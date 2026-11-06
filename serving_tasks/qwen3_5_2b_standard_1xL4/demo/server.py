#!/usr/bin/env python3
"""Correctness-first starter for qwen3_5_2b_standard_1xL4.

This deliberately slow OpenAI-compatible server uses plain Transformers,
one global request lock, and no cross-request prefix cache. ``AGENT_TASK.md``
in the workspace is the task's specification. Model, hardware, gates, timebox,
correctness, and concurrency values do not live in this reusable
implementation's comments.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
MODEL_CONFIG_OVERRIDES = {}
SERVED_AS = os.environ.get("SERVED_MODEL_NAME", "target")
PORT = int(os.environ.get("PORT", "8000"))

_STATE: dict = {"model": None, "tokenizer": None, "ready": False, "error": ""}
_COUNTERS: dict = {"requests": 0, "failures": 0,
                   "prompt_tokens": 0, "output_tokens": 0,
                   "prefill_seconds": 0.0, "decode_seconds": 0.0}

#: ONE LOCK, HELD ACROSS THE WHOLE REQUEST.
#: Requests are served one at a time; a request that arrives while another is
#: running waits for it to finish.
_LOCK = threading.Lock()


def _load() -> None:
    """Load once, in the background; report failure as state rather than exit.

    A server that exits on startup gives the harness a connection error, which
    looks like a hundred other things. A 503 carrying the exception says what
    happened.
    """
    try:
        import torch
        from transformers import AutoConfig, AutoTokenizer

        source = (os.environ.get("SEED_MODEL_DIR")
                  or os.environ.get("MODEL_PATH") or MODEL_ID)
        kwargs: dict = {} if os.path.isdir(source) else {"revision": REVISION}
        tokenizer = AutoTokenizer.from_pretrained(str(source), **kwargs)
        config = AutoConfig.from_pretrained(str(source), **kwargs)
        for name, value in copy.deepcopy(MODEL_CONFIG_OVERRIDES).items():
            current = getattr(config, name, None)
            if isinstance(value, dict) and hasattr(current, "update"):
                current.update(value)
            else:
                setattr(config, name, value)

        # Prefer the causal auto-class. Multimodal wrappers may require the
        # image-to-text class even for text-only requests.
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                str(source), config=config, dtype=torch.bfloat16,
                device_map="auto", **kwargs)
        except (ValueError, KeyError):
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                str(source), config=config, dtype=torch.bfloat16,
                device_map="auto", **kwargs)
        model.eval()
        _STATE.update(model=model, tokenizer=tokenizer, ready=True)
        print(f"[server] ready: {source} across "
              f"{getattr(model, 'hf_device_map', 'one device')}", flush=True)
    except Exception as exc:                       # noqa: BLE001 - reported
        _STATE["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[server] LOAD FAILED {_STATE['error']}", flush=True)


def _prompt_of(messages: list, chat_template_kwargs: dict | None = None) -> str:
    """The chat template applied, or the raw concatenation if there is none."""
    tokenizer = _STATE["tokenizer"]
    template_kwargs = {"enable_thinking": False}
    template_kwargs.update(chat_template_kwargs or {})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **template_kwargs)
    except Exception:                              # noqa: BLE001
        return "\n".join(str(m.get("content") or "") for m in messages)


def _eos_ids(model, tokenizer) -> set:
    """Every id the checkpoint stops on: its generation config and tokenizer."""
    ids = set()
    for value in (getattr(model.generation_config, "eos_token_id", None),
                  tokenizer.eos_token_id):
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, (list, tuple)):
            ids.update(v for v in value if isinstance(v, int))
    return ids


def _max_model_len(model) -> int:
    """The context window the checkpoint itself declares."""
    config = getattr(model.config, "text_config", None) or model.config
    return int(getattr(config, "max_position_embeddings", 0)
               or getattr(model.config, "max_position_embeddings", 0) or 0)


class _TokenSink:
    """Receives each token from `generate()` as it is produced.

    The streamer protocol: the first `put` is the prompt, every later one is
    the token just generated, and `end` follows the last.
    """

    def __init__(self, on_token):
        self.on_token = on_token
        self.prompt_seen = False

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        for token in value.reshape(-1).tolist():
            self.on_token(int(token))

    def end(self):
        pass


def _generate(body: dict, on_token=None):
    """Greedy decode, no cache reuse, one request at a time.

    `on_token(id)` is called for every token as soon as it is generated, so
    the handler can stream it. Returns (prompt_tokens, produced ids,
    finish_reason).
    """
    import torch

    messages = body.get("messages")
    if not messages:
        raise ValueError("a request needs at least one message")
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
    if max_tokens is None:
        raise ValueError(
            "a request needs an explicit max_tokens. The scenario sets "
            "ignore_eos, so max_tokens is exactly how many tokens this turn is "
            "worth and defaulting it changes how much work the arm does")

    # `ignore_eos` arrives in the body because the scenario injects it. Honour
    # it: generating to EOS instead would answer a shorter question than the
    # one asked, on every turn, and the replay-fidelity invariant catches it.
    # Without it, generation stops at the checkpoint's end-of-sequence token.
    ignore_eos = bool(body.get("ignore_eos", False))

    model, tokenizer = _STATE["model"], _STATE["tokenizer"]
    text = _prompt_of(messages, body.get("chat_template_kwargs"))
    eos = _eos_ids(model, tokenizer)

    with _LOCK:
        started = time.perf_counter()
        encoded = tokenizer(text, return_tensors="pt",
                            add_special_tokens=False)
        device = next(model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[1])

        # THE WHOLE PREFILL, EVERY TIME.
        #
        # No prefix cache, no reuse across turns of a session: every request's
        # whole prompt is prefilled, including the part an earlier turn of the
        # same session already sent.
        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=int(max_tokens),
                min_new_tokens=int(max_tokens) if ignore_eos else None,
                do_sample=False,
                eos_token_id=(None if ignore_eos else sorted(eos)),
                pad_token_id=tokenizer.pad_token_id or min(eos),
                streamer=_TokenSink(on_token) if on_token else None)
        finished = time.perf_counter()

    produced = out[0][prompt_tokens:].tolist()
    stopped = (not ignore_eos and bool(produced) and produced[-1] in eos)
    _COUNTERS["prompt_tokens"] += prompt_tokens
    _COUNTERS["output_tokens"] += len(produced)
    _COUNTERS["decode_seconds"] += finished - started
    return prompt_tokens, produced, ("stop" if stopped else "length")


class _Detokenizer:
    """Incremental detokenisation: emits text as tokens arrive, and the
    concatenation equals decoding the whole sequence at once."""

    def __init__(self, tokenizer, skip_special_tokens: bool):
        self.tokenizer = tokenizer
        self.skip = skip_special_tokens
        self.ids: list[int] = []
        self.sent = ""

    def _decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=self.skip,
                                     clean_up_tokenization_spaces=False)

    def push(self, token: int) -> str:
        self.ids.append(token)
        text = self._decode(self.ids)
        if text.endswith("�") or not text.startswith(self.sent):
            return ""                     # an incomplete multi-byte character
        piece, self.sent = text[len(self.sent):], text
        return piece

    def flush(self) -> str:
        text = self._decode(self.ids)
        piece = text[len(self.sent):] if text.startswith(self.sent) else ""
        self.sent = text
        return piece


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "serverbench-seed/1.0"

    def log_message(self, fmt, *args):
        print(f"[server] {fmt % args}", flush=True)

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _stream(self, body: dict) -> None:
        """Server-sent events in the OpenAI shape, one chunk per token as it
        is generated, so TTFT and TPOT measure this server's own speed."""
        rid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())

        def chunk(delta: dict, finish=None) -> bytes:
            return b"data: " + json.dumps({
                "id": rid, "object": "chat.completion.chunk",
                "created": created, "model": SERVED_AS,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish}]}).encode() + b"\n\n"

        detok = _Detokenizer(_STATE["tokenizer"],
                             bool(body.get("skip_special_tokens", True)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(chunk({"role": "assistant", "content": ""}))
        self.wfile.flush()

        def on_token(token: int) -> None:
            piece = detok.push(token)
            if piece:
                self.wfile.write(chunk({"content": piece}))
                self.wfile.flush()

        try:
            prompt_tokens, produced, finish = _generate(body, on_token)
        except Exception as exc:                   # noqa: BLE001
            _COUNTERS["failures"] += 1
            self.wfile.write(b"data: " + json.dumps(
                {"error": f"{type(exc).__name__}: {exc}"}).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            return
        rest = detok.flush()
        if rest:
            self.wfile.write(chunk({"content": rest}))
        if body.get("return_token_ids"):
            from native_tokens import native_token_event
            self.wfile.write(native_token_event(
                produced, request_id=rid, model=SERVED_AS, created=created))
        self.wfile.write(chunk({}, finish))
        self.wfile.write(b"data: " + json.dumps({
            "id": rid, "object": "chat.completion.chunk", "created": created,
            "model": SERVED_AS, "choices": [],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": len(produced),
                      "total_tokens": prompt_tokens + len(produced)}}).encode()
            + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            self._send(200 if _STATE["ready"] else 503,
                       {"status": "ok"} if _STATE["ready"]
                       else {"status": "loading", "error": _STATE["error"]})
        elif self.path == "/v1/models":
            ready = _STATE["ready"]
            self._send(200 if ready else 503, {
                "object": "list",
                "data": [{"id": SERVED_AS, "object": "model",
                          "model_id": MODEL_ID, "revision": REVISION,
                          "max_model_len": (_max_model_len(_STATE["model"])
                                            if ready else None)}],
                "ready": ready, "error": _STATE["error"]})
        elif self.path == "/metrics":
            done = max(1, _COUNTERS["requests"] - _COUNTERS["failures"])
            self._send(200, {**_COUNTERS,
                             "mean_seconds_per_request": round(
                                 _COUNTERS["decode_seconds"] / done, 3)})
        else:
            self._send(404, {"error": f"no such path {self.path}"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": f"no such path {self.path}"})
            return
        if not _STATE["ready"]:
            self._send(503, {"error": "model still loading",
                             "load_error": _STATE["error"]})
            return
        length = int(self.headers.get("Content-Length") or 0)
        _COUNTERS["requests"] += 1
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if body.get("stream"):
                self._stream(body)
                return
            prompt_tokens, produced, finish = _generate(body)
            answer = _STATE["tokenizer"].decode(
                produced, skip_special_tokens=bool(body.get("skip_special_tokens", True)),
                clean_up_tokenization_spaces=False)
            self._send(200, {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion", "created": int(time.time()),
                "model": SERVED_AS,
                "choices": [{"index": 0, "finish_reason": finish,
                             "message": {"role": "assistant",
                                         "content": answer}}],
                "usage": {"prompt_tokens": prompt_tokens,
                          "completion_tokens": len(produced),
                          "total_tokens": prompt_tokens + len(produced)}})
        except ValueError as exc:
            _COUNTERS["failures"] += 1
            self._send(400, {"error": str(exc)})
        except Exception as exc:                   # noqa: BLE001
            _COUNTERS["failures"] += 1
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    threading.Thread(target=_load, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[server] listening on 0.0.0.0:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
