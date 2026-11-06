"""Native-token SSE adapter and a small, unscored endpoint self-check.

Keep streaming normal text as it becomes available. Before the finish frame and
[DONE], write native_token_event(accepted_generated_ids). IDs must come from the
decoder, never by encoding the response text. One final ID event avoids adding
a GPU synchronisation or an HTTP flush to every decode step.

The replay gate reads the same IDs back out of AgentX's raw export
(`exact_output_lengths`): under ignore_eos every reply must carry exactly the
requested number of native IDs.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def checked_ids(ids) -> list[int]:
    if not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("token_ids must be a list of nonnegative native integer IDs")
    return ids


def native_token_event(ids: list[int], *, request_id: str = "native-tokens",
                       model: str = "target", created: int = 0) -> bytes:
    """Encode accepted output IDs once, all at the end or in disjoint chunks.

    Pass a CPU list already collected by your decoder. Exclude prompt, padding
    and rejected speculative IDs. This helper neither tokenizes nor detokenizes.
    """
    event = {"id": request_id, "object": "chat.completion.chunk",
             "created": created, "model": model,
             "choices": [{"index": 0, "delta": {"token_ids": checked_ids(ids)},
                          "finish_reason": None}]}
    return b"data: " + json.dumps(event).encode("utf-8") + b"\n\n"


def validate_native_output(ids: list[int], text: str, tokenizer, *,
                           expected_count: int) -> None:
    checked_ids(ids)
    if not ids or len(ids) != expected_count:
        raise ValueError(f"received {len(ids)} native IDs; expected {expected_count}")
    # Decode the whole sequence: decoding individual byte-fallback tokens and
    # joining can corrupt Unicode. Never reconstruct IDs from response text.
    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    decoded = backend.decode(ids, skip_special_tokens=False)
    if decoded != text:
        raise ValueError("native IDs do not decode exactly to the streamed text")


def collect_stream(lines) -> dict:
    """Collect the single-choice stream, including a final metadata-only event."""
    ids, pieces = [], []
    finish, usage, done = None, None, False
    for raw in lines:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if done:
            raise ValueError("data received after [DONE]")
        if data == "[DONE]":
            done = True
            continue
        if not data:
            continue
        event = json.loads(data)
        if event.get("error"):
            raise ValueError(f"server error: {event['error']}")
        if event.get("usage") is not None:
            usage = event["usage"].get("completion_tokens")
        for choice in event.get("choices") or []:
            if choice.get("index", 0) != 0:
                raise ValueError("expected one output choice")
            delta = choice.get("delta") or {}
            pieces.extend([delta.get("reasoning_content") or "",
                           delta.get("content") or ""])
            if "token_ids" in delta:
                ids.extend(checked_ids(delta["token_ids"]))
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    if not done:
        raise ValueError("stream ended without [DONE]")
    return {"token_ids": ids, "text": "".join(pieces),
            "finish_reason": finish, "usage_tokens": usage}


RAW_EXPORT = "profile_export_raw.jsonl"


def raw_export(artifact_dir) -> Path | None:
    """AgentX's per-request raw export under `artifact_dir`, or None."""
    found = sorted(Path(artifact_dir).rglob(RAW_EXPORT))
    return found[-1] if found else None


def record_token_ids(record: dict) -> list[int]:
    """The native output IDs one raw-export record streamed, in order."""
    ids: list[int] = []
    for response in record.get("responses") or []:
        for packet in (response or {}).get("packets") or []:
            try:
                event = json.loads((packet or {}).get("value", ""))
            except (ValueError, TypeError):
                continue
            if not isinstance(event, dict):
                continue
            for choice in event.get("choices") or []:
                if choice.get("index", 0) != 0:
                    continue
                chunk = (choice.get("delta") or {}).get("token_ids")
                if chunk is None:
                    chunk = choice.get("token_ids")
                if chunk is not None:
                    ids.extend(checked_ids(chunk))
    return ids


def requested_output_tokens(payload: dict) -> int | None:
    """The output length a request asked for (`max_tokens` or its alias)."""
    for key in ("max_tokens", "max_completion_tokens"):
        value = (payload or {}).get(key)
        if type(value) is int and value > 0:
            return value
    return None


def exact_output_lengths(artifact_dir, *, examples: int = 5) -> dict:
    """Every committed profiling reply returned exactly the tokens it asked for.

    With `ignore_eos` the server is told how many tokens to produce, so the
    count of native output IDs must equal the request's `max_tokens`: not
    within a tolerance, exactly. A reply without native IDs fails. Errored,
    cancelled and skipped requests are counted by the aggregate instead.
    """
    source = raw_export(artifact_dir)
    result = {"rule": "exact_output_length", "holds": False, "checked": 0,
              "mismatched": 0, "missing_native_ids": 0, "examples": [],
              "raw_export": str(source or ""), "why": ""}
    if source is None:
        result["why"] = (f"no {RAW_EXPORT} was written, so the length of each "
                         "reply cannot be verified")
        return result
    first = ""
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue
                metadata = record.get("metadata") or {}
                payload = record.get("payload")
                if (metadata.get("benchmark_phase") != "profiling"
                        or record.get("error") or metadata.get("was_cancelled")
                        or metadata.get("context_overflow_skip")
                        or not isinstance(payload, dict)
                        or payload.get("ignore_eos") is not True):
                    continue
                wanted = requested_output_tokens(payload)
                if wanted is None:
                    continue
                result["checked"] += 1
                try:
                    got = len(record_token_ids(record))
                    problem = ""
                except ValueError as exc:
                    got, problem = 0, str(exc)
                if got == wanted and not problem:
                    continue
                if got == 0:
                    result["missing_native_ids"] += 1
                else:
                    result["mismatched"] += 1
                where = "/".join(str(metadata.get(key)) for key in
                                 ("session_num", "turn_index")
                                 if metadata.get(key) is not None)
                example = {"x_request_id": metadata.get("x_request_id"),
                           "session_turn": where, "requested_tokens": wanted,
                           "native_token_ids": got}
                if problem:
                    example["error"] = problem
                if len(result["examples"]) < examples:
                    result["examples"].append(example)
                if not first:
                    first = (f"request {example['x_request_id']}"
                             + (f" (session/turn {where})" if where else "")
                             + (f" returned {got} native token ids"
                                if got else " carried no native token ids")
                             + f"; it asked for exactly {wanted} with ignore_eos")
    except OSError as exc:
        result["why"] = f"{RAW_EXPORT} is unreadable: {exc}"
        return result
    bad = result["mismatched"] + result["missing_native_ids"]
    result["holds"] = not bad
    if bad:
        result["why"] = (f"{bad} of {result['checked']} profiling replies did "
                         f"not return exactly the requested number of tokens. "
                         f"First: {first}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.self_test:
        result = collect_stream([native_token_event([1, 2]), b"data: [DONE]\n"])
        assert result["token_ids"] == [1, 2]
        print("native-token adapter self-test passed (no model or GPU required)")
        return 0
    if args.model_dir is None or args.tokens < 1:
        parser.error("provide --model-dir and a positive --tokens")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    payload = {"model": "target", "messages": [{"role": "user",
               "content": "Write a short paragraph about a quiet morning."}],
               "temperature": 0, "max_completion_tokens": args.tokens,
               "ignore_eos": True, "return_token_ids": True,
               "skip_special_tokens": False, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    endpoint = args.endpoint if "://" in args.endpoint else "http://" + args.endpoint
    request = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            output = collect_stream(response)
        validate_native_output(output["token_ids"], output["text"], tokenizer,
                               expected_count=args.tokens)
        if output["finish_reason"] != "length" or output["usage_tokens"] != args.tokens:
            raise ValueError("ignore_eos requires finish_reason=length and exact usage")
    except Exception as exc:
        print(json.dumps({"holds": False, "why": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps({"holds": True, "native_tokens": len(output["token_ids"]),
                      "note": "protocol only; not a probability correctness verdict"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
