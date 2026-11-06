"""The final verdict's correctness stage. HOST-ONLY: never packaged.

The coding agent must not learn how output correctness is checked. It is told
that correctness is verified privately, within a tolerance, and that it must
build its own verification. So everything
that says HOW -- the plain-checkpoint launch, the held-out questions, the
accuracy rule and the end-of-sequence padding audit -- lives here, and
`package_grader` does not copy this module (or `agentx_reference`) into any
task's `tests/final_verdict/`. `scenario.py` reaches it through a lazy import;
the in-workspace evaluator copy has no such module, so it can measure but
cannot run or read a correctness verdict.

Output correctness is three checks, all hard:

1. **replay fidelity** (in `scenario`): the timed replay's protocol, lengths
   and context integrity.
2. **answer accuracy.** Judged on answers rather than by teacher-forcing
   the timed replies, so an exact but differently-batched system is not
   failed for its batching. After the timed replay, while the candidate is
   still up, a seeded draw of held-out questions with known
   numeric answers is put to it -- greedy, a natural stop. The plain
   checkpoint answers the same questions after the candidate is gone. The
   candidate may lose at most `max_absolute_accuracy_drop` (0.20) accuracy
   against the plain checkpoint: at most four fewer correct answers out of 20.
3. **the EOS gate** (tasks whose `reference.yaml` sets
   `correctness.eos_audit: true`). Under `ignore_eos` a server is told exactly
   how many tokens to return. A server that stops at the model's
   end-of-sequence token and fills the remainder with EOS/pad tokens -- or with
   any token it did not compute -- returns the right COUNT for a fraction of
   the work, and replay fidelity cannot see it:

   * **stop probe.** One short streamed request WITHOUT `ignore_eos`. The
     candidate must finish with `finish_reason == "stop"` before `max_tokens`.
     If the plain reference does not stop either, the probe is inconclusive
     rather than a failure.
   * **padding audit.** Every committed timed reply's native token IDs are
     scanned. A reply is SUSPECT when an EOS-class token is followed by any
     further token -- whatever the following tokens are, so padding with
     ordinary or varied tokens is caught as surely as padding with EOS -- or
     when it contains a run of `REPEAT_RUN` identical tokens anywhere. Up to
     `SUSPECT_AUDIT_CAP` suspects (seeded choice, after the replay) are
     teacher-forced on the plain checkpoint under `max_token_regret`. A model
     that genuinely continues past EOS, or a greedy continuation that really
     does loop, passes; fabricated tokens do not. A timed reply without native
     IDs fails the gate, and so does one whose native IDs do not decode to
     exactly the text it streamed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import secrets
import shlex
import threading
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from final_validate import agentx_reference, native_tokens

REPEAT_RUN = 64
SUSPECT_AUDIT_CAP = 16
STOP_PROBE_MAX_TOKENS = 64
STOP_PROBE_PROMPT = "Reply with the single word OK and nothing else."


# ------------------------------------------------------------------ settings

def _settings(contract) -> dict[str, Any]:
    """The task's private correctness block, from its `reference.yaml`."""
    from final_validate import task_spec
    return task_spec.reference(contract.task_dir).get("correctness") or {}


def suspect_regret(contract) -> float:
    """The max-token logprob-regret limit for audited EOS-gate suspects."""
    regret = float(_settings(contract).get(
        "max_token_regret", agentx_reference.DEFAULT_MAX_TOKEN_LOGPROB_REGRET))
    if not 0.0 <= regret < float("inf"):
        raise SystemExit("reference.yaml correctness.max_token_regret must be "
                         "finite and nonnegative")
    return regret


def eos_audit_enabled(contract) -> bool:
    """Whether this task's reference.yaml turns the EOS gate on."""
    return bool(_settings(contract).get("eos_audit"))


def plain_reference_serve_extra(contract) -> str:
    """Launch the correctness oracle without the candidate-speed recipe.

    A task's ordinary SGLang arm may deliberately enable speculative decoding.
    That is a useful performance baseline, but it is neither a plain checkpoint
    oracle nor always small enough to coexist with the target model on the
    task's cards. Preserve context/RoPE and memory flags and admit one request
    at a time. Correctness uses the deterministic Triton path: prefix/extend partitioning and batch shape must
    not select different attention reduction paths for the same scored token.
    The performance arms retain their ordinary kernel configuration.
    """
    from final_validate import task_spec
    from final_validate.scenario import fill_concurrency

    template = task_spec.serve_extra(contract.task_dir, "sglang")
    filled = fill_concurrency(template, 1)
    parts = shlex.split(filled)
    plain: list[str] = []
    index = 0
    while index < len(parts):
        token = parts[index]
        if (token.startswith("--speculative-") or
                token.split("=", 1)[0] in {
                    "--attention-backend", "--prefill-attention-backend",
                    "--decode-attention-backend", "--max-running-requests",
                    "--chunked-prefill-size"}):
            index += 1
            if ("=" not in token and index < len(parts)
                    and not parts[index].startswith("--")):
                index += 1
            continue
        plain.append(token)
        index += 1
    # Deterministic Triton prefill only cuts chunks at multiples of
    # SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE (4096). SGLang's default
    # chunk on a card under 35 GB (L4, A10) is 2048, below that alignment, so
    # a prompt longer than one chunk is never scheduled and the server wedges.
    # Chunk boundaries are aligned either way, so the scored logprobs do not
    # depend on this value.
    plain.extend(["--attention-backend", "triton",
                  "--max-running-requests", "1",
                  "--chunked-prefill-size", "4096"])
    if "--enable-deterministic-inference" not in plain:
        plain.append("--enable-deterministic-inference")
    # Cached prefix boundaries still change BF16 intermediates in the
    # pinned Gemma implementation, even with deterministic attention.
    # Every audited reply starts from a fresh reference state.
    if "--disable-radix-cache" not in plain:
        plain.append("--disable-radix-cache")
    return shlex.join(plain)


def plain_reference_server_env() -> dict[str, str]:
    """Environment for the plain reference server.

    Thousands of output-token logits otherwise materialize a multi-GiB
    temporary tensor beside the preallocated KV cache.
    """
    return {"SGLANG_LOGPROB_CHUNK_SIZE": "256"}


def combined_correctness(replay: dict[str, Any],
                         answers: dict[str, Any],
                         eos_gate: dict[str, Any] | None = None,
                         ) -> dict[str, Any]:
    """Replay fidelity AND output correctness (answer accuracy + EOS gate)."""
    answer = dict(answers)
    if eos_gate is not None:
        answer["eos_gate"] = eos_gate
        answer["holds"] = bool(answer.get("holds")) and eos_gate.get("holds") is True
        if eos_gate.get("holds") is not True:
            answer["why"] = "; ".join(filter(None, [
                answer.get("why", ""), "eos gate: " + str(eos_gate.get("why", ""))]))
    evaluable = bool(answer.get("evaluable", answer.get("available", False)))
    return {
        "holds": (bool(replay.get("holds")) and evaluable
                  and bool(answer.get("holds"))),
        "evaluable": evaluable,
        "rule": "replay_fidelity_and_held_out_answers_vs_plain_checkpoint",
        "replay_fidelity": replay,
        "output_correctness": answer,
        "why": "; ".join(filter(None, [replay.get("why", ""),
                                         answer.get("why", "")])),
    }


# ------------------------------------------------------------------ answers

#: Held-out questions with known numeric answers, disjoint from anything a
#: workspace or a development run can reach. Local to the verdict host and
#: gitignored; the Modal backend ships a per-verdict selection instead.
ANSWER_POOL = (Path(__file__).resolve().parents[2] / "artifacts" / "eval_samples"
               / "gsm8k-final_validation.json")
#: Where a Modal candidate sandbox finds the questions (no answers) chosen
#: for this verdict on the host. Unset on the Docker path.
QUESTIONS_FILE_ENV = "SERVERBENCH_PRIVATE_QUESTIONS"
DEFAULT_ANSWER_SAMPLE = 20
DEFAULT_MAX_ABSOLUTE_ACCURACY_DROP = 0.2
ANSWER_MAX_TOKENS = 1024
ANSWER_WORKERS = 4
ANSWER_TIMEOUT_S = 300.0
#: Consecutive unanswered questions after which the rest are not attempted
#: (a crashed server would otherwise cost timeout x questions).
ANSWER_GIVE_UP = 8
ANSWER_PROMPT = ("Solve the following problem. Reason step by step, then give "
                 "the final answer as a number on its own last line, in the "
                 "form '#### <number>'.\n\n{question}")

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
_FINAL = re.compile(r"####\s*([^\n]*)")


def answer_settings(contract) -> tuple[int, float]:
    """Question count and maximum absolute accuracy drop (not a ratio)."""
    configured = _settings(contract)
    if "min_relative_accuracy" in configured:
        raise SystemExit("obsolete min_relative_accuracy; set "
                         "max_absolute_accuracy_drop explicitly")
    size = int(configured.get("answer_sample_size", DEFAULT_ANSWER_SAMPLE))
    raw_drop = configured.get("max_absolute_accuracy_drop",
                              DEFAULT_MAX_ABSOLUTE_ACCURACY_DROP)
    drop = float(raw_drop)
    if size < 1:
        raise SystemExit("reference.yaml correctness.answer_sample_size must "
                         "be positive")
    if isinstance(raw_drop, bool) or not 0.0 <= drop <= 1.0:
        raise SystemExit("reference.yaml correctness.max_absolute_accuracy_drop "
                         "must be in [0, 1]")
    return size, drop


def _as_number(text: str) -> float | None:
    try:
        return float(text.replace(",", "").strip())
    except ValueError:
        return None


def expected_answer(item: dict[str, Any]) -> float | None:
    """The number an item declares after `####`."""
    text = str(item.get("reference_answer") or "")
    marked = _FINAL.findall(text)
    found = _NUMBER.findall(marked[-1] if marked else text)
    return _as_number(found[-1]) if found else None


def extracted_answer(text: str | None) -> float | None:
    """The reply's final number: the one after its last `####`, else the last
    number anywhere in it (a correct reply that ignored the format counts)."""
    text = text or ""
    marked = [m for m in _FINAL.findall(text) if _NUMBER.search(m)]
    found = _NUMBER.findall(marked[-1] if marked else text)
    return _as_number(found[0] if marked else found[-1]) if found else None


def _pool() -> list[dict[str, Any]]:
    body = json.loads(ANSWER_POOL.read_text("utf-8"))
    return [item for item in body.get("items") or []
            if expected_answer(item) is not None]


def select_questions(size: int, seed: int) -> list[dict[str, Any]]:
    """A seeded draw from the pool: id, question and expected answer."""
    items = _pool()
    if len(items) < size:
        raise ValueError(f"the held-out pool has {len(items)} usable items; "
                         f"{size} are required")
    chosen = random.Random(int(seed)).sample(items, size)
    return [{"id": item["id"], "question": item["question"],
             "answer": expected_answer(item)} for item in chosen]


def answer_body(question: str) -> dict[str, Any]:
    return {"model": "target", "stream": True, "temperature": 0,
            "max_tokens": ANSWER_MAX_TOKENS, "ignore_eos": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user",
                          "content": ANSWER_PROMPT.format(question=question)}]}


def ask(url: str, questions: list[dict[str, Any]], *, workers: int = 1,
        destination: Path | None = None) -> list[dict[str, Any]]:
    """Every question, once, retried once on a transport error. Greedy."""
    lock = threading.Lock()
    failures = [0]

    def one(question: dict[str, Any]) -> dict[str, Any]:
        with lock:
            if failures[0] >= ANSWER_GIVE_UP:
                return {"id": question["id"], "error": "not attempted: the "
                        f"server failed {ANSWER_GIVE_UP} questions in a row"}
        error = ""
        for _ in range(2):
            try:
                got = _stream_once(url, answer_body(question["question"]),
                                   timeout=ANSWER_TIMEOUT_S)
                with lock:
                    failures[0] = 0
                return {"id": question["id"], "text": got.get("text", ""),
                        "finish_reason": got.get("finish_reason")}
            except Exception as exc:                          # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
        with lock:
            failures[0] += 1
        return {"id": question["id"], "error": error}

    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        replies = list(pool.map(one, questions))
    if destination is not None:
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_text(json.dumps(replies, indent=1) + "\n", "utf-8")
    return replies


def _questions_for_this_verdict(contract) -> list[dict[str, Any]]:
    """Drawn now, after the timed replay, so no run could have known them.
    On Modal the host has drawn them already and shipped the questions."""
    shipped = os.environ.get(QUESTIONS_FILE_ENV)
    if shipped:
        return json.loads(Path(shipped).read_text("utf-8"))
    size, _ = answer_settings(contract)
    return [{"id": q["id"], "question": q["question"]}
            for q in select_questions(size, secrets.randbits(64))]


def answer_probe(url: str, contract, destination: Path, *,
                 concurrency: int | None = None) -> dict[str, Any]:
    """The candidate half: its replies to the held-out questions."""
    try:
        questions = _questions_for_this_verdict(contract)
    except (OSError, ValueError) as exc:
        return {"questions": [], "replies": [],
                "error": f"held-out questions unavailable: {exc}"}
    workers = min(ANSWER_WORKERS, max(1, int(concurrency or 1)))
    return {"questions": questions,
            "replies": ask(url, questions, workers=workers,
                           destination=destination)}


def _correct(reply: dict[str, Any] | None, want: float | None) -> bool:
    if not reply or reply.get("error") or want is None:
        return False
    got = extracted_answer(reply.get("text"))
    return got is not None and abs(got - want) < 1e-6


def judge_answers(probe: dict[str, Any], reference: list[dict[str, Any]],
                  answers: dict[str, float], max_absolute_drop: float
                  ) -> dict[str, Any]:
    """Allow an absolute accuracy drop: reference correct minus floor(N * drop)."""
    ids = [q["id"] for q in probe.get("questions") or []]
    base = {"schema": "final_validate.answer_accuracy/2", "available": bool(ids),
            "rule": "held_out_answer_accuracy_absolute_drop_vs_plain_checkpoint",
            "max_absolute_accuracy_drop": max_absolute_drop, "questions": len(ids)}
    if (isinstance(max_absolute_drop, bool)
            or not isinstance(max_absolute_drop, (int, float))
            or not 0 <= max_absolute_drop <= 1):
        return {**base, "evaluable": False, "holds": False,
                "why": "invalid absolute accuracy drop"}
    if not ids:
        return {**base, "evaluable": False, "holds": False,
                "why": probe.get("error") or "no held-out question was asked"}
    unknown = [i for i in ids if answers.get(i) is None]
    if unknown:
        return {**base, "evaluable": False, "holds": False,
                "why": f"{len(unknown)} questions have no expected answer"}
    candidate = {r["id"]: r for r in probe.get("replies") or []}
    plain = {r["id"]: r for r in reference}
    rows = [{"id": i, "expected": answers[i],
             "candidate": _correct(candidate.get(i), answers[i]),
             "reference": _correct(plain.get(i), answers[i])} for i in ids]
    got = sum(r["candidate"] for r in rows)
    want = sum(r["reference"] for r in rows)
    candidate_errors = sum(1 for i in ids if (candidate.get(i) or {"error": 1}).get("error"))
    reference_errors = sum(1 for i in ids if (plain.get(i) or {"error": 1}).get("error"))
    detail = {**base, "candidate_correct": got, "reference_correct": want,
              "candidate_accuracy": got / len(ids),
              "reference_accuracy": want / len(ids),
              "relative_accuracy": (got / want) if want else None,
              "candidate_errors": candidate_errors,
              "reference_errors": reference_errors, "per_question": rows}
    if want == 0 or reference_errors > len(ids) // 10:
        return {**detail, "evaluable": False, "holds": False,
                "why": (f"the plain checkpoint answered {want}/{len(ids)} with "
                        f"{reference_errors} errors; no usable reference")}
    allowed = math.floor(max_absolute_drop * len(ids) + 1e-9)
    required = max(0, want - allowed)
    detail.update(allowed_fewer_correct=allowed, min_candidate_correct=required,
                  absolute_accuracy_drop=(want - got) / len(ids))
    holds = got >= required
    return {**detail, "evaluable": True, "holds": holds,
            "why": "" if holds else (
                f"answer accuracy {got}/{len(ids)} is below required {required}/{len(ids)}; "
                f"plain checkpoint {want}/{len(ids)}, at most {allowed} fewer correct "
                f"({max_absolute_drop:g} absolute accuracy drop)"
                + (f" ({candidate_errors} unanswered)" if candidate_errors else ""))}


# ------------------------------------------------------------------ EOS gate

def eos_token_ids(model_dir: Path, tokenizer) -> set[int]:
    """Every id the checkpoint treats as end-of-sequence or padding."""
    ids: set[int] = set()
    for attr in ("eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int):
            ids.add(value)
    for name in ("generation_config.json", "config.json"):
        try:
            body = json.loads((Path(model_dir) / name).read_text("utf-8"))
        except (OSError, ValueError):
            continue
        sections = [body, body.get("text_config") or {}]
        for section in sections:
            for key in ("eos_token_id", "pad_token_id"):
                value = section.get(key)
                if isinstance(value, int):
                    ids.add(value)
                elif isinstance(value, list):
                    ids.update(v for v in value if isinstance(v, int))
    return ids


def _suspicion(tokens: list[int], eos_ids: set[int]) -> str:
    """Why this reply looks padded, or '' when it does not.

    Any token after an EOS-class token is suspect: past EOS a server can
    return whatever it likes, and only the teacher-forced audit can tell the
    model's own continuation from tokens it never computed.
    """
    first = next((i for i, t in enumerate(tokens) if t in eos_ids), None)
    if first is not None and first < len(tokens) - 1:
        return (f"{len(tokens) - 1 - first} token(s) follow the first "
                f"end-of-sequence/pad token (position {first})")
    run, best, previous = 0, 0, None
    for token in tokens:
        run = run + 1 if token == previous else 1
        previous = token
        best = max(best, run)
    if best >= REPEAT_RUN:
        return f"a run of {best} identical tokens"
    return ""


def _streamed_text(record: dict[str, Any]) -> str:
    """The text one raw-export record streamed, joined as the protocol probe
    joins it (`native_tokens.collect_stream`)."""
    pieces: list[str] = []
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
                delta = choice.get("delta") or {}
                pieces.extend([str(delta.get("reasoning_content") or ""),
                               str(delta.get("content") or "")])
    return "".join(pieces)


def padding_suspects(artifact_dir: Path, eos_ids: set[int], *,
                     tokenizer=None) -> dict[str, Any]:
    """Scan every profiling reply's native ids. Never samples.

    With `tokenizer`, also require that each reply's native ids decode to
    exactly the text it streamed: ids that are not the streamed text are not
    the reply, whatever their count.
    """
    source = agentx_reference._raw_export(artifact_dir)
    scanned = missing = 0
    suspects: dict[str, str] = {}
    decode_mismatches: list[str] = []
    with Path(source).open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            metadata = record.get("metadata") or {}
            if (metadata.get("benchmark_phase") != "profiling"
                    or record.get("error") or metadata.get("was_cancelled")):
                continue
            request_id = str(metadata.get("x_request_id") or "")
            try:
                tokens = agentx_reference._stream_token_ids(record)
            except ValueError:
                tokens = []
            if not tokens:
                missing += 1
                continue
            scanned += 1
            why = _suspicion(tokens, eos_ids)
            if why:
                suspects[request_id] = why
            if tokenizer is not None:
                try:
                    native_tokens.validate_native_output(
                        tokens, _streamed_text(record), tokenizer,
                        expected_count=len(tokens))
                except (ValueError, TypeError, IndexError, OverflowError):
                    decode_mismatches.append(request_id)
    out = {"scanned": scanned, "missing_native_ids": missing,
           "suspects": suspects, "raw_export": str(source)}
    if tokenizer is not None:
        out["decode_checked"] = scanned
        out["decode_mismatches"] = len(decode_mismatches)
        out["decode_mismatch_examples"] = decode_mismatches[:5]
    return out


def _stream_once(url: str, body: dict[str, Any], *, timeout: float = 300.0
                 ) -> dict[str, Any]:
    """One streamed chat request, parsed by the same strict reader as the
    protocol probe (`native_tokens.collect_stream`)."""
    data = json.dumps(body).encode("utf-8")
    # Every replayed request carries an X-Correlation-ID (a uuid4 per
    # session), and the protocol a candidate is built against is that one.
    # These requests carry one too, in the same form, so a probe is
    # indistinguishable from replay traffic and does not exercise a path the
    # replay never sends: a server may key its queue on the id.
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions", data=data, method="POST",
        headers={"content-type": "application/json",
                 "X-Correlation-ID": str(uuid.uuid4())})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return native_tokens.collect_stream(response)


def stop_probe_body() -> dict[str, Any]:
    return {"model": "target", "stream": True, "temperature": 0,
            "max_tokens": STOP_PROBE_MAX_TOKENS, "ignore_eos": False,
            "return_token_ids": True, "skip_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": STOP_PROBE_PROMPT}]}


def run_stop_probe(url: str, destination: Path | None = None) -> dict[str, Any]:
    """The candidate half of the stop probe. Judged after the reference half."""
    try:
        got = _stream_once(url, stop_probe_body())
    except Exception as exc:                                  # noqa: BLE001
        got = {"error": f"{type(exc).__name__}: {exc}"}
    if destination is not None:
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_text(json.dumps(got, indent=1) + "\n", "utf-8")
    return got


def _stopped(result: dict[str, Any]) -> bool:
    return (result.get("finish_reason") == "stop"
            and 0 < len(result.get("token_ids") or [])
            < STOP_PROBE_MAX_TOKENS)


def judge_eos_gate(*, candidate_stop: dict[str, Any],
                   reference_stop: dict[str, Any] | None,
                   scan: dict[str, Any], audit: dict[str, Any] | None
                   ) -> dict[str, Any]:
    why: list[str] = []
    reference_stops = reference_stop is not None and _stopped(reference_stop)
    stop_ok = _stopped(candidate_stop) or not reference_stops
    if not stop_ok and candidate_stop.get("error"):
        why.append("the candidate did not answer the end-of-sequence stop "
                   f"probe ({candidate_stop['error']})")
    elif not stop_ok:
        why.append("with ignore_eos=false the candidate did not stop at the "
                   "model's end-of-sequence token "
                   f"(finish_reason={candidate_stop.get('finish_reason')!r}, "
                   f"{len(candidate_stop.get('token_ids') or [])} tokens)")
    if scan.get("missing_native_ids"):
        why.append(f"{scan['missing_native_ids']} timed replies carried no "
                   "native token ids, so they could not be scanned")
    if scan.get("decode_mismatches"):
        first = (scan.get("decode_mismatch_examples") or ["?"])[0]
        why.append(f"{scan['decode_mismatches']} timed replies streamed text "
                   "that their native token ids do not decode to (first: "
                   f"request {first})")
    audit_ok = True
    if scan.get("suspects"):
        audit_ok = bool(audit and audit.get("evaluable") and audit.get("holds"))
        if not audit_ok:
            count = len(scan["suspects"])
            if audit and audit.get("evaluable"):
                why.append(f"{count} padded-looking replies; the audited ones "
                           "were not the checkpoint's own tokens")
            else:
                reason = str((audit or {}).get("why") or "").strip() or (
                    "no audit result was produced" if audit is None
                    else "the audit was not evaluable")
                why.append(f"{count} padded-looking replies could not be "
                           f"audited: {reason}")
    return {
        "schema": "final_validate.eos_gate/1",
        "holds": (stop_ok and not scan.get("missing_native_ids")
                  and not scan.get("decode_mismatches") and audit_ok),
        "stop_probe": {"candidate": candidate_stop, "reference": reference_stop,
                       "reference_stops": reference_stops, "holds": stop_ok},
        "scan": {k: v for k, v in scan.items() if k != "suspects"},
        "suspect_count": len(scan.get("suspects") or {}),
        "suspects": dict(list((scan.get("suspects") or {}).items())[:64]),
        "audit": audit,
        "why": "; ".join(why),
    }


def audit_suspects(url: str, artifact_dir: Path, suspects: dict[str, str], *,
                   tokenizer, max_token_regret: float, seed: int,
                   destination: Path | None = None) -> dict[str, Any]:
    ids = sorted(suspects)
    if len(ids) > SUSPECT_AUDIT_CAP:
        ids = sorted(random.Random(seed).sample(ids, SUSPECT_AUDIT_CAP))
    samples = agentx_reference.select_timed_requests(
        artifact_dir, seed=seed, sample_size=len(ids), request_ids=ids)
    return agentx_reference.replay_selected_and_compare(
        url, samples, tokenizer=tokenizer, max_token_regret=max_token_regret,
        destination=destination)


# ------------------------------------------------------------------ verdict

def candidate_probes(url: str, run_dir: Path, target: str, contract, *,
                     concurrency: int | None = None) -> dict[str, Any]:
    """Probes that must reach the CANDIDATE, run while it is still up."""
    base = Path(run_dir) / "agentx" / target
    probes: dict[str, Any] = {}
    if eos_audit_enabled(contract):
        probes["eos_stop_probe"] = run_stop_probe(url, base / "eos-stop-probe.json")
    probes["answer_probe"] = answer_probe(
        url, contract, base / "answer-probe-candidate.json",
        concurrency=concurrency)
    return probes


def eos_gate(contract, *, reference_url: str, artifact_dir: Path,
             model_dir: Path, tokenizer, candidate_stop: dict[str, Any]
             ) -> dict[str, Any] | None:
    """The EOS gate against a running plain reference, or None when off."""
    if not eos_audit_enabled(contract):
        return None
    try:
        scan = padding_suspects(artifact_dir, eos_token_ids(model_dir, tokenizer),
                                tokenizer=tokenizer)
    except (OSError, ValueError) as exc:
        scan = {"scanned": 0, "missing_native_ids": 0, "suspects": {},
                "error": str(exc)}
    audit = (audit_suspects(
        reference_url, artifact_dir, scan["suspects"], tokenizer=tokenizer,
        max_token_regret=suspect_regret(contract), seed=secrets.randbits(64),
        destination=artifact_dir / "eos-suspect-audit.json")
        if scan.get("suspects") else None)
    gate = judge_eos_gate(candidate_stop=candidate_stop,
                          reference_stop=run_stop_probe(reference_url),
                          scan=scan, audit=audit)
    if scan.get("error"):
        gate["holds"] = False
        gate["why"] = f"raw export unreadable: {scan['error']}"
    (artifact_dir / "eos-gate.json").write_text(
        json.dumps(gate, indent=1) + "\n", "utf-8")
    return gate


def expected_answers(questions: list[dict[str, Any]]) -> dict[str, float]:
    """The pool's answers for these question ids (the Docker host path)."""
    wanted = {q["id"] for q in questions}
    return {item["id"]: expected_answer(item) for item in _pool()
            if item["id"] in wanted}


def reference_stage(contract, *, reference_url: str, artifact_dir: Path,
                    model_dir: Path, tokenizer, row: dict[str, Any],
                    answers: dict[str, float] | None = None) -> dict[str, Any]:
    """Everything judged against the plain reference, for either backend.

    `row` is the candidate's measured row, carrying its `answer_probe`;
    returns `row['correctness']`. `answers` maps question id to the expected
    number (read from the pool when omitted). Docker and Modal both call this,
    so the two verdicts cannot drift apart.
    """
    probe = row.get("answer_probe") or {}
    questions = probe.get("questions") or []
    _, max_absolute_drop = answer_settings(contract)
    try:
        expected = answers if answers is not None else expected_answers(questions)
    except (OSError, ValueError) as exc:
        expected = {}
        probe = {**probe, "error": f"held-out answers unavailable: {exc}"}
    reference = (ask(reference_url, questions, workers=ANSWER_WORKERS,
                     destination=artifact_dir / "answer-probe-reference.json")
                 if questions else [])
    row["output_correctness"] = judge_answers(probe, reference, expected,
                                              max_absolute_drop)
    (artifact_dir / "answer-accuracy.json").write_text(
        json.dumps(row["output_correctness"], indent=1) + "\n", "utf-8")
    gate = eos_gate(contract, reference_url=reference_url,
                    artifact_dir=artifact_dir, model_dir=model_dir,
                    tokenizer=tokenizer,
                    candidate_stop=row.get("eos_stop_probe") or {})
    replay = (row.get("correctness") or {}).get("replay_fidelity") \
        or row.get("correctness") or {}
    return combined_correctness(replay, row["output_correctness"], gate)


def final_correctness(*, contract, args, row: dict[str, Any],
                      full_replay_fidelity: dict[str, Any],
                      tokenizer, launch, teardown) -> dict[str, Any]:
    """Docker: stand the plain reference up and run `reference_stage`.

    The candidate is gone before the reference starts, so both systems use the
    same cards and cannot contend. The questions were drawn only after the
    candidate had committed every timed reply.
    """
    artifact_dir = Path(args.run_dir) / "agentx" / args.target
    if not args.launch:
        # The Modal backend attaches to an endpoint here and finishes in its
        # own reference sandbox (`modal_worker.finish_reference`).
        unavailable = {
            "available": False, "evaluable": False, "holds": False,
            "rule": "held_out_answer_accuracy_absolute_drop_vs_plain_checkpoint",
            "why": ("the reference stage needs --launch, or the Modal "
                    "reference sandbox"),
        }
        row["output_correctness"] = unavailable
        gate = ({"holds": False, "why": "the reference stage did not run"}
                if eos_audit_enabled(contract) else None)
        return combined_correctness(full_replay_fidelity, unavailable, gate)
    reference_args = argparse.Namespace(**vars(args))
    reference_args.target = "sglang"
    reference_args.image = ""
    reference_args.candidate_snapshot = None
    reference_args.serve_extra_override = plain_reference_serve_extra(contract)
    reference_args.server_env = plain_reference_server_env()
    print("\n  private correctness stage on the plain checkpoint", flush=True)
    try:
        host, port = launch(contract, reference_args)
        row["correctness"] = {"replay_fidelity": full_replay_fidelity}
        return reference_stage(
            contract, reference_url=f"http://{host}:{port}",
            artifact_dir=artifact_dir, model_dir=Path(args.model_dir),
            tokenizer=tokenizer, row=row)
    finally:
        teardown()
