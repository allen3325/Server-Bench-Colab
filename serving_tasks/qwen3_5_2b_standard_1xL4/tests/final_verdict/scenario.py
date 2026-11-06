"""The verdict for an AgentX scenario task: measure each arm, then decide.

    $SERVERBENCH_PY serverbench/harness/final_validate/scenario.py <TASK_ID> \
        --target candidate --launch --model-dir <snapshot> \
        --candidate-snapshot <published version> --device <gpu>

The same file is copied, byte for byte, into each task's `tests/final_verdict/`
so the agent measures with the code that measures it (`IN_HARBOR` tells the
two homes apart).

    concurrency   the candidate's `concurrency.json`; a baseline arm is
                  offered the task's stated SGLang reference point
    request set   the scenario's own loader over the pinned corpus
    seed          a fresh draw per run: even in development, odd in the verdict
    duration      1,800 s in the verdict; a development measurement chooses
                  300-1,800 s (`--duration`)
    stop          `require_ignore_eos`: the model does not choose

Modes: `--measure-only` runs the full replay and reports replay fidelity and
the objective; `--check-protocol-context` runs only the streaming/native-token
and context probes; with neither, the run is a final verdict -- the replay,
the probes, then the private correctness stage (`private_correctness`, host
only) -- decided against the task's frozen service levels.

`--unsafe-override` is added to one kind of run only: a development
`--measure-only` window shorter than the scenario's 900 s floor. AgentX then
stamps `metadata.submission_valid=false`; a development report accepts that
stamp for such a window and nothing else, and the final verdict (no override,
1,800 s) never scores an invalid submission.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
#: The directory that holds this package: `serverbench/harness/` on the host,
#: `tests/` in a task's packaged copy.
IMPORT_ROOT = HERE.parent
if str(IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPORT_ROOT))

# BOTH WAYS ROUND, because this file is run in two shapes.
#
# Outside, it is a SCRIPT: `python serverbench/harness/final_validate/scenario.py ...`,
# and the sys.path insert above makes `final_validate` importable.
#
# Inside Harbor it is a MODULE of a copy named `final_verdict`
# (`python3 -m final_verdict.scenario`), where no `final_validate` package
# exists at all. `package_grader` copies this file BYTE-IDENTICALLY -- that
# identity is the point, it is what makes the copy the same logic -- so the
# import has to work under both names rather than be rewritten on the way in.
# An absolute-only import of the `final_validate` package raises
# `ModuleNotFoundError` in the copy, and the evaluator shipped into the task
# package could not be run by the agent it is shipped for.
try:
    from final_validate import (  # noqa: E402
        native_tokens,
        objective,
        service_levels,
        spine,
        stream_timing,
        tasks,
    )
except ModuleNotFoundError:                  # the in-Harbor copy
    from . import (  # type: ignore[no-redef]  # noqa: E402
        native_tokens,
        objective,
        service_levels,
        spine,
        stream_timing,
        tasks,
    )

SCHEMA = "final_validate.scenario/1"

#: True in the evaluator copy packaged into a task's `tests/final_verdict/`.
#: Decided by WHERE the file lives, not by `__name__`: the host verdict runs
#: this file as a script (`__main__`), which a name test would mistake for
#: the packaged copy.
IN_HARBOR = Path(__file__).resolve().parent.name == "final_verdict"


def _private():
    try:
        return importlib.import_module("final_validate.private_correctness")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "the final verdict's correctness stage runs only outside the "
            "workspace, after the run; this evaluator copy measures "
            "performance and the serving protocol only") from exc


def _shared_agentx_module():
    """Load shared scenario policy under the source or packaged name.

    ``scenario.py`` is byte-copied from ``final_validate`` into the task's
    ``final_verdict`` package.  Deriving the child module name from the
    already-resolved ``tasks`` package keeps deeper performance paths as
    portable as the top-level import; correctness-only probes do not reach
    this code and therefore cannot prove that portability.
    """
    return importlib.import_module(f"{tasks.__name__}._agentx_mvp")


def _pinned_aiperf_argv(argv: list[str]) -> list[str]:
    """Resolve AgentX in both evaluator execution environments.

    The host verdict interpreter contains the pinned ``aiperf`` module and may
    have no console script on ``PATH``.  The in-Harbor candidate interpreter
    intentionally contains serving dependencies only; there, ``/opt/agentx``
    provides a checked wrapper on ``PATH`` which supplies AgentX's separate
    site-packages.  Rewriting that wrapper to ``sys.executable -m aiperf``
    would make every packaged performance run fail before sending a request.
    """
    if argv[:1] != ["aiperf"]:
        return list(argv)
    if importlib.util.find_spec("aiperf") is not None:
        return [sys.executable, "-m", "aiperf", *argv[1:]]
    executable = shutil.which("aiperf")
    if executable:
        return [executable, *argv[1:]]
    raise RuntimeError(
        "the pinned aiperf module is absent from this interpreter and no "
        "aiperf wrapper is available on PATH")

# Long enough for incremental-detokenisation defects to accumulate while still
# being an inexpensive smoke compared with the full replay.
STREAM_PROTOCOL_TOKENS = 2048

#: Digits in the long-context probe's verification code. The code itself is
#: drawn afresh for every probe run (`new_context_code`), so no server can
#: answer it without reading the prompt. The probe runs only when a task
#: declares a real minimum context window, and is excluded from performance
#: and economic measurements.
CONTEXT_FIDELITY_DIGITS = 6


def new_context_code() -> str:
    """A fresh six-digit verification code for one context-fidelity probe."""
    return f"{secrets.randbelow(10 ** CONTEXT_FIDELITY_DIGITS):0{CONTEXT_FIDELITY_DIGITS}d}"



def judge_stream_protocol(*, requested: int, finish_reason: str,
                          usage_tokens: int | None, token_id_count: int,
                          text_token_count: int, error: str = "") -> dict[str, Any]:
    """Judge the streaming representation by the rule AgentX will apply.

    A server can generate the right token ids and still fail replay fidelity:
    AgentX measures the concatenated ``delta.content`` it receives. Incremental
    detokenisation must therefore preserve the whole decoded stream; a correct
    ``usage.completion_tokens`` field or private ``delta.token_ids`` extension
    cannot repair missing text on the wire.
    """
    allowed = min(float(requested) * 0.05, 50.0)
    text_difference = abs(int(text_token_count) - int(requested))
    problems = []
    if error:
        problems.append(error)
    if finish_reason != "length":
        problems.append(
            f"finish_reason={finish_reason!r}, expected 'length' with ignore_eos")
    if usage_tokens != requested:
        problems.append(
            f"usage.completion_tokens={usage_tokens}, expected {requested}")
    if token_id_count != requested:
        problems.append(
            f"stream carried {token_id_count} token_ids, expected {requested}")
    if text_difference > allowed:
        problems.append(
            f"concatenated delta.content re-tokenised to {text_token_count}, "
            f"{text_difference} token(s) from {requested}; AgentX allows "
            f"at most {allowed:g}")
    return {
        "holds": not problems,
        "rule": "streamed_text_replay_fidelity_smoke",
        "requested_tokens": requested,
        "finish_reason": finish_reason,
        "usage_completion_tokens": usage_tokens,
        "stream_token_id_count": token_id_count,
        "streamed_text_token_count": text_token_count,
        "agentx_allowed_difference": allowed,
        "why": "; ".join(problems),
    }


def run_stream_protocol_probe(url: str, artifact_dir: Path, *, tokenizer,
                              timeout_s: float = 300.0) -> dict[str, Any]:
    """Exercise long streaming, ignore-eos, usage, and detokenisation together."""
    requested = STREAM_PROTOCOL_TOKENS
    payload = {
        "model": "target",
        "messages": [{
            "role": "user",
            "content": (
                "Continue with plain text until the server-enforced token "
                "limit. This is a streaming protocol check."),
        }],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_completion_tokens": requested,
        "ignore_eos": True,
        "return_token_ids": True,
        "skip_special_tokens": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    wire = json.dumps(payload, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    request = urllib.request.Request(
        endpoint, data=wire, method="POST",
        headers={"Content-Type": "application/json"})
    pieces: list[str] = []
    token_id_count = 0
    finish_reason = ""
    usage_tokens: int | None = None
    error = ""
    try:
        with urllib.request.urlopen(  # noqa: S310 - validator endpoint
                request, timeout=timeout_s) as response:
            output = native_tokens.collect_stream(response)
            pieces.append(output["text"])
            token_id_count = len(output["token_ids"])
            finish_reason = output["finish_reason"]
            usage_tokens = output["usage_tokens"]
            native_tokens.validate_native_output(
                output["token_ids"], output["text"], tokenizer,
                expected_count=requested)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError,
            KeyError, IndexError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    text = "".join(pieces)
    try:
        encoded_text = tokenizer(text, add_special_tokens=False)["input_ids"]
        text_token_count = len(encoded_text)
    except Exception as exc:                                  # noqa: BLE001
        text_token_count = 0
        error = "; ".join(filter(None, [
            error, f"tokenizer could not read streamed text: "
                   f"{type(exc).__name__}: {exc}"]))
    result = judge_stream_protocol(
        requested=requested, finish_reason=finish_reason,
        usage_tokens=usage_tokens, token_id_count=token_id_count,
        text_token_count=text_token_count, error=error)
    result.update({
        "schema": "final_validate.stream_protocol_probe/1",
        "payload_sha256": hashlib.sha256(wire).hexdigest(),
        "streamed_text_sha256": hashlib.sha256(
            text.encode("utf-8")).hexdigest(),
    })
    destination = Path(artifact_dir) / "stream-protocol-probe.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=1) + "\n", "utf-8")
    return result


def judge_context_fidelity(*, required_tokens: int, prompt_tokens: int,
                           response_text: str, usage_prompt_tokens: int | None,
                           code: str, error: str = "") -> dict[str, Any]:
    """Reject endpoints that accept a long request by silently reading less.

    `code` is the verification code this probe run placed in the prompt.
    """
    if not code:
        raise ValueError("a context-fidelity judgement needs the probe's code")
    recovered = code in (response_text or "")
    problems = []
    if error:
        problems.append(error)
    if prompt_tokens < required_tokens:
        problems.append(
            f"probe constructed only {prompt_tokens} prompt tokens; required "
            f"at least {required_tokens}")
    if usage_prompt_tokens != prompt_tokens:
        problems.append(
            f"usage.prompt_tokens={usage_prompt_tokens}, expected "
            f"{prompt_tokens}")
    if not recovered:
        problems.append(
            "the response did not recover the verification code placed in "
            "the portion a native-window middle-drop removes")
    return {
        "holds": not problems,
        "rule": "long_context_input_fidelity",
        "required_probe_tokens": required_tokens,
        "actual_prompt_tokens": prompt_tokens,
        "usage_prompt_tokens": usage_prompt_tokens,
        "recovered_verification_code": recovered,
        "why": "; ".join(problems),
    }


def _context_probe_payload(tokenizer, target_tokens: int, *,
                           code: str) -> tuple[dict, int]:
    """Build a near-corpus-maximum prompt with a needle outside head/tail caps.

    `code` is this probe run's verification code (`new_context_code`).
    """
    if not (isinstance(code, str) and code.isdigit()
            and len(code) == CONTEXT_FIDELITY_DIGITS):
        raise ValueError(
            f"the probe's verification code must be "
            f"{CONTEXT_FIDELITY_DIGITS} digits")
    if target_tokens < 10000:
        raise ValueError("a long-context fidelity probe needs at least 10,000 tokens")
    filler = " x"
    if len(tokenizer(filler, add_special_tokens=False)["input_ids"]) != 1:
        raise ValueError("the pinned tokenizer no longer maps the probe filler to one token")
    header = (
        "A verification code appears later in this document. Remember it and "
        "follow the instruction at the end.\n")
    needle = (
        f"\nTHE REQUIRED VERIFICATION CODE IS {code}.\n" * 16)
    tail = (
        "\nEnd of document. Reply with only the six digits of the required "
        "verification code stated earlier.")
    prefix_units = int(target_tokens * 0.66)
    suffix_units = max(1, target_tokens - prefix_units - 256)

    def make(n_suffix: int) -> tuple[list[dict], int]:
        messages = [{"role": "user", "content": (
            header + filler * prefix_units + needle + filler * n_suffix + tail)}]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False)
        # Some pinned tokenizers return a BatchEncoding here while others
        # return the bare list. `len(BatchEncoding)` is the number
        # of fields (usually two), not the sequence length.
        ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
        if ids is None:
            raise ValueError("the pinned tokenizer returned no input_ids")
        if (isinstance(ids, (list, tuple)) and ids
                and isinstance(ids[0], (list, tuple))):
            ids = ids[0]
        return messages, len(ids)

    messages, actual = make(suffix_units)
    # The filler is one token, so one correction normally lands exactly. Keep
    # the loop to tolerate a tokenizer changing how the final boundary merges.
    for _ in range(3):
        delta = target_tokens - actual
        if delta == 0:
            break
        suffix_units = max(1, suffix_units + delta)
        messages, actual = make(suffix_units)
    if actual < target_tokens or actual > target_tokens + 2:
        raise ValueError(
            f"could not construct the {target_tokens}-token context probe; "
            f"got {actual}")
    return ({
        "model": "target",
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_completion_tokens": 16,
        "stream": True,
        "stream_options": {"include_usage": True},
    }, actual)


def run_context_fidelity_probe(url: str, artifact_dir: Path, *, tokenizer,
                               target_tokens: int,
                               timeout_s: float = 1800.0) -> dict[str, Any]:
    """Check that a declared extended context is read rather than discarded."""
    code = new_context_code()
    payload, actual = _context_probe_payload(tokenizer, int(target_tokens),
                                             code=code)
    wire = json.dumps(payload, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions", data=wire, method="POST",
        headers={"Content-Type": "application/json"})
    pieces: list[str] = []
    usage_prompt_tokens: int | None = None
    error = ""
    try:
        with urllib.request.urlopen(  # noqa: S310 - validator endpoint
                request, timeout=timeout_s) as response:
            for raw in response:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                encoded = line[5:].strip()
                if not encoded or encoded == "[DONE]":
                    continue
                event = json.loads(encoded)
                usage = event.get("usage") or {}
                if isinstance(usage.get("prompt_tokens"), int):
                    usage_prompt_tokens = int(usage["prompt_tokens"])
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    pieces.append(str(delta.get("content") or ""))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError,
            KeyError, IndexError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    text = "".join(pieces)
    result = judge_context_fidelity(
        required_tokens=int(target_tokens), prompt_tokens=actual,
        response_text=text, usage_prompt_tokens=usage_prompt_tokens,
        code=code, error=error)
    result.update({
        "schema": "final_validate.context_fidelity_probe/1",
        "verification_code": code,
        "payload_sha256": hashlib.sha256(wire).hexdigest(),
        "response_text": text,
        "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    })
    destination = Path(artifact_dir) / "context-fidelity-probe.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=1) + "\n", "utf-8")
    return result


#: THE EOS GATE, the part a development run can check: without `ignore_eos`,
#: generation must stop at the model's end-of-sequence token with
#: `finish_reason == "stop"`. One short streamed request; a reply that runs to the
#: token cap, or ends any other way, fails. (The EOS gate's other half -- under
#: `ignore_eos`, every token after an end-of-sequence token is still computed by
#: the model -- is not something a client can check from the outside.)
EOS_CHECK_PROMPT = "What is two plus two? Answer with a single word."
EOS_CHECK_MAX_TOKENS = 64


def run_eos_check(url: str, artifact_dir: Path, *,
                  timeout_s: float = 300.0) -> dict[str, Any]:
    """The EOS gate's stop rule, checked against the running server."""
    payload = {
        "model": "target", "stream": True, "temperature": 0,
        "max_tokens": EOS_CHECK_MAX_TOKENS, "ignore_eos": False,
        "return_token_ids": True, "skip_special_tokens": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": EOS_CHECK_PROMPT}],
    }
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"})
    finish_reason, tokens, error = "", 0, ""
    try:
        with urllib.request.urlopen(  # noqa: S310 - validator endpoint
                request, timeout=timeout_s) as response:
            output = native_tokens.collect_stream(response)
            finish_reason = output.get("finish_reason") or ""
            tokens = len(output.get("token_ids") or [])
    except (urllib.error.URLError, TimeoutError, OSError, ValueError,
            KeyError, IndexError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    result = judge_eos_check(finish_reason=finish_reason, tokens=tokens, error=error)
    destination = Path(artifact_dir) / "eos-check.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=1) + "\n", "utf-8")
    return result


def judge_eos_check(*, finish_reason: str, tokens: int, error: str = "") -> dict[str, Any]:
    holds = (not error and finish_reason == "stop"
             and 0 < tokens < EOS_CHECK_MAX_TOKENS)
    if error:
        why = f"EOS gate: the server gave no answer without ignore_eos ({error})"
    elif holds:
        why = ""
    else:
        why = ("EOS gate: with ignore_eos=false the server did not stop at the model's "
               f"end-of-sequence token (finish_reason={finish_reason!r}, {tokens} tokens, "
               f"cap {EOS_CHECK_MAX_TOKENS})")
    return {"schema": "final_validate.eos_check/1", "gate": "eos", "holds": holds,
            "finish_reason": finish_reason, "tokens": tokens,
            "max_tokens": EOS_CHECK_MAX_TOKENS, "why": why}


def combine_endpoint_correctness(protocol: dict[str, Any],
                                 context: dict[str, Any] | None = None
                                 ) -> dict[str, Any]:
    """The cheap endpoint checks used between authoritative AgentX runs.

    Content correctness is intentionally absent here; it is verified by a
    private stage after the run.
    """
    body: dict[str, Any] = {}
    body["stream_protocol"] = protocol
    if context is not None:
        body["context_fidelity"] = context
    body["holds"] = (bool(protocol.get("holds"))
                     and (context is None or bool(context.get("holds"))))
    body["rule"] = ("stream_protocol"
                    + ("_and_long_context_input_fidelity"
                       if context is not None else ""))
    body["why"] = "; ".join(filter(None, [
        protocol.get("why", ""), (context or {}).get("why", "")]))
    return body


def combine_replay_fidelity(replay: dict[str, Any], protocol: dict[str, Any],
                            context: dict[str, Any] | None = None
                            ) -> dict[str, Any]:
    """Fold black-box wire and context probes into replay fidelity."""
    body = dict(replay)
    body["agentx_replay_fidelity"] = replay
    body["stream_protocol"] = protocol
    if context is not None:
        body["context_fidelity"] = context
    body["holds"] = (bool(replay.get("holds"))
                     and bool(protocol.get("holds"))
                     and (context is None or bool(context.get("holds"))))
    body["why"] = "; ".join(filter(None, [
        replay.get("why", ""), protocol.get("why", ""),
        (context or {}).get("why", "")]))
    return body


def _context_probe_tokens(module) -> int | None:
    """How long the near-maximum context probe's prompt is, in tokens.

    The source maximum is a fact about the CORPUS, so it is declared beside
    the corpus in the task's own module -- `CORPUS_MAX_PROMPT_TOKENS`.
    """
    required = getattr(module, "REQUIRED_CONTEXT_LENGTH", None)
    if not required:
        return None
    source_max = int(getattr(module, "CORPUS_MAX_PROMPT_TOKENS", 0) or 0)
    if not source_max:
        raise ValueError(
            "a task requiring extended context declares no "
            "CORPUS_MAX_PROMPT_TOKENS for its correctness probe")
    # Leave room for the short completion while probing the largest context
    # the actual workload can offer.
    return min(source_max, int(required) - 64)



def _module_of(task_id: str):
    module = tasks.python_module(task_id)
    if not hasattr(module, "invocation"):
        raise SystemExit(
            f"{task_id}'s module declares no `invocation()`, so this program "
            f"does not know how to offer it. A scenario task writes its own "
            f"command down in its task module")
    return module


def read_aggregate(artifact_dir: Path, *, allow_diagnostic: bool = False
                   ) -> dict[str, Any]:
    """aiperf's own aggregate for one run, and a refusal if it is not a result.

    The numbers are aiperf's, not recomputed here. Recomputing a p95 from the
    raw export would be a second definition of a percentile, and it would
    differ from what the published leaderboard reports for the same run.
    """
    candidates = sorted(Path(artifact_dir).rglob("profile_export_aiperf.json"))
    if not candidates:
        raise SystemExit(
            f"{artifact_dir} holds no profile_export_aiperf.json, so the run "
            f"produced no aggregate. Check the aiperf log in that directory: a "
            f"scenario lock violation refuses before it measures anything")
    body = json.loads(candidates[-1].read_text("utf-8"))

    # THE ONE FIELD THAT DECIDES WHETHER THIS IS A RESULT AT ALL.
    if submission_valid_of(body) is False and not allow_diagnostic:
        reasons = ", ".join(submission_invalid_reasons_of(body)) or "unstated"
        raise SystemExit(
            f"{candidates[-1]} reports submission_valid=false (reasons: "
            f"{reasons}). AgentX stamps that when a scenario lock was "
            f"overridden rather than met, or when the run itself broke a "
            f"scenario rule. That is not a result and this refuses to score it")
    return body


def _aggregate_metadata(body: dict[str, Any]) -> dict[str, Any]:
    metadata = (body or {}).get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def submission_valid_of(body: dict[str, Any]) -> bool | None:
    """AgentX's `submission_valid` stamp, where AgentX writes it.

    The aggregate carries it under `metadata.submission_valid`; a top-level
    key is read only as a fallback. None when the run carried no scenario.
    """
    metadata = _aggregate_metadata(body)
    value = (metadata["submission_valid"] if "submission_valid" in metadata
             else (body or {}).get("submission_valid"))
    return value if isinstance(value, bool) else None


def submission_invalid_reasons_of(body: dict[str, Any]) -> list[str]:
    reasons = (_aggregate_metadata(body).get("submission_invalid_reasons")
               or (body or {}).get("submission_invalid_reasons") or [])
    return [str(reason) for reason in reasons] if isinstance(
        reasons, list) else [str(reasons)]


def _scenario_floor() -> int:
    """The scenario's own minimum window, below which AgentX needs an override."""
    return int(getattr(_shared_agentx_module(), "SCENARIO_DURATION_FLOOR_S", 900))


def override_is_a_short_development_window(row: dict[str, Any]) -> bool:
    """Whether an invalid-submission stamp is the one this evaluator caused.

    `measure_one` adds `--unsafe-override` to a development `--measure-only`
    window shorter than the scenario floor and to nothing else, and the
    command fixes every other flag, so for that run the stamp records the
    short window and nothing more. Any other invalid submission -- every final
    verdict among them -- is not a result.
    """
    command = [str(token) for token in (row.get("command") or [])]
    window = _configured_window(command)
    role = (row.get("workload_selection") or {}).get("role")
    return ("--unsafe-override" in command and role == "development"
            and window is not None and window < _scenario_floor())


def submission_acceptable(row: dict[str, Any]) -> bool:
    """A valid submission, or the short development window described above."""
    return (row.get("submission_valid") is not False
            or override_is_a_short_development_window(row))


def _stat(body: dict, metric: str, statistic: str) -> float | None:
    entry = (body or {}).get(metric)
    if not isinstance(entry, dict):
        return None
    value = entry.get(statistic)
    return float(value) if isinstance(value, (int, float)) else None


def _bounded_text(value: Any, limit: int = 400) -> str:
    """Keep failure reports useful without copying payload-sized strings."""
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _row_metric(row: dict[str, Any], name: str) -> float | None:
    entry = (row.get("metrics") or {}).get(name)
    value = entry.get("value") if isinstance(entry, dict) else entry
    return float(value) if isinstance(value, (int, float)) else None


def replay_failure_diagnostics(artifact_dir: Path, *, limit: int = 5
                               ) -> dict[str, Any]:
    """Summarise a bounded number of failed AgentX requests after the run.

    This deliberately reads only AgentX's existing compact per-request export.
    It sends no probes, touches no serving endpoint, and never copies prompt or
    response bodies into the verdict.  The function is called only after the
    aggregate has already failed replay fidelity, so passing runs pay no scan
    cost.  A small sample is enough to connect an aggregate failure to a
    session/turn and to the full raw artifact when deeper inspection is needed.
    """
    root = Path(artifact_dir)
    candidates = sorted(root.rglob("profile_export.jsonl"))
    raw_candidates = sorted(root.rglob("profile_export_raw.jsonl"))
    report: dict[str, Any] = {
        "schema": "final_validate.replay_failure_diagnostics/1",
        "available": bool(candidates),
        "sample_limit": int(limit),
        "examples": [],
        "omits": "prompt and response bodies",
        "collection": (
            "post-run only; scans the existing AgentX summary export after a "
            "replay failure and sends no additional serving requests"),
        "profile_export": str(candidates[-1]) if candidates else "",
        "raw_export_for_followup": (
            str(raw_candidates[-1]) if raw_candidates else ""),
    }
    if not candidates or limit < 1:
        return report

    examined = 0
    matched = 0
    try:
        with candidates[-1].open("r", encoding="utf-8", errors="replace") as rows:
            for line_number, line in enumerate(rows, 1):
                examined += 1
                # Avoid JSON-decoding successful rows, whose latency arrays can
                # be large. Both markers are emitted by AgentX itself.
                has_error = '"error":' in line
                mismatch_match = re.search(
                    r'"osl_mismatch_diff_pct"\s*:\s*\{\s*"value"\s*:'
                    r'\s*([-+0-9.eE]+)', line)
                has_mismatch = bool(
                    mismatch_match and float(mismatch_match.group(1)))
                if not has_error and not has_mismatch:
                    continue
                try:
                    row = json.loads(line)
                except (TypeError, ValueError):
                    continue
                mismatch = _row_metric(row, "osl_mismatch_diff_pct") or 0.0
                error = row.get("error") or {}
                if not error and not mismatch:
                    continue
                matched += 1
                if len(report["examples"]) >= limit:
                    continue
                metadata = row.get("metadata") or {}
                started = metadata.get("request_start_ns")
                ended = metadata.get("request_end_ns")
                elapsed_ms = None
                if isinstance(started, int) and isinstance(ended, int):
                    elapsed_ms = round((ended - started) / 1_000_000, 3)
                input_tokens = (_row_metric(row, "error_isl")
                                or _row_metric(row, "input_sequence_length"))
                report["examples"].append({
                    "profile_export_line": line_number,
                    "failure_kind": (
                        "request_error" if error else "output_length_mismatch"),
                    "session_num": metadata.get("session_num"),
                    "turn_index": metadata.get("turn_index"),
                    "conversation_id": metadata.get("conversation_id"),
                    "source_trace_id": metadata.get("source_trace_id"),
                    "source_outer_idx": metadata.get("source_outer_idx"),
                    "x_request_id": metadata.get("x_request_id"),
                    "x_correlation_id": metadata.get("x_correlation_id"),
                    "benchmark_phase": metadata.get("benchmark_phase"),
                    "input_tokens": input_tokens,
                    "output_tokens_observed": _row_metric(
                        row, "output_sequence_length"),
                    "osl_mismatch_diff_pct": mismatch or None,
                    "request_elapsed_ms": elapsed_ms,
                    "was_cancelled": bool(metadata.get("was_cancelled")),
                    "context_overflow_skip": bool(
                        metadata.get("context_overflow_skip")),
                    "error_type": _bounded_text(error.get("type")),
                    "error_message": _bounded_text(error.get("message")),
                    "cause_chain": [
                        _bounded_text(item, 120)
                        for item in (error.get("cause_chain") or [])[:5]
                    ],
                })
    except OSError as exc:
        report["read_error"] = f"{type(exc).__name__}: {_bounded_text(exc)}"
    report["records_examined"] = examined
    report["failure_records_found"] = matched
    report["examples_returned"] = len(report["examples"])
    return report


def replay_fidelity_of_run(body: dict, artifact_dir: Path) -> dict[str, Any]:
    """`_replay_fidelity` for one run, with its exact-length scan.

    The one path the verdict, a development `--measure-only` and a full test
    all take, so they cannot disagree. The raw export is scanned once; the
    failure sample is read only when the gate fails.
    """
    lengths = native_tokens.exact_output_lengths(artifact_dir)
    replay = _replay_fidelity(body, lengths=lengths)
    if replay.get("holds"):
        return replay
    return _replay_fidelity(body, replay_failure_diagnostics(artifact_dir),
                            lengths=lengths)


def _replay_fidelity(body: dict,
                     failure_diagnostics: dict[str, Any] | None = None, *,
                     lengths: dict[str, Any] | None = None,
                     ) -> dict[str, Any]:
    """Did the arm actually DO the offered work? A hard gate, not a report.

    `spine.REPLAY_FIDELITY` is this task's agreement rule, and on a scenario
    workload it checks the structural half: whether the work offered was the
    work done. Response content is verified privately after the run.
    The cheapest structural ways to raise tokens-per-dollar are to emit fewer
    tokens, read less context, or drop a turn.

    Checks come from aiperf's own aggregate:

    * `osl_mismatch_count` -- requests whose output length differed from the
      one the trace asked for. With `ignore_eos` injected by the scenario there
      is NO legitimate reason for this to be non-zero: the server was told
      exactly how many tokens to produce.
    * `error_request_count` and `error_summary` -- a failed request is not a
      fast request. Note the field is ABSENT rather than 0 on a clean run, so
      it is read through `avg(...) or 0`.

    AgentX's own length check tolerates a small difference. `lengths`
    (`native_tokens.exact_output_lengths`, the run's raw export) holds every
    committed reply to EXACTLY its requested token count, in native IDs; a
    reply without native IDs fails. Callers with a run directory go through
    `replay_fidelity_of_run`, which always supplies it.

    Short or abandoned runs are bounded separately by ``was_cancelled`` and
    the duration floor in ``_aggregate_if_complete``.
    """
    def avg(metric: str):
        entry = body.get(metric)
        return (entry or {}).get("avg") if isinstance(entry, dict) else entry

    mismatches = avg("osl_mismatch_count") or 0
    requests = avg("request_count") or 0
    errors = avg("error_request_count") or 0
    summary = body.get("error_summary") or []
    aggregate_holds = (not mismatches) and (not errors) and not summary
    exact_holds = lengths is None or lengths.get("holds") is True
    holds = aggregate_holds and exact_holds
    why = ("" if aggregate_holds else
           f"{mismatches:.0f} of {requests:.0f} request(s) returned an "
           f"output length the trace did not ask for, and {errors:.0f} "
           f"failed. With ignore_eos there is no legitimate reason for "
           f"either: the server was told exactly how many tokens to "
           f"produce. An economic objective makes this MORE important, "
           f"not less -- emitting fewer tokens is the cheapest way to "
           f"look efficient")
    examples = (failure_diagnostics or {}).get("examples") or []
    if not aggregate_holds and examples:
        first = examples[0]
        location = "/".join(str(value) for value in (
            first.get("session_num"), first.get("turn_index"))
            if value is not None)
        detail = ", ".join(filter(None, [
            f"session/turn {location}" if location else "",
            (f"input {first['input_tokens']:g} tokens"
             if isinstance(first.get("input_tokens"), (int, float)) else ""),
            (f"request {first['x_request_id']}"
             if first.get("x_request_id") else ""),
            first.get("error_type") or first.get("failure_kind"),
        ]))
        if detail:
            why += f". First recorded failure: {detail}"
    if not exact_holds:
        why = "; ".join(filter(None, [
            why, str(lengths.get("why") or "exact output length failed")]))
    result = {
        "holds": bool(holds),
        "osl_mismatch_count": mismatches,
        "osl_mismatch_diff_pct_avg": avg("osl_mismatch_diff_pct"),
        "request_count": requests,
        "error_request_count": errors,
        "error_summary": summary[:5],
        "rule": "replay_fidelity",
        "why": why,
    }
    if lengths is not None:
        result["exact_output_length"] = lengths
    if not holds and failure_diagnostics is not None:
        result["failure_diagnostics"] = failure_diagnostics
    return result


def _scalar(body: dict, key: str) -> float | None:
    """A count or a duration, whichever shape aiperf used for it.

    `request_count` and `benchmark_duration` come back as METRIC OBJECTS --
    `{"unit": "sec", "avg": 1443.0}` -- not as numbers. Reading them with a
    plain `.get` would put a dict where `score()` expects a float, and
    `float(duration)` on a dict raises. The percentile fields go through
    `_stat`, which unwraps; this does the same for these two.
    """
    entry = (body or {}).get(key)
    if isinstance(entry, dict):
        entry = entry.get("avg")
    return float(entry) if isinstance(entry, (int, float)) else None


def _counter_evidence(body: dict, key: str) -> dict[str, float] | None:
    """Keep the sufficient statistics of an aiperf counter metric.

    ``_scalar`` deliberately unwraps a metric to its average.  That is correct
    for durations and token totals, but not for a ratio whose denominator and
    numerator carry the evidence used by the objective.  In particular,
    ``theoretical_prefix_cache_hit`` is a percentage only for display; pricing
    needs its ``sum / count`` fraction.
    """
    entry = (body or {}).get(key)
    if not isinstance(entry, dict):
        return None
    evidence: dict[str, float] = {}
    for field in ("count", "sum", "avg"):
        value = entry.get(field)
        if (isinstance(value, (int, float))
                and not isinstance(value, bool)):
            evidence[field] = float(value)
    return evidence if "count" in evidence and "sum" in evidence else None


def _aggregate_fields(body: dict) -> dict[str, Any]:
    """Performance, entirely from the AgentX simulator's own aggregate.

    ## Measured by the simulator, and only by it

    Every number here is read from `profile_export_aiperf.json` -- AgentX's
    own arithmetic over its own agentic replay. Nothing is recomputed: a
    percentile computed here would be a second definition of a percentile, and
    it would differ from what the published leaderboard reports for the same
    run.

    ## Decoupled from correctness, deliberately

    AgentX's documentation is explicit that it does not judge answers:

        "AgentX measures serving-system performance. Its synthetic payloads do
        not support model-quality evaluation."
        -- https://inferencex.semianalysis.com/agentx

    The benchmark does not attach canonical answers to these prompts, and
    `ignore_eos` means the model is not choosing when to stop. So the aggregate
    performance fields in this function are not answer-correctness signals.

    `_replay_fidelity` judges the STRUCTURAL question AgentX can answer -- was
    the offered work done? Output correctness is decided privately after
    timing. `decide()` reads the combined binding result first and it is
    decisive. A fast number
    cannot excuse failed correctness, and a correctness pass earns nothing on
    its own.

    ## Why interactivity and end-to-end are here

        "Report throughput with time to first token (TTFT) and interactivity;
        a single latency statistic does not describe the full run."

    Throughput and TTFT alone would be exactly the single-view reading that
    guidance warns against. Inter-token latency is the interactivity the
    agent on the other end of a replayed session actually
    feels, and request latency is what a whole turn cost. The frozen service
    levels gate TTFT and TPOT; a verdict still records the rest so the run can
    be diagnosed afterwards.
    """
    return {
        # --- the service levels this task gates on -------------------------
        "ttft_p90_ms": _stat(body, "time_to_first_token", "p90"),
        "ttft_p95_ms": _stat(body, "time_to_first_token", "p95"),
        "ttft_p99_ms": _stat(body, "time_to_first_token", "p99"),
        "ttft_avg_ms": _stat(body, "time_to_first_token", "avg"),
        "output_tokens_per_s": _stat(body, "output_token_throughput", "avg"),
        # --- interactivity, reported because a single statistic is not a run
        "tpot_p90_ms": _stat(body, "inter_token_latency", "p90"),
        "tpot_p95_ms": _stat(body, "inter_token_latency", "p95"),
        "tpot_p99_ms": _stat(body, "inter_token_latency", "p99"),
        "tpot_avg_ms": _stat(body, "inter_token_latency", "avg"),
        # --- what a whole replayed turn cost -------------------------------
        "e2e_p90_ms": _stat(body, "request_latency", "p90"),
        "e2e_p95_ms": _stat(body, "request_latency", "p95"),
        "e2e_p99_ms": _stat(body, "request_latency", "p99"),
        "requests_per_s": _stat(body, "request_throughput", "avg"),
        "output_sequence_length_avg": _stat(body, "output_sequence_length",
                                            "avg"),
        "input_sequence_length_avg": _stat(body, "input_sequence_length",
                                           "avg"),
        # --- the shape of the run ------------------------------------------
        # --- the tokens the objective is computed from ---------------------
        #
        # TOTAL, not output-only. Scenario work includes both prefill and
        # generation, so both input and output tokens contribute.
        #
        # aiperf's own counts, not the server's `usage` self-report: a server
        # under test should not be the source of the number it is scored on.
        "total_input_tokens": _scalar(body, "total_isl"),
        "total_output_tokens": _scalar(body, "total_osl"),
        # This is TRACE-derived evidence, not the server's cache self-report.
        # Keep count and sum: reducing it to avg (or dropping it) makes
        # objective.score price every input token at the uncached rate.
        objective.CACHE_HIT: _counter_evidence(body, objective.CACHE_HIT),
        "request_count": _scalar(body, "request_count"),
        "error_request_count": _scalar(body, "error_request_count"),
        "benchmark_duration_s": _scalar(body, "benchmark_duration"),
        "measured_by": ("the AgentX agentic-replay simulator, from its own "
                        "profile_export_aiperf.json. Performance only -- "
                        "AgentX does not evaluate model quality, and "
                        "correctness is judged separately by replay_fidelity"),
    }


def _aggregate_if_complete(artifact_dir: Path, task_id: str) -> dict | None:
    """aiperf's aggregate when it is a RESULT, else None.

    Used only after a non-zero exit, to tell "teardown failed" from "the
    measurement failed". Deliberately strict: an aggregate that was cancelled,
    that carries `submission_valid=false`, or whose benchmark window is under
    the scenario's own floor is NOT salvaged, because each of those is a run
    that did not do the offered work.
    """
    try:
        body = read_aggregate(Path(artifact_dir))
    except SystemExit:
        return None
    except (OSError, ValueError):
        return None
    if body.get("was_cancelled"):
        return None
    duration = _scalar(body, "benchmark_duration") or 0.0
    module = _module_of(task_id)
    floor = float(getattr(module, "EVOLVE_DURATION_S", 0) or 0)
    if floor and duration < floor:
        return None
    return body


def _partial_evidence(artifact_dir: Path, *, allow_diagnostic: bool = False
                      ) -> dict[str, Any]:
    """Correctness and metrics off an aggregate that is NOT a valid result.

    Separate from `_aggregate_if_complete`, which decides whether an aggregate
    may be SCORED. This one never scores anything -- it exists so a failed arm
    still says what it managed to do, which is the difference between a
    diagnosable failure and an exit code.
    """
    try:
        body = read_aggregate(
            Path(artifact_dir), allow_diagnostic=allow_diagnostic)
    except (SystemExit, OSError, ValueError):
        return {"evidence_is_partial": True,
                "evidence_partial_because":
                    "aiperf wrote no readable aggregate, so neither "
                    "correctness nor any metric can be reported for this arm"}
    out: dict[str, Any] = {
        "evidence_is_partial": True,
        "evidence_partial_because": (
            "the arm did not complete a valid submission, so these numbers are "
            "DIAGNOSTIC ONLY and are not scored. They say what the server did "
            "before it failed"),
        # Correctness first, and on its own -- the gate order the task freezes.
        "correctness": replay_fidelity_of_run(body, Path(artifact_dir)),
        "aggregate": _aggregate_fields(body),
        "was_cancelled": bool(body.get("was_cancelled")),
        "benchmark_duration_s": _scalar(body, "benchmark_duration"),
        "request_count": _scalar(body, "request_count"),
    }
    return out


def _warmup_watch_module():
    """`warmup_watch`, under whichever package name this copy is loaded as.

    BOTH NAMES, like every other import in this file. Outside it is
    `final_validate`; inside Harbor the copy ships as `final_verdict` and no
    `final_validate` package exists at all, so a bare import raises
    `ModuleNotFoundError` exactly where the agent needs this most -- in its own
    measurement. The file is copied BYTE-IDENTICALLY, which is what makes the
    copy the same logic, so the import has to work under both names rather than
    be rewritten on the way in.

    A separate function rather than four lines inside the supervisor, because
    a deferred import that can only be reached by starting a benchmark is a
    deferred import no test will ever exercise.
    """
    try:
        from final_validate import warmup_watch
    except ModuleNotFoundError:              # the in-Harbor copy
        from . import warmup_watch  # type: ignore[no-redef]
    return warmup_watch


def _run_watching_warmup(argv: list[str], environment: dict, log: Path, *,
                         timeout: float):
    """Run aiperf, and do not sit in silence while its warmup hangs.

    `subprocess.run` waits silently while AgentX can keep printing elapsed time
    for a warmup request that never returns.  Read the log as it is written so
    the caller can distinguish progress from a stall.

    So the log is read while it is written. Progress is `returned` going up;
    `elapsed` going up is not progress. A warmup that produces nothing for
    `warmup_watch.SAY_SOMETHING_S` says so, repeatedly, in a sentence naming
    what is stuck -- and one that produces nothing for
    `warmup_watch.STALL_S` is ended, so the run records a cause instead of
    an abandoned directory.

    Returns the finished process and the watch, or the process and None when
    the run never reached a warmup phase at all.
    """
    warmup_watch = _warmup_watch_module()
    watch = warmup_watch.WarmupWatch()
    deadline = time.time() + timeout
    aborted_at: float | None = None
    with log.open("w", encoding="utf-8") as sink:
        process = subprocess.Popen(                        # noqa: S603
            argv, env=environment, stdout=sink,
            stderr=subprocess.STDOUT)
        with log.open("r", encoding="utf-8", errors="replace") as reader:
            try:
                while process.poll() is None:
                    time.sleep(_WARMUP_POLL_S)
                    now = time.time()
                    chunk = reader.read()
                    watch.feed(chunk, now)
                    # aiperf can report its own abort while its controller
                    # remains alive.  The warmup stall watch only covers a
                    # running warmup, so give an already-aborted controller a
                    # short grace period and then record the abort.
                    if aborted_at is None and any(
                            marker in chunk for marker in _AIPERF_ABORTED):
                        aborted_at = now
                    if (aborted_at is not None
                            and now - aborted_at > _AIPERF_EXIT_GRACE_S):
                        print(f"aiperf reported its own abort "
                              f"{now - aborted_at:.0f}s ago and has not "
                              f"exited; ending it so the verdict can record "
                              f"the abort", flush=True)
                        _end(process)
                        break
                    if watch.should_say_something(now):
                        print(watch.describe(now), flush=True)
                    if watch.stalled(now):
                        print(watch.why_it_failed(now), flush=True)
                        _end(process)
                        break
                    if now > deadline:
                        _end(process)
                        raise subprocess.TimeoutExpired(argv, timeout)
            finally:
                if process.poll() is None:
                    _end(process)
        process.wait()
    return process, (watch if watch.saw_progress_line else None)


#: How often the supervisor looks at the log. Short enough that a stall is
#: reported near its threshold, long enough to cost nothing over 1,800 s.
_WARMUP_POLL_S = 15.0

#: aiperf's own words for "this run is over and failed". Read from its log,
#: not inferred: these are the lines it writes immediately before it SHOULD
#: exit non-zero.
_AIPERF_ABORTED = ("Run aborted (", "Will exit non-zero", "ProfileAborted")

#: How long aiperf gets to exit on its own after declaring an abort. Its
#: normal shutdown -- records flushed, services stopped -- takes seconds.
_AIPERF_EXIT_GRACE_S = 120.0


def _end(process: subprocess.Popen) -> None:
    """Stop aiperf, politely and then not. Never leaves it running."""
    with contextlib.suppress(OSError):
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError):
            process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=30)


def measure_one(task_id: str, arm: str, *, url: str, tokenizer: Path,
                run_dir: Path, duration_s: int, verbose: bool = False,
                diagnostic: bool = False, private_workload: bool = False,
                concurrency: int | None = None,
                ) -> dict[str, Any]:
    """Offer the scenario to ONE arm and return what it did."""
    module = _module_of(task_id)
    artifact_dir = Path(run_dir) / "agentx" / arm
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # AgentX 0.12's offline tokenizer loader resolves a cache identity with
    # ``snapshot_download``. A mounted snapshot path passes its early check but
    # then fails independently in dataset and record-processor workers as an
    # invalid Hub repository id. The task's pinned model id + revision is the
    # portable identity; ``tokenizer`` is used only for a module with no
    # ``MODEL_ID``.
    tokenizer_identity = str(getattr(module, "MODEL_ID", "") or tokenizer)
    extra = {} if concurrency is None else {"concurrency": int(concurrency)}
    argv = list(module.invocation(url=url, tokenizer=tokenizer_identity,
                                  artifact_dir=str(artifact_dir),
                                  duration_s=duration_s, **extra))
    if private_workload and not IN_HARBOR:
        # A final seed is odd and every development seed even, so the verdict
        # never replays a request mix the agent measured. The copy inside a
        # workspace never draws from the final space, whatever mode it is run
        # in: only the verdict, outside the container, does.
        seed_index = argv.index("--random-seed") + 1
        argv[seed_index] = str(_shared_agentx_module().final_seed())
    # On the host the checked AgentX is a module in this interpreter. Inside
    # Harbor it is a checked wrapper whose separate site-packages are not part
    # of the candidate's serving environment. Resolve the form that exists.
    argv = _pinned_aiperf_argv(argv)
    # The scenario's validity floor and the task's full measurement window are
    # different numbers. Only a diagnostic below the floor is unsafe.
    shared = _shared_agentx_module()
    scenario_floor = int(getattr(module, "SCENARIO_DURATION_FLOOR_S", 0)
                         or getattr(shared, "SCENARIO_DURATION_FLOOR_S", 900))
    if diagnostic and scenario_floor and duration_s < scenario_floor:
        # A short metrics diagnostic deliberately violates only that lock;
        # aiperf records submission_valid=false, and this path never scores it.
        argv.append("--unsafe-override")

    environment = dict(os.environ)
    environment.update(getattr(module, "ENVIRONMENT", {}) or {})

    started = time.time()
    log = artifact_dir / "aiperf.log"
    completed, watch = _run_watching_warmup(
        argv, environment, log, timeout=max(3600.0, duration_s * 4.0))
    wall = time.time() - started

    row: dict[str, Any] = {
        "arm": arm, "exit_code": completed.returncode,
        "wall_seconds": round(wall, 1), "log": str(log),
        "artifact_dir": str(artifact_dir),
        "command": argv,
        "offered_concurrency": (int(argv[argv.index("--concurrency") + 1])
                                if "--concurrency" in argv else None),
        "workload_selection": {
            "role": "final_validation" if private_workload else "development",
            "seed_space": "odd (final)" if private_workload else "even (development)",
            "seed": (int(argv[argv.index("--random-seed") + 1])
                     if "--random-seed" in argv else None),
        },
    }
    # A warmup that never finished is not a measurement.  Name the stall
    # before consulting the exit code, because this module terminated the
    # process and that code cannot explain the original cause.
    if watch is not None and watch.stalled(time.time()):
        row["warmup_stalled"] = True
        row["not_run_because"] = watch.why_it_failed(time.time())
        row.update(_partial_evidence(
            artifact_dir, allow_diagnostic=diagnostic))
        return row
    # A NON-ZERO EXIT IS NOT THE SAME AS NO MEASUREMENT.
    #
    # aiperf writes its aggregate and THEN tears down, and teardown can fail on
    # its own: a worker that outlives the message-bus grace period is SIGKILLed
    # and the process exits 1 with every export already on disk.
    #
    # So the exit code decides nothing on its own. What decides is whether the
    # aggregate is a RESULT: present, a valid submission, not cancelled, and
    # over the scenario's own duration floor. The exit code is recorded either
    # way, because a teardown that fails is worth seeing.
    if completed.returncode != 0 and not diagnostic:
        salvaged = _aggregate_if_complete(artifact_dir, task_id)
        if salvaged is not None:
            row["exit_code_note"] = (
                f"aiperf exited {completed.returncode} AFTER writing a "
                f"complete aggregate -- teardown failed, the measurement did "
                f"not. Scored from the aggregate; the exit code is recorded "
                f"rather than treated as a missing arm")
            body = salvaged
            row["correctness"] = replay_fidelity_of_run(body, artifact_dir)
            row["aggregate"] = _aggregate_fields(body)
            row["submission_valid"] = submission_valid_of(body) is not False
            row["was_cancelled"] = bool(body.get("was_cancelled"))
            return row
    if completed.returncode != 0:
        # SAY WHAT WENT WRONG, NOT JUST THAT SOMETHING DID.
        #
        # "aiperf exited 1" names a program and a number, and sends whoever
        # reads it off to find a log file. aiperf already writes the reason
        # in two machine-findable places: a boxed `AIPerf System Exit Errors`
        # summary, and the request error that caused it. Both belong in the
        # verdict, because the DIFFERENCE between them is the whole question:
        #
        #   ProfileAborted / warmup_failure          -> the candidate answered
        #     + ClientPayloadError: TransferEncoding    badly; a real finding
        #   connection refused                       -> it never came up
        #   dataset resolve 404                      -> our plumbing, not the
        #                                               candidate's fault
        #
        detail = _why_aiperf_failed(artifact_dir, log)
        # Distinguish a candidate that rejected the workload from missing
        # infrastructure.  A missing performance number is not automatically
        # an infrastructure failure.
        rejected = _candidate_rejected_workload(artifact_dir, log)
        if rejected:
            row["rejected_workload"] = rejected
        row["not_run_because"] = (
            f"aiperf exited {completed.returncode}. An arm that could not be "
            f"offered the workload is NOT_RUN with this reason, never a zero: "
            f"a missing arm makes a task look easier than it is."
            + (f"\n{detail}" if detail else "")
            + f"\nFull log: {log}")
        # Preserve any partial correctness and metric evidence separately for
        # diagnosis. It never turns an incomplete arm into a scored result.
        row.update(_partial_evidence(
            artifact_dir, allow_diagnostic=diagnostic))
        return row

    # Read even an invalid submission, so the arm says what it did; `score()`
    # then refuses to score it unless it is the short development window
    # this module itself ran with `--unsafe-override`.
    body = read_aggregate(artifact_dir, allow_diagnostic=True)
    row["correctness"] = replay_fidelity_of_run(body, artifact_dir)
    row["aggregate"] = _aggregate_fields(body)
    valid = submission_valid_of(body)
    row["submission_valid"] = valid is not False
    if valid is False:
        row["submission_invalid_reasons"] = submission_invalid_reasons_of(body)
    row["was_cancelled"] = bool(body.get("was_cancelled"))
    # WHEN the output arrived, not only how much. See `stream_timing`: a
    # server that flushes a finished response reports a TPOT that is the speed
    # of a socket write, and it reads as a pass.
    row["stream_timing"] = stream_timing.assess(artifact_dir)
    return row


def objective_detail(task_id: str) -> dict:
    """The task's objective, including its model-specific token prices."""
    try:
        from final_validate import task_spec
    except ModuleNotFoundError:          # the in-Harbor copy
        from . import task_spec  # type: ignore[no-redef]

    def _prices_of(body: dict[str, Any]) -> task_spec.TokenPrices:
        got = body.get("token_prices") or {}
        return task_spec.TokenPrices(
            model=got.get("model", ""),
            usd_per_million_input=float(got.get("usd_per_million_input", 0)),
            usd_per_million_output=float(got.get("usd_per_million_output", 0)),
            usd_per_million_cached_input=float(
                got.get("usd_per_million_cached_input", 0)),
            source=got.get("source", ""),
            cache_rate_is=got.get("cache_rate_is", ""))

    try:
        con = task_spec.contract(spine.resolve_task_dir(task_id))
    except (OSError, ValueError, KeyError):
        return {}
    devices, rate = int(con["devices"]), float(con["usd_per_gpu_hour"])
    return {
        "objective": con.get("objective") or objective.OBJECTIVE,
        "request_scoring": dict(con.get("request_scoring") or {}),
        "devices": devices,
        "devices_charged": devices,
        "usd_per_gpu_hour": rate,
        # Derived from the two numbers above rather than stored beside them,
        # so a total can never disagree with its own per-card rate.
        "usd_per_hour_total": rate * devices,
        # BOTH UNITS, DERIVED FROM ONE. `contract.yaml` states a rate once,
        # per million tokens, because that is how an operator reads it.
        # `token_prices.table()` bills in dollars per TOKEN and falls back to
        # nothing else, so both forms are emitted from the one declaration.
        "token_prices": {**dict(con.get("token_prices") or {}),
                         **_prices_of(con).per_token()},
    }


def score(task_id: str, row: dict[str, Any], usd_per_hour: float,
          checks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The task's declared objective, from the shared definition in `objective.py`.

    The objective and the gates come from the task's own `contract.yaml`; nothing
    here restates a threshold.
    """
    aggregate = row.get("aggregate") or {}
    duration = (aggregate.get("benchmark_duration_s")
                or row.get("wall_seconds"))
    # Use this task's declared prices rather than the module fallback.
    detail = objective_detail(task_id)
    scored = objective.score(aggregate, usd_per_hour=usd_per_hour,
                             duration_s=duration, prices=detail)
    if detail.get("request_scoring"):
        unavailable = ""
        if (row.get("evidence_is_partial") or row.get("was_cancelled")
                or row.get("not_run_because")):
            unavailable = "incomplete or invalid timed replay"
        elif not submission_acceptable(row):
            reasons = ", ".join(row.get("submission_invalid_reasons") or [])
            unavailable = ("AgentX stamped submission_valid=false"
                           + (f" ({reasons})" if reasons else "")
                           + "; an invalid submission is not a result")
        if (row.get("stream_timing") or {}).get("buffered"):
            unavailable = "buffered stream: TPOT is not measurable"
        scored.update(objective.request_scored(
            scored.get(objective.GROSS), checks or [],
            detail.get("request_scoring"), artifact_dir=row.get("artifact_dir"),
            expected_requests=aggregate.get("request_count"),
            expected_errors=aggregate.get("error_request_count"),
            expected_input_tokens=aggregate.get("total_input_tokens"),
            expected_output_tokens=aggregate.get("total_output_tokens"),
            prices=detail, usd_per_hour=usd_per_hour, duration_s=duration,
            unavailable_reason=unavailable))
    return scored


def eligibility_gates(correctness: dict[str, Any] | None
                      ) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Return replay and answer-correctness gates as independent constraints.

    A final run carries ``combined_correctness`` with both children. Read the
    children rather than trusting its cached conjunction, so neither a stale
    nor malformed top-level ``holds`` can let one failed gate through. Public
    performance-only evidence carries replay fidelity alone; treating that
    single record as the combined gate preserves development reporting
    without weakening the real final path.
    """
    combined = correctness or {}
    replay = combined.get("replay_fidelity")
    answer = combined.get("output_correctness")
    if isinstance(replay, dict) or isinstance(answer, dict):
        replay = replay if isinstance(replay, dict) else {}
        answer = answer if isinstance(answer, dict) else {}
        answer_evaluable = bool(answer.get(
            "evaluable", answer.get("available", False)))
        holds = (replay.get("holds") is True
                 and answer_evaluable and answer.get("holds") is True)
        return replay, answer, holds
    # Development shape: the one record is the gate. Final
    # validation never emits this shape after ``combined_correctness`` runs.
    return combined, combined, combined.get("holds") is True


#: Where a non-holding arm's measured objective is kept: a diagnostic of what
#: the run would have scored had every hard gate held. It ranks nothing.
IF_ELIGIBLE = "_if_eligible"


def withhold_objective(scored: dict[str, Any]) -> dict[str, Any]:
    """An arm that failed a hard gate scores nothing.

    The objective (and `objective_value`, where present) becomes None; the
    measured number moves to `<name>_if_eligible`, which no ranking reads.
    The gross figure stays: it was never a score.
    """
    out = dict(scored)
    for key in (objective.OBJECTIVE, "objective_value"):
        if key in out:
            out[key + IF_ELIGIBLE] = out[key]
            out[key] = None
    return out


def decide(contract: spine.Contract, rows: list[dict[str, Any]],
           usd_per_hour: float) -> dict[str, Any]:
    """The verdict: both service levels, on every arm, against the frozen SLO."""
    slo = contract.slo or {}
    try:
        levels = service_levels.read(slo, task_id=contract.task_id)
    except service_levels.ContractError as exc:
        raise SystemExit(
            f"{exc}. A verdict with no bar is not a verdict") from exc

    decisions = []
    for row in rows:
        arm = row["arm"]
        aggregate = row.get("aggregate") or {}
        if not aggregate and row.get("rejected_workload"):
            # EVALUATED, AND FAILED. The arm was offered the workload and its
            # server dropped it, which is exactly the offered-work invariant
            # replay fidelity exists to enforce. A MISS with no score -- never
            # NOT_RUN, which says "we could not evaluate" about a candidate we
            # evaluated perfectly well.
            fidelity = {"holds": False, "evaluable": True,
                        "rule": "replay_fidelity",
                        "why": row["rejected_workload"]}
            decisions.append({
                "arm": arm, "outcome": "MISSES",
                "correctness": fidelity,
                "replay_gate": fidelity,
                "correctness_gate": {"holds": False, "evaluable": False,
                                     "why": "workload rejection prevented "
                                            "answer correctness evaluation"},
                "eligibility_gates_hold": False,
                "service_levels": [],
                "why_misses": row["rejected_workload"],
                objective.OBJECTIVE: None, objective.GROSS: None,
                "tokens_per_usd": None})
            continue
        if not aggregate:
            decisions.append({"arm": arm, "outcome": "NOT_RUN",
                              "why": row.get("not_run_because", "")})
            continue
        # CORRECTNESS FIRST, and it is decisive.
        #
        # A system that did not do the offered work has not earned its
        # throughput, so its objective is reported as context and never as a
        # score. Checked before the service levels because an arm that emitted
        # fewer tokens than asked will also look FASTER on every one of them --
        # reading those numbers first is how a fidelity failure gets recorded
        # as a win.
        fidelity = row.get("correctness") or {}
        replay_gate, correctness_gate, eligibility_holds = eligibility_gates(
            fidelity)
        timing = row.get("stream_timing") or {}
        buffered = bool(timing.get("buffered"))
        # A BUFFERED STREAM'S TPOT IS NOT MEASURED. It measured a socket write,
        # and an unmeasured gate never holds -- `service_levels.check`'s own
        # rule, so this can fail a TPOT gate the wire made look like a pass and
        # can never pass one. TTFT is left alone: the client really did wait
        # that long for its first token.
        checks = service_levels.check(
            levels,
            ttft_by_percentile={int(level["percentile"]):
                                aggregate.get(f"ttft_p{int(level['percentile'])}_ms")
                                for level in levels},
            tpot_by_percentile={int(level["percentile"]):
                                (None if buffered else
                                 aggregate.get(f"tpot_p{int(level['percentile'])}_ms"))
                                for level in levels})
        scored = score(contract.task_id, row, usd_per_hour, checks)
        # Latency is priced per request into the objective, not gated: a run
        # holds when replay fidelity and correctness hold and the objective is
        # computable, even when some requests score zero.
        score_available = scored.get(objective.OBJECTIVE) is not None
        held = eligibility_holds and score_available
        if not held:
            scored = withhold_objective(scored)
        decisions.append({
            "arm": arm,
            "outcome": "HOLDS" if held else "MISSES",
            "correctness": fidelity,
            "replay_gate": replay_gate,
            "correctness_gate": correctness_gate,
            "eligibility_gates_hold": eligibility_holds,
            "service_levels": checks,
            "stream_timing": timing,
            "why_misses": ("" if held else "; ".join(filter(None, [
                replay_gate.get("why", ""),
                correctness_gate.get("why", ""),
                timing.get("why", "") if buffered else "",
                (scored.get("request_scoring_why")
                 or f"{objective.OBJECTIVE} not computable"
                 if not score_available else ""),
            ]))),
            **scored,
        })

    objective_name = objective.OBJECTIVE
    scored = [d for d in decisions
              if d["outcome"] == "HOLDS"
              and d.get(objective_name) is not None]
    best = max(scored, key=lambda d: d[objective_name]) if scored else None
    return {
        "schema": SCHEMA,
        "task_id": contract.task_id,
        "objective": objective_name,
        "usd_per_hour": usd_per_hour,
        "service_levels": levels,
        "both_must_hold": bool(slo.get("both_must_hold", True)),
        "arms": decisions,
        "best_holding_arm": best["arm"] if best else None,
        "best_objective_value": best[objective_name] if best else None,
        "best_gross_dollars_per_dollar": (
            best.get(objective.GROSS) if best else None),
        # A reported diagnostic, not the selector when the task declares a
        # different objective.
        "best_tokens_per_usd": best["tokens_per_usd"] if best else None,
        "note": (
            f"{objective_name} is the task's only objective: the token value of "
            "completed profiling requests meeting BOTH p90 bars divided by "
            "whole-window GPU cost. Replay fidelity and output correctness are "
            "hard eligibility gates. Aggregate percentiles and "
            f"{objective.GROSS} are diagnostics and decide nothing."),
    }


def _configured_window(command: list[str]) -> int | None:
    """The `--benchmark-duration` this AgentX invocation asked for."""
    for index, token in enumerate(command[:-1]):
        if token == "--benchmark-duration":
            try:
                return int(float(command[index + 1]))
            except ValueError:
                return None
    return None


def development_report(contract: spine.Contract, row: dict[str, Any], *,
                       usd_per_hour: float, mode: str,
                       candidate_tree_sha: str = "",
                       candidate_version: str = "",
                       source_snapshot: str = "") -> dict[str, Any]:
    """Machine-readable *public* evidence, never an independent final verdict.

    Exit zero in measure-only means replay fidelity, not latency acceptance.
    Keep the three decisions separate, and never turn a partial export into a
    qualification. Call the same gate/objective readers as the final verdict.
    """
    report: dict[str, Any] = {
        "schema": "final_validate.development_evidence/1",
        "task_id": contract.task_id, "mode": mode, "row": row,
        "final_verdict": False, "public_correctness_holds": None,
        "answer_correctness_holds": None,
        "correctness_scope": "protocol_context_only" if mode == "correctness"
                             else "replay_fidelity_only",
        "replay_fidelity_holds": None, "hard_gates_hold": None,
        "measured_gates_hold": None,
        "full_window_complete": False, "objective": contract.objective,
        "objective_value": None,
        # Public evidence is useful only for the immutable source it actually
        # measured.  The trial helper supplies these after freezing and
        # hashing the tree.  Direct endpoint calls remain usable diagnostics,
        # but say UNBOUND rather than silently attaching themselves to the
        # next checkpoint the agent happens to publish.
        "candidate_tree_sha": candidate_tree_sha or None,
        "version_id": candidate_version or None,
        "source_snapshot": source_snapshot or None,
        "measurement_status": (
            "COMPLETE" if candidate_tree_sha and source_snapshot else "UNBOUND"),
    }
    if mode == "correctness":
        report["public_correctness_holds"] = (
            (row.get("correctness") or {}).get("holds") is True)
        return report
    aggregate = row.get("aggregate") or {}
    elapsed = aggregate.get("benchmark_duration_s")
    command = row.get("command") or []
    window = _configured_window(command)
    # COMPLETE = the window that was asked for ran to its end. NOT "the last
    # response arrived at or after 1800 s": with trace delays and long
    # requests the last COMPLETED response often lands before the cut, and
    # that rule would report every slow but valid run as INCOMPLETE with no
    # objective. The verdict judges completeness the same way.
    # A development window below the scenario's 900 s floor is run with
    # `--unsafe-override`, which aiperf stamps `submission_valid=false`; our
    # command fixes every other flag, so that is the only override there is
    # (`override_is_a_short_development_window`, which `score()` reads too).
    ours = override_is_a_short_development_window(row)
    valid = row.get("submission_valid") is True or ours
    complete = (window is not None
                and isinstance(elapsed, (int, float)) and elapsed > 0
                and ("--unsafe-override" not in command or ours)
                and valid
                and not row.get("was_cancelled")
                and not row.get("not_run_because")
                and not row.get("warmup_stalled"))
    decision = decide(contract, [row], usd_per_hour)["arms"][0]
    latency_holds = service_levels.holds(decision.get("service_levels") or [])
    replay_holds = (row.get("correctness") or {}).get("holds") is True
    measured_holds = replay_holds
    report.update(
        window_seconds=window,
        window_is_verdict_length=window == 1800,
        full_window_complete=bool(complete),
        replay_fidelity_holds=replay_holds,
        # Output correctness is verified privately after the run, never by
        # this mode, so the complete hard-gate conjunction is UNKNOWN here.
        hard_gates_hold=None,
        measured_gates_hold=measured_holds,
        latency_gates_hold=latency_holds,
        request_scoring_policy=decision.get("request_scoring_policy"),
        request_multiplier=decision.get("request_multiplier"),
        request_attainment=decision.get("request_attainment"),
        billable_input_tokens=decision.get("billable_input_tokens"),
        billable_output_tokens=decision.get("billable_output_tokens"),
        billable_value_usd=decision.get("billable_value_usd"),
        billing_gpu_cost_usd=decision.get("billing_gpu_cost_usd"),
        request_scoring_why=decision.get("request_scoring_why"),
        service_levels=decision.get("service_levels") or [],
        stream_timing=row.get("stream_timing") or {},
        objective=objective.OBJECTIVE,
        objective_value=decision.get(objective.OBJECTIVE) if complete else None,
        # Measured, but withheld because a gate this mode reads failed. It
        # says what the window would have scored; it is never a score.
        objective_value_if_eligible=(
            decision.get(objective.OBJECTIVE + IF_ELIGIBLE)
            if complete else None),
        gross_dollars_per_dollar=(decision.get(objective.GROSS)
                                  if complete else None),
    )
    if not complete:
        report["measurement_status"] = "INCOMPLETE"
        report["incomplete_why"] = "; ".join(why for bad, why in (
            (window is None, "no --benchmark-duration in the recorded command"),
            (not isinstance(elapsed, (int, float)) or not elapsed,
             "the benchmark reported no duration"),
            (not valid, "aiperf stamped submission_valid=false"),
            (bool(row.get("was_cancelled")), "the run was cancelled"),
            (bool(row.get("not_run_because")),
             f"not run: {row.get('not_run_because')}"),
            (bool(row.get("warmup_stalled")), "the warmup stalled"),
        ) if bad) or "unknown"
    return report


def _print_request_scores(report: dict[str, Any]) -> None:
    """The same request-count explanation in development and final output."""
    evidence = report.get("request_attainment") or {}
    if not evidence.get("evaluable") or report.get("request_scoring_why"):
        print(f"  request scores UNAVAILABLE: "
              f"{report.get('request_scoring_why') or 'missing evidence'}")
        return
    counts = evidence["request_score_counts"]
    total = evidence["request_count"]
    print(f"  request scores     both p90 bars met: {counts['p90_both']}/{total} x 1; "
          f"otherwise: {counts['otherwise']}/{total} x 0")
    print(f"  request attainment {report['request_multiplier']:.6f} (diagnostic only)")
    print(f"  billable tokens    input {report.get('billable_input_tokens')}; "
          f"output {report.get('billable_output_tokens')}; "
          f"revenue ${report.get('billable_value_usd')} / "
          f"GPU cost ${report.get('billing_gpu_cost_usd')}")
    value = report.get(objective.OBJECTIVE, report.get("objective_value"))
    print(f"  {objective.OBJECTIVE:<22}{value}   <- the objective")
    if value is None and report.get("incomplete_why"):
        print(f"  NOT MEASURED: {report['incomplete_why']}")
    withheld = report.get(objective.OBJECTIVE + IF_ELIGIBLE,
                          report.get("objective_value" + IF_ELIGIBLE))
    if value is None and withheld is not None:
        print(f"  WITHHELD: a hard gate failed, so this run scores nothing "
              f"(it measured {withheld}; that number decides nothing)")
    gross = report.get(objective.GROSS)
    if gross is not None:
        print(f"  {objective.GROSS:<22}{gross}   (every request counted as if "
              "it met the bars; decides nothing)")


def _write_development_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n", "utf-8")
    temporary.replace(path)


def _capture_live_logs() -> None:
    """`deploy.capture_live_logs`, where `deploy` exists at all."""
    try:
        from final_validate import deploy
    except ModuleNotFoundError:          # the in-Harbor copy has no `deploy`
        return
    with contextlib.suppress(Exception):
        deploy.capture_live_logs()


def _teardown() -> None:
    """Hand back whatever `_launch` took, whatever happened.

    A verdict that dies holding a card is worse than a verdict that fails: the
    next arm, and every other job on the host, then waits on a container
    nobody is watching.
    """
    try:
        from final_validate import deploy
    except ModuleNotFoundError:          # the in-Harbor copy has no `deploy`
        return
    with contextlib.suppress(Exception):
        deploy._release_all()


#: The one placeholder in a frozen launch line, substituted BY NAME: the lines
#: carry JSON (`--json-model-override-args '{...}'`), which `str.format` would
#: read as replacement fields.
CONCURRENCY_SLOT = "{concurrency}"

#: The server admission limit a baseline arm is launched with. Above any
#: offered client load (`concurrency.json` allows at most 256), so the gates,
#: not the server's queue, are what an arm is judged by.
ADMISSION_LIMIT = 512


def fill_concurrency(template: str, concurrency: int) -> str:
    return (template or "").replace(CONCURRENCY_SLOT, str(int(concurrency)))


def baseline_serve_extra(task_dir: Path, target: str) -> str:
    """A baseline arm's launch flags, as the task's `reference.yaml` freezes them.

    Frozen with the task because some of them are not tuning (context length,
    RoPE overrides): an arm launched without them cannot be offered the work.
    Only the admission limit is filled in. Empty for the candidate, whose
    launch line is its own.
    """
    if target == "candidate":
        return ""
    try:
        from final_validate import task_spec
    except ModuleNotFoundError:          # the in-Harbor copy
        from . import task_spec  # type: ignore[no-redef]
    return fill_concurrency(task_spec.serve_extra(task_dir, target),
                            ADMISSION_LIMIT)


def _serve_extra(contract: spine.Contract, target: str) -> str:
    return baseline_serve_extra(contract.task_dir, target)


def _launch(contract: spine.Contract, args) -> tuple[str, int]:
    """Stand one arm up on a leased card and return where it is listening.

    Everything that can refuse happens before a card is taken: holding a
    shared device through a failing `docker image inspect` blocks a neighbour.
    """
    import uuid

    # LAUNCHING IS NOT AVAILABLE IN THE PACKAGED COPY, and it says so.
    #
    # `deploy`, `lease` and `store` stand a container up on a
    # leased card. None is shipped to an agent: a container cannot lease a
    # host GPU or start a sibling container, and shipping the code would only
    # let the failure happen later and less clearly.
    #
    # Without this the copy would raise `ModuleNotFoundError: No module named
    # 'final_validate'` -- a message about packaging, for an operation that
    # cannot work here. `--endpoint` is the form that works in here and is
    # what the brief tells the agent to use.
    try:
        from final_validate.deploy import (
            DockerControlPlaneError,
            ServerDidNotStart,
            inspect_image,
            materialise,
        )
        from final_validate.deploy import launch as go
        from final_validate.lease import acquire
    except ModuleNotFoundError:
        raise SystemExit(
            "--launch is not available in this copy of the verdict. Standing a "
            "server up needs a GPU lease and a sibling container, which is an "
            "operation for the harness outside, not for anything running in "
            "here.\n\n"
            "Start your own server and point this at it instead:\n"
            "    --target candidate --endpoint 127.0.0.1:8000\n\n"
            "That measures exactly what the verdict outside measures, against "
            "the server you are actually building.") from None

    wanted = args.image or (spine.image_for(args.target)
                            if args.target in spine.ARMS else "")
    if not wanted:
        raise SystemExit(
            f"no image for target {args.target!r}. A candidate is measured in "
            f"the task's own agent image; pass --image")
    if not inspect_image(wanted):
        raise SystemExit(
            f"the image {wanted!r} is not on this host, so {args.target} "
            f"cannot be measured. Nothing was leased and no card was touched. "
            f"This is NOT a result about {args.target}: an arm that did not "
            f"run is not an arm that lost")

    candidate = None
    if args.target == "candidate":
        if args.candidate_snapshot is None:
            raise SystemExit("--target candidate needs --candidate-snapshot")
        candidate = materialise(args.candidate_snapshot,
                                args.run_dir / "candidate")

    devices = [int(d) for d in str(args.device).split(",") if d.strip()]
    if devices:
        if len(devices) != contract.devices:
            raise SystemExit(
                f"--device names {devices} but {contract.task_id} declares "
                f"{contract.devices} device(s). Measuring a topology task on "
                f"the wrong device count answers a different question")
        # `ExternalLease`, NOT a hand-built string.
        #
        # `--gpus` parses its value as CSV, so the spec has to be the quoted
        # `"device=4,8"` form; a bare `4,8` sets docker's *Count* instead of
        # naming devices. That is not a visible error -- the container starts,
        # with the wrong number of GPUs.
        #
        # `ExternalLease` also re-derives foreign compute processes at the
        # moment of use and releases nothing, which is what a card claimed by
        # the driver needs.
        from final_validate.lease import ExternalLease

        lease = ExternalLease(devices)
        device_spec = lease.device_spec  # the quoted "device=4,8" form
    else:
        deny = tuple(int(d) for d in str(args.deny).split(",") if d.strip())
        lease = acquire(contract.devices, timeout_s=1800, deny=deny)
        device_spec = lease.device_spec

    try:
        try:
            from final_validate import model_identity
        except ModuleNotFoundError:          # the in-Harbor copy
            from . import model_identity  # type: ignore[no-redef]
        declared_id, _ = model_identity.declared_for(contract.task_dir)
        # A caller may supply the whole launch line and extra environment
        # for an arm (the host-only stages do); otherwise the task's own.
        serve_extra = (getattr(args, "serve_extra_override", None)
                       or _serve_extra(contract, args.target))
        server_env = {"MODEL_ID": declared_id} if declared_id else {}
        server_env.update(getattr(args, "server_env", None) or {})
        server = go(args.target, image=wanted,
                    model_dir=args.model_dir or Path("/nonexistent"),
                    device_spec=device_spec, port=args.port or 0,
                    name=f"finalval-{contract.task_id.lower()}-"
                         f"{uuid.uuid4().hex[:8]}",
                    log=args.run_dir / f"{args.target}.serverlog.txt",
                    candidate=candidate,
                    # The document promises `MODEL_ID` in the environment, so
                    # the verdict exports it too. See `deploy.launch`.
                    env=server_env or None,
                    serve_extra=serve_extra,
                    tp=contract.devices if contract.devices > 1 else 1)
    except ServerDidNotStart as exc:
        raise SystemExit(f"{args.target} did not start: {exc}") from None
    except DockerControlPlaneError as exc:
        raise SystemExit(
            f"INFRASTRUCTURE_FAILURE: Docker control plane: {exc}") from None
    return server.host, server.port


#: Lines aiperf writes when it gives up, in the order a reader wants them.
#: The boxed summary says WHAT aborted; the request error says WHY.
_AIPERF_ABORT_MARKERS = ("AIPerf System Exit Errors", "Run aborted",
                         "ProfileAborted")
_AIPERF_CAUSE_MARKERS = ("Error in aiohttp request:", "ERROR ",
                         "Terminal warmup failure", "Connection refused")


#: aiperf's words for "the SERVER cut this request off or refused it". Each is
#: a response the server under test produced, or a connection it closed after
#: accepting a request -- not a failure to reach it.
_SERVER_SEVERED = ("ServerDisconnectedError", "ClientPayloadError",
                   "TransferEncodingError", "Response payload is not completed",
                   "status=500", "status=502", "status=503", "status=504",
                   "500 Internal Server Error")

#: aiperf's words for "the run ended because a WORKLOAD request failed".
_WORKLOAD_ABORT = ("Run aborted (warmup_failure)", "Terminal warmup failure")


def _candidate_rejected_workload(artifact_dir: Path, log: Path) -> str:
    """aiperf's evidence that the SERVER failed an offered request, or "".

    Both halves are required, and both come from aiperf's own log:

    * the run was aborted BECAUSE a workload request failed, and
    * the failure was the server severing or rejecting that request.

    A refused connection is deliberately NOT here: a server that never came up
    is DID_NOT_START, and one we could not reach may be our network. Neither
    is anything but NOT_RUN without more evidence than this.

    This only ever runs after the arm passed readiness -- aiperf is not started
    against a server that did not answer /health -- so "the server was up and
    then dropped the work" is what the two markers together establish.
    """
    for path in (Path(artifact_dir) / "logs" / "aiperf.log",
                 Path(artifact_dir) / "aiperf.log", Path(log)):
        try:
            text = path.read_text("utf-8", errors="ignore")
        except OSError:
            continue
        if not any(marker in text for marker in _WORKLOAD_ABORT):
            continue
        severed = [line.strip() for line in text.splitlines()
                   if any(marker in line for marker in _SERVER_SEVERED)]
        if severed:
            return ("the server was ready and then severed or rejected a "
                    "request from the offered workload during warmup, so "
                    "aiperf aborted the run: " + severed[0][-240:])
    return ""


def _why_aiperf_failed(artifact_dir: Path, log: Path,
                       limit: int = 8) -> str:
    """The reason out of aiperf's own log, or "" if it does not say.

    Reads the structured log rather than stdout: the boxed exit summary is
    drawn with box characters and wrapped, so the useful text is easier to take
    from the log lines that produced it.
    """
    candidates = [Path(artifact_dir) / "logs" / "aiperf.log",
                  Path(artifact_dir) / "aiperf.log", Path(log)]
    for path in candidates:
        try:
            if not path.is_file():
                continue
            lines = path.read_text("utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        picked: list[str] = []
        for line in lines:
            if any(marker in line for marker in
                   (*_AIPERF_ABORT_MARKERS, *_AIPERF_CAUSE_MARKERS)):
                text = line.strip()
                if text and text not in picked:
                    picked.append(text)
        if picked:
            # The last few are the ones that ended the run.
            chosen = picked[-limit:]
            return ("aiperf's own reason:\n  "
                    + "\n  ".join(item[:300] for item in chosen))
    return ""


def _tasks_root(explicit: Path | None) -> Path:
    """Where to look for the task contract, in the order that actually works.

    The verdict runs in two places. Outside, `serving_tasks/` sits beside this
    package and `spine.TASKS_ROOT` is right. Inside an agent container there is
    no repository at all, and the task folder is delivered to
    `task_context/<TASK_ID>/` -- see `lifecycle/workspace._deliver_task_context`.

    Tried in order so an explicit flag always wins and the offline case needs
    no flag at all.
    """
    import os

    if explicit:
        return Path(explicit)
    named = os.environ.get("SERVERBENCH_TASK_CONTEXT", "").strip()
    if named and Path(named).is_dir():
        return Path(named)
    for guess in (Path.cwd() / "task_context",
                  Path("/workspace/task_context")):
        if spine.task_dirs(guess):
            return guess
    return Path(spine.TASKS_ROOT)


def _require_workspace_agentx(task_id: str, run_dir: Path) -> None:
    """Keep in-workspace evidence under the workspace evidence root, where
    `./measurements` and the run record can read it."""
    del task_id
    if not IN_HARBOR or not Path("/workspace/AGENT_TASK.md").is_file():
        return
    root = Path("/workspace/agentx").resolve()
    destination = Path(run_dir).resolve()
    if destination != root and root not in destination.parents:
        raise SystemExit(
            f"measurements must use a fresh --run-dir below "
            f"{root}; got {destination}")


def resolve_concurrency(module, args) -> int | None:
    """The offered load for this run, or None for the module's fixed value.

    A task whose module sets `CONCURRENCY_POLICY = "candidate"` offers the
    load the CANDIDATE declares in `concurrency.json` at the root of its tree:
    the published snapshot in the verdict, the workspace in development.
    `--concurrency N` lets a development measurement try another load; the
    verdict never takes it.
    """
    requested = getattr(args, "concurrency", None)
    if getattr(module, "CONCURRENCY_POLICY", "fixed") != "candidate":
        fixed = getattr(module, "OFFERED_CONCURRENCY", None)
        if requested is not None and requested != fixed:
            raise SystemExit(
                f"this task offers a fixed concurrency ({fixed}); "
                f"--concurrency {requested} is refused")
        return None
    shared = _shared_agentx_module()
    final = bool(getattr(args, "launch", False)) and not (
        getattr(args, "measure_only", False)
        or getattr(args, "development_workload", False)
        or getattr(args, "check_protocol_context", False))
    if args.target == "candidate":
        if final and args.candidate_snapshot is not None:
            if requested is not None:
                raise SystemExit("the verdict offers the load the published "
                                 "tree declares; --concurrency is refused")
            tree = args.candidate_snapshot
        elif requested is not None:
            return int(requested)
        elif getattr(args, "candidate_snapshot", None) is not None:
            tree = args.candidate_snapshot
        else:
            tree = Path(os.environ.get("APP_DIR") or Path.cwd())
            if not (tree / shared.CANDIDATE_CONCURRENCY_FILE).is_file() \
                    and Path("/workspace").is_dir():
                tree = Path("/workspace")
        try:
            return shared.candidate_concurrency(tree)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    # A baseline arm is measured at the task's stated SGLang operating point.
    offered = requested or getattr(module, "REFERENCE_CONCURRENCY", None)
    if offered is None:
        raise SystemExit(
            f"{args.target}: offered concurrency is undecided -- this task "
            "states no SGLang reference point; pass --concurrency N")
    return int(offered)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--target", default="all",
                    help="an arm name, or 'all'")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--endpoint", default="",
                    help="HOST:PORT of an already-running server, for one arm")
    ap.add_argument("--run-dir", type=Path, required=True)
    # CORRECTNESS WITHOUT THE TIMED REPLAY, FOR USE INSIDE THE LOOP.
    #
    # The full replay is appropriate for a result but expensive for a quick
    # development check.
    # `--check-protocol-context` does not load the scenario corpus. Loading
    # the scenario expands 393 traces into thousands of conversations and builds
    # a multi-gigabyte mmap before sending one request; paying that six-minute
    # setup cost to ask a few independent probes is wasteful. A task declaring
    # extended context also gets one near-maximum prompt here; that is slower
    # and is meant for architecture milestones, because a cheap short probe
    # cannot expose silent middle truncation.
    # Replay fidelity remains part of every real scenario measurement and of
    # final validation. This mode answers only the independent inner-loop
    # endpoint question: does the service preserve streamed text and the
    # task's required input context? Output correctness is verified privately
    # after the run.
    development_mode = ap.add_mutually_exclusive_group()
    development_mode.add_argument(
        "--check-protocol-context", "--check-correctness",
        dest="check_protocol_context", action="store_true",
        help=("protocol/native-token/context integrity only; no timed replay, "
              "answer fidelity, or score. --check-correctness is an "
              "alias"))
    # THE OTHER HALF, AND THE REASON THE FIRST ONE IS NOT ENOUGH.
    # `--measure-only` runs the complete performance window and reports its
    # metrics without combining them with answer correctness or a score.
    development_mode.add_argument(
        "--measure-only", action="store_true",
        help="AgentX performance replay (--duration, 300-1800 s, default "
             "1800), metrics and dollars_per_dollar only; no correctness "
             "verdict")
    ap.add_argument("--duration", type=int, default=0,
                    help="performance seconds: 300-1800 for a development "
                         "--measure-only window (default 1800); the final "
                         "verdict is always 1800")
    # The validator can either attach to an endpoint or launch the selected arm.
    ap.add_argument("--launch", action="store_true",
                    help="stand the arm up here rather than taking --endpoint")
    ap.add_argument("--development-workload", action="store_true",
                    help="run the complete verifier on public development "
                         "traffic; never an authoritative final result")
    ap.add_argument("--image", default="",
                    help="the image to serve in; required for --target candidate")
    ap.add_argument("--candidate-snapshot", type=Path, default=None,
                    help="the published version directory to serve")
    ap.add_argument("--device", default="",
                    help="comma-separated device indices to use instead of a lease")
    ap.add_argument("--deny", default="",
                    help="comma-separated device indices a lease may not take")
    ap.add_argument("--port", type=int, default=0)
    # WHERE THE TASK CONTRACT IS, for the copy that runs inside Harbor.
    #
    # Outside, `serving_tasks/` is beside this package and the default finds
    # it. Inside an agent container there is no repository, and the contract
    # is delivered to `task_context/<TASK_ID>/`. Without this the copy would
    # raise `KeyError: ... names no task under .../tests/serving_tasks.
    # Known: []` and could not be run.
    ap.add_argument("--task-context", type=Path, default=None,
                    help="directory holding <TASK_ID>/task.yaml; defaults to "
                         "$SERVERBENCH_TASK_CONTEXT, then ./task_context, then "
                         "the repository's serving_tasks/")
    ap.add_argument("--candidate-tree-sha", default="",
                    help="immutable source digest supplied by a freezing trial helper")
    ap.add_argument("--candidate-version", default="",
                    help="version id bound to --candidate-tree-sha, when materialised")
    ap.add_argument("--source-snapshot", default="",
                    help="immutable candidate directory that the endpoint serves")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="development only, on a task whose candidate "
                         "declares its own offered load: measure at N instead "
                         "of the value in concurrency.json")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    tasks_root = _tasks_root(args.task_context)
    task_id = spine.resolve_task_dir(args.task, tasks_root=tasks_root).name
    _require_workspace_agentx(task_id, args.run_dir)
    contract = spine.load_contract(task_id, tasks_root=tasks_root)
    module = _module_of(task_id)
    verdict_duration = int(getattr(module, "VERDICT_DURATION_S", 0) or 0)
    evolve_duration = int(getattr(module, "EVOLVE_DURATION_S", 0) or 0)
    if not args.check_protocol_context:
        # The VERDICT has exactly one window. A development measurement
        # (`--measure-only`) may choose a shorter one; it is reported as such
        # and never stands in for the
        # verdict, which always runs the full window.
        if (evolve_duration, verdict_duration) != (1800, 1800):
            raise SystemExit(
                f"{task_id} declares evolve/verdict performance windows "
                f"{evolve_duration}/{verdict_duration} s; both must be 1800 s")
        floor = int(getattr(_shared_agentx_module(),
                            "DEVELOPMENT_MIN_DURATION_S", 300))
        development_window = bool(args.measure_only)
        if development_window:
            duration = int(args.duration or 1800)
            if not floor <= duration <= 1800:
                raise SystemExit(
                    f"--duration {duration}: a development window is "
                    f"{floor}-1800 s (the final verdict always runs 1800 s)")
        elif args.duration and args.duration != 1800:
            raise SystemExit(
                f"--duration {args.duration}: the final verdict runs the "
                "complete 1800 s replay; a shorter window is a development "
                "measurement (--measure-only)")
        else:
            duration = 1800
    else:
        duration = args.duration or verdict_duration or 1800

    # The rate is the TASK's, never this program's: an objective whose
    # denominator could be set by the thing doing the measuring is not an
    # objective.
    # THE COST DENOMINATOR, from `contract.yaml`: the per-card rate times the
    # card count, both frozen with the task. No third field restates their
    # product, so a total can never disagree with its own rate.
    detail = objective_detail(task_id)
    usd_per_hour = float(detail.get("usd_per_hour_total") or 0.0)
    if not usd_per_hour:
        raise SystemExit(
            f"{task_id}'s contract.yaml declares no `usd_per_gpu_hour` and "
            f"`devices`, so dollars-per-dollar is not computable. The rate is "
            f"frozen with the task on purpose: an objective whose denominator "
            f"the measurer could pick is not an objective")

    if not args.endpoint and not args.launch:
        raise SystemExit(
            "--measure needs --endpoint HOST:PORT, or --launch to stand the "
            "target up here")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    offered = (None if args.check_protocol_context
               else resolve_concurrency(module, args))
    if offered is not None:
        print(f"  offered concurrency {offered}", flush=True)
    full_replay_fidelity: dict[str, Any] | None = None
    try:
        if args.launch:
            host, port = _launch(contract, args)
        else:
            host, _, port = args.endpoint.rpartition(":")
            if "://" in args.endpoint or not port.isdigit():
                raise SystemExit(
                    f"--endpoint takes HOST:PORT, got {args.endpoint!r}")
            port = int(port)
        url = f"http://{host}:{port}"
        context_target = _context_probe_tokens(module)
        protocol_tokenizer = None
        if args.check_protocol_context or not args.measure_only:
            try:
                from transformers import AutoTokenizer
                protocol_tokenizer = AutoTokenizer.from_pretrained(
                    args.model_dir, local_files_only=True)
            except (ImportError, OSError, ValueError) as exc:
                raise SystemExit(
                    "correctness cannot verify streamed output and context "
                    "without the staged checkpoint tokenizer: "
                    f"{type(exc).__name__}: {exc}") from exc
        # Wire representation and any declared long-context input-fidelity
        # requirement. Output correctness is verified privately after the run.
        # This path deliberately avoids the timed AgentX replay.
        if args.check_protocol_context:
            # Do not invoke measure_one here: that loads all 393 traces and
            # builds the conversation mmap. The streaming probe catches the
            # incremental-detokenisation failures that cost replay fidelity.
            row = {"arm": args.target}
            row["protocol_probe"] = run_stream_protocol_probe(
                url, args.run_dir / "agentx" / args.target,
                tokenizer=protocol_tokenizer)
            row["context_probe"] = (
                run_context_fidelity_probe(
                    url, args.run_dir / "agentx" / args.target,
                    tokenizer=protocol_tokenizer,
                    target_tokens=context_target)
                if context_target is not None else None)
            row["correctness"] = combine_endpoint_correctness(
                row["protocol_probe"], row["context_probe"])
            # The EOS gate, separately (and so in both of ./full-test's stages).
            row["eos_gate"] = run_eos_check(url, args.run_dir / "agentx" / args.target)
        else:
            row = measure_one(
                task_id, args.target, url=url, tokenizer=args.model_dir,
                run_dir=args.run_dir, duration_s=duration,
                verbose=args.verbose, diagnostic=args.measure_only,
                private_workload=not (args.measure_only or args.development_workload),
                concurrency=offered)
            # Save the server's output NOW, while the evidence of what it did
            # during the replay is certainly still in the container. Anything
            # after this -- the correctness probes -- can hang, and a hung
            # verdict is killed with SIGKILL, which no teardown survives.
            _capture_live_logs()
            if not args.measure_only:
                # Protocol and declared context integrity remain black-box
                # checks on the candidate.
                row["protocol_probe"] = run_stream_protocol_probe(
                    url, args.run_dir / "agentx" / args.target,
                    tokenizer=protocol_tokenizer)
                row["context_probe"] = (
                    run_context_fidelity_probe(
                        url, args.run_dir / "agentx" / args.target,
                        tokenizer=protocol_tokenizer,
                        target_tokens=context_target)
                    if context_target is not None else None)
                replay_fidelity = combine_replay_fidelity(
                    row.get("correctness") or {}, row["protocol_probe"],
                    row["context_probe"])
                full_replay_fidelity = replay_fidelity
                if not IN_HARBOR and not args.development_workload:
                    row.update(_private().candidate_probes(
                        url, args.run_dir, args.target, contract,
                        concurrency=offered))
    finally:
        _teardown()

    # THE PRIVATE CORRECTNESS STAGE. Host-only: see `private_correctness`.
    # A development full test (`--development-workload`) is public evidence
    # returned to the agent, so it never runs this stage: its correctness
    # record is replay fidelity alone and says so.
    if (not args.check_protocol_context and not args.measure_only
            and full_replay_fidelity is not None
            and args.development_workload):
        row["correctness"] = {
            **full_replay_fidelity, "evaluable": True,
            "scope": "replay fidelity and protocol only; output correctness "
                     "is verified privately after the run"}
    elif (not args.check_protocol_context and not args.measure_only
            and full_replay_fidelity is not None):
        row["correctness"] = _private().final_correctness(
            contract=contract, args=args, row=row,
            full_replay_fidelity=full_replay_fidelity,
            tokenizer=protocol_tokenizer, launch=_launch, teardown=_teardown)

    if args.measure_only:
        # Performance metrics, the per-request p90 scoring, AND THE PART OF
        # CORRECTNESS THIS REPLAY ACTUALLY OBSERVED. Output correctness and the
        # independent wire/context probes are named as unread below. Calling
        # all correctness unread would hide an OSL mismatch already present in
        # this row and point at --check-protocol-context, which deliberately runs
        # no AgentX replay and cannot observe it.
        row["stream_timing"] = (row.get("stream_timing") or stream_timing.assess(
            Path(args.run_dir) / "agentx" / args.target))
        measured_report = development_report(
            contract, row, usd_per_hour=usd_per_hour, mode="full",
            candidate_tree_sha=args.candidate_tree_sha,
            candidate_version=args.candidate_version,
            source_snapshot=args.source_snapshot)
        _write_development_report(
            args.run_dir / "development-evidence.json", measured_report)
        body = row.get("aggregate") or {}
        if not body:
            raw = _partial_evidence(Path(args.run_dir) / "agentx" / args.target)
            body = raw.get("aggregate") or {}
        print(f"\n{task_id} / {args.target} -- DEVELOPMENT METRICS ONLY "
              f"({duration} s window; "
              + ("the same length as the final verdict)" if duration == 1800
                 else "the final verdict runs 1800 s, so its tail can differ)"))
        if not body:
            print("  aiperf wrote no aggregate. See the log above.")
            return 1
        print(f"  {'offered_concurrency':22}{row.get('offered_concurrency')}")
        print(f"  {'workload_seed':22}"
              f"{(row.get('workload_selection') or {}).get('seed')}")
        for field in ("ttft_p90_ms", "ttft_p95_ms", "tpot_p90_ms",
                      "tpot_p95_ms",
                      "output_tokens_per_s", "request_count",
                      "benchmark_duration_s"):
            if body.get(field) is not None:
                print(f"  {field:22}{body[field]}")

        # AGAINST THE BAR, because raw percentiles are not a signal.
        #
        # Show the measured values against the task's declared gates. This is
        # important when objective improvements trade against tail latency.
        levels = list((contract.slo or {}).get("service_levels") or [])
        # The same assessment the verdict makes, from the same module, so the
        # agent is told what the verdict will conclude rather than a pass the
        # verdict will withdraw.
        timing = (row.get("stream_timing")
                  or stream_timing.assess(
                      Path(args.run_dir) / "agentx" / args.target))
        if timing.get("buffered"):
            print(f"\n  STREAM NOT MEASURABLE: {timing['why']}")
        if levels:
            print("\n  aggregate percentiles against this task's p90 bars:")
            for level in levels:
                pct = int(level["percentile"])
                ttft = body.get(f"ttft_p{pct}_ms")
                tpot = (None if timing.get("buffered")
                        else body.get(f"tpot_p{pct}_ms"))
                ttft_ceiling = float(level["ttft_ms"])
                tpot_ceiling = float(level["tpot_ms"])
                for name, got, bar, ok in (
                        (f"p{pct} TTFT", ttft, ttft_ceiling,
                         ttft is not None and ttft <= ttft_ceiling),
                        (f"p{pct} TPOT", tpot, tpot_ceiling,
                         tpot is not None and tpot <= tpot_ceiling)):
                    mark = "BELOW" if ok else "ABOVE"
                    shown = f"{got:,.0f}" if got is not None else "-"
                    print(f"    {mark}  {name:18} {shown:>10}  "
                          f"<= {bar:,.2f}")
            print("  (percentiles are diagnostic; each request is scored on "
                  "its own TTFT and TPOT)")
        _print_request_scores(measured_report)
        replay = row.get("correctness") or {}
        print("\n  actual-workload replay fidelity  "
              f"{'HOLDS' if replay.get('holds') else 'MISSES'}")
        print(f"    OSL mismatches    "
              f"{replay.get('osl_mismatch_count', '-')}")
        print(f"    request errors    "
              f"{replay.get('error_request_count', '-')}")
        exact = replay.get("exact_output_length") or {}
        if exact:
            wrong = (int(exact.get("mismatched") or 0)
                     + int(exact.get("missing_native_ids") or 0))
            checked = int(exact.get("checked") or 0)
            print(f"    exact lengths     {checked - wrong}/{checked} replies "
                  "carried exactly the requested native token ids")
        if replay.get("why"):
            print(f"    why               {replay['why']}")
        print("\n  PROTOCOL/CONTEXT checks were NOT run by this mode. "
              "Run --check-protocol-context separately. Its pass does not repair "
              "a replay-fidelity miss above, because that mode runs no "
              "AgentX replay.")
        print("  OUTPUT CORRECTNESS WAS NOT READ: it is verified privately "
              "after the run, and nothing in this evaluator measures it.")
        print("  NOT an accepted final score: development-evidence.json "
              "records the measured objective and latency evidence, but "
              "output correctness remains unmeasured.")
        return 0 if (replay.get("holds")
                     and measured_report.get("objective_value") is not None) else 1

    if args.check_protocol_context:
        _write_development_report(
            args.run_dir / "development-evidence.json",
            development_report(contract, row, usd_per_hour=usd_per_hour,
                               mode="correctness",
                               candidate_tree_sha=args.candidate_tree_sha,
                               candidate_version=args.candidate_version,
                               source_snapshot=args.source_snapshot))
        # Correctness probes only, and it says so. Replay fidelity is checked
        # by a real scenario measurement; silently loading that scenario here
        # would make an independent check take many minutes before request 1.
        # A declared near-maximum context probe may itself take minutes,
        # because there is no short substitute for reading that input.
        got = row.get("correctness") or {}
        eos = row.get("eos_gate") or {}
        print(f"\n{task_id} / {args.target} -- PROTOCOL/CONTEXT and EOS checks")
        print("  ANSWER FIDELITY: NOT RUN")
        print("  THIS IS NOT A COMPLETE CORRECTNESS PASS")
        print(f"  REPLAY GATE, protocol/context probes   {'PASS' if got.get('holds') else 'FAIL'}")
        print(f"  EOS GATE, stop at end-of-sequence      {'PASS' if eos.get('holds') else 'FAIL'}")
        print("  CORRECTNESS GATE                       NOT RUN (verified privately after the run)")
        if not got:
            print("  could not be judged: no endpoint evidence was written")
            return 1
        protocol = got.get("stream_protocol") or {}
        print(f"\n  replay gate: streamed text  "
              f"{'HOLDS' if protocol.get('holds') else 'MISSES'} "
              f"({protocol.get('stream_token_id_count', 0)}/"
              f"{protocol.get('requested_tokens', 0)} native tokens)")
        context = got.get("context_fidelity")
        if context is not None:
            print(f"  replay gate: long context   "
                  f"{'HOLDS' if context.get('holds') else 'MISSES'} "
                  f"({context.get('actual_prompt_tokens', 0):,} tokens; "
                  f"middle code "
                  f"{'recovered' if context.get('recovered_verification_code') else 'missing'})")
        print(f"  EOS gate: ignore_eos=false  "
              f"{'HOLDS' if eos.get('holds') else 'MISSES'} "
              f"(finish_reason={eos.get('finish_reason')!r}, {eos.get('tokens', 0)} tokens)")
        for reason in (got.get("why", "") if not got.get("holds") else "",
                       eos.get("why", "")):
            if reason:
                print(f"\n  {reason}")
        print("\n  Output correctness and performance were NOT measured. "
              "This quick mode checks the replay gate's streaming, output-length and "
              "long-context probes, and the EOS gate's stop rule, only.")
        # 1: a replay-gate probe failed; 4: only the EOS gate failed.
        if not got.get("holds"):
            return 1
        return 0 if eos.get("holds") else 4

    verdict = decide(contract, [row], usd_per_hour)
    out = args.run_dir / f"scenario-{args.target}.json"
    out.write_text(json.dumps({"row": row, "verdict": verdict,
                              "evaluation_scope": ("development_full_test"
                                                   if args.development_workload
                                                   else "final_validation"),
                              "candidate_tree_sha": args.candidate_tree_sha or None}, indent=1) + "\n",
                   "utf-8")

    print(f"\n{task_id} / {args.target}")
    for d in verdict["arms"]:
        print(f"  outcome           {d['outcome']}")
        _print_request_scores(d)
        why = d.get("why_misses") or d.get("why")
        if why:
            print(f"  why               {why[:150]}")
        for c in d.get("service_levels", []):
            mark = "below" if c["ttft_holds"] and c["tpot_holds"] else "above"
            print(f"  [{mark}] aggregate p{c['percentile']}  ttft "
                  f"{c['ttft_ms']} <= {c['ttft_ms_gate']} ms   "
                  f"tpot {c['tpot_ms']} <= {c['tpot_ms_gate']} ms")
        if d.get(objective.GROSS) is not None:
            print(f"  work value        ${d['value_usd']:.4f} of tokens for "
                  f"${d['usd']:.4f} of GPU time")
        if d.get("tokens_per_usd") is not None:
            print(f"  tokens_per_usd     {d['tokens_per_usd']:,.0f} "
                  f"({d['total_tokens']:,} total tokens for ${d['usd']:.4f})")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
