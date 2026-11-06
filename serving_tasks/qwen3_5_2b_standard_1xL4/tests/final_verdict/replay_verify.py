#!/usr/bin/env python3
"""Run a small AgentX replay-fidelity preflight against a live endpoint.

This is deliberately not a shortened final benchmark. It sends a fixed,
public, multi-turn fixture through the same pinned AgentX executable, streaming
chat endpoint, ``ignore_eos`` contract, tokenizer identity, and output-length
accounting used by the full scenario. It answers one development question:

    can this endpoint complete a small AgentX replay without dropping a turn,
    returning an error, or losing enough streamed text to miss requested OSL?

It does not load the frozen Weka corpus, exercise its long-context population,
verify output correctness (that is verified privately after the run), score
requests against the TTFT/TPOT bars, or compute an objective. A PASS here is
therefore a cheap rejection filter, never evidence that the 1,800-second
replay, the complete correctness gate, or ``dollars_per_dollar`` holds.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
#: The directory that holds this package: `serverbench/harness/` on the host,
#: `tests/` in a task's packaged copy.
IMPORT_ROOT = HERE.parent
if str(IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPORT_ROOT))

# Like scenario.py, this file runs both from the source package and from the
# generated in-workspace package named ``final_verdict``.
try:
    from final_validate import scenario, spine, tasks  # noqa: E402
except ModuleNotFoundError:
    from . import scenario, spine, tasks  # type: ignore[no-redef]  # noqa: E402


SCHEMA = "final_validate.lightweight_replay_verification/1"
FIXTURE_SCHEMA = "final_validate.lightweight_replay_fixture/1"
SESSION_OUTPUT_LENGTHS = (
    (16, 48, 24),
    (24, 64, 32),
    (32, 96, 48),
    (48, 128, 64),
)
SESSION_COUNT = len(SESSION_OUTPUT_LENGTHS)
REQUEST_COUNT = sum(len(turns) for turns in SESSION_OUTPUT_LENGTHS)
#: The fixture is public and fixed, so its sampling seed is too. It is not a
#: corpus draw and never shares a request with a timed measurement.
FIXTURE_SEED = 0
TIMEOUT_SECONDS = 600.0


def fixture_rows() -> list[dict[str, Any]]:
    """Return the fixed public multi-turn work shape used by every task."""
    shared = (
        "This is a public serving-system replay-integrity preflight. "
        "Keep producing ordinary plain text until the server-enforced output "
        "limit. Do not call tools. "
        + " ".join(
            f"Shared prefix record {index:03d} describes deterministic "
            "scheduler and cache verification."
            for index in range(16)
        )
    )
    rows = []
    for index, output_lengths in enumerate(SESSION_OUTPUT_LENGTHS):
        rows.append({
            "type": "multi_turn",
            "session_id": f"replay-verification-{index:02d}",
            "turns": [
                {
                    "text": (
                        f"{shared}\nSession {index}, first turn. Continue with "
                        "plain text until the requested token limit."),
                    "role": "user",
                    "output_length": int(output_lengths[0]),
                },
                {
                    "text": (
                        f"Session {index}, second turn. Continue again with "
                        "plain text until this turn's requested token limit."),
                    "role": "user",
                    "output_length": int(output_lengths[1]),
                },
                {
                    "text": (
                        f"Session {index}, third turn after prior state reuse. "
                        "Continue with plain text until this turn's limit."),
                    "role": "user",
                    "output_length": int(output_lengths[2]),
                },
            ],
        })
    return rows


def write_fixture(destination: Path) -> dict[str, Any]:
    """Write the AgentX ``multi_turn`` JSONL and describe its contract."""
    rows = fixture_rows()
    output_lengths = [
        int(turn["output_length"])
        for row in rows for turn in row["turns"]
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        "utf-8")
    return {
        "schema": FIXTURE_SCHEMA,
        "fixture_revision": 2,
        "dataset_type": "multi_turn",
        "session_header": "X-Correlation-ID",
        "sessions": len(rows),
        "requests": sum(len(row["turns"]) for row in rows),
        "output_lengths": output_lengths,
        "maximum_output_tokens_per_request": max(output_lengths),
        "total_requested_output_tokens": sum(output_lengths),
        "path": str(destination),
    }


def build_command(module, *, url: str, fixture: Path, artifact_dir: Path,
                  model_dir: Path, executable: str = "aiperf") -> list[str]:
    """Build the small run without invoking the locked final scenario."""
    tokenizer = str(getattr(module, "MODEL_ID", "") or model_dir)
    revision = str(getattr(module, "MODEL_REVISION", "") or "")
    offered = int(getattr(module, "OFFERED_CONCURRENCY", 0) or 1)
    concurrency = min(SESSION_COUNT, max(1, offered))
    seed = FIXTURE_SEED
    command = [
        executable, "profile",
        "--model", "target",
        "--url", url,
        "--endpoint-type", "chat",
        "--streaming",
        "--input-file", str(fixture),
        "--custom-dataset-type", "multi_turn",
        "--conversation-num", str(SESSION_COUNT),
        "--concurrency", str(concurrency),
        "--tokenizer", tokenizer,
        "--random-seed", str(seed),
        # The timed requests' fields, plus the ignore_eos the scenario
        # injects itself on the real replay.
        "--extra-inputs", json.dumps(
            {"ignore_eos": True,
             **scenario._shared_agentx_module().TIMED_REQUEST_FIELDS},
            sort_keys=True, separators=(",", ":")),
        # Match the final AgentX path: its turns carry one correlation ID and
        # no synthetic X-Session-ID. This exposes state-key and slot-lifetime
        # bugs without adding another replay or a long-context fixture.
        "--session-header", "X-Correlation-ID",
        "--connection-reuse-strategy", "sticky-user-sessions",
        "--request-timeout-seconds", "300",
        "--output-artifact-dir", str(artifact_dir),
        "--export-level", "raw",
        "--export-outputs-json",
        "--ui", "simple",
    ]
    if revision:
        marker = command.index("--random-seed")
        command[marker:marker] = ["--tokenizer-revision", revision]
    return command


def _metric(body: dict[str, Any], name: str) -> float | None:
    value = body.get(name)
    if isinstance(value, dict):
        value = value.get("avg")
    return float(value) if isinstance(value, (int, float)) else None


def assess(body: dict[str, Any], *, expected_requests: int = REQUEST_COUNT,
           failure_diagnostics: dict[str, Any] | None = None,
           lengths: dict[str, Any] | None = None,
           ) -> dict[str, Any]:
    """Apply the final scenario's replay-fidelity rule plus completeness.

    `lengths` is the run's exact-output-length scan
    (`native_tokens.exact_output_lengths`), the same one the verdict applies.
    """
    replay = scenario._replay_fidelity(body, failure_diagnostics,
                                       lengths=lengths)
    request_count = _metric(body, "request_count")
    cancelled = bool(body.get("was_cancelled"))
    problems = []
    if not replay.get("holds"):
        problems.append(str(replay.get("why") or "replay fidelity failed"))
    if request_count is None:
        problems.append("AgentX exported no request_count")
    elif int(request_count) != int(expected_requests):
        problems.append(
            f"AgentX exported {request_count:g} request(s), expected "
            f"{int(expected_requests)} from the public fixture")
    if cancelled:
        problems.append("AgentX marked the run cancelled")
    return {
        **replay,
        "holds": not problems,
        "rule": "lightweight_agentx_replay_fidelity_preflight",
        "expected_request_count": int(expected_requests),
        "request_count": request_count,
        "was_cancelled": cancelled,
        "why": "; ".join(problems),
    }


def _endpoint_url(endpoint: str) -> str:
    host, separator, port = str(endpoint).rpartition(":")
    if "://" in endpoint or not separator or not host or not port.isdigit():
        raise SystemExit(f"--endpoint takes HOST:PORT, got {endpoint!r}")
    return f"http://{host}:{int(port)}"


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", "utf-8")


def _print(report: dict[str, Any]) -> None:
    result = report.get("replay_fidelity") or {}
    outcome = "PASS" if result.get("holds") else "FAIL"
    print(f"\n{report['task_id']} -- LIGHTWEIGHT AGENTX REPLAY {outcome}")
    print(
        f"  requests          {result.get('request_count')} / "
        f"{result.get('expected_request_count')}")
    print(f"  OSL mismatches     {result.get('osl_mismatch_count', '-')}")
    print(f"  request errors     {result.get('error_request_count', '-')}")
    if result.get("why"):
        print(f"  why               {result['why']}")
    if report.get("note"):
        print(f"  process note      {report['note']}")
    print(f"  evidence          {report['report_path']}")
    print(
        "\n  Scope: public multi-turn AgentX transport/work-shape preflight only.\n"
        "  It does NOT run the frozen corpus, establish the complete correctness\n"
        "  gate, read a hard TTFT/TPOT gate, or measure dollars_per_dollar.\n"
        "  Only the exact 1,800-second frozen replay can establish the actual\n"
        "  replay fidelity and hard-gate result for these tasks.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task")
    parser.add_argument("--endpoint", required=True, help="live HOST:PORT")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-context", type=Path, default=None)
    args = parser.parse_args(argv)

    tasks_root = scenario._tasks_root(args.task_context)
    task_id = spine.resolve_task_dir(args.task, tasks_root=tasks_root).name
    scenario._require_workspace_agentx(task_id, args.run_dir)
    module = tasks.python_module(task_id)
    url = _endpoint_url(args.endpoint)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    attempt = args.run_dir / "lightweight-agentx" / f"{stamp}-{os.getpid()}"
    artifact_dir = attempt / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    fixture = write_fixture(attempt / "replay-fixture.jsonl")
    command = build_command(
        module, url=url, fixture=Path(fixture["path"]),
        artifact_dir=artifact_dir, model_dir=args.model_dir)
    command = scenario._pinned_aiperf_argv(command)
    environment = dict(os.environ)
    environment.update(getattr(module, "ENVIRONMENT", {}) or {})
    log = attempt / "aiperf.log"
    report_path = args.run_dir / "replay-verification.json"
    began = time.time()
    exit_code: int | None = None
    note = ""
    try:
        process, watch = scenario._run_watching_warmup(
            command, environment, log, timeout=TIMEOUT_SECONDS)
        exit_code = int(process.returncode)
        if watch is not None and watch.stalled(time.time()):
            note = watch.why_it_failed(time.time())
    except subprocess.TimeoutExpired:
        note = f"AgentX did not finish within {TIMEOUT_SECONDS:.0f} seconds"

    aggregate_path = None
    aggregate: dict[str, Any] = {}
    candidates = sorted(artifact_dir.rglob("profile_export_aiperf.json"))
    if candidates:
        aggregate_path = candidates[-1]
        try:
            aggregate = json.loads(aggregate_path.read_text("utf-8")) or {}
        except (OSError, ValueError) as exc:
            note = "; ".join(filter(None, [
                note,
                f"AgentX aggregate is unreadable: {type(exc).__name__}: {exc}"]))

    if aggregate:
        lengths = scenario.native_tokens.exact_output_lengths(artifact_dir)
        result = assess(aggregate, lengths=lengths)
        if not result.get("holds"):
            result = assess(
                aggregate,
                failure_diagnostics=scenario.replay_failure_diagnostics(
                    artifact_dir), lengths=lengths)
        if exit_code not in (None, 0) and result.get("holds"):
            note = "; ".join(filter(None, [
                note,
                f"aiperf exited {exit_code} after writing a complete passing "
                "aggregate; replay evidence was retained like the final "
                "verdict's post-export teardown salvage"]))
    else:
        result = {
            "holds": False,
            "rule": "lightweight_agentx_replay_fidelity_preflight",
            "expected_request_count": REQUEST_COUNT,
            "request_count": None,
            "osl_mismatch_count": None,
            "error_request_count": None,
            "why": note or (
                "AgentX wrote no profile_export_aiperf.json; inspect aiperf.log"),
        }

    report = {
        "schema": SCHEMA,
        "task_id": task_id,
        "kind": "lightweight_replay_verification",
        "authoritative_full_replay_seconds": int(
            getattr(module, "VERDICT_DURATION_S", 1800) or 1800),
        "fixture": fixture,
        "command": command,
        "exit_code": exit_code,
        "wall_seconds": round(time.time() - began, 1),
        "log": str(log),
        "aggregate": str(aggregate_path) if aggregate_path else "",
        "replay_fidelity": result,
        "note": note,
        "scope": {
            "checks": [
                "AgentX can execute all public multi-turn fixture requests",
                "three turns reuse one X-Correlation-ID per short session",
                "no AgentX request error is recorded",
                "streamed output length satisfies the final replay-fidelity rule",
            ],
            "does_not_check": [
                "the frozen Weka scenario or its long-context population",
                "output correctness",
                "per-request TTFT/TPOT scoring",
                "dollars_per_dollar",
            ],
        },
        "report_path": str(report_path),
    }
    _write_report(report_path, report)
    _print(report)
    return 0 if result.get("holds") else 1


if __name__ == "__main__":
    raise SystemExit(main())
