#!/usr/bin/env python3
"""Measure a published version -- and, on request, the baselines -- through the final verdict.

    $SERVERBENCH_PY serverbench/harness/final_validate/tools/measure_published.py \
        --task qwen3_5_2b_standard_1xL40 \
        --version server_output/<lineage>/<version> \
        --image serverbench/evolve-agent:cu128 --model-dir <snapshot> --gpu 7

Every arm runs `final_validate/scenario.py --launch` once, candidate first
(no baseline is worth measuring against a version that will not start). The
evidence lands under the private results root, which is checked to be
unreachable from any agent workspace, and a copy plus a `FINAL.json` pointer is
deposited beside the version in its lineage. Nothing is reported back to the
coding agent.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
HARNESS = PACKAGE.parent
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from final_validate import spine  # noqa: E402
from final_validate.store import default_root, refuse_if_agent_reachable  # noqa: E402

SCENARIO = PACKAGE / "scenario.py"

#: A scenario verdict, in the vocabulary `run.py::_final_validation` parses.
#: NOT_RUN is infrastructure, never a loss: a missing arm makes a task look
#: easier than it is.
SCENARIO_STATUS = {
    "HOLDS": "VALID_PASS",
    "MISSES": "VALID_FAIL",
    "NOT_RUN": "INVALID_INFRA",
}


def run(argv: list[str], *, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        done = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONPATH": str(HARNESS)})
    return done.returncode


def unstarted_decision(log_tail: str) -> dict:
    """Classify the absence of a scenario result from the causal log marker."""
    infrastructure = ("INFRASTRUCTURE_FAILURE:" in log_tail
                      or ("subprocess.TimeoutExpired:" in log_tail
                          and "'docker', 'inspect'" in log_tail))
    measurement_failed = "MEASUREMENT_FAILED:" in log_tail
    if infrastructure:
        because = (
            "validation infrastructure did not complete, so candidate startup "
            "or the independent reference audit could not be observed. "
            "This is invalid infrastructure evidence, "
            "not a candidate loss. Its log ends:\n" + log_tail)
        correctness_why = (
            "not fully evaluated -- infrastructure failed during startup or reference validation")
    elif measurement_failed:
        because = "the server started, but full verification did not finish. Its log ends:\n" + log_tail
        correctness_why = "not fully evaluated -- inspect the retained measurement evidence"
    else:
        because = (
            "the candidate's server never answered /health, so no workload "
            "was offered and neither correctness nor any metric could be "
            "measured. This is a non-result, not a zero: a missing arm makes "
            "a task look easier than it is. Its log ends:\n" + log_tail)
        correctness_why = (
            "not evaluated -- the server did not start, so the offered work "
            "was never offered")
    status = "NOT_RUN" if infrastructure or measurement_failed else "DID_NOT_START"
    return {
        "status": status,
        "because": because,
        "correctness_holds": None,
        "correctness_why": correctness_why,
        "dollars_per_dollar": None,
        "gross_dollars_per_dollar": None,
        "tokens_per_usd": None,
    }


def _decision_from(run_dir: Path, arms: list[str]) -> dict | None:
    """The candidate's verdict from the first `scenario-<arm>.json` written."""
    for arm in arms:
        path = run_dir / f"scenario-{arm}.json"
        if not path.is_file():
            continue
        parsed = json.loads(path.read_text("utf-8"))
        measured = parsed.get("row") or {}
        for d in (parsed.get("verdict") or {}).get("arms", []):
            correctness = d.get("correctness") or measured.get("correctness") or {}
            # A verdict that failed a hard gate scores nothing, whatever
            # number its evidence carries.
            score = (d.get("dollars_per_dollar")
                     if d.get("outcome") == "HOLDS" else None)
            print(f"  {arm:14s} {d['outcome']}  "
                  f"dollars_per_dollar={score}  "
                  f"(gross {d.get('gross_dollars_per_dollar')}, diagnostic)  "
                  f"tokens_per_usd={d.get('tokens_per_usd')}")
            return {
                "status": d["outcome"],
                # A NOT_RUN arm has no `why_misses`; its reason is on the row.
                "because": d.get("why_misses") or measured.get("not_run_because") or "",
                "correctness_holds": correctness.get("holds"),
                "correctness_why": correctness.get("why", ""),
                "dollars_per_dollar": score,
                "gross_dollars_per_dollar": d.get("gross_dollars_per_dollar"),
                "tokens_per_usd": d.get("tokens_per_usd"),
            }
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--version", type=Path, required=True,
                    help="the published version directory to measure")
    ap.add_argument("--image", default="",
                    help="the task's agent image, which the published version "
                         "is served in. Baselines use their own pins")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--gpu", default="",
                    help="a device index ALREADY CLAIMED for this run. Omit to "
                         "let the verdict take a lease")
    ap.add_argument(
        "--backend", default="docker", choices=("docker", "modal", "local"),
        help="where the candidate and the private reference stage run")
    ap.add_argument("--deny", default="",
                    help="device indices to fence off, forwarded to the verdict")
    ap.add_argument("--port", type=int, default=0,
                    help="loopback port for the arm's server; default 8800 + "
                         "the first device index, so two verdicts on two cards "
                         "cannot collide")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="private evidence directory. Defaults under the "
                         "private root, which is checked to be unreachable "
                         "from an agent workspace")
    ap.add_argument("--arm", action="append", default=None,
                    choices=list(spine.LIVE_TARGETS),
                    help="repeatable; defaults to the candidate and every "
                         "baseline this task declares, candidate first")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    task_dir = spine.resolve_task_dir(args.task)
    if not args.version.is_dir():
        raise SystemExit(f"{args.version} is not a directory")

    # The verdict must measure the model the task froze; `--model-dir` is the
    # argument most easily copied from another task's command line.
    from final_validate import model_identity

    want_model, want_revision = model_identity.declared_for(task_dir)
    try:
        if args.backend != "modal":
            model_identity.check(model_dir=args.model_dir,
                                 declared_model_id=want_model,
                                 declared_revision=want_revision,
                                 task_dir=task_dir)
    except model_identity.ModelMismatch as exc:
        raise SystemExit(
            f"MODEL_MISMATCH: {exc}\n\nNothing was leased and no card "
            f"was touched. This is NOT a result about the candidate."
            ) from None

    run_dir = args.run_dir or (
        default_root(spine.REPO) / task_dir.name / "published"
        / f"{args.version.name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    refuse_if_agent_reachable(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    first = str(args.gpu).split(",")[0].strip()
    port = args.port or ((8800 + int(first)) if first.isdigit() else 8793)

    arms = args.arm or list(spine.live_targets_for(task_dir.name))
    print(f"{task_dir.name}: {len(arms)} arm(s), published version "
          f"{args.version.name}")
    print(f"  evidence {run_dir}")

    outcomes: dict[str, int] = {}
    for arm in arms:
        command = [
            sys.executable, str(SCENARIO), task_dir.name,
            "--target", arm, "--launch",
            "--model-dir", str(args.model_dir),
            "--run-dir", str(run_dir), "--port", str(port),
        ]
        if args.gpu and args.backend == "docker":
            command += ["--device", args.gpu]
        if args.deny:
            command += ["--deny", args.deny]
        if arm == "candidate" and args.backend == "docker":
            if not args.image:
                raise SystemExit(
                    "--image is required to measure the candidate: a published "
                    "version is served in the task's own agent image")
            command += ["--image", args.image,
                        "--candidate-snapshot", str(args.version)]
        if args.verbose:
            command += ["-v"]
        if args.backend == "modal":
            from lifecycle.backend import modal_python
            command = [modal_python(), "-m", "final_validate.modal_verdict",
                       "--task", task_dir.name, "--version", str(args.version),
                       "--target", arm, "--out", str(run_dir / f"modal-{arm}")]
        if args.backend == "local":
            command = [sys.executable, "-m", "final_validate.local_verdict",
                       "--task", task_dir.name, "--version", str(args.version),
                       "--model-dir", str(args.model_dir),
                       "--target", arm, "--out", str(run_dir / f"local-{arm}")]

        print(f"\n=== {arm} ===", flush=True)
        code = run(command, log=run_dir / f"{arm}.log")
        if args.backend in ("modal", "local"):
            result = run_dir / f"{args.backend}-{arm}" / f"scenario-{arm}.json"
            if code == 0 and result.is_file():
                shutil.copy2(result, run_dir / result.name)
        outcomes[arm] = code
        print(f"  exit {code}  ({run_dir / f'{arm}.log'})", flush=True)
        if arm == "candidate" and code != 0:
            print("\nthe published version did not measure, so the baselines "
                  "are not run: there is nothing to compare them to.", flush=True)
            break

    # A measured verdict wins; otherwise an arm that never started still gets
    # a status and a reason, read from its log.
    decision = _decision_from(run_dir, arms)
    if decision is None and outcomes.get("candidate", 0) != 0:
        log_path = run_dir / "candidate.log"
        try:
            tail = "\n".join(log_path.read_text(
                "utf-8", errors="ignore").splitlines()[-25:])
        except OSError:
            tail = f"(no readable log at {log_path})"
        decision = unstarted_decision(tail)

    # The status line `run.py::_final_validation` parses.
    if decision and decision.get("status") in SCENARIO_STATUS:
        mapped = SCENARIO_STATUS[decision["status"]]
        print(f"\n{mapped}: {decision.get('because') or decision['status']} "
              f"evidence at {run_dir}")

    (run_dir / "MEASURED.json").write_text(json.dumps({
        "task_id": task_dir.name,
        "version": str(args.version),
        "arms": outcomes,
        "decided": bool(decision),
        "status": (decision or {}).get("status"),
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "instrument": ("final_validate.scenario -- the pinned AgentX. Nothing "
                       "here is reported back to the coding agent"),
    }, indent=1) + "\n", "utf-8")

    deposit_into_lineage(args.version, run_dir, task_dir.name, decision)
    return 0 if all(code == 0 for code in outcomes.values()) else 1


def deposit_into_lineage(version_dir: Path, run_dir: Path, task_id: str,
                         decision: dict | None) -> Path | None:
    """Copy the verdict's evidence beside the version it judged, and point
    `FINAL.json` at it.

    A copy, so the lineage keeps its verdict if the artifacts tree is pruned.
    Never fatal: the verdict is still in the artifacts tree.
    """
    try:
        version_dir = Path(version_dir).resolve()
        lineage = version_dir.parent
        if not (lineage / "ACCEPTED_HEAD.json").is_file():
            return None                    # not a lineage root; nothing to do
        into = lineage / "FINAL_VALIDATION" / version_dir.name
        if into.exists():
            shutil.rmtree(into)
        # Keep symlinks (a candidate `.venv` points into the container) and
        # skip the venv itself: gigabytes, rebuildable from the wheelhouse.
        shutil.copytree(run_dir, into, symlinks=True,
                        ignore=shutil.ignore_patterns(
                            ".venv", "__pycache__", "*.pyc"))

        head = json.loads((lineage / "ACCEPTED_HEAD.json").read_text("utf-8"))

        # A non-result never replaces a measurement of the same version.
        status = str((decision or {}).get("status") or "")
        existing = {}
        if (lineage / "FINAL.json").is_file():
            try:
                existing = json.loads(
                    (lineage / "FINAL.json").read_text("utf-8")) or {}
            except (OSError, ValueError):
                existing = {}
        measured = {"HOLDS", "MISSES"}
        if (status not in measured
                and str(existing.get("status") or "") in measured
                and existing.get("head_version_dir") == version_dir.name):
            print(f"\nKEPT the existing verdict for {version_dir.name}: "
                  f"{existing.get('status')} was measured, and this attempt "
                  f"returned {status or 'nothing'}. A non-result does not "
                  f"replace a measurement of the same version; this run's "
                  f"evidence is in {into.relative_to(lineage)}/")
            return into
        (lineage / "FINAL.json").write_text(json.dumps({
            "kind": "final_verdict_pointer",
            "task_id": task_id,
            "head_version_id": head.get("version_id"),
            "head_version_dir": version_dir.name,
            "is_accepted_head": version_dir.name.endswith(
                f"_{head.get('version_id')}"),
            "status": (decision or {}).get("status"),
            "because": (decision or {}).get("because", ""),
            "correctness_holds": (decision or {}).get("correctness_holds"),
            "correctness_why": (decision or {}).get("correctness_why", ""),
            "dollars_per_dollar": (
                decision or {}).get("dollars_per_dollar"),
            "gross_dollars_per_dollar": (
                decision or {}).get("gross_dollars_per_dollar"),
            "tokens_per_usd": (decision or {}).get("tokens_per_usd"),
            "evidence": str(into.relative_to(lineage)),
            "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": ("the verdict for this lineage's accepted head, measured "
                     "by final_validate outside the run. A copy: the record of "
                     "every measurement is serverbench/artifacts/final_validation/"),
        }, indent=1) + "\n", "utf-8")
        print(f"\ndeposited the verdict into {lineage.name}/FINAL.json "
              f"and {into.relative_to(lineage)}/")
        return into
    except (OSError, ValueError, KeyError) as exc:             # noqa: BLE001
        print(f"\ncould not deposit the verdict into the lineage: "
              f"{type(exc).__name__}: {exc}")
        return None


if __name__ == "__main__":
    raise SystemExit(main())
