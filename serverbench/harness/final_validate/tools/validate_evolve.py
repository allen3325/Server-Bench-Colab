#!/usr/bin/env python3
"""Take one finished evolve run's accepted head to the final verdict.

Called by `lifecycle/run.py::_final_validation` at run end, once, while the
card is still held. It selects nothing and decides nothing: it reads which
version the run accepted and hands that to `measure_published.py`, which
measures the candidate through the final verdict.

## Nothing here reaches the agent

The run is over and the container is down by the time this runs, and the
results land under the private root that `store.refuse_if_agent_reachable`
checks. The coding agent gets no feedback from the final verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
HARNESS = PACKAGE.parent
REPO = HARNESS.parents[1]
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))


def accepted_head(run_root: Path) -> tuple[str, Path] | None:
    """The version this run accepted, and its directory. `None` if there is none.

    TWO SOURCES, AND THEY HOLD DIFFERENT HALVES OF THE ANSWER.

    * `SUMMARY.json` names the accepted head, and its `versions` rows carry
      `version_id`, `candidate_id`, `tree_sha`, `predecessor` and `status` --
      and NOT `directory`.
    * the journal's `version_registered` events carry `directory`.

    So the head comes from whichever source names it and the directory always
    comes from the journal. A head with no registration is reported rather than
    guessed: measuring the wrong tree is worse than measuring nothing.
    """
    head = ""
    summary = run_root / "SUMMARY.json"
    if summary.is_file():
        head = str(json.loads(summary.read_text("utf-8")).get("accepted_head") or "")

    directories: dict[str, Path] = {}
    advanced: list[str] = []
    journal = run_root / "journal.jsonl"
    if journal.is_file():
        for line in journal.read_text("utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            name = event.get("event")
            if name == "version_registered" and event.get("directory"):
                directories[str(event["version_id"])] = Path(event["directory"])
            elif name == "head_advanced" and event.get("version_id"):
                advanced.append(str(event["version_id"]))

    # The journal is authoritative for WHICH head is last, because a run that
    # ended without writing a summary still has one.
    if not head and advanced:
        head = advanced[-1]
    if not head:
        return None
    if head not in directories:
        raise SystemExit(
            f"REFUSED: this run accepted {head} and no version_registered "
            f"event records a directory for it. There is no tree to measure, "
            f"and guessing one would validate a different version")
    return head, directories[head]


def task_of(run_root: Path) -> str:
    """Which task this run was for. From what the run actually writes.

    `SUMMARY.json`'s `task_id`, then the journal's `run_created`.

    Not inferred from the run directory's name: that is a path convention, not
    a record, and a run moved or copied would validate a version against
    another task's workload.
    """
    summary = run_root / "SUMMARY.json"
    if summary.is_file():
        found = str(json.loads(summary.read_text("utf-8")).get("task_id") or "")
        if found:
            return found
    journal = run_root / "journal.jsonl"
    if journal.is_file():
        for line in journal.read_text("utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") == "run_created" and event.get("task_id"):
                return str(event["task_id"])
    return ""


def env_input_of(run_root: Path) -> Path | None:
    """Return the exact offline build-input bundle used by this run.

    The agent launcher records its resolved host-side mount in
    ``container.json``. Final validation must reuse that value rather than
    derive a similarly named directory from whichever checkout happens to run
    the verdict: evolve may deliberately receive ``--env-input`` from another
    portable artifact root.

    A record without the field returns ``None``, and the validator's
    environment/default resolution applies.
    """
    record = run_root / "container.json"
    if not record.is_file():
        return None
    try:
        raw = json.loads(record.read_text("utf-8")).get("env_input")
    except (OSError, ValueError, AttributeError):
        return None
    return Path(str(raw)).expanduser() if raw else None


def verdict_environment(run_root: Path,
                        requested: Path | None = None) -> dict[str, str]:
    """Child environment carrying the run's frozen wheelhouse location.

    An explicit controller argument wins. Otherwise the run record supplies
    the location.
    """
    env = dict(os.environ)
    source = requested or env_input_of(run_root)
    if source is None:
        return env
    source = source.resolve()
    markers = ("wheelhouse", "resolved.requirements.txt")
    missing = [name for name in markers if not (source / name).exists()]
    if not source.is_dir() or missing:
        detail = f"; missing {missing}" if missing else ""
        raise SystemExit(
            f"REFUSED: the run's recorded build-input bundle {source} is not "
            f"usable{detail}. This is infrastructure, not a candidate "
            f"startup failure.")
    env["SERVERBENCH_ENV_INPUT"] = str(source)
    return env


def backend_of(run_root: Path) -> str:
    """Which backend served this run, from its own record.

    `harbor_env.py` and `modal_env.py` both write `container.json`, and the
    container's NAME says which: a Modal sandbox is recorded as
    `modal:<sandbox id>`. That is the same string `lifecycle/backend.py`
    dispatches on, so the verdict and the driver cannot disagree about where a
    run happened.

    A run whose environment is on another machine cannot be measured on this
    one: handing `measure_published.py` LOCAL card indices for a Modal run
    would launch the candidate on a host that never had it. Saying which
    backend applies is this function's whole job.
    """
    record = run_root / "container.json"
    if not record.is_file():
        return "docker"
    try:
        body = json.loads(record.read_text("utf-8"))
    except (OSError, ValueError):
        return "docker"
    name = str(body.get("container") or "")
    declared = str(body.get("backend") or "")
    if name.startswith("local:") or declared == "local":
        return "local"
    return "modal" if name.startswith("modal:") or declared == "modal" else "docker"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--image", default="")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument(
        "--env-input", type=Path, default=None,
        help=("the frozen build-input bundle used during evolve. The lifecycle "
              "passes it explicitly; when omitted for a recorded run, "
              "container.json supplies the exact original mount"))
    ap.add_argument("--deny", default="",
                    help="device indices to fence off -- typically the cards "
                         "another run still holds. Forwarded to "
                         "measure_published.py")
    ap.add_argument("--port", type=int, default=0,
                    help="host port for the arm's server. 0 derives it from "
                         "the card, which is what keeps two concurrent "
                         "verdicts from colliding")
    ap.add_argument("--gpu", default="",
                    help="a device index ALREADY CLAIMED for this verdict. "
                         "Without it the lease picks a card itself, and it "
                         "respects every live claim -- including this "
                         "run's own, which is how a card we hold blocks us. "
                         "Hand the card in when you are holding it")
    args = ap.parse_args(argv)
    run_root = args.run.resolve()

    stop_file = run_root / "STOP_REEVALUATION"
    if stop_file.exists():
        print(f"REFUSED: reevaluation disabled by operator: {stop_file}", file=sys.stderr)
        return 1

    from final_validate import spine

    found = accepted_head(run_root)
    if found is None:
        print("REFUSED: this run accepted no version, so there is nothing to "
              "validate. The baselines alone are a datasheet, not a verdict.")
        return 0
    version_id, version_dir = found
    if not version_dir.is_dir():
        print(f"REFUSED: {version_dir} is not on disk, so the accepted head "
              f"{version_id} cannot be measured")
        return 0

    task_id = task_of(run_root)
    if not task_id:
        print("REFUSED: this run records no task id in SUMMARY.json or its "
              "journal, and guessing one would validate a version against a "
              "different task's workload")
        return 0

    task_dir = spine.resolve_task_dir(task_id)
    # THE CANDIDATE ONLY. The baselines are not re-measured per run.
    #
    # `measure_published` defaults to candidate + SGLang + vLLM + TensorRT-LLM,
    # which is right when an operator is drawing a task's datasheet and wrong
    # at the end of every evolve. Each arm is a 1800 s AgentX scenario plus a
    # model load and a corpus rebuilt from KV block hashes, so four arms spend
    # hours of GPU re-deriving three numbers that cannot have changed.
    #
    # They cannot change because everything they depend on is frozen: the task,
    # the model and its revision, the workload, the tokenizer, the pinned arm
    # images and the hardware. A baseline is measured ONCE per task and
    # recorded. A run's verdict measures the thing that actually changed.
    #
    # This is NOT a smaller sample. The candidate is offered exactly the
    # workload the baselines were offered, for exactly as long; what is skipped
    # is repetition, not measurement. Re-measure the baselines deliberately
    # when a pin moves, because a pin change invalidates every number taken
    # under the old one.
    argv_out = [
        sys.executable, str(HERE / "measure_published.py"),
        "--task", task_dir.name,
        "--version", str(version_dir),
        "--model-dir", str(args.model_dir),
        "--arm", "candidate",
    ]
    backend = backend_of(run_root)
    if backend == "modal":
        # THE CARDS ARE NOT HERE. A Modal run's GPUs, checkpoint and image all
        # live in a sandbox, so a local card index, a local image tag and a
        # local `--deny` list describe a machine this version never ran on.
        # `measure_published --backend modal` serves it in a fresh read-only
        # sandbox instead; everything else about the verdict is unchanged.
        argv_out += ["--backend", "modal"]
    elif backend == "local":
        # NO DOCKER ON THIS HOST. The version is served in fresh namespace
        # sandboxes on this machine's card (`final_validate/local_verdict.py`),
        # after every process of the agent's is gone.
        argv_out += ["--backend", "local"]
    else:
        if args.image:
            argv_out += ["--image", args.image]
        if args.gpu:
            argv_out += ["--gpu", args.gpu]
    if args.deny and backend == "docker":
        argv_out += ["--deny", args.deny]
    if args.port and backend == "docker":
        argv_out += ["--port", str(args.port)]
    print(f"final verdict for {task_dir.name} {version_id}: {version_dir}")
    print(f"  backend {backend}"
          + ("  -- a fresh read-only sandbox, not the agent's"
             if backend == "modal" else ""))
    got = subprocess.run(
        argv_out, cwd=str(REPO), check=False,
        env=verdict_environment(run_root, args.env_input))
    return got.returncode


if __name__ == "__main__":
    raise SystemExit(main())
