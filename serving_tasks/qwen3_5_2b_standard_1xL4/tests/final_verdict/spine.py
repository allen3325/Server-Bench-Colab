"""A task's frozen contract, where its files live, and what a task module is.

Shared by the host verdict (`scenario.py`), the Modal backend and the evaluator
copy packaged into each task's `tests/final_verdict/`. It reads the task's two
data files -- `contract.yaml` (bars, prices, devices, window) and
`reference.yaml` (the reference arms' launch lines) -- and
resolves a task id to its directory in either home: the repository's
`serving_tasks/`, or a workspace's delivered `task_context/`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "serverbench" / "harness") not in sys.path:
    sys.path.insert(0, str(REPO / "serverbench" / "harness"))

from .arms import ARMS, arms_for, image_for, spec_for  # noqa: E402, F401

#: A task's short handle: the first hex digits of the SHA-256 of its id. A
#: task's id names what it is (model, profile, hardware) and carries no
#: ordinal, so no task is "task 1"; the hash stands in wherever a short,
#: stable handle is needed (run ids, `evolve-task.sh`).
TASK_HASH_LENGTH = 8


def task_hash(task_id: str) -> str:
    """The task's hashed id, derived from its full id and nothing else."""
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:TASK_HASH_LENGTH]


def is_task_dir(path: Path) -> bool:
    """A directory with a `contract.yaml` is a task; one without is not."""
    return (path.is_dir() and not path.name.startswith(("_", "."))
            and (path / "contract.yaml").is_file())


def task_dirs(root: Path) -> list[Path]:
    """Every task directory directly under `root`, sorted by name."""
    return sorted(p for p in Path(root).iterdir() if is_task_dir(p)) \
        if Path(root).is_dir() else []


def _tasks_root() -> Path:
    """Where task contracts live: the repository, or a workspace's delivery.

    `SERVERBENCH_TASK_ROOT` selects another host-side tree (tooling, tests).
    `SERVERBENCH_TASK_CONTEXT` names a container's delivered copy; otherwise a
    `task_context/` beside the packaged evaluator or the workspace is used.
    """
    root = os.environ.get("SERVERBENCH_TASK_ROOT", "").strip()
    if root and Path(root).is_dir():
        return Path(root).resolve()
    named = os.environ.get("SERVERBENCH_TASK_CONTEXT", "").strip()
    if named and Path(named).is_dir():
        return Path(named)
    here = Path(__file__).resolve().parent
    for guess in (here.parent / "task_context",          # tests/task_context
                  here.parent.parent / "task_context",   # workspace/task_context
                  Path.cwd() / "task_context",
                  Path("/workspace/task_context")):
        if task_dirs(guess):
            return guess
    return REPO / "serving_tasks"


TASKS_ROOT = _tasks_root()

#: The candidate plus the pinned baseline frameworks, for `--target`.
LIVE_TARGETS = ("candidate", *sorted(ARMS))


def live_targets_for(task_id: str) -> tuple[str, ...]:
    """The arms measured for this task, candidate first."""
    allowed = set(arms_for(task_id))
    return tuple(t for t in LIVE_TARGETS if t == "candidate" or t in allowed)


@dataclasses.dataclass(frozen=True)
class AgreementRule:
    """What "did the offered work" means for a task, in words."""

    name: str
    what: str
    why: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


#: The one rule the AgentX scenario tasks use. Output correctness is judged
#: separately and privately (`private_correctness`).
REPLAY_FIDELITY = AgreementRule(
    name="replay_fidelity",
    what=("every offered turn is answered with exactly the output token count "
          "its trace records, its input served whole, and the session's own "
          "inter-turn delays honoured. Exactness is available because "
          "ignore_eos removes the one legitimate reason two correct systems "
          "differ on output length"),
    why=("a scenario workload has no canonical response text, so replay "
         "fidelity protects the shape of the work; response content is "
         "verified separately"),
)


@dataclasses.dataclass(frozen=True)
class Contract:
    """The frozen facts a verdict needs, read from the task's own files."""

    task_id: str
    task_dir: Path
    objective: str
    devices: int
    #: `service_levels` and `request_scoring`.
    slo: dict[str, Any]


def load_contract(task_id: str, *, tasks_root: Path | None = None) -> Contract:
    """One task's contract, from `contract.yaml` and `reference.yaml`."""
    task_dir = resolve_task_dir(task_id, tasks_root=Path(tasks_root or TASKS_ROOT))
    try:
        from final_validate import task_spec
    except ModuleNotFoundError:          # the in-Harbor copy
        from . import task_spec  # type: ignore[no-redef]
    con = task_spec.contract(task_dir)
    return Contract(
        task_id=task_dir.name, task_dir=task_dir,
        objective=str(con.get("objective") or ""),
        devices=int(con["devices"]),
        slo={
            "service_levels": list(con.get("service_levels") or []),
            "request_scoring": dict(con.get("request_scoring") or {}),
        })


def resolve_task_dir(task_id: str, *, tasks_root: Path | None = None) -> Path:
    """The full directory name (any case) or the task's hash resolve. Nothing
    else."""
    root = Path(tasks_root or TASKS_ROOT)
    wanted = str(task_id).strip()
    candidates = task_dirs(root)
    exact = root / wanted
    if is_task_dir(exact):
        return exact
    for path in candidates:
        if wanted.lower() in (path.name.lower(), task_hash(path.name)):
            return path
    raise KeyError(
        f"{task_id!r} names no task under {root}. Known: "
        f"{[f'{p.name} ({task_hash(p.name)})' for p in candidates]}")


@dataclasses.dataclass(frozen=True)
class TaskModule:
    """What every per-task module in `tasks/` exports as `MODULE`."""

    task_id: str
    #: One line: what this task's workload is, in its own terms.
    workload_shape: str
    agreement: AgreementRule
    #: The metric the task is scored on.
    reported_objective: str
    #: Which program validates this task, for a caller that asked another.
    refuses: str = ""


class CannotValidate(RuntimeError):
    """This task cannot be validated by the requested program, with the reason."""
