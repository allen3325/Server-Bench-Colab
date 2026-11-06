"""One module per live task, and the registry that discovers them.

A task module declares its model, devices, context filter and reference
operating point, and re-exports the shared AgentX contract (`_agentx_mvp`).
`available()` derives the registry from files on disk; `load()` refuses a task
with no module rather than substituting a generic one.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:                                  # pragma: no cover
    from ..spine import TaskModule

HERE = Path(__file__).resolve().parent

#: Files here that are machinery rather than a task. Anything whose name starts
#: with `_` is shared machinery by convention -- `_agentx_mvp.py` holds the one
#: execution contract the live scenario tasks all inherit -- and `available()`
#: skips it rather than demanding a TASK_ID of it.
_NOT_A_TASK = {"__init__"}


def python_module(task_id: str):
    """The imported python module for a task, not its `MODULE` dataclass."""
    table = available()
    resolved = spine_resolve(task_id)
    if resolved not in table:
        raise NoModuleForTask(
            f"{task_id} has no module in {HERE}; known: {sorted(table)}")
    return importlib.import_module(f"{__name__}.{table[resolved]}")


def spine_resolve(task_id: str) -> str:
    from .. import spine

    return spine.resolve_task_dir(task_id).name


class NoModuleForTask(KeyError):
    """This task has no final-validation module, and nothing generic stands in."""


def available() -> dict[str, str]:
    """`{task_id: module name}` for every task module on disk, discovered."""
    out: dict[str, str] = {}
    for path in sorted(HERE.glob("*.py")):
        if (path.stem in _NOT_A_TASK or path.stem.startswith("test_")
                or path.stem.startswith("_")):
            continue
        module = importlib.import_module(f"{__name__}.{path.stem}")
        task_id = getattr(module, "TASK_ID", "")
        if not task_id:
            raise RuntimeError(
                f"{path} defines no TASK_ID, so nothing can dispatch to it. A "
                f"module that cannot be reached is not a validator")
        if not hasattr(module, "MODULE"):
            raise RuntimeError(
                f"{path} defines no MODULE, so the registry cannot use it")
        out[task_id] = path.stem
    return out


def load(task_id: str) -> TaskModule:
    """The `TaskModule` for one task. Refuses rather than substituting a default.

    Accepts the full directory name (any case) or the task's hash.
    """
    from .. import spine

    resolved = spine.resolve_task_dir(task_id).name
    table = available()
    if resolved not in table:
        raise NoModuleForTask(
            f"{resolved} has no module in {HERE}. Final validation is built "
            f"task by task on purpose, and there is deliberately no generic "
            f"fallback: a task validated by somebody else's workload has not "
            f"been validated. Modules present: {sorted(table)}")
    module = importlib.import_module(f"{__name__}.{table[resolved]}")
    got = module.MODULE
    if got.task_id != resolved:
        raise RuntimeError(
            f"{table[resolved]}.py declares TASK_ID {got.task_id!r} but lives "
            f"where {resolved} was asked for. A dispatch that lands on the "
            f"wrong module produces a measurement of the wrong workload under "
            f"the right name")
    return got
