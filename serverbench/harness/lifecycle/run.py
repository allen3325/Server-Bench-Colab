"""Production lifecycle: build, develop, checkpoint, preserve, then validate."""

from __future__ import annotations

import argparse
import atexit
import contextlib
import dataclasses
import datetime
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from . import agent as agent_mod
from . import backend, boundary_checks
from . import billing as billing_mod
from . import instructions as instructions_mod
from . import publish as publish_mod
from . import record as ids
from . import report as report_mod
from . import workspace as ws_mod
from .boundary import CaptureStore, VersionRegistry
from .publish import CASConflict, Publisher
from .record import (
    CandidateState,
    Journal,
    RunState,
    candidate_state,
    candidates,
    move_candidate,
    move_run,
    run_state,
)

# Nothing in a run measures the candidate: the agent measures itself in-Harbor
# with the generated copy of the final verdict, and the record comes from the
# final verdict run OUTSIDE Harbor afterwards.
from .utils import describe
from .verdict import FailureClass
from .workspace import Ownership, container_is_up, device_uuid, foreign_processes

#: A RUNAWAY CEILING, not a continuation budget.
#:
#: A launch that WORKED is resumed for as long as the deadline allows, so the
#: budget is `--hours` rather than a launch count; a flat count would let a
#: loop of short idle turns end a productive run early. What is bounded is
#: CONSECUTIVE IDLE launches, below. This number exists only so a pathological
#: loop cannot hold a card until the deadline.
MAX_LAUNCHES = 400

#: How many launches in a row may end having done nothing -- see
#: `agent.was_idle` for what that means -- before the
#: run stops asking. The first idle launch is a turn that ended early, and the
#: next launch is given the task's `## Idle restart` section, which states
#: what this environment does not do. Four of those in a row, each after a
#: backoff, is an agent that is not going to start.
MAX_IDLE_LAUNCHES = 4

#: The wait before an idle relaunch: this many seconds per launch of the
#: current idle streak, capped, so an idle loop cannot burn launches in
#: seconds. The wait is NOT added back to the deadline: the agent had budget
#: and did not use it, and `--hours` still bounds its wall time.
IDLE_BACKOFF_S = 30.0
IDLE_BACKOFF_CAP_S = 300.0

#: Below this much budget a cold restart is not worth the model load and
#: the workspace rebuild; the run stops instead.
COLD_RESTART_FLOOR_S = 15 * 60

#: Fallback ceiling for the closing final validation, used when the task's own
#: cost cannot be derived (`final_validation_timeout`). Bounded, because the
#: card is still held and a run that never ends never gives it back.
FINAL_VALIDATION_TIMEOUT_S = 5400.0

#: What one arm costs BESIDES its benchmark: loading the weights onto the
#: cards, rebuilding a scenario's corpus from KV block hashes, warmup, and the
#: teardown that follows. Generous on purpose: the cost of overestimating is a
#: timeout that never fires.
ARM_STARTUP_ALLOWANCE_S = 900.0

#: The verdict does not end with the benchmark. Its private correctness stage
#: serves the plain checkpoint as the reference and runs the answer, EOS-stop
#: and context-fidelity probes against both. Sized for a large task whose
#: warmup runs long prefills, and for the reference engine's own 1800 s
#: readiness limit (`deploy.BASELINE_READY_TIMEOUT_S`).
CORRECTNESS_STAGE_ALLOWANCE_S = 6300.0


def _agent_benchmarks(workspace) -> int:
    """How many non-preflight aiperf runs the agent completed.

    Counts finished ones only -- an export directory appears when a benchmark
    STARTS and its summary is written at the end -- and ignores the harness's
    `.review/` snapshots, which are copies rather than runs. The lightweight
    replay verifier also writes an aiperf aggregate, but it is deliberately not
    a performance measurement; counting it here would erase the distinction
    its output and the task brief make explicit.
    """
    if not workspace:
        return 0
    root = Path(workspace)
    if not root.is_dir():
        return 0
    found = 0
    for path in root.rglob("profile_export_aiperf.json"):
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part.startswith(".") for part in parts[:-1]):
            continue
        if "lightweight-agentx" in parts:
            continue
        found += 1
    return found


def _agent_replay_verifications(workspace) -> int:
    """How many lightweight AgentX replay preflights wrote a report."""
    if not workspace:
        return 0
    root = Path(workspace)
    if not root.is_dir():
        return 0
    found = 0
    for path in root.rglob("replay-verification.json"):
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part.startswith(".") for part in parts[:-1]):
            continue
        found += 1
    return found


def _agent_correctness_runs(workspace) -> int:
    """How many CORRECTNESS checks the agent completed in its own workspace.

    Counted apart from performance runs because they are different acts and
    the difference is what an operator needs: an agent that checked
    correctness but never timed a replay has done half the job, not none.
    `--check-protocol-context` runs the endpoint probes and no timed replay, so
    it writes `stream-protocol-probe.json`, not a `profile_export_aiperf.json`.
    Each directory is counted once. The file does not prove that the full
    final correctness gate passed.
    """
    if not workspace:
        return 0
    root = Path(workspace)
    if not root.is_dir():
        return 0
    found = set()
    for path in root.rglob("stream-protocol-probe.json"):
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part.startswith(".") for part in parts[:-1]):
            continue
        found.add(path.parent)
    return len(found)


def _objective_detail(task_dir) -> dict:
    """The task's frozen objective inputs, from `contract.yaml`, or `{}`.

    The denominator and the token prices are ONE contract, so both come from
    one read; otherwise a trajectory point could price tokens with a table the
    task did not declare.

    The hourly denominator is DERIVED here -- the rate times the card count --
    rather than read from a stored product that could disagree with its
    factors. The per-token prices are derived the same way.
    """
    try:
        import yaml
        body = yaml.safe_load(
            (Path(task_dir) / "contract.yaml").read_text("utf-8")) or {}
        devices, rate = int(body["devices"]), float(body["usd_per_gpu_hour"])
        prices = dict(body.get("token_prices") or {})
        for kind in ("input", "output", "cached_input"):
            per_million = prices.get(f"usd_per_million_{kind}")
            if per_million is not None:
                prices[f"usd_per_{kind}_token"] = float(per_million) / 1e6
        return {"objective": body.get("objective") or "dollars_per_dollar",
                "devices": devices, "devices_charged": devices,
                "usd_per_gpu_hour": rate,
                "usd_per_hour_total": rate * devices,
                "token_prices": prices}
    except Exception:                                          # noqa: BLE001
        return {}


def _usd_per_hour(task_dir) -> float:
    """The task's frozen `objective_detail.usd_per_hour_total`, or 0."""
    try:
        return float(_objective_detail(task_dir)
                     .get("usd_per_hour_total") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def verdict_seconds(task_dir) -> float:
    """How long the verdict's scored window is, from the task's own module.

    The development window and the scored one are the SAME length, so a pass
    the agent measures is a pass it gets. A summary that reports a request
    count should still say what the window was, because the request count is a
    property of the draw and the window is not.
    """
    try:
        from .utils import ensure_harness_on_path
        ensure_harness_on_path()
        from final_validate import tasks as tasks_mod

        module = tasks_mod.python_module(Path(task_dir).name)
        return float(getattr(module, "VERDICT_DURATION_S", 0) or 0)
    except Exception:                                          # noqa: BLE001
        return 0.0


def _service_levels():
    """`final_validate.service_levels`, imported lazily.

    Same reason as `_model_identity`: the gates belong to the verdict, and
    `tests/integrity/test_the_trajectory_never_reaches_the_verdict.py` forbids
    the traffic in the OTHER direction only. One module decides what a gate is,
    and the driver asks it rather than keeping a second copy that could drift.
    """
    from .utils import ensure_harness_on_path
    ensure_harness_on_path()
    from final_validate import service_levels

    return service_levels


def _version_order(version_id: str) -> tuple[int, str]:
    """`v10` after `v9`, which a string sort gets backwards."""
    text = str(version_id)
    digits = text[1:] if text[:1] == "v" else text
    return (int(digits), text) if digits.isdigit() else (-1, text)


def _model_identity():
    """`final_validate.model_identity`, imported lazily.

    It lives on the verdict's side of the line
    `tests/integrity/test_the_trajectory_never_reaches_the_verdict.py` draws:
    BOTH the driver and the verdict must ask whether the weights on the card
    are the weights the task froze, and that test forbids the verdict importing
    anything from `lifecycle`. Lazily, because the harness core has to stay
    importable inside a bare image.
    """
    from .utils import ensure_harness_on_path
    ensure_harness_on_path()
    from final_validate import model_identity

    return model_identity


def declared_model(task_dir) -> tuple[str, str]:
    """The task's frozen `(model_id, revision)`, or `("", "")`."""
    return _model_identity().declared_for(task_dir)


def final_validation_timeout(task_dir) -> float:
    """How long this task's verdict may take, derived rather than assumed.

    A scenario verdict replays its window after loading the model and
    rebuilding sessions from their KV block hashes, then runs the correctness
    stage; a fixed ceiling would kill a verdict that was measuring correctly.
    Derived from the task's own module, so a task with a different duration
    is right without editing this.
    """
    try:
        from .utils import ensure_harness_on_path
        ensure_harness_on_path()
        from final_validate import tasks as tasks_mod

        task_id = Path(task_dir).name
        module = tasks_mod.python_module(task_id)
        duration = float(getattr(module, "VERDICT_DURATION_S", 0) or 0)
        if not duration:
            return FINAL_VALIDATION_TIMEOUT_S
        # ONE arm: `validate_evolve.py` passes `--arm candidate`, because the
        # baselines are frozen and measured once per task rather than at the
        # end of every run. Sized for two so a slow load or a retry has room.
        return 2 * (duration + ARM_STARTUP_ALLOWANCE_S) + CORRECTNESS_STAGE_ALLOWANCE_S
    except Exception:                                          # noqa: BLE001
        return FINAL_VALIDATION_TIMEOUT_S


#: The task files a run is judged against. Their combined digest is the epoch.
#: `contract.yaml` is the bars and the denominators; `reference.yaml` is the
#: measured evidence a candidate is compared against.
EPOCH_FILES = ("contract.yaml", "reference.yaml")


def control_epoch(task_dir: Path) -> str:
    h = hashlib.sha256()
    for rel in EPOCH_FILES:
        path = Path(task_dir) / rel
        h.update(rel.encode())
        h.update(path.read_bytes() if path.exists() else b"absent")
    return f"{Path(task_dir).name}-{h.hexdigest()[:12]}"


def agent_slug(model: str, effort: str = "") -> str:
    """The agent that built a lineage, as a directory-name fragment.

    A lineage is one `(task, model, agent)` triple, and comparing agents on
    the same task is the benchmark's whole point, so the agent is part of the
    directory name rather than something a reader has to open `SUMMARY.json`
    to find.

    Effort is part of the identity because it changes what the agent is: opus
    at `high` and opus at `xhigh` are different budgets for the same weights,
    and a run comparing them would otherwise produce two names differing only
    in a timestamp. It is appended only when set.

        claude-opus-5, high      -> opus-5-high
        z-ai/glm-5.3-flash       -> glm-5.3-flash
    """
    name = (model or "unknown").strip().lower()
    name = name.rsplit("/", 1)[-1]          # drop an OpenRouter vendor prefix
    for prefix in ("claude-", "anthropic-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    slug = "".join(c if (c.isalnum() or c in "-.") else "-" for c in name)
    slug = "-".join(part for part in slug.split("-") if part)
    effort = "".join(c for c in (effort or "").strip().lower() if c.isalnum())
    return f"{slug}-{effort}" if effort else (slug or "unknown")


def persisted_agent_session(journal: Journal, provider: str) -> str:
    """Return the newest provider session recorded by a completed run."""
    for event in reversed(journal.events()):
        kind = event.get("event")
        session_id = str(event.get("session_id") or "")
        if not session_id:
            continue
        if kind in {"agent_session", "agent_session_store_imported",
                    "agent_session_imported"}:
            if event.get("provider") == provider:
                return session_id
        elif kind == "agent_launched" and event.get("agent_cli") == provider:
            return session_id
    return ""


@dataclasses.dataclass
class RunPlan:
    task_dir: Path
    repo: Path
    #: Every device this run holds, in the order the task's topology wants them.
    #: `gpu` is the FIRST of them and stays an int, because the bench service,
    #: the profiler and the headroom check each address one card by index and
    #: asking them to average four would make a per-device reading meaningless.
    gpus: tuple[int, ...]
    hours: float
    image: str
    model_dir: Path
    env_input: Path
    lineage_root: Path
    seed: Path | None = None
    agent_cli: str = "claude"
    agent_model: str = "claude-opus-5"
    agent_effort: str = "xhigh"
    #: A per-run KEY=VALUE file merged into the container's environment, for a
    #: run whose agent speaks to a different model backend. Empty means the
    #: host default. Recorded by PATH in `run_created`, never by content: which
    #: backend a run used is evidence, the credential is not.
    agent_env: Path | None = None
    credentials: Path | None = None
    profile: str = "opus"
    namespace: str = ids.PRODUCTION
    #: How long `finish` waits for extraction, deterministic checks and version
    #: materialisation after a checkpoint request.
    after_capture_timeout: float = 900.0
    tasks_root: Path | None = None
    brief: str = ""
    lineage_id: str = ""
    model_id: str = "Qwen/Qwen3-4B"
    stop: str = "timebox"
    agents_dir: Path | None = None
    #: Claude Code project files layered into the workspace's `.claude/`.
    claude_project_dir: Path | None = None
    memory_dir: Path | None = None
    skill_dir: Path | None = None
    #: The second of the skills the operator's allowlist admits. Named
    #: separately rather than globbed, so adding one is a deliberate edit.
    ncu_skill_dir: Path | None = None
    #: The third.
    kernel_skill_dir: Path | None = None
    #: Complete pinned KernelWiki, delivered only to the instructed treatment.
    kernelwiki_skill_dir: Path | None = None
    #: Parent-owned L40/skill metadata layered over the clean KernelWiki
    #: submodule only in the agent's materialized copy.
    kernelwiki_overlay_dir: Path | None = None
    #: A completed Codex run whose provider thread supplies conversation state
    #: to this separately identified make-up run. Source code still comes from
    #: the lineage's published head unless `resume_workspace` is explicitly
    #: selected for recovery from an externally interrupted run.
    resume_run: Path | None = None
    #: Recover the source files that were present in `resume_run/workspace` at
    #: interruption time.  Harness-owned files are regenerated; only the same
    #: candidate manifest that publication would capture crosses the boundary.
    resume_workspace: bool = False
    resume_session_id: str = ""
    dry_run: bool = False

    @property
    def gpu(self) -> int:
        """The primary device.

        Everything that addresses ONE card reads this: the bench service's
        `device_index`, the profiler, and the headroom check. Averaging four
        cards into one reading would make each of those meaningless, so a
        multi-device run still has a primary and the rest are held, checked and
        granted to the container alongside it.
        """
        return self.gpus[0]

    @property
    def device_spec(self) -> str:
        """What docker's `--gpus device=` wants: every card, comma separated."""
        return ",".join(str(g) for g in self.gpus)

    def __post_init__(self) -> None:
        if not self.gpus:
            raise ValueError("a run holds at least one device")
        if self.dry_run and self.namespace == ids.PRODUCTION:
            # A dry run never consumes a production identity.
            self.namespace = ids.SCRATCH
        # NO SKILLS OR AGENT DEFINITIONS ARE DEFAULTED BY THE DATA CLASS.
        #
        # What the agent is told is ONE document per task,
        # `serving_tasks/<TASK>/instruction.md`, and `instructions.py` only
        # loads it. The parser later calls `_guidance_defaults`: standard and
        # instructed task IDs select different, explicit skill roots. No
        # standalone knowledge tree exists in the plan or materializer; the
        # instructed model/framework notes live inside its skill. Skill fields
        # remain explicitly overridable for an audited experiment.
        #
        # What this lineage already learned, when there is a lineage to learn
        # from. A fresh one has none, and that is not a failure.
        if self.memory_dir is None and self.lineage_id:
            found = Path(self.lineage_root) / self.lineage_id / "MEMORY"
            self.memory_dir = found if found.is_dir() else None


class Lifecycle:
    def __init__(self, plan: RunPlan) -> None:
        self.plan = plan
        self.identity: ids.RunIdentity | None = None
        self.journal: Journal | None = None
        self.own: Ownership | None = None
        self.store: CaptureStore | None = None
        self.publisher: Publisher | None = None
        self.registry: VersionRegistry | None = None
        self.built: ws_mod.BuiltWorkspace | None = None
        self.container = ""
        self._claimed_device_uuids: dict[int, str] | None = None
        self._round = 0
        self._head_at_start = ""
        self._last_tree = ""
        self._last_commit = ""
        self._seen: set[str] = set()
        self._automatic_checkpoints: set[str] = set()
        #: A concrete version explicitly selected by the agent is a rollback
        #: override.  The ordinary selection rule is the final working tree,
        #: captured automatically at run end; ``promote latest`` is therefore
        #: only a request to catch up, not an instruction to pin an older tree.
        self._explicit_head_override = ""
        #: Candidates whose verdict this harness could not reach. Recorded
        #: once each, and asked again at run end.
        self._withheld: set[str] = set()
        self._threads: list[threading.Thread] = []
        #: The final validation's child process, once started. Held so a
        #: second caller waits for it instead of launching a rival.
        self._final_validation_child: subprocess.Popen | None = None
        #: Set once the run is wrapping up so no new capture begins after the
        #: remaining publication work has been joined.
        self._ending = threading.Event()

    # ------------------------------------------------------------------ create
    def create(self) -> ids.RunIdentity:
        plan = self.plan
        epoch = control_epoch(plan.task_dir)
        deadline = (datetime.datetime.now(datetime.timezone.utc)
                    + datetime.timedelta(hours=plan.hours)
                    ).strftime("%Y-%m-%dT%H:%M:%SZ")
        # TIMESTAMP FIRST, so `ls` is chronological order.
        #
        # `server_output/` accumulates one directory per lineage and is
        # read by eye more often than by anything else, so the newest lineage
        # sorts last. `lineage_of` derives a lineage from the PATH a seed sits
        # inside, by splitting against `lineage_root`; nothing parses this
        # string for its parts.
        lineage_id = plan.lineage_id or (
            f"{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}-"
            f"{Path(plan.task_dir).name.lower()}-"
            f"{agent_slug(plan.agent_model, plan.agent_effort)}")
        self.identity = ids.allocate(
            repo=plan.repo, task_id=Path(plan.task_dir).name,
            lineage_id=lineage_id, control_epoch=epoch,
            deadline_utc=deadline, namespace=plan.namespace)
        self.journal = Journal(self.identity.root)
        self.own = Ownership(self.journal)
        if plan.resume_run is not None:
            source_root = Path(plan.resume_run).resolve()
            source_identity = ids.load(source_root)
            if source_identity.lineage_id != lineage_id:
                raise ValueError(
                    f"resume run {source_identity.run_id} belongs to lineage "
                    f"{source_identity.lineage_id!r}, not {lineage_id!r}")
            source_journal = Journal(source_root)
            provider = plan.agent_cli
            session_id = persisted_agent_session(source_journal, provider)
            if not session_id:
                raise ValueError(
                    f"resume run {source_identity.run_id} records no "
                    f"{provider} session")
            if provider == "codex":
                # Codex keeps the rollout in a local session store, so both
                # the id and its JSONL must cross the run boundary. Claude's
                # OAuth-backed CLI resolves its recorded session remotely and
                # needs only the id; agent.launch already issues --resume.
                source_store = source_root / "codex-sessions"
                if not source_store.is_dir() or not any(
                        session_id in path.name
                        for path in source_store.rglob("*.jsonl")):
                    raise ValueError(
                        f"resume run {source_identity.run_id} does not retain "
                        f"the rollout for Codex thread {session_id}")
                destination = self.identity.root / "codex-sessions"
                shutil.copytree(source_store, destination)
            plan.resume_session_id = session_id
            # Claude Code keeps the conversation in ~/.claude/projects INSIDE
            # the container, which a make-up run's new container does not have:
            # `--resume <id>` would find nothing and the run would cold-restart.
            # An operator who saved the old container's projects dir to
            # `<source run>/claude-projects` gets it copied in before the first
            # launch.
            seed = source_root / "claude-projects"
            self._claude_session_seed = (
                seed if provider == "claude"
                and (seed / "-workspace" / f"{session_id}.jsonl").is_file() else None)
            self.journal.append(
                "agent_session", provider=provider, session_id=session_id,
                imported=True, source_run=source_identity.run_id)
            import_event = ("agent_session_store_imported"
                            if provider == "codex"
                            else "agent_session_imported")
            fields = {
                "provider": provider, "session_id": session_id,
                "source_run": source_identity.run_id,
                "source_root": str(source_root),
                "why": ("a separately identified make-up run continues the "
                        "persisted provider conversation while rebuilding its "
                        "workspace from the lineage's published head"),
            }
            if provider == "codex":
                fields["destination"] = str(destination)
            self.journal.append(import_event, **fields)
        lineage_dir = Path(plan.lineage_root) / lineage_id
        # Outside the published tree: server_output/<lineage>/ is a claim
        # that a lineage exists, and this is an object database. It is keyed by
        # LINEAGE and not by run, because a lineage is the chain of trees --
        # which is also what lets a later run carry its head forward and prove
        # descent from it.
        self.store = CaptureStore(
            Path(plan.lineage_root) / ".objects" / lineage_id)
        self.publisher = Publisher(lineage_dir)
        self.registry = VersionRegistry(
            lineage_dir, task_dir=plan.task_dir,
            agent={"model": plan.agent_model, "effort": plan.agent_effort})
        self._head_at_start = self.publisher.head_version()
        # What this run was ASKED FOR, beside what it was given.
        #
        # `stop` is half the acceptance rule -- `beat-all-baselines` and
        # `timebox` are different experiments with the same task, card and seed
        # -- so it is journalled here rather than left recoverable only from
        # the rendered brief.
        # `agent_model` and `agent_effort` are here for the same reason: a
        # lineage is one (task, model, agent) triple, so the triple has to be
        # readable off the evidence rather than off whoever launched it.
        self.journal.append("run_created", **self.identity.to_dict(),
                            hours=plan.hours, gpu=plan.gpu,
                            gpus=list(plan.gpus),
                            dry_run=plan.dry_run,
                            stop=plan.stop,
                            agent_cli=plan.agent_cli,
                            agent_model=plan.agent_model,
                            agent_effort=plan.agent_effort,
                            resume_run=str(plan.resume_run or ""),
                            resume_workspace=plan.resume_workspace,
                            accepted_head_at_start=self._head_at_start)
        return self.identity

    def resume(self, *, run_id: str, task_id: str, lineage_id: str,
               workspace_id: str, control_epoch_id: str) -> ids.RunIdentity:
        """Reopen exactly one run. Every field must match; nothing is guessed."""
        self.identity = ids.open_exact(
            repo=self.plan.repo, run_id=run_id, task_id=task_id,
            lineage_id=lineage_id, workspace_id=workspace_id,
            control_epoch=control_epoch_id, namespace=self.plan.namespace)
        self.journal = Journal(self.identity.root)
        self.own = Ownership(self.journal)
        lineage_dir = Path(self.plan.lineage_root) / lineage_id
        self.store = CaptureStore(
            Path(self.plan.lineage_root) / ".objects" / lineage_id)
        self.publisher = Publisher(lineage_dir)
        self.registry = VersionRegistry(
            lineage_dir, task_dir=self.plan.task_dir,
            agent={"model": self.plan.agent_model,
                   "effort": self.plan.agent_effort})
        self._head_at_start = self.publisher.head_version()
        self.journal.append("run_resumed", run_id=run_id,
                            accepted_head_at_start=self._head_at_start)
        return self.identity

    # -------------------------------------------------------------- workspace
    def build_workspace(self) -> ws_mod.BuiltWorkspace:
        plan = self.plan
        carried: list[tuple[str, bytes]] = []
        if plan.resume_workspace:
            source_workspace = Path(plan.resume_run) / "workspace"
            files = ws_mod.candidate_manifest(source_workspace)
            carried = [
                (relative, (source_workspace / relative).read_bytes())
                for relative in files
            ]
            self.journal.append(
                "interrupted_workspace_imported",
                source_run=ids.load(Path(plan.resume_run)).run_id,
                source_workspace=str(source_workspace),
                files=len(files),
                why=("explicit recovery retained unpublished candidate files "
                     "after the prior run was externally interrupted; fresh "
                     "harness controls are delivered by this run"),
            )
        else:
            head_tree = self.publisher.head_tree()
            if head_tree and self.store.exists(head_tree):
                staged = self.identity.root / ".head-source"
                self.store.extract(head_tree, staged)
                for path in sorted(staged.rglob("*")):
                    if path.is_file():
                        carried.append((str(path.relative_to(staged)),
                                        path.read_bytes()))
        # A RUN NEEDS AN OPERATING POINT BEFORE IT NEEDS A BRIEF.
        #
        # `_agentx_mvp.invocation` refuses a task whose offered concurrency is
        # undecided, and the verdict therefore cannot run -- but that refusal
        # fires at MEASUREMENT time, hours in, after the agent has optimised
        # for a load nobody would score it at.
        #
        # Refuse here instead: before a card is held, before an image is built,
        # while the fix is one calibration run.
        _require_an_operating_point(plan.task_dir)

        def text(workspace):
            return plan.brief or instructions_mod.build_brief(
                task_dir=plan.task_dir, model_id=plan.model_id,
                hours=plan.hours, carried=bool(carried),
                stop=plan.stop, workspace=workspace)
        self.built = ws_mod.build(
            identity=self.identity, task_dir=plan.task_dir, repo=plan.repo,
            journal=self.journal, seed=plan.seed, from_tree=carried,
            agents_dir=plan.agents_dir, skill_dir=plan.skill_dir,
            claude_project_dir=plan.claude_project_dir,
            ncu_skill_dir=plan.ncu_skill_dir,
            kernel_skill_dir=plan.kernel_skill_dir,
            kernelwiki_skill_dir=plan.kernelwiki_skill_dir,
            kernelwiki_overlay_dir=plan.kernelwiki_overlay_dir,
            brief=text)
        plan.brief = (self.built.path / "AGENT_TASK.md").read_text()
        self.recall_memory()
        move_run(self.journal, self.identity.run_id, run_state(self.journal),
                 RunState.WORKSPACE_READY, files=self.built.delivered_files,
                 seconds=self.built.build_seconds)
        return self.built

    def recall_memory(self) -> None:
        """Record that no lineage memory is delivered into the workspace.

        The agent sees its task, its own workspace and the evolve loop, and no
        previous evolve. The lineage's `memory.md` is collected at session end
        for the operator; it is never handed to a later agent. Journalled either way, so a run
        that delivered nothing and a driver that never called this differ.
        """
        lineage_id = (self.identity.lineage_id if self.identity is not None
                      else self.plan.lineage_id)
        self.journal.append(
            "memory_not_delivered",
            why=("no previous evolve is visible to the agent. The lineage "
                 "memory is kept for the operator and is not delivered into a "
                 "workspace"),
            lineage_id=lineage_id,
            had_memory_to_deliver=bool(self.plan.memory_dir),
            looked_in=(str(Path(self.plan.lineage_root) / lineage_id)
                       if lineage_id else ""))
        return

    def persist_memory(self) -> None:
        """Take back whatever the agent wrote, atomically, at session end.

        The agent owns the content; the harness owns only that it survives and
        that the previous version is kept beside it. No summarisation, no
        ranking, no rewriting of what the agent said.
        """
        if self.plan.memory_dir is None or self.built is None:
            return
        lineage_dir = Path(self.plan.memory_dir).parent
        try:
            from . import basic_memory
            report = basic_memory.collect(self.built.path, lineage_dir)
        except Exception as exc:                               # noqa: BLE001
            self.journal.append("memory_persist_failed", why=repr(exc)[:400],
                                classification=FailureClass.HARNESS.value)
            return
        self.journal.append("memory_persisted", lineage=str(lineage_dir),
                            **report)

    # -------------------------------------------------------------- container
    @property
    def on_modal(self) -> bool:
        """Is this run's environment a Modal sandbox rather than a container
        on this host? Asked in one place so the guards cannot drift apart."""
        return os.environ.get("SERVERBENCH_BACKEND", "").lower() == "modal"

    @property
    def on_local(self) -> bool:
        """Is this run's environment a namespace sandbox on this host (no
        docker; `serving_evolve/local_sandbox.py`)? The host is the run's own
        machine -- a Colab runtime -- so there is no fleet to claim cards
        from, no neighbour to hold them against and no daemon to ask."""
        return os.environ.get("SERVERBENCH_BACKEND", "").lower() == "local"

    def claim_devices(self) -> dict[int, str]:
        """Atomically claim every clean card before any expensive setup.

        Model identity verification hashes the staged weights and can take tens
        of seconds. Leaving the fleet claim until container setup would let a
        card selected by the launcher be taken while that check was running.
        This method is idempotent so ``main`` can reserve before verification
        and ``start_container`` can still enforce the invariant for direct
        callers.
        """
        if self.on_modal:
            # THE CARDS ARE NOT ON THIS MACHINE. There is nothing here to
            # claim, nothing to contend with, and nothing a holder process
            # could make visible to a neighbour -- the sandbox's GPUs are
            # Modal's to allocate and are exclusive by construction. Claiming
            # local indices would reserve someone else's cards on this host for
            # a run that never touches them.
            self.journal.append(
                "devices_not_local", backend="modal",
                why=("this run's GPUs are allocated by Modal inside its "
                     "sandbox; the local device claim, the foreign-process "
                     "check and the holder processes all describe this host "
                     "and do not apply"))
            self._claimed_device_uuids = {}
            return {}
        if self.on_local:
            # A SINGLE-TENANT MACHINE WITH NO CONTAINER RUNTIME. The fleet
            # claim asks docker which containers reserve a card, and there is
            # no docker here; the machine (a Colab runtime) is this run's
            # alone. What is recorded instead is that the card was idle.
            busy = foreign_processes(self.plan.gpu)
            if busy:
                raise ws_mod.HarnessSetupFailed(
                    f"GPU {self.plan.gpu} already carries {len(busy)} compute "
                    f"process(es) before the run started: {busy[:3]}")
            uuids = {index: device_uuid(index) for index in self.plan.gpus}
            self.journal.append(
                "devices_not_claimed", backend="local", uuids=uuids,
                why=("a single-tenant host with no container runtime: the "
                     "card was checked idle, and there is no fleet to claim "
                     "it from or neighbour to hold it against"))
            self._claimed_device_uuids = uuids
            return uuids
        if self._claimed_device_uuids is not None:
            return self._claimed_device_uuids
        if self.own is None:
            raise ws_mod.HarnessSetupFailed(
                "the run must be created before its devices can be claimed")
        plan = self.plan
        # EVERY card, not just the primary. A four-device run that checked
        # only its first card could start with somebody else's engine on the
        # other three, and a contended device is not a measurement.
        uuids = {index: device_uuid(index) for index in plan.gpus}
        for index in plan.gpus:
            reserved = ws_mod.container_reservations(index, uuids[index])
            if reserved:
                raise ws_mod.HarnessSetupFailed(
                    f"GPU {index} ({uuids[index]}) is reserved by running "
                    f"container(s): {reserved[:3]}. An idle container may "
                    "start GPU work again; nothing here will share its card")
            foreign = foreign_processes(index)
            if foreign:
                raise ws_mod.HarnessSetupFailed(
                    f"GPU {index} ({uuids[index]}) already carries "
                    f"{len(foreign)} compute process(es) that are not this "
                    f"run's. A contended device is not a measurement; nothing "
                    f"here will signal them")
        for index in plan.gpus:
            if not self.own.took_device(index, uuids[index]):
                raise ws_mod.HarnessSetupFailed(
                    f"GPU {index} ({uuids[index]}) could not be claimed for "
                    f"this run; another live owner may have selected it")
        self._claimed_device_uuids = uuids
        return uuids

    def start_container(self) -> str:
        """Compose and start the run's environment. Harbor, and only Harbor.

        There is no fallback to a raw `docker run`: a run that took one would
        be indistinguishable in the journal from one that did not, with a
        different set of mounts and a different uid owning the workspace. So a
        host that cannot drive Harbor REFUSES. `harbor/.venv` is
        gitignored and a fresh clone has none, so the refusal names the command
        that makes one.
        """
        uuids = self.claim_devices()
        self.container = self._start_through_harbor(uuids)
        return self.container

    def _start_on_modal(self, python: Path) -> str:
        """Stand the environment up as a Modal GPU sandbox.

        The same contract as the Harbor path: a subprocess under an interpreter
        that has the SDK, a `harbor_environment.json` naming the container, and
        a name the rest of the lifecycle can act on.
        """
        plan = self.plan
        script = plan.repo / "serverbench/scripts/agent/modal_env.py"
        if not script.is_file():
            raise ws_mod.HarnessSetupFailed(f"{script} is missing")
        hours = max(
            0.1,
            agent_mod.remaining_seconds(
                self.identity,
                paused_s=billing_mod.paused_seconds(self.journal)) / 3600.0)
        got = subprocess.run(
            [str(python), str(script),
             "--workspace", str(self.built.path),
             "--task", str(plan.task_dir),
             "--out", str(self.identity.root),
             "--hours", f"{hours:.3f}",
             "--agent-cli", plan.agent_cli,
             "--agent-model", plan.agent_model or "",
             "--agent-effort", plan.agent_effort or "",
             *(["--agent-env", str(plan.agent_env)] if plan.agent_env else []),
             *(["--credentials", str(plan.credentials)]
               if getattr(plan, "credentials", None) else [])],
            cwd=str(plan.repo), capture_output=True, text=True,
            timeout=3600, check=False)
        if got.stdout:
            print(got.stdout[-3000:])
        if got.returncode != 0:
            raise ws_mod.HarnessSetupFailed(
                f"modal_env.py exited {got.returncode}: "
                f"{(got.stdout + got.stderr)[-800:]}")
        record = self.identity.root / "harbor_environment.json"
        try:
            container = json.loads(record.read_text("utf-8"))["container"]
        except Exception as exc:                               # noqa: BLE001
            raise ws_mod.HarnessSetupFailed(
                f"the Modal environment named no container: {exc!r}") from None
        self.journal.append("container_backend",
                            **backend.describe(container))
        self.own.took_container(container)
        return container

    def _start_on_local(self) -> str:
        """Stand the environment up as a namespace sandbox on this host.

        The same contract as the other two: a setup subprocess that writes
        `harbor_environment.json` naming the container (`local:<name>`), and
        an egress broker this run must claim so teardown can stop it.
        """
        plan = self.plan
        script = plan.repo / "serverbench/scripts/agent/local_env.py"
        if not script.is_file():
            raise ws_mod.HarnessSetupFailed(f"{script} is missing")
        got = subprocess.run(
            [sys.executable, str(script),
             "--workspace", str(self.built.path),
             "--model", str(plan.model_dir),
             "--env-input", str(plan.env_input),
             "--gpu", plan.device_spec,
             "--out", str(self.identity.root),
             "--task", str(plan.task_dir),
             "--agent-cli", plan.agent_cli,
             "--agent-model", plan.agent_model or "",
             "--agent-effort", plan.agent_effort or "",
             *(["--agent-env", str(plan.agent_env)] if plan.agent_env else []),
             *(["--credentials", str(plan.credentials)]
               if getattr(plan, "credentials", None) else []),
             "--name", f"sb{self.identity.run_id.replace('-', '')[-20:]}"],
            cwd=str(plan.repo), capture_output=True, text=True,
            timeout=1800, check=False)
        if got.stdout:
            print(got.stdout[-3000:])
        if got.returncode != 0:
            raise ws_mod.HarnessSetupFailed(
                f"local_env.py exited {got.returncode}: "
                f"{(got.stdout + got.stderr)[-800:]}")
        record = self.identity.root / "harbor_environment.json"
        try:
            composed = json.loads(record.read_text("utf-8"))
            container = composed["container"]
        except Exception as exc:                               # noqa: BLE001
            raise ws_mod.HarnessSetupFailed(
                f"the local environment named no container: {exc!r}") from None
        broker_pid = ((composed.get("egress_broker") or {}).get("pid") or 0)
        if broker_pid:
            self.own.took_process(
                int(broker_pid),
                what="egress broker (the sandbox's only route out)",
                argv=[str((composed.get("egress_broker") or {}).get("endpoint")
                          or "")])
        self.journal.append("container_backend",
                            **backend.describe(container))
        self.own.took_container(container)
        return container

    def _start_through_harbor(self, uuids: dict) -> str:
        """Stand the environment up with Harbor, or raise. Returns its name.

        Harbor runs as a SUBPROCESS under `harbor/.venv/bin/python`, never
        imported here: it declares 22 runtime dependencies -- pydantic,
        litellm, datasets, fastapi, supabase, and `requests`/`httpx`, which are
        already forbidden in the core -- and needs Python 3.12 where the
        harness promises a 3.10 floor. The core has to import inside a bare
        image to verify a candidate there, so `harbor` is on
        `tests/repo/test_the_core_imports_in_a_bare_image.py`'s FORBIDDEN list; that
        check cannot see transitive imports, and without the name an `import
        harbor` would drag two forbidden packages in with the checker green.

        Every failure below raises `HarnessSetupFailed`, which `main()` turns
        into a typed `HARNESS_SETUP_FAILED` exit. None of them falls back.
        """
        plan = self.plan
        python = plan.repo / "harbor" / ".venv" / "bin" / "python"
        # ONE BACKEND PER ENVIRONMENT KIND, chosen here and nowhere else.
        # `modal_env.py` writes the same `harbor_environment.json` in the same
        # shape and names its container `modal:<sandbox>`, so everything below
        # this line is identical for both.
        if self.on_modal:
            return self._start_on_modal(python)
        if self.on_local:
            return self._start_on_local()
        script = plan.repo / "serverbench/scripts/agent/harbor_env.py"
        if not python.is_file():
            raise ws_mod.HarnessSetupFailed(
                f"no Harbor interpreter at {python}. Harbor composes and "
                f"starts every run's environment and there is no second way "
                f"to do it; `harbor/.venv` is gitignored, so run "
                f"`cd harbor && uv sync` on this host")
        if not script.is_file():
            raise ws_mod.HarnessSetupFailed(f"{script} is missing")
        got = subprocess.run(
            [str(python), str(script),
             "--workspace", str(self.built.path),
             "--model", str(plan.model_dir),
             "--env-input", str(plan.env_input),
             "--image", plan.image,
             "--gpu", plan.device_spec,
             "--out", str(self.identity.root),
             "--task", str(plan.task_dir),
             "--agent-cli", plan.agent_cli,
             "--agent-model", plan.agent_model,
             "--agent-effort", plan.agent_effort,
             *(["--agent-env", str(self.plan.agent_env)]
               if self.plan.agent_env else []),
             *(["--credentials", str(plan.credentials)] if plan.credentials else []),
             "--session-id", f"sb{self.identity.run_id.replace('-', '')[-20:]}"],
            cwd=str(plan.repo), capture_output=True, text=True,
            timeout=1800, check=False)
        if got.stdout:
            print(got.stdout[-2000:])
        if got.returncode != 0:
            raise ws_mod.HarnessSetupFailed(
                f"harbor_env.py exited {got.returncode}: "
                f"{(got.stdout + got.stderr)[-600:]}")
        record = self.identity.root / "harbor_environment.json"
        try:
            composed = json.loads(record.read_text("utf-8"))
            container = composed["container"]
        except Exception as exc:                               # noqa: BLE001
            raise ws_mod.HarnessSetupFailed(
                f"harbor reported success and named no container: {exc!r}"
            ) from None
        # CLAIM THE BROKER. It is started detached so it survives the setup
        # subprocess and keeps relaying while the agent works, so it must be
        # claimed or `release_all` cannot stop it and brokers accumulate, one
        # port each on the docker bridge. `Ownership` is the right place rather
        # than a kill in teardown, because a run may only signal what its
        # journal says it took.
        broker_pid = ((composed.get("egress_broker") or {}).get("pid") or 0)
        if broker_pid:
            self.own.took_process(
                int(broker_pid),
                what="egress broker (the container's only route out)",
                argv=[str((composed.get("egress_broker") or {}).get("endpoint")
                          or "")])
        if not container_is_up(container):
            raise ws_mod.HarnessSetupFailed(
                f"harbor's container {container} is not running")
        self.journal.append("container_backend", backend="harbor",
                            container=container,
                            image=plan.image, **_image_identity(plan.image),
                            driver="harbor.environments.docker.DockerEnvironment")
        self.own.took_container(container)
        self._hold_cards()
        return container

    #: How much each holder reserves, over and above its CUDA context. Small on
    #: purpose: it is memory the agent cannot use, and on a task whose whole
    #: point is KV capacity that cost has to be tiny and stated.
    HOLD_MIB = 256

    def _hold_cards(self) -> None:
        """Make this run's cards LOOK held to everyone, not just to us.

        Skipped entirely when the cards are not on this machine: see
        `claim_devices`.

        A card this run owns reads 0 MiB for most of the run -- the agent
        spends its first hour writing code, and the card is idle again between
        every teardown and restart of its server. Three mechanisms say it is
        taken and none is visible to another user: docker's `device_ids`
        governs only what our container sees, and the claims and lock
        directories are read by ServerBench tooling alone.

        On a shared box a neighbour can pick a card in the window where ours
        looks empty. A small allocation puts the reservation in the one place
        everybody actually reads, `nvidia-smi`.

        Recorded with `took_process`, so the run releases it the same way it
        releases everything else it started -- and released BEFORE the final
        verdict, because the verdict needs the card clear. `--parent-pid` is a
        second, independent guarantee: a holder whose driver is SIGKILLed exits
        on its own rather than becoming stale GPU memory.

        Never fatal. A host with no torch-capable interpreter simply does not
        get the reservation, and the run says so.
        """
        if self.on_modal or self.on_local:
            return
        interpreter = ws_mod.holder_interpreter()
        script = plan_repo = self.plan.repo / "serverbench/scripts/gpu_holder.py"
        if not interpreter or not script.is_file():
            self.journal.append(
                "cards_not_held",
                why=("no interpreter on this host can import torch and see a "
                     "GPU, so the cards are claimed but will not look busy to "
                     "a neighbour reading nvidia-smi"
                     if not interpreter else f"{plan_repo} is missing"))
            return
        for index in self.plan.gpus:
            try:
                argv = [interpreter, str(script), "--device", str(index),
                        "--mib", str(self.HOLD_MIB),
                        "--parent-pid", str(os.getpid())]
                # ITS OUTPUT IS EVIDENCE, NOT NOISE.
                #
                # The holder prints every take and every release; that is the
                # record needed to say whether a measurement was taken on a
                # clean card.
                log = self.identity.root / f"gpu-holder.{int(index)}.log"
                handle = log.open("a", encoding="utf-8")
                proc = subprocess.Popen(                       # noqa: S603
                    argv, stdout=handle, stderr=subprocess.STDOUT,
                    start_new_session=True)
                self.own.took_process(proc.pid, "gpu-holder", argv)
                self.journal.append(
                    "card_held", index=int(index), pid=proc.pid,
                    mib=self.HOLD_MIB, log=str(log),
                    note=("holds only until the card is first genuinely busy, "
                          "then releases and exits -- a holder resident on a "
                          "card the agent is benchmarking on changes what the "
                          "benchmark measures"))
            except (OSError, subprocess.SubprocessError) as exc:
                self.journal.append("cards_not_held", index=int(index),
                                    why=f"{type(exc).__name__}: {exc}"[:200])

    def _record_in_the_ledger(self) -> None:
        """Append this run's verdict to its task's CSV under evaluation_progress.

        ONE TASK, ONE FILE, so two tasks never share a column and a number is
        never compared across hardware by accident. Rows are ordered by
        `dollars_per_dollar`, which is the objective the suite exists to move.

        Reads only what the verdict already wrote -- FINAL.json and
        `scenario-candidate.json` -- so this cannot disagree with the verdict.
        A repeat of the same configuration is a NEW row, keyed by run id, not a
        replacement: the second measurement of a setting is the thing you
        compare the first against.

        Never fatal, and last. Every piece of evidence is already on disk by
        the time this runs; losing a spreadsheet row because a path moved must
        not cost the record of what happened.
        """
        if self.identity is None or self.plan.dry_run:
            return
        recorder = (self.plan.repo / "evaluation_progress" / "record_run.py")
        if not recorder.is_file():
            return
        try:
            got = subprocess.run(
                [sys.executable, str(recorder), "--run",
                 str(self.identity.root)],
                cwd=str(self.plan.repo), capture_output=True, text=True,
                timeout=300, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            self.journal.append("ledger_not_updated", why=describe(exc))
            return
        line = (got.stdout or got.stderr or "").strip()[:300]
        self.journal.append(
            "ledger_updated" if got.returncode == 0 else "ledger_not_updated",
            detail=line)

    def _release_holders(self, why: str) -> None:
        """Let go of the cards, by exact recorded pid.

        Called BEFORE the final verdict: the verdict refuses a card carrying a
        compute process it does not own, and a holder is exactly that from its
        point of view. Called again on teardown, which is harmless -- a pid
        already gone is recorded as such.
        """
        for event in self.own.processes:
            if event.get("what") != "gpu-holder":
                continue
            pid = int(event.get("pid") or 0)
            if not pid:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                how = "signal 15"
            except ProcessLookupError:
                how = "already gone"
            except OSError as exc:
                how = f"{type(exc).__name__}: {exc}"[:80]
            self.journal.append("card_released", pid=pid, how=how, why=why)

    def boot(self) -> ws_mod.BootReport:
        report = ws_mod.boot_contract(
            workspace=self.built.path, container=self.container,
            python="python3",
            journal=self.journal,
            withheld=("oracle", "references-qualification.json",
                      "references-final.json", "serving_evolve"))
        if not report.ok:
            self.journal.append("harness_setup_failed", why=report.failure())
        return report

    # ------------------------------------------------------------- development
    #: How often to re-check the cards for somebody else's work, in seconds.
    #: Cheap -- one `nvidia-smi` query -- and the window it closes is hours
    #: wide, so it does not need to be frequent.
    CONTENTION_EVERY_S = 300.0

    #: How often the run re-copies the operator's credentials into its own.
    #:
    #: Short, because the window this closes is measured in seconds: a host
    #: token refresh revokes the token the container holds, and the agent's
    #: next request fails.
    CREDENTIALS_EVERY_S = 60.0

    def _refresh_credentials(self, elapsed: float) -> None:
        """Keep the container's token the same token the host is using.

        The credentials reach the container as a bind of ONE FILE, and a file
        bind follows the inode it resolved at container start. Claude Code
        refreshes its OAuth token by atomic replace -- new inode -- and the
        refresh REVOKES the previous token. So without this, any refresh by
        anything else logged in as this operator kills the run with a 401.

        `harbor_env.stage_credentials` gives the run its own copy at an inode
        it owns, and this rewrites that copy IN PLACE, so the container -- which
        is holding exactly that inode -- sees the new bytes without a remount.

        Nothing about the VALUE is journalled, only that a refresh happened.
        """
        if self.plan.agent_cli != "claude":
            return

        import time as _time

        now = _time.monotonic()
        last = getattr(self, "_credentials_checked", 0.0)
        if now - last < self.CREDENTIALS_EVERY_S:
            return
        self._credentials_checked = now
        from lifecycle import credentials as creds_mod

        source = creds_mod.source_path()
        staged = creds_mod.staged_path(self.identity.root)
        if not staged.is_file():
            return
        try:
            changed = creds_mod.refresh(staged, source)
        except Exception as exc:                               # noqa: BLE001
            self.journal.append("credentials_refresh_failed",
                                why=repr(exc)[:200],
                                elapsed_seconds=round(elapsed, 1))
            return
        if not changed:
            return
        # A SANDBOX IS NOT HOLDING THIS INODE.
        #
        # The in-place rewrite above is the whole fix on docker, where the file
        # is a bind of exactly that inode. On Modal there are no bind mounts:
        # `modal_env.py` copied the token in once at setup, so rewriting the
        # host file reaches nothing and the sandbox keeps a token the refresh
        # has just revoked.
        #
        # Pushed through `backend.put_file`, which sends the bytes on stdin --
        # never on a command line -- and chowns the file to the agent at 0600,
        # exactly as the docker path leaves it.
        if self.container and (backend.is_modal(self.container)
                               or backend.is_local(self.container)):
            try:
                backend.put_file(self.container,
                                 staged.read_text("utf-8"),
                                 creds_mod.IN_CONTAINER)
            except Exception as exc:                           # noqa: BLE001
                self.journal.append(
                    "credentials_not_pushed_to_sandbox",
                    elapsed_seconds=round(elapsed, 1), why=repr(exc)[:200],
                    because=("the sandbox still holds the token the host "
                             "refresh revoked, so the agent's next request "
                             "will fail with 401 OAuth access token has been "
                             "revoked"))
                return
            self.journal.append(
                "credentials_refreshed", elapsed_seconds=round(elapsed, 1),
                backend="modal",
                why=("the operator's token changed on the host and the new "
                     "bytes were written into the sandbox, which holds a copy "
                     "rather than a bind and would otherwise keep using a "
                     "token the refresh had revoked"))
            return
        self.journal.append(
            "credentials_refreshed", elapsed_seconds=round(elapsed, 1),
            why=("the operator's token changed on the host and was copied "
                 "into this run's own file in place. The container holds "
                 "that inode, so its next request uses the new token "
                 "instead of one the refresh had revoked"))

    def _watch_for_contention(self, elapsed: float) -> None:
        """Record a foreign process that arrives on our card MID-RUN.

        `start_container` refuses a card that already carries somebody else's
        compute, but a card that is clean when a run starts can be shared an
        hour later.

        OWNERSHIP IS DECIDED BY THE CONTAINER, not by `foreign_processes`,
        which is a run-START question ("nothing of ours is on the card yet").
        Mid-run it would record the AGENT'S OWN SERVER, which restarts with a
        new pid every time the agent iterates, and a check that fires
        constantly is a check nobody reads.

        This does NOT stop the run, and deliberately so: the process is not
        ours to signal, the agent may be mid-benchmark, and killing a run over
        a neighbour's arrival throws away hours of good work for a fact that
        can simply be recorded. What it does is make the window a FACT about
        the run -- so a number measured inside it is known to have been, rather
        than quietly averaged in with the rest.
        """
        if self.on_modal or self.on_local:
            # Local: a single-tenant host with no daemon to attribute a
            # process to a container; nothing else runs here.
            # THE CARDS ARE NOT ON THIS MACHINE. Watching local indices here
            # would report a neighbour's process on this host as contention on
            # a run whose GPUs are Modal's.
            return
        import time as _time

        now = _time.monotonic()
        last = getattr(self, "_contention_checked", 0.0)
        if now - last < self.CONTENTION_EVERY_S:
            return
        self._contention_checked = now
        for index in self.plan.gpus:
            try:
                foreign = ws_mod.processes_not_in(
                    int(index), self.container or "",
                    # PROCESSES THIS RUN STARTED ARE OURS, container or not.
                    #
                    # Ownership by container alone is not enough: the GPU
                    # holders run on the HOST by necessity, because the harness
                    # interpreter carries no torch, and would otherwise be
                    # reported as neighbours. The ownership register the
                    # driver keeps answers exactly this question.
                    mine=tuple(str(e.get("pid")) for e in self.own.processes))
            except Exception:                              # noqa: BLE001
                continue                                   # never fatal
            if not foreign:
                continue
            seen = getattr(self, "_contention_seen", set())
            key = (int(index), tuple(sorted(foreign)))
            if key in seen:
                continue
            seen.add(key)
            self._contention_seen = seen
            full_test_thread = getattr(self, '_development_full_test_thread', None)
            if full_test_thread is not None and full_test_thread.is_alive():
                self.journal.append(
                    'device_activity_during_development_full_test', index=int(index),
                    elapsed_seconds=round(elapsed, 1), outside_agent_container=foreign[:8],
                    why=('the full verifier runs sibling containers on this allocation; '
                         'being outside the agent container alone does not establish foreign contention'))
                continue
            self.journal.append(
                "device_contended", index=int(index),
                elapsed_seconds=round(elapsed, 1),
                foreign=foreign[:8],
                why=("a process that is not this run's appeared on a card this "
                     "run holds, after the start-up check had passed. The run "
                     "continues -- the process is not ours to signal and the "
                     "agent may be mid-benchmark -- but every measurement from "
                     "here shares the card, so this window is recorded rather "
                     "than averaged in"))


    def develop(self) -> agent_mod.AgentRun:
        move_run(self.journal, self.identity.run_id, run_state(self.journal),
                 RunState.AGENT_RUNNING)

        def tick(elapsed: float) -> None:
            # None of these may take the agent down with them, and none may
            # fail silently. A swallowed boundary is invisible: the run looks
            # healthy and simply never produces a version.
            self.take_requested_boundaries(elapsed)
            self.take_requested_promotions(elapsed)
            self.take_evidence_boundaries(elapsed)
            self._watch_for_contention(elapsed)
            self._refresh_credentials(elapsed)

        # THE CONTINUATION LOOP.
        #
        # An agent that stops with budget left is resumed into the SAME
        # session, the same workspace and the same original deadline, so an
        # agent that exits early does not end the run as COMPLETED with its
        # budget unspent.
        #
        # What this is NOT: a supervisor. There is no new run, no new lineage,
        # no new budget and no new agent. `remaining_seconds` is computed from
        # the deadline set at creation, so continuing can never buy time.
        run: agent_mod.AgentRun | None = None
        cold_next = False
        idle_next = False
        idle_streak = 0
        self._seed_claude_session()
        while True:
            run = agent_mod.launch(
                identity=self.identity, journal=self.journal,
                container=self.container, brief=self.plan.brief,
                # The task owns every word an agent reads, including the
                # continuation and restart prompts. See
                # `lifecycle/instructions.py`.
                task_dir=self.plan.task_dir,
                model_flag=f"--model {self.plan.agent_model}",
                effort_flag=(f"--effort {self.plan.agent_effort}"
                             if self.plan.agent_effort else ""),
                agent_cli=self.plan.agent_cli,
                agent_model=self.plan.agent_model,
                agent_effort=self.plan.agent_effort,
                on_tick=tick, cold_start=cold_next,
                idle_restart=idle_next,
                resume_session_id=(
                    self.plan.resume_session_id
                    if not self.journal.last("agent_launched") else ""))
            cold_next = False
            idle_next = False
            if not run.stopped_early:
                return run
            # A CONFIGURATION FAILURE IS NOT AN AGENT THAT STOPPED.
            #
            # The loop below exists for an agent with budget left -- a crash, a
            # context exhaustion, a dropped stream -- and asking again is right
            # for all of those. It is wrong when the agent never reached its
            # model: every attempt fails exactly like the first, and the
            # run ends up reporting something about the AGENT for a defect in
            # its own command line. See `agent.terminal_failure`.
            if run.terminal_because:
                # RUNNING OUT OF MONEY IS A PAUSE, NOT A VERDICT.
                #
                # Every other terminal cause here is a configuration error that
                # no one is coming to fix mid-run: the model does not exist, no
                # credential reached the container. A spend limit is different
                # in exactly one way that matters -- an operator CAN fix it,
                # from a dashboard, in under a minute. Ending the run instead
                # discards the workspace, the lineage position and every
                # remaining GPU-hour on cards already paid for.
                #
                # So the run parks. See `lifecycle/billing.py` for what ends
                # the wait (the balance recovering, or the credentials file
                # changing -- a REPLACED key is a different token whose old
                # balance never recovers, so the file has to be watched too).
                # The pause is added back to the deadline, because `--hours`
                # bounds the AGENT's work and a parked run has no agent.
                if billing_mod.is_billing(run.terminal_because):
                    parked = billing_mod.wait_for_credit(
                        secrets_dir=self.identity.root / "secrets",
                        journal=self.journal)
                    if parked.get("resumed"):
                        continue
                self.journal.append(
                    "agent_cannot_start", why=run.terminal_because,
                    session_id=run.session_id,
                    launches=len(self.journal.events("agent_launched")),
                    remaining_seconds=round(
                        agent_mod.remaining_seconds(
                            self.identity,
                            paused_s=billing_mod.paused_seconds(self.journal)),
                        1))
                return run
            left = agent_mod.remaining_seconds(
                self.identity,
                paused_s=billing_mod.paused_seconds(self.journal))
            launches = len(self.journal.events("agent_launched"))
            if launches >= MAX_LAUNCHES:
                self.journal.append(
                    "continuation_declined", why=(
                        f"{launches} launches in this run, which is the "
                        f"runaway ceiling of {MAX_LAUNCHES}"),
                    launches=launches,
                    remaining_seconds=round(left, 1))
                return run
            # A SESSION LIMIT IS A CLOCK, NOT A CRASH. Park, do not relaunch.
            #
            # An agent that exits with "You've hit your session limit - resets
            # <time>" is not in TERMINAL_MARKERS; relaunching it would burn the
            # idle cap in seconds against a wall that clears on its own.
            #
            # Same handling as a billing pause, for the same reason: a wall
            # that clears on a clock costs the operator wall-time and must cost
            # the AGENT nothing, so the wait is added back to the deadline.
            wait_s = agent_mod.session_limited(run.stdout_path)
            if wait_s is not None and left > COLD_RESTART_FLOOR_S:
                cap = min(float(wait_s) + 60.0, billing_mod.MAX_PAUSE_S)
                self.journal.append(
                    "session_limit_paused", session_id=run.session_id,
                    wait_seconds=round(cap, 1),
                    remaining_seconds=round(left, 1),
                    why=("the agent hit its platform session limit, which "
                         "clears on a clock. The run is PARKED rather than "
                         "relaunched: retrying before the reset burns launches "
                         "against a wall that has not moved. The pause is "
                         "added back to the deadline"))
                from lifecycle import credentials as creds_mod
                credential_source = creds_mod.source_path()
                staged_credentials = creds_mod.staged_path(self.identity.root)
                credentials_before = billing_mod.fingerprint(
                    credential_source)
                slept = billing_mod.wait_out(
                    cap, journal=self.journal,
                    event="session_limit_resumed",
                    secrets_dir=credential_source,
                    why="the session limit's reset time has passed")
                del slept
                credentials_changed = (
                    billing_mod.fingerprint(credential_source)
                    != credentials_before)
                if credentials_changed:
                    creds_mod.refresh(staged_credentials, credential_source)
                    # A replacement token may belong to a different account.
                    # Start a provider session under that credential instead
                    # of resuming the old account's rate-limited session.
                    cold_next = True
                    self.journal.append(
                        "session_limit_credentials_changed",
                        why=("credentials changed while parked; the next "
                             "launch is a new provider session in the same "
                             "run and workspace"))
                continue

            # AN AGENT THAT WORKED IS RESUMED; ONE THAT DID NOTHING IS TOLD WHY.
            #
            # These two endings look the same to a launch count and deserve
            # opposite answers. See `agent.was_idle` for the measurement.
            idle = agent_mod.was_idle(run)
            idle_streak = idle_streak + 1 if idle else 0
            if idle_streak >= MAX_IDLE_LAUNCHES:
                self.journal.append(
                    "continuation_declined", why=(
                        f"{idle_streak} launches in a row ended having done "
                        f"nothing -- each under "
                        f"{agent_mod.IDLE_LAUNCH_S:.0f}s with at most "
                        f"{agent_mod.IDLE_LAUNCH_CALLS} tool calls, and all "
                        f"but the first already told what this environment "
                        f"does not do. An agent that will not start is not "
                        f"made to by asking again"),
                    idle_launches=idle_streak, launches=launches,
                    seconds=round(run.seconds, 1),
                    tool_calls=agent_mod.tool_calls(run.stdout_path),
                    remaining_seconds=round(left, 1))
                return run

            cold = False
            if not agent_mod.can_continue(
                    container=self.container, session_id=run.session_id,
                    agent_cli=self.plan.agent_cli):
                # The platform cannot resume the recorded session.
                #
                # A cold agent in the same workspace is a different experiment,
                # so what follows is NOT called a continuation. Stopping the
                # run instead would waste the rest of the budget on held cards.
                #
                # So: a cold restart, in the same run, the same lineage and
                # the same deadline, with a new session id, `continued=False`
                # on the launch, its own journal event, and a prompt that
                # opens by telling the agent its predecessor's conversation is
                # gone.
                if left <= COLD_RESTART_FLOOR_S:
                    self.journal.append(
                        "continuation_unavailable", session_id=run.session_id,
                        remaining_seconds=round(left, 1),
                        why=(f"the platform could not resume this session, and "
                             f"{left / 60:.0f} min is under the "
                             f"{COLD_RESTART_FLOOR_S / 60:.0f} min floor for a "
                             f"cold restart to be worth the model load"))
                    return run
                cold = True
                self.journal.append(
                    "cold_restart", session_id=run.session_id,
                    remaining_seconds=round(left, 1),
                    why=("the platform could not resume this session. A NEW "
                         "agent starts in the same workspace, with the same "
                         "deadline and no added budget. This is NOT a "
                         "continuation and is not recorded as one: the "
                         "conversation is gone and only the workspace carries "
                         "forward"))
            move_run(self.journal, self.identity.run_id,
                     run_state(self.journal), RunState.STOPPED_EARLY,
                     exit_code=run.exit_code,
                     remaining_seconds=round(left, 1))
            if idle and not cold:
                backoff = min(IDLE_BACKOFF_S * idle_streak,
                              IDLE_BACKOFF_CAP_S, max(left - 60.0, 0.0))
                self.journal.append(
                    "idle_restart", session_id=run.session_id,
                    idle_launches=idle_streak,
                    seconds=round(run.seconds, 1),
                    tool_calls=agent_mod.tool_calls(run.stdout_path),
                    backoff_seconds=round(backoff, 1),
                    remaining_seconds=round(left, 1),
                    why=("the launch ended without doing anything. The same "
                         "session is resumed, after a backoff, with the "
                         "task's `## Idle restart` section instead of its "
                         "plain continuation"))
                if backoff > 0:
                    time.sleep(backoff)
            move_run(self.journal, self.identity.run_id,
                     RunState.STOPPED_EARLY, RunState.AGENT_RUNNING,
                     continuing=not cold, cold_restart=cold,
                     idle_restart=idle and not cold,
                     remaining_seconds=round(left, 1))
            cold_next = cold
            idle_next = idle and not cold

    def take_requested_boundaries(self, elapsed: float = 0.0) -> list[str]:
        """Act on the version requests the agent has made, and only those.

        A round is a unit of the AGENT's work, and only the agent knows when
        one has finished. The harness polls for the request it made; it does
        not decide on a clock: a timer would freeze trees mid-edit and make the
        cadence a property of a launch flag rather than of the work.

        A method rather than a closure so a test can drive it: the two
        mutations that matter are taking a boundary nobody asked for, and
        acting on one request twice.
        """
        if backend.is_modal(self.container) and (
                time.monotonic() - getattr(self, '_last_request_sync', 0) >= 10):
            self._last_request_sync = time.monotonic()
            try:
                backend.sync_requests(self.container, self.built.path)
            except Exception as exc:
                self.journal.append('request_sync_failed', why=describe(exc))
                return []
        taken = []
        for request in ws_mod.pending(self.built.path, self._seen):
            if request['action'] == 'full_test':
                active = getattr(self, '_development_full_test_thread', None)
                if active is not None and active.is_alive():
                    continue
                self._seen.add(request['file'])
                self.journal.append('development_full_test_requested', **request)
                from . import full_test
                thread = threading.Thread(target=full_test.execute,
                                          args=(self, request), daemon=True,
                                          name='development-full-test')
                self._development_full_test_thread = thread
                self._threads.append(thread)
                thread.start()
                continue
            if request["action"] == "promote":
                continue
            self._seen.add(request["file"])
            self.journal.append("version_requested", **request)
            try:
                taken.append(self.boundary(
                    note=request["note"],
                    advance_head=request["action"] == "publish"))
            except Exception as exc:                       # noqa: BLE001
                self.journal.append(
                    "boundary_failed", why=repr(exc)[:400],
                    classification=FailureClass.HARNESS.value,
                    elapsed_seconds=elapsed)
        return taken

    def take_requested_promotions(self, elapsed: float = 0.0) -> list[str]:
        """Advance the instructed task's head without capturing a new tree."""
        promoted = []
        for request in ws_mod.pending(self.built.path, self._seen):
            if request["action"] != "promote":
                continue
            try:
                version_id = self.promote_checkpoint(
                    request["target"] or "latest", note=request["note"])
            except Exception as exc:                       # noqa: BLE001
                self._seen.add(request["file"])
                self.journal.append(
                    "promotion_failed", request_file=request["file"],
                    target=request["target"], why=repr(exc)[:400],
                    classification=FailureClass.HARNESS.value,
                    elapsed_seconds=elapsed)
                continue
            if not version_id:
                # A checkpoint request immediately followed by `promote
                # latest` may still be materialising on another thread. Keep
                # the request pending and retry on the next tick.
                continue
            self._seen.add(request["file"])
            self.journal.append("promotion_requested", **request,
                                version_id=version_id)
            promoted.append(version_id)
        return promoted

    def take_evidence_boundaries(self, elapsed: float = 0.0) -> list[str]:
        """Checkpoint the first successful/complete public evidence events.

        This is intentionally limited to instructed workspaces and to reports
        beneath `/workspace/agentx` (the host mirror is ``built.path/agentx``).
        It does not impose an order: replay, correctness, and a complete full
        window are three independent milestones.
        """
        if not self._is_instructed:
            return []
        recorded = {
            str(e.get("milestone"))
            for e in self.journal.events("automatic_checkpoint_requested")
        }
        self._automatic_checkpoints.update(recorded)
        root = self.built.path / "agentx"
        if not root.is_dir():
            return []
        found: dict[str, Path] = {}
        report_paths = []
        if "replay_passed" not in self._automatic_checkpoints:
            report_paths.extend(root.rglob("replay-verification.json"))
        if ({"replay_passed", "correctness_passed", "full_window_completed"}
                - self._automatic_checkpoints):
            report_paths.extend(root.rglob("development-evidence.json"))
        for path in sorted(report_paths):
            try:
                body = json.loads(path.read_text("utf-8"))
            except (OSError, ValueError):
                continue
            if body.get("task_id") != self.identity.task_id:
                continue
            if (path.name == "replay-verification.json"
                    and body.get("schema") ==
                    "final_validate.lightweight_replay_verification/1"
                    and (body.get("replay_fidelity") or {}).get("holds") is True):
                found.setdefault("replay_passed", path)
            elif (body.get("schema") == "final_validate.development_evidence/1"
                  and body.get("mode") == "correctness"
                  and body.get("public_correctness_holds") is True):
                found.setdefault("correctness_passed", path)
            elif (body.get("schema") == "final_validate.development_evidence/1"
                  and body.get("mode") == "full"
                  and body.get("full_window_complete") is True):
                found.setdefault("full_window_completed", path)
                if body.get("replay_fidelity_holds") is True:
                    found.setdefault("replay_passed", path)
        taken = []
        for milestone in ("replay_passed", "correctness_passed",
                          "full_window_completed"):
            if milestone in self._automatic_checkpoints or milestone not in found:
                continue
            evidence = found[milestone]
            try:
                candidate_id = self.boundary(
                    note=f"automatic checkpoint: first {milestone}",
                    advance_head=False, force=True)
            except Exception as exc:                       # noqa: BLE001
                self.journal.append(
                    "automatic_checkpoint_failed", milestone=milestone,
                    evidence=str(evidence.relative_to(self.built.path)),
                    why=repr(exc)[:400], elapsed_seconds=elapsed)
                continue
            self._automatic_checkpoints.add(milestone)
            self.journal.append(
                "automatic_checkpoint_requested", milestone=milestone,
                evidence=str(evidence.relative_to(self.built.path)),
                candidate_id=candidate_id,
                note=("an empty candidate id means this evidence already "
                      "described the most recently checkpointed source tree"))
            taken.append(candidate_id)
        return taken

    @property
    def _is_instructed(self) -> bool:
        return "_instructed_" in self.identity.task_id

    def _bring_the_tree_home(self, why: str) -> None:
        """Make the host workspace the agent's tree before anything reads it.

        A no-op on docker, where it already is one -- the workspace is a bind
        mount. On Modal the tree lives only in the sandbox, and everything that
        captures, publishes or validates reads the HOST path.

        Skipping this is not staleness, it is data loss: the host would capture
        the starter while the sandbox, terminated afterwards, held the agent's
        work.

        A failed pull is journalled loudly and refused, because what it would
        leave behind is a capture of the wrong tree.
        """
        if not self.container or not backend.is_modal(self.container):
            return
        try:
            detail = backend.pull_workspace(self.container, self.built.path)
        except Exception as exc:                               # noqa: BLE001
            self.journal.append(
                "workspace_not_pulled", why=why, error=describe(exc),
                because=("the host copy is what gets captured and published, "
                         "so whatever is read after this is NOT what the agent "
                         "built"))
            raise ws_mod.HarnessSetupFailed(
                "Modal workspace transfer failed; refusing to publish stale host source") from exc
        self.journal.append("workspace_pulled", why=why, detail=detail)

    # ---------------------------------------------------------- round boundary
    def boundary(self, note: str = "", *, advance_head: bool = True,
                 force: bool = False) -> str:
        """Capture, check in parallel, materialise, register.

        Returns the candidate id, or "" when nothing changed since the last
        boundary. The agent is not stopped: capture is an index write, and
        everything after it reads the frozen tree.
        """
        self._round += 1
        state = run_state(self.journal)
        move_run(self.journal, self.identity.run_id, state,
                 RunState.ROUND_BOUNDARY, round_index=self._round)
        self._bring_the_tree_home("a version was asked for")
        delivery = ws_mod.candidate_delivery_report(self.built.path)
        files = delivery["files"]
        (self.built.path / "DELIVERY_REPORT.json").write_text(
            json.dumps(delivery, indent=2) + "\n")
        captured = self.store.capture(
            workspace=self.built.path, files=files, round_index=self._round,
            run_id=self.identity.run_id, workspace_id=self.identity.workspace_id,
            accepted_head=self.publisher.head_tree(),
            parent=self._last_commit or self.publisher.head_commit())
        # The agent is free again from here. Everything below is on the frozen
        # tree, and the run goes back to AGENT_RUNNING immediately.
        move_run(self.journal, self.identity.run_id, RunState.ROUND_BOUNDARY,
                 RunState.AGENT_RUNNING, round_index=self._round)
        if captured.tree_sha == self._last_tree and not force:
            self.journal.append("boundary_skipped", round_index=self._round,
                                why=("the agent asked for a version and nothing "
                                     "has changed since the last one"),
                                note=note, tree_sha=captured.tree_sha)
            return ""
        self._last_tree = captured.tree_sha
        self._last_commit = captured.commit_sha
        self.journal.append("candidate_captured", **captured.to_dict())
        # The agent's own sentence about WHY it wanted this version, paired
        # with the candidate the boundary produced. Journalled rather than kept
        # on the object so it survives a resume.
        self.journal.append("candidate_noted",
                            candidate_id=captured.candidate_id,
                            round_index=self._round, note=note)

        # The candidate enters its machine here, at CAPTURED. Everything that
        # follows runs off the agent's path, on the frozen tree.
        self.journal.append("candidate_state", subject=captured.candidate_id,
                            frm=CandidateState.CAPTURED.value,
                            to=CandidateState.CAPTURED.value,
                            note="entered the candidate machine")
        thread = threading.Thread(
            target=self._after_capture,
            args=(captured, files, advance_head, delivery), daemon=True,
            name=f"boundary-{captured.candidate_id}")
        thread.start()
        self._threads.append(thread)
        return captured.candidate_id

    def _after_capture(self, captured, files: list[str],
                       advance_head: bool = True, delivery: dict | None = None) -> None:
        """Check, materialise, register and publish -- off the agent's path."""
        try:
            self._after_capture_inner(captured, files, advance_head, delivery)
        except Exception as exc:                           # noqa: BLE001
            # A thread that dies takes its traceback with it. This one says so.
            self.journal.append("after_capture_failed",
                                candidate_id=captured.candidate_id,
                                why=repr(exc)[:400],
                                classification=FailureClass.HARNESS.value)

    def _after_capture_inner(self, captured, files: list[str],
                             advance_head: bool = True, delivery: dict | None = None) -> None:
        cid = captured.candidate_id
        move_candidate(self.journal, cid, CandidateState.CAPTURED,
                       CandidateState.CHECKING)
        tree_dir = self.built.path / ".review" / cid
        self.store.extract(captured.tree_sha, tree_dir)
        result = self._boundary_checks(captured, cid, tree_dir, delivery)
        move_candidate(self.journal, cid, CandidateState.CHECKING,
                       CandidateState.CHECKS_COMPLETE,
                       seconds=result.seconds, failed=list(result.failed and
                                                           [c["name"] for c
                                                            in result.failed]))
        version = self.registry.materialize(
            captured=captured, store=self.store, identity=self.identity,
            journal=self.journal, review=result.to_dict(),
            predecessor=self.publisher.head_version(),
            memory_dir=self._memory_this_run_wrote(),
            workspace=self.built.path)
        move_candidate(self.journal, cid, CandidateState.CHECKS_COMPLETE,
                       CandidateState.MATERIALIZED, version_id=version.version_id)
        move_candidate(self.journal, cid, CandidateState.MATERIALIZED,
                       CandidateState.REGISTERED, version_id=version.version_id)
        # PUBLISHED HERE, UNJUDGED, AND THAT IS THE DESIGN.
        #
        # Nothing in the run measures a version; the repository has one
        # measurement path, the final verdict. So publishing is cheap and
        # unconditional: every captured version is
        # kept and takes the head if its tree descends from the current one.
        # What a version is WORTH is decided afterwards, by the final verdict
        # outside Harbor, on evidence the agent never sees.
        if advance_head:
            self._publish(cid, version, captured)
        else:
            self._save_checkpoint(cid, version)

    def _save_checkpoint(self, cid: str, version) -> None:
        """Keep a materialised version while leaving the final head unchanged."""
        existing = version.qualification or {}
        cost = existing.get("api_cost") or self._api_cost_for(version)
        point = existing.get("trajectory")
        if not isinstance(point, dict):
            point = self._trajectory_point(version, cost)
        self.registry.record_outcome(
            version.version_id, status="CHECKPOINTED",
            qualification={
                "judged": False,
                "judged_note": (
                    "immutable checkpoint saved; final-head promotion is a "
                    "separate explicit operation"),
                "api_cost": cost, "trajectory": point,
            }, journal=self.journal)
        move_candidate(self.journal, cid, CandidateState.REGISTERED,
                       CandidateState.CHECKPOINTED,
                       head=self.publisher.head_version(),
                       note="checkpoint kept; final head unchanged")
        self.journal.append("checkpoint_saved", candidate_id=cid,
                            version_id=version.version_id,
                            head=self.publisher.head_version())

    def _seed_claude_session(self) -> None:
        """Copy a saved Claude session into the container before the first launch."""
        seed = getattr(self, "_claude_session_seed", None)
        if seed is None or backend.is_modal(self.container):
            return
        dest = "/home/agent/.claude/projects"
        if backend.is_local(self.container):
            from serving_evolve import local_sandbox
            uid, gid = local_sandbox.agent_ids()
            try:
                shutil.copytree(seed, dest, dirs_exist_ok=True)
                for path in [Path(dest), *Path(dest).rglob("*")]:
                    os.chown(path, uid, gid)
                ok, why = True, ""
            except OSError as exc:
                ok, why = False, str(exc)[:300]
            self.journal.append("agent_session_seeded", source=str(seed),
                                destination=dest, ok=ok, why=why)
            return
        made = subprocess.run(
            backend.exec_argv(self.container, "", f"mkdir -p {dest}"),
            capture_output=True, text=True, timeout=120, check=False)
        got = subprocess.run(["docker", "cp", f"{seed}/.", f"{self.container}:{dest}/"],
                             capture_output=True, text=True, timeout=600, check=False)
        self.journal.append(
            "agent_session_seeded", source=str(seed), destination=dest,
            ok=made.returncode == 0 and got.returncode == 0,
            why=(got.stderr or made.stderr or "").strip()[:300])

    def _boundary_checks(self, captured, cid: str, tree_dir, delivery=None):
        """Record deterministic form and portability facts for a checkpoint."""
        result = boundary_checks.run(
            tree_dir, candidate_id=cid, tree_sha=captured.tree_sha,
            # So the scan can tell what WE put in the workspace from what the
            # candidate wrote; otherwise `source_scan` would flag the task's
            # own README.
            delivered_from=self.plan.task_dir,
            delivery_report=delivery,
            evidence_root=(self.built.path / "agentx") if self._is_instructed else None)
        self.journal.append(
            "boundary_checks", candidate_id=cid, tree_sha=captured.tree_sha,
            seconds=round(result.seconds, 3),
            failed=[c["name"] for c in result.failed],
            outcomes={c["name"]: c["outcome"] for c in result.checks})
        tested = self._full_test_of(tree_dir, delivery)
        self._deliver_boundary_checks(result, cid, tested)
        return result

    def _full_test_of(self, tree_dir, delivery) -> dict | None:
        """Has a development full test passed on exactly this version's tree?

        Recorded, never gating: the agent is told, so a deadline publish of an
        untested tree is a choice it makes knowingly."""
        from . import full_test
        try:
            tested = full_test.passed_on(tree_dir, list((delivery or {})["files"]),
                                         self.built.path / full_test.REPORTS)
        except (OSError, KeyError, TypeError):
            return None
        self.journal.append("version_full_test", **tested)
        return tested

    def _time_left_lines(self) -> list[str]:
        left = int(agent_mod.remaining_seconds(
            self.identity, paused_s=billing_mod.paused_seconds(self.journal)))
        hours, rest = divmod(max(0, left), 3600)
        return [f"Time left: {hours:d}:{rest // 60:02d} (remaining_seconds={left}). "
                "The final verdict judges the head as it stands at the deadline.", ""]

    def _deliver_boundary_checks(self, result, cid: str, tested: dict | None = None) -> None:
        """Write the latest checks into the workspace, best effort."""
        try:
            failed = list(result.failed)
            lines = [f"# Boundary checks — {cid}", ""]
            lines += self._time_left_lines()
            if tested is not None and tested.get("tested"):
                lines += ["## Full test of this tree", "",
                          f"Passed: `{tested['full_test']}` ran on exactly this "
                          f"tree ({tested['finished_utc']}).", ""]
            elif tested is not None:
                newest = tested.get("newest_passed")
                lines += ["## WARNING: no full test has passed on this tree", ""]
                if newest:
                    shown = ", ".join(f"`{n}`" for n in newest["changed"][:8])
                    more = len(newest["changed"]) - 8
                    lines += [f"The newest passed full test, `{newest['full_test']}` "
                              f"({newest['finished_utc']}), ran on a different tree: "
                              f"{len(newest['changed'])} file(s) differ ({shown}"
                              f"{f', and {more} more' if more > 0 else ''}).", ""]
                else:
                    lines += ["No development full test has passed in this run.", ""]
                lines += ["If this is the version judged at the deadline, it is "
                          "judged untested. Run `./full-test --wait` on it, or "
                          "publish a tree that has passed one.", ""]
            if failed:
                lines += [
                    "## Failed checks", "",
                    "These checks do not run correctness or performance "
                    "evaluation and do not reject the checkpoint. They report "
                    "form or portability issues in the published tree.", ""]
                for check in result.failed:
                    lines += [f"### {check['name']}  — FAIL", "",
                              str(check.get("detail", "")), ""]
            else:
                lines += ["Every check passed.", ""]
            lines += ["## All checks", "",
                      "| check | outcome |", "|---|---|"]
            for check in result.checks:
                lines.append(f"| `{check['name']}` | {check['outcome']} |")
            lines.append("")
            path = self.built.path / "BOUNDARY_CHECKS.md"
            path.write_text("\n".join(lines), "utf-8")
        except OSError:
            pass

    def _memory_this_run_wrote(self) -> Path | None:
        """What THIS run's agent wrote, for the operator's record.

        Published so a version carries the notes made alongside it. What is NOT
        published or delivered is a PREVIOUS run's memory: the agent sees its
        prompt, its task, its own workspace and the evolve loop, and no
        previous evolve.

        So this reads the workspace only. `plan.memory_dir` -- the inherited
        lineage memory -- is deliberately not consulted: it would be a channel
        by which a run could see what an earlier run had concluded.
        """
        if self.built is not None:
            wrote = self.built.path / "MEMORY"
            if wrote.is_dir() and any(wrote.iterdir()):
                return wrote
        return None

    # ------------------------------------------------------------- collection
    def promote_checkpoint(self, target: str, *, note: str = "") -> str:
        """Move the final head to a saved version; capture no workspace files.

        Returns an empty string while ``latest`` has not materialised yet, so a
        request queued immediately after a checkpoint can be retried safely.
        A concrete target may be either the short version id shown by the
        registry (``v7``) or the candidate id shown beside benchmark evidence
        (``r007-...``).  An unknown concrete target is an error, not a pending
        checkpoint, so a bad request cannot sit pending until run end.
        """
        ours = [v for v in self.registry.all()
                if v.run_id == self.identity.run_id]
        if target == "latest":
            if any(thread.is_alive() for thread in self._threads):
                return ""
            if not ours:
                return ""
            version = max(ours, key=lambda v: (v.round_index, v.version_id))
        else:
            version = self.registry.get(target)
            if version is None:
                matches = [v for v in ours if v.candidate_id == target]
                if len(matches) == 1:
                    version = matches[0]
                else:
                    raise ValueError(
                        f"{target} is not a version id or candidate id from "
                        f"run {self.identity.run_id}")
            if version.run_id != self.identity.run_id:
                raise ValueError(
                    f"{target} belongs to {version.run_id}, not this run")
        self._publish(version.candidate_id, version, captured=None,
                      explicit_selection=True)
        if self.publisher.head_version() != version.version_id:
            raise RuntimeError(
                f"{version.version_id} was kept but did not become the head")
        self.journal.append("checkpoint_promoted", version_id=version.version_id,
                            candidate_id=version.candidate_id, note=note,
                            head=self.publisher.head_version())
        # `./promote latest` withdraws an earlier concrete selection: the run
        # end then takes the newest checkpoint, as the script says.
        self._explicit_head_override = version.version_id if target != "latest" else ""
        return version.version_id

    def settle_head_after_harness_failure(self) -> str:
        """Give an instructed run that died on the host the head its run end would have.

        Instructed checkpoints do not advance the head; `finish()` selects one at
        run end. A host failure mid-run (the agent's container vanished, so the
        next prompt could not be staged) skips `finish()`, which would leave a
        run with saved checkpoints headless and its verdict REFUSED ("accepted no
        version"). Only the host-side registry is used: the container may be gone, so no
        new tree is captured. Never raises.
        """
        if not self._is_instructed:
            return ""
        with contextlib.suppress(Exception):
            self.take_requested_promotions()
        try:
            return self.default_instructed_head_at_run_end()
        except Exception as exc:                               # noqa: BLE001
            with contextlib.suppress(Exception):
                self.journal.append("head_default_failed", why=repr(exc)[:400],
                                    classification=FailureClass.HARNESS.value)
            return ""

    def default_instructed_head_at_run_end(self) -> str:
        """Select the newest saved checkpoint unless a rollback was selected.

        Explicit promotion remains useful when the best checkpoint is not the
        newest one.  It is not useful as a mandatory piece of bookkeeping that
        can turn a run with materialised checkpoints into ``REFUSED``.  The
        run-end path has already captured the last working tree before this
        method runs.  A concrete explicit promotion is treated as a
        rollback override; otherwise the newest immutable version wins.
        """
        ours = [v for v in self.registry.all()
                if v.run_id == self.identity.run_id]
        if not ours:
            self.journal.append(
                "head_default_unavailable",
                why="the run ended without a materialised checkpoint")
            return ""
        current = self.publisher.head_version()
        explicit_override = getattr(self, "_explicit_head_override", "")
        if explicit_override:
            override = next((v for v in ours
                             if v.version_id == explicit_override),
                            None)
            if override is not None:
                if current != override.version_id:
                    self._publish(override.candidate_id, override, captured=None,
                                  explicit_selection=True)
                self.journal.append(
                    "head_rollback_override_at_run_end",
                    version_id=override.version_id,
                    candidate_id=override.candidate_id,
                    selection_rule="explicit_concrete_version")
                return self.publisher.head_version()
        latest = max(ours, key=lambda v: (v.round_index, v.version_id))
        if current == latest.version_id:
            return current
        try:
            version_id = self.promote_checkpoint(
                latest.version_id,
                note=("run ended without an explicit selection; defaulting "
                      "to the newest materialised checkpoint"))
        except Exception as exc:                           # noqa: BLE001
            self.journal.append(
                "head_default_failed", version_id=latest.version_id,
                candidate_id=latest.candidate_id, why=repr(exc)[:400],
                classification=FailureClass.HARNESS.value)
            return ""
        self.journal.append(
            "head_defaulted_at_run_end", version_id=version_id,
            candidate_id=latest.candidate_id,
            selection_rule="newest_materialised_checkpoint",
            why="no checkpoint from this run had been explicitly promoted")
        return version_id

    def _publish(self, cid: str, version, captured,
                 explicit_selection: bool = False) -> None:
        """Move the accepted head to this version, or record why it stayed.

        The only writer of the head, and the only remaining check is ANCESTRY:
        is the current head's commit an ancestor of this candidate's? That is
        not a gate on quality. It stops the head walking backwards when
        candidates are published out of order, for instance when an end-of-run
        backlog drains in registry order and an older version whose tree is an
        ancestor of the head's arrives last.

        Nothing is reported to the agent. A version's worth is decided by the
        final verdict outside Harbor, and an agent told what that evidence says
        is an agent developing against it.
        """
        # An explicit instructed promotion may intentionally select an older
        # saved fallback in THIS run. Automatic/out-of-order publications keep
        # the ancestry guard. Identity was checked by promote_checkpoint.
        stale = ("" if explicit_selection
                 else self._may_not_take_the_head(version.version_id))
        if stale:
            self.registry.record_outcome(
                version.version_id, status="PUBLISHED_NOT_HEAD",
                qualification={"why_not": stale,
                               "judged": False,
                               "judged_note": (
                                   "nothing in the run judges a version. The "
                                   "final verdict outside Harbor does")},
                journal=self.journal)
            self.journal.append("head_not_advanced",
                                version_id=version.version_id, why=stale)
            if candidate_state(self.journal, cid) is CandidateState.REGISTERED:
                move_candidate(self.journal, cid, CandidateState.REGISTERED,
                               CandidateState.ACCEPTED,
                               head=self.publisher.head_version(),
                               note="kept and visible; the head did not move")
            return
        try:
            head = self.publisher.advance(
                authorization=publish_mod.Authorization.unconditional(
                    version_id=version.version_id,
                    tree_sha=version.tree_sha,
                    candidate_hash=version.tree_sha,
                    commit_sha=version.commit_sha),
                expected=self._head_this_candidate_descends_from(),
                journal=self.journal)
        except CASConflict as exc:
            self.registry.record_outcome(
                version.version_id, status="PUBLISHED_NOT_HEAD",
                qualification={"cas_conflict": str(exc), "judged": False},
                journal=self.journal)
            if candidate_state(self.journal, cid) is CandidateState.REGISTERED:
                move_candidate(self.journal, cid, CandidateState.REGISTERED,
                               CandidateState.ACCEPTED,
                               note=f"head moved underneath: {exc}")
            return
        # THE TRAJECTORY POINT, TAKEN AS THE VERSION IS PUBLISHED.
        #
        # The measurement of a version happens in parallel with its
        # publication, from the AGENT's own AgentX runs, so the run can be
        # plotted as performance against evolve time and API cost.
        #
        # It costs no GPU time: the agent has already measured -- that is why
        # it published -- so this reads artifacts that are on disk already. Not
        # a verdict, never shown to the agent, and never read by
        # `final_validate`, which measures this same version independently on a
        # partition the agent has never seen.
        existing = version.qualification or {}
        cost = existing.get("api_cost") or self._api_cost_for(version)
        point = existing.get("trajectory")
        if not isinstance(point, dict):
            point = self._trajectory_point(version, cost)
        self.registry.record_outcome(
            version.version_id, status="ACCEPTED",
            qualification={"judged": False,
                           "judged_note": (
                               "published unjudged. Correctness and metrics "
                               "come from the final verdict, run outside "
                               "Harbor after the evolve, and are not shown to "
                               "the agent"),
                           "api_cost": cost,
                           "trajectory": point},
            journal=self.journal)
        state = candidate_state(self.journal, cid)
        if state in {CandidateState.REGISTERED, CandidateState.CHECKPOINTED}:
            move_candidate(self.journal, cid, state,
                           CandidateState.ACCEPTED, head=head.version_id)

    def _trajectory_point(self, version, cost: dict) -> dict:
        """This version's point on the evolve curve. Never fatal.

        Harvested from the agent's own aiperf artifacts at publication time --
        see `lifecycle/trajectory.py` for why the agent's numbers are the right
        ones for a curve and the wrong ones for a verdict.

        A failure here returns a body saying so. A trajectory is a nice figure;
        it is not worth failing a publication over, and a run that lost its
        curve still has its verdict.
        """
        from . import trajectory

        try:
            point = trajectory.harvest(
                workspace=self.built.path,
                version_id=version.version_id,
                captured_unix=_unix_of(version.captured_utc
                                       or version.registered_utc) or 0.0,
                # The run's own start, so `evolve_seconds` is time
                # since the run began rather than since the epoch.
                started_unix=_unix_of(self.identity.created_at),
                api_cost=cost,
                round_index=getattr(version, "round_index", None),
                # The task's own frozen denominator AND numerator prices. A
                # partial contract would put the curve and verdict on
                # different price conventions.
                usd_per_hour=_usd_per_hour(self.plan.task_dir),
                objective_detail=_objective_detail(self.plan.task_dir))
            written = trajectory.append(self.identity.root, point)
            self.journal.append(
                "trajectory_point", version=version.version_id,
                file=str(written.relative_to(self.identity.root)),
                evolve_seconds=point.get("evolve_seconds"),
                agentx_runs_found=point.get("agentx_runs_found"))
            return point
        except Exception as exc:                               # noqa: BLE001
            self.journal.append(
                "trajectory_point_failed", version=version.version_id,
                why=f"{type(exc).__name__}: {exc}"[:300],
                because=("a trajectory is a figure, not a verdict. The "
                         "version is published and the final verdict is "
                         "unaffected"))
            return {"available": False,
                    "why": f"{type(exc).__name__}: {exc}"[:300]}

    def _api_cost_for(self, version) -> dict:
        """What producing this version cost, attributed from the agent's stream.

        The agent records its own `total_cost_usd`, so that is what the total
        comes from; this apportions it across the versions the run produced, by
        the usage recorded between one capture and the next. A version is
        charged for the work that made it.

        Never fatal. A missing or unreadable stream returns a body saying so --
        absent, not zero, because a run whose stream we could not read did not
        cost nothing, and a zero here would go into a published record.
        """
        from . import api_cost

        try:
            stream = self.identity.root / "agent.stream.jsonl"
            # DEDUPED BY VERSION ID, and that is not cosmetic.
            #
            # `materialize` has already registered this version by the time
            # `_publish` runs, so `registry.all()` contains it already.
            # `attribute` differences consecutive captures, so a second entry
            # for the same version would difference against the first and the
            # version's own slice would come out as exactly zero.
            captured: dict[str, float] = {}
            for v in list(self.registry.all()) + [version]:
                when = _unix_of(v.captured_utc or v.registered_utc)
                if when:
                    captured[v.version_id] = when
            attributed = api_cost.attribute(
                stream, sorted(captured.items(), key=lambda kv: kv[1]))
            if not attributed.get("available"):
                return attributed
            mine = (attributed.get("versions") or {}).get(version.version_id)
            return {
                "cost_usd": (mine or {}).get("cost_usd"),
                "usage": (mine or {}).get("usage"),
                "model": attributed.get("model"),
                "run_total_cost_usd": attributed.get("run_total_cost_usd"),
                "run_total_from": attributed.get("run_total_from"),
                "split_is_approximate": attributed.get("split_is_approximate"),
                "split_caveat": attributed.get("split_caveat"),
                "reconciliation": attributed.get("reconciliation"),
            }
        except Exception as exc:                    # noqa: BLE001
            # A cost figure is worth having and is not worth losing a
            # publication over.
            return {"available": False,
                    "unavailable_because": f"{type(exc).__name__}: {exc}"}

    def _may_not_take_the_head(self, version_id: str) -> str:
        """Why this version may not become the accepted head, or "" if it may.

        The question is ancestry and it is asked of git: is the head's commit
        an ancestor of this candidate's? That holds within a run (each capture
        is parented on the last) and across runs (a run seeded from a head
        parents its first capture on that head's commit), for every task, and
        it does not depend on ordering -- which is only meaningful inside one
        run and would refuse every legitimate continuation. Without it, an
        older version whose tree is an ANCESTOR of the head's could take the
        head and walk it backwards through the agent's work.

        Unprovable is not permitted. A version with no commit cannot show
        descent, and this returns a refusal saying so rather than guessing in
        either direction.
        """
        head = self.publisher.head()
        if head is None:
            return ""
        versions = {v.version_id: v for v in self.registry.all()}
        mine = versions.get(version_id)
        if mine is None:
            return f"{version_id} is not in this lineage's registry"
        if mine.version_id == head.version_id:
            return ""
        descends = self.store.descends_from(head.commit_sha, mine.commit_sha)
        if descends is True:
            return ""
        if descends is False:
            return (f"the accepted head {head.version_id} is not an ancestor of "
                    f"this tree, so this version does not descend from what is "
                    f"already accepted. It is measured and visible; it does not "
                    f"move the head")
        return (f"descent from the accepted head {head.version_id} could not be "
                f"established (one of the two has no commit recorded). A head "
                f"is not advanced on unproven provenance")

    def _head_this_candidate_descends_from(self) -> str:
        """The head a candidate of THIS run may land on.

        Every candidate in a run comes out of the same workspace, edited
        forward, so each descends from the one before it. The head therefore
        advances along that chain, and the right `expected` is the head as it
        stands now -- not the head as it stood when this candidate was
        captured, which is a different thing the moment two candidates are in
        flight at once.

        Passing the capture-time head to the CAS would be wrong: candidate 2,
        captured while candidate 1 was still in flight, would carry predecessor
        "" and conflict against the v1 that candidate 1 had just become, so
        every run would publish exactly one version and then refuse.

        What must still refuse is a head this run did NOT produce: another run
        publishing into the same lineage means this candidate does not descend
        from what is there, and no amount of ordering makes it so. Returning
        the stale value makes the publisher raise, which is the intent.
        """
        head = self.publisher.head_version()
        if not head:
            return ""
        ours = {v.version_id for v in self.registry.all()
                if v.run_id == self.identity.run_id}
        if head in ours or head == self._head_at_start:
            return head
        self.journal.append(
            "foreign_head_detected", head=head, run_id=self.identity.run_id,
            why=("the accepted head was moved by a run other than this one; a "
                 "candidate from this workspace does not descend from it"))
        return "\u0000not-ours"

    # ------------------------------------------------------------------ finish
    def finish(self, agent_run: agent_mod.AgentRun) -> dict:
        if self._is_instructed:
            # Honour requests that landed between the last tick and exit, then
            # freeze the actual last working tree, so the final verdict grades
            # the final implementation rather than an hours-old checkpoint
            # while the last work stays in the soon-to-be-destroyed workspace.
            self.take_requested_boundaries()
            self.take_evidence_boundaries()
            try:
                candidate_id = self.boundary(
                    note=("timebox ended; automatically snapshotting the last "
                          "working tree for final validation"),
                    advance_head=True)
                self.journal.append(
                    "final_working_tree_snapshot_requested",
                    candidate_id=candidate_id,
                    selection_rule=("last_working_tree_unless_explicit_"
                                    "rollback_override"))
            except Exception as exc:                       # noqa: BLE001
                self.journal.append(
                    "final_working_tree_snapshot_failed", why=repr(exc)[:400],
                    classification=FailureClass.HARNESS.value)
            self._ending.set()
        else:
            # Standard tasks take a final safety capture.
            try:
                self.boundary(note="the run ended; capturing the final state")
                self._ending.set()
            except Exception as exc:                       # noqa: BLE001
                self.journal.append("boundary_failed", why=repr(exc)[:400],
                                    classification=FailureClass.HARNESS.value)
        for t in list(self._threads):
            t.join(timeout=self.plan.after_capture_timeout + 60)
        if self._is_instructed:
            # A queued `promote latest` may have been waiting for the final
            # checkpoint thread to materialise its version.
            self.take_requested_promotions()
            self.default_instructed_head_at_run_end()
        # NOTHING TO WAIT FOR AT RUN END.
        #
        # Publication is synchronous and takes no device, so a version is
        # published the moment it is registered and there is no backlog to
        # drain past `deadline_utc`.
        end = (RunState.STOPPED_EARLY if agent_run.stopped_early
               else RunState.COMPLETED)
        return self.preserve(
            end, remaining_seconds=agent_mod.remaining_seconds(
                self.identity,
                paused_s=billing_mod.paused_seconds(self.journal)))

    def preserve(self, end: RunState, **why) -> dict:
        """Close the run and write down everything it produced. ONE path.

        An operator's Ctrl-C or SIGTERM takes this same path, so a cancelled
        run keeps its trajectory, its figure and the final capture of whatever
        the agent built since its last request.

        A run's evidence is not a reward for ending the approved way. Whatever
        state it ends in, what it produced is written down.
        """
        move_run(self.journal, self.identity.run_id, run_state(self.journal),
                 end, **why)
        # Memory is taken back FIRST, before anything else can fail. Whatever
        # state the run ends in -- an operator's Ctrl-C included -- the notes
        # the agent wrote about what it tried are part of what the run
        # produced, and losing them makes the next session re-derive the same
        # dead ends at full device cost.
        with contextlib.suppress(Exception):
            self.persist_memory()
        # The trajectory is collected before the summary is written, so a
        # reader has the run's whole story and the run's own account of it side
        # by side, and both name the same journal.
        try:
            report_mod.collect(identity=self.identity, journal=self.journal,
                                   agent_note=self.plan.agent_model)
        except Exception as exc:                           # noqa: BLE001
            self.journal.append("trajectory_failed", why=repr(exc)[:400],
                                classification=FailureClass.HARNESS.value)
        self._deliver_trajectory_to_versions()
        self._deliver_reasoning_to_versions()
        self._audit_models()
        # PRIVATE FINAL VALIDATION -- the closing stage of the evolve, and the
        # last thing that needs the card.
        #
        # It runs HERE, before `release()` gives the device back, and after the
        # agent and its container are gone. The coding agent never invokes it
        # and never sees its result: private final outcomes stay out of the
        # optimisation loop, so this writes to the operator-facing store and
        # nothing reaches the workspace.
        #
        # A refusal is an OUTCOME, not an error. A run whose candidates were all
        # rejected has no accepted head and therefore no final candidate, and
        # recording that is the honest end state -- inventing one would put a
        # candidate the lineage never accepted in front of the held-out
        # partition.
        #
        # ORDER MATTERS. `validate_evolve` resolves the lineage from this run's
        # `SUMMARY.json`, so the summary is written before it is called.
        self._bring_the_tree_home("the run is ending")
        summary = self.summary()
        path = self.identity.root / "SUMMARY.json"
        path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
        self._final_validation()
        self._record_in_the_ledger()
        # Rewritten so the run's own record carries what final validation
        # found; the first write exists for the validator to read, this one for
        # whoever reads the run afterwards. Guarded because the summary on disk
        # is already correct-as-of-the-first-write, and losing the outcome line
        # is better than losing the file.
        try:
            summary = self.summary()
            path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
        except Exception as exc:                          # noqa: BLE001
            self.journal.append("summary_not_rewritten", why=describe(exc),
                                because=("the summary written before final "
                                         "validation stands; only the final "
                                         "outcome line is missing from it"))
        return summary

    def _final_validation(self) -> None:
        """Take this run's accepted head to private final validation.

        Delegates to `final_validate/tools/validate_evolve.py`, which drives
        the final verdict -- the one measurement path. Nothing here
        selects a candidate, measures anything or decides a verdict; what it
        does is decide WHEN, which is once, at the end, while the card is
        still held.

        Run as a SUBPROCESS rather than imported. The final validator is
        deliberately independent of the grading path, and the driver already
        carries `serving_evolve` in its process; importing it here would put
        both on one interpreter and make the boundary a matter
        of discipline rather than of construction.

        Never fatal. The run's evidence is already written by this point, and
        losing a summary because a final validation could not start would
        destroy the record of everything that did happen.
        """
        if self.identity is None or self.plan.dry_run:
            return
        # ONCE PER RUN, AND THE SECOND CALLER WAITS FOR THE FIRST.
        #
        # An operator's SIGTERM arriving while the end-of-run validation runs
        # raises KeyboardInterrupt in the DRIVER; the child keeps running --
        # the signal goes to the driver's pid, not its group -- and the
        # preserving path then calls this again. A second validation would
        # find the card held by its own predecessor, refuse, and write that
        # refusal into the lineage's `FINAL.json` as `status: DID_NOT_START`
        # over the first validation's real verdict.
        if self._final_validation_child is not None:
            running = self._final_validation_child
            if running.poll() is None:
                self.journal.append(
                    "final_validation_awaited",
                    why=("a final validation for this run is already running; "
                         "waiting for it rather than starting a second one, "
                         "which would refuse the card it is holding and "
                         "record that refusal as the head's verdict"))
                try:
                    running.wait(timeout=final_validation_timeout(
                        self.plan.task_dir))
                except subprocess.TimeoutExpired:
                    self.journal.append(
                        "final_validation_failed",
                        why="the in-flight final validation did not finish",
                        classification=FailureClass.HARNESS.value)
            else:
                self.journal.append(
                    "final_validation_skipped",
                    why="this run has already been finally validated once")
            return
        script = (self.plan.repo / "serverbench" / "harness" / "final_validate"
                  / "tools" / "validate_evolve.py")
        if not script.is_file():
            self.journal.append(
                "final_validation_skipped",
                why=f"{script} is not on this host")
            return
        # THE RUN'S OWN CARD IS WHAT THE VERDICT MEASURES ON, and it has to be
        # handed in rather than fenced off.
        #
        # The verdict runs here precisely BECAUSE the device is still held;
        # denying it would leave the verdict polling a device lease for a card
        # this run is holding against it.
        #
        # The agent's container has to go first. It is still up at this point
        # and its processes are FOREIGN to the verdict, which refuses a
        # contended card -- correctly. Stopped by exact recorded name, never by
        # pattern: `docker rm $(docker ps -q)` takes down somebody else's work.
        # The holders go with the container, and for the same reason: the
        # verdict refuses a card carrying a compute process it does not own,
        # and from its point of view a holder is one.
        self._release_holders("before final validation, so the verdict finds "
                              "the card clear")
        if self.container and backend.measures_from_inside(self.container):
            # THE VERDICT DOES NOT RUN IN THIS ONE. It stands up a FRESH
            # sandbox, with the evaluator owned by root and stripped of write
            # before the candidate starts -- a verdict the measured party can
            # edit is not a verdict, and this sandbox is the agent's, writable
            # by the agent, for the whole run.
            #
            # So this one goes now. The tree came home at the boundary above,
            # so nothing here is lost; held open it would bill the task's full
            # GPU allocation a second time for the length of the replay.
            try:
                how = backend.remove(self.container)
                self.journal.append(
                    "container_released", name=self.container, backend="modal",
                    how=how,
                    why=("the verdict stands up its own read-only sandbox, so "
                         "holding the agent's would bill a second GPU pair "
                         "for nothing. The tree was pulled back before this"))
            except (OSError, subprocess.SubprocessError) as exc:
                self.journal.append(
                    "container_not_released", name=self.container,
                    why=f"{type(exc).__name__}: {exc}"[:300],
                    because=("a sandbox left standing holds GPUs until its "
                             "own timeout expires"))
        elif self.container:
            try:
                backend.remove(self.container)
                self.journal.append(
                    "container_released", name=self.container, how="stopped",
                    why=("before final validation, so the card this run holds "
                         "carries no foreign compute process when the verdict "
                         "asks for it"))
            except (OSError, subprocess.SubprocessError) as exc:
                self.journal.append(
                    "container_not_released", name=self.container,
                    why=f"{type(exc).__name__}: {exc}"[:300],
                    because=("the verdict will refuse the card as contended "
                             "rather than measure against a neighbour"))
        argv = [sys.executable, str(script), "--run", str(self.identity.root),
                "--image", self.plan.image,
                "--model-dir", str(self.plan.model_dir),
                "--env-input", str(self.plan.env_input),
                "--gpu", ",".join(str(g) for g in self.plan.gpus)]
        self.journal.append("final_validation_started",
                            script=str(script.relative_to(self.plan.repo)))
        try:
            # `Popen`, not `run`, so an interrupt arriving mid-validation
            # leaves a handle the preserving path can WAIT on. See the guard
            # at the top of this method.
            self._final_validation_child = subprocess.Popen(  # noqa: S603
                argv, cwd=str(self.plan.repo), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = self._final_validation_child.communicate(
                timeout=final_validation_timeout(self.plan.task_dir))
            got = subprocess.CompletedProcess(
                argv, self._final_validation_child.returncode, out, err)
        except (OSError, subprocess.SubprocessError) as exc:
            self.journal.append("final_validation_failed",
                                why=f"{type(exc).__name__}: {exc}"[:400],
                                classification=FailureClass.HARNESS.value)
            return
        tail = ((got.stdout or "") + (got.stderr or "")).strip()
        refused = tail.startswith("REFUSED") or "\nREFUSED" in tail
        status, evidence = "", ""
        for line in reversed(tail.splitlines()):
            if line.strip().startswith("VALID_") or line.strip().startswith(
                    "INVALID_") or line.strip().startswith("INCOMPLETE"):
                parts = line.split("evidence at")
                status = parts[0].strip()
                evidence = parts[1].strip() if len(parts) > 1 else ""
                break
        self.journal.append(
            "final_validation",
            outcome=status or ("REFUSED" if refused else "NO_RESULT"),
            evidence=evidence,
            returncode=got.returncode,
            # The refusal text is the finding when there is no head to validate.
            because=tail[-800:],
            note=("the coding agent neither invoked this nor sees its result; "
                  "private final outcomes are the operator's record and must "
                  "not become another optimisation feedback cycle"))

    def _audit_models(self) -> None:
        """Record which models and provider hosts the agent used (the brief
        forbids any model but the session's own), for the operator."""
        from . import model_audit
        try:
            result = model_audit.write(self.identity.root, self.plan.agent_model)
        except Exception as exc:                           # noqa: BLE001
            self.journal.append("model_audit_failed", why=repr(exc)[:300],
                                classification=FailureClass.HARNESS.value)
            return
        self.journal.append(
            "model_audit", holds=result["holds"],
            other_models=result["other_models"],
            unexpected_hosts=result["unexpected_hosts"],
            direct_api_commands=result["direct_api_command_count"])

    def _deliver_reasoning_to_versions(self) -> None:
        """`DECISIONS.md` beside every version this run produced.

        The agent's own words at the moment of each change, paired with what it
        changed immediately afterwards, extracted from the event stream by
        `serving_evolve.timeline` -- not written by hand and not summarised, so
        a published version carries an account of why its code was written,
        not just the code and the numbers.
        """
        try:
            from . import decisions as decisions_mod
            entries = decisions_mod.decisions(
                self.identity.root / "agent.stream.jsonl")
            body = decisions_mod.render_decisions(entries)
        except Exception as exc:                           # noqa: BLE001
            self.journal.append("decisions_failed", why=repr(exc)[:300],
                                classification=FailureClass.HARNESS.value)
            return
        if not body.strip():
            return
        for version in self.registry.all():
            if version.run_id != self.identity.run_id:
                continue
            with contextlib.suppress(Exception):
                (Path(version.directory) / "DECISIONS.md").write_text(
                    body, encoding="utf-8")
        self.journal.append("decisions_delivered", entries=len(entries))

    def _deliver_trajectory_to_versions(self) -> None:
        """Every version this run produced gets the run's trajectory.

        A version directory that holds the source and the numbers but not the
        story of how it was arrived at is half an artifact; the trajectory is
        what lets a later reader ask why, and it is small.
        """
        import shutil
        source = self.identity.root / "trajectory.atif.json"
        if not source.is_file():
            return
        # And a readable rendering beside it. The JSON is large and nested;
        # nobody opens it to find out what a run did.
        readable = ""
        try:
            readable = report_mod.as_markdown(
                json.loads(source.read_text("utf-8")))
        except Exception as exc:                           # noqa: BLE001
            self.journal.append("trajectory_render_failed", why=repr(exc)[:300],
                                classification=FailureClass.HARNESS.value)
        for version in self.registry.all():
            if version.run_id != self.identity.run_id:
                continue
            with contextlib.suppress(Exception):
                shutil.copy2(source, Path(version.directory) /
                             "trajectory.atif.json")
                if readable:
                    (Path(version.directory) / "TRAJECTORY.md").write_text(
                        readable, encoding="utf-8")

    def summary(self) -> dict:
        head = self.publisher.head()
        versions = self.registry.all()
        return {
            "kind": "run_summary", "schema_version": 1,
            "run_id": self.identity.run_id, "task_id": self.identity.task_id,
            "lineage_id": self.identity.lineage_id,
            "control_epoch": self.identity.control_epoch,
            "workspace_id": self.identity.workspace_id,
            "namespace": self.identity.namespace,
            # What "done" meant for this run. `control_epoch` above says what
            # the TASK was frozen as; this says what the RUN was asked for, so
            # two runs of the same task under different bars read differently.
            "stop": self.plan.stop,
            "state": run_state(self.journal).value,
            "accepted_head": head.version_id if head else "",
            "versions": [{"version_id": v.version_id, "status": v.status,
                          "candidate_id": v.candidate_id,
                          "predecessor": v.predecessor,
                          "tree_sha": v.tree_sha} for v in versions],
            "candidates": {k: v.value for k, v in candidates(self.journal).items()},
            "trajectory": str(self.identity.root / "trajectory.atif.json"),
            "publication": self._publication_health(),
            "note": ("versions are unjudged checkpoints during development; "
                     "correctness and performance come from final validation"),
        }

    def _head_concerns(self, points: list[dict], accepted: set[str]) -> list[str]:
        """What the HEAD's own numbers say, against the task's frozen bar.

        The head is the tree the final verdict will measure, so it is the one
        version whose own development numbers are worth reading in a summary.
        Two things about it are reported:

        A THIN YES. A head judged correct by its own benchmark on a small
        sample (`well_evidenced: false`) cannot see an intermittent fault the
        verdict outside may find.

        A HEAD THAT MISSES ITS OWN GATES. The head's own export compared
        against the task's frozen bar.

        Neither is a harness fault and neither fails the run: publication is
        unconditional by design, and an agent is allowed to publish a version
        that is not there yet. But "healthy, no concerns" is the wrong sentence
        to put in front of an operator about it, and it is the line they read
        first.

        NOT AUTHORITATIVE, and the wording says so. These are the agent's own
        measurements on prompts it may read. `final_validate` decides.
        """
        if not points or not accepted:
            return []
        head = max(accepted, key=_version_order)
        point = next((p for p in reversed(points)
                      if p.get("version_id") == head), None)
        if point is None:
            return []

        concerns: list[str] = []
        correctness = point.get("correctness") or {}
        if (correctness.get("holds") is True
                and correctness.get("well_evidenced") is False):
            scored = verdict_seconds(self.plan.task_dir)
            concerns.append(
                f"the head {head} passed correctness on its OWN benchmark but "
                f"the evidence is weak: {correctness.get('caveat', '')}. A "
                f"sample this small cannot see an intermittent fault"
                + (f", and the verdict outside runs {scored:.0f}s"
                   if scored else ""))

        metrics = point.get("metrics") or {}
        ttft = metrics.get("ttft_ms") or {}
        try:
            levels = _service_levels().read(self._slo(), task_id=head)
        except Exception:                                      # noqa: BLE001
            return concerns
        checks = _service_levels().check(
            levels,
            ttft_by_percentile={row["percentile"]:
                                ttft.get(f"p{row['percentile']}")
                                for row in levels},
            output_tps=metrics.get("output_tokens_per_s"),
        )
        if any(row["measured"] for row in checks) \
                and not _service_levels().holds(checks):
            concerns.append(
                f"the head {head}'s OWN last benchmark misses the frozen bar: "
                f"{_service_levels().why(checks)}. Not authoritative -- the "
                f"verdict measures outside, on a card it leases -- but the "
                f"agent published this as its best work")
        return concerns

    def _slo(self) -> dict:
        """The task's frozen service levels, in the shape its readers expect."""
        try:
            import yaml
            body = yaml.safe_load(
                (Path(self.plan.task_dir) / "contract.yaml").read_text("utf-8"))
            return {"service_levels": (body or {}).get("service_levels") or []}
        except Exception:                                      # noqa: BLE001
            return {}

    def _publication_health(self) -> dict:
        """Whether this run actually published, and if not, exactly why.

        A run that produces no version must not do so QUIETLY: an empty
        accepted head alone reads like an agent that improved nothing.

        So the summary carries the arithmetic: boundaries taken, versions made
        visible, versions accepted, and a plain sentence naming the first step
        that lost them. `versions_measured` is always 0: nothing in a run
        measures a version; the final verdict does.
        """
        journal = self.journal
        captured = journal.events("candidate_captured")
        registered = journal.events("version_registered")

        # Distinct versions that hold the head, not the number of times it
        # moved: an ascent of eleven steps is eleven advances and may be fewer
        # versions if one is superseded.
        accepted = {e["version_id"] for e in journal.events("head_advanced")}

        concerns = []
        if not captured:
            concerns.append("no boundary produced a candidate: the agent never "
                            "ran `./version`, or every request it made found "
                            "the workspace unchanged")
        if captured and len(registered) < len(captured):
            concerns.append(
                f"{len(captured) - len(registered)} candidate(s) were captured "
                f"but never became a version -- see after_capture_failed "
                f"and boundary_failed")

        # DID THE AGENT EVER RUN ITS OWN SERVER?
        #
        # Publishing is unconditional and unjudged, so a run can register
        # versions and advance the head while every one of those versions is
        # a tree that has never been started.
        #
        # None of that is a harness fault and none of it should fail a run.
        # But "healthy, no concerns" is the wrong thing to say about it, and
        # it is the line an operator reads first. A run that never measured
        # itself is a fact about the run, and the summary should carry it.
        workspace_path = self.built.path if self.built else None
        benchmarks = _agent_benchmarks(workspace_path)
        corrections = _agent_correctness_runs(workspace_path)
        replay_checks = _agent_replay_verifications(workspace_path)
        if registered and not benchmarks and not corrections and not replay_checks:
            concerns.append(
                f"the agent published {len(registered)} version(s) and never "
                f"ran the evaluator at all: neither a performance export nor "
                f"a correctness probe exists anywhere in the workspace. "
                f"Nothing here was measured by anyone before the run ended, "
                f"so the versions are unvalidated trees rather than "
                f"known-working ones")
        elif registered and not benchmarks:
            # SAY WHICH HALF IS MISSING. An agent that checked correctness and
            # never timed anything has done something real, and calling that
            # "never benchmarked its own server" both misreports the run and
            # buries the evidence that exists.
            concerns.append(
                f"the agent published {len(registered)} version(s) and "
                f"ran {replay_checks} lightweight replay verification(s) and "
                f"{corrections} endpoint check(s) "
                f"(not the final correctness gate), but "
                f"never completed a full-window performance replay. The "
                f"versions have cheap preflight evidence and are not known to "
                f"cross a hard gate, so no latency or dollars-per-dollar claim "
                f"in this run rests on a complete measurement")
        # DID THE AGENT'S OWN MEASUREMENTS EVER SERVE THE WORK CORRECTLY?
        #
        # Benchmarking is not the same as benchmarking something correct: an
        # agent's own export can show failing requests that nothing read.
        #
        # Read from the trajectory points, which read it from the agent's own
        # aiperf exports -- the same fields the verdict's rule reads, so the
        # two cannot disagree about what correctness means. Costs no GPU time
        # and adds no second measurement path.
        #
        # `holds: None` is UNKNOWN and counted separately: an agent that
        # published without benchmarking is already reported above, and
        # calling that a correctness failure would be a second alarm for one
        # fact.
        points = [e.get("trajectory") or {}
                  for e in journal.events("version_outcome")
                  if isinstance(e.get("trajectory"), dict)]
        judged = [p["correctness"] for p in points
                  if isinstance(p.get("correctness"), dict)
                  and p["correctness"].get("holds") is not None]
        failing = [c for c in judged if c.get("holds") is False]
        if judged and len(failing) == len(judged):
            first = failing[0]
            concerns.append(
                f"every version whose own benchmark could be judged FAILED "
                f"correctness ({len(failing)} of {len(judged)}): "
                f"{first.get('why', '')[:160]} -- the verdict checks this "
                f"first, so tokens_per_usd is unscorable until it holds")
        elif failing:
            concerns.append(
                f"{len(failing)} of {len(judged)} version(s) whose own "
                f"benchmark could be judged FAILED correctness. The head is "
                f"what the verdict measures, so check whether it is one of "
                f"them")

        concerns.extend(self._head_concerns(points, accepted))

        if registered and not accepted:
            concerns.append(
                f"{len(registered)} version(s) were registered and the accepted "
                f"head never moved. Publication is unconditional, so the "
                f"only thing that withholds the head is ANCESTRY -- see "
                f"head_not_advanced for which tree did not descend from which")
        for conflict in journal.events("head_cas_conflict"):
            concerns.append(f"a compare-and-swap was refused: expected "
                            f"{conflict['expected'] or 'none'}, found "
                            f"{conflict['found'] or 'none'}")

        return {
            "boundaries": len(captured),
            "versions_registered": len(registered),
            "versions_measured": 0,
            "versions_accepted": len(accepted),
            "accepted_versions": sorted(accepted),
            "healthy": bool(registered) and not concerns,
            "concerns": concerns or ["none"],
        }

    def release(self) -> None:
        # HOLDERS FIRST, AND AGAIN.
        #
        # `_release_holders` already runs before final validation, which is the
        # case that matters: the verdict refuses a card carrying a compute
        # process it does not own, and a holder is one from its point of view.
        # This second call covers every path that never reaches there -- a run
        # that fails in setup, an operator's Ctrl-C, a HarnessSetupFailed --
        # because a holder left behind is stale GPU memory.
        #
        # Idempotent: a pid already gone is journalled as "already gone".
        with contextlib.suppress(Exception):
            self._release_holders("run teardown")
        if self.own is not None:
            self.own.release_all()
        if self.identity is not None:
            self._reclaim_disk()

    def _reclaim_disk(self) -> None:
        """Give the run's disposable bulk back, and shred its staged token.

        A finished run holds gigabytes it will never need again -- a `.venv` of
        torch and transformers, rebuildable in minutes from the frozen
        wheelhouse that was the only place the agent could install from.
        Accumulated across runs, that fills the filesystem other measurements
        need.

        What is NEVER removed is the evidence -- journal, summary, trajectory,
        agent logs, and the agent's own source tree. `reclaim.reclaimable`
        refuses any run whose driver is alive or whose state is not terminal,
        so this cannot take a workspace out from under a running agent.

        It also shreds `secrets/`. `credentials.py` stages a live OAuth token
        there so a host refresh can reach the container, and a finished run has
        no use for one.

        Never fatal: a run that cannot free its disk still ends properly.
        """
        if self.identity is None:
            return
        try:
            from lifecycle import reclaim as reclaim_mod

            got = reclaim_mod.reclaim(self.identity.root)
        except Exception as exc:                               # noqa: BLE001
            with contextlib.suppress(Exception):
                self.journal.append("disk_not_reclaimed",
                                    why=repr(exc)[:200])
            return
        with contextlib.suppress(Exception):
            if got.get("skipped"):
                self.journal.append("disk_not_reclaimed", why=got["skipped"])
            elif got.get("bytes"):
                self.journal.append(
                    "disk_reclaimed", bytes=got["bytes"],
                    gigabytes=round(got["bytes"] / 1e9, 2),
                    paths=len(got.get("paths") or []),
                    note=("the run's .venv, caches and staged credentials. "
                          "Every piece of evidence -- journal, summary, "
                          "trajectory, agent logs, agent source -- is kept"))


def _unix_of(stamp: str) -> float:
    """A `...Z` timestamp as unix seconds, or 0.0 if it is not one.

    0.0 rather than an exception: a version with an unparseable timestamp
    should not stop a publication, and `attribute` skips a zero.
    """
    import datetime as _dt

    try:
        return _dt.datetime.strptime(
            str(stamp), "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=_dt.timezone.utc).timestamp()
    except (TypeError, ValueError):
        return 0.0


def devices_declared(task_dir: Path) -> int:
    """How many devices this task is billed for, from `contract.yaml`.

    One number, stated once, because the device count is the denominator of
    the objective. How a candidate organises the cards it is given is its own
    choice rather than the task's.

    yaml is imported here rather than at module scope because the harness core
    must import inside a bare image; `tests/repo` enforces that.
    """
    import yaml

    body = yaml.safe_load(
        (Path(task_dir) / "contract.yaml").read_text("utf-8")) or {}
    return int(body["devices"])


# ------------------------------------------------------------------------ main
def _install_stop_handlers() -> None:
    """Route catchable terminal-loss signals through evidence preservation."""

    def _on_stop(signum, frame):                              # noqa: ARG001
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, _on_stop)
    # A launcher that deliberately ignored hangups (`nohup` without `setsid`)
    # asked for the run to outlive its terminal. Installing a handler would
    # turn that terminal closing into a cancel, so an inherited SIG_IGN stays.
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        signal.signal(signal.SIGHUP, _on_stop)


def _image_identity(image: str) -> dict:
    """What the run actually ran in, beyond the name it asked for.

    A TAG IS NOT AN IDENTITY. An image rebuilt under the same tag between two
    runs would leave both recording the same string while running different
    content.

    So: the image ID and its RepoDigest, which are the content. Best effort --
    a host without a docker CLI still gets a run, and a missing digest is
    recorded as missing rather than as an error.
    """
    import subprocess

    try:
        done = subprocess.run(
            ["docker", "image", "inspect", image, "--format",
             "{{.Id}}\t{{join .RepoDigests \",\"}}\t{{.Created}}"],
            capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"image_digest": "", "image_unresolved": str(exc)[:200]}
    if done.returncode != 0:
        return {"image_digest": "",
                "image_unresolved": (done.stderr or "").strip()[:200]}
    parts = (done.stdout or "").strip().split("\t")
    # RepoDigests is empty for an image never pushed or pulled, which is every
    # locally built one. The image ID is still the content hash, so it is the
    # field that actually answers "was this the same image".
    return {"image_digest": parts[0] if parts else "",
            "image_repo_digest": parts[1] if len(parts) > 1 else "",
            "image_created": parts[2] if len(parts) > 2 else ""}


#: Written into a lineage while a run is extending it, so a second run cannot
#: start extending the same one.
LINEAGE_LOCK = ".live-run.json"


def lineage_live_writer(lineage_dir: Path) -> dict | None:
    """The run currently extending this lineage, if one still exists.

    A lock whose process is gone is not a lock: a driver killed mid-run would
    otherwise fence the lineage off forever. So liveness is the pid, not the
    file.
    """
    path = Path(lineage_dir) / LINEAGE_LOCK
    try:
        body = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    pid = body.get("pid")
    if not isinstance(pid, int):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        # Alive and owned by someone else. Still alive.
        return body
    return body


def claim_lineage(lineage_dir: Path, run_id: str) -> None:
    """Record that THIS process is the one extending this lineage."""
    path = Path(lineage_dir) / LINEAGE_LOCK
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"schema": "serverbench.lineage_live_run/1", "run_id": run_id,
             "pid": os.getpid(),
             "since_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime())}, indent=1) + "\n",
            "utf-8")
    except OSError:
        # A lineage that cannot be locked is still publishable; the guard is a
        # safety net, not a precondition for doing the work.
        pass


def release_lineage(lineage_dir: Path) -> None:
    with contextlib.suppress(OSError):
        (Path(lineage_dir) / LINEAGE_LOCK).unlink(missing_ok=True)


def lineage_of(seed: Path | None, lineage_root: Path) -> str:
    """The lineage a seed path is inside, or "" for a fresh one.

    A run started from a prototype or workspace extends THE LINEAGE THAT PATH
    IS INSIDE. Deriving it from `--seed` rather than requiring a flag keeps
    that automatic: seeding from `server_output/<lineage>/<version>/`
    continues that lineage, and seeding from the starter opens a new one.
    """
    if seed is None:
        return ""
    try:
        relative = Path(seed).resolve().relative_to(Path(lineage_root).resolve())
    except (ValueError, OSError):
        return ""            # the starter, or anywhere outside the lineage tree
    return relative.parts[0] if relative.parts else ""


#: A skill bundle in this checkout, or None when it is not on disk.
#:
#: Derived from the repository rather than named absolutely, because
#: `serverbench/skills/ncu-report-skill` is a SUBMODULE: a fresh clone has the
#: directory and no files in it until `git submodule update --init`, and
#: `materialize`'s `deliver_skill` treats a missing path as nothing to deliver.
#: Returning the path of an empty directory is harmless for the same reason.
def _skill_default(name: str) -> Path | None:
    path = Path(__file__).resolve().parents[2] / "skills" / name
    return path if path.is_dir() else None


def _guidance_defaults(task_dir: Path) -> dict[str, Path | None]:
    """Reference inputs for the task's named prompt treatment.

    `standard` keeps the reduced live bundles; its kernel skill carries the
    kernel knowledge corpus under `knowledge/`.
    `instructed` gets a separate, full guidance bundle plus the L40-specific
    profiler/kernel updates.  Selection from the full task id is
    what keeps two concurrent variants in different workspace manifests.
    """
    serverbench = Path(__file__).resolve().parents[2]
    instructed = "_instructed_" in Path(task_dir).name.lower()
    skills = serverbench / ("skills_instructed" if instructed else "skills")

    def present(path: Path) -> Path | None:
        return path if path.is_dir() else None

    return {
        "skill_dir": present(
            skills / ("serving-system-evolution-instructed"
                      if instructed else "serving-system-evolution")),
        "kernel_skill_dir": present(skills / "kernel-evolution"),
        "kernelwiki_skill_dir": (
            present(skills / "KernelWiki") if instructed else None),
        "kernelwiki_overlay_dir": (
            present(skills / "_overlays" / "KernelWiki")
            if instructed else None),
        "ncu_skill_dir": present(skills / "ncu-report-skill"),
        # The instructed team: the two subagent definitions `taskctl render`
        # writes into the task, and the hook that re-states their workflow.
        "agents_dir": (present(Path(task_dir) / "agents")
                       if instructed else None),
        "claude_project_dir": (
            present(serverbench.parent / "serving_tasks" / "_manage"
                    / "instructed_claude") if instructed else None),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", type=Path, required=True)
    ap.add_argument("--gpu", required=True,
                    help="device index, or a comma-separated list for a task "
                         "whose contract declares more than one device")
    ap.add_argument("--hours", type=float, required=True)
    ap.add_argument("--image", default="serverbench/evolve-agent:cu128")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--env-input", type=Path, required=True)
    ap.add_argument("--lineage-root", type=Path, required=True)
    ap.add_argument("--seed", type=Path, default=None)
    ap.add_argument("--lineage-id", default="",
                    help="continue this lineage instead of opening a new one; "
                         "the run receives its MEMORY and publishes into it. "
                         "Derived from --seed when the seed is inside a lineage.")
    ap.add_argument("--agent-cli", choices=("codex", "claude"),
                    default="claude")
    ap.add_argument("--agent-model", default=None)
    ap.add_argument("--agent-effort", default=None)
    ap.add_argument("--credentials", type=Path, default=None,
                    help="Explicit agent login file; defaults to the provider's host login.")
    ap.add_argument("--agent-env", type=Path, default=None,
                    help="a per-run KEY=VALUE file merged into the container's "
                         "environment. Names a different model backend for "
                         "THIS run without repointing every other run.")
    ap.add_argument(
        "--resume-run", type=Path, default=None,
        help=("completed agent run whose persisted provider session this "
              "make-up run continues; source is the lineage's published head "
              "and --hours is the new make-up budget"))
    ap.add_argument(
        "--resume-workspace", action="store_true",
        help=("with --resume-run, recover that interrupted run's unpublished "
              "candidate workspace instead of the lineage's published head; "
              "harness-owned files are regenerated"))
    ap.add_argument("--brief-file", type=Path, default=None)
    ap.add_argument("--agents-dir", type=Path, default=None)
    ap.add_argument("--claude-project-dir", type=Path, default=None)
    # Prompt treatments select their skill defaults after parsing, while every
    # skill path remains explicitly overridable for an audited experiment.
    ap.add_argument("--skill-dir", type=Path, default=None)
    ap.add_argument("--kernel-skill-dir", type=Path, default=None)
    ap.add_argument("--kernelwiki-skill-dir", type=Path, default=None)
    ap.add_argument("--kernelwiki-overlay-dir", type=Path, default=None)
    ap.add_argument("--ncu-skill-dir", type=Path, default=None)
    ap.add_argument("--model-id", default="",
                    help="default: the model the task declares")
    ap.add_argument("--stop", default="timebox",
                    choices=sorted(instructions_mod.STOP_CONDITIONS))
    ap.add_argument("--dry-run", action="store_true",
                    help="allocate in the scratch namespace and stop after the "
                         "workspace boots; consumes no production run id")
    args = ap.parse_args(argv)

    guidance = _guidance_defaults(args.task)
    for field, default in guidance.items():
        if getattr(args, field) is None:
            setattr(args, field, default)

    if args.agent_model is None:
        args.agent_model = (
            "gpt-5.6-sol" if args.agent_cli == "codex" else "claude-opus-5")
    if args.agent_effort is None:
        args.agent_effort = "max" if args.agent_cli == "codex" else "xhigh"

    # A task's topology says how many devices it needs; this driver provisions
    # exactly as many `--gpu` devices as the task declares: a workspace on
    # fewer or more cards than its reference arms answers a different question.
    needed = devices_declared(args.task.resolve())
    try:
        gpus = tuple(int(g) for g in str(args.gpu).split(",") if g.strip())
    except ValueError:
        print(f"--gpu takes device indices, got {args.gpu!r}", file=sys.stderr)
        return 2
    if not gpus:
        print("--gpu named no device", file=sys.stderr)
        return 2
    if len(set(gpus)) != len(gpus):
        print(f"--gpu names a device twice: {args.gpu}", file=sys.stderr)
        return 2
    # The count is the task's, not the caller's. Too FEW is the obvious error;
    # too many is also one, because the agent would be handed a card its task
    # never declared and the baselines never had, and the comparison is between
    # arrangements as much as implementations.
    if len(gpus) != needed:
        print(f"{Path(args.task).name} declares a topology on {needed} "
              f"device(s) and {len(gpus)} were given (--gpu {args.gpu}). "
              f"Running it on a different number is not a degraded run of this "
              f"task, it is a different experiment with the same name: its "
              f"baselines were measured on {needed}. "
              f"Pass --gpu with {needed} free index"
              f"{'es' if needed > 1 else ''}.", file=sys.stderr)
        return 2

    repo = Path(__file__).resolve().parents[3]

    # A seed that is not there is not a seed. `materialize.deliver` skips a
    # source that does not exist -- deliberately, so an absent optional bundle
    # is absent rather than fatal -- and the seed travels through that same
    # call. So a `--seed` pointing at a path that has been deleted would build
    # a workspace with NO starting source and say nothing.
    if args.seed is not None and not Path(args.seed).exists():
        print(f"--seed {args.seed} does not exist. A missing optional bundle "
              f"is skipped on purpose; a missing SEED is not, because it is "
              f"the candidate's starting source and a run without one begins "
              f"from an empty workspace", file=sys.stderr)
        return 2

    lineage_id = args.lineage_id or lineage_of(args.seed, args.lineage_root)
    # A resume is the SAME run continuing, so its own lock must not fence it
    # out of its own lineage.
    resuming_run_id = ""
    if getattr(args, "resume_run", None):
        with contextlib.suppress(Exception):
            resuming_run_id = ids.load(Path(args.resume_run).resolve()).run_id
    if args.resume_run is not None:
        try:
            source_identity = ids.load(args.resume_run.resolve())
        except Exception as exc:                              # noqa: BLE001
            print(f"--resume-run is not a readable run: {exc}", file=sys.stderr)
            return 2
        if lineage_id and lineage_id != source_identity.lineage_id:
            print(f"--resume-run belongs to lineage "
                  f"{source_identity.lineage_id!r}, but the seed/flag selects "
                  f"{lineage_id!r}", file=sys.stderr)
            return 2
        lineage_id = source_identity.lineage_id
    elif args.resume_workspace:
        print("--resume-workspace requires --resume-run", file=sys.stderr)
        return 2
    if args.resume_workspace:
        source_workspace = args.resume_run.resolve() / "workspace"
        if not source_workspace.is_dir():
            print(f"--resume-workspace source {source_workspace} does not "
                  "exist", file=sys.stderr)
            return 2

    # Continuing a lineage whose directory is gone REBUILDS it. `lineage_of` is
    # lexical -- it splits the seed path against the lineage root and never asks
    # the filesystem -- and `Publisher` creates its root on the first advance.
    # Together those mean a run seeded into a deleted lineage would silently
    # re-make the tree and number its first version v1 inside a name that had
    # already had several. An operator who deletes a prototype has made a decision, and
    # a harness that quietly undoes it is worse than one that stops.
    #
    # Refuse rather than fall back to a new lineage: the caller asked to
    # continue THIS ascent, and continuing a different one under a name they
    # did not choose is a substitution, not a recovery.
    if lineage_id:
        lineage_dir = Path(args.lineage_root).resolve() / lineage_id
        # ONE WRITER PER LINEAGE.
        #
        # A lineage is one ascent, and its version numbers are allocated from
        # the directory rather than from a run's own journal. Two live runs in
        # one lineage would therefore interleave, and one run's accepted head
        # could become a tree it never captured, which its final validation
        # refuses.
        live = lineage_live_writer(lineage_dir)
        if live and live.get("run_id") != resuming_run_id:
            print(f"lineage {lineage_id!r} is already being extended by run "
                  f"{live.get('run_id')!r} (pid {live.get('pid')}, since "
                  f"{live.get('since_utc')}).\n\n"
                  f"Version numbers in a lineage are allocated from its "
                  f"DIRECTORY, not from a run's journal, so a second run here "
                  f"would interleave versions with that one and each would "
                  f"end up with an accepted head the other wrote. To run two "
                  f"agents on this task at once, give this one its own "
                  f"lineage -- seed from the starter, or pass an explicit "
                  f"--lineage-id.", file=sys.stderr)
            return 2
        if not lineage_dir.is_dir():
            print(f"lineage {lineage_id!r} would be continued and "
                  f"{lineage_dir} does not exist. Publishing into it would "
                  f"RE-CREATE a lineage that was deleted, and number this "
                  f"run's first version v1 under a name that may already have "
                  f"had several. If the deletion was deliberate, start a new "
                  f"lineage by seeding from the starter; if it was not, "
                  f"restore the tree yourself -- this harness will not rebuild "
                  f"it from evidence held anywhere else", file=sys.stderr)
            return 2
    # A METERED KEY IS ONLY EVER GIVEN TO A MODEL THAT IS REACHED THROUGH IT.
    #
    # `claude-*` runs on the operator's OAuth credentials and `gpt-*`/codex on
    # theirs; both reach the container as a mounted file and cost a metered key
    # nothing. A run that ALSO names an `--agent-env` carrying an `sk-or-`
    # token would put that token in the container's environment, where the CLI
    # PREFERS it over the mounted credentials -- so an opus or codex run would
    # quietly spend a budget reserved for the OpenRouter-hosted arms.
    #
    # THE TEST IS THE MODEL ID'S SHAPE, not a list of names. Every
    # OpenRouter-routed model is `vendor/model` -- `z-ai/glm-5.3-flash`,
    # `deepseek/deepseek-v4.1-flash` -- and every first-party one is bare:
    # `claude-opus-5`, `gpt-5.6-sol`. A name list would need editing each time
    # a model is added. This needs no maintenance, and it refuses BEFORE any
    # card is claimed.
    if args.agent_env and "/" not in str(args.agent_model):
        token_file = Path(args.agent_env).expanduser()
        try:
            carries_key = "sk-or-" in token_file.read_text("utf-8")
        except OSError:
            carries_key = False
        if carries_key:
            print(f"REFUSED: --agent-model {args.agent_model} is a first-party "
                  f"model (no `vendor/` prefix), so it runs on the operator's "
                  f"own credentials, but --agent-env names {token_file}, which "
                  f"carries a metered third-party API key. Passing it would "
                  f"bill this run to that key and drain a budget reserved for "
                  f"the OpenRouter-hosted arms. Launch it with no "
                  f"--agent-env.", file=sys.stderr)
            return 2

    plan = RunPlan(
        task_dir=args.task.resolve(), repo=repo, gpus=gpus,
        hours=args.hours, image=args.image, model_dir=args.model_dir.resolve(),
        env_input=args.env_input.resolve(), lineage_id=lineage_id,
        lineage_root=args.lineage_root.resolve(), seed=args.seed,
        agent_cli=args.agent_cli, agent_model=args.agent_model,
        agent_effort=args.agent_effort,
        agent_env=(args.agent_env.resolve() if args.agent_env else None),
        credentials=(args.credentials.expanduser().resolve() if args.credentials else None),
        resume_run=(args.resume_run.resolve() if args.resume_run else None),
        resume_workspace=args.resume_workspace,
        brief=(args.brief_file.read_text() if args.brief_file else ""),
        agents_dir=args.agents_dir, skill_dir=args.skill_dir,
        claude_project_dir=args.claude_project_dir,
        ncu_skill_dir=args.ncu_skill_dir,
        kernel_skill_dir=args.kernel_skill_dir,
        kernelwiki_skill_dir=args.kernelwiki_skill_dir,
        kernelwiki_overlay_dir=args.kernelwiki_overlay_dir,
        dry_run=args.dry_run,
        model_id=(args.model_id or declared_model(args.task.resolve())[0]
                  or RunPlan.model_id),
        stop=args.stop)

    life = Lifecycle(plan)

    # An operator's stop is a stop. Without this, SIGTERM ends this process and
    # nothing else: the agent, its container and the service all
    # run on with no deadline over them. SIGHUP matters for the same reason:
    # an agent launched from a transient PTY would otherwise outlive its
    # driver, leaving no SUMMARY.json or final validation and its container
    # and broker alive.
    #
    # Both signals take the same preserving path as Ctrl-C, so the `finally`
    # below releases only what this run owns -- by recorded id. This cannot
    # catch SIGKILL; production launchers must still use a durable supervisor
    # (tmux/systemd) rather than make the lifecycle a child of an ephemeral
    # terminal.
    _install_stop_handlers()

    identity = life.create()
    # ONE WRITER PER LINEAGE: taken now that the run has an id, released when
    # this process ends however it ends. A lock whose pid is gone is ignored by
    # `lineage_live_writer`, so a killed driver cannot fence a lineage forever.
    if identity is not None and getattr(identity, "lineage_id", ""):
        _lineage_dir = Path(args.lineage_root).resolve() / identity.lineage_id
        claim_lineage(_lineage_dir, identity.run_id)
        atexit.register(release_lineage, _lineage_dir)
    print(f"run       {identity.run_id}")
    print(f"namespace {identity.namespace}")
    print(f"lineage   {identity.lineage_id}")
    print(f"epoch     {identity.control_epoch}")
    print(f"deadline  {identity.deadline_utc}")
    try:
        # Reserve first. Model identity verification hashes the staged weights
        # and takes long enough for another fleet scheduler to seize an
        # apparently idle card between selection and container startup.
        life.claim_devices()

        # THE WEIGHTS ON THE CARD MUST BE THE WEIGHTS THE TASK FROZE.
        #
        # A host-wide MODEL_DIR can be right for one task and wrong for the
        # next, and a run briefed on one model while serving another wastes
        # its whole budget. The check runs before any container starts; its
        # short-lived fleet claim is released by `finally` on mismatch.
        model_identity = _model_identity()
        want_model, want_revision = declared_model(plan.task_dir)
        try:
            model_identity.check(
                model_dir=plan.model_dir,
                declared_model_id=want_model,
                declared_revision=want_revision,
                # The task's own `required_assets` digests. A manifest that
                # nothing reads is documentation.
                task_dir=plan.task_dir)
        except model_identity.ModelMismatch as exc:
            print(f"MODEL_MISMATCH: {exc}", file=sys.stderr)
            # The run exists by now (its cards are claimed before hashing), so
            # it must end in a terminal state with a SUMMARY.json rather than
            # look, afterwards, like a run that is still going.
            with contextlib.suppress(Exception):
                life.preserve(RunState.FAILED,
                              why=f"MODEL_MISMATCH: {exc}"[:400],
                              classification=FailureClass.HARNESS.value)
            return 2

        life.build_workspace()
        life.start_container()
        report = life.boot()
        if not report.ok:
            print(f"HARNESS_SETUP_FAILED: {report.failure()}")
            return 2
        if args.dry_run:
            print("dry run: the workspace booted and no agent was launched")
            return 0
        agent_run = life.develop()
        summary = life.finish(agent_run)
        print(json.dumps(summary, indent=1))
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted: stopping the agent and the "
              "container this run owns, and preserving what it produced")
        if life.journal is not None:
            life.preserve(RunState.CANCELLED, why="stopped by the operator")
        return 130
    except Exception as exc:
        # Anything the HOST does wrong: docker down, a contended card refused
        # in `start_container`, a CaptureError, an OSError out of the workspace
        # build. Each ends the run in `RunState.FAILED` rather than a
        # NON-TERMINAL state with no SUMMARY.json, which would be
        # indistinguishable afterwards from a run still going.
        #
        # `preserve` is the same one path a cancel takes: trajectory, figure and
        # SUMMARY.json, so a run that died on the host still says what it built.
        print(f"\nHARNESS FAILURE: {exc!r}\n"
              f"preserving what this run produced and releasing what it owns",
              file=sys.stderr)
        traceback.print_exc()
        if life.journal is not None:
            with contextlib.suppress(Exception):
                life.settle_head_after_harness_failure()
            with contextlib.suppress(Exception):
                life.preserve(RunState.FAILED, why=repr(exc)[:400],
                              classification=FailureClass.HARNESS.value)
        return 1
    finally:
        life.release()


if __name__ == "__main__":
    raise SystemExit(main())


def _require_an_operating_point(task_dir) -> None:
    """Refuse a run on a task whose offered concurrency is undecided.

    The number is a property of the model, the hardware, the gates and the
    serving stack together, so it is measured per task and never inherited --
    which means a task can legitimately sit without one. What must not happen
    is a run starting anyway.

    Read from the task's own module, where `OFFERED_CONCURRENCY` is what the
    verdict offers the load at; the module is the one statement both the
    search and the verdict use.
    """
    name = Path(task_dir).name
    try:
        from .utils import ensure_harness_on_path
        ensure_harness_on_path()
        from final_validate import tasks as task_modules
        module = task_modules.python_module(name)
        selected = getattr(module, "OFFERED_CONCURRENCY", None)
        # A task whose candidate declares its own load (concurrency.json in
        # the published tree) has no fixed point to require.
        if getattr(module, "CONCURRENCY_POLICY", "fixed") == "candidate":
            return
    except Exception as exc:                                   # noqa: BLE001
        raise SystemExit(
            f"REFUSED: {name} has no readable task module "
            f"({type(exc).__name__}: {exc}), so the load it is measured at "
            f"cannot be established") from None
    if selected:
        return
    raise SystemExit(
        f"REFUSED: {name} has no offered concurrency, so there "
        f"is no load to measure it at and its verdict would refuse after the "
        f"run had already spent its hours.\n\n"
        f"Every live task declares CONCURRENCY_POLICY = \"candidate\"; a "
        f"fixed-load task needs OFFERED_CONCURRENCY set in its module.\n")
