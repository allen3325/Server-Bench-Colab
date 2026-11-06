"""What a serving task IS, as parameters -- so the next one is filled in, not written.

    $SERVERBENCH_PY serverbench/harness/final_validate/task_spec.py --check
    $SERVERBENCH_PY serverbench/harness/final_validate/task_spec.py --show <task>

Every live task replays the same corpus through the same scenario and differs
only in parameters: model, hardware, prices, p90 bars, context limit and the
SGLang reference point. `TaskSpec` is the single list, and `check()` reads the
live tasks back and refuses if any restatement disagrees with its source.

## Core parameters, and the escape hatch

Everything in `TaskSpec` is required of every task. `model_specific` is an open
dict for what only some models need -- a whole-trace context limit, a separate
draft model -- and it is open on purpose.

A closed parameter list would force a template change for the third model, and
the change that gets forgotten is the one that matters. So the shape is fixed
and the contents are not: anything under `model_specific` is recorded in the
task, reaches the launch line, and is named in the brief — but nothing here
predicts what it will be.

## Generated authority and specialised extensions

`serving_tasks/_manage/instruction_render.py` renders the agent's document
from this parsed spec; this module composes no agent-facing prose.
Task rationale and model-specific notes may remain authored extensions, but
they cannot override the model, hardware, operating point, prices, window,
or gates the evaluator reads. `taskctl render` writes the
generated brief, rebuilds the packaged verdict, and checks both copies.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
#: `serverbench/harness/`: the package root `final_validate` and the harness
#: packages import from.
HARNESS = HERE.parents[1]
REPO = HARNESS.parents[1]
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

SCHEMA = "final_validate.task_spec/1"


@dataclass
class ServiceLevel:
    """One percentile, two ceilings. Never one shared number."""

    percentile: int
    ttft_ms: float
    tpot_ms: float


@dataclass
class TokenPrices:
    """What the served tokens are WORTH, in THIS model's market.

    A task's `dollars_per_dollar` is denominated in its own model's revenue.
    Two tasks serving different models are each meaningful against their own
    market and are NOT directly comparable with one another.

    `cache_rate_is` is prose on purpose: for most models nobody publishes a
    cache-read price, so the figure is a stated convention rather than a quote,
    and it is the single most consequential number in the objective. When
    nearly all input is a cache hit, the cache rate alone can move dollars
    per dollar several-fold.
    """

    model: str
    usd_per_million_input: float
    usd_per_million_output: float
    usd_per_million_cached_input: float
    source: str
    cache_rate_is: str

    def per_token(self) -> dict[str, float]:
        return {
            "usd_per_input_token": self.usd_per_million_input / 1e6,
            "usd_per_output_token": self.usd_per_million_output / 1e6,
            "usd_per_cached_input_token":
                self.usd_per_million_cached_input / 1e6,
        }


@dataclass
class TaskSpec:
    task_id: str
    model_id: str
    model_revision: str

    #: The hardware this task is ABOUT. It is in the task's name because every
    #: gate, baseline and calibration is a statement about one device set: the
    #: same model under the same gates saturates at a different load on two
    #: L40s than on two H200s.
    hardware: str
    devices: int
    usd_per_gpu_hour: float

    service_levels: list[ServiceLevel]
    token_prices: TokenPrices

    #: Sessions, not requests. `None` until the SGLang search has run on this
    #: hardware, and the verdict REFUSES rather than inventing one.
    offered_concurrency: int | None

    objective: str = "dollars_per_dollar"
    request_scoring: dict[str, Any] = field(default_factory=dict)

    #: "fixed" -- the verdict offers `offered_concurrency`; "candidate" -- the
    #: published tree declares its own load in `concurrency.json`, and
    #: `reference_concurrency` is the SGLang operating point the task STATES
    #: as a reference without imposing it.
    concurrency_policy: str = "fixed"
    reference_concurrency: int | None = None

    #: ONE WINDOW. Both default to the same number on purpose -- a task built
    #: without an explicit window must not silently get a separated design.
    #: The default is live, not a comment: if the two differed, `TaskSpec(...)`
    #: with no window argument would produce a task whose development
    #: measurement does not predict its verdict.
    verdict_duration_s: int = 1800
    evolve_duration_s: int = 1800

    #: Whatever only this model needs. Open by design -- see the module note.
    model_specific: dict[str, Any] = field(default_factory=dict)

    #: The task's hashed id (`spine.task_hash`), as `contract.yaml` states it.
    task_hash: str = ""

    # ---------------------------------------------------------------- checks

    def problems(self) -> list[str]:
        """Everything wrong with this spec, in the words of what it costs."""
        bad: list[str] = []
        if self.task_hash != _spine().task_hash(self.task_id):
            bad.append(
                f"{self.task_id}: contract.yaml task_hash is "
                f"{self.task_hash or 'missing'!r}, its id hashes to "
                f"{_spine().task_hash(self.task_id)!r}")
        if not self.task_id.endswith(self.hardware):
            bad.append(
                f"{self.task_id} does not carry its hardware ({self.hardware}) "
                f"in its name, so its numbers will eventually be read as being "
                f"about a machine that never ran them")
        levels = {lv.percentile: lv for lv in self.service_levels}
        # Every task carries the p90 row alone. Nothing else.
        if sorted(levels) != [90]:
            bad.append(f"{self.task_id}: expected the p90 row alone, got "
                       f"{sorted(levels)}")
        p = self.token_prices
        if p.model.split("/")[-1].lower().replace("-", "") not in \
                self.model_id.lower().replace("-", "").replace("_", ""):
            bad.append(
                f"{self.task_id} serves {self.model_id} but prices "
                f"{p.model}, so its objective is denominated in another "
                f"model's revenue")
        if p.usd_per_million_cached_input > p.usd_per_million_input:
            bad.append(f"{self.task_id}: cached input priced above uncached")
        try:
            from final_validate import objective
        except ModuleNotFoundError:  # packaged as final_verdict
            from . import objective
        if self.objective != objective.OBJECTIVE:
            bad.append(f"{self.task_id}: objective is {self.objective!r}, "
                       f"expected {objective.OBJECTIVE!r}")
        error = objective.request_scoring_error(self.request_scoring)
        if error:
            bad.append(f"{self.task_id}: {error}")
        if not self.request_scoring.get("reference_task"):
            bad.append(f"{self.task_id}: request_scoring does not name the task "
                       "whose workload and operating point it carries")
        # ONE 1,800-SECOND WINDOW. This is an exact execution-contract value,
        # not a lower bound. A shorter replay can accept a candidate whose
        # TTFT p90 then misses at the 1,800 s verdict: cache eviction and
        # session churn in the long tail are not represented by the shorter
        # replay. Equality alone is insufficient: two equally wrong 900 s
        # values must not validate.
        for role, value in (("verdict", self.verdict_duration_s),
                            ("evolve", self.evolve_duration_s)):
            if value != 1800:
                bad.append(
                    f"{self.task_id}: the {role} performance window is "
                    f"{value} s; every evolve benchmark and final verdict "
                    f"must use the same 1,800 s replay")
        required_context = self.model_specific.get("required_context_length")
        filtered_context = self.model_specific.get("max_context_length")
        if required_context is not None and int(required_context) < 1:
            bad.append(f"{self.task_id}: required_context_length must be positive")
        if filtered_context is not None and int(filtered_context) < 1:
            bad.append(f"{self.task_id}: max_context_length must be positive")
        if required_context is not None and filtered_context is not None:
            bad.append(
                f"{self.task_id}: cannot require full context through "
                f"{required_context} and filter it at {filtered_context}")
        return bad

    # ------------------------------------------------------------- rendering

    def slo_block(self) -> dict[str, Any]:
        return {"service_levels": [
            {"percentile": lv.percentile, "ttft_ms": lv.ttft_ms,
             "tpot_ms": lv.tpot_ms} for lv in self.service_levels],
            "request_scoring": dict(self.request_scoring)}

    def objective_detail(self) -> dict[str, Any]:
        return {"objective": self.objective,
                "request_scoring": dict(self.request_scoring),
                "usd_per_hour_total": self.usd_per_gpu_hour * self.devices,
                "usd_per_gpu_hour": self.usd_per_gpu_hour,
                "devices": self.devices,
                "token_prices": {**self.token_prices.per_token(),
                                 "model": self.token_prices.model,
                                 "source": self.token_prices.source,
                                 "cache_rate_is": self.token_prices.cache_rate_is,
                                 "usd_per_million_input":
                                     self.token_prices.usd_per_million_input,
                                 "usd_per_million_output":
                                     self.token_prices.usd_per_million_output,
                                 "usd_per_million_cached_input":
                                     self.token_prices.usd_per_million_cached_input}}


def task_root() -> Path:
    """Where the task tree lives, honouring `SERVERBENCH_TASK_ROOT`.

    One resolver rather than four copies of `REPO / "serving_tasks"`. The
    override exists so the task tooling can be exercised on a disposable copy:
    these tasks are evolved against for hours, `taskctl set` rewrites a dozen
    files across a task, and checking that a parameter change actually reaches
    the agent's brief otherwise has no safe subject.

    `serving_tasks/_manage/sites.py` reads the same variable. Both must,
    because `taskctl` resolves paths through `sites` and the CONTRACT through
    this module -- overriding one and not the other would send the same command
    to two different trees, failing on a file it had just written.
    """
    import os
    return Path(os.environ.get("SERVERBENCH_TASK_ROOT")
                or REPO / "serving_tasks").resolve()


#: The two data files a task carries, and the only ones: `contract.yaml`
#: (the latency bars, prices, devices and window) and `reference.yaml` (the
#: SGLang operating-point evidence and the reference arms' launch lines).
#: Identity, scenario, corpus, durations and the context ceiling are in the
#: task's own Python module.
CONTRACT_FILE = "contract.yaml"
REFERENCE_FILE = "reference.yaml"


def contract(task_dir: Path | str) -> dict[str, Any]:
    """One task's frozen bars. THE reader, so there is one parse of them."""
    import yaml
    body = yaml.safe_load(
        (Path(task_dir) / CONTRACT_FILE).read_text("utf-8")) or {}
    return body


def reference(task_dir: Path | str) -> dict[str, Any]:
    """One task's measured evidence. Absent fields mean not measured yet."""
    import yaml
    path = Path(task_dir) / REFERENCE_FILE
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text("utf-8")) or {}


def serve_extra(task_dir: Path | str, arm: str) -> str:
    """The launch line the task freezes for one reference arm."""
    for entry in reference(task_dir).get("arms") or []:
        if entry.get("framework") == arm:
            return entry.get("serve_extra") or ""
    return ""


def sglang_settings(task_dir: Path | str) -> dict[str, Any]:
    """The SGLang reference arm's launch flags as `{flag: value}`, in order.

    Read from the one launch line the task freezes (`arms[sglang].serve_extra`
    in `reference.yaml`); `taskctl render` writes this mapping back beside it
    as `sglang_settings` and into the instruction. A flag with no value maps to
    True; `{concurrency}` is filled per run with the offered load.
    """
    import shlex

    parts = shlex.split(serve_extra(task_dir, "sglang"))
    out: dict[str, Any] = {}
    index = 0
    while index < len(parts):
        token = parts[index]
        if not token.startswith("--"):
            raise ValueError(f"unexpected SGLang argument {token!r}")
        flag, _, inline = token[2:].partition("=")
        if inline:
            out[flag] = inline
        elif index + 1 < len(parts) and not parts[index + 1].startswith("--"):
            index += 1
            out[flag] = parts[index]
        else:
            out[flag] = True
        index += 1
    return out


def model_overrides(task_dir: Path | str) -> dict[str, Any]:
    """Effective checkpoint overrides frozen in the SGLang reference arm."""
    import shlex

    parts = shlex.split(serve_extra(task_dir, "sglang"))
    for index, token in enumerate(parts):
        if token.split("=", 1)[0] != "--json-model-override-args":
            continue
        value = token.split("=", 1)[1] if "=" in token else parts[index + 1]
        overrides = json.loads(value)
        if not isinstance(overrides, dict):
            raise ValueError("model overrides must be a JSON object")
        return overrides
    return {}


def read(task_id: str) -> TaskSpec:
    """Read a live task back AS a spec. The template is derived from reality."""
    try:
        from final_validate import tasks as task_modules
    except ModuleNotFoundError:          # the in-Harbor copy
        from . import tasks as task_modules  # type: ignore[no-redef]
    root = task_root() / task_id
    module = task_modules.python_module(task_id)
    con = contract(root)
    prices = con.get("token_prices") or {}

    extra: dict[str, Any] = {}
    for name in ("MAX_CONTEXT_LENGTH", "REQUIRED_CONTEXT_LENGTH"):
        if getattr(module, name, None) is not None:
            extra[name.lower()] = getattr(module, name)

    return TaskSpec(
        task_id=task_id,
        task_hash=str(con.get("task_hash") or ""),
        model_id=module.MODEL_ID,
        model_revision=module.MODEL_REVISION,
        hardware=con.get("hardware") or task_id.rsplit("_", 1)[-1],
        # No default. A task that does not state its device count is a task
        # whose cost denominator nobody has stated.
        devices=int(con["devices"]),
        usd_per_gpu_hour=float(con["usd_per_gpu_hour"]),
        service_levels=[ServiceLevel(int(r["percentile"]), float(r["ttft_ms"]),
                                     float(r["tpot_ms"]))
                        for r in con.get("service_levels") or []],
        token_prices=TokenPrices(
            model=prices.get("model", ""),
            usd_per_million_input=float(prices.get("usd_per_million_input", 0)),
            usd_per_million_output=float(prices.get("usd_per_million_output", 0)),
            usd_per_million_cached_input=float(
                prices.get("usd_per_million_cached_input", 0)),
            source=prices.get("source", ""),
            cache_rate_is=prices.get("cache_rate_is", "")),
        offered_concurrency=module.OFFERED_CONCURRENCY,
        concurrency_policy=str(getattr(module, "CONCURRENCY_POLICY", "fixed")),
        reference_concurrency=getattr(module, "REFERENCE_CONCURRENCY", None),
        objective=str(con.get("objective") or "dollars_per_dollar"),
        request_scoring=dict(con.get("request_scoring") or {}),
        verdict_duration_s=int(module.VERDICT_DURATION_S),
        evolve_duration_s=int(module.EVOLVE_DURATION_S),
        model_specific=extra)


def live_tasks() -> list[str]:
    return [p.name for p in _spine().task_dirs(task_root())]


def _spine():
    try:
        from final_validate import spine
    except ModuleNotFoundError:          # the in-Harbor copy
        from . import spine  # type: ignore[no-redef]
    return spine


def check() -> list[str]:
    """Every live task, read back and validated. Empty means conforming."""
    bad: list[str] = []
    for task in live_tasks():
        try:
            bad.extend(read(task).problems())
        except Exception as exc:                               # noqa: BLE001
            bad.append(f"{task}: could not be read as a spec: "
                       f"{type(exc).__name__}: {exc}")
    return bad


def main(argv: list[str] | None = None) -> int:
    import argparse
    import dataclasses
    import json

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true",
                   help="read every live task back as a spec and validate it")
    g.add_argument("--show", metavar="TASK",
                   help="print one task as a spec")
    args = ap.parse_args(argv)

    if args.show:
        print(json.dumps(dataclasses.asdict(read(args.show)), indent=1))
        return 0

    problems = check()
    tasks = live_tasks()
    if problems:
        print(f"{len(problems)} problem(s) across {len(tasks)} task(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"{len(tasks)} task(s) conform to the spec: {', '.join(tasks)}")
    for task in tasks:
        s = read(task)
        levels = {lv.percentile: lv for lv in s.service_levels}
        print(f"\n  {s.task_id}")
        print(f"    model        {s.model_id} @ {s.model_revision[:12]}")
        print(f"    hardware     {s.hardware}  ({s.devices} devices, "
              f"${s.usd_per_gpu_hour:.2f}/GPU-hour)")
        print("    p90 bars     " + " · ".join(
            f"p{pc} TTFT {levels[pc].ttft_ms:,.0f} ms / "
            f"TPOT {levels[pc].tpot_ms:g} ms" for pc in sorted(levels)))
        print("    concurrency  "
              + (f"candidate-declared (SGLang reference "
                 f"{s.reference_concurrency})"
                 if s.concurrency_policy == "candidate"
                 else f"{s.offered_concurrency}"))
        print(f"    prices       ${s.token_prices.usd_per_million_input}/M in · "
              f"${s.token_prices.usd_per_million_output}/M out · "
              f"${s.token_prices.usd_per_million_cached_input}/M cached "
              f"({s.token_prices.model})")
        print(f"    window       {s.verdict_duration_s} s scored / "
              f"{s.evolve_duration_s} s development")
        print(f"    model-only   {s.model_specific or 'nothing'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
