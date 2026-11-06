"""The p90 bars: exact values per task, inclusive boundaries, no free pass.

Every live task scores requests against one p90 row (`request_p90_billable_tokens_v4`). The bars
are typed here by hand in `P90_ONLY`, on purpose: reading them from the task
would make these tests pass for any bar, including one that quietly moved.
`final_validate/service_levels.py` decides whether a measurement holds a bar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "serverbench" / "harness"))

TASKS_ROOT = REPO / "serving_tasks"


@pytest.fixture(scope="module")
def sl():
    from final_validate import service_levels
    return service_levels


#: Every live task's p90 bar, typed by hand: (TTFT ms, TPOT ms). A task with
#: its own SGLang measurements carries the worst p90 of five random-seed runs
#: at its reference point; a task that shares another's bars names it in
#: `derived_from` (`contract.yaml`).
P90_ONLY = {
    "qwen3_5_2b_standard_1xL40": (2707.8, 12.19),
    "qwen3_5_2b_instructed_1xL40": (2707.8, 12.19),
    # Provisional until `serverbench/scripts/colab/calibrate.py record` runs
    # on an L4; update this line to the bars it writes.
    "qwen3_5_2b_standard_1xL4": (4100.0, 35.0),
    "qwen3_5_9b_standard_2xL40": (4100.2848398999995, 23.497353185237426),
    "gemma4_e4b_it_standard_1xL40": (2526.9273697000003, 121.70566402153604),
    "qwen3_5_9b_standard_modal_2xL40S": (10000, 60.0),
    "gemma4_e4b_it_standard_modal_1xL40S": (3000, 80.0),
    "qwen3_5_9b_instructed_2xL40": (4100.2848398999995, 23.497353185237426),
    "gemma4_e4b_it_instructed_1xL40": (2526.9273697000003, 121.70566402153604),
    # SGLang v0.5.19 at c=4, worst p90 of five seeds, rounded UP to taskctl's
    # stored precision from 10298.7629 ms and 48.37687 ms.
    "qwen3_8_27b_int4_standard_2xL40": (10298.8, 48.3769),
}


@pytest.mark.parametrize("task_dir", sorted(P90_ONLY))
def test_a_v3_task_carries_exactly_its_p90_row(task_dir: str) -> None:
    import yaml
    body = yaml.safe_load(
        (TASKS_ROOT / task_dir / "contract.yaml").read_text("utf-8"))
    rows = body["service_levels"]
    assert [int(r["percentile"]) for r in rows] == [90], (
        f"{task_dir} must carry the p90 row alone")
    assert (float(rows[0]["ttft_ms"]), float(rows[0]["tpot_ms"])) \
        == P90_ONLY[task_dir]
    assert body["request_scoring"]["kind"] == "request_p90_billable_tokens_v4"
    assert body["request_scoring"]["gross_objective"] == "gross_dollars_per_dollar"
    assert body["objective"] == "dollars_per_dollar"


def _frozen(task_dir: str) -> tuple[float, float, float, float]:
    import yaml
    body = yaml.safe_load(
        (TASKS_ROOT / task_dir / "contract.yaml").read_text("utf-8"))
    rows = {int(r["percentile"]): r for r in body["service_levels"]}
    return tuple(float(rows[pc][f"{m}_ms"])
                 for pc in sorted(rows) for m in ("ttft", "tpot"))


def test_the_pin_covers_every_live_task() -> None:
    """A live task absent from `P90_ONLY` would be checked by nothing here."""
    live = {p.name for p in TASKS_ROOT.glob("*")
            if (p / "contract.yaml").is_file()}
    unpinned = sorted(live - set(P90_ONLY))
    assert not unpinned, (
        "these live tasks have no pinned bar in P90_ONLY:\n  "
        + "\n  ".join(f"{t}: {_frozen(t)}" for t in unpinned))
    stale = sorted(set(P90_ONLY) - live)
    assert not stale, f"P90_ONLY pins tasks that are not on disk: {stale}"


# --------------------------------------------------------------------------
# boundaries
# --------------------------------------------------------------------------

@pytest.mark.parametrize("task_dir", sorted(P90_ONLY))
def test_a_measurement_exactly_on_every_threshold_holds(sl, task_dir) -> None:
    """Both comparisons are INCLUSIVE."""
    t90, f90 = P90_ONLY[task_dir]
    levels = sl.read({"service_levels": [
        {"percentile": 90, "ttft_ms": t90, "tpot_ms": f90}]})
    checks = sl.check(levels, ttft_by_percentile={90: t90},
                      tpot_by_percentile={90: f90})
    assert sl.holds(checks), (
        f"{task_dir}: a measurement sitting exactly on every threshold was "
        f"refused. `<=` and `>=` mean the boundary holds")


def test_one_millisecond_over_a_ceiling_misses(sl) -> None:
    levels = sl.read({"service_levels": [
        {"percentile": 90, "ttft_ms": 3000, "tpot_ms": 12}]})
    over = sl.check(levels, ttft_by_percentile={90: 3001},
                    tpot_by_percentile={90: 12})
    assert not sl.holds(over)
    assert "p90 ttft 3001 > 3000 ms" in sl.why(over)
    assert "tpot" not in sl.why(over), (
        "a TTFT miss reported a TPOT failure too; the two bars are "
        "independent and must fail independently")


def test_a_hair_over_a_ceiling_misses(sl) -> None:
    """TPOT is a ceiling, not a floor."""
    levels = sl.read({"service_levels": [
        {"percentile": 90, "ttft_ms": 3000, "tpot_ms": 12}]})
    checks = sl.check(levels, ttft_by_percentile={90: 10},
                      tpot_by_percentile={90: 12.001})
    assert not sl.holds(checks)
    assert "p90 tpot 12.0 > 12 ms" in sl.why(checks)
    assert "ttft" not in sl.why(checks), (
        "a TPOT miss reported a TTFT failure too; the two bars are "
        "independent and must fail independently")


# --------------------------------------------------------------------------
# correctness, and evidence that is not evidence
# --------------------------------------------------------------------------

def test_a_huge_token_count_does_not_buy_a_correctness_failure() -> None:
    """The objective is not a currency the gates accept."""
    from final_validate import scenario, spine

    # A live task identity with a synthetic SLO: this is a unit test of the
    # verdict's decision, not of the task's bars.
    task_id = "qwen3_5_9b_standard_2xL40"
    contract = spine.Contract(
        task_id=task_id,
        task_dir=TASKS_ROOT / task_id,
        objective="tokens_per_usd", devices=2,
        slo={"service_levels": [
            {"percentile": 90, "ttft_ms": 6000, "tpot_ms": 24},
            {"percentile": 95, "ttft_ms": 10000, "tpot_ms": 32}]})
    row = {
        "arm": "candidate",
        "aggregate": {"ttft_p90_ms": 10.0,
                      "output_tokens_per_s": 900.0,
                      "benchmark_duration": 1800.0,
                      "total_input_tokens": 10 ** 9,
                      "total_output_tokens": 10 ** 7},
        "correctness": {"holds": False,
                        "why": "8 of 38 requests returned the wrong length"},
    }
    out = scenario.decide(contract, [row], usd_per_hour=2.0)
    decision = out["arms"][0]
    assert decision["outcome"] == "MISSES", (
        "an arm with every service gate held and a FAILED correctness gate was "
        "not recorded as missing. Correctness is checked first and is decisive")
    assert "wrong length" in decision["why_misses"]
    assert out["best_holding_arm"] is None, (
        "a best arm was named among arms that miss")
    assert out["best_tokens_per_usd"] is None


def test_absent_and_non_finite_evidence_never_holds_a_gate(sl) -> None:
    levels = sl.read({"service_levels": [
        {"percentile": 90, "ttft_ms": 3000, "tpot_ms": 12}]})
    for bad in (None, float("nan"), float("inf"), float("-inf"), "12",
                True, [12]):
        rows = sl.check(levels, ttft_by_percentile={90: 100},
                        tpot_by_percentile={90: bad})
        assert not sl.holds(rows), f"{bad!r} was accepted as a throughput"
        assert all(not r["measured"] for r in rows), (
            f"{bad!r} was recorded as a measurement")
    rows = sl.check(levels, ttft_by_percentile={}, tpot_by_percentile={90: 10.0})
    assert not sl.holds(rows)
    assert "not measured" in sl.why(rows)


def test_a_task_with_no_bar_is_refused_rather_than_defaulted(sl) -> None:
    for broken in ({}, {"service_levels": []},
                   {"service_levels": [{"percentile": 90, "ttft_ms": 1}]},
                   {"service_levels": [{"percentile": 90, "ttft_ms": 1,
                                        "tpot_ms": 1},
                                       {"percentile": 90, "ttft_ms": 2,
                                        "tpot_ms": 2}]}):
        with pytest.raises(sl.ContractError):
            sl.read(broken, task_id="probe_1xL40")


# --------------------------------------------------------------------------
# the shared contract
# --------------------------------------------------------------------------

SHARED_KEYS = ("SCENARIO", "CORPUS", "CORPUS_SHA256", "LOADER",
               "VERDICT_DURATION_S", "EVOLVE_DURATION_S", "ENVIRONMENT")


def test_all_the_tasks_share_the_scenario_but_keep_model_specific_load() -> None:
    """Corpus semantics are shared; model identity and load stay per task."""
    import yaml
    from final_validate import tasks
    from final_validate.tasks import _agentx_mvp as shared

    seen: dict[str, set] = {key: set() for key in SHARED_KEYS}
    contracts: dict[str, set] = {}
    for task_dir in sorted(P90_ONLY):
        module = tasks.python_module(task_dir)
        for key in SHARED_KEYS:
            value = getattr(module, key)
            seen[key].add(repr(value))
            assert value == getattr(shared, key), (
                f"{task_dir}.{key} does not come from `_agentx_mvp`. These "
                f"tasks share one execution contract by construction; a module "
                f"that restates a value has opted out of that guarantee")
        offered = module.OFFERED_CONCURRENCY or 1
        command = module.invocation("http://server", "/model", "/artifacts",
                                    concurrency=offered)
        assert command[command.index("--concurrency") + 1] == str(offered)

        # IDENTITY AND SCENARIO COME FROM THE MODULE, the bars and the
        # denominators from `contract.yaml`; neither restates the other.
        body = yaml.safe_load(
            (TASKS_ROOT / task_dir / "contract.yaml").read_text("utf-8"))
        contracts.setdefault("model", set()).add(
            (module.MODEL_ID, module.MODEL_REVISION))
        contracts.setdefault("devices", set()).add(body["devices"])
        contracts.setdefault("scenario", set()).add(
            (module.SCENARIO, module.CORPUS, module.CORPUS_SHA256,
             module.EVOLVE_DURATION_S, module.VERDICT_DURATION_S))
        contracts.setdefault("window", set()).add(body["window_seconds"])
        contracts.setdefault("objective", set()).add(body["objective"])

    for key, values in seen.items():
        assert len(values) == 1, f"{key} differs across the tasks: {values}"
    # WHAT IS SHARED AND WHAT IS PER TASK.
    #
    # `model` and `devices` are per task -- a task's id carries both, because
    # every gate, baseline and calibration in it is a statement about one
    # (model, device set) pair, and the suite exists to compare hardware.
    # Objective is also per task: a task may keep the same workload and
    # score it differently.
    per_task = ("model", "devices", "objective")
    for key, values in contracts.items():
        if key in per_task:
            continue
        assert len(values) == 1, (
            f"the model-specific tasks declare {len(values)} different {key}: "
            f"{values}. They share one AgentX workload contract")

    # The device count is per task, but it is NOT free: it has to be the
    # number in the task's own name. That is the invariant worth pinning --
    # a task called `_1xL40` whose topology declares two devices bills and
    # reports hardware it does not have, and nothing else would catch it.
    from final_validate import scenario as scenario_mod

    for task_dir in sorted(P90_ONLY):
        body = yaml.safe_load(
            (TASKS_ROOT / task_dir / "contract.yaml").read_text("utf-8"))
        suffix = task_dir.rsplit("_", 1)[-1]
        named = int(suffix.split("x")[0])
        assert body["devices"] == named, (
            f"{task_dir} is named for {named} device(s) but its contract "
            f"declares {body['devices']}")
        assert body["hardware"] == suffix, (
            f"{task_dir} is named for {suffix} and its contract says "
            f"{body['hardware']}")
        # And the cost denominator scenario.py actually bills is the product of
        # the two numbers the contract states, never a third stored field that
        # could disagree with them.
        detail = scenario_mod.objective_detail(task_dir)
        assert abs(float(detail["usd_per_hour_total"])
                   - named * float(body["usd_per_gpu_hour"])) < 1e-9, (
            f"{task_dir}: the hourly denominator scenario.py bills is not "
            f"devices x rate")

    models = {(m[0], m[1]) for m in contracts["model"]}
    assert models == {
        ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
        ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
        ("google/gemma-4-E4B-it",
         "ee0ef6023621cff504d758262d4e04895a5af4a2"),
        ("RedHatAI/Qwen3.8-27B-INT4", "91bd022d5b49442a868bc35008f6c21e1860edfa"),
    }
    # dtype is the checkpoint's own and the module does not restate it.
    # Read the durations rather than restating them: they are a shared
    # contract, and a literal here is a third copy that goes stale the day one
    # moves.
    from final_validate.tasks import _agentx_mvp as shared
    assert next(iter(contracts["scenario"]))[3:] == (
        shared.EVOLVE_DURATION_S, shared.VERDICT_DURATION_S), (
        "the tasks disagree with the shared contract about how long a "
        "development run and a scored run are")
    # ONE WINDOW. The development measurement and the verdict are the SAME
    # duration, so a pass the agent measures is a pass it gets: a tail
    # percentile measured over a shorter window can hold a bar that the full
    # window misses.
    assert shared.EVOLVE_DURATION_S == shared.VERDICT_DURATION_S, (
        f"the development window ({shared.EVOLVE_DURATION_S} s) is not the "
        f"scored window ({shared.VERDICT_DURATION_S} s), so a version can hold "
        f"the gates in development and miss them at the window that decides")
    assert shared.EVOLVE_DURATION_S >= 900, (
        "the shared window is below the scenario's own 900 s floor, so every "
        "measurement would need --unsafe-override and be stamped invalid")


@pytest.mark.parametrize("task_dir", sorted(P90_ONLY))
def test_each_task_dispatches_to_its_own_module(task_dir: str) -> None:
    from final_validate import tasks

    module = tasks.load(task_dir)
    assert module.task_id == task_dir
    assert module.agreement.name == "replay_fidelity"
    assert "scenario.py" in module.refuses
    # Every live task is scored per request; the id does not say so.
    assert module.reported_objective == "dollars_per_dollar"


@pytest.mark.parametrize("task_dir", sorted(P90_ONLY))
def test_each_task_ships_a_runnable_starter_demo(task_dir: str) -> None:
    """The agent is told to start from `demo/`, so `demo/` has to be there."""
    demo = TASKS_ROOT / task_dir / "demo"
    # `instruction.md` is the one document the agent is given; the starter is
    # source, with no document of its own.
    for name in ("server.py", "serve.sh", "start_server.sh", "stop_server.sh",
                 "gpu_env.sh", "smoke.sh", "example_requests.py"):
        assert (demo / name).is_file(), f"{task_dir}/demo/{name} is missing"
    body = (demo / "server.py").read_text("utf-8")
    assert "/v1/chat/completions" in body and "stream" in body, (
        f"{task_dir}'s starter does not serve the measured endpoint")
    t90, f90 = P90_ONLY[task_dir]
    header = (demo / "serve.sh").read_text("utf-8")
    assert f"p90 TTFT <= {t90:,g} ms" in header and f"p90 TPOT <= {f90:g} ms" in header, (
        f"{task_dir}/demo/serve.sh states a bar that is not this task's")
    assert "p95" not in header, f"{task_dir}/demo/serve.sh states a p95 bar"
