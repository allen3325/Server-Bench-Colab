#!/usr/bin/env python3
"""Derive a task's SGLang operating point and p90 bars on THIS machine's card.

OPERATIONS.md section 3 and docs/harness/bar-derivation.md, on a host with no
docker (a Colab L4 runtime). Everything about the measurement is the harness's
own -- `final_validate.concurrency` (aiperf at the pin, the task's scenario,
loader and window, the gates) -- and only how the SGLang server is started
changes: the pinned SGLang release runs from its own interpreter
(`SERVERBENCH_LOCAL_SGLANG_BIN`) on the host, with the same launch line the
task freezes, instead of `docker run lmsysorg/sglang:<pin>`.

Two steps, each RESUMABLE (a point already measured under `--out-dir` is
read back, not re-run), because a Colab session can end mid-way:

1. `search`: one 1,800 s window per offered load (default 1,2,4,8,16); the
   adopted point maximises gross dollars-per-dollar over p90 end-to-end
   latency (`max_gross_dollars_per_dollar_over_p90_request_latency`, the
   rule the Qwen3.5 2B L40 task used).
2. `seeds`: five more windows at the adopted point, each on a fresh seed; each
   bar is the worst p90 of the five (multiplier 1.0).

`record` then writes `concurrency_search` (points, adoption, derivation) into
the task's `reference.yaml` and sets the bars and the reference concurrency
through `taskctl set`, so the task stays managed.

    calibrate.py search --task T --model-dir M --out-dir D [--points 1,2,4,8,16]
    calibrate.py seeds  --task T --model-dir M --out-dir D [--seeds 5]
    calibrate.py record --task T --out-dir D
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
# The repository's tasks, never a guessed root: `spine` would otherwise look
# at /workspace/task_context, the agent's writable copy, if one is showing.
os.environ["SERVERBENCH_TASK_ROOT"] = str(REPO / "serving_tasks")
sys.path.insert(0, str(REPO / "serverbench" / "harness"))
sys.path.insert(0, str(REPO))

SGLANG_BIN = os.environ.get("SERVERBENCH_LOCAL_SGLANG_BIN",
                            "/opt/serverbench/sglang-venv/bin")
RULE = "max_gross_dollars_per_dollar_over_p90_request_latency"

_LIVE: list[subprocess.Popen] = []


class _Server:
    def __init__(self, process, log):
        self.process, self.log = process, log

    def stop(self):
        _stop(self.process)


def _stop(process: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def local_launch(target, *, image="", model_dir, device_spec="", port, name,
                 log, candidate=None, tp=1, dp=1, ready_timeout_s=None,
                 env=None, draft_model_dir=None, serve_extra=""):
    """`deploy.launch` for a baseline arm, as a host process instead of docker."""
    from final_validate.arms import spec_for
    from final_validate.deploy import ServerDidNotStart

    if target != "sglang":
        raise ServerDidNotStart(f"only the sglang arm runs locally, not {target!r}")
    spec = spec_for(target)
    command = shlex.split(spec["serve"].format(model=str(model_dir), port=port,
                                               parallel=""))
    command[0] = str(Path(SGLANG_BIN) / "python3")
    command += shlex.split(serve_extra or "")
    child_env = dict(os.environ)
    child_env.pop("PYTHONPATH", None)
    # FlashInfer JIT-compiles kernels at startup and runs `ninja` and `nvcc` off
    # PATH; `ninja` lives in SGLang's own venv, which nothing else puts on PATH
    # (local_verdict.py does the same for the reference arm).
    cuda_home = os.environ.get("SERVERBENCH_LOCAL_CUDA_HOME", "/usr/local/cuda")
    child_env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                      "PATH": f"{SGLANG_BIN}:{cuda_home}/bin:" + child_env.get("PATH", ""),
                      "CUDA_HOME": child_env.get("CUDA_HOME", cuda_home),
                      "PYTHONPATH": str(REPO / "serverbench/harness/final_validate/sglang_compat")})
    child_env.update(env or {})
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("w")
    handle.write("$ " + shlex.join(command) + "\n")
    handle.flush()
    process = subprocess.Popen(command, env=child_env, stdout=handle,
                               stderr=subprocess.STDOUT, start_new_session=True)
    _LIVE.append(process)
    deadline = time.time() + float(ready_timeout_s or 1800)
    request_timeout = float(spec.get("ready_request_timeout_s", 5))
    while time.time() < deadline:
        if process.poll() is not None:
            raise ServerDidNotStart(f"sglang exited {process.returncode}; "
                                    f"{log.read_text(errors='replace')[-1500:]}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}"
                                        + spec.get("ready_path", "/health"),
                                        timeout=request_timeout) as response:
                if 200 <= response.status < 300:
                    return _Server(process, log)
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    _stop(process)
    raise ServerDidNotStart("sglang did not answer /health in time")


def _release_all() -> None:
    while _LIVE:
        _stop(_LIVE.pop())


def _quiet(device_spec, *, timeout_s=900.0, quiet_mib=256) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        got = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, check=False)
        if not got.stdout.strip():
            return ""
        time.sleep(5)
    return "the card did not go quiet: " + got.stdout.strip()[:200]


def _patch() -> None:
    from final_validate import deploy
    from final_validate.concurrency import point
    deploy.launch = local_launch
    deploy._release_all = _release_all
    point.wait_for_quiet_cards = _quiet


def _plan(args, out_dir: Path, points: list[int]):
    from final_validate.concurrency import Plan
    return Plan(task=args.task, arm="sglang", model_dir=Path(args.model_dir),
                gpu=args.gpu, out_dir=out_dir, duration=1800, port=args.port,
                ready_timeout=1800.0, serve_extra="", random_seed=None,
                start=4, ceiling=512, resolution=2, points=points)


def _summary(task: str, run_dir: Path) -> dict | None:
    """One measured window: p90s, request count, gross $/$ -- from its export."""
    from final_validate import objective, scenario
    from final_validate.concurrency.point import aggregate
    body = aggregate(run_dir / "agentx" / "sglang")
    if body is None:
        return None

    def stat(metric, statistic):
        entry = body.get(metric)
        return entry.get(statistic) if isinstance(entry, dict) else None

    detail = scenario.objective_detail(task)
    rate = float(detail["usd_per_hour_total"])
    gross = objective.score(body, usd_per_hour=rate, prices=detail).get(objective.GROSS)
    e2e = stat("request_latency", "p90")
    return {
        "random_seed": (body.get("run_info") or {}).get("random_seed"),
        "requests": stat("request_count", "avg"),
        "ttft_p90_ms": round(stat("time_to_first_token", "p90"), 1),
        "tpot_p90_ms": round(stat("inter_token_latency", "p90"), 2),
        "e2e_p90_s": round(e2e / 1000.0, 2) if e2e is not None else None,
        "output_tokens_per_s": round(stat("output_token_throughput", "avg") or 0, 2),
        "gross_dollars_per_dollar": gross,
        "replay_holds": scenario._replay_fidelity(body)["holds"],
    }


def _measure(args, label: str, out_root: Path, offered: int) -> dict | None:
    """Measure one window unless `out_root/label` already holds one."""
    from final_validate.concurrency.strategy import sweep
    run_root = out_root / label
    found = _summary(args.task, run_root / f"c{offered}")
    if found is not None:
        print(f"{label}: already measured, reusing ({found['ttft_p90_ms']} ms / "
              f"{found['tpot_p90_ms']} ms)", flush=True)
        return found
    _patch()
    sweep(_plan(args, run_root, [offered]))
    found = _summary(args.task, run_root / f"c{offered}")
    if found is None:
        print(f"{label}: produced no aggregate; see {run_root}", flush=True)
    return found


def _adopt(points: list[dict]) -> dict:
    usable = [p for p in points if p.get("gross_dollars_per_dollar")
              and p.get("e2e_p90_s")]
    if not usable:
        raise SystemExit("no usable search point to adopt")
    return max(usable, key=lambda p: (float(p["gross_dollars_per_dollar"])
                                      / float(p["e2e_p90_s"]), -p["offered"]))


def cmd_search(args) -> int:
    out = Path(args.out_dir)
    rows = []
    for offered in [int(x) for x in args.points.split(",") if x.strip()]:
        found = _measure(args, f"search-c{offered}", out, offered)
        if found:
            rows.append({"label": f"search-c{offered}", "offered": offered, **found})
        (out / "search.json").write_text(json.dumps(rows, indent=1))
    adopted = _adopt(rows)
    print(f"\nadopted c{adopted['offered']} under {RULE}")
    for row in rows:
        print(f"  c{row['offered']:<3} ttft p90 {row['ttft_p90_ms']:>9} ms  "
              f"tpot p90 {row['tpot_p90_ms']:>7} ms  e2e p90 {row['e2e_p90_s']} s  "
              f"gross $/$ {row['gross_dollars_per_dollar']}")
    return 0


def cmd_seeds(args) -> int:
    out = Path(args.out_dir)
    rows = json.loads((out / "search.json").read_text())
    adopted = int(args.at or _adopt(rows)["offered"])
    runs = []
    for index in range(2, 2 + args.seeds):
        label = f"slo-c{adopted}-s{index}"
        found = _measure(args, label, out, adopted)
        if found:
            runs.append({"label": label, **found})
        (out / "seeds.json").write_text(json.dumps(
            {"adopted": adopted, "runs": runs}, indent=1))
    if runs:
        print(f"\nworst of {len(runs)}: ttft p90 "
              f"{max(r['ttft_p90_ms'] for r in runs)} ms, tpot p90 "
              f"{max(r['tpot_p90_ms'] for r in runs)} ms")
    return 0


def cmd_record(args) -> int:
    import yaml
    out = Path(args.out_dir)
    rows = json.loads((out / "search.json").read_text())
    seeds = json.loads((out / "seeds.json").read_text())
    adopted, runs = int(seeds["adopted"]), seeds["runs"]
    if len(runs) < args.min_seeds:
        raise SystemExit(f"only {len(runs)} seed run(s); need {args.min_seeds}")
    import math
    ttft = max(r["ttft_p90_ms"] for r in runs)
    tpot = max(r["tpot_p90_ms"] for r in runs)
    published_ttft = math.ceil(ttft * 10) / 10
    published_tpot = math.ceil(tpot * 100) / 100
    task_dir = REPO / "serving_tasks" / args.task
    reference = task_dir / "reference.yaml"
    body = yaml.safe_load(reference.read_text("utf-8"))
    image = (body.get("sglang_settings") or {}).get("image", "")
    body["concurrency_search"] = {
        "arm": "sglang", "image": image, "window_seconds": 1800,
        "selection_rule": RULE, "adopted": adopted,
        "random_seeds": "fresh per run",
        "host": f"Google Colab runtime ({_gpu_name()}); SGLang at the pinned release from "
                "its own interpreter rather than the pinned image "
                "(serverbench/scripts/colab/calibrate.py)",
        "points": [{k: r[k] for k in ("label", "offered", "random_seed",
                                      "requests", "ttft_p90_ms", "tpot_p90_ms",
                                      "e2e_p90_s", "output_tokens_per_s",
                                      "gross_dollars_per_dollar")}
                   for r in rows],
        "slo_derivation": {
            "rule": (f"the worst (highest) over {len(runs)} random seeds of "
                     f"SGLang's per-run p90 TTFT and p90 TPOT, at the searched "
                     f"operating point"),
            "sglang_worst_p90_ttft_ms": ttft, "sglang_worst_p90_tpot_ms": tpot,
            "multiplier": 1.0, "p90_ttft_ms": published_ttft,
            "p90_tpot_ms": published_tpot,
            "runs": [{k: r[k] for k in ("label", "random_seed", "ttft_p90_ms",
                                        "tpot_p90_ms", "gross_dollars_per_dollar")}
                     for r in runs],
        },
    }
    reference.write_text(yaml.safe_dump(body, sort_keys=False, width=200), "utf-8")
    taskctl = [sys.executable, str(REPO / "serving_tasks/_manage/taskctl.py")]
    reason = (f"measured on this {_gpu_name()}: worst p90 of {len(runs)} SGLang seeds at "
              f"c{adopted} ({out})")
    subprocess.run(taskctl + ["set", args.task, "--p90-ttft", str(published_ttft),
                              "--p90-tpot", str(published_tpot),
                              "--reference-concurrency", str(adopted),
                              "--reason", reason], check=True)
    subprocess.run(taskctl + ["render", args.task], check=True)
    got = subprocess.run(taskctl + ["check", args.task], check=False)
    print(f"\n{args.task}: bars {published_ttft} ms / {published_tpot} ms, "
          f"reference concurrency {adopted}")
    return got.returncode


def _gpu_name() -> str:
    got = subprocess.run(["nvidia-smi", "-i", os.environ.get("GPU", "0"),
                          "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True)
    return got.stdout.strip().splitlines()[0] if got.stdout.strip() else "unknown GPU"


def _hardware_check(task: str) -> None:
    """A task's bars are its own card's: refuse to measure them on another."""
    got = subprocess.run([sys.executable, str(HERE / "hardware.py"), "check", task])
    if got.returncode != 0:
        raise SystemExit(got.returncode)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    for name in ("search", "seeds", "record"):
        p = sub.add_parser(name)
        p.add_argument("--task", required=True)
        p.add_argument("--out-dir", required=True)
        if name != "record":
            p.add_argument("--model-dir", required=True)
            p.add_argument("--gpu", default="0")
            p.add_argument("--port", type=int, default=8931)
    sub.choices["search"].add_argument("--points", default="1,2,4,8,16")
    sub.choices["seeds"].add_argument("--seeds", type=int, default=5)
    sub.choices["seeds"].add_argument("--at", type=int, default=0,
                                      help="override the adopted point")
    sub.choices["record"].add_argument("--min-seeds", type=int, default=5)
    args = ap.parse_args(argv)
    if args.verb in ("search", "seeds"):
        _hardware_check(args.task)
    try:
        return {"search": cmd_search, "seeds": cmd_seeds,
                "record": cmd_record}[args.verb](args)
    finally:
        _release_all()


if __name__ == "__main__":
    raise SystemExit(main())
