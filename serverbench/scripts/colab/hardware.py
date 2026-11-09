#!/usr/bin/env python3
"""Which GPU this runtime has, and whether a task was made for it.

A task's numbers mean something only on the hardware it names: its p90 bars
are SGLang's, measured on that card, and its dollars are that card's hours.
Nothing else in the harness compares the card it is given with the card the
task declares, so an L4 task run on an A100 would quietly score A100 numbers
against L4 bars. This is that comparison, for the Colab scripts:

    hardware.py detect            # this card's task suffix, e.g. L4
    hardware.py task              # the Qwen3.5-2B task id for this card
    hardware.py check <task>      # exit 1 unless this card is the task's card
    hardware.py new-task [--usd-per-gpu-hour X]
                                  # make the task for this card (once per card)

A new task copies everything from `qwen3_5_2b_standard_1xL4` (model, token
prices, framework flags, correctness settings) except the hardware, and starts
with that task's provisional bars until `calibrate.py` measures its own.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BASE_TASK = "qwen3_5_2b_standard_1xL4"
FAMILY = "qwen3_5_2b_standard_1x"

#: Colab cards: task suffix -> (pattern on `nvidia-smi --query-gpu=name`,
#: memory in GiB, support). The suffix goes into the task id, whose hardware
#: part is letters and digits only.
CARDS = {
    "L4":         (r"\bL4\b", 24, "supported"),
    "A100":       (r"\bA100\b(?!.*80GB)", 40, "expected"),
    "A10080GB":   (r"\bA100\b.*80GB", 80, "expected"),
    "H100":       (r"\bH100\b", 80, "expected"),
    "RTXPRO6000": (r"RTX PRO 6000", 96, "experimental"),
    "T4":         (r"\bT4\b", 16, "unsupported"),
}
SUPPORT = {
    "supported": "the card this fork was built and tested for",
    "expected": "Ampere/Hopper with bf16: SGLang and the toolchain support it; not run end to end here",
    "experimental": "Blackwell (sm_120): CUDA 12.8 supports it; SGLang 0.5.19's kernels are unverified",
    "unsupported": "Turing (sm_75) has no bf16, which the model and the task's SGLang flags use",
}


def gpu_name(index: str = os.environ.get("GPU", "0")) -> str:
    got = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=name",
                          "--format=csv,noheader"], capture_output=True, text=True)
    if got.returncode != 0 or not got.stdout.strip():
        raise SystemExit(f"nvidia-smi cannot see GPU {index}: {got.stderr.strip()}")
    return got.stdout.strip().splitlines()[0]


def suffix_of(name: str) -> str:
    for suffix, (pattern, _, _) in CARDS.items():
        if re.search(pattern, name):
            return suffix
    raise SystemExit(f"no task suffix for {name!r}; known cards: {', '.join(CARDS)}")


def task_hardware(task: str) -> str:
    found = re.search(r"_\d+x([A-Za-z0-9]+)$", task)
    if not found:
        raise SystemExit(f"{task} does not end in its hardware (e.g. _1xL4)")
    return found.group(1)


def cmd_detect(_args) -> int:
    name = gpu_name()
    suffix = suffix_of(name)
    support = CARDS[suffix][2]
    print(suffix)
    print(f"{name}: {support} -- {SUPPORT[support]}", file=sys.stderr)
    return 0


def cmd_task(_args) -> int:
    print(FAMILY + suffix_of(gpu_name()))
    return 0


def cmd_check(args) -> int:
    want = task_hardware(args.task)
    name = gpu_name()
    override = os.environ.get("SB_ALLOW_HARDWARE_MISMATCH") == "1"
    try:
        have = suffix_of(name)
    except SystemExit as unknown:
        print(f"REFUSED: {unknown}", file=sys.stderr)
        return 0 if override else 1
    if have != want:
        print(f"REFUSED: {args.task} is a {want} task and this runtime has {name} "
              f"({have}). Its bars and its dollars are the {want}'s, so a run here "
              f"would score {have} numbers against them. Use (or make) the task for "
              f"this card: `hardware.py new-task` -> {FAMILY}{have}. "
              f"SB_ALLOW_HARDWARE_MISMATCH=1 overrides, for debugging only.",
              file=sys.stderr)
        return 0 if override else 1
    support = CARDS[have][2]
    if support == "unsupported" and os.environ.get("SB_ALLOW_UNSUPPORTED_GPU") != "1":
        print(f"REFUSED: {name} is unsupported: {SUPPORT[support]}. "
              f"SB_ALLOW_UNSUPPORTED_GPU=1 tries anyway.", file=sys.stderr)
        return 1
    if support != "supported":
        print(f"note: {name} is {support}: {SUPPORT[support]}", file=sys.stderr)
    return 0


def _yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text("utf-8")) or {}


def cmd_new_task(args) -> int:
    name = gpu_name()
    suffix = args.suffix or suffix_of(name)
    task = FAMILY + suffix
    task_dir = REPO / "serving_tasks" / task
    if task_dir.exists():
        print(f"{task} already exists ({task_dir})")
        return 0
    base_dir = REPO / "serving_tasks" / BASE_TASK
    contract = _yaml(base_dir / "contract.yaml")
    reference = _yaml(base_dir / "reference.yaml")
    arms = {a["framework"]: a["serve_extra"] for a in reference["arms"]}
    prices = contract["token_prices"]
    levels = contract["service_levels"][0]
    model = contract["model_config"]
    regret = (reference.get("correctness") or {}).get("max_token_regret", 4.0)
    rate = args.usd_per_gpu_hour
    label = f"one {suffix}"
    env = dict(os.environ, PYTHONPATH=f"{REPO}/serverbench/harness:{REPO}",
               SERVERBENCH_TASK_ROOT=str(REPO / "serving_tasks"))
    env.setdefault("HF_HOME", "/opt/serverbench/hf")
    taskctl = [sys.executable, str(REPO / "serving_tasks/_manage/taskctl.py")]
    subprocess.run(taskctl + [
        "new", task, "--mode", "docker",
        "--model-id", model["repo"], "--model-revision", model["revision"],
        "--max-context-length", "262144",
        "--title", f"Qwen3.5 2B Standard ({suffix})",
        "--question", (f"Serve Qwen3.5-2B on {label} for the AgentX coding-session "
                       "replay at a concurrency the system chooses itself, and earn "
                       "the most token value per GPU dollar from requests that meet "
                       "the p90 TTFT and TPOT bars."),
        "--usd-per-gpu-hour", str(rate),
        "--price-in", str(prices["usd_per_million_input"]),
        "--price-out", str(prices["usd_per_million_output"]),
        "--price-cached", str(prices["usd_per_million_cached_input"]),
        "--price-source", (f"Operator-supplied task tariff, the same token prices as "
                           f"{BASE_TASK}; the {suffix} hourly rate is an operator-set "
                           f"value ({rate} USD)"),
        "--p90-ttft", str(levels["ttft_ms"]), "--p90-tpot", str(levels["tpot_ms"]),
        "--sglang-serve-extra", arms["sglang"],
        "--vllm-serve-extra", arms["vllm"],
        "--trtllm-serve-extra", arms["tensorrt-llm"],
        "--from-demo", BASE_TASK,
    ], check=True, env=env, cwd=REPO)
    subprocess.run(taskctl + ["set", task, "--max-token-regret", str(regret)],
                   check=True, env=env, cwd=REPO)
    print(f"\n{task}: made for {suffix} (this runtime: {name}). Its bars ({levels['ttft_ms']} ms / "
          f"{levels['tpot_ms']} ms) are the L4 task's PROVISIONAL ones until "
          f"`calibrate.py search | seeds | record --task {task}` measures this card.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="verb", required=True)
    sub.add_parser("detect")
    sub.add_parser("task")
    check = sub.add_parser("check")
    check.add_argument("task")
    new = sub.add_parser("new-task")
    new.add_argument("--usd-per-gpu-hour", type=float, default=0.80,
                     help="the card's hourly rate (operator-set; it scales every "
                          "run's dollars equally, so it does not change a "
                          "comparison between agents on this card)")
    new.add_argument("--suffix", default="", help="override the detected card")
    args = ap.parse_args(argv)
    return {"detect": cmd_detect, "task": cmd_task, "check": cmd_check,
            "new-task": cmd_new_task}[args.verb](args)


if __name__ == "__main__":
    raise SystemExit(main())
