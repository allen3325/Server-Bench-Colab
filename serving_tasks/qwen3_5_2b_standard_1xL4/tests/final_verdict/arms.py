"""Pinned baseline arms and the settings that make their work comparable.

Candidate and baseline arms are measured by the same instrument; only the
serving system changes. Task-specific payload settings keep reasoning modes,
sampling, and output bounds aligned across frameworks.

WHERE THIS TABLE LIVES
----------------------
This module is the only copy. A task's own launch line for an arm lives in
its `reference.yaml` and takes precedence.

THREE THINGS THIS TABLE IS CAREFUL ABOUT
----------------------------------------
**Semantic equivalence before speed.** A reasoning model emits a thinking block by default
and every framework turns that on differently. A baseline that is "thinking" is
doing a different and much longer computation than the oracle did, so its latency
is not comparable to anything. `extra_body` is what makes each arm do the same
work, and a run verifies it with a probe rather than trusting the flag.

**A framework that did not run is not a framework that lost.** An arm whose image
is absent, whose server never came up, or whose semantics could not be matched is
`NOT_RUN` with a reason. An empty column is a gap in the datasheet, never a zero
in a comparison.

**The pins are the pins.** `IMAGES` is the single place the three are named. A
floating tag makes a measurement unreproducible from its own record, and an old
tag produces findings about the pin's age that read exactly like findings about
the framework.
"""

from __future__ import annotations

from typing import Any

#: Changing a pin invalidates measurements and support findings taken under it.
IMAGES = {
    "sglang": "lmsysorg/sglang:v0.5.19",
    "vllm": "vllm/vllm-openai:v0.28.0",
    "tensorrt-llm": "nvcr.io/nvidia/tensorrt-llm/release:1.2.1",
    # Stable tags only: a release candidate is a moving target with a version
    # number.
}


def arms_for(task_id: str) -> tuple[str, ...]:
    """The baseline arms a task may name. Every live task may name each one."""
    del task_id
    return tuple(ARMS)


def image_for(arm: str) -> str:
    """The pinned image for one arm. The only way to name one."""
    try:
        return IMAGES[arm]
    except KeyError:
        raise KeyError(
            f"{arm!r} is not one of the pinned baseline arms "
            f"{sorted(IMAGES)}. There is no floating tag and no default"
        ) from None


#: How each baseline is launched, and what makes it do the same work as the
#: oracle. `serve` is formatted with the model mount and port; `extra_body` is
#: merged into every request; `ready_path` is polled until it answers.
ARMS: dict[str, dict[str, Any]] = {
    "sglang": {
        "dp_flag": " --dp-size {dp}",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "image": "sglang",
        "ready_path": "/health",
        # SGLang's /health performs a real generation. Architectures that
        # cannot use CUDA graphs may compile Triton kernels on this first
        # request; aborting it every five seconds prevents readiness from
        # ever completing. Let that request use the startup budget.
        "ready_request_timeout_s": 900,
        "serve": ("python3 -m sglang.launch_server --model-path {model} "
                  "--served-model-name target --host 0.0.0.0 --port "
                  "{port} --dtype bfloat16 "
                  "--trust-remote-code{parallel}"),
        "shm": "8g",
        "tp_flag": " --tp-size {tp}",
    },
    "tensorrt-llm": {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "image": "tensorrt-llm",
        "ready_path": "/health",
        "serve": ("trtllm-serve {model} --host 0.0.0.0 --port "
                  "{port} --backend pytorch{parallel}"),
        "shm": "8g",
        "tp_flag": " --tp_size {tp}",
    },
    "vllm": {
        "dp_flag": " --data-parallel-size {dp}",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "image": "vllm",
        "ready_path": "/health",
        "serve": ("--model {model} --served-model-name target --host "
                  "0.0.0.0 --port {port} --dtype bfloat16 "
                  "--trust-remote-code{parallel}"),
        "shm": "8g",
        "tp_flag": " --tensor-parallel-size {tp}",
    },
}

#: Every arm's image comes from the one pin table above.
for _arm, _spec in ARMS.items():
    _spec["image"] = image_for(_arm)

#: The one target that is not a baseline. The candidate is launched from its own
#: `serve.sh` with no `extra_body` at all -- getting semantic equivalence right
#: is part of what the task asks of it, and handing it the flag would answer the
#: question on its behalf.
CANDIDATE = {
    "image": "",                      # supplied per run: the task's agent image
    "serve": "bash /app/serve.sh",
    "extra_body": {},
    "ready_path": "/health",
    "shm": "8g",
}


def spec_for(target: str) -> dict[str, Any]:
    """`"candidate"` or one of the three arm names."""
    if target == "candidate":
        return dict(CANDIDATE)
    if target in ARMS:
        return dict(ARMS[target])
    raise KeyError(
        f"{target!r} is not a measurement target. Use 'candidate' "
        f"or one of {sorted(ARMS)}")
