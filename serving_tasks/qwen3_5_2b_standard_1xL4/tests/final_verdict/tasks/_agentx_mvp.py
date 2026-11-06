"""The one execution contract the AgentX scenario tasks share.

Each live task is a model- and hardware-specific profile over one workload.
What a task module declares: its model and revision, its devices, its whole-
trace context filter, and the reference operating point its instruction states.
Everything else is shared here so the task modules cannot drift:

    scenario      inferencex-agentx-mvp (a lock: its own loader, its own
                  900 s floor, recorded inter-turn delays, ignore_eos)
    corpus        semianalysisai/cc-traces-weka-062126, uncapped
    concurrency   declared by the candidate in `concurrency.json`
    seed          a fresh draw per run: even in development, odd in the verdict
    window        1800 s in the verdict; a development measurement chooses
                  300-1800 s

The replay is closed-loop against a wall clock, so how far each session gets
inside the window depends on the system under test: request counts and length
distributions are properties of the run, recorded by `scenario.py` per arm.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path

from .. import spine

#: The AgentX scenario. A lock, not a suggestion: it requires one of its own
#: loaders and refuses any other.
SCENARIO = "inferencex-agentx-mvp"

#: THE FULL, UNCAPPED CORPUS. 393 real Claude Code sessions, 58,495 turns,
#: input up to 989,824 tokens -- the workload
#: <https://inferencex.semianalysis.com/inference> reports against.
CORPUS = "semianalysisai/cc-traces-weka-062126"
CORPUS_SHA256 = (
    "29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276")
LOADER = "semianalysis_cc_traces_weka_062126"

#: The longest prompt the corpus offers, measured over all 58,495 turns. The
#: near-maximum context probe builds its prompt from this, so it lives beside
#: the corpus it is a fact about.
CORPUS_MAX_PROMPT_TOKENS = 989_824

#: How requests arrive: the sessions' own recorded inter-turn delays, which
#: the scenario refuses to let a server skip.
ARRIVAL_PROCESS = "agentic_replay"

#: Named so it can be refused by name. The 256k-capped sibling is in the
#: scenario's allowed-loader list and is NOT used by any task here: capping
#: removes 28,354 of 58,495 turns, every one of them a turn whose input exceeds
#: the model's native 262,144 context -- which is exactly the population that
#: forces eviction and re-prefill. Capping therefore makes the workload look
#: markedly more profitable and shortens its TTFT tail.
CAPPED_SIBLING_NOT_USED = "semianalysis_cc_traces_weka_062126_256k"

DTYPE = "bfloat16"

#: The final verdict's window, and the default development window.
VERDICT_DURATION_S = 1800
EVOLVE_DURATION_S = 1800

#: A development measurement may be shorter: the agent decides how long to
#: measure. Tail latency on this stateful replay depends on run length, so a
#: shorter window is noisier and can miss the cache-eviction tail; the agent
#: is told so, and the verdict always runs `VERDICT_DURATION_S`.
DEVELOPMENT_MIN_DURATION_S = 300

#: WHAT THE SCENARIO ITSELF REFUSES BELOW. Not a task's choice.
#:
#: `inferencex-agentx-mvp` sets `min_benchmark_duration_seconds=900`, and a run
#: under it needs `--unsafe-override` -- which makes aiperf stamp
#: `submission_valid=false`, so the evidence can never be scored.
#:
#: This is distinct from the full development/verdict window above.  Only a
#: diagnostic below the scenario floor needs `--unsafe-override`.
SCENARIO_DURATION_FLOOR_S = 900

#: NO PUBLIC SEED. Every development measurement draws a fresh workload seed
#: (`development_seed`, even) and every final verdict draws its own
#: (`final_seed`, odd), so a verdict never replays a request mix the agent
#: measured. The scenario enters each session at a random depth, so each draw
#: is a different ~100-200-turn sample of a heavy-tailed corpus: compare
#: systems over repeated runs, not one draw. Kept as a name for task modules
#: that restate it.
RANDOM_SEED = None

#: THE TIMEOUTS THE DEFAULTS DO NOT ALLOW FOR.
#:
#: 393 sessions are rebuilt from their KV block hashes before a single request
#: is sent. That is on the order of 80 s of parallel reconstruction plus
#: assembly, and the worker waits on `_dataset_configured_event` with a 300 s
#: default, raising a bare `asyncio.TimeoutError` whose `str()` is EMPTY -- so
#: the failure reads as
#:
#:     Failed to handle command profile_configure with hook ... :
#:
#: with nothing after the colon. `PROFILE_CONFIGURE_TIMEOUT` must be >=
#: `CONFIGURATION_TIMEOUT`; an aiperf validator enforces it, so both move.
ENVIRONMENT = {
    "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "1800",
    "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "1800",
}


#: CANDIDATE-DECLARED CONCURRENCY (tasks whose module sets
#: `CONCURRENCY_POLICY = "candidate"`). The published tree carries
#: `concurrency.json` = {"offered_concurrency": N}; the verdict and every
#: development measurement offer the scenario at N. The instruction states the
#: SGLang operating point (`REFERENCE_CONCURRENCY`) as a reference only.
CANDIDATE_CONCURRENCY_FILE = "concurrency.json"
CANDIDATE_CONCURRENCY_MIN = 1
CANDIDATE_CONCURRENCY_MAX = 256


def candidate_concurrency(tree: str | Path) -> int:
    """The offered load a candidate tree declares, or ValueError saying why not."""
    path = Path(tree) / CANDIDATE_CONCURRENCY_FILE
    try:
        body = json.loads(path.read_text("utf-8"))
    except OSError as exc:
        raise ValueError(
            f"{path} is missing ({type(exc).__name__}); this task offers the "
            f"load the candidate declares there as "
            f'{{"offered_concurrency": N}}') from None
    except ValueError as exc:
        raise ValueError(f"{path} is not JSON: {exc}") from None
    value = body.get("offered_concurrency") if isinstance(body, dict) else None
    if (type(value) is not int
            or not CANDIDATE_CONCURRENCY_MIN <= value <= CANDIDATE_CONCURRENCY_MAX):
        raise ValueError(
            f"{path}: offered_concurrency must be an integer in "
            f"[{CANDIDATE_CONCURRENCY_MIN}, {CANDIDATE_CONCURRENCY_MAX}], "
            f"got {value!r}")
    return value


#: DISJOINT SEED SPACES: development draws are even, final draws odd.
def development_seed() -> int:
    return secrets.randbits(62) << 1


def final_seed() -> int:
    return secrets.randbits(63) | 1


#: Body fields every timed request carries. Greedy decoding makes the output
#: a property of the checkpoint; the native token ids and special tokens make
#: it auditable; the non-thinking template pins the prompt prefix. JSON
#: rather than aiperf's `key:value` spelling, which can lose a numeric zero.
TIMED_REQUEST_FIELDS = {
    "temperature": 0,
    "return_token_ids": True,
    "skip_special_tokens": False,
    "chat_template_kwargs": {"enable_thinking": False},
}


def invocation(url: str, tokenizer: str, artifact_dir: str,
               duration_s: int = VERDICT_DURATION_S,
               concurrency: int | None = None,
               max_context_length: int | None = None,
               tokenizer_revision: str | None = None,
               random_seed: int | None = None) -> list[str]:
    """The one command that measures these tasks, wherever it is run from.

    The agent's in-Harbor copy, the verdict and an operator all read this list.
    `concurrency` is the offered session load; AGENTIC_REPLAY governs delays
    within each session. `random_seed` defaults to a fresh development draw.

    This list never carries `--unsafe-override`. `scenario.measure_one` adds
    it to a development window shorter than the scenario floor and to nothing
    else; AgentX then stamps `submission_valid=false`, which only that
    development report accepts and the verdict never does.
    `--hf-weka-dataset` reaches the full corpus; `--public-dataset` could
    select the 256k-capped sibling instead.
    """
    if not DEVELOPMENT_MIN_DURATION_S <= int(duration_s) <= VERDICT_DURATION_S:
        raise ValueError(
            f"an AgentX performance window is {DEVELOPMENT_MIN_DURATION_S}-"
            f"{VERDICT_DURATION_S} s; got {duration_s}")
    seed = development_seed() if random_seed is None else int(random_seed)
    if concurrency is None or int(concurrency) < 1:
        raise ValueError(
            "offered concurrency is undecided: pass the load the candidate "
            "declares in concurrency.json")
    return [
        "aiperf", "profile",
        "--scenario", SCENARIO,
        "--hf-weka-dataset", CORPUS,
        "--model", "target", "--url", url,
        "--endpoint-type", "chat", "--streaming",
        "--tokenizer", tokenizer,
        *(["--tokenizer-revision", tokenizer_revision]
          if tokenizer_revision else []),
        # TIMEBOX, never count. `--benchmark-duration` is the stop rule for
        # every run this suite takes, and `--request-count` is never passed.
        #
        # The objective has GPU-hours in its denominator, so a fixed box makes
        # the denominator a constant and the measurement becomes purely "how
        # much work for the same money". A count rule would fix the numerator
        # and let cost float, which measures the inverse and is not comparable
        # with any number already recorded.
        #
        # It is also not really optional here: the scenario auto-fills
        # `default_benchmark_duration_seconds=1800` onto any profiling phase
        # whose duration is unset, so `--request-count` alone would still run
        # under an 1800 s box and merely add an earlier stop. The scenario
        # refuses a window under its 900 s floor unless `--unsafe-override`
        # is passed, which only a short development measurement does.
        "--benchmark-duration", str(int(duration_s)),
        "--concurrency", str(int(concurrency)),
        "--extra-inputs", json.dumps(TIMED_REQUEST_FIELDS,
                                     separators=(",", ":")),
        # Always explicit, so the run's seed is on its own recorded command.
        "--random-seed", str(seed),
        "--output-artifact-dir", artifact_dir,
        "--export-level", "raw", "--export-outputs-json",
        "--ui", "simple",
        # A HARDWARE CEILING, DECLARED BY THE TASK, NOT A CONVENIENCE.
        #
        # Weka loaders DROP WHOLE TRACES whose recorded peak prompt+output
        # exceeds this, at load time. That is a filter, not a truncation, so it
        # does not touch `forbid_input_truncation`: every turn that IS replayed
        # is replayed in full, and the ones that are not are absent rather than
        # shortened. A truncating alternative would silently change what each
        # request asks for and report it under the task's name.
        #
        # Only tasks whose model cannot hold the corpus set this.  The task's
        # contract carries its current ceiling.  `--context-length` alone is
        # not equivalent: it sets a positional limit while the KV pool may set
        # a lower practical limit.
        *(["--max-context-length", str(int(max_context_length))]
          if max_context_length else []),
    ]


def refusal(task_id: str) -> str:
    """Why no program but `scenario.py` can measure this task."""
    return (
        f"{task_id}: this task's workload is the AgentX scenario `{SCENARIO}` "
        f"(recorded inter-turn delays, its own loader and duration floor, "
        f"ignore_eos). Its verdict is `final_validate/scenario.py`, which "
        f"offers it through `invocation()` and decides against the frozen "
        f"service levels.")


def module(*, task_id: str, profile: str,
           reported_objective: str = "dollars_per_dollar",
           workload_shape: str | None = None) -> spine.TaskModule:
    """A `TaskModule` for one task over the shared AgentX workload.

    `profile` is the task's own title; `workload_shape` is its one-line
    description of the workload it replays.
    """
    return spine.TaskModule(
        task_id=task_id,
        workload_shape=workload_shape or (
            f"AgentX {SCENARIO} over {CORPUS}. Profile: {profile}"),
        agreement=spine.REPLAY_FIDELITY,
        refuses=refusal(task_id),
        reported_objective=reported_objective,
    )
