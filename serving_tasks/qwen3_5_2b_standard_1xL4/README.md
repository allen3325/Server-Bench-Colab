# qwen3_5_2b_standard_1xL4 -- Qwen3.5 2B Standard (L4)

## What this task is

A serving-system task. A coding agent is given a correctness-first Hugging
Face server -- one request at a time under a global lock, plain
`generate()`, no cross-request prefix cache -- and one frozen serving problem:
serve `Qwen/Qwen3.5-2B` at revision `15852e8c16360a2fea060d615a32b45270f8a8fc` on
1xL4 (1 device(s)) for 1800
seconds of real traffic. The agent owns the scheduler, the batching, the KV
cache, the kernels and the offered load; the harness owns the task, the
environment and the measurement.

## Background

The traffic is `semianalysisai/cc-traces-weka-062126`, replayed by the AgentX scenario
`inferencex-agentx-mvp` (a fresh random seed on every run): real coding sessions, offered at the sessions'
own recorded inter-turn delays. Requests are long, reusable conversational
contexts -- most of a turn's input is the previous turn's context plus a
little -- so what a serving system pays for is the prefill it cannot reuse, and
what it earns is the tokens it serves inside the p90 bars. Traces whose peak prompt and output exceed 262,144 tokens -- the model's native window -- are dropped whole at load; every replayed turn is served in full.

The cards are allocated and billed for the whole window whether they are busy
or idle. Their internal organisation is the candidate's choice.

## The goal

Maximise **`dollars_per_dollar`**: sum the input and output token value of
completed profiling requests whose TTFT and TPOT both meet the p90 bars,
then divide by the whole-window GPU cost (policy `request_p90_billable_tokens_v4`).
Missed, errored and cancelled requests earn nothing. Request-count attainment
and `gross_dollars_per_dollar` are diagnostics and decide nothing.
There is no hard latency gate, but a request outside either bar earns nothing,
so latency decides how much of the value counts. There are three separate hard
gates -- the replay gate, the EOS gate and the private correctness gate -- and
each must hold. A system the reviewer finds hacked scores 0.

The offered session concurrency is declared by the candidate itself in
`concurrency.json`; no SGLang reference operating point is stated.

## The p90 bars

| percentile | time to first token | time per output token |
|---|---|---|
| **p90** | <= **4100 ms** | <= **35 ms** |

## The three hard gates

1. **Replay gate.** Every turn the scenario offers is answered with exactly the
   output token count its trace records. The workload sets `ignore_eos`, so
   the server is told how many tokens to produce; fewer tokens, a dropped
   turn, a truncated input, or text lost between the tokens generated and the
   text streamed is a failure.
2. **EOS gate.** Every output token must come from the checkpoint's own
   computation, including past an end-of-sequence token under `ignore_eos`;
   padding after an early stop is fabricated output. Without `ignore_eos` the
   server must stop at end-of-sequence; the agent's `--check-protocol-context`
   and `./full-test` check this rule and report it separately.
3. **Correctness gate.** The checkpoint's own function at its own precision
   with greedy decoding. Bit-identical output is not required: it is judged
   within a tolerance for the numerical differences batching, kernels and
   speculative decoding legitimately introduce. It is verified privately after
   the run; its method is not disclosed to the agent and is not in the
   delivered evaluator. The agent is told to measure its own accuracy inside
   its container, and to prove the published tree with `./full-test`.

## What to read

| path | what it is |
|---|---|
| `instruction.md` | **the specification a coding agent is given**, and the only document it receives |
| `contract.yaml` | the objective and its request scoring, the p90 bars, token prices, devices, hourly rate, and scored window |
| `reference.yaml` | the SGLang concurrency search and the reference arms with their launch lines; its private correctness block is stripped from the delivered copy |
| `demo/` | the starter the agent begins from and its entry point `serve.sh`; it declares no offered concurrency -- the agent writes `concurrency.json` |
| `tests/final_verdict/` | the evaluator's measurement code, byte-for-byte; the private correctness stage is host-only |
| `task.toml` | the Harbor task definition |

## How it is judged

Once, after the run's clock stops, outside the container, on the final version
the agent published. The verdict starts that tree by its own `serve.sh` with
the checkpoint mounted, runs the complete 1800-second scenario at the
tree's declared concurrency, verifies output correctness privately, scores
every request against the p90 bars, and reports `HOLDS`, `MISSES` or
`NOT_RUN`. The agent never sees the result.

Every number in this file is generated from `contract.yaml`, `reference.yaml`
and the task's own module.
