# Running ServerBench on Google Colab (one GPU)

[繁體中文](COLAB.zh-TW.md)

This fork adds a third environment backend, **local**, beside Harbor/docker and
Modal. It runs an evolve on the machine the driver is on, with no container
runtime, which is what a Colab runtime is: root, one GPU, `CAP_SYS_ADMIN`, but
no docker and no `CAP_NET_ADMIN`.

The notebook is [`serverbench/scripts/colab/ServerBench_Colab.ipynb`](serverbench/scripts/colab/ServerBench_Colab.ipynb).
Open it in Colab with a GPU runtime (**L4** by default; see *Switching GPUs*) and run it top to bottom.

## What stays the same

Everything above "where the agent's processes live": the task brief, the
workspace contents, `./version`, `./full-test`, `./measurements`, the journal,
the trajectory, the AgentX replay, the private correctness stage and the
scoring. The verdict reuses the Modal backend's in-sandbox worker
(`final_validate/modal_worker.py`) unchanged.

## What the local backend does instead of docker

| docker (Harbor) | local |
|---|---|
| container with an iptables allowlist (`restrict-network.sh`) | the agent runs in its own **network namespace** with only `lo` (`serving_evolve/local_sandbox.py`); no route, no DNS |
| egress broker on the docker bridge | the same broker on a **unix socket**, relayed to `127.0.0.1:3128` inside the namespace; it allows CONNECT to the provider's hosts only and logs every attempt |
| `USER agent` in the image | the `agent` user (uid ≠ 0), entered with `nsenter --setuid` |
| bind mounts | bind mounts for paths inside the repository when the host allows `mount`, symlinks otherwise; the model and HF cache under `/opt/serverbench` are always symlinked |
| the host's files are not in the container | no mount namespace: private directories (verdict evidence, answer pool, secrets, other runs, published lineages) are made root-only, and the repository root too when binds work |
| verdict in a fresh container | verdict in two fresh sandboxes (`final_validate/local_verdict.py`), no egress at all, after every process of the agent user is killed and confirmed gone |
| `lmsysorg/sglang:v0.5.19` image | SGLang 0.5.19 in its own interpreter (`/opt/serverbench/sglang-venv`) |
| the agent image's python, CUDA 12.8 and Nsight | `/opt/serverbench/agent-python` (the image's pins, from the frozen wheelhouse), and the image's own CUDA 12.8 / Nsight `.deb`s with their checksums |

`serverbench/scripts/colab/selfcheck.py` proves the isolation on the machine it
runs on: not root, no direct internet, only the provider through the broker,
private directories unreadable, loopback and GPU working, nothing left behind.

## Order of work

1. `serverbench/scripts/colab/setup.sh` -- per session (a Colab VM is discarded).
2. `serverbench/scripts/colab/persist.sh restore` -- earlier results, from the
   rclone remote.
3. `serverbench/scripts/colab/calibrate.py search | seeds | record` -- once per
   task: the SGLang operating point and the p90 bars **on the L4**
   (`docs/harness/bar-derivation.md`). The L4 task
   `qwen3_5_2b_standard_1xL4` ships with provisional bars (4100 ms / 35 ms)
   until this runs; `record` replaces them and sets the reference concurrency.
   Every point is resumable.
4. `serverbench/scripts/server_evolve/local-evolve.sh <task> --hours 3` -- once
   per agent (`AGENT_CLI=claude|codex`).

## Switching GPUs

Supported, with one rule: **one task per GPU type, calibrated on that card**.
A task's p90 bars are SGLang's on its card and its dollars are that card's
hours; an L4 task run on an A100 would score A100 numbers against L4 bars.
Nothing upstream compares the card a run gets with the card its task names,
so `serverbench/scripts/colab/hardware.py` does: `local-evolve.sh`,
`verdict-only.sh` and `calibrate.py search|seeds` refuse a mismatch
(`SB_ALLOW_HARDWARE_MISMATCH=1` overrides, for debugging only).

| Colab GPU | task suffix | status |
|---|---|---|
| L4 (24 GB) | `L4` | supported; built and tested for (sm_89) |
| A100 40 GB | `A100` | expected to work (Ampere, bf16); not run end to end |
| A100 80 GB | `A10080GB` | as above; a separate task (different memory, different SGLang defaults) |
| H100 80 GB | `H100` | expected to work (Hopper) |
| RTX PRO 6000 (G4) | `RTXPRO6000` | experimental: Blackwell sm_120, SGLang 0.5.19 kernels unverified |
| T4 (16 GB) | `T4` | unsupported: no bf16 on Turing (`SB_ALLOW_UNSUPPORTED_GPU=1` tries anyway) |

Steps (the notebook does them): pick the runtime's GPU; leave `TASK = None`,
which becomes `qwen3_5_2b_standard_1x<suffix>` for the detected card; set its
hourly rate in `GPU_RATE`; after setup and restore, cell 5b runs
`hardware.py new-task`, which copies the L4 task (model, token prices,
framework flags, correctness settings) with the new hardware and rate and
saves it to the remote; then **calibrate** (section 6) before comparing
agents. By hand:

```
python3 serverbench/scripts/colab/hardware.py detect
serverbench/.venv/bin/python serverbench/scripts/colab/hardware.py new-task --usd-per-gpu-hour 1.50
export TASK=$(python3 serverbench/scripts/colab/hardware.py task)
```

A new task's files are not in git; `persist.sh` keeps them. To commit one,
also pin its bars in `serverbench/tests/contracts/test_the_p90_bars.py`.

## Credentials: subscriptions on both

| agent | subscription | API key |
|---|---|---|
| Claude Code | `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token` | `ANTHROPIC_API_KEY` |
| Codex | `codex login --device-auth` in the notebook, or your `~/.codex/auth.json` as `CODEX_AUTH_JSON` | `OPENAI_API_KEY` |

## Disconnects and checkpoints

A browser disconnect or a kernel restart does not stop a run (it is started
with `setsid nohup`); `reattach()` in the notebook follows it again. A
reclaimed VM loses its disk, so with an rclone config (`RCLONE_CONF`, a remote
named `sb`: Google Drive with `scope=drive.file`, or a GCS bucket)
`local-evolve.sh` starts `serverbench/scripts/colab/checkpoint.py` beside the
driver. It saves the run every 10 minutes and soon after every new version,
never while a measurement runs, and once more when the driver exits; it also
copies the agent's Claude transcripts to `<run>/claude-projects`, where a
make-up run (`run.py --resume-run`) looks for them.

It uploads as root from the host network namespace, with nothing mounted: the
agent's sandbox gains no route and no path. rclone and its token are
root-only, symlinks are skipped rather than followed (a run directory holds
files the agent wrote), and every `secrets/` directory is excluded.

If the VM goes during the final verdict, restore and run
`serverbench/scripts/colab/verdict-only.sh [RUN_DIR]`: the verdict alone, on
the run's accepted head, with no agent and no API cost.

## Known differences from the paper's setup

* The hardware is a Colab GPU (an L4 by default: sm_89, 24 GB, ~300 GB/s), not
  an L40; numbers are comparable only with runs of the same task (same card).
* The hourly rate in `contract.yaml` (0.80 USD by default) is an operator-set
  placeholder. It scales every run's `dollars_per_dollar` by the same factor,
  so it does not change a comparison between agents.
* The agent can see the host filesystem (with an unprivileged user's
  permissions); docker would show it only the container's.
* `ncu` may need profiling permissions the Colab VM does not grant to a
  non-root user.
* After calibration, update the pinned bars for the L4 task in
  `serverbench/tests/contracts/test_the_p90_bars.py` (`P90_ONLY`).

## Fixes

The plain reference (`final_validate/private_correctness.py`) runs SGLang with
`--enable-deterministic-inference` on Triton, which cuts prefill chunks only at
multiples of 4096 tokens. SGLang's default chunk on a card under 35 GB (the L4)
is 2048, so any prompt longer than one chunk was never scheduled and the EOS
audit's long teacher-forced requests wedged the verdict. The plain reference's
chunk is now pinned to 4096; boundaries are aligned either way, so the scored
logprobs do not change.

Host-side task root (a fix to the local backend itself). Without
`SERVERBENCH_TASK_ROOT`, the harness guesses `/workspace/task_context` as its
task root. A docker host has no `/workspace`, but the local backend placed one
on the host, pointing at the agent's writable workspace, and left it after the
run -- so a host tool, the driver and the verdict included, could read the
agent's copy of the task contract. Every host entry point now pins
`SERVERBENCH_TASK_ROOT=<repo>/serving_tasks`, and `local_sandbox.terminate`
removes the symlinks and binds `local_env` placed.
