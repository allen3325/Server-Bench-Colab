# Running ServerBench on Google Colab (one L4)

This fork adds a third environment backend, **local**, beside Harbor/docker and
Modal. It runs an evolve on the machine the driver is on, with no container
runtime, which is what a Colab runtime is: root, one GPU, `CAP_SYS_ADMIN`, but
no docker and no `CAP_NET_ADMIN`.

The notebook is [`serverbench/scripts/colab/ServerBench_Colab.ipynb`](serverbench/scripts/colab/ServerBench_Colab.ipynb).
Open it in Colab with an **L4** runtime and run it top to bottom.

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
2. `serverbench/scripts/colab/calibrate.py search | seeds | record` -- once per
   task: the SGLang operating point and the p90 bars **on the L4**
   (`docs/harness/bar-derivation.md`). The L4 task
   `qwen3_5_2b_standard_1xL4` ships with provisional bars (4100 ms / 35 ms)
   until this runs; `record` replaces them and sets the reference concurrency.
   Every point is resumable.
3. `serverbench/scripts/server_evolve/local-evolve.sh <task> --hours 3` -- once
   per agent (`AGENT_CLI=claude|codex`).
4. `serverbench/scripts/colab/persist.sh save|restore` -- results to and from
   Google Drive. Drive must be **unmounted** while an agent runs;
   `local-evolve.sh` refuses to start otherwise.

## Known differences from the paper's setup

* The hardware is an L4 (sm_89, 24 GB, ~300 GB/s), not an L40; numbers are
  comparable only with other runs of the L4 task.
* The L4 hourly rate in `contract.yaml` (0.80 USD) is an operator-set
  placeholder. It scales every run's `dollars_per_dollar` by the same factor,
  so it does not change a comparison between agents.
* The agent can see the host filesystem (with an unprivileged user's
  permissions); docker would show it only the container's.
* `ncu` may need profiling permissions the Colab VM does not grant to a
  non-root user.
* After calibration, update the pinned bars for the L4 task in
  `serverbench/tests/contracts/test_the_p90_bars.py` (`P90_ONLY`).
