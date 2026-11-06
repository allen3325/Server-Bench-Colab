# Docker mode

The run happens in a **local docker container** on a host that has the cards.
Harbor's `DockerEnvironment` stands it up; `serverbench/scripts/agent/harbor_env.py`
composes it.

```bash
bash serverbench/scripts/server_evolve/evolve-task.sh <TASK> --hours 8 --stop timebox
```

| what | where it comes from |
|---|---|
| the cards | this host's own, picked by `GPU=auto` and excluded with `EXCLUDE_GPUS` |
| the image | `serverbench/evolve-agent:cu128`, built by `serverbench/scripts/deploy/03-build-images.sh` and refused if it was not built from this checkout's Dockerfiles |
| the checkpoint | the host's Hugging Face cache, mounted read-only by exact repository slug |
| the corpus | the same cache, mounted by exact dataset slug -- the task's `CORPUS` and nothing else |
| the offline build inputs | `serverbench/artifacts/env-input`, mounted at `/opt/env-input` |
| the way out | the run's model provider only. An egress broker relays HTTPS to that one provider's host -- the `ANTHROPIC_BASE_URL` host for a third-party model, Anthropic's for an Anthropic model -- records every attempt, and refuses everything else; the container carries that provider's one credential and no other |

**The hardware suffix is this host's card.** `_2xL40`, `_1xL40`: whatever
`nvidia-smi` reports. Nothing checks the name against a catalogue, because the
cards are the ones in the machine.

**A run's container is removed at teardown;** what the agent built survives in
the run directory, its published versions and the lineage, where it can still
be measured by hand. Nothing in this mode deletes an image, a model cache or a
wheelhouse: they are shared with every other run on the host, and removing
them because one campaign ended would take weights out from under a neighbour.

Before a task's first run on this hardware, measure SGLang's operating point
here and record it with `taskctl set <task> --reference-concurrency N`; without
it the instruction states no reference point.
