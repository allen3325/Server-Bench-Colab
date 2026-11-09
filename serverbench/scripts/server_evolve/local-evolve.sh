#!/bin/bash
# Run one serving-system evolve on THIS machine's GPU, with no docker: the
# agent's environment is a namespace sandbox (`serving_evolve/local_sandbox.py`)
# and the final verdict runs in fresh sandboxes on the same card
# (`final_validate/local_verdict.py`). Written for a Google Colab runtime set up
# by `serverbench/scripts/colab/setup.sh`.
#
#   AGENT_CLI=claude AGENT_MODEL=claude-opus-5 AGENT_ENV=<env file> \
#       bash serverbench/scripts/server_evolve/local-evolve.sh <task> --hours 3
#
#   AGENT_CLI=codex  AGENT_MODEL=gpt-5.6-sol AGENT_EFFORT=max \
#       bash serverbench/scripts/server_evolve/local-evolve.sh <task> --hours 3
#
# Everything else -- the brief, versions, ./full-test, the journal, the
# trajectory, the verdict's scoring -- is the same code as the docker path.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

TASK="${1:?usage: local-evolve.sh <task> [--hours N] [entrypoint args...]}"
shift
HOURS="3"
if [ "${1:-}" = "--hours" ]; then HOURS="$2"; shift 2; fi

[ "$(id -u)" = 0 ] || { echo "the local backend runs as root" >&2; exit 2; }
[ -d "$REPO/serving_tasks/$TASK" ] || { echo "no such task: $TASK" >&2; exit 2; }

# The repository's tasks, never a guessed root: `spine` would otherwise look at
# /workspace/task_context -- the agent's writable copy -- whenever one shows.
export SERVERBENCH_TASK_ROOT="$REPO/serving_tasks"
# A task's bars and dollars are its own card's.
"$REPO/serverbench/.venv/bin/python" "$REPO/serverbench/scripts/colab/hardware.py" check "$TASK" || exit 2
SB=/opt/serverbench
export HF_HOME="${HF_HOME:-$SB/hf}"
export PATH="$SB/bin:$PATH"
if [ -z "${MODEL_DIR:-}" ]; then
    module="$REPO/serverbench/harness/final_validate/tasks/${TASK,,}.py"
    id="$(sed -n 's/^MODEL_ID = "\(.*\)"$/\1/p' "$module")"
    rev="$(sed -n 's/^MODEL_REVISION = "\(.*\)"$/\1/p' "$module")"
    MODEL_DIR="$HF_HOME/hub/models--${id//\//--}/snapshots/$rev"
fi
[ -f "$MODEL_DIR/config.json" ] || { echo "no model at $MODEL_DIR (run setup.sh)" >&2; exit 2; }

if mountpoint -q /content/drive 2>/dev/null; then
    echo "Google Drive is mounted at /content/drive. Unmount it before a run" \
         "(from google.colab import drive; drive.flush_and_unmount()):" \
         "the agent must not see your Drive." >&2
    exit 2
fi

printf '%s %s locally, %s h, %s %s\n' "$(date -u '+%FT%TZ')" "$TASK" "$HOURS" \
    "${AGENT_CLI:-claude}" "${AGENT_MODEL:-}"
started="$(date +%s)"
# The driver gets its own session, so a signal to this script's process group
# does not ALSO reach it directly: it must see exactly one stop. A second
# signal lands in its evidence-preserving teardown and cuts that short,
# leaving the sandbox standing.
SERVERBENCH_BACKEND=local GPU="${GPU:-0}" MODEL_DIR="$MODEL_DIR" \
    TASK_ROOT="$REPO/serving_tasks" \
    setsid bash serverbench/scripts/server_evolve/evolve-task.sh "$TASK" \
        --hours "$HOURS" --stop timebox "$@" &
driver=$!
echo "$driver" > /run/serverbench-local-driver.pid
forwarded=0
forward_stop() {
    [ "$forwarded" = 1 ] && return
    forwarded=1
    kill -TERM "$driver" 2>/dev/null
}
trap forward_stop TERM INT HUP

# Off-VM checkpoints while the run lasts (serverbench/scripts/colab/checkpoint.py):
# a Colab VM can be reclaimed mid-run, and its disk goes with it. Root, host
# network, no mount -- the agent's sandbox is untouched. Skipped without an
# rclone config; SB_CHECKPOINT=0 turns it off.
checkpointer=""
if [ "${SB_CHECKPOINT:-1}" != 0 ] && [ -f "$REPO/serverbench/artifacts/secrets/rclone.conf" ]; then
    "$REPO/serverbench/.venv/bin/python" serverbench/scripts/colab/checkpoint.py \
        --task "$TASK" --driver-pid "$driver" --since "$started" \
        --interval "${SB_CHECKPOINT_INTERVAL:-600}" &
    checkpointer=$!
else
    echo "no off-VM checkpoints (no rclone config): a reclaimed VM loses this run"
fi

# `wait` returns early when a trapped signal arrives; keep waiting for the
# driver, which handles the signal itself and preserves the run.
rc=0
while kill -0 "$driver" 2>/dev/null; do
    wait "$driver"; rc=$?
done
[ -n "$checkpointer" ] && wait "$checkpointer"
rm -f /run/serverbench-local-driver.pid
printf '%s run finished rc=%s\n' "$(date -u '+%FT%TZ')" "$rc"
exit "$rc"
