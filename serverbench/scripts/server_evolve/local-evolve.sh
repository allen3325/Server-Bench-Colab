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
SERVERBENCH_BACKEND=local GPU="${GPU:-0}" MODEL_DIR="$MODEL_DIR" \
    TASK_ROOT="$REPO/serving_tasks" \
    bash serverbench/scripts/server_evolve/evolve-task.sh "$TASK" \
        --hours "$HOURS" --stop timebox "$@"
rc=$?
printf '%s run finished rc=%s\n' "$(date -u '+%FT%TZ')" "$rc"
exit "$rc"
