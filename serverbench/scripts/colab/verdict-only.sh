#!/bin/bash
# Run (or re-run) ONLY the final verdict of a finished evolve, on this machine.
#
# For when the agent's part is done but the verdict never finished: the VM was
# reclaimed during it, or the driver died. Restore the run first
# (`persist.sh restore`), then:
#
#   bash serverbench/scripts/colab/verdict-only.sh [RUN_DIR] [--force]
#
# RUN_DIR defaults to the newest run of $TASK. The verdict is the same one the
# driver runs (`final_validate/tools/validate_evolve.py`, local backend, fresh
# sandboxes, no egress) on the run's accepted head. It costs GPU time and no
# API calls. Refuses when the run already has a verdict, unless --force.
# Afterwards the results are saved to the remote when an rclone config exists.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
TASK="${TASK:-qwen3_5_2b_standard_1xL4}"
RUN="" FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        *) RUN="$arg" ;;
    esac
done

[ "$(id -u)" = 0 ] || { echo "the local backend runs as root" >&2; exit 2; }
if mountpoint -q /content/drive 2>/dev/null; then
    echo "Google Drive is mounted at /content/drive; unmount it first." >&2; exit 2
fi
if [ -z "$RUN" ]; then
    RUN="$(ls -td "$REPO/serverbench/evolve_space/evolve/${TASK,,}"/*/ 2>/dev/null | head -1)"
fi
RUN="${RUN%/}"
[ -f "$RUN/journal.jsonl" ] || { echo "not a run directory: $RUN" >&2; exit 2; }

PY="$REPO/serverbench/.venv/bin/python"
LINEAGE="$("$PY" -c 'import json,sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("event") == "run_created":
        print(r["lineage_id"]); break' "$RUN/journal.jsonl")"
FINAL="$REPO/server_output/$LINEAGE/FINAL.json"
if [ -f "$FINAL" ] && [ "$FORCE" != 1 ]; then
    echo "this run already has a verdict: $FINAL (pass --force to measure again)" >&2
    exit 1
fi

SB=/opt/serverbench
export HF_HOME="${HF_HOME:-$SB/hf}" PATH="$SB/bin:$PATH"
module="$REPO/serverbench/harness/final_validate/tasks/${TASK,,}.py"
id="$(sed -n 's/^MODEL_ID = "\(.*\)"$/\1/p' "$module")"
rev="$(sed -n 's/^MODEL_REVISION = "\(.*\)"$/\1/p' "$module")"
MODEL_DIR="${MODEL_DIR:-$HF_HOME/hub/models--${id//\//--}/snapshots/$rev}"
[ -f "$MODEL_DIR/config.json" ] || { echo "no model at $MODEL_DIR (run setup.sh)" >&2; exit 2; }

# Nothing of the agent's may be alive on the card the verdict will measure.
PYTHONPATH="$REPO/serverbench/harness" "$PY" -I -c '
import sys; sys.path.insert(0, sys.argv[1])
from serving_evolve import local_sandbox as s
left = s.kill_agent_processes(timeout=10)
sys.exit(1 if left else 0)' "$REPO/serverbench/harness" \
    || { echo "agent processes survived; refusing to measure" >&2; exit 1; }

PYTHONPATH="$REPO/serverbench/harness" "$PY" -c '
import sys; sys.path.insert(0, sys.argv[1])
from lifecycle.record import Journal
from pathlib import Path
Journal(Path(sys.argv[2])).append("final_validation_rerun", force=sys.argv[3] == "1",
    why="the verdict is run on its own, after the evolve: the driver did not finish it")
' "$REPO/serverbench/harness" "$RUN" "$FORCE"

printf '%s verdict for %s (lineage %s)\n' "$(date -u '+%FT%TZ')" "$(basename "$RUN")" "$LINEAGE"
SERVERBENCH_BACKEND=local PYTHONPATH="$REPO/serverbench/harness" \
    "$PY" serverbench/harness/final_validate/tools/validate_evolve.py \
        --run "$RUN" --model-dir "$MODEL_DIR" --gpu "${GPU:-0}"
rc=$?
printf '%s verdict finished rc=%s\n' "$(date -u '+%FT%TZ')" "$rc"
if [ -f "$REPO/serverbench/artifacts/secrets/rclone.conf" ]; then
    TASK="$TASK" bash serverbench/scripts/colab/persist.sh save || true
fi
exit "$rc"
