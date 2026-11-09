#!/bin/bash
# Save this runtime's ServerBench results off the VM, or restore them into a
# fresh runtime. A Colab VM is discarded at the end of every session; the
# remote is what survives.
#
#   bash serverbench/scripts/colab/persist.sh save    [REMOTE]
#   bash serverbench/scripts/colab/persist.sh restore [REMOTE]
#   bash serverbench/scripts/colab/persist.sh status  [REMOTE]
#
# Uploads with rclone, as root, from the HOST network namespace: nothing is
# mounted, so the agent never sees the remote, and its sandbox gains no route.
# The rclone config (the remote's token) lives in the root-only secrets dir.
# REMOTE defaults to $SB_REMOTE, else `sb:serverbench-colab` -- the `sb` remote
# of that config (Google Drive or a GCS bucket; see the notebook).
#
# What crosses: results, lineages, run directories, the calibrated task. What
# never does: any `secrets/` directory (a run stages its login there), caches,
# virtualenvs, model weights -- and symlinks, which are skipped rather than
# followed: a run directory holds files the agent wrote, and a link it planted
# must not make a root process read something the agent cannot.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VERB="${1:?usage: persist.sh save|restore|status [REMOTE]}"
REMOTE="${2:-${SB_REMOTE:-sb:serverbench-colab}}"
TASK="${TASK:-qwen3_5_2b_standard_1xL4}"
CALIBRATION="${CALIBRATION:-/content/sb-calibration}"
RCLONE="${RCLONE:-/root/bin/rclone}"
export RCLONE_CONFIG="${RCLONE_CONFIG:-$REPO/serverbench/artifacts/secrets/rclone.conf}"

[ -x "$RCLONE" ] || { echo "no rclone at $RCLONE (run setup.sh)" >&2; exit 2; }
[ -f "$RCLONE_CONFIG" ] || { echo "no rclone config at $RCLONE_CONFIG (set the RCLONE_CONF secret)" >&2; exit 2; }

FILTERS=(--skip-links
         --exclude "secrets/**" --exclude ".venv/**" --exclude "uv-cache/**"
         --exclude ".uv-cache/**" --exclude "__pycache__/**"
         --exclude ".pytest_cache/**" --exclude "*.safetensors")
# Gentle by default: a checkpoint can land while a full test measures.
TUNING=(--transfers "${SB_TRANSFERS:-2}" --checkers 4
        --bwlimit "${SB_BWLIMIT:-20M}" --retries 3 --low-level-retries 5)

# Repository paths (relative) that hold results or a calibrated task.
PATHS=(
    "serving_tasks/$TASK"
    "serverbench/harness/final_validate/tasks/${TASK,,}.py"
    "server_output"
    "evaluation_progress"
    "serverbench/evolve_space"
    "serverbench/artifacts/final_validation"
)

rc() { nice -n 19 ionice -c3 "$RCLONE" "$@"; }

case "$VERB" in
    save)
        for rel in "${PATHS[@]}"; do
            src="$REPO/$rel"
            [ -e "$src" ] || continue
            if [ -d "$src" ]; then
                # copy, not sync: a remote file is never deleted because this
                # VM no longer has it (an earlier VM may have written it).
                rc copy "${FILTERS[@]}" "${TUNING[@]}" "$src" "$REMOTE/repo/$rel"
            else
                rc copyto "${TUNING[@]}" "$src" "$REMOTE/repo/$rel"
            fi
        done
        if [ -d "$CALIBRATION" ]; then
            rc copy "${FILTERS[@]}" "${TUNING[@]}" "$CALIBRATION" "$REMOTE/calibration"
        fi
        echo "saved to $REMOTE"
        ;;
    restore)
        if ! "$RCLONE" lsf "$REMOTE/repo" >/dev/null 2>&1; then
            echo "nothing saved at $REMOTE yet"; exit 0
        fi
        for rel in "${PATHS[@]}"; do
            if "$RCLONE" lsf "$REMOTE/repo/$(dirname "$rel")" 2>/dev/null \
                    | grep -qx "$(basename "$rel")/\?"; then
                if [[ "$rel" == *.py ]]; then
                    "$RCLONE" copyto "$REMOTE/repo/$rel" "$REPO/$rel"
                else
                    mkdir -p "$REPO/$rel"
                    "$RCLONE" copy "${FILTERS[@]}" "$REMOTE/repo/$rel" "$REPO/$rel"
                fi
            fi
        done
        if "$RCLONE" lsf "$REMOTE/calibration" >/dev/null 2>&1; then
            mkdir -p "$CALIBRATION"
            "$RCLONE" copy "$REMOTE/calibration" "$CALIBRATION"
        fi
        echo "restored from $REMOTE"
        ;;
    status)
        "$RCLONE" cat "$REMOTE/checkpoint.json" 2>/dev/null || echo "no checkpoint at $REMOTE"
        ;;
    *) echo "usage: persist.sh save|restore|status [REMOTE]" >&2; exit 2 ;;
esac
