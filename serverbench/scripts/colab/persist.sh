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
        --bwlimit "${SB_BWLIMIT:-20M}" --retries 3 --low-level-retries 10)

# Repository paths (relative) that hold results or a calibrated task.
PATHS=(
    "serving_tasks/$TASK"
    "serverbench/harness/final_validate/tasks/${TASK,,}.py"
    "server_output"
    "evaluation_progress"
    "serverbench/evolve_space"
    "serverbench/artifacts/final_validation"
)

# Every rclone call goes through `rc`, which tries again with a growing pause.
# A Drive remote without its own client_id shares rclone's public API project,
# whose per-minute quota other people exhaust: the 403 then arrives while the
# remote is being opened ("couldn't find root directory ID"), which rclone's
# own --retries does not cover. Exit 3 (directory not found) and 4 (file not
# found) are answers, not failures, and are returned at once.
rc() {
    local try=1 tries="${SB_RCLONE_TRIES:-6}" pause="${SB_RCLONE_PAUSE:-20}" code
    while :; do
        code=0
        nice -n 19 ionice -c3 "$RCLONE" "$@" || code=$?
        case "$code" in 0|3|4) return "$code" ;; esac
        [ "$try" -lt "$tries" ] || return "$code"
        echo "rclone $1 failed (exit $code); try $((try + 1))/$tries in ${pause}s" >&2
        sleep "$pause"; try=$((try + 1)); pause=$((pause * 2))
    done
}
# has REMOTE_DIR [NAME]: 0 when the directory (or NAME in it) is there, 1 when
# it is not, and the script stops when the remote could not be asked -- "the
# quota ran out" must never read as "nothing was saved".
has() {
    local listing code=0
    listing=$(rc lsf "$1" 2>"$ERR") || code=$?
    case "$code" in
        0) [ -z "${2:-}" ] || grep -qx "$2/\?" <<<"$listing" ;;
        3) return 1 ;;
        *) cat "$ERR" >&2; echo "could not list $1 (rclone exit $code)" >&2; exit 1 ;;
    esac
}
ERR="$(mktemp)"; trap 'rm -f "$ERR"' EXIT

case "$VERB" in
    save)
        # One path failing does not stop the others: what can be saved is.
        failed=()
        for rel in "${PATHS[@]}"; do
            src="$REPO/$rel"
            [ -e "$src" ] || continue
            if [ -d "$src" ]; then
                # copy, not sync: a remote file is never deleted because this
                # VM no longer has it (an earlier VM may have written it).
                rc copy "${FILTERS[@]}" "${TUNING[@]}" "$src" "$REMOTE/repo/$rel" || failed+=("$rel")
            else
                rc copyto "${TUNING[@]}" "$src" "$REMOTE/repo/$rel" || failed+=("$rel")
            fi
        done
        if [ -d "$CALIBRATION" ]; then
            rc copy "${FILTERS[@]}" "${TUNING[@]}" "$CALIBRATION" "$REMOTE/calibration" || failed+=("$CALIBRATION")
        fi
        if [ ${#failed[@]} -gt 0 ]; then
            echo "NOT saved to $REMOTE: ${failed[*]} -- still only on this VM; run save again" >&2
            exit 1
        fi
        echo "saved to $REMOTE"
        ;;
    restore)
        if ! has "$REMOTE/repo"; then
            echo "nothing saved at $REMOTE yet"; exit 0
        fi
        for rel in "${PATHS[@]}"; do
            if has "$REMOTE/repo/$(dirname "$rel")" "$(basename "$rel")"; then
                if [[ "$rel" == *.py ]]; then
                    rc copyto "$REMOTE/repo/$rel" "$REPO/$rel"
                else
                    mkdir -p "$REPO/$rel"
                    rc copy "${FILTERS[@]}" "$REMOTE/repo/$rel" "$REPO/$rel"
                fi
            fi
        done
        if has "$REMOTE/calibration"; then
            mkdir -p "$CALIBRATION"
            rc copy "$REMOTE/calibration" "$CALIBRATION"
        fi
        echo "restored from $REMOTE"
        ;;
    status)
        rc cat "$REMOTE/checkpoint.json" 2>/dev/null || echo "no checkpoint at $REMOTE"
        ;;
    *) echo "usage: persist.sh save|restore|status [REMOTE]" >&2; exit 2 ;;
esac
