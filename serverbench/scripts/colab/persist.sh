#!/bin/bash
# Save this runtime's ServerBench results to Google Drive, or restore them into a
# fresh runtime. A Colab VM is discarded at the end of every session; Drive is
# what survives.
#
#   bash serverbench/scripts/colab/persist.sh save    [DEST]
#   bash serverbench/scripts/colab/persist.sh restore [DEST]
#
# DEST defaults to /content/drive/MyDrive/serverbench-colab. Drive must be
# mounted -- and must NOT be mounted while an agent runs (local-evolve.sh
# refuses to start if it is).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VERB="${1:?usage: persist.sh save|restore [DEST]}"
DEST="${2:-/content/drive/MyDrive/serverbench-colab}"
TASK="${TASK:-qwen3_5_2b_standard_1xL4}"
CALIBRATION="${CALIBRATION:-/content/sb-calibration}"

command -v rsync >/dev/null || apt-get -qq install -y rsync >/dev/null
EXCLUDES=(--exclude .venv --exclude uv-cache --exclude .uv-cache
          --exclude __pycache__ --exclude .pytest_cache --exclude '*.safetensors')

# Repository paths (relative) that hold results or a calibrated task.
PATHS=(
    "serving_tasks/$TASK"
    "serverbench/harness/final_validate/tasks/${TASK,,}.py"
    "server_output"
    "evaluation_progress"
    "serverbench/evolve_space"
    "serverbench/artifacts/final_validation"
)

case "$VERB" in
    save)
        mkdir -p "$DEST/repo"
        for rel in "${PATHS[@]}"; do
            [ -e "$REPO/$rel" ] || continue
            mkdir -p "$DEST/repo/$(dirname "$rel")"
            rsync -a "${EXCLUDES[@]}" "$REPO/$rel" "$DEST/repo/$(dirname "$rel")/"
        done
        [ -d "$CALIBRATION" ] && rsync -a "${EXCLUDES[@]}" "$CALIBRATION/" "$DEST/calibration/"
        echo "saved to $DEST"
        ;;
    restore)
        [ -d "$DEST" ] || { echo "nothing saved at $DEST yet"; exit 0; }
        for rel in "${PATHS[@]}"; do
            [ -e "$DEST/repo/$rel" ] || continue
            mkdir -p "$REPO/$(dirname "$rel")"
            rsync -a "$DEST/repo/$rel" "$REPO/$(dirname "$rel")/"
        done
        [ -d "$DEST/calibration" ] && mkdir -p "$CALIBRATION" \
            && rsync -a "$DEST/calibration/" "$CALIBRATION/"
        echo "restored from $DEST"
        ;;
    *) echo "usage: persist.sh save|restore [DEST]" >&2; exit 2 ;;
esac
