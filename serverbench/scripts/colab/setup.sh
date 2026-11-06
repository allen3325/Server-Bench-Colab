#!/bin/bash
# Prepare a Google Colab runtime (root, one L4) for ServerBench's local backend.
#
# Idempotent: every stage checks its own result and skips when it is already
# there, so a re-run after a disconnect only does what is missing. Everything
# lands under /opt/serverbench (world-readable, because the agent user runs the
# tools) except the repository's own artifacts, which `local_env.py` makes
# root-only where they are private.
#
#   bash serverbench/scripts/colab/setup.sh            # everything
#   STAGES="model corpus" bash .../setup.sh            # just some stages
#
# Stages, in order: system uv submodules harness agentx wheelhouse answers
#                   agentpython cuda clis sglang model corpus user check
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SB=/opt/serverbench
TASK="${TASK:-qwen3_5_2b_standard_1xL4}"
STAGES="${STAGES:-system uv submodules harness agentx wheelhouse answers agentpython cuda clis sglang model corpus user check}"
NODE_VERSION="${NODE_VERSION:-v22.11.0}"
SGLANG_SPEC="${SGLANG_SPEC:-sglang[all]==0.5.19}"

export UV_PYTHON_INSTALL_DIR="$SB/python"     # interpreters the agent can exec
export HF_HOME="$SB/hf"                       # the model and corpus cache
export PATH="$SB/bin:$PATH"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mok\033[0m   %s\n' "$*"; }
die()  { printf '   \033[31mfail\033[0m %s\n' "$*" >&2; exit 1; }
want() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

[ "$(id -u)" = 0 ] || die "run as root (a Colab runtime is root)"
mkdir -p "$SB/bin" "$HF_HOME"
chmod 755 /opt "$SB"

PIN_FILE="$REPO/serverbench/harness/final_validate/agentx_pin.py"
AGENTX_COMMIT="$(sed -n 's/^PINNED_COMMIT = "\([0-9a-f]\{40\}\)"$/\1/p' "$PIN_FILE")"
AGENTX_REMOTE="$(sed -n 's/^PINNED_REMOTE = "\(.*\)"$/\1/p' "$PIN_FILE")"
PY="$REPO/serverbench/.venv/bin/python"
RUNTIME="$REPO/serverbench/artifacts/agentx-runtime"
model_id()  { sed -n 's/^MODEL_ID = "\(.*\)"$/\1/p' "$REPO/serverbench/harness/final_validate/tasks/${TASK,,}.py"; }
model_rev() { sed -n 's/^MODEL_REVISION = "\(.*\)"$/\1/p' "$REPO/serverbench/harness/final_validate/tasks/${TASK,,}.py"; }

if want system; then
    say "system packages"
    export DEBIAN_FRONTEND=noninteractive
    apt-get -qq update >/dev/null
    apt-get -qq install -y curl wget git iproute2 util-linux procps \
        build-essential ca-certificates xz-utils >/dev/null
    ok "$(nsenter --version | head -1)"
fi

if want uv; then
    say "uv and a managed Python 3.12"
    if [ ! -x "$SB/bin/uv" ]; then
        curl -LsSf https://astral.sh/uv/install.sh \
            | env UV_INSTALL_DIR="$SB/bin" UV_NO_MODIFY_PATH=1 sh >/dev/null
    fi
    uv python install 3.12 >/dev/null
    ok "$(uv --version); python $(uv python find 3.12)"
fi

if want submodules; then
    say "submodules (AgentX source, skill bundles)"
    git config --global --get-all safe.directory | grep -qx '\*' \
        || git config --global --add safe.directory '*'
    git -C "$REPO" submodule update --init --recursive >/dev/null 2>&1 \
        || git -C "$REPO" submodule update --init agentx-harness serverbench/skills/ncu-report-skill
    ok "$(git -C "$REPO" submodule status | wc -l) submodule(s)"
fi

if want harness; then
    say "harness interpreter ($PY) with the pinned AgentX"
    [ -x "$PY" ] || uv venv -q "$REPO/serverbench/.venv" --python 3.12
    uv pip install -q --python "$PY" -e "$REPO/serverbench[dev]" \
        "aiperf @ git+$AGENTX_REMOTE@$AGENTX_COMMIT"
    (cd "$REPO/serverbench/harness" && env -u PYTHONPATH "$PY" -m final_validate.agentx_pin >/dev/null) \
        || die "the installed AgentX is not the pinned one"
    ok "AgentX ${AGENTX_COMMIT:0:12} matches the pin"
fi

if want agentx; then
    say "AgentX runtime the agent measures with ($RUNTIME)"
    [ -x "$RUNTIME/venv/bin/python" ] || \
        uv venv -q "$RUNTIME/venv" --python 3.12 --managed-python --relocatable
    uv pip install -q --python "$RUNTIME/venv/bin/python" \
        "aiperf @ git+$AGENTX_REMOTE@$AGENTX_COMMIT"
    mkdir -p "$RUNTIME/bin"
    cat > "$RUNTIME/bin/aiperf" <<'SH'
#!/bin/sh
exec "$(dirname "$0")/../venv/bin/python" -m aiperf "$@"
SH
    chmod +x "$RUNTIME/bin/aiperf"
    "$RUNTIME/bin/aiperf" --help >/dev/null || die "the AgentX runtime does not start"
    chmod -R a+rX "$RUNTIME"
    ok "aiperf at $RUNTIME/bin/aiperf"
fi

if want wheelhouse; then
    say "offline wheelhouse (/opt/env-input in the sandbox)"
    PINS="$REPO/serverbench/artifacts/env-input/resolved.requirements.txt"
    if [ -f "$REPO/serverbench/artifacts/env-input/wheelhouse.manifest.json" ]; then
        ok "already built"
    else
        "$PY" "$REPO/serverbench/scripts/agent/build_env_bundle.py" \
            --out "$REPO/serverbench/artifacts/env-input" --pins "$PINS" --uv "$SB/bin/uv"
    fi
    chmod -R a+rX "$REPO/serverbench/artifacts/env-input"
fi

if want answers; then
    say "held-out answer pool (root-only)"
    "$PY" "$REPO/serverbench/scripts/deploy/build_answer_pool.py"
    chmod 700 "$REPO/serverbench/artifacts/eval_samples"
fi

if want agentpython; then
    say "the image's python3 ($SB/agent-python): torch and the tokenizer stack at the image's pins"
    [ -x "$SB/agent-python/bin/python" ] || uv venv -q --seed "$SB/agent-python" --python 3.12
    uv pip install -q --python "$SB/agent-python/bin/python" --no-index \
        --find-links "$REPO/serverbench/artifacts/env-input/wheelhouse" \
        "torch==2.8.0" "transformers==5.16.1" "tokenizers==0.23.1" \
        "protobuf==7.36.1" "sentencepiece==0.2.2" "huggingface-hub==1.29.0" \
        "safetensors==0.8.0" "accelerate==1.14.0" "numpy>=1.26,<3" \
        "fastapi==0.141.1" "uvicorn==0.52.4" "httpx==0.28.1" "pytest==9.1.1" \
        "pyyaml>=6" ninja triton
    chmod -R a+rX "$SB/agent-python" "$SB/python"
    "$SB/agent-python/bin/python" -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"
fi

if want cuda; then
    say "CUDA 12.8 toolchain and Nsight, the agent image's own .debs"
    if [ -x /usr/local/cuda-12.8/bin/nvcc ] && [ -x "$SB/bin/ncu" ]; then
        ok "already installed"
    else
        "$PY" - "$REPO/serverbench/environments/evolve-agent/Dockerfile" <<'PY'
import hashlib, re, subprocess, sys, tempfile, urllib.request
from pathlib import Path
text = Path(sys.argv[1]).read_text()
args = dict(re.findall(r"^ARG (\w+)=(\S+)$", text, re.M))
repo = args["CUDA_REPO"]
debs = [(args[k], args[k[:-4] + "_SHA256"]) for k in args
        if k.endswith("_DEB") and k[:-4] + "_SHA256" in args]
work = Path(tempfile.mkdtemp())
headers_only = {"CUBLAS_DEV", "CUSPARSE_DEV", "CUSOLVER_DEV"}
for key in [k for k in args if k.endswith("_DEB")]:
    name, digest = args[key], args.get(key[:-4] + "_SHA256")
    if not digest:
        continue
    path = work / name
    urllib.request.urlretrieve(f"{repo}/{name}", path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"checksum mismatch: {name}")
    if key[:-4] in headers_only:
        tmp = work / "hdr"
        subprocess.run(["dpkg", "-x", str(path), str(tmp)], check=True)
        inc = tmp / "usr/local/cuda-12.8/targets/x86_64-linux/include"
        subprocess.run(["cp", "-a", f"{inc}/.",
                        "/usr/local/cuda-12.8/targets/x86_64-linux/include/"], check=True)
        subprocess.run(["rm", "-rf", str(tmp)], check=True)
    else:
        subprocess.run(["dpkg", "-x", str(path), "/"], check=True)
    print("   installed", name)
PY
        NSYS="$(sed -n 's/^ARG NSYS_VERSION=//p' "$REPO/serverbench/environments/evolve-agent/Dockerfile" | head -1)"
        NCU="$(sed -n 's/^ARG NCU_VERSION=//p' "$REPO/serverbench/environments/evolve-agent/Dockerfile" | head -1)"
        ln -sf "/opt/nvidia/nsight-systems/$NSYS/target-linux-x64/nsys" "$SB/bin/nsys"
        ln -sf "/opt/nvidia/nsight-compute/$NCU/ncu" "$SB/bin/ncu"
    fi
    # The brief names /usr/local/cuda: point it at the 12.8 toolchain torch
    # is built against (Colab's own torch carries its CUDA in wheels).
    ln -sfn /usr/local/cuda-12.8 /usr/local/cuda
    echo /usr/local/cuda/lib64 > /etc/ld.so.conf.d/cuda.conf && ldconfig
    /usr/local/cuda/bin/nvcc --version | tail -1
fi

if want clis; then
    say "agent CLIs (claude, codex) on a private Node"
    if [ ! -x "$SB/node/bin/node" ]; then
        curl -fsSL "https://nodejs.org/dist/$NODE_VERSION/node-$NODE_VERSION-linux-x64.tar.xz" \
            | tar -xJ -C "$SB" && mv "$SB/node-$NODE_VERSION-linux-x64" "$SB/node"
    fi
    PATH="$SB/node/bin:$PATH" "$SB/node/bin/npm" install -g --silent \
        --prefix "$SB/npm" @anthropic-ai/claude-code @openai/codex >/dev/null
    for tool in node; do ln -sf "$SB/node/bin/$tool" "$SB/bin/$tool"; done
    for tool in claude codex; do ln -sf "$SB/npm/bin/$tool" "$SB/bin/$tool"; done
    chmod -R a+rX "$SB/node" "$SB/npm"
    ok "claude $("$SB/bin/claude" --version 2>/dev/null | head -1); codex $("$SB/bin/codex" --version 2>/dev/null | head -1)"
fi

if want sglang; then
    say "SGLang reference arm ($SGLANG_SPEC) in its own interpreter"
    [ -x "$SB/sglang-venv/bin/python" ] || uv venv -q --seed "$SB/sglang-venv" --python 3.12
    uv pip install -q --python "$SB/sglang-venv/bin/python" "$SGLANG_SPEC"
    ln -sf python "$SB/sglang-venv/bin/python3"
    chmod -R a+rX "$SB/sglang-venv"
    "$SB/sglang-venv/bin/python" -c "import sglang; print('sglang', sglang.__version__)"
fi

if want model; then
    say "the task's model at its pinned revision ($HF_HOME)"
    "$RUNTIME/venv/bin/python" - "$(model_id)" "$(model_rev)" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo, rev = sys.argv[1], sys.argv[2]
path = snapshot_download(repo, revision=rev)
refs = os.path.join(os.environ["HF_HOME"], "hub", "models--" + repo.replace("/", "--"), "refs")
main = os.path.join(refs, "main")
if not (os.path.isfile(main) and open(main).read().strip() == rev):
    os.makedirs(refs, exist_ok=True)
    open(main, "w").write(rev)
print("   model", path)
PY
fi

if want corpus; then
    say "the replay corpus, and its processed datasets cache"
    "$RUNTIME/venv/bin/python" - "$REPO" "$TASK" <<'PY'
import os, sys
sys.path.insert(0, os.path.join(sys.argv[1], "serverbench", "harness"))
from final_validate import tasks
corpus = tasks.python_module(sys.argv[2]).CORPUS
from huggingface_hub import snapshot_download
path = snapshot_download(corpus, repo_type="dataset", max_workers=8)
refs = os.path.join(os.environ["HF_HOME"], "hub", "datasets--" + corpus.replace("/", "--"), "refs")
head = os.path.basename(path.rstrip("/"))
main = os.path.join(refs, "main")
if not (os.path.isfile(main) and open(main).read().strip() == head):
    os.makedirs(refs, exist_ok=True)
    open(main, "w").write(head)
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ["HF_HOME"], "datasets")
from datasets import load_dataset
built = load_dataset(corpus, split="train")
print("   corpus", corpus, built.num_rows, "rows")
PY
    chmod -R a+rX "$HF_HOME" 2>/dev/null || true
fi

if want user; then
    say "the agent user"
    id agent >/dev/null 2>&1 || useradd -m -d /home/agent -s /bin/bash agent
    ok "$(id agent)"
fi

if want check; then
    say "sandbox self-check (namespaces, broker, GPU)"
    "$PY" "$REPO/serverbench/scripts/colab/selfcheck.py"
fi

say "ready"
echo "   MODEL_DIR=$HF_HOME/hub/models--$(model_id | sed 's#/#--#')/snapshots/$(model_rev)"
