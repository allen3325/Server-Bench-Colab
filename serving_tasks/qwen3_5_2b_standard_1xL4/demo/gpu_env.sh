#!/bin/bash
# RUNTIME GPU DETECTION — the task hardcodes NO specific card.
#
# The image is built once and may run on any CUDA GPU the wheels support (Ada/L40 sm_89,
# Hopper/H100/H200 sm_90, Blackwell/B200 sm_100, ...). Nothing here is baked at build time:
# every arch-dependent value is derived from the card that is ACTUALLY PRESENT, at run time.
#
# Source it (do not execute) to get the vars exported into your shell:
#     source "${APP_DIR:-/app}/gpu_env.sh"
# The image's /etc/profile.d/mle_gpu.sh sources it from $APP_DIR, so login shells
# pick it up automatically, and serve.sh sources it before starting the server.
#
# Exports:
#   GPU_NAME              e.g. "NVIDIA H200 NVL"
#   GPU_COUNT             number of visible GPUs
#   GPU_CC                compute capability as reported, e.g. "9.0"
#   GPU_SM                the same without the dot, e.g. "90"  (-> sm_90)
#   GPU_MEM_MIB           total memory of the first visible GPU, in MiB
#   TORCH_CUDA_ARCH_LIST  set to GPU_CC unless already set  (torch extension builds)
#   NVCC_ARCH_FLAG        "-arch=native" when nvcc supports it, else "-arch=sm_<GPU_SM>"
#
# Build custom CUDA with:   nvcc $NVCC_ARCH_FLAG ...
# Never hardcode sm_90 / sm_100 / 10.0 anywhere — read these instead.

_gpu_query() { nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -1 | sed 's/^ *//; s/ *$//'; }

GPU_NAME="$(_gpu_query name)"
GPU_MEM_MIB="$(_gpu_query memory.total)"
GPU_CC="$(_gpu_query compute_cap)"
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"

# Fallback for drivers whose nvidia-smi predates the compute_cap field: ask torch.
if [ -z "${GPU_CC}" ] || [ "${GPU_CC}" = "[N/A]" ]; then
    GPU_CC="$(python3 -c 'import torch;print("%d.%d"%torch.cuda.get_device_capability(0))' 2>/dev/null)"
fi

GPU_SM="$(echo "${GPU_CC}" | tr -d '.')"

# TORCH_CUDA_ARCH_LIST drives torch's C++/CUDA extension builds. Detected, never baked —
# an explicit value already in the environment always wins.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ] && [ -n "${GPU_CC}" ]; then
    export TORCH_CUDA_ARCH_LIST="${GPU_CC}"
fi

# `nvcc -arch=native` compiles for the GPU in the machine (CUDA >= 11.5). Fall back to an
# explicit sm_<cc> when the toolkit is older or detection failed.
if nvcc --help 2>/dev/null | grep -q -- '-arch=native\|native'; then
    NVCC_ARCH_FLAG="-arch=native"
elif [ -n "${GPU_SM}" ]; then
    NVCC_ARCH_FLAG="-arch=sm_${GPU_SM}"
else
    NVCC_ARCH_FLAG=""
fi

export GPU_NAME GPU_COUNT GPU_CC GPU_SM GPU_MEM_MIB NVCC_ARCH_FLAG

# Executed directly (not sourced) -> print the readout instead of silently doing nothing.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "GPU_NAME=${GPU_NAME}"
    echo "GPU_COUNT=${GPU_COUNT}"
    echo "GPU_CC=${GPU_CC}   (sm_${GPU_SM})"
    echo "GPU_MEM_MIB=${GPU_MEM_MIB}"
    echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-<unset>}"
    echo "NVCC_ARCH_FLAG=${NVCC_ARCH_FLAG}"
    echo
    echo "source this file to export them:  source ${APP_DIR:-/app}/gpu_env.sh"
fi
