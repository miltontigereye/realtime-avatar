#!/usr/bin/env bash
set -euo pipefail
cd /workspace/avatar/research/aura
export HF_HOME=/workspace/cache/huggingface HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/workspace/avatar/research/aura/experiments/rtx1-vae-002/.inductor-cache
export TORCHINDUCTOR_COMPILE_THREADS=8
exec .venv/bin/python -u experiments/rtx1-fp8-linear-001/experiment.py "$@"
