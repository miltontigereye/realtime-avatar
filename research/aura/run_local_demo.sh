#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
model="${1:-both}"
if [[ "$model" != both && "$model" != aura-1.3b && "$model" != aura-5b ]]; then
  echo "Usage: $0 [both|aura-1.3b|aura-5b] [portrait] [audio]" >&2
  exit 2
fi
image="${2:-demo/portrait.jpg}"
audio="${3:-demo/speech.wav}"
mkdir -p demo/outputs demo/logs
run_model() {
  local selected="$1"
  local options=(--offload-text-encoder)
  if [[ "$selected" == aura-5b ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    options+=(--bf16-transformer --vae-decode-group 1 --offload-transformer-for-decode)
  fi
  .venv/bin/python -u generate.py --model "$selected" --image "$image" --audio "$audio" \
    --output "demo/outputs/$selected.mp4" "${options[@]}" \
    2>&1 | tee "demo/logs/$selected.log"
}
if [[ "$model" == both ]]; then
  run_model aura-1.3b
  run_model aura-5b
else
  run_model "$model"
fi
