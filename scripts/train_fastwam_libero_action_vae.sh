#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTWAM_ROOT="${FASTWAM_ROOT:-${ROOT}/FastWAM}"

usage() {
  cat <<'EOF'
Train the published FastWAM LIBERO K8/d16 Action VAE.

Usage:
  bash scripts/train_fastwam_libero_action_vae.sh

Required environment:
  PREDECODED_ACTION_CACHE  Cached LIBERO actions with padding and suite IDs
  K8D16_VAE_OUTPUT        New output checkpoint path

Optional environment:
  FASTWAM_PYTHON          Python executable (default: python3)
  GPU_ID                  Training GPU (default: 0)
  TIME_LOG                Resource report path (default: beside checkpoint)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
[[ $# -eq 0 ]] || { echo "unknown argument: $1" >&2; usage >&2; exit 2; }

cache="${PREDECODED_ACTION_CACHE:-}"
output="${K8D16_VAE_OUTPUT:-}"
python_bin="${FASTWAM_PYTHON:-python3}"
gpu="${GPU_ID:-0}"

[[ -f "$cache" ]] || {
  echo "PREDECODED_ACTION_CACHE must name the cached LIBERO actions: $cache" >&2
  exit 2
}
[[ -n "$output" ]] || {
  echo "set K8D16_VAE_OUTPUT to a new output checkpoint path" >&2
  exit 2
}
[[ -d "$FASTWAM_ROOT" ]] || {
  echo "FASTWAM_ROOT must name the FastWAM checkout: $FASTWAM_ROOT" >&2
  exit 2
}
if [[ "$python_bin" == */* ]]; then
  [[ -x "$python_bin" ]] || {
    echo "FASTWAM_PYTHON must name an executable: $python_bin" >&2
    exit 2
  }
else
  command -v "$python_bin" >/dev/null || {
    echo "FASTWAM_PYTHON is not on PATH: $python_bin" >&2
    exit 2
  }
  python_bin="$(command -v "$python_bin")"
fi

cache="$(realpath "$cache")"
output="$(realpath -m "$output")"
FASTWAM_ROOT="$(realpath "$FASTWAM_ROOT")"
[[ ! -e "$output" ]] || {
  echo "refusing to overwrite existing Action VAE checkpoint: $output" >&2
  exit 2
}
mkdir -p "$(dirname "$output")"
time_log="${TIME_LOG:-${output%.*}_time.txt}"
time_log="$(realpath -m "$time_log")"
mkdir -p "$(dirname "$time_log")"

declare -a timer=()
if [[ -x /usr/bin/time ]]; then
  timer=(/usr/bin/time -v -o "$time_log")
else
  echo "warning: /usr/bin/time is unavailable; no resource report will be written" >&2
fi

cd "$FASTWAM_ROOT"
export CUDA_VISIBLE_DEVICES="$gpu"
export HIP_VISIBLE_DEVICES="$gpu"
exec "${timer[@]}" "$python_bin" scripts/train_action_ae.py \
  --predecoded "$cache" \
  --num-tokens 8 \
  --latent-dim 16 \
  --encoder-hidden 128 \
  --decoder-hidden 128 \
  --video-cond-dim 0 \
  --num-suites 4 \
  --variational \
  --beta-kl 1e-4 \
  --free-bits 0.1 \
  --kl-warmup-epochs 100 \
  --epochs 50 \
  --batch-size 256 \
  --output "$output"
