#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTWAM_ROOT="${FASTWAM_ROOT:-${ROOT}/FastWAM}"
TASK="libero_uncond_2cam224_1e-4_32hdim_latent_K8d16_freebits_pernorm_joint"

usage() {
  cat <<'EOF'
Train Compact Action DiT in the FastWAM LIBERO K8/d16 latent space.

Usage:
  bash scripts/train_fastwam_libero_latent_policy.sh [hydra_overrides...]

Required environment:
  COMPACT_INIT_CHECKPOINT  Compact Action DiT initializer
  K8D16_VAE_CHECKPOINT    Trained K8/d16 Action VAE

Optional environment:
  NPROC_PER_NODE           Worker/GPU count (default: 2)
  RUN_ID                   Output run ID (default: UTC timestamp)
  TIME_LOG                 Resource report path (default: run directory)
  FASTWAM_ROOT             Default: <repo>/FastWAM

The LIBERO datasets, per-suite statistics, text cache, and Wan assets must
match the paths in the task configuration. Additional arguments are forwarded
as Hydra overrides.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

compact_init="${COMPACT_INIT_CHECKPOINT:-}"
action_vae="${K8D16_VAE_CHECKPOINT:-}"
nproc="${NPROC_PER_NODE:-2}"
run_id="${RUN_ID:-$(date -u +%Y-%m-%d_%H-%M-%S)}"

[[ -f "$compact_init" ]] || {
  echo "COMPACT_INIT_CHECKPOINT must name the compact initializer: $compact_init" >&2
  exit 2
}
[[ -f "$action_vae" ]] || {
  echo "K8D16_VAE_CHECKPOINT must name the trained Action VAE: $action_vae" >&2
  exit 2
}
[[ "$nproc" =~ ^[1-9][0-9]*$ ]] || {
  echo "NPROC_PER_NODE must be a positive integer, got: $nproc" >&2
  exit 2
}
[[ -d "$FASTWAM_ROOT" ]] || {
  echo "FASTWAM_ROOT must name the FastWAM checkout: $FASTWAM_ROOT" >&2
  exit 2
}

compact_init="$(realpath "$compact_init")"
action_vae="$(realpath "$action_vae")"
FASTWAM_ROOT="$(realpath "$FASTWAM_ROOT")"
run_dir="${FASTWAM_ROOT}/runs/${TASK}/${run_id}"
mkdir -p "$run_dir"
time_log="${TIME_LOG:-${run_dir}/training_time.txt}"
time_log="$(realpath -m "$time_log")"
mkdir -p "$(dirname "$time_log")"

declare -a timer=()
if [[ -x /usr/bin/time ]]; then
  timer=(/usr/bin/time -v -o "$time_log")
else
  echo "warning: /usr/bin/time is unavailable; no resource report will be written" >&2
fi

cd "$FASTWAM_ROOT"
export RUN_ID="$run_id"
exec "${timer[@]}" bash scripts/train_zero1.sh "$nproc" \
  "task=${TASK}" \
  "model.action_dit_pretrained_path=${compact_init}" \
  "model.action_vae_ckpt=${action_vae}" \
  "$@"
