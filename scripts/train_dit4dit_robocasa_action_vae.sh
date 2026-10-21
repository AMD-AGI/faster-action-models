#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIT4DIT_ROOT="${DIT4DIT_ROOT:-${ROOT}/DiT4DiT}"
CANDIDATE="k8d48h256_none_align"

usage() {
  cat <<'EOF'
Train the CAE-DiT4DiT RoboCasa full-24 K8/d48 Action VAE.

Usage:
  bash scripts/train_dit4dit_robocasa_action_vae.sh

Required inputs (defaults shown):
  FULL24_ACTION_CACHE
    <DiT4DiT>/playground/action_vae/full24_action_chunks_per_task.pt
  FULL24_LANGUAGE_EMBEDDINGS
    <cache stem>.instruction_embeddings.pt

Optional environment:
  DIT4DIT_PYTHON        Python executable (default: python3)
  GPU_ID                Training GPU (default: 0)
  FULL24_VAE_ROOT       Checkpoint output directory
  FULL24_VAE_STATE_ROOT Logs, sentinels, and timing directory
  TIME_LOG              Resource report path
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
[[ $# -eq 0 ]] || { echo "unknown argument: $1" >&2; usage >&2; exit 2; }
[[ -d "$DIT4DIT_ROOT" ]] || {
  echo "DIT4DIT_ROOT must name the DiT4DiT checkout: $DIT4DIT_ROOT" >&2
  exit 2
}
DIT4DIT_ROOT="$(realpath "$DIT4DIT_ROOT")"

export PYTHON_BIN="${DIT4DIT_PYTHON:-${PYTHON_BIN:-python3}}"
export FULL24_ACTION_CACHE="${FULL24_ACTION_CACHE:-${DIT4DIT_ROOT}/playground/action_vae/full24_action_chunks_per_task.pt}"
export FULL24_LANGUAGE_EMBEDDINGS="${FULL24_LANGUAGE_EMBEDDINGS:-${FULL24_ACTION_CACHE%.pt}.instruction_embeddings.pt}"
export FULL24_VAE_ROOT="${FULL24_VAE_ROOT:-${DIT4DIT_ROOT}/playground/action_vae/full24_sweep}"
export FULL24_VAE_STATE_ROOT="${FULL24_VAE_STATE_ROOT:-${DIT4DIT_ROOT}/playground/full24_vae_sweep}"

[[ -f "$FULL24_ACTION_CACHE" ]] || {
  echo "missing full-24 action cache: $FULL24_ACTION_CACHE" >&2
  exit 2
}
[[ -f "$FULL24_LANGUAGE_EMBEDDINGS" ]] || {
  echo "missing frozen instruction embeddings: $FULL24_LANGUAGE_EMBEDDINGS" >&2
  exit 2
}
if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || {
    echo "DIT4DIT_PYTHON must name an executable: $PYTHON_BIN" >&2
    exit 2
  }
  # Absolute, but symlinks kept: a virtualenv's bin/python is a symlink, and resolving it drops the venv.
  PYTHON_BIN="$(cd "$(dirname "$PYTHON_BIN")" && pwd)/$(basename "$PYTHON_BIN")"
else
  command -v "$PYTHON_BIN" >/dev/null || {
    echo "DIT4DIT_PYTHON is not on PATH: $PYTHON_BIN" >&2
    exit 2
  }
  PYTHON_BIN="$(command -v "$PYTHON_BIN")"
fi
export PYTHON_BIN
FULL24_ACTION_CACHE="$(realpath "$FULL24_ACTION_CACHE")"
FULL24_LANGUAGE_EMBEDDINGS="$(realpath "$FULL24_LANGUAGE_EMBEDDINGS")"
FULL24_VAE_ROOT="$(realpath -m "$FULL24_VAE_ROOT")"
FULL24_VAE_STATE_ROOT="$(realpath -m "$FULL24_VAE_STATE_ROOT")"
export FULL24_ACTION_CACHE FULL24_LANGUAGE_EMBEDDINGS
export FULL24_VAE_ROOT FULL24_VAE_STATE_ROOT

mkdir -p "$FULL24_VAE_STATE_ROOT"
time_log="${TIME_LOG:-${FULL24_VAE_STATE_ROOT}/${CANDIDATE}_time.txt}"
time_log="$(realpath -m "$time_log")"
mkdir -p "$(dirname "$time_log")"

declare -a timer=()
if [[ -x /usr/bin/time ]]; then
  timer=(/usr/bin/time -v -o "$time_log")
else
  echo "warning: /usr/bin/time is unavailable; no resource report will be written" >&2
fi

exec "${timer[@]}" bash \
  "${DIT4DIT_ROOT}/examples/Robocasa_tabletop/train_files/run_full24_action_vae_candidate.sh" \
  "$CANDIDATE"
