#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIT4DIT_ROOT="${DIT4DIT_ROOT:-${ROOT}/DiT4DiT}"
DEFAULT_RUN_ID="dit4dit_robocasa_full24_latent_k8_d48_mi325x8_b16_opt_s42"

usage() {
  cat <<'EOF'
Train the 200K DiT4DiT RoboCasa full-24 latent policy (CAE-DiT4DiT).

Usage:
  bash scripts/train_dit4dit_robocasa_latent_policy.sh

Required environment:
  SOURCE_DATA_ROOT               Full-24 RoboCasa-GR1 dataset
  ACTION_VAE_CHECKPOINT          K8/d48 Action VAE
  POLICY_ACTION_STATISTICS_PATH  Full-24 policy normalization statistics

Optional environment:
  DIT4DIT_PYTHON       Python executable (default: python3)
  STAGED_DATA_ROOT     Dataset staging path under /dev/shm
  HIP_VISIBLE_DEVICES Default: 0,1,2,3,4,5,6,7
  NUM_PROCESSES        Default: 8
  RUN_ID               Training run ID
  TIME_LOG             Resource report path (default: run directory)

The native launcher retains the production defaults: global batch 128,
200,000 steps, seed 42, eager Cosmos, prompt caching, and VAE encode reuse.
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

export SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-}"
export ACTION_VAE_CHECKPOINT="${ACTION_VAE_CHECKPOINT:-${ACTION_VAE:-}}"
export POLICY_ACTION_STATISTICS_PATH="${POLICY_ACTION_STATISTICS_PATH:-${POLICY_STATS:-}}"
export PYTHON_BIN="${DIT4DIT_PYTHON:-${PYTHON_BIN:-python3}}"
export RUN_ID="${RUN_ID:-${DEFAULT_RUN_ID}}"

[[ -d "$SOURCE_DATA_ROOT" ]] || {
  echo "SOURCE_DATA_ROOT must name the full-24 dataset: $SOURCE_DATA_ROOT" >&2
  exit 2
}
[[ -f "$ACTION_VAE_CHECKPOINT" ]] || {
  echo "ACTION_VAE_CHECKPOINT must name the K8/d48 Action VAE: $ACTION_VAE_CHECKPOINT" >&2
  exit 2
}
[[ -f "$POLICY_ACTION_STATISTICS_PATH" ]] || {
  echo "POLICY_ACTION_STATISTICS_PATH must name the policy statistics: $POLICY_ACTION_STATISTICS_PATH" >&2
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
SOURCE_DATA_ROOT="$(realpath "$SOURCE_DATA_ROOT")"
ACTION_VAE_CHECKPOINT="$(realpath "$ACTION_VAE_CHECKPOINT")"
POLICY_ACTION_STATISTICS_PATH="$(realpath "$POLICY_ACTION_STATISTICS_PATH")"
export PYTHON_BIN SOURCE_DATA_ROOT ACTION_VAE_CHECKPOINT
export POLICY_ACTION_STATISTICS_PATH

output_dir="${DIT4DIT_ROOT}/playground/Checkpoints_robocasa/${RUN_ID}"
mkdir -p "$output_dir"
time_log="${TIME_LOG:-${output_dir}/training_time.txt}"
time_log="$(realpath -m "$time_log")"
mkdir -p "$(dirname "$time_log")"

declare -a timer=()
if [[ -x /usr/bin/time ]]; then
  timer=(/usr/bin/time -v -o "$time_log")
else
  echo "warning: /usr/bin/time is unavailable; no resource report will be written" >&2
fi

exec "${timer[@]}" bash \
  "${DIT4DIT_ROOT}/examples/Robocasa_tabletop/train_files/run_robocasa_full24_latent_mi325x8.sh"
