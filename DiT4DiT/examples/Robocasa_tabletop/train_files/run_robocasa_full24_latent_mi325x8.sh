#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_ID=${RUN_ID:-dit4dit_robocasa_full24_latent_k8_d48_mi325x8_b16_opt_s42}
OUTPUT_DIR="${REPO_ROOT}/playground/Checkpoints_robocasa/${RUN_ID}"
LOCK_PATH="${REPO_ROOT}/playground/locks/${RUN_ID}.lock"
SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT:-"${REPO_ROOT}/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"}
STAGED_DATA_ROOT=${STAGED_DATA_ROOT:-/dev/shm/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim}
ACTION_VAE_CHECKPOINT=${ACTION_VAE_CHECKPOINT:-"${REPO_ROOT}/playground/action_vae/full24_sweep/k8d48h256_none_align.pt"}
POLICY_ACTION_STATISTICS_PATH=${POLICY_ACTION_STATISTICS_PATH:-"${REPO_ROOT}/playground/Checkpoints_robocasa/benchmark_full24_b16_eager/dataset_statistics.json"}

[[ -d "${SOURCE_DATA_ROOT}" ]] || {
  echo "Missing source full-24 dataset: ${SOURCE_DATA_ROOT}" >&2
  exit 1
}
[[ -f "${ACTION_VAE_CHECKPOINT}" ]] || {
  echo "Missing language-aware ActionVAE: ${ACTION_VAE_CHECKPOINT}" >&2
  exit 1
}
[[ -f "${POLICY_ACTION_STATISTICS_PATH}" ]] || {
  echo "Missing full-24 policy action statistics: ${POLICY_ACTION_STATISTICS_PATH}" >&2
  exit 1
}
case "${STAGED_DATA_ROOT}" in
  /dev/shm/*) ;;
  *)
    echo "STAGED_DATA_ROOT must be beneath /dev/shm: ${STAGED_DATA_ROOT}" >&2
    exit 1
    ;;
esac

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOCK_PATH}")"
exec 9>"${LOCK_PATH}"
flock -n 9 || {
  echo "Run lock is already held: ${LOCK_PATH}" >&2
  exit 1
}

source_bytes=$(du -sb "${SOURCE_DATA_ROOT}" | cut -f1)
stage_signature="${SOURCE_DATA_ROOT}:${source_bytes}"
stage_marker="${STAGED_DATA_ROOT}/.dit4dit_stage_complete"
if [[ ! -f "${stage_marker}" ]] || [[ "$(<"${stage_marker}")" != "${stage_signature}" ]]; then
  echo "Staging ${source_bytes} bytes of RoboCasa data into ${STAGED_DATA_ROOT}"
  rm -rf "${STAGED_DATA_ROOT}"
  available_bytes=$(df --output=avail -B1 /dev/shm | awk 'NR == 2 {print $1}')
  reserve_bytes=$((64 * 1024 * 1024 * 1024))
  if (( available_bytes < source_bytes + reserve_bytes )); then
    echo "Insufficient /dev/shm space: require dataset plus 64 GiB reserve" >&2
    exit 1
  fi
  partial_stage="${STAGED_DATA_ROOT}.partial.$$"
  rm -rf "${partial_stage}"
  mkdir -p "${partial_stage}"
  rsync -a --delete --info=progress2 \
    "${SOURCE_DATA_ROOT}/" "${partial_stage}/"
  printf '%s\n' "${stage_signature}" >"${partial_stage}/.dit4dit_stage_complete"
  mv "${partial_stage}" "${STAGED_DATA_ROOT}"
else
  echo "Reusing validated staged dataset at ${STAGED_DATA_ROOT}"
fi

export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export RUN_ID
export DATA_MIX=fourier_gr1_unified_1000
export DATA_ROOT_DIR="${STAGED_DATA_ROOT}"
export CONFIG_YAML="${REPO_ROOT}/DiT4DiT/config/robocasa/dit4dit_robocasa_gr1_latent_k8_d48.yaml"
export ACTION_VAE_CHECKPOINT
export POLICY_ACTION_STATISTICS_PATH
export NUM_PROCESSES=${NUM_PROCESSES:-8}
export PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
export ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-false}
export CACHE_PROMPT_EMBEDDINGS=${CACHE_PROMPT_EMBEDDINGS:-true}
export REUSE_CAUSAL_VAE_ENCODE=${REUSE_CAUSAL_VAE_ENCODE:-true}
export MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-200000}
export NUM_WARMUP_STEPS=${NUM_WARMUP_STEPS:-5000}
export BASE_LEARNING_RATE=${BASE_LEARNING_RATE:-3e-5}
export BACKBONE_LEARNING_RATE=${BACKBONE_LEARNING_RATE:-1e-5}
export ACTION_LEARNING_RATE=${ACTION_LEARNING_RATE:-1e-4}
export MIN_LEARNING_RATE=${MIN_LEARNING_RATE:-5e-7}
export LOGGING_FREQUENCY=${LOGGING_FREQUENCY:-10}
export EVAL_INTERVAL=${EVAL_INTERVAL:-100}
export SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
export FULL_STATE_SAVE_INTERVAL=${FULL_STATE_SAVE_INTERVAL:-5000}
export MAX_FULL_TRAINING_STATES=${MAX_FULL_TRAINING_STATES:-1}
export SAVE_FULL_TRAINING_STATE=${SAVE_FULL_TRAINING_STATE:-false}
export SAVE_TRAINABLE_ONLY_CHECKPOINTS=${SAVE_TRAINABLE_ONLY_CHECKPOINTS:-true}
export FULL_MODEL_SAVE_INTERVAL=${FULL_MODEL_SAVE_INTERVAL:-50000}
export SKIP_FINAL_CHECKPOINT=${SKIP_FINAL_CHECKPOINT:-false}
export BENCHMARK_ENABLED=${BENCHMARK_ENABLED:-false}
export COMPILE_MODULE=${COMPILE_MODULE:-none}
export NUM_WORKERS=${NUM_WORKERS:-4}
export PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-false}
export PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
export MASTER_PORT=${MASTER_PORT:-29500}
export SEED=${SEED:-42}
export WANDB_MODE=${WANDB_MODE:-offline}

cd "${REPO_ROOT}"
"${PYTHON_BIN}" \
  examples/Robocasa_tabletop/train_files/preflight_robocasa_full24.py \
  --repo-root "${REPO_ROOT}" \
  --base-model "${COSMOS_BASE:-${REPO_ROOT}/Cosmos-Predict2.5-2B}" \
  --data-root "${DATA_ROOT_DIR}" \
  --output "${OUTPUT_DIR}/preflight.json" \
  --required-gpus "${NUM_PROCESSES}" \
  --min-free-gib "${MIN_FREE_GIB:-500}"

"${PYTHON_BIN}" - "${OUTPUT_DIR}/launch_config.json" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

keys = [
    "RUN_ID",
    "DATA_MIX",
    "DATA_ROOT_DIR",
    "CONFIG_YAML",
    "ACTION_VAE_CHECKPOINT",
    "POLICY_ACTION_STATISTICS_PATH",
    "NUM_PROCESSES",
    "PER_DEVICE_BATCH_SIZE",
    "GRADIENT_ACCUMULATION_STEPS",
    "ENABLE_GRADIENT_CHECKPOINTING",
    "CACHE_PROMPT_EMBEDDINGS",
    "REUSE_CAUSAL_VAE_ENCODE",
    "MAX_TRAIN_STEPS",
    "NUM_WARMUP_STEPS",
    "BASE_LEARNING_RATE",
    "BACKBONE_LEARNING_RATE",
    "ACTION_LEARNING_RATE",
    "MIN_LEARNING_RATE",
    "SAVE_INTERVAL",
    "FULL_STATE_SAVE_INTERVAL",
    "MAX_FULL_TRAINING_STATES",
    "SAVE_TRAINABLE_ONLY_CHECKPOINTS",
    "FULL_MODEL_SAVE_INTERVAL",
    "SEED",
]
payload = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "environment": {key: os.environ.get(key) for key in keys},
    "global_batch_size": (
        int(os.environ["NUM_PROCESSES"])
        * int(os.environ["PER_DEVICE_BATCH_SIZE"])
        * int(os.environ["GRADIENT_ACCUMULATION_STEPS"])
    ),
    "optimizer_states_retained": int(os.environ["MAX_FULL_TRAINING_STATES"]),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n")
PY

cp "$0" "${OUTPUT_DIR}/"
exec bash examples/Robocasa_tabletop/train_files/run_robocasa.sh
