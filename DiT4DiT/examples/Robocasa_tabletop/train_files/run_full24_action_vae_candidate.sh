#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
CANDIDATE=${1:?Usage: run_full24_action_vae_candidate.sh CANDIDATE}
GPU_ID=${GPU_ID:-0}
CACHE=${FULL24_ACTION_CACHE:-"${REPO_ROOT}/playground/action_vae/full24_action_chunks_per_task.pt"}
LANGUAGE_EMBEDDINGS=${FULL24_LANGUAGE_EMBEDDINGS:-"${CACHE%.pt}.instruction_embeddings.pt"}
ARTIFACT_ROOT=${FULL24_VAE_ROOT:-"${REPO_ROOT}/playground/action_vae/full24_sweep"}
STATE_ROOT=${FULL24_VAE_STATE_ROOT:-"${REPO_ROOT}/playground/full24_vae_sweep"}

case "${CANDIDATE}" in
  k8d32h128_encdec_noalign)
    TOKENS=8; LATENT=32; HIDDEN=128; MODE=encoder_decoder; ALIGNMENT=0 ;;
  k8d32h256_encdec_noalign)
    TOKENS=8; LATENT=32; HIDDEN=256; MODE=encoder_decoder; ALIGNMENT=0 ;;
  k8d48h256_encdec_noalign)
    TOKENS=8; LATENT=48; HIDDEN=256; MODE=encoder_decoder; ALIGNMENT=0 ;;
  k8d32h256_none_align)
    TOKENS=8; LATENT=32; HIDDEN=256; MODE=none; ALIGNMENT=0.001 ;;
  k8d32h256_enconly_align)
    TOKENS=8; LATENT=32; HIDDEN=256; MODE=encoder_only; ALIGNMENT=0.001 ;;
  k8d48h256_none_align)
    TOKENS=8; LATENT=48; HIDDEN=256; MODE=none; ALIGNMENT=0.001 ;;
  *)
    echo "Unknown full24 VAE candidate: ${CANDIDATE}" >&2
    exit 2
    ;;
esac

OUTPUT="${ARTIFACT_ROOT}/${CANDIDATE}.pt"
LOG_PATH="${STATE_ROOT}/${CANDIDATE}.log"
DONE_PATH="${STATE_ROOT}/${CANDIDATE}.done.json"
FAILED_PATH="${STATE_ROOT}/${CANDIDATE}.failed.json"
LOCK_PATH="${REPO_ROOT}/playground/locks/full24_vae_${CANDIDATE}.lock"

[[ -f "${CACHE}" ]] || { echo "Missing full24 action cache: ${CACHE}" >&2; exit 1; }
if [[ "${ALIGNMENT}" != 0 ]]; then
  [[ -f "${LANGUAGE_EMBEDDINGS}" ]] || {
    echo "Missing frozen instruction embeddings: ${LANGUAGE_EMBEDDINGS}" >&2
    exit 1
  }
fi
mkdir -p "${ARTIFACT_ROOT}" "${STATE_ROOT}" "$(dirname "${LOCK_PATH}")"
exec 9>"${LOCK_PATH}"
flock -n 9 || {
  echo "Candidate lock is already held: ${LOCK_PATH}" >&2
  exit 1
}
if [[ -f "${OUTPUT}" && -f "${DONE_PATH}" ]]; then
  echo "Reusing completed candidate: ${OUTPUT}"
  exit 0
fi
if [[ -e "${OUTPUT}" || -e "${DONE_PATH}" ]]; then
  echo "Refusing to overwrite incomplete or inconsistent candidate state: ${OUTPUT}" >&2
  exit 1
fi

write_sentinel() {
  local status=$1
  local target=$2
  local temporary="${target}.tmp.$$"
  "${PYTHON_BIN}" - "${temporary}" "${status}" "${OUTPUT}" "${CANDIDATE}" "${GPU_ID}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, status, checkpoint, candidate, gpu = sys.argv[1:]
Path(path).write_text(json.dumps({
    "status": status,
    "candidate": candidate,
    "gpu": int(gpu),
    "checkpoint": checkpoint,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
PY
  mv "${temporary}" "${target}"
}

trap 'code=$?; write_sentinel "failed:${code}" "${FAILED_PATH}"; exit "${code}"' ERR
rm -f "${DONE_PATH}" "${FAILED_PATH}"
cd "${REPO_ROOT}"

alignment_args=()
if [[ "${ALIGNMENT}" != 0 ]]; then
  alignment_args=(
    --language-embeddings "${LANGUAGE_EMBEDDINGS}"
    --alignment-weight "${ALIGNMENT}"
  )
fi

HIP_VISIBLE_DEVICES="${GPU_ID}" CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON_BIN}" examples/Robocasa_tabletop/train_files/train_action_vae.py \
  --cache "${CACHE}" \
  --output "${OUTPUT}" \
  --num-tokens "${TOKENS}" \
  --latent-dim "${LATENT}" \
  --encoder-hidden "${HIDDEN}" \
  --decoder-hidden "${HIDDEN}" \
  --num-tasks 24 \
  --conditioning-mode "${MODE}" \
  --epochs 50 \
  --batch-size 256 \
  --seed 42 \
  "${alignment_args[@]}" \
  2>&1 | tee "${LOG_PATH}"
write_sentinel completed "${DONE_PATH}"
