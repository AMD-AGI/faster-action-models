#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# FastWAM LIBERO evaluation: a thin wrapper over eval/native/run_fastwam_libero.sh, FastWAM's own evaluator.
# Every flag goes to the native launcher unchanged; the only default this wrapper adds is one GPU (--gpus 0,
# unless --gpus or EVAL_GPUS says otherwise), as this script always defaulted to.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NATIVE="${ROOT}/eval/native/run_fastwam_libero.sh"
# shellcheck source=eval_flags.sh
source "${ROOT}/scripts/eval_flags.sh"

for a in "$@"; do
  if [[ "$a" == -h || "$a" == --help ]]; then
    cat <<'EOF'
scripts/eval_fastwam_libero.sh: FastWAM LIBERO evaluation (baseline, GEAK or CAE-FastWAM arm).

This is a wrapper over eval/native/run_fastwam_libero.sh: it passes every flag through and defaults to one GPU
(--gpus 0; EVAL_GPUS or --gpus override it). Results land under <output-dir>/<suite>/ (FastWAM's per-task result
files and rollout videos) and <output-dir>/summary.json. --port, --clients-per-gpu and --shards are refused: the
evaluator runs no policy server; each GPU in --gpus runs one worker.

The native launcher's help follows.

EOF
    exec bash "$NATIVE" --help
  fi
done

# Variables of the former policy-server engine; the native launcher fixes the source tree and configs per --arm
# and takes everything else from its flags and eval/PROTOCOL.md.
eval_ignore_env "the native evaluator fixes the source tree and configs per --arm" \
  MODEL_CONFIGS BASELINE_SRC OPTIMIZED_SRC CAE_SRC
eval_ignore_env "the native evaluator runs no policy server; the protocol and the flags set it" \
  PORT EVAL_PORT SEED WAIT SHARDS SUITES_CSV EVAL_ARM_LABEL

export EVAL_GPUS="${EVAL_GPUS:-0}"
exec bash "$NATIVE" "$@"
