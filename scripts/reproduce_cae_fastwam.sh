#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# CAE-FastWAM on LIBERO: baseline, GEAK, and CAE-FastWAM at 10 and 4 steps -- success rate and latency.
#
# One run of each configuration. Every evaluation runs on CPU MuJoCo (--sim mujoco); the GPU-native run of
# the same configuration on AMD Instinct™ GPUs (--sim instinct) sits under it, commented out -- uncomment
# it to run it. Results go under $REPRO_DIR (default eval/outputs/reproduce_cae_fastwam).
#
# Environment: what each launcher's --help lists (LIBERO_ROOT, EVAL_IMAGE, EVAL_HARNESS, ROBOSUITE_OVERLAY,
# INSTINCT_SIM_BUILD for --sim instinct, ...), with the checkpoints named per model below:
#   FASTWAM_CKPT, FASTWAM_DATASET_STATS          FastWAM checkpoint and its normalization statistics
#   CAE_FASTWAM_CKPT, CAE_FASTWAM_ACTION_VAE_CKPT, CAE_FASTWAM_STATS_TEMPLATE ({suite}), CAE_FASTWAM_TASK_CONFIG
#                                                CAE-FastWAM checkpoint, its action VAE, per-suite statistics, task config
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
R="${REPRO_DIR:-eval/outputs/reproduce_cae_fastwam}"
mkdir -p "$R"

fastwam() { MODEL_CKPT="$FASTWAM_CKPT" DATASET_STATS="$FASTWAM_DATASET_STATS" "$@"; }
cae_fastwam() { MODEL_CKPT="$CAE_FASTWAM_CKPT" ACTION_VAE_CKPT="$CAE_FASTWAM_ACTION_VAE_CKPT" DATASET_STATS_TEMPLATE="$CAE_FASTWAM_STATS_TEMPLATE" TASK_CONFIG="$CAE_FASTWAM_TASK_CONFIG" "$@"; }

# FastWAM LIBERO, baseline (upstream FastWAM), 10 steps
fastwam bash scripts/eval_fastwam_libero.sh --sim mujoco --arm baseline --full --output-dir "$R/fastwam-libero-baseline-10step-mujoco"
# fastwam bash eval/native/run_fastwam_libero.sh --sim instinct --arm baseline --full --output-dir "$R/fastwam-libero-baseline-10step-instinct"

# FastWAM LIBERO, GEAK, 10 steps
fastwam bash scripts/eval_fastwam_libero.sh --sim mujoco --arm geak --steps 10 --full --output-dir "$R/fastwam-libero-geak-10step-mujoco"
# fastwam bash eval/native/run_fastwam_libero.sh --sim instinct --arm geak --steps 10 --full --output-dir "$R/fastwam-libero-geak-10step-instinct"

# FastWAM LIBERO, CAE-FastWAM (Compact Action Expert, GEAK), 10 steps
cae_fastwam bash scripts/eval_fastwam_libero.sh --sim mujoco --arm cae --steps 10 --full --output-dir "$R/fastwam-libero-cae-10step-mujoco"
# cae_fastwam bash eval/native/run_fastwam_libero.sh --sim instinct --arm cae --steps 10 --full --output-dir "$R/fastwam-libero-cae-10step-instinct"

# FastWAM LIBERO, CAE-FastWAM (Compact Action Expert, GEAK), 4 steps
cae_fastwam bash scripts/eval_fastwam_libero.sh --sim mujoco --arm cae --steps 4 --full --output-dir "$R/fastwam-libero-cae-4step-mujoco"
# cae_fastwam bash eval/native/run_fastwam_libero.sh --sim instinct --arm cae --steps 4 --full --output-dir "$R/fastwam-libero-cae-4step-instinct"

# FastWAM latency: baseline (eager), naive torch.compile and GEAK cells, interleaved on one GPU
fastwam python3 scripts/compare_artifacts.py --arms eager,compile,geak --rounds 5 --json-out "$R/latency-fastwam.json"

# CAE-FastWAM latency: 10 and 4 steps, interleaved on one GPU
cae_fastwam python3 scripts/cae_fastwam_latency.py --arms cae@10,cae@4 --rounds 5 --json-out "$R/latency-cae-fastwam.json"
