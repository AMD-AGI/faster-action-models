#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Baseline success rates: FastWAM and DiT4DiT on LIBERO, DiT4DiT on RoboCasa-GR1, unmodified upstream trees.
#
# One run of each configuration. Every evaluation runs on CPU MuJoCo (--sim mujoco); the GPU-native run of
# the same configuration on AMD Instinct™ GPUs (--sim instinct) sits under it, commented out -- uncomment
# it to run it. Results go under $REPRO_DIR (default eval/outputs/reproduce_baselines).
#
# Environment: what each launcher's --help lists (LIBERO_ROOT, EVAL_IMAGE, EVAL_HARNESS, ROBOSUITE_OVERLAY,
# INSTINCT_SIM_BUILD for --sim instinct, ...), with the checkpoints named per model below:
#   FASTWAM_CKPT, FASTWAM_DATASET_STATS          FastWAM checkpoint and its normalization statistics
#   DIT4DIT_LIBERO_CKPT                           DiT4DiT LIBERO checkpoint
#   DIT4DIT_ROBOCASA_CKPT                         the DiT4DiT authors' RoboCasa-GR1 checkpoint
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
R="${REPRO_DIR:-eval/outputs/reproduce_baselines}"
mkdir -p "$R"

fastwam() { MODEL_CKPT="$FASTWAM_CKPT" DATASET_STATS="$FASTWAM_DATASET_STATS" "$@"; }

# FastWAM LIBERO, baseline (upstream FastWAM), 10 steps
fastwam bash scripts/eval_fastwam_libero.sh --sim mujoco --arm baseline --full --output-dir "$R/fastwam-libero-baseline-10step-mujoco"
# fastwam bash eval/native/run_fastwam_libero.sh --sim instinct --arm baseline --full --output-dir "$R/fastwam-libero-baseline-10step-instinct"

# DiT4DiT LIBERO, baseline (upstream DiT4DiT), 4 steps
bash scripts/eval_dit4dit_libero.sh --sim mujoco --arm baseline --steps 4 --full --output-dir "$R/dit4dit-libero-baseline-4step-mujoco"
# bash scripts/eval_dit4dit_libero.sh --sim instinct --arm baseline --steps 4 --full --output-dir "$R/dit4dit-libero-baseline-4step-instinct"

# DiT4DiT RoboCasa-GR1 full-24, baseline (upstream DiT4DiT), 4 steps
bash scripts/eval_dit4dit_robocasa.sh --sim mujoco --arm baseline --steps 4 --full --output-dir "$R/dit4dit-robocasa-baseline-4step-mujoco"
# bash scripts/eval_dit4dit_robocasa.sh --sim instinct --arm baseline --steps 4 --full --output-dir "$R/dit4dit-robocasa-baseline-4step-instinct"
