#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Every documented evaluation launch: each launcher, arm and advertised step count, one run each.
#
# One run of each configuration. Every evaluation runs on CPU MuJoCo (--sim mujoco); the GPU-native run of
# the same configuration on AMD Instinct™ GPUs (--sim instinct) sits under it, commented out -- uncomment
# it to run it. Results go under $REPRO_DIR (default eval/outputs/run_all_evals).
#
# Environment: what each launcher's --help lists (LIBERO_ROOT, EVAL_IMAGE, EVAL_HARNESS, ROBOSUITE_OVERLAY,
# INSTINCT_SIM_BUILD for --sim instinct, ...), with the checkpoints named per model below:
#   FASTWAM_CKPT, FASTWAM_DATASET_STATS          FastWAM checkpoint and its normalization statistics
#   CAE_FASTWAM_CKPT, CAE_FASTWAM_ACTION_VAE_CKPT, CAE_FASTWAM_STATS_TEMPLATE ({suite}), CAE_FASTWAM_TASK_CONFIG
#                                                CAE-FastWAM checkpoint, its action VAE, per-suite statistics, task config
#   DIT4DIT_LIBERO_CKPT                           DiT4DiT LIBERO checkpoint
#   DIT4DIT_ROBOCASA_CKPT                         the DiT4DiT authors' RoboCasa-GR1 checkpoint
#   CAE_DIT4DIT_CKPT                              CAE-DiT4DiT (K8 x d48 latent-action) policy checkpoint
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
R="${REPRO_DIR:-eval/outputs/run_all_evals}"
mkdir -p "$R"

fastwam() { MODEL_CKPT="$FASTWAM_CKPT" DATASET_STATS="$FASTWAM_DATASET_STATS" "$@"; }
cae_fastwam() { MODEL_CKPT="$CAE_FASTWAM_CKPT" ACTION_VAE_CKPT="$CAE_FASTWAM_ACTION_VAE_CKPT" DATASET_STATS_TEMPLATE="$CAE_FASTWAM_STATS_TEMPLATE" TASK_CONFIG="$CAE_FASTWAM_TASK_CONFIG" "$@"; }
cae_dit4dit() { DIT4DIT_ROBOCASA_CKPT="$CAE_DIT4DIT_CKPT" "$@"; }

# FastWAM LIBERO, baseline (upstream FastWAM), 10 steps
fastwam bash scripts/eval_fastwam_libero.sh --sim mujoco --arm baseline --full --output-dir "$R/fastwam-libero-baseline-10step-mujoco"
# fastwam bash eval/native/run_fastwam_libero.sh --sim instinct --arm baseline --full --output-dir "$R/fastwam-libero-baseline-10step-instinct"

# FastWAM LIBERO, GEAK, 10 steps
fastwam bash scripts/eval_fastwam_libero.sh --sim mujoco --arm geak --steps 10 --full --output-dir "$R/fastwam-libero-geak-10step-mujoco"
# fastwam bash eval/native/run_fastwam_libero.sh --sim instinct --arm geak --steps 10 --full --output-dir "$R/fastwam-libero-geak-10step-instinct"

# FastWAM LIBERO, GEAK, 2 steps
fastwam bash scripts/eval_fastwam_libero.sh --sim mujoco --arm geak --steps 2 --full --output-dir "$R/fastwam-libero-geak-2step-mujoco"
# fastwam bash eval/native/run_fastwam_libero.sh --sim instinct --arm geak --steps 2 --full --output-dir "$R/fastwam-libero-geak-2step-instinct"

# FastWAM LIBERO, CAE-FastWAM (Compact Action Expert, GEAK), 10 steps
cae_fastwam bash scripts/eval_fastwam_libero.sh --sim mujoco --arm cae --steps 10 --full --output-dir "$R/fastwam-libero-cae-10step-mujoco"
# cae_fastwam bash eval/native/run_fastwam_libero.sh --sim instinct --arm cae --steps 10 --full --output-dir "$R/fastwam-libero-cae-10step-instinct"

# FastWAM LIBERO, CAE-FastWAM (Compact Action Expert, GEAK), 4 steps
cae_fastwam bash scripts/eval_fastwam_libero.sh --sim mujoco --arm cae --steps 4 --full --output-dir "$R/fastwam-libero-cae-4step-mujoco"
# cae_fastwam bash eval/native/run_fastwam_libero.sh --sim instinct --arm cae --steps 4 --full --output-dir "$R/fastwam-libero-cae-4step-instinct"

# DiT4DiT LIBERO, baseline, 4 steps
bash scripts/eval_dit4dit_libero.sh --sim mujoco --arm baseline --steps 4 --full --output-dir "$R/dit4dit-libero-baseline-4step-mujoco"
# bash scripts/eval_dit4dit_libero.sh --sim instinct --arm baseline --steps 4 --full --output-dir "$R/dit4dit-libero-baseline-4step-instinct"

# DiT4DiT LIBERO, baseline, 2 steps
bash scripts/eval_dit4dit_libero.sh --sim mujoco --arm baseline --steps 2 --full --output-dir "$R/dit4dit-libero-baseline-2step-mujoco"
# bash scripts/eval_dit4dit_libero.sh --sim instinct --arm baseline --steps 2 --full --output-dir "$R/dit4dit-libero-baseline-2step-instinct"

# DiT4DiT LIBERO, GEAK, 4 steps
bash scripts/eval_dit4dit_libero.sh --sim mujoco --arm geak --steps 4 --full --output-dir "$R/dit4dit-libero-geak-4step-mujoco"
# bash scripts/eval_dit4dit_libero.sh --sim instinct --arm geak --steps 4 --full --output-dir "$R/dit4dit-libero-geak-4step-instinct"

# DiT4DiT LIBERO, GEAK, 2 steps
bash scripts/eval_dit4dit_libero.sh --sim mujoco --arm geak --steps 2 --full --output-dir "$R/dit4dit-libero-geak-2step-mujoco"
# bash scripts/eval_dit4dit_libero.sh --sim instinct --arm geak --steps 2 --full --output-dir "$R/dit4dit-libero-geak-2step-instinct"

# DiT4DiT RoboCasa-GR1 full-24, baseline, 4 steps
bash scripts/eval_dit4dit_robocasa.sh --sim mujoco --arm baseline --steps 4 --full --output-dir "$R/dit4dit-robocasa-baseline-4step-mujoco"
# bash scripts/eval_dit4dit_robocasa.sh --sim instinct --arm baseline --steps 4 --full --output-dir "$R/dit4dit-robocasa-baseline-4step-instinct"

# DiT4DiT RoboCasa-GR1 full-24, baseline, 2 steps
bash scripts/eval_dit4dit_robocasa.sh --sim mujoco --arm baseline --steps 2 --full --output-dir "$R/dit4dit-robocasa-baseline-2step-mujoco"
# bash scripts/eval_dit4dit_robocasa.sh --sim instinct --arm baseline --steps 2 --full --output-dir "$R/dit4dit-robocasa-baseline-2step-instinct"

# DiT4DiT RoboCasa-GR1 full-24, GEAK, 4 steps
bash scripts/eval_dit4dit_robocasa.sh --sim mujoco --arm geak --steps 4 --full --output-dir "$R/dit4dit-robocasa-geak-4step-mujoco"
# bash scripts/eval_dit4dit_robocasa.sh --sim instinct --arm geak --steps 4 --full --output-dir "$R/dit4dit-robocasa-geak-4step-instinct"

# DiT4DiT RoboCasa-GR1 full-24, GEAK, 2 steps
bash scripts/eval_dit4dit_robocasa.sh --sim mujoco --arm geak --steps 2 --full --output-dir "$R/dit4dit-robocasa-geak-2step-mujoco"
# bash scripts/eval_dit4dit_robocasa.sh --sim instinct --arm geak --steps 2 --full --output-dir "$R/dit4dit-robocasa-geak-2step-instinct"

# DiT4DiT RoboCasa-GR1 full-24, CAE-DiT4DiT (Compact Action Expert)
cae_dit4dit bash scripts/eval_dit4dit_robocasa.sh --sim mujoco --arm cae --full --output-dir "$R/dit4dit-robocasa-cae-mujoco"
# cae_dit4dit bash scripts/eval_dit4dit_robocasa.sh --sim instinct --arm cae --full --output-dir "$R/dit4dit-robocasa-cae-instinct"
