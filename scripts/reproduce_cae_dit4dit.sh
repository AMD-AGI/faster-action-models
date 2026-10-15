#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# CAE-DiT4DiT on RoboCasa-GR1 full-24: the K8 x d48 latent-action policy at its checkpoint's step count.
#
# One run of each configuration. Every evaluation runs on CPU MuJoCo (--sim mujoco); the GPU-native run of
# the same configuration on AMD Instinct™ GPUs (--sim instinct) sits under it, commented out -- uncomment
# it to run it. Results go under $REPRO_DIR (default eval/outputs/reproduce_cae_dit4dit).
#
# Environment: what each launcher's --help lists (LIBERO_ROOT, EVAL_IMAGE, EVAL_HARNESS, ROBOSUITE_OVERLAY,
# INSTINCT_SIM_BUILD for --sim instinct, ...), with the checkpoints named per model below:
#   CAE_DIT4DIT_CKPT                              CAE-DiT4DiT (K8 x d48 latent-action) policy checkpoint
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
R="${REPRO_DIR:-eval/outputs/reproduce_cae_dit4dit}"
mkdir -p "$R"

cae_dit4dit() { DIT4DIT_ROBOCASA_CKPT="$CAE_DIT4DIT_CKPT" "$@"; }

# DiT4DiT RoboCasa-GR1 full-24, CAE-DiT4DiT (Compact Action Expert)
cae_dit4dit bash scripts/eval_dit4dit_robocasa.sh --sim mujoco --arm cae --full --output-dir "$R/dit4dit-robocasa-cae-mujoco"
# cae_dit4dit bash scripts/eval_dit4dit_robocasa.sh --sim instinct --arm cae --full --output-dir "$R/dit4dit-robocasa-cae-instinct"
