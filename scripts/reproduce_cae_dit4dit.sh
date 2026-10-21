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
# Environment (MODEL_ZOO.md names every file): what each launcher's --help lists (LIBERO_ROOT, EVAL_IMAGE, ROBOSUITE_OVERLAY,
# INSTINCT_SIM_BUILD for --sim instinct, ...), with the checkpoints named per model below:
#   CAE_DIT4DIT_CKPT                              CAE-DiT4DiT (K8 x d48 latent-action) policy checkpoint that you train
#                                                (not released; docs/cae-dit4dit-robocasa.md)
#   DIT4DIT_MODEL_PYTHON, DIT4DIT_PYOVERLAY       the DiT4DiT environments DiT4DiT/environments/setup_dit4dit_rocm_envs.sh
#   DIT4DIT_LIBERO_PYTHON, ROBOCASA_PYTHON        builds and prints (interpreters inside EVAL_IMAGE; DIT4DIT_PYTHON sets
#                                                any of the three Pythons left unset); docs/side-by-side-source-trees.md
#
# Runtime, approximate, per configuration on one AMD Instinct™ MI355X node (measured on our runs):
#   DiT4DiT RoboCasa-GR1 full-24 (1,200 episodes): about 7 h on 2 GPUs with --clients-per-gpu 6 (--sim instinct);
#   runs execute one after another, so the whole script takes many hours; run it under nohup or tmux.
# The reference MI355X results were measured with --sim instinct: to match them, run the commented twins instead.
# They are means over several full runs (each guide gives the count); a single run varies around the mean by about their spread.
#
# Where the runs execute: start the script on the host (it needs docker). FastWAM runs in EVAL_IMAGE containers
# the launcher starts; each DiT4DiT launch runs inside EVAL_IMAGE through scripts/in_eval_image.sh. A DiT4DiT
# LIBERO launch takes the first GPU of EVAL_GPUS (one policy server); RoboCasa runs one server per listed GPU.
#
# Flags: --help, --dry-run (print every resolved run without assets), --smoke (1 trial per task and one latency
# round with the real checkpoints, under <default>-smoke), --only REGEX (only the launches whose command line
# matches), --check (check the environment only).
# Every variable above is checked before the first run starts.
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "$SELF")/.."
# shellcheck source=scripts/reproduce_common.sh
source scripts/reproduce_common.sh
repro_init "$SELF" "$@"
if [[ -z "${CAE_DIT4DIT_CKPT:-}" ]] && ((!REPRO_DRY_RUN)); then
  echo "CAE_DIT4DIT_CKPT is not set. No CAE-DiT4DiT checkpoint is released: train your own with" >&2
  echo "scripts/train_dit4dit_robocasa_action_vae.sh and scripts/train_dit4dit_robocasa_latent_policy.sh" >&2
  echo "(docs/cae-dit4dit-robocasa.md), then set CAE_DIT4DIT_CKPT to the trained policy checkpoint." >&2
fi
repro_require CAE_DIT4DIT_CKPT ROBOSUITE_OVERLAY EVAL_IMAGE DIT4DIT_PYOVERLAY "DIT4DIT_MODEL_PYTHON|DIT4DIT_PYTHON" "ROBOCASA_PYTHON|DIT4DIT_PYTHON"
R="$(repro_dir eval/outputs/reproduce_cae_dit4dit)"
mkdir -p "$R"

cae_dit4dit() { DIT4DIT_ROBOCASA_CKPT="${CAE_DIT4DIT_CKPT:-}" "$@"; }

# DiT4DiT RoboCasa-GR1 full-24, CAE-DiT4DiT (Compact Action Expert)
cae_dit4dit dit4dit run bash scripts/eval_dit4dit_robocasa.sh --sim mujoco --arm cae --full --output-dir "$R/dit4dit-robocasa-cae-mujoco"
# cae_dit4dit dit4dit run bash scripts/eval_dit4dit_robocasa.sh --sim instinct --arm cae --full --output-dir "$R/dit4dit-robocasa-cae-instinct"
