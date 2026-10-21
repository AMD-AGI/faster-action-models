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
# Environment (MODEL_ZOO.md names every file): what each launcher's --help lists (LIBERO_ROOT, EVAL_IMAGE, ROBOSUITE_OVERLAY,
# INSTINCT_SIM_BUILD for --sim instinct, ...), with the checkpoints named per model below:
#   FASTWAM_CKPT, FASTWAM_DATASET_STATS          FastWAM checkpoint and its normalization statistics: yuanty/fastwam
#                                                libero_uncond_2cam224.pt and libero_uncond_2cam224_dataset_stats.json
#   DIT4DIT_LIBERO_CKPT                           DiT4DiT LIBERO checkpoint: mondo-robotics/dit4dit-model
#                                                dit4dit_libero/final_model/pytorch_model.pt
#   DIT4DIT_ROBOCASA_CKPT                         the DiT4DiT authors' RoboCasa-GR1 checkpoint: mondo-robotics/dit4dit-model
#                                                dit4dit_robocasa_gr1/final_model/pytorch_model.pt
#
# Runtime, approximate, per configuration on one AMD Instinct™ MI355X node (measured on our runs):
#   FastWAM LIBERO --full (2,000 episodes), FastWAM's own evaluator, one worker per GPU in EVAL_GPUS (default 0):
#   about 12-20 h per configuration on CPU MuJoCo (OSMesa) and 5-10 h with --sim instinct; with
#   --sim instinct --mode batched about 2 h. EVAL_GPUS=0,1,2,3 spreads the 40 tasks over four GPUs;
#   DiT4DiT RoboCasa-GR1 full-24 (1,200 episodes): about 7 h on 2 GPUs with --clients-per-gpu 6 (--sim instinct);
#   runs execute one after another, so the whole script takes many hours; run it under nohup or tmux.
# The reference MI355X results were measured with --sim instinct: to match them, run the commented twins instead.
# They are means over several full runs (each guide gives the count); a single run varies around the mean by about their spread.
# The FastWAM runs use eval/native/run_fastwam_libero.sh, FastWAM's own evaluator under the repository protocol
# (eval/PROTOCOL.md).
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
repro_require EVAL_IMAGE LIBERO_ROOT FASTWAM_CKPT FASTWAM_DATASET_STATS DIT4DIT_LIBERO_CKPT DIT4DIT_ROBOCASA_CKPT ROBOSUITE_OVERLAY
R="$(repro_dir eval/outputs/reproduce_baselines)"
mkdir -p "$R"

fastwam() { EVAL_GPUS="${EVAL_GPUS:-0}" MODEL_CKPT="${FASTWAM_CKPT:-}" DATASET_STATS="${FASTWAM_DATASET_STATS:-}" "$@"; }

# FastWAM LIBERO, baseline (upstream FastWAM), 10 steps
fastwam run bash eval/native/run_fastwam_libero.sh --sim mujoco --arm baseline --full --output-dir "$R/fastwam-libero-baseline-10step-mujoco"
# fastwam run bash eval/native/run_fastwam_libero.sh --sim instinct --arm baseline --full --output-dir "$R/fastwam-libero-baseline-10step-instinct"

# DiT4DiT LIBERO, baseline (upstream DiT4DiT), 4 steps
run bash scripts/eval_dit4dit_libero.sh --sim mujoco --arm baseline --steps 4 --full --output-dir "$R/dit4dit-libero-baseline-4step-mujoco"
# run bash scripts/eval_dit4dit_libero.sh --sim instinct --arm baseline --steps 4 --full --output-dir "$R/dit4dit-libero-baseline-4step-instinct"

# DiT4DiT RoboCasa-GR1 full-24, baseline (upstream DiT4DiT), 4 steps
run bash scripts/eval_dit4dit_robocasa.sh --sim mujoco --arm baseline --steps 4 --full --output-dir "$R/dit4dit-robocasa-baseline-4step-mujoco"
# run bash scripts/eval_dit4dit_robocasa.sh --sim instinct --arm baseline --steps 4 --full --output-dir "$R/dit4dit-robocasa-baseline-4step-instinct"
