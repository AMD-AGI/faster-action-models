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
# Environment (MODEL_ZOO.md names every file): what each launcher's --help lists (LIBERO_ROOT, EVAL_IMAGE, ROBOSUITE_OVERLAY,
# INSTINCT_SIM_BUILD for --sim instinct, ...), with the checkpoints named per model below:
#   FASTWAM_CKPT, FASTWAM_DATASET_STATS          FastWAM checkpoint and its normalization statistics: yuanty/fastwam
#                                                libero_uncond_2cam224.pt and libero_uncond_2cam224_dataset_stats.json
#   CAE_FASTWAM_CKPT, CAE_FASTWAM_ACTION_VAE_CKPT, CAE_FASTWAM_STATS_TEMPLATE ({suite}), CAE_FASTWAM_TASK_CONFIG
#                                                CAE-FastWAM checkpoint, its action VAE, per-suite statistics, task config:
#                                                amd/CAE-FastWAM policy.pt, action_vae.pt, stats/{suite}.json and
#                                                libero_uncond_2cam224_1e-4_32hdim_latent_K8d16_freebits_pernorm_joint
#   MODEL_SRC_ROOT, HF_CACHE_DIR, MODEL_PYTHON   for the FastWAM latency tool (scripts/compare_artifacts.py runs the GEAK
#                                                task's `performance` command): a FastWAM checkout (this repository's
#                                                FastWAM/), a writable cache directory, and the image's Python
#                                                (/opt/venv/bin/python3); docs/fastwam-geak-inference-optimization.md
#
# Runtime, approximate, per configuration on one AMD Instinct™ MI355X node (measured on our runs):
#   FastWAM LIBERO --full (2,000 episodes), FastWAM's own evaluator, one worker per GPU in EVAL_GPUS (default 0):
#   about 12-20 h per configuration on CPU MuJoCo (OSMesa) and 5-10 h with --sim instinct; with
#   --sim instinct --mode batched about 2 h. EVAL_GPUS=0,1,2,3 spreads the 40 tasks over four GPUs;
#   runs execute one after another, so the whole script takes many hours; run it under nohup or tmux.
# The reference MI355X results were measured with --sim instinct: to match them, run the commented twins instead.
# They are means of 11 full runs per configuration (5 for CAE-FastWAM at 4 steps; docs/cae-fastwam-libero.md);
# a single run of this script varies around the mean by about the stated spread.
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
repro_require EVAL_IMAGE LIBERO_ROOT MODEL_SRC_ROOT HF_CACHE_DIR MODEL_PYTHON FASTWAM_CKPT FASTWAM_DATASET_STATS CAE_FASTWAM_CKPT CAE_FASTWAM_ACTION_VAE_CKPT CAE_FASTWAM_STATS_TEMPLATE CAE_FASTWAM_TASK_CONFIG
R="$(repro_dir eval/outputs/reproduce_cae_fastwam)"
mkdir -p "$R"

fastwam() { EVAL_GPUS="${EVAL_GPUS:-0}" MODEL_CKPT="${FASTWAM_CKPT:-}" DATASET_STATS="${FASTWAM_DATASET_STATS:-}" "$@"; }
cae_fastwam() { EVAL_GPUS="${EVAL_GPUS:-0}" MODEL_CKPT="${CAE_FASTWAM_CKPT:-}" ACTION_VAE_CKPT="${CAE_FASTWAM_ACTION_VAE_CKPT:-}" DATASET_STATS_TEMPLATE="${CAE_FASTWAM_STATS_TEMPLATE:-}" TASK_CONFIG="${CAE_FASTWAM_TASK_CONFIG:-}" "$@"; }

# FastWAM LIBERO, baseline (upstream FastWAM), 10 steps
fastwam run bash eval/native/run_fastwam_libero.sh --sim mujoco --arm baseline --full --output-dir "$R/fastwam-libero-baseline-10step-mujoco"
# fastwam run bash eval/native/run_fastwam_libero.sh --sim instinct --arm baseline --full --output-dir "$R/fastwam-libero-baseline-10step-instinct"

# FastWAM LIBERO, GEAK, 10 steps
fastwam run bash eval/native/run_fastwam_libero.sh --sim mujoco --arm geak --steps 10 --full --output-dir "$R/fastwam-libero-geak-10step-mujoco"
# fastwam run bash eval/native/run_fastwam_libero.sh --sim instinct --arm geak --steps 10 --full --output-dir "$R/fastwam-libero-geak-10step-instinct"

# FastWAM LIBERO, CAE-FastWAM (Compact Action Expert, GEAK), 10 steps
cae_fastwam run bash eval/native/run_fastwam_libero.sh --sim mujoco --arm cae --steps 10 --full --output-dir "$R/fastwam-libero-cae-10step-mujoco"
# cae_fastwam run bash eval/native/run_fastwam_libero.sh --sim instinct --arm cae --steps 10 --full --output-dir "$R/fastwam-libero-cae-10step-instinct"

# FastWAM LIBERO, CAE-FastWAM (Compact Action Expert, GEAK), 4 steps
cae_fastwam run bash eval/native/run_fastwam_libero.sh --sim mujoco --arm cae --steps 4 --full --output-dir "$R/fastwam-libero-cae-4step-mujoco"
# cae_fastwam run bash eval/native/run_fastwam_libero.sh --sim instinct --arm cae --steps 4 --full --output-dir "$R/fastwam-libero-cae-4step-instinct"

# FastWAM latency: baseline (eager), naive torch.compile and GEAK cells, interleaved on one GPU
fastwam latency python3 scripts/compare_artifacts.py --arms eager,compile,geak --rounds 5 --json-out "$R/latency-fastwam.json"

# CAE-FastWAM latency: 10 and 4 steps, interleaved on one GPU
cae_fastwam latency python3 scripts/cae_fastwam_latency.py --arms cae@10,cae@4 --rounds 5 --json-out "$R/latency-cae-fastwam.json"
