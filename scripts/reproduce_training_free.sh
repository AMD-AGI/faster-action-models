#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Training-free optimization: baseline, GEAK, and GEAK with fewer denoising steps -- success rate and latency.
#
# One run of each configuration. Every evaluation runs on CPU MuJoCo (--sim mujoco); the GPU-native run of
# the same configuration on AMD Instinct™ GPUs (--sim instinct) sits under it, commented out -- uncomment
# it to run it. Results go under $REPRO_DIR (default eval/outputs/reproduce_training_free).
#
# Environment (MODEL_ZOO.md names every file): what each launcher's --help lists (LIBERO_ROOT, EVAL_IMAGE, ROBOSUITE_OVERLAY,
# INSTINCT_SIM_BUILD for --sim instinct, ...), with the checkpoints named per model below:
#   FASTWAM_CKPT, FASTWAM_DATASET_STATS          FastWAM checkpoint and its normalization statistics: yuanty/fastwam
#                                                libero_uncond_2cam224.pt and libero_uncond_2cam224_dataset_stats.json
#   DIT4DIT_LIBERO_CKPT                           DiT4DiT LIBERO checkpoint: mondo-robotics/dit4dit-model
#                                                dit4dit_libero/final_model/pytorch_model.pt
#   DIT4DIT_ROBOCASA_CKPT                         the DiT4DiT authors' RoboCasa-GR1 checkpoint: mondo-robotics/dit4dit-model
#                                                dit4dit_robocasa_gr1/final_model/pytorch_model.pt
#   MODEL_SRC_ROOT, HF_CACHE_DIR, MODEL_PYTHON   for the FastWAM latency tool (scripts/compare_artifacts.py runs the GEAK
#                                                task's `performance` command): a FastWAM checkout (this repository's
#                                                FastWAM/), a writable cache directory, and the image's Python
#                                                (/opt/venv/bin/python3); docs/fastwam-geak-inference-optimization.md
#   DIT4DIT_DIFFUSERS_SRC                         for the DiT4DiT latency tool: the model environment's diffusers 3996788
#                                                (envs/model/lib/python3.12/site-packages/diffusers); DIT4DIT_PYOVERLAY
#                                                must also hold transformers 4.57.6 and tokenizers 0.22.2
#                                                (envs/pyoverlay:envs/d4pins; scripts/dit4dit_latency.py --help, RECIPE)
#   DIT4DIT_MODEL_PYTHON, DIT4DIT_PYOVERLAY       the DiT4DiT environments DiT4DiT/environments/setup_dit4dit_rocm_envs.sh
#   DIT4DIT_LIBERO_PYTHON, ROBOCASA_PYTHON        builds and prints (interpreters inside EVAL_IMAGE; DIT4DIT_PYTHON sets
#                                                any of the three Pythons left unset); docs/side-by-side-source-trees.md
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
repro_require EVAL_IMAGE LIBERO_ROOT MODEL_SRC_ROOT HF_CACHE_DIR MODEL_PYTHON FASTWAM_CKPT FASTWAM_DATASET_STATS DIT4DIT_LIBERO_CKPT DIT4DIT_ROBOCASA_CKPT ROBOSUITE_OVERLAY DIT4DIT_PYOVERLAY "DIT4DIT_MODEL_PYTHON|DIT4DIT_PYTHON" "DIT4DIT_LIBERO_PYTHON|DIT4DIT_PYTHON" "ROBOCASA_PYTHON|DIT4DIT_PYTHON" DIT4DIT_DIFFUSERS_SRC
R="$(repro_dir eval/outputs/reproduce_training_free)"
mkdir -p "$R"

fastwam() { EVAL_GPUS="${EVAL_GPUS:-0}" MODEL_CKPT="${FASTWAM_CKPT:-}" DATASET_STATS="${FASTWAM_DATASET_STATS:-}" "$@"; }

# FastWAM LIBERO, baseline (upstream FastWAM), 10 steps
fastwam run bash eval/native/run_fastwam_libero.sh --sim mujoco --arm baseline --full --output-dir "$R/fastwam-libero-baseline-10step-mujoco"
# fastwam run bash eval/native/run_fastwam_libero.sh --sim instinct --arm baseline --full --output-dir "$R/fastwam-libero-baseline-10step-instinct"

# FastWAM LIBERO, GEAK, 10 steps
fastwam run bash eval/native/run_fastwam_libero.sh --sim mujoco --arm geak --steps 10 --full --output-dir "$R/fastwam-libero-geak-10step-mujoco"
# fastwam run bash eval/native/run_fastwam_libero.sh --sim instinct --arm geak --steps 10 --full --output-dir "$R/fastwam-libero-geak-10step-instinct"

# FastWAM LIBERO, GEAK, 2 steps
fastwam run bash eval/native/run_fastwam_libero.sh --sim mujoco --arm geak --steps 2 --full --output-dir "$R/fastwam-libero-geak-2step-mujoco"
# fastwam run bash eval/native/run_fastwam_libero.sh --sim instinct --arm geak --steps 2 --full --output-dir "$R/fastwam-libero-geak-2step-instinct"

# DiT4DiT LIBERO, baseline, 4 steps
dit4dit run bash scripts/eval_dit4dit_libero.sh --sim mujoco --arm baseline --steps 4 --full --output-dir "$R/dit4dit-libero-baseline-4step-mujoco"
# dit4dit run bash scripts/eval_dit4dit_libero.sh --sim instinct --arm baseline --steps 4 --full --output-dir "$R/dit4dit-libero-baseline-4step-instinct"

# DiT4DiT LIBERO, GEAK, 4 steps
dit4dit run bash scripts/eval_dit4dit_libero.sh --sim mujoco --arm geak --steps 4 --full --output-dir "$R/dit4dit-libero-geak-4step-mujoco"
# dit4dit run bash scripts/eval_dit4dit_libero.sh --sim instinct --arm geak --steps 4 --full --output-dir "$R/dit4dit-libero-geak-4step-instinct"

# DiT4DiT LIBERO, GEAK, 2 steps
dit4dit run bash scripts/eval_dit4dit_libero.sh --sim mujoco --arm geak --steps 2 --full --output-dir "$R/dit4dit-libero-geak-2step-mujoco"
# dit4dit run bash scripts/eval_dit4dit_libero.sh --sim instinct --arm geak --steps 2 --full --output-dir "$R/dit4dit-libero-geak-2step-instinct"

# DiT4DiT RoboCasa-GR1 full-24, baseline, 4 steps
dit4dit run bash scripts/eval_dit4dit_robocasa.sh --sim mujoco --arm baseline --steps 4 --full --output-dir "$R/dit4dit-robocasa-baseline-4step-mujoco"
# dit4dit run bash scripts/eval_dit4dit_robocasa.sh --sim instinct --arm baseline --steps 4 --full --output-dir "$R/dit4dit-robocasa-baseline-4step-instinct"

# DiT4DiT RoboCasa-GR1 full-24, GEAK, 4 steps
dit4dit run bash scripts/eval_dit4dit_robocasa.sh --sim mujoco --arm geak --steps 4 --full --output-dir "$R/dit4dit-robocasa-geak-4step-mujoco"
# dit4dit run bash scripts/eval_dit4dit_robocasa.sh --sim instinct --arm geak --steps 4 --full --output-dir "$R/dit4dit-robocasa-geak-4step-instinct"

# DiT4DiT RoboCasa-GR1 full-24, GEAK, 2 steps
dit4dit run bash scripts/eval_dit4dit_robocasa.sh --sim mujoco --arm geak --steps 2 --full --output-dir "$R/dit4dit-robocasa-geak-2step-mujoco"
# dit4dit run bash scripts/eval_dit4dit_robocasa.sh --sim instinct --arm geak --steps 2 --full --output-dir "$R/dit4dit-robocasa-geak-2step-instinct"

# FastWAM latency: baseline (eager), naive torch.compile and GEAK cells, interleaved on one GPU
fastwam latency python3 scripts/compare_artifacts.py --arms eager,compile,geak,geak@2 --rounds 5 --json-out "$R/latency-fastwam.json"

# DiT4DiT LIBERO latency: baseline@4, naive torch.compile@4, GEAK@4 and GEAK@2, interleaved on one GPU
latency python3 scripts/dit4dit_latency.py --bench libero --arms baseline@4,compile@4,geak@4,geak@2 --rounds 5 --json-out "$R/latency-dit4dit-libero.json"

# DiT4DiT RoboCasa latency: baseline@4, naive torch.compile@4, GEAK@4 and GEAK@2, interleaved on one GPU
latency python3 scripts/dit4dit_latency.py --bench robocasa --arms baseline@4,compile@4,geak@4,geak@2 --rounds 5 --json-out "$R/latency-dit4dit-robocasa.json"
