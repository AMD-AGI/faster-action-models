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
# Environment: what each launcher's --help lists (LIBERO_ROOT, EVAL_IMAGE, EVAL_HARNESS, ROBOSUITE_OVERLAY,
# INSTINCT_SIM_BUILD for --sim instinct, ...), with the checkpoints named per model below:
#   FASTWAM_CKPT, FASTWAM_DATASET_STATS          FastWAM checkpoint and its normalization statistics
#   DIT4DIT_LIBERO_CKPT                           DiT4DiT LIBERO checkpoint
#   DIT4DIT_ROBOCASA_CKPT                         the DiT4DiT authors' RoboCasa-GR1 checkpoint
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
R="${REPRO_DIR:-eval/outputs/reproduce_training_free}"
mkdir -p "$R"

fastwam() { MODEL_CKPT="$FASTWAM_CKPT" DATASET_STATS="$FASTWAM_DATASET_STATS" "$@"; }

# FastWAM LIBERO, baseline (upstream FastWAM), 10 steps
fastwam bash scripts/eval_fastwam_libero.sh --sim mujoco --arm baseline --full --output-dir "$R/fastwam-libero-baseline-10step-mujoco"
# fastwam bash eval/native/run_fastwam_libero.sh --sim instinct --arm baseline --full --output-dir "$R/fastwam-libero-baseline-10step-instinct"

# FastWAM LIBERO, GEAK, 10 steps
fastwam bash scripts/eval_fastwam_libero.sh --sim mujoco --arm geak --steps 10 --full --output-dir "$R/fastwam-libero-geak-10step-mujoco"
# fastwam bash eval/native/run_fastwam_libero.sh --sim instinct --arm geak --steps 10 --full --output-dir "$R/fastwam-libero-geak-10step-instinct"

# FastWAM LIBERO, GEAK, 2 steps
fastwam bash scripts/eval_fastwam_libero.sh --sim mujoco --arm geak --steps 2 --full --output-dir "$R/fastwam-libero-geak-2step-mujoco"
# fastwam bash eval/native/run_fastwam_libero.sh --sim instinct --arm geak --steps 2 --full --output-dir "$R/fastwam-libero-geak-2step-instinct"

# DiT4DiT LIBERO, baseline, 4 steps
bash scripts/eval_dit4dit_libero.sh --sim mujoco --arm baseline --steps 4 --full --output-dir "$R/dit4dit-libero-baseline-4step-mujoco"
# bash scripts/eval_dit4dit_libero.sh --sim instinct --arm baseline --steps 4 --full --output-dir "$R/dit4dit-libero-baseline-4step-instinct"

# DiT4DiT LIBERO, GEAK, 4 steps
bash scripts/eval_dit4dit_libero.sh --sim mujoco --arm geak --steps 4 --full --output-dir "$R/dit4dit-libero-geak-4step-mujoco"
# bash scripts/eval_dit4dit_libero.sh --sim instinct --arm geak --steps 4 --full --output-dir "$R/dit4dit-libero-geak-4step-instinct"

# DiT4DiT LIBERO, GEAK, 2 steps
bash scripts/eval_dit4dit_libero.sh --sim mujoco --arm geak --steps 2 --full --output-dir "$R/dit4dit-libero-geak-2step-mujoco"
# bash scripts/eval_dit4dit_libero.sh --sim instinct --arm geak --steps 2 --full --output-dir "$R/dit4dit-libero-geak-2step-instinct"

# DiT4DiT RoboCasa-GR1 full-24, baseline, 4 steps
bash scripts/eval_dit4dit_robocasa.sh --sim mujoco --arm baseline --steps 4 --full --output-dir "$R/dit4dit-robocasa-baseline-4step-mujoco"
# bash scripts/eval_dit4dit_robocasa.sh --sim instinct --arm baseline --steps 4 --full --output-dir "$R/dit4dit-robocasa-baseline-4step-instinct"

# DiT4DiT RoboCasa-GR1 full-24, GEAK, 4 steps
bash scripts/eval_dit4dit_robocasa.sh --sim mujoco --arm geak --steps 4 --full --output-dir "$R/dit4dit-robocasa-geak-4step-mujoco"
# bash scripts/eval_dit4dit_robocasa.sh --sim instinct --arm geak --steps 4 --full --output-dir "$R/dit4dit-robocasa-geak-4step-instinct"

# DiT4DiT RoboCasa-GR1 full-24, GEAK, 2 steps
bash scripts/eval_dit4dit_robocasa.sh --sim mujoco --arm geak --steps 2 --full --output-dir "$R/dit4dit-robocasa-geak-2step-mujoco"
# bash scripts/eval_dit4dit_robocasa.sh --sim instinct --arm geak --steps 2 --full --output-dir "$R/dit4dit-robocasa-geak-2step-instinct"

# FastWAM latency: baseline (eager), naive torch.compile and GEAK cells, interleaved on one GPU
fastwam python3 scripts/compare_artifacts.py --arms eager,compile,geak,geak@2 --rounds 5 --json-out "$R/latency-fastwam.json"

# DiT4DiT LIBERO latency: baseline@4, naive torch.compile@4, GEAK@4 and GEAK@2, interleaved on one GPU
python3 scripts/dit4dit_latency.py --bench libero --arms baseline@4,compile@4,geak@4,geak@2 --rounds 5 --json-out "$R/latency-dit4dit-libero.json"

# DiT4DiT RoboCasa latency: baseline@4, naive torch.compile@4, GEAK@4 and GEAK@2, interleaved on one GPU
python3 scripts/dit4dit_latency.py --bench robocasa --arms baseline@4,compile@4,geak@4,geak@2 --rounds 5 --json-out "$R/latency-dit4dit-robocasa.json"
