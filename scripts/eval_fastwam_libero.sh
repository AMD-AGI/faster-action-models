#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=eval_flags.sh
source "${ROOT}/scripts/eval_flags.sh"

usage() {
  cat <<'EOF'
Run a FastWAM LIBERO evaluation (baseline, GEAK or CAE-FastWAM arm).

Usage:
  bash scripts/eval_fastwam_libero.sh [--smoke | --full] [options]

Options:
  --arm NAME        baseline (default), geak or cae:
                      baseline  src/fastwam-baseline, the original FastWAM
                      geak      src/fastwam-geak, its GEAK-optimized tree
                      cae       CAE-FastWAM (Compact Action Expert: the
                                32-hidden-dimension Action DiT with the K8 x d16
                                action VAE, GEAK-optimized), src/cae-fastwam,
                                one policy server per suite
  --steps N         Denoise steps (default 10). Advertised: baseline 10,
                    geak 10 and 2, cae 10 and 4.
  --smoke           Run 1 trial per task.
  --full            Run 50 trials per task (default).
  --trials N        Override trials per task.
  --suites CSV      Evaluate selected LIBERO suites (default: all four).
  --sim NAME        mujoco (default). The harness evaluator runs LIBERO on CPU
                    MuJoCo only; for the GPU simulator use
                    eval/native/run_fastwam_libero.sh --sim instinct.
  --gpus ID         GPU of the policy server. This launcher runs one server,
                    so it takes exactly one id (default 0).
  --port PORT       Policy-server port (default: 8300 + GPU id for baseline
                    and geak, 8400 + GPU id for cae).
  --clients-per-gpu K
                    Harness clients (task shards) run at once against the
                    policy server (default 10).
  --output-dir DIR  Run directory (default: EVAL_OUTPUT_ROOT, else
                    eval/outputs/fastwam-libero-<arm>-<N>step-<sim>-<UTC>).
                    A rerun into the same directory resumes it only for the
                    same cell (tree, checkpoint, steps, trials) and the same
                    --clients-per-gpu (result files are per client shard, so
                    another count would pool stale shards); a directory
                    holding results of a different run is refused.
  --allow-baseline-steps
                    Permit --steps other than 10 on the baseline arm. Step
                    reduction is an optimized-side claim, so this is opt-in.
  --record-video    Write one agent-view video per episode under
                    <output-dir>/<cell>/videos/<suite>/. Needs a harness whose
                    LIBERO benchmark accepts save_video_dir; refused otherwise.
  --dry-run         Print the resolved arm, tree, configs, steps,
                    trials, suites, GPU, port and output directory, then exit.
                    Needs no checkpoint or container.
  -h, --help        Show this help.

The run ends by writing <output-dir>/summary.json (pooled and per-suite
successes and episodes) and printing one "eval complete: ..." line.

Deprecated spellings (warn; removed in the next release): --shards N (use
--clients-per-gpu).

Source trees (side-by-side, one per validated configuration):
  baseline  src/fastwam-baseline
  geak      src/fastwam-geak
  cae       src/cae-fastwam

Required environment:
  all arms:        MODEL_CKPT (the arm's checkpoint), EVAL_HARNESS,
                   LIBERO_ROOT, EVAL_IMAGE
  baseline, geak:  DATASET_STATS            normalization statistics
  cae:             TASK_CONFIG              CAE-FastWAM task config (base name)
                   DATASET_STATS_TEMPLATE   must contain the token {suite}
                   ACTION_VAE_CKPT          action-VAE checkpoint CAE-FastWAM
                                            was trained with

Optional environment:
  MODEL_CONFIGS     Default: <repo>/FastWAM/configs_baseline for baseline and
                    geak (the config tree those source trees were evaluated
                    with), and <repo>/FastWAM/configs for cae (holds the
                    CAE-FastWAM task config and its per-suite variants)
  BASELINE_SRC      Overrides the resolved baseline tree
  OPTIMIZED_SRC     Overrides the resolved geak tree
  CAE_SRC           Overrides the resolved cae tree
  STEPS, TRIALS     Defaults for --steps, --trials
  EVAL_GPUS         Default for --gpus
  EVAL_OUTPUT_ROOT  Default for --output-dir
  PORT              Default for --port

MODEL_CONFIGS and the tree override of the selected arm are honoured, with a
warning when they differ from what --arm selects; the other arms' tree
overrides are not used (on --arm cae, a set BASELINE_SRC or OPTIMIZED_SRC
is ignored with a warning). The engines' own knobs (SEED, WAIT, SHARDS, SUITES_CSV,
EVAL_PORT, EVAL_ARM_LABEL, and TASK_CONFIG on baseline and geak) are ignored
with a warning: the seed and wait come from eval/PROTOCOL.md and the rest from
the flags above.
EOF
}

arm=baseline
steps="${STEPS:-}"
trials="${TRIALS:-50}"
gpus="${EVAL_GPUS:-0}"
port="${PORT:-}"
sim=mujoco
output_dir="${EVAL_OUTPUT_ROOT:-}"
allow_baseline_steps=0
record_video=0
dry_run=0
clients_per_gpu=10
suites_csv=""
suites_given=0
while (($#)); do
  case "$1" in
    --arm)
      [[ $# -ge 2 ]] || { echo "--arm requires a value" >&2; exit 2; }
      arm="$2"
      shift 2
      ;;
    --steps)
      [[ $# -ge 2 ]] || { echo "--steps requires a value" >&2; exit 2; }
      steps="$2"
      shift 2
      ;;
    --gpus)
      [[ $# -ge 2 ]] || { echo "--gpus requires a value" >&2; exit 2; }
      gpus="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || { echo "--port requires a value" >&2; exit 2; }
      port="$2"
      shift 2
      ;;
    --allow-baseline-steps)
      allow_baseline_steps=1
      shift
      ;;
    --record-video)
      record_video=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --smoke)
      trials=1
      shift
      ;;
    --full)
      trials=50
      shift
      ;;
    --trials)
      [[ $# -ge 2 ]] || { echo "--trials requires a value" >&2; exit 2; }
      trials="$2"
      shift 2
      ;;
    --suites)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
      suites_csv="$2"
      suites_given=1
      shift 2
      ;;
    --clients-per-gpu|--shards)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
      [[ "$1" == --shards ]] && eval_deprecated --shards --clients-per-gpu
      eval_check_posint "$1" "$2"
      clients_per_gpu="$2"
      shift 2
      ;;
    --sim)
      [[ $# -ge 2 ]] || { echo "--sim requires a value" >&2; exit 2; }
      sim="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
      output_dir="$2"
      shift 2
      ;;
    --mode|--num-envs|--pad-batch)
      eval_unsupported "$1" "the batched N-world mode is eval/native/run_fastwam_libero.sh --sim instinct --mode batched"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

eval_check_posint --trials "$trials"
eval_check_arm "$arm" baseline geak cae
eval_check_sim "$sim"
if [[ "$sim" == instinct ]]; then
  eval_fail "the harness evaluator runs CPU MuJoCo; use eval/native/run_fastwam_libero.sh --sim instinct"
fi
# eval/run_eval.sh runs one config per GPU and this launcher selects one
# config; eval/run_eval_cae_fastwam.sh is sequential. Either way one server runs.
eval_check_gpus "$gpus" 1
if [[ -n "$port" ]]; then eval_check_port --port "$port"; fi
if ((suites_given)); then eval_check_suites "$suites_csv"; fi
steps="${steps:-10}"
eval_check_fastwam_steps "$arm" "$steps" "$allow_baseline_steps"

# Tree resolution. This table IS the side-by-side layout: an arm is a whole
# tree, never an env flag over a shared tree, because the GEAK variants were
# generated separately from their baselines and validated as whole trees.
# Each arm's tree variable: the override the launcher honours for that arm.
case "$arm" in
  baseline) arm_tree=src/fastwam-baseline; tree_var=BASELINE_SRC ;;
  geak)     arm_tree=src/fastwam-geak;     tree_var=OPTIMIZED_SRC ;;
  cae)      arm_tree=src/cae-fastwam;      tree_var=CAE_SRC ;;
esac
base_tree=src/fastwam-baseline

# The FastWAM baseline and GEAK trees predate the upstream config schema in
# FastWAM/configs (their create_fastwam rejects keys such as
# compile_training_denoise), so each arm defaults to the config tree its source
# tree was evaluated with.
configs_dir="${ROOT}/FastWAM/configs_baseline"
[[ "$arm" == cae ]] && configs_dir="${ROOT}/FastWAM/configs"
# An exported override (a guide's GEAK-task preamble, say) still wins, but it
# changes what is served under this cell's label, so say so.
cell="--arm ${arm}"
eval_warn_override MODEL_CONFIGS "$configs_dir" "$cell"
eval_warn_override "$tree_var" "${ROOT}/${arm_tree}" "$cell"
served_tree="${!tree_var:-${ROOT}/${arm_tree}}"
output_dir="${output_dir:-$(eval_default_output_dir fastwam libero "$arm" "$steps" "$sim")}"
# The engines take the run directory as EVAL_OUTPUT_ROOT and need it absolute.
output_dir="$(realpath -m "$output_dir")"
# The engine's cell directory inside the run directory. eval/run_eval.sh names
# the optimized arm after EVAL_ARM_LABEL, so it carries the arm the user passed.
if [[ "$arm" == cae ]]; then
  engine_cell="cae_${steps}step"
  default_port="$((8400 + gpus)) (8400 + GPU id)"
else
  engine_cell="${arm}_${steps}step"
  default_port="$((8300 + gpus)) (8300 + GPU id)"
fi

if ((dry_run)); then
  eval_dry_run "model=fastwam" "benchmark=libero" "arm=${arm}" \
    "tree=$(realpath -m "$served_tree")" "configs=$(realpath -m "${MODEL_CONFIGS:-$configs_dir}")" \
    "steps=${steps}" "trials=${trials}" \
    "suites=${suites_csv:-$(IFS=,; echo "${EVAL_LIBERO_SUITES[*]}")}" \
    "gpus=${gpus}" "clients_per_gpu=${clients_per_gpu}" "port=${port:-$default_port}" \
    "sim=${sim}" "mujoco_gl=$(eval_mujoco_gl_dry_run)" "record_video=${record_video}" \
    "output=${output_dir}"
fi

eval_require MODEL_CKPT "the FastWAM checkpoint"
eval_require EVAL_HARNESS "the vla-evaluation-harness checkout"
eval_require LIBERO_ROOT "the LIBERO checkout"
eval_require EVAL_IMAGE "the evaluation container image"
if [[ "$arm" == cae ]]; then
  eval_require ACTION_VAE_CKPT "the action-VAE checkpoint (--arm cae)"
else
  eval_require DATASET_STATS "its normalization statistics (baseline, geak)"
fi

export MODEL_CONFIGS="${MODEL_CONFIGS:-${configs_dir}}"
if [[ "$arm" == cae ]]; then
  # CAE-FastWAM has no baseline tree; the other arms' overrides do not apply.
  eval_ignore_env "it selects a FastWAM baseline or GEAK tree, not --arm cae" BASELINE_SRC OPTIMIZED_SRC
  export CAE_SRC="$served_tree"
else
  export BASELINE_SRC="${BASELINE_SRC:-${ROOT}/${base_tree}}"
fi
export EVAL_OUTPUT_ROOT="$output_dir"
export EVAL_GPUS="$gpus"
export RECORD_VIDEO="$record_video"
# The engines read these from the environment when run standalone. A launcher
# run takes the seed and wait from eval/PROTOCOL.md (the engines' defaults)
# and everything else from its flags, never from a stray export -- SEED, for
# one, is also a training variable.
eval_ignore_env "the evaluation protocol and this launcher's flags set it" \
  SEED WAIT SHARDS SUITES_CSV EVAL_PORT EVAL_ARM_LABEL
if [[ "$arm" != cae ]]; then
  # The baseline and GEAK servers use their default task config; TASK_CONFIG
  # names CAE-FastWAM's.
  eval_ignore_env "TASK_CONFIG is an --arm cae setting" TASK_CONFIG
fi
if [[ -n "$port" ]]; then export EVAL_PORT="$port"; fi
if [[ "$arm" == geak ]]; then export EVAL_ARM_LABEL="$arm"; fi

# Distinct trees must never share a compile cache. Keyed by arm so a rerun of
# the same cell reuses its own cache and never another tree's.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/fastwam_inductor_${arm}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/fastwam_triton_${arm}}"

# FastWAM GEAK: the K/V slab lever trips an inductor lowering bug on a cold
# compile (a GPU memory fault, and silently wrong tensors at layers that stay in
# bounds), so it is always off here. That alone is not enough: the inductor
# cache key does not see the switch, so a cache that ever held a slab-on build
# serves it back. Hence a fresh cache per run, overriding any caller cache.
if [[ "$arm" == geak ]]; then
  export FASTWAM_KVSLAB=0
  export TORCHINDUCTOR_CACHE_DIR="${EVAL_OUTPUT_ROOT}/.compile_cache/inductor"
  export TRITON_CACHE_DIR="${EVAL_OUTPUT_ROOT}/.compile_cache/triton"
fi

if [[ "$arm" == geak ]]; then
  export OPTIMIZED_SRC="${OPTIMIZED_SRC:-${ROOT}/${arm_tree}}"
fi
declare -a suite_args=()
if [[ -n "$suites_csv" ]]; then suite_args=(--suites "$suites_csv"); fi

# THE ONE ARM-NAME TRANSLATION. scripts/compare_artifacts.py and this launcher
# name the comparison arm `geak`; eval/run_eval.sh validates
# `baseline|optimized`. Translate here and nowhere else -- the engine never
# learns the word geak (EVAL_ARM_LABEL only names its result directories).
run_arm=baseline
[[ "$arm" == geak ]] && run_arm=optimized

rc=0
if [[ "$arm" != cae ]]; then
  # --arm is a FILTER over CONFIGS, not a setter, so both must be set together.
  export CONFIGS="${run_arm}:${steps}"
  bash "${ROOT}/eval/run_eval.sh" \
    --arm "$run_arm" \
    --trials "$trials" \
    --gpus "$gpus" \
    --shards "$clients_per_gpu" \
    "${suite_args[@]+"${suite_args[@]}"}" || rc=$?
else
  eval_require TASK_CONFIG "the CAE-FastWAM task config base name (--arm cae)"
  eval_require DATASET_STATS_TEMPLATE "the per-suite statistics path, with the token {suite} (--arm cae)"
  export ACTION_VAE_CKPT
  # CAE-FastWAM uses its own engine: the Compact Action Expert is
  # suite-conditioned and needs one server per suite. See
  # eval/run_eval_cae_fastwam.sh for why run_eval.sh cannot express this.
  bash "${ROOT}/eval/run_eval_cae_fastwam.sh" \
    --steps "$steps" \
    --trials "$trials" \
    --gpus "$gpus" \
    --shards "$clients_per_gpu" \
    "${suite_args[@]+"${suite_args[@]}"}" || rc=$?
fi
# Exit 2 is a refused invocation (an environment check, or a run directory that
# holds another run): there is nothing of this run to summarize.
((rc == 2)) && exit 2
eval_summary --model fastwam --bench libero --arm "$arm" --steps "$steps" \
  --sim "$sim" --trials "$trials" --output-dir "$output_dir" --rc "$rc" \
  --harness "${output_dir}/${engine_cell}" || true
exit "$rc"
