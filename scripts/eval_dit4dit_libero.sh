#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Deliberately NOT defaulted here: the default depends on --arm, which is
# parsed below. An explicit DIT4DIT_ROOT still wins, with a warning.
# shellcheck source=dit4dit_diffusers_overlay.sh
source "${ROOT}/scripts/dit4dit_diffusers_overlay.sh"
# shellcheck source=instinct_sim_env.sh
source "${ROOT}/scripts/instinct_sim_env.sh"
# shellcheck source=eval_flags.sh
source "${ROOT}/scripts/eval_flags.sh"

usage() {
  cat <<'EOF'
Run a DiT4DiT LIBERO evaluation (upstream baseline or GEAK arm).

Usage:
  bash scripts/eval_dit4dit_libero.sh [--smoke | --full] [options]

Options:
  --arm NAME        baseline (default) or geak. (--arm cae is the RoboCasa
                    Compact Action Expert policy; see
                    scripts/eval_dit4dit_robocasa.sh.)
  --steps N         Action-head denoise steps: 4 (paper default) or 2.
                    Plumbed as the num_inference_timesteps ATTRIBUTE, not the
                    dead num_inference_steps kwarg; see the note below.
  --smoke           Run 1 trial per task.
  --full            Run 50 trials per task (default).
  --trials N        Override trials per task.
  --gpus ID         GPU of the policy server. This launcher runs one server,
                    so it takes exactly one id (default 0).
  --port PORT       Policy-server port (default 5694).
  --output-dir DIR  Run directory for logs and videos (default:
                    EVAL_OUTPUT_ROOT, else
                    eval/outputs/dit4dit-libero-<arm>-<N>step-<sim>-<UTC>).
  --clients-per-gpu K
                    LIBERO clients run at once against the policy server,
                    one task each (default 1: one client runs the whole suite,
                    task by task). Results are per task. Needs a tree whose
                    eval_libero.py takes a task range (geak); on baseline it
                    falls back to 1.
  --suites CSV      Suites to evaluate (default: all four); checked before
                    anything starts.
  --sim NAME        mujoco (default): LIBERO on CPU MuJoCo.
                    instinct: LIBERO's physics and camera images GPU-native on AMD Instinct™ GPUs
                    (simulate/, INSTINCT_SIM_BUILD from `simulate/build.sh
                    libero`); the client is unchanged, its OffScreenRenderEnv is
                    swapped at import. See simulate/README.md.
  --record-video    Accepted for a common command line; a no-op here: the
                    unchanged upstream client always writes one video per
                    episode under <output-dir>/videos/<suite>[_task<T>]/.
  --dry-run         Print the resolved arm, tree, steps, trials,
                    suites, GPU, port, clients and output directory, then
                    exit. Needs no checkpoint or Python environment.
  -h, --help        Show this help.

The run ends by writing <output-dir>/summary.json (pooled and per-suite
successes and episodes) and printing one "eval complete: ..." line.

Deprecated spellings (warn; removed in the next release): --gpu ID (use
--gpus), --parallel-tasks N (use --clients-per-gpu), GPU_ID (use EVAL_GPUS),
MODEL_PYTHON (use DIT4DIT_MODEL_PYTHON), LIBERO_PYTHON (use
DIT4DIT_LIBERO_PYTHON).

Source trees (side-by-side, one per arm):
  baseline   src/dit4dit-baseline    Unmodified upstream DiT4DiT (source subset).
  geak       src/dit4dit-geak        Upstream plus earlier inference work, the
                                     GEAK-optimized Cosmos25.py, ActionDiT.py and
                                     DiT4DiT.py, a LIBERO client that takes a task
                                     range, and two diffusers patches (see
                                     below). src/README.md lists every difference.

WHY AN ARM IS A WHOLE TREE
  Running the GEAK tree with every gate off is NOT the baseline. Several of
  its optimizations read no environment variable at all and are always on, so
  a gates-off GEAK run is a partly-optimized tree. Each arm therefore selects a
  different TREE, never the same tree with flags flipped.

GEAK_* GATES ARE PINNED OFF ON EVERY ARM
  GEAK_ACTION_COMPILE, GEAK_COSMOS_COMPILE, GEAK_VAE_T1 and GEAK_ACTION_BF16
  are exported as 0 for every arm. That is the configuration the GEAK arm was
  validated in; with the compile gates on, inductor regenerates a user-defined
  Triton kernel into a module that lacks the source's `_geak_tl` alias and the
  server fails with a NameError. The gates are not overridable here.

DIFFUSERS OVERLAY
  Part of the GEAK arm lives in diffusers, not in the tree:
  src/dit4dit-geak/patches/ holds patches to autoencoder_kl_wan.py and
  transformer_cosmos.py against diffusers 3996788 (0.37.0.dev0). For every
  arm the script copies a diffusers package into the output directory, checks
  that its version is 0.37.0.dev0 and that both target files are byte-identical
  to that upstream commit, applies the patches only on the geak arm, and puts
  the copy first on the server's PYTHONPATH. The model environment's own
  diffusers is never modified, so one environment serves every arm and a
  patched install cannot leak into the baseline.

HOW --steps IS PLUMBED
  `num_inference_steps=` is a DEAD kwarg here. DiT4DiT.predict_action takes
  **kwargs and, on the legacy branch the released LIBERO checkpoint uses
  (`training: joint` -> policy_mode false), calls the action head with no step
  argument -- the kwarg is dropped. The step count is the
  `num_inference_timesteps` ATTRIBUTE, read by ActionDiT.predict_action in
  every tree and used as the `for t in range(num_steps)` bound.

  So the server is started through scripts/dit4dit_serve_policy.py, which
  imports scripts/dit4dit_step_override.py. That patches
  baseframework.from_pretrained to write the attribute after construction and
  ASSERT THE READ-BACK, then logs the value it actually installed. No tree is
  modified, and no new server flag is added -- upstream's
  argparser is handed off to unchanged.

  DIT4DIT_INFER_STEPS is exported on every run, including --steps 4, so the
  server log always carries a verified step count. The output directory name
  interpolates the step count, so a silent override failure would mislabel the
  artifact; the read-back assertion is what makes that impossible.

Required environment:
  DIT4DIT_LIBERO_CKPT    DiT4DiT LIBERO checkpoint: mondo-robotics/dit4dit-model
                         dit4dit_libero/final_model/pytorch_model.pt. The
                         config.yaml above final_model/ must point
                         framework.cosmos25.base_model at a local copy of the
                         gated nvidia/Cosmos-Predict2.5-2B (accept its licence
                         on Hugging Face and log in to download it;
                         docs/cae-dit4dit-robocasa.md#cosmos-predict25-2b).
  DIT4DIT_MODEL_PYTHON   Python in the DiT4DiT model environment (a path, or a
                         name on PATH; default: DIT4DIT_PYTHON)
  DIT4DIT_LIBERO_PYTHON  Python in the LIBERO simulator environment (default:
                         DIT4DIT_PYTHON)
  LIBERO_ROOT            LIBERO checkout

Optional environment:
  DIT4DIT_PYTHON         Default for both Pythons above
  STEPS, TRIALS          Defaults for --steps, --trials
  EVAL_GPUS              Default for --gpus
  EVAL_OUTPUT_ROOT       Default for --output-dir
  PORT                   Default for --port
  LIBERO_CONFIG_PATH     LIBERO path configuration (usually ~/.libero)
  DIT4DIT_ROOT           Overrides the tree selected by --arm, with a warning;
                         the default output directory then ends in
                         -custom-tree
  DIT4DIT_DIFFUSERS_SRC  diffusers package directory to copy into the overlay
                         (default: the model environment's installed diffusers;
                         required when that environment ships no diffusers)
  DIT4DIT_PYOVERLAY      Third-party wheel overlay added to PYTHONPATH
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
                         Set to 1 for the LIBERO clients unless exported:
                         LIBERO's init-state files need a full torch.load on
                         PyTorch >= 2.6
EOF
}

arm=baseline
steps="${STEPS:-4}"
checkpoint="${DIT4DIT_LIBERO_CKPT:-}"
libero_root="${LIBERO_ROOT:-}"
trials="${TRIALS:-50}"
if [[ -n "${EVAL_GPUS:-}" ]]; then
  gpus="$EVAL_GPUS"
elif [[ -n "${GPU_ID:-}" ]]; then
  eval_deprecated GPU_ID EVAL_GPUS
  gpus="$GPU_ID"
else
  gpus=0
fi
port="${PORT:-5694}"
output_dir="${EVAL_OUTPUT_ROOT:-}"
sim=mujoco
clients_per_gpu=1
suites_csv=libero_spatial,libero_object,libero_goal,libero_10
dry_run=0

while (($#)); do
  case "$1" in
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
    --arm)
      [[ $# -ge 2 ]] || { echo "--arm requires a value" >&2; exit 2; }
      arm="$2"
      shift 2
      ;;
    --record-video)
      # Always on: the upstream client writes every episode's video.
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --mode|--num-envs|--pad-batch)
      eval_unsupported "$1" "the batched N-world mode is FastWAM's (eval/native/run_fastwam_libero.sh)"
      ;;
    --steps)
      [[ $# -ge 2 ]] || { echo "--steps requires a value" >&2; exit 2; }
      steps="$2"
      shift 2
      ;;
    --gpus|--gpu)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
      [[ "$1" == --gpu ]] && eval_deprecated --gpu --gpus
      gpus="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || { echo "--port requires a value" >&2; exit 2; }
      port="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
      output_dir="$2"
      shift 2
      ;;
    --clients-per-gpu|--parallel-tasks)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
      [[ "$1" == --parallel-tasks ]] && eval_deprecated --parallel-tasks --clients-per-gpu
      clients_per_gpu="$2"
      shift 2
      ;;
    --suites)
      [[ $# -ge 2 ]] || { echo "--suites requires a value" >&2; exit 2; }
      suites_csv="$2"
      shift 2
      ;;
    --sim)
      [[ $# -ge 2 ]] || { echo "--sim requires a value" >&2; exit 2; }
      sim="$2"
      shift 2
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
eval_check_port --port "$port"
if [[ "$arm" == cae ]]; then
  eval_fail "--arm cae is the RoboCasa Compact Action Expert policy; DiT4DiT LIBERO has baseline and geak" \
    "(CAE-DiT4DiT runs with scripts/eval_dit4dit_robocasa.sh --arm cae)."
fi
eval_check_arm "$arm" baseline geak
eval_check_sim "$sim"
eval_check_gpus "$gpus" 1
gpu="$gpus"
eval_check_posint --clients-per-gpu "$clients_per_gpu"
eval_check_posint --steps "$steps"
# `num_inference_steps=` is a DEAD kwarg in these trees -- it drives the Cosmos
# video scheduler, not the action head, whose step count comes from the
# `num_inference_timesteps` ATTRIBUTE read by ActionDiT.predict_action.
# It is plumbed by writing that attribute at server start, via
# scripts/dit4dit_serve_policy.py -> scripts/dit4dit_step_override.py, which
# asserts the read-back. 10 stays refused: it is a FastWAM convention that was
# never validated for DiT4DiT, not a missing wire.
eval_check_dit4dit_steps "$steps"
eval_check_suites "$suites_csv"
IFS=, read -r -a suites <<< "$suites_csv"

# Tree resolution: an arm is a whole tree. See the --help note on why the
# baseline arm cannot be the GEAK tree with its gates turned off.
arm_tree="${ROOT}/src/dit4dit-${arm}"
tree_tag=""
if [[ -n "${DIT4DIT_ROOT:-}" ]]; then
  # Still honoured, but it serves another tree under this arm's label (and the
  # diffusers patches still follow --arm), so say so and mark the directory.
  eval_warn_override DIT4DIT_ROOT "$arm_tree" "--arm ${arm}"
  [[ "$(realpath -m "$DIT4DIT_ROOT")" != "$(realpath -m "$arm_tree")" ]] && tree_tag=custom-tree
else
  DIT4DIT_ROOT="$arm_tree"
fi
# One client per task needs a tree whose eval_libero.py takes a task range
# (geak); otherwise one client runs each suite. Decided before anything starts.
if (( clients_per_gpu > 1 )) && ! grep -q "start_task" "${DIT4DIT_ROOT}/examples/LIBERO/eval_files/eval_libero.py" 2>/dev/null; then
  echo "note: this tree's eval_libero.py has no --args.start-task/--args.end-task; running one client per suite" >&2
  clients_per_gpu=1
fi
output_dir="${output_dir:-$(eval_default_output_dir dit4dit libero "$arm" "$steps" "$sim" "$tree_tag")}"

if ((dry_run)); then
  eval_dry_run "model=dit4dit" "benchmark=libero" "arm=${arm}" \
    "tree=$(realpath -m "$DIT4DIT_ROOT")" "steps=${steps}" "trials=${trials}" "suites=${suites_csv}" \
    "gpus=${gpu}" "port=${port}" "clients_per_gpu=${clients_per_gpu}" "sim=${sim}" \
    "mujoco_gl=$([[ "$sim" == instinct ]] && echo "osmesa (--sim instinct helper contexts)" \
      || eval_mujoco_gl_dry_run)" \
    "record_video=1 (always)" "output=$(realpath -m "$output_dir")"
fi

# Pythons: DIT4DIT_MODEL_PYTHON / DIT4DIT_LIBERO_PYTHON, else the deprecated
# MODEL_PYTHON / LIBERO_PYTHON (so an old setup keeps its interpreter), else the
# DIT4DIT_PYTHON umbrella (as on RoboCasa).
model_python="${DIT4DIT_MODEL_PYTHON:-}"
libero_python="${DIT4DIT_LIBERO_PYTHON:-}"
if [[ -z "$model_python" && -n "${MODEL_PYTHON:-}" ]]; then
  eval_deprecated MODEL_PYTHON DIT4DIT_MODEL_PYTHON
  model_python="$MODEL_PYTHON"
fi
if [[ -z "$libero_python" && -n "${LIBERO_PYTHON:-}" ]]; then
  eval_deprecated LIBERO_PYTHON DIT4DIT_LIBERO_PYTHON
  libero_python="$LIBERO_PYTHON"
fi
model_python="${model_python:-${DIT4DIT_PYTHON:-}}"
libero_python="${libero_python:-${DIT4DIT_PYTHON:-}}"

eval_require DIT4DIT_LIBERO_CKPT "the DiT4DiT LIBERO checkpoint file"
eval_require LIBERO_ROOT "the LIBERO checkout"
[[ -f "$checkpoint" ]] || {
  echo "DIT4DIT_LIBERO_CKPT must name a checkpoint file: $checkpoint" >&2
  exit 2
}
model_python="$(eval_resolve_python DIT4DIT_MODEL_PYTHON "$model_python")" || exit 2
libero_python="$(eval_resolve_python DIT4DIT_LIBERO_PYTHON "$libero_python")" || exit 2
[[ -d "$libero_root" ]] || {
  echo "LIBERO_ROOT must name the LIBERO checkout: $libero_root" >&2
  exit 2
}
[[ -d "$DIT4DIT_ROOT" ]] || {
  echo "DIT4DIT_ROOT must name the DiT4DiT checkout: $DIT4DIT_ROOT" >&2
  exit 2
}
if [[ "$sim" == instinct ]]; then instinct_sim_check libero "$libero_python" || exit 2; fi

checkpoint="$(realpath "$checkpoint")"
libero_root="$(realpath "$libero_root")"
DIT4DIT_ROOT="$(realpath "$DIT4DIT_ROOT")"
# CPU MuJoCo renders with GPU OpenGL (EGL) when the client interpreter can
# create an EGL context in the client's environment (its GPU and PYTHONPATH),
# else with OSMesa; an exported MUJOCO_GL is kept. Under --sim instinct the GPU
# simulator renders the cameras and robosuite's helper MuJoCo contexts take
# OSMesa, as in scripts/eval_dit4dit_robocasa.sh: left unset, robosuite would
# pick EGL, which fails on GPUs without a graphics engine (EGL_BAD_ALLOC).
if [[ "$sim" == instinct ]]; then
  eval_mujoco_gl_instinct
fi
if [[ "$sim" == mujoco ]]; then
  eval_mujoco_gl env CUDA_VISIBLE_DEVICES="$gpu" HIP_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="${DIT4DIT_ROOT}:${libero_root}${DIT4DIT_PYOVERLAY:+:${DIT4DIT_PYOVERLAY}}:${PYTHONPATH:-}" \
    "$libero_python" || exit 2
fi
mkdir -p "$output_dir/logs" "$output_dir/videos"
output_dir="$(cd "$output_dir" && pwd)"
server_log="${output_dir}/logs/server_gpu${gpu}_port${port}.log"

"$model_python" - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise SystemExit(f"policy-server port {port} is unavailable: {exc}")
PY

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# Diffusers overlay: a per-run copy of diffusers 3996788, patched on the geak
# arm only. Sets diffusers_overlay.
build_diffusers_overlay "$model_python" "$arm" "${output_dir}/diffusers-overlay" || exit $?

# GEAK levers, pinned off on EVERY arm: this is the configuration the GEAK arm
# was validated in (see --help). Each is exported explicitly because an unset
# gate falls back to the source default, which is on for some of them.
gate_action_compile=0; gate_cosmos_compile=0; gate_vae_t1=0; gate_action_bf16=0

# No current tree's server_policy.py accepts --seed, and the validated runs
# passed none. Passing an unsupported flag is an argparse hard failure, so a
# tree that does accept it is detected rather than assumed.
declare -a seed_args=()
if grep -q -- '"--seed"' "${DIT4DIT_ROOT}/deployment/model_server/server_policy.py" 2>/dev/null; then
  seed_args=(--seed 7)
fi

echo "Starting DiT4DiT policy server on GPU ${gpu}, port ${port} (arm=${arm})"
(
  cd "$DIT4DIT_ROOT"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export HIP_VISIBLE_DEVICES="$gpu"
  export PYTHONPATH="${DIT4DIT_ROOT}:${diffusers_overlay}${DIT4DIT_PYOVERLAY:+:${DIT4DIT_PYOVERLAY}}:${PYTHONPATH:-}"
  export GEAK_ACTION_COMPILE="$gate_action_compile"
  export GEAK_COSMOS_COMPILE="$gate_cosmos_compile"
  export GEAK_VAE_T1="$gate_vae_t1"
  export GEAK_ACTION_BF16="$gate_action_bf16"
  # Always exported, including for the default 4, so the server log always
  # carries a step count that was asserted rather than assumed.
  export DIT4DIT_INFER_STEPS="$steps"
  exec "$model_python" "${ROOT}/scripts/dit4dit_serve_policy.py" \
    --ckpt_path "$checkpoint" \
    --port "$port" \
    "${seed_args[@]+"${seed_args[@]}"}" \
    --idle_timeout -1
) >"$server_log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 300); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "policy server exited before becoming ready; see $server_log" >&2
    wait "$server_pid" || true
    exit 1
  fi
  if "$model_python" - "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
    pass
PY
  then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" -eq 1 ]] || {
  echo "policy server did not become ready; see $server_log" >&2
  exit 1
}

# One eval_libero.py client. $1 = suite, $2 = task (empty: the whole suite).
run_libero_client() {
  local suite="$1" task="${2:-}"
  local -a task_args=()
  [[ -n "$task" ]] && task_args=(--args.start-task "$task" --args.end-task "$task")
  (
    cd "$DIT4DIT_ROOT"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export HIP_VISIBLE_DEVICES="$gpu"
    export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
    # LIBERO loads its own init-state files with a bare torch.load; PyTorch >= 2.6 defaults to
    # weights_only=True, which rejects them (the FastWAM evaluator passes weights_only=False instead).
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
    export PYTHONPATH="${DIT4DIT_ROOT}:${libero_root}${DIT4DIT_PYOVERLAY:+:${DIT4DIT_PYOVERLAY}}:${PYTHONPATH:-}"
    if [[ "$sim" == instinct ]]; then instinct_sim_client_env libero "$libero_python"; fi
    "$libero_python" examples/LIBERO/eval_files/eval_libero.py \
      --args.pretrained-path "$checkpoint" \
      --args.host 127.0.0.1 \
      --args.port "$port" \
      --args.task-suite-name "$suite" \
      --args.num-trials-per-task "$trials" \
      --args.seed 7 \
      "${task_args[@]+"${task_args[@]}"}" \
      --args.video-out-path "${output_dir}/videos/${suite}${task:+_task${task}}"
  )
}

# eval_libero.py logs a running "# successes: N (P%)" after every episode and "Total episodes: N" at the end; the
# last of each per log is that client's total.
summarize_logs() {
  local s=0 e=0 f n m
  for f in "$@"; do
    n="$(grep -oE '# successes: [0-9]+' "$f" | tail -n 1 | grep -oE '[0-9]+$' || true)"
    m="$(grep -oE 'Total episodes: [0-9]+' "$f" | tail -n 1 | grep -oE '[0-9]+$' || true)"
    s=$((s + ${n:-0})); e=$((e + ${m:-0}))
  done
  echo "${s}/${e}"
}

# One client or K, a failed client is counted and the remaining suites still
# run; the run then ends with the incomplete message and exit 1.
failed_clients=0
declare -a suite_counts=()
for suite in "${suites[@]}"; do
  echo "Evaluating ${suite} (${trials} trials/task, ${clients_per_gpu} client(s))"
  suite_failed=0
  if (( clients_per_gpu == 1 )); then
    suite_logs=("${output_dir}/logs/eval_${suite}.log")
    see="${suite_logs[0]}"
    # In the background, as the K>1 clients are, so the client keeps errexit
    # and its failure is counted rather than ending the run.
    run_libero_client "$suite" 2>&1 | tee "${suite_logs[0]}" &
    wait "$!" || suite_failed=1
  else
    client_pids=()
    for task in 0 1 2 3 4 5 6 7 8 9; do
      while (($(jobs -rp | grep -cvx "$server_pid") >= clients_per_gpu)); do
        wait -n || true
      done
      run_libero_client "$suite" "$task" >"${output_dir}/logs/eval_${suite}_task${task}.log" 2>&1 &
      client_pids+=("$!")
    done
    for pid in "${client_pids[@]}"; do
      wait "$pid" || suite_failed=$((suite_failed + 1))
    done
    suite_logs=("${output_dir}"/logs/eval_"${suite}"_task*.log)
    see="${output_dir}/logs/eval_${suite}_task*.log"
  fi
  counts="$(summarize_logs "${suite_logs[@]}")"
  suite_counts+=("${suite}=${counts}")
  echo "${suite}: ${counts} successful episodes"
  if (( suite_failed > 0 )); then
    echo "${suite}: ${suite_failed} client(s) failed; see ${see}" >&2
    failed_clients=$((failed_clients + suite_failed))
  fi
done

rc=0
if (( failed_clients > 0 )); then
  echo "DiT4DiT LIBERO evaluation incomplete: ${failed_clients} client(s) failed ($output_dir)" >&2
  rc=1
fi
eval_summary --model dit4dit --bench libero --arm "$arm" --steps "$steps" --sim "$sim" \
  --trials "$trials" --output-dir "$output_dir" --rc "$rc" --counts "${suite_counts[@]}" || true
exit "$rc"
