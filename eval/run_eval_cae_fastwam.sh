#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# CAE-FastWAM closed-loop LIBERO evaluation.
#
# CAE-FastWAM (Compact Action Expert: the 32-hidden-dimension Action DiT with the
# K8 x d16 action VAE, GEAK-optimized) is served from src/cae-fastwam.
#
# WHY THIS IS A SEPARATE ENGINE FROM eval/run_eval.sh
# --------------------------------------------------
# The CAE-FastWAM Compact Action Expert is suite-conditioned in TWO ways, and
# both are baked in for the lifetime of a server process:
#
#   1. per-suite normalization statistics (DATASET_STATS_TEMPLATE), and
#   2. a per-suite soft-prompt embedding in the action-VAE decoder, selected by
#      `model.action_vae_suite_id` inside the per-suite task config.
#
# eval/run_eval.sh writes all four suites into ONE eval.yaml behind ONE server
# and accepts exactly one DATASET_STATS file, so it structurally cannot express
# this. Rather than restructure a script whose literal contents are pinned by
# tests/test_fastwam_geak_tooling.py, CAE-FastWAM runs ONE SERVER PER SUITE,
# sequentially.
# That is the layout already validated by the reference implementation.
#
# eval/run_eval.sh ALSO never passes --task_config, so it silently inherits the
# harness default `libero_uncond_2cam224_1e-4` -- the FastWAM baseline config.
# Running the CAE-FastWAM tree through it is a silent-wrong-config path even
# before the stats
# problem. This engine always passes --task_config explicitly.
#
# No performance numbers are recorded in this repository.
set -uo pipefail
# shellcheck source=../scripts/eval_flags.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/eval_flags.sh"

usage() {
  cat <<'EOF'
CAE-FastWAM (Compact Action Expert) LIBERO evaluation engine. Internal: run
it through scripts/eval_fastwam_libero.sh --arm cae, which maps
--clients-per-gpu onto this engine's --shards.

Usage:
  bash eval/run_eval_cae_fastwam.sh [options]

Options:
  --steps N         Denoise steps (default 10).
  --trials N        Episodes per task (default 50).
  --shards N        Harness clients per server (default 10).
  --suites CSV      Suites to evaluate.
  --gpus CSV        GPU pool; one GPU is used (this engine is sequential).
  -h, --help        Show this help.

The server listens on EVAL_PORT when it is set, else on 8400 + the GPU id.
Results go to EVAL_OUTPUT_ROOT/cae_<N>step.

Required environment (absolute paths):
  MODEL_CKPT             CAE-FastWAM checkpoint
  MODEL_CONFIGS          Config root containing the CAE-FastWAM task
                         config and its per-suite variants (the repo's
                         FastWAM/configs)
  ACTION_VAE_CKPT        Action-VAE checkpoint the model was trained with; the
                         per-suite task configs read it from the environment
  EVAL_HARNESS           vla-evaluation-harness checkout
  LIBERO_ROOT            LIBERO checkout
  EVAL_OUTPUT_ROOT       Output root
  EVAL_IMAGE             Evaluation container image
  CAE_SRC                CAE-FastWAM source tree (the repo's src/cae-fastwam)
  TASK_CONFIG            Task config BASE name; "_<suite>" is appended per suite
  DATASET_STATS_TEMPLATE Path containing the literal token {suite}, e.g.
                         /.../per_suite_stats/dataset_stats_{suite}.json
EOF
}

STEPS=10
TRIALS="${TRIALS:-50}"; SHARDS="${SHARDS:-10}"
SEED="${SEED:-42}"; WAIT="${WAIT:-30}"
IFS=, read -r -a SUITES <<< "${SUITES_CSV:-libero_spatial,libero_object,libero_goal,libero_10}"
IFS=, read -r -a GPUS <<< "${EVAL_GPUS:-0}"

while [ $# -gt 0 ]; do case "$1" in
  --steps)  STEPS="$2"; shift 2 ;;
  --trials) TRIALS="$2"; shift 2 ;;
  --shards) SHARDS="$2"; shift 2 ;;
  --suites) IFS=, read -r -a SUITES <<< "$2"; shift 2 ;;
  --gpus)   IFS=, read -r -a GPUS <<< "$2"; shift 2 ;;
  -h|--help) usage; exit 0 ;;
  *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac; done

[[ "$STEPS"  =~ ^[1-9][0-9]*$ ]] || { echo "--steps must be a positive integer, got: $STEPS" >&2; exit 2; }
[[ "$TRIALS" =~ ^[1-9][0-9]*$ ]] || { echo "--trials must be a positive integer, got: $TRIALS" >&2; exit 2; }
[[ "$SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "--shards must be a positive integer, got: $SHARDS" >&2; exit 2; }
if [ "${#GPUS[@]}" -eq 0 ]; then echo "no GPUs configured; pass --gpus CSV" >&2; exit 2; fi
for g in "${GPUS[@]}"; do
  [[ "$g" =~ ^[0-9]+$ ]] || { echo "GPU ids must be non-negative integers, got: $g" >&2; exit 2; }
done
G="${GPUS[0]}"

REQUIRED_PATH_VARS=(MODEL_CKPT ACTION_VAE_CKPT MODEL_CONFIGS EVAL_HARNESS LIBERO_ROOT EVAL_OUTPUT_ROOT CAE_SRC)
for v in "${REQUIRED_PATH_VARS[@]}"; do
  if [[ -z "${!v:-}" ]]; then echo "missing required environment variable $v" >&2; exit 2; fi
  if [[ "${!v}" != /* ]]; then echo "$v must be an absolute path, got: ${!v}" >&2; exit 2; fi
done
: "${EVAL_IMAGE:?set EVAL_IMAGE}"
: "${TASK_CONFIG:?set TASK_CONFIG to the CAE-FastWAM task config name}"

ARM_SRC="$CAE_SRC"

# The Compact Action Expert is suite-conditioned, so statistics arrive as a
# per-suite template rather than a single file.
: "${DATASET_STATS_TEMPLATE:?set DATASET_STATS_TEMPLATE (must contain the token {suite})}"
case "$DATASET_STATS_TEMPLATE" in
  /*) ;;
  *) echo "DATASET_STATS_TEMPLATE must be an absolute path" >&2; exit 2 ;;
esac
case "$DATASET_STATS_TEMPLATE" in
  *"{suite}"*) ;;
  *) echo "DATASET_STATS_TEMPLATE must contain the literal token {suite}: $DATASET_STATS_TEMPLATE" >&2; exit 2 ;;
esac
# Every per-suite file must exist BEFORE a GPU is touched. A missing stats
# file discovered three suites in wastes hours and yields a partial matrix.
for S in "${SUITES[@]}"; do
  f="${DATASET_STATS_TEMPLATE//\{suite\}/$S}"
  if [[ ! -f "$f" || ! -r "$f" ]]; then
    echo "per-suite statistics missing for $S: $f" >&2; exit 2
  fi
done

for v in MODEL_CKPT ACTION_VAE_CKPT; do
  if [[ ! -f "${!v}" || ! -r "${!v}" ]]; then echo "$v must name a readable file: ${!v}" >&2; exit 2; fi
done
for v in MODEL_CONFIGS EVAL_HARNESS LIBERO_ROOT; do
  if [[ ! -d "${!v}" || ! -r "${!v}" || ! -x "${!v}" ]]; then
    echo "$v must name a readable, searchable directory: ${!v}" >&2; exit 2
  fi
done
# Same up-front rule for the per-suite task configs: Hydra would only fail
# inside the server, one suite at a time.
for S in "${SUITES[@]}"; do
  f="$MODEL_CONFIGS/task/${TASK_CONFIG}_${S}.yaml"
  if [[ ! -f "$f" ]]; then echo "per-suite task config missing for $S: $f" >&2; exit 2; fi
done
if [[ ! -d "$ARM_SRC" || ! -r "$ARM_SRC" || ! -x "$ARM_SRC" ]]; then
  echo "selected source tree must be a readable directory: $ARM_SRC" >&2; exit 2
fi
# Opt-in per-episode rollout videos. Recording is a harness feature (the
# LIBERO benchmark's save_video_dir param), not every harness revision has it,
# and an unknown param would only fail inside the clients -- so check up front.
RECORD_VIDEO="${RECORD_VIDEO:-0}"
if [[ "$RECORD_VIDEO" == 1 ]] && ! grep -q save_video_dir \
    "$EVAL_HARNESS/src/vla_eval/benchmarks/libero/benchmark.py" 2>/dev/null; then
  echo "RECORD_VIDEO=1 needs a harness whose LIBERO benchmark accepts" >&2
  echo "save_video_dir; $EVAL_HARNESS does not." >&2
  exit 2
fi
mkdir -p "$EVAL_OUTPUT_ROOT" || { echo "cannot create EVAL_OUTPUT_ROOT: $EVAL_OUTPUT_ROOT" >&2; exit 2; }
if [[ ! -w "$EVAL_OUTPUT_ROOT" || ! -x "$EVAL_OUTPUT_ROOT" ]]; then
  echo "EVAL_OUTPUT_ROOT must be a writable directory: $EVAL_OUTPUT_ROOT" >&2; exit 2
fi

declare -A MAXSTEPS=( [libero_spatial]=400 [libero_object]=400 [libero_goal]=400 [libero_10]=700 )
for S in "${SUITES[@]}"; do
  if [[ -z "${MAXSTEPS[$S]:-}" ]]; then echo "unknown suite: $S" >&2; exit 2; fi
done

if [[ -n "${EVAL_PORT:-}" ]] && { [[ ! "$EVAL_PORT" =~ ^[0-9]+$ ]] || ((EVAL_PORT < 1 || EVAL_PORT > 65535)); }; then
  echo "EVAL_PORT must be an integer from 1 to 65535, got: $EVAL_PORT" >&2; exit 2
fi
TAG="cae_${STEPS}step"
PORT="${EVAL_PORT:-$((8400 + G))}"
LOGS="$EVAL_OUTPUT_ROOT/$TAG/logs"; mkdir -p "$LOGS"

PW="/tmp/ctr_passwd_$(id -u)"
if ! getent passwd "$(id -u)" > "$PW" 2>/dev/null; then
  printf 'eval:x:%s:%s:Eval user:%s:/bin/bash\n' "$(id -u)" "$(id -g)" "${HOME:-/tmp}" > "$PW"
fi
RG="$(getent group render | cut -d: -f3 || true)"; RG="${RG:-109}"
VG="$(getent group video  | cut -d: -f3 || true)"; VG="${VG:-44}"
CHOME="${HOME:-/tmp/eval_home_$(id -u)}"; [ -w "$CHOME" ] || CHOME=/tmp/eval_home_$(id -u)
mkdir -p "$CHOME" || { echo "cannot create container HOME: $CHOME" >&2; exit 2; }

# Distinct trees must NEVER share a compile cache: the source trees' mot.py
# files differ by thousands of lines and a shared inductor cache silently serves
# the wrong kernels. Keyed by tag, which encodes the step count.
INDUCTOR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/cae_fastwam_inductor_${TAG}}"
TRITON="${TRITON_CACHE_DIR:-/tmp/cae_fastwam_triton_${TAG}}"
mkdir -p "$INDUCTOR" "$TRITON" || { echo "cannot create compile cache dirs" >&2; exit 2; }

declare -a CAE_DOCKER_ARGS=(
  --rm
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined
  --group-add "$VG" --group-add "$RG" --network host
  --user "$(id -u):$(id -g)"
  --env "HOME=$CHOME"
  --volume "$PW:/etc/passwd:ro"
  --volume /mnt:/mnt
  --volume /tmp:/tmp
  --env "PYTHONPATH=$EVAL_HARNESS/src:$LIBERO_ROOT"
  --env "HIP_VISIBLE_DEVICES=$G"
  --env "TORCHINDUCTOR_CACHE_DIR=$INDUCTOR"
  --env "TRITON_CACHE_DIR=$TRITON"
)
declare -A CAE_MOUNTED=(["/mnt"]=rw ["/tmp"]=rw)
cae_bind_path() {
  local path="$1" mode="$2"
  if [[ -z "${CAE_MOUNTED[$path]+x}" ]]; then
    CAE_DOCKER_ARGS+=(--volume "$path:$path:$mode")
    CAE_MOUNTED["$path"]=1
  fi
}
cae_bind_path "$CHOME" rw
cae_bind_path "$EVAL_OUTPUT_ROOT" rw
for v in HF_CACHE_DIR HF_HOME HUGGINGFACE_HUB_CACHE TORCH_HOME XDG_CACHE_HOME DIFFSYNTH_MODEL_BASE_PATH; do
  if [[ -n "${!v:-}" ]]; then
    if [[ "${!v}" != /* ]]; then echo "$v must be an absolute path, got: ${!v}" >&2; exit 2; fi
    mkdir -p "${!v}" || { echo "cannot create $v: ${!v}" >&2; exit 2; }
    cae_bind_path "${!v}" rw
    CAE_DOCKER_ARGS+=(--env "$v")
  fi
done
for v in DIFFSYNTH_DOWNLOAD_SOURCE DIFFSYNTH_SKIP_DOWNLOAD; do
  if [[ -n "${!v:-}" ]]; then CAE_DOCKER_ARGS+=(--env "$v"); fi
done
for v in MODEL_CKPT ACTION_VAE_CKPT MODEL_CONFIGS EVAL_HARNESS LIBERO_ROOT; do
  cae_bind_path "${!v}" ro
done
CAE_DOCKER_ARGS+=(--env ACTION_VAE_CKPT)
cae_bind_path "$ARM_SRC" ro
if [[ -n "${LIBERO_CONFIG_PATH:-}" ]]; then
  if [[ "$LIBERO_CONFIG_PATH" != /* || ! -e "$LIBERO_CONFIG_PATH" ]]; then
    echo "LIBERO_CONFIG_PATH must be an existing absolute path: $LIBERO_CONFIG_PATH" >&2; exit 2
  fi
  cae_bind_path "$LIBERO_CONFIG_PATH" ro
  CAE_DOCKER_ARGS+=(--env LIBERO_CONFIG_PATH)
fi

# MuJoCo rendering: GPU OpenGL (EGL) when the image can create an EGL context
# with this run's devices, else OSMesa; an exported MUJOCO_GL (the launcher's
# choice, or the user's) is kept.
eval_mujoco_gl docker run "${CAE_DOCKER_ARGS[@]}" "$EVAL_IMAGE" python3 || exit 2
CAE_DOCKER_ARGS+=(--env MUJOCO_GL --env PYOPENGL_PLATFORM)

cae_ctr() {  # $1=container name ("" for none) rest=cmd
  local nm="$1"; shift
  local -a args=("${CAE_DOCKER_ARGS[@]}")
  if [[ -n "$nm" ]]; then args+=(--name "$nm"); fi
  docker run "${args[@]}" --workdir "$EVAL_HARNESS" "$EVAL_IMAGE" "$@"
}

SPID=""
start_server() {  # $1=server tag $2=stats $3=task config $4=log
  local stag="$1" stats="$2" tcfg="$3" slog="$4"
  if ss -ltn 2>/dev/null | grep -qE ":$PORT +"; then
    echo "[$stag] ABORT: port $PORT already bound (stale server)" >&2; return 1
  fi
  docker rm -f "cae_srv_$stag" >/dev/null 2>&1 || true
  cae_ctr "cae_srv_$stag" python3 -m vla_eval.model_servers.fastwam \
    --ckpt "$MODEL_CKPT" --dataset_stats_path "$stats" \
    --task_config "$tcfg" \
    --fastwam_src "$ARM_SRC" --config_path "$MODEL_CONFIGS" \
    --num_inference_steps "$STEPS" --seed "$SEED" --port "$PORT" > "$slog" 2>&1 &
  SPID=$!
  for _ in $(seq 1 600); do
    grep -qE "Starting server on ws://|server listening on" "$slog" 2>/dev/null && return 0
    kill -0 "$SPID" 2>/dev/null || {
      echo "[$stag] SERVER DIED" >&2; tail -25 "$slog" >&2
      wait "$SPID" || true; docker rm -f "cae_srv_$stag" >/dev/null 2>&1 || true; return 1
    }
    sleep 2
  done
  echo "[$stag] server never ready" >&2
  docker rm -f "cae_srv_$stag" >/dev/null 2>&1 || true; wait "$SPID" || true; return 1
}
stop_server() {
  docker rm -f "cae_srv_$1" >/dev/null 2>&1 || true
  [[ -n "$SPID" ]] && { wait "$SPID" 2>/dev/null || true; }
  SPID=""
}

write_cfg() {  # $1=file $2=outdir rest=suites
  local f="$1" out="$2"; shift 2
  { echo "server:"; echo "  url: \"ws://localhost:$PORT\""; echo "  timeout: 600.0"
    echo "output_dir: \"$out\""; echo "benchmarks:"
    for S in "$@"; do
      echo "  - benchmark: \"vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark\""
      echo "    subname: $S"; echo "    mode: sync"
      echo "    episodes_per_task: $TRIALS"
      echo "    shard_by_task: true"          # protocol: trials of a task stay in one env
      echo "    params:"
      echo "      suite: $S"; echo "      seed: $SEED"
      echo "      num_steps_wait: $WAIT"
      echo "      max_steps: ${MAXSTEPS[$S]}"
      echo "      quat_no_antipodal: true"    # LOAD-BEARING -- see PROTOCOL.md
      if [[ "$RECORD_VIDEO" == 1 ]]; then
        echo "      save_video_dir: \"$out/videos/$S\""
      fi
    done
  } > "$f"
}

run_clients() {  # $1=cfg $2=out $3=phase
  local cfg="$1" out="$2" phase="$3" rc=0; local pids=()
  for k in $(seq 0 $((SHARDS-1))); do
    cae_ctr "" python3 src/vla_eval/cli/main.py run -c "$cfg" \
      --server-url "ws://localhost:$PORT" --output-dir "$out" --no-docker -y \
      --shard-id "$k" --num-shards "$SHARDS" > "$LOGS/${phase}.shard${k}.log" 2>&1 &
    pids+=($!)
  done
  for q in "${pids[@]}"; do wait "$q" || rc=1; done
  return $rc
}

OUT="$EVAL_OUTPUT_ROOT/$TAG"; mkdir -p "$OUT"

# RESUME GUARD. A finished suite is skipped on a rerun, and summarize.py pools
# every result file under $OUT, so resuming a directory written with another
# trial count, tree or checkpoint would silently mix the two runs. The run
# directory and every .done_<suite> marker record the run's identity; a rerun
# resumes only with the same one.
IDENTITY="$(printf '%s\n' \
  "steps=$STEPS" \
  "tree=$(realpath -m "$ARM_SRC")" \
  "configs=$(realpath -m "$MODEL_CONFIGS")" "task_config=$TASK_CONFIG" \
  "checkpoint=$(eval_file_identity "$MODEL_CKPT")" \
  "action_vae=$(eval_file_identity "$ACTION_VAE_CKPT")" \
  "dataset_stats_template=$DATASET_STATS_TEMPLATE" \
  "trials=$TRIALS" "seed=$SEED" "num_steps_wait=$WAIT" "shards=$SHARDS")"
has_results=0
compgen -G "$OUT/.done_*" >/dev/null && has_results=1
compgen -G "$OUT/LIBEROBenchmark_*.json" >/dev/null && has_results=1
eval_claim_run_dir "$OUT/.run_identity" "$IDENTITY" "$has_results" || exit 2

rc=0
echo "[$TAG] start gpu=$G port=$PORT tree=$ARM_SRC steps=$STEPS"

# Suite-conditioned: one server per suite, sequential, resumable.
for S in "${SUITES[@]}"; do
  if [[ -f "$OUT/.done_$S" ]]; then
    # A marker from another run (or an empty one, from before markers recorded
    # the run) is refused rather than trusted.
    if [[ "$(<"$OUT/.done_$S")" != "$IDENTITY" ]]; then
      echo "[$TAG] refusing to skip $S: $OUT/.done_$S was written by a different run;" >&2
      echo "use a fresh --output-dir" >&2
      rc=2; break
    fi
    echo "[$TAG] skip $S (already done)"; continue
  fi
  stag="${TAG}_${S}"
  stats="${DATASET_STATS_TEMPLATE//\{suite\}/$S}"
  if ! start_server "$stag" "$stats" "${TASK_CONFIG}_${S}" "$LOGS/${S}.server.log"; then rc=1; break; fi
  CFG="$OUT/eval_${S}.yaml"; write_cfg "$CFG" "$OUT" "$S"
  if run_clients "$CFG" "$OUT" "$S"; then printf '%s\n' "$IDENTITY" > "$OUT/.done_$S"; else rc=1; fi
  stop_server "$stag"
  [[ "$rc" -eq 0 ]] || break
done

if ! python3 "$(dirname "$0")/summarize.py" "$EVAL_OUTPUT_ROOT"; then
  echo "evaluation summary failed" >&2; rc=1
fi
if [ "$rc" -ne 0 ]; then echo "evaluation failed" >&2; exit "$rc"; fi
echo "EVAL-DONE"
