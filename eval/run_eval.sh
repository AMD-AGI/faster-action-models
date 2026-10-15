#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Closed-loop LIBERO evaluation. One CONFIG per GPU, all configs concurrent,
# N client shards per GPU sharing that GPU's model server.
#
# Protocol comes from the MODEL'S OWN eval code, not the harness defaults --
# see PROTOCOL.md. The values below are FastWAM's; change them for your model
# and re-derive them from its evaluation source, not from a reproduction doc.
set -uo pipefail
# shellcheck source=../scripts/eval_flags.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/eval_flags.sh"

if [[ " $* " == *" -h "* || " $* " == *" --help "* ]]; then
  cat <<'EOF'
Internal engine for scripts/eval_fastwam_libero.sh (the FastWAM baseline and its
optimized tree; not --arm cae); run that launcher.

Usage:
  bash eval/run_eval.sh [--arm baseline|optimized] [--gpus CSV] [--trials N]
                        [--shards N] [--suites CSV] [--task-config NAME]

--arm filters the CONFIGS list (arm:steps pairs; default
"baseline:10 optimized:10 optimized:2"); each selected config runs
on its own GPU from --gpus (default EVAL_GPUS, else 0,1,2,3). --shards is the
number of harness clients per policy server (default 10). Paths come from the
environment: MODEL_CKPT DATASET_STATS MODEL_CONFIGS EVAL_HARNESS LIBERO_ROOT
BASELINE_SRC OPTIMIZED_SRC EVAL_OUTPUT_ROOT EVAL_IMAGE.

Config i's server listens on EVAL_PORT+i when EVAL_PORT is set, else on
8300 + its GPU id. EVAL_ARM_LABEL names the optimized arm's result directory
(<label>_<N>step; default optimized); the launcher sets it to the arm the user
passed.
EOF
  exit 0
fi

REQUIRED_PATH_VARS=(
  MODEL_CKPT DATASET_STATS MODEL_CONFIGS EVAL_HARNESS LIBERO_ROOT
  BASELINE_SRC EVAL_OUTPUT_ROOT
)
for v in "${REQUIRED_PATH_VARS[@]}"; do
  if [[ -z "${!v:-}" ]]; then
    echo "missing required environment variable $v" >&2
    exit 2
  fi
  if [[ "${!v}" != /* ]]; then
    echo "$v must be an absolute path, got: ${!v}" >&2
    exit 2
  fi
done
: "${EVAL_IMAGE:?set EVAL_IMAGE}"
# Source trees per arm: baseline (unmodified) and optimized (post-GEAK).
# SILENT-WRONG-NUMBER GUARD. Leaving OPTIMIZED_SRC unset used to fall back to
# BASELINE_SRC, so a pristine run was written into a directory named
# optimized_Nstep. Record the fallback here and refuse it below if -- and only
# if -- an `optimized` config is actually selected. A baseline-only run
# legitimately never sets OPTIMIZED_SRC, so the check cannot be unconditional.
OPTIMIZED_SRC_DEFAULTED=0
if [[ -z "${OPTIMIZED_SRC:-}" ]]; then
  OPTIMIZED_SRC_DEFAULTED=1
fi
OPTIMIZED_SRC="${OPTIMIZED_SRC:-$BASELINE_SRC}"
if [[ "$OPTIMIZED_SRC" != /* ]]; then
  echo "OPTIMIZED_SRC must be an absolute path, got: $OPTIMIZED_SRC" >&2
  exit 2
fi

for v in MODEL_CKPT DATASET_STATS; do
  if [[ ! -f "${!v}" || ! -r "${!v}" ]]; then
    echo "$v must name a readable file: ${!v}" >&2
    exit 2
  fi
done
for v in MODEL_CONFIGS EVAL_HARNESS LIBERO_ROOT BASELINE_SRC OPTIMIZED_SRC; do
  if [[ ! -d "${!v}" || ! -r "${!v}" || ! -x "${!v}" ]]; then
    echo "$v must name a readable, searchable directory: ${!v}" >&2
    exit 2
  fi
done
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
if [[ -n "${EVAL_PORT:-}" ]] && { [[ ! "$EVAL_PORT" =~ ^[0-9]+$ ]] || ((EVAL_PORT < 1 || EVAL_PORT > 65535)); }; then
  echo "EVAL_PORT must be an integer from 1 to 65535, got: $EVAL_PORT" >&2
  exit 2
fi
if [[ -n "${EVAL_ARM_LABEL:-}" && ( ! "$EVAL_ARM_LABEL" =~ ^[a-z0-9][a-z0-9-]*$ || "$EVAL_ARM_LABEL" == baseline ) ]]; then
  echo "EVAL_ARM_LABEL must be a lower-case name other than baseline, got: $EVAL_ARM_LABEL" >&2
  exit 2
fi
mkdir -p "$EVAL_OUTPUT_ROOT" || {
  echo "cannot create EVAL_OUTPUT_ROOT: $EVAL_OUTPUT_ROOT" >&2
  exit 2
}
if [[ ! -d "$EVAL_OUTPUT_ROOT" || ! -w "$EVAL_OUTPUT_ROOT" || ! -x "$EVAL_OUTPUT_ROOT" ]]; then
  echo "EVAL_OUTPUT_ROOT must be a writable directory: $EVAL_OUTPUT_ROOT" >&2
  exit 2
fi

SUITES=(libero_spatial libero_object libero_goal libero_10)
# Per-suite horizons from the model's own _get_max_steps. NOT the harness map.
declare -A MAXSTEPS=( [libero_spatial]=400 [libero_object]=400 [libero_goal]=400 [libero_10]=700 )
SEED="${SEED:-42}"          # model's eval seed, NOT the harness default
WAIT="${WAIT:-30}"          # model's num_steps_wait, NOT the harness default
TRIALS="${TRIALS:-50}"
SHARDS="${SHARDS:-10}"      # 10 tasks/suite -> 1 task per shard per suite
# GPU pool. Overridable because a shared box rarely has 0-3 free: on this host
# a GEAK campaign holds 3 and 4. --gpus wins over EVAL_GPUS, which wins over the
# default. Spelled --gpus CSV to match scripts/eval_dit4dit_robocasa.sh.
IFS=, read -r -a GPUS <<< "${EVAL_GPUS:-0,1,2,3}"
# arm:steps. Baseline only needs its reference point; step reduction is an
# optimized-side claim.
CONFIGS="${CONFIGS:-baseline:10 optimized:10 optimized:2}"

while [ $# -gt 0 ]; do case "$1" in
  --arm)
    filtered=""
    for spec in $CONFIGS; do
      if [[ "${spec%%:*}" == "$2" ]]; then filtered+="${spec} "; fi
    done
    CONFIGS="$filtered"
    shift 2
    ;;
  --gpus) IFS=, read -r -a GPUS <<< "$2"; shift 2 ;;
  --task-config) TASK_CONFIG="$2"; shift 2 ;;
  --trials) TRIALS="$2"; shift 2 ;;
  --shards) SHARDS="$2"; shift 2 ;;
  --suites) IFS=, read -r -a SUITES <<< "$2"; shift 2 ;;
  *) echo "unknown arg $1"; exit 2 ;;
esac; done

read -r -a CONFIG_LIST <<< "$CONFIGS"
if [ "${#CONFIG_LIST[@]}" -eq 0 ]; then
  echo "no evaluation configs selected" >&2
  exit 2
fi
if [ "${#CONFIG_LIST[@]}" -gt "${#GPUS[@]}" ]; then
  echo "selected ${#CONFIG_LIST[@]} configs but only ${#GPUS[@]} GPUs are configured" >&2
  exit 2
fi
for spec in "${CONFIG_LIST[@]}"; do
  arm="${spec%%:*}"; steps="${spec##*:}"
  if [[ "$arm" != baseline && "$arm" != optimized ]]; then
    echo "invalid config arm '$arm' in '$spec' (expected baseline or optimized)" >&2
    exit 2
  fi
  if [[ ! "$steps" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid denoise step count in config '$spec'" >&2
    exit 2
  fi
done

if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "no GPUs configured; pass --gpus CSV or set EVAL_GPUS" >&2
  exit 2
fi
for g in "${GPUS[@]}"; do
  if [[ ! "$g" =~ ^[0-9]+$ ]]; then
    echo "GPU ids must be non-negative integers, got: $g" >&2
    exit 2
  fi
done

# Refuse the silent fallback only when it would actually mislabel a run.
if [ "$OPTIMIZED_SRC_DEFAULTED" -eq 1 ]; then
  for spec in "${CONFIG_LIST[@]}"; do
    if [[ "${spec%%:*}" == optimized ]]; then
      echo "OPTIMIZED_SRC is unset but config '$spec' selects the optimized arm;" >&2
      echo "refusing to run BASELINE_SRC under an optimized_Nstep label" >&2
      exit 2
    fi
  done
fi

PW="/tmp/ctr_passwd_$(id -u)"
if ! getent passwd "$(id -u)" > "$PW" 2>/dev/null; then
  printf 'eval:x:%s:%s:Eval user:%s:/bin/bash\n' \
    "$(id -u)" "$(id -g)" "${HOME:-/tmp}" > "$PW"
fi
RG="$(getent group render | cut -d: -f3 || true)"; RG="${RG:-109}"
VG="$(getent group video | cut -d: -f3 || true)";  VG="${VG:-44}"
CHOME="${HOME:-/tmp/eval_home_$(id -u)}"; [ -w "$CHOME" ] || CHOME=/tmp/eval_home_$(id -u)
mkdir -p "$CHOME" || {
  echo "cannot create container HOME: $CHOME" >&2
  exit 2
}
if [[ ! -d "$CHOME" || ! -w "$CHOME" || ! -x "$CHOME" ]]; then
  echo "container HOME must be a writable directory: $CHOME" >&2
  exit 2
fi

arm_src() { [ "$1" = baseline ] && echo "$BASELINE_SRC" || echo "$OPTIMIZED_SRC"; }
# A config's result directory, <label>_<N>step. The optimized arm is labelled
# EVAL_ARM_LABEL when the launcher sets it, so result paths and the summary
# carry the arm name the user passed.
config_tag() {  # $1=arm:steps
  local arm="${1%%:*}" steps="${1##*:}"
  [[ "$arm" == optimized ]] && arm="${EVAL_ARM_LABEL:-optimized}"
  echo "${arm}_${steps}step"
}

declare -a EVAL_DOCKER_ARGS=(
  --rm
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined
  --group-add "$VG" --group-add "$RG" --network host
  --user "$(id -u):$(id -g)"
  --env "HOME=$CHOME"
  --volume "$PW:/etc/passwd:ro"
  --volume /mnt:/mnt
  --volume /tmp:/tmp
  --env "PYTHONPATH=$EVAL_HARNESS/src:$LIBERO_ROOT"
)

declare -A EVAL_MOUNTED=(["/mnt"]=rw ["/tmp"]=rw)
eval_bind_path() {
  local path="$1" mode="$2"
  if [[ -z "${EVAL_MOUNTED[$path]+x}" ]]; then
    EVAL_DOCKER_ARGS+=(--volume "$path:$path:$mode")
    EVAL_MOUNTED["$path"]=1
  fi
}

# Writable locations first so an overlapping exact path is never downgraded.
eval_bind_path "$CHOME" rw
eval_bind_path "$EVAL_OUTPUT_ROOT" rw
# Compile-cache dirs are path-valued, so they go through the bind loop -- an
# --env alone would name a directory the container cannot see. Distinct source
# trees must never share an inductor cache; the wrapper sets these per arm.
for v in HF_CACHE_DIR HF_HOME HUGGINGFACE_HUB_CACHE TORCH_HOME XDG_CACHE_HOME \
         TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR DIFFSYNTH_MODEL_BASE_PATH; do
  if [[ -n "${!v:-}" ]]; then
    if [[ "${!v}" != /* ]]; then
      echo "$v must be an absolute path, got: ${!v}" >&2
      exit 2
    fi
    mkdir -p "${!v}" || {
      echo "cannot create $v: ${!v}" >&2
      exit 2
    }
    if [[ ! -d "${!v}" || ! -w "${!v}" || ! -x "${!v}" ]]; then
      echo "$v must be a writable directory: ${!v}" >&2
      exit 2
    fi
    eval_bind_path "${!v}" rw
    EVAL_DOCKER_ARGS+=(--env "$v")
  fi
done
# Scalar (non-path) pass-through. An unforwarded variable is a SILENT NO-OP:
# the switch never fires and the run is labelled as though it did.
for v in DIFFSYNTH_DOWNLOAD_SOURCE DIFFSYNTH_SKIP_DOWNLOAD FASTWAM_KVSLAB; do
  if [[ -n "${!v:-}" ]]; then
    EVAL_DOCKER_ARGS+=(--env "$v")
  fi
done
for v in MODEL_CKPT DATASET_STATS MODEL_CONFIGS EVAL_HARNESS LIBERO_ROOT \
         BASELINE_SRC OPTIMIZED_SRC; do
  eval_bind_path "${!v}" ro
done
if [[ -n "${LIBERO_CONFIG_PATH:-}" ]]; then
  if [[ "$LIBERO_CONFIG_PATH" != /* || ! -e "$LIBERO_CONFIG_PATH" ]]; then
    echo "LIBERO_CONFIG_PATH must be an existing absolute path: $LIBERO_CONFIG_PATH" >&2
    exit 2
  fi
  eval_bind_path "$LIBERO_CONFIG_PATH" ro
  EVAL_DOCKER_ARGS+=(--env LIBERO_CONFIG_PATH)
fi

# MuJoCo rendering: GPU OpenGL (EGL) when the image can create an EGL context
# with this run's devices, else OSMesa; an exported MUJOCO_GL (the launcher's
# choice, or the user's) is kept. Probed once, in the image, on the first GPU.
eval_mujoco_gl docker run "${EVAL_DOCKER_ARGS[@]}" --env "HIP_VISIBLE_DEVICES=${GPUS[0]}" \
  "$EVAL_IMAGE" python3 || exit 2
EVAL_DOCKER_ARGS+=(--env MUJOCO_GL --env PYOPENGL_PLATFORM)

evalctr() {  # $1=gpu $2=arm rest=cmd   (NOT named `ctr`: that is the containerd CLI)
  local gpu="$1" arm="$2"; shift 2
  local -a args=("${EVAL_DOCKER_ARGS[@]}")
  if [[ -n "${EVAL_CNAME:-}" ]]; then args+=(--name "$EVAL_CNAME"); fi
  args+=(--env "HIP_VISIBLE_DEVICES=$gpu")
  # Baseline means NO compile of any kind. Unmodified sources may still enable
  # torch.compile by default, so disable every compile path explicitly.
  if [[ "$arm" == baseline ]]; then
    args+=(
      --env FASTWAM_MOT_COMPILE_MODE=off
      --env FASTWAM_VAE_COMPILE=0
      --env FASTWAM_COMPILE=0
    )
  fi
  docker run "${args[@]}" --workdir "$EVAL_HARNESS" "$EVAL_IMAGE" "$@"
}

run_config() {  # $1=arm:steps $2=gpu $3=config index
  local arm="${1%%:*}" steps="${1##*:}" G="$2"
  local TAG PORT=$((8300 + G))
  TAG="$(config_tag "$1")"
  if [[ -n "${EVAL_PORT:-}" ]]; then PORT=$((EVAL_PORT + $3)); fi
  local OUT="$EVAL_OUTPUT_ROOT/$TAG"; mkdir -p "$OUT"
  local SLOG="$OUT/server.log" CNAME="eval_srv_${TAG}"

  # PORT GUARD. A stale server keeps its port; the new one dies with Errno 98 and
  # the clients then silently evaluate the WRONG arm under this run's label.
  if ss -ltn 2>/dev/null | grep -qE ":$PORT +"; then
    echo "[$TAG] ABORT: port $PORT already bound (stale server)"; return 1
  fi
  docker rm -f "$CNAME" >/dev/null 2>&1 || true

  local -a tcfg=()
  if [[ -n "${TASK_CONFIG:-}" ]]; then tcfg=(--task_config "$TASK_CONFIG"); fi
  EVAL_CNAME="$CNAME" evalctr "$G" "$arm" python3 -m vla_eval.model_servers.fastwam \
    --ckpt "$MODEL_CKPT" --dataset_stats_path "$DATASET_STATS" \
    --fastwam_src "$(arm_src "$arm")" --config_path "$MODEL_CONFIGS" \
    "${tcfg[@]+"${tcfg[@]}"}" \
    --num_inference_steps "$steps" --seed "$SEED" --port "$PORT" > "$SLOG" 2>&1 &
  local SPID=$!
  local ready=0
  for _ in $(seq 1 300); do
    # Readiness banners differ per server implementation; match either.
    grep -qE "Starting server on ws://|server listening on" "$SLOG" 2>/dev/null && { ready=1; break; }
    kill -0 "$SPID" 2>/dev/null || {
      echo "[$TAG] SERVER DIED"
      tail -20 "$SLOG"
      wait "$SPID" || true
      docker rm -f "$CNAME" >/dev/null 2>&1 || true
      return 1
    }
    sleep 2
  done
  [ "$ready" = 1 ] || {
    echo "[$TAG] server never ready"
    docker rm -f "$CNAME" >/dev/null 2>&1 || true
    wait "$SPID" || true
    return 1
  }

  local CFG="$OUT/eval.yaml"
  { echo "server:"; echo "  url: \"ws://localhost:$PORT\""; echo "  timeout: 600.0"
    echo "output_dir: \"$OUT\""; echo "benchmarks:"
    for S in "${SUITES[@]}"; do
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
        echo "      save_video_dir: \"$OUT/videos/$S\""
      fi
    done
  } > "$CFG" || {
    echo "[$TAG] failed to write config $CFG" >&2
    docker rm -f "$CNAME" >/dev/null 2>&1 || true
    wait "$SPID" || true
    return 1
  }

  local pids=()
  for k in $(seq 0 $((SHARDS-1))); do
    evalctr "$G" "$arm" python3 src/vla_eval/cli/main.py run -c "$CFG" \
      --server-url "ws://localhost:$PORT" --output-dir "$OUT" --no-docker -y \
      --shard-id "$k" --num-shards "$SHARDS" > "$OUT/shard${k}.log" 2>&1 &
    pids+=($!)
  done
  local rc=0; for q in "${pids[@]}"; do wait "$q" || rc=1; done
  if [ "$(docker inspect -f '{{.State.Running}}' "$CNAME" 2>/dev/null)" != true ]; then
    echo "[$TAG] model server exited before evaluation completed" >&2
    rc=1
  fi
  # Stop the CONTAINER, not just the docker client -- otherwise the port stays bound.
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  wait "$SPID" || true
  echo "[$TAG] done rc=$rc"
  return $rc
}

# RESUME GUARD. Each config writes into $EVAL_OUTPUT_ROOT/<arm>_<N>step and
# summarize.py pools every result file there, so a rerun into a directory
# holding another tree's, checkpoint's or trial count's results would silently
# merge them. Every config directory records what it holds, checked before any
# server starts.
config_identity() {  # $1=arm:steps
  local arm="${1%%:*}" steps="${1##*:}"
  printf '%s\n' \
    "arm=$arm" "steps=$steps" \
    "tree=$(realpath -m "$(arm_src "$arm")")" \
    "configs=$(realpath -m "$MODEL_CONFIGS")" \
    "task_config=${TASK_CONFIG:-}" \
    "checkpoint=$(eval_file_identity "$MODEL_CKPT")" \
    "dataset_stats=$(eval_file_identity "$DATASET_STATS")" \
    "trials=$TRIALS" "seed=$SEED" "num_steps_wait=$WAIT" "shards=$SHARDS" \
    "fastwam_kvslab=${FASTWAM_KVSLAB:-}"
}
for spec in "${CONFIG_LIST[@]}"; do
  cdir="$EVAL_OUTPUT_ROOT/$(config_tag "$spec")"
  mkdir -p "$cdir"
  has_results=0
  compgen -G "$cdir/LIBEROBenchmark_*.json" >/dev/null && has_results=1
  eval_claim_run_dir "$cdir/.run_identity" "$(config_identity "$spec")" "$has_results" || exit 2
done

i=0; pids=()
for spec in "${CONFIG_LIST[@]}"; do
  run_config "$spec" "${GPUS[$i]}" "$i" &
  pids+=($!)
  i=$((i+1))
done

rc=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then rc=1; fi
done
if ! python3 "$(dirname "$0")/summarize.py" "$EVAL_OUTPUT_ROOT"; then
  echo "evaluation summary failed" >&2
  rc=1
fi
if [ "$rc" -ne 0 ]; then
  echo "evaluation failed" >&2
  exit "$rc"
fi
echo "EVAL-DONE"
