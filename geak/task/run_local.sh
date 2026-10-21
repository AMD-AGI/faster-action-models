#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Container wrapper for the GEAK task commands.
#
# Every command runs INSIDE the image: an optimization campaign must measure the
# stack it will ship on, and the host frequently has no matching torch at all.
#
#   bash run_local.sh <compile|oracle|oracle-refresh|correctness|rollout|performance>
set -euo pipefail
CMD="${1:?usage: run_local.sh <compile|oracle|oracle-refresh|correctness|rollout|performance>}"
if [[ "$CMD" == -h || "$CMD" == --help ]]; then
  cat <<'EOF'
usage: bash geak/task/run_local.sh <compile|oracle|oracle-refresh|correctness|rollout|performance>

Runs one GEAK task command in EVAL_IMAGE (geak/task/scripts/README.md). Required environment: EVAL_IMAGE,
MODEL_SRC_ROOT, MODEL_CKPT, DATASET_STATS, HF_CACHE_DIR, MODEL_PYTHON (the interpreter inside the image); for
correctness and rollout also LIBERO_ROOT and LIBERO_CONFIG_PATH. Optional: HIP_VISIBLE_DEVICES (default 0), FWE_SRC,
FWE_CONFIG_DIR, DIFFSYNTH_*, GATE_*, ORACLE_*, FWE_*, EVAL_DOCKER_ARGS (extra docker run flags).
See docs/fastwam-geak-inference-optimization.md.
EOF
  exit 0
fi
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${HIP_VISIBLE_DEVICES:-0}"
IMAGE="${EVAL_IMAGE:?set EVAL_IMAGE}"

REQUIRED_VARS=(
  MODEL_SRC_ROOT MODEL_CKPT DATASET_STATS HF_CACHE_DIR MODEL_PYTHON
)
# Only `correctness` (and `rollout`, its closed-loop half alone) runs LIBERO episodes (FastWAM's own evaluator);
# the other commands need no LIBERO paths.
LIBERO_VARS=(LIBERO_ROOT LIBERO_CONFIG_PATH)
if [[ "$CMD" == correctness || "$CMD" == rollout ]]; then
  REQUIRED_VARS+=("${LIBERO_VARS[@]}")
fi
for v in "${REQUIRED_VARS[@]}"; do
  if [[ -z "${!v:-}" ]]; then
    printf 'missing required environment variable %s; see docs/fastwam-geak-inference-optimization.md\n' "$v" >&2
    exit 2
  fi
  if [[ "${!v}" != /* ]]; then
    printf '%s must be an absolute path, got: %s\n' "$v" "${!v}" >&2
    exit 2
  fi
done

HOST_PATH_VARS=(MODEL_SRC_ROOT MODEL_CKPT DATASET_STATS HF_CACHE_DIR)
if [[ "$CMD" == correctness || "$CMD" == rollout ]]; then
  HOST_PATH_VARS+=("${LIBERO_VARS[@]}")
fi
# MODEL_PYTHON intentionally is not a host path: it names the interpreter in
# EVAL_IMAGE. task_runner.py validates it after the container starts.
for v in "${HOST_PATH_VARS[@]}"; do
  if [[ ! -e "${!v}" ]]; then
    printf '%s does not exist on the host: %s\n' "$v" "${!v}" >&2
    exit 2
  fi
done
if [[ -n "${FWE_SRC:-}" ]]; then
  if [[ "$FWE_SRC" != /* || ! -d "$FWE_SRC" ]]; then
    printf 'FWE_SRC must be an existing absolute source directory: %s\n' "$FWE_SRC" >&2
    exit 2
  fi
fi
if [[ -n "${FWE_CONFIG_DIR:-}" ]]; then
  if [[ "$FWE_CONFIG_DIR" != /* || ! -d "$FWE_CONFIG_DIR" ]]; then
    printf 'FWE_CONFIG_DIR must be an existing absolute config directory: %s\n' "$FWE_CONFIG_DIR" >&2
    exit 2
  fi
fi

PW="/tmp/ctr_passwd_$(id -u)"
if ! getent passwd "$(id -u)" > "$PW" 2>/dev/null; then
  printf 'geak:x:%s:%s:GEAK user:%s:/bin/bash\n' \
    "$(id -u)" "$(id -g)" "${HOME:-/tmp}" > "$PW"
fi
RG="$(getent group render | cut -d: -f3 || true)"; RG="${RG:-109}"
VG="$(getent group video | cut -d: -f3 || true)";  VG="${VG:-44}"
# Container HOME: use the caller's when usable, else a per-uid /tmp dir. Hard-
# coding a home path breaks on any node where it is absent or unwritable, and
# only regenerable autotune caches live there -- do NOT carry those between
# nodes, since MIOpen/hipBLASLt select algorithms per box.
CHOME="${HOME:-/tmp/geak_home_$(id -u)}"; [ -w "$CHOME" ] || CHOME=/tmp/geak_home_$(id -u)
mkdir -p "$CHOME"

# Forward knobs ONLY when actually set: an empty value is not neutral downstream
# (e.g. an empty threshold crashes float()). Model kill-switches are forwarded by
# prefix so an OFF arm can actually be run -- a wrapper that silently drops them
# makes every kill-switch check a no-op that reports the optimized path as the
# baseline.
declare -a DOCKER_ARGS=(
  --rm
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined
  --group-add "$VG" --group-add "$RG" --network host
  --user "$(id -u):$(id -g)"
  --env "HOME=$CHOME"
  --volume "$PW:/etc/passwd:ro"
  --volume "$CHOME:$CHOME"
  --volume /mnt:/mnt
  --volume /tmp:/tmp
  --env "HIP_VISIBLE_DEVICES=$GPU"
  --env "HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}"
)

declare -A FORWARDED=()
forward_env() {
  local name="$1"
  if [[ -v "$name" && -z "${FORWARDED[$name]+x}" ]]; then
    DOCKER_ARGS+=(--env "$name")
    FORWARDED["$name"]=1
  fi
}

for v in "${REQUIRED_VARS[@]}" \
         GATE_SR_THRESHOLD GATE_SUITES GATE_ATTEMPTS \
         GATE_EPISODES_PER_TASK FWE_WARMUP FWE_ITERS FWE_STAGE_ITERS \
         FWE_STEPS FWE_SRC FWE_CONFIG_DIR FWE_NAIVE_COMPILE ORACLE_BASE \
         ORACLE_TOL ORACLE_STEPS ORACLE_WARMUP ORACLE_SAMPLES \
         TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR; do
  forward_env "$v"
done
while IFS= read -r v; do
  case "$v" in
    FASTWAM_*|GEAK_*|MODEL_*|DIFFSYNTH_*) forward_env "$v" ;;
  esac
done < <(compgen -v)

# Bind every selected host path at the same absolute location. Mounting only
# /mnt and /tmp made paths under workspaces, home directories, and other storage
# roots disappear inside the container.
declare -A MOUNTED=()
bind_path() {
  local path="$1" mode="${2:-ro}"
  if [[ -z "${MOUNTED[$path]+x}" ]]; then
    DOCKER_ARGS+=(--volume "$path:$path:$mode")
    MOUNTED["$path"]=1
  fi
}
bind_path "$TASK_DIR" rw
for v in "${HOST_PATH_VARS[@]}"; do
  if [[ "$v" == HF_CACHE_DIR ]]; then
    bind_path "${!v}" rw
  else
    bind_path "${!v}" ro
  fi
done
if [[ -n "${FWE_SRC:-}" ]]; then bind_path "$FWE_SRC" ro; fi
# The Wan assets: DIFFSYNTH_MODEL_BASE_PATH when set (forwarded above), else HF_CACHE_DIR.
if [[ -n "${DIFFSYNTH_MODEL_BASE_PATH:-}" ]]; then
  if [[ "$DIFFSYNTH_MODEL_BASE_PATH" != /* || ! -d "$DIFFSYNTH_MODEL_BASE_PATH" ]]; then
    printf 'DIFFSYNTH_MODEL_BASE_PATH must be an existing absolute directory: %s\n' "$DIFFSYNTH_MODEL_BASE_PATH" >&2
    exit 2
  fi
  bind_path "$DIFFSYNTH_MODEL_BASE_PATH" rw
fi
if [[ -n "${FWE_CONFIG_DIR:-}" ]]; then bind_path "$FWE_CONFIG_DIR" ro; fi
# Compile caches are keyed on the source text, not on every env switch that
# changes the traced graph, so a candidate run with different FASTWAM_* levers
# can be served another configuration's kernels. Give each such run its own
# cache dir; it has to be mounted as well as forwarded to be seen.
for v in TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR; do
  if [[ -n "${!v:-}" ]]; then
    [[ "${!v}" == /* ]] || { printf '%s must be an absolute path: %s\n' "$v" "${!v}" >&2; exit 2; }
    mkdir -p "${!v}"
    bind_path "${!v}" rw
  fi
done

# MuJoCo rendering, for `correctness` and `rollout` (the commands that run LIBERO
# episodes): an exported MUJOCO_GL is kept; otherwise GPU OpenGL (EGL) when the
# image can create an EGL context with these devices, else OSMesa -- the same
# choice as the evaluation launchers (scripts/eval_flags.sh). task_runner.py
# falls back to OSMesa when nothing is forwarded.
EVAL_FLAGS_SH="$TASK_DIR/../../scripts/eval_flags.sh"
if [[ ( "$CMD" == correctness || "$CMD" == rollout ) && -z "${MUJOCO_GL:-}" && -f "$EVAL_FLAGS_SH" ]]; then
  # shellcheck source=../../scripts/eval_flags.sh
  source "$EVAL_FLAGS_SH"
  eval_mujoco_gl docker run "${DOCKER_ARGS[@]}" "$IMAGE" "$MODEL_PYTHON" || exit 2
fi
forward_env MUJOCO_GL
forward_env PYOPENGL_PLATFORM

# EVAL_DOCKER_ARGS: extra `docker run` flags, split on whitespace (for example
# "--name geak-perf-g3 --cpuset-cpus 0-31" to name the container and pin its CPUs, as the reference
# latency runs did).
declare -a EXTRA_DOCKER_ARGS=()
if [[ -n "${EVAL_DOCKER_ARGS:-}" ]]; then read -r -a EXTRA_DOCKER_ARGS <<< "$EVAL_DOCKER_ARGS"; fi

exec docker run "${DOCKER_ARGS[@]}" "${EXTRA_DOCKER_ARGS[@]+"${EXTRA_DOCKER_ARGS[@]}"}" \
  --workdir "$TASK_DIR" \
  "$IMAGE" "$MODEL_PYTHON" "$TASK_DIR/scripts/task_runner.py" "$CMD"
