#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run a command inside the evaluation image (EVAL_IMAGE, docker/Dockerfile.eval), the way the DiT4DiT launchers
# need it: the Python environments DiT4DiT/environments/setup_dit4dit_rocm_envs.sh builds use the image's
# site-packages, so they work only inside that image.
#
#   bash scripts/in_eval_image.sh bash scripts/eval_dit4dit_libero.sh --arm geak --steps 2 --gpus 0 --smoke
#   bash scripts/in_eval_image.sh bash scripts/eval_dit4dit_robocasa.sh --arm baseline --gpus 0,1 --smoke
#
# The container gets every GPU (/dev/kfd, /dev/dri; the launchers pick theirs with --gpus), the video and render
# groups, the invoking user (with a passwd entry and USER, so getpass works for any host uid), the host network
# (policy-server ports), the host IPC namespace (shared memory), /tmp, and the repository at its own path, with the
# repository as the working directory. It forwards the variables the launchers read (DIT4DIT_*, ROBOCASA_*,
# ROBOSUITE_*, LIBERO_*, EVAL_*, INSTINCT_*, HF_*, TORCH*, TRITON_*, NUMBA_*, MUJOCO_GL, PYOPENGL_PLATFORM, STEPS,
# TRIALS, PORT, ...) and mounts each absolute path they name at its own path; for a *_CKPT file that is its run
# directory (two levels up, where config.yaml lives) plus the existing absolute paths that config.yaml names (the
# local Cosmos-Predict2.5-2B copy; for a file, its directory), read-only. An absolute --output-dir is created and mounted writable.
#
# Environment:
#   EVAL_IMAGE           the evaluation image (required)
#   IN_IMAGE_MOUNTS      extra host paths to mount at their own path, colon-separated
#   EVAL_DOCKER_ARGS     extra `docker run` flags (for example --cpuset-cpus 0-31)
# Inside the image (FAM_IN_EVAL_IMAGE=1 is exported there) the command runs directly, so a script that calls this
# helper also works when it is itself started inside the image.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -eq 0 || "$1" == -h || "$1" == --help ]]; then
  sed -n '4,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  [[ $# -eq 0 ]] && exit 2 || exit 0
fi
if [[ "${FAM_IN_EVAL_IMAGE:-}" == 1 ]]; then
  exec "$@"
fi
[[ -n "${EVAL_IMAGE:-}" ]] || { echo "in_eval_image.sh: set EVAL_IMAGE (docker/Dockerfile.eval)" >&2; exit 2; }
command -v docker >/dev/null || { echo "in_eval_image.sh: docker is not on PATH" >&2; exit 2; }

uid="$(id -u)"; gid="$(id -g)"
user="$(id -un 2>/dev/null || echo "u${uid}")"
# A passwd entry for the invoking user, written atomically: other tools write files of the same kind.
pw="$(mktemp "/tmp/fam_passwd_${uid}.XXXXXX")"
getent passwd "$uid" > "$pw" 2>/dev/null || echo "${user}:x:${uid}:${gid}::/tmp:/bin/bash" > "$pw"
chmod 644 "$pw"
home="/tmp/fam_home_${uid}"; mkdir -p "$home"
rg="$(getent group render | cut -d: -f3 || true)"; vg="$(getent group video | cut -d: -f3 || true)"

DOCKER=(docker run --rm --init --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined
  --group-add "${vg:-44}" --group-add "${rg:-109}" --network host --ipc host --user "${uid}:${gid}"
  -v "$pw:/etc/passwd:ro" -e "HOME=$home" -e "USER=$user" -e "LOGNAME=$user" -e FAM_IN_EVAL_IMAGE=1
  -e PYTHONUNBUFFERED=1 -v /tmp:/tmp -v "$ROOT:$ROOT" -w "$ROOT"
  --name "fam_in_image_${user//[^A-Za-z0-9_.-]/_}_$$")
MOUNTED=(/tmp:rw "$ROOT:rw")
mount_path() {  # $1=path $2=ro|rw: mount an existing absolute path at its own path, once
  local p m i j
  [[ "$1" == /* && -e "$1" ]] || return 0
  p="$(realpath -ms "$1")"
  # never shadow the image's own system and environment directories
  case "$p" in
    /|/usr|/usr/*|/bin|/bin/*|/sbin|/sbin/*|/lib|/lib/*|/lib64|/lib64/*|/etc|/etc/*|/proc|/proc/*|/sys|/sys/*|/dev|/dev/*) return 0 ;;
    /run|/run/*|/boot|/boot/*|/root|/root/*|/opt|/opt/venv|/opt/venv/*|/opt/LIBERO|/opt/LIBERO/*|/opt/libero_config*|/opt/rocm*) return 0 ;;
  esac
  # The same path can arrive through a checkpoint (ro) and a cache/root variable (rw).
  # Upgrade the existing exact mount rather than emitting a duplicate destination, which Docker rejects.
  for i in "${!MOUNTED[@]}"; do
    m="${MOUNTED[i]}"
    if [[ "$p" == "${m%:*}" ]]; then
      if [[ "$2" == rw && "${m##*:}" == ro ]]; then
        for j in "${!DOCKER[@]}"; do
          if [[ "${DOCKER[j]}" == "$p:$p:ro" ]]; then DOCKER[j]="$p:$p:rw"; break; fi
        done
        MOUNTED[i]="$p:rw"
      fi
      return 0
    fi
  done
  # covered by a mount already made: a writable one, or a read-only one when read-only is enough
  for m in "${MOUNTED[@]}"; do
    [[ ( "$p" == "${m%:*}" || "$p" == "${m%:*}"/* ) && ( "${m##*:}" == rw || "$2" == ro ) ]] && return 0
  done
  DOCKER+=(-v "$p:$p:$2"); MOUNTED+=("$p:$2")
}

FORWARD_RE='^(DIT4DIT_|ROBOCASA_|ROBOSUITE_|LIBERO_|EVAL_|INSTINCT_|HF_|HUGGINGFACE_|TRANSFORMERS_|TORCH|TRITON_|NUMBA_|COSMOS_|CAE_|FASTWAM_)|^(MUJOCO_GL|PYOPENGL_PLATFORM|STEPS|TRIALS|EPISODES|PORT|BASE_PORT|GPU_ID|POLICY_200K|MODEL_PYTHON|LIBERO_PYTHON|MIN_FREE_GIB|TOKENIZERS_PARALLELISM)$'
while IFS= read -r name; do
  [[ "$name" =~ $FORWARD_RE ]] || continue
  [[ "$name" == EVAL_DOCKER_ARGS ]] && continue
  value="${!name}"
  DOCKER+=(-e "$name")
  case "$name" in
    *_PYTHON) ;;  # interpreters inside the image; their environments are mounted through the repository or *_OVERLAY
    *_CKPT)
      if [[ -f "$value" ]]; then
        run_dir="$(dirname "$(dirname "$value")")"
        mount_path "$run_dir" ro
        if [[ -f "$run_dir/config.yaml" ]]; then
          while IFS= read -r p; do
            if [[ -d "$p" ]]; then mount_path "$p" ro; elif [[ -f "$p" ]]; then mount_path "$(dirname "$p")" ro; fi
          done \
            < <(grep -oE '/[^[:space:]"'"'"',]+' "$run_dir/config.yaml" | sort -u)
        fi
      else
        mount_path "$value" ro
      fi ;;
    EVAL_OUTPUT_ROOT)
      [[ "$value" == /* ]] && mkdir -p "$value"
      mount_path "$value" rw ;;
    *)
      IFS=: read -r -a parts <<< "$value"
      for p in "${parts[@]+"${parts[@]}"}"; do
        [[ -f "$p" ]] && p="$(dirname "$p")"
        mount_path "$p" rw
      done ;;
  esac
done < <(compgen -e | sort)
IFS=: read -r -a extra_mounts <<< "${IN_IMAGE_MOUNTS:-}"
for p in "${extra_mounts[@]+"${extra_mounts[@]}"}"; do mount_path "$p" rw; done
# An absolute --output-dir: create it and mount it writable.
args=("$@")
for ((i = 0; i + 1 < ${#args[@]}; i++)); do
  if [[ "${args[i]}" == --output-dir && "${args[i+1]}" == /* ]]; then
    mkdir -p "${args[i+1]}"; mount_path "${args[i+1]}" rw
  fi
done
EXTRA=()
if [[ -n "${EVAL_DOCKER_ARGS:-}" ]]; then read -r -a EXTRA <<< "$EVAL_DOCKER_ARGS"; fi

trap 'rm -f "$pw"' EXIT
rc=0
"${DOCKER[@]}" "${EXTRA[@]+"${EXTRA[@]}"}" "$EVAL_IMAGE" "$@" || rc=$?
exit "$rc"
