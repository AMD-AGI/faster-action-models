# shellcheck shell=bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared by scripts/reproduce_*.sh and scripts/run_all_evals.sh (sourced, not run). Gives each script
#   --help         print the script header (what it runs, the variables it needs, the expected runtime)
#   --dry-run      pass --dry-run to every launch (prints each resolved run; needs no assets) and skip the
#                  latency tools, which have no dry run
#   --smoke        a short end-to-end run with the real checkpoints: pass --smoke to every launch (1 trial per
#                  task) and run each latency tool for one round; results go under <default>-smoke unless
#                  REPRO_DIR is set (a run directory refuses a later --full run into it, see the launchers' --help)
#   --only REGEX   run only the launches whose command line matches REGEX (bash =~), e.g. --only 'arm cae|cae-fastwam'
#                  (REPRO_ONLY sets the same filter)
#   --check        only check the environment, then exit
# and checks every variable the script needs before the first run starts, so a missing one fails in
# seconds instead of after hours of evaluation.
#
# Where each launch runs: the FastWAM launcher and the latency tools run on the host and start their own containers
# of EVAL_IMAGE; the DiT4DiT launchers run inside EVAL_IMAGE (their Python environments live there), through
# scripts/in_eval_image.sh (see `dit4dit` below). So every script runs from the host, with docker.

REPRO_DRY_RUN=0
REPRO_SMOKE=0
REPRO_ONLY="${REPRO_ONLY:-}"

# repro_init SCRIPT_PATH "$@": parse the common flags.
repro_init() {
  local script="$1" check_only=0
  shift
  while (($#)); do
    case "$1" in
      -h|--help)
        # The header comment: every line after the shebang up to the first non-comment line.
        sed -n '2,/^[^#]/{/^#/!d;/Copyright\|SPDX-License/d;s/^# \{0,1\}//;p}' "$script"
        exit 0 ;;
      --dry-run) REPRO_DRY_RUN=1 ;;
      --smoke) REPRO_SMOKE=1 ;;
      --only)
        [[ $# -ge 2 ]] || { echo "$(basename "$script"): --only requires a value" >&2; exit 2; }
        REPRO_ONLY="$2"; shift ;;
      --check) check_only=1 ;;
      *) echo "$(basename "$script"): unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
  done
  REPRO_CHECK_ONLY=$check_only
}

# repro_dir DEFAULT: the results directory, REPRO_DIR when set, else DEFAULT (DEFAULT-smoke with --smoke, so a
# smoke run never lands in the directory of a full run).
repro_dir() {
  if [[ -n "${REPRO_DIR:-}" ]]; then
    echo "$REPRO_DIR"
  elif ((REPRO_SMOKE)); then
    echo "$1-smoke"
  else
    echo "$1"
  fi
}

# repro_require SPEC...: each SPEC is a variable name, or alternatives "A|B" (the first set one counts).
# Names ending in _CKPT, _STATS, _ROOT, _OVERLAY or _BUILD must name an existing path, _SRC a directory, names
# ending in _PYTHON (but MODEL_PYTHON) an executable inside EVAL_IMAGE (checked in one container), and
# *_STATS_TEMPLATE a path for every LIBERO suite. Skipped with --dry-run.
repro_require() {
  local spec name value missing=() s suite image_pythons=() line
  ((REPRO_DRY_RUN)) && return 0
  for spec in "$@"; do
    value="" name=""
    IFS='|' read -r -a alts <<<"$spec"
    for s in "${alts[@]}"; do
      if [[ -n "${!s:-}" ]]; then name="$s" value="${!s}"; break; fi
    done
    if [[ -z "$name" ]]; then
      missing+=("${spec//|/ or } is not set")
      continue
    fi
    case "$name" in
      *_STATS_TEMPLATE)
        for suite in libero_spatial libero_object libero_goal libero_10; do
          [[ -f "${value//\{suite\}/$suite}" ]] || missing+=("$name: no file ${value//\{suite\}/$suite}")
        done ;;
      *_CKPT|*_STATS|*_ROOT|*_OVERLAY|*_BUILD)
        [[ -e "$value" ]] || missing+=("$name: no such path: $value") ;;
      MODEL_PYTHON) ;;  # the interpreter inside EVAL_IMAGE; geak/task/run_local.sh checks it in the container
      *_PYTHON)
        # the DiT4DiT interpreters live inside EVAL_IMAGE: checked there, below
        image_pythons+=("$name=$value") ;;
      *_SRC)
        [[ -d "$value" ]] || missing+=("$name: no such directory: $value") ;;
    esac
  done
  if ((${#image_pythons[@]})) && [[ -n "${EVAL_IMAGE:-}" ]]; then
    local out
    # shellcheck disable=SC2016
    if out="$(bash scripts/in_eval_image.sh sh -c 'for nv; do command -v "${nv#*=}" >/dev/null 2>&1 || echo "${nv%%=*}: not an executable in EVAL_IMAGE: ${nv#*=}"; done' _ "${image_pythons[@]}" 2>&1)"; then
      [[ -n "$out" ]] && while IFS= read -r line; do missing+=("$line"); done <<< "$out"
    else
      missing+=("could not start EVAL_IMAGE ($EVAL_IMAGE) to check ${image_pythons[*]%%=*}: ${out##*$'\n'}")
    fi
  fi
  if ((${#missing[@]})); then
    printf 'environment check failed:\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
    echo "See --help, MODEL_ZOO.md and each launcher's --help." >&2
    exit 2
  fi
  echo "environment check passed"
  if ((REPRO_CHECK_ONLY)); then exit 0; fi
}

# dit4dit CMD...: a DiT4DiT launch (`dit4dit run bash scripts/eval_dit4dit_*.sh ...`): run it inside EVAL_IMAGE
# through scripts/in_eval_image.sh (a dry run stays on the host: it needs no environment). The DiT4DiT LIBERO
# launcher runs one policy server, so it gets the first GPU of EVAL_GPUS; RoboCasa takes the whole list.
dit4dit() {
  local fn="$1"; shift
  local gpus="${EVAL_GPUS:-}"
  [[ "$*" == *eval_dit4dit_libero.sh* ]] && gpus="${gpus%%,*}"
  if ((REPRO_DRY_RUN)); then
    EVAL_GPUS="$gpus" "$fn" "$@"
  else
    EVAL_GPUS="$gpus" "$fn" bash scripts/in_eval_image.sh "$@"
  fi
}

# repro_selected CMD...: true unless --only is set and the command line does not match it.
repro_selected() {
  [[ -z "$REPRO_ONLY" ]] && return 0
  [[ "$*" =~ $REPRO_ONLY ]] && return 0
  echo "skipped (--only ${REPRO_ONLY}): $*"
  return 1
}

# run CMD...: run one launch; with --dry-run, the same launch with --dry-run appended; with --smoke, with --smoke
# appended (the launchers take the last of --smoke/--full).
run() {
  repro_selected "$@" || return 0
  local extra=()
  ((REPRO_SMOKE)) && extra+=(--smoke)
  ((REPRO_DRY_RUN)) && extra+=(--dry-run)
  "$@" "${extra[@]+"${extra[@]}"}"
}

# latency CMD...: run a latency tool; with --dry-run, only print it; with --smoke, for one round. A latency tool
# times on one GPU: the first of EVAL_GPUS.
latency() {
  repro_selected "$@" || return 0
  local extra=()
  ((REPRO_SMOKE)) && extra+=(--rounds 1)
  if ((REPRO_DRY_RUN)); then
    echo "dry run, not timing: $* ${extra[*]+"${extra[*]}"}"
  else
    EVAL_GPUS="${EVAL_GPUS:+${EVAL_GPUS%%,*}}" "$@" "${extra[@]+"${extra[@]}"}"
  fi
}
