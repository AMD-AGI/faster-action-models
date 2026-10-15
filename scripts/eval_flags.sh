# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Helpers shared by the evaluation launchers (and the FastWAM engines under
# eval/), so that a flag means the same thing in every one of them. Sourced,
# not executed.
#
# Flags common to the launchers:
#   --arm NAME        baseline (default) | geak | cae: the source tree / policy
#                     to serve (cae: FastWAM LIBERO, DiT4DiT RoboCasa)
#   --steps N         denoise steps, each model's documented default
#   --sim NAME        mujoco | instinct
#   --gpus CSV        GPU ids; one policy server (or worker) per listed GPU
#   --smoke | --full  1 trial per task | the full protocol
#   --trials N        trials (episodes) per task
#   --suites CSV      LIBERO suites
#   --output-dir DIR  the run directory itself (default: $EVAL_OUTPUT_ROOT)
#   --clients-per-gpu K
#                     concurrent simulator clients per policy server
#   --port PORT       first policy-server port; server i listens on PORT+i
#   --record-video    rollout videos
#   --dry-run         print the resolved run, start nothing
# Environment defaults: STEPS, TRIALS, EVAL_GPUS, EVAL_OUTPUT_ROOT, PORT.
# MUJOCO_GL: unset (the default) renders CPU MuJoCo's cameras with GPU OpenGL
# (egl) when an EGL context can be created in the simulator's environment, else
# with OSMesa (osmesa); egl|osmesa|glfw forces one (eval_mujoco_gl below).
#
# Every refusal exits 2 with one message format: "--flag must be ..., got: X".

EVAL_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_LIBERO_SUITES=(libero_spatial libero_object libero_goal libero_10)

# eval_deprecated OLD NEW: warn that OLD is a deprecated spelling of NEW.
eval_deprecated() {
  echo "warning: $1 is deprecated and will be removed in the next release; use $2" >&2
}

# eval_check_gpus CSV MAX: CSV must be comma-separated GPU ids; MAX > 0 caps
# how many (a launcher that runs one policy server takes exactly one).
eval_check_gpus() {
  local csv="$1" max="$2" ids gpu
  IFS=, read -r -a ids <<< "$csv"
  ((${#ids[@]} > 0)) || { echo "--gpus must name at least one GPU" >&2; exit 2; }
  for gpu in "${ids[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || {
      echo "--gpus must be a comma-separated list of GPU ids, got: $csv" >&2
      exit 2
    }
  done
  if ((max > 0 && ${#ids[@]} > max)); then
    echo "this launcher runs one policy server; pass one GPU id to --gpus, got: $csv" >&2
    exit 2
  fi
}

# eval_ignore_env REASON NAME...: unset each NAME, warning when it was set to a
# value. For the engines' standalone-use knobs, which a launcher run must take
# from its flags and the protocol, never from a stray export.
eval_ignore_env() {
  local reason="$1" name
  shift
  for name in "$@"; do
    if [[ -n "${!name:-}" ]]; then
      echo "warning: ignoring ${name}=${!name}: ${reason}" >&2
    fi
    unset "$name"
  done
}

# eval_warn_override NAME RESOLVED WHAT: warn when NAME is set to a path other
# than RESOLVED, the one WHAT selects. The override is honoured.
eval_warn_override() {
  local name="$1" resolved="$2" what="$3"
  if [[ -n "${!name:-}" && "$(realpath -m "${!name}")" != "$(realpath -m "$resolved")" ]]; then
    echo "warning: ${name}=${!name} overrides ${resolved}, which ${what} selects" >&2
  fi
}

# eval_file_identity PATH: "<absolute path> <bytes> <mtime>", enough to tell
# two checkpoints (or statistics files) apart without hashing gigabytes.
eval_file_identity() {
  local path
  path="$(realpath -m "$1")"
  echo "${path} $(stat -L -c '%s %Y' "$path" 2>/dev/null || echo missing)"
}

# eval_claim_run_dir FILE IDENTITY HAS_RESULTS: a run directory records the
# run it holds (IDENTITY, key=value lines) in FILE. A rerun into it resumes
# only when its identity is the same; HAS_RESULTS=1 (the directory already
# holds results) with a different or missing record is refused, because
# resuming would pool results of another arm, tree, checkpoint, step count or
# trial count under this run's label. A directory without results is (re)claimed.
eval_claim_run_dir() {
  local file="$1" identity="$2" has_results="$3" dir recorded=""
  dir="$(dirname "$file")"
  [[ -f "$file" ]] && recorded="$(<"$file")"
  if [[ "$recorded" == "$identity" ]]; then
    return 0
  fi
  if [[ "$has_results" == 1 ]]; then
    if [[ -n "$recorded" ]]; then
      echo "refusing to resume into ${dir}: it holds results of a different run" >&2
      diff <(printf '%s\n' "$recorded") <(printf '%s\n' "$identity") \
        | sed -n 's/^< /  recorded: /p; s/^> /  this run: /p' >&2
    else
      echo "refusing to resume into ${dir}: it holds results but no record of the run" >&2
      echo "that wrote them ($(basename "$file"))" >&2
    fi
    echo "use a fresh --output-dir" >&2
    return 2
  fi
  printf '%s\n' "$identity" > "$file"
}

# eval_fail LINE...: print each line to stderr and exit 2 (a refused invocation).
eval_fail() {
  printf '%s\n' "$@" >&2
  exit 2
}

# eval_check_choice FLAG VALUE CHOICE...: VALUE must be one of the CHOICEs.
eval_check_choice() {
  local flag="$1" value="$2" choice list=""
  shift 2
  for choice in "$@"; do
    [[ "$value" == "$choice" ]] && return 0
  done
  if (($# == 1)); then
    list="$1"
  else
    list="$(printf '%s, ' "${@:1:$#-1}")"
    list="${list%, } or ${!#}"
  fi
  eval_fail "${flag} must be ${list}, got: ${value}"
}

# eval_check_arm ARM CHOICE...: ARM must be one of the launcher's arms.
eval_check_arm() { eval_check_choice --arm "$@"; }
eval_check_sim() { eval_check_choice --sim "$1" mujoco instinct; }

# eval_check_posint FLAG VALUE
eval_check_posint() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || eval_fail "$1 must be a positive integer, got: $2"
}

# eval_check_port FLAG VALUE [COUNT]: VALUE..VALUE+COUNT-1 must be valid ports.
eval_check_port() {
  local flag="$1" port="$2" count="${3:-1}"
  if [[ ! "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port + count - 1 > 65535)); then
    eval_fail "${flag} must be an integer from 1 to 65535, got: ${port}"
  fi
}

# eval_check_suites CSV: every entry must be a LIBERO suite. Checked right after
# argument parsing, so a typo never waits for a policy server.
eval_check_suites() {
  local csv="$1" suites suite known
  IFS=, read -r -a suites <<< "$csv"
  ((${#suites[@]} > 0)) || eval_fail "--suites must name at least one suite, got: ${csv}"
  for suite in "${suites[@]}"; do
    for known in "${EVAL_LIBERO_SUITES[@]}"; do
      [[ "$suite" == "$known" ]] && continue 2
    done
    eval_fail "--suites must be a comma-separated list of $(IFS=,; echo "${EVAL_LIBERO_SUITES[*]}"), got: ${csv}"
  done
}

# eval_check_fastwam_steps ARM STEPS ALLOW_BASELINE_STEPS: FastWAM's step range
# is open (for the GEAK arm's reduced-step gate); its baseline runs only at its
# 10-step reference point unless the caller opts in.
eval_check_fastwam_steps() {
  local arm="$1" steps="$2" allow="$3"
  eval_check_posint --steps "$steps"
  if [[ "$arm" == baseline && "$steps" != 10 && "$allow" != 1 ]]; then
    eval_fail "baseline with --steps $steps is not a validated cell: step reduction is an" \
      "optimized-side claim and the baseline only needs its 10-step reference point." \
      "Pass --allow-baseline-steps to run it anyway."
  fi
}

# eval_check_dit4dit_steps STEPS: DiT4DiT runs its validated ladder on both arms.
# 10 stays refused: it is a FastWAM convention never validated for DiT4DiT.
eval_check_dit4dit_steps() {
  case "$1" in
    1|2|4) ;;
    *)
      eval_fail "--steps $1 is not a DiT4DiT cell. The advertised step counts are 4 (paper default)" \
        "and 2; see docs/side-by-side-source-trees.md."
      ;;
  esac
}

# eval_unsupported FLAG REASON: refuse a common flag this launcher cannot honour.
eval_unsupported() {
  eval_fail "$1 is not available with this launcher: $2"
}

# eval_require VAR HINT: VAR must be set (and non-empty).
eval_require() {
  [[ -n "${!1:-}" ]] || eval_fail "missing required environment variable $1: set it to $2"
}

# eval_env_alias NEW OLD: when NEW is unset and the deprecated OLD is set, warn
# and use OLD's value for NEW.
eval_env_alias() {
  local new="$1" old="$2"
  if [[ -z "${!new:-}" && -n "${!old:-}" ]]; then
    eval_deprecated "$old" "$new"
    printf -v "$new" '%s' "${!old}"
  fi
}

# eval_resolve_python NAME VALUE: an interpreter path, or a name on PATH.
eval_resolve_python() {
  local name="$1" value="$2"
  [[ -n "$value" ]] || { echo "set ${name} to a Python interpreter" >&2; return 2; }
  if [[ "$value" == */* ]]; then
    [[ -x "$value" ]] || { echo "${name} must name an executable: $value" >&2; return 2; }
    echo "$value"
  else
    command -v "$value" >/dev/null || { echo "${name} is not on PATH: $value" >&2; return 2; }
    command -v "$value"
  fi
}

# eval_steps_label STEPS: "10step"; a non-numeric STEPS (a checkpoint's own
# count) is used as it is.
eval_steps_label() {
  if [[ "$1" =~ ^[0-9]+$ ]]; then echo "${1}step"; else echo "$1"; fi
}

# eval_default_output_dir MODEL BENCH ARM STEPS SIM [MODE]: the one default
# run-directory layout,
#   eval/outputs/<model>-<bench>-<arm>-<N>step-<sim>[-<mode>]-<UTC>
eval_default_output_dir() {
  local name="$1-$2-$3-$(eval_steps_label "$4")-$5"
  [[ -n "${6:-}" ]] && name+="-$6"
  echo "${EVAL_REPO_ROOT}/eval/outputs/${name}-$(date -u +%Y%m%dT%H%M%SZ)"
}

# eval_dry_run KEY=VALUE...: print the resolved run, one key=value per line, and
# exit 0. Called after flag validation and before any environment or asset check.
eval_dry_run() {
  echo "dry run: nothing is started"
  printf '%s\n' "$@"
  exit 0
}

# MuJoCo rendering (CPU MuJoCo, --sim mujoco). The logic is scripts/eval_flags.py's
# resolve_mujoco_gl, so the bash launchers and the Python tools decide alike.
EVAL_MUJOCO_GL_AUTO="auto (EGL, falls back to OSMesa)"

# eval_mujoco_gl_dry_run: the dry-run value, the explicit MUJOCO_GL or the auto
# default. Probes nothing, so a dry run needs no image or interpreter.
eval_mujoco_gl_dry_run() {
  echo "${MUJOCO_GL:-$EVAL_MUJOCO_GL_AUTO}"
}

# eval_mujoco_gl PREFIX...: resolve and export MUJOCO_GL and PYOPENGL_PLATFORM
# for CPU MuJoCo. PREFIX runs Python in the simulator's own environment (a
# `docker run ... IMAGE python3` with the run's devices and groups, or the host
# interpreter, behind `env` for its variables). An explicit MUJOCO_GL is kept
# unchanged and nothing is probed; otherwise PREFIX tries to create an EGL
# context (egl on success, osmesa on any failure, crash or timeout). The choice
# is printed once per run: EVAL_MUJOCO_GL_ANNOUNCED=1 is exported with it, so an
# engine a launcher starts keeps the launcher's choice silently.
eval_mujoco_gl() {
  local out line
  out="$(python3 "${EVAL_REPO_ROOT}/scripts/eval_flags.py" mujoco-gl -- "$@")" || return 2
  while IFS= read -r line; do
    case "$line" in
      MUJOCO_GL=*) export MUJOCO_GL="${line#MUJOCO_GL=}" ;;
      PYOPENGL_PLATFORM=*)
        if [[ -n "${line#PYOPENGL_PLATFORM=}" ]]; then
          export PYOPENGL_PLATFORM="${line#PYOPENGL_PLATFORM=}"
        else
          unset PYOPENGL_PLATFORM
        fi
        ;;
    esac
  done <<< "$out"
  [[ -n "${MUJOCO_GL:-}" ]] || return 2
  export EVAL_MUJOCO_GL_ANNOUNCED=1
}

# eval_mujoco_gl_instinct: --sim instinct. The GPU simulator renders the
# cameras itself; the helper MuJoCo contexts its LIBERO and RoboCasa adapters
# open take OSMesa, as they always have, so egl never reaches an instinct run.
# Exported the way eval_mujoco_gl exports its choice; nothing is printed.
eval_mujoco_gl_instinct() {
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa EVAL_MUJOCO_GL_ANNOUNCED=1
}

# eval_summary ARGS...: write <output-dir>/summary.json and print the closing
# "eval complete: ..." line (scripts/eval_summary.py).
eval_summary() {
  python3 "${EVAL_REPO_ROOT}/scripts/eval_summary.py" "$@"
}
