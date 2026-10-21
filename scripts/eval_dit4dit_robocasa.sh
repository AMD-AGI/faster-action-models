#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=dit4dit_diffusers_overlay.sh
source "${ROOT}/scripts/dit4dit_diffusers_overlay.sh"
# shellcheck source=instinct_sim_env.sh
source "${ROOT}/scripts/instinct_sim_env.sh"
# shellcheck source=eval_flags.sh
source "${ROOT}/scripts/eval_flags.sh"

usage() {
  cat <<'EOF'
Run the DiT4DiT RoboCasa full-24 evaluation: the repository's one RoboCasa
protocol, for every checkpoint and source tree.

Usage:
  bash scripts/eval_dit4dit_robocasa.sh [--smoke | --full] [options]

Options:
  --arm NAME        Which model and policy server run (and which checkpoint
                    DIT4DIT_ROBOCASA_CKPT must name):
                      baseline   (default) the DiT4DiT authors' RoboCasa-GR1
                                 checkpoint on src/dit4dit-baseline, the
                                 upstream DiT4DiT tree.
                      geak       the same checkpoint on src/dit4dit-geak.
                      cae        CAE-DiT4DiT (Compact Action Expert: the
                                 K8 x d48 latent policy; train it with
                                 scripts/train_dit4dit_robocasa_*.sh) on the
                                 vendored DiT4DiT/ tree's server, unchanged.
  --steps N         baseline and geak: action-head denoise steps, 4 (paper
                    default) or 2. Plumbed as in eval_dit4dit_libero.sh (see
                    its --help). The cae server runs with its checkpoint's own
                    step count, so --steps is refused with --arm cae (a STEPS
                    export is ignored there, with a note).
  --smoke           Run 1 episode per task.
  --full            Run 50 episodes per task (default).
  --trials N        Episodes per task: 1, 10 or 50, the counts the full-24
                    summarizer validates.
  --gpus CSV        GPU IDs, one policy server per GPU (default 0,1,2,3,4,5).
  --port PORT       First policy-server port; server i listens on PORT+i
                    (default 8098).
  --clients-per-gpu K
                    Simulator clients per policy server (default 1). Each
                    server's (task, episode) list is cut into K disjoint
                    shares; every episode keeps its seed and episode index, and
                    a task's shares are merged into the same result file one
                    client writes. Recommended: 6 with --sim instinct (each
                    client holds a GPU engine on its server's GPU); on CPU
                    MuJoCo, as many as the host's cores allow.
  --output-dir DIR  Run directory (default: EVAL_OUTPUT_ROOT, else
                    eval/outputs/dit4dit-robocasa-<arm>-<N>step-<sim>-<UTC>;
                    <N>step is checkpoint-steps on cae). A rerun into the same directory
                    skips finished tasks (and finished shares), and is refused
                    when that directory holds results of a different run
                    (arm, server tree, checkpoint, episodes per task, steps or
                    simulator).
  --sim NAME        mujoco (default): RoboCasa on CPU MuJoCo 3.2.6.
                    instinct: physics and camera images GPU-native on AMD Instinct™ GPUs
                    (simulate/, INSTINCT_SIM_BUILD from `simulate/build.sh
                    robocasa`, INSTINCT_SIM_MUJOCO_PATH = mujoco 3.3.2). The
                    clients are unchanged; every client's robocasa envs get the
                    GPU-native backend at import, on the GPU of the policy
                    server it talks to.
  --record-video    Write one rollout video per episode under
                    <output-dir>/videos/<task>/. Off by default: it slows the
                    run and the files are large. Recording does not change
                    what the policy sees.
  --dry-run         Print the resolved arm, server and harness trees,
                    steps, trials, GPUs, ports and output directory, and the
                    server/client episode plan, then exit. Needs no
                    checkpoint or simulator; it reads the task list from the
                    DiT4DiT harness tree (DIT4DIT_HARNESS_ROOT).
  -h, --help        Show this help.

There is no --suites: the full-24 protocol always runs its 24 task families.
The run ends by writing <output-dir>/summary.json (pooled and per-task
successes and episodes) and printing one "eval complete: ..." line; the
full-24 summarizer's own output is kept as full24_summary.json.

Deprecated spellings (warn; removed in the next release): --episodes N (use
--trials), EPISODES (use TRIALS),
DIT4DIT_GPU_LIST (use EVAL_GPUS), BASE_PORT (use PORT), DIT4DIT_ROOT (use
DIT4DIT_HARNESS_ROOT), POLICY_200K (use DIT4DIT_ROBOCASA_CKPT). The former
--no-result-gate flag is accepted with a warning and has no effect.

PROTOCOL
  The full-24 rollout-matching protocol of the vendored DiT4DiT/ tree
  (examples/Robocasa_tabletop/eval_files/): 24 task families x 50 episodes,
  720 max episode steps, 12 action steps per chunk, one sim env per client,
  evaluation seed 7 with an explicit reset of every episode to a SHA-256
  derived seed (schema-2 result files, initial-observation fingerprints),
  policy seed 7, server in bf16. scripts/robocasa_full24.py drives it: it
  imports the vendored harness, starts one policy server per GPU, runs the
  vendored simulation_env.py clients, writes protocol.json and results/, and
  summarizes with the vendored summarize_full24_policy.py. Only the policy
  server differs between arms. On baseline and geak it is started
  through scripts/dit4dit_serve_policy.py with the policy seed in
  DIT4DIT_POLICY_SEED (their server_policy.py has no --seed), the four GEAK_*
  gates pinned to 0 and a per-run diffusers overlay, as in
  eval_dit4dit_libero.sh.

  The DiT4DiT authors' own batch script
  (examples/Robocasa_tabletop/eval_files/batch_eval_args.sh) is left in every
  tree, unchanged; it seeds each env's episodes from one reset sequence
  instead of per episode, and is not a protocol of this repository.

DETERMINISM
  The policy draws its sampling noise from each server's random stream in
  request order, so a rerun replays the same policy calls only when every
  server sees the same requests in the same order: the same arm, simulator,
  episode count, GPU count and clients per GPU. More clients per GPU or another
  GPU count keep every episode's seed and initial state, but not the noise each
  policy call draws. Even then an exact replay needs the pinned environment of
  docs/cae-dit4dit-robocasa.md: the policy's GPU arithmetic is not
  bit-reproducible across GPU models and software versions, so elsewhere a
  rerun can differ in a few episodes.

Where to run it: the ROCm environments DiT4DiT/environments/setup_dit4dit_rocm_envs.sh
builds use the evaluation image's site-packages, so run this launcher inside
that image: bash scripts/in_eval_image.sh bash scripts/eval_dit4dit_robocasa.sh ...
(docs/side-by-side-source-trees.md, "DiT4DiT Python environments on ROCm").

Required environment:
  DIT4DIT_ROBOCASA_CKPT  Policy checkpoint for the selected arm (see --arm).
                         baseline/geak: the DiT4DiT authors' checkpoint,
                         mondo-robotics/dit4dit-model
                         dit4dit_robocasa_gr1/final_model/pytorch_model.pt;
                         cae: the CAE-DiT4DiT checkpoint you train (not
                         released; docs/cae-dit4dit-robocasa.md). The run
                         directory's config.yaml (above final_model/ or
                         checkpoints/) must point
                         framework.cosmos25.base_model at a local copy of the
                         gated nvidia/Cosmos-Predict2.5-2B (accept its licence
                         on Hugging Face and log in to download it).
  ROBOSUITE_OVERLAY      Pinned robosuite 1.5.1 site-packages overlay

Optional environment:
  DIT4DIT_PYTHON         Python for the server and the clients (default: python3)
  DIT4DIT_MODEL_PYTHON   Policy-server Python (default: DIT4DIT_PYTHON)
  ROBOCASA_PYTHON        Simulator-client Python, also runs the driver
                         (default: DIT4DIT_PYTHON); mujoco 3.2.6, numpy 1.26.4,
                         gymnasium >= 1.0
  ROBOCASA_ROOT          RoboCasa-GR1 checkout carrying
                         DiT4DiT/patches/robocasa-gr1-deterministic-reset.patch
                         (DiT4DiT/environments/setup_robocasa_gr1.sh applies it;
                         default: <repo>/DiT4DiT/robocasa-gr1-tabletop-tasks)
  DIT4DIT_HARNESS_ROOT   Tree holding the full-24 harness and client, and with
                         --arm cae the policy server (default: <repo>/DiT4DiT)
  DIT4DIT_PYOVERLAY      Third-party wheel overlay for the policy server
  DIT4DIT_DIFFUSERS_SRC  baseline, geak: diffusers package directory to copy
                         into the overlay (default: the model environment's
                         installed diffusers; required when that environment
                         ships no diffusers)
  STEPS, TRIALS          Defaults for --steps, --trials
  EVAL_GPUS              Default for --gpus
  EVAL_OUTPUT_ROOT       Default for --output-dir
  PORT                   Default for --port (8098)
  NUMBA_OPT              Set to 0 for the clients on AMD Zen 5 hosts unless
                         exported (numba 0.61 segfaults optimizing for them)
EOF
}

eval_env_alias DIT4DIT_HARNESS_ROOT DIT4DIT_ROOT
harness_root="${DIT4DIT_HARNESS_ROOT:-${ROOT}/DiT4DiT}"
arm=baseline
steps="${STEPS:-}"
steps_flag=0
checkpoint="${DIT4DIT_ROBOCASA_CKPT:-}"
python_bin="${DIT4DIT_PYTHON:-python3}"
model_python="${DIT4DIT_MODEL_PYTHON:-$python_bin}"
robocasa_python="${ROBOCASA_PYTHON:-$python_bin}"
overlay="${ROBOSUITE_OVERLAY:-}"
robocasa_root="${ROBOCASA_ROOT:-${harness_root}/robocasa-gr1-tabletop-tasks}"
if [[ -n "${TRIALS:-}" ]]; then
  trials="$TRIALS"
elif [[ -n "${EPISODES:-}" ]]; then
  eval_deprecated EPISODES TRIALS
  trials="$EPISODES"
else
  trials=50
fi
if [[ -n "${EVAL_GPUS:-}" ]]; then
  gpu_list="$EVAL_GPUS"
elif [[ -n "${DIT4DIT_GPU_LIST:-}" ]]; then
  eval_deprecated DIT4DIT_GPU_LIST EVAL_GPUS
  gpu_list="$DIT4DIT_GPU_LIST"
else
  gpu_list=0,1,2,3,4,5
fi
clients_per_gpu=1
if [[ -n "${PORT:-}" ]]; then
  base_port="$PORT"
elif [[ -n "${BASE_PORT:-}" ]]; then
  eval_deprecated BASE_PORT PORT
  base_port="$BASE_PORT"
else
  base_port=8098
fi
output_dir="${EVAL_OUTPUT_ROOT:-}"
record_video=0
dry_run=0
sim=mujoco

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
    --trials|--episodes)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
      [[ "$1" == --episodes ]] && eval_deprecated --episodes --trials
      trials="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || { echo "--port requires a value" >&2; exit 2; }
      base_port="$2"
      shift 2
      ;;
    --arm)
      [[ $# -ge 2 ]] || { echo "--arm requires a value" >&2; exit 2; }
      arm="$2"
      shift 2
      ;;
    --steps)
      [[ $# -ge 2 ]] || { echo "--steps requires a value" >&2; exit 2; }
      steps="$2"
      steps_flag=1
      shift 2
      ;;
    --suites)
      eval_unsupported "$1" "the full-24 protocol always runs its 24 task families"
      ;;
    --mode|--num-envs|--pad-batch)
      eval_unsupported "$1" "the batched N-world mode is FastWAM's (eval/native/run_fastwam_libero.sh)"
      ;;
    --gpus)
      [[ $# -ge 2 ]] || { echo "--gpus requires a value" >&2; exit 2; }
      gpu_list="$2"
      shift 2
      ;;
    --clients-per-gpu)
      [[ $# -ge 2 ]] || { echo "--clients-per-gpu requires a value" >&2; exit 2; }
      clients_per_gpu="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
      output_dir="$2"
      shift 2
      ;;
    --no-result-gate)
      echo "warning: --no-result-gate is deprecated and has no effect; it will be removed in the next release" >&2
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

eval_check_arm "$arm" baseline geak cae
eval_check_sim "$sim"
if [[ "$arm" == cae ]]; then
  if ((steps_flag)); then
    eval_fail "--steps is not available with --arm cae: its server runs unchanged with the" \
      "checkpoint's own step count. --steps applies to --arm baseline and geak."
  fi
  if [[ -n "$steps" ]]; then
    echo "note: ignoring STEPS=${steps}: the CAE-DiT4DiT server runs with its checkpoint's own step count" >&2
  fi
  steps=checkpoint-steps
  server_root="$harness_root"
  if [[ -z "$checkpoint" && -n "${POLICY_200K:-}" ]]; then
    eval_deprecated POLICY_200K DIT4DIT_ROBOCASA_CKPT
    checkpoint="$POLICY_200K"
  fi
  label_arm=cae
else
  steps="${steps:-4}"
  eval_check_dit4dit_steps "$steps"
  server_root="${ROOT}/src/dit4dit-${arm}"
  label_arm="$arm"
fi
case "$trials" in
  1|10|50) ;;
  *) eval_fail "--trials (episodes per task) must be one of 1, 10, or 50, got: $trials" ;;
esac
eval_check_posint --clients-per-gpu "$clients_per_gpu"
eval_check_gpus "$gpu_list" 0
IFS=, read -r -a gpus <<< "$gpu_list"
eval_check_port --port "$base_port" "${#gpus[@]}"

output_dir="${output_dir:-$(eval_default_output_dir dit4dit robocasa "$arm" "$steps" "$sim")}"
output_dir="$(realpath -m "$output_dir")"

# The dry run reads the harness's task list, so check the harness tree first.
[[ -f "$harness_root/DiT4DiT/dataloader/gr00t_lerobot/full24_tasks.py" ]] ||
  eval_fail "DIT4DIT_HARNESS_ROOT must name the DiT4DiT checkout (the full-24 harness), got: $harness_root"

if [[ "$dry_run" -eq 1 ]]; then
  ports=()
  for i in "${!gpus[@]}"; do ports+=("$((base_port + i))"); done
  echo "dry run: nothing is started"
  printf '%s\n' "model=dit4dit" "benchmark=robocasa" "arm=${arm}" \
    "tree=$(realpath -m "$server_root")" "harness=$(realpath -m "$harness_root")" \
    "steps=${steps}" "trials=${trials}" "gpus=${gpu_list}" "port=$(IFS=,; echo "${ports[*]}")" \
    "clients_per_gpu=${clients_per_gpu}" "sim=${sim}" \
    "mujoco_gl=$([[ "$sim" == instinct ]] && echo "osmesa (--sim instinct helper contexts)" \
      || eval_mujoco_gl_dry_run)" \
    "record_video=${record_video}" "output=${output_dir}"
  exec python3 "${ROOT}/scripts/robocasa_full24.py" run --dry-run \
    --dit4dit-root "$harness_root" --condition direct --episodes "$trials" \
    --gpu-list "$gpu_list" --base-port "$base_port" --clients-per-gpu "$clients_per_gpu"
fi

# checkpoint may also come from the deprecated POLICY_200K (--arm cae).
[[ -n "$checkpoint" ]] || eval_fail "missing required environment variable DIT4DIT_ROBOCASA_CKPT: set it to" \
  "the policy checkpoint for --arm ${arm}"
eval_require ROBOSUITE_OVERLAY "the pinned site-packages overlay"
[[ -f "$checkpoint" ]] || {
  echo "DIT4DIT_ROBOCASA_CKPT must name the policy checkpoint for --arm ${arm}: $checkpoint" >&2
  exit 2
}
[[ -d "$overlay" ]] || {
  echo "ROBOSUITE_OVERLAY must name the pinned site-packages overlay: $overlay" >&2
  exit 2
}
[[ -d "$robocasa_root" ]] || {
  echo "ROBOCASA_ROOT must name the RoboCasa-GR1 checkout: $robocasa_root" >&2
  exit 2
}
grep -q 'self.env.rng = np.random.default_rng(seed)' \
  "${robocasa_root}/robocasa/utils/gym_utils/gymnasium_basic.py" 2>/dev/null || {
  echo "ROBOCASA_ROOT lacks DiT4DiT/patches/robocasa-gr1-deterministic-reset.patch, without which an" >&2
  echo "episode's scene does not follow its seed; apply it (DiT4DiT/environments/setup_robocasa_gr1.sh): $robocasa_root" >&2
  exit 2
}
[[ -d "$harness_root" ]] || {
  echo "DIT4DIT_HARNESS_ROOT must name the DiT4DiT checkout: $harness_root" >&2
  exit 2
}
[[ -d "$server_root" ]] || {
  echo "policy-server tree not found: $server_root" >&2
  exit 2
}
model_python="$(eval_resolve_python DIT4DIT_MODEL_PYTHON "$model_python")" || exit 2
robocasa_python="$(eval_resolve_python ROBOCASA_PYTHON "$robocasa_python")" || exit 2

checkpoint="$(realpath "$checkpoint")"
overlay="$(realpath "$overlay")"
robocasa_root="$(realpath "$robocasa_root")"
harness_root="$(realpath "$harness_root")"
server_root="$(realpath "$server_root")"

# The clients rely on gymnasium 1.x vector-env autoreset timing; under 0.29 a
# client crashes on its final reset. Refuse rather than run it.
PYTHONPATH="${overlay}:${robocasa_root}:${PYTHONPATH:-}" "$robocasa_python" - <<'PY' || exit 2
import sys

import gymnasium

if int(gymnasium.__version__.split(".")[0]) < 1:
    sys.exit(f"the RoboCasa client needs gymnasium >= 1.0, found "
             f"{gymnasium.__version__} at {gymnasium.__file__}")
PY

if [[ "$sim" == instinct ]]; then
  instinct_sim_check robocasa "$robocasa_python" || exit 2
fi

declare -a server=(--server-python "$model_python" --server-root "$server_root")
if [[ "$arm" == cae ]]; then
  # The vendored server, unchanged, with --seed: run_full24_evaluation.py's own server command.
  server+=(--server-seed flag --server-pythonpath "${DIT4DIT_PYOVERLAY:-}")
else
  mkdir -p "$output_dir"
  # A per-run copy of diffusers 3996788, patched on the geak arm only. Sets diffusers_overlay.
  build_diffusers_overlay "$model_python" "$arm" "${output_dir}/diffusers-overlay" || exit $?
  # GEAK levers pinned off on every arm, and the step count asserted at load; see eval_dit4dit_libero.sh --help.
  server+=(
    --server-script "${ROOT}/scripts/dit4dit_serve_policy.py"
    --server-seed env
    --server-pythonpath "${diffusers_overlay}${DIT4DIT_PYOVERLAY:+:${DIT4DIT_PYOVERLAY}}"
    --server-env GEAK_ACTION_COMPILE=0
    --server-env GEAK_COSMOS_COMPILE=0
    --server-env GEAK_VAE_T1=0
    --server-env GEAK_ACTION_BF16=0
    --server-env "DIT4DIT_INFER_STEPS=${steps}"
    --server-env TOKENIZERS_PARALLELISM=false
  )
fi

declare -a evaluation=(
  "$robocasa_python" "${ROOT}/scripts/robocasa_full24.py" run
  --dit4dit-root "$harness_root"
  --checkpoint "$checkpoint"
  --condition direct
  --episodes "$trials"
  --gpu-list "$gpu_list"
  --base-port "$base_port"
  --policy-seed 7
  --eval-seed 7
  --max-episode-steps 720
  --n-action-steps 12
  --python-bin "$robocasa_python"
  --robosuite-overlay "$overlay"
  --robocasa-root "$robocasa_root"
  --output-dir "$output_dir"
  --clients-per-gpu "$clients_per_gpu"
  --server-timeout-seconds 900
  --label "arm=${label_arm} steps=${steps/checkpoint-steps/checkpoint} sim=${sim}"
  "${server[@]}"
)
if [[ "$record_video" -eq 1 ]]; then
  evaluation+=(--record-video)
fi

# numba 0.61 / llvmlite 0.44 (pinned by robocasa-gr1) do not know AMD Zen 5 CPUs (family 26, the EPYC hosts of
# MI350X/MI355X systems): optimizing robosuite's @njit transforms segfaults every RoboCasa client, on both
# simulators. NUMBA_OPT=0 compiles them unoptimized; an exported NUMBA_OPT is kept.
if [[ -z "${NUMBA_OPT:-}" ]] && grep -q '^vendor_id[[:space:]]*: AuthenticAMD' /proc/cpuinfo 2>/dev/null \
  && grep -qE '^cpu family[[:space:]]*: 26$' /proc/cpuinfo 2>/dev/null; then
  echo "AMD Zen 5 host CPU: setting NUMBA_OPT=0 for the RoboCasa clients (numba 0.61 cannot optimize for it)"
  export NUMBA_OPT=0
fi

if [[ "$sim" == instinct ]]; then
  # The driver starts each client as a subprocess with this environment. instinct_sim_robocasa's sitecustomize
  # attaches the backend when a process imports robocasa (the policy servers never do) and puts a client's engine
  # on its server's GPU (HP_PORT_GPU: server port -> GPU, one server per GPU from --port up).
  instinct_sim_client_env robocasa "$robocasa_python"
  # The GPU simulator renders the cameras; the clients' helper MuJoCo contexts stay on OSMesa.
  eval_mujoco_gl_instinct
  export PYTHONPATH="${ROOT}/simulate/python/instinct_sim_robocasa/sitecustomize:${PYTHONPATH}"
  export HP_ROBOCASA_BACKEND=hip
  port_gpu=()
  for i in "${!gpus[@]}"; do port_gpu+=("$((base_port + i)):${gpus[$i]}"); done
  HP_PORT_GPU="$(IFS=,; echo "${port_gpu[*]}")"
  export HP_PORT_GPU
fi

# Under --sim mujoco the driver picks the clients' MUJOCO_GL once (an exported one is kept,
# else EGL when the client interpreter can create a context, else OSMesa) and prints it.
# Run from the harness tree, as run_full24_evaluation.py is run, so every process inherits the same environment.
cd "$harness_root"
"${evaluation[@]}"

declare -a summary=(
  "$robocasa_python" "${ROOT}/scripts/robocasa_full24.py" summarize "$output_dir"
  --dit4dit-root "$harness_root"
  --episodes-per-task "$trials"
  --sim "$sim"
)
"${summary[@]}" > "${output_dir}/full24_summary.json"
cat "${output_dir}/full24_summary.json"
eval_summary --model dit4dit --bench robocasa --arm "$arm" --steps "$steps" --sim "$sim" \
  --trials "$trials" --output-dir "$output_dir" --robocasa "${output_dir}/full24_summary.json"
