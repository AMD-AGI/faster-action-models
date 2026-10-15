#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../../scripts/eval_flags.sh
source "$ROOT/scripts/eval_flags.sh"

usage() {
  cat <<'EOF'
FastWAM LIBERO evaluation with FastWAM's own evaluator, no harness; LIBERO on CPU MuJoCo or GPU-native (--sim instinct).

Usage:
  bash eval/native/run_fastwam_libero.sh [options]

Options:
  --sim NAME      mujoco (default): LIBERO on CPU MuJoCo.
                  instinct: LIBERO's physics and camera images GPU-native on AMD Instinct™ GPUs (simulate/, INSTINCT_SIM_BUILD from
                    `simulate/build.sh libero`); LIBERO's OffScreenRenderEnv is swapped for the GPU one at
                    import (instinct_sim_libero). See simulate/README.md.
  --mode NAME     serial (default): FastWAM's evaluator (FastWAM/experiments/libero/eval_libero_single.py,
                    persistent worker per GPU), unchanged.
                  batched: eval/native/fastwam_libero_batched.py -- the same episode loop for all trials of a
                    task in lockstep on one N-world engine, one model call per replan. Needs --sim instinct;
                    --arm baseline and geak only.
  --arm NAME      baseline (default), geak or cae (CAE-FastWAM: Compact Action Expert, the 32-hidden-dimension
                    Action DiT with the K8 x d16 action VAE, GEAK-optimized; one suite at a time).
  --steps N       Denoise steps (default 10). Advertised: baseline 10, geak 10 and 2, cae 10 and 4. Baseline at
                    other than 10 needs --allow-baseline-steps, as in scripts/eval_fastwam_libero.sh.
  --smoke         1 trial per task.   --full  50 trials per task (default).   --trials N  trials per task.
  --suites CSV    Default: libero_spatial,libero_object,libero_goal,libero_10.
  --gpus CSV      GPUs (default 0,1,2,3): one worker per GPU pulling (suite, task) items. There is no protocol behind
                    this default (the protocol fixes the episodes, not the workers); it only spreads the 40 tasks.
  --output-dir DIR  Run directory (default: EVAL_OUTPUT_ROOT, else
                    eval/outputs/fastwam-libero-<arm>-<N>step-<sim>-<mode>-<UTC>). A rerun into it skips
                    finished tasks, and is refused when it holds results of a different cell.
  --num-envs N    batched: worlds per task (default 50).   --pad-batch N  batched: model batch (default 64).
  --record-video  Accepted for a common command line. Serial: a no-op, FastWAM's evaluator always writes one video
                    per episode under <output-dir>/<suite>/videos/. Batched writes no videos and refuses it.
  --dry-run       Print the resolved arm, tree, configs, steps, trials, suites, GPUs, mode, simulator and
                    output directory, then exit. Needs no checkpoint or container.
  -h, --help

There is no --clients-per-gpu or --port: each GPU runs one worker that steps its own env, with no policy server.
The run ends by writing <output-dir>/summary.json (pooled and per-suite successes and episodes) and printing one
"eval complete: ..." line; FastWAM's own summary is kept as summary.txt and fastwam_summary.json.

Source trees: baseline src/fastwam-baseline, geak src/fastwam-geak, cae src/cae-fastwam. Configs:
FastWAM/configs_baseline (baseline, geak), FastWAM/configs (cae). Protocol (seed 42, num_steps_wait 30, replan 10, max_steps 400/400/400/700,
binarized gripper) is FastWAM's own sim_libero.yaml + eval_libero_single.py; see eval/PROTOCOL.md.

Required environment:
  MODEL_CKPT, LIBERO_ROOT, EVAL_IMAGE; --sim instinct: INSTINCT_SIM_BUILD (a `simulate/build.sh libero` build dir)
  baseline, geak: DATASET_STATS     cae: TASK_CONFIG (base name), DATASET_STATS_TEMPLATE ({suite}), ACTION_VAE_CKPT
Optional: STEPS, TRIALS (defaults for --steps, --trials), EVAL_GPUS (default for --gpus), EVAL_OUTPUT_ROOT (default
  for --output-dir), EVAL_PYOVERLAY (extra PYTHONPATH entries, e.g. FastWAM eval deps the image lacks),
  LIBERO_CONFIG_PATH, HF_HOME, DIFFSYNTH_MODEL_BASE_PATH, DIFFSYNTH_DOWNLOAD_SOURCE (forwarded when set).
EOF
}

SIM=mujoco MODE=serial ARM=baseline STEPS="${STEPS:-}" TRIALS="${TRIALS:-50}" NUM_ENVS=50 PAD_BATCH=64
ALLOW_BASELINE_STEPS=0 RECORD_VIDEO=0 DRY_RUN=0
SUITES_CSV=libero_spatial,libero_object,libero_goal,libero_10 GPUS_CSV="${EVAL_GPUS:-0,1,2,3}"
OUTPUT_DIR="${EVAL_OUTPUT_ROOT:-}"
while [ $# -gt 0 ]; do
  case "$1" in --sim|--mode|--arm|--steps|--trials|--suites|--gpus|--output-dir|--num-envs|--pad-batch)
    [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; } ;;
  esac
  case "$1" in
  --sim) SIM="$2"; shift 2 ;;
  --mode) MODE="$2"; shift 2 ;;
  --arm) ARM="$2"; shift 2 ;;
  --steps) STEPS="$2"; shift 2 ;;
  --trials) TRIALS="$2"; shift 2 ;;
  --smoke) TRIALS=1; shift ;;
  --full) TRIALS=50; shift ;;
  --allow-baseline-steps) ALLOW_BASELINE_STEPS=1; shift ;;
  --record-video) RECORD_VIDEO=1; shift ;;
  --dry-run) DRY_RUN=1; shift ;;
  --suites) SUITES_CSV="$2"; shift 2 ;;
  --gpus) GPUS_CSV="$2"; shift 2 ;;
  --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
  --clients-per-gpu|--shards|--parallel-tasks|--port)
    eval_unsupported "$1" "it runs no policy server; each GPU in --gpus runs one worker (serial) or one N-world engine (batched, --num-envs)" ;;
  --num-envs) NUM_ENVS="$2"; shift 2 ;;
  --pad-batch) PAD_BATCH="$2"; shift 2 ;;
  -h|--help) usage; exit 0 ;;
  *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac; done

eval_check_arm "$ARM" baseline geak cae
eval_check_sim "$SIM"
eval_check_choice --mode "$MODE" serial batched
case "$ARM" in
  baseline) SRC=src/fastwam-baseline ;;
  geak) SRC=src/fastwam-geak ;;
  cae) SRC=src/cae-fastwam ;;
esac
STEPS="${STEPS:-10}"
eval_check_fastwam_steps "$ARM" "$STEPS" "$ALLOW_BASELINE_STEPS"
eval_check_posint --trials "$TRIALS"
eval_check_posint --num-envs "$NUM_ENVS"
eval_check_posint --pad-batch "$PAD_BATCH"
eval_check_gpus "$GPUS_CSV" 0
eval_check_suites "$SUITES_CSV"
[[ "$MODE" == batched && "$SIM" != instinct ]] && \
  eval_fail "--mode batched runs N worlds on one GPU engine; it needs --sim instinct"
[[ "$MODE" == batched && "$ARM" == cae ]] && eval_fail "--mode batched supports --arm baseline and geak only"
[[ "$MODE" == batched && "$RECORD_VIDEO" == 1 ]] && \
  eval_unsupported --record-video "--mode batched writes no rollout videos; use --mode serial, which always writes them"
if [[ "$ARM" != cae ]]; then
  CONFIGS="$ROOT/FastWAM/configs_baseline"; TASK=libero_uncond_2cam224_1e-4
else
  CONFIGS="$ROOT/FastWAM/configs"
fi
OUT="${OUTPUT_DIR:-$(eval_default_output_dir fastwam libero "$ARM" "$STEPS" "$SIM" "$MODE")}"
if ((DRY_RUN)); then
  dry=("model=fastwam" "benchmark=libero" "arm=$ARM" "tree=$ROOT/$SRC" "configs=$CONFIGS"
    "steps=$STEPS" "trials=$TRIALS" "suites=$SUITES_CSV" "gpus=$GPUS_CSV" "port=none (no policy server)"
    "mode=$MODE" "sim=$SIM")
  # --sim instinct renders cameras on the GPU simulator; its helper MuJoCo contexts stay on OSMesa.
  if [[ "$SIM" == instinct ]]; then dry+=("mujoco_gl=osmesa (--sim instinct helper contexts)")
  else dry+=("mujoco_gl=$(eval_mujoco_gl_dry_run)"); fi
  [[ "$MODE" == batched ]] && dry+=("num_envs=$NUM_ENVS" "pad_batch=$PAD_BATCH")
  eval_dry_run "${dry[@]}" "record_video=$([[ "$MODE" == serial ]] && echo "1 (always)" || echo 0)" \
    "output=$(realpath -m "$OUT")"
fi
eval_require MODEL_CKPT "the FastWAM checkpoint"
eval_require LIBERO_ROOT "the LIBERO checkout"
eval_require EVAL_IMAGE "the evaluation container image"
if [[ "$ARM" != cae ]]; then
  eval_require DATASET_STATS "its normalization statistics (baseline, geak)"
else
  eval_require TASK_CONFIG "the CAE-FastWAM task config base name (--arm cae)"
  eval_require DATASET_STATS_TEMPLATE "the per-suite statistics path, with the token {suite} (--arm cae)"
  eval_require ACTION_VAE_CKPT "the action-VAE checkpoint (--arm cae)"
fi
if [[ "$SIM" == instinct ]]; then
  [[ -n "${INSTINCT_SIM_BUILD:-}" ]] || { echo "--sim instinct needs INSTINCT_SIM_BUILD (run simulate/build.sh libero)" >&2; exit 2; }
  ls "$INSTINCT_SIM_BUILD"/python/instinct_sim*.so >/dev/null 2>&1 || {
    echo "INSTINCT_SIM_BUILD has no python/instinct_sim*.so: $INSTINCT_SIM_BUILD (run simulate/build.sh libero)" >&2; exit 2; }
fi
IFS=, read -r -a SUITES <<< "$SUITES_CSV"; IFS=, read -r -a GPUS <<< "$GPUS_CSV"
mkdir -p "$OUT/logs"; OUT="$(cd "$OUT" && pwd)"
TAG="${ARM}_${STEPS}_${MODE}_${SIM}"

# Resume guard: a task whose result file exists is skipped and the summary
# pools every result file under $OUT, so resuming a directory written by
# another cell would silently mix the two. $OUT records the run it holds.
identity=("arm=$ARM" "tree=$ROOT/$SRC" "steps=$STEPS" "trials=$TRIALS" "mode=$MODE"
  "sim=$SIM" "configs=$CONFIGS" "checkpoint=$(eval_file_identity "$MODEL_CKPT")")
if [[ "$ARM" != cae ]]; then
  identity+=("task=$TASK" "dataset_stats=$(eval_file_identity "$DATASET_STATS")")
else
  identity+=("task_config=$TASK_CONFIG" "dataset_stats_template=$DATASET_STATS_TEMPLATE"
    "action_vae=$(eval_file_identity "$ACTION_VAE_CKPT")")
fi
[[ "$SIM" == instinct ]] && identity+=("instinct_sim_build=$(realpath -m "$INSTINCT_SIM_BUILD")")
# batched result files are per GPU label, so another GPU list would replay finished tasks
[[ "$MODE" == batched ]] && identity+=("num_envs=$NUM_ENVS" "pad_batch=$PAD_BATCH" "gpus=$GPUS_CSV")
has_results=0
compgen -G "$OUT/*/gpu*_task*_results.json" >/dev/null && has_results=1
eval_claim_run_dir "$OUT/.run_identity" "$(printf '%s\n' "${identity[@]}")" "$has_results" || exit 2

PW="/tmp/ctr_passwd_$(id -u)"; getent passwd "$(id -u)" > "$PW" || true
RG="$(getent group render | cut -d: -f3 || true)"; VG="$(getent group video | cut -d: -f3 || true)"
HP="$ROOT/simulate/python"
DOCKER=(docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined
  --group-add "${VG:-44}" --group-add "${RG:-109}" --network host --user "$(id -u):$(id -g)" --ipc host
  -e HOME=/tmp -v "$PW:/etc/passwd:ro" -v /mnt:/mnt -v /tmp:/tmp -v "$ROOT:$ROOT"
  -e PYTHONUNBUFFERED=1 -e PYTHONNOUSERSITE=1
  -e "LD_LIBRARY_PATH=${INSTINCT_SIM_BUILD:+$INSTINCT_SIM_BUILD:}/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib"
  # several engine/model processes share a GPU
  -e HSA_ENABLE_SDMA=0 -e GPU_MAX_HW_QUEUES=2)
if [[ "$SIM" == instinct ]]; then
  # The GPU simulator renders the cameras itself; its helper MuJoCo contexts
  # (instinct_sim_libero) stay on OSMesa, whatever MUJOCO_GL says.
  eval_mujoco_gl_instinct
else
  # CPU MuJoCo: GPU OpenGL (EGL) when the image can create an EGL context with
  # these devices, else OSMesa; an exported MUJOCO_GL is kept. Probed once.
  eval_mujoco_gl "${DOCKER[@]}" -e "HIP_VISIBLE_DEVICES=${GPUS[0]}" "$EVAL_IMAGE" /opt/venv/bin/python3 || exit 2
fi
DOCKER+=(-e MUJOCO_GL -e PYOPENGL_PLATFORM)
for v in LIBERO_CONFIG_PATH HF_HOME DIFFSYNTH_MODEL_BASE_PATH DIFFSYNTH_DOWNLOAD_SOURCE ACTION_VAE_CKPT; do
  [[ -n "${!v:-}" ]] && DOCKER+=(-e "$v")
done
ARMENV=()
case "$ARM" in
  baseline) ARMENV=(-e FASTWAM_MOT_COMPILE_MODE=off -e FASTWAM_VAE_COMPILE=0 -e FASTWAM_COMPILE=0) ;;
  geak) ARMENV=(-e FASTWAM_KVSLAB=0) ;;     # KVSLAB=1 is the known aliasing bug
esac

# Work items: (suite, task_id) for every task of the selected suites (10 per LIBERO suite).
ITEMS=(); for S in "${SUITES[@]}"; do for t in $(seq 0 9); do ITEMS+=("$S,$t"); done; done
echo "[$TAG] tree=$SRC steps=$STEPS trials=$TRIALS gpus=${GPUS[*]} items=${#ITEMS[@]} out=$OUT"

worker() {  # $1=gpu $2=suite-or-"" (cae runs one suite at a time) $3..=items (batched only)
  local G="$1" S="$2"; shift 2
  local cache="$OUT/.compile_cache_gpu$G"; mkdir -p "$cache/inductor" "$cache/triton"
  local PP="$ROOT/$SRC:$LIBERO_ROOT${EVAL_PYOVERLAY:+:$EVAL_PYOVERLAY}" sim_env=()
  if [[ "$SIM" == instinct ]]; then
    # sitecustomize swaps LIBERO's OffScreenRenderEnv for instinct_sim_libero's in every worker process
    PP="$HP/instinct_sim_libero/sitecustomize:$ROOT/$SRC:$HP:$INSTINCT_SIM_BUILD/python:$LIBERO_ROOT${EVAL_PYOVERLAY:+:$EVAL_PYOVERLAY}"
    sim_env=(-e HP_LIBERO_BACKEND=hip)
  fi
  local -a run=("${DOCKER[@]}" "${ARMENV[@]+"${ARMENV[@]}"}" --name "native_${TAG}_g$G${S:+_$S}"
    -e "HIP_VISIBLE_DEVICES=$G" -e "TORCHINDUCTOR_CACHE_DIR=$cache/inductor" -e "TRITON_CACHE_DIR=$cache/triton")
  if [[ "$MODE" == batched ]]; then
    local specs=() it; for it in "$@"; do specs+=("${it/,/:}"); done
    local csv; csv="$(IFS=,; echo "${specs[*]}")"
    "${run[@]}" -e "PYTHONPATH=$PP" -e "HP_GPU_LABEL=$G" -w "$ROOT" "$EVAL_IMAGE" \
      /opt/venv/bin/python3 eval/native/fastwam_libero_batched.py --config-dir "$CONFIGS" --task "$TASK" \
      --ckpt "$MODEL_CKPT" --dataset-stats "$DATASET_STATS" --steps "$STEPS" --tasks "$csv" --output-dir "$OUT" \
      --trials "$TRIALS" --num-envs "$NUM_ENVS" --pad-batch "$PAD_BATCH" \
      --override model.redirect_common_files=false
    return
  fi
  local task="${TASK:-}" stats="${DATASET_STATS:-}" extra=() pool="$OUT/.pool"
  if [[ "$ARM" == cae ]]; then
    task="${TASK_CONFIG}_$S"; stats="${DATASET_STATS_TEMPLATE//\{suite\}/$S}"; pool="$OUT/.pool_$S"
    extra=("EVALUATION.task_suite_name=$S")
  fi
  "${run[@]}" -e "PYTHONPATH=$PP" "${sim_env[@]+"${sim_env[@]}"}" \
    -e LIBERO_WORKER_MODE=1 -e "LIBERO_WORKER_ID=$G" -e "LIBERO_WORKER_PENDING_FILE=$pool/pending.txt" \
    -e "LIBERO_WORKER_LOCK_FILE=$pool/pending.lock" -e "LIBERO_WORKER_STOP_FILE=$pool/stop" \
    -e "LIBERO_WORKER_STATUS_DIR=$pool/status" -e "LIBERO_WORKER_FAILED_FILE=$pool/failed.txt" \
    -w "$ROOT/FastWAM" "$EVAL_IMAGE" \
    /opt/venv/bin/python3 experiments/libero/eval_libero_single.py --config-path "$CONFIGS" --config-name sim_libero \
    "task=$task" "ckpt=$MODEL_CKPT" "gpu_id=$G" seed=42 "EVALUATION.num_trials=$TRIALS" \
    "EVALUATION.output_dir=$OUT" "EVALUATION.dataset_stats_path=$stats" "EVALUATION.num_inference_steps=$STEPS" \
    model.redirect_common_files=false "${extra[@]+"${extra[@]}"}"
}

# Serial workers pull items from FastWAM's own worker-pool queue (experiments/libero/worker_pool.py) and exit
# when it is empty and the stop file exists; batched workers get a fixed round-robin share of the items.
run_pool() {  # $1=suite-or-"" $2..=items
  local S="$1"; shift
  local pool="$OUT/.pool${S:+_$S}"; mkdir -p "$pool/status"; : > "$pool/failed.txt"; rm -f "$pool/stop"
  printf '%s\n' "$@" > "$pool/pending.txt"
  local pids=() rc=0 i=0 retried=0
  for G in "${GPUS[@]}"; do
    if [[ "$MODE" == batched ]]; then
      local mine=() j=0; for it in "$@"; do (( j % ${#GPUS[@]} == i )) && mine+=("$it"); j=$((j+1)); done
      (( ${#mine[@]} )) && { worker "$G" "$S" "${mine[@]}" > "$OUT/logs/worker_g$G${S:+_$S}.log" 2>&1 & pids+=($!); }
    else
      worker "$G" "$S" > "$OUT/logs/worker_g$G${S:+_$S}.log" 2>&1 & pids+=($!)
    fi
    i=$((i+1))
  done
  if [[ "$MODE" == serial ]]; then   # workers idle on an empty queue until told to stop
    while :; do
      sleep 20
      local left; left=$(grep -c . "$pool/pending.txt" 2>/dev/null || true)
      local busy; busy=$(grep -l '"running"' "$pool"/status/*.json 2>/dev/null | wc -l || true)
      if [[ -s "$pool/failed.txt" ]]; then
        # torch.compile can intermittently fail a worker's first model call (an InductorError on a cold cache);
        # a second attempt usually succeeds. Requeue the failed items and restart the dead workers once.
        local dead=() k
        for k in "${!pids[@]}"; do kill -0 "${pids[$k]}" 2>/dev/null || dead+=("$k"); done
        (( ${#dead[@]} )) || continue          # the failing worker is still exiting; look again next tick
        echo "[$TAG] worker failure:"; cat "$pool/failed.txt"
        (( retried )) && { touch "$pool/stop"; rc=1; break; }
        retried=1; echo "[$TAG] requeueing the failed items and restarting ${#dead[@]} worker(s) once"
        # same mkdir lock as worker_pool.task_queue_lock, so a live worker's pop is not lost
        until mkdir "$pool/pending.lock.lockdir" 2>/dev/null; do sleep 0.05; done
        cut -d, -f2,3 "$pool/failed.txt" >> "$pool/pending.txt"; : > "$pool/failed.txt"
        rmdir "$pool/pending.lock.lockdir"
        for k in "${dead[@]}"; do
          wait "${pids[$k]}" 2>/dev/null || true; G="${GPUS[$k]}"
          worker "$G" "$S" >> "$OUT/logs/worker_g$G${S:+_$S}.log" 2>&1 & pids[$k]=$!
        done
        continue
      fi
      [[ "${left:-0}" == 0 && "${busy:-0}" == 0 ]] && { touch "$pool/stop"; break; }
      local alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
      [[ $alive == 0 ]] && break
    done
  fi
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}

rc=0
if [[ "$ARM" == cae && "$MODE" == serial ]]; then
  for S in "${SUITES[@]}"; do        # per-suite model: task config, action-VAE suite id and stats differ
    sitems=(); for it in "${ITEMS[@]}"; do [[ "${it%%,*}" == "$S" ]] && sitems+=("$it"); done
    run_pool "$S" "${sitems[@]}" || rc=1
  done
else
  run_pool "" "${ITEMS[@]}" || rc=1
fi
"${DOCKER[@]}" -e "PYTHONPATH=$ROOT/FastWAM" -w "$ROOT/FastWAM" "$EVAL_IMAGE" /opt/venv/bin/python3 \
  experiments/libero/summarize_results.py --output_dir "$OUT" > "$OUT/summary.txt" 2>&1 || rc=1
cat "$OUT/summary.txt"
# FastWAM's summarizer writes its own summary.json; keep it under another name
# so <output-dir>/summary.json is the common one every launcher writes.
if [[ -f "$OUT/summary.json" ]] && ! grep -q '"schema_version"' "$OUT/summary.json"; then
  mv -f "$OUT/summary.json" "$OUT/fastwam_summary.json"
fi
eval_summary --model fastwam --bench libero --arm "$ARM" --steps "$STEPS" --sim "$SIM" \
  --mode "$MODE" --trials "$TRIALS" --output-dir "$OUT" --rc "$rc" --native "$OUT" || true
exit $rc
