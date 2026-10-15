#!/usr/bin/env bash
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export WANDB_API_KEY=${WANDB_API_KEY:-""}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

Framework_name=DiT4DiT
base_model=${COSMOS_BASE:-"${REPO_ROOT}/Cosmos-Predict2.5-2B"}
freeze_module_list="backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae"
DIT_TYPE="DiT-B"
data_root_dir=${DATA_ROOT_DIR:-"${REPO_ROOT}/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"}
data_mix=${DATA_MIX:-fourier_gr1_unified_1000}
config_yaml=${CONFIG_YAML:-"${REPO_ROOT}/DiT4DiT/config/robocasa/dit4dit_robocasa_gr1.yaml"}
ACTION_VAE_CHECKPOINT=${ACTION_VAE_CHECKPOINT:-}
POLICY_ACTION_STATISTICS_PATH=${POLICY_ACTION_STATISTICS_PATH:-}
ACTION_VAE_ARGS=()
if [[ -n "${ACTION_VAE_CHECKPOINT}" ]]; then
  freeze_module_list="${freeze_module_list},action_vae"
  ACTION_VAE_ARGS=(--framework.action_vae.checkpoint_path "${ACTION_VAE_CHECKPOINT}")
  if [[ -n "${POLICY_ACTION_STATISTICS_PATH}" ]]; then
    ACTION_VAE_ARGS+=(
      --framework.action_vae.policy_action_statistics_path
      "${POLICY_ACTION_STATISTICS_PATH}"
    )
  fi
fi


run_root_dir=${RUN_ROOT_DIR:-"${REPO_ROOT}/playground/Checkpoints_robocasa"}
run_id=${RUN_ID:-dit4dit_robocasa_gr1}

export WANDB_MODE=${WANDB_MODE:-offline}

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
# mv this script to the output dir
cp "$0" "${output_dir}/"

NUM_PROCESSES=${NUM_PROCESSES:-2}
MASTER_PORT=${MASTER_PORT:-29500}
SEED=${SEED:-42}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-200000}
SAVE_INTERVAL=${SAVE_INTERVAL:-50000}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
NUM_WARMUP_STEPS=${NUM_WARMUP_STEPS:-5000}
LOGGING_FREQUENCY=${LOGGING_FREQUENCY:-10}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
IS_RESUME=${IS_RESUME:-false}
CONTINUATION_CHECKPOINT=${CONTINUATION_CHECKPOINT:-}
CONTINUATION_START_STEP=${CONTINUATION_START_STEP:-0}
RESUME_STATE_PATH=${RESUME_STATE_PATH:-}
REPLAY_DATA_POSITION=${REPLAY_DATA_POSITION:-false}
SAVE_FULL_TRAINING_STATE=${SAVE_FULL_TRAINING_STATE:-false}
FULL_STATE_SAVE_INTERVAL=${FULL_STATE_SAVE_INTERVAL:-0}
MAX_FULL_TRAINING_STATES=${MAX_FULL_TRAINING_STATES:-0}
SAVE_TRAINABLE_ONLY_CHECKPOINTS=${SAVE_TRAINABLE_ONLY_CHECKPOINTS:-false}
FULL_MODEL_SAVE_INTERVAL=${FULL_MODEL_SAVE_INTERVAL:-0}
SKIP_FINAL_CHECKPOINT=${SKIP_FINAL_CHECKPOINT:-false}
BENCHMARK_ENABLED=${BENCHMARK_ENABLED:-false}
BENCHMARK_WARMUP_STEPS=${BENCHMARK_WARMUP_STEPS:-10}
COMPILE_MODULE=${COMPILE_MODULE:-none}
COMPILE_MODE=${COMPILE_MODE:-default}
NUM_WORKERS=${NUM_WORKERS:-4}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-false}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-true}
CACHE_PROMPT_EMBEDDINGS=${CACHE_PROMPT_EMBEDDINGS:-false}
REUSE_CAUSAL_VAE_ENCODE=${REUSE_CAUSAL_VAE_ENCODE:-false}
BASE_LEARNING_RATE=${BASE_LEARNING_RATE:-3e-5}
BACKBONE_LEARNING_RATE=${BACKBONE_LEARNING_RATE:-1e-5}
ACTION_LEARNING_RATE=${ACTION_LEARNING_RATE:-1e-4}
MIN_LEARNING_RATE=${MIN_LEARNING_RATE:-5e-7}
CONTINUATION_ARGS=()
if [[ -n "${CONTINUATION_CHECKPOINT}" ]]; then
  CONTINUATION_ARGS=(
    --trainer.continuation_checkpoint "${CONTINUATION_CHECKPOINT}"
    --trainer.continuation_start_step "${CONTINUATION_START_STEP}"
    --trainer.replay_data_position "${REPLAY_DATA_POSITION}"
    --trainer.scheduler_specific_kwargs.min_lr null
    --trainer.scheduler_specific_kwargs.min_lr_rate 0.25
  )
elif [[ -n "${RESUME_STATE_PATH}" ]]; then
  CONTINUATION_ARGS=(
    --trainer.resume_state_path "${RESUME_STATE_PATH}"
    --trainer.continuation_start_step "${CONTINUATION_START_STEP}"
  )
fi

"${ACCELERATE_BIN:-accelerate}" launch \
  --config_file "${REPO_ROOT}/DiT4DiT/config/deepseeds/deepspeed_zero2.yaml" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MASTER_PORT}" \
  "${REPO_ROOT}/DiT4DiT/training/train.py" \
  --config_yaml "${config_yaml}" \
  --seed ${SEED} \
  --framework.name ${Framework_name} \
  --framework.cosmos25.base_model "${base_model}" \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --datasets.vla_data.num_workers ${NUM_WORKERS} \
  --datasets.vla_data.persistent_workers ${PERSISTENT_WORKERS} \
  --datasets.vla_data.prefetch_factor ${PREFETCH_FACTOR} \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps ${MAX_TRAIN_STEPS} \
  --trainer.save_interval ${SAVE_INTERVAL} \
  --trainer.gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --trainer.is_resume ${IS_RESUME} \
  --trainer.logging_frequency ${LOGGING_FREQUENCY} \
  --trainer.eval_interval ${EVAL_INTERVAL} \
  --trainer.learning_rate.base ${BASE_LEARNING_RATE} \
  --trainer.learning_rate.backbone_interface ${BACKBONE_LEARNING_RATE} \
  --trainer.learning_rate.action_model ${ACTION_LEARNING_RATE} \
  --trainer.scheduler_specific_kwargs.min_lr ${MIN_LEARNING_RATE} \
  --trainer.num_warmup_steps ${NUM_WARMUP_STEPS} \
  --trainer.save_full_training_state ${SAVE_FULL_TRAINING_STATE} \
  --trainer.full_state_save_interval ${FULL_STATE_SAVE_INTERVAL} \
  --trainer.max_full_training_states ${MAX_FULL_TRAINING_STATES} \
  --trainer.save_trainable_only_checkpoints ${SAVE_TRAINABLE_ONLY_CHECKPOINTS} \
  --trainer.full_model_save_interval ${FULL_MODEL_SAVE_INTERVAL} \
  --trainer.skip_final_checkpoint ${SKIP_FINAL_CHECKPOINT} \
  --trainer.enable_gradient_checkpointing ${ENABLE_GRADIENT_CHECKPOINTING} \
  --trainer.benchmark.enabled ${BENCHMARK_ENABLED} \
  --trainer.benchmark.warmup_steps ${BENCHMARK_WARMUP_STEPS} \
  --trainer.compile.module ${COMPILE_MODULE} \
  --trainer.compile.mode ${COMPILE_MODE} \
  --framework.cosmos25.cache_prompt_embeddings ${CACHE_PROMPT_EMBEDDINGS} \
  --framework.cosmos25.reuse_causal_vae_encode ${REUSE_CAUSAL_VAE_ENCODE} \
  --framework.cosmos25.extract_layer 17 \
  --framework.cosmos25.flow_matching.time_distribution uniform \
  --framework.cosmos25.flow_matching.high_sigma_ratio null \
  --framework.cosmos25.flow_matching.high_sigma_min null \
  --framework.cosmos25.conditional_frame_timestep 0.0001 \
  --datasets.vla_data.action_video_freq_ratio 2 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project DiT4DiT_robocasa \
  --wandb_entity ${WANDB_ENTITY:-local} \
  "${ACTION_VAE_ARGS[@]}" \
  "${CONTINUATION_ARGS[@]}"

