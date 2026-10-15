export WANDB_API_KEY=your_wandb_key
export PYTHONPATH=$(pwd)

Framework_name=DiT4DiT
base_vlm=/path/to/Cosmos-Predict2.5-2B
freeze_module_list="backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae"
DIT_TYPE="DiT-B"
data_root_dir=/path/to/your_unitree_g1_dataset
data_mix=g1_decoupled_wbc


run_root_dir=./playground/Checkpoints_real
run_id=dit4dit_real_g1
pretrained_ckpt=/path/to/your/pretrained/ckpt

export WANDB_MODE=offline

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

accelerate launch \
  --config_file DiT4DiT/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 2 \
  DiT4DiT/training/train.py \
  --config_yaml ./DiT4DiT/config/real_robot/dit4dit_g1.yaml \
  --framework.name ${Framework_name} \
  --framework.cosmos25.base_model ${base_vlm} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --framework.action_model.action_horizon 50 \
  --framework.action_model.future_action_window_size 49 \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 2 \
  --datasets.vla_data.max_action_dim 36 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.pretrained_checkpoint ${pretrained_ckpt} \
  --trainer.max_train_steps 16000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.learning_rate.base 3e-5 \
  --trainer.learning_rate.vlm_interface 1e-5 \
  --trainer.learning_rate.action_model 1e-4 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project Dit4Dit_real_g1 \
  --wandb_entity your_wandb_entity \
  --framework.cosmos25.extract_layer 17 \



