# Copyright 2025 DiT4DiT team. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].
# Modified by [Teli Ma/ HKUST GZ] in [2025]. 
# Modification: [modify more efficient distributed training and from pre-training mode to training mode].
# Modified by Advanced Micro Devices, Inc. Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.


# Standard Library
import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import re

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from DiT4DiT.training.trainer_utils.trainer_tools import normalize_dotlist_args
from DiT4DiT.model.framework import build_framework
from DiT4DiT.training.trainer_utils.trainer_tools import TrainerUtils
from DiT4DiT.training.trainer_utils.trainer_tools import build_param_lr_groups
from DiT4DiT.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig
from DiT4DiT.model.checkpointing import build_trainable_delta_checkpoint
# 获取本地 Rank（Ray 通常会自动设置 LOCAL_RANK 环境变量）
# local_rank = int(os.environ.get("LOCAL_RANK", 0))

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)


def build_accelerator(cfg) -> Accelerator:
    """Create Accelerate/DeepSpeed with the configured accumulation window."""
    accumulation_steps = int(cfg.trainer.gradient_accumulation_steps)
    if accumulation_steps < 1:
        raise ValueError("trainer.gradient_accumulation_steps must be at least 1")
    deepspeed_config = (
        Path(__file__).resolve().parents[1] / "config/deepseeds/ds_config.yaml"
    )
    deepspeed_plugin = DeepSpeedPlugin(
        hf_ds_config=str(deepspeed_config),
        gradient_accumulation_steps=accumulation_steps,
    )
    configured_accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=accumulation_steps,
    )
    if configured_accelerator.gradient_accumulation_steps != accumulation_steps:
        raise RuntimeError(
            "Accelerate gradient accumulation mismatch: "
            f"requested {accumulation_steps}, got "
            f"{configured_accelerator.gradient_accumulation_steps}"
        )
    configured_accelerator.print(configured_accelerator.state)
    return configured_accelerator


def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    return fast_tokenizer


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # # save config
        # OmegaConf.save(cfg, output_dir / "config.yaml")
        # with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
        #     yaml_cfg = yaml.safe_load(f_yaml)
        #     json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    return model


def continuation_scheduler_steps(trainer_config) -> int:
    start_step = int(
        getattr(trainer_config, "continuation_start_step", 0) or 0
    )
    total_steps = int(trainer_config.max_train_steps)
    if getattr(trainer_config, "continuation_checkpoint", None):
        total_steps -= start_step
        if total_steps <= 0:
            raise ValueError(
                "Continuation max_train_steps must exceed continuation_start_step"
            )
    return total_steps


def continuation_consumed_batches(trainer_config, completed_steps: int) -> int:
    configured_batches = getattr(
        trainer_config,
        "continuation_consumed_batches",
        None,
    )
    if configured_batches is not None:
        configured_batches = int(configured_batches)
        if configured_batches < 0:
            raise ValueError("continuation_consumed_batches cannot be negative")
        return configured_batches
    gradient_accumulation = int(trainer_config.gradient_accumulation_steps)
    eval_interval = int(trainer_config.eval_interval)
    return (
        completed_steps * gradient_accumulation
        + completed_steps // eval_interval
    )


# here changes need to 📦 encapsulate Dataloader
from DiT4DiT.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA data loader
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    # Access in main process so this key is tracked and persisted by AccessTrackedConfig.
    action_video_freq_ratio = cfg.datasets.vla_data.get("action_video_freq_ratio", 1)
    logger.info(f"Using action_video_freq_ratio={action_video_freq_ratio}")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()

    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    scheduler_training_steps = continuation_scheduler_steps(cfg.trainer)
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=scheduler_training_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # training status tracking
        self.completed_steps = 0
        self.weights_only_continuation = False
        self.pending_resume_state = None
        self.resume_data_batches_consumed = None
        self.data_batches_consumed = 0
        self.total_batch_size = self._calculate_total_batch_size()
        benchmark_cfg = getattr(self.config.trainer, "benchmark", None)
        self.benchmark_enabled = bool(
            getattr(benchmark_cfg, "enabled", False)
        )
        self.benchmark_warmup_steps = int(
            getattr(benchmark_cfg, "warmup_steps", 10)
        )
        self.benchmark_samples = []
        self.benchmark_eval_samples = []
    
    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # load pretrained weights
        self._init_checkpointing()

        # 根据  resume 调整 lr_scheduler
        self._adjust_lr_scheduler_for_resume()

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        gradient_checkpointing_enabled = bool(
            getattr(
                self.config.trainer,
                "enable_gradient_checkpointing",
                False,
            )
        )
        checkpoint_targets = {
            "action_model": getattr(
                getattr(self.model, "action_model", None),
                "model",
                None,
            ),
            "backbone_interface": getattr(
                getattr(
                    getattr(self.model, "backbone_interface", None),
                    "extractor",
                    None,
                ),
                "transformer",
                None,
            ),
        }
        configured_targets = []
        for target_name, target in checkpoint_targets.items():
            method_name = (
                "enable_gradient_checkpointing"
                if gradient_checkpointing_enabled
                else "disable_gradient_checkpointing"
            )
            method = getattr(target, method_name, None)
            if callable(method):
                method()
                configured_targets.append(target_name)
        logger.info(
            "Gradient checkpointing enabled=%s for targets=%s",
            gradient_checkpointing_enabled,
            configured_targets,
        )

        compile_cfg = getattr(self.config.trainer, "compile", None)
        compile_module = str(getattr(compile_cfg, "module", "none")).lower()
        compile_mode = str(getattr(compile_cfg, "mode", "default"))
        if compile_module != "none":
            if compile_module not in {
                "action_model",
                "backbone_interface",
                "cosmos_transformer",
            }:
                raise ValueError(
                    "trainer.compile.module must be one of: "
                    "none, action_model, backbone_interface, cosmos_transformer"
                )
            if compile_module == "cosmos_transformer":
                extractor = self.model.backbone_interface.extractor
                extractor.transformer = torch.compile(
                    extractor.transformer,
                    mode=compile_mode,
                    dynamic=False,
                )
            else:
                target = getattr(self.model, compile_module)
                setattr(
                    self.model,
                    compile_module,
                    torch.compile(
                        target,
                        mode=compile_mode,
                        dynamic=False,
                    ),
                )
            logger.info(
                "Enabled torch.compile for %s with mode=%s",
                compile_module,
                compile_mode,
            )

        # Guard: if everything is frozen, Deepspeed ZeRO optimizer init will crash with an opaque
        # `torch.cat(): expected a non-empty list of Tensors`.
        # any_trainable = any(p.requires_grad for p in self.model.parameters())
        # if not any_trainable:
        #     # Show a few top-level module names to help locate what got frozen.
        #     top_modules = list(dict(self.model.named_children()).keys())
        #     raise RuntimeError(
        #         "No trainable parameters found after freezing. "
        #         "Please check `trainer.freeze_modules` and any backbone wrappers that set requires_grad_(False).\n"
        #         f"- trainer.freeze_modules: {freeze_modules!r}\n"
        #         f"- top-level modules: {top_modules}\n"
        #         "Fix: ensure at least the action head stays trainable (e.g., do not freeze `action_model`)."
        #     )

        # # IMPORTANT: the optimizer was built before we freeze modules (see main()).
        # # Deepspeed ZeRO assumes optimizer param groups contain trainable params; if a group becomes empty after
        # # filtering `requires_grad`, it may crash during flattening.
        # if self.optimizer is not None and hasattr(self.optimizer, "param_groups"):
        #     for group in self.optimizer.param_groups:
        #         group["params"] = [p for p in group.get("params", []) if getattr(p, "requires_grad", False)]
        #     # drop empty groups
        #     self.optimizer.param_groups = [g for g in self.optimizer.param_groups if len(g.get("params", [])) > 0]
        #     total_opt_params = sum(len(g.get("params", [])) for g in self.optimizer.param_groups)
        #     if total_opt_params == 0:
        #         raise RuntimeError(
        #             "Optimizer has 0 trainable parameters after freezing/pruning. "
        #             "This will crash Deepspeed ZeRO.\n"
        #             f"- trainer.freeze_modules: {freeze_modules!r}\n"
        #             "Fix: ensure your action head parameters are included in the optimizer param_groups."
        #         )

        # Print parameter statistics after freezing
        if not dist.is_initialized() or dist.get_rank() == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            logger.info("=" * 80)
            logger.info("📊 Model Parameter Statistics (after freezing):")
            logger.info(f"  Total parameters:      {total_params:,} ({total_params / 10**6:.3f}M)")
            logger.info(f"  Trainable parameters:  {trainable_params:,} ({trainable_params / 10**6:.3f}M)")
            logger.info(f"  Frozen parameters:     {frozen_params:,} ({frozen_params / 10**6:.3f}M)")
            logger.info(f"  Trainable ratio:       {trainable_params / total_params * 100:.2f}%")
            logger.info("=" * 80)

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )
        self.accelerator.register_for_checkpointing(self.lr_scheduler)
        if self.pending_resume_state:
            self.accelerator.load_state(self.pending_resume_state)
            logger.info(
                "Restored full model/optimizer/scheduler state from %s",
                self.pending_resume_state,
            )

        self._init_wandb()


    def _adjust_lr_scheduler_for_resume(self):
        """根据已完成的步数调整学习率调度器状态"""
        if self.weights_only_continuation or self.pending_resume_state:
            logger.info(
                "Continuation scheduler starts at local step zero; "
                "global completed_steps=%s",
                self.completed_steps,
            )
            return
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            
            # 方法1: 直接模拟已完成的步数（适用于大多数调度器）
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            
            # 或者方法2: 对于某些调度器，可以直接设置最后步数
            # if hasattr(self.lr_scheduler, '_step_count'):
            #     self.lr_scheduler._step_count = self.completed_steps
            
            logger.info(f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}")

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 获取预训练检查点和是否恢复训练的标志
        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint
        continuation_checkpoint = getattr(
            self.config.trainer,
            "continuation_checkpoint",
            None,
        )
        continuation_start_step = int(
            getattr(self.config.trainer, "continuation_start_step", 0) or 0
        )
        resume_state_path = getattr(
            self.config.trainer,
            "resume_state_path",
            None,
        )
        if continuation_checkpoint and (is_resume or resume_state_path):
            raise ValueError(
                "continuation_checkpoint cannot be combined with is_resume "
                "or resume_state_path"
            )
        if continuation_checkpoint:
            if continuation_start_step <= 0:
                raise ValueError(
                    "continuation_start_step must be positive for weights-only "
                    "continuation"
                )
            if not os.path.isfile(continuation_checkpoint):
                raise FileNotFoundError(
                    f"Continuation checkpoint not found: {continuation_checkpoint}"
                )
            self.model = self.load_pretrained_backbones(
                self.model,
                continuation_checkpoint,
                reload_modules=None,
            )
            self.completed_steps = continuation_start_step
            self.weights_only_continuation = True
            self.resume_from_checkpoint = continuation_checkpoint
            if not dist.is_initialized() or dist.get_rank() == 0:
                metadata_path = Path(self.config.output_dir) / "continuation.json"
                metadata_path.write_text(
                    json.dumps(
                        {
                            "mode": "weights_only_optimizer_restart",
                            "source_checkpoint": str(
                                Path(continuation_checkpoint).resolve()
                            ),
                            "continuation_start_step": continuation_start_step,
                            "target_step": int(
                                self.config.trainer.max_train_steps
                            ),
                            "optimizer_state_restored": False,
                            "data_position_replayed": bool(
                                getattr(
                                    self.config.trainer,
                                    "replay_data_position",
                                    False,
                                )
                            ),
                        },
                        indent=2,
                    )
                    + "\n"
                )
            logger.info(
                "Continuing from model weights %s at global step %s with a "
                "fresh optimizer",
                continuation_checkpoint,
                continuation_start_step,
            )
            return
        if resume_state_path:
            if not os.path.isdir(resume_state_path):
                raise FileNotFoundError(
                    f"Full training state not found: {resume_state_path}"
                )
            progress_path = Path(resume_state_path) / "trainer_progress.json"
            if progress_path.is_file():
                progress = json.loads(progress_path.read_text())
                saved_step = int(progress["completed_steps"])
                if (
                    continuation_start_step > 0
                    and continuation_start_step != saved_step
                ):
                    raise ValueError(
                        "continuation_start_step does not match the full-state "
                        f"checkpoint: {continuation_start_step} != {saved_step}"
                    )
                continuation_start_step = saved_step
                self.resume_data_batches_consumed = int(
                    progress["data_batches_consumed"]
                )
            elif continuation_start_step <= 0:
                match = re.search(
                    r"steps_(\d+)_training_state/?$",
                    str(resume_state_path),
                )
                if not match:
                    raise ValueError(
                        "Could not infer continuation_start_step from "
                        f"resume_state_path: {resume_state_path}"
                    )
                continuation_start_step = int(match.group(1))
            self.completed_steps = continuation_start_step
            self.pending_resume_state = resume_state_path
            return
        if is_resume:
            # 恢复训练状态
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}")
                return None
            else:
                logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
                self.completed_steps = 0

        # 加载预训练权重
        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0
    

    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _prune_full_training_states(self):
        max_states = int(
            getattr(self.config.trainer, "max_full_training_states", 0) or 0
        )
        if max_states <= 0 or not self.accelerator.is_main_process:
            return
        states = []
        for path in Path(self.checkpoint_dir).glob(
            "steps_*_training_state"
        ):
            match = re.fullmatch(
                r"steps_(\d+)_training_state",
                path.name,
            )
            if match and path.is_dir():
                states.append((int(match.group(1)), path))
        for _, path in sorted(states)[:-max_states]:
            shutil.rmtree(path)
            logger.info("Removed old full training state %s", path)

    def _save_checkpoint(
        self,
        save_model: bool = True,
        save_training_state: bool = False,
    ):
        """save current training state"""

        if save_model and self.accelerator.is_main_process:

            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # save model state
            state_dict = self.accelerator.get_state_dict(self.model)
            save_trainable_only = bool(
                getattr(
                    self.config.trainer,
                    "save_trainable_only_checkpoints",
                    False,
                )
            )
            full_model_interval = int(
                getattr(
                    self.config.trainer,
                    "full_model_save_interval",
                    0,
                )
                or 0
            )
            is_full_model_milestone = (
                full_model_interval > 0
                and self.completed_steps % full_model_interval == 0
            )
            if save_trainable_only and not is_full_model_milestone:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                action_vae_config = getattr(
                    self.config.framework,
                    "action_vae",
                    None,
                )
                checkpoint_payload = build_trainable_delta_checkpoint(
                    unwrapped_model,
                    state_dict,
                    base_artifacts={
                        "cosmos_base_model": str(
                            self.config.framework.cosmos25.base_model
                        ),
                        "action_vae_checkpoint": (
                            str(action_vae_config.checkpoint_path)
                            if action_vae_config is not None
                            else None
                        ),
                    },
                )
                checkpoint_kind = "trainable delta"
            else:
                checkpoint_payload = state_dict
                checkpoint_kind = "full model"
            torch.save(
                checkpoint_payload,
                checkpoint_path + "_pytorch_model.pt",
            )

            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(
                f"✅ {checkpoint_kind} checkpoint saved at {checkpoint_path}"
            )
            # ✅ Save accessed configuration only
            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                # self.config.save_accessed_config(
                #     output_dir / "config.json", 
                #     use_original_values=False
                # )
                self.config.save_accessed_config(
                    output_dir / "config.yaml", 
                    use_original_values=False 
                )
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()
        if save_training_state:
            state_path = os.path.join(
                self.checkpoint_dir,
                f"steps_{self.completed_steps}_training_state",
            )
            self.accelerator.save_state(state_path)
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                progress = {
                    "completed_steps": self.completed_steps,
                    "data_batches_consumed": self.data_batches_consumed,
                    "gradient_accumulation_steps": int(
                        self.config.trainer.gradient_accumulation_steps
                    ),
                    "eval_interval": int(
                        self.config.trainer.eval_interval
                    ),
                }
                (
                    Path(state_path) / "trainer_progress.json"
                ).write_text(json.dumps(progress, indent=2) + "\n")
            self.accelerator.print(
                f"✅ Full optimizer/scheduler state saved at {state_path}"
            )
            self.accelerator.wait_for_everyone()
            self._prune_full_training_states()
            self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                # add learning rate for each param group
                for group, lr in zip(self.optimizer.param_groups, self.lr_scheduler.get_last_lr()):
                    metrics[f"learning_rate/{group['name']}"] = lr

                # add epoch info
                metrics["epoch"] = round(self.completed_steps * self.config.trainer.gradient_accumulation_steps  / len(self.vla_train_dataloader), 2)

                # record to W&B
                wandb.log(metrics, step=self.completed_steps)
                # debug output
                logger.info(f"[Exp: {self.config.run_id}] Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """create data iterators"""
        replay_data_position = bool(
            getattr(self.config.trainer, "replay_data_position", False)
        )
        should_restore_position = (
            (self.weights_only_continuation and replay_data_position)
            or getattr(self, "pending_resume_state", None) is not None
        )
        if should_restore_position:
            resumed_batches = getattr(
                self,
                "resume_data_batches_consumed",
                None,
            )
            configured_batches = (
                resumed_batches
                if resumed_batches is not None
                else continuation_consumed_batches(
                    self.config.trainer,
                    self.completed_steps,
                )
            )
            batches_per_epoch = len(self.vla_train_dataloader)
            completed_epochs, batches_to_skip = divmod(
                configured_batches,
                batches_per_epoch,
            )
            self.vla_epoch_count = completed_epochs
            sampler = getattr(self.vla_train_dataloader, "sampler", None)
            if callable(getattr(sampler, "set_epoch", None)):
                sampler.set_epoch(completed_epochs)
            if batches_to_skip:
                replay_loader = self.accelerator.skip_first_batches(
                    self.vla_train_dataloader,
                    batches_to_skip,
                )
                self.vla_iter = iter(replay_loader)
            else:
                self.vla_iter = iter(self.vla_train_dataloader)
            self.data_batches_consumed = configured_batches
            logger.info(
                "Restored dataloader position at epoch=%s, batch=%s "
                "(%s batches consumed)",
                completed_epochs,
                batches_to_skip,
                configured_batches,
            )
        else:
            self.vla_iter = iter(self.vla_train_dataloader)
        # self.vlm_iter = iter(self.vlm_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        self.data_batches_consumed += 1
        return batch_vla

    def _synchronize_device(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _write_benchmark_report(self):
        if not self.benchmark_enabled:
            return
        if not self.benchmark_samples:
            raise RuntimeError(
                "Benchmark collected no measured steps; reduce "
                "trainer.benchmark.warmup_steps"
            )

        updates = np.asarray(
            [sample["update_seconds"] for sample in self.benchmark_samples],
            dtype=np.float64,
        )
        data_times = np.asarray(
            [sample["data_seconds"] for sample in self.benchmark_samples],
            dtype=np.float64,
        )
        model_times = np.asarray(
            [sample["model_seconds"] for sample in self.benchmark_samples],
            dtype=np.float64,
        )
        eval_times = np.asarray(
            self.benchmark_eval_samples,
            dtype=np.float64,
        )
        losses = np.asarray(
            [sample["total_loss"] for sample in self.benchmark_samples],
            dtype=np.float64,
        )
        local_values = torch.tensor(
            [
                len(updates),
                updates.sum(),
                data_times.sum(),
                model_times.sum(),
                len(eval_times),
                eval_times.sum(),
                torch.cuda.max_memory_allocated(),
                torch.cuda.max_memory_reserved(),
                float(np.isfinite(losses).all()),
                losses[-1],
                losses.min(),
                losses.max(),
            ],
            dtype=torch.float64,
            device=self.accelerator.device,
        )
        gathered = [
            torch.zeros_like(local_values)
            for _ in range(self.accelerator.num_processes)
        ]
        dist.all_gather(gathered, local_values)
        rank_values = [values.cpu().tolist() for values in gathered]

        if self.accelerator.is_main_process:
            mean_updates = [
                values[1] / values[0]
                for values in rank_values
                if values[0] > 0
            ]
            mean_data = [
                values[2] / values[0]
                for values in rank_values
                if values[0] > 0
            ]
            mean_model = [
                values[3] / values[0]
                for values in rank_values
                if values[0] > 0
            ]
            mean_evals = [
                values[5] / values[4]
                for values in rank_values
                if values[4] > 0
            ]
            bottleneck_update = max(mean_updates)
            eval_overhead = max(mean_evals) if mean_evals else 0.0
            eval_interval = int(self.config.trainer.eval_interval)
            projected_step_seconds = (
                bottleneck_update + eval_overhead / eval_interval
            )
            report = {
                "run_id": self.config.run_id,
                "world_size": self.accelerator.num_processes,
                "per_device_batch_size": int(
                    self.config.datasets.vla_data.per_device_batch_size
                ),
                "gradient_accumulation_steps": int(
                    self.config.trainer.gradient_accumulation_steps
                ),
                "gradient_checkpointing_enabled": bool(
                    getattr(
                        self.config.trainer,
                        "enable_gradient_checkpointing",
                        False,
                    )
                ),
                "cache_prompt_embeddings": bool(
                    getattr(
                        self.config.framework.cosmos25,
                        "cache_prompt_embeddings",
                        False,
                    )
                ),
                "reuse_causal_vae_encode": bool(
                    getattr(
                        self.config.framework.cosmos25,
                        "reuse_causal_vae_encode",
                        False,
                    )
                ),
                "global_batch_size": self.total_batch_size,
                "num_workers": int(
                    self.config.datasets.vla_data.get("num_workers", 4)
                ),
                "prefetch_factor": int(
                    self.config.datasets.vla_data.get("prefetch_factor", 2)
                ),
                "warmup_steps": self.benchmark_warmup_steps,
                "measured_steps": int(min(values[0] for values in rank_values)),
                "compile_module": str(
                    getattr(
                        getattr(self.config.trainer, "compile", None),
                        "module",
                        "none",
                    )
                ),
                "mean_update_seconds": bottleneck_update,
                "rank0_update_p50_seconds": float(np.median(updates)),
                "rank0_update_p95_seconds": float(
                    np.percentile(updates, 95)
                ),
                "mean_data_seconds": max(mean_data),
                "mean_model_seconds": max(mean_model),
                "mean_eval_seconds": eval_overhead,
                "losses_finite_on_all_ranks": all(
                    bool(values[8]) for values in rank_values
                ),
                "last_loss_by_rank": [values[9] for values in rank_values],
                "minimum_loss_by_rank": [values[10] for values in rank_values],
                "maximum_loss_by_rank": [values[11] for values in rank_values],
                "rank0_last_loss": float(losses[-1]),
                "rank0_min_loss": float(losses.min()),
                "rank0_max_loss": float(losses.max()),
                "examples_per_second": (
                    self.total_batch_size / bottleneck_update
                ),
                "projected_seconds_per_step_with_eval": (
                    projected_step_seconds
                ),
                "projected_200k_days": (
                    projected_step_seconds * 200_000 / 86_400
                ),
                "peak_allocated_gib_by_rank": [
                    values[6] / (1024**3) for values in rank_values
                ],
                "peak_reserved_gib_by_rank": [
                    values[7] / (1024**3) for values in rank_values
                ],
            }
            report_path = (
                Path(self.config.output_dir) / "benchmark_metrics.json"
            )
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            logger.info("Benchmark report written to %s", report_path)
        self.accelerator.wait_for_everyone()

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()
        if self.benchmark_enabled:
            self._synchronize_device()
            torch.cuda.reset_peak_memory_stats()

        # create progress bar
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )
        accumulated_update_start = None
        accumulated_data_seconds = 0.0
        accumulated_model_seconds = 0.0

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            if self.benchmark_enabled and accumulated_update_start is None:
                self._synchronize_device()
                accumulated_update_start = time.perf_counter()
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()
            if self.benchmark_enabled:
                accumulated_data_seconds += t_end_data - t_start_data

            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            if self.benchmark_enabled:
                self._synchronize_device()
            t_end_model = time.perf_counter()
            if self.benchmark_enabled:
                accumulated_model_seconds += t_end_model - t_start_model

            # update progress
            did_step = bool(self.accelerator.sync_gradients)
            if did_step:
                progress_bar.update(1)
                self.completed_steps += 1
            
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                        {
                            "data_times": f"{t_end_data - t_start_data:.3f}",
                            "model_times": f"{t_end_model - t_start_model:.3f}",
                        }
                    )

            # Only eval/log/save on real optimizer steps (end of accumulation window)
            if did_step:
                # evaluate model (skip action eval in video-only mode)
                video_fm_only = getattr(self.model, "video_fm_only", False)
                eval_seconds = None
                if (not video_fm_only) and (self.completed_steps % self.config.trainer.eval_interval == 0):
                    if self.benchmark_enabled:
                        self._synchronize_device()
                        t_start_eval = time.perf_counter()
                    step_metrics = self.eval_action_model(step_metrics)
                    if self.benchmark_enabled:
                        self._synchronize_device()
                        eval_seconds = time.perf_counter() - t_start_eval

                # record metrics
                step_metrics["data_time"] = t_end_data - t_start_data
                step_metrics["model_time"] = t_end_model - t_start_model
                if (
                    self.benchmark_enabled
                    and self.completed_steps > self.benchmark_warmup_steps
                ):
                    self.benchmark_samples.append(
                        {
                            "update_seconds": (
                                t_end_model - accumulated_update_start
                            ),
                            "data_seconds": accumulated_data_seconds,
                            "model_seconds": accumulated_model_seconds,
                            "total_loss": float(step_metrics["total_loss"]),
                        }
                    )
                    if eval_seconds is not None:
                        self.benchmark_eval_samples.append(eval_seconds)
                if self.benchmark_enabled:
                    accumulated_update_start = None
                    accumulated_data_seconds = 0.0
                    accumulated_model_seconds = 0.0
                self._log_metrics(step_metrics)

                # save checkpoint
                save_model = (
                    self.completed_steps > 0
                    and self.completed_steps
                    % int(self.config.trainer.save_interval)
                    == 0
                )
                full_state_interval = int(
                    getattr(
                        self.config.trainer,
                        "full_state_save_interval",
                        0,
                    )
                    or 0
                )
                save_training_state = (
                    (
                        full_state_interval > 0
                        and self.completed_steps % full_state_interval == 0
                    )
                    or (
                        bool(
                            getattr(
                                self.config.trainer,
                                "save_full_training_state",
                                False,
                            )
                        )
                        and save_model
                    )
                )
                if save_model or save_training_state:
                    self._save_checkpoint(
                        save_model=save_model,
                        save_training_state=save_training_state,
                    )

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._write_benchmark_report()
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        examples = self._get_next_batch()
        score = 0.0
        num_samples = len(examples)
        actions = [example["action"] for example in examples]  # label
        # Predict actions using the model
        action_mask = [example["action_mask"] for example in examples] # [B, len, action_dim]
        output_dict = self.model.predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]  # B, T, D
            actions = np.array(actions)  # convert actions to numpy.ndarray
            action_mask = np.array(action_mask)  # convert action_mask to numpy.ndarray [B, T, D]
            # Apply action_mask: only compute MSE on valid (True) dimensions
            masked_diff = (normalized_actions - actions) * action_mask
            mse = (masked_diff ** 2).sum() / action_mask.sum()
            step_metrics["mse_score"] = mse

        del examples
        dist.barrier()  # ensure all processes are synchronized
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
            logger.info(f"  len(vla_train_dataloader) = {len(self.vla_train_dataloader)}")
            logger.info(f"  len(vla_train_dataloader.dataset) = {len(self.vla_train_dataloader.dataset)}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):
            # VLA task forward propagation
            # NOTE: In some DeepSpeed/Accelerate configurations, autograd can be globally disabled in engine.forward().
            # We force-enable grad here to ensure `loss.grad_fn` exists in training.

            with torch.enable_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output_dict = self.model.forward(batch_vla)

                    action_loss = output_dict.get("action_loss", None)
                    future_video_loss = output_dict.get("future_video_loss", None)

                    # Validate at least one loss exists
                    if action_loss is None and future_video_loss is None:
                        raise KeyError(
                            "Model output must contain either `action_loss` or `future_video_loss` for training."
                        )
                    if action_loss is not None and not torch.is_tensor(action_loss):
                        raise TypeError(
                            f"`action_loss` must be a torch.Tensor, got {type(action_loss)}. "
                            "This usually means model.forward returned a Python float or did `.item()`."
                        )
                    if future_video_loss is not None and not torch.is_tensor(future_video_loss):
                        raise TypeError(
                            f"`future_video_loss` must be a torch.Tensor, got {type(future_video_loss)}."
                        )

                    # Compute total loss based on training mode:
                    #   video:  total = future_video_loss * scale
                    #   action: total = action_loss
                    #   joint:  total = action_loss + future_video_loss * scale
                    future_video_loss_scaled = None
                    video_loss_scale = 1.0
                    if future_video_loss is not None:
                        try:
                            video_loss_scale = float(getattr(getattr(self.config.trainer, "loss_scale", None), "future_video", 1.0))
                        except Exception:
                            video_loss_scale = 1.0
                        future_video_loss_scaled = future_video_loss * video_loss_scale

                    if action_loss is not None and future_video_loss is not None:
                        # joint: action + video auxiliary
                        total_loss = action_loss + future_video_loss_scaled
                    elif action_loss is not None:
                        # action only
                        total_loss = action_loss
                    else:
                        # video only
                        total_loss = future_video_loss_scaled

                    # VLA backward propagation (keep inside enable_grad scope)
                    self.accelerator.backward(total_loss)

            # Only step optimizer / scheduler when gradients are synchronized (i.e., end of accumulation window).
            if self.accelerator.sync_gradients:
                # gradient clipping
                if self.config.trainer.gradient_clipping is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

                # optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

        step_metrics = {}
        if action_loss is not None:
            step_metrics["action_dit_loss"] = action_loss.item()
        if future_video_loss is not None:
            # raw auxiliary loss
            step_metrics["future_video_loss"] = future_video_loss.item() if torch.is_tensor(future_video_loss) else float(future_video_loss)
        if future_video_loss_scaled is not None:
            step_metrics["future_video_loss_scaled"] = (
                future_video_loss_scaled.item()
                if torch.is_tensor(future_video_loss_scaled)
                else float(future_video_loss_scaled)
            )
        # total loss (for quick monitoring only)
        step_metrics["total_loss"] = total_loss.item() if torch.is_tensor(total_loss) else float(total_loss)
        return step_metrics

    def _finalize_training(self):
        """training end processing"""
        # save final model
        skip_final_checkpoint = bool(
            getattr(self.config.trainer, "skip_final_checkpoint", False)
        )
        if self.accelerator.is_main_process and not skip_final_checkpoint:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")
        elif self.accelerator.is_main_process:
            logger.info("Skipping final checkpoint as configured")


        # close W&B
        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")
    accelerator = build_accelerator(cfg)

    #  Wrap config to enable access tracking
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    # build model
    vla = build_framework(cfg)
    # prepare data
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="xxx.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    main(cfg)
