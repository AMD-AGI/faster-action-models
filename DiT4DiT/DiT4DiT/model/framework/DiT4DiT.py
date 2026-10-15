# Copyright 2025 DiT4DiT team. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Teli Ma/ HKUST GZ] in [2025]. 
# Modified by Advanced Micro Devices, Inc. Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.


import json
import sys
from pathlib import Path
# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image



from DiT4DiT.training.trainer_utils import initialize_overwatch


logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from DiT4DiT.model.framework.base_framework import baseframework
from DiT4DiT.model.modules.vlm import get_backbone_model
from DiT4DiT.model.modules.action_model.ActionDiT import get_action_model, FlowmatchingActionHead
from DiT4DiT.model.modules.action_model.action_vae import load_action_vae
from DiT4DiT.dataloader.gr00t_lerobot.full24_tasks import FULL24_TASKS
from DiT4DiT.dataloader.gr00t_lerobot.pnp6_tasks import PNP6_TASKS
from DiT4DiT.training.trainer_utils.trainer_tools import resize_images
from DiT4DiT.model.tools import FRAMEWORK_REGISTRY


def _match_action_vae_registry(
    checkpoint_datasets: object,
    num_tasks: int,
) -> tuple[str, list[str]]:
    supported_registries = {
        "pnp6": [task.dataset_name for task in PNP6_TASKS],
        "full24": [task.dataset_name for task in FULL24_TASKS],
    }
    matched_registry = next(
        (
            name
            for name, datasets in supported_registries.items()
            if checkpoint_datasets == datasets
        ),
        None,
    )
    if matched_registry is None:
        raise ValueError(
            "ActionVAE task registry must match canonical PnP-6 "
            "or full-24 dataset order"
        )
    expected_datasets = supported_registries[matched_registry]
    if num_tasks != len(expected_datasets):
        raise ValueError(
            "ActionVAE num_tasks does not match its canonical "
            f"{matched_registry} registry: {num_tasks} "
            f"!= {len(expected_datasets)}"
        )
    return matched_registry, expected_datasets


@FRAMEWORK_REGISTRY.register("DiT4DiT")
class DiT4DiT(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config

        # Determine training mode from config: "video", "action", or "joint"
        training_mode = config.framework.cosmos25.training.lower() if config is not None else "action"
        self.video_fm_only = (training_mode == "video")

        self.backbone_interface = get_backbone_model(config=self.config)

        # -------- Align DiT cross-attention dim with backbone output dim --------
        # GR00T ActionHead uses `diffusion_model_cfg.cross_attention_dim` to match vl_embs' last dim.
        vl_hidden_dim = None
        if hasattr(self.backbone_interface, "model") and hasattr(self.backbone_interface.model, "config"):
            vl_hidden_dim = getattr(self.backbone_interface.model.config, "hidden_size", None)
        if vl_hidden_dim is None and hasattr(self.backbone_interface, "extractor"):
            vl_hidden_dim = getattr(self.backbone_interface.extractor, "hidden_size", None)
        if vl_hidden_dim is None:
            vl_hidden_dim = getattr(self.config.framework.cosmos25, "vl_hidden_dim", None)

        if not self.video_fm_only:
            if vl_hidden_dim is None:
                raise ValueError(
                    "Cannot infer `vl_hidden_dim` for the selected backbone. "
                    "Please set `framework.cosmos25.vl_hidden_dim` in your config."
                )
            self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

            self.future_action_window_size = config.framework.action_model.future_action_window_size
            self.past_action_window_size = config.framework.action_model.past_action_window_size
            self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

            action_vae_config = getattr(config.framework, "action_vae", None)
            self.use_per_task_vae_action_norm = False
            self.use_action_vae = bool(
                action_vae_config is not None
                and getattr(action_vae_config, "enabled", False)
            )
            if self.use_action_vae:
                checkpoint_path = getattr(action_vae_config, "checkpoint_path", None)
                if not checkpoint_path:
                    raise ValueError(
                        "framework.action_vae.checkpoint_path is required when the "
                        "latent action VAE is enabled"
                    )
                self.action_vae, action_vae_checkpoint = load_action_vae(
                    checkpoint_path
                )
                self.action_vae.requires_grad_(False)
                self.action_vae.eval()
                vae_config = self.action_vae.config
                if vae_config.action_dim != 29 or vae_config.action_horizon != 16:
                    raise ValueError(
                        "RoboCasa latent training expects an ActionVAE over [16, 29], "
                        f"got [{vae_config.action_horizon}, {vae_config.action_dim}]"
                    )
                if vae_config.num_tasks > 0:
                    cache_metadata = action_vae_checkpoint.get(
                        "cache_metadata",
                        {},
                    )
                    checkpoint_datasets = cache_metadata.get("task_datasets")
                    _, expected_datasets = _match_action_vae_registry(
                        checkpoint_datasets,
                        vae_config.num_tasks,
                    )
                    if cache_metadata.get("normalization") == "per-task":
                        vae_action_low = torch.as_tensor(
                            cache_metadata.get("action_low"),
                            dtype=torch.float32,
                        )
                        vae_action_high = torch.as_tensor(
                            cache_metadata.get("action_high"),
                            dtype=torch.float32,
                        )
                        expected_shape = (len(expected_datasets), vae_config.action_dim)
                        if (
                            tuple(vae_action_low.shape) != expected_shape
                            or tuple(vae_action_high.shape) != expected_shape
                        ):
                            raise ValueError(
                                "Per-task ActionVAE bounds must both have shape "
                                f"{expected_shape}, got {tuple(vae_action_low.shape)} "
                                f"and {tuple(vae_action_high.shape)}"
                            )
                        statistics_path = getattr(
                            action_vae_config,
                            "policy_action_statistics_path",
                            None,
                        ) or cache_metadata.get("statistics_path")
                        if not statistics_path:
                            raise ValueError(
                                "Per-task ActionVAE requires policy-global action "
                                "statistics via framework.action_vae."
                                "policy_action_statistics_path or cache metadata"
                            )
                        statistics = json.loads(Path(statistics_path).read_text())
                        if "gr1" in statistics:
                            action_statistics = statistics["gr1"]["action"]
                        elif len(statistics) == 1:
                            action_statistics = next(iter(statistics.values()))[
                                "action"
                            ]
                        else:
                            raise ValueError(
                                f"Cannot select action statistics from {statistics_path}"
                            )
                        policy_action_low = torch.as_tensor(
                            action_statistics["min"],
                            dtype=torch.float32,
                        )
                        policy_action_high = torch.as_tensor(
                            action_statistics["max"],
                            dtype=torch.float32,
                        )
                        if (
                            tuple(policy_action_low.shape) != (vae_config.action_dim,)
                            or tuple(policy_action_high.shape) != (vae_config.action_dim,)
                        ):
                            raise ValueError(
                                "Policy-global ActionVAE bridge bounds must both have "
                                f"shape ({vae_config.action_dim},)"
                            )
                        self.register_buffer(
                            "vae_action_low",
                            vae_action_low,
                            persistent=True,
                        )
                        self.register_buffer(
                            "vae_action_high",
                            vae_action_high,
                            persistent=True,
                        )
                        self.register_buffer(
                            "policy_action_low",
                            policy_action_low,
                            persistent=True,
                        )
                        self.register_buffer(
                            "policy_action_high",
                            policy_action_high,
                            persistent=True,
                        )
                        self.use_per_task_vae_action_norm = True
                self.raw_action_horizon = vae_config.action_horizon
                if (
                    config.framework.action_model.action_dim != vae_config.latent_dim
                    or config.framework.action_model.action_horizon
                    != vae_config.num_tokens
                ):
                    raise ValueError(
                        "ActionDiT latent dimensions do not match the ActionVAE: "
                        f"head=[{config.framework.action_model.action_horizon}, "
                        f"{config.framework.action_model.action_dim}], "
                        f"vae=[{vae_config.num_tokens}, {vae_config.latent_dim}]"
                    )
                latent_mean = action_vae_checkpoint.get(
                    "latent_mean",
                    torch.zeros(vae_config.num_tokens, vae_config.latent_dim),
                ).float()
                latent_std = action_vae_checkpoint.get(
                    "latent_std",
                    torch.ones(vae_config.num_tokens, vae_config.latent_dim),
                ).float()
                self.register_buffer("action_latent_mean", latent_mean, persistent=True)
                self.register_buffer(
                    "action_latent_std", latent_std.clamp_min(1e-4), persistent=True
                )
                self.normalize_action_latents = bool(
                    action_vae_checkpoint.get("apply_latent_normalization", False)
                )
            else:
                self.action_vae = None
                self.normalize_action_latents = False
                self.use_per_task_vae_action_norm = False
                self.raw_action_horizon = self.future_action_window_size + 1
        else:
            # Video-only mode: skip action model entirely
            self.action_model = None
            self.future_action_window_size = 0
            self.past_action_window_size = 0
            self.chunk_len = 0
            self.use_action_vae = False
            self.action_vae = None
            self.normalize_action_latents = False
            self.use_per_task_vae_action_norm = False

    @staticmethod
    def _normalize_action_space(
        actions: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> torch.Tensor:
        span = high - low
        safe_span = torch.where(span.abs() > 1e-8, span, torch.ones_like(span))
        normalized = 2.0 * (actions - low) / safe_span - 1.0
        normalized = torch.where(
            (span.abs() > 1e-8).expand_as(normalized),
            normalized,
            torch.zeros_like(normalized),
        )
        return normalized.clamp(-1.0, 1.0)

    @staticmethod
    def _unnormalize_action_space(
        actions: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> torch.Tensor:
        return (actions.clamp(-1.0, 1.0) + 1.0) / 2.0 * (high - low) + low

    def _global_to_vae_actions(
        self,
        actions: torch.Tensor,
        task_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not getattr(self, "use_per_task_vae_action_norm", False):
            return actions
        if task_ids is None:
            raise ValueError(
                "Per-task ActionVAE normalization requires task_ids"
            )
        task_ids = task_ids.to(device=actions.device, dtype=torch.long)
        task_low = self.vae_action_low.to(actions)[task_ids].unsqueeze(1)
        task_high = self.vae_action_high.to(actions)[task_ids].unsqueeze(1)
        policy_low = self.policy_action_low.to(actions).view(1, 1, -1)
        policy_high = self.policy_action_high.to(actions).view(1, 1, -1)
        physical = self._unnormalize_action_space(
            actions,
            policy_low,
            policy_high,
        )
        return self._normalize_action_space(physical, task_low, task_high)

    def _vae_to_global_actions(
        self,
        actions: torch.Tensor,
        task_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not getattr(self, "use_per_task_vae_action_norm", False):
            return actions
        if task_ids is None:
            raise ValueError(
                "Per-task ActionVAE normalization requires task_ids"
            )
        task_ids = task_ids.to(device=actions.device, dtype=torch.long)
        task_low = self.vae_action_low.to(actions)[task_ids].unsqueeze(1)
        task_high = self.vae_action_high.to(actions)[task_ids].unsqueeze(1)
        policy_low = self.policy_action_low.to(actions).view(1, 1, -1)
        policy_high = self.policy_action_high.to(actions).view(1, 1, -1)
        physical = self._unnormalize_action_space(
            actions,
            task_low,
            task_high,
        )
        return self._normalize_action_space(
            physical,
            policy_low,
            policy_high,
        )

    def _encode_action_targets(
        self,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_action_vae:
            return actions, action_mask
        self.action_vae.eval()
        vae_dtype = next(self.action_vae.parameters()).dtype
        with torch.no_grad():
            vae_actions = self._global_to_vae_actions(
                actions[..., :29].to(dtype=vae_dtype),
                task_ids,
            )
            encoded = self.action_vae.encode(
                vae_actions,
                task_ids=task_ids,
            )
            latent = encoded["mu"] if "mu" in encoded else encoded["z"]
        if self.normalize_action_latents:
            latent = (latent - self.action_latent_mean.to(latent)) / (
                self.action_latent_std.to(latent)
            )
        active_steps = action_mask[..., :29].any(dim=-1)
        ratio = actions.shape[1] // latent.shape[1]
        latent_steps = active_steps.reshape(
            active_steps.shape[0], latent.shape[1], ratio
        ).any(dim=-1)
        latent_mask = latent_steps.unsqueeze(-1).expand_as(latent)
        return latent.to(dtype=actions.dtype), latent_mask

    def _decode_action_predictions(
        self,
        latent: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.use_action_vae:
            return latent
        self.action_vae.eval()
        vae_dtype = next(self.action_vae.parameters()).dtype
        latent = latent.to(dtype=vae_dtype)
        if self.normalize_action_latents:
            latent = latent * self.action_latent_std.to(latent) + (
                self.action_latent_mean.to(latent)
            )
        decoded = self.action_vae.decode(latent, task_ids=task_ids)
        decoded = self._vae_to_global_actions(decoded, task_ids)
        return F.pad(decoded, (0, 3))
        

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  #  [B, [frame_0, frame_1, ..., frame_T-1]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        # Step 1: backbone input format
        # All video frames (condition + future) are already in batch_images;
        # build_cosmos_inputs splits them into videos (cond) and future_videos internally.
        backbone_inputs = self.backbone_interface.build_cosmos_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_outputs = self.backbone_interface(
                **backbone_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            if not self.video_fm_only:
                last_hidden = backbone_outputs.hidden_states[-1]  # [B, L, H] ##2560-4b
            else:
                last_hidden = None
            future_video_loss = getattr(backbone_outputs, "future_video_loss", None)

        # Video-only FM training: no action branch.
        if self.video_fm_only:
            if future_video_loss is None:
                raise ValueError(
                    "video_fm_only is enabled (cosmos25.training='video') but `future_video_loss` is None. "
                    "Please provide `image_next` (or future_images) and set "
                    "`framework.cosmos25.future_loss_type=flow_matching`."
                )
            return {"future_video_loss": future_video_loss}

        actions = [example["action"] for example in examples]  # label [B， len, 7]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        action_mask = [example["action_mask"] for example in examples]  # [B, len, action_dim]
        task_ids = None
        if self.use_action_vae and self.action_vae.config.num_tasks > 0:
            missing = [index for index, example in enumerate(examples) if "task_id" not in example]
            if missing:
                raise ValueError(
                    "Task-conditioned ActionVAE requires task_id in every example; "
                    f"missing batch indices: {missing}"
                )
            task_ids = torch.as_tensor(
                [example["task_id"] for example in examples],
                device=last_hidden.device,
                dtype=torch.long,
            )

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.raw_action_horizon :, :]
            action_mask = torch.from_numpy(np.stack(action_mask)).to(last_hidden.device)
            action_mask = action_mask[:, -self.raw_action_horizon :, :]
            actions_target, action_mask = self._encode_action_targets(
                actions_target,
                action_mask,
                task_ids=task_ids,
            )

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            action_mask = action_mask.repeat(repeated_diffusion_steps, 1, 1)
            ###no state
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(last_hidden_repeated, actions_target_repeated, action_mask, state_repeated)  # (B, chunk_len, action_dim)

        out = {"action_loss": action_loss}
        if future_video_loss is not None:
            out["future_video_loss"] = future_video_loss
        return out

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with backbone (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = []
        for ex in examples:
            img = ex["image"]
            if isinstance(img, (list, tuple)) and len(img) > 0:
                batch_images.append(img)
            else:
                batch_images.append([img])
        instructions = [example["lang"] for example in examples]  # [B, str]
        task_ids = None
        if self.use_action_vae and self.action_vae.config.num_tasks > 0:
            missing = [index for index, example in enumerate(examples) if "task_id" not in example]
            if missing:
                raise ValueError(
                    "Task-conditioned ActionVAE requires task_id in every example; "
                    f"missing batch indices: {missing}"
                )
            task_ids = torch.as_tensor(
                [example["task_id"] for example in examples],
                device=next(self.action_vae.parameters()).device,
                dtype=torch.long,
            )
    
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
    
        # Step 1: backbone input format
        backbone_inputs = self.backbone_interface.build_cosmos_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_outputs = self.backbone_interface(
                **backbone_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = backbone_outputs.hidden_states[-1]   # [B, L, H]

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        
        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)  # (B, chunk_len, action_dim)
            pred_actions = self._decode_action_predictions(
                pred_actions,
                task_ids=task_ids,
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}



