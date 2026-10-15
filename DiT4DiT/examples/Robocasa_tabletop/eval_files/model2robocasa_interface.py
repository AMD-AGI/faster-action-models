# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
from collections import deque
from typing import Optional, Sequence
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

from examples.Robocasa_tabletop.eval_files.adaptive_ensemble import AdaptiveEnsembler
from typing import Dict
import numpy as np
from pathlib import Path

from DiT4DiT.model.framework.share_tools import read_mode_config
from DiT4DiT.model.modules.action_model.action_vae import load_action_vae
from DiT4DiT.dataloader.gr00t_lerobot.full24_tasks import FULL24_TASKS
from DiT4DiT.dataloader.gr00t_lerobot.pnp6_tasks import PNP6_TASKS

CEILING_TASK_REGISTRIES = {
    "pnp6": PNP6_TASKS,
    "full24": FULL24_TASKS,
}


def match_ceiling_task_registry(metadata: dict, num_tasks: int):
    """Return the canonical registry named by checkpoint cache metadata."""
    task_datasets = metadata.get("task_datasets")
    matches = [
        (name, tasks)
        for name, tasks in CEILING_TASK_REGISTRIES.items()
        if list(task_datasets or ()) == [task.dataset_name for task in tasks]
        and (num_tasks == 0 or num_tasks == len(tasks))
    ]
    if len(matches) != 1:
        supported = {
            name: len(tasks) for name, tasks in CEILING_TASK_REGISTRIES.items()
        }
        raise ValueError(
            "Ceiling VAE cache task registry does not match a canonical "
            f"registry ({supported})"
        )
    return matches[0]



class PolicyWarper:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble = False, 
        action_ensemble_horizon: Optional[int] = 3, 
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha = 0.1,
        host="0.0.0.0",
        port=10095,
        n_action_steps=2,
        ceiling_vae_path: Optional[str] = None,
        task_id: Optional[int] = None,
    ) -> None:
        
        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon #0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.n_action_steps = n_action_steps
        self.task_id = task_id
        self.ceiling_vae = None
        self.ceiling_vae_action_stats = None
        self.ceiling_registry_name = None
        self.ceiling_tasks = None
        if ceiling_vae_path:
            self.ceiling_vae, checkpoint = load_action_vae(ceiling_vae_path)
            self.ceiling_vae.eval()
            self.ceiling_vae.requires_grad_(False)
            metadata = checkpoint.get("cache_metadata", {})
            if "action_low" not in metadata or "action_high" not in metadata:
                raise ValueError(
                    "Ceiling VAE checkpoint is missing cache normalization metadata"
                )
            self.ceiling_vae_action_stats = {
                "min": np.asarray(metadata["action_low"], dtype=np.float32),
                "max": np.asarray(metadata["action_high"], dtype=np.float32),
            }
            self.ceiling_registry_name, self.ceiling_tasks = (
                match_ceiling_task_registry(
                    metadata,
                    self.ceiling_vae.config.num_tasks,
                )
            )
            uses_family_bounds = self.ceiling_vae_action_stats["min"].ndim == 2
            if self.ceiling_vae.config.num_tasks > 0 or uses_family_bounds:
                if self.task_id is None:
                    raise ValueError(
                        "A task_id is required for a task-conditioned or "
                        "per-family-normalized ceiling VAE"
                    )
                if not 0 <= self.task_id < len(self.ceiling_tasks):
                    raise ValueError(
                        f"Invalid {self.ceiling_registry_name} task_id: "
                        f"{self.task_id}"
                    )
            print(f"*** ae_execute ceiling enabled: {ceiling_vae_path} ***")

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str or tuple) -> None:
       
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def _build_vla_example(
        self,
        images: list[np.ndarray],
        instruction: str,
        input_state: np.ndarray,
    ) -> dict:
        example = {
            "image": images,
            "lang": instruction,
            "state": np.pad(
                input_state,
                ((0, 0), (0, 64 - input_state.shape[1])),
                "constant",
            ),
        }
        if self.task_id is not None:
            example["task_id"] = self.task_id
        return example

    def _ceiling_stats_for_task(self) -> dict[str, np.ndarray]:
        stats = self.ceiling_vae_action_stats
        if stats["min"].ndim == 1:
            return stats
        if self.ceiling_tasks is None or self.task_id is None:
            raise ValueError(
                "Per-family ceiling normalization requires a matched registry "
                "and task_id"
            )
        expected_shape = (len(self.ceiling_tasks), 29)
        if (
            stats["min"].shape != expected_shape
            or stats["max"].shape != expected_shape
        ):
            raise ValueError(
                "Per-family ceiling normalization must have shape "
                f"{list(expected_shape)} for {self.ceiling_registry_name}"
            )
        return {
            "min": stats["min"][self.task_id],
            "max": stats["max"][self.task_id],
        }

    def _apply_ceiling_vae(self, normalized_actions: np.ndarray) -> np.ndarray:
        if self.ceiling_vae is None:
            return normalized_actions
        ceiling_stats = self._ceiling_stats_for_task()
        physical_actions = self.unnormalize_actions(
            normalized_actions[:, :, :29],
            self.action_norm_stats,
        )
        vae_normalized_actions = self.normalize_actions(
            physical_actions,
            ceiling_stats,
        )
        active_actions = torch.as_tensor(
            vae_normalized_actions,
            dtype=torch.float32,
        )
        task_ids = None
        if self.ceiling_vae.config.num_tasks > 0:
            task_ids = torch.full(
                (active_actions.shape[0],),
                self.task_id,
                dtype=torch.long,
            )
        with torch.inference_mode():
            encoded = self.ceiling_vae.encode(
                active_actions,
                task_ids=task_ids,
            )
            active_actions = self.ceiling_vae.decode(
                encoded["mu"] if "mu" in encoded else encoded["z"],
                task_ids=task_ids,
            )
        reconstructed_physical_actions = self.unnormalize_actions(
            active_actions.numpy(),
            ceiling_stats,
        )
        reconstructed = normalized_actions.copy()
        reconstructed[:, :, :29] = self.normalize_actions(
            reconstructed_physical_actions,
            self.action_norm_stats,
        )
        return reconstructed


    def step(
        self, 
        observations,
        **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        执行一步推理
        :param image: 输入图像 (H, W, 3) uint8格式
        :param task_description: 任务描述文本
        :return: (原始动作, 处理后的动作)
        """

        task_description = observations['annotation.human.coarse_action'][0] # tuple       
        ego_view = observations['video.ego_view']  # (N, 1, H, W, 3)
        images = ego_view
        state = {}
        state['left_arm'] = observations['state.left_arm']     
        state['right_arm'] = observations['state.right_arm']             # (N, 1, 7)
        state['left_hand'] = observations['state.left_hand']              # (N, 1, 6)
        state['right_hand'] = observations['state.right_hand']            # (N, 1, 6)
        state['waist'] = observations['state.waist']                      # (N, 1, 3)
        

        state = self.normalize_state(state)
        input_state = []
        for key in state.keys():
            input_state.append(state[key])
        input_state = np.concatenate(input_state, axis=-1)

        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        # image: Image.Image = Image.fromarray(image)
        images = [[self._resize_image(img) for img in sample] for sample in images] # (B, N_view, H, W, 3)
        input_state = [input_s for input_s in input_state] # B, state_dim*(sin, cos)

        # prepare vla input
        examples = []
        batch_size = len(images)
        # Ensure instructions list has the same length as batch_size
        if isinstance(self.task_description, str):
            # If task_description is a string, repeat it for all batch elements
            instructions = [self.task_description] * batch_size
        elif isinstance(self.task_description, list):
            if len(self.task_description) == batch_size:
                instructions = self.task_description
            elif len(self.task_description) == 1:
                # If only one instruction, repeat it for all batch elements
                instructions = self.task_description * batch_size
            elif len(self.task_description) > batch_size:
                # If more instructions than batch_size, take first batch_size elements
                instructions = self.task_description[:batch_size]
            else:
                # If fewer instructions than batch_size, pad with last element
                instructions = self.task_description + [self.task_description[-1]] * (batch_size - len(self.task_description))
        else:
            # Fallback: treat as single instruction and repeat
            instructions = [self.task_description] * batch_size

        for b in range(batch_size):
            examples.append(
                self._build_vla_example(
                    images[b],
                    instructions[b],
                    input_state[b],
                )
            )
        
        vla_input = {
            "examples": examples,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        
        response = self.client.predict_action(vla_input)

        # unnormalize the action
        normalized_actions = response["data"]["normalized_actions"] # B, chunk, D        
        normalized_actions = self._apply_ceiling_vae(normalized_actions)
        
        # unnormalize actions in batch form
        raw_actions = self.unnormalize_actions(normalized_actions=normalized_actions[:,:,:29], action_norm_stats=self.action_norm_stats)
        
        # raw_actions shape: (B, chunk, D)
        if self.action_ensemble:
            # 对batch中的每个样本进行ensemble
            batch_size = raw_actions.shape[0]
            ensembled_actions = []
            for b in range(batch_size):
                ensembled = self.action_ensembler.ensemble_action(raw_actions[b])[None]  # (1, D)
                ensembled_actions.append(ensembled)
            raw_actions = np.stack(ensembled_actions, axis=0)  # (B, 1, D)

        raw_action = {
            "action.left_arm": raw_actions[:, :self.n_action_steps, :7],      # (B, n_action_steps, 7)
            "action.right_arm": raw_actions[:, :self.n_action_steps, 7:14],   # (B, n_action_steps, 7)
            "action.left_hand": raw_actions[:, :self.n_action_steps, 14:20],  # (B, n_action_steps, 6)
            "action.right_hand": raw_actions[:, :self.n_action_steps, 20:26], # (B, n_action_steps, 6)
            "action.waist": raw_actions[:, :self.n_action_steps, 26:29],      # (B, n_action_steps, 3)
        }

        return {"actions": raw_action}

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Args:
            normalized_actions: shape (B, chunk, D) (chunk, D)
            action_norm_stats:
        Returns:
            actions
        """
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        
        normalized_actions = np.clip(normalized_actions, -1, 1)
        
        actions = np.where(
            mask,
            (normalized_actions + 1) / 2 * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    @staticmethod
    def normalize_actions(
        actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]
    ) -> np.ndarray:
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["min"], dtype=bool)
        )
        action_high = np.asarray(action_norm_stats["max"])
        action_low = np.asarray(action_norm_stats["min"])
        span = action_high - action_low
        safe_span = np.where(np.abs(span) > 1e-8, span, 1.0)
        normalized = np.where(
            mask,
            2.0 * (actions - action_low) / safe_span - 1.0,
            actions,
        )
        normalized = np.where(
            mask & (np.abs(span) <= 1e-8),
            0.0,
            normalized,
        )
        return np.clip(normalized, -1.0, 1.0)

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = PolicyWarper._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]



    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
    
    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
    
    def normalize_state(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Normalize the state
        """
        for key in state.keys():
            sin_state = np.sin(state[key])
            cos_state = np.cos(state[key])
            state[key] = np.concatenate([sin_state, cos_state], axis=-1)
        return state
    
    