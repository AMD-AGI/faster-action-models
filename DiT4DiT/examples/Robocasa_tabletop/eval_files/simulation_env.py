# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by Advanced Micro Devices, Inc. Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import tyro
import dataclasses
import json
import logging
import os

# Required for robocasa environments
import robocasa  # noqa: F401
import robosuite  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
from examples.Robocasa_tabletop.eval_files.base_config import ModalityConfig
from examples.Robocasa_tabletop.eval_files.base_config import BasePolicy
from examples.Robocasa_tabletop.eval_files.wrappers.multistep_wrapper import MultiStepWrapper
from examples.Robocasa_tabletop.eval_files.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)

from examples.Robocasa_tabletop.eval_files.model2robocasa_interface import PolicyWarper
from DiT4DiT.dataloader.gr00t_lerobot.full24_tasks import (
    FULL24_ENV_TO_TASK_ID,
    FULL24_TASKS,
)
from DiT4DiT.dataloader.gr00t_lerobot.pnp6_tasks import PNP6_ENV_TO_TASK_ID
from examples.Robocasa_tabletop.eval_files.full24_rollout_matching import (
    ROLLOUT_MATCHING_PROTOCOL,
    run_matched_full24_episodes,
)


@dataclass
class VideoConfig:
    """Configuration for video recording settings."""

    video_dir: Optional[str] = None
    steps_per_render: int = 2 # 和10 是什么关系
    fps: int = 10 # BUG 数据集中应该是 20？
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings."""

    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 1440


@dataclass
class SimulationConfig:
    """Main configuration for simulation environment."""

    env_name: str
    n_episodes: int = 2
    n_envs: int = 1
    seed: int = 7
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


class SimulationInferenceEnv:
    """Client for running simulations with a model."""

    def __init__(self, model: Optional[BasePolicy] = None):
        """Initialize the simulation client with a model."""
        self.model = model
        self.env = None
        self.last_episode_records: List[Dict[str, Any]] = []

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get action from the model based on observations."""
        # NOTE(YL)!
        # hot fix to change the video.ego_view_bg_crop_pad_res256_freq20 to video.ego_view
        if "video.ego_view_bg_crop_pad_res256_freq20" in observations:
            observations["video.ego_view"] = observations.pop(
                "video.ego_view_bg_crop_pad_res256_freq20"
            )
        return self.model.step(observations)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """Get modality configuration from the model."""
        return self.model.get_modality_config()

    def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
        """Set up the simulation environment based on the provided configuration."""
        # Create environment functions for each parallel environment
        env_fns = [partial(_create_single_env, config=config, idx=i) for i in range(config.n_envs)]
        # Create vector environment (sync for single env, async for multiple)
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        else:
            return gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )

    def run_simulation(self, config: SimulationConfig, model: Optional[BasePolicy] = None) -> Tuple[str, List[bool]]:
        """Run the simulation for the specified number of episodes.
        
        Args:
            config: Configuration for the simulation
            model: The model to use for inference. If None, uses the model from __init__
        """
        # Use the provided model or fall back to the instance model
        if model is not None:
            self.model = model
        
        if self.model is None:
            raise ValueError("No model provided. Please provide a model either in __init__ or run_simulation")
        is_full24 = config.env_name in FULL24_ENV_TO_TASK_ID
        if is_full24 and config.n_envs != 1:
            raise ValueError(
                "Matched full24 evaluation requires n_envs=1 so every "
                "episode can be explicitly reset with its derived seed"
            )
        
        start_time = time.time()
        self.last_episode_records = []
        print(
            f"Running {config.n_episodes} episodes for {config.env_name} with {config.n_envs} environments"
        )
        # Set up the environment
        self.env = self.setup_environment(config)
        if is_full24:
            episode_successes, episode_lengths = (
                self._run_matched_full24_episodes(config)
            )
        else:
            episode_successes, episode_lengths = self._run_legacy_episodes(config)
        # Clean up
        self.env.close()
        self.env = None
        print(
            f"Collecting {config.n_episodes} episodes took {time.time() - start_time:.2f} seconds"
        )
        assert (
            len(episode_successes) >= config.n_episodes
        ), f"Expected at least {config.n_episodes} episodes, got {len(episode_successes)}"
        return config.env_name, episode_successes

    def _run_matched_full24_episodes(
        self, config: SimulationConfig
    ) -> Tuple[List[bool], List[int]]:
        """Run full24 episodes with an explicit deterministic reset each time."""
        episode_successes, records = run_matched_full24_episodes(
            env=self.env,
            get_actions=self._get_actions_from_model,
            env_name=config.env_name,
            n_episodes=config.n_episodes,
            base_eval_seed=config.seed,
        )
        self.last_episode_records.extend(records)
        return episode_successes, [record["length"] for record in records]

    def _run_legacy_episodes(
        self, config: SimulationConfig
    ) -> Tuple[List[bool], List[int]]:
        """Preserve the prior vectorized behavior for non-full24 evaluations."""
        # Initialize tracking variables
        episode_lengths = []
        current_rewards = [0] * config.n_envs
        current_lengths = [0] * config.n_envs
        completed_episodes = 0
        current_successes = [False] * config.n_envs
        episode_successes = []
        # Initial environment reset
        obs, _ = self.env.reset(seed=config.seed)
        current_instructions = [
            _instruction_for_env(obs, env_idx)
            for env_idx in range(config.n_envs)
        ]
        # Main simulation loop
        while completed_episodes < config.n_episodes:
            # Process observations and get actions from the model
            actions = self._get_actions_from_model(obs)
            # Step the environment
            next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)
            # Update episode tracking
            for env_idx in range(config.n_envs):
                current_successes[env_idx] |= bool(env_infos["success"][env_idx][0])
                current_rewards[env_idx] += rewards[env_idx]
                current_lengths[env_idx] += 1
                # If episode ended, store results
                if terminations[env_idx] or truncations[env_idx]:
                    if completed_episodes >= config.n_episodes:
                        continue
                    episode_lengths.append(current_lengths[env_idx])
                    episode_successes.append(current_successes[env_idx])
                    self.last_episode_records.append(
                        {
                            "episode_index": completed_episodes,
                            "env_index": env_idx,
                            "instruction": current_instructions[env_idx],
                            "success": bool(current_successes[env_idx]),
                            "length": current_lengths[env_idx],
                        }
                    )
                    current_successes[env_idx] = False
                    completed_episodes += 1
                    # Reset trackers for this environment
                    current_rewards[env_idx] = 0
                    current_lengths[env_idx] = 0
            obs = next_obs
            current_instructions = [
                _instruction_for_env(obs, env_idx)
                for env_idx in range(config.n_envs)
            ]
        return episode_successes, episode_lengths

    def _get_actions_from_model(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Process observations and get actions from the model."""
        # Get actions from the model
        action_dict = self.get_action(observations)
        # Extract actions from the response
        if "actions" in action_dict:
            actions = action_dict["actions"]
        else:
            actions = action_dict
        # Add batch dimension to actions
        return actions


def _instruction_for_env(observations: Dict[str, Any], env_idx: int) -> str:
    """Extract a stable, JSON-safe instruction from vector observations."""
    value = observations.get("annotation.human.coarse_action")
    if value is None:
        return ""
    try:
        value = value[env_idx]
    except (IndexError, KeyError, TypeError):
        pass
    while isinstance(value, (list, tuple, np.ndarray)) and len(value) == 1:
        value = value[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        flattened = np.asarray(value, dtype=object).reshape(-1).tolist()
        return " ".join(str(item) for item in flattened if str(item))
    return str(value)


def _create_single_env(config: SimulationConfig, idx: int) -> gym.Env:
    """Create a single environment with appropriate wrappers."""
    # Create base environment
    env = gym.make(config.env_name, enable_render=True)
    # Add video recording wrapper if needed (only for the first environment)
    if config.video.video_dir is not None:
        video_recorder = VideoRecorder.create_h264(
            fps=config.video.fps,
            codec=config.video.codec,
            input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf,
            thread_type=config.video.thread_type,
            thread_count=config.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(config.video.video_dir),
            steps_per_render=config.video.steps_per_render,
        )
    # Add multi-step wrapper
    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
    )
    return env


def run_evaluation(
    env_name: str,
    model: BasePolicy,
    video_dir: Optional[str] = None,
    n_episodes: int = 2,
    n_envs: int = 1,
    n_action_steps: int = 2,
    max_episode_steps: int = 100,
    seed: int = 7,
    result_out_path: Optional[str] = None,
) -> Tuple[str, List[bool]]:
    """
    Simple entry point to run a simulation evaluation.
    Args:
        env_name: Name of the environment to run
        model: The model to use for inference
        video_dir: Directory to save videos (None for no videos)
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        n_action_steps: Number of action steps per environment step
        max_episode_steps: Maximum number of steps per episode
    Returns:
        Tuple of environment name and list of episode success flags
    """
    # Create configuration
    config = SimulationConfig(
        env_name=env_name,
        n_episodes=n_episodes,
        n_envs=n_envs,
        seed=seed,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps, max_episode_steps=max_episode_steps
        ),
    )
    # Create client and run simulation
    client = SimulationInferenceEnv(model=model)
    results = client.run_simulation(config)
    # Print results
    print(f"Results for {env_name}:")
    print(f"Success rate: {np.mean(results[1]):.2f}")
    print(f"Episode successes: {json.dumps([bool(value) for value in results[1]])}")
    if result_out_path is not None:
        task_id = FULL24_ENV_TO_TASK_ID.get(
            env_name, PNP6_ENV_TO_TASK_ID.get(env_name)
        )
        task = FULL24_TASKS[task_id] if task_id is not None else None
        payload = {
            "schema_version": 2,
            "env_name": env_name,
            "task_id": task_id,
            "label": task.label if task is not None else None,
            "group": task.group if task is not None else None,
            "seed": seed,
            "rollout_matching": ROLLOUT_MATCHING_PROTOCOL,
            "episodes": client.last_episode_records,
            "success_rate": float(np.mean(results[1])),
        }
        output = Path(result_out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(output)
        print(f"Machine-readable results: {output}")
    return results


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 5678
    resize_size = [224,224]

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    env_name: str = "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    n_episodes: int = 50  # Number of steps to wait for objects to stabilize i n sim
    n_envs: int = 1  # Number of rollouts per task
    max_episode_steps: int = 360 # 
    n_action_steps: int = 3

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: Optional[str] = None

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = "results/Checkpoints/1029_qwenGR00T_fourier_gr1_unified_1000_PnPMilkToMicrowaveClose_gpus_woPretrain_wState/checkpoints/steps_20000_pytorch_model.pt"
    ceiling_vae_path: Optional[str] = None
    result_out_path: Optional[str] = None



def eval_gr1_unified(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    task_id = FULL24_ENV_TO_TASK_ID.get(
        args.env_name, PNP6_ENV_TO_TASK_ID.get(args.env_name)
    )
    model = PolicyWarper(
        policy_ckpt_path=args.pretrained_path, # to get unnormalization stats
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        n_action_steps=args.n_action_steps,
        ceiling_vae_path=args.ceiling_vae_path,
        task_id=task_id,
    )
    run_evaluation(
        env_name=args.env_name,
        model=model,
        video_dir=args.video_out_path,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        result_out_path=args.result_out_path,
    )

if __name__ == "__main__":
    tyro.cli(eval_gr1_unified)