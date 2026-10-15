# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from DiT4DiT.dataloader.gr00t_lerobot.full24_tasks import FULL24_TASKS
from examples.Robocasa_tabletop.eval_files.full24_rollout_matching import (
    full24_episode_seed,
    run_matched_full24_episodes,
)


class FakeMatchedVectorEnv:
    def __init__(self, trajectory_lengths):
        self.trajectory_lengths = trajectory_lengths
        self.reset_seeds = []
        self.episode = -1
        self.step_count = 0

    def reset(self, *, seed):
        self.reset_seeds.append(seed)
        self.episode += 1
        self.step_count = 0
        return (
            {
                "state.robot": np.asarray(
                    [[seed % 997, seed % 101]], dtype=np.int64
                ),
                "video.ego_view": np.full(
                    (1, 1, 2, 2, 3), seed % 251, dtype=np.uint8
                ),
                "annotation.human.coarse_action": np.asarray(
                    [f"instruction-{seed % 89}"], dtype=object
                ),
            },
            {},
        )

    def step(self, _actions):
        self.step_count += 1
        done = self.step_count >= self.trajectory_lengths[self.episode]
        return (
            {},
            np.asarray([0.0]),
            np.asarray([done]),
            np.asarray([False]),
            {"success": np.asarray([[False]])},
        )


def test_full24_rollouts_reset_and_match_each_episode():
    task = FULL24_TASKS[0]
    direct_env = FakeMatchedVectorEnv([1, 3, 2])
    candidate_env = FakeMatchedVectorEnv([3, 1, 4])

    _, direct_records = run_matched_full24_episodes(
        env=direct_env,
        get_actions=lambda _obs: {},
        env_name=task.env_name,
        n_episodes=3,
        base_eval_seed=23,
    )
    _, candidate_records = run_matched_full24_episodes(
        env=candidate_env,
        get_actions=lambda _obs: {},
        env_name=task.env_name,
        n_episodes=3,
        base_eval_seed=23,
    )

    expected_seeds = [
        full24_episode_seed(23, task.env_name, index)
        for index in range(3)
    ]
    assert direct_env.reset_seeds == candidate_env.reset_seeds == expected_seeds
    assert len(set(expected_seeds)) == 3
    for direct, candidate in zip(direct_records, candidate_records):
        assert direct["episode_seed"] == candidate["episode_seed"]
        assert direct["instruction"] == candidate["instruction"]
        assert (
            direct["initial_observation_fingerprint"]
            == candidate["initial_observation_fingerprint"]
        )
    assert [row["length"] for row in direct_records] != [
        row["length"] for row in candidate_records
    ]
