# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import json

import pytest

from DiT4DiT.dataloader.gr00t_lerobot.full24_tasks import FULL24_TASKS
from examples.Robocasa_tabletop.eval_files.full24_rollout_matching import (
    ROLLOUT_MATCHING_PROTOCOL,
    full24_episode_seed,
)
from examples.Robocasa_tabletop.eval_files.summarize_full24_policy import (
    validate_and_summarize,
)


def write_run(root, successes=600):
    protocol = {
        "schema_version": 2,
        "registry": "full24",
        "condition": "direct",
        "episodes_per_task": 50,
        "max_episode_steps": 720,
        "n_action_steps": 12,
        "policy_seed": 7,
        "eval_seed": 7,
        "rollout_matching": ROLLOUT_MATCHING_PROTOCOL,
        "runtime_versions": {
            "mujoco": "3.2.6",
            "robosuite": "1.5.1",
            "numpy": "1.26.4",
        },
        "tasks": [dataclasses.asdict(task) for task in FULL24_TASKS],
    }
    (root / "protocol.json").write_text(json.dumps(protocol) + "\n")
    result_dir = root / "results"
    result_dir.mkdir()
    remaining = successes
    for task in FULL24_TASKS:
        task_successes = min(50, remaining)
        remaining -= task_successes
        episodes = [
            {
                "episode_index": index,
                "episode_seed": full24_episode_seed(7, task.env_name, index),
                "instruction": f"instruction {index}",
                "initial_observation_fingerprint": "sha256:" + "a" * 64,
                "success": index < task_successes,
            }
            for index in range(50)
        ]
        payload = {
            "schema_version": 2,
            "task_id": task.task_id,
            "label": task.label,
            "env_name": task.env_name,
            "group": task.group,
            "seed": 7,
            "rollout_matching": ROLLOUT_MATCHING_PROTOCOL,
            "episodes": episodes,
        }
        (result_dir / f"{task.label}.json").write_text(
            json.dumps(payload) + "\n"
        )
    assert remaining == 0


def test_summary_validates_expected_result(tmp_path):
    write_run(tmp_path)
    summary = validate_and_summarize(tmp_path, expected_successes=600)
    assert summary["successes"] == 600
    assert summary["episodes"] == 1_200
    assert summary["success_rate"] == pytest.approx(0.5)


def test_summary_rejects_wrong_expected_result(tmp_path):
    write_run(tmp_path)
    with pytest.raises(ValueError, match="expected 700 successes"):
        validate_and_summarize(tmp_path, expected_successes=700)
