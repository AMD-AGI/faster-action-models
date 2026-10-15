#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize a schema-2 RoboCasa full-24 policy evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from DiT4DiT.dataloader.gr00t_lerobot.full24_tasks import FULL24_TASKS
from examples.Robocasa_tabletop.eval_files.full24_rollout_matching import (
    ROLLOUT_MATCHING_PROTOCOL,
    full24_episode_seed,
)

EXPECTED_RUNTIME = {
    "mujoco": "3.2.6",
    "robosuite": "1.5.1",
    "numpy": "1.26.4",
}


def validate_and_summarize(
    run_dir: Path,
    *,
    expected_successes: int | None = None,
    expected_episodes_per_task: int = 50,
    expected_seed: int = 7,
) -> dict:
    protocol = json.loads((run_dir / "protocol.json").read_text())
    if protocol.get("schema_version") != 2:
        raise ValueError("protocol must use schema_version 2")
    if protocol.get("registry") != "full24":
        raise ValueError("protocol registry must be full24")
    if protocol.get("condition") != "direct":
        raise ValueError("policy result must use direct execution")
    if protocol.get("episodes_per_task") != expected_episodes_per_task:
        raise ValueError("unexpected episodes_per_task")
    if (
        protocol.get("policy_seed") != expected_seed
        or protocol.get("eval_seed") != expected_seed
    ):
        raise ValueError("unexpected policy/evaluation seed")
    if protocol.get("rollout_matching") != ROLLOUT_MATCHING_PROTOCOL:
        raise ValueError("unexpected rollout-matching protocol")
    if protocol.get("max_episode_steps") != 720:
        raise ValueError("max_episode_steps must be 720")
    if protocol.get("n_action_steps") != 12:
        raise ValueError("n_action_steps must be 12")
    versions = protocol.get("runtime_versions", {})
    for package, version in EXPECTED_RUNTIME.items():
        if versions.get(package) != version:
            raise ValueError(
                f"unexpected {package} version: {versions.get(package)!r}"
            )

    task_rows = []
    total_successes = 0
    total_episodes = 0
    result_dir = run_dir / "results"
    expected_names = {f"{task.label}.json" for task in FULL24_TASKS}
    actual_names = {path.name for path in result_dir.glob("*.json")}
    if actual_names != expected_names:
        raise ValueError(
            f"result files differ: missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )

    for task in FULL24_TASKS:
        payload = json.loads((result_dir / f"{task.label}.json").read_text())
        if payload.get("schema_version") != 2:
            raise ValueError(f"{task.label}: schema_version must be 2")
        for field in ("task_id", "label", "env_name", "group"):
            if payload.get(field) != getattr(task, field):
                raise ValueError(f"{task.label}: unexpected {field}")
        if payload.get("seed") != expected_seed:
            raise ValueError(f"{task.label}: unexpected seed")
        if payload.get("rollout_matching") != ROLLOUT_MATCHING_PROTOCOL:
            raise ValueError(f"{task.label}: unexpected rollout protocol")
        episodes = payload.get("episodes", [])
        if len(episodes) != expected_episodes_per_task:
            raise ValueError(f"{task.label}: unexpected episode count")
        for index, episode in enumerate(episodes):
            if episode.get("episode_index") != index:
                raise ValueError(f"{task.label}: non-canonical episode index")
            expected_episode_seed = full24_episode_seed(
                expected_seed, task.env_name, index
            )
            if episode.get("episode_seed") != expected_episode_seed:
                raise ValueError(f"{task.label}: unexpected episode seed")
            if not episode.get("instruction"):
                raise ValueError(f"{task.label}: missing instruction")
            fingerprint = episode.get("initial_observation_fingerprint", "")
            if not fingerprint.startswith("sha256:") or len(fingerprint) != 71:
                raise ValueError(f"{task.label}: invalid observation fingerprint")
        successes = sum(bool(episode.get("success")) for episode in episodes)
        total_successes += successes
        total_episodes += len(episodes)
        task_rows.append(
            {
                "label": task.label,
                "successes": successes,
                "episodes": len(episodes),
                "success_rate": successes / len(episodes),
            }
        )

    if expected_successes is not None and total_successes != expected_successes:
        raise ValueError(
            f"expected {expected_successes} successes, got {total_successes}"
        )
    return {
        "successes": total_successes,
        "episodes": total_episodes,
        "success_rate": total_successes / total_episodes,
        "tasks": task_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-successes", type=int)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    summary = validate_and_summarize(
        args.run_dir,
        expected_successes=args.expected_successes,
        expected_episodes_per_task=args.episodes_per_task,
        expected_seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
