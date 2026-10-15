# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic rollout identity for matched full-24 evaluations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from DiT4DiT.dataloader.gr00t_lerobot.full24_tasks import (
    FULL24_ENV_TO_TASK_ID,
    FULL24_TASKS,
)

ROLLOUT_MATCHING_STRATEGY = "explicit-per-episode-reset"
ROLLOUT_MATCHING_VERSION = 2
ROLLOUT_MATCHING_PROTOCOL = {
    "strategy": ROLLOUT_MATCHING_STRATEGY,
    "version": ROLLOUT_MATCHING_VERSION,
    "seed_derivation": (
        "sha256(base_eval_seed,family_id,env_name,episode_index)->uint32; "
        "reseed numpy, Gymnasium, and RoboCasa env.rng"
    ),
    "initial_observation_fingerprint": "sha256-canonical-observation-v1",
}


def robosuite_overlay_identity(path: Path) -> dict[str, str]:
    """Bind the simulator overlay to its package version and RECORD contents."""
    resolved = path.resolve()
    dist_infos = sorted(resolved.glob("robosuite-*.dist-info"))
    if len(dist_infos) != 1:
        raise ValueError(
            f"Expected one robosuite dist-info under {resolved}, got {dist_infos}"
        )
    metadata = (dist_infos[0] / "METADATA").read_text()
    version = next(
        line.split(":", 1)[1].strip()
        for line in metadata.splitlines()
        if line.startswith("Version:")
    )
    record = dist_infos[0] / "RECORD"
    return {
        "path": str(resolved),
        "version": version,
        "record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
    }


def full24_episode_seed(
    base_eval_seed: int,
    env_name: str,
    episode_index: int,
) -> int:
    """Derive a stable uint32 seed for one full-24 episode."""
    try:
        task = FULL24_TASKS[FULL24_ENV_TO_TASK_ID[env_name]]
    except KeyError as exc:
        raise ValueError(f"Not a canonical full24 environment: {env_name}") from exc
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    identity = {
        "base_eval_seed": int(base_eval_seed),
        "episode_index": int(episode_index),
        "env_name": env_name,
        "family_id": task.family_id,
        "strategy": ROLLOUT_MATCHING_STRATEGY,
        "version": ROLLOUT_MATCHING_VERSION,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _update_hash(hasher: Any, value: Any) -> None:
    if isinstance(value, dict):
        hasher.update(b"dict\0")
        for key in sorted(value):
            _update_hash(hasher, str(key))
            _update_hash(hasher, value[key])
        return
    if isinstance(value, np.ndarray):
        hasher.update(b"array\0")
        hasher.update(value.dtype.str.encode())
        hasher.update(json.dumps(value.shape).encode())
        if value.dtype.hasobject:
            for item in value.reshape(-1):
                _update_hash(hasher, item)
        else:
            hasher.update(np.ascontiguousarray(value).tobytes())
        return
    if isinstance(value, np.generic):
        _update_hash(hasher, value.item())
        return
    if isinstance(value, (list, tuple)):
        hasher.update(b"sequence\0")
        hasher.update(str(len(value)).encode())
        for item in value:
            _update_hash(hasher, item)
        return
    if isinstance(value, bytes):
        hasher.update(b"bytes\0")
        hasher.update(str(len(value)).encode())
        hasher.update(value)
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        hasher.update(b"string\0")
        hasher.update(str(len(encoded)).encode())
        hasher.update(encoded)
        return
    if value is None:
        hasher.update(b"none\0")
        return
    hasher.update(f"{type(value).__name__}\0{value!r}".encode())


def initial_observation_fingerprint(
    observations: dict[str, Any],
    env_idx: int = 0,
) -> str:
    """Fingerprint one environment's initial vector observation."""
    selected = {}
    for key, value in observations.items():
        try:
            selected[key] = value[env_idx]
        except (IndexError, KeyError, TypeError):
            selected[key] = value
    hasher = hashlib.sha256()
    _update_hash(hasher, selected)
    return f"sha256:{hasher.hexdigest()}"


def run_matched_full24_episodes(
    *,
    env: Any,
    get_actions: Callable[[dict[str, Any]], dict[str, Any]],
    env_name: str,
    n_episodes: int,
    base_eval_seed: int,
) -> tuple[list[bool], list[dict[str, Any]]]:
    """Run single-vector-env episodes with explicit deterministic resets."""
    successes = []
    records = []
    for episode_index in range(n_episodes):
        episode_seed = full24_episode_seed(
            base_eval_seed, env_name, episode_index
        )
        obs, _ = env.reset(seed=episode_seed)
        instruction = _instruction_for_vector_observation(obs, 0)
        fingerprint = initial_observation_fingerprint(obs, 0)
        success = False
        length = 0
        while True:
            next_obs, _, terminations, truncations, env_infos = env.step(
                get_actions(obs)
            )
            success |= bool(env_infos["success"][0][0])
            length += 1
            if terminations[0] or truncations[0]:
                successes.append(success)
                records.append(
                    {
                        "episode_index": episode_index,
                        "env_index": 0,
                        "episode_seed": episode_seed,
                        "instruction": instruction,
                        "initial_observation_fingerprint": fingerprint,
                        "success": bool(success),
                        "length": length,
                    }
                )
                break
            obs = next_obs
    return successes, records


def _instruction_for_vector_observation(
    observations: dict[str, Any], env_idx: int
) -> str:
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
