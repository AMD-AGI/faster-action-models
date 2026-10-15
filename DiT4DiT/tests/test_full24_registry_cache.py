# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from DiT4DiT.dataloader.gr00t_lerobot.full24_tasks import (
    FULL24_DATASET_TO_TASK_ID,
    FULL24_ENV_TO_TASK_ID,
    FULL24_TASKS,
)
from DiT4DiT.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from DiT4DiT.dataloader.gr00t_lerobot.pnp6_tasks import PNP6_TASKS
from examples.Robocasa_tabletop.train_files.prepare_pnp3_action_cache import (
    CACHE_SCHEMA_VERSION,
    PNP_TASK_DATASETS,
    combine_task_splits,
    compute_action_bounds,
    load_instruction_texts,
    parse_args,
    sample_episode_chunks,
    save_cache_immutably,
    split_episode_paths,
    tensor_payload_fingerprint,
    validate_cache,
)


def _write_episode(
    path: Path,
    action_value: float,
    instruction_ids: list[int],
) -> None:
    actions = [
        np.full(44, action_value + row / 100.0, dtype=np.float32).tolist()
        for row in range(len(instruction_ids))
    ]
    table = pa.table(
        {
            "action": pa.array(actions),
            "task_index": pa.array(instruction_ids, type=pa.int64()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_full24_registry_matches_mixture_order_and_ids():
    mixture_names = [
        dataset_name
        for dataset_name, weight, robot_type in DATASET_NAMED_MIXTURES[
            "fourier_gr1_unified_1000"
        ]
        if weight == 1.0 and robot_type == "fourier_gr1_arms_waist"
    ]
    assert len(FULL24_TASKS) == 24
    assert mixture_names == [task.dataset_name for task in FULL24_TASKS]
    assert [task.task_id for task in FULL24_TASKS] == list(range(24))
    assert [task.family_id for task in FULL24_TASKS] == list(range(24))
    assert [task.dataset_name for task in FULL24_TASKS[:6]] == [
        task.dataset_name for task in PNP6_TASKS
    ]
    assert {task.group for task in FULL24_TASKS} == {"pnp6", "novel"}
    assert all(task.source and task.destination for task in FULL24_TASKS)
    assert FULL24_DATASET_TO_TASK_ID == {
        task.dataset_name: task.task_id for task in FULL24_TASKS
    }
    assert FULL24_ENV_TO_TASK_ID == {
        task.env_name: task.task_id for task in FULL24_TASKS
    }


def test_full24_cache_balance_and_episode_disjoint_split():
    paths = [Path(f"episode_{index:06d}.parquet") for index in range(1000)]
    train, val = split_episode_paths(paths, 0.1, np.random.default_rng(42))
    assert len(train) == 900
    assert len(val) == 100
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(paths)

    count = 3
    splits = [
        {
            "actions": torch.zeros(count, 16, 29),
            "valid_steps": torch.ones(count, 16, dtype=torch.bool),
            "task_ids": torch.full((count,), family_id),
            "episode_ids": torch.arange(count),
            "instruction_ids": torch.ones(count, dtype=torch.int64),
        }
        for family_id in range(24)
    ]
    combined = combine_task_splits(splits)
    assert torch.bincount(combined["task_ids"]).tolist() == [count] * 24
    assert set(combined) == {
        "actions",
        "valid_steps",
        "task_ids",
        "episode_ids",
        "instruction_ids",
    }


def test_bounds_use_only_supplied_train_episodes(tmp_path):
    train_path = tmp_path / "episode_000000.parquet"
    val_path = tmp_path / "episode_000001.parquet"
    _write_episode(train_path, 1.0, [1, 1])
    _write_episode(val_path, 99.0, [1, 1])
    low, high = compute_action_bounds([train_path])
    assert np.all(low >= 1.0)
    assert np.all(high < 2.0)


def test_instruction_ids_and_nonempty_text_metadata(tmp_path):
    dataset_path = tmp_path / "dataset"
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    tasks_path.parent.mkdir(parents=True)
    tasks_path.write_text(
        "\n".join(
            [
                json.dumps({"task_index": 0, "task": ""}),
                json.dumps({"task_index": 7, "task": "pick the cup"}),
            ]
        )
        + "\n"
    )
    assert load_instruction_texts(dataset_path) == {7: "pick the cup"}

    episode_path = dataset_path / "data/chunk-000/episode_000000.parquet"
    _write_episode(episode_path, 0.0, [7, 7, 7, 7])
    split = sample_episode_chunks(
        [episode_path],
        target_count=4,
        horizon=2,
        task_id=3,
        low=np.full(29, -1.0, dtype=np.float32),
        high=np.full(29, 1.0, dtype=np.float32),
        rng=np.random.default_rng(0),
        instruction_id_map={7: 11},
    )
    assert split["instruction_ids"].tolist() == [11, 11, 11, 11]
    assert split["parquet_instruction_ids"].tolist() == [7, 7, 7, 7]


def test_pnp3_pnp6_and_global_cli_remain_compatible(monkeypatch):
    assert len(PNP_TASK_DATASETS["pnp3"]) == 3
    assert tuple(PNP_TASK_DATASETS["pnp6"]) == tuple(
        task.dataset_name for task in PNP6_TASKS
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare-cache", "--task-set", "pnp6", "--normalization", "global"],
    )
    global_args = parse_args()
    assert global_args.statistics.name == "dataset_statistics.json"

    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare-cache", "--task-set", "full24", "--normalization", "per-task"],
    )
    per_task_args = parse_args()
    assert per_task_args.statistics is None
    assert per_task_args.train_chunks_per_task == 100_000
    assert per_task_args.val_chunks_per_task == 10_000


def _small_fingerprinted_cache(fingerprint: str = "cache-a") -> dict:
    split = {
        "actions": torch.zeros(2, 2, 29, dtype=torch.float16),
        "valid_steps": torch.ones(2, 2, dtype=torch.bool),
        "task_ids": torch.tensor([0, 1]),
        "episode_ids": torch.tensor([1, 2], dtype=torch.int32),
        "instruction_ids": torch.tensor([0, 1]),
        "parquet_instruction_ids": torch.tensor([1, 1]),
    }
    cache = {
        "train": {key: value.clone() for key, value in split.items()},
        "val": {key: value.clone() for key, value in split.items()},
        "metadata": {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "cache_fingerprint": fingerprint,
            "task_datasets": ["family-a", "family-b"],
            "instruction_texts": ["instruction-a", "instruction-b"],
            "horizon": 2,
            "split_chunk_counts": {"train": 2, "val": 2},
            "episode_split": {
                "family-a": {
                    "train_episode_ids": [1],
                    "val_episode_ids": [2],
                }
            },
        },
    }
    cache["metadata"]["tensor_payload_sha256"] = tensor_payload_fingerprint(cache)
    return cache


def test_cache_is_atomically_reused_and_mismatch_is_rejected(tmp_path):
    output = tmp_path / "cache.pt"
    cache = _small_fingerprinted_cache()
    save_cache_immutably(cache, output)
    save_cache_immutably(cache, output)
    loaded = torch.load(output, map_location="cpu", weights_only=False)
    validate_cache(loaded)

    mismatch = _small_fingerprinted_cache("cache-b")
    with pytest.raises(FileExistsError, match="Refusing to replace cache"):
        save_cache_immutably(mismatch, output)


def test_cache_tensor_corruption_is_detected():
    cache = _small_fingerprinted_cache()
    cache["val"]["actions"][0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="payload fingerprint mismatch"):
        validate_cache(cache)

