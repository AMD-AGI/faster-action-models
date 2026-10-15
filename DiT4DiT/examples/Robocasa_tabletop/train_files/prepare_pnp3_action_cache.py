#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a balanced normalized action-chunk cache for RoboCasa task families."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from DiT4DiT.dataloader.gr00t_lerobot.full24_tasks import FULL24_TASKS  # noqa: E402
from DiT4DiT.dataloader.gr00t_lerobot.pnp6_tasks import PNP6_TASKS  # noqa: E402


PNP_TASK_DATASETS = {
    "pnp3": (
        "gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
        "gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
        "gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
    ),
    "pnp6": tuple(task.dataset_name for task in PNP6_TASKS),
    "full24": tuple(task.dataset_name for task in FULL24_TASKS),
}

# Order used by FourierGr1ArmsWaistDataConfig:
# left arm, right arm, left hand, right hand, waist.
ACTIVE_ACTION_INDICES = np.asarray(
    [*range(0, 7), *range(22, 29), *range(7, 13), *range(29, 35), *range(41, 44)],
    dtype=np.int64,
)
CACHE_SCHEMA_VERSION = 2
SOURCE_METADATA_FILES = ("info.json", "stats.json", "tasks.jsonl")


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _jsonable(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def source_dataset_fingerprint(data_root: Path, datasets: tuple[str, ...]) -> dict:
    """Fingerprint the ordered source inventory without rereading all parquet payloads."""
    digest = hashlib.sha256()
    per_dataset: dict[str, dict[str, object]] = {}
    for dataset_name in datasets:
        dataset_path = data_root / dataset_name
        metadata_hashes = {}
        for name in SOURCE_METADATA_FILES:
            path = dataset_path / "meta" / name
            content = path.read_bytes()
            metadata_hashes[name] = hashlib.sha256(content).hexdigest()
            digest.update(_canonical_json((dataset_name, f"meta/{name}", metadata_hashes[name])))
        parquet_paths = sorted(
            (dataset_path / "data").glob("chunk-*/episode_*.parquet")
        )
        inventory = [
            (str(path.relative_to(dataset_path)), path.stat().st_size)
            for path in parquet_paths
        ]
        inventory_digest = hashlib.sha256(_canonical_json(inventory)).hexdigest()
        digest.update(_canonical_json((dataset_name, inventory_digest)))
        per_dataset[dataset_name] = {
            "parquet_count": len(parquet_paths),
            "inventory_sha256": inventory_digest,
            "metadata_sha256": metadata_hashes,
        }
    return {"sha256": digest.hexdigest(), "datasets": per_dataset}


def tensor_payload_fingerprint(cache: dict) -> str:
    """Hash cache tensors in bounded views so validation does not duplicate the cache."""
    digest = hashlib.sha256()
    for split_name in ("train", "val"):
        split = cache[split_name]
        for key in sorted(split):
            tensor = split[key].detach().cpu().contiguous()
            digest.update(_canonical_json((split_name, key, str(tensor.dtype), list(tensor.shape))))
            rows = max(1, min(4096, tensor.shape[0] if tensor.ndim else 1))
            if tensor.ndim == 0:
                digest.update(memoryview(tensor.numpy()))
                continue
            for start in range(0, tensor.shape[0], rows):
                digest.update(memoryview(tensor[start : start + rows].numpy()))
    return digest.hexdigest()


def validate_cache(cache: dict, verify_tensor_hash: bool = True) -> None:
    metadata = cache.get("metadata", {})
    if metadata.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported action-cache schema: "
            f"{metadata.get('cache_schema_version')!r}"
        )
    task_count = len(metadata.get("task_datasets", []))
    instruction_count = len(metadata.get("instruction_texts", []))
    horizon = int(metadata.get("horizon", -1))
    expected_counts = metadata.get("split_chunk_counts", {})
    for split_name in ("train", "val"):
        split = cache.get(split_name)
        if not isinstance(split, dict):
            raise ValueError(f"Missing cache split: {split_name}")
        required = {
            "actions",
            "valid_steps",
            "task_ids",
            "episode_ids",
            "instruction_ids",
            "parquet_instruction_ids",
        }
        if set(split) != required:
            raise ValueError(
                f"{split_name} cache fields mismatch: {sorted(set(split) ^ required)}"
            )
        count = split["actions"].shape[0]
        if count != int(expected_counts.get(split_name, -1)):
            raise ValueError(f"{split_name} chunk count does not match metadata")
        if split["actions"].shape[1:] != (horizon, 29):
            raise ValueError(f"Invalid {split_name} action shape: {split['actions'].shape}")
        if split["valid_steps"].shape != (count, horizon):
            raise ValueError(f"Invalid {split_name} valid-step shape")
        if any(value.shape != (count,) for key, value in split.items() if key not in {"actions", "valid_steps"}):
            raise ValueError(f"Invalid {split_name} sample metadata shape")
        if not torch.isfinite(split["actions"]).all():
            raise ValueError(f"{split_name} actions contain non-finite values")
        if split["actions"].abs().max().item() > 1.0001:
            raise ValueError(f"{split_name} normalized actions exceed [-1, 1]")
        task_ids = split["task_ids"]
        if task_ids.min().item() < 0 or task_ids.max().item() >= task_count:
            raise ValueError(f"{split_name} task IDs are out of range")
        counts = torch.bincount(task_ids, minlength=task_count)
        if counts.min().item() != counts.max().item():
            raise ValueError(f"{split_name} is not balanced across task families")
        instruction_ids = split["instruction_ids"]
        if (
            instruction_ids.min().item() < 0
            or instruction_ids.max().item() >= instruction_count
        ):
            raise ValueError(f"{split_name} instruction IDs are out of range")
    for dataset_name, split in metadata.get("episode_split", {}).items():
        train_ids = set(split["train_episode_ids"])
        val_ids = set(split["val_episode_ids"])
        if train_ids & val_ids:
            raise ValueError(f"Episode split overlap for {dataset_name}")
    expected_payload_hash = metadata.get("tensor_payload_sha256")
    if verify_tensor_hash and tensor_payload_fingerprint(cache) != expected_payload_hash:
        raise ValueError("Action-cache tensor payload fingerprint mismatch")


def save_cache_immutably(cache: dict, output: Path) -> str:
    """Atomically create or exactly reuse a fingerprinted cache."""
    validate_cache(cache)
    fingerprint = cache["metadata"]["cache_fingerprint"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = torch.load(output, map_location="cpu", weights_only=False)
        validate_cache(existing)
        existing_fingerprint = existing["metadata"].get("cache_fingerprint")
        if existing_fingerprint != fingerprint:
            raise FileExistsError(
                f"Refusing to replace cache {output}: existing fingerprint "
                f"{existing_fingerprint}, requested {fingerprint}"
            )
        print(f"Reusing verified immutable cache {output} ({fingerprint})")
        return fingerprint
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        torch.save(cache, temporary)
        written = torch.load(temporary, map_location="cpu", weights_only=False)
        validate_cache(written)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-set", choices=tuple(PNP_TASK_DATASETS), default="pnp3")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT
        / "playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
    )
    parser.add_argument(
        "--statistics",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--train-chunks-per-task", type=int, default=100_000)
    parser.add_argument("--val-chunks-per-task", type=int, default=10_000)
    parser.add_argument("--val-episode-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalization",
        choices=("global", "per-task"),
        default="global",
    )
    args = parser.parse_args()
    if args.statistics is None and args.normalization == "global":
        default_runs = {
            "pnp3": "dit4dit_robocasa_pnp3_25k",
            "pnp6": "dit4dit_robocasa_pnp6_stats",
            "full24": "dit4dit_robocasa_gr1_mi325x4_b16_eager_s42",
        }
        args.statistics = (
            REPO_ROOT
            / "playground/Checkpoints_robocasa"
            / default_runs[args.task_set]
            / "dataset_statistics.json"
        )
    if args.output is None:
        args.output = (
            REPO_ROOT
            / "playground/action_vae"
            / f"{args.task_set}_action_chunks.pt"
        )
    return args


def load_normalization(statistics_path: Path) -> tuple[np.ndarray, np.ndarray]:
    statistics = json.loads(statistics_path.read_text())
    if "gr1" in statistics:
        action_stats = statistics["gr1"]["action"]
    elif len(statistics) == 1:
        action_stats = next(iter(statistics.values()))["action"]
    else:
        raise KeyError(f"Cannot select action statistics from {statistics_path}")
    low = np.asarray(action_stats["min"], dtype=np.float32)
    high = np.asarray(action_stats["max"], dtype=np.float32)
    if low.shape != (29,) or high.shape != (29,):
        raise ValueError(
            f"Expected 29-dimensional RoboCasa statistics, got {low.shape} and {high.shape}"
        )
    return low, high


def normalize_actions(actions: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    span = high - low
    safe_span = np.where(np.abs(span) > 1e-8, span, 1.0)
    normalized = 2.0 * (actions - low) / safe_span - 1.0
    normalized[..., np.abs(span) <= 1e-8] = 0.0
    return np.clip(normalized, -1.0, 1.0)


def sample_episode_chunks(
    episode_paths: list[Path],
    target_count: int,
    horizon: int,
    task_id: int,
    low: np.ndarray,
    high: np.ndarray,
    rng: np.random.Generator,
    instruction_id_map: dict[int, int] | None = None,
) -> dict[str, torch.Tensor]:
    action_chunks: list[np.ndarray] = []
    valid_chunks: list[np.ndarray] = []
    episode_chunks: list[np.ndarray] = []
    instruction_chunks: list[np.ndarray] = []
    remaining = target_count

    shuffled_paths = list(episode_paths)
    rng.shuffle(shuffled_paths)
    for position, episode_path in enumerate(shuffled_paths):
        if remaining <= 0:
            break
        table = pq.read_table(episode_path, columns=["action", "task_index"])
        raw = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        instruction_ids = np.asarray(table["task_index"], dtype=np.int64)
        actions = normalize_actions(raw[:, ACTIVE_ACTION_INDICES], low, high)
        episodes_left = len(shuffled_paths) - position
        requested = min(len(actions), math.ceil(remaining / episodes_left))
        starts = rng.choice(len(actions), size=requested, replace=False)
        time_indices = starts[:, None] + np.arange(horizon, dtype=np.int64)[None, :]
        valid = time_indices < len(actions)
        time_indices = np.minimum(time_indices, len(actions) - 1)
        action_chunks.append(actions[time_indices].astype(np.float16))
        valid_chunks.append(valid)
        episode_index = int(episode_path.stem.rsplit("_", 1)[-1])
        episode_chunks.append(np.full(requested, episode_index, dtype=np.int32))
        instruction_chunks.append(instruction_ids[starts])
        remaining -= requested

    if remaining > 0:
        raise RuntimeError(
            f"Only built {target_count - remaining}/{target_count} chunks for task {task_id}"
        )
    actions = np.concatenate(action_chunks, axis=0)[:target_count]
    valid_steps = np.concatenate(valid_chunks, axis=0)[:target_count]
    episode_ids = np.concatenate(episode_chunks, axis=0)[:target_count]
    parquet_instruction_ids = np.concatenate(instruction_chunks, axis=0)[
        :target_count
    ]
    if instruction_id_map is None:
        instruction_ids = parquet_instruction_ids
    else:
        try:
            instruction_ids = np.asarray(
                [
                    instruction_id_map[int(instruction_id)]
                    for instruction_id in parquet_instruction_ids
                ],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                f"Parquet task_index {exc.args[0]} has no nonempty instruction text"
            ) from exc
    return {
        "actions": torch.from_numpy(actions),
        "valid_steps": torch.from_numpy(valid_steps),
        "task_ids": torch.full((target_count,), task_id, dtype=torch.int64),
        "episode_ids": torch.from_numpy(episode_ids),
        "instruction_ids": torch.from_numpy(instruction_ids),
        "parquet_instruction_ids": torch.from_numpy(parquet_instruction_ids),
    }


def combine_task_splits(splits: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.cat([split[key] for split in splits], dim=0)
        for key in splits[0]
    }


def compute_action_bounds(episode_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    low = np.full(29, np.inf, dtype=np.float32)
    high = np.full(29, -np.inf, dtype=np.float32)
    for episode_path in episode_paths:
        raw = np.asarray(
            pq.read_table(episode_path, columns=["action"])["action"].to_pylist(),
            dtype=np.float32,
        )[:, ACTIVE_ACTION_INDICES]
        low = np.minimum(low, raw.min(axis=0))
        high = np.maximum(high, raw.max(axis=0))
    return low, high


def load_instruction_texts(dataset_path: Path) -> dict[int, str]:
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Missing instruction metadata: {tasks_path}")
    instructions: dict[int, str] = {}
    for line_number, line in enumerate(tasks_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        instruction_id = int(record["task_index"])
        text = str(record["task"]).strip()
        if not text:
            continue
        if instruction_id in instructions:
            raise ValueError(
                f"Duplicate task_index {instruction_id} in {tasks_path}:{line_number}"
            )
        instructions[instruction_id] = text
    if not instructions:
        raise ValueError(f"No nonempty instructions in {tasks_path}")
    return instructions


def split_episode_paths(
    episode_paths: list[Path],
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[Path], list[Path]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_episode_fraction must be strictly between 0 and 1")
    if len(episode_paths) < 2:
        raise ValueError("At least two episodes are required for a disjoint split")
    permutation = rng.permutation(len(episode_paths))
    val_count = max(
        1,
        min(
            len(episode_paths) - 1,
            round(len(episode_paths) * val_fraction),
        ),
    )
    val_paths = [episode_paths[index] for index in permutation[:val_count]]
    train_paths = [episode_paths[index] for index in permutation[val_count:]]
    return train_paths, val_paths


def main() -> None:
    args = parse_args()
    global_bounds = None
    if args.normalization == "global":
        if args.statistics is None:
            raise ValueError("--statistics is required for global normalization")
        global_bounds = load_normalization(args.statistics)
    train_splits: list[dict[str, torch.Tensor]] = []
    val_splits: list[dict[str, torch.Tensor]] = []
    split_manifest: dict[str, dict[str, list[int]]] = {}
    task_specs = []
    instruction_texts: list[str] = []
    instruction_registry: list[dict[str, object]] = []
    instruction_id_maps: list[dict[int, int]] = []

    task_datasets = PNP_TASK_DATASETS[args.task_set]
    source_fingerprint = source_dataset_fingerprint(args.data_root, task_datasets)
    for task_id, dataset_name in enumerate(task_datasets):
        dataset_path = args.data_root / dataset_name
        episode_paths = sorted(
            (dataset_path / "data").glob("chunk-*/episode_*.parquet")
        )
        if not episode_paths:
            raise FileNotFoundError(f"No parquet episodes found for {dataset_name}")
        family_instructions = load_instruction_texts(dataset_path)
        family_id_map: dict[int, int] = {}
        for parquet_task_index, text in sorted(family_instructions.items()):
            instruction_id = len(instruction_texts)
            family_id_map[parquet_task_index] = instruction_id
            instruction_texts.append(text)
            instruction_registry.append(
                {
                    "instruction_id": instruction_id,
                    "family_id": task_id,
                    "parquet_task_index": parquet_task_index,
                    "text": text,
                }
            )
        instruction_id_maps.append(family_id_map)
        rng = np.random.default_rng(args.seed + task_id)
        train_paths, val_paths = split_episode_paths(
            episode_paths, args.val_episode_fraction, rng
        )
        split_manifest[dataset_name] = {
            "train_episode_ids": [
                int(path.stem.rsplit("_", 1)[-1]) for path in train_paths
            ],
            "val_episode_ids": [
                int(path.stem.rsplit("_", 1)[-1]) for path in val_paths
            ],
        }
        task_specs.append((task_id, dataset_name, train_paths, val_paths, rng))

    if args.normalization == "per-task":
        task_bounds = [
            compute_action_bounds(train_paths)
            for _, _, train_paths, _, _ in task_specs
        ]
        metadata_low = torch.from_numpy(np.stack([low for low, _ in task_bounds]))
        metadata_high = torch.from_numpy(np.stack([high for _, high in task_bounds]))
    else:
        assert global_bounds is not None
        global_low, global_high = global_bounds
        task_bounds = [(global_low, global_high) for _ in task_specs]
        metadata_low = torch.from_numpy(global_low)
        metadata_high = torch.from_numpy(global_high)

    for task_spec, (low, high) in zip(task_specs, task_bounds):
        task_id, dataset_name, train_paths, val_paths, rng = task_spec
        train_split = sample_episode_chunks(
            train_paths,
            args.train_chunks_per_task,
            args.horizon,
            task_id,
            low,
            high,
            rng,
            instruction_id_maps[task_id],
        )
        val_split = sample_episode_chunks(
            val_paths,
            args.val_chunks_per_task,
            args.horizon,
            task_id,
            low,
            high,
            rng,
            instruction_id_maps[task_id],
        )
        known_instruction_ids = set(instruction_id_maps[task_id].values())
        for split_name, split in (("train", train_split), ("val", val_split)):
            unknown = set(split["instruction_ids"].tolist()) - known_instruction_ids
            if unknown:
                raise ValueError(
                    f"{dataset_name} {split_name} references task_index values "
                    f"without nonempty text: {sorted(unknown)}"
                )
        train_splits.append(train_split)
        val_splits.append(val_split)
        print(
            f"{dataset_name}: {len(train_paths)} train episodes, "
            f"{len(val_paths)} validation episodes"
        )

    combined_train = combine_task_splits(train_splits)
    combined_val = combine_task_splits(val_splits)
    cache = {
        "train": combined_train,
        "val": combined_val,
        "metadata": {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "task_set": args.task_set,
            "normalization": args.normalization,
            "task_datasets": list(task_datasets),
            "instruction_texts": instruction_texts,
            "instruction_registry": instruction_registry,
            "active_action_indices": ACTIVE_ACTION_INDICES.tolist(),
            "action_low": metadata_low,
            "action_high": metadata_high,
            "horizon": args.horizon,
            "seed": args.seed,
            "statistics_path": (
                str(args.statistics) if args.statistics is not None else None
            ),
            "episode_split": split_manifest,
            "split_chunk_counts": {
                "train": len(task_datasets) * args.train_chunks_per_task,
                "val": len(task_datasets) * args.val_chunks_per_task,
            },
            "source_dataset_fingerprint": source_fingerprint,
            "instruction_counts": {
                split_name: torch.bincount(
                    combine["instruction_ids"], minlength=len(instruction_texts)
                ).tolist()
                for split_name, combine in (
                    ("train", combined_train),
                    ("val", combined_val),
                )
            },
        },
    }
    cache["metadata"]["tensor_payload_sha256"] = tensor_payload_fingerprint(cache)
    manifest_for_hash = {
        key: value
        for key, value in cache["metadata"].items()
        if key != "cache_fingerprint"
    }
    cache["metadata"]["cache_fingerprint"] = hashlib.sha256(
        _canonical_json(manifest_for_hash)
    ).hexdigest()
    fingerprint = save_cache_immutably(cache, args.output)
    print(
        f"Saved {len(cache['train']['actions']):,} train and "
        f"{len(cache['val']['actions']):,} validation chunks to {args.output} "
        f"(fingerprint {fingerprint})"
    )


if __name__ == "__main__":
    main()
