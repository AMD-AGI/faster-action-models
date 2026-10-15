#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict, read-only structural and readability audit for RoboCasa full-24."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from DiT4DiT.dataloader.gr00t_lerobot.full24_tasks import FULL24_TASKS  # noqa: E402

EXPECTED_EPISODE_IDS = set(range(1000))
REQUIRED_METADATA = (
    "episodes.jsonl",
    "info.json",
    "modality.json",
    "stats.json",
    "tasks.jsonl",
)
VIDEO_KEY = "observation.images.ego_view"


def _episode_ids(paths: list[Path], suffix: str) -> set[int]:
    ids: set[int] = set()
    for path in paths:
        try:
            episode_id = int(path.stem.removeprefix("episode_"))
        except ValueError as exc:
            raise ValueError(f"Invalid {suffix} episode filename: {path}") from exc
        if episode_id in ids:
            raise ValueError(f"Duplicate {suffix} episode {episode_id}: {path}")
        ids.add(episode_id)
    return ids


def _validate_instruction_metadata(tasks_path: Path) -> int:
    records = [
        json.loads(line)
        for line in tasks_path.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Empty task metadata: {tasks_path}")
    task_ids: set[int] = set()
    nonempty_count = 0
    for record in records:
        task_id = int(record["task_index"])
        if task_id in task_ids:
            raise ValueError(f"Duplicate task_index {task_id} in {tasks_path}")
        task_ids.add(task_id)
        if str(record["task"]).strip():
            nonempty_count += 1
    if nonempty_count == 0:
        raise ValueError(f"No nonempty task text in {tasks_path}")
    return nonempty_count


def _sample_ids(sample_count: int) -> list[int]:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if sample_count == 1:
        return [0]
    return sorted(
        {
            round(index * 999 / (sample_count - 1))
            for index in range(sample_count)
        }
    )


def _read_parquet_sample(path: Path) -> None:
    table = pq.read_table(path, columns=["action", "task_index"])
    if table.num_rows == 0:
        raise ValueError(f"Parquet contains no rows: {path}")
    if table["action"][0].as_py() is None or table["task_index"][0].as_py() is None:
        raise ValueError(f"Parquet sample contains null required values: {path}")


def _read_video_sample(path: Path) -> None:
    with av.open(str(path)) as container:
        frame = next(container.decode(video=0), None)
        if frame is None or frame.width <= 0 or frame.height <= 0:
            raise ValueError(f"Could not decode a video frame: {path}")


def audit_full24(data_root: Path, sample_count: int = 3) -> dict[str, object]:
    expected_names = {task.dataset_name for task in FULL24_TASKS}
    actual_names = {
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            "Dataset directory mismatch: "
            f"missing={missing or 'none'}, unexpected={unexpected or 'none'}"
        )

    samples = _sample_ids(sample_count)
    instruction_count = 0
    for task in FULL24_TASKS:
        dataset_path = data_root / task.dataset_name
        for metadata_name in REQUIRED_METADATA:
            metadata_path = dataset_path / "meta" / metadata_name
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Missing required metadata: {metadata_path}")

        info_path = dataset_path / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        for key in ("total_episodes", "total_videos"):
            if int(info.get(key, -1)) != 1000:
                raise ValueError(f"{info_path}: expected {key}=1000")
        instruction_count += _validate_instruction_metadata(
            dataset_path / "meta" / "tasks.jsonl"
        )

        parquet_paths = sorted(
            (dataset_path / "data").glob("chunk-*/episode_*.parquet")
        )
        video_paths = sorted(
            (dataset_path / "videos").glob(
                f"chunk-*/{VIDEO_KEY}/episode_*.mp4"
            )
        )
        parquet_ids = _episode_ids(parquet_paths, "parquet")
        video_ids = _episode_ids(video_paths, "video")
        if parquet_ids != EXPECTED_EPISODE_IDS:
            raise ValueError(
                f"{task.dataset_name}: parquet IDs do not equal exactly 0..999"
            )
        if video_ids != EXPECTED_EPISODE_IDS:
            raise ValueError(
                f"{task.dataset_name}: video IDs do not equal exactly 0..999"
            )

        parquet_by_id = {
            int(path.stem.removeprefix("episode_")): path for path in parquet_paths
        }
        video_by_id = {
            int(path.stem.removeprefix("episode_")): path for path in video_paths
        }
        for episode_id in samples:
            _read_parquet_sample(parquet_by_id[episode_id])
            _read_video_sample(video_by_id[episode_id])

    return {
        "dataset_count": len(FULL24_TASKS),
        "parquet_count": len(FULL24_TASKS) * 1000,
        "video_count": len(FULL24_TASKS) * 1000,
        "nonempty_instruction_count": instruction_count,
        "sampled_episode_ids": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT
        / "playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
    )
    parser.add_argument("--sample-count", type=int, default=3)
    args = parser.parse_args()
    summary = audit_full24(args.data_root, args.sample_count)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
