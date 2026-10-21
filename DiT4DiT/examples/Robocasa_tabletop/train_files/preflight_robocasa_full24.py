#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and record the inputs for a four-GPU RoboCasa training run."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
import diffusers
import numpydantic
import albumentations

from DiT4DiT.dataloader.gr00t_lerobot.mixtures import (
    DATASET_NAMED_MIXTURES,
)


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-gpus", type=int, default=4)
    parser.add_argument("--min-free-gib", type=float, default=300.0)
    args = parser.parse_args()

    mixture = DATASET_NAMED_MIXTURES["fourier_gr1_unified_1000"]
    expected_datasets = [args.data_root / item[0] for item in mixture]
    if len(expected_datasets) != 24:
        raise SystemExit(
            f"Expected 24 datasets in the full mixture, got "
            f"{len(expected_datasets)}"
        )
    invalid_datasets = []
    for path in expected_datasets:
        info_path = path / "meta" / "info.json"
        required_paths = [
            info_path,
            path / "meta" / "stats.json",
            path / "data" / "chunk-000" / "episode_000000.parquet",
            path / "data" / "chunk-000" / "episode_000999.parquet",
            path
            / "videos"
            / "chunk-000"
            / "observation.images.ego_view"
            / "episode_000000.mp4",
            path
            / "videos"
            / "chunk-000"
            / "observation.images.ego_view"
            / "episode_000999.mp4",
        ]
        missing_paths = [
            str(required) for required in required_paths if not required.is_file()
        ]
        if missing_paths:
            invalid_datasets.extend(missing_paths)
            continue
        info = json.loads(info_path.read_text())
        if int(info.get("total_episodes", 0)) != 1000:
            invalid_datasets.append(
                f"{info_path}: expected total_episodes=1000"
            )
    if invalid_datasets:
        raise SystemExit(
            "Incomplete full-task datasets:\n"
            + "\n".join(invalid_datasets)
        )
    if not args.base_model.is_dir():
        raise SystemExit(f"Missing Cosmos base model: {args.base_model}")
    if not hasattr(diffusers, "Cosmos2_5_PredictBasePipeline"):
        raise SystemExit(
            "Installed diffusers does not provide "
            "Cosmos2_5_PredictBasePipeline"
        )
    if transformers.__version__ != "4.57.6":
        raise SystemExit(
            "RoboCasa training requires transformers 4.57.6, got "
            f"{transformers.__version__}"
        )

    gpu_count = torch.cuda.device_count()
    if gpu_count != args.required_gpus:
        raise SystemExit(
            f"Expected exactly {args.required_gpus} visible GPUs, got "
            f"{gpu_count}. Set HIP_VISIBLE_DEVICES explicitly."
        )
    gpu_names = [
        torch.cuda.get_device_name(index) for index in range(gpu_count)
    ]
    # The reference run used eight MI325X GPUs. Other AMD Instinct GPUs work
    # but are not the recorded configuration; set REQUIRE_GPU_MODEL (for
    # example REQUIRE_GPU_MODEL=MI325X) to make a mismatch fatal.
    required_model = os.environ.get("REQUIRE_GPU_MODEL", "")
    if required_model and not all(required_model in name for name in gpu_names):
        raise SystemExit(f"Expected {required_model} GPUs, got: {gpu_names}")
    if not all("MI325X" in name for name in gpu_names):
        print(
            f"warning: the reference run used MI325X GPUs; got {gpu_names}. "
            "Throughput, memory headroom and results can differ.",
            file=sys.stderr,
        )

    disk = shutil.disk_usage(args.repo_root)
    free_gib = disk.free / (1024**3)
    if free_gib < args.min_free_gib:
        raise SystemExit(
            f"Only {free_gib:.1f} GiB free; require "
            f"{args.min_free_gib:.1f} GiB"
        )

    git_commit = command_output(
        ["git", "-C", str(args.repo_root), "rev-parse", "HEAD"]
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "numpydantic": version("numpydantic"),
        "albumentations": albumentations.__version__,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "gpu_memory_gib": [
            torch.cuda.get_device_properties(index).total_memory / (1024**3)
            for index in range(gpu_count)
        ],
        "dataset_mix": "fourier_gr1_unified_1000",
        "dataset_count": len(expected_datasets),
        "data_root": str(args.data_root.resolve()),
        "base_model": str(args.base_model.resolve()),
        "free_disk_gib": free_gib,
        "git_commit": git_commit.strip() if git_commit else None,
        "rocm_topology": command_output(["rocm-smi", "--showtopo"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
