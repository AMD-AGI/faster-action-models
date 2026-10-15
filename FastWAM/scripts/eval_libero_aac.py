# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LIBERO evaluation for the Compact Action Expert (Action DiT plus Action VAE).

This is a small CLI over the standard single-process LIBERO evaluator. It
keeps the normal rollout path and writes per-trial success records.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.eval_libero_single import eval_single_process


def parse_task_ids(value: str | None) -> list[int]:
    if not value:
        return list(range(10))
    task_ids = [int(item) for item in value.split(",")]
    if any(task_id < 0 or task_id > 9 for task_id in task_ids):
        raise ValueError("task ids must be between 0 and 9")
    return task_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastwam-ckpt", required=True)
    parser.add_argument(
        "--action-vae-ckpt",
        help="Override model.action_vae_ckpt for Compact Action Expert evaluation.",
    )
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--task-suite", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--task-ids")
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--compile-action-infer", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = str(PROJECT_ROOT / "configs")
    evaluator = getattr(eval_single_process, "__wrapped__", eval_single_process)

    for task_id in parse_task_ids(args.task_ids):
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(
                config_name="sim_libero",
                overrides=[f"task={args.task_config}"],
            )
        cfg.ckpt = args.fastwam_ckpt
        if args.action_vae_ckpt is not None:
            cfg.model.action_vae_ckpt = args.action_vae_ckpt
        cfg.seed = args.seed
        cfg.EVALUATION.task_suite_name = args.task_suite
        cfg.EVALUATION.task_id = task_id
        cfg.EVALUATION.num_trials = args.num_trials
        cfg.EVALUATION.output_dir = str(output_dir)
        cfg.EVALUATION.dataset_stats_path = args.dataset_stats
        cfg.EVALUATION.device = args.device
        cfg.EVALUATION.replan_steps = args.replan_steps
        cfg.EVALUATION.num_inference_steps = args.num_inference_steps
        cfg.EVALUATION.compile_action_infer = args.compile_action_infer
        results = evaluator(cfg)

        successes = set(results["success_episodes"])
        trial_file = output_dir / f"task{task_id}_trials.jsonl"
        trial_file.write_text(
            "".join(
                json.dumps(
                    {
                        "task_id": task_id,
                        "trial": trial,
                        "success": trial in successes,
                    }
                )
                + "\n"
                for trial in range(args.num_trials)
            )
        )


if __name__ == "__main__":
    main()
