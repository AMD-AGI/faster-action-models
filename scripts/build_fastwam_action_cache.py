#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the CAE-FastWAM action cache (PREDECODED_ACTION_CACHE) from the public LIBERO data.

scripts/train_fastwam_libero_action_vae.sh trains the K8 x d16 Action VAE on a single .pt file holding every
training action chunk of the four LIBERO suites:

  actions        float32 [N, 32, 7]  per-suite normalized action chunks
  action_is_pad  bool    [N, 32]     chunk steps past the end of the episode
  suite_ids      int64   [N]         0 spatial, 1 object, 2 goal, 3 libero_10 (the evaluator's suite ids)

This script makes that file from the LeRobot LIBERO datasets published as yuanty/LIBERO-fastwam. It reads only
the parquet action columns (no video decoding), so the small LeRobot 3.0 copy is enough:

  hf download yuanty/LIBERO-fastwam --repo-type dataset \\
      --include "lerobot_v30/*.parquet" --local-dir data/LIBERO-fastwam
  hf download amd/CAE-FastWAM --include "stats/*" --local-dir checkpoints/CAE-FastWAM
  python3 scripts/build_fastwam_action_cache.py \\
      --data-root data/LIBERO-fastwam/lerobot_v30 \\
      --stats-dir checkpoints/CAE-FastWAM/stats \\
      --output data/fastwam_libero_action_cache.pt

The extracted LeRobot 2.1 tarballs (data/libero_mujoco3.3.2/, the layout the latent-policy training reads) work
as --data-root too. Each training sample is built the way FastWAM's training dataset builds it
(FastWAM/configs/data/libero_2cam.yaml): one chunk per frame, the next 32 actions, steps past the episode end
repeat the last action and are marked as padding, the delta dimensions of padded steps are zeroed, and the chunk
is normalized with its suite's min/max statistics (the stats/libero_{suite}.json files published with
amd/CAE-FastWAM, which are the statistics CAE-FastWAM was trained with).

Run it with the FastWAM Python environment or in the evaluation image (docker/Dockerfile.eval); it imports
FastWAM's normalizer and needs no GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FASTWAM_ROOT = ROOT / "FastWAM"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DATA_CONFIG = FASTWAM_ROOT / "configs" / "data" / "libero_2cam.yaml"


def suite_dir(data_root: Path, suite: str) -> Path:
    return data_root / f"{suite}_no_noops_lerobot"


def stats_file(stats_dir: Path, suite: str) -> Path:
    for name in (f"{suite}.json", f"dataset_stats_{suite}.json"):
        if (stats_dir / name).is_file():
            return stats_dir / name
    return stats_dir / f"{suite}.json"


def parquet_files(ds_dir: Path) -> list[Path]:
    return sorted((ds_dir / "data").rglob("*.parquet"))


def load_processor_settings() -> dict:
    """The action settings of FastWAM's LIBERO data config, checked against what this script implements."""
    from omegaconf import OmegaConf

    proc = OmegaConf.load(DATA_CONFIG).train.processor
    if proc.get("action_state_transforms") is not None:
        raise SystemExit(f"{DATA_CONFIG}: action_state_transforms is set; this builder does not apply them")
    if proc.get("norm_exception_mode") is not None:
        raise SystemExit(f"{DATA_CONFIG}: norm_exception_mode is set; this builder does not apply it")
    return {
        "delta_mask": [bool(x) for x in proc.delta_action_dim_mask["default"]],
        "stepwise": bool(proc.use_stepwise_action_norm),
        "mode": str(proc.norm_default_mode),
        "action_dim": int(proc.action_output_dim),
    }


def load_episodes(ds_dir: Path, max_episodes: int | None):
    """Yield (episode_index, actions[T, D]) in episode order, from LeRobot 2.1 or 3.0 parquet files."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    tables = [pq.read_table(f, columns=["episode_index", "frame_index", "action"]) for f in parquet_files(ds_dir)]
    if not tables:
        raise SystemExit(f"no parquet files under {ds_dir}/data")
    table = pa.concat_tables(tables)
    order = pc.sort_indices(table, sort_keys=[("episode_index", "ascending"), ("frame_index", "ascending")])
    table = table.take(order)
    episode = table.column("episode_index").to_numpy()
    frame = table.column("frame_index").to_numpy()
    action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, episode[1:] != episode[:-1]])
    ends = np.r_[starts[1:], len(episode)]
    for n, (s, e) in enumerate(zip(starts, ends)):
        if max_episodes is not None and n >= max_episodes:
            break
        if not np.array_equal(frame[s:e], np.arange(e - s)):
            raise SystemExit(f"{ds_dir}: episode {int(episode[s])} has missing or repeated frames")
        yield int(episode[s]), action[s:e]


def build_suite(ds_dir: Path, stats_path: Path, settings: dict, horizon: int, max_episodes: int | None):
    import torch

    sys.path.insert(0, str(FASTWAM_ROOT / "src"))
    try:
        import git  # noqa: F401  GitPython, in FastWAM's own environment
    except ImportError:
        # FastWAM's normalizer module imports GitPython only to name its dataset-statistics cache files, which this
        # script does not use; a stand-in lets it run in the evaluation image (docker/Dockerfile.eval) too.
        import types

        def _no_repo(*_args, **_kwargs):
            raise RuntimeError("GitPython is not installed")

        sys.modules["git"] = types.ModuleType("git")
        sys.modules["git"].Repo = _no_repo
    from fastwam.datasets.lerobot.utils.normalizer import (  # noqa: E402
        SingleFieldLinearNormalizer,
        load_dataset_stats_from_json,
    )

    stats = load_dataset_stats_from_json(str(stats_path))["action"]["default"]
    prefix = "stepwise_" if settings["stepwise"] else "global_"
    normalizer = SingleFieldLinearNormalizer(
        stats={k.removeprefix(prefix): v for k, v in stats.items() if k.startswith(prefix)},
        mode=settings["mode"],
    )
    delta = torch.tensor(settings["delta_mask"], dtype=torch.bool)
    offsets = torch.arange(horizon)
    chunks, pads, episodes = [], [], 0
    for _, actions in load_episodes(ds_dir, max_episodes):
        a = torch.from_numpy(actions)
        if a.shape[-1] != settings["action_dim"]:
            raise SystemExit(f"{ds_dir}: action dim {a.shape[-1]} != {settings['action_dim']}")
        t = a.shape[0]
        index = torch.arange(t)[:, None] + offsets[None, :]  # [T, H]
        is_pad = index >= t
        chunk = a[index.clamp(max=t - 1)].clone()  # [T, H, D]; past the end repeats the last action
        chunk[is_pad[..., None] & delta] = 0.0
        chunks.append(normalizer.forward(chunk).float())
        pads.append(is_pad)
        episodes += 1
    return torch.cat(chunks), torch.cat(pads), episodes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n\n".join(__doc__.split("\n\n")[1:]),
    )
    parser.add_argument("--data-root", required=True, type=Path,
                        help="directory holding libero_{spatial,object,goal,10}_no_noops_lerobot (LeRobot 2.1 or 3.0)")
    parser.add_argument("--stats-dir", required=True, type=Path,
                        help="per-suite statistics: libero_{suite}.json (amd/CAE-FastWAM stats/) "
                             "or dataset_stats_libero_{suite}.json")
    parser.add_argument("--output", required=True, type=Path, help="new .pt file (PREDECODED_ACTION_CACHE)")
    parser.add_argument("--suites", default=",".join(SUITES),
                        help="comma-separated suites, in suite-id order (default: all four)")
    parser.add_argument("--horizon", type=int, default=32, help="action chunk length (default 32)")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="use only the first N episodes of each suite (smoke tests)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing --output")
    parser.add_argument("--dry-run", action="store_true",
                        help="check the inputs and print the plan, without reading the data")
    args = parser.parse_args()

    suites = [s for s in args.suites.split(",") if s]
    unknown = [s for s in suites if s not in SUITES]
    if unknown:
        parser.error(f"unknown suite(s) {unknown}; choose from {list(SUITES)}")
    suite_ids = {s: SUITES.index(s) for s in suites}

    missing = []
    for s in suites:
        if not parquet_files(suite_dir(args.data_root, s)):
            missing.append(f"no parquet data under {suite_dir(args.data_root, s)}/data")
        if not stats_file(args.stats_dir, s).is_file():
            missing.append(f"no statistics file {stats_file(args.stats_dir, s)}")
    if args.output.exists() and not args.force:
        missing.append(f"{args.output} exists (pass --force to overwrite)")
    for s in suites:
        print(f"{s:15s} id={suite_ids[s]}  data={suite_dir(args.data_root, s)}  stats={stats_file(args.stats_dir, s)}")
    print(f"output: {args.output}  horizon: {args.horizon}"
          + (f"  max episodes per suite: {args.max_episodes}" if args.max_episodes else ""))
    if missing:
        for m in missing:
            print(f"error: {m}", file=sys.stderr)
        return 2
    if args.dry_run:
        return 0

    import torch

    settings = load_processor_settings()
    actions, pads, ids = [], [], []
    for s in suites:
        a, p, n = build_suite(suite_dir(args.data_root, s), stats_file(args.stats_dir, s), settings,
                              args.horizon, args.max_episodes)
        print(f"{s:15s} {n} episodes, {a.shape[0]} chunks, normalized range [{a.min():.3f}, {a.max():.3f}]")
        actions.append(a)
        pads.append(p)
        ids.append(torch.full((a.shape[0],), suite_ids[s], dtype=torch.long))
    payload = {
        "actions": torch.cat(actions).contiguous(),
        "action_is_pad": torch.cat(pads).contiguous(),
        "suite_ids": torch.cat(ids),
        "suites": suites,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(args.output)
    print(f"wrote {args.output}: actions {tuple(payload['actions'].shape)}, "
          f"{int(payload['action_is_pad'].sum())} padded steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
