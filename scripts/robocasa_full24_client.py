# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the full-24 ``simulation_env.py`` client on part of a task's episodes.

    python robocasa_full24_client.py FIRST <tree>/examples/Robocasa_tabletop/eval_files/simulation_env.py [--args.* ...]

plays episodes FIRST .. FIRST + n_episodes - 1 of the task, where episode i is
reset with ``full24_episode_seed(eval_seed, env_name, i)``, its seed in a run
that plays the whole task in one client. Its records carry those episode
indices. The script is then run unchanged as ``__main__`` with the given
arguments; only ``run_matched_full24_episodes``, which simulation_env.py imports
from ``full24_rollout_matching``, is wrapped. robocasa_full24.py merges the
parts of a task into the result file a single client writes.
"""

from __future__ import annotations

import importlib
import os
import runpy
import sys
from typing import Any, Callable

MATCHING = "examples.Robocasa_tabletop.eval_files.full24_rollout_matching"


def episode_range_runner(matching: Any, first: int) -> Callable[..., Any]:
    """Wrap matching.run_matched_full24_episodes to play episodes first, first + 1, ...

    The wrapped loop is the vendored one: it derives episode i's seed through
    ``matching.full24_episode_seed``, which is offset by ``first`` for the
    duration of the call. The returned records are renumbered to match.
    """
    run_episodes = matching.run_matched_full24_episodes
    episode_seed = matching.full24_episode_seed

    def offset_seed(base_eval_seed: int, env_name: str, episode_index: int) -> int:
        return episode_seed(base_eval_seed, env_name, episode_index + first)

    def run_episode_range(**kwargs: Any):
        matching.full24_episode_seed = offset_seed
        try:
            successes, records = run_episodes(**kwargs)
        finally:
            matching.full24_episode_seed = episode_seed
        for record in records:
            record["episode_index"] += first
        return successes, records

    return run_episode_range


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    first = int(sys.argv[1])
    if first < 0:
        sys.exit("FIRST must be non-negative")
    script = os.path.abspath(sys.argv[2])
    # As `python simulation_env.py ...`: the script's directory first on sys.path, its arguments in argv.
    sys.path[0] = os.path.dirname(script)
    sys.argv = [script] + sys.argv[3:]
    # simulation_env.py's own import order up to the evaluation package (robocasa before any DiT4DiT module).
    for name in ("gymnasium", "numpy", "tyro", "robocasa", "robosuite"):
        importlib.import_module(name)
    matching = importlib.import_module(MATCHING)
    matching.run_matched_full24_episodes = episode_range_runner(matching, first)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
