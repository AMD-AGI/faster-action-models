#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool success rates across tasks and suites for every run directory under a results root.

A run directory is one launcher run (eval/native/run_fastwam_libero.sh --output-dir): FastWAM's evaluator
writes <run>/<suite>/gpu<G>_task<T>_results.json. Usage: python3 eval/summarize.py ROOT, where ROOT holds run
directories (a scripts/reproduce_*.sh REPRO_DIR, say) or is one itself.

Reports POOLED SR (episode-weighted), not the mean of per-suite means: the
suites contribute unequal episode counts once a subset of tasks or trials
is used, and averaging the means silently reweights them.

Per-suite is always printed alongside, because degradation is rarely uniform --
reduced denoise steps typically cost accuracy on long-horizon and goal-
conditioned suites while leaving object-centric suites untouched, and a single
pooled number hides exactly that.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import defaultdict


def tally(d: pathlib.Path):
    """Per-suite [successes, episodes] of one run directory."""
    per = defaultdict(lambda: [0.0, 0])
    for f in sorted(d.glob("*/gpu*_task*_results.json")):
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        n = int(j.get("total_episodes", 0))
        if not n:
            continue
        per[f.parent.name][0] += int(j.get("successes", 0))
        per[f.parent.name][1] += n
    return per


def main(root: str) -> int:
    rp = pathlib.Path(root)
    # Dot-directories are launcher scratch (e.g. .compile_cache), not runs.
    if any(rp.glob("*/gpu*_task*_results.json")):
        arms = [rp]  # ROOT is itself one run directory
    else:
        arms = sorted(p for p in rp.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not arms:
        print(f"no result directories under {root}")
        return 1
    for a in arms:
        per = tally(a)
        succ = sum(v[0] for v in per.values())
        eps = sum(v[1] for v in per.values())
        if not eps:
            print(f"{a.name:<40} no episodes")
            continue
        sr = succ / eps
        # binomial standard error, so a reader can size the difference that
        # actually matters rather than reading noise as a result
        se = math.sqrt(max(sr * (1 - sr), 0.0) / eps)
        print(f"{a.name:<40} pooled SR={sr:.4f} +/-{se:.4f}  ({int(round(succ))}/{eps} eps)")
        for s, (x, n) in sorted(per.items()):
            print(f"    {s:<18} {x / n:.4f}  ({int(round(x))}/{n})")
    print("\nDifferences smaller than ~2 standard errors are not distinguishable.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
