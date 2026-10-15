#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool success rates across shards and suites for every arm in a results root.

Reports POOLED SR (episode-weighted), not the mean of per-suite means: the
suites contribute unequal episode counts once a subset or an uneven shard split
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
    per = defaultdict(lambda: [0.0, 0])
    for f in sorted(d.glob("LIBEROBenchmark_*.json")):
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        suite = f.name.split("LIBEROBenchmark_")[1].split("_sync")[0].split("_shard")[0]
        for t in j.get("tasks", []):
            n = t.get("num_episodes", 0)
            if not n:
                continue
            per[suite][0] += (t.get("mean_success") or 0.0) * n
            per[suite][1] += n
    return per


def main(root: str) -> int:
    rp = pathlib.Path(root)
    # Dot-directories are launcher scratch (e.g. .compile_cache), not arms.
    arms = sorted(p for p in rp.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not arms:
        print(f"no result directories under {root}")
        return 1
    for a in arms:
        per = tally(a)
        succ = sum(v[0] for v in per.values())
        eps = sum(v[1] for v in per.values())
        if not eps:
            print(f"{a.name:<24} no episodes")
            continue
        sr = succ / eps
        # binomial standard error, so a reader can size the difference that
        # actually matters rather than reading noise as a result
        se = math.sqrt(max(sr * (1 - sr), 0.0) / eps)
        print(f"{a.name:<24} pooled SR={sr:.4f} +/-{se:.4f}  ({int(round(succ))}/{eps} eps)")
        for s, (x, n) in sorted(per.items()):
            print(f"    {s:<18} {x / n:.4f}  ({int(round(x))}/{n})")
    print("\nDifferences smaller than ~2 standard errors are not distinguishable.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
