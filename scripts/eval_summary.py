#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The common end of every evaluation launcher.

Each launcher ends by calling this with the run's labels and where its results
are. It writes ``<output-dir>/summary.json`` (pooled successes and episodes,
and the same per suite or per task) and prints one closing line:

    eval complete: <model> <bench> <arm> <N>step <sim>: S/E -> <dir>

("eval incomplete" when the launcher's run failed; the counts are then those of
the episodes that finished). The result sources it reads:

  --native DIR     FastWAM evaluator <suite>/gpu*_task*_results.json files
  --counts K=S/E   counts the launcher tallied itself (DiT4DiT LIBERO), repeated
  --robocasa FILE  the full-24 summarizer's JSON output (DiT4DiT RoboCasa)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA_VERSION = 1


def native_counts(out_dir: Path) -> dict[str, tuple[int, int]]:
    """Per-suite (successes, episodes) of FastWAM evaluator result files."""
    per: dict[str, list[int]] = {}
    for path in sorted(out_dir.glob("*/gpu*_task*_results.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        row = per.setdefault(path.parent.name, [0, 0])
        row[0] += int(payload.get("successes", 0))
        row[1] += int(payload.get("total_episodes", 0))
    return {suite: (s, n) for suite, (s, n) in per.items()}


def parse_counts(items: list[str]) -> dict[str, tuple[int, int]]:
    """["libero_goal=S/E", ...] -> {"libero_goal": (S, E), ...}."""
    counts = {}
    for item in items:
        match = re.fullmatch(r"([^=]+)=(\d+)/(\d+)", item)
        if not match:
            raise ValueError(f"--counts must be NAME=SUCCESSES/EPISODES, got: {item}")
        counts[match.group(1)] = (int(match.group(2)), int(match.group(3)))
    return counts


def robocasa_counts(summary_json: Path) -> dict[str, tuple[int, int]]:
    """Per-task (successes, episodes) of summarize_full24_policy.py's output."""
    payload = json.loads(summary_json.read_text())
    return {row["label"]: (int(row["successes"]), int(row["episodes"])) for row in payload.get("tasks", [])}


def rate(successes: int, episodes: int) -> float | None:
    return successes / episodes if episodes else None


def build_summary(meta: dict, groups: dict[str, tuple[int, int]], group_kind: str,
                  complete: bool) -> dict:
    """The summary.json payload: META labels, pooled counts and GROUP_KIND rows."""
    successes = sum(s for s, _ in groups.values())
    episodes = sum(n for _, n in groups.values())
    return {
        "schema_version": SCHEMA_VERSION,
        **meta,
        "status": "complete" if complete else "incomplete",
        "successes": successes,
        "episodes": episodes,
        "success_rate": rate(successes, episodes),
        f"per_{group_kind}": {
            name: {"successes": s, "episodes": n, "success_rate": rate(s, n)}
            for name, (s, n) in sorted(groups.items())
        },
    }


def steps_label(steps: str) -> str:
    return f"{steps}step" if steps.isdigit() else steps


def closing_line(summary: dict) -> str:
    cell = [summary["model"], summary["benchmark"], summary["arm"],
            steps_label(str(summary["steps"])), summary["sim"]]
    if summary.get("mode"):
        cell.append(summary["mode"])
    return (f"eval {summary['status']}: {' '.join(cell)}: "
            f"{summary['successes']}/{summary['episodes']} -> {summary['output_dir']}")


def write_summary(output_dir: Path, summary: dict) -> Path:
    path = output_dir / "summary.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    tmp.replace(path)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--model", "--bench", "--arm", "--steps", "--sim", "--output-dir"):
        ap.add_argument(name, required=True)
    ap.add_argument("--mode", default="")
    ap.add_argument("--trials", type=int)
    ap.add_argument("--rc", type=int, default=0, help="the run's exit status (non-zero: incomplete)")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--native", type=Path)
    source.add_argument("--counts", nargs="+")
    source.add_argument("--robocasa", type=Path)
    a = ap.parse_args(argv)

    try:
        if a.native is not None:
            groups, kind = native_counts(a.native), "suite"
        elif a.counts is not None:
            groups, kind = parse_counts(a.counts), "suite"
        else:
            groups, kind = robocasa_counts(a.robocasa), "task"
    except (OSError, ValueError, KeyError) as exc:
        print(f"eval_summary: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(a.output_dir)
    meta = {"model": a.model, "benchmark": a.bench, "arm": a.arm,
            "steps": int(a.steps) if a.steps.isdigit() else a.steps, "sim": a.sim,
            "mode": a.mode or None, "trials_per_task": a.trials, "output_dir": str(output_dir)}
    summary = build_summary(meta, groups, kind, complete=a.rc == 0)
    write_summary(output_dir, summary)
    print(closing_line(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
