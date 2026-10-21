#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare optimization artifacts in ONE run: eager, torch.compile, GEAK, reduced steps.

    python3 scripts/compare_artifacts.py --arms eager,compile,geak --steps 10
    python3 scripts/compare_artifacts.py --arms eager,compile,geak,geak@2 --steps 10

LATENCY ONLY. Each arm is one `geak/task/run_local.sh performance` container that
times FastWAM.infer_action on synthetic inputs (25 warmup calls, median of 40).
The arms' outputs are never compared: check numerics with the GEAK task's
`run_local.sh oracle` / `correctness` gates and accuracy with the evaluation
launchers. Prerequisites: docs/fastwam-geak-inference-optimization.md.

WHY ONE SCRIPT. Latency is not portable across nodes, driver versions or even
sessions -- two nodes on an identical stack can differ substantially.
Numbers collected in separate runs are therefore not comparable, and a table
assembled from them is misleading. This measures every arm in a SINGLE session,
INTERLEAVED (for round: for arm), so all arms share one drift epoch and the
comparison survives whatever the box was doing at the time.

Nothing is committed. Re-run it on the machine you care about.

ARMS
  eager         unmodified sources, ALL compile disabled. Note that "unmodified
                sources" is NOT eager: these trees commonly enable torch.compile
                by default, so the baseline must disable it explicitly or it is
                already an optimized arm. Getting this wrong makes every later
                speedup look smaller than it is.
  compile       naive torch.compile at default settings, from the eager tree.
                Note `torch.compile(model)` compiles forward(); if the entry
                point is another method it does nothing at all -- compile the
                callable actually invoked. Timed at --steps only (compile@N
                with another N is refused), with a compile cache of its own;
                the cell fails unless dynamo compiled graphs and the timed
                calls did not recompile.
  geak          the post-GEAK source tree, timed as eval/native/run_fastwam_libero.sh
                evaluates it: FASTWAM_KVSLAB=0 and a compile cache of its own
                (the K/V slab lever trips an inductor lowering bug, and the
                inductor cache key does not see the switch).

Append @<steps> to an arm to time it at that denoise-step count instead of
--steps (geak@2), as scripts/cae_fastwam_latency.py spells it. The former
geak-reduced arm (geak at --reduced-steps, default 2) still works with a
deprecation warning.

Source trees: BASELINE_SRC (eager, compile; default src/fastwam-baseline) and
OPTIMIZED_SRC (geak; default src/fastwam-geak), the FastWAM baseline and GEAK
trees the eval launcher serves. Those trees are timed with
FastWAM/configs_baseline; any other tree with MODEL_SRC_ROOT/configs, unless
FWE_CONFIG_DIR names a config tree. --task-dir defaults to GEAK_TASK_DIR, else the repository's
geak/task.

--gpus takes the one HIP device to time on (default: EVAL_GPUS, else 0); the
former --gpu still works with a deprecation warning.
EVAL_DOCKER_ARGS adds `docker run` flags to every timed container, e.g.
EVAL_DOCKER_ARGS="--cpuset-cpus 104-127" pins it to the CPUs of the GPU's NUMA node
(the reference latency runs pinned each container that way, away from other work).

ENVIRONMENT (geak/task/run_local.sh, which runs each cell)
  required  EVAL_IMAGE        the evaluation image (docker/Dockerfile.eval)
            MODEL_SRC_ROOT    a FastWAM checkout (this repository's FastWAM/)
            MODEL_CKPT        FastWAM checkpoint (yuanty/fastwam libero_uncond_2cam224.pt)
            DATASET_STATS     its statistics (libero_uncond_2cam224_dataset_stats.json)
            HF_CACHE_DIR      an existing writable cache directory on the host
            MODEL_PYTHON      the image's Python (/opt/venv/bin/python3)
  optional  DIFFSYNTH_MODEL_BASE_PATH  the Wan-AI files (MODEL_ZOO.md; default HF_CACHE_DIR)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_flags import deprecated, resolve_one_gpu  # noqa: E402  (scripts/eval_flags.py)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The FastWAM baseline and GEAK trees eval/native/run_fastwam_libero.sh evaluates,
# when the source variables are unset.
DEFAULT_SRC = {
    "BASELINE_SRC": os.path.join(REPO, "src", "fastwam-baseline"),
    "OPTIMIZED_SRC": os.path.join(REPO, "src", "fastwam-geak"),
}
# The FastWAM baseline and GEAK trees take the older config schema of
# FastWAM/configs_baseline; the GEAK task otherwise composes
# MODEL_SRC_ROOT/configs, which FastWAM/src (and src/cae-fastwam) takes.
BASELINE_SCHEMA_TREES = tuple(DEFAULT_SRC.values())
BASELINE_CONFIG_DIR = os.path.join(REPO, "FastWAM", "configs_baseline")

ARM_SPECS = {
    # name: (src env var, extra env, description)
    "eager":        ("BASELINE_SRC",  {"FASTWAM_COMPILE": "0",
                                       "FASTWAM_MOT_COMPILE_MODE": "off",
                                       "FASTWAM_VAE_COMPILE": "0"},
                     "unmodified sources, all compile off"),
    "compile":      ("BASELINE_SRC",  {"FASTWAM_COMPILE": "0",
                                       "FASTWAM_MOT_COMPILE_MODE": "off",
                                       "FASTWAM_VAE_COMPILE": "0",
                                       "FWE_NAIVE_COMPILE": "1"},
                     "naive torch.compile of the inference entry point"),
    # The eval launcher always runs the FastWAM GEAK tree with the K/V slab
    # off; time the same configuration it evaluates.
    "geak":         ("OPTIMIZED_SRC", {"FASTWAM_KVSLAB": "0"}, "post-GEAK tree"),
    "geak-reduced": ("OPTIMIZED_SRC", {"FASTWAM_KVSLAB": "0"},
                     "post-GEAK tree, reduced steps"),
}
# Arms that get a compile cache of their own for the session: a cache that ever
# held a slab-on build would serve it back, whatever FASTWAM_KVSLAB says; the
# naive compile arm must not reuse kernels another tree or session compiled.
FRESH_CACHE_ARMS = ("compile", "geak", "geak-reduced")
# Arms timed only at the --steps (baseline) denoise-step count.
BASELINE_STEPS_ONLY_ARMS = ("compile",)


def parse_arm(token: str, steps: int, reduced_steps: int) -> tuple[str, str, int]:
    """An --arms entry -> (label, ARM_SPECS name, steps).

    `geak@2` -> ("geak@2", "geak", 2); `geak` -> ("geak", "geak", STEPS); the
    deprecated `geak-reduced` -> ("geak-reduced", "geak-reduced", REDUCED_STEPS).
    """
    name, _, at = token.partition("@")
    if name not in ARM_SPECS:
        raise ValueError(f"unknown arm {name!r}; known: {', '.join(ARM_SPECS)}")
    if name == "geak-reduced":
        if at:
            raise ValueError("geak-reduced takes no @<steps>; use geak@<steps>")
        deprecated("the geak-reduced arm", "geak@<steps>")
        return token, name, reduced_steps
    if at:
        if not at.isdigit() or int(at) < 1:
            raise ValueError(f"arm steps must be a positive integer, got: {token}")
        if name in BASELINE_STEPS_ONLY_ARMS and int(at) != steps:
            raise ValueError(
                f"{token}: the naive torch.compile arm is timed at the baseline step "
                f"count only (--steps {steps}); use {name} or {name}@{steps}")
        return token, name, int(at)
    return token, name, steps


def arm_src(arm: str) -> str:
    """The source tree ARM times: its variable, else the launcher's tree for it."""
    var = ARM_SPECS[arm][0]
    return os.environ.get(var) or DEFAULT_SRC[var]


def config_dir_for(src: str) -> str | None:
    """The config tree SRC needs: configs_baseline for the FastWAM baseline or
    GEAK tree (or a path inside one), else None (the GEAK task's default,
    MODEL_SRC_ROOT/configs)."""
    path = os.path.realpath(src)
    for tree in BASELINE_SCHEMA_TREES:
        root = os.path.realpath(tree)
        if os.path.commonpath([path, root]) == root:
            return BASELINE_CONFIG_DIR
    return None


def arm_env(arm: str, steps: int, gpu: str, src: str, caches: dict[str, str],
            key: str | None = None) -> dict[str, str]:
    """The environment one `performance` invocation of ARM runs with.

    KEY (default ARM) selects the arm's own compile cache in CACHES.
    """
    key = arm if key is None else key
    env = dict(os.environ, HIP_VISIBLE_DEVICES=gpu, FWE_STEPS=str(steps),
               FWE_SRC=src, **ARM_SPECS[arm][1])
    # An explicit FWE_CONFIG_DIR wins; otherwise follow the source tree.
    if not os.environ.get("FWE_CONFIG_DIR"):
        cfg = config_dir_for(src)
        if cfg:
            env["FWE_CONFIG_DIR"] = cfg
        else:
            env.pop("FWE_CONFIG_DIR", None)
    if key in caches:
        env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(caches[key], "inductor")
        env["TRITON_CACHE_DIR"] = os.path.join(caches[key], "triton")
    return env


def measure(label: str, arm: str, steps: int, gpu: str, task_dir: str,
            caches: dict[str, str]) -> float | None:
    """One `performance` invocation; returns the scored median ms, or None."""
    src = arm_src(arm)
    env = arm_env(arm, steps, gpu, src, caches, key=label)
    p = subprocess.run(["bash", os.path.join(task_dir, "run_local.sh"), "performance"],
                       env=env, capture_output=True, text=True, timeout=7200)
    if p.returncode != 0:
        tail = "\n".join((p.stdout + p.stderr).splitlines()[-60:])
        print(f"  [{label}] performance command failed (rc={p.returncode})\n{tail}",
              file=sys.stderr)
        return None
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("Perf:"):
            try:
                return float(line.split()[1])
            except (IndexError, ValueError):
                pass
    print(f"  [{label}] no Perf line (rc={p.returncode})", file=sys.stderr)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 1)[1])
    ap.add_argument("--arms", default="eager,compile,geak")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--reduced-steps", type=int, default=2,
                    help="steps of the deprecated geak-reduced arm (use geak@N)")
    ap.add_argument("--rounds", type=int, default=3,
                    help="interleaved rounds; 3 is the minimum that shows spread")
    ap.add_argument("--gpus", default=None,
                    help="HIP device to time on, one id (default: EVAL_GPUS, else 0)")
    ap.add_argument("--gpu", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--task-dir",
                    default=os.environ.get("GEAK_TASK_DIR") or os.path.join(REPO, "geak", "task"))
    ap.add_argument("--json-out", default="")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    try:
        cells = [parse_arm(x, a.steps, a.reduced_steps) for x in a.arms.split(",") if x]
        gpu = resolve_one_gpu(a.gpus, a.gpu)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    if len({label for label, _, _ in cells}) != len(cells):
        print("duplicate arm in --arms", file=sys.stderr)
        return 2
    caches = {label: tempfile.mkdtemp(prefix=f"compare_artifacts_{arm}_")
              for label, arm, _ in cells if arm in FRESH_CACHE_ARMS}
    try:
        return run_arms(a, cells, gpu, caches)
    finally:
        for path in caches.values():
            shutil.rmtree(path, ignore_errors=True)


def run_arms(a, cells: list[tuple[str, str, int]], gpu: str, caches: dict[str, str]) -> int:

    # INTERLEAVED: for round -> for arm. Never all rounds of one arm in a block;
    # arms measured in the same round share a drift epoch, which is the whole
    # point of running them together.
    samples: dict[str, list[float]] = {label: [] for label, _, _ in cells}
    measurement_failed = False
    for r in range(a.rounds):
        for label, arm, steps in cells:
            v = measure(label, arm, steps, gpu, a.task_dir, caches)
            if v is not None:
                samples[label].append(v)
            else:
                measurement_failed = True
            if not a.quiet:
                print(f"  round {r + 1} {label:<14} {v if v is None else f'{v:.3f} ms'}")

    rows = []
    for label, _, steps in cells:
        s = samples[label]
        if not s:
            continue
        med = statistics.median(s)
        spread = (max(s) - min(s)) / min(s) * 100 if len(s) > 1 else 0.0
        rows.append({"arm": label, "median_ms": med, "spread_pct": spread,
                     "steps": steps, "samples": s})

    if not rows:
        print("no successful measurements", file=sys.stderr)
        return 1

    ref = rows[0]["median_ms"]
    if not a.quiet:
        print(f"\n{'arm':<14}{'steps':>6}{'median_ms':>12}{'spread':>9}{'vs ' + rows[0]['arm']:>14}")
    for row in rows:
        row["speedup_vs_first"] = ref / row["median_ms"]
        if a.quiet:
            print(f"{row['arm']} {row['median_ms']:.3f}")
        else:
            print(f"{row['arm']:<14}{row['steps']:>6}{row['median_ms']:>12.3f}"
                  f"{row['spread_pct']:>8.2f}%{row['speedup_vs_first']:>13.3f}x")

    if not a.quiet:
        print("\nSpeedups are relative to the FIRST arm listed. If that is not `eager`,"
              "\nsay so when quoting them: 'Nx over torch.compile' and 'Nx over eager'"
              "\nare different claims and differ substantially.")
        print("Treat differences below the reported spread as unresolved.")

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"rounds": a.rounds, "gpu": gpu, "rows": rows}, f, indent=2)
        if not a.quiet:
            print(f"\nwrote {a.json_out}")
    if measurement_failed:
        print("one or more requested measurements failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
