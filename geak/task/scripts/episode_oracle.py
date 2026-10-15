# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact-match numerics pre-filter for the FastWAM 1024-dim episode task.

WHY THIS EXISTS (it did not on the MI250 version of this task)
--------------------------------------------------------------
`correctness` here was closed-loop LIBERO ONLY: ~6 min of real MuJoCo rollout for
every candidate, including candidates that are arithmetically broken and would
have been rejected by a 30-second tensor compare. This adds the cheap pre-filter
the sibling K8d16 task has, so `correctness` becomes:

    oracle       cheap pre-filter, catches arithmetic drift          (~1 min)
    closed-loop  the AUTHORITY, catches cross-episode state staleness (~6 min)

READ THIS BEFORE TRUSTING A GREEN ORACLE
----------------------------------------
The oracle is STRUCTURALLY BLIND to the bug class this campaign is most likely to
introduce. It replays ONE input, so a cached video-KV buffer or a captured HIP
graph that is never invalidated across episodes looks perfect here: the first
episode is the only episode. On the sibling model exactly that patch passed a
numerics oracle at rel 1.4e-2 and then scored end-to-end SR 0% (down from 100%).

So this file is a pre-filter and NOTHING MORE. A candidate that passes the oracle
has not been validated; it has merely earned the right to spend 6 minutes in
MuJoCo. Never promote the oracle to the authority to save wall-clock.

TOLERANCE IS MEASURED CROSS-PROCESS, AND THAT DISTINCTION IS THE WHOLE POINT
---------------------------------------------------------------------------
The first version of this file measured the model's self-consistency WITHIN one
process, got a clean 0.000e+00 twice, concluded "bit-deterministic", and set the
tolerance to exact equality. It then failed the UNMODIFIED tree.

Measured on MI355X: `infer_action` is bit-identical within a process (0.000e+00)
but two independent processes running byte-identical code differ by a fixed,
perfectly reproducible 6.250e-02 at 1 step and 3.125e-02 at 10. Against outputs
whose max magnitude is ~7.44 and ~4.84, those are **1-2 bf16 ULP** -- the
smallest representable difference, not a numeric divergence. Something in the
autotune/algorithm-selection path resolves per-process and then stays fixed.

The golden lives on disk and is therefore ALWAYS compared across a process
boundary, so the within-process figure is simply the wrong quantity. `--refresh`
records the golden three times in three separate processes, takes the worst
pairwise disagreement as the floor, and applies a 2x safety factor.

VERIFIED SENSITIVITY -- a gate nobody has tried to fail is not a gate
--------------------------------------------------------------------
Measured on this box, by perturbing `ActionDiT.post_dit` (which IS on the live
path -- note `ActionHead` in the same file is DEAD CODE and perturbing it changes
nothing, which is how the first attempt at this test fooled itself):

    unmodified src      max 3.1e-2 / tol 1.25e-1   mean 9.0e-3 / tol 2.25e-2  PASS
    post_dit x 1.01     max 1.3e-1 / tol 1.25e-1   mean 2.6e-2 / tol 2.25e-2  FAIL

The FAIL is carried by mean|d| (2.59e-2 vs 2.25e-2), and only at 1 step -- at 10
steps the same perturbation dilutes to a PASS. The margin is ~15%, so the honest
sensitivity floor is a systematic error of roughly 1%. BELOW ~1% THIS ORACLE
SEES NOTHING, and bf16's own resolution (2^-8, ~0.39%) is not far under that.

So: a cheap pre-filter that rejects obviously-wrong candidates before they cost
six minutes of MuJoCo. NOT a numerics guarantee. The closed-loop gate remains the
authority for every candidate that passes here.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

# Reuse the perf harness's loader/input builder verbatim. Sharing them is the
# point: an oracle that built its own inputs could drift from the shapes the
# scored path actually runs, and the whole task has already lost a round to a
# harness measuring a shape the model never sees.
from episode_perf import build_inputs, infer, load_model  # noqa: E402

GOLDEN = Path(os.environ.get(
    "ORACLE_GOLDEN",
    str(Path(__file__).resolve().parent.parent / "oracle_golden.pt"),
))
CASES = [int(s) for s in os.environ.get("ORACLE_STEPS", "1,10").split(",")]

# WARMUP IS LOAD-BEARING, NOT POLITENESS. Measured: with no warmup the golden and
# a check of the SAME BYTES disagree by a bit-identical 6.250e-02 every time --
# a stable cold-vs-warm offset, not noise. The refresh process was the first to
# touch a fresh tree (cold MIOpen conv autotune); every later check ran warm and
# they agreed with each other but not with the golden. The perf harness warms 25
# times for the same reason ("the VAE keeps autotuning its convs well past the
# 8th call"). Both legs must therefore reach the SAME steady state before the
# comparison means anything.
WARMUP = int(os.environ.get("ORACLE_WARMUP", "25"))


def _flatten(out):
    """infer_action may return a tensor or a container; compare all tensors."""
    if torch.is_tensor(out):
        return [out.detach().float().cpu()]
    if isinstance(out, dict):
        return [t for _, v in sorted(out.items()) for t in _flatten(v)]
    if isinstance(out, (list, tuple)):
        return [t for v in out for t in _flatten(v)]
    return []


def _maxdiff(a: list[torch.Tensor], b: list[torch.Tensor]) -> tuple[float, float]:
    """(max|d|, mean|d|).

    BOTH are needed and max alone is not enough. These outputs are bf16 with max
    magnitude ~7.4, where one ULP is 0.03125. Cross-process noise sits at 2 ULP
    (6.25e-2) and a 1% error on the same tensor is ~2.4 ULP (1.25e-1) -- so on
    max|d| a real 1% regression and pure noise are one step apart and no threshold
    separates them. mean|d| does separate them: ULP noise moves a handful of
    elements, while a systematic error moves EVERY element, so the mean floor is
    ~an order of magnitude below the mean of a real regression. Measured here:
    noise mean 3.3e-3 vs 1%-perturbation mean 3.4e-2, a 10x gap.

    The third quantity, max relative, is DIAGNOSTIC ONLY -- it divides by
    near-zero action components and reports absurdities like 1.7e+04. Never gate
    on it.
    """
    if len(a) != len(b):
        raise AssertionError(f"output arity changed: golden {len(a)} vs candidate {len(b)}")
    abs_d = 0.0
    tot, n = 0.0, 0
    for x, y in zip(a, b):
        if x.shape != y.shape:
            raise AssertionError(f"output shape changed: golden {tuple(x.shape)} vs {tuple(y.shape)}")
        d = (x - y).abs()
        abs_d = max(abs_d, float(d.max()))
        tot += float(d.sum())
        n += d.numel()
    return abs_d, (tot / n if n else 0.0)


def _warm(model, inputs) -> None:
    for steps in CASES:
        for _ in range(WARMUP):
            infer(model, *inputs, steps)
    torch.cuda.synchronize()


SAFETY = float(os.environ.get("ORACLE_SAFETY", "2.0"))


def refresh() -> int:
    """Record ONE sample of the golden. Calibration across samples is `--calibrate`."""
    model = load_model()
    inputs = build_inputs(model)
    _warm(model, inputs)
    golden, selfdiff = {}, 0.0
    for steps in CASES:
        a = _flatten(infer(model, *inputs, steps))
        b = _flatten(infer(model, *inputs, steps))   # same inputs, same seed
        d, _ = _maxdiff(a, b)
        selfdiff = max(selfdiff, d)
        golden[str(steps)] = a
        print(f"[oracle:refresh] steps={steps} within-process max|d|={d:.3e}", flush=True)

    torch.save({"cases": golden, "within_process_abs": selfdiff,
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0)}, GOLDEN)
    print(f"[oracle:refresh] wrote {GOLDEN}", flush=True)
    print("ORACLE_REFRESH_DONE", flush=True)
    return 0


def calibrate(paths: list[str]) -> int:
    """Fold N independently-recorded samples into the golden the gate will use.

    The tolerance is the worst pairwise disagreement between samples -- i.e. the
    floor this comparison can actually resolve across a process boundary -- times
    a safety factor. Sample 0's tensors become the reference.
    """
    samples = [torch.load(p, weights_only=False) for p in paths]
    floor, floor_mean = 0.0, 0.0
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            for k in samples[i]["cases"]:
                d, m = _maxdiff(samples[i]["cases"][k], samples[j]["cases"][k])
                floor, floor_mean = max(floor, d), max(floor_mean, m)
                print(f"[oracle:calibrate] sample{i} vs sample{j} steps={k} "
                      f"max|d|={d:.3e} mean|d|={m:.3e}", flush=True)

    tol, tol_mean = floor * SAFETY, floor_mean * SAFETY
    out = dict(samples[0])
    out["cross_process_abs"] = floor
    out["cross_process_mean"] = floor_mean
    out["tolerance_abs"] = tol
    out["tolerance_mean"] = tol_mean
    out["n_samples"] = len(samples)
    torch.save(out, GOLDEN)
    print(f"[oracle:calibrate] cross-process floor: max={floor:.3e} mean={floor_mean:.3e} "
          f"(x{SAFETY:g}) -> tol max={tol:.3e} mean={tol_mean:.3e}", flush=True)
    print(f"[oracle:calibrate] wrote {GOLDEN}", flush=True)
    print("ORACLE_CALIBRATE_DONE", flush=True)
    return 0


def check() -> int:
    if not GOLDEN.exists():
        print(f"[oracle] no golden at {GOLDEN} -- run `oracle-refresh` on a pristine tree first")
        return 1
    g = torch.load(GOLDEN, weights_only=False)
    if "tolerance_abs" not in g and "ORACLE_TOL" not in os.environ:
        print("[oracle] golden was never calibrated -- run `oracle-refresh`, which "
              "records 3 samples and measures the cross-process floor")
        return 1
    # The gate's floor is the CROSS-process disagreement, because the golden is
    # always read back from disk. Using the within-process figure (0.0 here) fails
    # the unmodified tree -- measured, not hypothetical.
    tol = float(os.environ.get("ORACLE_TOL", g.get("tolerance_abs", 0.0)))
    tol_mean = float(os.environ.get("ORACLE_TOL_MEAN", g.get("tolerance_mean", 0.0)))

    model = load_model()
    inputs = build_inputs(model)
    _warm(model, inputs)   # same steady state the golden was recorded in
    worst, worst_mean = 0.0, 0.0
    for steps in CASES:
        key = str(steps)
        if key not in g["cases"]:
            print(f"[oracle] golden has no case steps={steps}")
            return 1
        abs_d, mean_d = _maxdiff(g["cases"][key], _flatten(infer(model, *inputs, steps)))
        worst, worst_mean = max(worst, abs_d), max(worst_mean, mean_d)
        ok = abs_d <= tol + 1e-12 and mean_d <= tol_mean + 1e-12
        print(f"[oracle] steps={steps} max|d|={abs_d:.3e}/{tol:.3e} "
              f"mean|d|={mean_d:.3e}/{tol_mean:.3e} -> {'PASS' if ok else 'FAIL'}", flush=True)

    passed = worst <= tol + 1e-12 and worst_mean <= tol_mean + 1e-12
    print(f"[oracle] worst max|d|={worst:.3e}/{tol:.3e} "
          f"mean|d|={worst_mean:.3e}/{tol_mean:.3e} -> "
          f"{'PASS' if passed else 'FAIL'}", flush=True)
    if passed:
        print("ORACLE_OK", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--refresh" in argv:
        sys.exit(refresh())
    if "--calibrate" in argv:
        sys.exit(calibrate(argv[argv.index("--calibrate") + 1:]))
    sys.exit(check())