# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end `FastWAM.infer_action` latency — the number that actually ships.

Runs the REAL model (real checkpoint, real VAE, real 2-camera observation) and
times the whole action-inference call, not just the denoise step. Two cases:

    episode_1step   num_inference_steps=1   -> prologue-dominated
    episode_10step  num_inference_steps=10  -> denoise-loop-dominated

Both are scored (geomean). Covering both regimes is deliberate: a patch that
speeds the 30-layer action step but slows the once-per-episode video prefill
looks like a win at 10 steps and a loss at 2, and only measuring one of them
hides that.

Stage breakdown (VAE / pre_dit / prefill / per-step) is measured in a SECOND
pass, because the per-stage syncs it needs would inflate the primary total.

Random text context is fine: this measures kernel + dispatch time, not task
success. Success rate is the `correctness` command's job (real LIBERO episodes).
"""
import json
import os
import statistics
import sys
import time

CKPT = os.environ["FWE_CKPT"]
CONFIG_DIR = os.environ["FWE_CONFIG_DIR"]
# 25 warmups, not the usual 8: the VAE encode keeps autotuning its convs well past
# the 8th call, which put ~13% of samples at 2.3x the median (p90 439 vs median 193).
# At 25 the 2-step p90 sits within 10% of the median.
WARMUP = int(os.environ.get("FWE_WARMUP", "25"))
ITERS = int(os.environ.get("FWE_ITERS", "40"))
STAGE_ITERS = int(os.environ.get("FWE_STAGE_ITERS", "10"))
# Scored at 1 and 10 denoise steps. 1 step is the most prologue-dominated regime
# there is (the VAE + video prefill are paid in full for a single step), so it is
# where a prologue win shows up largest; 10 steps is what the correctness gate's
# server actually runs and is denoise-dominated. Optimizing the geomean of the two
# means neither stage can be traded away for the other.
CASES = [int(s) for s in os.environ.get("FWE_STEPS", "1,10").split(",")]

# Deployed observation: agentview + wrist, each center-cropped to 224x224 and
# concatenated horizontally. This is what makes Sv 98 (7x14 patches) rather than
# the 49 a single 224x224 camera would give.
IMG_H, IMG_W = 224, 448
# Deployed context is 129: 128 text tokens + 1 proprio token appended by
# _append_proprio_to_context. Both asserted below against what prefill receives.
TEXT_CTX_LEN, TEXT_DIM, ACTION_HORIZON = 128, 4096, 32
EXPECT_SV, EXPECT_CTX = 98, 129

import torch  # noqa: E402  (after env reads so a missing var fails fast and cheap)


def load_model():
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("max", lambda x: max(x), replace=True)
    OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)], replace=True)
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="sim_libero",
            overrides=[f"ckpt={CKPT}", "model.load_text_encoder=false"],
        )
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(cfg.ckpt))
    return model.to("cuda").eval()


def build_inputs(model):
    g = torch.Generator(device="cpu").manual_seed(1234)
    img = (torch.rand((1, 3, IMG_H, IMG_W), generator=g) * 2 - 1).to("cuda", torch.bfloat16)
    ctx = torch.randn((1, TEXT_CTX_LEN, TEXT_DIM), generator=g).to("cuda", torch.bfloat16)
    mask = torch.ones((1, TEXT_CTX_LEN), dtype=torch.bool, device="cuda")
    pd = getattr(model, "proprio_dim", None)
    proprio = None if not pd else torch.randn((1, int(pd)), generator=g).to("cuda", torch.bfloat16)
    return img, ctx, mask, proprio


def require_method(owner, name, owner_label):
    """Return a required callable from the active FastWAM inference API."""
    method = getattr(owner, name, None)
    if not callable(method):
        raise RuntimeError(
            "FastWAM API mismatch: required callable "
            f"{owner_label}.{name} is absent"
        )
    return method


def install_shape_probe(model, seen):
    """Record the shapes prefill actually receives, so the harness cannot lie.

    One dict write per episode; no sync, so it does not perturb the timing.
    """
    mot = getattr(model, "mot", None)
    if mot is None:
        raise RuntimeError("FastWAM API mismatch: required model.mot is absent")
    # Trees prefill through either mot.prefill_video_cache_tensor(video_tokens=...,
    # video_context=...) or mot.prefill_video_cache(video_tokens, ...,
    # video_context_payload={"context": ...}). Probe whichever the tree has.
    name = ("prefill_video_cache_tensor" if hasattr(mot, "prefill_video_cache_tensor")
            else "prefill_video_cache")
    orig = require_method(mot, name, "model.mot")

    def wrapped(*a, **k):
        vt = k.get("video_tokens", a[0] if a else None)
        if vt is not None:
            seen["Sv"] = int(vt.shape[1])
        c = k.get("video_context")
        if c is None:
            pl = k.get("video_context_payload", a[3] if len(a) > 3 else None)
            c = pl.get("context") if isinstance(pl, dict) else None
        if c is not None:
            seen["ctx"] = int(c.shape[1])
        return orig(*a, **k)

    setattr(mot, name, wrapped)
    seen["_probe"] = name
    return orig


# NAIVE torch.compile arm (FWE_NAIVE_COMPILE=1).
#
# "Naive" = what you get with zero effort and no knowledge of the model. Note
# that the obvious `torch.compile(model)` does NOTHING here: it compiles
# `forward()`, and the entry point is `infer_action()`. So the honest naive
# application is to compile the callable actually invoked, at default settings --
# no mode tuning, no region selection, no graph capture, no fullgraph.
#
# This is measured from the TRULY EAGER tree (FASTWAM_COMPILE=0,
# MOT_COMPILE_MODE=off, VAE_COMPILE=0), so it isolates what compile alone buys
# over eager -- distinct from the pristine tree's OWN shipped torch.compile,
# which already targets the video prefill and the denoise loop specifically.
_NAIVE = os.environ.get("FWE_NAIVE_COMPILE", "0") == "1"
_naive_fn = None


def infer(model, img, ctx, mask, proprio, steps):
    global _naive_fn
    if _NAIVE:
        if _naive_fn is None:
            _naive_fn = torch.compile(model.infer_action)
        with torch.no_grad():
            return _naive_fn(
                prompt=None, input_image=img, action_horizon=ACTION_HORIZON,
                proprio=proprio, context=ctx, context_mask=mask,
                num_inference_steps=steps, seed=42, rand_device="cpu",
            )
    with torch.no_grad():
        return model.infer_action(
            prompt=None,
            input_image=img,
            action_horizon=ACTION_HORIZON,
            proprio=proprio,
            context=ctx,
            context_mask=mask,
            num_inference_steps=steps,
            seed=42,
            rand_device="cpu",
        )


def time_total(model, inputs, steps):
    for _ in range(WARMUP):  # triggers torch.compile / graph capture if the patch adds them
        infer(model, *inputs, steps)
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        infer(model, *inputs, steps)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return ts


def eager_call(model, img, ctx, mask, proprio, steps):
    with torch.no_grad():
        return model.infer_action(
            prompt=None, input_image=img, action_horizon=ACTION_HORIZON,
            proprio=proprio, context=ctx, context_mask=mask,
            num_inference_steps=steps, seed=42, rand_device="cpu",
        )


def dynamo_snapshot():
    """torch._dynamo counters that show whether (re)compilation happened."""
    from torch._dynamo.utils import counters
    try:
        from torch._dynamo.utils import compilation_time_metrics
        compiles = len(compilation_time_metrics.get("_compile.compile_inner", []))
    except ImportError:
        compiles = 0
    return {"frames": int(counters["frames"].get("total", 0)),
            "unique_graphs": int(counters["stats"].get("unique_graphs", 0)),
            "graph_breaks": int(sum(counters["graph_break"].values())),
            "compiles": compiles}


def naive_case(model, inputs, steps, probe_name, probe_orig):
    """The naive torch.compile arm, with proof that it compiled.

    The shape probe is checked on an eager call first; the probe wrapper is then
    removed so torch.compile traces the unmodified tree, not our Python closure.
    The first compiled call is timed on its own, then the same WARMUP + ITERS
    loop as time_total. Refuses the cell if nothing was compiled or the timed
    calls recompiled.
    """
    eager_call(model, *inputs, steps)
    torch.cuda.synchronize()
    setattr(model.mot, probe_name, probe_orig)
    t0 = time.perf_counter()
    infer(model, *inputs, steps)
    torch.cuda.synchronize()
    first_call_s = time.perf_counter() - t0
    s1 = dynamo_snapshot()
    # > 0 rather than "grew": a second FWE_STEPS case reuses the compiled fn.
    if s1["unique_graphs"] == 0:
        raise AssertionError(f"naive torch.compile produced no graphs: {s1}")
    for _ in range(WARMUP - 1):
        infer(model, *inputs, steps)
    torch.cuda.synchronize()
    s2 = dynamo_snapshot()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        infer(model, *inputs, steps)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    s3 = dynamo_snapshot()
    if s3 != s2:
        raise AssertionError(f"the timed calls compiled or recompiled: {s2} -> {s3}")
    info = {"target": "model.infer_action (torch.compile at defaults)",
            "first_call_s": first_call_s, "unique_graphs": s3["unique_graphs"],
            "graph_breaks": s3["graph_breaks"],
            "warmup_recompiles": s2["frames"] - s1["frames"],
            "timed_recompiled": False}
    print("COMPILE_JSON " + json.dumps(info), flush=True)
    return ts, info


def time_stages(model, inputs, steps):
    """Per-stage medians. Diagnostics only — the syncs here inflate the total."""
    stage = {"vae": [], "pre_dit": [], "prefill": []}
    steps_ms = {}
    idx = {"i": 0}
    o_vae = model._encode_input_image_latents_tensor
    o_pre = model.video_expert.pre_dit
    o_pf = require_method(
        model.mot, "prefill_video_cache_tensor", "model.mot"
    )
    o_denoise = require_method(
        model, "_denoise_action_with_video_cache", "model"
    )

    def timed(key, fn):
        def w(*a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = fn(*a, **k)
            torch.cuda.synchronize()
            stage[key].append((time.perf_counter() - t0) * 1000.0)
            return r
        return w

    def timed_step(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = o_denoise(*a, **k)
        torch.cuda.synchronize()
        steps_ms.setdefault(idx["i"], []).append((time.perf_counter() - t0) * 1000.0)
        idx["i"] += 1
        return r

    model._encode_input_image_latents_tensor = timed("vae", o_vae)
    model.video_expert.pre_dit = timed("pre_dit", o_pre)
    timed_pf = timed("prefill", o_pf)
    model.mot.prefill_video_cache_tensor = timed_pf
    model._denoise_action_with_video_cache = timed_step

    # infer_action caches compiled callables after the primary timing pass.
    # Point those aliases at the same wrappers for this diagnostic pass so the
    # active path is measured, then restore them exactly.
    compiled_aliases = {}
    for name, wrapped in (
        ("_prefill_video_cache_compiled", timed_pf),
        ("_denoise_action_with_video_cache_compiled", timed_step),
    ):
        if hasattr(model, name):
            compiled_aliases[name] = getattr(model, name)
            setattr(model, name, wrapped)
    try:
        for _ in range(STAGE_ITERS):
            idx["i"] = 0
            infer(model, *inputs, steps)
    finally:
        model._encode_input_image_latents_tensor = o_vae
        model.video_expert.pre_dit = o_pre
        model.mot.prefill_video_cache_tensor = o_pf
        model._denoise_action_with_video_cache = o_denoise
        for name, original in compiled_aliases.items():
            setattr(model, name, original)

    med = {k: statistics.median(v) for k, v in stage.items() if v}
    med["denoise_steps"] = [statistics.median(v) for _, v in sorted(steps_ms.items())]
    return med


def main():
    print(f"[device] {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[torch]  {torch.__version__}", flush=True)
    model = load_model()
    inputs = build_inputs(model)
    # The stage breakdown (diagnostic only, not scored) needs these hooks; trees
    # without them, and the naive compile arm, skip it.
    has_stage_hooks = (hasattr(model, "_denoise_action_with_video_cache")
                       and hasattr(model.mot, "prefill_video_cache_tensor"))
    seen = {}
    shape_probe_original = install_shape_probe(model, seen)
    probe_name = seen["_probe"]

    report = {"cases": {}, "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}
    try:
        for steps in CASES:
            compile_info = None
            if _NAIVE:
                ts, compile_info = naive_case(model, inputs, steps, probe_name,
                                              shape_probe_original)
            else:
                ts = time_total(model, inputs, steps)
            if seen.get("Sv") != EXPECT_SV or seen.get("ctx") != EXPECT_CTX:
                raise AssertionError(
                    f"harness is measuring the wrong shape: got Sv={seen.get('Sv')} "
                    f"ctx={seen.get('ctx')}, deployed is Sv={EXPECT_SV} ctx={EXPECT_CTX}. "
                    "A previous campaign lost a full round to exactly this class of mismatch."
                )
            med = statistics.median(ts)
            stages = (time_stages(model, inputs, steps)
                      if has_stage_hooks and not _NAIVE else {})
            case = f"episode_{steps}step"
            report["cases"][case] = {
                "num_inference_steps": steps,
                "median_ms": med,
                "min_ms": min(ts),
                "p90_ms": sorted(ts)[int(0.9 * (len(ts) - 1))],
                "stage_ms_diagnostic": stages,
            }
            if compile_info is not None:
                report["cases"][case]["compile"] = compile_info
            print(f"Perf: {med:.4f} ms ({case})", flush=True)
    finally:
        setattr(model.mot, probe_name, shape_probe_original)

    meds = [c["median_ms"] for c in report["cases"].values()]
    geo = statistics.geometric_mean(meds)
    report["geomean_ms"] = geo
    report["shape"] = {"Sv": seen["Sv"], "ctx": seen["ctx"], "img": [IMG_H, IMG_W]}

    print(f"\n=== END-TO-END infer_action (median of {ITERS} iters) ===", flush=True)
    for case, c in report["cases"].items():
        s = c["stage_ms_diagnostic"]
        loop = sum(s.get("denoise_steps", []))
        prologue = s.get("vae", 0.0) + s.get("pre_dit", 0.0) + s.get("prefill", 0.0)
        print(f"{case:<16} total {c['median_ms']:7.2f} ms   "
              f"[vae {s.get('vae', float('nan')):5.2f} + pre_dit {s.get('pre_dit', float('nan')):5.2f} "
              f"+ prefill {s.get('prefill', float('nan')):6.2f} = prologue {prologue:6.2f} | "
              f"denoise {loop:7.2f} over {c['num_inference_steps']} steps]", flush=True)
    print(f"GEAK_EPISODE_GEOMEAN_MS={geo:.4f}", flush=True)
    print(json.dumps(report), flush=True)
    print("PERF_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())