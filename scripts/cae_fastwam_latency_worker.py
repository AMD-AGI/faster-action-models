#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One CAE-FastWAM latency measurement, run INSIDE the eval container.

Launched by `scripts/cae_fastwam_latency.py`, once per (round, arm). The arm's source
tree is the first entry on PYTHONPATH; this script never selects a tree itself.

It loads the real CAE-FastWAM model (real checkpoint, real action VAE),
proves the cell is the one it claims to be, then times the whole inference
call:

  1. CONTRACT (untimed). One call with probes on the video prefill and on the
     per-step action denoise. Aborts unless Sv=98, ctx=129, Sa=8, the output is
     (32, 7), the denoise ran exactly the requested number of steps, and the
     denoise batch is 1. A mislabelled cell is refused, never reported.
  2. WARMUP. 25 calls, not 8: the VAE keeps autotuning its convs well past the
     8th call.
  3. TIMED. Median over ITERS calls, each fenced by a device synchronize.

Inputs are fixed and synthetic (seeded): a 224x448 two-camera image, a 128-token
text context and a proprio vector. This measures kernel and dispatch time, not
task success. The text context is precomputed, so the text encoder never runs
inside the timed region; `--text-encoder` only controls whether it is resident.

Prints `Perf: <median> ms` and one `RESULT <json>` line. Nothing is written.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import sys
import time

IMG_H, IMG_W = 224, 448          # agentview + wrist, 224x224 each, side by side
TEXT_CTX_LEN, TEXT_DIM = 128, 4096
ACTION_HORIZON = 32
EXPECT_SV, EXPECT_CTX, EXPECT_SA = 98, 129, 8   # ctx = 128 text + 1 proprio token
DENOISE = "_denoise_action_with_video_cache"
PREFILL = "prefill_video_cache_tensor"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--compile-infer", choices=("0", "1"), default="1",
                    help="compile_action_infer for infer_action")
    ap.add_argument("--text-encoder", choices=("0", "1"), default="0")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--suite-id", type=int, default=0)
    ap.add_argument("--expect-src", default="",
                    help="abort unless the loaded fastwam package lives under this path")
    return ap.parse_args()


def load_model(a, torch):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("max", lambda x: max(x), replace=True)
    OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)],
                                    replace=True)
    ae = os.environ["ACTION_VAE_CKPT"]
    overrides = [
        f"ckpt={os.environ['MODEL_CKPT']}",
        f"task={os.environ['TASK_CONFIG']}",
        f"model.load_text_encoder={'true' if a.text_encoder == '1' else 'false'}",
        # The action expert denoises the action-VAE latent: action_dim is the
        # VAE latent_dim (16), not the raw 7. Without this the checkpoint fails
        # to load on a [16,32] vs [7,32] head.
        "model.action_dit_config.action_dim=16",
        # Absolute path: the config's relative one only resolves from one cwd.
        f"++model.action_vae_ckpt={ae}",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=os.environ["MODEL_CONFIGS"]):
        cfg = compose(config_name="sim_libero", overrides=overrides)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    vae = getattr(model, "action_vae", None)
    if vae is None:
        raise AssertionError("action VAE did not load; this is not the Compact Action Expert")
    # Suite-conditioned decoder: decode raises unless a suite is selected. The
    # cost is the same for every suite (one embedding added to the decoder input).
    model.action_vae_suite_id = a.suite_id
    model.load_checkpoint(str(cfg.ckpt))
    return model.to("cuda").eval()


def build_inputs(model, torch):
    g = torch.Generator(device="cpu").manual_seed(1234)
    img = (torch.rand((1, 3, IMG_H, IMG_W), generator=g) * 2 - 1).to("cuda", torch.bfloat16)
    ctx = torch.randn((1, TEXT_CTX_LEN, TEXT_DIM), generator=g).to("cuda", torch.bfloat16)
    mask = torch.ones((1, TEXT_CTX_LEN), dtype=torch.bool, device="cuda")
    pd = getattr(model, "proprio_dim", None)
    proprio = (None if not pd else
               torch.randn((1, int(pd)), generator=g).to("cuda", torch.bfloat16))
    return img, ctx, mask, proprio


def install_probes(model, seen):
    """Record what prefill and the denoise loop actually receive."""
    orig_pf = getattr(model.mot, PREFILL)
    sig = inspect.signature(orig_pf)

    def prefill(*args, **kw):
        b = sig.bind(*args, **kw).arguments
        vt, vc = b.get("video_tokens"), b.get("video_context")
        if vt is not None:
            seen["Sv"] = int(vt.shape[1])
        if vc is not None:
            seen["ctx"] = int(vc.shape[1])
        return orig_pf(*args, **kw)

    setattr(model.mot, PREFILL, prefill)

    orig_dn = getattr(model, DENOISE)

    def denoise(*args, **kw):
        la = kw.get("latents_action", args[0] if args else None)
        if la is not None and hasattr(la, "shape"):
            seen.setdefault("denoise_B", set()).add(int(la.shape[0]))
        seen["calls"] = seen.get("calls", 0) + 1
        return orig_dn(*args, **kw)

    setattr(model, DENOISE, denoise)

    # The context is precomputed, so the text encoder must never run. Count it
    # rather than assume it, in case a tree re-encodes the prompt.
    if getattr(model, "encode_prompt", None) is not None:
        orig_ep = model.encode_prompt

        def encode_prompt(*args, **kw):
            seen["encode_prompt"] = seen.get("encode_prompt", 0) + 1
            return orig_ep(*args, **kw)

        model.encode_prompt = encode_prompt


def main() -> int:
    a = parse_args()
    import torch

    model = load_model(a, torch)
    import fastwam
    loaded_from = os.path.dirname(os.path.dirname(os.path.abspath(fastwam.__file__)))
    if a.expect_src and os.path.realpath(loaded_from) != os.path.realpath(a.expect_src):
        raise AssertionError(f"loaded fastwam from {loaded_from}, expected {a.expect_src}")
    img, ctx, mask, proprio = build_inputs(model, torch)

    def call():
        with torch.no_grad():
            return model.infer_action(
                prompt=None, input_image=img, action_horizon=ACTION_HORIZON,
                proprio=proprio, context=ctx, context_mask=mask,
                num_inference_steps=a.steps, seed=42, rand_device="cpu",
                compile_action_infer=a.compile_infer == "1")

    seen: dict = {}
    install_probes(model, seen)

    out = call()
    act = out["action"] if isinstance(out, dict) else out
    shape = tuple(int(x) for x in act.shape)
    expect = (ACTION_HORIZON, 7)
    sa = int(model.action_vae.cfg.num_tokens)
    contract = {"Sv": seen.get("Sv"), "ctx": seen.get("ctx"), "Sa": sa,
                "action_out": list(shape), "denoise_steps": seen.get("calls", 0),
                "denoise_B": sorted(seen.get("denoise_B", ()))}
    print(f"[contract] {contract}", flush=True)
    problems = []
    if contract["Sv"] != EXPECT_SV or contract["ctx"] != EXPECT_CTX:
        problems.append(f"prefill saw Sv={contract['Sv']} ctx={contract['ctx']}, "
                        f"deployed is Sv={EXPECT_SV} ctx={EXPECT_CTX}")
    if sa != EXPECT_SA:
        problems.append(f"Sa={sa}, expected {EXPECT_SA}")
    if shape != expect:
        problems.append(f"action {shape}, expected {expect}")
    if contract["denoise_steps"] != a.steps:
        problems.append(f"denoise ran {contract['denoise_steps']}x, requested {a.steps}")
    if contract["denoise_B"] != [1]:
        problems.append(f"denoise batch {contract['denoise_B']}, expected [1]")
    if seen.get("encode_prompt", 0):
        problems.append(f"text encoder ran {seen['encode_prompt']}x inside the call")
    if problems:
        print("CONTRACT FAILED: " + "; ".join(problems), file=sys.stderr, flush=True)
        return 3

    for _ in range(a.warmup):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)

    med = statistics.median(ts)
    result = {"median_ms": med, "min_ms": min(ts),
              "p90_ms": sorted(ts)[int(0.9 * (len(ts) - 1))],
              "steps": a.steps, "compile_infer": a.compile_infer == "1",
              "text_encoder_resident": getattr(model, "text_encoder", None) is not None,
              "warmup": a.warmup, "iters": a.iters, "contract": contract,
              "device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    print(f"Perf: {med:.4f} ms", flush=True)
    print("RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
