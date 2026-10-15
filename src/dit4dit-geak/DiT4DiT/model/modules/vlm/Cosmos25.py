# Copyright 2025 DiT4DiT team. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Teli Ma/ HKUST GZ] in [2025]. 
# Modified by Advanced Micro Devices, Inc. Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.

"""
Cosmos 2.5 feature extractor backend for DiT4DiT.

Goal:
  - Replace VLM backbone with a video diffusion transformer feature extractor.
  - Expose a lightweight interface compatible with DiT4DiT frameworks:
      - build_cosmos_inputs(images, instructions, ...) -> dict
      - forward(**inputs) -> object/dict containing `hidden_states` (list[Tensor])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
def _exists(v) -> bool:
    return v is not None


def _geak_env_on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False")


# GEAK: torch.compile the LIVE Cosmos transformer blocks (0..extract_layer).
# The backbone is 71% of the call and, on the kernel side, ~27 ms / 1700 launches
# of un-fused pointwise chains (three AdaLN units + three gated residual adds +
# GELU FF + the attn-processor reshape/RMS/RoPE pile, x18 blocks). There is no
# single fat kernel to rewrite -- inductor fusion is the only thing that reaches
# this surface.  Set GEAK_COSMOS_COMPILE=0 to fall back to the eager path.
_GEAK_COSMOS_COMPILE = _geak_env_on("GEAK_COSMOS_COMPILE", "1")


def _geak_nx_shadow() -> bool:
    """TorchInductor imports networkx; the ~/.local pin is networkx 2.2, which is
    broken on py3.10 (`from collections import Mapping`). A py3.10-safe networkx
    lives in nx_shadow/ -- prepend it so inductor can import.

    NOTE: prepending sys.path alone is NOT sufficient. The failed first import
    leaves half-initialised `networkx.*` entries in sys.modules and the shadow
    then dies with "cannot import name 'argmap'". Every networkx* key must be
    popped after the path insert."""
    try:
        import networkx  # noqa: F401
        return True
    except Exception:
        pass
    # Optional fallback: a directory holding a private networkx copy,
    # used only when the normal import above fails. Set GEAK_NX_SHADOW
    # to point at one; unset, this returns False and the caller degrades.
    shadow = os.environ.get("GEAK_NX_SHADOW", "")
    if not shadow:
        return False
    if not os.path.isdir(shadow):
        return False
    if shadow not in sys.path:
        sys.path.insert(0, shadow)
    for name in [n for n in sys.modules if n == "networkx" or n.startswith("networkx.")]:
        sys.modules.pop(name, None)
    try:
        import networkx  # noqa: F401
        return True
    except Exception:
        return False


def _geak_flag(name: str, default: str = "1") -> bool:
    import os as _o
    return str(_o.environ.get(name, default)).strip() not in ("0", "false", "False", "")


# Explicit stop-at-layer instead of the exception-based tail elision (an exception
# cannot cross a graph capture). Falls back to the hook+exception path when off.
_GEAK_STOPLAYER = _geak_flag("GEAK_DIT_STOPLAYER")
_GEAK_PROMPT_UID = _geak_flag("GEAK_DIT_TXTCACHE")
_GEAK_UID_COUNTER = [0]

# ---------------------------------------------------------------------------
# GEAK r2_d0: kill the per-call blocking host<->device memcpies.
#
# MEASURED (torch.profiler, 4-step arm, this tree): 5 x hipMemcpyWithStream per
# iteration costing 5.166 ms of SELF CPU.  A traced run (patching Tensor.to /
# Tensor.cpu / torch.tensor and dumping the python stack of every CPU<->CUDA
# transition) attributes them to:
#     Cosmos25.prepare_latents:630  self.latents_mean.to(device)   (1,16,1,1,1) CPU->GPU
#     Cosmos25.prepare_latents:631  self.latents_std.to(device)    (1,16,1,1,1) CPU->GPU
#     Cosmos25.forward:1008         scheduler.set_timesteps(...)   -> timesteps H2D
#     Cosmos25.forward:1085         scheduler.sigmas.to(device)    (S+1,)       CPU->GPU
#     Cosmos25._build_cond_videos_gpu:1612  image                  (1,224,224,3) u8 H2D
# plus the terminal D2H in DiT4DiT.predict_action (outside any capture region).
#
# Each of these is a copy out of PAGEABLE host memory, which ROCm implements as
# a BLOCKING hipMemcpyWithStream: the host stalls until the whole stream drains.
# Four of the five are compile-time constants of the call (VAE latent stats,
# scheduler sigma/timestep tables) that are re-uploaded every single iteration.
# Cache the device-resident copy keyed by (device, dtype) and the sync is gone.
#
# The five .item() sites named in the round-2 direction (Cosmos25:936/1178/1208/
# 1324/1388, DiT4DiT:381) were checked with the same tracer and are NOT executed
# on this deployment's path -- they all sit inside `training_mode != "legacy"`
# or `gt_future_videos is not None` branches, and DiT4DiT:381 is in
# `_predict_action_policy` which `policy_mode=False` never reaches.  The tracer
# records exactly ONE device->host sync per predict_action call (the terminal
# .cpu() at DiT4DiT.predict_action), so there is nothing to constant-fold there.
# They are left untouched rather than "fixed" blind.
_GEAK_DEVCONST = _geak_flag("GEAK_DIT_DEVCONST")
_GEAK_PINNED_H2D = _geak_flag("GEAK_DIT_PINH2D", "0")  # measured NEUTRAL-to-worse (11.702 vs 11.688 geomean): the extra numpy->pinned copy costs what the async DMA saves. Kept behind the switch, default OFF.
_GEAK_FUSE_LATSCALE = _geak_flag("GEAK_DIT_FUSESCALE")
_GEAK_VERIFY = _geak_flag("GEAK_DIT_HOSTVERIFY", "0")
# Collapse the 0/1-mask arithmetic that surrounds the video DiT input into
# single kernels.  Every one of these rewrites is BITWISE exact because
# `cond_indicator` is built as `zeros(); [:, :, :n] = 1.0`, i.e. its values are
# exactly 0.0 and 1.0.  Measured inside the captured region these tiny 5-D
# broadcasts are NOT free: TensorIterator hands a (B,1,T,1,1) x (B,C,T,H,W)
# broadcast to a single-CTA `vectorized_templated_elementwise_kernel`.
_GEAK_MASKFUSE = _geak_flag("GEAK_DIT_MASKFUSE")

# ---------------------------------------------------------------------------
# GEAK r4_d2 (region B, pre-DiT).
#
# (1) GEAK_DIT_VAESAMPLE.  `DiagonalGaussianDistribution.sample()` in diffusers
#     evaluates `mean + std * noise`.  On this checkpoint `mean` is the bf16
#     chunk of the encoder output while `std = exp(0.5*logvar)` comes back FP32,
#     so BOTH the mul and the add fall onto TensorIterator's mixed-dtype
#     `vectorized_templated_elementwise_kernel` (dynamic cast per element).
#     GRAPH-TIMED on this box, (1,16,1,28,28):
#         mean + std*noise                 63.4 us   (31.2 mul + 32.2 add)
#         torch.addcmul(mean, std, noise)   4.3 us
#     One fused ternary op, same fp32 arithmetic, 59 us off BOTH arms.  This is
#     the pair r3_d0 was reported to have removed and had not.
_GEAK_VAESAMPLE = _geak_flag("GEAK_DIT_VAESAMPLE")
# ROUND 6 -- GEAK_DIT_SAMPDT.  The addcmul above is still MIXED DTYPE: `mean`
# and `noise` are bf16 (the encoder's autocast dtype) while `std = exp(0.5*logvar)`
# comes back fp32 because `torch.exp` is on autocast's fp32 promote list.  Type
# promotion therefore makes the whole ternary fp32, and TensorIterator inserts
# TWO explicit `bfloat16tofloat32` cast dispatches in front of it (visible in the
# kernel trace at 7 CTAs each, pure floor) plus a cast of the fp32 result back
# down.  Rounding `std` ONCE to the activation dtype makes the op homogeneous:
# 4 nodes -> 2.  The only numerical difference is that the std factor is rounded
# to bf16 before the FMA instead of after it; the result is stored as bf16 either
# way.  Kill switch: GEAK_DIT_SAMPDT=0 keeps the mixed-dtype form.
_GEAK_SAMPDT = _geak_flag("GEAK_DIT_SAMPDT")
# (2) GEAK_DIT_PREB.  Region B is 98 dispatches averaging ~5 us at <=8 CTAs with
#     essentially no arithmetic in them.  Every 0/1 mask, the padding mask, the
#     conditional-timestep field and the per-step timestep vector are pure
#     functions of (shape, dtype, device, schedule) -- i.e. call-INVARIANT for a
#     fixed episode -- so they are built once and memoised on the module.  The
#     cache OWNS a strong reference, which is exactly what a HIP-graph capture
#     needs from a tensor allocated outside its private pool.
_GEAK_PREB = _geak_flag("GEAK_DIT_PREB")
# (3) GEAK_DIT_DTYPEMEMO -- the last host bubble in the model, and by far the
#     cheapest thing on the board.  `ModelMixin.dtype` is a PROPERTY that walks
#     every parameter and buffer of the module.  MEASURED on this checkpoint,
#     pure python, no device work:
#         extractor.vae.dtype            0.191 ms   per access
#         extractor.transformer.dtype    0.828 ms   per access
#         extractor.device               0.003 ms
#     `_build_cond_videos_gpu` reads `vae.dtype` once per call and it sits in the
#     UN-CAPTURED image-preprocess prologue, so the GPU idles through all 191 us
#     of it -- this is the ~207 us head-of-call gap in the kernel trace.  The
#     module dtype is fixed for the life of the process (the harness casts the
#     model once, at load, before any predict_action), so it is memoised on the
#     module's __dict__ (bypassing nn.Module.__setattr__).
_GEAK_DTYPEMEMO = _geak_flag("GEAK_DIT_DTYPEMEMO")
# (4) GEAK_DIT_PREGRAPH (r5_d2) -- the head-of-call bubble, re-measured.
#     MEASURED on this box (probe, 4-step, medians over 60 calls):
#         hipGraphLaunch, backbone graph (~950 nodes): 0.022 ms of HOST CPU,
#             and only 0.004 ms of it is EXPOSED (intercept of wall-vs-K for K
#             back-to-back replays; slope = 4.650 ms of GPU).
#         hipGraphLaunch, action-head graph:           0.006 ms host / 0.009 exposed.
#     So the ~204 us head-of-call gap is NOT graph launch -- graph launch is
#     ~13 us of the whole call.  What IS exposed is the UN-CAPTURED image
#     preprocess: `build_cosmos_inputs` costs 0.15 ms of host CPU with the GPU
#     idle, because it issues ~8 eager torch ops (permute / to(f32) / div_ /
#     to(bf16) / mul_ / sub_ / unsqueeze / contiguous) plus an H2D, and the
#     first ops after the terminal .cpu() block run on a cold host.
#     FIX: make the GRAPH INPUT the raw uint8 (B,H,W,3) frame and move the whole
#     preprocess chain INSIDE the capture.  The op sequence is unchanged, so the
#     result is bitwise identical; the host per call drops to
#     numpy-stack + from_numpy + one H2D copy_ + replay.
_GEAK_PREGRAPH = _geak_flag("GEAK_DIT_PREGRAPH")


def _geak_mod_device(mod, default=None):
    """`Module.device` is a diffusers property (`next(self.parameters()).device`).
    Cheap (0.003 ms) but it sits in the exposed prologue; the device is fixed for
    the life of the process here, so memoise it exactly like the dtype."""
    if mod is None:
        return default
    if not _GEAK_DTYPEMEMO:
        return getattr(mod, "device", default)
    d = mod.__dict__.get("_geak_device_memo", None)
    if d is None:
        d = getattr(mod, "device", default)
        try:
            mod.__dict__["_geak_device_memo"] = d
        except Exception:
            pass
    return d


# ==================================================================
# GEAK r2_d0 (deep_explore) -- PREPROCESS NODE COLLAPSE.
#
# MEASURED CONTEXT.  `_geak_pre_from_u8` is inside the backbone CUDA graph, so
# every one of its ops is a graph NODE, and this box's in-graph node floor was
# re-derived this session at 1.93 us/node (112-node head chain with every body
# replaced by a same-grid nop: 216.5 us).  The chain is SIX nodes for 150 KB of
# pixels:
#     permute+contiguous (u8)  ->  .to(float32)  ->  div_(255)
#     ->  .to(bf16)  ->  mul_(2)  ->  sub_(1)
# i.e. ~11.6 us of pure launch floor on BOTH scored arms for a transform that is
# one pass over 50176 pixels.
#
# BITWISE IDENTITY (this is why it is safe).  PyTorch runs bf16 elementwise ops
# with an fp32 `opmath` accumulator and rounds once on store, so the deployed
# value chain is exactly
#     a = rn_bf16(u8 / 255.0)          # fp32 divide, then one bf16 rounding
#     b = rn_bf16(fp32(a) * 2.0)       # exact: scaling by 2 only moves the exponent
#     c = rn_bf16(fp32(b) - 1.0)
# and the kernel below evaluates that same expression in that same order with
# the same roundings.  Verified elementwise-equal (max|delta| = 0, exact bit
# equality on all 150528 elements) under GEAK_PRE_VERIFY=1.
#
# Kill switch: GEAK_PRE_FUSE=0 restores the six-op chain.
# ==================================================================
_GEAK_PREFUSE = _geak_env_on("GEAK_PRE_FUSE", "1")
_GEAK_PREVERIFY = _geak_env_on("GEAK_PRE_VERIFY", "0")
_GEAK_PRE_TRITON = True
try:
    import triton as _geak_triton
    import triton.language as _geak_tl
except Exception:  # pragma: no cover
    _GEAK_PRE_TRITON = False

if _GEAK_PRE_TRITON:

    @_geak_triton.jit
    def _geak_pre_u8_kernel(U8, O, HW, BLOCK: _geak_tl.constexpr,
                            NORM: _geak_tl.constexpr):
        """(B,H,W,3) uint8 -> (B,3,1,H,W) bf16 in [-1,1], one launch."""
        pid = _geak_tl.program_id(0)
        b = _geak_tl.program_id(1)
        off = pid * BLOCK + _geak_tl.arange(0, BLOCK)
        m = off < HW
        base = b * HW * 3
        for c in _geak_tl.static_range(3):
            v = _geak_tl.load(U8 + base + off * 3 + c, mask=m, other=0)
            v = v.to(_geak_tl.float32) / 255.0
            a = v.to(_geak_tl.bfloat16)
            if NORM:
                r = (a.to(_geak_tl.float32) * 2.0 - 1.0).to(_geak_tl.bfloat16)
            else:
                r = a
            _geak_tl.store(O + (b * 3 + c) * HW + off, r, mask=m)


def _geak_pre_fused(t_u8, do_normalize, out_dtype):
    """One-launch replacement for the six-node u8 preprocess chain, or None."""
    if not (_GEAK_PREFUSE and _GEAK_PRE_TRITON):
        return None
    try:
        if (t_u8.dtype != torch.uint8 or t_u8.ndim != 4 or t_u8.shape[-1] != 3
                or not t_u8.is_cuda or not t_u8.is_contiguous()
                or out_dtype != torch.bfloat16):
            return None
        B, H, W, _ = (int(v) for v in t_u8.shape)
        hw = H * W
        out = torch.empty((B, 3, 1, H, W), device=t_u8.device,
                          dtype=torch.bfloat16)
        blk = 1024
        _geak_pre_u8_kernel[((hw + blk - 1) // blk, B)](
            t_u8, out, hw, BLOCK=blk, NORM=bool(do_normalize),
            num_warps=4, num_stages=1)
        return out
    except Exception:
        return None


def _geak_mod_dtype(mod, default=None):
    if mod is None:
        return default
    if not _GEAK_DTYPEMEMO:
        return getattr(mod, "dtype", default)
    d = mod.__dict__.get("_geak_dtype_memo", None)
    if d is None:
        d = getattr(mod, "dtype", default)
        try:
            mod.__dict__["_geak_dtype_memo"] = d
        except Exception:
            pass
    return d


# Hard bound on the region-B memo.  Every key is a shape/dtype tuple or the id of
# another entry in the same (never-evicted) dict, so on the deployed path the cache
# saturates at a handful of entries.  A caller that rebuilds cond_indicator per call
# (the non-legacy action/state-latent injection branch) would otherwise grow it
# without limit.  The cap STOPS INSERTING rather than evicting -- evicting a tensor
# a captured HIP graph still reads is a hard memory fault, so nothing here is ever
# dropped once handed out.
_GEAK_PREB_MAX = 64


def _geak_preb_cache(owner):
    c = owner.__dict__.get("_geak_preb_cache_d")
    if c is None:
        c = {}
        owner.__dict__["_geak_preb_cache_d"] = c
    return c


def _geak_preb_put(cache, key, value):
    if len(cache) < _GEAK_PREB_MAX:
        cache[key] = value
    return value


def _geak_dist_sample(dist, generator):
    """`dist.sample()` with the mul+add collapsed into one addcmul.

    Bit-for-bit this is an FMA where the stock path rounds twice; the operand
    dtypes and the fp32 accumulation are identical.  Falls back to the stock
    call for anything that is not a plain DiagonalGaussianDistribution."""
    if not _GEAK_VAESAMPLE:
        return dist.sample(generator=generator)
    try:
        mean = dist.mean
        std = dist.std
        params = dist.parameters
    except Exception:
        return dist.sample(generator=generator)
    if not (torch.is_tensor(mean) and torch.is_tensor(std) and torch.is_tensor(params)):
        return dist.sample(generator=generator)
    if tuple(mean.shape) != tuple(std.shape):
        return dist.sample(generator=generator)
    try:
        from diffusers.utils.torch_utils import randn_tensor as _rt
        noise = _rt(mean.shape, generator=generator, device=params.device, dtype=params.dtype)
        if _GEAK_SAMPDT and std.dtype != mean.dtype and noise.dtype == mean.dtype:
            std = std.to(mean.dtype)
        out = torch.addcmul(mean, std, noise)
    except Exception:
        return dist.sample(generator=generator)
    if _GEAK_VERIFY:
        try:
            print("[GEAK_VERIFY] vae_sample dtypes mean=%s std=%s out=%s" %
                  (mean.dtype, std.dtype, out.dtype), flush=True)
        except Exception:
            pass
    return out

# scheduler fields written by *MultistepScheduler.set_timesteps -- memoising the
# call means restoring every one of them, including the mutable ones that
# .step() would advance (copied, never aliased).
_GEAK_SCHED_FIELDS = (
    "sigmas", "timesteps", "num_inference_steps", "model_outputs",
    "lower_order_nums", "last_sample", "_step_index", "_begin_index",
)


def _geak_dev_const(owner, key, src, device, dtype):
    """Device-resident, dtype-resolved copy of a host constant, memoised on `owner`.

    `src` is a tensor that never changes for the life of the module (VAE latent
    mean/std, scheduler sigma table).  Returns the SAME device tensor on every
    later call, so the pageable H2D memcpy -- and the full-stream drain it forces
    -- happens once instead of once per predict_action."""
    if not _GEAK_DEVCONST or src is None:
        return src.to(device=device, dtype=dtype) if src is not None else None
    cache = owner.__dict__.get("_geak_const_cache")
    if cache is None:
        cache = {}
        owner.__dict__["_geak_const_cache"] = cache
    k = (key, str(device), dtype, tuple(src.shape), src.data_ptr())
    hit = cache.get(k)
    if hit is None:
        hit = src.to(device=device, dtype=dtype)
        cache[k] = hit
    return hit


def _geak_set_timesteps(owner, scheduler, num_steps, device):
    """Memoised scheduler.set_timesteps.

    set_timesteps is deterministic in (num_steps, device): it rebuilds the sigma
    table on the host with numpy and then does a `torch.from_numpy(...).to(device)`
    -- one blocking pageable H2D per call -- and resets the solver state.  Cache
    every field it writes and restore them (mutable containers copied so a later
    .step() can never write through into the cache)."""
    if not _GEAK_DEVCONST:
        scheduler.set_timesteps(num_steps, device=device)
        return
    cache = owner.__dict__.get("_geak_sched_cache")
    if cache is None:
        cache = {}
        owner.__dict__["_geak_sched_cache"] = cache
    k = (id(scheduler), int(num_steps), str(device))
    ent = cache.get(k)
    if ent is None:
        scheduler.set_timesteps(num_steps, device=device)
        try:
            ent = {f: getattr(scheduler, f) for f in _GEAK_SCHED_FIELDS if hasattr(scheduler, f)}
            if getattr(scheduler, "solver_p", None) is not None:
                ent = None  # nested scheduler state -- do not memoise
        except Exception:
            ent = None
        if ent is not None:
            cache[k] = ent
        return
    for f, v in ent.items():
        setattr(scheduler, f, list(v) if isinstance(v, list) else v)
# ---------------------------------------------------------------------------


class _GeakTailElided(Exception):
    """Private sentinel: unwinds the Cosmos transformer immediately after the extraction block.

    See `Cosmos25FeatureExtractor.forward` for the read-set argument that makes the remaining
    transformer blocks + norm_out + proj_out + unpatchify provably dead work on the inference
    (`training_mode="legacy"`, `need_generate=False`, `stop_i==0`) path.
    """

    __slots__ = ()


@dataclass
class BackboneOutput:
    """Minimal output container."""

    hidden_states: List[torch.Tensor]
    future_video_loss: Optional[torch.Tensor] = None
    pred_future_video: Optional[torch.Tensor] = None  # (B,T,3,H,W) in [0,1] if returned
    action_loss: Optional[torch.Tensor] = None         # flow matching loss on action latent frames
    action_decoded_loss: Optional[torch.Tensor] = None  # MSE on decoded actions from predicted x0


class _DefaultDummySafetyChecker:
    """Used only to construct Cosmos pipeline without importing/downloading guardrail."""

    def __init__(self, *args, **kwargs):
        pass

    def to(self, device):
        return self

    def check_text_safety(self, text):
        return True

    def check_video_safety(self, video):
        return video


def _os_env_cache_max():
    """Max distinct prompts held in the text-embedding cache (env-tunable)."""
    import os as _o
    return _o.environ.get("DIT_PROMPT_CACHE_MAX", "32")


class Cosmos25FeatureExtractor(nn.Module):
    """
    Feature extractor based on `diffusers.Cosmos2_5_PredictBasePipeline`.
      - returns a hidden feature tensor (hooked from a transformer block)
      - optionally returns repeated per-layer hidden list for Layerwise DiT conditioning
    """

    def __init__(
        self,
        model_id_or_path: str,
        *,
        revision: str = "diffusers/base/post-trained",
        torch_dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = True,
        extract_layer: int = 0,
        max_sequence_length: int = 512,
        trainable: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        safety_checker: Any = None,
        config: Optional[Any] = None,
    ):
        super().__init__()
        self.config = config

        from diffusers import Cosmos2_5_PredictBasePipeline

        self.model_id_or_path = model_id_or_path
        self.revision = revision
        self.torch_dtype = torch_dtype
        self.local_files_only = local_files_only
        self.extract_layer = extract_layer
        self.max_sequence_length = max_sequence_length
        self.trainable = trainable

        if safety_checker is None:
            safety_checker = _DefaultDummySafetyChecker()

        pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
            model_id_or_path,
            revision=revision,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            safety_checker=safety_checker,
        )

        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.transformer = pipe.transformer
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler
        self.video_processor = pipe.video_processor

        self.latents_mean = getattr(pipe, "latents_mean", None)
        # DIFFUSERS 0.40 STORES THE RECIPROCAL. In <=0.37 `pipe.latents_std` is
        # the std itself; 0.40 changed it to `1.0 / latents_std`. This code
        # divides by the attribute, so on 0.40 the conditioning latents get
        # scaled by std^2 (~8x) instead of normalized -- finite actions, healthy
        # plumbing, and a 0% success rate. Normalize against the VAE config,
        # which is version-stable, and only fall back to the attribute.
        _std = getattr(pipe, "latents_std", None)
        _cfg_std = getattr(getattr(pipe, "vae", None), "config", None)
        _cfg_std = getattr(_cfg_std, "latents_std", None) if _cfg_std is not None else None
        if _std is not None and _cfg_std is not None:
            import torch as _t
            _ref = _t.tensor(_cfg_std).view(1, -1, 1, 1, 1).float().to(_std.device)
            # reciprocal iff std*ref ~= 1 elementwise
            if bool((_std.float() * _ref - 1.0).abs().max() < 1e-3):
                _std = 1.0 / _std
        self.latents_std = _std
        self.vae_scale_factor_temporal = getattr(pipe, "vae_scale_factor_temporal", None)
        self.vae_scale_factor_spatial = getattr(pipe, "vae_scale_factor_spatial", None)

        self._hook_handle = None
        self._cached_hidden: List[torch.Tensor] = []
        # Used to freeze hidden capture at step 0 (so later denoising steps don't overwrite it)
        self._capture_hidden_enabled: bool = True
        # Opt-IN flag (default OFF): when the caller has positively verified that the transformer's
        # RETURN VALUE is dead and only the extraction-layer hidden state is read, the capture hook
        # raises `_GeakTailElided` to unwind out of the remaining transformer blocks. Any caller that
        # does not set this gets the original, bit-for-bit full-transformer path.
        self._geak_elide_tail: bool = False
        self._register_hook()
        self._geak_compile_blocks()

        # Training-only. Upstream DiT4DiT never enables this; under
        # torch.inference_mode() the checkpointed blocks try to save inference
        # tensors for backward and predict_action raises. The measured serving
        # runs disabled it on every module after load, which this reproduces.
        if self.trainable:
            self.transformer.enable_gradient_checkpointing()

        if _exists(device):
            self.to(device)

        # Free memory
        del pipe



    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def hidden_size(self) -> Optional[int]:
        # Best-effort inference from common config names; frameworks may override via config if None.
        cfg = getattr(self.transformer, "config", None)
        if cfg is None:
            return None
        for key in ("hidden_size", "inner_dim", "model_dim", "cross_attention_dim"):
            v = getattr(cfg, key, None)
            if isinstance(v, int) and v > 0:
                return v
        return None

    def __del__(self):
        if _exists(self._hook_handle):
            try:
                self._hook_handle.remove()
            except Exception:
                pass

    def _geak_compile_blocks(self):
        """torch.compile the transformer blocks that actually execute (0..extract_layer).

        WHY: the Cosmos backbone is ~71% of `predict_action` and its device profile is
        ~27 ms of elementwise work spread over ~1700 launches and 50 distinct kernels with
        no single site above 3.3 ms -- three `CosmosAdaLayerNormZero` units (SiLU ->
        Linear(1024->256) -> Linear(256->3*2048) -> add -> chunk -> 3x unsqueeze ->
        LayerNorm -> h*(1+scale)+shift), three gated residual adds, a GELU FeedForward, and
        the attn-processor unflatten/transpose/RMS-q/k/RoPE/repeat_interleave/flatten pile,
        times 18 live blocks. That surface is only reachable by FUSION, and inductor is the
        automatic path for it.

        THREE THINGS THAT MAKE THIS WORK (each cost a measurement cycle to find):

        1. `extract_layer` is a CHECKPOINT property (config.yaml overrides the code default
           of 19). Read the live value; measured census here is extract_layer=17 with 18
           blocks executing (0..17) at both 1- and 4-step.

        2. All 28 blocks are structurally identical, but each carries its OWN
           `CosmosAttnProcessor2_0` INSTANCE, and dynamo guards on `___check_obj_id(
           self._modules['attn1'].processor, ...)`. Without intervention every block is a
           distinct graph, the 8-entry recompile limit trips at block 8 and the remaining
           blocks silently fall back to eager (measured: 1.16x instead of 1.45x). The
           processors are STATELESS (`CosmosAttnProcessor2_0.__init__` only checks that
           `F.scaled_dot_product_attention` exists; `__call__` reads nothing off `self`), so
           pointing every block at ONE shared processor instance is semantics-preserving and
           collapses all 18 blocks onto a single compiled graph -- 68 inductor kernels
           compiled once in ~0.7 s.

        3. THE TAIL-ELISION EXCEPTION IS SAFE HERE. `_GeakTailElided` is raised from a
           forward HOOK, and `nn.Module._call_impl` runs hooks OUTSIDE the value returned by
           `self.forward` -- so the exception never crosses a compiled frame and needs no
           graph break. Verified: the census still shows blocks 0..17 executing and the
           unwind still fires.

        Wrapping is instance-level `block.forward = torch.compile(block.forward)`, which
        DOES take effect (`_call_impl` reads `self.forward`) -- verified by census, not
        assumed. Compilation is lazy, so this costs nothing at construction time and the
        one-off ~0.7 s trace is absorbed by the harness's 20 warmups. Any failure falls back
        to the untouched eager path.
        """
        if not _GEAK_COSMOS_COMPILE:
            return
        blocks = getattr(self.transformer, "transformer_blocks", None)
        if not isinstance(blocks, (list, nn.ModuleList)) or len(blocks) == 0:
            return
        nx_ok = _geak_nx_shadow()
        n_live = 0
        if nx_ok:
            try:
                import torch._dynamo as _dynamo

                # 18 live blocks share one graph, but leave headroom for the recompiles
                # dynamo does on the first few (dtype/stride) specialisations.
                _dynamo.config.cache_size_limit = max(
                    32, int(getattr(_dynamo.config, "cache_size_limit", 8))
                )
                _dynamo.config.accumulated_cache_size_limit = max(
                    256, int(getattr(_dynamo.config, "accumulated_cache_size_limit", 64))
                )

                last = min(int(self.extract_layer), len(blocks) - 1)
                shared_p1 = getattr(blocks[0], "attn1", None)
                shared_p2 = getattr(blocks[0], "attn2", None)
                shared_p1 = getattr(shared_p1, "processor", None)
                shared_p2 = getattr(shared_p2, "processor", None)
                for i in range(last + 1):
                    b = blocks[i]
                    # Collapse the per-block processor identities (stateless objects).
                    if shared_p1 is not None and hasattr(b, "attn1"):
                        b.attn1.processor = shared_p1
                    if shared_p2 is not None and hasattr(b, "attn2"):
                        b.attn2.processor = shared_p2
                    b.forward = torch.compile(b.forward, dynamic=False)
                    n_live += 1
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[geak] cosmos block compile FAILED ({exc!r}); staying eager", flush=True)
                n_live = 0
        print(
            f"[geak] cosmos-block compile: nx={nx_ok} compiled_blocks={n_live} "
            f"extract_layer={self.extract_layer} nblocks={len(blocks)}",
            flush=True,
        )

    def _register_hook(self):
        if not hasattr(self.transformer, "transformer_blocks"):
            return
        blocks = getattr(self.transformer, "transformer_blocks")
        if not isinstance(blocks, (list, nn.ModuleList)) or len(blocks) == 0:
            return
        if self.extract_layer < 0 or self.extract_layer >= len(blocks):
            raise ValueError(f"extract_layer={self.extract_layer} out of bounds for {len(blocks)} blocks")

        target_layer = blocks[self.extract_layer]

        def hook_fn(module, inp, out):
            if not getattr(self, "_capture_hidden_enabled", True):
                return
            captured = False
            if torch.is_tensor(out):
                self._cached_hidden.append(out.detach())
                captured = True
            elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                self._cached_hidden.append(out[0].detach())
                captured = True
            # Dead-tail elision (opt-in, and only once we actually hold the hidden state).
            if captured and getattr(self, "_geak_elide_tail", False):
                raise _GeakTailElided()

        self._hook_handle = target_layer.register_forward_hook(hook_fn)

    def _geak_disable_hook(self):
        """Remove the capture hook: on the stop_at_layer path the hidden state comes
        back as the transformer's RETURN VALUE, so the hook (and the framework hook
        walk it forces on every block call) is pure overhead."""
        h = self._hook_handle
        if h is not None:
            try:
                h.remove()
            except Exception:
                pass
            self._hook_handle = None

    def _geak_enable_hook(self):
        if self._hook_handle is None:
            self._register_hook()

    def _get_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
        max_sequence_length: int,
    ) -> torch.Tensor:
        prompt_list = [prompt] if isinstance(prompt, str) else prompt

        input_ids_batch = []
        for sample_idx in range(len(prompt_list)):
            conversations = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant who will provide prompts to an image generator.",
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt_list[sample_idx]}]},
            ]

            input_ids = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=False,
                max_length=max_sequence_length,
                truncation=True,
                padding="max_length",
            )
            # transformers >=5.x returns a BatchEncoding from
            # apply_chat_template(tokenize=True); 4.x returned a flat list of
            # ids. Iterating the BatchEncoding yields its KEYS (strings), so
            # torch.LongTensor() raised "'str' object cannot be interpreted as
            # an integer". Unwrap when needed -- works on both versions.
            if hasattr(input_ids, "input_ids"):
                input_ids = input_ids["input_ids"]
            if input_ids and isinstance(input_ids[0], (list, tuple)):
                input_ids = input_ids[0]      # batched-of-one
            input_ids_batch.append(torch.LongTensor(input_ids))

        input_ids_batch = torch.stack(input_ids_batch, dim=0).to(device)
        outputs = self.text_encoder(input_ids_batch, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        normalized_hidden_states = []
        for layer_idx in range(1, len(hidden_states)):
            hs = hidden_states[layer_idx]
            hs = (hs - hs.mean(dim=-1, keepdim=True)) / (hs.std(dim=-1, keepdim=True) + 1e-8)
            normalized_hidden_states.append(hs)

        prompt_embeds = torch.cat(normalized_hidden_states, dim=-1).to(dtype=dtype, device=device)
        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        *,
        guidance_scale: float = 1.0,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = device or self.device
        dtype = dtype or _geak_mod_dtype(self.text_encoder)
        max_sequence_length = max_sequence_length or self.max_sequence_length

        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt_list) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            # Cache the text-encoder output. `encode_prompt` is called from
            # forward(), i.e. once per policy call, but the LIBERO instruction
            # is constant for a whole episode -- so the 8.3B Qwen2.5-VL was
            # re-encoding the same string hundreds of times per rollout
            # (measured 91.5 ms of a 570 ms call, 16%).
            #
            # Same inputs => same embedding, so this is bit-exact, not an
            # approximation. Key on everything that changes the result. The
            # cache is capped because a long-running server sees many distinct
            # instructions across episodes and this tensor is not small.
            _key = (tuple(prompt_list), int(max_sequence_length), str(device), str(dtype))
            _cache = getattr(self, "_prompt_embed_cache", None)
            if _cache is None:
                _cache = self._prompt_embed_cache = {}
            prompt_embeds = _cache.get(_key)
            if prompt_embeds is None:
                prompt_embeds = self._get_prompt_embeds(
                    prompt_list, device=device, dtype=dtype,
                    max_sequence_length=max_sequence_length,
                )
                if len(_cache) >= int(_os_env_cache_max()):
                    _cache.pop(next(iter(_cache)))
                _cache[_key] = prompt_embeds
            _, seq_len, _ = prompt_embeds.shape
            if num_videos_per_prompt != 1 or prompt_embeds.shape[0] != batch_size:
                prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
            # Mint a content-derived id for this prompt embedding so downstream
            # consumers (crossattn_proj, per-block cross-attn K/V) can memoize a
            # pure function of the prompt.  A different prompt / max_len / device /
            # dtype is a different `_key`, hence a different uid, hence a guaranteed
            # cache MISS -- never a stale hit.
            if _GEAK_PROMPT_UID:
                _umap = getattr(self, "_geak_prompt_uidmap", None)
                if _umap is None:
                    _umap = self._geak_prompt_uidmap = {}
                _uid = _umap.get(_key)
                if _uid is None:
                    _GEAK_UID_COUNTER[0] += 1
                    _uid = _umap[_key] = _GEAK_UID_COUNTER[0]
                    if len(_umap) > 4096:
                        _umap.clear()
                        _umap[_key] = _uid
                try:
                    prompt_embeds._geak_prompt_uid = _uid
                except Exception:
                    pass

        # Feature extraction path uses guidance_scale=1.0 -> no negative embeds needed.
        _ = guidance_scale
        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        *,
        video: Optional[Union[torch.Tensor, List[Any]]],
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames_in: int,
        num_frames_out: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from diffusers.utils.torch_utils import randn_tensor

        B = batch_size
        C = num_channels_latents
        if not _exists(self.vae_scale_factor_temporal) or not _exists(self.vae_scale_factor_spatial):
            raise ValueError("missing VAE scale factors")

        T = (num_frames_out - 1) // int(self.vae_scale_factor_temporal) + 1
        H = height // int(self.vae_scale_factor_spatial)
        W = width // int(self.vae_scale_factor_spatial)
        shape = (B, C, T, H, W)

        if num_frames_in == 0:
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            cond_mask = torch.zeros((B, 1, T, H, W), dtype=latents.dtype, device=latents.device)
            cond_indicator = torch.zeros((B, 1, T, 1, 1), dtype=latents.dtype, device=latents.device)
            cond_latents = torch.zeros_like(latents)
            return latents, cond_latents, cond_mask, cond_indicator

        if video is None:
            raise ValueError("`video` must be provided when `num_frames_in` > 0")

        needs_preprocessing = not (
            isinstance(video, torch.Tensor) and video.ndim == 5 and (video.shape[1] == 3 or video.shape[2] == 3)
        )
        if needs_preprocessing:
            raise ValueError("`video` must be a 5D torch.Tensor in (B,3,T,H,W) or (B,T,3,H,W)")

        # Normalize to VAE dtype before encode
        video_t = video.to(device=device, dtype=_geak_mod_dtype(self.vae))
        if video_t.shape[1] == 3:
            video_bcthw = video_t
        else:
            # preprocess_video expects (B,T,C,H,W) and returns (B,C,T,H,W)
            video_bcthw = self.video_processor.preprocess_video(video_t, height=height, width=width)

        # Pad video in pixel space to num_frames_out BEFORE VAE encode
        # This matches the official Cosmos pipeline behavior
        temporal_factor = int(self.vae_scale_factor_temporal) if _exists(self.vae_scale_factor_temporal) else 1
        temporal_factor = max(1, temporal_factor)
        T_in_pix = int(video_bcthw.shape[2])

        # --- Elide the all-zero pixel pad whose latents are provably never read -------------------
        # The zero pad below exists only to satisfy the video pipeline's num_frames_out contract.
        # `AutoencoderKLWan._encode` chunks the pixel timeline as [0:1], [1:1+tf], [1+tf:1+2tf], ...
        # so latent frame j is produced from pixel frames [1+tf*(j-1), 1+tf*j).  To materialise the
        # first `n_lat_read` latent frames we therefore only need the first `1 + tf*(n_lat_read-1)`
        # pixel frames.  Any pixel frame dropped beyond that is (a) part of the ALL-ZERO pad this
        # function is about to write (index >= T_in_pix -- known from shapes only, no host sync),
        # and (b) only feeds latent frames at index >= n_lat_read.
        #
        # Whether those trailing latent frames are read is CALLER-DEPENDENT, so this is opt-IN:
        #   * `forward()`'s video flow-matching / world-model branches DO read
        #     `cond_latents[:, :, cond_count:]` as their future-video target.
        #   * `forward()`'s LEGACY branch (which is what `DiT4DiT.predict_action` drives in this
        #     deployment) never does: `num_cond_latent_frames = min(T_lat, (num_frames_in-1)//tf+1)`
        #     is 1 here, `cond_mask`/`cond_indicator` are 0 for every later frame, and every use of
        #     `cond_latents` in that branch is `cond_mask_t * cond_latents` or
        #     `(latents - cond_latents) * cond_mask_t`, i.e. multiplied by exactly zero.
        # The caller therefore sets `_geak_elide_pad` for the frames it will not read; the shorter
        # cond_latents is re-expanded to the full (B,C,T_out_lat,H,W) contract by the
        # `cond_latents.shape[2] != T_out_lat` fixup below (which zero-fills the dropped frames --
        # numerically identical to what the all-zero pixel pad would have produced only up to the
        # VAE's bias terms, and those frames are masked to zero anyway).
        _n_read = getattr(self, "_geak_elide_pad", None)
        _elide_ok = (
            _n_read is not None
            and torch.is_inference_mode_enabled()
            and not self.training
        )
        _n_pix_keep = None
        if _elide_ok:
            T_out_lat_pre = (int(num_frames_out) - 1) // temporal_factor + 1
            n_lat_read = max(1, min(int(T_out_lat_pre), int(_n_read)))
            n_pix_needed = 1 + temporal_factor * (n_lat_read - 1)
            # never drop a real (non-pad) pixel frame
            _n_pix_keep = max(T_in_pix, n_pix_needed)

        if _elide_ok and _n_pix_keep is not None and _n_pix_keep < num_frames_out:
            # Everything from _n_pix_keep..num_frames_out-1 would have been zero pad feeding
            # latent frames the caller has declared it will not read -> skip the encoder work.
            video_bcthw_padded = video_bcthw[:, :, :_n_pix_keep]
        elif T_in_pix < num_frames_out:
            # Pad with zeros in pixel space (matching official implementation)
            n_pad = num_frames_out - T_in_pix
            zero_pad = torch.zeros(
                (video_bcthw.shape[0], video_bcthw.shape[1], n_pad, video_bcthw.shape[3], video_bcthw.shape[4]),
                device=video_bcthw.device,
                dtype=video_bcthw.dtype,
            )
            video_bcthw_padded = torch.cat([video_bcthw, zero_pad], dim=2)
            # video_bcthw_padded = video_bcthw.repeat(1, 1, num_frames_out, 1, 1)
        else:
            video_bcthw_padded = video_bcthw

        # Encode the padded video (no_grad since VAE is frozen)
        with torch.no_grad():
            if isinstance(generator, list):
                cond_latents_list = [
                    _geak_dist_sample(self.vae.encode(video_bcthw_padded[i].unsqueeze(0)).latent_dist, generator[i])
                    for i in range(batch_size)
                ]
            else:
                cond_latents_list = [
                    _geak_dist_sample(self.vae.encode(vid.unsqueeze(0)).latent_dist, generator)
                    for vid in video_bcthw_padded
                ]

        if _GEAK_VAESAMPLE and len(cond_latents_list) == 1 and cond_latents_list[0].dtype == dtype \
                and cond_latents_list[0].is_contiguous():
            # `torch.cat` on a 1-element list is a full copy of an already
            # contiguous tensor -- one dispatch of pure launch floor.
            cond_latents = cond_latents_list[0]
        else:
            cond_latents = torch.cat(cond_latents_list, dim=0).to(dtype)

        if self.latents_mean is None or self.latents_std is None:
            raise ValueError("VAE configuration must define both `latents_mean` and `latents_std`.")

        latents_mean = _geak_dev_const(self, "lat_mean", self.latents_mean, device, dtype)
        latents_std = _geak_dev_const(self, "lat_std", self.latents_std, device, dtype)
        if _GEAK_FUSE_LATSCALE:
            # (x - m) / s  ==  x * (1/s) + (-m/s).  Two broadcast elementwise
            # kernels (measured 71.5 + 73.4 us on this box -- a 1-CTA
            # TensorIterator over a 5-D broadcast) collapse into one addcmul on
            # a contiguous operand.  The reciprocal/offset are themselves cached
            # device constants, so nothing extra is computed per call.
            cc = self.__dict__.setdefault("_geak_scale_cache", {})
            ck = (str(device), dtype, latents_std.data_ptr(), latents_mean.data_ptr())
            aff = cc.get(ck)
            if aff is None:
                aff = (torch.reciprocal(latents_std).contiguous(),
                       (-latents_mean / latents_std).contiguous())
                cc[ck] = aff
            if _GEAK_VERIFY:
                _ref = (cond_latents - latents_mean) / latents_std
            cond_latents = torch.addcmul(aff[1], cond_latents.contiguous(), aff[0])
            if _GEAK_VERIFY:
                print("[GEAK_VERIFY] latscale max|delta| =",
                      float((cond_latents - _ref).abs().max()), flush=True)
        else:
            cond_latents = (cond_latents - latents_mean) / latents_std

        # Compute output latent shape
        T_out_lat = (num_frames_out - 1) // temporal_factor + 1

        H_lat = int(cond_latents.shape[-2])
        W_lat = int(cond_latents.shape[-1])
        shape_out = (B, C, int(T_out_lat), H_lat, W_lat)

        if latents is None:
            # Generate noise with same pattern across batch dimension
            if generator is not None:
                single_noise = randn_tensor((1, C, int(T_out_lat), H_lat, W_lat), generator=generator, device=device, dtype=dtype)
                latents = single_noise.repeat(B, 1, 1, 1, 1)
            else:
                latents = randn_tensor(shape_out, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
            if tuple(latents.shape) != shape_out:
                raise ValueError(f"Provided `latents` shape {tuple(latents.shape)} != expected latent shape {shape_out}")

        # cond_latents should now already have the correct shape from VAE encode
        # Just ensure it matches shape_out
        if cond_latents.shape[2] != T_out_lat:
            # Truncate or pad if needed (should rarely happen)
            T_copy = min(int(cond_latents.shape[2]), int(T_out_lat))
            if _GEAK_PREB:
                # The zero-pad buffer is call-invariant except for the [:, :, :T_copy]
                # slot that is overwritten every call anyway, so the new_zeros fill is
                # a pure launch-floor dispatch.  Memoised (and therefore pinned for a
                # graph capture) on the module.
                _bk = ("condpad", tuple(int(x) for x in shape_out), dtype, str(device))
                _pc = _geak_preb_cache(self)
                cond_latents_adjusted = _pc.get(_bk)
                if cond_latents_adjusted is None:
                    cond_latents_adjusted = _geak_preb_put(_pc, _bk, latents.new_zeros(shape_out))
            else:
                cond_latents_adjusted = latents.new_zeros(shape_out)
            cond_latents_adjusted[:, :, :T_copy] = cond_latents[:, :, :T_copy].to(dtype=dtype, device=device)
            cond_latents = cond_latents_adjusted

        _, _, T_lat, H_lat, W_lat = shape_out
        num_cond_latent_frames = min(T_lat, (num_frames_in - 1) // temporal_factor + 1)

        if _GEAK_PREB:
            # cond_indicator / cond_mask are a pure function of
            # (B, T_lat, H_lat, W_lat, n_cond, dtype, device): 3 dispatches
            # (zeros + fill + contiguous-expand) of pure launch floor, once per
            # call, on a value that never changes for the life of the episode.
            _pk = ("mask", int(B), int(T_lat), int(H_lat), int(W_lat),
                   int(num_cond_latent_frames), latents.dtype, str(latents.device))
            _pc = _geak_preb_cache(self)
            _hit = _pc.get(_pk)
            if _hit is None:
                _ci = latents.new_zeros(B, 1, T_lat, 1, 1)
                _ci[:, :, 0:num_cond_latent_frames] = 1.0
                _cm = _ci.expand(B, 1, T_lat, H_lat, W_lat).contiguous()
                _hit = _geak_preb_put(_pc, _pk, (_ci, _cm))
            cond_indicator, cond_mask = _hit
            return latents, cond_latents, cond_mask, cond_indicator

        cond_indicator = latents.new_zeros(B, 1, T_lat, 1, 1)
        cond_indicator[:, :, 0:num_cond_latent_frames] = 1.0
        if _GEAK_MASKFUSE:
            # ones_padding is all 1.0 and zeros_padding all 0.0, so
            #   cond_indicator*1 + (1-cond_indicator)*0  ==  cond_indicator
            # broadcast to (B,1,T,H,W).  Six kernels (2 fills + mul + rsub + mul
            # + add) collapse to one contiguous copy.
            cond_mask = cond_indicator.expand(B, 1, T_lat, H_lat, W_lat).contiguous()
            if _GEAK_VERIFY:
                _op = latents.new_ones((B, 1, T_lat, H_lat, W_lat))
                _zp = latents.new_zeros((B, 1, T_lat, H_lat, W_lat))
                _ref = cond_indicator * _op + (1.0 - cond_indicator) * _zp
                print("[GEAK_VERIFY] cond_mask max|delta| =",
                      float((cond_mask - _ref).abs().max()), flush=True)
        else:
            ones_padding = latents.new_ones((B, 1, T_lat, H_lat, W_lat))
            zeros_padding = latents.new_zeros((B, 1, T_lat, H_lat, W_lat))
            cond_mask = cond_indicator * ones_padding + (1.0 - cond_indicator) * zeros_padding

        return latents, cond_latents, cond_mask, cond_indicator

    def _denorm_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Inverse of normalization done in `prepare_latents`:
            latents_norm = (latents_raw - mean) / std
        so:
            latents_raw = latents_norm * std + mean
        Supports mean/std with or without a temporal dimension.
        """
        if self.latents_mean is None or self.latents_std is None:
            raise ValueError("VAE configuration must define both `latents_mean` and `latents_std`.")

        mean = _geak_dev_const(self, "lat_mean", self.latents_mean, latents.device, latents.dtype)
        std = _geak_dev_const(self, "lat_std", self.latents_std, latents.device, latents.dtype)

        # If mean/std have a temporal dimension, slice to match
        if mean.ndim == 5 and latents.ndim == 5 and mean.shape[2] >= latents.shape[2]:
            mean = mean[:, :, : latents.shape[2]]
        if std.ndim == 5 and latents.ndim == 5 and std.shape[2] >= latents.shape[2]:
            std = std[:, :, : latents.shape[2]]

        return latents * std + mean

    def _match_num_frames(self, video: torch.Tensor, target_num_frames: int) -> torch.Tensor:
        if target_num_frames <= 0 or video.shape[2] == target_num_frames:
            return video

        frames_per_latent = max(self.vae_scale_factor_temporal, 1)
        video = torch.repeat_interleave(video, repeats=frames_per_latent, dim=2)

        current_frames = video.shape[2]
        if current_frames < target_num_frames:
            pad = video[:, :, -1:, :, :].repeat(1, 1, target_num_frames - current_frames, 1, 1)
            video = torch.cat([video, pad], dim=2)
        elif current_frames > target_num_frames:
            video = video[:, :, :target_num_frames]

        return video

    def _decode_latents_to_video(self, latents_norm: torch.Tensor) -> torch.Tensor:
        """
        Decode normalized VAE latents to pixel video in [0,1].
        Returns shape (B,T,3,H,W).
        """


        latents_raw = self._denorm_latents(latents_norm).to(dtype=_geak_mod_dtype(self.vae))
        out = self.vae.decode(latents_raw, return_dict=False)[0]
        x = out

        # Normalize layout to (B,T,3,H,W)
        if x.ndim == 5:
            # common: (B,3,T,H,W)
            if x.shape[1] == 3:
                x = x.permute(0, 2, 1, 3, 4).contiguous()
            # else assume already (B,T,3,H,W)
        elif x.ndim == 4:
            # (B,3,H,W) -> single frame
            if x.shape[1] == 3:
                x = x.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected decoded tensor shape {tuple(x.shape)}")
        else:
            raise ValueError(f"Unexpected decoded tensor ndim={x.ndim} shape={tuple(x.shape)}")

        x = x.float()
        min_val, max_val = x.min().item(), x.max().item()

        if min_val < -0.5:  # 真正的 [-1, 1] 范围
            x = (x + 1.0) / 2.0
        x = x.clamp(0.0, 1.0)


        return x


    def _encode_video_to_latents_norm(self, video_bcthw: torch.Tensor) -> torch.Tensor:
        """
        Encode pixel video (B,3,T,H,W) into normalized latents (B,C,T_lat,H_lat,W_lat).
        Deterministic: uses encoder mean.
        """
        if video_bcthw.ndim != 5 or video_bcthw.shape[1] != 3:
            raise ValueError(f"video_bcthw must be (B,3,T,H,W), got {tuple(video_bcthw.shape)}")
        if self.latents_mean is None or self.latents_std is None:
            raise ValueError("VAE configuration must define both `latents_mean` and `latents_std`.")

        vb = video_bcthw.to(device=self.device, dtype=_geak_mod_dtype(self.vae))
        enc = self.vae.encode(vb)
        if hasattr(enc, "latent_dist"):
            latents_raw = enc.latent_dist.mean
        else:
            latents_raw = enc

        latents_raw = latents_raw.to(dtype=torch.float32)
        mean = _geak_dev_const(self, "lat_mean", self.latents_mean, latents_raw.device, latents_raw.dtype)
        std = _geak_dev_const(self, "lat_std", self.latents_std, latents_raw.device, latents_raw.dtype)

        if mean.ndim == 5 and mean.shape[2] >= latents_raw.shape[2]:
            mean = mean[:, :, : latents_raw.shape[2]]
        if std.ndim == 5 and std.shape[2] >= latents_raw.shape[2]:
            std = std[:, :, : latents_raw.shape[2]]

        return (latents_raw - mean) / std

    def _coerce_videos_to_bcthw(self, videos: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        if not isinstance(videos, torch.Tensor) or videos.ndim != 5:
            raise ValueError("`videos` must be a 5D torch.Tensor")

        if videos.shape[1] == 3:
            return videos
        if videos.shape[2] == 3:
            return self.video_processor.preprocess_video(videos, height=height, width=width)
        raise ValueError(f"Unsupported `videos` shape {tuple(videos.shape)}; expected (B,3,T,H,W) or (B,T,3,H,W).")

    def forward(
        self,
        *,
        prompts: Union[str, List[str]],
        videos: torch.Tensor,
        height: int,
        width: int,
        num_inference_steps: int = 1,
        capture_step_index: int = 0,
        conditional_frame_timestep: float = 0.001,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        detach: bool = True,
        num_frames_out: Optional[int] = None,
        gt_future_videos: Optional[torch.Tensor] = None,  # (B,Tf,3,H,W) float in [0,1]
        return_pred_future_video: bool = False,
        fixed_seed: Optional[int] = None,
        action_latents: Optional[torch.Tensor] = None,    # (B,16,T_a,28,28) action latent frames
        state_latents: Optional[torch.Tensor] = None,     # (B,16,T_s,28,28) state latent frames
        training_mode: str = "legacy",                     # "legacy"|"policy"|"world_model"
        action_codec=None,                                 # ActionLatentCodec for decoded action loss
    ) -> torch.Tensor:
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 16 but are {height} and {width}.")

        device = self.device
        transformer_dtype = _geak_mod_dtype(self.transformer)

        b = int(videos.shape[0])
        if isinstance(prompts, str):
            prompts_list: List[str] = [prompts] * b
        else:
            prompts_list = prompts
            if len(prompts_list) != b:
                raise ValueError(f"len(prompts)={len(prompts_list)} must match batch size B={b}")

        # Create fixed generator if seed is provided
        if fixed_seed is not None and generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(fixed_seed)

        with torch.no_grad():
            prompt_embeds, _ = self.encode_prompt(
                prompt=prompts_list,
                guidance_scale=1.0,
                num_videos_per_prompt=1,
                device=device,
                max_sequence_length=self.max_sequence_length,
            )
        _geak_puid = getattr(prompt_embeds, "_geak_prompt_uid", None)
        prompt_embeds = prompt_embeds.detach()
        if _geak_puid is not None:
            # `.detach()` makes a fresh python object and drops python attributes;
            # carry the prompt identity across so the backbone can memoize on it.
            try:
                prompt_embeds._geak_prompt_uid = _geak_puid
            except Exception:
                pass

        with torch.enable_grad():

            videos_bcthw = self._coerce_videos_to_bcthw(videos, height=height, width=width)
            videos_bcthw = videos_bcthw.to(device=device, dtype=_geak_mod_dtype(self.vae)) ##[-1,1]
            num_frames_in = int(videos_bcthw.shape[2])
            # NOTE: Cosmos VAE is temporally downsampled (vae_scale_factor_temporal > 1).
            temporal_factor = int(getattr(self, "vae_scale_factor_temporal", 1) or 1)
            temporal_factor = max(1, temporal_factor)
            min_frames_out_for_future = 1 + temporal_factor

            if num_frames_out is None:
                if gt_future_videos is not None and torch.is_tensor(gt_future_videos):
                    # gt_future_videos: (B,T_f,3,H,W) or (B,3,T_f,H,W)
                    gt_t_dim = gt_future_videos.shape[1] if gt_future_videos.shape[2] == 3 else gt_future_videos.shape[2]
                    num_frames_out = num_frames_in + int(gt_t_dim)
                else:
                    num_frames_out = num_frames_in
                # Ensure enough output frames for at least one future latent slot
                num_frames_out = max(int(num_frames_out), int(min_frames_out_for_future))

            # If we need any future prediction/supervision, ensure enough output frames to create a future latent slot.
            # if (gt_future_videos is not None and torch.is_tensor(gt_future_videos)) or bool(return_pred_future_video):
            #     num_frames_out = max(int(num_frames_out), int(min_frames_out_for_future))

            num_channels_latents = int(self.transformer.config.in_channels) - 1

            # --- Declare how many cond_latent frames THIS call will actually read --------------
            # Only the pure-inference legacy path (no future-video target, no action/state latent
            # injection, no video generation) is provably confined to the conditioned prefix:
            # there `cond_latents` only ever appears as `cond_mask_t * cond_latents` or
            # `(latents - cond_latents) * cond_mask_t`, and `cond_mask_t` is zero for every frame
            # at index >= num_cond_latent_frames. Every other branch reads the trailing frames as
            # its future-video / flow-matching target, so elision stays OFF for them.
            _uses_action_latents = (training_mode != "legacy") and (
                action_latents is not None or state_latents is not None
            )
            _legacy_infer_only = (
                gt_future_videos is None
                and not bool(return_pred_future_video)
                and not _uses_action_latents
                and torch.is_inference_mode_enabled()
                and not self.training
            )
            if _legacy_infer_only:
                self._geak_elide_pad = (int(num_frames_in) - 1) // temporal_factor + 1
            try:
                latents, cond_latents, cond_mask, cond_indicator = self.prepare_latents(
                    video=videos_bcthw,
                    batch_size=b,
                    num_channels_latents=num_channels_latents,
                    height=height,
                    width=width,
                    num_frames_in=num_frames_in,
                    num_frames_out=int(num_frames_out),
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                    latents=None,
                )
            finally:
                # exception-safe: never leave the flag set for a later caller
                self._geak_elide_pad = None

            # --- Action/state latent frame injection (Cosmos-policy style) ---
            # Track how many action temporal frames were appended (for extraction later)
            n_action_frames = 0
            n_state_frames = 0
            if training_mode != "legacy" and (action_latents is not None or state_latents is not None):
                _, _, T_vid, H_lat, W_lat = latents.shape

                def _match_spatial(x: torch.Tensor) -> torch.Tensor:
                    """Interpolate action/state latent frames to match video latent spatial dims."""
                    if x.shape[-2] == H_lat and x.shape[-1] == W_lat:
                        return x
                    B_, C_, T_, _, _ = x.shape
                    flat = x.reshape(B_ * C_ * T_, 1, x.shape[-2], x.shape[-1])
                    flat = F.interpolate(flat, size=(H_lat, W_lat), mode="bilinear", align_corners=False)
                    return flat.reshape(B_, C_, T_, H_lat, W_lat)

                if state_latents is not None:
                    state_latents = _match_spatial(state_latents.to(device=device, dtype=latents.dtype))
                    n_state_frames = state_latents.shape[2]

                if action_latents is not None:
                    action_latents = _match_spatial(action_latents.to(device=device, dtype=latents.dtype))
                    n_action_frames = action_latents.shape[2]

                # Sequence layout: [video_cond | action | future_video | state]
                # Build conditioning based on training_mode:
                #   policy:      condition on video_cond + state, denoise action + future_video
                #   world_model: condition on video_cond + action + state, denoise future_video
                cond_count_vid = int(cond_indicator[0, 0, :, 0, 0].sum().item())

                parts_latents = [cond_latents[:, :, :cond_count_vid]]  # video conditioning frames
                parts_noise = [latents[:, :, :cond_count_vid]]         # noise for same slots
                parts_cond = [torch.ones(cond_count_vid, device=device)]  # always conditioned

                if action_latents is not None:
                    action_noise = torch.randn_like(action_latents)
                    parts_latents.append(action_latents)
                    parts_noise.append(action_noise)
                    if training_mode == "policy":
                        parts_cond.append(torch.zeros(n_action_frames, device=device))  # denoise
                    else:
                        parts_cond.append(torch.ones(n_action_frames, device=device))   # condition

                future_count = T_vid - cond_count_vid
                if future_count > 0:
                    parts_latents.append(cond_latents[:, :, cond_count_vid:])
                    parts_noise.append(latents[:, :, cond_count_vid:])
                    parts_cond.append(torch.zeros(future_count, device=device))  # always denoised

                if state_latents is not None:
                    parts_latents.append(state_latents)
                    parts_noise.append(torch.randn_like(state_latents))
                    parts_cond.append(torch.ones(n_state_frames, device=device))  # always conditioned

                cond_latents = torch.cat(parts_latents, dim=2)
                latents = torch.cat(parts_noise, dim=2)
                T_total = cond_latents.shape[2]

                cond_vec = torch.cat(parts_cond, dim=0)  # (T_total,)
                cond_indicator = cond_vec.view(1, 1, T_total, 1, 1).expand(b, 1, T_total, 1, 1)
                ones_pad = latents.new_ones((b, 1, T_total, H_lat, W_lat))
                zeros_pad = latents.new_zeros((b, 1, T_total, H_lat, W_lat))
                cond_mask = cond_indicator * ones_pad + (1.0 - cond_indicator) * zeros_pad
            # --- End action/state latent frame injection ---

            if _GEAK_PREB:
                # Six launch-floor dispatches on tensors that are a pure function of
                # (cond_indicator identity, cond_mask identity, transformer_dtype,
                #  conditional_frame_timestep, H, W).  cond_indicator/cond_mask are
                # themselves memoised above, so identity is a valid key AND the cache
                # holds them alive for a graph's lifetime.
                _fk = ("fwd", id(cond_indicator), id(cond_mask), transformer_dtype,
                       float(conditional_frame_timestep), int(height), int(width))
                _pc = _geak_preb_cache(self)
                _hit = _pc.get(_fk)
                if _hit is None:
                    _hit = (
                        torch.full_like(cond_indicator, float(conditional_frame_timestep)),
                        cond_mask.to(transformer_dtype),
                        (cond_indicator > 0.5),
                        (cond_mask > 0.5),
                        latents.new_zeros(1, 1, height, width, dtype=transformer_dtype),
                        cond_indicator, cond_mask,      # keep the keys alive
                    )
                    _geak_preb_put(_pc, _fk, _hit)
                cond_timestep, cond_mask_t, _cond_sel, _cond_sel_full, padding_mask = _hit[:5]
                if not _GEAK_MASKFUSE:
                    _cond_sel = None
                    _cond_sel_full = None
            else:
                if _GEAK_MASKFUSE:
                    cond_timestep = torch.full_like(cond_indicator, float(conditional_frame_timestep))
                else:
                    cond_timestep = torch.ones_like(cond_indicator) * float(conditional_frame_timestep)
                cond_mask_t = cond_mask.to(transformer_dtype)
                # boolean selector reused every step; cond_indicator is exactly 0.0/1.0
                _cond_sel = (cond_indicator > 0.5) if _GEAK_MASKFUSE else None
                _cond_sel_full = (cond_mask > 0.5) if _GEAK_MASKFUSE else None
                padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)

            self._cached_hidden.clear()
            self._capture_hidden_enabled = True

            # Resolve future loss type early so we can skip expensive sampling/decoding when training with
            # latent-space flow-matching loss only.
            loss_type = None
            if gt_future_videos is not None:
                cosmos_cfg = getattr(getattr(self, "config", None), "framework", None)
                cosmos_cfg = getattr(cosmos_cfg, "cosmos25", None) if cosmos_cfg is not None else None
                if cosmos_cfg is not None:
                    if isinstance(cosmos_cfg, dict):
                        loss_type = str(cosmos_cfg.get("future_loss_type", "pixel_l1")).lower()
                    else:
                        loss_type = str(getattr(cosmos_cfg, "future_loss_type", "pixel_l1")).lower()
                else:
                    loss_type = "pixel_l1"

            flow_matching_types = ("flow_matching", "latent_flow_matching", "rectified_flow", "rf")
            train_flow_matching_only = (gt_future_videos is not None) and (loss_type in flow_matching_types) and (not return_pred_future_video)

            # If we need generated video / future supervision, run the full denoising loop.
            # Otherwise, we can early-exit after `capture_step_index` (default 0).
            #
            # Match diffusers' Cosmos2.5 Predict pipeline behavior:
            # - Use UniPCMultistepScheduler with `prediction_type="flow_prediction"` + `use_flow_sigmas=True`
            # - Treat transformer output as velocity (a.k.a. flow) and replace conditioned frames with gt_velocity
            # For pure flow-matching training, we don't need to run the sampling loop or decode videos.
            need_generate = ((gt_future_videos is not None) or bool(return_pred_future_video)) and (not train_flow_matching_only)

            # If we aren't generating, only run enough steps to reach capture_step_index (usually 0).
            steps_for_hidden = int(num_inference_steps) if need_generate else max(1, int(capture_step_index) + 1)
            _geak_set_timesteps(self, self.scheduler, steps_for_hidden, device)
            timesteps = self.scheduler.timesteps

            stop_i = (len(timesteps) - 1) if need_generate else capture_step_index
            if stop_i < 0 or stop_i >= len(timesteps):
                raise ValueError(f"stop_i={stop_i} out of range for {len(timesteps)} steps")

            # Determine whether scheduler expects flow/velocity outputs.
            scheduler_cfg = getattr(self.scheduler, "config", None)
            prediction_type = str(getattr(scheduler_cfg, "prediction_type", "")).lower() if scheduler_cfg is not None else ""
            use_flow_prediction = prediction_type == "flow_prediction"

            # In diffusers pipeline_cosmos2_5_predict.py:
            #   gt_velocity = (latents - cond_latent) * cond_mask
            # where `latents` are the initial noise latents and `cond_latent` are the (padded) conditional latents.
            gt_velocity = None
            _gt_vel_deferred = False
            if use_flow_prediction:
                # `gt_velocity` is read ONLY at `if gt_velocity is not None and not
                # _tail_elided`.  On the stop-at-layer path `_tail_elided` is True for
                # every iteration, so the three kernels below are pure dead work.  The
                # decision needs `_stop_layer`, computed a few lines down, so defer.
                _gt_vel_deferred = bool(_GEAK_MASKFUSE)
                if not _gt_vel_deferred:
                    gt_velocity = (latents - cond_latents).to(dtype=transformer_dtype) * cond_mask_t

            # --- Dead transformer-tail elision gate (OPT-IN, all conditions verified at runtime) ----
            # READ-SET ARGUMENT: on this path `model_out` -- the transformer's return value -- is
            # never read again.  It is only consumed by `self.scheduler.step(...)` under
            # `if need_generate:`.  When `need_generate` is False the scheduler never runs, `latents`
            # is never advanced, the loop breaks at `i == stop_i == capture_step_index == 0`, and the
            # ONLY value that escapes is `hidden`, which comes from the forward hook installed on
            # `transformer_blocks[self.extract_layer]` -- not from the transformer's return value.
            # Everything the transformer computes AFTER that block (blocks extract_layer+1..N-1,
            # norm_out, proj_out, unpatchify) is therefore pure dead work.
            #
            # This gate is opt-IN: it fires only when every one of the following is positively true,
            # so any unconsidered caller falls through to the original full-transformer path.
            # All predicates are python bools / ints / len() -- no `.item()`, no device-data tests,
            # hence no data-dependent host sync added to an already host-paced call.
            # (The `gt_velocity + model_out * (1 - cond_mask_t)` recombination at the end of the
            # loop body is likewise dead when `need_generate` is False, so it is skipped too.)
            try:
                _nblocks = len(getattr(self.transformer, "transformer_blocks", ()))
                _elide_tail = (
                    getattr(self, "_geak_tail_elision_enabled", True)  # debug master switch
                    and (not need_generate)                   # scheduler.step never runs
                    and int(stop_i) == 0                      # loop is a single iteration
                    and len(timesteps) >= 1                   # ... and it does execute once
                    and bool(self._capture_hidden_enabled)    # hook will actually store the hidden
                    and _nblocks > 0
                    and 0 <= int(self.extract_layer) < _nblocks - 1   # there IS a tail to elide
                )
            except Exception:
                _elide_tail = False

            # --- Explicit stop-at-layer (replaces the exception-based unwind) ---------
            # Same read-set argument as above; the difference is purely mechanical: the
            # transformer takes a `stop_at_layer` argument and RETURNS the extraction
            # block's output, so there is no forward hook and no Python exception on the
            # path.  (A Python exception cannot cross a graph capture; an argument can.)
            _stop_layer = None
            try:
                if (
                    _GEAK_STOPLAYER
                    and _elide_tail
                    and getattr(self.transformer, "_geak_supports_stop_at_layer", False)
                ):
                    _stop_layer = int(self.extract_layer)
            except Exception:
                _stop_layer = None
            if _stop_layer is not None:
                self._geak_disable_hook()
            else:
                self._geak_enable_hook()
                _elide_tail = bool(_elide_tail) and self._hook_handle is not None

            # Hoist the per-step device->host sync out of the loop: `sigmas[i].item()`
            # is a sync on an already host-paced call.  Move the whole sigma vector to
            # the device once and index it there.
            _sig_dev = None
            try:
                _sig = self.scheduler.sigmas
                if torch.is_tensor(_sig):
                    _sig_dev = _geak_dev_const(self, "sched_sigmas", _sig, device, transformer_dtype)
            except Exception:
                _sig_dev = None

            if _gt_vel_deferred and _stop_layer is None:
                gt_velocity = (latents - cond_latents).to(dtype=transformer_dtype) * cond_mask_t

            hidden_first: Optional[torch.Tensor] = None
            for i, t in enumerate(timesteps):
                # NOTE: for Cosmos2.5 Predict with flow sigmas, scheduler.sigmas are mapped to [0,1]
                if _sig_dev is not None:
                    sigma_t = _sig_dev[i].reshape(1)
                else:
                    sigma_t = (
                        torch.tensor(self.scheduler.sigmas[i].item()).unsqueeze(0).to(device=device, dtype=transformer_dtype)
                    )
                if _GEAK_MASKFUSE:
                    # mask is exactly 0/1 -> the affine blend IS a select, bitwise.
                    in_latents = torch.where(_cond_sel_full, cond_latents, latents).to(transformer_dtype)
                    if _GEAK_PREB and _sig_dev is not None:
                        # in_timestep = [conditional_frame_timestep on the cond frames,
                        # sigma[i] elsewhere] -- both operands are memoised device
                        # constants, so the whole vector is call-invariant.  Keyed on
                        # object identity of the memoised operands (which the cache
                        # itself keeps alive), never on tensor VALUES: no sync.
                        _tk = ("ts", id(_cond_sel), id(cond_timestep), id(_sig_dev), int(i))
                        _pc = _geak_preb_cache(self)
                        _th = _pc.get(_tk)
                        if _th is None:
                            _th = (torch.where(_cond_sel, cond_timestep,
                                               sigma_t.to(cond_timestep.dtype)),
                                   _cond_sel, cond_timestep, _sig_dev)
                            _geak_preb_put(_pc, _tk, _th)
                        in_timestep = _th[0]
                    else:
                        in_timestep = torch.where(_cond_sel, cond_timestep, sigma_t.to(cond_timestep.dtype))
                    if _GEAK_VERIFY:
                        _rl = (cond_mask_t * cond_latents + (1 - cond_mask_t) * latents).to(transformer_dtype)
                        _rt = cond_indicator * cond_timestep + (1 - cond_indicator) * sigma_t
                        print("[GEAK_VERIFY] in_latents max|delta| =",
                              float((in_latents.float() - _rl.float()).abs().max()),
                              " in_timestep max|delta| =",
                              float((in_timestep.float() - _rt.float()).abs().max()), flush=True)
                else:
                    in_latents = cond_mask_t * cond_latents + (1 - cond_mask_t) * latents
                    in_latents = in_latents.to(transformer_dtype)
                    in_timestep = cond_indicator * cond_timestep + (1 - cond_indicator) * sigma_t

                _tail_elided = False
                if _stop_layer is not None:
                    model_out = None
                    _tail_elided = True
                    self._cached_hidden.append(
                        self.transformer(
                            hidden_states=in_latents,
                            condition_mask=cond_mask_t,
                            timestep=in_timestep,
                            encoder_hidden_states=prompt_embeds,
                            padding_mask=padding_mask,
                            return_dict=False,
                            stop_at_layer=_stop_layer,
                        )[0]
                    )
                else:
                    if _elide_tail:
                        self._geak_elide_tail = True
                    try:
                        model_out = self.transformer(
                            hidden_states=in_latents,
                            condition_mask=cond_mask_t,
                            timestep=in_timestep,
                            encoder_hidden_states=prompt_embeds,
                            padding_mask=padding_mask,
                            return_dict=False,
                        )[0]
                    except _GeakTailElided:
                        # Unwound from inside transformer_blocks[extract_layer]'s forward hook.
                        # The hidden state we need is already in self._cached_hidden.
                        model_out = None
                        _tail_elided = True
                    finally:
                        self._geak_elide_tail = False

                # Replace conditional-frame velocity with gt_velocity (diffusers behavior)
                if gt_velocity is not None and not _tail_elided:
                    model_out = gt_velocity + model_out * (1 - cond_mask_t)


                # Capture hidden from the FIRST transformer call and freeze capture afterwards.
                if i == 0 and hidden_first is None and len(self._cached_hidden) > 0:
                    hidden_first = self._cached_hidden[-1]
                    self._capture_hidden_enabled = False

                # Only advance diffusion if we actually need generated video / future loss.
                if need_generate:
                    latents = self.scheduler.step(model_out, t, latents, return_dict=False)[0]

                if i == stop_i:
                    break

            if len(self._cached_hidden) == 0:
                raise RuntimeError(
                    "No transformer hidden captured. Your transformer may not expose `transformer_blocks`, "
                    "or `extract_layer` is not a valid block index."
                )

            hidden = hidden_first if hidden_first is not None else self._cached_hidden[-1]

            # Optional future-video prediction + auxiliary supervision loss
            future_loss = None
            pred_video = None
            action_loss = None
            action_decoded_loss = None

            # --- Cosmos-policy flow matching loss (action latent frames) ---
            if training_mode != "legacy" and n_action_frames > 0:
                # Build full GT x0 for the non-conditioned slots.
                # Layout: [video_cond(C) | action(D) | future_video(D) | state(C)]
                # where C=conditioned, D=denoised (in policy mode).
                # We build x0 for ALL temporal frames; cond_indicator masks the loss.
                T_total_now = latents.shape[2]
                x0_full = cond_latents.clone().to(dtype=torch.float32, device=device)
                # Action slots: GT is the original action_latents
                # Future video slots: GT comes from VAE-encoding the full video
                cond_count_now = int(cond_indicator[0, 0, :, 0, 0].sum().item())

                # Get video GT latents for future video slots
                if gt_future_videos is not None:
                    gt = gt_future_videos
                    if gt.ndim == 5 and gt.shape[1] == 3:
                        gt = gt.permute(0, 2, 1, 3, 4).contiguous()
                    gt_bcthw = gt.permute(0, 2, 1, 3, 4).contiguous()
                    if gt_bcthw.min() >= 0:
                        gt_bcthw = gt_bcthw * 2.0 - 1.0
                    full_bcthw = torch.cat([videos_bcthw.to(gt_bcthw.device, dtype=gt_bcthw.dtype), gt_bcthw], dim=2)
                    temporal_factor_fm = int(getattr(self, "vae_scale_factor_temporal", 1) or 1)
                    temporal_factor_fm = max(1, temporal_factor_fm)
                    min_full_frames = 1 + temporal_factor_fm
                    if int(full_bcthw.shape[2]) < int(min_full_frames):
                        n_pad_fm = int(min_full_frames - int(full_bcthw.shape[2]))
                        last = full_bcthw[:, :, -1:, :, :].repeat(1, 1, n_pad_fm, 1, 1)
                        full_bcthw = torch.cat([full_bcthw, last], dim=2)
                    with torch.no_grad():
                        gt_vid_latents = self._encode_video_to_latents_norm(full_bcthw)
                    # Place video GT latents into x0_full at the correct positions
                    vid_cond_count = (num_frames_in - 1) // max(1, int(self.vae_scale_factor_temporal or 1)) + 1
                    vid_future_start = vid_cond_count + n_action_frames  # after action slots
                    future_count_now = T_vid - vid_cond_count
                    if future_count_now > 0:
                        gt_future_lat = gt_vid_latents[:, :, vid_cond_count:vid_cond_count + future_count_now]
                        x0_full[:, :, vid_future_start:vid_future_start + gt_future_lat.shape[2]] = gt_future_lat.to(dtype=torch.float32)

                # Compute flow matching loss on all non-conditioned frames
                denoise_mask = (1.0 - cond_indicator[0, 0, :, 0, 0]).bool()  # (T_total,)
                n_denoise = int(denoise_mask.sum().item())
                if n_denoise > 0:
                    x0_denoise = x0_full[:, :, denoise_mask]  # (B, C, n_denoise, H, W)

                    B_fm = latents.shape[0]
                    fm_cfg = getattr(cosmos_cfg, "flow_matching", None) if cosmos_cfg is not None else None
                    time_dist = str(getattr(fm_cfg, "time_distribution", "logit_normal")) if fm_cfg else "logit_normal"
                    _hsr = getattr(fm_cfg, "high_sigma_ratio", 0.05) if fm_cfg else 0.05
                    _hsm = getattr(fm_cfg, "high_sigma_min", 0.98) if fm_cfg else 0.98
                    high_sigma_ratio = None if _hsr is None else float(_hsr)
                    high_sigma_min = None if _hsm is None else float(_hsm)

                    if time_dist == "logit_normal":
                        t = torch.sigmoid(torch.randn((B_fm,), device=device, dtype=torch.float32))
                    else:
                        t = torch.rand((B_fm,), device=device, dtype=torch.float32)
                    if high_sigma_ratio is not None and high_sigma_ratio > 0:
                        high_mask_t = torch.rand((B_fm,), device=device) < high_sigma_ratio
                        high_t = torch.rand((B_fm,), device=device, dtype=torch.float32) * (1.0 - high_sigma_min) + high_sigma_min
                        t = torch.where(high_mask_t, high_t, t)
                    t = t.view(B_fm, 1, 1, 1, 1)

                    z_denoise = torch.randn_like(x0_denoise)
                    xt_denoise = (1.0 - t) * x0_denoise + t * z_denoise

                    xt_full = torch.randn_like(latents.float())
                    xt_full[:, :, denoise_mask] = xt_denoise

                    t_B1T11 = latents.new_zeros(cond_indicator.shape, dtype=torch.float32)
                    t_B1T11[:, :, denoise_mask] = t
                    t_B1T11 = t_B1T11.to(dtype=transformer_dtype)

                    in_latents_fm = cond_mask_t * cond_latents + (1 - cond_mask_t) * xt_full.to(transformer_dtype)
                    in_timestep_fm = cond_indicator * cond_timestep + (1 - cond_indicator) * t_B1T11

                    v_pred = self.transformer(
                        hidden_states=in_latents_fm,
                        condition_mask=cond_mask_t,
                        timestep=in_timestep_fm,
                        encoder_hidden_states=prompt_embeds,
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0]

                    v_tgt_denoise = (z_denoise - x0_denoise).to(device=v_pred.device, dtype=v_pred.dtype)
                    v_pred_denoise = v_pred[:, :, denoise_mask]

                    combined_loss = F.mse_loss(v_pred_denoise.float(), v_tgt_denoise.float())

                    # Split loss into action and video components for logging
                    if n_action_frames > 0 and training_mode == "policy":
                        # Action frames are the first n_action_frames in the denoised set
                        v_pred_action = v_pred_denoise[:, :, :n_action_frames]
                        v_tgt_action = v_tgt_denoise[:, :, :n_action_frames]
                        action_loss = F.mse_loss(v_pred_action.float(), v_tgt_action.float())

                        # Direct decoded action supervision
                        if action_codec is not None:
                            xt_action = xt_denoise[:, :, :n_action_frames]
                            x0_pred_action = xt_action - t * v_pred_action.float()
                            x0_gt_action = x0_denoise[:, :, :n_action_frames]
                            decoded_pred = action_codec.decode_actions(x0_pred_action, action_stats=None)
                            decoded_gt = action_codec.decode_actions(x0_gt_action, action_stats=None)
                            action_decoded_loss = F.mse_loss(decoded_pred, decoded_gt)

                        if v_pred_denoise.shape[2] > n_action_frames:
                            v_pred_vid = v_pred_denoise[:, :, n_action_frames:]
                            v_tgt_vid = v_tgt_denoise[:, :, n_action_frames:]
                            future_loss = F.mse_loss(v_pred_vid.float(), v_tgt_vid.float())
                        else:
                            future_loss = torch.tensor(0.0, device=device)
                    else:
                        future_loss = combined_loss
                else:
                    future_loss = torch.tensor(0.0, device=device)
                    action_loss = torch.tensor(0.0, device=device)

            # --- Legacy flow matching loss (original DiT4DiT path) ---
            elif gt_future_videos is not None or return_pred_future_video:
                need_decode = bool(return_pred_future_video) or (
                    (gt_future_videos is not None) and (loss_type is not None) and (loss_type not in ("latent_mse",) + flow_matching_types)
                )
                pred_video_full = None
                if need_decode:
                    pred_video_full = self._decode_latents_to_video(latents)
                    if return_pred_future_video:
                        pred_video = pred_video_full

                if gt_future_videos is not None:
                    gt = gt_future_videos
                    if gt.ndim == 5 and gt.shape[2] == 3:
                        pass
                    elif gt.ndim == 5 and gt.shape[1] == 3:
                        gt = gt.permute(0, 2, 1, 3, 4).contiguous()
                    else:
                        raise ValueError(f"gt_future_videos must be (B,T,3,H,W) or (B,3,T,H,W), got {tuple(gt.shape)}")

                    loss_type = loss_type or "pixel_l1"

                    if loss_type in flow_matching_types:
                        gt_bcthw = gt.permute(0, 2, 1, 3, 4).contiguous()
                        if gt_bcthw.min() >= 0:
                            gt_bcthw = gt_bcthw * 2.0 - 1.0
                        full_bcthw = torch.cat([videos_bcthw.to(gt_bcthw.device, dtype=gt_bcthw.dtype), gt_bcthw], dim=2)

                        temporal_factor = int(getattr(self, "vae_scale_factor_temporal", 1) or 1)
                        temporal_factor = max(1, temporal_factor)
                        min_full_frames = 1 + temporal_factor
                        if int(full_bcthw.shape[2]) < int(min_full_frames):
                            n_pad = int(min_full_frames - int(full_bcthw.shape[2]))
                            last = full_bcthw[:, :, -1:, :, :].repeat(1, 1, n_pad, 1, 1)
                            full_bcthw = torch.cat([full_bcthw, last], dim=2)

                        with torch.no_grad():
                            gt_latents_norm = self._encode_video_to_latents_norm(full_bcthw)

                        cond_count = int(cond_indicator[0, 0, :, 0, 0].sum().item())
                        pred_latents_future = latents[:, :, cond_count:]
                        gt_latents_future = gt_latents_norm[:, :, cond_count : cond_count + pred_latents_future.shape[2]]

                        if pred_latents_future.numel() == 0 or gt_latents_future.numel() == 0:
                            future_loss = torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
                        else:
                            T_sup = min(int(pred_latents_future.shape[2]), int(gt_latents_future.shape[2]))
                            x0_future = gt_latents_future[:, :, :T_sup].to(device=latents.device, dtype=torch.float32)

                            B_fm = latents.shape[0]
                            fm_cfg = getattr(cosmos_cfg, "flow_matching", None) if cosmos_cfg is not None else None
                            time_dist = str(getattr(fm_cfg, "time_distribution", "logit_normal")) if fm_cfg else "logit_normal"
                            _hsr = getattr(fm_cfg, "high_sigma_ratio", 0.05) if fm_cfg else 0.05
                            _hsm = getattr(fm_cfg, "high_sigma_min", 0.98) if fm_cfg else 0.98
                            high_sigma_ratio = None if _hsr is None else float(_hsr)
                            high_sigma_min = None if _hsm is None else float(_hsm)

                            if time_dist == "logit_normal":
                                t = torch.sigmoid(torch.randn((B_fm,), device=latents.device, dtype=torch.float32))
                            else:
                                t = torch.rand((B_fm,), device=latents.device, dtype=torch.float32)

                            if high_sigma_ratio is not None and high_sigma_ratio > 0:
                                high_mask = torch.rand((B_fm,), device=latents.device) < high_sigma_ratio
                                high_t = torch.rand((B_fm,), device=latents.device, dtype=torch.float32) * (1.0 - high_sigma_min) + high_sigma_min
                                t = torch.where(high_mask, high_t, t)

                            t = t.view(B_fm, 1, 1, 1, 1)

                            z_future = torch.randn_like(x0_future)
                            xt_future = (1.0 - t) * x0_future + t * z_future

                            xt_full = torch.randn_like(latents[:, :, :, :, :].float())
                            xt_full[:, :, cond_count : cond_count + T_sup] = xt_future

                            t_B1T11 = latents.new_zeros(cond_indicator.shape, dtype=torch.float32)
                            t_B1T11[:, :, cond_count : cond_count + T_sup] = t
                            t_B1T11 = t_B1T11.to(dtype=transformer_dtype)

                            in_latents_fm = cond_mask_t * cond_latents + (1 - cond_mask_t) * xt_full.to(transformer_dtype)
                            in_timestep_fm = cond_indicator * cond_timestep + (1 - cond_indicator) * t_B1T11

                            v_pred = self.transformer(
                                hidden_states=in_latents_fm,
                                condition_mask=cond_mask_t,
                                timestep=in_timestep_fm,
                                encoder_hidden_states=prompt_embeds,
                                padding_mask=padding_mask,
                                return_dict=False,
                            )[0]

                            v_tgt_future = (z_future - x0_future).to(device=v_pred.device, dtype=v_pred.dtype)
                            v_pred_future = v_pred[:, :, cond_count : cond_count + T_sup]

                            future_loss = F.mse_loss(v_pred_future.float(), v_tgt_future.float())

                    elif loss_type == "latent_mse":
                        gt_bcthw = gt.permute(0, 2, 1, 3, 4).contiguous()
                        full_bcthw = torch.cat([videos_bcthw.to(gt_bcthw.device, dtype=gt_bcthw.dtype), gt_bcthw], dim=2)

                        with torch.no_grad():
                            gt_latents_norm = self._encode_video_to_latents_norm(full_bcthw)

                        cond_count = int(cond_indicator[0, 0, :, 0, 0].sum().item())
                        pred_latents_future = latents[:, :, cond_count:]
                        gt_latents_future = gt_latents_norm[:, :, cond_count : cond_count + pred_latents_future.shape[2]]

                        if pred_latents_future.numel() == 0 or gt_latents_future.numel() == 0:
                            future_loss = torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
                        else:
                            future_loss = F.mse_loss(
                                pred_latents_future.float(),
                                gt_latents_future.to(pred_latents_future.device, dtype=torch.float32),
                            )
                    else:
                        if pred_video_full is None:
                            future_loss = torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
                        else:
                            tf = int(gt.shape[1])
                            if pred_video_full.shape[1] >= 1 + tf:
                                pred_future = pred_video_full[:, 1 : 1 + tf]
                            else:
                                pred_future = pred_video_full[:, -tf:]
                            gt_pix = gt.to(pred_future.device, dtype=pred_future.dtype)
                            if loss_type in ("pixel_mse", "mse"):
                                future_loss = F.mse_loss(pred_future, gt_pix)
                            else:
                                future_loss = torch.abs(pred_future - gt_pix).mean()

            if detach:
                hidden = hidden.detach()
                if pred_video is not None:
                    pred_video = pred_video.detach()

            if training_mode != "legacy" and n_action_frames > 0:
                return hidden, pred_video, future_loss, action_loss, action_decoded_loss

            if gt_future_videos is not None or return_pred_future_video:
                return hidden, pred_video, future_loss

            return hidden


class _Cosmos25_Interface(nn.Module):
    """
    Interface wrapper around Cosmos25FeatureExtractor.

    Expected usage pattern:
      inputs = build_cosmos_inputs(images=batch_images, instructions=instructions)
      out = self(**inputs, output_hidden_states=True, return_dict=True)
      out.hidden_states -> list[Tensor] each (B, S, D)
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = config

        cosmos_cfg = getattr(config.framework, "cosmos25", None) if config is not None else None
        if cosmos_cfg is None:
            cosmos_cfg = {}

        model_id = cosmos_cfg.get("model_id_or_path", None) or cosmos_cfg.get("base_model", None)
        if not model_id:
            raise ValueError("Cosmos25 backend requires `framework.cosmos25.model_id_or_path`")

        self.extract_layer = int(cosmos_cfg.get("extract_layer", 19))
        self.max_sequence_length = int(cosmos_cfg.get("max_sequence_length", 512))
        self.revision = cosmos_cfg.get("revision", "diffusers/base/post-trained")
        self.local_files_only = bool(cosmos_cfg.get("local_files_only", True))
        self.trainable = bool(cosmos_cfg.get("trainable", False))

        dtype_str = cosmos_cfg.get("torch_dtype", "bfloat16")
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
            str(dtype_str).lower(), torch.bfloat16
        )

        self.extractor = Cosmos25FeatureExtractor(
            model_id_or_path=model_id,
            revision=self.revision,
            torch_dtype=torch_dtype,
            local_files_only=self.local_files_only,
            extract_layer=self.extract_layer,
            max_sequence_length=self.max_sequence_length,
            trainable=self.trainable,
            device=cosmos_cfg.get("device", None),
            config=config,
        )

        # Try to set vl_hidden_dim in config if missing
        if config is not None and getattr(config.framework.cosmos25, "vl_hidden_dim", None) in (None, 0):
            hs = self.extractor.hidden_size
            if hs is not None:
                config.framework.cosmos25.vl_hidden_dim = int(hs)

    @property
    def device(self) -> torch.device:
        return self.extractor.device

    @staticmethod
    def _to_chw_float01(x) -> torch.Tensor:
        """
        Accept PIL / numpy(HWC) / torch(CHW or HWC) and return torch float tensor in CHW, normalized to [0,1].
        """
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu()
            if t.ndim == 3 and t.shape[0] in (1, 3, 4):  # CHW
                if t.shape[0] == 4:
                    t = t[:3]
            elif t.ndim == 3 and t.shape[-1] in (1, 3, 4):  # HWC
                if t.shape[-1] == 4:
                    t = t[..., :3]
                t = t.permute(2, 0, 1).contiguous()
            else:
                raise ValueError(f"Unsupported torch image shape {tuple(t.shape)}; expected CHW or HWC.")
            t = t.float()
            if t.max() > 1.0:
                t = t / 255.0
            return t.contiguous()

        if isinstance(x, np.ndarray):
            arr = x
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            if arr.ndim != 3:
                raise ValueError(f"Expected numpy image with shape (H,W,C), got {arr.shape}")
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.shape[2] != 3:
                raise ValueError(f"Expected RGB image, got shape={arr.shape}")
            t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
            if t.max() > 1.0:
                t = t / 255.0
            return t.contiguous()

        # PIL.Image.Image or compatible
        arr = np.array(x, dtype=np.uint8)  # (H,W,3)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected RGB image, got shape={arr.shape}")
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
        return t.contiguous()

    @staticmethod
    def _as_uint8_hwc3(x):
        """
        Return an (H, W, 3) contiguous **uint8** numpy view of `x`, or None if `x`
        is not provably a uint8 RGB image.

        This is a *dtype/attribute* test only -- it never inspects pixel VALUES, so
        it costs nothing and (crucially) introduces no device->host synchronization.
        Anything that is not provably uint8 falls back to the original slow path,
        so the [0,1] scaling invariant (`/255` iff uint8) is established by
        construction rather than by probing `t.max()`.
        """
        arr = None
        if isinstance(x, np.ndarray):
            arr = x
        elif isinstance(x, torch.Tensor):
            if x.device.type == "cpu" and x.dtype == torch.uint8:
                arr = x.detach().numpy()
            else:
                return None
        elif hasattr(x, "mode") and hasattr(x, "size"):  # PIL.Image.Image
            if getattr(x, "mode", None) not in ("RGB", "RGBA"):
                return None
            arr = np.asarray(x, dtype=np.uint8)
        else:
            return None

        if arr is None or arr.dtype != np.uint8 or arr.ndim != 3:
            return None
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.shape[2] != 3:
            return None
        return arr

    def _build_cond_videos_gpu(self, cond_views) -> Optional[torch.Tensor]:
        """
        Fast path for the policy hot loop: build the conditioning tensor
        DIRECTLY on the GPU, already in the (B, C=3, T=1, H, W) layout and the
        [-1,1] range that `VideoProcessor.preprocess_video` would have produced.

        Returning the (B,3,T,H,W) layout makes the consumer
        (`DiT4DiT._predict_action_policy`, and `_coerce_videos_to_bcthw` here)
        take its `videos.shape[1] == 3` identity branch, so the resize/normalize
        never runs on the CPU in fp32 while the GPU sits idle.

        Returns None if the fast path does not apply (caller falls back).
        """
        arrs = []
        for v in cond_views:
            a = self._as_uint8_hwc3(v)
            if a is None:
                return None
            arrs.append(a)
        if len(arrs) == 0:
            return None
        h0, w0 = arrs[0].shape[0], arrs[0].shape[1]
        for a in arrs:
            if a.shape[0] != h0 or a.shape[1] != w0:
                return None

        vp = getattr(self.extractor, "video_processor", None)
        cfg = getattr(vp, "config", None)
        if cfg is None:
            return None
        # Replicate VaeImageProcessor semantics exactly, but decided from
        # host-side attributes only (no value probes, no syncs).
        if bool(getattr(cfg, "do_binarize", False)):
            return None
        if bool(getattr(cfg, "do_convert_grayscale", False)):
            return None
        do_resize = bool(getattr(cfg, "do_resize", True))
        do_normalize = bool(getattr(cfg, "do_normalize", True))
        vsf = int(getattr(cfg, "vae_scale_factor", 8) or 1)
        # get_default_height_width(): floor to a multiple of vae_scale_factor
        tgt_h = h0 - (h0 % vsf) if vsf > 0 else h0
        tgt_w = w0 - (w0 % vsf) if vsf > 0 else w0

        device = self.extractor.device
        if getattr(device, "type", "cpu") != "cuda":
            return None

        # One contiguous uint8 HtoD copy (4x smaller than the fp32 one it
        # replaces); everything after this is on-device.
        stacked = np.stack(arrs, axis=0) if len(arrs) > 1 else arrs[0][None]
        stacked = np.ascontiguousarray(stacked)                      # (B,H,W,3) u8
        if _GEAK_PINNED_H2D:
            # `torch.from_numpy(x).to(cuda, non_blocking=True)` out of PAGEABLE
            # memory is NOT asynchronous: ROCm falls back to a blocking
            # hipMemcpyWithStream that drains the whole stream.  Stage through a
            # persistent PINNED buffer so the copy is a real async DMA.
            pin = self.__dict__.get("_geak_pin_buf")
            if (pin is None or tuple(pin.shape) != stacked.shape):
                try:
                    pin = torch.empty(stacked.shape, dtype=torch.uint8, pin_memory=True)
                    self.__dict__["_geak_pin_buf"] = pin
                except Exception:
                    pin = None
                    self.__dict__["_geak_pin_buf"] = None
            if pin is not None:
                np.copyto(pin.numpy(), stacked, casting="no")
                t = pin.to(device=device, non_blocking=True)
            else:
                t = torch.from_numpy(stacked).to(device=device, non_blocking=True)
        else:
            t = torch.from_numpy(stacked).to(device=device, non_blocking=True)
        t = t.permute(0, 3, 1, 2)                                    # (B,3,H,W) u8
        t = t.to(dtype=torch.float32).div_(255.0)                    # [0,1]
        # The reference path casts to the VAE dtype BEFORE resize/normalize
        # (DiT4DiT does `videos.to(dtype=vae.dtype)` ahead of preprocess_video).
        # Match that exactly so the resulting tensor is bit-identical.
        vae_dtype = _geak_mod_dtype(getattr(self.extractor, "vae", None), None)
        if vae_dtype is not None:
            t = t.to(dtype=vae_dtype)
        if do_resize and (tgt_h != h0 or tgt_w != w0):
            # VaeImageProcessor.resize() for tensors uses F.interpolate default
            # mode, i.e. "nearest".
            t = torch.nn.functional.interpolate(t, size=(tgt_h, tgt_w))
        if do_normalize:
            t = t.mul_(2.0).sub_(1.0)                                # [-1,1]
        return t.unsqueeze(2).contiguous()                           # (B,3,T=1,H,W)

    # ---- GEAK r5_d2: raw-uint8 graph input -------------------------------
    def _geak_raw_build(self, images, instructions):
        """HOST-ONLY prologue for the captured backbone.

        Returns the raw uint8 (B, H, W, 3) frame plus the host-side decisions
        `_build_cond_videos_gpu` would have made, or None when the GPU fast path
        does not apply (caller then falls back to `build_cosmos_inputs`).

        Not a single torch op runs here -- no allocation, no dispatch, no
        module-property walk -- so the exposed head-of-call host CPU collapses to
        numpy stacking.  The device half is `_geak_pre_from_u8`, which runs
        INSIDE the graph.
        """
        if not _GEAK_PREGRAPH:
            return None
        if len(images) != len(instructions):
            return None
        if not all(isinstance(v, (list, tuple)) and len(v) == 1 for v in images):
            return None
        arrs = []
        for v in images:
            a = self._as_uint8_hwc3(v[0])
            if a is None:
                return None
            arrs.append(a)
        if len(arrs) == 0:
            return None
        h0, w0 = int(arrs[0].shape[0]), int(arrs[0].shape[1])
        for a in arrs:
            if int(a.shape[0]) != h0 or int(a.shape[1]) != w0:
                return None
        vp = getattr(self.extractor, "video_processor", None)
        cfg = getattr(vp, "config", None)
        if cfg is None:
            return None
        if bool(getattr(cfg, "do_binarize", False)):
            return None
        if bool(getattr(cfg, "do_convert_grayscale", False)):
            return None
        do_resize = bool(getattr(cfg, "do_resize", True))
        do_normalize = bool(getattr(cfg, "do_normalize", True))
        vsf = int(getattr(cfg, "vae_scale_factor", 8) or 1)
        tgt_h = h0 - (h0 % vsf) if vsf > 0 else h0
        tgt_w = w0 - (w0 % vsf) if vsf > 0 else w0
        device = _geak_mod_device(self.extractor, None)
        if getattr(device, "type", "cpu") != "cuda":
            return None
        # PERSISTENT PINNED STAGING, and the whole descriptor memoised.
        # `np.stack` + `np.ascontiguousarray` + `torch.from_numpy` allocate three
        # objects per call on the exposed path; a pinned buffer that is filled in
        # place removes all three AND turns the pageable H2D (measured 0.023 ms,
        # the single largest exposed item once the preprocess moved in-graph)
        # into a real async DMA.  The buffer is rewritten only at the START of a
        # call, i.e. after the previous call's terminal .cpu() has drained the
        # stream, so the async copy can never be racing a live DMA.
        B = len(arrs)
        shape = (B, h0, w0, 3)
        mk = (shape, tuple(instructions), do_resize, do_normalize,
              int(tgt_h), int(tgt_w), str(device))
        memo = self.__dict__.get("_geak_raw_memo")
        if memo is None:
            memo = {}
            self.__dict__["_geak_raw_memo"] = memo
        raw = memo.get(mk)
        if raw is None:
            host = None
            try:
                host = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
            except Exception:
                host = torch.empty(shape, dtype=torch.uint8)
            raw = {
                "host": host,
                "host_np": host.numpy(),
                "shape": shape,
                "prompts": list(instructions),
                "height": int(tgt_h),
                "width": int(tgt_w),
                "meta": (do_resize, do_normalize, h0, w0, int(tgt_h), int(tgt_w)),
                "device": device,
            }
            if len(memo) < _GEAK_PREB_MAX:
                memo[mk] = raw
        hnp = raw["host_np"]
        if B == 1:
            np.copyto(hnp[0], arrs[0], casting="no")
        else:
            for _i, _a in enumerate(arrs):
                np.copyto(hnp[_i], _a, casting="no")
        return raw

    def _geak_pre_from_u8(self, t_u8, meta):
        """DEVICE half of `_build_cond_videos_gpu`, from the H2D onwards.

        Byte-for-byte the same op sequence in the same order as the eager path,
        so the produced (B,3,1,H,W) tensor is bitwise identical -- the only
        change is that these ops become graph NODES instead of eager launches on
        an idle GPU."""
        do_resize, do_normalize, h0, w0, tgt_h, tgt_w = meta
        # Do the NHWC->NCHW transpose on the UINT8 tensor, not at the end on the
        # bf16 one: same values (a permutation is exact, so this stays bitwise
        # identical), but the copy moves 1 byte/element instead of 2 AND every
        # later op then runs on contiguous memory instead of a permuted view --
        # and the trailing `.contiguous()` becomes a no-op because
        # `unsqueeze(2)` on a contiguous (B,3,H,W) is already contiguous.
        vae_dtype = _geak_mod_dtype(getattr(self.extractor, "vae", None), None)
        # GEAK r2_d0: six graph nodes -> one.  Only when no resize is needed
        # (the deployed 224x224 case); anything else falls through untouched.
        if not (do_resize and (tgt_h != h0 or tgt_w != w0)):
            _f = _geak_pre_fused(t_u8, do_normalize, vae_dtype)
            if _f is not None:
                if _GEAK_PREVERIFY:
                    _r = t_u8.permute(0, 3, 1, 2).contiguous()
                    _r = _r.to(dtype=torch.float32).div_(255.0)
                    _r = _r.to(dtype=vae_dtype)
                    if do_normalize:
                        _r = _r.mul_(2.0).sub_(1.0)
                    _r = _r.unsqueeze(2).contiguous()
                    print("[GEAK_PRE_VERIFY] bitwise_equal=%s max|d|=%.6g n=%d"
                          % (bool(torch.equal(_f, _r)),
                             float((_f.float() - _r.float()).abs().max()),
                             _f.numel()), flush=True)
                return _f
        t = t_u8.permute(0, 3, 1, 2).contiguous()                    # (B,3,H,W) u8
        t = t.to(dtype=torch.float32).div_(255.0)                    # [0,1]
        if vae_dtype is not None:
            t = t.to(dtype=vae_dtype)
        if do_resize and (tgt_h != h0 or tgt_w != w0):
            t = torch.nn.functional.interpolate(t, size=(tgt_h, tgt_w))
        if do_normalize:
            t = t.mul_(2.0).sub_(1.0)                                # [-1,1]
        return t.unsqueeze(2).contiguous()                           # (B,3,T=1,H,W)

    def build_cosmos_inputs(
        self,
        images: Sequence[Sequence[Any]],
        instructions: Sequence[str],
        **kwargs,
    ) -> dict:
        """
        Convert multi-frame images + instruction into Cosmos inputs.

        `images` is a list of lists: images[b] = [frame_0, frame_1, ..., frame_T-1]
        where frames are ordered temporally after pixel-space downsampling.

        Frame 0 is used as the conditioning frame (videos), and the remaining
        frames are future frames for video flow-matching supervision (future_videos).
        """

        if len(images) != len(instructions):
            raise ValueError("images and instructions must have the same batch size")

        # ---- GPU fast path (policy inference: exactly one view per sample) ----
        if all(isinstance(v, (list, tuple)) and len(v) == 1 for v in images):
            fast = self._build_cond_videos_gpu([v[0] for v in images])
            if fast is not None:
                return {
                    "prompts": list(instructions),
                    "videos": fast,                       # (B,3,1,H,W), on CUDA, [-1,1]
                    "height": int(fast.shape[-2]),
                    "width": int(fast.shape[-1]),
                    "future_videos": None,
                }

        all_cond = []      # frame 0 per sample
        all_future = []    # frames 1..T-1 per sample
        n_future = None

        for views in images:
            if not isinstance(views, (list, tuple)) or len(views) == 0:
                raise ValueError("Each sample must provide at least 1 view image")

            chw_frames = [self._to_chw_float01(v) for v in views]

            all_cond.append(chw_frames[0])
            if len(chw_frames) > 1:
                future = torch.stack(chw_frames[1:], dim=0)  # (T_f, C, H, W)
                all_future.append(future)
                if n_future is None:
                    n_future = future.shape[0]

        # videos (condition): (B, T=1, C, H, W)
        videos = torch.stack(all_cond, dim=0).unsqueeze(1)
        height, width = int(videos.shape[-2]), int(videos.shape[-1])

        # future_videos: (B, T_f, C, H, W) or None
        future_videos = None
        if len(all_future) > 0:
            future_videos = torch.stack(all_future, dim=0)  # (B, T_f, C, H, W)

        return {
            "prompts": list(instructions),
            "videos": videos,
            "height": height,
            "width": width,
            "future_videos": future_videos,
        }

    @staticmethod
    def _hidden_to_bsd(hidden: torch.Tensor) -> torch.Tensor:
        """
        Normalize different possible hidden shapes to (B, S, D) for DiT conditioning.
        Supported:
          - (B, S, D): returned as-is
          - (B, C, T, H, W): flattened to tokens (S=T*H*W), D=C
        """
        if hidden.ndim == 3:
            return hidden
        if hidden.ndim == 5:
            b, c, t, h, w = hidden.shape
            x = hidden.permute(0, 2, 3, 4, 1).contiguous().view(b, t * h * w, c)
            return x
        raise ValueError(f"Unsupported hidden shape {tuple(hidden.shape)}; expected 3D or 5D tensor.")

    def forward(
        self,
        *,
        prompts: Union[str, List[str]],
        videos: torch.Tensor,
        height: int,
        width: int,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        future_videos: Optional[torch.Tensor] = None,
        predict_future: bool = False,
        action_latents: Optional[torch.Tensor] = None,
        state_latents: Optional[torch.Tensor] = None,
        training_mode: str = "legacy",
        action_codec=None,
        **kwargs,
    ):
        _ = kwargs

        future_loss = None
        action_loss = None
        action_decoded_loss = None
        pred_future_video = None
        fixed_seed = 42

        cosmos_cfg = getattr(self.config.framework, "cosmos25", None) if self.config is not None else None
        conditional_frame_timestep = float(getattr(cosmos_cfg, "conditional_frame_timestep", 0.001)) if cosmos_cfg is not None else 0.001

        train_num_frames_out = None
        if cosmos_cfg is not None and self.config is not None:
            data_cfg = getattr(getattr(self.config, "datasets", None), "vla_data", None)
            if data_cfg is not None:
                video_delta = getattr(data_cfg, "video_delta_indices", None)
                if video_delta is not None:
                    n_raw = len(video_delta)
                    ratio = int(getattr(data_cfg, "action_video_freq_ratio", 1))
                    train_num_frames_out = len(range(0, n_raw, ratio))

        use_action_latents = training_mode != "legacy" and action_latents is not None
        need_fm = predict_future or (future_videos is not None) or use_action_latents

        if need_fm:
            future_steps = int(getattr(cosmos_cfg, "future_num_inference_steps", 2)) if cosmos_cfg is not None else 2
            result = self.extractor(
                prompts=prompts,
                videos=videos,
                height=height,
                width=width,
                detach=True,
                gt_future_videos=future_videos,
                return_pred_future_video=False,
                num_inference_steps=future_steps,
                num_frames_out=train_num_frames_out,
                conditional_frame_timestep=conditional_frame_timestep,
                action_latents=action_latents,
                state_latents=state_latents,
                training_mode=training_mode,
                action_codec=action_codec,
            )
            if use_action_latents:
                hidden, pred_future_video, future_loss, action_loss, action_decoded_loss = result
            else:
                hidden, pred_future_video, future_loss = result
        else:
            hidden = self.extractor(
                prompts=prompts,
                videos=videos,
                height=height,
                width=width,
                detach=True,
                num_frames_out=train_num_frames_out,
                conditional_frame_timestep=conditional_frame_timestep,
            )

        bsd = self._hidden_to_bsd(hidden)

        out = BackboneOutput(
            hidden_states=[bsd],
            future_video_loss=future_loss,
            pred_future_video=pred_future_video,
            action_loss=action_loss,
            action_decoded_loss=action_decoded_loss,
        )
        return out


