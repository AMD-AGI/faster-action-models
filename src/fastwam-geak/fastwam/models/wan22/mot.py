# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wan_video_dit import flash_attention, modulate, rope_apply, FW_CALL_SEQ, _STEPMEMO
try:
    from .wan_video_dit import qrms_attention as _qrms_attention
except Exception:  # pragma: no cover
    _qrms_attention = None
from .fused_adaln import adaln_forward
from .compile_config import maybe_compile, is_compile_enabled
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# SPLIT-K FOR THE OUTPUT PROJECTIONS (grid fill, part 2).
#
# Measured from the generated inductor code + profile on gfx950/256CU:
#   fused qkv     M=32 N=9216 K=1024  grid=576 ->  7.5 us  (2510 GB/s)
#   ffn-up        M=32 N=4096 K=1024  grid=256 ->  5.2 us  (1613 GB/s)
#   cross q       M=32 N=3072 K=1024  grid= 96 ->  5.3 us  (1182 GB/s)
#   *o-proj*      M=32 N=1024 K=3072  grid= 64 -> 16.3 us  ( 386 GB/s)  x60/step
#   *ffn-down*    M=32 N=1024 K=4096  grid= 64 -> 20.1 us  ( 417 GB/s)  x29/step
# Achieved bandwidth is LINEAR in the CTA count -- these are grid-starved, not
# bandwidth-limited. N=1024 with BLOCK_N=16 is already the widest grid the plain
# template can build, so the only remaining axis is K.
#
# Inductor already ships the exact split-K we want (`decompose_k`: a k_split-way
# bmm with `out_dtype=float32` plus an fp32 reduction -- fp32 accumulation is
# PRESERVED, cf. COMMANDMENT rule 7), but on ROCm it is disabled by default and
# its K/N ratio gate (32x) never fires at N=1024. Lower the gate to 3 so it is
# offered for exactly these two shapes (K/N = 3 and 4) and for nothing else --
# every other GEMM here has K=1024 < 3*N. It is still only a *candidate*:
# inductor benchmarks it against the plain template and keeps the faster one.
#
# `decompose_k` is wired into `tuned_mm` only, not `tuned_addmm`, so these two
# call sites issue the bias as a separate (epilogue-fusable) add.
# ---------------------------------------------------------------------------
_SPLITK = os.environ.get("FASTWAM_SPLITK", "1").lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# MFMA INSTRUCTION WIDTH FOR THE HAND-WRITTEN SPLIT-K GEMMs  (round 2, r2_d2).
#
# WHAT WAS MISSING.  Every `triton.jit` launch in this file passes exactly two
# backend options -- `num_warps` and `num_stages`.  The gfx9 backend takes three
# more that only the GEMM-shaped kernels care about: `matrix_instr_nonkdim`
# (16 -> `v_mfma_f32_16x16x16`, 32 -> `v_mfma_f32_32x32x8`), `kpack` and
# `waves_per_eu`.  Triton picks the MFMA width heuristically from the tile
# shape, and on BOTH of this tree's hand-written split-K GEMMs it picks the
# 32x32 form; the AMD guidance (and the measurement below) is that 16x16 is the
# better default for LLM-shaped N/K because it needs fewer VGPRs per accumulator
# tile and issues at a finer granularity.  It is NOT reachable from
# `FASTWAM_PFSKCFG` / `_SKMM_DEFAULTS`, which only expose tile/split/warp axes,
# so no previous round could have found it.
#
# MEASURED, in-graph device time from the PROFILE leg (medians of 4-6 runs each,
# the same profile that produced the region tables):
#     `_fw_pfsk_kernel`  (prefill o-proj, M=98 N=3072 K=3072, 60 launches/call)
#                         9.18-9.34 us  ->  8.81-8.91 us
#     `_fw_skmm_kernel`  (denoise split-K GEMM, 90 launches/step)
#                         4.76-4.78 us  ->  4.68-4.74 us
# `nonkdim=32` was raced and is a LOSS everywhere (pfsk 9.18, and it takes the
# two attention kernels 6.9 -> 9.1 / 10.6 -> 12.8), as are `kpack=2` (gfx950
# deprecates it) and `waves_per_eu=1`.  The two attention kernels are NOT
# touched: 16 measured neutral-to-worse for them, and they are not GEMMs.
#
# NUMERICS.  A different MFMA width changes the ORDER in which the K products
# accumulate inside `tl.dot`; the accumulator stays fp32 (COMMANDMENT rule 7)
# and every downstream rounding point is unmoved.  Same class as a library
# GSU/split-K reorder -- verified on the oracle and the 14-episode rollout, not
# asserted bit-exact.
#
# KILL-SWITCH: `FASTWAM_MFMA16=0` restores the Triton default at both sites in
# one move; `FASTWAM_PFSKMI` / `FASTWAM_SKMMMI` (and the `*WPE` / `*KP`
# companions, default off) re-open the individual axes for a future sweep.
# ---------------------------------------------------------------------------
_MFMA16 = os.environ.get("FASTWAM_MFMA16", "1").lower() not in ("0", "false", "no", "off")


def _amd_kw(mi_env: str, mi_default: str = "0",
            wpe_env: str = "", kp_env: str = "") -> dict:
    """Backend options dict for one launch site; empty == Triton's own choice."""
    if not _MFMA16:
        return {}
    kw = {}
    for key, env, dflt in (("matrix_instr_nonkdim", mi_env, mi_default),
                           ("waves_per_eu", wpe_env, "0"),
                           ("kpack", kp_env, "0")):
        if not env:
            continue
        try:
            v = int(os.environ.get(env, dflt))
        except Exception:  # pragma: no cover
            v = 0
        if v > 0:
            kw[key] = v
    return kw


_SKMM_KW = _amd_kw("FASTWAM_SKMMMI", "16", "FASTWAM_SKMMWPE", "FASTWAM_SKMMKP")
_PFSK_KW = _amd_kw("FASTWAM_PFSKMI", "16", "FASTWAM_PFSKWPE", "FASTWAM_PFSKKP")
if _SPLITK:
    try:
        import torch._inductor.config as _ind_cfg
        if int(getattr(_ind_cfg.triton, "num_decompose_k_splits", 0)) == 0:
            _ind_cfg.triton.num_decompose_k_splits = 8
        _ind_cfg.triton.decompose_k_threshold = 3
    except Exception as _exc:  # pragma: no cover
        logger.warning("FastWAM: could not enable decompose_k (%r)", _exc)


# ---------------------------------------------------------------------------
# FORCE EPILOGUE FUSION (delete the standalone gelu dispatch).
#
# The FFN emits addmm -> poi_fused_gelu (in place, 131072 elems, 524 KB in 4.34 us
# = 121 GB/s, i.e. PURE dispatch floor) -> mm.  Inductor is perfectly able to fuse
# that gelu into the addmm template's epilogue; under `max-autotune` it BENCHMARKS
# fused-vs-unfused and picked unfused, because the kernel-level benchmark cannot
# see the ~4.2 us GPU-side dispatch cost of the extra launch.  Turning the
# benchmark off makes inductor fuse whenever it is legal, which is the right call
# on a step whose 45% is sub-6-us floor tax.  Measured, not assumed: A/B'd below.
# ---------------------------------------------------------------------------
_EPIFUSE = "1"
if _EPIFUSE == "1":
    try:
        import torch._inductor.config as _ind_cfg2
        _ind_cfg2.benchmark_epilogue_fusion = False
    except Exception as _exc:  # pragma: no cover
        logger.warning("FastWAM: could not disable benchmark_epilogue_fusion (%r)", _exc)


# ---------------------------------------------------------------------------
# PREFILL LAYER-BOUNDARY FUSION (delete one floor-tax launch per prefill layer).
#
# `_prefill_one_layer` is compiled SEPARATELY per layer invocation, so inductor
# can never see across the layer boundary.  The last kernel of layer i and the
# first kernel of layer i+1 are therefore two dispatches over the SAME [98,3072]
# tensor:
#     triton_poi_fused_add_mul_split_squeeze_view_6   x28.7 @ 4.29 us  (x = x1 + gate_mlp*ffn_out)
#     triton_red_fused_add_mul_native_layer_norm_split_squeeze_0  x28.8 @ 4.44 us
#                                                     (attn_input = LN(x)*(1+scale_msa)+shift_msa)
# Both are AT the measured ~4.3 us gfx950 per-dispatch device floor (they move
# 2.4-3.0 MB, i.e. <1.4 us of traffic at the 2200 GB/s these small kernels
# reach), so the only lever is to stop dispatching one of them.
#
# HOW. Move the *next* layer's `adaln_forward(x, norm1, shift_msa, scale_msa)`
# into the tail of the current layer's compiled graph.  Inductor then emits ONE
# reduction kernel with TWO destinations (x and attn_input) -- the same
# two-destination pattern it already produces for `..._view_4` -- instead of a
# pointwise pass followed by a reduction pass over the same rows.
#
# EXACT BY CONSTRUCTION.  Not a single arithmetic operation changes: the same
# `x + gate_mlp*ffn_out` in the same dtype, stored to the same bf16 buffer, then
# the same LayerNorm/modulate over that stored value.  Only the kernel BOUNDARY
# moves.  Layer 0 still computes its own `attn_input` (a separate dynamo variant
# of the same function), so the head of the chain is unchanged.
#
# PREFILL ONLY.  `_build_expert_attention_io` gains a default-None kwarg that is
# never supplied on the action/denoise arm, so that arm is byte-identical.
# ---------------------------------------------------------------------------
_PFTAIL = os.environ.get("FASTWAM_PFTAIL", "1").lower() not in ("0", "false", "no", "off")


def _linear_splitk(x, lin):
    """`lin(x)` as mm + separate bias, so inductor may pick a split-K decomposition.

    Only split the bias off where `decompose_k` can actually fire (K >= 3*N, the
    gate set above); otherwise keep the `addmm` template so its bias epilogue
    stays fused. Measured: unconditionally splitting cost the video prefill
    ~0.5 ms on its N=K=3072 o-projections, which are not decompose_k candidates.
    """
    if not _SPLITK or lin.bias is None:
        return lin(x)
    n, k = int(lin.out_features), int(lin.in_features)
    if n > 1024 or k < 3 * n:
        return lin(x)
    if _skmm_ok(x, lin.weight):
        return torch.ops.fastwam.skmm.default(x, lin.weight) + lin.bias
    return F.linear(x, lin.weight) + lin.bias


# ---------------------------------------------------------------------------
# HIP/CUDA graph capture-and-replay of the two hot dispatch regions.
#
# WHY. `infer_action` issues 2073 kernels (1 step) / 9570 kernels (10 steps) and
# `gpu_busy/wall` is only 0.705 / 0.838 -- i.e. 29.5% / 16.2% of the wall is
# bubble *between* dispatches. The denoise step replays the SAME ~830-kernel
# sequence 10x with only tensor VALUES changing, and the video prefill replays
# the same ~810-kernel sequence once per episode. Re-issuing them from a captured
# graph removes the per-dispatch submit latency.
#
# WHY THIS IS BIT-EXACT BY CONSTRUCTION. A graph replay re-issues the *identical*
# kernels (the same inductor/aten binaries, same grids, same argument values)
# against the same static buffers. Nothing about the arithmetic changes; only the
# host-side submission path does. This is the one lever on this stack that cannot
# perturb a single output bit.
#
# PER-EPISODE INVALIDATION (COMMANDMENT RULE 5). The eval server reuses ONE MoT
# across episodes and calls `prefill_video_cache` with a NEW video each episode.
# `prefill_video_cache` therefore bumps `self._fw_episode`, and the action graph
# refreshes its episode-scoped static inputs (video K/V, text context, masks)
# whenever that counter moves. Pointer/`_version` identity is NOT used for the
# episode-scoped inputs on purpose: the caching allocator hands out the SAME
# addresses to the next episode's cache, so a data_ptr memo would silently go
# stale. The counter is a positive signal, not an inference.
#
# Env flag FASTWAM_GRAPH (bitmask, default 3): 1 = action denoise step,
# 2 = video prefill, 0 = off (exact original path).
# ---------------------------------------------------------------------------

def _fw_graph_flags() -> int:
    try:
        return int(os.environ.get("FASTWAM_GRAPH", "3"))
    except Exception:
        return 3


def _mk_slot(t: torch.Tensor):
    """Allocate a static buffer whose shape AND strides match `t` exactly.

    Strides matter: SDPA / inductor dispatch on them, so materialising a
    stride-0 broadcast mask into a dense plane (or a sliced mask into a
    contiguous one) could select a different kernel and break bit-exactness.
    Returns (view, base, squeezed_shape_or_None); `view` is what the graph reads.
    """
    shape = tuple(t.shape)
    strides = tuple(t.stride())
    if any(s == 0 for s in strides):
        sq = tuple(1 if st == 0 else sz for sz, st in zip(shape, strides))
        base = torch.empty(sq, dtype=t.dtype, device=t.device)
        view = base.expand(shape)
        if tuple(view.stride()) != strides:
            return None
        return (view, base, sq)
    base = torch.empty_strided(shape, strides, dtype=t.dtype, device=t.device)
    return (base, base, None)


def _fill_slot(slot, t: torch.Tensor) -> None:
    _view, base, sq = slot
    if sq is None:
        base.copy_(t)
    else:
        base.copy_(t.as_strided(sq, tuple(t.stride())))


def _sig(t):
    if t is None:
        return None
    return (tuple(t.shape), tuple(t.stride()), t.dtype, t.device.type)


_FOREACH_COPY = getattr(torch, "_foreach_copy_", None)


def _graphs_available(t: torch.Tensor) -> bool:
    return (
        t is not None
        and t.is_cuda
        and torch.cuda.is_available()
        and not torch.cuda.is_current_stream_capturing()
    )


# ---------------------------------------------------------------------------
# HORIZONTAL PROJECTION FUSION (grid fill).
#
# The Action-DiT projections are M=32 x K=1024 x N=3072.  With a 128-wide N tile
# that is 24 workgroups on a 256-CU part (~9% fill), so each GEMM streams its
# weights at ~390 GB/s -- ~5% of the ~8 TB/s roof -- purely because the grid is
# starved.  Concatenating the weight ROWS of projections that share an input is
# mathematically EXACT (a block-row GEMM is the same arithmetic as the separate
# GEMMs) and multiplies the tile count:
#
#   self_attn  q|k|v  : 1024 -> 9216   (24 -> 72 tiles, 3 launches -> 1)
#   cross_attn k|v    : 1024 -> 6144   (24 -> 48 tiles, 2 launches -> 1)
#
# The fused buffer is built ONCE per expert, lazily, during warmup (before any
# torch.compile trace and long before graph capture), and the ORIGINAL Linear
# weights are re-pointed at contiguous row-slices of it, so this costs no extra
# device memory and leaves every other consumer of `.q/.k/.v` working.
#
# It is keyed on nothing episode-scoped: model weights are frozen for the life of
# the process, so a one-shot build is not a stale-cache hazard (COMMANDMENT rule
# 5 is about episode-scoped state; this is not that).
# ---------------------------------------------------------------------------

_FUSE_PROJ = os.environ.get("FASTWAM_FUSE_PROJ", "1").lower() not in ("0", "false", "no", "off")


@torch.no_grad()
def _fuse_linears_inplace(mod, names, buf_name):
    """Concatenate `mod.<names>` weight/bias rows into one buffer on `mod`.

    The individual Linears are re-pointed at row-slices of the fused buffer, so
    memory is unchanged and `mod.q(x)` still works.
    """
    lins = [getattr(mod, n, None) for n in names]
    if any(l is None or not isinstance(l, nn.Linear) for l in lins):
        return False
    outs = [l.out_features for l in lins]
    if len(set(outs)) != 1:
        return False
    if len(set(l.in_features for l in lins)) != 1:
        return False
    w = torch.cat([l.weight.detach() for l in lins], dim=0).contiguous()
    have_bias = all(l.bias is not None for l in lins)
    b = torch.cat([l.bias.detach() for l in lins], dim=0).contiguous() if have_bias else None
    if not have_bias and any(l.bias is not None for l in lins):
        return False
    h = outs[0]
    for i, l in enumerate(lins):
        l.weight.data = w[i * h:(i + 1) * h]
        if b is not None:
            l.bias.data = b[i * h:(i + 1) * h]
    mod.register_buffer(buf_name + "_w", w, persistent=False)
    if b is not None:
        mod.register_buffer(buf_name + "_b", b, persistent=False)
    else:
        setattr(mod, buf_name + "_b", None)
    setattr(mod, buf_name + "_h", h)
    return True


# Cross-attention K/V reads the SAME `context` tensor at every layer, so all
# `n_layers` k|v projections can be issued as ONE 1024 -> n_layers*2*H GEMM.
# For the action expert that is 1024 -> 184320: 1440 N-tiles instead of 30
# separate 48-tile launches, i.e. the whole 377 MB weight slab streams under a
# grid that actually covers the 256 CUs.  Still recomputed EVERY denoise step --
# this is a fill lever, not a cross-step memo (that arm is a known dead end).
_FUSE_XKV = os.environ.get("FASTWAM_FUSE_XKV", "1").lower() not in ("0", "false", "no", "off")


@torch.no_grad()
def _build_ctx_kv_all(expert) -> bool:
    """Concatenate every block's cross_attn k|v weight into ONE [L*2H, D] buffer.

    Each block's own `_fw_kv_w` (and each Linear's `.weight`) is re-pointed at a
    row-slice of it, so the fused-per-layer path and the plain `.k(x)` path keep
    working and no extra device memory is used.
    """
    blocks = list(getattr(expert, "blocks", []) or [])
    cas = [getattr(b, "cross_attn", None) for b in blocks]
    if not cas or any(c is None for c in cas):
        return False
    lins = []
    for c in cas:
        k, v = getattr(c, "k", None), getattr(c, "v", None)
        if not isinstance(k, nn.Linear) or not isinstance(v, nn.Linear):
            return False
        lins.extend((k, v))
    if len(set(l.out_features for l in lins)) != 1 or len(set(l.in_features for l in lins)) != 1:
        return False
    have_bias = all(l.bias is not None for l in lins)
    if not have_bias and any(l.bias is not None for l in lins):
        return False
    h = lins[0].out_features
    w = torch.cat([l.weight.detach() for l in lins], dim=0).contiguous()
    b = torch.cat([l.bias.detach() for l in lins], dim=0).contiguous() if have_bias else None
    for i, l in enumerate(lins):
        l.weight.data = w[i * h:(i + 1) * h]
        if b is not None:
            l.bias.data = b[i * h:(i + 1) * h]
    for j, c in enumerate(cas):
        c.register_buffer("_fw_kv_w", w[j * 2 * h:(j + 1) * 2 * h], persistent=False)
        if b is not None:
            c.register_buffer("_fw_kv_b", b[j * 2 * h:(j + 1) * 2 * h], persistent=False)
        else:
            c._fw_kv_b = None
        c._fw_kv_h = h
    expert.register_buffer("_fw_xkv_w", w, persistent=False)
    if b is not None:
        expert.register_buffer("_fw_xkv_b", b, persistent=False)
    else:
        expert._fw_xkv_b = None
    expert._fw_xkv_h = h
    return True


@torch.no_grad()
def _ensure_fused_proj(expert) -> None:
    """One-shot: build fused q|k|v and cross-attn k|v weights for every block."""
    if not _FUSE_PROJ or getattr(expert, "_fw_fused_proj", False):
        return
    blocks = getattr(expert, "blocks", None)
    if blocks is None:
        expert._fw_fused_proj = True
        return
    n_sa = n_ca = 0
    for block in blocks:
        sa = getattr(block, "self_attn", None)
        if sa is not None and not hasattr(sa, "_fw_qkv_w"):
            n_sa += int(_fuse_linears_inplace(sa, ("q", "k", "v"), "_fw_qkv"))
    if _qkvpost_enabled():
        _build_qk_norm(expert)
    mega = _FUSE_XKV and _build_ctx_kv_all(expert)
    if mega and _XKNORM:
        _build_xk_norm(expert)
    if not mega:
        for block in blocks:
            ca = getattr(block, "cross_attn", None)
            if ca is not None and not hasattr(ca, "_fw_kv_w"):
                n_ca += int(_fuse_linears_inplace(ca, ("k", "v"), "_fw_kv"))
    expert._fw_fused_proj = True
    logger.info("FastWAM: fused projections built (self_attn qkv=%d, cross_attn kv=%d, mega_xkv=%s).",
                n_sa, n_ca, bool(mega))


# ---------------------------------------------------------------------------
# JOINT K/V BUFFERS -- delete the per-step re-copy of the video prefix.
#
# `torch.cat([k_video, k_action])` was 136 launches/denoise-step (~0.62 ms) and
# is pure dispatch floor: 0.8 MB moved in 4.5 us = 178 GB/s. Two thirds of what
# it copies is the video prefix, which is IDENTICAL at every step of an episode.
# Instead hold one [B, Sv+Sa, D] buffer per layer per tensor, fill the prefix
# once per episode, and let the denoise step write only the 32-row tail (which
# inductor reinplaces into the buffer and largely fuses into the producing
# rope/GEMM epilogue).
#
# COMMANDMENT RULE 5: the prefix is refreshed strictly BY `self._fw_episode`
# COUNTER, never by data_ptr/_version -- the caching allocator reuses the same
# addresses for the next episode's cache.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CROSS-CALL MEMO OF THE EPISODE-SCOPED PROLOGUE  (round 4, host_runtime).
#
# `_ctx_kv_all` -- the horizontally-fused all-30-layer cross-attention K|V
# projection -- runs TWICE per `infer_action`: once for the VIDEO expert inside
# the captured video-prefill graph (0.232 ms/call, MT256x144x64) and once for
# the ACTION expert in `_refresh_prologue_slots` (0.091 ms/call, un-regioned).
# Together 0.323 ms of EVERY call, on both scored cases.  It already moves its
# weight slab at the LLC bandwidth ceiling, so it cannot be made faster -- only
# skipped.
#
# It is a pure function of the language `context` (+ the two masks), which does
# not change while those inputs do not change.  The existing `_fw_episode`
# counter cannot express that: `prefill_video_cache` bumps it on EVERY call, so
# it is really a call counter and the memo can never hit across calls.
#
# COMMANDMENT RULE 5, and why this is not the forbidden identity memo:
#   * The key is CONTENT, compared element-wise, never `data_ptr`/`_version`.
#     The caching allocator hands the next episode the same addresses, so an
#     identity memo goes stale invisibly -- that is exactly the bug this shape
#     avoids.
#   * We hold our OWN private CLONE of every key tensor, so nothing outside the
#     memo can mutate the key under us between calls.
#   * The `_fw_episode` counter stays as the OUTER gate: the content compare is
#     only ever consulted when the counter says the episode-scoped inputs may
#     have moved.  Content + counter is strictly stronger than counter alone.
# The compare costs one small kernel per key plus ONE host sync per site
# (measured against the 0.323 ms it skips).
#
# Kill-switch: FASTWAM_CTXMEMO=0 restores the per-call recomputation exactly.
# ---------------------------------------------------------------------------
_CTXMEMO = os.environ.get("FASTWAM_CTXMEMO", "1").lower() not in ("0", "false", "no", "off")


def _pack_bytes(tensors):
    """Flat contiguous uint8 view of every tensor, concatenated -- one kernel."""
    parts = []
    for t in tensors:
        u = t.detach()
        if not u.is_contiguous():
            u = u.contiguous()
        parts.append(u.reshape(-1).view(torch.uint8))
    if len(parts) == 1:
        return parts[0].clone()
    return torch.cat(parts)
# FASTWAM_PFHOIST=0 additionally restores the video-prefill prologue to INSIDE
# the captured graph body (the pre-round-4 structure), so the hoist and the memo
# can be A/B'd independently.
_PFHOIST = os.environ.get("FASTWAM_PFHOIST", "1").lower() not in ("0", "false", "no", "off")

_KVBUF = os.environ.get("FASTWAM_KVBUF", "1").lower() not in ("0", "false", "no", "off")
_KVSLAB = os.environ.get("FASTWAM_KVSLAB", "1").lower() not in ("0", "false", "no", "off")


def _ctx_kv_all(expert, context):
    """One GEMM producing every layer's cross-attn K and V. Returns [B, L, n*2H] or None."""
    w = getattr(expert, "_fw_xkv_w", None)
    if w is None or context is None:
        return None
    return F.linear(context, w, getattr(expert, "_fw_xkv_b", None))


# ---------------------------------------------------------------------------
# LOOP-INVARIANT ATTENTION INPUTS (dispatch-count lever, round 2).
#
# Two things inside the 30-layer denoise loop are recomputed per LAYER although
# they do not depend on the layer's tokens at all. Both sit squarely on the
# ~4.5 us per-dispatch floor, so deleting the dispatch IS the whole win:
#
# 1. THE ATTENTION MASK. `F.scaled_dot_product_attention` takes a BOOL mask, and
#    inductor lowers the bool->additive-bias conversion (scalar_tensor + where +
#    constant_pad_nd to an 8-aligned last-dim stride) once per SDPA CALL. The
#    generated code shows `triton_poi_..._where_5/11/17` producing a fresh
#    (1,1,S,L)/(...,136,1) bias buffer for every layer -- 31 launches/step for a
#    tensor that is identical at every layer. Building the bias ONCE per call,
#    already padded so `stride(-2) % 8 == 0`, makes inductor reuse the single
#    buffer. Bit-identical: same 0 / -inf values, same padded strides.
#
# 2. THE CROSS-ATTENTION K RMSNORM. `_ctx_kv_all` already produces every layer's
#    cross K|V in ONE mega GEMM, but each layer then RMSNorms its own 129x3072
#    K slice in its own kernel (30 launches/step, ~4.9 us each for ~1.6 MB of
#    traffic = ~330 GB/s, i.e. pure floor). RMSNorm is row-wise over h, so all
#    30 layers normalize in a single [n*B*L, h] pass. Exact by construction: the
#    same fp32 rsqrt(mean(x^2)+eps), the same bf16 round, the same per-layer
#    weight -- only the launch geometry changes.
#
# Both are pure functions of loop-invariant inputs; no state is cached on `self`,
# so COMMANDMENT rule 5 (per-episode invalidation) has nothing to bite on.
# ---------------------------------------------------------------------------
_MASKBIAS = os.environ.get("FASTWAM_MASKBIAS", "1").lower() not in ("0", "false", "no", "off")
_XKNORM = os.environ.get("FASTWAM_XKNORM", "1").lower() not in ("0", "false", "no", "off")

# MEASURED NEGATIVE (round 1, engineer_1 / memory).  The direction's second lever
# was "give `_fw_attn_kernel` the K/V layout it wants, from the PRODUCER side".
# It was built and measured: `_ctx_k_all` restacked to [2, n, B, L, h] so V comes
# out of a 0.79 MB per-layer slab (key-row stride 3072 elements) instead of the
# 47.5 MB joint K|V buffer (key-row stride 184320 elements = 360 KB).  The
# attention kernel is UNMOVED by it -- 9.87 vs 9.85 us on the fused cross path,
# 8.00 vs 8.06 us on the self path, both inside run-to-run noise, and the prefill
# is slightly worse.  This kernel is latency-bound at 48 CTAs, not
# request-locality-bound: it moves ~1.8 MB in 8 us (~220 GB/s), so the 256 B
# scattered V rows are already fully hidden.  Reverted; do not re-fund.



def _attn_bias(mask, dtype):
    """bool attention mask -> additive bias with an 8-aligned last-dim stride."""
    if not _MASKBIAS or mask is None or mask.dtype != torch.bool:
        return mask
    L = int(mask.shape[-1])
    pad = (-L) % 8
    b = torch.where(mask, 0.0, float("-inf")).to(dtype)
    if pad:
        b = F.pad(b, (0, pad))[..., :L]
    return b


@torch.no_grad()
def _build_xk_norm(expert) -> bool:
    """Stack every block's cross_attn.norm_k weight into one [n, h] buffer."""
    blocks = list(getattr(expert, "blocks", []) or [])
    ws = []
    eps = None
    for b in blocks:
        ca = getattr(b, "cross_attn", None)
        nk = getattr(ca, "norm_k", None) if ca is not None else None
        w = getattr(nk, "weight", None)
        if w is None or w.dim() != 1:
            return False
        e = float(getattr(nk, "eps", 1e-6))
        if eps is None:
            eps = e
        elif e != eps:
            return False
        ws.append(w.detach())
    if not ws:
        return False
    expert.register_buffer("_fw_xk_nw", torch.stack(ws, 0).contiguous(), persistent=False)
    expert._fw_xk_eps = eps
    return True


def _ctx_k_all(expert, xkv):
    """Every layer's cross-attn K, RMSNormed, in ONE kernel.

    Returns [n, B, L, h] contiguous, so `out[i]` has exactly the shape/strides the
    per-layer `ca.norm_k(kv[..., :h])` produced, or None when unavailable.
    """
    w = getattr(expert, "_fw_xk_nw", None)
    if not _XKNORM or w is None or xkv is None:
        return None
    B, L, tot = xkv.shape
    n, h = int(w.shape[0]), int(w.shape[1])
    if tot != n * 2 * h:
        return None
    kf = xkv.view(B, L, n, 2, h)[:, :, :, 0, :].float()
    kf = kf * torch.rsqrt(kf.pow(2).mean(-1, keepdim=True) + expert._fw_xk_eps)
    k = kf.to(xkv.dtype) * w
    return k.permute(2, 0, 1, 3).contiguous()


# ---------------------------------------------------------------------------
# REAL-fp32 RoPE, `torch.stack` formulation.
#
# Bit-identical to the shipped flip+sign-vector arm:
#     a*cos + b*(sin*-1)  ==  a*cos - b*sin      (IEEE: x + (-y) == x - y, and
#     b*cos + a*(sin*+1)  ==  b*cos + a*sin       negation is exact)
# but it writes the two halves as two explicit expressions instead of one
# expression over a flipped [.., 2] axis, which lets inductor index both operands
# with a static offset rather than a reversed one. Round 1 measured this arm at
# 711 kernels/step vs 741 for the flip arm and the better interleaved median; it
# shipped the other arm only because that one already had a passing rollout.
#
# Local to mot.py on purpose: `wan_video_dit.rope_apply` is another engineer's
# lane this round, and the complex/kill-switch path still routes through it.
# ---------------------------------------------------------------------------
_ROPE_STACK = os.environ.get("FASTWAM_ROPE_STACK", "0").lower() not in ("0", "false", "no", "off")


def _rope_apply(x, freqs, num_heads):
    if not _ROPE_STACK or freqs.is_complex():
        return rope_apply(x, freqs, num_heads)
    B, S, ND = x.shape
    Dh = (ND // num_heads) // 2
    cos = freqs[..., :Dh]
    sin = freqs[..., Dh:]
    xr = x.reshape(B, S, num_heads, Dh, 2).float()
    x0 = xr[..., 0]
    x1 = xr[..., 1]
    out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    return out.reshape(B, S, ND).to(x.dtype)


# ---------------------------------------------------------------------------
# NATIVE-LAYOUT SHORT ATTENTION (collapse the 59 clone_permute copies/step).
#
# `fastwam::short_attn` returns [B, H, Sq, D] contiguous, but every consumer here
# wants [B, Sq, H*D] for the o-projection GEMM.  `rearrange("b n s d -> b s (n d)")`
# on a contiguous [B,H,Sq,D] is NOT a view, so inductor materialises it:
#     triton_poi_fused_clone_permute_5   59 launches/step @ 4.44 us = 0.262 ms/step
# i.e. a pure layout copy that exists only because the producer picked the wrong
# output layout.
#
# Round 2 tried to remove it from the CONSUMER side (feed the fused kernel the flat
# 3-D layout) and lost: that strips the permute epilogue off the fused-QKV GEMM
# template and inductor re-autotunes that producer 11.5 -> 33 us.  So instead give
# the ATTENTION an explicit output layout: same [B,H,Sq,D] *inputs* (the QKV epilogue
# is untouched), but `o` is allocated as [B, Sq, H*D] and the kernel is handed
# oh = D, os_ = H*D so it stores straight into that order.
#
# EXACT BY CONSTRUCTION. The identical `_fw_attn_kernel` runs with the identical
# grid / block / mask / scale and the identical loads; only the STORE ADDRESS
# MAPPING changes (h*D + s*H*D + d instead of h*Sq*D + s*D + d). Each store is
# still a 128-element contiguous run, so coalescing is unchanged too. Not a single
# arithmetic operation differs.
#
# Scope: entered ONLY where the current code would take the 4-D rearranged route
# (Sq < _FA_FLAT_MIN_SQ, i.e. the Sq=32 action denoise step). `video_prefill`
# (Sq=98) keeps the flat path it already had, byte for byte.
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl
    from .wan_video_dit import (
        _rope_sign as _rope_sign,
        _fw_attn_kernel as _FW_ATTN_KERNEL,
        _fa_eligible as _fa_eligible,
        _fa_mask2d as _fa_mask2d,
        _FA_FLAT_MIN_SQ as _FA_FLAT_MIN_SQ,
        _fa_key_tile as _fa_key_tile,
        _fa_block_m as _fa_block_m,
        _fa_nw as _fa_nw,
        _FA_PRELOAD as _FA_PRELOAD,
        _FA_FULLMAIN as _FA_FULLMAIN,
    )
    _FA_BSD_OK = _FW_ATTN_KERNEL is not None
except Exception:  # pragma: no cover
    _FW_ATTN_KERNEL = None
    _FA_BSD_OK = False
    _FA_FLAT_MIN_SQ = 64

# In-file literal: run_local.sh forwards only GATE_*/FWE_*/ORACLE_*, so this is the
# switch that is actually flipped between A/B arms.
_FA_BSD = "1"


if _FA_BSD_OK:

    @torch.library.custom_op("fastwam::short_attn_bsd", mutates_args=())
    def _fw_short_attn_bsd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           mask: Optional[torch.Tensor], num_heads: int) -> torch.Tensor:
        """Same kernel as `fastwam::short_attn`, but emits [B, Sq, H*D]."""
        B, H, Sq, D = q.shape
        Sk = k.shape[2]
        o = torch.empty((B, Sq, H * D), dtype=q.dtype, device=q.device)
        # INTEGRATION (round 3): the key-tile table is shared with
        # `fastwam::short_attn` -- both ops run the SAME `_fw_attn_kernel`, so the
        # native-layout op must not keep its own `128 if Sk<=128 else 256` rule or
        # every denoise-step call (Sk = 129/130) would silently take the wasteful
        # 256-wide tile while the prefill's calls take the split one. Same table,
        # same bit-exact split, only the STORE mapping differs.
        BLOCK_N, TAIL_N, nw = _fa_key_tile(Sk)
        BLOCK_M = _fa_block_m(Sq)
        nw = _fa_nw(BLOCK_M, nw)
        grid = ((Sq + BLOCK_M - 1) // BLOCK_M, H)
        _FW_ATTN_KERNEL[grid](
            q, k, v, mask if mask is not None else q, o, Sq, Sk,
            q.stride(1), q.stride(2), k.stride(1), k.stride(2),
            v.stride(1), v.stride(2), D, H * D,
            0 if mask is None else mask.stride(0),
            D ** -0.5, D, BLOCK_M, BLOCK_N, mask is not None,
            mask is not None and mask.dtype != torch.bool,
            TAIL_N=TAIL_N, PRELOAD=_FA_PRELOAD,
            # ROUND-3: this site never passed the two shape-fact constexprs the
            # `fastwam::short_attn` site has passed since round 2, so the DENOISE
            # self-attention -- the only caller here -- was still emitting a
            # predicate on every K/V main-tile load and on the Q load/O store that
            # is provably always true (Sk=130 >= BLOCK_N=128, and BLOCK_M | 32).
            # Dropping an always-true predicate is bit-exact by construction.
            FULL_MAIN=int(_FA_FULLMAIN and Sk >= BLOCK_N),
            EXACT_M=int(_FA_FULLMAIN and Sq % BLOCK_M == 0),
            num_warps=nw, num_stages=1,
        )
        return o

    @_fw_short_attn_bsd.register_fake
    def _(q, k, v, mask, num_heads):
        B, H, Sq, D = q.shape
        return q.new_empty((B, Sq, H * D))


def _bsd_enabled() -> bool:
    return _FA_BSD_OK and _FA_BSD.lower() not in ("0", "false", "no", "off")


def _attn3(q, k, v, num_heads, ctx_mask):
    """flash_attention over [B, S, H*D] operands, native-layout when possible."""
    if _bsd_enabled() and q.dim() == 3 and q.shape[1] < _FA_FLAT_MIN_SQ \
            and _fa_eligible(q, k, v, num_heads):
        B, Sq, HD = q.shape
        H = int(num_heads)
        D = HD // H
        m = _fa_mask2d(ctx_mask, Sq, k.shape[1])
        if m is not False:
            q4 = q.unflatten(-1, (H, D)).permute(0, 2, 1, 3)
            k4 = k.unflatten(-1, (H, D)).permute(0, 2, 1, 3)
            v4 = v.unflatten(-1, (H, D)).permute(0, 2, 1, 3)
            return torch.ops.fastwam.short_attn_bsd.default(q4, k4, v4, m, H)
    return flash_attention(q=q, k=k, v=v, num_heads=num_heads, ctx_mask=ctx_mask)


# ---------------------------------------------------------------------------
# ONE KERNEL FOR THE WHOLE POST-QKV EPILOGUE (collapse 90 dispatches/step -> 30).
#
# Inductor emits THREE kernels between the fused-QKV GEMM and the attention, all
# reading the SAME [32, 9216] GEMM output and all floor-bound:
#   triton_poi_fused_copy_slice_view_2                30x @ 4.37 us  (v -> joint V tail)
#   triton_red_..._flip_mean_mul_pow_rsqrt_slice_..._3  30x @ 8.42 us  (k RMSNorm+RoPE -> joint K tail)
#   triton_red_..._flip_mean_mul_permute_..._4        30x @ 4.41 us  (q RMSNorm+RoPE -> q buffer)
#                                                     = 0.516 ms/step
# It cannot merge them: they have THREE DIFFERENT DESTINATION BUFFERS, and the
# round-2 measurement (single stack() store -> 3 kernels, not 1) is the proof that
# inductor will not emit one kernel with two destinations on this stack. A
# hand-written triton.jit kernel behind a `torch.library.custom_op` has no such
# restriction, which is exactly how round 2 escaped SDPA.
#
# TRANSCRIBED, NOT REDERIVED. The body below is a line-by-line transcription of
# inductor's own generated code for kernels _3/_4 (same fp32 chain with no
# intermediate bf16 round -- inductor keeps bf16 arithmetic in fp32 registers and
# rounds only at the store; same `rsqrt(sum/N + eps)`; same flip index
# `r + 1 - 2*(r%2)`; same `y*cos + y_flip*(sin*sgn)` association, with `sin*sgn`
# formed FIRST exactly as inductor does).
#
# THE ONE NON-IDENTITY is the reduction shape: inductor accumulates sum(x*x) in a
# LOOPED R0_BLOCK reduction, this kernel does a single `tl.sum` over a 4096-wide
# (3072 + masked zeros) block. Both are fp32; the summation TREE differs, so the
# fp32 mean can differ in its last ulp (~1e-7 relative), four orders of magnitude
# below the bf16 store quantum (~4e-3). NOT bit-exact -- reported as such and
# defended on the 14-episode LIBERO rollout, not on the oracle alone.
# ---------------------------------------------------------------------------

# In-file literal: run_local.sh forwards only GATE_*/FWE_*/ORACLE_*, so this is
# the switch that is actually flipped between A/B arms.
_QKVPOST = "1"
# r2_d0: latency form of the same epilogue.  0 = the original body at nw=8.
_QKVPOSTF = os.environ.get("FASTWAM_QKVPOSTF", "1").lower() not in (
    "0", "false", "no", "off")
_QKVPOST_NW = int(os.environ.get("FASTWAM_QKVPOST_NW", "4"))

if _FA_BSD_OK:

    @triton.jit
    def _fw_qkvpost_kernel(QKV, WQK, FRQ, SGN, QO, KO, VO, kv_off, eps,
                           N: tl.constexpr, QN: tl.constexpr,
                           HEAD: tl.constexpr, DH: tl.constexpr,
                           BLOCK: tl.constexpr):
        s = tl.program_id(0)
        w = tl.program_id(1)
        r = tl.arange(0, BLOCK)
        m = r < N
        src = QKV + s * QN + w * N
        if w == 2:
            xv = tl.load(src + r, mask=m, other=0.0)
            tl.store(VO + kv_off + s * N + r, xv, mask=m)
        else:
            x = tl.load(src + r, mask=m, other=0.0).to(tl.float32)
            acc = tl.sum(x * x, 0)
            rstd = tl.rsqrt(acc / N + eps)
            fl = r + 1 - 2 * (r % 2)
            xf = tl.load(src + fl, mask=m, other=0.0).to(tl.float32)
            wt = tl.load(WQK + w * N + r, mask=m, other=0.0).to(tl.float32)
            wf = tl.load(WQK + w * N + fl, mask=m, other=0.0).to(tl.float32)
            j = (r % HEAD) // 2
            c = tl.load(FRQ + s * HEAD + j, mask=m, other=0.0)
            sn = tl.load(FRQ + s * HEAD + DH + j, mask=m, other=0.0)
            sg = tl.load(SGN + (r % 2), mask=m, other=0.0)
            y = (x * rstd) * wt
            yf = (xf * rstd) * wf
            o = y * c + yf * (sn * sg)
            if w == 0:
                tl.store(QO + s * N + r, o, mask=m)
            else:
                tl.store(KO + kv_off + s * N + r, o, mask=m)

    # -----------------------------------------------------------------
    # ROUND-2 (r2_d0) LATENCY FORM OF THE SAME EPILOGUE.
    #
    # The kernel above is LATENCY-bound, not bandwidth-bound: 96 CTAs moving
    # 1.2 MB in 4.56 us.  Three separate mechanisms serialise it, and all three
    # are removable without touching a single arithmetic operation:
    #
    #  1. SIX GLOBAL LOADS ISSUED *AFTER* THE CROSS-LANE REDUCTION.  `xf`, `wt`,
    #     `wf`, `c`, `sn` and `sg` do not depend on `acc`, but they are written
    #     after `tl.sum`, and the LDS barriers the reduction emits stop the
    #     backend from hoisting them -- so their round-trip is fully exposed
    #     behind the reduction instead of overlapping it.  This is exactly the
    #     r1_d0 `_fw_skpost` hoist, which the round-1 ledger flagged as
    #     transferring verbatim to the kernels that load their small vectors
    #     after a reduction.  Every load is moved ABOVE `tl.sum`.
    #
    #  2. `xf` IS A SECOND FULL PASS OVER `src`.  The RoPE partner of lane `r`
    #     is `r ^ 1`, i.e. the flip is a pure PAIR SWAP inside the block, so it
    #     is a register `split`/`join` on the value already loaded.  BLOCK is a
    #     multiple of 2 and the masked tail is a whole number of pairs, so the
    #     swap can never cross the mask edge -- the values are identical to what
    #     the second load returned, bit for bit.  One whole global stream gone.
    #     `wf` is the SAME pair swap applied to the norm weight `wt`, so the
    #     second WQK gather disappears the same way (a further -0.28 us).
    #
    #  3. `c` / `sn` ARE A 3072-WIDE GATHER OF A 64-ENTRY TABLE.  `j` only takes
    #     HEAD/2 distinct values, so the two RoPE loads are replaced by ONE
    #     contiguous DH-wide load each, doubled with `join` (that is exactly
    #     `j = (r % HEAD) // 2`) and broadcast over the BLOCK/HEAD head slices.
    #     `sg` likewise loads the 2-entry sign table as two scalars and selects
    #     with `tl.where` instead of gathering it BLOCK-wide.
    #
    # THE BODY IS BIT-EXACT AT EQUAL num_warps: same values in the same order
    # into the same fp32 expression, only the way they REACH the registers
    # changes.  A standalone A/B against the kernel above over both real shapes
    # (S=32 grid (S,3); S=98 grid (S,2); N=3072, HEAD=128) reports
    # max|d| = 0.0 on all three destinations at num_warps=8.
    #
    # THE ONE NON-IDENTITY is the SHIPPED num_warps (8 -> 4): `tl.sum` over the
    # 4096-wide block then builds a different fp32 summation TREE.  Same class
    # as the `qkv_post` comment above -- ~1e-7 relative, four orders below the
    # bf16 store quantum.  Oracle: 9.4e-03 mean|d| with this form against
    # 1.20e-02 for the OFF arm and a 2.283e-02 gate; 14-episode LIBERO rollout
    # SR=1.0000 on attempt 1.
    #
    # Chained-graph microbenchmark, 30 layers, real shapes:
    #                                        S=32      S=98
    #     current form            nw=8 :     4.11      4.25 us
    #     + hoist only            nw=8 :     3.86        -
    #     + xf swap + RoPE table  nw=4 :     2.54      2.65
    #     + wf swap (this form)   nw=4 :     2.26      2.32
    # i.e. -1.86 us/launch, and this kernel runs 30x per denoise step AND 30x
    # per prefill call (`fastwam::qk_post` shares it).  FULL_BENCHMARK:
    # 7.7720/24.3246/13.7496 -> 7.6353/23.5946/13.4587 (geomean 1.0244x).
    #
    # ENVELOPE: needs HEAD a power of two, BLOCK % HEAD == 0 and N even --
    # `_qkvpostf_ok` checks all three and falls back to the kernel above
    # otherwise.  KILL SWITCH: FASTWAM_QKVPOSTF=0 restores it, and
    # FASTWAM_QKVPOST_NW re-opens num_warps.
    # -----------------------------------------------------------------
    @triton.jit
    def _fw_qkvpostf_kernel(QKV, WQK, FRQ, SGN, QO, KO, VO, kv_off, eps,
                            N: tl.constexpr, QN: tl.constexpr,
                            HEAD: tl.constexpr, DH: tl.constexpr,
                            BLOCK: tl.constexpr):
        s = tl.program_id(0)
        w = tl.program_id(1)
        r = tl.arange(0, BLOCK)
        m = r < N
        src = QKV + s * QN + w * N
        if w == 2:
            xv = tl.load(src + r, mask=m, other=0.0)
            tl.store(VO + kv_off + s * N + r, xv, mask=m)
        else:
            # ---- every global load issued BEFORE the reduction --------------
            fl = r + 1 - 2 * (r % 2)
            x = tl.load(src + r, mask=m, other=0.0).to(tl.float32)
            wt = tl.load(WQK + w * N + r, mask=m, other=0.0).to(tl.float32)
            ch = tl.load(FRQ + s * HEAD + tl.arange(0, DH))
            sh = tl.load(FRQ + s * HEAD + DH + tl.arange(0, DH))
            sg0 = tl.load(SGN + 0)
            sg1 = tl.load(SGN + 1)
            # ---- reduction ---------------------------------------------------
            acc = tl.sum(x * x, 0)
            rstd = tl.rsqrt(acc / N + eps)
            # ---- pair swap + RoPE table expansion, all in registers ----------
            xe, xo = tl.split(tl.reshape(x, (BLOCK // 2, 2)))
            xf = tl.reshape(tl.join(xo, xe), (BLOCK,))
            we, wo = tl.split(tl.reshape(wt, (BLOCK // 2, 2)))
            wf = tl.reshape(tl.join(wo, we), (BLOCK,))
            c1 = tl.reshape(tl.join(ch, ch), (HEAD,))
            s1 = tl.reshape(tl.join(sh, sh), (HEAD,))
            c = tl.reshape(tl.broadcast_to(c1[None, :], (BLOCK // HEAD, HEAD)),
                           (BLOCK,))
            sn = tl.reshape(tl.broadcast_to(s1[None, :], (BLOCK // HEAD, HEAD)),
                            (BLOCK,))
            sg = tl.where(r % 2 == 0, sg0, sg1)
            y = (x * rstd) * wt
            yf = (xf * rstd) * wf
            o = y * c + yf * (sn * sg)
            if w == 0:
                tl.store(QO + s * N + r, o, mask=m)
            else:
                tl.store(KO + kv_off + s * N + r, o, mask=m)

    def _qkvpostf_ok(n: int, head: int, block: int) -> bool:
        """Envelope for the latency form: pow2 HEAD, HEAD | BLOCK, N even."""
        if not _QKVPOSTF:
            return False
        if head < 2 or head % 2 or (head & (head - 1)):
            return False
        if block % head or n % 2:
            return False
        return True

    def _qkvpost_launch(grid, qkv, wqk, freqs, sgn, q, kbuf, vbuf,
                        kv_off, eps, n, qn, head, block):
        if _qkvpostf_ok(n, head, block):
            _fw_qkvpostf_kernel[grid](
                qkv, wqk, freqs, sgn, q, kbuf, vbuf, kv_off, eps,
                n, qn, head, head // 2, block,
                num_warps=_QKVPOST_NW,
            )
        else:
            _fw_qkvpost_kernel[grid](
                qkv, wqk, freqs, sgn, q, kbuf, vbuf, kv_off, eps,
                n, qn, head, head // 2, block,
                num_warps=8, num_stages=1,
            )

    @torch.library.custom_op("fastwam::qkv_post", mutates_args={"kbuf", "vbuf"})
    def _fw_qkv_post(qkv: torch.Tensor, wqk: torch.Tensor, freqs: torch.Tensor,
                     sgn: torch.Tensor, kbuf: torch.Tensor, vbuf: torch.Tensor,
                     kv_off: int, head: int, eps: float) -> torch.Tensor:
        S, qn = qkv.shape
        N = qn // 3
        q = torch.empty((1, S, N), dtype=qkv.dtype, device=qkv.device)
        BLOCK = 1
        while BLOCK < N:
            BLOCK *= 2
        _qkvpost_launch((S, 3), qkv, wqk, freqs, sgn, q, kbuf, vbuf,
                        kv_off * N, eps, N, qn, head, BLOCK)
        return q

    @_fw_qkv_post.register_fake
    def _(qkv, wqk, freqs, sgn, kbuf, vbuf, kv_off, head, eps):
        S, qn = qkv.shape
        return qkv.new_empty((1, S, qn // 3))

    # -----------------------------------------------------------------
    # VIDEO-PREFILL SIBLING of `fastwam::qkv_post`.
    #
    # The prefill arm cannot use `qkv_post`: that op writes K and V straight
    # into the action expert's joint K/V cache buffers at `kv_off`, and the
    # prefill has no such buffers -- it must HAND BACK k and v as the cache
    # entries themselves.  So `_qkvpost_ok`'s envelope excludes it and the
    # prefill still runs inductor's
    #   triton_red_fused__to_copy_add_flip_mean_mul_pow_rsqrt_short_attn_slice
    # at 6.75 us x 28.7 layers = 0.194 ms.  That kernel is a TWO-PASS looped
    # reduction (welford-style r0 loop, then a second pass that re-reads the
    # same rows) over only 98 rows -- 2.4 MB in 6.75 us = 356 GB/s, i.e. ~2.5 us
    # ABOVE the ~4.3 us gfx950 dispatch floor, purely L2 round-trip latency
    # serialised across the reduction loop (the r1_d1 finding).
    #
    # This op runs the EXACT SAME `_fw_qkvpost_kernel` -- same fp32 chain, same
    # flip index, same `sin*sgn`-first association, same single-block `tl.sum`
    # -- with grid (S, 2) instead of (S, 3): `w == 2` (the V copy) is simply not
    # launched, because the prefill's V is a zero-copy strided view of the fused
    # QKV buffer and must stay that way.  Two contiguous destinations, one
    # dispatch, one pass over the rows.
    #
    # NUMERICS: identical deviation class to `qkv_post` (single-block fp32 sum
    # vs inductor's looped tree, and no intermediate bf16 round between the
    # RMSNorm and the norm weight). Defended on the oracle + 14-episode rollout.
    # -----------------------------------------------------------------
    @torch.library.custom_op("fastwam::qk_post", mutates_args=())
    def _fw_qk_post(qkv: torch.Tensor, wqk: torch.Tensor, freqs: torch.Tensor,
                    sgn: torch.Tensor, head: int, eps: float) -> List[torch.Tensor]:
        S, qn = qkv.shape
        N = qn // 3
        q = torch.empty((1, S, N), dtype=qkv.dtype, device=qkv.device)
        k = torch.empty((1, S, N), dtype=qkv.dtype, device=qkv.device)
        BLOCK = 1
        while BLOCK < N:
            BLOCK *= 2
        _qkvpost_launch((S, 2), qkv, wqk, freqs, sgn, q, k, k,
                        0, eps, N, qn, head, BLOCK)
        return [q, k]

    @_fw_qk_post.register_fake
    def _(qkv, wqk, freqs, sgn, head, eps):
        S, qn = qkv.shape
        N = qn // 3
        return [qkv.new_empty((1, S, N)), qkv.new_empty((1, S, N))]


# ---------------------------------------------------------------------------
# HAND-WRITTEN SPLIT-K GEMM FOR THE TWO GRID-STARVED N=1024 DENOISE PROJECTIONS.
#
# WHY. Profiled on gfx950/256CU, both of these run the SAME inductor template
# (BLOCK_M=32, BLOCK_N=16, BLOCK_K=256, num_warps=2):
#   o-proj   M=32 N=1024 K=3072  -> grid_0 = N/16 = 64 CTAs (25% fill), 11.34 us
#   ffn-down M=32 N=1024 K=4096  -> grid_0 = N/16 = 64 CTAs (25% fill), 13.57 us
# M=32 <= BLOCK_M so grid_m == 1 and CTAs == N/BLOCK_N exactly; achieved
# bandwidth across five kernels sharing that template is monotone in CTA count
# (64 -> 578 GB/s, 192 -> 1274, 256 -> 1546, 576 -> 2605). Not bandwidth-bound
# (7-8% of the ~8 TB/s roof), not register-bound (vgpr 114, zero spills) --
# simply never enough blocks launched. N is exhausted, so K is the only axis
# left. Inductor's own `decompose_k` is enabled above and did NOT fire (its
# isolated autotune sees the extra fp32-reduction dispatch and rejects it), so
# the split is forced here instead.
#
# NUMERICS (COMMANDMENT rule 7). Every partition accumulates in fp32 exactly
# like the template does; the cross-partition combine is also fp32, in a fixed
# s = 0..SPLIT_K-1 order out of a deterministic workspace (no float atomics,
# nothing order-dependent), and only the combined value is rounded to bf16 --
# one rounding, same as today. The only difference from the single-pass template
# is the fp32 summation TREE over K, and the oracle moved 1.105e-02 -> 1.137e-02
# mean|d| against a 2.283e-02 gate.
#
# WHY TWO KERNELS AND NOT A FUSED FIX-UP (measured, do not re-derive). Every
# extra launch in the captured denoise step costs ~4.4 us of DEVICE time no
# matter how few bytes it moves, so a same-launch combine looks like the obvious
# win. It was built and measured: last-arrival fix-up, one device-scope atomic
# per workgroup, combine masked to the winner. It needs a real acquire on the
# partial reads -- release-only + plain loads is wrong (max|d| 199), and
# `sem="acq_rel"` on the counter invalidates L2 in all 256 workgroups
# (28.4 us/launch vs 10.2 for the bare GEMM). release + `.cv` loads is correct
# and is the cheapest form, but the winner's uncached tail still lands at
# 12.47 us/launch in-graph against 4.68 + 4.39 = 9.07 for GEMM + reduce, i.e.
# the serialized fix-up tail costs MORE than the dispatch it removes. Kept
# behind FASTWAM_SKFIX=1 as the recorded negative result; default is the
# two-kernel form.
# ---------------------------------------------------------------------------

_SKMM = os.environ.get("FASTWAM_SKMM", "1").lower() not in ("0", "false", "no", "off")
_SKFIX = os.environ.get("FASTWAM_SKFIX", "0").lower() not in ("0", "false", "no", "off")
_SKRED = tuple(int(v) for v in os.environ.get("FASTWAM_SKRED", "256,4").split(","))
# skpost: bitmask, 1 = self-o site, 2 = cross-o + ffn-down sites (those two
# are COUPLED -- site 2 hands its unrounded fp32 residual to site 3).
_SKPOST = int(os.environ.get("FASTWAM_SKPOST", "3"))
_SKPOST_NW = int(os.environ.get("FASTWAM_SKPOST_NW", "4"))
_SKPOST_NS = int(os.environ.get("FASTWAM_SKPOST_NS", "1"))


def _skmm_cfg(n: int, k: int):
    """(BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages) for one shape."""
    env = os.environ.get("FASTWAM_SKCFG_%d_%d" % (n, k))
    if env:
        p = [int(v) for v in env.split(",")]
        return tuple(p) if len(p) == 6 else None
    return _SKMM_DEFAULTS.get((n, k))


_SKMM_DEFAULTS = {
    (1024, 3072): (32, 32, 128, 8, 2, 2),
    (1024, 4096): (32, 32, 128, 8, 2, 2),
}


if _FA_BSD_OK:

    @triton.jit
    def _fw_skmm_kernel(A, B, C, N: tl.constexpr, K: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
                        M: tl.constexpr, FP32_OUT: tl.constexpr):
        pid = tl.program_id(0)
        pid_k = tl.program_id(1)
        num_n = N // BLOCK_N
        pid_m = pid // num_n
        pid_n = pid % num_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        KPS: tl.constexpr = K // SPLIT_K
        k0 = pid_k * KPS
        a_ptrs = A + offs_m[:, None] * K + (k0 + offs_k)[None, :]
        b_ptrs = B + offs_n[None, :] * K + (k0 + offs_k)[:, None]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in tl.range(0, KPS // BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        if FP32_OUT:
            o_ptrs = C + pid_k * (M * N) + offs_m[:, None] * N + offs_n[None, :]
            tl.store(o_ptrs, acc)
        else:
            o_ptrs = C + offs_m[:, None] * N + offs_n[None, :]
            tl.store(o_ptrs, acc.to(C.dtype.element_ty))

    @triton.jit
    def _fw_skmm_fx_kernel(A, B, C, P, LOCK, FLAG, N: tl.constexpr, K: tl.constexpr,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                           BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
                           M: tl.constexpr):
        """Split-K with the fp32 combine FUSED INTO THE SAME LAUNCH.

        Every extra kernel in the captured denoise step costs ~4.4 us of device
        time regardless of the bytes it moves (measured: the five <0.3 MB
        epilogue kernels all land at 4.3-4.6 us against a ~1.96 us chained no-op
        floor), so a two-kernel split-K pays the whole win back in the reduction
        dispatch. Here the LAST partition to finish a tile does the combine.

        Workgroups that are not last mask off every load and store of the
        combine, so they issue no combine traffic at all.

        Deterministic (fixed s = 0..SPLIT_K-1 combine order, no float atomics)
        and fp32 throughout: identical arithmetic to the two-kernel form.
        """
        pid = tl.program_id(0)
        tile = pid // SPLIT_K
        sk = pid % SPLIT_K
        num_n: tl.constexpr = N // BLOCK_N
        offs_m = (tile // num_n) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = (tile % num_n) * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        KPS: tl.constexpr = K // SPLIT_K
        a_ptrs = A + offs_m[:, None] * K + (sk * KPS + offs_k)[None, :]
        b_ptrs = B + offs_n[None, :] * K + (sk * KPS + offs_k)[:, None]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in tl.range(0, KPS // BLOCK_K):
            acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        off = offs_m[:, None] * N + offs_n[None, :]
        tl.store(P + sk * (M * N) + off, acc)
        one = tl.arange(0, 1)
        # ONE device-scope RELEASE atomic per workgroup: publishes this
        # workgroup's partial before the arrival is announced. The matching
        # acquire is done per-LOAD below with `.cv` instead of on the atomic:
        # `sem="acq_rel"` emits a whole-L2 invalidate in all 256 workgroups and
        # measured 28.4 us/launch vs 15.3 us for release + `.cv` (and 10.2 us
        # for the bare GEMM). Plain loads with a release-only atomic are NOT
        # safe -- measured max|d| 199 -- so the `.cv` is load-bearing.
        old = tl.atomic_add(LOCK + tile + one, 1, sem="release", scope="gpu")
        # Broadcast the arrival number to every lane THROUGH MEMORY. A register
        # broadcast (`tl.max(old, 0)`) is silently wrong on this build: measured,
        # 226 of 256 workgroups then saw a non-uniform value and the kernel was
        # deterministically incorrect. A per-element lock fixes that but costs
        # BLOCK_M*BLOCK_N device-scope atomics per workgroup, which measured
        # +1.75 ms/denoise-step. One scalar store + a workgroup barrier + a
        # uniform reload is exact and free.
        tl.store(FLAG + pid + one, old)
        tl.debug_barrier()
        last = tl.load(FLAG + pid + tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)) == SPLIT_K - 1
        tot = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for s in tl.static_range(0, SPLIT_K):
            tot += tl.load(P + s * (M * N) + off, mask=last, other=0.0,
                           cache_modifier=".cv")
        tl.store(C + off, tot.to(C.dtype.element_ty), mask=last)
        # leave the arrival counter zeroed for the next launch
        tl.store(LOCK + tile + one, 0, mask=(old == SPLIT_K - 1))

    @triton.jit
    def _fw_skred_kernel(P, C, MN: tl.constexpr, SPLIT_K: tl.constexpr,
                         BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        acc = tl.load(P + offs)
        for s in tl.static_range(1, SPLIT_K):
            acc += tl.load(P + s * MN + offs)
        tl.store(C + offs, acc.to(C.dtype.element_ty))

    _SKMM_LOCK = {}
    _SKMM_FLAG_OFF = 4096      # [0:4096) arrival counters, [4096:) per-CTA flags

    def _skmm_lock(dev, n_elem: int):
        """Persistent arrival-counter buffer, zero on entry AND on exit.

        Allocated lazily on the first (eager, pre-capture) call and cached, so
        the captured graph replays against a fixed address. Never allocated
        while a graph is capturing -- that would put it in the graph's private
        pool; the two-kernel path covers that case.
        """
        key = (dev.type, dev.index, n_elem)
        t = _SKMM_LOCK.get(key)
        if t is None:
            try:
                if torch.cuda.is_current_stream_capturing():
                    return None
            except Exception:  # pragma: no cover
                pass
            t = torch.zeros(n_elem, dtype=torch.int32, device=dev)
            _SKMM_LOCK[key] = t
        return t

    @torch.library.custom_op("fastwam::skmm", mutates_args=())
    def _fw_skmm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        M = x.shape[-2]
        K = x.shape[-1]
        N = w.shape[0]
        cfg = _skmm_cfg(N, K)
        BM, BN, BK, SK, NW, NS = cfg
        out = torch.empty((1, M, N), dtype=x.dtype, device=x.device)
        grid = ((M // BM) * (N // BN), SK)
        lock = _skmm_lock(x.device, 2 * _SKMM_FLAG_OFF) if (SK > 1 and _SKFIX) else None
        if lock is not None:
            part = torch.empty((SK, M, N), dtype=torch.float32, device=x.device)
            _fw_skmm_fx_kernel[((M // BM) * (N // BN) * SK,)](
                x, w, out, part, lock, lock[_SKMM_FLAG_OFF:], N, K, BM, BN, BK, SK, M,
                num_warps=NW, num_stages=NS, **_SKMM_KW,
            )
        elif SK == 1:
            _fw_skmm_kernel[grid](
                x, w, out, N, K, BM, BN, BK, SK, M, False,
                num_warps=NW, num_stages=NS, **_SKMM_KW,
            )
        else:
            part = torch.empty((SK, M, N), dtype=torch.float32, device=x.device)
            _fw_skmm_kernel[grid](
                x, w, part, N, K, BM, BN, BK, SK, M, True,
                num_warps=NW, num_stages=NS, **_SKMM_KW,
            )
            MN = M * N
            RB, RW = _SKRED
            _fw_skred_kernel[(MN // RB,)](
                part, out, MN, SK, RB, num_warps=RW, num_stages=1,
            )
        return out

    @_fw_skmm.register_fake
    def _(x, w):
        return x.new_empty((1, x.shape[-2], w.shape[0]))

    # -----------------------------------------------------------------
    # SPLIT-K COMBINE FUSED INTO ITS RESIDUAL + GATE + LAYERNORM CONSUMER
    # ("skpost").  3 dispatches -> 2.
    #
    # Every one of the three `fastwam::skmm` calls per denoise layer runs
    #     _fw_skmm_kernel (256 CTAs, 4.71 us)
    #  -> _fw_skred_kernel (128 CTAs, 4.34 us, 1.05 MB at 242 GB/s)
    #  -> triton_per_fused_add_mul_native_layer_norm_split_* (32 CTAs, 4.34 us)
    # and the middle kernel exists only to add SPLIT_K fp32 slices.  On gfx950
    # a dispatch inside the captured denoise graph costs ~4.4 us of DEVICE time
    # regardless of CTAs or bytes (128 CTAs/1.05 MB and 32 CTAs/0.20 MB are
    # both 4.34 us), so the only remedy for that class is FEWER dispatches.
    #
    # This kernel deletes the middle one by reading the fp32 partials directly
    # in the consumer:  sum over SPLIT_K -> +bias -> [*gate] -> +residual ->
    # LayerNorm -> (affine | adaLN modulate) -> split.  Grid is one CTA per row
    # (M=32), i.e. exactly the grid inductor's own consumer used; the extra
    # traffic is 8 x 4 KB of fp32 partials per row instead of one 2 KB bf16
    # row, issued as SPLIT_K independent loads so they overlap.
    #
    # NUMERICS: bit-identical to the two-kernel path by construction.  The
    # combine keeps `_fw_skred_kernel`'s exact sequential s = 0..SPLIT_K-1
    # fp32 order and rounds the combined value to bf16 before the bias add,
    # which is what the separate reduce + inductor consumer did; the residual,
    # gate, LayerNorm and modulate arithmetic is transcribed from the inductor
    # kernel (fp32 throughout, x stored bf16 but the UNROUNDED fp32 value fed
    # to the LayerNorm, exactly as inductor codegens it).
    #
    # NOT the naive route: returning fp32 partials and letting inductor's
    # `.sum(0)` fuse into its own consumer was measured a loser -- that
    # consumer runs at ~45 GB/s, so handing it 1 MB instead of 64 KB costs
    # more than the launch it saves.  Going 3 -> 1 (combine back inside the
    # GEMM) is the FASTWAM_SKFIX path above: correct but slower.
    # -----------------------------------------------------------------

    @triton.jit
    def _fw_skpost_kernel(P, RES, BIAS, MODG, MODA, TMOD, LNW, LNB, XO, YO,
                          MN: tl.constexpr, N: tl.constexpr,
                          SPLIT_K: tl.constexpr, GI: tl.constexpr,
                          SI: tl.constexpr, HI: tl.constexpr,
                          HAS_GATE: tl.constexpr, AFFINE: tl.constexpr,
                          EPS: tl.constexpr, BN: tl.constexpr):
        m = tl.program_id(0)
        offs = tl.arange(0, BN)
        base = m * N + offs
        acc = tl.load(P + base)
        for s in tl.static_range(1, SPLIT_K):
            acc += tl.load(P + s * MN + base)
        y = acc.to(tl.bfloat16).to(tl.float32) + tl.load(BIAS + offs).to(tl.float32)
        if HAS_GATE:
            g = (tl.load(MODG + GI * N + offs).to(tl.float32)
                 + tl.load(TMOD + GI * N + offs).to(tl.float32))
            y = g * y
        x = tl.load(RES + base).to(tl.float32) + y
        tl.store(XO + base, x.to(XO.dtype.element_ty))
        mu = tl.sum(x, 0) / N
        d = x - mu
        var = tl.sum(d * d, 0) / N
        o = d * tl.rsqrt(var + EPS)
        if AFFINE:
            o = (o * tl.load(LNW + offs).to(tl.float32)
                 + tl.load(LNB + offs).to(tl.float32))
        else:
            sc = (tl.load(MODA + SI * N + offs).to(tl.float32)
                  + tl.load(TMOD + SI * N + offs).to(tl.float32))
            sh = (tl.load(MODA + HI * N + offs).to(tl.float32)
                  + tl.load(TMOD + HI * N + offs).to(tl.float32))
            o = o * (sc + 1.0) + sh
        tl.store(YO + base, o.to(YO.dtype.element_ty))

    @torch.library.custom_op("fastwam::skpost", mutates_args=())
    def _fw_skpost(a: torch.Tensor, w: torch.Tensor, bias: torch.Tensor,
                   res: torch.Tensor, modg: torch.Tensor, moda: torch.Tensor,
                   tmod: torch.Tensor, lnw: Optional[torch.Tensor],
                   lnb: Optional[torch.Tensor], gi: int, si: int, hi: int,
                   eps: float, xdt: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        M = a.shape[-2]
        K = a.shape[-1]
        N = w.shape[0]
        BM, BN, BK, SK, NW, NS = _skmm_cfg(N, K)
        part = torch.empty((SK, M, N), dtype=torch.float32, device=a.device)
        _fw_skmm_kernel[((M // BM) * (N // BN), SK)](
            a, w, part, N, K, BM, BN, BK, SK, M, True,
            num_warps=NW, num_stages=NS, **_SKMM_KW,
        )
        xo = torch.empty((1, M, N),
                         dtype=torch.float32 if xdt else a.dtype,
                         device=a.device)
        yo = torch.empty((1, M, N), dtype=a.dtype, device=a.device)
        _fw_skpost_kernel[(M,)](
            part, res, bias, modg, moda, tmod,
            bias if lnw is None else lnw, bias if lnb is None else lnb,
            xo, yo, M * N, N, SK, gi, si, hi,
            gi >= 0, lnw is not None, float(eps), N,
            num_warps=_SKPOST_NW, num_stages=_SKPOST_NS,
        )
        return xo, yo

    @_fw_skpost.register_fake
    def _(a, w, bias, res, modg, moda, tmod, lnw, lnb, gi, si, hi, eps, xdt=0):
        n = w.shape[0]
        xd = torch.float32 if xdt else a.dtype
        return (a.new_empty((1, a.shape[-2], n), dtype=xd),
                a.new_empty((1, a.shape[-2], n)))


def _skmm_ok(x, w) -> bool:
    """Static-shape envelope: only the two profiled denoise projections."""
    if not (_SKMM and _FA_BSD_OK):
        return False
    try:
        if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
            return False
        if x.dim() != 3 or x.shape[0] != 1 or w.dim() != 2:
            return False
        if not x.is_contiguous() or not w.is_contiguous():
            return False
        M, K = int(x.shape[1]), int(x.shape[2])
        N = int(w.shape[0])
        if int(w.shape[1]) != K:
            return False
        cfg = _skmm_cfg(N, K)
        if cfg is None:
            return False
        BM, BN, BK, SK, NW, NS = cfg
        if M % BM or N % BN or K % (SK * BK):
            return False
        if BN < 64 and NW >= 4:      # COMMANDMENT rule 7
            return False
        if _SKFIX and (M // BM) * (N // BN) * SK > _SKMM_FLAG_OFF:
            return False
        if SK > 1 and not _SKFIX and (M * N) % _SKRED[0]:
            return False
        return True
    except Exception:  # pragma: no cover
        return False


def _skpost_ok(x, w, res, mod, tmod, bias) -> bool:
    """Envelope for the fused split-K combine + residual/gate/LayerNorm op.

    Everything `_skmm_ok` demands of the GEMM, plus: a real SPLIT_K, a dense
    bf16 [1, M, N] residual, and the two [1, 6, N] modulation planes the
    consumer reads its gate/shift/scale chunks out of (unmaterialised in the
    inductor consumer today -- passing chunk VIEWS into a custom op would
    force inductor to materialise `block.modulation + t_mod` in its own extra
    launch, which is exactly the dispatch this direction is deleting).
    """
    if not _SKPOST or not _skmm_ok(x, w):
        return False
    try:
        if bias is None or bias.dtype != torch.bfloat16 or not bias.is_contiguous():
            return False
        N = int(w.shape[0])
        M = int(x.shape[1])
        if int(bias.numel()) != N:
            return False
        cfg = _skmm_cfg(N, int(x.shape[2]))
        if cfg is None or int(cfg[3]) <= 1:
            return False
        if res is None or res.dtype not in (torch.bfloat16, torch.float32):
            return False
        if tuple(res.shape) != (1, M, N) or not res.is_contiguous():
            return False
        for t in (mod, tmod):
            if t is None or t.dtype != torch.bfloat16 or not t.is_contiguous():
                return False
            if t.dim() != 3 or tuple(t.shape) != (1, 6, N):
                return False
        return True
    except Exception:  # pragma: no cover
        return False


def _skpost_ok3(block, ref, nmod, tmod) -> bool:
    """Will the ffn-down site fuse for this layer?

    Answered from module attributes only, BEFORE the ffn hidden tensor exists,
    because the cross-o site must decide whether to emit its residual in fp32
    (site 3 consumes it) or bf16 (inductor consumes it).
    """
    if not (_SKMM and _FA_BSD_OK and (_SKPOST & 2) and _SPLITK):
        return False
    try:
        ffn = block.ffn
        if len(ffn) != 3 or not isinstance(ffn[2], nn.Linear):
            return False
        lin = ffn[2]
        if lin.bias is None or lin.weight.dtype != torch.bfloat16:
            return False
        if not lin.weight.is_contiguous() or not lin.bias.is_contiguous():
            return False
        N = int(lin.out_features)
        K = int(lin.in_features)
        M = int(ref.shape[-2])
        cfg = _skmm_cfg(N, K)
        if cfg is None:
            return False
        BM, BN, BK, SK, NW, NS = cfg
        if SK <= 1 or M % BM or N % BN or K % (SK * BK):
            return False
        if BN < 64 and NW >= 4:      # COMMANDMENT rule 7
            return False
        for t in (nmod, tmod):
            if t is None or t.dtype != torch.bfloat16 or not t.is_contiguous():
                return False
            if t.dim() != 3 or tuple(t.shape) != (1, 6, N):
                return False
        return True
    except Exception:  # pragma: no cover
        return False


def _qkvpost_enabled() -> bool:
    return _FA_BSD_OK and _QKVPOST.lower() not in ("0", "false", "no", "off")


def _qkvpost_ok(expert, freqs, cache0, sv: int, sa_len: int) -> bool:
    """Static-shape envelope for the fused post-QKV kernel."""
    try:
        if freqs is None or freqs.is_complex() or freqs.dtype != torch.float32:
            return False
        if int(freqs.shape[0]) != int(sa_len) or not freqs.is_contiguous():
            return False
        head = int(freqs.shape[-1])
        if head % 2 or int(freqs.numel()) != sa_len * head:
            return False
        k0 = cache0["k"]
        v0 = cache0["v"]
        if k0.dim() != 3 or k0.shape[0] != 1 or v0.shape != k0.shape:
            return False
        if int(k0.shape[1]) != sv + sa_len:
            return False
        if not k0.is_contiguous() or not v0.is_contiguous():
            return False
        n = int(k0.shape[2])
        for b in (getattr(expert, "blocks", None) or []):
            sa = getattr(b, "self_attn", None)
            w = getattr(sa, "_fw_qk_nw", None)
            if w is None or int(w.shape[0]) != 2 or int(w.shape[1]) != n:
                return False
            if getattr(sa, "_fw_qkv_h", None) != n:
                return False
            if n % head:
                return False
        return True
    except Exception:  # pragma: no cover
        return False


# ---------------------------------------------------------------------------
# SINGLE-BLOCK RESIDUAL + LAYERNORM + MODULATION FOR THE PREFILL (M=98).
#
# Inductor emits these two per prefill layer, both `red` (LOOPED) reductions
# over xnumel=98 / r0_numel=3072 with `R0_BLOCK` chunks and TWO passes over the
# rows (welford pass, then an apply pass that re-loads everything):
#   ..._native_layer_norm_split_squeeze_view_1   5.08 us   norm3 input  (x + gate_msa*self_o)
#   ..._native_layer_norm_split_squeeze_view_2   6.73 us   mlp  input  (+ cross_o, +mod)
# At 98 rows the grid is 98 CTAs on 256 CU, one wave, one workgroup per CU --
# so the r0 loop has NOTHING to hide its L2 round-trip behind and the second
# pass pays the traffic twice.  That is exactly the r1_d1 finding, and the
# in-run existence proof is `_fw_qkvpost_kernel`: a SINGLE-BLOCK (no r0 loop)
# kernel over the same 98 rows moving the same 2.4 MB runs at 4.54 us where
# inductor's looped equivalent ran at 6.75 us.
#
# So: one program per row, the whole 3072-wide row resident in registers
# (BLOCK=4096, num_warps=8 -> 8 elements/lane), mean and variance from that one
# resident copy, no second pass, no r0 loop.
#
# TRANSCRIBED, NOT REDERIVED -- read off inductor's own generated bodies:
#   * every intermediate stays fp32; only the STORES round to bf16;
#   * mean/var are taken from the UNROUNDED fp32 residual, but the normalised
#     value is taken from the bf16-ROUNDED residual wherever inductor stores it
#     to `in_out_ptr0` and re-loads it (that is `_view_2`, not `_view_1`) --
#     hence the separate ROUND_X flag;
#   * `norm3` keeps its own [N] weight/bias (AF=0); `norm2` is affine-free and
#     is modulated by `(1 + scale_mlp)` / `shift_mlp` read straight out of
#     `block.modulation + t_mod` (AF=1), so no modulation tensor is ever
#     materialised and the fused op costs inductor nothing it was not paying.
# The ONE non-identity is the reduction shape (single-block sum vs a looped
# welford tree) -- same fp32, different summation order, ~1 ulp.
#
# PREFILL ONLY: reached solely from `_pf_post_block`, which is only called when
# `_prefill_one_layer*` passes `pf_mod`.  The action/denoise arm never does.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# HIPBLASLT SOLUTION PIN FOR THE FUSED q|k|v PREFILL GEMM -- frozen TunableOp
# (r2_d0).
#
# TARGET.  video_prefill's Tensile block is 2.7425 ms = 32% of the 1-step call.
# Per site, measured (profile, OFF arm): fused q|k|v [98,3072]x[3072,9216]
# 23.33 us, ffn.0 [.,3072]x[3072,14336] 26.67 us, ffn.2 [.,14336]x[14336,3072]
# 23.02 us + 5.25 us of `PostGSU8` reduction, cross-attn q [.,3072]x[3072,3072]
# 13.15 us -- all x30 layers.
#
# LEVER.  Do NOT out-write Tensile (round 1 measured hand Triton LOSING here:
# q|k|v 25.23 vs 24.08 us, cross-q 19.55 vs 13.09).  Instead make the LIBRARY
# hand back a different solution.  PyTorch's TunableOp is the only enumerator on
# this box (the container ships no `hipblaslt-bench`): it races the real
# hipBLASLt pool per (transA,transB,M,N,K,dtype,epilogue) and records the winner
# by NAME, and the name resolves to a Tensile kernel at library level -- so this
# is pure BACKEND SELECTION, no new kernel and no new dispatch.
#
# WHAT ACTUALLY WON, and it is ONE shape.  The pool was raced exhaustively:
# 264 / 492 / 267 / 805 *supported* solutions for the four shapes above (the
# other ~19k per shape are rejected by `matmulIsAlgoSupported`), in two regimes
# -- isolated with a 512 MB rotating buffer, and chained cold in a HIP graph
# over 30 distinct weight sets.  Only q|k|v moved: `Gemm_Hipblaslt_16750` swaps
# `MT64x128x128` (144 CTAs, 23.33 us, 2.43 TB/s) for `MT80x64x256` (232 CTAs,
# 21.40 us, 2.64 TB/s) -- +61% CTAs on the same weight stream, which is the
# axis the per-site measurement says is binding.  Pinning anything else is a
# MEASURED LOSS and is therefore NOT in the table below (see the notes at the
# end of this comment).
#
# TWO-PHASE, so the scored run pays ZERO tuning cost:
#   FASTWAM_TUNEOP=2 + FASTWAM_TUNEOP_FILE=<path>  -> race and write the table.
#       Run it with FASTWAM_GRAPH=0: a tuning call inside a HIP-graph capture
#       breaks the capture.
#   FASTWAM_TUNEOP=1 (DEFAULT)  -> FROZEN.  The winning table is embedded below,
#       written once to a temp file at import and loaded READ-ONLY with
#       `tuning_enable(False)`.  Selection then happens at graph-CAPTURE time,
#       so steady-state replay pays no lookup at all.
#   FASTWAM_TUNEOP=0            -> kill-switch: the untouched library heuristic.
#
# SAFETY.  Shapes absent from the table fall through to the stock heuristic, so
# every denoise GEMM is byte-identical (only `tn_9216_98_3072` is listed, and
# the denoise expert runs M=32 with N in {1024,3072,4096}).  Same kernel count
# (396) and same 6 launch-API calls per prefill => capture intact.  A solution
# name that fails to resolve degrades silently to the framework path, which is
# why engagement is asserted from the PROFILE (a different `Cijk_*` name at that
# site), not from the absence of an error.
#
# MEASURED, ON THE PROFILE (OFF -> ON): q|k|v site 23.33 -> 21.40 us x30;
# Tensile block 2.7425 -> 2.6778 ms; video_prefill wall 4.5113 -> 4.4577 ms;
# infer_action 8.4823 -> 8.3986 (1 step) and 25.7901 -> 25.6970 (10 step).
#
# FALSIFIED HERE, do not re-derive.  (a) The "a GSU/split-U solution exists and
# was simply never requested" thesis: `PostGSU8` in this region belongs to
# ffn.2, which ALREADY uses split-U, and no GSU solution wins any of the other
# three sites out of the full raced pool.  (b) Workspace was not the gate:
# raising `HIPBLASLT_WORKSPACE_SIZE` to 256 MB moves the supported-solution
# count 189->209 / 525->776 and the best time by <1%.  (c) `OPTIMIZE_EPILOGUE=1`
# selects the identical kernels (prefill 4.5027 vs 4.5113 ms, inside noise).
# (d) Dropping the bias epilogue to widen the pool (`GemmTunableOp` instead of
# `GemmAndBias`) does not help: 23.00 vs 23.25 us on q|k|v, worse elsewhere.
# (e) Transplanting 16750 onto ffn.0 (27.78 us) or onto cross-q/ffn.2 (Tensile
# block 3.0785 ms) is a large LOSS.  (f) cross-attn q is not fixable by
# selection: its `MT32x64x256` re-reads the B panel ceil(98/32)=4x, which puts
# it at ~5.7 TB/s of EFFECTIVE traffic -- it is already at the memory ceiling,
# it is the ADDRESSING that is wrong, and no pooled solution tiles it better.
# ---------------------------------------------------------------------------
_TUNEOP = int(os.environ.get("FASTWAM_TUNEOP", "1"))

# Frozen TunableOp result table (validators + winning solution per shape).
_TUNEOP_BLOB = """\
Validator,PT_VERSION,2.11.0
Validator,HIP_VERSION,714
Validator,HIPBLASLT_VERSION,100401-cd957402
Validator,GCN_ARCH_NAME,gfx950:sramecc+:xnack-
Validator,ROCBLAS_VERSION,5.5.0.cd957402
GemmAndBiasTunableOp_BFloat16_TN,tn_9216_98_3072_ld_3072_3072_9216,Gemm_Hipblaslt_16750,0.023161
"""


def _tuneop_init() -> None:
    if _TUNEOP <= 0:
        return
    try:
        import torch.cuda.tunable as _tun
    except Exception:  # pragma: no cover
        return
    try:
        if _TUNEOP >= 2:
            fn = os.environ.get("FASTWAM_TUNEOP_FILE", "")
            if not fn:
                logger.warning("FastWAM: FASTWAM_TUNEOP=2 but FASTWAM_TUNEOP_FILE unset")
                return
            _tun.enable(True)
            _tun.set_rotating_buffer_size(int(os.environ.get("FASTWAM_TUNEOP_ROT", "512")))
            _tun.set_max_tuning_duration(int(os.environ.get("FASTWAM_TUNEOP_MS", "80")))
            _tun.set_max_tuning_iterations(int(os.environ.get("FASTWAM_TUNEOP_IT", "100")))
            _tun.set_filename(fn, False)
            _tun.tuning_enable(True)
            return
        blob = _TUNEOP_BLOB
        if not blob.strip():
            return
        import hashlib
        import tempfile
        d = os.path.join(tempfile.gettempdir(), "fastwam_tuneop_%d" % os.getuid())
        os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, "tuned_%s.csv" % hashlib.md5(blob.encode()).hexdigest()[:12])
        if not os.path.exists(fn):
            tmp = "%s.%d" % (fn, os.getpid())
            with open(tmp, "w") as f:
                f.write(blob)
            os.replace(tmp, fn)
        _tun.enable(True)
        _tun.tuning_enable(False)
        _tun.set_filename(fn, False)
        try:
            _tun.read_file(fn)
        except Exception:
            pass
    except Exception as _exc:  # pragma: no cover
        logger.warning("FastWAM: TunableOp init failed (%r)", _exc)


_tuneop_init()


# ---------------------------------------------------------------------------
# PREFILL GEMM BACKEND OVERRIDE (r3_d0).
#
# WHY.  `max-autotune` ranks GEMM backends by benchmarking each shape in
# ISOLATION, i.e. with the whole B panel resident in the 256 MB Infinity Cache.
# The video prefill walks 30 layers x ~1.5 GB of DISTINCT weights, so in the
# real run every B panel arrives COLD from HBM and the ranking flips.  Measured
# on this box with a rotating weight set larger than the LLC, M=98, bf16,
# chained inside a HIP graph (so the isolation artefact is removed):
#
#     N=9216  K=3072 (q|k|v) : hipBLASLt 22.7-23.2 us
#                              vs the shipped `triton_tem_fused_addmm_t_view_0`
#                              at 0.7670 ms / 28.1 calls = 27.30 us
#     N=14336 K=3072 (ffn.0) : hipBLASLt + fused GELU epilogue 25.3 us
#                              vs `triton_tem_fused_addmm_gelu_t_view_1`
#                              at 0.8609 ms / 28.0 calls = 30.75 us
#
# Wrapping the call in an OPAQUE custom op is what routes it to aten (and thus
# hipBLASLt); flipping inductor's global `max_autotune_gemm_backends` would also
# re-pick every DENOISE GEMM, which is out of scope and must stay byte-identical.
#
# ffn.0 keeps its GELU IN THE GEMM EPILOGUE via `torch._addmm_activation`, so
# this is still ONE dispatch -- a plain aten addmm plus a standalone gelu
# measured 28.5 us and would have given the win straight back to the ~4.4 us
# dispatch floor.
#
# PREFILL ONLY, BY CONSTRUCTION.  Site 1 is inside the `pf` branch of
# `_build_expert_attention_io` (the denoise arm never sets `pf`); site 2 is
# inside `_pf_post_block`, which only runs when `_prefill_one_layer` supplies
# `pf_mod`.  `_pfgemm_ok` additionally demands M >= 64 and N >= 8192, and the
# denoise expert runs M=32 with N in {1024, 3072, 4096}, so neither clause can
# be satisfied on that arm even if a call site were reached by accident.
# NOTHING in `_skmm_ok` / `_SKMM_DEFAULTS` / `_skpost_ok3` is touched.
# ---------------------------------------------------------------------------
_PFGEMM = int(os.environ.get("FASTWAM_PFGEMM", "3"))   # bit1 = q|k|v, bit2 = ffn.0


@torch.library.custom_op("fastwam::pf_mm", mutates_args=())
def _fw_pf_mm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.linear(x, w, b)


@_fw_pf_mm.register_fake
def _(x, w, b):
    return x.new_empty(tuple(x.shape[:-1]) + (int(w.shape[0]),))


@torch.library.custom_op("fastwam::pf_mm_gelu", mutates_args=())
def _fw_pf_mm_gelu(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x2 = x.reshape(-1, int(x.shape[-1]))
    o = torch._addmm_activation(b, x2, w.t(), use_gelu=True)
    return o.view(tuple(x.shape[:-1]) + (int(w.shape[0]),))


@_fw_pf_mm_gelu.register_fake
def _(x, w, b):
    return x.new_empty(tuple(x.shape[:-1]) + (int(w.shape[0]),))


def _pfgemm_ok(x, w, b, bit: int) -> bool:
    """Prefill-only envelope for the aten/hipBLASLt GEMM override."""
    if not (_PFGEMM & bit):
        return False
    try:
        if x is None or w is None or b is None:
            return False
        if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
            return False
        if b.dtype != torch.bfloat16 or b.dim() != 1:
            return False
        if x.dim() != 3 or int(x.shape[0]) != 1 or w.dim() != 2:
            return False
        if not x.is_contiguous() or not w.is_contiguous():
            return False
        if not b.is_contiguous():
            return False
        M, K = int(x.shape[1]), int(x.shape[2])
        N = int(w.shape[0])
        if int(w.shape[1]) != K or int(b.shape[0]) != N:
            return False
        if M < 64 or N < 8192:      # disjoint from the denoise arm (M=32)
            return False
        return True
    except Exception:  # pragma: no cover
        return False


_PFRES = os.environ.get("FASTWAM_PFRES", "1").lower() not in ("0", "false", "no", "off")
# r1_d1: the layer-boundary tail fusion.  FASTWAM_PFLNT=0 restores inductor's
# two-pass looped welford at the layer boundary, byte for byte.
_PFLNT = os.environ.get("FASTWAM_PFLNT", "1").lower() not in ("0", "false", "no", "off")
# r2_d2: LOAD-SCHEDULING HOIST for the prefill row-combine chain.
# `_fw_resln_kernel`, `_fw_reslnsk_kernel` and `_fw_qcomb_kernel` all issue
# their small cold [N] vectors (norm weight/bias, modulation gate/shift/scale,
# GEMM bias) AFTER the split-K reduction and/or AFTER the two cross-lane
# LayerNorm sums, so those HBM round trips sit on the critical path with
# nothing left to overlap them.  With only 98 CTAs on 256 CUs there is no
# other wave to hide them behind, and between two launches of the same kernel
# the prefill streams ~90 MB of layer weights through the LLC, so every one of
# those vectors is cold every time.  Issuing them all UP FRONT puts them in
# flight together, behind the row loads and the reduction.
# BIT-IDENTICAL: same operands, same fp32 expression order, only issue order
# moves.  Kill switch FASTWAM_PFLDH=0 restores the original bodies verbatim.
_PFLDH = int(os.environ.get("FASTWAM_PFLDH", "1"))
# r2_d2: num_warps=4 on the 98-CTA row kernels.  With BLOCK=4096 over N=3072
# these are latency-bound, not ALU-bound, and halving the warp count raises the
# per-lane load count (8 -> 16 elements) so each wave has more independent
# loads in flight.  Measured 4/4 lower device time (reslnsk 489-493 -> 477-479
# us/call over 90 launches).  Restore with FASTWAM_RESLNSKNW=8 etc.
_RESLN_NW = int(os.environ.get("FASTWAM_RESLNNW", "4"))
_RESLNSK_NW = int(os.environ.get("FASTWAM_RESLNSKNW", "4"))

if _FA_BSD_OK:

    @triton.jit
    def _fw_resln_kernel(X, O1, O2, MB, MB2, TM, WN, BN_, XO, NO, eps,
                         N: tl.constexpr, MS: tl.constexpr,
                         GO: tl.constexpr, SHO: tl.constexpr, SCO: tl.constexpr,
                         AF: tl.constexpr, HAS_O2: tl.constexpr,
                         WRITE_X: tl.constexpr, ROUND_X: tl.constexpr,
                         BLOCK: tl.constexpr, H: tl.constexpr = 0):
        s = tl.program_id(0)
        r = tl.arange(0, BLOCK)
        m = r < N
        base = s * N + r
        trow = TM + s * MS
        if H:
            # r2_d2 hoist: every cold [N] vector issued ahead of the two sums.
            if AF == 0:
                wv = tl.load(WN + r, mask=m, other=0.0).to(tl.float32)
                bv = tl.load(BN_ + r, mask=m, other=0.0).to(tl.float32)
            else:
                sc = (tl.load(MB2 + SCO + r, mask=m, other=0.0).to(tl.float32)
                      + tl.load(trow + SCO + r, mask=m, other=0.0).to(tl.float32))
                sh = (tl.load(MB2 + SHO + r, mask=m, other=0.0).to(tl.float32)
                      + tl.load(trow + SHO + r, mask=m, other=0.0).to(tl.float32))
        x = tl.load(X + base, mask=m, other=0.0).to(tl.float32)
        o = tl.load(O1 + base, mask=m, other=0.0).to(tl.float32)
        g = (tl.load(MB + GO + r, mask=m, other=0.0).to(tl.float32)
             + tl.load(trow + GO + r, mask=m, other=0.0).to(tl.float32))
        t = x + g * o
        if HAS_O2:
            t = t + tl.load(O2 + base, mask=m, other=0.0).to(tl.float32)
        if WRITE_X:
            tl.store(XO + base, t.to(tl.bfloat16), mask=m)
        tn = t
        if ROUND_X:
            tn = t.to(tl.bfloat16).to(tl.float32)
        mu = tl.sum(t, 0) / N
        d = tl.where(m, t - mu, 0.0)
        var = tl.sum(d * d, 0) / N
        rstd = tl.rsqrt(var + eps)
        y = (tn - mu) * rstd
        if AF == 0:
            if not H:
                wv = tl.load(WN + r, mask=m, other=0.0).to(tl.float32)
                bv = tl.load(BN_ + r, mask=m, other=0.0).to(tl.float32)
            y = y * wv + bv
        else:
            if not H:
                sc = (tl.load(MB2 + SCO + r, mask=m, other=0.0).to(tl.float32)
                      + tl.load(trow + SCO + r, mask=m, other=0.0).to(tl.float32))
                sh = (tl.load(MB2 + SHO + r, mask=m, other=0.0).to(tl.float32)
                      + tl.load(trow + SHO + r, mask=m, other=0.0).to(tl.float32))
            y = y * (sc + 1.0) + sh
        tl.store(NO + base, y.to(tl.bfloat16), mask=m)

    def _resln_block(n: int) -> int:
        b = 1
        while b < n:
            b *= 2
        return b

    @torch.library.custom_op("fastwam::res_ln1", mutates_args=())
    def _fw_res_ln1(x: torch.Tensor, o: torch.Tensor, mb: torch.Tensor,
                    tm: torch.Tensor, wn: torch.Tensor, bn: torch.Tensor,
                    go: int, eps: float) -> torch.Tensor:
        S, N = int(x.shape[1]), int(x.shape[2])
        out = torch.empty((1, S, N), dtype=x.dtype, device=x.device)
        _fw_resln_kernel[(S,)](
            x, o, o, mb, mb, tm, wn, bn, out, out, eps,
            N, int(tm.shape[1]), go, 0, 0, 0, False, False, False,
            _resln_block(N), _PFLDH, num_warps=8, num_stages=1,
        )
        return out

    @_fw_res_ln1.register_fake
    def _(x, o, mb, tm, wn, bn, go, eps):
        return x.new_empty((1, int(x.shape[1]), int(x.shape[2])))

    @torch.library.custom_op("fastwam::res_ln2", mutates_args=())
    def _fw_res_ln2(x: torch.Tensor, o1: torch.Tensor, o2: torch.Tensor,
                    mb: torch.Tensor, mb2: torch.Tensor, tm: torch.Tensor,
                    go: int, sho: int, sco: int, eps: float) -> List[torch.Tensor]:
        S, N = int(x.shape[1]), int(x.shape[2])
        xo = torch.empty((1, S, N), dtype=x.dtype, device=x.device)
        no = torch.empty((1, S, N), dtype=x.dtype, device=x.device)
        _fw_resln_kernel[(S,)](
            x, o1, o2, mb, mb2, tm, mb, mb, xo, no, eps,
            N, int(tm.shape[1]), go, sho, sco, 1, True, True, True,
            _resln_block(N), _PFLDH, num_warps=8, num_stages=1,
        )
        return [xo, no]

    @_fw_res_ln2.register_fake
    def _(x, o1, o2, mb, mb2, tm, go, sho, sco, eps):
        S, N = int(x.shape[1]), int(x.shape[2])
        return [x.new_empty((1, S, N)), x.new_empty((1, S, N))]

    # ---- r1_d1: the LAYER-BOUNDARY tail, same kernel, one `o` -------------
    # `_prefill_one_layer_t` ends every layer with
    #     x = x1 + gate_mlp * ffn_out ;  nxt = adaLN(x, next.norm1, sh, sc)
    # which inductor codegens as `triton_red_..._split_squeeze_view_0`: a TWO
    # PASS looped welford over xnumel=98 / r0_numel=3072 that re-loads the whole
    # 0.6 MB residual for the apply pass.  Measured in-region 6.59 us against
    # 4.5-5.3 us for the single-block kernels already in this file that move the
    # same bytes -- the same defect r2_d1 fixed for the other two prefill
    # LayerNorms and simply never reached the layer boundary.
    #
    # TRANSCRIBED FROM INDUCTOR'S OWN BODY, not re-derived (read out of the
    # generated `..._view_0`): the residual is accumulated in fp32 and STORED
    # bf16; mean/var are taken from the UNROUNDED fp32 residual; the normalised
    # value is taken from the bf16-ROUNDED reload (inductor's second pass loads
    # `in_out_ptr0` back) -> ROUND_X=True; norm1 is affine-free and modulated by
    # the NEXT block's `(1 + scale_msa)` / `shift_msa` read straight out of
    # `mod + t_mod` (AF=1).  The one non-identity is the reduction shape:
    # single-block sum vs a looped welford tree, both fp32.
    @torch.library.custom_op("fastwam::res_ln_tail", mutates_args=())
    def _fw_res_ln_tail(x: torch.Tensor, o: torch.Tensor, mb: torch.Tensor,
                        mb2: torch.Tensor, tm: torch.Tensor, go: int,
                        sho: int, sco: int, eps: float) -> List[torch.Tensor]:
        S, N = int(x.shape[1]), int(x.shape[2])
        xo = torch.empty((1, S, N), dtype=x.dtype, device=x.device)
        no = torch.empty((1, S, N), dtype=x.dtype, device=x.device)
        _fw_resln_kernel[(S,)](
            x, o, o, mb, mb2, tm, mb, mb, xo, no, eps,
            N, int(tm.shape[1]), go, sho, sco, 1, False, True, True,
            _resln_block(N), _PFLDH, num_warps=_RESLN_NW, num_stages=1,
        )
        return [xo, no]

    @_fw_res_ln_tail.register_fake
    def _(x, o, mb, mb2, tm, go, sho, sco, eps):
        S, N = int(x.shape[1]), int(x.shape[2])
        return [x.new_empty((1, S, N)), x.new_empty((1, S, N))]


# ---------------------------------------------------------------------------
# PREFILL SPLIT-K o-PROJECTION, COMBINE FUSED INTO THE RESIDUAL+LN CONSUMER
# (r3_d0).
#
# THE DEFECT.  The three M=98 N=3072 K=3072 projections route to hipBLASLt
# `..._MT32x64x256_MI16x16x1`.  MT32x64 at M=98 forces grid_m = ceil(98/32) = 4,
# so the whole 18.9 MB B panel is streamed up to 4x, and the launch carries only
# 192 CTAs.  Measured on this box (rotating >LLC weight set, chained in a HIP
# graph): 12.03-12.26 us, i.e. ~1565 GB/s counting B once, against a sibling in
# the same region (ffn.0) at 3106 GB/s.
#
# WHAT WAS RACED FIRST, AND LOST.  A straight Triton port is a closed axis here
# and the numbers say so: single-launch grid_m=1 tops out at 18.7 us (BN=16) /
# 21.4 us (BN=64) because 48-96 CTAs cannot pull 18.9 MB; NN layout (weights
# pre-transposed) is 14.4 us vs 12.3; padding M to 128 is 12.23 vs 12.26.  The
# ONE thing that wins is BLOCK_M=128 (grid_m=1, B read once) + SPLIT_K=4
# (192 CTAs at constant B traffic): 9.05 us, 1.33x.  But a standalone fp32
# combine kernel costs a whole ~4.4 us dispatch and hands the win straight back
# (9.05 + 3.3 = 12.37 us, i.e. 0.98x).
#
# THE FIX.  This site already HAS a hand-written consumer -- `_fw_resln_kernel`,
# the prefill residual + LayerNorm pass -- sitting on the dispatch floor at
# 4.47 us for 2.4 MB.  Fold the SPLIT_K fp32 combine and the bias into it.  Net
# per site: one 9.05 us GEMM + a slightly fatter (but still floor-bound)
# consumer, instead of a 12.03 us GEMM + the same consumer.  This is the same
# trick r2_d0 used on the denoise arm (`skpost`), applied to the prefill arm.
#
# NUMERICS.  The combine is fp32 and is rounded to bf16 EXACTLY ONCE, at the
# same place the GEMM's own bf16 output was rounded, and the rounded value is
# what feeds the LayerNorm -- so the only change is the accumulation ORDER
# inside the GEMM (4 fp32 partial sums instead of hipBLASLt's single pass).
# The residual `x2 = x + gate*o + c` intermediate stays fp32 exactly as before.
#
# DISJOINT FROM THE DENOISE ARM.  Both call sites live in `_pf_post_block`,
# which only runs when `_prefill_one_layer` supplies `pf_mod`.  `_pfsk_ok`
# additionally requires 64 <= M <= 128, and the action/denoise expert runs
# M=32.  `_skmm_ok`, `_SKMM_DEFAULTS` and `_skpost_ok3` are NOT touched, so the
# denoise ffn-down fusion is unaffected.
# ---------------------------------------------------------------------------
_PFSK = os.environ.get("FASTWAM_PFSK", "1").lower() not in ("0", "false", "no", "off")
# BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, pipeline stages
_PFSK_CFG = tuple(int(v) for v in
                  os.environ.get("FASTWAM_PFSKCFG", "128,64,128,4,8,2").split(","))
_PFSK_SHARE = os.environ.get("FASTWAM_PFSKSHARE", "1").lower() not in ("0", "false", "no", "off")
# Non-temporal hint on the read-once weight stream of `_fw_pfsk_kernel`.
# MEASURED NEGATIVE, kept as a default-OFF knob so the finding is not re-derived:
# `.cg` on the W load is a clear win in an isolated 30-launch replay graph
# (8.91 -> 8.26 us) and a clear LOSS in situ (in-graph 9.37 -> 9.82 us), because
# the real prefill interleaves this kernel with three library GEMMs that have
# already reshaped L2 -- an isolated back-to-back replay is NOT a valid ranker
# for cache-policy hints here.  Same for `matrix_instr_nonkdim=32` (micro 8.38,
# in situ 9.21 vs 9.37 -- inside noise, and round 2 measured it a loss).
_PFSK_CG = int(os.environ.get("FASTWAM_PFSKCG", "0"))
# r2_d2: the ffn.2 down-projection is a DIFFERENT regime from the three
# K=3072 sites -- its B panel is 88.1 MB (4.7x bigger) and read exactly once,
# so it is pure streaming bandwidth and wants CTAs, not reuse.  Its geometry is
# therefore tuned separately.  BM,BN,BK,SPLIT_K,num_warps,stages.
_PFFFN_CFG = tuple(int(v) for v in
                   os.environ.get("FASTWAM_PFFFNCFG", "128,128,128,8,8,2").split(","))
# `.cg` on the ffn.2 weight stream only: unlike the K=3072 sites (where it was
# measured a loss in situ) this panel is 88.1 MB read exactly once and is large
# enough to evict everything else from L2/MALL on its way through.
_PFFFN_CG = int(os.environ.get("FASTWAM_PFFFNCG", "0"))

if _FA_BSD_OK:

    @triton.jit
    def _fw_pfsk_kernel(A, W, P, N: tl.constexpr, K: tl.constexpr, M: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        SK: tl.constexpr, NS2: tl.constexpr, CG: tl.constexpr = 0):
        pid = tl.program_id(0)
        pk = tl.program_id(1)
        offs_m = tl.arange(0, BM)
        ma = offs_m < M
        offs_n = pid * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        KPS: tl.constexpr = K // SK
        a_ptrs = A + offs_m[:, None] * K + (pk * KPS + offs_k)[None, :]
        b_ptrs = W + offs_n[None, :] * K + (pk * KPS + offs_k)[:, None]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in tl.range(0, KPS // BK, num_stages=NS2):
            # ROUND 3 (r3_d2): `.cg` on the WEIGHT stream only.  W is a
            # read-once 18.87 MB panel -- nothing in this kernel or the next
            # ever re-reads it -- so caching it in L1/CU-local storage only
            # evicts the A panel, which IS re-read by all 48 n-tiles.  A is
            # left on the default policy for exactly that reason.  Bit-exact:
            # a cache hint changes no value and no order.
            if CG:
                b = tl.load(b_ptrs, cache_modifier=".cg")
            else:
                b = tl.load(b_ptrs)
            acc += tl.dot(tl.load(a_ptrs, mask=ma[:, None], other=0.0),
                          b, out_dtype=tl.float32)
            a_ptrs += BK
            b_ptrs += BK
        tl.store(P + pk * (M * N) + offs_m[:, None] * N + offs_n[None, :],
                 acc, mask=ma[:, None])

    # ONE scratch buffer for every layer and both sites.  A fresh
    # `torch.empty((4, 98, 3072), fp32)` per site is 4.8 MB, and the prefill
    # graph holds 30 layers x 2 sites LIVE in its private pool = 288 MB, which
    # is larger than the 256 MB Infinity Cache and evicts the very weight panels
    # this direction is trying to stop re-streaming.  The partials are consumed
    # by the very next kernel in a strictly serial chain, so one buffer is
    # enough.  Allocated on the pre-capture warm-up call (never while a graph is
    # capturing -- that would put it in the graph's private pool), same rule as
    # `_skmm_lock`.
    _PFSK_BUF = {}
    # Fixed leading dim so the custom op's FAKE output shape does not move with
    # SPLIT_K.  Inductor's FX-graph cache does not key on it, so a buffer whose
    # shape follows SK makes a cached artifact assert the PREVIOUS SK at replay;
    # the actual partition count travels as an int argument instead.
    _PFSK_SKMAX = 8

    def _pfsk_part(dev, SK: int, M: int, N: int):
        key = (dev.type, dev.index, SK, M, N)
        t = _PFSK_BUF.get(key)
        if t is None:
            try:
                if torch.cuda.is_current_stream_capturing():
                    return torch.empty((SK, M, N), dtype=torch.float32, device=dev)
            except Exception:  # pragma: no cover
                pass
            t = torch.empty((SK, M, N), dtype=torch.float32, device=dev)
            _PFSK_BUF[key] = t
        return t

    @torch.library.custom_op("fastwam::pf_skmm", mutates_args=())
    def _fw_pf_skmm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        M = int(x.shape[-2])
        K = int(x.shape[-1])
        N = int(w.shape[0])
        BM, BN, BK, SK, NW, NS2 = _PFSK_CFG
        part = (_pfsk_part(x.device, _PFSK_SKMAX, M, N) if _PFSK_SHARE
                else torch.empty((_PFSK_SKMAX, M, N), dtype=torch.float32,
                                 device=x.device))
        _fw_pfsk_kernel[(N // BN, SK)](
            x, w, part, N, K, M, BM, BN, BK, SK, NS2, _PFSK_CG,
            num_warps=NW, num_stages=1, **_PFSK_KW,
        )
        return part

    @_fw_pf_skmm.register_fake
    def _(x, w):
        return x.new_empty((_PFSK_SKMAX, int(x.shape[-2]), int(w.shape[0])),
                           dtype=torch.float32)

    # A SEPARATE partial buffer for the ffn.2 site, with its own FIXED leading
    # dim, so (a) SPLIT_K here can exceed the o-projections' 8 without moving
    # any FAKE output shape (inductor's FX cache does not key on it) and (b) the
    # ffn producer never aliases the o-projection partials at all.
    _PFFFN_SKMAX = 16

    @torch.library.custom_op("fastwam::pf_skmm_big", mutates_args=())
    def _fw_pf_skmm_big(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        M = int(x.shape[-2])
        K = int(x.shape[-1])
        N = int(w.shape[0])
        BM, BN, BK, SK, NW, NS2 = _PFFFN_CFG
        part = (_pfsk_part(x.device, _PFFFN_SKMAX, M, N) if _PFSK_SHARE
                else torch.empty((_PFFFN_SKMAX, M, N), dtype=torch.float32,
                                 device=x.device))
        _fw_pfsk_kernel[(N // BN, SK)](
            x, w, part, N, K, M, BM, BN, BK, SK, NS2, _PFFFN_CG,
            num_warps=NW, num_stages=1, **_PFSK_KW,
        )
        return part

    @_fw_pf_skmm_big.register_fake
    def _(x, w):
        return x.new_empty((_PFFFN_SKMAX, int(x.shape[-2]), int(w.shape[0])),
                           dtype=torch.float32)

    @triton.jit
    def _fw_reslnsk_kernel(X, O1, P, BI, MB, MB2, TM, WN, BN_, XO, NO, OO, eps, MN,
                           N: tl.constexpr, MS: tl.constexpr,
                           GO: tl.constexpr, SHO: tl.constexpr, SCO: tl.constexpr,
                           AF: tl.constexpr, HAS_O2: tl.constexpr,
                           WRITE_X: tl.constexpr, ROUND_X: tl.constexpr,
                           SKP: tl.constexpr, WRITE_O: tl.constexpr,
                           BLOCK: tl.constexpr, H: tl.constexpr = 0):
        s = tl.program_id(0)
        r = tl.arange(0, BLOCK)
        m = r < N
        base = s * N + r
        trow = TM + s * MS
        if H:
            # r2_d2 hoist: BI, the gate, the norm/modulation vectors and the
            # two row streams all issued BEFORE the split-K reduction, so the
            # cold [N] round trips overlap it instead of following it.
            bih = tl.load(BI + r, mask=m, other=0.0).to(tl.float32)
            g = (tl.load(MB + GO + r, mask=m, other=0.0).to(tl.float32)
                 + tl.load(trow + GO + r, mask=m, other=0.0).to(tl.float32))
            if AF == 0:
                wv = tl.load(WN + r, mask=m, other=0.0).to(tl.float32)
                bv = tl.load(BN_ + r, mask=m, other=0.0).to(tl.float32)
            else:
                sc = (tl.load(MB2 + SCO + r, mask=m, other=0.0).to(tl.float32)
                      + tl.load(trow + SCO + r, mask=m, other=0.0).to(tl.float32))
                sh = (tl.load(MB2 + SHO + r, mask=m, other=0.0).to(tl.float32)
                      + tl.load(trow + SHO + r, mask=m, other=0.0).to(tl.float32))
            if HAS_O2:
                o = tl.load(O1 + base, mask=m, other=0.0).to(tl.float32)
        x = tl.load(X + base, mask=m, other=0.0).to(tl.float32)
        # SPLIT_K combine + bias, in fp32, rounded to bf16 exactly once -- the
        # same single rounding the GEMM's own bf16 output carried.
        p = tl.load(P + base, mask=m, other=0.0)
        for i in tl.static_range(1, SKP):
            p += tl.load(P + i * MN + base, mask=m, other=0.0)
        if H:
            p += bih
        else:
            p += tl.load(BI + r, mask=m, other=0.0).to(tl.float32)
        pb = p.to(tl.bfloat16)
        if WRITE_O:
            tl.store(OO + base, pb, mask=m)
        if not H:
            if HAS_O2:
                o = tl.load(O1 + base, mask=m, other=0.0).to(tl.float32)
        if not HAS_O2:
            o = pb.to(tl.float32)
        if not H:
            g = (tl.load(MB + GO + r, mask=m, other=0.0).to(tl.float32)
                 + tl.load(trow + GO + r, mask=m, other=0.0).to(tl.float32))
        t = x + g * o
        if HAS_O2:
            t = t + pb.to(tl.float32)
        if WRITE_X:
            tl.store(XO + base, t.to(tl.bfloat16), mask=m)
        tn = t
        if ROUND_X:
            tn = t.to(tl.bfloat16).to(tl.float32)
        mu = tl.sum(t, 0) / N
        d = tl.where(m, t - mu, 0.0)
        var = tl.sum(d * d, 0) / N
        rstd = tl.rsqrt(var + eps)
        y = (tn - mu) * rstd
        if AF == 0:
            if not H:
                wv = tl.load(WN + r, mask=m, other=0.0).to(tl.float32)
                bv = tl.load(BN_ + r, mask=m, other=0.0).to(tl.float32)
            y = y * wv + bv
        else:
            if not H:
                sc = (tl.load(MB2 + SCO + r, mask=m, other=0.0).to(tl.float32)
                      + tl.load(trow + SCO + r, mask=m, other=0.0).to(tl.float32))
                sh = (tl.load(MB2 + SHO + r, mask=m, other=0.0).to(tl.float32)
                      + tl.load(trow + SHO + r, mask=m, other=0.0).to(tl.float32))
            y = y * (sc + 1.0) + sh
        tl.store(NO + base, y.to(tl.bfloat16), mask=m)

    @torch.library.custom_op("fastwam::res_ln1_sk", mutates_args=())
    def _fw_res_ln1_sk(x: torch.Tensor, p: torch.Tensor, bi: torch.Tensor,
                       mb: torch.Tensor, tm: torch.Tensor, wn: torch.Tensor,
                       bn: torch.Tensor, go: int, eps: float,
                       sk: int) -> List[torch.Tensor]:
        S, N = int(x.shape[1]), int(x.shape[2])
        out = torch.empty((1, S, N), dtype=x.dtype, device=x.device)
        oo = torch.empty((1, S, N), dtype=x.dtype, device=x.device)
        _fw_reslnsk_kernel[(S,)](
            x, x, p, bi, mb, mb, tm, wn, bn, out, out, oo, eps, S * N,
            N, int(tm.shape[1]), go, 0, 0, 0, False, False, False,
            int(sk), True, _resln_block(N), _PFLDH,
            num_warps=_RESLNSK_NW, num_stages=1,
        )
        return [out, oo]

    @_fw_res_ln1_sk.register_fake
    def _(x, p, bi, mb, tm, wn, bn, go, eps, sk):
        S, N = int(x.shape[1]), int(x.shape[2])
        return [x.new_empty((1, S, N)), x.new_empty((1, S, N))]

    @torch.library.custom_op("fastwam::res_ln2_sk", mutates_args=())
    def _fw_res_ln2_sk(x: torch.Tensor, o1: torch.Tensor, p: torch.Tensor,
                       bi: torch.Tensor, mb: torch.Tensor, mb2: torch.Tensor,
                       tm: torch.Tensor, go: int, sho: int, sco: int,
                       eps: float, sk: int) -> List[torch.Tensor]:
        S, N = int(x.shape[1]), int(x.shape[2])
        xo = torch.empty((1, S, N), dtype=x.dtype, device=x.device)
        no = torch.empty((1, S, N), dtype=x.dtype, device=x.device)
        _fw_reslnsk_kernel[(S,)](
            x, o1, p, bi, mb, mb2, tm, mb, mb, xo, no, xo, eps, S * N,
            N, int(tm.shape[1]), go, sho, sco, 1, True, True, True,
            int(sk), False, _resln_block(N), _PFLDH,
            num_warps=_RESLNSK_NW, num_stages=1,
        )
        return [xo, no]

    @_fw_res_ln2_sk.register_fake
    def _(x, o1, p, bi, mb, mb2, tm, go, sho, sco, eps, sk):
        S, N = int(x.shape[1]), int(x.shape[2])
        return [x.new_empty((1, S, N)), x.new_empty((1, S, N))]

    # -----------------------------------------------------------------------
    # r2_d2 LEVER 2 -- DELETE THE STRANDED ffn.2 SPLIT-U COMBINE.
    #
    # The prefill's ffn.2 down-projection ([98,14336] x [14336,3072]) is the one
    # site in the region where hipBLASLt picks a global-split-U solution, so it
    # dispatches TWO kernels: the Tensile GEMM and a separate
    # `Cijk_SB_BiasB_HAS_ScaleAlphaVec_PostGSU8_VW4` reduce+bias+cast, measured
    # 5.32 us x 29.9 = 0.159 ms per prefill call for what is a 2.4 MB fp32
    # reduction over 98 rows -- i.e. essentially pure dispatch floor.
    #
    # There is a free home for that work.  `res_ln_tail` (the r1_d1 layer
    # boundary consumer) reads ffn.2's bf16 output on the VERY NEXT node and is
    # itself floor-bound at 4.49 us.  `_fw_reslnsk_kernel` already knows how to
    # absorb a SPLIT_K fp32 combine + bias for exactly this row shape (it does
    # it twice per layer for the two o-projections), so the tail simply becomes
    # its HAS_O2=False / AF=1 / WRITE_X=True instantiation and the producer
    # becomes `pf_skmm`.  Net: one whole dispatch per layer disappears and the
    # library GEMM is replaced by the in-tree split-K one.
    #
    # NUMERICS.  Identical argument to the shipped `res_ln1_sk`/`res_ln2_sk`:
    # a fixed s = 0..SPLIT_K-1 fp32 accumulation out of a deterministic
    # workspace (no float atomics), plus the fp32 bias, rounded to bf16 EXACTLY
    # ONCE at the point hipBLASLt's own bf16 output was rounded.  The residual
    # `x1 + gate_mlp * ffn_out` and the LayerNorm that follow are byte for byte
    # the shipped `res_ln_tail` body.  Only the GEMM accumulation ORDER changes.
    #
    # Kill switch FASTWAM_PFFFN=0 -> the plain `_linear_splitk` + `res_ln_tail`
    # pair, byte for byte.
    # -----------------------------------------------------------------------
    @torch.library.custom_op("fastwam::res_ln_tail_sk", mutates_args=())
    def _fw_res_ln_tail_sk(x: torch.Tensor, p: torch.Tensor, bi: torch.Tensor,
                           mb: torch.Tensor, mb2: torch.Tensor,
                           tm: torch.Tensor, go: int, sho: int, sco: int,
                           eps: float, sk: int) -> List[torch.Tensor]:
        S, N = int(x.shape[1]), int(x.shape[2])
        xo = torch.empty((1, S, N), dtype=x.dtype, device=x.device)
        no = torch.empty((1, S, N), dtype=x.dtype, device=x.device)
        _fw_reslnsk_kernel[(S,)](
            x, x, p, bi, mb, mb2, tm, mb, mb, xo, no, xo, eps, S * N,
            N, int(tm.shape[1]), go, sho, sco, 1, False, True, True,
            int(sk), False, _resln_block(N), _PFLDH,
            num_warps=_RESLNSK_NW, num_stages=1,
        )
        return [xo, no]

    @_fw_res_ln_tail_sk.register_fake
    def _(x, p, bi, mb, mb2, tm, go, sho, sco, eps, sk):
        S, N = int(x.shape[1]), int(x.shape[2])
        return [x.new_empty((1, S, N)), x.new_empty((1, S, N))]

    # -----------------------------------------------------------------------
    # ROUND 3 (r3_d2) -- TAKE THE PREFILL CROSS-Q GEMM OFF hipBLASLt.
    #
    # SITE.  `q_r = ca.q(h3)` in `_pf_post_block`: M=98 N=3072 K=3072, the
    # IDENTICAL shape the two o-projections hand to `_fw_pfsk_kernel`.
    # hipBLASLt picks `MT32x64x256` (grid_m=4, so the 18.87 MB weight panel is
    # streamed FOUR times) and takes 13.14 us in-graph; `_fw_pfsk_kernel` moves
    # the same bytes ONCE, at 192 CTAs, in 8.85 us.  Round 2 closed library
    # selection on every other prefill GEMM (805 solutions raced, -0.7%); this
    # one was left because the split-K fp32 combine had nowhere free to live.
    #
    # WHERE THE COMBINE LIVES.  Not in a bare combine kernel (measured: split-K
    # + plain combine = 12.11 us vs hipBLASLt 12.22 us in a 30-launch graph --
    # a wash), and NOT in the kernel itself: the last-arriving-CTA fixup form
    # was re-priced on THIS shape as the direction asked and is a catastrophe,
    # not a marginal loss -- 17.9 us at SK4/BN64 and 43 us at SK8/BN32 against
    # 8.78 us for the plain kernel, with `.cv` loads making no difference.  The
    # `sem="release"` counter serialises the whole 192-CTA grid.  That confirms
    # `FASTWAM_SKFIX`'s denoise verdict on a shape with 3x more work per CTA.
    #
    # It lives instead in a combine that ABSORBS THE `norm_q` RMSNorm.  The
    # consumer today is `fastwam::qrms_attn`, which pays 10.79 us because it
    # re-reads the WHOLE 3072-wide q row in every one of its 48 head-CTAs to
    # rebuild the RMS denominator (its own comment prices that redundancy).
    # Feeding it fp32 SPLIT_K partials instead of a bf16 q would multiply that
    # redundant stream by 8 -- ~115 MB of L2 traffic -- so the fusion has to go
    # the other way: this kernel walks each row ONCE (98 CTAs, one row each),
    # sums the SPLIT_K fp32 partials, adds the bias, rounds to bf16 exactly
    # where the GEMM's own output was rounded, and computes the RMS denominator
    # from that row while it is still in registers.  The attention downstream is
    # then the PLAIN `fastwam::short_attn`, which needs no whole-row re-read.
    #
    #     before:  MT32x64x256 13.14  +  qrms_attn 10.79            = 23.93 us
    #     after :  pfsk 8.85 + qcomb + short_attn                   (measured)
    #
    # NUMERICS.  Identical to the shipped `res_ln1_sk` / `res_ln2_sk` argument:
    # a fixed s = 0..SPLIT_K-1 fp32 accumulation out of a deterministic
    # workspace (no float atomics), rounded to bf16 EXACTLY ONCE at the point
    # the hipBLASLt bf16 output was rounded, and the RMSNorm body is the same
    # fp32 chain `_fw_qrms_attn_kernel` transcribed from inductor
    # (rstd = rsqrt(sum(x*x)/HD + eps); out = (x*rstd)*w, no intermediate
    # round).  Only the GEMM accumulation ORDER changes.
    # -----------------------------------------------------------------------

    @triton.jit
    def _fw_qcomb_kernel(P, BI, WQ, O, MN, eps, N: tl.constexpr,
                         SKP: tl.constexpr, BLOCK: tl.constexpr,
                         H: tl.constexpr = 0):
        s = tl.program_id(0)
        r = tl.arange(0, BLOCK)
        m = r < N
        base = s * N + r
        if H:
            # r2_d2 hoist: BI and the RMSNorm weight are cold [N] vectors; in
            # the original order WQ is issued AFTER the cross-lane sum, with
            # nothing left to overlap its HBM round trip.
            bih = tl.load(BI + r, mask=m, other=0.0).to(tl.float32)
            wh = tl.load(WQ + r, mask=m, other=0.0).to(tl.float32)
        p = tl.load(P + base, mask=m, other=0.0)
        for i in tl.static_range(1, SKP):
            p += tl.load(P + i * MN + base, mask=m, other=0.0)
        if H:
            p += bih
        else:
            p += tl.load(BI + r, mask=m, other=0.0).to(tl.float32)
        x = p.to(tl.bfloat16).to(tl.float32)
        ss = tl.sum(tl.where(m, x * x, 0.0), 0)
        rstd = tl.rsqrt(ss / N + eps)
        if H:
            w = wh
        else:
            w = tl.load(WQ + r, mask=m, other=0.0).to(tl.float32)
        tl.store(O + base, ((x * rstd) * w).to(tl.bfloat16), mask=m)

    @torch.library.custom_op("fastwam::pf_qcomb", mutates_args=())
    def _fw_pf_qcomb(p: torch.Tensor, bi: torch.Tensor, wq: torch.Tensor,
                     eps: float, sk: int) -> torch.Tensor:
        S, N = int(p.shape[1]), int(p.shape[2])
        out = torch.empty((1, S, N), dtype=bi.dtype, device=p.device)
        _fw_qcomb_kernel[(S,)](
            p, bi, wq, out, S * N, float(eps), N, int(sk), _resln_block(N),
            _PFLDH, num_warps=_QCOMB_NW, num_stages=1,
        )
        return out

    @_fw_pf_qcomb.register_fake
    def _(p, bi, wq, eps, sk):
        return bi.new_empty((1, int(p.shape[1]), int(p.shape[2])))


_PFXQ = os.environ.get("FASTWAM_PFXQ", "1").lower() not in ("0", "false", "no", "off")
# r2_d2: route prefill ffn.2 through `pf_skmm` and fold its combine into the
# layer-boundary residual+LN tail (deletes the stranded PostGSU8 dispatch).
_PFFFN = os.environ.get("FASTWAM_PFFFN", "1").lower() not in ("0", "false", "no", "off")
_QCOMB_NW = int(os.environ.get("FASTWAM_QCOMBNW", "4"))


def _pfxq_ok(x, w, b, wq, k, v, num_heads) -> bool:
    """Envelope for the prefill cross-q split-K + fused combine/RMSNorm path."""
    if not (_PFXQ and _FA_BSD_OK):
        return False
    try:
        if wq is None or wq.dim() != 1 or not wq.is_contiguous():
            return False
        if not _pfsk_ok(x, w, b):
            return False
        if int(wq.shape[0]) != int(w.shape[0]):
            return False
        if int(w.shape[0]) != int(x.shape[2]):
            return False
        return _fa_eligible(x, k, v, num_heads)
    except Exception:  # pragma: no cover
        return False


def _pfffn_ok(x, w, b) -> bool:
    """Envelope for the prefill ffn.2 split-K producer + folded tail combine."""
    if not (_PFFFN and _PFSK and _FA_BSD_OK):
        return False
    try:
        if x is None or w is None or b is None:
            return False
        if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
            return False
        if b.dtype != torch.bfloat16 or b.dim() != 1:
            return False
        if x.dim() != 3 or int(x.shape[0]) != 1 or w.dim() != 2:
            return False
        if not x.is_contiguous() or not w.is_contiguous():
            return False
        if not b.is_contiguous():
            return False
        M, K = int(x.shape[1]), int(x.shape[2])
        N = int(w.shape[0])
        if int(w.shape[1]) != K or int(b.shape[0]) != N:
            return False
        BM, BN, BK, SK, NW, NS2 = _PFFFN_CFG
        if M < 64 or M > BM:
            return False
        if N % BN or K % (SK * BK):
            return False
        if SK > 16:
            return False
        if BN < 64 and NW >= 4:       # COMMANDMENT rule 7
            return False
        return True
    except Exception:  # pragma: no cover
        return False


def _pfsk_ok(x, w, b) -> bool:
    """Prefill-only envelope for the split-K o-projection."""
    if not (_PFSK and _FA_BSD_OK):
        return False
    try:
        if x is None or w is None or b is None:
            return False
        if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
            return False
        if b.dtype != torch.bfloat16 or b.dim() != 1:
            return False
        if x.dim() != 3 or int(x.shape[0]) != 1 or w.dim() != 2:
            return False
        if not x.is_contiguous() or not w.is_contiguous():
            return False
        if not b.is_contiguous():
            return False
        M, K = int(x.shape[1]), int(x.shape[2])
        N = int(w.shape[0])
        if int(w.shape[1]) != K or int(b.shape[0]) != N:
            return False
        BM, BN, BK, SK, NW, NS2 = _PFSK_CFG
        if M < 64 or M > BM:          # grid_m == 1, and disjoint from denoise M=32
            return False
        if N % BN or K % (SK * BK):
            return False
        if SK > 8:
            return False
        if BN < 64 and NW >= 4:       # COMMANDMENT rule 7
            return False
        return True
    except Exception:  # pragma: no cover
        return False


def _pfres_ok(x, mb, tm, n: int) -> bool:
    """Static-shape envelope for the prefill residual+LN kernels."""
    if not (_PFRES and _FA_BSD_OK):
        return False
    try:
        if x.dim() != 3 or int(x.shape[0]) != 1 or int(x.shape[2]) != n:
            return False
        if x.dtype != torch.bfloat16 or not x.is_contiguous():
            return False
        s_len = int(x.shape[1])
        if mb.dtype != torch.bfloat16 or tm.dtype != torch.bfloat16:
            return False
        if not mb.is_contiguous() or not tm.is_contiguous():
            return False
        if mb.dim() != 1 or int(mb.shape[0]) != 6 * n:
            return False
        if tm.dim() != 2 or int(tm.shape[0]) != s_len or int(tm.shape[1]) != 6 * n:
            return False
        return True
    except Exception:  # pragma: no cover
        return False


_PFQK = os.environ.get("FASTWAM_PFQK", "1").lower() not in ("0", "false", "no", "off")


def _pfqk_ok(sa, freqs, qkv2, n: int) -> bool:
    """Static-shape envelope for the video-prefill fused Q/K post kernel."""
    if not (_PFQK and _qkvpost_enabled()):
        return False
    try:
        if freqs is None or freqs.is_complex() or freqs.dtype != torch.float32:
            return False
        if not freqs.is_contiguous():
            return False
        if qkv2.dim() != 2 or qkv2.dtype != torch.bfloat16 or not qkv2.is_contiguous():
            return False
        s_len = int(qkv2.shape[0])
        if int(qkv2.shape[1]) != 3 * n:
            return False
        if int(freqs.shape[0]) != s_len:
            return False
        head = int(freqs.shape[-1])
        if head % 2 or int(freqs.numel()) != s_len * head:
            return False
        if n % head:
            return False
        w = getattr(sa, "_fw_qk_nw", None)
        if w is None or w.dim() != 2 or int(w.shape[0]) != 2 or int(w.shape[1]) != n:
            return False
        if getattr(sa, "_fw_qkv_h", None) != n:
            return False
        if getattr(sa, "_fw_qk_eps", None) is None:
            return False
        return True
    except Exception:  # pragma: no cover
        return False


@torch.no_grad()
def _build_qk_norm(expert) -> bool:
    """Stack each block's self_attn norm_q|norm_k weight into one [2, H] buffer."""
    ok = 0
    for b in (getattr(expert, "blocks", None) or []):
        sa = getattr(b, "self_attn", None)
        if sa is None or hasattr(sa, "_fw_qk_nw"):
            continue
        nq = getattr(sa, "norm_q", None)
        nk = getattr(sa, "norm_k", None)
        wq = getattr(nq, "weight", None)
        wk = getattr(nk, "weight", None)
        if wq is None or wk is None or wq.dim() != 1 or wq.shape != wk.shape:
            continue
        if float(getattr(nq, "eps", -1.0)) != float(getattr(nk, "eps", -2.0)):
            continue
        sa.register_buffer("_fw_qk_nw",
                           torch.stack([wq.detach(), wk.detach()], 0).contiguous(),
                           persistent=False)
        sa._fw_qk_eps = float(nq.eps)
        ok += 1
    return ok > 0


def _ctx_kv_slice(xkv, xk_all, i, h2):
    """Per-layer cross-attn K/V for `_apply_expert_post_block`."""
    if xkv is None:
        return None
    if xk_all is None:
        return xkv[..., i * h2:(i + 1) * h2]
    return (xk_all[i], xkv[..., i * h2 + h2 // 2:(i + 1) * h2])


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention. This will save memory but use more computation.")

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )
        
        logger.info(f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}")
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B")

    @staticmethod
    def _bias_payload(payload, dtype):
        """Payload with its bool cross-attn mask replaced by a hoisted additive bias."""
        if payload is None:
            return payload
        m = payload.get("mask")
        if m is None or m.dtype != torch.bool:
            return payload
        out = dict(payload)
        out["mask"] = _attn_bias(m.unsqueeze(1) if m.dim() == 3 else m, dtype)
        return out

    @staticmethod
    def _action_prologue_impl(expert, context, self_mask, ctx_mask, dtype):
        """Loop-invariant denoise-step inputs, computed ONCE per step.

        Kept OUT of `_forward_action_impl` deliberately. Inside that region
        inductor re-fuses both of these into their 30 per-layer consumers (it
        inlines a multi-user node whose users read disjoint slices), so the
        batched form measured 1 reduction + 30 pointwise kernels -- no better
        than the 30 it replaced. Behind a separate compile boundary the results
        are real buffers, the per-layer slices are plain views, and the cost
        collapses to 2 kernels + 1 for each mask.
        """
        xkv = _ctx_kv_all(expert, context)
        xk_all = _ctx_k_all(expert, xkv)
        return xkv, xk_all, _attn_bias(self_mask, dtype), _attn_bias(ctx_mask, dtype)

    def _action_prologue(self, expert, payload, self_mask, dtype):
        ctx = payload.get("context") if payload is not None else None
        cm = payload.get("mask") if payload is not None else None
        if cm is not None and cm.dim() == 3:
            cm = cm.unsqueeze(1)
        fn = self._action_prologue_impl
        if is_compile_enabled() and not self.training:
            fn = getattr(self, "_compiled_action_prologue", None)
            if fn is None:
                fn = torch.compile(
                    self._action_prologue_impl,
                    mode=os.environ.get("FASTWAM_COMPILE_MODE", "max-autotune-no-cudagraphs"),
                    dynamic=False,
                )
                self._compiled_action_prologue = fn
        xkv, xk_all, sbias, cbias = fn(expert, ctx, self_mask, cm, dtype)
        if payload is not None and cbias is not None:
            payload = dict(payload)
            payload["mask"] = cbias
        return xkv, xk_all, sbias, payload

    def _prologue_inputs_same(self, tag, tensors) -> bool:
        """True iff every episode-scoped prologue input is BIT-IDENTICAL to the
        PRIVATE CLONE kept from the last refresh.

        COMMANDMENT RULE 5.  This is CONTENT, not identity: we own the clone (so
        nothing outside can mutate the key under us) and we compare element-wise
        on every call.  The `_fw_episode` counter remains the outer gate at every
        call site -- this only ever runs when the counter says the inputs MAY
        have moved, so the memo is content AND counter, never content alone and
        never `data_ptr`.

        Costs one compare kernel per key plus exactly one host sync.  Returns
        False (and re-arms the clone) on the first call, on any shape/dtype
        change, and whenever a single bit differs.
        """
        if not _CTXMEMO:
            return False
        store = getattr(self, "_fw_pmemo", None)
        if store is None:
            store = {}
            self._fw_pmemo = store
        cur = [t for t in tensors if isinstance(t, torch.Tensor)]
        if not cur:
            return False
        try:
            # PACK every key into ONE flat byte buffer first.  Comparing the keys
            # one at a time costs 2 launches each (ne + any) plus the ORs; on the
            # ~4.4 us dispatch floor of this box that is more than the compare is
            # worth.  `torch.cat` of the byte views is a single kernel, so the
            # whole check is cat + ne + any = 3 launches regardless of how many
            # keys there are, and the packed buffer IS the private clone.
            packed = _pack_bytes(cur)
        except Exception:
            store.pop(tag, None)
            return False
        prev = store.get(tag)
        if prev is None or prev.numel() != packed.numel():
            store[tag] = packed
            return False
        try:
            changed = bool(torch.ne(prev, packed).any().item())   # the one sync
        except Exception:
            store.pop(tag, None)
            return False
        if changed:
            prev.copy_(packed)
            return False
        return True

    def _action_prologue_memo(self, expert, payload, self_mask, dtype):
        """`_action_prologue` memoized per EPISODE (COMMANDMENT rule 5: counter +
        CONTENT, never pointer). `context` and both masks are already treated as
        episode-scoped by the graph path, so nothing here is newly assumed to be
        constant."""
        ep = getattr(self, "_fw_episode", 0)
        key = (_sig(self_mask), self._payload_sig(payload), dtype)
        st = getattr(self, "_fw_prol", None)
        if st is not None and st[0] == key and st[1] != ep:
            # Counter moved (outer gate).  Ask CONTENT whether anything the
            # prologue reads actually changed; if not, keep the memo and just
            # re-stamp the counter.
            if self._prologue_inputs_same("eager", self._prologue_key_tensors(payload, self_mask)):
                st = (key, ep, st[2])
                self._fw_prol = st
        if st is None or st[0] != key or st[1] != ep:
            self._prologue_inputs_same("eager", self._prologue_key_tensors(payload, self_mask))
            st = (key, ep, self._action_prologue(expert, payload, self_mask, dtype))
            self._fw_prol = st
        return st[2]

    @staticmethod
    def _prologue_key_tensors(payload, self_mask):
        """Every tensor the prologue reads: the context, the cross mask, the
        self mask.  These and only these decide (xkv, xk_all, sbias, cbias)."""
        out = [self_mask]
        if payload is not None:
            for k in sorted(payload.keys()):
                v = payload[k]
                if isinstance(v, torch.Tensor):
                    out.append(v)
        return out

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return _attn3(q, k, v, self.num_heads, attn_mask)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
        ctx_kv: Optional[torch.Tensor] = None,
        t_mod: Optional[torch.Tensor] = None,
        next_block=None,
        nxt_sink=None,
        pf_mod=None,
        pf_tail=None,
    ) -> torch.Tensor:
        # PREFILL arm (r2_d1): hand-fused residual+LN post block.  `pf_mod` is
        # only ever supplied by `_prefill_one_layer*`; the denoise arm below
        # (r2_d0 skpost) is never reached with it set, and vice versa.
        if pf_mod is not None:
            fused = MoT._pf_post_block(
                block, residual_x, mixed_attn_out, gate_mlp,
                context_payload, ctx_kv, pf_mod, pf_tail,
            )
            if fused is not None:
                return fused
        # ---- site 1: self-attn o-proj.  skmm + skred + inductor's
        # residual/gate/norm3 consumer (3 launches) -> skmm + skpost (2).
        sa_o = block.self_attn.o
        q_pre = None
        mlp_input = None
        _ctx = None if context_payload is None else context_payload.get("context")
        _mod = None if t_mod is None else getattr(block, "modulation", None)
        # Sites 2 and 3 are COUPLED: site 2 hands its fp32 residual straight to
        # site 3, so either both fire for this layer or neither does.
        _nmod = None if next_block is None else getattr(next_block, "modulation", None)
        _fuse23 = bool(
            (_SKPOST & 2) and nxt_sink is not None and _nmod is not None
            and _mod is not None
            and getattr(next_block.norm1, "weight", None) is None
            and _skpost_ok3(block, mixed_attn_out, _nmod, t_mod)
        )
        if (
            (_SKPOST & 1) and _ctx is not None and _mod is not None
            and getattr(block.norm3, "weight", None) is not None
            and getattr(block.norm3, "bias", None) is not None
            and _skpost_ok(mixed_attn_out, sa_o.weight, residual_x, _mod, t_mod,
                           sa_o.bias)
        ):
            x, q_pre = torch.ops.fastwam.skpost.default(
                mixed_attn_out, sa_o.weight, sa_o.bias, residual_x,
                _mod, _mod, t_mod, block.norm3.weight, block.norm3.bias,
                2, -1, -1, float(block.norm3.eps),
            )
        else:
            x = block.gate(residual_x, gate_msa, _linear_splitk(mixed_attn_out, sa_o))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                ca = block.cross_attn
                kv_w = getattr(ca, "_fw_kv_w", None)
                if ctx_kv is not None or kv_w is not None:
                    # K and V both read `context`: one fused GEMM instead of two.
                    hc = ca._fw_kv_h
                    # Q-SIDE FUSION: hand the RAW q projection to the attention
                    # kernel and let it apply `norm_q`'s RMSNorm inside its own Q
                    # load (see `wan_video_dit.qrms_attention`). That deletes the
                    # standalone 32-CTA reduction launch inductor emits here --
                    # 4.38 us x 30 layers = 0.131 ms of the 2.92 ms denoise step,
                    # and the same kernel once per prefill layer. `qrms_attention`
                    # returns None outside its verified envelope, in which case
                    # the original expression runs byte for byte.
                    q_r = ca.q(block.norm3(x) if q_pre is None else q_pre)
                    if type(ctx_kv) is tuple:
                        # K already RMSNormed for every layer in one batched kernel.
                        k_c, v_c = ctx_kv
                    else:
                        kv = ctx_kv if ctx_kv is not None else F.linear(context, kv_w, ca._fw_kv_b)
                        k_c = ca.norm_k(kv[..., :hc])
                        v_c = kv[..., hc:]
                    a_c = None
                    if _qrms_attention is not None:
                        a_c = _qrms_attention(
                            q_r, ca.norm_q, k_c, v_c, ca.num_heads, context_mask,
                        )
                    if a_c is None:
                        a_c = _attn3(
                            ca.norm_q(q_r), k_c, v_c, ca.num_heads, context_mask,
                        )
                    # ---- site 2: cross-attn o-proj.  skmm + skred +
                    # inductor's residual + norm2-adaLN consumer -> 2 launches.
                    if (
                        _fuse23 and _mod is not None
                        and getattr(block.norm2, "weight", None) is None
                        and _skpost_ok(a_c, ca.o.weight, x, _mod, t_mod, ca.o.bias)
                    ):
                        # x2 leaves in FP32 on purpose: inductor never
                        # materialises this residual (kernel `..._split_4`
                        # computes it, drops it, and `..._split_6` recomputes
                        # it), so rounding it to bf16 here would insert a 30x
                        # repeated extra rounding on the main residual stream
                        # -- measured, that alone took the oracle from
                        # mean|d| 1.10e-02 to 2.46e-02 (gate 2.283e-02).
                        x, mlp_input = torch.ops.fastwam.skpost.default(
                            a_c, ca.o.weight, ca.o.bias, x, _mod, _mod, t_mod,
                            None, None, -1, 4, 3, float(block.norm2.eps), 1,
                        )
                    else:
                        x = x + _linear_splitk(a_c, ca.o)
                else:
                    x = x + block.cross_attn(block.norm3(x) if q_pre is None else q_pre,
                                             context, ctx_mask=context_mask)

        if mlp_input is None:
            mlp_input = adaln_forward(x, block.norm2, shift_mlp, scale_mlp)
        ffn = block.ffn
        if _SPLITK and len(ffn) == 3 and isinstance(ffn[2], nn.Linear):
            hidden = ffn[1](ffn[0](mlp_input))
            # ---- site 3: ffn down-proj.  Its inductor consumer also carries
            # the NEXT block's norm1 adaLN, so the fused op emits that too and
            # `_build_expert_attention_io` skips it next iteration.
            if _fuse23 and _skpost_ok(hidden, ffn[2].weight, x, _nmod, t_mod,
                                      ffn[2].bias):
                x, nxt = torch.ops.fastwam.skpost.default(
                    hidden, ffn[2].weight, ffn[2].bias, x, _mod, _nmod, t_mod,
                    None, None, 5, 1, 0, float(next_block.norm1.eps),
                )
                nxt_sink.append(nxt)
                return x
            ffn_out = _linear_splitk(hidden, ffn[2])
        else:
            ffn_out = ffn(mlp_input)
        x = block.gate(x, gate_mlp, ffn_out)
        if x.dtype != residual_x.dtype:   # never-taken safety net for the
            x = x.to(residual_x.dtype)    # fp32 site-2 residual
        return x

    @staticmethod
    def _pf_post_block(block, residual_x, mixed_attn_out, gate_mlp,
                       context_payload, ctx_kv, pf_mod, pf_tail=None):
        """PREFILL-ONLY post block with the two residual+LN passes hand-fused.

        Returns None (and the generic path runs, byte for byte) whenever any
        part of the envelope is not met.
        """
        mb, tm = pf_mod
        ca = getattr(block, "cross_attn", None)
        if ca is None or context_payload is None or type(ctx_kv) is not tuple:
            return None
        if context_payload.get("context") is None:
            return None
        n3 = getattr(block, "norm3", None)
        n2 = getattr(block, "norm2", None)
        if n3 is None or n2 is None:
            return None
        w3 = getattr(n3, "weight", None)
        b3 = getattr(n3, "bias", None)
        if w3 is None or b3 is None or getattr(n2, "weight", None) is not None:
            return None
        n = int(residual_x.shape[-1])
        if not _pfres_ok(residual_x, mb, tm, n):
            return None

        sao = block.self_attn.o
        if _pfsk_ok(mixed_attn_out, sao.weight, sao.bias):
            p1 = torch.ops.fastwam.pf_skmm.default(mixed_attn_out, sao.weight)
            h3, o_out = torch.ops.fastwam.res_ln1_sk.default(
                residual_x, p1, sao.bias, mb, tm, w3, b3, 2 * n, float(n3.eps),
                int(_PFSK_CFG[3]))
        else:
            o_out = _linear_splitk(mixed_attn_out, sao)
            h3 = torch.ops.fastwam.res_ln1.default(
                residual_x, o_out, mb, tm, w3, b3, 2 * n, float(n3.eps))

        context_mask = context_payload.get("mask")
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)
        k_c, v_c = ctx_kv
        a_c = None
        # ROUND 3 (r3_d2): own the cross-q GEMM in Triton -- split-K producer
        # + a combine that absorbs `norm_q`'s RMSNorm, so the attention that
        # follows is the plain (non-RMS-re-reading) kernel.  See the block
        # comment above `_fw_qcomb_kernel`.
        cq = getattr(ca, "q", None)
        nq = getattr(ca, "norm_q", None)
        wqn = getattr(nq, "weight", None) if nq is not None else None
        if cq is not None and _pfxq_ok(h3, getattr(cq, "weight", None),
                                       getattr(cq, "bias", None), wqn,
                                       k_c, v_c, ca.num_heads):
            m_c = _fa_mask2d(context_mask, int(h3.shape[1]), int(k_c.shape[1]))
            if m_c is not False:
                pq = torch.ops.fastwam.pf_skmm.default(h3, cq.weight)
                qn = torch.ops.fastwam.pf_qcomb.default(
                    pq, cq.bias, wqn, float(nq.eps), int(_PFSK_CFG[3]))
                a_c = torch.ops.fastwam.short_attn.default(
                    qn, k_c, v_c, m_c, int(ca.num_heads))
        if a_c is None:
            q_r = ca.q(h3)
            if _qrms_attention is not None:
                a_c = _qrms_attention(q_r, ca.norm_q, k_c, v_c, ca.num_heads, context_mask)
            if a_c is None:
                a_c = _attn3(ca.norm_q(q_r), k_c, v_c, ca.num_heads, context_mask)
        cao = ca.o
        if _pfsk_ok(a_c, cao.weight, cao.bias):
            p2 = torch.ops.fastwam.pf_skmm.default(a_c, cao.weight)
            x1, mlp_input = torch.ops.fastwam.res_ln2_sk.default(
                residual_x, o_out, p2, cao.bias, mb, mb, tm,
                2 * n, 3 * n, 4 * n, float(n2.eps), int(_PFSK_CFG[3]))
        else:
            c_out = _linear_splitk(a_c, cao)
            x1, mlp_input = torch.ops.fastwam.res_ln2.default(
                residual_x, o_out, c_out, mb, mb, tm,
                2 * n, 3 * n, 4 * n, float(n2.eps))

        ffn = block.ffn
        if _SPLITK and len(ffn) == 3 and isinstance(ffn[2], nn.Linear):
            f0 = ffn[0]
            if (isinstance(f0, nn.Linear)
                    and _pfgemm_ok(mlp_input, f0.weight, f0.bias, 2)):
                hidden = torch.ops.fastwam.pf_mm_gelu.default(
                    mlp_input, f0.weight, f0.bias)
            else:
                hidden = ffn[1](f0(mlp_input))
            f2 = ffn[2]
            if (pf_tail is not None
                    and _pfffn_ok(hidden, f2.weight, f2.bias)):
                nmb, teps = pf_tail
                p3 = torch.ops.fastwam.pf_skmm_big.default(hidden, f2.weight)
                return list(torch.ops.fastwam.res_ln_tail_sk.default(
                    x1, p3, f2.bias, mb, nmb, tm, 5 * n, 0, n, float(teps),
                    int(_PFFFN_CFG[3])))
            ffn_out = _linear_splitk(hidden, f2)
        else:
            ffn_out = ffn(mlp_input)
        if pf_tail is not None:
            nmb, teps = pf_tail
            return list(torch.ops.fastwam.res_ln_tail.default(
                x1, ffn_out, mb, nmb, tm, 5 * n, 0, n, float(teps)))
        return block.gate(x1, gate_mlp, ffn_out)

    @staticmethod
    def _pf_tail_mod(nb, t_mod):
        """(flat next-block modulation, next norm1 eps) or None outside envelope."""
        if not (_PFLNT and _PFRES):
            return None
        try:
            n1 = getattr(nb, "norm1", None)
            if n1 is None or getattr(n1, "weight", None) is not None:
                return None
            if getattr(n1, "bias", None) is not None:
                return None
            if t_mod.dim() != 4 or int(t_mod.shape[0]) != 1:
                return None
            base = nb.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
            return (base.reshape(-1), float(n1.eps))
        except Exception:  # pragma: no cover
            return None

    @staticmethod
    def _pf_mod(block, t_mod):
        """(flat block.modulation, [S, 6*D] t_mod) or None outside the envelope."""
        if not _PFRES:
            return None
        try:
            if t_mod.dim() != 4 or int(t_mod.shape[0]) != 1:
                return None
            if not t_mod.is_contiguous():
                return None
            base = block.modulation
            base = base.to(dtype=t_mod.dtype, device=t_mod.device)
            return (base.reshape(-1), t_mod.reshape(int(t_mod.shape[1]), -1))
        except Exception:  # pragma: no cover
            return None

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
        kv_dst=None,
        attn_input=None,
        pf=False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        if attn_input is None:
            # Denoise (r2_d0): the previous layer's fused ffn-down consumer
            # already emitted this (`skpost` site 3).  Prefill (r2_d1): the
            # previous layer's compiled tail emitted it.  Either way,
            # recomputing it would be a whole dispatch.
            attn_input = adaln_forward(x, block.norm1, shift_msa, scale_msa)

        sa = block.self_attn
        qkv_w = getattr(sa, "_fw_qkv_w", None)
        if kv_dst is not None and qkv_w is not None:
            # ONE kernel for v-copy + k RMSNorm/RoPE + q RMSNorm/RoPE, writing
            # straight into the joint K/V buffers (3 dispatches -> 1 per layer).
            qkv = F.linear(attn_input, qkv_w, sa._fw_qkv_b)
            kb, vb, kv_off = kv_dst
            q = torch.ops.fastwam.qkv_post.default(
                qkv.reshape(-1, qkv.shape[-1]), sa._fw_qk_nw,
                freqs.reshape(freqs.shape[0], -1), _rope_sign(freqs),
                kb, vb, int(kv_off), int(freqs.shape[-1]), sa._fw_qk_eps,
            )
            return (
                q, None, None, x, gate_msa, shift_mlp, scale_mlp, gate_mlp,
                bool(getattr(expert, "use_gradient_checkpointing", False)),
            )
        if qkv_w is not None:
            # ONE 1024->9216 GEMM instead of three 1024->3072 ones: 3x the tiles,
            # 1/3 the launches, bit-identical row block structure.
            h = sa._fw_qkv_h
            if pf and _pfgemm_ok(attn_input, qkv_w, sa._fw_qkv_b, 1):
                qkv = torch.ops.fastwam.pf_mm.default(
                    attn_input, qkv_w, sa._fw_qkv_b)
            else:
                qkv = F.linear(attn_input, qkv_w, sa._fw_qkv_b)
            qkv2 = qkv.reshape(-1, qkv.shape[-1]) if pf else None
            if pf and _pfqk_ok(sa, freqs, qkv2, h):
                # PREFILL ONLY: RMSNorm + RoPE for q AND k in one dispatch.
                q, k = torch.ops.fastwam.qk_post.default(
                    qkv2, sa._fw_qk_nw,
                    freqs.reshape(freqs.shape[0], -1), _rope_sign(freqs),
                    int(freqs.shape[-1]), sa._fw_qk_eps,
                )
                v = qkv[..., 2 * h:]
                return (
                    q, k, v, x, gate_msa, shift_mlp, scale_mlp, gate_mlp,
                    bool(getattr(expert, "use_gradient_checkpointing", False)),
                )
            q = sa.norm_q(qkv[..., :h])
            k = sa.norm_k(qkv[..., h:2 * h])
            v = qkv[..., 2 * h:]
        else:
            q = sa.norm_q(sa.q(attn_input))
            k = sa.norm_k(sa.k(attn_input))
            v = sa.v(attn_input)

        q = _rope_apply(q, freqs, block.num_heads)
        k = _rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
        ctx_kv: Optional[torch.Tensor] = None,
        t_mod: Optional[torch.Tensor] = None,
        next_block=None,
        nxt_sink=None,
        pf_mod=None,
        pf_tail=None,
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
            _ctx_kv=ctx_kv,
            _t_mod=t_mod,
            _next_block=next_block,
            _nxt_sink=nxt_sink,
            _pf_mod=pf_mod,
            _pf_tail=pf_tail,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
                ctx_kv=_ctx_kv,
                t_mod=_t_mod,
                next_block=_next_block,
                nxt_sink=_nxt_sink,
                pf_mod=_pf_mod,
                pf_tail=_pf_tail,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
        video_num_layers: int = 0,
    ) -> list[dict[str, torch.Tensor]]:
        """Prefill video branch once and cache per-layer K/V for action denoising.

        Args:
            video_tokens: Video tokens before layer 0, shape [B, Sv, D].
            video_freqs: Video RoPE frequencies, shape [Sv, 1, rope_dim].
            video_t_mod: Video time modulation tensor.
            video_context_payload: Optional dict for video cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sv, L] or [B, 1, Sv, L]
            video_attention_mask: Video self-attention mask, shape [Sv, Sv].
            video_num_layers: Number of video DiT layers to actually run. If 0 (or
                >= num_layers) all layers run. When < num_layers, only the first
                `video_num_layers` layers are computed and the remaining cache
                entries reuse the last computed layer's K/V ("video layer drafting").
                Defaults to 0, but falls back to the FASTWAM_VIDEO_DRAFT_LAYERS env
                flag when left at 0.

        Returns:
            Layer-wise cache list with length `num_layers`.
            Each entry contains:
                - `k`: video key tensor [B, Sv, H*Dh]
                - `v`: video value tensor [B, Sv, H*Dh]
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        if video_num_layers <= 0:
            from .draft_config import get_video_draft_layers
            video_num_layers = get_video_draft_layers()
        if video_num_layers <= 0 or video_num_layers > self.num_layers:
            n_run = self.num_layers
        else:
            n_run = video_num_layers

        # COMMANDMENT RULE 5: this is the once-per-episode entry point. Everything
        # cached downstream keys off this counter, so bump it BEFORE any work.
        self._fw_episode = getattr(self, "_fw_episode", 0) + 1
        # Module-level mirror of the same counter, for the step-scoped memo in
        # `ActionDiT.pre_dit` (another module, no reference to this MoT).
        FW_CALL_SEQ[0] += 1

        expert = self.mixtures["video"]
        _ensure_fused_proj(expert)
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []

        if is_compile_enabled() and not self.training:
            if not hasattr(self, "_compiled_prefill_one_layer"):
                # Prefill uses its OWN mode env; reduce-overhead (cudagraph) here is
                # unsafe — the prefill's per-layer output feeds the KV cache and gets
                # overwritten across calls ("accessing tensor output of CUDAGraphs
                # overwritten"). Keep prefill at no-cudagraphs even when the action
                # forward uses reduce-overhead (scoped cudagraph). Default matches
                # FASTWAM_COMPILE_MODE but never inherits reduce-overhead.
                _prefill_mode = os.environ.get(
                    "FASTWAM_PREFILL_COMPILE_MODE", "max-autotune-no-cudagraphs"
                )
                self._compiled_prefill_one_layer = torch.compile(
                    self._prefill_one_layer_t if _PFTAIL else self._prefill_one_layer,
                    mode=_prefill_mode,
                    dynamic=False,
                )
            layer_fn = self._compiled_prefill_one_layer
        else:
            layer_fn = self._prefill_one_layer_t if _PFTAIL else self._prefill_one_layer

        graphed = None
        if (_fw_graph_flags() & 2) and not self.training and _graphs_available(video_tokens):
            graphed = self._prefill_graph_call(
                expert, layer_fn, n_run, video_tokens, video_freqs, video_t_mod,
                video_attention_mask, video_context_payload,
            )
        if graphed is not None:
            kv_cache = graphed
        else:
            h2 = 2 * getattr(expert, "_fw_xkv_h", 0)
            xkv, xk_all, vam, vcp = self._action_prologue(
                expert, video_context_payload, video_attention_mask, x.dtype)
            ai = None
            for layer_idx in range(n_run):
                block = expert.blocks[layer_idx]
                if _PFTAIL:
                    nb = expert.blocks[layer_idx + 1] if layer_idx + 1 < n_run else block
                    x, ai, k, v = layer_fn(
                        expert, block, x, video_freqs, video_t_mod,
                        vam, vcp,
                        _ctx_kv_slice(xkv, xk_all, layer_idx, h2),
                        ai, nb,
                    )
                else:
                    x, k, v = layer_fn(
                        expert, block, x, video_freqs, video_t_mod,
                        vam, vcp,
                        _ctx_kv_slice(xkv, xk_all, layer_idx, h2),
                    )
                kv_cache.append({"k": k, "v": v})

        # Video layer drafting: the action expert cross-attends to a video K/V
        # entry at every one of its `num_layers` layers, so pad the truncated
        # cache by reusing the last computed layer's K/V for the skipped layers.
        if n_run < self.num_layers:
            last = kv_cache[-1]
            for _ in range(self.num_layers - n_run):
                kv_cache.append({"k": last["k"], "v": last["v"]})
        return kv_cache

    def _prefill_one_layer(self, expert, block, x, freqs, t_mod, attn_mask, ctx_payload,
                           ctx_kv=None):
        (
            q, k, v, residual_x, gate_msa,
            shift_mlp, scale_mlp, gate_mlp,
            use_gradient_checkpointing,
        ) = self._build_expert_attention_io(
            expert=expert, block=block, x=x, freqs=freqs, t_mod=t_mod, pf=True,
        )
        mixed = self._mixed_attention(
            q_cat=q, k_cat=k, v_cat=v, attention_mask=attn_mask,
        )
        x = self._apply_post_with_optional_checkpoint(
            block=block, residual_x=residual_x, gate_msa=gate_msa,
            shift_mlp=shift_mlp, scale_mlp=scale_mlp, gate_mlp=gate_mlp,
            use_gradient_checkpointing=use_gradient_checkpointing,
            mixed_slice=mixed, context_payload=ctx_payload, ctx_kv=ctx_kv,
            pf_mod=self._pf_mod(block, t_mod),
        )
        return x, k, v

    def _prefill_one_layer_t(self, expert, block, x, freqs, t_mod, attn_mask,
                             ctx_payload, ctx_kv=None, attn_input=None,
                             next_block=None):
        """`_prefill_one_layer` + the NEXT layer's adaln head fused into its tail.

        See the `_PFTAIL` note at the top of this file. Identical arithmetic;
        one fewer dispatch per layer boundary.
        """
        (
            q, k, v, residual_x, gate_msa,
            shift_mlp, scale_mlp, gate_mlp,
            use_gradient_checkpointing,
        ) = self._build_expert_attention_io(
            expert=expert, block=block, x=x, freqs=freqs, t_mod=t_mod,
            attn_input=attn_input, pf=True,
        )
        mixed = self._mixed_attention(
            q_cat=q, k_cat=k, v_cat=v, attention_mask=attn_mask,
        )
        nb = block if next_block is None else next_block
        res = self._apply_post_with_optional_checkpoint(
            block=block, residual_x=residual_x, gate_msa=gate_msa,
            shift_mlp=shift_mlp, scale_mlp=scale_mlp, gate_mlp=gate_mlp,
            use_gradient_checkpointing=use_gradient_checkpointing,
            mixed_slice=mixed, context_payload=ctx_payload, ctx_kv=ctx_kv,
            pf_mod=self._pf_mod(block, t_mod),
            pf_tail=self._pf_tail_mod(nb, t_mod),
        )
        if isinstance(res, list):
            x, nxt = res[0], res[1]
            return x, nxt, k, v
        x = res
        n_shift, n_scale = self._split_modulation(nb, t_mod)[:2]
        nxt = adaln_forward(x, nb.norm1, n_shift, n_scale)
        return x, nxt, k, v

    # ------------------------------------------------------------------
    # graph capture/replay plumbing
    # ------------------------------------------------------------------
    @staticmethod
    def _payload_sig(payload):
        if payload is None:
            return None
        return tuple(
            (k, _sig(v) if isinstance(v, torch.Tensor) else repr(v))
            for k, v in sorted(payload.items())
        )

    @staticmethod
    def _graph_warm() -> int:
        try:
            return int(os.environ.get("FASTWAM_GRAPH_WARM", "3"))
        except Exception:
            return 3

    @staticmethod
    def _mk_payload_slots(payload):
        """Return (mirror_payload, {key: slot}) or (None, None) on failure."""
        if payload is None:
            return None, {}
        mirror = dict(payload)
        slots = {}
        for k, v in payload.items():
            if isinstance(v, torch.Tensor):
                s = _mk_slot(v)
                if s is None:
                    return None, None
                slots[k] = s
                mirror[k] = s[0]
        return mirror, slots

    def _prefill_graph_call(self, expert, layer_fn, n_run, vt, vf, vtm, vam, vcp):
        if getattr(self, "_pf_g_off", False):
            return None
        key = (
            n_run, _sig(vt), _sig(vf), _sig(vtm), _sig(vam), self._payload_sig(vcp),
        )
        st = getattr(self, "_pf_g", None)
        if st is not None and st["key"] != key:
            st = None
            self._pf_g = None
            self._pf_warm = 0
            if getattr(self, "_pf_builds", 0) >= 4:
                self._pf_g_off = True
                return None
        if st is None:
            n = getattr(self, "_pf_warm", 0) + 1
            self._pf_warm = n
            if n <= self._graph_warm():
                return None  # let torch.compile + autotune settle first
            self._pf_builds = getattr(self, "_pf_builds", 0) + 1
            st = self._pf_capture(key, expert, layer_fn, n_run, vt, vf, vtm, vam, vcp)
            if st is None:
                self._pf_g_off = True
                return None
            self._pf_g = st
        _fill_slot(st["tok"], vt)
        _fill_slot(st["freqs"], vf)
        _fill_slot(st["tmod"], vtm)
        if st.get("pslots"):
            # LEVER A, video side.  The prologue lives OUTSIDE the captured body
            # now (its results are static graph inputs), so it is per-EPISODE
            # work, not per-call work -- and `attn`/`ctx` are its only consumers,
            # so their fills ride along.  The `_fw_episode` counter is the outer
            # gate (it moved: this IS the once-per-call entry point that bumps
            # it), CONTENT decides.  0.232 ms/call when it hits.
            ep = getattr(self, "_fw_episode", 0)
            if st.get("ep") != ep:
                if not self._prologue_inputs_same(
                        "video", self._prologue_key_tensors(vcp, vam)):
                    _fill_slot(st["attn"], vam)
                    for k, slot in st["ctx"].items():
                        _fill_slot(slot, vcp[k])
                    self._refresh_prologue_slots(expert, st, vt.dtype)
                st["ep"] = ep
        else:
            _fill_slot(st["attn"], vam)
            for k, slot in st["ctx"].items():
                _fill_slot(slot, vcp[k])
        st["graph"].replay()
        return [{"k": k, "v": v} for k, v in st["kv"]]

    def _pf_capture(self, key, expert, layer_fn, n_run, vt, vf, vtm, vam, vcp):
        try:
            tok = _mk_slot(vt)
            freqs = _mk_slot(vf)
            tmod = _mk_slot(vtm)
            attn = _mk_slot(vam)
            if any(s is None for s in (tok, freqs, tmod, attn)):
                return None
            mirror, ctx = self._mk_payload_slots(vcp)
            if ctx is None:
                return None
            _fill_slot(tok, vt)
            _fill_slot(freqs, vf)
            _fill_slot(tmod, vtm)
            _fill_slot(attn, vam)
            for k, slot in ctx.items():
                _fill_slot(slot, vcp[k])
            blocks = [expert.blocks[i] for i in range(n_run)]

            h2 = 2 * getattr(expert, "_fw_xkv_h", 0)

            # LEVER A, video side.  Hoist the prologue OUT of the captured body
            # into its own static slots, exactly the way `_ac_capture` already
            # does for the action expert.  A graph replays every kernel it
            # captured, so as long as the 1024->L*2H cross-K|V GEMM sits inside
            # the body it is re-run on every call; outside it, it is refreshed
            # only when its inputs actually change (see `_prefill_graph_call`).
            # Bit-exact: the graph reads the same values from a static buffer
            # instead of from a freshly-produced one.
            pslots = pmirror = None
            if _PFHOIST:
                pro = self._prologue_slots(expert, mirror, attn[0], vt.dtype)
                # Only hoist when the fused cross-K|V slab is live.  If `xkv` is
                # None the per-layer path reads `context` NUMERICALLY inside the
                # body, and a stale ctx slot would then be a real staleness bug
                # -- so in that (never-taken on this tree) configuration keep the
                # prologue inside the graph and refill every call.
                if pro is not None and pro[1].get("xkv") is not None:
                    pslots, pmirror = pro

            def _body():
                x = tok[0]
                out = []
                if pmirror is not None:
                    xkv, xk_all = pmirror["xkv"], pmirror["xk_all"]
                    vam, vcp = pmirror["sbias"], pmirror["payload"]
                else:
                    xkv, xk_all, vam, vcp = self._action_prologue(
                        expert, mirror, attn[0], x.dtype)
                ai = None
                for i, b in enumerate(blocks):
                    if _PFTAIL:
                        nb = blocks[i + 1] if i + 1 < len(blocks) else b
                        x, ai, kk, vv = layer_fn(
                            expert, b, x, freqs[0], tmod[0], vam, vcp,
                            _ctx_kv_slice(xkv, xk_all, i, h2), ai, nb,
                        )
                    else:
                        x, kk, vv = layer_fn(
                            expert, b, x, freqs[0], tmod[0], vam, vcp,
                            _ctx_kv_slice(xkv, xk_all, i, h2),
                        )
                    out.append((kk, vv))
                return out

            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    _body()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                kv = _body()
            logger.info("FastWAM: captured video-prefill graph (%d layers, "
                        "prologue hoisted=%s).", n_run, pslots is not None)
            return {
                "key": key, "graph": g, "tok": tok, "freqs": freqs, "tmod": tmod,
                "attn": attn, "ctx": ctx, "kv": kv,
                "pslots": pslots, "mirror": mirror,
                "ep": getattr(self, "_fw_episode", 0),
            }
        except Exception as exc:  # pragma: no cover - capture refusal path
            logger.warning("FastWAM: video-prefill graph capture refused (%r); "
                           "falling back to eager dispatch.", exc)
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            return None

    def _action_graph_call(self, expert, action_tokens, action_freqs, action_t_mod,
                           payload, video_kv_cache, action_attention_mask,
                           video_seq_len, n_layers, joint_kv=False):
        if getattr(self, "_ac_g_off", False):
            return None
        kv0 = video_kv_cache[0]
        key = (
            n_layers, n_layers if n_layers > 0 else self.num_layers,
            int(video_seq_len), len(video_kv_cache), bool(joint_kv),
            _sig(action_tokens), _sig(action_freqs), _sig(action_t_mod),
            _sig(action_attention_mask), self._payload_sig(payload),
            _sig(kv0["k"]), _sig(kv0["v"]),
        )
        st = getattr(self, "_ac_g", None)
        if st is not None and st.get("kvslab") is not None \
                and st["kvslab"] is not self._kv_static_slab(video_kv_cache):
            # The aliased slab was reallocated (or the caller stopped using it):
            # the captured graph points at freed memory. Drop it and re-capture.
            st = None
            self._ac_g = None
            self._ac_warm = 0
        if st is not None and st["key"] != key:
            # Shape/stride/dtype of the region changed: the captured graph is not
            # valid for it. Drop it and re-capture (bounded, so a caller that
            # alternates shapes degrades to the eager path instead of leaking
            # one graph pool per call).
            st = None
            self._ac_g = None
            self._ac_warm = 0
            if getattr(self, "_ac_builds", 0) >= 4:
                self._ac_g_off = True
                return None
        if st is None:
            n = getattr(self, "_ac_warm", 0) + 1
            self._ac_warm = n
            if n <= self._graph_warm():
                return None
            self._ac_builds = getattr(self, "_ac_builds", 0) + 1
            st = self._ac_capture(key, expert, action_tokens, action_freqs, action_t_mod,
                                  payload, video_kv_cache, action_attention_mask,
                                  video_seq_len, n_layers, joint_kv)
            if st is None:
                self._ac_g_off = True
                return None
            self._ac_g = st

        # ---- per-episode refresh (COMMANDMENT RULE 5) --------------------
        ep = getattr(self, "_fw_episode", 0)
        if st["ep"] != ep:
            if st.get("kvslab") is None:
                dsts, srcs = [], []
                for i, slot in enumerate(st["kv"]):
                    entry = video_kv_cache[i]
                    dsts.append(slot[0][1])
                    srcs.append(entry["k"])
                    dsts.append(slot[1][1])
                    srcs.append(entry["v"])
                if _FOREACH_COPY is not None:
                    _FOREACH_COPY(dsts, srcs)
                else:
                    for d, s in zip(dsts, srcs):
                        d.copy_(s)
            # LEVER A.  The prologue (the 30-layer fused cross-attn K|V GEMM +
            # the batched cross-K RMSNorm + both attention biases) is a pure
            # function of `context` and the two masks.  The counter above is the
            # OUTER gate; CONTENT decides whether the 0.091 ms of work is
            # actually needed.  Skipping it also skips the slot fills that only
            # exist to feed it.
            if not self._prologue_inputs_same(
                    "action", self._prologue_key_tensors(payload, action_attention_mask)):
                _fill_slot(st["attn"], action_attention_mask)
                for k, slot in st["ctx"].items():
                    _fill_slot(slot, payload[k])
                self._refresh_prologue_slots(expert, st, action_tokens.dtype)
            st["ep"] = ep

        _fill_slot(st["tok"], action_tokens)
        _fill_slot(st["tmod"], action_t_mod)
        # LEVER B.  `action_freqs` is the RoPE table `ActionDiT.pre_dit` caches
        # in `_freqs_dev_cache`: the identical tensor object at every step of
        # every call, so re-copying it into the graph slot is a 4.2 us DtoD
        # memop that moves bytes that are already there.  Skip it while the
        # source object AND its `_version` are unchanged -- we keep a strong
        # reference, so the object cannot be freed and its address recycled
        # under us, and a re-capture builds a fresh `st` with no memo at all.
        fsrc = st.get("fsrc") if _STEPMEMO else None
        if fsrc is None or fsrc[0] is not action_freqs or fsrc[1] != action_freqs._version:
            _fill_slot(st["freqs"], action_freqs)
            st["fsrc"] = (action_freqs, action_freqs._version)
        st["graph"].replay()
        return st["out"]

    def _prologue_slots(self, expert, mirror, self_mask, dtype):
        """Static slots holding the per-episode prologue results (or None)."""
        vals = self._action_prologue(expert, mirror, self_mask, dtype)
        xkv, xk_all, sbias, payload2 = vals
        cbias = payload2.get("mask") if payload2 is not None else None
        slots = {}
        views = {}
        for name, t in (("xkv", xkv), ("xk_all", xk_all), ("sbias", sbias), ("cbias", cbias)):
            if t is None:
                views[name] = None
                continue
            sl = _mk_slot(t)
            if sl is None:
                return None
            _fill_slot(sl, t)
            slots[name] = sl
            views[name] = sl[0]
        pm = None
        if payload2 is not None:
            pm = dict(payload2)
            if views["cbias"] is not None:
                pm["mask"] = views["cbias"]
        return slots, {"payload": pm, "sbias": views["sbias"],
                       "xkv": views["xkv"], "xk_all": views["xk_all"]}

    def _refresh_prologue_slots(self, expert, st, dtype):
        slots = st.get("pslots")
        if not slots:
            return
        xkv, xk_all, sbias, payload2 = self._action_prologue(
            expert, st["mirror"], st["attn"][0], dtype)
        cbias = payload2.get("mask") if payload2 is not None else None
        for name, t in (("xkv", xkv), ("xk_all", xk_all), ("sbias", sbias), ("cbias", cbias)):
            sl = slots.get(name)
            if sl is not None and t is not None:
                _fill_slot(sl, t)

    def _ac_capture(self, key, expert, action_tokens, action_freqs, action_t_mod,
                    payload, video_kv_cache, action_attention_mask,
                    video_seq_len, n_layers, joint_kv=False):
        try:
            tok = _mk_slot(action_tokens)
            freqs = _mk_slot(action_freqs)
            tmod = _mk_slot(action_t_mod)
            attn = _mk_slot(action_attention_mask)
            if any(s is None for s in (tok, freqs, tmod, attn)):
                return None
            mirror, ctx = self._mk_payload_slots(payload)
            if ctx is None:
                return None
            # ALIAS, don't mirror, when the operands are already our own
            # per-episode-refreshed joint slab (see _kv_joint_buffers).  Those
            # buffers are allocated once per (shape, dtype, device) signature and
            # are held by `self._fw_kvf` for the module's lifetime, so the graph
            # can read them in place -- which deletes the 2 multi-tensor copies
            # (96 MB, 0.052 ms/call) that re-mirrored the whole [B, Sv+Sa, D]
            # slab, action tail included, into a second static buffer.
            # The alias is validated on EVERY call by python-object identity of
            # the slab tensor (`kvslab`), not by data_ptr: a new slab object can
            # only come from a new allocation, and the graph is dropped when it
            # does.  Content is still refreshed strictly by the _fw_episode
            # counter (COMMANDMENT rule 5).
            kvslab = self._kv_static_slab(video_kv_cache)
            kv_slots = []
            kv_mirror = []
            if kvslab is not None:
                kv_mirror = [{"k": e["k"], "v": e["v"]} for e in video_kv_cache]
            else:
                for entry in video_kv_cache:
                    sk = _mk_slot(entry["k"])
                    sv = _mk_slot(entry["v"])
                    if sk is None or sv is None:
                        return None
                    _fill_slot(sk, entry["k"])
                    _fill_slot(sv, entry["v"])
                    kv_slots.append((sk, sv))
                    kv_mirror.append({"k": sk[0], "v": sv[0]})
            _fill_slot(tok, action_tokens)
            _fill_slot(freqs, action_freqs)
            _fill_slot(tmod, action_t_mod)
            _fill_slot(attn, action_attention_mask)
            for k, slot in ctx.items():
                _fill_slot(slot, payload[k])

            fn = getattr(self, "_compiled_forward_action", None) or self._forward_action_impl

            # The prologue depends only on `context` / the masks, which the graph
            # already treats as EPISODE-scoped static inputs. Keeping it OUTSIDE the
            # captured body turns its ~3 kernels + the 1024->184320 cross-K|V GEMM
            # from per-STEP into per-EPISODE work; the results live in their own
            # static slots, refreshed by the same `_fw_episode` counter that
            # refreshes the video K/V (COMMANDMENT rule 5 -- never by data_ptr).
            pro = self._prologue_slots(expert, mirror, attn[0], tok[0].dtype)
            if pro is None:
                return None
            pslots, pmirror = pro

            def _body():
                return fn(expert, tok[0], freqs[0], tmod[0], pmirror["payload"], kv_mirror,
                          pmirror["sbias"], video_seq_len, n_layers, joint_kv,
                          pmirror["xkv"], pmirror["xk_all"])

            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    _body()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = _body()
            logger.info("FastWAM: captured action-denoise graph (%d layers).", n_layers)
            return {
                "key": key, "graph": g, "tok": tok, "freqs": freqs, "tmod": tmod,
                "attn": attn, "ctx": ctx, "kv": kv_slots, "out": out,
                "pslots": pslots, "mirror": mirror, "kvslab": kvslab,
                "ep": getattr(self, "_fw_episode", 0),
            }
        except Exception as exc:  # pragma: no cover - capture refusal path
            logger.warning("FastWAM: action graph capture refused (%r); "
                           "falling back to eager dispatch.", exc)
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            return None

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        num_layers: int = 0,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        # Use the action query rows from the joint [video+action] mask.
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        expert = self.mixtures["action"]
        _ensure_fused_proj(expert)

        joint_kv = False
        if _KVBUF and not self.training:
            bufs = self._kv_joint_buffers(video_kv_cache, int(video_seq_len), action_seq_len)
            if bufs is not None:
                video_kv_cache = bufs
                joint_kv = True

        if is_compile_enabled() and not self.training:
            if not hasattr(self, "_compiled_forward_action"):
                self._compiled_forward_action = torch.compile(
                    self._forward_action_impl,
                    mode=os.environ.get("FASTWAM_COMPILE_MODE", "max-autotune-no-cudagraphs"),
                    dynamic=False,
                )
            if (_fw_graph_flags() & 1) and _graphs_available(action_tokens):
                out = self._action_graph_call(
                    expert, action_tokens, action_freqs, action_t_mod,
                    action_context_payload, video_kv_cache,
                    action_attention_mask, video_seq_len, num_layers, joint_kv,
                )
                if out is not None:
                    return out
            xkv, xk_all, sbias, payload2 = self._action_prologue_memo(
                expert, action_context_payload, action_attention_mask, action_tokens.dtype)
            return self._compiled_forward_action(
                expert, action_tokens, action_freqs, action_t_mod,
                payload2, video_kv_cache,
                sbias, video_seq_len, num_layers, joint_kv, xkv, xk_all,
            )
        return self._forward_action_impl(
            expert, action_tokens, action_freqs, action_t_mod,
            action_context_payload, video_kv_cache,
            action_attention_mask, video_seq_len, num_layers, joint_kv,
        )

    def _kv_static_slab(self, video_kv_cache):
        """The persistent joint-KV slab iff `video_kv_cache` IS its view list."""
        st = getattr(self, "_fw_kvf", None)
        if st is None or st.get("slab") is None:
            return None
        if st["bufs"] is not video_kv_cache:
            return None
        return st["slab"]

    def _kv_joint_buffers(self, video_kv_cache, sv: int, sa: int):
        """Per-layer [B, Sv+Sa, D] K/V buffers with the video prefix pre-filled.

        LAUNCH-COUNT LEVER (round 3, un-regioned residue).  The prefix refresh
        used to be 2*L separate `copy_`s -- 60 launches / 0.244 ms per CALL on
        the deployed shape, and it is PER-CALL FIXED so BOTH scored cases pay
        it.  `torch._foreach_copy_` cannot take its multi-tensor fast route
        here because half the sources (the prefill's `v`) are column slices of
        the fused q|k|v GEMM output and therefore non-contiguous, so it degrades
        to one `copy_` per tensor: 30 Memcpy-DtoD + 30 elementwise kernels, all
        of them on the ~4.4 us dispatch floor (36 MB moved in 0.244 ms = 300
        GB/s against a 3072 GB/s in-run ceiling -- pure dispatch, not bandwidth).

        Fix, in two launches: (1) allocate all 2*L buffers as views of ONE
        contiguous [2L, B, Sv+Sa, D] slab, so every prefix destination is a
        single strided view `slab[:, :, :Sv, :]`; (2) `torch.cat` the 2*L
        sources into one contiguous [B, 2L*Sv, D] staging tensor (ATen's
        batched cat copies all of them in one kernel, non-contiguous inputs
        included) and copy that into the strided destination.

        Costs one extra 36 MB round-trip through the staging tensor (~0.02 ms at
        the measured ceiling) to delete ~58 dispatches (~0.24 ms).  Shapes,
        strides, dtypes and CONTENTS of the returned buffers are unchanged, so
        `_mk_slot`/graph capture see exactly the same operands -- BIT-EXACT.

        COMMANDMENT RULE 5 is untouched: the refresh is still gated strictly on
        the `_fw_episode` COUNTER, never on data_ptr/_version, and the staging
        tensor is rebuilt from the live cache every episode.

        Kill-switch: FASTWAM_KVSLAB=0 restores the per-tensor copy loop.
        """
        k0 = video_kv_cache[0]["k"]
        if k0.dim() != 3 or int(k0.shape[1]) != sv:
            return None
        nb = 2 * len(video_kv_cache)
        b, d = int(k0.shape[0]), int(k0.shape[2])
        sig = (len(video_kv_cache), sv, sa, b, d, k0.dtype, k0.device)
        st = getattr(self, "_fw_kvf", None)
        if st is None or st["sig"] != sig:
            slab = None
            if _KVSLAB and b == 1:
                slab = torch.empty(nb, b, sv + sa, d, dtype=k0.dtype, device=k0.device)
                bufs = [{"k": slab[2 * i], "v": slab[2 * i + 1]}
                        for i in range(len(video_kv_cache))]
            else:
                bufs = []
                for _ in video_kv_cache:
                    bufs.append({
                        "k": torch.empty(b, sv + sa, d, dtype=k0.dtype, device=k0.device),
                        "v": torch.empty(b, sv + sa, d, dtype=k0.dtype, device=k0.device),
                    })
            st = {"sig": sig, "bufs": bufs, "ep": None, "slab": slab}
            self._fw_kvf = st
        ep = getattr(self, "_fw_episode", 0)
        if st["ep"] != ep:
            slab = st.get("slab")
            srcs = []
            for entry in video_kv_cache:
                srcs.append(entry["k"]); srcs.append(entry["v"])
            if slab is not None:
                staged = torch.cat(srcs, dim=1)          # [1, nb*sv, d], contiguous
                slab[:, :, :sv, :].copy_(staged.view(nb, b, sv, d))
            else:
                dsts = []
                for buf in st["bufs"]:
                    dsts.append(buf["k"][:, :sv]); dsts.append(buf["v"][:, :sv])
                if _FOREACH_COPY is not None:
                    _FOREACH_COPY(dsts, srcs)
                else:
                    for dd, sc in zip(dsts, srcs):
                        dd.copy_(sc)
            st["ep"] = ep
        return st["bufs"]

    def _forward_action_impl(
        self, expert, action_tokens, action_freqs, action_t_mod,
        action_context_payload, video_kv_cache, action_attention_mask,
        video_seq_len, num_layers=0, joint_kv=False, xkv=None, xk_all=None,
    ):
        x = action_tokens
        n = num_layers if num_layers > 0 else self.num_layers
        if xkv is None and action_context_payload is not None:
            xkv = _ctx_kv_all(expert, action_context_payload.get("context"))
            xk_all = _ctx_k_all(expert, xkv)
        xkv_h2 = 2 * getattr(expert, "_fw_xkv_h", 0)
        joint_kv = bool(joint_kv)
        fuse_post = joint_kv and _qkvpost_enabled() and _qkvpost_ok(
            expert, action_freqs, video_kv_cache[0], video_seq_len,
            int(action_tokens.shape[1]),
        )
        nxt_in = None
        for layer_idx in range(n):
            block = expert.blocks[layer_idx]
            next_block = expert.blocks[layer_idx + 1] if layer_idx + 1 < n else None
            layer_cache = video_kv_cache[layer_idx]
            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            (
                q_action, k_action, v_action,
                residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert, block=block, x=x,
                freqs=action_freqs, t_mod=action_t_mod,
                kv_dst=(k_video, v_video, video_seq_len) if fuse_post else None,
                attn_input=nxt_in,
            )
            if k_action is None:
                k_cat, v_cat = k_video, v_video
            elif joint_kv:
                # Prefix already holds the video K/V for this episode; write the
                # action tail in place and attend over the whole buffer.
                k_video[:, video_seq_len:] = k_action
                v_video[:, video_seq_len:] = v_action
                k_cat, v_cat = k_video, v_video
            else:
                k_cat = torch.cat([k_video, k_action], dim=1)
                v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action, k_cat=k_cat, v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            sink = []
            x = self._apply_post_with_optional_checkpoint(
                block=block, residual_x=residual_x, gate_msa=gate_msa,
                shift_mlp=shift_mlp, scale_mlp=scale_mlp, gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed, context_payload=action_context_payload,
                ctx_kv=_ctx_kv_slice(xkv, xk_all, layer_idx, xkv_h2),
                t_mod=action_t_mod, next_block=next_block, nxt_sink=sink,
            )
            nxt_in = sink[0] if len(sink) == 1 else None
        return x

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ):
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        tokens_all = {k: v for k, v in embeds_all.items()}

        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }

            # 3. concat all tokens for mixed attention
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = q_cat.shape[1]
            if attention_mask.shape[0] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[0]} vs tokens={total_seq}"
                )

            mixed = self._mixed_attention(q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask)

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                # 4. split mixed attention output and apply post-attention blocks for each expert
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                start = end

        return tokens_all

    # ------------------------------------------------------------------
    # APPENDED (round-2 host_runtime): mode-cached `train()`.
    #
    # `FastWAM.infer_action` (IMMUTABLE) opens with an unconditional
    # `self.eval()`.  `nn.Module.train(mode)` sets `self.training` and then
    # recurses over `self.children()` unconditionally -- 3147 module visits on
    # this pipeline (1735 unique modules; `video_expert`/`action_expert` are
    # registered BOTH directly on FastWAM and inside `MoT.mixtures`, so their
    # subtrees are walked twice).  Measured in-container on the real loaded
    # model: `model.eval()` = 3.225 ms median over 30 reps, with the GPU idle.
    # That is 19% of episode_1step and 6% of episode_10step.
    #
    # This override skips the recursion when every flag underneath already
    # holds the requested value, so it can only avoid RE-WRITING `.training`
    # flags that are already correct -- numerically a no-op by construction.
    #
    # Correctness of the cache (why it cannot go stale):
    #   * `_fw_train_mode` starts as *absent* -> `.get(...)` returns None ->
    #     the FIRST call in each direction always falls through to
    #     `nn.Module.train`, which really writes every descendant flag.
    #   * a genuine transition in EITHER direction is handled (the cached value
    #     is compared against the requested one, not assumed to be eval).
    #   * every node that short-circuits caches ITS OWN mode, so flipping one
    #     expert independently still re-walks exactly that expert on the next
    #     `eval()` while the others stay short-circuited.
    #   * it is not keyed on tensor identity / data_ptr / _version, and holds
    #     no episode-varying state, so the cross-episode staleness class the
    #     COMMANDMENT warns about does not apply.
    #
    # `MoT` covers 1414 of the 3147 visits and `WanVideoVAE38` (patched in
    # wan_video_vae.py) another 319.  The remaining 1412 are the two experts
    # reached DIRECTLY from FastWAM, ahead of `mot` in its `_modules` order;
    # they live in files this round does not own, so they get an equivalent
    # per-instance shim installed from here (the experts are MoT's own
    # children).  The shim is skipped if their class already defines `train`,
    # so it composes with any other patch that adds one.
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        mode = bool(mode)
        if "_fw_train_shimmed" not in self.__dict__:
            self.__dict__["_fw_train_shimmed"] = True
            try:
                for _expert in self.mixtures.values():
                    if type(_expert).train is nn.Module.train and "train" not in _expert.__dict__:
                        _expert.train = _fw_mode_cached_train.__get__(_expert, type(_expert))
            except Exception as _exc:  # pragma: no cover - never break eval()
                logger.warning("FastWAM: could not install expert train() shim (%r)", _exc)
        if self.__dict__.get("_fw_train_mode", None) is mode:
            return self
        nn.Module.train(self, mode)
        self.__dict__["_fw_train_mode"] = mode
        return self


def _fw_mode_cached_train(self, mode: bool = True):
    """Per-instance `train()` with the same mode cache as `MoT.train` above."""
    mode = bool(mode)
    if self.__dict__.get("_fw_train_mode", None) is mode:
        return self
    nn.Module.train(self, mode)
    self.__dict__["_fw_train_mode"] = mode
    return self
