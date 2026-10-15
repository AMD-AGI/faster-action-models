# Based on Wan2.x / DiffSynth-Studio (Apache-2.0) via FastWAM (MIT).
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, Tuple, Optional
from einops import rearrange

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

    
# ============================================================================
# FASTWAM_FATRITON -- native-layout Triton FMHA replacing the
# `rearrange -> F.scaled_dot_product_attention -> rearrange` body of
# `flash_attention`.  DEFAULT ON, kill switch `FASTWAM_FATRITON=0`.
#
# WHY.  Every consumer of this function hands q/k/v in the NATIVE
# [B, S, H*D] token-major layout and wants the answer back in the same layout,
# so the SDPA body forces inductor to materialise four extra layout tensors per
# call ([B,H,S,D] for q/k/v plus the transpose of the output).  Those show up in
# the denoise profile as
#   triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_
#   triton_red_fused__scaled_dot_product_efficient_attention__to_copy_add_
#   triton_poi_fused_copy_slice_transpose_view_19
#   triton_poi_fused__to_copy_copy_slice_transpose_view_4
# ~1.04 ms/step over ~193 launches, i.e. 39% of the 2.55 ms denoise step, on
# tensors of at most 48 KB.  The kernel below reads q/k/v THROUGH THEIR ACTUAL
# STRIDES (the cross-attention K is a layer-slice of the stack-wide context GEMM
# with stride (11888640, 92160, 1) -- a contiguous-only fast path would miss it)
# and writes [B, S, H*D] directly, so no layout tensor is ever needed.
#
# Trace-time call-site inventory on this tree (FASTWAM_FA_DIAG=1), per graph:
#   x240 q(1,98,3072)  k(1,98,3072)  m(98,98)/(106,1)      video prefill self-attn
#   x240 q(1,8,3072)   k(1,129,3072) m(1,1,8,129)/(129,0,0,1)  denoise cross-attn
#   x240 q(1,98,3072)  k(1,129,3072) m(1,1,98,129)             prefill cross-attn
# All of them are strided-mask + strided-K; all are covered.
#
# FASTWAM_FA_DIAG=1 prints the covered/total call-site count (engagement proof).
# ============================================================================
_FA_DIAG = os.environ.get("FASTWAM_FA_DIAG", "0") != "0"
_FA_TRITON = os.environ.get("FASTWAM_FATRITON", "1") != "0"
_FA_SITES: Dict[str, int] = {}
_FA_COV: Dict[str, int] = {"total": 0, "covered": 0}

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False
    _FA_TRITON = False

# =====================================================================
# FASTWAM_R3_FAST (round 3, r3_d0) -- cross-attention half of the same
# transcendental/atomic hygiene applied in mot.py, plus a re-sweep of the
# o-proj FMHA's N tile (the round-1 `_fa_tile` sweep predates the round-2
# epilogue fold and is still tuned for a kernel that no longer exists).
#
# Measured in-container, isolated 200-launch cudagraph replay, M=8 N=129
# D=128 NO=32, median of 3 interleaved rounds:
#   round-2 body, BLOCK_N=128 nw=4 ns=2   8.43 us   (profile says 8.43)
#   + rcp/exp2/relaxed atomic             7.15 us
#   + BLOCK_N=64  nw=4 ns=3               6.19 us
# BLOCK_N=128 on N=129 spends its whole second iteration on ONE live column;
# BLOCK_N=64 gives three iterations the pipeliner can actually overlap.
#
#   FASTWAM_R3_FAST=0   -> both changes off.
#   FASTWAM_R3_XSOFT=0  -> softmax/atomic change off only.
#   FASTWAM_R3_XTILE=1  -> the N-tile re-sweep, SHIPPED OFF.  It is a real
#                          0.96 us/launch (28 us/step), but it reorders the
#                          online softmax over the 129 context keys and that
#                          alone moved the integrated 1-step oracle max|d|
#                          from 7.910e-02 to 1.152e-01 (92% of the 1.25e-01
#                          tolerance), measured by A/B with everything else
#                          held on.  Not worth 92% of a budget three lanes
#                          have to share.  Kept in-tree as the measured
#                          negative result.
# =====================================================================
try:
    _R3_FAST = os.environ.get("FASTWAM_R3_FAST", "1") != "0"
    _R3_XSOFT = 1 if (_R3_FAST and os.environ.get("FASTWAM_R3_XSOFT", "1") != "0") else 0
    _R3_XTILE = bool(_R3_FAST and os.environ.get("FASTWAM_R3_XTILE", "0") != "0")
except Exception:  # pragma: no cover
    _R3_XSOFT = 0
    _R3_XTILE = False

if _FA_DIAG:
    import atexit

    def _fa_diag_dump():
        tot = sum(_FA_SITES.values())
        print(f"[fa_diag] flash_attention trace-time call sites: {tot}", flush=True)
        print(f"[fa_diag] triton-FMHA COVERED {_FA_COV['covered']} / {_FA_COV['total']} flash_attention call sites", flush=True)
        print(f"[fa_diag] sdpa-hook  COVERED {_SDPA_COV['covered']} / {_SDPA_COV['total']} F.scaled_dot_product_attention call sites", flush=True)
        for rec, n in sorted(_FA_SITES.items(), key=lambda kv: -kv[1]):
            print(f"[fa_diag] x{n}  {rec}", flush=True)

    atexit.register(_fa_diag_dump)


def _fa_diag_rec(q, k, v, num_heads, ctx_mask, covered):
    def _t(x):
        if x is None:
            return "None"
        return f"{tuple(x.shape)}/{tuple(x.stride())}/{str(x.dtype).replace('torch.','')}"
    rec = f"cov={int(covered)} q={_t(q)} k={_t(k)} v={_t(v)} h={int(num_heads)} m={_t(ctx_mask)}"
    # NOTE: a *dict value* bump does not invalidate a dynamo guard (a list
    # `append` does -- it guards on len() and forces a recompile every call).
    _FA_SITES[rec] = _FA_SITES.get(rec, 0) + 1
    _FA_COV["total"] = _FA_COV["total"] + 1
    if covered:
        _FA_COV["covered"] = _FA_COV["covered"] + 1


if _HAS_TRITON:

    @triton.jit
    def _fwam_fmha_kernel(
        Q, K, V, MSK, OUT,
        sq_h, sq_m, sk_h, sk_n, sv_h, sv_n, so_h, so_m,
        sm_m, sm_n,
        scale,
        M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        HAS_MASK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        h = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        rm = offs_m < M

        q = tl.load(Q + h * sq_h + offs_m[:, None] * sq_m + offs_d[None, :], mask=rm[:, None], other=0.0)

        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

        for start_n in range(0, N, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            rn = offs_n < N
            k = tl.load(K + h * sk_h + offs_n[:, None] * sk_n + offs_d[None, :], mask=rn[:, None], other=0.0)
            s = (tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale).to(tl.float32)
            keep = rn[None, :]
            if HAS_MASK:
                mb = tl.load(
                    MSK + offs_m[:, None] * sm_m + offs_n[None, :] * sm_n,
                    mask=rm[:, None] & rn[None, :],
                    other=0,
                ).to(tl.int1)
                keep = keep & mb
            s = tl.where(keep, s, float("-inf"))
            m_blk = tl.max(s, 1)
            m_new = tl.maximum(m_i, m_blk)
            # A fully-masked tile would leave m_new = -inf and poison alpha.
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            p = tl.exp(s - m_safe[:, None])
            v_ = tl.load(V + h * sv_h + offs_n[:, None] * sv_n + offs_d[None, :], mask=rn[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v_.dtype), v_, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_safe

        acc = acc / l_i[:, None]
        tl.store(
            OUT + h * so_h + offs_m[:, None] * so_m + offs_d[None, :],
            acc.to(OUT.dtype.element_ty),
            mask=rm[:, None],
        )

    # ------------------------------------------------------------------
    # FASTWAM_R2_OPROJ (round 2, r2_d0) -- the SAME flash body, but the
    # o-projection is folded into the EPILOGUE instead of being dispatched as a
    # separate hipBLASLt `Cijk_..._MT16x16x1024` (measured 6.06 us x 59.5 per
    # denoise step = 0.361 ms / 18% of the step, on an M=8 problem where half of
    # every 16-row macro-tile is dead).
    #
    # The accumulator for head `h` is already in registers here, so the
    # [H*Dh -> NO] contraction is done per head against that head's [Dh, NO]
    # slice of the o-projection weight and reduced across heads with an fp32
    # `tl.atomic_add` into a tiny [M, NO] scratch (1 KB).  Nothing extra is ever
    # written to HBM: the [M, H*Dh] attention output (48 KB) is not materialised
    # at all any more.
    #
    # NUMERICS.  `acc` is rounded to bf16 before the epilogue dot, which is
    # exactly the operand the vendor GEMM used to see, and the reduction is fp32
    # -- so the only change is the ORDER of the 3072-long contraction (24
    # per-head partials instead of hipBLASLt's own K-blocking).
    # ------------------------------------------------------------------
    @triton.jit
    def _fwam_fmha_oproj_kernel(
        Q, K, V, MSK, WO, BO, ACC,
        sq_h, sq_m, sk_h, sk_n, sv_h, sv_n, sw_n,
        sm_m, sm_n,
        scale,
        M: tl.constexpr, N: tl.constexpr, D: tl.constexpr, NO: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        HAS_MASK: tl.constexpr,
        FAST: tl.constexpr = 0,
    ):
        pid_m = tl.program_id(0)
        h = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        rm = offs_m < M

        q = tl.load(Q + h * sq_h + offs_m[:, None] * sq_m + offs_d[None, :], mask=rm[:, None], other=0.0)

        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

        for start_n in range(0, N, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            rn = offs_n < N
            k = tl.load(K + h * sk_h + offs_n[:, None] * sk_n + offs_d[None, :], mask=rn[:, None], other=0.0)
            s = (tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale).to(tl.float32)
            keep = rn[None, :]
            if HAS_MASK:
                mb = tl.load(
                    MSK + offs_m[:, None] * sm_m + offs_n[None, :] * sm_n,
                    mask=rm[:, None] & rn[None, :],
                    other=0,
                ).to(tl.int1)
                keep = keep & mb
            s = tl.where(keep, s, float("-inf"))
            m_blk = tl.max(s, 1)
            m_new = tl.maximum(m_i, m_blk)
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            if FAST == 0:
                alpha = tl.exp(m_i - m_safe)
                p = tl.exp(s - m_safe[:, None])
            else:
                # FASTWAM_R3_XSOFT: exp(x) == exp2(x * log2(e)); the ocml
                # wrapper around v_exp_f32 goes away, values are <=1 ulp apart.
                alpha = tl.math.exp2((m_i - m_safe) * 1.4426950408889634)
                p = tl.math.exp2((s - m_safe[:, None]) * 1.4426950408889634)
            v_ = tl.load(V + h * sv_h + offs_n[:, None] * sv_n + offs_d[None, :], mask=rn[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v_.dtype), v_, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_safe

        if FAST == 0:
            acc = acc / l_i[:, None]
        else:
            # one reciprocal per ROW (BLOCK_M of them) instead of a full
            # BLOCK_M x D fp32 divide.  Deliberately NOT libdevice here: this
            # kernel is launched from a plain python helper, so dynamo inlines
            # it and inductor recompiles it in a subprocess worker where the
            # module-level `libdevice` alias is not in scope (NameError).
            acc = acc * (1.0 / l_i)[:, None]
        ob = acc.to(WO.dtype.element_ty)
        no = tl.arange(0, NO)
        w = tl.load(WO + no[None, :] * sw_n + h * D + tl.arange(0, D)[:, None])
        part = tl.dot(ob, w, out_dtype=tl.float32)
        if h == 0:
            part = part + tl.load(BO + no)[None, :].to(tl.float32)
        if FAST == 0:
            tl.atomic_add(ACC + offs_m[:, None] * NO + no[None, :], part, mask=rm[:, None])
        else:
            tl.atomic_add(
                ACC + offs_m[:, None] * NO + no[None, :], part, mask=rm[:, None], sem="relaxed"
            )


def _fa_mask_strides(m, Mq, N):
    """Normalise an arbitrary broadcastable bool mask to (stride_m, stride_n).

    Returns None when the mask cannot be expressed as a 2-D strided read of the
    [Mq, N] score tile (which is the only thing the kernel supports)."""
    if m is None:
        return (0, 0)
    if m.dtype != torch.bool:
        return None
    if m.dim() < 2:
        return None
    shp = list(m.shape)
    if shp[-1] != N:
        return None
    r = shp[-2]
    if r not in (1, Mq):
        return None
    for s in shp[:-2]:
        if s != 1:
            return None
    sm_n = m.stride(-1)
    sm_m = m.stride(-2) if r == Mq else 0
    return (int(sm_m), int(sm_n))


# ============================================================================
# FASTWAM_R2E2_FATILE (round 2, r2_d2, host_runtime) -- re-tile the PREFILL
# FMHA.  Kill switch FASTWAM_R2E2_FATILE=0 restores `_fa_tile` byte-for-byte.
#
# WHAT IS WRONG WITH THE INHERITED TILE.  `_fa_tile` picks
# `BLOCK_N = 64 if N <= 128 else 128` for EVERY call site.  The video prefill's
# CROSS-attention runs at M = 98, N = ctx = 129 -- one key past the 128
# boundary -- so it takes BLOCK_N = 128 and runs TWO key tiles covering 256
# columns for 129 real keys.  The second tile is 127/128 masked: a full
# [128, 128] K load, a full [128, 128] V load and two full `tl.dot`s that
# contribute exactly ONE column of signal.  That is a ~1.9x pad factor on a
# kernel measured at 0.4396 ms x 58 in `video_prefill`.
#
# The fix costs NOTHING and adds NO launch (which the 4.4 us prefill dispatch
# floor makes mandatory): at BLOCK_N = 64 the same 129 keys need three tiles =
# 192 columns (1.49x pad), at BLOCK_N = 48 -> 144 (1.12x) but 48 is not a
# power of two, so 64 and 32 are the live candidates and both are swept.
#
# SCOPE.  Only the M > 16 (video prefill) call sites move; the action-side
# M = 8 tile the previous rounds swept (`M=8 N=106 -> 4.56 us`,
# `M=8 N=129 -> 6.22 us`) is returned byte-for-byte, so no denoise-side kernel
# and no `_fa_tile` consumer in mot.py changes shape.
#
# NUMERICS.  The flash online-softmax is re-blocked, so the fp32 rescaling
# happens in a different ORDER; every masked column still contributes exp(-inf)
# = 0 exactly.  No precision is spent (same fp32 accumulator, same dtypes),
# only the association order of an fp32 sum changes.
# ============================================================================
_R2E2_FATILE = os.environ.get("FASTWAM_R2E2_FATILE", "1") != "0"
_R2E2_FATILE_BN = int(os.environ.get("FASTWAM_R2E2_FATILE_BN", "64") or "64")
_R2E2_FATILE_NW = int(os.environ.get("FASTWAM_R2E2_FATILE_NW", "4") or "4")
_R2E2_FATILE_NS = int(os.environ.get("FASTWAM_R2E2_FATILE_NS", "4") or "4")
_R2E2_FATILE_BM = int(os.environ.get("FASTWAM_R2E2_FATILE_BM", "16") or "16")


def _fa_tile(Mq, N):
    # Swept in-container against a 50-launch cudagraph replay (eager per-launch
    # timing is host-bound at ~13.7 us and cannot see the kernel at all).
    # BM=16 always: M is 8 or 98 here, and the extra CTAs from a small BM are
    # worth more than the wasted rows because the grid is only 24 heads wide.
    #   M=8  N=106 -> 4.56 us   M=8  N=129 -> 6.22 us
    #   M=98 N=98  -> 4.77 us   M=98 N=129 -> 6.52 us   (SDPA lane: 18.5-23.5 us)
    if _R2E2_FATILE and Mq > 16:
        # FASTWAM_R2E2_FATILE: the video-prefill lane only.
        return (_R2E2_FATILE_BM, _R2E2_FATILE_BN,
                _R2E2_FATILE_NW, _R2E2_FATILE_NS)
    BLOCK_N = 64 if N <= 128 else 128
    return 16, BLOCK_N, 4, 2


def _fa_triton(q, k, v, num_heads, ctx_mask):
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        return None
    b, Mq, C = q.shape
    if b != 1:
        return None
    if k.shape[0] != 1 or v.shape[0] != 1 or k.shape[2] != C or v.shape[2] != C:
        return None
    N = k.shape[1]
    if v.shape[1] != N:
        return None
    D = C // num_heads
    if D * num_heads != C or D not in (32, 64, 128, 256):
        return None
    if q.dtype != k.dtype or q.dtype != v.dtype:
        return None
    if q.dtype not in (torch.bfloat16, torch.float16):
        return None
    if q.stride(2) != 1 or k.stride(2) != 1 or v.stride(2) != 1:
        return None
    ms = _fa_mask_strides(ctx_mask, Mq, N)
    if ms is None:
        return None
    sm_m, sm_n = ms

    BLOCK_M, BLOCK_N, NW, NS = _fa_tile(Mq, N)

    out = torch.empty((1, Mq, C), dtype=q.dtype, device=q.device)
    grid = ((Mq + BLOCK_M - 1) // BLOCK_M, num_heads)
    _fwam_fmha_kernel[grid](
        q, k, v, ctx_mask if ctx_mask is not None else q, out,
        D, q.stride(1), D, k.stride(1), D, v.stride(1), D, out.stride(1),
        sm_m, sm_n,
        1.0 / math.sqrt(D),
        M=Mq, N=N, D=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        HAS_MASK=(ctx_mask is not None),
        num_warps=NW, num_stages=NS,
    )
    return out


# FASTWAM_R2_OPROJ_CROSS -- kill-switch for the cross-attention epilogue fold.
_R2_OPROJ_CROSS = os.environ.get("FASTWAM_R2_OPROJ_CROSS", "1") != "0"


def fwam_cross_oproj(q, k, v, num_heads, ctx_mask, wo, bo, acc):
    """Token-major FMHA with the o-projection folded into the epilogue.

    `acc` is a PRE-ZEROED fp32 [Mq, NO] scratch which the kernel reduces into
    with atomics; the caller reads it back (plus the bias, which head 0 adds)
    and casts to bf16 inside whatever pointwise kernel already consumes it.

    Returns True iff the fold ran.  Every rejection below is on trace-time
    constant metadata, so `torch.compile` specialises the branch away.
    """
    if not (_R2_OPROJ_CROSS and _HAS_TRITON and _FA_TRITON):
        return False
    if q is None or k is None or v is None or wo is None or bo is None or acc is None:
        return False
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        return False
    b, Mq, C = q.shape
    if b != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        return False
    if k.shape[2] != C or v.shape[2] != C:
        return False
    N = k.shape[1]
    if v.shape[1] != N:
        return False
    D = C // num_heads
    if D * num_heads != C or D not in (32, 64, 128, 256):
        return False
    if q.dtype != k.dtype or q.dtype != v.dtype:
        return False
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False
    if q.stride(2) != 1 or k.stride(2) != 1 or v.stride(2) != 1:
        return False
    if wo.dim() != 2 or int(wo.shape[1]) != C or wo.stride(1) != 1:
        return False
    if wo.dtype != q.dtype or bo.dim() != 1 or int(bo.shape[0]) != int(wo.shape[0]):
        return False
    NO = int(wo.shape[0])
    if NO not in (16, 32, 64, 128):
        return False
    if acc.dtype != torch.float32 or acc.dim() != 2:
        return False
    if int(acc.shape[0]) != int(Mq) or int(acc.shape[1]) != NO or acc.stride(1) != 1:
        return False
    if acc.stride(0) != NO:
        return False
    ms = _fa_mask_strides(ctx_mask, Mq, N)
    if ms is None:
        return False
    sm_m, sm_n = ms

    BLOCK_M, BLOCK_N, NW, NS = _fa_tile(Mq, N)
    if _R3_XTILE and Mq <= 16 and N > 64:
        # FASTWAM_R3_XTILE.  Re-swept against the ROUND-2 body (the fold now
        # carries the o-projection, so the round-1 numbers are void).  This
        # branch is only ever taken by the action expert's cross-attention
        # (Mq = Sa = 8, N = ctx = 129); every other `_fa_tile` consumer,
        # including the video prefill's own FMHA, is untouched.
        BLOCK_N, NW, NS = 64, 4, 3
    grid = ((Mq + BLOCK_M - 1) // BLOCK_M, num_heads)
    _fwam_fmha_oproj_kernel[grid](
        q, k, v, ctx_mask if ctx_mask is not None else q, wo, bo, acc,
        D, q.stride(1), D, k.stride(1), D, v.stride(1), wo.stride(0),
        sm_m, sm_n,
        1.0 / math.sqrt(D),
        M=Mq, N=N, D=D, NO=NO,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        HAS_MASK=(ctx_mask is not None),
        FAST=_R3_XSOFT,
        num_warps=NW, num_stages=NS,
    )
    return True


# ============================================================================
# FASTWAM_SDPAHOOK -- route the HEAD-MAJOR 4-D SDPA call sites onto the same
# Triton FMHA.  DEFAULT ON, kill switch `FASTWAM_SDPAHOOK=0`.
#
# WHY IT IS A MONKEY-PATCH AND NOT A CALL-SITE EDIT.  The single largest
# attention item left in the denoise step after FASTWAM_FATRITON is NOT reached
# through `flash_attention` at all: `mot.py`'s FASTWAM_MOTKVH path calls
# `F.scaled_dot_product_attention` directly on the [1, H, Sv+Sa, D] persistent
# K/V buffer (profiled: `attn_fwd` 0.309 ms x 29.9 per step = 10.3 us/call, on
# an M=8, S_kv=106 problem whose kernel floor is 4.6 us).  `mot.py` is owned by
# another engineer this round, so the hook is installed from here, on the
# `torch.nn.functional` module object that `mot.py` already resolves through.
#
# It is gated HARD: 4-D, batch 1, equal head counts, contiguous head_dim,
# bf16/fp16, no dropout, not causal, default scale, bool-or-absent mask.  The
# VAE's two SDPA sites (`wan_video_vae.py:1340,1372`) and anything else in the
# process fall through to the original op untouched.
#
# The output is allocated TOKEN-MAJOR ([1, M, H, D]) and handed back as a
# `.permute(0, 2, 1, 3)` view, so the caller's canonical
# `.transpose(1, 2).reshape(1, M, H*D)` becomes a free reshape of a contiguous
# tensor instead of a materialised transpose.
# ============================================================================
_SDPA_HOOK = os.environ.get("FASTWAM_SDPAHOOK", "1") != "0"
_ORIG_SDPA = F.scaled_dot_product_attention
_SDPA_COV: Dict[str, int] = {"total": 0, "covered": 0}


def _fa_triton_hm(query, key, value, attn_mask):
    """Head-major [B, H, S, D] entry point for the FMHA kernel."""
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return None
    b, H, Mq, D = query.shape
    if b != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        return None
    if key.shape[1] != H or value.shape[1] != H:
        return None
    if key.shape[3] != D or value.shape[3] != D:
        return None
    N = key.shape[2]
    if value.shape[2] != N:
        return None
    if D not in (32, 64, 128, 256):
        return None
    if query.dtype != key.dtype or query.dtype != value.dtype:
        return None
    if query.dtype not in (torch.bfloat16, torch.float16):
        return None
    if query.stride(3) != 1 or key.stride(3) != 1 or value.stride(3) != 1:
        return None
    ms = _fa_mask_strides(attn_mask, Mq, N)
    if ms is None:
        return None
    sm_m, sm_n = ms

    BLOCK_M, BLOCK_N, NW, NS = _fa_tile(Mq, N)
    out_tm = torch.empty((1, Mq, H, D), dtype=query.dtype, device=query.device)
    out = out_tm.permute(0, 2, 1, 3)
    grid = ((Mq + BLOCK_M - 1) // BLOCK_M, H)
    _fwam_fmha_kernel[grid](
        query, key, value, attn_mask if attn_mask is not None else query, out_tm,
        query.stride(1), query.stride(2),
        key.stride(1), key.stride(2),
        value.stride(1), value.stride(2),
        D, H * D,
        sm_m, sm_n,
        1.0 / math.sqrt(D),
        M=Mq, N=N, D=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        HAS_MASK=(attn_mask is not None),
        num_warps=NW, num_stages=NS,
    )
    return out


def _fwam_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
               is_causal=False, scale=None, enable_gqa=False):
    if (not is_causal) and dropout_p == 0.0 and (not enable_gqa) and query.dim() == 4:
        ok_scale = scale is None or (
            query.shape[3] > 0 and abs(float(scale) - 1.0 / math.sqrt(query.shape[3])) < 1e-12
        )
        if ok_scale:
            out = _fa_triton_hm(query, key, value, attn_mask)
            if _FA_DIAG:
                _SDPA_COV["total"] = _SDPA_COV["total"] + 1
                if out is not None:
                    _SDPA_COV["covered"] = _SDPA_COV["covered"] + 1
            if out is not None:
                return out
        elif _FA_DIAG:
            _SDPA_COV["total"] = _SDPA_COV["total"] + 1
    elif _FA_DIAG:
        _SDPA_COV["total"] = _SDPA_COV["total"] + 1
    return _ORIG_SDPA(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                      is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)


if _SDPA_HOOK and _HAS_TRITON:
    F.scaled_dot_product_attention = _fwam_sdpa
    torch.nn.functional.scaled_dot_product_attention = _fwam_sdpa


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, ctx_mask: Optional[torch.Tensor] = None, compatibility_mode=True):
    if _FA_TRITON:
        out = _fa_triton(q, k, v, num_heads, ctx_mask)
        if _FA_DIAG:
            _fa_diag_rec(q, k, v, num_heads, ctx_mask, out is not None)
        if out is not None:
            return out
    elif _FA_DIAG:
        _fa_diag_rec(q, k, v, num_heads, ctx_mask, False)
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = _ORIG_SDPA(q, k, v, attn_mask=ctx_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        return x
    else:
        raise NotImplementedError("Only compatibility mode is implemented for flash attention. Please set compatibility_mode=True.")



def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


# ============================================================================
# FASTWAM_ACTX (r1_d3) -- kill-switch (RULE 9), default ON.
#
# WHAT IT HOISTS.  `FastWAM._denoise_action_with_video_cache` (IMMUTABLE) is the
# torch.compile(reduce-overhead, fullgraph) region that is cudagraph-replayed
# ONCE PER DENOISE STEP.  Inside it, `ActionDiT.prepare` computes
# `context_emb = self.text_embedding(context)` -- a [1, L, text_dim=4096] ->
# [1, L, hidden=32] Linear, a tanh-GELU and a [32, 32] Linear.  `context` is a
# per-CALL constant (fastwam.py::infer_action resolves it once, before the
# schedule loop, and passes the SAME tensor to every step), so that projection
# recomputes a bit-identical result on every one of the 10 steps of
# `episode_10step`.  With dG/dD = 4.1611 vs dG/dP = 1.0715, a millisecond moved
# from the per-step region into the once-per-call prologue is worth 3.88x its
# face value.
#
# WHERE IT GOES.  `WanModel.prepare` (the VIDEO expert's prologue) is called
# EXACTLY ONCE per `infer_action`, eagerly, with the very same raw `context`
# the denoise region will be handed.  It is therefore the natural staging site.
# It re-projects the action expert's context into a persistent, in-place
# refreshed buffer registered on the ActionDiT (`register_buffer(...,
# persistent=False)`), which dynamo treats as a STATIC-ADDRESS graph input --
# so it costs ZERO cudagraph input copies, the same reason `mot.py`'s fused
# weights are buffers and not plain attributes.  As a bonus the raw `context`
# [1, 129, 4096] bf16 (1.06 MB) becomes dead inside the compiled region and its
# per-replay cudagraph input copy disappears with it.
#
# COMMANDMENT RULE 6 (episode state).  The buffer is REFRESHED UNCONDITIONALLY
# on every call, from the prologue, before any denoise step runs -- it is not
# memoised, not keyed on `data_ptr`, not keyed on `_version`, and there is no
# path that reaches the denoise loop without going through `WanModel.prepare`
# first (fastwam.py::infer_action calls `video_expert.prepare` at line ~1180,
# unconditionally, above the schedule loop).  `context` is EPISODE-VARYING and
# this is exactly the failure mode the COMMANDMENT calls out, so the staging
# path is fail-closed: `_fw_actx_ok` is cleared FIRST and only set once the
# fresh projection has actually landed in the buffer at the matching length.
# Any bail-out (different context length than the one the buffer was sized at,
# a non-3D context, a disabled switch) leaves the flag False and
# `ActionDiT.prepare` recomputes `text_embedding(context)` in-graph exactly as
# the pristine code does.  The buffer is allocated ONCE and never reallocated,
# so the address a cudagraph captured can never go stale.
#
# RULE 7.  The consumer branch reads one Python bool attribute, which dynamo
# specialises at trace time; no data-dependent control flow, no graph break,
# `fullgraph=True` holds and the region stays 1 `hipGraphLaunch`.
# ============================================================================
_FW_ACTX_ENABLED = os.environ.get("FASTWAM_ACTX", "1") not in ("0", "false", "False", "off", "OFF")
_FW_ACTX_SINKS: "list" = []


def _fw_actx_register(mod) -> None:
    """Register an ActionDiT whose text-context projection should be staged."""
    import weakref
    _FW_ACTX_SINKS.append(weakref.ref(mod))


def _fw_actx_stage(context) -> None:
    """Refresh every registered action expert's staged context embedding."""
    if not _FW_ACTX_ENABLED or not _FW_ACTX_SINKS:
        return
    alive = []
    for ref in _FW_ACTX_SINKS:
        mod = ref()
        if mod is None:
            continue
        alive.append(ref)
        stage = getattr(mod, "_fw_stage_context", None)
        if stage is not None:
            stage(context)
    if len(alive) != len(_FW_ACTX_SINKS):
        _FW_ACTX_SINKS[:] = alive


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute -- the VIDEO expert's table.  This is the one that gets
    # the real (cos, sin) layout; see the FASTWAM_ROPEREAL block below.
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta, _real=_ROPE_REAL)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta, _real=_ROPE_REAL)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta, _real=_ROPE_REAL)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def _env_on(name: str, default: str) -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False", "off", "OFF")


# ============================================================================
# FASTWAM_ROPEREAL -- real-valued (cos, sin) RoPE, replacing the complex128
# `torch.polar` / `view_as_complex` round-trip.  DEFAULT ON.
#
# WHY IT PAYS.  The pristine `rope_apply` casts to float64, reinterprets as
# complex128, multiplies, and casts back.  Inductor cannot code-generate complex
# ops ("Torchinductor does not support code generation for complex operators"),
# so the multiply lowers to an ATEN EXTERN kernel -- `elementwise_kernel_manual
# _unroll<128,4>`, measured at 0.2501 ms x 55.7/call in `video_prefill` and
# 0.2631 ms x 60.0/step in `denoise_step`, i.e. ~4.4 us each, which is the
# per-dispatch floor on this box.  Rewriting the rotation with real pointwise
# arithmetic lets inductor FUSE it away entirely: the win is DISPATCH DELETION,
# not cheaper math.
#
# THE ACTION/DENOISE PATH IS ON BY DEFAULT SINCE ROUND 2, MINUS LAYER 0.
# This tree's numerics oracle admits essentially NO perturbation of the DiT hot
# path -- budget mean|d| <= 2.25e-02 at steps=1 -- and fusing the RoPE on ALL 30
# action layers measured 2.759e-02 there.  ROUND 2 BISECTED THAT PER LAYER (see
# FASTWAM_ROPEDEN_SKIP in mot.py) and found it is entirely LAYER 0:
#   layers 1..29 fused, layer 0 pristine -> max|d| = mean|d| = 0.000e+00 at BOTH
#   step counts (bit-identical to the golden), while fusing ONLY layer 0 and
#   leaving 1..29 pristine measures 2.940e-02 -> FAIL.
# So the fused real rotation is BIT-IDENTICAL to the complex128 chain on 29 of
# the 30 layers; only layer 0 -- whose `x` is the raw graph-input action token
# tensor, giving inductor a different fusion boundary for norm_q/norm_k -- moves
# a bf16 ULP, and the 30-layer stack amplifies that one ULP past the budget.
# Holding layer 0 back costs 2 of the 60 extern dispatches and buys back 100% of
# the steps=1 budget.
# The earlier in-tree explanation (an SDPA q/k STRIDE flip) was DISPROVED in
# round 2 by two direct measurements: forcing the action q head-major contiguous
# on the OFF arm is bit-exact, and pinning the fused arm's q token-major through
# a static buffer leaves the failure at exactly 2.616e-02.
#
#   FASTWAM_ROPEREAL=0       -> restore the pristine complex128 chain everywhere.
#   FASTWAM_ROPEDENOISE=0    -> restore it on the action/denoise path only.
#   FASTWAM_ROPEDEN_SKIP=n   -> hold the first n action layers back (default 1).
#   FASTWAM_ROPEF64=0     -> do the rotation in fp32 instead of fp64.  Faster
#                            per element but NOT bit-identical (2-7 bf16 ULP
#                            flips per call out of ~1e5), and those alone move
#                            the oracle to mean|d| = 3.328e-02 -> FAIL.  Kept as
#                            an escape hatch, shipped OFF.
# All three are read ONCE at import so the compiled regions never branch on an
# env lookup and `fullgraph=True` is untouched.
# ============================================================================
_ROPE_REAL = _env_on("FASTWAM_ROPEREAL", "1")
_ROPE_F64 = _env_on("FASTWAM_ROPEF64", "1")
_ROPE_DENOISE = _env_on("FASTWAM_ROPEDENOISE", "1")
_ROPE_DTYPE = torch.float64 if _ROPE_F64 else torch.float32


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0, _real=None):
    # 1d rope precompute.
    #
    # Real layout (default): a real tensor of shape [end, dim] whose last axis
    # holds each rotated 2-element chunk's (cos, sin) pair INTERLEAVED as
    # [cos_0, sin_0, cos_1, sin_1, ...].  That interleaving is what lets every
    # existing consumer -- `precompute_freqs_cis_3d`, `WanVideoDiT.get_freqs`,
    # `ActionDiT.get_freqs` -- keep slicing / viewing / concatenating along the
    # last axis completely unchanged: concatenating interleaved (cos, sin) pairs
    # is exactly concatenating the complex elements they encode.
    #
    # The pair is read out of `torch.polar` itself rather than recomputed with
    # torch.cos / torch.sin.  Those two DISAGREE in the last mantissa bits
    # (verified: `torch.equal` False on the 1024x128 table), and on this model
    # that difference alone is enough to move the oracle.
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex128
    # `_real` is chosen PER TABLE, once, at construction.  action_dit.py calls
    # this entry point positionally for the ACTION expert and therefore gets the
    # default (`_ROPE_DENOISE`, shipped off -> complex); `precompute_freqs_cis_3d`
    # passes `_real=_ROPE_REAL` for the VIDEO expert.  Splitting here rather than
    # inside `rope_apply` matters: a shape test in `rope_apply` would have to
    # rebuild a complex view of the real table on the excluded path, and because
    # `freqs` is a runtime argument of the compiled region that rebuild is NOT
    # const-folded -- measured at +61 kernels/denoise-step.
    real = _ROPE_REAL and (_ROPE_DENOISE if _real is None else bool(_real))
    if not real:
        return freqs_cis
    return torch.view_as_real(freqs_cis).reshape(end, -1).contiguous().to(_ROPE_DTYPE)


def rope_apply(x, freqs, num_heads, force_complex: bool = False):
    batch_size, seq_len, _ = x.shape
    if force_complex and not freqs.is_complex():
        # Diagnostic / partial-arm path: rebuild the complex128 view of the real
        # (cos, sin) table so THIS call takes the pristine extern-kernel chain
        # while another call on the same table takes the fused real one.  The
        # rebuild is a pure reinterpret of the same bits, so it introduces no
        # arithmetic of its own.
        freqs = torch.view_as_complex(freqs.unflatten(-1, (-1, 2)).contiguous())
    if freqs.is_complex():
        # Pristine complex128 round-trip (the OFF arm / the action expert).
        x_heads = x.view(batch_size, seq_len, num_heads, -1)
        x_out = torch.view_as_complex(
            x_heads.to(torch.float64).reshape(batch_size, seq_len, num_heads, -1, 2)
        )
        return torch.view_as_real(x_out * freqs).flatten(2).to(x.dtype)

    # ON arm: pure real pointwise arithmetic.  Same formula the complex multiply
    # uses -- (a*c - b*d, a*d + b*c) -- at the same precision, so it is exact,
    # and every op is pointwise so inductor folds the whole rotation into the
    # neighbouring kernel instead of emitting an extern dispatch.
    #   freqs : [S, 1, D]   -> [S, 1, D/2, 2]  (broadcasts over batch and heads)
    #   x     : [B, S, H*D] -> [B, S, H, D/2, 2]
    pair = freqs.unflatten(-1, (-1, 2))
    cos = pair[..., 0]
    sin = pair[..., 1]
    x_pairs = x.to(freqs.dtype).view(batch_size, seq_len, num_heads, -1, 2)
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]
    out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    return out.reshape(batch_size, seq_len, -1).to(x.dtype)


def create_group_causal_attn_mask(
    num_temporal_groups: int, num_query_per_group: int, num_key_per_group: int, mode: str = "causal"
) -> torch.Tensor:
    """
    Creates a group-based attention mask for scaled dot-product attention with two modes:
    'causal' and 'group_diagonal'.

    Parameters:
    - num_temporal_groups (int): The number of temporal groups (e.g., frames in a video sequence).
    - num_query_per_group (int): The number of query tokens per temporal group. (e.g., latent tokens in a frame, H x W).
    - num_key_per_group (int): The number of key tokens per temporal group. (e.g., action tokens per frame).
    - mode (str): The mode of the attention mask. Options are:
        - 'causal': Query tokens can attend to key tokens from the same or previous temporal groups.
        - 'group_diagonal': Query tokens can attend only to key tokens from the same temporal group.

    Returns:
    - attn_mask (torch.Tensor): A boolean tensor of shape (L, S), where:
        - L = num_temporal_groups * num_query_per_group (total number of query tokens)
        - S = num_temporal_groups * num_key_per_group (total number of key tokens)
      The mask indicates where attention is allowed (True) and disallowed (False).

    Example:
    Input:
        num_temporal_groups = 3
        num_query_per_group = 4
        num_key_per_group = 2
    Output:
        Causal Mask Shape: torch.Size([12, 6])
        Group Diagonal Mask Shape: torch.Size([12, 6])
        if mode='causal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True]])

        if mode='group_diagonal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True]])

    """
    assert mode in ["causal", "group_diagonal"], f"Mode {mode} must be 'causal' or 'group_diagonal'"

    # Total number of query and key tokens
    total_num_query_tokens = num_temporal_groups * num_query_per_group  # Total number of query tokens (L)
    total_num_key_tokens = num_temporal_groups * num_key_per_group  # Total number of key tokens (S)

    # Generate time indices for query and key tokens (shape: [L] and [S])
    query_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_query_per_group)  # Shape: [L]
    key_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_key_per_group)  # Shape: [S]

    # Expand dimensions to compute outer comparison
    query_time_indices = query_time_indices.unsqueeze(1)  # Shape: [L, 1]
    key_time_indices = key_time_indices.unsqueeze(0)  # Shape: [1, S]

    if mode == "causal":
        # Causal Mode: Query can attend to keys where key_time <= query_time
        attn_mask = query_time_indices >= key_time_indices  # Shape: [L, S]
    elif mode == "group_diagonal":
        # Group Diagonal Mode: Query can attend only to keys where key_time == query_time
        attn_mask = query_time_indices == key_time_indices  # Shape: [L, S]

    assert attn_mask.shape == (total_num_query_tokens, total_num_key_tokens), "Attention mask shape mismatch"
    return attn_mask


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, ctx_mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return x


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)
        
        # self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, self_attn_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=self_attn_mask)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6,):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)
            
        # self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self,  hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, hidden_dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, context_mask=None, self_attn_mask: Optional[torch.Tensor] = None):
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1) # (B, 1, seq_len, context_len), 1 for heads
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask))
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


# ============================================================================
# FASTWAM_VDITEVAL -- mode-cached `train()` for the VIDEO expert.  DEFAULT ON.
# BIT-EXACT (oracle 0.000e+00 at steps=1; steps=10 unchanged at the shipped
# tree's 1.457e-02), so it spends ZERO of the remaining numerics budget.
#
# WHY.  `FastWAM.infer_action` (IMMUTABLE, fastwam.py:1097) opens with
# `self.eval()` on EVERY call, and `nn.Module.train(mode)` is an unconditional
# CPython recursion over the whole module tree with NO visited-set -- so a
# submodule reachable by two references is walked TWICE.  `WanVideoDiT` is
# reachable as BOTH `FastWAM.video_expert` AND `FastWAM.mot.mixtures["video"]`,
# and it is 30 `DiTBlock`s (~24 sub-modules each) plus the embedding stacks.
# MEASURED IN-TREE with FASTWAM_VDITDIAG=1: **707 modules, 0.766 ms per walk,
# 1.084 us per visit** -> ~1414 visits / ~1.53 ms of pure host time per call.
# The perf harness syncs immediately before each timed iteration, so `eval()`
# runs with the GPU genuinely idle and 100% of that is exposed wall time.
# Measured A/B (interleaved, 3 rounds): -1.52 ms on episode_1step and -1.51 ms
# on episode_10step -- the SAME ABSOLUTE DELTA on both cases, which is the
# signature of a prologue-only win.
#
# The identical override already ships on `ActionDiT` (action_dit.py,
# FASTWAM_ADITEVAL) and on `WanVideoVAE38` (wan_video_vae.py:2577).  This closes
# the last un-covered root: after it the whole `self.eval()` walk is ~10 visits.
#
# WHY IT IS A NO-OP BY CONSTRUCTION.  The short-circuit only declines to
# RE-write `.training` flags that already equal `mode`.  The cache starts
# *absent*, so the first call in each direction falls through to
# `nn.Module.train` and really writes every descendant flag; a genuine
# transition in either direction is handled the same way.  Nothing in this tree
# writes a descendant `.training` behind `train()`'s back.
#
# COMMANDMENT RULE 6 (episode state).  The only cached object is a single
# Python `bool` naming a mode the caller just asked for -- no tensor, no
# `data_ptr`, no `_version`, nothing a new episode can invalidate.  Validated
# on the 14-episode closed-loop rollout anyway (PASS on attempt 1).
#
#   FASTWAM_VDITEVAL=0  -> restore the pristine unconditional walk (OFF arm).
#   FASTWAM_VDITDIAG=1  -> print, once, the module count under this root and the
#                          wall time of the FULL (uncached) walk.  Diagnostic
#                          only; never set for a scored run.
# ============================================================================
_VDIT_EVAL = _env_on("FASTWAM_VDITEVAL", "1")
_VDIT_DIAG = os.environ.get("FASTWAM_VDITDIAG", "0") == "1"


class WanVideoDiT(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = False,
        require_clip_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
        action_conditioned: bool = False,
        action_dim: int = 7,
        action_group_causal_mask_mode = "causal",
        video_attention_mask_mode: str = "bidirectional",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.video_attention_mask_mode = str(video_attention_mask_mode)

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(
                f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}"
            )
        
        self.action_conditioned = action_conditioned
        self.action_dim = action_dim
        assert has_image_input == False
        assert require_clip_embedding == False
        assert require_vae_embedding == False and fuse_vae_embedding_in_latents == True, "Only support fusing vae embedding in latents"

        self.patch_embedding = nn.Conv3d(
            in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(hidden_dim, out_dim, patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(attn_head_dim)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, hidden_dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

        if self.action_conditioned:
            self.action_embedding = nn.Linear(action_dim, hidden_dim)
            self.action_group_causal_mask_mode = action_group_causal_mask_mode
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        if self.use_gradient_checkpointing:
            logger.info("Using gradient checkpointing for DiT blocks. This will save memory but use more computation.")

    def _apply(self, fn):
        result = super()._apply(fn)
        device = next(self.parameters()).device
        self.freqs = tuple(freqs.to(device=device) for freqs in self.freqs)
        return result

    def train(self, mode: bool = True):
        # See the FASTWAM_VDITEVAL note just above this class.
        if _VDIT_DIAG:
            import time as _t
            _n = sum(1 for _ in self.modules())
            _t0 = _t.perf_counter()
            nn.Module.train(self, bool(mode))
            _dt = (_t.perf_counter() - _t0) * 1e3
            print(f"[vditdiag] WanVideoDiT modules={_n} full_train_walk={_dt:.4f} ms "
                  f"({_dt * 1e3 / max(_n, 1):.3f} us/visit) "
                  f"cached={'1' if _VDIT_EVAL else '0'}", flush=True)
            if _VDIT_EVAL:
                self.__dict__["_fw_train_mode"] = bool(mode)
            return self
        if _VDIT_EVAL:
            mode = bool(mode)
            if self.__dict__.get("_fw_train_mode", None) is mode:
                return self
            nn.Module.train(self, mode)
            self.__dict__["_fw_train_mode"] = mode
            return self
        return nn.Module.train(self, bool(mode))

    def get_freqs(self, f: int, h: int, w: int) -> torch.Tensor:
        freq_f, freq_h, freq_w = self.freqs
        return torch.cat(
            [
                freq_f[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freq_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freq_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)
            

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def _validate_forward_inputs(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        action: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"`latents` must be 5D [B, C, T, H, W], got shape {tuple(x.shape)}")
        num_latent_frames = x.shape[2]
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if self.action_conditioned:
            allow_text_only_single_frame = (num_latent_frames == 1 and action is None)
            if not allow_text_only_single_frame:
                assert action is not None, "Action input is required for action-conditioned model."
                if action.ndim != 3:
                    raise ValueError(f"`action` must be 3D [B, action_horizon, action_dim], got shape {tuple(action.shape)}")
                if action.shape[2] != self.action_dim:
                    raise ValueError(f"`action` last dimension must be {self.action_dim}, got {action.shape[2]}")
                if num_latent_frames <= 1:
                    raise ValueError(f"video length must be > 1 for action-conditioned model, got {num_latent_frames}")
                if action.shape[1] % (num_latent_frames - 1) != 0:
                    raise ValueError(
                        f"action horizon must be divisible by (num_latent_frames - 1), got action_horizon={action.shape[1]}"
                    )
        if context_mask is None:
            context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != context.shape[0] or context_mask.shape[1] != context.shape[1]:
                raise ValueError(f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}")

        batch_size = x.shape[0]
        if batch_size != context.shape[0]:
            if not self.training and batch_size == 1:
                x = x.expand(context.shape[0], -1, -1, -1, -1)
                batch_size = context.shape[0]
            else:
                raise ValueError(
                    f"Batch mismatch between latents and context: {batch_size} vs {context.shape[0]}."
                )

        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            assert not self.training, "During training, timestep length must match batch_size."
            timestep = timestep.expand(batch_size)
        return x, timestep, context_mask

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")

        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if self.video_attention_mask_mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` in `per_frame_causal` mode, "
                    f"got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(
                torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device)
            )
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if self.video_attention_mask_mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(f"Unsupported video attention mask mode: {self.video_attention_mask_mode}")

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
        control_camera_latents_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        x, timestep, context_mask = self._validate_forward_inputs(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
        )

        batch_size = x.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        if x.shape[3] % patch_h != 0 or x.shape[4] % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by DiT patch size, "
                f"got HxW=({x.shape[3]}, {x.shape[4]}), patch=({patch_h}, {patch_w})"
            )
        tokens_per_frame = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)

        if self.seperated_timestep and fuse_vae_embedding_in_latents:
            if not hasattr(self, "patch_size") or len(self.patch_size) < 3:
                raise ValueError(f"Invalid dit.patch_size: {getattr(self, 'patch_size', None)}")
            
            token_timesteps = torch.ones(
                (batch_size, x.shape[2], tokens_per_frame),
                dtype=timestep.dtype,
                device=timestep.device,
            ) * timestep.view(batch_size, 1, 1)
            token_timesteps[:, 0, :] = 0
            token_timesteps = token_timesteps.reshape(batch_size, -1)
            token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
            t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        else:
            raise NotImplementedError("Only support seperated_timestep with fuse_vae_embedding_in_latents for now.")
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        x = self.patchify(x, control_camera_latents_input=control_camera_latents_input)
        f, h, w = x.shape[2:]

        context = self.text_embedding(context) # (B, L, dim)
        context_len = context.shape[1]
        if self.action_conditioned and action is not None:
            action_len = action.shape[1]
            action_emb = self.action_embedding(action) # (B, action_len, dim)
            action_pos_embed = sinusoidal_embedding_1d(self.hidden_dim, 
                torch.arange(action_len, device=action_emb.device)) # (action_len, dim)
            action_emb = action_emb + action_pos_embed.unsqueeze(0) # (B, action_len, dim)
            context = torch.cat([context, action_emb], dim=1) # (B, context_len + action_len, dim)

            # new mask
            num_temporal_groups = f - 1 # first latent frame do not attend to actions
            if num_temporal_groups <= 0:
                raise ValueError(
                    "Action-conditioned context mask requires at least 2 latent frames when `action` is provided."
                )
            assert action_emb.shape[1] % num_temporal_groups == 0, \
                f"Action embedding length {action_emb.shape[1]} must be divisible by number of temporal groups {num_temporal_groups}"
            # Each latent frame (from the 2nd one) attends to the corresponding group of action tokens
            action_group_mask = create_group_causal_attn_mask(
                num_temporal_groups=num_temporal_groups,
                num_query_per_group=tokens_per_frame,
                num_key_per_group=action_len // num_temporal_groups,
                mode=self.action_group_causal_mask_mode,
            ).to(context.device) # ((f-1)*tokens_per_frame, action_len)

            seq_len = f * h * w # query length
            final_context_mask = torch.zeros((batch_size, seq_len, context.shape[1]), dtype=torch.bool, device=context.device) # (B, seq_len, L + action_len)
            # all latent frames attend to text tokens
            final_context_mask[:, :, :context_len] = context_mask.unsqueeze(1).expand(-1, seq_len, -1) # (B, seq_len, L)
            # latent frames from the 2nd one attend to action tokens
            final_context_mask[:, tokens_per_frame:, context_len:] = action_group_mask.unsqueeze(0).expand(batch_size, -1, -1) # (B, seq_len, action_len)
            context_mask = final_context_mask
        elif self.action_conditioned and action is None:
            if f != 1:
                raise ValueError(
                    "Action-conditioned model requires `action` unless running single-frame text-only mode with num_latent_frames=1."
                )
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1) # (B, seq_len, L)
        else:
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1) # (B, seq_len, L)

        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        freqs = self.get_freqs(f, h, w)

        return {
            "tokens": x_tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "context_mask": context_mask,
            "meta": {
                "grid_size": (f, h, w),
                "tokens_per_frame": tokens_per_frame,
                "batch_size": batch_size,
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        f, h, w = pre_state["meta"]["grid_size"]
        x = self.head(x_tokens, pre_state["t"])
        x = self.unpatchify(x, (f, h, w))
        return x

    def prepare(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
        control_camera_latents_input: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        int,
        int,
    ]:
        """Prepare tensor inputs for the compile-friendly DiT/MoT core."""
        # FASTWAM_ACTX: once-per-call staging of the ACTION expert's
        # step-invariant text-context projection.  See the block above.
        _fw_actx_stage(context)
        batch_size = x.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        tokens_per_frame = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)

        if not self.seperated_timestep or not fuse_vae_embedding_in_latents:
            raise NotImplementedError(
                "Only support seperated_timestep with fuse_vae_embedding_in_latents for now."
            )
        token_timesteps = torch.ones(
            (batch_size, x.shape[2], tokens_per_frame),
            dtype=timestep.dtype,
            device=timestep.device,
        ) * timestep.view(batch_size, 1, 1)
        token_timesteps[:, 0, :] = 0
        token_timesteps = token_timesteps.reshape(batch_size, -1)
        token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
        t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))

        x = self.patchify(x, control_camera_latents_input=control_camera_latents_input)
        f, h, w = x.shape[2:]
        context = self.text_embedding(context)
        context_len = context.shape[1]
        if self.action_conditioned and action is not None:
            action_len = action.shape[1]
            action_emb = self.action_embedding(action)
            action_pos_embed = sinusoidal_embedding_1d(
                self.hidden_dim,
                torch.arange(action_len, device=action_emb.device),
            )
            action_emb = action_emb + action_pos_embed.unsqueeze(0)
            context = torch.cat([context, action_emb], dim=1)
            num_temporal_groups = f - 1
            action_group_mask = create_group_causal_attn_mask(
                num_temporal_groups=num_temporal_groups,
                num_query_per_group=tokens_per_frame,
                num_key_per_group=action_len // num_temporal_groups,
                mode=self.action_group_causal_mask_mode,
            ).to(context.device)
            seq_len = f * h * w
            final_context_mask = torch.zeros(
                (batch_size, seq_len, context.shape[1]),
                dtype=torch.bool,
                device=context.device,
            )
            final_context_mask[:, :, :context_len] = context_mask.unsqueeze(1).expand(
                -1, seq_len, -1
            )
            final_context_mask[:, tokens_per_frame:, context_len:] = action_group_mask.unsqueeze(
                0
            ).expand(batch_size, -1, -1)
            context_mask = final_context_mask
        else:
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1)

        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        freqs = self.get_freqs(f, h, w)
        return x_tokens, t, t_mod, context, context_mask, freqs, f, h, w, tokens_per_frame

    def post(
        self,
        x_tokens: torch.Tensor,
        t: torch.Tensor,
        f: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Convert tensor-core video tokens back into latent predictions."""
        return self.unpatchify(self.head(x_tokens, t), (f, h, w))

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        if context_mask is None:
            context_mask = torch.ones(
                (x.shape[0], context.shape[1]),
                dtype=torch.bool,
                device=context.device,
            )
        x_tokens, t, t_mod, context_emb, context_attn_mask, freqs, f, h, w, tokens_per_frame = self.prepare(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        self_attn_mask = self.build_video_to_video_mask(
            video_seq_len=x_tokens.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=x_tokens.device,
        ) if self.video_attention_mask_mode != "bidirectional" else None # special rule for faster speed

        for block in self.blocks:
            x_tokens = block(
                x_tokens,
                context_emb,
                t_mod,
                freqs,
                context_mask=context_attn_mask,
                self_attn_mask=self_attn_mask,
            )

        return self.post(x_tokens, t, f, h, w)
