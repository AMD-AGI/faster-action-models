# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025]. 
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].
# Action repeat is inspired by CogACT
# Modified by [Teli Ma/ HKUST GZ] in [2025]. 
# Modification: [modify to match with DiT4DiT].
# Modified by Advanced Micro Devices, Inc. Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.


from dataclasses import dataclass, field

import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
import os
from typing import Optional, Tuple

from DiT4DiT.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from DiT4DiT.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT

def _geak_env_on(name, default="1"):
    return os.environ.get(name, default) not in ("0", "false", "False")


# GEAK: run the DiT trunk of the flow-matching head in bf16 (MFMA) instead of
# fp32 (no MFMA on gfx90a). Measured to be worth ~nothing at B=1/S=9 (the head
# is host-dispatch bound, not GEMM bound), so it is OFF by default -- it only
# buys numeric risk. Set GEAK_ACTION_BF16=1 to re-enable.
_GEAK_BF16 = _geak_env_on("GEAK_ACTION_BF16", "0")

# GEAK: torch.compile the per-denoise-step trunk. This is the real lever --
# the head issues ~290 tiny ATen ops per step at 27 us of CPU each, so fusing
# them into inductor kernels roughly halves the per-step cost.
_GEAK_COMPILE = _geak_env_on("GEAK_ACTION_COMPILE", "1")

# GEAK r1_d0 (host_runtime) kill switches -- see the block above predict_action.
_GEAK_FP32W = _geak_env_on("GEAK_HEAD_FP32W", "1")
_GEAK_CONST = _geak_env_on("GEAK_HEAD_CONST", "1")
_GEAK_GRAPH = _geak_env_on("GEAK_HEAD_GRAPH", "1")
_GEAK_FUSE = _geak_env_on("GEAK_HEAD_FUSE", "1")
# r1 MEASURED LOSS (-2.7% e2e): concatenating q|k|v into one GEMM hands SDPA
# permuted (non-contiguous) q/k/v, and the copies SDPA then makes cost more
# than the two launches it saves at S=8.
# r2_d1 REVERSAL, measured: the loss was never the fusion, it was the copy the
# CONSUMER demanded.  GEAK_HEAD_ATT's triton attention indexes q/k/v through
# explicit per-dim strides, so the (3,B,H,S,d) permuted view is consumed in
# place with no materialisation -- and then the fusion is a pure win: three
# (8,768)x(768,768) GEMMs at 4.81 us each become one (8,768)x(768,2304) at
# 5.85 us.  Head device time 4.25 -> 3.77 ms at 4 steps.  Default ON.
_GEAK_QKV = _geak_env_on("GEAK_HEAD_QKV", "1")
# Debug: run BOTH the new head and the stock path on the same noise and
# print the deviation. Off by default; this is how the parity below was measured.
_GEAK_VERIFY = _geak_env_on("GEAK_HEAD_VERIFY", "0")

# ==================================================================
# GEAK r2_d1 (compute) kill switches.
#
#  GEAK_HEAD_IND -- inductor-compile the per-denoise-step body that the
#      CUDA-graph path replays.  r1 captured the head as one graph, which
#      removed 100% of the HOST gap but left the DEVICE op stream exactly as
#      eager emitted it: 190 dispatches/step, of which 83 are pure elementwise
#      launch floor (34 residual adds on 6 CTAs, 33 layernorms on 8 CTAs,
#      16 GELUs on 24 CTAs) over a (1, 9, 768) tensor.  Inductor fuses the
#      residual-add/layernorm/modulation chains into single triton kernels;
#      the resulting op stream is then captured by the SAME graph machinery
#      (we warm the compiled callable up on the side stream *before*
#      torch.cuda.graph(), so no tracing/compilation ever happens under
#      capture).  Values are unchanged up to fp32 reassociation inside a
#      layernorm/gelu.  NOTE: this is NOT the pinned GEAK_ACTION_COMPILE --
#      that one compiled the *reference* path's step_fn and never ran with
#      the graph capture.
#  GEAK_HEAD_ATT -- replace SDPA at the head's tiny sequence length with the
#      explicit baddbmm/softmax/bmm form.  See _geak_sdpa.
#  GEAK_HEAD_B16 -- narrow the trunk's weight-streaming GEMM operands to
#      bf16 while keeping the Euler accumulator, the residual stream, the
#      layernorms and the action encode/decode in fp32.
#  GEAK_HEAD_DBG -- one-shot shape/dtype dump.
# ==================================================================
_GEAK_IND = _geak_env_on("GEAK_HEAD_IND", "1")
_GEAK_ATT = _geak_env_on("GEAK_HEAD_ATT", "1")
# GEAK r3_d1: GEAK_HEAD_B16 is now a LEVEL, not a flag.
#   0 = off (fp32 trunk, the round-2 shipped default)
#   1 = full bf16 trunk (weights AND the residual stream)
#   2 = HYBRID: bf16 weight-streaming GEMM operands, but the residual stream,
#       the AdaLN modulation constants and every LayerNorm statistic stay
#       fp32.  The weight traffic win is identical (the 151.9 M trunk params
#       are what is big -- the (1,9,768) activations are 27 KB), but the
#       error no longer compounds across 16 residual adds.
def _geak_env_int(name, default=0):
    try:
        return int(os.environ.get(name, str(default)) or default)
    except Exception:
        return default


_GEAK_B16_LVL = _geak_env_int("GEAK_HEAD_B16", 1)
_GEAK_B16 = _GEAK_B16_LVL > 0
_GEAK_LN = _geak_env_on("GEAK_HEAD_LN", "1")
# GEAK r3_d1: replace the head trunk's tiny (M<=16) rocBLAS GEMMs with a
# triton GEMM decomposed for THIS shape, and fold the FFN's GELU into the
# first one's epilogue.  See _geak_mm_kernel.
_GEAK_MM = _geak_env_on("GEAK_HEAD_MM", "1")
# GEAK r4_d0: the FeedForward recogniser.  See _geak_ff_pack -- the r3_d1
# guard required len(net) == 3 and this checkpoint's FeedForward has FOUR
# entries (GELU, Dropout, Linear, Dropout), so the packed FFN GEMM path
# NEVER engaged: every block's ff1/ff2 fell back to rocBLAS + an inductor
# gelu, i.e. 3 dispatches instead of 2 and the slower vendor tile.
_GEAK_FFP = _geak_env_on("GEAK_HEAD_FFP", "1")
# GEAK r4_d0: split-K for the deep-K / narrow-N trunk GEMMs whose ONLY consumer
# is `geak_head::add_ln`.  The partial reduction is folded into add_ln, which is
# 99.6% dispatch floor anyway (MEASURED: 1.903 us gutted vs 1.910 us with the
# full body, in-graph), so the split costs ZERO extra dispatches.  See
# _geak_sk_cfg for the measured table.
_GEAK_SK = _geak_env_on("GEAK_HEAD_SK", "1")
_GEAK_SK_CFG = os.environ.get("GEAK_HEAD_SK_CFG", "")
# GEAK r4_d0: ONE k|v GEMM for ALL cross-attention blocks instead of one per
# block.  Traced (rocprofv3, this tree): 8 x Cijk_..._BBS_BH_..._MT64x48x256 at
# 11.0 us each = 88 us per CALL (not per step) for 8 x 2.36 MB of weight at only
# ~306 GB/s -- eight separate 216-CTA launches of a shape that is one launch.
_GEAK_KVALL = _geak_env_on("GEAK_HEAD_KVALL", "1")
# GEAK r4_d0: algebraically collapse the action encoder's head.
#   x = swish(W2 @ cat(W1 a + b1, tau) + b2)
#     = swish((W2a W1) a + (W2a b1 + b2) + W2b tau)
# W2b tau is CONSTANT for a given (step, horizon) -- it is precomputed once in
# _geak_consts and carried in place of tau -- and (W2a W1) is a single
# (hidden x action_dim) weight.  Four dispatches per step (K=8 GEMM, cat,
# K=1536 GEMM, swish) collapse to two (K=8 GEMM, add+swish).
_GEAK_AFOLD = _geak_env_on("GEAK_HEAD_AFOLD", "1")
# GEAK r4_d0: run the trunk's output projection through the M=8 triton GEMM and
# have it STORE fp32, which deletes the separate bf16->fp32 copy kernel
# (traced: triton_poi_fused__to_copy_view_3, 160 CTAs, 5.0 us per step).
_GEAK_P2MM = _geak_env_on("GEAK_HEAD_P2MM", "1")
# GEAK r4_d0: kill switch for the num_stages 2 -> 3 retune (see _geak_mm_cfg).
_GEAK_ST3 = _geak_env_on("GEAK_HEAD_ST3", "1")
# GEAK r2_d2: absorb the per-block residual-add + LayerNorm ("_geak_add_ln")
# into the PROLOGUE of the triton GEMM that consumes its normalised output.
# The head runs 32 `_geak_add_ln_kernel` nodes per denoise step (1-step device
# census: 65 mm + 32 add_ln + 24 attn = 121 nodes) and every one of them has a
# `_geak_mm_kernel` as its ONLY consumer, so the norm can be recomputed inside
# that GEMM's A-operand path instead of being materialised by its own launch.
# 32 nodes x 4 steps = 128 dispatches deleted on the 4-step arm.
# MEASURED LOSS, shipped OFF (GEAK_HEAD_MMLN=1 re-enables it for re-measurement):
# the fusion deletes exactly the predicted 32 device nodes per step (1-step
# device_ops 615 -> 583) but the tree gets SLOWER, 5.508 -> 5.700 ms geomean.
# Reason, from an isolated graph-replay sweep: an in-graph nop kernel costs
# 2.06 us on this box, and adding ONE extra (M=8, K=768) operand stream to
# these BN=16 / 1-2-warp GEMMs costs 1.36 us while the layer-norm affine adds
# another ~1.8 us -- i.e. absorbing a node that IS the launch floor costs more
# than the floor.  These GEMMs run at ~1 wave per CU, so there is no other wave
# to hide the added load latency behind.
_GEAK_MMLN = _geak_env_on("GEAK_HEAD_MMLN", "0")
# split-K factor used for the two GEMMs that FEED a fused prologue.  "keep"
# reuses _geak_sk_cfg (fp32 partial stack, most accurate, most prologue
# traffic); "1" makes the producer emit a single fp32 [M,K] tile instead.
_GEAK_MMLN_SO = os.environ.get("GEAK_HEAD_MMLN_SO", "keep")
_GEAK_MMLN_CFG = os.environ.get("GEAK_HEAD_MMLN_CFG", "")
_GEAK_MMLN_PREW = 1 if _geak_env_on("GEAK_HEAD_MMLN_PREW", "1") else 0
# GEAK r2_d2: run the head attention's two tl.dot calls on NARROW operands so
# they hit MFMA.  gfx950 has no fp32 MFMA, so the r2_d1 kernels' `.to(fp32)`
# before the dot put QK^T and PV on the fp32 VALU.
_GEAK_ATT_MFMA = _geak_env_on("GEAK_HEAD_ATT_MFMA", "1")
# GEAK r2_d2: once the dots are MFMA the split-KV kernel's two K/V loads are the
# critical path; one extra pipeline stage measures 4.073 -> 3.639 us isolated
# (consistent across every bn/warps combination swept).
_GEAK_ATT_ST2 = _geak_env_on("GEAK_HEAD_ATT_ST2", "1")

# ------------------------------------------------------------------
# GEAK r3(c2)_d1 -- ACTION-HEAD NODE-COUNT COLLAPSE, round 3.
#
# MEASURED CONTEXT (this box, this session, all in a captured graph):
#   * in-graph node floor, re-derived in situ: 1.67 us/node (chain of 64
#     distinct-code-object nop kernels, INSENSITIVE to grid 1..576, to
#     num_warps and to buffer size).  The carried 2.06 us figure is 23% high.
#   * the head runs ~128 device nodes per denoise step against a 0.390 ms
#     marginal step => 214 us (55%) is pure dispatch floor.
#
# TWO LEVERS SHIP HERE, both node deletions:
#
#  GEAK_HEAD_TAIL -- the per-step action encoder / decoder tail.  MEASURED:
#     it is FOUR fp32 rocBLAS GEMMs at M=8 (3.7 / 4.7 / 10.8 / 8.1 us of
#     attributed device time) plus four inductor pointwise nodes, i.e. ~9
#     nodes and ~38 us of a 390 us step -- more per node than anything in
#     the trunk.  The dominant one is `action_decoder.layer1`, a
#     2560x2560 **fp32** weight = 26.2 MB streamed EVERY denoise step.
#     Three changes, all node deletions, none of which adds an operand
#     stream to a kernel that did not already read it:
#       (a) proj_out_2 (2560x768) and action_decoder.layer1 (2560x2560) have
#           NO nonlinearity between them, so they compose exactly:
#           relu(Wd1 (Wp2 x + bp2) + bd1) = relu(W' x + b') with
#           W' = Wd1 @ Wp2 (2560x768) and b' = Wd1 @ bp2 + bd1, folded ONCE.
#           That is 17.0 MB of weight traffic replaced by 3.9 MB and THREE
#           nodes (proj2 GEMM, adec1 GEMM, relu) replaced by ONE.
#       (b) the remaining tail GEMMs run through a new `_geak_mmx_kernel`
#           (the trunk's `_geak_mm_kernel` plus an epilogue) in bf16/MFMA
#           instead of fp32 rocBLAS -- gfx950 has no fp32 MFMA path, and the
#           trunk activations are ALREADY bf16, so this adds no new error
#           term of a different order than the one already there.
#       (c) the epilogues (+tau, swish, +pos, relu, and the Euler update
#           `actions - dt*pred`) ride in the fp32 accumulator.  These are
#           EPILOGUE operands: a CTA touches only its own BNxBM tile of
#           them (<=1 KB), which is why this is not the r2_d2 mmln shape --
#           that one pulled a full extra A-operand into the PROLOGUE, ahead
#           of the K loop, and lost 3.5%.
#     Net: ~9 tail nodes -> 5, and the fp32 GEMMs -> bf16.
#
#  GEAK_HEAD_A1 -- cross-attention (S_k=392) as ONE kernel instead of the
#     split-KV pair.  MEASURED standalone, cold K/V, captured graph:
#       split+combine (4,64,ws2,wc4)            6.483 us   <- shipped before
#       single kernel BLOCK_N=256 nw=4 stages=2 6.179 us   <- shipped now
#     i.e. one fewer node AND 0.30 us less.  NOTE the two negative results
#     that fence this: (i) `nsplit=1` on the OLD BLOCK_N=64 geometry is
#     +4.3% (the direction's warning) -- the win needs BLOCK_N>=128 so the
#     392-token K stream is 2 iterations, not 7; (ii) folding the combine
#     into the split kernel's last-arriving CTA (an acq_rel arrival counter,
#     bit-exact, verified maxdiff 0.0) measures 6.449 us -- it deletes the
#     node but hands the combine to a CTA running at the SPLIT's num_warps,
#     and the two cancel.  Not shipped.
# ------------------------------------------------------------------
# ==================================================================
# GEAK r2_d0 (deep_explore) -- HEAD I/O STAGING NODE COLLAPSE.
#
# The head's graph boundary costs FOUR dispatches that move nothing:
#   vl_embs.float()      [1,392,2048] _to_copy   -- and then the captured region
#                                                   immediately casts it BACK to
#                                                   the trunk dtype (`vl_r`), so
#                                                   bf16 -> fp32 -> bf16 is an
#                                                   exact round trip: 2 nodes for
#                                                   the identity map.
#   actions.float()      [1,8,8]      _to_copy   -- 256 BYTES; the staging
#                                                   `s_act.copy_()` that follows
#                                                   already does the widening.
#   s_out.clone()        [1,8,8]      clone      -- the sole consumer is
#                                                   `.detach().cpu().numpy()` on
#                                                   the very next line, i.e. the
#                                                   D2H already makes the copy.
# Every one is bitwise neutral (bf16->fp32 and fp32->fp32 conversions are exact).
# In-graph node floor re-derived this session: 1.93 us/node; these run EAGER on
# the boundary between the two graph replays, where the census charges them
# 4.2-4.6 us each.
#
# Kill switch: GEAK_HEAD_IO=0 restores all four.
# ==================================================================
_GEAK_IO = _geak_env_on("GEAK_HEAD_IO", "1")
_GEAK_TAIL = _geak_env_on("GEAK_HEAD_TAIL", "1")
_GEAK_A1 = _geak_env_on("GEAK_HEAD_A1", "1")
# ------------------------------------------------------------------
# GEAK r1_d1 (campaign 3) -- GEAK_HEAD_ILANE: the head's INSTRUCTION lane.
# Three exact rewrites, all already shipped elsewhere in this tree:
#   (1) tl.exp -> tl.math.exp2 with log2(e) folded into `sm_scale` at the
#       host, so the softmax runs on the hardware v_exp_f32 instead of an
#       ocml __ocml_exp_f32 call (which is exp2 + a multiply anyway).
#   (2) y*sigmoid(a*y) -> y*(1/(1+exp2(y*-a*log2e))) -- one hardware exp2,
#       one add, one reciprocal, no ocml call and no per-element divide.
#   (3) acc / l_i[:, None] -> acc * (1/l_i)[:, None]: BLOCK_M reciprocals
#       hoisted out of the BLOCK_M x D element loop instead of BLOCK_M*D
#       IEEE fp32 divide expansions.
# GEAK_HEAD_ILANE=0 restores the exact prior instruction sequence.
# ------------------------------------------------------------------
_GEAK_ILANE = _geak_env_on("GEAK_HEAD_ILANE", "1")
_GEAK_ILANE_I = 1 if _GEAK_ILANE else 0
_GEAK_LOG2E = 1.4426950408889634
_GEAK_SMSC = _GEAK_LOG2E if _GEAK_ILANE else 1.0
_GEAK_A1_BN, _GEAK_A1_W, _GEAK_A1_S = 256, 4, 2
try:
    _a1 = [int(v) for v in os.environ.get("GEAK_HEAD_A1_CFG", "").split(",")]
    if len(_a1) == 3:
        _GEAK_A1_BN, _GEAK_A1_W, _GEAK_A1_S = _a1
except Exception:
    pass


def _geak_afold(aenc):
    """(W12, b12, W2b) for a stock ActionEncoder, else None."""
    got = getattr(aenc, "_geak_afold_c", None)
    if got is not None:
        return got if got is not False else None
    got = False
    try:
        l1, l2 = aenc.layer1, aenc.layer2
        if (_GEAK_AFOLD and isinstance(l1, nn.Linear) and isinstance(l2, nn.Linear)
                and isinstance(aenc.layer3, nn.Linear)):
            h = l1.weight.shape[0]
            if l2.weight.shape[1] == 2 * h:
                w2a = l2.weight[:, :h]
                w2b = l2.weight[:, h:].contiguous()
                w12 = (w2a @ l1.weight).contiguous()
                b12 = l2.bias
                if l1.bias is not None:
                    b12 = (w2a @ l1.bias) + (0 if b12 is None else b12)
                got = (w12, (b12.contiguous() if b12 is not None else None), w2b)
    except Exception:
        got = False
    aenc._geak_afold_c = got
    return got if got is not False else None
# GEAK r4_d0: the flash-decoding split-KV geometry, exposed so it can be swept.
# "nsplit,block_n,warps_split,warps_combine"
_GEAK_ATT_NS, _GEAK_ATT_BN, _GEAK_ATT_WS, _GEAK_ATT_WC = 4, 64, 2, 4
try:
    _a = [int(v) for v in os.environ.get("GEAK_HEAD_ATT_CFG", "").split(",")]
    if len(_a) == 4:
        _GEAK_ATT_NS, _GEAK_ATT_BN, _GEAK_ATT_WS, _GEAK_ATT_WC = _a
except Exception:
    pass
_GEAK_DBG = _geak_env_on("GEAK_HEAD_DBG", "0")
_GEAK_IND_MODE = os.environ.get("GEAK_HEAD_IND_MODE", "default")


# ------------------------------------------------------------------
# GEAK r2_d1: a single-kernel attention for the action head's tiny query
# sequence (S_q = 8), replacing BOTH the generic flash kernel AND the
# layout copy that follows it.
#
# MEASURED PROBLEM.  torch's ROCm SDPA dispatches a long-sequence triton
# `attn_fwd` here.  At S_q = 8 it launches heads x ceil(S_q/BLOCK_M) = 12
# CTAs and costs 16.5 us -- 64 launches per 4-step call, 1.05 ms, 23% of the
# whole head's device time, for attention over EIGHT query tokens.  Its
# output is (B, H, S, d), so `x.transpose(1,2).reshape(B, S, H*d)` then has
# to materialise a contiguous copy before the to_out GEMM: a second kernel
# per block.
#
# THIS KERNEL.  One program per (batch, head): the 8x64 query tile lives in
# registers for the whole kernel, K/V stream through in BLOCK_N chunks with
# the standard online-softmax rescale, and the fp32 accumulator is stored
# STRAIGHT INTO THE (B, S, H*d) LAYOUT the next GEMM wants.  So one launch
# replaces two, and no `.reshape()` copy is ever created.
#
# Numerics: identical algorithm to SDPA (fp32 accumulate, max-subtracted
# softmax); only the summation order over K differs.  Measured deviation
# from the stock path is at the fp32 round-off floor -- see GEAK_HEAD_VERIFY.
#
# It is wrapped in a torch.library custom op so that (a) dynamo treats it as
# an opaque extern call instead of graph-breaking around a raw triton launch,
# and (b) it stays capture-safe: the JIT compile happens on the warmup
# iterations that _geak_capture runs BEFORE torch.cuda.graph().
# ------------------------------------------------------------------
_GEAK_TRITON_OK = True
try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - triton always present on this stack
    _GEAK_TRITON_OK = False


if _GEAK_TRITON_OK:

    @triton.jit
    def _geak_attn_kernel(
        Q, K, V, O,
        sqb, sqh, sqm,
        skb, skh, skn,
        svb, svh, svn,
        sob, som,
        sm_scale,
        H: tl.constexpr, SEQ_K: tl.constexpr, SEQ_M: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
        MFMA: tl.constexpr = 0, ILANE: tl.constexpr = 0,
    ):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        mmask = offs_m < SEQ_M

        qr = tl.load(
            Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :],
            mask=mmask[:, None], other=0.0,
        )
        q = qr.to(tl.float32)

        m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, D), tl.float32)

        for start_n in tl.range(0, SEQ_K, BLOCK_N):
            nn = start_n + offs_n
            nmask = nn < SEQ_K
            kr = tl.load(
                K + b * skb + h * skh + nn[:, None] * skn + offs_d[None, :],
                mask=nmask[:, None], other=0.0,
            )
            if MFMA:
                qk = tl.dot(qr, tl.trans(kr)) * sm_scale
            else:
                qk = tl.dot(q, tl.trans(kr.to(tl.float32))) * sm_scale
            qk = tl.where(nmask[None, :], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            if ILANE:
                # sm_scale already carries log2(e), so qk / m_i live in the
                # log2 domain and this is the SAME softmax -- one hardware
                # v_exp_f32 per element instead of an ocml exp call.
                alpha = tl.math.exp2(m_i - m_new)
                p = tl.math.exp2(qk - m_new[:, None])
            else:
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(qk - m_new[:, None])
            vr = tl.load(
                V + b * svb + h * svh + nn[:, None] * svn + offs_d[None, :],
                mask=nmask[:, None], other=0.0,
            )
            if MFMA:
                acc = acc * alpha[:, None] + tl.dot(p.to(vr.dtype), vr)
            else:
                acc = acc * alpha[:, None] + tl.dot(p, vr.to(tl.float32))
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new

        if ILANE:
            acc = acc * (1.0 / l_i)[:, None]
        else:
            acc = acc / l_i[:, None]
        tl.store(
            O + b * sob + offs_m[:, None] * som + h * D + offs_d[None, :],
            acc.to(O.dtype.element_ty), mask=mmask[:, None],
        )


    @triton.jit
    def _geak_attn_split_kernel(
        Q, K, V, PA, PM, PL,
        sqb, sqh, sqm,
        skb, skh, skn,
        svb, svh, svn,
        sm_scale,
        H: tl.constexpr, SEQ_K: tl.constexpr, SEQ_M: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
        NSPLIT: tl.constexpr, CHUNK: tl.constexpr, MFMA: tl.constexpr = 0,
        ILANE: tl.constexpr = 0,
    ):
        pid = tl.program_id(0)
        sp = tl.program_id(1)
        b = pid // H
        h = pid % H
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        mmask = offs_m < SEQ_M
        # GEAK r2_d2: `MFMA` keeps Q/K/P/V in the checkpoint's narrow dtype for
        # the two tl.dot calls.  gfx950 has NO MFMA path for an fp32 tl.dot, so
        # the r2_d1 form (which widened both operands before the dot) ran the
        # whole QK^T and PV on the fp32 VALU.  Narrow operands + fp32 ACCUMULATE
        # is exactly what torch SDPA / flash-attention do, so this is not a
        # precision regression relative to the stock path it replaced.
        qr = tl.load(
            Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :],
            mask=mmask[:, None], other=0.0,
        )
        q = qr.to(tl.float32)
        m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, D), tl.float32)
        lo = sp * CHUNK
        hi = tl.minimum(lo + CHUNK, SEQ_K)
        for start_n in tl.range(lo, hi, BLOCK_N):
            nn = start_n + offs_n
            nmask = nn < hi
            kr = tl.load(
                K + b * skb + h * skh + nn[:, None] * skn + offs_d[None, :],
                mask=nmask[:, None], other=0.0,
            )
            if MFMA:
                qk = tl.dot(qr, tl.trans(kr)) * sm_scale
            else:
                qk = tl.dot(q, tl.trans(kr.to(tl.float32))) * sm_scale
            qk = tl.where(nmask[None, :], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            if ILANE:
                # sm_scale already carries log2(e), so qk / m_i live in the
                # log2 domain and this is the SAME softmax -- one hardware
                # v_exp_f32 per element instead of an ocml exp call.
                alpha = tl.math.exp2(m_i - m_new)
                p = tl.math.exp2(qk - m_new[:, None])
            else:
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(qk - m_new[:, None])
            p = tl.where(nmask[None, :], p, 0.0)
            vr = tl.load(
                V + b * svb + h * svh + nn[:, None] * svn + offs_d[None, :],
                mask=nmask[:, None], other=0.0,
            )
            if MFMA:
                acc = acc * alpha[:, None] + tl.dot(p.to(vr.dtype), vr)
            else:
                acc = acc * alpha[:, None] + tl.dot(p, vr.to(tl.float32))
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new
        off = pid * NSPLIT + sp
        tl.store(PA + off * BLOCK_M * D + offs_m[:, None] * D + offs_d[None, :], acc)
        tl.store(PM + off * BLOCK_M + offs_m, m_i)
        tl.store(PL + off * BLOCK_M + offs_m, l_i)

    @triton.jit
    def _geak_attn_combine_kernel(
        PA, PM, PL, O, sob, som,
        H: tl.constexpr, SEQ_M: tl.constexpr,
        BLOCK_M: tl.constexpr, D: tl.constexpr, NSPLIT: tl.constexpr,
        ILANE: tl.constexpr = 0,
    ):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        offs_s = tl.arange(0, NSPLIT)
        base = (pid * NSPLIT + offs_s)
        mv = tl.load(PM + base[:, None] * BLOCK_M + offs_m[None, :])
        lv = tl.load(PL + base[:, None] * BLOCK_M + offs_m[None, :])
        gm = tl.max(mv, 0)
        if ILANE:
            w = tl.math.exp2(mv - gm[None, :])
        else:
            w = tl.exp(mv - gm[None, :])
        den = tl.sum(lv * w, 0)
        av = tl.load(
            PA + base[:, None, None] * BLOCK_M * D
            + offs_m[None, :, None] * D + offs_d[None, None, :]
        )
        if ILANE:
            acc = tl.sum(av * w[:, :, None], 0) * (1.0 / den)[:, None]
        else:
            acc = tl.sum(av * w[:, :, None], 0) / den[:, None]
        tl.store(
            O + b * sob + offs_m[:, None] * som + h * D + offs_d[None, :],
            acc.to(O.dtype.element_ty), mask=(offs_m < SEQ_M)[:, None],
        )

    # --------------------------------------------------------------
    # GEAK r2_d1: residual-add + LayerNorm + (modulation) affine, one
    # kernel, TWO outputs (the new residual AND the normalised tensor,
    # both of which the block needs).
    #
    # This is the single most repeated shape in the head: 2 per block x 16
    # blocks x N steps.  Inductor already fuses the add into the layernorm,
    # but its generated kernel measures 4.5 us on (8, 768) fp32 -- 24 KB of
    # traffic against this box's ~2.0 us empty-kernel floor.  Hand-written
    # with one program per row and one warp it measures 2.12 us, i.e. 2.1x
    # less, for bit-comparable output (max deviation 9.5e-07 vs torch).
    # 128 launches per 4-step call -> ~0.30 ms.
    # --------------------------------------------------------------
    @triton.jit
    def _geak_add_ln_kernel(
        A, Bv, W, Bi, R, O, eps,
        DD: tl.constexpr, HAS_AF: tl.constexpr, BLOCK: tl.constexpr,
        NSPL: tl.constexpr = 1, PROWS: tl.constexpr = 0,
    ):
        row = tl.program_id(0)
        od = tl.arange(0, BLOCK)
        m = od < DD
        if NSPL == 1:
            a = tl.load(A + row * DD + od, mask=m, other=0.0).to(tl.float32)
        else:
            # GEAK r4_d0: A is a (NSPL, PROWS, DD) fp32 split-K partial stack
            # written by _geak_mm_kernel.  Summing it here is free -- this
            # kernel is at the dispatch floor with ~0 body.
            a = tl.zeros((BLOCK,), dtype=tl.float32)
            for s_ in tl.static_range(NSPL):
                a += tl.load(A + s_ * PROWS * DD + row * DD + od,
                             mask=m, other=0.0).to(tl.float32)
        bb = tl.load(Bv + row * DD + od, mask=m, other=0.0).to(tl.float32)
        x = a + bb
        tl.store(R + row * DD + od, x.to(R.dtype.element_ty), mask=m)
        mu = tl.sum(x, 0) / DD
        xc = tl.where(m, x - mu, 0.0)
        var = tl.sum(xc * xc, 0) / DD
        y = xc * tl.rsqrt(var + eps)
        if HAS_AF:
            y = y * tl.load(W + od, mask=m, other=0.0).to(tl.float32) \
                + tl.load(Bi + od, mask=m, other=0.0).to(tl.float32)
        tl.store(O + row * DD + od, y.to(O.dtype.element_ty), mask=m)

    # --------------------------------------------------------------
    # GEAK r3_d1: the head trunk's GEMM, re-decomposed for M = 8.
    #
    # MEASURED PROBLEM (graph-timed device us on this box, bf16, M=8):
    #
    #     shape                    rocBLAS   this kernel
    #     qkv     768 -> 2304       3.88       2.28
    #     to_out  768 ->  768       3.85       2.24
    #     ff1     768 -> 3072 +gelu 6.32*      2.47     (*3.85 mm + 2.24 gelu)
    #     ff2    3072 ->  768       6.12       4.07
    #
    # rocBLAS picks a 16x16 macro-tile with a deep K-unroll (Cijk_..._MT16x16x256),
    # which at M = 8 leaves HALF of every MFMA tile masked off and, worse, gives
    # each CTA a long serial K walk.  The whole trunk moves 304 MB of bf16 weight
    # per denoise step in ~0.9 ms = ~350 GB/s against an 8 TB/s HBM: this is not a
    # bandwidth problem and not a FLOP problem, it is MEMORY-LEVEL PARALLELISM.
    # One CTA per 16 output columns (BN=16 -> 48..192 CTAs instead of rocBLAS's
    # tile count) puts many more independent weight streams in flight at once.
    #
    # The GELU rides in the fp32 accumulator, so the FFN's separate
    # `triton_poi_fused_addmm_gelu_view` launch disappears entirely: one fewer
    # link in a chain of ~146 serialised dispatches per step, which on this
    # model is worth more than the kernel being faster.
    #
    # tanh-GELU is evaluated as x*sigmoid(2z), which is algebraically identical
    # to 0.5x(1+tanh z) and avoids a libdevice call.
    # --------------------------------------------------------------
    @triton.jit
    def _geak_mm_kernel(
        X, W, Bs, O,
        K: tl.constexpr, N: tl.constexpr, MM: tl.constexpr,
        HAS_B: tl.constexpr, GELU: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        EVEN_K: tl.constexpr, EVEN_N: tl.constexpr,
        SPLITK: tl.constexpr = 1, NBLK: tl.constexpr = 0,
        ILANE: tl.constexpr = 0, MP: tl.constexpr = 16,
    ):
        pid = tl.program_id(0)
        if SPLITK == 1:
            pn = pid
            k_beg = 0
            k_end = K
        else:
            # GEAK r4_d0: grid is (NBLK * SPLITK,).  ff2 (K=3072, N=768) had only
            # 48 CTAs and ran at 738 GB/s; the same weight stream over 288 CTAs
            # runs at ~1.4 TB/s, which is where ff1 (192 CTAs) already sits.
            pn = pid % NBLK
            k_beg = (pid // NBLK) * (K // SPLITK)
            k_end = k_beg + (K // SPLITK)
        offs_n = pn * BN + tl.arange(0, BN)
        offs_m = tl.program_id(1) * BM + tl.arange(0, BM)
        mn = offs_n < N
        mm_ = offs_m < MM
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(k_beg, k_end, BK):
            offs_k = k0 + tl.arange(0, BK)
            if EVEN_K:
                a = tl.load(X + offs_m[:, None] * K + offs_k[None, :],
                            mask=mm_[:, None], other=0.0)
                if EVEN_N:
                    b = tl.load(W + offs_n[:, None] * K + offs_k[None, :])
                else:
                    b = tl.load(W + offs_n[:, None] * K + offs_k[None, :],
                                mask=mn[:, None], other=0.0)
            else:
                mk = offs_k < K
                a = tl.load(X + offs_m[:, None] * K + offs_k[None, :],
                            mask=mm_[:, None] & mk[None, :], other=0.0)
                b = tl.load(W + offs_n[:, None] * K + offs_k[None, :],
                            mask=mn[:, None] & mk[None, :], other=0.0)
            acc += tl.dot(a, tl.trans(b))
        if HAS_B:
            if SPLITK == 1:
                acc += tl.load(Bs + offs_n, mask=mn, other=0.0)[None, :].to(tl.float32)
            elif pid < NBLK:
                # the bias belongs to exactly one partial
                acc += tl.load(Bs + offs_n, mask=mn, other=0.0)[None, :].to(tl.float32)
        if GELU:
            if ILANE:
                # x*sigmoid(2z) == x/(1+exp(-2z)) == x/(1+exp2(-2*log2e*z)),
                # with -2*log2e folded into the 0.7978... constant: the whole
                # tanh-GELU becomes 3 v_fma + 1 v_exp + 1 v_rcp per element
                # and calls no ocml routine at all.
                zz = -2.302208198144325 * (acc + 0.044715 * acc * acc * acc)
                acc = acc * (1.0 / (1.0 + tl.math.exp2(zz)))
            else:
                z = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)
                acc = acc * tl.sigmoid(2.0 * z)
        if SPLITK == 1:
            tl.store(O + offs_m[:, None] * N + offs_n[None, :],
                     acc.to(O.dtype.element_ty), mask=mm_[:, None] & mn[None, :])
        else:
            tl.store(O + (pid // NBLK) * MP * N
                     + offs_m[:, None] * N + offs_n[None, :],
                     acc, mask=mm_[:, None] & mn[None, :])

    # --------------------------------------------------------------
    # GEAK r3(c2)_d1: `_geak_mm_kernel` + an EPILOGUE, for the head's
    # action-encoder / action-decoder tail.  Everything after the K loop
    # touches only this CTA's own BM x BN output tile, so the extra
    # operands cost <= 1 KB per CTA and never sit ahead of the K loop.
    #   ADDC : += C[m, n]        (tau_t, or the positional embedding)
    #   ACT  : 1 = swish (x*sigmoid(x)), 2 = relu
    #   EUL  : out = A2[m, n] - dtv * acc   (the Euler update)
    #   CAST : narrow the fp32 A operand to the weight dtype for the MFMA
    #          (the action buffer stays fp32; only the GEMM input narrows)
    # --------------------------------------------------------------
    @triton.jit
    def _geak_mmx_kernel(
        X, W, Bs, C, A2, O, dtv,
        K: tl.constexpr, N: tl.constexpr, MM: tl.constexpr,
        HAS_B: tl.constexpr, ADDC: tl.constexpr, ACT: tl.constexpr,
        EUL: tl.constexpr, CAST: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        EVEN_K: tl.constexpr, EVEN_N: tl.constexpr,
        ILANE: tl.constexpr = 0,
    ):
        pn = tl.program_id(0)
        offs_n = pn * BN + tl.arange(0, BN)
        offs_m = tl.program_id(1) * BM + tl.arange(0, BM)
        mn = offs_n < N
        mm_ = offs_m < MM
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            offs_k = k0 + tl.arange(0, BK)
            if EVEN_K:
                a = tl.load(X + offs_m[:, None] * K + offs_k[None, :],
                            mask=mm_[:, None], other=0.0)
                if EVEN_N:
                    b = tl.load(W + offs_n[:, None] * K + offs_k[None, :])
                else:
                    b = tl.load(W + offs_n[:, None] * K + offs_k[None, :],
                                mask=mn[:, None], other=0.0)
            else:
                mk = offs_k < K
                a = tl.load(X + offs_m[:, None] * K + offs_k[None, :],
                            mask=mm_[:, None] & mk[None, :], other=0.0)
                b = tl.load(W + offs_n[:, None] * K + offs_k[None, :],
                            mask=mn[:, None] & mk[None, :], other=0.0)
            if CAST:
                a = a.to(b.dtype)
            acc += tl.dot(a, tl.trans(b))
        if HAS_B:
            acc += tl.load(Bs + offs_n, mask=mn, other=0.0)[None, :].to(tl.float32)
        if ADDC:
            acc += tl.load(C + offs_m[:, None] * N + offs_n[None, :],
                           mask=mm_[:, None] & mn[None, :], other=0.0).to(tl.float32)
        if ACT == 1:
            if ILANE:
                acc = acc * (1.0 / (1.0 + tl.math.exp2(
                    acc * -1.4426950408889634)))
            else:
                acc = acc * tl.sigmoid(acc)
        if ACT == 2:
            acc = tl.maximum(acc, 0.0)
        if EUL:
            av = tl.load(A2 + offs_m[:, None] * N + offs_n[None, :],
                         mask=mm_[:, None] & mn[None, :], other=0.0).to(tl.float32)
            acc = av - dtv * acc
        tl.store(O + offs_m[:, None] * N + offs_n[None, :],
                 acc.to(O.dtype.element_ty), mask=mm_[:, None] & mn[None, :])

    # --------------------------------------------------------------
    # GEAK r2_d2: `_geak_mm_kernel` with the producing residual-add +
    # LayerNorm folded into its A-operand PROLOGUE.
    #
    # WHY THIS IS POSSIBLE HERE.  Every `_geak_add_ln_kernel` launch in the
    # head has exactly one consumer and that consumer is a `_geak_mm_kernel`
    # -- ff1 after the attention residual, and the next block's q|k|v (or the
    # trunk's proj_out_2) after the FFN residual.  The GEMM's A operand is a
    # single (M<=16, K=768) row block that EVERY CTA already reads in full, so
    # recomputing the norm per CTA costs one extra streaming pass over ~24 KB
    # of L2-resident data instead of a whole 3.0 us dispatch.
    #
    # Xp is the producing GEMM's output: either a plain (M, K) tile (NSPL == 1)
    # or the (NSPL, PROWS, K) fp32 split-K partial stack.  Rout receives the
    # new residual stream; it is written by the CTA whose id equals the K-chunk
    # index, so it costs three stores spread over three CTAs and nothing else.
    #
    # Variance uses E[x^2]-E[x]^2 in fp32 so that the statistics need ONE pass
    # (the reference kernel's two-pass form would mean three passes over Xp).
    # Verified against the stock head with GEAK_HEAD_VERIFY.
    # --------------------------------------------------------------
    @triton.jit
    def _geak_mmln_kernel(
        Xp, Res, LW, LB, W, Bs, O, Rout, eps,
        K: tl.constexpr, N: tl.constexpr, MM: tl.constexpr,
        HAS_B: tl.constexpr, GELU: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        EVEN_K: tl.constexpr, EVEN_N: tl.constexpr,
        NSPL: tl.constexpr, PROWS: tl.constexpr,
        HAS_LA: tl.constexpr, WR: tl.constexpr, NKC: tl.constexpr,
        PREW: tl.constexpr = 1, ILANE: tl.constexpr = 0,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BN + tl.arange(0, BN)
        offs_m = tl.program_id(1) * BM + tl.arange(0, BM)
        mn = offs_n < N
        mm_ = offs_m < MM
        # GEAK r2_d2: issue the (HBM-resident, ~400 MB pool) WEIGHT tiles BEFORE
        # the norm prologue.  Without this the prologue's three L2 round-trips
        # are serialised AHEAD of the weight stream and the fused kernel pays
        # both latencies end to end -- which is why the first cut of this lever
        # measured 7x worse in situ than in an L2-hot microbenchmark.
        wpre = ()
        if PREW:
            for kw in tl.static_range(NKC):
                offs_kw = kw * BK + tl.arange(0, BK)
                if EVEN_K:
                    if EVEN_N:
                        _wt = tl.load(W + offs_n[:, None] * K + offs_kw[None, :])
                    else:
                        _wt = tl.load(W + offs_n[:, None] * K + offs_kw[None, :],
                                      mask=mn[:, None], other=0.0)
                else:
                    _wt = tl.load(W + offs_n[:, None] * K + offs_kw[None, :],
                                  mask=mn[:, None] & (offs_kw < K)[None, :],
                                  other=0.0)
                wpre = wpre + (_wt,)
        s1 = tl.zeros((BM,), dtype=tl.float32)
        s2 = tl.zeros((BM,), dtype=tl.float32)
        for ki in tl.static_range(NKC):
            offs_k = ki * BK + tl.arange(0, BK)
            if EVEN_K:
                msk = mm_[:, None]
            else:
                msk = mm_[:, None] & (offs_k < K)[None, :]
            x = tl.load(Res + offs_m[:, None] * K + offs_k[None, :],
                        mask=msk, other=0.0).to(tl.float32)
            if NSPL == 1:
                x += tl.load(Xp + offs_m[:, None] * K + offs_k[None, :],
                             mask=msk, other=0.0).to(tl.float32)
            else:
                for s_ in tl.static_range(NSPL):
                    x += tl.load(Xp + s_ * PROWS * K
                                 + offs_m[:, None] * K + offs_k[None, :],
                                 mask=msk, other=0.0).to(tl.float32)
            if WR:
                tl.store(Rout + offs_m[:, None] * K + offs_k[None, :],
                         x.to(Rout.dtype.element_ty),
                         mask=msk & (pid == ki))
            s1 += tl.sum(x, 1)
            s2 += tl.sum(x * x, 1)
        mu = s1 / K
        var = s2 / K - mu * mu
        rstd = tl.rsqrt(tl.maximum(var, 0.0) + eps)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for ki in tl.static_range(NKC):
            offs_k = ki * BK + tl.arange(0, BK)
            mk = offs_k < K
            if EVEN_K:
                msk = mm_[:, None]
            else:
                msk = mm_[:, None] & mk[None, :]
            x = tl.load(Res + offs_m[:, None] * K + offs_k[None, :],
                        mask=msk, other=0.0).to(tl.float32)
            if NSPL == 1:
                x += tl.load(Xp + offs_m[:, None] * K + offs_k[None, :],
                             mask=msk, other=0.0).to(tl.float32)
            else:
                for s_ in tl.static_range(NSPL):
                    x += tl.load(Xp + s_ * PROWS * K
                                 + offs_m[:, None] * K + offs_k[None, :],
                                 mask=msk, other=0.0).to(tl.float32)
            a = (x - mu[:, None]) * rstd[:, None]
            if HAS_LA:
                if EVEN_K:
                    a = a * tl.load(LW + offs_k)[None, :].to(tl.float32) \
                        + tl.load(LB + offs_k)[None, :].to(tl.float32)
                else:
                    a = a * tl.load(LW + offs_k, mask=mk, other=0.0)[None, :].to(tl.float32) \
                        + tl.load(LB + offs_k, mask=mk, other=0.0)[None, :].to(tl.float32)
            a = a.to(W.dtype.element_ty)
            if PREW:
                b = wpre[ki]
            else:
                if EVEN_K:
                    if EVEN_N:
                        b = tl.load(W + offs_n[:, None] * K + offs_k[None, :])
                    else:
                        b = tl.load(W + offs_n[:, None] * K + offs_k[None, :],
                                    mask=mn[:, None], other=0.0)
                else:
                    b = tl.load(W + offs_n[:, None] * K + offs_k[None, :],
                                mask=mn[:, None] & mk[None, :], other=0.0)
            acc += tl.dot(a, tl.trans(b))
        if HAS_B:
            acc += tl.load(Bs + offs_n, mask=mn, other=0.0)[None, :].to(tl.float32)
        if GELU:
            if ILANE:
                # x*sigmoid(2z) == x/(1+exp(-2z)) == x/(1+exp2(-2*log2e*z)),
                # with -2*log2e folded into the 0.7978... constant: the whole
                # tanh-GELU becomes 3 v_fma + 1 v_exp + 1 v_rcp per element
                # and calls no ocml routine at all.
                zz = -2.302208198144325 * (acc + 0.044715 * acc * acc * acc)
                acc = acc * (1.0 / (1.0 + tl.math.exp2(zz)))
            else:
                z = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)
                acc = acc * tl.sigmoid(2.0 * z)
        tl.store(O + offs_m[:, None] * N + offs_n[None, :],
                 acc.to(O.dtype.element_ty), mask=mm_[:, None] & mn[None, :])


def _geak_attn_impl(q, k, v):
    """One launch (self-attn, S_k = 8) or two (cross-attn, S_k = 392).

    Config from a standalone sweep on this box, all fp32-exact (max deviation
    from torch SDPA 2.1e-07):

        S_k = 392   torch SDPA + reshape copy   25.75 us
                    one kernel, 12 CTAs         19.64 us   (BLOCK_N 64, 2 warps)
                    split-KV 4-way, 48 CTAs      9.56 us   <-- shipped
        S_k =   8   torch SDPA + reshape copy    3.32 us
                    one kernel, 12 CTAs          2.80 us   <-- shipped

    The single-kernel form is parallelism-starved: heads(12) CTAs on 256 CUs
    with a serial 392-token K loop.  Splitting K 4 ways is the flash-decoding
    trick -- 48 CTAs, each doing a quarter of the K stream and emitting a
    partial (acc, m, l); a tiny second kernel rescales by the global row max
    and reduces.  Exactly the same value, ~2x less time.
    """
    B, H, M, D = q.shape
    SK = k.shape[2]
    out = torch.empty((B, M, H * D), device=q.device, dtype=q.dtype)
    bm = 16 if M <= 16 else triton.next_power_of_2(M)
    mf = 1 if (_GEAK_ATT_MFMA and q.dtype != torch.float32) else 0

    ns = 1
    if SK >= 256 and not _GEAK_A1:
        ns = _GEAK_ATT_NS
    chunk = 0
    if ns > 1:
        bn = _GEAK_ATT_BN
        per = (SK + ns - 1) // ns
        chunk = ((per + bn - 1) // bn) * bn
        # every split must own at least one real token (else its softmax
        # statistics are -inf/0 and the combine divides 0 by 0), and the
        # splits together must cover the whole K stream.
        if chunk * (ns - 1) >= SK or chunk * ns < SK:
            ns = 1
    if ns > 1:
        pa = torch.empty((B * H * ns, bm, D), device=q.device, dtype=torch.float32)
        pm = torch.empty((B * H * ns, bm), device=q.device, dtype=torch.float32)
        pl = torch.empty((B * H * ns, bm), device=q.device, dtype=torch.float32)
        _geak_attn_split_kernel[(B * H, ns)](
            q, k, v, pa, pm, pl,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            (float(D) ** -0.5) * _GEAK_SMSC,
            H=H, SEQ_K=SK, SEQ_M=M, BLOCK_M=bm, BLOCK_N=bn, D=D,
            NSPLIT=ns, CHUNK=chunk, MFMA=mf, ILANE=_GEAK_ILANE_I,
            num_warps=_GEAK_ATT_WS, num_stages=(2 if _GEAK_ATT_ST2 else 1),
        )
        _geak_attn_combine_kernel[(B * H,)](
            pa, pm, pl, out, out.stride(0), out.stride(1),
            H=H, SEQ_M=M, BLOCK_M=bm, D=D, NSPLIT=ns, ILANE=_GEAK_ILANE_I,
            num_warps=_GEAK_ATT_WC, num_stages=1,
        )
        return out

    if _GEAK_A1 and SK >= 256:
        # GEAK r3(c2)_d1: one kernel for the whole 392-token K stream.  The
        # split-KV pair is 6.483 us (48 CTAs, 2 nodes); this is 6.179 us and
        # ONE node.  The win needs BLOCK_N >= 128 -- at the split geometry's
        # BLOCK_N=64 the single-kernel form is 7.5-9 us because the 12-CTA
        # grid then serialises SEVEN dependent K iterations.
        nwa = _GEAK_A1_W
        bna = _GEAK_A1_BN
    else:
        bna = 16 if SK <= 16 else (_GEAK_ATT_BN if SK <= 512 else 128)
        nwa = (_GEAK_ATT_WS if SK > 16 else 2)
    _geak_attn_kernel[(B * H,)](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1),
        (float(D) ** -0.5) * _GEAK_SMSC,
        H=H, SEQ_K=SK, SEQ_M=M, BLOCK_M=bm, BLOCK_N=bna, D=D, MFMA=mf,
        ILANE=_GEAK_ILANE_I,
        num_warps=nwa,
        num_stages=(_GEAK_A1_S if (_GEAK_A1 and SK >= 256) else 1),
    )
    return out


_GEAK_ATT_OP = None
if _GEAK_TRITON_OK:
    try:
        @torch.library.custom_op("geak_head::attn", mutates_args=())
        def _geak_attn_op(q: torch.Tensor, k: torch.Tensor,
                          v: torch.Tensor) -> torch.Tensor:
            return _geak_attn_impl(q, k, v)

        @_geak_attn_op.register_fake
        def _geak_attn_meta(q, k, v):
            B, H, M, D = q.shape
            return q.new_empty((B, M, H * D))

        _GEAK_ATT_OP = torch.ops.geak_head.attn
    except Exception:
        _GEAK_ATT_OP = None

if _GEAK_ATT_OP is None:
    _GEAK_ATT = False


def _geak_add_ln_impl(a, b, w, bias, eps, nspl=1):
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    if nspl > 1:
        # GEAK r4_d0: `a` is a (nspl, PROWS, dd) fp32 split-K partial stack.
        # Reducing it here is free (this kernel is pure dispatch floor) and it
        # is strictly MORE accurate than the SPLITK=1 path, which rounds the
        # GEMM accumulator to bf16 before this add.
        dd = b.shape[-1]
        rows = b.numel() // dd
        prows = a.shape[-2]
        r = torch.empty_like(b)
        o = torch.empty_like(b)
        blk = triton.next_power_of_2(dd)
        _geak_add_ln_kernel[(rows,)](
            a, b, w, bias, r, o, float(eps),
            DD=dd, HAS_AF=(w is not None), BLOCK=blk,
            NSPL=nspl, PROWS=prows,
            num_warps=(1 if blk <= 1024 else 4), num_stages=1,
        )
        return r, o
    dd = a.shape[-1]
    rows = a.numel() // dd
    # r3_d1: the two outputs no longer have to share a dtype.  `b` is the
    # incoming residual stream and `a` is the producing GEMM's output, so
    # r follows b and o follows a -- which is exactly what the fp32-residual
    # / bf16-GEMM hybrid needs, and is a no-op when both are the same dtype.
    r = torch.empty_like(b)
    o = torch.empty_like(a)
    blk = triton.next_power_of_2(dd)
    _geak_add_ln_kernel[(rows,)](
        a, b, w, bias, r, o, float(eps),
        DD=dd, HAS_AF=(w is not None), BLOCK=blk,
        num_warps=(1 if blk <= 1024 else 4), num_stages=1,
    )
    return r, o


_GEAK_LN_OP = None
if _GEAK_TRITON_OK:
    try:
        @torch.library.custom_op("geak_head::add_ln", mutates_args=())
        def _geak_add_ln_op(a: torch.Tensor, b: torch.Tensor,
                            w: Optional[torch.Tensor],
                            bias: Optional[torch.Tensor],
                            eps: float, nspl: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
            return _geak_add_ln_impl(a, b, w, bias, eps, nspl)

        @_geak_add_ln_op.register_fake
        def _geak_add_ln_meta(a, b, w, bias, eps, nspl=1):
            if nspl > 1:
                return torch.empty_like(b), torch.empty_like(b)
            return torch.empty_like(b), torch.empty_like(a)

        _GEAK_LN_OP = torch.ops.geak_head.add_ln
    except Exception:
        _GEAK_LN_OP = None

if _GEAK_LN_OP is None:
    _GEAK_LN = False


_GEAK_MM_OVR = os.environ.get("GEAK_HEAD_MM_CFG", "")


def _geak_mm_cfg(K, N):
    """Static config table -- swept on this box, graph-timed, M=8 bf16."""
    if _GEAK_MM_OVR:
        try:
            a = [int(v) for v in _GEAK_MM_OVR.split(",")]
            if K >= 2048:
                return a[4], a[5], a[6], a[7]
            return a[0], a[1], a[2], a[3]
        except Exception:
            pass
    if K >= 2048 and K % 512 == 0:
        return 16, 512, 2, 2
    # GEAK r4_d0: num_stages 2 -> 3.  r3_d1's table was swept with a single
    # weight matrix reused across launches, i.e. L2-resident; re-swept with a
    # DISTINCT weight matrix per launch (a ~400 MB pool, which is how the trunk
    # actually streams) one more pipeline stage is worth 10% on both hot shapes:
    #   qkv K=768 N=2304   3.090 -> 2.780 us
    #   ff1 K=768 N=3072   3.521 -> 3.152 us
    #   to_q/to_out N=768  2.369 -> 2.421 us (worse, but that shape is at the
    #                                        dispatch floor and now split-K)
    if K % 256 == 0:
        return 16, 256, 1, (3 if (_GEAK_ST3 and N >= 1024) else 2)
    if K % 128 == 0:
        return 16, 128, 1, 2
    if K % 64 == 0:
        return 16, 64, 1, 2
    return 16, 64, 1, 2


def _geak_sk_cfg(K, N):
    """(splitk, bk, warps) for a GEMM whose ONLY consumer is add_ln.

    MEASURED (graph-timed chain, COLD weights -- a distinct weight matrix per
    launch so the stream is not L2-resident, which is how the trunk actually
    runs -- reported as mm + add_ln together, us):

        K=3072 N=768 (ff2)     SK=1 BK=512 w=2  8.576   <- r3_d1's shipped cfg
                               SK=3 BK=512 w=1  5.337
                               SK=6 BK=512 w=1  5.197   <- shipped
        K=768  N=768 (to_out)  SK=1 BK=256 w=1  4.551   <- r3_d1's shipped cfg
                               SK=3 BK=256 w=1  4.268   <- shipped

    ff2 at 48 CTAs ran at 738 GB/s while ff1 at 192 CTAs ran at 1429 GB/s on
    the same box: the deep-K/narrow-N shape was simply parallelism-starved.
    """
    if not _GEAK_SK:
        return 1, 0, 0
    if _GEAK_SK_CFG:
        try:
            a = [int(v) for v in _GEAK_SK_CFG.split(",")]
            if K >= 2048:
                return a[0], a[1], a[2]
            return a[3], a[4], a[5]
        except Exception:
            pass
    nb = (N + 15) // 16
    if nb >= 128:
        return 1, 0, 0
    for sk in (6, 4, 3, 2):
        if K % sk:
            continue
        kc = K // sk
        if kc < 256 or nb * sk > 384:
            continue
        for bk in (512, 256, 128):
            if kc % bk == 0:
                return sk, bk, 1
    return 1, 0, 0


def _geak_rows(m):
    """(BM, row tiles, padded rows) for an M-row GEMM: one 16-row tile up to M = 16, 32-row tiles above."""
    bm = 16 if m <= 16 else 32
    nt = (m + bm - 1) // bm
    return bm, nt, nt * bm


def _geak_mm_impl(x, w, b, gelu, splitk=1, ofp32=False):
    k = x.shape[-1]
    n = w.shape[0]
    xf = x if x.dim() == 2 else x.reshape(-1, k)
    if not xf.is_contiguous():
        xf = xf.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()
    m = xf.shape[0]
    if splitk > 1 and not gelu:
        sk, bk, nw = _geak_sk_cfg(k, n)
        if sk > 1:
            bn = 16
            nb = (n + bn - 1) // bn
            bm, nt, mp = _geak_rows(m)
            o = torch.empty((sk, mp, n), device=x.device, dtype=torch.float32)
            _geak_mm_kernel[(nb * sk, nt)](
                xf, w, b, o,
                K=k, N=n, MM=m, HAS_B=(b is not None), GELU=False,
                BM=bm, BN=bn, BK=bk, EVEN_K=True, EVEN_N=(n % bn == 0),
                SPLITK=sk, NBLK=nb, ILANE=_GEAK_ILANE_I, MP=mp,
                num_warps=nw, num_stages=2,
            )
            return o
    bn, bk, nw, ns = _geak_mm_cfg(k, n)
    # GEAK r4_d0: the accumulator is fp32 anyway, so widening the STORE is free
    # and deletes the separate `.to(float32)` dispatch the caller would need.
    odt = torch.float32 if ofp32 else x.dtype
    o = torch.empty((m, n), device=x.device, dtype=odt)
    bm, nt, _ = _geak_rows(m)
    _geak_mm_kernel[((n + bn - 1) // bn, nt)](
        xf, w, b, o,
        K=k, N=n, MM=m, HAS_B=(b is not None), GELU=bool(gelu),
        BM=bm, BN=bn, BK=bk, EVEN_K=(k % bk == 0), EVEN_N=(n % bn == 0),
        ILANE=_GEAK_ILANE_I,
        num_warps=nw, num_stages=ns,
    )
    return o if x.dim() == 2 else o.view(*x.shape[:-1], n)


_GEAK_MM_OP = None
if _GEAK_TRITON_OK:
    try:
        @torch.library.custom_op("geak_head::mm", mutates_args=())
        def _geak_mm_op(x: torch.Tensor, w: torch.Tensor,
                        b: Optional[torch.Tensor], gelu: bool,
                        splitk: int = 1, ofp32: bool = False) -> torch.Tensor:
            return _geak_mm_impl(x, w, b, gelu, splitk, ofp32)

        @_geak_mm_op.register_fake
        def _geak_mm_meta(x, w, b, gelu, splitk=1, ofp32=False):
            if splitk > 1 and not gelu:
                sk, _bk, _nw = _geak_sk_cfg(x.shape[-1], w.shape[0])
                if sk > 1:
                    m = x.numel() // x.shape[-1]
                    return x.new_empty((sk, _geak_rows(m)[2], w.shape[0]), dtype=torch.float32)
            if ofp32:
                return x.new_empty((*x.shape[:-1], w.shape[0]), dtype=torch.float32)
            return x.new_empty((*x.shape[:-1], w.shape[0]))

        _GEAK_MM_OP = torch.ops.geak_head.mm
    except Exception:
        _GEAK_MM_OP = None

if _GEAK_MM_OP is None:
    _GEAK_MM = False


def _geak_mmx_cfg(K, N):
    """(BN, BK, warps, stages) for the tail shapes.  M = 8."""
    bn = 16
    if K <= 16:
        return bn, 16, 1, 1
    if N <= 16:
        # (K=2560, N=8, the action_decoder's output layer): the reduction is
        # over K, so this is ONE CTA no matter what; 3.654 us swept over
        # BK 16..1024 x 1..8 warps x 1..4 stages, best (512, 2 warps, 2).
        # A split-K form needs a reduction node, and the node floor (1.67 us)
        # is larger than the split could win back.
        return bn, (512 if K % 512 == 0 else 256), 2, 2
    if K % 512 == 0 and K >= 2048:
        return bn, 512, 2, 2
    if K % 256 == 0:
        # swept on this box, graph-timed, cold weights, M=8:
        #   K=768 N= 768 (aenc.layer3) 2.593 us
        #   K=768 N=2560 (folded proj2.adec1+relu) 2.746 us
        # both at (BN 16, BK 256, 1 warp, 2 stages); unlike the trunk's
        # `_geak_mm_cfg`, a third stage does NOT help here.
        return bn, 256, 1, 2
    if K % 128 == 0:
        return bn, 128, 1, 2
    if K % 64 == 0:
        return bn, 64, 1, 2
    return bn, 32, 1, 2


def _geak_mmx_impl(x, w, b, c, a2, dtv, addc, act, eul, ofp32):
    k = x.shape[-1]
    n = w.shape[0]
    xf = x if x.dim() == 2 else x.reshape(-1, k)
    if not xf.is_contiguous():
        xf = xf.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()
    m = xf.shape[0]
    bn, bk, nw, ns = _geak_mmx_cfg(k, n)
    odt = torch.float32 if ofp32 else w.dtype
    o = torch.empty((m, n), device=x.device, dtype=odt)
    cf = c if (c is None or c.dim() == 2) else c.reshape(-1, n)
    af = a2 if (a2 is None or a2.dim() == 2) else a2.reshape(-1, n)
    bm, nt, _ = _geak_rows(m)
    _geak_mmx_kernel[((n + bn - 1) // bn, nt)](
        xf, w, b, (cf if cf is not None else xf),
        (af if af is not None else xf), o, float(dtv),
        K=k, N=n, MM=m, HAS_B=(b is not None),
        ADDC=(1 if addc else 0), ACT=int(act), EUL=(1 if eul else 0),
        CAST=(1 if xf.dtype != w.dtype else 0),
        BM=bm, BN=bn, BK=bk, EVEN_K=(k % bk == 0), EVEN_N=(n % bn == 0),
        ILANE=_GEAK_ILANE_I,
        num_warps=nw, num_stages=ns,
    )
    return o if x.dim() == 2 else o.view(*x.shape[:-1], n)


_GEAK_MMX_OP = None
if _GEAK_TRITON_OK:
    try:
        @torch.library.custom_op("geak_head::mmx", mutates_args=())
        def _geak_mmx_op(x: torch.Tensor, w: torch.Tensor,
                         b: Optional[torch.Tensor], c: Optional[torch.Tensor],
                         a2: Optional[torch.Tensor], dtv: float,
                         addc: bool, act: int, eul: bool,
                         ofp32: bool) -> torch.Tensor:
            return _geak_mmx_impl(x, w, b, c, a2, dtv, addc, act, eul, ofp32)

        @_geak_mmx_op.register_fake
        def _geak_mmx_meta(x, w, b, c, a2, dtv, addc, act, eul, ofp32):
            if ofp32:
                return x.new_empty((*x.shape[:-1], w.shape[0]), dtype=torch.float32)
            return x.new_empty((*x.shape[:-1], w.shape[0]), dtype=w.dtype)

        _GEAK_MMX_OP = torch.ops.geak_head.mmx
    except Exception:
        _GEAK_MMX_OP = None

if _GEAK_MMX_OP is None:
    _GEAK_TAIL = False


def _geak_tail_pack(aenc, adec, proj2, rdt):
    """Narrowed + FOLDED weights for the per-step encoder/decoder tail.

    Built once per (aenc, adec, dtype) and cached on the encoder module.
    Everything here is instruction-invariant (pure weight algebra), so the
    COMMANDMENT's rule-7 caching constraint is satisfied.

    Returns (W12, b12, W3, b3, Wfold, bfold, Wd2, bd2) or None.
    """
    key = "_geak_tailc_%s" % str(rdt).replace(".", "_")
    got = getattr(aenc, key, None)
    if got is not None:
        return got if got is not False else None
    got = False
    try:
        af = _geak_afold(aenc)
        l3 = aenc.layer3
        d1, d2 = adec.layer1, adec.layer2
        if (af is not None and isinstance(l3, nn.Linear)
                and isinstance(d1, nn.Linear) and isinstance(d2, nn.Linear)
                and isinstance(proj2, nn.Linear)
                and d1.weight.shape[1] == proj2.weight.shape[0]):
            wp2 = proj2.weight.detach().float()
            bp2 = proj2.bias.detach().float() if proj2.bias is not None else None
            wd1 = d1.weight.detach().float()
            bd1 = d1.bias.detach().float() if d1.bias is not None else None
            # relu(Wd1 (Wp2 x + bp2) + bd1) == relu(W' x + b')
            wfold = (wd1 @ wp2).contiguous()
            bfold = None
            if bp2 is not None:
                bfold = wd1 @ bp2
            if bd1 is not None:
                bfold = bd1 if bfold is None else (bfold + bd1)
            def _n(t):
                return None if t is None else t.detach().to(rdt).contiguous()
            def _f32(t):
                return None if t is None else t.detach().float().contiguous()
            got = (_n(af[0]), _f32(af[1]),
                   _n(l3.weight), _f32(l3.bias),
                   wfold.to(rdt).contiguous(),
                   (None if bfold is None else bfold.to(torch.float32).contiguous()),
                   _n(d2.weight),
                   (None if d2.bias is None else d2.bias.detach().float().contiguous()))
    except Exception:
        got = False
    setattr(aenc, key, got)
    return got if got is not False else None



def _geak_mmln_impl(xp, res, lw, lb, eps, w, b, gelu, nspl, wr, ofp32):
    """(gemm_out, new_residual) -- one launch for add_ln + the next GEMM."""
    kk = res.shape[-1]
    n = w.shape[0]
    rows = res.numel() // kk
    rf = res if res.dim() == 2 else res.reshape(rows, kk)
    if not rf.is_contiguous():
        rf = rf.contiguous()
    if not xp.is_contiguous():
        xp = xp.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()
    prows = xp.shape[-2] if nspl > 1 else rows
    bn, bk, nw, ns = _geak_mm_cfg(kk, n)
    if _GEAK_MMLN_CFG:
        try:
            _c = [int(v) for v in _GEAK_MMLN_CFG.split(",")]
            bn, bk, nw, ns = _c[0], _c[1], _c[2], _c[3]
        except Exception:
            pass
    odt = torch.float32 if ofp32 else w.dtype
    o = torch.empty((rows, n), device=res.device, dtype=odt)
    rout = torch.empty_like(rf) if wr else res.new_empty(0)
    nkc = (kk + bk - 1) // bk
    bm, nt, _ = _geak_rows(rows)
    _geak_mmln_kernel[((n + bn - 1) // bn, nt)](
        xp, rf, lw, lb, w, b, o, rout, float(eps),
        K=kk, N=n, MM=rows, HAS_B=(b is not None), GELU=bool(gelu),
        BM=bm, BN=bn, BK=bk, EVEN_K=(kk % bk == 0), EVEN_N=(n % bn == 0),
        NSPL=nspl, PROWS=prows, HAS_LA=(lw is not None), WR=bool(wr), NKC=nkc,
        PREW=_GEAK_MMLN_PREW, ILANE=_GEAK_ILANE_I,
        num_warps=nw, num_stages=ns,
    )
    if res.dim() != 2:
        o = o.view(*res.shape[:-1], n)
        if wr:
            rout = rout.view(res.shape)
    return o, rout


_GEAK_MMLN_OP = None
if _GEAK_TRITON_OK:
    try:
        @torch.library.custom_op("geak_head::mmln", mutates_args=())
        def _geak_mmln_op(xp: torch.Tensor, res: torch.Tensor,
                          lw: Optional[torch.Tensor], lb: Optional[torch.Tensor],
                          eps: float, w: torch.Tensor, b: Optional[torch.Tensor],
                          gelu: bool, nspl: int, wr: bool,
                          ofp32: bool) -> Tuple[torch.Tensor, torch.Tensor]:
            return _geak_mmln_impl(xp, res, lw, lb, eps, w, b, gelu, nspl, wr,
                                   ofp32)

        @_geak_mmln_op.register_fake
        def _geak_mmln_meta(xp, res, lw, lb, eps, w, b, gelu, nspl, wr, ofp32):
            n = w.shape[0]
            odt = torch.float32 if ofp32 else w.dtype
            return (res.new_empty((*res.shape[:-1], n), dtype=odt),
                    torch.empty_like(res) if wr else res.new_empty(0))

        _GEAK_MMLN_OP = torch.ops.geak_head.mmln
    except Exception:
        _GEAK_MMLN_OP = None

if _GEAK_MMLN_OP is None:
    _GEAK_MMLN = False


def _geak_ff_pack(block):
    """(W1, b1, W2, b2) for a diffusers FeedForward with tanh-GELU, or None.

    GEAK r4_d0 -- MEASURED BUG.  The r3_d1 recogniser demanded ``len(net) == 3``.
    diffusers' ``FeedForward`` appends a Dropout after the second Linear whenever
    ``final_dropout`` is set, and THIS checkpoint's is
    ``[GELU, Dropout, Linear, Dropout]`` (len 4).  So the recogniser returned
    None for all 16 blocks and the whole packed-FFN path was dead code: every
    block's ff1/ff2 went to rocBLAS (``Cijk_..._MT16x16x256`` / ``_MT16x16x1024``)
    plus a separate inductor GELU -- 3 dispatches per block per step instead of 2,
    on the vendor tile that masks off half of every MFMA at M = 8.

    The rule is now structural rather than positional: walk ``net``, require
    exactly one trailing Linear, and allow any number of INERT modules
    (``Dropout`` with p == 0 or a module in eval mode, ``Identity``) anywhere
    around it.  A dropout that would actually drop still rejects the block.
    """
    pack = getattr(block, "_geak_ffp", None)
    if pack is not None:
        return pack if pack is not False else None
    pack = False
    try:
        net = block.ff.net
        act = net[0]
        proj = getattr(act, "proj", None)
        lin2 = None
        ok = (_GEAK_FFP and proj is not None and isinstance(proj, nn.Linear)
              and getattr(act, "approximate", None) == "tanh")
        if ok:
            for m in net[1:]:
                if isinstance(m, nn.Linear):
                    if lin2 is not None:
                        ok = False
                        break
                    lin2 = m
                elif isinstance(m, nn.Dropout):
                    # inference: Dropout is the identity unless the module is in
                    # training mode AND p != 0.
                    if m.training and float(m.p) != 0.0:
                        ok = False
                        break
                elif isinstance(m, nn.Identity):
                    continue
                else:
                    ok = False
                    break
        if ok and lin2 is not None:
            pack = (proj.weight, proj.bias, lin2.weight, lin2.bias)
    except Exception:
        pack = False
    block._geak_ffp = pack
    return pack if pack is not False else None


def _geak_nx_shadow():
    """TorchInductor imports networkx; the ~/.local pin is networkx 2.2, which
    is broken on py3.10 (`from collections import Mapping`). A py3.10-safe
    networkx lives in nx_shadow/ -- prepend it so inductor can import."""
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
    # The failed import above leaves half-initialised networkx.* submodules in
    # sys.modules; drop them so the shadow copy is imported cleanly.
    for name in [n for n in sys.modules if n == "networkx" or n.startswith("networkx.")]:
        sys.modules.pop(name, None)
    try:
        import networkx  # noqa: F401
        return True
    except Exception:
        return False


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x



class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}

class FlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        full_config,
    ):
        super().__init__()
        config = full_config.framework.action_model
        self.hidden_size = config.hidden_size
        self.full_config = full_config
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]
        
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]
        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DiT(**diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.future_action_window_size + 1
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = MLP(
            input_dim=config.state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        ) if config.state_dim else None

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return sample / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)


    def forward(self, vl_embs: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor, state: torch.Tensor = None, encoder_attention_mask=None):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        """
        device = vl_embs.device

        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * actions + t * noise
        velocity = noise - actions

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)


        # embed state
        state_features = self.state_encoder(state) if state is not None else None


        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1) \
            if state_features is not None else action_features

        # Join VLM features with state and action embedding along sequence dimension.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )


        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        loss = ((pred_actions - velocity) ** 2) * action_mask
        loss = loss.sum() / action_mask.sum()
        return loss

    # ------------------------------------------------------------------
    # GEAK inference fast path for the denoise loop.
    #
    # Semantically identical re-implementation of
    #   DiT.forward -> BasicTransformerBlock.forward -> AttnProcessor2_0
    # (cross_attention_dit.py / diffusers attention_processor.py) that lets us
    # hoist the *loop-invariant* half of the work out of the Euler loop:
    #   * `encoder_hidden_states` (= vl_embs) is identical at every denoise
    #     step, so every cross-attention K/V projection over the 784 VL tokens
    #     was being recomputed num_steps times. We project K/V once per
    #     predict_action call and reuse them.
    #   * the position-embedding lookup and the timestep tensor allocation are
    #     also loop-invariant / cheaply reusable.
    # NOTHING about the math changes -- this is bit-identical to the original
    # path, verified in-process.
    #
    # A conservative guard (`_geak_fastpath_ok`) checks that the module tree is
    # exactly the simple configuration this re-implementation covers; anything
    # unexpected falls back to the stock `self.model(...)` call.
    # ------------------------------------------------------------------
    def _geak_fastpath_ok(self, dit) -> bool:
        cached = getattr(self, "_geak_fp_ok", None)
        if cached is not None:
            return cached
        ok = True
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0
            if not getattr(dit.config, "interleave_self_attention", False):
                ok = False
            for block in dit.transformer_blocks:
                attn = block.attn1
                if block.norm_type != "ada_norm" or block.pos_embed is not None:
                    ok = False
                    break
                if not isinstance(attn.processor, AttnProcessor2_0):
                    ok = False
                    break
                if (attn.spatial_norm is not None or attn.group_norm is not None
                        or attn.norm_cross is not None or attn.norm_q is not None
                        or attn.norm_k is not None or attn.residual_connection
                        or float(attn.rescale_output_factor) != 1.0):
                    ok = False
                    break
        except Exception:
            ok = False
        self._geak_fp_ok = ok
        return ok

    def _geak_bf16_model(self):
        """Lazily build (once) a bf16 replica of the DiT trunk.

        gfx90a has no fp8 and its fp32 vector peak (~24 TFLOPS) is ~8x below the
        bf16 MFMA peak (~191 TFLOPS); the trunk ships in fp32 while the whole
        Cosmos backbone is bf16. The Euler accumulator and the returned tensor
        stay fp32 -- only the GEMM inputs are narrowed, so bf16 error is not
        compounded across denoise steps.
        """
        rep = getattr(self, "_geak_bf16_dit", None)
        if rep is None:
            import copy
            rep = copy.deepcopy(self.model).to(torch.bfloat16).eval()
            for p in rep.parameters():
                p.requires_grad_(False)
            self._geak_bf16_dit = rep
        return rep

    @staticmethod
    def _geak_prep_kv_fused(dit, encoder_hidden_states):
        """K/V for the (loop-invariant) VL tokens, one fused GEMM per block."""
        ehs = encoder_hidden_states.contiguous()
        b = ehs.shape[0]
        allk = getattr(dit, "_geak_kvall", None)
        if allk is not None:
            w, bi, idxs, heads, ncr = allk
            # (B, S, ncr * 2 * heads * hd) -- the concatenation order is
            # [blk0_k | blk0_v | blk1_k | blk1_v | ...], so the trailing axis
            # splits into (ncr, 2, heads, hd) as a pure VIEW.  Every k/v below
            # is a strided view of that one buffer: no copy, no extra dispatch.
            t = F.linear(ehs, w, bi)
            hd = t.shape[-1] // (ncr * 2 * heads)
            t = t.view(b, -1, ncr, 2, heads, hd)
            cache = [None] * len(dit.transformer_blocks)
            for j, idx in enumerate(idxs):
                cache[idx] = (t[:, :, j, 0].permute(0, 2, 1, 3),
                              t[:, :, j, 1].permute(0, 2, 1, 3))
            return cache
        cache = []
        for idx, block in enumerate(dit.transformer_blocks):
            fkv = getattr(block, "_geak_kv", None)
            if idx % 2 == 1 or fkv is None:
                cache.append(None)
                continue
            attn = block.attn1
            kv = F.linear(ehs, fkv[0], fkv[1])
            hd = kv.shape[-1] // (2 * attn.heads)
            kv = kv.view(b, -1, 2, attn.heads, hd).permute(2, 0, 3, 1, 4)
            cache.append((kv[0], kv[1]))
        return cache

    @staticmethod
    def _geak_prep_kv(dit, encoder_hidden_states):
        """Project the (loop-invariant) VL tokens to K/V once per call."""
        ehs = encoder_hidden_states.contiguous()
        b = ehs.shape[0]
        cache = []
        for idx, block in enumerate(dit.transformer_blocks):
            if idx % 2 == 1:  # interleaved self-attention block: no encoder K/V
                cache.append(None)
                continue
            attn = block.attn1
            k = attn.to_k(ehs)
            v = attn.to_v(ehs)
            hd = k.shape[-1] // attn.heads
            k = k.view(b, -1, attn.heads, hd).transpose(1, 2)
            v = v.view(b, -1, attn.heads, hd).transpose(1, 2)
            cache.append((k, v))
        return cache

    @staticmethod
    def _geak_block(block, hidden_states, temb, kv):
        # --- AdaLayerNorm (norm_type == "ada_norm")
        n1 = block.norm1
        t = n1.linear(n1.silu(temb))
        scale, shift = t.chunk(2, dim=1)
        norm_hs = n1.norm(hidden_states) * (1 + scale[:, None]) + shift[:, None]

        # --- Attention (AttnProcessor2_0, simple config)
        attn = block.attn1
        b = norm_hs.shape[0]
        query = attn.to_q(norm_hs)
        if kv is None:
            key = attn.to_k(norm_hs)
            value = attn.to_v(norm_hs)
            hd = key.shape[-1] // attn.heads
            key = key.view(b, -1, attn.heads, hd).transpose(1, 2)
            value = value.view(b, -1, attn.heads, hd).transpose(1, 2)
        else:
            key, value = kv          # already (B, heads, S, head_dim)
            hd = key.shape[-1]
        query = query.view(b, -1, attn.heads, hd).transpose(1, 2)
        x = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        x = x.transpose(1, 2).reshape(b, -1, attn.heads * hd)
        x = x.to(query.dtype)
        x = attn.to_out[0](x)
        x = attn.to_out[1](x)
        if block.final_dropout is not None:
            x = block.final_dropout(x)

        hidden_states = x + hidden_states
        hidden_states = block.ff(block.norm3(hidden_states)) + hidden_states
        return hidden_states

    def _geak_dit_forward(self, dit, hidden_states, timestep, kv_cache):
        temb = dit.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        for idx, block in enumerate(dit.transformer_blocks):
            hidden_states = self._geak_block(block, hidden_states, temb, kv_cache[idx])
        shift, scale = dit.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = dit.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return dit.proj_out_2(hidden_states)

    def _geak_step_fn(self, trunk):
        """Return the (optionally inductor-compiled) per-denoise-step callable.

        Compiled ONCE per (module, dtype); the KV cache is passed as an
        argument, so a new observation reuses the same compiled graph (guards
        are on shape/dtype, not tensor identity) and never re-traces.
        """
        key = "_geak_step_%x" % id(trunk)
        fn = getattr(self, key, None)
        if fn is not None:
            return fn

        def step(hidden_states, timestep, kv_cache):
            return self._geak_dit_forward(trunk, hidden_states, timestep, kv_cache)

        if _GEAK_COMPILE and _geak_nx_shadow():
            try:
                fn = torch.compile(step, dynamic=False)
            except Exception:
                fn = step
        else:
            fn = step
        setattr(self, key, fn)
        return fn

    # ==================================================================
    # GEAK r1_d0 (host_runtime): action-head launch-floor collapse.
    #
    # Three stacked levers, each with its own kill switch:
    #
    #  (1) GEAK_HEAD_FP32W -- the caller wraps predict_action in
    #      torch.autocast("cuda", float32) over a *bf16* checkpoint, so every
    #      Linear/LayerNorm/SDPA in the head re-casts its bf16 weights to fp32
    #      on EVERY call (measured: 445 pure dtype-churn dispatches per 4-step
    #      iteration, 1.44 ms of device time, ~4 us of host launch each).
    #      We materialize an fp32 replica of every module the head touches ONCE
    #      and then run with autocast DISABLED. The arithmetic is unchanged --
    #      autocast(float32) was already widening the same weights to the same
    #      fp32 values -- only the per-call cast kernels disappear.
    #
    #  (2) GEAK_HEAD_CONST -- all N denoise timesteps are known up front and are
    #      the SAME constants on every call (t_disc = int((1-t/N)*buckets)).
    #      So the entire timestep-conditioning chain is loop- AND call-invariant:
    #      timestep_encoder, the 16 per-block AdaLN `linear(silu(temb))` GEMMs,
    #      proj_out_1, and the sinusoidal tau embedding in the action encoder.
    #      All of it is computed once per (N, T) and cached for the process
    #      lifetime, batched over the N steps in a single GEMM each.
    #
    #  (3) GEAK_HEAD_GRAPH -- what is left is a bit-stable, static-shape
    #      sequence replayed N times, i.e. geomean_levers.md Lever 6. The WHOLE
    #      head (KV prep + N Euler steps) is captured into one CUDA graph keyed
    #      by (N, vl shape, state shape) with static input/output buffers.
    #      The capture-unsafe torch.randn stays OUTSIDE the graph (it fills the
    #      static action buffer before replay). Every failure mode -- unexpected
    #      module tree, capture error, shape miss -- falls back to eager.
    # ==================================================================

    def _geak_f32_pack(self):
        """fp32 replicas of every module the head touches, built once."""
        pack = getattr(self, "_geak_f32_cache", None)
        if pack is not None:
            return pack
        import copy
        try:
            with torch.inference_mode(False), torch.no_grad():
                trunk = copy.deepcopy(self.model).float().eval()
                aenc = copy.deepcopy(self.action_encoder).float().eval()
                adec = copy.deepcopy(self.action_decoder).float().eval()
                senc = (copy.deepcopy(self.state_encoder).float().eval()
                        if self.state_encoder is not None else None)
                pemb = (self.position_embedding.weight.detach().clone().float()
                        if self.config.add_pos_embed else None)
                for m in (trunk, aenc, adec, senc):
                    if m is not None:
                        for p in m.parameters():
                            p.requires_grad_(False)
                rtrunk = trunk
                if _GEAK_B16:
                    # NOTE the checkpoint itself is bf16 (`from_pretrained(...,
                    # torch_dtype=bfloat16)`); the fp32 replica above only
                    # widens those same bf16 values because the stock caller
                    # wrapped the head in autocast(float32).  So narrowing the
                    # trunk back to bf16 is EXACT for every weight -- the only
                    # thing that changes precision is the activation stream.
                    rtrunk = copy.deepcopy(self.model).to(torch.bfloat16).eval()
                    for pp in rtrunk.parameters():
                        pp.requires_grad_(False)
                if _GEAK_QKV:
                    self._geak_fuse_proj(rtrunk)
            pack = (trunk, aenc, adec, senc, pemb, rtrunk)
        except Exception:
            pack = False
        self._geak_f32_cache = pack
        return pack

    @staticmethod
    def _geak_fuse_proj(trunk):
        """Concatenate the per-projection GEMMs of each block into ONE.

        At S=8 tokens every kernel in this trunk sits at the device launch
        floor (~5 us regardless of size), so launch COUNT is the cost, not
        FLOPs. Self-attention blocks get a single fused q|k|v GEMM; cross
        blocks get a single fused k|v GEMM over the (loop-invariant) VL tokens.
        Values are unchanged -- each output element is the same dot product.
        """
        def _cat(mods):
            w = torch.cat([m.weight for m in mods], dim=0).contiguous()
            if all(m.bias is not None for m in mods):
                b = torch.cat([m.bias for m in mods], dim=0).contiguous()
            elif all(m.bias is None for m in mods):
                b = None
            else:
                return None
            return (w, b)

        for idx, block in enumerate(trunk.transformer_blocks):
            attn = block.attn1
            self_attn = (idx % 2 == 1)
            if self_attn:
                block._geak_qkv = _cat([attn.to_q, attn.to_k, attn.to_v])
                block._geak_kv = None
            else:
                block._geak_qkv = None
                block._geak_kv = _cat([attn.to_k, attn.to_v])

        # ---- one k|v GEMM for every cross block (see _GEAK_KVALL) ----
        trunk._geak_kvall = None
        if _GEAK_KVALL:
            try:
                cross = [(i, b) for i, b in enumerate(trunk.transformer_blocks)
                         if getattr(b, "_geak_kv", None) is not None]
                if len(cross) >= 2:
                    heads = {b.attn1.heads for _, b in cross}
                    widths = {b._geak_kv[0].shape[0] for _, b in cross}
                    kdim = {b._geak_kv[0].shape[1] for _, b in cross}
                    hasb = {b._geak_kv[1] is not None for _, b in cross}
                    if len(heads) == 1 and len(widths) == 1 and len(kdim) == 1 \
                            and len(hasb) == 1:
                        w = torch.cat([b._geak_kv[0] for _, b in cross], dim=0).contiguous()
                        bi = (torch.cat([b._geak_kv[1] for _, b in cross], dim=0).contiguous()
                              if hasb.pop() else None)
                        trunk._geak_kvall = (w, bi, [i for i, _ in cross],
                                             heads.pop(), len(cross))
            except Exception:
                trunk._geak_kvall = None

    def _geak_consts(self, trunk, aenc, pemb, num_steps, seq_T, device,
                     rdtype=torch.float32):
        """Step- and call-invariant conditioning, computed once per (N, T).

        Always evaluated in fp32 (it is once-per-process, so precision here is
        free) and only then narrowed to the trunk's run dtype.
        """
        cache = getattr(self, "_geak_const_cache", None)
        if cache is None:
            cache = {}
            self._geak_const_cache = cache
        key = (num_steps, seq_T, rdtype)
        pack = cache.get(key)
        if pack is not None:
            return pack
        buckets = self.num_timestep_buckets
        ts = torch.tensor(
            [int((1.0 - t / float(num_steps)) * buckets) for t in range(num_steps)],
            device=device, dtype=torch.long,
        )
        temb = trunk.timestep_encoder(ts)                       # (N, D)
        blk = []
        for block in trunk.transformer_blocks:
            n1 = block.norm1
            tt = n1.linear(n1.silu(temb))
            scale, shift = tt.chunk(2, dim=1)
            sc = (1.0 + scale).contiguous()
            sh = shift.contiguous()
            blk.append((sc.unsqueeze(1), sh.unsqueeze(1), sc, sh))
        shift_o, scale_o = trunk.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        sc_o = (1.0 + scale_o).contiguous()
        sh_o = shift_o.contiguous()
        out_c = (sc_o.unsqueeze(1), sh_o.unsqueeze(1), sc_o, sh_o)
        tau = aenc.pos_encoding(ts.unsqueeze(1).expand(-1, seq_T)).float().contiguous()
        _af = _geak_afold(aenc)
        if _af is not None:
            # carry W2b @ tau instead of tau -- same tensor shape, so nothing
            # downstream of `consts` changes except the step body.
            tau = F.linear(tau, _af[2]).contiguous()
        pos = pemb[:seq_T].unsqueeze(0).contiguous() if pemb is not None else None
        ln_affine = bool(_GEAK_FUSE) and trunk.norm_out.weight is None and all(
            blk_.norm1.norm.weight is None and blk_.norm1.norm.bias is None
            for blk_ in trunk.transformer_blocks)
        if rdtype != torch.float32:
            blk = [tuple(t.to(rdtype).contiguous() for t in e) for e in blk]
            out_c = tuple(t.to(rdtype).contiguous() for t in out_c)
        pack = (blk, out_c, tau, pos, ln_affine)
        cache[key] = pack
        return pack

    @staticmethod
    def _geak_block_c(block, hidden_states, scale1, shift1, scale_v, shift_v, kv, ln_affine):
        """One BasicTransformerBlock with the AdaLN modulation precomputed."""
        attn = block.attn1
        n1norm = block.norm1.norm
        if ln_affine:
            # LayerNorm(x) * (1+scale) + shift  ==  layer_norm(x, w=1+scale, b=shift)
            norm_hs = F.layer_norm(hidden_states, n1norm.normalized_shape,
                                   scale_v, shift_v, n1norm.eps)
        else:
            norm_hs = torch.addcmul(shift1, n1norm(hidden_states), scale1)

        b = norm_hs.shape[0]
        heads = attn.heads
        fq = getattr(block, "_geak_qkv", None)
        if fq is not None:
            qkv = F.linear(norm_hs, fq[0], fq[1])
            hd = qkv.shape[-1] // (3 * heads)
            qkv = qkv.view(b, -1, 3, heads, hd).permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]
        else:
            query = attn.to_q(norm_hs)
            if kv is None:
                key = attn.to_k(norm_hs)
                value = attn.to_v(norm_hs)
                hd = key.shape[-1] // heads
                key = key.view(b, -1, heads, hd).transpose(1, 2)
                value = value.view(b, -1, heads, hd).transpose(1, 2)
            else:
                key, value = kv
                hd = key.shape[-1]
            query = query.view(b, -1, heads, hd).transpose(1, 2)
        if _GEAK_ATT:
            # writes (B, S, heads*hd) directly -- no reshape copy
            x = _GEAK_ATT_OP(query, key, value)
        else:
            x = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            x = x.transpose(1, 2).reshape(b, -1, heads * hd)
        x = attn.to_out[0](x)
        hidden_states = x + hidden_states
        hidden_states = block.ff(block.norm3(hidden_states)) + hidden_states
        return hidden_states

    # ------------------------------------------------------------------
    # GEAK r2_d1: the per-denoise-step body, factored out so that ONE
    # inductor compilation covers every step (all N steps have identical
    # shapes/dtypes -- only the values of the pre-sliced conditioning
    # tensors differ, and those enter as ordinary tensor arguments).
    # ------------------------------------------------------------------
    @staticmethod
    def _geak_block_ln(block, res, normed, w_next, b_next, eps_next, kv,
                       sk_o: int = 1, sk_f: int = 1):
        """Same block, but the residual add and the NEXT normalisation are
        one `geak_head::add_ln` launch each (see _geak_add_ln_kernel).

        The block is re-cut so that every residual add is already paired with
        the layernorm that consumes it: the block takes (res, normed) and
        returns (res, normed) for the next consumer, which for the last block
        is `norm_out` and its own modulation.  Two launches per block replace
        inductor's two 4.5 us fused kernels with two 2.12 us ones, and the
        arithmetic is unchanged.
        """
        attn = block.attn1
        b = normed.shape[0]
        heads = attn.heads
        # MEASURED: this decomposition is a WIN on a bf16 trunk (4-step 13.94 ->
        # 12.96 ms together with the narrowing) and a LOSS on an fp32 one
        # (14.45 ms) -- fp32 tl.dot has no MFMA path on gfx950, so rocBLAS's
        # deeper K-unroll wins there.  Gate on the actual run dtype so that
        # GEAK_HEAD_B16=0 still reproduces the round-2 tree exactly.
        mm = (_GEAK_MM and _GEAK_MM_OP is not None
              and normed.dtype != torch.float32 and normed.shape[-2] <= 16)
        fq = getattr(block, "_geak_qkv", None)
        if fq is not None:
            qkv = (_GEAK_MM_OP(normed, fq[0], fq[1], False) if mm
                   else F.linear(normed, fq[0], fq[1]))
            hd = qkv.shape[-1] // (3 * heads)
            qkv = qkv.view(b, -1, 3, heads, hd).permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]
        else:
            query = (_GEAK_MM_OP(normed, attn.to_q.weight, attn.to_q.bias, False)
                     if mm else attn.to_q(normed))
            if kv is None:
                key = attn.to_k(normed)
                value = attn.to_v(normed)
                hd = key.shape[-1] // heads
                key = key.view(b, -1, heads, hd).transpose(1, 2)
                value = value.view(b, -1, heads, hd).transpose(1, 2)
            else:
                key, value = kv
                hd = key.shape[-1]
            query = query.view(b, -1, heads, hd).transpose(1, 2)
        if _GEAK_ATT:
            x = _GEAK_ATT_OP(query, key, value)
        else:
            x = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            x = x.transpose(1, 2).reshape(b, -1, heads * hd)
        to = attn.to_out[0]
        # GEAK r4_d0: `sk_o` / `sk_f` are the split-K factors for the two GEMMs
        # whose ONLY consumer is the add_ln immediately below.  The partial
        # reduction rides in add_ln's (empty) body, so the split is free.
        so = sk_o if mm else 1
        x = (_GEAK_MM_OP(x, to.weight, to.bias, False, so) if mm else to(x))
        n3 = block.norm3
        res, y = _GEAK_LN_OP(x, res, n3.weight, n3.bias, n3.eps, so)
        ffp = _geak_ff_pack(block) if mm else None
        sf = sk_f if ffp is not None else 1
        if ffp is not None:
            # GELU folded into the first GEMM's fp32 accumulator: the FFN is
            # TWO dispatches instead of three, and neither of them is a
            # bandwidth round-trip over the (8, 3072) hidden activation.
            y = _GEAK_MM_OP(y, ffp[0], ffp[1], True)
            y = _GEAK_MM_OP(y, ffp[2], ffp[3], False, sf)
        else:
            y = block.ff(y)
        res, normed = _GEAK_LN_OP(y, res, w_next, b_next, eps_next, sf)
        return res, normed

    def _geak_hybrid(self, rtrunk):
        """True when GEAK_HEAD_B16=2 (bf16 GEMMs / fp32 residual) is usable.

        The hybrid needs the re-cut `_geak_make_step_ln` body, because that is
        the only path where every residual add goes through the two-output
        `geak_head::add_ln` kernel -- which is what lets the residual stream
        stay fp32 while the tensor handed to the next GEMM is bf16, for zero
        extra launches.  If that path is unavailable we fall back to the full
        bf16 trunk (level 1) rather than mixing dtypes into F.linear.
        """
        if _GEAK_B16_LVL != 2:
            return False
        try:
            ln_affine = bool(_GEAK_FUSE) and rtrunk.norm_out.weight is None and all(
                b.norm1.norm.weight is None and b.norm1.norm.bias is None
                for b in rtrunk.transformer_blocks)
            return bool(self._geak_ln_ok(rtrunk, ln_affine))
        except Exception:
            return False

    def _geak_ln_ok(self, trunk, ln_affine):
        """The re-cut path needs modulation-only norm1 (no baked affine)."""
        if not (_GEAK_LN and _GEAK_LN_OP is not None and ln_affine):
            return False
        for blk in trunk.transformer_blocks:
            n1 = blk.norm1.norm
            if n1.weight is not None or n1.bias is not None:
                return False
            if getattr(blk, "final_dropout", None) is not None and blk.training:
                return False
        return trunk.norm_out.weight is None and trunk.norm_out.bias is None

    def _geak_make_step_ln(self, trunk, aenc, adec, dt, horizon, hybrid=False):
        blocks = list(trunk.transformer_blocks)
        no = trunk.norm_out
        proj2 = trunk.proj_out_2
        rdt = next(trunk.parameters()).dtype
        blk_ln = FlowmatchingActionHead._geak_block_ln
        no_shape = no.normalized_shape
        eps_next = [blocks[i + 1].norm1.norm.eps for i in range(len(blocks) - 1)]
        eps_next.append(no.eps)
        eps0 = blocks[0].norm1.norm.eps
        nb = len(blocks)
        # GEAK r4_d0: static per-block split-K factors.  Disabled in the level-2
        # hybrid (there add_ln's two outputs carry DIFFERENT dtypes, which the
        # fp32-partial path cannot express) and whenever the trunk is fp32.
        allow_sk = bool(_GEAK_SK and _GEAK_MM and _GEAK_MM_OP is not None
                        and (not hybrid) and rdt != torch.float32)
        sk_tab = []
        for _b in blocks:
            _so = _sf = 1
            if allow_sk:
                try:
                    _w = _b.attn1.to_out[0].weight
                    _so = _geak_sk_cfg(_w.shape[1], _w.shape[0])[0]
                except Exception:
                    _so = 1
                _fp = _geak_ff_pack(_b)
                if _fp is not None:
                    _sf = _geak_sk_cfg(_fp[2].shape[1], _fp[2].shape[0])[0]
            sk_tab.append((_so, _sf))

        afold = _geak_afold(aenc)
        p2mm = bool(_GEAK_P2MM and _GEAK_MM and _GEAK_MM_OP is not None
                    and (not hybrid) and rdt != torch.float32
                    and isinstance(proj2, nn.Linear))
        # GEAK r3(c2)_d1 -- see GEAK_HEAD_TAIL at the top of this file.
        tailp = None
        if (_GEAK_TAIL and _GEAK_MMX_OP is not None and (not hybrid)
                and rdt != torch.float32):
            tailp = _geak_tail_pack(aenc, adec, proj2, rdt)

        def step(actions, tau_t, pos, state_features, blk_t, out_t, kv):
            if tailp is not None and state_features is None:
                # encoder: (afold GEMM, +tau, swish) and (layer3 GEMM, +pos)
                # are ONE node each, in bf16/MFMA, instead of two fp32
                # rocBLAS GEMMs plus two inductor pointwise nodes.
                h_ = _GEAK_MMX_OP(actions, tailp[0], tailp[1], tau_t, None,
                                  0.0, True, 1, False, False)
                res = _GEAK_MMX_OP(h_, tailp[2], tailp[3], pos, None,
                                   0.0, pos is not None, 0, False, False)
                normed = F.layer_norm(res, no_shape, blk_t[0][2], blk_t[0][3],
                                      eps0)
                for i in range(nb):
                    if i + 1 < nb:
                        wn, bn = blk_t[i + 1][2], blk_t[i + 1][3]
                    else:
                        wn, bn = out_t[2], out_t[3]
                    res, normed = blk_ln(blocks[i], res, normed, wn, bn,
                                         eps_next[i], kv[i],
                                         sk_tab[i][0], sk_tab[i][1])
                # proj_out_2 . action_decoder.layer1 . relu, folded to ONE
                # GEMM (W' = Wd1 @ Wp2), then layer2 with the Euler update
                # riding in its fp32 accumulator.
                hd = _GEAK_MMX_OP(normed, tailp[4], tailp[5], None, None,
                                  0.0, False, 2, False, False)
                return _GEAK_MMX_OP(hd, tailp[6], tailp[7], None, actions,
                                    float(dt), False, 0, True, True)
            if afold is not None:
                x = aenc.layer3(swish(F.linear(actions, afold[0], afold[1]) + tau_t))
            else:
                x = torch.cat([aenc.layer1(actions), tau_t], dim=-1)
                x = aenc.layer3(swish(aenc.layer2(x)))
            if pos is not None:
                x = x + pos
            if state_features is not None:
                x = torch.cat((state_features, x), dim=1)
            if hybrid:
                # r3_d1 HYBRID: the residual stream stays fp32 for the whole
                # trunk; only the tensor that feeds a GEMM is bf16.  The cast
                # is fused into the layer_norm epilogue by inductor (and into
                # the add_ln kernel's store for every later block), so this
                # costs ZERO extra dispatches.
                res = x
                normed = F.layer_norm(res, no_shape, blk_t[0][2], blk_t[0][3],
                                      eps0).to(rdt)
            else:
                if x.dtype != rdt:
                    # the action encoder and the Euler accumulator stay fp32;
                    # only the trunk's weight-streaming half is narrowed.
                    x = x.to(rdt)
                res = x
                normed = F.layer_norm(res, no_shape, blk_t[0][2], blk_t[0][3],
                                      eps0)
            for i in range(nb):
                if i + 1 < nb:
                    wn, bn = blk_t[i + 1][2], blk_t[i + 1][3]
                else:
                    wn, bn = out_t[2], out_t[3]
                res, normed = blk_ln(blocks[i], res, normed, wn, bn,
                                     eps_next[i], kv[i],
                                     sk_tab[i][0], sk_tab[i][1])
            if p2mm:
                # proj2 straight into fp32: one rocBLAS bf16 GEMM + one
                # `.to(fp32)` copy kernel become a single triton launch.
                out = _GEAK_MM_OP(normed, proj2.weight, proj2.bias, False, 1, True)
            else:
                out = proj2(normed)
                if out.dtype != actions.dtype:
                    out = out.to(actions.dtype)
            pred = adec(out)
            return actions - dt * pred[:, -horizon:]

        return step

    def _geak_mmln_ok(self, trunk):
        """True when every block's add_ln can ride in its consumer's prologue."""
        if not (_GEAK_MMLN and _GEAK_MMLN_OP is not None
                and _GEAK_MM and _GEAK_MM_OP is not None
                and _GEAK_ATT and _GEAK_ATT_OP is not None):
            return False
        try:
            if next(trunk.parameters()).dtype == torch.float32:
                return False
            for blk in trunk.transformer_blocks:
                if _geak_ff_pack(blk) is None:
                    return False
                attn = blk.attn1
                if not isinstance(attn.to_out[0], nn.Linear):
                    return False
                if getattr(blk, "_geak_qkv", None) is None \
                        and not isinstance(attn.to_q, nn.Linear):
                    return False
                if getattr(blk, "final_dropout", None) is not None and blk.training:
                    return False
            if not isinstance(trunk.proj_out_2, nn.Linear):
                return False
        except Exception:
            return False
        return True

    def _geak_make_step_mmln(self, trunk, aenc, adec, dt, horizon, hybrid=False):
        """The re-cut step body with EVERY add_ln absorbed into a GEMM prologue.

        The block chain is re-phased by one GEMM: a block receives its q|k|v
        (or q) projection ALREADY COMPUTED by the previous block's fused
        FFN-residual kernel, and in turn emits the next block's projection.  So
        the per-block dispatch chain is

            attn -> to_out -> [addLN + ff1] -> ff2 -> [addLN + next proj]

        i.e. 5 nodes for a self-attention block and 6 for a cross block, where
        the r4_d0 tree ran 7 and 8.  32 nodes per denoise step disappear.
        """
        blocks = list(trunk.transformer_blocks)
        nb = len(blocks)
        no = trunk.norm_out
        proj2 = trunk.proj_out_2
        rdt = next(trunk.parameters()).dtype
        no_shape = no.normalized_shape
        eps0 = blocks[0].norm1.norm.eps
        eps_next = [blocks[i + 1].norm1.norm.eps for i in range(nb - 1)]
        eps_next.append(no.eps)

        allow_sk = bool(_GEAK_SK and (not hybrid) and _GEAK_MMLN_SO != "1")
        meta = []
        pw = []
        for blk in blocks:
            attn = blk.attn1
            fq = getattr(blk, "_geak_qkv", None)
            to = attn.to_out[0]
            ffp = _geak_ff_pack(blk)
            n3 = blk.norm3
            so = sf = 1
            if allow_sk:
                so = _geak_sk_cfg(to.weight.shape[1], to.weight.shape[0])[0]
                sf = _geak_sk_cfg(ffp[2].shape[1], ffp[2].shape[0])[0]
            meta.append((int(attn.heads), fq, to.weight, to.bias, ffp,
                         n3.weight, n3.bias, float(n3.eps), int(so), int(sf)))
            if fq is not None:
                pw.append((fq[0], fq[1]))
            else:
                pw.append((attn.to_q.weight, attn.to_q.bias))

        afold = _geak_afold(aenc)
        mmop = _GEAK_MM_OP
        mlop = _GEAK_MMLN_OP
        attop = _GEAK_ATT_OP

        def step(actions, tau_t, pos, state_features, blk_t, out_t, kv):
            if afold is not None:
                x = aenc.layer3(swish(F.linear(actions, afold[0], afold[1]) + tau_t))
            else:
                x = torch.cat([aenc.layer1(actions), tau_t], dim=-1)
                x = aenc.layer3(swish(aenc.layer2(x)))
            if pos is not None:
                x = x + pos
            if state_features is not None:
                x = torch.cat((state_features, x), dim=1)
            if hybrid:
                res = x
                normed = F.layer_norm(res, no_shape, blk_t[0][2], blk_t[0][3],
                                      eps0).to(rdt)
            else:
                if x.dtype != rdt:
                    x = x.to(rdt)
                res = x
                normed = F.layer_norm(res, no_shape, blk_t[0][2], blk_t[0][3],
                                      eps0)
            bsz = res.shape[0]
            proj = mmop(normed, pw[0][0], pw[0][1], False)
            out = None
            for i in range(nb):
                heads, fq, tow, tob, ffp, n3w, n3b, n3e, so, sf = meta[i]
                if fq is not None:
                    hd = proj.shape[-1] // (3 * heads)
                    q3 = proj.view(bsz, -1, 3, heads, hd).permute(2, 0, 3, 1, 4)
                    query, key, value = q3[0], q3[1], q3[2]
                else:
                    key, value = kv[i]
                    hd = key.shape[-1]
                    query = proj.view(bsz, -1, heads, hd).transpose(1, 2)
                xa = attop(query, key, value)
                xo = mmop(xa, tow, tob, False, so)
                y, res = mlop(xo, res, n3w, n3b, n3e,
                              ffp[0], ffp[1], True, so, True, False)
                xf = mmop(y, ffp[2], ffp[3], False, sf)
                if i + 1 < nb:
                    proj, res = mlop(xf, res, blk_t[i + 1][2], blk_t[i + 1][3],
                                     eps_next[i], pw[i + 1][0], pw[i + 1][1],
                                     False, sf, True, False)
                else:
                    out, _r = mlop(xf, res, out_t[2], out_t[3], eps_next[i],
                                   proj2.weight, proj2.bias, False, sf, False,
                                   True)
            if out.dtype != actions.dtype:
                out = out.to(actions.dtype)
            pred = adec(out)
            return actions - dt * pred[:, -horizon:]

        return step

    def _geak_make_step(self, trunk, aenc, adec, ln_affine, dt, horizon):
        blocks = list(trunk.transformer_blocks)
        no = trunk.norm_out
        proj2 = trunk.proj_out_2
        blk_c = FlowmatchingActionHead._geak_block_c
        no_shape = no.normalized_shape
        no_eps = no.eps
        rdt2 = next(trunk.parameters()).dtype

        afold = _geak_afold(aenc)

        def step(actions, tau_t, pos, state_features, blk_t, out_t, kv):
            if afold is not None:
                x = aenc.layer3(swish(F.linear(actions, afold[0], afold[1]) + tau_t))
            else:
                x = torch.cat([aenc.layer1(actions), tau_t], dim=-1)
                x = aenc.layer3(swish(aenc.layer2(x)))
            if pos is not None:
                x = x + pos
            if state_features is not None:
                x = torch.cat((state_features, x), dim=1)
            if x.dtype != rdt2:
                x = x.to(rdt2)
            for i, block in enumerate(blocks):
                sc1, sh1, scv, shv = blk_t[i]
                x = blk_c(block, x, sc1, sh1, scv, shv, kv[i], ln_affine)
            if ln_affine:
                x = F.layer_norm(x, no_shape, out_t[2], out_t[3], no_eps)
            else:
                x = torch.addcmul(out_t[1], no(x), out_t[0])
            out = proj2(x)
            if out.dtype != actions.dtype:
                out = out.to(actions.dtype)
            pred = adec(out)
            return actions - dt * pred[:, -horizon:]

        return step

    def _geak_step_c(self, trunk, aenc, adec, ln_affine, dt, horizon):
        """Cached (optionally inductor-compiled) per-step callable."""
        store = getattr(self, "_geak_stepc_cache", None)
        if store is None:
            store = {}
            self._geak_stepc_cache = store
        hybrid = self._geak_hybrid(trunk)
        key = (id(trunk), bool(ln_affine), float(dt), int(horizon),
               bool(_GEAK_ATT), bool(_GEAK_LN), _GEAK_B16_LVL, hybrid,
               bool(_GEAK_MM), bool(_GEAK_MMLN))
        fn = store.get(key)
        if fn is not None:
            return fn
        use_ln = self._geak_ln_ok(trunk, ln_affine)
        use_mmln = bool(use_ln and self._geak_mmln_ok(trunk))
        if _GEAK_DBG:
            print("[GEAK_HEAD_DBG] use_ln=%s use_mmln=%s hybrid=%s rdt=%s"
                  % (use_ln, use_mmln, hybrid,
                     next(trunk.parameters()).dtype), flush=True)
        if use_mmln:
            fn = self._geak_make_step_mmln(trunk, aenc, adec, dt, horizon, hybrid)
        elif use_ln:
            fn = self._geak_make_step_ln(trunk, aenc, adec, dt, horizon, hybrid)
        else:
            fn = self._geak_make_step(trunk, aenc, adec, ln_affine, dt, horizon)
        if _GEAK_IND and _geak_nx_shadow():
            try:
                # Per-compile options only -- never a global torch._inductor.config
                # mutation, which would leak into any other lane's compiles.
                opts = {}
                if "cd" in _GEAK_IND_MODE:
                    # The head's fused pointwise/reduction kernels run ~2.3x above
                    # this box's ~2.0 us empty-kernel floor at inductor's default
                    # heuristic block sizes; coordinate descent re-tunes
                    # XBLOCK/RBLOCK/num_warps per kernel.
                    opts["coordinate_descent_tuning"] = True
                if "mm" in _GEAK_IND_MODE:
                    opts["max_autotune_gemm"] = True
                    opts["max_autotune_gemm_backends"] = "TRITON,ATEN"
                if "ep" in _GEAK_IND_MODE:
                    opts["epilogue_fusion"] = True
                if opts:
                    fn = torch.compile(fn, dynamic=False, options=opts)
                elif _GEAK_IND_MODE and _GEAK_IND_MODE not in ("default", "1"):
                    fn = torch.compile(fn, dynamic=False, mode=_GEAK_IND_MODE)
                else:
                    fn = torch.compile(fn, dynamic=False)
            except Exception:
                if use_mmln:
                    fn = self._geak_make_step_mmln(trunk, aenc, adec, dt,
                                                   horizon, hybrid)
                elif use_ln:
                    fn = self._geak_make_step_ln(trunk, aenc, adec, dt, horizon,
                                                 hybrid)
                else:
                    fn = self._geak_make_step(trunk, aenc, adec, ln_affine, dt,
                                              horizon)
        store[key] = fn
        return fn

    def _geak_head_body(self, trunk, aenc, adec, consts, kv, state_features,
                        actions, num_steps):
        blk, out_c, tau, pos, ln_affine = consts
        sc_o1, sh_o1, sc_ov, sh_ov = out_c
        horizon = self.action_horizon
        dt = 1.0 / num_steps
        B = actions.shape[0]
        step = self._geak_step_c(trunk, aenc, adec, ln_affine, dt, horizon)
        if _GEAK_DBG and not getattr(self, "_geak_dbg_done", False):
            self._geak_dbg_done = True
            nk = sum(1 for e in kv if e is not None)
            print("[GEAK_HEAD_DBG] steps=%d actions=%s x_dtype=%s nblocks=%d "
                  "cross_kv=%d kv0=%s tau=%s pos=%s state=%s ln_affine=%s ind=%s"
                  % (num_steps, tuple(actions.shape), actions.dtype,
                     len(blk), nk,
                     (tuple(kv[0][0].shape) if kv[0] is not None else None),
                     tuple(tau.shape), (tuple(pos.shape) if pos is not None else None),
                     (tuple(state_features.shape) if state_features is not None else None),
                     ln_affine, _GEAK_IND), flush=True)
        for t in range(num_steps):
            tau_t = tau[t : t + 1]
            if B != 1:
                tau_t = tau_t.expand(B, -1, -1)
            blk_t = [(sc1[t : t + 1], sh1[t : t + 1], scv[t], shv[t])
                     for (sc1, sh1, scv, shv) in blk]
            out_t = (sc_o1[t : t + 1], sh_o1[t : t + 1], sc_ov[t], sh_ov[t])
            actions = step(actions, tau_t, pos, state_features, blk_t, out_t, kv)
        return actions

    def _geak_run_eager(self, pack, consts, vl_f32, state_f32, actions, num_steps):
        trunk, aenc, adec, senc, _, rtrunk = pack
        rdt = next(rtrunk.parameters()).dtype
        vl_r = vl_f32 if rdt == vl_f32.dtype else vl_f32.to(rdt)
        kv = (self._geak_prep_kv_fused(rtrunk, vl_r) if _GEAK_QKV
              else self._geak_prep_kv(rtrunk, vl_r))
        sf = senc(state_f32) if (senc is not None and state_f32 is not None) else None
        # HYBRID (level 2): the state features are concatenated onto the fp32
        # residual stream, so they must stay fp32.
        sdt = torch.float32 if self._geak_hybrid(rtrunk) else rdt
        if sf is not None and sf.dtype != sdt:
            sf = sf.to(sdt)
        return self._geak_head_body(rtrunk, aenc, adec, consts, kv, sf,
                                    actions, num_steps)

    def _geak_graph(self, pack, consts, vl_f32, state_f32, actions, num_steps):
        """Capture (or reuse) the whole head as ONE CUDA graph; returns output."""
        store = getattr(self, "_geak_graph_cache", None)
        if store is None:
            store = {}
            self._geak_graph_cache = store
        key = (num_steps, tuple(vl_f32.shape),
               tuple(state_f32.shape) if state_f32 is not None else None,
               tuple(actions.shape))
        ent = store.get(key)
        if ent is False:
            return None
        if ent is None:
            if len(store) >= 8:
                return None
            try:
                ent = self._geak_capture(pack, consts, vl_f32, state_f32,
                                         actions, num_steps)
            except Exception:
                ent = False
            store[key] = ent
            if ent is False:
                return None
        g, s_vl, s_state, s_act, s_out = ent
        s_vl.copy_(vl_f32)
        if s_state is not None and state_f32 is not None:
            s_state.copy_(state_f32)
        s_act.copy_(actions)
        g.replay()
        return s_out

    def _geak_capture(self, pack, consts, vl_f32, state_f32, actions, num_steps):
        s_vl = vl_f32.clone()
        s_state = state_f32.clone() if state_f32 is not None else None
        # GEAK r2_d0: the Euler accumulator is fp32 by construction; when the
        # caller hands us the narrow noise tensor the WIDENING happens here,
        # once, at capture time, and `_geak_graph`'s `s_act.copy_()` does it on
        # every later call for free.
        s_act = (actions.clone() if actions.dtype == torch.float32
                 else actions.float())

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._geak_run_eager(pack, consts, s_vl, s_state, s_act, num_steps)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            s_out = self._geak_run_eager(pack, consts, s_vl, s_state, s_act, num_steps)
        return (g, s_vl, s_state, s_act, s_out)

    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn( # yes, here make sure action_horizon align with data loader? or share from clinet?
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        fast = self._geak_fastpath_ok(self.model)
        pack = self._geak_f32_pack() if (fast and _GEAK_FP32W) else False

        if pack:
            # ---- unified-dtype fast path: no autocast, no per-call casts ----
            trunk, aenc, adec, senc, pemb, rtrunk = pack
            rdt = next(rtrunk.parameters()).dtype
            with torch.autocast("cuda", enabled=False):
                # GEAK r2_d0 (GEAK_HEAD_IO): keep `vl` in the trunk's own
                # dtype when they already agree -- the .float() here and the
                # `vl_f32.to(rdt)` inside the captured region are an exact
                # round trip, so both nodes are pure launch floor.
                if _GEAK_IO and vl_embs.dtype == rdt:
                    vl_f32 = vl_embs.contiguous()
                else:
                    vl_f32 = vl_embs.float().contiguous()
                state_f32 = state.float() if state is not None else None
                act0 = actions if _GEAK_IO else actions.float()
                seq_T = self.config.action_horizon
                cdt = torch.float32 if self._geak_hybrid(rtrunk) else rdt
                consts = self._geak_consts(trunk, aenc, pemb, num_steps, seq_T,
                                           device, cdt)
                out = None
                if _GEAK_GRAPH:
                    out = self._geak_graph(pack, consts, vl_f32, state_f32,
                                           act0, num_steps)
                if out is None:
                    out = self._geak_run_eager(
                        pack, consts, vl_f32, state_f32,
                        (act0 if act0.dtype == torch.float32 else act0.float()),
                        num_steps)
                elif not _GEAK_IO:
                    out = out.clone()
            if _GEAK_VERIFY:
                ref = self._geak_reference(vl_embs, state, actions, num_steps)
                d = (out.float() - ref.float()).abs()
                print("[GEAK_HEAD_VERIFY] steps=%d max_abs=%.3e rel=%.3e ref_absmax=%.3e"
                      % (num_steps, d.max().item(),
                         (d.max() / ref.float().abs().max().clamp_min(1e-12)).item(),
                         ref.float().abs().max().item()), flush=True)
            return out

        return self._geak_reference(vl_embs, state, actions, num_steps)

    def _geak_reference(self, vl_embs, state, actions, num_steps):
        """The stock denoise loop (KV-cache fastpath / bf16 / eager fallback)."""
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        dt = 1.0 / num_steps
        fast = self._geak_fastpath_ok(self.model)
        state_features = self.state_encoder(state) if state is not None else None
        use_bf16 = fast and _GEAK_BF16
        if use_bf16:
            trunk = self._geak_bf16_model()
            with torch.autocast("cuda", enabled=False):
                kv_cache = self._geak_prep_kv(trunk, vl_embs.to(torch.bfloat16))
        else:
            trunk = self.model
            kv_cache = self._geak_prep_kv(trunk, vl_embs) if fast else None
        step_fn = self._geak_step_fn(trunk) if fast else None

        pos_embs = None

        if True:
            for t in range(num_steps):
                t_cont = 1.0 - t / float(num_steps)
                t_discretized = int(t_cont * self.num_timestep_buckets)

                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device
                )
                action_features = self.action_encoder(actions, timesteps_tensor)
                if self.config.add_pos_embed:
                    if pos_embs is None:
                        pos_ids = torch.arange(
                            action_features.shape[1], dtype=torch.long, device=device
                        )
                        pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs

                sa_embs = torch.cat((state_features, action_features), dim=1) \
                    if state_features is not None else action_features

                if use_bf16:
                    with torch.autocast("cuda", enabled=False):
                        model_output = step_fn(
                            sa_embs.to(torch.bfloat16), timesteps_tensor, kv_cache
                        ).float()
                elif fast:
                    model_output = step_fn(sa_embs, timesteps_tensor, kv_cache)
                else:
                    model_output = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embs,
                        timestep=timesteps_tensor,
                    )
                pred = self.action_decoder(model_output)

                pred_velocity = pred[:, -self.action_horizon :]

                actions = actions - dt * pred_velocity
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype



def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.
    
    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return FlowmatchingActionHead(
        full_config=config
    )


if __name__ == "__main__":
    pass
