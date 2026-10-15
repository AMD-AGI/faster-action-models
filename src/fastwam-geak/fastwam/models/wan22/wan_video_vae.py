# Based on Wan2.x / DiffSynth-Studio (Apache-2.0) via FastWAM (MIT).
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
import os as _os
from typing import Optional as _Optional

from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from .compile_config import maybe_compile
except Exception:  # pragma: no cover - standalone import fallback
    def maybe_compile(fn):
        return fn

CACHE_T = 2


# ---------------------------------------------------------------------------
# HIP stream-ordered mem-pool tuning  (pure host-side; ZERO numeric effect)
#
# MEASURED (round-1 profile of `vae_encode`, PROF_KEEP_TRACES trace):
#   9 x hipMallocAsync + 9 x hipFreeAsync per encode, all of them nested
#   directly under `aten::miopen_convolution` -> they are MIOpen's OWN internal
#   workspace / NCHW<->NHWC transpose buffers, not PyTorch caching-allocator
#   traffic.  Per steady-state call the 9 frees cost ~0.735 ms of HOST time
#   (one single free is ~0.69 ms of it) -- the largest host line item anywhere
#   in the tree, and the reason `vae_encode` is the only hot region with an
#   exposed bubble (gpu_idle 0.41 ms; the 10-step call shows 0.000 end to end).
#
# CAUSE: the HIP device default mem-pool ships with
# `hipMemPoolAttrReleaseThreshold = 0`, so every hipFreeAsync hands the pages
# straight back to the driver instead of parking them in the pool, and the next
# hipMallocAsync re-maps them.  Raising the threshold makes the pool CACHE the
# workspace across calls: same sizes, same solvers, no driver round-trip.
#
# WHAT THE FIX ACTUALLY BUYS (measured A/B on the profile, NOT assumed):
#   the 9 frees still cost ~0.74 ms of HOST time afterwards -- that number does
#   not move.  What moves is the DEVICE side: the eager unmap made the GPU queue
#   drain, so `vae_encode` gpu_idle 0.415 -> 0.120 ms and its wall 2.889 -> 2.607
#   ms, with gpu_busy flat (2.474 -> 2.486).  The residual 0.69 ms free is now
#   pure host wait fully overlapped with GPU work, and the region is GPU-bound
#   again (0.12 ms of slack left), so there is nothing more for a host lever here.
#   hipMallocAsync collapses from 0.065 ms/call to 0.009 ms/call as a bonus.
#
# This is a runtime allocator policy knob only.  It changes no shape, no dtype,
# no kernel selection and no numerics -- the oracle must stay BIT-EXACT.  It is
# also NOT episode state (no tensor is retained, nothing is keyed on data_ptr),
# so COMMANDMENT rule 5's counter-invalidation requirement does not apply: there
# is nothing to invalidate.  Idempotent and applied once per process.
#
# Kill-switch: FASTWAM_MEMPOOL=0 restores the stock behaviour.
# ---------------------------------------------------------------------------
_FW_MEMPOOL_STATE = {}

# hipMemPoolAttr enum (hip_runtime_api.h)
_HIP_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4


def _fw_tune_hip_mempool():
    """Park MIOpen's stream-ordered conv workspaces in the HIP pool.

    Best-effort: any failure (non-HIP build, symbol missing, driver refusal)
    leaves the process exactly as it was.  Returns True iff the attribute was
    actually set, so the profile A/B can assert it took effect.
    """
    if _FW_MEMPOOL_STATE.get("done"):
        return _FW_MEMPOOL_STATE.get("ok", False)
    try:
        # Never mutate a pool attribute while a HIP graph is being captured, and
        # never let dynamo try to trace the ctypes call. Both just defer to the
        # next eager call (`done` is deliberately NOT set on these paths).
        if torch.compiler.is_compiling():
            return False
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return False
    except Exception:
        pass
    _FW_MEMPOOL_STATE["done"] = True
    _FW_MEMPOOL_STATE["ok"] = False
    try:
        import os
        if os.environ.get("FASTWAM_MEMPOOL", "1") == "0":
            return False
        if not torch.cuda.is_available():
            return False
        import ctypes
        # The container ships libamdhip64.so.7 with no unversioned symlink and
        # no ldconfig entry, so a bare CDLL("libamdhip64.so") raises OSError and
        # this whole helper silently no-ops. Probe the sonames, then fall back to
        # the process's already-loaded global symbols (torch has dlopened HIP).
        lib = None
        for _cand in ("libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"):
            try:
                lib = ctypes.CDLL(_cand)
                break
            except OSError:
                continue
        if lib is None:
            lib = ctypes.CDLL(None)
        if not hasattr(lib, "hipDeviceGetDefaultMemPool"):
            return False
        dev = int(torch.cuda.current_device())          # forces lazy HIP init
        lib.hipDeviceGetDefaultMemPool.restype = ctypes.c_int
        lib.hipDeviceGetDefaultMemPool.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        lib.hipMemPoolSetAttribute.restype = ctypes.c_int
        lib.hipMemPoolSetAttribute.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        pool = ctypes.c_void_p()
        if lib.hipDeviceGetDefaultMemPool(ctypes.byref(pool), dev) != 0:
            return False
        # 4 GiB of parked workspace is far beyond what MIOpen asks for here
        # (its largest single buffer in this encode is a few hundred MB) and is
        # negligible against the 288 GB HBM on this part; a finite cap keeps the
        # pool from ever competing with the torch caching allocator.
        thresh = ctypes.c_uint64(4 << 30)
        rc = lib.hipMemPoolSetAttribute(
            pool, _HIP_MEMPOOL_ATTR_RELEASE_THRESHOLD, ctypes.byref(thresh))
        _FW_MEMPOOL_STATE["ok"] = (rc == 0)
        return _FW_MEMPOOL_STATE["ok"]
    except Exception:
        return False


def check_is_instance(model, module_class):
    if isinstance(model, module_class):
        return True
    if hasattr(model, "module") and isinstance(model.module, module_class):
        return True
    return False


def block_causal_mask(x, block_size):
    # params
    b, n, s, _, device = *x.size(), x.device
    assert s % block_size == 0
    num_blocks = s // block_size

    # build mask
    mask = torch.zeros(b, n, s, s, dtype=torch.bool, device=device)
    for i in range(num_blocks):
        mask[:, :,
             i * block_size:(i + 1) * block_size, :(i + 1) * block_size] = 1
    return mask


# ---------------------------------------------------------------------------
# F = 1 causal-conv collapse  (kill-switch: FASTWAM_VAE2D=0)
#
# STRUCTURAL FACT.  `VideoVAE38_.encode` takes a fast path guarded on
# `x.shape[2] <= 1`; `patchify(patch_size=2)` hands `Encoder3d_38.forward_nocache`
# a [1, 12, 1, 112, 224] tensor, and NOTHING in that straight-line path ever
# grows the temporal extent (Resample38.forward_nocache skips `time_conv`
# entirely, AvgDown3D has factor_t applied to T=1, AttentionBlock is Conv2d).
# So EVERY CausalConv3d in the encode path runs at T_in = 1.
#
# `CausalConv3d.__init__` rewrites the padding to `(pw,pw,ph,ph, 2*pt, 0)` and
# sets `self.padding = (0,0,0)`, so `forward` F.pads `2*pt` ZERO planes IN FRONT
# of the temporal axis.  With T_in = 1, pt = 1, kT = 3 the padded input is
# literally `[0, 0, x0]`, and
#
#     y[:, :, 0] = sum_k W[:, :, k] * xpad[:, :, k] = W[:, :, 2] * x0
#
# i.e. the two leading temporal taps multiply EXACT ZEROS.  Every such conv is
# algebraically `conv2d(x0, W[:, :, -1])`.  (Same identity for the kT = 1
# 1x1x1 shortcuts, where 2*pt = 0 and the slice is a no-op.)
#
# WHY IT PAYS (measured, region profile of `vae_encode`):
#   * MIOpen/CK's 3-D coverage falls back to `Im3d2Col` + NCHW<->NHWC
#     `batched_transpose` pre/post passes -- 0.633 ms / 57 launches of the
#     2.457 ms region that computes NOTHING.  The dense 2-D path has none of it.
#   * the conv reads 3x its true input volume (two planes of zeros), and the
#     explicit `F.pad` is fused by inductor into the PRECEDING pointwise, which
#     therefore writes 3x the bytes it needs to.  Dropping the pad and using
#     conv2d's implicit spatial padding removes both.
#
# NUMERICS: adding exact zeros is exact, so the identity itself is bit-exact.
# A different vendor kernel is still a different accumulation order, so this is
# measured against the oracle, not assumed.
#
# The collapsed weight is a small CONSTANT cache (one [Cout,Cin,kh,kw] slice per
# conv, ~1/3 of the 3-D weight) built once per process from frozen eval-mode
# weights and re-keyed on the owning weight's data_ptr.  It is NOT episode state
# -- no activation, no cache_x, nothing that varies per call is retained -- so
# the counter-invalidation rule for episode caches does not apply.
# ---------------------------------------------------------------------------
_FW_VAE2D = _os.environ.get("FASTWAM_VAE2D", "1") not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# ONE NHWC LAYOUT DOMAIN for the encode fast path  (kill-switch: FASTWAM_VAENHWC=0)
#
# MEASURED (round-2 profile of `vae_encode`): 76 `batched_transpose_*` launches
# carrying 0.3821 ms of the 1.641 ms region -- 23% of it -- around only 18
# CK/MIOpen conv calls.  They are MIOpen's OWN NCHW<->NHWC pre/post passes: its
# fast solvers are NHWC-native, so an NCHW activation costs a transpose in, a
# transpose of the weight, and a transpose out, PER CONV.  Round 1's F=1
# collapse turned 8 3-D convs into 18 2-D convs and each of them now round-trips
# its layout independently, which is why the count GREW 46 -> 76.
#
# The remedy is NOT per-conv `channels_last` (measured a LOSS in round 1: the
# NHWC tensors leaked out of the encoder and downstream `pre_dit` paid for it).
# It is a SINGLE LAYOUT DOMAIN: convert ONCE at the encoder entry, keep the whole
# 18-conv + attention chain resident in NHWC, and convert ONCE at the exit with
# an explicit `.contiguous()` so that NOTHING in NHWC can ever reach pre_dit.
# The leak is bounded by construction, not by hope.
#
# Mechanics that make it free rather than fought:
#   * every activation is [B, C, 1, H, W]; `channels_last_3d` strides are
#     (CTHW, 1, CHW, CW, C), so the `x[:, :, 0]` the F=1 conv collapse already
#     takes is EXACTLY a 4-D `channels_last` view, and the `.unsqueeze(2)` after
#     it is exactly `channels_last_3d` again.  No copy on either side.
#   * conv weights are materialized channels_last ONCE, eagerly, in
#     `_fw_prepare_conv2d` / `_fw_prepare_nhwc` -- outside any compiled region.
#     That is what makes ATen/MIOpen pick the NHWC solver instead of transposing.
#   * `AttentionBlock` gets an NHWC twin: in NHWC the [B,3C,H,W] qkv tensor is
#     ALREADY laid out as (B,H,W,3C), so the reshape+permute+`.contiguous()` at
#     the heart of the stock block (which exists only to build that order) is a
#     free view, and the same is true of the SDPA output on the way back.
#
# NUMERICS: a layout change is algebraically exact.  MIOpen/CK may select a
# different solver with a different accumulation order, so this is measured
# against the oracle, not assumed.
# ---------------------------------------------------------------------------
_FW_VAENHWC = _os.environ.get("FASTWAM_VAENHWC", "1") not in ("0", "false", "no", "off")

# THE SWITCH THAT MAKES THE WHOLE DOMAIN PAY.  ROCm PyTorch's
# `_conv_determine_backend_memory_format` refuses ChannelsLast for MIOpen unless
# `PYTORCH_MIOPEN_SUGGEST_NHWC` is set, so WITHOUT it a channels_last activation
# is copied back to NCHW before the call and the result comes back NCHW as well.
# Measured on this box over the eight encoder conv shapes (eager, bf16, 50 iters
# after 15 warmups), NHWC vs NCHW total:
#     unset : 665.2 us vs 457.8 us  -> 1.453x SLOWER, and out.is_channels_last
#             is FALSE, i.e. the layout cannot even stay resident
#     "1"   : 329.6 us vs 448.1 us  -> 0.736x FASTER, out stays channels_last
# Per-shape the win ranges 0.585x (the stride-2 downsample) to 0.803x.  That
# reverses the sign of this entire direction: it is why round 1's per-conv
# channels_last measured a LOSS, and it is why NHWC-without-this-flag measured a
# loss here too (region 1.641 -> 1.716 ms).
#
# It has to be set IN-PROCESS.  `run_local.sh` forwards only `FASTWAM_*` plus a
# fixed list, so exporting `PYTORCH_MIOPEN_SUGGEST_NHWC` outside the container is
# SILENTLY DROPPED -- the same trap that made a `PYTORCH_TUNABLEOP_*` export a
# no-op for a whole campaign.  c10 caches the lookup on first read, which happens
# at the first convolution, so module-import time is safely early.  `setdefault`
# leaves an explicit outside value alone, and FASTWAM_VAENHWC=0 skips it entirely
# so the OFF arm really is the previous tree.
if _FW_VAENHWC:
    _os.environ.setdefault("PYTORCH_MIOPEN_SUGGEST_NHWC", "1")


# ---------------------------------------------------------------------------
# WHY A CUSTOM OP AND NOT `torch._inductor.config.layout_optimization`.
#
# Inductor DOES have a channels_last conv pass (off by default on ROCm) and
# turning it on genuinely deletes all 76 transposes -- but it also hands every
# conv to a Triton template, which is SLOWER than the vendor kernel here
# (18 `triton_tem_*_convolution_*` at 0.9672 ms vs 0.6082 ms for CK), so the
# region went 1.641 -> 1.804 ms.  Forcing `max_autotune_conv_backends="ATEN"`
# instead was worse still (2.158 ms: the transposes came BACK and inductor added
# 50 `elementwise_kernel` relayout copies worth 0.484 ms).  And turning the pass
# on process-wide faulted the GPU outright in an unrelated graph
# (`Memory Fault Error ... triton_poi_fused_as_strided_clone_34`).
#
# Worse, `PYTORCH_MIOPEN_SUGGEST_NHWC=1` is INCOMPATIBLE with inductor's aten
# conv on this build.  `_meta_registrations.meta_conv`'s `pick_memory_format()`
# reads
#       if is_channels_last(self):
#           if device_hint(self) == "cuda":  return torch.contiguous_format
# i.e. on CUDA/HIP the META ALWAYS predicts an NCHW output, even for a
# channels_last input.  Verified directly under FakeTensorMode: with the flag at
# both "0" and "1", a channels_last [1,640,29,57] input meta-predicts stride
# (250880, 392, 28, 1).  That matches stock ROCm (where the runtime really does
# return NCHW) but NOT the flag-on runtime, so inductor emits
#   assert_size_stride(buf71, (1,640,14,28), (250880,392,28,1))
# and it fires -- measured.  With size asserts off it would silently index a
# channels_last buffer as if it were NCHW, which is far worse than a crash.
#
# So the conv is wrapped in an opaque custom op whose FAKE kernel declares the
# channels_last output the runtime actually produces.  Inductor then reasons
# about a layout that is true, keeps the whole chain resident in NHWC (the
# default `needs_fixed_stride_order` constraint makes the producing pointwise
# WRITE channels_last rather than write NCHW and copy), and never sees an
# `aten.convolution` node at all -- which also makes `decide_layout_opt` a no-op
# for this graph, so none of the layout-pass hazards above can fire.
# ---------------------------------------------------------------------------

def _fw_nhwc_conv2d_impl(x: torch.Tensor, w: torch.Tensor,
                         b: _Optional[torch.Tensor],
                         sh: int, sw: int, ph: int, pw: int) -> torch.Tensor:
    # Deliberately MINIMAL. This runs ~30x per encode from inside the compiled
    # region and the region is host-bound once the transposes are gone
    # (gpu_busy 0.989 ms against a 1.406 ms wall), so every extra ATen call here
    # is exposed latency, not free. The caller already emitted the input
    # relayout as a graph node, so no input `.contiguous()` is needed; the
    # trailing one is the guarantee that the op can never violate its own fake
    # kernel, and returns `self` unchanged whenever
    # PYTORCH_MIOPEN_SUGGEST_NHWC did its job (i.e. always, here).
    return F.conv2d(x, w, b, (sh, sw), (ph, pw)).contiguous(
        memory_format=torch.channels_last)


def _fw_nhwc_conv2d_meta(x: torch.Tensor, w: torch.Tensor,
                         b: _Optional[torch.Tensor],
                         sh: int, sw: int, ph: int, pw: int) -> torch.Tensor:
    n = x.shape[0]
    co, kh, kw = w.shape[0], w.shape[2], w.shape[3]
    ho = (x.shape[2] + 2 * ph - kh) // sh + 1
    wo = (x.shape[3] + 2 * pw - kw) // sw + 1
    return x.new_empty((n, co, ho, wo)).to(memory_format=torch.channels_last)


_FW_NHWC_CONV = None
if _FW_VAENHWC:
    try:
        # Low-level `Library` rather than `torch.library.custom_op`: the op is
        # called ~30x per encode from inside the compiled region, and the
        # CustomOpDef wrapper's per-call schema/device validation showed up as
        # 0.283 ms of EXPOSED host bubble (vae_encode gpu_idle went 0.000 ->
        # 0.283 while gpu_busy fell 1.650 -> 1.123). Own namespace so it cannot
        # collide with mot.py's `fastwam::` ops. Inference-only by construction
        # (the caller is under torch.no_grad), so no Autograd kernel is needed.
        _fw_lib = torch.library.Library("fastwam_vae", "FRAGMENT")
        _fw_lib.define(
            "conv2d_nhwc(Tensor x, Tensor w, Tensor? b, "
            "int sh, int sw, int ph, int pw) -> Tensor")
        _fw_lib.impl("conv2d_nhwc", _fw_nhwc_conv2d_impl, "CUDA")
        _fw_lib.impl("conv2d_nhwc", _fw_nhwc_conv2d_meta, "Meta")
        _FW_NHWC_CONV = torch.ops.fastwam_vae.conv2d_nhwc.default
    except Exception:
        try:
            _FW_NHWC_CONV = torch.ops.fastwam_vae.conv2d_nhwc.default
        except Exception:
            _FW_NHWC_CONV = None


# ---------------------------------------------------------------------------
# THE ENCODER CONVS ARE GEMMs THAT MIOpen IS RUNNING AS CONVOLUTIONS
#     (kill-switches: FASTWAM_VAECONV=0 for the whole block,
#      FASTWAM_VAEC1=0 / FASTWAM_VAEC3=0 for the two halves separately)
#
# MEASURED on this box, eager, bf16, NHWC (PYTORCH_MIOPEN_SUGGEST_NHWC=1).  Two
# numbers per shape, because they answer different questions: DEV is device time
# from a HIP-graph replay of 20 back-to-back calls (no host in the loop at all),
# HOST is the wall time of a plain async python loop.  The 29 conv calls the
# encode fast path issues cost 769.7 us of device time in total:
#
#   (H,W,Cin,Cout,k,s)         dev us  x n   total   TFLOP/s
#   (112,224,160,160,3,1)       29.2   x4    120.8     383
#   ( 56,112,320,320,3,1)       28.0   x3     82.1     423
#   ( 28, 56,640,640,3,1)       38.6   x3    117.1     296
#   ( 14, 28,640,640,3,1)       27.3   x8    230.3     100   <- 30% of the block
#   ( 14, 28,640, 96,3,1)       20.0   x1     22.7      19
#   ( 28, 56,320,640,1,1)        4.2   x1     23.8*     27   <- a 1x1
#   ( 14, 28,640,640,1,1)        4.6   x1     23.6*     14   <- a 1x1
#   ( 14, 28,640,1920,1,1)       5.3   x1     16.5*     58   <- a 1x1
#   ( 56,112,160,320,1,1)        4.5   x1     17.3*     37   <- a 1x1
#   + the three stride-2 downsamples                   61.7
#   (* the four 1x1s are HOST-bound, see below: the async-loop totals in the
#    "total" column are 3-5x their device time.)
#
# WHY THE 3x3s ARE SLOW -- and it is NOT what the TFLOP/s column suggests.  The
# implicit GEMM for a 3x3 at 14x28 is M = H*W = 392, N = Cout = 640,
# K = Cin*3*3 = 5760.  Two things are true at once:
#
#   (a) GRID FILL.  With CK's 128x128 tile that is 4*5 = 20 CTAs on a 256-CU
#       part.  The per-CTA cost is unchanged from the 28x56 shape, which is
#       exactly why 1.85 GFLOP costs the same 27-39 us as the 7.4 GFLOP
#       112x224 conv.
#   (b) RE-READ.  The obvious remedy -- shrink the tile to make more CTAs --
#       does not work, because total traffic is 2*K*(M_pad*n_tiles +
#       N_pad*m_tiles): halving the tile DOUBLES the bytes.  Measured ceiling
#       for a cache-resident working set on this box is 5.1-6.9 TB/s (a 64 MB
#       device copy runs at 6865 GB/s), so a 64x64 tile moving 143 MB for a
#       12.4 MB problem is ~22 us of pure traffic however well it computes.
#
# BOTH were measured, not reasoned about.  A hand-written Triton implicit-GEMM
# with a two-axis (tap x c-block) split-K was built and swept over ~1700
# configs; its best point is (BM,BN,BK,ST,SC,warps,stages,nonkdim) =
# (32,64,128,1,1,8,3,16) at 22.7 us -- a real 1.19x, but it lands exactly on the
# 6.3 TB/s traffic wall (143 MB in 22.7 us), and every larger-tile/split-K
# config that lowers the traffic pays it straight back in the fp32 combine.
# That whole family is a DEAD END at ~22 us; recorded so it is not re-derived.
#
# WHAT WINS is to stop hand-writing the GEMM and give the shape to hipBLASLt,
# which already has a better answer for M=392 N=640 K=5760 than either CK's conv
# solver or a Triton tile: the DENSE `[392,5760] x [5760,640]` runs in 13.6 us
# against the conv's 27.3.  The only thing between the two is the im2col gather,
# and at these spatial sizes the expanded matrix is only 4.5 MB:
#
#   (14,28,640,640):  27.25 -> 18.34 us dev  (host 35.70 -> 27.32)   x8
#   (28,56,640,640):  38.58 -> 31.03 us dev  (host 37.58 -> 31.10)   x3
#   (14,28,640, 96):  20.00 -> 12.83 us dev  (host 36.99 -> 28.10)   x1
#
# and it is a LOSS at every larger spatial size ((56,112,*) 28.0 -> 31.9,
# (112,224,160,160) 29.2 -> 40.6), because there the expanded matrix is 25-72 MB
# and writing it costs more than the conv saves.  Hence a WIN-LIST, not a
# policy: a shape not in `_FW_I2C_SHAPES` keeps the vendor kernel.
#
# THE GATHER IS TRACED TENSOR OPS, NOT A TRITON KERNEL -- and that is the whole
# result of this round's first experiment, which is worth more than the patch.
# A Triton `_fw_i2c_kernel` was written, swept and measured: 2.7 us of DEVICE
# time for the gather, and in an isolated async loop the pair beat the conv on
# BOTH axes (host 35.7 -> 27.3 us, device 27.3 -> 18.3).  END TO END IT LOST
# 0.25 ms (10-step 25.22 -> 25.53, reproduced in two interleaved rounds).
# `vae_encode` is the ONE un-graphed region in this tree, so a Triton kernel
# launched from PYTHON pays its full launcher cost on the critical path once per
# conv, and 12 of those swamp a 0.10 ms device win.  Expressed instead as tensor
# ops inside the already-compiled `_encode_single_chunk`, the SAME gather is
# emitted by inductor and launched from generated code, and because
# `constant_pad_nd` + the strided copy are pointwise-shaped it is a candidate
# for fusion into the RMS-norm/SiLU that produces the activation.
#
#   x [N,C,H,W] channels_last -> permute -> [N,H,W,C] contiguous (a VIEW)
#   pad (1,1) on H and W                                   (fusable pointwise)
#   unfold(H,3,1).unfold(W,3,1) -> [N,H,W,C,3,3]           (a VIEW)
#   permute to [N,H,W,3,3,C] and reshape -> [N*H*W, 9C]    (the materialization)
#   F.linear with the weight viewed as [Cout, 9C]          (hipBLASLt)
#
# The weight view is FREE: a channels_last [Cout,Cin,3,3] IS [Cout,3,3,Cin] in
# byte order, i.e. exactly [Cout, 9*Cin] with the taps in (r,s,c) order -- the
# same order `unfold` produces.  No weight copy, no cache, no workspace.
#
# THE FOUR 1x1 CONVS ARE A HOST STORY, NOT A DEVICE ONE, and separating the two
# is why the DEV/HOST pair is quoted above.  A 1x1 stride-1 conv on an NHWC
# activation IS `[N*H*W, Cin] @ [Cin, Cout]`: the activation is ALREADY the
# GEMM's A matrix in the right byte order, and the weight's channels_last byte
# order [Cout,Cin,1,1] is exactly `F.linear`'s [out,in].  On DEVICE the vendor
# conv is already fine and the rewrite is a wash (4.20-5.28 -> 4.79-6.05 us).
# On HOST it is not close -- 37.8-38.4 us -> 11.2-12.1 us per call, because an
# `F.conv2d` on ANY of these shapes drags MIOpen's dispatch,
# `hipPointerGetAttribute` traffic and workspace alloc/free behind it.
#
# NUMERICS.  im2col is an exact data movement (its zero border IS the conv's own
# zero padding).  What changes is that hipBLASLt accumulates the same K=5760
# products in a different order than CK's conv solver -- the same class of
# reordering as every other split-K in this tree, fp32 accumulator throughout.
# Measured on the oracle (steps=1 max|d| 6.250e-02 mean|d| 9.984e-03, i.e. the
# mean IMPROVES on the shipped 1.188e-02) and on the 14-episode rollout; not
# asserted bit-exact.  Max relative deviation vs `F.conv2d` on random bf16
# inputs: 3.0e-3 (3x3), <1e-3 (1x1).
#
# INDUCTOR SAFETY.  These paths REPLACE the opaque `fastwam_vae::conv2d_nhwc`
# call for their shapes rather than hiding inside it, so for those convs the
# graph contains no convolution node of any kind -- neither `aten.convolution`
# (whose `meta_conv` mispredicts NCHW on HIP, the hazard documented above) nor
# the custom op.  Every remaining shape still goes through the custom op exactly
# as before.  Nothing about the layout domain changes: the input is
# channels_last on the way in and the output is a channels_last view on the way
# out, which is what the rest of the encoder already expects.
# ---------------------------------------------------------------------------

def _fw_off(v):
    return v in ("0", "false", "no", "off")


# Master switch plus one per sub-lever: the two halves have DIFFERENT cost
# structures (the 1x1 rewrite deletes a dispatch, the 3x3 one adds a gather
# kernel), so integrate must be able to price them apart.  FASTWAM_VAECONV=0
# turns the whole block off and restores the previous tree exactly.
_FW_VAECONV = not _fw_off(_os.environ.get("FASTWAM_VAECONV", "1"))
_FW_VAEC1 = _FW_VAECONV and not _fw_off(_os.environ.get("FASTWAM_VAEC1", "1"))
# `_FW_VAEC3` DEFAULTS OFF.  It is kept, wired and documented because it is the
# only thing in this round that moved the REGION -- and because the reason it
# does not move the CALL is the most useful thing measured here.  3 interleaved
# rounds, medians, single built tree, arms driven by these switches:
#
#   arm    1step    10step   geomean   vae stage   prologue
#   OFF    7.9865   25.2023  14.1773     1.270       5.900
#   C1     7.9189   25.1498  14.1005     1.250       5.880   1.0054x
#   C3     8.0065   25.2765  14.2217     1.110       5.740   0.9969x  <-- LOSS
#   BOTH   7.9240   25.1174  14.1078     1.100       5.720   1.0056x
#
# C3 takes 0.16 ms off the PROLOGUE stage and 0.16 ms off the `vae_encode`
# region (gpu_busy 0.988 -> 0.864 ms in the profile, `grouped_conv_fwd`
# 0.778 -> 0.33 ms) and still loses 0.3% of the CALL, reproducibly, in all three
# rounds.  That is the COMMANDMENT's own misreading guard firing: `vae_encode`
# is host-bound and un-graphed, so device time removed from it is NOT time
# removed from the call, and the per-stage diagnostic (which syncs) reports a
# win the wall clock does not pay.  BOTH is indistinguishable from C1 alone
# (14.1078 vs 14.1005, 0.05% apart) -- C3 contributes nothing on top.
#
# So the DEFAULT ships only the half that measured: set FASTWAM_VAEC3=1 to
# re-enable the gather path (it is numerically validated -- the 14-episode
# rollout was run with BOTH halves on).
_FW_VAEC3 = _FW_VAECONV and not _fw_off(_os.environ.get("FASTWAM_VAEC3", "0"))

# (H, W, Cin, Cout) of the 3x3 stride-1 pad-1 convs where im2col + GEMM beat the
# vendor conv on device time.  See the table above for the losers.
_FW_I2C_SHAPES = frozenset((
    (14, 28, 640, 640),
    (28, 56, 640, 640),
    (14, 28, 640, 96),
))


def _fw_c3(x, w):
    """3x3 stride-1 pad-1 NHWC conv as im2col + GEMM.  None => not ours."""
    n, c, h, wd = x.shape
    co = w.shape[0]
    if n != 1 or (h, wd, c, co) not in _FW_I2C_SHAPES:
        return None
    xp = F.pad(x.permute(0, 2, 3, 1), (0, 0, 1, 1, 1, 1))
    u = (xp.unfold(1, 3, 1).unfold(2, 3, 1)
           .permute(0, 1, 2, 4, 5, 3)
           .reshape(n * h * wd, 9 * c))
    y = F.linear(u, w.permute(0, 2, 3, 1).reshape(co, 9 * c))
    return y.view(n, h, wd, co).permute(0, 3, 1, 2)


def _fw_c1(x, w):
    """1x1 stride-1 conv on an NHWC activation == `F.linear` on the same bytes.

    `x` is channels_last, so `x.permute(0,2,3,1)` is contiguous [N,H,W,Cin] and
    the reshape to [N*H*W, Cin] is a VIEW -- there is no relayout anywhere here.
    The result is [N,H,W,Cout] contiguous and the closing permute makes it the
    channels_last [N,Cout,H,W] the rest of the encoder expects.
    """
    n, c, h, wd = x.shape
    y = F.linear(x.permute(0, 2, 3, 1).reshape(-1, c),
                 w.reshape(w.shape[0], c))
    return y.view(n, h, wd, -1).permute(0, 3, 1, 2)


def _fw_conv2d(x, w, b, stride, padding):
    """conv2d that stays inside the NHWC domain, with an eager fallback."""
    if _FW_NHWC_CONV is None:
        return F.conv2d(x, w, b, stride, padding)
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (padding, padding) if isinstance(padding, int) else padding
    # Relayout OUTSIDE the opaque op, so inductor sees a real `clone` node and
    # folds it into the producing pointwise (which then simply WRITES NHWC).
    x = x.contiguous(memory_format=torch.channels_last)
    # THE GEMM WIN-LIST (see the long note above).  Taken HERE, in the traced
    # body, rather than inside the opaque op: these shapes are faster as
    # im2col + hipBLASLt than as a vendor conv, and the gather has to be a
    # graph node for inductor to emit and fuse it -- a Triton kernel launched
    # from python inside the op measured a 0.25 ms end-to-end LOSS in this
    # un-graphed region even though its device time was 0.10 ms better.
    if _FW_VAECONV:
        y = None
        if (_FW_VAEC1 and w.shape[2] == 1 and w.shape[3] == 1
                and sh == 1 and sw == 1 and ph == 0 and pw == 0):
            y = _fw_c1(x, w)
        elif (_FW_VAEC3 and w.shape[2] == 3 and w.shape[3] == 3
                and sh == 1 and sw == 1 and ph == 1 and pw == 1):
            y = _fw_c3(x, w)
        if y is not None:
            return y if b is None else y + b.view(1, -1, 1, 1)
    # PEEL THE BIAS, exactly as inductor's own conv lowering does
    # (`if bias is not None and device_type != "cpu": convolution(x, w, None)`
    # then a separate add).  Left inside the opaque op the bias add is an ATen
    # broadcast: 30 `elementwise_kernel_manual_unroll` launches worth 0.1454 ms.
    # Peeled, it is a graph node inductor folds into the NEXT fused pointwise --
    # the RMS-norm/SiLU kernel that already reads this tensor -- for free.
    y = _FW_NHWC_CONV(x, w, None, int(sh), int(sw), int(ph), int(pw))
    if b is not None:
        y = y + b.view(1, -1, 1, 1)
    return y


def _fw_conv2d_mod(m, x):
    """Run a plain nn.Conv2d module through the NHWC op."""
    if _FW_NHWC_CONV is None or m.groups != 1 or m.dilation != (1, 1):
        return m(x)
    return _fw_conv2d(x, m.weight, m.bias, m.stride, m.padding)


def _fw_prepare_nhwc(root):
    """Materialize every plain nn.Conv2d weight under `root` in channels_last.

    In-place on `.data`, value-preserving, idempotent.  Runs EAGERLY (never
    inside a compiled region / graph capture).
    """
    n = 0
    for m in root.modules():
        if isinstance(m, nn.Conv2d) and not isinstance(m, nn.Conv3d):
            w = m.weight
            if w.dim() == 4 and not w.is_contiguous(memory_format=torch.channels_last):
                m.weight.data = w.data.contiguous(memory_format=torch.channels_last)
            n += 1
    return n


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)
        # set eagerly (outside any compiled region) by _fw_prepare_conv2d()
        self._fw_w2d = None

    def forward(self, x, cache_x=None):
        w2 = self._fw_w2d
        if w2 is not None and cache_x is None and x.shape[2] == 1:
            # padded T == kT and all leading planes are zero -> pure 2-D conv
            if _FW_VAENHWC and self.groups == 1 and self.dilation[1:] == (1, 1):
                return _fw_conv2d(
                    x[:, :, 0], w2, self.bias, self.stride[1:],
                    (self._padding[2], self._padding[0])).unsqueeze(2)
            return F.conv2d(
                x[:, :, 0], w2, self.bias, self.stride[1:],
                (self._padding[2], self._padding[0]),
                self.dilation[1:], self.groups).unsqueeze(2)
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


def _fw_can_collapse(m):
    """True iff `m` at T_in == 1 is exactly conv2d with the last temporal tap."""
    return (m.stride[0] == 1
            and m.dilation[0] == 1
            and m._padding[5] == 0
            and 1 + m._padding[4] == m.kernel_size[0])


def _fw_prepare_conv2d(root):
    """Materialize `_fw_w2d` on every collapsible CausalConv3d under `root`.

    Runs EAGERLY (never inside a compiled region / graph capture): the compiled
    `_encode_single_chunk` then just sees a plain tensor attribute.
    """
    n = 0
    for m in root.modules():
        if not isinstance(m, CausalConv3d):
            continue
        if _fw_can_collapse(m):
            _w = m.weight[:, :, -1]
            m._fw_w2d = (_w.contiguous(memory_format=torch.channels_last)
                         if _FW_VAENHWC else _w.contiguous())
            n += 1
        else:
            m._fw_w2d = None
    return n


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(dim,
                                          dim * 2, (3, 1, 1),
                                          padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim,
                                          dim, (3, 1, 1),
                                          stride=(2, 1, 1),
                                          padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x, feat_cache, feat_idx

    def forward_nocache(self, x):
        """feat_cache-free equivalent of forward() for the FIRST chunk.

        Bit-exact with forward(x, feat_cache, feat_idx) when feat_cache[idx] is
        None (which is always true at iter_==0 right after clear_cache()):
          * upsample3d  -> the `feat_cache[idx] is None` branch only marks 'Rep'
                           and skips time_conv entirely.
          * downsample3d-> the `feat_cache[idx] is None` branch only stores
                           x.clone() and skips time_conv entirely.
        So the only surviving work is the spatial resample.
        """
        t = x.shape[2]
        if t == 1 and x.shape[0] == 1:
            # (b t) with b == t == 1 is a pure squeeze/unsqueeze of a size-1
            # axis -- do it with views so a channels_last_3d input stays
            # channels_last through the spatial resample (einops' reshape would
            # be free too, but this makes the layout invariant explicit).
            y = x[:, :, 0]
            for _m in (self.resample if isinstance(self.resample, nn.Sequential)
                       else [self.resample]):
                y = _fw_conv2d_mod(_m, y) if isinstance(_m, nn.Conv2d) else _m(y)
            return y.unsqueeze(2)
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        conv_weight.data[:, :, 1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)



def patchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(x,
                      "b c f (h q) (w r) -> b (c r q) f h w",
                      q=patch_size,
                      r=patch_size)
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")
    return x


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(x,
                      "b (c r q) f h w -> b c f (h q) (w r)",
                      q=patch_size,
                      r=patch_size)
    return x


class Resample38(Resample):

    def __init__(self, dim, mode):
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        super(Resample, self).__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
        else:
            self.resample = nn.Identity()

class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h, feat_cache, feat_idx

    def forward_nocache(self, x):
        # feat_cache is None -> every CausalConv3d is called with cache_x=None
        # and the clone/cat ladder is dead. Bit-exact.
        h = self.shortcut(x)
        for layer in self.residual:
            x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(
            0, 1, 3, 2).contiguous().chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            #attn_mask=block_causal_mask(q, block_size=h * w)
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity

    def forward_nocache(self, x):
        """Layout-transparent twin of forward() for the F == 1 encode path.

        Algebraically identical to forward(); the difference is that it never
        asks for an NCHW-contiguous intermediate.  The stock block builds
        [B, HW, 3C] by `reshape(B,1,3C,HW).permute(0,1,3,2).contiguous()` -- but
        (B, H, W, 3C) IS the channels_last byte order of the conv output, so on
        an NHWC tensor that whole dance is a free view, and so is the inverse on
        the SDPA result.  Falls back to forward() when the domain is off or the
        shape is not the single-frame one.
        """
        if not _FW_VAENHWC or x.shape[0] != 1 or x.shape[2] != 1:
            return self.forward(x)
        identity = x
        b, c, t, h, w = x.size()
        y = self.norm(x[:, :, 0])
        qkv = _fw_conv2d_mod(self.to_qkv, y).permute(0, 2, 3, 1).reshape(
            b, 1, h * w, c * 3)
        q, k, v = qkv.chunk(3, dim=-1)
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.reshape(b, h, w, c).permute(0, 3, 1, 2)
        y = _fw_conv2d_mod(self.proj, y)
        return y.unsqueeze(2) + identity


class AvgDown3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class Down_ResidualBlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, dropout, mult, temperal_downsample=False, down_flag=False
    ):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample38(out_dim, mode=mode))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x_copy = x.clone()
        for module in self.downsamples:
            x, feat_cache, feat_idx = module(x, feat_cache, feat_idx)

        return x + self.avg_shortcut(x_copy), feat_cache, feat_idx

    def forward_nocache(self, x):
        # No in-place writes downstream, so the defensive clone of x is dead.
        x_copy = x
        for module in self.downsamples:
            x = module.forward_nocache(x)
        return x + self.avg_shortcut(x_copy)


class Up_ResidualBlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, dropout, mult, temperal_upsample=False, up_flag=False
    ):
        super().__init__()
        # Shortcut path with upsample
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        # Main path with residual blocks and upsample
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final upsample block
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample38(out_dim, mode=mode))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        x_main = x.clone()
        for module in self.upsamples:
            x_main, feat_cache, feat_idx = module(x_main, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            x_main = x_main + x_shortcut
        return x_main, feat_cache, feat_idx


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim, dropout),
                                    AttentionBlock(out_dim),
                                    ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(),
                                  CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x, feat_cache, feat_idx


class Encoder3d_38(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = (
                temperal_downsample[i] if i < len(temperal_downsample) else False
            )
            downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        # # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )


    def forward(self, x, feat_cache=None, feat_idx=[0]):

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x, feat_cache, feat_idx

    def forward_nocache(self, x):
        """Straight-line encode of a single chunk with feat_cache disabled.

        Equivalent (bit-exact) to forward(x, feat_cache=[None]*n, feat_idx=[0]):
        every branch that fires when feat_cache[idx] is None either (a) passes
        cache_x=None to the CausalConv3d, which is exactly what happens when
        feat_cache itself is None, or (b) only writes a clone into feat_cache
        that this call never reads back. All the surviving work is the conv /
        norm / attention chain below.
        """
        x = self.conv1(x)
        for layer in self.downsamples:
            x = layer.forward_nocache(x)
        for layer in self.middle:
            if isinstance(layer, (ResidualBlock, AttentionBlock)):
                x = layer.forward_nocache(x)
            else:
                x = layer(x)
        for layer in self.head:
            x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0], dropout),
                                    AttentionBlock(dims[0]),
                                    ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(),
                                  CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x, feat_cache, feat_idx



class Decoder3d_38(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)
        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0], dropout),
                                    AttentionBlock(dims[0]),
                                    ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(
                Up_ResidualBlock(in_dim=in_dim,
                                 out_dim=out_dim,
                                 dropout=dropout,
                                 mult=num_res_blocks + 1,
                                 temperal_upsample=t_up_flag,
                                 up_flag=i != len(dim_mult) - 1))
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(),
                                  CausalConv3d(out_dim, 12, 3, padding=1))


    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x, feat_cache, feat_idx


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class VideoVAE_(nn.Module):

    def __init__(self,
                 dim=96,
                 z_dim=16,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        self.clear_cache()
        ## cache
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4

        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out, self._enc_feat_map, self._enc_conv_idx = self.encoder(x[:, :, :1, :, :],
                                   feat_cache=self._enc_feat_map,
                                   feat_idx=self._enc_conv_idx)
            else:
                out_, self._enc_feat_map, self._enc_conv_idx = self.encoder(x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                                    feat_cache=self._enc_feat_map,
                                    feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=z.dtype, device=z.device)
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out, self._feat_map, self._conv_idx = self.decoder(x[:, :, i:i + 1, :, :],
                                   feat_cache=self._feat_map,
                                   feat_idx=self._conv_idx)
            else:
                out_, self._feat_map, self._conv_idx = self.decoder(x[:, :, i:i + 1, :, :],
                                    feat_cache=self._feat_map,
                                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2) # may add tensor offload
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        # count_conv3d() walks EVERY submodule of the decoder and of the encoder.
        # The module tree is structurally frozen after __init__, so the two counts
        # are constants; the original code recomputed them on every encode/decode
        # call (twice per encode). Memoize by structure, not by data.
        counts = self.__dict__.get("_conv3d_counts")
        if counts is None:
            counts = (count_conv3d(self.decoder), count_conv3d(self.encoder))
            self.__dict__["_conv3d_counts"] = counts
        self._conv_num, self._enc_conv_num = counts
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


class WanVideoVAE(nn.Module):

    def __init__(self, z_dim=16):
        super().__init__()

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = VideoVAE_(z_dim=z_dim).eval().requires_grad_(False)
        self.upsampling_factor = 8
        self.temporal_downsample_factor = 4
        self.z_dim = z_dim


    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x


    def build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])

        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)

        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask


    def tiled_decode(self, hidden_states, device, tile_size, tile_stride):
        _, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if (h-stride_h >= 0 and h-stride_h+size_h >= H): continue
            for w in range(0, W, stride_w):
                if (w-stride_w >= 0 and w-stride_w+size_w >= W): continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device

        out_T = T * 4 - 3
        weight = torch.zeros((1, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor), dtype=hidden_states.dtype, device=data_device)
        values = torch.zeros((1, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor), dtype=hidden_states.dtype, device=data_device)

        for h, h_, w, w_ in tqdm(tasks, desc="VAE decoding"):
            hidden_states_batch = hidden_states[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.decode(hidden_states_batch, self.scale).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h==0, h_>=H, w==0, w_>=W),
                border_width=((size_h - stride_h) * self.upsampling_factor, (size_w - stride_w) * self.upsampling_factor)
            ).to(dtype=hidden_states.dtype, device=data_device)

            target_h = h * self.upsampling_factor
            target_w = w * self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h: target_h + hidden_states_batch.shape[3],
                target_w: target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        values = values.clamp_(-1, 1)
        return values


    def tiled_encode(self, video, device, tile_size, tile_stride):
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if (h-stride_h >= 0 and h-stride_h+size_h >= H): continue
            for w in range(0, W, stride_w):
                if (w-stride_w >= 0 and w-stride_w+size_w >= W): continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device

        out_T = (T + 3) // 4
        weight = torch.zeros((1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor), dtype=video.dtype, device=data_device)
        values = torch.zeros((1, self.z_dim, out_T, H // self.upsampling_factor, W // self.upsampling_factor), dtype=video.dtype, device=data_device)

        for h, h_, w, w_ in tqdm(tasks, desc="VAE encoding"):
            hidden_states_batch = video[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.encode(hidden_states_batch, self.scale).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h==0, h_>=H, w==0, w_>=W),
                border_width=((size_h - stride_h) // self.upsampling_factor, (size_w - stride_w) // self.upsampling_factor)
            ).to(dtype=video.dtype, device=data_device)

            target_h = h // self.upsampling_factor
            target_w = w // self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h: target_h + hidden_states_batch.shape[3],
                target_w: target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        return values


    def single_encode(self, video, device):
        video = video.to(device)
        x = self.model.encode(video, self.scale)
        return x


    def single_decode(self, hidden_state, device):
        hidden_state = hidden_state.to(device)
        video = self.model.decode(hidden_state, self.scale)
        return video.clamp_(-1, 1)


    def encode(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        # videos = [video.to("cpu") for video in videos]
        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            if tiled:
                raise NotImplementedError("Tiled encoding is not allowed yet.")
                tile_size = (tile_size[0] * self.upsampling_factor, tile_size[1] * self.upsampling_factor)
                tile_stride = (tile_stride[0] * self.upsampling_factor, tile_stride[1] * self.upsampling_factor)
                hidden_state = self.tiled_encode(video, device, tile_size, tile_stride)
            else:
                hidden_state = self.single_encode(video, device)
            hidden_state = hidden_state.squeeze(0)
            hidden_states.append(hidden_state)
        hidden_states = torch.stack(hidden_states)
        return hidden_states


    def decode(self, hidden_states, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        hidden_states = [hidden_state.to("cpu") for hidden_state in hidden_states]
        videos = []
        for hidden_state in hidden_states:
            hidden_state = hidden_state.unsqueeze(0)
            if tiled:
                video = self.tiled_decode(hidden_state, device, tile_size, tile_stride)
            else:
                video = self.single_decode(hidden_state, device)
            video = video.squeeze(0)
            videos.append(video)
        videos = torch.stack(videos)
        return videos


    @staticmethod
    def state_dict_converter():
        return WanVideoVAEStateDictConverter()


class WanVideoVAEStateDictConverter:

    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        state_dict_ = {}
        if 'model_state' in state_dict:
            state_dict = state_dict['model_state']
        for name in state_dict:
            state_dict_['model.' + name] = state_dict[name]
        return state_dict_


class VideoVAE38_(VideoVAE_):

    def __init__(self,
                 dim=160,
                 z_dim=48,
                 dec_dim=256,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[False, True, True],
                 dropout=0.0):
        super(VideoVAE_, self).__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d_38(dim, z_dim * 2, dim_mult, num_res_blocks,
                                    attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d_38(dec_dim, z_dim, dim_mult, num_res_blocks,
                                    attn_scales, self.temperal_upsample, dropout)


    @maybe_compile
    def _encode_single_chunk(self, x, s0, s1):
        """Whole single-chunk encode as ONE traceable region.

        Only valid when iter_ == 1 (t <= 1 after patchify), i.e. exactly the
        deployed FastWAM path: infer_action feeds ONE frame, clear_cache() has
        just run so every feat_cache slot is None, and the clone-and-store ladder
        the general path executes is write-only within the call.
        """
        x = patchify(x, patch_size=2)
        if _FW_VAENHWC:
            # ENTER the NHWC domain once.  Fused by inductor into the patchify
            # pointwise, so it costs no extra pass over memory.
            x = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.encoder.forward_nocache(x)
        mu, log_var = self.conv1(x).chunk(2, dim=1)
        mu = (mu - s0) * s1
        if _FW_VAENHWC:
            # LEAVE it once, explicitly: nothing NHWC may escape into pre_dit.
            mu = mu.contiguous()
        return mu

    def _scale_on_device(self, scale, dtype, device):
        """Memoized device copy of the (constant) latent mean / inv-std.

        `scale` is built once in WanVideoVAE38.__init__ from two literal python
        lists and never mutated, so this is a constant, NOT episode state. The
        memo is keyed by the identity of the OWNING python list plus dtype/device
        -- never by tensor data_ptr / _version -- and a different list object
        recomputes. Without it every encode() did two pageable H2D copies.
        """
        key = (id(scale), dtype, str(device))
        cached = self.__dict__.get("_scale_dev_cache")
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        s0 = scale[0].to(dtype=dtype, device=device).view(1, self.z_dim, 1, 1, 1)
        s1 = scale[1].to(dtype=dtype, device=device).view(1, self.z_dim, 1, 1, 1)
        # keep a strong ref to the source list so id() cannot be recycled
        self.__dict__["_scale_dev_cache"] = (key, s0, s1, scale)
        return s0, s1

    def encode(self, x, scale):
        # One-shot, process-global host knob: stop MIOpen's per-conv workspace
        # frees from round-tripping through the driver (see the long note at the
        # top of this file). Pure allocator policy -- no numeric effect.
        _fw_tune_hip_mempool()
        # FAST PATH: single temporal chunk (the only shape infer_action ever
        # produces). Bit-exact with the loop below -- see forward_nocache().
        if (
            x.dim() == 5
            and x.shape[2] <= 1
            and isinstance(scale, (list, tuple))
            and isinstance(scale[0], torch.Tensor)
        ):
            self.clear_cache()
            w = self.conv1.weight
            # One-shot, eager (never traced / captured): collapse every F=1
            # causal 3-D conv on the ENCODE path to its algebraically identical
            # 2-D form. Re-keyed on the owning weight so a reload rebuilds it.
            if _FW_VAE2D:
                _key = (w.data_ptr(), self.encoder.conv1.weight.data_ptr())
                if self.__dict__.get("_fw_w2d_key") != _key:
                    _fw_prepare_conv2d(self.encoder)
                    _fw_prepare_conv2d(self.conv1)
                    if _FW_VAENHWC:
                        # the plain nn.Conv2d modules on the encode path
                        # (Resample38 spatial convs, AttentionBlock 1x1s)
                        _fw_prepare_nhwc(self.encoder)
                    self.__dict__["_fw_w2d_key"] = _key
            s0, s1 = self._scale_on_device(scale, w.dtype, w.device)
            mu = self._encode_single_chunk(x, s0, s1)
            self.clear_cache()
            return mu

        self.clear_cache()
        x = patchify(x, patch_size=2)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out, self._enc_feat_map, self._enc_conv_idx = self.encoder(x[:, :, :1, :, :],
                                   feat_cache=self._enc_feat_map,
                                   feat_idx=self._enc_conv_idx)
            else:
                out_, self._enc_feat_map, self._enc_conv_idx = self.encoder(x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                                    feat_cache=self._enc_feat_map,
                                    feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu


    def decode(self, z, scale):
        self.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=z.dtype, device=z.device)
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out, self._feat_map, self._conv_idx = self.decoder(x[:, :, i:i + 1, :, :],
                                   feat_cache=self._feat_map,
                                   feat_idx=self._conv_idx,
                                   first_chunk=True)
            else:
                out_, self._feat_map, self._conv_idx = self.decoder(x[:, :, i:i + 1, :, :],
                                    feat_cache=self._feat_map,
                                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        out = unpatchify(out, patch_size=2)
        self.clear_cache()
        return out


class WanVideoVAE38(WanVideoVAE):

    def __init__(self, z_dim=48, dim=160):
        super(WanVideoVAE, self).__init__()

        mean = [
            -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,  0.1838,  0.1557,
            -0.1382,  0.0542,  0.2813,  0.0891,  0.1570, -0.0098,  0.0375, -0.1825,
            -0.2246, -0.1207, -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
            -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899, -0.2799, -0.1230,
            -0.0313, -0.1649,  0.0117,  0.0723, -0.2839, -0.2083, -0.0520,  0.3748,
            0.0152,  0.1957,  0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667
        ]
        std = [
            0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
            0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
            0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
            0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
            0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
            0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744
        ]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = VideoVAE38_(z_dim=z_dim, dim=dim).eval().requires_grad_(False)
        self.upsampling_factor = 16
        self.temporal_downsample_factor = 4
        self.z_dim = z_dim

    # ------------------------------------------------------------------
    # Mode-cached `train()` -- see the long note on `MoT.train` in mot.py.
    #
    # `FastWAM.infer_action` (IMMUTABLE) calls `self.eval()` on EVERY call and
    # `nn.Module.train(mode)` unconditionally recurses the whole module tree.
    # `WanVideoVAE38` is 319 of the pipeline's 3147 per-call module visits
    # (0.338 ms of the measured 3.225 ms `model.eval()`), all of it CPython
    # with the GPU idle.
    #
    # Short-circuiting only skips RE-writing `.training` flags that already
    # hold the requested value, so it is a no-op by construction.  The cache
    # starts *absent*, so the first call in each direction falls through to
    # `nn.Module.train` and really writes every descendant flag; a genuine
    # transition in either direction is handled the same way.  No tensor
    # identity / data_ptr keying and no episode-varying state is involved.
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        mode = bool(mode)
        if self.__dict__.get("_fw_train_mode", None) is mode:
            return self
        nn.Module.train(self, mode)
        self.__dict__["_fw_train_mode"] = mode
        return self
