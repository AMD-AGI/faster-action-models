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
# WHICH MIOpen SOLVER THE 3x3 640-CHANNEL CONVS LAND ON
#     (kill-switch: FASTWAM_VAESOLVER=0)
#
# MEASURED (prof_out/profile_summary.json, `vae_encode`, shipped tree): NINE of
# the 25 vendor convs are served by MIOpen's assembly implicit-GEMM solver
#   igemm_fwd_gtcx35_nhwc_bf16_bx0_ex1_bt128x64x32_...   0.2080 ms x 9
# and they drag TWO MIOpen housekeeping kernels each behind them:
#   SubTensorOpWithCastTensor1d  0.0380 ms x 9
#   SubTensorOpWithScalar1d      0.0360 ms x 9
# i.e. 0.074 ms -- 4.7% of the whole 1.56 ms region -- spent casting/zeroing
# that solver's workspace, plus 18 extra dispatches inside the replayed graph.
# The CK `kernel_grouped_conv_fwd` family that serves the other 16 convs has no
# such prologue.  `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS_NHWC=0`
# removes that solver from the candidate list so those nine shapes fall through
# to the next-best (CK) solution.
#
# It must be set IN-PROCESS, same trap as PYTORCH_MIOPEN_SUGGEST_NHWC above:
# `run_local.sh` forwards only `FASTWAM_*` plus a fixed list, so an export from
# outside the container is silently dropped.  MIOpen caches the lookup on first
# use, which is the first convolution, so module-import time is early enough.
# `setdefault` leaves an explicit outside value alone.
# ---------------------------------------------------------------------------
_FW_VAESOLVER = _os.environ.get("FASTWAM_VAESOLVER", "0") not in (
    "0", "false", "no", "off")
if _FW_VAESOLVER:
    _os.environ.setdefault(
        "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS_NHWC", "0")


# ---------------------------------------------------------------------------
# THE GLUE THAT THE CAPTURE BACKEND CANNOT FUSE  (kill-switch: FASTWAM_VAEGLUE=0)
#
# MEASURED, eager region trace (`prof_eager/trace_vae_encode.json`, per call,
# every kernel attributed to the ATen op that launched it):
#
#   aten::miopen_convolution      0.7427 ms x 25
#   aten::linalg_vector_norm      0.1223 ms x 22
#   fastwam_vae::rmsnorm_tail     0.1063 ms x 22
#   aten::add                     0.1057 ms x 23
#   aten::copy_                   0.0923 ms x 12   <-- pure relayout / padding
#   aten::fill_                   0.0221 ms x  6
#
# and, in the CAPTURED region, one more family that is invisible eagerly:
#   aten::_foreach_copy_          0.0802 ms x  3   <-- graph input staging
#
# The shipped backend is `cudagraphs` (aot-eager + capture), NOT inductor, so
# NOTHING in this region is fused for us -- every one of those ops is a real
# kernel inside the replayed graph at the 4.7-5.6 us launch floor.  Two of them
# are pure waste and are deleted here, both BIT-EXACTLY:
#
#  (a) `AvgDown3D.forward` unconditionally calls `F.pad(x, (0,0,0,0,pad_t,0))`.
#      On the encode path T == 1 and factor_t == 1, so `pad_t == 0` and that is
#      a full-tensor COPY that changes nothing -- three of them (the three
#      `Down_ResidualBlock.avg_shortcut`es with factor_s == 2, at 160x112x224,
#      320x56x112 and 640x28x56).  Skipping a zero-width pad is an identity.
#
#  (b) the encode graph's three non-parameter inputs are `x`, `s0` and `s1`.
#      `s0`/`s1` are the CONSTANT latent mean / inv-std, already memoized on
#      `self` by `_scale_on_device` and never reallocated, so the capture
#      backend re-copies 2 tensors per call for nothing.
#      `torch._dynamo.mark_static_address` tells it their address is fixed and
#      the staging copy is dropped.  This is a CONSTANT, not episode state:
#      `_scale_on_device`'s memo is keyed on `id(scale)` + dtype + device and a
#      different scale list rebuilds both the tensors and the mark (COMMANDMENT
#      rule 6 has nothing to invalidate -- no activation is retained).
#
# Neither touches any arithmetic: (a) removes a copy of a tensor onto itself,
# (b) removes a copy of a tensor whose address the graph can read directly.
# ---------------------------------------------------------------------------
_FW_VAEGLUE = _os.environ.get("FASTWAM_VAEGLUE", "1") not in (
    "0", "false", "no", "off")

# ---------------------------------------------------------------------------
# THE 3x3 640-CHANNEL CONVS AS ONE-KERNEL im2col + GEMM  (FASTWAM_VAEI2C)
#
# `FASTWAM_VAEC3` (shipped OFF) already proved the GEMM half: hipBLASLt runs
# [392,5760]x[5760,640] in 13.6 us where the vendor conv needs 23-27.  What
# sank it was the GATHER: it was expressed as traced tensor ops
# (`pad -> unfold -> permute -> reshape`), which under the shipped `cudagraphs`
# (aot-eager) backend is TWO full materializations of the expanded matrix, not
# the one inductor would have fused.  Re-measured on the captured tree at the
# top of this round: 3 interleaved rounds, C3 loses 0.46% of the CALL and its
# own `vae` stage diagnostic goes UP (1.605 -> 1.646 ms).  That direction is
# dead in this form.
#
# This is the same algebra with ONE gather kernel: ATen's own `im2col`
# (`F.unfold`), which writes the [9C, L] expanded matrix in a single pass.  The
# weight matrix is materialized ONCE, eagerly, in `_fw_prepare_conv2d` in the
# NCHW (c, kh, kw) order `im2col` produces, so there is no per-call reshape.
# ---------------------------------------------------------------------------
_FW_VAEI2C = _os.environ.get("FASTWAM_VAEI2C", "0") not in (
    "0", "false", "no", "off")


# ---------------------------------------------------------------------------
# THE 8 x 3x3 [640,640] @ 14x28 CONVS AS *ONE* TRITON GATHER + ONE GEMM
#     (kill-switch: FASTWAM_VAEGATH=0)
#
# READ OFF THE AOT GRAPH, then priced against the region profile (both on THIS
# tree, `prof_out_base/profile_summary.json`, vae_encode wall 1.0721 ms):
#
#   igemm_fwd_gtcx35_nhwc_bf16_..._bt128x64x32   0.1823 ms x 8   (22.8 us ea)
#   SubTensorOpWithCastTensor1d                  0.0409 ms x 9
#   SubTensorOpWithScalar1d                      0.0379 ms x 9
#
# The two SubTensorOp families are NOT dead work and NOT deletable on their own:
# they are MIOpen's split-K workspace management for that ASM solver -- a
# `SetTensor` zeroing the fp32 accumulator and a `CastTensor` narrowing it back
# to bf16, one pair per launch.  Proof (measured, 3 profile arms this round):
# `FASTWAM_VAESOLVER=1` removes the solver, both SubTensorOp families and 18
# dispatches -- and the region gets SLOWER, 1.0721 -> 1.1049 ms, because the CK
# fallback for those shapes costs more than the solver plus its housekeeping.
# So the 0.075 ms is the PRICE of the fastest vendor conv, not a saving.
#
# The way to actually collect it is to stop calling a vendor conv for those
# eight shapes.  `FASTWAM_VAEC3` / `FASTWAM_VAEI2C` already proved the GEMM half
# (hipBLASLt does [392,5760]x[5760,640] in ~13.6 us where the vendor conv needs
# 22.8 + 9.9 of housekeeping) and both still LOSE, measured again here
# (1.1021 / 1.1125 ms), because of the GATHER:
#   VAEC3 : pad->unfold->permute->reshape as traced tensor ops == TWO full
#           materializations under the aot-eager capture backend
#           (elementwise_kernel_manual_unroll 0.0569 x12 -> 0.2121 x36).
#   VAEI2C: `aten::im2col` 0.0904 x12 PLUS `__amd_rocclr_copyBuffer` 0.0531 x12
#           for the transposed operand.
#
# This is the same algebra with the gather written as ONE Triton kernel that
# emits the expanded matrix DIRECTLY in the [L, 9*Cin] row-major order the GEMM
# wants, straight out of the channels_last activation -- no pad, no permute, no
# transpose, no second pass.  [392, 5760] bf16 = 4.5 MB written once.
#
# The weight is materialized ONCE, eagerly, in `_fw_prepare_conv2d`, in the
# matching (kh, kw, c) order, so `F.linear(U, wgat)` is a plain NT GEMM whose
# OUTPUT is already the [L, Cout] byte order of a channels_last activation --
# nothing relayouts on either side of it.
#
# NUMERICS: this is a different conv algorithm (hipBLASLt fp32-accumulate GEMM
# instead of MIOpen's split-K fp32 accumulator), so it is NOT bit-exact and is
# measured on the oracle `mean|d|` and on the 14-episode closed-loop rollout,
# never asserted.  The gather itself is pure data movement (exact); only the
# contraction order changes.  FASTWAM_VAEGATH=0 restores the vendor conv.
#
# EPISODE STATE: none.  `_fw_wgat` is a frozen eval-mode weight view rebuilt on
# the same weight-identity `_key` the F=1 collapse already uses; no activation
# is retained and nothing is keyed on a data_ptr.
# ---------------------------------------------------------------------------
_FW_VAEGATH = _os.environ.get("FASTWAM_VAEGATH", "1") not in (
    "0", "false", "no", "off")

# (H, W, Cin, Cout) of the 3x3 stride-1 pad-1 convs served by the gather+GEMM.
# Override with FASTWAM_VAEGATH_SHAPES="h,w,ci,co;h,w,ci,co;..." (measurement
# knob -- the shipped default is the set that measured).
def _fw_gath_shapes():
    raw = _os.environ.get("FASTWAM_VAEGATH_SHAPES", "").strip()
    if not raw:
        # MEASURED this round, `vae_encode` gpu_busy, one profile per set
        # (base 1.0637 ms):
        #   {(14,28,640,640)}                        1.0014
        #   + (14,28,640,96)   [the head conv]        0.9777  <-- SHIPPED
        #   + (28,56,*,640)                           0.9854
        #   + every 3x3 s1p1 conv in the encoder      1.1685
        # The win is confined to the shapes whose spatial extent is too small
        # for the vendor conv to fill the GPU (14x28 = 392 pixels: MIOpen's ASM
        # solver needs 22.8 us + 9.9 us of split-K housekeeping for 2.89 GFLOP
        # = 89 TFLOP/s).  At 28x56 and above CK is already faster than gather +
        # GEMM, and forcing them all through it LOSES 0.10 ms.
        return frozenset(((14, 28, 640, 640), (14, 28, 640, 96)))
    out = []
    for grp in raw.split(";"):
        grp = grp.strip()
        if not grp:
            continue
        try:
            a, b, c, d = (int(v) for v in grp.split(","))
        except Exception:
            continue
        out.append((a, b, c, d))
    return frozenset(out)


_FW_GATH_SHAPES = _fw_gath_shapes()
_FW_GATH_STATE = {}


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

# ---------------------------------------------------------------------------
# GRAPH THE ONE UN-GRAPHED REGION  (kill-switch: FASTWAM_VAEGRAPH=0)
#
# MEASURED (round-1 baseline profile, EVAL_DIR/profiling_summary.md region
# table): `vae_encode` runs 263 kernels behind 263 `hipLaunchKernel` -- it is
# the ONLY region of `infer_action` that is not behind a graph.  Every other hot
# region replays 761-783 kernels behind 1-2 launch-API calls.  The cost of that
# asymmetry: cpu_launch_ms = 1.548 and gpu_idle = 0.881 ms (30%) of a 2.937 ms
# wall against a gpu_busy floor of 2.055 ms.  At the CALL level this region is
# essentially all of the 8.331 ms / 36.5% gpu_idle the 1-step case carries (the
# 10-step case measures gpu_idle = 0.000, i.e. the bubble is exclusively here).
#
# `_encode_single_chunk` was already written as ONE traceable region for exactly
# this purpose (see its docstring) and carried a `@maybe_compile` decorator --
# but `compile_config.maybe_compile` does not exist in this tree, so the import
# falls back to the identity and the decorator has been an inert no-op.  This
# block supplies the missing compile site locally, inside the one file this lane
# owns, instead of resurrecting a cross-file infrastructure module.
#
# BACKEND (FASTWAM_VAEGRAPH_BACKEND, informational -- the shipped default is the
# one that measured):
#   cudagraphs  (DEFAULT) torch.compile(backend="cudagraphs", fullgraph=True) --
#               aot-eager kernels, capture only.  No fusion, no codegen: keeps
#               the vendor / custom-op kernel sequence byte-for-byte and only
#               removes the host dispatch.  Oracle measures
#               max|d| = mean|d| = 0.000e+00 at BOTH step counts, i.e. this
#               direction spends ZERO of the campaign's numerics budget.  This
#               is also the backend `FastWAM._encode_video_latents` already uses
#               for the (unused-here) batched video encode path.
#   inductor    torch.compile(mode="reduce-overhead", fullgraph=True) -- inductor
#               codegen PLUS cudagraph-trees capture.  MEASURED FASTER (see the
#               report) and **REJECTED**: its pointwise fusion / reduction
#               re-association moves the encode output enough to burn 91% of the
#               mean|d| budget at 1 step (2.048e-02 / 2.250e-02, max 9.375e-02 /
#               1.250e-01) and it then FAILS the 14-episode closed-loop rollout
#               -- pooled SR = 0.9286 on all 3 attempts, libero_spatial 0.75,
#               deterministically.  Kept selectable so the finding is
#               reproducible, never defaulted.  `inductor_ep` adds scoped
#               `emulate_precision_casts` and is WORSE (oracle max|d| 2.344e-01).
#   eager       no capture at all == FASTWAM_VAEGRAPH=0.
#
# RULE 6 (episode state).  The only things cached on `self` here are (a) the
# compiled callable, re-keyed on the SAME weight-identity key the F=1 collapse
# already uses, so a checkpoint reload rebuilds it, and (b) whatever cudagraph
# trees keeps internally.  Neither is episode-varying DATA: cudagraph trees
# copies non-static inputs into its own static placeholders on every call, so a
# new episode's frame can never be served from a stale buffer, and the result is
# `.clone()`d at the boundary below so a cudagraph's internal output buffer is
# NEVER handed back to `infer_action` (which then feeds it to `video_expert.
# prepare` and, transitively, to the prefill graph).  Validated by the
# 14-episode closed-loop rollout, not by the oracle.
#
# NUMERICS: the shipped `cudagraphs` backend replays the SAME kernels the eager
# region dispatched, so the encode is BIT-EXACT (oracle 0.000e+00 at 1 and 10
# steps) and the whole tolerance budget stays unspent for the numerically active
# lanes.  Capture changes no math; only the inductor backend does, which is
# exactly why it is not the default.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# STOP THE CUDAGRAPH INPUT RE-STAGING  (kill-switch: FASTWAM_VAESTATIC=0)
#
# MEASURED (prof_out/profile_summary.json on the shipped tree, `vae_encode`):
#
#   multi_tensor_apply_kernel   0.0833 ms x 3      (6.3% of a 1.319 ms region)
#   hipLaunchKernel             0.0097 ms x 3      (the ONLY 3 eager launches
#                                                   left; everything else is
#                                                   behind the 1 hipGraphLaunch)
#
# Those three kernels are `aten::_foreach_copy_`: cudagraph-trees staging the
# graph's inputs into its own static placeholders before every replay.  They
# compute NOTHING -- they copy the encoder's weights, the collapsed 2-D weight
# slices and the two latent-scale constants onto themselves, byte for byte, at
# 320/262 CTAs, on every single call.
#
# WHY `mark_static_address` ALONE DID NOT REMOVE THEM (round-1 finding, and the
# reason this needs a backend and not another decorator).  Read
# `torch/_dynamo/backends/cudagraphs.py`: the stock `cudagraphs` backend passes
#
#     fixed = num_fw_fixed_arguments(len(dynamo_inputs), len(aot_inputs))
#     cudagraphify_impl(interp, aot_inputs, range(fixed), ...)
#
# i.e. its static-input set is *positional* -- the params AOTAutograd lifted in
# front of the real arguments -- and `num_fw_fixed_arguments` is literally
# `aot_fw_gm_num_inputs - dynamo_gm_num_inputs`, which is ZERO under
# `inline_inbuilt_nn_modules` (the default since 2.6) because dynamo lifts the
# parameters itself.  The backend NEVER consults `_dynamo_static_input_type`,
# so `mark_static_address` is inert here: only inductor's `compile_fx` path
# reads that attribute.  That is exactly what the round-1 A/B saw.
#
# THE FIX is the same backend with the static set computed by IDENTITY instead
# of by position: every input tensor that has been explicitly marked static
# (`_fw_mark_static` below marks the encoder's parameters/buffers, the eagerly
# built `_fw_w2d` / `_fw_wmat` collapsed weights, and the two latent-scale
# constants) is handed to `cudagraphify_impl` as a static index, so the graph
# reads it in place.  `x` -- the one input that really does change every call --
# is deliberately NOT marked, so it keeps its staging copy and a new episode's
# frame can never be served from a stale buffer.
#
# CORRECTNESS / RULE 6.  `mark_static_address(guard=True)` installs a data_ptr
# GUARD, so if any marked tensor is ever reallocated dynamo RECOMPILES rather
# than replaying a graph that points at freed memory; on top of that the marks
# and the compiled graph are both re-keyed on the same weight-identity `_key`
# the F=1 collapse uses, so a checkpoint reload rebuilds everything.  Nothing
# episode-varying is marked: weights are frozen eval-mode constants, `s0`/`s1`
# are memoized constants built from two literal python lists.
#
# NUMERICS: removing a copy of a tensor onto itself changes no arithmetic.  The
# replayed kernel sequence is identical (oracle must stay bit-exact).
# ---------------------------------------------------------------------------
_FW_VAESTATIC = not _fw_off(_os.environ.get("FASTWAM_VAESTATIC", "1"))
_FW_VAESTATIC_STATE = {}


def _fw_mark_static(*objs):
    """mark_static_address every parameter/buffer/collapsed weight in `objs`.

    Idempotent, eager-only, no-op on any failure.  Marking is what makes the
    identity-based backend below able to see them.
    """
    if not _FW_VAESTATIC:
        return 0
    try:
        mark = torch._dynamo.mark_static_address
    except Exception:
        return 0
    n = 0
    for o in objs:
        if o is None:
            continue
        ts = []
        if isinstance(o, torch.Tensor):
            ts.append(o)
        else:
            try:
                ts.extend(o.parameters())
                ts.extend(o.buffers())
                for m in o.modules():
                    for a in ("_fw_w2d", "_fw_wmat", "_fw_wgat"):
                        t = getattr(m, a, None)
                        if isinstance(t, torch.Tensor):
                            ts.append(t)
            except Exception:
                continue
        for t in ts:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                try:
                    mark(t)
                    n += 1
                except Exception:
                    pass
    return n


def _fw_cudagraphs_static(dynamo_model, dynamo_inputs):
    """`backend="cudagraphs"` with an IDENTITY-based static-input set.

    Byte-identical to torch's own backend except for the `static_input_idxs`
    argument: see the note above.  Any import/API drift falls straight back to
    the stock backend, never to a wrong answer.
    """
    import functools as _ft
    from torch._dynamo.backends.common import aot_autograd
    from torch._dynamo.backends.debugging import boxed_nop
    from torch._dynamo.backends.cudagraphs import (
        check_for_skip, find_input_mutations, get_device_index,
        get_stack_traces)
    from torch._inductor.compile_fx import get_static_input_idxs
    from torch._inductor.cudagraph_trees import cudagraphify_impl
    from torch._inductor.cudagraph_utils import get_placeholder_info
    from torch._inductor.utils import num_fw_fixed_arguments

    def _fw(aot_model, aot_inputs, is_inference=False):
        interp = boxed_nop(aot_model, aot_inputs)
        fixed = num_fw_fixed_arguments(len(dynamo_inputs), len(aot_inputs))
        if check_for_skip(aot_model, fixed):
            return interp
        # EXACTLY what inductor's own cudagraph path uses:
        # `TracingContext.fw_metadata.static_input_indices`, i.e. the inputs AOT
        # recorded as parameters or as explicitly `mark_static_address`ed --
        # identity, not position.  Falls back to `range(fixed)` (the stock
        # backend's answer) if the tracing context is absent.
        static = list(get_static_input_idxs(fixed))
        if not _FW_VAESTATIC_STATE.get("noted"):
            _FW_VAESTATIC_STATE["noted"] = True
            try:
                print(f"[fw-vaestatic] aot_inputs={len(aot_inputs)} "
                      f"dynamo_inputs={len(dynamo_inputs)} fixed={fixed} "
                      f"static={len(static)}", flush=True)
            except Exception:
                pass
        out = cudagraphify_impl(
            interp, aot_inputs, static,
            device_index=get_device_index(aot_model),
            is_backward=False, is_inference=is_inference,
            stack_traces=get_stack_traces(aot_model),
            placeholders=get_placeholder_info(aot_model.graph),
            mutated_input_idxs=find_input_mutations(aot_model.graph),
        )
        out._boxed_call = True
        return out

    return aot_autograd(
        fw_compiler=_fw,
        inference_compiler=_ft.partial(_fw, is_inference=True),
        keep_inference_input_mutations=(
            torch._dynamo.config.cudagraph_backend_keep_input_mutation),
    )(dynamo_model, dynamo_inputs)


_FW_VAEGRAPH = not _fw_off(_os.environ.get("FASTWAM_VAEGRAPH", "1"))
_FW_VAEGRAPH_BACKEND = _os.environ.get("FASTWAM_VAEGRAPH_BACKEND", "cudagraphs").strip().lower()
_FW_VAEGRAPH_STATE = {}


def _fw_build_encode_graph(fn):
    """Wrap the single-chunk encode in a capture backend.  None => stay eager."""
    if not _FW_VAEGRAPH or _FW_VAEGRAPH_BACKEND in ("eager", "off", "none"):
        return None
    try:
        if _FW_VAEGRAPH_BACKEND == "cudagraphs":
            g = torch.compile(
                fn,
                backend=(_fw_cudagraphs_static if _FW_VAESTATIC
                         else "cudagraphs"),
                fullgraph=True)
        elif _FW_VAEGRAPH_BACKEND == "inductor_nocg":
            g = torch.compile(fn, fullgraph=True)
        else:
            if _FW_VAEGRAPH_BACKEND == "inductor_ep":
                # Make inductor's fused pointwise chains round-trip through
                # bf16 exactly where eager does, instead of carrying fp32
                # intermediates across a fusion boundary.  Passed as `options`
                # (NOT as a global `torch._inductor.config` write) so it is
                # scoped to THIS compilation and cannot leak into the two
                # `infer_action` regions this lane does not own.
                # `triton.cudagraphs=True` is what mode="reduce-overhead" sets.
                g = torch.compile(fn, fullgraph=True, options={
                    "triton.cudagraphs": True,
                    "emulate_precision_casts": True,
                })
            else:
                g = torch.compile(fn, mode="reduce-overhead", fullgraph=True)
    except Exception:
        return None
    return g


def _fw_vaegraph_note(tag):
    """One line per process proving which encode path actually engaged."""
    if _FW_VAEGRAPH_STATE.get(tag):
        return
    _FW_VAEGRAPH_STATE[tag] = True
    try:
        print(f"[fw-vaegraph] {tag} (FASTWAM_VAEGRAPH={'1' if _FW_VAEGRAPH else '0'} "
              f"backend={_FW_VAEGRAPH_BACKEND})", flush=True)
    except Exception:
        pass


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


def _fw_c3_i2c(x, wmat):
    """3x3 s1 p1 conv as ONE `aten::im2col` + one hipBLASLt GEMM.

    `wmat` is the pre-materialized [Cout, 9*Cin] view of the weight in im2col's
    own (c, kh, kw) order.  The GEMM is written so its OUTPUT is already the
    [L, Cout] byte order of a channels_last activation -- no relayout on either
    side of it.
    """
    n, c, h, wd = x.shape
    co = wmat.shape[0]
    u = F.unfold(x, 3, padding=1)          # [1, 9C, L]
    y = torch.mm(u.view(9 * c, h * wd).t(), wmat.t())   # [L, Cout]
    return y.view(n, h, wd, co).permute(0, 3, 1, 2)


def _fw_c1(x, w, b=None):
    """1x1 stride-1 conv on an NHWC activation == `F.linear` on the same bytes.

    `x` is channels_last, so `x.permute(0,2,3,1)` is contiguous [N,H,W,Cin] and
    the reshape to [N*H*W, Cin] is a VIEW -- there is no relayout anywhere here.
    The result is [N,H,W,Cout] contiguous and the closing permute makes it the
    channels_last [N,Cout,H,W] the rest of the encoder expects.

    `b` rides in the GEMM's own epilogue (the Tensile kernels this lands on are
    the `..._Bias_...` variants and take it for free) instead of a separate
    full-tensor `aten::add` launch -- four of those, 0.017 ms, at the four 1x1
    sites (the two Down_ResidualBlock shortcuts and the AttentionBlock's
    to_qkv / proj).  It also drops one bf16 rounding: the bias is applied to the
    fp32 accumulator instead of to an already-rounded bf16 product.
    """
    n, c, h, wd = x.shape
    y = F.linear(x.permute(0, 2, 3, 1).reshape(-1, c),
                 w.reshape(w.shape[0], c), b)
    return y.view(n, h, wd, -1).permute(0, 3, 1, 2)


def _fw_conv2d(x, w, b, stride, padding, res=None, wmat=None,
               defer_bias=False, wgat=None):
    """conv2d that stays inside the NHWC domain, with an eager fallback.

    `res` (optional) is a skip connection the caller would otherwise add in a
    SEPARATE elementwise kernel.  When the bias is peeled (below) the tail is
    already `y + b`, so `(y + b) + h` costs one fused dispatch instead of two
    and one pass over HBM instead of two.  Both rounds are preserved, so it is
    bit-exact with `(conv + bias) + h`.
    """
    if _FW_NHWC_CONV is None:
        if defer_bias and res is None:
            return F.conv2d(x, w, None, stride, padding), b
        y = F.conv2d(x, w, b, stride, padding)
        return y if res is None else y + res
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
            if _FW_VAER4 and res is None and not defer_bias:
                # bias in the GEMM epilogue, no separate elementwise pass
                return _fw_c1(x, w, b)
            y = _fw_c1(x, w)
        elif (_FW_VAEGATH and wgat is not None and w.shape[2] == 3
                and w.shape[3] == 3 and sh == 1 and sw == 1
                and ph == 1 and pw == 1 and x.shape[0] == 1
                and (x.shape[2], x.shape[3], w.shape[1], w.shape[0])
                in _FW_GATH_SHAPES):
            y = _fw_gath_conv(x, wgat)
        elif (_FW_VAEI2C and wmat is not None and w.shape[2] == 3
                and w.shape[3] == 3 and sh == 1 and sw == 1
                and ph == 1 and pw == 1 and x.shape[0] == 1
                and (x.shape[2], x.shape[3], w.shape[1], w.shape[0])
                in _FW_I2C_SHAPES):
            y = _fw_c3_i2c(x, wmat)
        elif (_FW_VAEC3 and w.shape[2] == 3 and w.shape[3] == 3
                and sh == 1 and sw == 1 and ph == 1 and pw == 1):
            y = _fw_c3(x, w)
        if y is not None:
            if defer_bias and res is None:
                return y, b
            # Same epilogue fusion the vendor-conv path below already gets:
            # `(y + bias) + res` in ONE dispatch instead of two, both rounds
            # preserved (bit-exact).  Under the shipped `cudagraphs` (aot-eager)
            # capture backend nothing fuses these for us, so each of these adds
            # is a real ~5.6 us elementwise launch inside the replayed graph.
            if (res is not None and b is not None and _FW_VAEFUSE_ADD
                    and _fw_can_fuse(y, b) and b.numel() == y.shape[1]
                    and res.shape == y.shape and res.dtype == y.dtype
                    and _fw_cl_flat(y) is not None
                    and _fw_cl_flat(res) is not None):
                return torch.ops.fastwam_vae.biasadd_res.default(y, b, res)
            if b is not None:
                y = y + b.view(1, -1, 1, 1)
            return y if res is None else y + res
    # PEEL THE BIAS, exactly as inductor's own conv lowering does
    # (`if bias is not None and device_type != "cpu": convolution(x, w, None)`
    # then a separate add).  Left inside the opaque op the bias add is an ATen
    # broadcast: 30 `elementwise_kernel_manual_unroll` launches worth 0.1454 ms.
    # Peeled, it is a graph node inductor folds into the NEXT fused pointwise --
    # the RMS-norm/SiLU kernel that already reads this tensor -- for free.
    y = _FW_NHWC_CONV(x, w, None, int(sh), int(sw), int(ph), int(pw))
    if defer_bias and res is None:
        return y, b
    if (res is not None and b is not None and _FW_VAEFUSE_ADD
            and _fw_can_fuse(y, b) and b.numel() == y.shape[1]
            and res.shape == y.shape and res.dtype == y.dtype
            and _fw_cl_flat(res) is not None):
        return torch.ops.fastwam_vae.biasadd_res.default(y, b, res)
    if b is not None:
        y = y + b.view(1, -1, 1, 1)
    return y if res is None else y + res


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
        self._fw_wmat = None
        self._fw_wgat = None

    def forward(self, x, cache_x=None, res=None, defer_bias=False):
        w2 = self._fw_w2d
        if w2 is not None and cache_x is None and x.shape[2] == 1:
            # padded T == kT and all leading planes are zero -> pure 2-D conv
            if _FW_VAENHWC and self.groups == 1 and self.dilation[1:] == (1, 1):
                _r = _fw_conv2d(
                    x[:, :, 0], w2, self.bias, self.stride[1:],
                    (self._padding[2], self._padding[0]),
                    res=None if res is None else res[:, :, 0],
                    wmat=self._fw_wmat, defer_bias=defer_bias,
                    wgat=self._fw_wgat)
                if defer_bias and res is None:
                    return _r[0].unsqueeze(2), _r[1]
                return _r.unsqueeze(2)
            y = F.conv2d(
                x[:, :, 0], w2, self.bias, self.stride[1:],
                (self._padding[2], self._padding[0]),
                self.dilation[1:], self.groups).unsqueeze(2)
            return y if res is None else y + res
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        y = super().forward(x)
        return y if res is None else y + res


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
            m._fw_wmat = None
            m._fw_wgat = None
            if (_FW_VAEI2C and _w.dim() == 4 and _w.shape[2] == 3
                    and _w.shape[3] == 3 and _w.shape[1] >= 320):
                # [Cout, Cin*3*3] in the (c, kh, kw) order aten::im2col emits.
                m._fw_wmat = _w.contiguous().reshape(_w.shape[0], -1)
            if (_FW_VAEGATH and _w.dim() == 4 and _w.shape[2] == 3
                    and _w.shape[3] == 3):
                # [Cout, 3*3*Cin] in the (kh, kw, c) order `_fw_i2c_gather_k`
                # emits -- so `F.linear(U, wgat)` is a plain NT GEMM and its
                # output is already channels_last byte order.
                m._fw_wgat = (_w.permute(0, 2, 3, 1)
                                .reshape(_w.shape[0], -1).contiguous())
            n += 1
        else:
            m._fw_w2d = None
            m._fw_wmat = None
            m._fw_wgat = None
    return n


# ---------------------------------------------------------------------------
# HAND-FUSE THE ELEMENTWISE TAIL OF THE ENCODE  (kill-switch: FASTWAM_VAEFUSE=0)
#
# WHY THIS AND NOT MORE CAPTURE.  Round 1 graphed `_encode_single_chunk`, so
# `vae_encode` now replays 276 kernels behind 4 launch-API calls with
# `gpu_idle` EXACTLY 0.000 and `cpu_launch_ms` 0.080.  There is no host bubble
# left in this region -- it is DEVICE-DISPATCH-COUNT bound.  Measured today
# (`prof_out/profile_summary.json`, quiet box):
#
#   region gpu_busy 2.0963 ms over 276.4 kernels, of which ONE family is math:
#     ck kernel_grouped_conv_fwd            0.6978 ms x 22.8  @ 30.6us  (33%)
#   and the rest is elementwise/reduce GLUE pinned at the 4.7-5.6us floor:
#     elementwise_kernel_manual_unroll<128,8> 0.4974 x 89.1  @ 5.58us  (24%)
#     vectorized_elementwise<8> (anon)        0.1987 x 42.6  @ 4.66us
#     vectorized_elementwise<8> CUDAFunctor   0.1626 x 32.7  @ 4.98us
#     reduce_kernel<512,1,ReduceOp<BFloat16>> 0.1097 x 21.8  @ 5.03us
#     vectorized_elementwise<8> AUnaryFunctor 0.1052 x 21.8  @ 4.83us
#
# The `x 21.8` pair is the signature of the encoder's **22 RMS_norm modules**
# (8 in the four Down_ResidualBlocks' 8 ResidualBlocks, 4 in the two middle
# ResidualBlocks, 1 in the AttentionBlock, 1 in `head`; each ResidualBlock has
# two).  `RMS_norm.forward` is
#
#     F.normalize(x, dim=1) * self.scale * self.gamma + self.bias
#
# and `F.normalize` is itself `x / x.norm(2, dim, keepdim=True).clamp_min(eps)`.
# Eager therefore dispatches SIX kernels per norm --
#   norm (reduce) | clamp_min | div | mul-by-scalar | mul-by-gamma | add-bias
# -- five of which are pure pointwise passes over the SAME activation.  At
# stage 0 that activation is 160 x 112 x 224 = 4.0 M bf16 elements, so each of
# those passes moves 16 MB to compute one multiply.  22 x 6 = 132 dispatches,
# i.e. HALF the region's kernel count, for one normalize.
#
# `backend="cudagraphs"` CAPTURES those launches but does not FUSE them, which
# is exactly why the count survived round 1.  This block collapses the five
# pointwise kernels into ONE Triton kernel: 132 -> 44 dispatches, and four
# fewer round trips through HBM per norm.
#
# NUMERICS -- THE CONSTRAINT THAT DECIDES THE IMPLEMENTATION.  The committed
# stack already spends 75-76% of the end-to-end tolerance, and round 1 proved
# that a fusion which merely *approximates* eager (inductor, which keeps fp32
# intermediates across a fused pointwise chain) passes the oracle at 91% of the
# mean|d| budget and then DETERMINISTICALLY FAILS the 14-episode rollout at
# pooled SR=0.9286.  So this kernel is written to be BIT-EXACT, not "close":
#
#   * every intermediate is ROUNDED BACK TO THE STORAGE DTYPE after each op,
#     exactly where eager's TensorIterator rounds, instead of being carried in
#     fp32 across the chain.  That is the single difference between this and
#     what inductor did, and it is the whole ballgame.
#   * arithmetic is done in fp32 (`opmath_t`) like ATen's own bf16 kernels.
#   * the REDUCTION is left to ATen (`x.norm(2, 1, keepdim=True)`) and is not
#     re-implemented, because a hand-written reduction would have a different
#     summation order and a 1-ULP fp32 difference there rounds to a different
#     bf16 norm for ~1e-5 of elements -- and this workload amplifies a single
#     bf16 flip chaotically (rope_apply is exact to 1 ULP standalone yet eats
#     86% of the e2e budget).  Only the POINTWISE tail is fused; the reduce
#     keeps its own kernel.
#   * `+ self.bias` is kept as a literal `+ 0.0f` in fp32 rather than dropped,
#     so even the signed-zero behaviour matches.
#
# VERIFIED: oracle `max|d| = mean|d| = 0.000e+00` at BOTH 1 and 10 steps with
# this on -- i.e. the fusion spends ZERO of the remaining numerics budget.  See
# the report.  If a future torch changes ATen's rounding this shows up as a
# non-zero oracle immediately; FASTWAM_VAEFUSE=0 is the escape hatch.
#
# SUB-SWITCHES (all default ON, each independently killable):
#   FASTWAM_VAEFUSE       master.  0 = stock eager RMS_norm / AvgDown3D.
#   FASTWAM_VAEFUSE_SILU  also absorb the `nn.SiLU()` that follows 21 of the 22
#                         norms, using ATen's own formula `x/(1+exp(-x))` in
#                         fp32.  Deletes another 21 dispatches and 21 passes.
#   FASTWAM_VAEFUSE_ADD   absorb the ResidualBlock / Down_ResidualBlock skip
#                         connection into the producing conv's peeled bias add.
#   FASTWAM_VAEFUSE_AVG   short-circuit the group=1, factor=1 `AvgDown3D` in
#                         the LAST Down_ResidualBlock, which is `pad(0) ->
#                         permute().contiguous() -> mean(dim of size 1)` == a
#                         three-kernel identity.
#
# EPISODE STATE: none.  This block caches nothing on `self` except a boolean
# capability flag and the module-level Triton handle; no tensor, no data_ptr,
# nothing that varies per call.  COMMANDMENT rule 6 has nothing to invalidate.
# ---------------------------------------------------------------------------
_FW_VAEFUSE = not _fw_off(_os.environ.get("FASTWAM_VAEFUSE", "1"))
_FW_VAEFUSE_SILU = _FW_VAEFUSE and not _fw_off(_os.environ.get("FASTWAM_VAEFUSE_SILU", "1"))
_FW_VAEFUSE_ADD = _FW_VAEFUSE and not _fw_off(_os.environ.get("FASTWAM_VAEFUSE_ADD", "1"))
_FW_VAEFUSE_AVG = _FW_VAEFUSE and not _fw_off(_os.environ.get("FASTWAM_VAEFUSE_AVG", "1"))

_FW_TL = None
if _FW_VAEFUSE:
    try:
        import triton as _fw_triton
        import triton.language as _fw_tl

        @_fw_triton.jit
        def _fw_rmsnorm_tail_k(X, D, G, Y, NUMEL, SCALE,
                               C: _fw_tl.constexpr, SILU: _fw_tl.constexpr,
                               BLOCK: _fw_tl.constexpr):
            """One kernel for  ((x / max(d,eps)) * scale) * gamma + 0  [+ silu].

            Rounds to the storage dtype after EVERY op, exactly where ATen's
            TensorIterator does, so the result is bit-identical to the eager
            five-kernel chain.  `C` is a constexpr so `// C` and `% C` become
            multiply-shift, not integer division.
            """
            pid = _fw_tl.program_id(0)
            offs = pid * BLOCK + _fw_tl.arange(0, BLOCK)
            mask = offs < NUMEL
            dt = Y.dtype.element_ty
            row = offs // C
            col = offs % C
            x = _fw_tl.load(X + offs, mask=mask, other=0).to(_fw_tl.float32)
            d = _fw_tl.load(D + row, mask=mask, other=1).to(_fw_tl.float32)
            g = _fw_tl.load(G + col, mask=mask, other=0).to(_fw_tl.float32)
            # clamp_min(eps) -- ATen writes the clamped value back as `dt`
            d = _fw_tl.maximum(d, 1e-12).to(dt).to(_fw_tl.float32)
            y = (x / d).to(dt).to(_fw_tl.float32)
            y = (y * SCALE).to(dt).to(_fw_tl.float32)
            y = (y * g).to(dt).to(_fw_tl.float32)
            y = (y + 0.0).to(dt).to(_fw_tl.float32)
            if SILU:
                # ATen silu: opmath x / (1 + exp(-x)), then round once.
                y = (y / (1.0 + _fw_tl.exp(-y))).to(dt).to(_fw_tl.float32)
            _fw_tl.store(Y + offs, y.to(dt), mask=mask)

        @_fw_triton.jit
        def _fw_biasadd_res_k(X, B, H, Y, NUMEL, C: _fw_tl.constexpr,
                              BLOCK: _fw_tl.constexpr):
            """`(x + bias[c]) + h`, both rounds kept, one dispatch."""
            pid = _fw_tl.program_id(0)
            offs = pid * BLOCK + _fw_tl.arange(0, BLOCK)
            mask = offs < NUMEL
            dt = Y.dtype.element_ty
            col = offs % C
            x = _fw_tl.load(X + offs, mask=mask, other=0).to(_fw_tl.float32)
            b = _fw_tl.load(B + col, mask=mask, other=0).to(_fw_tl.float32)
            h = _fw_tl.load(H + offs, mask=mask, other=0).to(_fw_tl.float32)
            y = (x + b).to(dt).to(_fw_tl.float32)
            y = (y + h).to(dt)
            _fw_tl.store(Y + offs, y, mask=mask)


        @_fw_triton.jit
        def _fw_addchain_k(X, B, HY, HB, HH, K, Y, NUMEL,
                           C: _fw_tl.constexpr, HAS_B: _fw_tl.constexpr,
                           HAS_H: _fw_tl.constexpr, HAS_HB: _fw_tl.constexpr,
                           HAS_HH: _fw_tl.constexpr, HAS_K: _fw_tl.constexpr,
                           BLOCK: _fw_tl.constexpr):
            """r5_d2 -- `(((x + bias) + ((hy + hbias) + hh)) + k)`, ONE dispatch.

            Generalises `_fw_biasadd_res_k`: the `h` a residual block adds back
            is often ITSELF the previous block's `(conv + bias) + skip`, whose
            only consumers are this add and the next norm -- both of which
            already stream the same bytes.  Carrying it as the un-materialised
            triple `(hy, hbias, hh)` deletes a launch AND a full write+read of
            the intermediate.  `k` picks up a trailing shortcut add
            (`Down_ResidualBlock`'s factor==1 avg shortcut) for free.
            Every round back to the storage dtype sits exactly where the
            shipped chain of `aten::add` / `_fw_biasadd_res_k` puts it, so this
            is BIT-EXACT.
            """
            pid = _fw_tl.program_id(0)
            offs = pid * BLOCK + _fw_tl.arange(0, BLOCK)
            mask = offs < NUMEL
            dt = Y.dtype.element_ty
            col = offs % C
            y = _fw_tl.load(X + offs, mask=mask, other=0).to(_fw_tl.float32)
            if HAS_B:
                b = _fw_tl.load(B + col, mask=mask, other=0).to(_fw_tl.float32)
                y = (y + b).to(dt).to(_fw_tl.float32)
            if HAS_H:
                h = _fw_tl.load(HY + offs, mask=mask,
                                other=0).to(_fw_tl.float32)
                if HAS_HB:
                    hb = _fw_tl.load(HB + col, mask=mask,
                                     other=0).to(_fw_tl.float32)
                    h = (h + hb).to(dt).to(_fw_tl.float32)
                if HAS_HH:
                    hh = _fw_tl.load(HH + offs, mask=mask,
                                     other=0).to(_fw_tl.float32)
                    h = (h + hh).to(dt).to(_fw_tl.float32)
                y = (y + h).to(dt).to(_fw_tl.float32)
            if HAS_K:
                kk = _fw_tl.load(K + offs, mask=mask,
                                 other=0).to(_fw_tl.float32)
                y = (y + kk).to(dt).to(_fw_tl.float32)
            _fw_tl.store(Y + offs, y.to(dt), mask=mask)

        @_fw_triton.jit
        def _fw_rmsnorm_fullr_k(X, G, Y, BI, R, SCALE, ROWS,
                                C: _fw_tl.constexpr, BC: _fw_tl.constexpr,
                                SILU: _fw_tl.constexpr,
                                ROWS_PER: _fw_tl.constexpr):
            """r5_d2 -- `_fw_rmsnorm_full_k` with the producer's `+ bias + res`
            absorbed.  Byte-for-byte the shipped kernel plus the two rounds
            `_fw_biasadd_res_k` performs, in the same order, so the whole
            (biasadd_res -> rmsnorm_full) pair collapses to ONE launch and one
            pass BIT-EXACTLY."""
            pid = _fw_tl.program_id(0)
            cols = _fw_tl.arange(0, BC)[None, :]
            rows = pid * ROWS_PER + _fw_tl.arange(0, ROWS_PER)[:, None]
            dt = Y.dtype.element_ty
            m = (cols < C) & (rows < ROWS)
            off = rows * C + cols
            g = _fw_tl.load(G + cols, mask=cols < C, other=0).to(_fw_tl.float32)
            x = _fw_tl.load(X + off, mask=m, other=0).to(_fw_tl.float32)
            bb = _fw_tl.load(BI + cols, mask=cols < C,
                             other=0).to(_fw_tl.float32)
            x = (x + bb).to(dt).to(_fw_tl.float32)
            rr = _fw_tl.load(R + off, mask=m, other=0).to(_fw_tl.float32)
            x = (x + rr).to(dt).to(_fw_tl.float32)
            ss = _fw_tl.sum(x * x, axis=1)[:, None]
            d = _fw_tl.sqrt(ss).to(dt).to(_fw_tl.float32)
            d = _fw_tl.maximum(d, 1e-12).to(dt).to(_fw_tl.float32)
            y = (x / d).to(dt).to(_fw_tl.float32)
            y = (y * SCALE).to(dt).to(_fw_tl.float32)
            y = (y * g).to(dt).to(_fw_tl.float32)
            y = (y + 0.0).to(dt).to(_fw_tl.float32)
            if SILU:
                y = (y / (1.0 + _fw_tl.exp(-y))).to(dt).to(_fw_tl.float32)
            _fw_tl.store(Y + off, y.to(dt), mask=m)


        @_fw_triton.jit
        def _fw_i2c_gather_k(X, U, NUMEL,
                             H: _fw_tl.constexpr, W: _fw_tl.constexpr,
                             C: _fw_tl.constexpr, BLOCK: _fw_tl.constexpr):
            """[H*W, 9*C] im2col of a channels_last [1,C,H,W], in ONE pass.

            `U[p, (kh*3 + kw)*C + c] = x[h + kh - 1, w + kw - 1, c]`, zero
            outside the frame -- i.e. a 3x3 stride-1 pad-1 patch matrix in the
            (kh, kw, c) order `_fw_prepare_conv2d` materializes the weight in.
            Pure data movement: every value is copied, never arithmetic'd, so
            the gather itself is exact.  `c` is the innermost axis on BOTH
            sides, so a contiguous span of `U` reads a contiguous span of `x`.
            """
            pid = _fw_tl.program_id(0)
            offs = pid * BLOCK + _fw_tl.arange(0, BLOCK)
            mask = offs < NUMEL
            K: _fw_tl.constexpr = 9 * C
            p = offs // K
            k = offs % K
            kh = k // (3 * C)
            r = k % (3 * C)
            kw = r // C
            c = r % C
            h = p // W
            w = p % W
            ih = h + kh - 1
            iw = w + kw - 1
            ok = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            v = _fw_tl.load(X + (ih * W + iw) * C + c, mask=ok, other=0)
            _fw_tl.store(U + offs, v, mask=mask)

        @_fw_triton.jit
        def _fw_patch_nhwc_k(X, Y, NUMEL,
                             C: _fw_tl.constexpr, H: _fw_tl.constexpr,
                             W: _fw_tl.constexpr, P: _fw_tl.constexpr,
                             BLOCK: _fw_tl.constexpr):
            """`patchify(x, P)` AND the channels_last_3d relayout, one pass.

            r3_d0.  The captured encode region is built with the `cudagraphs`
            backend, which fuses NOTHING: `rearrange` + `.contiguous(
            channels_last_3d)` are three separate ATen copy kernels at the
            ~4 us in-graph floor for a 0.6 MB tensor.  This is the same index
            algebra with one gather.  PURE DATA MOVEMENT -- every value is
            copied, never arithmetic'd, so it is bit-exact by construction.

            X : [C, H*P, W*P] contiguous (b == f == 1).
            Y : [H*W, C*P*P] channel-innermost == channels_last_3d of
                [1, C*P*P, 1, H, W].
            Mapping (einops `b c f (h q) (w r) -> b (c r q) f h w`):
                oc = c*P*P + r*P + q ,  src_h = h*P + q ,  src_w = w*P + r
            """
            pid = _fw_tl.program_id(0)
            offs = pid * BLOCK + _fw_tl.arange(0, BLOCK)
            mask = offs < NUMEL
            OC: _fw_tl.constexpr = C * P * P
            oc = offs % OC
            sp = offs // OC
            w = sp % W
            h = sp // W
            c = oc // (P * P)
            rem = oc % (P * P)
            r = rem // P
            q = rem % P
            src = (c * (H * P) + h * P + q) * (W * P) + w * P + r
            v = _fw_tl.load(X + src, mask=mask, other=0)
            _fw_tl.store(Y + offs, v, mask=mask)

        @_fw_triton.jit
        def _fw_vaetail_k(V, S0, S1, O, NUMEL,
                          C: _fw_tl.constexpr, CIN: _fw_tl.constexpr,
                          HW: _fw_tl.constexpr, BLOCK: _fw_tl.constexpr):
            """`((v[:C] - s0) * s1).contiguous()` of the encode head, one pass.

            r3_d0.  Replaces `chunk` + `sub` + `mul` + `contiguous` -- three
            ATen kernels at the in-graph launch floor on a 37 KB tensor.  The
            two rounds are kept exactly where ATen's TensorIterator puts them
            (round back to the storage dtype after EVERY op), so this is
            bit-exact with the eager pair, not an approximation.

            V : [HW, CIN] channel-innermost (channels_last_3d conv output).
            O : [C, HW] contiguous (the plain NCDHW `mu` the caller returns).
            """
            pid = _fw_tl.program_id(0)
            offs = pid * BLOCK + _fw_tl.arange(0, BLOCK)
            mask = offs < NUMEL
            dt = O.dtype.element_ty
            c = offs // HW
            sp = offs % HW
            v = _fw_tl.load(V + sp * CIN + c, mask=mask,
                            other=0).to(_fw_tl.float32)
            a = _fw_tl.load(S0 + c, mask=mask, other=0).to(_fw_tl.float32)
            b = _fw_tl.load(S1 + c, mask=mask, other=0).to(_fw_tl.float32)
            y = (v - a).to(dt).to(_fw_tl.float32)
            y = (y * b).to(dt)
            _fw_tl.store(O + offs, y, mask=mask)

        @_fw_triton.jit
        def _fw_rmsnorm_full_k(X, G, Y, BI, SCALE, ROWS,
                               C: _fw_tl.constexpr, BC: _fw_tl.constexpr,
                               SILU: _fw_tl.constexpr, HAS_B: _fw_tl.constexpr,
                               ROWS_PER: _fw_tl.constexpr):
            """ONE kernel for the whole RMS_norm: reduce AND pointwise tail.

            The shipped pair is `aten::linalg_vector_norm` (0.1064 ms x 22)
            followed by `_fw_rmsnorm_tail_k` (0.1073 ms x 22) -- two launches
            and TWO full reads of the activation to produce one output.  The
            channel axis is the INNERMOST axis of the channels_last domain, so
            a row (one spatial position, C values) fits in registers and the
            reduce can be done where the tail already has the data.

            The pointwise arithmetic below is byte-for-byte the tail kernel's
            (round back to the storage dtype after every op, exactly where
            ATen's TensorIterator rounds).  The ONLY thing that changes is the
            summation ORDER inside the L2 norm, and its fp32 result is rounded
            to bf16 before use -- the same bf16 `d` the two-kernel path stores
            -- so a 1-ULP fp32 difference has to straddle a bf16 rounding
            boundary to be visible at all.  Validated on the oracle mean|d| and
            on the 14-episode closed-loop rollout, not asserted bit-exact.
            """
            pid = _fw_tl.program_id(0)
            cols = _fw_tl.arange(0, BC)[None, :]
            rows = pid * ROWS_PER + _fw_tl.arange(0, ROWS_PER)[:, None]
            dt = Y.dtype.element_ty
            m = (cols < C) & (rows < ROWS)
            off = rows * C + cols
            g = _fw_tl.load(G + cols, mask=cols < C, other=0).to(_fw_tl.float32)
            x = _fw_tl.load(X + off, mask=m, other=0).to(_fw_tl.float32)
            if HAS_B:
                # the producing conv's bias, added HERE instead of in its own
                # `aten::add` pass (same round, one dispatch and one full pass
                # over the activation less).
                bb = _fw_tl.load(BI + cols, mask=cols < C,
                                 other=0).to(_fw_tl.float32)
                x = (x + bb).to(dt).to(_fw_tl.float32)
            ss = _fw_tl.sum(x * x, axis=1)[:, None]
            d = _fw_tl.sqrt(ss).to(dt).to(_fw_tl.float32)
            d = _fw_tl.maximum(d, 1e-12).to(dt).to(_fw_tl.float32)
            y = (x / d).to(dt).to(_fw_tl.float32)
            y = (y * SCALE).to(dt).to(_fw_tl.float32)
            y = (y * g).to(dt).to(_fw_tl.float32)
            y = (y + 0.0).to(dt).to(_fw_tl.float32)
            if SILU:
                y = (y / (1.0 + _fw_tl.exp(-y))).to(dt).to(_fw_tl.float32)
            _fw_tl.store(Y + off, y.to(dt), mask=m)


        @_fw_triton.jit
        def _fw_rmsnorm_full2_k(X, G, Y, BI, SCALE, ROWS,
                                C: _fw_tl.constexpr, BCC: _fw_tl.constexpr,
                                NCH: _fw_tl.constexpr, SILU: _fw_tl.constexpr,
                                HAS_B: _fw_tl.constexpr,
                                ROWS_PER: _fw_tl.constexpr):
            """r4_d2 -- the SAME RMS_norm, with the padded lanes deleted.

            `_fw_rmsnorm_full_k` tiles the channel axis at `BC = next_pow2(C)`.
            Every C this VAE uses is `5 * 2^k` (160 / 320 / 640), so BC is
            always `8 * 2^k`: **37.5% of the vector lanes in every one of the
            22 launches are masked-off padding**, and the mask costs full ALU
            issue on the div / sqrt / silu / bf16-round chain, which the
            ablation shows is 7.3 us of the big launch's 10.9 us (a pure copy
            of the same bytes is 3.6 us).  Here the row is walked as
            `NCH = C / BCC` chunks of `BCC = ` the largest power of two that
            DIVIDES C, so every lane is live.  The loaded chunks are kept in
            registers across the reduce, so the activation is still read once.

            NUMERICS: every arithmetic op, and every round back to the storage
            dtype, is byte-for-byte the one-tile kernel's.  The only change is
            the summation order of the L2 norm (`NCH` partial `tl.sum`s
            accumulated in fp32, instead of one `tl.sum` over BC lanes of which
            the top 37.5% are exact zeros) -- the same class of change r3_d0
            already validated, and its fp32 result is rounded to bf16 before
            use.  Measured over the encode's own shapes: 96 of 4014080 output
            elements differ on ONE of the ten shape/bias combinations and zero
            on the other nine.  Kill-switch FASTWAM_VAENORM2=0.
            """
            pid = _fw_tl.program_id(0)
            cols = _fw_tl.arange(0, BCC)[None, :]
            rows = pid * ROWS_PER + _fw_tl.arange(0, ROWS_PER)[:, None]
            dt = Y.dtype.element_ty
            rm = rows < ROWS
            base = rows * C
            xs = []
            ss = _fw_tl.zeros([ROWS_PER, 1], dtype=_fw_tl.float32)
            for j in _fw_tl.static_range(NCH):
                o = base + (j * BCC + cols)
                x = _fw_tl.load(X + o, mask=rm, other=0).to(_fw_tl.float32)
                if HAS_B:
                    bb = _fw_tl.load(BI + j * BCC + cols).to(_fw_tl.float32)
                    x = (x + bb).to(dt).to(_fw_tl.float32)
                xs = xs + [x]
                ss += _fw_tl.sum(x * x, axis=1)[:, None]
            d = _fw_tl.sqrt(ss).to(dt).to(_fw_tl.float32)
            d = _fw_tl.maximum(d, 1e-12).to(dt).to(_fw_tl.float32)
            for j in _fw_tl.static_range(NCH):
                o = base + (j * BCC + cols)
                g = _fw_tl.load(G + j * BCC + cols).to(_fw_tl.float32)
                y = (xs[j] / d).to(dt).to(_fw_tl.float32)
                y = (y * SCALE).to(dt).to(_fw_tl.float32)
                y = (y * g).to(dt).to(_fw_tl.float32)
                y = (y + 0.0).to(dt).to(_fw_tl.float32)
                if SILU:
                    y = (y / (1.0 + _fw_tl.exp(-y))).to(dt).to(_fw_tl.float32)
                _fw_tl.store(Y + o, y.to(dt), mask=rm)


        @_fw_triton.jit
        def _fw_avgdown_add_k(M, X, Y, NUMEL, W, C: _fw_tl.constexpr,
                              O: _fw_tl.constexpr, W2: _fw_tl.constexpr,
                              FT: _fw_tl.constexpr, BLOCK: _fw_tl.constexpr):
            """`main + AvgDown3D(x)` for the T == 1 encode path, ONE kernel.

            See the long note on `_FW_VAEAVG2`.  The shortcut is exactly a 2x2
            spatial average (channel-doubled, with the even output channels
            identically zero, when factor_t == 2), and the `mean` is written
            the way ATen's own reduce writes it: accumulate in fp32, scale by
            1/factor, round to the storage dtype ONCE.
            """
            pid = _fw_tl.program_id(0)
            offs = pid * BLOCK + _fw_tl.arange(0, BLOCK)
            mask = offs < NUMEL
            dt = Y.dtype.element_ty
            o = offs % O
            p = offs // O
            w2 = p % W2
            h2 = p // W2
            if FT == 2:
                c = o // 2
                k = o % 2
            else:
                c = o
                k = o - o + 1
            base = (h2 * 2 * W + w2 * 2) * C + c
            a0 = _fw_tl.load(X + base, mask=mask, other=0).to(_fw_tl.float32)
            a1 = _fw_tl.load(X + base + C, mask=mask, other=0).to(_fw_tl.float32)
            a2 = _fw_tl.load(X + base + W * C, mask=mask,
                             other=0).to(_fw_tl.float32)
            a3 = _fw_tl.load(X + base + W * C + C, mask=mask,
                             other=0).to(_fw_tl.float32)
            v = (((a0 + a1) + a2) + a3) * 0.25
            v = v.to(dt).to(_fw_tl.float32)
            if FT == 2:
                v = _fw_tl.where(k == 1, v, 0.0)
            m = _fw_tl.load(M + offs, mask=mask, other=0).to(_fw_tl.float32)
            _fw_tl.store(Y + offs, (m + v).to(dt), mask=mask)

        _FW_TL = _fw_triton
    except Exception:  # pragma: no cover
        _FW_TL = None


def _fw_cl_flat(x):
    """Flat view of `x` in channel-innermost order, or None if it is not so laid out."""
    nd = x.dim()
    if nd < 2:
        return None
    xp = x.permute([0] + list(range(2, nd)) + [1])
    if not xp.is_contiguous():
        return None
    return xp.reshape(-1)


def _fw_rmsnorm_tail_impl(x, d, g, scale, silu):
    y = torch.empty_like(x)
    n = x.numel()
    blk = 1024
    _fw_rmsnorm_tail_k[((n + blk - 1) // blk,)](
        _fw_cl_flat(x), d.reshape(-1), g.reshape(-1), _fw_cl_flat(y), n,
        float(scale), C=int(x.shape[1]), SILU=bool(silu), BLOCK=blk,
        num_warps=4)
    return y


def _fw_biasadd_res_impl(x, b, h):
    y = torch.empty_like(x)
    n = x.numel()
    blk = 1024
    _fw_biasadd_res_k[((n + blk - 1) // blk,)](
        _fw_cl_flat(x), b.reshape(-1), _fw_cl_flat(h), _fw_cl_flat(y), n,
        C=int(x.shape[1]), BLOCK=blk, num_warps=4)
    return y


def _fw_addchain_impl(x, b, hy, hb, hh, k):
    y = torch.empty_like(x)
    n = x.numel()
    blk = 1024
    _fw_addchain_k[((n + blk - 1) // blk,)](
        _fw_cl_flat(x), None if b is None else b.reshape(-1),
        None if hy is None else _fw_cl_flat(hy),
        None if hb is None else hb.reshape(-1),
        None if hh is None else _fw_cl_flat(hh),
        None if k is None else _fw_cl_flat(k),
        _fw_cl_flat(y), n, C=int(x.shape[1]),
        HAS_B=b is not None, HAS_H=hy is not None, HAS_HB=hb is not None,
        HAS_HH=hh is not None, HAS_K=k is not None, BLOCK=blk, num_warps=4)
    return y


def _fw_rmsnorm_fullr_impl(x, g, scale, silu, bi, r):
    y = torch.empty_like(x)
    c = int(x.shape[1])
    rows = x.numel() // c
    bc = _fw_next_pow2(c)
    tile = int(_os.environ.get("FASTWAM_VAERES_TILE", "2048"))
    rows_per = max(1, min(_fw_next_pow2(rows), tile // bc))
    nw = int(_os.environ.get("FASTWAM_VAERES_W", "4"))
    _fw_rmsnorm_fullr_k[((rows + rows_per - 1) // rows_per,)](
        _fw_cl_flat(x), g.reshape(-1), _fw_cl_flat(y), bi.reshape(-1),
        _fw_cl_flat(r), float(scale), rows,
        C=c, BC=bc, SILU=bool(silu), ROWS_PER=rows_per, num_warps=nw)
    return y


_FW_VAENORM1 = _os.environ.get("FASTWAM_VAENORM1", "1") not in (
    "0", "false", "no", "off")


def _fw_next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


_FW_GATH_BLK = int(_os.environ.get("FASTWAM_VAEGATH_BLK", "1024"))
_FW_GATH_W = int(_os.environ.get("FASTWAM_VAEGATH_W", "4"))


def _fw_i2c_gather_impl(x, ph, pw):
    n, c, h, w = x.shape
    u = x.new_empty((h * w, 9 * c))
    numel = u.numel()
    blk = _FW_GATH_BLK
    _fw_i2c_gather_k[((numel + blk - 1) // blk,)](
        _fw_cl_flat(x), u, numel, H=int(h), W=int(w), C=int(c), BLOCK=blk,
        num_warps=_FW_GATH_W)
    return u


# ---------------------------------------------------------------------------
# r3_d0 -- the two ENDS of the captured encode region.
#
# The region is captured with backend="cudagraphs" (aot_eager + cudagraph), so
# nothing inductor-fuses inside it: every ATen pointwise op is its own kernel
# and every kernel pays the ~4.2 us in-graph floor this tree measures.  The
# head (`patchify` + `.contiguous(channels_last_3d)`) and the tail (`chunk` +
# `-s0` + `*s1` + `.contiguous()`) are 7 such kernels for 0.64 MB of traffic.
# Both are collapsed to ONE Triton kernel each.  Kill-switches, default ON:
#   FASTWAM_VAEHEAD=0   restore rearrange + contiguous
#   FASTWAM_VAETAIL=0   restore chunk + sub + mul + contiguous
# NUMERICS: the head is pure data movement (exact).  The tail keeps both bf16
# rounds where ATen puts them, so it is bit-exact too -- neither spends any
# numerics budget.  EPISODE STATE: none; both are pure functions of their args.
# ---------------------------------------------------------------------------
_FW_VAEHEAD = _os.environ.get("FASTWAM_VAEHEAD", "1") not in (
    "0", "false", "no", "off")
_FW_VAETAIL = _os.environ.get("FASTWAM_VAETAIL", "1") not in (
    "0", "false", "no", "off")
_FW_VAEXBUF = _os.environ.get("FASTWAM_VAEXBUF", "1") not in (
    "0", "false", "no", "off")


def _fw_patch_nhwc_out(x, p):
    c, hp, wp = int(x.shape[1]), int(x.shape[3]), int(x.shape[4])
    return torch.empty((1, c * p * p, 1, hp // p, wp // p),
                       dtype=x.dtype, device=x.device,
                       memory_format=torch.channels_last_3d)


def _fw_patch_nhwc_impl(x, p):
    y = _fw_patch_nhwc_out(x, p)
    numel = y.numel()
    blk = 1024
    _fw_patch_nhwc_k[((numel + blk - 1) // blk,)](
        x.reshape(-1), _fw_cl_flat(y), numel,
        C=int(x.shape[1]), H=int(x.shape[3]) // p, W=int(x.shape[4]) // p,
        P=int(p), BLOCK=blk, num_warps=4)
    return y


def _fw_patch_nhwc_meta(x, p):
    return _fw_patch_nhwc_out(x, p)


def _fw_vaetail_out(v, c):
    return torch.empty((1, c, 1, int(v.shape[3]), int(v.shape[4])),
                       dtype=v.dtype, device=v.device)


def _fw_vaetail_impl(v, s0, s1, c):
    o = _fw_vaetail_out(v, c)
    hw = int(v.shape[3]) * int(v.shape[4])
    numel = c * hw
    blk = 1024
    _fw_vaetail_k[((numel + blk - 1) // blk,)](
        _fw_cl_flat(v), s0.reshape(-1), s1.reshape(-1), o.reshape(-1), numel,
        C=int(c), CIN=int(v.shape[1]), HW=int(hw), BLOCK=blk, num_warps=4)
    return o


def _fw_vaetail_meta(v, s0, s1, c):
    return _fw_vaetail_out(v, c)


def _fw_i2c_gather_meta(x, ph, pw):
    return x.new_empty((int(x.shape[2]) * int(x.shape[3]), 9 * int(x.shape[1])))


def _fw_gath_conv(x, wgat):
    """3x3 s1 p1 NHWC conv as ONE Triton gather + ONE GEMM.  None => not ours."""
    if not (_FW_VAEGATH and _FW_GATH_OP and wgat is not None):
        return None

    n, c, h, w = x.shape
    co = wgat.shape[0]
    if n != 1 or (h, w, c, co) not in _FW_GATH_SHAPES:
        return None
    if _fw_cl_flat(x) is None:
        return None
    # NO `print` HERE.  r2_d2: the round-1 version of this function carried a
    # one-shot `print(f"[fw-vaegath] engaged ...")` diagnostic guarded by a dict
    # flag.  Whether that costs anything depends on WHICH call runs first: if
    # the flag is still unset when `torch.compile(fullgraph=True)` traces the
    # encode, dynamo hits `Failed to trace builtin operator print` (gb0059),
    # `_fw_build_encode_graph`'s caller falls back to eager FOR THE WHOLE
    # PROCESS, and `vae_encode` goes from 2 launch-API calls to 102 with 1.73 ms
    # of newly EXPOSED gpu_idle.  Measured on this tree: paired A/B geomean
    # 10.02 -> 12.24 ms (+22%), i.e. the diagnostic was 40x more expensive than
    # the optimisation it was announcing.  The engaged/not-engaged signal is
    # already visible in the profile (`_fw_i2c_gather_k` x9 replacing
    # `igemm_fwd_gtcx35_...` x8), so it is simply deleted rather than re-guarded.
    u = torch.ops.fastwam_vae.i2c_gather.default(x, 1, 1)
    y = F.linear(u, wgat)                      # [H*W, Cout], NHWC byte order
    return y.view(n, h, w, co).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# r4_d2 -- WHICH RMS_norm KERNEL, AND WITH WHICH TILE.  (FASTWAM_VAENORM2=0)
#
# `_fw_rmsnorm_full_k` pads the channel axis to next_pow2(C).  Every C in this
# encode is 5 * 2^k, so 37.5% of its vector lanes are dead in ALL 22 launches,
# and the per-element chain (one fp32 divide, one exp, six bf16 rounds) is
# expensive enough that those dead lanes are real time: in-graph, on this box,
# the 8 MB launch costs 11.0 us where a bare copy of the same bytes costs 3.6.
# `_fw_rmsnorm_full2_k` walks the row in NCH = C/BCC fully-live chunks.
#
# The (ROWS_PER, num_warps) below are the winners of a JOINT 6x4 sweep (per the
# round-3 finding that a one-at-a-time sweep closes live axes), measured on
# 22-launch cudagraph replays at the encode's OWN shapes -- not on eager
# launches, which are ~12.6 us of host overhead each and hide the kernel
# entirely.  Table is keyed (C, rows); a shape that is NOT in it keeps the
# shipped one-tile kernel unchanged, which is also what the (392, 640) rows do
# deliberately: at 0.5 MB the chunked form is slower (2.77 vs 2.57 us), the
# launch floor dominates and there is nothing to win.
#
#   (C, rows)        shipped   chunked  (ROWS_PER, num_warps)
#   (160, 25088)  11.0/14.9 -> 8.6/8.9        (4, 1)
#   (160,  6272)       4.5  -> 3.9            (8, 1)
#   (320,  6272)   6.5/7.8  -> 5.7/5.9        (1, 1)
#   (320,  1568)       3.1  -> 3.1            (1, 1)
#   (640,  1568)   4.8/5.1  -> 4.0/4.2        (1, 1)
#   (640,   392)       ----  keeps _fw_rmsnorm_full_k ----
# Total over the region's 22 launches: 123.2 -> 98.0 us predicted.
# ---------------------------------------------------------------------------
_FW_VAENORM2 = not _fw_off(_os.environ.get("FASTWAM_VAENORM2", "1"))

_FW_NORM2_CFG = {
    (160, 25088): (4, 1),
    (160, 6272): (8, 1),
    (320, 6272): (1, 1),
    (320, 1568): (1, 1),
    (640, 1568): (1, 1),
}


def _fw_lowest_pow2_div(c):
    """Largest power of two that divides `c`."""
    p = 1
    while c % (p * 2) == 0:
        p *= 2
    return p


def _fw_rmsnorm_full_impl(x, g, scale, silu, bi=None):
    y = torch.empty_like(x)
    c = int(x.shape[1])
    rows = x.numel() // c
    cfg = _FW_NORM2_CFG.get((c, rows)) if _FW_VAENORM2 else None
    if cfg is not None:
        bcc = _fw_lowest_pow2_div(c)
        rows_per = min(cfg[0], _fw_next_pow2(rows))
        _fw_rmsnorm_full2_k[((rows + rows_per - 1) // rows_per,)](
            _fw_cl_flat(x), g.reshape(-1), _fw_cl_flat(y),
            None if bi is None else bi.reshape(-1), float(scale), rows,
            C=c, BCC=bcc, NCH=c // bcc, SILU=bool(silu),
            HAS_B=bi is not None, ROWS_PER=rows_per, num_warps=cfg[1])
        return y
    bc = _fw_next_pow2(c)
    tile = int(_os.environ.get("FASTWAM_VAENORM1_TILE", "2048"))
    rows_per = max(1, min(_fw_next_pow2(rows), tile // bc))
    nw = int(_os.environ.get("FASTWAM_VAENORM1_W", "4"))
    _fw_rmsnorm_full_k[((rows + rows_per - 1) // rows_per,)](
        _fw_cl_flat(x), g.reshape(-1), _fw_cl_flat(y),
        None if bi is None else bi.reshape(-1), float(scale), rows,
        C=c, BC=bc, SILU=bool(silu), HAS_B=bi is not None,
        ROWS_PER=rows_per, num_warps=nw)
    return y



# ---------------------------------------------------------------------------
# THE Down_ResidualBlock SHORTCUT IS A 2x2 AVERAGE POOL  (FASTWAM_VAEAVG2=0)
#
# MEASURED, `__torch_dispatch__` log of one eager encode (every relayout the
# region performs, with its strides) -- the three `AvgDown3D` shortcuts owned
# EIGHT of the region's ten copies and were its whole `aten::copy_` line
# (0.0784 ms x 9 per call in the eager attribution, plus `aten::mean` 0.0181 x3
# and `aten::fill_` 0.0077 x3):
#
#   clone (1,160,1,2,2,1,56,112)   8.0 MB   <- stage 0 permute().contiguous()
#   pad   (1,160,1,56,112)->T=2    2.0->4.0 MB
#   clone (1,160,2,2,2,1,28,56)    4.0 MB   <- stage 1
#   pad   (1,320,1,28,56)->T=2     1.0->2.0 MB
#   clone (1,320,2,2,2,1,14,28)    2.0 MB   <- stage 2
#
# STRUCTURAL FACT (exact, not an approximation).  `dims = [160,160,320,640,640]`
# and every encode-path `AvgDown3D` has `factor_s = 2`, `group_size = 4`:
#
#   factor_t == 1 (stage 0):  e = c*4 + sh*2 + sw, o = e // 4 = c
#       -> out[c] is the mean of the 2x2 spatial block.  O == C.
#   factor_t == 2 (stages 1,2):  T == 1 so `pad_t == 1` and the pad puts a plane
#       of EXACT ZEROS in FRONT: xpad = [0, x0].  Then
#       e = c*8 + t*4 + sh*2 + sw and o = e // 4 = 2*c + t, so
#           out[2c]   = mean of four zeros            = 0
#           out[2c+1] = mean of the 2x2 spatial block
#       -- i.e. HALF of every one of those tensors is identically zero, and the
#       pad exists only to manufacture the zeros.  O == 2*C.
#
# So all three shortcuts are one 2x2 average pool (channel-interleaved with a
# zero channel when factor_t == 2), and the `+ x` that follows rides along in
# the same kernel: `_fw_avgdown_add_k` reads the block input ONCE, reads the
# main-path activation once and writes the sum once.  Per call it replaces
# 11 kernels (3 pads, 3 clones, 3 means, 3 adds -- one is the FUSE_AVG identity
# already deleted) with 3, and ~46 MB of HBM traffic with ~10 MB.
#
# MEASURED (`run_profile.sh`, `vae_encode`, on top of FASTWAM_VAESTATIC):
#   wall 1.2483 -> 1.1547 ms   gpu_busy 1.2497 -> 1.1588   kernels 134 -> 124
#   `_fw_avgdown_add_k` costs 0.0135 ms x3 in total.
#
# NUMERICS: the mean is written the way ATen's own reduce writes it -- fp32
# accumulate in the SAME (sh,sw) order the strided reduce walks, one multiply
# by 1/4 (exact), one round to bf16 -- and the `+ x` keeps its own round.  Not
# asserted bit-exact (a 4-element fp32 sum can be re-associated by ATen), so it
# is measured: oracle steps=1 max|d| 6.250e-02 mean|d| 8.143e-03 (36% of budget,
# BELOW the shipped tree's own 9.035e-03) and the 14-episode rollout.
#
# EPISODE STATE: none -- no tensor is cached, nothing is keyed on data_ptr.
# ---------------------------------------------------------------------------
_FW_VAEAVG2 = _FW_VAEFUSE and not _fw_off(_os.environ.get("FASTWAM_VAEAVG2", "1"))


# ---------------------------------------------------------------------------
# THREE DEAD OPS THE CAPTURED GRAPH STILL EXECUTES  (kill-switch: FASTWAM_VAER4=0)
#
# Read off the AOT graph of `_encode_single_chunk` itself (every node printed
# with its meta `val` shape AND strides, from inside `_fw_cudagraphs_static`),
# not off a kernel-name profile -- the region is graph-replayed so the kernel
# names carry no attribution and these three hid behind generic names.
#
#  (a) `nn.Dropout(0.0)` IN `ResidualBlock.residual`.  In eval mode eager it is
#      an identity, but under AOT tracing `aten::dropout(x, p, train=False)`
#      lowers to `aten.clone.default`, and the graph therefore contains TEN
#      full-activation clones whose output strides are IDENTICAL to their input
#      strides -- e.g. (1,160,1,112,224) (4014080,1,4014080,35840,160) -> the
#      same.  They are the region's `__amd_rocclr_copyBuffer 0.0454 ms x 10`
#      line, i.e. 3.9% of a 1.16 ms region spent copying 8/4/2/1 MB tensors onto
#      themselves.  Skipping a p == 0 / eval-mode Dropout is the identity.
#
#  (b) `F.scaled_dot_product_attention` in the AttentionBlock has no mask, so it
#      takes the MATH backend, and torch's math path (i) `_to_copy`s all three
#      [1,1,392,640] chunk views to contiguous because they are columns of the
#      fused [392,1920] qkv, and (ii) uses `aten::_safe_softmax`, which drags
#      `isneginf` + `all` + `where` + `fill_` behind it to handle fully-masked
#      rows that cannot exist here.  Written as two `bmm`s and a plain
#      `softmax`, both disappear.  With no mask `_safe_softmax` IS `softmax`
#      (its extra branch only fires on an all -inf row), and the two `* E**-0.25`
#      scalings are kept exactly where the math backend puts them, so the
#      arithmetic is unchanged.
#
#  (c) The head's `CausalConv3d(640, 96, 3)` is followed by `self.conv1`, a
#      1x1x1 `CausalConv3d(96, 96)`, with NOTHING between them, and only the
#      FIRST 48 of those 96 channels are used (`mu, log_var = ....chunk(2)`;
#      `log_var` is dead on this path).  Two linear ops with no nonlinearity
#      between them compose exactly, and the trailing `(mu - s0) * s1` is
#      itself affine, so all four collapse into ONE 3x3 conv with 48 output
#      channels:  W = s1 * (Wq[:48] . Wh),  b = (Wq[:48] . bh + bq[:48] - s0)*s1.
#      That deletes a conv, its 1x1 GEMM epilogue, the chunk and two full
#      elementwise passes, and halves the head conv's output extent.
#      NOT bit-exact: the composed weight is contracted in fp32 and stored bf16,
#      where the stock path rounds the 96-channel intermediate ACTIVATION to
#      bf16 instead.  Measured on the oracle mean|d| and the rollout.
#      Separately switchable: FASTWAM_VAEHEAD=0.
# ---------------------------------------------------------------------------
_FW_VAER4 = not _fw_off(_os.environ.get("FASTWAM_VAER4", "1"))
# r4_d2 HOUSEKEEPING: this assignment SHADOWED r3_d0's `_FW_VAEHEAD` (the
# patch_nhwc gather head, defined ~200 lines above).  Both default True so
# shipped behaviour was what was measured, but the shadow silently made
# FASTWAM_VAER4=0 disable the gather head too.  Renamed; nothing reads it.
_FW_VAEHEAD_R4 = _FW_VAER4 and not _fw_off(_os.environ.get("FASTWAM_VAEHEAD", "1"))


def _fw_avgdown_add_impl(m, x, ft):
    y = torch.empty_like(m)
    n = m.numel()
    blk = 1024
    o = int(m.shape[1])
    _fw_avgdown_add_k[((n + blk - 1) // blk,)](
        _fw_cl_flat(m), _fw_cl_flat(x), _fw_cl_flat(y), n, int(x.shape[-1]),
        C=int(x.shape[1]), O=o, W2=int(m.shape[-1]), FT=int(ft), BLOCK=blk,
        num_warps=4)
    return y


def _fw_avgdown_add_meta(m, x, ft):
    return torch.empty_like(m)


def _fw_avgdown_add(m, x, ft, gs):
    """`m + AvgDown3D(x)` in one kernel, or None when this is not our shape."""
    if not (_FW_VAEAVG2 and _FW_TL is not None and _FW_FUSE_OPS):
        return None
    if gs != 4 or ft not in (1, 2):
        return None
    if m.dim() != 5 or x.dim() != 5 or x.shape[0] != 1 or x.shape[2] != 1:
        return None
    c, h, w = int(x.shape[1]), int(x.shape[3]), int(x.shape[4])
    o = int(m.shape[1])
    if (h % 2) or (w % 2):
        return None
    if m.shape[2] != 1 or m.shape[3] != h // 2 or m.shape[4] != w // 2:
        return None
    if o != (c * 2 if ft == 2 else c):
        return None
    if (m.dtype != x.dtype or not x.is_cuda
            or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)):
        return None
    if _fw_cl_flat(x) is None or _fw_cl_flat(m) is None:
        return None
    return torch.ops.fastwam_vae.avgdown_add.default(m, x, int(ft))


def _fw_like_meta(x, *_a, **_k):
    return torch.empty_like(x)


_FW_FUSE_OPS = False
_FW_GATH_OP = False
_FW_HT_OP = False
if _FW_TL is not None:
    try:
        # Low-level `Library`, for the SAME reason `conv2d_nhwc` uses it: the
        # `torch.library.custom_op` wrapper's per-call schema/device validation
        # cost 0.283 ms of exposed host bubble in this region last round.
        _fw_lib2 = torch.library.Library("fastwam_vae", "FRAGMENT")
        _fw_lib2.define(
            "rmsnorm_tail(Tensor x, Tensor d, Tensor g, float scale, "
            "bool silu) -> Tensor")
        _fw_lib2.impl("rmsnorm_tail", _fw_rmsnorm_tail_impl, "CUDA")
        _fw_lib2.impl("rmsnorm_tail", _fw_like_meta, "Meta")
        _fw_lib2.define(
            "rmsnorm_full(Tensor x, Tensor g, float scale, bool silu, "
            "Tensor? bi) -> Tensor")
        _fw_lib2.impl("rmsnorm_full", _fw_rmsnorm_full_impl, "CUDA")
        _fw_lib2.impl("rmsnorm_full", _fw_like_meta, "Meta")
        _fw_lib2.define("biasadd_res(Tensor x, Tensor b, Tensor h) -> Tensor")
        _fw_lib2.impl("biasadd_res", _fw_biasadd_res_impl, "CUDA")
        _fw_lib2.impl("biasadd_res", _fw_like_meta, "Meta")
        _fw_lib2.define(
            "addchain(Tensor x, Tensor? b, Tensor? hy, Tensor? hb, "
            "Tensor? hh, Tensor? k) -> Tensor")
        _fw_lib2.impl("addchain", _fw_addchain_impl, "CUDA")
        _fw_lib2.impl("addchain", _fw_like_meta, "Meta")
        _fw_lib2.define(
            "rmsnorm_fullr(Tensor x, Tensor g, float scale, bool silu, "
            "Tensor bi, Tensor r) -> Tensor")
        _fw_lib2.impl("rmsnorm_fullr", _fw_rmsnorm_fullr_impl, "CUDA")
        _fw_lib2.impl("rmsnorm_fullr", _fw_like_meta, "Meta")
        _fw_lib2.define("avgdown_add(Tensor m, Tensor x, int ft) -> Tensor")
        _fw_lib2.impl("avgdown_add", _fw_avgdown_add_impl, "CUDA")
        _fw_lib2.impl("avgdown_add", _fw_avgdown_add_meta, "Meta")
        _fw_lib2.define("i2c_gather(Tensor x, int ph, int pw) -> Tensor")
        _fw_lib2.impl("i2c_gather", _fw_i2c_gather_impl, "CUDA")
        _fw_lib2.impl("i2c_gather", _fw_i2c_gather_meta, "Meta")
        _fw_lib2.define("patch_nhwc(Tensor x, int p) -> Tensor")
        _fw_lib2.impl("patch_nhwc", _fw_patch_nhwc_impl, "CUDA")
        _fw_lib2.impl("patch_nhwc", _fw_patch_nhwc_meta, "Meta")
        _fw_lib2.define(
            "vae_tail(Tensor v, Tensor s0, Tensor s1, int c) -> Tensor")
        _fw_lib2.impl("vae_tail", _fw_vaetail_impl, "CUDA")
        _fw_lib2.impl("vae_tail", _fw_vaetail_meta, "Meta")
        _FW_FUSE_OPS = True
        _FW_GATH_OP = True
        _FW_HT_OP = True
    except Exception:
        try:
            torch.ops.fastwam_vae.rmsnorm_tail.default
            torch.ops.fastwam_vae.biasadd_res.default
            _FW_FUSE_OPS = True
        except Exception:
            _FW_FUSE_OPS = False
        try:
            torch.ops.fastwam_vae.i2c_gather.default
            _FW_GATH_OP = True
        except Exception:
            _FW_GATH_OP = False
        try:
            torch.ops.fastwam_vae.patch_nhwc.default
            torch.ops.fastwam_vae.vae_tail.default
            _FW_HT_OP = True
        except Exception:
            _FW_HT_OP = False


# ---------------------------------------------------------------------------
# r5_d2 -- ABSORB `_fw_biasadd_res_k` INTO ITS CONSUMER.  (FASTWAM_VAERES=0)
#
# A ResidualBlock ends in `(conv + bias) + shortcut`, shipped as one
# `_fw_biasadd_res_k` launch (0.0525 ms x10 in the encode profile).  For FIVE
# of those ten the result is consumed by exactly two things that ALREADY read
# the same bytes: the next block's `norm1`, and that block's own closing
# `+ h` (its shortcut is `nn.Identity`, so `h` IS this tensor).  Instead of
# materialising the sum we carry it as the deferred triple `(y, bias, res)`
# and let both consumers do the add themselves --
# `_fw_rmsnorm_fullr_k` / `_fw_biasadd_res2_k`.  Net per site: one launch and
# one full write+read of an 8/4/2 MB intermediate deleted, one extra streamed
# read added to each of the two consumers.
#
# BIT-EXACT: the add is elementwise, every round back to bf16 is kept where
# `_fw_biasadd_res_k` put it, and the reduction that follows sees identical
# inputs.  The deferral is dropped (materialised through the shipped op) the
# moment any tensor is not channel-innermost contiguous.
# ---------------------------------------------------------------------------
_FW_VAERES = not _fw_off(_os.environ.get("FASTWAM_VAERES", "1"))


def _fw_mat(t):
    """Materialise a deferred `(y, bias, res)` triple (or pass a tensor through)."""
    if not isinstance(t, tuple):
        return t
    y, b, h = t
    if (b is not None and _FW_VAEFUSE_ADD and _fw_can_fuse(y, b)
            and b.numel() == y.shape[1] and h.shape == y.shape
            and h.dtype == y.dtype and _fw_cl_flat(h) is not None):
        return torch.ops.fastwam_vae.biasadd_res.default(y, b, h)
    if b is not None:
        y = y + b.view([-1] + [1] * (y.dim() - 2))
    return y + h


def _fw_defer_ok(t):
    """True iff every member of the triple is a shape the fused ops accept."""
    if not isinstance(t, tuple):
        return False
    y, b, h = t
    return (_FW_VAERES and _FW_FUSE_OPS and b is not None
            and _fw_can_fuse(y, b) and b.numel() == y.shape[1]
            and h.shape == y.shape and h.dtype == y.dtype
            and _fw_cl_flat(h) is not None)


def _fw_can_fuse(x, g):
    """Cheap capability test -- must be false for anything the kernel cannot do."""
    return (_FW_TL is not None and _FW_FUSE_OPS
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and g.dtype == x.dtype
            and x.dim() >= 3
            and x.numel() > 0
            and _fw_cl_flat(x) is not None)


def _fw_norm_tail(x, d, gamma, scale, silu):
    """Fused `((x/clamp_min(d,eps))*scale)*gamma + 0` [+ silu], or None."""
    if not _fw_can_fuse(x, gamma):
        return None
    if gamma.numel() != x.shape[1]:
        return None
    return torch.ops.fastwam_vae.rmsnorm_tail.default(x, d, gamma, float(scale),
                                                      bool(silu))


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x, silu=False, pre_bias=None, pre_res=None):
        if (pre_res is not None and self.channel_first and _FW_VAEFUSE
                and isinstance(self.bias, float) and self.bias == 0.0
                and _FW_VAENORM1 and _fw_can_fuse(x, self.gamma)
                and _fw_next_pow2(int(x.shape[1])) <= 1024
                and self.gamma.numel() == x.shape[1]
                and pre_bias is not None
                and pre_bias.numel() == x.shape[1]
                and pre_res.shape == x.shape and pre_res.dtype == x.dtype
                and _fw_cl_flat(pre_res) is not None):
            # r5_d2: ONE kernel for [producer bias + producer skip] + reduce
            # + tail -- the producing block's `_fw_biasadd_res_k` launch.
            y = torch.ops.fastwam_vae.rmsnorm_fullr.default(
                x, self.gamma, float(self.scale),
                bool(silu and _FW_VAEFUSE_SILU), pre_bias, pre_res)
            if y is not None:
                return y if (not silu or _FW_VAEFUSE_SILU) else F.silu(y)
        if pre_res is not None:
            x = _fw_mat((x, pre_bias, pre_res))
            pre_bias = None
        if self.channel_first and _FW_VAEFUSE and isinstance(self.bias, float) \
                and self.bias == 0.0 and _fw_can_fuse(x, self.gamma):
            # SAME reduce ATen's F.normalize uses -- only the pointwise tail is
            # fused, so the summation order is untouched (see the long note).
            if _FW_VAENORM1 and _fw_next_pow2(int(x.shape[1])) <= 1024 \
                    and self.gamma.numel() == x.shape[1] \
                    and (pre_bias is None
                         or pre_bias.numel() == x.shape[1]):
                # ONE kernel: [+ producer bias] + reduce + tail.
                y = torch.ops.fastwam_vae.rmsnorm_full.default(
                    x, self.gamma, float(self.scale),
                    bool(silu and _FW_VAEFUSE_SILU), pre_bias)
            else:
                if pre_bias is not None:
                    x = x + pre_bias.view([-1] + [1] * (x.dim() - 2))
                # SAME reduce ATen's F.normalize uses.
                d = x.norm(2, 1, keepdim=True)
                y = _fw_norm_tail(x, d, self.gamma, self.scale,
                                  silu and _FW_VAEFUSE_SILU)
            if y is not None:
                return y if (not silu or _FW_VAEFUSE_SILU) else F.silu(y)
        if pre_bias is not None:
            x = x + pre_bias.view([-1] + [1] * (x.dim() - 2))
        y = F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias
        return F.silu(y) if silu else y


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
            _mods = list(self.resample) if isinstance(
                self.resample, nn.Sequential) else [self.resample]
            if (_FW_VAEGLUE and _FW_NHWC_CONV is not None and len(_mods) == 2
                    and isinstance(_mods[0], nn.ZeroPad2d)
                    and tuple(_mods[0].padding) == (0, 1, 0, 1)
                    and isinstance(_mods[1], nn.Conv2d)
                    and _mods[1].kernel_size == (3, 3)
                    and _mods[1].stride == (2, 2)
                    and _mods[1].padding == (0, 0)
                    and _mods[1].groups == 1
                    and _mods[1].dilation == (1, 1)):
                # `ZeroPad2d((0,1,0,1)) -> conv3x3 s2 p0` is EXACTLY
                # `conv3x3 s2 p2` with the first output row and column
                # dropped: with pad 2 the windows start at input row -2, so
                # window j covers rows [2j-2, 2j] and window j=1 is the
                # zero-padded variant's window 0, while the last window ends
                # on the single trailing zero the ZeroPad2d supplied.  The
                # pad module costs a full-tensor `fill_` + `copy_` (two
                # dispatches over 8 MB at the first downsample) and nothing
                # else; the slice costs nothing because the bias add that
                # follows materializes it anyway.
                _m = _mods[1]
                _r = _fw_conv2d(y, _m.weight, _m.bias, 2, 2, defer_bias=True)
                if isinstance(_r, tuple):
                    y = _r[0][:, :, 1:, 1:]
                    if _r[1] is not None:
                        y = y + _r[1].view(1, -1, 1, 1)
                else:
                    y = _r[:, :, 1:, 1:]
                return y.unsqueeze(2)
            for _m in _mods:
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

    def forward_nocache(self, x, defer=False, extra=None):
        # feat_cache is None -> every CausalConv3d is called with cache_x=None
        # and the clone/cat ladder is dead. Bit-exact.
        #
        # r5_d2: `x` may arrive as a DEFERRED triple `(y, bias, res)` meaning
        # `(y + bias) + res` -- the previous block's un-materialised output.
        # `defer=True` asks this block to return its own output the same way.
        pres = None
        pend0 = None
        if isinstance(x, tuple):
            if (_FW_VAEFUSE and isinstance(self.shortcut, nn.Identity)
                    and _fw_defer_ok(x)):
                # the shortcut IS the deferred sum; norm1 absorbs the add and
                # the closing `+ h` re-derives it (both already read the bytes)
                h = x
                x, pend0, pres = x
            else:
                x = _fw_mat(x)
                h = self.shortcut(x)
        else:
            h = self.shortcut(x)
        if _FW_VAEFUSE:
            # The block is literally [RMS_norm, SiLU, Conv, RMS_norm, SiLU,
            # Dropout, Conv].  Each (RMS_norm, SiLU) pair becomes ONE kernel,
            # and the trailing `+ h` rides along on the last conv's peeled bias
            # add.  10 ResidualBlocks x 3 dispatches = 30 kernels deleted.
            mods = list(self.residual)
            i, n = 0, len(mods)
            pend = pend0 if pres is not None else None
            while i < n:
                m = mods[i]
                last = (i == n - 1)
                if (isinstance(m, RMS_norm) and i + 1 < n
                        and isinstance(mods[i + 1], nn.SiLU)):
                    x = m(x, silu=True, pre_bias=pend, pre_res=pres)
                    pend = None
                    pres = None
                    i += 2
                    continue
                if last and isinstance(m, CausalConv3d):
                    if (_FW_VAERES
                            and (defer or isinstance(h, tuple)
                                 or extra is not None)
                            and m.bias is not None and m._fw_w2d is not None
                            and x.shape[2] == 1):
                        _o = m(x, defer_bias=True)
                        if isinstance(_o, tuple):
                            y, b = _o
                            hy, hb, hh = (h if isinstance(h, tuple)
                                          else (h, None, None))
                            if (defer and extra is None
                                    and not isinstance(h, tuple)
                                    and _fw_defer_ok((y, b, h))):
                                return (y, b, h)
                            if (_fw_can_fuse(y, b) and b.numel() == y.shape[1]
                                    and _fw_cl_flat(hy) is not None
                                    and (hh is None
                                         or _fw_cl_flat(hh) is not None)
                                    and (extra is None
                                         or _fw_cl_flat(extra) is not None)):
                                return torch.ops.fastwam_vae.addchain.default(
                                    y, b, hy, hb, hh, extra)
                            r = _fw_mat((y, b, _fw_mat(h)))
                            return r if extra is None else r + extra
                        x = _o
                        r = x + _fw_mat(h)
                        return r if extra is None else r + extra
                    x = m(x, res=_fw_mat(h))
                    return x if extra is None else x + extra
                if (_FW_VAENORM1 and isinstance(m, CausalConv3d)
                        and i + 1 < n and isinstance(mods[i + 1], RMS_norm)
                        and m.bias is not None and m._fw_w2d is not None
                        and x.shape[2] == 1):
                    # the next thing that touches this conv's output is the
                    # norm -- hand the bias to it instead of paying a separate
                    # full-tensor `aten::add` pass for it.
                    _o = m(x, defer_bias=True)
                    if isinstance(_o, tuple):
                        x, pend = _o
                    else:
                        x = _o
                    i += 1
                    continue
                if _FW_VAER4 and isinstance(m, nn.Dropout) and not m.training:
                    # eval-mode Dropout is the identity, but AOT lowers
                    # `aten::dropout(train=False)` to a full `aten.clone` --
                    # ten of them, 0.045 ms of copying tensors onto themselves.
                    i += 1
                    continue
                x = m(x)
                i += 1
            r = x + _fw_mat(h)
            return r if extra is None else r + extra
        for layer in self.residual:
            x = layer(x)
        r = x + _fw_mat(h)
        return r if extra is None else r + extra


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
        # r5_d2: `x` may be the producer's un-materialised `(y, bias, res)`.
        # Both of this block's consumers of it -- `self.norm` and the closing
        # `+ identity` -- already stream those bytes, so the producer's
        # `_fw_biasadd_res_k` launch never has to run.
        _pb = _pr = None
        if isinstance(x, tuple):
            if _FW_VAERES and _fw_defer_ok(x) and _FW_VAENHWC \
                    and x[0].shape[0] == 1 and x[0].shape[2] == 1:
                identity = x
                _y4, _pb, _pr = x[0][:, :, 0], x[1], x[2][:, :, 0]
                if _fw_cl_flat(_pr) is None:
                    identity = _fw_mat(x)
                    _y4 = identity[:, :, 0]
                    _pb = _pr = None
            else:
                x = _fw_mat(x)
        if not isinstance(x, tuple):
            if not _FW_VAENHWC or x.shape[0] != 1 or x.shape[2] != 1:
                return self.forward(x)
            identity = x
            _y4 = x[:, :, 0]
        b, c, t, h, w = (identity[0] if isinstance(identity, tuple)
                         else identity).size()
        y = self.norm(_y4, pre_bias=_pb, pre_res=_pr)
        if _FW_VAER4:
            # torch's math SDPA on these column-views costs three `_to_copy`s
            # plus `_safe_softmax`'s isneginf/all/where/fill_ ladder, none of
            # which can do anything here (no mask).  Same arithmetic, 7 fewer
            # dispatches.  The two `E ** -0.25` scalings are placed exactly
            # where `scaled_dot_product_attention`'s math path places them.
            qkv = _fw_conv2d_mod(self.to_qkv, y).permute(0, 2, 3, 1).reshape(
                b, h * w, c * 3)
            sc = float(c) ** -0.25
            q = qkv[:, :, 0:c] * sc
            k = qkv[:, :, c:2 * c] * sc
            v = qkv[:, :, 2 * c:3 * c]
            a = torch.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)
            y = torch.bmm(a, v)
            y = y.reshape(b, h, w, c).permute(0, 3, 1, 2)
        else:
            qkv = _fw_conv2d_mod(self.to_qkv, y).permute(0, 2, 3, 1).reshape(
                b, 1, h * w, c * 3)
            q, k, v = qkv.chunk(3, dim=-1)
            y = F.scaled_dot_product_attention(q, k, v)
            y = y.reshape(b, h, w, c).permute(0, 3, 1, 2)
        y = _fw_conv2d_mod(self.proj, y)
        y = y.unsqueeze(2)
        if isinstance(identity, tuple):
            hy, hb, hh = identity
            if (_fw_can_fuse(y, hb) and _fw_cl_flat(hy) is not None
                    and _fw_cl_flat(hh) is not None):
                return torch.ops.fastwam_vae.addchain.default(
                    y, None, hy, hb, hh, None)
            identity = _fw_mat(identity)
        return y + identity


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
        if (_FW_VAEFUSE_AVG and self.factor == 1 and self.group_size == 1
                and self.in_channels == self.out_channels
                and x.shape[2] % self.factor_t == 0):
            # factor == 1 -> no pad, the permute is the identity permutation of
            # unit-size axes, and `mean` reduces a dim of size 1.  Stock torch
            # still spends three full copies on it (constant_pad_nd clone +
            # permute().contiguous() + reduce).  Returning `x` is bit-exact:
            # mean over exactly one element is that element.
            return x
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        if pad_t or not _FW_VAEGLUE:
            # a zero-width `F.pad` is a full-tensor copy that changes nothing;
            # on the encode path (T == 1, factor_t == 1) it always is one.
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
        _a0 = self.avg_shortcut
        _avg_id = (isinstance(_a0, AvgDown3D) and _FW_VAEFUSE_AVG
                   and _a0.factor == 1 and _a0.group_size == 1
                   and _a0.in_channels == _a0.out_channels
                   and x.shape[2] % _a0.factor_t == 0)
        _avg_done = False
        mods = list(self.downsamples)
        nm = len(mods)
        for j, module in enumerate(mods):
            nxt = mods[j + 1] if j + 1 < nm else None
            # r5_d2: hand the block's closing `(conv+bias)+skip` straight to
            # the NEXT block, whose norm1 and closing add both consume it.
            if (_FW_VAERES and isinstance(module, ResidualBlock)
                    and isinstance(nxt, ResidualBlock)
                    and isinstance(nxt.shortcut, nn.Identity)):
                x = module.forward_nocache(x, defer=True)
            elif (_FW_VAERES and nxt is None
                  and isinstance(module, ResidualBlock) and _avg_id):
                # the whole shortcut is the identity (factor == 1), so the
                # closing `x + x_copy` rides on the block's own residual add
                x = module.forward_nocache(x, extra=x_copy)
                _avg_done = True
            else:
                x = module.forward_nocache(x)
        x = _fw_mat(x)
        if _avg_done:
            return x
        a = self.avg_shortcut
        if isinstance(a, AvgDown3D) and a.factor_s == 2:
            y = _fw_avgdown_add(x, x_copy, a.factor_t, a.group_size)
            if y is not None:
                return y
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
        head = list(self.head)
        n = len(head)
        # r5_d2: the LAST middle block's `(conv+bias)+skip` is consumed by
        # exactly one thing -- the head norm -- so it never has to exist.
        _hdefer = (_FW_VAERES and n >= 2 and isinstance(head[0], RMS_norm)
                   and isinstance(head[1], nn.SiLU) and _FW_VAEFUSE)
        mid = list(self.middle)
        for j, layer in enumerate(mid):
            if isinstance(layer, ResidualBlock):
                _nx = mid[j + 1] if j + 1 < len(mid) else None
                _df = ((_hdefer and j == len(mid) - 1)
                       or (_FW_VAERES and isinstance(_nx, AttentionBlock)))
                x = layer.forward_nocache(x, defer=_df)
            elif isinstance(layer, AttentionBlock):
                x = layer.forward_nocache(x)
            else:
                x = layer(_fw_mat(x))
        i = 0
        while i < n:
            m = head[i]
            if (_FW_VAEFUSE and isinstance(m, RMS_norm) and i + 1 < n
                    and isinstance(head[i + 1], nn.SiLU)):
                if isinstance(x, tuple):
                    _y, _b, _r = x
                    x = m(_y, silu=True, pre_bias=_b, pre_res=_r)
                else:
                    x = m(x, silu=True)
                i += 2
                continue
            x = _fw_mat(x)
            x = m(x)
            i += 1
        return _fw_mat(x)


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
        self.register_buffer("mean", torch.tensor(mean), persistent=False)
        self.register_buffer("inv_std", torch.tensor(std).reciprocal(), persistent=False)

        # init model
        self.model = VideoVAE_(z_dim=z_dim).eval().requires_grad_(False)
        self.upsampling_factor = 8
        self.temporal_downsample_factor = 4
        self.z_dim = z_dim

    @property
    def scale(self):
        return [self.mean, self.inv_std]


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
        raise NotImplementedError(
            "WanVideoVAE.encode() legacy per-video loop is disabled. "
            "Call `vae.model.encode(batched_video, vae.scale)` for fixed-shape batched encode."
        )


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
        if (_FW_VAEHEAD and _FW_VAENHWC and _FW_HT_OP
                and x.dim() == 5 and x.shape[0] == 1 and x.shape[2] == 1
                and x.is_contiguous()
                and int(x.shape[3]) % 2 == 0 and int(x.shape[4]) % 2 == 0
                and x.is_cuda):
            # r3_d0: patchify + the channels_last_3d relayout in ONE gather.
            x = torch.ops.fastwam_vae.patch_nhwc.default(x, 2)
        else:
            x = patchify(x, patch_size=2)
            if _FW_VAENHWC:
                # ENTER the NHWC domain once.  Fused by inductor into the
                # patchify pointwise, so it costs no extra pass over memory.
                x = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.encoder.forward_nocache(x)
        v = self.conv1(x)
        if (_FW_VAETAIL and _FW_VAENHWC and _FW_HT_OP
                and v.dim() == 5 and v.shape[0] == 1 and v.shape[2] == 1
                and int(v.shape[1]) == 2 * self.z_dim and v.is_cuda
                and _fw_cl_flat(v) is not None):
            # r3_d0: chunk + (-s0) + (*s1) + contiguous in ONE pass, both bf16
            # rounds preserved.  Also LEAVES the NHWC domain -- the output is
            # the plain contiguous `mu` nothing NHWC escapes through.
            return torch.ops.fastwam_vae.vae_tail.default(
                v, s0, s1, int(self.z_dim))
        mu, log_var = v.chunk(2, dim=1)
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
        if _FW_VAEGLUE:
            # Fixed address => the capture backend does not have to stage-copy
            # these two constants into the graph on every call.
            try:
                torch._dynamo.mark_static_address(s0)
                torch._dynamo.mark_static_address(s1)
            except Exception:
                pass
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
            # Weight-identity key: a checkpoint reload changes it and rebuilds
            # BOTH the collapsed 2-D weights and the captured encode graph.
            _key = (w.data_ptr(), self.encoder.conv1.weight.data_ptr())
            # One-shot, eager (never traced / captured): collapse every F=1
            # causal 3-D conv on the ENCODE path to its algebraically identical
            # 2-D form. Re-keyed on the owning weight so a reload rebuilds it.
            if _FW_VAE2D:
                if self.__dict__.get("_fw_w2d_key") != _key:
                    _fw_prepare_conv2d(self.encoder)
                    _fw_prepare_conv2d(self.conv1)
                    if _FW_VAENHWC:
                        # the plain nn.Conv2d modules on the encode path
                        # (Resample38 spatial convs, AttentionBlock 1x1s)
                        _fw_prepare_nhwc(self.encoder)
                    self.__dict__["_fw_w2d_key"] = _key
            # Identity-based static inputs for the capture backend: mark the
            # frozen weights ONCE, re-keyed on the same weight identity, so the
            # replayed graph reads them in place instead of re-staging them.
            if _FW_VAESTATIC and self.__dict__.get("_fw_static_key") != _key:
                _fw_mark_static(self.encoder, self.conv1)
                self.__dict__["_fw_static_key"] = _key
            s0, s1 = self._scale_on_device(scale, w.dtype, w.device)
            # ---- r3_d0: STATIC-ADDRESS the graph's one dynamic input --------
            # MEASURED on the chrome trace of this region: every replay runs a
            # `multi_tensor_apply_kernel` (grid [5,1,1], 13.4-13.8 us) BEFORE
            # the graph, and it is cudagraph_trees copying `x` into the capture
            # placeholder because `x` is the only input that is neither a
            # parameter nor `mark_static_address`ed.  5 CTAs on 256 CUs for
            # 0.6 MB is ~90 GB/s.  Staging it ourselves into a persistent,
            # marked buffer moves that copy into the static-input set: the
            # per-replay foreach copy disappears and the one `copy_` left runs
            # at full width.
            # EPISODE STATE: the buffer is REWRITTEN unconditionally on every
            # call before it is read, so it can never serve a stale frame (it
            # is an input staging slot, not a memo -- nothing is keyed on it).
            # NUMERICS: a copy of the same bytes; bit-exact by construction.
            if _FW_VAEXBUF:
                xb = self.__dict__.get("_fw_xbuf")
                if (xb is None or xb.shape != x.shape or xb.dtype != x.dtype
                        or xb.device != x.device
                        or xb.stride() != x.stride()):
                    xb = torch.empty_strided(x.shape, x.stride(),
                                             dtype=x.dtype, device=x.device)
                    try:
                        torch._dynamo.mark_static_address(xb)
                    except Exception:
                        xb = None
                    self.__dict__["_fw_xbuf"] = xb
                if xb is not None:
                    xb.copy_(x)
                    x = xb
            # ---- graph/compile the encode (see FASTWAM_VAEGRAPH note above) --
            g = None
            if _FW_VAEGRAPH and not torch.compiler.is_compiling():
                ent = self.__dict__.get("_fw_enc_graph")
                if ent is not None and ent[0] == _key:
                    g = ent[1]
                else:
                    g = _fw_build_encode_graph(self._encode_single_chunk)
                    self.__dict__["_fw_enc_graph"] = (_key, g)
                    _fw_vaegraph_note("built" if g is not None else "build-failed")
            if g is not None:
                try:
                    # NEVER return a cudagraph's internal static output buffer.
                    mu = g(x, s0, s1).clone()
                    _fw_vaegraph_note("engaged")
                except Exception as _e:
                    # Any capture/compile failure degrades to the eager region
                    # for the rest of the process, never to a wrong answer.
                    if _os.environ.get("FASTWAM_VAEDBG", "0") != "0":
                        import traceback as _tb
                        _tb.print_exc()
                    self.__dict__["_fw_enc_graph"] = (_key, None)
                    _fw_vaegraph_note("fallback-eager")
                    mu = self._encode_single_chunk(x, s0, s1)
            else:
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
        self.register_buffer("mean", torch.tensor(mean), persistent=False)
        self.register_buffer("inv_std", torch.tensor(std).reciprocal(), persistent=False)

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
