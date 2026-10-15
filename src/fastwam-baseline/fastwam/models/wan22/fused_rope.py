# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-valued RoPE for the ActionDiT, replacing the complex-valued fp64 path.

The baseline `rope_apply` (wan_video_dit.py) casts to float64, builds a complex
view, multiplies by complex `freqs`, then casts back. Two problems:

  1. The fp64 cast is unnecessary — the rotation is exact in fp32.
  2. TorchInductor cannot codegen complex ops ("Torchinductor does not support
     code generation for complex operators"), so under torch.compile the RoPE
     falls back to aten extern kernels. That extern call sits between the
     surrounding elementwise work (RMSNorm, modulation) and forces graph
     breaks, so RoPE never fuses with its neighbours. This is the gap between
     the isolated-stack compile speedup (3.8-4.2x) and the real-model one (2.86x).

Real rotation (identical math):

    out[2i]   = x[2i] * cos[i] - x[2i+1] * sin[i]
    out[2i+1] = x[2i] * sin[i] + x[2i+1] * cos[i]

where cos = freqs.real, sin = freqs.imag (freqs = polar(1, angle)).

IMPORTANT (compile correctness): the cos/sin must be derived from the complex
`freqs` OUTSIDE the compiled region and passed in as a *real* tensor. Deriving
them inside the graph (or via a data_ptr cache) re-introduces a complex op / a
graph break and is *slower* than the baseline on these tiny tensors. Use
`precompute_rope_real(freqs)` once in eager code (e.g. the model's pre-forward)
and pass the resulting real cache into the blocks. `rope_apply_real` accepts
either the real cache (fast, fusible) or the raw complex freqs (drop-in, but
eager-only fast).

The real cache is packed as a real tensor of shape [S, 1, D] where
[..., :D/2] = cos and [..., D/2:] = sin.

Selection via env flag FASTWAM_FUSED_ROPE: "torch" | "triton" | "0"/off.
"""

import os
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton optional
    _HAS_TRITON = False


def fused_rope_mode() -> str:
    """Return the RoPE mode from FASTWAM_FUSED_ROPE: 'torch', 'triton', or 'off'."""
    val = os.environ.get("FASTWAM_FUSED_ROPE", "0").lower()
    if val in ("torch", "real"):
        return "torch"
    if val == "triton":
        return "triton"
    if val in ("1", "true", "yes", "on"):
        return "torch"
    return "off"


# ---------------------------------------------------------------------------
# Eager precompute of the packed real cos/sin cache
# ---------------------------------------------------------------------------
_REAL_CACHE: dict = {}


def precompute_rope_real(freqs: torch.Tensor, compute_dtype=torch.float32) -> torch.Tensor:
    """Convert complex `freqs` [S, 1, D/2] into a packed real cache [S, 1, D].

    Layout: result[..., :D/2] = cos, result[..., D/2:] = sin. fp32 by default so
    the rotation stays within 1e-3 of the fp64 baseline in bf16.

    Call this in EAGER code (outside torch.compile); the result is a plain real
    tensor that inductor can fuse when passed into compiled blocks.
    """
    if not freqs.is_complex():
        return freqs  # already packed real cache
    key = (freqs.data_ptr(), tuple(freqs.shape), str(freqs.device), str(compute_dtype))
    cached = _REAL_CACHE.get(key)
    if cached is not None:
        return cached
    cos = freqs.real.to(dtype=compute_dtype)
    sin = freqs.imag.to(dtype=compute_dtype)
    packed = torch.cat([cos, sin], dim=-1).contiguous()  # [S, 1, D]
    _REAL_CACHE[key] = packed
    return packed


def clear_rope_cache() -> None:
    _REAL_CACHE.clear()


def _split_cos_sin(rope_cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    Dh = rope_cache.shape[-1] // 2
    return rope_cache[..., :Dh], rope_cache[..., Dh:]


# ---------------------------------------------------------------------------
# Pure-torch real-valued RoPE
# ---------------------------------------------------------------------------
def rope_apply_real(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Real-valued RoPE, numerically equal to the complex `rope_apply`.

    x: [B, S, N*D].  `freqs` may be either the packed real cache [S, 1, D]
    from `precompute_rope_real` (fast, compile-fusible) or the raw complex
    freqs [S, 1, D/2] (drop-in; converted eagerly, do not do this in-graph).
    """
    in_dtype = x.dtype
    B, S, ND = x.shape
    Dh = (ND // num_heads) // 2

    rope_cache = precompute_rope_real(freqs) if freqs.is_complex() else freqs
    cos, sin = _split_cos_sin(rope_cache)          # each [S, 1, Dh]

    xr = x.reshape(B, S, num_heads, Dh, 2).float()
    x0 = xr[..., 0]
    x1 = xr[..., 1]

    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos

    out = torch.stack((out0, out1), dim=-1).reshape(B, S, ND)
    return out.to(in_dtype)


# ---------------------------------------------------------------------------
# Triton real-valued RoPE
# ---------------------------------------------------------------------------
if _HAS_TRITON:

    @triton.jit
    def _rope_kernel(
        x_ptr, out_ptr, cos_ptr, sin_ptr,
        S, N, D, Dh,
        row_stride, cs_stride,
        BLOCK: tl.constexpr,
    ):
        # One program per row (b, s); a row is N heads * D dims contiguous.
        pid = tl.program_id(0)
        s = pid % S

        n_pairs = N * Dh
        offs = tl.arange(0, BLOCK)
        mask = offs < n_pairs

        head = offs // Dh
        j = offs % Dh

        base = pid * row_stride
        x0_off = base + head * D + 2 * j
        x1_off = x0_off + 1

        cs_off = s * cs_stride + j

        x0 = tl.load(x_ptr + x0_off, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + x1_off, mask=mask, other=0.0).to(tl.float32)
        cos = tl.load(cos_ptr + cs_off, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + cs_off, mask=mask, other=0.0).to(tl.float32)

        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos

        tl.store(out_ptr + x0_off, out0, mask=mask)
        tl.store(out_ptr + x1_off, out1, mask=mask)

    def rope_apply_triton(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
        in_dtype = x.dtype
        B, S, ND = x.shape
        D = ND // num_heads
        Dh = D // 2

        rope_cache = precompute_rope_real(freqs) if freqs.is_complex() else freqs
        cos, sin = _split_cos_sin(rope_cache)
        cos = cos.reshape(S, Dh).contiguous()
        sin = sin.reshape(S, Dh).contiguous()

        xf = x.reshape(B * S, ND).contiguous()
        out = torch.empty_like(xf)

        n_pairs = num_heads * Dh
        BLOCK = triton.next_power_of_2(n_pairs)
        grid = (B * S,)
        _rope_kernel[grid](
            xf, out, cos, sin,
            S, num_heads, D, Dh,
            ND, Dh,
            BLOCK=BLOCK,
        )
        return out.reshape(B, S, ND).to(in_dtype)

else:  # pragma: no cover
    def rope_apply_triton(x, freqs, num_heads):
        return rope_apply_real(x, freqs, num_heads)


# ---------------------------------------------------------------------------
# Dispatcher matching the original signature
# ---------------------------------------------------------------------------
def rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    mode = fused_rope_mode()
    if mode == "triton" and _HAS_TRITON:
        return rope_apply_triton(x, freqs, num_heads)
    return rope_apply_real(x, freqs, num_heads)
