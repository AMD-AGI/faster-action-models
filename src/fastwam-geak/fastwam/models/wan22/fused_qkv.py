# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused QKV projection variants for the ActionDiT attention modules.

The baseline SelfAttention/CrossAttention use 3 separate nn.Linear projections
for Q, K, V. Each is a tiny GEMM (e.g. [32, 1024] x [1024, 3072]) that is
launch-bound rather than compute-bound, so collapsing them into a single larger
GEMM removes kernel-launch and dispatch overhead.

- SelfAttention: Q/K/V all read from the same input x  -> fuse all three into
  one qkv projection (1 launch instead of 3), then chunk the output.
- CrossAttention: Q reads from x, but K/V read from ctx -> fuse only K+V into a
  single kv projection (2 launches instead of 3).

Numerically these are identical to the separate projections: concatenating the
weight rows and splitting the output is exactly equivalent to running the
projections independently. See `fuse_attention_state_dict` for loading existing
separate-projection checkpoints into the fused modules.

Selection is controlled by the FASTWAM_FUSED_QKV env flag (default OFF); see
DiTBlock in wan_video_dit.py.
"""

import os
from typing import Optional

import torch
import torch.nn as nn

from .wan_video_dit import RMSNorm, rope_apply, flash_attention


def fused_qkv_enabled() -> bool:
    """Return True when the FASTWAM_FUSED_QKV feature flag is set."""
    return os.environ.get("FASTWAM_FUSED_QKV", "0").lower() in ("1", "true", "yes", "on")


class FusedSelfAttention(nn.Module):
    """SelfAttention with a single fused Q/K/V projection.

    Drop-in replacement for SelfAttention: same constructor signature, same
    forward signature, and a state_dict layout that can be produced from a
    separate-projection checkpoint via `fuse_attention_state_dict`.
    """

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        # Single fused projection: hidden_dim -> 3 * attn_hidden_dim
        self.qkv = nn.Linear(hidden_dim, 3 * self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x, freqs, self_attn_mask: Optional[torch.Tensor] = None):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.norm_q(q)
        k = self.norm_k(k)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=self_attn_mask)
        return self.o(x)


class FusedCrossAttention(nn.Module):
    """CrossAttention with a fused K/V projection.

    Q reads from x and stays separate; K and V both read from ctx and are fused
    into a single kv projection (2 launches instead of 3).
    """

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        # Fused K/V projection: hidden_dim -> 2 * attn_hidden_dim
        self.kv = nn.Linear(hidden_dim, 2 * self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        kv = self.kv(ctx)
        k, v = kv.chunk(2, dim=-1)
        k = self.norm_k(k)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)


def _fuse_linear_weights(*linears: nn.Linear) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Concatenate the weight (and bias) rows of several nn.Linear modules."""
    weight = torch.cat([lin.weight.data for lin in linears], dim=0)
    if all(lin.bias is not None for lin in linears):
        bias = torch.cat([lin.bias.data for lin in linears], dim=0)
    else:
        bias = None
    return weight, bias


@torch.no_grad()
def fuse_self_attention(src) -> FusedSelfAttention:
    """Build a FusedSelfAttention initialised from a separate-projection SelfAttention."""
    dst = FusedSelfAttention(src.hidden_dim, src.attn_head_dim, src.num_heads,
                             eps=src.norm_q.eps)
    weight, bias = _fuse_linear_weights(src.q, src.k, src.v)
    dst.qkv.weight.data.copy_(weight)
    if bias is not None:
        dst.qkv.bias.data.copy_(bias)
    dst.o.load_state_dict(src.o.state_dict())
    dst.norm_q.load_state_dict(src.norm_q.state_dict())
    dst.norm_k.load_state_dict(src.norm_k.state_dict())
    return dst.to(src.q.weight.device, src.q.weight.dtype)


@torch.no_grad()
def fuse_cross_attention(src) -> FusedCrossAttention:
    """Build a FusedCrossAttention initialised from a separate-projection CrossAttention."""
    dst = FusedCrossAttention(src.hidden_dim, src.attn_head_dim, src.num_heads,
                              eps=src.norm_q.eps)
    dst.q.load_state_dict(src.q.state_dict())
    weight, bias = _fuse_linear_weights(src.k, src.v)
    dst.kv.weight.data.copy_(weight)
    if bias is not None:
        dst.kv.bias.data.copy_(bias)
    dst.o.load_state_dict(src.o.state_dict())
    dst.norm_q.load_state_dict(src.norm_q.state_dict())
    dst.norm_k.load_state_dict(src.norm_k.state_dict())
    return dst.to(src.q.weight.device, src.q.weight.dtype)


def fuse_attention_state_dict(state_dict: dict) -> dict:
    """Convert a checkpoint with separate Q/K/V projections into the fused layout.

    Rewrites keys in place-style (returns a new dict):
      self_attn: {q,k,v}.{weight,bias}         -> self_attn.qkv.{weight,bias}
      cross_attn: {k,v}.{weight,bias}          -> cross_attn.kv.{weight,bias}
                  (cross_attn.q.* is left unchanged)

    Any keys that are not part of a fusible triple/pair are passed through
    untouched, so this is safe to run over a full model state_dict.
    """
    out: dict = {}
    consumed: set = set()

    # Group keys by the module prefix that owns the projections.
    keys = list(state_dict.keys())

    def _prefixes(suffix_set):
        found = set()
        for k in keys:
            for suffix in suffix_set:
                if k.endswith(suffix):
                    found.add(k[: -len(suffix)])
        return found

    # Self-attention: fuse q,k,v -> qkv where all three exist under the prefix.
    for prefix in _prefixes({".q.weight"}):
        q_w, k_w, v_w = prefix + ".q.weight", prefix + ".k.weight", prefix + ".v.weight"
        if q_w in state_dict and k_w in state_dict and v_w in state_dict:
            # Distinguish self-attn (q from same source) from cross-attn by name.
            if "self_attn" in prefix:
                out[prefix + ".qkv.weight"] = torch.cat(
                    [state_dict[q_w], state_dict[k_w], state_dict[v_w]], dim=0)
                consumed.update({q_w, k_w, v_w})
                for b in (prefix + ".q.bias", prefix + ".k.bias", prefix + ".v.bias"):
                    if b in state_dict:
                        pass
                bs = [prefix + ".q.bias", prefix + ".k.bias", prefix + ".v.bias"]
                if all(b in state_dict for b in bs):
                    out[prefix + ".qkv.bias"] = torch.cat([state_dict[b] for b in bs], dim=0)
                    consumed.update(bs)

    # Cross-attention: fuse k,v -> kv (leave q alone).
    for prefix in _prefixes({".k.weight"}):
        if "cross_attn" not in prefix:
            continue
        k_w, v_w = prefix + ".k.weight", prefix + ".v.weight"
        if k_w in state_dict and v_w in state_dict:
            out[prefix + ".kv.weight"] = torch.cat([state_dict[k_w], state_dict[v_w]], dim=0)
            consumed.update({k_w, v_w})
            bs = [prefix + ".k.bias", prefix + ".v.bias"]
            if all(b in state_dict for b in bs):
                out[prefix + ".kv.bias"] = torch.cat([state_dict[b] for b in bs], dim=0)
                consumed.update(bs)

    for k, val in state_dict.items():
        if k not in consumed:
            out[k] = val
    return out
