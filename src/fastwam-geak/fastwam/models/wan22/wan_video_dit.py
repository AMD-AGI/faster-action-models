# Based on Wan2.x / DiffSynth-Studio (Apache-2.0) via FastWAM (MIT).
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, Tuple, Optional
from einops import rearrange
from .helpers.gradient import gradient_checkpoint_forward

from fastwam.utils.logging_config import get_logger
from .fused_adaln import adaln_forward

logger = get_logger(__name__)

    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, ctx_mask: Optional[torch.Tensor] = None, compatibility_mode=True):
    if compatibility_mode:
        # Short-sequence fused path (see the block at the bottom of this file).
        # Falls through to the untouched SDPA call for anything outside the
        # verified envelope.
        if q.shape[1] >= _FA_FLAT_MIN_SQ and _fa_eligible(q, k, v, num_heads):
            _m = _fa_mask2d(ctx_mask, q.shape[1], k.shape[1])
            if _m is not False:
                return _fw_short_attn(q, k, v, _m, num_heads)
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        if _fa_eligible(q, k, v, num_heads):
            _m = _fa_mask2d(ctx_mask, q.shape[2], k.shape[2])
            if _m is not False:
                x = _fw_short_attn(q, k, v, _m, num_heads)
                return rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        return x
    else:
        raise NotImplementedError("Only compatibility mode is implemented for flash attention. Please set compatibility_mode=True.")



def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


# ---------------------------------------------------------------------------
# STEP-INVARIANT WORK IN THE DENOISE PROLOGUE (round 4, host_runtime, lever B).
#
# `ActionDiT.pre_dit` runs EAGERLY once per denoise step, OUTSIDE the captured
# graph, and it is the whole of the step's 29 non-transformer launches (0.143 ms
# = 6.7% of the 2.14 ms step, paid 10x in the 10-step case).  From the round-3
# chrome trace, per step:
#     arange/neg/pow/fill/div  ~24.6 us   <-- a MATHEMATICAL CONSTANT, rebuilt
#                                             from scratch on every single step
#     outer/cos/sin/cat/to     ~22.2 us   <-- depends on the timestep
#     time_embedding + time_projection (3 tiny GEMMs + 2 SiLU) ~29 us
#     text_embedding(context)  ~27.0 us   <-- pure function of `context`, which
#                                             is IDENTICAL at every step of a call
#     action_encoder            ~6.2 us   <-- depends on the latents: real work
#
# Two of those pools are pure recomputation and are deleted here:
#  (1) the sinusoid frequency vector `10000**(-arange(dim/2)/(dim/2))` depends on
#      `dim` and NOTHING else, so it is built once per (dim, device).  Five
#      launches on the ~4 us do-nothing dispatch floor per step.  It is a
#      constant, so it carries no cross-episode staleness risk whatsoever.
#  (2) `torch.cat([cos, sin])` becomes two `out=` writes into one buffer, which
#      deletes the cat launch.  Bit-identical: same fp64 cos/sin, same fp64
#      buffer, same contiguous [B, dim] layout, same final `.to(dtype)`.
#
# Kill-switch: FASTWAM_STEPMEMO=0 restores the original expression exactly.
# ---------------------------------------------------------------------------
_STEPMEMO = os.environ.get("FASTWAM_STEPMEMO", "1").lower() not in ("0", "false", "no", "off")

# ROUND-1 host_runtime LEVER 1 kill-switch: WanModel.pre_dit's provably-constant
# `t`/`t_mod` cache (only ever engaged when the latent has exactly one frame, in
# which case `token_timesteps` is identically zero by construction).
_TCONST = os.environ.get("FASTWAM_TCONST", "1").lower() not in ("0", "false", "no", "off")

# ROUND-1 host_runtime LEVER 2 kill-switch: WanModel.pre_dit's patchify+rearrange
# collapse (Conv3d with stride == kernel is exactly a per-patch matmul).
_PATCHMM = os.environ.get("FASTWAM_PATCHMM", "1").lower() not in ("0", "false", "no", "off")

# CALL counter, bumped once per `infer_action` by `MoT.prefill_video_cache`
# (the same place that bumps `MoT._fw_episode`).  It is the OUTER gate for the
# step-scoped memo in `ActionDiT.pre_dit`, which lives in another module and has
# no reference to the MoT.  COMMANDMENT rule 5: a counter, never a data_ptr.
FW_CALL_SEQ = [0]

_SIN_FREQ_CACHE = {}


def _sin_freq(dim, device):
    key = (int(dim), device.type, device.index)
    f = _SIN_FREQ_CACHE.get(key)
    if f is None:
        f = torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64,
                                           device=device).div(dim // 2))
        _SIN_FREQ_CACHE[key] = f
    return f


def _is_compiling() -> bool:
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:
        return False


def sinusoidal_embedding_1d(dim, position):
    if not _STEPMEMO or _is_compiling():
        sinusoid = torch.outer(position.type(torch.float64), torch.pow(
            10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
        x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
        return x.to(position.dtype)
    half = dim // 2
    sinusoid = torch.outer(position.type(torch.float64), _sin_freq(dim, position.device))
    x = torch.empty((sinusoid.shape[0], 2 * half), dtype=torch.float64,
                    device=sinusoid.device)
    torch.cos(sinusoid, out=x[:, :half])
    torch.sin(sinusoid, out=x[:, half:])
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


# ---------------------------------------------------------------------------
# Real-valued RoPE
#
# The original path upcasts x to fp64, builds a complex view, multiplies by the
# complex `freqs` and views back.  TorchInductor cannot codegen complex ops, so
# under torch.compile that becomes three separate dispatches per q/k per layer:
#   * a `..._view_as_complex_` reduction kernel (the RMSNorm, fused only up to
#     the complex view),
#   * an aten `elementwise_kernel_manual_unroll` extern for the complex mul,
#   * a `..._sdpa__to_copy_add_` kernel to lay the result out for SDPA.
# On this tree every one of those sits on the ~4.2 us per-dispatch floor.
#
# The identical rotation in real arithmetic
#       out[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
#       out[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
# is pure elementwise, so inductor folds RMSNorm + rotation + the SDPA layout
# copy into ONE kernel per q/k.  fp32 throughout (the operands are bf16, whose
# 8-bit mantissa is ~4e-3 -- fp32 cos/sin are ~1e-7, i.e. four orders of
# magnitude below the representation error of the output).
#
# `freqs` is handed in already packed as a REAL tensor [S, 1, D] with
# [..., :D/2] = cos and [..., D/2:] = sin (built once per (shape, device) in the
# two `pre_dit`s -- the complex->real conversion must stay OUT of the compiled
# region or it re-introduces the extern call).  A complex `freqs` still works and
# takes the original fp64 path, so the env kill-switch below is exact.
# ---------------------------------------------------------------------------

def real_rope_enabled() -> bool:
    return os.environ.get("FASTWAM_REAL_ROPE", "1").lower() not in ("0", "false", "no", "off")


def pack_rope_real(freqs: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """Complex freqs [S, 1, D/2] -> packed real [S, 1, D] = cat(cos, sin)."""
    if not freqs.is_complex():
        return freqs
    return torch.cat([freqs.real.to(dtype), freqs.imag.to(dtype)], dim=-1).contiguous()


_ROPE_SIGN = {}


def _rope_sign(freqs: torch.Tensor) -> torch.Tensor:
    """Constant [-1, +1] pair-sign vector, cached per (device, dtype)."""
    key = (freqs.device, freqs.dtype)
    t = _ROPE_SIGN.get(key)
    if t is None:
        t = torch.tensor([-1.0, 1.0], device=freqs.device, dtype=freqs.dtype)
        _ROPE_SIGN[key] = t
    return t


def rope_apply(x, freqs, num_heads):
    if not freqs.is_complex():
        # Packed real cos/sin cache -- inductor-fusible fast path.
        #   out = x * cos_pair + flip(x_pair) * (+/-) sin_pair
        # Expressed with a flip + a constant [-1, +1] sign vector instead of a
        # stack of two half-width expressions: every load is then contiguous and
        # the whole rotation is ONE elementwise expression over [B,S,N,Dh,2], so
        # inductor keeps it in the same kernel as the RMSNorm that produced x and
        # as the SDPA layout store that consumes it.
        B, S, ND = x.shape
        Dh = (ND // num_heads) // 2
        cos = freqs[..., :Dh].unsqueeze(-1)
        sin = freqs[..., Dh:].unsqueeze(-1)
        sgn = _rope_sign(freqs)
        xr = x.reshape(B, S, num_heads, Dh, 2).float()
        out = xr * cos + xr.flip(-1) * (sin * sgn)
        return out.reshape(B, S, ND).to(x.dtype)
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


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

        from .fused_qkv import (
            fused_qkv_enabled,
            FusedSelfAttention,
            FusedCrossAttention,
        )
        if fused_qkv_enabled():
            self.self_attn = FusedSelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
            self.cross_attn = FusedCrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        else:
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
        input_x = adaln_forward(x, self.norm1, shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask))
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        input_x = adaln_forward(x, self.norm2, shift_mlp, scale_mlp)
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
            

    def _fw_patch_tokens(self, x: torch.Tensor):
        """(x_tokens [B, f*h*w, D], (f, h, w)) in TWO launches, or None.

        ROUND-1 host_runtime, LEVER 2 (dispatch collapse).  `patchify` +
        the `rearrange(... "b c f h w -> b (f h w) c").contiguous()` that always
        follows it cost FOUR launches / 25.7 us on this box (round-0 chrome
        trace): `Im3d2Col` 4.36 + Tensile `MT64x32x64` 8.80 + an NCHW->bias
        elementwise 5.36, then a second full 0.6 MB permute-copy 7.20.

        `patch_embedding` is a Conv3d whose stride EQUALS its kernel and which
        has no padding/dilation/groups, so it is exactly a per-patch matmul:
        gather the disjoint patches into rows and `addmm` them against the
        flattened weight.  Two launches -- one tiny gather (the latent is only
        tens of KB) and one GEMM -- and it lands in the [B, N, D] token layout
        the caller wants, so the trailing permute-copy disappears as well.

        The flattened weight view `W.reshape(D, C*pt*ph*pw)` is the conv's own
        memory in its own order (out, in, kT, kH, kW), and the gather emits the
        patch elements in that same (c, kt, kh, kw) order, so this is the same
        contraction over the same K=C*pt*ph*pw with the same fp32 accumulation
        the im2col path performs -- it is a re-association of the vendor GEMM,
        not a different computation.

        Returns None (-> original path) for anything outside the envelope:
        a control adapter, a non-contiguous or non-5D input, padding/dilation/
        groups, a stride that differs from the kernel, or a ragged grid.
        """
        if not _PATCHMM or _is_compiling():
            return None
        conv = getattr(self, "patch_embedding", None)
        if conv is None or self.control_adapter is not None:
            return None
        if not isinstance(conv, nn.Conv3d) or conv.bias is None:
            return None
        if tuple(conv.padding) != (0, 0, 0) or tuple(conv.dilation) != (1, 1, 1) \
                or conv.groups != 1:
            return None
        kt, kh, kw = (int(v) for v in conv.kernel_size)
        if tuple(int(v) for v in conv.stride) != (kt, kh, kw):
            return None
        if x.ndim != 5 or not x.is_contiguous():
            return None
        B, C, T, H, W = (int(v) for v in x.shape)
        if C != int(conv.in_channels) or T % kt or H % kh or W % kw:
            return None
        f, h, w = T // kt, H // kh, W // kw
        wt = conv.weight
        if not wt.is_contiguous():
            return None
        D = int(conv.out_channels)
        K = C * kt * kh * kw
        # (B,C,f,kt,h,kh,w,kw) -> (B,f,h,w,C,kt,kh,kw): one gather kernel.
        rows = (x.view(B, C, f, kt, h, kh, w, kw)
                 .permute(0, 2, 4, 6, 1, 3, 5, 7)
                 .reshape(B * f * h * w, K))
        out = torch.addmm(conv.bias, rows, wt.reshape(D, K).t())
        return out.view(B, f * h * w, D), (f, h, w)

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

            # ---------------------------------------------------------------
            # ROUND-1 host_runtime, LEVER 1: the video expert's `t` / `t_mod`
            # are a MATHEMATICAL CONSTANT whenever the latent has ONE frame.
            #
            #   token_timesteps = ones(B, F, TPF) * timestep.view(B,1,1)
            #   token_timesteps[:, 0, :] = 0
            #
            # With F == 1 the second line overwrites the WHOLE tensor, so
            # `token_timesteps` is identically ZERO for ANY `timestep` value.
            # That is a SHAPE fact, not a data fact, so establishing it needs
            # no host sync and no device compare.  `infer_action` always takes
            # this path (`first_frame_latents` is one latent frame, and the
            # caller even passes `timestep = zeros`).
            #
            # Everything downstream (sinusoid -> time_embedding ->
            # time_projection) is then a pure function of the module WEIGHTS
            # and the shape, so it is cached.  Measured from the round-0 chrome
            # trace: 13 eager launches / 104 us of device time per call, of
            # which the [98,3072]x[3072,18432] `time_projection` GEMM alone
            # streams 113 MB of weights (34.2 us) to produce a constant.
            #
            # COMMANDMENT RULE 5 / staleness: the cache is CONTENT-INDEPENDENT
            # (same class as the `_freqs_dev_cache` below) -- there is no input
            # whose change it could miss, so no episode counter can make it
            # stale.  Belt-and-braces, the key still carries data_ptr AND
            # `_version` of all three projection weights, so a checkpoint
            # reload or an in-place weight edit rebuilds it.  The cached
            # tensors are handed out READ-ONLY: `prefill_video_cache` only
            # `_fill_slot`-copies `t_mod` into its own static graph buffer, and
            # `t` is unused by `infer_action`.
            #
            # Kill-switch: FASTWAM_TCONST=0 restores the original expression.
            # ---------------------------------------------------------------
            _tc_key = None
            if _TCONST and int(x.shape[2]) == 1 and not self.training:
                _w0 = self.time_embedding[0].weight
                _w2 = self.time_embedding[2].weight
                _wp = self.time_projection[1].weight
                _tc_key = (
                    int(batch_size), int(tokens_per_frame), int(self.freq_dim),
                    int(self.hidden_dim), str(timestep.device), timestep.dtype,
                    _w0.data_ptr(), _w0._version,
                    _w2.data_ptr(), _w2._version,
                    _wp.data_ptr(), _wp._version,
                )

            _cc = getattr(self, "_fw_tconst", None) if _tc_key is not None else None
            if _cc is not None and _cc[0] == _tc_key:
                t, t_mod = _cc[1], _cc[2]
            else:
                if _tc_key is not None:
                    # Provably all-zero; skip the ones/mul/fill_ trio entirely.
                    token_timesteps = torch.zeros(
                        (batch_size, x.shape[2] * tokens_per_frame),
                        dtype=timestep.dtype,
                        device=timestep.device,
                    )
                else:
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
                if _tc_key is not None:
                    self._fw_tconst = (_tc_key, t, t_mod)
        else:
            raise NotImplementedError("Only support seperated_timestep with fuse_vae_embedding_in_latents for now.")
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        _pt = (self._fw_patch_tokens(x)
               if control_camera_latents_input is None else None)
        if _pt is not None:
            _tok_early, (f, h, w) = _pt
        else:
            _tok_early = None
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

        x_tokens = (_tok_early if _tok_early is not None
                    else rearrange(x, "b c f h w -> b (f h w) c").contiguous())

        # Counter-free but staleness-free cache: the 3D RoPE table is a pure
        # function of (f, h, w) and the constant CPU `self.freqs` buffers, so a
        # shape-keyed cache can never serve stale data across episodes.  This
        # deletes the per-call expand/cat/reshape AND the HtoD copy.
        _fkey = (int(f), int(h), int(w), str(x_tokens.device), bool(real_rope_enabled()))
        _fc = getattr(self, "_freqs_dev_cache", None)
        if _fc is None or _fc[0] != _fkey:
            _fr = torch.cat([
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(f * h * w, 1, -1).to(x_tokens.device)
            if real_rope_enabled():
                _fr = pack_rope_real(_fr)
            self._freqs_dev_cache = (_fkey, _fr)
        freqs = self._freqs_dev_cache[1]

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

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        pre_state = self.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        x_tokens = pre_state["tokens"]
        context_emb = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_attn_mask = pre_state["context_mask"]
        self_attn_mask = self.build_video_to_video_mask(
            video_seq_len=x_tokens.shape[1],
            video_tokens_per_frame=int(pre_state["meta"]["tokens_per_frame"]),
            device=x_tokens.device,
        ) if self.video_attention_mask_mode != "bidirectional" else None # special rule for faster speed

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x_tokens = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x_tokens, context_emb, t_mod, freqs, context_mask=context_attn_mask, self_attn_mask=self_attn_mask
                )
            else:
                x_tokens = block(x_tokens, context_emb, t_mod, freqs, context_mask=context_attn_mask, self_attn_mask=self_attn_mask)

        return self.post_dit(x_tokens, pre_state)


# ===========================================================================
# SHORT-SEQUENCE FUSED ATTENTION  (round-2, memory lane)
#
# WHY.  `attn_fwd` (the aten memory-efficient-attention kernel SDPA selects once
# an `attn_mask` is present) is the single largest kernel on this tree: 0.867 ms
# of the 4.14 ms denoise step (21%) over 60 launches, plus 0.750 ms of the 6.02 ms
# video prefill over 58 launches -- ~9.4 ms of the 51 ms of GPU on episode_10step.
# Every one of those launches is a *tiny* problem: 24 heads x Sq in {32, 98} x
# Sk in {98, 129, 130} x head_dim 128.  At 14.5 us/launch that is ~5 GB/s per CU
# and ~3 TFLOP/s -- it is neither bandwidth- nor MFMA-bound, it is one long
# latency chain issued at 24-CTA occupancy.
#
# WHAT.  Because Sk <= 256 here, the whole attention fits in ONE pass: no online
# softmax, no K/V loop.  A single Triton kernel loads K and V whole, does
# QK^T -> mask -> softmax(fp32) -> PV, and -- crucially -- reads and writes the
# [B, S, H*Dh] layout the rest of the stack already uses, so the two `rearrange`
# round-trips (and the `_scaled_dot_product_efficient_attention` layout-copy
# kernels inductor emits for them) disappear as well.  Tiling BLOCK_M = 16
# instead of one CTA per head raises the grid from 24 to 48/168 CTAs, which is
# where nearly all of the win comes from.
#
# Standalone, CUDA-graph timed (so the numbers are serialized GPU time, not the
# inflated back-to-back-launch numbers a naive loop reports):
#     shape                     aten attn_fwd (in-model)   this kernel
#     denoise self  32 x 130            14.5 us              10.25 us
#     denoise cross 32 x 129            14.5 us              10.27 us
#     prefill self  98 x  98            13.0 us               6.00 us
#     prefill cross 98 x 129            13.0 us              10.42 us
#
# SAFETY.  Strictly a fast path: anything outside the verified envelope (batch
# > 1, head_dim not 64/128, Sk > 256, a per-head mask, a non-unit last stride,
# a non-bf16/fp16 dtype, Triton missing) falls through to the original
# `F.scaled_dot_product_attention` call, which is left byte-for-byte in place.
# ===========================================================================

def _fa_fast_enabled() -> bool:
    return os.environ.get("FASTWAM_FUSED_ATTN", _FA_DEFAULT).lower() not in ("0", "false", "no", "off")


# `run_local.sh` forwards only GATE_*/FWE_*/ORACLE_*, so no FASTWAM_* name
# reaches the container: this literal is the switch that is actually flipped
# between A/B arms.
_FA_DEFAULT = "1"

try:
    import triton
    import triton.language as tl
    _FA_TRITON = True
except Exception:  # pragma: no cover
    _FA_TRITON = False


# ---------------------------------------------------------------------------
# ROUND-2 arm kill-switch.  `run_local.sh` / `run_profile.sh` forward every set
# FASTWAM_* into the container, so `FASTWAM_ATTN_PRELOAD=0` restores the exact
# shipped load order in BOTH attention kernels (the constexpr collapses the new
# code away at trace time -- the OFF arm compiles to the byte-identical body).
#
# MEASURED, ROUND 2 -- what this arm is and is NOT.  The two knobs below are the
# only two bit-exact per-launch levers that survived a sweep of the whole launch
# axis on the chained-graph-replay bench (30 links, median of 60 replays, 30
# DISTINCT K/V pairs so nothing sits hot in L2) plus the PROFILE leg:
#
#   lever                          den_self 32x130   verdict
#   shipped                              5.39 us     --
#   PRELOAD + FULL_MAIN (this arm)       5.30 us     kept
#   num_warps 2 / 8                 5.90 / 6.57 us   worse (table says 4)
#   BLOCK_M 32 / 64                 7.02 / 7.27 us   much worse (re-confirms
#                                                    the closed BLOCK_M retile)
#   launch num_stages=2                  4.97 us     no-op (no loop to pipeline)
#   waves_per_eu 1 / 2              4.96 / 5.00 us   no-op
#
# The direction's 18.2 -> 13 us target is NOT reachable from inside these two
# kernels: a diagnostic build that DELETES the whole RMSNorm reduction AND the
# whole tail key tile from `_fw_qrms_attn_kernel` only takes it 10.05 -> 8.43 us
# in the denoise step and moved the denoise_step WALL by 0.00 ms.  The residue is
# the ~4.34-4.64 us per-dispatch device floor this box charges every kernel
# regardless of CTA count or byte count -- these kernels are ~50% floor, and the
# remedy for that class is FEWER LAUNCHES, which is a mot.py lever.
_FA_FAST = os.environ.get("FASTWAM_ATTN_FAST", "1").lower() \
    not in ("0", "false", "no", "off")
# `PRELOAD`: issue every global load (K/V main tile, K/V tail tile, both mask
# tiles) in ONE basic block before any dependent arithmetic.
_FA_PRELOAD = int(_FA_FAST)
# `FULL_MAIN`/`EXACT_M`: launch-site constexprs asserting Sk >= BLOCK_N and
# BLOCK_M | Sq, which lets the always-true load predicates and the always-true
# `tl.where(-inf)` on the main tile be dropped.
_FA_FULLMAIN = int(_FA_FAST)


# ===========================================================================
# ROUND-3 ARM `R3` -- in-kernel score-path rework for the attention FAMILY.
# Kill-switch: `FASTWAM_ATTN_R3=0` restores the byte-identical shipped body
# (every knob below is a `tl.constexpr`, so the OFF arm traces to the old code).
#
# Four independent, individually-priceable levers, all inside the two kernels:
#
#  AMASK  the {0, -inf} additive bias that `mot.py::_attn_bias` already hands us
#         is ADDED to the scores instead of driving `cmp -> and -> select`.
#         3 VALU ops per score element -> 1.  BIT-EXACT for MASK_BIAS masks
#         (0 + s == s, -inf + s == -inf); the bool-mask path is untouched.
#  EXP2   `tl.exp(x)` on gfx950 is `v_mul_f32 (x*log2e)` + `v_exp_f32`.  Folding
#         log2(e) into the (scalar) `scale` deletes that multiply per element.
#  RECIP  `p / den` is a full-precision fp32 divide on every score element
#         (~10 VALU each).  One reciprocal per ROW plus a multiply per element
#         instead.  ~1 ulp fp32, four orders under the bf16 store quantum.
#  DEFER  (qrms only, the structural one)  RMSNorm's `rstd` is a PER-ROW SCALAR,
#         so  (x*rstd*w) @ k^T == rstd * ((x*w) @ k^T).  Moving the multiply to
#         after the QK dot takes the whole HD=3072 sum-of-squares reduction OFF
#         the dot's dependency chain -- the MFMA issues while the reduction's
#         L2 round trips are still in flight, instead of after all of them.
#         Numerically it is the same bf16 quantisation applied to `x*w` rather
#         than to `x*rstd*w`; identical relative error, different rounding.
#  POSTN  normalise the OUTPUT by 1/den after the PV dot instead of normalising
#         `p` before it (fewer elements, shorter chain, one bf16 ulp).
# ===========================================================================
_FA_R3 = int(os.environ.get("FASTWAM_ATTN_R3", "1"))
_FA_AMASK = int(os.environ.get("FASTWAM_ATTN_AMASK", "1"))
_FA_EXP2 = int(os.environ.get("FASTWAM_ATTN_EXP2", "1"))
_FA_RECIP = int(os.environ.get("FASTWAM_ATTN_RECIP", "1"))
_FA_DEFER = int(os.environ.get("FASTWAM_ATTN_DEFER", "0"))
_FA_POSTN = int(os.environ.get("FASTWAM_ATTN_POSTN", "1"))
_LOG2E = 1.4426950408889634
# DIAGNOSTIC ONLY (default 0 = inert).  Each one DELETES one candidate cost from
# the NEW score path so it can be priced in isolation on the chained bench:
#   DMAX  skip the row-max cross-warp reduction   DDEN  skip the softmax denominator
#   DTAIL skip the whole tail key tile            DDOT  skip the PV dot
_FA_DMAX = int(os.environ.get("FASTWAM_ATTN_DMAX", "0"))
_FA_DDEN = int(os.environ.get("FASTWAM_ATTN_DDEN", "0"))
_FA_DTAIL = int(os.environ.get("FASTWAM_ATTN_DTAIL", "0"))
_FA_DRED = int(os.environ.get("FASTWAM_ATTN_DRED", "0"))
# TMAX: softmax is shift-invariant for ANY per-row constant, so the stabiliser
# does NOT have to be the exact row max.  Taking it from the MAIN key tile only
# deletes one cross-warp row reduction AND the [BLOCK_M,TAIL_N] -> [BLOCK_M,
# BLOCK_N] layout conversion that `tl.maximum(max(s), max(s2))` forces.
# TSEP: sum the two PV dots instead of chaining the second onto the first's
# accumulator (different layout constraint on `o`).
_FA_TMAX = int(os.environ.get("FASTWAM_ATTN_TMAX", "0"))
_FA_TSEP = int(os.environ.get("FASTWAM_ATTN_TSEP", "0"))


# ===========================================================================
# ROUND-2 (v5) ARM `XCD` -- XCD-LOCAL GRID SWIZZLE for the attention family.
# Kill-switch: `FASTWAM_ATTN_XCD=0` restores the byte-identical shipped body
# (the swizzle is a `tl.constexpr`, so the OFF arm traces to the old code).
#
# WHY.  These kernels are K/V-LOAD-LATENCY bound (measured: replacing the two
# K/V tiles with constants takes den_self 5.83 -> 2.74 us).  The grid is
# (Sq/BLOCK_M, H) and EVERY m-tile CTA of a given head loads the SAME K/V tile
# (72 KB at Sk=130) -- 4 CTAs/head in denoise, 7 in prefill.  MI355X is 8 XCD
# chiplets with a PRIVATE 4 MiB L2 each, and the HWS round-robins workgroups
# across XCDs by linear block id, so with grid0 = 4 the four m-tiles of one head
# land on FOUR DIFFERENT XCDs: each one misses its own L2 and pulls the same
# 72 KB over the fabric.  Total demand is 96 x 72 KB = 6.9 MB for 1.6 MB of
# unique bytes.
#
# WHAT.  Permute the (m, h) -> hardware-block-id mapping so that all CTAs
# sharing a head land on the SAME XCD (`new = (lin % XCD) * (NT/XCD) + lin/XCD`,
# the documented CDNA3/4 XCD remap).  One XCD then holds NT/(XCD*GM) whole heads,
# the first CTA's miss fills that XCD's L2, and its 3 (denoise) / 6 (prefill)
# siblings hit L2 instead of the fabric.
#
# BIT-EXACT BY CONSTRUCTION: it is a permutation of which CTA computes which
# (m, h) output tile.  Every tile is still computed exactly once, by the same
# code, over the same operands, in the same order.  Nothing is reduced across
# CTAs.  Requires NT % XCD == 0 and (NT/XCD) % GM == 0 so the remap is a
# bijection AND whole heads stay together; the launcher passes XCD=0 otherwise.
_FA_XCD = int(os.environ.get("FASTWAM_ATTN_XCD", "8"))


# ===========================================================================
# ROUND-2 (v5) ARM `DSPL` -- SPLIT THE OUTPUT (head-dim) AXIS ACROSS CTAs.
# Kill-switch: `FASTWAM_ATTN_DSPL=1` restores the byte-identical shipped body.
#
# THE MEASUREMENT THIS IS BUILT ON (chained bench, 30 distinct K/V, den 32x130):
#   key slots per CTA   72 KB->  144   80    64    32
#   us                          5.81  4.47  3.74  3.19
# i.e. cost = 2.44 us + 0.0234 us per key slot (512 B of K+V), and DOUBLING THE
# CTA COUNT AT CONSTANT PER-CTA BYTES IS FREE (96 CTAs x 64 slots = 3.74 us,
# 192 CTAs x 64 slots = 3.79 us).  The per-CTA delivery rate is pinned at
# ~22 GB/s -- 10 B/clk against a 64 B/clk L1 -- because at 96 CTAs on 256 CUs
# every CU holds ONE workgroup, four waves, one per SIMD, and there is nothing
# to hide a miss behind.  The lever is therefore FEWER BYTES PER CTA, MORE CTAs.
#
# The direction's KEY-axis split does that but needs a cross-CTA flash combine.
# The OUTPUT-COLUMN axis does it for FREE: `o[:, c]` for a column block c needs
# the full `p` (hence all of K) but only `V[:, c]`, and the column blocks are
# DISJOINT, so there is nothing to reduce -- no lock, no atomic, no partial
# workspace, no second launch.  Per-CTA bytes go 144*(K+V) -> 144*(K + V/DSPL).
# Measured cost of the redundant score path is inside the 2.44 us floor.
#
# BIT-EXACT BY CONSTRUCTION.  Every CTA recomputes the SAME `p` from the same
# operands with the same instruction sequence (softmax reduces along the KEY
# axis, which is not split), and the PV dot's reduction axis is also the key
# axis, tiled identically -- only its N extent shrinks.  Verified max|d| = 0
# against the DSPL=1 body on all four in-model shapes.
_FA_DSPL = int(os.environ.get("FASTWAM_ATTN_DSPL", "2"))
_FA_DSPL_QR = int(os.environ.get("FASTWAM_ATTN_DSPLQ", "2"))


# Number of CUs the split is allowed to fill.  The whole point of `DSPL` is to
# put a SECOND workgroup's worth of memory-level parallelism on a CU that is
# holding exactly one, so it only pays while the split grid still fits in one
# pass over the part.  MEASURED, chained bench, XCD swizzle on:
#   shape             CTAs 1 -> 2      DSPL=1    DSPL=2
#   den self 32x130     96 -> 192       5.72      5.59   <- keep
#   den cross 32x129    96 -> 192       6.74      6.50   <- keep
#   pre self  98x98    168 -> 336       4.95      5.54   <- reject (oversubscribed)
#   pre cross 98x129   168 -> 336       6.32      6.83   <- reject
_FA_DSPL_MAXCTA = int(os.environ.get("FASTWAM_ATTN_DSPLCTA", "256"))

# DIAGNOSTIC ONLY (default 0 = inert; the constexpr collapses it away).  `DKV=1`
# replaces the K/V tiles with CONSTANTS -- a WRONG answer that exists only to
# price the K/V load IN SITU, inside the real captured graph, per the
# COMMANDMENT's "strip the body and read `wall_ms_no_profiler`" rule.
_FA_DKV = int(os.environ.get("FASTWAM_ATTN_DKV", "0"))


def _fa_dspl(D: int, qrms: int = 0, ncta: int = 0) -> int:
    """Output-column split factor for head_dim `D` (1 = shipped single-CTA form).

    `ncta` is the UNSPLIT workgroup count; the split is refused when it would
    oversubscribe the part (see `_FA_DSPL_MAXCTA`).
    """
    d = _FA_DSPL_QR if qrms else _FA_DSPL
    if d <= 1:
        return 1
    while d > 1 and (D % d or (D // d) < 32):
        d //= 2
    while d > 1 and ncta > 0 and ncta * d > _FA_DSPL_MAXCTA:
        d //= 2
    return max(d, 1)


def _fa_xcd_kw(grid) -> dict:
    """(XCD, GM, NT) constexprs for the XCD-local swizzle, or {} when illegal."""
    x = _FA_XCD
    if x <= 1:
        return {}
    gm = int(grid[0])
    nt = gm * int(grid[1])
    if gm <= 0 or nt % x or (nt // x) % gm:
        return {}
    return {"XCD": x, "GM": gm, "NT": nt}


if _FA_TRITON:

    @triton.jit
    def _fw_attn_kernel(Q, K, V, MSK, O, Sq, Sk,
                        qh, qs, kh, ks, vh, vs, oh, os_, sm_s, scale,
                        D: tl.constexpr, BLOCK_M: tl.constexpr,
                        BLOCK_N: tl.constexpr, HAS_MASK: tl.constexpr,
                        MASK_BIAS: tl.constexpr, TAIL_N: tl.constexpr = 0,
                        PRELOAD: tl.constexpr = _FA_PRELOAD,
                        FULL_MAIN: tl.constexpr = 0, EXACT_M: tl.constexpr = 0,
                        R3: tl.constexpr = _FA_R3,
                        AMASK: tl.constexpr = _FA_AMASK,
                        EXP2: tl.constexpr = _FA_EXP2,
                        RECIP: tl.constexpr = _FA_RECIP,
                        POSTN: tl.constexpr = _FA_POSTN,
                        DMAX: tl.constexpr = _FA_DMAX,
                        DDEN: tl.constexpr = _FA_DDEN,
                        DTAIL: tl.constexpr = _FA_DTAIL,
                        TMAX: tl.constexpr = _FA_TMAX,
                        TSEP: tl.constexpr = _FA_TSEP,
                        XCD: tl.constexpr = 0, GM: tl.constexpr = 0,
                        NT: tl.constexpr = 0, DSPL: tl.constexpr = 1,
                        DKV: tl.constexpr = _FA_DKV):
        """Single-pass short-sequence attention.

        ROUND-3 CHANGE -- SPLIT KEY TILE (`TAIL_N`).  `tl.arange` needs a power of
        two, so a key length of 129/130 (denoise self+cross, prefill cross) used to
        round up to BLOCK_N=256: HALF the K/V tile, half the QK^T MFMAs and half the
        exp()s were spent on columns that are masked off.  That single fact -- not
        occupancy -- was what made this kernel slow: the one shape that fits a 128
        tile (prefill self, Sk=98) already ran at 5.6 us while every Sk>128 shape ran
        at 9.7-10.1 us with the SAME CTA count.

        So the key axis is now covered by a main tile of BLOCK_N plus an optional
        small tail tile of TAIL_N (129 -> 128+16 instead of 256).  It is
        BIT-IDENTICAL to the single-tile form, by construction and verified
        (max|d| = 0 over all four in-model shapes):
          * `tl.max` is order-independent;
          * the softmax denominator is `sum(main) + sum(tail)`, which is exactly the
            top node of the binary tree the 256-wide `tl.sum` builds (the columns
            between Sk and BLOCK_N+TAIL_N contribute exp(-inf) = +0.0);
          * `p` is divided by the denominator BEFORE the bf16 cast and BEFORE P@V,
            exactly as the single-tile form does -- hence the accumulator chain sees
            the same bf16 inputs, and the tail dot continues the SAME fp32
            accumulator rather than being summed in afterwards.
        Normalising after the dot instead is ~9% faster but perturbs the output by
        one bf16 ulp; that is not worth spending against an oracle gate already at
        ~90% consumed, so the exact ordering is what ships.
        """
        pid_m = tl.program_id(0)
        h = tl.program_id(1)
        if XCD > 1:
            # XCD-local remap (see `_FA_XCD`).  Pure permutation of tile owners.
            lin = pid_m + GM * h
            per: tl.constexpr = NT // XCD
            nlin = (lin % XCD) * per + (lin // XCD)
            pid_m = nlin % GM
            h = nlin // GM
        # `DSPL` (see `_FA_DSPL`): grid axis 0 carries m-tile MAJOR, column-block
        # MINOR, so the CTAs sharing a head's K/V stay adjacent for the XCD remap.
        DV: tl.constexpr = D // DSPL
        if DSPL > 1:
            dj = pid_m % DSPL
            pid_m = pid_m // DSPL
            ov = dj * DV + tl.arange(0, DV)
        else:
            ov = tl.arange(0, DV)
        om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        on = tl.arange(0, BLOCK_N)
        od = tl.arange(0, D)
        # ROUND-2 `FULL_MAIN` / `EXACT_M`: launch-site constexprs that say the
        # MAIN key tile is entirely in range (Sk >= BLOCK_N) and that BLOCK_M
        # divides Sq.  Both are shape facts the kernel could not see before, so
        # every load carried a predicate and the main tile paid a full
        # [BLOCK_M, BLOCK_N] `tl.where(-inf)` that could never fire.  Removing a
        # predicate that is provably always true is BIT-EXACT.
        mm = om < Sq
        mn = on < Sk
        # ROUND-3: the reworked score path needs the {0,-inf} ADDITIVE bias form
        # of the mask; a bool mask (FASTWAM_MASKBIAS=0) keeps the shipped body.
        NEW: tl.constexpr = R3 and HAS_MASK and MASK_BIAS and PRELOAD
        # AMASK level: 0 off, 1 always, 2 only on the single-tile (TAIL_N == 0)
        # shapes -- the fp32 bias plane costs registers that the two-tile shapes
        # cannot afford (measured: helps 98x98 by 0.24 us, costs the tail shapes).
        AM: tl.constexpr = NEW and (AMASK == 1 or (AMASK == 2 and TAIL_N == 0))
        if EXACT_M:
            q = tl.load(Q + h * qh + om[:, None] * qs + od[None, :])
        else:
            q = tl.load(Q + h * qh + om[:, None] * qs + od[None, :], mask=mm[:, None], other=0.0)
        if DKV:
            k = tl.full([BLOCK_N, D], 0.01, q.dtype)
        elif FULL_MAIN:
            k = tl.load(K + h * kh + on[:, None] * ks + od[None, :])
        else:
            k = tl.load(K + h * kh + on[:, None] * ks + od[None, :], mask=mn[:, None], other=0.0)
        # ---- ROUND-2 `PRELOAD`: issue EVERY global load in one basic block ---
        # Pure re-ordering of loads.  Not one arithmetic op below is touched, so
        # this is BIT-IDENTICAL by construction (verified max|d| = 0).  The
        # shipped order was  load k -> dot -> load k2 -> dot -> softmax ->
        # load v -> load v2 -> dot: FOUR serialised L2 round trips on a kernel
        # that runs 48 CTAs on 256 CU (1 CTA/CU, 1 wave/SIMD) with nothing else
        # resident to hide them.  `_fw_qrms_attn_kernel` already hoisted k+v for
        # exactly this reason and never hoisted the tail; here nothing was
        # hoisted at all.  VGPRs are free at this occupancy.
        if PRELOAD:
            if DKV:
                v = tl.full([BLOCK_N, DV], 0.01, q.dtype)
            elif FULL_MAIN:
                v = tl.load(V + h * vh + on[:, None] * vs + ov[None, :])
            else:
                v = tl.load(V + h * vh + on[:, None] * vs + ov[None, :], mask=mn[:, None], other=0.0)
            if TAIL_N > 0:
                on2 = BLOCK_N + tl.arange(0, TAIL_N)
                mn2 = on2 < Sk
                if DKV:
                    k2 = tl.full([TAIL_N, D], 0.01, K.dtype.element_ty)
                    v2 = tl.full([TAIL_N, DV], 0.01, K.dtype.element_ty)
                else:
                    k2 = tl.load(K + h * kh + on2[:, None] * ks + od[None, :],
                                 mask=mn2[:, None], other=0.0)
                    v2 = tl.load(V + h * vh + on2[:, None] * vs + ov[None, :],
                                 mask=mn2[:, None], other=0.0)
            if HAS_MASK:
                if AM:
                    # `_attn_bias` guarantees exactly {0, -inf}, so `s + bias` is
                    # BIT-EXACT with `where(bias == 0, s, -inf)`.  Loading the
                    # out-of-range columns as -inf folds the `mn` range predicate
                    # into the same add (again bit-exact).
                    if FULL_MAIN and EXACT_M:
                        mk = tl.load(MSK + om[:, None] * sm_s + on[None, :]).to(tl.float32)
                    else:
                        mk = tl.load(MSK + om[:, None] * sm_s + on[None, :],
                                     mask=mm[:, None] & mn[None, :],
                                     other=float("-inf")).to(tl.float32)
                    if TAIL_N > 0:
                        mk2 = tl.load(MSK + om[:, None] * sm_s + on2[None, :],
                                      mask=mm[:, None] & mn2[None, :],
                                      other=float("-inf")).to(tl.float32)
                else:
                    mk = tl.load(MSK + om[:, None] * sm_s + on[None, :],
                                 mask=mm[:, None] & mn[None, :], other=0)
                    if TAIL_N > 0:
                        mk2 = tl.load(MSK + om[:, None] * sm_s + on2[None, :],
                                      mask=mm[:, None] & mn2[None, :], other=0)
        if NEW:
            if EXP2:
                qk = scale * 1.4426950408889634
            else:
                qk = scale
            s = tl.dot(q, tl.trans(k)) * qk
            if AM:
                s = s + mk
            else:
                keep = mn[None, :] & (mk == 0)
                s = tl.where(keep, s, float("-inf"))
            if TAIL_N > 0 and not DTAIL:
                s2 = tl.dot(q, tl.trans(k2)) * qk
                if AM:
                    s2 = s2 + mk2
                else:
                    s2 = tl.where(mn2[None, :] & (mk2 == 0), s2, float("-inf"))
                if DMAX:
                    mx = tl.zeros([BLOCK_M, 1], tl.float32)
                elif TMAX:
                    mx = tl.max(s, 1)[:, None]
                else:
                    mx = tl.maximum(tl.max(s, 1), tl.max(s2, 1))[:, None]
                if EXP2:
                    p = tl.exp2(s - mx)
                    p2 = tl.exp2(s2 - mx)
                else:
                    p = tl.exp(s - mx)
                    p2 = tl.exp(s2 - mx)
                if DDEN:
                    den = tl.full([BLOCK_M, 1], 1.0, tl.float32)
                else:
                    den = (tl.sum(p, 1) + tl.sum(p2, 1))[:, None]
                if POSTN:
                    if TSEP:
                        o = tl.dot(p.to(q.dtype), v) + tl.dot(p2.to(q.dtype), v2)
                    else:
                        o = tl.dot(p.to(q.dtype), v)
                        o = tl.dot(p2.to(q.dtype), v2, o)
                    if RECIP:
                        o = o * (1.0 / den)
                    else:
                        o = o / den
                else:
                    if RECIP:
                        rden = 1.0 / den
                        o = tl.dot((p * rden).to(q.dtype), v)
                        o = tl.dot((p2 * rden).to(q.dtype), v2, o)
                    else:
                        o = tl.dot((p / den).to(q.dtype), v)
                        o = tl.dot((p2 / den).to(q.dtype), v2, o)
            else:
                if DMAX:
                    mx = tl.zeros([BLOCK_M, 1], tl.float32)
                else:
                    mx = tl.max(s, 1)[:, None]
                if EXP2:
                    p = tl.exp2(s - mx)
                else:
                    p = tl.exp(s - mx)
                if DDEN:
                    den = tl.full([BLOCK_M, 1], 1.0, tl.float32)
                else:
                    den = tl.sum(p, 1)[:, None]
                if POSTN:
                    o = tl.dot(p.to(q.dtype), v)
                    if RECIP:
                        o = o * (1.0 / den)
                    else:
                        o = o / den
                else:
                    if RECIP:
                        p = p * (1.0 / den)
                    else:
                        p = p / den
                    o = tl.dot(p.to(q.dtype), v)
            if EXACT_M:
                tl.store(O + h * oh + om[:, None] * os_ + ov[None, :], o.to(q.dtype))
            else:
                tl.store(O + h * oh + om[:, None] * os_ + ov[None, :], o.to(q.dtype),
                         mask=mm[:, None])
        else:
            s = tl.dot(q, tl.trans(k)) * scale
            keep = mn[None, :]
            if HAS_MASK:
                if not PRELOAD:
                    mk = tl.load(MSK + om[:, None] * sm_s + on[None, :],
                                 mask=mm[:, None] & mn[None, :], other=0)
                if MASK_BIAS:
                    # INTEGRATION (round-2 hand-merge): mot.py's `_attn_bias` hoists
                    # the bool->additive-bias conversion out of the 30-layer loop, so
                    # the mask that reaches here is an exactly-{0, -inf} float bias
                    # rather than a bool.  `bias == 0` is the same predicate as
                    # `bool_mask != 0`, so this branch is bit-identical to the other.
                    keep = keep & (mk == 0)
                else:
                    keep = keep & (mk != 0)
            if HAS_MASK or not FULL_MAIN:
                s = tl.where(keep, s, float("-inf"))
            if TAIL_N > 0:
                if not PRELOAD:
                    on2 = BLOCK_N + tl.arange(0, TAIL_N)
                    mn2 = on2 < Sk
                    k2 = tl.load(K + h * kh + on2[:, None] * ks + od[None, :],
                                 mask=mn2[:, None], other=0.0)
                s2 = tl.dot(q, tl.trans(k2)) * scale
                keep2 = mn2[None, :]
                if HAS_MASK:
                    if not PRELOAD:
                        mk2 = tl.load(MSK + om[:, None] * sm_s + on2[None, :],
                                      mask=mm[:, None] & mn2[None, :], other=0)
                    if MASK_BIAS:
                        keep2 = keep2 & (mk2 == 0)
                    else:
                        keep2 = keep2 & (mk2 != 0)
                s2 = tl.where(keep2, s2, float("-inf"))
                mx = tl.maximum(tl.max(s, 1), tl.max(s2, 1))[:, None]
                p = tl.exp(s - mx)
                p2 = tl.exp(s2 - mx)
                den = (tl.sum(p, 1) + tl.sum(p2, 1))[:, None]
                p = p / den
                p2 = p2 / den
                if not PRELOAD:
                    if FULL_MAIN:
                        v = tl.load(V + h * vh + on[:, None] * vs + ov[None, :])
                    else:
                        v = tl.load(V + h * vh + on[:, None] * vs + ov[None, :], mask=mn[:, None], other=0.0)
                    v2 = tl.load(V + h * vh + on2[:, None] * vs + ov[None, :], mask=mn2[:, None], other=0.0)
                o = tl.dot(p.to(q.dtype), v)
                o = tl.dot(p2.to(q.dtype), v2, o)
            else:
                if not PRELOAD:
                    if FULL_MAIN:
                        v = tl.load(V + h * vh + on[:, None] * vs + ov[None, :])
                    else:
                        v = tl.load(V + h * vh + on[:, None] * vs + ov[None, :], mask=mn[:, None], other=0.0)
                s = s - tl.max(s, 1)[:, None]
                p = tl.exp(s)
                p = p / tl.sum(p, 1)[:, None]
                o = tl.dot(p.to(q.dtype), v)
            if EXACT_M:
                tl.store(O + h * oh + om[:, None] * os_ + ov[None, :], o.to(q.dtype))
            else:
                tl.store(O + h * oh + om[:, None] * os_ + ov[None, :], o.to(q.dtype),
                         mask=mm[:, None])


# ---------------------------------------------------------------------------
# ROUND-3 LEVER: BLOCK_M DOWNWARD RETILE (16 -> 8) for the SHORT (Sq=32) shapes.
#
# WHY.  Both attention kernels run grid=(Sq/BLOCK_M, H) = (2, 24) = 48 CTAs on a
# 256-CU part -- 18.75% fill, the only fill-starved kernel class left in the run.
# The UPWARD retile (BLOCK_M 32/64) is closed twice (7.02 / 7.27 us vs 5.39): it
# moves the grid the WRONG way (48 -> 24).  DOWNWARD doubles it to 96 CTAs and had
# never been tried.  COMMANDMENT rule 7 bans BLOCK_N < 64 at num_warps=4; it does
# NOT constrain BLOCK_M.
#
# Numerics: no reduction whose ORDER changes -- softmax reduces along the KEY axis
# per row and rows are independent of the M tiling; the QK^T / PV dots accumulate
# over D / BLOCK_N, neither of which moves.  Only the MFMA M-tile the backend picks
# can change, so this is verified on the oracle rather than asserted bit-exact.
#
# Guarded to Sq <= _FA_BM_MAX_SQ so the video-prefill shape (Sq=98, 7 -> 13 CTAs
# per head, and 98 % 8 != 0 so EXACT_M stays off) is untouched.
_FA_BM = int(os.environ.get("FASTWAM_ATTN_BM", "8"))
_FA_BM_MAX_SQ = int(os.environ.get("FASTWAM_ATTN_BM_MAXSQ", "32"))


_FA_BM_NW = int(os.environ.get("FASTWAM_ATTN_BM_NW", "0"))


# ROUND-3: the M-tile is now per-KERNEL, not per-family.  With the reworked score
# path the two kernels want DIFFERENT tiles on the same Sq=32 denoise shape --
# measured on the chained bench (median of 60 replays, 30 distinct K/V):
#   BLOCK_M    _fw_attn (32x130)   _fw_qrms_attn (32x129)
#      8            6.79 us              7.71 us   <- round-2 setting, both
#     16            5.92 us              8.03 us
# The split is mechanical: at BLOCK_M=8 the MFMA cannot tile M, so the backend
# splits the 4 warps along N and BOTH softmax row-reductions become cross-warp
# (deleting them is worth 1.06 us at BM=8 but only 0.24 us at BM=16).  `_fw_attn`
# therefore wants 16.  `_fw_qrms_attn` does not, because doubling BLOCK_M also
# doubles the rows of the HD=3072 sum-of-squares its CTA has to stream.
_FA_BM_ATTN = int(os.environ.get("FASTWAM_ATTN_BMA", "8"))


def _fa_block_m(Sq: int, qrms: int = 0) -> int:
    """M-tile for the attention family.  16 everywhere except the short shapes.

    ROUND-2 (v5): values ABOVE 16 are now honoured too.  The whole M-tile table
    in this file was measured on the chained graph-replay bench, which is
    L2-hot; in situ the attention kernels re-fetch a head's WHOLE K/V once per
    m-tile CTA (4x in denoise, 7x in prefill) out of another XCD's dirty L2, so
    the in-situ optimum can sit on the opposite side of the microbench's.
    """
    bm = _FA_BM if qrms else _FA_BM_ATTN
    if bm > 16:
        while bm > 16 and bm > Sq:
            bm //= 2
        return bm
    if bm < 16 and Sq <= _FA_BM_MAX_SQ:
        return bm
    return 16


def _fa_nw(BLOCK_M: int, nw: int) -> int:
    """num_warps.  `_fa_key_tile`'s value stands unless the M-tile was retiled
    down, in which case the half-height MFMA tile may want fewer lanes."""
    if BLOCK_M < 16 and _FA_BM_NW > 0:
        return _FA_BM_NW
    return nw


_FA_TN = int(os.environ.get("FASTWAM_ATTN_TN", "0"))
_FA_NW = int(os.environ.get("FASTWAM_ATTN_NW", "0"))
_FA_NW128 = int(os.environ.get("FASTWAM_ATTN_NW128", "2" if _FA_R3 else "4"))
_FA_BN256 = int(os.environ.get("FASTWAM_ATTN_BN256", "0"))
_FA_MFMA = int(os.environ.get("FASTWAM_ATTN_MFMA", "16"))
_FA_WPE = int(os.environ.get("FASTWAM_ATTN_WPE", "0"))
_FA_KPACK = int(os.environ.get("FASTWAM_ATTN_KPACK", "0"))


def _fa_extra_kw() -> dict:
    """gfx950 backend launch hints (empty = backend default).

    Part of the `FASTWAM_ATTN_R3` arm, so the OFF arm re-measures the exact
    shipped launch configuration.
    """
    kw = {}
    if not _FA_R3:
        return kw
    if _FA_MFMA:
        kw["matrix_instr_nonkdim"] = _FA_MFMA
    if _FA_WPE:
        kw["waves_per_eu"] = _FA_WPE
    if _FA_KPACK:
        kw["kpack"] = _FA_KPACK
    return kw


class _FaHinted:
    """`[grid](...)` proxy that injects the gfx950 backend launch hints.

    `mot.py::fastwam::short_attn_bsd` imports `_fw_attn_kernel` and launches it
    itself (that is the DENOISE self-attention, Sq=32/Sk=130).  `mot.py` belongs
    to another owner this round, so `matrix_instr_nonkdim` has to travel WITH the
    kernel object rather than with the call site.  Measured on the chained bench:
    the hint is worth 6.57 -> 5.62 us there and 7.62 -> 6.70 us on the cross
    kernel; `matrix_instr_nonkdim=32` is worse on every shape.
    """

    __slots__ = ("_k",)

    def __init__(self, k):
        self._k = k

    def __getitem__(self, grid):
        k = self._k

        def _run(*args, **kwargs):
            for hk, hv in _fa_extra_kw().items():
                kwargs.setdefault(hk, hv)
            # The DSPL / XCD decisions have to travel WITH the kernel object for
            # the same reason `matrix_instr_nonkdim` does: `mot.py::short_attn_bsd`
            # (the denoise self-attention, 30 launches/step) launches this kernel
            # itself and belongs to another owner this round.  `D` is positional
            # argument 17 at every call site, so the grid can only be finalised
            # here, once the arguments are known.
            g = grid
            if "DSPL" not in kwargs and len(args) > 17:
                nc = 1
                for gv in grid:
                    nc *= int(gv)
                d = _fa_dspl(int(args[17]), ncta=nc)
                if d > 1:
                    kwargs["DSPL"] = d
                    g = (grid[0] * d,) + tuple(grid[1:])
            for hk, hv in _fa_xcd_kw(g).items():
                kwargs.setdefault(hk, hv)
            return k[g](*args, **kwargs)

        return _run


if _FA_TRITON:
    _fw_attn_kernel = _FaHinted(_fw_attn_kernel)


def _fa_key_tile(Sk: int):
    """(BLOCK_N, TAIL_N, num_warps) covering `Sk` keys with the least wasted tile.

    Graph-replay timings of a DEPENDENCY-CHAINED sequence (o feeds the next q, 30
    links, median of 60 replays -- a `for _ in range(N)` loop overlaps back-to-back
    launches on a 256-CU card and overstates this kernel by ~1.7x, so it is not
    used).  Chained no-op floor on this box is 1.96 us.

        shape                   old (BLOCK_N only)   this table
        denoise self  32x130     9.83 us (BN=256)     6.71 us (128+16)
        denoise cross 32x129     9.73 us (BN=256)     6.70 us (128+16)
        prefill cross 98x129    10.11 us (BN=256)     6.80 us (128+16)
        prefill self  98x 98      5.57 us (BN=128)    unchanged (BN=128)

    Sk <= 128 deliberately keeps the exact single-tile launch this file already
    shipped -- a 64+64 split measured 3% faster there but is NOT bit-exact, and 3%
    of the one shape that was already fast is not worth the numerics.
    """
    if _FA_BN256 and 128 < Sk <= 256:
        return 256, 0, (_FA_NW or _FA_BN256)
    if _FA_TN and 128 < Sk <= 128 + _FA_TN:
        return 128, _FA_TN, (_FA_NW or 4)
    if Sk <= 64:
        return 64, 0, (_FA_NW or _FA_NW128)
    if Sk <= 128:
        return 128, 0, (_FA_NW or _FA_NW128)
    if Sk <= 144:
        return 128, 16, (_FA_NW or 4)
    if Sk <= 160:
        return 128, 32, 4
    if Sk <= 192:
        return 128, 64, 4
    return 256, 0, 2


@torch.library.custom_op("fastwam::short_attn", mutates_args=())
def _fw_short_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   mask: Optional[torch.Tensor], num_heads: int) -> torch.Tensor:
    if q.dim() == 4:
        B, H, Sq, D = q.shape
        Sk = k.shape[2]
        qh, qs = q.stride(1), q.stride(2)
        kh, ks = k.stride(1), k.stride(2)
        vh, vs = v.stride(1), v.stride(2)
        o = torch.empty((B, H, Sq, D), dtype=q.dtype, device=q.device)
        oh, os_ = o.stride(1), o.stride(2)
    else:
        B, Sq, HD = q.shape
        H = num_heads
        D = HD // H
        Sk = k.shape[1]
        qh, qs = D, q.stride(1)
        kh, ks = D, k.stride(1)
        vh, vs = D, v.stride(1)
        o = torch.empty((B, Sq, HD), dtype=q.dtype, device=q.device)
        oh, os_ = D, o.stride(1)
    BLOCK_N, TAIL_N, nw = _fa_key_tile(Sk)
    BLOCK_M = _fa_block_m(Sq)
    nw = _fa_nw(BLOCK_M, nw)
    grid = ((Sq + BLOCK_M - 1) // BLOCK_M, H)
    _fw_attn_kernel[grid](
        q, k, v, mask if mask is not None else q, o, Sq, Sk,
        qh, qs, kh, ks, vh, vs, oh, os_,
        0 if mask is None else mask.stride(0),
        D ** -0.5, D, BLOCK_M, BLOCK_N, mask is not None,
        mask is not None and mask.dtype != torch.bool,
        TAIL_N=TAIL_N, PRELOAD=_FA_PRELOAD,
        FULL_MAIN=int(_FA_FULLMAIN and Sk >= BLOCK_N),
        EXACT_M=int(_FA_FULLMAIN and Sq % BLOCK_M == 0),
        num_warps=nw, num_stages=1,
    )
    return o


@_fw_short_attn.register_fake
def _(q, k, v, mask, num_heads):
    return q.new_empty(q.shape)


_FA_OK_D = (64, 128)


def _fa_mask2d(m, Sq: int, Sk: int):
    """Collapse an SDPA-broadcastable mask to a [Sq, Sk] plane, or None/False.

    Returns None for "no mask", False for "not representable -> fall back".
    """
    if m is None:
        return None
    # bool mask (pristine call sites) OR the {0, -inf} additive bias that
    # mot.py's `_attn_bias` hoists out of the per-layer loop (round-2 merge).
    if m.dtype != torch.bool and m.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if m.dim() == 4:
        if m.shape[0] != 1 or m.shape[1] != 1:
            return False
        m = m[0, 0]
    elif m.dim() == 3:
        if m.shape[0] != 1:
            return False
        m = m[0]
    elif m.dim() != 2:
        return False
    if tuple(m.shape) != (Sq, Sk) or m.stride(1) != 1:
        return False
    return m


# Below this query length the fast path is entered on the ALREADY-REARRANGED
# [B, H, S, Dh] tensors instead of on [B, S, H*Dh].  Measured reason (profile,
# 3 interleaved runs): feeding the fused kernel the flat layout deletes the SDPA
# layout-copy kernels, which is a 0.64 ms win on `video_prefill` (Sq = 98) -- but
# in the action denoise step (Sq = 32) it also strips the `permute` epilogue off
# the expert's fused-QKV GEMM template, and inductor then re-autotunes that GEMM
# from 11.5 us to 33 us per launch, costing +1.29 ms/step.  Keeping the
# rearranges for the short-query call sites preserves that epilogue.
_FA_FLAT_MIN_SQ = 64


def _fa_eligible(q, k, v, num_heads: int) -> bool:
    """Common envelope check; `q` may be [B, S, H*Dh] or [B, H, S, Dh]."""
    if not (_FA_TRITON and _fa_fast_enabled()):
        return False
    if q.dim() != k.dim() or q.dim() != v.dim():
        return False
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        return False
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        return False
    if q.dim() == 4:
        if q.shape[1] != num_heads or k.shape[1] != num_heads or v.shape[1] != num_heads:
            return False
        if q.shape[3] not in _FA_OK_D or k.shape[3] != q.shape[3] or v.shape[3] != q.shape[3]:
            return False
        return k.shape[2] == v.shape[2] and k.shape[2] <= 256
    if q.dim() != 3:
        return False
    HD = q.shape[2]
    if HD != k.shape[2] or HD != v.shape[2] or HD % num_heads:
        return False
    if (HD // num_heads) not in _FA_OK_D:
        return False
    return k.shape[1] == v.shape[1] and k.shape[1] <= 256


# ===========================================================================
# CROSS-ATTENTION Q-SIDE FUSION: absorb `norm_q`'s RMSNorm into the attention
# kernel's Q LOAD.
#
# WHY.  `q_c = ca.norm_q(ca.q(block.norm3(x)))` compiles to a standalone reduction
# kernel -- `triton_red_fused__to_copy_add_mean_mul_permute_pow_rsqrt_select_
# short_attn_bsd_slice_view_5`, xnumel=32 rnumel=3072, XBLOCK=1 so 32 CTAs =
# 12.5% of a 256-CU part -- that runs IN PLACE on the GEMM output buffer and
# moves 2 x 0.196 MB.  Measured 4.38 us x 30 layers = 0.131 ms/step of the
# 2.92 ms denoise step, against a chained no-op floor of ~1.96 us: it is a launch,
# not work.  The prefill pays the same kernel once per layer (4.5 us x 29).
#
# WHAT.  RMSNorm here normalises over the FULL attn_hidden_dim (H*Dh = 3072, all
# 24 heads), so it cannot be expressed as a per-head epilogue -- which is exactly
# why inductor emits it as its own reduction and why folding it into the GEMM
# would need a cross-CTA (atomic, non-deterministic) reduction over N.  The fused
# kernel below instead recomputes the per-row sum-of-squares REDUNDANTLY in each
# of the H head-CTAs: 16 rows x 3072 bf16 = 98 KB per CTA, all of it L2-resident
# (the whole q buffer is 196 KB and every CTA reads the same bytes).  That buys
# back a whole kernel launch for a few hundred ns of L2 traffic.
#
# NUMERICS.  Transcribed from inductor's generated body, not re-derived:
#   rstd = rsqrt(sum(x_f32 * x_f32) / HD + eps);  out = (x_f32 * rstd) * w_f32
# with the SINGLE bf16 rounding at the (here: register) store -- inductor drops
# the `.to(dtype)` in `RMSNorm.forward` (`tmp14 = tmp13.to(tl.float32)` is a
# no-op in the generated kernel), so the intermediate is NOT rounded.  The one
# non-identity is the fp32 summation TREE: inductor accumulates over a looped
# R0_BLOCK, this kernel over a `RB`-wide block.  Both fp32; ~1e-7 relative, four
# orders below the bf16 store quantum -- same class as the shipped
# `_fw_qkvpost_kernel`, and gated on `run_local.sh oracle` + the 14-episode
# rollout, not asserted bit-exact.
#
# INTEGRATION (COMMANDMENT rule 9).  The key-tile rule is NOT re-derived here:
# `_fa_key_tile(Sk)` is the single source of truth, and `TAIL_N` is a REQUIRED
# keyword at every launch site (no constexpr default, so a future copy cannot
# silently fall back to a 256-wide tile).  BLOCK_N < 64 at num_warps=4 is never
# produced by that table.
# ===========================================================================

if _FA_TRITON:

    @triton.jit
    def _fw_qrms_attn_kernel(Q, WQ, K, V, MSK, O, Sq, Sk,
                             qs, kh, ks, vh, vs, oh, os_, sm_s, scale, eps,
                             D: tl.constexpr, BLOCK_M: tl.constexpr,
                             BLOCK_N: tl.constexpr, HAS_MASK: tl.constexpr,
                             MASK_BIAS: tl.constexpr, HD: tl.constexpr,
                             RB: tl.constexpr, EXACT_M: tl.constexpr,
                             RSTAGES: tl.constexpr, TAIL_N: tl.constexpr,
                             PRELOAD: tl.constexpr = _FA_PRELOAD,
                             FULL_MAIN: tl.constexpr = 0,
                             R3: tl.constexpr = _FA_R3,
                             AMASK: tl.constexpr = _FA_AMASK,
                             EXP2: tl.constexpr = _FA_EXP2,
                             RECIP: tl.constexpr = _FA_RECIP,
                             DEFER: tl.constexpr = _FA_DEFER,
                             POSTN: tl.constexpr = _FA_POSTN,
                             DMAX: tl.constexpr = _FA_DMAX,
                             DDEN: tl.constexpr = _FA_DDEN,
                             DTAIL: tl.constexpr = _FA_DTAIL,
                             DRED: tl.constexpr = _FA_DRED,
                             TMAX: tl.constexpr = _FA_TMAX,
                             TSEP: tl.constexpr = _FA_TSEP,
                             XCD: tl.constexpr = 0, GM: tl.constexpr = 0,
                             NT: tl.constexpr = 0, DSPL: tl.constexpr = 1,
                             DKV: tl.constexpr = _FA_DKV):
        """`_fw_attn_kernel` with `RMSNorm(HD)` folded into the Q load.

        Q is the RAW [B, Sq, HD] projection output (row stride `qs`); this CTA
        owns head `h = program_id(1)`, i.e. columns [h*D, h*D+D).  Everything from
        `s = q @ k^T` onward is a line-for-line copy of `_fw_attn_kernel`.

        SCHEDULING.  Every global load -- K, V, the mask, the head tile of Q AND
        the HD/RB whole-row reduction tiles -- is issued in ONE basic block at the
        top, before any dependent arithmetic, and the reduction loop is a
        `tl.static_range` so it is fully unrolled into that block.  This matters
        more than anything else here: the kernel runs at 48 CTAs on a 256-CU part
        (1 CTA/CU, 1 wave/SIMD), so VGPRs are free and occupancy is not a
        constraint, while the ROLLED form serialises HD/RB dependent L2 round
        trips and costs ~2.7 us -- more than the 4.38 us launch this fusion
        removes.  Overlapped, the redundant read hides inside the latency the
        attention was already paying.
        """
        pid_m = tl.program_id(0)
        h = tl.program_id(1)
        if XCD > 1:
            # XCD-local remap (see `_FA_XCD`).  Pure permutation of tile owners.
            lin = pid_m + GM * h
            per: tl.constexpr = NT // XCD
            nlin = (lin % XCD) * per + (lin // XCD)
            pid_m = nlin % GM
            h = nlin // GM
        # `DSPL` (see `_FA_DSPL`): grid axis 0 carries m-tile MAJOR, column-block
        # MINOR, so the CTAs sharing a head's K/V stay adjacent for the XCD remap.
        DV: tl.constexpr = D // DSPL
        if DSPL > 1:
            dj = pid_m % DSPL
            pid_m = pid_m // DSPL
            ov = dj * DV + tl.arange(0, DV)
        else:
            ov = tl.arange(0, DV)
        om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        on = tl.arange(0, BLOCK_N)
        od = tl.arange(0, D)
        mm = om < Sq
        mn = on < Sk
        NEW: tl.constexpr = R3 and HAS_MASK and MASK_BIAS and PRELOAD
        AM: tl.constexpr = NEW and (AMASK == 1 or (AMASK == 2 and TAIL_N == 0))
        DF: tl.constexpr = DEFER == 1 or (DEFER == 2 and BLOCK_M >= 16)
        qrow = Q + om[:, None] * qs
        qc = h * D + od
        # ---- issue every global load first (independent, all in flight) ------
        if DKV:
            k = tl.full([BLOCK_N, D], 0.01, Q.dtype.element_ty)
            v = tl.full([BLOCK_N, DV], 0.01, Q.dtype.element_ty)
        elif FULL_MAIN:
            k = tl.load(K + h * kh + on[:, None] * ks + od[None, :])
            v = tl.load(V + h * vh + on[:, None] * vs + ov[None, :])
        else:
            k = tl.load(K + h * kh + on[:, None] * ks + od[None, :], mask=mn[:, None], other=0.0)
            v = tl.load(V + h * vh + on[:, None] * vs + ov[None, :], mask=mn[:, None], other=0.0)
        # ROUND-2 `PRELOAD`: k/v were hoisted here already, but the TAIL tile and
        # the mask were not -- they sat behind the RMS reduction loop AND behind
        # the first dot.  Hoisting them is a pure load re-order (bit-identical)
        # and puts them in flight underneath the HD/RB reduction.
        if PRELOAD:
            if TAIL_N > 0:
                on2 = BLOCK_N + tl.arange(0, TAIL_N)
                mn2 = on2 < Sk
                if DKV:
                    k2 = tl.full([TAIL_N, D], 0.01, K.dtype.element_ty)
                    v2 = tl.full([TAIL_N, DV], 0.01, K.dtype.element_ty)
                else:
                    k2 = tl.load(K + h * kh + on2[:, None] * ks + od[None, :],
                                 mask=mn2[:, None], other=0.0)
                    v2 = tl.load(V + h * vh + on2[:, None] * vs + ov[None, :],
                                 mask=mn2[:, None], other=0.0)
            if HAS_MASK:
                if AM:
                    if FULL_MAIN and EXACT_M:
                        mk = tl.load(MSK + om[:, None] * sm_s + on[None, :]).to(tl.float32)
                    else:
                        mk = tl.load(MSK + om[:, None] * sm_s + on[None, :],
                                     mask=mm[:, None] & mn[None, :],
                                     other=float("-inf")).to(tl.float32)
                    if TAIL_N > 0:
                        mk2 = tl.load(MSK + om[:, None] * sm_s + on2[None, :],
                                      mask=mm[:, None] & mn2[None, :],
                                      other=float("-inf")).to(tl.float32)
                else:
                    mk = tl.load(MSK + om[:, None] * sm_s + on[None, :],
                                 mask=mm[:, None] & mn[None, :], other=0)
                    if TAIL_N > 0:
                        mk2 = tl.load(MSK + om[:, None] * sm_s + on2[None, :],
                                      mask=mm[:, None] & mn2[None, :], other=0)
        if EXACT_M:
            xq = tl.load(qrow + qc[None, :])
        else:
            xq = tl.load(qrow + qc[None, :], mask=mm[:, None], other=0.0)
        wq = tl.load(WQ + qc)
        if NEW:
            if EXP2:
                qk = scale * 1.4426950408889634
            else:
                qk = scale
            if DF:
                # DEFERRED rstd.  `rstd` is a PER-ROW SCALAR, so
                #     (x * rstd * w) @ k^T  ==  rstd * ((x * w) @ k^T)
                # and the whole HD-wide sum-of-squares comes OFF the QK dot's
                # dependency chain: the MFMAs issue first and the reduction's L2
                # round trips run underneath them.  The only numerical difference
                # is that the single bf16 rounding lands on `x*w` instead of on
                # `x*rstd*w` -- same relative quantum, different rounding.
                q = ((xq.to(tl.float32)) * wq.to(tl.float32)).to(Q.dtype.element_ty)
                s = tl.dot(q, tl.trans(k))
                if TAIL_N > 0:
                    s2 = tl.dot(q, tl.trans(k2))
            if DRED == 2:
                rstd = tl.full([BLOCK_M], 1.0, tl.float32)
            else:
                acc = tl.zeros([BLOCK_M, RB], tl.float32)
                for r0 in tl.range(0, HD, RB, num_stages=RSTAGES):
                    ri = r0 + tl.arange(0, RB)
                    if EXACT_M:
                        xr = tl.load(qrow + ri[None, :]).to(tl.float32)
                    else:
                        xr = tl.load(qrow + ri[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
                    acc += xr * xr
                if DRED == 1:
                    rstd = tl.rsqrt(tl.max(acc, 1) / HD + eps)
                else:
                    rstd = tl.rsqrt(tl.sum(acc, 1) / HD + eps)
            if DF:
                rs = (rstd * qk)[:, None]
                s = s * rs
                if TAIL_N > 0:
                    s2 = s2 * rs
            else:
                q = ((xq.to(tl.float32) * rstd[:, None]) * wq.to(tl.float32)).to(Q.dtype.element_ty)
                s = tl.dot(q, tl.trans(k)) * qk
                if TAIL_N > 0:
                    s2 = tl.dot(q, tl.trans(k2)) * qk
            if AM:
                s = s + mk
            else:
                s = tl.where(mn[None, :] & (mk == 0), s, float("-inf"))
            if TAIL_N > 0 and not DTAIL:
                if AM:
                    s2 = s2 + mk2
                else:
                    s2 = tl.where(mn2[None, :] & (mk2 == 0), s2, float("-inf"))
                if DMAX:
                    mx = tl.zeros([BLOCK_M, 1], tl.float32)
                elif TMAX:
                    mx = tl.max(s, 1)[:, None]
                else:
                    mx = tl.maximum(tl.max(s, 1), tl.max(s2, 1))[:, None]
                if EXP2:
                    p = tl.exp2(s - mx)
                    p2 = tl.exp2(s2 - mx)
                else:
                    p = tl.exp(s - mx)
                    p2 = tl.exp(s2 - mx)
                if DDEN:
                    den = tl.full([BLOCK_M, 1], 1.0, tl.float32)
                else:
                    den = (tl.sum(p, 1) + tl.sum(p2, 1))[:, None]
                if POSTN:
                    if TSEP:
                        o = tl.dot(p.to(Q.dtype.element_ty), v) + tl.dot(p2.to(Q.dtype.element_ty), v2)
                    else:
                        o = tl.dot(p.to(Q.dtype.element_ty), v)
                        o = tl.dot(p2.to(Q.dtype.element_ty), v2, o)
                    if RECIP:
                        o = o * (1.0 / den)
                    else:
                        o = o / den
                else:
                    if RECIP:
                        rden = 1.0 / den
                        o = tl.dot((p * rden).to(Q.dtype.element_ty), v)
                        o = tl.dot((p2 * rden).to(Q.dtype.element_ty), v2, o)
                    else:
                        o = tl.dot((p / den).to(Q.dtype.element_ty), v)
                        o = tl.dot((p2 / den).to(Q.dtype.element_ty), v2, o)
            else:
                if DMAX:
                    mx = tl.zeros([BLOCK_M, 1], tl.float32)
                else:
                    mx = tl.max(s, 1)[:, None]
                if EXP2:
                    p = tl.exp2(s - mx)
                else:
                    p = tl.exp(s - mx)
                if DDEN:
                    den = tl.full([BLOCK_M, 1], 1.0, tl.float32)
                else:
                    den = tl.sum(p, 1)[:, None]
                if POSTN:
                    o = tl.dot(p.to(Q.dtype.element_ty), v)
                    if RECIP:
                        o = o * (1.0 / den)
                    else:
                        o = o / den
                else:
                    if RECIP:
                        p = p * (1.0 / den)
                    else:
                        p = p / den
                    o = tl.dot(p.to(Q.dtype.element_ty), v)
            if EXACT_M:
                tl.store(O + h * oh + om[:, None] * os_ + ov[None, :], o.to(Q.dtype.element_ty))
            else:
                tl.store(O + h * oh + om[:, None] * os_ + ov[None, :], o.to(Q.dtype.element_ty),
                         mask=mm[:, None])
        else:
            acc = tl.zeros([BLOCK_M, RB], tl.float32)
            for r0 in tl.range(0, HD, RB, num_stages=RSTAGES):
                ri = r0 + tl.arange(0, RB)
                if EXACT_M:
                    xr = tl.load(qrow + ri[None, :]).to(tl.float32)
                else:
                    xr = tl.load(qrow + ri[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
                acc += xr * xr
            # ---- RMSNorm, transcribed from inductor's generated body -------------
            rstd = tl.rsqrt(tl.sum(acc, 1) / HD + eps)
            q = ((xq.to(tl.float32) * rstd[:, None]) * wq.to(tl.float32)).to(Q.dtype.element_ty)
            # ---- unchanged single-pass attention ---------------------------------
            s = tl.dot(q, tl.trans(k)) * scale
            keep = mn[None, :]
            if HAS_MASK:
                if not PRELOAD:
                    mk = tl.load(MSK + om[:, None] * sm_s + on[None, :],
                                 mask=mm[:, None] & mn[None, :], other=0)
                if MASK_BIAS:
                    keep = keep & (mk == 0)
                else:
                    keep = keep & (mk != 0)
            if HAS_MASK or not FULL_MAIN:
                s = tl.where(keep, s, float("-inf"))
            if TAIL_N > 0:
                if not PRELOAD:
                    on2 = BLOCK_N + tl.arange(0, TAIL_N)
                    mn2 = on2 < Sk
                    k2 = tl.load(K + h * kh + on2[:, None] * ks + od[None, :],
                                 mask=mn2[:, None], other=0.0)
                s2 = tl.dot(q, tl.trans(k2)) * scale
                keep2 = mn2[None, :]
                if HAS_MASK:
                    if not PRELOAD:
                        mk2 = tl.load(MSK + om[:, None] * sm_s + on2[None, :],
                                      mask=mm[:, None] & mn2[None, :], other=0)
                    if MASK_BIAS:
                        keep2 = keep2 & (mk2 == 0)
                    else:
                        keep2 = keep2 & (mk2 != 0)
                s2 = tl.where(keep2, s2, float("-inf"))
                mx = tl.maximum(tl.max(s, 1), tl.max(s2, 1))[:, None]
                p = tl.exp(s - mx)
                p2 = tl.exp(s2 - mx)
                den = (tl.sum(p, 1) + tl.sum(p2, 1))[:, None]
                p = p / den
                p2 = p2 / den
                if not PRELOAD:
                    v2 = tl.load(V + h * vh + on2[:, None] * vs + ov[None, :], mask=mn2[:, None], other=0.0)
                o = tl.dot(p.to(Q.dtype.element_ty), v)
                o = tl.dot(p2.to(Q.dtype.element_ty), v2, o)
            else:
                s = s - tl.max(s, 1)[:, None]
                p = tl.exp(s)
                p = p / tl.sum(p, 1)[:, None]
                o = tl.dot(p.to(Q.dtype.element_ty), v)
            if EXACT_M:
                tl.store(O + h * oh + om[:, None] * os_ + ov[None, :], o.to(Q.dtype.element_ty))
            else:
                tl.store(O + h * oh + om[:, None] * os_ + ov[None, :], o.to(Q.dtype.element_ty),
                         mask=mm[:, None])


# Reduction block for the fused RMSNorm.  Must DIVIDE HD exactly (so the r loop
# needs no column mask and the summation tree is uniform) and stay small enough
# that the [BLOCK_M, RB] fp32 accumulator does not spill at num_warps=4.
# Reduction tile and loop pipeline depth, both swept on the PROFILE leg against
# the in-graph kernel time of `_fw_qrms_attn_kernel` (denoise Sq=32/Sk=129,
# prefill Sq=98/Sk=129), 1 run each:
#   RB   RS   denoise us   note
#   256   1     10.74      rolled, no pipelining
#   256   1     10.77      + hand-unrolled x4 (no help: not issue-bound)
#   512   1     10.97      fully unrolled `static_range` (spills)
#  1024   1     11.82      accumulator too wide -> spills
#   512   2     10.35
#   512   3      9.87      <- shipped
#   512   4     10.29
#   512   6     10.44
#  1024   3     11.46
# num_warps is INHERITED from `_fa_key_tile` (4 here); 8 measured 11.15 and 16
# measured 25.41, so the reduction is not helped by more lanes -- it is L2
# ROUND-TRIP LATENCY, which only `tl.range(..., num_stages=RS)` hides.  This is
# the same `num_stages=2` software pipeline inductor puts on its own reduction
# loop; RS=3 is the depth this shape wants.
_QRMS_RB = 512
_QRMS_RS = 3


def _qrms_rb(HD: int) -> int:
    """Reduction tile: must divide HD exactly so the fully-unrolled r loop needs
    no column mask and every row sees the same summation tree."""
    rb = _QRMS_RB
    while rb > 32 and HD % rb:
        rb //= 2
    return rb if HD % rb == 0 else 0


# In-file literal: run_local.sh forwards only GATE_*/FWE_*/ORACLE_*, so this is
# the switch that is actually flipped between A/B arms.
_QRMS = os.environ.get("FASTWAM_QRMS", "1")


def _qrms_enabled() -> bool:
    return _FA_TRITON and _QRMS.lower() not in ("0", "false", "no", "off")


@torch.library.custom_op("fastwam::qrms_attn", mutates_args=())
def _fw_qrms_attn(qraw: torch.Tensor, wq: torch.Tensor, k: torch.Tensor,
                  v: torch.Tensor, mask: Optional[torch.Tensor],
                  num_heads: int, eps: float) -> torch.Tensor:
    B, Sq, HD = qraw.shape
    H = int(num_heads)
    D = HD // H
    Sk = k.shape[1]
    o = torch.empty((B, Sq, HD), dtype=qraw.dtype, device=qraw.device)
    # COMMANDMENT rule 9: the key-tile table is the ONE source of truth for every
    # op that launches this family of kernels.  Never re-derive it here.
    BLOCK_N, TAIL_N, nw = _fa_key_tile(Sk)
    BLOCK_M = _fa_block_m(Sq, qrms=1)
    nw = _fa_nw(BLOCK_M, nw)
    gm = (Sq + BLOCK_M - 1) // BLOCK_M
    dspl = _fa_dspl(D, qrms=1, ncta=gm * H)
    grid = (gm * dspl, H)
    _fw_qrms_attn_kernel[grid](
        qraw, wq, k, v, mask if mask is not None else qraw, o, Sq, Sk,
        qraw.stride(1), D, k.stride(1), D, v.stride(1), D, o.stride(1),
        0 if mask is None else mask.stride(0),
        D ** -0.5, eps, D, BLOCK_M, BLOCK_N, mask is not None,
        mask is not None and mask.dtype != torch.bool,
        HD, _qrms_rb(HD), Sq % BLOCK_M == 0, _QRMS_RS,
        TAIL_N=TAIL_N, PRELOAD=_FA_PRELOAD,
        FULL_MAIN=int(_FA_FULLMAIN and Sk >= BLOCK_N),
        DSPL=dspl,
        num_warps=nw, num_stages=1, **_fa_extra_kw(), **_fa_xcd_kw(grid),
    )
    return o


@_fw_qrms_attn.register_fake
def _(qraw, wq, k, v, mask, num_heads, eps):
    return qraw.new_empty(qraw.shape)


def _qrms_eligible(qraw, wq, k, v, num_heads: int) -> bool:
    """Static-shape envelope for `fastwam::qrms_attn` (3-D [B, S, H*Dh] only)."""
    if not (_qrms_enabled() and _fa_fast_enabled()):
        return False
    if qraw.dim() != 3 or k.dim() != 3 or v.dim() != 3 or wq.dim() != 1:
        return False
    if qraw.dtype not in (torch.bfloat16, torch.float16):
        return False
    if k.dtype != qraw.dtype or v.dtype != qraw.dtype or wq.dtype != qraw.dtype:
        return False
    if qraw.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        return False
    if qraw.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        return False
    HD = qraw.shape[2]
    if HD != k.shape[2] or HD != v.shape[2] or HD != wq.shape[0]:
        return False
    if num_heads <= 0 or HD % num_heads or (HD // num_heads) not in _FA_OK_D:
        return False
    if not wq.is_contiguous() or _qrms_rb(HD) == 0:
        return False
    return k.shape[1] == v.shape[1] and k.shape[1] <= 256


def qrms_attention(qraw, norm_q, k, v, num_heads: int, ctx_mask=None):
    """`flash_attention(norm_q(qraw), k, v, ...)` with the RMSNorm fused in.

    Falls back to the exact original expression whenever anything is outside the
    verified envelope, so this is strictly a fast path.
    """
    wq = getattr(norm_q, "weight", None)
    if wq is not None and _qrms_eligible(qraw, wq, k, v, num_heads):
        m = _fa_mask2d(ctx_mask, qraw.shape[1], k.shape[1])
        if m is not False:
            return torch.ops.fastwam.qrms_attn.default(
                qraw, wq, k, v, m, int(num_heads), float(norm_q.eps),
            )
    return None
