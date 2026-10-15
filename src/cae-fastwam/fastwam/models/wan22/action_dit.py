# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import os
import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from fastwam.utils.logging_config import get_logger

from .wan_video_dit import (
    DiTBlock,
    sinusoidal_embedding_1d,
    precompute_freqs_cis,
    _FW_ACTX_ENABLED,
    _fw_actx_register,
)

logger = get_logger(__name__)


# FASTWAM_CTXKVH (r2_d1): imported lazily so `action_dit` keeps its current
# import order (`mot` imports `wan_video_dit`, not `action_dit`, so there is no
# cycle -- the lazy form simply makes that independent of future edits).
def _fw_ctxkvh_disarm():
    from .mot import _fw_ctxkvh_disarm as _d
    _d()


def _fw_ctxkvh_stage(owner, emb):
    from .mot import _fw_ctxkvh_stage as _s
    _s(owner, emb)


def _fw_off(v):
    return str(v).strip().lower() in ("0", "off", "false", "no")


# ---------------------------------------------------------------------------
# MODE-CACHED `train()` FOR THE ACTION EXPERT  (kill-switch: FASTWAM_ADITEVAL=0)
#
# WHY.  `FastWAM.infer_action` (IMMUTABLE, fastwam.py:1097) opens with
# `self.eval()` on EVERY call.  `nn.Module.train(mode)` is an unconditional
# CPython recursion over the whole module tree -- it re-writes `.training` on
# every descendant even when every one of them already holds the requested
# value.  It runs at the very top of the call with NOTHING queued on the GPU,
# so 100% of its cost is exposed wall time and it is a pure host bubble.
#
# The ported `wan_video_vae.py` already ships exactly this override on
# `WanVideoVAE38` (see its note: 319 of ~3147 per-call module visits,
# 0.338 ms of a measured 3.225 ms `model.eval()`, i.e. ~1.06 us per visit).
# The ACTION EXPERT is by far the bigger share and was never covered:
# `ActionDiT` is 30 `DiTBlock`s (each ~24 sub-modules: SelfAttention +
# CrossAttention + 3 LayerNorms + the FFN Sequential + GateModule) plus the
# embedding stacks -- ~735 modules -- and `MoT.mixtures` holds a SECOND
# reference to the same object (`FastWAM.action_expert` and
# `FastWAM.mot.mixtures["action"]`), and `nn.Module.train` has no visited-set,
# so the whole subtree is walked TWICE per call.  Truncating at this one root
# removes ~1.4-1.5 k of the ~3.1 k per-call module visits.
#
# WHY IT IS A NO-OP BY CONSTRUCTION.  The short-circuit only declines to
# RE-write flags that already equal `mode`.  The cache starts *absent*, so the
# first call in each direction falls through to `nn.Module.train` and really
# writes every descendant flag; a genuine transition in either direction is
# handled the same way.  Nothing in this tree writes a descendant `.training`
# behind `train()`'s back.
#
# COMMANDMENT RULE 6 (episode state).  There is no episode-varying state here:
# the only thing cached is a single Python `bool` describing a mode the caller
# just asked for.  No tensor, no `data_ptr`, no `_version`, nothing that a new
# observation or a new episode can invalidate.  Validated on the 14-episode
# closed-loop rollout regardless.
# ---------------------------------------------------------------------------
_FW_ADITEVAL = not _fw_off(os.environ.get("FASTWAM_ADITEVAL", "1"))


# ===========================================================================
# FASTWAM_ATMOD (r1_d3) -- kill-switch (RULE 9), default ON.
#
# WHAT IT COLLAPSES.  `ActionDiT.prepare` runs inside the once-per-STEP
# cudagraph-replayed region.  After the sinusoidal embedding it issues a chain
# of B=1 / hidden_dim=32 launches, every one of which is on this box's ~4 us
# per-dispatch floor and none of which does any real work:
#     time_embedding.0   Linear[256 -> 32]   (Cijk ... MT16x16x1024,  ~7.1 us)
#     time_embedding.1   SiLU                (triton_poi_fused_silu,  ~4.2 us)
#     time_embedding.2   Linear[32 -> 32]    (Cijk ... MT16x16x32,    ~4.3 us)
#     time_projection.0  SiLU                (triton_poi_fused_silu,  ~4.2 us)
#     time_projection.1  Linear[32 -> 192]   (Cijk ... MT16x16x32,    ~4.3 us)
#     action_encoder     Linear[16 -> 32]    (Cijk ... MT16x16x32,    ~4.3 us)
# i.e. ~28 us/step of dispatch floor for at most 16 kFLOP.  ONE Triton program
# does the whole chain in registers and writes `t`, `t_mod` and the encoded
# action tokens.  The win is DISPATCH DELETION, not cheaper math.
#
# NUMERICS (RULE 7 -- fp32 accumulation kept).  Each matvec mirrors what the
# vendor GEMM does: bf16 operands widened to fp32, accumulated in fp32, the
# bias added in fp32, and the result rounded to bf16 exactly once.  Each SiLU
# mirrors `at::native::silu`'s bf16 path (opmath_t = float): widen, x*sigmoid(x)
# in fp32, round once.  The reduction ORDER differs (tl.sum is a tree, Tensile
# is a DepthU MAC chain), so it is not promised bit-exact; it is oracle-measured
# on `mean|d|` at both step counts, per HAZARD 1 / RULE 12.
#
# RULE 6.  Nothing is cached: the op is a pure function of its arguments and
# holds no per-episode state.  RULE 7: it is a `torch.library.custom_op` with a
# `register_fake`, static shapes taken from python ints at call time, so
# `fullgraph=True` holds and the region stays 1 `hipGraphLaunch`.
# RULE 11: sized at M=8 (`seq_a`), and the `S` extent is read from the tensor,
# never assumed.
# ===========================================================================
_FW_ATMOD = not _fw_off(os.environ.get("FASTWAM_ATMOD", "1"))
_FW_ATMOD_WARPS = int(os.environ.get("FASTWAM_ATMOD_WARPS", "4"))
# FASTWAM_ASINE -- sub-lever: also fold `sinusoidal_embedding_1d` into the
# fused program (one more launch deleted).  Default ON.
_FW_ASINE = not _fw_off(os.environ.get("FASTWAM_ASINE", "1"))

_FW_TRITON_OK = False
try:  # pragma: no cover - environment dependent
    import triton
    import triton.language as tl
    _FW_TRITON_OK = True
except Exception:  # pragma: no cover
    triton = None
    tl = None


if _FW_TRITON_OK:

    @triton.jit
    def _fwam_adit_tmod_kernel(
        TE, INV, W0, B0, W1, B1, W2, B2, ATOK, AW, AB,
        TOUT, TMOUT, TKOUT,
        FD: tl.constexpr, H: tl.constexpr, H6: tl.constexpr, H6P: tl.constexpr,
        S: tl.constexpr, SP: tl.constexpr, A: tl.constexpr,
        FUSE_SIN: tl.constexpr,
    ):
        h = tl.arange(0, H)
        fd = tl.arange(0, FD)

        # ---- sinusoidal_embedding_1d, folded in (FASTWAM_ASINE) ---------
        # Mirrors `sinusoidal_embedding_1d` exactly: the inverse-frequency
        # vector is the SAME float64 tensor `torch.pow(10000, -arange/…)`
        # produces (precomputed once, on device), the angle and the cos/sin are
        # taken in float64, and the result is rounded to the position dtype
        # (bfloat16) exactly once -- which is also why float64 is kept here:
        # the angles reach ~1e3 rad and an fp32 product would perturb the last
        # bfloat16 bit on a few of the 256 lanes.
        if FUSE_SIN:
            half = fd % (FD // 2)
            inv = tl.load(INV + half)
            ang = tl.load(TE).to(tl.float64) * inv
            te = tl.where(fd < (FD // 2), tl.cos(ang), tl.sin(ang)).to(tl.bfloat16).to(tl.float32)
        else:
            te = tl.load(TE + fd).to(tl.float32)                  # [FD]

        # ---- time_embedding.0 : Linear[FD -> H] -------------------------
        w0 = tl.load(W0 + h[:, None] * FD + fd[None, :]).to(tl.float32)
        acc = tl.sum(w0 * te[None, :], 1) + tl.load(B0 + h).to(tl.float32)
        x = acc.to(tl.bfloat16)

        # ---- SiLU (aten bf16 path: widen, fp32 math, round once) --------
        xf = x.to(tl.float32)
        x = (xf * tl.sigmoid(xf)).to(tl.bfloat16)

        # ---- time_embedding.2 : Linear[H -> H] --------------------------
        xf = x.to(tl.float32)
        w1 = tl.load(W1 + h[:, None] * H + h[None, :]).to(tl.float32)
        t = (tl.sum(w1 * xf[None, :], 1) + tl.load(B1 + h).to(tl.float32)).to(tl.bfloat16)
        tl.store(TOUT + h, t)

        # ---- time_projection : SiLU then Linear[H -> 6H] ----------------
        tf = t.to(tl.float32)
        y = ((tf * tl.sigmoid(tf)).to(tl.bfloat16)).to(tl.float32)
        n6 = tl.arange(0, H6P)
        m6 = n6 < H6
        w2 = tl.load(W2 + n6[:, None] * H + h[None, :], mask=m6[:, None], other=0.0).to(tl.float32)
        tm = tl.sum(w2 * y[None, :], 1) + tl.load(B2 + n6, mask=m6, other=0.0).to(tl.float32)
        tl.store(TMOUT + n6, tm.to(tl.bfloat16), mask=m6)

        # ---- action_encoder : Linear[A -> H] on [S, A] ------------------
        sp = tl.arange(0, SP)
        a = tl.arange(0, A)
        smask = sp < S
        at = tl.load(ATOK + sp[:, None] * A + a[None, :], mask=smask[:, None], other=0.0).to(tl.float32)
        aw = tl.load(AW + h[:, None] * A + a[None, :]).to(tl.float32)
        tok = tl.sum(at[:, None, :] * aw[None, :, :], 2) + tl.load(AB + h)[None, :].to(tl.float32)
        tl.store(TKOUT + sp[:, None] * H + h[None, :], tok.to(tl.bfloat16), mask=smask[:, None])


_FW_ATMOD_OK = False
if _FW_TRITON_OK and hasattr(torch, "library") and hasattr(torch.library, "custom_op"):
    try:

        @torch.library.custom_op("fastwam::adit_tmod", mutates_args=())
        def _fwam_adit_tmod(
            te: torch.Tensor,
            inv: torch.Tensor,
            w0: torch.Tensor, b0: torch.Tensor,
            w1: torch.Tensor, b1: torch.Tensor,
            w2: torch.Tensor, b2: torch.Tensor,
            atok: torch.Tensor, aw: torch.Tensor, ab: torch.Tensor,
            fuse_sin: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            te = te.contiguous()
            atok = atok.contiguous()
            fd = int(w0.shape[1]) if fuse_sin else int(te.shape[-1])
            h = int(w0.shape[0])
            h6 = int(w2.shape[0])
            s = int(atok.shape[1])
            a = int(atok.shape[2])
            odt = atok.dtype
            t = torch.empty((1, h), dtype=odt, device=te.device)
            tm = torch.empty((1, h6), dtype=odt, device=te.device)
            tk = torch.empty((1, s, h), dtype=odt, device=te.device)
            _fwam_adit_tmod_kernel[(1,)](
                te, inv, w0, b0, w1, b1, w2, b2, atok, aw, ab, t, tm, tk,
                FD=fd, H=h, H6=h6, H6P=max(1, 1 << (h6 - 1).bit_length()), S=s,
                SP=max(1, 1 << (s - 1).bit_length()), A=a,
                FUSE_SIN=int(fuse_sin),
                num_warps=_FW_ATMOD_WARPS,
            )
            return t, tm, tk

        @_fwam_adit_tmod.register_fake
        def _(te, inv, w0, b0, w1, b1, w2, b2, atok, aw, ab, fuse_sin):
            return (
                atok.new_empty((1, w0.shape[0])),
                atok.new_empty((1, w2.shape[0])),
                atok.new_empty((1, atok.shape[1], w0.shape[0])),
            )

        _FW_ATMOD_OK = True
    except Exception:  # pragma: no cover
        _FW_ATMOD_OK = False


class ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        return self.proj(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class ActionDiT(nn.Module):
    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")
    ACTION_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, action_dim)
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)

        # FASTWAM_ACTX (see wan_video_dit.py): the once-per-call prologue
        # (`WanModel.prepare`) re-stages this expert's step-invariant text
        # context projection into `_fw_actx_emb` before every denoise loop.
        # `_fw_actx_ok` is FAIL-CLOSED: False here, and False again at the top
        # of every staging attempt, so any path that reaches `prepare` without
        # a fresh projection recomputes it in-graph.
        self._fw_actx_ok = False
        self._fw_actx_len = -1
        # FASTWAM_ATMOD engagement predicate -- pure trace-time python metadata
        # (RULE 11: every extent is read from the built modules, nothing is
        # assumed about `seq_a`).
        self._fw_atmod_ok = bool(
            _FW_ATMOD
            and _FW_ATMOD_OK
            and (hidden_dim & (hidden_dim - 1)) == 0
            and (freq_dim & (freq_dim - 1)) == 0
            and (action_dim & (action_dim - 1)) == 0
        )
        # FASTWAM_ASINE: fold `sinusoidal_embedding_1d` into the same program.
        # `_fw_sin_inv` is the IDENTICAL float64 inverse-frequency vector
        # `sinusoidal_embedding_1d` builds, materialised once as a
        # non-persistent buffer (static-address graph input, zero cudagraph
        # input copies) instead of being rebuilt by an `arange`/`pow` chain on
        # every replay.
        self.register_buffer(
            "_fw_sin_inv",
            torch.pow(
                10000,
                -torch.arange(freq_dim // 2, dtype=torch.float64).div(freq_dim // 2),
            ),
            persistent=False,
        )
        self._fw_asine_ok = bool(self._fw_atmod_ok and _FW_ASINE)
        if _FW_ACTX_ENABLED:
            _fw_actx_register(self)

        self.use_gradient_checkpointing = use_gradient_checkpointing

    def _apply(self, fn):
        result = super()._apply(fn)
        device = next(self.parameters()).device
        self.freqs = self.freqs.to(device=device)
        # FASTWAM_ASINE: `Module.to(dtype=bfloat16)` would cast this float64
        # buffer, so rebuild it here -- on the DEVICE, with exactly the ops
        # `sinusoidal_embedding_1d` uses, so the angles are bit-identical.
        self._fw_sin_inv = torch.pow(
            10000,
            -torch.arange(self.freq_dim // 2, dtype=torch.float64, device=device).div(
                self.freq_dim // 2
            ),
        )
        return result

    def train(self, mode: bool = True):
        # See the FASTWAM_ADITEVAL note at the top of this file.  Skips the
        # ~735-module (x2 references) CPython recursion when the requested
        # mode already holds everywhere below this root.
        if _FW_ADITEVAL:
            mode = bool(mode)
            if self.__dict__.get("_fw_train_mode", None) is mode:
                return self
            nn.Module.train(self, mode)
            self.__dict__["_fw_train_mode"] = mode
            return self
        return nn.Module.train(self, bool(mode))

    def get_freqs(self, seq_len: int) -> torch.Tensor:
        return self.freqs[:seq_len].view(seq_len, 1, -1)

    @torch.no_grad()
    def _fw_stage_context(self, context: torch.Tensor) -> None:
        """FASTWAM_ACTX: refresh the staged text-context projection.

        Called ONCE PER `infer_action` from the video expert's eager prologue,
        with the same raw `context` the per-step compiled region receives.
        Fail-closed by construction (RULE 6): the validity flag is cleared
        before anything else and is only re-armed once the fresh projection has
        actually been copied into the persistent buffer.
        """
        self._fw_actx_ok = False
        # FASTWAM_CTXKVH (r2_d1): the MoT's staged cross-attention context K/V
        # is derived from the buffer this method refreshes, so it is disarmed
        # FIRST and only re-armed below, after the fresh embedding has landed.
        _fw_ctxkvh_disarm()
        if not _FW_ACTX_ENABLED:
            return
        if not isinstance(context, torch.Tensor) or context.ndim != 3:
            return
        if int(context.shape[-1]) != int(self.text_dim):
            return
        emb = self.text_embedding(context)
        buf = self._buffers.get("_fw_actx_emb", None)
        if buf is None:
            # Allocated exactly ONCE and never reallocated -- the address a
            # cudagraph captures for this static input can therefore never go
            # stale.  A later call with a different context length simply does
            # not re-arm the flag and falls back to the in-graph projection.
            self.register_buffer("_fw_actx_emb", emb.detach().clone(), persistent=False)
            self._fw_actx_len = int(context.shape[1])
        else:
            if buf.shape != emb.shape or buf.dtype != emb.dtype:
                return
            buf.copy_(emb)
        self._fw_actx_ok = True
        # FASTWAM_CTXKVH: the action expert's whole-stack cross-attn context
        # K/V is a pure function of the embedding just staged, so hoist it out
        # of the captured per-step region too.  Unconditional per call, never
        # memoised (COMMANDMENT RULE 6: `context` is EPISODE-VARYING).
        _fw_ctxkvh_stage(self, self._buffers["_fw_actx_emb"])

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiT":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiT.from_pretrained().")
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert randomly and expecting checkpoint override."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if not action_dit_pretrained_path:
            logger.info("No `action_dit_pretrained_path` provided, initializing ActionDiT with random weights.")
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        from pathlib import Path
        p = Path(action_dit_pretrained_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[4] / p
        action_dit_pretrained_path = str(p)
        if not os.path.isfile(action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`action_dit_pretrained_path` does not exist: {action_dit_pretrained_path}"
            )

        action_cfg = dict(action_dit_config)
        action_expert = cls(**action_cfg).to(device=device, dtype=torch_dtype)
        action_state = action_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(action_state.keys())

        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
            )
        
        policy = payload.get("policy", {})
        if policy:
            logger.info(f"ActionDiT backbone payload policy: {policy}")

        meta = payload.get("meta")
        expected_meta = {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        }
        for key in cls.ACTION_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {action_dit_pretrained_path}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {action_dit_pretrained_path}, "
                f"got {type(backbone_state_dict)}"
            )

        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Action backbone key mismatch in preprocessed payload. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(action_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{key}]` must be torch.Tensor in {action_dit_pretrained_path}, "
                    f"got {type(value)}"
                )
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        action_expert.load_state_dict(merged_state, strict=True)
        logger.info(
            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
        )
        return action_expert.to(device=device, dtype=torch_dtype)

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, action_dim], got shape {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if context.ndim != 3:
            raise ValueError(
                f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}"
            )

        batch_size = action_tokens.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between action tokens and text context: {batch_size} vs {context.shape[0]}"
            )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        seq_len = action_tokens.shape[1]
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        tokens = self.action_encoder(action_tokens)
        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        freqs = self.get_freqs(seq_len)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(tokens)

    def prepare(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare tensor inputs for the compile-friendly DiT/MoT core."""
        seq_len = action_tokens.shape[1]
        # FASTWAM_ATMOD: one Triton program for the whole B=1 time chain +
        # the action-token encoder (six launches -> one).  See the block above.
        if self._fw_atmod_ok:
            if self._fw_asine_ok:
                te = timestep
            else:
                te = sinusoidal_embedding_1d(self.freq_dim, timestep)
            t, t_mod_flat, tokens = torch.ops.fastwam.adit_tmod(
                te,
                self._fw_sin_inv,
                self.time_embedding[0].weight, self.time_embedding[0].bias,
                self.time_embedding[2].weight, self.time_embedding[2].bias,
                self.time_projection[1].weight, self.time_projection[1].bias,
                action_tokens,
                self.action_encoder.weight, self.action_encoder.bias,
                int(self._fw_asine_ok),
            )
            t_mod = t_mod_flat.unflatten(1, (6, self.hidden_dim))
        else:
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
            tokens = self.action_encoder(action_tokens)
        # FASTWAM_ACTX: `context` is a per-CALL constant, so its projection is
        # step-invariant.  When the prologue staged it (the only path that
        # reaches here in `infer_action`), read the persistent static-address
        # buffer instead of recomputing the two Linears + GELU on every replay.
        if _FW_ACTX_ENABLED and self._fw_actx_ok:
            context_emb = self._fw_actx_emb
        else:
            context_emb = self.text_embedding(context)
        # FASTWAM_MASKSTAGE (r1_d1): `context_mask` is a per-CALL constant that
        # `infer_action` hands to the compiled denoise callable as a FRESH
        # tensor object every call, so cudagraph-trees re-copies its 129 bytes
        # into the graph placeholder on EVERY step.  The host-side wrapper in
        # mot.py stages it into this non-persistent buffer once per call (same
        # trick `_fw_actx_emb` above already uses for `context`); reading the
        # buffer drops the argument out of the traced graph entirely.
        _cmb = getattr(self, "_fw_cm_buf", None)
        if _cmb is not None and _cmb.shape == context_mask.shape:
            context_mask = _cmb
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        freqs = self.get_freqs(seq_len)
        return tokens, t, t_mod, context_emb, context_attn_mask, freqs

    def post(self, tokens: torch.Tensor) -> torch.Tensor:
        """Project action tokens produced by the tensor core."""
        return self.head(tokens)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context_mask is None:
            context_mask = torch.ones(
                (action_tokens.shape[0], context.shape[1]),
                dtype=torch.bool,
                device=context.device,
            )
        # The eager `forward` entry point is not fed by the prologue, so it
        # must never consume a staged buffer (FASTWAM_ACTX, RULE 6).
        self._fw_actx_ok = False
        x, _t, t_mod, context_emb, context_attn_mask, freqs = self.prepare(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )

        for block in self.blocks:
            x = block(
                x,
                context_emb,
                t_mod,
                freqs,
                context_mask=context_attn_mask,
            )

        return self.post(x)
