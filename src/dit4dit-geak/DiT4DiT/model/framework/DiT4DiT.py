# Copyright 2025 DiT4DiT team. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Teli Ma/ HKUST GZ] in [2025]. 
# Modified by Advanced Micro Devices, Inc. Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.


import sys
from pathlib import Path
# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image



from DiT4DiT.training.trainer_utils import initialize_overwatch


logger = initialize_overwatch(__name__)

import os as _geak_osmod


def _geak_os_env(name, default):
    return _geak_osmod.environ.get(name, default)



# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from DiT4DiT.model.framework.base_framework import baseframework
from DiT4DiT.model.modules.vlm import get_backbone_model
from DiT4DiT.model.modules.action_model.ActionDiT import get_action_model, FlowmatchingActionHead
from DiT4DiT.training.trainer_utils.trainer_tools import resize_images
from DiT4DiT.model.tools import FRAMEWORK_REGISTRY



# ---------------------------------------------------------------------------
# GEAK r2_d0: HIP-graph capture of the once-per-call backbone.
#
# Round 1 captured the action head's denoise loop and left it 96.6% device-bound
# with 0.005 ms of inter-kernel gap.  Everything that is still host-paced lives
# in the ONE-SHOT backbone (image preprocess -> VAE encode -> prepare_latents ->
# 18 video-DiT blocks -> hidden), which is ~2000 dispatches and ~5.5 ms of HIP
# launch CPU per call against a 13.8 / 17.2 ms wall.  That whole region is
# static-shape and, after the device-constant caching in Cosmos25.py, contains
# NO host<->device sync: the only remaining transfer is the input image H2D,
# which stays outside the capture, and the terminal .cpu() which is also outside.
#
# The capture boundary is `self.backbone_interface(**backbone_inputs)`:
#   in  : backbone_inputs["videos"]  (B,3,1,H,W) already on device
#   out : backbone_outputs.hidden_states[-1]
# `prompts`/`height`/`width` are host-side constants folded into the capture and
# re-checked in the cache key, so a different instruction re-captures instead of
# silently replaying a stale prompt.
_GEAK_BBGRAPH = str(_geak_os_env("GEAK_DIT_BBGRAPH", "1")).strip() not in ("0", "false", "False", "")
_GEAK_BBVERIFY = str(_geak_os_env("GEAK_DIT_BBVERIFY", "0")).strip() not in ("0", "false", "False", "")
_GEAK_BBWARM = int(_geak_os_env("GEAK_DIT_BBWARM", "3") or 3)
# GEAK r5_d2: PART A, MEASURED.  The ~204 us head-of-call gap is NOT the
# backbone graph's launch CPU.  Timed directly on this box (medians, 4-step):
#   backbone graph (~950 nodes): hipGraphLaunch = 0.022 ms of host CPU, of which
#     only 0.004 ms is EXPOSED (intercept of a wall-vs-K fit over K back-to-back
#     replays; the slope, 4.650 ms, is the GPU time).
#   action-head graph:           0.006 ms host, 0.009 ms exposed.
#   => ~13 us of the whole call is graph launch.  The 0.269 ms the torch.profiler
#      CPU table attributes to 2 hipGraphLaunch calls is profiler instrumentation
#      overhead, not launch cost.  DO NOT re-fund "shrink the graph launch".
# What IS exposed is the un-captured image preprocess in front of the graph:
# `build_cosmos_inputs` is 0.15 ms of host CPU with the GPU completely idle.
# GEAK_DIT_PREGRAPH makes the RAW uint8 frame the graph input and captures the
# preprocess chain with the backbone, leaving only numpy stacking + one H2D +
# hipGraphLaunch on the exposed host path.
_GEAK_PREGRAPH = str(_geak_os_env("GEAK_DIT_PREGRAPH", "1")).strip() not in ("0", "false", "False", "")


@FRAMEWORK_REGISTRY.register("DiT4DiT")
class DiT4DiT(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Supports two action prediction modes:
      - "joint"/"action": Separate action DiT head (legacy DiT4DiT)
      - "policy": Cosmos-policy latent frame injection (action-as-latent-frame)

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config

        training_mode = config.framework.cosmos25.training.lower() if config is not None else "action"
        self.video_fm_only = (training_mode == "video")
        self.policy_mode = (training_mode == "policy")

        self.backbone_interface = get_backbone_model(config=self.config)

        vl_hidden_dim = None
        if hasattr(self.backbone_interface, "model") and hasattr(self.backbone_interface.model, "config"):
            vl_hidden_dim = getattr(self.backbone_interface.model.config, "hidden_size", None)
        if vl_hidden_dim is None and hasattr(self.backbone_interface, "extractor"):
            vl_hidden_dim = getattr(self.backbone_interface.extractor, "hidden_size", None)
        if vl_hidden_dim is None:
            vl_hidden_dim = getattr(self.config.framework.cosmos25, "vl_hidden_dim", None)

        if self.policy_mode:
            from DiT4DiT.model.modules.action_latent import ActionLatentCodec
            from DiT4DiT.model.modules.action_latent.action_latent_codec import ActionStats

            al_cfg = getattr(config.framework, "action_latent", None) or {}
            if isinstance(al_cfg, dict):
                _get = al_cfg.get
            else:
                _get = lambda k, d=None: getattr(al_cfg, k, d)

            self.action_codec = ActionLatentCodec(
                action_dim=int(_get("action_dim", 7)),
                action_horizon=int(_get("action_horizon", 8)),
                state_dim=int(_get("state_dim", 8)),
                latent_channels=int(_get("latent_channels", 16)),
                latent_h=int(_get("latent_h", 28)),
                latent_w=int(_get("latent_w", 28)),
            )

            stats_path = _get("dataset_statistics", None)
            if stats_path:
                import json
                with open(stats_path, "r") as f:
                    ds = json.load(f)
                robot_key = list(ds.keys())[0]
                act_dim = self.action_codec.action_dim
                st_dim = self.action_codec.state_dim
                self.register_buffer("action_min", torch.tensor(ds[robot_key]["action"]["min"][:act_dim], dtype=torch.float32))
                self.register_buffer("action_max", torch.tensor(ds[robot_key]["action"]["max"][:act_dim], dtype=torch.float32))
                self.register_buffer("state_min", torch.tensor(ds[robot_key]["state"]["min"][:st_dim], dtype=torch.float32))
                self.register_buffer("state_max", torch.tensor(ds[robot_key]["state"]["max"][:st_dim], dtype=torch.float32))
            else:
                self.register_buffer("action_min", None)
                self.register_buffer("action_max", None)
                self.register_buffer("state_min", None)
                self.register_buffer("state_max", None)

            self.action_model = None
            self.future_action_window_size = 0
            self.past_action_window_size = 0
            self.chunk_len = 0

        elif not self.video_fm_only:
            if vl_hidden_dim is None:
                raise ValueError(
                    "Cannot infer `vl_hidden_dim` for the selected backbone. "
                    "Please set `framework.cosmos25.vl_hidden_dim` in your config."
                )
            self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

            self.future_action_window_size = config.framework.action_model.future_action_window_size
            self.past_action_window_size = config.framework.action_model.past_action_window_size
            self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        else:
            self.action_model = None
            self.future_action_window_size = 0
            self.past_action_window_size = 0
            self.chunk_len = 0

    def _get_action_stats(self):
        from DiT4DiT.model.modules.action_latent.action_latent_codec import ActionStats
        if self.action_min is not None:
            return ActionStats(min=self.action_min, max=self.action_max)
        return None

    def _get_state_stats(self):
        from DiT4DiT.model.modules.action_latent.action_latent_codec import ActionStats
        if self.state_min is not None:
            return ActionStats(min=self.state_min, max=self.state_max)
        return None
        

    def forward(
        self,
        examples: List[dict] = None,
        training_mode: Optional[str] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]

        backbone_inputs = self.backbone_interface.build_cosmos_inputs(images=batch_images, instructions=instructions)

        # --- Cosmos-policy latent frame injection path ---
        if self.policy_mode:
            actions = [example["action"] for example in examples]
            state = [example["state"] for example in examples] if "state" in examples[0] else None

            device = next(self.backbone_interface.parameters()).device
            act_dim = self.action_codec.action_dim
            horizon = self.action_codec.action_horizon

            actions_t = torch.tensor(np.array(actions), device=device, dtype=torch.float32)
            # Take last `horizon` steps, trim to actual action_dim (remove padding)
            actions_t = actions_t[:, -horizon:, :act_dim]  # (B, horizon, act_dim)

            action_stats = self._get_action_stats()
            action_latents = self.action_codec.encode_actions(actions_t, action_stats)  # (B,16,1,28,28)

            state_latents = None
            if state is not None:
                st_dim = self.action_codec.state_dim
                state_t = torch.tensor(np.array(state), device=device, dtype=torch.float32)
                if state_t.ndim == 3:
                    state_t = state_t[:, -1, :st_dim]  # take last timestep
                else:
                    state_t = state_t[:, :st_dim]
                state_stats = self._get_state_stats()
                state_latents = self.action_codec.encode_state(state_t, state_stats)  # (B,16,1,28,28)

            tm = training_mode or "policy"

            with torch.autocast("cuda", dtype=torch.bfloat16):
                backbone_outputs = self.backbone_interface(
                    **backbone_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                    action_latents=action_latents,
                    state_latents=state_latents,
                    training_mode=tm,
                    action_codec=self.action_codec,
                )

            out = {}
            if backbone_outputs.action_loss is not None:
                out["action_loss"] = backbone_outputs.action_loss
            if backbone_outputs.future_video_loss is not None:
                out["future_video_loss"] = backbone_outputs.future_video_loss
            if backbone_outputs.action_decoded_loss is not None:
                out["action_decoded_loss"] = backbone_outputs.action_decoded_loss
            return out

        # --- Legacy DiT4DiT path (separate action DiT head) ---
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_outputs = self.backbone_interface(
                **backbone_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            if not self.video_fm_only:
                last_hidden = backbone_outputs.hidden_states[-1]
            else:
                last_hidden = None
            future_video_loss = getattr(backbone_outputs, "future_video_loss", None)

        if self.video_fm_only:
            if future_video_loss is None:
                raise ValueError(
                    "video_fm_only is enabled (cosmos25.training='video') but `future_video_loss` is None. "
                    "Please provide `image_next` (or future_images) and set "
                    "`framework.cosmos25.future_loss_type=flow_matching`."
                )
            return {"future_video_loss": future_video_loss}

        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        action_mask = [example["action_mask"] for example in examples]

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size+1):, :]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            action_mask = torch.from_numpy(np.stack(action_mask)).to(last_hidden.device)
            action_mask = action_mask.repeat(repeated_diffusion_steps, 1, 1)
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(last_hidden_repeated, actions_target_repeated, action_mask, state_repeated)

        out = {"action_loss": action_loss}
        if future_video_loss is not None:
            out["future_video_loss"] = future_video_loss
        return out

    # ---- GEAK r2_d0: captured backbone -------------------------------------
    def _geak_backbone_eager(self, backbone_inputs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.backbone_interface(
                **backbone_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            return out.hidden_states[-1]

    def _geak_bb_key(self, backbone_inputs):
        v = backbone_inputs.get("videos", None)
        if not torch.is_tensor(v) or v.device.type != "cuda":
            return None
        if backbone_inputs.get("future_videos", None) is not None:
            return None
        return (tuple(v.shape), v.dtype, v.device.index,
                tuple(backbone_inputs.get("prompts", ()) or ()),
                int(backbone_inputs.get("height", 0) or 0),
                int(backbone_inputs.get("width", 0) or 0))

    def _geak_bb_capture(self, backbone_inputs):
        """Warm up on a side stream, then capture the whole backbone into one graph."""
        v = backbone_inputs["videos"]
        s_vid = v.detach().clone()
        static_inputs = dict(backbone_inputs)
        static_inputs["videos"] = s_vid

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(max(1, _GEAK_BBWARM)):
                self._geak_backbone_eager(static_inputs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            s_out = self._geak_backbone_eager(static_inputs)
        return (g, s_vid, s_out)

    # ---- GEAK r5_d2: raw-uint8 capture ------------------------------------
    def _geak_bb_key_raw(self, raw):
        # The descriptor dict itself is memoised per (shape, prompt, config) on
        # the interface, so the graph key can be memoised INTO it: identity of
        # the descriptor already implies every component of the key.  Built once
        # per episode instead of once per call.
        k = raw.get("_key")
        if k is None:
            k = ("geak_raw", tuple(int(x) for x in raw["shape"]),
                 tuple(raw["prompts"] or ()), int(raw["height"]), int(raw["width"]),
                 raw["meta"], str(raw["device"]))
            raw["_key"] = k
        return k

    def _geak_bb_capture_raw(self, raw):
        """Capture preprocess + backbone as ONE graph whose only input is uint8."""
        be = self.backbone_interface
        s_u8 = torch.empty(tuple(int(x) for x in raw["shape"]),
                           dtype=torch.uint8, device=raw["device"])
        s_u8.copy_(raw["host"])
        meta = raw["meta"]
        prompts, height, width = raw["prompts"], raw["height"], raw["width"]

        def _run():
            vids = be._geak_pre_from_u8(s_u8, meta)
            return self._geak_backbone_eager({
                "prompts": prompts, "videos": vids,
                "height": height, "width": width, "future_videos": None,
            })

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(max(1, _GEAK_BBWARM)):
                _run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            s_out = _run()
        return (g, s_u8, s_out)

    def _geak_backbone_hidden_raw(self, raw):
        """Replay the raw-input backbone graph; None if it is unavailable."""
        if not (_GEAK_BBGRAPH and _GEAK_PREGRAPH) or not torch.cuda.is_available():
            return None
        key = self._geak_bb_key_raw(raw)
        store = getattr(self, "_geak_bb_store", None)
        if store is None:
            store = {}
            self._geak_bb_store = store
        bad = getattr(self, "_geak_bb_bad", None)
        if bad is None:
            bad = set()
            self._geak_bb_bad = bad
        if key in bad:
            return None
        ent = store.get(key)
        if ent is None:
            # Same "at most ONE live graph" invariant as the cooked path: the
            # per-block cross-attention K/V memo holds only the newest prompt, so
            # replaying an older graph is a hard memory fault.  This shares the
            # SAME store as `_geak_backbone_hidden`, so a raw graph also evicts a
            # cooked one and the invariant holds across both entry points.
            if store:
                store.clear()
                torch.cuda.synchronize()
            try:
                ent = self._geak_bb_capture_raw(raw)
            except Exception as e:
                if _GEAK_BBVERIFY:
                    print(f"[GEAK_VERIFY] raw backbone capture FAILED: "
                          f"{type(e).__name__}: {e}", flush=True)
                bad.add(key)
                return None
            store[key] = ent
        g, s_u8, s_out = ent
        s_u8.copy_(raw["host"], non_blocking=True)
        g.replay()
        if _GEAK_BBVERIFY:
            _hn = raw["host_np"]
            ref = self.backbone_interface._geak_pre_from_u8(s_u8, raw["meta"])
            cooked = self.backbone_interface.build_cosmos_inputs(
                images=[[_hn[i]] for i in range(_hn.shape[0])],
                instructions=list(raw["prompts"]))["videos"]
            print("[GEAK_VERIFY] pregraph videos max|d|=%.6g  shape=%s/%s dtype=%s/%s"
                  % (float((ref.float() - cooked.float()).abs().max()),
                     tuple(ref.shape), tuple(cooked.shape), ref.dtype, cooked.dtype),
                  flush=True)
        return s_out

    def _geak_backbone_hidden(self, backbone_inputs):
        if not _GEAK_BBGRAPH or not torch.cuda.is_available():
            return self._geak_backbone_eager(backbone_inputs)
        key = self._geak_bb_key(backbone_inputs)
        if key is None:
            return self._geak_backbone_eager(backbone_inputs)
        store = getattr(self, "_geak_bb_store", None)
        if store is None:
            store = {}
            self._geak_bb_store = store
        bad = getattr(self, "_geak_bb_bad", None)
        if bad is None:
            bad = set()
            self._geak_bb_bad = bad
        if key in bad:
            return self._geak_backbone_eager(backbone_inputs)
        ent = store.get(key)
        if ent is None:
            # ONE graph per (video shape, prompt), and -- this is load bearing --
            # AT MOST ONE LIVE GRAPH AT A TIME.
            #
            # The prompt is BAKED IN, not a graph input: `encode_prompt`'s
            # embedding and the per-block cross-attention K/V memo keyed on its
            # content-derived uid both live inside the captured region, so a
            # replay with a different instruction would silently reuse the
            # captured prompt's K/V.  Keying the graph on the prompt turns that
            # into a capture MISS.
            #
            # MEASURED TRAP (this cost a debug cycle): keeping several captured
            # graphs alive and replaying an older one is a HARD MEMORY FAULT --
            #   `Memory Fault Error ... kernel: aiter::fmha_fwd_hd128_bf16`
            # -- because that K/V memo holds only the NEWEST uid per module.
            # Capturing prompt B evicts prompt A's K/V tensors, the caching
            # allocator hands their storage to someone else, and graph A's nodes
            # still point at the old addresses.  Any tensor a graph reads that
            # was allocated OUTSIDE its private pool must be pinned for the life
            # of the graph; a newest-only memo does not pin it.  So: drop every
            # older graph before capturing a new one.  A LIBERO episode holds one
            # instruction throughout, so the steady state is still zero
            # re-captures; a task switch pays one warmup+capture (~50 ms).
            if store:
                store.clear()
                torch.cuda.synchronize()
            try:
                ent = self._geak_bb_capture(backbone_inputs)
            except Exception as e:
                if _GEAK_BBVERIFY:
                    print(f"[GEAK_VERIFY] backbone capture FAILED: {type(e).__name__}: {e}",
                          flush=True)
                bad.add(key)
                return self._geak_backbone_eager(backbone_inputs)
            store[key] = ent
        g, s_vid, s_out = ent
        s_vid.copy_(backbone_inputs["videos"])
        g.replay()
        if _GEAK_BBVERIFY:
            # The backbone is INTRINSICALLY stochastic: latent frame 1 of the
            # (B,C,2,28,28) sequence is unconditioned, so `prepare_latents`
            # seeds it with fresh torch.randn every call and full attention
            # spreads that noise over all 392 tokens.  An exact graph-vs-eager
            # comparison is therefore impossible; the meaningful check is that
            # graph-vs-eager sits inside the eager-vs-eager band.
            ref1 = self._geak_backbone_eager(backbone_inputs).float()
            ref2 = self._geak_backbone_eager(backbone_inputs).float()
            o = s_out.float()
            print("[GEAK_VERIFY] backbone graph  shape=%s dtype=%s\n"
                  "              graph-vs-eager max|d|=%.6g   eager-vs-eager max|d|=%.6g\n"
                  "              mean g/e1/e2 = %.6g / %.6g / %.6g   std = %.6g / %.6g / %.6g"
                  % (tuple(s_out.shape), s_out.dtype,
                     float((o - ref1).abs().max()), float((ref1 - ref2).abs().max()),
                     float(o.mean()), float(ref1.mean()), float(ref2.mean()),
                     float(o.std()), float(ref1.std()), float(ref2.std())),
                  flush=True)
        return s_out

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = []
        for ex in examples:
            img = ex["image"]
            if isinstance(img, (list, tuple)) and len(img) > 0:
                batch_images.append(img)
            else:
                batch_images.append([img])
        instructions = [example["lang"] for example in examples]

        state = [example["state"] for example in examples] if "state" in examples[0] else None

        # GEAK r5_d2: exposed head-of-call host CPU.  On the legacy path the raw
        # uint8 frame IS the graph input, so `build_cosmos_inputs` (0.15 ms of
        # host CPU against an idle GPU) never runs; everything it did is a graph
        # node.  Falls through to the original path for any shape/config the raw
        # builder declines, and for policy_mode.
        last_hidden = None
        if _GEAK_PREGRAPH and not self.policy_mode:
            _rb = getattr(self.backbone_interface, "_geak_raw_build", None)
            if _rb is not None:
                try:
                    _raw = _rb(batch_images, instructions)
                except Exception:
                    _raw = None
                if _raw is not None:
                    last_hidden = self._geak_backbone_hidden_raw(_raw)

        if last_hidden is None:
            backbone_inputs = self.backbone_interface.build_cosmos_inputs(images=batch_images, instructions=instructions)

            if self.policy_mode:
                return self._predict_action_policy(backbone_inputs, state, **kwargs)

            # --- Legacy path ---
            last_hidden = self._geak_backbone_hidden(backbone_inputs)

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    @torch.inference_mode()
    def _predict_action_policy(
        self,
        backbone_inputs: dict,
        state: Optional[List] = None,
        num_inference_steps: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """Denoise action latent frames through the Cosmos backbone."""
        device = next(self.backbone_interface.parameters()).device
        B = backbone_inputs["videos"].shape[0]

        # Encode state conditioning
        state_latents = None
        if state is not None:
            st_dim = self.action_codec.state_dim
            state_t = torch.tensor(np.array(state), device=device, dtype=torch.float32)
            if state_t.ndim == 3:
                state_t = state_t[:, -1, :st_dim]
            else:
                state_t = state_t[:, :st_dim]
            state_stats = self._get_state_stats()
            state_latents = self.action_codec.encode_state(state_t, state_stats)

        # Initialize action latent frames from noise
        C = self.action_codec.latent_channels
        H = self.action_codec.latent_h
        W = self.action_codec.latent_w
        action_noise = torch.randn(B, C, 1, H, W, device=device, dtype=torch.float32)

        cosmos_cfg = getattr(self.config.framework, "cosmos25", None)
        if num_inference_steps is None:
            num_inference_steps = int(getattr(cosmos_cfg, "future_num_inference_steps", 10)) if cosmos_cfg else 10

        scheduler = self.backbone_interface.extractor.scheduler
        scheduler.set_timesteps(num_inference_steps, device=device)
        transformer_dtype = self.backbone_interface.extractor.transformer.dtype

        # Encode video and text once
        videos = backbone_inputs["videos"].to(device=device, dtype=self.backbone_interface.extractor.vae.dtype)
        if videos.shape[1] == 3:
            videos_bcthw = videos
        else:
            videos_bcthw = self.backbone_interface.extractor.video_processor.preprocess_video(
                videos, height=backbone_inputs["height"], width=backbone_inputs["width"]
            )

        prompt_embeds, _ = self.backbone_interface.extractor.encode_prompt(
            prompt=backbone_inputs["prompts"],
            guidance_scale=1.0,
            num_videos_per_prompt=1,
            device=device,
            max_sequence_length=self.backbone_interface.extractor.max_sequence_length,
        )

        num_channels_latents = int(self.backbone_interface.extractor.transformer.config.in_channels) - 1
        height, width = backbone_inputs["height"], backbone_inputs["width"]
        num_frames_in = int(videos_bcthw.shape[2])
        num_frames_out = num_frames_in

        latents, cond_latents, cond_mask, cond_indicator = self.backbone_interface.extractor.prepare_latents(
            video=videos_bcthw,
            batch_size=B,
            num_channels_latents=num_channels_latents,
            height=height, width=width,
            num_frames_in=num_frames_in,
            num_frames_out=num_frames_out,
            dtype=torch.float32,
            device=device,
        )

        # Build sequence: [video_cond | action_noise | state]
        _, _, T_vid, H_lat, W_lat = cond_latents.shape
        cond_count_vid = int(cond_indicator[0, 0, :, 0, 0].sum().item())

        # Recreate action noise at the correct spatial resolution
        action_noise = torch.randn(B, C, 1, H_lat, W_lat, device=device, dtype=torch.float32)

        def _match_spatial_infer(x: torch.Tensor) -> torch.Tensor:
            if x.shape[-2] == H_lat and x.shape[-1] == W_lat:
                return x
            B_, C_, T_, _, _ = x.shape
            flat = x.reshape(B_ * C_ * T_, 1, x.shape[-2], x.shape[-1])
            flat = F.interpolate(flat, size=(H_lat, W_lat), mode="bilinear", align_corners=False)
            return flat.reshape(B_, C_, T_, H_lat, W_lat)

        parts_cond_lat = [cond_latents[:, :, :cond_count_vid]]
        parts_noise = [latents[:, :, :cond_count_vid]]
        parts_cond_vec = [torch.ones(cond_count_vid, device=device)]

        parts_cond_lat.append(action_noise)
        parts_noise.append(action_noise.clone())
        parts_cond_vec.append(torch.zeros(1, device=device))  # action = denoised

        if state_latents is not None:
            state_latents_matched = _match_spatial_infer(state_latents.to(dtype=torch.float32))
            parts_cond_lat.append(state_latents_matched)
            parts_noise.append(torch.randn_like(state_latents_matched, dtype=torch.float32))
            parts_cond_vec.append(torch.ones(1, device=device))

        cond_latents_full = torch.cat(parts_cond_lat, dim=2)
        action_latents_iter = torch.cat(parts_noise, dim=2)
        T_total = cond_latents_full.shape[2]

        cond_vec = torch.cat(parts_cond_vec, dim=0)
        cond_ind = cond_vec.view(1, 1, T_total, 1, 1).expand(B, 1, T_total, 1, 1)
        ones_p = action_latents_iter.new_ones((B, 1, T_total, H_lat, W_lat))
        zeros_p = action_latents_iter.new_zeros((B, 1, T_total, H_lat, W_lat))
        cond_m = cond_ind * ones_p + (1.0 - cond_ind) * zeros_p

        conditional_frame_timestep = float(getattr(cosmos_cfg, "conditional_frame_timestep", 0.001)) if cosmos_cfg else 0.001
        cond_timestep = torch.ones_like(cond_ind) * conditional_frame_timestep
        cond_m_t = cond_m.to(transformer_dtype)
        padding_mask = action_latents_iter.new_zeros(1, 1, height, width, dtype=transformer_dtype)

        scheduler_cfg = getattr(scheduler, "config", None)
        prediction_type = str(getattr(scheduler_cfg, "prediction_type", "")).lower() if scheduler_cfg else ""
        use_flow_prediction = prediction_type == "flow_prediction"
        gt_velocity = None
        if use_flow_prediction:
            gt_velocity = (action_latents_iter - cond_latents_full).to(dtype=transformer_dtype) * cond_m_t

        # Iterative denoising
        for i, t in enumerate(scheduler.timesteps):
            sigma_t = torch.tensor(scheduler.sigmas[i].item()).unsqueeze(0).to(device=device, dtype=transformer_dtype)
            in_lat = cond_m_t * cond_latents_full + (1 - cond_m_t) * action_latents_iter
            in_lat = in_lat.to(transformer_dtype)
            in_ts = cond_ind * cond_timestep + (1 - cond_ind) * sigma_t

            model_out = self.backbone_interface.extractor.transformer(
                hidden_states=in_lat,
                condition_mask=cond_m_t,
                timestep=in_ts,
                encoder_hidden_states=prompt_embeds,
                padding_mask=padding_mask,
                return_dict=False,
            )[0]

            if gt_velocity is not None:
                model_out = gt_velocity + model_out * (1 - cond_m_t)

            action_latents_iter = scheduler.step(model_out, t, action_latents_iter, return_dict=False)[0]

        # Extract action latent frame and decode
        action_frame = action_latents_iter[:, :, cond_count_vid:cond_count_vid + 1]  # (B,16,1,H_lat,W_lat)
        codec_h, codec_w = self.action_codec.latent_h, self.action_codec.latent_w
        if action_frame.shape[-2] != codec_h or action_frame.shape[-1] != codec_w:
            B_, C_, T_, _, _ = action_frame.shape
            flat = action_frame.reshape(B_ * C_ * T_, 1, action_frame.shape[-2], action_frame.shape[-1])
            flat = F.interpolate(flat, size=(codec_h, codec_w), mode="bilinear", align_corners=False)
            action_frame = flat.reshape(B_, C_, T_, codec_h, codec_w)
        # Return actions in normalized [-1,1] space — client handles unnormalization
        pred_actions = self.action_codec.decode_actions(action_frame, action_stats=None)  # (B, horizon, act_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        # Pad to max_action_dim for compatibility
        max_action_dim = int(getattr(self.config.datasets.vla_data, "max_action_dim", 8))
        if normalized_actions.shape[-1] < max_action_dim:
            pad_width = max_action_dim - normalized_actions.shape[-1]
            normalized_actions = np.pad(normalized_actions, ((0, 0), (0, 0), (0, pad_width)), "constant")

        return {"normalized_actions": normalized_actions}



