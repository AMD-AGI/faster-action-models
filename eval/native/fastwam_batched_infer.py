# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""True B>1 inference for FastWAM: N DISTINCT observations in one model call.

`FastWAM.infer_action` is hard-gated to B=1 (`input_image.shape[0] != 1`), so a
batched eval cannot call it with N observations.

A multi-env eval has N DIFFERENT observations, so every stage -- VAE, video
prefill, denoise -- runs at B=N here.

This reimplements the same pipeline as `infer_action`, batched, by calling the
stage methods directly instead of going through the gate.

Each row draws its noise from its own generator, exactly as N serial calls
would; pass the same seed for every row to reproduce N serial calls made with
that seed (FastWAM's eval passes cfg.seed on every call).
"""
from __future__ import annotations

from typing import Any


def infer_action_multi(
    model,
    prompts: list[str],
    input_images,          # [B,3,H,W] tensor, B DISTINCT observations
    action_horizon: int,
    proprios,              # [B,D] tensor or None
    num_inference_steps: int,
    sigma_shift: float | None = None,
    seeds: list[int] | None = None,
    rand_device: str = "cpu",
    tiled: bool = False,
) -> dict[str, Any]:
    """Batched twin of FastWAM.infer_action for B distinct observations.

    Mirrors infer_action's stage order exactly: VAE encode -> pre_dit ->
    attention mask -> prefill_video_cache -> denoise loop -> post. Divergence
    from that order is what would break equivalence, so it is kept literal.
    """
    import torch

    model.eval()
    if str(getattr(model.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
        raise ValueError("requires video_attention_mask_mode='first_frame_causal'")

    if input_images.ndim != 4 or input_images.shape[1] != 3:
        raise ValueError(f"input_images must be [B,3,H,W], got {tuple(input_images.shape)}")
    B, _, height, width = input_images.shape
    if height % 16 or width % 16:
        raise ValueError(f"H,W must be multiples of 16, got {height}x{width}")
    if len(prompts) != B:
        raise ValueError(f"{len(prompts)} prompts for B={B}")

    device, dtype = model.device, model.torch_dtype

    # Per-observation noise, drawn the way N serial calls would draw it: one
    # generator per seed, stacked.
    if seeds is None:
        seeds = [None] * B
    rows = []
    for s in seeds:
        g = None if s is None else torch.Generator(device=rand_device).manual_seed(int(s))
        rows.append(torch.randn((1, action_horizon, model.action_expert.action_dim),
                                generator=g, device=rand_device, dtype=torch.float32))
    latents_action = torch.cat(rows, dim=0).to(device=device, dtype=dtype)

    x = input_images.to(device=device, dtype=dtype)

    # VAE: the public list API, one entry per observation, as the serial route.
    lat = [x[i].unsqueeze(1) for i in range(B)]          # each [3,1,H,W]
    first_frame_latents = model.vae.encode(lat, device=device, tiled=tiled)
    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))

    # Prompts are usually identical across a batch (same task); encode once and
    # repeat when so, else encode per prompt and stack.
    uniq = set(prompts)
    if len(uniq) == 1:
        ctx1, cm1 = model.encode_prompt(prompts[0])
        context = ctx1.repeat(B, *([1] * (ctx1.ndim - 1)))
        context_mask = cm1.repeat(B, *([1] * (cm1.ndim - 1)))
    else:
        cs, ms = zip(*(model.encode_prompt(p) for p in prompts))
        context = torch.cat(list(cs), dim=0)
        context_mask = torch.cat(list(ms), dim=0)

    if proprios is not None:
        proprios = proprios.to(device=device, dtype=dtype)
        context, context_mask = model._append_proprio_to_context(
            context=context, context_mask=context_mask, proprio=proprios)

    timestep_video = torch.zeros((first_frame_latents.shape[0],),
                                 dtype=first_frame_latents.dtype, device=device)
    video_pre = model.video_expert.pre_dit(
        x=first_frame_latents, timestep=timestep_video, context=context,
        context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=fuse_flag)
    video_seq_len = int(video_pre["tokens"].shape[1])
    attention_mask = model._build_mot_attention_mask(
        video_seq_len=video_seq_len, action_seq_len=latents_action.shape[1],
        video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
        device=video_pre["tokens"].device)
    video_kv_cache = model.mot.prefill_video_cache(
        video_tokens=video_pre["tokens"], video_freqs=video_pre["freqs"],
        video_t_mod=video_pre["t_mod"],
        video_context_payload={"context": video_pre["context"],
                               "mask": video_pre["context_mask"]},
        video_attention_mask=attention_mask[:video_seq_len, :video_seq_len])

    ts, deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps, device=device,
        dtype=latents_action.dtype, shift_override=sigma_shift)
    for step_t, step_delta in zip(ts, deltas):
        timestep_action = step_t.reshape(1).expand(B).to(dtype=latents_action.dtype, device=device)
        pred = model._predict_action_noise_with_cache(
            latents_action=latents_action, timestep_action=timestep_action,
            context=context, context_mask=context_mask,
            video_kv_cache=video_kv_cache, attention_mask=attention_mask,
            video_seq_len=video_seq_len)
        latents_action = model.infer_action_scheduler.step(
            pred, step_delta, latents_action)

    return {"action": latents_action}
