# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layer drafting configuration for FastWAM inference.

Env flags:
  FASTWAM_DRAFT_LAYERS        — number of layers for action draft steps (default: 0 = disabled)
  FASTWAM_DRAFT_STEPS         — number of draft steps before refinement (default: 0)
  FASTWAM_VIDEO_DRAFT_LAYERS  — number of video DiT layers to actually run during the
                                video KV-cache prefill (default: 0 = run all num_layers).
                                Layers beyond this count reuse the last computed layer's
                                K/V, trading video-feature refinement for prefill latency.

When enabled, the denoise loop uses draft_layers for the first draft_steps steps,
then full num_layers for the remaining refinement steps.

Example: FASTWAM_DRAFT_LAYERS=8 FASTWAM_DRAFT_STEPS=1 with num_inference_steps=2
  Step 0: 8/30 layers (draft)
  Step 1: 30/30 layers (refinement)
"""
import os


_DRAFT_LAYERS = int(os.environ.get("FASTWAM_DRAFT_LAYERS", "0"))
_DRAFT_STEPS = int(os.environ.get("FASTWAM_DRAFT_STEPS", "0"))
_VIDEO_DRAFT_LAYERS = int(os.environ.get("FASTWAM_VIDEO_DRAFT_LAYERS", "0"))


def get_draft_layers() -> int:
    return _DRAFT_LAYERS


def get_draft_steps() -> int:
    return _DRAFT_STEPS


def get_video_draft_layers() -> int:
    """Number of video DiT layers to run during prefill (0 = all layers)."""
    return _VIDEO_DRAFT_LAYERS


def is_draft_enabled() -> bool:
    return _DRAFT_LAYERS > 0 and _DRAFT_STEPS > 0


def get_num_layers_for_step(step_idx: int, total_layers: int) -> int:
    if not is_draft_enabled():
        return total_layers
    if step_idx < _DRAFT_STEPS:
        return min(_DRAFT_LAYERS, total_layers)
    return total_layers
