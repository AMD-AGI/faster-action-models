# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""torch.compile configuration for FastWAM inference.

Env flags:
  FASTWAM_COMPILE      — "1"/"true" enables, "0"/"false" disables.
                         Default: ON for inference, OFF for training.
  FASTWAM_COMPILE_MODE — torch.compile mode string.
                         Default: "max-autotune-no-cudagraphs"

The training entry point calls apply_training_launch_default() to disable
compile for the whole process unless FASTWAM_COMPILE was explicitly set.
"""
import os
import functools
import torch

_EXPLICIT = "FASTWAM_COMPILE" in os.environ

_ENABLED = os.environ.get("FASTWAM_COMPILE", "1").lower() in ("1", "true", "yes", "on")

_MODE = os.environ.get("FASTWAM_COMPILE_MODE", "max-autotune-no-cudagraphs")

_compile_cache = {}


def is_compile_enabled() -> bool:
    return _ENABLED


def set_compile_enabled(enabled: bool):
    global _ENABLED
    _ENABLED = enabled


def apply_training_launch_default():
    """Call from the trainer to disable compile unless explicitly set."""
    global _ENABLED
    if not _EXPLICIT:
        _ENABLED = False


def maybe_compile(fn):
    """Decorator: wraps fn with torch.compile when enabled, lazy + memoized."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not _ENABLED:
            return fn(*args, **kwargs)
        compiled = _compile_cache.get(fn)
        if compiled is None:
            compiled = torch.compile(fn, mode=_MODE, dynamic=False)
            _compile_cache[fn] = compiled
        return compiled(*args, **kwargs)
    return wrapper
