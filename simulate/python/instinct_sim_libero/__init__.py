# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LIBERO on the GPU simulator: physics and camera images on the GPU, everything else LIBERO's own.

Two entry points, both driving LIBERO's unmodified ``OffScreenRenderEnv`` for scene construction, resets,
init states, observation templates and success checks:

* ``HipOffScreenRenderEnv`` -- a drop-in for ``OffScreenRenderEnv`` (one world). ``install()`` swaps it in for
  ``libero.libero.envs.OffScreenRenderEnv``, so a model's own LIBERO eval script runs unchanged; see
  ``instinct_sim_libero.run`` and ``sitecustomize/``.
* ``HipLiberoVectorEnv`` -- N episodes of one task in lockstep on ONE N-world engine (batched evals).

Settings below are the validated LIBERO defaults; set the variable to override.
"""
import os

_DEFAULTS = {
    "HP_RENDERER": "rt",          # HIP ray tracer
    "HP_OBS_LAG": "2",            # robosuite-timed proprio and camera images (pose/image lag = 3 substeps)
    "HP_PERCAM_MAX_ENVS": "0",
    "HP_SUCCESS_MODE": "robosuite",   # LIBERO's own _check_success on the GPU simulator's state
}
for _k, _v in _DEFAULTS.items():
    os.environ.setdefault(_k, _v)

from .env import HipOffScreenRenderEnv, install  # noqa: E402
from .vector import HipLiberoVectorEnv  # noqa: E402

__all__ = ["HipOffScreenRenderEnv", "HipLiberoVectorEnv", "install"]
