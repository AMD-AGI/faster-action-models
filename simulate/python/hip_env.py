# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim: ``import hip_env`` still works.

The implementation moved to ``instinct_sim_env/hip_env.py`` when the wrapper became
a real package. This module stays because existing callers do::

    sys.path.insert(0, f"{hip_repo}/python")
    import hip_env

and breaking those to save one file is not a trade worth making. New code should
prefer ``from instinct_sim_env import BatchedInstinctSimEnv``.

`import *` is deliberate: callers reach for a wide surface here (module-level
constants, helpers, both env classes), and enumerating it would silently drop
whatever was missed.
"""
from instinct_sim_env.hip_env import *  # noqa: F401,F403
from instinct_sim_env.hip_env import (  # noqa: F401  explicit: not covered by *
    BatchedInstinctSimEnv,
    InstinctSimEnv,
    NativeSuccessChecker,
    inject_mujoco_inertial,
)
