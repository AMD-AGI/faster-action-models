# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU Python environment wrapper.

Preferred import::

    from instinct_sim_env import BatchedInstinctSimEnv, InstinctSimEnv

``hip_env`` and ``osc_controller`` remain importable as TOP-LEVEL modules too
(``import hip_env``), because existing callers do exactly that after putting ``<repo>/python``
on ``sys.path``.
Two things keep that working:

  * ``python/hip_env.py`` and ``python/osc_controller.py`` are shims that
    re-export from this package, so the flat import path still resolves;
  * inside this package, ``hip_env`` imports ``osc_controller`` via a relative
    import, so the package does not depend on ``sys.path`` being set up.

Do not "clean up" the shims without first repointing those call sites.
"""

from instinct_sim_env.hip_env import (  # noqa: F401
    BatchedInstinctSimEnv,
    InstinctSimEnv,
    NativeSuccessChecker,
    inject_mujoco_inertial,
)
from instinct_sim_env.osc_controller import OSCController, quat_to_mat  # noqa: F401

__all__ = [
    "BatchedInstinctSimEnv",
    "InstinctSimEnv",
    "NativeSuccessChecker",
    "inject_mujoco_inertial",
    "OSCController",
    "quat_to_mat",
]
