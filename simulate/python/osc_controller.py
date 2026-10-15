# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim: ``import osc_controller`` still works.

Implementation moved to ``instinct_sim_env/osc_controller.py``. Kept for the same
reason as ``hip_env.py`` -- existing call sites import it flat after putting
``<repo>/python`` on ``sys.path``. New code should use
``from instinct_sim_env import OSCController``.
"""
from instinct_sim_env.osc_controller import *  # noqa: F401,F403
from instinct_sim_env.osc_controller import OSCController, quat_to_mat  # noqa: F401
