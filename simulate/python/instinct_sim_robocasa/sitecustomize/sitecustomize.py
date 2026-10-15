# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-wide GPU RoboCasa backend, for harnesses that start the sim clients as subprocesses.

With this directory on PYTHONPATH and HP_ROBOCASA_BACKEND=hip, every Python process that imports robocasa does so
under MuJoCo 3.3.2 (``instinct_sim_robocasa.compat``) and gets the GPU-native backend attached to every env it builds
through gymnasium.make (``instinct_sim_robocasa.run_client.install``), exactly as ``python -m
instinct_sim_robocasa.run_client`` does. Processes that never import robocasa (e.g. the policy server) are unaffected.

INSTINCT_SIM_MUJOCO_PATH (a mujoco==3.3.2 install) is moved to the front of sys.path, ahead of any overlay a harness
prepends. HP_PORT_GPU="<port>:<gpu>,...": a client started with ``--args.port <port>`` puts its engine on <gpu>, the device of
the policy server it talks to (HIP_VISIBLE_DEVICES, set before HIP initializes). Without HP_ROBOCASA_BACKEND=hip
this file does nothing. An existing sitecustomize further down sys.path still runs.
"""
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_TARGET = "robocasa"


def _chain_next_sitecustomize():
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        if not entry or os.path.abspath(entry) == here:
            continue
        spec = importlib.machinery.PathFinder.find_spec("sitecustomize", [entry])
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return


def _pin_gpu_from_port():
    mapping = os.environ.get("HP_PORT_GPU", "").strip()
    argv = list(getattr(sys, "argv", []))
    if not mapping or "--args.port" not in argv:
        return
    i = argv.index("--args.port")
    if i + 1 >= len(argv):
        return
    gpus = dict(item.split(":", 1) for item in mapping.split(",") if ":" in item)
    gpu = gpus.get(argv[i + 1])
    if gpu is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["HP_GPU_ID"] = "0"


class _InstallOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        exec_module = spec.loader.exec_module

        def exec_under_compat(module):
            # robocasa asserts mujoco 3.2.6 at import; the GPU simulator follows 3.3.2 (see instinct_sim_robocasa.compat)
            sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _InstallOnImport)]
            import mujoco
            real = mujoco.__version__
            mujoco.__version__ = "3.2.6"
            try:
                exec_module(module)
            finally:
                mujoco.__version__ = real
            from instinct_sim_robocasa.compat import tolerate_tiny_meshes
            from instinct_sim_robocasa.run_client import install
            tolerate_tiny_meshes()
            install()

        spec.loader.exec_module = exec_under_compat
        return spec


if os.environ.get("HP_ROBOCASA_BACKEND", "").strip().lower() == "hip":
    _pin_gpu_from_port()
    # A harness may put a pinned simulator overlay (with mujoco 3.2.6) ahead of everything else on PYTHONPATH;
    # the GPU simulator's MuJoCo 3.3.2 (INSTINCT_SIM_MUJOCO_PATH) must win.
    _mj = os.environ.get("INSTINCT_SIM_MUJOCO_PATH", "").strip()
    if _mj:
        sys.path[:] = [_mj] + [p for p in sys.path if os.path.abspath(p) != os.path.abspath(_mj)]
    sys.meta_path.insert(0, _InstallOnImport())
_chain_next_sitecustomize()
