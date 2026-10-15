# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-wide GPU LIBERO backend, for eval scripts that spawn their own Python workers.

With this directory on PYTHONPATH and HP_LIBERO_BACKEND=hip, every Python process swaps
``libero.libero.envs.OffScreenRenderEnv`` for ``instinct_sim_libero.HipOffScreenRenderEnv`` the moment LIBERO's env
package is imported (``instinct_sim_libero.install``), so worker processes need no changes. Without
HP_LIBERO_BACKEND=hip this file does nothing. An existing sitecustomize further down sys.path still runs.
"""
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_TARGET = "libero.libero.envs"


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


class _InstallAfterImport(importlib.abc.MetaPathFinder):
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
        loader = spec.loader
        exec_module = loader.exec_module

        def exec_and_install(module):
            exec_module(module)
            sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _InstallAfterImport)]
            import instinct_sim_libero
            instinct_sim_libero.install()

        loader.exec_module = exec_and_install
        return spec


if os.environ.get("HP_LIBERO_BACKEND", "").strip().lower() == "hip":
    sys.meta_path.insert(0, _InstallAfterImport())
_chain_next_sitecustomize()
