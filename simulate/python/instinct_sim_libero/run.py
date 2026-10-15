# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a model's own LIBERO eval script with physics and camera images on the GPU simulator.

    python3 -m instinct_sim_libero.run <path/to/eval_script.py> [its arguments]

Nothing in the script changes: ``instinct_sim_libero.install()`` swaps ``libero.libero.envs.OffScreenRenderEnv`` for
``HipOffScreenRenderEnv`` before the script imports LIBERO, then the script runs as ``__main__``. Scripts that spawn
their own Python workers (e.g. FastWAM's run_libero_manager.py) need the swap in every worker process: put
``<repo>/simulate/python/instinct_sim_libero/sitecustomize`` on PYTHONPATH and set HP_LIBERO_BACKEND=hip instead.

HP_GPU_ID selects the device (index among the visible GPUs, default 0).
"""
import os
import runpy
import sys


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    import instinct_sim_libero
    instinct_sim_libero.install()
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
