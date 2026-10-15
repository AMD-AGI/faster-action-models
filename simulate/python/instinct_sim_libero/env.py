# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drop-in ``OffScreenRenderEnv`` whose physics and camera images run on the GPU simulator.

``HipOffScreenRenderEnv(**env_args)`` takes LIBERO's own constructor arguments and builds the real
``OffScreenRenderEnv`` from them. That reference env provides the scene, every reset (fixture sampling), the
init states, the observation template and the success check; it is never stepped. The GPU engine is built
from it lazily, on the first reset, so ``env.seed(...)`` lands on the reference env first, as on CPU.

``install()`` replaces ``libero.libero.envs.OffScreenRenderEnv`` (and the ``env_wrapper`` module attribute), so
eval scripts that do ``from libero.libero.envs import OffScreenRenderEnv`` get this class without edits. It must
run before such a script imports LIBERO; ``instinct_sim_libero.run`` and ``sitecustomize/`` arrange that.

HP_GPU_ID selects the device (index among the visible GPUs, default 0).
"""
import os

_ORIGINAL = None


def _original_cls():
    global _ORIGINAL
    if _ORIGINAL is None:
        from libero.libero.envs import env_wrapper
        _ORIGINAL = env_wrapper.OffScreenRenderEnv
    return _ORIGINAL


def _success_metadata(ref_env):
    """Metadata for the GPU simulator's own success checker (HP_SUCCESS_MODE=native). The default mode asks LIBERO's
    _check_success instead and does not need it, so an extraction failure only matters in native mode."""
    from .success_metadata import extract_success_metadata
    try:
        return extract_success_metadata(ref_env)
    except Exception as error:
        if os.environ.get("HP_SUCCESS_MODE") == "native":
            raise RuntimeError(f"HP_SUCCESS_MODE=native needs success metadata: {error}") from error
        return None


class HipOffScreenRenderEnv:
    """LIBERO OffScreenRenderEnv with GPU physics + rendering (one world)."""

    def __init__(self, **env_args):
        h, w = env_args.get("camera_heights", 128), env_args.get("camera_widths", 128)
        if h != w:
            raise ValueError(f"GPU LIBERO renders square cameras, got {h}x{w}")
        self._ref = _original_cls()(**env_args)
        self._resolution = int(h)
        self._gpu_id = int(os.environ.get("HP_GPU_ID", "0"))
        self._facade = None

    def _hip(self):
        if self._facade is None:
            from .facade import GpuResidentLiberoEnv
            self._facade = GpuResidentLiberoEnv(
                self._ref, gpu_id=self._gpu_id, resolution=self._resolution,
                success_metadata=_success_metadata(self._ref), env_slot=0, num_envs=1, shared=None)
        return self._facade

    # ---- the OffScreenRenderEnv surface eval loops use ----------------------------------------------------------
    def seed(self, seed):
        return self._ref.seed(seed)

    def reset(self):
        return self._hip().reset()

    def set_init_state(self, init_state):
        return self._hip().set_init_state(init_state)

    def step(self, action):
        return self._hip().step(action)

    def check_success(self):
        """LIBERO's own success predicate, evaluated on the GPU simulator's current state."""
        return self._hip()._robosuite_success()

    def close(self):
        if self._facade is not None:
            self._facade.close()      # also closes the reference env
            self._facade = None
        else:
            self._ref.close()

    @property
    def robots(self):
        return self._ref.robots

    @property
    def env(self):
        return self._ref.env

    def __getattr__(self, name):
        # Anything else would read the reference env, which is never stepped: its sim state is stale.
        raise AttributeError(
            f"HipOffScreenRenderEnv does not provide {name!r} (the reference OffScreenRenderEnv is setup-only and "
            "never stepped, so forwarding would return stale CPU state)")


def install():
    """Make ``libero.libero.envs.OffScreenRenderEnv`` the GPU env. Idempotent."""
    import libero.libero.envs as envs
    from libero.libero.envs import env_wrapper
    _original_cls()
    envs.OffScreenRenderEnv = HipOffScreenRenderEnv
    env_wrapper.OffScreenRenderEnv = HipOffScreenRenderEnv
    print(f"[instinct_sim_libero] OffScreenRenderEnv -> the GPU simulator (renderer={os.environ.get('HP_RENDERER')} "
          f"aa={os.environ.get('HP_RT_AA') or 'ssaa1'} obs_lag={os.environ.get('HP_OBS_LAG')} "
          f"success={os.environ.get('HP_SUCCESS_MODE')} gpu={os.environ.get('HP_GPU_ID', '0')})", flush=True)
