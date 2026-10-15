# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locked site-visual bookkeeping for the GPU-resident LIBERO env.

No simulator imports, physics changes, scoring, or rendering occur here except
the caller's existing single step_all. Each returned image carries its own
copied, image-used state, before callbacks prepare the NEXT image. Callback
atomicity is per world: any partial batch failure permanently poisons the
engine's barrier instead of claiming whole-batch rollback.
"""
from __future__ import annotations

from contextlib import contextmanager
from itertools import count
import numpy as np

from .site_visual_state import from_reference


VISUAL_KEY = "_gpu_site_visual_state"
_ENGINE_GENERATIONS = count(1)


class SiteVisualCoordinator:
    def __init__(self, env, barrier):
        self._env, self._barrier = env, barrier
        self._engine_generation = next(_ENGINE_GENERATIONS)
        self._count = int(env.num_envs)
        self._nsite, self._nq = int(env.model.nsite), int(env.model.nq)
        self._names = tuple(env.model.site_name(i) for i in range(self._nsite))
        if (self._count <= 0 or barrier._n_envs != self._count
                or any(not isinstance(name, str) or not name for name in self._names)
                or len(set(self._names)) != self._nsite):
            raise ValueError("Invalid shared visual model/world identity")
        self._states = [None] * self._count
        self._generation = [0] * self._count
        self._reset_generation = [0] * self._count
        self._round = 0
        self._failure = None
        with barrier._cv:
            self._rgba = self._rgba_array(
                env._renderer.get_site_rgba_batch(env.engine),
                (self._count, self._nsite, 4))

    @staticmethod
    def _rgba_array(value, shape):
        value = np.asarray(value)
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Malformed native-order float32 visual RGBA")
        return value.copy(order="C")

    def _locked(self):
        if not self._barrier._cv._is_owned():
            raise RuntimeError("Visual bookkeeping requires the shared engine lock")

    def ensure_healthy(self):
        self._locked()
        self._barrier.raise_if_poisoned()
        if self._failure is not None:
            raise self._failure

    def fail(self, error):
        self._locked()
        if self._failure is None:
            self._failure = error
        self._barrier.poison(self._failure)

    @contextmanager
    def transaction(self):
        """Preserve the engine lock and make any native-operation error fatal."""
        with self._barrier._cv:
            self.ensure_healthy()
            try:
                yield
            except BaseException as error:
                self.fail(error)
                raise

    def _slot(self, slot):
        if type(slot) is not int or not 0 <= slot < self._count:
            raise ValueError("Visual world index outside the actual engine")

    def reset_from_reference(self, slot, reference):
        """Publish a fresh post-init snapshot BEFORE both existing reset gathers.

        The caller has already run reference.set_init_state. This does not
        advance callbacks, physics, controllers, or the global physical round.
        """
        self.ensure_healthy()
        try:
            self._slot(slot)
            state = from_reference(reference, self._env.model)
            rgba = self._rgba_array(state.hp_rgba, (self._nsite, 4))
            self._env._renderer.set_site_rgba(self._env.engine, rgba, env_idx=slot)
            self._states[slot] = state
            self._rgba[slot] = rgba
            self._generation[slot] = 0
            self._reset_generation[slot] += 1
        except BaseException as error:
            self.fail(error)
            raise

    def snapshot(self, slot, *, physics_round=None, kind="reset_final"):
        self.ensure_healthy()
        self._slot(slot)
        return {
            "schema": "resident-image-site-state-v1",
            "engine_generation": self._engine_generation,
            "world": slot,
            "frame_kind": kind,
            "site_names": self._names,
            "rgba": self._rgba[slot].copy(),
            "visual_generation": self._generation[slot],
            "reset_generation": self._reset_generation[slot],
            "physics_round": self._round if physics_round is None else physics_round,
            "initialized": self._states[slot] is not None,
        }

    def attach(self, obs, slot):
        """Attach a copied snapshot to an existing no-step reset observation."""
        return self._attach_snapshot(obs, self.snapshot(slot))

    @staticmethod
    def _attach_snapshot(obs, snapshot, *, copy_arrays=True):
        if VISUAL_KEY in obs:
            raise ValueError("An observation already has visual-state metadata")
        copied = ({k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()}
                  if copy_arrays else dict(obs))
        snapshot = {**snapshot, "rgba": snapshot["rgba"].copy() if copy_arrays else snapshot["rgba"],
                    "camera_keys": tuple(k for k, v in copied.items()
                                         if k.endswith("_image") and isinstance(v, np.ndarray)
                                         and v.ndim == 3 and v.shape[-1] == 3)}
        return {**copied, VISUAL_KEY: snapshot}

    def step_batch(self, actions, render_rows=None, step_rows=None):
        """Exactly one original step/render, then one advance per initialized row.

        render_rows: rows whose images this round's caller consumes;
        forwarded to step_all only when given (None = unchanged call).
        step_rows (world skip): rows to simulate; the rest are frozen this round.
        """
        self.ensure_healthy()
        try:
            if np.shape(actions) != (self._count, 7):
                raise ValueError("Action batch does not cover the actual engine rows")
            next_round = self._round + 1
            snapshots = [self.snapshot(i, physics_round=next_round, kind="step")
                         for i in range(self._count)]
            kw = {}
            if render_rows is not None:
                kw["render_rows"] = render_rows
            if step_rows is not None:
                kw["step_rows"] = step_rows
            result = self._env.step_all(actions, **kw)
            if not isinstance(result, tuple) or len(result) != 4:
                raise ValueError("Unexpected original step_all return contract")
            obs, rewards, dones, infos = result
            if (len(obs) != self._count or any(not isinstance(row, dict) or VISUAL_KEY in row
                                              for row in obs)):
                raise ValueError("Original observations do not match the engine rows")
            copied_obs = [self._attach_snapshot(row, snapshots[i], copy_arrays=False)
                          for i, row in enumerate(obs)]
            qpos = np.asarray(self._env.engine.data.read_qpos_d())
            if (qpos.shape != (self._count, self._nq) or qpos.dtype != np.float64
                    or not np.isfinite(qpos).all()):
                raise ValueError("Visual update requires the full finite native double-qpos batch")
            qpos = qpos.copy()
            next_rgba = self._rgba.copy()
            # Every physical row advances in step_all, not only barrier participants.
            for slot, state in enumerate(self._states):
                if state is not None:
                    next_rgba[slot] = self._rgba_array(state.advance(qpos[slot]), (self._nsite, 4))
            if any(state is not None for state in self._states):
                self._env._renderer.set_site_rgba_batch(self._env.engine, next_rgba)
            self._rgba = next_rgba
            for slot, state in enumerate(self._states):
                if state is not None:
                    self._generation[slot] += 1
            self._round = next_round
            return copied_obs, rewards, dones, infos
        except BaseException as error:
            self.fail(error)
            raise


def snapshot_arrays(observation):
    """Serialize ONLY image-associated metadata, never shared current state."""
    state = observation[VISUAL_KEY]
    if not isinstance(state, dict) or state.get("schema") != "resident-image-site-state-v1":
        raise ValueError("Missing/unsupported image-used visual state")
    names = state["site_names"]
    if (not isinstance(names, tuple) or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("Invalid image-used site-name order")
    rgba = SiteVisualCoordinator._rgba_array(state["rgba"], (len(names), 4))
    fields = ("visual_generation", "reset_generation", "physics_round", "engine_generation", "world")
    if any(type(state[k]) is not int or not 0 <= state[k] <= np.iinfo(np.int64).max for k in fields):
        raise ValueError("Invalid image-used generation")
    if type(state["initialized"]) is not bool:
        raise ValueError("Invalid image-used initialization state")
    if state.get("frame_kind") not in ("reset_final", "step"):
        raise ValueError("Invalid image-used frame kind")
    cameras = state.get("camera_keys")
    actual_cameras = {k for k, v in observation.items()
                      if isinstance(k, str) and k.endswith("_image")
                      and isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 3}
    if (not isinstance(cameras, tuple) or len(set(cameras)) != len(cameras)
            or set(cameras) != actual_cameras
            or any(not isinstance(key, str) or key not in observation
                   or np.asarray(observation[key]).ndim != 3
                   or np.asarray(observation[key]).shape[-1] != 3 for key in cameras)):
        raise ValueError("Visual receipt does not match the observed cameras")
    return {
        "visual_site_names": np.asarray(names, dtype=str),
        "visual_site_rgba": rgba,
        **{"visual_" + k if not k.startswith("visual_") else k:
           np.asarray(state[k], dtype=np.int64) for k in fields},
        "visual_initialized": np.asarray(state["initialized"], dtype=np.bool_),
        "visual_frame_kind": np.asarray(state["frame_kind"], dtype=str),
        "visual_camera_keys": np.asarray(cameras, dtype=str),
    }
