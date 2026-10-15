# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""N episodes of one LIBERO task in lockstep on ONE N-world GPU engine.

Each slot owns a real LIBERO ``OffScreenRenderEnv`` (setup, resets, init states, observation template, success
check; never stepped) and a ``GpuResidentLiberoEnv`` facade over one shared world of the engine. A round of
``step`` submits every given slot's action and runs exactly one batched physics step (``StepBarrier``); slots
that are not stepping must have left the round (``finish``), or the round would wait for them.

    venv = HipLiberoVectorEnv(env_args, num_envs=50, seed=42)
    obs = venv.reset(init_states)              # one episode per slot, slots 0..len(init_states)-1
    out = venv.step({slot: action, ...})       # {slot: (obs, reward, done, info)}
    venv.finish(slot)                          # episode over: stop waiting for this slot
    venv.close()

Every slot's reset is LIBERO's own hard reset of its reference env (fixture placements are sampled as on CPU);
with N reference envs the draws differ from one env's sequential resets, so episodes are statistically, not
pairwise, equivalent to a serial run.
"""
import os
from concurrent.futures import ThreadPoolExecutor

from .env import _original_cls, _success_metadata


class HipLiberoVectorEnv:
    def __init__(self, env_args, num_envs, seed=None, gpu_id=None):
        from .facade import GpuResidentLiberoEnv
        h, w = env_args.get("camera_heights", 128), env_args.get("camera_widths", 128)
        if h != w:
            raise ValueError(f"GPU LIBERO renders square cameras, got {h}x{w}")
        self.num_envs = int(num_envs)
        gpu_id = int(os.environ.get("HP_GPU_ID", "0")) if gpu_id is None else int(gpu_id)
        cls = _original_cls()
        self._refs = []
        for _ in range(self.num_envs):
            ref = cls(**env_args)
            if seed is not None:
                ref.seed(seed)
            self._refs.append(ref)
        meta = _success_metadata(self._refs[0])
        self._slots = []
        shared = None
        for k, ref in enumerate(self._refs):
            f = GpuResidentLiberoEnv(ref, gpu_id=gpu_id, resolution=int(h), success_metadata=meta,
                                     env_slot=k, num_envs=self.num_envs, shared=shared)
            shared = f.shared
            self._slots.append(f)
        self._barrier = self._slots[0]._barrier
        # A round waits for every submitter; the first one after a reset includes engine warm-up.
        self._barrier._timeout = float(os.environ.get("HP_BARRIER_TIMEOUT", "900"))
        # Construction registered every slot as a participant; only slots with an episode take part in rounds.
        for k in range(self.num_envs):
            self._barrier.leave(k)
        self._pool = ThreadPoolExecutor(max_workers=self.num_envs, thread_name_prefix="hip_slot")

    def reset(self, init_states):
        """Start one episode per slot (slot i gets init_states[i]); returns the first observations."""
        if len(init_states) > self.num_envs:
            raise ValueError(f"{len(init_states)} episodes for {self.num_envs} slots")
        obs = []
        for k, state in enumerate(init_states):
            f = self._slots[k]
            f.reset()
            obs.append(f.set_init_state(state))
        return obs

    def step(self, actions):
        """One physics round for the slots in ``actions`` ({slot: 7-d action}); every other slot must be finished."""
        # Register every participant BEFORE any submits: a reset left its slot out of the round, and the first
        # submitter would otherwise find itself the only active slot and run the round alone.
        for k in actions:
            self._barrier.rejoin(k)
        futures = {k: self._pool.submit(self._slots[k].step, a) for k, a in actions.items()}
        return {k: fut.result() for k, fut in futures.items()}

    def finish(self, slot):
        """The slot's episode is over: later rounds no longer wait for it (it rejoins on its next step)."""
        self._barrier.leave(slot)

    def check_success(self, slot):
        return self._slots[slot]._robosuite_success()

    def close(self):
        for f in self._slots:
            try:
                f.cleanup()
            except Exception:
                pass
            try:
                f.close()
            except Exception:
                pass
        self._pool.shutdown(wait=False)
        self._slots, self._refs = [], []
        import gc
        gc.collect()
