# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step barrier for N GPU-resident LIBERO slots sharing ONE BatchedInstinctSimEnv.

WHY A BARRIER. ``BatchedInstinctSimEnv.step_all(actions)`` advances EVERY env in
one call. If N slot facades
each called it, physics would advance N steps per logical round and every slot
would see peers' extra steps. So stepping needs a REAL barrier: all active slots
submit an action, exactly one call to step_all happens, all slots then read their
own row of the result.

THE HARD PART IS NOT THE BARRIER, IT IS THE PARTICIPANT SET.
Episodes have very different lengths -- 128 to 284 steps measured on
libero_object/0 -- so slots leave the barrier at different times. A fixed
participant count deadlocks the moment the first slot finishes. Every exit path
must deregister, INCLUDING exceptions, which is why ``leave()`` is idempotent and
callers use try/finally.

THREADING MODEL. Each slot steps from its own worker thread (HipLiberoVectorEnv
uses a thread pool); a blocking ``threading.Condition`` here parks that thread
until every participant has submitted.

A TIMEOUT IS A BUG, NOT A FALLBACK. If a slot fails to arrive the run is already
wrong, and silently stepping the subset would advance physics for slots that
never submitted an action. The timeout therefore RAISES rather than proceeding.
"""
from __future__ import annotations

import threading
from typing import Any, Callable

import numpy as np


class StepBarrier:
    """Rendezvous N slots, run one batched step, hand each slot its own result.

    Usage per slot, per step::

        obs_row = barrier.step(slot, action_7d)

    and at episode/worker exit::

        barrier.leave(slot)
    """

    def __init__(self, num_slots: int, step_fn: Callable[[np.ndarray], Any],
                 timeout: float = 120.0, startup_timeout: float = 900.0):
        self._n = int(num_slots)
        self._n_envs = int(num_slots)
        self._step_fn = step_fn
        self._timeout = float(timeout)
        self._startup_timeout = float(startup_timeout)
        self._cv = threading.Condition()
        # ONLINE vs ACTIVE. Slots are NOT participants until their facade is
        # constructed: each builds its own robosuite env inside its own worker
        # thread, serialised behind slot 0's engine build and 195 MB texture
        # upload, so slot 0 reaches its first step MINUTES before slot 3 exists.
        # Pre-registering all N made slot 0 wait on peers that had not been
        # created yet and die with "active=[1,2,3] submitted=[]".
        # So: _online grows as slots call online(); the first step blocks until
        # every expected slot is online, then normal lockstep applies.
        self._online: set[int] = set()
        self._started = False
        self._active: set[int] = set()
        self._actions: dict[int, np.ndarray] = {}
        self._inflight: dict[int, dict] = {}
        self._pending_tickets: dict[int, dict] = {}
        self._result: Any = None
        self._epoch = 0
        self._error: BaseException | None = None
        # Only explicit visual-transaction failure sets this permanent latch.
        self._fatal_error: BaseException | None = None
        self.rounds = 0
        self.participants_total = 0

    # ---- participant set --------------------------------------------------
    def raise_if_poisoned(self) -> None:
        with self._cv:
            if self._fatal_error is not None:
                raise self._fatal_error

    def poison(self, error: BaseException) -> None:
        """Fail every waiter and forbid later physical rounds for this engine."""
        with self._cv:
            if self._fatal_error is None:
                self._fatal_error = error
            self._error = self._fatal_error
            self._result = None
            self._actions.clear()
            self._pending_tickets.clear()
            self._epoch += 1
            self._cv.notify_all()

    def online(self, slot: int) -> None:
        """Declare this slot constructed and ready to participate.

        Called from the facade once its env exists. Until every expected slot
        has done this, the first ``step`` blocks in ``_await_start_locked``
        rather than treating absent-because-not-yet-built as absent-because-hung.
        """
        with self._cv:
            self.raise_if_poisoned()
            self._online.add(slot)
            self._active.add(slot)
            if len(self._online) >= self._n:
                self._started = True
            self._cv.notify_all()

    def _await_start_locked(self, slot: int) -> None:
        """Block until all N slots are online. Caller holds the lock."""
        self.raise_if_poisoned()
        if self._started:
            return
        if not self._cv.wait_for(lambda: self._started or self._fatal_error is not None,
                                 timeout=self._startup_timeout):
            raise TimeoutError(
                f"StepBarrier: slot {slot} waited {self._startup_timeout}s for "
                f"peers to come online; online={sorted(self._online)} of {self._n}. "
                "Each slot builds its own robosuite env serialised behind slot 0's "
                "engine build and texture upload, so this window is minutes, not "
                "seconds -- but it is bounded."
            )
        self.raise_if_poisoned()

    def leave(self, slot: int, permanent: bool = False) -> None:
        """Deregister a slot. IDEMPOTENT: called from finally blocks.

        ``permanent=True`` means the slot has exhausted its work queue and will
        never step again, so it also drops out of ``_online`` and the expected
        cohort size. A TRANSIENT leave (episode boundary in ``set_init_state``)
        must NOT shrink the cohort -- the slot is coming right back.
        """
        with self._cv:
            if slot in self._inflight and self._fatal_error is None:
                error = RuntimeError("Concurrent reset/departure and step for one slot is unsupported")
                self.poison(error)
                raise error
            was_active = slot in self._active
            self._active.discard(slot)
            self._actions.pop(slot, None)
            if permanent:
                # Shrink the cohort, or peers block forever on a slot that has
                # gone home.
                self._online.discard(slot)
                self._n = max(1, len(self._online)) if self._online else 1
                self._started = True
            if was_active or permanent:
                # A departure can complete a round that the remaining slots are
                # already blocked on -- without this they wait out the timeout.
                if self._fatal_error is None and self._actions and self._ready_locked():
                    self._fire_locked()
                self._cv.notify_all()

    def rejoin(self, slot: int) -> None:
        with self._cv:
            self.raise_if_poisoned()
            self._active.add(slot)
            self._cv.notify_all()

    @property
    def active(self) -> int:
        with self._cv:
            return len(self._active)

    # ---- the barrier ------------------------------------------------------
    def _ready_locked(self) -> bool:
        return self._active and all(s in self._actions for s in self._active)

    def submitted_slots_locked(self) -> tuple[int, ...]:
        """The actual submitted rows for this round, not all physical rows.

        Only the locked step callback consumes this snapshot. Keeping the set
        identical to _fire_locked's batch construction avoids extra success
        checks for inactive or zero-filled unsubmitted worlds.
        """
        if not self._cv._is_owned():
            raise RuntimeError("Submitted-slot snapshot requires the shared engine lock")
        return tuple(sorted(self._actions))

    def _fire_locked(self) -> None:
        """Run ONE step_all for every submitted action. Caller holds the lock.

        Holding the lock across the GPU call is deliberate: step_all mutates
        shared engine state (_step_counts, the OSC env-state buffer), so no peer
        may touch the engine concurrently. The call is the round's whole cost,
        so there is nothing to overlap anyway.
        """
        slots = sorted(self._actions)
        tickets = [self._pending_tickets[s] for s in slots]
        try:
            self.raise_if_poisoned()
            # _n_envs, NOT _n: sized to the engine, which keeps every row alive
            # even after slots depart.
            batch = np.zeros((self._n_envs, 7), dtype=np.float32)
            for s in slots:
                batch[s] = self._actions[s]
            self._result = self._step_fn(batch)
            self._error = None
        except BaseException as exc:            # noqa: BLE001 - re-raised in every waiter
            self._result = None
            self._error = exc
        self.rounds += 1
        self.participants_total += len(slots)
        # A forced later round may run before an earlier waiter reacquires the
        # condition. Each submission retains ITS outcome, not shared latest.
        for ticket in tickets:
            ticket.update(complete=True, result=self._result, error=self._error)
        self._actions.clear()
        self._pending_tickets.clear()
        self._epoch += 1
        self._cv.notify_all()

    def step(self, slot: int, action: np.ndarray, await_cohort: bool | None = None) -> Any:
        """Submit this slot's action; block until the round runs; return result.

        NEVER BLOCKS ON THE MAIN THREAD: a call from the main thread runs its
        round immediately (a single-world env stepped by a plain eval loop, or
        the main thread of a driver that has not started its peers yet). Worker
        threads wait for the cohort. Pass ``await_cohort`` explicitly to
        override.
        """
        if await_cohort is None:
            await_cohort = threading.current_thread() is not threading.main_thread()
        with self._cv:
            self.raise_if_poisoned()
            if type(slot) is not int or not 0 <= slot < self._n_envs:
                raise ValueError("Step slot outside the actual engine")
            if slot in self._inflight:
                error = RuntimeError("Concurrent steps for one slot are unsupported")
                self.poison(error)
                raise error
            ticket = {"complete": False}
            self._inflight[slot] = ticket
            try:
                # Readiness, warmup forcing and zero-filled engine row count
                # are unchanged. Only delivery is bound to the actual round.
                if await_cohort:
                    self._await_start_locked(slot)
                if slot not in self._active:
                    self._active.add(slot)
                self._actions[slot] = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
                self._pending_tickets[slot] = ticket
                if self._ready_locked() or not await_cohort:
                    self._fire_locked()
                elif not self._cv.wait_for(
                        lambda: ticket["complete"] or self._fatal_error is not None,
                        timeout=self._timeout):
                    self._active.discard(slot)
                    self._actions.pop(slot, None)
                    self._pending_tickets.pop(slot, None)
                    self._cv.notify_all()
                    raise TimeoutError(
                        f"StepBarrier: slot {slot} waited {self._timeout}s; "
                        f"active={sorted(self._active)} submitted={sorted(self._actions)}. "
                        "A slot failed to arrive -- stepping the subset would "
                        "advance physics for slots that never submitted, so this raises instead.")
                self.raise_if_poisoned()
                if ticket["error"] is not None:
                    raise ticket["error"]
                return ticket["result"]
            finally:
                self._inflight.pop(slot, None)

    def stats(self) -> dict:
        with self._cv:
            avg = (self.participants_total / self.rounds) if self.rounds else 0.0
            return {"rounds": self.rounds, "avg_participants": avg,
                    "active": len(self._active)}
