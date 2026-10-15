# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Set DiT4DiT's action-head denoise step count at server start.

WHY THIS EXISTS
---------------
`num_inference_steps=` is a DEAD kwarg in these trees. `DiT4DiT.predict_action`
takes `**kwargs` and, on the legacy branch (`training: joint` -> `policy_mode`
false, which is what the released LIBERO checkpoint uses), calls
`self.action_model.predict_action(last_hidden, state)` with no step argument at
all -- the kwarg is silently dropped. Only `_predict_action_policy` reads it, and
there it drives the Cosmos *video* scheduler, not the action head.

The action step count is the `num_inference_timesteps` ATTRIBUTE on the flow
matching head:

    num_steps = self.num_inference_timesteps    (ActionDiT.predict_action)
    for t in range(num_steps):

So a `--steps` that went over the wire would run 4 steps while reporting 2. This
module writes the attribute instead, after construction, and **asserts the
read-back** -- a mislabelled run is worse than a refused one.

Nothing else changes: the checkpoint config is untouched, and with
DIT4DIT_INFER_STEPS unset this module is a no-op, so the 4-step run stays
byte-identical to what was validated.

Deliberately NOT included: the `step_override.py` the validated runs used also
disabled gradient checkpointing on every module after load. The GEAK tree now
enables it only when the backbone is trainable, and the baseline tree never
enables it, which gives the same serving state without a load-time hook. Gradient checkpointing does not
change inference numerics either way.
"""
from __future__ import annotations

import os

_ENV = "DIT4DIT_INFER_STEPS"
_ATTR = "num_inference_timesteps"


def _mirror_config(model, n: int) -> bool:
    """Best-effort: keep config-derived provenance from reporting a stale 4.

    Cosmetic only -- the head attribute is what the denoise loop reads. Failing
    to mirror (struct-mode OmegaConf, missing node) is not fatal.
    """
    for holder in (getattr(getattr(model, "config", None), "framework", None),
                   getattr(model, "config", None)):
        node = getattr(holder, "action_model", None)
        if node is None:
            continue
        try:
            setattr(node, _ATTR, n)
            return True
        except Exception:
            continue
    return False


def apply(n: int, model):
    """Write the step count onto `model`'s action head; verify or raise."""
    head = getattr(model, "action_model", None)
    if head is None:
        raise RuntimeError(
            f"{_ENV}={n} was set but this model has no action head "
            "(action_model is None -- video/policy-mode checkpoint). Refusing "
            "to run: the step count would be whatever the video scheduler uses "
            "and the artifact would be mislabelled."
        )
    if not hasattr(head, _ATTR):
        raise RuntimeError(
            f"{_ENV}={n} was set but the action head has no {_ATTR!r} attribute. "
            "Refusing to run a mislabelled ablation."
        )
    setattr(head, _ATTR, n)
    got = getattr(head, _ATTR)
    if got != n:
        raise RuntimeError(
            f"{_ENV}={n} did not stick: action_model.{_ATTR} reads back as "
            f"{got!r}. Refusing to run a mislabelled ablation."
        )
    mirrored = _mirror_config(model, n)
    print(
        f"[dit4dit_step_override] action_model.{_ATTR} = {got} "
        f"(read back; config mirror {'ok' if mirrored else 'skipped'})",
        flush=True,
    )
    return model


def install() -> None:
    raw = os.environ.get(_ENV)
    if not raw:
        return
    try:
        n = int(raw)
    except ValueError:
        raise SystemExit(f"{_ENV} must be a positive integer, got: {raw!r}")
    if n < 1:
        raise SystemExit(f"{_ENV} must be a positive integer, got: {raw!r}")

    from DiT4DiT.model.framework.base_framework import baseframework

    inner = baseframework.__dict__.get("from_pretrained")
    orig = inner.__func__ if isinstance(inner, classmethod) else baseframework.from_pretrained

    def _patched(cls, *a, **k):
        return apply(n, orig(cls, *a, **k))

    baseframework.from_pretrained = classmethod(_patched)


install()
