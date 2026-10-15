# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep inductor's fallback log message from stalling the GEAK arm's first call.

WHY THIS EXISTS
---------------
The GEAK arm's action head calls its own custom ops (`geak_head::mm` and
friends), which have no inductor lowering. When `torch.compile` meets one,
`GraphLowering.call_function` creates an implicit fallback and logs it with

    log.info("Creating implicit fallback for:\n%s",
             error.operator_str(target, args, kwargs))

The message is built before `log.info` checks the level, so it is built even
when nothing is logged. `operator_str` does `str()` on each input `TensorBox`,
which prints the input's whole IR tree. Those trees share nodes, and `str()`
walks a shared node again every time it is referenced, so the text grows
exponentially with depth. On the RoboCasa GR1 head graph the first such call did
not finish in over an hour; the server thread sat in `ir.py __str__` with the
inductor cache still empty.

WHAT IT CHANGES
---------------
Only `OperatorIssue.operator_str`: it now names each argument by type, size and
dtype instead of printing its IR. The function is used only for that log line
and for the text of inductor's lowering exceptions. Which ops fall back, how
they are compiled and every numeric result are unchanged.

The module refuses to install if torch's `operator_str` is not the one this was
written against, so a torch update cannot silently change what is replaced.
"""
from __future__ import annotations

import hashlib
import inspect
import textwrap
from typing import Any

from torch._inductor.exc import OperatorIssue

# sha256 of inspect.getsource(OperatorIssue.operator_str) in torch 2.11.0.
_UPSTREAM_OPERATOR_STR_SHA256 = "dbb0f7c79535687274ee37079f04dd3024df2bbc67ea8dc1cd283743d2497407"


def _brief(arg: Any) -> str:
    get_size = getattr(arg, "get_size", None)
    if callable(get_size):
        try:
            size = list(get_size())
        except Exception:
            size = "?"
        try:
            dtype = arg.get_dtype()
        except Exception:
            dtype = "?"
        return f"{type(arg).__name__}(size={size}, dtype={dtype})"
    if isinstance(arg, (list, tuple)):
        inner = ", ".join(_brief(a) for a in arg)
        if isinstance(arg, list):
            return f"[{inner}]"
        return f"({inner},)" if len(arg) == 1 else f"({inner})"
    return repr(arg)


@staticmethod
def _operator_str(target: Any, args: list[Any], kwargs: dict[str, Any]) -> str:
    lines = [f"target: {target}"] + [
        f"args[{i}]: {_brief(arg)}" for i, arg in enumerate(args)
    ]
    if kwargs:
        inner = ", ".join(f"{k!r}: {_brief(v)}" for k, v in kwargs.items())
        lines.append(f"kwargs: {{{inner}}}")
    return textwrap.indent("\n".join(lines), "  ")


def install() -> None:
    src = inspect.getsource(OperatorIssue.operator_str)
    got = hashlib.sha256(src.encode()).hexdigest()
    if got != _UPSTREAM_OPERATOR_STR_SHA256:
        raise RuntimeError(
            "torch._inductor.exc.OperatorIssue.operator_str differs from the one "
            f"{__file__} replaces (sha256 {got}); re-check it before serving")
    OperatorIssue.operator_str = _operator_str


install()
