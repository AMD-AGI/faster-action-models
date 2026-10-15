# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy-server entry point that applies the step override, then hands off.

`deployment/model_server/server_policy.py` is byte-identical in every
DiT4DiT tree and exposes exactly four flags -- `--ckpt_path`, `--port`,
`--use_bf16`, `--idle_timeout`. It has no step flag and no `--seed`. Rather than
edit a tree whose whole value is being the exact validated code, this launcher
wraps it: import the override (a no-op unless DIT4DIT_INFER_STEPS is set), then
call upstream's own `main(build_argparser().parse_args())` unchanged. Every
server flag therefore stays upstream's, so there is no argparse surface to drift
out of sync with the runner.

`dit4dit_serve_keepalive` runs each request's policy call on a worker thread so
the server keeps answering websocket keepalive pings while a slow first call
(the GEAK arm compiles on it) is in flight. See that module for why.

`dit4dit_serve_inductor_logstr` stops inductor from printing whole IR trees
into a log line it builds for every GEAK custom op it falls back on; on the
RoboCasa head graph that string alone stalled the first call for over an hour.

`blockflax` is an optional, user-provided environment shim for hosts with a
broken flax install in ~/.local: it stubs flax out before diffusers eagerly
imports its JAX UNet. It must precede any diffusers import, hence the position
here. It is a property of the host, not of this repository, so the import is
optional -- on a host without it, nothing is stubbed and nothing needs to be.
"""
from __future__ import annotations

import os

try:  # must precede diffusers; absent on hosts that do not need the shim
    import blockflax  # noqa: F401
except ImportError:
    pass

import dit4dit_step_override  # noqa: F401  -- patches from_pretrained if env set
import dit4dit_serve_keepalive  # noqa: F401  -- policy off the event loop
import dit4dit_serve_inductor_logstr  # noqa: F401  -- bounded fallback log text

from deployment.model_server.server_policy import build_argparser, main


def seed_policy_from_environment() -> None:
    """DIT4DIT_POLICY_SEED=N seeds torch as the vendored server's --seed does.

    The vendored DiT4DiT/ server seeds torch and every CUDA device before it
    loads the model; the side-by-side trees' servers have no --seed. The RoboCasa
    full-24 driver sets this variable for them so every arm runs with the same
    policy seed. Unset, nothing is seeded (the LIBERO launcher's behavior).
    """
    value = os.environ.get("DIT4DIT_POLICY_SEED", "").strip()
    if not value:
        return
    import torch

    seed = int(value)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[dit4dit_serve_policy] policy seed {seed}", flush=True)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, force=True)
    seed_policy_from_environment()
    main(build_argparser().parse_args())
