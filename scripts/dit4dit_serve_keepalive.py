# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# The replacement handler is copied from DiT4DiT (MIT, (c) 2025 Teli Ma &
# Mondo Robotics), whose websocket server derives from openpi (Apache-2.0).
"""Keep the policy server answering websocket keepalive pings during inference.

WHY THIS EXISTS
---------------
Every DiT4DiT tree's `WebsocketPolicyServer._handler` calls the policy
synchronously on the asyncio event loop. While `predict_action` runs, the loop
answers nothing, including the keepalive pings the tree's own
`websocket_policy_client.py` sends (`ping_interval=20`, `ping_timeout=20`).
A call that blocks the loop for longer than that drops every connected client
with "1011 keepalive ping timeout".

Steady-state calls are far shorter, but the GEAK arm's first call compiles its
action-head step with `torch.compile` (`GEAK_HEAD_IND`, on by default and
not one of the four gates the launchers pin). On a new input shape, or on a
loaded host, that first call can outlast the ping window, and the client dies
before the first action comes back. The baseline arm has no
such compile, which is why the failure is arm-specific.

WHAT IT CHANGES
---------------
Only where `_route_message` runs: on one dedicated worker thread instead of on
the event loop, and serialized by a lock, so requests are still served one at a
time in arrival order, exactly as before. The loop stays free to answer pings.
Messages, responses, error handling and the policy call are unchanged. Using a
single long-lived thread keeps thread-local torch state (current stream,
compiled-graph caches) the same on every call.

The replaced handler is a copy of the tree's with one line changed. The module
refuses to install if the tree's handler is not the one it was copied from, so
a tree update cannot silently lose behavior.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import inspect
import logging
import time
import traceback

import websockets
import websockets.frames

from deployment.model_server.tools import msgpack_numpy
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

# sha256 of inspect.getsource(WebsocketPolicyServer._handler) in every DiT4DiT
# tree (the whole module is byte-identical across them).
_UPSTREAM_HANDLER_SHA256 = "3663e350cad0e24f7cc2eabdfd42f0b4b2084ef2d2facc045d5ade384b821c8f"

_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="dit4dit-policy")


async def _handler(self, websocket):
    logging.info(f"Connection from {websocket.remote_address} opened")
    packer = msgpack_numpy.Packer()

    await websocket.send(packer.pack(self._metadata))

    loop = asyncio.get_running_loop()
    while True:
        try:
            msg = msgpack_numpy.unpackb(await websocket.recv())
            self._last_active = time.time()
            # The only change from the tree's handler: run the policy off the
            # event loop. The single-worker executor serializes calls.
            ret = await loop.run_in_executor(_executor, self._route_message, msg)
            await websocket.send(packer.pack(ret))
        except websockets.ConnectionClosed:
            logging.info(f"Connection from {websocket.remote_address} closed")
            break
        except Exception:
            await websocket.send(traceback.format_exc())
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="Internal server error. Traceback included in previous frame.",
            )
            raise


def install() -> None:
    src = inspect.getsource(WebsocketPolicyServer._handler)
    got = hashlib.sha256(src.encode()).hexdigest()
    if got != _UPSTREAM_HANDLER_SHA256:
        raise RuntimeError(
            "WebsocketPolicyServer._handler differs from the copy in "
            f"{__file__} (sha256 {got}); update the copy before serving")
    WebsocketPolicyServer._handler = _handler


install()
