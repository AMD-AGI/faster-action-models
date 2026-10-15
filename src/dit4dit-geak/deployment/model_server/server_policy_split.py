# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy server with split-GPU model loading for Cosmos-policy.

Loads text_encoder+VAE on GPU1, transformer on GPU0.
Uses the custom predict_action_split for cross-device inference.

Usage:
    HIP_VISIBLE_DEVICES=4,5 python -m deployment.model_server.server_policy_split \
        --ckpt_path results/Checkpoints/cosmos-policy/checkpoints/steps_30000_pytorch_model.pt \
        --port 10093
"""
import logging
import socket
import argparse
import gc

import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


class SplitGPUPolicyWrapper:
    """Wraps a split-GPU model to provide the predict_action interface expected by the server."""

    def __init__(self, model):
        self.model = model
        from smoke_test_inference import predict_action_split
        self._predict_fn = predict_action_split
        self._call_count = 0

    def predict_action(self, examples, **kwargs):
        num_steps = kwargs.get("num_ddim_steps", 10)
        result = self._predict_fn(self.model, examples, num_inference_steps=num_steps)
        self._call_count += 1
        if self._call_count % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        return result


def main(args):
    from smoke_test_inference import load_model_split

    logging.info("Loading model with split-GPU placement...")
    model = load_model_split(args.ckpt_path)
    logging.info("Model loaded successfully")

    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        logging.info(f"GPU {i}: {alloc:.2f} GB")

    wrapper = SplitGPUPolicyWrapper(model)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(f"Creating server (host: {hostname}, ip: {local_ip}, port: {args.port})")

    server = WebsocketPolicyServer(
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "libero"},
    )
    logging.info("Server running...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--idle_timeout", type=int, default=3600)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
