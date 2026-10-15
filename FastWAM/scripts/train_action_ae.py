# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train the K8d16-compatible action VAE from cached action data."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastwam.models.action_ae import ActionAE, ActionAEConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("train_action_ae")


@torch.no_grad()
def update_ema(ema_model: ActionAE, model: ActionAE, decay: float) -> None:
    for ema_parameter, parameter in zip(
        ema_model.parameters(), model.parameters()
    ):
        ema_parameter.lerp_(parameter, 1 - decay)


def load_data(
    path: str, batch_size: int, predecoded: bool
) -> tuple[DataLoader, bool]:
    payload = torch.load(path, map_location="cpu")
    actions = payload["actions"]
    if predecoded:
        action_is_pad = payload["action_is_pad"]
        has_suite_ids = "suite_ids" in payload
        tensors = (
            (actions, action_is_pad, payload["suite_ids"])
            if has_suite_ids
            else (actions, action_is_pad)
        )
    else:
        has_suite_ids = False
        tensors = (actions, payload["video_cond"])
    logger.info("Loaded %d action chunks from %s", len(actions), path)
    return (
        DataLoader(
            TensorDataset(*tensors),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        ),
        has_suite_ids,
    )


@torch.no_grad()
def evaluate(
    model: ActionAE,
    loader: DataLoader,
    device: str,
    predecoded: bool,
    has_suite_ids: bool,
) -> dict:
    model.eval()
    total_mse = total_velocity_mse = 0.0
    per_dof = None
    sample_count = 0
    for batch in loader:
        actions = batch[0].to(device)
        video_cond = None if predecoded else batch[1].to(device)
        suite_ids = (
            batch[2].to(device)
            if predecoded and has_suite_ids
            else None
        )
        reconstruction = model(
            actions, video_cond, suite_ids=suite_ids
        )["recon"]
        mse = F.mse_loss(reconstruction, actions, reduction="none")
        current_per_dof = mse.mean(dim=(0, 1))
        per_dof = (
            current_per_dof * actions.shape[0]
            if per_dof is None
            else per_dof + current_per_dof * actions.shape[0]
        )
        total_mse += mse.mean().item() * actions.shape[0]
        total_velocity_mse += F.mse_loss(
            reconstruction[:, 1:] - reconstruction[:, :-1],
            actions[:, 1:] - actions[:, :-1],
        ).item() * actions.shape[0]
        sample_count += actions.shape[0]
    return {
        "mse": total_mse / max(sample_count, 1),
        "vel_mse": total_velocity_mse / max(sample_count, 1),
        "per_dof_mse": (
            (per_dof / max(sample_count, 1)).cpu().tolist()
            if per_dof is not None
            else []
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--predecoded")
    data.add_argument("--cached-features")
    parser.add_argument("--output", required=True)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--encoder-hidden", type=int, default=128)
    parser.add_argument("--decoder-hidden", type=int, default=128)
    parser.add_argument("--video-cond-dim", type=int, default=12288)
    parser.add_argument("--variational", action="store_true")
    parser.add_argument("--beta-kl", type=float, default=1e-2)
    parser.add_argument("--kl-warmup-epochs", type=int, default=100)
    parser.add_argument("--free-bits", type=float, default=0.0)
    parser.add_argument(
        "--recon-loss", choices=["mse", "l1", "huber"], default="mse"
    )
    parser.add_argument("--num-suites", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predecoded = args.predecoded is not None
    loader, has_suite_ids = load_data(
        args.predecoded or args.cached_features,
        args.batch_size,
        predecoded,
    )
    config = ActionAEConfig(
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        num_tokens=args.num_tokens,
        latent_dim=args.latent_dim,
        encoder_hidden=args.encoder_hidden,
        decoder_hidden=args.decoder_hidden,
        video_cond_dim=0 if predecoded else args.video_cond_dim,
        variational=args.variational,
        beta_kl=args.beta_kl,
        free_bits=args.free_bits,
        recon_loss=args.recon_loss,
        num_suites=args.num_suites,
    )
    model = ActionAE(config).to(args.device)
    ema_model = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_mse = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        beta = (
            min(1.0, epoch / args.kl_warmup_epochs) * args.beta_kl
            if args.variational and args.kl_warmup_epochs > 0
            else args.beta_kl if args.variational else 0.0
        )
        for batch in loader:
            actions = batch[0].to(args.device)
            video_cond = None if predecoded else batch[1].to(args.device)
            action_is_pad = batch[1].to(args.device) if predecoded else None
            suite_ids = (
                batch[2].to(args.device)
                if predecoded and has_suite_ids
                else None
            )
            losses = model.compute_loss(
                actions,
                video_cond,
                beta_kl=beta,
                action_is_pad=action_is_pad,
                suite_ids=suite_ids,
            )
            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(ema_model, model, args.ema_decay)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(
                ema_model, loader, args.device, predecoded, has_suite_ids
            )
            metrics["epoch"] = epoch
            history.append(metrics)
            logger.info("Epoch %d: mse=%.8f", epoch, metrics["mse"])
            if metrics["mse"] < best_mse:
                best_mse = metrics["mse"]
                torch.save(
                    {
                        "config": config.__dict__,
                        "model": ema_model.state_dict(),
                        "epoch": epoch,
                        "best_mse": best_mse,
                    },
                    output,
                )
        if epoch % args.save_every == 0:
            torch.save(
                {"config": config.__dict__, "model": ema_model.state_dict()},
                output.with_name(f"{output.stem}_ep{epoch}{output.suffix}"),
            )

    output.with_name(f"{output.stem}_metrics.json").write_text(
        json.dumps({"best_mse": best_mse, "history": history}, indent=2)
    )


if __name__ == "__main__":
    main()
