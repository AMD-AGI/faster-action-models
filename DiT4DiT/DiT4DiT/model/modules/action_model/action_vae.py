# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporal variational autoencoder for normalized robot action chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActionVAEConfig:
    action_dim: int = 29
    action_horizon: int = 16
    num_tokens: int = 4
    latent_dim: int = 64
    encoder_hidden: int = 128
    decoder_hidden: int = 128
    variational: bool = True
    beta_kl: float = 1e-4
    free_bits: float = 0.1
    recon_loss: str = "mse"
    velocity_weight: float = 0.1
    num_tasks: int = 0
    conditioning_mode: Optional[
        Literal["none", "encoder_only", "encoder_decoder"]
    ] = None

    def __post_init__(self) -> None:
        # Missing conditioning_mode is the legacy checkpoint/config behavior.
        if self.conditioning_mode is None:
            self.conditioning_mode = (
                "encoder_decoder" if self.num_tasks > 0 else "none"
            )
        if self.conditioning_mode not in {
            "none",
            "encoder_only",
            "encoder_decoder",
        }:
            raise ValueError(
                "conditioning_mode must be one of none, encoder_only, "
                f"encoder_decoder; got {self.conditioning_mode!r}"
            )
        if self.conditioning_mode != "none" and self.num_tasks <= 0:
            raise ValueError(
                f"conditioning_mode={self.conditioning_mode!r} requires num_tasks > 0"
            )

    def to_dict(self) -> dict:
        return asdict(self)


class ResidualConv1dBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(min(8, channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _factorise_ratio(ratio: int) -> list[int]:
    if ratio == 1:
        return [1]
    factors: list[int] = []
    for factor in (4, 2, 3):
        while ratio % factor == 0:
            factors.append(factor)
            ratio //= factor
    if ratio > 1:
        factors.append(ratio)
    return factors


class ActionEncoder(nn.Module):
    def __init__(self, config: ActionVAEConfig):
        super().__init__()
        if config.action_horizon % config.num_tokens:
            raise ValueError(
                "action_horizon must be divisible by num_tokens, got "
                f"{config.action_horizon} and {config.num_tokens}"
            )
        self.config = config
        self.input_proj = nn.Linear(config.action_dim, config.encoder_hidden)
        self.task_embedding: Optional[nn.Embedding] = None
        if config.conditioning_mode in {"encoder_only", "encoder_decoder"}:
            self.task_embedding = nn.Embedding(config.num_tasks, config.encoder_hidden)
            nn.init.zeros_(self.task_embedding.weight)

        layers: list[nn.Module] = []
        for stride in _factorise_ratio(config.action_horizon // config.num_tokens):
            layers.extend(
                [
                    nn.Conv1d(
                        config.encoder_hidden,
                        config.encoder_hidden,
                        kernel_size=stride * 2 - 1,
                        stride=stride,
                        padding=stride - 1,
                    ),
                    nn.GroupNorm(min(8, config.encoder_hidden), config.encoder_hidden),
                    nn.SiLU(),
                    ResidualConv1dBlock(config.encoder_hidden),
                ]
            )
        self.down = nn.Sequential(*layers)
        output_dim = config.latent_dim * (2 if config.variational else 1)
        self.bottleneck = nn.Linear(config.encoder_hidden, output_dim)

    def forward(
        self, actions: torch.Tensor, task_ids: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        x = self.input_proj(actions)
        if self.task_embedding is not None:
            if task_ids is None:
                raise ValueError("task_ids are required for a task-conditioned ActionVAE")
            x = x + self.task_embedding(task_ids).unsqueeze(1)
        x = self.down(x.transpose(1, 2)).transpose(1, 2)
        output = self.bottleneck(x)
        if not self.config.variational:
            return {"z": output}
        mu, log_var = output.chunk(2, dim=-1)
        log_var = log_var.clamp(-20.0, 10.0)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        return {"z": z, "mu": mu, "log_var": log_var}


class ActionDecoder(nn.Module):
    def __init__(self, config: ActionVAEConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.latent_dim, config.decoder_hidden)
        self.task_embedding: Optional[nn.Embedding] = None
        if config.conditioning_mode == "encoder_decoder":
            self.task_embedding = nn.Embedding(config.num_tasks, config.decoder_hidden)
            nn.init.zeros_(self.task_embedding.weight)

        layers: list[nn.Module] = []
        for stride in _factorise_ratio(config.action_horizon // config.num_tokens):
            layers.extend(
                [
                    nn.ConvTranspose1d(
                        config.decoder_hidden,
                        config.decoder_hidden,
                        kernel_size=stride * 2,
                        stride=stride,
                        padding=stride // 2,
                    ),
                    nn.GroupNorm(min(8, config.decoder_hidden), config.decoder_hidden),
                    nn.SiLU(),
                    ResidualConv1dBlock(config.decoder_hidden),
                ]
            )
        self.up = nn.Sequential(*layers)
        self.output_proj = nn.Linear(config.decoder_hidden, config.action_dim)

    def forward(
        self, latent: torch.Tensor, task_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.input_proj(latent)
        if self.task_embedding is not None:
            if task_ids is None:
                raise ValueError("task_ids are required for a task-conditioned ActionVAE")
            x = x + self.task_embedding(task_ids).unsqueeze(1)
        x = self.up(x.transpose(1, 2))
        target_horizon = self.config.action_horizon
        if x.shape[-1] > target_horizon:
            x = x[..., :target_horizon]
        elif x.shape[-1] < target_horizon:
            x = F.pad(x, (0, target_horizon - x.shape[-1]))
        return self.output_proj(x.transpose(1, 2))


class ActionVAE(nn.Module):
    def __init__(self, config: ActionVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = ActionEncoder(config)
        self.decoder = ActionDecoder(config)

    def encode(
        self, actions: torch.Tensor, task_ids: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        return self.encoder(actions, task_ids=task_ids)

    def decode(
        self, latent: torch.Tensor, task_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.decoder(latent, task_ids=task_ids)

    def forward(
        self, actions: torch.Tensor, task_ids: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode(actions, task_ids=task_ids)
        reconstruction = self.decode(encoded["z"], task_ids=task_ids)
        return {"reconstruction": reconstruction, **encoded}

    def compute_loss(
        self,
        actions: torch.Tensor,
        valid_steps: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
        beta_kl: Optional[float] = None,
        return_latent: bool = False,
    ) -> dict[str, torch.Tensor]:
        output = self(actions, task_ids=task_ids)
        reconstruction = output["reconstruction"]
        if self.config.recon_loss == "l1":
            element_error = (reconstruction - actions).abs()
        elif self.config.recon_loss == "huber":
            element_error = F.smooth_l1_loss(
                reconstruction, actions, reduction="none"
            )
        else:
            element_error = F.mse_loss(reconstruction, actions, reduction="none")

        if valid_steps is not None:
            valid = valid_steps.to(element_error.dtype).unsqueeze(-1)
            reconstruction_loss = (element_error * valid).sum() / (
                valid.sum() * actions.shape[-1]
            ).clamp_min(1)
        else:
            reconstruction_loss = element_error.mean()

        velocity_error = (
            (reconstruction[:, 1:] - reconstruction[:, :-1])
            - (actions[:, 1:] - actions[:, :-1])
        ).pow(2)
        if valid_steps is not None:
            valid_velocity = (valid_steps[:, 1:] & valid_steps[:, :-1]).to(
                velocity_error.dtype
            ).unsqueeze(-1)
            velocity_loss = (velocity_error * valid_velocity).sum() / (
                valid_velocity.sum() * actions.shape[-1]
            ).clamp_min(1)
        else:
            velocity_loss = velocity_error.mean()

        kl_loss = actions.new_zeros(())
        effective_beta = self.config.beta_kl if beta_kl is None else beta_kl
        if self.config.variational and effective_beta > 0:
            kl_per_dim = -0.5 * (
                1 + output["log_var"] - output["mu"].pow(2) - output["log_var"].exp()
            )
            if self.config.free_bits > 0:
                kl_per_dim = kl_per_dim.clamp_min(self.config.free_bits)
            kl_loss = kl_per_dim.mean()

        total = (
            reconstruction_loss
            + self.config.velocity_weight * velocity_loss
            + effective_beta * kl_loss
        )
        losses = {
            "loss": total,
            "loss_reconstruction": reconstruction_loss.detach(),
            "loss_velocity": velocity_loss.detach(),
            "loss_kl": kl_loss.detach(),
        }
        if return_latent:
            losses["latent"] = output["mu"] if "mu" in output else output["z"]
        return losses


def load_action_vae(
    checkpoint_path: str, map_location: str | torch.device = "cpu"
) -> tuple[ActionVAE, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    config = ActionVAEConfig(**checkpoint["config"])
    model = ActionVAE(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint
