# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic or variational temporal action autoencoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActionAEConfig:
    action_dim: int = 7
    action_horizon: int = 32
    num_tokens: int = 4
    latent_dim: int = 8
    encoder_hidden: int = 128
    decoder_hidden: int = 128
    video_cond_dim: int = 12288
    variational: bool = False
    beta_kl: float = 1e-2
    free_bits: float = 0.0
    recon_loss: str = "mse"
    num_suites: int = 0
    sigreg_lambda: float = 0.0


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


class ActionEncoder(nn.Module):
    def __init__(self, cfg: ActionAEConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.action_horizon % cfg.num_tokens:
            raise ValueError("action_horizon must be divisible by num_tokens")
        self.input_proj = nn.Linear(cfg.action_dim, cfg.encoder_hidden)
        self.cond_proj = (
            nn.Sequential(
                nn.Linear(cfg.video_cond_dim, cfg.encoder_hidden),
                nn.SiLU(),
                nn.Linear(cfg.encoder_hidden, cfg.encoder_hidden),
            )
            if cfg.video_cond_dim > 0
            else None
        )
        ratio = cfg.action_horizon // cfg.num_tokens
        self.strides = self._factorise_ratio(ratio)
        layers: list[nn.Module] = []
        for stride in self.strides:
            layers.extend(
                [
                    nn.Conv1d(
                        cfg.encoder_hidden,
                        cfg.encoder_hidden,
                        kernel_size=stride * 2 - 1,
                        stride=stride,
                        padding=stride - 1,
                    ),
                    nn.GroupNorm(min(8, cfg.encoder_hidden), cfg.encoder_hidden),
                    nn.SiLU(),
                    ResidualConv1dBlock(cfg.encoder_hidden),
                ]
            )
        self.down = nn.Sequential(*layers)
        self.suite_embed: Optional[nn.Embedding] = None
        if cfg.num_suites > 0:
            self.suite_embed = nn.Embedding(cfg.num_suites, cfg.encoder_hidden)
            nn.init.zeros_(self.suite_embed.weight)
        output_dim = cfg.latent_dim * (2 if cfg.variational else 1)
        self.bottleneck = nn.Linear(cfg.encoder_hidden, output_dim)

    @staticmethod
    def _factorise_ratio(ratio: int) -> list[int]:
        if ratio == 1:
            return [1]
        factors = []
        for factor in (4, 2, 3):
            while ratio % factor == 0:
                factors.append(factor)
                ratio //= factor
        if ratio > 1:
            factors.append(ratio)
        return factors

    def forward(
        self,
        actions: torch.Tensor,
        video_cond: Optional[torch.Tensor] = None,
        suite_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        x = self.input_proj(actions)
        if self.cond_proj is not None and video_cond is not None:
            x = x + self.cond_proj(video_cond).unsqueeze(1)
        if self.suite_embed is not None and suite_ids is not None:
            x = x + self.suite_embed(suite_ids).unsqueeze(1)
        x = self.down(x.transpose(1, 2)).transpose(1, 2)
        output = self.bottleneck(x)
        if not self.cfg.variational:
            return {"z": output}
        mu, log_var = output.chunk(2, dim=-1)
        z = mu + (0.5 * log_var).exp() * torch.randn_like(mu)
        return {"z": z, "mu": mu, "log_var": log_var}


class ActionDecoder(nn.Module):
    def __init__(self, cfg: ActionAEConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.latent_dim, cfg.decoder_hidden)
        strides = ActionEncoder._factorise_ratio(
            cfg.action_horizon // cfg.num_tokens
        )
        layers: list[nn.Module] = []
        for stride in strides:
            layers.extend(
                [
                    nn.ConvTranspose1d(
                        cfg.decoder_hidden,
                        cfg.decoder_hidden,
                        kernel_size=stride * 2,
                        stride=stride,
                        padding=stride // 2,
                    ),
                    nn.GroupNorm(min(8, cfg.decoder_hidden), cfg.decoder_hidden),
                    nn.SiLU(),
                    ResidualConv1dBlock(cfg.decoder_hidden),
                ]
            )
        self.up = nn.Sequential(*layers)
        self.output_proj = nn.Linear(cfg.decoder_hidden, cfg.action_dim)
        self.suite_embed: Optional[nn.Embedding] = None
        if cfg.num_suites > 0:
            self.suite_embed = nn.Embedding(cfg.num_suites, cfg.decoder_hidden)
            nn.init.zeros_(self.suite_embed.weight)

    def forward(
        self, z: torch.Tensor, suite_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.input_proj(z)
        if self.suite_embed is not None and suite_ids is not None:
            x = x + self.suite_embed(suite_ids).unsqueeze(1)
        x = self.up(x.transpose(1, 2))
        target = self.cfg.action_horizon
        if x.shape[2] > target:
            x = x[:, :, :target]
        elif x.shape[2] < target:
            x = F.pad(x, (0, target - x.shape[2]))
        return self.output_proj(x.transpose(1, 2))


class ActionAE(nn.Module):
    def __init__(self, cfg: ActionAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ActionEncoder(cfg)
        self.decoder = ActionDecoder(cfg)

    def encode(
        self,
        actions: torch.Tensor,
        video_cond: Optional[torch.Tensor] = None,
        suite_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        return self.encoder(actions, video_cond, suite_ids=suite_ids)

    def decode(
        self, z: torch.Tensor, suite_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.decoder(z, suite_ids=suite_ids)

    def forward(
        self,
        actions: torch.Tensor,
        video_cond: Optional[torch.Tensor] = None,
        suite_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode(actions, video_cond, suite_ids=suite_ids)
        output = {
            "recon": self.decode(encoded["z"], suite_ids=suite_ids),
            "z": encoded["z"],
        }
        if "mu" in encoded:
            output.update(mu=encoded["mu"], log_var=encoded["log_var"])
        return output

    def compute_loss(
        self,
        actions: torch.Tensor,
        video_cond: Optional[torch.Tensor] = None,
        beta_kl: float = 0.0,
        action_is_pad: Optional[torch.Tensor] = None,
        suite_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        output = self.forward(actions, video_cond, suite_ids=suite_ids)
        recon = output["recon"]
        if self.cfg.recon_loss == "l1":
            error = (recon - actions).abs().mean(dim=-1)
        elif self.cfg.recon_loss == "huber":
            error = F.smooth_l1_loss(
                recon, actions, reduction="none"
            ).mean(dim=-1)
        else:
            error = F.mse_loss(recon, actions, reduction="none").mean(dim=-1)
        if action_is_pad is not None:
            valid = (~action_is_pad).float().to(error.device)
            error = (error * valid).sum(1) / valid.sum(1).clamp(min=1)
        else:
            error = error.mean(1)
        loss_recon = error.mean()
        velocity_true = actions[:, 1:] - actions[:, :-1]
        velocity_pred = recon[:, 1:] - recon[:, :-1]
        loss_vel = F.mse_loss(velocity_pred, velocity_true)
        loss = loss_recon + 0.1 * loss_vel
        loss_kl = torch.tensor(0.0, device=actions.device)
        if self.cfg.variational and beta_kl > 0 and "mu" in output:
            mu, log_var = output["mu"], output["log_var"]
            kl_per_dim = -0.5 * (
                1 + log_var - mu.pow(2) - log_var.exp()
            )
            if self.cfg.free_bits > 0:
                kl_per_dim = torch.clamp(
                    kl_per_dim, min=self.cfg.free_bits
                )
            loss_kl = kl_per_dim.mean()
            loss = loss + beta_kl * loss_kl
        return {
            "loss": loss,
            "loss_recon": loss_recon.detach(),
            "loss_vel": loss_vel.detach(),
            "loss_kl": loss_kl.detach(),
        }
