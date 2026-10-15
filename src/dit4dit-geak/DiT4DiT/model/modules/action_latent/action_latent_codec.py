# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class ActionStats:
    """Min-max normalization statistics for mapping to [-1, 1]."""
    min: torch.Tensor  # (dim,)
    max: torch.Tensor  # (dim,)


class ActionLatentCodec(nn.Module):
    """Encodes/decodes robot actions and state to/from Cosmos latent frame format.

    Follows the Cosmos-policy latent frame injection approach:
    normalize to [-1,1] → flatten → duplicate to fill latent volume → reshape.
    """

    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 8,
        state_dim: int = 9,
        latent_channels: int = 16,
        latent_h: int = 28,
        latent_w: int = 28,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.state_dim = state_dim
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w

        self.latent_volume = latent_channels * latent_h * latent_w  # 12544
        self.action_vector_len = action_horizon * action_dim         # 56
        self.state_vector_len = state_dim                            # 9

        self.action_repeat = self.latent_volume // self.action_vector_len  # 224
        self.action_remainder = self.latent_volume % self.action_vector_len
        self.state_repeat = self.latent_volume // self.state_vector_len
        self.state_remainder = self.latent_volume % self.state_vector_len

    def _normalize(self, x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """Min-max normalize to [-1, 1]."""
        span = (hi - lo).clamp(min=1e-6)
        return 2.0 * (x - lo) / span - 1.0

    def _unnormalize(self, x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """Inverse min-max: [-1, 1] → [lo, hi]."""
        span = hi - lo
        return (x + 1.0) * 0.5 * span + lo

    def _vec_to_latent_frame(self, vec: torch.Tensor, repeat_factor: int, remainder: int) -> torch.Tensor:
        """(B, D) → (B, C, 1, H, W) via repeat+reshape."""
        B = vec.shape[0]
        repeated = vec.repeat(1, repeat_factor)  # (B, D * repeat_factor)
        if remainder > 0:
            repeated = torch.cat([repeated, vec[:, :remainder]], dim=1)
        latent = repeated[:, :self.latent_volume]
        return latent.reshape(B, self.latent_channels, 1, self.latent_h, self.latent_w)

    def _latent_frame_to_vec(self, latent: torch.Tensor, vec_len: int, repeat_factor: int) -> torch.Tensor:
        """(B, C, 1, H, W) → (B, D) via reshape+average over repeats."""
        B = latent.shape[0]
        flat = latent.reshape(B, -1)  # (B, latent_volume)
        usable = flat[:, :repeat_factor * vec_len]
        chunks = usable.reshape(B, repeat_factor, vec_len)
        return chunks.mean(dim=1)  # (B, vec_len)

    def encode_actions(
        self,
        actions: torch.Tensor,
        action_stats: Optional[ActionStats] = None,
    ) -> torch.Tensor:
        """Encode action chunk to a single latent frame.

        Args:
            actions: (B, action_horizon, action_dim) raw actions
            action_stats: normalization statistics, or None to skip normalization

        Returns:
            (B, 16, 1, 28, 28) latent frame
        """
        B = actions.shape[0]
        if action_stats is not None:
            lo = action_stats.min.to(actions.device)
            hi = action_stats.max.to(actions.device)
            actions = self._normalize(actions, lo, hi)
        flat = actions.reshape(B, self.action_vector_len)  # (B, 56)
        return self._vec_to_latent_frame(flat, self.action_repeat, self.action_remainder)

    def decode_actions(
        self,
        action_latent: torch.Tensor,
        action_stats: Optional[ActionStats] = None,
    ) -> torch.Tensor:
        """Decode action latent frame back to action chunk.

        Args:
            action_latent: (B, 16, 1, 28, 28) denoised latent frame
            action_stats: normalization statistics, or None to skip un-normalization

        Returns:
            (B, action_horizon, action_dim) actions
        """
        flat = self._latent_frame_to_vec(action_latent, self.action_vector_len, self.action_repeat)
        actions = flat.reshape(-1, self.action_horizon, self.action_dim)
        if action_stats is not None:
            lo = action_stats.min.to(actions.device)
            hi = action_stats.max.to(actions.device)
            actions = self._unnormalize(actions, lo, hi)
        return actions

    def encode_state(
        self,
        state: torch.Tensor,
        state_stats: Optional[ActionStats] = None,
    ) -> torch.Tensor:
        """Encode proprioceptive state to a single latent frame.

        Args:
            state: (B, state_dim) or (B, 1, state_dim)
            state_stats: normalization statistics

        Returns:
            (B, 16, 1, 28, 28)
        """
        if state.ndim == 3:
            state = state.squeeze(1)
        B = state.shape[0]
        if state_stats is not None:
            lo = state_stats.min.to(state.device)
            hi = state_stats.max.to(state.device)
            state = self._normalize(state, lo, hi)
        return self._vec_to_latent_frame(state, self.state_repeat, self.state_remainder)

    def decode_state(
        self,
        state_latent: torch.Tensor,
        state_stats: Optional[ActionStats] = None,
    ) -> torch.Tensor:
        """(B, 16, 1, 28, 28) → (B, state_dim)"""
        state = self._latent_frame_to_vec(state_latent, self.state_vector_len, self.state_repeat)
        if state_stats is not None:
            lo = state_stats.min.to(state.device)
            hi = state_stats.max.to(state.device)
            state = self._unnormalize(state, lo, hi)
        return state
