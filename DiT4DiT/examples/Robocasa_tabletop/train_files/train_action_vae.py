#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train and diagnose a temporal action VAE on a balanced PnP cache."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from DiT4DiT.model.modules.action_model.action_vae import (  # noqa: E402
    ActionVAE,
    ActionVAEConfig,
)
from examples.Robocasa_tabletop.train_files.prepare_pnp3_action_cache import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    validate_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT / "playground/action_vae/pnp3_action_chunks.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-tokens", type=int, required=True)
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--encoder-hidden", type=int, default=128)
    parser.add_argument("--decoder-hidden", type=int, default=128)
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument(
        "--conditioning-mode",
        choices=("none", "encoder_only", "encoder_decoder"),
        default=None,
        help=(
            "Family conditioning location. Omitted preserves legacy behavior: "
            "encoder_decoder when num_tasks > 0, otherwise none."
        ),
    )
    parser.add_argument(
        "--language-embeddings",
        type=Path,
        default=None,
        help=(
            "Optional sidecar produced by precompute_action_vae_instruction_embeddings.py. "
            "If omitted, attached cache embeddings are used when present."
        ),
    )
    parser.add_argument("--alignment-weight", type=float, default=0.0)
    parser.add_argument("--alignment-dim", type=int, default=256)
    parser.add_argument("--alignment-temperature", type=float, default=0.07)
    parser.add_argument(
        "--shuffle-language-seed",
        type=int,
        default=None,
        help=(
            "Train against a deterministic derangement of language rows. "
            "Original instruction IDs remain the multi-positive grouping labels."
        ),
    )
    parser.add_argument("--perturbation-std", type=float, default=0.01)
    parser.add_argument("--beta-kl", type=float, default=1e-4)
    parser.add_argument("--free-bits", type=float, default=0.1)
    parser.add_argument("--kl-warmup-epochs", type=int, default=5)
    parser.add_argument("--velocity-weight", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def deterministic_derangement(size: int, seed: int) -> torch.Tensor:
    """Return a seeded, fixed-point-free permutation using Sattolo's algorithm."""
    if size < 2:
        raise ValueError("A language derangement requires at least two instructions")
    generator = np.random.default_rng(seed)
    permutation = np.arange(size, dtype=np.int64)
    for index in range(size - 1, 0, -1):
        swap_index = int(generator.integers(0, index))
        permutation[index], permutation[swap_index] = (
            permutation[swap_index],
            permutation[index],
        )
    if np.any(permutation == np.arange(size)):
        raise RuntimeError("Internal error: generated language permutation has a fixed point")
    return torch.from_numpy(permutation)


def language_rows_for_instruction_ids(
    instruction_ids: torch.Tensor,
    permutation: Optional[torch.Tensor],
) -> torch.Tensor:
    """Map semantic instruction IDs to trained language-table rows."""
    if permutation is None:
        return instruction_ids
    return permutation.to(instruction_ids.device)[instruction_ids]


def build_language_alignment_metadata(
    instruction_count: int,
    instruction_texts: Optional[list[str]],
    shuffle_seed: Optional[int],
    permutation: Optional[torch.Tensor],
    cache_fingerprint: Optional[str],
) -> dict:
    """Describe the semantic labels and exact language rows used for training."""
    if permutation is None:
        permutation_list = list(range(instruction_count))
        mapping = "semantic"
    else:
        permutation_list = permutation.tolist()
        mapping = "shuffled_derangement"
    text_sha256 = (
        hashlib.sha256("\n".join(instruction_texts).encode()).hexdigest()
        if instruction_texts is not None
        else None
    )
    return {
        "enabled": True,
        "mapping": mapping,
        "shuffle_seed": shuffle_seed,
        "permutation": permutation_list,
        "instruction_count": instruction_count,
        "instruction_text_sha256": text_sha256,
        "cache_fingerprint": cache_fingerprint,
        "multi_positive_grouping": "original_instruction_ids",
        "semantic_mapping": "language_embeddings[instruction_ids]",
        "trained_mapping": (
            "language_embeddings[permutation[instruction_ids]]"
            if permutation is not None
            else "language_embeddings[instruction_ids]"
        ),
        "fixed_point_count": sum(
            index == row for index, row in enumerate(permutation_list)
        ),
    }


class ActionLanguageAlignmentHead(nn.Module):
    """Training-only projections for CLIP-style action/language alignment."""

    def __init__(
        self,
        action_feature_dim: int,
        language_feature_dim: int,
        projection_dim: int,
    ):
        super().__init__()
        self.action_projection = nn.Sequential(
            nn.LayerNorm(action_feature_dim),
            nn.Linear(action_feature_dim, projection_dim),
        )
        self.language_projection = nn.Sequential(
            nn.LayerNorm(language_feature_dim),
            nn.Linear(language_feature_dim, projection_dim),
        )

    def forward(
        self,
        latent: torch.Tensor,
        language_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_features = self.action_projection(latent.flatten(1))
        text_features = self.language_projection(language_features)
        return F.normalize(action_features, dim=-1), F.normalize(
            text_features, dim=-1
        )


def multi_positive_alignment_loss(
    action_features: torch.Tensor,
    text_features: torch.Tensor,
    instruction_ids: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE where every equal-instruction pair is positive."""
    if action_features.shape != text_features.shape:
        raise ValueError(
            "action_features and text_features must have identical shapes, got "
            f"{tuple(action_features.shape)} and {tuple(text_features.shape)}"
        )
    if instruction_ids.ndim != 1 or instruction_ids.shape[0] != action_features.shape[0]:
        raise ValueError("instruction_ids must have shape [batch]")
    if temperature <= 0:
        raise ValueError("alignment temperature must be positive")
    logits = action_features @ text_features.transpose(0, 1) / temperature
    positives = instruction_ids[:, None] == instruction_ids[None, :]
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    action_to_text = torch.logsumexp(logits, dim=1) - torch.logsumexp(
        positive_logits, dim=1
    )
    text_to_action = torch.logsumexp(logits, dim=0) - torch.logsumexp(
        positive_logits, dim=0
    )
    return 0.5 * (action_to_text.mean() + text_to_action.mean())


def _optional_split_tensor(
    split: dict[str, torch.Tensor],
    names: tuple[str, ...],
    default: torch.Tensor,
) -> torch.Tensor:
    for name in names:
        if name in split:
            return split[name]
    return default


def make_loader(
    split: dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    count = split["actions"].shape[0]
    dataset = TensorDataset(
        split["actions"],
        split["valid_steps"],
        split["task_ids"],
        _optional_split_tensor(
            split,
            ("instruction_ids",),
            torch.full((count,), -1, dtype=torch.int64),
        ),
        _optional_split_tensor(
            split,
            ("episode_ids",),
            torch.full((count,), -1, dtype=torch.int64),
        ),
        _optional_split_tensor(
            split,
            ("start_steps", "chunk_start_steps", "step_ids"),
            torch.full((count,), -1, dtype=torch.int64),
        ),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
    )


def balanced_diagnostic_split(
    split: dict[str, torch.Tensor],
    samples_per_task: int = 1_000,
) -> dict[str, torch.Tensor]:
    """Select a deterministic, balanced subset for expensive diagnostics."""
    task_ids = split["task_ids"]
    indices = torch.cat(
        [
            (task_ids == task_id).nonzero(as_tuple=False).flatten()[
                :samples_per_task
            ]
            for task_id in torch.unique(task_ids, sorted=True).tolist()
        ]
    )
    return {key: value[indices] for key, value in split.items()}


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter, 1.0 - decay)


@torch.no_grad()
def validation_mse(
    model: ActionVAE, loader: DataLoader, device: torch.device, num_tasks: int
) -> float:
    model.eval()
    squared_error = 0.0
    valid_elements = 0
    for actions, valid_steps, task_ids, _, _, _ in loader:
        actions = actions.to(device=device, dtype=torch.float32, non_blocking=True)
        valid_steps = valid_steps.to(device=device, non_blocking=True)
        task_ids = task_ids.to(device=device, non_blocking=True) if num_tasks else None
        encoded = model.encode(actions, task_ids=task_ids)
        reconstruction = model.decode(
            encoded["mu"] if "mu" in encoded else encoded["z"],
            task_ids=task_ids,
        )
        valid = valid_steps.unsqueeze(-1)
        squared_error += ((reconstruction - actions).pow(2) * valid).sum().item()
        valid_elements += int(valid.sum().item()) * actions.shape[-1]
    return squared_error / max(valid_elements, 1)


def _masked_numpy(tensor: torch.Tensor, valid_steps: torch.Tensor) -> np.ndarray:
    return tensor[valid_steps.unsqueeze(-1).expand_as(tensor)].reshape(
        -1, tensor.shape[-1]
    ).numpy()


def _bounded_rows(values: np.ndarray, maximum_rows: int = 20_000) -> np.ndarray:
    """Deterministically bound arrays used for expensive percentile diagnostics."""
    if values.shape[0] <= maximum_rows:
        return values
    indices = np.linspace(
        0, values.shape[0] - 1, num=maximum_rows, dtype=np.int64
    )
    return values[indices]


@torch.no_grad()
def detailed_diagnostics(
    model: ActionVAE,
    loader: DataLoader,
    device: torch.device,
    num_tasks: int,
    task_names: list[str],
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    instruction_texts: Optional[list[str]] = None,
    alignment_head: Optional[ActionLanguageAlignmentHead] = None,
    language_embeddings: Optional[torch.Tensor] = None,
    language_permutation: Optional[torch.Tensor] = None,
    perturbation_std: float = 0.01,
) -> tuple[dict, torch.Tensor, torch.Tensor, bool]:
    model.eval()
    if alignment_head is not None:
        alignment_head.eval()
    normalized_errors: list[torch.Tensor] = []
    physical_errors: list[torch.Tensor] = []
    velocity_errors: list[torch.Tensor] = []
    physical_velocity_errors: list[torch.Tensor] = []
    latents: list[torch.Tensor] = []
    sample_mses: list[torch.Tensor] = []
    all_task_ids: list[torch.Tensor] = []
    all_instruction_ids: list[torch.Tensor] = []
    all_episode_ids: list[torch.Tensor] = []
    all_start_steps: list[torch.Tensor] = []
    perturbation_latent_rms: list[float] = []
    perturbation_gain: list[float] = []
    shifted_chunk_latent_rms: list[float] = []
    adjacent_token_latent_rms: list[float] = []
    aligned_action_features: list[torch.Tensor] = []
    semantic_text_features: list[torch.Tensor] = []
    trained_text_features: list[torch.Tensor] = []
    aligned_instruction_ids: list[torch.Tensor] = []
    alignment_diagnostic_count = 0
    task_normalized_errors: list[list[np.ndarray]] = [
        [] for _ in task_names
    ]
    task_physical_errors: list[list[np.ndarray]] = [[] for _ in task_names]
    task_velocity_errors: list[list[np.ndarray]] = [[] for _ in task_names]
    per_task_squared = torch.zeros(len(task_names), dtype=torch.float64)
    per_task_count = torch.zeros(len(task_names), dtype=torch.float64)
    scale = (action_high.float() - action_low.float()) / 2.0

    for batch_index, (
        actions,
        valid_steps,
        task_ids_cpu,
        instruction_ids_cpu,
        episode_ids,
        start_steps,
    ) in enumerate(loader, start=1):
        actions = actions.to(device=device, dtype=torch.float32, non_blocking=True)
        valid_steps_device = valid_steps.to(device=device, non_blocking=True)
        task_ids = (
            task_ids_cpu.to(device=device, non_blocking=True) if num_tasks else None
        )
        encoded = model.encode(actions, task_ids=task_ids)
        latent = encoded["mu"] if "mu" in encoded else encoded["z"]
        reconstruction = model.decode(latent, task_ids=task_ids)
        error = (reconstruction - actions).cpu()
        batch_scale = (
            scale.view(1, 1, -1)
            if scale.ndim == 1
            else scale[task_ids_cpu].unsqueeze(1)
        )
        normalized_errors.append(_masked_numpy(error.abs(), valid_steps))
        physical_errors.append(
            _masked_numpy(error.abs() * batch_scale, valid_steps)
        )
        valid_velocity = valid_steps[:, 1:] & valid_steps[:, :-1]
        velocity_error = (
            (reconstruction[:, 1:] - reconstruction[:, :-1])
            - (actions[:, 1:] - actions[:, :-1])
        ).abs().cpu()
        velocity_errors.append(_masked_numpy(velocity_error, valid_velocity))
        physical_velocity_errors.append(
            _masked_numpy(
                velocity_error * batch_scale[:, :1],
                valid_velocity,
            )
        )
        latents.append(latent.cpu())
        all_task_ids.append(task_ids_cpu)
        all_instruction_ids.append(instruction_ids_cpu)
        all_episode_ids.append(episode_ids)
        all_start_steps.append(start_steps)

        sample_mse = (
            error.pow(2) * valid_steps.unsqueeze(-1)
        ).sum(dim=(1, 2)) / (
            valid_steps.sum(dim=1).clamp_min(1) * error.shape[-1]
        )
        sample_mses.append(sample_mse)
        if latent.shape[1] > 1:
            adjacent_token_latent_rms.append(
                float((latent[:, 1:] - latent[:, :-1]).pow(2).mean().sqrt())
            )
        shifted_actions = torch.cat((actions[:, 1:], actions[:, -1:]), dim=1)
        shifted_encoded = model.encode(shifted_actions, task_ids=task_ids)
        shifted_latent = (
            shifted_encoded["mu"]
            if "mu" in shifted_encoded
            else shifted_encoded["z"]
        )
        shifted_chunk_latent_rms.append(
            float((shifted_latent - latent).pow(2).mean().sqrt())
        )
        if perturbation_std > 0:
            noise = (
                torch.randn_like(actions)
                * perturbation_std
                * valid_steps_device.unsqueeze(-1)
            )
            perturbed = (actions + noise).clamp(-1.0, 1.0)
            perturbed_encoded = model.encode(perturbed, task_ids=task_ids)
            perturbed_latent = (
                perturbed_encoded["mu"]
                if "mu" in perturbed_encoded
                else perturbed_encoded["z"]
            )
            latent_rms = (perturbed_latent - latent).pow(2).mean().sqrt()
            action_rms = (perturbed - actions).pow(2).mean().sqrt().clamp_min(1e-8)
            perturbation_latent_rms.append(float(latent_rms))
            perturbation_gain.append(float(latent_rms / action_rms))
        valid_instruction = instruction_ids_cpu >= 0
        if (
            alignment_head is not None
            and language_embeddings is not None
            and valid_instruction.any()
            and alignment_diagnostic_count < 4096
        ):
            selected_indices = valid_instruction.nonzero(as_tuple=False).flatten()
            selected_indices = selected_indices[
                : 4096 - alignment_diagnostic_count
            ]
            selected_indices_device = selected_indices.to(device)
            ids = instruction_ids_cpu[selected_indices].to(device)
            trained_rows = language_rows_for_instruction_ids(
                ids, language_permutation
            )
            action_features, trained_features = alignment_head(
                latent[selected_indices_device],
                language_embeddings[trained_rows],
            )
            _, semantic_features = alignment_head(
                latent[selected_indices_device],
                language_embeddings[ids],
            )
            aligned_action_features.append(action_features.cpu())
            semantic_text_features.append(semantic_features.cpu())
            trained_text_features.append(trained_features.cpu())
            aligned_instruction_ids.append(ids.cpu())
            alignment_diagnostic_count += int(ids.numel())
        for task_id in range(len(task_names)):
            selected = task_ids_cpu == task_id
            per_task_squared[task_id] += sample_mse[selected].double().sum()
            per_task_count[task_id] += selected.sum()
            if selected.any():
                task_normalized_errors[task_id].append(
                    _masked_numpy(error[selected].abs(), valid_steps[selected])
                )
                task_physical_errors[task_id].append(
                    _masked_numpy(
                        error[selected].abs() * batch_scale[selected],
                        valid_steps[selected],
                    )
                )
                task_velocity_errors[task_id].append(
                    _masked_numpy(
                        velocity_error[selected],
                        valid_velocity[selected],
                    )
                )
        if batch_index % 200 == 0:
            print(
                f"Diagnostics inference: {batch_index}/{len(loader)} batches",
                flush=True,
            )

    print("Diagnostics inference complete; aggregating metrics", flush=True)
    normalized = _bounded_rows(np.concatenate(normalized_errors))
    physical = _bounded_rows(np.concatenate(physical_errors))
    velocity = _bounded_rows(np.concatenate(velocity_errors))
    physical_velocity = _bounded_rows(np.concatenate(physical_velocity_errors))
    latent = torch.cat(latents).float()
    sample_mse_all = torch.cat(sample_mses).float()
    task_ids_all = torch.cat(all_task_ids)
    instruction_ids_all = torch.cat(all_instruction_ids)
    episode_ids_all = torch.cat(all_episode_ids)
    start_steps_all = torch.cat(all_start_steps)
    latent_mean = latent.mean(dim=0)
    latent_std = latent.std(dim=0).clamp_min(1e-4)
    flat_std = latent_std.flatten()
    variance_ratio = float(
        (flat_std.square().max() / flat_std.square().min()).item()
    )
    collapsed_fraction = float((flat_std < 0.1).float().mean().item())
    collapsed_dimensions = int((flat_std < 0.1).sum().item())
    latent_variances = flat_std.square()
    effective_dimensions = float(
        latent_variances.sum().square()
        / latent_variances.square().sum().clamp_min(1e-12)
    )

    sample_count = min(5000, latent.shape[0])
    sampled = latent[:sample_count].reshape(sample_count, -1)
    sorted_latent = sampled.sort(dim=0).values
    probabilities = (
        torch.arange(sample_count, dtype=torch.float32) + 0.5
    ) / sample_count
    normal_quantiles = math.sqrt(2.0) * torch.erfinv(
        (2.0 * probabilities - 1.0).clamp(-0.999999, 0.999999)
    )
    w1 = float(
        (sorted_latent - normal_quantiles.unsqueeze(1)).abs().mean().item()
    )
    apply_normalization = bool(
        (flat_std < 0.5).any()
        or (flat_std > 2.0).any()
        or variance_ratio > 10.0
    )

    def group_metrics(values: np.ndarray) -> dict[str, dict[str, float]]:
        groups = {"arms": slice(0, 14), "hands": slice(14, 26), "waist": slice(26, 29)}
        return {
            name: {
                "mean": float(values[:, indices].mean()),
                "p95": float(np.quantile(values[:, indices], 0.95)),
                "p99": float(np.quantile(values[:, indices], 0.99)),
                "max": float(values[:, indices].max()),
            }
            for name, indices in groups.items()
        }

    def abs_error_metrics(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(values.max()),
        }

    def reconstruction_groups(
        ids: torch.Tensor,
        names: Optional[list[str]] = None,
    ) -> dict:
        groups = {}
        for group_id in torch.unique(ids[ids >= 0]).tolist():
            selected = sample_mse_all[ids == group_id]
            label = (
                names[group_id]
                if names is not None and group_id < len(names)
                else str(group_id)
            )
            groups[label] = {
                "id": int(group_id),
                "count": int(selected.numel()),
                "normalized_mse_mean": float(selected.mean()),
                "normalized_mse_p95": float(torch.quantile(selected, 0.95)),
                "normalized_mse_max": float(selected.max()),
            }
        worst = max(
            groups,
            key=lambda name: groups[name]["normalized_mse_mean"],
            default=None,
        )
        return {"groups": groups, "worst_mean_group": worst}

    adjacent_deltas = []
    adjacent_rates = []
    if (start_steps_all >= 0).any():
        ordering = sorted(
            range(latent.shape[0]),
            key=lambda index: (
                int(task_ids_all[index]),
                int(episode_ids_all[index]),
                int(start_steps_all[index]),
            ),
        )
        flattened_latent = latent.flatten(1)
        for left, right in zip(ordering, ordering[1:]):
            same_episode = (
                task_ids_all[left] == task_ids_all[right]
                and episode_ids_all[left] == episode_ids_all[right]
            )
            step_delta = int(start_steps_all[right] - start_steps_all[left])
            if same_episode and step_delta > 0:
                latent_delta = float(
                    (flattened_latent[right] - flattened_latent[left])
                    .pow(2)
                    .mean()
                    .sqrt()
                )
                adjacent_deltas.append(latent_delta)
                adjacent_rates.append(latent_delta / step_delta)

    per_task_diagnostics = {}
    for task_id, task_name in enumerate(task_names):
        task_normalized = _bounded_rows(
            np.concatenate(task_normalized_errors[task_id])
        )
        task_physical = _bounded_rows(
            np.concatenate(task_physical_errors[task_id])
        )
        task_velocity = _bounded_rows(
            np.concatenate(task_velocity_errors[task_id])
        )
        per_task_diagnostics[task_name] = {
            "normalized_mse": float(np.square(task_normalized).mean()),
            "normalized_abs_error": abs_error_metrics(task_normalized),
            "normalized_abs_error_by_group": group_metrics(task_normalized),
            "physical_abs_error_by_group": group_metrics(task_physical),
            "velocity_abs_error_mean": float(task_velocity.mean()),
        }
        print(
            f"Diagnostics aggregation: {task_id + 1}/{len(task_names)} families",
            flush=True,
        )

    def prompt_metrics(embedding: torch.nn.Embedding) -> dict:
        weights = embedding.weight.detach().float().cpu()
        norms = weights.norm(dim=-1)
        distances = torch.pdist(weights)
        return {
            "per_task_norm": {
                task_name: float(norms[task_id])
                for task_id, task_name in enumerate(task_names)
            },
            "pairwise_distance": {
                "mean": float(distances.mean()),
                "min": float(distances.min()),
                "max": float(distances.max()),
            },
        }

    diagnostics = {
        "normalized_mse": float(np.square(normalized).mean()),
        "normalized_abs_error": {
            "mean": float(normalized.mean()),
            "p95": float(np.quantile(normalized, 0.95)),
            "p99": float(np.quantile(normalized, 0.99)),
            "max": float(normalized.max()),
        },
        "physical_abs_error": group_metrics(physical),
        "physical_abs_error_overall": abs_error_metrics(physical),
        "normalized_abs_error_by_group": group_metrics(normalized),
        "velocity_abs_error_mean": float(velocity.mean()),
        "hand_transition_abs_error": abs_error_metrics(velocity[:, 14:26]),
        "hand_transition_physical_abs_error": abs_error_metrics(
            physical_velocity[:, 14:26]
        ),
        "per_dof_normalized_mse": np.square(normalized).mean(axis=0).tolist(),
        "per_task_normalized_mse": (
            per_task_squared / per_task_count.clamp_min(1)
        ).tolist(),
        "per_task_normalized_mse_by_name": {
            task_name: float(per_task_squared[task_id] / per_task_count[task_id].clamp_min(1))
            for task_id, task_name in enumerate(task_names)
        },
        "per_task": per_task_diagnostics,
        "latent": {
            "mean_abs": float(latent_mean.abs().mean().item()),
            "std_mean": float(flat_std.mean().item()),
            "std_min": float(flat_std.min().item()),
            "std_max": float(flat_std.max().item()),
            "variance_ratio": variance_ratio,
            "collapsed_fraction": collapsed_fraction,
            "collapsed_dimensions": collapsed_dimensions,
            "total_dimensions": int(flat_std.numel()),
            "effective_dimensions": effective_dimensions,
            "w1_from_standard_normal": w1,
            "apply_affine_normalization": apply_normalization,
        },
    }
    diagnostics["family_reconstruction"] = reconstruction_groups(
        task_ids_all, task_names
    )
    diagnostics["instruction_reconstruction"] = reconstruction_groups(
        instruction_ids_all, instruction_texts
    )
    diagnostics["stability"] = {
        "perturbation_std": perturbation_std,
        "perturbation_latent_rms_mean": (
            float(np.mean(perturbation_latent_rms))
            if perturbation_latent_rms
            else None
        ),
        "perturbation_latent_to_action_gain_mean": (
            float(np.mean(perturbation_gain)) if perturbation_gain else None
        ),
        "perturbation_latent_to_action_gain_p95": (
            float(np.quantile(perturbation_gain, 0.95))
            if perturbation_gain
            else None
        ),
        "perturbation_latent_to_action_gain_p99": (
            float(np.quantile(perturbation_gain, 0.99))
            if perturbation_gain
            else None
        ),
        "one_step_shift_latent_rms_mean": float(
            np.mean(shifted_chunk_latent_rms)
        ),
        "within_chunk_adjacent_token_rms_mean": (
            float(np.mean(adjacent_token_latent_rms))
            if adjacent_token_latent_rms
            else None
        ),
        "adjacent_pair_count": len(adjacent_deltas),
        "adjacent_latent_rms_mean": (
            float(np.mean(adjacent_deltas)) if adjacent_deltas else None
        ),
        "adjacent_latent_rms_per_step_mean": (
            float(np.mean(adjacent_rates)) if adjacent_rates else None
        ),
        "adjacent_available": bool(adjacent_deltas),
    }
    if aligned_action_features:
        action_features = torch.cat(aligned_action_features)[:4096]
        semantic_features = torch.cat(semantic_text_features)[:4096]
        trained_features = torch.cat(trained_text_features)[:4096]
        aligned_ids = torch.cat(aligned_instruction_ids)[:4096]
        semantic_nearest = (
            action_features @ semantic_features.transpose(0, 1)
        ).argmax(dim=1)
        trained_nearest = (
            action_features @ trained_features.transpose(0, 1)
        ).argmax(dim=1)
        diagnostics["language_alignment"] = {
            "sample_count": int(action_features.shape[0]),
            "semantic_mapping": {
                "paired_cosine_mean": float(
                    (action_features * semantic_features).sum(dim=-1).mean()
                ),
                "multi_positive_recall_at_1": float(
                    (aligned_ids[semantic_nearest] == aligned_ids).float().mean()
                ),
            },
            "trained_mapping": {
                "kind": (
                    "shuffled"
                    if language_permutation is not None
                    else "semantic"
                ),
                "paired_cosine_mean": float(
                    (action_features * trained_features).sum(dim=-1).mean()
                ),
                "multi_positive_recall_at_1": float(
                    (aligned_ids[trained_nearest] == aligned_ids).float().mean()
                ),
            },
        }
        # Preserve legacy fields for consumers of unshuffled checkpoints.
        if language_permutation is None:
            diagnostics["language_alignment"].update(
                diagnostics["language_alignment"]["semantic_mapping"]
            )
    if model.encoder.task_embedding is not None or model.decoder.task_embedding is not None:
        diagnostics["task_prompts"] = {
            "num_tasks": model.config.num_tasks,
            "conditioning_mode": model.config.conditioning_mode,
            "encoder": (
                prompt_metrics(model.encoder.task_embedding)
                if model.encoder.task_embedding is not None
                else None
            ),
            "decoder": (
                prompt_metrics(model.decoder.task_embedding)
                if model.decoder.task_embedding is not None
                else None
            ),
        }
    return diagnostics, latent_mean, latent_std, apply_normalization


def _instruction_texts_from_mapping(mapping: dict) -> Optional[list[str]]:
    for key in ("instruction_texts", "instructions"):
        value = mapping.get(key)
        if isinstance(value, (list, tuple)) and all(
            isinstance(text, str) for text in value
        ):
            return list(value)
    registry = mapping.get("instruction_registry")
    if isinstance(registry, (list, tuple)):
        texts = []
        for entry in registry:
            if isinstance(entry, str):
                texts.append(entry)
            elif isinstance(entry, dict):
                text = entry.get("text") or entry.get("instruction")
                if not isinstance(text, str):
                    return None
                texts.append(text)
            else:
                return None
        return texts
    return None


def load_language_embeddings(
    cache: dict,
    sidecar_path: Optional[Path],
) -> tuple[Optional[torch.Tensor], Optional[list[str]], Optional[dict]]:
    payload = None
    if sidecar_path is not None:
        payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    else:
        for container_name, container in (
            ("attached-cache", cache),
            ("attached-cache-metadata", cache.get("metadata", {})),
        ):
            for key in (
                "instruction_embeddings",
                "language_embeddings",
                "frozen_language_embeddings",
            ):
                if key in container:
                    payload = {
                        "instruction_embeddings": container[key],
                        "instruction_texts": _instruction_texts_from_mapping(
                            cache.get("metadata", {})
                        ),
                        "source": container_name,
                    }
                    break
            if payload is not None:
                break
    if payload is None:
        return None, _instruction_texts_from_mapping(cache.get("metadata", {})), None
    if torch.is_tensor(payload):
        embeddings = payload
        payload_metadata = {"source": str(sidecar_path)}
        texts = _instruction_texts_from_mapping(cache.get("metadata", {}))
    elif isinstance(payload, dict):
        embeddings = payload.get(
            "instruction_embeddings",
            payload.get("language_embeddings", payload.get("embeddings")),
        )
        payload_metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"instruction_embeddings", "embeddings"}
        }
        texts = _instruction_texts_from_mapping(payload)
        if texts is None:
            texts = _instruction_texts_from_mapping(cache.get("metadata", {}))
    else:
        raise TypeError("Language embedding sidecar must contain a tensor or dictionary")
    if not torch.is_tensor(embeddings) or embeddings.ndim != 2:
        raise ValueError("instruction_embeddings must be a rank-2 tensor [instructions, dim]")
    embeddings = embeddings.detach().float().contiguous()
    if not torch.isfinite(embeddings).all():
        raise ValueError("instruction_embeddings contains non-finite values")
    if texts is not None and len(texts) != embeddings.shape[0]:
        raise ValueError(
            f"Found {len(texts)} instruction texts but {embeddings.shape[0]} embeddings"
        )
    cache_fingerprint = cache.get("metadata", {}).get("cache_fingerprint")
    if cache_fingerprint is not None and isinstance(payload, dict):
        sidecar_fingerprint = payload.get("cache_fingerprint")
        if sidecar_fingerprint != cache_fingerprint:
            raise ValueError(
                "Language embedding sidecar cache fingerprint mismatch: "
                f"{sidecar_fingerprint!r} != {cache_fingerprint!r}"
            )
        expected_text_hash = payload.get("text_sha256")
        actual_text_hash = hashlib.sha256("\n".join(texts or ()).encode()).hexdigest()
        if expected_text_hash != actual_text_hash:
            raise ValueError("Language embedding sidecar text fingerprint mismatch")
    return embeddings, texts, payload_metadata


def _family_names(metadata: dict, num_tasks: int) -> list[str]:
    for key in ("task_datasets", "family_names", "family_datasets"):
        names = metadata.get(key)
        if isinstance(names, (list, tuple)) and len(names) == num_tasks:
            return list(names)
    return [f"family_{index}" for index in range(num_tasks)]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("metadata", {}).get("cache_schema_version") is not None:
        if cache["metadata"]["cache_schema_version"] != CACHE_SCHEMA_VERSION:
            raise ValueError("Full24 cache schema does not match this trainer")
        validate_cache(cache)
    language_embeddings, instruction_texts, language_metadata = (
        load_language_embeddings(cache, args.language_embeddings)
    )
    if args.alignment_weight < 0:
        raise ValueError("alignment-weight must be non-negative")
    if args.shuffle_language_seed is not None and args.alignment_weight == 0:
        raise ValueError(
            "shuffle-language-seed requires a positive alignment-weight"
        )
    language_permutation = None
    alignment_metadata = None
    if args.alignment_weight > 0:
        if language_embeddings is None:
            raise ValueError(
                "alignment-weight > 0 requires --language-embeddings or attached "
                "cache instruction_embeddings"
            )
        for split_name in ("train", "val"):
            instruction_ids = cache[split_name].get("instruction_ids")
            if instruction_ids is None or (instruction_ids < 0).any():
                raise ValueError(
                    f"Alignment requires non-negative {split_name} instruction_ids"
                )
            if int(instruction_ids.max()) >= language_embeddings.shape[0]:
                raise ValueError(
                    f"{split_name} instruction_ids exceed the embedding table"
                )
        if args.shuffle_language_seed is not None:
            language_permutation = deterministic_derangement(
                language_embeddings.shape[0],
                args.shuffle_language_seed,
            )
        alignment_metadata = build_language_alignment_metadata(
            instruction_count=language_embeddings.shape[0],
            instruction_texts=instruction_texts,
            shuffle_seed=args.shuffle_language_seed,
            permutation=language_permutation,
            cache_fingerprint=cache.get("metadata", {}).get(
                "cache_fingerprint"
            ),
        )
    train_loader = make_loader(
        cache["train"], args.batch_size, True, args.num_workers
    )
    val_loader = make_loader(cache["val"], args.batch_size, False, args.num_workers)
    config = ActionVAEConfig(
        action_dim=cache["train"]["actions"].shape[-1],
        action_horizon=cache["train"]["actions"].shape[1],
        num_tokens=args.num_tokens,
        latent_dim=args.latent_dim,
        encoder_hidden=args.encoder_hidden,
        decoder_hidden=args.decoder_hidden,
        variational=True,
        beta_kl=args.beta_kl,
        free_bits=args.free_bits,
        velocity_weight=args.velocity_weight,
        num_tasks=args.num_tasks,
        conditioning_mode=args.conditioning_mode,
    )
    model = ActionVAE(config).to(device)
    ema_model = copy.deepcopy(model).eval()
    alignment_head = None
    ema_alignment_head = None
    language_embeddings_device = None
    language_permutation_device = None
    parameters = list(model.parameters())
    if args.alignment_weight > 0:
        language_embeddings_device = language_embeddings.to(device)
        language_permutation_device = (
            language_permutation.to(device)
            if language_permutation is not None
            else None
        )
        alignment_head = ActionLanguageAlignmentHead(
            config.num_tokens * config.latent_dim,
            language_embeddings.shape[1],
            args.alignment_dim,
        ).to(device)
        ema_alignment_head = copy.deepcopy(alignment_head).eval()
        parameters.extend(alignment_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    best_mse = float("inf")
    history: list[dict] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Training K={config.num_tokens}, dz={config.latent_dim}, "
        f"total latent={config.num_tokens * config.latent_dim}, "
        f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        if alignment_head is not None:
            alignment_head.train()
        totals = {
            "loss": 0.0,
            "reconstruction": 0.0,
            "velocity": 0.0,
            "kl": 0.0,
            "alignment": 0.0,
        }
        beta = min(1.0, epoch / max(args.kl_warmup_epochs, 1)) * args.beta_kl
        for actions, valid_steps, task_ids, instruction_ids, _, _ in train_loader:
            actions = actions.to(device=device, dtype=torch.float32, non_blocking=True)
            valid_steps = valid_steps.to(device=device, non_blocking=True)
            task_ids = task_ids.to(device=device, non_blocking=True) if args.num_tasks else None
            losses = model.compute_loss(
                actions,
                valid_steps=valid_steps,
                task_ids=task_ids,
                beta_kl=beta,
                return_latent=alignment_head is not None,
            )
            alignment_loss = actions.new_zeros(())
            if alignment_head is not None:
                instruction_ids = instruction_ids.to(device, non_blocking=True)
                language_rows = language_rows_for_instruction_ids(
                    instruction_ids,
                    language_permutation_device,
                )
                action_features, text_features = alignment_head(
                    losses["latent"],
                    language_embeddings_device[language_rows],
                )
                alignment_loss = multi_positive_alignment_loss(
                    action_features,
                    text_features,
                    instruction_ids,
                    args.alignment_temperature,
                )
            total_loss = losses["loss"] + args.alignment_weight * alignment_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            update_ema(ema_model, model, args.ema_decay)
            if alignment_head is not None:
                update_ema(ema_alignment_head, alignment_head, args.ema_decay)
            totals["loss"] += total_loss.item()
            totals["reconstruction"] += losses["loss_reconstruction"].item()
            totals["velocity"] += losses["loss_velocity"].item()
            totals["kl"] += losses["loss_kl"].item()
            totals["alignment"] += alignment_loss.item()

        batches = len(train_loader)
        row = {
            "epoch": epoch,
            "beta_kl": beta,
            **{key: value / batches for key, value in totals.items()},
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            row["val_mse"] = validation_mse(
                ema_model, val_loader, device, args.num_tasks
            )
            if row["val_mse"] < best_mse:
                best_mse = row["val_mse"]
                torch.save(
                    {
                        "config": config.to_dict(),
                        "model": ema_model.state_dict(),
                        "alignment_head": (
                            ema_alignment_head.state_dict()
                            if ema_alignment_head is not None
                            else None
                        ),
                        "alignment_config": (
                            {
                                "weight": args.alignment_weight,
                                "projection_dim": args.alignment_dim,
                                "temperature": args.alignment_temperature,
                                "language_feature_dim": language_embeddings.shape[1],
                            }
                            if alignment_head is not None
                            else None
                        ),
                        "language_alignment_metadata": alignment_metadata,
                        "epoch": epoch,
                        "best_mse": best_mse,
                    },
                    args.output,
                )
        history.append(row)
        print(json.dumps(row), flush=True)

    best = torch.load(args.output, map_location="cpu", weights_only=False)
    ema_model.load_state_dict(best["model"], strict=True)
    ema_model.to(device)
    if ema_alignment_head is not None:
        ema_alignment_head.load_state_dict(best["alignment_head"], strict=True)
        ema_alignment_head.to(device)
    metadata = cache["metadata"]
    diagnostic_family_count = int(cache["val"]["task_ids"].max()) + 1
    diagnostic_loader = make_loader(
        balanced_diagnostic_split(cache["val"]),
        args.batch_size,
        False,
        args.num_workers,
    )
    diagnostics, latent_mean, latent_std, apply_normalization = detailed_diagnostics(
        ema_model,
        diagnostic_loader,
        device,
        args.num_tasks,
        _family_names(metadata, diagnostic_family_count),
        metadata["action_low"],
        metadata["action_high"],
        instruction_texts=instruction_texts,
        alignment_head=ema_alignment_head,
        language_embeddings=language_embeddings_device,
        language_permutation=language_permutation_device,
        perturbation_std=args.perturbation_std,
    )
    best.update(
        {
            "diagnostics": diagnostics,
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "apply_latent_normalization": apply_normalization,
            "cache_path": str(args.cache),
            "cache_metadata": cache["metadata"],
            "language_embedding_metadata": language_metadata,
        }
    )
    torch.save(best, args.output)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "config": config.to_dict(),
                "best_epoch": best["epoch"],
                "best_mse": best_mse,
                "language_alignment_metadata": alignment_metadata,
                "diagnostics": diagnostics,
                "history": history,
            },
            indent=2,
        )
    )
    print(f"Saved checkpoint to {args.output}")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
