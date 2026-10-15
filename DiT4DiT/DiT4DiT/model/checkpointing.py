# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint helpers for full and trainable-delta model artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


TRAINABLE_DELTA_FORMAT = "dit4dit_trainable_delta_v1"


def build_trainable_delta_checkpoint(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    base_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep trainable parameters and buffers while recording the full key contract."""
    trainable_names = {
        name
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if parameter.requires_grad
    }
    # ``named_buffers`` includes non-persistent buffers (for example rotary
    # embedding caches), while ``state_dict`` intentionally excludes them.
    buffer_names = {
        name
        for name, _ in model.named_buffers(remove_duplicate=False)
        if name in state_dict
    }
    retained_names = trainable_names | buffer_names
    missing_trainable = trainable_names - set(state_dict)
    if missing_trainable:
        raise ValueError(
            "Gathered state dict is missing trainable parameters: "
            + ", ".join(sorted(missing_trainable))
        )
    delta_state = {
        name: tensor
        for name, tensor in state_dict.items()
        if name in retained_names
    }
    return {
        "checkpoint_format": TRAINABLE_DELTA_FORMAT,
        "state_dict": delta_state,
        "expected_model_keys": sorted(state_dict),
        "trainable_parameter_names": sorted(trainable_names),
        "buffer_names": sorted(buffer_names),
        "base_artifacts": dict(base_artifacts or {}),
    }


def checkpoint_state_dict(payload: object) -> Mapping[str, torch.Tensor]:
    """Return the state mapping from either a full or delta checkpoint payload."""
    if (
        isinstance(payload, Mapping)
        and payload.get("checkpoint_format") == TRAINABLE_DELTA_FORMAT
    ):
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("Delta checkpoint is missing its state_dict mapping")
        return state_dict
    if not isinstance(payload, Mapping):
        raise TypeError("Model checkpoint must contain a state-dict mapping")
    return payload


def load_model_checkpoint(
    model: nn.Module,
    payload: object,
    *,
    strict_full: bool = True,
) -> None:
    """Load a full checkpoint strictly or a validated trainable delta."""
    if not (
        isinstance(payload, Mapping)
        and payload.get("checkpoint_format") == TRAINABLE_DELTA_FORMAT
    ):
        model.load_state_dict(checkpoint_state_dict(payload), strict=strict_full)
        return

    state_dict = checkpoint_state_dict(payload)
    expected_keys = payload.get("expected_model_keys")
    trainable_names = payload.get("trainable_parameter_names")
    buffer_names = payload.get("buffer_names")
    if not all(
        isinstance(names, list) and all(isinstance(name, str) for name in names)
        for names in (expected_keys, trainable_names, buffer_names)
    ):
        raise ValueError("Delta checkpoint has an invalid key manifest")

    current_keys = set(model.state_dict())
    if current_keys != set(expected_keys):
        missing = sorted(set(expected_keys) - current_keys)
        unexpected = sorted(current_keys - set(expected_keys))
        raise RuntimeError(
            "Delta checkpoint architecture mismatch; "
            f"missing current keys={missing}, unexpected current keys={unexpected}"
        )

    expected_key_set = set(expected_keys)
    if not set(trainable_names) <= expected_key_set:
        invalid = sorted(set(trainable_names) - expected_key_set)
        raise RuntimeError(
            "Delta checkpoint trainable manifest contains non-state keys: "
            f"{invalid}"
        )

    # Early v1 deltas recorded every named buffer, including non-persistent
    # buffers that are absent from both state_dict and expected_model_keys.
    # Ignore only those explicitly non-state buffer entries for compatibility.
    required_delta_keys = set(trainable_names) | (
        set(buffer_names) & expected_key_set
    )
    if set(state_dict) != required_delta_keys:
        missing = sorted(required_delta_keys - set(state_dict))
        unexpected = sorted(set(state_dict) - required_delta_keys)
        raise RuntimeError(
            "Delta checkpoint payload does not match its manifest; "
            f"missing keys={missing}, unexpected keys={unexpected}"
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    expected_missing = current_keys - required_delta_keys
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Delta checkpoint load contract failed; "
            f"missing keys={incompatible.missing_keys}, "
            f"unexpected keys={incompatible.unexpected_keys}"
        )
