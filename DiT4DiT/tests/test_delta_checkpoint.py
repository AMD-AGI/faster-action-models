# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
import torch
from torch import nn

from DiT4DiT.model.checkpointing import (
    TRAINABLE_DELTA_FORMAT,
    build_trainable_delta_checkpoint,
    load_model_checkpoint,
)


class TinyModel(nn.Module):
    def __init__(self, extra_layer: bool = False):
        super().__init__()
        self.frozen = nn.Linear(3, 3)
        self.trainable = nn.Linear(3, 2)
        self.register_buffer("running_value", torch.zeros(2))
        self.register_buffer(
            "runtime_cache",
            torch.ones(2),
            persistent=False,
        )
        if extra_layer:
            self.extra = nn.Linear(2, 1)


def make_model() -> TinyModel:
    torch.manual_seed(7)
    model = TinyModel()
    model.frozen.requires_grad_(False)
    return model


def test_trainable_delta_round_trip(tmp_path: Path):
    source = make_model()
    with torch.no_grad():
        source.trainable.weight.add_(2.0)
        source.trainable.bias.sub_(1.0)
        source.running_value.copy_(torch.tensor([4.0, 5.0]))

    payload = build_trainable_delta_checkpoint(
        source,
        source.state_dict(),
        base_artifacts={"base_model": "/models/base"},
    )
    assert payload["checkpoint_format"] == TRAINABLE_DELTA_FORMAT
    assert set(payload["state_dict"]) == {
        "trainable.weight",
        "trainable.bias",
        "running_value",
    }
    assert "runtime_cache" not in payload["buffer_names"]
    assert not any(name.startswith("frozen.") for name in payload["state_dict"])

    path = tmp_path / "delta.pt"
    torch.save(payload, path)
    restored = make_model()
    load_model_checkpoint(
        restored,
        torch.load(path, map_location="cpu", weights_only=True),
    )
    for name, value in source.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value), name


def test_early_delta_ignores_nonpersistent_buffer_manifest_entry():
    source = make_model()
    payload = build_trainable_delta_checkpoint(source, source.state_dict())
    payload["buffer_names"].append("runtime_cache")

    restored = make_model()
    load_model_checkpoint(restored, payload)
    for name, value in source.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value), name


def test_trainable_delta_rejects_architecture_mismatch():
    source = make_model()
    payload = build_trainable_delta_checkpoint(source, source.state_dict())
    target = TinyModel(extra_layer=True)
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        load_model_checkpoint(target, payload)


def test_full_checkpoint_stays_strict():
    source = make_model()
    target = TinyModel(extra_layer=True)
    with pytest.raises(RuntimeError):
        load_model_checkpoint(target, source.state_dict(), strict_full=True)
