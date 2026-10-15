# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import DiT4DiT.training.train as train_module
from DiT4DiT.training.train import (
    VLATrainer,
    build_accelerator,
    continuation_consumed_batches,
    continuation_scheduler_steps,
)


def trainer_config(**overrides):
    values = {
        "max_train_steps": 50_000,
        "continuation_checkpoint": "/tmp/source.pt",
        "continuation_start_step": 10_000,
        "gradient_accumulation_steps": 1,
        "eval_interval": 100,
        "replay_data_position": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_accelerator_receives_configured_gradient_accumulation(monkeypatch):
    calls = {}

    class FakeDeepSpeedPlugin:
        def __init__(self, **kwargs):
            calls["plugin"] = kwargs

    class FakeAccelerator:
        def __init__(self, **kwargs):
            calls["accelerator"] = kwargs
            self.gradient_accumulation_steps = kwargs[
                "gradient_accumulation_steps"
            ]

        def print(self, _value):
            return None

        state = "test-state"

    monkeypatch.setattr(train_module, "DeepSpeedPlugin", FakeDeepSpeedPlugin)
    monkeypatch.setattr(train_module, "Accelerator", FakeAccelerator)
    cfg = SimpleNamespace(
        trainer=trainer_config(gradient_accumulation_steps=4)
    )

    accelerator = build_accelerator(cfg)

    assert accelerator.gradient_accumulation_steps == 4
    assert calls["plugin"]["gradient_accumulation_steps"] == 4
    assert calls["accelerator"]["gradient_accumulation_steps"] == 4


def test_accelerator_rejects_invalid_gradient_accumulation():
    cfg = SimpleNamespace(
        trainer=trainer_config(gradient_accumulation_steps=0)
    )
    with pytest.raises(ValueError, match="must be at least 1"):
        build_accelerator(cfg)


def test_continuation_uses_local_scheduler_horizon():
    config = trainer_config()
    assert continuation_scheduler_steps(config) == 40_000
    assert continuation_consumed_batches(config, 10_000) == 10_100

    config.continuation_consumed_batches = 12_345
    assert continuation_consumed_batches(config, 10_000) == 12_345

    with pytest.raises(ValueError, match="must exceed"):
        continuation_scheduler_steps(
            trainer_config(max_train_steps=10_000)
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        continuation_consumed_batches(
            trainer_config(continuation_consumed_batches=-1),
            10_000,
        )


def test_continuation_replays_dataloader_position():
    class FakeAccelerator:
        def __init__(self):
            self.skipped = None

        def skip_first_batches(self, dataloader, count):
            self.skipped = count
            return dataloader[count:]

    trainer = object.__new__(VLATrainer)
    trainer.config = SimpleNamespace(trainer=trainer_config())
    trainer.weights_only_continuation = True
    trainer.completed_steps = 10_000
    trainer.vla_train_dataloader = list(range(10_200))
    trainer.accelerator = FakeAccelerator()

    trainer._create_data_iterators()

    assert trainer.accelerator.skipped == 10_100
    assert next(trainer.vla_iter) == 10_100


def test_weights_only_continuation_records_restart(tmp_path: Path):
    source = tmp_path / "steps_10000_pytorch_model.pt"
    source.touch()
    output = tmp_path / "continued"
    output.mkdir()
    config = trainer_config(continuation_checkpoint=str(source))

    trainer = object.__new__(VLATrainer)
    trainer.config = SimpleNamespace(
        output_dir=str(output),
        trainer=config,
    )
    trainer.model = object()
    trainer.load_pretrained_backbones = (
        lambda model, checkpoint, reload_modules: "loaded-model"
    )

    trainer._init_checkpointing()

    assert trainer.model == "loaded-model"
    assert trainer.completed_steps == 10_000
    assert trainer.weights_only_continuation
    metadata = json.loads((output / "continuation.json").read_text())
    assert metadata["optimizer_state_restored"] is False
    assert metadata["continuation_start_step"] == 10_000
    assert metadata["data_position_replayed"] is True


def test_full_state_resume_restores_recorded_progress(tmp_path: Path):
    state = tmp_path / "steps_100000_training_state"
    state.mkdir()
    (state / "trainer_progress.json").write_text(
        json.dumps(
            {
                "completed_steps": 100_000,
                "data_batches_consumed": 101_000,
                "gradient_accumulation_steps": 1,
                "eval_interval": 100,
            }
        )
    )
    trainer = object.__new__(VLATrainer)
    trainer.config = SimpleNamespace(
        output_dir=str(tmp_path / "run"),
        trainer=trainer_config(
            continuation_checkpoint=None,
            continuation_start_step=0,
            resume_state_path=str(state),
        )
    )
    trainer.model = object()

    trainer._init_checkpointing()

    assert trainer.completed_steps == 100_000
    assert trainer.pending_resume_state == str(state)
    assert trainer.resume_data_batches_consumed == 101_000


def test_full_state_resume_uses_epoch_relative_batch_offset():
    class FakeAccelerator:
        def __init__(self):
            self.skipped = None

        def skip_first_batches(self, dataloader, count):
            self.skipped = count
            return dataloader[count:]

    trainer = object.__new__(VLATrainer)
    trainer.config = SimpleNamespace(trainer=trainer_config())
    trainer.weights_only_continuation = False
    trainer.pending_resume_state = "/tmp/steps_20_training_state"
    trainer.resume_data_batches_consumed = 25
    trainer.completed_steps = 20
    trainer.vla_train_dataloader = list(range(10))
    trainer.accelerator = FakeAccelerator()

    trainer._create_data_iterators()

    assert trainer.vla_epoch_count == 2
    assert trainer.accelerator.skipped == 5
    assert next(trainer.vla_iter) == 5
    assert trainer.data_batches_consumed == 25
