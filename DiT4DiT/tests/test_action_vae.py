# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
import torch
from torch import nn

from DiT4DiT.dataloader.gr00t_lerobot.full24_tasks import FULL24_TASKS
from DiT4DiT.dataloader.gr00t_lerobot.pnp6_tasks import PNP6_TASKS
from DiT4DiT.model.framework.DiT4DiT import (
    DiT4DiT,
    _match_action_vae_registry,
)
from DiT4DiT.model.modules.action_model.action_vae import (
    ActionVAE,
    ActionVAEConfig,
    load_action_vae,
)
from examples.Robocasa_tabletop.train_files.train_action_vae import (
    ActionLanguageAlignmentHead,
    build_language_alignment_metadata,
    detailed_diagnostics,
    deterministic_derangement,
    language_rows_for_instruction_ids,
    make_loader,
    multi_positive_alignment_loss,
)


def test_action_vae_candidate_shapes():
    actions = torch.randn(2, 16, 29)
    for num_tokens, latent_dim in ((4, 64), (8, 32)):
        model = ActionVAE(
            ActionVAEConfig(num_tokens=num_tokens, latent_dim=latent_dim)
        )
        encoded = model.encode(actions)
        reconstructed = model.decode(encoded["mu"])
        assert encoded["mu"].shape == (2, num_tokens, latent_dim)
        assert reconstructed.shape == actions.shape


def test_action_vae_checkpoint_round_trip(tmp_path: Path):
    config = ActionVAEConfig(num_tokens=8, latent_dim=32)
    model = ActionVAE(config)
    checkpoint_path = tmp_path / "action_vae.pt"
    torch.save(
        {
            "config": config.to_dict(),
            "model": model.state_dict(),
            "latent_mean": torch.zeros(8, 32),
            "latent_std": torch.ones(8, 32),
            "apply_latent_normalization": True,
            "cache_metadata": {
                "task_datasets": [f"pnp_task_{index}" for index in range(6)],
                "action_low": torch.full((29,), -1.0),
                "action_high": torch.full((29,), 1.0),
            },
        },
        checkpoint_path,
    )
    loaded, checkpoint = load_action_vae(str(checkpoint_path))
    loaded.requires_grad_(False)
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    assert checkpoint["latent_mean"].shape == (8, 32)
    assert len(checkpoint["cache_metadata"]["task_datasets"]) == 6
    assert set(model.state_dict()) == set(loaded.state_dict())


@pytest.mark.parametrize(
    ("name", "tasks"),
    (("pnp6", PNP6_TASKS), ("full24", FULL24_TASKS)),
)
def test_action_vae_registry_matching(name, tasks):
    datasets = [task.dataset_name for task in tasks]
    assert _match_action_vae_registry(datasets, len(tasks)) == (name, datasets)


def test_action_vae_registry_matching_rejects_wrong_size():
    datasets = [task.dataset_name for task in FULL24_TASKS]
    with pytest.raises(ValueError, match="num_tasks"):
        _match_action_vae_registry(datasets, 6)


class _LatentActionHarness(nn.Module):
    _encode_action_targets = DiT4DiT._encode_action_targets
    _decode_action_predictions = DiT4DiT._decode_action_predictions
    _global_to_vae_actions = DiT4DiT._global_to_vae_actions
    _vae_to_global_actions = DiT4DiT._vae_to_global_actions
    _normalize_action_space = staticmethod(DiT4DiT._normalize_action_space)
    _unnormalize_action_space = staticmethod(
        DiT4DiT._unnormalize_action_space
    )

    def __init__(self, num_tasks: int = 0, per_task_norm: bool = False):
        super().__init__()
        config = ActionVAEConfig(
            num_tokens=8,
            latent_dim=32,
            num_tasks=num_tasks,
        )
        self.action_vae = ActionVAE(config).eval().requires_grad_(False)
        self.use_action_vae = True
        self.use_per_task_vae_action_norm = per_task_norm
        self.normalize_action_latents = True
        self.register_buffer("action_latent_mean", torch.zeros(8, 32))
        self.register_buffer("action_latent_std", torch.ones(8, 32))
        if per_task_norm:
            if num_tasks != 6:
                raise ValueError("Per-task test harness requires six tasks")
            policy_low = torch.full((29,), -2.0)
            policy_high = torch.full((29,), 2.0)
            policy_low[[19, 25]] = 3.0
            policy_high[[19, 25]] = 3.0
            task_low = torch.stack(
                [torch.full((29,), -1.0 + task_id * 0.02) for task_id in range(6)]
            )
            task_high = torch.stack(
                [torch.full((29,), 1.0 - task_id * 0.02) for task_id in range(6)]
            )
            task_low[:, [19, 25]] = 3.0
            task_high[:, [19, 25]] = 3.0
            self.register_buffer("policy_action_low", policy_low)
            self.register_buffer("policy_action_high", policy_high)
            self.register_buffer("vae_action_low", task_low)
            self.register_buffer("vae_action_high", task_high)


def test_latent_framework_boundary_shapes():
    harness = _LatentActionHarness()
    raw_actions = torch.randn(2, 16, 32)
    raw_mask = torch.zeros(2, 16, 32, dtype=torch.bool)
    raw_mask[..., :29] = True
    latent, latent_mask = harness._encode_action_targets(raw_actions, raw_mask)
    decoded = harness._decode_action_predictions(latent)
    assert latent.shape == (2, 8, 32)
    assert latent_mask.shape == latent.shape
    assert latent_mask.all()
    assert decoded.shape == raw_actions.shape
    assert torch.equal(decoded[..., 29:], torch.zeros_like(decoded[..., 29:]))


def test_task_conditioned_action_vae_boundaries():
    harness = _LatentActionHarness(num_tasks=6)
    raw_actions = torch.randn(2, 16, 32)
    raw_mask = torch.zeros(2, 16, 32, dtype=torch.bool)
    raw_mask[..., :29] = True
    task_ids = torch.tensor([0, 5])
    latent, latent_mask = harness._encode_action_targets(
        raw_actions,
        raw_mask,
        task_ids=task_ids,
    )
    decoded = harness._decode_action_predictions(latent, task_ids=task_ids)
    assert latent.shape == (2, 8, 32)
    assert latent_mask.all()
    assert decoded.shape == raw_actions.shape

    with pytest.raises(ValueError, match="task_ids are required"):
        harness._encode_action_targets(raw_actions, raw_mask)
    with pytest.raises((IndexError, RuntimeError)):
        harness._decode_action_predictions(
            latent,
            task_ids=torch.tensor([0, 6]),
        )


def test_per_task_action_space_bridge_round_trip():
    harness = _LatentActionHarness(num_tasks=6, per_task_norm=True)
    task_ids = torch.tensor([0, 3, 5])
    physical = torch.linspace(-0.25, 0.25, 3 * 16 * 29).reshape(3, 16, 29)
    physical[..., [19, 25]] = 3.0
    policy_low = harness.policy_action_low.view(1, 1, -1)
    policy_high = harness.policy_action_high.view(1, 1, -1)
    global_actions = harness._normalize_action_space(
        physical,
        policy_low,
        policy_high,
    )
    vae_actions = harness._global_to_vae_actions(global_actions, task_ids)
    reconstructed = harness._vae_to_global_actions(vae_actions, task_ids)
    assert torch.allclose(reconstructed, global_actions, atol=2e-6)
    assert torch.equal(
        reconstructed[..., [19, 25]],
        torch.zeros_like(reconstructed[..., [19, 25]]),
    )
    assert not torch.allclose(vae_actions[0], vae_actions[2])

    with pytest.raises(ValueError, match="requires task_ids"):
        harness._global_to_vae_actions(global_actions, None)
    with pytest.raises((IndexError, RuntimeError)):
        harness._vae_to_global_actions(
            vae_actions,
            torch.tensor([0, 3, 6]),
        )


def test_task_conditioned_checkpoint_round_trip(tmp_path: Path):
    config = ActionVAEConfig(num_tokens=8, latent_dim=32, num_tasks=6)
    model = ActionVAE(config)
    with torch.no_grad():
        model.encoder.task_embedding.weight.copy_(
            torch.arange(6 * 128).reshape(6, 128)
        )
    checkpoint_path = tmp_path / "task_conditioned_action_vae.pt"
    torch.save(
        {"config": config.to_dict(), "model": model.state_dict()},
        checkpoint_path,
    )
    loaded, _ = load_action_vae(str(checkpoint_path))
    assert loaded.config.num_tasks == 6
    assert torch.equal(
        loaded.encoder.task_embedding.weight,
        model.encoder.task_embedding.weight,
    )


@pytest.mark.parametrize(
    ("mode", "encoder_conditioned", "decoder_conditioned"),
    [
        ("none", False, False),
        ("encoder_only", True, False),
        ("encoder_decoder", True, True),
    ],
)
def test_action_vae_conditioning_modes(
    mode: str,
    encoder_conditioned: bool,
    decoder_conditioned: bool,
):
    model = ActionVAE(
        ActionVAEConfig(
            num_tokens=8,
            latent_dim=32,
            num_tasks=3,
            conditioning_mode=mode,
        )
    )
    actions = torch.randn(2, 16, 29)
    task_ids = torch.tensor([0, 2])
    assert (model.encoder.task_embedding is not None) is encoder_conditioned
    assert (model.decoder.task_embedding is not None) is decoder_conditioned

    if encoder_conditioned:
        with pytest.raises(ValueError, match="task_ids are required"):
            model.encode(actions)
        encoded = model.encode(actions, task_ids)
    else:
        encoded = model.encode(actions)
    latent = encoded["mu"]

    if decoder_conditioned:
        with pytest.raises(ValueError, match="task_ids are required"):
            model.decode(latent)
        reconstruction = model.decode(latent, task_ids)
    else:
        reconstruction = model.decode(latent)
        assert torch.equal(
            reconstruction,
            model.decode(latent, torch.tensor([2, 0])),
        )
    assert reconstruction.shape == actions.shape


def test_legacy_task_checkpoint_defaults_to_encoder_decoder(tmp_path: Path):
    config = ActionVAEConfig(num_tokens=8, latent_dim=32, num_tasks=3)
    model = ActionVAE(config)
    legacy_config = config.to_dict()
    legacy_config.pop("conditioning_mode")
    checkpoint_path = tmp_path / "legacy_action_vae.pt"
    torch.save(
        {"config": legacy_config, "model": model.state_dict()},
        checkpoint_path,
    )

    loaded, _ = load_action_vae(str(checkpoint_path))
    assert loaded.config.conditioning_mode == "encoder_decoder"
    assert loaded.encoder.task_embedding is not None
    assert loaded.decoder.task_embedding is not None
    assert set(loaded.state_dict()) == set(model.state_dict())


def test_multi_positive_alignment_loss_and_gradients():
    instruction_ids = torch.tensor([0, 0, 1, 1])
    matched = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        requires_grad=True,
    )
    text = matched.detach().clone()
    mismatched_text = text.flip(0)

    matched_loss = multi_positive_alignment_loss(
        matched,
        text,
        instruction_ids,
        temperature=0.1,
    )
    mismatched_loss = multi_positive_alignment_loss(
        matched,
        mismatched_text,
        instruction_ids,
        temperature=0.1,
    )
    assert matched_loss < mismatched_loss
    matched_loss.backward()
    assert matched.grad is not None
    assert torch.isfinite(matched.grad).all()


def test_language_derangement_is_deterministic_and_fixed_point_free():
    first = deterministic_derangement(186, seed=42)
    second = deterministic_derangement(186, seed=42)
    different = deterministic_derangement(186, seed=43)

    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert torch.equal(first.sort().values, torch.arange(186))
    assert not torch.any(first == torch.arange(186))


def test_true_language_lookup_is_unchanged_and_shuffle_preserves_labels():
    instruction_ids = torch.tensor([0, 2, 2, 4])
    assert (
        language_rows_for_instruction_ids(instruction_ids, None)
        is instruction_ids
    )
    permutation = deterministic_derangement(5, seed=7)
    rows = language_rows_for_instruction_ids(instruction_ids, permutation)
    assert torch.equal(rows, permutation[instruction_ids])
    assert torch.equal(instruction_ids, torch.tensor([0, 2, 2, 4]))


def test_language_alignment_checkpoint_metadata_round_trip(tmp_path: Path):
    permutation = deterministic_derangement(4, seed=44)
    metadata = build_language_alignment_metadata(
        instruction_count=4,
        instruction_texts=["a", "b", "c", "d"],
        shuffle_seed=44,
        permutation=permutation,
        cache_fingerprint="cache-sha",
    )
    checkpoint = tmp_path / "metadata.pt"
    torch.save({"language_alignment_metadata": metadata}, checkpoint)
    loaded = torch.load(checkpoint, weights_only=False)

    assert loaded["language_alignment_metadata"] == metadata
    assert metadata["multi_positive_grouping"] == "original_instruction_ids"
    assert metadata["fixed_point_count"] == 0
    assert metadata["permutation"] == permutation.tolist()


def test_alignment_head_does_not_change_decoder_contract():
    model = ActionVAE(
        ActionVAEConfig(
            num_tokens=8,
            latent_dim=32,
            num_tasks=3,
            conditioning_mode="none",
        )
    )
    actions = torch.randn(2, 16, 29)
    latent = model.encode(actions)["mu"]
    head = ActionLanguageAlignmentHead(8 * 32, 16, 8)
    action_features, text_features = head(latent, torch.randn(2, 16))

    assert action_features.shape == text_features.shape == (2, 8)
    assert model.decode(latent).shape == actions.shape
    assert not any("alignment" in key for key in model.state_dict())


def test_alignment_and_stability_diagnostics():
    model = ActionVAE(
        ActionVAEConfig(
            action_horizon=16,
            num_tokens=8,
            latent_dim=16,
            encoder_hidden=16,
            decoder_hidden=16,
            num_tasks=2,
            conditioning_mode="encoder_only",
        )
    )
    split = {
        "actions": torch.randn(4, 16, 29),
        "valid_steps": torch.ones(4, 16, dtype=torch.bool),
        "task_ids": torch.tensor([0, 0, 1, 1]),
        "instruction_ids": torch.tensor([0, 0, 1, 1]),
        "episode_ids": torch.tensor([0, 0, 0, 0]),
        "start_steps": torch.tensor([0, 1, 0, 1]),
    }
    loader = make_loader(split, batch_size=2, shuffle=False, num_workers=0)
    language_embeddings = torch.randn(2, 12)
    alignment_head = ActionLanguageAlignmentHead(8 * 16, 12, 8)

    diagnostics, latent_mean, latent_std, _ = detailed_diagnostics(
        model,
        loader,
        torch.device("cpu"),
        num_tasks=2,
        task_names=["family_a", "family_b"],
        action_low=torch.full((2, 29), -1.0),
        action_high=torch.full((2, 29), 1.0),
        instruction_texts=["instruction a", "instruction b"],
        alignment_head=alignment_head,
        language_embeddings=language_embeddings,
        language_permutation=deterministic_derangement(2, seed=42),
        perturbation_std=0.01,
    )

    assert latent_mean.shape == latent_std.shape == (8, 16)
    assert diagnostics["family_reconstruction"]["worst_mean_group"] in {
        "family_a",
        "family_b",
    }
    assert set(diagnostics["instruction_reconstruction"]["groups"]) == {
        "instruction a",
        "instruction b",
    }
    assert diagnostics["stability"]["adjacent_pair_count"] == 2
    assert diagnostics["stability"]["one_step_shift_latent_rms_mean"] >= 0
    assert diagnostics["language_alignment"]["sample_count"] == 4
    assert set(diagnostics["language_alignment"]) >= {
        "semantic_mapping",
        "trained_mapping",
    }
    assert diagnostics["language_alignment"]["trained_mapping"]["kind"] == "shuffled"
    assert diagnostics["latent"]["effective_dimensions"] > 0
    assert diagnostics["latent"]["collapsed_dimensions"] >= 0
    assert diagnostics["hand_transition_abs_error"]["p99"] >= 0
