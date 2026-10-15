#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Precompute frozen Cosmos/Qwen text features for an action-cache sidecar.

Contract:
  * The cache metadata contains an ordered ``instruction_texts``/``instructions``
    list or ``instruction_registry`` entries with ``text``/``instruction``.
  * Cache split ``instruction_ids`` index that exact order.
  * This script writes ``<cache>.instruction_embeddings.pt`` by default.
  * ``train_action_vae.py --language-embeddings <sidecar>`` consumes it without
    loading any text model during VAE training or ceiling evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from examples.Robocasa_tabletop.train_files.prepare_pnp3_action_cache import (  # noqa: E402
    validate_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "Cosmos-Predict2.5-2B",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    return parser.parse_args()


def extract_instruction_texts(metadata: dict) -> list[str]:
    for key in ("instruction_texts", "instructions"):
        value = metadata.get(key)
        if isinstance(value, (list, tuple)) and all(
            isinstance(text, str) for text in value
        ):
            return list(value)
    registry = metadata.get("instruction_registry")
    if isinstance(registry, (list, tuple)):
        texts = []
        for entry in registry:
            if isinstance(entry, str):
                texts.append(entry)
            elif isinstance(entry, dict):
                text = entry.get("text") or entry.get("instruction")
                if not isinstance(text, str):
                    break
                texts.append(text)
            else:
                break
        if len(texts) == len(registry):
            return texts
    raise KeyError(
        "Cache metadata needs ordered instruction_texts, instructions, or "
        "instruction_registry entries"
    )


def _chat_prompt(tokenizer, instruction: str) -> str:
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a helpful assistant who will provide prompts "
                        "to an image generator."
                    ),
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": instruction}]},
    ]
    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False,
        add_vision_id=False,
    )


def model_inventory(path: Path) -> list[dict[str, int | str]]:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return [
        {
            "path": str(file.relative_to(resolved)),
            "size": file.stat().st_size,
            "mtime_ns": file.stat().st_mtime_ns,
        }
        for file in sorted(resolved.rglob("*"))
        if file.is_file()
    ]


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = args.cache.with_suffix(".instruction_embeddings.pt")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    validate_cache(cache)
    texts = extract_instruction_texts(cache["metadata"])
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Instruction registry contains empty text")
    for split_name in ("train", "val"):
        ids = cache[split_name].get("instruction_ids")
        if ids is None:
            raise KeyError(f"Cache {split_name} split has no instruction_ids")
        if int(ids.min()) < 0 or int(ids.max()) >= len(texts):
            raise ValueError(
                f"{split_name} instruction_ids must be in [0, {len(texts)})"
            )

    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    tokenizer_path = args.model / "tokenizer"
    text_encoder_path = args.model / "text_encoder"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path if tokenizer_path.is_dir() else args.model,
        local_files_only=True,
    )
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        text_encoder_path if text_encoder_path.is_dir() else args.model,
        torch_dtype=dtype,
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval().to(args.device)
    model.requires_grad_(False)

    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            prompts = [
                _chat_prompt(tokenizer, text)
                for text in texts[start : start + args.batch_size]
            ]
            tokens = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(args.device)
            output = model(
                **tokens,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden = output.hidden_states[-1].float()
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            embeddings.append(F.normalize(pooled, dim=-1).cpu())
            print(f"Embedded {min(start + args.batch_size, len(texts))}/{len(texts)}")

    instruction_embeddings = torch.cat(embeddings)
    text_digest = hashlib.sha256("\n".join(texts).encode()).hexdigest()
    model_files = model_inventory(args.model)
    model_digest = hashlib.sha256(repr(model_files).encode()).hexdigest()
    payload = {
        "instruction_embeddings": instruction_embeddings,
        "instruction_texts": texts,
        "model": str(args.model.resolve()),
        "model_inventory_sha256": model_digest,
        "encoder": "Cosmos2.5 Qwen2.5-VL final hidden state",
        "pooling": "attention-mask mean, L2 normalized",
        "max_length": args.max_length,
        "text_sha256": text_digest,
        "cache_path": str(args.cache.resolve()),
        "cache_fingerprint": cache["metadata"]["cache_fingerprint"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = torch.load(args.output, map_location="cpu", weights_only=False)
        identity_keys = (
            "cache_fingerprint",
            "text_sha256",
            "model_inventory_sha256",
            "encoder",
            "pooling",
            "max_length",
        )
        mismatches = {
            key: (existing.get(key), payload[key])
            for key in identity_keys
            if existing.get(key) != payload[key]
        }
        embeddings_match = (
            torch.is_tensor(existing.get("instruction_embeddings"))
            and existing["instruction_embeddings"].shape
            == instruction_embeddings.shape
            and torch.equal(
                existing["instruction_embeddings"], instruction_embeddings
            )
        )
        if mismatches or not embeddings_match:
            raise FileExistsError(
                f"Refusing to replace mismatched embedding sidecar {args.output}: "
                f"{mismatches or 'embedding payload differs'}"
            )
        print(f"Reusing identical frozen embeddings at {args.output}")
        return
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        written = torch.load(temporary, map_location="cpu", weights_only=False)
        if (
            written.get("cache_fingerprint")
            != cache["metadata"]["cache_fingerprint"]
            or not torch.equal(
                written["instruction_embeddings"], instruction_embeddings
            )
        ):
            raise RuntimeError("Embedding sidecar failed write verification")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"Saved {tuple(instruction_embeddings.shape)} frozen embeddings "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
