---
license: apache-2.0
base_model: Wan-AI/Wan2.2-TI2V-5B
datasets:
  - yuanty/LIBERO-fastwam
library_name: pytorch
tags:
  - robotics
  - world-action-model
  - libero
pipeline_tag: robotics
---

# CAE-FastWAM

A latent world-action policy for the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark, fine-tuned by Advanced Micro Devices, Inc. from [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).

Code: [AMD-AGI/faster-action-models](https://github.com/AMD-AGI/faster-action-models).
Write-up and reference results: [Optimizing World Action Models on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/faster-action-models/README.html).

Part of the [Compact Action Expert (CAE) collection](https://huggingface.co/collections/amd/compact-action-expert-cae-6ace112e7bd665ff8c107ed2).

## Model

The Compact Action Expert combines a 32-dimensional action expert (about 30M parameters) with a suite-conditioned K8 × d16 action VAE. The VAE encodes each 32 × 7 action chunk as 8 latent tokens of width 16, and the action expert denoises those latent chunks.

## Results

LIBERO (all four suites, 50 episodes per task) on one AMD Instinct™ MI355X GPU with ROCm 7.14.0 and PyTorch 2.11.0, simulated on the GPU (`--sim instinct`), with the GEAK-optimized inference path from the code repository. Reference results, from Table 3 of the [write-up](https://rocm.blogs.amd.com/artificial-intelligence/faster-action-models/README.html):

| Configuration | Success (%) | Latency per policy decision (ms) |
|---|---|---|
| FastWAM, no optimization (10 denoising steps) | 96.95 ± 0.22 | 222.55 |
| FastWAM + GEAK (10 steps) | 96.86 ± 0.30 | 24.17 |
| CAE-FastWAM + GEAK (10 steps) | 97.28 ± 0.20 | 12.82 |
| CAE-FastWAM + GEAK (4 steps) | 97.33 ± 0.26 | 8.83 |

Success is the mean ± standard deviation over 11 full LIBERO runs (5 for the 4-step row). The FastWAM rows use the authors' [`yuanty/fastwam`](https://huggingface.co/yuanty/fastwam) checkpoint. Reproduce the table with `scripts/reproduce_cae_fastwam.sh` in the code repository. Training took 22.7 hours on two AMD Instinct™ MI325X GPUs. Results are simulator measurements and may vary with configuration and software versions.

## Files

| File | Description |
|---|---|
| `policy.pt` | Inference weights for the video and action experts |
| `action_vae.pt` | Suite-conditioned action VAE |
| `stats/libero_spatial.json` | Normalization statistics for LIBERO-Spatial |
| `stats/libero_object.json` | Normalization statistics for LIBERO-Object |
| `stats/libero_goal.json` | Normalization statistics for LIBERO-Goal |
| `stats/libero_10.json` | Normalization statistics for LIBERO-10 |

Use the statistics file that matches the suite under evaluation. `policy.pt` contains inference weights only.

SHA-256 checksums are in [`ASSET_SHA256SUMS`](ASSET_SHA256SUMS).

## Download

Run the downloads with a current `huggingface_hub` (`pip install -U huggingface_hub`), which provides the `hf` command. Releases before 0.34 only have the older spelling, `huggingface-cli download`. Download into a directory of its own, for example inside a clone of the code repository:

```bash
hf download amd/CAE-FastWAM --local-dir checkpoints/CAE-FastWAM
(cd checkpoints/CAE-FastWAM && grep -E '\.(pt|json)$' ASSET_SHA256SUMS | sha256sum -c -)
```

The evaluation launchers in the code repository take the files as:

```bash
export MODEL_CKPT=$PWD/checkpoints/CAE-FastWAM/policy.pt
export ACTION_VAE_CKPT=$PWD/checkpoints/CAE-FastWAM/action_vae.pt
export DATASET_STATS_TEMPLATE=$PWD/checkpoints/CAE-FastWAM/stats/{suite}.json
export TASK_CONFIG=libero_uncond_2cam224_1e-4_32hdim_latent_K8d16_freebits_pernorm_joint
```

## Runtime weights

This repository does not include the Wan2.2 VAE, the umT5-XXL text encoder, or the umT5-XXL tokenizer. Download them from their upstream repositories:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth \
  --local-dir checkpoints/Wan-AI/Wan2.2-TI2V-5B
hf download Wan-AI/Wan2.1-T2V-1.3B \
  --include "google/umt5-xxl/*" \
  --local-dir checkpoints/Wan-AI/Wan2.1-T2V-1.3B
export DIFFSYNTH_MODEL_BASE_PATH=$PWD/checkpoints
```

The full Wan video DiT is not required. `policy.pt` already contains the fine-tuned video and action experts.

LIBERO simulator assets, BDDL task files, and initial states are also required. The evaluation commands are in [`MODEL_ZOO.md`](https://github.com/AMD-AGI/faster-action-models/blob/main/MODEL_ZOO.md) and [`docs/cae-fastwam-libero.md`](https://github.com/AMD-AGI/faster-action-models/blob/main/docs/cae-fastwam-libero.md).

## Intended use

Research on world-action models and robot-policy inference efficiency, in simulation.

## Limitations

- Evaluated only in the LIBERO simulator. Not validated on physical robots.
- Not intended for safety-critical or high-stakes applications. Users are responsible for evaluating the policy and adding appropriate safety mechanisms.
- Use must also follow the Wan2.2 model card.

## License

These weights are licensed under the Apache License 2.0 ([`LICENSE`](LICENSE)). See [`NOTICE`](NOTICE).

They are a modified version of [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) (Apache-2.0, Wan Team, Alibaba), fine-tuned by Advanced Micro Devices, Inc. The training recipe is [FastWAM](https://github.com/yuantianyuan01/FastWAM) (MIT). That license covers the training code, not these weights.

Training data: [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (CC BY 4.0), in the LeRobot format published as [yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam) (CC BY 4.0). This is the dataset used by the FastWAM LIBERO recipe, and it is used as published. No changes were made to the training data. No extra training data was used beyond the FastWAM recipe.

## Citing

```bibtex
@misc{CAE-FastWAM,
    title = {CAE-FastWAM: Compact Action Expert for FastWAM},
    url = {https://huggingface.co/amd/CAE-FastWAM},
    author = {Hitesh Kandala, Raghav Goyal, Pratik Prabhanjan Brahma, Emad Barsoum},
    month = {October},
    year = {2026}
}
```

- T. Yuan, Z. Dong, Y. Liu, and H. Zhao, "Fast-WAM: Do World Action Models Need Test-time Future Imagination?" arXiv:2603.16666, 2026.
- B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone, "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning," NeurIPS 2023.
