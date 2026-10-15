# Faster Action Models

Inference optimization and Compact Action Experts for World Action Models
(FastWAM, DiT4DiT) on AMD Instinct™ GPUs.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Full write-up and results: the
[ROCm blog post](https://rocm.blogs.amd.com/artificial-intelligence/faster-action-models/README.html).
Released checkpoint: [`amd/CAE-FastWAM`](https://huggingface.co/amd/CAE-FastWAM), in the [CAE collection](https://huggingface.co/collections/amd/compact-action-expert-cae).

## Overview

World Action Models (WAMs) condition an action expert on a pretrained video
backbone. This repository makes two of them faster in two ways, and provides
the tools to evaluate and reproduce the results:

- **Training-free optimization**: GEAK-optimized inference trees for
  [FastWAM](https://github.com/yuantianyuan01/FastWAM) and
  [DiT4DiT](https://github.com/Mondo-Robotics/DiT4DiT), plus fewer denoising
  steps, with the weights unchanged. [GEAK](https://www.amd.com/en/developer/resources/technical-articles/2026/geak-v4.html)
  is AMD's agentic GPU-kernel optimization framework.
- **Compact Action Expert (CAE)**: a Compact Action DiT (hidden size 1024 -> 32)
  that denoises continuous latents from an Action VAE. CAE-FastWAM (LIBERO,
  32x7 action chunk -> 8x16 latents) and CAE-DiT4DiT (RoboCasa-GR1, 16x29 ->
  8x48).
- **Evaluation launchers** for LIBERO and RoboCasa-GR1 (24 task families), on
  CPU MuJoCo or GPU-native on AMD Instinct™ GPUs (`--sim instinct`).
- **Latency tools**, **training scripts** and **reproduction scripts** for every
  configuration in the blog.

## Models

| Model | Benchmark | Checkpoint |
|---|---|---|
| CAE-FastWAM | LIBERO | [`amd/CAE-FastWAM`](https://huggingface.co/amd/CAE-FastWAM) (`huggingface-cli download amd/CAE-FastWAM`); paths, hashes and download steps in [`MODEL_ZOO.md`](MODEL_ZOO.md); model card in [`model_cards/fastwam-libero/`](model_cards/fastwam-libero/) |
| CAE-DiT4DiT | RoboCasa-GR1 | Not released. Train it with the scripts in [Training](#training) ([guide](docs/cae-dit4dit-robocasa.md)). |

The FastWAM and DiT4DiT baseline checkpoints come from their authors.

## Requirements

- Linux with ROCm on AMD Instinct™ GPUs, or a Radeon™ GPU for the CPU-MuJoCo
  path; `--sim instinct` targets gfx950 and gfx942.
- A ROCm PyTorch environment per model. Keep the installed ROCm PyTorch: install
  the vendored trees with `pip install -e . --no-deps`, never with the
  CUDA-pinned dependencies from `FastWAM/pyproject.toml`.
- FastWAM: the ROCm evaluation container (`EVAL_IMAGE`), the
  [vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness)
  (`EVAL_HARNESS`), and a LIBERO checkout (`LIBERO_ROOT`) with `mujoco==3.3.2`.
  Setup: [FastWAM GEAK guide](docs/fastwam-geak-inference-optimization.md).
- DiT4DiT: separate policy and simulator Python environments (LIBERO), and the
  pinned RoboCasa-GR1 stack from the [CAE-DiT4DiT guide](docs/cae-dit4dit-robocasa.md)
  (robosuite 1.5.1 overlay, deterministic-reset patch, gated Cosmos-Predict2.5-2B).
- `--sim instinct`: build the GPU simulator with `simulate/build.sh libero|robocasa`
  and set `INSTINCT_SIM_BUILD` ([`simulate/README.md`](simulate/README.md)).
  Each launcher's `--help` lists the environment variables it needs.

## Quick start

```bash
git clone https://github.com/AMD-AGI/faster-action-models.git
cd faster-action-models

# Print the resolved run without starting anything (needs no checkpoint).
bash scripts/eval_fastwam_libero.sh --arm cae --steps 4 --smoke --dry-run

# One rollout per task, once MODEL_CKPT, ACTION_VAE_CKPT, TASK_CONFIG,
# DATASET_STATS_TEMPLATE, EVAL_HARNESS, LIBERO_ROOT and EVAL_IMAGE are set.
bash scripts/eval_fastwam_libero.sh --arm cae --steps 4 --smoke
```

## Evaluation

| Launcher | Model, benchmark | Arms |
|---|---|---|
| `scripts/eval_fastwam_libero.sh` | FastWAM, LIBERO (evaluation harness, CPU MuJoCo) | `baseline`, `geak`, `cae` |
| `eval/native/run_fastwam_libero.sh` | FastWAM, LIBERO (FastWAM's own evaluator; the FastWAM path for `--sim instinct`) | `baseline`, `geak`, `cae` |
| `scripts/eval_dit4dit_libero.sh` | DiT4DiT, LIBERO | `baseline`, `geak` |
| `scripts/eval_dit4dit_robocasa.sh` | DiT4DiT, RoboCasa-GR1 full-24 | `baseline`, `geak`, `cae` |

| Configuration | Benchmark | Arm | Denoise steps |
|---|---|---|---|
| FastWAM baseline | LIBERO | `--arm baseline` | 10 |
| FastWAM GEAK | LIBERO | `--arm geak` | 10, 2 |
| CAE-FastWAM | LIBERO | `--arm cae` | 10, 4 |
| DiT4DiT baseline, GEAK | LIBERO | `--arm baseline`, `--arm geak` | 4 (paper default), 2 |
| DiT4DiT baseline, GEAK | RoboCasa-GR1 full-24 | `--arm baseline`, `--arm geak` | 4 (paper default), 2 |
| CAE-DiT4DiT | RoboCasa-GR1 full-24 | `--arm cae` | checkpoint default |

- Shared flags: `--arm`, `--steps`, `--sim mujoco|instinct` (`instinct` on all
  launchers except `scripts/eval_fastwam_libero.sh`), `--gpus`,
  `--smoke`/`--full`/`--trials`, `--suites` (LIBERO), `--output-dir`,
  `--clients-per-gpu` and `--port` (launchers that start a policy server),
  `--record-video`, `--dry-run`. Details:
  [common flags](docs/side-by-side-source-trees.md#common-flags).
- `--sim mujoco` (default) runs physics on the CPU; `--sim instinct` runs physics
  and camera rendering GPU-native on AMD Instinct™ GPUs.
- MuJoCo renders camera images with GPU OpenGL (EGL) when available and falls
  back to OSMesa; set `MUJOCO_GL` to force one.
- Every run writes `<output-dir>/summary.json` and prints one `eval complete: ...`
  line. A rerun into the same `--output-dir` resumes the same run only.
- Protocol: [`eval/PROTOCOL.md`](eval/PROTOCOL.md). Command per configuration:
  [`docs/side-by-side-source-trees.md`](docs/side-by-side-source-trees.md).
  Rollout videos: [`docs/rollout-videos.md`](docs/rollout-videos.md).

## Reproducing the blog results

Each script runs one set of published configurations on CPU MuJoCo, with the
`--sim instinct` twin of each run commented out beneath it. Its header lists the
checkpoints it needs.

| Script | Runs |
|---|---|
| [`scripts/reproduce_baselines.sh`](scripts/reproduce_baselines.sh) | FastWAM and DiT4DiT on LIBERO, DiT4DiT on RoboCasa-GR1 |
| [`scripts/reproduce_training_free.sh`](scripts/reproduce_training_free.sh) | Baseline, GEAK, and GEAK with fewer denoising steps: success rate and latency |
| [`scripts/reproduce_cae_fastwam.sh`](scripts/reproduce_cae_fastwam.sh) | CAE-FastWAM next to the FastWAM baseline and GEAK: success rate and latency |
| [`scripts/reproduce_cae_dit4dit.sh`](scripts/reproduce_cae_dit4dit.sh) | CAE-DiT4DiT on RoboCasa-GR1 full-24; needs a CAE-DiT4DiT checkpoint you train (see [Training](#training)) |
| [`scripts/run_all_evals.sh`](scripts/run_all_evals.sh) | Every documented launch: each launcher, arm and step count |

## Latency tools

Each tool times its arms interleaved in one session on one GPU (GEAK campaign
setup: [`geak/README.md`](geak/README.md)):

```bash
python3 scripts/compare_artifacts.py --arms eager,compile,geak,geak@2 --rounds 5   # FastWAM
python3 scripts/cae_fastwam_latency.py --arms cae@10,cae@4 --rounds 5             # CAE-FastWAM
python3 scripts/dit4dit_latency.py --bench libero --dry-run                       # DiT4DiT (libero|robocasa)
```

## Training

```bash
bash scripts/train_fastwam_libero_action_vae.sh
bash scripts/train_fastwam_libero_latent_policy.sh
bash scripts/train_dit4dit_robocasa_action_vae.sh
bash scripts/train_dit4dit_robocasa_latent_policy.sh
```

Each wrapper packages the published hyperparameters, calls the model's native
trainer, and writes a GNU `time -v` resource report when `/usr/bin/time` exists.
`--help` lists inputs and hardware defaults. Guides:
[CAE-FastWAM](docs/cae-fastwam-libero.md),
[CAE-DiT4DiT](docs/cae-dit4dit-robocasa.md).

## Repository layout

```text
FastWAM/, DiT4DiT/   vendored upstream code with the CAE models and trainers
src/                 side-by-side inference trees: fastwam-{baseline,geak}, cae-fastwam, dit4dit-{baseline,geak}
geak/                GEAK task setup for an action model
eval/                evaluation harness, protocol, native FastWAM evaluator
simulate/            GPU-native simulator for --sim instinct
scripts/             launchers, latency tools, training and reproduction scripts
docs/                guides; model_cards/ holds the Hugging Face model card
```

## Limitations

- Research checkpoints evaluated in simulation only (LIBERO, RoboCasa-GR1).
- Not validated on physical robots; not for safety-critical use.
- Not bit-reproducible across GPU models and software versions; reruns can differ.

## Citation

If you use this repository, please cite the blog post and the upstream models:

```bibtex
@misc{kandala2026faster,
  title  = {Optimizing World Action Models on AMD GPUs},
  author = {Kandala, Hitesh and Goyal, Raghav and Brahma, Pratik Prabhanjan and Barsoum, Emad},
  year   = {2026},
  howpublished = {ROCm Blogs},
  url    = {https://rocm.blogs.amd.com/artificial-intelligence/faster-action-models/README.html}
}
```

Upstream: FastWAM ([arXiv:2603.16666](https://arxiv.org/abs/2603.16666)),
DiT4DiT ([arXiv:2603.10448](https://arxiv.org/abs/2603.10448)), LIBERO
([arXiv:2306.03310](https://arxiv.org/abs/2306.03310)), RoboCasa
([arXiv:2406.02523](https://arxiv.org/abs/2406.02523)).

## License

- AMD code: Apache-2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)).
- Vendored FastWAM and DiT4DiT code: MIT, under their own `LICENSE` files; files
  with their own license header keep it.
- CAE-FastWAM checkpoint: Apache-2.0, fine-tuned from Wan2.2-TI2V-5B
  ([`model_cards/fastwam-libero/`](model_cards/fastwam-libero/)).
- Third-party models, datasets and simulator assets are not redistributed; they are
  downloaded under their own terms ([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).

## Contributing, security and contact

- Contributions: [`CONTRIBUTING.md`](CONTRIBUTING.md), [Code of Conduct](CODE_OF_CONDUCT.md).
- Security: report vulnerabilities privately per [`SECURITY.md`](SECURITY.md),
  not in public issues.
- Maintainers: @AMD-AGI/brain-genai ([`.github/CODEOWNERS`](.github/CODEOWNERS)).
