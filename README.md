# Faster Action Models

Inference optimization and Compact Action Experts for World Action Models
(FastWAM, DiT4DiT) on AMD Instinct™ GPUs.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

ROCm blog with the reference results:
[Optimizing World Action Models on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/faster-action-models/README.html).
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
- **Compact Action Expert (CAE)**: two changes together, a smaller Action DiT
  (hidden size 1024 -> 32) and an Action VAE whose continuous latents it
  denoises. CAE-FastWAM (LIBERO,
  32x7 action chunk -> 8x16 latents) and CAE-DiT4DiT (RoboCasa-GR1, 16x29 ->
  8x48).
- **Evaluation launchers** for LIBERO and RoboCasa-GR1 (24 task families), on
  CPU MuJoCo or GPU-native on AMD Instinct™ MI350X/MI355X GPUs (`--sim instinct`).
- **Latency tools**, **training scripts** and **reproduction scripts** for every
  configuration of the reference results.

## Models

| Model | Benchmark | Checkpoint |
|---|---|---|
| CAE-FastWAM | LIBERO | [`amd/CAE-FastWAM`](https://huggingface.co/amd/CAE-FastWAM) (`hf download amd/CAE-FastWAM --local-dir checkpoints/CAE-FastWAM`); files, hashes and launcher variables in [`MODEL_ZOO.md`](MODEL_ZOO.md); model card in [`model_cards/fastwam-libero/`](model_cards/fastwam-libero/) |
| CAE-DiT4DiT | RoboCasa-GR1 | Not released. Train it with the scripts in [Training](#training) ([guide](docs/cae-dit4dit-robocasa.md)). |

The FastWAM and DiT4DiT baseline checkpoints come from their authors
([`yuanty/fastwam`](https://huggingface.co/yuanty/fastwam),
[`mondo-robotics/dit4dit-model`](https://huggingface.co/mondo-robotics/dit4dit-model));
[`MODEL_ZOO.md`](MODEL_ZOO.md#baseline-checkpoints-from-their-authors) names the
files each launcher takes. Run the `hf download` commands from an environment
with a current `huggingface_hub` (`pip install -U huggingface_hub`); releases
before 0.34 spell the command `huggingface-cli download`.

## Requirements

- Linux with ROCm on AMD Instinct™ GPUs, or a Radeon™ GPU for the CPU-MuJoCo
  path. `--sim instinct` needs an MI350X/MI355X (gfx950); on MI300X/MI325X
  (gfx942) and other GPUs use `--sim mujoco` (the default).
- A ROCm PyTorch environment per model. Keep the installed ROCm PyTorch: install
  the vendored trees with `pip install -e . --no-deps`, never with the
  CUDA-pinned dependencies from `FastWAM/pyproject.toml`.
- FastWAM: the ROCm evaluation container (`EVAL_IMAGE`, built from
  [`docker/Dockerfile.eval`](docker/Dockerfile.eval):
  `docker build -f docker/Dockerfile.eval -t fam-eval:rocm7.14 .`) and a
  LIBERO checkout at `8f1084e` (`LIBERO_ROOT`) with `mujoco==3.3.2`. FastWAM is
  evaluated with its own evaluator through `eval/native/run_fastwam_libero.sh`. Setup:
  [the evaluation image](eval/native/README.md#the-evaluation-image),
  [FastWAM GEAK guide](docs/fastwam-geak-inference-optimization.md).
- DiT4DiT: separate policy and simulator Python environments (LIBERO), and the
  pinned RoboCasa-GR1 stack from the [CAE-DiT4DiT guide](docs/cae-dit4dit-robocasa.md)
  (robosuite 1.5.1 overlay, deterministic-reset patch), built inside the
  evaluation image by `DiT4DiT/environments/setup_dit4dit_rocm_envs.sh`; the
  DiT4DiT launchers run in that image through `scripts/in_eval_image.sh`
  ([setup](docs/side-by-side-source-trees.md#dit4dit-python-environments-on-rocm)). Every DiT4DiT run also
  needs `nvidia/Cosmos-Predict2.5-2B`, which is gated: accept its licence on
  Hugging Face and log in before downloading it
  ([steps](docs/cae-dit4dit-robocasa.md#cosmos-predict25-2b)).
- `--sim instinct`: build the GPU simulator with `simulate/build.sh libero|robocasa`
  and set `INSTINCT_SIM_BUILD` ([`simulate/README.md`](simulate/README.md)).
  Each launcher's `--help` lists the environment variables it needs.

## Quick start

```bash
git clone https://github.com/AMD-AGI/faster-action-models.git
cd faster-action-models

# Print the resolved run without starting anything (needs no checkpoint).
bash eval/native/run_fastwam_libero.sh --arm cae --steps 4 --smoke --gpus 0 --dry-run

# One rollout per task, once MODEL_CKPT, ACTION_VAE_CKPT, TASK_CONFIG,
# DATASET_STATS_TEMPLATE, LIBERO_ROOT and EVAL_IMAGE are set.
bash eval/native/run_fastwam_libero.sh --arm cae --steps 4 --smoke --gpus 0
```

## Evaluation

| Launcher | Model, benchmark | Arms |
|---|---|---|
| `eval/native/run_fastwam_libero.sh` | FastWAM, LIBERO (FastWAM's own evaluator; [`eval/README.md`](eval/README.md)) | `baseline`, `geak`, `cae` |
| `scripts/eval_fastwam_libero.sh` | FastWAM, LIBERO (a wrapper over `eval/native/run_fastwam_libero.sh` that defaults to one GPU) | `baseline`, `geak`, `cae` |
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

- Shared flags: `--arm`, `--steps`, `--sim mujoco|instinct`, `--gpus`,
  `--smoke`/`--full`/`--trials`, `--suites` (LIBERO), `--output-dir`,
  `--clients-per-gpu` and `--port` (launchers that start a policy server),
  `--record-video`, `--dry-run`. Details:
  [common flags](docs/side-by-side-source-trees.md#common-flags).
- `--sim mujoco` (default) runs physics on the CPU; `--sim instinct` runs physics
  and camera rendering GPU-native on AMD Instinct™ MI350X/MI355X GPUs (gfx950 only).
- MuJoCo renders camera images with GPU OpenGL (EGL) when available and falls
  back to OSMesa; set `MUJOCO_GL` to force one.
- Every run writes `<output-dir>/summary.json` and prints one `eval complete: ...`
  line. A rerun into the same `--output-dir` resumes the same run only.
- Protocol: [`eval/PROTOCOL.md`](eval/PROTOCOL.md). Command per configuration:
  [`docs/side-by-side-source-trees.md`](docs/side-by-side-source-trees.md).
  Rollout videos: [`docs/rollout-videos.md`](docs/rollout-videos.md).

## Reproducing the reference results

```bash
# Baselines: FastWAM and DiT4DiT on LIBERO, DiT4DiT on RoboCasa-GR1
bash scripts/reproduce_baselines.sh

# Training-free optimization: baseline, GEAK, and GEAK with fewer denoising steps
bash scripts/reproduce_training_free.sh

# CAE-FastWAM on LIBERO, next to the FastWAM baseline and GEAK
bash scripts/reproduce_cae_fastwam.sh

# CAE-DiT4DiT on RoboCasa-GR1 full-24 (needs a checkpoint you train; see Training)
bash scripts/reproduce_cae_dit4dit.sh

# Every documented launch: each launcher, arm and step count
bash scripts/run_all_evals.sh
```

Each script runs its configurations once on CPU MuJoCo, with the
`--sim instinct` twin of each run commented out beneath it, and writes results
under `REPRO_DIR` (default `eval/outputs/<script name>`). `--help` prints the
checkpoints and environment variables each script needs and the approximate
runtime (a full FastWAM LIBERO run takes about 12-20 h on one GPU on CPU MuJoCo,
5-10 h with `--sim instinct`, and `EVAL_GPUS=0,1,2,3` spreads it over four GPUs;
a DiT4DiT LIBERO run uses the first of them, a RoboCasa-GR1 full-24 run one
policy server per listed GPU and takes about 7 h; the runs execute one after
another). Start the scripts on the host, with docker: the FastWAM launcher and
the latency tools start `EVAL_IMAGE` containers themselves, and each DiT4DiT
launch runs inside `EVAL_IMAGE` through
[`scripts/in_eval_image.sh`](scripts/in_eval_image.sh), with the environments
from [DiT4DiT Python environments on ROCm](docs/side-by-side-source-trees.md#dit4dit-python-environments-on-rocm).
`--dry-run` prints every resolved run without assets, `--check` only checks the
environment (every script checks it before the first run starts), `--smoke` runs
every configuration end to end with the real checkpoints at one trial per task
(and each latency tool for one round) under `eval/outputs/<script name>-smoke`
unless `REPRO_DIR` is set, and
`--only REGEX` keeps the launches whose command line matches (for example
`bash scripts/reproduce_cae_fastwam.sh --only 'arm cae|cae-fastwam'` runs only
the CAE-FastWAM cells and latency). The reference MI355X results were measured
with `--sim instinct`, as means over several full runs.

## Latency tools

Each tool times its arms interleaved in one session on one GPU (GEAK campaign
setup: [`geak/README.md`](geak/README.md)):

```bash
python3 scripts/compare_artifacts.py --arms eager,compile,geak,geak@2 --rounds 5   # FastWAM
python3 scripts/cae_fastwam_latency.py --arms cae@10,cae@4 --rounds 5             # CAE-FastWAM
python3 scripts/dit4dit_latency.py --bench libero --dry-run                       # DiT4DiT (libero|robocasa)
```

They run on the host and start their own `EVAL_IMAGE` containers. Besides
`EVAL_IMAGE`, `compare_artifacts.py` needs `MODEL_SRC_ROOT` (`FastWAM/`),
`MODEL_CKPT`, `DATASET_STATS`, an existing `HF_CACHE_DIR` and `MODEL_PYTHON`
(`/opt/venv/bin/python3`); `cae_fastwam_latency.py` needs `MODEL_CKPT` and
`ACTION_VAE_CKPT`; `dit4dit_latency.py` needs the checkpoint
of its `--bench`, `DIT4DIT_DIFFUSERS_SRC` and a `DIT4DIT_PYOVERLAY` with the
pinned `transformers`/`tokenizers` (its RECIPE). Each tool's `--help` lists them.

## Training

```bash
python3 scripts/build_fastwam_action_cache.py --help   # CAE-FastWAM action cache from public data
bash scripts/train_fastwam_libero_action_vae.sh
bash scripts/train_fastwam_libero_latent_policy.sh
bash scripts/train_dit4dit_robocasa_action_vae.sh
bash scripts/train_dit4dit_robocasa_latent_policy.sh
```

Each wrapper packages the published hyperparameters, calls the model's native
trainer, and writes a GNU `time -v` resource report when `/usr/bin/time` exists.
`--help` lists inputs and hardware defaults. All training data is public:
[CAE-FastWAM training data](docs/cae-fastwam-libero.md#training-data) builds
the action cache from `yuanty/LIBERO-fastwam`, and the CAE-DiT4DiT guide
builds its cache from GR00T-X-Embodiment-Sim. Guides:
[CAE-FastWAM](docs/cae-fastwam-libero.md),
[CAE-DiT4DiT](docs/cae-dit4dit-robocasa.md).

## Repository layout

```text
FastWAM/, DiT4DiT/   vendored upstream code with the CAE models and trainers
src/                 side-by-side inference trees: fastwam-{baseline,geak}, cae-fastwam, dit4dit-{baseline,geak}
geak/                GEAK task setup for an action model
eval/                protocol, native FastWAM evaluator, summarizers
simulate/            GPU-native simulator for --sim instinct
scripts/             launchers, latency tools, training and reproduction scripts
docs/                guides; model_cards/ holds the Hugging Face model card
```

## Limitations

- Research checkpoints evaluated in simulation only (LIBERO, RoboCasa-GR1).
- Not validated on physical robots; not for safety-critical use.
- Not bit-reproducible across GPU models and software versions; reruns can differ.

## Citation

If you use this repository, please cite the ROCm blog and the upstream models:

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
