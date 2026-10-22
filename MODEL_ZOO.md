# Model Zoo

One checkpoint is released: **CAE-FastWAM** on LIBERO,
[`amd/CAE-FastWAM`](https://huggingface.co/amd/CAE-FastWAM), in the
[Compact Action Expert (CAE) Hugging Face collection](https://huggingface.co/collections/amd/compact-action-expert-cae).
The [ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/faster-action-models/README.html)
has the full reference results.

CAE-DiT4DiT (RoboCasa-GR1) is not released as a checkpoint; train it with the
scripts and guide in [`docs/cae-dit4dit-robocasa.md`](docs/cae-dit4dit-robocasa.md).

## License

| Checkpoint | License | Model card |
|---|---|---|
| CAE-FastWAM LIBERO (32-hidden-dimension Action DiT and K8 x d16 Action VAE) | Apache-2.0; contains a modified version of Wan2.2-TI2V-5B | [`model_cards/fastwam-libero/`](model_cards/fastwam-libero/) |

The Hugging Face repository ships the same `LICENSE` and `NOTICE` as the
model-card directory; [`model_cards/fastwam-libero/README.md`](model_cards/fastwam-libero/README.md)
is a copy of its model card. Third-party models, datasets, and simulator
assets are downloaded from their publishers and are not redistributed; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Download

The checkpoint is published as
[`amd/CAE-FastWAM`](https://huggingface.co/amd/CAE-FastWAM) in the
[Compact Action Expert (CAE) collection](https://huggingface.co/collections/amd/compact-action-expert-cae).

Run the downloads with a current `huggingface_hub` (`pip install -U huggingface_hub`
in any Python environment), which provides the `hf` command. Releases before
0.34 only have the older spelling, `huggingface-cli download`, with the same
arguments. From the repository root, download into a directory of its own (not
into `FastWAM/`, whose `LICENSE` and `README.md` the model card's files would
overwrite) and verify the weights and statistics:

```bash
hf download amd/CAE-FastWAM --local-dir checkpoints/CAE-FastWAM
(cd checkpoints/CAE-FastWAM && grep -E '\.(pt|json)$' ASSET_SHA256SUMS | sha256sum -c -)
```

The repository is flat:

| File | Description | Launcher variable |
|---|---|---|
| `policy.pt` | Inference weights for the video and action experts | `MODEL_CKPT` |
| `action_vae.pt` | Suite-conditioned K8 x d16 Action VAE | `ACTION_VAE_CKPT` |
| `stats/libero_{spatial,object,goal,10}.json` | Per-suite normalization statistics | `DATASET_STATS_TEMPLATE` (`{suite}` placeholder) |
| `videos/*.mp4`, `replay.mp4` | Example rollouts (not needed to run) | |

```bash
export CAE_FASTWAM_DIR="$PWD/checkpoints/CAE-FastWAM"
export MODEL_CKPT="$CAE_FASTWAM_DIR/policy.pt"
export ACTION_VAE_CKPT="$CAE_FASTWAM_DIR/action_vae.pt"
export DATASET_STATS_TEMPLATE="$CAE_FASTWAM_DIR/stats/{suite}.json"
export TASK_CONFIG=libero_uncond_2cam224_1e-4_32hdim_latent_K8d16_freebits_pernorm_joint
```

`scripts/reproduce_cae_fastwam.sh` and `scripts/run_all_evals.sh` take the same
files as `CAE_FASTWAM_CKPT`, `CAE_FASTWAM_ACTION_VAE_CKPT`,
`CAE_FASTWAM_STATS_TEMPLATE` and `CAE_FASTWAM_TASK_CONFIG`.

## CAE-FastWAM

CAE-FastWAM is FastWAM with the Compact Action Expert (CAE), GEAK-optimized. The
CAE combines two changes: an Action DiT whose hidden dimension is reduced from
1024 to 32 (about 30M action-expert parameters), and a suite-conditioned
variational Action VAE that compresses each `[32, 7]` action chunk into 8 tokens
with 16 dimensions per token (K8 x d16). The Action DiT denoises in that latent
space.

- Evaluation: the four LIBERO suites, 50 episodes per task.
- Action scheduler: `train_shift=5.0`, `infer_shift=5.0`.
- Config:
  `FastWAM/configs/task/libero_uncond_2cam224_1e-4_32hdim_latent_K8d16_freebits_pernorm_joint.yaml`.

| File | SHA-256 |
|---|---|
| `policy.pt` | `b2cd77e9402b7897ba0d680eb45e4a38fb752e89641e3a9487a7cd4e947dad6c` |
| `action_vae.pt` | `c485bd2ee62dbb1828eea5c5ea2bb7b0178f60d809329ebfd15bdf731f592f9a` |

Use the matching normalization file for each suite: `stats/libero_spatial.json`,
`stats/libero_object.json`, `stats/libero_goal.json`, `stats/libero_10.json`.
The launchers pick it per suite from `DATASET_STATS_TEMPLATE`.

The shareable evaluator and exact checkpoint completed a one-rollout
LIBERO-Spatial task 0 smoke test successfully on an AMD Instinct™ MI325X GPU with ROCm 7.2.

## Baseline checkpoints (from their authors)

The baseline and GEAK arms run the authors' published checkpoints unchanged:

| Model | Download | Launcher variables |
|---|---|---|
| FastWAM, LIBERO | `hf download yuanty/fastwam libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json --local-dir checkpoints/fastwam` | `MODEL_CKPT` (or `FASTWAM_CKPT`) = `checkpoints/fastwam/libero_uncond_2cam224.pt`, `DATASET_STATS` (or `FASTWAM_DATASET_STATS`) = `checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json` |
| DiT4DiT, LIBERO | `hf download mondo-robotics/dit4dit-model --include "dit4dit_libero/*" --local-dir checkpoints/dit4dit-model` | `DIT4DIT_LIBERO_CKPT` = `checkpoints/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt` |
| DiT4DiT, RoboCasa-GR1 | `hf download mondo-robotics/dit4dit-model --include "dit4dit_robocasa_gr1/*" --local-dir checkpoints/dit4dit-model` | `DIT4DIT_ROBOCASA_CKPT` = `checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt` |

Each DiT4DiT checkpoint reads its `config.yaml` and `dataset_statistics.json`
from the directory above `final_model/`. Both DiT4DiT checkpoints also need the
gated `nvidia/Cosmos-Predict2.5-2B` backbone; see
[Cosmos-Predict2.5-2B](docs/cae-dit4dit-robocasa.md#cosmos-predict25-2b).

## Shared runtime assets

The checkpoint restores the trained video and action experts itself.
Evaluation additionally requires only these Wan assets (Apache-2.0), which are
downloaded from Hugging Face rather than redistributed:

```text
checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
checkpoints/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
```

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth \
  --local-dir checkpoints/Wan-AI/Wan2.2-TI2V-5B
hf download Wan-AI/Wan2.1-T2V-1.3B \
  --include "google/umt5-xxl/*" \
  --local-dir checkpoints/Wan-AI/Wan2.1-T2V-1.3B
```

The full Wan video-DiT safetensors are not required for evaluation because
`skip_dit_load_from_pretrain=true`. Every evaluation and timing config loads the
VAE and text encoder from these `Wan-AI/...` files (`redirect_common_files: false`);
no ModelScope download is needed.

Training (the CAE-FastWAM initializer and latent policy,
[docs/cae-fastwam-libero.md](docs/cae-fastwam-libero.md#training-data)) also needs the
Wan2.2-TI2V-5B video DiT, about 20 GB, in the same directory:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  config.json diffusion_pytorch_model.safetensors.index.json \
  diffusion_pytorch_model-00001-of-00003.safetensors \
  diffusion_pytorch_model-00002-of-00003.safetensors \
  diffusion_pytorch_model-00003-of-00003.safetensors \
  --local-dir checkpoints/Wan-AI/Wan2.2-TI2V-5B
```

Without them the loader stops with
`Cannot detect model type for wan_video_dit. File: []`.

Set the runtime environment (from the directory that holds `checkpoints/`, the
repository root if you followed the commands above):

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export TOKENIZERS_PARALLELISM=false
```

Leave `MUJOCO_GL` unset: the evaluation launchers render MuJoCo's camera
images with GPU OpenGL (EGL) when it is available and fall back to OSMesa
otherwise. Export `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa` only to force one.

The official LIBERO package, its BDDL/init-state/assets configuration, and
`mujoco==3.3.2` are also required. On ROCm, preserve the existing ROCm PyTorch
installation and use `pip install -e . --no-deps`; FastWAM's development
dependencies pin CUDA wheels. LIBERO itself is imported from `LIBERO_ROOT` on
`PYTHONPATH` (its editable install does not make `import libero` work), and its
first import asks interactive questions unless `$LIBERO_CONFIG_PATH/config.yaml`
exists. [`docker/Dockerfile.eval`](docker/Dockerfile.eval) handles both, and
[`eval/native/README.md`](eval/native/README.md#the-evaluation-image) shows the
config file for a host checkout.

Training-only artifacts are not published: the pretrained Wan video DiT
(download it from `Wan-AI/Wan2.2-TI2V-5B`), the compact initializer, the LeRobot
datasets (`yuanty/LIBERO-fastwam`) and the predecoded action cache. The
[training data section](docs/cae-fastwam-libero.md#training-data) downloads the
datasets and builds the initializer and the action cache from public data.

See [`docs/cae-fastwam-libero.md`](docs/cae-fastwam-libero.md) for
training and evaluation commands.
