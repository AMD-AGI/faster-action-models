# Model Zoo

One checkpoint is released: **CAE-FastWAM** on LIBERO,
[`amd/CAE-FastWAM`](https://huggingface.co/amd/CAE-FastWAM), in the
[Compact Action Expert (CAE) Hugging Face collection](https://huggingface.co/collections/amd/compact-action-expert-cae).
The [blog post](https://rocm.blogs.amd.com/artificial-intelligence/faster-action-models/README.html)
has the full write-up and results.

CAE-DiT4DiT (RoboCasa-GR1) is not released as a checkpoint; train it with the
scripts and guide in [`docs/cae-dit4dit-robocasa.md`](docs/cae-dit4dit-robocasa.md).

## License

| Checkpoint | License | Model card |
|---|---|---|
| CAE-FastWAM LIBERO (32-hidden-dimension Action DiT and K8 x d16 Action VAE) | Apache-2.0; contains a modified version of Wan2.2-TI2V-5B | [`model_cards/fastwam-libero/`](model_cards/fastwam-libero/) |

The Hugging Face repository ships the `LICENSE`, `NOTICE`, and `README.md`
from the model-card directory. Third-party models, datasets, and simulator
assets are downloaded from their publishers and are not redistributed; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Download

The checkpoint is published as
[`amd/CAE-FastWAM`](https://huggingface.co/amd/CAE-FastWAM) in the
[Compact Action Expert (CAE) collection](https://huggingface.co/collections/amd/compact-action-expert-cae).

Download the repository into the imported FastWAM tree and verify it:

```bash
cd /path/to/faster-action-models/FastWAM
huggingface-cli download amd/CAE-FastWAM --local-dir .
sha256sum -c ASSET_SHA256SUMS
```

The repository preserves the original `checkpoints/` and `runs/` paths, so no
reorganization is required after download.

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

FastWAM checkpoint:

```text
runs/libero_uncond_2cam224_1e-4_32hdim_latent_K8d16_freebits_pernorm_joint/2026-08-06_09-15-14/checkpoints/weights/step_021480.pt
```

Checkpoint SHA-256:

```text
b2cd77e9402b7897ba0d680eb45e4a38fb752e89641e3a9487a7cd4e947dad6c
```

K8 x d16 Action VAE:

```text
checkpoints/cascade/vae_approachB/vae_B4_allsuites_softprompt_pernorm_freebits_K8_dz16.pt
```

VAE SHA-256:

```text
c485bd2ee62dbb1828eea5c5ea2bb7b0178f60d809329ebfd15bdf731f592f9a
```

Use the matching normalization file for each suite:

```text
runs/per_suite_stats/dataset_stats_libero_spatial.json
runs/per_suite_stats/dataset_stats_libero_object.json
runs/per_suite_stats/dataset_stats_libero_goal.json
runs/per_suite_stats/dataset_stats_libero_10.json
```

The shareable evaluator and exact checkpoint completed a one-rollout
LIBERO-Spatial task 0 smoke test successfully on an AMD Instinct™ MI325X GPU with ROCm 7.2.

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
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth \
  --local-dir checkpoints/Wan-AI/Wan2.2-TI2V-5B
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --include "google/umt5-xxl/*" \
  --local-dir checkpoints/Wan-AI/Wan2.1-T2V-1.3B
```

The full Wan video-DiT safetensors are not required for evaluation because
`skip_dit_load_from_pretrain=true`.

Set the runtime environment:

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
dependencies pin CUDA wheels.

Training-only artifacts are intentionally excluded: the pretrained Wan video
DiT, compact initializer, LeRobot datasets, and predecoded action caches.

See [`docs/cae-fastwam-libero.md`](docs/cae-fastwam-libero.md) for
training and evaluation commands.
