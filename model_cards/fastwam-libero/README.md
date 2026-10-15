---
license: apache-2.0
base_model: Wan-AI/Wan2.2-TI2V-5B
datasets:
  - yuanty/LIBERO-fastwam
tags:
  - robotics
  - world-action-model
  - libero
---

# CAE-FastWAM LIBERO

A world-action-model policy for the LIBERO benchmark, fine-tuned by AMD from
Wan2.2-TI2V-5B with the [FastWAM](https://github.com/yuantianyuan01/FastWAM)
recipe. It accompanies the
[`faster-action-models`](https://github.com/AMD-AGI/faster-action-models)
repository; see its
[`MODEL_ZOO.md`](https://github.com/AMD-AGI/faster-action-models/blob/main/MODEL_ZOO.md)
for paths, hashes, and download instructions:

```bash
huggingface-cli download amd/CAE-FastWAM --local-dir .
```

Part of the [Compact Action Expert (CAE) collection](https://huggingface.co/collections/amd/compact-action-expert-cae);
the full write-up and results are in the
[blog post](https://rocm.blogs.amd.com/artificial-intelligence/faster-action-models/README.html).

## Model

CAE-FastWAM (Compact Action Expert: the 32-hidden-dimension Compact Action DiT with the
K8 x d16 Action VAE, GEAK-optimized) is one model with two parts:

| Component | Description |
|---|---|
| Compact Action DiT | Hidden size reduced from 1024 to 32 (about 30M action-expert parameters); denoises latent action chunks |
| K8 x d16 Action VAE | Suite-conditioned; encodes each 32 x 7 action chunk as 8 latent tokens of width 16 |

The evaluation protocol is described in
[`docs/cae-fastwam-libero.md`](https://github.com/AMD-AGI/faster-action-models/blob/main/docs/cae-fastwam-libero.md).

## Intended use

Research on world-action models and robot-policy inference efficiency, in
simulation.

## Bias, risks, and limitations

- Evaluated only in simulation (LIBERO). Not validated on physical robots and not
  intended for safety-critical, medical, or other high-stakes applications.
- Success rates may not transfer to real-world settings. Users are responsible
  for evaluating the policy and adding appropriate safety mechanisms.
- Do not use this model or its outputs in ways that violate applicable laws,
  cause harm, spread misinformation, or target vulnerable populations, consistent
  with the Wan2.2 model card.

## License and attribution

- License: Apache License 2.0 ([`LICENSE`](LICENSE)). See [`NOTICE`](NOTICE).
- Contains a modified version of [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
  (Apache-2.0), fine-tuned by Advanced Micro Devices, Inc. Runtime assets
  (Wan2.2 VAE, umt5-xxl encoder and tokenizer) are downloaded from Hugging Face
  and are not redistributed with these checkpoints.
- Training data: [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
  (CC BY 4.0) in the LeRobot format published as
  [yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
  (CC BY 4.0).

## Citation

Please cite FastWAM and LIBERO:

- T. Yuan, Z. Dong, Y. Liu, and H. Zhao, "Fast-WAM: Do World Action Models Need
  Test-time Future Imagination?" arXiv:2603.16666, 2026.
- B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone, "LIBERO:
  Benchmarking Knowledge Transfer for Lifelong Robot Learning," NeurIPS 2023.
