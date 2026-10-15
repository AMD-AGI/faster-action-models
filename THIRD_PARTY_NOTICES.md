# Third-party notices

Code authored by Advanced Micro Devices, Inc. in this repository is licensed
under the Apache License, Version 2.0 (see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE)). This repository also contains third-party code, which
keeps its original license, and it instructs users to download third-party
models, datasets, and simulator assets, which are **not** redistributed here.

## 1. Third-party code in this repository

| Component | Location | Upstream | License | License text |
|---|---|---|---|---|
| FastWAM | `FastWAM/`, `src/fastwam-*/`, `src/cae-fastwam/` | [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM) at `7faa711` (FastWAM baseline and GEAK trees: an earlier upstream revision) | MIT, © 2026 The FastWAM Authors | [`FastWAM/LICENSE`](FastWAM/LICENSE), copied into each `src/fastwam-*/LICENSE` and `src/cae-fastwam/LICENSE` |
| DiT4DiT | `DiT4DiT/`, `src/dit4dit-*/` | [Mondo-Robotics/DiT4DiT](https://github.com/Mondo-Robotics/DiT4DiT) at `66a6f3a` | MIT, © 2025 Teli Ma & Mondo Robotics | [`DiT4DiT/LICENSE`](DiT4DiT/LICENSE), `src/dit4dit-*/LICENSE` |
| Wan2.x video DiT and VAE modules, via DiffSynth-Studio (`wan_video_dit.py`, `wan_video_vae.py`, `wan_video_text_encoder.py` in the FastWAM trees) | `FastWAM/src/fastwam/models/wan22/`, `src/fastwam-*/fastwam/models/wan22/`, `src/cae-fastwam/fastwam/models/wan22/` | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2), [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) | Apache-2.0 | [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt) |
| LeRobot dataset utilities (files carrying HuggingFace Apache-2.0 headers) | `FastWAM/src/fastwam/datasets/lerobot*/`, `src/fastwam-*/fastwam/datasets/lerobot*/`, `src/cae-fastwam/fastwam/datasets/lerobot*/` | [huggingface/lerobot](https://github.com/huggingface/lerobot) | Apache-2.0 | [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt) |
| NVIDIA GR00T data loaders, flow-matching head, and RoboCasa wrappers (files carrying NVIDIA Apache-2.0 headers) | `DiT4DiT/DiT4DiT/dataloader/gr00t_lerobot/`, `DiT4DiT/DiT4DiT/model/modules/action_model/flow_matching_head/`, `DiT4DiT/examples/Robocasa_tabletop/eval_files/`, and the same paths in `src/dit4dit-*/` | NVIDIA Isaac GR00T | Apache-2.0 | [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt) |
| NVIDIA utility (`fastwam/utils/misc.py`, SPDX Apache-2.0 header) | FastWAM trees | NVIDIA | Apache-2.0 | [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt) |
| diffusers patch (modifies an installed `diffusers` package; not vendored source) | `src/dit4dit-geak/patches/` | [huggingface/diffusers](https://github.com/huggingface/diffusers) | Apache-2.0 | [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt) |
| PyTorch3D rotation utilities (`transforms/rotation.py` with a Meta BSD header, and `utils/rotation.py` "adapted from pytorch3d") | `FastWAM/src/fastwam/datasets/lerobot/`, `src/fastwam-*/fastwam/datasets/lerobot/`, `src/cae-fastwam/fastwam/datasets/lerobot/` | [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d) | BSD-3-Clause, © Meta Platforms, Inc. and affiliates | [`licenses/BSD-3-Clause-PyTorch3D.txt`](licenses/BSD-3-Clause-PyTorch3D.txt) |
| openpi websocket policy server (`websocket_policy_server.py`, and the handler copied from it in `scripts/dit4dit_serve_keepalive.py`) | `DiT4DiT/deployment/model_server/tools/`, `src/dit4dit-*/deployment/model_server/tools/`, `scripts/` | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi), via DiT4DiT | Apache-2.0 | [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt) |
| starVLA action-head adaptations (credited in the `ActionDiT.py` header, which also notes action repeat is inspired by CogACT) | `DiT4DiT/DiT4DiT/model/modules/action_model/ActionDiT.py`, `src/dit4dit-*/DiT4DiT/model/modules/action_model/ActionDiT.py` | [starVLA/starVLA](https://github.com/starVLA/starVLA), via DiT4DiT | MIT (the file itself keeps its NVIDIA Apache-2.0 header) | [`DiT4DiT/LICENSE`](DiT4DiT/LICENSE), [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt) |
| MuJoCo collision routines (box-box `engine_collision_box.c`, GJK/EPA `engine_collision_gjk.c` / `engine_collision_convex.c`, `engine_util_blas.c`), transcribed to HIP | `simulate/src/kernels/physics_collision.hip`, `simulate/include/hipgl/physics/physics_mjccd.h` | [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) 3.9.0 | Apache-2.0, © 2021 DeepMind Technologies Limited | [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt) |
| MuJoCo Warp ray/primitive intersection routines (`ray.py`, `render.py`), ported to C++/HIP | `simulate/src/kernels/rt_intersect.inc` | [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp) | Apache-2.0, © 2025 The Newton Developers | [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt) |
| TinyXML-2 | `simulate/third_party/tinyxml2.*` | [leethomason/tinyxml2](https://github.com/leethomason/tinyxml2) | zlib, Lee Thomason | header of `simulate/third_party/tinyxml2.h` |

Files that carry their own license header keep it. Where AMD modified a
third-party file, a "Modifications Copyright (c) 2026 Advanced Micro Devices,
Inc." or "Modified by Advanced Micro Devices, Inc., 2026" line is added below the
original header.

Upstream subdirectories that this project does not use (RoboTwin, the
GR00T-WholeBodyControl `decoupled_wbc/` stack, and the XR-Robotics Pico runtime
packages) are not included. See [`FastWAM/README.vendor.md`](FastWAM/README.vendor.md)
and [`DiT4DiT/README.vendor.md`](DiT4DiT/README.vendor.md).

## 2. Third-party assets downloaded by users (not redistributed)

Users obtain these directly from their publishers and must comply with their
terms.

| Asset | Used for | Source | License |
|---|---|---|---|
| Wan2.2-TI2V-5B (video DiT, VAE, umt5-xxl text encoder) | FastWAM base model and runtime assets | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | Apache-2.0 |
| Wan2.1-T2V-1.3B / `google/umt5-xxl` tokenizer | FastWAM tokenizer | [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B), [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) | Apache-2.0 |
| Cosmos-Predict2.5-2B | DiT4DiT video backbone | [nvidia/Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) (gated) | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| LIBERO datasets | FastWAM training and evaluation | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | CC BY 4.0 (code: MIT) |
| LIBERO in LeRobot format | FastWAM training | [yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam) | CC BY 4.0 |
| PhysicalAI-Robotics-GR00T-X-Embodiment-Sim | DiT4DiT training | [nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) | CC BY 4.0 |
| RoboCasa kitchen assets | RoboCasa simulation | [robocasa/robocasa](https://github.com/robocasa/robocasa) (`download_kitchen_assets.py`) | CC BY 4.0 per upstream; includes Objaverse objects with per-object licenses (code: MIT) |
| RoboCasa-GR1 tabletop assets | RoboCasa-GR1 evaluation | [nvidia/PhysicalAI-DigitalCousin-Assets](https://huggingface.co/datasets/nvidia/PhysicalAI-DigitalCousin-Assets) | **CC BY-NC 4.0 (non-commercial)** (task code [robocasa-gr1-tabletop-tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks): MIT) |
| robosuite | Simulation framework | [ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite) | MIT |
| MuJoCo | Physics engine | [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) | Apache-2.0 |
| vla-evaluation-harness | FastWAM LIBERO closed-loop evaluation | [allenai/vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness) | Apache-2.0 |
| GEAK | GEAK kernel optimization | [AMD-AGI/GEAK](https://github.com/AMD-AGI/GEAK) | Apache-2.0 |

## 3. Model checkpoints

One fine-tuned checkpoint, CAE-FastWAM (LIBERO), is distributed separately from
this repository in the
[Compact Action Expert (CAE) Hugging Face collection](https://huggingface.co/collections/amd/compact-action-expert-cae).
It is licensed under Apache-2.0, as a modified version of Wan2.2-TI2V-5B, and
its Hugging Face repository ships the license, notice, and model card from
[`model_cards/fastwam-libero/`](model_cards/fastwam-libero/).

No CAE-DiT4DiT checkpoint is distributed. CAE-DiT4DiT training and evaluation
require the Cosmos-Predict2.5-2B backbone listed in section 2, which users
download from NVIDIA under its own license.
