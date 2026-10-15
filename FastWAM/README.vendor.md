# Vendored FastWAM source

This directory is imported from
[yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM) at commit
`7faa711`. The imported upstream project is licensed under the MIT License; see
[`LICENSE`](LICENSE). Files that carry their own license header (for example the
LeRobot-derived dataset utilities and NVIDIA utilities, Apache-2.0) keep it.

## Local changes

The Compact Action Expert changes add CAE-FastWAM (Compact Action Expert: the
32-hidden-dimension Action DiT with the K8 x d16 action VAE; its
GEAK-optimized inference tree is `../src/cae-fastwam`): the Action VAE
model and trainer, per-suite normalization, and its LIBERO evaluator. Files
added or modified by Advanced Micro Devices, Inc. carry an AMD copyright line
in their header.

Added:

- `scripts/train_action_ae.py`
- `scripts/eval_libero_aac.py`
- `src/fastwam/models/action_ae.py`
- CAE-FastWAM task and model configs under `configs/`
- `configs_baseline/`, the config tree of the FastWAM baseline and GEAK source
  trees under `../src/`

Modified:

- `experiments/libero/eval_libero_single.py`
- `scripts/preprocess_action_dit_backbone.py`
- `src/fastwam/datasets/lerobot/processors/fastwam_processor.py`
- `src/fastwam/datasets/lerobot/robot_video_dataset.py`
- `src/fastwam/datasets/lerobot/utils/normalizer.py`
- `src/fastwam/models/wan22/fastwam.py`
- `src/fastwam/runtime.py`

## Divergence from upstream: RoboTwin removed

Upstream vendors the RoboTwin benchmark under `third_party/RoboTwin/` and ships
RoboTwin evaluation entry points and configs. This project evaluates on LIBERO
only, so the following are **not included**:

- `third_party/RoboTwin/`
- `experiments/robotwin/`
- `configs/sim_robotwin.yaml`, `configs/data/robotwin.yaml`, and
  `configs/task/robotwin_*.yaml`

The `"robotwin"` camera-layout option in `robot_video_dataset.py` is left in
place. To run the RoboTwin benchmark, use the upstream repository.

Model weights, datasets, and the Wan2.2 runtime assets are not vendored. See
[`../MODEL_ZOO.md`](../MODEL_ZOO.md) and
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
