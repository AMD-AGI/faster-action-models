# Vendored DiT4DiT source

This directory is a git subtree of
[Mondo-Robotics/DiT4DiT](https://github.com/Mondo-Robotics/DiT4DiT), imported
from commit `66a6f3a12e2c8740157c1e478795952c040d31dd`.

The CAE-DiT4DiT changes add the RoboCasa full-24 K8/d48 action-VAE policy,
reproducible cache and training tools, optimized MI325X training, deterministic
schema-2 rollout evaluation, and focused tests. The imported upstream project
is licensed under the MIT License; see [`LICENSE`](LICENSE). Files that carry
their own license header (for example the NVIDIA Apache-2.0 GR00T data loaders
and flow-matching head) keep it. Files added or modified by Advanced Micro
Devices, Inc. carry an AMD copyright line in their header.

Datasets, model weights, simulator assets, virtual environments, action caches,
and experiment outputs are not vendored. See
[`../docs/cae-dit4dit-robocasa.md`](../docs/cae-dit4dit-robocasa.md),
[`../MODEL_ZOO.md`](../MODEL_ZOO.md), and
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for their pinned
sources, hashes, and licenses.

## Divergence from upstream: `decoupled_wbc/` removed

Upstream vendors NVIDIA's
[GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
stack under `decoupled_wbc/`. It is used only for real-robot (Unitree G1)
teleoperation and whole-body control, which nothing in this repository uses, and
it carries separately licensed content: Apache-2.0 code, policy checkpoints under
the NVIDIA Open Model License, the BSD-3-Clause Unitree SDK, robot meshes, and
XR-Robotics Pico runtime packages. It is therefore **not included**. The
`examples/Real_G1/` instructions refer to it; obtain it from the upstream
GR00T-WholeBodyControl repository if you need that workflow.
