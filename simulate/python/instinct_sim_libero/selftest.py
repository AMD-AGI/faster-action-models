# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quick check of the GPU LIBERO env against LIBERO on CPU MuJoCo (no policy needed).

    python3 -m instinct_sim_libero.selftest [--suite libero_spatial --task 0 --steps 60 --num-envs 4]

Runs the same open-loop actions (30 no-op warm-up steps, then a smooth arm motion) from the same init state:
  1. LIBERO's OffScreenRenderEnv (CPU MuJoCo) vs HipOffScreenRenderEnv: end-effector position gap and image MAE;
  2. HipLiberoVectorEnv with --num-envs worlds stepping identical episodes: every world must match world 0.
Both engines integrate the same contact-rich dynamics with different floating-point order, so small gaps
(sub-millimetre eef, a few grey levels) are expected; a broken setup shows centimetres or a black/garbled image.
"""
import argparse
import pathlib

import numpy as np


def actions(n_wait, n):
    out = [[0, 0, 0, 0, 0, 0, -1]] * n_wait
    for i in range(n):
        out.append([0.3 * np.sin(0.1 * i), 0.3 * np.cos(0.13 * i), -0.2, 0, 0, 0.1 * np.sin(0.05 * i), -1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import instinct_sim_libero
    from instinct_sim_libero.env import HipOffScreenRenderEnv, _original_cls
    from libero.libero import benchmark, get_libero_path
    import torch

    task = benchmark.get_benchmark_dict()[a.suite]().get_task(a.task)
    init = torch.load(pathlib.Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                      weights_only=False)[0]
    env_args = {"bddl_file_name": str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder /
                                      task.bddl_file), "camera_heights": 256, "camera_widths": 256}
    acts = actions(30, a.steps)

    def rollout(env):
        env.seed(a.seed)
        env.reset()
        obs = env.set_init_state(init)
        for act in acts:
            obs, _, _, _ = env.step(act)
        return obs

    cpu = _original_cls()(**env_args)
    o_cpu = rollout(cpu)
    cpu.close()
    hip = HipOffScreenRenderEnv(**env_args)
    o_hip = rollout(hip)
    hip.close()
    eef = float(np.abs(o_cpu["robot0_eef_pos"] - o_hip["robot0_eef_pos"]).max())
    mae = {k: float(np.abs(o_cpu[k].astype(np.float64) - o_hip[k].astype(np.float64)).mean())
           for k in ("agentview_image", "robot0_eye_in_hand_image")}
    print(f"[selftest] {a.suite} t{a.task}: CPU vs hip after {len(acts)} steps: eef max|d| {eef * 1000:.3f} mm, "
          f"image MAE agent {mae['agentview_image']:.2f} wrist {mae['robot0_eye_in_hand_image']:.2f} (0-255)")

    venv = instinct_sim_libero.HipLiberoVectorEnv(env_args, num_envs=a.num_envs, seed=a.seed)
    obs = venv.reset([init] * a.num_envs)
    for act in acts:
        out = venv.step({k: act for k in range(a.num_envs)})
        obs = [out[k][0] for k in range(a.num_envs)]
    for k in range(a.num_envs):
        venv.finish(k)
    venv.close()
    spread = max(float(np.abs(o["robot0_eef_pos"] - obs[0]["robot0_eef_pos"]).max()) for o in obs)
    vs_single = float(np.abs(obs[0]["robot0_eef_pos"] - o_hip["robot0_eef_pos"]).max())
    print(f"[selftest] vector env, {a.num_envs} worlds: eef spread across worlds {spread * 1000:.4f} mm, "
          f"world 0 vs single-world {vs_single * 1000:.4f} mm")
    ok = eef < 0.01 and max(mae.values()) < 20 and spread < 1e-3
    print("[selftest] PASS" if ok else "[selftest] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
