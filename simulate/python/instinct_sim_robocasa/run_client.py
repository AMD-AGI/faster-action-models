# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the DiT4DiT RoboCasa sim client (simulation_env.py) with physics on the GPU simulator.

    python3 -m instinct_sim_robocasa.run_client <path/to/simulation_env.py> [its --args.* flags]

Nothing in the client changes: robocasa is imported under MuJoCo 3.3.2 (compat), and every env the client
builds through gymnasium.make gets the GPU-native backend attached (backend.attach). HP_GPU_ID selects the
device (container-visible index, default 0); HP_PHYSICS=cpu runs the same launcher on CPU MuJoCo.
HP_SEED=<int> makes episode i's scene a function of (HP_SEED, i) only (paired CPU-vs-hip runs, see seed_episodes).
Harnesses that start the client as a subprocess use ``sitecustomize/`` with HP_ROBOCASA_BACKEND=hip instead.
"""
import os
import runpy
import sys


def seed_episodes(env, base):
    """Episode i (0-based count of resets) samples its scene from seed base + i: robosuite's env rng (layout,
    objects, placements) and the global RNGs are reseeded right before each reset. The client's own resets pass
    seed=None, which RoboCasaEnv forwards to np.random.seed (OS entropy), so the seed is substituted there."""
    import random
    import numpy as np
    genv = env.unwrapped
    orig_reset = genv.reset
    count = [0]

    def reset(seed=None, options=None):
        s = base + count[0]
        count[0] += 1
        random.seed(s)
        genv.env.rng = np.random.default_rng(s)
        print(f"[run_client] episode reset {count[0] - 1} seed {s}", flush=True)
        return orig_reset(seed=s, options=options)

    genv.reset = reset


def record_episodes(env, out_dir):
    """HP_RECORD_DIR: per episode, the robosuite pre-step state (qpos, qvel, qacc_warmstart, act) and the action
    dict of every env step, saved as <out_dir>/<pid>_ep<k>_seed<s>.npz at the next reset (for offline replays)."""
    import numpy as np
    genv = env.unwrapped
    os.makedirs(out_dir, exist_ok=True)
    buf = {"states": [], "actions": [], "success": [], "seed": None, "k": 0}
    orig_step, orig_reset = genv.step, genv.reset

    def flush():
        if buf["actions"]:
            st = np.array(buf["states"]); keys = sorted(buf["actions"][0])
            np.savez_compressed(os.path.join(out_dir, f"{os.getpid()}_ep{buf['k']}_seed{buf['seed']}.npz"),
                                states=st, success=np.array(buf["success"]), seed=buf["seed"],
                                **{"act_" + k: np.array([a[k] for a in buf["actions"]]) for k in keys})
            buf["k"] += 1
        buf["states"].clear(); buf["actions"].clear(); buf["success"].clear()

    def step(action):
        d = genv.env.sim.data._data
        buf["states"].append(np.concatenate([d.qpos, d.qvel, d.qacc_warmstart, d.act]))
        buf["actions"].append({k: np.array(v, copy=True) for k, v in action.items()})
        r = orig_step(action)
        buf["success"].append(bool(r[-1].get("success", False)) if isinstance(r[-1], dict) else False)
        return r

    def reset(seed=None, options=None):
        flush()
        buf["seed"] = seed
        return orig_reset(seed=seed, options=options)

    genv.step, genv.reset = step, reset


def install():
    """Attach the GPU-native backend (HP_PHYSICS != cpu) to every robocasa env built through gymnasium.make.
    robocasa must already be imported under compat.import_robocasa. Idempotent."""
    import gymnasium as gym
    if getattr(gym.make, "_hp_installed", False):
        return
    if os.environ.get("HP_SUCCESS_DEBUG") == "1":
        from .success_debug import install as install_success_debug
        install_success_debug()
    hip = os.environ.get("HP_PHYSICS", "hip") != "cpu"
    seed = os.environ.get("HP_SEED")
    if hip or seed is not None:
        gpu = int(os.environ.get("HP_GPU_ID", "0"))
        orig_make = gym.make

        def make(*a, **k):
            if seed is not None:
                import random
                import numpy as np
                random.seed(int(seed)); np.random.seed(int(seed))   # construction-time scene sampling
                k.setdefault("seed", int(seed))
            env = orig_make(*a, **k)
            if "robocasa" in str(type(env.unwrapped).__module__) or hasattr(env.unwrapped, "env"):
                if os.environ.get("HP_RECORD_DIR"):
                    record_episodes(env, os.environ["HP_RECORD_DIR"])   # inner: sees the seeded reset's seed
                if seed is not None:
                    seed_episodes(env, int(seed))
                if hip:
                    from .backend import attach
                    env._hip_backend = attach(env, gpu_id=gpu)
            return env

        make._hp_installed = True
        gym.make = make
    import mujoco
    print(f"[run_client] mujoco {mujoco.__version__}, physics {os.environ.get('HP_PHYSICS', 'hip')}", flush=True)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script] + sys.argv[2:]
    # The GPU simulator follows MuJoCo 3.3.2: INSTINCT_SIM_MUJOCO_PATH wins over any simulator overlay on PYTHONPATH
    mj = os.environ.get("INSTINCT_SIM_MUJOCO_PATH", "").strip()
    if mj and os.environ.get("HP_PHYSICS", "hip") != "cpu":
        sys.path[:] = [mj] + [p for p in sys.path if os.path.abspath(p) != os.path.abspath(mj)]
    from .compat import import_robocasa
    import_robocasa()
    install()
    sys.path.insert(0, os.path.dirname(script))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
