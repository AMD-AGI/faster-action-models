#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CAE-FastWAM latency: every cell in ONE interleaved session.

CAE-FastWAM (Compact Action Expert: the 32-hidden-dimension Action DiT with the
K8 x d16 action VAE, GEAK-optimized) is timed from its source tree,
src/cae-fastwam, at the step counts you ask for.

    python3 scripts/cae_fastwam_latency.py --gpus 0
    python3 scripts/cae_fastwam_latency.py --arms cae@10,cae@4 --rounds 3

Each (round, cell) runs `scripts/cae_fastwam_latency_worker.py` in a fresh eval
container with the arm's source tree as PYTHONPATH. The worker checks the cell
before timing it (Sv/ctx/Sa, output shape, executed denoise steps) and times
the whole `infer_action` call. See its
docstring for the input and timing contract.

Latency is not portable across nodes, drivers or sessions, so cells are measured
INTERLEAVED (for round: for cell) and only compared within one run. Nothing is
committed. Re-run it on the machine you care about.

ARMS  (append @<steps> to override --steps for that cell, e.g. cae@4)
  cae        src/cae-fastwam, compile_action_infer=True.

--gpus takes the one HIP device to time on (default: EVAL_GPUS, else 0); the
former --gpu still works with a deprecation warning.
EVAL_DOCKER_ARGS adds `docker run` flags to every timed container, e.g.
EVAL_DOCKER_ARGS="--cpuset-cpus 104-127" pins it to the CPUs of the GPU's NUMA node
(the reference latency runs pinned each container that way, away from other work).

ENVIRONMENT
  required  MODEL_CKPT        CAE-FastWAM checkpoint (.pt)
            ACTION_VAE_CKPT   action-VAE checkpoint the model was trained with
            EVAL_IMAGE        container image with the FastWAM dependencies
  optional  MODEL_CONFIGS     config root (default: <repo>/FastWAM/configs)
            TASK_CONFIG       task config name (default below)
            HF_HOME HF_CACHE_DIR HUGGINGFACE_HUB_CACHE TORCH_HOME XDG_CACHE_HOME
            DIFFSYNTH_MODEL_BASE_PATH DIFFSYNTH_DOWNLOAD_SOURCE
            DIFFSYNTH_SKIP_DOWNLOAD   forwarded to the container when set

The latency cells always pass compile_action_infer explicitly. The LIBERO eval
server does not (it calls infer_action with the default, False), so closed-loop
success rates for CAE-FastWAM come from the uncompiled path. The
two paths compute the same function; only their speed differs.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_flags import extra_docker_args, resolve_one_gpu  # noqa: E402  (scripts/eval_flags.py)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(REPO, "scripts", "cae_fastwam_latency_worker.py")
DEFAULT_TASK = "libero_uncond_2cam224_1e-4_32hdim_latent_K8d16_freebits_pernorm_joint"
DEFAULT_ARMS = "cae@10,cae@4"

ARM_SPECS = {
    # name: (tree under src/, compile_action_infer, description)
    "cae": ("cae-fastwam", True, "CAE-FastWAM (GEAK-optimized)"),
}
FORWARD_PATH_VARS = ("HF_CACHE_DIR", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME",
                     "XDG_CACHE_HOME", "DIFFSYNTH_MODEL_BASE_PATH")
FORWARD_VARS = ("DIFFSYNTH_DOWNLOAD_SOURCE", "DIFFSYNTH_SKIP_DOWNLOAD")


def parse_arm(token: str, default_steps: int) -> tuple[str, int]:
    """`cae@4` -> ("cae", 4); `cae` -> ("cae", default_steps)."""
    name, sep, steps = token.partition("@")
    if name not in ARM_SPECS:
        raise ValueError(f"unknown arm {name!r}; known: {', '.join(ARM_SPECS)}")
    if not sep:
        return name, default_steps
    if not steps.isdigit() or int(steps) < 1:
        raise ValueError(f"bad step count in {token!r}")
    return name, int(steps)


def label(name: str, steps: int) -> str:
    return f"{name}@{steps}"


def worker_argv(name: str, steps: int, a) -> list[str]:
    tree, compile_infer, _ = ARM_SPECS[name]
    return ["python3", WORKER, "--steps", str(steps),
            "--compile-infer", "1" if compile_infer else "0",
            "--text-encoder", "1" if a.text_encoder else "0",
            "--warmup", str(a.warmup), "--iters", str(a.iters),
            "--suite-id", str(a.suite_id),
            "--expect-src", os.path.join(REPO, "src", tree)]


def docker_argv(name: str, gpu: str, env: dict) -> list[str]:
    """Container flags, following eval/native/run_fastwam_libero.sh; EVAL_DOCKER_ARGS adds more."""
    uid, gid = os.getuid(), os.getgid()
    home = env.get("HOME") or f"/tmp/cae_fastwam_latency_home_{uid}"
    if not os.access(home, os.W_OK):
        home = f"/tmp/cae_fastwam_latency_home_{uid}"
    os.makedirs(home, exist_ok=True)
    pw = f"/tmp/ctr_passwd_{uid}"
    with open(f"{pw}.{os.getpid()}", "w") as f:  # written atomically: other tools write the same file
        subprocess.run(["getent", "passwd", str(uid)], stdout=f, check=False)
    os.replace(f"{pw}.{os.getpid()}", pw)

    def gid_of(group: str, fallback: str) -> str:
        p = subprocess.run(["getent", "group", group], capture_output=True, text=True)
        return p.stdout.split(":")[2] if p.returncode == 0 and p.stdout.count(":") >= 2 \
            else fallback

    tree = os.path.join(REPO, "src", ARM_SPECS[name][0])
    # Distinct trees must never share a compile cache: a shared inductor cache
    # can serve one tree's kernels to another.
    cache = f"/tmp/cae_fastwam_latency_{uid}_{ARM_SPECS[name][0]}"
    os.makedirs(f"{cache}/inductor", exist_ok=True)
    os.makedirs(f"{cache}/triton", exist_ok=True)

    mounts = {REPO: "ro", home: "rw", "/tmp": "rw"}
    for v in ("MODEL_CKPT", "ACTION_VAE_CKPT", "MODEL_CONFIGS"):
        mounts.setdefault(os.path.dirname(env[v]) if v != "MODEL_CONFIGS" else env[v], "ro")
    args = ["docker", "run", "--rm",
            "--device=/dev/kfd", "--device=/dev/dri", "--security-opt", "seccomp=unconfined",
            "--group-add", gid_of("video", "44"), "--group-add", gid_of("render", "109"),
            "--network", "host", "--user", f"{uid}:{gid}",
            "--volume", f"{pw}:/etc/passwd:ro", "--env", f"HOME={home}"]
    for v in FORWARD_PATH_VARS:
        if env.get(v):
            os.makedirs(env[v], exist_ok=True)
            mounts.setdefault(env[v], "rw")
            args += ["--env", f"{v}={env[v]}"]
    for v in FORWARD_VARS:
        if env.get(v):
            args += ["--env", f"{v}={env[v]}"]
    for path, mode in mounts.items():
        args += ["--volume", f"{path}:{path}:{mode}"]
    for v in ("MODEL_CKPT", "ACTION_VAE_CKPT", "MODEL_CONFIGS", "TASK_CONFIG"):
        args += ["--env", f"{v}={env[v]}"]
    args += ["--env", f"HIP_VISIBLE_DEVICES={gpu}",
             "--env", f"PYTHONPATH={tree}",
             "--env", f"TORCHINDUCTOR_CACHE_DIR={cache}/inductor",
             "--env", f"TRITON_CACHE_DIR={cache}/triton",
             # No MUJOCO_GL: the worker times the model on synthetic inputs and renders nothing.
             ] + extra_docker_args(env) + ["--workdir", tree, env["EVAL_IMAGE"]]
    return args


def measure(name: str, steps: int, a, env: dict) -> dict | None:
    """One container, one arm; returns the worker's RESULT dict or None."""
    cmd = docker_argv(name, a.gpu, env) + worker_argv(name, steps, a)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
    except subprocess.TimeoutExpired:
        print(f"  [{name}@{steps}] timed out after {a.timeout}s", file=sys.stderr)
        return None
    result = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
    if p.returncode != 0 or result is None:
        tail = "\n".join((p.stdout + p.stderr).splitlines()[-60:])
        print(f"  [{name}@{steps}] failed (rc={p.returncode})\n{tail}", file=sys.stderr)
        return None
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 1)[1])
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--steps", type=int, default=10,
                    help="denoise steps for arms without an @<steps> suffix")
    ap.add_argument("--rounds", type=int, default=3,
                    help="interleaved rounds; 3 is the minimum that shows spread")
    ap.add_argument("--gpus", default=None,
                    help="HIP device index to time on, one id (not the rocm-smi card id; "
                         "default: EVAL_GPUS, else 0)")
    ap.add_argument("--gpu", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--text-encoder", action="store_true",
                    help="keep the text encoder resident (it never runs: context is precomputed)")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--suite-id", type=int, default=0,
                    help="action-VAE suite id; decode cost is the same for every suite")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    try:
        cells = [parse_arm(t, a.steps) for t in a.arms.split(",") if t]
        a.gpu = resolve_one_gpu(a.gpus, a.gpu)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    if len(set(cells)) != len(cells):
        print("duplicate cell in --arms", file=sys.stderr)
        return 2

    env = dict(os.environ)
    env.setdefault("MODEL_CONFIGS", os.path.join(REPO, "FastWAM", "configs"))
    env.setdefault("TASK_CONFIG", DEFAULT_TASK)
    missing = [v for v in ("MODEL_CKPT", "ACTION_VAE_CKPT", "EVAL_IMAGE") if not env.get(v)]
    if missing:
        print(f"set {', '.join(missing)} (see --help)", file=sys.stderr)
        return 2
    for v in ("MODEL_CKPT", "ACTION_VAE_CKPT", "MODEL_CONFIGS"):
        if not os.path.isabs(env[v]) or not os.path.exists(env[v]):
            print(f"{v} must be an existing absolute path: {env[v]}", file=sys.stderr)
            return 2

    # INTERLEAVED: for round -> for arm, so arms in one round share a drift epoch.
    samples: dict[tuple[str, int], list[float]] = {c: [] for c in cells}
    first: dict[tuple[str, int], dict] = {}
    failed = False
    for r in range(a.rounds):
        for c in cells:
            res = measure(*c, a, env)
            if res is None:
                failed = True
            else:
                samples[c].append(res["median_ms"])
                first.setdefault(c, res)
            v = "FAILED" if res is None else f"{res['median_ms']:.3f} ms"
            print(f"  round {r + 1} {label(*c):<18} {v}", flush=True)

    rows = []
    for c in cells:
        s = samples[c]
        if not s:
            continue
        rows.append({"arm": c[0], "steps": c[1], "label": label(*c),
                     "median_ms": statistics.median(s),
                     "spread_pct": (max(s) - min(s)) / min(s) * 100 if len(s) > 1 else 0.0,
                     "samples": s, "contract": first[c]["contract"],
                     "device": first[c]["device"]})
    if not rows:
        print("no successful measurements", file=sys.stderr)
        return 1

    ref = rows[0]["median_ms"]
    print(f"\n{'arm':<18}{'median_ms':>12}{'spread':>9}{'vs ' + rows[0]['label']:>18}")
    for row in rows:
        row["speedup_vs_first"] = ref / row["median_ms"]
        print(f"{row['label']:<18}{row['median_ms']:>12.3f}{row['spread_pct']:>8.2f}%"
              f"{row['speedup_vs_first']:>17.3f}x")
    print("\nSpeedups are relative to the FIRST cell listed. Treat differences below"
          "\nthe spread as unresolved.")
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"rounds": a.rounds, "gpu": a.gpu,
                       "text_encoder_resident": a.text_encoder,
                       "warmup": a.warmup, "iters": a.iters, "rows": rows}, f, indent=2)
        print(f"wrote {a.json_out}")
    if failed:
        print("one or more requested measurements failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
