#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DiT4DiT latency: every arm in ONE interleaved session.

    python3 scripts/dit4dit_latency.py --bench libero --gpus 0
    python3 scripts/dit4dit_latency.py --bench robocasa --arms baseline@4,compile@4,geak@4,geak@2 --rounds 5
    python3 scripts/dit4dit_latency.py --bench libero --dry-run

Each (round, arm) runs `scripts/dit4dit_latency_worker.py` in a FRESH container
with that arm's source tree and diffusers overlay, set up the way
scripts/eval_dit4dit_libero.sh and scripts/eval_dit4dit_robocasa.sh start the
policy server: PYTHONPATH = <tree>:<diffusers overlay>[:DIT4DIT_PYOVERLAY],
cwd = <tree>, the four GEAK_* gates pinned to 0, DIT4DIT_INFER_STEPS = the
cell's step count (applied and read back by scripts/dit4dit_step_override.py),
TOKENIZERS_PARALLELISM=false, and one inductor/Triton cache per tree and arm.
The worker checks the cell before timing it (tree, diffusers, gates, executed step
count, request and action shapes) and times the policy server's call. See its
docstring for the input and timing contract.

Latency is not portable across nodes, drivers or sessions, so arms are measured
INTERLEAVED (for round: for arm) and only compared within one run. The number
reported per arm is the median over rounds of the worker's median, with the
round-to-round spread. Nothing is committed. Re-run it on the machine you care
about.

ARMS  (append @<steps> to override --steps for that arm, e.g. geak@2)
  baseline   src/dit4dit-baseline, unmodified upstream diffusers 3996788.
             The comparison point of the GEAK arm.
  compile    naive torch.compile: the baseline arm's tree and diffusers overlay,
             unchanged, with torch.compile at default settings applied by the
             worker to the heavy modules of each call (see
             scripts/dit4dit_latency_worker.py, --naive-compile modules). Own
             compile cache. 4 steps only (the baseline step count).
  geak       src/dit4dit-geak, diffusers 3996788 with src/dit4dit-geak/patches.
Step counts are the DiT4DiT ones the launchers advertise: 4 (paper default)
and 2. The default --arms is baseline@4,compile@4,geak@4,geak@2, so one default
session covers the naive torch.compile point too.

DIFFUSERS OVERLAYS
  Built once per run and arm, in a container of EVAL_IMAGE, by
  scripts/dit4dit_diffusers_overlay.sh -- the same builder the DiT4DiT
  launchers use: a copy of DIT4DIT_DIFFUSERS_SRC (default: the image's
  installed diffusers), checked to be unmodified 3996788, patched and
  re-checked on the geak arm only (the compile arm reuses the baseline overlay).
  They live under --work-dir, next to the compile caches.

--gpus takes the one HIP device to time on (default: EVAL_GPUS, else 0); the
former --gpu still works with a deprecation warning.

ENVIRONMENT
  required  DIT4DIT_LIBERO_CKPT    DiT4DiT LIBERO checkpoint (--bench libero)
            DIT4DIT_ROBOCASA_CKPT  DiT4DiT RoboCasa-GR1 checkpoint (--bench robocasa)
            EVAL_IMAGE             container image with the DiT4DiT model environment
  optional  DIT4DIT_DIFFUSERS_SRC  diffusers package directory to copy into the
                                   overlay (default: the model environment's
                                   installed diffusers; required when that
                                   environment ships no diffusers). The model
                                   environment is EVAL_IMAGE.
            DIT4DIT_PYOVERLAY      third-party wheel overlay(s), colon-separated,
                                   added to PYTHONPATH after the diffusers overlay.
                                   Together with the image it must provide the
                                   transformers and tokenizers versions pinned in
                                   the tree's requirements.txt; the worker refuses
                                   the cell otherwise and records torch, diffusers,
                                   transformers and tokenizers versions in its result.
            HF_HOME HF_CACHE_DIR HUGGINGFACE_HUB_CACHE TORCH_HOME XDG_CACHE_HOME
            HF_HUB_OFFLINE TRANSFORMERS_OFFLINE   forwarded to the container when set

Absolute paths named in the checkpoint's run-directory config.yaml (the local
Cosmos-Predict2.5-2B base model) are mounted read-only into the container.

The RoboCasa cells load the policy in bf16, as the RoboCasa launcher's server
does (--use_bf16); the LIBERO cells do not, as the LIBERO launcher's server does
not.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_flags import resolve_one_gpu  # noqa: E402  (scripts/eval_flags.py)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(REPO, "scripts", "dit4dit_latency_worker.py")
OVERLAY_BUILDER = os.path.join(REPO, "scripts", "dit4dit_diffusers_overlay.sh")
DEFAULT_ARMS = "baseline@4,compile@4,geak@4,geak@2"
DIT4DIT_STEPS = (4, 2)
CONTAINER_PYTHON = "python3"

ARM_SPECS = {
    # name: (tree under src/, description)
    "baseline": ("dit4dit-baseline", "upstream DiT4DiT, upstream diffusers"),
    "compile":  ("dit4dit-baseline", "upstream DiT4DiT, upstream diffusers, naive torch.compile"),
    "geak":     ("dit4dit-geak",     "GEAK tree, patched diffusers"),
}
# The diffusers overlay each arm runs with (built once per overlay name).
ARM_OVERLAY = {"baseline": "baseline", "compile": "baseline", "geak": "geak"}
# Arms the worker compiles naively (--naive-compile), and the only step counts
# they are timed at: the baseline ones.
ARM_NAIVE = {"compile": "modules"}
NAIVE_STEPS = (4,)
BENCH_CKPT = {"libero": "DIT4DIT_LIBERO_CKPT", "robocasa": "DIT4DIT_ROBOCASA_CKPT"}
# Pinned off on every arm, as the DiT4DiT launchers pin them.
GATES = ("GEAK_ACTION_COMPILE", "GEAK_COSMOS_COMPILE", "GEAK_VAE_T1", "GEAK_ACTION_BF16")
FORWARD_PATH_VARS = ("HF_CACHE_DIR", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME",
                     "XDG_CACHE_HOME")
FORWARD_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def parse_arm(token: str, default_steps: int) -> tuple[str, int]:
    """`geak@2` -> ("geak", 2); `geak` -> ("geak", default_steps)."""
    name, sep, steps = token.partition("@")
    if name not in ARM_SPECS:
        raise ValueError(f"unknown arm {name!r}; known: {', '.join(ARM_SPECS)}")
    if sep and not steps.isdigit():
        raise ValueError(f"bad step count in {token!r}")
    n = int(steps) if sep else default_steps
    if n not in DIT4DIT_STEPS:
        raise ValueError(f"{token!r}: {n} is not a DiT4DiT step count; the advertised "
                         f"step counts are 4 (paper default) and 2")
    if name in ARM_NAIVE and n not in NAIVE_STEPS:
        raise ValueError(f"{token!r}: the naive torch.compile arm is timed at the baseline "
                         f"step count only ({name}@4); reduced steps are a GEAK-arm "
                         f"comparison (geak@{n})")
    return name, n


def label(name: str, steps: int) -> str:
    return f"{name}@{steps}"


def tree_of(name: str) -> str:
    return os.path.join(REPO, "src", ARM_SPECS[name][0])


def overlay_root(work: str, overlay: str) -> str:
    """Directory the overlay builder fills; the package is <root>/diffusers.

    OVERLAY is an overlay name (a value of ARM_OVERLAY)."""
    return os.path.join(work, "overlays", overlay)


def arm_overlay_root(work: str, name: str) -> str:
    """The overlay arm NAME runs with."""
    return overlay_root(work, ARM_OVERLAY[name])


def cache_root(work: str, name: str) -> str:
    # Distinct trees must never share a compile cache: a shared inductor cache
    # can serve one tree's kernels to another. The naive torch.compile arm runs
    # the baseline tree but gets a cache of its own (one per tree and arm).
    tree = ARM_SPECS[name][0]
    return os.path.join(work, "cache", f"{tree}-{name}" if name in ARM_NAIVE else tree)


def pyoverlay_entries(env: dict) -> list[str]:
    return [p for p in env.get("DIT4DIT_PYOVERLAY", "").split(":") if p]


def config_mounts(ckpt: str) -> list[str]:
    """Absolute, existing paths named in the checkpoint's run-dir config.yaml.

    DiT4DiT reads <run_dir>/config.yaml, run_dir = the checkpoint's grandparent;
    the config names the local Cosmos base model by absolute path.
    """
    cfg = os.path.join(os.path.dirname(os.path.dirname(ckpt)), "config.yaml")
    try:
        with open(cfg) as f:
            text = f.read()
    except OSError:
        return []
    found = []
    for m in re.finditer(r":\s*['\"]?(/[^\s'\"#]+)", text):
        p = m.group(1).rstrip("/")
        if p and os.path.exists(p) and p not in found:
            found.append(p)
    return found


def _gid_of(group: str, fallback: str) -> str:
    p = subprocess.run(["getent", "group", group], capture_output=True, text=True)
    return p.stdout.split(":")[2] if p.returncode == 0 and p.stdout.count(":") >= 2 \
        else fallback


def _passwd_file() -> str:
    return f"/tmp/ctr_passwd_{os.getuid()}"


def _container_base(env: dict, work: str, mounts: dict, gpu_devices: bool) -> list[str]:
    """`docker run` flags shared by the overlay build and the timed cells."""
    uid, gid = os.getuid(), os.getgid()
    args = ["docker", "run", "--rm"]
    if gpu_devices:
        args += ["--device=/dev/kfd", "--device=/dev/dri", "--security-opt", "seccomp=unconfined",
                 "--group-add", _gid_of("video", "44"), "--group-add", _gid_of("render", "109")]
    args += ["--network", "host", "--user", f"{uid}:{gid}",
             "--volume", f"{_passwd_file()}:/etc/passwd:ro",
             "--env", f"HOME={os.path.join(work, 'home')}"]
    m = {REPO: "ro", work: "rw"}
    for k, v in mounts.items():
        m.setdefault(k, v)
    for v in FORWARD_PATH_VARS:
        if env.get(v):
            m.setdefault(env[v], "rw")
            args += ["--env", f"{v}={env[v]}"]
    for v in FORWARD_VARS:
        if env.get(v):
            args += ["--env", f"{v}={env[v]}"]
    for p in pyoverlay_entries(env):
        m.setdefault(p, "ro")
    if env.get("DIT4DIT_DIFFUSERS_SRC"):
        m.setdefault(env["DIT4DIT_DIFFUSERS_SRC"], "ro")
    for path, mode in m.items():
        args += ["--volume", f"{path}:{path}:{mode}"]
    return args


def overlay_argv(name: str, env: dict, work: str) -> list[str]:
    """Build one diffusers overlay (an ARM_OVERLAY name) with the launchers' own builder."""
    args = _container_base(env, work, {}, gpu_devices=False)
    for v in ("DIT4DIT_DIFFUSERS_SRC", "DIT4DIT_PYOVERLAY"):
        if env.get(v):
            args += ["--env", f"{v}={env[v]}"]
    args += ["--env", "PYTHONNOUSERSITE=1", env["EVAL_IMAGE"],
             "bash", "-c", 'source "$1" && build_diffusers_overlay "$2" "$3" "$4"', "overlay",
             OVERLAY_BUILDER, CONTAINER_PYTHON, name, overlay_root(work, name)]
    return args


def worker_argv(bench: str, name: str, steps: int, a, ckpt: str) -> list[str]:
    return [CONTAINER_PYTHON, WORKER, "--bench", bench, "--ckpt", ckpt,
            "--steps", str(steps),
            "--expect-tree", tree_of(name),
            "--expect-diffusers", os.path.join(arm_overlay_root(a.work_dir, name), "diffusers"),
            "--naive-compile", ARM_NAIVE.get(name, "none"),
            "--warmup", str(a.warmup), "--iters", str(a.iters)]


def docker_argv(name: str, steps: int, gpu: str, env: dict, work: str, ckpt: str) -> list[str]:
    """Container flags for one timed cell, as the DiT4DiT launchers start the server."""
    tree = tree_of(name)
    cache = cache_root(work, name)
    mounts = {}
    if os.path.isabs(ckpt):   # a dry run may carry a placeholder
        # The checkpoint, and its run directory (config.yaml, dataset statistics).
        mounts[os.path.dirname(os.path.dirname(ckpt))] = "ro"
        for p in config_mounts(ckpt):
            mounts.setdefault(p, "ro")
    args = _container_base(env, work, mounts, gpu_devices=True)
    pythonpath = [tree, arm_overlay_root(work, name)] + pyoverlay_entries(env)
    args += ["--env", f"HIP_VISIBLE_DEVICES={gpu}", "--env", f"CUDA_VISIBLE_DEVICES={gpu}"]
    for g in GATES:
        args += ["--env", f"{g}=0"]
    args += ["--env", f"DIT4DIT_INFER_STEPS={steps}",
             "--env", "TOKENIZERS_PARALLELISM=false",
             "--env", f"TORCHINDUCTOR_CACHE_DIR={cache}/inductor",
             "--env", f"TRITON_CACHE_DIR={cache}/triton",
             "--env", "PYTHONPATH=" + ":".join(pythonpath),
             "--env", "PYTHONNOUSERSITE=1", "--env", "PYTHONUNBUFFERED=1",
             "--env", "PYTHONDONTWRITEBYTECODE=1",
             "--workdir", tree, env["EVAL_IMAGE"]]
    return args


def prepare_host(env: dict, work: str, names) -> None:
    """Side effects the container commands rely on."""
    with open(_passwd_file(), "w") as f:
        subprocess.run(["getent", "passwd", str(os.getuid())], stdout=f, check=False)
    os.makedirs(os.path.join(work, "home"), exist_ok=True)
    for n in names:
        os.makedirs(f"{cache_root(work, n)}/inductor", exist_ok=True)
        os.makedirs(f"{cache_root(work, n)}/triton", exist_ok=True)
    for v in FORWARD_PATH_VARS:
        if env.get(v):
            os.makedirs(env[v], exist_ok=True)


def build_overlays(names, env: dict, a) -> bool:
    for n in dict.fromkeys(ARM_OVERLAY[x] for x in names):
        print(f"building diffusers overlay for {n} -> {overlay_root(a.work_dir, n)}", flush=True)
        try:
            p = subprocess.run(overlay_argv(n, env, a.work_dir), capture_output=True, text=True,
                               timeout=1800)
        except subprocess.TimeoutExpired:
            print(f"  overlay build for {n} timed out", file=sys.stderr)
            return False
        if p.returncode != 0:
            tail = "\n".join((p.stdout + p.stderr).splitlines()[-15:])
            print(f"  overlay build for {n} failed (rc={p.returncode})\n{tail}", file=sys.stderr)
            return False
    return True


def measure(bench: str, name: str, steps: int, a, env: dict, ckpt: str) -> dict | None:
    """One container, one arm; returns the worker's RESULT dict or None."""
    cmd = docker_argv(name, steps, a.gpu, env, a.work_dir, ckpt) + \
        worker_argv(bench, name, steps, a, ckpt)
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
        tail = "\n".join((p.stdout + p.stderr).splitlines()[-15:])
        print(f"  [{name}@{steps}] failed (rc={p.returncode})\n{tail}", file=sys.stderr)
        return None
    return result


def check_env(bench: str, env: dict) -> str | None:
    """Return an error message, or None when everything a run needs is present."""
    ckpt_var = BENCH_CKPT[bench]
    missing = [v for v in (ckpt_var, "EVAL_IMAGE") if not env.get(v)]
    if missing:
        return f"set {', '.join(missing)} (see --help)"
    ckpt = env[ckpt_var]
    if not os.path.isabs(ckpt) or not os.path.isfile(ckpt) or not ckpt.endswith(".pt"):
        return f"{ckpt_var} must be an existing absolute path to a .pt checkpoint: {ckpt}"
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    for f in ("config.yaml", "dataset_statistics.json"):
        if not os.path.isfile(os.path.join(run_dir, f)):
            return (f"{ckpt_var}: DiT4DiT expects {f} in the checkpoint's run directory "
                    f"(<run_dir>/<subdir>/<name>.pt): {run_dir}")
    src = env.get("DIT4DIT_DIFFUSERS_SRC")
    if src and (not os.path.isabs(src) or not os.path.isfile(os.path.join(src, "__init__.py"))):
        return f"DIT4DIT_DIFFUSERS_SRC must be an absolute diffusers package directory: {src}"
    for p in pyoverlay_entries(env):
        if not os.path.isabs(p) or not os.path.exists(p):
            return f"DIT4DIT_PYOVERLAY entries must be existing absolute paths: {p}"
    for v in FORWARD_PATH_VARS:
        if env.get(v) and not os.path.isabs(env[v]):
            return f"{v} must be an absolute path: {env[v]}"
    if shutil.which("docker") is None:
        return "docker not found on PATH"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 1)[1])
    ap.add_argument("--bench", choices=tuple(BENCH_CKPT), required=True)
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--steps", type=int, default=4,
                    help="denoise steps for arms without an @<steps> suffix (4 or 2)")
    ap.add_argument("--rounds", type=int, default=5,
                    help="interleaved rounds; 3 is the minimum that shows spread")
    ap.add_argument("--gpus", default=None,
                    help="HIP device index to time on, one id (not the rocm-smi card id; "
                         "default: EVAL_GPUS, else 0)")
    ap.add_argument("--gpu", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--warmup", type=int, default=25,
                    help="untimed calls per cell, the first of which is the contract check")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=7200, help="seconds per cell")
    ap.add_argument("--work-dir", default=None,
                    help="diffusers overlays and compile caches "
                         "(default: /tmp/dit4dit_latency_<uid>)")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and every container command, then exit")
    a = ap.parse_args()

    try:
        cells = [parse_arm(t, a.steps) for t in a.arms.split(",") if t]
        a.gpu = resolve_one_gpu(a.gpus, a.gpu)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    if not cells:
        print("--arms is empty", file=sys.stderr)
        return 2
    if len(set(cells)) != len(cells):
        print("duplicate arm in --arms", file=sys.stderr)
        return 2
    for flag, v in (("--rounds", a.rounds), ("--warmup", a.warmup), ("--iters", a.iters),
                    ("--timeout", a.timeout)):
        if v < 1:
            print(f"{flag} must be a positive integer, got: {v}", file=sys.stderr)
            return 2
    a.work_dir = os.path.abspath(a.work_dir or f"/tmp/dit4dit_latency_{os.getuid()}")
    names = list(dict.fromkeys(c[0] for c in cells))
    env = dict(os.environ)
    ckpt_var = BENCH_CKPT[a.bench]

    if a.dry_run:
        for v in (ckpt_var, "EVAL_IMAGE"):   # placeholders: a dry run needs no assets
            if not env.get(v):
                env[v] = f"<{v}>"
        ckpt = env[ckpt_var]
        print(f"bench={a.bench} gpus={a.gpu} rounds={a.rounds} warmup={a.warmup} "
              f"iters={a.iters} work_dir={a.work_dir}")
        print("order (every round): " + ", ".join(label(*c) for c in cells))
        for n in dict.fromkeys(ARM_OVERLAY[x] for x in names):
            print(f"\n# diffusers overlay, {n}\n" + shlex.join(overlay_argv(n, env, a.work_dir)))
        for c in cells:
            cmd = docker_argv(c[0], c[1], a.gpu, env, a.work_dir, ckpt) + \
                worker_argv(a.bench, c[0], c[1], a, ckpt)
            print(f"\n# cell {label(*c)}\n" + shlex.join(cmd))
        return 0

    err = check_env(a.bench, env)
    if err:
        print(err, file=sys.stderr)
        return 2
    ckpt = os.path.realpath(env[ckpt_var])
    prepare_host(env, a.work_dir, names)
    if not build_overlays(names, env, a):
        return 1

    # INTERLEAVED: for round -> for arm, so arms in one round share a drift epoch.
    samples: dict[tuple[str, int], list[float]] = {c: [] for c in cells}
    first: dict[tuple[str, int], dict] = {}
    failed = False
    for r in range(a.rounds):
        for c in cells:
            res = measure(a.bench, *c, a, env, ckpt)
            if res is None:
                failed = True
            else:
                samples[c].append(res["median_ms"])
                first.setdefault(c, res)
            v = "FAILED" if res is None else f"{res['median_ms']:.3f} ms"
            cc = (res or {}).get("compile")
            if cc:
                v += (f"  [compile: {cc['unique_graphs']} graphs, {cc['graph_breaks']} breaks, "
                      f"first call {cc['first_call_s']:.1f} s]")
            print(f"  round {r + 1} {label(*c):<14} {v}", flush=True)

    rows = []
    for c in cells:
        s = samples[c]
        if not s:
            continue
        rows.append({"arm": c[0], "steps": c[1], "label": label(*c),
                     "median_ms": statistics.median(s),
                     "spread_pct": (max(s) - min(s)) / min(s) * 100 if len(s) > 1 else 0.0,
                     "samples": s, "contract": first[c]["contract"],
                     "compile": first[c].get("compile"), "device": first[c]["device"]})
    if not rows:
        print("no successful measurements", file=sys.stderr)
        return 1

    ref = rows[0]["median_ms"]
    print(f"\n{'arm':<14}{'median_ms':>12}{'spread':>9}{'vs ' + rows[0]['label']:>18}")
    for row in rows:
        row["speedup_vs_first"] = ref / row["median_ms"]
        print(f"{row['label']:<14}{row['median_ms']:>12.3f}{row['spread_pct']:>8.2f}%"
              f"{row['speedup_vs_first']:>17.3f}x")
    print("\nSpeedups are relative to the FIRST arm listed. Treat differences below"
          "\nthe spread as unresolved.")
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"bench": a.bench, "rounds": a.rounds, "gpu": a.gpu,
                       "warmup": a.warmup, "iters": a.iters, "rows": rows}, f, indent=2)
        print(f"wrote {a.json_out}")
    if failed:
        print("one or more requested measurements failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
