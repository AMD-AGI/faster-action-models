#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GEAK harness for FastWAM action inference — END-TO-END metric, real shape.

This is the campaign target. It differs from `fastwam_action_real` in ONE way:
the primary metric is the whole `FastWAM.infer_action` call, not just one denoise
step. Same closed-loop correctness gate, same real deployed shape.

WHY END-TO-END (read before changing the metric)
------------------------------------------------
At low step counts the once-per-episode prologue (observation encode + video
prefill) dominates the call, so a large win on the denoise step is a small win on
what ships. Scoring the step alone also makes the prologue invisible to the
optimizer even when it lives in the same editable file. Scoring the FULL call
makes both legitimate targets and makes step/prologue trade-offs visible instead
of silent. Measure your own split before deciding which to weight.

SHAPES ARE MEASURED, NOT ASSUMED. `episode_perf.py` intercepts what
`prefill_video_cache_tensor` actually receives and hard-fails unless
Sv=98 / ctx=129.
A previous campaign lost a round to a harness that timed a fictional 9-frame
Sv=882 cache that `infer_action` never builds.

  compile      -> overlay src/ onto a pristine FastWAM tree and import it
  correctness  -> oracle pre-filter, then 14 real MuJoCo LIBERO episodes across
                  all 4 suites; PASS iff SR >= threshold on any of GATE_ATTEMPTS
                  independent attempts (the gate flakes ~37% on pristine) AND
                  every activation marker the candidate's source emits appears
                  in the server log (see ACTIVATION_MARKERS)
  performance  -> median end-to-end infer_action ms at 1 and 10 denoise steps
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent
os.chdir(TASK_DIR)

def _required_path(
    name: str,
    *,
    kind: str = "any",
    executable: bool = False,
) -> Path:
    """Resolve and validate one documented harness path."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"missing required environment variable {name}; "
            "see docs/fastwam-geak-inference-optimization.md"
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SystemExit(f"{name} must be an absolute path, got: {value}")
    if not path.exists():
        raise SystemExit(f"{name} does not exist: {path}")
    if kind == "dir" and not path.is_dir():
        raise SystemExit(f"{name} must name a directory: {path}")
    if kind == "file" and not path.is_file():
        raise SystemExit(f"{name} must name a file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise SystemExit(f"{name} is not executable: {path}")
    return path


MODEL_SRC_ROOT = _required_path("MODEL_SRC_ROOT", kind="dir")
MODEL_SRC = MODEL_SRC_ROOT / "src"
if not (MODEL_SRC / "fastwam").is_dir():
    raise SystemExit(
        f"MODEL_SRC_ROOT must contain an importable src/fastwam tree: {MODEL_SRC_ROOT}"
    )
if not (MODEL_SRC_ROOT / "configs").is_dir():
    raise SystemExit(f"MODEL_SRC_ROOT must contain configs/: {MODEL_SRC_ROOT}")
CKPT = _required_path("MODEL_CKPT", kind="file")
STATS = _required_path("DATASET_STATS", kind="file")
HARNESS = _required_path("EVAL_HARNESS", kind="dir")
if not (HARNESS / "src").is_dir():
    raise SystemExit(f"EVAL_HARNESS must contain src/: {HARNESS}")
LIBERO = _required_path("LIBERO_ROOT", kind="dir")
LIBERO_CONFIG = _required_path("LIBERO_CONFIG_PATH")
HF_CACHE = _required_path("HF_CACHE_DIR", kind="dir")
MODEL_PYTHON = _required_path("MODEL_PYTHON", kind="file", executable=True)


def _optional_source_tree() -> Path | None:
    """Return FWE_SRC as an import root, when a comparison selected one."""
    value = os.environ.get("FWE_SRC")
    if not value:
        return None
    src = Path(value).expanduser()
    if not src.is_absolute():
        raise SystemExit(f"FWE_SRC must be an absolute path, got: {value}")
    if not src.is_dir():
        raise SystemExit(f"FWE_SRC does not exist or is not a directory: {src}")
    if not (src / "fastwam").is_dir():
        raise SystemExit(
            f"FWE_SRC must directly contain the importable fastwam package: {src}"
        )
    return src


def _config_dir(pristine_only: bool = False) -> str:
    """The Hydra config tree matching the source tree this invocation imports.

    A comparison run that selects FWE_SRC may also select FWE_CONFIG_DIR (the
    FastWAM baseline and GEAK trees need FastWAM/configs_baseline, whose schema
    differs from MODEL_SRC_ROOT/configs). Otherwise, and for the pristine tree, the config
    tree is MODEL_SRC_ROOT/configs.
    """
    value = os.environ.get("FWE_CONFIG_DIR")
    if pristine_only or not value or _optional_source_tree() is None:
        return str(MODEL_SRC_ROOT / "configs")
    cfg = Path(value).expanduser()
    if not cfg.is_absolute() or not (cfg / "sim_libero.yaml").is_file():
        raise SystemExit(
            f"FWE_CONFIG_DIR must be an absolute config tree with sim_libero.yaml: {value}"
        )
    return str(cfg)

# The editable surface. The MI250 version of this task had only mot.py and
# wan_video_dit.py, which is why its final report lists the VAE as "unmodifiable"
# and a large share of the call as untouchable BY CONSTRUCTION -- that was a
# property of the task definition, not of the model. VAE + action DiT are in
# scope here, so that floor is gone.
# Keep in sync with config.yaml:source_file_path.
EDITABLE = ("mot.py", "wan_video_dit.py", "wan_video_vae.py", "action_dit.py")

# Closed-loop gate: an EXPLICIT task manifest, not a shard index.
#
# v1 selected tasks with `--shard-id 0 --num-shards 5` and got, in libero_goal,
# "push the plate to the front of the stove", a task the policy does not complete
# reliably. With 2 episodes of it and a threshold of SR=1.0, the gate demanded two
# consecutive successes at that task, so the unmodified tree failed the gate about
# as often as it passed. That ONE task was the entire source of the "flaky gate".
#
# It is dropped; every remaining task is one the policy completes reliably, and
# GATE_ATTEMPTS retries absorb the rest.
#
# Naming tasks explicitly also removes sharding from the gate entirely, so task
# selection no longer depends on an index into a list order we do not control, and
# every task's trials run contiguously in one env (the ordering that agreed with the
# reference more often when we compared the two).
#
# 7 tasks x 2 episodes = 14 episodes on ONE persistent model. Multi-episode across
# a CHANGING scene is the point -- it is what catches cross-episode staleness.
GATE_TASKS = {
    "libero_spatial": [
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "pick up the black bowl on the ramekin and place it on the plate",
    ],
    "libero_object": [
        "pick up the alphabet soup and place it in the basket",
        "pick up the tomato sauce and place it in the basket",
    ],
    "libero_goal": [
        "open the middle drawer of the cabinet",
        # "push the plate to the front of the stove" -- DROPPED, not completed reliably
    ],
    "libero_10": [
        "pick up the book and place it in the back compartment of the caddy",
        "put both the alphabet soup and the tomato sauce in the basket",
    ],
}
GATE_SUITES = [s for s in os.environ.get(
    "GATE_SUITES", ",".join(GATE_TASKS)).split(",") if s]
unknown_suites = sorted(set(GATE_SUITES) - set(GATE_TASKS))
if unknown_suites:
    raise SystemExit(
        "GATE_SUITES contains unsupported suites: " + ", ".join(unknown_suites)
    )
GATE_EPISODES_PER_TASK = int(os.environ.get("GATE_EPISODES_PER_TASK", "2"))
GATE_EXPECTED_EPISODES = sum(
    len(GATE_TASKS[s]) for s in GATE_SUITES) * GATE_EPISODES_PER_TASK
SR_THRESHOLD = float(os.environ.get("GATE_SR_THRESHOLD", "1.0"))
# Independent rollout attempts before declaring FAIL. Justification and the
# measured false-failure rate this defends against are in run_correctness().
GATE_ATTEMPTS = int(os.environ.get("GATE_ATTEMPTS", "3"))
GPU = os.environ.get("HIP_VISIBLE_DEVICES", os.environ.get("GPU_ID", "0"))
# Port is DERIVED FROM THE GPU, not fixed. The first campaign lost rollouts to
# this: `gpu_lock.sh` serializes GPUs but not the model server's websocket port,
# so two engineers on different GPUs raced for 8093 and the loser got
# `OSError: [Errno 98] address already in use` -- which surfaces as a rollout FAIL
# indistinguishable from a real correctness failure. Keying the port to the GPU
# removes the collision entirely rather than serializing lanes behind a lock.
def _default_port() -> int:
    try:
        return 8093 + int(str(GPU).split(",")[0].strip())
    except (ValueError, IndexError):
        return 8093


PORT = int(os.environ.get("GATE_PORT", _default_port()))

# Activation markers: a GEAK lever that fails to engage falls back to the
# pristine path and still produces correct actions, so SR alone cannot tell an
# optimized candidate from one that silently runs the fallback. Each marker is a
# log line the lever emits when it engages. A marker is REQUIRED only when the
# candidate's own source contains it -- a lever the candidate ships must be seen
# to engage, while a candidate that drops a lever (or the pristine tree, which
# has none) is not held to it. Any FALLBACK_MARKERS line fails the gate outright.
ACTIVATION_MARKERS = (
    "FastWAM: fused projections built",
    "FastWAM: captured video-prefill graph",
    "FastWAM: captured action-denoise graph",
)
FALLBACK_MARKERS = ("graph capture refused",)
# Kill-switch for manual probing only: GATE_REQUIRE_MARKERS=0.
REQUIRE_MARKERS = os.environ.get("GATE_REQUIRE_MARKERS", "1") != "0"


class MarkerFailure(Exception):
    """A shipped lever did not engage. Deterministic, so never retried."""


def _expected_markers(src: Path) -> list:
    """Markers whose literal text appears in the candidate's model source."""
    text = "".join(
        f.read_text(errors="ignore")
        for f in sorted((src / "fastwam" / "models").rglob("*.py"))
    )
    return [m for m in ACTIVATION_MARKERS if m in text]


def _check_markers(expected: list, log_text: str) -> list:
    """Return one failure string per missing marker or fallback line."""
    failures = [f"missing activation marker: {m!r}"
                for m in expected if m not in log_text]
    failures += [f"fallback engaged: {m!r}"
                 for m in FALLBACK_MARKERS if m in log_text]
    return failures


def _gate_max_steps(suite: str) -> int:
    """Return FastWAM's deployed LIBERO horizon for a supported suite."""
    if suite not in GATE_TASKS:
        raise ValueError(f"unsupported LIBERO suite: {suite}")
    return 700 if suite == "libero_10" else 400


def _overlay_tree(pristine_only: bool = False) -> Path:
    """Build an importable FastWAM src tree for this invocation.

    The server auto-detects FastWAM's configs/ as a sibling of src/, so mirror
    that layout (symlink configs + checkpoints rather than copy).

    FWE_SRC selects a complete source tree for comparison runs. Without it, copy
    MODEL_SRC_ROOT/src and overlay the GEAK-edited files from task/src.
    `pristine_only` always uses the unmodified MODEL_SRC_ROOT/src tree.
    """
    tmp = Path(tempfile.mkdtemp(prefix="fw_ep_overlay_"))
    dst = tmp / "src"
    selected = None if pristine_only else _optional_source_tree()
    shutil.copytree(selected or MODEL_SRC, dst, symlinks=True)
    w22 = dst / "fastwam" / "models" / "wan22"
    if not pristine_only and selected is None:
        for f in EDITABLE:
            editable = TASK_DIR / "src" / f
            if not editable.is_file():
                shutil.rmtree(tmp, ignore_errors=True)
                raise SystemExit(
                    f"missing GEAK editable file {editable}; seed geak/task/src "
                    "as documented before running the task"
                )
            shutil.copy2(editable, w22 / f)
    for sib in ("configs", "checkpoints", "experiments"):
        s = MODEL_SRC_ROOT / sib
        if s.exists() and not (tmp / sib).exists():
            (tmp / sib).symlink_to(s, target_is_directory=True)
    return dst


def _env(src_dir: Path) -> dict:
    e = dict(os.environ)
    import_roots = [str(src_dir), str(HARNESS / "src"), str(LIBERO)]
    if e.get("PYTHONPATH"):
        import_roots.append(e["PYTHONPATH"])
    e["PYTHONPATH"] = os.pathsep.join(import_roots)
    # run_local.sh forwards the MuJoCo rendering backend it resolved (EGL when
    # the image can create a context, else OSMesa) or the user's; unset, OSMesa.
    mujoco_gl = e.setdefault("MUJOCO_GL", "osmesa")
    if mujoco_gl in ("egl", "osmesa"):
        e.setdefault("PYOPENGL_PLATFORM", mujoco_gl)
    e["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
    e["DIFFSYNTH_MODEL_BASE_PATH"] = str(HF_CACHE)
    e["HF_HOME"] = str(HF_CACHE)
    e["LIBERO_CONFIG_PATH"] = str(LIBERO_CONFIG)
    e["HIP_VISIBLE_DEVICES"] = GPU
    return e


def run_compile():
    src = _overlay_tree()
    p = subprocess.run(
        [str(MODEL_PYTHON), "-c",
         "import torch;"
         "from fastwam.models.wan22.mot import MoT;"
         "from fastwam.models.wan22.wan_video_dit import DiTBlock;"
         "from fastwam.models.wan22.wan_video_vae import WanVideoVAE, Encoder3d;"
         "from fastwam.models.wan22.action_dit import ActionDiT, ActionHead;"
         "print('import OK')"],
        env=_env(src), capture_output=True, text=True, timeout=600,
    )
    shutil.rmtree(src.parent, ignore_errors=True)
    if p.returncode != 0:
        print("Compile failed:\n" + p.stdout + p.stderr)
        return False
    print("Compile OK")
    return True


def _write_gate_config(out_dir: Path) -> Path:
    cfg = out_dir / "gate.yaml"
    head = (
        "server:\n"
        f"  url: \"ws://localhost:{PORT}\"\n"
        "  timeout: 120.0\n"
        "docker:\n"
        "  image: ghcr.io/allenai/vla-evaluation-harness/libero:latest\n"
        "  gpus: \"none\"\n"
        f"output_dir: \"{out_dir}\"\n"
        "benchmarks:\n"
    )
    # `subname` is required with multiple suites: it disambiguates the per-suite
    # result filenames, which otherwise collide and trip the harness's own
    # "Result file already exists" guard.
    def _block(suite: str) -> str:
        tasks = "".join(f"      - \"{t}\"\n" for t in GATE_TASKS[suite])
        return (
            "  - benchmark: \"vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark\"\n"
            f"    subname: {suite}\n"
            "    mode: sync\n"
            f"    episodes_per_task: {GATE_EPISODES_PER_TASK}\n"
            "    tasks:\n" + tasks +
            "    params:\n"
            f"      suite: {suite}\n"
            "      seed: 42\n"
            "      num_steps_wait: 30\n"
            f"      max_steps: {_gate_max_steps(suite)}\n"
            "      quat_no_antipodal: true\n"
        )

    body = "".join(_block(suite) for suite in GATE_SUITES)
    cfg.write_text(head + body)
    return cfg


def _collect_sr(out: Path):
    """Aggregate across the per-suite result files.

    One file per suite, so taking res[0] (what the single-suite version did)
    would silently score ONE suite and ignore the other three. Weight by episode
    count so the aggregate is the true pooled success rate.
    """
    files = sorted(out.glob("LIBEROBenchmark_*.json"))
    if not files:
        return None, 0, {}
    total_eps, total_succ, per_suite = 0, 0.0, {}
    for f in files:
        d = json.loads(f.read_text())
        n = sum(t.get("num_episodes", 0) for t in d.get("tasks", []))
        sr = float(d.get("mean_success", 0.0))
        if n == 0:
            continue
        total_eps += n
        total_succ += sr * n
        per_suite[f.stem] = {"episodes": n, "sr": sr}
    if total_eps == 0:
        return None, 0, per_suite
    return total_succ / total_eps, total_eps, per_suite


def _run_gate():
    """Serve the overlaid model, run the multi-episode rollout, return (sr, secs)."""
    src = _overlay_tree()
    expected = _expected_markers(src)
    out = Path(tempfile.mkdtemp(prefix="fw_ep_gate_"))
    env = _env(src)
    srv_log = out / "server.log"
    srv = subprocess.Popen(
        [str(MODEL_PYTHON), "-m", "vla_eval.model_servers.fastwam",
         "--ckpt", str(CKPT), "--dataset_stats_path", str(STATS), "--seed", "42",
         "--port", str(PORT), "--num_inference_steps", "10",
         "--chunk_size", "10", "--action_ensemble", "newest"],
        cwd=str(HARNESS), env=env,
        stdout=open(srv_log, "w"), stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(300):
            if srv.poll() is not None:
                print("SERVER DIED:\n" + srv_log.read_text()[-3000:])
                return None, None
            if "Starting server on ws://" in srv_log.read_text(errors="ignore"):
                break
            time.sleep(2)
        else:
            print("server never became ready")
            return None, None

        cfg = _write_gate_config(out)
        t0 = time.time()
        p = subprocess.run(
            # No --shard-id/--num-shards: tasks are named explicitly in gate.yaml,
            # so selection no longer depends on an index into the suite's order.
            [str(MODEL_PYTHON), "src/vla_eval/cli/main.py", "run", "-c", str(cfg),
             "--server-url", f"ws://localhost:{PORT}", "--output-dir", str(out),
             "--no-docker", "-y"],
            cwd=str(HARNESS), env=env, capture_output=True, text=True, timeout=3600,
        )
        secs = time.time() - t0
        if p.returncode != 0:
            print("gate run failed:\n" + p.stdout[-2000:] + p.stderr[-2000:])
            return None, None

        sr, n, per_suite = _collect_sr(out)
        if sr is None:
            print("no result json produced")
            return None, None
        for name, s in sorted(per_suite.items()):
            print(f"[gate]   {name}: {s['episodes']} eps SR={s['sr']:.4f}")
        if n != GATE_EXPECTED_EPISODES:
            print(f"[gate] WARNING: ran {n} episodes, manifest expects "
                  f"{GATE_EXPECTED_EPISODES} -- task names may not match the suite")
        print(f"[gate] episodes={n} (expected {GATE_EXPECTED_EPISODES}) "
              f"pooled SR={sr:.4f}")

        # Read after the rollout: graph capture happens on the first request.
        failures = _check_markers(expected, srv_log.read_text(errors="ignore"))
        print(f"[gate] activation markers: {len(expected)} expected, "
              f"{len(failures)} problem(s)")
        for f in failures:
            print(f"[gate]   {f}")
        if failures and REQUIRE_MARKERS:
            raise MarkerFailure("; ".join(failures))
        return sr, secs
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except Exception:
            srv.kill()
        shutil.rmtree(src.parent, ignore_errors=True)


ORACLE_SAMPLES = int(os.environ.get("ORACLE_SAMPLES", "3"))

# Which tree the golden is recorded FROM.
#
# Recording from PRISTINE means the oracle measures CUMULATIVE drift since the
# original model, which is the right thing for one campaign and the wrong thing
# for a chain of them: after campaign 2 the tree sat at mean|d| 2.023e-02 against
# a 2.251e-02 tolerance -- 90% of the budget spent, with 87% of that from a single
# accepted sub-lever. A third campaign would start rejecting GOOD patches for
# inherited drift, which is exactly the failure mode the rollout gate just had.
#
# Recording from the CURRENT src/ re-bases the pre-filter on the checkpoint we
# so it answers "did THIS round change numerics" with a full budget again.
#
# What that gives up: the oracle no longer tracks drift back to pristine. That is
# acceptable ONLY because the oracle was never the authority -- the closed-loop
# rollout is, it runs every round, and it compares against real task success rather
# than against a stored tensor. Re-base only from a tree that has PASSED the rollout.
ORACLE_BASE_PRISTINE = os.environ.get("ORACLE_BASE", "current").lower() == "pristine"


def _oracle_proc(extra: list, pristine: bool, golden: str | None = None):
    """One episode_oracle.py invocation in its own process."""
    src = _overlay_tree(pristine_only=pristine)
    env = _env(src)
    env["FWE_CKPT"] = str(CKPT)
    env["FWE_CONFIG_DIR"] = _config_dir(pristine_only=pristine)
    env.setdefault("HF_HUB_OFFLINE", "1")
    if golden:
        env["ORACLE_GOLDEN"] = golden
    p = subprocess.run(
        [str(MODEL_PYTHON), str(TASK_DIR / "scripts" / "episode_oracle.py")] + extra,
        env=env, capture_output=True, text=True, timeout=3600,
    )
    shutil.rmtree(src.parent, ignore_errors=True)
    print(p.stdout.strip())
    if p.returncode != 0:
        print("oracle failed:\n" + p.stderr[-3000:])
    return p.returncode == 0


def run_oracle(refresh: bool = False):
    """Cheap numerics pre-filter. NOT the authority -- see episode_oracle.py."""
    if not refresh:
        return _oracle_proc([], pristine=False)

    # Record N samples in N SEPARATE processes, from the PRISTINE tree. Separate
    # processes is the whole point: the golden is read back across a process
    # boundary, and this model differs by 1-2 bf16 ULP across processes while
    # being bit-identical within one. Calibrating within a process yields a 0.0
    # tolerance that then fails the unmodified tree.
    tmp = Path(tempfile.mkdtemp(prefix="fw_ep_cal_"))
    try:
        samples = []
        for i in range(ORACLE_SAMPLES):
            s = str(tmp / f"sample{i}.pt")
            print(f"[oracle] recording sample {i + 1}/{ORACLE_SAMPLES} "
                  f"from {'PRISTINE' if ORACLE_BASE_PRISTINE else 'CURRENT src/'}")
            if not _oracle_proc(["--refresh"], pristine=ORACLE_BASE_PRISTINE, golden=s):
                return False
            samples.append(s)
        return _oracle_proc(["--calibrate"] + samples,
                            pristine=ORACLE_BASE_PRISTINE)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_correctness():
    # Two gates, both must pass. The oracle is a PRE-FILTER ONLY: it replays one
    # input, so it cannot see cross-episode cache/graph staleness -- the exact bug
    # class compile + graph capture introduces, and the one that took a sibling
    # model from SR 1.0 to 0.0 while a numerics oracle stayed green at rel 1.4e-2.
    # Closed-loop multi-episode LIBERO is the authority.
    if not run_oracle():
        print("[correctness] oracle FAILED -- skipping the rollout")
        return False

    # BEST-OF-N, and the asymmetry is the whole justification.
    #
    # The v1 gate was a SINGLE rollout against an absolute SR>=1.0. Measured over
    # the first campaign, that fails BYTE-IDENTICAL PRISTINE SOURCE ~37% of the
    # time (5 PASS / 3 FAIL), always on the same one boundary episode of
    # libero_goal (0.75 = 3/4 -> pooled 0.9375). It cost that run ~5 wasted
    # rollouts, sank two directions on verdicts that were probably noise, and
    # discarded the one patch that actually worked.
    #
    # Retrying is sound here ONLY because the two failure modes have different
    # shapes: the false failure is STOCHASTIC (an env/render boundary episode that
    # flips run to run) while a real regression is DETERMINISTIC (broken numerics
    # or stale cross-episode state fail every time). So N independent attempts
    # drive the false-failure rate to 0.37^N -- ~5% at N=3 -- while a genuinely
    # broken candidate still fails all N.
    #
    # This is NOT "retry until green". A candidate that fails every attempt FAILS,
    # and every attempt is printed so a reader can see whether the run was clean
    # or scraped through on attempt 3. Watch for a candidate that needs the retry
    # every single time: that is a partial regression, not noise.
    attempts = []
    for i in range(1, GATE_ATTEMPTS + 1):
        try:
            sr, _ = _run_gate()
        except MarkerFailure as exc:
            # Not the flaky episode: a lever that does not engage fails the same
            # way every attempt, so retrying would only burn rollouts.
            print(f"[correctness] attempt {i}/{GATE_ATTEMPTS}: FAIL, the candidate "
                  f"did not run the levers it ships ({exc})")
            return False
        if sr is None:
            print(f"[correctness] attempt {i}/{GATE_ATTEMPTS}: gate produced no result")
            attempts.append(None)
            continue
        attempts.append(sr)
        ok = sr >= SR_THRESHOLD - 1e-9
        print(f"[correctness] attempt {i}/{GATE_ATTEMPTS} SR={sr:.4f} "
              f"threshold={SR_THRESHOLD} -> {'PASS' if ok else 'FAIL'}", flush=True)
        if ok:
            if i > 1:
                print(f"[correctness] NOTE: passed on attempt {i}. Earlier attempts "
                      f"{[f'{a:.4f}' if a is not None else 'ERR' for a in attempts[:-1]]} "
                      f"are consistent with the known flaky libero_goal episode, but a "
                      f"candidate that needs a retry EVERY time is a partial regression.")
            print(f"[correctness] closed-loop PASS (best of {i})")
            return True

    good = [a for a in attempts if a is not None]
    print(f"[correctness] closed-loop FAIL: {GATE_ATTEMPTS} independent attempts, "
          f"SR={[f'{a:.4f}' for a in good]}, none reached {SR_THRESHOLD}. "
          f"Pristine clears this gate within 3 attempts ~95% of the time, so this "
          f"is a real regression, not the flaky episode.")
    return False


def run_performance():
    """End-to-end infer_action latency on the real model at 1 and 10 steps."""
    src = _overlay_tree()
    env = _env(src)
    env["FWE_CKPT"] = str(CKPT)
    env["FWE_CONFIG_DIR"] = _config_dir()
    env.setdefault("HF_HUB_OFFLINE", "1")
    p = subprocess.run(
        [str(MODEL_PYTHON), str(TASK_DIR / "scripts" / "episode_perf.py")],
        env=env, capture_output=True, text=True, timeout=3600,
    )
    shutil.rmtree(src.parent, ignore_errors=True)
    print(p.stdout.strip())
    if p.returncode != 0 or "PERF_DONE" not in p.stdout:
        print("performance failed:\n" + p.stderr[-3000:])
        return False
    return True


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if cmd == "compile":
        sys.exit(0 if run_compile() else 1)
    if cmd == "correctness":
        if not run_correctness():
            print("Validation failed")
            sys.exit(1)
        print("Correctness OK")
        sys.exit(0)
    if cmd == "performance":
        sys.exit(0 if run_performance() else 1)
    if cmd == "oracle":            # pre-filter alone, for manual probing
        sys.exit(0 if run_oracle() else 1)
    if cmd == "oracle-refresh":    # record the golden from the PRISTINE tree
        sys.exit(0 if run_oracle(refresh=True) else 1)
    print(f"unknown command {cmd}")
    sys.exit(2)