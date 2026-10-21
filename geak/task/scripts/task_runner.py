#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GEAK task runner for FastWAM action inference — END-TO-END metric, real shape.

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
A previous campaign lost a round to a benchmark that timed a fictional 9-frame
Sv=882 cache that `infer_action` never builds.

  compile      -> overlay src/ onto a pristine FastWAM tree and import it
  correctness  -> oracle pre-filter, then 14 real MuJoCo LIBERO episodes across
                  all 4 suites, run by FastWAM's own evaluator
                  (experiments/libero/eval_libero_single.py, one persistent
                  worker); PASS iff SR >= threshold on any of GATE_ATTEMPTS
                  independent attempts (see run_correctness on retries) AND
                  every activation marker the candidate's source emits appears
                  in the worker log (see ACTIVATION_MARKERS)
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
# Only `correctness` runs LIBERO episodes; the other commands need no LIBERO paths. Imported as a module (the
# tests), the runner validates everything `correctness` needs.
COMMAND = (sys.argv[1] if len(sys.argv) > 1 else "correctness") if __name__ == "__main__" else "correctness"
LIBERO_COMMANDS = ("correctness", "rollout")

def _required_path(
    name: str,
    *,
    kind: str = "any",
    executable: bool = False,
) -> Path:
    """Resolve and validate one documented task path."""
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
if COMMAND in LIBERO_COMMANDS:
    LIBERO = _required_path("LIBERO_ROOT", kind="dir")
    LIBERO_CONFIG = _required_path("LIBERO_CONFIG_PATH")
    if not (MODEL_SRC_ROOT / "experiments" / "libero" / "eval_libero_single.py").is_file():
        raise SystemExit(
            "MODEL_SRC_ROOT must contain FastWAM's evaluator "
            f"experiments/libero/eval_libero_single.py: {MODEL_SRC_ROOT}"
        )
else:
    LIBERO = Path(os.environ["LIBERO_ROOT"]) if os.environ.get("LIBERO_ROOT") else None
    LIBERO_CONFIG = (Path(os.environ["LIBERO_CONFIG_PATH"])
                     if os.environ.get("LIBERO_CONFIG_PATH") else None)
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
        # "pick up the alphabet soup and place it in the basket" -- REPLACED: FastWAM's own
        # evaluator fails its first episode in this manifest on the unmodified tree, every time
        "pick up the ketchup and place it in the basket",
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
    import_roots = [str(src_dir)] + ([str(LIBERO)] if LIBERO else [])
    if e.get("PYTHONPATH"):
        import_roots.append(e["PYTHONPATH"])
    e["PYTHONPATH"] = os.pathsep.join(import_roots)
    # run_local.sh forwards the MuJoCo rendering backend it resolved (EGL when
    # the image can create a context, else OSMesa) or the user's; unset, OSMesa.
    mujoco_gl = e.setdefault("MUJOCO_GL", "osmesa")
    if mujoco_gl in ("egl", "osmesa"):
        e.setdefault("PYOPENGL_PLATFORM", mujoco_gl)
    # Defaults only: run_local.sh forwards (and mounts) the caller's DIFFSYNTH_* settings, which win.
    e.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
    e.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(HF_CACHE))
    # Hydra otherwise reports a failed import as "Error locating target ..." and hides the cause.
    e.setdefault("HYDRA_FULL_ERROR", "1")
    e["HF_HOME"] = str(HF_CACHE)
    if LIBERO_CONFIG:
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


# Resolves the gate's task names to LIBERO task ids, in the evaluator's own environment.
_TASK_IDS_CODE = """
import json, sys
from libero.libero import benchmark
d = benchmark.get_benchmark_dict()
out = {}
for s in sys.argv[1:]:
    suite = d[s]()
    out[s] = {suite.get_task(i).language: i for i in range(suite.get_num_tasks())}
print("TASK_IDS " + json.dumps(out))
"""


def _gate_items(env: dict) -> list:
    """The manifest as (suite, task_id) work items, matched on the task language."""
    p = subprocess.run(
        [str(MODEL_PYTHON), "-c", _TASK_IDS_CODE] + GATE_SUITES,
        env=env, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL,
    )
    line = next((x for x in p.stdout.splitlines() if x.startswith("TASK_IDS ")), None)
    if p.returncode != 0 or line is None:
        raise SystemExit("could not list the LIBERO tasks:\n" + p.stdout[-2000:] + p.stderr[-2000:])
    ids = json.loads(line[len("TASK_IDS "):])
    items, missing = [], []
    for suite in GATE_SUITES:
        for name in GATE_TASKS[suite]:
            if name in ids[suite]:
                items.append((suite, ids[suite][name]))
            else:
                missing.append(f"{suite}: {name!r}")
    if missing:
        raise SystemExit("gate tasks not found in LIBERO: " + "; ".join(missing))
    return items


def _collect_sr(out: Path, items: list):
    """Pool FastWAM's per-task result files (<out>/<suite>/gpu*_task<id>_results.json).

    Weighted by episode count, as experiments/libero/summarize_results.py pools them.
    """
    total_eps, total_succ, per_suite = 0, 0, {}
    for suite, task_id in items:
        files = sorted((out / suite).glob(f"gpu*_task{task_id}_results.json"))
        if not files:
            continue
        d = json.loads(files[0].read_text())
        n, k = int(d.get("total_episodes", 0)), int(d.get("successes", 0))
        total_eps += n
        total_succ += k
        e = per_suite.setdefault(suite, {"episodes": 0, "successes": 0})
        e["episodes"] += n
        e["successes"] += k
    for e in per_suite.values():
        e["sr"] = e["successes"] / e["episodes"] if e["episodes"] else 0.0
    if total_eps == 0:
        return None, 0, per_suite
    return total_succ / total_eps, total_eps, per_suite


def _run_gate():
    """Run the multi-episode rollout with FastWAM's own evaluator, return (sr, secs).

    One persistent worker (eval_libero_single.py with LIBERO_WORKER_MODE=1) loads the
    overlaid model once and runs every manifest task from its file-backed queue, so
    all 14 episodes share one model, as the deployed policy does. The protocol is
    FastWAM's sim_libero.yaml (seed 42, num_steps_wait 30, replan 10, max_steps
    400/700, binarized gripper; eval/PROTOCOL.md) at 10 denoise steps.
    """
    src = _overlay_tree()
    expected = _expected_markers(src)
    out = Path(tempfile.mkdtemp(prefix="fw_ep_gate_"))
    env = _env(src)
    pool = out / ".pool"
    (pool / "status").mkdir(parents=True)
    worker_log = out / "worker.log"
    proc = None
    try:
        items = _gate_items(env)
        (pool / "pending.txt").write_text("".join(f"{s},{t}\n" for s, t in items))
        (pool / "failed.txt").write_text("")
        env.update({
            "LIBERO_WORKER_MODE": "1", "LIBERO_WORKER_ID": "0",
            "LIBERO_WORKER_PENDING_FILE": str(pool / "pending.txt"),
            "LIBERO_WORKER_LOCK_FILE": str(pool / "pending.lock"),
            "LIBERO_WORKER_STOP_FILE": str(pool / "stop"),
            "LIBERO_WORKER_STATUS_DIR": str(pool / "status"),
            "LIBERO_WORKER_FAILED_FILE": str(pool / "failed.txt"),
        })
        evaluator = src.parent / "experiments" / "libero" / "eval_libero_single.py"
        t0 = time.time()
        log_handle = open(worker_log, "w")
        proc = subprocess.Popen(
            [str(MODEL_PYTHON), str(evaluator),
             "--config-path", _config_dir(), "--config-name", "sim_libero",
             f"ckpt={CKPT}", "gpu_id=0", "seed=42",
             f"EVALUATION.num_trials={GATE_EPISODES_PER_TASK}",
             f"EVALUATION.output_dir={out}",
             f"EVALUATION.dataset_stats_path={STATS}",
             "EVALUATION.num_inference_steps=10",
             "model.redirect_common_files=false"],
            cwd=str(out), env=env, stdin=subprocess.DEVNULL,
            stdout=log_handle, stderr=subprocess.STDOUT,
        )
        log_handle.close()  # the child holds its own descriptor
        # The worker idles on an empty queue until the stop file appears.
        while True:
            done = sum(1 for s, t in items if any((out / s).glob(f"gpu*_task{t}_results.json")))
            if done == len(items) or (pool / "failed.txt").read_text().strip():
                break
            if proc.poll() is not None:
                break
            if time.time() - t0 > 3600:
                print("gate run timed out after 3600 s")
                return None, None
            time.sleep(2)
        secs = time.time() - t0
        (pool / "stop").touch()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_text = worker_log.read_text(errors="ignore")
        failed = (pool / "failed.txt").read_text().strip()
        if failed or proc.returncode not in (0, None):
            print("gate run failed:\n" + failed + "\n" + log_text[-3000:])
            return None, None

        sr, n, per_suite = _collect_sr(out, items)
        if sr is None:
            print("no result json produced:\n" + log_text[-3000:])
            return None, None
        for name, st in sorted(per_suite.items()):
            print(f"[gate]   {name}: {st['episodes']} eps SR={st['sr']:.4f}")
        if n != GATE_EXPECTED_EPISODES:
            print(f"[gate] WARNING: ran {n} episodes, manifest expects "
                  f"{GATE_EXPECTED_EPISODES}")
        print(f"[gate] episodes={n} (expected {GATE_EXPECTED_EPISODES}) "
              f"pooled SR={sr:.4f}")

        # Read after the rollout: graph capture happens on the first model call.
        failures = _check_markers(expected, log_text)
        print(f"[gate] activation markers: {len(expected)} expected, "
              f"{len(failures)} problem(s)")
        for f in failures:
            print(f"[gate]   {f}")
        if failures and REQUIRE_MARKERS:
            raise MarkerFailure("; ".join(failures))
        return sr, secs
    finally:
        if proc is not None and proc.poll() is None:
            (pool / "stop").touch()
            proc.kill()
        shutil.rmtree(src.parent, ignore_errors=True)
        print(f"[gate] results, videos and worker.log: {out}")


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
    # With FastWAM's own evaluator (seed 42 once per worker, one persistent model) a
    # rollout of the UNMODIFIED tree repeats exactly: it passed this manifest 14/14 on
    # MI355X (CPU MuJoCo, OSMesa), and a manifest task that failed once failed every
    # attempt, so it was replaced. A candidate's numerics still move trajectories at
    # boundary episodes (kernel-level nondeterminism), which is what the retries absorb.
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
            # Not a boundary episode: a lever that does not engage fails the same
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
                      f"are consistent with a boundary episode that flips run to run, but a "
                      f"candidate that needs a retry EVERY time is a partial regression.")
            print(f"[correctness] closed-loop PASS (best of {i})")
            return True

    good = [a for a in attempts if a is not None]
    print(f"[correctness] closed-loop FAIL: {GATE_ATTEMPTS} independent attempts, "
          f"SR={[f'{a:.4f}' for a in good]}, none reached {SR_THRESHOLD}. "
          f"The unmodified tree passes this manifest, so this is a real "
          f"regression, not a boundary episode.")
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
    cmd = COMMAND
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
    if cmd == "rollout":           # one closed-loop rollout alone (no oracle), for manual probing
        try:
            sr, _ = _run_gate()
        except MarkerFailure as exc:
            print(f"[rollout] FAIL: {exc}")
            sys.exit(1)
        ok = sr is not None and sr >= SR_THRESHOLD - 1e-9
        print(f"[rollout] SR={sr} threshold={SR_THRESHOLD} -> {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    if cmd == "oracle-refresh":    # record the golden from the PRISTINE tree
        sys.exit(0 if run_oracle(refresh=True) else 1)
    print(f"unknown command {cmd}")
    sys.exit(2)