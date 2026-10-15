# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Silent-wrong-result guards of the evaluation launchers, checked without a GPU.

* A rerun into an existing output directory resumes only the same run; a
  directory holding results of another arm, tree, checkpoint, step or trial
  count is refused (RoboCasa protocol.json, the FastWAM engines' and the native
  launcher's .run_identity, the CAE-FastWAM engine's .done_<suite> markers).
* The FastWAM launcher takes the protocol and its flags, never the engines'
  standalone env knobs, and warns when a tree or config override differs from
  what --arm selects.
* DiT4DiT LIBERO counts a failed client the same way with one client or K.
* The latency tools take --gpus and time the FastWAM GEAK arm as it is evaluated.

The engines run against a stand-in `docker` that fails at once, so they stop
at the first server start; everything checked here happens before that.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_FLAGS = ROOT / "scripts" / "eval_flags.sh"


def load(path: Path):
    name = f"t_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


class Sandbox:
    """A temp tree with stand-in assets and a `docker` that always fails."""

    def __init__(self, root: Path):
        self.root = root
        bin_dir = root / "bin"
        bin_dir.mkdir()
        docker = bin_dir / "docker"
        docker.write_text("#!/usr/bin/env bash\nexit 1\n")
        docker.chmod(0o755)
        for name in ("harness", "libero", "home", "inductor", "triton", "configs/task"):
            (root / name).mkdir(parents=True)
        for name in ("model.pt", "vae.pt", "stats.json"):
            (root / name).write_text("x\n")
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
            (root / f"stats_{suite}.json").write_text("{}\n")
            (root / "configs" / "task" / f"cae_{suite}.yaml").write_text("{}\n")
        self.out = root / "out"
        self.env = {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": str(root / "home"),
            "EVAL_IMAGE": "test-image",
            "EVAL_HARNESS": str(root / "harness"),
            "LIBERO_ROOT": str(root / "libero"),
            "MODEL_CKPT": str(root / "model.pt"),
            "DATASET_STATS": str(root / "stats.json"),
            "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor"),
            "TRITON_CACHE_DIR": str(root / "triton"),
        }

    def cae_env(self) -> dict:
        return {
            **self.env,
            "ACTION_VAE_CKPT": str(self.root / "vae.pt"),
            "MODEL_CONFIGS": str(self.root / "configs"),
            "TASK_CONFIG": "cae",
            "DATASET_STATS_TEMPLATE": str(self.root / "stats_{suite}.json"),
            "CAE_SRC": str(ROOT / "src" / "cae-fastwam"),
            "EVAL_OUTPUT_ROOT": str(self.out),
        }

    def run(self, script: Path, *args, env: dict, timeout: int = 60):
        return subprocess.run(["bash", str(script), *args], env=env, capture_output=True,
                              text=True, timeout=timeout)


class ClaimRunDirTests(unittest.TestCase):
    def claim(self, file: Path, identity: str, has_results: int):
        return subprocess.run(
            ["bash", "-c", 'source "$1"; eval_claim_run_dir "$2" "$3" "$4"', "_",
             str(EVAL_FLAGS), str(file), identity, str(has_results)],
            capture_output=True, text=True,
        )

    def test_same_run_resumes_and_a_different_one_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / ".run_identity"
            self.assertEqual(self.claim(marker, "arm=a\ntrials=50", 0).returncode, 0)
            self.assertEqual(marker.read_text(), "arm=a\ntrials=50\n")
            self.assertEqual(self.claim(marker, "arm=a\ntrials=50", 1).returncode, 0)
            refused = self.claim(marker, "arm=a\ntrials=1", 1)
            self.assertEqual(refused.returncode, 2)
            self.assertIn("recorded: trials=50", refused.stderr)
            self.assertIn("this run: trials=1", refused.stderr)
            self.assertIn("fresh --output-dir", refused.stderr)

    def test_results_without_a_record_are_refused_and_an_empty_dir_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / ".run_identity"
            refused = self.claim(marker, "arm=a", 1)
            self.assertEqual(refused.returncode, 2)
            self.assertIn("no record of the run", refused.stderr)
            self.assertFalse(marker.exists())
            marker.write_text("arm=b\n")
            self.assertEqual(self.claim(marker, "arm=a", 0).returncode, 0)
            self.assertEqual(marker.read_text(), "arm=a\n")


class FastwamBaselineGeakResumeTests(unittest.TestCase):
    LAUNCHER = ROOT / "scripts" / "eval_fastwam_libero.sh"

    def test_launcher_ignores_engine_knobs_and_resumes_only_the_same_cell(self):
        with tempfile.TemporaryDirectory() as td:
            box = Sandbox(Path(td))
            env = {**box.env, "SEED": "7", "WAIT": "5", "SHARDS": "3",
                   "SUITES_CSV": "libero_goal", "TASK_CONFIG": "cae"}
            first = box.run(self.LAUNCHER, "--smoke", "--output-dir", str(box.out), env=env)
            self.assertEqual(first.returncode, 1, first.stderr)  # the stand-in docker fails
            for name in ("SEED", "WAIT", "SHARDS", "SUITES_CSV", "TASK_CONFIG"):
                self.assertIn(f"warning: ignoring {name}=", first.stderr)
            identity = (box.out / "baseline_10step" / ".run_identity").read_text().splitlines()
            for line in ("seed=42", "num_steps_wait=30", "shards=10", "trials=1", "task_config=",
                         f"tree={ROOT / 'src' / 'fastwam-baseline'}",
                         f"configs={ROOT / 'FastWAM' / 'configs_baseline'}"):
                self.assertIn(line, identity)

            (box.out / "baseline_10step" / "LIBEROBenchmark_libero_spatial_shard0.json").write_text("{}")
            refused = box.run(self.LAUNCHER, "--full", "--output-dir", str(box.out), env=box.env)
            self.assertEqual(refused.returncode, 2, refused.stderr)
            self.assertIn("refusing to resume", refused.stderr)
            self.assertIn("recorded: trials=1", refused.stderr)

            same = box.run(self.LAUNCHER, "--smoke", "--output-dir", str(box.out), env=box.env)
            self.assertNotIn("refusing to resume", same.stderr)

    def test_unrecorded_results_are_refused(self):
        with tempfile.TemporaryDirectory() as td:
            box = Sandbox(Path(td))
            cell = box.out / "baseline_10step"
            cell.mkdir(parents=True)
            (cell / "LIBEROBenchmark_libero_spatial_shard0.json").write_text("{}")
            result = box.run(self.LAUNCHER, "--output-dir", str(box.out), env=box.env)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("no record of the run", result.stderr)

    def test_tree_and_config_overrides_warn(self):
        with tempfile.TemporaryDirectory() as td:
            box = Sandbox(Path(td))
            other = str(ROOT / "src" / "cae-fastwam")
            env = {**box.env, "BASELINE_SRC": other, "MODEL_CONFIGS": str(ROOT / "FastWAM" / "configs")}
            result = box.run(self.LAUNCHER, "--smoke", "--output-dir", str(box.out), env=env)
            self.assertIn(f"warning: BASELINE_SRC={other} overrides", result.stderr)
            self.assertIn("warning: MODEL_CONFIGS=", result.stderr)
            self.assertIn("which --arm baseline selects", result.stderr)
            # The resolved values themselves are not overrides.
            env = {**box.env, "BASELINE_SRC": str(ROOT / "src" / "fastwam-baseline"),
                   "OPTIMIZED_SRC": str(ROOT / "src" / "fastwam-geak")}
            result = box.run(self.LAUNCHER, "--smoke", "--output-dir", str(box.out), env=env)
            self.assertNotIn("overrides", result.stderr)
            # --arm cae has its own override; the other arms' are ignored, with a warning.
            env = {**box.cae_env(), "BASELINE_SRC": str(ROOT / "src" / "fastwam-baseline"),
                   "OPTIMIZED_SRC": str(ROOT / "src" / "fastwam-geak")}
            result = box.run(self.LAUNCHER, "--arm", "cae", "--smoke", "--output-dir", str(box.out / "cae"),
                             env=env)
            self.assertNotRegex(result.stderr, r"_SRC=\S+ overrides")
            self.assertIn("warning: ignoring OPTIMIZED_SRC=", result.stderr)
            self.assertIn("warning: ignoring BASELINE_SRC=", result.stderr)
            env = {**box.cae_env(), "CAE_SRC": str(ROOT / "src" / "fastwam-geak")}
            result = box.run(self.LAUNCHER, "--arm", "cae", "--smoke", "--output-dir", str(box.out / "cae2"),
                             env=env)
            self.assertIn(f"warning: CAE_SRC={ROOT / 'src' / 'fastwam-geak'} overrides", result.stderr)


class CaeFastwamResumeTests(unittest.TestCase):
    ENGINE = ROOT / "eval" / "run_eval_cae_fastwam.sh"
    ARGS = ("--suites", "libero_spatial", "--gpus", "0")

    def test_done_markers_record_the_run_and_refuse_another(self):
        with tempfile.TemporaryDirectory() as td:
            box = Sandbox(Path(td))
            env = box.cae_env()
            first = box.run(self.ENGINE, *self.ARGS, "--trials", "50", env=env)
            self.assertEqual(first.returncode, 1, first.stderr)
            out = box.out / "cae_10step"
            identity = (out / ".run_identity").read_text()
            self.assertIn("trials=50\n", identity)
            self.assertIn("steps=10\n", identity)
            self.assertIn(f"tree={ROOT / 'src' / 'cae-fastwam'}\n", identity)

            # A finished suite of this run is skipped ...
            (out / ".done_libero_spatial").write_text(identity)
            same = box.run(self.ENGINE, *self.ARGS, "--trials", "50", env=env)
            self.assertEqual(same.returncode, 0, same.stderr)
            self.assertIn("skip libero_spatial (already done)", same.stdout)

            # ... but never under a run with another trial count.
            other = box.run(self.ENGINE, *self.ARGS, "--trials", "1", env=env)
            self.assertEqual(other.returncode, 2, other.stderr)
            self.assertIn("recorded: trials=50", other.stderr)
            self.assertNotIn("skip libero_spatial", other.stdout)

    def test_bare_done_marker_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as td:
            box = Sandbox(Path(td))
            out = box.out / "cae_10step"
            out.mkdir(parents=True)
            (out / ".done_libero_spatial").touch()
            result = box.run(self.ENGINE, *self.ARGS, env=box.cae_env())
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("no record of the run", result.stderr)

    def test_launcher_passes_shards_and_drops_suites_csv(self):
        with tempfile.TemporaryDirectory() as td:
            box = Sandbox(Path(td))
            env = {**box.cae_env(), "SHARDS": "3", "SUITES_CSV": "libero_goal", "SEED": "1"}
            env.pop("CAE_SRC")
            env.pop("MODEL_CONFIGS")
            result = box.run(ROOT / "scripts" / "eval_fastwam_libero.sh", "--arm", "cae",
                             "--smoke", "--output-dir", str(box.out), env=env)
            # FastWAM/configs carries no task "cae"; the engine stops at that check, after
            # the launcher has dropped the knobs.
            self.assertIn("warning: ignoring SHARDS=3", result.stderr)
            self.assertIn("warning: ignoring SUITES_CSV=libero_goal", result.stderr)
            self.assertIn("per-suite task config missing for libero_spatial", result.stderr)


class NativeResumeTests(unittest.TestCase):
    LAUNCHER = ROOT / "eval" / "native" / "run_fastwam_libero.sh"

    def test_foreign_results_are_refused_before_any_worker(self):
        with tempfile.TemporaryDirectory() as td:
            box = Sandbox(Path(td))
            suite = box.out / "libero_spatial"
            suite.mkdir(parents=True)
            (suite / "gpu0_task0_results.json").write_text("{}")
            result = box.run(self.LAUNCHER, "--output-dir", str(box.out), "--gpus", "0", env=box.env)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("no record of the run", result.stderr)

            (box.out / ".run_identity").write_text("arm=geak\n")
            result = box.run(self.LAUNCHER, "--output-dir", str(box.out), "--gpus", "0", env=box.env)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("recorded: arm=geak", result.stderr)
            self.assertIn("this run: arm=baseline", result.stderr)


class RobocasaResumeTests(unittest.TestCase):
    def setUp(self):
        self.driver = load(ROOT / "scripts" / "robocasa_full24.py")
        self.protocol = {
            "checkpoint": {"path": "/c/model.pt", "size": 10, "mtime_ns": 1},
            "ceiling_vae": None,
            "condition": "direct",
            "episodes_per_task": 50,
            "max_episode_steps": 720,
            "n_action_steps": 12,
            "policy_seed": 7,
            "eval_seed": 7,
            "rollout_matching": {"strategy": "s", "version": 1},
            "simulation_robosuite_overlay": {"version": "1.5.1"},
            "runtime_versions": {"mujoco": "3.2.6", "python": "3.11"},
            "gpu_list": ["0", "1"],
            "clients_per_gpu": 1,
            "policy_server": {"root": "/r/src/dit4dit-baseline", "script": "/r/serve.py",
                              "seed": "DIT4DIT_POLICY_SEED",
                              "label": "arm=baseline steps=4 sim=mujoco"},
        }

    def changed(self, path, value):
        other = json.loads(json.dumps(self.protocol))
        target = other
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return other

    def test_same_run_with_another_split_resumes(self):
        self.assertEqual(self.driver.resume_mismatches(self.protocol, self.protocol), [])
        split = self.changed(("clients_per_gpu",), 6)
        split["gpu_list"] = ["0", "1", "2"]
        self.assertEqual(self.driver.resume_mismatches(self.protocol, split), [])

    def test_another_arm_checkpoint_count_steps_or_sim_is_refused(self):
        cases = {
            "policy_server.root": (("policy_server", "root"), "/r/src/dit4dit-geak"),
            "policy_server.label": (("policy_server", "label"), "arm=baseline steps=2 sim=mujoco"),
            "checkpoint": (("checkpoint", "mtime_ns"), 2),
            "episodes_per_task": (("episodes_per_task",), 10),
            "runtime_versions.mujoco": (("runtime_versions", "mujoco"), "3.3.2"),
        }
        for name, (path, value) in cases.items():
            with self.subTest(field=name):
                mismatches = self.driver.resume_mismatches(self.protocol, self.changed(path, value))
                self.assertEqual(len(mismatches), 1, mismatches)
                self.assertTrue(mismatches[0].startswith(name + ":"), mismatches)

    def test_launcher_help_states_the_refusal(self):
        text = subprocess.run(["bash", str(ROOT / "scripts" / "eval_dit4dit_robocasa.sh"), "--help"],
                              capture_output=True, text=True).stdout
        self.assertIn("is refused", text)


class Dit4ditLiberoClientFailureTests(unittest.TestCase):
    def test_one_client_failure_is_counted_like_k(self):
        text = (ROOT / "scripts" / "eval_dit4dit_libero.sh").read_text()
        # The K=1 client runs as a job whose status is collected, as the K>1
        # clients are, never as a foreground pipeline that errexit would end.
        self.assertIn('run_libero_client "$suite" 2>&1 | tee "${suite_logs[0]}" &\n'
                      '    wait "$!" || suite_failed=1', text)
        self.assertNotIn('run_libero_client "$suite" 2>&1 | tee "${output_dir}', text)
        self.assertIn("failed_clients=$((failed_clients + suite_failed))", text)

    def test_pipeline_status_reaches_the_counter(self):
        script = (
            "set -euo pipefail\n"
            "client() { ( false; echo unreachable ); }\n"
            "failed=0\n"
            "client 2>&1 | tee /dev/null &\n"
            'wait "$!" || failed=1\n'
            'echo "failed=$failed"\n'
        )
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "failed=1")


class LatencyToolTests(unittest.TestCase):
    def setUp(self):
        self.compare = load(ROOT / "scripts" / "compare_artifacts.py")
        self.flags = load(ROOT / "scripts" / "eval_flags.py")

    def test_gpus_defaults_to_eval_gpus_and_takes_one_id(self):
        saved = os.environ.pop("EVAL_GPUS", None)
        try:
            self.assertEqual(self.flags.resolve_one_gpu(None), "0")
            os.environ["EVAL_GPUS"] = "3"
            self.assertEqual(self.flags.resolve_one_gpu(None), "3")
            self.assertEqual(self.flags.resolve_one_gpu("5"), "5")
            for bad in ("0,1", "x", ""):
                with self.subTest(gpus=bad), self.assertRaises(ValueError):
                    self.flags.resolve_one_gpu(bad)
        finally:
            os.environ.pop("EVAL_GPUS", None)
            if saved is not None:
                os.environ["EVAL_GPUS"] = saved

    def test_gpu_is_a_deprecated_alias(self):
        for script in ("compare_artifacts.py", "cae_fastwam_latency.py"):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / script), "--gpu", "0,1"],
                    capture_output=True, text=True,
                    env={k: v for k, v in os.environ.items() if k != "EVAL_GPUS"},
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("warning: --gpu is deprecated and will be removed in the next "
                              "release; use --gpus", result.stderr)
                self.assertIn("pass one GPU id to --gpus", result.stderr)

    def test_geak_arms_time_with_the_kv_slab_off_and_a_cache_of_their_own(self):
        for arm in ("geak", "geak-reduced"):
            with self.subTest(arm=arm):
                env = self.compare.arm_env(arm, 2, "4", "/src", {arm: "/tmp/cache"})
                self.assertEqual(env["FASTWAM_KVSLAB"], "0")
                self.assertEqual(env["HIP_VISIBLE_DEVICES"], "4")
                self.assertEqual(env["TORCHINDUCTOR_CACHE_DIR"], "/tmp/cache/inductor")
                self.assertEqual(env["TRITON_CACHE_DIR"], "/tmp/cache/triton")
        self.assertEqual(set(self.compare.FRESH_CACHE_ARMS), {"compile", "geak", "geak-reduced"})
        self.assertNotIn("FASTWAM_KVSLAB", self.compare.ARM_SPECS["eager"][1])

    def test_reduced_step_default_is_two(self):
        text = (ROOT / "scripts" / "compare_artifacts.py").read_text()
        self.assertIn('ap.add_argument("--reduced-steps", type=int, default=2,', text)


if __name__ == "__main__":
    unittest.main()
