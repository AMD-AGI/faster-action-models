# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Silent-wrong-result guards of the evaluation launchers, checked without a GPU.

* A rerun into an existing output directory resumes only the same run; a
  directory holding results of another arm, tree, checkpoint, step or trial
  count is refused (RoboCasa protocol.json, the FastWAM launcher's .run_identity).
* scripts/eval_fastwam_libero.sh, a wrapper over the native FastWAM launcher,
  ignores the former policy-server engine's env knobs with a warning.
* DiT4DiT LIBERO counts a failed client the same way with one client or K.
* The latency tools take --gpus and time the FastWAM GEAK arm as it is evaluated.

The launchers run against a stand-in `docker` that fails at once, so they stop
at the first container start; everything checked here happens before that.
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
        for name in ("libero", "home", "inductor", "triton", "configs/task"):
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


class FastwamWrapperTests(unittest.TestCase):
    """scripts/eval_fastwam_libero.sh: the native launcher, one GPU by default, former engine knobs ignored."""

    LAUNCHER = ROOT / "scripts" / "eval_fastwam_libero.sh"
    RESULT = Path("libero_spatial") / "gpu0_task0_results.json"

    def test_wrapper_ignores_engine_knobs_and_resumes_only_the_same_run(self):
        with tempfile.TemporaryDirectory() as td:
            box = Sandbox(Path(td))
            env = {**box.env, "SEED": "7", "WAIT": "5", "SHARDS": "3", "SUITES_CSV": "libero_goal",
                   "PORT": "9000", "EVAL_PORT": "9001", "EVAL_ARM_LABEL": "x"}
            env.pop("EVAL_GPUS", None)
            first = box.run(self.LAUNCHER, "--smoke", "--output-dir", str(box.out), env=env)
            self.assertEqual(first.returncode, 1, first.stderr)  # the stand-in docker fails
            for name in ("SEED", "WAIT", "SHARDS", "SUITES_CSV", "PORT", "EVAL_PORT", "EVAL_ARM_LABEL"):
                self.assertIn(f"warning: ignoring {name}=", first.stderr)
            identity = (box.out / ".run_identity").read_text().splitlines()
            for line in ("arm=baseline", "trials=1", "steps=10", "mode=serial", "sim=mujoco",
                         f"tree={ROOT / 'src' / 'fastwam-baseline'}",
                         f"configs={ROOT / 'FastWAM' / 'configs_baseline'}"):
                self.assertIn(line, identity)
            self.assertIn("gpus=0 ", first.stdout)  # one worker on GPU 0, the wrapper's default

            (box.out / self.RESULT).parent.mkdir(parents=True)
            (box.out / self.RESULT).write_text("{}")
            refused = box.run(self.LAUNCHER, "--full", "--output-dir", str(box.out), env=box.env)
            self.assertEqual(refused.returncode, 2, refused.stderr)
            self.assertIn("recorded: trials=1", refused.stderr)

            same = box.run(self.LAUNCHER, "--smoke", "--output-dir", str(box.out), env=box.env)
            self.assertNotIn("recorded:", same.stderr)

    def test_unrecorded_results_are_refused(self):
        with tempfile.TemporaryDirectory() as td:
            box = Sandbox(Path(td))
            (box.out / self.RESULT).parent.mkdir(parents=True)
            (box.out / self.RESULT).write_text("{}")
            result = box.run(self.LAUNCHER, "--output-dir", str(box.out), env=box.env)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("no record of the run", result.stderr)

    def test_tree_and_config_overrides_are_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            box = Sandbox(Path(td))
            env = {**box.cae_env(), "BASELINE_SRC": str(ROOT / "src" / "fastwam-geak"),
                   "OPTIMIZED_SRC": str(ROOT / "src" / "fastwam-baseline")}
            result = box.run(self.LAUNCHER, "--arm", "cae", "--smoke", "--suites", "libero_spatial",
                             "--output-dir", str(box.out), env=env)
            for name in ("MODEL_CONFIGS", "BASELINE_SRC", "OPTIMIZED_SRC", "CAE_SRC"):
                self.assertIn(f"warning: ignoring {name}=", result.stderr)
            self.assertIn(f"tree={ROOT / 'src' / 'cae-fastwam'}", (box.out / ".run_identity").read_text())

    def test_policy_server_flags_are_refused(self):
        with tempfile.TemporaryDirectory() as td:
            box = Sandbox(Path(td))
            for flag in ("--port", "--clients-per-gpu", "--shards"):
                with self.subTest(flag=flag):
                    result = box.run(self.LAUNCHER, flag, "1", "--dry-run", env=box.env)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("runs no policy server", result.stderr)

    def test_dry_run_matches_the_native_launcher(self):
        env = {k: v for k, v in os.environ.items() if k not in ("EVAL_GPUS", "EVAL_OUTPUT_ROOT")}
        args = ("--arm", "geak", "--steps", "2", "--smoke", "--output-dir", "/tmp/x", "--dry-run")
        wrapper = subprocess.run(["bash", str(self.LAUNCHER), *args], env=env, capture_output=True, text=True)
        native = subprocess.run(["bash", str(ROOT / "eval" / "native" / "run_fastwam_libero.sh"), *args,
                                 "--gpus", "0"], env=env, capture_output=True, text=True)
        self.assertEqual(wrapper.returncode, 0, wrapper.stderr)
        self.assertEqual(wrapper.stdout, native.stdout)


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


class NativeWanPreflightTests(unittest.TestCase):
    LAUNCHER = ROOT / "eval" / "native" / "run_fastwam_libero.sh"

    def test_missing_wan_assets_are_named_before_any_worker(self):
        with tempfile.TemporaryDirectory() as td:
            box = Sandbox(Path(td))
            (box.root / "wan").mkdir()
            env = {**box.env, "DIFFSYNTH_MODEL_BASE_PATH": str(box.root / "wan"), "DIFFSYNTH_SKIP_DOWNLOAD": "true"}
            result = box.run(self.LAUNCHER, "--output-dir", str(box.out), "--gpus", "0", env=env)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("missing Wan asset", result.stderr)
            self.assertIn("Wan2.2_VAE.pth", result.stderr)
            self.assertFalse((box.out / ".run_identity").exists())


class NativeMountTests(unittest.TestCase):
    """eval/native/run_fastwam_libero.sh mounts every path variable, not only the repository, /mnt and /tmp."""

    LAUNCHER = ROOT / "eval" / "native" / "run_fastwam_libero.sh"

    @unittest.skipUnless(os.access("/var/tmp", os.W_OK), "needs a writable /var/tmp (outside the /tmp mount)")
    def test_path_variables_outside_the_default_mounts_are_mounted(self):
        with tempfile.TemporaryDirectory(dir="/var/tmp") as td:
            box = Sandbox(Path(td))
            log = box.root / "docker_args.txt"
            (box.root / "bin" / "docker").write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> {log}\nexit 1\n')
            for name in ("ckpts/stats", "ckpts/model", "wan", "hfhome", "libero_config", "overlay"):
                (box.root / name).mkdir(parents=True)
            for wan in ("Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth", "Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
                        "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json"):
                (box.root / "wan" / wan).parent.mkdir(parents=True, exist_ok=True)
                (box.root / "wan" / wan).write_text("x\n")
            (box.root / "ckpts" / "model" / "model.pt").write_text("x\n")
            env = {**box.env, "MUJOCO_GL": "osmesa", "DIFFSYNTH_MODEL_BASE_PATH": str(box.root / "wan"),
                   "HF_HOME": str(box.root / "hfhome"), "LIBERO_CONFIG_PATH": str(box.root / "libero_config"),
                   "EVAL_PYOVERLAY": str(box.root / "overlay"), "MODEL_CKPT": str(box.root / "ckpts/model/model.pt"),
                   "DATASET_STATS": str(box.root / "ckpts" / "stats" / "s.json"),
                   "DIFFSYNTH_SKIP_DOWNLOAD": "true", "TOKENIZERS_PARALLELISM": "false"}
            box.run(self.LAUNCHER, "--output-dir", str(box.out), "--gpus", "0", "--suites", "libero_spatial",
                    "--trials", "1", env=env, timeout=120)
            args = log.read_text().splitlines()
            r = box.root.resolve()
            for path, mode in ((box.out, "rw"), (r / "wan", "rw"), (r / "hfhome", "rw"), (r / "libero", "ro"),
                               (r / "libero_config", "ro"), (r / "ckpts" / "model", "ro"),
                               (r / "ckpts" / "stats", "ro"), (r / "overlay", "ro")):
                with self.subTest(path=str(path)):
                    self.assertIn(f"{Path(path).resolve()}:{Path(path).resolve()}:{mode}", args)
            self.assertNotIn(f"{r}:{r}:ro", args)   # only the paths themselves, not their parent
            self.assertNotIn(f"{r / 'out'}:{r / 'out'}:ro", args)
            for v in ("DIFFSYNTH_SKIP_DOWNLOAD", "TOKENIZERS_PARALLELISM"):
                self.assertIn(v, args)


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
