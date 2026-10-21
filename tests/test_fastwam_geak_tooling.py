# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TASK_RUNNER = ROOT / "geak" / "task" / "scripts" / "task_runner.py"
COMPARE_ARTIFACTS = ROOT / "scripts" / "compare_artifacts.py"
GEAK_LOCAL_RUNNER = ROOT / "geak" / "task" / "run_local.sh"
EDITABLE = ("mot.py", "wan_video_dit.py", "wan_video_vae.py", "action_dit.py")


# A stand-in for FastWAM's experiments/libero/eval_libero_single.py in persistent-worker mode: it pops
# "suite,task" items from LIBERO_WORKER_PENDING_FILE, writes FastWAM's per-task result file, and exits when the
# stop file appears.
FAKE_EVALUATOR = r'''
import json, os, sys, time
from pathlib import Path
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a and not a.startswith("--"))
out = Path(args["EVALUATION.output_dir"]); trials = int(args["EVALUATION.num_trials"])
assert os.environ["LIBERO_WORKER_MODE"] == "1" and args["seed"] == "42"
assert args["EVALUATION.num_inference_steps"] == "10" and args["model.redirect_common_files"] == "false"
if os.environ.get("FAKE_EVAL_MARKER"):
    print(os.environ["FAKE_EVAL_MARKER"], flush=True)
pending = Path(os.environ["LIBERO_WORKER_PENDING_FILE"]); stop = Path(os.environ["LIBERO_WORKER_STOP_FILE"])
while not stop.exists():
    lines = [l for l in pending.read_text().splitlines() if l.strip()]
    if not lines:
        time.sleep(0.05); continue
    pending.write_text("".join(l + "\n" for l in lines[1:]))
    suite, task = lines[0].split(",")
    ok = 0 if os.environ.get("FAKE_EVAL_FAIL_SUITE") == suite else trials
    (out / suite).mkdir(parents=True, exist_ok=True)
    (out / suite / f"gpu0_task{task}_results.json").write_text(
        json.dumps({"successes": ok, "total_episodes": trials}))
'''


def write_fake_libero(root: Path) -> None:
    """A LIBERO package whose suites hold the gate's task names (as ids 3, 4, ...) among others."""
    pkg = root / "libero" / "libero"
    pkg.mkdir(parents=True)
    (root / "libero" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    runner_src = TASK_RUNNER.read_text()
    start = runner_src.index("GATE_TASKS = {")
    end = runner_src.index("\n}\n", start) + 3
    (pkg / "benchmark.py").write_text(runner_src[start:end] + '''
class _Task:
    def __init__(self, language):
        self.language = language

class _Suite:
    def __init__(self, names):
        self.names = ["other task a", "other task b", "other task c"] + list(names)
    def get_num_tasks(self):
        return len(self.names)
    def get_task(self, i):
        return _Task(self.names[i])

def get_benchmark_dict():
    return {s: (lambda s=s: _Suite(GATE_TASKS[s])) for s in GATE_TASKS}
''')


def load_module(path: Path):
    name = f"fastwam_geak_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TaskRunnerSourceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.old_cwd = Path.cwd()
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.model = root / "model"
        self.model_src = self.model / "src"
        self.wan22 = self.model_src / "fastwam" / "models" / "wan22"
        self.wan22.mkdir(parents=True)
        for filename in EDITABLE:
            (self.wan22 / filename).write_text(f"model:{filename}\n")
        (self.model_src / "fastwam" / "marker.py").write_text("model\n")
        for sibling in ("configs", "checkpoints", "experiments"):
            (self.model / sibling).mkdir()
        (self.model / "experiments" / "libero").mkdir()
        (self.model / "experiments" / "libero" / "eval_libero_single.py").write_text(FAKE_EVALUATOR)

        self.ckpt = self.model / "checkpoints" / "model.pt"
        self.stats = self.model / "checkpoints" / "stats.json"
        self.ckpt.touch()
        self.stats.write_text("{}\n")
        self.libero = root / "libero"
        write_fake_libero(self.libero)
        self.libero_config = root / ".libero"
        self.libero_config.write_text("benchmark_root: test\n")
        self.hf_cache = root / "hf"
        self.hf_cache.mkdir()
        self.task = root / "task"
        (self.task / "src").mkdir(parents=True)
        for filename in EDITABLE:
            (self.task / "src" / filename).write_text(f"edited:{filename}\n")

        self.env = {
            "MODEL_SRC_ROOT": str(self.model),
            "MODEL_CKPT": str(self.ckpt),
            "DATASET_STATS": str(self.stats),
            "LIBERO_ROOT": str(self.libero),
            "LIBERO_CONFIG_PATH": str(self.libero_config),
            "HF_CACHE_DIR": str(self.hf_cache),
            "MODEL_PYTHON": sys.executable,
        }
        self.env_patch = mock.patch.dict(os.environ, self.env, clear=False)
        self.env_patch.start()
        os.environ.pop("FWE_SRC", None)
        self.runner = load_module(TASK_RUNNER)
        self.runner.TASK_DIR = self.task
        # The gate keeps its output directory; keep the tests' under their own temporary directory.
        tempdir_patch = mock.patch.object(tempfile, "tempdir", str(root))
        tempdir_patch.start()
        self.addCleanup(tempdir_patch.stop)

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.env_patch.stop()
        self.tmp.cleanup()

    def test_default_source_overlays_geak_editables_and_links_siblings(self):
        src = self.runner._overlay_tree()
        self.addCleanup(shutil.rmtree, src.parent, True)

        for filename in EDITABLE:
            actual = src / "fastwam" / "models" / "wan22" / filename
            self.assertEqual(actual.read_text(), f"edited:{filename}\n")
        self.assertEqual((src / "fastwam" / "marker.py").read_text(), "model\n")
        for sibling in ("configs", "checkpoints", "experiments"):
            link = src.parent / sibling
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), (self.model / sibling).resolve())

    def test_fwe_src_selects_complete_import_tree_without_geak_overlay(self):
        selected = Path(self.tmp.name) / "selected"
        selected_wan22 = selected / "fastwam" / "models" / "wan22"
        selected_wan22.mkdir(parents=True)
        for filename in EDITABLE:
            (selected_wan22 / filename).write_text(f"selected:{filename}\n")
        with mock.patch.dict(os.environ, {"FWE_SRC": str(selected)}):
            src = self.runner._overlay_tree()
        self.addCleanup(shutil.rmtree, src.parent, True)

        for filename in EDITABLE:
            actual = src / "fastwam" / "models" / "wan22" / filename
            self.assertEqual(actual.read_text(), f"selected:{filename}\n")

    def test_pristine_source_ignores_fwe_src_and_geak_overlay(self):
        selected = Path(self.tmp.name) / "selected"
        (selected / "fastwam").mkdir(parents=True)
        with mock.patch.dict(os.environ, {"FWE_SRC": str(selected)}):
            src = self.runner._overlay_tree(pristine_only=True)
        self.addCleanup(shutil.rmtree, src.parent, True)

        for filename in EDITABLE:
            actual = src / "fastwam" / "models" / "wan22" / filename
            self.assertEqual(actual.read_text(), f"model:{filename}\n")

    def test_config_dir_follows_a_selected_source_tree(self):
        default = str(self.model / "configs")
        self.assertEqual(self.runner._config_dir(), default)
        selected = Path(self.tmp.name) / "selected"
        (selected / "fastwam").mkdir(parents=True)
        configs = Path(self.tmp.name) / "configs_baseline"
        configs.mkdir()
        (configs / "sim_libero.yaml").write_text("{}\n")
        with mock.patch.dict(os.environ, {"FWE_CONFIG_DIR": str(configs)}):
            # Without FWE_SRC the task's own tree, and its configs, are used.
            self.assertEqual(self.runner._config_dir(), default)
        with mock.patch.dict(os.environ, {"FWE_SRC": str(selected), "FWE_CONFIG_DIR": str(configs)}):
            self.assertEqual(self.runner._config_dir(), str(configs))
            self.assertEqual(self.runner._config_dir(pristine_only=True), default)
        with mock.patch.dict(os.environ, {"FWE_SRC": str(selected), "FWE_CONFIG_DIR": str(selected)}):
            with self.assertRaises(SystemExit):
                self.runner._config_dir()

    def test_markers_are_required_only_when_the_source_emits_them(self):
        src = self.runner._overlay_tree()
        self.addCleanup(shutil.rmtree, src.parent, True)
        self.assertEqual(self.runner._expected_markers(src), [])
        mot = src / "fastwam" / "models" / "wan22" / "mot.py"
        mot.write_text('logger.info("FastWAM: captured action-denoise graph '
                       '(%d layers).", n)\n')
        self.assertEqual(self.runner._expected_markers(src),
                         ["FastWAM: captured action-denoise graph"])

    def test_missing_marker_or_fallback_fails(self):
        marker = "FastWAM: captured action-denoise graph"
        engaged = f"INFO mot: {marker} (30 layers).\n"
        self.assertEqual(self.runner._check_markers([marker], engaged), [])
        self.assertEqual(len(self.runner._check_markers([marker], "")), 1)
        refused = engaged + "WARNING mot: FastWAM: video-prefill graph capture refused\n"
        self.assertEqual(len(self.runner._check_markers([marker], refused)), 1)
        self.assertEqual(self.runner._check_markers([], ""), [])

    def test_geak_tree_emits_every_activation_marker(self):
        # If a marker string is renamed in the shipped tree, the gate would
        # silently stop requiring it; pin the set against the real source.
        tree = ROOT / "src" / "fastwam-geak"
        self.assertEqual(self.runner._expected_markers(tree),
                         list(self.runner.ACTIVATION_MARKERS))
        self.assertEqual(
            self.runner._expected_markers(ROOT / "src" / "fastwam-baseline"), [])

    def test_gate_tasks_resolve_to_libero_task_ids_by_language(self):
        src = self.runner._overlay_tree()
        self.addCleanup(shutil.rmtree, src.parent, True)
        items = self.runner._gate_items(self.runner._env(src))
        self.assertEqual(len(items), sum(len(v) for v in self.runner.GATE_TASKS.values()))
        # write_fake_libero numbers each suite's manifest tasks from 3 in manifest order
        self.assertEqual(items[:2], [("libero_spatial", 3), ("libero_spatial", 4)])

    def test_gate_runs_fastwams_evaluator_and_pools_its_result_files(self):
        sr, secs = self.runner._run_gate()
        self.assertEqual(sr, 1.0)
        self.assertIsNotNone(secs)

    def test_gate_scores_partial_success_from_result_files(self):
        with mock.patch.dict(os.environ, {"FAKE_EVAL_FAIL_SUITE": "libero_goal"}):
            sr, _ = self.runner._run_gate()
        n = self.runner.GATE_EXPECTED_EPISODES
        self.assertAlmostEqual(sr, (n - self.runner.GATE_EPISODES_PER_TASK) / n)

    def test_gate_requires_the_markers_the_candidate_ships(self):
        mot = self.task / "src" / "mot.py"
        mot.write_text('logger.info("FastWAM: captured action-denoise graph")\n')
        with self.assertRaises(self.runner.MarkerFailure):
            self.runner._run_gate()
        with mock.patch.dict(os.environ, {"FAKE_EVAL_MARKER": "FastWAM: captured action-denoise graph"}):
            sr, _ = self.runner._run_gate()
        self.assertEqual(sr, 1.0)


class CompareArtifactArmTests(unittest.TestCase):
    def test_geak_is_the_comparison_arm_name(self):
        compare = load_module(COMPARE_ARTIFACTS)
        self.assertIn("geak", compare.ARM_SPECS)
        self.assertNotIn("optimized", compare.ARM_SPECS)


class ContainerWrapperTests(unittest.TestCase):
    def test_geak_wrapper_does_not_require_model_python_on_host(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            capture = root / "docker-args"
            docker = fake_bin / "docker"
            docker.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$DOCKER_CAPTURE"\n'
            )
            docker.chmod(0o755)

            model = root / "model"
            libero = root / "libero"
            hf_cache = root / "hf"
            home = root / "home"
            for directory in (model, libero, hf_cache, home):
                directory.mkdir()
            ckpt = root / "model.pt"
            stats = root / "stats.json"
            libero_config = root / ".libero"
            for file in (ckpt, stats, libero_config):
                file.touch()

            container_python = "/container/only/python"
            env = dict(
                os.environ,
                PATH=f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                HOME=str(home),
                DOCKER_CAPTURE=str(capture),
                EVAL_IMAGE="test-image",
                MODEL_SRC_ROOT=str(model),
                MODEL_CKPT=str(ckpt),
                DATASET_STATS=str(stats),
                LIBERO_ROOT=str(libero),
                LIBERO_CONFIG_PATH=str(libero_config),
                HF_CACHE_DIR=str(hf_cache),
                MODEL_PYTHON=container_python,
            )
            result = subprocess.run(
                ["bash", str(GEAK_LOCAL_RUNNER), "compile"],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(container_python, capture.read_text().splitlines())


    def _run_local(self, command, extra_env=None):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        root = Path(td.name)
        fake_bin = root / "bin"
        fake_bin.mkdir()
        capture = root / "docker-args"
        docker = fake_bin / "docker"
        docker.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$DOCKER_CAPTURE"\n')
        docker.chmod(0o755)
        for directory in ("model", "hf", "home"):
            (root / directory).mkdir()
        for file in ("model.pt", "stats.json"):
            (root / file).touch()
        env = {k: v for k, v in os.environ.items() if k not in ("LIBERO_ROOT", "LIBERO_CONFIG_PATH", "EVAL_DOCKER_ARGS")}
        env.update(PATH=f"{fake_bin}{os.pathsep}{os.environ['PATH']}", HOME=str(root / "home"),
                   DOCKER_CAPTURE=str(capture), EVAL_IMAGE="test-image", MODEL_SRC_ROOT=str(root / "model"),
                   MODEL_CKPT=str(root / "model.pt"), DATASET_STATS=str(root / "stats.json"),
                   HF_CACHE_DIR=str(root / "hf"), MODEL_PYTHON="/container/python", MUJOCO_GL="osmesa")
        env.update(extra_env or {})
        result = subprocess.run(["bash", str(GEAK_LOCAL_RUNNER), command], env=env,
                                capture_output=True, text=True, timeout=10)
        args = capture.read_text().splitlines() if capture.exists() else []
        return result, args

    def test_only_correctness_needs_libero(self):
        for command in ("compile", "oracle", "performance"):
            result, _ = self._run_local(command)
            self.assertEqual(result.returncode, 0, f"{command}: {result.stderr}")
        for command in ("correctness", "rollout"):
            result, _ = self._run_local(command)
            self.assertEqual(result.returncode, 2, command)
            self.assertIn("LIBERO_ROOT", result.stderr)
        self.assertNotIn("EVAL_HARNESS", GEAK_LOCAL_RUNNER.read_text())

    def test_eval_docker_args_reach_the_container(self):
        result, args = self._run_local("performance", {"EVAL_DOCKER_ARGS": "--name geak-perf-g3 --cpuset-cpus 0-7"})
        self.assertEqual(result.returncode, 0, result.stderr)
        joined = "\n".join(args)
        self.assertIn("--name\ngeak-perf-g3\n--cpuset-cpus\n0-7\n--workdir", joined)


if __name__ == "__main__":
    unittest.main()
