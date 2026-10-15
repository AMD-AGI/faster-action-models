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
EVAL_RUNNER = ROOT / "eval" / "run_eval.sh"
GEAK_LOCAL_RUNNER = ROOT / "geak" / "task" / "run_local.sh"
EDITABLE = ("mot.py", "wan_video_dit.py", "wan_video_vae.py", "action_dit.py")


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

        self.ckpt = self.model / "checkpoints" / "model.pt"
        self.stats = self.model / "checkpoints" / "stats.json"
        self.ckpt.touch()
        self.stats.write_text("{}\n")
        self.harness = root / "harness"
        (self.harness / "src").mkdir(parents=True)
        self.libero = root / "libero"
        self.libero.mkdir()
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
            "EVAL_HARNESS": str(self.harness),
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

    def test_suite_horizons_match_fastwam_protocol(self):
        for suite in ("libero_spatial", "libero_object", "libero_goal"):
            self.assertEqual(self.runner._gate_max_steps(suite), 400)
        self.assertEqual(self.runner._gate_max_steps("libero_10"), 700)


class CompareArtifactArmTests(unittest.TestCase):
    def test_geak_is_the_comparison_arm_name(self):
        compare = load_module(COMPARE_ARTIFACTS)
        self.assertIn("geak", compare.ARM_SPECS)
        self.assertNotIn("optimized", compare.ARM_SPECS)


class ContainerWrapperTests(unittest.TestCase):
    def test_eval_runner_builds_safe_same_path_mount_arguments(self):
        script = EVAL_RUNNER.read_text()
        self.assertIn('local -a args=("${EVAL_DOCKER_ARGS[@]}")', script)
        self.assertIn('docker run "${args[@]}"', script)
        for variable in (
            "MODEL_CKPT",
            "DATASET_STATS",
            "MODEL_CONFIGS",
            "EVAL_HARNESS",
            "LIBERO_ROOT",
            "BASELINE_SRC",
            "OPTIMIZED_SRC",
        ):
            self.assertIn(variable, script)
        self.assertIn('eval_bind_path "$EVAL_OUTPUT_ROOT" rw', script)
        self.assertIn('eval_bind_path "$CHOME" rw', script)

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
            harness = root / "harness"
            libero = root / "libero"
            hf_cache = root / "hf"
            home = root / "home"
            for directory in (model, harness, libero, hf_cache, home):
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
                EVAL_HARNESS=str(harness),
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


if __name__ == "__main__":
    unittest.main()
