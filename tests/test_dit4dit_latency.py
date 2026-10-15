# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import importlib.util
import os
import py_compile
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dit4dit_latency.py"
ENV_VARS = ("DIT4DIT_LIBERO_CKPT", "DIT4DIT_ROBOCASA_CKPT", "EVAL_IMAGE", "EVAL_GPUS",
            "DIT4DIT_DIFFUSERS_SRC", "DIT4DIT_PYOVERLAY")


def _load():
    spec = importlib.util.spec_from_file_location("dit4dit_latency", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in ENV_VARS}
    env.update(extra)
    return env


def _envs(argv):
    return [argv[i + 1] for i, x in enumerate(argv) if x == "--env"]


class DiT4DiTLatencyHarnessTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load()
        self.env = {"EVAL_IMAGE": "img:tag"}
        self.work = "/tmp/dit4dit_latency_test"
        self.ckpt = "/ckpts/dit4dit_libero/final_model/pytorch_model.pt"

    def test_every_arm_points_at_a_real_tree(self):
        for name, (tree, _desc) in self.mod.ARM_SPECS.items():
            with self.subTest(arm=name):
                self.assertTrue((ROOT / "src" / tree / "DiT4DiT").is_dir(), tree)
                self.assertTrue((ROOT / "src" / tree / "examples" / "LIBERO" / "eval_files"
                                 / "model2libero_interface.py").is_file())
                self.assertTrue((ROOT / "src" / tree / "examples" / "Robocasa_tabletop"
                                 / "eval_files" / "model2robocasa_interface.py").is_file())

    def test_default_arms_parse(self):
        cells = [self.mod.parse_arm(t, 4) for t in self.mod.DEFAULT_ARMS.split(",")]
        self.assertEqual(cells, [("baseline", 4), ("compile", 4), ("geak", 4), ("geak", 2)])

    def test_compile_arm_is_baseline_steps_only(self):
        self.assertEqual(self.mod.parse_arm("compile@4", 4), ("compile", 4))
        self.assertEqual(self.mod.parse_arm("compile", 4), ("compile", 4))
        for bad in ("compile@2", "compile"):
            with self.subTest(token=bad), self.assertRaises(ValueError) as cm:
                self.mod.parse_arm(bad, 2)
            self.assertIn("baseline step count", str(cm.exception))

    def test_compile_arm_refused_at_two_steps_before_launch(self):
        p = subprocess.run([sys.executable, str(SCRIPT), "--bench", "libero", "--dry-run",
                            "--arms", "baseline@4,compile@2"], env=_clean_env(),
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("compile@2", p.stderr)
        self.assertIn("baseline step count", p.stderr)

    def test_compile_arm_runs_the_baseline_tree_with_its_own_cache(self):
        argv = self.mod.docker_argv("compile", 4, "6", self.env, self.work, self.ckpt)
        envs = _envs(argv)
        tree = str(ROOT / "src" / "dit4dit-baseline")
        for gate in self.mod.GATES:
            self.assertIn(f"{gate}=0", envs)
        self.assertIn("DIT4DIT_INFER_STEPS=4", envs)
        cache = f"{self.work}/cache/dit4dit-baseline-compile"
        self.assertIn(f"TORCHINDUCTOR_CACHE_DIR={cache}/inductor", envs)
        self.assertIn(f"TRITON_CACHE_DIR={cache}/triton", envs)
        pp = [e for e in envs if e.startswith("PYTHONPATH=")][0][len("PYTHONPATH="):]
        self.assertEqual(pp.split(":")[:2], [tree, f"{self.work}/overlays/baseline"])
        self.assertEqual(argv[argv.index("--workdir") + 1], tree)
        caches = {self.mod.cache_root(self.work, n) for n in self.mod.ARM_SPECS}
        self.assertEqual(len(caches), len(self.mod.ARM_SPECS))

    def test_worker_argv_carries_the_naive_compile_switch(self):
        a = argparse.Namespace(warmup=25, iters=40, work_dir=self.work)
        argv = self.mod.worker_argv("libero", "compile", 4, a, self.ckpt)
        self.assertEqual(argv[argv.index("--naive-compile") + 1], "modules")
        self.assertEqual(argv[argv.index("--expect-tree") + 1],
                         str(ROOT / "src" / "dit4dit-baseline"))
        self.assertEqual(argv[argv.index("--expect-diffusers") + 1],
                         f"{self.work}/overlays/baseline/diffusers")
        for name in ("baseline", "geak"):
            argv = self.mod.worker_argv("libero", name, 4, a, self.ckpt)
            self.assertEqual(argv[argv.index("--naive-compile") + 1], "none", name)

    def test_worker_accepts_the_naive_compile_switch(self):
        text = (ROOT / "scripts" / "dit4dit_latency_worker.py").read_text()
        self.assertIn('choices=("none", "modules")', text)
        for target in ("text_encoder.compile()", "transformer.compile()",
                       "torch.compile(ext.vae.encode)", "action_model.model.compile()"):
            self.assertIn(target, text)

    def test_arm_step_suffix(self):
        self.assertEqual(self.mod.parse_arm("geak@2", 4), ("geak", 2))
        self.assertEqual(self.mod.parse_arm("baseline", 4), ("baseline", 4))
        for bad in ("nope", "nope@4", "eager@4", "geak@", "geak@0", "geak@x", "geak@10"):
            with self.subTest(token=bad), self.assertRaises(ValueError):
                self.mod.parse_arm(bad, 4)

    def test_worker_argv_names_the_arm_tree_overlay_and_steps(self):
        a = argparse.Namespace(warmup=25, iters=40, work_dir=self.work)
        argv = self.mod.worker_argv("robocasa", "geak", 2, a, self.ckpt)
        self.assertEqual(argv[argv.index("--bench") + 1], "robocasa")
        self.assertEqual(argv[argv.index("--steps") + 1], "2")
        self.assertEqual(argv[argv.index("--expect-tree") + 1], str(ROOT / "src" / "dit4dit-geak"))
        self.assertEqual(argv[argv.index("--expect-diffusers") + 1],
                         f"{self.work}/overlays/geak/diffusers")
        self.assertEqual(argv[argv.index("--warmup") + 1], "25")
        self.assertEqual(argv[argv.index("--iters") + 1], "40")

    def test_docker_argv_pins_gates_steps_and_per_tree_cache(self):
        for name, steps in (("baseline", 4), ("geak", 2)):
            with self.subTest(arm=name):
                argv = self.mod.docker_argv(name, steps, "6", self.env, self.work, self.ckpt)
                envs = _envs(argv)
                for gate in self.mod.GATES:
                    self.assertIn(f"{gate}=0", envs)
                self.assertIn(f"DIT4DIT_INFER_STEPS={steps}", envs)
                self.assertIn("TOKENIZERS_PARALLELISM=false", envs)
                self.assertIn("HIP_VISIBLE_DEVICES=6", envs)
                self.assertIn("CUDA_VISIBLE_DEVICES=6", envs)
                tree = str(ROOT / "src" / f"dit4dit-{name}")
                cache = f"{self.work}/cache/dit4dit-{name}"
                self.assertIn(f"TORCHINDUCTOR_CACHE_DIR={cache}/inductor", envs)
                self.assertIn(f"TRITON_CACHE_DIR={cache}/triton", envs)
                pp = [e for e in envs if e.startswith("PYTHONPATH=")][0][len("PYTHONPATH="):]
                self.assertEqual(pp.split(":")[:2], [tree, f"{self.work}/overlays/{name}"])
                self.assertEqual(argv[argv.index("--workdir") + 1], tree)
                self.assertEqual(argv[-1], "img:tag")
                self.assertIn(f"{ROOT}:{ROOT}:ro", argv)
                self.assertIn("/ckpts/dit4dit_libero:/ckpts/dit4dit_libero:ro", argv)

    def test_trees_never_share_a_compile_cache(self):
        self.assertNotEqual(self.mod.cache_root(self.work, "baseline"),
                            self.mod.cache_root(self.work, "geak"))

    def test_pyoverlay_follows_the_diffusers_overlay(self):
        env = dict(self.env, DIT4DIT_PYOVERLAY="/ovl/a:/ovl/b")
        argv = self.mod.docker_argv("geak", 4, "0", env, self.work, self.ckpt)
        pp = [e for e in _envs(argv) if e.startswith("PYTHONPATH=")][0]
        self.assertTrue(pp.endswith(f"{self.work}/overlays/geak:/ovl/a:/ovl/b"), pp)
        self.assertIn("/ovl/a:/ovl/a:ro", argv)

    def test_overlay_is_built_by_the_launchers_builder(self):
        for name in ("baseline", "geak"):
            with self.subTest(arm=name):
                argv = self.mod.overlay_argv(name, self.env, self.work)
                self.assertIn(str(ROOT / "scripts" / "dit4dit_diffusers_overlay.sh"), argv)
                self.assertEqual(argv[-2:], [name, f"{self.work}/overlays/{name}"])
                self.assertTrue(any("build_diffusers_overlay" in x for x in argv))
        env = dict(self.env, DIT4DIT_DIFFUSERS_SRC="/src/diffusers")
        argv = self.mod.overlay_argv("geak", env, self.work)
        self.assertIn("DIT4DIT_DIFFUSERS_SRC=/src/diffusers", _envs(argv))
        self.assertIn("/src/diffusers:/src/diffusers:ro", argv)

    def test_dry_run_needs_no_assets(self):
        p = subprocess.run([sys.executable, str(SCRIPT), "--bench", "libero", "--dry-run",
                            "--gpus", "5"], env=_clean_env(), capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("baseline@4, compile@4, geak@4, geak@2", p.stdout)
        self.assertEqual(p.stdout.count("# cell "), 4)
        self.assertEqual(p.stdout.count("# diffusers overlay, "), 2)   # compile reuses baseline
        self.assertIn("HIP_VISIBLE_DEVICES=5", p.stdout)
        self.assertIn("DIT4DIT_INFER_STEPS=2", p.stdout)
        self.assertNotIn("--volume :", p.stdout)   # no mount for a placeholder path

    def test_gpus_flag(self):
        p = subprocess.run([sys.executable, str(SCRIPT), "--bench", "libero", "--dry-run"],
                           env=_clean_env(EVAL_GPUS="3"), capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("gpus=3", p.stdout)
        p = subprocess.run([sys.executable, str(SCRIPT), "--bench", "libero", "--dry-run",
                            "--gpus", "0,1"], env=_clean_env(), capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("one GPU id", p.stderr)
        p = subprocess.run([sys.executable, str(SCRIPT), "--bench", "libero", "--dry-run",
                            "--gpu", "2"], env=_clean_env(), capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("gpus=2", p.stdout)
        self.assertIn("deprecated", p.stderr)

    def test_missing_env_fails_before_launch(self):
        for bench, var in (("libero", "DIT4DIT_LIBERO_CKPT"),
                           ("robocasa", "DIT4DIT_ROBOCASA_CKPT")):
            with self.subTest(bench=bench):
                p = subprocess.run([sys.executable, str(SCRIPT), "--bench", bench],
                                   env=_clean_env(), capture_output=True, text=True)
                self.assertEqual(p.returncode, 2)
                self.assertIn(var, p.stderr)
                self.assertIn("EVAL_IMAGE", p.stderr)

    def test_bad_checkpoint_fails_before_launch(self):
        p = subprocess.run([sys.executable, str(SCRIPT), "--bench", "libero"],
                           env=_clean_env(DIT4DIT_LIBERO_CKPT="/nonexistent/x.pt",
                                          EVAL_IMAGE="img"),
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("DIT4DIT_LIBERO_CKPT", p.stderr)

    def test_unknown_arm_fails_before_launch(self):
        p = subprocess.run([sys.executable, str(SCRIPT), "--bench", "libero",
                            "--arms", "geak,bogus"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("bogus", p.stderr)

    def test_bench_is_required(self):
        p = subprocess.run([sys.executable, str(SCRIPT), "--dry-run"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("--bench", p.stderr)

    def test_worker_compiles(self):
        # Compile to a temporary file so the test leaves no scripts/__pycache__.
        with tempfile.TemporaryDirectory() as tmp:
            py_compile.compile(str(ROOT / "scripts" / "dit4dit_latency_worker.py"),
                               cfile=os.path.join(tmp, "worker.pyc"), doraise=True)

    def test_worker_pins_transformers_and_tokenizers_from_each_tree(self):
        spec = importlib.util.spec_from_file_location(
            "dit4dit_latency_worker", ROOT / "scripts" / "dit4dit_latency_worker.py")
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        for tree in ("dit4dit-baseline", "dit4dit-geak"):
            pins = worker.requirement_pins(str(ROOT / "src" / tree))
            self.assertEqual(set(pins), {"transformers", "tokenizers"}, tree)
            self.assertTrue(all(pins.values()), tree)


if __name__ == "__main__":
    unittest.main()
