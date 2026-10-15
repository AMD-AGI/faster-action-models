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
SCRIPT = ROOT / "scripts" / "cae_fastwam_latency.py"


def _load():
    spec = importlib.util.spec_from_file_location("cae_fastwam_latency", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CaeFastwamLatencyHarnessTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load()

    def test_every_arm_points_at_a_real_tree(self):
        for name, (tree, *_rest) in self.mod.ARM_SPECS.items():
            with self.subTest(arm=name):
                self.assertTrue((ROOT / "src" / tree / "fastwam").is_dir(), tree)

    def test_only_the_cae_arm(self):
        self.assertEqual(set(self.mod.ARM_SPECS), {"cae"})

    def test_default_arms_parse(self):
        cells = [self.mod.parse_arm(t, 10) for t in self.mod.DEFAULT_ARMS.split(",")]
        self.assertEqual(cells, [("cae", 10), ("cae", 4)])

    def test_arm_step_suffix(self):
        self.assertEqual(self.mod.parse_arm("cae@4", 10), ("cae", 4))
        self.assertEqual(self.mod.parse_arm("cae", 10), ("cae", 10))
        for bad in ("nope", "nope@4", "cae@", "cae@0", "cae@x", "eager", "geak"):
            with self.subTest(token=bad), self.assertRaises(ValueError):
                self.mod.parse_arm(bad, 10)

    def test_worker_argv_compiles_the_cae_tree(self):
        a = argparse.Namespace(text_encoder=False, warmup=25, iters=40, suite_id=0)
        argv = self.mod.worker_argv("cae", 4, a)
        self.assertEqual(argv[argv.index("--compile-infer") + 1], "1")
        self.assertEqual(argv[argv.index("--steps") + 1], "4")
        self.assertTrue(argv[argv.index("--expect-src") + 1].endswith("cae-fastwam"))

    def test_missing_env_fails_before_launch(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("MODEL_CKPT", "ACTION_VAE_CKPT", "EVAL_IMAGE")}
        p = subprocess.run([sys.executable, str(SCRIPT)], env=env,
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("MODEL_CKPT", p.stderr)

    def test_unknown_arm_fails_before_launch(self):
        p = subprocess.run([sys.executable, str(SCRIPT), "--arms", "cae,bogus"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("bogus", p.stderr)

    def test_worker_compiles(self):
        # Compile to a temporary file so the test leaves no scripts/__pycache__.
        with tempfile.TemporaryDirectory() as tmp:
            py_compile.compile(str(ROOT / "scripts" / "cae_fastwam_latency_worker.py"),
                               cfile=os.path.join(tmp, "worker.pyc"), doraise=True)


if __name__ == "__main__":
    unittest.main()
