# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SCRIPTS = {
    "FastWAM LIBERO": ROOT / "scripts" / "eval_fastwam_libero.sh",
    "DiT4DiT LIBERO": ROOT / "scripts" / "eval_dit4dit_libero.sh",
    "DiT4DiT RoboCasa": ROOT / "scripts" / "eval_dit4dit_robocasa.sh",
}
TRAINING_SCRIPTS = {
    "FastWAM LIBERO K8/d16 Action VAE": (
        ROOT / "scripts" / "train_fastwam_libero_action_vae.sh"
    ),
    "Compact Action DiT": (
        ROOT / "scripts" / "train_fastwam_libero_latent_policy.sh"
    ),
    "DiT4DiT RoboCasa full-24 K8/d48 Action VAE": (
        ROOT / "scripts" / "train_dit4dit_robocasa_action_vae.sh"
    ),
    "200K DiT4DiT RoboCasa": (
        ROOT / "scripts" / "train_dit4dit_robocasa_latent_policy.sh"
    ),
}
SCRIPTS = {**EVALUATION_SCRIPTS, **TRAINING_SCRIPTS}


class ModelLauncherScriptTests(unittest.TestCase):
    def test_scripts_have_valid_bash_syntax(self):
        for name, script in SCRIPTS.items():
            with self.subTest(name=name):
                result = subprocess.run(
                    ["bash", "-n", str(script)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_help_does_not_require_runtime_assets(self):
        for name, script in SCRIPTS.items():
            with self.subTest(name=name):
                result = subprocess.run(
                    ["bash", str(script), "--help"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(name, result.stdout)

    def test_invalid_protocol_sizes_fail_before_launch(self):
        cases = (
            ("eval_fastwam_libero.sh", ("--trials", "0"), "positive integer"),
            ("eval_dit4dit_libero.sh", ("--port", "0"), "1 to 65535"),
            (
                "eval_dit4dit_robocasa.sh",
                ("--trials", "2"),
                "one of 1, 10, or 50",
            ),
            (
                "eval_dit4dit_robocasa.sh",
                ("--clients-per-gpu", "0"),
                "positive integer",
            ),
        )
        for filename, arguments, expected in cases:
            with self.subTest(script=filename):
                result = subprocess.run(
                    ["bash", str(ROOT / "scripts" / filename), *arguments],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)


if __name__ == "__main__":
    unittest.main()
