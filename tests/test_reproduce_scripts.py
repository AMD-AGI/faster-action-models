# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The reproduction scripts: every launch resolves, and each CPU MuJoCo launch has its GPU-native twin."""
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("reproduce_baselines.sh", "reproduce_training_free.sh", "reproduce_cae_fastwam.sh",
           "reproduce_cae_dit4dit.sh", "run_all_evals.sh")
LAUNCHERS = ("scripts/eval_fastwam_libero.sh", "eval/native/run_fastwam_libero.sh",
             "scripts/eval_dit4dit_libero.sh", "scripts/eval_dit4dit_robocasa.sh")


def launches(text):
    """(line, commented) for every evaluation launch in a script."""
    out = []
    for line in text.splitlines():
        body = line[2:] if line.startswith("# ") else line
        if any(f"bash {l} " in body for l in LAUNCHERS):
            out.append((body, line.startswith("# ")))
    return out


class ReproduceScriptTests(unittest.TestCase):
    def test_scripts_parse(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)

    def test_each_mujoco_launch_has_a_commented_instinct_twin(self):
        for name in SCRIPTS:
            pairs = launches((ROOT / "scripts" / name).read_text())
            self.assertTrue(pairs, name)
            for (active, ac), (twin, tc) in zip(pairs[::2], pairs[1::2]):
                with self.subTest(script=name, line=active):
                    self.assertFalse(ac)
                    self.assertTrue(tc)
                    self.assertIn("--sim mujoco", active)
                    self.assertIn("--sim instinct", twin)
                    norm = lambda s: re.sub(r"(--sim \w+|-mujoco\"|-instinct\"|scripts/eval_fastwam_libero\.sh|eval/native/run_fastwam_libero\.sh)", "", s)
                    self.assertEqual(norm(active), norm(twin))

    def test_every_launch_resolves_with_dry_run(self):
        # Each launch (MuJoCo and the commented GPU-native twin) must get past argument
        # validation: run it with --dry-run, which needs no assets.
        with tempfile.TemporaryDirectory() as out:
            env = {"PATH": "/usr/bin:/bin", "HOME": out, "REPRO_DIR": out}
            for name in SCRIPTS:
                for line, _ in launches((ROOT / "scripts" / name).read_text()):
                    cmd = line[line.index("bash "):].replace("$R", out).replace('"', "")
                    with self.subTest(script=name, line=cmd):
                        r = subprocess.run(cmd.split() + ["--dry-run"], cwd=ROOT, env=env,
                                           capture_output=True, text=True)
                        self.assertEqual(r.returncode, 0, r.stderr[-800:])


if __name__ == "__main__":
    unittest.main()
