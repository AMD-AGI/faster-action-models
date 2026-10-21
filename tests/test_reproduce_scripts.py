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

    def test_help_dry_run_and_environment_check(self):
        # --help prints the header, --dry-run resolves every launch without assets, and a run with
        # nothing set fails the environment check before anything starts.
        with tempfile.TemporaryDirectory() as out:
            env = {"PATH": "/usr/bin:/bin", "HOME": out, "REPRO_DIR": out}
            for name in SCRIPTS:
                script = str(ROOT / "scripts" / name)
                with self.subTest(script=name, flag="--help"):
                    r = subprocess.run(["bash", script, "--help"], env=env, capture_output=True, text=True)
                    self.assertEqual(r.returncode, 0, r.stderr)
                    self.assertIn("Runtime", r.stdout)
                with self.subTest(script=name, flag="--dry-run"):
                    r = subprocess.run(["bash", script, "--dry-run"], env=env, capture_output=True, text=True)
                    self.assertEqual(r.returncode, 0, r.stderr[-800:])
                with self.subTest(script=name, flag="none"):
                    r = subprocess.run(["bash", script], env=env, capture_output=True, text=True, timeout=60)
                    self.assertEqual(r.returncode, 2, r.stderr[-800:])
                    self.assertIn("environment check failed", r.stderr)

    def test_smoke_appends_smoke_to_every_launch_and_one_round_to_latency(self):
        smoke_dir = ROOT / "eval" / "outputs" / "reproduce_cae_fastwam-smoke"
        if not smoke_dir.exists():  # the script creates its default results directory; leave no trace
            self.addCleanup(lambda: smoke_dir.is_dir() and not any(smoke_dir.iterdir()) and smoke_dir.rmdir())
        with tempfile.TemporaryDirectory() as out:
            env = {"PATH": "/usr/bin:/bin", "HOME": out}
            r = subprocess.run(["bash", str(ROOT / "scripts" / "reproduce_cae_fastwam.sh"), "--dry-run", "--smoke"],
                               cwd=out, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            trials = re.findall(r"^trials=(\d+)$", r.stdout, re.M)
            self.assertEqual(trials, ["1"] * 4, r.stdout)
            outputs = re.findall(r"^output=(.*)$", r.stdout, re.M)
            self.assertTrue(outputs)
            for path in outputs:
                self.assertIn("/eval/outputs/reproduce_cae_fastwam-smoke/", path)
            latency = [l for l in r.stdout.splitlines() if l.startswith("dry run, not timing:")]
            self.assertEqual(len(latency), 2)
            for line in latency:
                self.assertTrue(line.endswith("--rounds 1"), line)

    def test_only_keeps_the_matching_launches(self):
        with tempfile.TemporaryDirectory() as out:
            env = {"PATH": "/usr/bin:/bin", "HOME": out, "REPRO_DIR": out}
            r = subprocess.run(["bash", str(ROOT / "scripts" / "reproduce_cae_fastwam.sh"), "--dry-run",
                                "--only", "arm cae|cae-fastwam"], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            self.assertEqual(re.findall(r"^arm=(\w+)$", r.stdout, re.M), ["cae", "cae"])
            self.assertEqual(r.stdout.count("skipped (--only"), 3)  # baseline, geak and the FastWAM latency
            self.assertIn("dry run, not timing: python3 scripts/cae_fastwam_latency.py", r.stdout)

    def test_latency_tools_time_on_the_first_gpu(self):
        script = ('source scripts/reproduce_common.sh; REPRO_DRY_RUN=0; '
                  'EVAL_GPUS=3,4,6 latency bash -c \'echo "gpus=$EVAL_GPUS $*"\' _ --x')
        r = subprocess.run(["bash", "-c", script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "gpus=3 --x")

    def test_no_script_needs_a_harness(self):
        for name in SCRIPTS + ("reproduce_common.sh",):
            with self.subTest(script=name):
                text = (ROOT / "scripts" / name).read_text()
                self.assertNotIn("HARNESS", text.replace("DIT4DIT_HARNESS_ROOT", ""))
                self.assertNotIn("vla-evaluation-harness", text)


if __name__ == "__main__":
    unittest.main()
