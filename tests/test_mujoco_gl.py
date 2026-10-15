# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo rendering (MUJOCO_GL): GPU OpenGL (EGL) by default, OSMesa fallback.

* scripts/eval_flags.py resolve_mujoco_gl: an explicit MUJOCO_GL is kept with
  no probe; a probe that creates an EGL context gives egl; a failed, crashed or
  timed-out probe gives osmesa with a short reason; the dry-run value is auto.
  The probe is the real snippet, run by this Python against stand-in mujoco
  modules.
* scripts/eval_flags.sh eval_mujoco_gl exports the same choice.
* No launcher or tool hard-codes MUJOCO_GL=osmesa any more: only the
  --sim instinct helpers, and fallbacks that honour a caller's setting.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
import uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
EVAL_FLAGS_SH = ROOT / "scripts" / "eval_flags.sh"


def load(path: Path):
    name = f"t_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FLAGS = load(ROOT / "scripts" / "eval_flags.py")

STUBS = {
    "ok": """
        class GLContext:
            def __init__(self, w, h): pass
            def make_current(self): pass
            def free(self): pass
    """,
    "no_libegl": """
        raise AttributeError("'NoneType' object has no attribute 'eglQueryString'")
    """,
    "compute_chip": """
        import sys
        print("radeonsi: error: can't create a graphics context on a compute chip", file=sys.stderr)
        class GLContext:
            def __init__(self, w, h):
                raise RuntimeError("EGL_BAD_ALLOC")
    """,
    "hang": """
        import time
        time.sleep(60)
    """,
    "crash": """
        import os
        os.abort()
    """,
}


class StubPython:
    """A temp dir whose mujoco.py is one of STUBS, and the env that imports it."""

    def __init__(self, td: Path, stub: str):
        (td / "mujoco.py").write_text(textwrap.dedent(STUBS[stub]))
        self.prefix = [sys.executable]
        self.env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(td),
                    "PYTHONNOUSERSITE": "1"}


def resolve(prefix, env=None, environ=None, timeout=None):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        resolved = FLAGS.resolve_mujoco_gl(prefix, env=env, environ=environ or {}, timeout=timeout)
    return resolved, err.getvalue()


class ResolveTests(unittest.TestCase):
    def test_explicit_setting_is_kept_without_a_probe(self):
        with mock.patch.object(FLAGS, "probe_egl", side_effect=AssertionError("probed")):
            for value, platform in (("egl", "egl"), ("osmesa", "osmesa"), ("glfw", None)):
                with self.subTest(value=value):
                    resolved, err = resolve(["/nonexistent"], environ={"MUJOCO_GL": value})
                    self.assertEqual(resolved, {"MUJOCO_GL": value, "PYOPENGL_PLATFORM": platform})
                    self.assertEqual(err.strip(), f"MuJoCo rendering: {value} (set by MUJOCO_GL)")
            resolved, err = resolve(["/nonexistent"], environ={
                "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", FLAGS.MUJOCO_GL_ANNOUNCED: "1"})
            self.assertEqual(resolved["MUJOCO_GL"], "osmesa")
            self.assertEqual(err, "")  # a launcher already printed it

    def test_probe_success_gives_egl(self):
        with tempfile.TemporaryDirectory() as td:
            stub = StubPython(Path(td), "ok")
            resolved, err = resolve(stub.prefix, env=stub.env)
        self.assertEqual(resolved, {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"})
        self.assertEqual(err.strip(), "MuJoCo rendering: egl (GPU OpenGL)")

    def test_probe_failures_give_osmesa_with_a_reason(self):
        cases = {
            "no_libegl": "libEGL not found",
            "compute_chip": "the GPU has no graphics engine",
            "hang": "probe timed out after 2s",
            "crash": "probe crashed (signal 6)",
        }
        for stub_name, reason in cases.items():
            with self.subTest(stub=stub_name), tempfile.TemporaryDirectory() as td:
                stub = StubPython(Path(td), stub_name)
                resolved, err = resolve(stub.prefix, env=stub.env, timeout=2)
                self.assertEqual(resolved, {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"})
                self.assertEqual(err.strip(), f"MuJoCo rendering: osmesa (EGL unavailable: {reason})")

    def test_a_stuck_launcher_and_a_missing_interpreter_fall_back(self):
        with tempfile.TemporaryDirectory() as td:
            stuck = Path(td) / "stuck"
            stuck.write_text("#!/bin/sh\nexec sleep 30\n")
            stuck.chmod(0o755)
            ok, reason = FLAGS.probe_egl([str(stuck)], timeout=1)
            self.assertFalse(ok)
            self.assertEqual(reason, "probe timed out after 2s")
        ok, reason = FLAGS.probe_egl(["/nonexistent/python"], timeout=1)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("cannot run /nonexistent/python"), reason)

    def test_dry_run_shows_auto(self):
        self.assertEqual(FLAGS.mujoco_gl_dry_run({}), "auto (EGL, falls back to OSMesa)")
        self.assertEqual(FLAGS.mujoco_gl_dry_run({"MUJOCO_GL": "osmesa"}), "osmesa")
        bash = subprocess.run(["bash", "-c", 'source "$1"; eval_mujoco_gl_dry_run', "_", str(EVAL_FLAGS_SH)],
                              capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
        self.assertEqual(bash.stdout.strip(), FLAGS.MUJOCO_GL_AUTO)

    def test_apply_sets_the_environment_and_marks_it_announced(self):
        environ = {"PYOPENGL_PLATFORM": "x"}
        FLAGS.apply_mujoco_gl({"MUJOCO_GL": "glfw", "PYOPENGL_PLATFORM": None}, environ)
        self.assertEqual(environ, {"MUJOCO_GL": "glfw", FLAGS.MUJOCO_GL_ANNOUNCED: "1"})


class BashHelperTests(unittest.TestCase):
    def run_helper(self, env: dict, *prefix: str):
        script = ('source "$1"; shift; eval_mujoco_gl "$@" || exit 9; '
                  'echo "gl=$MUJOCO_GL py=${PYOPENGL_PLATFORM-unset} announced=$EVAL_MUJOCO_GL_ANNOUNCED"')
        return subprocess.run(["bash", "-c", script, "_", str(EVAL_FLAGS_SH), *prefix],
                              capture_output=True, text=True, timeout=60,
                              env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **env})

    def test_probe_result_is_exported(self):
        for stub_name, expected in (("ok", "egl"), ("no_libegl", "osmesa")):
            with self.subTest(stub=stub_name), tempfile.TemporaryDirectory() as td:
                stub = StubPython(Path(td), stub_name)
                result = self.run_helper({}, "env", f"PYTHONPATH={td}", *stub.prefix)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), f"gl={expected} py={expected} announced=1")
                self.assertIn(f"MuJoCo rendering: {expected} (", result.stderr)

    def test_explicit_setting_is_kept_without_a_probe(self):
        result = self.run_helper({"MUJOCO_GL": "glfw"}, "/nonexistent/python")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "gl=glfw py=unset announced=1")
        self.assertEqual(result.stderr.strip(), "MuJoCo rendering: glfw (set by MUJOCO_GL)")

    def test_instinct_helper_exports_osmesa_silently(self):
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; eval_mujoco_gl_instinct; echo "$MUJOCO_GL $PYOPENGL_PLATFORM"',
             "_", str(EVAL_FLAGS_SH)], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin",
                                                                            "MUJOCO_GL": "egl"})
        self.assertEqual((result.stdout.strip(), result.stderr), ("osmesa osmesa", ""))


class DirectRobocasaRunnerTests(unittest.TestCase):
    """run_full24_evaluation.py (the documented direct runner) uses the shared check.

    The runner imports DiT4DiT's training stack, so its two MUJOCO_GL functions
    are taken from the source and run on their own.
    """

    RUNNER = ROOT / "DiT4DiT" / "examples" / "Robocasa_tabletop" / "eval_files" / "run_full24_evaluation.py"

    def functions(self, repo_root: Path):
        import ast
        import argparse

        tree = ast.parse(self.RUNNER.read_text())
        wanted = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name in ("client_pythonpath", "resolve_mujoco_gl")]
        self.assertEqual(len(wanted), 2)
        namespace = {"argparse": argparse, "os": os, "REPO_ROOT": repo_root}
        exec(compile(ast.Module(body=wanted, type_ignores=[]), str(self.RUNNER), "exec"), namespace)
        return namespace["resolve_mujoco_gl"]

    def args(self):
        return mock.Mock(python_bin="/nonexistent/python", robosuite_overlay="/o", robocasa_root="/r")

    def test_unset_probes_with_the_client_python(self):
        resolve = self.functions(self.RUNNER.parents[3])
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "EVAL_MUJOCO_GL_ANNOUNCED"):
                os.environ.pop(key, None)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                resolve(self.args())
            self.assertEqual(os.environ["MUJOCO_GL"], "osmesa")
            self.assertEqual(os.environ["PYOPENGL_PLATFORM"], "osmesa")
        self.assertIn("MuJoCo rendering: osmesa (EGL unavailable: cannot run /nonexistent/python", err.getvalue())

    def test_explicit_setting_is_kept(self):
        resolve = self.functions(self.RUNNER.parents[3])
        with mock.patch.dict(os.environ, {"MUJOCO_GL": "egl", "EVAL_MUJOCO_GL_ANNOUNCED": "1"}):
            os.environ.pop("PYOPENGL_PLATFORM", None)
            resolve(self.args())
            self.assertEqual((os.environ["MUJOCO_GL"], os.environ["PYOPENGL_PLATFORM"]), ("egl", "egl"))

    def test_outside_the_repository_it_leaves_the_osmesa_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            resolve = self.functions(Path(tmp) / "DiT4DiT")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MUJOCO_GL", None)
                resolve(self.args())
                self.assertNotIn("MUJOCO_GL", os.environ)


class NoHardCodedOsmesaTests(unittest.TestCase):
    """MUJOCO_GL / PYOPENGL_PLATFORM are never pinned to osmesa outside the allowed places."""

    ASSIGNMENT = re.compile(r"(MUJOCO_GL|PYOPENGL_PLATFORM)\W{0,6}osmesa")
    # path -> why osmesa may appear there
    ALLOWED = {
        # --sim instinct: the GPU simulator renders; its helper contexts stay on OSMesa.
        "scripts/eval_flags.sh": "eval_mujoco_gl_instinct",
        "simulate/python/instinct_sim_libero/success_metadata.py": "instinct helper (setdefault)",
        "simulate/python/instinct_sim_render_worker.py": "instinct helper (setdefault)",
        # Standalone fallbacks that honour a caller's MUJOCO_GL (setdefault).
        "DiT4DiT/examples/Robocasa_tabletop/eval_files/run_full24_evaluation.py": "standalone runner",
        "geak/task/scripts/task_runner.py": "run_local.sh forwards the resolved choice",
    }

    def files(self):
        listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT,
                                capture_output=True, text=True)
        if listed.returncode == 0:
            return [ROOT / line for line in listed.stdout.splitlines()]
        return [p for p in ROOT.rglob("*") if ".git" not in p.parts]

    def test_no_hard_coded_osmesa(self):
        offenders = []
        for path in self.files():
            rel = path.relative_to(ROOT).as_posix()
            if path.suffix not in (".sh", ".py", ".md") or rel.startswith("tests/") or not path.is_file():
                continue
            for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if path.suffix == ".md" and not line.lstrip().startswith("export "):
                    continue  # prose may name MUJOCO_GL=osmesa as the way to force it
                if not self.ASSIGNMENT.search(line):
                    continue
                if rel in self.ALLOWED and (rel == "scripts/eval_flags.sh" or "setdefault" in line):
                    continue
                offenders.append(f"{rel}:{number}: {line.strip()}")
        self.assertEqual(offenders, [])

    def test_allowed_places_still_hold_their_osmesa(self):
        for rel in self.ALLOWED:
            with self.subTest(path=rel):
                self.assertRegex((ROOT / rel).read_text(), self.ASSIGNMENT)

    def test_no_launcher_exports_egl_into_an_instinct_run(self):
        native = (ROOT / "eval" / "native" / "run_fastwam_libero.sh").read_text()
        self.assertRegex(native, r'if \[\[ "\$SIM" == instinct \]\]; then\n(  #.*\n)*  eval_mujoco_gl_instinct\n')
        robocasa = (ROOT / "scripts" / "eval_dit4dit_robocasa.sh").read_text()
        self.assertIn('instinct_sim_client_env robocasa "$robocasa_python"', robocasa)
        self.assertRegex(robocasa, r"instinct_sim_client_env robocasa[^\n]*\n(  #.*\n)*  eval_mujoco_gl_instinct\n")
        libero = (ROOT / "scripts" / "eval_dit4dit_libero.sh").read_text()
        self.assertRegex(libero, r'if \[\[ "\$sim" == mujoco \]\]; then\n  eval_mujoco_gl ')


if __name__ == "__main__":
    unittest.main()
