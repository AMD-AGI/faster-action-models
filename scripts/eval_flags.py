# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Python tools' share of scripts/eval_flags.sh: one meaning per flag.

Also the one MUJOCO_GL resolution (EGL, falling back to OSMesa) that the bash
launchers run as `python3 scripts/eval_flags.py mujoco-gl -- PREFIX...`.
"""
from __future__ import annotations

import os
import subprocess
import sys


def deprecated(old: str, new: str) -> None:
    """eval_deprecated: warn that OLD is a deprecated spelling of NEW."""
    print(f"warning: {old} is deprecated and will be removed in the next release; use {new}",
          file=sys.stderr)


def resolve_one_gpu(gpus: str | None, gpu: str | None = None) -> str:
    """--gpus (default EVAL_GPUS, else 0) for a tool that runs on one GPU.

    GPU is the deprecated --gpu spelling; an explicit --gpus wins over it.
    Raises ValueError unless the result is exactly one GPU id.
    """
    if gpu is not None:
        deprecated("--gpu", "--gpus")
        if gpus is None:
            gpus = gpu
    if gpus is None:
        gpus = os.environ.get("EVAL_GPUS") or "0"
    ids = gpus.split(",")
    if not all(i.isdigit() for i in ids):
        raise ValueError(f"--gpus must be a comma-separated list of GPU ids, got: {gpus}")
    if len(ids) != 1:
        raise ValueError(f"this tool times on one GPU; pass one GPU id to --gpus, got: {gpus}")
    return ids[0]


# ---------------------------------------------------------------------------
# MuJoCo rendering backend (MUJOCO_GL)
#
# CPU MuJoCo (--sim mujoco) keeps physics on the CPU and renders camera images
# with OpenGL. The default is GPU OpenGL through EGL; when no EGL context can be
# created in the simulator's own environment (no libEGL in it, or a GPU without
# a graphics engine) it falls back to OSMesa, software OpenGL on the CPU. An
# explicit MUJOCO_GL is always honoured unchanged and never probed. The bash
# launchers reach this through eval_mujoco_gl in scripts/eval_flags.sh.
# ---------------------------------------------------------------------------
MUJOCO_GL_AUTO = "auto (EGL, falls back to OSMesa)"
MUJOCO_GL_PROBE_TIMEOUT = 30  # seconds inside the probe; process startup gets as long again
MUJOCO_GL_ANNOUNCED = "EVAL_MUJOCO_GL_ANNOUNCED"

# Run as `<python> -c EGL_PROBE <timeout>` in the simulator's environment. It
# creates the contexts the simulator will: MuJoCo's own (mujoco picks its GL
# backend at import) and, when robosuite is installed, robosuite's, which
# LIBERO and RoboCasa render through and which picks its EGL device from
# CUDA_VISIBLE_DEVICES.
EGL_PROBE = r"""
import faulthandler, os, sys
faulthandler.dump_traceback_later(int(sys.argv[1]), exit=True)  # also when stuck inside the GL driver
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
try:
    import mujoco
    context = mujoco.GLContext(64, 64)
    context.make_current()
    context.free()
    try:
        from robosuite.renderers.context.egl_context import EGLGLContext
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("robosuite"):
            raise
    else:
        context = EGLGLContext(64, 64, -1)  # device -1: CUDA_VISIBLE_DEVICES, as the clients
        context.make_current()
        context.free()
except BaseException as exc:
    message = (str(exc).strip().splitlines() or [""])[0]
    print(f"EGL_PROBE_FAIL {type(exc).__name__}: {message}", flush=True)
    os._exit(1)
print("EGL_PROBE_OK", flush=True)
os._exit(0)
"""


def mujoco_gl_dry_run(environ=None) -> str:
    """The dry-run value: the explicit MUJOCO_GL, else the auto default."""
    environ = os.environ if environ is None else environ
    return environ.get("MUJOCO_GL") or MUJOCO_GL_AUTO


def _probe_reason(output: str) -> str:
    """A short reason from a failed probe's combined output."""
    if "compute chip" in output:
        return "the GPU has no graphics engine"
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    marked = [line[len("EGL_PROBE_FAIL "):] for line in lines if line.startswith("EGL_PROBE_FAIL ")]
    reason = marked[-1] if marked else (lines[-1] if lines else "")
    if "eglQueryString" in reason or "libEGL" in reason:
        return "libEGL not found"
    return reason[:120] or "no EGL context"


def probe_egl(prefix, env=None, timeout: int | None = None, cwd=None) -> tuple[bool, str]:
    """(True, "") when PREFIX (a command that runs Python in the simulator's
    environment) can create an EGL context, else (False, short reason)."""
    if timeout is None:
        timeout = int(os.environ.get("EVAL_MUJOCO_GL_PROBE_TIMEOUT") or MUJOCO_GL_PROBE_TIMEOUT)
    try:
        result = subprocess.run([*prefix, "-c", EGL_PROBE, str(timeout)], env=env, cwd=cwd,
                                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                errors="replace", timeout=2 * timeout)
    except subprocess.TimeoutExpired:
        return False, f"probe timed out after {2 * timeout}s"
    except OSError as exc:
        return False, f"cannot run {prefix[0]}: {exc.strerror or exc}"
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode == 0 and "EGL_PROBE_OK" in result.stdout:
        return True, ""
    if "Timeout (" in output and "EGL_PROBE_FAIL" not in output:  # the probe's own deadline
        return False, f"probe timed out after {timeout}s"
    if "EGL_PROBE_FAIL" in output or "compute chip" in output:
        return False, _probe_reason(output)
    code = result.returncode
    if code < 0 or code > 128:
        signal_number = -code if code < 0 else code - 128
        return False, f"probe crashed (signal {signal_number})"
    return False, f"probe exited with status {code}: {_probe_reason(output)}"


def _pyopengl_for(mujoco_gl: str, environ) -> str | None:
    if environ.get("PYOPENGL_PLATFORM"):
        return environ["PYOPENGL_PLATFORM"]
    return mujoco_gl if mujoco_gl in ("egl", "osmesa") else None


def resolve_mujoco_gl(prefix, env=None, timeout: int | None = None, cwd=None,
                      environ=None, announce: bool = True) -> dict[str, str | None]:
    """MUJOCO_GL and PYOPENGL_PLATFORM for a run, printing the choice once.

    ENVIRON (default os.environ) holds the user's settings; an explicit
    MUJOCO_GL is returned unchanged without a probe (its PYOPENGL_PLATFORM is the
    user's, else egl/osmesa to match, else None: leave it unset). Otherwise
    PREFIX runs the EGL probe with ENV: egl on success, osmesa on any failure.
    """
    environ = os.environ if environ is None else environ
    explicit = environ.get("MUJOCO_GL")
    if explicit:
        if announce and environ.get(MUJOCO_GL_ANNOUNCED) != "1":
            print(f"MuJoCo rendering: {explicit} (set by MUJOCO_GL)", file=sys.stderr, flush=True)
        return {"MUJOCO_GL": explicit, "PYOPENGL_PLATFORM": _pyopengl_for(explicit, environ)}
    ok, reason = probe_egl(prefix, env=env, timeout=timeout, cwd=cwd)
    chosen = "egl" if ok else "osmesa"
    if announce:
        note = "GPU OpenGL" if ok else f"EGL unavailable: {reason}"
        print(f"MuJoCo rendering: {chosen} ({note})", file=sys.stderr, flush=True)
    return {"MUJOCO_GL": chosen, "PYOPENGL_PLATFORM": chosen}


def apply_mujoco_gl(resolved: dict, environ=None) -> None:
    """Set RESOLVED in ENVIRON (default os.environ) and mark it announced."""
    environ = os.environ if environ is None else environ
    environ["MUJOCO_GL"] = resolved["MUJOCO_GL"]
    if resolved.get("PYOPENGL_PLATFORM"):
        environ["PYOPENGL_PLATFORM"] = resolved["PYOPENGL_PLATFORM"]
    else:
        environ.pop("PYOPENGL_PLATFORM", None)
    environ[MUJOCO_GL_ANNOUNCED] = "1"


def main(argv=None) -> int:
    """`eval_flags.py mujoco-gl [--dry-run] -- PREFIX...`: print MUJOCO_GL=...
    and PYOPENGL_PLATFORM=... (empty: leave unset), the choice on stderr."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "mujoco-gl":
        print("usage: eval_flags.py mujoco-gl [--dry-run] -- PREFIX...", file=sys.stderr)
        return 2
    argv = argv[1:]
    if argv[:1] == ["--dry-run"]:
        print(f"MUJOCO_GL={mujoco_gl_dry_run()}")
        return 0
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv and not os.environ.get("MUJOCO_GL"):
        print("eval_flags.py mujoco-gl: give the command that runs the simulator's Python",
              file=sys.stderr)
        return 2
    resolved = resolve_mujoco_gl(argv)
    print(f"MUJOCO_GL={resolved['MUJOCO_GL']}")
    print(f"PYOPENGL_PLATFORM={resolved['PYOPENGL_PLATFORM'] or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
