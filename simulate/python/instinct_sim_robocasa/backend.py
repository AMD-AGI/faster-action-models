# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a robosuite / robocasa env's physics on the GPU simulator.

attach(env) swaps the physics substep of a robosuite env (MjSim.step, or MjSim.step2 under robosuite's
lite_physics) for a GPU step from the same MuJoCo state and ctrl; the new qpos / qvel / time are
written back into MuJoCo's mjData. Everything else stays on the MuJoCo side exactly as in the CPU
reference: robocasa's scene generation and resets, the GR1 composite controller (it computes ctrl from
mjData every substep), observations, rendering and robocasa's own success checks. Like mj_step2, the
substep leaves mjData's position-dependent quantities (xpos, contacts, ...) at their pre-step values;
robosuite's next sim.forward() / step1() recomputes them from the GPU state.

A new GPU engine is built whenever robosuite builds a new MjSim (robocasa hard-resets a fresh scene
every episode).

HP_RENDERER=rt: camera images (MjSim.render) come from the GPU ray tracer instead of MuJoCo's GL
renderer, configured to match robosuite's OSMesa images (msaa4 = the 4x MSAA of MuJoCo's offscreen buffer,
llvmpipe sampler, robosuite's geom groups; MAE <= 0.03 / 255). HP_RT_AA overrides the antialiasing. The ray tracer renders the configuration MuJoCo's own
renderer would show: the qpos of mjData's last kinematics pass (the pre-step state after a substep, as
mj_step leaves xpos; or whatever MjSim.forward last ran on, e.g. after a reset).
"""
import os
import time

import numpy as np


class HipBackend:
    def __init__(self, gpu_id=0, verbose=True):
        self.gpu_id = gpu_id
        self.verbose = verbose
        self.sim = None
        self.eng = None
        self.model = None
        self.build_s = []          # engine build seconds per scene
        self.nstep = 0
        self.step_s = 0.0
        self.dump_dir = os.environ.get("HP_DUMP_DIR")
        self.dump_qvel = float(os.environ.get("HP_DUMP_QVEL", "20"))
        self._hist = []
        self._dumped = False
        self.rt = os.environ.get("HP_RENDERER", "").strip().lower() == "rt"
        self._renderers = {}       # (width, height) -> RtCameraRenderer for the current engine
        self._kin_qpos = None      # qpos of mjData's last kinematics pass (what the renderer shows)
        self.render_s = 0.0
        self.nrender = 0

    # -- engine lifecycle ---------------------------------------------------------------------------
    def bind(self, sim):
        """Build an engine for sim's model and route sim's physics substep through it."""
        from .engine import build_engine
        t0 = time.time()
        mj = sim.model._model
        self.model, self.eng, _ = build_engine(mj, sim.model.get_xml(), gpu_id=self.gpu_id)
        self.build_s.append(time.time() - t0)
        if self.verbose:
            print(f"[hip-backend] engine for nq {mj.nq} nv {mj.nv} nu {mj.nu} ngeom {mj.ngeom} built in "
                  f"{self.build_s[-1]:.1f}s", flush=True)
        self.sim = sim
        self._hist = []
        self._dumped = False
        self._nq, self._nv, self._nu = mj.nq, mj.nv, mj.nu
        self._dt = float(mj.opt.timestep)
        # HP_GPU_PHYSICS=0 (with HP_RENDERER=rt): physics stays on CPU MuJoCo, the GPU simulator only renders (the
        # kinematics qpos is recorded at MuJoCo's own step1 / step / forward)
        self.physics = os.environ.get("HP_GPU_PHYSICS", "1") != "0"
        if self.physics:
            sim.step = self._step
            sim.step2 = self._step2
        else:
            o_step1, o_step = sim.step1, sim.step
            def step1(*a, **k):
                self._kin_qpos = sim.data._data.qpos.copy(); return o_step1(*a, **k)
            def step(*a, **k):
                self._kin_qpos = sim.data._data.qpos.copy(); return o_step(*a, **k)
            sim.step1, sim.step = step1, step
        self._renderers = {}
        self._kin_qpos = None
        if self.rt:
            self._xml = sim.model.get_xml()
            self._orig_render = sim.render
            self._orig_forward = sim.forward
            sim.render = self._render
            sim.forward = self._forward

    def _push(self, d):
        e = self.eng.data
        e.write_qpos_d_env(np.ascontiguousarray(d.qpos, dtype=np.float64).reshape(1, self._nq), 0)
        e.write_qvel_d_env(np.ascontiguousarray(d.qvel, dtype=np.float64).reshape(1, self._nv), 0)
        if self._nu:
            e.write_ctrl(np.ascontiguousarray(d.ctrl, dtype=np.float32).reshape(1, self._nu))

    def _advance(self):
        import mujoco  # noqa: F401
        d = self.sim.data._data
        t0 = time.perf_counter()
        if self.dump_dir:
            self._hist.append((d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), float(d.time)))
            del self._hist[:-50]
        self._push(d)
        self._kin_qpos = d.qpos.copy()   # mj_step leaves xpos at the pre-step state
        self.eng.forward()
        self.eng.step()
        self.eng.sync()
        d.qpos[:] = np.asarray(self.eng.data.read_qpos_d()).reshape(-1)[:self._nq]
        d.qvel[:] = np.asarray(self.eng.data.read_qvel_d()).reshape(-1)[:self._nv]
        d.time += self._dt
        self.step_s += time.perf_counter() - t0
        self.nstep += 1
        if self.dump_dir and not self._dumped and np.abs(d.qvel).max() > self.dump_qvel:
            self._dump()

    # -- fast path (fast_step.py): many substeps per GPU call ------------------------------------------
    def advance_n(self, n):
        """n substeps on the GPU from mjData's state, PD actuators driven by the GPU joint PD; mjData gets the
        final qpos / qvel / time (its kinematics are left as they are)."""
        if n <= 0:
            return
        import mujoco
        d = self.sim.data._data
        t0 = time.perf_counter()
        self._push(d)
        self.eng.forward()
        self.eng.step_n(n)
        self.eng.sync()
        d.qpos[:] = np.asarray(self.eng.data.read_qpos_d()).reshape(-1)[:self._nq]
        d.qvel[:] = np.asarray(self.eng.data.read_qvel_d()).reshape(-1)[:self._nv]
        d.time += n * self._dt
        self.step_s += time.perf_counter() - t0
        self.nstep += n

    def advance_last_kinematic(self):
        """The control step's last substep, leaving mjData as mj_step would: kinematics (xpos, contacts, ...)
        computed at the pre-step state, qpos / qvel / time after the step."""
        import mujoco
        d = self.sim.data._data
        mujoco.mj_forward(self.sim.model._model, d)   # kinematics + ctrl-independent quantities at pre-step
        self._kin_qpos = d.qpos.copy()
        t0 = time.perf_counter()
        self._push(d)
        self.eng.forward()
        self.eng.step()
        self.eng.sync()
        d.qpos[:] = np.asarray(self.eng.data.read_qpos_d()).reshape(-1)[:self._nq]
        d.qvel[:] = np.asarray(self.eng.data.read_qvel_d()).reshape(-1)[:self._nv]
        d.time += self._dt
        self.step_s += time.perf_counter() - t0
        self.nstep += 1

    def _dump(self):
        import mujoco
        self._dumped = True
        tag = f"{os.getpid()}_{self.nstep}"
        os.makedirs(self.dump_dir, exist_ok=True)
        base = os.path.join(self.dump_dir, tag)
        mujoco.mj_saveModel(self.sim.model._model, base + ".mjb", None)
        open(base + ".xml", "w").write(self.sim.model.get_xml())
        np.savez(base + ".npz", qpos=np.array([h[0] for h in self._hist]), qvel=np.array([h[1] for h in self._hist]),
                 ctrl=np.array([h[2] for h in self._hist]), time=np.array([h[3] for h in self._hist]),
                 post_qpos=self.sim.data._data.qpos.copy(), post_qvel=self.sim.data._data.qvel.copy())
        print(f"[hip-backend] |qvel| {np.abs(self.sim.data._data.qvel).max():.1f} > {self.dump_qvel}: dumped {base}.*",
              flush=True)

    # -- rendering (HP_RENDERER=rt) -------------------------------------------------------------------
    def _forward(self, *a, **k):
        r = self._orig_forward(*a, **k)
        self._kin_qpos = self.sim.data._data.qpos.copy()
        return r

    def _renderer(self, width, height):
        key = (int(width), int(height))
        if key not in self._renderers:
            import re
            import sys
            from instinct_sim_env.rt_renderer import RtCameraRenderer
            repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            sys.path.insert(0, os.path.join(repo, "tests"))
            from instinct_sim_render_worker import extract_textures
            if not self._renderers:   # textures once per engine, before the first RayTracer init
                # the XML as of the first render (after robosuite's reset), not of engine bind
                tex = extract_textures(re.sub(r' colorspace="[^"]*"', '', self.sim.model.get_xml()))
                tw = [t["width"] for t in tex["texture_info"]["textures"]]
                th = [t["height"] for t in tex["texture_info"]["textures"]]
                if tw:
                    self.model.set_textures(tex["tex_data"], tw, th, tex["geom_texids"].tolist(),
                                            tex["geom_texrepeat"], tex_iscube=tex["tex_iscube"])
            rr = RtCameraRenderer(self.sim.model._model, aa=os.environ.get("HP_RT_AA", "") or "msaa4",
                                 sampler="llvmpipe", verbose=self.verbose)
            self._renderers[key] = rr.init(self.model, *key)
            self._groups = None
        rr = self._renderers[key]
        # robosuite sets its context's vopt.geomgroup (collision meshes hidden) during reset, after the
        # first render may already have happened: mirror it on every render
        ctx = getattr(self.sim, "_render_context_offscreen", None)
        groups = tuple(int(v) for v in ctx.vopt.geomgroup) if ctx is not None else (0, 1, 1, 0, 0, 0)
        if getattr(rr, "_hp_groups", None) != groups:
            for g, v in enumerate(groups):
                rr.raytracer.set_geomgroup(g, bool(v))
            rr._hp_groups = groups
        # robosuite edits the live model's site_rgba during reset (hides its visualization sites); the
        # engine was built from the XML before that, so mirror the live colours (hip sites follow MuJoCo order)
        srgba = np.ascontiguousarray(self.sim.model._model.site_rgba, dtype=np.float32)
        if srgba.shape[0] == self.model.nsite and getattr(rr, "_hp_site_rgba", None) != srgba.tobytes():
            rr.set_site_rgba(self.eng, srgba)
            rr._hp_site_rgba = srgba.tobytes()
        return rr

    def _render(self, width=None, height=None, *, camera_name=None, depth=False, mode="offscreen",
                device_id=-1, segmentation=False):
        if depth or segmentation or camera_name is None or mode != "offscreen":
            return self._orig_render(width, height, camera_name=camera_name, depth=depth, mode=mode,
                                     device_id=device_id, segmentation=segmentation)
        t0 = time.perf_counter()
        d = self.sim.data._data
        q = self._kin_qpos if self._kin_qpos is not None else d.qpos
        e = self.eng.data
        e.write_qpos_d_env(np.ascontiguousarray(q, dtype=np.float64).reshape(1, self._nq), 0)
        self.eng.forward()
        img = np.asarray(self._renderer(width, height).render(self.eng, self.model.camera_name2id(camera_name)))
        self.render_s += time.perf_counter() - t0
        self.nrender += 1
        return np.ascontiguousarray(img[::-1, :, :3])   # MjSim.render returns GL row order (bottom first)

    def _step(self, with_udd=True):
        # mj_step = mj_step1 (position / velocity stage) + mj_step2 (acceleration + integration)
        import mujoco
        mujoco.mj_step1(self.sim.model._model, self.sim.data._data)
        self._advance()

    def _step2(self):
        self._advance()


def attach(env, gpu_id=0, verbose=True):
    """Route env's physics through the GPU simulator. env: a gymnasium robocasa env or the robosuite env.
    Returns the HipBackend."""
    rs = env
    while not hasattr(rs, "_initialize_sim") and hasattr(rs, "env"):
        rs = rs.env
    if hasattr(rs, "unwrapped") and not hasattr(rs, "_initialize_sim"):
        rs = rs.unwrapped
    be = HipBackend(gpu_id=gpu_id, verbose=verbose)
    orig_init = rs._initialize_sim

    def init_sim(*a, **k):
        r = orig_init(*a, **k)
        be.bind(rs.sim)
        return r

    rs._initialize_sim = init_sim
    if getattr(rs, "sim", None) is not None:
        be.bind(rs.sim)
    if os.environ.get("HP_FAST_STEP") == "1":
        from .fast_step import install
        install(rs, be)
    return be
