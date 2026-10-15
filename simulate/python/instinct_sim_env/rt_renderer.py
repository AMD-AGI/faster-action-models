# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renderer switch for the GPU envs: the HIP ray tracer (default) or the GPU rasterizer.

The switch is read ONCE, at env construction:

    HP_RENDERER   = rt (default) | raster (fallback: the CameraRenderer setup); unset or empty means rt.
    HP_RT_AA      = ssaa1 (default, no antialiasing) | msaa4 (MuJoCo's offscreen buffer is 4x MSAA,
                    closest to MuJoCo's images) | ssaa4
    HP_RT_SAMPLER = unset (the 'mujoco' preset's sampler, hw) | hw | hw_mips | llvmpipe

Envs that already take renderer options also take them as constructor kwargs (``renderer=``,
``rt_aa=``, ``rt_sampler=``); a kwarg wins over the environment variable.

With ``raster`` nothing changes: the envs build exactly the CameraRenderer(s) they always did.

With ``rt`` the env needs its MuJoCo model (``mj_model=``; the GPU-resident LIBERO env passes
robosuite's LIVE model). When the RT was chosen by default (HP_RENDERER unset) and no mj_model was
passed, the env raises instead of rendering something else: pass mj_model, or set
HP_RENDERER=raster. An EXPLICIT ``rt`` (HP_RENDERER=rt / renderer='rt') without mj_model compiles
the env's xml_string, as before (mujoco_model_for).

With ``rt`` an env owns ONE ``RtCameraRenderer``: a CameraRenderer-shaped facade over one
``instinct_sim.RayTracer``, configured by ``instinct_sim_rt_mujoco.configure`` from the MuJoCo model the
env already holds (mesh normals, texgen planes, sRGB texels, lights, geom colours, ellipsoids). It
exposes the subset of the CameraRenderer API the env code and the LIBERO evals call, so callers
that reach into ``env._renderer`` (GPU-resident env: ssaa re-init, geom groups, sort metadata;
SiteVisualCoordinator: site rgba) keep working unchanged:

  * ``init(model, w, h, ssaa, msaa)`` (re)builds the RayTracer. The RT samples the model's textures
    at init, so a caller that re-uploads textures (e.g. with tex_iscube) must re-init, exactly as
    the GPU-resident env already does for its ssaa re-init. The raster ``ssaa``/``msaa`` arguments
    are recorded and ignored: the RT's antialiasing is ``aa`` (HP_RT_AA). Geom/site group choices
    survive a re-init; site rgba state does not (set it after init, as the env code does).
  * geom groups, site groups, site rgba (per world / batch, get/clear): forwarded. Site visibility
    is therefore driven by the env's own site-rgba state (LIBERO: SiteVisualCoordinator /
    HipVisualRenderOnlyEnv push the live model's site_rgba every step) -- the same mechanism the
    raster uses; no name rule is involved (instinct_sim_rt_mujoco.libero_site_rgba is only a test
    helper for references recorded without site rgba).
  * ``render`` / ``render_gpu_batch``: forwarded (same row order and flips as the raster).
    ``render_all_gpu_batch(engine, camera_ids, num_envs, active)`` renders every camera in ONE
    launch -> device (C, N, H, W, 3) u8 (``__cuda_array_interface__``; ``.numpy()`` is one D2H).
  * fog, gradient background, clip planes, bg colour: forwarded to the RT's own equivalents.
  * raster-only appearance knobs the MuJoCo preset already decides are no-ops:
    ``set_gouraud_shading`` (preset shades Gouraud like MuJoCo GL), ``set_shadow(False)``,
    ``set_num_streams``, ``set_transparency_sort_metadata`` (the RT resolves transparency per ray).
    ``set_shadow(True)`` warns (the RT has no shadows; LIBERO lights are castshadow=false).
"""
from __future__ import annotations

import os
import sys
import warnings

RENDERERS = ("raster", "rt")
RT_AAS = ("msaa4", "ssaa1", "ssaa4")
RT_SAMPLERS = ("hw", "hw_mips", "llvmpipe")


DEFAULT_RENDERER = "rt"


def renderer_settings(renderer=None, rt_aa=None, rt_sampler=None) -> dict:
    """Resolve the renderer switch: kwargs win, then HP_RENDERER / HP_RT_AA / HP_RT_SAMPLER, then the
    default (rt). Returns {'renderer', 'rt_aa', 'rt_sampler', 'renderer_source'} with
    renderer_source = 'kwarg' | 'env' | 'default'; raises ValueError on an unknown value."""
    if renderer is not None:
        r, src = renderer, "kwarg"
    elif os.environ.get("HP_RENDERER", "").strip():
        r, src = os.environ["HP_RENDERER"], "env"
    else:
        r, src = DEFAULT_RENDERER, "default"
    r = str(r).strip().lower()
    if r not in RENDERERS:
        raise ValueError(f"HP_RENDERER / renderer= must be one of {RENDERERS}, got {r!r}")
    aa = str(rt_aa if rt_aa is not None else os.environ.get("HP_RT_AA", "") or "ssaa1").strip().lower()
    if aa not in RT_AAS:
        raise ValueError(f"HP_RT_AA / rt_aa= must be one of {RT_AAS}, got {aa!r}")
    sm = rt_sampler if rt_sampler is not None else (os.environ.get("HP_RT_SAMPLER", "") or None)
    if sm is not None:
        sm = str(sm).strip().lower()
        if sm not in RT_SAMPLERS:
            raise ValueError(f"HP_RT_SAMPLER / rt_sampler= must be one of {RT_SAMPLERS}, got {sm!r}")
    return {"renderer": r, "rt_aa": aa, "rt_sampler": sm, "renderer_source": src}


def _rt_mujoco():
    try:
        import instinct_sim_rt_mujoco as hm
    except ImportError:   # package imported without <repo>/python on sys.path
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import instinct_sim_rt_mujoco as hm
    return hm


class RtCameraRenderer:
    """CameraRenderer-shaped facade over one instinct_sim.RayTracer (see module docstring)."""

    is_rt = True

    def __init__(self, mj_model, aa: str = "msaa4", sampler: str | None = None, instinct_sim_module=None,
                 verbose: bool = True):
        if mj_model is None:
            raise ValueError("RtCameraRenderer needs the env's MuJoCo model (mesh normals, texgen, lights, colours)")
        if aa not in RT_AAS:
            raise ValueError(f"aa must be one of {RT_AAS}")
        if instinct_sim_module is None:
            import instinct_sim as instinct_sim_module
        if not hasattr(instinct_sim_module, "RayTracer"):
            raise RuntimeError(f"HP_RENDERER=rt but this build has no RayTracer "
                               f"({getattr(instinct_sim_module, '__file__', '?')})")
        self._hp = instinct_sim_module
        self.mj_model = mj_model
        self.aa = aa
        self.sampler = sampler
        self.verbose = verbose
        self._rt = None
        self._model = None
        self._groups = {}       # group -> visible, replayed on re-init
        self._sitegroups = {}
        self._view = {}         # fog / gradient / clip / bg, replayed on re-init
        self.raster_request = None
        self.width = self.height = 0

    # ---------------------------------------------------------------- setup
    def init(self, model, width: int = 256, height: int = 256, ssaa: int = 1, msaa: int = 1):
        """(Re)build the RayTracer for `model` at width x height (raster ssaa/msaa ignored)."""
        hm = _rt_mujoco()
        rt = self._hp.RayTracer()
        rt.init(model, int(width), int(height), ssaa=4 if self.aa == "ssaa4" else 1)
        for g, v in sorted(self._groups.items()):
            rt.set_geomgroup(g, v)
        for g, v in sorted(self._sitegroups.items()):
            rt.set_sitegroup(g, v)
        rt.set_mode("hipgl")
        hm.configure(rt, self.mj_model, model, aa="msaa4" if self.aa == "msaa4" else "ssaa",
                     sampler=self.sampler)
        for name, args in self._view.items():
            getattr(rt, name)(*args)
        self._rt, self._model = rt, model
        self.width, self.height = int(width), int(height)
        self.raster_request = (int(ssaa), int(msaa))
        if self.verbose:
            print(f"[instinct_sim] renderer=rt aa={self.aa} sampler={self.sampler or 'preset(hw)'} "
                  f"{self.width}x{self.height} (raster ssaa={int(ssaa)} msaa={int(msaa)} request not used)",
                  flush=True)
        return self

    @property
    def raytracer(self):
        """The underlying instinct_sim.RayTracer (None before init)."""
        return self._rt

    def _need(self):
        if self._rt is None:
            raise RuntimeError("RtCameraRenderer not initialized")
        return self._rt

    # ---------------------------------------------------------------- visibility
    def set_geomgroup(self, group: int, visible: bool):
        self._groups[int(group)] = bool(visible)
        if self._rt is not None:
            self._rt.set_geomgroup(int(group), bool(visible))

    def set_sitegroup(self, group: int, visible: bool):
        self._sitegroups[int(group)] = bool(visible)
        if self._rt is not None:
            self._rt.set_sitegroup(int(group), bool(visible))

    def set_site_rgba(self, engine, rgba, env_idx: int = 0):
        return self._need().set_site_rgba(engine, rgba, env_idx=env_idx)

    def set_site_rgba_batch(self, engine, rgba):
        return self._need().set_site_rgba_batch(engine, rgba)

    def get_site_rgba(self, engine, env_idx: int = 0):
        return self._need().get_site_rgba(engine, env_idx)

    def get_site_rgba_batch(self, engine):
        return self._need().get_site_rgba_batch(engine)

    def clear_site_rgba(self, engine, env_idx=None):
        return self._need().clear_site_rgba(engine, env_idx)

    # ---------------------------------------------------------------- view state (RT equivalents)
    def _view_call(self, name, *args):
        self._view[name] = args
        if self._rt is not None:
            getattr(self._rt, name)(*args)

    def set_fog(self, start, end, r, g, b):
        self._view_call("set_fog", float(start), float(end), float(r), float(g), float(b))

    def set_gradient_bg(self, bot_r, bot_g, bot_b, top_r, top_g, top_b):
        self._view_call("set_gradient_bg", *(float(v) for v in (bot_r, bot_g, bot_b, top_r, top_g, top_b)))

    def set_clip_planes(self, znear, zfar):
        self._view_call("set_clip_planes", float(znear), float(zfar))

    def set_bg_color(self, r, g, b, a=1.0):
        self._view_call("set_bg_color", float(r), float(g), float(b), float(a))

    # ---------------------------------------------------------------- raster-only knobs
    def set_gouraud_shading(self, enabled: bool):
        """No-op: the RT's MuJoCo preset already shades like MuJoCo GL (Gouraud)."""

    def set_shadow(self, enabled: bool):
        if enabled:
            warnings.warn("RtCameraRenderer: the RT draws no shadows; set_shadow(True) ignored", stacklevel=2)

    def set_num_streams(self, n: int = 4):
        """No-op: the RT renders every camera and world in one launch."""

    def set_transparency_sort_metadata(self, positions, radii):
        """No-op: the RT resolves transparency per ray (no draw-order sort)."""

    # ---------------------------------------------------------------- rendering
    def render(self, engine, camera_idx: int, env_idx: int = 0):
        """(H, W, 3) u8 host image, row 0 = top (CameraRenderer.render's order)."""
        return self._need().render(engine, camera_idx=camera_idx, env_idx=env_idx)

    def render_gpu_batch(self, engine, camera_idx: int, num_envs: int, flip_v: bool = False,
                         flip_h: bool = False, active=None):
        """(N, H, W, 3) u8 device batch for one camera (CameraRenderer.render_gpu_batch's contract)."""
        return self._need().render_gpu_batch(engine, camera_idx, num_envs, flip_v, flip_h,
                                             None if active is None else list(active))

    def render_all_gpu_batch(self, engine, camera_ids, num_envs: int, active=None):
        """All cameras in ONE launch -> device GpuArray (C, N, H, W, 3) u8, row 0 = top."""
        rgb, _, _ = self._need().render_all_gpu_batch(
            engine, [int(c) for c in camera_ids], int(num_envs),
            None if active is None else list(active), outputs="rgb")
        return rgb

    def render_gpu(self, *a, **k):
        raise NotImplementedError("RtCameraRenderer: use render_gpu_batch / render_all_gpu_batch")

    def stats(self):
        return self._need().stats()

    def settings(self):
        return self._need().settings()


def make_rt_renderer(model, width, height, mj_model, settings: dict, instinct_sim_module=None,
                     verbose: bool = True) -> RtCameraRenderer:
    """An initialised RtCameraRenderer for `settings` (renderer_settings())."""
    r = RtCameraRenderer(mj_model, aa=settings["rt_aa"], sampler=settings["rt_sampler"],
                         instinct_sim_module=instinct_sim_module, verbose=verbose)
    return r.init(model, width, height)


def _no_model_message(why: str) -> str:
    return (f"instinct_sim: the ray tracer is configured from the env's MuJoCo model, and {why}. "
            "Pass mj_model=<the env's live mujoco.MjModel, e.g. robosuite env.sim.model._model>, or set "
            "HP_RENDERER=raster (renderer='raster') to use the rasterizer. The ray tracer is the default "
            "renderer; HP_RENDERER=rt / renderer='rt' set "
            "explicitly configures it from a recompile of xml_string instead.")


def mujoco_model_for(xml_string: str, mj_model=None, settings: dict | None = None):
    """The MuJoCo model the RT is configured from: the caller's (live) model if given. Live is
    preferred: a get_xml() recompile is not guaranteed identical.

    Without mj_model:
      * RT chosen by DEFAULT (settings['renderer_source'] == 'default'): RuntimeError. A caller that
        never asked for the RT and has no model gets a clear error, not a silently different image
        path; it passes mj_model or sets HP_RENDERER=raster.
      * RT requested explicitly (HP_RENDERER=rt / renderer='rt'): xml_string compiled, as before;
        a compile failure (no mujoco, unresolvable assets) raises the same RuntimeError."""
    if mj_model is not None:
        return mj_model
    if settings is not None and settings.get("renderer_source") == "default":
        raise RuntimeError(_no_model_message("no mj_model was passed (HP_RENDERER is unset, so the "
                                             "default renderer, rt, was selected)"))
    if not xml_string:
        raise RuntimeError(_no_model_message("neither mj_model nor an xml_string to compile was given"))
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_string(xml_string)
    except Exception as e:  # noqa: BLE001 -- re-raised with the fix in the message
        raise RuntimeError(_no_model_message(f"no mj_model was passed and compiling xml_string failed "
                                             f"({type(e).__name__}: {e})")) from e
    print("[instinct_sim] renderer=rt: no mj_model passed; configuring the RT from a recompile of "
          "xml_string (pass the live model to avoid recompile differences)", flush=True)
    return m
