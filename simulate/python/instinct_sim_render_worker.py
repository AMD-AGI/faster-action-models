#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Subprocess worker for GPU rendering.

Runs in a separate process with system ROCm 7.1 (no PyTorch) to avoid
ROCm version conflicts with torch's bundled ROCm 6.2.4.

Communicates with the main eval process via multiprocessing.Connection pipes.
"""

import os
import sys
import traceback
from pathlib import Path

import numpy as np

# Ensure the GPU simulator and hip_env are importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "build" / "python"))
sys.path.insert(0, str(REPO_ROOT / "python"))

os.environ.setdefault("MUJOCO_GL", "osmesa")

import instinct_sim
import mujoco
from hip_env import inject_mujoco_inertial

LIBERO_ENV_RESOLUTION = 256


def extract_textures(xml_string):
    """Extract texture data from MuJoCo model for the GPU simulator."""
    m = mujoco.MjModel.from_xml_string(xml_string)
    result = {
        "texture_info": {"textures": []},
        "tex_data": None,
        "geom_texids": None,
        "geom_texrepeat": None,
        "tex_iscube": None,
    }
    if m.ntex == 0:
        result["geom_texids"] = np.full(m.ngeom, -1, dtype=np.int32)
        return result

    COLORSPACE_SRGB = 2
    # mjModel.tex_colorspace landed in MuJoCo 3.6; older runtimes (e.g. the
    # FastWAM venv on 3.3.2) do not have it. Those versions store texture data
    # already in the colorspace hipGL expects, so treat "absent" as "no sRGB
    # conversion needed" rather than crashing the whole render path.
    tex_colorspace = getattr(m, "tex_colorspace", None)
    # mjtTexture: 0 = 2D, 1 = CUBE, 2 = SKYBOX. Cube textures are bound as
    # GL_TEXTURE_CUBE_MAP by MuJoCo and sampled with a DIRECTION, which is a
    # different mapping from the 2D path -- treating them as 2D was leaving
    # table_visual (55% of the agentview frame) mis-textured.
    tex_type = getattr(m, "tex_type", None)
    tex_list, all_data, tex_iscube = [], [], []
    for i in range(m.ntex):
        w, h = int(m.tex_width[i]), int(m.tex_height[i])
        adr = m.tex_adr[i]
        ttype = int(tex_type[i]) if tex_type is not None else 0
        is_cube = 1 if ttype == 1 else 0
        if is_cube:
            # Normalise both MuJoCo cube layouts to 6 stacked square faces so
            # the sampler arithmetic is uniform (render_context.c:1428-1447):
            #   square (w == h): the SAME image goes on all six faces
            #   h == 6*w:        six faces already stacked, in GL order
            if h == w:
                one = m.tex_data[adr:adr + w * w * 3].copy()
                data = np.tile(one, 6)
            elif h == 6 * w:
                data = m.tex_data[adr:adr + w * h * 3].copy()
            else:
                # Not a layout MuJoCo produces; fall back to 2D rather than
                # silently sampling garbage.
                is_cube = 0
                data = m.tex_data[adr:adr + w * h * 3].copy()
            if is_cube:
                h = 6 * w          # stored height after normalisation
        else:
            data = m.tex_data[adr:adr + w * h * 3].copy()
        tex_iscube.append(is_cube)
        if tex_colorspace is not None and tex_colorspace[i] == COLORSPACE_SRGB:
            srgb = data.astype(np.float32) / 255.0
            linear = np.where(srgb <= 0.04045, srgb / 12.92,
                              ((srgb + 0.055) / 1.055) ** 2.4)
            data = np.clip(linear * 255.0, 0, 255).astype(np.uint8)
        all_data.append(data)
        tex_list.append({"width": int(w), "height": int(h)})

    result["texture_info"]["textures"] = tex_list
    result["tex_iscube"] = tex_iscube
    result["tex_data"] = np.concatenate(all_data).astype(np.uint8)

    geom_texids = np.full(m.ngeom, -1, dtype=np.int32)
    geom_texrepeat = np.ones((m.ngeom, 2), dtype=np.float32)
    for gi in range(m.ngeom):
        matid = m.geom_matid[gi]
        if matid >= 0:
            texid = int(m.mat_texid[matid, 1])
            if texid >= 0:
                geom_texids[gi] = texid
                rep_s = float(m.mat_texrepeat[matid, 0])
                rep_t = float(m.mat_texrepeat[matid, 1])
                if m.mat_texuniform[matid]:
                    gtype = m.geom_type[gi]
                    if gtype in (0, 6):
                        gsx = float(m.geom_size[gi, 0])
                        gsy = float(m.geom_size[gi, 1])
                        if gsx > 0: rep_s *= gsx
                        if gsy > 0: rep_t *= gsy
                geom_texrepeat[gi, 0] = rep_s
                geom_texrepeat[gi, 1] = rep_t

    result["geom_texids"] = geom_texids
    result["geom_texrepeat"] = geom_texrepeat
    return result


_WRIST_DARK_BAND_ALPHA = None


def _build_wrist_dark_band_alpha(h, w, dark_end=39, transition_end=45):
    alpha = np.ones((h, w, 1), dtype=np.float32)
    for r in range(min(transition_end, h)):
        if r < dark_end:
            alpha[r, :, :] = 0.0
        else:
            alpha[r, :, :] = (r - dark_end) / (transition_end - dark_end)
    return alpha


def apply_wrist_dark_band(img, dark_end=39, transition_end=45,
                          dark_color=(52, 52, 52)):
    global _WRIST_DARK_BAND_ALPHA
    h, w = img.shape[:2]
    if (_WRIST_DARK_BAND_ALPHA is None
            or _WRIST_DARK_BAND_ALPHA.shape[0] != h):
        _WRIST_DARK_BAND_ALPHA = _build_wrist_dark_band_alpha(
            h, w, dark_end, transition_end)
    alpha = _WRIST_DARK_BAND_ALPHA
    dc = np.array(dark_color, dtype=np.float32)
    blended = alpha * img.astype(np.float32) + (1.0 - alpha) * dc
    return blended.clip(0, 255).astype(np.uint8)


class _RenderState:
    """Holds GPU engine/renderer state for a single scene."""
    def __init__(self):
        self.hip_model = None
        self.hip_engine = None
        self.hip_renderer = None
        self.cam_ids = {}


def _do_setup(state, xml_string, gpu_id, skip_group0):
    """Create GPU engine + renderer for a scene."""
    mj_model = mujoco.MjModel.from_xml_string(xml_string)
    biw = mj_model.body_invweight0.copy()
    diw = mj_model.dof_invweight0.copy()
    tex = extract_textures(xml_string)

    xml_string = inject_mujoco_inertial(xml_string, mj_model)

    # render_only=True: this worker only ever calls engine.forward() (FK + geom
    # and site transforms) from MuJoCo-supplied qpos/qvel — it never integrates,
    # so the dense solver's nv <= KERNEL_MAXNV limit does not apply. Without the
    # flag, create_engine REFUSES every nv>48 scene (all 10 libero_object tasks
    # at nv=51, 2 libero_10 tasks at nv=57); the caller would then skip those
    # tasks and report an SR averaged over only the tasks that happened to load.
    hip_model, hip_engine = instinct_sim.create_engine(
        xml_string, num_envs=1, gpu_id=gpu_id,
        body_invweight0=biw, dof_invweight0=diw,
        render_only=True,
    )

    if tex["tex_data"] is not None and tex["geom_texids"] is not None:
        tex_widths = [t["width"] for t in tex["texture_info"].get("textures", [])]
        tex_heights = [t["height"] for t in tex["texture_info"].get("textures", [])]
        _kw = {}
        if tex.get("tex_iscube") is not None:
            _kw["tex_iscube"] = [int(v) for v in tex["tex_iscube"]]
        hip_model.set_textures(tex["tex_data"], tex_widths, tex_heights,
                               tex["geom_texids"].tolist(), tex["geom_texrepeat"], **_kw)

    hip_renderer = instinct_sim.CameraRenderer()
    _ssaa = int(os.environ.get('HP_SR_SSAA', '1'))
    print(f'[instinct_sim] renderer ssaa={_ssaa}', flush=True)
    hip_renderer.init(hip_model, LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, _ssaa, 1)
    if skip_group0:
        hip_renderer.set_geomgroup(0, False)

    cam_ids = {}
    for name in ["agentview", "robot0_eye_in_hand"]:
        cid = hip_model.camera_name2id(name)
        if cid >= 0:
            cam_ids[name] = cid

    state.hip_model = hip_model
    state.hip_engine = hip_engine
    state.hip_renderer = hip_renderer
    state.cam_ids = cam_ids
    return cam_ids


def _do_render(state, qpos, qvel):
    """Sync state and render all cameras. Returns {cam_name: np.ndarray}."""
    engine = state.hip_engine
    renderer = state.hip_renderer

    engine.data.write_qpos_d_env(qpos.astype(np.float32), 0)
    engine.data.write_qvel_d_env(qvel.astype(np.float32), 0)
    engine.forward()
    engine.sync()

    images = {}
    for cam_name, cam_id in state.cam_ids.items():
        img = np.array(renderer.render(engine, camera_idx=cam_id, env_idx=0))
        img = img[::-1].copy()
        if cam_name == "robot0_eye_in_hand":
            img = apply_wrist_dark_band(img)
        images[cam_name] = img
    return images


def _do_teardown(state):
    """Clean up GPU resources."""
    del state.hip_renderer
    del state.hip_engine
    del state.hip_model
    state.hip_renderer = None
    state.hip_engine = None
    state.hip_model = None
    state.cam_ids = {}


def worker_main(conn, gpu_id):
    """Main loop for the GPU render subprocess.

    Protocol:
        conn.recv() → (command, *args)
        conn.send() → result or ("error", traceback_string)

    Commands:
        ("setup", xml_string, skip_group0) → cam_ids dict
        ("render", qpos, qvel) → {cam_name: np.ndarray}
        ("teardown",) → "ok"
        ("quit",) → exits process
    """
    state = _RenderState()

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break

        cmd = msg[0]
        try:
            if cmd == "setup":
                xml_string, skip_group0 = msg[1], msg[2]
                cam_ids = _do_setup(state, xml_string, gpu_id, skip_group0)
                conn.send(cam_ids)
            elif cmd == "render":
                qpos, qvel = msg[1], msg[2]
                images = _do_render(state, qpos, qvel)
                conn.send(images)
            elif cmd == "teardown":
                _do_teardown(state)
                conn.send("ok")
            elif cmd == "quit":
                _do_teardown(state)
                conn.send("ok")
                break
            else:
                conn.send(("error", f"Unknown command: {cmd}"))
        except Exception:
            conn.send(("error", traceback.format_exc()))

    conn.close()
