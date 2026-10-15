# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU engine for a compiled robocasa MjModel (MuJoCo 3.3.2 semantics).

Built the way the LIBERO GPU-resident env builds it: inertials and invweight0 from the live MjModel, joint
anchors re-origined (mjcf_fix), MuJoCo's convex-collision data (mjc_Convex port) and the model doubles.
"""
import os
import re

import numpy as np

# RoboCasa scenes queue ~1600 box-mesh / mesh-mesh broadphase pairs per world (objects are 20-30 convex
# pieces among the fixture boxes); the binding's default 256-pair queue overflows into the legacy collider.
CONVEX_QUEUE_CAP = 4096


def build_engine(mj, xml=None, gpu_id=0, num_envs=1, reorigin=True):
    """Returns (model, engine, model-doubles report). mj: compiled MjModel (the live one). xml: its MJCF
    (robosuite's get_xml() is fine; default mj_saveLastXML)."""
    import mujoco
    import instinct_sim
    from instinct_sim_env import hip_env
    from instinct_sim_env.mujoco_convex import mujoco_convex_data, sync_model_doubles
    from .mjcf_fix import reorigin_joint_anchors
    if xml is None:
        import tempfile
        with tempfile.NamedTemporaryFile("r", suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, mj)
            xml = f.read()
    # every body (named or not) gets MuJoCo's compiled inertial; must precede the anchor re-origin,
    # which shifts the inertial pos with the body frame
    from .mjcf_fix import inject_inertials
    xml = inject_inertials(re.sub(r' colorspace="[^"]*"', '', xml), mj)
    # robocasa's objaverse objects / fixtures are textured through their OBJ vt coordinates, as MuJoCo
    # samples them; the GPU simulator replaces OBJ texcoords with texgen unless this extension is set.
    xml = xml.replace("</mujoco>", '<custom><numeric name="instinct_sim_obj_texcoords" data="1"/></custom></mujoco>', 1) \
        if "instinct_sim_obj_texcoords" not in xml else xml
    moved = {}
    if reorigin:
        xml, moved = reorigin_joint_anchors(xml, mj)
    model, eng = instinct_sim.create_engine(
        xml, num_envs=num_envs, gpu_id=gpu_id,
        body_invweight0=mj.body_invweight0.flatten().astype(np.float64),
        dof_invweight0=mj.dof_invweight0.astype(np.float64))
    # The mjc_Convex port needs MuJoCo >= 3.3 polygon data (mesh_polynum, ...).
    if not hasattr(mj, "mesh_polynum"):
        raise RuntimeError(f"MuJoCo {mujoco.__version__} has no mesh polygon data; the GPU simulator's RoboCasa path "
                           "follows MuJoCo 3.3.2 (put a mujoco==3.3.2 install first on PYTHONPATH)")
    if os.environ.get("HP_MUJOCO_CONVEX", "1") == "1":
        cd = mujoco_convex_data(model, mj)
        if moved:
            # geom_mj_pos is MuJoCo's body-relative geom_pos; re-origined bodies moved their frame to
            # the joint anchor p, so their geoms sit at geom_pos - p in the new frame (as in the XML).
            from instinct_sim_env.mujoco_convex import geom_map
            gmap = geom_map(model, mj)
            shift = {mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, b): p for b, p in moved.items()}
            for k, mid in enumerate(gmap):
                p = shift.get(int(mj.geom_bodyid[mid]))
                if p is not None:
                    cd["geom_mj_pos"][k] -= p
        eng.set_mujoco_convex(cd, queue_cap_per_env=CONVEX_QUEUE_CAP)
    rep = sync_model_doubles(model, mj)
    return model, eng, rep
