# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MJCF rewrites that make a robocasa scene exact under the GPU simulator's conventions.

reorigin_joint_anchors: the GPU simulator assumes every joint sits at its body's origin. Forward
kinematics only subtracts R_world * jnt_pos (it never adds MuJoCo's R_pre * jnt_pos, and it
also subtracts for slide joints, whose anchor MuJoCo ignores), and the constraint Jacobian takes
the hinge lever arm from the body origin. A joint with pos != 0 therefore puts the body (and
everything on it) R_pre * jnt_pos away from MuJoCo: robocasa's cabinet / drawer slide joints
(pos 0 -0.375 0) and door hinges sat 0.15-0.375 m off, colliding with their own cabinets.

Moving the body frame onto the anchor removes the anchor from the model without changing its
physics: body pos += R_body * p, and every child expressed in the body frame (geoms, sites,
cameras, lights, child bodies, inertial, joints) gets -p. Geom / site / camera / child world
poses then match MuJoCo exactly; only the origin of a re-origined body moves (by R * p), so
read world poses through sites or MuJoCo, not through those bodies' xpos.
"""
import xml.etree.ElementTree as ET

import numpy as np

_SHIFT_TAGS = ("geom", "site", "camera", "light", "body", "inertial", "joint", "frame")


def _vec(s, n=3):
    return np.array([float(x) for x in s.split()][:n]) if s else np.zeros(n)


def _fmt(v):
    return " ".join(repr(float(x)) for x in v)


def _rotate(q, v):
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R @ v


def reorigin_joint_anchors(xml_string, mj_model, verbose=False):
    """Returns (xml, moved): moved maps each re-origined body name to its anchor p (body frame). Anything
    else built from mj_model's body-relative poses (e.g. mujoco_convex_data's geom_mj_pos) must apply
    the same -p to that body's children: see engine.build_engine. mj_model: the compiled MuJoCo model of xml_string; supplies
    each body's local orientation (body_quat, whatever MJCF form it was written in) and each
    joint's anchor (jnt_pos, including default-class values)."""
    import mujoco
    root = ET.fromstring(xml_string)
    moved = {}
    for body in root.iter("body"):
        joints = [c for c in body if c.tag in ("joint", "freejoint")]
        if not joints or joints[0].tag == "freejoint" or joints[0].get("type") == "free":
            continue
        name = body.get("name")
        jname = joints[0].get("name")
        if not name or not jname:
            continue
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if bid < 0 or jid < 0:
            continue
        p = np.array(mj_model.jnt_pos[jid], dtype=np.float64)
        if not np.any(p):
            continue
        anchors = {tuple(np.round(mj_model.jnt_pos[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j.get("name"))], 12))
                   for j in joints if j.get("name")}
        if len(anchors) > 1:
            raise ValueError(f"body {name!r}: joints with different anchors {sorted(anchors)}; "
                             "the GPU simulator composes one anchor per body")
        body.set("pos", _fmt(_vec(body.get("pos")) + _rotate(mj_model.body_quat[bid], p)))
        for c in body:
            if c.tag not in _SHIFT_TAGS:
                continue
            if c.tag == "joint":
                jid_c = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, c.get("name")) if c.get("name") else -1
                base = np.array(mj_model.jnt_pos[jid_c]) if jid_c >= 0 else _vec(c.get("pos"))
                c.set("pos", _fmt(base - p))
                continue
            if c.get("fromto"):
                ft = _vec(c.get("fromto"), 6)
                c.set("fromto", _fmt(np.r_[ft[:3] - p, ft[3:] - p]))
            else:
                c.set("pos", _fmt(_vec(c.get("pos")) - p))
        moved[name] = p
        if verbose:
            print(f"[reorigin] body {name!r}: anchor {p.tolist()} of joint {jname!r} moved to the body origin")
    return ET.tostring(root, encoding="unicode"), moved


def inject_inertials(xml_string, mj_model):
    """Set every body's <inertial> to MuJoCo's compiled mass / COM / orientation / principal inertia.

    hip_env.inject_mujoco_inertial matches bodies by name, so unnamed bodies keep the GPU simulator's own geom-based
    inertia; robocasa puts each object's and container's geoms on an unnamed child body, which the GPU simulator
    then weighed from all its geoms, visual meshes included (a pan far heavier than in MuJoCo). MuJoCo
    numbers bodies depth-first in document order, as ElementTree iterates them, so bodies are matched by
    index (names are checked where present). Massless bodies (mocap targets) are left as they are."""
    import mujoco
    root = ET.fromstring(xml_string)
    wb = root.find("worldbody")
    bodies = list(wb.iter("body"))
    if len(bodies) != mj_model.nbody - 1:
        raise ValueError(f"{len(bodies)} <body> elements vs MuJoCo nbody-1 = {mj_model.nbody - 1}")
    for i, b in enumerate(bodies, start=1):
        nm = b.get("name")
        if nm and mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) != nm:
            raise ValueError(f"body order mismatch at {i}: XML {nm!r} vs MuJoCo "
                             f"{mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)!r}")
        mass = float(mj_model.body_mass[i])
        if mass <= 0.0:
            continue
        for old in b.findall("inertial"):
            b.remove(old)
        inert = ET.Element("inertial", {
            "pos": _fmt(mj_model.body_ipos[i]), "quat": _fmt(mj_model.body_iquat[i]),
            "mass": repr(mass), "diaginertia": _fmt(mj_model.body_inertia[i])})
        b.insert(0, inert)
    return ET.tostring(root, encoding="unicode")
