# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs for PhysicsEngine.set_mujoco_convex(): MuJoCo's own convex-collision data.

The GPU simulator routes box-mesh and mesh-mesh pairs through a bit-exact port of MuJoCo 3.9's
mjc_Convex (include/hipgl/physics/physics_mjccd.h) when this data is uploaded. The port needs
MuJoCo's compiled mesh data (recentred float mesh_vert, hill-climb graph, merged polygons) and
MuJoCo's geom local poses (mesh geoms live in the recentred mesh frame), so everything comes
from the live MuJoCo model rather than from the GPU simulator's own MJCF parse.
"""
import os

import numpy as np

_MJ_GEOM_BOX = 6
_MJ_GEOM_MESH = 7
_HP_GEOM_BOX = 1
_HP_GEOM_MESH = 4
_DSBL_NATIVECCD = 1 << 18   # mjDSBL_NATIVECCD (MuJoCo 3.9 mjtDisableBit)
_DSBL_MULTICCD = 1 << 19    # mjDSBL_MULTICCD


def _disable_bit(mujoco, name, default):
    try:
        return int(getattr(mujoco.mjtDisableBit, name))
    except Exception:
        return default


def geom_map(hp_model, mj):
    """GPU geom index -> MuJoCo geom index (by name; by index for unnamed geoms)."""
    import mujoco
    ng = int(hp_model.ngeom)
    out = np.empty(ng, dtype=np.int64)
    for k in range(ng):
        name = hp_model.geom_name(k)
        mid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, name) if name else -1
        if mid < 0:
            if k >= mj.ngeom:
                raise ValueError(f"GPU geom {k} ({name!r}) has no MuJoCo counterpart")
            mid = k
        out[k] = mid
    return out


_GEOM_ARRAYS = (("geom_pos", 3), ("geom_quat", 4), ("geom_size", 3), ("geom_friction", 3),
                ("geom_solref", 2), ("geom_solimp", 5))
_JNT_ARRAYS = (("jnt_range", 2), ("jnt_stiffness", 1), ("jnt_damping", 1), ("jnt_armature", 1),
               ("jnt_frictionloss", 1), ("jnt_solref_lim", 2), ("jnt_solimp_lim", 5), ("jnt_pos", 3),
               ("jnt_axis", 3))
_BODY_ARRAYS = (("body_pos", 3), ("body_quat", 4), ("body_mass", 1), ("body_inertia", 3), ("body_ipos", 3),
                ("body_iquat", 4))
# GPU array -> MjModel field (joint rows: MuJoCo keeps damping / armature / frictionloss per dof,
# one dof per row here: hinge / slide; free and ball joints have none of the three).
_MJ_FIELD = {"jnt_solref_lim": "jnt_solref", "jnt_solimp_lim": "jnt_solimp", "jnt_damping": "dof_damping",
             "jnt_armature": "dof_armature", "jnt_frictionloss": "dof_frictionloss"}


def sync_model_doubles(hp_model, mj, tol=1e-5, verbose=None):
    """Overwrite the GPU simulator's double model shadows with the compiled MjModel's doubles.

    load_from_xml parses geom / joint parameters as float and widens them, and the XML it reads prints
    some values to 6 digits (robot0_joint3 armature 1.66667 vs MuJoCo's 5/3). A plate's 5 cm box
    half-sizes and local poses then carry ~1e-9 m of rounding: a few nm on a ~3 um resting box-box
    contact, 0.2% of its reference acceleration. Each row is replaced only when the GPU simulator's value already
    agrees with MuJoCo's to tol (relative, floor 1), i.e. both use the same convention; rows that differ
    by more (mesh geoms in another frame, free-joint ranges, other inertia frames) keep the GPU simulator's value.
    Call before any per-world body pose exists. Returns {array: (rows replaced, rows kept)}.
    Requires PhysicsModel.model_array_d.
    """
    import mujoco, os
    if verbose is None:
        verbose = os.environ.get("HP_MODEL_DOUBLES_VERBOSE") == "1"
    gmap = geom_map(hp_model, mj)

    def by_name(n, label, obj):
        out = np.full(n, -1, dtype=np.int64)
        for k in range(n):
            nm = label(k)
            out[k] = mujoco.mj_name2id(mj, obj, nm) if nm else -1
        return out
    jmap = by_name(int(hp_model.njnt), hp_model.joint_name, mujoco.mjtObj.mjOBJ_JOINT)
    bmap = by_name(int(hp_model.nbody), hp_model.body_name, mujoco.mjtObj.mjOBJ_BODY)
    report = {}
    for name, w in _GEOM_ARRAYS + _JNT_ARRAYS + _BODY_ARRAYS:
        kind = name.split("_")[0]
        idx, label = {"geom": (gmap, hp_model.geom_name), "jnt": (jmap, hp_model.joint_name),
                      "body": (bmap, hp_model.body_name)}[kind]
        field = np.asarray(getattr(mj, _MJ_FIELD.get(name, name)), dtype=np.float64)
        try:
            cur = np.asarray(hp_model.model_array_d(name), dtype=np.float64).reshape(-1, w)
        except ValueError as e:      # array not exposed by this build
            report[name] = str(e)
            continue
        new = cur.copy()
        done = kept = 0
        for k in range(cur.shape[0]):
            i = int(idx[k])
            if i < 0:
                kept += 1
                if verbose:
                    print(f"  sync_model_doubles kept {name}[{k}] {label(k)!r}: no MuJoCo counterpart")
                continue
            src = np.atleast_1d(field[int(mj.jnt_dofadr[i])] if name in ("jnt_damping", "jnt_armature",
                                                                         "jnt_frictionloss") else field[i])
            if np.all(np.abs(src - cur[k]) <= tol * np.maximum(1.0, np.abs(src))):
                new[k] = src
                done += 1
            else:
                kept += 1
                if verbose:
                    print(f"  sync_model_doubles kept {name}[{k}] {label(k)!r}: hip {cur[k].tolist()} mj {src.tolist()}")
        if np.any(new != cur):
            hp_model.model_array_d(name, np.ascontiguousarray(new.reshape(-1)))
        report[name] = (done, kept)
    return report


def mujoco_convex_data(hp_model, mj):
    """Dict for PhysicsEngine.set_mujoco_convex(data) from a compiled mujoco.MjModel.

    mj must be the LIVE model the reference simulation uses (e.g. robosuite's
    env.sim.model._model), not a recompile of get_xml().
    """
    import mujoco
    gmap = geom_map(hp_model, mj)
    ng = len(gmap)
    geom_mj_mesh = np.full(ng, -1, dtype=np.int32)
    for k, mid in enumerate(gmap):
        hp_t, mj_t = int(hp_model.geom_type(k)), int(mj.geom_type[mid])
        if (hp_t == _HP_GEOM_MESH) != (mj_t == _MJ_GEOM_MESH) or (hp_t == _HP_GEOM_BOX) != (mj_t == _MJ_GEOM_BOX):
            raise ValueError(f"geom {k} type mismatch: GPU {hp_t} vs MuJoCo {mj_t}")
        if mj_t == _MJ_GEOM_MESH:
            geom_mj_mesh[k] = int(mj.geom_dataid[mid])
    dflags = int(mj.opt.disableflags)
    d = dict(
        geom_mj_mesh=geom_mj_mesh,
        geom_mj_pos=np.ascontiguousarray(np.asarray(mj.geom_pos, dtype=np.float64)[gmap]),
        geom_mj_quat=np.ascontiguousarray(np.asarray(mj.geom_quat, dtype=np.float64)[gmap]),
        geom_mj_size=np.ascontiguousarray(np.asarray(mj.geom_size, dtype=np.float64)[gmap]),
        # margin > 0 pairs run MuJoCo's multiCCD perturbation loop, whose tolerance uses rbound
        geom_mj_rbound=np.ascontiguousarray(np.asarray(mj.geom_rbound, dtype=np.float64)[gmap]),
        mesh_vertadr=np.asarray(mj.mesh_vertadr, dtype=np.int32),
        mesh_vertnum=np.asarray(mj.mesh_vertnum, dtype=np.int32),
        mesh_vert=np.ascontiguousarray(np.asarray(mj.mesh_vert, dtype=np.float32).reshape(-1)),
        mesh_graphadr=np.asarray(mj.mesh_graphadr, dtype=np.int32),
        mesh_graph=np.asarray(mj.mesh_graph, dtype=np.int32),
        mesh_polynum=np.asarray(mj.mesh_polynum, dtype=np.int32),
        mesh_polyadr=np.asarray(mj.mesh_polyadr, dtype=np.int32),
        mesh_polynormal=np.ascontiguousarray(np.asarray(mj.mesh_polynormal, dtype=np.float64).reshape(-1)),
        mesh_polyvertadr=np.asarray(mj.mesh_polyvertadr, dtype=np.int32),
        mesh_polyvertnum=np.asarray(mj.mesh_polyvertnum, dtype=np.int32),
        mesh_polyvert=np.asarray(mj.mesh_polyvert, dtype=np.int32),
        mesh_polymapadr=np.asarray(mj.mesh_polymapadr, dtype=np.int32),
        mesh_polymapnum=np.asarray(mj.mesh_polymapnum, dtype=np.int32),
        mesh_polymap=np.asarray(mj.mesh_polymap, dtype=np.int32),
        ccd_iterations=int(mj.opt.ccd_iterations),
        ccd_tolerance=float(mj.opt.ccd_tolerance),
        disable_nativeccd=int(bool(dflags & _disable_bit(mujoco, "mjDSBL_NATIVECCD", _DSBL_NATIVECCD))),
        # MuJoCo <= 3.3.x: multiCCD is opt-in (mjENBL_MULTICCD, off in LIBERO), so box-mesh / mesh-mesh pairs get
        # one contact; 3.4+ made it a disable flag (on by default) and adds perturbed contacts on flat
        # face-to-face touches (gripper finger on a box: 2 contacts vs 3.3.2's 1). The SR references are
        # MuJoCo 3.3.2, so multiCCD stays off unless HP_MULTICCD=1 (then the live model's flags decide).
        disable_multiccd=(int(bool(dflags & _disable_bit(mujoco, "mjDSBL_MULTICCD", _DSBL_MULTICCD)))
                          if os.environ.get("HP_MULTICCD") == "1" else 1),
    )
    return d
