# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo-parity setup for instinct_sim.RayTracer.

The RT's default shading already follows MuJoCo's GL renderer (Gouraud at mesh corners and on
MuJoCo's primitive tessellation, MuJoCo texgen, Mesa/llvmpipe fixed-function and sampler
behaviour; textures are filtered the way GPU GL drivers do, configure(sampler=...) selects the
texture model). Three things hip's model does not carry the way MuJoCo does; this module hands them
over from the mjModel the environment already has:

  * mesh normals: hip's STL loader computes its own; MuJoCo's per-corner normals
    (mesh_normal[mesh_facenormal], rotated by mesh_quat) are set with set_mesh_normals;
  * texgen for UV-less meshes: MuJoCo applies settexture() planes in its RECENTRED mesh frame;
    hip baked them in the file frame. set_texgen_planes takes the planes re-expressed in hip's frame;
  * sRGB textures: hip stores them linearised; llvmpipe samples the sRGB bytes with its own
    decode and sRGB mip chain. set_texture_srgb(_data) gives the RT the raw bytes.

  * lights: MuJoCo's light list (active flag, type, lights inside bodies, <default> classes,
    world pose from mj_forward at qpos0) replaces hip's parsed one (set_lights);
  * geom colours: MuJoCo's setMaterial() rule (geom rgba wins over the material's unless it is
    the default 0.5 grey) replaces hip's (material always wins) for the RT (set_geom_rgba);

and two scene-state helpers:
  * libero_site_rgba: LIBERO's visualisation-site toggles (the stove's red burner disk);
  * configure(..., aa='ssaa') is the default (1 sample/pixel at init(..., ssaa=1): same policy SR
    as the raster ssaa4 on LIBERO, cheapest). aa='msaa4' reproduces MuJoCo's 4x MSAA offscreen
    buffer (shade once per covered surface at the pixel centre, u8 resolve) for the lowest MAE.

Usage:
    rt = instinct_sim.RayTracer(); rt.init(hip_model, 256, 256, ssaa=1)
    configure(rt, mjm, hip_model)                # ssaa1; aa="msaa4" for MuJoCo-like 4x MSAA
"""
from __future__ import annotations

import warnings

import numpy as np


def _quat_mat(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mesh_normals(mjm, hip_model):
    """{mesh_id: (3*nface, 3) float32} MuJoCo per-corner normals in hip soup order/frame.
    hip's soup has MuJoCo's face order; meshes whose corner positions disagree (> 1e-4) are skipped."""
    verts, adr, num = hip_model.mesh_soup()
    out = {}
    for mi in range(mjm.nmesh):
        nf = int(mjm.mesh_facenum[mi])
        if num[mi] != 3 * nf or mjm.mesh_normaladr[mi] < 0:
            continue
        R = _quat_mat(mjm.mesh_quat[mi])
        f = np.asarray(mjm.mesh_face[mjm.mesh_faceadr[mi]:mjm.mesh_faceadr[mi] + nf]).reshape(-1)
        vm = np.asarray(mjm.mesh_vert[mjm.mesh_vertadr[mi]:mjm.mesh_vertadr[mi] + mjm.mesh_vertnum[mi]])
        if np.abs(vm[f] @ R.T + mjm.mesh_pos[mi] - verts[adr[mi]:adr[mi] + num[mi]]).max() > 1e-4:
            continue
        fn = np.asarray(mjm.mesh_facenormal[mjm.mesh_faceadr[mi]:mjm.mesh_faceadr[mi] + nf]).reshape(-1)
        nm = np.asarray(mjm.mesh_normal[mjm.mesh_normaladr[mi]:mjm.mesh_normaladr[mi] + mjm.mesh_normalnum[mi]])
        out[mi] = (nm[fn] @ R.T).astype(np.float32)
    return out


def mesh_texgen_planes(mjm):
    """{geom_id: (3,4) float32} MuJoCo settexture() GL_OBJECT_LINEAR planes for textured mesh geoms
    whose mesh has no texcoords, over hip's geom-local position (x_mj = Rm^T (x_hip - pm)).
    2D: s = 0.5 scl0 x - 0.5, t = -0.5 scl1 y - 0.5, scl = texrepeat / geom_size (texuniform:
    texrepeat); cube: (s, t, r) = x_mj (texuniform: x_mj * size)."""
    out = {}
    for g in range(mjm.ngeom):
        if mjm.geom_type[g] != 7 or mjm.geom_matid[g] < 0:
            continue
        mat = mjm.geom_matid[g]
        tex = int(mjm.mat_texid[mat, 1])
        mid = int(mjm.geom_dataid[g])
        if tex < 0 or mid < 0 or mjm.mesh_texcoordadr[mid] >= 0:
            continue
        R = _quat_mat(mjm.mesh_quat[mid])
        pm = np.asarray(mjm.mesh_pos[mid], np.float64)
        size = np.asarray(mjm.geom_size[g], np.float64)
        uni = bool(mjm.mat_texuniform[mat])
        P = np.zeros((3, 4))
        if int(mjm.tex_type[tex]) == 0:
            rep = np.asarray(mjm.mat_texrepeat[mat], np.float64)
            scl = np.array([rep[k] / max(1e-15, size[k]) if size[k] > 0 else rep[k] for k in range(2)])
            if uni:
                scl = np.array([scl[k] * size[k] if size[k] > 0 else scl[k] for k in range(2)])
            for k, (a, b) in enumerate([(0.5 * scl[0], -0.5), (-0.5 * scl[1], -0.5)]):
                e = R[:, k]
                P[k, :3] = a * e
                P[k, 3] = -a * e @ pm + b
        else:
            for k in range(3):
                sk = size[k] if uni else 1.0
                e = R[:, k]
                P[k, :3] = sk * e
                P[k, 3] = -sk * e @ pm
        out[g] = P.astype(np.float32)
    return out


def texture_colorspace(rt, mjm):
    """Flag MuJoCo's sRGB textures and hand the RT their raw sRGB 2D texels."""
    cs = getattr(mjm, "tex_colorspace", None)
    if cs is None:
        return
    srgb = [int(c) == 2 for c in np.asarray(cs)]
    rt.set_texture_srgb(srgb)
    for t, on in enumerate(srgb):
        if on and int(mjm.tex_type[t]) == 0 and int(mjm.tex_nchannel[t]) == 3:
            w, h, a = int(mjm.tex_width[t]), int(mjm.tex_height[t]), int(mjm.tex_adr[t])
            rt.set_texture_srgb_data(t, np.asarray(mjm.tex_data[a:a + w * h * 3], np.uint8).reshape(h, w, 3))


def _light_types(mjm):
    """Per-light 'spot' / 'directional' / 'point' (MuJoCo >= 3.3 light_type; older light_directional)."""
    lt = getattr(mjm, "light_type", None)
    if lt is not None:   # mjtLightType: SPOT 0, DIRECTIONAL 1, POINT 2, IMAGE 3
        return [{0: "spot", 1: "directional"}.get(int(t), "point") for t in np.asarray(lt).reshape(-1)]
    return ["directional" if int(d) else "spot" for d in np.asarray(mjm.light_directional).reshape(-1)]


def mujoco_lights(mjm, warn: bool = True):
    """MuJoCo's light table for RayTracer.set_lights: headlight (mjModel.vis.headlight) then the
    ACTIVE scene lights in model order (mjv makeLights), posed at qpos0 (mj_forward). MuJoCo's GL
    renderer ignores point/image light parameters (Mesa then lights along the eye +Z); the RT
    does the same. Warns about what the RT does not model: shadow-casting lights (castshadow
    defaults to true in MJCF; MuJoCo adds them in a separate shadow-mapped pass) and lights on
    moving bodies (the RT keeps the qpos0 pose for every world)."""
    import mujoco
    d = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, d)
    h = mjm.vis.headlight
    out = [{"type": "directional", "active": bool(h.active), "ambient": list(h.ambient),
            "diffuse": list(h.diffuse), "specular": list(h.specular)}]
    types = _light_types(mjm)
    shadow, moving = [], []
    for i in range(mjm.nlight):
        if not int(mjm.light_active[i]):
            continue
        b = int(mjm.light_bodyid[i])
        nm = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_LIGHT, i) or f"light{i}"
        if int(mjm.light_castshadow[i]):
            shadow.append(nm)
        if b > 0 and (int(mjm.body_weldid[b]) != 0 or int(mjm.body_mocapid[b]) >= 0):
            moving.append(nm)
        out.append({"type": types[i], "active": True, "castshadow": bool(mjm.light_castshadow[i]),
                    "pos": [float(v) for v in d.light_xpos[i]], "dir": [float(v) for v in d.light_xdir[i]],
                    "ambient": [float(v) for v in mjm.light_ambient[i]],
                    "diffuse": [float(v) for v in mjm.light_diffuse[i]],
                    "specular": [float(v) for v in mjm.light_specular[i]],
                    "attenuation": [float(v) for v in mjm.light_attenuation[i]],
                    "cutoff": float(mjm.light_cutoff[i]), "exponent": float(mjm.light_exponent[i])})
    if warn and shadow:
        warnings.warn(f"instinct_sim_rt_mujoco: lights {shadow} cast shadows in MuJoCo (castshadow, the MJCF "
                      "default); the RT does not model shadows and lights them unshadowed. Set "
                      "castshadow='false' in the scene for parity.", stacklevel=3)
    if warn and moving:
        warnings.warn(f"instinct_sim_rt_mujoco: lights {moving} are attached to moving bodies; the RT uses "
                      "their qpos0 pose in every world.", stacklevel=3)
    if len(out) > 9:   # the RT table holds headlight + 8; Mesa lights only 8 anyway
        out = out[:9]
    return out


def mujoco_geom_rgba(mjm, hip_model=None):
    """{geom_id: rgba} MuJoCo's displayed geom colour (engine_vis_visualize.c setMaterial): the
    material's rgba, unless the geom's own rgba differs from the default (0.5, 0.5, 0.5, 1) or
    there is no material. With hip_model, returns {} unless geom ids/names agree."""
    import mujoco
    if hip_model is not None:
        try:
            if int(hip_model.ngeom) != int(mjm.ngeom):
                return {}
        except AttributeError:
            pass
    out = {}
    for g in range(mjm.ngeom):
        rgba = np.asarray(mjm.geom_rgba[g], np.float32)
        mat = int(mjm.geom_matid[g])
        if mat >= 0 and np.array_equal(rgba, np.array([0.5, 0.5, 0.5, 1.0], np.float32)):
            rgba = np.asarray(mjm.mat_rgba[mat], np.float32)
        out[g] = rgba
    if hip_model is not None and hasattr(hip_model, "geom_name"):
        for g in range(mjm.ngeom):
            if (hip_model.geom_name(g) or "") != (mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) or ""):
                return {}
    return out


def libero_site_rgba(mjm, qpos, turn_on: float = 0.5):
    """(nsite, 4) site rgba as LIBERO shows them for joint state qpos: model rgba with LIBERO's
    set_visualization() toggle (alpha -> 1 - alpha) on FlatStove '<prefix>burner' sites whose
    '<prefix>button' hinge is >= turn_on (articulated_objects.py turn_on; LIBERO's update_state()
    runs it before set_visualization() every step). LIBERO-specific (matches sites by name); an
    environment integration should drive the site rgba from its own visualisation step instead."""
    import mujoco
    rgba = np.asarray(mjm.site_rgba, np.float32).copy()
    for i in range(mjm.nsite):
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_SITE, i) or ""
        if not name.endswith("burner"):
            continue
        j = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, name[:-len("burner")] + "button")
        if j < 0:
            continue
        on = float(qpos[mjm.jnt_qposadr[j]]) >= turn_on
        if (rgba[i, 3] <= 0 and on) or (rgba[i, 3] > 0 and not on):
            rgba[i, 3] = 1.0 - rgba[i, 3]
    return rgba


def apply_lights(rt, mjm, warn: bool = True):
    rt.set_lights(mujoco_lights(mjm, warn=warn))


def apply_geom_rgba(rt, mjm, hip_model):
    n = 0
    for g, rgba in mujoco_geom_rgba(mjm, hip_model).items():
        n += bool(rt.set_geom_rgba(g, [float(v) for v in rgba]))
    return n


def apply_ellipsoids(rt, mjm, hip_model):
    """MJCF ellipsoid geoms (hip's physics parses them as boxes) drawn as ellipsoids."""
    n = 0
    if int(getattr(hip_model, "ngeom", mjm.ngeom)) != int(mjm.ngeom):
        return 0
    for g in range(mjm.ngeom):
        if int(mjm.geom_type[g]) == 4:   # mjGEOM_ELLIPSOID
            n += bool(rt.set_geom_ellipsoid(g, [float(v) for v in mjm.geom_size[g]]))
    return n


SAMPLERS = {
    # hardware-GL filtering (float bilinear/trilinear, exact log2 LOD) on the model's mip pyramid
    # (the 'mujoco' preset's sampler)
    "hw": dict(mesa_tex=False, hw_tex=True, hw_mips=False),
    # + an RT-owned glGenerateMipmap box pyramid: closest to GPU-GL frames (the LIBERO datasets'),
    #   but minified textures move away from llvmpipe renders (whose pyramid is double-rounded)
    "hw_mips": dict(mesa_tex=False, hw_tex=True, hw_mips=True),
    # Mesa llvmpipe's 8-bit AoS sampler: bit-faithful to MuJoCo OSMesa/llvmpipe renders
    "llvmpipe": dict(mesa_tex=True, hw_tex=False, hw_mips=False),
}


def configure(rt, mjm, hip_model, aa="ssaa", lights: bool = True, colors: bool = True,
              sampler: str | None = None, look: str = "classic"):
    """MuJoCo-parity RayTracer: preset shading flags plus the mjModel data above.
    aa: 'ssaa' (default: the init ssaa, 1 sample/pixel with init(..., ssaa=1)), 'msaa4' (needs
        init(..., ssaa=1); closest to MuJoCo's 4x MSAA images) or None (leave as is).
    sampler: texture model, one of SAMPLERS, or None for the preset's ('llvmpipe' matches OSMesa renders).
    look: 'classic' (MuJoCo's GL renderer), the only look this build includes."""
    if look != "classic":
        raise ValueError(f"look must be 'classic', got {look!r}: no other look is included")
    rt.set_shading_preset("mujoco")
    if sampler is not None:
        if sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {sorted(SAMPLERS)} or None")
        rt.set_shading(**SAMPLERS[sampler])
    if lights:
        apply_lights(rt, mjm)
    if colors:
        apply_geom_rgba(rt, mjm, hip_model)
    apply_ellipsoids(rt, mjm, hip_model)
    texture_colorspace(rt, mjm)
    for mi, n in mesh_normals(mjm, hip_model).items():
        rt.set_mesh_normals(mi, n)
    for g, P in mesh_texgen_planes(mjm).items():
        rt.set_texgen_planes(g, P)
    if aa:
        rt.set_aa(aa)
    return rt
