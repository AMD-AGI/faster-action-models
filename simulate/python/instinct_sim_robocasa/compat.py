# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import robocasa under MuJoCo 3.3.2 (the version the GPU simulator follows).

robocasa/__init__.py asserts mujoco.__version__ == "3.2.6". This presents that string only
while robocasa is imported, without editing the shared robocasa checkout the 3.2.6 CPU reference
client also uses.
"""
import sys


def import_robocasa():
    if "robocasa" in sys.modules:
        return sys.modules["robocasa"]
    import mujoco
    real = mujoco.__version__
    mujoco.__version__ = "3.2.6"
    try:
        import robocasa
    finally:
        mujoco.__version__ = real
    tolerate_tiny_meshes()
    return robocasa


def tolerate_tiny_meshes():
    """MuJoCo 3.2.6 takes shell inertia per geom (<geom shellinertia="true">, robocasa's hamilton_beach
    microwave panels); 3.3 moved it to the mesh asset (<mesh inertia="shell">) and computes volume-based
    inertia for those meshes instead, so their thin panels fail with "mesh volume is too small". Translate
    the 3.2.6 form before compiling: mesh geoms with shellinertia="true" set inertia="shell" on their mesh
    and drop the geom attribute (same surface-based mass as 3.2.6). Logged per scene."""
    import xml.etree.ElementTree as ET
    import mujoco
    if getattr(mujoco.MjModel, "_hp_shell_xlate", False):
        return
    orig = mujoco.MjModel.from_xml_string

    def from_xml_string(xml, *a, **k):
        if "shellinertia" in xml:
            root = ET.fromstring(xml)
            shell, other = set(), set()
            for g in root.iter("geom"):
                mesh = g.get("mesh")
                if g.get("shellinertia") is not None and mesh:
                    if g.get("shellinertia") in ("true", "1"):
                        shell.add(mesh)
                    del g.attrib["shellinertia"]
                elif mesh:
                    other.add(mesh)
            for me in root.iter("mesh"):
                if me.get("name") in shell:
                    me.set("inertia", "shell")
            if shell & other:
                print(f"[compat] WARNING meshes used with and without shellinertia: {sorted(shell & other)}", flush=True)
            if shell:
                print(f"[compat] shellinertia -> mesh inertia=shell for {len(shell)} meshes", flush=True)
            xml = ET.tostring(root, encoding="unicode")
        return orig(xml, *a, **k)

    mujoco.MjModel.from_xml_string = staticmethod(from_xml_string)
    mujoco.MjModel._hp_shell_xlate = True
