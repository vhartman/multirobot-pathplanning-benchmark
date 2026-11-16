"""MuJoCo models built from an external RoboBallet asset directory.

The release provides the Panda and welding models. Packing layouts are
recreations with rigid mocap attachments, not published cuArms instances.
"""

from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

HOME = np.array([0.0, -0.7854, 0.0, -2.1817, 0.0, 1.3962, 0.0])


def quaternion(matrix):
    """Return a scalar-first quaternion for one rotation matrix."""
    rotation = Rotation.from_matrix(matrix).as_quat()
    return rotation[[3, 0, 1, 2]]


def make_welding_model(
    assets,
    robots,
    boxes,
    targets=(),
    geometry="source",
    hulls=None,
    sphere_model=None,
    workpiece_geometry="auto",
):
    """Retain visual meshes; select frozen hulls or enclosing workpiece boxes."""
    if workpiece_geometry not in ("auto", "mesh", "boxes"):
        raise ValueError("workpiece_geometry must be auto, mesh or boxes")
    if workpiece_geometry == "mesh" and hulls is None:
        raise ValueError("Mesh workpiece requested without prepared hulls")
    use_hulls = hulls is not None and (
        workpiece_geometry == "mesh"
        or (workpiece_geometry == "auto" and geometry == "source")
    )
    path = Path(assets) / "data/mujoco_world" / f"{robots}_pandas_world.xml"
    spec = mujoco.MjSpec.from_file(str(path))
    workpiece = spec.body("workpiece")
    shift = 0.6 if robots == 8 else 0.0
    workpiece.pos = [0, 0, shift]
    for geom in spec.geoms:
        if geom.name.startswith("cube/"):
            geom.contype = geom.conaffinity = 0
    # The release's collision geom is unnamed; disable by its mesh instead.
    for geom in spec.geoms:
        if "table_cube_collision" in geom.meshname:
            geom.contype = geom.conaffinity = 0
    if use_hulls:
        for i, hull in enumerate(hulls):
            name = f"weld_hull_{i}"
            spec.add_mesh(
                name=name,
                uservert=np.asarray(hull["vertices"]).ravel(),
                userface=np.asarray(hull["faces"]).ravel(),
            )
            spec.worldbody.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
                rgba=[0.25, 0.3, 0.35, 0.25],
                group=3,
                contype=1,
                conaffinity=1,
            )
    else:
        for i, box in enumerate(boxes):
            spec.worldbody.add_geom(
                name=f"weld_obstacle_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=box["position"],
                quat=box["quaternion"],
                size=np.asarray(box["size"]) / 2,
                rgba=[0.25, 0.3, 0.35, 0.25],
                group=3,
                contype=1,
                conaffinity=1,
            )
    for i, target in enumerate(targets):
        spec.worldbody.add_site(
            name=f"weld_target_{i}",
            pos=target["position"],
            size=[0.009, 0.009, 0.009],
            rgba=[1.0, 0.12, 0.03, 1.0],
            group=2,
        )
    if geometry == "source":
        return spec.compile()
    from .roboballet_geometry import compile_geometry

    return compile_geometry(
        spec,
        geometry,
        sphere_model,
        preserve_meshes=(
            {f"weld_hull_{i}" for i in range(len(hulls))} if use_hulls else ()
        ),
    )


def make_packing_model(assets, instance, geometry="source"):
    import xml.etree.ElementTree as ET

    root = ET.Element("mujoco")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "model",
        name="panda",
        file=str(
            Path(assets).resolve()
            / "data/mujoco_world/assets/franka_emika_panda/panda.xml"
        ),
    )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 3", diffuse=".8 .8 .8")
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="box",
        pos="0 0 -.045",
        size=".7 .5 .045",
        rgba=".65 .66 .69 1",
    )
    region = instance.get("packing_region_half_size", [0.18, 0.16])
    ET.SubElement(
        world,
        "geom",
        name="region",
        type="box",
        pos="0 0 .0002",
        size=f"{region[0]} {region[1]} .0002",
        rgba=".18 .18 .2 1",
        contype="0",
        conaffinity="0",
    )
    for i, (x, yaw) in enumerate([(-0.58, 0), (0.58, 180)]):
        body = ET.SubElement(
            world, "body", name=f"panda{i+1}", pos=f"{x} 0 .003", euler=f"0 0 {yaw}"
        )
        ET.SubElement(
            body, "attach", model="panda", body="link0", prefix=f"panda{i+1}/"
        )
    colors = [".85 .1 .08 1", ".2 .75 .05 1", ".1 .6 .85 1", ".6 .1 .8 1"]
    for i, obj in enumerate(instance["objects"]):
        body = ET.SubElement(
            world,
            "body",
            name=f"block_{i}",
            mocap="true",
            pos=" ".join(map(str, obj["start"])),
        )
        ET.SubElement(
            body,
            "geom",
            name=f"block_geom_{i}",
            type="box",
            size=".03 .03 .03",
            rgba=colors[i % 4],
        )
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    for i in range(2):
        spec.body(f"panda{i+1}/hand").add_site(
            name=f"panda{i+1}/grasp_tcp", pos=[0, 0, 0.1034], size=[0.003] * 3, group=4
        )
    if geometry == "source":
        return spec.compile()
    from .roboballet_geometry import compile_geometry

    return compile_geometry(spec, geometry, instance.get("sphere_model"))
