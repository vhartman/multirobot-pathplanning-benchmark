"""Collision-geometry alternatives retaining the released visual meshes.

Generic box/capsule fits enclose the fitted convex mesh pieces. The supplied
Panda capsule and sphere models instead replace moving links and use their own
self-collision exclusions; they are not guaranteed mesh enclosures. These Panda
variants retain precise hands, fingers and welding tools. Audit resulting paths
against source geometry and report goal feasibility separately.
"""

import math
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh


def fit_primitive(vertices, allow_capsule=True):
    points = np.unique(np.asarray(vertices, dtype=float), axis=0)
    hull = ConvexHull(points)
    points = points[hull.vertices]
    transform, spans = trimesh.bounds.oriented_bounds(points)
    pose = np.linalg.inv(transform)
    spans = np.maximum(spans, 1e-7) + 2e-7
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    if np.max(np.abs(local) - spans / 2) > 1e-8:
        raise ValueError("Box fit does not contain collision vertices")
    box = dict(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=spans / 2,
        pos=pose[:3, 3],
        rotation=pose[:3, :3],
        volume=np.prod(spans),
    )
    if not allow_capsule:
        return box
    # Try all OBB axes; retain the smallest containing capsule.
    best = box
    for axis in range(3):
        z = pose[:3, axis]
        x = pose[:3, (axis + 1) % 3]
        y = np.cross(z, x)
        R = np.column_stack([x, y, z])
        center = points.mean(axis=0)
        q = (points - center) @ R
        radius = float(np.max(np.linalg.norm(q[:, :2], axis=1))) + 1e-7
        reach = np.sqrt(np.maximum(0, radius**2 - np.sum(q[:, :2] ** 2, axis=1)))
        low = float(np.min(q[:, 2] + reach))
        high = float(np.max(q[:, 2] - reach))
        if low > high:
            low = high = (low + high) / 2
        length = high - low
        pos = center + z * (low + high) / 2
        volume = math.pi * radius**2 * length + 4 / 3 * math.pi * radius**3
        if volume < best["volume"]:
            best = dict(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=np.array([radius, length / 2, 0.0]),
                pos=pos,
                rotation=R,
                volume=volume,
            )
    return best


def partition_hull(points, count):
    """Split a convex solid; include cut-plane vertices, not just input samples."""
    parts = [np.unique(points, axis=0)]
    while len(parts) < count:
        scores = []
        for p in parts:
            hull = ConvexHull(p)
            _, spans = trimesh.bounds.oriented_bounds(p[hull.vertices])
            scores.append(max(0.0, np.prod(spans) - hull.volume))
        i = int(np.argmax(scores))
        if scores[i] < 1e-10:
            break
        p = parts.pop(i)
        hull = ConvexHull(p)
        T, spans = trimesh.bounds.oriented_bounds(p[hull.vertices])
        axis = T[int(np.argmax(spans)), :3]
        d = p @ axis
        cut = (d.min() + d.max()) / 2
        edges = set()
        for tri in hull.simplices:
            for j in range(3):
                edges.add(tuple(sorted((int(tri[j]), int(tri[(j + 1) % 3])))))
        intersections = []
        for a, b in edges:
            if (d[a] - cut) * (d[b] - cut) < 0:
                intersections.append(
                    p[a] + (p[b] - p[a]) * (cut - d[a]) / (d[b] - d[a])
                )
        cross = np.asarray(intersections).reshape(-1, 3)
        parts.extend(
            [
                np.unique(np.vstack([p[d <= cut], cross]), axis=0),
                np.unique(np.vstack([p[d >= cut], cross]), axis=0),
            ]
        )
    return parts


def compile_geometry(spec, geometry="source", sphere_model=None, preserve_meshes=()):
    if geometry not in ("source", "primitives", "panda_capsules", "panda_spheres"):
        raise ValueError("Unknown collision geometry variant")
    if geometry == "source":
        return spec.compile()
    # Stable names let us map compiled mesh coordinates back to their spec.
    for i, g in enumerate(spec.geoms):
        if not g.name:
            g.name = f"primitive_source_{i}"
    # Keep handles: adding geoms invalidates name-to-index lookup in MuJoCo 3.3.5.
    model = spec.compile()
    cache = {}
    source_geoms = {g.name: g for g in spec.geoms}
    capsule_bodies = {}
    sphere_bodies = {}
    if geometry == "panda_spheres":
        if sphere_model is None:
            raise ValueError("Prepare sphere_model from a Panda URDF/SRDF first")
        import re

        for b in range(model.nbody):
            match = re.fullmatch(r"(panda\d+)/(link[1-7])", model.body(b).name)
            if match:
                robot, link = match.groups()
                joint = sphere_model["joints"][link]
                expected = Rotation.from_euler("xyz", joint["rpy"]).as_matrix()
                actual = Rotation.from_quat(np.roll(model.body_quat[b], -1)).as_matrix()
                if not np.allclose(
                    model.body_pos[b], joint["position"], atol=1e-8
                ) or not np.allclose(actual, expected, atol=1e-8):
                    raise ValueError(f"Sphere-model frame mismatch for {robot}/{link}")
                sphere_bodies[b] = (robot, link)
    if geometry == "panda_capsules":
        # Use the shipped RAI Panda capsule model, with its two-joint
        # neighborhood exclusions. It is an approximation, not an enclosure.
        import robotic as ry

        canonical = ry.Config()
        canonical.addFile(ry.raiPath("panda/panda.g"))
        import re

        for b in range(model.nbody):
            name = model.body(b).name
            # The shipped base capsule extends below the mounting plane.
            # Retain the enclosing fit for link0 at the release's table height.
            match = re.fullmatch(r"(panda\d+)/link([1-7])", name)
            if match:
                capsule_bodies[b] = (match[1], int(match[2]))
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or not (
            model.geom_contype[g] or model.geom_conaffinity[g]
        ):
            continue
        if model.geom(g).name in preserve_meshes:
            continue
        mid = int(model.geom_dataid[g])
        body = model.body(int(model.geom_bodyid[g])).name
        if (
            body.endswith(("/hand", "/left_finger", "/right_finger"))
            or (geometry in ("panda_capsules", "panda_spheres") and "/tool" in body)
            or (geometry == "panda_spheres" and body.endswith("/link0"))
        ):
            continue
        if (
            int(model.geom_bodyid[g]) in capsule_bodies
            or int(model.geom_bodyid[g]) in sphere_bodies
        ):
            geom = source_geoms[model.geom(g).name]
            geom.contype = geom.conaffinity = 0
            continue
        ancestor = int(model.geom_bodyid[g])
        robot = False
        while ancestor:
            if model.body(ancestor).name.startswith("panda"):
                robot = True
                break
            ancestor = int(model.body_parentid[ancestor])
        if not robot:
            # The bench is axis-aligned in its body frame. A minimum-volume
            # rotated box can protrude above its flat mounting surface.
            start = int(model.mesh_vertadr[mid])
            n = int(model.mesh_vertnum[mid])
            R = Rotation.from_quat(np.roll(model.geom_quat[g], -1)).as_matrix()
            points = model.mesh_vert[start : start + n] @ R.T + model.geom_pos[g]
            lo, hi = points.min(axis=0), points.max(axis=0)
            target = source_geoms[model.geom(g).name]
            target.type = mujoco.mjtGeom.mjGEOM_BOX
            target.meshname = ""
            target.pos = (lo + hi) / 2
            target.quat = [1, 0, 0, 0]
            target.size = (hi - lo) / 2 + 1e-7
            continue
        key = (mid, robot)
        if key not in cache:
            start = int(model.mesh_vertadr[mid])
            n = int(model.mesh_vertnum[mid])
            points = model.mesh_vert[start : start + n].astype(float)
            points = points[ConvexHull(points).vertices]
            cache[key] = [
                fit_primitive(part, robot)
                for part in partition_hull(points, 8 if robot else 32)
            ]
        original = source_geoms[model.geom(g).name]
        R = Rotation.from_quat(np.roll(model.geom_quat[g], -1)).as_matrix()
        for i, fit in enumerate(cache[key]):
            target = (
                original
                if i == 0
                else original.parent.add_geom(
                    name=f"{original.name}_part_{i}",
                    contype=int(model.geom_contype[g]),
                    conaffinity=int(model.geom_conaffinity[g]),
                    group=int(model.geom_group[g]),
                    rgba=model.geom_rgba[g],
                )
            )
            target.type = fit["type"]
            target.meshname = ""
            target.size = fit["size"]
            target.pos = model.geom_pos[g] + R @ fit["pos"]
            target.quat = np.roll(
                Rotation.from_matrix(R @ fit["rotation"]).as_quat(), 1
            )
    if geometry == "panda_spheres":
        for b, (robot, link) in sphere_bodies.items():
            for i, sphere in enumerate(sphere_model["links"][link]):
                spec.body(model.body(b).name).add_geom(
                    name=f"{robot}/panda_sphere_{link}_{i}",
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[sphere["radius"], 0, 0],
                    pos=sphere["position"],
                    contype=1,
                    conaffinity=1,
                    group=3,
                    rgba=[0.4, 0.6, 0.8, 0.0],
                )
        for robot in sorted({robot for robot, _ in sphere_bodies.values()}):
            for a, b in sphere_model["exclusions"]:
                spec.add_exclude(bodyname1=f"{robot}/{a}", bodyname2=f"{robot}/{b}")
    if geometry == "panda_capsules":
        for b, (robot, link) in capsule_bodies.items():
            frame = canonical.getFrame(f"panda_coll{link}")
            size = frame.getSize()
            spec.body(model.body(b).name).add_geom(
                name=f"{robot}/panda_capsule_{link}",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[size[-1], size[-2] / 2, 0],
                pos=frame.getRelativePosition(),
                quat=frame.getRelativeQuaternion(),
                contype=1,
                conaffinity=1,
                group=3,
                rgba=[0.4, 0.6, 0.8, 0.0],
            )
        # The supplied capsule model excludes its two-joint neighborhood,
        # including the fixed base (MuJoCo does not auto-exclude world parents).
        base_bodies = {
            model.body(f"{robot}/link0").id: (robot, 0)
            for robot, _ in capsule_bodies.values()
        }
        neighbors = {**base_bodies, **capsule_bodies}
        for a, (robot, i) in neighbors.items():
            for b, (other, j) in neighbors.items():
                if a < b and robot == other and abs(i - j) <= 2:
                    spec.add_exclude(
                        bodyname1=model.body(a).name, bodyname2=model.body(b).name
                    )
    return spec.compile()


def decompose_workpiece(mesh):
    """VHACD with the RoboBallet release's workpiece settings.

    Only numeric geometry is read from the release; no release Python modules
    or training dependencies are imported.
    """
    import vhacdx

    faces = np.column_stack((np.full(len(mesh.faces), 3, dtype=np.int32), mesh.faces))
    return vhacdx.compute_vhacd(
        points=np.asarray(mesh.vertices),
        faces=faces.ravel(),
        maxConvexHulls=30,
        minimumVolumePercentErrorAllowed=1.0,
        maxRecursionDepth=8,
        findBestPlane=True,
        asyncACD=False,
    )
