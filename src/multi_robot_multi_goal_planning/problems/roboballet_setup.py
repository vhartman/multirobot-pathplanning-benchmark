"""In-memory scene construction and goal preparation for RoboBallet environments."""

from pathlib import Path
import re

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from .roboballet_env import RoboBalletEnvironment
from .roboballet_models import HOME, quaternion
from .roboballet_ik import solve_ik, solve_bounded_ik, refine_goal_region


def make_instance(assets, *, kind="welding", robots=None, targets=40,
                  episode=0, objects=4, seed=0, ik_attempts=12,
                  geometry="source", workpiece_geometry="boxes", drop_uncovered=False):
    """Build a scene description in memory; no files are written.

    Welding uses a released target episode and 4 or 8 robots. Pack/unpack uses
    two robots and 4 or 8 blocks. Reuse the returned mapping to share geometry,
    assignments and IK goals between collision backends.
    """
    if kind == "welding":
        return make_welding_instance(
            assets, robots=8 if robots is None else robots, target_count=targets,
            episode=episode, seed=seed, ik_attempts=ik_attempts,
            geometry=geometry, workpiece_geometry=workpiece_geometry,
            drop_uncovered=drop_uncovered,
        )
    if kind in ("pack", "unpack"):
        if robots not in (None, 2):
            raise ValueError("Packing and unpacking use exactly two robots")
        return make_packing_instance(assets, kind=kind, object_count=objects,
                                     seed=seed, geometry=geometry)
    raise ValueError("kind must be welding, pack or unpack")



def read_targets(assets, count, episode, robots):
    # Parse only the release's documented PredefinedTasks text export. Refuse
    # unexpected shapes instead of silently interpreting another proto schema.
    text = (assets / f"data/paper_test_tasksets/test_{count}targets.pbtxt").read_text()
    episodes = re.split(r"(?m)^target_sets\s*\{", text)[1:]
    if not 0 <= episode < len(episodes):
        raise ValueError(f"Episode {episode} is outside the released {len(episodes)} episodes")
    targets = []
    for entry in re.split(r"\btargets\s*\{", episodes[episode])[1:]:
        arrays = []
        for name, shape in [("position", (3,)), ("rotation", (3, 3))]:
            block = re.search(name + r"\s*\{([^}]+)\}", entry).group(1)
            dims = tuple(map(int, re.findall(r"\bdims:\s*(\d+)", block)))
            if dims != shape:
                raise ValueError(f"Unexpected {name} dimensions: {dims}")
            values = np.array(
                list(map(float, re.findall(r"float_values:\s*([^\s}]+)", block)))
            )
            arrays.append(values.reshape(shape))
        p, R = arrays
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or np.linalg.det(R) < 0.9999:
            raise ValueError("Target rotation is invalid")
        p[2] += 0.6 if robots == 8 else 0.0
        targets.append(dict(position=p.tolist(), rotation=R.tolist()))
    if len(targets) != count:
        raise ValueError("Target count mismatch")
    return targets


def prepare_workpiece(assets):
    """Return enclosing boxes and convex hulls in the workpiece's local frame."""
    import trimesh
    from .roboballet_geometry import decompose_workpiece

    source = Path(assets) / "data/mujoco_world/assets/table_cube/table_cube_collision.stl"
    parts = decompose_workpiece(trimesh.load(source, force="mesh"))
    boxes, hulls = [], []
    for vertices, faces in parts:
        to_origin, spans = trimesh.bounds.oriented_bounds(
            trimesh.Trimesh(vertices, faces, process=False)
        )
        pose = np.linalg.inv(to_origin)
        local = trimesh.transform_points(vertices, to_origin)
        if np.any(np.abs(local) > spans / 2 + 1e-6):
            raise ValueError("OBB does not enclose its collision hull")
        boxes.append(dict(position=pose[:3, 3].tolist(),
                          quaternion=quaternion(pose[:3, :3]).tolist(),
                          size=spans.tolist()))
        hulls.append(dict(vertices=np.asarray(vertices).tolist(),
                          faces=np.asarray(faces).tolist()))
    return boxes, hulls



def make_welding_instance(assets, *, robots=8, target_count=40, episode=0,
                          seed=0, ik_attempts=12, geometry="source",
                          workpiece_geometry="boxes", drop_uncovered=False):
    """Construct welding geometry, assign targets, and solve collision-free IK."""
    assets = Path(assets).expanduser().resolve()
    if robots not in (4, 8) or target_count not in (10, 20, 40):
        raise ValueError("Welding needs 4/8 robots and 10/20/40 targets")
    if episode < 0 or seed < 0 or ik_attempts < 1:
        raise ValueError("Episode/seed must be nonnegative and IK attempts positive")
    if workpiece_geometry not in ("boxes", "mesh"):
        raise ValueError("workpiece_geometry must be boxes or mesh")
    targets = read_targets(assets, target_count, episode, robots)
    boxes, hulls = prepare_workpiece(assets)
    if robots == 8:
        for box in boxes:
            box["position"][2] += 0.6
        for hull in hulls:
            hull["vertices"] = (np.asarray(hull["vertices"]) + [0, 0, 0.6]).tolist()
    instance = dict(
        kind="welding",
        robots=robots,
        targets=targets,
        boxes=boxes,
        tasks=[],
        episode=episode,
        seed=seed,
        collision_resolution=0.05,
        geometry_note="Release meshes; VHACD <=30 parts enclosed by saved OBBs; 8-arm workpiece and targets lifted 0.6m.",
        objective_note="Geometric max per-robot L2 length plus sum tie-break; no dwell or acceleration timing.",
    )
    instance["workpiece_geometry"] = workpiece_geometry
    if workpiece_geometry == "mesh":
        instance["hulls"] = hulls
    env = RoboBalletEnvironment(assets, instance, geometry=geometry)
    if not env.is_collision_free(env.start_pos, env.start_mode):
        raise ValueError(f"Start collides: {env.colliding_pairs()}")
    rng = np.random.default_rng(seed)
    banks = [[[] for _ in targets] for r in env.robots]
    near_errors = np.full((len(env.robots), len(targets)), np.inf)
    for r, robot in enumerate(env.robots):
        idx = env.robot_idx[robot]
        lo, hi = env.limits[:, idx]
        previous = HOME.copy()
        for t, target in enumerate(targets):
            if banks[r][t]:
                continue
            for attempt in range(ik_attempts):
                ik_seed = (
                    previous
                    if attempt == 0
                    else (
                        HOME
                        if attempt == 1
                        else (
                            np.clip(HOME + rng.normal(0, 0.8, 7), lo, hi)
                            if attempt < 5
                            else rng.uniform(lo, hi)
                        )
                    )
                )
                q = solve_ik(env, r, target, ik_seed, return_best=True)
                full = env.start_pos.state().copy()
                full[idx] = q
                env.set_configuration(full, collision=False)
                dp = np.linalg.norm(env.data.site_xpos[env.tcp[r]] - target["position"])
                dr = Rotation.from_matrix(
                    np.asarray(target["rotation"])
                    @ env.data.site_xmat[env.tcp[r]].reshape(3, 3).T
                ).magnitude()
                near_errors[r, t] = min(near_errors[r, t], dp + 0.3 * dr)
                if dp >= 3e-4 or dr >= 2e-3:
                    continue
                if not env.is_collision_free(full, env.start_mode):
                    continue
                if all(np.linalg.norm(q - np.asarray(x)) > 0.1 for x in banks[r][t]):
                    banks[r][t].append(q.tolist())
                    previous = q
                if len(banks[r][t]) >= 2:
                    break
    uncovered = [
        t
        for t in range(len(targets))
        if not any(banks[r][t] for r in range(len(env.robots)))
    ]
    # Some published target centres are not exactly reachable by a collision-free
    # tool. RoboBallet evaluates a 25 mm / 15 degree region. Search that region
    # only for otherwise uncovered targets, keeping the original target intact.
    for t in uncovered:
        nominal = targets[t]
        for r in np.argsort(near_errors[:, t]):
            if near_errors[r, t] > 0.3:
                continue
            robot = env.robots[r]
            idx = env.robot_idx[robot]
            lo, hi = env.limits[:, idx]
            for attempt in range(180):
                if attempt < 60:
                    direction = -np.asarray(nominal["rotation"])[:, 2]
                    radius = [0.005, 0.01, 0.015, 0.02, 0.024][attempt // 12]
                else:
                    direction = rng.normal(size=3)
                    direction /= np.linalg.norm(direction)
                    radius = rng.uniform(0, 0.024)
                position = np.asarray(nominal["position"]) + direction * radius
                axis = rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                rotation = (
                    Rotation.from_rotvec(
                        axis * (0.0 if attempt < 60 else rng.uniform(0, 0.24))
                    ).as_matrix()
                    @ nominal["rotation"]
                )
                adjusted = dict(position=position, rotation=rotation)
                ik_seed = (
                    np.clip(HOME + rng.normal(0, 1.0, 7), lo, hi)
                    if attempt < 60
                    else rng.uniform(lo, hi)
                )
                q = solve_ik(env, r, adjusted, ik_seed)
                if q is None:
                    continue
                full = env.start_pos.state().copy()
                full[idx] = q
                if env.is_collision_free(full, env.start_mode):
                    banks[r][t].append(q.tolist())
                    if len(banks[r][t]) >= 2:
                        break
    uncovered = [
        t
        for t in range(len(targets))
        if not any(banks[r][t] for r in range(len(env.robots)))
    ]
    for t in uncovered:
        for r in np.argsort(near_errors[:, t]):
            robot = env.robots[r]
            idx = env.robot_idx[robot]
            for attempt in range(40):
                q = solve_bounded_ik(
                    env,
                    r,
                    targets[t],
                    rng.uniform(*env.limits[:, idx]),
                    rotation_weight=[0.1, 0.2, 0.3][attempt % 3],
                )
                full = env.start_pos.state().copy()
                full[idx] = q
                env.set_configuration(full, collision=False)
                dp = np.linalg.norm(
                    env.data.site_xpos[env.tcp[r]] - targets[t]["position"]
                )
                dr = Rotation.from_matrix(
                    np.asarray(targets[t]["rotation"])
                    @ env.data.site_xmat[env.tcp[r]].reshape(3, 3).T
                ).magnitude()
                if dp > 0.12 or dr > 0.6:
                    continue
                q = refine_goal_region(env, r, targets[t], q)
                if q is not None:
                    banks[r][t].append(q.tolist())
                    break
    uncovered = [
        t
        for t in range(len(targets))
        if not any(banks[r][t] for r in range(len(env.robots)))
    ]
    instance["uncovered_targets"] = uncovered
    if uncovered and not drop_uncovered:
        raise RuntimeError(
            f"No collision-free IK found for targets {uncovered}; increase IK attempts or inspect geometry"
        )
    if uncovered:
        import warnings
        warnings.warn(f"Dropping welding targets without collision-free IK: {uncovered}")
    current = np.tile(HOME, (len(env.robots), 1))
    remaining = [t for t in range(len(targets)) if t not in uncovered]
    while remaining:
        costs = np.full((len(env.robots), len(remaining)), 1e6)
        choices = {}
        for r in range(len(env.robots)):
            for col, t in enumerate(remaining):
                for candidate in banks[r][t]:
                    q = np.asarray(candidate)
                    # Prefer small travel and comfortable distance from joint limits.
                    lo, hi = env.limits[:, env.robot_idx[env.robots[r]]]
                    score = np.linalg.norm(q - current[r]) + 0.03 * np.sum(
                        1 / (0.1 + np.minimum(q - lo, hi - q))
                    )
                    if score < costs[r, col]:
                        costs[r, col] = score
                        choices[r, col] = q
        rows, cols = linear_sum_assignment(costs)
        selected = []
        for r, col in zip(rows, cols):
            if costs[r, col] >= 1e6:
                continue
            t, q = remaining[col], choices[r, col]
            instance["tasks"].append(
                dict(name=f"weld_{t:02d}", robot=int(r), target=t, q=q.tolist())
            )
            current[r] = q
            selected.append(t)
        if not selected:
            raise RuntimeError("Assignment made no progress")
        remaining = [t for t in remaining if t not in selected]
    errors = []
    for task in instance["tasks"]:
        full = env.start_pos.state().copy()
        r = task["robot"]
        full[env.robot_idx[env.robots[r]]] = task["q"]
        env.set_configuration(full)
        target = targets[task["target"]]
        p_error = np.linalg.norm(env.data.site_xpos[env.tcp[r]] - target["position"])
        r_error = Rotation.from_matrix(
            np.asarray(target["rotation"])
            @ env.data.site_xmat[env.tcp[r]].reshape(3, 3).T
        ).magnitude()
        if p_error >= 0.025 or r_error >= np.deg2rad(15) or env.colliding_pairs():
            raise ValueError("Selected goal fails collision/Cartesian validation")
        errors.append(
            dict(target=task["target"], position=float(p_error), angle=float(r_error))
        )
    instance["goal_errors"] = errors
    instance["ik_attempts"] = ik_attempts
    env.close()
    return instance



def make_packing_instance(assets, *, kind="pack", object_count=4, seed=0,
                          geometry="source"):
    """Construct a two-Panda pack/unpack task with fixed grasp transforms."""
    if kind not in ("pack", "unpack") or object_count not in (4, 8) or seed < 0:
        raise ValueError("Use pack/unpack, 4/8 objects, and a nonnegative seed")
    if object_count == 4:
        outside = [
            [-0.28, -0.23, 0.032],
            [-0.28, 0.23, 0.032],
            [0.28, -0.23, 0.032],
            [0.28, 0.23, 0.032],
        ]
        inside = [
            [-0.07, -0.07, 0.032],
            [-0.07, 0.07, 0.032],
            [0.07, -0.07, 0.032],
            [0.07, 0.07, 0.032],
        ]
    else:
        outside = [
            [x, y, 0.032] for x in (-0.28, 0.28) for y in (-0.30, -0.10, 0.10, 0.30)
        ]
        inside = [
            [x, y, 0.032] for x in (-0.07, 0.07) for y in (-0.18, -0.06, 0.06, 0.18)
        ]
    starts, ends = (outside, inside) if kind == "pack" else (inside, outside)
    instance = dict(
        kind=kind,
        robots=2,
        targets=[],
        boxes=[],
        tasks=[],
        seed=seed,
        objects=[dict(start=a, goal=b) for a, b in zip(starts, ends)],
        collision_resolution=0.05,
        packing_region_half_size=[0.18, 0.16 if object_count == 4 else 0.23],
        geometry_note=f"Two original Panda models; recreated {object_count}-block pack/unpack layout, not an exact cuArms instance.",
        objective_note="Geometric joint-space cost, no gripper actuation or acceleration timing.",
    )
    env = RoboBalletEnvironment(assets, instance, geometry=geometry)
    if not env.is_collision_free(env.start_pos, env.start_mode):
        raise ValueError(f"Packing start collides: {env.colliding_pairs()}")
    rng = np.random.default_rng(seed)
    for r in range(2):
        current = HOME.copy()
        idx = env.robot_idx[env.robots[r]]
        lo, hi = env.limits[:, idx]
        per_robot = object_count // 2
        for obj in range(per_robot * r, per_robot * (r + 1)):
            for action, pos in [("pick", starts[obj]), ("place", ends[obj])]:
                target = dict(position=pos, rotation=np.diag([1.0, -1.0, -1.0]))
                candidates = []
                for attempt in range(40):
                    ik_seed = (
                        current
                        if attempt == 0
                        else np.clip(HOME + rng.normal(0, 0.8, 7), lo, hi)
                    )
                    q = solve_ik(env, r, target, ik_seed)
                    if q is None:
                        continue
                    full = env.start_pos.state().copy()
                    full[idx] = q
                    if not env.is_collision_free(full, env.start_mode):
                        continue
                    candidates.append(q)
                    if len(candidates) >= 4:
                        break
                if not candidates:
                    raise RuntimeError(
                        f"No packing IK: robot {r}, object {obj}, {action}"
                    )
                q = min(candidates, key=lambda q: np.linalg.norm(q - current))
                instance["tasks"].append(
                    dict(
                        name=f"{action}_{obj}",
                        robot=r,
                        object=obj,
                        action=action,
                        q=q.tolist(),
                    )
                )
                current = q
    env.close()
    return instance



def add_welding_goal_set(assets, instance, target, *, attempts=1500, seed=91,
                         geometry="source"):
    """Return a copied description with IK alternatives for one weld target."""
    if attempts < 1 or seed < 0:
        raise ValueError("Attempts must be positive and seed nonnegative")
    env = RoboBalletEnvironment(assets, instance, geometry=geometry)
    if env.packing:
        raise ValueError("Goal alternatives require a welding instance")
    tasks = [t for t in env.instance["tasks"] if t.get("target") == target]
    if len(tasks) != 1:
        raise ValueError("Expected exactly one task for the requested welding target")
    task = tasks[0]
    robot = task["robot"]
    idx = env.robot_idx[env.robots[robot]]
    lo, hi = env.limits[:, idx]
    goal = env.instance["targets"][target]
    rotation, position = np.asarray(goal["rotation"]), np.asarray(goal["position"])
    rng = np.random.default_rng(seed)
    candidates = []
    for attempt in range(attempts):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        pos = position + axis * rng.uniform(0, .024)
        rot = Rotation.from_rotvec(rng.normal(size=3) * .10).as_matrix() @ rotation
        initial = (HOME if attempt % 4 == 0 else
                   np.clip(HOME + rng.normal(size=7), lo, hi) if attempt % 4 == 1
                   else rng.uniform(lo, hi))
        q = solve_ik(env, robot, dict(position=pos, rotation=rot), initial, return_best=True)
        full = env.start_pos.state().copy()
        full[idx] = q
        free = env.is_collision_free(full, env.start_mode)
        dp = np.linalg.norm(env.data.site_xpos[env.tcp[robot]] - position)
        dr = Rotation.from_matrix(
            rotation @ env.data.site_xmat[env.tcp[robot]].reshape(3, 3).T
        ).magnitude()
        if free and dp < .025 and dr < np.deg2rad(15):
            candidates.append(q.tolist())
    if not candidates:
        raise RuntimeError("No valid IK alternatives found")
    existing = task["q_candidates"] if "q_candidates" in task else [task["q"]]
    task["q_candidates"] = existing + candidates
    env.close()
    return env.instance
