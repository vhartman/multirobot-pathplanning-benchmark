"""Load frozen scene descriptions without running IK or importing release code."""

from copy import deepcopy
from pathlib import Path
import json

import numpy as np


def _array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def load_instance(source):
    """Read and copy a prepared instance, checking its runtime-facing fields.

    Empty task lists are allowed while preparing goals. Extra provenance fields
    are preserved. The caller's mapping is never retained by reference.
    """
    instance = (
        deepcopy(source)
        if isinstance(source, dict)
        else json.loads(Path(source).read_text())
    )
    if not isinstance(instance, dict):
        raise ValueError("An instance must be a JSON object")
    kind = instance.get("kind")
    if kind not in ("welding", "pack", "unpack"):
        raise ValueError("Instance kind must be welding, pack or unpack")
    robots = instance.get("robots")
    allowed = (4, 8) if kind == "welding" else (2,)
    if type(robots) is not int or robots not in allowed:
        raise ValueError(f"{kind} requires robot count in {allowed}")
    resolution = float(instance.get("collision_resolution", 0.05))
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("collision_resolution must be finite and positive")

    objects = instance.get("objects", [])
    if kind != "welding":
        if not objects:
            raise ValueError("Packing scenes require objects")
        for i, obj in enumerate(objects):
            for key in ("start", "goal"):
                _array(obj[key], (3,), f"objects[{i}].{key}")
        region = _array(
            instance.get("packing_region_half_size", [0.18, 0.16]),
            (2,),
            "packing_region_half_size",
        )
        if np.any(region <= 0):
            raise ValueError("packing_region_half_size must be positive")
    else:
        for i, box in enumerate(instance.get("boxes", [])):
            _array(box["position"], (3,), f"boxes[{i}].position")
            quat = _array(box["quaternion"], (4,), f"boxes[{i}].quaternion")
            size = _array(box["size"], (3,), f"boxes[{i}].size")
            if np.linalg.norm(quat) == 0 or np.any(size <= 0):
                raise ValueError(f"Invalid box pose or size at {i}")
        if not instance.get("boxes") and not instance.get("hulls"):
            raise ValueError("Welding requires prepared workpiece boxes or hulls")
        for i, target in enumerate(instance.get("targets", [])):
            _array(target["position"], (3,), f"targets[{i}].position")
            rotation = _array(target["rotation"], (3, 3), f"targets[{i}].rotation")
            if (
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or np.linalg.det(rotation) < 0.9999
            ):
                raise ValueError(f"Invalid target rotation at {i}")

    names = {"terminal"}
    for task in instance.get("tasks", []):
        name = task.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"Invalid or duplicate task name: {name!r}")
        names.add(name)
        robot = task.get("robot")
        if type(robot) is not int or not 0 <= robot < robots:
            raise ValueError(f"Invalid robot for task {name}")
        goals = task.get("q_candidates", [task.get("q")])
        if not goals:
            raise ValueError(f"Empty goal set for task {name}")
        for goal in goals:
            _array(goal, (7,), f"goal of {name}")
        if kind == "welding":
            if "action" in task:
                raise ValueError("Welding tasks cannot attach objects")
            target = task.get("target")
            if type(target) is not int or not 0 <= target < len(
                instance.get("targets", [])
            ):
                raise ValueError(f"Invalid weld target for task {name}")
        else:
            obj = task.get("object")
            if (
                task.get("action") not in ("pick", "place")
                or type(obj) is not int
                or not 0 <= obj < len(objects)
            ):
                raise ValueError(f"Invalid object action for task {name}")
    return instance


def resolve_assets(directory, instance):
    """Resolve the release root and fail before model compilation if missing."""
    directory = Path(directory).expanduser().resolve()
    relative = (
        f"data/mujoco_world/{instance['robots']}_pandas_world.xml"
        if instance["kind"] == "welding"
        else "data/mujoco_world/assets/franka_emika_panda/panda.xml"
    )
    if not (directory / relative).is_file():
        raise FileNotFoundError(
            f"Missing RoboBallet asset: {directory / relative}. Pass the release root as assets."
        )
    return directory
