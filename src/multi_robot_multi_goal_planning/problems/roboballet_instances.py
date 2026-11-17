"""Load frozen scene descriptions without running IK or importing release code."""

from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os

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


RELEASE = dict(
    record="https://zenodo.org/records/16546403",  # roboballet.tar.zst, 1.1 GB, Apache-2.0
    md5="659f63c725f2237b3ccd10a9ba8f5717",
)
# The 44 MB of the release that MRMG reads (data/mujoco_world, data/paper_test_tasksets), unmodified, with the
# release's LICENSE and a NOTICE, plus frozen instances of the registered presets.
PACKAGED_ASSETS = Path(__file__).resolve().parent.parent / "assets" / "roboballet"
_CACHE_VERSION = 1  # bump when instance construction changes


def _cache_root():
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mrmg" / "roboballet"


def default_assets():
    """$MRMG_ROBOBALLET_ASSETS if set, else the release subset packaged with MRMG."""
    return Path(os.environ.get("MRMG_ROBOBALLET_ASSETS") or PACKAGED_ASSETS)


def _assets_id(assets):
    """The packaged subset is identified by the release checksum, so frozen instances apply wherever MRMG is
    installed. Other asset trees are identified by their path."""
    path = Path(assets).expanduser().resolve()
    return f"zenodo-16546403-{RELEASE['md5']}" if path == PACKAGED_ASSETS.resolve() else str(path)


def instance_name(options):
    key = dict(options, assets=_assets_id(options["assets"]), cache_version=_CACHE_VERSION)
    return f"{hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]}.json"


def cached_instance(build, options):
    """Return the instance for these construction options without rerunning IK when possible.

    Lookup order: the frozen instances packaged with MRMG (assets/roboballet/instances, written by
    scripts/freeze_roboballet_instances.py), then the user cache (~/.cache/mrmg/roboballet/instances), else build()
    and store it in the user cache. Preparation runs IK and convex decomposition, minutes for welding. Set
    MRMG_ROBOBALLET_CACHE=0 to always rebuild.
    """
    if os.environ.get("MRMG_ROBOBALLET_CACHE", "1") == "0":
        return build()
    name = instance_name(options)
    for path in (PACKAGED_ASSETS / "instances" / name, _cache_root() / "instances" / name):
        if path.is_file():
            return json.loads(path.read_text())
    instance = build()
    path = _cache_root() / "instances" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(instance))
    tmp.replace(path)
    return instance


def resolve_assets(directory, instance):
    """Resolve the release root and fail before model compilation if missing."""
    directory = Path(default_assets() if directory is None else directory).expanduser().resolve()
    relative = (
        f"data/mujoco_world/{instance['robots']}_pandas_world.xml"
        if instance["kind"] == "welding"
        else "data/mujoco_world/assets/franka_emika_panda/panda.xml"
    )
    if not (directory / relative).is_file():
        raise FileNotFoundError(
            f"Missing RoboBallet asset: {directory / relative}. MRMG ships the needed part of the release in "
            f"{PACKAGED_ASSETS}. To use another copy of {RELEASE['record']}, pass its release root (the "
            f"'roboballet' directory of the unpacked archive) as assets or in MRMG_ROBOBALLET_ASSETS."
        )
    return directory
