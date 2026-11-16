"""Coordination environments: welding lines, shared bins, lidded containers and corridor swaps.

Arm keyframes use deterministic IK and single-robot reachability checks. Shared
rail goals are checked with neighbouring carriages moved aside; this does not
certify a complete coordinated path. Corridor feasibility is not certified by
the builder. Container goals follow rigid pick/place attachment semantics.

Sequence and dependency-graph variants are registered below. Models are loaded
from the package assets; scene generation leaves global random streams intact.
"""
from pathlib import Path
import logging

import numpy as np
import robotic as ry
from scipy.optimize import least_squares

from ..core.dependency_graph import DependencyGraph
from ..core.goals import SingleGoal
from ..core.registry import register
from ..planning_env import (
    BaseModeLogic,
    DependencyGraphMixin,
    DependencyType,
    GoalType,
    ManipulationType,
    SafePoseType,
    SequenceMixin,
    Task,
)
from ..rai_base_env import rai_env
from .rai_config import get_robot_joints

ASSETS = Path(__file__).resolve().parents[2] / "assets/models/rai"
_LOG = logging.getLogger(__name__)
SILVER, DARK = [0.74, 0.76, 0.80], [0.28, 0.29, 0.31]
PALETTE = [[0.90, 0.40, 0.10], [0.15, 0.50, 0.85], [0.20, 0.65, 0.30], [0.80, 0.20, 0.50], [0.55, 0.35, 0.80],
           [0.85, 0.70, 0.10], [0.10, 0.65, 0.70], [0.60, 0.30, 0.20]]


def _qz(angle):
    return [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]


def _box(C, name, size, pos, color=SILVER, contact=1, parent="world"):
    return C.addFrame(name, parent).setShape(ry.ST.box, list(size)).setPosition(list(pos)).setColor(color).setContact(contact)


def _robot_indices(C, robot):
    return [k for k, n in enumerate(C.getJointNames()) if n.startswith(robot)]


def _moving_frames(C, rng):
    """Frames whose pose depends on some joint (found by perturbing all joints once)."""
    q0 = C.getJointState().copy()
    X0 = C.getFrameState().copy()
    lim = C.getJointLimits()
    C.setJointState(rng.uniform(lim[0], lim[1]))
    X1 = C.getFrameState().copy()
    C.setJointState(q0)
    return {n for n, a, b in zip(C.getFrameNames(), X0, X1) if np.abs(a - b).max() > 1e-9}


def _robot_from_frame(name, robots, carried=None):
    if carried and name in carried:
        return carried[name]
    return next((r for r in robots if name.startswith(r)), None)


def _check_pose(C, robot, robots, moving, carried=None):
    """(ok, conflicts): robot's current pose vs the static scene, itself and other robots' fixed links: no
    penetration at all; contacts with other robots' moving links are allowed and returned. carried: {frame: robot}
    for objects a robot holds (they count as that robot's links)."""
    C.computeCollisions()
    conflicts = set()
    for a, b, d in C.getCollisions():
        if d >= 0.0:
            continue
        oa, ob = _robot_from_frame(a, robots, carried), _robot_from_frame(b, robots, carried)
        if robot not in (oa, ob):
            continue
        other, owner = (b, ob) if oa == robot else (a, oa)
        if owner is not None and owner != robot and other in moving:
            conflicts.add(owner)
            continue
        return False, None
    return True, conflicts


# TODO: Replace this manual IK implementation with a shared solver.
def _ik(C, robot, tip, pos, axis_fs, axis, rng, accept, attempts=16, keep=3, weight=0.5, extra=()):
    """Least-squares IK for tip position + tool axis (rotation about the axis free; `extra` (feature, axis) pairs
    fix more axes, e.g. a full grasp pose) over the robot's joints, random restarts. Among up to `keep` accepted
    solutions the one closest to the robot's current (home) pose wins."""
    names = get_robot_joints(C, robot)
    idx = _robot_indices(C, robot)
    lim = C.getJointLimits()
    lo, hi = lim[0, idx], lim[1, idx]
    q_all0 = C.getJointState().copy()
    home = q_all0[idx].copy()
    pos, axis = np.asarray(pos, float), np.asarray(axis, float)

    def res_jac(q):
        C.setJointState(q, names)
        p, Jp = C.eval(ry.FS.position, [tip])
        v, Jv = C.eval(axis_fs, [tip])
        res, jac = [p - pos, weight * (v - axis)], [Jp[:, idx], weight * Jv[:, idx]]
        for fs, ax in extra:
            u, Ju = C.eval(fs, [tip])
            res.append(weight * (u - np.asarray(ax, float)))
            jac.append(weight * Ju[:, idx])
        return np.concatenate(res), np.vstack(jac)

    found = []
    for attempt in range(attempts):
        q0 = home if attempt == 0 else rng.uniform(lo, hi)
        fit = least_squares(lambda q: res_jac(q)[0], np.clip(q0, lo + 1e-7, hi - 1e-7), jac=lambda q: res_jac(q)[1],
                            bounds=(lo, hi), max_nfev=150, gtol=1e-12, ftol=1e-12, xtol=1e-12)
        r, _ = res_jac(fit.x)
        if np.linalg.norm(r[:3]) < 1e-4 and np.linalg.norm(r[3:]) / weight < 1e-3:
            out = accept()
            if out is not None:
                found.append((np.linalg.norm(fit.x - home), fit.x.copy(), out))
                if len(found) >= keep:
                    break
    C.setJointState(q_all0)
    if not found:
        return None
    _, q, out = min(found, key=lambda f: f[0])
    return q, out


def _reachable_from_home(C, robot, robots, moving, q_goal, rng, iterations=1500, step=0.6, resolution=0.02,
                         carried=None):
    """Single-robot RRT-Connect from the robot's current (home) pose to q_goal, the other robots fixed where they are
    (at home, or pushed along a shared rail), under the strict pose check with no contact with anything, edges checked
    every `resolution` rad (max norm). True if connected within `iterations` extensions (a count, not seconds: the
    scene must not depend on the machine's load); restores the configuration."""
    names = get_robot_joints(C, robot)
    idx = _robot_indices(C, robot)
    q_all0 = C.getJointState().copy()
    lim = C.getJointLimits()
    lo, hi = lim[0, idx], lim[1, idx]
    home = q_all0[idx].copy()

    def edge(a, b):
        n = max(1, int(np.ceil(np.abs(b - a).max() / resolution)))
        for k in range(1, n + 1):
            C.setJointState(a + (b - a) * (k / n), names)
            ok, conf = _check_pose(C, robot, robots, moving, carried)
            if not ok or conf:
                return False
        return True

    try:
        if edge(home, q_goal):
            return True
        trees = [[home.copy()], [np.asarray(q_goal, float).copy()]]
        a, b = 0, 1
        for _ in range(iterations):
            x = trees[b][0] if rng.random() < 0.05 else rng.uniform(lo, hi)
            nodes = trees[a]
            near = nodes[int(np.argmin(np.linalg.norm(np.array(nodes) - x, axis=1)))]
            d = np.linalg.norm(x - near)
            x_new = x if d <= step else near + (x - near) * (step / d)
            if edge(near, x_new):
                nodes.append(x_new)
                other = trees[b]
                if edge(other[int(np.argmin(np.linalg.norm(np.array(other) - x_new, axis=1)))], x_new):
                    return True
            a, b = b, a
        return False
    finally:
        C.setJointState(q_all0)


def _mark_static(env):
    env.spec.manipulation = ManipulationType.STATIC


def _safe_home_if_unblocking(env):
    """Home is a safe pose only if no keyframe touches another robot's home pose."""
    if any(c for cs in env.scene["home_conflicts"].values() for c in cs):
        env.spec.home_pose = SafePoseType.HAS_NO_SAFE_HOME_POSE
    else:
        env._set_default_safe_pose()
