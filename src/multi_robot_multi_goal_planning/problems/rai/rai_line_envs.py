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


def _push_neighbours(C, robot, side_robots, robots, moving, gaps=(0.6, 0.7, 0.8, 0.5)):
    """Shared rail: with `robot` at its current pose and the others at home, push the other carriages on its rail out
    of the way (rail order kept, at least `gap` apart, within their limits, arms at home). The full joint vector of
    the first gap whose configuration has no robot of the rail penetrating anything, else None; restores the
    configuration."""
    q_all0 = C.getJointState().copy()
    names = C.getJointNames()
    lim = C.getJointLimits()
    ci = {r: names.index(r + "carriage") for r in side_robots}
    k = side_robots.index(robot)
    try:
        for gap in gaps:
            q = q_all0.copy()
            for j in range(k + 1, len(side_robots)):
                a, b = ci[side_robots[j]], ci[side_robots[j - 1]]
                q[a] = min(max(q[a], q[b] + gap), lim[1, a])
            for j in range(k - 1, -1, -1):
                a, b = ci[side_robots[j]], ci[side_robots[j + 1]]
                q[a] = max(min(q[a], q[b] - gap), lim[0, a])
            C.setJointState(q)
            if all((lambda ok_c: ok_c[0] and not ok_c[1])(_check_pose(C, rr, robots, moving)) for rr in side_robots):
                return q
        return None
    finally:
        C.setJointState(q_all0)


# --------------------------------------------------------------------------------------------- welding line
STEEL = [0.60, 0.63, 0.68]


def _prism_xz(C, name, poly_xz, y0, y1, color=STEEL):
    """Convex prism: a convex polygon in the x-z plane extruded over y in [y0, y1]. Collision shape = visual (rai's
    capsule-convex penetration is exact, unlike cylinder-cylinder)."""
    pts = np.array([[x, y, z] for x, z in poly_xz for y in (y0, y1)], float)
    return C.addFrame(name, "world").setConvexMesh(pts).setColor(color).setContact(1)


def _prism_yz(C, name, poly_yz, x0, x1, color=STEEL):
    pts = np.array([[x, y, z] for y, z in poly_yz for x in (x0, x1)], float)
    return C.addFrame(name, "world").setConvexMesh(pts).setColor(color).setContact(1)


def _roof_z(x):
    """Roof line (top of the roof rails): crown 1.44 m at x 0.2, about 1.37 m at the A and C pillar tops."""
    return 1.38 + 0.06 * (1.0 - ((x - 0.2) / 0.8) ** 2)


def _body_in_white(C):
    """Body in white on a jig, x along the line (front at -x), y across, z up: underbody with tunnel, rockers,
    hinge / slanted A, tapered B and slanted C pillars, crowned roof rails and roof bows, rear quarters with the wheel
    arch cut out, dash and cowl, parcel shelf, front rails, strut towers and upper rails, radiator support frame, rear
    rails and rear panel. Boxes and convex prisms; what is drawn is what collides. Returns the candidate weld spots
    (label, position, outward unit normal); the gun axis is -normal."""
    spots = []
    up = (0.0, 0.0, 1.0)

    def spot(label, p, n):
        n = np.asarray(n, float)
        spots.append((label, np.asarray(p, float), n / np.linalg.norm(n)))

    for y in (-0.40, 0.40):  # jig: skids on posts, locating pins under the underbody
        _box(C, f"jig_skid_{y}", [4.3, 0.10, 0.10], [0, y, 0.30], DARK)
        for x in (-1.7, 1.7):
            _box(C, f"jig_post_{x}_{y}", [0.08, 0.08, 0.25], [x, y, 0.125], DARK)
        for x in (-1.2, 0.0, 1.2):
            _box(C, f"jig_pin_{x}_{y}", [0.10, 0.10, 0.13], [x, y, 0.415], [0.85, 0.62, 0.12])
    _box(C, "underbody", [3.9, 1.44, 0.04], [0, 0, 0.50], STEEL)
    _box(C, "tunnel", [2.1, 0.24, 0.10], [-0.05, 0, 0.57], STEEL)
    for x in (-0.85, -0.6, -0.35, -0.1, 0.4, 0.65, 0.9):
        for s in (-1, 1):
            spot("floor", (x, s * 0.45, 0.52), up)

    for s in (-1, 1):
        side = (0.0, s, 0.0)
        frame_y = (s * 0.68, s * 0.76)  # side frame members
        # rocker
        _box(C, f"sill_{s}", [2.17, 0.12, 0.16], [-0.035, s * 0.72, 0.60], STEEL)
        for x in np.arange(-0.90, 0.91, 0.15):
            for z in (0.57, 0.63):
                spot("sill", (x, s * 0.78, z), side)
            if not 0.04 < x < 0.26:
                spot("sill_top", (x, s * 0.72, 0.68), up)
        # A pillar: vertical hinge pillar, slanted windshield pillar up to the roof rail
        _prism_xz(C, f"a_pillar_low_{s}", [(-1.12, 0.52), (-1.00, 0.52), (-1.00, 0.98), (-1.12, 0.98)], *frame_y)
        _prism_xz(C, f"a_pillar_{s}", [(-1.12, 0.96), (-1.00, 0.96), (-0.55, 1.375), (-0.66, 1.375)], *frame_y)
        for z in (0.75, 0.88):
            spot("pillar", (-1.06, s * 0.76, z), side)
        a0, a1 = np.array([-1.06, 0.97]), np.array([-0.605, 1.36])
        for t in (0.35, 0.7):
            x, z = a0 + t * (a1 - a0)
            spot("pillar", (x, s * 0.76, z), side)
        # B pillar, tapered
        _prism_xz(C, f"b_pillar_{s}", [(0.07, 0.66), (0.23, 0.66), (0.19, 1.385), (0.11, 1.385)], *frame_y)
        for z in (0.85, 1.05, 1.25):
            spot("pillar", (0.15, s * 0.76, z), side)
        # C pillar, slanted down to the quarter
        _prism_xz(C, f"c_pillar_{s}", [(0.93, 1.375), (1.03, 1.375), (1.48, 0.97), (1.30, 0.97)], *frame_y)
        c0, c1 = np.array([0.98, 1.36]), np.array([1.39, 0.98])
        for t in (0.3, 0.65):
            x, z = c0 + t * (c1 - c0)
            spot("pillar", (x, s * 0.76, z), side)
        # roof rail along the crowned roof line, 8 segments, 7 cm deep
        xs = np.linspace(-0.66, 1.03, 9)
        for k, (xa, xb) in enumerate(zip(xs[:-1], xs[1:])):
            za, zb = _roof_z(xa), _roof_z(xb)
            _prism_xz(C, f"roof_rail_{s}_{k}", [(xa, za - 0.07), (xb, zb - 0.07), (xb, zb), (xa, za)], *frame_y)
        for x in np.arange(-0.45, 0.91, 0.15):
            spot("roof_rail", (x, s * 0.76, _roof_z(x) - 0.035), side)
        for x in np.arange(-0.45, 0.91, 0.2):
            spot("roof_top", (x, s * 0.72, _roof_z(x)), (0.1875 * (x - 0.2), 0.0, 1.0))
        # rear quarter with the wheel arch (centre x 1.45, z 0.55, r 0.30) cut out
        qy = (s * 0.70, s * 0.75)
        _prism_xz(C, f"quarter_front_{s}", [(1.00, 0.55), (1.15, 0.55), (1.15, 1.00), (1.00, 1.00)], *qy)
        _prism_xz(C, f"quarter_rear_{s}", [(1.75, 0.55), (1.95, 0.55), (1.95, 1.00), (1.75, 1.00)], *qy)
        th = np.linspace(np.pi, 0.0, 9)
        for k in range(8):
            xa, za = 1.45 + 0.30 * np.cos(th[k]), 0.55 + 0.30 * np.sin(th[k])
            xb, zb = 1.45 + 0.30 * np.cos(th[k + 1]), 0.55 + 0.30 * np.sin(th[k + 1])
            _prism_xz(C, f"quarter_arch_{s}_{k}", [(xa, za), (xb, zb), (xb, 1.00), (xa, 1.00)], *qy)
        for x, z in ((1.07, 0.70), (1.07, 0.90), (1.30, 0.93), (1.45, 0.93), (1.60, 0.93), (1.85, 0.70), (1.85, 0.90)):
            spot("quarter", (x, s * 0.75, z), side)
        # front: lower rails, strut towers, upper rails from the hinge pillar to the radiator support
        _box(C, f"front_rail_{s}", [0.74, 0.10, 0.12], [-1.55, s * 0.42, 0.58], STEEL)
        _box(C, f"strut_tower_{s}", [0.32, 0.20, 0.34], [-1.46, s * 0.52, 0.71], STEEL)
        _prism_xz(C, f"upper_rail_{s}", [(-1.12, 0.92), (-1.12, 1.00), (-1.94, 0.86), (-1.94, 0.78)],
                  s * 0.60, s * 0.68)
        for x in (-1.85, -1.75, -1.68, -1.25):
            spot("front_rail", (x, s * 0.42, 0.64), up)
            spot("front_rail_side", (x, s * 0.47, 0.58), side)
        for x in (-1.55, -1.38):
            spot("strut", (x, s * 0.52, 0.88), up)
        u0, u1 = np.array([-1.12, 1.00]), np.array([-1.94, 0.86])
        for t in (0.25, 0.55, 0.8):
            x, z = u0 + t * (u1 - u0)
            spot("upper_rail", (x, s * 0.64, z), (-0.1707, 0.0, 1.0))
        # rear rails
        _box(C, f"rear_rail_{s}", [0.70, 0.10, 0.12], [1.60, s * 0.42, 0.58], STEEL)
        for x in (1.5, 1.65, 1.8, 1.9):
            spot("rear_rail", (x, s * 0.42, 0.64), up)
            spot("rear_rail_side", (x, s * 0.37, 0.58), (0.0, -s, 0.0))
        # radiator support post
        _box(C, f"rad_post_{s}", [0.06, 0.06, 0.34], [-1.94, s * 0.47, 0.69], STEEL)
        for z in (0.62, 0.74):
            spot("front", (-1.97, s * 0.47, z), (-1.0, 0.0, 0.0))
        for z in (0.72, 0.86):
            spot("dash", (-1.10, s * 0.40, z), (1.0, 0.0, 0.0))
        spot("cowl", (-1.06, s * 0.30, 0.99), up)
        for x in (1.12, 1.30):
            spot("shelf", (x, s * 0.35, 0.99), up)
    # cross members: dash and cowl, parcel shelf, radiator support frame, rear panel
    _box(C, "dash", [0.04, 1.36, 0.45], [-1.12, 0, 0.745], STEEL)
    _box(C, "cowl", [0.16, 1.36, 0.04], [-1.06, 0, 0.97], STEEL)
    _box(C, "parcel_shelf", [0.38, 1.36, 0.03], [1.21, 0, 0.975], STEEL)
    _box(C, "rad_top", [0.06, 1.32, 0.06], [-1.94, 0, 0.83], STEEL)
    _box(C, "rad_bottom", [0.06, 1.00, 0.06], [-1.94, 0, 0.55], STEEL)
    for y in (-0.3, 0.0, 0.3):
        spot("front", (-1.97, y, 0.83), (-1.0, 0.0, 0.0))
    for y in (-0.25, 0.25):
        spot("front", (-1.97, y, 0.55), (-1.0, 0.0, 0.0))
    _box(C, "rear_panel", [0.04, 1.36, 0.38], [1.95, 0, 0.71], STEEL)
    for y in (-0.4, -0.2, 0.0, 0.2, 0.4):
        for z in (0.62, 0.75, 0.85):
            spot("rear", (1.97, y, z), (1.0, 0.0, 0.0))
    # roof bows (front header, B, rear header), cross-car crown 2.5 cm, 6 segments
    for name, (x0, x1) in (("front", (-0.66, -0.56)), ("mid", (0.11, 0.19)), ("rear", (0.93, 1.03))):
        zc = _roof_z((x0 + x1) / 2)
        bow = lambda y: zc + 0.025 * (1.0 - (y / 0.72) ** 2)
        ys = np.linspace(-0.72, 0.72, 7)
        for k in range(6):
            ya, yb = ys[k], ys[k + 1]
            _prism_yz(C, f"roof_bow_{name}_{k}", [(ya, bow(ya) - 0.05), (yb, bow(yb) - 0.05), (yb, bow(yb)),
                                                  (ya, bow(ya))], x0, x1)
        for s in (-1, 1):
            spot("roof_bow", ((x0 + x1) / 2, s * 0.40, bow(s * 0.40)), (0.0, 0.0965 * s * 0.40, 1.0))
    return spots


def _line_layout(num_robots, rails):
    """Stations (x, y, yaw, pedestal height, rail travel or 0). Side robots alternate low/high pedestals.
    rails: none / some (every other side robot on its own short rail) / all / shared (one rail per side carrying all
    of that side's robots; a carriage may travel to its neighbours' stations, so robots share stretches of the body
    and cannot pass each other)."""
    xs = {8: (-1.35, -0.45, 0.45, 1.35), 12: (-1.6, -0.8, 0.0, 0.8, 1.6),
          16: (-1.75, -1.05, -0.35, 0.35, 1.05, 1.75)}[num_robots]
    travel = {8: 0.30, 12: 0.22, 16: 0.18}[num_robots]
    if rails == "shared":
        travel = xs[1] - xs[0]
    stations = []
    for s in (1, -1):
        for k, x in enumerate(xs):
            on_rail = rails in ("all", "shared") or (rails == "some" and k % 2 == 1)
            stations.append((x, s * 1.38, -s * np.pi / 2, (0.55, 0.85)[k % 2], travel if on_rail else 0.0))
    if num_robots == 12:
        stations += [(-2.55, 0.0, 0.0, 0.55, 0.0), (2.55, 0.0, np.pi, 0.55, 0.0)]
    elif num_robots == 16:
        stations += [(-2.5, 0.45, 0.0, 0.55, 0.0), (-2.5, -0.45, 0.0, 0.55, 0.0),
                     (2.5, 0.45, np.pi, 0.55, 0.0), (2.5, -0.45, np.pi, 0.55, 0.0)]
    return stations


def make_welding_line_env(num_robots=8, rails="some", spots_per_robot=4, seed=0):
    rng = np.random.default_rng(1000 + seed)
    C = ry.Config()
    C.addFrame("world")
    _box(C, "floor", [7.0, 4.4, 0.04], [0, 0, -0.02], [0.80, 0.80, 0.78])
    for y in (-1.78, 1.78):  # walkway markings, visual only
        _box(C, f"marking_{y}", [6.2, 0.06, 0.002], [0, y, 0.001], [0.95, 0.78, 0.10], contact=0)
    for x in (-3.1, 3.1):
        _box(C, f"marking_{x}", [0.06, 3.62, 0.002], [x, 0, 0.001], [0.95, 0.78, 0.10], contact=0)
    candidates = _body_in_white(C)
    stations = _line_layout(num_robots, rails)
    robots = [f"a{i + 1}_" for i in range(len(stations))]
    model = str(ASSETS / "ur10/ur_welding.g")
    tool_q = [0.70710678, 0.0, 0.70710678, 0.0]  # capsule z -> ee x (the flange axis)
    side_xs = {s: sorted(st[0] for st in stations if st[4] > 0 and np.sign(st[1]) == s) for s in (1, -1)}
    if rails == "shared":
        for s, xs_s in side_xs.items():
            sp = xs_s[1] - xs_s[0]
            _box(C, f"rail_side{s}", [xs_s[-1] - xs_s[0] + 2 * sp + 0.45, 0.30, 0.04], [0, s * 1.38, 0.02], DARK,
                 contact=0)
    for r, (x, y, yaw, h, travel) in zip(robots, stations):
        if rails == "shared" and travel > 0:
            # Shared side rail: the carriage may travel from the left to the right neighbour's station. Pedestals
            # collide as vertical capsules (rai 0.2.9 reports zero penetration for box-box and cylinder-cylinder
            # pairs, so box pedestals on one rail would pass through each other unnoticed); the cylinder and the
            # plate are visual only.
            xs_s = side_xs[int(np.sign(y))]
            k, sp = xs_s.index(x), xs_s[1] - xs_s[0]
            lo = xs_s[k - 1] if k > 0 else x - sp
            hi = xs_s[k + 1] if k < len(xs_s) - 1 else x + sp
            mount = C.addFrame(r + "carriage", f"rail_side{int(np.sign(y))}").setJoint(ry.JT.transX, limits=[lo, hi])
            mount.setJointState([x])
            C.addFrame(r + "plate", mount.name).setShape(ry.ST.box, [0.36, 0.36, 0.05]).setRelativePosition(
                [0, 0, 0.045]).setColor(DARK).setContact(0)
            ped = C.addFrame(r + "pedestal", mount.name).setShape(ry.ST.cylinder, [h - 0.09, 0.15])
            ped.setRelativePosition([0, 0, 0.07 + (h - 0.09) / 2]).setColor([0.95, 0.75, 0.15]).setContact(0)
            C.addFrame(r + "pedestal_coll", mount.name).setShape(ry.ST.capsule, [h - 0.45, 0.15]).setRelativePosition(
                [0, 0, (h - 0.01) / 2]).setColor([0.95, 0.75, 0.15]).setContact(1)
            top = (h - 0.09) / 2
        elif travel > 0:
            _box(C, f"rail_{r}", [2 * travel + 0.45, 0.30, 0.04], [x, y, 0.02], DARK, contact=0)
            # joint frame at the rail centre (z 0.02); plate 0.04-0.09, pedestal 0.09-h, robot base at h
            mount = C.addFrame(r + "carriage", f"rail_{r}").setJoint(ry.JT.transX, limits=[-travel, travel])
            C.addFrame(r + "plate", mount.name).setShape(ry.ST.box, [0.36, 0.36, 0.05]).setRelativePosition(
                [0, 0, 0.045]).setColor(DARK).setContact(1)
            ped = C.addFrame(r + "pedestal", mount.name).setShape(ry.ST.box, [0.30, 0.30, h - 0.09])
            ped.setRelativePosition([0, 0, 0.07 + (h - 0.09) / 2]).setColor([0.95, 0.75, 0.15]).setContact(1)
            top = (h - 0.09) / 2
        else:
            ped = _box(C, r + "pedestal", [0.30, 0.30, h], [x, y, h / 2], [0.95, 0.75, 0.15])
            top = h / 2
        C.addFile(model, namePrefix=r).setParent(ped).setRelativePosition([0, 0, top]).setRelativeQuaternion(
            _qz(yaw)).setJoint(ry.JT.rigid)
        C.delFrame(r + "gripper_fill")
        ee = r + "ur_ee_link"
        C.addFrame(r + "gun_body", ee).setShape(ry.ST.capsule, [0.16, 0.045]).setRelativePosition(
            [0.11, 0, 0]).setRelativeQuaternion(tool_q).setColor([0.85, 0.45, 0.12]).setContact(-1)
        C.addFrame(r + "gun_tip", ee).setShape(ry.ST.capsule, [0.08, 0.012]).setRelativePosition(
            [0.26, 0, 0]).setRelativeQuaternion(tool_q).setColor([0.25, 0.25, 0.25]).setContact(-1)
        C.addFrame(r + "gun_point", ee).setShape(ry.ST.marker, [0.03]).setRelativePosition([0.32, 0, 0])
    home = C.getJointState().copy()
    moving = _moving_frames(C, rng)
    for r in robots:
        ok, conf = _check_pose(C, r, robots, moving)
        if not ok or conf:
            raise RuntimeError(f"welding line home pose of {r} in collision")

    taken = set()
    keyframes, conflicts, labels = {}, {}, {}

    def reachable(k, x, y, h, travel):  # flange (gun point + 0.33 along the normal) within UR10 reach of the shoulder
        l, p, n = candidates[k]
        f = p + 0.33 * n - np.array([x, y, h + 0.127])
        return np.hypot(max(abs(f[0]) - travel, 0.0), f[1]) ** 2 + f[2] ** 2 < 1.27 ** 2

    pools = {}
    for r, (x, y, yaw, h, travel) in zip(robots, stations):
        if abs(y) > 1.0:  # side robot: own side, along the line
            mine = [k for k, (l, p, n) in enumerate(candidates) if p[1] * y > 0.2 and abs(p[0] - x) < 1.0 + travel]
        else:  # front / rear robot: its end of the body
            mine = [k for k, (l, p, n) in enumerate(candidates) if p[0] * x > 0 and abs(p[0]) > 1.2
                    and (y == 0 or p[1] * y >= -0.05)]
        pools[r] = [k for k in mine if reachable(k, x, y, h, travel)]
    rail_of = {}  # shared rail: the robots on each robot's rail, in rail order
    if rails == "shared":
        for s in (1, -1):
            on = sorted((st[0], r) for r, st in zip(robots, stations) if st[4] > 0 and np.sign(st[1]) == s)
            for _, r in on:
                rail_of[r] = [rr for _, rr in on]

    def accept(r):
        # (conflicts, pushed configuration) or None. Feasible by construction: off a shared rail the keyframe must be
        # free with everyone at home (robots one at a time is then a plan); on a shared rail the neighbours must be
        # able to make room (pushed along the rail, arms at home).
        ok, conf = _check_pose(C, r, robots, moving)
        if not ok:
            return None
        if r in rail_of:
            q_push = _push_neighbours(C, r, rail_of[r], robots, moving)
            return None if q_push is None else (conf, q_push)
        return None if conf else (conf, None)

    for r in sorted(robots, key=lambda r: (len(pools[r]), robots.index(r))):  # fewest candidates pick first
        mine = pools[r]
        groups = {}
        for k in mine:
            groups.setdefault(candidates[k][0], []).append(k)
        order = [list(rng.permutation(g)) for g in (groups[l] for l in rng.permutation(sorted(groups)))]
        keyframes[r], conflicts[r], labels[r] = [], [], []
        while len(keyframes[r]) < spots_per_robot and any(order):
            for g in order:
                if len(keyframes[r]) >= spots_per_robot:
                    break
                while g:
                    k = int(g.pop(0))
                    if k in taken:
                        continue
                    label, p, n = candidates[k]
                    sol = _ik(C, r, r + "gun_point", p + 0.01 * n, ry.FS.vectorX, -n, rng, lambda: accept(r))
                    if sol is None:
                        continue
                    q, (conf, q_push) = sol
                    if q_push is not None:  # reach it with the neighbours already pushed aside
                        q_start = q_push.copy()
                        q_start[_robot_indices(C, r)] = home[_robot_indices(C, r)]
                        C.setJointState(q_start)
                    reached = _reachable_from_home(C, r, robots, moving, q, rng)
                    C.setJointState(home)
                    if not reached:
                        continue
                    taken.add(k)
                    C.setJointState(q, get_robot_joints(C, r))
                    keyframes[r].append(C.getJointState()[_robot_indices(C, r)].copy())
                    C.setJointState(home)
                    conflicts[r].append(sorted(conf))
                    labels[r].append((label, p))
                    break
        if len(keyframes[r]) < spots_per_robot:
            raise RuntimeError(f"welding line: only {len(keyframes[r])} reachable spots for {r}")
    for i, r in enumerate(robots):
        for j, (label, p) in enumerate(labels[r]):
            C.addFrame(f"spot_{r}{j}").setShape(ry.ST.sphere, [0.018]).setPosition(list(p)).setColor(
                PALETTE[i % len(PALETTE)]).setContact(0)
    C.setJointState(home)
    scene = dict(stations=stations, spot_labels={r: [l for l, _ in labels[r]] for r in robots},
                 home_conflicts=conflicts, rails=rails)
    return C, robots, keyframes, scene


@register([
    ("rai.welding_line_8", {"num_robots": 8, "rails": "none"}),
    ("rai.welding_line_8_rails", {"num_robots": 8, "rails": "some"}),
    ("rai.welding_line_8_shared_rail", {"num_robots": 8, "rails": "shared"}),
    ("rai.welding_line_12_shared_rail", {"num_robots": 12, "rails": "shared"}),
    ("rai.welding_line_8_small", {"num_robots": 8, "rails": "some", "spots_per_robot": 2}),
    ("rai.welding_line_12", {"num_robots": 12, "rails": "some"}),
    ("rai.welding_line_16", {"num_robots": 16, "rails": "some"}),
])
class rai_welding_line(SequenceMixin, rai_env):
    """Car-body welding line, round robin: every robot's j-th spot before anyone's (j+1)-th, then all home."""

    def __init__(self, num_robots=8, rails="some", spots_per_robot=4, seed=0):
        self.C, self.robots, keyframes, self.scene = make_welding_line_env(num_robots, rails, spots_per_robot, seed)
        rai_env.__init__(self)
        self.tasks = [Task(f"{r}spot{j}", [r], SingleGoal(q)) for r in self.robots for j, q in enumerate(keyframes[r])]
        self.tasks.append(Task("terminal", self.robots, SingleGoal(self.C.getJointState())))
        self.sequence = self._make_sequence_from_names(
            [f"{r}spot{j}" for j in range(spots_per_robot) for r in self.robots] + ["terminal"])
        BaseModeLogic.__init__(self)
        self.collision_tolerance = 0.01
        _mark_static(self)
        _safe_home_if_unblocking(self)


@register([
    ("rai.welding_line_8_dep", {"num_robots": 8, "rails": "some"}),
    ("rai.welding_line_8_shared_rail_dep", {"num_robots": 8, "rails": "shared"}),
    ("rai.welding_line_12_dep", {"num_robots": 12, "rails": "some"}),
])
class rai_welding_line_dep(DependencyGraphMixin, rai_env):
    """Welding line with an independent chain of spots per robot (any interleaving), then all home."""

    def __init__(self, num_robots=8, rails="some", spots_per_robot=4, seed=0):
        self.C, self.robots, keyframes, self.scene = make_welding_line_env(num_robots, rails, spots_per_robot, seed)
        rai_env.__init__(self)
        self.tasks = [Task(f"{r}spot{j}", [r], SingleGoal(q)) for r in self.robots for j, q in enumerate(keyframes[r])]
        self.tasks.append(Task("terminal", self.robots, SingleGoal(self.C.getJointState())))
        self.graph = DependencyGraph()
        for r in self.robots:
            for j in range(1, spots_per_robot):
                self.graph.add_dependency(f"{r}spot{j}", f"{r}spot{j - 1}")
            self.graph.add_dependency("terminal", f"{r}spot{spots_per_robot - 1}")
        BaseModeLogic.__init__(self)
        self.collision_tolerance = 0.01
        _mark_static(self)
        self.spec.dependency = DependencyType.UNORDERED
        _safe_home_if_unblocking(self)


# ------------------------------------------------------------------------------------------------ shared bin
def make_shared_bin_env(num_arms=3, bin_width=0.26, targets_per_arm=3, wall_height=0.14, radius=0.56, seed=0):
    rng = np.random.default_rng(2000 + seed)
    C = ry.Config()
    C.addFrame("world")
    _box(C, "floor", [3.0, 3.0, 0.04], [0, 0, -0.02], [0.86, 0.86, 0.84], contact=0)
    _box(C, "table", [1.9, 1.9, 0.06], [0, 0, 0.37], [0.60, 0.48, 0.33])
    for x in (-0.85, 0.85):
        for y in (-0.85, 0.85):
            _box(C, f"table_leg_{x}_{y}", [0.06, 0.06, 0.34], [x, y, 0.17], DARK, contact=0)
    w, t, z0 = bin_width, 0.02, 0.40
    _box(C, "bin_floor", [w + 2 * t, w + 2 * t, 0.01], [0, 0, z0 + 0.005], [0.25, 0.35, 0.55])
    for s in (-1, 1):
        _box(C, f"bin_wall_x{s}", [t, w + 2 * t, wall_height], [s * (w / 2 + t / 2), 0, z0 + 0.01 + wall_height / 2],
             [0.25, 0.35, 0.55])
        _box(C, f"bin_wall_y{s}", [w, t, wall_height], [0, s * (w / 2 + t / 2), z0 + 0.01 + wall_height / 2],
             [0.25, 0.35, 0.55])
    robots = [f"a{i + 1}_" for i in range(num_arms)]
    model = str(ASSETS / "ur5_vamp/ur5_vamp.g")
    offset = np.pi / 4 if num_arms == 4 else np.pi / 2
    angles = [offset + 2 * np.pi * i / num_arms for i in range(num_arms)]
    for r, a in zip(robots, angles):
        base = _box(C, r + "riser", [0.16, 0.16, 0.02], [radius * np.cos(a), radius * np.sin(a), z0 + 0.01], DARK)
        C.addFile(model, namePrefix=r).setParent(base).setRelativePosition([0, 0, 0.01]).setRelativeQuaternion(
            _qz(a + np.pi)).setJoint(ry.JT.rigid)  # base x towards the bin: the home pose looks at it
    home = C.getJointState().copy()
    moving = _moving_frames(C, rng)
    for r in robots:
        ok, conf = _check_pose(C, r, robots, moving)
        if not ok or conf:
            raise RuntimeError(f"shared bin home pose of {r} in collision")
    d = 0.22 * w
    in_bin = [(0, 0), (d, d), (-d, -d), (d, -d), (-d, d)]
    down = np.array([0, 0, -1.0])
    keyframes, conflicts, names = {}, {}, {}
    accept = lambda r: (lambda ok_c: ok_c[1] if ok_c[0] and not ok_c[1] else None)(_check_pose(C, r, robots, moving))
    for i, (r, a) in enumerate(zip(robots, angles)):
        b = np.array([radius * np.cos(a), radius * np.sin(a)])
        keyframes[r], conflicts[r], names[r] = [], [], []
        for j in range(targets_per_arm):
            for kind in ("bin", "drop"):
                if kind == "bin":
                    tries = [np.array([*in_bin[(i + j + k) % len(in_bin)], z0 + 0.055]) for k in range(len(in_bin))]
                else:
                    da = a + np.pi + (1.0 if j % 2 == 0 else -1.0) * 1.9
                    tries = [np.array([*(b + rad * np.array([np.cos(da), np.sin(da)])), z0 + 0.05])
                             for rad in (0.34, 0.30, 0.38)]
                sol = None
                for p in tries:
                    sol = _ik(C, r, r + "ur_gripper_center", p, ry.FS.vectorZ, down, rng, lambda: accept(r))
                    if sol is not None and _reachable_from_home(C, r, robots, moving, sol[0], rng):
                        break
                    sol = None
                if sol is None:
                    raise RuntimeError(f"shared bin: no {kind} pose for {r} target {j}")
                q, conf = sol
                C.setJointState(q, get_robot_joints(C, r))
                keyframes[r].append(C.getJointState()[_robot_indices(C, r)].copy())
                C.setJointState(home)
                conflicts[r].append(sorted(conf))
                names[r].append(f"{r}{kind}{j}")
                if kind == "drop":
                    C.addFrame(f"pad_{r}{j}").setShape(ry.ST.box, [0.07, 0.07, 0.004]).setPosition(
                        [p[0], p[1], z0 + 0.002]).setColor(PALETTE[i % len(PALETTE)] + [0.7]).setContact(0)
    C.setJointState(home)
    return C, robots, keyframes, names, dict(bin_width=w, wall_height=wall_height, home_conflicts=conflicts)


@register([
    ("rai.shared_bin_2", {"num_arms": 2}),
    ("rai.shared_bin_3", {"num_arms": 3}),
    ("rai.shared_bin_4", {"num_arms": 4}),
    ("rai.shared_bin_4_wide", {"num_arms": 4, "bin_width": 0.50}),
    ("rai.shared_bin_3_deep", {"num_arms": 3, "wall_height": 0.22}),
])
class rai_shared_bin(SequenceMixin, rai_env):
    """Shared narrow bin, round robin: a1 reach in, a1 drop, a2 reach in, a2 drop, ..., then all home."""

    def __init__(self, num_arms=3, bin_width=0.26, targets_per_arm=3, wall_height=0.14, seed=0):
        self.C, self.robots, keyframes, names, self.scene = make_shared_bin_env(
            num_arms, bin_width, targets_per_arm, wall_height, seed=seed)
        rai_env.__init__(self)
        self.tasks = [Task(n, [r], SingleGoal(q)) for r in self.robots for n, q in zip(names[r], keyframes[r])]
        self.tasks.append(Task("terminal", self.robots, SingleGoal(self.C.getJointState())))
        order = [names[r][2 * j + k] for j in range(targets_per_arm) for r in self.robots for k in (0, 1)]
        self.sequence = self._make_sequence_from_names(order + ["terminal"])
        BaseModeLogic.__init__(self)
        self.collision_tolerance = 0.001
        _mark_static(self)
        _safe_home_if_unblocking(self)


@register([
    ("rai.shared_bin_3_dep", {"num_arms": 3}),
    ("rai.shared_bin_4_dep", {"num_arms": 4}),
])
class rai_shared_bin_dep(DependencyGraphMixin, rai_env):
    """Shared narrow bin, a chain per arm (reach in, drop, reach in, ...): the planner decides who goes first."""

    def __init__(self, num_arms=3, bin_width=0.26, targets_per_arm=3, wall_height=0.14, seed=0):
        self.C, self.robots, keyframes, names, self.scene = make_shared_bin_env(
            num_arms, bin_width, targets_per_arm, wall_height, seed=seed)
        rai_env.__init__(self)
        self.tasks = [Task(n, [r], SingleGoal(q)) for r in self.robots for n, q in zip(names[r], keyframes[r])]
        self.tasks.append(Task("terminal", self.robots, SingleGoal(self.C.getJointState())))
        self.graph = DependencyGraph()
        for r in self.robots:
            for a, b in zip(names[r][1:], names[r]):
                self.graph.add_dependency(a, b)
            self.graph.add_dependency("terminal", names[r][-1])
        BaseModeLogic.__init__(self)
        self.collision_tolerance = 0.001
        _mark_static(self)
        self.spec.dependency = DependencyType.UNORDERED
        _safe_home_if_unblocking(self)


# -------------------------------------------------------------------------------------- container with a lid
def make_container_env(task="unpack", layers=1, width=None, seed=0, num_arms=2):
    """UR5s (2, 3 or 4) at a table with a lidded container in the middle. unpack: a1 lifts the lid off by its
    handle and sets it on the table, then the arms take the objects out (top layer first) to pads on their own side.
    pack: the objects go from the pads into the container (bottom layer first), then a1 puts the lid on. Two arms
    face each other across the container; 3 or 4 stand around it (the lid goes to the free sector between a1 and
    a2, each object to the nearest arm, pads on both sides of each base). The sequence is
    simulated on a working copy with the benchmark's pick / place semantics (attach to the gripper, then to the
    table, contact -1) and every leg of the one-arm-at-a-time plan (home -> pick, carry -> place, place -> home) is
    checked by a single-robot RRT-Connect, so the scene is feasible by construction.
    Returns C (start state), robots, steps [(robot, object, q_pick, q_place)], scene."""
    rng = np.random.default_rng(3000 + seed)
    C = ry.Config()
    C.addFrame("world")
    _box(C, "floor", [3.0, 3.0, 0.04], [0, 0, -0.02], [0.80, 0.80, 0.78], contact=0)
    tx, ty = (1.7, 1.2) if num_arms == 2 else (1.9, 1.9)
    _box(C, "table", [tx, ty, 0.06], [0, 0, 0.37], [0.60, 0.48, 0.33])
    for x in (-tx / 2 + 0.07, tx / 2 - 0.07):
        for y in (-ty / 2 + 0.07, ty / 2 - 0.07):
            _box(C, f"table_leg_{x:.2f}_{y:.2f}", [0.06, 0.06, 0.34], [x, y, 0.17], DARK, contact=0)
    # the deep container is wider: reaching the bottom layer takes the gripper's palm below the rim
    z0, t, w = 0.40, 0.015, (width or (0.25 if layers == 1 else 0.33))
    hc = 0.075 if layers == 1 else 0.125
    blue = [0.20, 0.36, 0.58]
    _box(C, "container_floor", [w + 2 * t, w + 2 * t, 0.01], [0, 0, z0 + 0.005], blue)
    for sg in (-1, 1):
        _box(C, f"container_wall_x{sg}", [t, w + 2 * t, hc], [sg * (w / 2 + t / 2), 0, z0 + 0.01 + hc / 2], blue)
        _box(C, f"container_wall_y{sg}", [w, t, hc], [0, sg * (w / 2 + t / 2), z0 + 0.01 + hc / 2], blue)
    top = z0 + 0.01 + hc
    robots = [f"a{i + 1}_" for i in range(num_arms)]
    if num_arms == 2:
        bases = {"a1_": np.array([-0.56, 0.0]), "a2_": np.array([0.56, 0.0])}
        angles = {"a1_": np.pi, "a2_": 0.0}
    else:  # around the container; the lid's sector lies between a1 and a2
        offset = -np.pi / 2 if num_arms == 3 else np.pi / 4
        angles = {r: offset + 2 * np.pi * i / num_arms for i, r in enumerate(robots)}
        bases = {r: 0.56 * np.array([np.cos(a), np.sin(a)]) for r, a in angles.items()}
    model = str(ASSETS / "ur5_vamp/ur5_vamp.g")
    for r, b in bases.items():
        riser = _box(C, r + "riser", [0.16, 0.16, 0.02], [b[0], b[1], z0 + 0.01], DARK)
        yaw = (0.0 if b[0] < 0 else np.pi) if num_arms == 2 else angles[r] + np.pi  # base x towards the container
        C.addFile(model, namePrefix=r).setParent(riser).setRelativePosition([0, 0, 0.01]).setRelativeQuaternion(
            _qz(yaw)).setJoint(ry.JT.rigid)
    # lid with a handle; objects of slightly different sizes (identical aligned boxes are a degenerate rai pair).
    # Movable frames must have "obj" in their name (rai_env.get_scenegraph_info_for_mode), markers must not.
    lid_size = [w + 2 * t + 0.02, w + 2 * t + 0.02, 0.012]
    lid_closed = np.array([0.0, 0.0, top + 0.001 + 0.006])
    if num_arms == 2:
        lid_rest = np.array([0.0, 0.40, z0 + 0.001 + 0.006])
    else:
        a_lid = (angles["a1_"] + angles["a2_"]) / 2
        lid_rest = np.array([0.50 * np.cos(a_lid), 0.50 * np.sin(a_lid), z0 + 0.001 + 0.006])
    slots = [(sx, sy, layer) for layer in range(layers) for sx in (-0.06, 0.06) for sy in (-0.06, 0.06)]
    pads = {}
    for r, b in bases.items():
        if num_arms == 2:
            ang0 = 0.0 if b[0] < 0 else np.pi
            sign = -1.0 if b[0] < 0 else 1.0  # rotate towards -y
            pads[r] = [b + 0.32 * np.array([np.cos(ang0 + sign * a), np.sin(ang0 + sign * a)])
                       for a in np.radians([45, 70, 95, 120])]
        else:  # both sides of the base, sideways and a little behind
            to_c = angles[r] + np.pi
            pads[r] = [b + 0.28 * np.array([np.cos(to_c + a), np.sin(to_c + a)])
                       for a in np.radians([110, -110, 140, -140])]
    objects = []  # (name, size, slot centre, pad centre, robot)
    used = {r: 0 for r in robots}
    for k, (sx, sy, layer) in enumerate(slots):
        if num_arms == 2:
            r = "a1_" if sx < 0 else "a2_"
        else:  # nearest arm by direction
            r = max(robots, key=lambda rr: np.cos(np.arctan2(sy, sx) - angles[rr]))
        size = [0.044 + 0.002 * k, 0.046 + 0.0015 * k, 0.05]
        zc = z0 + 0.01 + 0.001 + 0.025 + layer * 0.051
        pad = pads[r][used[r]]
        used[r] += 1
        objects.append((f"obj{k}", size, np.array([sx, sy, zc]), np.array([pad[0], pad[1], z0 + 0.001 + 0.025]), r))
    unpack = task == "unpack"
    lid_start = lid_closed if unpack else lid_rest
    lid = C.addFrame("obj_lid", "table").setShape(ry.ST.box, lid_size).setPosition(list(lid_start)).setColor(
        [0.86, 0.86, 0.88]).setContact(1).setJoint(ry.JT.rigid)
    C.addFrame("obj_lid_handle", "obj_lid").setShape(ry.ST.box, [0.03, 0.10, 0.045]).setRelativePosition(
        [0, 0, 0.006 + 0.0225]).setColor([0.22, 0.22, 0.24]).setContact(-1)
    C.addFrame("lid_rest_pad").setShape(ry.ST.box, [lid_size[0], lid_size[1], 0.002]).setPosition(
        [lid_rest[0], lid_rest[1], z0 + 0.001]).setColor([0.5, 0.5, 0.5, 0.4]).setContact(0)
    for i, (name, size, slot, pad, r) in enumerate(objects):
        colour = PALETTE[i % len(PALETTE)]
        C.addFrame(name, "table").setShape(ry.ST.box, size).setPosition(list(slot if unpack else pad)).setColor(
            colour).setContact(1).setJoint(ry.JT.rigid)
        C.addFrame(f"pad_{i}").setShape(ry.ST.box, [0.07, 0.07, 0.002]).setPosition(
            [pad[0], pad[1], z0 + 0.001]).setColor(colour + [0.5]).setContact(0)
    home = C.getJointState().copy()
    moving = _moving_frames(C, rng)
    for r in robots:
        ok, conf = _check_pose(C, r, robots, moving)
        if not ok or conf:
            raise RuntimeError(f"container home pose of {r} in collision")
    # the steps: (robot, frame to attach, frames carried along, start centre, target centre, grasp offset)
    handle_off = np.array([0.0, 0.0, 0.006 + 0.0225 + 0.010])  # fingers ~1 cm above the lid
    obj_off = np.array([0.0, 0.0, 0.01])
    lid_step = ("a1_", "obj_lid", ("obj_lid", "obj_lid_handle"), lid_start, lid_rest if unpack else lid_closed, handle_off)
    if num_arms == 2:
        order = sorted(objects, key=lambda o: (-o[2][2] if unpack else o[2][2], o[4], o[0]))
    else:  # round robin over the arms within a layer
        rank = {o[0]: sum(1 for p in objects if p[4] == o[4] and p[2][2] == o[2][2] and p[0] < o[0]) for o in objects}
        order = sorted(objects, key=lambda o: (-o[2][2] if unpack else o[2][2], rank[o[0]], o[4]))
    obj_steps = [(r, n, (n,), slot if unpack else pad, pad if unpack else slot, obj_off)
                 for n, size, slot, pad, r in order]
    plan = [lid_step] + obj_steps if unpack else obj_steps + [lid_step]
    W = ry.Config()
    W.addConfigurationCopy(C)
    down = np.array([0.0, 0.0, -1.0])
    steps = []
    for r, frame, carried_frames, start, target, off in plan:
        names, idx = get_robot_joints(W, r), _robot_indices(W, r)
        home_r = home[idx].copy()
        snapshot = ry.Config()
        snapshot.addConfigurationCopy(W)
        done = None
        for psi in (0.0, np.pi / 2):
            xdir = [np.cos(psi), np.sin(psi), 0.0]
            strict = lambda carried=None: (lambda ok_c: ok_c[1] if ok_c[0] and not ok_c[1] else None)(
                _check_pose(W, r, robots, moving, carried))
            sol = _ik(W, r, r + "ur_gripper_center", start + off, ry.FS.vectorZ, down, rng, strict,
                      extra=[(ry.FS.vectorX, xdir)])
            if sol is None or not _reachable_from_home(W, r, robots, moving, sol[0], rng):
                _LOG.debug(f"{frame} psi {psi:.2f}: " + ("no pick IK" if sol is None else "pick not reachable"))
                continue
            q_pick = sol[0]
            W.setJointState(q_pick, names)
            W.attach(r + "ur_gripper", frame)
            W.getFrame(frame).setContact(-1)
            carried = {f: r for f in carried_frames}
            sol = _ik(W, r, r + "ur_gripper_center", target + off, ry.FS.vectorZ, down, rng,
                      lambda: strict(carried), extra=[(ry.FS.vectorX, xdir)])
            ok = sol is not None and _reachable_from_home(W, r, robots, moving, sol[0], rng, carried=carried)
            if ok:
                q_place = sol[0]
                W.setJointState(q_place, names)
                W.attach("table", frame)
                ok = _reachable_from_home(W, r, robots, moving, home_r, rng)
            if ok:
                W.setJointState(home)
                done = (q_pick, q_place)
                break
            _LOG.debug(f"{frame} psi {psi:.2f}: " + ("no place IK" if sol is None else "carry or return leg failed"))
            W.clear()
            W.addConfigurationCopy(snapshot)
        if done is None:
            raise RuntimeError(f"container: no feasible pick / place for {frame} by {r}")
        q_pick_full, q_place_full = home.copy(), home.copy()
        q_pick_full[idx], q_place_full[idx] = done
        steps.append((r, frame, q_pick_full[idx].copy(), q_place_full[idx].copy()))
    C.setJointState(home)
    return C, robots, steps, dict(task=task, layers=layers, width=w, wall_height=hc, home_conflicts={},
                                  num_arms=num_arms)


@register([
    ("rai.unpack_lid_2", {"task": "unpack", "layers": 1}),
    ("rai.unpack_lid_2_deep", {"task": "unpack", "layers": 2}),
    ("rai.pack_lid_2", {"task": "pack", "layers": 1}),
    ("rai.pack_lid_2_deep", {"task": "pack", "layers": 2}),
    ("rai.unpack_lid_3", {"task": "unpack", "layers": 1, "num_arms": 3}),
    ("rai.unpack_lid_3_deep", {"task": "unpack", "layers": 2, "num_arms": 3}),
    ("rai.pack_lid_3", {"task": "pack", "layers": 1, "num_arms": 3}),
    ("rai.pack_lid_3_deep", {"task": "pack", "layers": 2, "num_arms": 3}),
    ("rai.unpack_lid_4", {"task": "unpack", "layers": 1, "num_arms": 4}),
    ("rai.unpack_lid_4_deep", {"task": "unpack", "layers": 2, "num_arms": 4}),
    ("rai.pack_lid_4", {"task": "pack", "layers": 1, "num_arms": 4}),
    ("rai.pack_lid_4_deep", {"task": "pack", "layers": 2, "num_arms": 4}),
])
class rai_container_lid(SequenceMixin, rai_env):
    """Lidded container: lid off, objects out (unpack), or objects in, lid on (pack); then all arms home."""

    def __init__(self, task="unpack", layers=1, seed=0, num_arms=2):
        self.C, self.robots, steps, self.scene = make_container_env(task, layers, seed=seed, num_arms=num_arms)
        rai_env.__init__(self)
        self.manipulating_env = True
        self.tasks = []
        for r, frame, q_pick, q_place in steps:
            self.tasks.append(Task(f"pick_{frame}", [r], SingleGoal(q_pick), "pick", frames=[r + "ur_gripper", frame]))
            self.tasks.append(Task(f"place_{frame}", [r], SingleGoal(q_place), "place", frames=["table", frame]))
        self.tasks.append(Task("terminal", self.robots, SingleGoal(self.C.getJointState())))
        self.sequence = self._make_sequence_from_names([t.name for t in self.tasks])
        BaseModeLogic.__init__(self)
        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005
        self._set_default_safe_pose()


@register([
    ("rai.unpack_lid_2_dep", {"layers": 1}),
    ("rai.unpack_lid_3_dep", {"layers": 1, "num_arms": 3}),
    ("rai.unpack_lid_4_dep", {"layers": 1, "num_arms": 4}),
])
class rai_container_lid_dep(DependencyGraphMixin, rai_env):
    """Unpacking with a chain per arm: per-arm object chains, each starting after a1 removes the lid."""

    def __init__(self, layers=1, seed=0, num_arms=2):
        self.C, self.robots, steps, self.scene = make_container_env("unpack", layers, seed=seed, num_arms=num_arms)
        rai_env.__init__(self)
        self.manipulating_env = True
        self.tasks = []
        self.graph = DependencyGraph()
        last = {}
        for r, frame, q_pick, q_place in steps:
            self.tasks.append(Task(f"pick_{frame}", [r], SingleGoal(q_pick), "pick", frames=[r + "ur_gripper", frame]))
            self.tasks.append(Task(f"place_{frame}", [r], SingleGoal(q_place), "place", frames=["table", frame]))
            self.graph.add_dependency(f"place_{frame}", f"pick_{frame}")
            if r in last:
                self.graph.add_dependency(f"pick_{frame}", last[r])
            elif frame != "obj_lid":
                self.graph.add_dependency(f"pick_{frame}", "place_obj_lid")
            last[r] = f"place_{frame}"
        self.tasks.append(Task("terminal", self.robots, SingleGoal(self.C.getJointState())))
        for r in self.robots:
            self.graph.add_dependency("terminal", last[r])
        BaseModeLogic.__init__(self)
        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005
        self.spec.dependency = DependencyType.UNORDERED
        self._set_default_safe_pose()


# --------------------------------------------------------------------------------------------- swap corridor
def make_swap_corridor_env(robots_per_side=2, num_bays=1, corridor_width=None, corridor_length=1.6,
                           robot_radius=0.08, shape="disc"):
    """Two rooms joined by a corridor along x with passing bays on its +y side. Robots in the left room swap with
    the robots in the right room (slot k <-> mirrored slot k)."""
    r = robot_radius
    box = (0.30, 0.14) if shape == "box" else None
    lane = box[1] if box else 2 * r
    long_side = box[0] if box else 2 * r
    cw = lane + 0.06 if corridor_width is None else corridor_width
    room, Ly = 0.70, 1.10
    Lx = corridor_length + 2 * room
    C = ry.Config()
    C.addFrame("world")
    _box(C, "floor", [Lx + 0.1, Ly + 0.1, 0.04], [0, 0, -0.02], [0.86, 0.86, 0.84], contact=0)
    h, zc = 0.08, 0.04
    wall = [0.30, 0.32, 0.36]
    for s, (a, b) in (("x+", (Lx / 2, Lx / 2 + 0.05)), ("x-", (-Lx / 2 - 0.05, -Lx / 2))):
        _box(C, f"border_{s}", [b - a, Ly + 0.1, h], [(a + b) / 2, 0, zc], wall)
    for s, (a, b) in (("y+", (Ly / 2, Ly / 2 + 0.05)), ("y-", (-Ly / 2 - 0.05, -Ly / 2))):
        _box(C, f"border_{s}", [Lx, b - a, h], [0, (a + b) / 2, zc], wall)
    x0, x1 = -corridor_length / 2, corridor_length / 2
    _box(C, "corridor_low", [corridor_length, Ly / 2 - cw / 2, h], [0, -(Ly / 2 + cw / 2) / 2, zc], wall)
    bw, bd = long_side + 0.08, long_side + 0.06
    bays = [x0 + corridor_length * (k + 1) / (num_bays + 1) for k in range(num_bays)]
    cuts = [x0] + [c for xb in bays for c in (xb - bw / 2, xb + bw / 2)] + [x1]
    for k in range(0, len(cuts), 2):
        a, b = cuts[k], cuts[k + 1]
        _box(C, f"corridor_high_{k // 2}", [b - a, Ly / 2 - cw / 2, h], [(a + b) / 2, (Ly / 2 + cw / 2) / 2, zc], wall)
    for k, xb in enumerate(bays):
        y0 = cw / 2 + bd
        if y0 < Ly / 2:
            _box(C, f"bay_back_{k}", [bw, Ly / 2 - y0, h], [xb, (Ly / 2 + y0) / 2, zc], wall)
    n = robots_per_side
    cols = 1 if n <= 3 else 2
    rows = int(np.ceil(n / cols))
    slots = [(-Lx / 2 + 0.20 + 0.26 * c, (rr - (rows - 1) / 2) * 0.28) for rr in range(rows) for c in range(cols)][:n]
    robots, start, goal = [], [], []
    for side in (-1, 1):
        for k, (sx, sy) in enumerate(slots):
            name = f"a{len(robots) + 1}_"
            robots.append(name)
            p = np.array([side * -sx if side == 1 else sx, sy])  # left slots for side -1, mirrored for +1
            start.append(p)
            goal.append(np.array([-p[0], sy]))
    lim = [-Lx / 2 + r, Lx / 2 - r, -Ly / 2 + r, Ly / 2 - r]
    # Shapes: rai (0.2.9) reports zero penetration for cylinder-cylinder and box-box pairs, and the env only rejects
    # penetration above its tolerance, so discs are spheres and rectangles horizontal capsules (exact against each
    # other and against the box walls).
    for i, (name, p) in enumerate(zip(robots, start)):
        pre = C.addFrame(name + "pre", "world").setPosition([0, 0, zc])
        f = C.addFrame(name + "base", pre.name)
        if box:
            f.setJoint(ry.JT.transXYPhi, limits=lim + [-np.pi, np.pi]).setJointState([p[0], p[1], 0.0])
            C.addFrame(name + "body", f.name).setShape(ry.ST.capsule, [box[0] - box[1], box[1] / 2]).setRelativeQuaternion(
                [0.70710678, 0.0, 0.70710678, 0.0]).setColor(PALETTE[i % len(PALETTE)]).setContact(1)
        else:
            f.setShape(ry.ST.sphere, [r]).setColor(PALETTE[i % len(PALETTE)]).setContact(1)
            f.setJoint(ry.JT.transXY, limits=lim).setJointState(list(p))
    if box:
        start = [np.array([*p, 0.0]) for p in start]
        goal = [np.array([*p, 0.0]) for p in goal]
    for i, (name, g) in enumerate(zip(robots, goal)):
        C.addFrame(f"goal_{name}").setShape(ry.ST.cylinder, [0.004, 0.05]).setPosition([g[0], g[1], 0.002]).setColor(
            PALETTE[i % len(PALETTE)] + [0.6]).setContact(0)
    return C, robots, start, goal, dict(corridor_width=cw, bays=bays, bay_width=bw, bay_depth=bd, shape=shape)


@register([
    ("rai.swap_corridor_1v1_b1", {"robots_per_side": 1, "num_bays": 1}),
    ("rai.swap_corridor_2v2_b1", {"robots_per_side": 2, "num_bays": 1}),
    ("rai.swap_corridor_2v2_b2", {"robots_per_side": 2, "num_bays": 2}),
    ("rai.swap_corridor_3v3_b2", {"robots_per_side": 3, "num_bays": 2}),
    ("rai.swap_corridor_3v3_b3", {"robots_per_side": 3, "num_bays": 3}),
    ("rai.swap_corridor_2v2_wide", {"robots_per_side": 2, "num_bays": 0, "corridor_width": 0.42}),
    ("rai.swap_corridor_2v2_b2_box", {"robots_per_side": 2, "num_bays": 2, "shape": "box"}),
    ("rai.swap_corridor_2v2_b2_roundtrip", {"robots_per_side": 2, "num_bays": 2, "roundtrip": True}),
])
class rai_swap_corridor(SequenceMixin, rai_env):
    """One-lane corridor swap. Single goal (all swapped) or, with roundtrip, every robot first visits its mirrored
    slot (alternating sides), then all return to their start."""

    def __init__(self, robots_per_side=2, num_bays=1, corridor_width=None, corridor_length=1.6, shape="disc",
                 roundtrip=False):
        self.C, self.robots, start, goal, self.scene = make_swap_corridor_env(
            robots_per_side, num_bays, corridor_width, corridor_length, shape=shape)
        rai_env.__init__(self)
        if roundtrip:
            n = robots_per_side
            order = [self.robots[k + side * n] for k in range(n) for side in (0, 1)]
            self.tasks = [Task(f"{r}cross", [r], SingleGoal(goal[self.robots.index(r)])) for r in order]
            self.tasks.append(Task("terminal", self.robots, SingleGoal(np.concatenate(start))))
            self.sequence = self._make_sequence_from_names([f"{r}cross" for r in order] + ["terminal"])
        else:
            self.tasks = [Task("terminal", self.robots, SingleGoal(np.concatenate(goal)))]
            self.sequence = self._make_sequence_from_names(["terminal"])
        BaseModeLogic.__init__(self)
        self.collision_tolerance = 0.001  # 1 mm: with 0.01 the 16 cm discs visibly overlap
        self.collision_resolution = 0.005
        _mark_static(self)
        if not roundtrip:
            self.spec.goals = GoalType.SINGLE_GOAL
        self.spec.home_pose = SafePoseType.HAS_NO_SAFE_HOME_POSE  # a start slot is some other robot's goal
