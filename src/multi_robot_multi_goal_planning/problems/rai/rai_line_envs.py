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
