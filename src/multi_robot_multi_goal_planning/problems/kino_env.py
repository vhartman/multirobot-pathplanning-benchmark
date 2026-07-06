"""
Kinodynamic environments built on AbstractEnvironment's geometric collision backend.

Class hierarchy:
  KinodynamicPlanningProblem  (abstract dynamics interface, in kinodynamic_planning_env.py)
   └─ DoubleIntegratorEnvironment   (concrete second-order dynamics + sphere/rect collision
                                     via position projection, per-robot layout [pos, vel])
        └─ kino.point_mass_2d_swap  (concrete scenario)
"""

import numpy as np
from numba import njit  # ty: ignore[possibly-unbound-import]

from typing import List, Optional, Tuple
from numpy.typing import NDArray

from .planning_env import (
    BaseModeLogic,
    SequenceMixin,
    Mode,
    Task,
    ProblemSpec,
    AgentType,
    ConstraintType,
    ManipulationType,
    DependencyType,
    DynamicsType,
    GoalType,
    SafePoseType,
)
from .core.configuration import Configuration, NpConfiguration
from .core.goals import SingleGoal, GoalRegion
from .core.registry import register

from .abstract_env import AbstractEnvironment, Sphere, Rectangle
from .kinodynamic_planning_env import KinodynamicPlanningProblem


@njit(cache=True, fastmath=True)
def _di_min_horizon_njit(p0, v0, p1, v1, u_max, dt, max_bisect):
    """Compiled version of the min-effort minimum-feasible-horizon search (the RRT* hot path:
    ~half the profiled steering cost). Bracket by geometric expansion, then bisect. Returns T,
    or -1.0 if infeasible. Fully scalar loops (no numpy per-op overhead)."""
    n = p0.shape[0]
    T = dt
    feasible = False
    for _ in range(80):
        det = -(T * T * T * T) / 12.0
        pk = 0.0
        for i in range(n):
            dp = p1[i] - p0[i] - v0[i] * T
            dv = v1[i] - v0[i]
            a = (dv * (T * T * T / 6.0) - dp * (T * T / 2.0)) / det
            b = (T * dp - (T * T / 2.0) * dv) / det
            e2 = abs(a + b * T)
            e = abs(a)
            if e2 > e:
                e = e2
            if e > pk:
                pk = e
        if pk <= u_max + 1e-9:
            feasible = True
            break
        T *= 1.3
    if not feasible:
        return -1.0
    T_lo = T / 1.3 if T > dt else 0.0
    T_hi = T
    for _ in range(max_bisect):
        Tm = 0.5 * (T_lo + T_hi)
        det = -(Tm * Tm * Tm * Tm) / 12.0
        pk = 0.0
        for i in range(n):
            dp = p1[i] - p0[i] - v0[i] * Tm
            dv = v1[i] - v0[i]
            a = (dv * (Tm * Tm * Tm / 6.0) - dp * (Tm * Tm / 2.0)) / det
            b = (Tm * dp - (Tm * Tm / 2.0) * dv) / det
            e2 = abs(a + b * Tm)
            e = abs(a)
            if e2 > e:
                e = e2
            if e > pk:
                pk = e
        if pk <= u_max + 1e-9:
            T_hi = Tm
        else:
            T_lo = Tm
    return T_hi


class ProjectedKinodynamicEnvironment(KinodynamicPlanningProblem, AbstractEnvironment):
    """Base for kinodynamic envs whose collision check only depends on a low-dimensional
    *position* projection of the (possibly larger) dynamic state. It reuses the
    AbstractEnvironment sphere/rectangle collision backend by projecting every robot's
    state onto its position via robot_position(). Subclasses must set
    `position_dim_per_robot` (the collision dimensionality, e.g. 2 for planar robots)
    and implement robot_position / forward_dynamics / state_bounds / control_bounds."""

    position_dim_per_robot: int

    # Whether steer(x0, x1) returns an *exact* connection ending at x1. Exact steering
    # enables sensible RRT* rewiring; feedback/approximate steering does not.
    has_exact_steering: bool = False

    def __init__(self):
        AbstractEnvironment.__init__(self)

    # lazily-built (template config, extended-state index array) for the position
    # projection: is_collision_free runs per trajectory waypoint, so rebuilding an
    # NpConfiguration from per-robot slices every call (from_list => full __init__)
    # was a measurable share of the profile. from_flat shares the template's slice
    # metadata, and one fancy-index replaces the per-robot python loop.
    _pos_proj: Optional[Tuple[NpConfiguration, NDArray]] = None

    def _build_pos_proj(self) -> Tuple[NpConfiguration, NDArray]:
        n = self.start_pos.num_agents()
        # recover the extended-state indices of each robot's position entries by
        # projecting an index-valued state through robot_position (valid because every
        # robot_position implementation is a pure selection of state entries)
        dim = np.asarray(self.start_pos.state()).shape[0]
        probe = self.start_pos.from_flat(np.arange(dim, dtype=np.float64))
        idx = np.concatenate(
            [np.asarray(self.robot_position(probe, i)) for i in range(n)]
        ).astype(np.int64)
        per_robot = [np.asarray(self.robot_position(self.start_pos, i)) for i in range(n)]
        template = NpConfiguration.from_list(per_robot)
        # the index projection must reproduce the slow path
        assert np.array_equal(
            np.asarray(self.start_pos.state())[idx], np.concatenate(per_robot)
        )
        self._pos_proj = (template, idx)
        return self._pos_proj

    def _position_configuration(self, q: Configuration) -> NpConfiguration:
        proj = self._pos_proj
        if proj is None:
            proj = self._build_pos_proj()
        template, idx = proj
        return template.from_flat(np.asarray(q.state())[idx])

    def is_collision_free(
        self, q: Optional[Configuration], mode: Optional[Mode]
    ) -> bool:
        if q is None:
            raise ValueError
        return AbstractEnvironment.is_collision_free(
            self, self._position_configuration(q), mode
        )

    def show_config(self, q, blocking=True):
        if self.position_dim_per_robot != 2:
            return
        AbstractEnvironment.show_config(
            self, self._position_configuration(q), blocking=blocking
        )

    def show(self, blocking=True):
        if self.position_dim_per_robot != 2:
            return
        original_start = self.start_pos
        try:
            self.start_pos = self._position_configuration(original_start)
            AbstractEnvironment.show(self, blocking=blocking)
        finally:
            self.start_pos = original_start


class DoubleIntegratorEnvironment(ProjectedKinodynamicEnvironment):
    """Multi-robot point-mass env with second-order dynamics ẋ = [vel; u]
    and per-robot extended-state layout [pos..., vel...]. Reuses the
    AbstractEnvironment sphere/rectangle collision check via position projection."""

    position_dim_per_robot: int
    u_max: float
    has_exact_steering = True

    def forward_dynamics(self, x: NDArray, u: NDArray) -> NDArray:
        n = self.start_pos.num_agents()
        d = self.position_dim_per_robot
        x_dot = np.zeros_like(x)
        for i in range(n):
            base = i * 2 * d
            x_dot[base : base + d] = x[base + d : base + 2 * d]
            x_dot[base + d : base + 2 * d] = u[i * d : (i + 1) * d]
        return x_dot

    def _pos_vel_indices(self):
        """Index arrays that pair every position DOF with its velocity DOF, so the
        per-DOF double-integrator steering can be vectorized across all robots."""
        cached = getattr(self, "_pv_idx_cache", None)
        if cached is not None:
            return cached
        n = self.start_pos.num_agents()
        d = self.position_dim_per_robot
        pos_idx, vel_idx = [], []
        for i in range(n):
            base = i * 2 * d
            for k in range(d):
                pos_idx.append(base + k)
                vel_idx.append(base + d + k)
        cached = (np.array(pos_idx), np.array(vel_idx))
        self._pv_idx_cache = cached
        return cached

    def _di_coeffs(self, p0, v0, p1, v1, T):
        """Minimum-effort affine control u(t) = a + b*t that drives each decoupled
        double integrator from (p0, v0) to (p1, v1) in time T. Returns (a, b) and the
        peak |u| over [0, T] (which, being linear in t, is attained at an endpoint)."""
        dp = p1 - p0 - v0 * T
        dv = v1 - v0
        det = -(T**4) / 12.0
        a = (dv * (T**3 / 6.0) - dp * (T**2 / 2.0)) / det
        b = (T * dp - (T**2 / 2.0) * dv) / det
        peak = np.maximum(np.abs(a), np.abs(a + b * T))
        return a, b, peak

    def _min_feasible_horizon(self, p0, v0, p1, v1):
        """Smallest horizon T for which the minimum-effort connection respects the control
        bounds. First brackets a feasible T by geometric expansion, then BISECTS within the
        bracket to the exact boundary (peak control == u_max) instead of accepting the ~1.3x
        overshoot. Since the cost is trajectory time, this directly lowers it. The min-effort
        affine control is globally well-behaved (its peak decreases with T), so this is safe;
        a true time-optimal (bang-bang) profile would be faster still but needs a
        velocity-limited trapezoidal generator (the accel-limited min-time is non-monotonic
        in T for excursion targets)."""
        # 16 bisections reach ~dt/2^16 precision -- far finer than needed (we round the
        # horizon to a whole number of dt steps anyway). The compiled routine replaces the old
        # 40-iteration numpy bisection (the top steering hot path).
        T = _di_min_horizon_njit(
            np.ascontiguousarray(p0), np.ascontiguousarray(v0),
            np.ascontiguousarray(p1), np.ascontiguousarray(v1),
            float(self.u_max), float(self.integration_dt), 16,
        )
        return None if T < 0.0 else float(T)

    def steer(self, x0, x1, max_steps=None):
        """Exact analytic steering for the multi-robot double integrator.

        Finds the (bisected) smallest common horizon T for which the minimum-effort
        connection stays within the control bounds, then samples the closed-form trajectory
        at integration_dt. The final state equals x1 exactly (to floating-point precision),
        which is what makes zero-velocity mode-switch goals reachable to the 1e-8 tolerance
        used by is_transition/done."""
        x0 = np.asarray(x0, dtype=float)
        x1 = np.asarray(x1, dtype=float)
        pos_idx, vel_idx = self._pos_vel_indices()

        p0, v0 = x0[pos_idx], x0[vel_idx]
        p1, v1 = x1[pos_idx], x1[vel_idx]

        dt = self.integration_dt

        # already there
        if np.linalg.norm(x1 - x0) < 1e-12:
            return [x1.copy()]

        T = self._min_feasible_horizon(p0, v0, p1, v1)
        if T is None:
            return None

        # snap horizon to a whole number of integration steps (>= T keeps it feasible)
        N = max(1, int(np.ceil(T / dt)))
        T = N * dt
        a, b, _ = self._di_coeffs(p0, v0, p1, v1, T)

        # vectorised closed-form sampling of the double integrator under u(t) = a + b*t
        t = (np.arange(1, N + 1) * dt)[:, None]  # (N, 1)
        arr = np.tile(x0, (N, 1))
        arr[:, pos_idx] = p0 + v0 * t + a * t * t / 2.0 + b * t * t * t / 6.0
        arr[:, vel_idx] = v0 + a * t + b * t * t / 2.0
        arr[-1] = x1  # exact endpoint (guard against float drift at t = T)
        return list(arr)

    def steer_fixed_time(self, x0, x1, num_steps):
        """Affine (minimum-effort) connection from x0 to x1 over a *fixed* horizon
        T = num_steps * dt. Returns (states, peak_control_per_dof); the final state equals
        x1 exactly. Unlike steer(), the horizon is prescribed (not minimised), which is what
        per-robot shortcutting needs: re-steering one robot over exactly the same number of
        steps keeps every robot's timeline aligned. peak_control_per_dof is ordered like
        _pos_vel_indices (robot 0's dofs first, then robot 1's, ...), so a caller can check
        feasibility for just one robot's block."""
        x0 = np.asarray(x0, dtype=float)
        x1 = np.asarray(x1, dtype=float)
        pos_idx, vel_idx = self._pos_vel_indices()
        p0, v0 = x0[pos_idx], x0[vel_idx]
        p1, v1 = x1[pos_idx], x1[vel_idx]

        dt = self.integration_dt
        T = num_steps * dt
        a, b, peak = self._di_coeffs(p0, v0, p1, v1, T)

        states = []
        for j in range(1, num_steps + 1):
            t = j * dt
            x = x0.copy()
            x[pos_idx] = p0 + v0 * t + a * t**2 / 2.0 + b * t**3 / 6.0
            x[vel_idx] = v0 + a * t + b * t**2 / 2.0
            states.append(x)
        states[-1] = x1.copy()
        return states, peak

    def steer_cost(self, x0, x1):
        """Time cost of the steering connection WITHOUT generating the trajectory or
        checking collision. Used for cheap RRT* cost ranking (like batch_config_cost in the
        geometric planner), so collision checking can be deferred to only the accepted edges."""
        x0 = np.asarray(x0, dtype=float)
        x1 = np.asarray(x1, dtype=float)
        if np.linalg.norm(x1 - x0) < 1e-12:
            return 0.0
        pos_idx, vel_idx = self._pos_vel_indices()
        T = self._min_feasible_horizon(x0[pos_idx], x0[vel_idx], x1[pos_idx], x1[vel_idx])
        if T is None:
            return None
        return max(1, int(np.ceil(T / self.integration_dt))) * self.integration_dt

    def robot_position(self, q: Configuration, robot_index: int) -> NDArray:
        return np.asarray(q[robot_index])[: self.position_dim_per_robot]

    def state_bounds(self) -> NDArray:
        return self.limits

    def control_bounds(self) -> NDArray:
        n = self.start_pos.num_agents()
        d = self.position_dim_per_robot
        bounds = np.zeros((2, n * d))
        bounds[0, :] = -self.u_max
        bounds[1, :] = +self.u_max
        return bounds


def make_kino_middle_obstacle(num_agents=2, pos_dim=2):
    """State bounds and a middle obstacle for `num_agents` point masses in `pos_dim`
    dimensions with extended state [pos..., vel...] per robot. The obstacle lives in the
    first two position dimensions (extended trivially in higher dims)."""
    state_dim_per_robot = 2 * pos_dim
    total_dim = num_agents * state_dim_per_robot

    pos_lo, pos_hi = -2.0, 2.0
    vel_lo, vel_hi = -3.0, 3.0

    limits = np.zeros((2, total_dim))
    for i in range(num_agents):
        base = i * state_dim_per_robot
        limits[0, base : base + pos_dim] = pos_lo
        limits[1, base : base + pos_dim] = pos_hi
        limits[0, base + pos_dim : base + 2 * pos_dim] = vel_lo
        limits[1, base + pos_dim : base + 2 * pos_dim] = vel_hi

    sphere_center = np.zeros(pos_dim)
    rect_center = np.zeros(pos_dim)
    rect_center[1] = 0.4
    obstacles = [
        Sphere(sphere_center, 0.2),
        Rectangle(rect_center, np.ones(pos_dim) * 0.5),
    ]
    return limits, obstacles


def make_kino_middle_obstacle_2d():
    return make_kino_middle_obstacle(num_agents=2, pos_dim=2)


@register("kino.point_mass_2d_swap")
class kino_point_mass_2d_swap(SequenceMixin, DoubleIntegratorEnvironment):
    """Two 2D point-mass robots swap positions through a cluttered scene.
    Goal convention: come to rest (v=0) at goal position."""

    def __init__(self):
        DoubleIntegratorEnvironment.__init__(self)

        self.position_dim_per_robot = 2
        self.u_max = 2.0
        self.integration_dt = 0.05

        self.limits, self.obstacles = make_kino_middle_obstacle_2d()

        a1_start = np.array([-0.8, 0.0, 0.0, 0.0])
        a2_start = np.array([+0.8, 0.0, 0.0, 0.0])
        self.start_pos = NpConfiguration.from_list([a1_start, a2_start])

        self.agent_radii = [0.1, 0.1]

        self.robot_idx = {"a1": [0, 1, 2, 3], "a2": [4, 5, 6, 7]}
        self.robot_dims = {"a1": 4, "a2": 4}
        self.robots = ["a1", "a2"]

        a1_goal_state = np.array([+0.8, 0.0, 0.0, 0.0])
        a2_goal_state = np.array([-0.8, 0.0, 0.0, 0.0])
        terminal_state = np.concatenate([a2_goal_state, a1_goal_state])

        self.tasks = [
            Task("a1_goal", ["a1"], SingleGoal(a1_goal_state)),
            Task("a2_goal", ["a2"], SingleGoal(a2_goal_state)),
            Task("terminal", ["a1", "a2"], SingleGoal(terminal_state)),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a2_goal", "a1_goal", "terminal"]
        )

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self.spec = ProblemSpec(
            agent_type=AgentType.MULTI_AGENT,
            constraints=ConstraintType.UNCONSTRAINED,
            manipulation=ManipulationType.STATIC,
            dependency=DependencyType.FULLY_ORDERED,
            dynamics=DynamicsType.KINODYNAMIC,
            goals=GoalType.MULTI_GOAL,
            home_pose=SafePoseType.HAS_NO_SAFE_HOME_POSE,
        )

        BaseModeLogic.__init__(self)


@register([
    ("kino.point_mass_3d_swap", {"pos_dim": 3}),
    ("kino.point_mass_4d_swap", {"pos_dim": 4}),
])
class kino_point_mass_nd_swap(SequenceMixin, DoubleIntegratorEnvironment):
    """Two n-D point-mass robots swap positions through the middle obstacle.

    This is the "treat a robot arm as an n-dimensional point mass" case: the joint
    vector is the position, controlled by a double integrator, with the same
    zero-velocity mode-switch convention. Collision checking still uses the abstract
    sphere/rectangle world (obstacle acts on the first two coordinates)."""

    def __init__(self, pos_dim=3):
        DoubleIntegratorEnvironment.__init__(self)

        self.position_dim_per_robot = pos_dim
        self.u_max = 2.0
        self.integration_dt = 0.05

        self.limits, self.obstacles = make_kino_middle_obstacle(
            num_agents=2, pos_dim=pos_dim
        )

        state_dim = 2 * pos_dim

        def state(first_coord):
            s = np.zeros(state_dim)
            s[0] = first_coord
            return s

        a1_start = state(-0.8)
        a2_start = state(+0.8)
        self.start_pos = NpConfiguration.from_list([a1_start, a2_start])

        self.agent_radii = [0.1, 0.1]

        self.robot_idx = {
            "a1": list(range(0, state_dim)),
            "a2": list(range(state_dim, 2 * state_dim)),
        }
        self.robot_dims = {"a1": state_dim, "a2": state_dim}
        self.robots = ["a1", "a2"]

        a1_goal_state = state(+0.8)
        a2_goal_state = state(-0.8)
        terminal_state = np.concatenate([a1_goal_state, a2_goal_state])

        self.tasks = [
            Task("a1_goal", ["a1"], SingleGoal(a1_goal_state)),
            Task("a2_goal", ["a2"], SingleGoal(a2_goal_state)),
            Task("terminal", ["a1", "a2"], SingleGoal(terminal_state)),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a2_goal", "a1_goal", "terminal"]
        )

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self.spec = ProblemSpec(
            agent_type=AgentType.MULTI_AGENT,
            constraints=ConstraintType.UNCONSTRAINED,
            manipulation=ManipulationType.STATIC,
            dependency=DependencyType.FULLY_ORDERED,
            dynamics=DynamicsType.KINODYNAMIC,
            goals=GoalType.MULTI_GOAL,
            home_pose=SafePoseType.HAS_NO_SAFE_HOME_POSE,
        )

        BaseModeLogic.__init__(self)


def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class CarEnvironment(ProjectedKinodynamicEnvironment):
    """Multi-robot second-order unicycle ("car") env. Per-robot state is
    [x, y, theta, v] and control is [a, omega] (longitudinal acceleration, turn rate):

        dx = v cos(theta), dy = v sin(theta), dtheta = omega, dv = a

    The car can come to rest (v = 0), which is exactly the zero-velocity constraint
    required at mode switches. There is no closed-form two-point steering here, so the
    steering function is a simple pose-stabilizing *feedback* controller that drives the
    car towards a target position while braking to a stop -- a "nice" steering function in
    the sense the user asked for, without resorting to random propagation."""

    position_dim_per_robot = 2  # collision only sees (x, y)

    # dynamics limits
    v_max = 1.5
    a_max = 2.0
    omega_max = 3.0

    # feedback controller gains
    k_rho = 1.5
    k_alpha = 4.0
    k_v = 3.0
    reach_tol = 0.05
    v_stop_tol = 0.1

    @property
    def angular_dims(self):
        """Flat-state indices that are angles (per-robot heading theta), so a trajectory
        optimiser (iLQR) can wrap angle differences in its cost. Per-robot layout is
        [x, y, theta, v]."""
        return [i * 4 + 2 for i in range(self.start_pos.num_agents())]

    def forward_dynamics(self, x: NDArray, u: NDArray) -> NDArray:
        n = self.start_pos.num_agents()
        x_dot = np.zeros_like(x)
        for i in range(n):
            b = i * 4
            theta, v = x[b + 2], x[b + 3]
            a, omega = u[i * 2], u[i * 2 + 1]
            x_dot[b] = v * np.cos(theta)
            x_dot[b + 1] = v * np.sin(theta)
            x_dot[b + 2] = omega
            x_dot[b + 3] = a
        return x_dot

    def robot_position(self, q: Configuration, robot_index: int) -> NDArray:
        return np.asarray(q[robot_index])[:2]

    def state_bounds(self) -> NDArray:
        return self.limits

    def control_bounds(self) -> NDArray:
        n = self.start_pos.num_agents()
        bounds = np.zeros((2, n * 2))
        for i in range(n):
            bounds[0, i * 2] = -self.a_max
            bounds[1, i * 2] = +self.a_max
            bounds[0, i * 2 + 1] = -self.omega_max
            bounds[1, i * 2 + 1] = +self.omega_max
        return bounds

    def steer(self, x0, x1, max_steps=None):
        n = self.start_pos.num_agents()
        dt = self.integration_dt
        steps = max_steps if max_steps is not None else 400

        x = np.asarray(x0, dtype=float).copy()
        target = np.asarray(x1, dtype=float)

        states = []
        for _ in range(steps):
            u = np.zeros(n * 2)
            all_reached = True
            for i in range(n):
                b = i * 4
                px, py, th, v = x[b], x[b + 1], x[b + 2], x[b + 3]
                dx, dy = target[b] - px, target[b + 1] - py
                rho = float(np.hypot(dx, dy))

                if rho > self.reach_tol or abs(v) > self.v_stop_tol:
                    all_reached = False

                if rho < self.reach_tol:
                    v_des, omega = 0.0, 0.0
                else:
                    alpha = _wrap_angle(np.arctan2(dy, dx) - th)
                    v_des = np.clip(
                        self.k_rho * rho * np.cos(alpha), -self.v_max, self.v_max
                    )
                    omega = np.clip(
                        self.k_alpha * alpha, -self.omega_max, self.omega_max
                    )
                a = np.clip(self.k_v * (v_des - v), -self.a_max, self.a_max)
                u[i * 2], u[i * 2 + 1] = a, omega

            x = x + dt * self.forward_dynamics(x, u)
            for i in range(n):
                x[i * 4 + 2] = _wrap_angle(x[i * 4 + 2])
                x[i * 4 + 3] = np.clip(x[i * 4 + 3], -self.v_max, self.v_max)
            states.append(x.copy())
            if all_reached:
                break
        return states


def _car_goal_region(gx, gy, pos_tol=0.12, v_tol=0.15):
    """Box goal for a single car: position near (gx, gy), any heading, ~zero velocity."""
    lo = np.array([gx - pos_tol, gy - pos_tol, -4.0, -v_tol])
    hi = np.array([gx + pos_tol, gy + pos_tol, +4.0, +v_tol])
    return lo, hi


@register("kino.car_2d_swap")
class kino_car_2d_swap(SequenceMixin, CarEnvironment):
    """Two cars (second-order unicycles) swap sides through the cluttered scene AND then
    return to their starting positions, each coming to rest at every sub-goal. Heading at a
    goal is free; only position and (near) zero velocity are constrained at the mode switch."""

    def __init__(self):
        CarEnvironment.__init__(self)

        self.integration_dt = 0.05

        _, self.obstacles = make_kino_middle_obstacle(num_agents=2, pos_dim=2)

        lo_r = np.array([-2.0, -2.0, -np.pi, -self.v_max])
        hi_r = np.array([+2.0, +2.0, +np.pi, +self.v_max])
        self.limits = np.stack(
            [np.concatenate([lo_r, lo_r]), np.concatenate([hi_r, hi_r])]
        )

        a1_start = np.array([-0.8, 0.0, 0.0, 0.0])  # facing +x
        a2_start = np.array([+0.8, 0.0, np.pi, 0.0])  # facing -x
        self.start_pos = NpConfiguration.from_list([a1_start, a2_start])

        self.agent_radii = [0.1, 0.1]

        self.robot_idx = {"a1": [0, 1, 2, 3], "a2": [4, 5, 6, 7]}
        self.robot_dims = {"a1": 4, "a2": 4}
        self.robots = ["a1", "a2"]

        # swap goals (each car to the other's side) and return goals (back to own start)
        a1_swap = _car_goal_region(+0.8, 0.0)
        a2_swap = _car_goal_region(-0.8, 0.0)
        a1_home = _car_goal_region(-0.8, 0.0)
        a2_home = _car_goal_region(+0.8, 0.0)
        # terminal: both cars back at their starting positions
        terminal_limits = np.stack(
            [np.concatenate([a1_home[0], a2_home[0]]), np.concatenate([a1_home[1], a2_home[1]])]
        )

        self.tasks = [
            Task("a1_swap", ["a1"], GoalRegion(np.stack(a1_swap))),
            Task("a2_swap", ["a2"], GoalRegion(np.stack(a2_swap))),
            Task("a1_return", ["a1"], GoalRegion(np.stack(a1_home))),
            Task("a2_return", ["a2"], GoalRegion(np.stack(a2_home))),
            Task("terminal", ["a1", "a2"], GoalRegion(terminal_limits)),
        ]

        # swap first (both cars cross), then return to start, then terminate
        self.sequence = self._make_sequence_from_names(
            ["a2_swap", "a1_swap", "a2_return", "a1_return", "terminal"]
        )

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self.spec = ProblemSpec(
            agent_type=AgentType.MULTI_AGENT,
            constraints=ConstraintType.UNCONSTRAINED,
            manipulation=ManipulationType.STATIC,
            dependency=DependencyType.FULLY_ORDERED,
            dynamics=DynamicsType.KINODYNAMIC,
            goals=GoalType.MULTI_GOAL,
            home_pose=SafePoseType.HAS_NO_SAFE_HOME_POSE,
        )

        BaseModeLogic.__init__(self)


class TrailerEnvironment(ProjectedKinodynamicEnvironment):
    """Multi-robot tractor-trailer ("car with trailer"). Per-robot state is
    [x, y, theta, theta_t, v] (cab pose, absolute trailer heading, cab speed) and control is
    [a, omega]:

        dx = v cos(theta),  dy = v sin(theta),  dtheta = omega,
        dtheta_t = (v / L) sin(theta - theta_t),   dv = a

    The trailer lags the cab and can jackknife -- a genuinely harder nonholonomic system. It
    has no closed-form steering (feedback controller here, then iLQR shortcutting), and comes
    to rest (v = 0) at mode switches. Collision covers BOTH bodies (cab disc + a couple of
    discs along the trailer)."""

    position_dim_per_robot = 2  # collision projection / nearest-neighbour uses the cab (x, y)

    v_max = 1.5
    a_max = 2.0
    omega_max = 2.0
    L = 0.3            # hitch-to-trailer-axle length
    r_cab = 0.09
    r_trailer = 0.08

    # feedback controller gains (gentler turning than the plain car, to limit jackknifing)
    k_rho = 1.2
    k_alpha = 2.2
    k_v = 3.0
    reach_tol = 0.05
    v_stop_tol = 0.1

    @property
    def angular_dims(self):
        return [i * 5 + k for i in range(self.start_pos.num_agents()) for k in (2, 3)]

    def forward_dynamics(self, x: NDArray, u: NDArray) -> NDArray:
        n = self.start_pos.num_agents()
        xd = np.zeros_like(x)
        for i in range(n):
            b = i * 5
            th, tht, v = x[b + 2], x[b + 3], x[b + 4]
            a, om = u[i * 2], u[i * 2 + 1]
            xd[b] = v * np.cos(th)
            xd[b + 1] = v * np.sin(th)
            xd[b + 2] = om
            xd[b + 3] = (v / self.L) * np.sin(th - tht)
            xd[b + 4] = a
        return xd

    def robot_position(self, q: Configuration, robot_index: int) -> NDArray:
        return np.asarray(q[robot_index])[:2]

    def state_bounds(self) -> NDArray:
        return self.limits

    def control_bounds(self) -> NDArray:
        n = self.start_pos.num_agents()
        bounds = np.zeros((2, n * 2))
        for i in range(n):
            bounds[0, i * 2] = -self.a_max
            bounds[1, i * 2] = +self.a_max
            bounds[0, i * 2 + 1] = -self.omega_max
            bounds[1, i * 2 + 1] = +self.omega_max
        return bounds

    def _robot_discs(self, s: NDArray):
        """(center, radius) discs covering the cab and the trailer body for one robot."""
        x, y, tht = s[0], s[1], s[3]
        c, sn = np.cos(tht), np.sin(tht)
        return [
            (np.array([x, y]), self.r_cab),
            (np.array([x - 0.5 * self.L * c, y - 0.5 * self.L * sn]), self.r_trailer),
            (np.array([x - self.L * c, y - self.L * sn]), self.r_trailer),
        ]

    def robot_segments(self, q: Configuration, robot_index: int):
        """Line segment(s) to render the trailer (cab -> trailer axle). Used by the viewers."""
        s = np.asarray(q[robot_index])
        c, sn = np.cos(s[3]), np.sin(s[3])
        return [(np.array([s[0], s[1]]), np.array([s[0] - self.L * c, s[1] - self.L * sn]))]

    def is_collision_free(self, q: Optional[Configuration], mode: Optional[Mode]) -> bool:
        if q is None:
            raise ValueError
        n = q.num_agents()
        discs = [self._robot_discs(np.asarray(q[i])) for i in range(n)]
        for i in range(n):
            for center, radius in discs[i]:
                for o in self.obstacles:
                    if o.collides_with_sphere(center, radius):
                        return False
        for i in range(n):
            for j in range(i + 1, n):
                for ci, ri in discs[i]:
                    for cj, rj in discs[j]:
                        if np.linalg.norm(ci - cj) < ri + rj:
                            return False
        return True

    def steer(self, x0, x1, max_steps=None):
        """Feedback controller driving each cab to its target position and braking to rest;
        the trailer follows passively. Not exact (use iLQR shortcutting to clean up)."""
        n = self.start_pos.num_agents()
        dt = self.integration_dt
        steps = max_steps if max_steps is not None else 400
        x = np.asarray(x0, dtype=float).copy()
        target = np.asarray(x1, dtype=float)

        states = []
        for _ in range(steps):
            u = np.zeros(n * 2)
            all_reached = True
            for i in range(n):
                b = i * 5
                px, py, th, v = x[b], x[b + 1], x[b + 2], x[b + 4]
                dx, dy = target[b] - px, target[b + 1] - py
                rho = float(np.hypot(dx, dy))
                if rho > self.reach_tol or abs(v) > self.v_stop_tol:
                    all_reached = False
                if rho < self.reach_tol:
                    v_des, omega = 0.0, 0.0
                else:
                    alpha = _wrap_angle(np.arctan2(dy, dx) - th)
                    v_des = np.clip(self.k_rho * rho * np.cos(alpha), -self.v_max, self.v_max)
                    omega = np.clip(self.k_alpha * alpha, -self.omega_max, self.omega_max)
                a = np.clip(self.k_v * (v_des - v), -self.a_max, self.a_max)
                u[i * 2], u[i * 2 + 1] = a, omega
            x = x + dt * self.forward_dynamics(x, u)
            for i in range(n):
                x[i * 5 + 2] = _wrap_angle(x[i * 5 + 2])
                x[i * 5 + 3] = _wrap_angle(x[i * 5 + 3])
                x[i * 5 + 4] = np.clip(x[i * 5 + 4], -self.v_max, self.v_max)
            states.append(x.copy())
            if all_reached:
                break
        return states


def _trailer_goal_region(gx, gy, pos_tol=0.14, v_tol=0.15):
    """Box goal for a trailer: cab position near (gx, gy), any cab/trailer heading, ~rest."""
    lo = np.array([gx - pos_tol, gy - pos_tol, -4.0, -4.0, -v_tol])
    hi = np.array([gx + pos_tol, gy + pos_tol, +4.0, +4.0, +v_tol])
    return lo, hi


@register("kino.trailer_2d_swap")
class kino_trailer_2d_swap(SequenceMixin, TrailerEnvironment):
    """Two tractor-trailers swap sides through the cluttered scene AND then return to their
    starting positions, each coming to rest with the cab at its goal (trailer heading free).
    Nonholonomic + jackknife-prone; planned with the feedback controller and cleaned up with
    iLQR shortcutting."""

    def __init__(self):
        TrailerEnvironment.__init__(self)

        self.integration_dt = 0.05
        _, self.obstacles = make_kino_middle_obstacle(num_agents=2, pos_dim=2)

        lo_r = np.array([-2.0, -2.0, -np.pi, -np.pi, -self.v_max])
        hi_r = np.array([+2.0, +2.0, +np.pi, +np.pi, +self.v_max])
        self.limits = np.stack([np.concatenate([lo_r, lo_r]), np.concatenate([hi_r, hi_r])])

        # cab facing +x with trailer aligned behind (theta_t == theta); mirror for a2
        a1_start = np.array([-0.8, 0.0, 0.0, 0.0, 0.0])
        a2_start = np.array([+0.8, 0.0, np.pi, np.pi, 0.0])
        self.start_pos = NpConfiguration.from_list([a1_start, a2_start])

        self.agent_radii = [self.r_cab, self.r_cab]
        self.robot_idx = {"a1": [0, 1, 2, 3, 4], "a2": [5, 6, 7, 8, 9]}
        self.robot_dims = {"a1": 5, "a2": 5}
        self.robots = ["a1", "a2"]

        # swap goals (each trailer to the other's side) and return goals (back to own start)
        a1_swap = _trailer_goal_region(+0.8, 0.0)
        a2_swap = _trailer_goal_region(-0.8, 0.0)
        a1_home = _trailer_goal_region(-0.8, 0.0)
        a2_home = _trailer_goal_region(+0.8, 0.0)
        # terminal: both trailers back at their starting positions
        terminal_limits = np.stack(
            [np.concatenate([a1_home[0], a2_home[0]]), np.concatenate([a1_home[1], a2_home[1]])]
        )

        self.tasks = [
            Task("a1_swap", ["a1"], GoalRegion(np.stack(a1_swap))),
            Task("a2_swap", ["a2"], GoalRegion(np.stack(a2_swap))),
            Task("a1_return", ["a1"], GoalRegion(np.stack(a1_home))),
            Task("a2_return", ["a2"], GoalRegion(np.stack(a2_home))),
            Task("terminal", ["a1", "a2"], GoalRegion(terminal_limits)),
        ]
        # swap first (both trailers cross), then return to start, then terminate
        self.sequence = self._make_sequence_from_names(
            ["a2_swap", "a1_swap", "a2_return", "a1_return", "terminal"]
        )

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self.spec = ProblemSpec(
            agent_type=AgentType.MULTI_AGENT,
            constraints=ConstraintType.UNCONSTRAINED,
            manipulation=ManipulationType.STATIC,
            dependency=DependencyType.FULLY_ORDERED,
            dynamics=DynamicsType.KINODYNAMIC,
            goals=GoalType.MULTI_GOAL,
            home_pose=SafePoseType.HAS_NO_SAFE_HOME_POSE,
        )

        BaseModeLogic.__init__(self)


class RaiDoubleIntegratorEnvironment(DoubleIntegratorEnvironment):
    """Double-integrator dynamics laid *on top of* an existing rai environment, without
    modifying any rai code. Each rai robot's joint vector is treated as the position of
    an n-D point mass (extended per-robot state [q, qdot]); the rai forward-kinematics
    collision checker is reused by projecting the extended state onto its joint (position)
    part. The rai env's SingleGoal joint targets become zero-velocity kinodynamic goals.

    Requires the inner rai env to use uniform joint dimension across robots (so the shared
    [pos, vel] block layout of DoubleIntegratorEnvironment applies) and SingleGoal tasks.
    The inner rai env is used purely for collision/geometry; all mode/task logic lives here
    (SequenceMixin, extended goals)."""

    def _build_extended_state(self, inner_env_name, u_max=2.0, vel_max=2.0, integration_dt=0.05):
        """Shared setup: instantiate the inner rai env and derive the extended [q, qdot]
        state space (start_pos, limits, robot_idx/dims). Returns (d, inner)."""
        from .core.registry import get_env_by_name

        DoubleIntegratorEnvironment.__init__(self)

        inner = get_env_by_name(inner_env_name)
        self._inner = inner

        self.robots = list(inner.robots)
        dims = {inner.robot_dims[r] for r in self.robots}
        assert len(dims) == 1, "rai kino wrapper needs a uniform joint dimension per robot"
        d = dims.pop()

        self.position_dim_per_robot = d
        self.u_max = u_max
        self.integration_dt = integration_dt

        # inner joint-space slices, used to hand the rai collision checker a joint-only config
        self._inner_slices = inner.start_pos._array_slice
        # template + extended-state position indices for _joint_configuration (collision
        # hot path, called per trajectory waypoint): from_flat shares the template's slice
        # metadata and one fancy-index replaces per-robot concatenation + a full __init__
        self._joint_template = NpConfiguration(
            np.zeros(int(self._inner_slices[-1][-1])), self._inner_slices
        )
        self._joint_pos_idx = np.concatenate(
            [np.arange(i * 2 * d, i * 2 * d + d) for i in range(len(self.robots))]
        )

        state_dim = 2 * d
        n = len(self.robots)

        # extended start: [q, 0] per robot
        start_parts = []
        for i, r in enumerate(self.robots):
            q0 = np.asarray(inner.start_pos.robot_state(i), dtype=float)
            start_parts.append(np.concatenate([q0, np.zeros(d)]))
        self.start_pos = NpConfiguration.from_list(start_parts)

        self.robot_idx = {
            r: list(range(i * state_dim, (i + 1) * state_dim))
            for i, r in enumerate(self.robots)
        }
        self.robot_dims = {r: state_dim for r in self.robots}
        self.agent_radii = [0.05] * n  # only used by the fallback matplotlib viewer

        # extended limits: joint (position) block from inner, velocity block [-vel_max, vel_max]
        limits = np.zeros((2, n * state_dim))
        for i, r in enumerate(self.robots):
            s, e = self._inner_slices[i]
            base = i * state_dim
            limits[0, base : base + d] = inner.limits[0, s:e]
            limits[1, base : base + d] = inner.limits[1, s:e]
            limits[0, base + d : base + 2 * d] = -vel_max
            limits[1, base + d : base + 2 * d] = +vel_max
        self.limits = limits

        self.collision_resolution = getattr(inner, "collision_resolution", 0.01)
        self.collision_tolerance = getattr(inner, "collision_tolerance", 0.01)

        inner_manip = getattr(getattr(inner, "spec", None), "manipulation", ManipulationType.STATIC)
        self.spec = ProblemSpec(
            agent_type=AgentType.MULTI_AGENT,
            constraints=ConstraintType.UNCONSTRAINED,
            manipulation=inner_manip,
            dependency=DependencyType.FULLY_ORDERED,
            dynamics=DynamicsType.KINODYNAMIC,
            goals=GoalType.MULTI_GOAL,
            home_pose=SafePoseType.HAS_NO_SAFE_HOME_POSE,
        )
        return d, inner

    def _build_from_inner(self, inner_env_name, u_max=2.0, vel_max=2.0, integration_dt=0.05):
        """Static-env path: rebuild the inner tasks as zero-velocity extended goals and run
        our own SequenceMixin mode logic (valid because a static env has no scenegraph side
        effects at mode switches)."""
        d, inner = self._build_extended_state(inner_env_name, u_max, vel_max, integration_dt)

        self.tasks = [self._extend_task(t) for t in inner.tasks]
        seq_names = [inner.tasks[i].name for i in inner.sequence]
        self.sequence = self._make_sequence_from_names(seq_names)

        BaseModeLogic.__init__(self)

    def _extend_task(self, inner_task) -> Task:
        """[joint_goal] -> [joint_goal, zeros] per robot (zero-velocity mode switch)."""
        inner = self._inner
        goal_arr = np.asarray(inner_task.goal.goal, dtype=float)
        parts = []
        off = 0
        for r in inner_task.robots:
            d = inner.robot_dims[r]
            q_goal = goal_arr[off : off + d]
            off += d
            parts.append(np.concatenate([q_goal, np.zeros(d)]))
        return Task(inner_task.name, list(inner_task.robots), SingleGoal(np.concatenate(parts)))

    def _joint_configuration(self, q: Configuration) -> NpConfiguration:
        """Project the extended [q, qdot] config onto a joint-only config matching the
        inner rai env's joint-space slicing, so rai's setJointState gets the right length."""
        return self._joint_template.from_flat(np.asarray(q.state())[self._joint_pos_idx])

    def is_collision_free(self, q: Optional[Configuration], mode: Optional[Mode]) -> bool:
        if q is None:
            raise ValueError
        # inner env is STATIC: is_collision_free only uses q.state() for setJointState and
        # treats `mode` via a no-op set_to_mode, so passing our extended Mode is fine.
        return self._inner.is_collision_free(self._joint_configuration(q), mode)

    # --- visualization: delegate to the real rai viewer -----------------------------
    # The rai geometry lives inside the inner env's rai config, so we render the
    # kinodynamic trajectory by projecting each extended [q, qdot] state down to a joint
    # config and handing it to the inner rai viewer. The inner env is STATIC (set_to_mode
    # is a no-op), and our kino tasks are 1:1 (same order/names) with the inner tasks, so
    # passing our kino Mode straight through keeps mode-switch pausing and task labels
    # working.
    def _to_inner_path(self, path):
        from .planning_env import State as _State

        return [_State(self._joint_configuration(s.q), s.mode) for s in path]

    def show(self, blocking=True):
        self._inner.show(blocking)

    def show_config(self, q, blocking=True):
        self._inner.show_config(self._joint_configuration(q), blocking)

    def display_path(self, path, *args, **kwargs):
        self._inner.display_path(self._to_inner_path(path), *args, **kwargs)

    def display_path_viser(self, paths, *args, **kwargs):
        if paths and not isinstance(paths[0], list):
            inner_paths = self._to_inner_path(paths)
        else:
            inner_paths = [self._to_inner_path(p) for p in paths]
        self._inner.display_path_viser(inner_paths, *args, **kwargs)


@register("kino.rai_two_dim_single_goal")
class kino_rai_two_dim_single_goal(SequenceMixin, RaiDoubleIntegratorEnvironment):
    """Two 2-DOF planar rai robots (real rai collision geometry) driven as double-integrator
    point masses; single terminal goal, come to rest at the goal."""

    def __init__(self):
        self._build_from_inner("rai.two_dim_single_goal_no_rot")


@register("kino.rai_hallway")
class kino_rai_hallway(SequenceMixin, RaiDoubleIntegratorEnvironment):
    """Two 2-DOF planar rai robots in a hallway (real rai collision geometry), each with a
    short sequence of goals (multi-mode), driven as double-integrator point masses with
    zero-velocity mode switches. Reuses rai collision + tasks; optimal planning via
    kinodynamic RRT*."""

    def __init__(self):
        self._build_from_inner("rai.other_hallway_no_rot")


@register("kino.rai_random_2d")
class kino_rai_random_2d(SequenceMixin, RaiDoubleIntegratorEnvironment):
    """Three 2-DOF planar rai robots in a random-obstacle scene, each visiting several
    goals. A long-horizon (12 mode switches) stress case for the kinodynamic planner."""

    def __init__(self):
        self._build_from_inner("rai.random_2d_no_rot")


class RaiManipDoubleIntegratorEnvironment(RaiDoubleIntegratorEnvironment):
    """Kinodynamic (double-integrator over joint space) wrapper for a *manipulation* rai
    env (pick/place with scenegraph re-linking), e.g. box stacking with arms.

    Unlike the static wrapper, the mode/task logic is NOT rebuilt here: manipulation mode
    switches carry scenegraph side effects (which object is attached to which gripper) that
    only the inner rai env knows how to produce. So all mode-structure queries are delegated
    to the inner env, and its Mode objects (joint-space, with the correct scenegraph) are
    used directly. This wrapper adds only (a) the double-integrator dynamics + exact steering
    over joint space, and (b) zero-velocity extended goals for the mode-switch / done checks.
    Collision delegates to the inner env, which attaches the grasped object via set_to_mode
    before checking -- so the arm is checked against the object it is carrying."""

    def _build_manip(self, inner_env_name, u_max=2.0, vel_max=2.0, integration_dt=0.05):
        _, inner = self._build_extended_state(inner_env_name, u_max, vel_max, integration_dt)
        # keep inner tasks for task_id-indexed label lookups; mode logic is delegated
        self.tasks = inner.tasks
        self.start_mode = inner.get_start_mode()
        self.manipulating_env = False  # our env does not manipulate; the inner env does

    # --- mode / scenegraph logic: delegate to the inner rai env ---------------------
    def get_start_mode(self):
        return self.start_mode

    def is_terminal_mode(self, mode):
        return self._inner.is_terminal_mode(mode)

    def get_valid_next_task_combinations(self, mode):
        return self._inner.get_valid_next_task_combinations(mode)

    def get_scenegraph_info_for_mode(self, mode, is_start_mode=False):
        return self._inner.get_scenegraph_info_for_mode(mode, is_start_mode)

    def set_to_mode(self, mode):
        return self._inner.set_to_mode(mode)

    def get_current_seq_index(self, mode):
        return self._inner.get_current_seq_index(mode)

    def get_next_modes(self, q, mode):
        # the entry configuration of the next mode (where the grasp happens) must be the
        # joint config -- that is what the inner env uses to compute the object attachment
        return self._inner.get_next_modes(self._joint_configuration(q), mode)

    def get_active_task(self, mode, next_task_ids=None) -> Task:
        return self._extend_task(self._inner.get_active_task(mode, next_task_ids))

    # --- zero-velocity mode-switch / terminal checks --------------------------------
    def _goal_satisfied(self, q: Configuration, mode: Mode, inner_task) -> bool:
        ext = self._extend_task(inner_task)
        q_concat = np.concatenate(
            [q.robot_state(self.robots.index(r)) for r in inner_task.robots]
        )
        return ext.goal.satisfies_constraints(q_concat, mode=mode, tolerance=1e-8)

    def is_transition(self, q: Configuration, mode: Mode) -> bool:
        if self._inner.is_terminal_mode(mode):
            return False
        return self._goal_satisfied(q, mode, self._inner.get_active_task(mode, None))

    def done(self, q: Configuration, mode: Mode) -> bool:
        if not self._inner.is_terminal_mode(mode):
            return False
        return self._goal_satisfied(q, mode, self._inner.get_active_task(mode, None))


@register("kino.rai_box_stacking_one_robot")
class kino_rai_box_stacking_one_robot(RaiManipDoubleIntegratorEnvironment):
    """Single 6-DOF arm stacking boxes, driven as a double integrator over joint space with
    zero-velocity pick/place mode switches. Small env for debugging the manipulation path."""

    def __init__(self):
        self._build_manip("rai.box_stacking_one_robot")


@register("kino.rai_box_stacking_two_robots")
class kino_rai_box_stacking_two_robots(RaiManipDoubleIntegratorEnvironment):
    """Two 6-DOF arms stacking boxes, driven as double integrators over joint space.
    Mid-size manipulation test between the single-arm debug env and the full multi-arm one."""

    def __init__(self):
        self._build_manip("rai.box_stacking_two_robots")


@register("kino.rai_box_stacking")
class kino_rai_box_stacking(RaiManipDoubleIntegratorEnvironment):
    """Full multi-arm box stacking, driven as a double integrator over joint space.
    Performance / stress test for the kinodynamic planner on a manipulation problem."""

    def __init__(self):
        self._build_manip("rai.box_stacking")
