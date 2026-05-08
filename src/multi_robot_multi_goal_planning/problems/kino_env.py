"""
Kinodynamic environments built on AbstractEnvironment's geometric collision backend.

Class hierarchy:
  KinodynamicPlanningProblem  (abstract dynamics interface, in kinodynamic_planning_env.py)
   └─ DoubleIntegratorEnvironment   (concrete second-order dynamics + sphere/rect collision
                                     via position projection, per-robot layout [pos, vel])
        └─ kino.point_mass_2d_swap  (concrete scenario)
"""

import numpy as np

from typing import List, Optional
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
from .core.goals import SingleGoal
from .core.registry import register

from .abstract_env import AbstractEnvironment, Sphere, Rectangle
from .kinodynamic_planning_env import KinodynamicPlanningProblem


class DoubleIntegratorEnvironment(KinodynamicPlanningProblem, AbstractEnvironment):
    """Multi-robot point-mass env with second-order dynamics ẋ = [vel; u]
    and per-robot extended-state layout [pos..., vel...]. Reuses the
    AbstractEnvironment sphere/rectangle collision check via position projection."""

    position_dim_per_robot: int
    u_max: float

    def __init__(self):
        AbstractEnvironment.__init__(self)

    def forward_dynamics(self, x: NDArray, u: NDArray) -> NDArray:
        n = self.start_pos.num_agents()
        d = self.position_dim_per_robot
        x_dot = np.zeros_like(x)
        for i in range(n):
            base = i * 2 * d
            x_dot[base : base + d] = x[base + d : base + 2 * d]
            x_dot[base + d : base + 2 * d] = u[i * d : (i + 1) * d]
        return x_dot

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

    def _position_configuration(self, q: Configuration) -> NpConfiguration:
        per_robot = [self.robot_position(q, i) for i in range(q.num_agents())]
        return NpConfiguration.from_list(per_robot)

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


def make_kino_middle_obstacle_2d():
    num_agents = 2
    pos_dim = 2
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

    obstacles = [
        Sphere(np.zeros(pos_dim), 0.2),
        Rectangle(np.array([0.0, 0.4]), np.array([0.5, 0.5])),
    ]
    return limits, obstacles


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
