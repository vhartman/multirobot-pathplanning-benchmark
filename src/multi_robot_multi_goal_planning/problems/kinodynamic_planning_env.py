"""
Abstract base class for kinodynamic planning problems.

Extends BaseProblem with the dynamics interface that any kinodynamic environment
must implement: forward dynamics, state and control bounds, and a position-projection
hook so geometric collision checkers can be reused without committing to a state-layout
convention.

Default behavior:
  - propagate(): forward Euler over forward_dynamics. Override for RK4 etc.
  - is_trajectory_collision_free(): walks integrated states, calls is_collision_free.
  - is_edge_collision_free(): raises — straight-line C-space interpolation is
    meaningless under dynamics; the planner must propagate.

Concrete envs are expected to additionally provide a collision backend (e.g. by
mixing in AbstractEnvironment, a Rai/Pinocchio backend, etc.) so that
is_collision_free is wired up.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .planning_env import BaseProblem, Mode
from .core.configuration import Configuration


class KinodynamicPlanningProblem(BaseProblem, ABC):
    integration_dt: float = 0.05

    @abstractmethod
    def forward_dynamics(self, x: NDArray, u: NDArray) -> NDArray:
        """Continuous-time dynamics ẋ for the full multi-robot flat state."""
        ...

    @abstractmethod
    def state_bounds(self) -> NDArray:
        """(2, total_state_dim) lower/upper bounds on the extended state."""
        ...

    @abstractmethod
    def control_bounds(self) -> NDArray:
        """(2, total_control_dim) lower/upper bounds on the control."""
        ...

    @abstractmethod
    def robot_position(self, q: Configuration, robot_index: int) -> NDArray:
        """Position-only slice for the given robot. Lets the geometric collision
        check operate on (q, q̇) without committing to a state-layout convention."""
        ...

    def propagate(
        self,
        x: NDArray,
        u: NDArray,
        dt: Optional[float] = None,
        num_steps: int = 1,
    ) -> List[NDArray]:
        """Forward-Euler integrate x under constant control u for num_steps of length dt.
        Returns the list of intermediate states (excluding x_0, including x_N)."""
        if dt is None:
            dt = self.integration_dt
        traj: List[NDArray] = []
        x_curr = x.copy()
        for _ in range(num_steps):
            x_curr = x_curr + dt * self.forward_dynamics(x_curr, u)
            traj.append(x_curr)
        return traj

    def is_trajectory_collision_free(
        self, traj: List[NDArray], mode: Mode
    ) -> bool:
        """Check collision along a propagated trajectory of full extended states."""
        for x in traj:
            q = self.start_pos.from_flat(x)
            if not self.is_collision_free(q, mode):
                return False
        return True

    def is_edge_collision_free(self, *args, **kwargs) -> bool:
        raise NotImplementedError(
            "Geometric edge collision check is not meaningful in a kinodynamic problem. "
            "The kinodynamic planner must propagate dynamics and call is_collision_free "
            "at each integration step."
        )
