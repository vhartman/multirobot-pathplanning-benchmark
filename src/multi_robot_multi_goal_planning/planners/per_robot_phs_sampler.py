"""
Per-robot PHS-biased sampler — experimental sketch.

For each robot j in mode m, sample j's coordinates from a prolate hyperspheroid
in j's d_j-dim subspace. PHS foci are j's *previous forced goal* and *next
forced goal* on the task chain, found by walking the mode graph from m.

The cost bound c_j is supplied as a callable `(c_min, iteration) -> c_j` so the
caller can grow it over time. For asymptotic optimality, c_j should be
unbounded in the iteration count: as c_j -> inf the PHS covers the whole
robot subspace, recovering uniform sampling.

!!! BIG TODO !!!
The forward walk over `mode.next_modes` ignores branching: when len > 1 and
different branches assign different next tasks to robot j, this implementation
just picks one branch uniformly at sample time. That is WRONG in the general
case — must be revisited before using on real branching plans. Possible fixes:
  - take the union of per-branch PHSs (mixture sampler weighted by branch)
  - precompute per (mode, robot) the set of reachable next goals and sample
    from a mixture
The same caveat applies to robots with no remaining forced goal: currently we
fall back to uniform-in-limits.

!!! TODO (future) !!!
Cache prev/next forced goals per (mode, robot) so the graph walks happen once
during mode construction rather than every sample. Wire the live walks first;
only cache once the design is settled.

See memory/project_per_robot_ellipsoid_sampling.md for the full design notes.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.core.configuration import Configuration
from multi_robot_multi_goal_planning.problems.planning_env import BaseProblem, Mode

from .sampling_informed import compute_PHS_matrices, sample_phs_with_given_matrices


# (c_min_j, iter) -> c_j. Must satisfy c_j >= c_min_j; should be unbounded in
# `iter` for asymptotic optimality (sampling distribution -> uniform).
CostBoundFn = Callable[[float, int], float]


def default_cost_bound_fn(c_min: float, it: int) -> float:
    """Placeholder schedule — TUNE. Linear additive growth keeps the corridor
    widening even when c_min is near 0; multiplicative term ensures
    non-degenerate ellipsoid from the start."""
    return c_min * 1.5 + 0.01 * it


class PerRobotPHSSampler:
    def __init__(
        self,
        env: BaseProblem,
        c_bound_fn: CostBoundFn = default_cost_bound_fn,
    ):
        self.env = env
        self.c_bound_fn = c_bound_fn
        self._iter = 0  # caller can also drive this externally via tick()

    def tick(self) -> None:
        self._iter += 1

    def selected_sampler_name(self, mode: Optional[Mode] = None) -> str:
        # exposed for compatibility with collision_free_sampler.BaseCollisionFreeSampler
        # so planner dispatch can identify this sampler.
        return "per_robot_phs"

    def sample(
        self,
        mode: Mode,
        pinned: Optional[Dict[str, NDArray]] = None,
        max_attempts: int = 100,
    ) -> Configuration | None:
        """Returns a collision-free configuration, or None after `max_attempts`."""
        pinned = pinned or {}
        for _ in range(max_attempts):
            self._iter += 1
            q = self._draw(mode, pinned)
            if self.env.is_collision_free(q, mode):
                return q
        return None

    # ------------------------------------------------------------------
    # candidate proposal

    def _draw(self, mode: Mode, pinned: Dict[str, NDArray]) -> Configuration:
        # uniform fallback for any robot we can't bias (no forced goal,
        # degenerate foci, out-of-limits PHS draw, etc.)
        q = self.env.sample_config_uniform_in_limits()
        for robot, values in pinned.items():
            i = self.env.robots.index(robot)
            q[i] = values

        for i, robot in enumerate(self.env.robots):
            if robot in pinned:
                continue
            lims = self.env.limits[:, self.env.robot_idx[robot]]

            prev_goal = self._prev_goal(mode, robot, i)
            next_goal = self._next_goal(mode, robot, i)

            if next_goal is None:
                # no remaining forced goal — keep uniform fallback
                continue

            c_min = float(np.linalg.norm(next_goal - prev_goal))
            c = self.c_bound_fn(c_min, self._iter)

            if c_min < 1e-9:
                # foci coincide; PHS is undefined. For now stay at the goal.
                # (Could instead sample a small ball of radius (c)/2 here.)
                q[i] = prev_goal
                continue

            if c <= c_min:
                # bound too tight — pin to the line midpoint as safe fallback
                q[i] = 0.5 * (prev_goal + next_goal)
                continue

            rot, center = compute_PHS_matrices(prev_goal, next_goal, c)
            qr = sample_phs_with_given_matrices(rot, center, n=1)[:, 0]

            # PHS draws can fall outside robot limits. Cheap fallback: re-draw
            # uniformly. (Better: per-robot rejection inside the PHS, but keep
            # the sketch simple.)
            if np.any(qr < lims[0]) or np.any(qr > lims[1]):
                qr = np.random.uniform(lims[0], lims[1])
            q[i] = qr

        return q

    # ------------------------------------------------------------------
    # mode-graph walks (no caching — see TODO at top)

    def _prev_goal(self, mode: Mode, robot: str, robot_idx: int) -> NDArray:
        """Walk back via prev_mode. Return the goal of the most recent
        transition where robot j's task changed. Falls back to start config."""
        m = mode
        while m.prev_mode is not None:
            prev = m.prev_mode
            if prev.task_ids[robot_idx] != m.task_ids[robot_idx]:
                # robot j's task ended at the prev -> m transition
                task = self.env.tasks[prev.task_ids[robot_idx]]
                goal = task.goal.sample(prev)
                return self._slice_for_robot(goal, task, robot)
            m = prev
        return self.env.get_start_pos()[robot_idx].copy()

    def _next_goal(
        self, mode: Mode, robot: str, robot_idx: int
    ) -> Optional[NDArray]:
        """Walk forward via next_modes. Return the goal of the next transition
        where robot j's task changes. Returns None if no future task does so.

        !!! Branching is ignored — picks one next_mode uniformly. !!!
        """
        m = mode
        guard = 0
        while m.next_modes and guard < 1024:
            guard += 1
            nxt = random.choice(m.next_modes)
            if m.task_ids[robot_idx] != nxt.task_ids[robot_idx]:
                task = self.env.tasks[m.task_ids[robot_idx]]
                goal = task.goal.sample(m)
                return self._slice_for_robot(goal, task, robot)
            m = nxt
        return None

    def _slice_for_robot(self, goal: NDArray, task, robot: str) -> NDArray:
        """Tasks may pack multiple robots' goals into one flat array, ordered
        by task.robots. Pick out our robot's slice."""
        d_j = self.env.robot_dims[robot]
        if len(goal) == d_j:
            return goal
        offset = 0
        for r in task.robots:
            dim = self.env.robot_dims[r]
            if r == robot:
                return goal[offset : offset + dim]
            offset += dim
        # robot not in task.robots — shouldn't happen if caller invariant holds
        raise ValueError(f"Robot {robot} not in task.robots {task.robots}")
