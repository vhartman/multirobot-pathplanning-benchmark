import numpy as np
from typing import List

from .planning_env import State
from .configuration import config_dist


def path_cost(path: List[State], batch_cost_fun, agent_slices=None) -> float:
    """
    Computes the path cost via the batch cost function and summing it up.
    """
    if isinstance(path[0], State):
        pts = [start.q.state() for start in path]
        agent_slices = path[0].q._array_slice
        batch_costs = batch_cost_fun(pts, None, tmp_agent_slice=agent_slices)
    elif isinstance(path[0], np.ndarray) and agent_slices is not None:
        batch_costs = batch_cost_fun(path, None, tmp_agent_slice=agent_slices)
    else:
        raise ValueError("Arguments to path cost seem to be wrong.")
        
    # batch_costs = batch_cost_fun(path, None)
    # assert np.allclose(batch_costs, batch_costs_tmp)

    return np.sum(batch_costs)


def interpolate_path(path: List[State], resolution: float = 0.1) -> List[State]:
    """
    Takes a path and interpolates it at the given resolution.
    Uses the euclidean distance between states to do the resolution.
    """
    new_path = []

    # Discretize path
    for i in range(len(path) - 1):
        s_curr = path[i]
        s_next = path[i+1]

        # Check for skill flag
        is_skill = getattr(s_curr, 'is_skill_waypoint', False)

        if is_skill:
            # Do NOT create new State object, append exact original point
            new_path.append(s_curr)
        else:
            # Standard free space interpolation
            dist = config_dist(s_curr.q, s_next.q, "euclidean")
            N = int(dist / resolution)
            N = max(1, N)

            q0_state = s_curr.q.state()
            q1_state = s_next.q.state()
            dir = (q1_state - q0_state) / N

            for j in range(N):
                q = q0_state + dir * j
                new_path.append(State(s_curr.q.from_flat(q), s_curr.mode))

    # Add the final state (which is not added in the interpolation before) while preserving the object
    new_path.append(path[-1])

    return new_path