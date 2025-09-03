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

    # discretize path
    for i in range(len(path) - 1):
        q0 = path[i].q
        q1 = path[i + 1].q

        # if path[i].mode != path[i + 1].mode:
        #     new_path.append(State(config_type.from_list(q), path[i].mode))
        #     continue

        dist = config_dist(q0, q1, "euclidean")
        N = int(dist / resolution)
        N = max(1, N)

        q0_state = q0.state()
        q1_state = q1.state()
        dir = (q1_state - q0_state) / N

        for j in range(N):
            q = q0_state + dir * j
            new_path.append(State(q0.from_flat(q), path[i].mode))

    # add the final state (which is not added in the interpolation before)
    new_path.append(State(path[-1].q, path[-1].mode))

    return new_path