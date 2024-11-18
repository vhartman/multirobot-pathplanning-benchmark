import numpy as np
from numpy.typing import NDArray
from planning_env import *
from rai_envs import *

from typing import List

from jax import jit

def state_dist(start: State, end: State) -> float:
    # if not np.array_equal(n_start.mode, n_end.mode):
    # if np.linalg.norm(n_start.mode - n_end.mode) > 0:
    if start.mode.tolist() != end.mode.tolist():
        return np.inf

    return config_dist(start.q, end.q)


def state_cost(start: State, end: State) -> float:
    # if not np.array_equal(n_start.mode, n_end.mode):
    # if np.linalg.norm(n_start.mode - n_end.mode) > 0:
    if start.mode.tolist() != end.mode.tolist():
        return np.inf

    return config_cost(start.q, end.q)


def path_cost(path: List[State]) -> float:
    cost = 0

    batch_costs = batch_config_cost(path[:-1], path[1:])
    return np.sum(batch_costs)
    # print("A")
    # print(batch_costs)

    # costs = []

    for i in range(len(path) - 1):
        # costs.append(float(config_cost(path[i].q, path[i + 1].q)))
        cost += config_cost(path[i].q, path[i + 1].q)

    # print(costs)

    return cost