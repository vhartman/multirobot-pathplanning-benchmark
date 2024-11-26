import numpy as np
from typing import List

from planning_env import State


def path_cost(path: List[State], batch_cost_fun) -> float:
    batch_costs = batch_cost_fun(path[:-1], path[1:])
    return np.sum(batch_costs)
