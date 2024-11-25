import numpy as np
from planning_env import *
from rai_envs import *

from typing import List

# TODO:
# add cost/distance to the envs


def path_cost(path: List[State], batch_cost_fun) -> float:
    batch_costs = batch_cost_fun(path[:-1], path[1:])
    return np.sum(batch_costs)