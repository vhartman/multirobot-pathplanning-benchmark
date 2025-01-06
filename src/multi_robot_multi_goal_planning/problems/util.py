import numpy as np
from typing import List

from functools import cache
from collections import deque

from multi_robot_multi_goal_planning.problems.planning_env import State


def path_cost(path: List[State], batch_cost_fun) -> float:
    batch_costs = batch_cost_fun(path[:-1], path[1:])
    return np.sum(batch_costs)


@cache
def generate_binary_search_indices(N):
    sequence = []
    queue = deque([(0, N - 1)])
    while queue:
        start, end = queue.popleft()
        if start > end:
            continue
        mid = (start + end) // 2
        sequence.append(int(mid))
        queue.append((start, mid - 1))
        queue.append((mid + 1, end))
    return sequence
