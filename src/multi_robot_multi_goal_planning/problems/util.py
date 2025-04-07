import numpy as np
from typing import List

from functools import cache
from collections import deque

from multi_robot_multi_goal_planning.problems.planning_env import State
from multi_robot_multi_goal_planning.problems.configuration import config_dist


def path_cost(path: List[State], batch_cost_fun) -> float:
    batch_costs = batch_cost_fun(path[:-1], path[1:])
    return np.sum(batch_costs)


def interpolate_path(path: List[State], resolution: float = 0.1):
    config_type = type(path[0].q)
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
            # for k in range(q0.num_agents()):
            #     qr = q0.robot_state(k) + (q1.robot_state(k) - q0.robot_state(k)) / N * j
            #     q.append(qr)

                # env.C.setJointState(qr, get_robot_joints(env.C, env.robots[k]))

                # env.C.setJointState(qr, [env.robots[k]])

            # env.C.view(True)

            new_path.append(State(config_type(q, q0.array_slice), path[i].mode))

    new_path.append(State(path[-1].q, path[-1].mode))

    return new_path

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
    return tuple(sequence)
@cache
def generate_binary_search_indices_wo_start_and_end(N):
    sequence = []
    queue = deque([(0, N - 1)])
    while queue:
        start, end = queue.popleft()
        if start > end:
            continue
        mid = (start + end) // 2
        queue.append((start, mid - 1))
        queue.append((mid + 1, end))
        if mid == 0 or mid == N - 1:
            continue
        sequence.append(int(mid))
    return tuple(sequence)