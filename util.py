import numpy as np
from numpy.typing import NDArray
from planning_env import *
from rai_envs import *

from typing import List

from jax import jit


def config_dist(
    q_start: List[NDArray], q_end: List[NDArray], metric: str = "euclidean"
) -> float:
    num_agents = len(q_start)
    dists = np.zeros(num_agents)

    for robot_index in range(num_agents):
        # print(robot_index)
        # print(q_start)
        # print(q_end)
        # d = np.linalg.norm(q_start[robot_index] - q_end[robot_index])
        diff = q_start[robot_index] - q_end[robot_index]
        d = 0
        if metric == "euclidean":
            for j in range(len(diff)):
                d += (diff[j]) ** 2
            dists[robot_index] = d**0.5
        else:
            dists[robot_index] = np.max(np.abs(diff))

    return np.max(dists)


def state_dist(start: State, end: State) -> float:
    # if not np.array_equal(n_start.mode, n_end.mode):
    # if np.linalg.norm(n_start.mode - n_end.mode) > 0:
    if start.mode.tolist() != end.mode.tolist():
        return np.inf

    return config_dist(start.q, end.q)

def config_cost(
    q_start: List[NDArray], q_end: List[NDArray], metric: str = "euclidean"
) -> float:
    num_agents = len(q_start)
    dists = np.zeros(num_agents)

    for robot_index in range(num_agents):
        # print(robot_index)
        # print(q_start)
        # print(q_end)
        # d = np.linalg.norm(q_start[robot_index] - q_end[robot_index])
        diff = q_start[robot_index] - q_end[robot_index]
        if metric == "euclidean":
            d = 0
            for j in range(len(diff)):
                d += (diff[j]) ** 2
            dists[robot_index] = d**0.5
        else:
            dists[robot_index] = np.max(np.abs(diff))

    # dists = np.linalg.norm(np.array(q_start) - np.array(q_end), axis=1)
    return max(dists) + 0.01 * sum(dists)
    # return np.sum(dists)


def state_cost(start: State, end: State) -> float:
    # if not np.array_equal(n_start.mode, n_end.mode):
    # if np.linalg.norm(n_start.mode - n_end.mode) > 0:
    if start.mode.tolist() != end.mode.tolist():
        return np.inf

    return config_cost(start.q, end.q)


def path_cost(path: List[State]) -> float:
    cost = 0

    for i in range(len(path) - 1):
        cost += config_cost(path[i].q, path[i + 1].q)

    return cost
