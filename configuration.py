import numpy as np
import jax
import jax.numpy as jnp

from typing import List
from numpy.typing import NDArray

from abc import ABC, abstractmethod


class Configuration(ABC):
    @abstractmethod
    def num_agents(self) -> int:
      pass
    
    @abstractmethod
    def robot_state(self, ind: int) -> NDArray:
        pass

    @abstractmethod
    def state(self) -> NDArray:
        pass

    @abstractmethod
    def from_list(cls, q_list: List[NDArray]):
        pass


class ListConfiguration(Configuration):
    def __init__(self, q_list):
        self.q = q_list

    def __getitem__(self, ind):
        return self.robot_state(ind)

    def __setitem__(self, ind, data):
        self.q[ind] = data

    @classmethod
    def from_list(cls, q_list: List[NDArray]) -> "ListConfiguration":
        return cls(q_list)

    def robot_state(self, ind: int) -> NDArray:
        return self.q[ind]

    def state(self) -> NDArray:
        return np.concatenate(self.q)

    def num_agents(self):
      return len(self.q)


class NpConfiguration(Configuration):
    def __init__(self, q: NDArray, slice: List[int]):
        self.slice = slice
        self.q = q
    
    def num_agents(self):
        return len(self.slice)

    def __getitem__(self, ind):
        return self.robot_state(ind)

    def __setitem__(self, ind, data):
        self.q[self.slice[ind][0]: self.slice[ind][1]] = data

    @classmethod
    def from_list(cls, q_list: List[NDArray]) -> "NpConfiguration":
        slices = []
        s = 0
        for i in range(len(q_list)):
          slices.append((s, s+len(q_list[i])))
          s += len(q_list[i])
            
        return cls(np.concatenate(q_list), slices)

    def robot_state(self, ind: int) -> NDArray:
        return self.q[self.slice[ind][0]: self.slice[ind][1]]

    def state(self) -> NDArray:
        return self.q


def config_dist(
    q_start: Configuration, q_end: Configuration, metric: str = "euclidean"
) -> float:
    num_agents = q_start.num_agents()
    dists = np.zeros(num_agents)

    for robot_index in range(num_agents):
        # print(robot_index)
        # print(q_start)
        # print(q_end)
        # d = np.linalg.norm(q_start[robot_index] - q_end[robot_index])
        diff = q_start.robot_state(robot_index) - q_end.robot_state(robot_index)
        d = 0
        if metric == "euclidean":
            for j in range(len(diff)):
                d += (diff[j]) ** 2
            dists[robot_index] = d**0.5
        else:
            dists[robot_index] = np.max(np.abs(diff))

    return np.max(dists)


def config_cost(
    q_start: Configuration, q_end: Configuration, metric: str = "euclidean"
) -> float:
    num_agents = q_start.num_agents()
    dists = np.zeros(num_agents)

    for robot_index in range(num_agents):
        # print(robot_index)
        # print(q_start)
        # print(q_end)
        # d = np.linalg.norm(q_start[robot_index] - q_end[robot_index])
        diff = q_start.robot_state(robot_index) - q_end.robot_state(robot_index)
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
