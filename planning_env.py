import robotic as ry
import numpy as np

import time
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import random

from typing import List
from numpy.typing import NDArray

# from dependency_graph import DependencyGraph

# questions:
# - what to do with final mode? goal?
# - how to sample? in env? how to do goal sampling?
# - mode represenatition needs to be better


class Goal(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def satisfies_constraints(self, q):
        pass

    @abstractmethod
    def sample(self):
        pass


class GoalRegion(Goal):
    def __init__(self):
        pass

    def satisfies_constraints(self, q, tolerance):
        pass

    def sample(self):
        pass


class GoalSet(Goal):
    def __init__(self, goals):
        self.goals = goals

    def satisfies_constraints(self, q: NDArray, tolerance: float) -> bool:
        for g in self.goals:
            if np.linalg.norm(g - q) < tolerance:
                return True

        return False

    def sample(self) -> NDArray:
        rnd = np.random.randint(0, len(self.goals))
        return self.goals[rnd]


class SingleGoal(Goal):
    def __init__(self, goal: NDArray):
        self.goal = goal

    def satisfies_constraints(self, q: ry.Config, tolerance: float) -> bool:
        if np.linalg.norm(self.goal - q) < tolerance:
            return True

        return False

    def sample(self) -> NDArray:
        return self.goal


def make_mode_sequence_from_sequence(robots: List, sequence: List) -> List:
    # compute initial mode
    initial_mode = {}

    for s in sequence:
        r_str = s[0]
        robot_mode = s[1]

        if r_str not in initial_mode:
            initial_mode[r_str] = robot_mode

    initial_mode_list = [0 for _ in range(len(robots))]
    for k, v in initial_mode.items():
        r_idx = robots.index(k)
        mode = v

        initial_mode_list[r_idx] = mode

    mode_sequence = [initial_mode_list]

    for i, s in enumerate(sequence):
        r_str = s[0]
        r_idx = robots.index(r_str)
        robot_mode = s[1]

        # find next mode for robot
        next_robot_mode = robot_mode + 1

        if next_robot_mode is not None:
            new_mode = mode_sequence[-1].copy()
            new_mode[r_idx] = next_robot_mode

        mode_sequence.append(new_mode)

    return mode_sequence


class base_env(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def done(self, q, m):
        pass

    def get_robot_dim(self, r):
        return self.robot_dims[r]

    def get_start_pos(self):
        return self.start_pos

    def get_start_mode(self):
        return self.start_mode

    def get_sequence(self):
        return self.sequence

    def get_robot_sequence(self, r):
        pass

    def set_to_mode(self, m):
        pass

    @abstractmethod
    def get_next_mode(self, q, m):
        pass

    @abstractmethod
    def is_collision_free(self, q, m):
        pass

    @abstractmethod
    def is_edge_collision_free(self, q1, q2, m, resolution=5):
        pass

    @abstractmethod
    def is_path_collision_free(self, path):
        pass

    @abstractmethod
    def is_valid_plan(self, path):
        pass
