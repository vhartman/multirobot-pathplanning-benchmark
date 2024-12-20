import robotic as ry
import numpy as np
import random

from abc import ABC, abstractmethod

from typing import List, Dict, Optional
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
    config_dist,
)
from multi_robot_multi_goal_planning.problems.dependency_graph import DependencyGraph


class Goal(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def satisfies_constraints(self, q: NDArray, tolerance: float):
        pass

    @abstractmethod
    def sample(self):
        pass


# class DummyGoal(ABC):
#     def __init__(self):
#         pass

#     def satisfies_constraints(self, q, tolerance):
#         return True

#     def sample(self):
#         pass


class GoalRegion(Goal):
    def __init__(self, limits: NDArray):
        self.limits = limits

    def satisfies_constraints(self, q: NDArray, _):
        if np.all(q > self.limits[0, :]) and np.all(q < self.limits[1, :]):
            return True

    def sample(self):
        q = (
            np.random.rand(len(self.limits[0, :]))
            * (self.limits[1, :] - self.limits[0, :])
            + self.limits[0, :]
        )
        return q


# TODO: implement sampler to sample a goal
class ConstrainedGoal(Goal):
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


class Task:
    name: str
    robots: List[str]
    goal: Goal

    # things for manipulation
    type: str
    frames: List[str]
    side_effect: str

    # things for the future:
    constraints = List

    def __init__(self, robots: List[str], goal:NDArray, type=None, frames=None, side_effect=None):
        self.robots = robots
        self.goal = goal

        # constraints
        self.type = type
        self.frames = frames
        self.side_effect = side_effect


class State:
    q: Configuration
    m: List[int]

    def __init__(self, q: Configuration, m: List[int]):
        self.q = q
        self.mode = m


def state_dist(start: State, end: State) -> float:
    if start.mode != end.mode:
        return np.inf

    return config_dist(start.q, end.q)


# concrete implementations of the required abstract classes for the sequence-setting.
class SequenceMixin:
    def _make_sequence_from_names(self, names: List[str]) -> List[int]:
        sequence = []

        for name in names:
            no_task_with_name_found = True
            for idx, task in enumerate(self.tasks):
                if name == task.name:
                    sequence.append(idx)
                    no_task_with_name_found = False

            if no_task_with_name_found:
                raise ValueError(f"Task with name {name} not found.")

        return sequence

    def _make_start_mode_from_sequence(self) -> List[int]:
        mode_dict = {}

        for task_index in self.sequence:
            task_robots = self.tasks[task_index].robots

            for r in task_robots:
                if r not in mode_dict:
                    mode_dict[r] = task_index

        mode = []
        for r in self.robots:
            mode.append(mode_dict[r])

        return mode

    def _make_terminal_mode_from_sequence(self) -> List[int]:
        mode_dict = {}

        for task_index in self.sequence:
            task_robots = self.tasks[task_index].robots

            # difference to above: we do not check if the robot already has a task assigned
            for r in task_robots:
                mode_dict[r] = task_index

        mode = []
        for r in self.robots:
            mode.append(mode_dict[r])

        return mode

    def get_current_seq_index(self, mode: List[int]) -> int:
        # Approach: iterate through all indices, find them in the sequence, and check which is the one
        # that has to be fulfilled first
        min_sequence_pos = len(self.sequence) - 1
        for i, m in enumerate(mode):
            # print("robots in task:", self.tasks[m].robots, self.sequence.index(m))
            if m != self.terminal_mode[i]:
                min_sequence_pos = min(self.sequence.index(m), min_sequence_pos)

        return min_sequence_pos

    # TODO: is that really a good way to sample a mode?
    def sample_random_mode(self) -> List[int]:
        m = self.start_mode
        rnd = random.randint(0, len(self.sequence))

        for _ in range(rnd):
            m = self.get_next_mode(None, m)

        return m

    def get_sequence(self):
        return self.sequence

    def get_robot_sequence(self, robot: str):
        pass

    def get_next_mode(self, q: Configuration, mode: List[int]):
        pass


class DependencyGraphMixin(ABC):
    def get_next_mode(self, q, mode):
        pass


# TODO: split into env + problem specification
class base_env(ABC):
    robots: List[str]
    robot_dims: Dict[str, int]
    robot_idx: Dict[str, NDArray]
    start_pos: Configuration

    start_mode: List[int]
    terminal_mode: List[int]

    # visualization
    @abstractmethod
    def show_config(self, q: Configuration):
        pass

    @abstractmethod
    def show(self):
        pass

    ## General methods
    def get_start_pos(self):
        return self.start_pos

    def get_start_mode(self):
        return self.start_mode

    def get_robot_dim(self, robot: str):
        return self.robot_dims[robot]

    def get_all_bounds(self):
        self.bounds

    # def get_robot_bounds(self, robot):
    #     self.bounds

    # Task sequencing methods
    @abstractmethod
    def set_to_mode(self, mode: List[int]):
        pass

    @abstractmethod
    def sample_random_mode(self) -> List[int]:
        pass

    @abstractmethod
    def done(self, q: Configuration, mode: List[int]):
        pass

    @abstractmethod
    def is_transition(self, q: Configuration, m: List[int]) -> bool:
        pass

    @abstractmethod
    def get_next_mode(self, q: Configuration, mode: List[int]):
        pass

    @abstractmethod
    def get_active_task(self, mode: List[int]) -> Task:
        pass

    @abstractmethod
    def get_tasks_for_mode(self, mode: List[int]) -> List[Task]:
        pass

    # Collision checking and environment related methods
    @abstractmethod
    def is_collision_free(self, q: Optional[Configuration], mode: List[int]) -> bool:
        pass

    def is_collision_free_for_robot(
        self, r: str, q, m: List[int], collision_tolerance: float = 0.01
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_edge_collision_free(
        self,
        q1: Configuration,
        q2: Configuration,
        mode: List[int],
        resolution: float = 0.1,
    ) -> bool:
        pass

    @abstractmethod
    def is_path_collision_free(self, path: List[State]) -> bool:
        pass

    def is_valid_plan(self, path: List[State]) -> bool:
        # check if it is collision free and if all modes are passed in order
        # only take the configuration into account for that
        mode = self.start_mode
        collision = False
        for i in range(len(path)):
            # check if the state is collision free
            if not self.is_collision_free(path[i].q.state(), mode):
                print(f"There is a collision at index {i}")
                # col = self.C.getCollisionsTotalPenetration()
                # print(col)
                self.show()
                collision = True

            # if the next mode is a transition, check where to go
            if i < len(path) - 1 and self.is_transition(path[i].q, mode):
                # TODO: this does not work if multiple switches are possible at the same time
                next_mode = self.get_next_mode(path[i].q, mode)

                if path[i + 1].mode == next_mode:
                    mode = next_mode

        if not self.done(path[-1].q, path[-1].mode):
            print("Final mode not reached")
            return False

        if collision:
            print("There was a collision")
            return False

        return True

    @abstractmethod
    def config_cost(self, start: Configuration, goal: Configuration) -> float:
        pass

    @abstractmethod
    def batch_config_cost(
        self,
        starts: List[Configuration],
        ends: List[Configuration],
    ) -> List[float]:
        pass

    def state_cost(self, start: State, end: State) -> float:
        if start.mode != end.mode:
            return np.inf

        return self.config_cost(start.q, end.q)
