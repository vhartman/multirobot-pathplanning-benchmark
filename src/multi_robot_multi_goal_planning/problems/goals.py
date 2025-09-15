import numpy as np

from typing import List, Dict, Optional, Any, Tuple
from numpy.typing import NDArray

from abc import ABC, abstractmethod


class Goal(ABC):
    def __init__(self):
        pass

    # TODO: this should be a coficturaiotn not an ndarray
    @abstractmethod
    def satisfies_constraints(self, q: NDArray, mode: "Mode", tolerance: float) -> bool:
        pass

    @abstractmethod
    def sample(self, mode: "Mode") -> NDArray:
        pass

    @abstractmethod
    def serialize(self) -> List:
        pass

    @classmethod
    @abstractmethod
    def from_data(cls, data):
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

    def satisfies_constraints(self, q: NDArray, mode: "Mode", tolerance: float) -> bool:
        if np.all(q > self.limits[0, :]) and np.all(q < self.limits[1, :]):
            return True

        return False

    def sample(self, mode: "Mode") -> NDArray:
        q = (
            np.random.rand(len(self.limits[0, :]))
            * (self.limits[1, :] - self.limits[0, :])
            + self.limits[0, :]
        )
        return q

    def serialize(self) -> List:
        return self.limits.tolist()

    @classmethod
    def from_data(cls, data):
        return GoalRegion(np.array(data))


# TODO: implement sampler to sample a goal
class ConstrainedGoal(Goal):
    pass


class ConditionalGoal(Goal):
    def __init__(self, conditions, goals):
        self.conditions = conditions
        self.goals = goals

    def satisfies_constraints(self, q: NDArray, mode: "Mode", tolerance: float) -> bool:
        for c, g in zip(self.conditions, self.goals):
            if (
                np.linalg.norm(mode.entry_configuration[0] - c) < tolerance
                and np.linalg.norm(g - q) < tolerance
            ):
                return True

        return False

    def sample(self, mode: "Mode") -> NDArray:
        for c, g in zip(self.conditions, self.goals):
            if np.linalg.norm(mode.entry_configuration[0] - c) < 1e-8:
                return g

        raise ValueError("No feasible goal in mode")

    def serialize(self) -> List:
        print("This is not yet implemented")
        raise NotImplementedError

    @classmethod
    def from_data(cls, data):
        print("This is not yet implemented")
        raise NotImplementedError


class GoalSet(Goal):
    def __init__(self, goals):
        self.goals = goals

    def satisfies_constraints(self, q: NDArray, mode: "Mode", tolerance: float) -> bool:
        for g in self.goals:
            if np.linalg.norm(g - q) < tolerance:
                return True

        return False

    def sample(self, mode: "Mode") -> NDArray:
        rnd = np.random.randint(0, len(self.goals))
        return self.goals[rnd]

    def serialize(self) -> List:
        return [goal.tolist() for goal in self.goals]

    @classmethod
    def from_data(cls, data):
        return GoalSet([np.array(goal) for goal in data])


class SingleGoal(Goal):
    def __init__(self, goal: NDArray):
        self.goal = goal

    def satisfies_constraints(self, q, mode: "Mode", tolerance: float) -> bool:
        if np.linalg.norm(self.goal - q) < tolerance:
            return True

        return False

    def sample(self, mode: "Mode") -> NDArray:
        return self.goal

    def serialize(self) -> List:
        return self.goal.tolist()

    @classmethod
    def from_data(cls, data):
        return SingleGoal(np.array(data))