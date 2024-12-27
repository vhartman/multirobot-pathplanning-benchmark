import robotic as ry
import numpy as np
import random
import time

from typing import List, Dict, Optional
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.dependency_graph import DependencyGraph

from multi_robot_multi_goal_planning.problems.rai_config import *
from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    Task,
    SingleGoal,
    GoalSet,
    GoalRegion,
)
from multi_robot_multi_goal_planning.problems.rai_base_env import rai_env


class rai_two_dim_env(rai_env):
    def __init__(self, agents_can_rotate=True):
        self.C, keyframes = make_2d_rai_env(agents_can_rotate=agents_can_rotate)
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        super().__init__()

        self.tasks = [
            Task(
                ["a1", "a2"],
                SingleGoal(
                    np.concatenate(
                        [
                            keyframes[0][self.robot_idx["a1"]],
                            keyframes[2][self.robot_idx["a2"]],
                        ]
                    )
                ),
            ),
        ]

        self.tasks[0].name = "terminal"

        self.sequence = self._make_sequence_from_names(["terminal"])

        self.start_mode = self._make_start_mode_from_sequence()
        self.terminal_mode = self._make_terminal_mode_from_sequence()

        self.tolerance = 0.1


class rai_random_two_dim(rai_env):
    def __init__(self, num_robots=3, agents_can_rotate=False):
        self.C, keyframes = make_random_two_dim_single_goal(
            num_agents=num_robots,
            num_obstacles=10,
            agents_can_rotate=agents_can_rotate,
        )
        # self.C.view(True)

        self.robots = [f"a{i}" for i in range(num_robots)]

        super().__init__()

        self.tasks = []
        self.tasks.append(Task(self.robots, SingleGoal(keyframes[0])))
        self.tasks[-1].name = "terminal"

        self.sequence = []
        self.sequence.append(len(self.tasks) - 1)

        self.start_mode = self._make_start_mode_from_sequence()
        self.terminal_mode = self._make_terminal_mode_from_sequence()

        self.tolerance = 0.05

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        self.prev_mode = [0, 0]


class rai_hallway_two_dim(rai_env):
    def __init__(self, agents_can_rotate=True):
        self.C, keyframes = make_two_dim_tunnel_env(agents_can_rotate=agents_can_rotate)
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        super().__init__()

        self.tasks = []
        self.sequence = []

        self.tasks = [
            Task(
                ["a1", "a2"], SingleGoal(np.concatenate([keyframes[0], keyframes[1]]))
            ),
        ]

        self.tasks[0].name = "terminal"

        self.sequence = [0]

        self.start_mode = self._make_start_mode_from_sequence()
        self.terminal_mode = self._make_terminal_mode_from_sequence()

        self.tolerance = 0.05

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)
