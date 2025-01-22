import robotic as ry
import numpy as np
import random
import time

from typing import List, Dict, Optional
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.dependency_graph import DependencyGraph

from multi_robot_multi_goal_planning.problems.rai_config import *
from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseModeLogic,
    SequenceMixin,
    State,
    Task,
    SingleGoal,
    GoalSet,
    GoalRegion,
)
from multi_robot_multi_goal_planning.problems.rai_base_env import rai_env


class rai_two_dim_env(SequenceMixin, rai_env):
    def __init__(self, agents_can_rotate=True):
        self.C, keyframes = make_2d_rai_env(agents_can_rotate=agents_can_rotate)
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

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

        BaseModeLogic.__init__(self)

        self.tolerance = 0.1


class rai_random_two_dim(SequenceMixin, rai_env):
    def __init__(self, num_robots=3, agents_can_rotate=False):
        self.C, keyframes = make_random_two_dim_single_goal(
            num_agents=num_robots,
            num_obstacles=10,
            agents_can_rotate=agents_can_rotate,
        )
        # self.C.view(True)

        self.robots = [f"a{i}" for i in range(num_robots)]

        rai_env.__init__(self)

        self.tasks = []
        self.tasks.append(Task(self.robots, SingleGoal(keyframes[0])))
        self.tasks[-1].name = "terminal"

        self.sequence = []
        self.sequence.append(len(self.tasks) - 1)

        BaseModeLogic.__init__(self)

        self.tolerance = 0.05

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        self.prev_mode = [0, 0]

class rai_random_two_dim_single_agent(SequenceMixin, rai_env):
    def __init__(self, agents_can_rotate=False):
        self.C, keyframes = make_random_two_dim_single_goal(
            num_agents=1,
            num_obstacles=5,
            agents_can_rotate=agents_can_rotate,
        )
        # self.C.view(True)

        self.robots = [f"a{i}" for i in range(1)]

        rai_env.__init__(self)

        self.tasks = []
        self.tasks.append(Task(self.robots, SingleGoal(keyframes[0])))
        self.tasks[-1].name = "terminal"

        self.sequence = []
        self.sequence.append(len(self.tasks) - 1)

        BaseModeLogic.__init__(self)

        self.tolerance = 0.05

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        self.prev_mode = [0, 0]

class rai_hallway_two_dim(SequenceMixin, rai_env):
    def __init__(self, agents_can_rotate=True):
        self.C, keyframes = make_two_dim_tunnel_env(agents_can_rotate=agents_can_rotate)
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        self.tasks = []
        self.sequence = []

        self.tasks = [
            Task(
                ["a1", "a2"], SingleGoal(np.concatenate([keyframes[0], keyframes[1]]))
            ),
        ]

        self.tasks[0].name = "terminal"

        self.sequence = [0]

        BaseModeLogic.__init__(self)

        self.tolerance = 0.05

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)


class rai_multi_panda_arm_single_goal_env(SequenceMixin, rai_env):
    def __init__(self, num_robots: int = 3):
        self.C, keyframes = make_panda_single_joint_goal_env(num_robots=num_robots)

        # more efficient collision scene that only has the collidabe shapes (and the links)
        self.C_coll = ry.Config()
        self.C_coll.addConfigurationCopy(self.C)

        # go through all frames, and delete the ones that are only visual
        # that is, the frames that do not have a child, and are not
        # contact frames
        for f in self.C_coll.frames():
            info = f.info()
            if "shape" in info and info["shape"] == "mesh":
                self.C_coll.delFrame(f.name)

        self.C.clear()
        self.C.addConfigurationCopy(self.C_coll)

        self.robots = ["a0", "a1", "a2"]
        self.robots = self.robots[:num_robots]

        print(self.robots)

        rai_env.__init__(self)

        self.tasks = [Task(self.robots, SingleGoal(keyframes[0]))]
        self.tasks[0].name = "terminal"
        self.sequence = [0]

        BaseModeLogic.__init__(self)

        self.tolerance = 0.1


class rai_single_panda_arm_single_goal_env(SequenceMixin, rai_env):
    def __init__(self):
        self.C, keyframes = make_panda_single_joint_goal_env(num_robots=1)

        # more efficient collision scene that only has the collidabe shapes (and the links)
        self.C_coll = ry.Config()
        self.C_coll.addConfigurationCopy(self.C)

        # go through all frames, and delete the ones that are only visual
        # that is, the frames that do not have a child, and are not
        # contact frames
        for f in self.C_coll.frames():
            info = f.info()
            if "shape" in info and info["shape"] == "mesh":
                self.C_coll.delFrame(f.name)

        self.C.clear()
        self.C.addConfigurationCopy(self.C_coll)

        self.robots = ["a0"]

        print(self.robots)

        rai_env.__init__(self)

        self.tasks = [Task(self.robots, SingleGoal(keyframes[0]))]
        self.tasks[0].name = "terminal"
        self.sequence = [0]

        BaseModeLogic.__init__(self)

        self.tolerance = 0.1

class rai_ur10_handover_env(SequenceMixin, rai_env):
    def __init__(self):
        self.C, keyframes = make_handover_env()

        # more efficient collision scene that only has the collidabe shapes (and the links)
        self.C_coll = ry.Config()
        self.C_coll.addConfigurationCopy(self.C)

        # go through all frames, and delete the ones that are only visual
        # that is, the frames that do not have a child, and are not
        # contact frames
        for f in self.C_coll.frames():
            info = f.info()
            if "shape" in info and info["shape"] == "mesh":
                self.C_coll.delFrame(f.name)

        # self.C_coll.view(True)
        # self.C.view(True)

        self.C.clear()
        self.C.addConfigurationCopy(self.C_coll)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

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

        self.sequence = [0]

        BaseModeLogic.__init__(self)

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode.copy()

        self.tolerance = 0.1

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)
