from multi_robot_multi_goal_planning.assets import ASSET_ROOT
import numpy as np
import time
from typing import List, Optional
from numpy.typing import NDArray
from numba import jit, float64, int64  # ty: ignore[possibly-unbound-import]
from ...planning_env import generate_binary_search_indices
from ...core.configuration import Configuration, NpConfiguration, config_dist
from ...planning_env import (
    BaseProblem,
    BaseModeLogic,
    SequenceMixin,
    Mode,
    State,
    Task,
    ProblemSpec,
    AgentType,
    GoalType,
    ConstraintType,
    DynamicsType,
    ManipulationType,
    DependencyType,
    SafePoseType,
    SingleGoal,
)
from ...core.registry import register
import mujoco
import mujoco.viewer
import os.path

from multi_robot_multi_goal_planning.problems.mujoco.base import (
    MujocoEnvironment,
    OptimizedMujocoEnvironment,
    mul_quat,
    rot_vec_quat,
)

@register("mujoco.swap")
class swap_mujoco_env(SequenceMixin, OptimizedMujocoEnvironment):
    def __init__(self):
        path = str(ASSET_ROOT / "models/mujoco/mj_two_dim.xml")
        self.robots = [
            "a1",
            "a2",
        ]

        self.start_pos = NpConfiguration.from_list(
            [
                np.array([0, -1, 0]),
                np.array([0, 1, 0]),
            ]
        )

        OptimizedMujocoEnvironment.__init__(self, path)

        self.tasks = [
            Task("a1_goal", ["a1"], SingleGoal(np.array([0, 1, 0.0]))),
            Task("a2_goal", ["a2"], SingleGoal(np.array([0.0, -1, 0.0]))),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(self.start_pos.state()),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a1_goal", "a2_goal", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.00

        # self.show_config(self.start_pos)


@register("mujoco.hallway")
class simple_mujoco_env(SequenceMixin, MujocoEnvironment):
    def __init__(self):
        path = str(ASSET_ROOT / "models/mujoco/mj_hallway.xml")
        self.robots = [
            "a1",
            "a2",
        ]

        self.start_pos = NpConfiguration.from_list(
            [
                np.array([1.5, 0.0, 0]),
                np.array([-1.5, 0.0, 0]),
            ]
        )

        MujocoEnvironment.__init__(self, path)

        self.tasks = [
            Task("a1_goal", ["a1"], SingleGoal(np.array([-1.5, 1, np.pi / 2]))),
            Task("a2_goal", ["a2"], SingleGoal(np.array([1.5, 1, 0.0]))),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(self.start_pos.state()),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a1_goal", "a2_goal", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.00

        # self.show_config(self.start_pos)


@register("mujoco.manip")
class manip_mujoco_env(SequenceMixin, MujocoEnvironment):
    def __init__(self):
        path = str(ASSET_ROOT / "models/mujoco/mj_manip.xml")
        self.robots = [
            "a1",
            "a2",
        ]

        self.start_pos = NpConfiguration.from_list(
            [
                np.array([1.5, 0.0, 0]),
                np.array([-1.5, 0.0, 0]),
            ]
        )

        MujocoEnvironment.__init__(self, path)
        self.manipulating_env = True

        self.tasks = [
            Task(
                "a2_pick",
                ["a2"],
                SingleGoal(np.array([1.0, 0.6, 0])),
                type="pick",
                frames=["a2", "obj1"],
            ),
            Task("a1_goal", ["a1"], SingleGoal(np.array([-1.5, 1, 0.0]))),
            Task(
                "a2_place",
                ["a2"],
                SingleGoal(np.array([-1.0, -1.1, 0])),
                type="place",
                frames=["floor", "obj1"],
            ),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(self.start_pos.state()),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a2_pick", "a1_goal", "a2_place", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.00

        # self.show_config(self.start_pos)


@register("mujoco.piano")
class piano_mujoco_env(SequenceMixin, MujocoEnvironment):
    def __init__(self):
        path = str(ASSET_ROOT / "models/mujoco/mj_piano.xml")
        self.robots = [
            "a1",
            "a2",
        ]

        self.start_pos = NpConfiguration.from_list(
            [
                np.array([1.5, 0.0, 0]),
                np.array([-1.5, 0.0, 0]),
            ]
        )

        MujocoEnvironment.__init__(self, path)
        self.manipulating_env = True

        self.tasks = [
            Task(
                "a2_pick",
                ["a2"],
                SingleGoal(np.array([1.0, 0.6, 0])),
                type="pick",
                frames=["a2", "obj1"],
            ),
            Task(
                "a1_pick",
                ["a1"],
                SingleGoal(np.array([-1.0, -0.6, 0.0])),
                type="pick",
                frames=["a1", "obj2"],
            ),
            Task(
                "a2_place",
                ["a2"],
                SingleGoal(np.array([-1.0, -1.1, 0])),
                type="place",
                frames=["floor", "obj1"],
            ),
            Task(
                "a1_place",
                ["a1"],
                SingleGoal(np.array([1.0, 1.1, 0])),
                type="place",
                frames=["floor", "obj2"],
            ),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(self.start_pos.state()),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a2_pick", "a1_pick", "a2_place", "a1_place", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.00

        # self.show_config(self.start_pos)


# @register("mujoco.four_panda")
# class four_arm_mujoco_env(SequenceMixin, OptimizedMujocoEnvironment):
#     def __init__(self, agents_can_rotate=True):

#         self.robots = [
#             "panda1",
#             "panda2",
#             "panda3",
#             "panda4",
#         ]

#         self.start_pos = NpConfiguration.from_list(
#             [np.array([0, -0.5, 0, -2, 0, 2, -0.5]) for r in self.robots]
#         )

#         OptimizedMujocoEnvironment.__init__(self, path)

#         self.tasks = [
#             Task(
#                 ["panda1"], SingleGoal(np.array([-1, 0.05, 0.4, -2, 0.17, 2.5, -1.5]))
#             ),
#             Task(
#                 ["panda2"], SingleGoal(np.array([0.2, 0.05, 0.4, -2, 0.17, 2.5, -1.5]))
#             ),
#             Task(
#                 ["panda3"], SingleGoal(np.array([-1, 0.05, 0.4, -2, 0.17, 2.5, -1.5]))
#             ),
#             Task(
#                 ["panda4"], SingleGoal(np.array([0.2, 0.05, 0.4, -2, 0.17, 2.5, -1.5]))
#             ),
#             # terminal mode
#             Task(
#                 self.robots,
#                 SingleGoal(self.start_pos.state()),
#             ),
#         ]

#         self.tasks[0].name = "p1_goal"
#         self.tasks[1].name = "p2_goal"
#         self.tasks[2].name = "p3_goal"
#         self.tasks[3].name = "p4_goal"
#         self.tasks[4].name = "terminal"

#         self.sequence = self._make_sequence_from_names(
#             ["p1_goal", "p2_goal", "p3_goal", "p4_goal", "terminal"]
#         )

#         # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
#         BaseModeLogic.__init__(self)

#         self.collision_resolution = 0.01
#         self.collision_tolerance = 0.00


@register("mujoco.four_ur10")
class four_arm_ur10_mujoco_env(SequenceMixin, OptimizedMujocoEnvironment):
    def __init__(self, agents_can_rotate=True):
        path = str(ASSET_ROOT / "models/mujoco/mujoco_4_ur10_world_closer.xml")

        self.robots = [
            "ur10_1",
            "ur10_2",
            "ur10_3",
            "ur10_4",
        ]

        self.start_pos = NpConfiguration.from_list(
            [np.array([0, -2, 1.0, -1.0, -1.57, 1.0]) for r in self.robots]
        )

        OptimizedMujocoEnvironment.__init__(self, path)

        self.tasks = [
            Task("p1_goal", ["ur10_1"], SingleGoal(np.array([-1, -1, 1.3, -1.0, -1.57, 1.0]))),
            Task("p2_goal", ["ur10_2"], SingleGoal(np.array([1, -1, 1.3, -1.0, -1.57, 1.0]))),
            Task("p3_goal", ["ur10_3"], SingleGoal(np.array([-1, -1, 1.3, -1.0, -1.57, 1.0]))),
            Task("p4_goal", ["ur10_4"], SingleGoal(np.array([1, -1, 1.3, -1.0, -1.57, 1.0]))),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(self.start_pos.state()),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["p1_goal", "p2_goal", "p3_goal", "p4_goal", "terminal"]
        )

        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE
        ur10_home = np.array([0, -2, 1.0, -1.0, -1.57, 1.0])
        self.safe_pose = {r: ur10_home.copy() for r in self.robots}



# @register("mujoco.rfl")
# class rfl_mujoco(SequenceMixin, OptimizedMujocoEnvironment):
