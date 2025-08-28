import robotic as ry
import numpy as np
import random

from typing import List, Dict, Optional
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.dependency_graph import DependencyGraph

import multi_robot_multi_goal_planning.problems.rai_config as rai_config
from multi_robot_multi_goal_planning.problems.configuration import config_dist

# from multi_robot_multi_goal_planning.problems.rai_config import *
from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseModeLogic,
    SequenceMixin,
    DependencyGraphMixin,
    State,
    Task,
    SingleGoal,
    GoalSet,
    GoalRegion,
    ConditionalGoal,
    ProblemSpec,
    AgentType,
    GoalType,
    ConstraintType,
    DynamicsType,
    ManipulationType,
    DependencyType,
    SafePoseType,
    FrameRelativePoseConstraint,
    FrameOrientationConstraint,
    relative_pose
)
from multi_robot_multi_goal_planning.problems.rai_base_env import rai_env


class rai_two_dim_env_pose_constraint(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_2d_rai_env_no_obs(agents_can_rotate=True)
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        self.tasks = [
            Task(
                ["a1"],
                SingleGoal(np.array([-0.5, -0.5, 0])), constraints=[FrameOrientationConstraint("a1", np.array([1, 0, 0, 0]))]
            ),
            Task(
                ["a2"],
                SingleGoal(np.array([0.5, 0.5, 0])),
            ),
            # terminal mode
            Task(
                ["a1", "a2"],
                SingleGoal(home_pose),
            ),
        ]

        self.tasks[0].name = "joint_pick"
        self.tasks[1].name = "joint_place"
        self.tasks[2].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            ["joint_pick", "joint_place", "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.manipulation = ManipulationType.STATIC
        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


class rai_two_dim_env_relative_pose_constraint(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_2d_rai_env_no_obs(agents_can_rotate=False)
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        joint_goal = np.array([-0.5, -0.25, -0.5, 0.25])
        rel_movement_end = np.array([0.5, 0.25, 0.5, -0.25])

        home_pose = self.C.getJointState()

        self.C.setJointState(joint_goal)
        a1_pose = self.C.getFrame("a1").getPose()
        a2_pose = self.C.getFrame("a2").getPose()
        rel_pose = relative_pose(a1_pose, a2_pose)

        self.C.setJointState(home_pose)

        self.tasks = [
            # joint
            Task(
                ["a1", "a2"],
                SingleGoal(joint_goal),
            ),
            Task(
                ["a1", "a2"],
                SingleGoal(rel_movement_end), constraints=[FrameRelativePoseConstraint(["a1", "a2"], rel_pose)]
            ),            
            # terminal mode
            Task(
                ["a1", "a2"],
                SingleGoal(home_pose),
            ),
        ]

        self.tasks[0].name = "joint_pick"
        self.tasks[1].name = "joint_place"
        self.tasks[2].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            ["joint_pick", "joint_place", "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.manipulation = ManipulationType.STATIC
        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


class rai_two_arm_grasping(SequenceMixin, rai_env):
    def __init__(self):
        self.C, self.robots, keyframes = rai_config.make_bimanual_grasping_env()
        # self.C.view(True)

        rai_env.__init__(self)

        pick_pose = keyframes[0]
        place_pose = keyframes[-1]

        home_pose = self.C.getJointState()

        self.C.setJointState(pick_pose)
        a1_pose = self.C.getFrame("a1_ur_ee_marker").getPose()
        a2_pose = self.C.getFrame("a2_ur_ee_marker").getPose()
        rel_pose = relative_pose(a1_pose, a2_pose)

        self.C.setJointState(home_pose)

        self.manipulating_env = True

        self.tasks = [
            # joint
            Task(
                self.robots,
                SingleGoal(pick_pose),
                type="pick", 
                frames=["a1_ur_ee_marker", "obj1"]
            ),
            Task(
                self.robots,
                SingleGoal(place_pose),
                type="place", 
                frames=["table", "obj1"],
                constraints=[FrameRelativePoseConstraint(["a1_ur_ee_marker", "a2_ur_ee_marker"], rel_pose)]
            ),            
            # terminal mode
            Task(
                self.robots,
                SingleGoal(home_pose),
            ),
        ]

        self.tasks[0].name = "joint_pick"
        self.tasks[1].name = "joint_place"
        self.tasks[2].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            ["joint_pick", "joint_place", "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.manipulation = ManipulationType.STATIC
        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


class rai_hold_glass_upright(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_arm_orientation_env()
        # self.C.view(True)


class rai_stacking_with_holding(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_stacking_with_holding_env()
        # self.C.view(True)
