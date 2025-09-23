import robotic as ry
import numpy as np
import random

from typing import List, Dict, Optional
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.dependency_graph import DependencyGraph

import multi_robot_multi_goal_planning.problems.rai_config as rai_config
from .configuration import config_dist

# from multi_robot_multi_goal_planning.problems.rai_config import *
from .planning_env import (
    BaseModeLogic,
    SequenceMixin,
    DependencyGraphMixin,
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
)
from .goals import (
    SingleGoal,
    GoalSet,
    GoalRegion,
    ConditionalGoal,
)
from .rai_base_env import rai_env
from .constraints import (
    RelativeAffineTaskSpaceEqualityConstraint,
    AffineTaskSpaceEqualityConstraint,
    AffineConfigurationSpaceEqualityConstraint,
    AffineFrameOrientationConstraint,
    relative_pose
)

from .registry import register

@register("rai.constrained_pose")
class rai_two_dim_env_pose_constraint(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_2d_rai_env_no_obs(agents_can_rotate=True)
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        self.tasks = [
            Task(
                "joint_pick",
                ["a1"],
                SingleGoal(np.array([-0.5, -0.5, 0])), constraints=[AffineConfigurationSpaceEqualityConstraint(np.array([[0, 0, 1, 0, 0, 0]]), np.array([0]), 0) ]
            ),
            Task(
                "joint_place",
                ["a2"],
                SingleGoal(np.array([0.5, 0.5, 0])),
            ),
            # terminal mode
            Task(
                "terminal",
                ["a1", "a2"],
                SingleGoal(home_pose),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["joint_pick", "joint_place", "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.manipulation = ManipulationType.STATIC
        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


@register("rai.constrained_relative_pose")
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
                "joint_pick",
                ["a1", "a2"],
                SingleGoal(joint_goal),
            ),
            Task(
                "joint_place",
                ["a1", "a2"],
                SingleGoal(rel_movement_end), constraints=[RelativeAffineTaskSpaceEqualityConstraint(["a1", "a2"], np.eye(7), rel_pose)]
            ),            
            # terminal mode
            Task(
                "terminal",
                ["a1", "a2"],
                SingleGoal(home_pose),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["joint_pick", "joint_place", "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.manipulation = ManipulationType.STATIC
        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


@register("rai.grasping")
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
                "joint_pick",
                self.robots,
                SingleGoal(pick_pose),
                type="pick", 
                frames=["a1_ur_ee_marker", "obj1"]
            ),
            Task(
                "joint_place",
                self.robots,
                SingleGoal(place_pose),
                type="place", 
                frames=["table", "obj1"],
                constraints=[RelativeAffineTaskSpaceEqualityConstraint(["a1_ur_ee_marker", "a2_ur_ee_marker"], np.eye(7),  rel_pose)]
            ),            
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(home_pose),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["joint_pick", "joint_place", "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.manipulation = ManipulationType.STATIC
        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


class rai_bimanual_husky(SequenceMixin, rai_env):
    def __init__(self):
        pass


@register("rai.constrained_2d_puzzle")
class rai_linked_2d_puzzle(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_linked_puzzle_env(agents_can_rotate=True)
        # self.C.view(True)

        self.robots = ["a1", "a2"]
        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        r1_goal = np.array([-0.5, -0.5, 0])
        r2_goal = np.array([-0.5, 0.5, 0])

        self.tasks = [
            # joint
            Task(
                "r1_goal",
                ["a1"],
                SingleGoal(r1_goal),
                # constraints=[AffineConfigurationSpaceEqualityConstraint(np.array([1, 0, 0, -1, 0, 0]), 0)]
            ),
            Task(
                "r2_goal",
                ["a2"],
                SingleGoal(r2_goal),
                constraints=[AffineConfigurationSpaceEqualityConstraint(np.array([[1, 0, 0, -1, 0, 0]]), np.array([0]))]
            ),            
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(home_pose),
                constraints=[AffineConfigurationSpaceEqualityConstraint(np.array([[1, 0, 0, -1, 0, 0]]), np.array([0]))]
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["r1_goal", "r2_goal", "terminal"]
        )

        self.collision_tolerance = 0.00
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.manipulation = ManipulationType.STATIC
        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


class rai_rfl_two_only(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_two_arms_on_a_gantry()
        

class rai_rfl(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_four_arms_on_a_gantry()

@register("rai.arm_ee_pose")
class rai_hold_glass_upright(SequenceMixin, rai_env):
    def __init__(self):
        self.C, [r1_keyframes, r2_keyframes] = rai_config.make_ur10_arm_orientation_env()
        # self.C.view(True)

        self.robots = ["a1", "a2"]
        rai_env.__init__(self)
        self.manipulating_env = True

        home_pose = self.C.getJointState()

        self.tasks = [
            # joint
            Task(
                "r1_pick",
                ["a1"],
                SingleGoal(r1_keyframes[0]),
                "pick",
                frames=["a1_ur_vacuum", "obj_1"],
            ),
            Task(
                "r1_place",
                ["a1"],
                SingleGoal(r1_keyframes[1]),
                "place",
                frames=["table", "obj_1"],
                constraints=[AffineFrameOrientationConstraint("obj_1", np.array([[0, 0, 1]]), np.array([0]))]
            ),
            Task(
                "r2_pick",
                ["a2"],
                SingleGoal(r2_keyframes[0]),
                "pick",
                frames=["a2_ur_vacuum", "obj_2"],
            ),
            Task(
                "r2_place",
                ["a2"],
                SingleGoal(r2_keyframes[1]),
                "place",
                frames=["table", "obj_2"],
                constraints=[AffineFrameOrientationConstraint("obj_2", np.array([[0, 0, 1]]), np.array([0]))]
            ),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(home_pose),
            ),
        ]
        
        self.sequence = self._make_sequence_from_names(
            ["r1_pick", "r2_pick", "r1_place", "r2_place", "terminal"]
        )

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.manipulation = ManipulationType.STATIC
        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


class rai_stacking_with_holding(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_stacking_with_holding_env()
        # self.C.view(True)
