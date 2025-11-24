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
    AffineRelativeFrameOrientationConstraint,
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


@register([
    ("rai.constrained_relative_pose", {}),
    ("rai.constrained_relative_pose_with_obstacle", {'obstacle': True}),
])
class rai_two_dim_env_relative_pose_constraint(SequenceMixin, rai_env):
    def __init__(self, obstacle):
        self.C = rai_config.make_2d_rai_env_no_obs(agents_can_rotate=True)
        # self.C.view(True)

        if obstacle:
            self.C.addFrame("obs").setParent(self.C.getFrame("table")).setPosition(
                self.C.getFrame("table").getPosition() + [0, 0, 0.07]
            ).setShape(ry.ST.box, size=[0.2, 0.5 - 0.001, 0.06]).setContact(
                1
            ).setColor([0, 0, 0]).setJoint(ry.JT.rigid)

        self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        joint_goal = np.array([-0.5, -0.25, 0, -0.5, 0.25, 0])
        rel_movement_end = np.array([0.5, 0.25, np.pi, 0.5, -0.25, np.pi])

        home_pose = self.C.getJointState()

        self.C.setJointState(joint_goal)
        a1_pose = self.C.getFrame("a1").getPose()
        a2_pose = self.C.getFrame("a2").getPose()
        rel_pose = relative_pose(a1_pose, a2_pose)[:, None]

        self.C.setJointState(home_pose)

        pose_projection_matrix = np.zeros((3, 7))
        pose_projection_matrix[0, 0] = 1
        pose_projection_matrix[1, 1] = 1
        pose_projection_matrix[2, 2] = 1

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
                SingleGoal(rel_movement_end), constraints=[
                    RelativeAffineTaskSpaceEqualityConstraint(["a1", "a2"], pose_projection_matrix, rel_pose),
                    AffineRelativeFrameOrientationConstraint(["a1", "a2"], "y", np.array([0, 1, 0]), 1e-3)
                    # AffineRelativeFrameOrientationConstraint(["a1", "a2"], "y", np.array([0, 0, 0]), 1e-3)
                ]
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


@register([
    ("rai.grasping", {}),
    ("rai.grasping_with_obstacle", {'obstacle': True}),
])
class rai_two_arm_grasping(SequenceMixin, rai_env):
    def __init__(self, obstacle=False):
        self.C, self.robots, keyframes = rai_config.make_bimanual_grasping_env(obstacle)
        # self.C.view(True)

        rai_env.__init__(self)

        pick_pose = keyframes[0]
        place_pose = keyframes[-1]

        home_pose = self.C.getJointState()

        self.C.setJointState(pick_pose)
        a1_pose = self.C.getFrame("a1_ur_ee_marker").getPose()
        a2_pose = self.C.getFrame("a2_ur_ee_marker").getPose()
        rel_pose = relative_pose(a1_pose, a2_pose)[:, None]

        self.C.setJointState(home_pose)

        pose_projection_matrix = np.zeros((3, 7))
        pose_projection_matrix[0, 0] = 1
        pose_projection_matrix[1, 1] = 1
        pose_projection_matrix[2, 2] = 1

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
                constraints=[
                    RelativeAffineTaskSpaceEqualityConstraint(["a1_ur_ee_marker", "a2_ur_ee_marker"], pose_projection_matrix,  rel_pose, 1e-2),
                    AffineRelativeFrameOrientationConstraint(["a1_ur_ee_marker", "a2_ur_ee_marker"], "y", np.array([0, 0, 1]), 5e-2),
                    AffineRelativeFrameOrientationConstraint(["a1_ur_ee_marker", "a2_ur_ee_marker"], "z", np.array([0, -1, 0]), 5e-1)
                    # AffineRelativeFrameOrientationConstraint(["a1_ur_ee_marker", "a2_ur_ee_marker"], "y", np.array([1, -1, 0]), 5e-2),
                    # AffineRelativeFrameOrientationConstraint(["a1_ur_ee_marker", "a2_ur_ee_marker"], "z", np.array([-1, -1, 0]), 5e-2)
                ]
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


        # for _ in range(1000):
        #     q = self.sample_config_uniform_in_limits()

        #     for constraint in self.tasks[1].constraints:
        #         if constraint.is_fulfilled(q, )
        #         self.show_config(q)


@register("rai.husky_reach")
class rai_husky_reach(SequenceMixin, rai_env):
    def __init__(self):
        self.C, kl, kr = rai_config.make_goto_husky_env()

        self.robots = ["a1", "a2"]
        rai_env.__init__(self)
        self.manipulating_env = True

        home_pose = self.C.getJointState()

        lhs_constraint = np.zeros((3, 2*9), dtype=int)
        lhs_constraint[0, [0, 9]] = [1, -1]
        lhs_constraint[1, [1, 10]] = [1, -1]
        lhs_constraint[2, [2, 11]] = [1, -1]

        rhs_constraint = np.zeros((3, 1))

        self.constraints = [AffineConfigurationSpaceEqualityConstraint(lhs_constraint, rhs_constraint)]

        self.tasks = [
            # joint
            Task(
                "r1_reach",
                ["a1"],
                GoalSet(kl),
            ),
            Task(
                "r2_reach",
                ["a2"],
                GoalSet(kr),
            ),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(home_pose),
            ),
        ]
        
        self.sequence = self._make_sequence_from_names(
            ["r1_reach", "r2_reach", "terminal"]
        )

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        # for _ in range(100):
        #     q = self.sample_config_uniform_in_limits()
        #     q[1][0] = q[0][0]
        #     q[1][1] = q[0][1]
        #     q[1][2] = q[0][2]
        #     print(q.state())
        #     print(len(q.state()))
        #     self.show_config(q)

@register("rai.husky_single_arm_stacking")
class rai_single_arm_husky_stacking(SequenceMixin, rai_env):
    def __init__(self):
        self.C, [k1, k2, k3, k4] = rai_config.make_husky_single_arm_box_stacking_env()

        self.robots = ["a1", "a2"]
        rai_env.__init__(self)
        self.manipulating_env = True

        home_pose = self.C.getJointState()

        lhs_constraint = np.zeros((3, 2*9), dtype=int)
        lhs_constraint[0, [0, 9]] = [1, -1]
        lhs_constraint[1, [1, 10]] = [1, -1]
        lhs_constraint[2, [2, 11]] = [1, -1]

        rhs_constraint = np.zeros((3, 1))

        self.constraints = [AffineConfigurationSpaceEqualityConstraint(lhs_constraint, rhs_constraint)]

        self.tasks = [
            # joint
            Task(
                "r1_pick_0",
                ["a1"],
                SingleGoal(k1[0]),
                "pick",
                frames=["a1_ur_vacuum", "obj0"],
            ),
            Task(
                "r1_place_0",
                ["a1"],
                SingleGoal(k1[1]),
                "place",
                frames=["table", "obj0"],
            ),
            Task(
                "r2_pick_0",
                ["a2"],
                SingleGoal(k2[0]),
                "pick",
                frames=["a2_ur_vacuum", "obj1"],
            ),
            Task(
                "r2_place_0",
                ["a2"],
                SingleGoal(k2[1]),
                "place",
                frames=["table", "obj1"],
            ),
            Task(
                "r1_pick_1",
                ["a1"],
                SingleGoal(k3[0]),
                "pick",
                frames=["a1_ur_vacuum", "obj2"],
            ),
            Task(
                "r1_place_1",
                ["a1"],
                SingleGoal(k3[1]),
                "place",
                frames=["table", "obj2"],
                constraints=[]
            ),
            Task(
                "r2_pick_1",
                ["a2"],
                SingleGoal(k4[0]),
                "pick",
                frames=["a2_ur_vacuum", "obj3"],
            ),
            Task(
                "r2_place_1",
                ["a2"],
                SingleGoal(k4[1]),
                "place",
                frames=["table", "obj3"],
                constraints=[]
            ),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(home_pose),
            ),
        ]
        
        self.sequence = self._make_sequence_from_names(
            ["r1_pick_0", "r1_place_0", "r2_pick_0", "r1_pick_1", "r2_place_0", "r1_place_1", "r2_pick_1", "r2_place_1", "terminal"]
        )

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


@register("rai.husky_bimanual_stacking")
class rai_bimanual_husky_stacking(SequenceMixin, rai_env):
    def __init__(self):
        self.C, [k1, k2, k3, k4] = rai_config.make_husky_bimanual_box_stacking_env()

        self.robots = ["a1", "a2"]
        rai_env.__init__(self)
        self.manipulating_env = True

        home_pose = self.C.getJointState()

        lhs_constraint = np.zeros((3, 2*9), dtype=int)
        lhs_constraint[0, [0, 9]] = [1, -1]
        lhs_constraint[1, [1, 10]] = [1, -1]
        lhs_constraint[2, [2, 11]] = [1, -1]

        rhs_constraint = np.zeros((3, 1))

        self.constraints = [AffineConfigurationSpaceEqualityConstraint(lhs_constraint, rhs_constraint)]

        def get_picking_constraints(keyframe):
            self.C.setJointState(keyframe)
            a1_pose = self.C.getFrame("a1_ur_ee_marker").getPose()
            a2_pose = self.C.getFrame("a2_ur_ee_marker").getPose()
            rel_pose = relative_pose(a1_pose, a2_pose)[:, None]
            # rel_pos = self.C.getFrame("a2_ur_ee_marker").getPose() - self.C.getFrame("a1_ur_ee_marker").getPose()

            pose_projection_matrix = np.zeros((3, 7))
            pose_projection_matrix[0, 0] = 1
            pose_projection_matrix[1, 1] = 1
            pose_projection_matrix[2, 2] = 1
            position_constraint = RelativeAffineTaskSpaceEqualityConstraint(["a1_ur_ee_marker", "a2_ur_ee_marker"], pose_projection_matrix,  rel_pose, 1e-2)

            orientation_constraint_1 = AffineRelativeFrameOrientationConstraint(["a1_ur_ee_marker", "a2_ur_ee_marker"], "x", np.array([-1, 0, 0]), 1e-1)
            # orientation_constraint_2 = AffineRelativeFrameOrientationConstraint(["a1_ur_ee_marker", "a2_ur_ee_marker"], "z", np.array([0, -1, 0]), 5e-1)

            constraints = [position_constraint, orientation_constraint_1]

            return constraints

        self.tasks = [
            # joint
            Task(
                "pick_0",
                self.robots,
                SingleGoal(k1[0]),
                "pick",
                frames=["a2_ur_vacuum", "obj0"],
            ),
            Task(
                "place_0",
                self.robots,
                SingleGoal(k1[1]),
                "place",
                frames=["table", "obj0"],
                constraints = get_picking_constraints(k1[1])
            ),
            Task(
                "pick_1",
                self.robots,
                SingleGoal(k2[0]),
                "pick",
                frames=["a2_ur_vacuum", "obj1"],
            ),
            Task(
                "place_1",
                self.robots,
                SingleGoal(k2[1]),
                "place",
                frames=["table", "obj1"],
                constraints = get_picking_constraints(k2[1])
            ),
            Task(
                "pick_2",
                self.robots,
                SingleGoal(k3[0]),
                "pick",
                frames=["a2_ur_vacuum", "obj2"],
            ),
            Task(
                "place_2",
                self.robots,
                SingleGoal(k3[1]),
                "place",
                frames=["table", "obj2"],
                constraints = get_picking_constraints(k3[1])
            ),
            Task(
                "pick_3",
                self.robots,
                SingleGoal(k4[0]),
                "pick",
                frames=["a2_ur_vacuum", "obj3"],
            ),
            Task(
                "place_3",
                self.robots,
                SingleGoal(k4[1]),
                "place",
                frames=["table", "obj3"],
                constraints = get_picking_constraints(k4[1])
            ),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(home_pose),
            ),
        ]

        self.C.setJointState(home_pose)
        
        self.sequence = self._make_sequence_from_names(
            ["pick_0", "place_0", "pick_1", "place_1", "pick_2", "place_2", "pick_3", "place_3", "terminal"]
        )

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


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
        
        self.constraints=[AffineConfigurationSpaceEqualityConstraint(np.array([[1, 0, 0, -1, 0, 0]]), np.array([0]))]

        self.tasks = [
            # joint
            Task(
                "r1_goal",
                ["a1"],
                SingleGoal(r1_goal),
            ),
            Task(
                "r2_goal",
                ["a2"],
                SingleGoal(r2_goal),
            ),            
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(home_pose),
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

@register("rai.half_rfl")
class rai_rfl_two_only(SequenceMixin, rai_env):
    def __init__(self):
        self.C, [k1, k2, k3, k4] = rai_config.make_two_arms_on_a_gantry()
        
        self.robots = ["a1", "a2"]
        rai_env.__init__(self)
        self.manipulating_env = True

        home_pose = self.C.getJointState()

        self.constraints = [AffineConfigurationSpaceEqualityConstraint(np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0]]), np.array([0]))]

        self.tasks = [
            # joint
            Task(
                "r1_pick_0",
                ["a1"],
                SingleGoal(k1[0]),
                "pick",
                frames=["a1_ur_vacuum", "obj0"],
            ),
            Task(
                "r1_place_0",
                ["a1"],
                SingleGoal(k1[1]),
                "place",
                frames=["table", "obj0"],
            ),
            Task(
                "r1_pick_1",
                ["a1"],
                SingleGoal(k2[0]),
                "pick",
                frames=["a1_ur_vacuum", "obj1"],
            ),
            Task(
                "r1_place_1",
                ["a1"],
                SingleGoal(k2[1]),
                "place",
                frames=["table", "obj1"],
            ),
            Task(
                "r2_pick_0",
                ["a2"],
                SingleGoal(k3[0]),
                "pick",
                frames=["a2_ur_vacuum", "obj2"],
            ),
            Task(
                "r2_place_0",
                ["a2"],
                SingleGoal(k3[1]),
                "place",
                frames=["table", "obj2"],
                constraints=[]
            ),
            Task(
                "r2_pick_1",
                ["a2"],
                SingleGoal(k4[0]),
                "pick",
                frames=["a2_ur_vacuum", "obj3"],
            ),
            Task(
                "r2_place_1",
                ["a2"],
                SingleGoal(k4[1]),
                "place",
                frames=["table", "obj3"],
                constraints=[]
            ),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(home_pose),
            ),
        ]
        
        self.sequence = self._make_sequence_from_names(
            ["r1_pick_0", "r1_place_0", "r1_pick_1", "r2_pick_0", "r1_place_1", "r2_place_0", "r2_pick_1", "r2_place_1", "terminal"]
        )

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

@register("rai.rfl")
class rai_rfl(SequenceMixin, rai_env):
    def __init__(self):
        self.C, [k1, k2, k3, k4] = rai_config.make_four_arms_on_a_gantry()

        self.robots = ["a1", "a2", "a3", "a4"]
        rai_env.__init__(self)
        self.manipulating_env = True

        home_pose = self.C.getJointState()

        lhs_constraint = np.zeros((2, 4*9), dtype=int)
        lhs_constraint[0, [0, 9]] = [1, -1]
        lhs_constraint[1, [18, 27]] = [1, -1]

        rhs_constraint = np.zeros((2, 1))

        self.constraints = [AffineConfigurationSpaceEqualityConstraint(lhs_constraint, rhs_constraint)]

        self.tasks = [
            # joint
            Task(
                "r1_pick_0",
                ["a1"],
                SingleGoal(k1[0]),
                "pick",
                frames=["a1_ur_vacuum", "obj0"],
            ),
            Task(
                "r1_place_0",
                ["a1"],
                SingleGoal(k1[1]),
                "place",
                frames=["table", "obj0"],
            ),
            Task(
                "r1_pick_1",
                ["a2"],
                SingleGoal(k2[0]),
                "pick",
                frames=["a2_ur_vacuum", "obj1"],
            ),
            Task(
                "r1_place_1",
                ["a2"],
                SingleGoal(k2[1]),
                "place",
                frames=["table", "obj1"],
            ),
            Task(
                "r2_pick_0",
                ["a3"],
                SingleGoal(k3[0]),
                "pick",
                frames=["a3_ur_vacuum", "obj2"],
            ),
            Task(
                "r2_place_0",
                ["a3"],
                SingleGoal(k3[1]),
                "place",
                frames=["table", "obj2"],
                constraints=[]
            ),
            Task(
                "r2_pick_1",
                ["a4"],
                SingleGoal(k4[0]),
                "pick",
                frames=["a4_ur_vacuum", "obj3"],
            ),
            Task(
                "r2_place_1",
                ["a4"],
                SingleGoal(k4[1]),
                "place",
                frames=["table", "obj3"],
                constraints=[]
            ),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(home_pose),
            ),
        ]
        
        self.sequence = self._make_sequence_from_names(
            ["r1_pick_0", "r1_place_0", "r1_pick_1", "r2_pick_0", "r1_place_1", "r2_place_0", "r2_pick_1", "r2_place_1", "terminal"]
        )

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        # for _ in range(1000):
        #     q = self.sample_config_uniform_in_limits()

        #     if self.is_collision_free(q):
        #         self.show_config(q)

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
                constraints=[AffineFrameOrientationConstraint("obj_1", "z", np.array([0, 0, 1]), np.array([1e-3]))]
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
                constraints=[AffineFrameOrientationConstraint("obj_2", "z", np.array([0, 0, 1]), np.array([1e-3]))]
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

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


@register([
    ("rai.single_stick_upright", {}),
    ("rai.single_stick_upright_clutter", {"clutter": True}),
    ("rai.single_stick", {"stick_upright": False}),
    ("rai.single_stick_ineq", {"stick_upright": False, 
                               "ineq_orientation_constraint": True}),
])
class rai_keep_single_stick_on_ground(SequenceMixin, rai_env):
    def __init__(self, stick_upright=True, ineq_orientation_constraint=False, clutter=False):
        self.C, keyframes = rai_config.make_single_arm_stick_env(clutter=clutter)
        
        self.robots = ["a1"]
        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        h = 0.26

        constraints = [AffineTaskSpaceEqualityConstraint("a1_stick_ee", np.array([[0, 0, 1, 0, 0, 0, 0]]), np.array([h]))]
        if stick_upright:
            constraints.append(
                AffineFrameOrientationConstraint("a1_stick_ee", "z", np.array([0, 0, -1]), np.array([1e-3]))
            )
        
        if ineq_orientation_constraint:
            assert False

        self.tasks = [
            # joint
            Task(
                "r1_1",
                ["a1"],
                SingleGoal(keyframes[0]),
            ),
            Task(
                "r1_2",
                ["a1"],
                SingleGoal(keyframes[1]),
                constraints=constraints
            ),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(home_pose),
            ),
        ]
        
        self.sequence = self._make_sequence_from_names(
            ["r1_1", "r1_2", "terminal"]
        )

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

@register([
    ("rai.dual_stick_upright", {}),
    ("rai.dual_stick", {"stick_upright": False}),
    ("rai.dual_stick_clutter", {"clutter": True}),
    ("rai.dual_stick_ineq", {"stick_upright": False, 
                             "ineq_orientation_constraint": True}),
])
class rai_keep_dual_stick_on_ground(SequenceMixin, rai_env):
    def __init__(self, stick_upright=True, ineq_orientation_constraint=False, clutter=False):
        self.C, r1_keyframes, r2_keyframes = rai_config.make_dual_arm_stick_env(clutter=clutter)
        # self.C.view(True)

        self.robots = ["a1", "a2"]
        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        h = 0.26

        r1_constraints = [AffineTaskSpaceEqualityConstraint("a1_stick_ee", np.array([[0, 0, 1, 0, 0, 0, 0]]), np.array([h]))]
        r2_constraints = [AffineTaskSpaceEqualityConstraint("a2_stick_ee", np.array([[0, 0, 1, 0, 0, 0, 0]]), np.array([h]))]

        if stick_upright:
            r1_constraints.append(
                AffineFrameOrientationConstraint("a1_stick_ee", "z", np.array([0, 0, -1]), np.array([1e-3]))
            )

            r2_constraints.append(
                AffineFrameOrientationConstraint("a2_stick_ee", "z", np.array([0, 0, -1]), np.array([1e-3]))            
            )

        if ineq_orientation_constraint:
            assert False

        self.tasks = [
            # joint
            Task(
                "r1_1",
                ["a1"],
                SingleGoal(r1_keyframes[0]),
            ),
            Task(
                "r1_2",
                ["a1"],
                SingleGoal(r1_keyframes[1]),
                constraints=r1_constraints
            ),
            Task(
                "r2_1",
                ["a2"],
                SingleGoal(r2_keyframes[0]),
            ),
            Task(
                "r2_2",
                ["a2"],
                SingleGoal(r2_keyframes[1]),
                constraints=r2_constraints
            ),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(home_pose),
            ),
        ]
        
        self.sequence = self._make_sequence_from_names(
            ["r1_1", "r2_1", "r1_2", "r2_2", "terminal"]
        )

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

# assembly with predetermined assembly 'insertion directions'
# effectively an orientation constraint
class rai_assembly_with_insertions():
    pass

# environment with end effector path following for 'printing'
class rai_bimanual_printing(SequenceMixin, rai_env):
    pass

# TODO: also add orientation constraints
@register("rai.stacking_with_holding")
class rai_stacking_with_holding(SequenceMixin, rai_env):
    def __init__(self):
        self.C, [r1_keyframes, r2_keyframes] = rai_config.make_stacking_with_holding_env()

        self.robots = ["a1", "a2"]
        rai_env.__init__(self)
        self.manipulating_env = True

        home_pose = self.C.getJointState()

        r1_mat = np.zeros((6, 12))
        r1_mat[:, :6] = np.eye(6)
        
        r2_mat = np.zeros((6, 12))
        r2_mat[:, 6:] = np.eye(6)
        
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
                constraints=[]
            ),
            Task(
                "r1_hold",
                ["a1"],
                SingleGoal(r1_keyframes[1]),
                constraints=[AffineConfigurationSpaceEqualityConstraint(r1_mat, r1_keyframes[1][:, None])]
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
            ),
            Task(
                "r2_hold",
                ["a2"],
                SingleGoal(r2_keyframes[1]),
                constraints=[AffineConfigurationSpaceEqualityConstraint(r2_mat, r2_keyframes[1][:, None])]
            ),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(home_pose),
            ),
        ]
        
        self.sequence = self._make_sequence_from_names(
            ["r1_pick", "r2_pick", "r1_place", "r2_place", "r1_hold", "terminal"]
        )

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


# @register("rai.assembly_with_insertion")
class rai_assembly_with_insertion(SequenceMixin, rai_env):
    def __init__(self):
        self.C, [r1_keyframes, r2_keyframes] = rai_config.make_assembly_with_insertion()

        self.robots = ["a1", "a2"]
        rai_env.__init__(self)
        self.manipulating_env = True

        home_pose = self.C.getJointState()