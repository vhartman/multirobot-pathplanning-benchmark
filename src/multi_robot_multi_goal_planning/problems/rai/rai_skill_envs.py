import robotic as ry
import numpy as np
import random

from typing import List, Dict, Optional
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.dependency_graph import DependencyGraph

import multi_robot_multi_goal_planning.problems.rai.rai_config as rai_config
from ..configuration import config_dist

# from multi_robot_multi_goal_planning.problems.rai_config import *
from ..planning_env import (
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

from ..skills import (
    EEPoseGoalReaching,
    EEPositionGoalReaching,
    JogJoint,
    EndEffectorPositionFollowing,
    StochasticBinPick,
    DualRobotGrasping,
    ModelBasedInsertion
)

from ..constraints import (
    relative_pose
)

from ..goals import (
    SingleGoal,
    GoalSet,
    GoalRegion,
    ConditionalGoal,
)
from ..rai_base_env import rai_env

from ..registry import register

############
# Debugging/testing envs: single agent
############

@register("rai.single_agent_screw")
class rai_single_agent_screw(SequenceMixin, rai_env):
    def __init__(self):
        self.C, self.robots, [pick_pose, pre_screw_pose] = rai_config.make_ur10_screwing_env()
        # self.C.view(True)

        print(pick_pose, pre_screw_pose)

        rai_env.__init__(self)

        self.manipulating_env = True

        home_pose = self.C.getJointState()

        post_screw_pose = pre_screw_pose * 1.
        post_screw_pose[-1] += 2. * np.pi/2.

        self.tasks = [
            Task(
                "pick",
                [self.robots[0]],
                SingleGoal(pick_pose),
                type="pick",
                frames=["a1_ur_gripper_center", "obj1"]
            ),
            Task(
                "pre_screw",
                [self.robots[0]],
                SingleGoal(pre_screw_pose),
            ),
            Task(
                "screw",
                [self.robots[0]],
                SingleGoal(post_screw_pose),
                frames=["table", "obj1"],
                type="place",
                skill = JogJoint(speed=np.pi/2., idx=5, duration=2.) # just moving the final joint for a fixed time
            ),
            Task(
                "terminal",
                [self.robots[0]],
                SingleGoal(home_pose),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["pick", "pre_screw", "screw", "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE  
        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array(self.C.getJointState()[self.robot_idx[r]])

# Debugging for single agent timed skill
@register([
    ("rai.single_agent_drawing", {}),
    ("rai.single_agent_drawing_square", {'square': True}),
])
class rai_single_agent_drawing(SequenceMixin, rai_env):
    def __init__(self, square=False):
        self.C, poses = rai_config.make_single_agent_drawing()
        # self.C.view(True)

        self.robots = ["a1"]

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        #table_height = 0.1 
        table_height = 0.25 # Table top at z = 0.23
        if square:
            pts = [
                np.array([-0.5, 0.0, table_height]), 
                np.array([0.1, 0.0, table_height]),
                np.array([0.1, -0.5, table_height]),
                np.array([-0.5, -0.5, table_height]),
                np.array([-0.5, 0.0, table_height])
            ]
        else:
            pts = [
                np.array([-0.5, 0, table_height]), 
                np.array([0.0, 0, table_height])
            ]
        # path = LineSegment(pts)

        self.tasks = [
            Task(
                "pre_draw",
                ["a1"],
                SingleGoal(poses[0]),
            ),
            Task(
                "draw",
                ["a1"],
                SingleGoal(poses[0]), # TODO: figure out how to do skill goal checking (Valentin)
                skill = EndEffectorPositionFollowing("a1_stick_ee", pts)
            ),
            Task(
                "terminal",
                ["a1"],
                SingleGoal(home_pose),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["pre_draw", "draw", "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE
        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array(self.C.getJointState()[self.robot_idx[r]])

# TODO unfinished
@register("rai.single_agent_lego")
class rai_single_agent_lego(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_single_agent_lego()
        # self.C.view(True)

        self.robots = ["a1"]

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        lego_placement_path = CubicSpline()
        
        pick_pose = None
        pre_place_pose = None

        self.tasks = [
            Task(
                "pick",
                ["a1"],
                SingleGoal(pick_pose),
                type="pick",
                frames=["a1_ur_ee_marker", "obj1"]
            ),
            Task(
                "pre_place",
                ["a1"],
                SingleGoal(pre_place_pose),
            ),
            Task(
                "place",
                ["a1"],
                SingleGoal(np.array([0.5, 0.5, 0])),
                skill = EndEffectorPositionFollowing(lego_placement_path),
                type="place",
                frames=["table", "obj1"]
            ),
            Task(
                "terminal",
                ["a1"],
                SingleGoal(home_pose),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["pick", "pre_place", "place", "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE
        
        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array(self.C.getJointState()[self.robot_idx[r]])

# TODO: enable mode to only plan for a subset of dofs
@register("rai.single_agent_pick_and_place")
class rai_single_agent_pick_and_place(SequenceMixin, rai_env):
    def __init__(self):
        self.C, [pre_pick, pre_place] = rai_config.make_single_agent_pick_and_place()
        # self.C.view(True)

        self.robots = ["a1"]

        rai_env.__init__(self)

        self.manipulating_env = True

        home_pose = self.C.getJointState()

        pick_position = self.C.getFrame("obj1").getPose()
        place_position = self.C.getFrame("goal1").getPose()

        self.tasks = [
            Task(
                "pre_pick",
                ["a1"],
                SingleGoal(pre_pick),
            ),
            Task(
                "pick",
                ["a1"],
                SingleGoal(home_pose),
                frames=["a1_ur_ee_marker", "obj1"],
                type="pick",
                skill = EEPoseGoalReaching(pick_position, "a1_ur_ee_marker")
            ),
            Task(
                "pre_place",
                ["a1"],
                SingleGoal(pre_place),
            ),
            Task(
                "place",
                ["a1"],
                SingleGoal(home_pose),
                skill = EEPoseGoalReaching(place_position, "obj1"),
                type="place",
                frames=["table", "obj1"]
            ),
            Task(
                "terminal",
                ["a1"],
                SingleGoal(home_pose),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["pre_pick", "pick", "pre_place", "place",
            "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE
        
        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array(self.C.getJointState()[self.robot_idx[r]])

# TODO unfinished
@register("rai.single_agent_scripted_insert")
class rai_single_agent_scripted_insert(SequenceMixin, rai_env):
    def __init__(self):
        self.C, keyframes = rai_config.make_single_robot_insert()

        self.robots = ["a1"]

        rai_env.__init__(self)

        self.manipulating_env = True

        home_pose = self.C.getJointState()

        self.C.setJointState(keyframes[0][0])
        a1_pose = self.C.getFrame("a1_ur_ee_marker").getPose()
        self.C.setJointState(home_pose)

        self.tasks = []
        seq = []

        for i in range(3):
            pre_pick = keyframes[i][0]
            pre_place = keyframes[i][1]
            
            obj_name = f"obj{i+1}"
            goal_name = f"goal{i+1}"

            pick_pose = self.C.getFrame(obj_name).getPose()
            place_pose = self.C.getFrame(goal_name).getPose()

            pick_pose[3:] = a1_pose[3:]
            pick_pose[2] += 0.11

            self.tasks.extend([
                Task(
                    f"pre_pick_{i}",
                    ["a1"],
                    SingleGoal(pre_pick + np.random.rand(6) * 0.1),
                ),
                Task(
                    f"pick_{i}",
                    ["a1"],
                    SingleGoal(home_pose),
                    frames=["a1_ur_ee_marker", f"obj{i+1}"],
                    type="pick",
                    skill = EEPoseGoalReaching(pick_pose, "a1_ur_ee_marker")
                ),
                Task(
                    f"pre_place_{i}",
                    ["a1"],
                    SingleGoal(pre_place + np.random.rand(6) * 0.1),
                ),
                Task(
                    f"place_{i}",
                    ["a1"],
                    SingleGoal(home_pose),
                    # skill = EEPoseGoalReaching(place_pose, f"obj{i+1}"),
                    skill = ModelBasedInsertion(place_pose, f"obj{i+1}"),
                    type="place",
                    frames=["table", f"obj{i+1}"]
                )
            ])

            seq.extend([f"pre_pick_{i}", f"pick_{i}", f"pre_place_{i}", f"place_{i}"])

        self.tasks.append(
            Task(
                "terminal",
                ["a1"],
                SingleGoal(home_pose),
            ),
        )

        self.sequence = self._make_sequence_from_names(
            seq + ["terminal"]
        )

        self.collision_tolerance = 0.005
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE
        
        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array(self.C.getJointState()[self.robot_idx[r]])

class rai_multi_agent_scripted_insert_base(rai_env):
    def __init__(self):
        self.C, keyframes = rai_config.make_multi_robot_insert()

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        self.manipulating_env = True

        home_pose = self.C.getJointState()

        self.C.setJointState(keyframes[0][1] ,self.robot_joints["a1"])
        a1_pose = self.C.getFrame("a1_ur_ee_marker").getPose()

        self.C.setJointState(keyframes[1][1] ,self.robot_joints["a2"])
        a2_pose = self.C.getFrame("a2_ur_ee_marker").getPose()
        
        self.C.setJointState(home_pose)

        self.tasks = []

        for i in range(3):
            robot = keyframes[i][0]
            pre_pick = keyframes[i][1]
            pre_place = keyframes[i][2]
            
            obj_name = f"obj{i+1}"
            goal_name = f"goal{i+1}"

            pick_pose = self.C.getFrame(obj_name).getPose()
            place_pose = self.C.getFrame(goal_name).getPose()

            pick_pose[3:] = a1_pose[3:]
            pick_pose[2] += 0.11

            self.tasks.extend([
                Task(
                    f"pre_pick_{i}",
                    [robot],
                    SingleGoal(pre_pick + np.random.rand(6) * 0.05),
                ),
                Task(
                    f"pick_{i}",
                    [robot],
                    SingleGoal(pre_pick),
                    frames=[f"{robot}_ur_ee_marker", f"obj{i+1}"],
                    type="pick",
                    skill = EEPoseGoalReaching(pick_pose, f"{robot}_ur_ee_marker")
                ),
                Task(
                    f"pre_place_{i}",
                    [robot],
                    SingleGoal(pre_place + np.random.rand(6) * 0.05),
                ),
                Task(
                    f"place_{i}",
                    [robot],
                    SingleGoal(pre_place),
                    # skill = EEPoseGoalReaching(place_pose, f"obj{i+1}"),
                    skill = ModelBasedInsertion(place_pose, f"obj{i+1}"),
                    type="place",
                    frames=["table", f"obj{i+1}"]
                )
            ])

            # TEMORARY FIX
            self.tasks[-1].skill.joints = self.robot_joints[robot]
            self.tasks[-3].skill.joints = self.robot_joints[robot]

        self.tasks.append(
            Task(
                "terminal",
                ["a1", "a2"],
                SingleGoal(home_pose),
            ),
        )

@register("rai.multi_agent_scripted_insert")
class rai_multi_agent_scripted_insert(SequenceMixin, rai_multi_agent_scripted_insert_base):
    def __init__(self):
        rai_multi_agent_scripted_insert_base.__init__(self)

        seq = []
        for i in range(3):
            seq.extend([f"pre_pick_{i}", f"pick_{i}", f"pre_place_{i}", f"place_{i}"])

        self.sequence = self._make_sequence_from_names(
            seq + ["terminal"]
        )

        self.collision_tolerance = 0.005
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE
        
        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array(self.C.getJointState()[self.robot_idx[r]])

# TODO: add holding
@register("rai.dep_multi_agent_scripted_insert")
class rai_dep_multi_agent_scripted_insert(DependencyGraphMixin, rai_multi_agent_scripted_insert_base):
    def __init__(self):
        rai_multi_agent_scripted_insert_base.__init__(self)

        self.graph = DependencyGraph()

        for i in range(3):
            self.graph.add_dependency(f"pick_{i}", f"pre_pick_{i}")
            self.graph.add_dependency(f"pre_place_{i}", f"pick_{i}")
            self.graph.add_dependency(f"place_{i}", f"pre_place_{i}")

        self.graph.add_dependency("pre_pick_2", "place_0")

        self.graph.add_dependency("terminal", "place_1")
        self.graph.add_dependency("terminal", "place_2")        

        self.collision_tolerance = 0.005
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE
        
        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array(self.C.getJointState()[self.robot_idx[r]])


# TODO unfinished
@register("rai.single_agent_learned_insert")
class rai_single_agent_learned_insert(SequenceMixin, rai_env):
  pass

# TODO unfinished
# multi agent rearrangement with skills
@register("rai.multi_agent_rearrangement")
class rai_multi_agent_pick_and_place(SequenceMixin, rai_env):
    pass

# TODO unfinished
# multi agent rearrangement with skills
@register([
    ("rai.skill_box_stacking", {}),
    ("rai.skill_box_stacking_two_robots", {"num_robots": 2}),
    ("rai.skill_box_stacking_two_robots_four_obj", {"num_robots": 2, "num_boxes": 4}),
    ("rai.skill_box_stacking_three_robots", {"num_robots": 3}),
    ("rai.skill_box_stacking_one_robot", {"num_robots": 1, "num_boxes": 2}),
])
class rai_multi_agent_stacking(SequenceMixin, rai_env):
    def __init__(self, num_robots=4, num_boxes: int = 8):
        self.C, keyframes, self.robots = rai_config.make_box_stacking_env(
            num_robots, num_boxes, skill_starts=True
        )

        rai_env.__init__(self)

        self.manipulating_env = True

        home_pose = self.C.getJointState()

        self.tasks = []
        task_names = ["pick", "place"]
        for r, b, qs, g in keyframes:
            cnt = 0
            for t, k in zip(task_names, qs):
                task_name = r + t + "_" + b + "_" + str(cnt)
                if t == "pick":
                    ee_name = r + "gripper_center"
                    self.tasks.append(Task("pre_" + task_name, [r], SingleGoal(k)))

                    print(r)
                    self.C.setJointState(k, self.robot_joints[r])
                    robot_ee_pose = self.C.getFrame(ee_name).getPose()
                    self.C.setJointState(home_pose)
                    obj_pose = self.C.getFrame(b).getPose()

                    grasp_pose = np.concatenate([obj_pose[:3], robot_ee_pose[3:]])
                    self.tasks.append(Task(task_name, [r], SingleGoal(k), t, frames=[ee_name, b], 
                        skill=EEPoseGoalReaching(grasp_pose, ee_name)))
                else:
                    self.tasks.append(Task("pre_" + task_name, [r], SingleGoal(k)))

                    place_pose = self.C.getFrame(g).getPose()
                    self.tasks.append(Task(task_name, [r], SingleGoal(k), t, frames=["table", b], 
                        skill=EEPoseGoalReaching(place_pose, b)))

                cnt += 1

                # if b in action_names:
                #     action_names[b].append(self.tasks[-1].name)
                # else:
                #     action_names[b] = [self.tasks[-1].name]

        self.tasks.append(Task("terminal", self.robots, SingleGoal(self.C.getJointState())))

        self.sequence = self._make_sequence_from_names([t.name for t in self.tasks])

        BaseModeLogic.__init__(self)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.01
        # self.collision_resolution = 0.005
        self.collision_resolution = 0.01

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        for r in self.robots:
            print(self.C.getJointState()[0:6])
            self.safe_pose[r] = np.array(self.C.getJointState()[0:6])

# TODO unfinished
# multi agent rearrangement with skills
@register("rai.multi_agent_dexterous_stacking")
class rai_multi_agent_stacking(SequenceMixin, rai_env):
    pass

# TODO unfinished
# draw the crl logo with 3 robots:
# TODO this should be an unordered problem
@register("rai.multi_agent_drawing")
class rai_multi_agent_drawing(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_multi_agent_drawing()
        # self.C.view(True)

        self.robots = ["a1", "a2", "a3"]

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        self.tasks = [
            None
        ]

        self.sequence = self._make_sequence_from_names(
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

# TODO unfinished
# four robot, same welding env as before
# welding lines here
@register("rai.multi_agent_line_weld")
class rai_multi_agent_weld(SequenceMixin, rai_env):
    def __init__(self):
        self.C = rai_config.make_multi_agent_skill_welding_env(num_robots=4, num_lines=4)
        # self.C.view(True)

        self.robots = ["a1", "a2", "a3"]

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        self.tasks = [
            None
        ]

        self.sequence = self._make_sequence_from_names(
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

# TODO unfinished
@register("rai.multi_agent_pcb")
class rai_multi_agent_pcb(SequenceMixin, rai_env):
  pass

# TODO unfinished
# kids game: https://www.youtube.com/watch?v=Ddertj2CG3I
# vision based insertion?
# vision based grasping?
@register("rai.multi_agent_insert")
class rai_multi_agent_insert(SequenceMixin, rai_env):
  pass

@register([
    ("rai.dual_arm_transport", {}),
    ("rai.dual_arm_transport_rotation", {'rotation': True}),
])
class rai_dual_arm_transport(SequenceMixin, rai_env):
    def __init__(self, rotation=False):
        self.C, _, [self.pick_pose, _] = rai_config.make_bimanual_grasping_env(obstacle=False, rotate=rotation)
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        self.manipulating_env = True

        home_pose = self.C.getJointState()

        obj_start_pose = self.C.getFrame("obj1").getPose()
        obj_goal_pose = self.C.getFrame("goal1").getPose()

        ee_names = ["a1_ur_ee_marker", "a2_ur_ee_marker"]

        self.C.setJointState(self.pick_pose)
        
        a1_pose = self.C.getFrame("a1_ur_ee_marker").getPose()
        a2_pose = self.C.getFrame("a2_ur_ee_marker").getPose()
        obj_pose = self.C.getFrame("obj1").getPose()

        a1_transformation = relative_pose(a1_pose, obj_pose)
        a2_transformation = relative_pose(a2_pose, obj_pose)

        self.C.setJointState(home_pose)

        poses = [
            obj_start_pose, 
            obj_start_pose + np.array([0, 0, 0.1, 0, 0, 0, 0]), 
            obj_goal_pose + np.array([0, 0, 0.1, 0, 0, 0, 0]),
            obj_goal_pose
        ]

        self.tasks = [
            Task(
                "pick_1",
                ["a1", "a2"],
                SingleGoal(self.pick_pose),
                frames=["a1_ur_ee_marker", "obj1"],
                type="pick",
            ),
            Task(
                "move",
                ["a1", "a2"],
                SingleGoal(self.pick_pose),
                skill = DualRobotGrasping(ee_names, [a1_transformation, a2_transformation], poses),
                type="place",
                frames=["table", "obj1"]
            
            ),
            Task(
                "terminal",
                ["a1", "a2"],
                SingleGoal(home_pose),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["pick_1", "move", "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE
        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array(self.C.getJointState()[self.robot_idx[r]])

# TODO: figure out how we randomize stuff -> could be stored in mode?
# would then be made into a stochastic version of the bin picking problem
# allowign to simulate failure to grasp/rasping in diff. ways
# TODO unfinished
# pick 'any' item from a bin
@register("rai.single_agent_bin_picking")
class rai_single_agent_bin_picking(SequenceMixin, rai_env):
    def __init__(self):
        self.C, [pre_pick, pre_place_type_1, pre_place_type_2] = rai_config.make_single_agent_bin_picking_env()
        self.C.view(True)

        self.robots = ["a1"]

        rai_env.__init__(self)

        self.manipulating_env = True

        home_pose = self.C.getJointState()

        # assuming here that we have a place to set down an object, and we only need to go to a 'generic' position
        # above the bin for picking
        
        self.tasks = []

        self.C.setJointState(pre_pick, self.robot_joints["a1"])
        pose = self.C.getFrame("a1_ur_gripper_center").getPose()
        self.C.setJointState(home_pose)

        for i in range(1,5):
            if i%2 == 1:
                place_pose = pre_place_type_1
            else:
                place_pose = pre_place_type_2

            grasp_pose = self.C.getFrame(f"obj{i}").getPose() + np.array([0, 0, 0.05, 0, 0, 0, 0])
            grasp_pose[3:] = pose[3:]

            self.tasks.extend([
                Task(
                    f"pre_pick_{i}",
                    ["a1"],
                    SingleGoal(pre_pick),
                ),
                Task(
                    f"pick_{i}",
                    ["a1"],
                    SingleGoal(pre_pick),
                    frames=["a1_ur_gripper_center", f"obj{i}"],
                    type="pick",
                    skill = EEPoseGoalReaching(grasp_pose, "a1_ur_gripper_center")
                ),
                Task(
                    f"pre_place_{i}",
                    ["a1"],
                    SingleGoal(place_pose),
                ),
                Task(
                    f"place_{i}",
                    ["a1"],
                    SingleGoal(place_pose),
                    skill = EEPoseGoalReaching(self.C.getFrame(f"goal{i}").getPose(), f"obj{i}"),
                    type="place",
                    frames=["table", f"obj{i}"]
                )
            ])

        self.tasks.append(
            Task(
                "terminal",
                ["a1"],
                SingleGoal(home_pose),
            ))

        task_name_sequence = []
        for i in range(1,5):
            task_name_sequence.extend(
                [f"pre_pick_{i}", f"pick_{i}", f"pre_place_{i}", f"place_{i}"]
            )

        self.sequence = self._make_sequence_from_names(
            task_name_sequence +  ["terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array(self.C.getJointState()[self.robot_idx[r]])


# TODO unfinished
# in principle very similar to above, however here we have the insertion skill
# running in the end. Other main diff is that we take stuff from outside, and move them inside the bin
@register("rai.single_agent_bin_packing")
class rai_single_agent_bin_packing(SequenceMixin, rai_env):
    def __init__(self):
        self.C, [pre_pick_type_1, pre_pick_type_2, pre_place] = rai_config.make_single_agent_bin_packing_env()
        self.C.view(True)

        self.robots = ["a1"]

        rai_env.__init__(self)

        self.manipulating_env = True

        home_pose = self.C.getJointState()

        # assuming here that we have a place to set down an object, and we only need to go to a 'generic' position
        # above the bin for picking
        
        self.tasks = []

        self.C.setJointState(pre_place)
        pose = self.C.getFrame("a1_ur_ee_marker").getPose()
        self.C.setJointState(home_pose)

        ee_name = "ee_marker"

        for i in range(1,4):
            if i in [1,2]:
                pre_pick = pre_pick_type_1
            else:
                pre_pick = pre_pick_type_2
            
            grasp_pose = self.C.getFrame(f"obj{i}").getPose() + np.array([0, 0, 0.15, 0, 0, 0, 0])
            grasp_pose[3:] = 1.*pose[3:]

            self.tasks.extend([
                Task(
                    f"pre_pick_{i}",
                    ["a1"],
                    SingleGoal(pre_pick),
                ),
                Task(
                    f"pick_{i}",
                    ["a1"],
                    SingleGoal(pre_pick),
                    frames=["a1_ur_" + ee_name, f"obj{i}"],
                    type="pick",
                    skill = EEPoseGoalReaching(grasp_pose, "a1_ur_" + ee_name)
                ),
                Task(
                    f"pre_place_{i}",
                    ["a1"],
                    SingleGoal(pre_place),
                ),
                Task(
                    f"place_{i}",
                    ["a1"],
                    SingleGoal(pre_place),
                    skill = ModelBasedInsertion(self.C.getFrame(f"goal{i}").getPose(), f"obj{i}"),
                    # skill = EEPoseGoalReaching(self.C.getFrame(f"goal{i}").getPose(), f"obj{i}"),
                    type="place",
                    frames=["table", f"obj{i}"]
                )
            ])

        self.tasks.append(
            Task(
                "terminal",
                ["a1"],
                SingleGoal(home_pose),
            ))

        task_name_sequence = []
        for i in range(1,4):
            task_name_sequence.extend(
                [f"pre_pick_{i}", f"pick_{i}", f"pre_place_{i}", f"place_{i}"]
            )

        self.sequence = self._make_sequence_from_names(
            task_name_sequence +  ["terminal"]
        )

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array(self.C.getJointState()[self.robot_idx[r]])


# TODO: more robots, more things -> ideally programmatically
@register("rai.multi_agent_bin_packing")
class rai_multi_agent_bin_packing(SequenceMixin, rai_env):
    def __init__(self):
        self.C, \
            [a1_pre_pick_type_1, a1_pre_pick_type_2, a1_pre_place], \
            [a2_pre_pick_type_1, a2_pre_pick_type_2, a2_pre_place] = rai_config.make_multi_agent_bin_packing_env()
        self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        self.manipulating_env = True

        home_pose = self.C.getJointState()

        # assuming here that we have a place to set down an object, and we only need to go to a 'generic' position
        # above the bin for picking
        
        self.tasks = []

        self.C.setJointState(a1_pre_place, self.robot_joints["a1"])
        a1_pose = self.C.getFrame("a1_ur_ee_marker").getPose()

        self.C.setJointState(a2_pre_place, self.robot_joints["a2"])
        a2_pose = self.C.getFrame("a2_ur_ee_marker").getPose()
        self.C.setJointState(home_pose)

        ee_name = "ee_marker"

        for i in range(1,4):
            if i in [1,2]:
                robot = "a1"
                pre_pick = a1_pre_pick_type_1
                pose = a1_pose
                pre_place = a1_pre_place
            else:
                robot = "a2"
                pre_pick = a2_pre_pick_type_2
                pose = a2_pose
                pre_place = a2_pre_place
            

            grasp_pose = self.C.getFrame(f"obj{i}").getPose() + np.array([0, 0, 0.15, 0, 0, 0, 0])
            grasp_pose[3:] = 1.*pose[3:]

            self.tasks.extend([
                Task(
                    f"pre_pick_{i}",
                    [robot],
                    SingleGoal(pre_pick),
                ),
                Task(
                    f"pick_{i}",
                    [robot],
                    SingleGoal(pre_pick),
                    frames=[robot + "_ur_" + ee_name, f"obj{i}"],
                    type="pick",
                    skill = EEPoseGoalReaching(grasp_pose, robot + "_ur_" + ee_name)
                ),
                Task(
                    f"pre_place_{i}",
                    [robot],
                    SingleGoal(pre_place),
                ),
                Task(
                    f"place_{i}",
                    [robot],
                    SingleGoal(pre_place),
                    # skill = EEPoseGoalReaching(self.C.getFrame(f"goal{i}").getPose(), f"obj{i}"),
                    skill = ModelBasedInsertion(self.C.getFrame(f"goal{i}").getPose(), f"obj{i}"),
                    type="place",
                    frames=["table", f"obj{i}"]
                )
            ])

        self.tasks.append(
            Task(
                "terminal",
                ["a1", "a2"],
                SingleGoal(home_pose),
            ))

        task_name_sequence = []
        for i in [1,3,2]:
            task_name_sequence.extend(
                [f"pre_pick_{i}", f"pick_{i}", f"pre_place_{i}", f"place_{i}"]
            )

        self.sequence = self._make_sequence_from_names(
            task_name_sequence +  ["terminal"]
        )

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array(self.C.getJointState()[self.robot_idx[r]])

class rai_multi_agent_bin_picking_base(rai_env):
    def __init__(self):
        self.C, \
            [a1_pre_pick, a1_pre_place_type_1, a1_pre_place_type_2], \
            [a2_pre_pick, a2_pre_place_type_1, a2_pre_place_type_2] = \
            rai_config.make_multi_agent_bin_picking()
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        self.manipulating_env = True

        home_pose = self.C.getJointState()

        # assuming here that we have a place to set down an object, and we only need to go to a 'generic' position
        # above the bin for picking
        
        self.tasks = []

        self.C.setJointState(a1_pre_pick, self.robot_joints["a1"])
        pose_a1 = self.C.getFrame("a1_ur_gripper_center").getPose()
        self.C.setJointState(home_pose)

        self.C.setJointState(a2_pre_pick, self.robot_joints["a2"])
        pose_a2 = self.C.getFrame("a2_ur_gripper_center").getPose()
        self.C.setJointState(home_pose)

        for i in range(1,5):
            if i%2 == 1:
                pre_pick = a1_pre_pick
                place_pose = a1_pre_place_type_1
                robot = "a1"
                pose = pose_a1

            else:
                pre_pick = a2_pre_pick
                place_pose = a2_pre_place_type_2
                robot = "a2"
                pose = pose_a2
    
            grasp_pose = self.C.getFrame(f"obj{i}").getPose() + np.array([0, 0, 0.05, 0, 0, 0, 0])
            grasp_pose[3:] = pose[3:]

            self.tasks.extend([
                Task(
                    f"pre_pick_{i}",
                    [robot],
                    SingleGoal(pre_pick),
                ),
                Task(
                    f"pick_{i}",
                    [robot],
                    SingleGoal(pre_pick),
                    frames=[f"{robot}_ur_gripper_center", f"obj{i}"],
                    type="pick",
                    skill = EEPoseGoalReaching(grasp_pose, f"{robot}_ur_gripper_center")
                ),
                Task(
                    f"pre_place_{i}",
                    [robot],
                    SingleGoal(place_pose),
                ),
                Task(
                    f"place_{i}",
                    [robot],
                    SingleGoal(place_pose),
                    skill = EEPoseGoalReaching(self.C.getFrame(f"goal{i}").getPose(), f"obj{i}"),
                    type="place",
                    frames=["table", f"obj{i}"]
                )
            ])

            # TEMORARY FIX
            self.tasks[-1].skill.joints = self.robot_joints[robot]
            self.tasks[-3].skill.joints = self.robot_joints[robot]

        self.tasks.append(
            Task(
                "terminal",
                ["a1", "a2"],
                SingleGoal(home_pose),
            ))


# TODO unfinished
# pick 'any' item from a bin
# Stochastic skill
# TODO: can a skill determine which frame will be linked??
# possible solution: just add the object where the robot ends up
@register("rai.multi_agent_bin_picking")
class rai_multi_agent_bin_picking(SequenceMixin, rai_multi_agent_bin_picking_base):
    def __init__(self):
        rai_multi_agent_bin_picking_base.__init__(self)

        task_name_sequence = []
        for i in range(1,5):
            task_name_sequence.extend(
                [f"pre_pick_{i}", f"pick_{i}", f"pre_place_{i}", f"place_{i}"]
            )

        self.sequence = self._make_sequence_from_names(
            task_name_sequence +  ["terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array(self.C.getJointState()[self.robot_idx[r]])

@register("rai.dep_multi_agent_bin_picking")
class rai_dep_multi_agent_bin_picking(DependencyGraphMixin, rai_multi_agent_bin_picking_base):
    def __init__(self):
        rai_multi_agent_bin_picking_base.__init__(self)

        self.graph = DependencyGraph()

        for i in range(1,5):
            self.graph.add_dependency(f"pick_{i}", f"pre_pick_{i}")
            self.graph.add_dependency(f"pre_place_{i}", f"pick_{i}")
            self.graph.add_dependency(f"place_{i}", f"pre_place_{i}")

        self.graph.add_dependency("pre_pick_3", "place_1")
        self.graph.add_dependency("pre_pick_4", "place_2")

        self.graph.add_dependency("terminal", "place_3")
        self.graph.add_dependency("terminal", "place_4")        

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.prev_mode = self.start_mode

        self.spec.dependency = DependencyType.UNORDERED
        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array(self.C.getJointState()[self.robot_idx[r]])

# TODO unfinished
# skills: 
# - multiple robots -> fast pcb assembly?
# - bimanual skill with reorientation of obj? holding?
@register("rai.bimanual_assembly")
class rai_bimanual_assembly(SequenceMixin, rai_env):
  # one holding, the other adding something
  # skills might be 
  # - dual insertion where both do something
  # - single robot pick up
  # - idally both at some pt.
  pass

# TODO unfinished
# inspiration: https://arxiv.org/pdf/2511.04758
@register("rai.bimanual_sorting")
class rai_bimanual_sorting(SequenceMixin, rai_env):
    def __init__(self):
        self.C, a1_keyframes, a2_keyframes = rai_config.make_bimanual_sorting()
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)
        self.manipulating_env = True

        home_pose = self.C.getJointState()

        self.tasks = []

        for robot_name, [pre_pick, pre_place], obj_name, goal_name in zip(self.robots, [a1_keyframes, a2_keyframes], ["obj1", "obj2"], ["goal1", "goal2"]):
            # pick_position = self.C.getFrame(obj_name).getPose()
            # place_position = self.C.getFrame(goal_name).getPose()
            # place_position = np.concatenate([goal_pose[:3], downward_quat])
            # TODO (Liam) or change object definition in rai_config?
            obj_pose = self.C.getFrame(obj_name).getPose() + np.array([0, 0, 0.05, 0, 0, 0, 0])
            goal_pose = self.C.getFrame(goal_name).getPose() + np.array([0, 0, 0.05, 0, 0, 0, 0])
            downward_quat = [0., 1., 0., 0.]
            pick_position = np.concatenate([obj_pose[:3], downward_quat])
            place_position = goal_pose

            self.tasks.extend([
                Task(
                    robot_name + "_pre_pick",
                    [robot_name],
                    SingleGoal(pre_pick),
                ),
                Task(
                    robot_name + "_pick",
                    [robot_name],
                    SingleGoal(pre_pick),
                    frames=[robot_name + "_ur_gripper_center", obj_name],
                    type="pick",
                    skill = EEPoseGoalReaching(pick_position, robot_name + "_ur_gripper_center")
                ),
                Task(
                    robot_name + "_pre_place",
                    [robot_name],
                    SingleGoal(pre_place),
                ),
                Task(
                    robot_name + "_place",
                    [robot_name],
                    SingleGoal(pre_place),
                    skill = EEPoseGoalReaching(place_position, obj_name),
                    type="place",
                    frames=["table", obj_name]
                )
            ]
            )

        self.tasks.append(
            Task(
                "terminal",
                self.robots,
                SingleGoal(home_pose),
            ),
        )

        self.sequence = self._make_sequence_from_names(
            ["a1_pre_pick", "a1_pick", "a2_pre_pick", "a2_pick", "a1_pre_place", "a1_place", "a2_pre_place", "a2_place", "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array(self.C.getJointState()[self.robot_idx[r]])

# TODO unfinished
# inspiration: https://arxiv.org/pdf/2511.04758
@register("rai.so_100_sorting")
class rai_so_100_sorting(SequenceMixin, rai_env):
  pass

# TODO unfinished
# skills: 
# - screwing
# - placing
# - scaffolding stuff
@register("rai.husky_assembly")
class rai_husky_assembly(SequenceMixin, rai_env):
  pass

# TODO unfinished
# skills: 
# - grasping -> deterministic
# - insertion -> stochastic
# - do with four arms -> assemble fast
@register("rai.yijiang_corl")
class rai_yijiang_corl(SequenceMixin, rai_env):
  pass

# TODO unfinished
# skills: 
# - tying wire knots
# - inserting rods?
@register("rai.mesh")
class rai_mesh(SequenceMixin, rai_env):
  pass

# TODO unfinished
# skill to follow curved surface
@register("rai.polishing")
class rai_polising(SequenceMixin, rai_env):
  pass