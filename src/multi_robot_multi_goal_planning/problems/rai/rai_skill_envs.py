import robotic as ry
import numpy as np
import random

from typing import List, Dict, Optional
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.core.dependency_graph import DependencyGraph

import multi_robot_multi_goal_planning.problems.rai.rai_config as rai_config
from ..core.configuration import config_dist

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
    ModelBasedInsertion,
    RelativePoseReaching
)

from ..core.constraints import (
    relative_pose
)

from ..core.goals import (
    SingleGoal,
    GoalSet,
    GoalRegion,
    ConditionalGoal,
)
from ..rai_base_env import rai_env

from ..core.registry import register

############
# Debugging/testing envs: single agent
############

# TODO: 
# - make setting with goal region from which we start skill rather than
#   goal pose -> the controllers induce a funnel from which these skills
#   are possible to run.

@register("rai.single_agent_screw")
class rai_single_agent_screw(SequenceMixin, rai_env):
    def __init__(self):
        self.C, self.robots, [pick_pose, pre_screw_pose] = rai_config.make_ur10_screwing_env()
        # self.C.view(True)

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
                skill = JogJoint(joints=self.robot_joints[self.robots[0]], speed=np.pi/2., idx=5, duration=2.) # just moving the final joint for a fixed time
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

        self._set_default_safe_pose()

@register([
    ("rai.hallway_counterexample", {}),
    ("rai.hallway_counterexample_sweep", {'sweep': True}),
    ("rai.hallway_counterexample_wide_sweep", {'sweep': True, 'wide': True}),
])
class rai_skill_hallway(SequenceMixin, rai_env):
    def __init__(self, sweep=False, wide=False):
        self.C, self.keyframes = rai_config.make_only_short_tunnel()

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        home_pose = self.C.getJointState()

        if sweep:
            height = 1
            if wide:
                height = 1.4
            
            pts = [
                np.array([1.5, 0, 0.1]),
                np.array([-1.5, 0, 0.1]),
                np.array([-1.5, -height, 0.1]),
                np.array([-1.5, height, 0.1]),
            ]
            passage_skill = EndEffectorPositionFollowing(self.robot_joints["a1"], "a1", pts)
        else:
            passage_skill = JogJoint(joints=self.robot_joints[self.robots[0]], speed=-3 / 2, idx=0, duration=2.)

        self.tasks = [
            Task(
                "a1_pre_tunnel_passage",
                ["a1"],
                SingleGoal(np.array([1.5, 0.])),
            ),
            Task("a1_tunnel_passage",
                ["a1"],
                SingleGoal(np.array([1.5, 0.])),
                skill = passage_skill
            ),
            Task(
                "a2_goal",
                ["a2"],
                SingleGoal(self.keyframes[1]),
            ),
            Task(
                "a1_goal",
                ["a1"],
                SingleGoal(self.keyframes[0]),
            ),
            Task(
                "terminal",
                ["a1", "a2"],
                SingleGoal(self.keyframes[2]),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a1_pre_tunnel_passage", "a1_tunnel_passage", "a2_goal", "a1_goal", "terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self._set_default_safe_pose()

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
                skill = EndEffectorPositionFollowing(self.robot_joints["a1"], "a1_stick_ee", pts)
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

        self._set_default_safe_pose()

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

        self._set_default_safe_pose()

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
                skill = EEPoseGoalReaching(self.robot_joints["a1"], pick_position, "a1_ur_ee_marker")
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
                skill = EEPoseGoalReaching(self.robot_joints["a1"], place_position, "obj1"),
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

        self._set_default_safe_pose()

# TODO unfinished
@register([
    ("rai.single_agent_scripted_insert", {}),
    ("rai.single_agent_scripted_insert_goal_set", {'goal_set': True}),
])
class rai_single_agent_scripted_insert(SequenceMixin, rai_env):
    def __init__(self, goal_set=False):
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

            if goal_set:
                pre_pick_goal = GoalSet([pre_pick + np.random.rand(6) * 0.01 for _ in range(10)])
                pre_place_goal = GoalSet([pre_place + np.random.rand(6) * 0.01 for _ in range(10)])
            else:
                pre_pick_goal = SingleGoal(pre_pick + np.random.rand(6) * 0.1)
                pre_place_goal = SingleGoal(pre_place + np.random.rand(6) * 0.1)

            self.tasks.extend([
                Task(
                    f"pre_pick_{i}",
                    ["a1"],
                    pre_pick_goal,
                ),
                Task(
                    f"pick_{i}",
                    ["a1"],
                    SingleGoal(home_pose),
                    frames=["a1_ur_ee_marker", f"obj{i+1}"],
                    type="pick",
                    skill = EEPoseGoalReaching(self.robot_joints["a1"], pick_pose, "a1_ur_ee_marker")
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
                    skill = ModelBasedInsertion(self.robot_joints["a1"], place_pose, f"obj{i+1}"),
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

        self._set_default_safe_pose()

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
                    skill = EEPoseGoalReaching(self.robot_joints[robot], pick_pose, f"{robot}_ur_ee_marker")
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
                    skill = ModelBasedInsertion(self.robot_joints[robot], place_pose, f"obj{i+1}"),
                    type="place",
                    frames=["table", f"obj{i+1}"]
                )
            ])

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

        self._set_default_safe_pose()

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

        self._set_default_safe_pose()


# TODO unfinished
@register("rai.single_agent_learned_insert")
class rai_single_agent_learned_insert(SequenceMixin, rai_env):
  pass

# TODO unfinished
# multi agent rearrangement with skills
@register("rai.multi_agent_rearrangement")
class rai_multi_agent_pick_and_place(SequenceMixin, rai_env):
    pass

# multi agent stacking with skills
@register([
    ("rai.skill_box_stacking", {}),
    ("rai.skill_box_stacking_unordered_sequence", {"ordered_sequence": False}),
    ("rai.skill_box_stacking_two_robots", {"num_robots": 2}),
    ("rai.skill_box_stacking_two_robots_four_obj", {"num_robots": 2, "num_boxes": 4}),
    ("rai.skill_box_stacking_three_robots", {"num_robots": 3}),
    ("rai.skill_box_stacking_one_robot", {"num_robots": 1, "num_boxes": 2}),
])
class rai_multi_agent_stacking(SequenceMixin, rai_env):
    def __init__(self, num_robots=4, num_boxes: int = 8, ordered_sequence=True):
        self.C, keyframes, self.robots = rai_config.make_box_stacking_env(
            num_robots, num_boxes, skill_starts=True
        )

        rai_env.__init__(self)

        self.manipulating_env = True

        home_pose = self.C.getJointState()

        self.robot_objs = {r: [] for r in self.robots}

        self.tasks = []
        robot_chains: Dict[str, List[List[str]]] = {r: [] for r in self.robots}
        place_tasks_in_order: List[str] = []
        task_names = ["pick", "place"]
        for r, b, qs, g in keyframes:
            cnt = 0
            self.robot_objs[r].append(b)
            chain = []

            for t, k in zip(task_names, qs):
                task_name = r + t + "_" + b + "_" + str(cnt)
                chain.extend(["pre_" + task_name, task_name])
                if t == "pick":
                    ee_name = r + "gripper_center"
                    self.tasks.append(Task("pre_" + task_name, [r], SingleGoal(k)))

                    self.C.setJointState(k, self.robot_joints[r])
                    robot_ee_pose = self.C.getFrame(ee_name).getPose()
                    self.C.setJointState(home_pose)
                    obj_pose = self.C.getFrame(b).getPose()

                    grasp_pose = np.concatenate([obj_pose[:3], robot_ee_pose[3:]])
                    self.tasks.append(Task(task_name, [r], SingleGoal(k), t, frames=[ee_name, b],
                        skill=EEPoseGoalReaching(self.robot_joints[r], grasp_pose, ee_name)))
                else:
                    self.tasks.append(Task("pre_" + task_name, [r], SingleGoal(k)))

                    place_pose = self.C.getFrame(g).getPose()
                    self.tasks.append(Task(task_name, [r], SingleGoal(k), t, frames=["table", b],
                        skill=EEPoseGoalReaching(self.robot_joints[r], place_pose, b)))

                cnt += 1

                # if b in action_names:
                #     action_names[b].append(self.tasks[-1].name)
                # else:
                #     action_names[b] = [self.tasks[-1].name]

            robot_chains[r].append(chain)
            place_tasks_in_order.append(chain[-1])  # place task is always last in chain

        self.tasks.append(Task("terminal", self.robots, SingleGoal(self.C.getJointState())))

        if ordered_sequence:
            self.sequence = self._make_sequence_from_names([t.name for t in self.tasks])
        else:
            # Stacking requires place actions to stay in keyframe order (base before top).
            place_constraints = list(zip(place_tasks_in_order, place_tasks_in_order[1:]))

            task_name_sequence = make_task_sequence(robot_chains, constraints=place_constraints, pair_pre_tasks=True, seed=0)
        
            self.sequence = self._make_sequence_from_names(task_name_sequence + ["terminal"])

        BaseModeLogic.__init__(self)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.01
        # self.collision_resolution = 0.005
        self.collision_resolution = 0.01

        self._set_default_safe_pose()

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

        self._set_default_safe_pose()

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

        self._set_default_safe_pose()

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
                skill = DualRobotGrasping(self.robot_joints["a1"] + self.robot_joints["a2"], ee_names, [a1_transformation, a2_transformation], poses),
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

        self._set_default_safe_pose()

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
                    skill = EEPoseGoalReaching(self.robot_joints["a1"], grasp_pose, "a1_ur_gripper_center")
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
                    skill = EEPoseGoalReaching(self.robot_joints["a1"], self.C.getFrame(f"goal{i}").getPose(), f"obj{i}"),
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

        self._set_default_safe_pose()


# TODO unfinished
# in principle very similar to above, however here we have the insertion skill
# running in the end. Other main diff is that we take stuff from outside, and move them inside the bin
@register([
    ("rai.single_agent_bin_packing", {}),
    ("rai.single_agent_bin_packing_goal_set", {'goal_set': True}),
])
class rai_single_agent_bin_packing(SequenceMixin, rai_env):
    def __init__(self, goal_set=False):
        if goal_set:
            self.C, [pre_pick_type_1, pre_pick_type_2, pre_place] = rai_config.make_single_agent_bin_packing_env(True)
        else:
            self.C, [pre_pick_type_1, pre_pick_type_2, pre_place] = rai_config.make_single_agent_bin_packing_env()

        self.robots = ["a1"]

        rai_env.__init__(self)

        self.manipulating_env = True

        home_pose = self.C.getJointState()

        # assuming here that we have a place to set down an object, and we only need to go to a 'generic' position
        # above the bin for picking
        
        self.tasks = []

        if goal_set:
            self.C.setJointState(pre_place[0])
        else:
            self.C.setJointState(pre_place)

        pose = self.C.getFrame("a1_ur_ee_marker").getPose()
        self.C.setJointState(home_pose)

        ee_name = "ee_marker"

        for i in range(1,4):
            if i in [1,2]:
                pre_pick = pre_pick_type_1
            else:
                pre_pick = pre_pick_type_2
            
            if goal_set:
                pre_place_goal = GoalSet(pre_place)
            else:
                pre_place_goal = SingleGoal(pre_place)

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
                    skill = EEPoseGoalReaching(self.robot_joints["a1"], grasp_pose, "a1_ur_" + ee_name)
                ),
                Task(
                    f"pre_place_{i}",
                    ["a1"],
                    pre_place_goal,
                ),
                Task(
                    f"place_{i}",
                    ["a1"],
                    SingleGoal(pre_place),
                    skill = ModelBasedInsertion(self.robot_joints["a1"], self.C.getFrame(f"goal{i}").getPose(), f"obj{i}"),
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

        self._set_default_safe_pose()


def make_task_sequence(
    robot_chains: Dict[str, List[List[str]]],
    constraints: Optional[List[tuple]] = None,
    pair_pre_tasks: bool = False,
    seed: Optional[int] = None,
) -> List[str]:
    """Generic random topological sort over task chains with robot serialization.

    Args:
        robot_chains: per-robot list of chains. Each chain is an ordered list of
            task names that must be executed in that order.
        constraints: optional list of (task_a, task_b) pairs meaning task_a must
            appear before task_b in the sequence.
        pair_pre_tasks: if True and chains have exactly 4 tasks, treat
            (chain[0], chain[1]) and (chain[2], chain[3]) as atomic pairs — no
            other task will be interleaved within a pair.
        seed: optional random seed.

    Constraints enforced automatically:
    - Within each chain: tasks appear in the given order.
    - For same-robot chains: last task of chain i precedes first task of chain j.
      The ordering between chains is randomized, but respects any ordering already
      implied by the explicit constraints (to avoid creating cycles).
    """
    rng = random.Random(seed)
    all_chains = [chain for chains in robot_chains.values() for chain in chains]

    # Build atomic units. Node key = first task in the unit.
    unit_tasks: Dict[str, List[str]] = {}
    task_to_unit: Dict[str, str] = {}
    chain_unit_lists: List[List[str]] = []

    for chain in all_chains:
        if pair_pre_tasks and len(chain) == 4:
            unit_keys = [chain[0], chain[2]]
            unit_tasks[chain[0]] = [chain[0], chain[1]]
            unit_tasks[chain[2]] = [chain[2], chain[3]]
            for t in chain[:2]:
                task_to_unit[t] = chain[0]
            for t in chain[2:]:
                task_to_unit[t] = chain[2]
        else:
            unit_keys = list(chain)
            for t in chain:
                unit_tasks[t] = [t]
                task_to_unit[t] = t
        chain_unit_lists.append(unit_keys)

    # Unit → chain index
    unit_to_chain_idx: Dict[str, int] = {}
    for ci, unit_keys in enumerate(chain_unit_lists):
        for uk in unit_keys:
            unit_to_chain_idx[uk] = ci

    # Build predecessor graph (chain ordering + explicit constraints)
    predecessors: Dict[str, set] = {k: set() for k in unit_tasks}
    for unit_keys in chain_unit_lists:
        for j in range(1, len(unit_keys)):
            predecessors[unit_keys[j]].add(unit_keys[j - 1])
    for a, b in (constraints or []):
        predecessors[task_to_unit[b]].add(task_to_unit[a])

    # Compute forward reachability between chains so same-robot serialization
    # never introduces a cycle with the explicit constraints.
    # Forward adjacency: intra-chain unit edges + explicit constraint edges.
    unit_forward: Dict[str, List[str]] = {k: [] for k in unit_tasks}
    for unit_keys in chain_unit_lists:
        for j in range(len(unit_keys) - 1):
            unit_forward[unit_keys[j]].append(unit_keys[j + 1])
    for a, b in (constraints or []):
        unit_forward[task_to_unit[a]].append(task_to_unit[b])

    # BFS from each chain to find all transitively reachable chains
    chain_can_reach: List[set] = [set() for _ in range(len(all_chains))]
    for ci, unit_keys in enumerate(chain_unit_lists):
        visited: set = set()
        queue = list(unit_keys)
        while queue:
            u = queue.pop()
            if u in visited:
                continue
            visited.add(u)
            cj = unit_to_chain_idx[u]
            if cj != ci:
                chain_can_reach[ci].add(cj)
            queue.extend(unit_forward[u])

    # Same-robot serialization: random topological sort of chains per robot,
    # respecting the reachability-derived partial order to avoid cycles.
    chain_idx_offset = 0
    for chains in robot_chains.values():
        n = len(chains)
        my_unit_chains = chain_unit_lists[chain_idx_offset : chain_idx_offset + n]
        global_idxs = list(range(chain_idx_offset, chain_idx_offset + n))

        # local partial order: local_prec[j] = set of local indices that must precede j
        local_prec: Dict[int, set] = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i != j and global_idxs[j] in chain_can_reach[global_idxs[i]]:
                    local_prec[j].add(i)

        in_deg = {i: len(local_prec[i]) for i in range(n)}
        chain_ready = [i for i in range(n) if in_deg[i] == 0]
        chain_order: List[int] = []
        while chain_ready:
            rng.shuffle(chain_ready)
            chosen = chain_ready.pop()
            chain_order.append(chosen)
            for j in range(n):
                if chosen in local_prec[j]:
                    in_deg[j] -= 1
                    if in_deg[j] == 0:
                        chain_ready.append(j)

        for k in range(len(chain_order) - 1):
            i, j = chain_order[k], chain_order[k + 1]
            predecessors[my_unit_chains[j][0]].add(my_unit_chains[i][-1])

        chain_idx_offset += n

    # Random topological sort of units (Kahn's algorithm)
    in_degree = {t: len(preds) for t, preds in predecessors.items()}
    ready = [t for t, d in in_degree.items() if d == 0]
    result = []
    while ready:
        rng.shuffle(ready)
        chosen = ready.pop()
        result.extend(unit_tasks[chosen])
        for t, preds in predecessors.items():
            if chosen in preds:
                in_degree[t] -= 1
                if in_degree[t] == 0:
                    ready.append(t)
    return result


def make_pick_place_sequence(
    robot_objects: Dict[str, List[int]],
    pick_order: Optional[List[tuple]] = None,
    place_order: Optional[List[tuple]] = None,
    pair_pre_tasks: bool = False,
    seed: Optional[int] = None,
) -> List[str]:
    """Convenience wrapper around make_task_sequence for standard pick-place naming.

    robot_objects: per-robot list of integer object indices.
    pick_order:  list of (i, j) pairs meaning pick_i before pick_j.
    place_order: list of (i, j) pairs meaning place_i before place_j
                 (e.g. for stacking: base must be placed before the object on top).
    pair_pre_tasks: see make_task_sequence.
    """
    robot_chains = {
        r: [[f"pre_pick_{i}", f"pick_{i}", f"pre_place_{i}", f"place_{i}"] for i in objs]
        for r, objs in robot_objects.items()
    }
    constraints = (
        [(f"pick_{i}", f"pick_{j}") for i, j in (pick_order or [])]
        + [(f"place_{i}", f"place_{j}") for i, j in (place_order or [])]
    )
    return make_task_sequence(robot_chains, constraints or None, pair_pre_tasks, seed)


# TODO: more robots, more things -> ideally programmatically
@register([
    ("rai.multi_agent_bin_packing", {}),
    ("rai.multi_agent_bin_packing_unordered_sequence", {'ordered_sequence': False}),
])
class rai_multi_agent_bin_packing(SequenceMixin, rai_env):
    def __init__(self, ordered_sequence=True):
        self.C, \
            [a1_pre_pick_type_1, a1_pre_pick_type_2, a1_pre_place], \
            [a2_pre_pick_type_1, a2_pre_pick_type_2, a2_pre_place] = rai_config.make_multi_agent_bin_packing_env()
        # self.C.view(True)

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

        self.robot_objs = {"a1": [], "a2": []}

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

            self.robot_objs[robot].append(i)

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
                    skill = EEPoseGoalReaching(self.robot_joints[robot], grasp_pose, robot + "_ur_" + ee_name)
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
                    skill = ModelBasedInsertion(self.robot_joints[robot], self.C.getFrame(f"goal{i}").getPose(), f"obj{i}"),
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

        if ordered_sequence:
            task_name_sequence = []
            for i in [1,3,2]:
                task_name_sequence.extend(
                    [f"pre_pick_{i}", f"pick_{i}", f"pre_place_{i}", f"place_{i}"]
                )
        else:
            task_name_sequence = make_pick_place_sequence(self.robot_objs, pair_pre_tasks=True, seed=0)


        self.sequence = self._make_sequence_from_names(
            task_name_sequence +  ["terminal"]
        )

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self._set_default_safe_pose()

class rai_multi_agent_bin_picking_base(rai_env):
    def __init__(self, num_objects=4):
        self.C, \
            [a1_pre_pick, a1_pre_place_type_left, a1_pre_place_type_right],\
            [a2_pre_pick, a2_pre_place_type_left, a2_pre_place_type_right],\
            left_objs, right_objs = \
            rai_config.make_multi_agent_bin_picking(num_objects)
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

        self.robot_objs = {"a1": [], "a2": []}

        for i in range(1,num_objects+1):
            if i%2 == 1:
                pre_pick = a1_pre_pick
                if i-1 in left_objs:
                    place_pose = a1_pre_place_type_left
                else:
                    place_pose = a1_pre_place_type_right
                robot = "a1"
                pose = pose_a1

            else:
                pre_pick = a2_pre_pick
                if i-1 in left_objs:
                    place_pose = a2_pre_place_type_left
                else:
                    place_pose = a2_pre_place_type_right
                robot = "a2"
                pose = pose_a2

            self.robot_objs[robot].append(i)
    
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
                    skill = EEPoseGoalReaching(self.robot_joints[robot], grasp_pose, f"{robot}_ur_gripper_center")
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
                    skill = EEPoseGoalReaching(self.robot_joints[robot], self.C.getFrame(f"goal{i}").getPose(), f"obj{i}"),
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


# TODO unfinished
# pick 'any' item from a bin
# Stochastic skill
# TODO: can a skill determine which frame will be linked??
# possible solution: just add the object where the robot ends up
@register([
    ("rai.multi_agent_bin_picking", {}),
    ("rai.multi_agent_bin_picking_four_objs", {"num_objs": 4}),
    ("rai.multi_agent_bin_picking_unordered_sequence", {'ordered_sequence': False}),
])
class rai_multi_agent_bin_picking(SequenceMixin, rai_multi_agent_bin_picking_base):
    def __init__(self, num_objs=9, ordered_sequence = True):
        rai_multi_agent_bin_picking_base.__init__(self, num_objs)

        if ordered_sequence:
            task_name_sequence = []
            for i in range(1,num_objs+1):
                task_name_sequence.extend(
                    [f"pre_pick_{i}", f"pick_{i}", f"pre_place_{i}", f"place_{i}"]
                )
        else:
            # a1 handles odd objects (1, 3), a2 handles even objects (2, 4)
            task_name_sequence = make_pick_place_sequence(self.robot_objs, pair_pre_tasks=True, seed=2)

        self.sequence = self._make_sequence_from_names(
            task_name_sequence + ["terminal"]
        )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self._set_default_safe_pose()

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
        self._set_default_safe_pose()

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
@register([
    ("rai.bimanual_sorting", {}),
    ("rai.bimanual_sorting_swapped", {'swap': True}),
])
class rai_bimanual_sorting(SequenceMixin, rai_env):
    def __init__(self, swap=False):
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
                    skill = EEPoseGoalReaching(self.robot_joints[robot_name], pick_position, robot_name + "_ur_gripper_center")
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
                    skill = EEPoseGoalReaching(self.robot_joints[robot_name], place_position, obj_name),
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

        if swap:
            self.sequence = self._make_sequence_from_names(
                ["a1_pre_pick", "a2_pre_pick", "a1_pick", "a2_pick", "a1_pre_place", "a1_place", "a2_pre_place", "a2_place", "terminal"]
            )
        else:
            self.sequence = self._make_sequence_from_names(
                ["a1_pre_pick", "a1_pick", "a2_pre_pick", "a2_pick", "a1_pre_place", "a1_place", "a2_pre_place", "a2_place", "terminal"]
            )

        self.collision_tolerance = 0.001
        self.collision_resolution = 0.005

        BaseModeLogic.__init__(self)

        self._set_default_safe_pose()

@register("rai.skill_handover")
class rai_skill_handover(SequenceMixin, rai_env):
    def __init__(self):
        self.C, keyframes = rai_config.make_handover_env(skill=True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        self.manipulating_env = True

        home_pose = self.C.getJointState()

        self.C.setJointState(keyframes[2])
        a1_pose = self.C.getFrame("a1_ur_vacuum").getPose()
        a2_pose = self.C.getFrame("a2_ur_vacuum").getPose()
        self.C.setJointState(home_pose)

        self.C.view(True)

        offset = relative_pose(a2_pose, a1_pose)

        self.tasks = [
            Task(
                "a1_pick",
                ["a1"],
                SingleGoal(keyframes[0][self.robot_idx["a1"]]),
                type="pick",
                frames=["a1_ur_vacuum", "obj1"],
            ),
            Task(
                "pre_handover",
                ["a1", "a2"],
                SingleGoal(keyframes[1])
            ),
            Task(
                "handover",
                ["a1", "a2"],
                SingleGoal(keyframes[2]),
                type="handover",
                frames=["a2_ur_vacuum", "obj1"],
                skill = RelativePoseReaching(self.robot_joints["a1"] + self.robot_joints["a2"], "a1_ur_vacuum", "a2_ur_vacuum", offset)
            ),
            Task(
                "a2_place",
                ["a2"],
                SingleGoal(keyframes[3][self.robot_idx["a2"]]),
                type="place",
                frames=["table", "obj1"],
            ),
            Task("terminal", ["a1", "a2"], SingleGoal(keyframes[4])),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a1_pick", "pre_handover", "handover", "a2_place", "terminal"]
        )

        BaseModeLogic.__init__(self)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.05

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        dim = 6
        for i, r in enumerate(self.robots):
            self.safe_pose[r] = np.array(self.C.getJointState()[dim*i:dim*(i+1)])
            self.safe_pose[r][3] = -2

@register([
    ("rai.skill_flex_assembly", {}),
    ("rai.skill_flex_assembly_bottom", {"bottom": True}),
    ("rai.skill_flex_assembly_float", {"floating_ee": True}),
    ("rai.skill_flex_assembly_bottom_float", {"floating_ee": True, "bottom": True}),
])
class rai_ur10_arm_flex_assembly_env(SequenceMixin, rai_env):
    def __init__(self, floating_ee=False, bottom=False):
        self.C, keyframes = rai_config.make_flex_assembly(
            floating_ee=floating_ee, bottom_cubes=bottom, placement_offset = 0.1, align_cube_z_with_marker_x = True, view=False
        )

        self.robots = ["a1", "a2", "a3"]

        home = self.C.getJointState()
        home[3] = -np.pi/2
        home[9] = -np.pi/2
        home[15] = -np.pi/2

        self.C.setJointState(home)

        rai_env.__init__(self)
        self.manipulating_env = True

        self.tasks = []

        for i, (task, obj, robots, pose) in enumerate(keyframes):
            if i == 0:
                self.tasks.append(
                    Task("pick", robots, SingleGoal(pose), 
                    type="pick",
                    frames=[robots[0] + "_ur_vacuum", "obj_1"])
                )
            elif i == len(keyframes) - 1:
                self.tasks.append(
                    Task("place", robots, SingleGoal(pose),
                    type="place",
                    frames=["table", "obj_1"])
                )
            else:
                if task == "pick":
                    self.tasks.append(
                        Task(f"pick_{obj}", [robots[0]], SingleGoal(pose[:6]),
                        type="pick",
                        frames=[robots[0] + "_ur_vacuum", obj])
                    )
                else:
                    self.tasks.append(
                        Task(f"pre_place_{obj}", robots, SingleGoal(pose))
                    )
                    self.tasks.append(
                        Task(
                            f"placement_{obj}",
                            robots,
                            SingleGoal(pose),
                            type="place",
                            frames=["obj_1", obj],
                            skill = RelativePoseReaching(self.robot_joints[robots[0]] + self.robot_joints[robots[1]], obj, f"weld_pose_{int(obj[-1])-1}", np.array([0, 0, 0, 0, 0, 0, 0]))
                        )
                    )

        self.sequence = [i for i in range(len(self.tasks))]

        q_home = self.C.getJointState()
        self.tasks.append(Task("terminal", self.robots, SingleGoal(q_home)))

        self.sequence.append(len(self.tasks) - 1)

        BaseModeLogic.__init__(self)

        self.collision_tolerance = 0.01

        self._set_default_safe_pose()

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