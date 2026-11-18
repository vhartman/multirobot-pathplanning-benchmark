import copy
import numpy as np
import time
from typing import List, Optional
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from ...planning_env import generate_binary_search_indices
from ...core.configuration import Configuration, NpConfiguration, config_dist
from ...planning_env import (
    BaseModeLogic,
    SequenceMixin,
    DependencyGraphMixin,
    State,
    Mode,
    Task,
    BaseProblem,
    ProblemSpec,
    AgentType,
    GoalType,
    ConstraintType,
    DynamicsType,
    ManipulationType,
    DependencyType,
    SafePoseType,
)
from ...core.goals import SingleGoal, GoalSet, GoalRegion, ConditionalGoal
import sys
from ..base import mr_planner_core
from ...core.registry import register
import meshcat
import copy

from multi_robot_multi_goal_planning.problems.vampmr.base import VampEnv

@register("vampmr.quad_panda")
class vamp_quad_panda_env(SequenceMixin, VampEnv):
    def __init__(self, agents_can_rotate=True):
        VampEnv.__init__(self)
        self.env = mr_planner_core.VampEnvironment("panda_four")

        info = self.env.info()
        num_robots = int(info["num_robots"])

        self.robots = ["a1", "a2", "a3", "a4"]

        # self.env.enable_meshcat()
        self.env.reset_scene()

        panda_dim = 7
        self.limits = np.array([[-2] * 4 * panda_dim, [2] * 4 * panda_dim])

        self.start_pos = NpConfiguration.from_list([[0] * panda_dim, [0] * panda_dim, [0] * panda_dim, [0] * panda_dim])

        self.robot_dims = {"a1": panda_dim, "a2": panda_dim, "a3": panda_dim, "a4": panda_dim}
        self.robot_idx = {f"a{i+1}": list(range(i * panda_dim, (i + 1) * panda_dim)) for i in range(num_robots)}

        goal_pos = self.start_pos.q

        goal_tf = np.eye(4)
        goal_tf[2, 3] = 0.5

        print(goal_tf)

        ik_poses = {}
        for i, r in enumerate(self.robots):
            current_transform = self.env.end_effector_transform(i, [0]*7)
            current_transform = np.array(current_transform)
            current_transform[0, 3] = 0
            current_transform[1, 3] = 0
            current_transform[2, 3] = 0.2
            pose = self.env.inverse_kinematics(i, current_transform, max_restarts=100, tol_pos=0.01, tol_ang=1)
            print(pose)
            if pose:
                ik_poses[i] = pose

        self.tasks = [
            # r1
            Task("a1_goal", ["a1"], SingleGoal(np.array(ik_poses[0]))),
            # r2
            Task("a2_goal", ["a2"], SingleGoal(np.array(ik_poses[1]))),
            Task("a3_goal", ["a3"], SingleGoal(np.array(ik_poses[2]))),
            Task("a4_goal", ["a4"], SingleGoal(np.array(ik_poses[3]))),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(goal_pos),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a1_goal", "a2_goal", "a3_goal", "a4_goal", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self._set_zero_safe_pose(panda_dim)

@register("vampmr.hex_panda")
class vamp_hex_panda_env(SequenceMixin, VampEnv):
    def __init__(self, agents_can_rotate=True):
        VampEnv.__init__(self)
        self.env = mr_planner_core.VampEnvironment("panda_six")

        objects = self.env.get_scene_objects()
        print(objects)

        # move all the table segments down a bit
        for i in range(12):
            name = "panda_six_table_segment_" + str(i)
            self.env.remove_object(name)

        # self.env.remove_object("panda_six_plank")

        info = self.env.info()
        num_robots = int(info["num_robots"])

        self.robots = ["a1", "a2", "a3", "a4", "a5", "a6"]

        # self.env.enable_meshcat()
        self.env.update_scene()

        panda_dim = 7
        self.limits = np.array([[-2] * num_robots * panda_dim, [2] * num_robots * panda_dim])

        self.start_pos = NpConfiguration.from_list([[0] * panda_dim, [0] * panda_dim, [0] * panda_dim, [0] * panda_dim, [0] * panda_dim, [0] * panda_dim])

        self.robot_dims = {"a1": panda_dim, "a2": panda_dim, "a3": panda_dim, "a4": panda_dim, "a5": panda_dim, "a6": panda_dim}
        self.robot_idx = {f"a{i+1}": list(range(i * panda_dim, (i + 1) * panda_dim)) for i in range(num_robots)}
        # self.C.view(True)

        goal_pos = self.start_pos.q

        self.tasks = [
            # r1
            Task("a1_goal", ["a1"], SingleGoal(np.array([0.0, 0.5, -1, 0, -0, 0, 0]))),
            # r2
            Task("a2_goal", ["a2"], SingleGoal(np.array([-0.0, 0.5, -1, 0, 0, 0, 0]))),
            Task("a3_goal", ["a3"], SingleGoal(np.array([-0.0, 0.5, -1, 0, 0, 0, 0]))),
            Task("a4_goal", ["a4"], SingleGoal(np.array([-0.0, 0.5, -1, 0, 0, 0, 0]))),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(goal_pos),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a1_goal", "a2_goal", "a3_goal", "a4_goal", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self._set_zero_safe_pose(panda_dim)


@register("vampmr.dual_ur5")
class vamp_hex_panda_env(SequenceMixin, VampEnv):
    def __init__(self, num_repetitions: int = 2):
        VampEnv.__init__(self)
        self.env = mr_planner_core.VampEnvironment("dual_ur5")

        floor = mr_planner_core.Object()
        floor.name = "floor"
        floor.shape = mr_planner_core.Object.Box
        floor.state = mr_planner_core.Object.Static
        floor.x = 0.
        floor.y = 0.0
        floor.z = -0.0
        floor.qw = 1.0
        floor.qx = 0.0
        floor.qy = 0.0
        floor.qz = 0.0
        floor.width = 10   # x
        floor.height = 0.01  # y
        floor.length = 10  # z
        self.env.add_object(floor)

        self.env.set_allowed_collision("floor", "ur5_0_arm_base_link", True)
        self.env.set_allowed_collision("floor", "ur5_0_arm_shoulder_link", True)

        self.env.set_allowed_collision("floor", "ur5_1_arm_base_link", True)
        self.env.set_allowed_collision("floor", "ur5_1_arm_shoulder_link", True)

        def make_transform(angle_z, tx, ty, tz):
            c, s = np.cos(angle_z), np.sin(angle_z)
            return [
                [c, -s, 0, tx],
                [s,  c, 0, ty],
                [0,  0, 1, tz],
                [0,  0, 0,  1],
            ]

        q_offset = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        def rai_to_vamp_config(q_rai):
            return q_rai + q_offset

        self.env.set_robot_base_transform(0, make_transform(-np.pi/2,        0.0,        0.0,        -0.9144))
        self.env.set_robot_base_transform(1, make_transform(np.pi/2, 0.88128092, -0.01226491, -0.9144))

        info = self.env.info()
        num_robots = int(info["num_robots"])

        self.robots = ["a1", "a2"]

        # self.env.enable_meshcat()
        # self.env.update_scene()

        ur_dim = 6
        self.limits = np.array([[-4] * num_robots * ur_dim, [4] * num_robots * ur_dim])

        # start_config_rai = np.array([0, 0, 0, 0, 0, 0])
        start_config_rai = np.array([0, -.5, 1, .5, -1.57, 0])

        start_config = rai_to_vamp_config(start_config_rai)

        self.start_pos = NpConfiguration.from_list([start_config, start_config])

        self.robot_dims = {"a1": ur_dim, "a2": ur_dim}
        self.robot_idx = {f"a{i+1}": list(range(i * ur_dim, (i + 1) * ur_dim)) for i in range(num_robots)}
        # self.C.view(True)

        goal_pos = self.start_pos.q

        p1 = [ 6.05587010e-01, -1.82889078e-03, -1.86183192e+00, -7.57176381e-01, 8.65312213e-01, -8.24393655e-04]
        p2 = [ 0.52888098, -0.14924623, -1.7284803,  -0.64827086,  0.72253539,  0.01270607]

        goal_tasks = []
        sequence_names = []
        for rep in range(num_repetitions):
            suffix = f"_{rep + 1}" if num_repetitions > 1 else ""
            a1_name = f"a1_goal{suffix}"
            a2_name = f"a2_goal{suffix}"
            goal_tasks.append(Task(a1_name, ["a1"], SingleGoal(rai_to_vamp_config(np.array(p1)))))
            goal_tasks.append(Task(a2_name, ["a2"], SingleGoal(rai_to_vamp_config(np.array(p2)))))
            sequence_names += [a1_name, a2_name]

        self.tasks = goal_tasks + [
            Task("terminal", self.robots, SingleGoal(goal_pos)),
        ]

        self.sequence = self._make_sequence_from_names(sequence_names + ["terminal"])

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self._set_zero_safe_pose(ur_dim)


@register("vampmr.dual_ur5_with_box")
class vamp_hex_panda_env(SequenceMixin, VampEnv):
    def __init__(self, agents_can_rotate=True):
        VampEnv.__init__(self)
        self.env = mr_planner_core.VampEnvironment("dual_ur5")

        self.manipulating_env = True

        info = self.env.info()
        num_robots = int(info["num_robots"])

        self.robots = ["a1", "a2"]

        box = mr_planner_core.Object()
        box.name = "box1"
        box.shape = mr_planner_core.Object.Box
        box.state = mr_planner_core.Object.Static
        box.x = 0.5
        box.y = 0.0
        box.z = 0.5
        box.qw = 1.0
        box.qx = 0.0
        box.qy = 0.0
        box.qz = 0.0
        box.width = 0.3   # x
        box.height = 0.1  # y
        box.length = 0.1  # z
        self.env.add_object(box)
        self._initial_objects[box.name] = box

        floor = mr_planner_core.Object()
        floor.name = "floor"
        floor.shape = mr_planner_core.Object.Box
        floor.state = mr_planner_core.Object.Static
        floor.x = 0.
        floor.y = 0.0
        floor.z = -0.0
        floor.qw = 1.0
        floor.qx = 0.0
        floor.qy = 0.0
        floor.qz = 0.0
        floor.width = 10   # x
        floor.height = 0.01  # y
        floor.length = 10  # z
        self.env.add_object(floor)

        self.env.set_allowed_collision("floor", "ur5_0_arm_base_link", True)
        self.env.set_allowed_collision("floor", "ur5_0_arm_shoulder_link", True)

        self.env.set_allowed_collision("floor", "ur5_1_arm_base_link", True)
        self.env.set_allowed_collision("floor", "ur5_1_arm_shoulder_link", True)

        def make_transform(angle_z, tx, ty, tz):
            c, s = np.cos(angle_z), np.sin(angle_z)
            return [
                [c, -s, 0, tx],
                [s,  c, 0, ty],
                [0,  0, 1, tz],
                [0,  0, 0,  1],
            ]

        q_offset = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        def rai_to_vamp_config(q_rai):
            return q_rai + q_offset

        self.env.set_robot_base_transform(0, make_transform(-np.pi/2,        0.0,        0.0,        -0.9144))
        self.env.set_robot_base_transform(1, make_transform(np.pi/2, 0.88128092, -0.01226491, -0.9144))

        # self.env.enable_meshcat()
        self.env.update_scene()

        ur_dim = 6
        self.limits = np.array([[-4] * num_robots * ur_dim, [4] * num_robots * ur_dim])

        start_config_rai = np.array([0, -.5, 1, .5, -1.57, 0])
        start_config = rai_to_vamp_config(start_config_rai)

        self.start_pos = NpConfiguration.from_list([start_config, start_config])

        self.robot_dims = {"a1": ur_dim, "a2": ur_dim}
        self.robot_idx = {f"a{i+1}": list(range(i * ur_dim, (i + 1) * ur_dim)) for i in range(num_robots)}
        # self.C.view(True)

        goal_pos = self.start_pos.q

        self.initial_sg = {"box1": ("world", None, None, None, None)}

        self.tasks = [
            # r1
            Task("a1_goal", ["a1"], SingleGoal(start_config + np.array([0, 0.0, 0., 0, 0, 0])), type="pick", frames=["a1", "box1"]),
            # r2
            Task("a2_goal", ["a2"], SingleGoal(np.array([-0.0, -0.5, -1, 0, 0, 0]))),
            # Task("a1_place", ["a1"], SingleGoal(start_config + np.array([0, 0, 0, 0, 1, 0]))),
            Task("a1_place", ["a1"], SingleGoal(start_config + np.array([0, 0, 0, 0, 1, 0])), type="place", frames=["world", "box1"]),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(goal_pos),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a1_goal", "a2_goal", "a1_place", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self._set_zero_safe_pose(ur_dim)

@register("vampmr.quad_ur5")
class vamp_hex_panda_env(SequenceMixin, VampEnv):
    def __init__(self, num_repetitions=2):
        VampEnv.__init__(self)
        self.env = mr_planner_core.VampEnvironment("quad_ur5")

        floor = mr_planner_core.Object()
        floor.name = "floor"
        floor.shape = mr_planner_core.Object.Box
        floor.state = mr_planner_core.Object.Static
        floor.x = 0.
        floor.y = 0.0
        floor.z = -0.0
        floor.qw = 1.0
        floor.qx = 0.0
        floor.qy = 0.0
        floor.qz = 0.0
        floor.width = 10   # x
        floor.height = 0.01  # y
        floor.length = 10  # z
        self.env.add_object(floor)

        for i in range(4):
            self.env.set_allowed_collision("floor", f"ur5_{i}_arm_base_link", True)
            self.env.set_allowed_collision("floor", f"ur5_{i}_arm_shoulder_link", True)

        def make_transform(angle_z, tx, ty, tz):
            c, s = np.cos(angle_z), np.sin(angle_z)
            return [
                [c, -s, 0, tx],
                [s,  c, 0, ty],
                [0,  0, 1, tz],
                [0,  0, 0,  1],
            ]

        q_offset = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        def rai_to_vamp_config(q_rai):
            return q_rai + q_offset

        self.env.set_robot_base_transform(0, make_transform(np.pi/2,   0.36, -0.36, -0.9144))
        self.env.set_robot_base_transform(1, make_transform(-np.pi/2, -0.36, -0.36, -0.9144))
        self.env.set_robot_base_transform(2, make_transform(np.pi/2, 0.36,  0.36, -0.9144))
        self.env.set_robot_base_transform(3, make_transform(-np.pi/2, -0.36,  0.36, -0.9144))

        info = self.env.info()
        num_robots = int(info["num_robots"])

        self.robots = ["a1", "a2", "a3", "a4"]

        # self.env.enable_meshcat()
        # self.env.reset_scene()
        # self.env.update_scene()

        ur_dim = 6
        lower = rai_to_vamp_config(np.array([-np.pi, -2, -np.pi, -np.pi, -np.pi, -np.pi])).tolist()
        upper = rai_to_vamp_config(np.array([np.pi, 0.75, np.pi, np.pi, np.pi, np.pi])).tolist()
        self.limits = np.array([lower * num_robots,
                                upper * num_robots])

        start_config_rai = np.array([0, -.5, 1, .5, -1.57, 0])
        start_config = rai_to_vamp_config(start_config_rai)

        self.start_pos = NpConfiguration.from_list([start_config, start_config, start_config, start_config])

        self.env.set_joint_positions(self.start_pos.as_list())

        self.robot_dims = {"a1": ur_dim, "a2": ur_dim, "a3": ur_dim, "a4": ur_dim}
        self.robot_idx = {f"a{i+1}": list(range(i * ur_dim, (i + 1) * ur_dim)) for i in range(num_robots)}
        # self.C.view(True)

        goal_pos = self.start_pos.q

        p1 = np.array([-1.08443716e+00,  2.65812454e-01,  1.19333815e+00,4.95651623e-01, -1.51123655e+00, -6.12287039e-04])
        p2 = np.array([ 0.25372525,  0.15432183,  1.37953107,  0.4719497 , -1.11911701, 0.01556415])
        p3 = np.array([ 0.25500657,  0.15616105,  1.37700978,  0.47377805, -1.12129939, 0.00521846])
        p4 = np.array([-1.08587427,  0.27071415,  1.18725584,  0.50407107, -1.5087889, 0.00212636])

        goal_tasks = []
        sequence_names = []
        for rep in range(num_repetitions):
            suffix = f"_{rep + 1}" if num_repetitions > 1 else ""
            a1_name = f"a1_goal{suffix}"
            a2_name = f"a2_goal{suffix}"
            a3_name = f"a3_goal{suffix}"
            a4_name = f"a4_goal{suffix}"
            goal_tasks.append(Task(a1_name, ["a1"], SingleGoal(rai_to_vamp_config(p1))))
            goal_tasks.append(Task(a2_name, ["a2"], SingleGoal(rai_to_vamp_config(p2))))
            goal_tasks.append(Task(a3_name, ["a3"], SingleGoal(rai_to_vamp_config(p3))))
            goal_tasks.append(Task(a4_name, ["a4"], SingleGoal(rai_to_vamp_config(p4))))
            sequence_names += [a1_name, a2_name, a3_name, a4_name]

        self.tasks = goal_tasks + [
            Task("terminal", self.robots, SingleGoal(goal_pos)),
        ]

        self.sequence = self._make_sequence_from_names(sequence_names + ["terminal"])

        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self._set_zero_safe_pose(ur_dim)


@register("vampmr.ur5_box_stacking")
class vamp_ur5_box_stacking_env(SequenceMixin, VampEnv):
    def __init__(self, num_robots: int = 4, num_boxes: int = 8):
        VampEnv.__init__(self)
        self.env = mr_planner_core.VampEnvironment("quad_ur5")
        self.manipulating_env = True

        from ...rai.envs import builders as rai_config
        C, keyframes, all_robots = rai_config.make_box_stacking_env(
            num_robots=num_robots, num_boxes=num_boxes, robot_types="ur5"
        )

        self.robots = [f"a{i+1}" for i in range(num_robots)]
        ur_dim = 6

        def make_transform(angle_z, tx, ty, tz):
            c, s = np.cos(angle_z), np.sin(angle_z)
            return [
                [c, -s, 0, tx],
                [s,  c, 0, ty],
                [0,  0, 1, tz],
                [0,  0, 0,  1],
            ]

        q_offset = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        def rai_to_vamp_config(q_rai):
            return np.array(q_rai) + q_offset * 0

        # Robot base transforms — same layout as vampmr.quad_ur5; adjust offsets as needed
        individual_offsets = [
            [-0.1, 0.1, 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0.0, 0.02, 0.],
        ]
        for i in range(4):
            pos = C.getFrame(f"a{i+1}_ur_base").getPosition() + np.array(individual_offsets[i]) * 0
            z_offset = 0.9144 - 0.
            rot = np.pi/2
            if i>=2:
                rot = -np.pi/2
            self.env.set_robot_base_transform(i, make_transform(rot + np.pi/2, pos[0], pos[1], pos[2] - z_offset))

        for i in range(num_robots):
            self.env.set_allowed_collision("floor", f"ur5_{i}_arm_base_link", True)
            self.env.set_allowed_collision("floor", f"ur5_{i}_arm_shoulder_link", True)

        # Floor
        floor = mr_planner_core.Object()
        floor.name = "floor"
        floor.shape = mr_planner_core.Object.Box
        floor.state = mr_planner_core.Object.Static
        floor.x = 0.; floor.y = 0.; floor.z = 0.
        floor.qw = 1.; floor.qx = 0.; floor.qy = 0.; floor.qz = 0.
        floor.width = 10; floor.height = 0.01; floor.length = 10
        self.env.add_object(floor)

        # Table — get world pose from rai config
        table_pos  = C.getFrame("table").getPosition()   # [x, y, z]
        table_quat = C.getFrame("table").getQuaternion() # [w, x, y, z]

        table_obj = mr_planner_core.Object()
        table_obj.name = "table"
        table_obj.shape = mr_planner_core.Object.Box
        table_obj.state = mr_planner_core.Object.Static
        table_obj.x = float(table_pos[0]); table_obj.y = float(table_pos[1]); table_obj.z = float(table_pos[2])
        table_obj.qw = float(table_quat[3]); table_obj.qx = float(table_quat[0])
        table_obj.qy = float(table_quat[1]); table_obj.qz = float(table_quat[2])
        table_obj.width = 3.0; table_obj.height = 0.06; table_obj.length = 3.0
        self.env.add_object(table_obj)

        for i in range(num_robots):
            self.env.set_allowed_collision("table", f"ur5_{i}_arm_base_link", True)
            self.env.set_allowed_collision("table", f"ur5_{i}_arm_shoulder_link", True)

        # Boxes — extract world positions from rai config
        boxes_seen = []
        for r_rai, b, qs, g in keyframes:
            if b not in boxes_seen:
                boxes_seen.append(b)

        for box_name in boxes_seen:
            pos  = C.getFrame(box_name).getPosition()   # [x, y, z]
            quat = C.getFrame(box_name).getQuaternion() # [w, x, y, z]

            box_obj = mr_planner_core.Object()
            box_obj.name = box_name
            box_obj.shape = mr_planner_core.Object.Box
            box_obj.state = mr_planner_core.Object.Static
            box_obj.x = float(pos[0]); box_obj.y = float(pos[1]); box_obj.z = float(pos[2])
            box_obj.qw = float(quat[0]); box_obj.qx = float(quat[1])
            box_obj.qy = float(quat[2]); box_obj.qz = float(quat[3])
            box_obj.width = 0.05; box_obj.height = 0.05; box_obj.length = 0.05

            self.env.add_object(box_obj)
            self._initial_objects[box_name] = box_obj
            self.initial_sg[box_name] = ("world", None, None, None, None)

        # Joint limits and start config
        # lower = rai_to_vamp_config(np.array([-np.pi, -3, -np.pi, -np.pi, -np.pi, -np.pi])).tolist()
        # upper = rai_to_vamp_config(np.array([np.pi, 0.0, np.pi, np.pi, np.pi, np.pi])).tolist()
        # self.limits = np.array([lower * num_robots, upper * num_robots])

        self.robot_dims = {f"a{i+1}": ur_dim for i in range(num_robots)}
        self.robot_idx = {f"a{i+1}": list(range(i * ur_dim, (i + 1) * ur_dim)) for i in range(num_robots)}

        # start_config = rai_to_vamp_config(np.array([0, -.5, 1, .5, -1.57, 0]))
        # start_config = rai_to_vamp_config(np.array([0, -2., 1, -1, -1.57, 0]))
        self.start_pos = NpConfiguration.from_list([C.getJointState()[self.robot_idx[r]] for r in self.robots])
        self.env.set_joint_positions(self.start_pos.as_list())

        self.limits = C.getJointLimits()

        # Build pick/place tasks from rai keyframes
        # keyframe format: (robot_prefix, box_name, [pick_q, place_q], goal_name)
        # robot_prefix is like "a1_ur_"; split on "_" to get "a1"
        self.tasks = []
        task_names_seq = []

        for r_rai, b, qs, _g in keyframes[:num_boxes*2]:
            r = r_rai.split("_")[0]  # "a1_ur_" -> "a1"
            pick_q  = rai_to_vamp_config(qs[0])
            place_q = rai_to_vamp_config(qs[1])

            pick_name  = f"{r}_pick_{b}"
            place_name = f"{r}_place_{b}"

            self.tasks.append(Task(pick_name,  [r], SingleGoal(pick_q),  "pick",  frames=[r, b]))
            self.tasks.append(Task(place_name, [r], SingleGoal(place_q), "place", frames=["world", b]))
            task_names_seq += [pick_name, place_name]

            # print(place_name)
            # print(place_q)

        self.tasks.append(Task("terminal", self.robots, SingleGoal(self.start_pos.q)))
        task_names_seq.append("terminal")

        self.sequence = self._make_sequence_from_names(task_names_seq)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.00

        self._set_zero_safe_pose(ur_dim)
