import robotic as ry
import numpy as np
import random
import time

from typing import List, Dict, Optional
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseModeLogic,
    FreeMixin,
    State,
    Task,
    SingleGoal,
    GoalSet,
    GoalRegion,
    SafePoseType
)
from multi_robot_multi_goal_planning.problems.rai_base_env import rai_env
import multi_robot_multi_goal_planning.problems.rai_config as rai_config


class rai_two_dim_env(FreeMixin, rai_env):
    def __init__(self, agents_can_rotate=True):
        self.C = rai_config.make_2d_rai_env_no_obs(agents_can_rotate=agents_can_rotate)
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        # r1 starts at both negative
        r1_state = self.C.getJointState()[self.robot_idx["a1"]]
        # r2 starts at both positive
        r2_state = self.C.getJointState()[self.robot_idx["a2"]]

        r1_goal = r1_state * 1.0
        r1_goal[:2] = [-0.0, 0.0]

        r2_goal_1 = r2_state * 1.0
        r2_goal_1[:2] = [0.5, -0.5]
        r2_goal_2 = r2_state * 1.0
        r2_goal_2[:2] = [-0.5, -0.5]
        r2_goal_3 = r2_state * 1.0
        r2_goal_3[:2] = [-0.5, 0.5]

        self.tasks = [
            Task(
                ["a1", "a2"],
                SingleGoal(self.C.getJointState()),
            ),
            # r1
            Task(["a1"], SingleGoal(r1_goal)),
            Task(["a1"], SingleGoal(r2_goal_1)),
            Task(["a1"], SingleGoal(r2_goal_2)),
            Task(["a1"], SingleGoal(r2_goal_3)),
            # r2
            Task(["a2"], SingleGoal(r1_goal)),
            Task(["a2"], SingleGoal(r2_goal_1)),
            Task(["a2"], SingleGoal(r2_goal_2)),
            Task(["a2"], SingleGoal(r2_goal_3)),
            # terminal mode
            Task(
                ["a1", "a2"],
                SingleGoal(self.C.getJointState()),
            ),
        ]

        # self.tasks[0].name = "dummy_start"
        # self.tasks[1].name = "a1_goal"
        # self.tasks[2].name = "a2_goal_0"
        # self.tasks[3].name = "a2_goal_1"
        # self.tasks[4].name = "a2_goal_2"
        # self.tasks[5].name = "a2_goal_3"
        # self.tasks[6].name = "terminal"

        self.task_groups = [
            [(0, 1), (1, 5)],
            [(0, 2), (1, 6)],
            [(0, 3), (1, 7)],
            [(0, 4), (1, 8)],
        ]
        self.terminal_task = len(self.tasks) - 1
        self.task_dependencies = {}
        self.task_dependencies_any = {}

        self.collision_tolerance = 0.01

        BaseModeLogic.__init__(self)


def make_piano_mover_env(view: bool = False):
    C = rai_config.make_table_with_walls(2, 2)
    table = C.getFrame("table")

    pre_agent_1_frame = (
        C.addFrame("pre_agent_1_frame")
        .setParent(table)
        .setPosition(table.getPosition() + [0.0, 0.0, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    C.addFrame("a1").setParent(pre_agent_1_frame).setShape(
        ry.ST.cylinder, size=[0.1, 0.2, 0.06, 0.075]
    ).setColor([1, 0.5, 0]).setContact(1).setJoint(
        ry.JT.transXYPhi, limits=np.array([-1, 1, -1, 1, -3.14, 3.14])
    ).setJointState([0.0, -0.5, 0.0])

    pre_agent_2_frame = (
        C.addFrame("pre_agent_2_frame")
        .setParent(table)
        .setPosition(table.getPosition() + [0, 0.0, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    C.addFrame("a2").setParent(pre_agent_2_frame).setShape(
        ry.ST.cylinder, size=[4, 0.1, 0.06, 0.075]
    ).setColor([0.5, 0.5, 0]).setContact(1).setJoint(
        ry.JT.transXYPhi, limits=np.array([-1, 1, -1, 1, -3.14, 3.14])
    ).setJointState([0, 0.5, 0.0])

    C.addFrame("obj1").setParent(table).setShape(
        ry.ST.box, size=[0.4, 0.4, 0.06, 0.005]
    ).setColor([1, 0.5, 0, 1]).setContact(1).setRelativePosition(
        [+0.5, +0.5, 0.07]
    ).setJoint(ry.JT.rigid)

    C.addFrame("obj2").setParent(table).setShape(
        ry.ST.box, size=[0.3, 0.4, 0.06, 0.005]
    ).setColor([0.5, 0.5, 0, 1]).setContact(1).setRelativePosition(
        [0.5, -0.5, 0.07]
    ).setJoint(ry.JT.rigid)

    C.addFrame("goal1").setParent(table).setShape(
        ry.ST.box, size=[0.2, 0.2, 0.06, 0.005]
    ).setColor([1, 0.5, 0, 0.3]).setContact(0).setRelativePosition([-0.5, -0.5, 0.07])

    C.addFrame("goal2").setParent(table).setShape(
        ry.ST.box, size=[0.2, 0.2, 0.06, 0.005]
    ).setColor([0.5, 0.5, 0, 0.2]).setContact(0).setRelativePosition([-0.5, 0.5, 0.07])

    C.addFrame("obs1").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0.7, 0, 0.07]
    ).setShape(ry.ST.box, size=[0.6, 0.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("obs2").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [-0.7, 0, 0.07]
    ).setShape(ry.ST.box, size=[0.6, 0.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    if view:
        C.view(True)

    def pick_and_place(agent, object, goal):
        c_tmp = ry.Config()
        c_tmp.addConfigurationCopy(C)

        c_tmp.selectJointsBySubtree(c_tmp.getFrame(agent))

        q_home = c_tmp.getJointState()

        komo = ry.KOMO(
            c_tmp, phases=3, slicesPerPhase=1, kOrder=1, enableCollisions=True
        )
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [0.1])
        komo.addControlObjective([], 0, 1e-1)
        # komo.addControlObjective([], 1, 1e0)
        # komo.addControlObjective([], 2, 1e0)

        komo.addModeSwitch([1, 2], ry.SY.stable, [agent, object])
        komo.addObjective([1, 2], ry.FS.distance, [agent, object], ry.OT.eq, [1e1])
        komo.addObjective([2, -1], ry.FS.poseDiff, [object, goal], ry.OT.eq, [1e1])
        komo.addModeSwitch([2, -1], ry.SY.stable, ["table", object])

        # komo.addModeSwitch([1, 2], ry.SY.stable, ["a2", "obj2"])
        # komo.addObjective([1, 2], ry.FS.distance, ["a2", "obj2"], ry.OT.eq, [1e1])
        # komo.addModeSwitch([2, -1], ry.SY.stable, ["table", "obj2"])
        # komo.addObjective([2, -1], ry.FS.poseDiff, ["obj2", "goal2"], ry.OT.eq, [1e1])

        komo.addObjective(
            times=[3],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.eq,
            scale=[1e0],
            target=q_home,
        )

        # komo.addObjective([2], ry.FS.poseDiff, ["a2", "goal2"], ry.OT.eq, [1e1])

        # komo.addObjective([3, -1], ry.FS.poseDiff, ['a1', 'goal2'], ry.OT.eq, [1e1])
        # komo.addObjective(
        #     [3, -1], ry.FS.poseDiff, ["a2", "pre_agent_2_frame"], ry.OT.eq, [1e1]
        # )

        solver = ry.NLP_Solver(komo.nlp(), verbose=4)
        # options.nonStrictSteps = 50;

        solver.setOptions(damping=0.01, wolfe=0.001)
        solver.solve()

        if view:
            komo.view(True, "IK solution")

        keyframes = komo.getPath()
        print(keyframes)
        return keyframes

    all_keyframes = []

    for agent in ["a1", "a2"]:
        for object, goal in zip(["obj1", "obj2"], ["goal1", "goal2"]):
            keyframes = pick_and_place(agent, object, goal)
            all_keyframes.append((agent, object, keyframes))

    return C, all_keyframes


class rai_unassigned_piano_mover(FreeMixin, rai_env):
    def __init__(self, agents_can_rotate=True):
        self.C, all_keyframes = make_piano_mover_env()
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        self.manipulating_env = True

        self.tasks = [
            # Task(
            #     ["a1"],
            #     GoalRegion(self.limits[:, :3]),
            # ),
            # Task(
            #     ["a2"],
            #     GoalRegion(self.limits[:, 3:]),
            # ),
            Task(
                ["a1", "a2"],
                # GoalRegion(self.limits),
                SingleGoal(self.C.getJointState())
            ),
        ]

        self.task_dependencies = {}
        self.task_dependencies_any = {}

        pick_tasks = {"obj1": [], "obj2": []}
        place_tasks = {"obj1": [], "obj2": []}

        for agent, object, keyframes in all_keyframes:
            self.tasks.append(
                Task(
                    [agent],
                    SingleGoal(keyframes[0, :]),
                    type="pick",
                    frames=[agent, object],
                ),
            )
            self.tasks.append(
                Task(
                    [agent],
                    SingleGoal(keyframes[1, :]),
                    type="place",
                    frames=["table", object],
                )
            )

            pick_tasks[object].append((self.robots.index(agent), len(self.tasks) - 2))
            place_tasks[object].append((self.robots.index(agent), len(self.tasks) - 1))

            # place is dependent on pick
            self.task_dependencies[len(self.tasks) - 1] = [len(self.tasks) - 2]

            self.tasks[-2].name = f"pick_{agent}_{object}"
            self.tasks[-1].name = f"place_{agent}_{object}"

        # self.tasks[0].name = "dummy_start"
        # self.tasks[1].name = "a1_goal"
        # self.tasks[2].name = "a2_goal_0"
        # self.tasks[3].name = "a2_goal_1"
        # self.tasks[4].name = "a2_goal_2"
        # self.tasks[5].name = "a2_goal_3"
        # self.tasks[6].name = "terminal"

        self.task_groups = []

        for k, v in pick_tasks.items():
            self.task_groups.append(v)
        for k, v in place_tasks.items():
            self.task_groups.append(v)

        self.tasks.append(
            # terminal mode
            Task(
                ["a1", "a2"],
                SingleGoal(self.C.getJointState()),
            ),
        )
        self.terminal_task = len(self.tasks) - 1

        self.collision_tolerance = 0.01

        BaseModeLogic.__init__(self)


class rai_unassigned_pile_cleanup(FreeMixin, rai_env):
    def __init__(self, num_boxes = 5):
        self.C, keyframes = rai_config.make_box_pile_env(
            num_boxes=num_boxes, random_orientation=False, compute_all_keyframes=True
        )

        self.robots = ["a1_", "a2_"]

        rai_env.__init__(self)

        self.manipulating_env = True

        self.tasks = [
            Task(
                ["a1_", "a2_"],
                # GoalRegion(self.limits),
                SingleGoal(self.C.getJointState())
            ),
        ]
        self.tasks[-1].name = "dummy_start"

        pick_task_names = ["pick", "place"]
        handover_task_names = ["pick", "handover", "place"]

        self.task_dependencies = {}
        self.task_dependencies_any = {}

        pick_tasks = {}
        place_tasks = {}

        cnt = 0
        for primitive_type, robots, box_index, qs in keyframes:
            box_name = "obj" + str(box_index)

            if box_name not in pick_tasks:
                pick_tasks[box_name] = []
                place_tasks[box_name] = []

            robot_index = 0
            if robots[0] == "a2_":
                robot_index = 1
            
            # print("robot index", robot_index)

            print(primitive_type)
            if primitive_type == "pick":
                for t, k in zip(pick_task_names, qs[0]):
                    print(robots)
                    print(k)
                    if t == "pick":
                        ee_name = robots[0] + "ur_vacuum"
                        self.tasks.append(
                            Task(robots, SingleGoal(k), t, frames=[ee_name, box_name])
                        )
                        pick_tasks[box_name].append((robot_index, len(self.tasks) - 1))
                        
                    else:
                        self.tasks.append(
                            Task(robots, SingleGoal(k), t, frames=["tray", box_name])
                        )
                        self.task_dependencies[len(self.tasks) - 1] = [
                            len(self.tasks) - 2
                        ]
                        place_tasks[box_name].append((robot_index, len(self.tasks) - 1))

                    self.tasks[-1].name = (
                        robots[0] + t + "_" + box_name + "_" + str(cnt)
                    )
                    cnt += 1
            else:
                assert False

        self.task_groups = []

        for k, v in pick_tasks.items():
            self.task_groups.append(v)
        for k, v in place_tasks.items():
            self.task_groups.append(v)

        print(self.task_groups)
        print(self.task_dependencies)

        self.tasks.append(Task(self.robots, SingleGoal(self.C.getJointState())))
        self.tasks[-1].name = "terminal"

        self.terminal_task = len(self.tasks) - 1

        BaseModeLogic.__init__(self)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.00
        self.collision_resolution = 0.01

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        dim = 6
        for i, r in enumerate(self.robots):
            print(self.C.getJointState()[0:6])
            self.safe_pose[r] = np.array(self.C.getJointState()[dim*i:dim*(i+1)])


class rai_unassigned_stacking(FreeMixin, rai_env):
    def __init__(self, num_robots=4, num_boxes: int = 8):
        self.C, keyframes, self.robots = rai_config.make_box_stacking_env(
            num_robots, num_boxes, make_and_return_all_keyframes=True
        )

        rai_env.__init__(self)

        self.manipulating_env = True

        self.tasks = [
            Task(
                self.robots,
                # GoalRegion(self.limits),
                SingleGoal(self.C.getJointState())
            ),
        ]
        self.tasks[-1].name = "dummy_start"

        pick_task_names = ["pick", "place"]

        self.task_dependencies = {}

        pick_tasks = {}
        place_tasks = {}

        box_order = []
        for _, box_name, _ in keyframes:
            if len(box_order) == 0 or box_order[-1] != box_name:
                box_order.append(box_name)

        for r, box_name, qs in keyframes:
            if box_name not in pick_tasks:
                pick_tasks[box_name] = []
                place_tasks[box_name] = []

            robot_index = self.robots.index(r)
            print(robot_index)
            
            cnt = 0
            for t, k in zip(pick_task_names, qs):
                if t == "pick":
                    ee_name = r + "gripper_center"
                    self.tasks.append(Task([r], SingleGoal(k), t, frames=[ee_name, box_name]))
                    pick_tasks[box_name].append((robot_index, len(self.tasks) - 1))

                else:
                    self.tasks.append(Task([r], SingleGoal(k), t, frames=["table", box_name]))
                    place_tasks[box_name].append((robot_index, len(self.tasks) - 1))
                    self.task_dependencies[len(self.tasks) - 1] = [
                        len(self.tasks) - 2
                    ]

                self.tasks[-1].name = r + t + "_" + box_name + "_" + str(cnt)
                cnt += 1

                # if b in action_names:
                #     action_names[b].append(self.tasks[-1].name)
                # else:
                #     action_names[b] = [self.tasks[-1].name]
        self.task_dependencies_any = {}

        for i, box_name in enumerate(box_order):
            if i == 0:
                continue
            
            all_place_task_ids = place_tasks[box_name]
            all_prev_place_task_ids = place_tasks[box_order[i-1]]

            for r_id, task_id in all_place_task_ids:
                self.task_dependencies_any[task_id] = [task_id_others for _, task_id_others in all_prev_place_task_ids]

        self.task_groups = []

        for k, v in pick_tasks.items():
            self.task_groups.append(v)
        for k, v in place_tasks.items():
            self.task_groups.append(v)

        print("task groups", self.task_groups)
        print("dependencies", self.task_dependencies)

        print("Dependencies")
        for k, v in self.task_dependencies.items():
            print(self.tasks[k].name)
            for idx in v:
                print(self.tasks[idx].name)
            
            print()

        print("Dependencies any")
        for k, v in self.task_dependencies_any.items():
            print(self.tasks[k].name)
            for idx in v:
                print(self.tasks[idx].name)
            
            print()

        self.tasks.append(Task(self.robots, SingleGoal(self.C.getJointState())))
        self.tasks[-1].name = "terminal"

        self.terminal_task = len(self.tasks) - 1

        BaseModeLogic.__init__(self)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.00
        self.collision_resolution = 0.0025

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        dim = 6
        for i, r in enumerate(self.robots):
            print(self.C.getJointState()[0:6])
            self.safe_pose[r] = np.array(self.C.getJointState()[dim*i:dim*(i+1)])