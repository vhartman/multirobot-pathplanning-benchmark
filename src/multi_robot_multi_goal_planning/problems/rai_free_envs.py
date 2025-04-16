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
