import robotic as ry
import numpy as np
import argparse

from typing import List

import os.path
import random
import json

# make everything predictable
# np.random.seed(2)
# random.seed(2)


def get_robot_joints(C: ry.Config, prefix: str) -> List[str]:
    links = []

    for name in C.getJointNames():
        if prefix in name:
            name = name.split(":")[0]

            if name not in links:
                links.append(name)

    return links


def make_table_with_walls(width=2, length=2):
    C = ry.Config()

    table = (
        C.addFrame("table")
        .setPosition([0, 0, 1.0])
        .setShape(ry.ST.box, size=[width, length, 0.06, 0.005])
        .setColor([0.6, 0.6, 0.6])
        .setContact(1)
    )

    C.addFrame("wall1").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0, width / 2 + 0.1, 0.07]
    ).setShape(ry.ST.box, size=[width - 0.001, 0.2, 0.06, 0.005]).setContact(
        1
    ).setColor([0, 0, 0]).setJoint(ry.JT.rigid)

    C.addFrame("wall2").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0, -width / 2 - 0.1, 0.07]
    ).setShape(ry.ST.box, size=[width - 0.001, 0.2, 0.06, 0.005]).setContact(
        1
    ).setColor([0, 0, 0]).setJoint(ry.JT.rigid)

    C.addFrame("wall3").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [length / 2 + 0.1, 0, 0.07]
    ).setShape(ry.ST.box, size=[0.2, length + 0.2 * 2 - 0.001, 0.06, 0.005]).setContact(
        1
    ).setColor([0, 0, 0]).setJoint(ry.JT.rigid)

    C.addFrame("wall4").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [-length / 2 - 0.1, 0, 0.07]
    ).setShape(ry.ST.box, size=[0.2, length + 0.2 * 2 - 0.001, 0.06, 0.005]).setContact(
        1
    ).setColor([0, 0, 0]).setJoint(ry.JT.rigid)

    return C


def make_2d_rai_env_no_obs(view: bool = False, agents_can_rotate=True):
    if not isinstance(agents_can_rotate, list):
        agents_can_rotate = [agents_can_rotate] * 2
    else:
        assert len(agents_can_rotate) == 2

    C = make_table_with_walls(2, 2)
    table = C.getFrame("table")

    pre_agent_1_frame = (
        C.addFrame("pre_agent_1_frame")
        .setParent(table)
        # .setPosition(table.getPosition() + [0.0, -0.5, 0.07])
        .setPosition(table.getPosition() + [0.0, 0.0, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    if agents_can_rotate[0]:
        C.addFrame("a1").setParent(pre_agent_1_frame).setShape(
            ry.ST.cylinder, size=[0.1, 0.2, 0.06, 0.15]
        ).setColor([1, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXYPhi, limits=np.array([-1, 1, -1, 1, -3.14, 3.14])
        ).setJointState([-0.5, -0.5, 0])
    else:
        C.addFrame("a1").setParent(pre_agent_1_frame).setShape(
            ry.ST.cylinder, size=[0.1, 0.2, 0.06, 0.15]
        ).setColor([1, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXY, limits=np.array([-1, 1, -1, 1, -3.14, 3.14])
        ).setJointState([-0.5, -0.5])

    pre_agent_2_frame = (
        C.addFrame("pre_agent_2_frame")
        .setParent(table)
        # .setPosition(table.getPosition() + [0, 0.5, 0.07])
        .setPosition(table.getPosition() + [0, 0.0, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    if agents_can_rotate[1]:
        C.addFrame("a2").setParent(pre_agent_2_frame).setShape(
            ry.ST.box,
            size=[0.1, 0.2, 0.06, 0.005],
            # ry.ST.cylinder, size=[4, 0.1, 0.06, 0.075]
        ).setColor([0.5, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXYPhi, limits=np.array([-1, 1, -1, 1, -3.14, 3.14])
        ).setJointState([0.5, 0.5, 0])
    else:
        C.addFrame("a2").setParent(pre_agent_2_frame).setShape(
            ry.ST.box,
            size=[0.1, 0.2, 0.06, 0.005],
            # ry.ST.cylinder, size=[4, 0.1, 0.06, 0.075]
        ).setColor([0.5, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXY, limits=np.array([-1, 1, -1, 1, -3.14, 3.14])
        ).setJointState([0.5, 0.5])

    return C


def make_2d_rai_env_no_obs_three_agents(view: bool = False, agents_can_rotate=True):
    if not isinstance(agents_can_rotate, list):
        agents_can_rotate = [agents_can_rotate] * 3
    else:
        assert len(agents_can_rotate) == 3

    C = make_table_with_walls(2, 2)
    table = C.getFrame("table")

    pre_agent_1_frame = (
        C.addFrame("pre_agent_1_frame")
        .setParent(table)
        # .setPosition(table.getPosition() + [0.0, -0.5, 0.07])
        .setPosition(table.getPosition() + [0.0, 0.0, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    if agents_can_rotate[0]:
        C.addFrame("a1").setParent(pre_agent_1_frame).setShape(
            ry.ST.cylinder, size=[0.1, 0.2, 0.06, 0.15]
        ).setColor([1, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXYPhi, limits=np.array([-1, 1, -1, 1, -3.14, 3.14])
        ).setJointState([-0.5, -0.5, 0])
    else:
        C.addFrame("a1").setParent(pre_agent_1_frame).setShape(
            ry.ST.cylinder, size=[0.1, 0.2, 0.06, 0.15]
        ).setColor([1, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXY, limits=np.array([-1, 1, -1, 1, -3.14, 3.14])
        ).setJointState([-0.5, -0.5])

    pre_agent_2_frame = (
        C.addFrame("pre_agent_2_frame")
        .setParent(table)
        # .setPosition(table.getPosition() + [0, 0.5, 0.07])
        .setPosition(table.getPosition() + [0, 0.0, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    if agents_can_rotate[1]:
        C.addFrame("a2").setParent(pre_agent_2_frame).setShape(
            ry.ST.box,
            size=[0.1, 0.2, 0.06, 0.005],
            # ry.ST.cylinder, size=[4, 0.1, 0.06, 0.075]
        ).setColor([0.5, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXYPhi, limits=np.array([-1, 1, -1, 1, -3.14, 3.14])
        ).setJointState([0.5, 0.5, 0])
    else:
        C.addFrame("a2").setParent(pre_agent_2_frame).setShape(
            ry.ST.box,
            size=[0.1, 0.2, 0.06, 0.005],
            # ry.ST.cylinder, size=[4, 0.1, 0.06, 0.075]
        ).setColor([0.5, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXY, limits=np.array([-1, 1, -1, 1, -3.14, 3.14])
        ).setJointState([0.5, 0.5])

    pre_agent_3_frame = (
        C.addFrame("pre_agent_3_frame")
        .setParent(table)
        # .setPosition(table.getPosition() + [0, 0.5, 0.07])
        .setPosition(table.getPosition() + [0, 0.0, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    if agents_can_rotate[2]:
        C.addFrame("a3").setParent(pre_agent_3_frame).setShape(
            ry.ST.box,
            size=[0.1, 0.2, 0.06, 0.005],
            # ry.ST.cylinder, size=[4, 0.1, 0.06, 0.075]
        ).setColor([0.5, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXYPhi, limits=np.array([-1, 1, -1, 1, -3.14, 3.14])
        ).setJointState([0.0, 0.5, 0])
    else:
        C.addFrame("a3").setParent(pre_agent_3_frame).setShape(
            ry.ST.box,
            size=[0.1, 0.2, 0.06, 0.005],
            # ry.ST.cylinder, size=[4, 0.1, 0.06, 0.075]
        ).setColor([0.5, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXY, limits=np.array([-1, 1, -1, 1, -3.14, 3.14])
        ).setJointState([0.0, 0.5])

    return C


def make_2d_rai_env(view: bool = False, agents_can_rotate=True):
    if not isinstance(agents_can_rotate, list):
        agents_can_rotate = [agents_can_rotate] * 2
    else:
        assert len(agents_can_rotate) == 2

    C = make_table_with_walls(2, 2)
    table = C.getFrame("table")

    pre_agent_1_frame = (
        C.addFrame("pre_agent_1_frame")
        .setParent(table)
        # .setPosition(table.getPosition() + [0.0, -0.5, 0.07])
        .setPosition(table.getPosition() + [0.0, 0.0, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    if agents_can_rotate[0]:
        C.addFrame("a1").setParent(pre_agent_1_frame).setShape(
            ry.ST.cylinder, size=[0.1, 0.2, 0.06, 0.15]
        ).setColor([1, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXYPhi, limits=np.array([-1, 1, -1, 1, -3.14, 3.14])
        ).setJointState([0, -0.5, 0])
    else:
        C.addFrame("a1").setParent(pre_agent_1_frame).setShape(
            ry.ST.cylinder, size=[0.1, 0.2, 0.06, 0.15]
        ).setColor([1, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXY, limits=np.array([-1, 1, -1, 1, -3.14, 3.14])
        ).setJointState([0, -0.5])

    pre_agent_2_frame = (
        C.addFrame("pre_agent_2_frame")
        .setParent(table)
        # .setPosition(table.getPosition() + [0, 0.5, 0.07])
        .setPosition(table.getPosition() + [0, 0.0, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    if agents_can_rotate[1]:
        C.addFrame("a2").setParent(pre_agent_2_frame).setShape(
            ry.ST.box,
            size=[0.1, 0.2, 0.06, 0.005],
            # ry.ST.cylinder, size=[4, 0.1, 0.06, 0.075]
        ).setColor([0.5, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXYPhi, limits=np.array([-1, 1, -1, 1, -3.14, 3.14])
        ).setJointState([0, 0.5, 0])
    else:
        C.addFrame("a2").setParent(pre_agent_2_frame).setShape(
            ry.ST.box,
            size=[0.1, 0.2, 0.06, 0.005],
            # ry.ST.cylinder, size=[4, 0.1, 0.06, 0.075]
        ).setColor([0.5, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXY, limits=np.array([-1, 1, -1, 1, -3.14, 3.14])
        ).setJointState([0, 0.5])

    C.addFrame("goal1").setParent(table).setShape(
        ry.ST.box, size=[0.2, 0.2, 0.06, 0.005]
    ).setColor([1, 0.5, 0, 0.3]).setContact(0).setRelativePosition([+0.5, +0.5, 0.07])

    C.addFrame("goal2").setParent(table).setShape(
        ry.ST.box, size=[0.2, 0.2, 0.06, 0.005]
    ).setColor([0.5, 0.5, 0, 0.2]).setContact(0).setRelativePosition([-0.5, -0.5, 0.07])

    C.addFrame("obs1").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0.75, 0, 0.07]
    ).setShape(ry.ST.box, size=[0.5, 0.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("obs2").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [-0.75, 0, 0.07]
    ).setShape(ry.ST.box, size=[0.5, 0.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("obs3").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0.1, 0, 0.07]
    ).setShape(ry.ST.box, size=[0.3, 0.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    # pairs = C.getCollidablePairs()

    # for i in range(0, len(pairs), 2):
    #     print(pairs[i], pairs[i + 1])

    if view:
        C.view(True)

    komo = ry.KOMO(C, phases=3, slicesPerPhase=1, kOrder=1, enableCollisions=True)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])
    komo.addControlObjective([], 0, 1e-1)
    # komo.addControlObjective([], 1, 1e0)
    # komo.addControlObjective([], 2, 1e0)

    komo.addObjective([1, -1], ry.FS.poseDiff, ["a1", "goal1"], ry.OT.eq, [1e1])
    komo.addObjective([2], ry.FS.poseDiff, ["a2", "goal2"], ry.OT.eq, [1e1])

    # komo.addObjective([3, -1], ry.FS.poseDiff, ['a1', 'goal2'], ry.OT.eq, [1e1])
    komo.addObjective(
        [3, -1],
        ry.FS.positionDiff,
        ["a2", "pre_agent_2_frame"],
        ry.OT.eq,
        [1e1],
        [0, 0.5, 0],
    )

    solver = ry.NLP_Solver(komo.nlp(), verbose=4)
    # options.nonStrictSteps = 50;

    solver.setOptions(damping=0.01, wolfe=0.001)
    solver.solve()

    if view:
        komo.view(True, "IK solution")

    keyframes = komo.getPath()
    # print(keyframes)

    return C, keyframes


def make_random_two_dim(
    num_agents: int = 3,
    num_obstacles: int = 5,
    num_goals: int = 3,
    agents_can_rotate=True,
    view: bool = False,
):
    if not isinstance(agents_can_rotate, list):
        agents_can_rotate = [agents_can_rotate] * num_agents
    else:
        assert len(agents_can_rotate) == num_agents

    C = make_table_with_walls(4, 4)

    added_agents = 0
    agent_names = []

    colors = []

    while added_agents < num_agents:
        c_coll_tmp = ry.Config()
        c_coll_tmp.addConfigurationCopy(C)

        pos = np.random.rand(2) * 4 - 2
        rot = np.random.rand() * 6 - 3
        size = np.random.rand(2) * 0.2 + 0.3
        color = np.random.rand(3)

        pre_agent_1_frame = (
            c_coll_tmp.addFrame(f"pre_agent_{added_agents}_frame")
            .setParent(c_coll_tmp.getFrame("table"))
            .setPosition(c_coll_tmp.getFrame("table").getPosition() + [0, 0, 0.07])
            .setShape(ry.ST.marker, size=[0.05])
            .setContact(0)
            .setJoint(ry.JT.rigid)
        )

        if agents_can_rotate[added_agents]:
            c_coll_tmp.addFrame(f"a{added_agents}").setParent(
                pre_agent_1_frame
            ).setShape(
                # ry.ST.box, size=[size[0], size[1], 0.06, 0.2]
                ry.ST.cylinder,
                size=[4, 0.1, 0.06, 0.2],
            ).setColor(color).setContact(1).setJoint(
                ry.JT.transXYPhi, limits=np.array([-2, 2, -2, 2, -3.14, 3.14])
            ).setJointState([pos[0], pos[1], rot])
        else:
            c_coll_tmp.addFrame(f"a{added_agents}").setParent(
                pre_agent_1_frame
            ).setShape(
                # ry.ST.box, size=[size[0], size[1], 0.06, 0.2]
                ry.ST.cylinder,
                size=[4, 0.1, 0.06, 0.2],
            ).setColor(color).setContact(1).setJoint(
                ry.JT.transXY, limits=np.array([-2, 2, -2, 2, -3.14, 3.14])
            ).setJointState([pos[0], pos[1]])

        binary_collision_free = c_coll_tmp.getCollisionFree()
        if binary_collision_free:
            agent_names.append(f"a{added_agents}")
            added_agents += 1

            C.clear()
            C.addConfigurationCopy(c_coll_tmp)

            colors.append(color)
        else:
            colls = c_coll_tmp.getCollisions()
            for c in colls:
                if c[2] < 0:
                    print(c)

    added_obstacles = 0

    while added_obstacles < num_obstacles:
        c_coll_tmp = ry.Config()
        c_coll_tmp.addConfigurationCopy(C)

        pos = np.random.rand(2) * 4 - 2
        size = np.random.rand(2) * 0.5 + 0.2

        c_coll_tmp.addFrame(f"obs{added_obstacles}").setParent(
            c_coll_tmp.getFrame("table")
        ).setPosition(
            c_coll_tmp.getFrame("table").getPosition() + [pos[0], pos[1], 0.07]
        ).setShape(ry.ST.box, size=[size[0], size[1], 0.06, 0.005]).setContact(
            -2
        ).setColor([0, 0, 0]).setJoint(ry.JT.rigid)

        binary_collision_free = c_coll_tmp.getCollisionFree()
        if binary_collision_free:
            added_obstacles += 1

            C.clear()
            C.addConfigurationCopy(c_coll_tmp)
        # else:
        #     colls = c_coll_tmp.getCollisions()
        #     for c in colls:
        #         if c[2] < 0:
        #             print(c)

        #     c_coll_tmp.view(True)

    keyframes = []

    for i, agent in enumerate(agent_names):
        added_goals = 0
        while added_goals < num_goals:
            pos = np.random.rand(2) * 4 - 2
            rot = np.random.rand() * 6 - 3

            c_coll_tmp = ry.Config()
            c_coll_tmp.addConfigurationCopy(C)

            if agents_can_rotate[i]:
                q = np.array([pos[0], pos[1], rot])
            else:
                q = np.array([pos[0], pos[1]])

            c_coll_tmp.setJointState(q, get_robot_joints(c_coll_tmp, agent))

            binary_collision_free = c_coll_tmp.getCollisionFree()
            # c_coll_tmp.view(True)

            if binary_collision_free:
                color = colors[i]
                C.addFrame(f"goal_{agent}_{added_goals}").setParent(
                    C.getFrame("table")
                ).setShape(ry.ST.box, size=[0.1, 0.1, 0.06, 0.005]).setColor(
                    [color[0], color[1], color[2], 0.3]
                ).setContact(0).setRelativePosition([pos[0], pos[1], 0.07])

                keyframes.append(q)
                added_goals += 1

    if view:
        C.view(True)

    return C, keyframes


def make_random_two_dim_single_goal(
    num_agents: int = 3,
    num_obstacles: int = 5,
    agents_can_rotate=True,
    view: bool = False,
):
    if not isinstance(agents_can_rotate, list):
        agents_can_rotate = [agents_can_rotate] * num_agents
    else:
        assert len(agents_can_rotate) == num_agents

    C = make_table_with_walls(4, 4)

    added_agents = 0
    agent_names = []

    colors = []

    while added_agents < num_agents:
        c_coll_tmp = ry.Config()
        c_coll_tmp.addConfigurationCopy(C)

        pos = np.random.rand(2) * 4 - 2
        rot = np.random.rand() * 6 - 3
        size = np.random.rand(2) * 0.2 + 0.3
        color = np.random.rand(3)

        pre_agent_1_frame = (
            c_coll_tmp.addFrame(f"pre_agent_{added_agents}_frame")
            .setParent(c_coll_tmp.getFrame("table"))
            .setPosition(c_coll_tmp.getFrame("table").getPosition() + [0, 0, 0.07])
            .setShape(ry.ST.marker, size=[0.05])
            .setContact(0)
            .setJoint(ry.JT.rigid)
        )

        if agents_can_rotate[added_agents]:
            c_coll_tmp.addFrame(f"a{added_agents}").setParent(
                pre_agent_1_frame
            ).setShape(
                # ry.ST.box, size=[size[0], size[1], 0.06, 0.2]
                ry.ST.cylinder,
                size=[4, 0.1, 0.06, 0.2],
            ).setColor(color).setContact(1).setJoint(
                ry.JT.transXYPhi, limits=np.array([-2, 2, -2, 2, -3.14, 3.14])
            ).setJointState([pos[0], pos[1], rot])
        else:
            c_coll_tmp.addFrame(f"a{added_agents}").setParent(
                pre_agent_1_frame
            ).setShape(
                # ry.ST.box, size=[size[0], size[1], 0.06, 0.2]
                ry.ST.cylinder,
                size=[4, 0.1, 0.06, 0.2],
            ).setColor(color).setContact(1).setJoint(
                ry.JT.transXY, limits=np.array([-2, 2, -2, 2, -3.14, 3.14])
            ).setJointState([pos[0], pos[1]])

        binary_collision_free = c_coll_tmp.getCollisionFree()
        if binary_collision_free:
            agent_names.append(f"a{added_agents}")
            added_agents += 1

            C.clear()
            C.addConfigurationCopy(c_coll_tmp)

            colors.append(color)
        else:
            colls = c_coll_tmp.getCollisions()
            for c in colls:
                if c[2] < 0:
                    print(c)

    added_obstacles = 0

    while added_obstacles < num_obstacles:
        c_coll_tmp = ry.Config()
        c_coll_tmp.addConfigurationCopy(C)

        pos = np.random.rand(2) * 4 - 2
        size = np.random.rand(2) * 0.5 + 0.2

        c_coll_tmp.addFrame(f"obs{added_obstacles}").setParent(
            c_coll_tmp.getFrame("table")
        ).setPosition(
            c_coll_tmp.getFrame("table").getPosition() + [pos[0], pos[1], 0.07]
        ).setShape(ry.ST.box, size=[size[0], size[1], 0.06, 0.005]).setContact(
            -2
        ).setColor([0, 0, 0]).setJoint(ry.JT.rigid)

        binary_collision_free = c_coll_tmp.getCollisionFree()
        if binary_collision_free:
            added_obstacles += 1

            C.clear()
            C.addConfigurationCopy(c_coll_tmp)

    keyframes = []

    while True:
        c_coll_tmp = ry.Config()
        c_coll_tmp.addConfigurationCopy(C)

        for i, agent in enumerate(agent_names):
            pos = np.random.rand(2) * 4 - 2
            rot = np.random.rand() * 6 - 3

            if agents_can_rotate[i]:
                q = np.array([pos[0], pos[1], rot])
            else:
                q = np.array([pos[0], pos[1]])

            c_coll_tmp.setJointState(q, get_robot_joints(c_coll_tmp, agent))

        binary_collision_free = c_coll_tmp.getCollisionFree()
        # c_coll_tmp.view(True)

        if binary_collision_free:
            keyframes.append(c_coll_tmp.getJointState())
            break

    if view:
        C.view(True)

    return C, keyframes


def make_two_dim_handover(view: bool = False):
    C = make_table_with_walls(4, 4)
    table = C.getFrame("table")

    pre_agent_1_frame = (
        C.addFrame("pre_agent_1_frame")
        .setParent(table)
        .setPosition(table.getPosition() + [0, 0.0, 0.07])
        # .setPosition(table.getPosition() + [-0.5, 0.8, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    C.addFrame("a1").setParent(pre_agent_1_frame).setShape(
        ry.ST.cylinder, size=[0.04, 0.15]
    ).setContact(1).setJoint(
        ry.JT.transXYPhi, limits=np.array([-2, 2, -2, 2, -3.14, 3.14])
    ).setJointState([-0.5, 0.8, 0])

    C.addFrame("a1_dir").setParent(C.getFrame("a1")).setShape(
        ry.ST.box, size=[0.04, 0.15, 0.05]
    ).setColor([0, 0, 0]).setContact(0).setRelativePosition([0, 0, 0.0])

    pre_agent_2_frame = (
        C.addFrame("pre_agent_2_frame")
        .setParent(table)
        .setPosition(table.getPosition() + [0.0, 0.0, 0.07])
        # .setPosition(table.getPosition() + [0.0, -0.5, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    C.addFrame("a2").setParent(pre_agent_2_frame).setShape(
        ry.ST.cylinder, size=[0.04, 0.2]
    ).setColor([0.7, 0.7, 0.7]).setContact(1).setJoint(
        ry.JT.transXYPhi, limits=np.array([-2, 2, -2, 2, -3.14, 3.14])
    ).setJointState([0.0, -0.5, 0])

    C.addFrame("a2_dir").setParent(C.getFrame("a2")).setShape(
        ry.ST.box, size=[0.04, 0.2, 0.05]
    ).setColor([0, 0, 0]).setContact(0).setRelativePosition([0, 0, 0.0])

    C.addFrame("obj1").setParent(table).setShape(
        ry.ST.box, size=[0.4, 0.4, 0.06, 0.005]
    ).setColor([1, 0.5, 0, 1]).setContact(1).setRelativePosition(
        [0, 0.4, 0.07]
    ).setJoint(ry.JT.rigid)

    C.addFrame("obj2").setParent(table).setShape(
        ry.ST.box, size=[0.3, 0.4, 0.06, 0.005]
    ).setColor([0.5, 0.5, 0, 1]).setContact(1).setRelativePosition(
        [0.5, -1.5, 0.07]
    ).setJoint(ry.JT.rigid)

    C.addFrame("goal1").setParent(table).setShape(
        ry.ST.box, size=[0.4, 0.4, 0.06, 0.005]
    ).setColor([1, 0.5, 0, 0.3]).setContact(0).setRelativePosition([0.8, 0.4, 0.07])

    C.addFrame("goal2").setParent(table).setShape(
        ry.ST.box, size=[0.3, 0.4, 0.06, 0.005]
    ).setColor([0.5, 0.5, 0, 0.2]).setContact(0).setRelativePosition([0.8, 1.2, 0.07])

    C.addFrame("obs1").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0.0, 0, 0.07]
    ).setShape(ry.ST.box, size=[2.3, 0.2, 0.06, 0.005]).setContact(-2).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("obs2").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0.4, 1.05, 0.07]
    ).setShape(ry.ST.box, size=[0.2, 1.8, 0.06, 0.005]).setContact(-2).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("obs3").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [-0.4, -0.6, 0.07]
    ).setShape(ry.ST.box, size=[0.2, 0.9, 0.06, 0.005]).setContact(-2).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("obs4").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0.8, 0.8, 0.07]
    ).setShape(ry.ST.box, size=[0.6, 0.2, 0.06, 0.005]).setContact(-2).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    if view:
        C.view(True)

    def compute_handover():
        box = "obj1"

        q_home = C.getJointState()

        komo = ry.KOMO(C, phases=4, slicesPerPhase=1, kOrder=1, enableCollisions=True)
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1], [-0.0])

        komo.addControlObjective([], 0, 1e-1)
        # komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        komo.addModeSwitch([1, 2], ry.SY.stable, ["a1", box])
        komo.addObjective([1, 2], ry.FS.distance, ["a1", box], ry.OT.sos, [1e1], [-0.0])
        komo.addObjective(
            [1, 2],
            ry.FS.positionDiff,
            ["a1", box],
            ry.OT.sos,
            [1e0, 1e0, 1e0],
        )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.positionDiff,
        #     ["a1_" + "ur_ee_marker", box],
        #     ry.OT.sos,
        #     [1e0],
        # )

        # komo.addObjective(
        #     [2], ry.FS.position, ["a2"], ry.OT.sos, [1e0, 1e1, 0], [1., -0.5, 0]
        # )

        komo.addObjective(
            [2], ry.FS.position, [box], ry.OT.sos, [1e0, 1e0, 0], [1, -1, 0]
        )

        komo.addModeSwitch([2, 3], ry.SY.stable, ["a2", box])
        komo.addObjective([2, 3], ry.FS.distance, ["a2", box], ry.OT.sos, [1e1], [-0.0])
        komo.addObjective(
            [2, 3],
            ry.FS.positionDiff,
            ["a2", box],
            ry.OT.sos,
            [1e0, 1e0, 0e0],
        )
        # komo.addObjective(
        #     [2, 3],
        #     ry.FS.positionDiff,
        #     ["a2_" + "ur_ee_marker", box],
        #     ry.OT.sos,
        #     [1e0],
        # )

        komo.addModeSwitch([3, -1], ry.SY.stable, ["table", box])
        komo.addObjective([3, -1], ry.FS.poseDiff, ["goal1", box], ry.OT.eq, [1e1])

        komo.addObjective(
            times=[4],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.eq,
            scale=[1e0],
            target=q_home,
        )

        for _ in range(100):
            # komo.initRandom()
            komo.initWithConstant(np.random.rand(6) * 2)

            solver = ry.NLP_Solver(komo.nlp(), verbose=4)
            # options.nonStrictSteps = 50;

            # solver.setOptions(damping=0.01, wolfe=0.001)
            # solver.setOptions(damping=0.001)
            retval = solver.solve()
            retval = retval.dict()

            print(retval)

            if view:
                komo.view(True, "IK solution")

            keyframes = komo.getPath()

            # print(retval)

            if retval["ineq"] < 1 and retval["eq"] < 1 and retval["feasible"]:
                return keyframes

    def compute_place():
        box = "obj2"

        q_home = C.getJointState()

        komo = ry.KOMO(C, phases=3, slicesPerPhase=1, kOrder=1, enableCollisions=True)
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1], [-0.0])

        komo.addControlObjective([], 0, 1e-1)
        # komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        komo.addModeSwitch([1, 2], ry.SY.stable, ["a1", box])
        komo.addObjective([1, 2], ry.FS.distance, ["a1", box], ry.OT.sos, [1e1], [-0.0])
        komo.addObjective(
            [1, 2],
            ry.FS.positionDiff,
            ["a1", box],
            ry.OT.sos,
            [1e0, 1e0, 1e0],
        )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.positionDiff,
        #     ["a1_" + "ur_ee_marker", box],
        #     ry.OT.sos,
        #     [1e0],
        # )

        # komo.addObjective(
        #     [2], ry.FS.position, ["a2"], ry.OT.sos, [1e0, 1e1, 0], [1., -0.5, 0]
        # )

        # komo.addObjective(
        #     [2], ry.FS.position, [box], ry.OT.sos, [1e0, 1e0, 0], [0.8, -0.8, 0]
        # )

        komo.addModeSwitch([2, -1], ry.SY.stable, ["table", box])
        komo.addObjective([2, -1], ry.FS.poseDiff, ["goal2", box], ry.OT.eq, [1e1])

        komo.addObjective(
            times=[3],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.eq,
            scale=[1e0],
            target=q_home,
        )

        for _ in range(2000):
            # komo.initRandom()
            komo.initWithConstant(np.random.rand(6) * 2)

            solver = ry.NLP_Solver(komo.nlp(), verbose=4)
            # options.nonStrictSteps = 50;

            # solver.setOptions(damping=0.01, wolfe=0.001)
            # solver.setOptions(damping=0.001)
            retval = solver.solve()
            retval = retval.dict()

            print(retval)

            if view:
                komo.view(True, "IK solution")

            keyframes = komo.getPath()

            # print(retval)

            if retval["ineq"] < 1 and retval["eq"] < 1 and retval["feasible"]:
                return keyframes

    handover_keyframes = compute_handover()
    place_keyframes = compute_place()

    return C, np.concatenate([handover_keyframes, place_keyframes])


def make_single_agent_mover_env(num_goals=30, view: bool = False):
    C = make_table_with_walls(2, 2)
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

    C.addFrame("obj1").setParent(table).setShape(
        ry.ST.box, size=[0.4, 0.4, 0.06, 0.005]
    ).setColor([1, 0.5, 0, 1]).setContact(1).setRelativePosition(
        [+0.5, +0.5, 0.07]
    ).setJoint(ry.JT.rigid)

    C.addFrame("goal1").setParent(table).setShape(
        ry.ST.box, size=[0.2, 0.2, 0.06, 0.005]
    ).setColor([1, 0.5, 0, 0.3]).setContact(0).setRelativePosition([-0.5, -0.5, 0.07])

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

    qHome = C.getJointState()

    komo = ry.KOMO(C, phases=3, slicesPerPhase=1, kOrder=1, enableCollisions=True)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [0.1])
    komo.addControlObjective([], 0, 1e-1)
    # komo.addControlObjective([], 1, 1e0)
    # komo.addControlObjective([], 2, 1e0)

    komo.addModeSwitch([1, 2], ry.SY.stable, ["a1", "obj1"])
    komo.addObjective([1, 2], ry.FS.distance, ["a1", "obj1"], ry.OT.eq, [1e1])
    komo.addModeSwitch([2, -1], ry.SY.stable, ["table", "obj1"])

    komo.addObjective([2, -1], ry.FS.positionDiff, ["obj1", "goal1"], ry.OT.eq, [1e1])

    komo.addObjective(
        times=[3],
        feature=ry.FS.jointState,
        frames=[],
        type=ry.OT.eq,
        scale=[1e0],
        target=qHome,
    )

    # komo.addObjective([2], ry.FS.poseDiff, ["a2", "goal2"], ry.OT.eq, [1e1])

    # komo.addObjective([3, -1], ry.FS.poseDiff, ['a1', 'goal2'], ry.OT.eq, [1e1])
    # komo.addObjective(
    #     [3, -1], ry.FS.poseDiff, ["a2", "pre_agent_2_frame"], ry.OT.eq, [1e1]
    # )

    # we try to produce a couple different solutions
    sols = []
    for _ in range(num_goals):
        dim = len(C.getJointState())
        x_init = np.random.rand(dim) * 2 - 1.0
        komo.initWithConstant(x_init)

        solver = ry.NLP_Solver(komo.nlp(), verbose=4)
        # options.nonStrictSteps = 50;

        # solver.setOptions(damping=0.01, wolfe=0.001)
        # solver.setOptions(damping=0.001)
        retval = solver.solve()
        retval = retval.dict()

        # print(bottle, retval)

        if view:
            komo.view(True, "IK solution")

        keyframes = komo.getPath()

        # print(retval)

        if retval["ineq"] < 1 and retval["eq"] < 1 and retval["feasible"]:
            sols.append(keyframes)

    # # print(keyframes)

    return C, sols


def make_piano_mover_env(view: bool = False):
    C = make_table_with_walls(2, 2)
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

    qHome = C.getJointState()

    komo = ry.KOMO(C, phases=4, slicesPerPhase=1, kOrder=1, enableCollisions=True)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [0.1])
    komo.addControlObjective([], 0, 1e-1)
    # komo.addControlObjective([], 1, 1e0)
    # komo.addControlObjective([], 2, 1e0)

    komo.addModeSwitch([1, 2], ry.SY.stable, ["a1", "obj1"])
    komo.addObjective([1, 2], ry.FS.distance, ["a1", "obj1"], ry.OT.eq, [1e1])

    komo.addObjective([2, -1], ry.FS.poseDiff, ["obj1", "goal1"], ry.OT.eq, [1e1])

    komo.addModeSwitch([1, 2], ry.SY.stable, ["a2", "obj2"])
    komo.addObjective([1, 2], ry.FS.distance, ["a2", "obj2"], ry.OT.eq, [1e1])
    komo.addModeSwitch([2, -1], ry.SY.stable, ["table", "obj2"])
    komo.addModeSwitch([2, -1], ry.SY.stable, ["table", "obj1"])

    komo.addObjective([2, -1], ry.FS.poseDiff, ["obj2", "goal2"], ry.OT.eq, [1e1])

    komo.addObjective(
        times=[3],
        feature=ry.FS.jointState,
        frames=[],
        type=ry.OT.eq,
        scale=[1e0],
        target=qHome,
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
    # # print(keyframes)

    return C, keyframes


# environment to test informed sampling
def make_two_dim_tunnel_env(view: bool = False, agents_can_rotate=True):
    if not isinstance(agents_can_rotate, list):
        agents_can_rotate = [agents_can_rotate] * 2
    else:
        assert len(agents_can_rotate) == 2

    C = make_table_with_walls(4, 4)
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

    if agents_can_rotate[0]:
        C.addFrame("a1").setParent(pre_agent_1_frame).setShape(
            ry.ST.cylinder, size=[0.06, 0.15]
        ).setColor([1, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXYPhi, limits=np.array([-2, 2, -2, 2, -3.14, 3.14])
        ).setJointState([1.5, -0.0, 0])
    else:
        C.addFrame("a1").setParent(pre_agent_1_frame).setShape(
            ry.ST.cylinder, size=[0.06, 0.15]
        ).setColor([1, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXY, limits=np.array([-2, 2, -2, 2, -3.14, 3.14])
        ).setJointState([1.5, -0.0])

    pre_agent_2_frame = (
        C.addFrame("pre_agent_2_frame")
        .setParent(table)
        .setPosition(table.getPosition() + [0, 0.0, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    if agents_can_rotate[1]:
        C.addFrame("a2").setParent(pre_agent_2_frame).setShape(
            ry.ST.cylinder, size=[0.06, 0.15]
        ).setColor([0.5, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXYPhi, limits=np.array([-2, 2, -2, 2, -3.14, 3.14])
        ).setJointState([-1.5, -0.0, 0])
    else:
        C.addFrame("a2").setParent(pre_agent_2_frame).setShape(
            ry.ST.cylinder, size=[0.06, 0.15]
        ).setColor([0.5, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXY, limits=np.array([-2, 2, -2, 2, -3.14, 3.14])
        ).setJointState([-1.5, -0.0])

    if agents_can_rotate[0]:
        g1_state = np.array([-1.5, 0.5, 0])
        # g1_state = np.array([-1.5, -0.5, 0])
    else:
        g1_state = np.array([-1.5, 0.5])
        # g1_state = np.array([-1.5, -0.5, 0])

    if agents_can_rotate[1]:
        g2_state = np.array([0.5, +0.8, 0])
    else:
        g2_state = np.array([0.5, +0.8])

    C.addFrame("goal1").setParent(table).setShape(
        ry.ST.box, size=[0.2, 0.2, 0.06, 0.005]
    ).setColor([1, 0.5, 0, 0.3]).setContact(0).setRelativePosition(
        [g1_state[0], g1_state[1], 0.07]
    )

    C.addFrame("goal2").setParent(table).setShape(
        ry.ST.box, size=[0.2, 0.2, 0.06, 0.005]
    ).setColor([0.5, 0.5, 0, 0.3]).setContact(0).setRelativePosition(
        [g2_state[0], g2_state[1], 0.07]
    )

    C.addFrame("obs1").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0.0, 0.3, 0.07]
    ).setShape(ry.ST.box, size=[2, 0.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("obs2").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0.0, -0.3, 0.07]
    ).setShape(ry.ST.box, size=[2, 0.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("obs3").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0.0, -0.86, 0.07]
    ).setShape(ry.ST.box, size=[0.2, 0.9, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("obs4").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0.0, 1.2, 0.07]
    ).setShape(ry.ST.box, size=[0.2, 1.55, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    if view:
        C.view(True)

    keyframes = [g1_state, g2_state, C.getJointState()]

    print(keyframes)
    print(agents_can_rotate)

    # print(C.getJointLimits())

    # komo = ry.KOMO(C, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=True)
    # print(komo.nlp().getBounds())

    return C, keyframes


# environment to test informed sampling
def make_two_dim_short_tunnel_env(view: bool = False, agents_can_rotate=True):
    if not isinstance(agents_can_rotate, list):
        agents_can_rotate = [agents_can_rotate] * 2
    else:
        assert len(agents_can_rotate) == 2

    C = make_table_with_walls(4, 4)
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

    if agents_can_rotate[0]:
        C.addFrame("a1").setParent(pre_agent_1_frame).setShape(
            ry.ST.box, size=[0.1, 0.5, 0.06, 0.005]
        ).setColor([1, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXYPhi, limits=np.array([-2, 2, -2, 2, -3.14, 3.14])
        ).setJointState([1.5, -0.0, 0])
    else:
        C.addFrame("a1").setParent(pre_agent_1_frame).setShape(
            ry.ST.box, size=[0.1, 0.5, 0.06, 0.005]
        ).setColor([1, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXY, limits=np.array([-2, 2, -2, 2, -3.14, 3.14])
        ).setJointState([1.5, -0.0])

    pre_agent_2_frame = (
        C.addFrame("pre_agent_2_frame")
        .setParent(table)
        .setPosition(table.getPosition() + [0, 0.0, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    if agents_can_rotate[1]:
        C.addFrame("a2").setParent(pre_agent_2_frame).setShape(
            ry.ST.cylinder, size=[0.06, 0.15]
        ).setColor([0.5, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXYPhi, limits=np.array([-2, 2, -2, 2, -3.14, 3.14])
        ).setJointState([-1.5, -0.0, 0])
    else:
        C.addFrame("a2").setParent(pre_agent_2_frame).setShape(
            ry.ST.cylinder, size=[0.06, 0.15]
        ).setColor([0.5, 0.5, 0]).setContact(1).setJoint(
            ry.JT.transXY, limits=np.array([-2, 2, -2, 2, -3.14, 3.14])
        ).setJointState([-1.5, -0.0])

    if agents_can_rotate[0]:
        g1_state = np.array([-1.5, 1, np.pi / 2])
        # g1_state = np.array([-1.5, -0.5, 0])
    else:
        g1_state = np.array([-1.5, 1])
        # g1_state = np.array([-1.5, -0.5, 0])

    if agents_can_rotate[1]:
        g2_state = np.array([1.5, +1, 0])
    else:
        g2_state = np.array([1.5, +1])

    C.addFrame("goal1").setParent(table).setShape(
        ry.ST.box, size=[0.5, 0.1, 0.06, 0.005]
    ).setColor([1, 0.5, 0, 0.3]).setContact(0).setRelativePosition(
        [g1_state[0], g1_state[1], 0.07]
    )

    C.addFrame("goal2").setParent(table).setShape(
        ry.ST.box, size=[0.2, 0.2, 0.06, 0.005]
    ).setColor([0.5, 0.5, 0, 0.3]).setContact(0).setRelativePosition(
        [g2_state[0], g2_state[1], 0.07]
    )

    # C.addFrame("obs1").setParent(table).setPosition(
    #     C.getFrame("table").getPosition() + [0.0, 0.3, 0.07]
    # ).setShape(ry.ST.box, size=[2, 0.2, 0.06, 0.005]).setContact(1).setColor(
    #     [0, 0, 0]
    # ).setJoint(ry.JT.rigid)

    # C.addFrame("obs2").setParent(table).setPosition(
    #     C.getFrame("table").getPosition() + [0.0, -0.3, 0.07]
    # ).setShape(ry.ST.box, size=[2, 0.2, 0.06, 0.005]).setContact(1).setColor(
    #     [0, 0, 0]
    # ).setJoint(ry.JT.rigid)

    C.addFrame("obs3").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0.0, -0.5, 0.07]
    ).setShape(ry.ST.box, size=[1.4, 1.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("obs4").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0.0, 1.2, 0.07]
    ).setShape(ry.ST.box, size=[1.4, 1.5, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    if view:
        C.view(True)

    keyframes = [g1_state, g2_state, C.getJointState()]

    print(keyframes)
    print(agents_can_rotate)

    # print(C.getJointLimits())

    # komo = ry.KOMO(C, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=True)
    # print(komo.nlp().getBounds())

    return C, keyframes


def make_2d_rai_env_3_agents(view: bool = False):
    C = make_table_with_walls(2, 2)
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
        ry.ST.cylinder, size=[0.1, 0.2, 0.06, 0.15]
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
        ry.ST.box,
        size=[0.1, 0.2, 0.06, 0.005],
        # ry.ST.cylinder, size=[4, 0.1, 0.06, 0.075]
    ).setColor([0.5, 0.5, 0]).setContact(1).setJoint(
        ry.JT.transXYPhi, limits=np.array([-1, 1, -1, 1, -3.14, 3.14])
    ).setJointState([0, 0.4, 0.0])

    pre_agent_3_frame = (
        C.addFrame("pre_agent_3_frame")
        .setParent(table)
        .setPosition(table.getPosition() + [0.0, 0.0, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    C.addFrame("a3").setParent(pre_agent_3_frame).setShape(
        ry.ST.box,
        size=[0.3, 0.2, 0.06, 0.005],
        # ry.ST.cylinder, size=[4, 0.1, 0.06, 0.075]
    ).setColor([0.5, 0.5, 1]).setContact(1).setJoint(
        ry.JT.transXYPhi, limits=np.array([-1, 1, -1, 1, -3.14, 3.14])
    ).setJointState([0.5, -0.7, 0.0])

    C.addFrame("goal1").setParent(table).setShape(
        ry.ST.box, size=[0.2, 0.2, 0.06, 0.005]
    ).setColor([1, 0.5, 0, 0.3]).setContact(0).setRelativePosition([+0.5, +0.5, 0.07])

    C.addFrame("goal2").setParent(table).setShape(
        ry.ST.box, size=[0.2, 0.2, 0.06, 0.005]
    ).setColor([0.5, 0.5, 0, 0.2]).setContact(0).setRelativePosition([-0.5, -0.5, 0.07])

    C.addFrame("goal3").setParent(table).setShape(
        ry.ST.box, size=[0.2, 0.2, 0.06, 0.005]
    ).setColor([0.5, 0.5, 1, 0.2]).setContact(0).setRelativePosition([-0.6, 0.7, 0.07])

    C.addFrame("obs1").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0.75, 0, 0.07]
    ).setShape(ry.ST.box, size=[0.499, 0.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("obs2").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [-0.75, 0, 0.07]
    ).setShape(ry.ST.box, size=[0.499, 0.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("obs3").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0.1, 0, 0.07]
    ).setShape(ry.ST.box, size=[0.3, 0.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    if view:
        C.view(True)

    q_home = C.getJointState()

    komo = ry.KOMO(C, phases=6, slicesPerPhase=1, kOrder=1, enableCollisions=True)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])
    komo.addControlObjective([], 0, 1e-1)
    # komo.addControlObjective([], 1, 1e0)
    # komo.addControlObjective([], 2, 1e0)

    komo.addObjective([1], ry.FS.poseDiff, ["a1", "goal1"], ry.OT.eq, [1e1])
    komo.addObjective([2], ry.FS.poseDiff, ["a2", "goal2"], ry.OT.eq, [1e1])
    komo.addObjective([3], ry.FS.positionDiff, ["a3", "goal3"], ry.OT.eq, [1e1])
    komo.addObjective([4], ry.FS.positionDiff, ["a2", "goal3"], ry.OT.eq, [1e1])
    komo.addObjective([5], ry.FS.positionDiff, ["a1", "goal3"], ry.OT.eq, [1e1])

    komo.addObjective(
        times=[6],
        feature=ry.FS.jointState,
        frames=[],
        type=ry.OT.eq,
        scale=[1e0],
        target=q_home,
    )

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
    # print(keyframes)

    return C, keyframes


def small_angle_quaternion(axis, delta_theta):
    axis = np.asarray(axis) / np.linalg.norm(axis)

    # Compute quaternion components
    half_theta = delta_theta / 2
    q_w = 1.0  # cos(half_theta) approximated to 1 for small angles
    q_xyz = axis * half_theta  # sin(half_theta) ≈ half_theta

    return np.array([q_w, *q_xyz])


def make_box_sorting_env(view: bool = False):
    C = ry.Config()

    C.addFrame("floor").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(0)

    table = (
        C.addFrame("table")
        .setPosition([0, 0, 0.2])
        .setShape(ry.ST.box, size=[2, 3, 0.06, 0.005])
        .setColor([0.6, 0.6, 0.6])
        .setContact(1)
    )

    # C.addFile(ry.raiPath('panda/panda.g'), namePrefix='a1_') \
    #         .setParent(C.getFrame('table')) \
    #         .setRelativePosition([-0.3, 0.5, 0]) \
    #         .setRelativeQuaternion([0.7071, 0, 0, -0.7071]) \
    robot_path = os.path.join(os.path.dirname(__file__), "../models/ur10/ur10_vacuum.g")

    C.addFile(robot_path, namePrefix="a1_").setParent(
        C.getFrame("table")
    ).setRelativePosition([-0.5, 0.5, 0.0]).setRelativeQuaternion(
        [0.7071, 0, 0, -0.7071]
    ).setJoint(ry.JT.rigid)

    # C.getFrame('a1_ur_coll0').setContact(-2)

    C.addFile(robot_path, namePrefix="a2_").setParent(
        C.getFrame("table")
    ).setRelativePosition([+0.5, 0.5, 0.0]).setRelativeQuaternion(
        [0.7071, 0, 0, -0.7071]
    ).setJoint(ry.JT.rigid)

    # C.getFrame('a2_ur_coll0').setContact(-2)

    w = 2
    d = 2
    h = 2
    size = np.array([0.3, 0.2, 0.1])

    for k in range(d):
        for i in range(h):
            for j in range(w):
                axis = np.random.randn(3)
                axis /= np.linalg.norm(axis)
                delta_theta = np.random.rand() * 0 + 0.3
                perturbation_quaternion = small_angle_quaternion(axis, delta_theta * 0)

                pos = np.array(
                    [
                        j * size[0] * 1.3 - w / 2 * size[0] + size[0] / 2,
                        k * size[1] * 1.3 - 0.2,
                        i * size[2] * 1.3 + 0.05 + 0.4,
                    ]
                )
                C.addFrame("obj" + str(i) + str(j) + str(k)).setParent(table).setShape(
                    ry.ST.box, [size[0], size[1], size[2], 0.005]
                ).setPosition([pos[0], pos[1], pos[2]]).setMass(0.1).setColor(
                    np.random.rand(3)
                ).setContact(1).setQuaternion(perturbation_quaternion).setJoint(
                    ry.JT.rigid
                )

    C.addFrame("goal1").setParent(table).setShape(
        ry.ST.marker, [0.3, 0.2, 0.1, 0.005]
    ).setPosition([0, 1, 0.3]).setColor([0.1, 0.1, 0.1, 0.2]).setContact(0).setJoint(
        ry.JT.rigid
    )

    C.addFrame("goal2").setParent(table).setShape(
        ry.ST.marker, [0.3, 0.2, 0.1, 0.005]
    ).setPosition([0, 1, 0.3]).setColor([0.1, 0.1, 0.1, 0.2]).setContact(0)

    # sim = ry.Simulation(C, ry.SimulationEngine.physx, verbose=0) #try verbose=2

    # tau=.01

    # # C.view(True)

    # for t in range(int(3./tau)):
    #     # time.sleep(tau)

    #     sim.step([], tau, ry.ControlMode.spline)

    #     [X, q, V, qDot] = sim.getState()
    #     C.setFrameState(X)

    #     # C.view()
    #     # C.view_savePng('./z.vid/')

    #     #if (t%10)==0:
    #     #    C.view(False, f'Note: the sim operates *directly* on the given config\nt:{t:4d} = {tau*t:5.2f}sec')

    q_init = C.getJointState()
    q_init[0] += 1
    q_init[6] -= 1
    C.setJointState(q_init)

    if view:
        C.view(True)

    qHome = C.getJointState()

    komo = ry.KOMO(C, phases=3, slicesPerPhase=1, kOrder=1, enableCollisions=True)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1], [-0.1])

    komo.addControlObjective([], 0, 1e-1)
    # komo.addControlObjective([], 1, 1e-1)
    # komo.addControlObjective([], 2, 1e-1)

    box = "obj100"
    robot_prefix = "a1_"

    komo.addModeSwitch([1, 2], ry.SY.stable, [robot_prefix + "ur_vacuum", box])
    komo.addObjective(
        [1, 2],
        ry.FS.distance,
        [robot_prefix + "ur_vacuum", box],
        ry.OT.sos,
        [1e1],
    )
    komo.addObjective(
        [1, 2],
        ry.FS.positionDiff,
        [robot_prefix + "ur_ee_marker", box],
        ry.OT.sos,
        [1e1],
    )
    komo.addObjective(
        [1, 2],
        ry.FS.scalarProductYZ,
        [robot_prefix + "ur_ee_marker", box],
        ry.OT.sos,
        [1e1],
    )
    komo.addObjective(
        [1, 2],
        ry.FS.scalarProductZZ,
        [robot_prefix + "ur_ee_marker", box],
        ry.OT.sos,
        [1e1],
    )

    komo.addModeSwitch([2, -1], ry.SY.stable, ["table", box])
    komo.addObjective([2, -1], ry.FS.poseDiff, ["goal1", box], ry.OT.eq, [1e1])

    komo.addObjective(
        times=[3],
        feature=ry.FS.jointState,
        frames=[],
        type=ry.OT.eq,
        scale=[1e0],
        target=qHome,
    )

    solver = ry.NLP_Solver(komo.nlp(), verbose=4)
    solver.setOptions(damping=0.1, wolfe=0.001)
    solver.solve()

    if view:
        komo.view(True, "IK solution")

    keyframes_a1 = komo.getPath()

    komo = ry.KOMO(C, phases=3, slicesPerPhase=1, kOrder=1, enableCollisions=True)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1], [-0.1])

    komo.addControlObjective([], 0, 1e-1)
    # komo.addControlObjective([], 1, 1e-1)
    # komo.addControlObjective([], 2, 1e-1)

    box = "obj101"
    robot_prefix = "a2_"

    komo.addModeSwitch([1, 2], ry.SY.stable, [robot_prefix + "ur_vacuum", box])
    komo.addObjective(
        [1, 2], ry.FS.distance, [robot_prefix + "ur_vacuum", box], ry.OT.sos, [1e1]
    )
    komo.addObjective(
        [1, 2],
        ry.FS.positionDiff,
        [robot_prefix + "ur_ee_marker", box],
        ry.OT.sos,
        [1e1],
    )
    komo.addObjective(
        [1, 2],
        ry.FS.scalarProductYZ,
        [robot_prefix + "ur_ee_marker", box],
        ry.OT.sos,
        [1e1],
    )
    komo.addObjective(
        [1, 2],
        ry.FS.scalarProductZZ,
        [robot_prefix + "ur_ee_marker", box],
        ry.OT.sos,
        [1e1],
    )

    komo.addModeSwitch([2, -1], ry.SY.stable, ["table", box])
    komo.addObjective([2, -1], ry.FS.poseDiff, ["goal1", box], ry.OT.eq, [1e1])

    komo.addObjective(
        times=[3],
        feature=ry.FS.jointState,
        frames=[],
        type=ry.OT.eq,
        scale=[1e0],
        target=qHome,
    )

    solver = ry.NLP_Solver(komo.nlp(), verbose=4)
    solver.setOptions(damping=0.1, wolfe=0.001)
    solver.solve()

    if view:
        komo.view(True, "IK solution")

    keyframes_a2 = komo.getPath()

    keyframes = np.concatenate([keyframes_a1, keyframes_a2])
    # print(keyframes)
    return C, keyframes


def make_egg_carton_env(num_boxes=9, view: bool = False):
    C = ry.Config()

    C.addFrame("floor").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(0)

    table = (
        C.addFrame("table")
        .setPosition([0, 0, 0.2])
        .setShape(ry.ST.box, size=[2, 3, 0.06, 0.005])
        .setColor([0.6, 0.6, 0.6])
        .setContact(1)
    )

    robot_path = os.path.join(os.path.dirname(__file__), "../models/ur10/ur10_vacuum.g")

    C.addFile(robot_path, namePrefix="a1_").setParent(
        C.getFrame("table")
    ).setRelativePosition([-0.5, 0.5, 0]).setRelativeQuaternion(
        [0.7071, 0, 0, -0.7071]
    ).setJoint(ry.JT.rigid)

    # C.getFrame('a1_ur_coll0').setContact(-5)

    C.addFile(robot_path, namePrefix="a2_").setParent(
        C.getFrame("table")
    ).setRelativePosition([+0.5, 0.5, 0]).setRelativeQuaternion(
        [0.7071, 0, 0, -0.7071]
    ).setJoint(ry.JT.rigid)

    # C.getFrame('a2_ur_coll0').setContact(-5)

    w = 3
    d = 3
    h = 1
    size = np.array([0.3, 0.1, 0.07])

    all_boxes = []

    for k in range(d):
        for i in range(h):
            for j in range(w):
                axis = np.random.randn(3)
                axis /= np.linalg.norm(axis)
                delta_theta = np.random.rand() * 0.1 + 0.3
                perturbation_quaternion = small_angle_quaternion(axis, 0 * delta_theta)

                pos = np.array(
                    [
                        j * size[0] * 1.3 - w / 2 * size[0] + size[0] / 2,
                        k * size[1] * 1.3 - 0.4,
                        i * size[2] * 1.3 + 0.05 + 0.1,
                    ]
                )
                box_name = "obj" + str(i) + str(j) + str(k)
                all_boxes.append(box_name)
                C.addFrame(box_name).setParent(table).setShape(
                    ry.ST.box, [size[0], size[1], size[2], 0.005]
                ).setRelativePosition([pos[0], pos[1], pos[2]]).setMass(0.1).setColor(
                    np.random.rand(3)
                ).setContact(1).setQuaternion(perturbation_quaternion).setJoint(
                    ry.JT.rigid
                )

    C.addFrame("goal1").setParent(table).setShape(
        ry.ST.marker, [0.3, 0.2, 0.1, 0.005]
    ).setPosition([0, 1, 0.3]).setColor([0.1, 0.1, 0.1, 0.2]).setContact(0).setJoint(
        ry.JT.rigid
    )

    C.addFrame("goal2").setParent(table).setShape(
        ry.ST.marker, [0.3, 0.2, 0.1, 0.005]
    ).setPosition([0, 1, 0.3]).setColor([0.1, 0.1, 0.1, 0.2]).setContact(0)

    # pairs = C.getCollidablePairs()

    # for i in range(0, len(pairs), 2):
    #     print(pairs[i], pairs[i + 1])

    # sim = ry.Simulation(C, ry.SimulationEngine.physx, verbose=0)  # try verbose=2

    # tau = 0.01

    # # C.view(True)

    # for t in range(int(3.0 / tau)):
    #     # time.sleep(tau)

    #     sim.step([], tau, ry.ControlMode.spline)

    #     [X, q, V, qDot] = sim.getState()
    #     C.setFrameState(X)

    #     # C.view()
    #     # C.view_savePng('./z.vid/')

    #     # if (t%10)==0:
    #     #    C.view(False, f'Note: the sim operates *directly* on the given config\nt:{t:4d} = {tau*t:5.2f}sec')

    q_init = C.getJointState()
    q_init[0] += 1
    q_init[6] -= 1
    C.setJointState(q_init)

    if view:
        C.view(True)

    qHome = C.getJointState()

    def compute_keyframes_for_obj(robot_prefix, box, goal="goal1"):
        komo = ry.KOMO(C, phases=3, slicesPerPhase=1, kOrder=1, enableCollisions=True)
        komo.addObjective(
            [], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [-0.1]
        )

        komo.addControlObjective([], 0, 1e-1)
        komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        komo.addModeSwitch([1, 2], ry.SY.stable, [robot_prefix + "ur_vacuum", box])
        komo.addObjective(
            [1, 2],
            ry.FS.distance,
            [robot_prefix + "ur_vacuum", box],
            ry.OT.sos,
            [1e0],
            [0.05],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.positionDiff,
            [robot_prefix + "ur_vacuum", box],
            ry.OT.sos,
            [1e1, 1e1, 0],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.positionDiff,
            [robot_prefix + "ur_ee_marker", box],
            ry.OT.sos,
            [1e1],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.scalarProductYZ,
            [robot_prefix + "ur_ee_marker", box],
            ry.OT.sos,
            [1e1],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.scalarProductZZ,
            [robot_prefix + "ur_ee_marker", box],
            ry.OT.sos,
            [1e1],
        )

        komo.addModeSwitch([2, -1], ry.SY.stable, ["table", box])
        komo.addObjective([2, -1], ry.FS.poseDiff, [goal, box], ry.OT.eq, [1e1])

        komo.addObjective(
            times=[3],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.eq,
            scale=[1e0],
            target=qHome,
        )

        solver = ry.NLP_Solver(komo.nlp(), verbose=4)
        solver.setOptions(damping=0.1, wolfe=0.001)
        solver.solve()

        if view:
            komo.view(True, "IK solution")

        return komo.getPath()

    keyframes = {"a1_": [], "a2_": []}
    all_robots = ["a1_", "a2_"]

    for b in all_boxes[:num_boxes]:
        r = random.choice(all_robots)
        res = compute_keyframes_for_obj(r, b)
        keyframes[r].append((b, res))

    return C, keyframes


def make_crl_logo_rearrangement_env(num_robots=4, view: bool = False):
    assert num_robots <= 4, "Only a maximum of 4 robots are supported"

    C = ry.Config()

    C.addFrame("floor").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(0)

    table = (
        C.addFrame("table")
        .setPosition([0, 0, 0.2])
        .setShape(ry.ST.box, size=[2, 3, 0.06, 0.005])
        .setColor([0.6, 0.6, 0.6])
        .setContact(1)
    )

    robot_path = os.path.join(os.path.dirname(__file__), "../models/ur10/ur10_vacuum.g")

    C.addFile(robot_path, namePrefix="a1_").setParent(
        C.getFrame("table")
    ).setRelativePosition([-0.5, 0.5, 0]).setRelativeQuaternion(
        [0.7071, 0, 0, -0.7071]
    ).setJoint(ry.JT.rigid)

    # C.getFrame('a1_ur_coll0').setContact(-5)

    if num_robots >= 2:
        C.addFile(robot_path, namePrefix="a2_").setParent(
            C.getFrame("table")
        ).setRelativePosition([+0.5, 0.5, 0]).setRelativeQuaternion(
            [0.7071, 0, 0, -0.7071]
        ).setJoint(ry.JT.rigid)

    if num_robots >= 3:
        C.addFile(robot_path, namePrefix="a3_").setParent(
            C.getFrame("table")
        ).setRelativePosition([+0.5, -0.6, 0]).setRelativeQuaternion(
            [0.7071, 0, 0, 0.7071]
        ).setJoint(ry.JT.rigid)

    if num_robots >= 4:
        C.addFile(robot_path, namePrefix="a4_").setParent(
            C.getFrame("table")
        ).setRelativePosition([-0.5, -0.6, 0]).setRelativeQuaternion(
            [0.7071, 0, 0, 0.7071]
        ).setJoint(ry.JT.rigid)

    # C.getFrame('a2_ur_coll0').setContact(-5)

    w = 10
    d = 3
    size = np.array([0.1, 0.1, 0.1])

    boxes = []
    goals = []

    crl_logo = np.array(
        [
            [0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 1, 0, 1, 0, 1, 1],
        ]
    )

    def get_pos(j, k):
        stretch = 1.2
        pos = np.array(
            [
                j * size[0] * stretch
                - (w - 1) * stretch / 2 * size[0]
                + size[0] / 2
                - 0.1,
                k * size[1] * stretch - 0.2,
                0.085,
            ]
        )

        # if j > 2:
        #     pos[0] += 0.12
        # if j > 5:
        #     pos[0] += 0.12

        return pos

    def get_intermediate_pos(j, k):
        stretch = 1.2
        pos = np.array(
            [
                j * size[0] * stretch
                - (w - 1) * stretch / 2 * size[0]
                + size[0] / 2
                - 0.1,
                k * size[1] * stretch - 0.2,
                0.085,
            ]
        )

        return pos

    indices = []
    for k in range(d):
        for j in range(w):
            if j == 3 or j == 7:
                continue
            indices.append([k, j])

    sequence = [i for i, _ in enumerate(indices)]
    shuffled_sequence = sequence.copy()
    random.shuffle(shuffled_sequence)

    used_positions = []

    for i, s in enumerate(shuffled_sequence):
        k, j = indices[i]
        k_rnd, j_rnd = indices[s]

        pos = get_pos(j, k)
        rnd_pos = get_pos(j_rnd, k_rnd)

        used_positions.append(tuple(list(pos.tobytes())))

        color = [69 / 255, 144 / 255, 195 / 255, 1]
        # color = [46/255, 108/255, 164/255, 1]
        # color = [84/255, 188/255, 232/255, 1]
        if crl_logo[2 - k, j] == 0:
            color = [1, 1, 1, 1]

        C.addFrame("obj" + str(j_rnd) + str(k_rnd)).setParent(table).setShape(
            ry.ST.box, [size[0], size[1], size[2], 0.5]
        ).setRelativePosition([rnd_pos[0], rnd_pos[1], rnd_pos[2]]).setMass(
            0.1
        ).setColor(np.array(color)).setContact(1).setJoint(ry.JT.rigid)

        C.addFrame("goal" + str(j) + str(k)).setParent(table).setShape(
            ry.ST.marker, [size[0], size[1], size[2], 0.005]
        ).setRelativePosition([pos[0], pos[1], pos[2]]).setColor(color).setContact(
            0
        ).setJoint(ry.JT.rigid)

        boxes.append("obj" + str(j_rnd) + str(k_rnd))
        goals.append("goal" + str(j) + str(k))

    intermediate_goals = []

    for k in range(-1, d + 1):
        for j in range(-1, w + 1):
            pos = get_intermediate_pos(j, k)

            if tuple(list(pos.tobytes())) in used_positions:
                continue

            C.addFrame("intermediate_goal" + str(j) + str(k)).setParent(table).setShape(
                ry.ST.box, [size[0], size[1], size[2], 0.005]
            ).setRelativePosition([pos[0], pos[1], pos[2]]).setColor(
                [0, 0, 0.1, 0.1]
            ).setContact(0).setJoint(ry.JT.rigid)

            intermediate_goals.append("intermediate_goal" + str(j) + str(k))

    random.shuffle(intermediate_goals)

    if view:
        C.view(True)

    def compute_rearrangment(
        robot_prefix, box, intermediate_goal, goal, directly_place=False
    ):
        # set everything but the crrent box to non-contact
        c_tmp = ry.Config()
        c_tmp.addConfigurationCopy(C)

        robot_base = robot_prefix + "ur_base"
        c_tmp.selectJointsBySubtree(c_tmp.getFrame(robot_base))

        q_home = c_tmp.getJointState()

        for frame_name in boxes:
            if frame_name != box:
                c_tmp.getFrame(frame_name).setContact(0)

        komo = ry.KOMO(
            c_tmp, phases=5, slicesPerPhase=1, kOrder=1, enableCollisions=True
        )
        komo.addObjective(
            [], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [-0.0]
        )

        komo.addControlObjective([], 0, 1e-1)
        komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        komo.addModeSwitch([1, 2], ry.SY.stable, [robot_prefix + "ur_vacuum", box])
        komo.addObjective(
            [1, 2],
            ry.FS.distance,
            [robot_prefix + "ur_vacuum", box],
            ry.OT.ineq,
            [-1e0],
            [0.02],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.positionDiff,
            [robot_prefix + "ur_vacuum", box],
            ry.OT.sos,
            [1e1, 1e1, 0],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.scalarProductYZ,
            [robot_prefix + "ur_ee_marker", box],
            ry.OT.sos,
            [1e1],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.scalarProductZZ,
            [robot_prefix + "ur_ee_marker", box],
            ry.OT.sos,
            [1e1],
        )

        # for pick and place directly
        if directly_place:
            komo.addModeSwitch([2, -1], ry.SY.stable, ["table", box])
            komo.addObjective([2, -1], ry.FS.poseDiff, [goal, box], ry.OT.eq, [1e1])

            komo.addObjective(
                times=[3, -1],
                feature=ry.FS.jointState,
                frames=[],
                type=ry.OT.eq,
                scale=[1e0],
                target=q_home,
            )

        else:
            komo.addModeSwitch([2, 3], ry.SY.stable, ["table", box])
            komo.addObjective(
                [2, 3], ry.FS.poseDiff, [intermediate_goal, box], ry.OT.eq, [1e1]
            )

            komo.addModeSwitch([3, 4], ry.SY.stable, [robot_prefix + "ur_vacuum", box])
            komo.addObjective(
                [3, 4],
                ry.FS.distance,
                [robot_prefix + "ur_vacuum", box],
                ry.OT.ineq,
                [-1e0],
                [0.02],
            )
            komo.addObjective(
                [3, 4],
                ry.FS.positionDiff,
                [robot_prefix + "ur_vacuum", box],
                ry.OT.sos,
                [1e1, 1e1, 0],
            )
            komo.addObjective(
                [3, 4],
                ry.FS.scalarProductYZ,
                [robot_prefix + "ur_ee_marker", box],
                ry.OT.sos,
                [1e1],
            )
            komo.addObjective(
                [3, 4],
                ry.FS.scalarProductZZ,
                [robot_prefix + "ur_ee_marker", box],
                ry.OT.sos,
                [1e1],
            )

            komo.addModeSwitch([4, -1], ry.SY.stable, ["table", box])
            komo.addObjective([4, -1], ry.FS.poseDiff, [goal, box], ry.OT.eq, [1e1])

            komo.addObjective(
                times=[5],
                feature=ry.FS.jointState,
                frames=[],
                type=ry.OT.eq,
                scale=[1e0],
                target=q_home,
            )

        max_attempts = 5
        for num_attempt in range(max_attempts):
            # komo.initRandom()
            if num_attempt > 0:
                dim = len(c_tmp.getJointState())
                x_init = np.random.rand(dim) * 3 - 1.5
                komo.initWithConstant(x_init)
                # komo.initWithPath(np.random.rand(3, 12) * 5 - 2.5)

            solver = ry.NLP_Solver(komo.nlp(), verbose=4)
            # options.nonStrictSteps = 50;

            # solver.setOptions(damping=0.01, wolfe=0.001)
            # solver.setOptions(damping=0.001)
            retval = solver.solve()
            retval = retval.dict()

            # print(retval)

            # if view:
            # komo.view(True, "IK solution")

            print(retval)

            if retval["ineq"] < 1 and retval["eq"] < 1 and retval["feasible"]:
                # komo.view(True, "IK solution")
                keyframes = komo.getPath()
                return keyframes

        return None

        # solver = ry.NLP_Solver(komo.nlp(), verbose=10)
        # solver.setOptions(damping=0.1, wolfe=0.001)
        # retval = solver.solve()

        # print(retval.dict())

        # if view:
        #     komo.view(True, "IK solution")

        # keyframes = komo.getPath()

        # return keyframes

    # all_robots = ["a1_", "a2_"]
    all_robots = ["a1_", "a2_", "a3_", "a4_"]

    all_robots = all_robots[:num_robots]

    # direct_pick_place_keyframes = {"a1_": {}, "a2_": {}}
    # indirect_pick_place_keyframes = {"a1_": {}, "a2_": {}}

    direct_pick_place_keyframes = {}
    indirect_pick_place_keyframes = {}

    for r in all_robots:
        direct_pick_place_keyframes[r] = {}
        indirect_pick_place_keyframes[r] = {}

    for r in all_robots:
        for box, intermediate_goal, goal in zip(boxes, intermediate_goals, goals):
            r1 = compute_rearrangment(
                r, box, intermediate_goal, goal, directly_place=True
            )
            if r1 is not None:
                direct_pick_place_keyframes[r][box] = r1[:2]

            r2 = compute_rearrangment(r, box, intermediate_goal, goal)
            if r2 is not None:
                indirect_pick_place_keyframes[r][box] = r2[:4]

    all_objs = boxes.copy()
    random.shuffle(all_objs)

    box_goal = {}
    for b, g in zip(boxes, goals):
        box_goal[b] = g

    keyframes = []

    for i in range(len(all_objs)):
        obj_to_move = all_objs[i]
        goal = box_goal[obj_to_move]

        goal_location_is_free = False
        for prev_moved_obj in all_objs[:i]:
            # check if the 'coordinates' of a boxes goal location appear in the
            # objects that were already moved. If yes,
            # we already moved the object that previoulsy occupied
            # this location.
            if goal[-2:] == prev_moved_obj[-2:]:
                goal_location_is_free = True
                break

        place_directly = False
        if goal_location_is_free:
            place_directly = True

        while True:
            robot = random.choice(all_robots)

            if (
                (place_directly and obj_to_move in direct_pick_place_keyframes[robot])
                or obj_to_move in indirect_pick_place_keyframes[robot]
            ):
                break

            print("choosing robot")

        if place_directly:
            keyframes.append(
                (
                    robot,
                    obj_to_move,
                    direct_pick_place_keyframes[robot][obj_to_move],
                    goal,
                )
            )
        else:
            keyframes.append(
                (
                    robot,
                    obj_to_move,
                    indirect_pick_place_keyframes[robot][obj_to_move],
                    goal,
                )
            )

    return C, keyframes, all_robots


def make_box_rearrangement_env(num_robots=2, num_boxes=9, view: bool = False):
    assert num_boxes <= 9, "A maximum of 9 boxes are supported"
    assert num_robots <= 4, "A maximum of 4 robots are supported"

    C = ry.Config()

    C.addFrame("floor").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(0)

    table = (
        C.addFrame("table")
        .setPosition([0, 0, 0.2])
        .setShape(ry.ST.box, size=[2, 3, 0.06, 0.005])
        .setColor([0.6, 0.6, 0.6])
        .setContact(1)
    )

    robot_path = os.path.join(os.path.dirname(__file__), "../models/ur10/ur10_vacuum.g")

    C.addFile(robot_path, namePrefix="a1_").setParent(
        C.getFrame("table")
    ).setRelativePosition([-0.5, 0.5, 0]).setRelativeQuaternion(
        [0.7071, 0, 0, -0.7071]
    ).setJoint(ry.JT.rigid)

    # C.getFrame('a1_ur_coll0').setContact(-5)

    if num_robots >= 2:
        C.addFile(robot_path, namePrefix="a2_").setParent(
            C.getFrame("table")
        ).setRelativePosition([+0.5, 0.5, 0]).setRelativeQuaternion(
            [0.7071, 0, 0, -0.7071]
        ).setJoint(ry.JT.rigid)

    if num_robots >= 3:
        C.addFile(robot_path, namePrefix="a3_").setParent(
            C.getFrame("table")
        ).setRelativePosition([+0.5, -0.6, 0]).setRelativeQuaternion(
            [0.7071, 0, 0, 0.7071]
        ).setJoint(ry.JT.rigid)

    if num_robots >= 4:
        C.addFile(robot_path, namePrefix="a4_").setParent(
            C.getFrame("table")
        ).setRelativePosition([-0.5, -0.6, 0]).setRelativeQuaternion(
            [0.7071, 0, 0, 0.7071]
        ).setJoint(ry.JT.rigid)

    # C.getFrame('a2_ur_coll0').setContact(-5)

    w = 3
    d = 3
    size = np.array([0.1, 0.1, 0.1])

    boxes = []
    goals = []

    cnt = 0
    for k in range(d):
        for j in range(w):
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            delta_theta = np.random.rand() * 0.1 + 0.3
            perturbation_quaternion = small_angle_quaternion(axis, 0 * delta_theta)

            pos = np.array(
                [
                    j * size[0] * 1.5 - w / 2 * size[0] + size[0] / 2,
                    k * size[1] * 1.5 - 0.2,
                    0.085,
                ]
            )
            C.addFrame("obj" + str(j) + str(k)).setParent(table).setShape(
                ry.ST.ssBox, [size[0], size[1], size[2], 0.005]
            ).setRelativePosition([pos[0], pos[1], pos[2]]).setMass(0.1).setColor(
                np.random.rand(3)
            ).setContact(1).setQuaternion(perturbation_quaternion).setJoint(ry.JT.rigid)

            C.addFrame("goal" + str(j) + str(k)).setParent(table).setShape(
                ry.ST.marker, [size[0], size[1], size[2], 0.005]
            ).setRelativePosition([pos[0], pos[1], pos[2]]).setColor(
                [0, 0, 0.1, 0.5]
            ).setContact(0).setQuaternion(perturbation_quaternion).setJoint(ry.JT.rigid)

            boxes.append("obj" + str(j) + str(k))
            goals.append("goal" + str(j) + str(k))

            cnt += 1

            if cnt == num_boxes:
                break

        if cnt == num_boxes:
            break

    intermediate_goals = []

    for k in range(d + 2):
        for j in range(w + 2):
            if k == 0 or k == d + 1 or j == 0 or j == w + 1:
                pos = np.array(
                    [
                        j * size[0] * 1.5 - (w + 2) / 2 * size[0],
                        k * size[1] * 1.5 - 0.35,
                        0.085,
                    ]
                )

                C.addFrame("intermediate_goal" + str(j) + str(k)).setParent(
                    table
                ).setShape(
                    ry.ST.marker, [size[0], size[1], size[2], 0.005]
                ).setRelativePosition([pos[0], pos[1], pos[2]]).setColor(
                    [0, 0, 0.1, 0.1]
                ).setContact(0).setJoint(ry.JT.rigid)

                intermediate_goals.append("intermediate_goal" + str(j) + str(k))

    if view:
        C.view(True)

    # figure out what should go where
    # random.shuffle(boxes)
    random.shuffle(goals)

    while True:
        print()
        obj_in_same_place = False
        for i, obj in enumerate(boxes):
            print(obj)
            print(goals[i])
            if obj[-2:] == goals[i][-2:]:
                print(obj)
                obj_in_same_place = True

        if not obj_in_same_place:
            break

        random.shuffle(goals)

    random.shuffle(intermediate_goals)

    def compute_rearrangment(
        robot_prefix, box, intermediate_goal, goal, directly_place=False
    ):
        # set everything but the crrent box to non-contact
        c_tmp = ry.Config()
        c_tmp.addConfigurationCopy(C)

        robot_base = robot_prefix + "ur_base"
        c_tmp.selectJointsBySubtree(c_tmp.getFrame(robot_base))

        q_home = c_tmp.getJointState()

        for frame_name in boxes:
            if frame_name != box:
                c_tmp.getFrame(frame_name).setContact(0)

        komo = ry.KOMO(
            c_tmp, phases=5, slicesPerPhase=1, kOrder=1, enableCollisions=True
        )
        komo.addObjective(
            [], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [-0.0]
        )

        komo.addControlObjective([], 0, 1e-1)
        komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        komo.addModeSwitch([1, 2], ry.SY.stable, [robot_prefix + "ur_vacuum", box])
        komo.addObjective(
            [1, 2],
            ry.FS.distance,
            [robot_prefix + "ur_vacuum", box],
            ry.OT.sos,
            [1e0],
            [0.05],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.positionDiff,
            [robot_prefix + "ur_vacuum", box],
            ry.OT.sos,
            [1e1, 1e1, 1],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.scalarProductYZ,
            [robot_prefix + "ur_ee_marker", box],
            ry.OT.sos,
            [1e1],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.scalarProductZZ,
            [robot_prefix + "ur_ee_marker", box],
            ry.OT.sos,
            [1e1],
        )

        # for pick and place directly
        if directly_place:
            komo.addModeSwitch([2, -1], ry.SY.stable, ["table", box])
            komo.addObjective([2, -1], ry.FS.poseDiff, [goal, box], ry.OT.eq, [1e1])

            komo.addObjective(
                times=[3, -1],
                feature=ry.FS.jointState,
                frames=[],
                type=ry.OT.eq,
                scale=[1e0],
                target=q_home,
            )

        else:
            komo.addModeSwitch([2, 3], ry.SY.stable, ["table", box])
            komo.addObjective(
                [2, 3], ry.FS.poseDiff, [intermediate_goal, box], ry.OT.eq, [1e1]
            )

            komo.addModeSwitch([3, 4], ry.SY.stable, [robot_prefix + "ur_vacuum", box])
            komo.addObjective(
                [3, 4],
                ry.FS.distance,
                [robot_prefix + "ur_vacuum", box],
                ry.OT.sos,
                [1e0],
                [0.05],
            )
            komo.addObjective(
                [3, 4],
                ry.FS.positionDiff,
                [robot_prefix + "ur_vacuum", box],
                ry.OT.sos,
                [1e1, 1e1, 1],
            )
            komo.addObjective(
                [3, 4],
                ry.FS.scalarProductYZ,
                [robot_prefix + "ur_ee_marker", box],
                ry.OT.sos,
                [1e1],
            )
            komo.addObjective(
                [3, 4],
                ry.FS.scalarProductZZ,
                [robot_prefix + "ur_ee_marker", box],
                ry.OT.sos,
                [1e1],
            )

            komo.addModeSwitch([4, -1], ry.SY.stable, ["table", box])
            komo.addObjective([4, -1], ry.FS.poseDiff, [goal, box], ry.OT.eq, [1e1])

            komo.addObjective(
                times=[5],
                feature=ry.FS.jointState,
                frames=[],
                type=ry.OT.eq,
                scale=[1e0],
                target=q_home,
            )

        max_attempts = 5
        for num_attempt in range(max_attempts):
            # komo.initRandom()
            if num_attempt > 0:
                dim = len(c_tmp.getJointState())
                x_init = np.random.rand(dim) * 3 - 1.5
                komo.initWithConstant(x_init)
                # komo.initWithPath(np.random.rand(3, 12) * 5 - 2.5)

            solver = ry.NLP_Solver(komo.nlp(), verbose=4)
            # options.nonStrictSteps = 50;

            # solver.setOptions(damping=0.01, wolfe=0.001)
            # solver.setOptions(damping=0.001)
            retval = solver.solve()
            retval = retval.dict()

            # print(retval)

            # if view:
            # komo.view(True, "IK solution")

            print(retval)

            if retval["ineq"] < 1 and retval["eq"] < 1 and retval["feasible"]:
                # komo.view(True, "IK solution")
                keyframes = komo.getPath()
                return keyframes

        return None

    # all_robots = ["a1_", "a2_"]
    all_robots = ["a1_", "a2_", "a3_", "a4_"]

    all_robots = all_robots[:num_robots]

    # direct_pick_place_keyframes = {"a1_": {}, "a2_": {}}
    # indirect_pick_place_keyframes = {"a1_": {}, "a2_": {}}

    direct_pick_place_keyframes = {}
    indirect_pick_place_keyframes = {}

    for r in all_robots:
        direct_pick_place_keyframes[r] = {}
        indirect_pick_place_keyframes[r] = {}

    for r in all_robots:
        for box, intermediate_goal, goal in zip(boxes, intermediate_goals, goals):
            r1 = compute_rearrangment(
                r, box, intermediate_goal, goal, directly_place=True
            )
            direct_pick_place_keyframes[r][box] = r1[:2]

            r2 = compute_rearrangment(r, box, intermediate_goal, goal)
            indirect_pick_place_keyframes[r][box] = r2[:4]

            # r3 = compute_rearrangment(
            #     "a2_", box, intermediate_goal, goal, directly_place=True
            # )
            # direct_pick_place_keyframes["a2_"][box] = r3[:2]

            # r4 = compute_rearrangment("a2_", box, intermediate_goal, goal)
            # indirect_pick_place_keyframes["a2_"][box] = r4[:4]

    all_objs = boxes.copy()
    random.shuffle(all_objs)

    robot_to_use = []

    while True:
        robot_to_use = []

        for _ in range(len(all_objs)):
            r = random.choice(all_robots)
            robot_to_use.append(r)

            print(r)

        robot_unused = False
        for r in all_robots:
            if robot_to_use.count(r) == 0:
                robot_unused = True

        if not robot_unused:
            break

        print(robot_to_use)

    box_goal = {}
    for b, g in zip(boxes, goals):
        box_goal[b] = g

    keyframes = []

    for i in range(len(all_objs)):
        obj_to_move = all_objs[i]
        goal = box_goal[obj_to_move]

        goal_location_is_free = False
        for prev_moved_obj in all_objs[:i]:
            # check if the 'coordinates' of a boxes goal location appear in the
            # objects that were already moved. If yes,
            # we already moved the object that previoulsy occupied
            # this location.
            if goal[-2:] == prev_moved_obj[-2:]:
                goal_location_is_free = True
                break

        place_directly = False
        if goal_location_is_free:
            place_directly = True

        robot = robot_to_use[i]

        if place_directly:
            keyframes.append(
                (
                    robot,
                    obj_to_move,
                    direct_pick_place_keyframes[robot][obj_to_move],
                    goal,
                )
            )
        else:
            keyframes.append(
                (
                    robot,
                    obj_to_move,
                    indirect_pick_place_keyframes[robot][obj_to_move],
                    goal,
                )
            )

    return C, keyframes, all_robots


def make_box_stacking_env(
    num_robots=2, num_boxes=9, mixed_robots: bool = False, view: bool = False, make_and_return_all_keyframes: bool = False
):
    assert num_boxes <= 9, "A maximum of 9 boxes are supported"
    assert num_robots <= 4, "A maximum of 4 robots are supported"

    C = ry.Config()

    C.addFrame("floor").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(0)

    table = (
        C.addFrame("table")
        .setPosition([0, 0, 0.2])
        .setShape(ry.ST.box, size=[2, 3, 0.06, 0.005])
        .setColor([0.6, 0.6, 0.6])
        .setContact(1)
    )

    ur10_path = os.path.join(
        os.path.dirname(__file__), "../models/ur10/ur10_two_finger.g"
    )

    kuka_path = os.path.join(
        os.path.dirname(__file__), "../models/kuka_drake/kuka_two_finger.g"
    )

    def get_robot_and_type_prefix(idx: int):
        if not mixed_robots:
            return ur10_path, "ur_"
        else:
            if idx in [0, 2]:
                return ur10_path, "ur_"
            else:
                return kuka_path, "kuka_"

    # robot_path = ur10_path

    all_robots = []

    robot_path, robot_type_prefix = get_robot_and_type_prefix(0)
    all_robots.append(f"a1_{robot_type_prefix}")

    C.addFile(robot_path, namePrefix="a1_").setParent(
        C.getFrame("table")
    ).setRelativePosition([-0.5, 0.5, 0]).setRelativeQuaternion(
        [0.7071, 0, 0, -0.7071]
    ).setJoint(ry.JT.rigid)

    # C.getFrame('a1_ur_coll0').setContact(-5)

    if num_robots >= 2:
        robot_path, robot_type_prefix = get_robot_and_type_prefix(1)
        all_robots.append(f"a2_{robot_type_prefix}")

        C.addFile(robot_path, namePrefix="a2_").setParent(
            C.getFrame("table")
        ).setRelativePosition([+0.5, 0.5, 0]).setRelativeQuaternion(
            [0.7071, 0, 0, -0.7071]
        ).setJoint(ry.JT.rigid)

    if num_robots >= 3:
        robot_path, robot_type_prefix = get_robot_and_type_prefix(2)
        all_robots.append(f"a3_{robot_type_prefix}")

        C.addFile(robot_path, namePrefix="a3_").setParent(
            C.getFrame("table")
        ).setRelativePosition([+0.5, -0.6, 0]).setRelativeQuaternion(
            [0.7071, 0, 0, 0.7071]
        ).setJoint(ry.JT.rigid)

    if num_robots >= 4:
        robot_path, robot_type_prefix = get_robot_and_type_prefix(3)
        all_robots.append(f"a4_{robot_type_prefix}")

        C.addFile(robot_path, namePrefix="a4_").setParent(
            C.getFrame("table")
        ).setRelativePosition([-0.5, -0.6, 0]).setRelativeQuaternion(
            [0.7071, 0, 0, 0.7071]
        ).setJoint(ry.JT.rigid)

    # C.getFrame('a2_ur_coll0').setContact(-5)

    w = 3
    d = 3
    size = np.array([0.05, 0.05, 0.05])

    boxes = []
    goals = []

    height = 0.065

    def get_pos(j, k):
        pos = np.array(
            [
                j * size[0] * 3 - w / 2 * size[0] + size[0] / 2,
                k * size[1] * 3 - 0.2,
                height,
            ]
        )
        return pos

    cnt = 0
    for k in range(d):
        for j in range(w):
            if k == 1 and j == 1:
                continue

            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            delta_theta = np.random.rand() * 0.1 + 0.3
            perturbation_quaternion = small_angle_quaternion(axis, 0 * delta_theta)

            pos = get_pos(j, k)

            C.addFrame("obj" + str(j) + str(k)).setParent(table).setShape(
                ry.ST.ssBox, [size[0], size[1], size[2], 0.005]
            ).setRelativePosition([pos[0], pos[1], pos[2]]).setMass(0.1).setColor(
                np.random.rand(3)
            ).setContact(1).setQuaternion(perturbation_quaternion).setJoint(ry.JT.rigid)

            C.addFrame("goal" + str(j) + str(k)).setParent(table).setShape(
                ry.ST.box, [size[0], size[1], size[2], 0.005]
            ).setRelativePosition(
                [get_pos(1, 1)[0], get_pos(1, 1)[1], cnt * size[2] * 1.1 + height]
            ).setColor([0, 0, 0.1, 0.5]).setContact(0).setQuaternion(
                perturbation_quaternion
            ).setJoint(ry.JT.rigid)

            boxes.append("obj" + str(j) + str(k))
            goals.append("goal" + str(j) + str(k))

            cnt += 1

            if cnt == num_boxes:
                break

        if cnt == num_boxes:
            break

    if view:
        C.view(True)

    # figure out what should go where
    random.shuffle(boxes)

    def compute_rearrangment(c_tmp, robot_prefix, box, goal):
        # set everything but the current box to non-contact
        robot_base = robot_prefix + "base"
        c_tmp.selectJointsBySubtree(c_tmp.getFrame(robot_base))

        q_home = c_tmp.getJointState()

        komo = ry.KOMO(
            c_tmp, phases=2, slicesPerPhase=1, kOrder=1, enableCollisions=True
        )
        komo.addObjective(
            [], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [-0.0]
        )

        komo.addControlObjective([], 0, 1e-1)
        # komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        komo.addModeSwitch([1, 2], ry.SY.stable, [robot_prefix + "gripper", box])
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.distance,
        #     [robot_prefix + "ur_gripper_center", box],
        #     ry.OT.sos,
        #     [1e0],
        #     # [0.05],
        # )
        komo.addObjective(
            [1, 2],
            ry.FS.positionDiff,
            [robot_prefix + "gripper_center", box],
            ry.OT.sos,
            [1e1, 1e1, 1],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.scalarProductZZ,
            [robot_prefix + "gripper_center", box],
            ry.OT.sos,
            [1e1],
            [-1],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.scalarProductXX,
            [robot_prefix + "gripper_center", box],
            ry.OT.sos,
            [1e1],
            [1],
        )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.scalarProductZZ,
        #     [robot_prefix + "ur_gripper", box],
        #     ry.OT.sos,
        #     [1e1],
        # )

        # komo.addModeSwitch([1, 2], ry.SY.stable, [robot_prefix + "ur_vacuum", box])
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.distance,
        #     [robot_prefix + "ur_vacuum", box],
        #     ry.OT.sos,
        #     [1e0],
        #     [0.05],
        # )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.positionDiff,
        #     [robot_prefix + "ur_vacuum", box],
        #     ry.OT.sos,
        #     [1e1, 1e1, 1],
        # )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.scalarProductYZ,
        #     [robot_prefix + "ur_ee_marker", box],
        #     ry.OT.sos,
        #     [1e1],
        # )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.scalarProductZZ,
        #     [robot_prefix + "ur_ee_marker", box],
        #     ry.OT.sos,
        #     [1e1],
        # )

        # for pick and place directly
        # komo.addModeSwitch([2, -1], ry.SY.stable, ["table", box])
        komo.addObjective([2, -1], ry.FS.poseDiff, [goal, box], ry.OT.eq, [1e1])

        komo.addObjective(
            times=[3, -1],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.eq,
            scale=[1e0],
            target=q_home,
        )

        max_attempts = 5
        for num_attempt in range(max_attempts):
            # komo.initRandom()
            if num_attempt > 0:
                dim = len(c_tmp.getJointState())
                x_init = np.random.rand(dim) * 3 - 1.5
                komo.initWithConstant(x_init)
                # komo.initWithPath(np.random.rand(3, 12) * 5 - 2.5)

            solver = ry.NLP_Solver(komo.nlp(), verbose=4)
            # options.nonStrictSteps = 50;

            # solver.setOptions(damping=0.01, wolfe=0.001)
            # solver.setOptions(damping=0.001)
            retval = solver.solve()
            retval = retval.dict()

            # print(retval)

            # if view:
            # komo.view(True, "IK solution")

            print(retval)

            if retval["ineq"] < 1 and retval["eq"] < 1 and retval["feasible"]:
                # komo.view(True, "IK solution")
                keyframes = komo.getPath()
                return keyframes

        return None

    # all_robots = ["a1_", "a2_", "a3_", "a4_"]
    # all_robots = all_robots[:num_robots]

    if make_and_return_all_keyframes:
        direct_pick_place_keyframes = {}
        
        for r in all_robots:
            direct_pick_place_keyframes[r] = {}

        c_tmp = ry.Config()
        c_tmp.addConfigurationCopy(C)

        for box, goal in zip(boxes, goals):
            for r in all_robots:
                c_tmp_2 = ry.Config()
                c_tmp_2.addConfigurationCopy(c_tmp)
                # c_tmp_2.computeCollisions()

                r1 = compute_rearrangment(c_tmp_2, r, box, goal)

                if r1 is not None:
                    direct_pick_place_keyframes[r][box] = r1[:2]

            c_tmp.getFrame(box).setRelativePosition(
                c_tmp.getFrame(goal).getRelativePosition()
            )

        keyframes = []

        # for r, b, g in zip(robot_to_use, boxes, goals):
        for b in boxes:
            for r in all_robots:
                if b in direct_pick_place_keyframes[r]:
                    keyframes.append(
                        (
                            r,
                            b,
                            direct_pick_place_keyframes[r][b],
                        )
                    )
                    
        return C, keyframes, all_robots
    else:
        direct_pick_place_keyframes = {}

        for r in all_robots:
            direct_pick_place_keyframes[r] = {}

        c_tmp = ry.Config()
        c_tmp.addConfigurationCopy(C)

        robot_to_use = []

        for box, goal in zip(boxes, goals):
            c_tmp_2 = ry.Config()
            c_tmp_2.addConfigurationCopy(c_tmp)
            # c_tmp_2.computeCollisions()

            while True:
                r = random.choice(all_robots)
                r1 = compute_rearrangment(c_tmp_2, r, box, goal)

                if r1 is not None:
                    break

            direct_pick_place_keyframes[r][box] = r1[:2]
            robot_to_use.append(r)

            c_tmp.getFrame(box).setRelativePosition(
                c_tmp.getFrame(goal).getRelativePosition()
            )

        box_goal = {}
        for b, g in zip(boxes, goals):
            box_goal[b] = g

        keyframes = []

        for r, b, g in zip(robot_to_use, boxes, goals):
            keyframes.append(
                (
                    r,
                    b,
                    direct_pick_place_keyframes[r][b],
                    goal,
                )
            )

        return C, keyframes, all_robots
    
def make_pyramid_env(
    num_robots=2, num_boxes=6, mixed_robots: bool = False, view: bool = False
):
    assert num_boxes <= 9, "A maximum of 6 boxes are supported"
    assert num_robots <= 4, "A maximum of 4 robots are supported"

    C = ry.Config()

    C.addFrame("floor").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(0)

    table = (
        C.addFrame("table")
        .setPosition([0, 0, 0.2])
        .setShape(ry.ST.box, size=[2, 3, 0.06, 0.005])
        .setColor([0.6, 0.6, 0.6])
        .setContact(1)
    )

    ur10_path = os.path.join(
        os.path.dirname(__file__), "../models/ur10/ur10_two_finger.g"
    )

    kuka_path = os.path.join(
        os.path.dirname(__file__), "../models/kuka_drake/kuka_two_finger.g"
    )

    def get_robot_and_type_prefix(idx: int):
        if not mixed_robots:
            return ur10_path, "ur_"
        else:
            if idx in [0, 2]:
                return ur10_path, "ur_"
            else:
                return kuka_path, "kuka_"

    # robot_path = ur10_path

    all_robots = []

    robot_path, robot_type_prefix = get_robot_and_type_prefix(0)
    all_robots.append(f"a1_{robot_type_prefix}")

    C.addFile(robot_path, namePrefix="a1_").setParent(
        C.getFrame("table")
    ).setRelativePosition([-0.5, 0.5, 0]).setRelativeQuaternion(
        [0.7071, 0, 0, -0.7071]
    ).setJoint(ry.JT.rigid)

    # C.getFrame('a1_ur_coll0').setContact(-5)

    if num_robots >= 2:
        robot_path, robot_type_prefix = get_robot_and_type_prefix(1)
        all_robots.append(f"a2_{robot_type_prefix}")

        C.addFile(robot_path, namePrefix="a2_").setParent(
            C.getFrame("table")
        ).setRelativePosition([+0.5, 0.5, 0]).setRelativeQuaternion(
            [0.7071, 0, 0, -0.7071]
        ).setJoint(ry.JT.rigid)

    if num_robots >= 3:
        robot_path, robot_type_prefix = get_robot_and_type_prefix(2)
        all_robots.append(f"a3_{robot_type_prefix}")

        C.addFile(robot_path, namePrefix="a3_").setParent(
            C.getFrame("table")
        ).setRelativePosition([+0.5, -0.6, 0]).setRelativeQuaternion(
            [0.7071, 0, 0, 0.7071]
        ).setJoint(ry.JT.rigid)

    if num_robots >= 4:
        robot_path, robot_type_prefix = get_robot_and_type_prefix(3)
        all_robots.append(f"a4_{robot_type_prefix}")

        C.addFile(robot_path, namePrefix="a4_").setParent(
            C.getFrame("table")
        ).setRelativePosition([-0.5, -0.6, 0]).setRelativeQuaternion(
            [0.7071, 0, 0, 0.7071]
        ).setJoint(ry.JT.rigid)

    # C.getFrame('a2_ur_coll0').setContact(-5)
    height = 0.065

    size = np.array([0.05, 0.05, 0.05])

    poses = [
        np.array([size[0] * 1.1, 0, height]),
        np.array([0, 0, height]),
        np.array([-size[0] * 1.1, 0, height]),

        np.array([size[0] / 2  * 1.1, 0, height + size[2] * 1.1]),
        np.array([-size[0] / 2  * 1.1, 0, height + size[2] * 1.1]),

        np.array([0, 0, height + 2 * size[2] * 1.1]),
    ]

    boxes = []
    goals = []

    for i in range(num_boxes):
        pose = poses[i]

        goal_name = "goal" + str(i)
        C.addFrame(goal_name).setParent(table).setShape(
            ry.ST.box, [size[0], size[1], size[2], 0.005]
        ).setRelativePosition(
            [pose[0], pose[1], pose[2]]
        ).setColor([0, 0, 0.1, 0.5]).setContact(0).setJoint(ry.JT.rigid)

        goals.append(goal_name)

        obj_name = "obj" + str(i)
        C.addFrame(obj_name).setParent(table).setShape(
            ry.ST.ssBox, [size[0], size[1], size[2], 0.005]
        ).setRelativePosition([0, 0, -1]).setMass(0.1).setColor(
            np.random.rand(3)
        ).setContact(0).setJoint(ry.JT.rigid)

        boxes.append(obj_name)

    if view:
        C.view(True)

    def compute_rearrangment(c_tmp, robot_prefix, box, goal):
        # set everything but the current box to non-contact
        robot_base = robot_prefix + "base"
        c_tmp.selectJointsBySubtree(c_tmp.getFrame(robot_base))

        q_home = c_tmp.getJointState()

        komo = ry.KOMO(
            c_tmp, phases=2, slicesPerPhase=1, kOrder=1, enableCollisions=True
        )
        komo.addObjective(
            [], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [-0.0]
        )

        komo.addControlObjective([], 0, 1e-1)
        # komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        komo.addModeSwitch([1, 2], ry.SY.stable, [robot_prefix + "gripper", box])
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.distance,
        #     [robot_prefix + "ur_gripper_center", box],
        #     ry.OT.sos,
        #     [1e0],
        #     # [0.05],
        # )
        komo.addObjective(
            [1, 2],
            ry.FS.positionDiff,
            [robot_prefix + "gripper_center", box],
            ry.OT.sos,
            [1e1, 1e1, 1],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.scalarProductZZ,
            [robot_prefix + "gripper_center", box],
            ry.OT.sos,
            [1e1],
            [-1],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.scalarProductXX,
            [robot_prefix + "gripper_center", box],
            ry.OT.sos,
            [1e1],
            [1],
        )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.scalarProductZZ,
        #     [robot_prefix + "ur_gripper", box],
        #     ry.OT.sos,
        #     [1e1],
        # )

        # komo.addModeSwitch([1, 2], ry.SY.stable, [robot_prefix + "ur_vacuum", box])
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.distance,
        #     [robot_prefix + "ur_vacuum", box],
        #     ry.OT.sos,
        #     [1e0],
        #     [0.05],
        # )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.positionDiff,
        #     [robot_prefix + "ur_vacuum", box],
        #     ry.OT.sos,
        #     [1e1, 1e1, 1],
        # )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.scalarProductYZ,
        #     [robot_prefix + "ur_ee_marker", box],
        #     ry.OT.sos,
        #     [1e1],
        # )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.scalarProductZZ,
        #     [robot_prefix + "ur_ee_marker", box],
        #     ry.OT.sos,
        #     [1e1],
        # )

        # for pick and place directly
        # komo.addModeSwitch([2, -1], ry.SY.stable, ["table", box])
        komo.addObjective([2, -1], ry.FS.poseDiff, [goal, box], ry.OT.eq, [1e1])

        komo.addObjective(
            times=[3, -1],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.eq,
            scale=[1e0],
            target=q_home,
        )

        max_attempts = 5
        for num_attempt in range(max_attempts):
            # komo.initRandom()
            if num_attempt > 0:
                dim = len(c_tmp.getJointState())
                x_init = np.random.rand(dim) * 3 - 1.5
                komo.initWithConstant(x_init)
                # komo.initWithPath(np.random.rand(3, 12) * 5 - 2.5)

            solver = ry.NLP_Solver(komo.nlp(), verbose=4)
            # options.nonStrictSteps = 50;

            # solver.setOptions(damping=0.01, wolfe=0.001)
            # solver.setOptions(damping=0.001)
            retval = solver.solve()
            retval = retval.dict()

            # print(retval)

            # if view:
            # komo.view(True, "IK solution")

            print(retval)

            if retval["ineq"] < 1 and retval["eq"] < 1 and retval["feasible"]:
                # komo.view(True, "IK solution")
                keyframes = komo.getPath()
                return keyframes

        return None

    # all_robots = ["a1_", "a2_", "a3_", "a4_"]
    # all_robots = all_robots[:num_robots]

    direct_pick_place_keyframes = {}

    for r in all_robots:
        direct_pick_place_keyframes[r] = {}

    c_tmp = ry.Config()
    c_tmp.addConfigurationCopy(C)

    robot_to_use = []

    for box, goal in zip(boxes, goals):
        c_tmp_2 = ry.Config()
        c_tmp_2.addConfigurationCopy(c_tmp)
        # c_tmp_2.computeCollisions()

        c_tmp_2.getFrame(box).setRelativePosition([0, 0, 0.6])
        c_tmp_2.getFrame(box).setContact(1)

        while True:
            r = random.choice(all_robots)
            r1 = compute_rearrangment(c_tmp_2, r, box, goal)

            if r1 is not None:
                break

        direct_pick_place_keyframes[r][box] = r1[:2]
        robot_to_use.append(r)

        c_tmp.getFrame(box).setRelativePosition(
            c_tmp.getFrame(goal).getRelativePosition()
        )

    box_goal = {}
    for b, g in zip(boxes, goals):
        box_goal[b] = g

    keyframes = []

    for r, b, g in zip(robot_to_use, boxes, goals):
        keyframes.append(
            (
                r,
                b,
                direct_pick_place_keyframes[r][b],
                goal,
            )
        )

    return C, keyframes, all_robots


def make_handover_env(view: bool = False):
    C = ry.Config()

    C.addFrame("floor").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(0)

    table = (
        C.addFrame("table")
        .setPosition([0, 0, 0.2])
        .setShape(ry.ST.box, size=[2, 3, 0.06, 0.005])
        .setColor([0.6, 0.6, 0.6])
        .setContact(1)
    )

    # C.addFile(ry.raiPath('panda/panda.g'), namePrefix='a1_') \
    #         .setParent(C.getFrame('table')) \
    #         .setRelativePosition([-0.3, 0.5, 0]) \
    #         .setRelativeQuaternion([0.7071, 0, 0, -0.7071]) \
    robot_path = os.path.join(os.path.dirname(__file__), "../models/ur10/ur10_vacuum.g")

    print(robot_path)

    C.addFile(robot_path, namePrefix="a1_").setParent(table).setRelativePosition(
        [-0.5, -0.5, 0.0]
    ).setRelativeQuaternion([0.7071, 0, 0, 0.7071]).setJoint(ry.JT.rigid)

    # C.getFrame('a1_ur_coll0').setContact(-2)

    C.addFile(robot_path, namePrefix="a2_").setParent(table).setRelativePosition(
        [+0.5, 0.5, 0.0]
    ).setRelativeQuaternion([0.7071, 0, 0, -0.7071]).setJoint(ry.JT.rigid)

    C.addFrame("obj1").setParent(table).setShape(
        ry.ST.box, size=[0.4, 0.4, 0.2, 0.005]
    ).setColor([1, 0.5, 0, 1]).setContact(1).setRelativePosition(
        [+0.5, -1.0, 0.15]
    ).setJoint(ry.JT.rigid)

    C.addFrame("goal1").setParent(table).setShape(
        ry.ST.box, size=[0.4, 0.4, 0.2, 0.005]
    ).setColor([1, 0.5, 0, 0.2]).setContact(0).setRelativePosition(
        [-0.5, 1.0, 0.15]
    ).setJoint(ry.JT.rigid)

    C.addFrame("obs1").setParent(table).setRelativePosition([-0.5, 1, 0.7]).setShape(
        ry.ST.box, size=[1, 1, 0.1, 0.005]
    ).setColor([0.3, 0.3, 0.3]).setContact(1)

    C.addFrame("obs2").setParent(table).setRelativePosition([0.5, -1, 0.7]).setShape(
        ry.ST.box, size=[1, 1, 0.1, 0.005]
    ).setColor([0.3, 0.3, 0.3]).setContact(1)

    if view:
        C.view(True)

    qHome = C.getJointState()
    box = "obj1"

    komo = ry.KOMO(C, phases=4, slicesPerPhase=1, kOrder=1, enableCollisions=True)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [-0.01])

    komo.addControlObjective([], 0, 1e-1)
    komo.addControlObjective([], 1, 1e-1)
    # komo.addControlObjective([], 2, 1e-1)

    komo.addModeSwitch([1, 2], ry.SY.stable, ["a1_" + "ur_vacuum", box])
    komo.addObjective(
        [1, 2], ry.FS.distance, ["a1_" + "ur_vacuum", box], ry.OT.sos, [1e1], [-0.0]
    )
    komo.addObjective(
        [1, 2],
        ry.FS.positionDiff,
        ["a1_" + "ur_vacuum", box],
        ry.OT.sos,
        [1e1, 1e1, 1e1],
    )
    # komo.addObjective(
    #     [1, 2],
    #     ry.FS.positionDiff,
    #     ["a1_" + "ur_ee_marker", box],
    #     ry.OT.sos,
    #     [1e0],
    # )
    komo.addObjective(
        [1, 2],
        ry.FS.scalarProductYZ,
        ["a1_" + "ur_ee_marker", box],
        ry.OT.sos,
        [1e0],
    )
    komo.addObjective(
        [1, 2],
        ry.FS.scalarProductZZ,
        ["a1_" + "ur_ee_marker", box],
        ry.OT.sos,
        [1e0],
    )

    komo.addModeSwitch([2, 3], ry.SY.stable, ["a2_" + "ur_vacuum", box])
    komo.addObjective(
        [2, 3], ry.FS.distance, ["a2_" + "ur_vacuum", box], ry.OT.sos, [1e1], [-0.0]
    )
    komo.addObjective(
        [2, 3],
        ry.FS.positionDiff,
        ["a2_" + "ur_vacuum", box],
        ry.OT.sos,
        [1e1, 1e1, 1e1],
    )
    # komo.addObjective(
    #     [2, 3],
    #     ry.FS.positionDiff,
    #     ["a2_" + "ur_ee_marker", box],
    #     ry.OT.sos,
    #     [1e0],
    # )
    komo.addObjective(
        [2, 3],
        ry.FS.scalarProductYZ,
        ["a2_" + "ur_ee_marker", box],
        ry.OT.sos,
        [1e0],
    )
    komo.addObjective(
        [2, 3],
        ry.FS.scalarProductZZ,
        ["a2_" + "ur_ee_marker", box],
        ry.OT.sos,
        [1e0],
    )

    komo.addModeSwitch([3, -1], ry.SY.stable, ["table", box])
    komo.addObjective([3, -1], ry.FS.poseDiff, ["goal1", box], ry.OT.eq, [1e1])

    komo.addObjective(
        times=[4],
        feature=ry.FS.jointState,
        frames=[],
        type=ry.OT.eq,
        scale=[1e0],
        target=qHome,
    )

    komo.initRandom()

    solver = ry.NLP_Solver(komo.nlp(), verbose=4)
    solver.setOptions(damping=0.1, wolfe=0.001)
    retval = solver.solve()

    if view:
        print(retval.dict())
        komo.view(True, "IK solution")

    keyframes = komo.getPath()

    return C, keyframes


def make_bimanual_grasping_env(view: bool = False):
    C = ry.Config()

    C.addFrame("floor").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(0)

    table = (
        C.addFrame("table")
        .setPosition([0, 0, 0.2])
        .setShape(ry.ST.box, size=[2, 3, 0.06, 0.005])
        .setColor([0.6, 0.6, 0.6])
        .setContact(1)
    )

    # C.addFile(ry.raiPath('panda/panda.g'), namePrefix='a1_') \
    #         .setParent(C.getFrame('table')) \
    #         .setRelativePosition([-0.3, 0.5, 0]) \
    #         .setRelativeQuaternion([0.7071, 0, 0, -0.7071]) \
    robot_path = os.path.join(os.path.dirname(__file__), "../models/ur10/ur10_vacuum.g")

    print(robot_path)

    C.addFile(robot_path, namePrefix="a1_").setParent(table).setRelativePosition(
        [-0.75, 0, 0.0]
    ).setRelativeQuaternion([1, 0, 0, 0]).setJoint(ry.JT.rigid)

    # C.getFrame('a1_ur_coll0').setContact(-2)

    C.addFile(robot_path, namePrefix="a2_").setParent(table).setRelativePosition(
        [+0.75, 0, 0.0]
    ).setRelativeQuaternion([0, 0, 0, 1]).setJoint(ry.JT.rigid)

    C.addFrame("obj1").setParent(table).setShape(
        ry.ST.box, size=[0.2, 0.4, 0.2, 0.005]
    ).setColor([1, 0.5, 0, 1]).setContact(1).setRelativePosition(
        [0, -0.75, 0.15]
    ).setJoint(ry.JT.rigid)

    C.addFrame("goal1").setParent(table).setShape(
        ry.ST.box, size=[0.2, 0.4, 0.2, 0.005]
    ).setColor([1, 0.5, 0, 0.2]).setContact(0).setRelativePosition(
        [-0., 0., 0.15]
    ).setJoint(ry.JT.rigid).setRelativeQuaternion([0.7071, 0, 0, -0.7071])

    C.addFrame("obj_marker").setParent(
        C.getFrame("obj1")
    ).setShape(
        ry.ST.marker, [0.2]
    ).setRelativePosition([0, 0, 0]).setColor(
        [0, 0, 0.1, 0.1]
    ).setContact(0).setJoint(ry.JT.rigid)

    C.addFrame("goal_marker").setParent(
        C.getFrame("goal1")
    ).setShape(
        ry.ST.marker, [0.2]
    ).setRelativePosition([0, 0, 0]).setColor(
        [0, 0, 0.1, 0.1]
    ).setContact(0).setJoint(ry.JT.rigid)

    C.addFrame("ee_marker").setParent(
        C.getFrame("a1_ur_ee_marker")
    ).setShape(
        ry.ST.marker, [0.2]
    ).setRelativePosition([0, 0, 0]).setColor(
        [0, 0, 0.1, 0.1]
    ).setContact(0).setJoint(ry.JT.rigid)


    # C.addFrame("obs1").setParent(table).setRelativePosition([-0.5, 1, 0.7]).setShape(
    #     ry.ST.box, size=[1, 1, 0.1, 0.005]
    # ).setColor([0.3, 0.3, 0.3]).setContact(1)

    # C.addFrame("obs2").setParent(table).setRelativePosition([0.5, -1, 0.7]).setShape(
    #     ry.ST.box, size=[1, 1, 0.1, 0.005]
    # ).setColor([0.3, 0.3, 0.3]).setContact(1)

    robots = ["a1_", "a2_"]

    if view:
        C.view(True)

    qHome = C.getJointState()
    box = "obj1"

    komo = ry.KOMO(C, phases=3, slicesPerPhase=1, kOrder=1, enableCollisions=True)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [-0.0])

    komo.addControlObjective([], 0, 1e-1)
    # komo.addControlObjective([], 1, 1e-1)
    # komo.addControlObjective([], 2, 1e-1)

    komo.addModeSwitch([1, 2], ry.SY.stable, ["a1_" + "ur_vacuum", box])
    # komo.addObjective(
    #     [1, 2], ry.FS.distance, ["a1_" + "ur_vacuum", box], ry.OT.sos, [1e1], [-0.0]
    # )
    komo.addObjective(
        [1, 2],
        ry.FS.positionRel,
        ["a1_" + "ur_vacuum", box],
        ry.OT.eq,
        [1e1, 1e1, 1e1],
        [-0.2, 0, 0]
    )
    komo.addObjective(
        [1, 2],
        ry.FS.positionRel,
        ["a2_" + "ur_vacuum", box],
        ry.OT.eq,
        [1e1, 1e1, 1e1],
        [0.2, 0, 0]
    )

    # komo.addObjective(
    #     [1, 2],
    #     ry.FS.positionDiff,
    #     ["a1_" + "ur_ee_marker", box],
    #     ry.OT.sos,
    #     [1e0],
    # )
    komo.addObjective(
        [1, 2],
        ry.FS.scalarProductXZ,
        ["a1_" + "ur_ee_marker", box],
        ry.OT.eq,
        [1e1],
        [-1]
    )
    komo.addObjective(
        [1, 2],
        ry.FS.scalarProductXZ,
        ["a2_" + "ur_ee_marker", box],
        ry.OT.eq,
        [1e1],
        [-1]
    )

    komo.addObjective(
        [1, 2],
        ry.FS.scalarProductXZ,
        [box, "a1_" + "ur_ee_marker"],
        ry.OT.eq,
        [1e1],
        [1]
    )
    komo.addObjective(
        [1, 2],
        ry.FS.scalarProductYX,
        ["a2_" + "ur_ee_marker", box],
        ry.OT.eq,
        [1e1],
        [-1]
    )

    # komo.addObjective(
    #     [1, 2],
    #     ry.FS.scalarProductZZ,
    #     ["a1_" + "ur_ee_marker", box],
    #     ry.OT.sos,
    #     [1e0],
    # )

    komo.addModeSwitch([2, -1], ry.SY.stable, ["table", box])
    komo.addObjective([2, -1], ry.FS.poseDiff, ["goal1", box], ry.OT.eq, [1e1])

    komo.addObjective(
        times=[3],
        feature=ry.FS.jointState,
        frames=[],
        type=ry.OT.sos,
        scale=[1e0],
        target=qHome,
    )

    max_attempts = 30
    for i in range(max_attempts):
        if i > 0:
            komo.initRandom()
            # komo.initWithConstant(np.random.rand(len(q_home)) * 4)
            # x_init = q_home + np.random.randn(len(q_home)) * 0.1
            # komo.initWithConstant(x_init)

        solver = ry.NLP_Solver(komo.nlp(), verbose=4)
        # options.nonStrictSteps = 50;

        # solver.setOptions(damping=0.01, wolfe=0.001)
        # solver.setOptions(damping=0.001)
        retval = solver.solve()
        retval = retval.dict()

        print(retval)

        keyframes = komo.getPath()

        print(keyframes)

        if view:
            komo.view(True, "IK solution")
        # komo.view(True, "IK solution")

        # print(retval)

        if retval["ineq"] < 1 and retval["eq"] < 1 and retval["feasible"]:
            komo.view(True, "IK solution")

            keyframes = keyframes[:-1, :]
            break
            # return keyframes[:-1, :]

    return C, robots, keyframes


def make_panda_waypoint_env(
    num_robots: int = 3,
    num_waypoints: int = 6,
    mixed_robots: bool = True,
    view: bool = False,
):
    if num_robots > 3:
        raise NotImplementedError("More than three robot arms are not supported.")

    if num_waypoints > 6:
        raise NotImplementedError("More than six waypoints are not supported.")

    C = ry.Config()

    C.addFrame("floor").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(0)

    C.addFrame("table").setPosition([0, 0, 0.5]).setShape(
        ry.ST.box, size=[2, 2, 0.06, 0.005]
    ).setColor([0.6, 0.6, 0.6]).setContact(1)

    panda_path = ry.raiPath("panda/panda.g")
    ur10_path = os.path.join(
        os.path.dirname(__file__), "../models/ur10/ur10_two_finger.g"
    )
    kuka_path = os.path.join(
        os.path.dirname(__file__), "../models/kuka_drake/kuka_two_finger.g"
    )

    C.addFile(panda_path, namePrefix="a0_").setParent(
        C.getFrame("table")
    ).setRelativePosition([0.0, -0.5, 0]).setRelativeQuaternion([0.7071, 0, 0, 0.7071])

    ees = ["a0_gripper"]

    # this could likely be done nicer
    if num_robots > 1:
        path = panda_path
        ee = "a1_gripper"

        if mixed_robots:
            path = ur10_path
            ee = "a1_ur_gripper_center"
        ees.append(ee)

        C.addFile(path, namePrefix="a1_").setParent(
            C.getFrame("table")
        ).setRelativePosition([-0.4, 0.5, 0]).setRelativeQuaternion(
            [0.7071, 0, 0, -0.7071]
        )
    if num_robots > 2:
        path = panda_path
        ee = "a2_gripper"

        if mixed_robots:
            path = kuka_path
            ee = "a2_kuka_gripper_center"

        ees.append(ee)

        C.addFile(path, namePrefix="a2_").setParent(
            C.getFrame("table")
        ).setRelativePosition([+0.4, 0.5, 0]).setRelativeQuaternion(
            [0.7071, 0, 0, -0.7071]
        )

    C.addFrame("way1").setShape(ry.ST.marker, [0.1]).setPosition([0.3, 0.0, 1.0])
    C.addFrame("way2").setShape(ry.ST.marker, [0.1]).setPosition([0.3, 0.0, 1.4])
    C.addFrame("way3").setShape(ry.ST.marker, [0.1]).setPosition([-0.3, 0.0, 1.0])
    C.addFrame("way4").setShape(ry.ST.marker, [0.1]).setPosition([-0.3, 0.0, 1.4])
    C.addFrame("way5").setShape(ry.ST.marker, [0.1]).setPosition([-0.3, 0.0, 0.6])
    C.addFrame("way6").setShape(ry.ST.marker, [0.1]).setPosition([0.3, 0.0, 0.6])

    q_init = C.getJointState()
    q_init[1] -= 0.5

    if num_robots > 1 and not mixed_robots:
        q_init[8] -= 0.5
    if num_robots > 2:
        if not mixed_robots:
            q_init[15] -= 0.5

    C.setJointState(q_init)

    if view:
        C.view(True)

    qHome = C.getJointState()

    def compute_pose_for_robot(robot_ee):
        komo = ry.KOMO(
            C,
            phases=num_waypoints + 1,
            slicesPerPhase=1,
            kOrder=1,
            enableCollisions=True,
        )
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])

        komo.addControlObjective([], 0, 1e-1)
        komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        for i in range(num_waypoints):
            komo.addObjective(
                [i + 1],
                ry.FS.positionDiff,
                [robot_ee, "way" + str(i + 1)],
                ry.OT.eq,
                [1e1],
            )

        # for i in range(7):
        #     komo.addObjective([i], ry.FS.jointState, [], ry.OT.eq, [1e1], [], order=1)

        komo.addObjective(
            times=[num_waypoints + 1],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.eq,
            scale=[1e0],
            target=qHome,
        )

        ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
        # print(ret)
        q = komo.getPath()
        # print(q)

        if view:
            komo.view(True, "IK solution")

        return q

    keyframes = compute_pose_for_robot("a0_gripper")

    if num_robots > 1:
        keyframes_a1 = compute_pose_for_robot(ees[1])
        keyframes = np.concatenate([keyframes, keyframes_a1])
    if num_robots > 2:
        keyframes_a2 = compute_pose_for_robot(ees[2])
        keyframes = np.concatenate([keyframes, keyframes_a2])

    return C, keyframes


def make_panda_single_joint_goal_env(
    num_robots: int = 3, num_waypoints: int = 6, view: bool = False
):
    if num_robots > 3:
        raise NotImplementedError("More than three robot arms are not supported.")

    if num_waypoints > 6:
        raise NotImplementedError("More than six waypoints are not supported.")

    C = ry.Config()
    # C.addFile(ry.raiPath('scenarios/pandaSingle.g'))

    C.addFrame("floor").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(0)

    C.addFrame("table").setPosition([0, 0, 0.5]).setShape(
        ry.ST.box, size=[2, 2, 0.06, 0.005]
    ).setColor([0.6, 0.6, 0.6]).setContact(1)

    robot_path = ry.raiPath("panda/panda.g")
    # robot_path = "ur10/ur10_vacuum.g"

    C.addFile(robot_path, namePrefix="a0_").setParent(
        C.getFrame("table")
    ).setRelativePosition([0.0, -0.5, 0]).setRelativeQuaternion([0.7071, 0, 0, 0.7071])

    # this could likely be done nicer
    if num_robots > 1:
        C.addFile(robot_path, namePrefix="a1_").setParent(
            C.getFrame("table")
        ).setRelativePosition([-0.3, 0.5, 0]).setRelativeQuaternion(
            [0.7071, 0, 0, -0.7071]
        )
    if num_robots > 2:
        C.addFile(robot_path, namePrefix="a2_").setParent(
            C.getFrame("table")
        ).setRelativePosition([+0.3, 0.5, 0]).setRelativeQuaternion(
            [0.7071, 0, 0, -0.7071]
        )

    C.addFrame("way1").setShape(ry.ST.marker, [0.1]).setPosition([0.3, 0.0, 1.0])
    C.addFrame("way2").setShape(ry.ST.marker, [0.1]).setPosition([0.3, 0.0, 1.4])
    C.addFrame("way3").setShape(ry.ST.marker, [0.1]).setPosition([-0.3, 0.0, 1.0])
    C.addFrame("way4").setShape(ry.ST.marker, [0.1]).setPosition([-0.3, 0.0, 1.4])
    C.addFrame("way5").setShape(ry.ST.marker, [0.1]).setPosition([-0.3, 0.0, 0.6])
    C.addFrame("way6").setShape(ry.ST.marker, [0.1]).setPosition([0.3, 0.0, 0.6])

    waypoints = [f"way{i}" for i in range(1, 7)]

    q_init = C.getJointState()
    q_init[1] -= 0.5

    if num_robots > 1:
        q_init[8] -= 0.5
    if num_robots > 2:
        q_init[15] -= 0.5

    C.setJointState(q_init)

    if view:
        C.view(True)

    qHome = C.getJointState()

    def compute_pose_for_robot(ees, goals):
        komo = ry.KOMO(
            C,
            phases=num_waypoints + 1,
            slicesPerPhase=1,
            kOrder=1,
            enableCollisions=True,
        )
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])

        komo.addControlObjective([], 0, 1e-1)
        komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        for robot_ee, goal in zip(ees, goals):
            komo.addObjective(
                [1],
                ry.FS.positionDiff,
                [robot_ee, goal],
                ry.OT.eq,
                [1e1],
            )

        ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
        print(ret)
        q = komo.getPath()
        # print(q)

        if view:
            komo.view(True, "IK solution")

        return q

    ees = ["a0_gripper"]

    if num_robots > 1:
        ees.append("a1_gripper")
    if num_robots > 2:
        ees.append("a2_gripper")

    goals = ["way3", "way1", "way5"]
    goals = goals[: len(ees)]

    print(goals)

    keyframes = compute_pose_for_robot(ees, goals)

    return C, keyframes


def quaternion_from_z_rotation(angle):
    half_angle = angle / 2
    w = np.cos(half_angle)
    x = 0
    y = 0
    z = np.sin(half_angle)
    return np.array([w, x, y, z])


def make_welding_env(num_robots=4, num_pts=4, view: bool = False):
    assert num_robots <= 4, "Number of robots should be less than or equal to 4"

    C = ry.Config()

    C.addFrame("floor").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(0)

    robot_path = os.path.join(os.path.dirname(__file__), "../models/ur10/ur_welding.g")

    C.addFrame("table").setPosition([0, 0, 0.5]).setShape(
        ry.ST.box, size=[2, 2, 0.06, 0.005]
    ).setColor([0.6, 0.6, 0.6]).setContact(1)

    C.addFile(robot_path, namePrefix="a1_").setParent(
        C.getFrame("table")
    ).setRelativePosition([-0.7, 0.7, 0]).setJoint(ry.JT.rigid).setRelativeQuaternion(
        quaternion_from_z_rotation(-45 / 180 * np.pi)
    )
    if num_robots > 1:
        C.addFile(robot_path, namePrefix="a2_").setParent(
            C.getFrame("table")
        ).setRelativePosition([+0.7, 0.7, 0]).setJoint(
            ry.JT.rigid
        ).setRelativeQuaternion(quaternion_from_z_rotation(225 / 180 * np.pi))

    if num_robots > 2:
        C.addFile(robot_path, namePrefix="a3_").setParent(
            C.getFrame("table")
        ).setRelativePosition([-0.7, -0.7, 0]).setJoint(
            ry.JT.rigid
        ).setRelativeQuaternion(quaternion_from_z_rotation(45 / 180 * np.pi))

    if num_robots > 3:
        C.addFile(robot_path, namePrefix="a4_").setParent(
            C.getFrame("table")
        ).setRelativePosition([+0.7, -0.7, 0]).setJoint(
            ry.JT.rigid
        ).setRelativeQuaternion(quaternion_from_z_rotation(135 / 180 * np.pi))

    C.addFrame("obs1").setParent(C.getFrame("table")).setRelativePosition(
        [-0.2, -0.2, 0.3]
    ).setShape(ry.ST.box, size=[0.1, 0.1, 0.3, 0.005]).setColor(
        [0.3, 0.3, 0.3]
    ).setContact(1)

    C.addFrame("obs2").setParent(C.getFrame("table")).setRelativePosition(
        [-0.2, 0.2, 0.3]
    ).setShape(ry.ST.box, size=[0.1, 0.1, 0.3, 0.005]).setColor(
        [0.3, 0.3, 0.3]
    ).setContact(1)

    C.addFrame("obs3").setParent(C.getFrame("table")).setRelativePosition(
        [0.2, 0.2, 0.3]
    ).setShape(ry.ST.box, size=[0.1, 0.1, 0.3, 0.005]).setColor(
        [0.3, 0.3, 0.3]
    ).setContact(1)

    C.addFrame("obs4").setParent(C.getFrame("table")).setRelativePosition(
        [0.2, -0.2, 0.3]
    ).setShape(ry.ST.box, size=[0.1, 0.1, 0.3, 0.005]).setColor(
        [0.3, 0.3, 0.3]
    ).setContact(1)

    C.addFrame("obs5").setParent(C.getFrame("table")).setRelativePosition(
        [0.0, -0.2, 0.4]
    ).setShape(ry.ST.box, size=[0.3, 0.1, 0.1, 0.005]).setColor(
        [0.3, 0.3, 0.3]
    ).setContact(1)

    C.addFrame("obs6").setParent(C.getFrame("table")).setRelativePosition(
        [0.0, 0.2, 0.4]
    ).setShape(ry.ST.box, size=[0.3, 0.1, 0.1, 0.005]).setColor(
        [0.3, 0.3, 0.3]
    ).setContact(1)

    C.addFrame("obs7").setParent(C.getFrame("table")).setRelativePosition(
        [-0.2, 0, 0.4]
    ).setShape(ry.ST.box, size=[0.1, 0.3, 0.1, 0.005]).setColor(
        [0.3, 0.3, 0.3]
    ).setContact(1)

    C.addFrame("obs8").setParent(C.getFrame("table")).setRelativePosition(
        [0.2, 0, 0.4]
    ).setShape(ry.ST.box, size=[0.1, 0.3, 0.1, 0.005]).setColor(
        [0.3, 0.3, 0.3]
    ).setContact(1)

    if view:
        C.view(True)

    qHome = C.getJointState()

    def compute_pose_for_robot(robot_ee):
        komo = ry.KOMO(
            C,
            phases=num_pts + 1,
            slicesPerPhase=1,
            kOrder=1,
            enableCollisions=True,
        )
        komo.addObjective(
            [], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [5e1], [0.01]
        )

        komo.addControlObjective([], 0, 1e-1)
        # komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        for i in range(num_pts):
            komo.addObjective(
                [i + 1],
                ry.FS.distance,
                [robot_ee, "obs" + str(i + 1)],
                ry.OT.sos,
                [1e1],
                [-0.05],
            )

            komo.addObjective(
                [i + 1],
                ry.FS.positionDiff,
                [robot_ee, "obs" + str(i + 1)],
                ry.OT.sos,
                [1e-1],
            )

            komo.addObjective(
                [i + 1],
                ry.FS.scalarProductYZ,
                [robot_ee, "obs" + str(i + 1)],
                ry.OT.sos,
                [1e-1],
            )

        # for i in range(7):
        #     komo.addObjective([i], ry.FS.jointState, [], ry.OT.eq, [1e1], [], order=1)

        komo.addObjective(
            times=[],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.sos,
            scale=[1e-1],
            target=qHome,
        )

        komo.addObjective(
            times=[num_pts + 1],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.eq,
            scale=[1e0],
            target=qHome,
        )

        # print(komo.nlp().getBounds())
        komo.initRandom()

        ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
        # print(ret.dict())
        q = komo.getPath()
        # print(q)

        if view:
            komo.view(True, "IK solution")

        return q

    robots = ["a1", "a2", "a3", "a4"]

    keyframes = np.zeros((0, len(C.getJointState())))
    for r in robots[:num_robots]:
        k = compute_pose_for_robot(r + "_ur_vacuum")
        keyframes = np.concatenate([keyframes, k])

    return C, keyframes


def make_shelf_env(view: bool = False):
    pass


def make_bottle_insertion(remove_non_moved_bottles: bool = False, view: bool = False):
    C = ry.Config()

    C.addFrame("floor").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(0)

    path = os.path.join(os.path.dirname(__file__), "../models/bottle.g")
    C.addFile(path).setPosition([1, 0, 0.2])

    if remove_non_moved_bottles:
        moved_bottle_indices = [1, 12, 3, 5]
        for i in range(1, 15):
            if i not in moved_bottle_indices:
                print(f"removing bottle {i}")
                C.delFrame("bottle_" + str(i) + "_goal")

    if view:
        C.view(True)

    qHome = C.getJointState()

    def compute_insertion_poses(ee, bottle):
        komo = ry.KOMO(C, phases=3, slicesPerPhase=1, kOrder=1, enableCollisions=True)
        komo.addObjective(
            [], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [0.01]
        )
        komo.addControlObjective([], 0, 1e-1)
        # komo.addControlObjective([], 1, 1e0)
        # komo.addControlObjective([], 2, 1e0)

        komo.addModeSwitch([1, 2], ry.SY.stable, [ee, bottle])
        komo.addObjective(
            [1, 2], ry.FS.distance, [ee, bottle], ry.OT.eq, [1e1], [-0.01]
        )

        komo.addObjective([1, 2], ry.FS.vectorYDiff, [ee, bottle], ry.OT.sos, [1e2])

        komo.addObjective(
            [2, -1], ry.FS.poseDiff, [bottle, bottle + "_goal"], ry.OT.eq, [1e1]
        )

        # komo.addModeSwitch([1, 2], ry.SY.stable, ["a2", "obj2"])
        # komo.addObjective([1, 2], ry.FS.distance, ["a2", "obj2"], ry.OT.eq, [1e1])
        # komo.addModeSwitch([2, -1], ry.SY.stable, ["table", "obj2"])
        komo.addModeSwitch([2, -1], ry.SY.stable, ["table", bottle])

        # komo.addObjective([2, -1], ry.FS.poseDiff, ["obj2", "goal2"], ry.OT.eq, [1e1])

        komo.addObjective(
            times=[],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.sos,
            scale=[1e-1],
            target=qHome,
        )

        komo.addObjective(
            times=[3],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.eq,
            scale=[1e0],
            target=qHome,
        )

        # komo.addObjective([2], ry.FS.poseDiff, ["a2", "goal2"], ry.OT.eq, [1e1])

        # komo.addObjective([3, -1], ry.FS.poseDiff, ['a1', 'goal2'], ry.OT.eq, [1e1])
        # komo.addObjective(
        #     [3, -1], ry.FS.poseDiff, ["a2", "pre_agent_2_frame"], ry.OT.eq, [1e1]
        # )

        for _ in range(10):
            komo.initRandom()

            solver = ry.NLP_Solver(komo.nlp(), verbose=4)
            # options.nonStrictSteps = 50;

            # solver.setOptions(damping=0.01, wolfe=0.001)
            # solver.setOptions(damping=0.001)
            retval = solver.solve()
            retval = retval.dict()

            # print(bottle, retval)

            if view:
                komo.view(True, "IK solution")

            keyframes = komo.getPath()

            # print(retval)

            if retval["ineq"] < 1 and retval["eq"] < 1 and retval["feasible"]:
                return keyframes

            # else:
            #     print(retval)

        return 0

    a0_b1_keyframes = compute_insertion_poses("a0_ur_vacuum", "bottle_1")
    a0_b2_keyframes = compute_insertion_poses("a0_ur_vacuum", "bottle_12")

    a1_b3_keyframes = compute_insertion_poses("a1_ur_vacuum", "bottle_3")
    a1_b5_keyframes = compute_insertion_poses("a1_ur_vacuum", "bottle_5")

    return C, np.concatenate(
        [a0_b1_keyframes, a0_b2_keyframes, a1_b3_keyframes, a1_b5_keyframes]
    )


def generate_rnd_axis_quaternion(dont_rotate_z=False):
    def align_x_with_target(target_vector):
        # Default vector to align from (1, 0, 0)
        source_vector = np.array([0.0, 0.0, 1.0])

        # Compute the cross product and dot product
        v = np.cross(source_vector, target_vector)
        c = np.dot(source_vector, target_vector)

        # Handle the edge case where the vectors are opposite
        if np.isclose(c, -1.0, atol=1e-6):
            # Rotate 180 degrees around any perpendicular axis (e.g., Y-axis)
            return (0.0, 0.0, 1.0, 0.0)  # 180° rotation about Y-axis

        # Compute the quaternion
        s = np.sqrt((1 + c) * 2)
        q = (s * 0.5, v[0] / s, v[1] / s, v[2] / s)

        return q

    def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([w, x, y, z])

    def axis_to_quaternion(axis, angle):
        axis = np.array(axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm == 0:
            raise ValueError("Axis vector must not be zero.")

        axis = axis / norm  # Normalize the axis
        half_angle = angle / 2
        w = np.cos(half_angle)
        x, y, z = axis * np.sin(half_angle)

        return np.array([w, x, y, z])

    # Randomly select one of the major axes (±X, ±Y, ±Z)
    axes = [
        (1, 0, 0),  # +X
        (-1, 0, 0),  # -X
        (0, 1, 0),  # +Y
        (0, -1, 0),  # -Y
        (0, 0, 1),  # +Z
        (0, 0, -1),  # -Z
    ]

    # Choose one axis randomly
    if not dont_rotate_z:
        chosen_axis = random.choice(axes)
    else:
        chosen_axis = axes[np.random.randint(4, 5)]

    # Compute the quaternion to align the X-axis with the selected axis
    quat = align_x_with_target(chosen_axis)

    angle = np.random.random() * np.pi * 2 - np.pi
    rotate_around_axis_quat = axis_to_quaternion(chosen_axis, angle)

    result = quaternion_multiply(quat, rotate_around_axis_quat)

    return result


def is_z_axis_up(quaternion):
    w, x, y, z = quaternion

    # Rotation matrix from quaternion
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x**2 + y**2)

    # The Z-axis vector in the rotated frame
    z_axis = np.array([r20, r21, r22])

    # Check if Z-axis points up (dot product with (0, 0, 1) is close to 1)
    return np.isclose(np.dot(z_axis, np.array([0, 0, 1])), 1.0, atol=1e-6)


def make_box_pile_env(
    num_boxes=6,
    compute_multiple_keyframes: bool = False,
    random_orientation: bool = True,
    compute_all_keyframes = False,
    view: bool = False,
):
    assert num_boxes <= 9

    C = ry.Config()

    C.addFrame("floor").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(0)

    table = (
        C.addFrame("table")
        .setPosition([0, 0, 0.2])
        .setShape(ry.ST.box, size=[2, 3, 0.06, 0.005])
        .setColor([0.6, 0.6, 0.6])
        .setContact(1)
    )

    robot_path = os.path.join(os.path.dirname(__file__), "../models/ur10/ur10_vacuum.g")

    C.addFile(robot_path, namePrefix="a1_").setParent(
        C.getFrame("table")
    ).setRelativePosition([-0.5, 0.5, 0.0]).setRelativeQuaternion(
        [0.7071, 0, 0, -0.7071]
    ).setJoint(ry.JT.rigid)

    # C.getFrame('a1_ur_coll0').setContact(-2)

    C.addFile(robot_path, namePrefix="a2_").setParent(
        C.getFrame("table")
    ).setRelativePosition([+0.5, 0.5, 0.0]).setRelativeQuaternion(
        [0.7071, 0, 0, -0.7071]
    ).setJoint(ry.JT.rigid)

    C.addFrame("tray").setParent(C.getFrame("table")).setShape(
        ry.ST.box, [0.5, 0.5, 0.025, 0.005]
    ).setPosition([0, 0, 0.25]).setMass(0.1).setColor((1, 0.1, 0.2, 1)).setContact(
        0
    ).setJoint(ry.JT.rigid)

    C.addFrame("back_wall").setParent(C.getFrame("table")).setShape(
        ry.ST.box, [2, 0.1, 1, 0.005]
    ).setPosition([0, 1.5, 0.8]).setMass(0.1).setColor((0, 0, 0, 0.01)).setContact(
        1
    ).setJoint(ry.JT.rigid)

    C.addFrame("side_wall_r").setParent(C.getFrame("table")).setShape(
        ry.ST.box, [0.1, 2.8, 1, 0.005]
    ).setPosition([-1, 0, 0.8]).setMass(0.1).setColor((0, 0, 0, 0.01)).setContact(
        1
    ).setJoint(ry.JT.rigid)

    C.addFrame("side_wall_l").setParent(C.getFrame("table")).setShape(
        ry.ST.box, [0.1, 2.8, 1, 0.005]
    ).setPosition([1, 0, 0.8]).setMass(0.1).setColor((0, 0, 0, 0.01)).setContact(
        1
    ).setJoint(ry.JT.rigid)

    size = np.array([0.1, 0.1, 0.1])

    added_boxes = 0
    while added_boxes < num_boxes:
        c_coll_tmp = ry.Config()
        c_coll_tmp.addConfigurationCopy(C)

        pos = np.array(
            [
                (random.random() - 0.5) * 1.2,
                (random.random() - 0.7) * 0.6,
                0.285,
            ]
        )

        # goal position
        if pos[0] > -0.3 and pos[0] < 0.3 and pos[1] < 0.3 and pos[1] > -0.3:
            continue

        if random_orientation:
            keep_z_aligned = np.random.rand() > 0.8
        else:
            keep_z_aligned = True
            
        quat = generate_rnd_axis_quaternion(keep_z_aligned)

        color = np.random.rand(3)

        box = (
            c_coll_tmp.addFrame("obj" + str(added_boxes))
            .setParent(c_coll_tmp.getFrame("table"))
            .setShape(ry.ST.ssBox, [size[0], size[1], size[2], 0.005])
            .setPosition([pos[0], pos[1], pos[2]])
            .setMass(0.1)
            .setColor(color)
            .setContact(1)
            .setQuaternion(quat)
            .setJoint(ry.JT.rigid)
        )

        c_coll_tmp.addFrame("marker_obj" + str(added_boxes)).setParent(box).setShape(
            ry.ST.marker, [0.1]
        ).setContact(0).setJoint(ry.JT.rigid)

        binary_collision_free = c_coll_tmp.getCollisionFree()
        if binary_collision_free:
            added_boxes += 1

            C.clear()
            C.addConfigurationCopy(c_coll_tmp)

    # with open("box_poses.json", "w") as f:
    #     box_data = []
    #     for frame in C.getFrames():
    #         if "box" == frame.name[:3]:
    #             rel_pos = frame.getRelativePosition()
    #             rel_quat = frame.getRelativeQuaternion()
    #             box_data.append({"position": rel_pos.tolist(), "quaternion": rel_quat.tolist(), "name": frame.name})
    #     # write all data as a single JSON array
    #     json.dump(box_data, f)

    # C.view(True)

    # add goal positions
    for i in range(num_boxes):
        x_ind = i % 3
        y_ind = int(i / 3)
        pos = np.array(
            [
                x_ind * 0.15 - 0.15,
                y_ind * 0.15 - 0.15,
                0.07,
            ]
        )

        box = (
            C.addFrame("goal" + str(i))
            .setParent(c_coll_tmp.getFrame("tray"))
            .setShape(ry.ST.box, [size[0] * 0.9, size[1] * 0.9, size[2] * 0.9, 0.005])
            .setRelativePosition([pos[0], pos[1], pos[2]])
            .setMass(0.1)
            .setColor((0.1, 0.1, 0.1, 0.3))
            .setContact(0)
            .setJoint(ry.JT.rigid)
        )

    if view:
        C.view(True)

    keyframes = []

    def handover(
        r1, r2, box_num, num_keyframes=1, relative_pose_at_handover=None, ref_pose=None
    ):
        c_tmp = ry.Config()
        c_tmp.addConfigurationCopy(C)

        box = "obj" + str(box_num)
        goal = "goal" + str(box_num)

        q_home = c_tmp.getJointState()

        komo = ry.KOMO(
            c_tmp, phases=4, slicesPerPhase=1, kOrder=1, enableCollisions=True
        )
        komo.addObjective(
            [], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [-0.0]
        )

        komo.addControlObjective([], 0, 1e-1)
        komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        komo.addModeSwitch([1, 2], ry.SY.stable, [r1 + "ur_vacuum", box])
        komo.addObjective(
            [1, 2],
            ry.FS.distance,
            [r1 + "ur_vacuum", box],
            ry.OT.ineq,
            [-1e0],
            [0.00],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.positionDiff,
            [r1 + "ur_vacuum", box],
            ry.OT.sos,
            [1e1, 1e1, 0],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.scalarProductXZ,
            [r1 + "ur_ee_marker", "table"],
            ry.OT.sos,
            [2e1],
            [-1],
        )

        komo.addModeSwitch([2, 3], ry.SY.stable, [r2 + "ur_vacuum", box])
        komo.addObjective(
            [2, 3],
            ry.FS.distance,
            [r2 + "ur_vacuum", box],
            ry.OT.eq,
            [-1e0],
            [0.00],
        )
        komo.addObjective(
            [2, 3],
            ry.FS.positionDiff,
            [r2 + "ur_vacuum", box],
            ry.OT.sos,
            [1e1, 1e1, 0],
        )
        komo.addObjective(
            [2, 3],
            ry.FS.scalarProductXZ,
            [r2 + "ur_ee_marker", box],
            ry.OT.sos,
            [2e1],
            [-1],
        )

        if relative_pose_at_handover is not None:
            assert ref_pose is not None
            komo.addObjective([1, 1], ry.FS.qItself, [], ry.OT.eq, [1e1], ref_pose[0])
            komo.addObjective(
                [2, 2],
                ry.FS.poseDiff,
                [r2 + "ur_ee_marker", r1 + "ur_ee_marker"],
                ry.OT.eq,
                [1e1],
                relative_pose_at_handover,
            )

        komo.addModeSwitch([3, -1], ry.SY.stable, ["table", box])
        komo.addObjective([3, -1], ry.FS.poseDiff, [goal, box], ry.OT.eq, [1e1])

        komo.addObjective(
            times=[4, -1],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.eq,
            scale=[1e0],
            target=q_home,
        )

        all_keyframes = []
        for _ in range(num_keyframes):
            max_attempts = 20
            for i in range(max_attempts):
                if i > 0 or relative_pose_at_handover is not None:
                    komo.initRandom()
                    # komo.initWithConstant(np.random.rand(len(q_home)) * 4)
                    # x_init = q_home + np.random.randn(len(q_home)) * 0.1
                    # komo.initWithConstant(x_init)

                if relative_pose_at_handover is not None:
                    p = komo.getPath()
                    p += np.random.randn(p.shape[0], p.shape[1]) * 0.1
                    komo.initWithPath(p)

                solver = ry.NLP_Solver(komo.nlp(), verbose=4)
                # options.nonStrictSteps = 50;

                # solver.setOptions(damping=0.01, wolfe=0.001)
                # solver.setOptions(damping=0.001)
                retval = solver.solve()
                retval = retval.dict()

                print(retval)

                if view:
                    komo.view(True, "IK solution")
                # komo.view(True, "IK solution")

                keyframes = komo.getPath()

                # print(retval)

                if retval["ineq"] < 1 and retval["eq"] < 1 and retval["feasible"]:
                    # komo.view(True, "IK solution")

                    all_keyframes.append(keyframes[:-1, :])
                    break
                    # return keyframes[:-1, :]

        return all_keyframes

    def pick_and_place(robot_prefix, box_num):
        c_tmp = ry.Config()
        c_tmp.addConfigurationCopy(C)

        box = "obj" + str(box_num)
        goal = "goal" + str(box_num)

        robot_base = robot_prefix + "ur_base"
        c_tmp.selectJointsBySubtree(c_tmp.getFrame(robot_base))

        q_home = c_tmp.getJointState()

        komo = ry.KOMO(
            c_tmp, phases=3, slicesPerPhase=1, kOrder=1, enableCollisions=True
        )
        komo.addObjective(
            [], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [-0.0]
        )

        komo.addControlObjective([], 0, 1e-1)
        komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        komo.addModeSwitch([1, 2], ry.SY.stable, [robot_prefix + "ur_vacuum", box])
        komo.addObjective(
            [1, 2],
            ry.FS.distance,
            [robot_prefix + "ur_vacuum", box],
            ry.OT.eq,
            [-1e0],
            [0.00],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.positionDiff,
            [robot_prefix + "ur_vacuum", box],
            ry.OT.sos,
            [1e1, 1e1, 0],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.scalarProductXZ,
            [robot_prefix + "ur_ee_marker", box],
            ry.OT.sos,
            [2e1],
            [-1],
        )

        # for pick and place directly
        komo.addModeSwitch([2, -1], ry.SY.stable, ["table", box])
        komo.addObjective([2, -1], ry.FS.poseDiff, [goal, box], ry.OT.eq, [1e1])

        komo.addObjective(
            times=[3, -1],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.eq,
            scale=[1e0],
            target=q_home,
        )

        solver = ry.NLP_Solver(komo.nlp(), verbose=10)
        solver.setOptions(damping=0.1, wolfe=0.001)
        retval = solver.solve()

        print(retval.dict())

        if view:
            komo.view(True, "IK solution")

        sol = komo.getPath()[:-1, :]

        return sol

    selected_robots = set()
    for i in range(num_boxes):
        # check how the object is oriented, and if we can grasp and place it directly
        quat = C.getFrame("obj" + str(i)).getQuaternion()
        z_up = is_z_axis_up(quat)
        robots = ["a1_", "a2_"]
        if i == num_boxes - 1 and len(selected_robots) != len(robots):
            if "a2_" not in selected_robots:
                robots = ["a2_", "a1_"]
        else:
            random.shuffle(robots)

        if z_up:
            for r in robots:
                selected_robots.add(r)
                keyframe = pick_and_place(r, i)
                print(keyframe)
                if keyframe is None:
                    continue
                keyframes.append(("pick", [r], i, [keyframe]))

                if not compute_all_keyframes:
                    break
        else:
            # otherwise a handover/regrasp is required
            found_sol = False
            for r1 in robots:
                for r2 in robots:
                    if r1 == r2:
                        continue

                    poses = handover(r1, r2, i)

                    if len(poses) == 0:
                        continue

                    if compute_multiple_keyframes:
                        home = C.getJointState()
                        C.setJointState(poses[0][1])
                        ee_r1_pose = C.getFrame(r1 + "ur_ee_marker").getPose()
                        # ee_r1_pose = C.getFrame("box" + str(i)).getPose()
                        ee_r2_pose = C.getFrame(r2 + "ur_ee_marker").getPose()
                        relative_pose = ee_r2_pose - ee_r1_pose

                        C.setJointState(home)

                        print(relative_pose)

                        keyframes_with_same_relative_pose = handover(
                            r1,
                            r2,
                            i,
                            num_keyframes=10,
                            relative_pose_at_handover=relative_pose,
                            ref_pose=poses[0],
                        )

                        poses = poses + keyframes_with_same_relative_pose

                    print(poses)

                    keyframes.append(("handover", [r1, r2], i, poses))
                    found_sol = True
                if found_sol and not compute_all_keyframes:
                    break

    return C, keyframes


def make_mobile_manip_env(num_robots=5, view: bool = False):
    C = ry.Config()

    table = (
        C.addFrame("table")
        .setPosition([0, 0, -0.02])
        .setShape(ry.ST.box, size=[20, 20, 0.06, 0.005])
        .setColor([0.6, 0.6, 0.6])
        .setContact(1)
    )

    mobile_robot_path = os.path.join(
        os.path.dirname(__file__), "../models/mobile-manipulator-restricted.g"
    )

    robots = []
    for i in range(num_robots):
        prefix = f"a{i}_"
        C.addFile(mobile_robot_path, namePrefix=prefix).setPosition([0, 0, 0.25])
        C.getFrame(prefix + "base").setColor(np.random.rand(3))
        robots.append(prefix)

    q = C.getJointState()

    base_pos = np.array(
        [[2.5, -(num_robots - 1) / 2 + i, -np.pi / 2] for i in range(num_robots)]
    )

    for i in range(num_robots):
        q[6 * i] = base_pos[i, 0]
        q[6 * i + 1] = base_pos[i, 1]
        q[6 * i + 2] = base_pos[i, 2]

        print(base_pos[i, 0])

    C.setJointState(q)

    # build a wall in the middle

    w = num_robots
    h = 2
    size = np.array([0.5, 0.25, 0.15])

    all_boxes = []

    bottom_goal_names = []

    for i in range(h):
        for j in range(w):
            pos = np.array(
                [
                    j * size[0] * 1.075 - w / 2 * size[0] + size[0] / 2,
                    -1,
                    i * size[2] * 1.05 + 0.05 + 0.1,
                ]
            )

            color = np.random.rand(3)
            box_name = "obj_" + str(i) + str(j)
            all_boxes.append(box_name)
            C.addFrame(box_name).setParent(table).setShape(
                ry.ST.box, [size[0], size[1], size[2], 0.005]
            ).setRelativePosition([pos[0], pos[1], pos[2]]).setMass(0.1).setColor(
                color
            ).setContact(1).setJoint(ry.JT.rigid)

            goal_pos = np.array(
                [
                    j * size[0] * 1.075 - w / 2 * size[0] + size[0] / 2,
                    1,
                    (1 - i) * size[2] * 1.01 + 0.05 + 0.1,
                ]
            )
            goal_name = "obj_goal_" + str(i) + str(j)
            C.addFrame(goal_name).setParent(table).setShape(
                ry.ST.box, [size[0], size[1], size[2], 0.005]
            ).setRelativePosition(goal_pos).setMass(0.1).setColor(
                [color[0], color[1], color[2], 0.5]
            ).setContact(0).setJoint(ry.JT.rigid)

            if i == 1:
                bottom_goal_names.append(goal_name)

    if view:
        C.view(True)

    def compute_pick_and_place(c_tmp, box, goal, robot_prefix):
        ee = "gripper"

        robot_base = robot_prefix + "base"
        c_tmp.selectJointsBySubtree(c_tmp.getFrame(robot_base))

        q_home = c_tmp.getJointState()

        komo = ry.KOMO(
            c_tmp, phases=3, slicesPerPhase=1, kOrder=1, enableCollisions=True
        )
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1], [-0.0])

        komo.addControlObjective([], 0, 1e-1)
        # komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        komo.addModeSwitch([1, 2], ry.SY.stable, [robot_prefix + ee, box])
        komo.addObjective(
            [1, 2], ry.FS.distance, [robot_prefix + ee, box], ry.OT.sos, [1e1], [-0.0]
        )
        komo.addObjective(
            [1, 2],
            ry.FS.positionDiff,
            [robot_prefix + ee, box],
            ry.OT.sos,
            [1e1, 1e1, 1e0],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.scalarProductZZ,
            [robot_prefix + ee, box],
            ry.OT.sos,
            [1e1],
            [1],
        )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.positionDiff,
        #     ["a1_" + "ur_ee_marker", box],
        #     ry.OT.sos,
        #     [1e0],
        # )

        # komo.addObjective(
        #     [2], ry.FS.position, ["a2"], ry.OT.sos, [1e0, 1e1, 0], [1., -0.5, 0]
        # )

        # komo.addObjective(
        #     [2], ry.FS.position, [box], ry.OT.sos, [1e0, 1e0, 0], [1, -1, 0]
        # )

        komo.addModeSwitch([2, -1], ry.SY.stable, ["table", box])
        komo.addObjective([2, -1], ry.FS.poseDiff, [goal, box], ry.OT.eq, [1e1])

        komo.addObjective(
            times=[3],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.sos,
            scale=[1e0],
            target=q_home,
        )

        for _ in range(100):
            # komo.initRandom()
            # komo.initWithConstant(np.random.rand(6) * 2)

            solver = ry.NLP_Solver(komo.nlp(), verbose=4)
            # options.nonStrictSteps = 50;

            # solver.setOptions(damping=0.01, wolfe=0.001)
            # solver.setOptions(damping=0.001)
            retval = solver.solve()
            retval = retval.dict()

            print(retval)
            if view:
                komo.view(True, "IK solution")

            if retval["ineq"] < 1 and retval["eq"] < 1 and retval["feasible"]:
                keyframes = komo.getPath()
                return keyframes

    c_tmp = ry.Config()
    c_tmp.addConfigurationCopy(C)

    keyframes = {}

    for i in range(num_robots):
        robot_prefix = f"a{i}_"
        keyframes[robot_prefix] = []

        for j in [1, 0]:
            c_tmp_2 = ry.Config()
            c_tmp_2.addConfigurationCopy(c_tmp)
            # c_tmp_2.computeCollisions()

            box = f"obj_{j}{i}"
            box_goal = f"obj_goal_{j}{i}"

            for g in bottom_goal_names:
                if g != box_goal:
                    c_tmp_2.getFrame(g).setContact(1)

            res = compute_pick_and_place(c_tmp_2, box, box_goal, robot_prefix)

            keyframes[robot_prefix].append((box, res[:-1]))

            c_tmp.getFrame(box).setRelativePosition(
                c_tmp.getFrame(box_goal).getRelativePosition()
            ).setContact(0)

    return C, keyframes


def make_depalletizing_env():
    C = ry.Config()

    path = os.path.join(os.path.dirname(__file__), "../models/rollcage.g")
    C.addFile(path).setPosition([0, 0, 0.0])

    robot_path = os.path.join(os.path.dirname(__file__), "../models/ur10/ur10_vacuum.g")

    C.addFrame("robot_1_base").setPosition([0.4, 0.8, 0.1]).setShape(
        ry.ST.box, size=[0.2, 0.2, 0.2, 0.005]
    ).setColor([0.3, 0.3, 0.3]).setContact(1)  # .setQuaternion([ 0.924, 0, -0.383, 0])

    C.addFile(robot_path, namePrefix="a1_").setParent(
        C.getFrame("robot_1_base")
    ).setRelativePosition([-0.0, 0.0, 0.1]).setRelativeQuaternion(
        [0, 0, 0, 1]
    ).setJoint(ry.JT.rigid)

    C.addFrame("robot_2_base").setPosition([-0.4, 0.8, 0.1]).setShape(
        ry.ST.box, size=[0.2, 0.2, 0.2, 0.005]
    ).setColor([0.3, 0.3, 0.3]).setContact(1)  # .setQuaternion([ 0.924, 0, 0.383, 0])

    C.addFile(robot_path, namePrefix="a2_").setParent(
        C.getFrame("robot_2_base")
    ).setRelativePosition([0.0, 0.0, 0.1]).setJoint(ry.JT.rigid)

    C.addFrame("conveyor").setPosition([0.0, 2.0, 0.4]).setShape(
        ry.ST.box, size=[0.4, 1.6, 0.01, 0.005]
    ).setColor([0.3, 0.3, 0.3]).setContact(1)  # .setQuaternion([ 0.924, 0, 0.383, 0])

    size = np.array([0.2, 0.1, 0.1])

    C.addFrame("obj").setParent(C.getFrame("floor")).setShape(
        ry.ST.box, [size[0], size[1], size[2], 0.005]
    ).setRelativePosition([0, 0, 0.1]).setMass(0.1).setColor(
        np.random.rand(3)
    ).setContact(1).setJoint(ry.JT.rigid)

    C.addFrame("goal").setParent(C.getFrame("conveyor")).setShape(
        ry.ST.box, [size[0], size[1], size[2], 0.005]
    ).setRelativePosition([0.0, -0.5, 0.1]).setMass(0.1).setColor(
        np.random.rand(3)
    ).setContact(0).setJoint(ry.JT.rigid)

    # C.view(True)

    def compute_pick_and_place(box, goal, robot_prefix):
        ee = "ur_vacuum"

        q_home = C.getJointState()

        komo = ry.KOMO(C, phases=4, slicesPerPhase=1, kOrder=1, enableCollisions=True)
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1], [-0.0])

        komo.addControlObjective([], 0, 1e-1)
        # komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        komo.addModeSwitch([1, 2], ry.SY.stable, [robot_prefix + ee, box])
        komo.addObjective(
            [1, 2], ry.FS.distance, [robot_prefix + ee, box], ry.OT.sos, [1e1], [-0.0]
        )
        komo.addObjective(
            [1, 2],
            ry.FS.positionDiff,
            [robot_prefix + ee, box],
            ry.OT.sos,
            [1e0, 1e0, 1e0],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.scalarProductXZ,
            [robot_prefix + ee, box],
            ry.OT.sos,
            [1e1],
            [-1],
        )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.positionDiff,
        #     ["a1_" + "ur_ee_marker", box],
        #     ry.OT.sos,
        #     [1e0],
        # )

        # komo.addObjective(
        #     [2], ry.FS.position, ["a2"], ry.OT.sos, [1e0, 1e1, 0], [1., -0.5, 0]
        # )

        komo.addObjective(
            [2], ry.FS.position, [box], ry.OT.sos, [1e0, 1e0, 0], [1, -1, 0]
        )

        komo.addModeSwitch([2, -1], ry.SY.stable, ["conveyor", box])
        komo.addObjective([3, -1], ry.FS.poseDiff, ["goal", box], ry.OT.eq, [1e1])

        komo.addObjective(
            times=[4],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.eq,
            scale=[1e0],
            target=q_home,
        )

        for _ in range(100):
            # komo.initRandom()
            # komo.initWithConstant(np.random.rand(6) * 2)

            solver = ry.NLP_Solver(komo.nlp(), verbose=4)
            # options.nonStrictSteps = 50;

            # solver.setOptions(damping=0.01, wolfe=0.001)
            # solver.setOptions(damping=0.001)
            retval = solver.solve()
            retval = retval.dict()

            print(retval)
            # komo.view(True, "IK solution")

            if retval["ineq"] < 1 and retval["eq"] < 1 and retval["feasible"]:
                keyframes = komo.getPath()
                return keyframes

    box = "obj"
    robot_prefix = "a1_"
    compute_pick_and_place(box, "goal", robot_prefix)

    robot_prefix = "a2_"
    compute_pick_and_place(box, "goal", robot_prefix)

    return C


def quaternion_from_z_to_target(target_z):
    # Ensure the target vector is normalized
    target_z = target_z / np.linalg.norm(target_z)

    # The source vector is the unit z vector
    source_z = np.array([0, 0, 1])

    # Compute the axis of rotation (cross product)
    axis = np.cross(source_z, target_z)

    # Compute the angle of rotation using dot product
    cos_theta = np.dot(source_z, target_z)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # Handle the case where source and target are identical (no rotation needed)
    if np.isclose(angle, 0):
        return np.array([1, 0, 0, 0])  # Quaternion for no rotation

    # Handle the case where source and target are opposite (180-degree rotation)
    if np.isclose(angle, np.pi):
        # Choose an arbitrary orthogonal vector for the axis
        axis = np.cross(source_z, np.array([1, 0, 0]))
        if np.linalg.norm(axis) < 1e-6:  # If the axis is still zero, try another vector
            axis = np.cross(source_z, np.array([0, 1, 0]))
        axis = axis / np.linalg.norm(axis)

    # Normalize the rotation axis
    axis = axis / np.linalg.norm(axis)

    # Compute the quaternion components
    qw = np.cos(angle / 2)
    qx = axis[0] * np.sin(angle / 2)
    qy = axis[1] * np.sin(angle / 2)
    qz = axis[2] * np.sin(angle / 2)

    return np.array([qw, qx, qy, qz])


def make_strut_assembly_problem():
    C = ry.Config()

    C.addFrame("table").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(0)

    mobile_robot_path = os.path.join(
        os.path.dirname(__file__), "../models/mobile_manipulator_additional_gripper_rot.g"
    )

    C.addFile(mobile_robot_path, namePrefix="a0_").setPosition([1, 2, 0.2])
    C.addFile(mobile_robot_path, namePrefix="a1_").setPosition([-1, 2, 0.2])

    C.addFrame("strut_table").setParent(C.getFrame("table")).setRelativePosition([0, 2.5, 0.15]).setShape(
        ry.ST.box, size=[0.3, 1, 0.27, 0.005]
    ).setColor([0.3, 0.3, 0.3, 1]).setContact(1)

    robots = ["a0_", "a1_"]

    # assembly_path = os.path.join(os.path.dirname(__file__), "../models/strut_assemblies/yijiang_strut.json")
    assembly_path = os.path.join(os.path.dirname(__file__), "../models/strut_assemblies/florian_strut.json")
    # assembly_path = os.path.join(os.path.dirname(__file__), "../models/strut_assemblies/z_shape.json")
    # assembly_path = os.path.join(os.path.dirname(__file__), "../models/strut_assemblies/tower.json")
    # assembly_path = os.path.join(os.path.dirname(__file__), "../models/strut_assemblies/bridge.json")
    # assembly_path = os.path.join(
    #     os.path.dirname(__file__), "../models/strut_assemblies/roboarch.json"
    # )

    objects = []
    goals = []

    with open(assembly_path) as f:
        d = json.load(f)

        sequence = d["elements"]
        nodes = d["nodes"]

        num_parts = len(sequence)

        print(num_parts)

        if "assembly_sequence" in d and len(d["assembly_sequence"]) > 0:
            assembly_sequence = []
            for seq in d["assembly_sequence"]:
                assembly_sequence.extend(seq["installPartIDs"])
        else:
            assembly_sequence = np.arange(0, num_parts)

        for i in assembly_sequence:
            s = sequence[i]
            i1, i2 = s["end_node_inds"]
            print(i1, i2)

            p1 = np.array(nodes[i1]["point"])
            p2 = np.array(nodes[i2]["point"])

            z_vec = p2 - p1
            origin = p1 + (p2 - p1) / 2
            origin[2] += 0.05
            length = np.linalg.norm(p2 - p1) * 0.92

            quat = quaternion_from_z_to_target(z_vec)

            goal_name = "goal_" + str(i)
            obj_name = "obj_" + str(i)

            C.addFrame(goal_name).setParent(C.getFrame("table")).setRelativePosition(origin).setShape(
                ry.ST.box, size=[0.01, 0.01, length, 0.005]
            ).setColor([0.3, 0.3, 0.3, 0.5]).setContact(0).setQuaternion(quat)

            C.addFrame(obj_name).setParent(C.getFrame("table")).setPosition(np.array([0, 2.5, 0.3])).setShape(
                ry.ST.cylinder, size=[length, 0.007]
            ).setColor([1, 0.3, 0.3, 1]).setContact(0).setQuaternion([1, 1, 0, 0]).setJoint(ry.JT.rigid)

            # C.addFrame(f"marker_{i}").setParent(C.getFrame(obj_name)).setRelativePosition(np.array([0, 0, 0])).setShape(
            #     ry.ST.marker, size=[0.1]
            # ).setColor([1, 0.3, 0.3, 1])

            # C.view(True)

            objects.append(obj_name)
            goals.append(goal_name)

        # C.view(True)

    def compute_pick_and_place(c_tmp, robot_prefix, box, goal):
        ee = "gripper"

        robot_base = robot_prefix + "base"
        c_tmp.selectJointsBySubtree(c_tmp.getFrame(robot_base))

        q_home = c_tmp.getJointState()

        komo = ry.KOMO(
            c_tmp, phases=2, slicesPerPhase=1, kOrder=1, enableCollisions=True
        )
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e2], [-0.0])

        komo.addControlObjective([], 0, 1e-1)
        # komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        komo.addModeSwitch([1, 2], ry.SY.stable, [robot_prefix + ee, box])
        # komo.addObjective(
        #     [1, 2], ry.FS.distance, [robot_prefix + ee, box], ry.OT.sos, [1e1], [-0.01]
        # )
        komo.addObjective(
            [1, 2],
            ry.FS.positionDiff,
            [robot_prefix + "gripper_marker", box],
            ry.OT.eq,
            [1e0, 1e0, 1e1],
            [0, 0, 0.]
        )
        komo.addObjective(
            [1, 2],
            ry.FS.scalarProductZZ,
            [robot_prefix + ee, box],
            ry.OT.sos,
            [1e0],
            [0],
        )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.positionDiff,
        #     ["a1_" + "ur_ee_marker", box],
        #     ry.OT.sos,
        #     [1e0],
        # )

        # komo.addObjective(
        #     [2], ry.FS.position, ["a2"], ry.OT.sos, [1e0, 1e1, 0], [1., -0.5, 0]
        # )

        # komo.addObjective(
        #     [2], ry.FS.position, [box], ry.OT.sos, [1e0, 1e0, 0], [1, -1, 0]
        # )

        komo.addModeSwitch([2, -1], ry.SY.stable, [goal, box])
        komo.addObjective([2, -1], ry.FS.poseDiff, [goal, box], ry.OT.eq, [1e1])

        komo.addObjective(
            times=[3],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.sos,
            scale=[1e0],
            target=q_home,
        )

        max_attempts = 100
        for num_attempt in range(max_attempts):
            # komo.initRandom()
            if num_attempt > 0:
                dim = len(c_tmp.getJointState())
                x_init = np.random.rand(dim) * 5. - 2.5
                komo.initWithConstant(x_init)
                # komo.initWithPath(np.random.rand(3, 12) * 5 - 2.5)

            solver = ry.NLP_Solver(komo.nlp(), verbose=4)
            # options.nonStrictSteps = 50;

            # solver.setOptions(damping=0.01, wolfe=0.001)
            # solver.setOptions(damping=0.001)
            retval = solver.solve()
            retval = retval.dict()

            # print(retval)

            # if view:
            print(retval)

            # komo.view(True, "IK solution")

            # print(retval)

            if retval["ineq"] < 1 and retval["eq"] < 1 and retval["feasible"]:
                # komo.view(True, "IK solution")
                keyframes = komo.getPath()
                return keyframes

    keyframes = []
    assigned_robots = []

    c_tmp = ry.Config()
    c_tmp.addConfigurationCopy(C)

    for i, obj in enumerate(objects):
        c_tmp_2 = ry.Config()
        c_tmp_2.addConfigurationCopy(c_tmp)

        c_tmp_2.getFrame(obj).setContact(1)
        c_tmp_2.computeCollisions()

        obj_name = objects[i]
        goal_name = goals[i]

        while True:
            r = random.choice(robots)
            r1 = compute_pick_and_place(c_tmp_2, r, obj, goal_name)

            if r1 is not None:
                break

        ee_name = r + "gripper"
        assigned_robots.append(r)
        start_pose = np.concatenate([c_tmp.getFrame(obj).getRelativePosition(), c_tmp.getFrame(obj).getRelativeQuaternion()])

        keyframes.append(
            (
                r, ee_name, obj_name, r1[:2], start_pose
            )
        )

        c_tmp.getFrame(obj).setRelativePosition(
            c_tmp.getFrame(goal_name).getRelativePosition()
        )
        c_tmp.getFrame(obj).setRelativeQuaternion(
            c_tmp.getFrame(goal_name).getRelativeQuaternion()
        )

        c_tmp.getFrame(obj).setContact(1)
        # c_tmp.view(False)

    # c_tmp.view(True)

    for i, obj in enumerate(objects):
        C.getFrame(obj).setPosition([0, 0, -2])

    return C, robots, keyframes

def make_strut_nccr_env():
    from scipy.spatial.transform import Rotation as R

    def compute_quaternion_world_to_goal(x_g, y_g):
        # Normalize input vectors
        x_g = x_g / np.linalg.norm(x_g)
        y_g = y_g / np.linalg.norm(y_g)

        # Compute the z-axis of the goal frame
        z_g = np.cross(x_g, y_g)
        z_g /= np.linalg.norm(z_g)

        # Re-orthogonalize y_g in case of numerical errors
        y_g = np.cross(z_g, x_g)

        # Construct the rotation matrix (columns are the goal frame's axes in the world frame)
        R_goal = np.column_stack((x_g, y_g, z_g))

        # Convert the rotation matrix to a quaternion
        quaternion = R.from_matrix(R_goal).as_quat()  # Returns [x, y, z, w]
        quaternion = np.array([quaternion[-1], quaternion[0], quaternion[1], quaternion[2]])
        return quaternion
    
    C = ry.Config()
    robot_path = os.path.join(
        os.path.dirname(__file__), "../models/abb_robot/dual_cell.g"
    )

    C.addFile(robot_path)

    assembly_path = "/home/valentin/git/postdoc/robotic-venv/nccr/exported_boxes.json"
        
    robots = ["a0_", "a1_"]
    
    objects = []
    goals = []

    asssigned_robots = []
    start_poses = {}

    with open(assembly_path) as f:
        d = json.load(f)

        # for i in range(len(d)):
        for i in range(10):
            obj = d[i]
            goal_name = "goal_" + str(obj["index"])
            obj_name = "obj_" + str(obj["index"])

            print(goal_name)

            shape_dict = obj["dimensions"]
            shape = np.array([shape_dict[key] for key in ["x", "y", "z"]])
            shape[0] = shape[0] * 0.7
            shape[2] = shape[2] * 0.7

            x_axis = obj["x_axis"]
            y_axis = obj["y_axis"]

            goal_pose = obj["origin"]
            goal_pose[2] += 0.025
            goal_quat = compute_quaternion_world_to_goal(x_axis, y_axis)

            goal_frame = (
                C.addFrame(goal_name)
                .setParent(C.getFrame("table"))
                .setRelativePosition(goal_pose)
                .setShape(ry.ST.box, size=shape)
                .setColor([1, 0.3, 0.3, 0.5])
                .setContact(0)
                .setRelativeQuaternion(goal_quat)
            )

            # C.addFrame(f"goal_marker_{str(obj['index'])}").setParent(goal_frame).setShape(
            #     ry.ST.marker, size=[0.1]
            # )

            if obj["robot_id"] == "A":
                start_pos = np.array([-0.4, 0.0, 0.15])
            else:
                start_pos = np.array([1.6, 0.0, 0.15])
            start_quat = compute_quaternion_world_to_goal(
                np.array([1, 0, 0]), np.array([0, 1, 0])
            )

            start_frame = (
                C.addFrame(obj_name)
                .setParent(C.getFrame("table"))
                .setRelativePosition(start_pos)
                .setShape(ry.ST.box, size=shape)
                .setColor([0.3, 1, 0.3])
                .setContact(0)
                .setRelativeQuaternion(start_quat)
                .setJoint(ry.JT.rigid)
            )
            # C.addFrame(f"start_marker_{str(obj['index'])}").setParent(start_frame).setShape(
            #     ry.ST.marker, size=[0.1]
            # )

            start_poses[obj_name] = {
                "position": start_pos,
                "orientation": start_quat
            }

            objects.append(obj_name)
            goals.append(goal_name)

            asssigned_robots.append("a0_" if obj["robot_id"] == "A" else "a1_")

    # C.view(True)

    def compute_rearrangement(c_tmp, robot_prefix, box, goal, gripper_type="vacuum"):
        # set everything but the current box to non-contact
        robot_base = robot_prefix + "base_link"
        c_tmp.selectJointsBySubtree(c_tmp.getFrame(robot_base))

        q_home = c_tmp.getJointState()

        komo = ry.KOMO(
            c_tmp, phases=2, slicesPerPhase=1, kOrder=1, enableCollisions=True
        )
        komo.addObjective(
            [], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e2], [-0.0]
        )

        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e1], [-0.0])

        ee_name = robot_prefix + "ee_marker"

        komo.addControlObjective([], 0, 1e-1)
        komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        komo.addModeSwitch([1, 2], ry.SY.stable, [ee_name, box])
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.distance,
        #     [robot_prefix + "ur_gripper_center", box],
        #     ry.OT.sos,
        #     [1e0],
        #     # [0.05],
        # )
        
        komo.addObjective(
            [1, 2],
            ry.FS.distance,
            [ee_name, box],
            ry.OT.sos,
            [1e1],
        )
        komo.addObjective(
            [1, 2],
            ry.FS.positionDiff,
            [ee_name, box],
            ry.OT.sos,
            [1e0],
        )

        komo.addObjective(
            [1, 2],
            ry.FS.positionDiff,
            [ee_name, box],
            ry.OT.ineq,
            [0, 1e1, 0],
            [10, shape[1]/2 * 0.8, 10]
        )

        komo.addObjective(
            [1, 2],
            ry.FS.positionDiff,
            [ee_name, box],
            ry.OT.ineq,
            [0, -1e1, 0],
            [10, -shape[1]/2 * 0.8, 10]
        )
        komo.addObjective(
            [1, 2],
            ry.FS.scalarProductXY,
            [ee_name, box],
            ry.OT.sos,
            [1e1],
        )

        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.positionDiff,
        #     [ee_name, box],
        #     ry.OT.sos,
        #     [1e1, 1e1, 1],
        # )
        
        # if gripper_type == "two_finger":
        #     komo.addObjective(
        #         [1, 2],
        #         ry.FS.scalarProductXZ,
        #         [ee_name, box],
        #         ry.OT.eq,
        #         [1e1],
        #         [1],
        #     )
        komo.addObjective(
            [1, 2],
            ry.FS.positionDiff,
            [ee_name, box],
            ry.OT.eq,
            [1e0],
            [0, 0, 0]
        )
        # else:
        #     komo.addObjective(
        #         [1, 2],
        #         ry.FS.scalarProductXY,
        #         [ee_name, box],
        #         ry.OT.sos,
        #         [1e1],
        #         [0],
        #     )
        #     komo.addObjective(
        #         [1, 2],
        #         ry.FS.positionDiff,
        #         [ee_name, box],
        #         ry.OT.eq,
        #         [1e0],
        #         [0.0, 0, 0],
        #     )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.scalarProductZZ,
        #     [robot_prefix + "ur_gripper", box],
        #     ry.OT.sos,
        #     [1e1],
        # )

        # komo.addModeSwitch([1, 2], ry.SY.stable, [robot_prefix + "ur_vacuum", box])
        
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.positionDiff,
        #     [robot_prefix + "ur_vacuum", box],
        #     ry.OT.sos,
        #     [1e1, 1e1, 1],
        # )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.scalarProductYZ,
        #     [robot_prefix + "ur_ee_marker", box],
        #     ry.OT.sos,
        #     [1e1],
        # )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.scalarProductZZ,
        #     [robot_prefix + "ur_ee_marker", box],
        #     ry.OT.sos,
        #     [1e1],
        # )

        # for pick and place directly
        # komo.addModeSwitch([2, -1], ry.SY.stable, ["table", box])
        komo.addObjective([2, -1], ry.FS.poseDiff, [goal, box], ry.OT.eq, [1e1])

        komo.addObjective(
            times=[0, 3],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.sos,
            scale=[1e-1],
            target=q_home,
        )

        komo.addObjective(
            times=[3, -1],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.eq,
            scale=[1e0],
            target=q_home,
        )

        max_attempts = 200
        for num_attempt in range(max_attempts):
            # komo.initRandom()
            if num_attempt > 0:
                dim = len(c_tmp.getJointState())
                x_init = np.random.rand(dim) * 3 - 1.5
                komo.initWithConstant(x_init)
                # komo.initWithPath(np.random.rand(3, 12) * 5 - 2.5)

            solver = ry.NLP_Solver(komo.nlp(), verbose=4)
            # options.nonStrictSteps = 50;

            # solver.setOptions(damping=0.01, wolfe=0.001)
            # solver.setOptions(damping=0.001)
            retval = solver.solve()
            retval = retval.dict()

            # print(retval)

            # if view:
            print(retval)

            # komo.view(True, "IK solution")

            # print(retval)

            if retval["ineq"] < 1 and retval["eq"] < 1 and retval["feasible"]:
                # komo.view(True, "IK solution")
                keyframes = komo.getPath()
                return keyframes

        return None

    keyframes = []

    c_tmp = ry.Config()
    c_tmp.addConfigurationCopy(C)

    for i, obj in enumerate(objects):
        c_tmp_2 = ry.Config()
        c_tmp_2.addConfigurationCopy(c_tmp)

        c_tmp_2.getFrame(obj).setContact(1)

        obj_name = objects[i]
        goal_name = goals[i]
        robot = asssigned_robots[i]

        c_tmp_2.computeCollisions()

        keyframe = compute_rearrangement(c_tmp_2, robot, obj_name, goal_name)

        if keyframe is None:
            raise ValueError

        ee_name = robot + "ee_marker"

        start_pose = np.concatenate([start_poses[obj]["position"], start_poses[obj]["orientation"]])

        keyframes.append(
            (
                robot, ee_name, obj_name, keyframe[:2], start_pose
            )
        )

        c_tmp.getFrame(obj).setRelativePosition(
            c_tmp.getFrame(goal_name).getRelativePosition()
        )
        c_tmp.getFrame(obj).setRelativeQuaternion(
            c_tmp.getFrame(goal_name).getRelativeQuaternion()
        )

        c_tmp.getFrame(obj).setContact(1)

    for i, obj in enumerate(objects):
        C.getFrame(obj).setPosition([0, 0, -2])

    return C, robots, keyframes


def coop_tamp_architecture_env(assembly_name, robot_type="ur10", gripper_type="two_finger"):
    C = ry.Config()
    
    C.addFrame("table").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(1)

    if assembly_name == "three_robot_truss":
        assembly_filename = "three_robot_truss"
    elif assembly_name == "spiral_tower":
        assembly_filename = "spiral_tower_four"
    elif assembly_name == "spiral_tower_two":
        assembly_filename = "spiral_tower_two"
    elif assembly_name == "cube_four":
        assembly_filename = "cube_four"
    elif assembly_name == "extreme_beam_test":
        assembly_filename = "extreme_beam_test"
    else:
        raise ValueError("Assembly name not existent.")
        # path = os.path.join(
        #     "/home/valentin/Downloads/assembly_roadmap-master/data/problems/spiral_tower_side_welding/problem.json"
        #     # "/home/valentin/Downloads/assembly_roadmap-master/data/problems/truss_bridge_three_robots_welding/problem.json"
        #     # "/home/valentin/Downloads/assembly_roadmap-master/data/problems/truss_bridge/problem.json"
        #     # "/home/valentin/Downloads/assembly_roadmap-master/data/problems/truss_boat_three_robots/problem.json"
        #     # "/home/valentin/Downloads/assembly_roadmap-master/data/problems/extreme_beam_test/problem.json"
        #     # "/home/valentin/Downloads/assembly_roadmap-master/data/problems/cube_four_robots/problem.json"
        # )
    
    path = os.path.join(
        os.path.dirname(__file__), f"../problems/desc/{assembly_filename}.json"
    )

    if gripper_type == "vacuum":
        robot_path = os.path.join(os.path.dirname(__file__), "../models/ur10/ur10_vacuum.g")
    else:
        robot_path = os.path.join(os.path.dirname(__file__), "../models/ur10/ur10_two_finger.g")

    start_poses = {}
    goal_poses = {}
    objects = []
    robots = []

    with open(path) as f:
        d = json.load(f)

        agents = d["agents"]
        for i, agent in enumerate(agents):
            relative_base_pos = np.array(agent["base_link_position"])
            relative_base_quat = np.array(agent["base_link_orientation"])
            relative_base_quat = np.array([relative_base_quat[3], relative_base_quat[0], relative_base_quat[1], relative_base_quat[2]])

            C.addFile(robot_path, namePrefix=f"a{i}_").setParent(
                C.getFrame("table")
            ).setRelativePosition(relative_base_pos).setRelativeQuaternion(
                relative_base_quat
            ).setJoint(ry.JT.rigid)

            robots.append(f"a{i}_ur_")

        components = d["components"]
        for i, component in enumerate(components):
            start_position = np.array(component["start_position"])
            start_orientation = np.array(component["start_orientation"])
            start_orientation = np.array([start_orientation[3], start_orientation[0], start_orientation[1], start_orientation[2]])

            goal_position = np.array(component["goal_position"])
            goal_orientation = np.array(component["goal_orientation"])
            goal_orientation = np.array([goal_orientation[3], goal_orientation[0], goal_orientation[1], goal_orientation[2]])

            p1 = np.array(component["geometry_data"]["axis_points"][0])
            p2 = np.array(component["geometry_data"]["axis_points"][1])

            shrink = component["geometry_data"]["shrink"]

            length = np.linalg.norm(p2 - p1)
            shrinked_length = np.linalg.norm(p2 - p1) - 2  * shrink

            obj_name = "start_" + str(i)
            start_poses[obj_name] = {
                "position": start_position,
                "orientation": start_orientation
            }

            goal_poses[obj_name] = {
                "position": goal_position,
                "orientation": goal_orientation
            }

            r = component["geometry_data"]["radius"]

            C.addFrame(obj_name).setParent(
                C.getFrame("table")
            ).setPosition(start_position).setShape(
                ry.ST.cylinder, size=[r, shrinked_length, 0.005]
            ).setColor([1, 0.3, 0.3, 1]).setContact(0).setQuaternion(start_orientation).setJoint(ry.JT.rigid)

            C.addFrame("goal_" + str(i)).setParent(
                C.getFrame("table")
            ).setPosition(goal_position).setShape(
                ry.ST.cylinder, size=[r, length, 0.005]
            ).setColor([0.3, 0.3, 0.3, 0.2]).setContact(0).setQuaternion(goal_orientation)

            objects.append(obj_name)

            # C.view(True)
    
    def compute_rearrangment(c_tmp, robot_prefix, box, goal, gripper_type="vacuum"):
        # set everything but the current box to non-contact
        robot_base = robot_prefix + "base"
        c_tmp.selectJointsBySubtree(c_tmp.getFrame(robot_base))

        q_home = c_tmp.getJointState()

        komo = ry.KOMO(
            c_tmp, phases=2, slicesPerPhase=1, kOrder=1, enableCollisions=True
        )
        komo.addObjective(
            [], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [-0.0]
        )

        if gripper_type == "vacuum":
            ee_name = robot_prefix + "vacuum"
        else:
            ee_name = robot_prefix + "gripper_center"

        komo.addControlObjective([], 0, 1e-1)
        # komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        komo.addModeSwitch([1, 2], ry.SY.stable, [ee_name, box])
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.distance,
        #     [robot_prefix + "ur_gripper_center", box],
        #     ry.OT.sos,
        #     [1e0],
        #     # [0.05],
        # )
        komo.addObjective(
            [1, 2],
            ry.FS.positionDiff,
            [ee_name, box],
            ry.OT.sos,
            [1e1, 1e1, 1],
        )
        
        if gripper_type == "two_finger":
            komo.addObjective(
                [1, 2],
                ry.FS.scalarProductXZ,
                [ee_name, box],
                ry.OT.eq,
                [1e1],
                [1],
            )
            komo.addObjective(
                [1, 2],
                ry.FS.positionDiff,
                [ee_name, box],
                ry.OT.eq,
                [1e0],
                [0, 0, 0]
            )
        else:
            komo.addObjective(
                [1, 2],
                ry.FS.scalarProductXZ,
                [ee_name, box],
                ry.OT.sos,
                [1e1],
                [0],
            )
            komo.addObjective(
                [1, 2],
                ry.FS.distance,
                [ee_name, box],
                ry.OT.eq,
                [1e0],
                [0.05],
            )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.scalarProductZZ,
        #     [robot_prefix + "ur_gripper", box],
        #     ry.OT.sos,
        #     [1e1],
        # )

        # komo.addModeSwitch([1, 2], ry.SY.stable, [robot_prefix + "ur_vacuum", box])
        
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.positionDiff,
        #     [robot_prefix + "ur_vacuum", box],
        #     ry.OT.sos,
        #     [1e1, 1e1, 1],
        # )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.scalarProductYZ,
        #     [robot_prefix + "ur_ee_marker", box],
        #     ry.OT.sos,
        #     [1e1],
        # )
        # komo.addObjective(
        #     [1, 2],
        #     ry.FS.scalarProductZZ,
        #     [robot_prefix + "ur_ee_marker", box],
        #     ry.OT.sos,
        #     [1e1],
        # )

        # for pick and place directly
        # komo.addModeSwitch([2, -1], ry.SY.stable, ["table", box])
        komo.addObjective([2, -1], ry.FS.poseDiff, [goal, box], ry.OT.eq, [1e1])

        komo.addObjective(
            times=[0, 3],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.sos,
            scale=[1e0],
            target=q_home,
        )

        komo.addObjective(
            times=[3, -1],
            feature=ry.FS.jointState,
            frames=[],
            type=ry.OT.eq,
            scale=[1e0],
            target=q_home,
        )

        max_attempts = 5
        for num_attempt in range(max_attempts):
            # komo.initRandom()
            if num_attempt > 0:
                dim = len(c_tmp.getJointState())
                x_init = np.random.rand(dim) * 3 - 1.5
                komo.initWithConstant(x_init)
                # komo.initWithPath(np.random.rand(3, 12) * 5 - 2.5)

            solver = ry.NLP_Solver(komo.nlp(), verbose=4)
            # options.nonStrictSteps = 50;

            # solver.setOptions(damping=0.01, wolfe=0.001)
            # solver.setOptions(damping=0.001)
            retval = solver.solve()
            retval = retval.dict()

            # print(retval)

            # if view:
            # komo.view(True, "IK solution")

            print(retval)

            if retval["ineq"] < 1 and retval["eq"] < 1 and retval["feasible"]:
                # komo.view(True, "IK solution")
                keyframes = komo.getPath()
                return keyframes

        return None

    # C.view(True)
    
    # sort objects by z component of their goal position
    sorted_objects = sorted(objects, key=lambda obj: goal_poses[obj]["position"][2]) 
    objects = sorted_objects

    direct_pick_place_keyframes = {}

    for r in robots:
        direct_pick_place_keyframes[r] = {}

    c_tmp = ry.Config()
    c_tmp.addConfigurationCopy(C)

    robot_to_use = []

    for obj in objects:
        c_tmp_2 = ry.Config()
        c_tmp_2.addConfigurationCopy(c_tmp)

        c_tmp_2.getFrame(obj).setContact(1)

        # c_tmp_2.computeCollisions()

        goal = "goal" + obj[5:]

        while True:
            r = random.choice(robots)
            r1 = compute_rearrangment(c_tmp_2, r, obj, goal, gripper_type=gripper_type)

            if r1 is not None:
                break

        direct_pick_place_keyframes[r][obj] = r1[:2]
        robot_to_use.append(r)

        c_tmp.getFrame(obj).setRelativePosition(
            c_tmp.getFrame(goal).getRelativePosition()
        )
        c_tmp.getFrame(obj).setRelativeQuaternion(
            c_tmp.getFrame(goal).getRelativeQuaternion()
        )

        c_tmp.getFrame(obj).setContact(1)

    # set poses to somewhere hidden and collect couple of info things
    keyframes = []

    for i, obj in enumerate(objects):
        C.getFrame(obj).setPosition([0, 0, -2])
        start_pose = np.concatenate([start_poses[obj]["position"], start_poses[obj]["orientation"]])

        ee_name = robot_to_use[i] + "gripper_center"
        if gripper_type == "vacuum":
            ee_name = robot_to_use[i] + "vacuum"

        keyframes.append(
            (
                robot_to_use[i], ee_name, obj, direct_pick_place_keyframes[robot_to_use[i]][obj], start_pose
            )
        )

    return C, robots, keyframes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Env shower")
    parser.add_argument("env", nargs="?", default="default", help="env to show")

    args = parser.parse_args()

    if args.env == "piano":
        make_piano_mover_env(view=True)
    elif args.env == "simple_2d":
        make_2d_rai_env(True)
    elif args.env == "three_agents":
        make_2d_rai_env_3_agents(True)
    elif args.env == "box_sorting":
        make_box_sorting_env(True)
    elif args.env == "eggs":
        make_egg_carton_env(True)
    elif args.env == "triple_waypoints":
        make_panda_waypoint_env(view=True)
    elif args.env == "welding":
        make_welding_env(num_robots=4, num_pts=8, view=True)
    elif args.env == "mobile":
        make_mobile_manip_env(True)
    elif args.env == "bottle":
        make_bottle_insertion(view=True)
    elif args.env == "handover":
        make_handover_env(view=True)
    elif args.env == "2d_handover":
        make_two_dim_handover(view=True)
    elif args.env == "random_2d":
        make_random_two_dim(view=True)
    elif args.env == "optimality_test_2d":
        make_two_dim_tunnel_env(view=True)
    elif args.env == "box_rearrangement":
        make_box_rearrangement_env(view=True)
    elif args.env == "depalletizing":
        make_depalletizing_env()
    elif args.env == "struts":
        make_strut_assembly_problem()
    elif args.env == "crl":
        make_crl_logo_rearrangement_env()
    else:
        make_panda_waypoint_env(2, view=True)
