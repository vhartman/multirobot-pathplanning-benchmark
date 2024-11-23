import robotic as ry
import numpy as np
import argparse

import os.path
import random

# make everything predictable
np.random.seed(2)
random.seed(2)

def make_2d_rai_env(view: bool = False):
    C = ry.Config()

    table = (
        C.addFrame("table")
        .setPosition([0, 0, 1.0])
        .setShape(ry.ST.box, size=[2, 2, 0.06, 0.005])
        .setColor([0.3, 0.3, 0.3])
        .setContact(1)
    )

    pre_agent_1_frame = (
        C.addFrame("pre_agent_1_frame")
        .setParent(table)
        .setPosition(table.getPosition() + [0.0, -0.5, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    C.addFrame("a1").setParent(pre_agent_1_frame).setShape(
        ry.ST.cylinder, size=[0.1, 0.2, 0.06, 0.15]
    ).setColor([1, 0.5, 0]).setContact(1).setJoint(ry.JT.transXYPhi, limits=np.array([-2, 2, -2, 2, -3.14, 3.14]))

    pre_agent_2_frame = (
        C.addFrame("pre_agent_2_frame")
        .setParent(table)
        .setPosition(table.getPosition() + [0, 0.5, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    C.addFrame("a2").setParent(pre_agent_2_frame).setShape(
        ry.ST.box,
        size=[0.1, 0.2, 0.06, 0.005],
        # ry.ST.cylinder, size=[4, 0.1, 0.06, 0.075]
    ).setColor([0.5, 0.5, 0]).setContact(1).setJoint(ry.JT.transXYPhi, limits=np.array([-2, 2, -2, 2, -3.14, 3.14]))

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

    C.addFrame("wall1").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0, 1.1, 0.07]
    ).setShape(ry.ST.box, size=[2, 0.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("wall2").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0, -1.1, 0.07]
    ).setShape(ry.ST.box, size=[2, 0.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("wall3").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [1.1, 0, 0.07]
    ).setShape(ry.ST.box, size=[0.2, 2.4, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("wall4").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [-1.1, 0, 0.07]
    ).setShape(ry.ST.box, size=[0.2, 2.4, 0.06, 0.005]).setContact(1).setColor(
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
        [3, -1], ry.FS.poseDiff, ["a2", "pre_agent_2_frame"], ry.OT.eq, [1e1]
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

def make_two_dim_handover(view:bool=False):
    C = ry.Config()

    table = (
        C.addFrame("table")
        .setPosition([0, 0, 1.0])
        .setShape(ry.ST.box, size=[4, 4, 0.06, 0.005])
        .setColor([0.3, 0.3, 0.3])
        .setContact(1)
    )

    pre_agent_1_frame = (
        C.addFrame("pre_agent_1_frame")
        .setParent(table)
        .setPosition(table.getPosition() + [-.5, 0.8, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    C.addFrame("a1").setParent(pre_agent_1_frame).setShape(
        ry.ST.cylinder, size=[4, 0.1, 0.06, 0.2]
    ).setColor([0.5, 0.5, 0]).setContact(1).setJoint(ry.JT.transXYPhi, limits=np.array([-3, 3, -3, 3, -3.14, 3.14]))

    pre_agent_2_frame = (
        C.addFrame("pre_agent_2_frame")
        .setParent(table)
        .setPosition(table.getPosition() + [0.0, -0.5, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    C.addFrame("a2").setParent(pre_agent_2_frame).setShape(
        ry.ST.cylinder, size=[0.1, 0.2, 0.06, 0.2]
    ).setColor([1, 0.5, 0]).setContact(1).setJoint(ry.JT.transXYPhi, limits=np.array([-3, 3, -3, 3, -3.14, 3.14]))

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
        ry.ST.box, size=[0.2, 0.2, 0.06, 0.005]
    ).setColor([1, 0.5, 0, 0.3]).setContact(0).setRelativePosition([.8, 0.4, 0.07])

    C.addFrame("goal2").setParent(table).setShape(
        ry.ST.box, size=[0.2, 0.2, 0.06, 0.005]
    ).setColor([0.5, 0.5, 0, 0.2]).setContact(0).setRelativePosition([.8, 1.3, 0.07])

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
        C.getFrame("table").getPosition() + [-0.4, -.6, 0.07]
    ).setShape(ry.ST.box, size=[0.2, .9, 0.06, 0.005]).setContact(-2).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("obs4").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0.8, 0.8, 0.07]
    ).setShape(ry.ST.box, size=[0.6, .2, 0.06, 0.005]).setContact(-2).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("wall1").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0, 2.1, 0.07]
    ).setShape(ry.ST.box, size=[4, 0.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("wall2").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0, -2.1, 0.07]
    ).setShape(ry.ST.box, size=[4, 0.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("wall3").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [2.1, 0, 0.07]
    ).setShape(ry.ST.box, size=[0.2, 4.4, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("wall4").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [-2.1, 0, 0.07]
    ).setShape(ry.ST.box, size=[0.2, 4.4, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    if view:
        C.view(True)

    def compute_handover():
        box = "obj1"

        q_home= C.getJointState()

        komo = ry.KOMO(C, phases=4, slicesPerPhase=1, kOrder=1, enableCollisions=True)
        komo.addObjective(
            [], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1], [-0.0]
        )

        komo.addControlObjective([], 0, 1e-1)
        # komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        komo.addModeSwitch([1, 2], ry.SY.stable, ["a1", box])
        komo.addObjective(
            [1, 2], ry.FS.distance, ["a1", box], ry.OT.sos, [1e1], [-0.0]
        )
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
            [2], ry.FS.position, [box], ry.OT.sos, [1e0, 1e0, 0], [0.8, -0.8, 0]
        )

        komo.addModeSwitch([2, 3], ry.SY.stable, ["a2", box])
        komo.addObjective(
            [2, 3], ry.FS.distance, ["a2", box], ry.OT.sos, [1e1], [-0.0]

        )
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

        for _ in range(30):
            # komo.initRandom()
            komo.initWithConstant(np.random.rand(6)*2)

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

            if retval['ineq'] < 1 and retval['eq'] < 1 and retval['feasible']:
                return keyframes

    def compute_place():
        box = "obj2"

        q_home= C.getJointState()

        komo = ry.KOMO(C, phases=3, slicesPerPhase=1, kOrder=1, enableCollisions=True)
        komo.addObjective(
            [], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1], [-0.0]
        )

        komo.addControlObjective([], 0, 1e-1)
        # komo.addControlObjective([], 1, 1e-1)
        # komo.addControlObjective([], 2, 1e-1)

        komo.addModeSwitch([1, 2], ry.SY.stable, ["a1", box])
        komo.addObjective(
            [1, 2], ry.FS.distance, ["a1", box], ry.OT.sos, [1e1], [-0.0]
        )
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

        for _ in range(30):
            # komo.initRandom()
            komo.initWithConstant(np.random.rand(6)*2)

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

            if retval['ineq'] < 1 and retval['eq'] < 1 and retval['feasible']:
                return keyframes

    handover_keyframes = compute_handover()
    place_keyframes = compute_place()

    return C, np.concatenate([handover_keyframes, place_keyframes])

def make_piano_mover_env(view: bool = False):
    C = ry.Config()

    table = (
        C.addFrame("table")
        .setPosition([0, 0, 1.0])
        .setShape(ry.ST.box, size=[2, 2, 0.06, 0.005])
        .setColor([0.3, 0.3, 0.3])
        .setContact(1)
    )

    pre_agent_1_frame = (
        C.addFrame("pre_agent_1_frame")
        .setParent(table)
        .setPosition(table.getPosition() + [0.0, -0.5, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    C.addFrame("a1").setParent(pre_agent_1_frame).setShape(
        ry.ST.cylinder, size=[0.1, 0.2, 0.06, 0.075]
    ).setColor([1, 0.5, 0]).setContact(1).setJoint(ry.JT.transXYPhi, limits=np.array([-2, 2, -2, 2, -3.14, 3.14]))

    pre_agent_2_frame = (
        C.addFrame("pre_agent_2_frame")
        .setParent(table)
        .setPosition(table.getPosition() + [0, 0.5, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    C.addFrame("a2").setParent(pre_agent_2_frame).setShape(
        ry.ST.cylinder, size=[4, 0.1, 0.06, 0.075]
    ).setColor([0.5, 0.5, 0]).setContact(1).setJoint(ry.JT.transXYPhi, limits=np.array([-2, 2, -2, 2, -3.14, 3.14]))

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

    C.addFrame("wall1").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0, 1.1, 0.07]
    ).setShape(ry.ST.box, size=[2, 0.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("wall2").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0, -1.1, 0.07]
    ).setShape(ry.ST.box, size=[2, 0.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("wall3").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [1.1, 0, 0.07]
    ).setShape(ry.ST.box, size=[0.2, 2.4, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("wall4").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [-1.1, 0, 0.07]
    ).setShape(ry.ST.box, size=[0.2, 2.4, 0.06, 0.005]).setContact(1).setColor(
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


def make_2d_rai_env_3_agents(view: bool = False):
    C = ry.Config()

    table = (
        C.addFrame("table")
        .setPosition([0, 0, 1.0])
        .setShape(ry.ST.box, size=[2, 2, 0.06, 0.005])
        .setColor([0.3, 0.3, 0.3])
        .setContact(1)
    )

    pre_agent_1_frame = (
        C.addFrame("pre_agent_1_frame")
        .setParent(table)
        .setPosition(table.getPosition() + [0.0, -0.5, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    C.addFrame("a1").setParent(pre_agent_1_frame).setShape(
        ry.ST.cylinder, size=[0.1, 0.2, 0.06, 0.15]
    ).setColor([1, 0.5, 0]).setContact(1).setJoint(ry.JT.transXYPhi,limits=np.array([-2, 2, -2, 2, -3.14, 3.14]))

    pre_agent_2_frame = (
        C.addFrame("pre_agent_2_frame")
        .setParent(table)
        .setPosition(table.getPosition() + [0, 0.4, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    C.addFrame("a2").setParent(pre_agent_2_frame).setShape(
        ry.ST.box,
        size=[0.1, 0.2, 0.06, 0.005],
        # ry.ST.cylinder, size=[4, 0.1, 0.06, 0.075]
    ).setColor([0.5, 0.5, 0]).setContact(1).setJoint(ry.JT.transXYPhi, limits=np.array([-2, 2, -2, 2, -3.14, 3.14]))

    pre_agent_3_frame = (
        C.addFrame("pre_agent_3_frame")
        .setParent(table)
        .setPosition(table.getPosition() + [0.5, -0.7, 0.07])
        .setShape(ry.ST.marker, size=[0.05])
        .setColor([1, 0.5, 0])
        .setContact(0)
        .setJoint(ry.JT.rigid)
    )

    C.addFrame("a3").setParent(pre_agent_3_frame).setShape(
        ry.ST.box,
        size=[0.3, 0.2, 0.06, 0.005],
        # ry.ST.cylinder, size=[4, 0.1, 0.06, 0.075]
    ).setColor([0.5, 0.5, 1]).setContact(1).setJoint(ry.JT.transXYPhi, limits=np.array([-2, 2, -2, 2, -3.14, 3.14]))

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

    C.addFrame("wall1").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0, 1.1, 0.07]
    ).setShape(ry.ST.box, size=[2, 0.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("wall2").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [0, -1.1, 0.07]
    ).setShape(ry.ST.box, size=[2, 0.2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("wall3").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [1.1, 0, 0.07]
    ).setShape(ry.ST.box, size=[0.2, 2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    C.addFrame("wall4").setParent(table).setPosition(
        C.getFrame("table").getPosition() + [-1.1, 0, 0.07]
    ).setShape(ry.ST.box, size=[0.2, 2, 0.06, 0.005]).setContact(1).setColor(
        [0, 0, 0]
    ).setJoint(ry.JT.rigid)

    if view:
        C.view(True)

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
        [6], ry.FS.positionDiff, ["a1", "pre_agent_1_frame"], ry.OT.eq, [1e1]
    )
    komo.addObjective(
        [6], ry.FS.positionDiff, ["a2", "pre_agent_2_frame"], ry.OT.eq, [1e1]
    )
    komo.addObjective(
        [6], ry.FS.positionDiff, ["a3", "pre_agent_3_frame"], ry.OT.eq, [1e1]
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

    table = (
        C.addFrame("table")
        .setPosition([0, 0, 0.2])
        .setShape(ry.ST.box, size=[2, 3, 0.06, 0.005])
        .setColor([0.3, 0.3, 0.3])
        .setContact(1)
    )

    # C.addFile(ry.raiPath('panda/panda.g'), namePrefix='a1_') \
    #         .setParent(C.getFrame('table')) \
    #         .setRelativePosition([-0.3, 0.5, 0]) \
    #         .setRelativeQuaternion([0.7071, 0, 0, -0.7071]) \
    robot_path = os.path.join(os.path.dirname(__file__), "ur10/ur10_vacuum.g")

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
                C.addFrame("box" + str(i) + str(j) + str(k)).setParent(table).setShape(
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

    box = "box100"
    robot_prefix = "a1_"

    komo.addModeSwitch([1, 2], ry.SY.stable, [robot_prefix + "ur_vacuum", box])
    komo.addObjective(
        [1, 2], ry.FS.distance, [robot_prefix + "ur_vacuum", box], ry.OT.sos, [1e1],
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

    box = "box101"
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


def make_egg_carton_env(view: bool = False):
    C = ry.Config()

    table = (
        C.addFrame("table")
        .setPosition([0, 0, 0.2])
        .setShape(ry.ST.box, size=[2, 3, 0.06, 0.005])
        .setColor([0.3, 0.3, 0.3])
        .setContact(1)
    )

    robot_path = os.path.join(os.path.dirname(__file__), "ur10/ur10_vacuum.g")

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
                C.addFrame("box" + str(i) + str(j) + str(k)).setParent(table).setShape(
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
            [1, 2], ry.FS.distance, [robot_prefix + "ur_vacuum", box], ry.OT.sos, [1e0], [0.05]

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

    a1_boxes = ["box000", "box001", "box011", "box002"]
    a2_boxes = ["box021", "box010", "box012", "box022", "box020"]

    keyframes = np.zeros((0, 12))

    for b in a1_boxes:
        keyframes = np.concatenate([keyframes, compute_keyframes_for_obj("a1_", b)])

    for b in a2_boxes:
        keyframes = np.concatenate([keyframes, compute_keyframes_for_obj("a2_", b)])

    # print(keyframes)

    # keyframes = np.concatenate([keyframes_a1, keyframes_a2])
    return C, keyframes

def make_handover_env(view:bool=False):
    C = ry.Config()

    table = (
        C.addFrame("table")
        .setPosition([0, 0, 0.2])
        .setShape(ry.ST.box, size=[2, 3, 0.06, 0.005])
        .setColor([0.3, 0.3, 0.3])
        .setContact(1)
    )

    # C.addFile(ry.raiPath('panda/panda.g'), namePrefix='a1_') \
    #         .setParent(C.getFrame('table')) \
    #         .setRelativePosition([-0.3, 0.5, 0]) \
    #         .setRelativeQuaternion([0.7071, 0, 0, -0.7071]) \
    robot_path = os.path.join(os.path.dirname(__file__), "ur10/ur10_vacuum.g")

    C.addFile(robot_path, namePrefix="a1_").setParent(
        table
    ).setRelativePosition([-0.5, -0.5, 0.0]).setRelativeQuaternion(
        [0.7071, 0, 0, 0.7071]
    ).setJoint(ry.JT.rigid)

    # C.getFrame('a1_ur_coll0').setContact(-2)

    C.addFile(robot_path, namePrefix="a2_").setParent(
        table
    ).setRelativePosition([+0.5, 0.5, 0.0]).setRelativeQuaternion(
        [0.7071, 0, 0, -0.7071]
    ).setJoint(ry.JT.rigid)

    C.addFrame("obj1").setParent(table).setShape(
        ry.ST.box, size=[0.4, 0.4, 0.2, 0.005]
    ).setColor([1, 0.5, 0, 1]).setContact(1).setRelativePosition(
        [+0.5, -1., 0.15]
    ).setJoint(ry.JT.rigid)

    C.addFrame("goal1").setParent(table).setShape(
        ry.ST.box, size=[0.4, 0.4, 0.2, 0.005]
    ).setColor([1, 0.5, 0, 0.2]).setContact(0).setRelativePosition(
        [-0.5, 1., 0.15]
    ).setJoint(ry.JT.rigid)

    C.addFrame("obs1").setParent(table).setRelativePosition(
        [-0.5, 1, 0.7]
    ).setShape(ry.ST.box, size=[1, 1, 0.1, 0.005]).setColor(
        [0.3, 0.3, 0.3]
    ).setContact(1)

    C.addFrame("obs2").setParent(table).setRelativePosition(
        [0.5, -1, 0.7]
    ).setShape(ry.ST.box, size=[1, 1, 0.1, 0.005]).setColor(
        [0.3, 0.3, 0.3]
    ).setContact(1)

    if view:
        C.view(True)

    qHome = C.getJointState()
    box = "obj1"

    komo = ry.KOMO(C, phases=4, slicesPerPhase=1, kOrder=1, enableCollisions=True)
    komo.addObjective(
        [], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [-0.01]
    )

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


# TODO: the parameters are implemented horribly
def make_panda_waypoint_env(
    num_robots: int = 3, num_waypoints: int = 6, view: bool = False
):
    if num_robots > 3:
        raise NotImplementedError("More than three robot arms are not supported.")

    if num_waypoints > 6:
        raise NotImplementedError("More than six waypoints are not supported.")

    C = ry.Config()
    # C.addFile(ry.raiPath('scenarios/pandaSingle.g'))

    C.addFrame("table").setPosition([0, 0, 0.5]).setShape(
        ry.ST.box, size=[2, 2, 0.06, 0.005]
    ).setColor([0.3, 0.3, 0.3]).setContact(1)

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
        keyframes_a1 = compute_pose_for_robot("a1_gripper")
        keyframes = np.concatenate([keyframes, keyframes_a1])
    if num_robots > 2:
        keyframes_a2 = compute_pose_for_robot("a2_gripper")
        keyframes = np.concatenate([keyframes, keyframes_a2])

    return C, keyframes


def quaternion_from_z_rotation(angle):
    half_angle = angle / 2
    w = np.cos(half_angle)
    x = 0
    y = 0
    z = np.sin(half_angle)
    return np.array([w, x, y, z])


def make_welding_env(num_robots=4, num_pts=4, view: bool = False):
    C = ry.Config()

    robot_path = os.path.join(os.path.dirname(__file__), "ur10/ur_welding.g")

    C.addFrame("table").setPosition([0, 0, 0.5]).setShape(
        ry.ST.box, size=[2, 2, 0.06, 0.005]
    ).setColor([0.3, 0.3, 0.3]).setContact(1)

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
                [-0.05]
            )

            komo.addObjective(
                [i + 1],
                ry.FS.positionDiff,
                [robot_ee, "obs" + str(i + 1)],
                ry.OT.sos,
                [1e-1],
            )

            komo.addObjective([i+1], ry.FS.scalarProductYZ, [robot_ee, "obs"+str(i+1)], ry.OT.sos, [1e-1])


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


def make_bottle_insertion(remove_non_moved_bottles:bool=False, view: bool = False):
    C = ry.Config()

    path = os.path.join(os.path.dirname(__file__), "bottle.g")
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
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [0.01])
        komo.addControlObjective([], 0, 1e-1)
        # komo.addControlObjective([], 1, 1e0)
        # komo.addControlObjective([], 2, 1e0)

        komo.addModeSwitch([1, 2], ry.SY.stable, [ee, bottle])
        komo.addObjective([1, 2], ry.FS.distance, [ee, bottle], ry.OT.eq, [1e1], [-0.01])

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

            if retval['ineq'] < 1 and retval['eq'] < 1 and retval['feasible']:
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


def make_mobile_manip_env(view: bool = False):
    C = ry.Config()

    C.addFile("mobile-manipulator-restricted.g", namePrefix="a1_").setPosition(
        [1, 0, 0.2]
    )
    C.addFile("mobile-manipulator-restricted.g", namePrefix="a1_").setPosition(
        [-1, 0, 0.2]
    )

    C.view(True)


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
    else:
        make_panda_waypoint_env(2, view=True)
