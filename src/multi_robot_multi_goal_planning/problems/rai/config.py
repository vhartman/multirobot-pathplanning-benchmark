"""Reusable RAI joint and keyframe utilities."""
import robotic as ry
import numpy as np
from typing import List

def get_robot_joints(C: ry.Config, prefix: str) -> List[str]:
    links = []

    for name in C.getJointNames():
        if prefix in name:
            name = name.split(":")[0]

            if name not in links:
                links.append(name)

    return links


def solve_komo_problem(komo, max_attempts, C, view, mult=3, offset=-1.5, damping=None, wolfe=None):
    for num_attempt in range(max_attempts):
        # komo.initRandom()
        if num_attempt > 0:
            dim = len(C.getJointState())
            x_init = np.random.rand(dim) * mult + offset
            komo.initWithConstant(x_init)
            # komo.initWithPath(np.random.rand(3, 12) * 5 - 2.5)

        solver = ry.NLP_Solver(komo.nlp(), verbose=4)
        # options.nonStrictSteps = 50;

        # solver.setOptions(damping=0.01, wolfe=0.001)
        if damping is not None:
            solver.setOptions(damping=damping)
        if wolfe is not None:
            solver.setOptions(wolfe=wolfe)

        retval = solver.solve()
        retval = retval.dict()

        # print(retval)

        if view:
            print(retval)
            komo.view(True, "IK solution")

        # komo.view(True, "IK solution")

        if retval["ineq"] < 1 and retval["eq"] < 1 and retval["feasible"]:
            # komo.view(True, "IK solution")
            keyframes = komo.getPath()
            return keyframes

    return None
