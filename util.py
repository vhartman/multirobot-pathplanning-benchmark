from rai_envs import *

from typing import List

def display_path(
    env, path: List[State], stop: bool = True, export: bool = False
) -> None:
    for i in range(len(path)):
        env.set_to_mode(path[i].mode)
        for k in range(len(env.robots)):
            q = path[i].q[k]
            env.C.setJointState(q, get_robot_joints(env.C, env.robots[k]))

        env.C.view(stop)

        if export:
            env.C.view_savePng("./z.vid/")

        time.sleep(0.05)