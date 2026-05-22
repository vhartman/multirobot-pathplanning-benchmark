from multi_robot_multi_goal_planning.problems import get_env_by_name
from multi_robot_multi_goal_planning.problems.skills import EEPositionGoalReaching, EEPoseGoalReaching, EndEffectorPositionFollowing, DualRobotGrasping


import numpy as np
import random

def base_skill():
    np.random.seed(0)
    random.seed(0)
    
    env = get_env_by_name("rai.single_stick_upright")

    # ee_reaching_skill = EEPositionGoalReaching(np.array([0, 0, 0]), "a1_stick_ee")
    ee_reaching_skill = EEPoseGoalReaching(np.array([0, 0, 0, 0, 1, 0, 0]), "a1_stick_ee")
    
    for _ in range(100):
      qnew = ee_reaching_skill.step(env.C.getJointState(), env)

      env.C.setJointState(qnew)
      env.C.view(True)

def timed_skill():
    np.random.seed(0)
    random.seed(0)
    
    env = get_env_by_name("rai.single_stick_upright")

    curr_ee_pos = env.C.getFrame("a1_stick_ee").getPosition()
    ee_reaching_skill = EndEffectorPositionFollowing(curr_ee_pos, curr_ee_pos + np.array([0, 0, -0.5]), "a1_stick_ee")
    
    N = 100
    for i in range(N):
      qnew = ee_reaching_skill.step(i / N, env.C.getJointState(), env)

      env.C.setJointState(qnew)
      env.C.view(True)


def dual_arm_skill():
    np.random.seed(0)
    random.seed(0)
    
    # env = get_env_by_name("rai.dual_arm_transport")
    env = get_env_by_name("rai.dual_arm_transport_rotation")
    env.C.setJointState(env.pick_pose)

    skill = env.tasks[1].skill

    N = 100
    for i in range(N):
      qnew = skill.step(i / N, env.C.getJointState(), env)

      env.C.setJointState(qnew)
      env.C.view(True)


if __name__ == "__main__":
    # base_skill()
    # timed_skill()
    dual_arm_skill()