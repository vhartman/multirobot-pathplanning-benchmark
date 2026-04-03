from multi_robot_multi_goal_planning.problems import (
    get_env_by_name,
    get_all_environments,
)
import multi_robot_multi_goal_planning.problems as problems
from multi_robot_multi_goal_planning.problems.rai_envs import rai_env
from multi_robot_multi_goal_planning.problems.planning_env import Mode, State
from multi_robot_multi_goal_planning.problems.configuration import NpConfiguration

import numpy as np
import argparse
import time
import random


def benchmark_collision_checking(envs, N=10000):
    conf_type = type(envs[0].get_start_pos())

    def sample_next_modes(env, mode: Mode):
        if env.is_terminal_mode(mode):
            return None
        failed_attemps = 0
        while True:
            if failed_attemps > 1000:
                print("Failed to sample next mode")
                return None
            possible_next_task_combinations = env.get_valid_next_task_combinations(mode)
            if len(possible_next_task_combinations) > 0:
                ind = random.randint(0, len(possible_next_task_combinations) - 1)
                active_task = env.get_active_task(
                    mode, possible_next_task_combinations[ind]
                )
            else:
                active_task = env.get_active_task(mode, None)

            goals_to_sample = active_task.robots

            goal_sample = active_task.goal.sample(mode)

            q = []
            for i in range(len(env.robots)):
                r = env.robots[i]
                if r in goals_to_sample:
                    offset = 0
                    for _, task_robot in enumerate(active_task.robots):
                        if task_robot == r:
                            q.append(
                                goal_sample[
                                    offset : offset + env.robot_dims[task_robot]
                                ]
                            )
                            break
                        offset += env.robot_dims[task_robot]
                else:  # uniform sample
                    lims = env.limits[:, env.robot_idx[r]]
                    if lims[0, 0] < lims[1, 0]:
                        qr = (
                            np.random.rand(env.robot_dims[r])
                            * (lims[1, :] - lims[0, :])
                            + lims[0, :]
                        )
                    else:
                        qr = np.random.rand(env.robot_dims[r]) * 6 - 3

                    q.append(qr)

            q = conf_type.from_list(q)

            if env.is_collision_free(q, mode):                
                next_modes = env.get_next_modes(q, mode)
                # assert len(next_modes) == 1
                # next_mode = next_modes[0]
                return next_modes
            else:
                failed_attemps += 1

    modes_for_envs = []

    for env in envs:
        # create list of modes that we can reach
        print("Make mode list")
        reachable_modes = set([env.get_start_mode()])
        max_iter = 500
        for _ in range(max_iter):
            m_rnd = random.choice(tuple(reachable_modes))
            next_modes = sample_next_modes(env, m_rnd)
            print(next_modes)
            if next_modes is not None:
                reachable_modes.update(next_modes)
        reachable_modes = tuple(reachable_modes)
        print("Found", len(reachable_modes), "reachable modes")

        modes_for_envs.append(reachable_modes)
    
    is_collision_free_rai = envs[0].is_collision_free
    is_collision_free_vamp = envs[1].is_collision_free

    rai_time = 0
    vamp_time = 0

    # actually do the benchmarking
    print("Starting benchmark")
    for _ in range(N):

        idx = random.randint(0, len(modes_for_envs[0])-1)

        m_rai = modes_for_envs[0][idx]
        m_vamp = modes_for_envs[1][idx]

        q = env.sample_config_uniform_in_limits()
    
        start = time.time()
        is_collision_free_rai(q, m_rai)
        end = time.time()

        rai_time += end - start

        start = time.time()
        is_collision_free_vamp(q, m_vamp)
        end = time.time()

        vamp_time += end - start


    print(f"Took on avg. {(rai_time) / N * 1000} ms for a rai collision check.")
    print(f"Took on avg. {(vamp_time) / N * 1000} ms for a vamp collision check.")


def main():

    np.random.seed(0)
    random.seed(0)

    env_rai = get_env_by_name("rai.ur5_box_stacking")
    env_vampmr = get_env_by_name("vampmr.ur5_box_stacking")

    benchmark_collision_checking([env_rai, env_vampmr], 100_000)


if __name__ == "__main__":
    main()
