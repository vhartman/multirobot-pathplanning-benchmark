from multi_robot_multi_goal_planning.problems import (
    get_env_by_name,
    get_all_environments,
)
import multi_robot_multi_goal_planning.problems as problems
from multi_robot_multi_goal_planning.problems.rai_envs import rai_env
from multi_robot_multi_goal_planning.problems.planning_env import Mode
from multi_robot_multi_goal_planning.problems.configuration import NpConfiguration

import numpy as np
import argparse
import time
import random


def visualize_modes(env: rai_env, export_images: bool = False):
    env.show()

    q_home = env.start_pos

    m = env.start_mode
    while True:
        print("--------")
        print("Mode", m)

        q = []
        next_task_combos = env.get_valid_next_task_combinations(m)
        if len(next_task_combos) > 0:
            idx = random.randint(0, len(next_task_combos) - 1)
            task = env.get_active_task(m, next_task_combos[idx])
        else:
            task = env.get_active_task(m, None)
        switching_robots = task.robots
        goal_sample = task.goal.sample(m)

        if task.name is not None:
            print("Active Task name:", task.name)
        print("Involved robots: ", task.robots)

        print("Goal state:")
        print(goal_sample)

        print("switching robots: ", switching_robots)

        for j, r in enumerate(env.robots):
            if r in switching_robots:
                # TODO: need to check all goals here
                # figure out where robot r is in the goal description
                offset = 0
                for _, task_robot in enumerate(task.robots):
                    if task_robot == r:
                        q.append(
                            goal_sample[offset : offset + env.robot_dims[task_robot]]
                        )
                        break
                    offset += env.robot_dims[task_robot]
                # q.append(goal_sample)
            else:
                q.append(q_home.robot_state(j))

        print(q)

        print(
            "Is collision free: ",
            env.is_collision_free(type(env.get_start_pos()).from_list(q), m),
        )

        # colls = env.C.getCollisions()
        # for c in colls:
        #     if c[2] < 0:
        #         print(c)

        if export_images:
            env.show(False)
            env.C.view_savePng("./z.img/")
        else:
            env.show_config(type(env.get_start_pos()).from_list(q))

        if env.is_terminal_mode(m):
            break

        m = env.get_next_mode(type(env.get_start_pos()).from_list(q), m)


def benchmark_collision_checking(env: rai_env, N=10000):
    conf_type = type(env.get_start_pos())

    def sample_next_mode(mode: Mode):
        if env.is_terminal_mode(mode):
            return None
        
        while True:
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
                next_mode = env.get_next_mode(q, mode)
                return next_mode

    # create list of modes that we can reach
    print("Make mode list")
    reachable_modes = [env.get_start_mode()]
    max_iter = 500
    for _ in range(max_iter):
        m_rnd = random.choice(reachable_modes)
        next_mode = sample_next_mode(m_rnd)

        if next_mode is not None:
            reachable_modes.append(next_mode)

    is_collision_free = env.is_collision_free

    # actually do the benchmarking
    print("Starting benchmark")
    start = time.time()
    for _ in range(N):
        q = env.sample_config_uniform_in_limits()
        m = random.choice(reachable_modes)

        is_collision_free(q, m)

    end = time.time()

    print(f"Took on avg. {(end - start) / N * 1000} ms for a collision check.")


def main():
    # problems.rai_envs.rai_hallway_two_dim_dependency_graph()
    # print()
    # problems.rai_envs.rai_two_dim_three_agent_env_dependency_graph()

    parser = argparse.ArgumentParser(description="Env shower")
    parser.add_argument("env_name", nargs="?", default="default", help="env to show")
    parser.add_argument(
        "--mode",
        choices=["benchmark", "show", "modes", "list_all"],
        required=True,
        help="Select the mode of operation",
    )
    parser.add_argument(
        "--show_coll_config",
        action="store_true",
        help="Display the configuration used for collision checking. (default: False)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export images of modes. (default: False)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    args = parser.parse_args()

    if args.mode == "list_all":
        all_envs = get_all_environments()
        print("\n".join([name for name in all_envs.keys()]))
        return

    # check_all_modes()

    np.random.seed(args.seed)
    random.seed(args.seed)

    env = get_env_by_name(args.env_name)

    # make use of the original config
    if not args.show_coll_config and isinstance(env, rai_env):
        env.C_base = env.C_orig
        env.C = env.C_orig

    if args.mode == "show":
        print("Environment starting position")
        env.show()
    elif args.mode == "benchmark":
        benchmark_collision_checking(env)
    elif args.mode == "modes":
        print("Environment modes/goals")
        visualize_modes(env, args.export)


if __name__ == "__main__":
    main()
