import multi_robot_multi_goal_planning as mrmgp
import multi_robot_multi_goal_planning.problems as problems
from multi_robot_multi_goal_planning.problems.rai_envs import rai_env

import numpy as np
import argparse
import time


# TODO: make rai-independent
def visualize_modes(env: rai_env):
    env.show()

    q_home = env.start_pos

    m = env.start_mode
    for i in range(len(env.sequence)):
        print("mode", m)
        switching_robots = env.get_goal_constrained_robots(m)

        q = []
        task = env.get_active_task(m)
        goal_sample = task.goal.sample()

        print(task.name)
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
            "is collision free: ",
            env.is_collision_free(type(env.get_start_pos()).from_list(q).state(), m),
        )

        # colls = env.C.getCollisions()
        # for c in colls:
        #     if c[2] < 0:
        #         print(c)

        env.show()

        if m == env.terminal_mode:
            break

        m = env.get_next_mode(None, m)


def benchmark_collision_checking(env: rai_env, N=5000):
    start = time.time()
    for _ in range(N):
        q = []
        for i in range(len(env.robots)):
            lims = env.limits[:, env.robot_idx[env.robots[i]]]
            if lims[0, 0] < lims[1, 0]:
                qr = (
                    np.random.rand(env.robot_dims[env.robots[i]])
                    * (lims[1, :] - lims[0, :])
                    + lims[0, :]
                )
            else:
                qr = np.random.rand(env.robot_dims[env.robots[i]]) * 6 - 3
            q.append(qr)

        m = env.sample_random_mode()

        env.is_collision_free(type(env.get_start_pos()).from_list(q).state(), m)

    end = time.time()

    print(f"Took on avg. {(end-start)/N * 1000} ms for a collision check.")


if __name__ == "__main__":
    # problems.rai_envs.rai_hallway_two_dim_dependency_graph()
    # print()
    # problems.rai_envs.rai_two_dim_three_agent_env_dependency_graph()

    parser = argparse.ArgumentParser(description="Env shower")
    parser.add_argument("env_name", nargs="?", default="default", help="env to show")
    parser.add_argument(
        "--mode",
        choices=["benchmark", "show", "modes"],
        required=True,
        help="Select the mode of operation",
    )
    args = parser.parse_args()

    # check_all_modes()

    env = problems.rai_envs.get_env_by_name(args.env_name)

    if args.mode == "show":
        print("Environment starting position")
        env.show()
    elif args.mode == "benchmark":
        benchmark_collision_checking(env)
    elif args.mode == "modes":
        print("Environment modes/goals")
        visualize_modes(env)
