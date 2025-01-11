import argparse
from matplotlib import pyplot as plt
import time
import numpy as np
import random

from typing import List

from multi_robot_multi_goal_planning.problems import get_env_by_name
from multi_robot_multi_goal_planning.problems.rai_envs import display_path, rai_env
from multi_robot_multi_goal_planning.problems.planning_env import State
from multi_robot_multi_goal_planning.problems.configuration import config_dist
from multi_robot_multi_goal_planning.problems.util import path_cost

# planners
from multi_robot_multi_goal_planning.planners.prioritized_planner import (
    prioritized_planning,
)
from multi_robot_multi_goal_planning.planners.joint_prm_planner import joint_prm_planner
from multi_robot_multi_goal_planning.planners.tensor_prm_planner import (
    tensor_prm_planner,
)

# np.random.seed(100)


def robot_mode_shortcut(env: rai_env, path: List[State], max_iter: int = 1000):
    new_path = interpolate_path(path, 0.05)

    costs = [path_cost(new_path, env.batch_config_cost)]
    times = [0.0]

    start_time = time.time()

    config_type = type(env.get_start_pos())

    cnt = 0
    for iter in range(max_iter):
        i = np.random.randint(0, len(new_path))
        j = np.random.randint(0, len(new_path))

        if i > j:
            tmp = i
            i = j
            j = tmp

        if abs(j - i) < 2:
            continue

        robots_to_shortcut = [r for r in range(len(env.robots))]
        random.shuffle(robots_to_shortcut)
        # num_robots = np.random.randint(0, len(robots_to_shortcut))
        num_robots = 1
        robots_to_shortcut = robots_to_shortcut[:num_robots]

        can_shortcut_this = True
        for r in robots_to_shortcut:
            if new_path[i].mode.task_ids[r] != new_path[j].mode.task_ids[r]:
                can_shortcut_this = False

        if not can_shortcut_this:
            continue

        cnt += 1

        q0 = new_path[i].q
        q1 = new_path[j].q

        # constuct pth element for the shortcut
        path_element = []
        for k in range(j - i + 1):
            q = []
            for r in range(len(env.robots)):
                # print(r, i, j, k)
                if r in robots_to_shortcut:
                    q_interp = q0[r] + (q1[r] - q0[r]) / (j - i) * k
                    q.append(q_interp)
                else:
                    q.append(new_path[i + k].q[r])

            # print(q)
            path_element.append(State(config_type.from_list(q), new_path[i + k].mode))

        # check if the shortcut improves cost
        if path_cost(path_element, env.batch_config_cost) >= path_cost(
            new_path[i : j + 1], env.batch_config_cost
        ):
            continue

        # this is wrong for partial shortcuts atm.
        if env.is_path_collision_free(path_element):
            for k in range(j - i + 1):
                new_path[i + k].q = path_element[k].q

                # if not np.array_equal(new_path[i+k].mode, path_element[k].mode):
                # print('fucked up')

        current_time = time.time()
        times.append(current_time - start_time)
        costs.append(path_cost(new_path, env.batch_config_cost))

    assert new_path[-1].mode == path[-1].mode
    assert np.linalg.norm(new_path[-1].q.state() - path[-1].q.state()) < 1e-6
    assert np.linalg.norm(new_path[0].q.state() - path[0].q.state()) < 1e-6

    print("original cost:", path_cost(path, env.batch_config_cost))
    print("Attempted shortcuts", cnt)
    print("new cost:", path_cost(new_path, env.batch_config_cost))

    # plt.plot(times, costs)
    # plt.show()

    return new_path, [costs, times]

def interpolate_path(path: List[State], resolution: float = 0.1):
    config_type = type(path[0].q)
    new_path = []

    # discretize path
    for i in range(len(path) - 1):
        q0 = path[i].q
        q1 = path[i + 1].q

        if path[i].mode != path[i + 1].mode:
            continue

        dist = config_dist(q0, q1)
        N = int(dist / resolution)
        N = max(1, N)

        for j in range(N):
            q = []
            for k in range(q0.num_agents()):
                qr = q0.robot_state(k) + (q1.robot_state(k) - q0.robot_state(k)) / N * j
                q.append(qr)

                # env.C.setJointState(qr, get_robot_joints(env.C, env.robots[k]))

                # env.C.setJointState(qr, [env.robots[k]])

            # env.C.view(True)

            new_path.append(State(config_type.from_list(q), path[i].mode))

    new_path.append(State(path[-1].q, path[-1].mode))

    return new_path


def main():
    parser = argparse.ArgumentParser(description="Planner runner")
    parser.add_argument("env", nargs="?", default="default", help="env to show")
    parser.add_argument(
        "--optimize",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=True,
        help="Enable optimization (default: True)",
    )
    parser.add_argument(
        "--num_iters", type=int, default=2000, help="Number of iterations"
    )
    parser.add_argument(
        "--planner",
        choices=["joint_prm", "tensor_prm", "prioritized"],
        default="joint_prm",
        help="Planner to use (default: joint_prm)",
    )
    parser.add_argument(
        "--distance_metric",
        choices=["euclidean", "sum_euclidean", "max", "max_euclidean"],
        default="max",
        help="Distance metric to use (default: max)",
    )
    parser.add_argument(
        "--per_agent_cost_function",
        choices=["euclidean", "max"],
        default="max",
        help="Per agent cost function to use (default: max)",
    )
    parser.add_argument(
        "--cost_reduction",
        choices=["sum", "max"],
        default="max",
        help="How the agent specific cost functions are reduced to one single number (default: max)",
    )
    args = parser.parse_args()

    env = get_env_by_name(args.env)
    env.cost_reduction = args.cost_reduction
    env.cost_metric = args.per_agent_cost_function

    env.show()

    if args.planner == "joint_prm":
        path, info = joint_prm_planner(
            env,
            optimize=args.optimize,
            mode_sampling_type=None,
            max_iter=args.num_iters,
            distance_metric=args.distance_metric
        )
    elif args.planner == "tensor_prm":
        path, info = tensor_prm_planner(
            env,
            optimize=args.optimize,
            mode_sampling_type=None,
            max_iter=args.num_iters,
        )
    elif args.planner == "prioritized":
        path, info = prioritized_planning(env)

    shortcut_path, info_shortcut = robot_mode_shortcut(env, path, 10000)
    interpolated_path = interpolate_path(path, 0.05)

    print("Checking original path for validity")
    print(env.is_valid_plan(interpolated_path))

    print("Checking shortcutted path for validity")
    print(env.is_valid_plan(shortcut_path))


    print("cost", info["costs"])
    print("comp_time", info["times"])

    plt.figure()
    plt.plot(info["times"], info["costs"], "-o", drawstyle='steps-post')

    plt.figure()
    for info in [info_shortcut]:
        plt.plot(info[1], info[0], drawstyle='steps-post')

    plt.xlabel("time")
    plt.ylabel("cost")

    mode_switch_indices = []
    for i in range(len(interpolated_path)-1):
        if interpolated_path[i].mode != interpolated_path[i+1].mode:
            mode_switch_indices.append(i)

    plt.figure("Path cost")
    plt.plot(env.batch_config_cost(interpolated_path[:-1], interpolated_path[1:]), label="Original")
    plt.plot(env.batch_config_cost(shortcut_path[:-1], shortcut_path[1:]), label="Shortcut")
    plt.plot(mode_switch_indices, [0.1]*len(mode_switch_indices), 'o')

    plt.show()

    print("displaying path from planner")
    display_path(env, interpolated_path, stop=False, stop_at_end=True)

    print("displaying path from shortcut path")
    shortcut_discretized_path = interpolate_path(shortcut_path)
    display_path(env, shortcut_discretized_path, stop=False)

if __name__ == "__main__":
    main()
