import argparse
from matplotlib import pyplot as plt
import time
import numpy as np
import random

from typing import List
import os
import sys
current_file_dir = os.path.dirname(os.path.abspath(__file__))  # Current file's directory
project_root = os.path.abspath(os.path.join(current_file_dir, ".."))
src_path = os.path.abspath(os.path.join(project_root, "../src"))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))
from multi_robot_multi_goal_planning.problems import get_env_by_name
from multi_robot_multi_goal_planning.problems.rai_envs import display_path, rai_env

# from multi_robot_multi_goal_planning.problems.planning_env import State
# from multi_robot_multi_goal_planning.problems.configuration import config_dist
from multi_robot_multi_goal_planning.problems.util import interpolate_path

# planners
from multi_robot_multi_goal_planning.planners.prioritized_planner import (
    prioritized_planning,
)
from multi_robot_multi_goal_planning.planners.joint_prm_planner import joint_prm_planner
from multi_robot_multi_goal_planning.planners.tensor_prm_planner import (
    tensor_prm_planner,
)

# np.random.seed(100)


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
    parser.add_argument(
        "--prm_k_nearest",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=True,
        help="Use k-nearest (default: True)",
    )
    parser.add_argument(
        "--prm_sample_near_path",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=False,
        help="Generate samples near a previously found path (default: False)",
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
            distance_metric=args.distance_metric,
            try_sampling_around_path=args.prm_sample_near_path,
            use_k_nearest=args.prm_k_nearest,
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

    interpolated_path = interpolate_path(path, 0.05)

    print("Checking original path for validity")
    print(env.is_valid_plan(interpolated_path))

    print("cost", info["costs"])
    print("comp_time", info["times"])

    plt.figure()
    plt.plot(info["times"], info["costs"], "-o", drawstyle="steps-post")

    mode_switch_indices = []
    for i in range(len(interpolated_path) - 1):
        if interpolated_path[i].mode != interpolated_path[i + 1].mode:
            mode_switch_indices.append(i)

    plt.figure("Path cost")
    plt.plot(
        env.batch_config_cost(interpolated_path[:-1], interpolated_path[1:]),
        label="Original",
    )
    plt.plot(mode_switch_indices, [0.1] * len(mode_switch_indices), "o")

    plt.figure("Cumulative path cost")
    plt.plot(
        np.cumsum(env.batch_config_cost(interpolated_path[:-1], interpolated_path[1:])),
        label="Original",
    )
    plt.plot(mode_switch_indices, [0.1] * len(mode_switch_indices), "o")

    # plt.figure()

    # plt.plot([pt.q.state()[0] for pt in interpolated_path], [pt.q.state()[1] for pt in interpolated_path], 'o-')
    # plt.plot([pt.q.state()[3] for pt in interpolated_path], [pt.q.state()[4] for pt in interpolated_path], 'o-')

    # plt.plot([pt.q.state()[0] for pt in shortcut_discretized_path], [pt.q.state()[1] for pt in shortcut_discretized_path], 'o--')
    # plt.plot([pt.q.state()[3] for pt in shortcut_discretized_path], [pt.q.state()[4] for pt in shortcut_discretized_path], 'o--')

    plt.show()

    print("displaying path from planner")
    display_path(env, interpolated_path, stop=False, stop_at_end=True)


if __name__ == "__main__":
    main()
