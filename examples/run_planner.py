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
            use_k_nearest=args.prm_k_nearest
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
    plt.plot(info["times"], info["costs"], "o")

    plt.show()

    print("displaying path from planner")
    display_path(env, interpolated_path, stop=False, stop_at_end=True)


if __name__ == "__main__":
    main()
