import argparse
from matplotlib import pyplot as plt
import numpy as np

import datetime
import os
import random

from multi_robot_multi_goal_planning.problems import get_env_by_name

# from multi_robot_multi_goal_planning.problems.configuration import config_dist
from multi_robot_multi_goal_planning.problems.util import interpolate_path

# planners
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    IterationTerminationCondition,
    RuntimeTerminationCondition,
)

from multi_robot_multi_goal_planning.planners.prioritized_planner import (
    prioritized_planning,
)
from multi_robot_multi_goal_planning.planners.composite_prm_planner import (
    CompositePRMConfig,
    CompositePRM,
)
from multi_robot_multi_goal_planning.planners.shortcutting import (
    single_mode_shortcut,
    robot_mode_shortcut,
)
from multi_robot_multi_goal_planning.planners.tensor_prm_planner import (
    tensor_prm_planner,
)
from multi_robot_multi_goal_planning.planners.planner_rrtstar import RRTstar
from multi_robot_multi_goal_planning.planners.planner_birrtstar import (
    BidirectionalRRTstar,
)

from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTConfig
from multi_robot_multi_goal_planning.planners.itstar_base import BaseITConfig

from multi_robot_multi_goal_planning.planners.planner_aitstar import AITstar
from multi_robot_multi_goal_planning.planners.planner_eitstar import EITstar

from run_experiment import export_planner_data


def main():
    parser = argparse.ArgumentParser(description="Planner runner")
    parser.add_argument("env", nargs="?", default="default", help="env to show")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable optimization (default: True)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed")
    parser.add_argument(
        "--num_iters", type=int, help="Maximum number of iterations for termination."
    )
    parser.add_argument(
        "--max_time", type=float, help="Maximum runtime (in seconds) for termination."
    )
    parser.add_argument(
        "--planner",
        choices=[
            "joint_prm",
            "tensor_prm",
            "prioritized",
            "rrt_star",
            "birrt_star",
            "aitstar",
            "eitstar",
        ],
        default="joint_prm",
        help="Planner to use (default: joint_prm)",
    )
    parser.add_argument(
        "--distance_metric",
        choices=["euclidean", "sum_euclidean", "max", "max_euclidean"],
        default="euclidean",
        help="Distance metric to use (default: max)",
    )
    parser.add_argument(
        "--per_agent_cost_function",
        choices=["euclidean", "max"],
        default="euclidean",
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
        action="store_true",
        help="Use k-nearest (default: False)",
    )
    parser.add_argument(
        "--prm_sample_near_path",
        action="store_true",
        help="Generate samples near a previously found path (default: False)",
    )
    parser.add_argument(
        "--prm_informed_sampling",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=True,
        help="Generate samples near a previously found path (default: False)",
    )
    parser.add_argument(
        "--prm_shortcutting",
        action="store_true",
        help="Try shortcutting the solution.",
    )
    parser.add_argument(
        "--prm_locally_informed_sampling",
        action="store_true",
        help="Try shortcutting the solution.",
    )
    parser.add_argument(
        "--prm_direct_sampling",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=True,
        help="Generate samples near a previously found path (default: False)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Try shortcutting the solution.",
    )
    parser.add_argument(
        "--stop_at_mode",
        action="store_true",
        help="Generate samples near a previously found path (default: False)",
    )
    args = parser.parse_args()

    if args.num_iters is not None and args.max_time is not None:
        raise ValueError("Cannot specify both num_iters and max_time.")

    np.random.seed(args.seed)
    random.seed(args.seed)

    env = get_env_by_name(args.env)
    env.cost_reduction = args.cost_reduction
    env.cost_metric = args.per_agent_cost_function

    env.show()

    termination_condition = None
    if args.num_iters is not None:
        termination_condition = IterationTerminationCondition(args.num_iters)
    elif args.max_time is not None:
        termination_condition = RuntimeTerminationCondition(args.max_time)

    assert termination_condition is not None

    if args.planner == "joint_prm":
        config = CompositePRMConfig()

        config.mode_sampling_type = None
        config.distance_metric = args.distance_metric
        config.try_sampling_around_path = args.prm_sample_near_path
        config.use_k_nearest = args.prm_k_nearest
        config.try_informed_sampling = args.prm_informed_sampling
        config.try_shortcutting = args.prm_shortcutting
        config.try_direct_informed_sampling = args.prm_direct_sampling
        config.locally_informed_sampling = args.prm_locally_informed_sampling

        planner = CompositePRM(env, config)

        path, info = planner.plan(ptc=termination_condition, optimize=args.optimize)

        if args.save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # convention: alsways use "/" as trailing character
            experiment_folder = f"./out/{timestamp}_{args.env}/"

            # export_config(experiment_folder, config)

            if not os.path.isdir(experiment_folder):
                os.makedirs(experiment_folder)

            planner_folder = experiment_folder + args.planner + "/"
            export_planner_data(planner_folder, 0, info)
    elif args.planner == "rrt_star":
        config = BaseRRTConfig()

        config.distance_metric = args.distance_metric

        planner = RRTstar(env, config=config)
        path, info = planner.plan(ptc=termination_condition, optimize=args.optimize)
    elif args.planner == "birrt_star":
        config = BaseRRTConfig()

        config.distance_metric = args.distance_metric

        planner = BidirectionalRRTstar(env, config=config)
        path, info = planner.plan(ptc=termination_condition, optimize=args.optimize)
    elif args.planner == "aitstar":
        config = BaseITConfig()

        config.distance_metric = args.distance_metric

        planner = AITstar(env, config=config)
        path, info = planner.plan(ptc=termination_condition, optimize=args.optimize)

    elif args.planner == "eitstar":
        config = BaseITConfig()

        config.distance_metric = args.distance_metric

        planner = EITstar(env, config=config)
        path, info = planner.plan(ptc=termination_condition, optimize=args.optimize)

    # elif args.planner == "tensor_prm":
    #     path, info = tensor_prm_planner(
    #         env,
    #         optimize=args.optimize,
    #         mode_sampling_type=None,
    #         max_iter=args.num_iters,
    #     )
    # elif args.planner == "prioritized":
    #     path, info = prioritized_planning(env)

    print("robot-mode-shortcut")
    shortcut_path, info_shortcut = robot_mode_shortcut(
        env,
        path,
        1000,
        tolerance=env.collision_tolerance,
        resolution=env.collision_resolution,
    )

    print("task-shortcut")
    single_mode_shortcut_path, info_single_mode_shortcut = single_mode_shortcut(
        env, path, 1000
    )

    interpolated_path = interpolate_path(path, 0.05)

    print("Checking original path for validity")
    print(env.is_valid_plan(interpolated_path))

    print("Checking mode-shortcutted path for validity")
    print(env.is_valid_plan(single_mode_shortcut_path))

    print("Checking task shortcutted path for validity")
    print(env.is_valid_plan(shortcut_path))

    print("cost", info["costs"])
    print("comp_time", info["times"])

    plt.figure()
    plt.plot(info["times"], info["costs"], "-o", drawstyle="steps-post")

    plt.figure()
    for name, info in zip(
        ["task-shortcut", "mode-shortcut"], [info_shortcut, info_single_mode_shortcut]
    ):
        plt.plot(info[1], info[0], drawstyle="steps-post", label=name)

    plt.xlabel("time")
    plt.ylabel("cost")
    plt.legend()

    mode_switch_indices = []
    for i in range(len(interpolated_path) - 1):
        if interpolated_path[i].mode != interpolated_path[i + 1].mode:
            mode_switch_indices.append(i)

    plt.figure("Path cost")
    plt.plot(
        env.batch_config_cost(interpolated_path[:-1], interpolated_path[1:]),
        label="Original",
    )
    plt.plot(
        env.batch_config_cost(shortcut_path[:-1], shortcut_path[1:]), label="Shortcut"
    )
    plt.plot(mode_switch_indices, [0.1] * len(mode_switch_indices), "o")
    plt.legend()

    plt.figure("Cumulative path cost")
    plt.plot(
        np.cumsum(env.batch_config_cost(interpolated_path[:-1], interpolated_path[1:])),
        label="Original",
    )
    plt.plot(
        np.cumsum(env.batch_config_cost(shortcut_path[:-1], shortcut_path[1:])),
        label="Shortcut",
    )
    plt.plot(mode_switch_indices, [0.1] * len(mode_switch_indices), "o")
    plt.legend()

    shortcut_discretized_path = interpolate_path(shortcut_path)

    plt.figure()

    plt.plot(
        [pt.q[0][0] for pt in interpolated_path],
        [pt.q[0][1] for pt in interpolated_path],
        "o-",
    )
    plt.plot(
        [pt.q[1][0] for pt in interpolated_path],
        [pt.q[1][1] for pt in interpolated_path],
        "o-",
    )

    plt.plot(
        [pt.q[0][0] for pt in shortcut_discretized_path],
        [pt.q[0][1] for pt in shortcut_discretized_path],
        "o--",
    )
    plt.plot(
        [pt.q[1][0] for pt in shortcut_discretized_path],
        [pt.q[1][1] for pt in shortcut_discretized_path],
        "o--",
    )

    plt.show()

    print("displaying path from planner")
    env.display_path(
        interpolated_path,
        stop=False,
        stop_at_end=True,
        adapt_to_max_distance=True,
        stop_at_mode=args.stop_at_mode,
    )

    print("displaying path from shortcut path")
    env.display_path(
        shortcut_discretized_path,
        stop=False,
        adapt_to_max_distance=True,
        stop_at_mode=args.stop_at_mode,
    )


if __name__ == "__main__":
    main()
