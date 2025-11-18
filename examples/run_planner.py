import argparse
from simple_parsing import ArgumentParser

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np

import datetime
import os
import random

import copy

from multi_robot_multi_goal_planning.problems import get_env_by_name

from multi_robot_multi_goal_planning.problems.planning_env import State
from multi_robot_multi_goal_planning.problems.util import interpolate_path

from multi_robot_multi_goal_planning.planners.termination_conditions import (
    IterationTerminationCondition,
    RuntimeTerminationCondition,
)

from multi_robot_multi_goal_planning.planners import (
    PrioritizedPlanner,
    PrioritizedPlannerConfig,
    CompositePRM,
    CompositePRMConfig,
    single_mode_shortcut,
    robot_mode_shortcut,
    robot_mode_shortcut_nl,
    robot_mode_shortcut_nl_opt,
    BaseRRTConfig,
    RRTstar,
    BidirectionalRRTstar,
    BaseITConfig,
    AITstar,
    EITstar,
    RecedingHorizonConfig,
    RecedingHorizonPlanner,
)

from run_experiment import export_planner_data

import logging
# logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    # parser = argparse.ArgumentParser(description="Planner runner")
    parser = ArgumentParser(description="Planner runner")

    parser.add_argument("env", nargs="?", default="default", help="env to show")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable optimization (default: True)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed")
    parser.add_argument("--run_id", type=int, default=0, help="Run id. Used for debugging only.")
    parser.add_argument(
        "--num_iters", type=int, help="Maximum number of iterations for termination."
    )
    parser.add_argument(
        "--max_time", type=float, help="Maximum runtime (in seconds) for termination."
    )
    parser.add_argument(
        "--planner",
        choices=[
            "composite_prm",
            "prioritized",
            "rrt_star",
            "birrt_star",
            "aitstar",
            "eitstar",
            "short_horizon",
        ],
        default="composite_prm",
        help="Planner to use (default: composite_prm)",
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
        "--save",
        action="store_true",
        help="Try shortcutting the solution.",
    )
    parser.add_argument(
        "--stop_at_mode",
        action="store_true",
        help="Generate samples near a previously found path (default: False)",
    )
    parser.add_argument(
        "--insert_transition_nodes",
        action="store_true",
        help="Shortcut the path. (default: False)",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Show some analytics plots. (default: False)",
    )

    # Add planner-specific configs - this is the ONLY change needed!
    parser.add_arguments(CompositePRMConfig, dest="composite_prm_config", prefix="prm.")
    parser.add_arguments(BaseRRTConfig, dest="rrt_config", prefix="rrt.")
    parser.add_arguments(BaseITConfig, dest="it_config", prefix="it.")
    parser.add_arguments(PrioritizedPlannerConfig, dest="prioritized_config", prefix="prio.")
    parser.add_arguments(RecedingHorizonConfig, dest="horizon_config", prefix="horizon.")

    args = parser.parse_args()

    if args.num_iters is not None and args.max_time is not None:
        raise ValueError("Cannot specify both num_iters and max_time.")

    np.random.seed(args.seed)
    random.seed(args.seed)

    env = get_env_by_name(args.env)
    env.cost_reduction = args.cost_reduction
    env.cost_metric = args.per_agent_cost_function

    # env.show()

    termination_condition = None
    if args.num_iters is not None:
        termination_condition = IterationTerminationCondition(args.num_iters)
    elif args.max_time is not None:
        termination_condition = RuntimeTerminationCondition(args.max_time)

    assert termination_condition is not None

    # env_copy = copy.deepcopy(env)

    # Now just use the config objects directly!
    if args.planner == "composite_prm":
        config = args.composite_prm_config
        config.distance_metric = args.distance_metric  # Override if needed
        planner = CompositePRM(env, config)

    elif args.planner == "rrt_star":
        config = args.rrt_config
        config.distance_metric = args.distance_metric
        planner = RRTstar(env, config=config)

    elif args.planner == "birrt_star":
        config = args.rrt_config
        config.distance_metric = args.distance_metric
        planner = BidirectionalRRTstar(env, config=config)

    elif args.planner == "aitstar":
        config = args.it_config
        config.distance_metric = args.distance_metric
        planner = AITstar(env, config=config)

    elif args.planner == "eitstar":
        config = args.it_config
        config.distance_metric = args.distance_metric
        planner = EITstar(env, config=config)

    elif args.planner == "prioritized":
        config = args.prioritized_config
        planner = PrioritizedPlanner(env, config)

    elif args.planner == "short_horizon":
        config = args.horizon_config
        planner = RecedingHorizonPlanner(env, config)

    np.random.seed(args.seed + args.run_id)
    random.seed(args.seed + args.run_id)

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

    assert path is not None

    if args.insert_transition_nodes:
        path_w_doubled_modes = []
        for i in range(len(path)):
            path_w_doubled_modes.append(path[i])

            if i + 1 < len(path) and path[i].mode != path[i + 1].mode:
                path_w_doubled_modes.append(State(path[i].q, path[i + 1].mode))

        path = path_w_doubled_modes

    if False:
        print("robot-mode-shortcut")
        shortcut_path, info_shortcut = robot_mode_shortcut(
            env,
            path,
            1000,
            tolerance=env.collision_tolerance,
            resolution=env.collision_resolution,
            # planner=planner,
        )
    else:
        print("robot-mode-nl-shortcut")
        shortcut_path, info_shortcut = robot_mode_shortcut_nl_opt(
            env,
            path,
            1000,
            tolerance=env.collision_tolerance,
            resolution=env.collision_resolution,
            planner=planner,
        )

    print("task-shortcut")
    single_mode_shortcut_path, info_single_mode_shortcut = single_mode_shortcut(
        env, path, 1000,
    )

    interpolated_path = interpolate_path(path, 0.05)
    # interpolated_path_nl = planner.interpolate_path_nonlinear(path, 0.05)
    shortcut_discretized_path = interpolate_path(shortcut_path)
    # shortcut_discretized_path = planner.interpolate_path_nonlinear(shortcut_path)

    print("Checking original path for validity")
    print(env.is_valid_plan(interpolated_path))

    # print("Checking nonlinear interpolated path for validity")
    # print(env.is_valid_plan(interpolated_path_nl))

    print("Checking mode-shortcutted path for validity")
    print(env.is_valid_plan(single_mode_shortcut_path))

    print("Checking task shortcutted path for validity")
    print(env.is_valid_plan(shortcut_path))

    # print("Checking nonlinear interpolated path for validity")
    # print(env.is_valid_plan(shortcut_discretized_path))

    print("cost", info["costs"])
    print("comp_time", info["times"])

    if args.show_plots:
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
    # display starting configuration to not run it immediately
    env.show(blocking=True)
    env.display_path(
        interpolated_path,
        stop=False,
        stop_at_end=True,
        adapt_to_max_distance=True,
        stop_at_mode=args.stop_at_mode,
    )

    # print("displaying path from nonlinear interpolation")
    # env.display_path(
    #     interpolated_path_nl,
    #     stop=False,
    #     stop_at_end=True,
    #     adapt_to_max_distance=True,
    #     stop_at_mode=args.stop_at_mode,
    # )

    print("displaying path from shortcut path")
    env.display_path(
        shortcut_discretized_path,
        stop=False,
        adapt_to_max_distance=True,
        stop_at_mode=args.stop_at_mode,
    )

    if hasattr(env, "close"):
        env.close()

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     import cProfile
#     import pstats

#     # Profile the full run
#     with cProfile.Profile() as pr:
#         main()  # or whatever function kicks off the planner

#     pr.dump_stats("res.prof")