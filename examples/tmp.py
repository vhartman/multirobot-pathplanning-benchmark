import argparse
from matplotlib import pyplot as plt

import datetime
import json
import os
import random

import numpy as np

# from typing import Dict, Any, Callable, Tuple, List

from multi_robot_multi_goal_planning.problems import get_env_by_name

from multi_robot_multi_goal_planning.problems.planning_env import State
from multi_robot_multi_goal_planning.problems.util import interpolate_path, path_cost
from multi_robot_multi_goal_planning.planners.shortcutting import (
    single_mode_shortcut,
    robot_mode_shortcut,
)

# from multi_robot_multi_goal_planning.problems.configuration import config_dist


def load_path(filename):
    with open(filename) as f:
        d = json.load(f)
        return d


def convert_to_path(env, path_data):
    real_path = []
    prev_mode_ids = env.start_mode.task_ids

    modes = [env.start_mode]

    start_conf = env.get_start_pos()
    prev_config = start_conf

    for a in path_data:
        q_np = np.array(a["q"])
        q = type(env.get_start_pos())(q_np, start_conf.array_slice)

        if a["mode"] != prev_mode_ids and a["mode"] != env._terminal_task_ids:
            next_modes = env.get_next_modes(prev_config, modes[-1])
            # assert len(next_modes) == 1
            # next_mode = next_modes[0]
            if len(next_modes) == 1:
                next_mode = next_modes[0]
            else:
                next_modes_task_ids = [m.task_ids for m in next_modes]
                idx = next_modes_task_ids.index(a["mode"])
                next_mode = next_modes[idx]

            modes.append(next_mode)

        real_path.append(State(q, modes[-1]))
        prev_mode_ids = a["mode"]

        prev_config = q

        # env.set_to_mode(modes[-1])
        # env.show_config(q, True)

    return real_path


def make_mode_plot(path, env):
    data = []

    for p in path:
        data.append(p.mode.task_ids)

    data = np.array(data)
    num_robots = data.shape[1]

    fig, ax = plt.subplots(figsize=(10, 5))

    for robot_id in range(num_robots):
        active_value = None
        start_idx = None

        for t in range(data.shape[0]):
            if active_value is None or data[t, robot_id] != active_value:
                if active_value is not None:
                    # Draw a box from start_idx to t-1
                    color = f"C{active_value}"

                    if env.tasks[active_value].type is not None:
                        if env.tasks[active_value].type == "pick":
                            color = "tab:green"
                        elif env.tasks[active_value].type == "place":
                            color = "tab:orange"
                        else:
                            color = "tab:blue"

                    ax.add_patch(
                        plt.Rectangle(
                            (start_idx, robot_id + 0.25),
                            t - start_idx,
                            0.5,
                            color=color,
                            alpha=0.8,
                            edgecolor="black",
                            linewidth=1.5,
                        )
                    )
                active_value = data[t, robot_id]
                start_idx = t

        # Final segment
        if active_value is not None:
            ax.add_patch(
                plt.Rectangle(
                    (start_idx, robot_id + 0.25),
                    data.shape[0] - start_idx,
                    0.5,
                    color=f"C{active_value}",
                    alpha=0.6,
                )
            )

    ax.set_xlim(0, data.shape[0])
    ax.set_ylim(0, num_robots)
    ax.set_yticks(np.arange(num_robots) + 0.5)
    ax.set_yticklabels([f"Robot {i}" for i in range(num_robots)])
    ax.set_xlabel("Time")
    ax.set_ylabel("Robots")


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("path_filename_1", nargs="?", default="", help="filepath")
    parser.add_argument("path_filename_2", nargs="?", default="", help="filepath")
    parser.add_argument("env_name", nargs="?", default="", help="filepath")
    parser.add_argument(
        "--interpolate",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=True,
        help="Interpolate the path that is loaded. (default: True)",
    )
    parser.add_argument(
        "--shortcut",
        action="store_true",
        help="Shortcut the path. (default: False)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the path. (default: True)",
    )
    args = parser.parse_args()

    path_data_1 = load_path(args.path_filename_1)
    path_data_2 = load_path(args.path_filename_2)

    np.random.seed(1)
    random.seed(1)

    env = get_env_by_name(args.env_name)
    env.cost_reduction = "max"
    env.cost_metric = "euclidean"

    path_1 = convert_to_path(env, path_data_1)
    path_2 = convert_to_path(env, path_data_2)

    cost_1 = path_cost(path_1, env.batch_config_cost)
    cost_2 = path_cost(path_2, env.batch_config_cost)
    
    print("cost",cost_1)
    print("cost",cost_2)

    if args.plot:
        for path in [path_1, path_2]:
            plt.figure()
            for i in range(path[0].q.num_agents()):
                plt.plot([pt.q[i][0] for pt in path], [pt.q[i][1] for pt in path], "o-")
            
            for i in range(len(path)-1):
                if path[i].mode != path[i+1].mode:
                    for j in range(path[0].q.num_agents()):
                        plt.scatter([path[i].q[j][0]], [path[i].q[j][1]], s=100, color='black')
                    

            # plt.show(block=False)

            # make_mode_plot(path, env)

        plt.show()

    if args.interpolate:
        path_1 = interpolate_path(path_1, 0.01)
        path_2 = interpolate_path(path_2, 0.01)

    if args.shortcut:
        path, _ = robot_mode_shortcut(
            env,
            path,
            10000,
            resolution=env.collision_resolution,
            tolerance=env.collision_tolerance,
        )

        cost = path_cost(path, env.batch_config_cost)
        print("cost",cost)

    print("Attempting to display path")
    env.show()
    # display_path(env, real_path, True, True)

    env.display_path(
        path_1,
        False,
        export=False,
        pause_time=0.05,
        stop_at_end=True,
        adapt_to_max_distance=True,
    )

    env.show()
    env.display_path(
        path_2,
        False,
        export=False,
        pause_time=0.05,
        stop_at_end=True,
        adapt_to_max_distance=True,
    )


if __name__ == "__main__":
    main()
