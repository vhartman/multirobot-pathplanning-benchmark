import argparse
from matplotlib import pyplot as plt

import datetime
import json
import os

import numpy as np

# from typing import Dict, Any, Callable, Tuple, List

from multi_robot_multi_goal_planning.problems import get_env_by_name
from multi_robot_multi_goal_planning.problems.rai_envs import display_path

from multi_robot_multi_goal_planning.problems.planning_env import State
from multi_robot_multi_goal_planning.problems.util import interpolate_path
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

        if a["mode"] != prev_mode_ids:
            next_mode = env.get_next_mode(prev_config, modes[-1])
            modes.append(next_mode)

        real_path.append(State(q, modes[-1]))
        prev_mode_ids = a["mode"]

        prev_config = q

        # env.set_to_mode(modes[-1])
        # env.show_config(q, True)

    return real_path


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("path_filename", nargs="?", default="", help="filepath")
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

    path_data = load_path(args.path_filename)
    env = get_env_by_name(args.env_name)

    path = convert_to_path(env, path_data)

    if args.plot:
        plt.figure()
        for i in range(path[0].q.num_agents()):
            plt.plot([pt.q[i][0] for pt in path], [pt.q[i][1] for pt in path], "o-")

        # plt.show(block=False)
        plt.show()

    if args.interpolate:
        path = interpolate_path(path)

    if args.shortcut:
        env.cost_reduction = "max"
        env.cost_metric = "euclidean"
        path, _ = robot_mode_shortcut(env, path, 10000)

    print("Attempting to display path")
    env.show()
    # display_path(env, real_path, True, True)

    display_path(
        env,
        path,
        False,
        export=False,
        pause_time=0.05,
        stop_at_end=True,
        adapt_to_max_distance=True,
    )


if __name__ == "__main__":
    main()
