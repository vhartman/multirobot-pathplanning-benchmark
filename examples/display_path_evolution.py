import argparse
from matplotlib import pyplot as plt
import matplotlib.cm as cm

import datetime
import json
import os
from pathlib import Path
import re

import numpy as np
import random

# from typing import Dict, Any, Callable, Tuple, List

from multi_robot_multi_goal_planning.problems import get_env_by_name

from multi_robot_multi_goal_planning.problems.planning_env import State
from multi_robot_multi_goal_planning.problems.util import interpolate_path
# from multi_robot_multi_goal_planning.problems.configuration import config_dist
from run_experiment import load_experiment_config

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


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("folder", nargs="?", default="", help="filepath")
    parser.add_argument("env_name", nargs="?", default="", help="filepath")
    parser.add_argument(
        "--interpolate",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=True,
        help="Interpolate the path that is loaded. (default: True)",
    )
    parser.add_argument(
        "--plot",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=False,
        help="Plot the path. (default: True)",
    )
    args = parser.parse_args()

    files = [
        f
        for f in os.listdir(args.folder)
        if os.path.isfile(os.path.join(args.folder, f))
    ]
    path_nums = [int(f[5:-5]) for f in files]
    sorted_files = [x for _, x in sorted(zip(path_nums, files))]
    folder_path = re.match(r'(.*?/out/[^/]+)', args.folder).group(1)
    potential_config_path = os.path.join(folder_path, 'config.json')
    if Path(potential_config_path).exists():
        config = load_experiment_config(potential_config_path)
        seed = config["seed"]
    else:
        seed = 0
    
    np.random.seed(seed)
    random.seed(seed)

    env = get_env_by_name(args.env_name)

    path_data = load_path(args.folder + sorted_files[0])
    path = convert_to_path(env, path_data)

    num_agents = path[0].q.num_agents()

    # Create a single figure with subplots for each agent
    fig, axes = plt.subplots(1, num_agents, figsize=(5 * num_agents, 5))

    if num_agents == 1:
        axes = [axes]  # Ensure axes is always iterable
    
    for j, file in enumerate(sorted_files):
        path_data = load_path(args.folder + file)

        path = convert_to_path(env, path_data)

        for i in range(path[0].q.num_agents()):
        
            x = [pt.q[i][0] for pt in path]
            y = [pt.q[i][1] for pt in path]
            num_points = len(x)

            # Normalize the indices to be used with a colormap
            colors = cm.viridis(
                np.linspace(0, 1, num_points)
            )  # Change 'viridis' to any colormap

            for k in range(num_points - 1):
                axes[i].plot(
                    x[k : k + 2],
                    y[k : k + 2],
                    "o-",
                    color=colors[k],
                    alpha=j / (len(sorted_files) - 1),
                )  # Connect points with a gradient effect

            axes[i].set_title(f"Agent {i}")
            axes[i].set_xlabel("X")
            axes[i].set_ylabel("Y")
            axes[i].set_aspect("equal")

        # for i in range(path[0].q.num_agents()):
        #     plt.plot(
        #         [pt.q[i][0] for pt in path],
        #         [pt.q[i][1] for pt in path],
        #         "o-",
        #         color=agent_to_color[i],
        #         alpha=j / (len(sorted_files) - 1),
        #     )

    # plt.show(block=False)
    plt.show()

    if args.interpolate:
        path = interpolate_path(path)

    print("Attempting to display path")
    env.show()
    # display_path(env, real_path, True, True)

    env.display_path(path, False, export=False, stop_at_end=True)


if __name__ == "__main__":
    main()
