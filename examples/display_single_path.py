import argparse
from matplotlib import pyplot as plt

from pathlib import Path
import json
import os
import re
import random

import numpy as np

# from typing import Dict, Any, Callable, Tuple, List

from multi_robot_multi_goal_planning.problems import get_env_by_name
from multi_robot_multi_goal_planning.problems.rai_base_env import rai_env

from multi_robot_multi_goal_planning.problems.planning_env import State
from multi_robot_multi_goal_planning.problems.util import interpolate_path, path_cost
from multi_robot_multi_goal_planning.planners.shortcutting import (
    robot_mode_shortcut,
)
from run_experiment import load_experiment_config
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

    for i, a in enumerate(path_data):
        q_np = np.array(a["q"])
        q = env.get_start_pos().from_flat(q_np)

        # print(i)
        # print(q.state())

        if a["mode"] != prev_mode_ids:
        # if a["mode"] != prev_mode_ids and a["mode"] != env._terminal_task_ids:
            if a["mode"] in env.get_valid_next_task_combinations(modes[-1]):
                # print("current read mode", a["mode"])
                # print(q)
                # try:
                # print(prev_config.state(), modes[-1])
                next_modes = env.get_next_modes(prev_config, modes[-1])
                if len(next_modes) == 1:
                    next_mode = next_modes[0]
                else:
                    next_modes_task_ids = [m.task_ids for m in next_modes]
                    idx = next_modes_task_ids.index(a["mode"])
                    next_mode = next_modes[idx]


                if next_mode.task_ids == a["mode"]:
                    modes.append(next_mode)
                # except:
                #     break

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
        "--dont_insert_transition_nodes",
        action="store_false",
        help="Shortcut the path. (default: False)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the path. (default: False)",
    )
    parser.add_argument(
        "--cost_plot",
        action="store_true",
        help="Plot the path. (default: False)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Plot the path. (default: False)",
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Plot the path. (default: False)",
    )
    parser.add_argument(
        "--show_coll_config",
        action="store_true",
        help="Display the configuration used for collision checking. (default: False)",
    )
    args = parser.parse_args()

    folder_path = re.match(r'(.*?/out/[^/]+)', args.path_filename).group(1)
    potential_config_path = os.path.join(folder_path, 'config.json')
    if Path(potential_config_path).exists():
        config = load_experiment_config(potential_config_path)
        seed = config["seed"] 
    else:
        seed = 0
    
    np.random.seed(seed)
    random.seed(seed)

    path_data = load_path(args.path_filename)

    env = get_env_by_name(args.env_name)
    env.cost_reduction = "max"
    env.cost_metric = "euclidean"

    if not args.show_coll_config and isinstance(env, rai_env):
        env.C_base = env.C_orig
        env.C = env.C_orig

    path = convert_to_path(env, path_data)

    cost = path_cost(path, env.batch_config_cost)
    print("cost", cost)

    if args.plot:
        plt.figure()
        for i in range(path[0].q.num_agents()):
            plt.plot([pt.q[i][0] for pt in path], [pt.q[i][1] for pt in path], "o-")

        # plt.show(block=False)

        # make_mode_plot(path, env)
        plt.show()
    
    if args.cost_plot:
        pts = [start.q.state() for start in path]
        agent_slices = path[0].q._array_slice
        batch_costs = env.batch_config_cost(pts, None, tmp_agent_slice=agent_slices)

        mode_switch_idx = []
        prev_mode = None
        for i, s in enumerate(path):
            if prev_mode is None or s.mode != prev_mode:
                mode_switch_idx.append(i)
                prev_mode = s.mode
                print(s.mode)

        print(mode_switch_idx)

        plt.figure()
        plt.plot(batch_costs)

        # plt.show(block=False)

        # make_mode_plot(path, env)
        plt.show()

    if args.dont_insert_transition_nodes:
        path_w_doubled_modes = []
        for i in range(len(path)):
            path_w_doubled_modes.append(path[i])

            if i + 1 < len(path) and path[i].mode != path[i + 1].mode:
                path_w_doubled_modes.append(State(path[i].q, path[i + 1].mode))

        path = path_w_doubled_modes

    if args.interpolate:
        path = interpolate_path(path, 0.1)

    if args.shortcut:
        plt.figure()
        for i in range(path[0].q.num_agents()):
            plt.plot([pt.q[i][0] for pt in path], [pt.q[i][1] for pt in path], "o-")

        # plt.show(block=False)

        # make_mode_plot(path, env)

        path, _ = robot_mode_shortcut(
            env,
            path,
            10000,
            resolution=env.collision_resolution,
            tolerance=env.collision_tolerance,
        )

        plt.figure()
        for i in range(path[0].q.num_agents()):
            plt.plot([pt.q[i][0] for pt in path], [pt.q[i][1] for pt in path], "o-")

        # plt.show(block=False)

        # make_mode_plot(path, env)
        plt.show()

        cost = path_cost(path, env.batch_config_cost)
        print("cost", cost)

    print("Attempting to display path")
    env.show()
    # display_path(env, real_path, True, True)

    env.display_path(
        path,
        args.pause,
        export=args.export,
        pause_time=0.05,
        stop_at_end=True,
        adapt_to_max_distance=True,
        stop_at_mode=False
    )


if __name__ == "__main__":
    main()
