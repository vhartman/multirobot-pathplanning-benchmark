import argparse
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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

        if a["mode"] != prev_mode_ids:
            # if a["mode"] != prev_mode_ids and a["mode"] != env._terminal_task_ids:
            print(a["mode"])
            next_modes = env.get_next_modes(prev_config, modes[-1])
            assert len(next_modes) == 1
            next_mode = next_modes[0]

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
        "--insert_transition_nodes",
        action="store_true",
        help="Shortcut the path. (default: False)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the path. (default: True)",
    )
    args = parser.parse_args()

    np.random.seed(0)
    random.seed(0)

    path_data = load_path(args.path_filename)

    np.random.seed(0)
    random.seed(0)

    env = get_env_by_name(args.env_name)
    env.cost_reduction = "sum"
    env.cost_metric = "euclidean"

    path = convert_to_path(env, path_data)

    cost = path_cost(path, env.batch_config_cost)
    print("cost", cost)

    if args.interpolate:
        path = interpolate_path(path, 0.01)

    if args.plot:
        obstacles = [
            {"pos": [0.0, -0.5], "size": [1.4, 1.2]},
            {"pos": [0.0, 1.2], "size": [1.4, 1.5]},
        ]

        fig, ax = plt.subplots()
        ax.set_xlim(2, -2)
        ax.set_ylim(2, -2.0)
        ax.set_aspect("equal")
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Draw obstacles
        for obs in obstacles:
            x, y = obs["pos"]
            w, h = obs["size"]
            rect = patches.Rectangle((x - w / 2, y - h / 2), w, h, color="black")
            ax.add_patch(rect)

        # for i in range(path[0].q.num_agents()):
        #     plt.plot([pt.q[i][0] for pt in path], [pt.q[i][1] for pt in path], "o-")
        step = 50

        path = path[::step]

        cmap_1 = cm.get_cmap("cool", len(path))
        cmap_2 = cm.get_cmap("summer", len(path))
        norm = mcolors.Normalize(vmin=0, vmax=len(path) - 1)

        for i in range(2):
            x_vals = [pt.q[i][0] for pt in path]
            y_vals = [pt.q[i][1] for pt in path]
            theta_vals = [pt.q[i][2] for pt in path]
            
            if i == 0:
                shape = "rectangle"
            else:
                shape = "circle"
            
            for t, (x, y, theta) in enumerate(zip(x_vals, y_vals, theta_vals)):
              if shape == "rectangle":
                  color = cmap_1(norm(t))
                  # Simplified rectangle plotting
                  width, height = 0.09, 0.49
                  # Create rectangle at origin
                  rect = patches.Rectangle(
                      (-width/2, -height/2),  # Center the rectangle
                      width, height,
                      angle=np.degrees(theta),
                      rotation_point='center',
                      edgecolor=color,
                      facecolor='none',
                      linewidth=2
                  )
                  # Move rectangle to the right position
                  transform = plt.matplotlib.transforms.Affine2D().translate(x, y)
                  rect.set_transform(transform + ax.transData)
                  ax.add_patch(rect)

              else:
                  color = cmap_2(norm(t))
                  circle = patches.Circle((x, y), 0.15, edgecolor=color, facecolor='none',linewidth=2)
                  ax.add_patch(circle)
                

        # plt.show(block=False)

        # make_mode_plot(path, env)
        plt.show()

    if args.insert_transition_nodes:
        path_w_doubled_modes = []
        for i in range(len(path)):
            path_w_doubled_modes.append(path[i])

            if i + 1 < len(path) and path[i].mode != path[i + 1].mode:
                path_w_doubled_modes.append(State(path[i].q, path[i + 1].mode))

        path = path_w_doubled_modes

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
        False,
        export=False,
        pause_time=0.05,
        stop_at_end=True,
        adapt_to_max_distance=True,
    )


if __name__ == "__main__":
    main()
