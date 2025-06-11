import argparse
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import json
import os
import random
import re
import numpy as np
from pathlib import Path
# from typing import Dict, Any, Callable, Tuple, List

from multi_robot_multi_goal_planning.problems import get_env_by_name

from multi_robot_multi_goal_planning.problems.planning_env import State
from multi_robot_multi_goal_planning.problems.util import interpolate_path, path_cost
from multi_robot_multi_goal_planning.planners.shortcutting import (
    robot_mode_shortcut,
)
from run_experiment import load_experiment_config
# from multi_robot_multi_goal_planning.problems.configuration import config_dist

def get_infos_of_obstacles_and_table_2d(env):
    frames = env.C.getFrames()
    obstacles = []
    for frame in frames:
        #check if its an obstacle or the table
        name = frame.name
        if name.startswith("table"):
            size = frame.getSize()
            table_size = (size[0]/2, size[0]/2)
        if name.startswith("obs"):
            info = {}
            shape = frame.getShapeType()
            assert str(shape).startswith("ST.box"),(
                "This environment contains obstacles that aren't boxes, which is not supported by this function"
            )
            state = frame.getPosition()
            info["pos"] = list(state[:2])  # Extract [x, y]
            info["size"] = list(frame.getSize()[:2])
            obstacles.append(info)
    return obstacles, table_size

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
        q = env.get_start_pos().from_flat(q_np)

        if a["mode"] != prev_mode_ids:
            # if a["mode"] != prev_mode_ids and a["mode"] != env._terminal_task_ids:
            print(a["mode"])
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
    parser.add_argument("path_filename", nargs="?", default="", help="filepath")
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Interpolate the path that is loaded. (default: False)",
    )
    parser.add_argument(
        "--shortcut",
        action="store_true",
        help="Shortcut the path. (default: False)",
    )
    parser.add_argument(
        "--insert_transition_nodes",
        action="store_true",
        help="Insert transition nodes into the path. (default: False)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the path. (default: False)",
    )
    parser.add_argument(
        "--outline",
        action="store_true",
        help="Add outline to the visualization. (default: False)",
    )

    args = parser.parse_args()

    folder_path = re.match(r'(.*?/out/[^/]+)', args.path_filename).group(1)
    potential_config_path = os.path.join(folder_path, 'config.json')
    if Path(potential_config_path).exists():
        config = load_experiment_config(potential_config_path)
        seed = config["seed"] 
        env_name = config["environment"]
        cost_reduction = config["cost_reduction"]
        cost_metric = config["per_agent_cost"]
    else:
        raise FileNotFoundError(f"Could not find the configuration file at: '{potential_config_path}'. Please check the provided path {args.path_filename} and try again.")
    
    np.random.seed(seed)
    random.seed(seed)

    path_data = load_path(args.path_filename)
    env = get_env_by_name(env_name)
    env.cost_reduction = cost_reduction
    env.cost_metric = cost_metric

    path = convert_to_path(env, path_data)

    cost = path_cost(path, env.batch_config_cost)
    print("cost", cost)

    if args.interpolate:
        path = interpolate_path(path, 0.01)

    if args.plot:
        obstacles, table_size = get_infos_of_obstacles_and_table_2d(env)
        # obstacles = [
        #     {"pos": [0.0, -0.5], "size": [1.4, 1.2]},
        #     {"pos": [0.0, 1.2], "size": [1.4, 1.5]},
        # ]

        fig, ax = plt.subplots()
        ax.set_xlim(1*table_size[0], -1*table_size[0])
        ax.set_ylim(1*table_size[1], -1.0*table_size[1])
        ax.set_aspect("equal")
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if args.outline:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(10) 
                spine.set_edgecolor("black")
        
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
        num_steps = len(path)

        cmap_1 = cm.get_cmap("cool", num_steps)
        cmap_2 = cm.get_cmap("summer", num_steps)
        cmap_3 = cm.get_cmap("plasma", num_steps)
        cmap_4 = cm.get_cmap("viridis", num_steps)
        cmap_5 = cm.get_cmap("cividis", num_steps)

        cmap_list = [cmap_1, cmap_2, cmap_3, cmap_4, cmap_5]
        norm = mcolors.Normalize(vmin=0, vmax=len(path) - 1)
        available_shapes = ["rectangle", "circle", "triangle", "diamond", "hexagon"]
        for i in range(len(env.robots)):
            x_vals = [pt.q[i][0] for pt in path]
            y_vals = [pt.q[i][1] for pt in path]
            theta_vals = [pt.q[i][2] for pt in path]
            shape = available_shapes[i]
            color_map = cmap_list[i]

            for t, (x, y, theta) in enumerate(zip(x_vals, y_vals, theta_vals)):
                if shape == "rectangle":
                    color = color_map(norm(t))
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
                elif shape == "triangle":
                    color = color_map(norm(t))
                    size = 0.15  # side length of the equilateral triangle

                    # Height of an equilateral triangle with side length `size`
                    height = np.sqrt(3) / 2 * size

                    # Define triangle vertices (centered at origin, pointing "up")
                    triangle_pts = np.array([
                        [0, 2/3 * height],               # Top vertex
                        [-size / 2, -1/3 * height],      # Bottom left
                        [size / 2, -1/3 * height]        # Bottom right
                    ])

                    # Create rotation matrix
                    c, s = np.cos(theta), np.sin(theta)
                    rotation_matrix = np.array([[c, -s], [s, c]])

                    # Rotate and translate triangle
                    rotated_pts = triangle_pts @ rotation_matrix.T + np.array([x, y])

                    # Draw triangle
                    triangle = patches.Polygon(
                        rotated_pts,
                        edgecolor=color,
                        facecolor='none',
                        linewidth=2
                    )
                    ax.add_patch(triangle)
                elif shape == "diamond":
                    color = color_map(norm(t))
                    width, height = 0.09, 0.2  # Match rectangle footprint

                    diamond_pts = np.array([
                        [0, height / 2],             # Top
                        [width / 2, 0],              # Right
                        [0, -height / 2],            # Bottom
                        [-width / 2, 0],             # Left
                    ])

                    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    rotated = diamond_pts @ rot.T + np.array([x, y])

                    diamond = patches.Polygon(
                        rotated,
                        edgecolor=color,
                        facecolor='none',
                        linewidth=2
                    )
                    ax.add_patch(diamond)
                elif shape == "hexagon":
                    color = color_map(norm(t))
                    radius = 0.1  # Approximate radius to visually match the rectangle

                    hexagon_pts = []
                    for i in range(6):
                        angle = np.pi / 3 * i
                        px = radius * np.cos(angle)
                        py = radius * np.sin(angle)
                        hexagon_pts.append([px, py])
                    hexagon_pts = np.array(hexagon_pts)

                    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    rotated = hexagon_pts @ rot.T + np.array([x, y])

                    hexagon = patches.Polygon(
                        rotated,
                        edgecolor=color,
                        facecolor='none',
                        linewidth=2
                    )
                    ax.add_patch(hexagon)
                else:
                    color = color_map(norm(t))
                    circle = patches.Circle((x, y), 0.15, edgecolor=color, facecolor='none',linewidth=2)
                    ax.add_patch(circle)
                

        # plt.show(block=False)

        # make_mode_plot(path, env)
        dir_out = os.path.join(Path(args.path_filename).parent.parent, 'ColoredPath')
        os.makedirs(dir_out, exist_ok=True)
        next_file_number = max(
            (int(file.split('.')[0]) for file in os.listdir(dir_out)
            if file.endswith('.png') and file.split('.')[0].isdigit()),
            default=-1
        ) + 1
        output_path = os.path.join(dir_out, f"{next_file_number:04d}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)


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
