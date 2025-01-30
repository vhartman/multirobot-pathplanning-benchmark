import argparse
from matplotlib import pyplot as plt

import time
import datetime
import json
import os

import numpy as np
import random

from typing import List

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

from examples.make_plots import make_cost_plots

# np.random.seed(100)


def load_experiment_config(filepath: str):
    with open(filepath) as f:
        config = json.load(f)

    # TODO: sanity checks

    # config["planners"] = {}
    # config["planners"]["name"] = []
    # config["planners"]["type"] = []
    # config["planners"]["options"] = {}

    # config["environment_name"] = []

    return config


def run_single_planner(env, planner):
    path, data = planner(env)
    return data


def export_planner_data(planner_folder, run_id, planner_data):
    # we expect data from multiple runs
    #
    # resulting folderstructure:
    # - experiment_name
    # | - config.txt
    # | - planner_name
    #   | - config.txt
    #   | - timestamps.txt
    #   | - costs.txt
    #   | - paths
    #     | - ...

    run_folder = f"{planner_folder}{run_id}/"

    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)

    # write path to file
    paths = planner_data["paths"]
    for i, path in enumerate(paths):
        # export path
        file_path = f"{run_folder}path_{i}.json"
        with open(file_path, "w") as f:
            json.dump([state.to_dict() for state in path], f)

    # write all costs with their timestamps to file
    with open(planner_folder + "timestamps.txt", "ab") as f:
        timestamps = planner_data["times"]

        np.savetxt(f, timestamps, delimiter=",", newline=",")
        f.write(b"\n")

    with open(planner_folder + "costs.txt", "ab") as f:
        costs = planner_data["costs"]

        np.savetxt(f, costs, delimiter=",", newline=",")
        f.write(b"\n")


def export_config(path, config):
    with open(path + "config.json", "w") as f:
        json.dump(config, f)


def setup_planner(planner_config, optimize=True):
    name = planner_config["name"]

    if planner_config["type"] == "prm":

        def planner(env):
            options = planner_config["options"]
            return joint_prm_planner(
                env,
                optimize=optimize,
                max_iter=20000,
                distance_metric=options["distance_function"],
                try_sampling_around_path=options["sample_near_path"],
                use_k_nearest=options["connection_strategy"] == "k_nearest",
                try_informed_sampling=options["informed_sampling"],
                try_informed_transitions=options["informed_transition_sampling"],
                uniform_batch_size=options["batch_size"],
                uniform_transition_batch_size=options["transition_batch_size"],
                informed_batch_size=options["informed_batch_size"],
                informed_transition_batch_size=options[
                    "informed_transition_batch_size"
                ],
                path_batch_size=options["path_batch_size"],
                locally_informed_sampling=options["locally_informed_sampling"],
            )
    elif planner_config["type"] == "rrt":
        pass
    else:
        raise ValueError(f"Planner type {planner_config['type']} not implemented")

    return name, planner


def setup_env(env_config):
    pass


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("filepath", nargs="?", default="default", help="filepath")

    args = parser.parse_args()

    config = load_experiment_config(args.filepath)

    env = get_env_by_name(config["environment"])
    env.cost_reduction = config["cost_reduction"]
    env.cost_metric = config["per_agent_cost"]

    if False:
        env.show()

    planners = []
    for planner_config in config["planners"]:
        planners.append(setup_planner(planner_config, config["optimize"]))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # convention: alsways use "/" as trailing character
    experiment_folder = (
        f"./out/{timestamp}_{config['experiment_name']}_{config['environment']}/"
    )

    if not os.path.isdir(experiment_folder):
        os.makedirs(experiment_folder)

    export_config(experiment_folder, config)

    seed = config["seed"]

    all_experiment_data = {}
    for planner_name, _ in planners:
        all_experiment_data[planner_name] = []

    for run_id in range(config["num_runs"]):
        for planner_name, planner in planners:
            np.random.seed(seed + run_id)
            random.seed(seed + run_id)

            res = run_single_planner(env, planner)

            planner_folder = experiment_folder + f"{planner_name}/"
            if not os.path.isdir(planner_folder):
                os.makedirs(planner_folder)

            all_experiment_data[planner_name].append(res)

            # export planner data
            export_planner_data(planner_folder, run_id, res)

    make_cost_plots(all_experiment_data, config)

    plt.show()


if __name__ == "__main__":
    main()
