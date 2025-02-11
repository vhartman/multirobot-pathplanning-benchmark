import argparse
from matplotlib import pyplot as plt

import time
import datetime
import json
import os
import pathlib

import numpy as np
import random

from typing import List, Dict, Optional, Any


def load_data_from_folder(folder: str) -> Dict[str, List[Any]]:
    all_subfolders = [
        name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))
    ]

    planner_names = [f for f in all_subfolders if "plots" not in f]

    all_experiment_data = {}

    for planner_name in planner_names:
        print(planner_name)
        subfolder_path = folder + planner_name + "/"

        timestamps = []
        try:
            with open(subfolder_path + "timestamps.txt") as file:
                for line in file:
                    timestamps_this_run = []
                    if len(line) <= 1:
                        continue

                    for num in line.rstrip()[:-1].split(","):
                        timestamps_this_run.append(float(num))

                    timestamps.append(timestamps_this_run)
        except FileNotFoundError:
            continue

        costs = []
        with open(subfolder_path + "costs.txt") as file:
            for line in file:
                costs_this_run = []
                if len(line) <= 1:
                    continue

                for num in line.rstrip()[:-1].split(","):
                    costs_this_run.append(float(num))

                costs.append(costs_this_run)

        runs = [
            int(name)
            for name in os.listdir(subfolder_path)
            if os.path.isdir(os.path.join(subfolder_path, name))
        ]

        runs.sort()

        planner_data = []

        for i, run in enumerate(runs):
            run_data = {}

            run_subfolder = subfolder_path + str(run) + "/"
            onlyfiles = [
                f
                for f in os.listdir(run_subfolder)
                if os.path.isfile(os.path.join(run_subfolder, f))
            ]

            # sort files:
            path_nums = [int(f[5:-5]) for f in onlyfiles]

            sorted_files = [x for _, x in sorted(zip(path_nums, onlyfiles))]

            paths = []
            for file in sorted_files:
                with open(run_subfolder + file) as f:
                    d = json.load(f)
                    paths.append(d)

            run_data["paths"] = paths
            run_data["costs"] = costs[i]
            run_data["times"] = timestamps[i]

            planner_data.append(run_data)

        all_experiment_data[planner_name] = planner_data

    return all_experiment_data


def load_config_from_folder(filepath: str) -> Dict:
    with open(filepath + "config.json") as f:
        config = json.load(f)

    # TODO: sanity checks

    # config["planners"] = {}
    # config["planners"]["name"] = []
    # config["planners"]["type"] = []
    # config["planners"]["options"] = {}

    # config["environment_name"] = []

    return config


# TODO: move this to config? Add dome default behaviour
planner_name_to_color = {
    "informed_prm": "tab:orange",
    "uniform_prm": "tab:blue",
    "path_prm": "tab:green",
    "joint_prm": "tab:blue",
    "informed_path_prm": "tab:red",
    "informed_prm_k_nearest": "orange",
    "uniform_prm_k_nearest": "blue",
    "path_prm_k_nearest": "green",
    "informed_path_prm_k_nearest": "red",
    "euclidean_prm": "tab:orange",
    "sum_euclidean_prm": "tab:blue",
    "max_euclidean_prm": "tab:green",
    # "informed_prm_k_nearest": "green",
    "informed_prm_radius": "tab:blue",
    "locally_informed_path_prm": "tab:purple",
    "globally_informed_path_prm": "tab:brown",
    "locally_informed_prm": "darkgreen",
    "globally_informed_prm": "magenta",
    "locally_informed_prm_shortcutting": "tab:blue",
    "globally_informed_prm_shortcutting": "tab:green",
    "locally_informed_prm_rejection": "tab:orange",
    "locally_informed_prm_shortcutting_rejection": "tab:red",
    "globally_informed_prm_shortcutting_rejection": "tab:brown",
}


def make_cost_plots(
    all_experiment_data: Dict,
    config: Dict,
    save: bool = False,
    foldername: Optional[str] = None,
):
    plt.figure("Cost plot")

    max_time = 0

    for planner_name, results in all_experiment_data.items():
        all_initial_solution_times = []
        all_initial_solution_costs = []

        for single_run_result in results:
            solution_times = single_run_result["times"]
            solution_costs = single_run_result["costs"]

            initial_solution_time = solution_times[0]
            initial_solution_cost = solution_costs[0]

            all_initial_solution_times.append(initial_solution_time)
            all_initial_solution_costs.append(initial_solution_cost)

            max_time = max(max_time, solution_times[-1])

        median_initial_solution_cost = np.median(all_initial_solution_costs)
        median_initial_solution_time = np.median(all_initial_solution_times)

        lb_initial_solution_time = np.quantile(all_initial_solution_times, 0.1)
        ub_initial_solution_time = np.quantile(all_initial_solution_times, 0.9)

        lb_initial_solution_cost = np.quantile(all_initial_solution_costs, 0.1)
        ub_initial_solution_cost = np.quantile(all_initial_solution_costs, 0.9)

        plt.errorbar(
            [median_initial_solution_time],
            [median_initial_solution_cost],
            xerr=np.array(
                [
                    median_initial_solution_time - lb_initial_solution_time,
                    ub_initial_solution_time - median_initial_solution_time,
                ]
            )[:, None],
            yerr=np.array(
                [
                    median_initial_solution_cost - lb_initial_solution_cost,
                    ub_initial_solution_cost - median_initial_solution_cost,
                ]
            )[:, None],
            marker="o",
            color=planner_name_to_color[planner_name],
            capsize=5,
            # capthick=5,
        )

    time_discretization = 1e-2
    interpolated_solution_times = np.arange(0, max_time, time_discretization)

    def interpolate_costs(new_timesteps, times, costs):
        # if not times or not costs or len(times) != len(costs) or not new_timesteps:
        #     return []

        if any(times[i] >= times[i + 1] for i in range(len(times) - 1)):
            raise ValueError("times must be monotonically increasing")

        interpolated_costs = []
        j = 0  # Index for original times and costs

        for new_time in new_timesteps:
            if new_time < times[0]:  # Before the first time
                interpolated_costs.append(np.inf)
            elif new_time >= times[-1]:  # After or at the last time
                interpolated_costs.append(costs[-1])
            else:  # Within the original time range
                j = 0
                while j < len(times) - 1 and new_time > times[j + 1]:
                    j += 1
                interpolated_costs.append(costs[j])

        return interpolated_costs

    # plt.figure("Cost plot")

    max_non_inf_cost = 0
    min_non_inf_cost = np.inf

    for planner_name, results in all_experiment_data.items():
        all_solution_costs = []

        max_planner_solution_time = 0

        for single_run_result in results:
            solution_times = single_run_result["times"]
            solution_costs = single_run_result["costs"]

            discretized_solution_costs = interpolate_costs(
                interpolated_solution_times, solution_times, solution_costs
            )

            all_solution_costs.append(discretized_solution_costs)

            # max_planner_solution_time = max(
            #     max_planner_solution_time, solution_times[-1]
            # )
            max_planner_solution_time = max_time

        median_solution_cost = np.median(all_solution_costs, axis=0)

        lb_solution_cost = np.quantile(all_solution_costs, 0.1, axis=0)
        ub_solution_cost = np.quantile(all_solution_costs, 0.9, axis=0)

        min_solution_cost = np.min(all_solution_costs, axis=0)
        max_solution_cost = np.max(all_solution_costs, axis=0)

        max_non_inf_cost = max(
            max_non_inf_cost, np.max(max_solution_cost[np.isfinite(max_solution_cost)])
        )
        min_non_inf_cost = min(
            min_non_inf_cost, np.min(min_solution_cost[np.isfinite(min_solution_cost)])
        )

        ub_solution_cost[~np.isfinite(ub_solution_cost)] = 1e6

        plt.semilogx(
            interpolated_solution_times[
                interpolated_solution_times < max_planner_solution_time
            ],
            median_solution_cost[
                interpolated_solution_times < max_planner_solution_time
            ],
            label=planner_name,
            color=planner_name_to_color[planner_name],
        )
        plt.fill_between(
            interpolated_solution_times[
                interpolated_solution_times < max_planner_solution_time
            ],
            lb_solution_cost[interpolated_solution_times < max_planner_solution_time],
            ub_solution_cost[interpolated_solution_times < max_planner_solution_time],
            alpha=0.5,
            lw=2,
            color=planner_name_to_color[planner_name],
        )

    plt.ylim([0.9 * min_non_inf_cost, 1.1 * max_non_inf_cost])

    # plt.grid()
    plt.grid(which="major", ls="--")

    plt.ylabel(f"Cost ({config['cost_reduction']})")
    plt.xlabel("T [s]")

    plt.legend()

    if save:
        if foldername is None:
            raise ValueError("No path specified")

        pathlib.Path(f"{foldername}plots/").mkdir(parents=True, exist_ok=True)

        plt.savefig(
            f"{foldername}plots/cost_plot.pdf",
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )


def make_success_plot(all_experiment_data: Dict[str, Any], config: Dict):
    time_discretization = 1e-2
    interpolated_solution_times = np.arange(
        0, config["max_planning_time"], time_discretization
    )

    def interpolate_costs(new_timesteps, times, costs):
        # if not times or not costs or len(times) != len(costs) or not new_timesteps:
        #     return []

        if any(times[i] >= times[i + 1] for i in range(len(times) - 1)):
            raise ValueError("times must be monotonically increasing")

        interpolated_costs = []
        j = 0  # Index for original times and costs

        for new_time in new_timesteps:
            if new_time < times[0]:  # Before the first time
                interpolated_costs.append(np.inf)
            elif new_time >= times[-1]:  # After or at the last time
                interpolated_costs.append(costs[-1])
            else:  # Within the original time range
                j = 0
                while j < len(times) - 1 and new_time > times[j + 1]:
                    j += 1
                interpolated_costs.append(costs[j])

        return interpolated_costs

    plt.figure("Success plot")

    for planner_name, results in all_experiment_data.items():
        all_solution_costs = []

        for single_run_result in results:
            solution_times = single_run_result["times"]
            solution_costs = single_run_result["costs"]

            discretized_solution_costs = interpolate_costs(
                interpolated_solution_times, solution_times, solution_costs
            )

            all_solution_costs.append(discretized_solution_costs)

        solution_found = np.isfinite(all_solution_costs)
        percentage_solution_found = np.sum(solution_found, axis=0)

        plt.semilogx(
            interpolated_solution_times,
            percentage_solution_found,
            color=planner_name_to_color[planner_name],
            label=planner_name,
            drawstyle="steps-post",
        )

    plt.legend()


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("foldername", nargs="?", default="default", help="filepath")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the generated plot (default: False)",
    )
    parser.add_argument(
        "--use_paper_style",
        action="store_true",
        help="Use the paper style (default: False)",
    )
    args = parser.parse_args()

    if args.use_paper_style:
        plt.style.use("./examples/paper_2.mplstyle")

    foldername = args.foldername
    if foldername[-1] != "/":
        foldername += "/"

    all_experiment_data = load_data_from_folder(foldername)
    config = load_config_from_folder(foldername)

    make_cost_plots(all_experiment_data, config, args.save, foldername)
    # make_success_plot(all_experiment_data, config)

    plt.show()


if __name__ == "__main__":
    main()
