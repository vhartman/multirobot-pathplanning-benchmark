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

from multi_robot_multi_goal_planning.problems.planning_env import State
from compute_confidence_intervals import computeConfidenceInterval


def load_data_from_folder(folder: str, load_paths: bool = False) -> Dict[str, List[Any]]:
    all_subfolders = [
        name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))
    ]

    planner_names = [f for f in all_subfolders if "plots" not in f]

    all_experiment_data = {}

    loading_start_time = time.time()

    for planner_name in planner_names:
        print(f"Loading data for {planner_name}")
        subfolder_path = folder + planner_name + "/"

        timestamps = []
        # print("Loading timestamps")
        try:
            with open(subfolder_path + "timestamps.txt") as file:
                for line in file:
                    timestamps_this_run = []
                    if len(line) <= 1:
                        continue

                    for num in line.rstrip()[:-1].split(","):
                        timestamps_this_run.append(float(num))

                    timestamps.append(timestamps_this_run)
                # for line in file:
                #     line = line.strip()
                #     if not line:
                #         continue
                #     # Use map for type conversion
                #     timestamps_this_run = list(map(float, line.rstrip(',').split(",")))
                #     timestamps.append(timestamps_this_run)

        except FileNotFoundError:
            continue

        costs = []
        # print("Loading costs")
        with open(subfolder_path + "costs.txt") as file:
            for line in file:
                costs_this_run = []
                if len(line) <= 1:
                    continue

                for num in line.rstrip()[:-1].split(","):
                    costs_this_run.append(float(num))

                costs.append(costs_this_run)
            # for line in file:
            #     line = line.strip()
            #     if not line:
            #         continue
            #     # Use map for type conversion
            #     costs_this_run = list(map(float, line.rstrip(',').split(",")))
            #     costs.append(costs_this_run)

            #     costs.append(costs_this_run)

        runs = [
            int(name)
            for name in os.listdir(subfolder_path)
            if os.path.isdir(os.path.join(subfolder_path, name))
        ]

        runs.sort()

        planner_data = []
    # config["planners"] = {}
    # config["planners"]["name"] = []
    # config["planners"]["type"] = []
    # config["planners"]["options"] = {}

    # config["environment_name"] = []

        # print("Loading paths")
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

            if load_paths:
                paths = []
                for file in sorted_files:
                    with open(run_subfolder + file) as f:
                        path_data = json.load(f)
                        paths.append(path_data)

                run_data["paths"] = paths

            try:
                run_data["costs"] = costs[i]
                run_data["times"] = timestamps[i]
            except Exception:
                continue

            planner_data.append(run_data)

        all_experiment_data[planner_name] = planner_data

    # loading_end_time = time.time()
    # print(f"Loading took {loading_end_time - loading_start_time}s")

    return all_experiment_data


def load_config_from_folder(filepath: str) -> Dict:
    with open(filepath + "config.json") as f:
        config = json.load(f)

    # TODO: sanity checks

    return config


report_colors = {
    "rrt":            "#A01CBB",
    "rrt_ablation":   "#EE99CC",
    "birrt":          "black",
    "birrt_ablation": (158 / 255.0, 154 / 255.0, 161 / 255.0),
    "eit":            "#009E1A",
    "eit_ablation":   "#90E93D",
    "rheit":          (0.537, 0.0, 0.267), 
    "ait":            "#00C3FF",
    "ait_ablation":   "#306DDF",
    "rhait":          (132 / 255.0, 0 / 255.0, 255 / 255.0),
    "prm":            "#E21616",
    "prm_ablation":   "#FF6600",
}

# TODO: move this to config?
planner_name_to_color = {
    "prioritized": "#FFD61F",
    "rrtstar": report_colors["rrt"],
    "rrtstar_global_sampling": report_colors["rrt_ablation"],
    "rrtstar_no_shortcutting": report_colors["rrt_ablation"],
    "rrtstar uniform": report_colors["rrt_ablation"],
    "rrtstar without": report_colors["rrt_ablation"],

    "birrtstar": report_colors["birrt"],
    "birrtstar_global_sampling": report_colors["birrt_ablation"],
    "birrtstar_no_shortcutting": report_colors["birrt_ablation"],
    "birrtstar uniform": report_colors["birrt_ablation"],
    "birrtstar without": report_colors["birrt_ablation"], 

    "eitstar": report_colors["eit"],
    "eitstar same": report_colors["eit_ablation"],
    "long_horizon eitstar": report_colors["rheit"],
    "eitstar uniform": report_colors["eit_ablation"],
    "eitstar_no_shortcutting": report_colors["eit_ablation"],
    "eitstar without": report_colors["eit_ablation"],
    "eitstar_global_sampling": report_colors["eit_ablation"],

    "aitstar": report_colors["ait"],
    "aitstar same": report_colors["ait_ablation"],
    "long_horizon aitstar": report_colors["rhait"],
    "aitstar uniform": report_colors["ait_ablation"],
    "aitstar_no_shortcutting": report_colors["ait_ablation"],
    "aitstar_global_sampling": report_colors["ait_ablation"],
    "aitstar without": report_colors["ait_ablation"],

    "prm": report_colors["prm"],
    "informed_prm_k_nearest":  report_colors["prm_ablation"],
    "prm_no_shortcutting": report_colors["prm_ablation"],
    "prm same": report_colors["prm_ablation"],
    "prm uniform": report_colors["prm_ablation"],
    "prm without": report_colors["prm_ablation"],    
    "globally_informed_prm": report_colors["prm"],
}

planner_name_to_style = {
    "rrtstar without": "--",
    "rrtstar_global_sampling": ":",
    "rrtstar_no_shortcutting": ":",
    "rrtstar uniform": "--",

    "birrtstar_global_sampling": ":",
    "birrtstar_no_shortcutting": ":",
    "birrtstar uniform":"--",
    "birrtstar without":"--",

    "globally_informed_prm": "--",
    "prm_no_shortcutting": ":",
    "prm same":"--",
    "prm uniform":"--",
    "prm without":"--",

    "aitstar same": "--",
    "aitstar_no_shortcutting": ":",
    "aitstar uniform": "--",
    "aitstar without": "--",
    "aitstar_global_sampling": ":",

    "eitstar_no_shortcutting": ":",
    "eitstar same": "--",
    "eitstar uniform": "--",
    "eitstar without": "--",
    "eitstar_global_sampling": ":",
}


def interpolate_costs(new_timesteps, times, costs):
    # if not times or not costs or len(times) != len(costs) or not new_timesteps:
    #     return []
    new_timesteps = np.asarray(new_timesteps)
    times = np.asarray(times)
    costs = np.asarray(costs)

    # Verify times are monotonically increasing
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be monotonically increasing")

    # Find insertion points for all new_timesteps at once
    indices = np.searchsorted(times, new_timesteps, side="right") - 1

    # Create the output array
    result = np.empty_like(new_timesteps, dtype=float)

    # Handle cases before first time
    before_start = indices < 0
    result[before_start] = np.inf

    # Handle cases after or at last time
    after_end = indices >= len(times) - 1
    result[after_end] = costs[-1]

    # Handle cases within the time range
    within_range = ~(before_start | after_end)
    result[within_range] = costs[indices[within_range]]

    return result


def make_cost_plots(
    all_experiment_data: Dict,
    config: Dict,
    save: bool = False,
    foldername: Optional[str] = None,
    save_as_png: bool = False,
    add_legend: bool = True,
    baseline_cost=None,
    add_info: bool = False,
    final_max_time: Optional[float] = None,
    logscale: bool = False
):
    plt.figure("Cost plot")

    max_time = 0
    colors = {}
    for planner_name, results in all_experiment_data.items():
        all_initial_solution_times = []
        all_initial_solution_costs = []

        if len(results) == 0:
            print(f"Skipping {planner_name} since no solutions are available")
            continue

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

        lb_index, ub_index, _ = computeConfidenceInterval(len(results), 0.95)

        sorted_solution_times = np.sort(all_initial_solution_times)

        lb_initial_solution_time = sorted_solution_times[lb_index]
        ub_initial_solution_time = sorted_solution_times[ub_index - 1]

        sorted_solution_costs = np.sort(all_initial_solution_costs)

        lb_initial_solution_cost = sorted_solution_costs[lb_index]
        ub_initial_solution_cost = sorted_solution_costs[ub_index - 1]

        if planner_name in planner_name_to_color:
            color = planner_name_to_color[planner_name]
        else:
            color = np.random.rand(3,)
        colors[planner_name] = color

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
            color=color,
            capsize=5,
            capthick=2,
        )

    time_discretization = 1e-2
    if final_max_time is not None:
        max_time = final_max_time
    interpolated_solution_times = np.arange(0, max_time, time_discretization)

    # plt.figure("Cost plot")

    max_non_inf_cost = 0
    min_non_inf_cost = np.inf

    for planner_name, results in all_experiment_data.items():
        print(f"Constructing cost curve for {planner_name}")
        if len(results) == 0:
            print(f"Skipping {planner_name} since no solutions are available")
            continue
        
        all_solution_costs = []

        max_planner_solution_time = 0

        is_initial_solution_only = True

        for single_run_result in results:
            solution_times = single_run_result["times"]
            solution_costs = single_run_result["costs"]

            if len(solution_costs) > 1:
                is_initial_solution_only = False

            discretized_solution_costs = interpolate_costs(
                interpolated_solution_times, solution_times, solution_costs
            )

            all_solution_costs.append(discretized_solution_costs)

            max_planner_solution_time = max_time

        median_solution_cost = np.median(all_solution_costs, axis=0)

        lb_index, ub_index, _ = computeConfidenceInterval(len(results), 0.95)
        sorted_solution_costs = np.sort(all_solution_costs, axis=0)

        lb_solution_cost = sorted_solution_costs[lb_index, :]
        ub_solution_cost = sorted_solution_costs[ub_index - 1, :]

        min_solution_cost = np.min(all_solution_costs, axis=0)
        try:
            max_solution_cost = np.max(ub_solution_cost[np.isfinite(ub_solution_cost)])
        except:
            min_solution_cost = np.min(all_solution_costs, axis=0)
        if len(max_solution_cost[np.isfinite(max_solution_cost)]) > 0:
            max_non_inf_cost = max(
                max_non_inf_cost,
                max_solution_cost
            )

        if len(min_solution_cost[np.isfinite(min_solution_cost)]) > 0:
            min_non_inf_cost = min(
                min_non_inf_cost,
                np.min(min_solution_cost[np.isfinite(min_solution_cost)]),
            )

        ub_solution_cost[~np.isfinite(ub_solution_cost)] = 1e6

        color = colors[planner_name]

        ls = "-"
        if planner_name in planner_name_to_style:
            ls = planner_name_to_style[planner_name]
        
        if is_initial_solution_only:
            continue

        plt.semilogx(
            interpolated_solution_times[
                interpolated_solution_times < max_planner_solution_time
            ],
            median_solution_cost[
                interpolated_solution_times < max_planner_solution_time
            ],
            label=planner_name,
            color=color,
            ls = ls
        )
        plt.fill_between(
            interpolated_solution_times[
                interpolated_solution_times < max_planner_solution_time
            ],
            lb_solution_cost[interpolated_solution_times < max_planner_solution_time],
            ub_solution_cost[interpolated_solution_times < max_planner_solution_time],
            alpha=0.3,
            # lw=2,
            color=color,
        )

    if baseline_cost is not None:
        plt.axhline(y=baseline_cost, color="gray", linestyle="--")

        min_non_inf_cost = min(min_non_inf_cost, baseline_cost)

    plt.ylim([0.9 * min_non_inf_cost, 1.1 * max_non_inf_cost])

    plt.grid(which="both", axis="both", ls="--")

    plt.ylabel(f"Cost ({config['cost_reduction']})")
    plt.xlabel("Computation Time [s]")

    if logscale:
        plt.yscale("log")

    if add_legend:
        if add_info:
            legend_title = f"Environment: {config['environment']}\nNumber of runs: {config['num_runs']}"
            existing_handles, _ = plt.gca().get_legend_handles_labels()
            plt.legend(handles=existing_handles, title=legend_title, title_fontsize='medium', loc='best', alignment='left')
        else:

            plt.legend()

    if save:
        if foldername is None:
            raise ValueError("No path specified")

        pathlib.Path(f"{foldername}plots/").mkdir(parents=True, exist_ok=True)

        format = "pdf"
        if save_as_png:
            format = "png"

        scenario_name = config["environment"]

        plt.savefig(
            f"{foldername}plots/cost_plot_{scenario_name}.{format}",
            format=format,
            dpi=300,
            bbox_inches="tight",
        )


def make_success_plot(
        all_experiment_data: Dict[str, Any], 
        config: Dict,
        save: bool = False,
        foldername: Optional[str] = None,
        save_as_png: bool = False,
        add_legend: bool = True,
        add_info: bool = False,
        final_max_time: Optional[float] = None,
        ):
    time_discretization = 1e-2
    if final_max_time is None:
        interpolated_solution_times = np.arange(
            0, config["max_planning_time"], time_discretization
        )
    else:
        interpolated_solution_times = np.arange(
            0, final_max_time, time_discretization
        )

    plt.figure("Success plot")

    first_solution_found = 1e8
    len_results = config["num_runs"]

    for planner_name, results in all_experiment_data.items():
        if len(results) == 0:
            print(f"Skipping {planner_name} since no solutions are available")
            continue
        
        all_solution_costs = []

        for single_run_result in results:
            solution_times = single_run_result["times"]
            solution_costs = single_run_result["costs"]

            discretized_solution_costs = interpolate_costs(
                interpolated_solution_times, solution_times, solution_costs
            )

            all_solution_costs.append(discretized_solution_costs)

        solution_found = np.isfinite(all_solution_costs)
        percentage_solution_found = np.sum(solution_found, axis=0) / len_results

        for i in range(len(percentage_solution_found)):
            if percentage_solution_found[i] > 1e-3:
                first_solution_found = min(first_solution_found, i)
                break

        if planner_name in planner_name_to_color:
            color = planner_name_to_color[planner_name]
        else:
            color = np.random.rand(3,)

        ls = '-'
        if planner_name in planner_name_to_style:
            ls = planner_name_to_style[planner_name]
        plt.semilogx(
            interpolated_solution_times,
            percentage_solution_found,
            color=color,
            label=planner_name,
            drawstyle="steps-post",
            ls = ls
        )

    plt.xlim(
        [
            interpolated_solution_times[first_solution_found] * 0.9,
            interpolated_solution_times[-1],
        ]
    )

    # plt.legend()
    if add_legend:
        if add_info:
            legend_title = f"Environment: {config['environment']}\nNumber of runs: {config['num_runs']}"
            existing_handles, _ = plt.gca().get_legend_handles_labels()
            plt.legend(handles=existing_handles, title=legend_title, title_fontsize='medium', loc='best', alignment='left')
        else:

            plt.legend()
    plt.grid(which="both", axis="both", ls="--")  
    plt.ylabel(f"Success [%]")
    plt.xlabel("Computation Time [s]")

    if save:
        if foldername is None:
            raise ValueError("No path specified")

        pathlib.Path(f"{foldername}plots/").mkdir(parents=True, exist_ok=True)

        format = "pdf"
        if save_as_png:
            format = "png"

        scenario_name = config["environment"]

        plt.savefig(
            f"{foldername}plots/success_plot_{scenario_name}.{format}",
            format=format,
            dpi=300,
            bbox_inches="tight",
        )


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
    parser.add_argument(
        "--png",
        action="store_true",
        help="Use the paper style (default: False)",
    )

    parser.add_argument(
        "--legend",
        action="store_true",
        help="Add the legend to the plot (default: False)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Add general information to the plot legend (default: False)",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Display the resulting plots at the end. (default: False)",
    )
    parser.add_argument(
        "--logscale",
        action="store_true",
        help="Scale the y axis logarithmically. (default: False)",
    )
    parser.add_argument(
        "--baseline_cost", type=float, default=None, help="Baseline"
    )
    parser.add_argument(
        "--limited_max_time", type=float, default=None, help="Max time for the plot"
    )
    args = parser.parse_args()

    if args.use_paper_style:
        plt.style.use("./examples/paper_2.mplstyle")

    foldername = args.foldername
    if foldername[-1] != "/":
        foldername += "/"

    all_experiment_data = load_data_from_folder(foldername)
    config = load_config_from_folder(foldername)

    make_cost_plots(
        all_experiment_data,
        config,
        args.save,
        foldername,
        save_as_png=args.png,
        add_legend=args.legend,
        baseline_cost=args.baseline_cost,
        add_info = args.info,
        final_max_time=args.limited_max_time,
        logscale=args.logscale
    )
    make_success_plot(
        all_experiment_data, 
        config, 
        args.save,
        foldername,
        save_as_png=args.png,
        add_legend=args.legend,
        add_info = args.info,
        final_max_time=args.limited_max_time,
        )

    if not args.no_display:
        plt.show()


if __name__ == "__main__":
    main()
