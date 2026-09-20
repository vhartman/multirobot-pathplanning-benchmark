import argparse
from matplotlib import pyplot as plt

import json
import os
import pathlib

import numpy as np

from typing import List, Dict, Optional, Any

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_confidence_intervals import computeConfidenceInterval
from make_plots_stochastic_skill import apply_style, save_legend, FLAT


def load_data_from_folder(
    folder: str, load_paths: int = 0
) -> Dict[str, List[Any]]:
    all_subfolders = [
        name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))
    ]

    planner_names = [f for f in all_subfolders if "plots" not in f]

    all_experiment_data = {}


    for planner_name in planner_names:
        print(f"Loading data for {planner_name}")
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
            print(f"Did not find timestamps.txt at {subfolder_path}")
            all_experiment_data[planner_name] = []
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
                if os.path.isfile(os.path.join(run_subfolder, f)) and f.startswith("path_") and f.endswith(".json")
            ]

            path_nums = [int(f[5:-5]) for f in onlyfiles]

            sorted_files = [x for _, x in sorted(zip(path_nums, onlyfiles))]

            if load_paths:
                paths = []
                for j, file in enumerate(sorted_files):
                    if j >= load_paths:
                        break

                    with open(run_subfolder + file) as f:
                        path_data = json.load(f)
                        paths.append(path_data)

                run_data["paths"] = paths

            try:
                run_data["costs"] = costs[i]
                run_data["times"] = timestamps[i]
            except Exception:
                print("Exception saving data")
                continue

            planner_data.append(run_data)

        all_experiment_data[planner_name] = planner_data


    return all_experiment_data


def load_config_from_folder(filepath: str) -> Dict:
    with open(filepath + "config.json") as f:
        config = json.load(f)


    return config


report_colors = {
    "rrt": "#A01CBB",
    "rrt_ablation": "#EE99CC",
    "birrt": "black",
    "birrt_ablation": (158 / 255.0, 154 / 255.0, 161 / 255.0),
    "eit": "#009E1A",
    "eit_ablation": "#90E93D",
    "rheit": (0.537, 0.0, 0.267),
    "ait": "#00C3FF",
    "ait_ablation": "#306DDF",
    "rhait": (132 / 255.0, 0 / 255.0, 255 / 255.0),
    "prm": "#E21616",
    "prm_ablation": "#FF6600",
}

planner_name_to_color = {
    "prioritized": "#FFD61F",
    "rrt_skills": "#A01CBB",
    "rrg_skills": "#1F77B4",
    "rrt_conservative": "#ff7f0e",
    "rrg_reactive": "#2ca02c",

    "prio": "#FFD61F",
    "rrtstar": report_colors["rrt"],
    "rrtstar_global_sampling": report_colors["rrt_ablation"],
    "rrtstar_no_shortcutting": report_colors["rrt_ablation"],
    "rrtstar uniform": report_colors["rrt_ablation"],
    "rrtstar without": report_colors["rrt_ablation"],
    "birrtstar": report_colors["birrt"],
    "birrt": report_colors["birrt"],
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
    "informed_prm_k_nearest": report_colors["prm_ablation"],
    "prm_no_shortcutting": report_colors["prm_ablation"],
    "prm same": report_colors["prm_ablation"],
    "prm uniform": report_colors["prm_ablation"],
    "prm without": report_colors["prm_ablation"],
    "globally_informed_prm": report_colors["prm_ablation"],
}

planner_name_to_style = {
    "rrtstar without": "--",
    "rrtstar_global_sampling": ":",
    "rrtstar_no_shortcutting": "--",
    "rrtstar uniform": "--",
    "birrtstar_global_sampling": "--",
    "birrtstar_no_shortcutting": "--",
    "birrtstar uniform": "--",
    "birrtstar without": "--",
    "globally_informed_prm": "--",
    "prm_no_shortcutting": "--",
    "prm same": "--",
    "prm uniform": "--",
    "prm without": "--",
    "aitstar same": "--",
    "aitstar_no_shortcutting": ":",
    "aitstar uniform": "--",
    "aitstar without": "--",
    "aitstar_global_sampling": ":",
    "eitstar_no_shortcutting": "--",
    "eitstar same": "--",
    "eitstar uniform": "--",
    "eitstar without": "--",
    "eitstar_global_sampling": "--",
}

skill_benchmark_colors = {
    "rrtstar_old": "#A01CBB",
    "birrtstar_old": "black",
    "pp": "#FFD61F",
    "prioritized": "#FFD61F",
    "rrt_skills": "#A01CBB",
    "rrg_skills": "#1F77B4",
    "rrt_conservative": "#ff7f0e",
    "rrg_reactive": "#2ca02c",

    "prm": "#E21616",
    "prm_incremental_frozen_lane": "#E21616",
    "prm_incremental_stepwise_lane": "#FF6600",
    "prm_single_frozen_lane": "#8C0000",
    "prm_multi_frozen_lane": "#FF9E80",
    "prm_phase3_outside": "#E21616",
    "prm_phase3_inside": "#FF6600",
    "prm_phase1": "#8C0000",
    "prm_phase2": "#FF9E80",
    "pp_naive": "#FFD61F",
    "pp_conservative": "#B8860B",
    "reactive_shortcut": "#0050C8",
    "reactive_position_time": "#0050C8",
    "rrg_reactive_position": "#8C6BB1",
    "rrg_reactive_nowait": "#B15928",
    "rrg_reactive_f2": "#6A9E00",
    "rrg_reactive_f10": "#00857A",
}

rrt_skills_colors = [
    "#90E93D",
    "#1F77B4",
    "#17BECF",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
]

rrt_skills_styles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]


def get_planner_config(name, config):
    return next((p for p in config.get("planners", []) if p.get("name") == name), None)


def rrt_skills_index(name, config):
    names = [
        p["name"]
        for p in config.get("planners", [])
        if p.get("type", "") == "rrt_skills"
    ]
    return names.index(name) if name in names else None


def get_planner_color(name, config):
    if name in skill_benchmark_colors:
        return skill_benchmark_colors[name]

    idx = rrt_skills_index(name, config)
    if idx is not None:
        return rrt_skills_colors[idx % len(rrt_skills_colors)]

    p_cfg = get_planner_config(name, config)
    if p_cfg:
        p_type = p_cfg.get("type", "")
        if p_type in ["prioritized", "prio"]:
            return skill_benchmark_colors["pp"]
        if p_type == "prm":
            return skill_benchmark_colors["prm"]

    if name not in planner_name_to_color:
        planner_name_to_color[name] = np.random.rand(3,)
    return planner_name_to_color[name]


def get_ordered_planner_names(planner_names, config):
    planner_name_set = set(planner_names)
    ordered_names = [
        planner["name"]
        for planner in config.get("planners", [])
        if planner.get("name") in planner_name_set
    ]
    ordered_names.extend(planner_names)
    return list(dict.fromkeys(ordered_names))


def get_planner_style(name, config):
    idx = rrt_skills_index(name, config)
    if idx is not None:
        return rrt_skills_styles[idx % len(rrt_skills_styles)]
    return planner_name_to_style.get(name, "-")


def get_planner_label(name, config):
    lower_name = name.lower()
    if "prm" in lower_name:
        return "PRM*"
    if lower_name in ("pp", "prioritized", "prio") or lower_name.startswith("pp_"):
        return "PP"
    if "conservative" in lower_name:
        return "Conservative"
    if "reactive" in lower_name:
        return "Reactive"
    if "rrt" in lower_name:
        return "RRT*"
    if "rrg" in lower_name:
        return "RRG"
    return name


def interpolate_costs(new_timesteps, times, costs):
    new_timesteps = np.asarray(new_timesteps)
    times = np.asarray(times)
    costs = np.asarray(costs)

    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be monotonically increasing")

    indices = np.searchsorted(times, new_timesteps, side="right") - 1

    result = np.empty_like(new_timesteps, dtype=float)

    before_start = indices < 0
    result[before_start] = np.inf

    after_end = indices >= len(times) - 1
    result[after_end] = costs[-1]

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
    final_max_time: Optional[float] = None,
    logscale: bool = True,
    yticks: List[int] = [],
):
    plt.figure("Cost plot", figsize=(FLAT[0] * 2 / 3, FLAT[1]))

    max_time = 0
    planner_names = get_ordered_planner_names(all_experiment_data.keys(), config)
    for planner_name in planner_names:
        results = all_experiment_data[planner_name]
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

        color = get_planner_color(planner_name, config)

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
            label=get_planner_label(planner_name, config),
        )

    time_discretization = 1e-2
    if final_max_time is not None:
        max_time = final_max_time
    interpolated_solution_times = np.arange(0, max_time, time_discretization)


    max_non_inf_cost = 0
    min_non_inf_cost = np.inf

    for planner_name in planner_names:
        results = all_experiment_data[planner_name]
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
        finite_ub = ub_solution_cost[np.isfinite(ub_solution_cost)]
        if len(finite_ub) > 0:
            max_non_inf_cost = max(max_non_inf_cost, np.max(finite_ub))

        if len(min_solution_cost[np.isfinite(min_solution_cost)]) > 0:
            min_non_inf_cost = min(
                min_non_inf_cost,
                np.min(min_solution_cost[np.isfinite(min_solution_cost)]),
            )

        ub_solution_cost[~np.isfinite(ub_solution_cost)] = 1e6

        color = get_planner_color(planner_name, config)
        ls = get_planner_style(planner_name, config)

        if is_initial_solution_only:
            continue

        plt.semilogx(
            interpolated_solution_times[
                interpolated_solution_times < max_planner_solution_time
            ],
            median_solution_cost[
                interpolated_solution_times < max_planner_solution_time
            ],
            color=color,
            ls=ls,
        )
        plt.fill_between(
            interpolated_solution_times[
                interpolated_solution_times < max_planner_solution_time
            ],
            lb_solution_cost[interpolated_solution_times < max_planner_solution_time],
            ub_solution_cost[interpolated_solution_times < max_planner_solution_time],
            alpha=0.3,
            color=color,
        )

    if baseline_cost is not None:
        plt.axhline(y=baseline_cost, color="tab:blue", linestyle="--")

        min_non_inf_cost = min(min_non_inf_cost, baseline_cost)

    plt.grid(which="both", axis="both", ls="--")

    if "cost_reduction" in config:
        plt.ylabel(f"Cost ({config['cost_reduction']})", fontsize=18)
    else:
        plt.ylabel("Cost", fontsize=18)

    plt.xlabel("Computation Time [s]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    if logscale:
        plt.yscale("log")

    if len(yticks) > 0:
        ticks = np.array(yticks, dtype=float)
        if logscale:
            ticks = ticks[ticks > 0]

        if ticks.size > 0:
            plt.yticks(ticks, [int(t) for t in ticks])

    if np.isfinite(min_non_inf_cost) and np.isfinite(max_non_inf_cost) and max_non_inf_cost > 0:
        plt.ylim([0.9 * min_non_inf_cost, 1.1 * max_non_inf_cost])

    if logscale and len(yticks) == 0:
        ax = plt.gca()
        lo, hi = ax.get_ylim()
        if lo > 0 and hi > 0:
            from matplotlib.ticker import MaxNLocator
            ticks = [t for t in MaxNLocator(nbins=3).tick_values(lo, hi) if lo <= t <= hi]
            if not ticks:
                ticks = [lo, (lo+hi)/2.0, hi]
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(plt.ScalarFormatter())
            ax.yaxis.set_minor_formatter(plt.NullFormatter())

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

        if add_legend:
            handles, labels = plt.gca().get_legend_handles_labels()
            save_legend(handles, labels, f"{foldername}plots/legend_deterministic.{format}")


def make_success_plot(
    all_experiment_data: Dict[str, Any],
    config: Dict,
    save: bool = False,
    foldername: Optional[str] = None,
    save_as_png: bool = False,
    add_legend: bool = True,
    final_max_time: Optional[float] = None,
):
    time_discretization = 1e-2
    if final_max_time is None:
        interpolated_solution_times = np.arange(
            0, config["max_planning_time"], time_discretization
        )
    else:
        interpolated_solution_times = np.arange(0, final_max_time, time_discretization)

    plt.figure("Success plot")

    first_solution_found = 1e8
    len_results = config["num_runs"]
    planner_names = get_ordered_planner_names(all_experiment_data.keys(), config)

    for planner_name in planner_names:
        results = all_experiment_data[planner_name]
        if len(results) == 0:
            print(f"{planner_name} solved 0/{len_results} runs -- drawn at 0% success")
            plt.semilogx(
                interpolated_solution_times,
                np.zeros_like(interpolated_solution_times),
                color=get_planner_color(planner_name, config),
                label=f"{get_planner_label(planner_name, config)} (0/{len_results})",
                drawstyle="steps-post",
                ls=get_planner_style(planner_name, config),
            )
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

        color = get_planner_color(planner_name, config)
        ls = get_planner_style(planner_name, config)
        plt.semilogx(
            interpolated_solution_times,
            percentage_solution_found,
            color=color,
            label=f"{get_planner_label(planner_name, config)} ({len(results)}/{len_results})",
            drawstyle="steps-post",
            ls=ls,
        )

    if first_solution_found < len(interpolated_solution_times):
        plt.xlim(
            [
                interpolated_solution_times[first_solution_found] * 0.9,
                interpolated_solution_times[-1],
            ]
        )

    plt.grid(which="both", axis="both", ls="--")
    plt.ylabel(r"Success [\%]")
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

        if add_legend:
            handles, labels = plt.gca().get_legend_handles_labels()
            save_legend(handles, labels, f"{foldername}plots/legend_deterministic.{format}")


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("foldername", nargs="?", default="default", help="filepath")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process all subfolders containing a config.json",
    )
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
        help="Also save a separate, framed legend_deterministic.<fmt> next to the plots "
             "(default: False). The plots themselves never carry an in-panel legend.",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Display the resulting plots at the end. (default: False)",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use a linear y axis instead of the default log scale. An anytime curve opens "
             "5-10x above where it converges, and on a linear axis that drop flattens the part "
             "that decides the comparison.",
    )
    parser.add_argument(
        "--yticks",
        default="",
        type=str,
        help="Y ticks. (default: Lets matplotlib do it automatically.)",
    )
    parser.add_argument("--baseline_cost", type=float, default=None, help="Baseline")
    parser.add_argument(
        "--limited_max_time", type=float, default=None, help="Max time for the plot"
    )
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--cost_only",
        action="store_true",
        help="Only generate the cost plot",
    )
    plot_group.add_argument(
        "--success_only",
        action="store_true",
        help="Only generate the success plot",
    )
    args = parser.parse_args()

    make_cost = not args.success_only
    make_success = not args.cost_only

    apply_style(args.use_paper_style)

    yticks = []
    if len(args.yticks) > 0:
        yticks = list(map(int, args.yticks.split(",")))

    foldername = args.foldername
    if foldername[-1] != "/":
        foldername += "/"

    if args.recursive:
        root = foldername

        subdirs = [
            os.path.join(root, d, "")
            for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
            and os.path.exists(os.path.join(root, d, "config.json"))
        ]

        if len(subdirs) == 0:
            print("No valid experiment subfolders found.")
            return

        for subfolder in subdirs:
            try:
                print(f"\n=== Processing {subfolder} ===")
                all_experiment_data = load_data_from_folder(subfolder)
                config = load_config_from_folder(subfolder)

                if make_cost:
                    make_cost_plots(
                        all_experiment_data,
                        config,
                        args.save,
                        subfolder,
                        save_as_png=args.png,
                        add_legend=args.legend,
                        baseline_cost=args.baseline_cost,
                        final_max_time=args.limited_max_time,
                        logscale=not args.linear,
                        yticks=yticks,
                    )
                    plt.close()

                if make_success:
                    make_success_plot(
                        all_experiment_data,
                        config,
                        args.save,
                        subfolder,
                        save_as_png=args.png,
                        add_legend=args.legend,
                        final_max_time=args.limited_max_time,
                    )
                    plt.close()

            except Exception as e:
                print(f"failed plotting {subfolder}: {e}")

    else:
        all_experiment_data = load_data_from_folder(foldername)
        config = load_config_from_folder(foldername)

        if make_cost:
            make_cost_plots(
                all_experiment_data,
                config,
                args.save,
                foldername,
                save_as_png=args.png,
                add_legend=args.legend,
                baseline_cost=args.baseline_cost,
                final_max_time=args.limited_max_time,
                logscale=not args.linear,
                yticks=yticks,
            )
        if make_success:
            make_success_plot(
                all_experiment_data,
                config,
                args.save,
                foldername,
                save_as_png=args.png,
                add_legend=args.legend,
                final_max_time=args.limited_max_time,
            )

        if not args.no_display:
            plt.show()


if __name__ == "__main__":
    main()
