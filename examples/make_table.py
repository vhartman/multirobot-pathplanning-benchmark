import numpy as np
import random

from make_plots import load_data_from_folder, load_config_from_folder
import os
from string import Template

from multi_robot_multi_goal_planning.problems import get_env_by_name

def get_initial_solution_time(results):
    all_initial_solution_times = []

    if len(results) == 0:
        return None, None, None

    for single_run_result in results:
        solution_times = single_run_result["times"]
        initial_solution_time = solution_times[0]
        all_initial_solution_times.append(initial_solution_time)

    median_initial_solution_time = np.median(all_initial_solution_times)

    return median_initial_solution_time, 0, 0


def get_initial_solution_cost(results):
    all_initial_solution_costs = []

    if len(results) == 0:
        return None, None, None

    for single_run_result in results:
        solution_costs = single_run_result["costs"]
        initial_solution_cost = solution_costs[0]

        all_initial_solution_costs.append(initial_solution_cost)

    median_initial_solution_cost = np.median(all_initial_solution_costs)

    return median_initial_solution_cost, 0, 0


def get_final_solution_cost(results):
    all_final_solution_costs = []

    if len(results) == 0:
        return None, None, None

    for single_run_result in results:
        solution_costs = single_run_result["costs"]
        final_solution_cost = solution_costs[-1]

        all_final_solution_costs.append(final_solution_cost)

    median_final_solution_cost = np.median(all_final_solution_costs)

    return median_final_solution_cost, 0, 0

def get_success_rate(results, num_experiments):
    all_succ_rates = []

    if len(results) == 0:
        return 0, 0, 0

    return  len(results) / num_experiments, 0, 0

def get_dimensionality(all_experiment_data):
    for planner_name, results in all_experiment_data.items():
        for single_run_result in results:
            if "paths" not in single_run_result:
                continue

            for path in single_run_result["paths"]:
                for pt in path:
                    return len(pt['q'])

    return 0
    # return len(env.start_pos.state())

def get_num_tasks(all_experiment_data):
    for planner_name, results in all_experiment_data.items():
        for single_run_result in results:
            if "paths" not in single_run_result:
                continue

            for path in single_run_result["paths"]:
                set_indices = set()
                for pt in path:
                    for m in pt['mode']:
                        set_indices.add(m)

                return len(set_indices)

    return 0

    # return len(env.tasks)

def get_runtime(config):
    return config["max_planning_time"]

# def load_env(config):
#     np.random.seed(config["seed"])
#     random.seed(config["seed"])

#     env = get_env_by_name(config["environment"])
    
#     return env

def aggregate_data(folders):
    env_results = {}
    for experiment in folders:
        # TODO change behavior to add none
        all_experiment_data = load_data_from_folder(experiment, 1)
        config = load_config_from_folder(experiment)

        # env = load_env(config)

        env_name = experiment.split(".")[-1][:-1]

        env_results[env_name] = {}

        env_results[env_name]["runtime"] = get_runtime(config)
        
        env_results[env_name]["num_dims"] = get_dimensionality(all_experiment_data)
        env_results[env_name]["num_tasks"] = get_num_tasks(all_experiment_data)

        for planner_name, results in all_experiment_data.items():
            t_init_median, t_init_lb, t_init_ub = get_initial_solution_time(results)
            c_init_median, c_init_lb, c_init_ub = get_initial_solution_cost(results)
            c_final_median, c_final_lb, c_final_ub = get_final_solution_cost(results)

            success_rate, _, _ = get_success_rate(results, 25)

            env_results[env_name][planner_name] = {
                "success_rate": success_rate,
                "t_init_median": t_init_median,
                "t_init_lb": t_init_lb,
                "t_init_ub": t_init_ub,
                "c_init_median": c_init_median,
                "c_init_lb": c_init_lb,
                "c_init_ub": c_init_ub,
                "c_final_median": c_final_median,
                "c_final_lb": c_final_lb,
                "c_final_ub": c_final_ub,
            }

    return env_results


def print_table(aggregated_data):
    line_layout = ["t_init_median", "c_init_median", "c_final_median"]
    # print header
    print("env", end=" ")

    planner_names = list(next(iter(aggregated_data.values())).keys())
    for key in line_layout:
        print(key, end=" ")
    print()

    for key in line_layout:
        for planner_name in planner_names:
            print(planner_name, end=" ")
    print()

    # print stuff per line
    for env_name, all_planner_data in aggregated_data.items():
        print(env_name, end=" ")
        for key in line_layout:
            all_values = [
                data[key] if data[key] is not None else float("inf")
                for planner_name, data in all_planner_data.items()
            ]
            for v in all_values:
                if v == min(all_values):
                    print(v, end=" ")
                else:
                    print(v, end=" ")

        print()


def print_latex_table(aggregated_data):
    # planner_names = list(next(iter(aggregated_data.values())).keys())
    planner_names = [
        "prio", "birrt", "ait"
    ]
    planner_key_to_name = {
        "prio": "Prio",
        "birrt": "BiRRT*",
        "ait": "AIT*"
    }

    num_planners = len(planner_names)
    
    line_layout = [
        "num_dims",
        "num_tasks",
        "runtime",
        "success_rate",
        "t_init_median",
        "c_init_median", 
        "c_final_median"
    ]
    key_to_name = {
        "num_dims": "\#Dim",
        # "num_dims": "$N_d$",
        "runtime": "$t_\\text{{max}}$ [s]",
        "num_tasks": "\#Tasks",
        # "num_tasks": "$N_t$",
        "success_rate": "\\multicolumn{3}{c}{Success Rate}",
        "t_init_median": "\\multicolumn{3}{c}{$t_\\text{{init}}$ [s]}",
        "c_init_median": "\\multicolumn{3}{c}{$c_\\text{{init}}$}",
        "c_final_median": "\\multicolumn{3}{c}{$c_{{t_\\text{{max}}}}$}",
    }

    env_info = [
        "num_dims",
        "num_tasks",
        "runtime",
    ]

    num_cols = 1 + len(planner_names) * len(line_layout)

    colspec = "l |" + "r" * len(env_info) + 4 * ("|" + len(planner_names) * "r")

    header1 = " "
    for key in line_layout:
        header1 += f"& {key_to_name[key]} "

    header2 = " "
    for key in env_info:
        header2 += " & "

    for key in line_layout:
        if key in env_info:
            continue

        for planner_name in planner_names:
            header2 += f"& \multicolumn{{1}}{{c}}{{{planner_key_to_name[planner_name]}}}"

    body = ""

    # print stuff per line
    for env_name, all_planner_data in aggregated_data.items():
        escaped_env_name = env_name.replace("_", "\\_")
        body += f"{escaped_env_name} "
        for key in line_layout:
            if key in env_info:
                body += f"& ${all_planner_data[key]}$ "

            else:
                # all_values = [
                #     data[key] if data[key] is not None else float("inf")
                #     for planner_name, data in all_planner_data.items() if planner_name in planner_names
                # ]
                all_values = []
                for planner_name in planner_names:
                    if all_planner_data[planner_name][key] is not None:
                        all_values.append(all_planner_data[planner_name][key]) 
                    else:
                        all_values.append(float("inf"))

                for i, v in enumerate(all_values):
                    if key == "success_rate":
                        best_value = max(all_values)
                    else:
                        best_value = min(all_values)

                    if v == best_value:
                        body += f"& $\\mathbf{{ {v:.2f} }}$"
                    else:
                        body += f"& ${v:.2f}$ "

        body += "\\\\ \n"

    body = body.replace("inf", "-")

    TEX_TEMPLATE = Template(r"""
\begin{tabular}{$colspec}
\toprule
$header1 \\
$header2 \\
\midrule
$body
\bottomrule
\end{tabular}
""")
    print(
        TEX_TEMPLATE.substitute(
            colspec=colspec, header1=header1, header2=header2, body=body
        )
    )


def get_subfolders_from_main_folder(folder):
    if not os.path.isdir(folder):
        return []
    try:
        with os.scandir(folder) as it:
            subdirs = [
                os.path.join(folder, entry.name) + "/" for entry in it if entry.is_dir()
            ]
    except (FileNotFoundError, PermissionError):
        return []
    subdirs.sort()
    return subdirs


def main():
    main_dir = "./out/exp_many_envs_11/"
    folders = get_subfolders_from_main_folder(main_dir)

    group_envs_by = ["dep", "unordered", "unassigned"]

    grouped_folders = {}
    grouped_folders["other"] = []
    for k in group_envs_by:
        grouped_folders[k] = []

    for folder in folders:
        found = False
        for key in group_envs_by:
            if key in folder:
                grouped_folders[key].append(folder)
                found = True
                break
        if not found:
            grouped_folders["other"].append(folder)

    folders = [folder for group in grouped_folders.values() for folder in group]

    aggregated_data = aggregate_data(folders)

    # print_table(aggregated_data)

    print_latex_table(aggregated_data)


if __name__ == "__main__":
    main()
