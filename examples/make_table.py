import numpy as np

from make_plots import load_data_from_folder
import os
from string import Template


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


def aggregate_data(folders):
    env_results = {}
    for experiment in folders:
        # TODO change behavior to add none
        all_experiment_data = load_data_from_folder(experiment)
        env_name = experiment.split(".")[-1][:-1]

        env_results[env_name] = {}

        for planner_name, results in all_experiment_data.items():
            t_init_median, t_init_lb, t_init_ub = get_initial_solution_time(results)
            c_init_median, c_init_lb, c_init_ub = get_initial_solution_cost(results)
            c_final_median, c_final_lb, c_final_ub = get_final_solution_cost(results)

            env_results[env_name][planner_name] = {
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
    line_layout = ["t_init_median", "c_init_median", "c_final_median"]
    key_to_name = {
        "t_init_median": "\\multicolumn{2}{c}{$t_\\text{{init}}$}",
        "c_init_median": "\multicolumn{2}{c}{$c_\\text{{init}}$}",
        "c_final_median": "\multicolumn{2}{c}{$c_\\text{{final}}$}",
    }

    planner_names = list(next(iter(aggregated_data.values())).keys())

    num_cols = 1 + len(planner_names) * len(line_layout)

    colspec = "l " + len(line_layout) * ("|" + len(planner_names) * "c")

    header1 = " "
    for key in line_layout:
        header1 += f"& {key_to_name[key]} "

    header2 = " "
    for key in line_layout:
        for planner_name in planner_names:
            header2 += f"& {planner_name} "

    body = ""

    # print stuff per line
    for env_name, all_planner_data in aggregated_data.items():
        escaped_env_name = env_name.replace("_", "\\_")
        body += f"{escaped_env_name} "
        for key in line_layout:
            all_values = [
                data[key] if data[key] is not None else float("inf")
                for planner_name, data in all_planner_data.items()
            ]
            for i, v in enumerate(all_values):
                if v == min(all_values):
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
    main_dir = "./out/tmp/"
    folders = get_subfolders_from_main_folder(main_dir)

    group_envs_by = ["dep", "unassigned", "unordered"]

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
