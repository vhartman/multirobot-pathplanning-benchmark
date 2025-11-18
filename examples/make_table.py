import numpy as np

from make_plots import load_data_from_folder


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
        env_name = experiment.split('.')[-1][:-1]

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
    print('env', end=' ')

    planner_names = list(next(iter(aggregated_data.values())).keys())
    for key in line_layout:
        print(key, end=' ')
    print()

    for key in line_layout:
        for planner_name in planner_names:
          print(planner_name, end=' ')
    print()

    # print stuff per line
    for env_name, all_planner_data in aggregated_data.items():
        print(env_name, end=' ')
        for key in line_layout:
            all_values = [data[key] if data[key] is not None else float('inf') for planner_name, data in all_planner_data.items()]
            for v in all_values:
                if v == min(all_values):
                    print(v, end=' ')
                else:
                    print(v, end=' ')

        print()


def main():
    folders = [
        
    ]

    aggregated_data = aggregate_data(folders)

    print_table(aggregated_data)


if __name__ == "__main__":
    main()
