import argparse
from matplotlib import pyplot as plt

import datetime
import json
import os

import numpy as np
import random

import time
import sys
import multiprocessing
import copy
import traceback

import gc

from typing import Dict, Any, Callable, Tuple, List

from multi_robot_multi_goal_planning.problems import get_env_by_name

from multi_robot_multi_goal_planning.problems.planning_env import BaseProblem
from multi_robot_multi_goal_planning.problems.rai_base_env import rai_env

# planners
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    RuntimeTerminationCondition,
)
from multi_robot_multi_goal_planning.planners.prioritized_planner import (
    PrioritizedPlanner,
)
from multi_robot_multi_goal_planning.planners.composite_prm_planner import (
    CompositePRM,
    CompositePRMConfig,
)
from multi_robot_multi_goal_planning.planners.planner_rrtstar import RRTstar
from multi_robot_multi_goal_planning.planners.planner_birrtstar import (
    BidirectionalRRTstar,
)
from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTConfig
from multi_robot_multi_goal_planning.planners.itstar_base import BaseITConfig
from make_plots import make_cost_plots
from multi_robot_multi_goal_planning.planners.planner_aitstar import AITstar
from multi_robot_multi_goal_planning.planners.planner_eitstar import EITstar
# np.random.seed(100)


def merge_config(
    user_config: Dict[str, Any], default_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge user_config with default_config while preserving the order of user_config keys."""
    if not isinstance(user_config, dict):
        return user_config  # Base case for non-dict values

    for key, default_value in default_config.items():
        if key in user_config:
            if isinstance(user_config[key], dict) and isinstance(default_value, dict):
                merge_config(user_config[key], default_value)  # Recursive merge
        else:
            user_config[key] = default_value  # Add missing default keys at the end

    return user_config


def validate_config(config: Dict[str, Any]) -> None:
    pass


def load_experiment_config(filepath: str) -> Dict[str, Any]:
    with open(filepath) as f:
        config = json.load(f)

    planner_default_configs, planner_default_config_paths = {}, {}
    planner_default_config_paths["prm"] = "configs/defaults/composite_prm.json"
    planner_default_config_paths["rrtstar"] = "configs/defaults/rrtstar.json"
    planner_default_config_paths["birrtstar"] = "configs/defaults/birrtstar.json"
    planner_default_config_paths["aitstar"] = "configs/defaults/aitstar.json"
    planner_default_config_paths["eitstar"] = "configs/defaults/eitstar.json"

    for planner_type, default_config_path in planner_default_config_paths.items():
        with open(default_config_path) as f:
            planner_default_configs[planner_type] = json.load(f)

    for i, planner in enumerate(config["planners"]):
        planner_type = planner["type"]

        if planner_type in planner_default_configs:
            merged_planner = merge_config(
                planner, planner_default_configs[planner_type]
            )
            config["planners"][i] = merged_planner

    # TODO: sanity checks
    validate_config(config)

    return config


def run_single_planner(
    env: BaseProblem, planner: Callable[[BaseProblem], Tuple[Any, Dict]]
) -> Dict:
    _, data = planner(env)

    return data


def export_planner_data(planner_folder: str, run_id: int, planner_data: Dict):
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


def export_config(path: str, config: Dict):
    with open(path + "config.json", "w") as f:
        json.dump(config, f)


def setup_planner(
    planner_config, runtime: int, optimize: bool = True
) -> Callable[[BaseProblem], Tuple[Any, Dict]]:
    name = planner_config["name"]

    if planner_config["type"] == "prm":

        def planner(env):
            options = planner_config["options"]
            prm_config = CompositePRMConfig(
                # Map dictionary keys to dataclass attributes
                distance_metric=options["distance_metric"],
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
                try_shortcutting=options["shortcutting"],
                try_direct_informed_sampling=options["direct_informed_sampling"],
                # Corrected spelling here as well:
                inlcude_lb_in_informed_sampling=options[
                    "inlcude_lb_in_informed_sampling"
                ],
                init_mode_sampling_type=options["init_mode_sampling_type"],
                frontier_mode_sampling_probability=options[
                    "frontier_mode_sampling_probability"
                ],
                init_uniform_batch_size=options["init_uniform_batch_size"],
                init_transition_batch_size=options["init_transition_batch_size"],
                with_mode_validation=options["with_mode_validation"],
                with_noise=options["with_noise"],
            )

            return CompositePRM(env, config=prm_config).plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )
    elif planner_config["type"] == "rrtstar":

        def planner(env):
            options = planner_config["options"]
            rrtstar_config = BaseRRTConfig(
                general_goal_sampling=options["general_goal_sampling"],
                informed_sampling=options["informed_sampling"],
                informed_sampling_version=options["informed_sampling_version"],
                distance_metric=options["distance_metric"],
                p_goal=options["p_goal"],
                p_stay=options["p_stay"],
                p_uniform=options["p_uniform"],
                shortcutting=options["shortcutting"],
                init_mode_sampling_type=options["init_mode_sampling_type"],
                frontier_mode_sampling_probability=options[
                    "frontier_mode_sampling_probability"
                ],
                locally_informed_sampling=options["locally_informed_sampling"],
                informed_batch_size=options["informed_batch_size"],
                sample_near_path=options["sample_near_path"],
                remove_redundant_nodes=options["remove_redundant_nodes"],
                apply_long_horizon=options["apply_long_horizon"],
                horizon_length=options["horizon_length"],
                with_mode_validation=options["with_mode_validation"],
                with_noise=options["with_noise"],
                with_tree_visualization=options["with_tree_visualization"],
            )

            return RRTstar(env, config=rrtstar_config).plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )
    elif planner_config["type"] == "birrtstar":

        def planner(env):
            options = planner_config["options"]
            birrtstar_config = BaseRRTConfig(
                general_goal_sampling=options["general_goal_sampling"],
                informed_sampling=options["informed_sampling"],
                informed_sampling_version=options["informed_sampling_version"],
                distance_metric=options["distance_metric"],
                p_goal=options["p_goal"],
                p_stay=options["p_stay"],
                p_uniform=options["p_uniform"],
                shortcutting=options["shortcutting"],
                init_mode_sampling_type=options["init_mode_sampling_type"],
                frontier_mode_sampling_probability=options[
                    "frontier_mode_sampling_probability"
                ],
                locally_informed_sampling=options["locally_informed_sampling"],
                sample_near_path=options["sample_near_path"],
                transition_nodes=options["transition_nodes"],
                birrtstar_version=options["birrtstar_version"],
                informed_batch_size=options["informed_batch_size"],
                remove_redundant_nodes=options["remove_redundant_nodes"],
                apply_long_horizon=options["apply_long_horizon"],
                horizon_length=options["horizon_length"],
                with_mode_validation=options["with_mode_validation"],
                with_noise=options["with_noise"],
                with_tree_visualization=options["with_tree_visualization"],
            )
            return BidirectionalRRTstar(env, config=birrtstar_config).plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )
    elif planner_config["type"] == "aitstar":

        def planner(env):
            options = planner_config["options"]
            aitstar_config = BaseITConfig(
                init_mode_sampling_type=options["init_mode_sampling_type"],
                distance_metric=options["distance_metric"],
                try_sampling_around_path=options["sample_near_path"],
                try_informed_sampling=options["informed_sampling"],
                try_informed_transitions=options["informed_transition_sampling"],
                init_uniform_batch_size=options["init_uniform_batch_size"],
                init_transition_batch_size=options["init_transition_batch_size"],
                uniform_batch_size=options["batch_size"],
                uniform_transition_batch_size=options["transition_batch_size"],
                informed_batch_size=options["informed_batch_size"],
                informed_transition_batch_size=options[
                    "informed_transition_batch_size"
                ],
                path_batch_size=options["path_batch_size"],
                locally_informed_sampling=options["locally_informed_sampling"],
                try_shortcutting=options["shortcutting"],
                try_direct_informed_sampling=options["direct_informed_sampling"],
                inlcude_lb_in_informed_sampling=options[
                    "inlcude_lb_in_informed_sampling"
                ],
                remove_based_on_modes=options["remove_based_on_modes"],
                with_tree_visualization=options["with_tree_visualization"],
                apply_long_horizon=options["apply_long_horizon"],
                frontier_mode_sampling_probability=options[
                    "frontier_mode_sampling_probability"
                ],
                horizon_length=options["horizon_length"],
                with_rewiring=options["with_rewiring"],
                with_mode_validation=options["with_mode_validation"],
                with_noise=options["with_noise"],
            )
            return AITstar(env, config=aitstar_config).plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )
    elif planner_config["type"] == "eitstar":

        def planner(env):
            options = planner_config["options"]
            eitstar_config = BaseITConfig(
                init_mode_sampling_type=options["init_mode_sampling_type"],
                distance_metric=options["distance_metric"],
                try_sampling_around_path=options["sample_near_path"],
                try_informed_sampling=options["informed_sampling"],
                try_informed_transitions=options["informed_transition_sampling"],
                init_uniform_batch_size=options["init_uniform_batch_size"],
                init_transition_batch_size=options["init_transition_batch_size"],
                uniform_batch_size=options["batch_size"],
                uniform_transition_batch_size=options["transition_batch_size"],
                informed_batch_size=options["informed_batch_size"],
                informed_transition_batch_size=options[
                    "informed_transition_batch_size"
                ],
                path_batch_size=options["path_batch_size"],
                locally_informed_sampling=options["locally_informed_sampling"],
                try_shortcutting=options["shortcutting"],
                try_direct_informed_sampling=options["direct_informed_sampling"],
                inlcude_lb_in_informed_sampling=options[
                    "inlcude_lb_in_informed_sampling"
                ],
                remove_based_on_modes=options["remove_based_on_modes"],
                with_tree_visualization=options["with_tree_visualization"],
                apply_long_horizon=options["apply_long_horizon"],
                frontier_mode_sampling_probability=options[
                    "frontier_mode_sampling_probability"
                ],
                horizon_length=options["horizon_length"],
                with_rewiring=options["with_rewiring"],
                with_mode_validation=options["with_mode_validation"],
                with_noise=options["with_noise"],
            )
            return EITstar(env, config=eitstar_config).plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )

    else:
        raise ValueError(f"Planner type {planner_config['type']} not implemented")


def setup_env(env_config):
    pass


class Tee:
    """Custom stream to write to both stdout and a file."""

    def __init__(self, file, print_to_file_and_stdout: bool):
        self.file = file
        self.print_to_file_and_stdout = print_to_file_and_stdout

        if self.print_to_file_and_stdout:
            self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.file.flush()  # Ensure immediate writing

        if self.print_to_file_and_stdout:
            self.stdout.write(data)
            self.stdout.flush()

    def flush(self):
        self.file.flush()

        if self.print_to_file_and_stdout:
            self.stdout.flush()


def run_experiment(
    env: BaseProblem,
    planners: List[Tuple[str, Callable[[BaseProblem], Tuple[Any, Dict]]]],
    config: Dict,
    experiment_folder: str,
):
    seed = config["seed"]

    all_experiment_data = {}
    for planner_name, _ in planners:
        all_experiment_data[planner_name] = []

    for run_id in range(config["num_runs"]):
        for planner_name, planner in planners:
            planner_folder = experiment_folder + f"{planner_name}/"
            os.makedirs(planner_folder, exist_ok=True)

            log_file = f"{planner_folder}run_{run_id}.log"

            with open(log_file, "w", buffering=1) as f:  # Line-buffered writing
                # Redirect stdout and stderr
                sys.stdout = Tee(f, True)
                sys.stderr = Tee(f, True)

                try:
                    print(f"Run #{run_id} for {planner_name}")
                    print(f"Seed {seed + run_id}")

                    np.random.seed(seed + run_id)
                    random.seed(seed + run_id)

                    env_copy = copy.deepcopy(env)
                    res = run_single_planner(env_copy, planner)

                    if isinstance(env, rai_env):
                        del env_copy.C

                    del planner
                    gc.collect()

                    planner_folder = experiment_folder + f"{planner_name}/"
                    if not os.path.isdir(planner_folder):
                        os.makedirs(planner_folder)

                    all_experiment_data[planner_name].append(res)

                    # export planner data
                    export_planner_data(planner_folder, run_id, res)
                except Exception as e:
                    print(f"Error in {planner_name} run {run_id}: {e}")
                    tb = traceback.format_exc()  # Get the full traceback
                    print(
                        f"Error in {planner_name} run {run_id}: {e}\nTraceback:\n{tb}"
                    )

                finally:
                    # Restore stdout and stderr
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__

    return all_experiment_data


def run_planner_process(
    run_id: int,
    planner_name: str,
    planner: Callable[[BaseProblem], Tuple[Any, Dict]],
    seed: int,
    env: BaseProblem,
    experiment_folder: str,
    results: List,  # Changed from Queue to List
    semaphore,
    print_to_file_and_stdout: bool = False,
):
    try:
        semaphore.acquire()
        planner_folder = experiment_folder + f"{planner_name}/"
        os.makedirs(planner_folder, exist_ok=True)

        log_file = f"{planner_folder}run_{run_id}.log"

        with open(log_file, "w", buffering=1) as f:
            sys.stdout = Tee(f, print_to_file_and_stdout)
            sys.stderr = Tee(f, print_to_file_and_stdout)

            try:
                np.random.seed(seed + run_id)
                random.seed(seed + run_id)

                res = run_single_planner(env, planner)

                if isinstance(env, rai_env):
                    del env.C
                del planner
                gc.collect()

                planner_folder = experiment_folder + f"{planner_name}/"
                os.makedirs(planner_folder, exist_ok=True)

                export_planner_data(planner_folder, run_id, res)
                results.append((planner_name, res))

            except Exception as e:
                print(f"Error in {planner_name} run {run_id}: {e}")
                results.append((planner_name, None))

            finally:
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                sys.stdout.flush()
                sys.stderr.flush()

    finally:
        semaphore.release()
        print(f"DONE {planner_name} {run_id}")


def run_experiment_in_parallel(
    env: BaseProblem,
    planners,
    config: Dict,
    experiment_folder: str,
    max_parallel: int = 4,
):
    """Runs experiments in parallel with a fixed number of processes."""
    all_experiment_data = {planner_name: [] for planner_name, _ in planners}
    seed = config["seed"]

    # Use Manager instead of Queue for better cleanup
    with multiprocessing.Manager() as manager:
        results = manager.list()
        semaphore = manager.Semaphore(max_parallel)
        processes = []

        try:
            # Launch separate processes
            for run_id in range(config["num_runs"]):
                for planner_name, planner in planners:
                    env_copy = copy.deepcopy(env)
                    p = multiprocessing.Process(
                        target=run_planner_process,
                        args=(
                            run_id,
                            planner_name,
                            planner,
                            seed,
                            env_copy,
                            experiment_folder,
                            results,  # Use manager.list instead of Queue
                            semaphore,
                        ),
                    )
                    p.daemon = True  # Make processes daemon
                    p.start()
                    processes.append(p)

            # Wait for processes with timeout
            for p in processes:
                p.join()  # Add timeout to join

            # Collect results
            for planner_name, res in results:
                if res is not None:
                    all_experiment_data[planner_name].append(res)

        except KeyboardInterrupt:
            print("\nCaught KeyboardInterrupt, terminating processes...")
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    try:
                        p.join(timeout=0.1)
                    except TimeoutError:
                        pass

        finally:
            # Force terminate any remaining processes
            for p in processes:
                if p.is_alive():
                    try:
                        p.terminate()
                        p.join(timeout=0.1)
                    except (TimeoutError, Exception):
                        pass

    return all_experiment_data


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("filepath", nargs="?", default="default", help="filepath")
    parser.add_argument(
        "--parallel_execution",
        action="store_true",
        help="Run the experiments in parallel. (default: False)",
    )
    parser.add_argument(
        "--display_result",
        action="store_true",
        help="Display the resulting plots at the end. (default: False)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=2,
        help="Number of processes to run in parallel. (default: 2)",
    )

    args = parser.parse_args()
    config = load_experiment_config(args.filepath)

    # make sure that the environment is initializaed correctly
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    env = get_env_by_name(config["environment"])
    env.cost_reduction = config["cost_reduction"]
    env.cost_metric = config["per_agent_cost"]

    if False:
        env.show()

    planners = []
    for planner_config in config["planners"]:
        planners.append(
            setup_planner(
                planner_config, config["max_planning_time"], config["optimize"]
            )
        )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # convention: alsways use "/" as trailing character
    experiment_folder = (
        f"./out/{timestamp}_{config['experiment_name']}_{config['environment']}/"
    )

    if not os.path.isdir(experiment_folder):
        os.makedirs(experiment_folder)

    export_config(experiment_folder, config)

    if args.parallel_execution:
        all_experiment_data = run_experiment_in_parallel(
            env, planners, config, experiment_folder, max_parallel=args.num_processes
        )
    else:
        all_experiment_data = run_experiment(env, planners, config, experiment_folder)

    if args.display_result:
        make_cost_plots(all_experiment_data, config)
        plt.show()


if __name__ == "__main__":
    main()
