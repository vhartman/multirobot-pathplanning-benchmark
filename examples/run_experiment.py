import argparse
from matplotlib import pyplot as plt

import datetime
import json
import os

import numpy as np
import random

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
from make_plots import make_cost_plots
# np.random.seed(100)

from multi_robot_multi_goal_planning.planners import (
    PrioritizedPlanner,
    PrioritizedPlannerConfig,
    CompositePRM,
    CompositePRMConfig,
    BaseRRTConfig,
    RRTstar,
    BidirectionalRRTstar,
    BaseITConfig,
    AITstar,
    EITstar,
    RecedingHorizonConfig,
    RecedingHorizonPlanner,
)

def validate_config(config: Dict[str, Any]) -> None:
    pass


def load_experiment_config(filepath: str) -> Dict[str, Any]:
    with open(filepath) as f:
        config = json.load(f)

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
            prm_config = CompositePRMConfig()
            for k, v in options.items():
                setattr(prm_config, k, v)

            return CompositePRM(env, config=prm_config).plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )
    elif planner_config["type"] == "rrtstar":

        def planner(env):
            options = planner_config["options"]
            rrtstar_config = BaseRRTConfig()
            for k, v in options.items():
                setattr(rrtstar_config, k, v)

            return RRTstar(env, config=rrtstar_config).plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )
    elif planner_config["type"] == "birrtstar":

        def planner(env):
            options = planner_config["options"]
            birrtstar_config = BaseRRTConfig()
            for k, v in options.items():
                setattr(birrtstar_config, k, v)

            return BidirectionalRRTstar(env, config=birrtstar_config).plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )
    elif planner_config["type"] == "aitstar":

        def planner(env):
            options = planner_config["options"]
            aitstar_config = BaseITConfig()
            for k, v in options.items():
                setattr(aitstar_config, k, v)

            return AITstar(env, config=aitstar_config).plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )
    elif planner_config["type"] == "eitstar":

        def planner(env):
            options = planner_config["options"]
            eitstar_config = BaseITConfig()
            for k, v in options.items():
                setattr(eitstar_config, k, v)

            return EITstar(env, config=eitstar_config).plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )
    elif planner_config["type"] == "prioritized":

        def planner(env):
            options = planner_config["options"]
            prio_config = PrioritizedPlannerConfig()
            for k, v in options.items():
                setattr(prio_config, k, v)

            return PrioritizedPlanner(env, prio_config).plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )
    elif planner_config["type"] == "short_horizon":

        def planner(env):
            options = planner_config["options"]
            config = RecedingHorizonConfig()
            for k, v in options.items():
                setattr(config, k, v)

            return RecedingHorizonPlanner(env, config).plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )

    else:
        raise ValueError(f"Planner type {planner_config['type']} not implemented")

    return name, planner


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
    num_runs,
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
        current_datetime = datetime.datetime.now()
        readable_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        print(f"DONE {planner_name} {run_id}/{num_runs} at time {readable_time}")


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
                            config["num_runs"],
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
