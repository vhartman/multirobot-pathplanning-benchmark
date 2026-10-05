import argparse
from matplotlib import pyplot as plt

import datetime
import json
import os
import time
from dataclasses import asdict

import numpy as np
import random

import sys
import multiprocessing
import copy
import traceback

import gc

from typing import Dict, Any, Callable, Tuple, List, Optional

from multi_robot_multi_goal_planning.problems import get_env_by_name

from multi_robot_multi_goal_planning.problems.planning_env import BaseProblem
from multi_robot_multi_goal_planning.problems.rai_base_env import rai_env

# planners
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    RuntimeTerminationCondition,
)
from scripts.analysis.make_plots import make_cost_plots
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
    RRTSkills,
    RRTSkillsConfig,
    RRTSkillsConservative,
    RRTSkillsConservativeConfig
)
from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTstar
from multi_robot_multi_goal_planning.planners.rrt_skills_reactive import (
    ReactiveRoadmap,
)
from multi_robot_multi_goal_planning.planners.reactive_policy import ReactiveExecutor
from multi_robot_multi_goal_planning.problems.util import interpolate_path
from scripts.run_stochastic_experiments import (
    ReactiveExperimentConfig,
    build_reactive_configs,
    draw_execution_base_seed,
    evaluate_open_loop,
    evaluate_reactive_policy,
    _execution_seed,
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
    paths = list(planner_data["paths"]) + list(planner_data.get("execution_paths") or [])
    for i, path in enumerate(paths):
        if not path:
            continue
        # export path
        file_path = f"{run_folder}path_{i}.json"
        with open(file_path, "w") as f:
            json.dump([state.to_dict() for state in path], f)
        execution_index = i - len(planner_data["paths"])
        executions = planner_data.get("executions") or []
        if execution_index >= 0 and execution_index < len(executions):
            timing_file = f"{run_folder}timing_{i}.json"
            with open(timing_file, "w") as f:
                json.dump({"timestamps": executions[execution_index].get("state_timestamps", [])}, f)

    # write all costs with their timestamps to file
    with open(planner_folder + "timestamps.txt", "ab") as f:
        timestamps = planner_data["times"]

        np.savetxt(f, timestamps, delimiter=",", newline=",")
        f.write(b"\n")

    with open(planner_folder + "costs.txt", "ab") as f:
        costs = planner_data["costs"]

        np.savetxt(f, costs, delimiter=",", newline=",")
        f.write(b"\n")

    if planner_data.get("realized_costs"):
        with open(planner_folder + "realized_timestamps.txt", "ab") as f:
            np.savetxt(f, planner_data["realized_times"], delimiter=",", newline=",")
            f.write(b"\n")
        with open(planner_folder + "realized_costs.txt", "ab") as f:
            np.savetxt(f, planner_data["realized_costs"], delimiter=",", newline=",")
            f.write(b"\n")

    if "executions" in planner_data:
        with open(f"{run_folder}executions.json", "w") as f:
            json.dump(planner_data["executions"], f, indent=2)

    if "shortcut_history" in planner_data:
        with open(f"{run_folder}shortcut_history.json", "w") as f:
            json.dump({"build_time": planner_data.get("build_time"),
                       "planning_failed": planner_data.get("planning_failed", False),
                       "phase_a_points": planner_data.get("phase_a_points"),
                       "history": planner_data["shortcut_history"]}, f, indent=2)


def export_config(path: str, config: Dict):
    with open(path + "config.json", "w") as f:
        json.dump(config, f)


def setup_planner(
    planner_config, runtime: int, optimize: bool = True, cost_model: str = "geometric",
    problem_env: Optional[BaseProblem] = None,
) -> Tuple[str, Callable[[BaseProblem], Tuple[Any, Dict]], Any]:
    name = planner_config["name"]

    if planner_config["type"] == "prm":
        options = planner_config["options"]
        config = CompositePRMConfig()
        for k, v in options.items():
            setattr(config, k, v)

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
        options = planner_config["options"]
        config = BaseRRTConfig()
        for k, v in options.items():
            setattr(config, k, v)

        def planner(env):
            p = RRTstar(env, config=config)
            result = p.plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )
            planner.last_instance = p
            return result
    elif planner_config["type"] == "birrtstar":
        options = planner_config["options"]
        config = BaseRRTConfig()
        for k, v in options.items():
            setattr(config, k, v)

        def planner(env):
            p = BidirectionalRRTstar(env, config=config)
            result = p.plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )
            planner.last_instance = p
            return result
    elif planner_config["type"] == "aitstar":
        options = planner_config["options"]
        config = BaseITConfig()
        for k, v in options.items():
            setattr(config, k, v)

        def planner(env):
            return AITstar(env, config=config).plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )
    elif planner_config["type"] == "eitstar":
        options = planner_config["options"]
        config = BaseITConfig()
        for k, v in options.items():
            setattr(config, k, v)

        def planner(env):
            return EITstar(env, config=config).plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )
    elif planner_config["type"] == "prioritized":
        planner_cls, config_cls = PrioritizedPlanner, PrioritizedPlannerConfig

        options = dict(planner_config["options"])
        n_executions = options.pop("num_executions", 0)
        config = config_cls()
        for k, v in options.items():
            setattr(config, k, v)

        def planner(env):
            seed_base = draw_execution_base_seed()
            path, info = planner_cls(env, config).plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )
            if n_executions > 0 and path is not None:
                info["executions"], info["execution_paths"] = evaluate_open_loop(
                    env, path, n_executions, seed_base)
            return path, info

    elif planner_config["type"] == "short_horizon":
        options = planner_config["options"]
        config = RecedingHorizonConfig()
        for k, v in options.items():
            setattr(config, k, v)

        def planner(env):
            return RecedingHorizonPlanner(env, config).plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )
    elif planner_config["type"] == "rrt_skills":
        options = dict(planner_config["options"])
        n_executions = options.pop("num_executions", 0)
        config = RRTSkillsConfig()
        for k, v in options.items():
            setattr(config, k, v)

        def planner(env):
            seed_base = draw_execution_base_seed()
            rrt_config = RRTSkillsConfig()
            for k, v in options.items():
                setattr(rrt_config, k, v)

            path, info = RRTSkills(env, config=rrt_config).plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )
            if n_executions > 0 and path is not None:
                nominal = interpolate_path(path, 0.05, kind="euclidean")
                info["executions"], info["execution_paths"] = evaluate_open_loop(
                    env, nominal, n_executions, seed_base)
            return path, info
    elif planner_config["type"] == "rrt_skills_conservative":
        options = dict(planner_config["options"])
        n_executions = options.pop("num_executions", 0)
        config = RRTSkillsConservativeConfig()
        for k, v in options.items():
            setattr(config, k, v)

        def planner(env):
            seed_base = draw_execution_base_seed()
            rrt_config = RRTSkillsConservativeConfig()
            for k, v in options.items():
                setattr(rrt_config, k, v)

            planner_obj = RRTSkillsConservative(env, config=rrt_config)
            path, info = planner_obj.plan(
                ptc=RuntimeTerminationCondition(runtime),
                optimize=optimize,
            )
            if n_executions > 0 and path is not None:
                nominal = interpolate_path(path, 0.05, kind="euclidean")
                info["executions"], info["execution_paths"] = evaluate_open_loop(
                    env, nominal, n_executions, seed_base)
            return path, info

    elif planner_config["type"] == "reactive":
        options = dict(planner_config["options"])
        n_executions = options.pop("num_executions", 100)
        options.pop("batched", None)  # compatibility with older experiment configs
        (resolved_roadmap, resolved_mdp, resolved_shortcut,
         resolved_batched) = build_reactive_configs(
            options, runtime, cost_model, problem_env)
        config = ReactiveExperimentConfig(
            roadmap=asdict(resolved_roadmap),
            mdp=asdict(resolved_mdp),
            shortcut=asdict(resolved_shortcut),
            batched=asdict(resolved_batched),
            num_executions=n_executions,
        )

        def planner(env):
            seed_base = draw_execution_base_seed()
            roadmap_config, mdp_config, shortcut_config, batched_config = build_reactive_configs(
                options, runtime, cost_model, env)

            t0 = time.time()
            deadline = t0 + runtime
            from multi_robot_multi_goal_planning.planners.reactive_batched import (
                batched_reactive_plan,
            )
            roadmap = ReactiveRoadmap(env, roadmap_config)
            build_time = time.time() - t0
            mdp, shortcut_history = batched_reactive_plan(
                roadmap, mdp_config, shortcut_config, batched_config,
                deadline=deadline)
            plan_time = time.time() - t0

            planning_failed = not np.isfinite(mdp.get_start_cost_to_go())
            if planning_failed:
                print("[PLANNING FAILURE] V(start) is infinite -- no policy exists for this "
                      "roadmap, so no executions are run. This run counts against PLANNING "
                      "success, not against execution success.")
                executions, execution_paths = [], []
            else:
                executions, execution_paths = evaluate_reactive_policy(
                    env, mdp, n_executions, seed_base)

            times = [float(t) for t in getattr(roadmap, "phase_a_times", [])]
            costs = [float(c) for c in getattr(roadmap, "phase_a_costs", [])]
            wall = build_time
            for h in shortcut_history:
                wall += h["improve_time"] + h["solve_time"]
                times.append(wall)
                costs.append(float(h["V_start"]))

            strictly_increasing = ([0] + [i for i in range(1, len(times)) if times[i] > times[i - 1]]
                                   if times else [])
            n_phase_a = sum(1 for i in strictly_increasing if i < len(roadmap.phase_a_times))
            times = [times[i] for i in strictly_increasing]
            costs = [costs[i] for i in strictly_increasing]

            best = min((e for e in executions if e["success"]),
                       key=lambda e: e["cost"], default=None)
            representative = best["index"] if best else 0
            np.random.seed(_execution_seed(seed_base, representative))
            random.seed(_execution_seed(seed_base, representative))
            path = None if planning_failed else ReactiveExecutor(mdp).run().path

            return path, {
                "planning_failed": planning_failed,
                "paths": [path],
                "times": times or [plan_time],
                "costs": costs or [float(mdp.get_start_cost_to_go())],
                "executions": executions,
                "execution_paths": execution_paths,
                "shortcut_history": shortcut_history,
                "build_time": build_time,
                "phase_a_points": n_phase_a,
            }

    else:
        raise ValueError(f"Planner type {planner_config['type']} not implemented")

    return name, planner, config


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

                    rrt_instance = getattr(planner, "last_instance", None)
                    del planner
                    gc.collect()

                    planner_folder = experiment_folder + f"{planner_name}/"
                    if not os.path.isdir(planner_folder):
                        os.makedirs(planner_folder)

                    all_experiment_data[planner_name].append(res)

                    # export planner data
                    export_planner_data(planner_folder, run_id, res)

                    # export RRT timing breakdown if available
                    if isinstance(rrt_instance, BaseRRTstar):
                        p = rrt_instance
                        timing_file = planner_folder + "timing.csv"
                        write_header = not os.path.exists(timing_file)
                        with open(timing_file, "a") as tf:
                            if write_header:
                                tf.write("run_id,sampling_time,edge_success_time,edge_failure_time,coll_checking_time\n")
                            tf.write(
                                f"{run_id},"
                                f"{p._sampling_time:.6f},"
                                f"{p._edge_check_time_success:.6f},"
                                f"{p._edge_check_time_failure:.6f},"
                                f"{p._coll_checking_time:.6f}\n"
                            )
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

                rrt_instance = getattr(planner, "last_instance", None)
                del planner
                gc.collect()

                planner_folder = experiment_folder + f"{planner_name}/"
                os.makedirs(planner_folder, exist_ok=True)

                export_planner_data(planner_folder, run_id, res)
                results.append((planner_name, res))

                if isinstance(rrt_instance, BaseRRTstar):
                    p = rrt_instance
                    timing_file = planner_folder + "timing.csv"
                    write_header = not os.path.exists(timing_file)
                    with open(timing_file, "a") as tf:
                        if write_header:
                            tf.write("run_id,sampling_time,edge_success_time,edge_failure_time,coll_checking_time\n")
                        tf.write(
                            f"{run_id},"
                            f"{p._sampling_time:.6f},"
                            f"{p._edge_check_time_success:.6f},"
                            f"{p._edge_check_time_failure:.6f},"
                            f"{p._coll_checking_time:.6f}\n"
                        )

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
    cost_model = config.get("cost_model", "geometric")
    env.cost_model = cost_model
    env.v_ref = config.get("v_ref", 1.0)

    if False:
        env.show()

    planners = []
    for planner_config in config["planners"]:
        name, planner_fn, resolved_config = setup_planner(
            planner_config, config["max_planning_time"], config["optimize"],
            cost_model=cost_model, problem_env=env,
        )
        planners.append(
            (name, planner_fn)
        )
        planner_config["options"] = asdict(resolved_config)


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # convention: alsways use "/" as trailing character
    base_out = os.environ.get("MRMG_OUTPUT_DIR", "./out").rstrip("/")
    experiment_folder = (
        f"{base_out}/{timestamp}_{config['experiment_name']}_{config['environment']}/"
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
