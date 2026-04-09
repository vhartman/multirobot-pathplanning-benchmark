import argparse
import copy
import datetime
import os
import random
import sys

import numpy as np
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_robot_multi_goal_planning.problems.util import compute_reachable_modes

from run_experiment import (
    setup_planner,
    run_experiment,
    run_experiment_in_parallel,
    export_config,
    load_experiment_config,
)

from multi_robot_multi_goal_planning.problems.rai_envs import rai_ur10_arm_box_stack_env, rai_mobile_manip_wall, rai_isolated_arm_box_stack_env


DEFAULT_PLANNER_CONFIGS = [
    {"name": "rrt", "type": "birrtstar", "options": {"with_mode_validation": False}},
    {"name": "ait", "type": "aitstar", "options": {"with_mode_validation": False}},
    {"name": "prioritized", "type": "prioritized", "options": {}},
]

DEFAULT_CONFIG = {
    "seed": 3,
    "num_runs": 10,
    "optimize": False,
    "max_planning_time": 500,
    "per_agent_cost": "euclidean",
    "cost_reduction": "max",
}


def run_stacking_scaling(
    base_config: dict,
    parallel: bool,
    num_processes: int,
):
    for num_robots in range(1, 5):
        for num_boxes in range(1, 9):
            while True:
                np.random.seed(base_config["seed"])
                random.seed(base_config["seed"])

                print(f"\n=== Scaling test: {num_robots} robots, {num_boxes} boxes ===")

                env = rai_ur10_arm_box_stack_env(num_robots=num_robots, num_boxes=num_boxes)
                env.cost_reduction = base_config["cost_reduction"]
                env.cost_metric = base_config["per_agent_cost"]

                # make copy of the env, and test if we can reach the terminal mode
                env_tmp = copy.deepcopy(env)
                modes = compute_reachable_modes(env_tmp)
                goal_mode_reachable = False
                for m in modes:
                    if env_tmp.is_terminal_mode(m):
                        goal_mode_reachable = True
                        break

                if goal_mode_reachable:
                    del env_tmp
                    break


            config = copy.deepcopy(base_config)
            config["experiment_name"] = "scaling"
            config["environment"] = f"stacking_r{num_robots}_b{num_boxes}"

            planners = []
            for planner_config in config["planners"]:
                name, planner_fn, resolved_config = setup_planner(
                    planner_config, config["max_planning_time"], config["optimize"]
                )
                planners.append((name, planner_fn))
                planner_config["options"] = asdict(resolved_config)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_folder = (
                f"./out/{timestamp}_{config['experiment_name']}_{config['environment']}/"
            )
            os.makedirs(experiment_folder, exist_ok=True)
            export_config(experiment_folder, config)

            if parallel:
                run_experiment_in_parallel(
                    env, planners, config, experiment_folder,
                    max_parallel=num_processes,
                )
            else:
                run_experiment(env, planners, config, experiment_folder)

def run_isolated_stacking(
    base_config: dict,
    parallel: bool,
    num_processes: int,
):
    for num_robots in range(1, 8 + 1):
        for num_boxes in range(1, 5):
            np.random.seed(base_config["seed"])
            random.seed(base_config["seed"])

            print(f"\n=== Scaling test: {num_robots} robots, {num_boxes} boxes ===")

            env = rai_isolated_arm_box_stack_env(num_robots=num_robots, num_boxes=num_boxes)
            env.cost_reduction = base_config["cost_reduction"]
            env.cost_metric = base_config["per_agent_cost"]

            config = copy.deepcopy(base_config)
            config["experiment_name"] = "scaling"
            config["environment"] = f"stacking_r{num_robots}_b{num_boxes}"

            planners = []
            for planner_config in config["planners"]:
                name, planner_fn, resolved_config = setup_planner(
                    planner_config, config["max_planning_time"], config["optimize"]
                )
                planners.append((name, planner_fn))
                planner_config["options"] = asdict(resolved_config)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_folder = (
                f"./out/{timestamp}_{config['experiment_name']}_{config['environment']}/"
            )
            os.makedirs(experiment_folder, exist_ok=True)
            export_config(experiment_folder, config)

            if parallel:
                run_experiment_in_parallel(
                    env, planners, config, experiment_folder,
                    max_parallel=num_processes,
                )
            else:
                run_experiment(env, planners, config, experiment_folder)

def run_mobile_scaling(
    base_config: dict,
    parallel: bool,
    num_processes: int,
):
    for num_robots in range(1, 8):
        for num_x_boxes, num_z_boxes in [[2,2], [3,2], [3,3], [4,3], [4,4], [5,4], [5,5]]:
                np.random.seed(base_config["seed"])
                random.seed(base_config["seed"])

                total_boxes = num_x_boxes * num_z_boxes
                print(f"\n=== Scaling test: {num_robots} robots, {total_boxes} boxes === (x={num_x_boxes}, z={num_z_boxes})")

                env = rai_mobile_manip_wall(num_robots=num_robots, wall_x = num_x_boxes, wall_z = num_z_boxes)
                env.cost_reduction = base_config["cost_reduction"]
                env.cost_metric = base_config["per_agent_cost"]

                config = copy.deepcopy(base_config)
                config["experiment_name"] = "scaling"
                config["environment"] = f"mobile_r{num_robots}_x{num_x_boxes}_z{num_z_boxes}"

                planners = []
                for planner_config in config["planners"]:
                    name, planner_fn, resolved_config = setup_planner(
                        planner_config, config["max_planning_time"], config["optimize"]
                    )
                    planners.append((name, planner_fn))
                    planner_config["options"] = asdict(resolved_config)

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                experiment_folder = (
                    f"./out/{timestamp}_{config['experiment_name']}_{config['environment']}/"
                )
                os.makedirs(experiment_folder, exist_ok=True)
                export_config(experiment_folder, config)

                if parallel:
                    run_experiment_in_parallel(
                        env, planners, config, experiment_folder,
                        max_parallel=num_processes,
                    )
                else:
                    run_experiment(env, planners, config, experiment_folder)



def main():
    parser = argparse.ArgumentParser(description="Scaling test for box stacking environments")
    parser.add_argument(
        "--parallel", action="store_true", help="Run experiments in parallel (default: False)"
    )
    parser.add_argument(
        "--num_processes", type=int, default=10, help="Number of parallel processes (default: 2)"
    )
    parser.add_argument(
        "--mode", type=str, choices=["stacking", "mobile", "isolated_stacking"],
        default="stacking", help="Which scaling mode to run (default: mobile)"
    )

    args = parser.parse_args()

    base_config = copy.deepcopy(DEFAULT_CONFIG)
    base_config["planners"] = copy.deepcopy(DEFAULT_PLANNER_CONFIGS)

    if args.mode == "stacking":
        run_stacking_scaling(
            base_config=base_config,
            parallel=args.parallel,
            num_processes=args.num_processes,
        )
    elif args.mode == "isolated_stacking":
        run_isolated_stacking(
            base_config=base_config,
            parallel=args.parallel,
            num_processes=args.num_processes,
        )
    elif args.mode == "mobile":
        run_mobile_scaling(
            base_config=base_config,
            parallel=args.parallel,
            num_processes=args.num_processes,
        )
    

if __name__ == "__main__":
    main()
