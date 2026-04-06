import argparse
import copy
import datetime
import os
import random
import sys

import numpy as np
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_experiment import (
    setup_planner,
    run_experiment,
    run_experiment_in_parallel,
    export_config,
    load_experiment_config,
)

from multi_robot_multi_goal_planning.problems.rai_envs import rai_ur10_arm_box_stack_env


DEFAULT_PLANNER_CONFIGS = [
    {"name": "rrt", "type": "birrtstar", "options": {}},
    {"name": "prioritized", "type": "prioritized", "options": {}},
]

DEFAULT_CONFIG = {
    "seed": 2,
    "num_runs": 5,
    "optimize": False,
    "max_planning_time": 100,
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
            np.random.seed(base_config["seed"])
            random.seed(base_config["seed"])

            print(f"\n=== Scaling test: {num_robots} robots, {num_boxes} boxes ===")

            env = rai_ur10_arm_box_stack_env(num_robots=num_robots, num_boxes=num_boxes)
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


def main():
    parser = argparse.ArgumentParser(description="Scaling test for box stacking environments")
    parser.add_argument(
        "--parallel", action="store_true", help="Run experiments in parallel (default: False)"
    )
    parser.add_argument(
        "--num_processes", type=int, default=2, help="Number of parallel processes (default: 2)"
    )

    args = parser.parse_args()

    base_config = copy.deepcopy(DEFAULT_CONFIG)
    base_config["planners"] = copy.deepcopy(DEFAULT_PLANNER_CONFIGS)

    run_stacking_scaling(
        base_config=base_config,
        parallel=args.parallel,
        num_processes=args.num_processes,
    )


if __name__ == "__main__":
    main()
