import traceback
import datetime
import os

import random
import numpy as np

import importlib
import robotic

from run_experiment import run_experiment_in_parallel, export_config

from multi_robot_multi_goal_planning.problems import (
    get_all_environments,
    get_env_by_name,
)

from multi_robot_multi_goal_planning.planners import (
    BaseRRTConfig,
    BidirectionalRRTstar,
    PrioritizedPlannerConfig,
    PrioritizedPlanner,
)

from multi_robot_multi_goal_planning.planners.termination_conditions import (
    RuntimeTerminationCondition,
)


def make_planners(env, runtime):
    def birrt_planner(env):
        config = BaseRRTConfig()
        config.distance_metric = "max_euclidean"
        return BidirectionalRRTstar(env, config=config).plan(
            RuntimeTerminationCondition(runtime), optimize=True
        )

    def prio_planner(env):
        config = PrioritizedPlannerConfig()
        return PrioritizedPlanner(env, config).plan(
            RuntimeTerminationCondition(runtime), optimize=True
        )

    planners = [("birrt", birrt_planner), ("prio", prio_planner)]
    return planners


def main():
    # all_envs = get_all_environments()
    representative_envs = [
        "rai.simple",
        "rai.one_agent_many_goals",
        "rai.two_agents_many_goals_dep",
        "rai.three_agent_many_goals",
        "rai.single_agent_mover",
        "rai.piano_dep",
        "rai.2d_handover",
        "rai.random_2d",
        "rai.other_hallway",
        "rai.three_agents",
        "rai.triple_waypoints",
        "rai.welding",
        "rai.handover",
        "rai.eggs",
        "rai.bottles",
        "rai.box_rearrangement",
        "rai.box_reorientation",
        "rai.box_reorientation_dep",
        "rai.pyramid",
        "rai.box_stacking",
        "rai.box_stacking_dep",
        "rai.mobile_wall_four",
        "rai.mobile_wall_four_dep",
        "rai.mobile_strut",
        "rai.three_robot_truss",
        "rai.spiral_tower",
        "rai.spiral_tower_two",
        "rai.cube_four",
        "rai.unordered_box_reorientation",
        "rai.unassigned_two_dim",
        "rai.unassigned_cleanup",
        "rai.unassigned_stacking",
        "rai.unordered_bottles",
    ]

    num_parallel = 10
    num_runs = 5
    max_runtime = 500

    seed = 0

    for env_name in representative_envs:
        try:
            importlib.reload(robotic)
            
            print(f"Running env {env_name}")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = "convergence"

            # convention: alsways use "/" as trailing character
            experiment_folder = f"./out/{timestamp}_{experiment_name}_{env_name}/"

            if not os.path.isdir(experiment_folder):
                os.makedirs(experiment_folder)

            config = {}
            config["seed"] = seed
            config["num_runs"] = num_runs
            config["environment"] = env_name
            config["max_planning_time"] = max_runtime
            config["planners"] = []

            export_config(experiment_folder, config)

            np.random.seed(seed)
            random.seed(seed)

            env = get_env_by_name(env_name)

            planners = make_planners(env, max_runtime)

            _ = run_experiment_in_parallel(
                env, planners, config, experiment_folder, max_parallel=num_parallel
            )
        except Exception as e:
            print(traceback.format_exc())


if __name__ == "__main__":
    main()
