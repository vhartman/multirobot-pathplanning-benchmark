from simple_parsing import ArgumentParser
import numpy as np
import random
import datetime
import os

from run_experiment import export_planner_data, _execution_seed
from multi_robot_multi_goal_planning.problems import get_env_by_name
from multi_robot_multi_goal_planning.problems.planning_env import State
from multi_robot_multi_goal_planning.problems.util import interpolate_path
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    IterationTerminationCondition,
    RuntimeTerminationCondition,
)
from multi_robot_multi_goal_planning.planners import (
    RRTSkillsConservative,
    RRTSkillsConservativeConfig,
)
from multi_robot_multi_goal_planning.planners.rrt_skills_conservative import (
    replay_stochastic_path,
)

def main():
    parser = ArgumentParser(description="RRTSkillsConservative runner")

    parser.add_argument("env", nargs="?", default="default", help="env to run")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable optimization if the planner supports it. (default: False)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--run_id", type=int, default=0, help="Run id. Used for debugging only.")
    parser.add_argument(
        "--num_iters", type=int, help="Maximum number of iterations for termination."
    )
    parser.add_argument(
        "--max_time", type=float, help="Maximum runtime (in seconds) for termination."
    )
    parser.add_argument(
        "--distance_metric",
        choices=["euclidean", "sum_euclidean", "max", "max_euclidean"],
        default="max_euclidean",
        help="Distance metric to use (default: max_euclidean)",
    )
    parser.add_argument(
        "--per_agent_cost_function",
        choices=["euclidean", "max"],
        default="euclidean",
        help="Per agent cost function to use (default: euclidean)",
    )
    parser.add_argument(
        "--cost_reduction",
        choices=["sum", "max"],
        default="max",
        help="How the agent specific cost functions are reduced to one single number (default: max)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="save the computed solutions. (default: False)",
    )
    parser.add_argument(
        "--insert_transition_nodes",
        action="store_true",
        help="Insert transition nodes to ensure they are doubled. (default: False)",
    )

    parser.add_argument(
        "--viser",
        action="store_true",
        help="Show the paths using viser. (default: False)",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=5,
        help="Number of fresh noisy skill realizations to simulate (default: 5)",
    )

    parser.add_arguments(RRTSkillsConservativeConfig, dest="rrt_skills_conservative_config", prefix="rrtss.")

    args = parser.parse_args()

    if args.num_iters is not None and args.max_time is not None:
        raise ValueError("Cannot specify both num_iters and max_time.")

    np.random.seed(args.seed)
    random.seed(args.seed)

    env = get_env_by_name(args.env)
    env.cost_reduction = args.cost_reduction
    env.cost_metric = args.per_agent_cost_function

    termination_condition = None
    if args.num_iters is not None:
        termination_condition = IterationTerminationCondition(args.num_iters)
    elif args.max_time is not None:
        termination_condition = RuntimeTerminationCondition(args.max_time)

    assert termination_condition is not None

    config = args.rrt_skills_conservative_config
    config.distance_metric = args.distance_metric

    planner = RRTSkillsConservative(env, config)

    np.random.seed(args.seed + args.run_id)
    random.seed(args.seed + args.run_id)

    path, info = planner.plan(ptc=termination_condition, optimize=args.optimize)

    assert path is not None

    if args.insert_transition_nodes:
        path_w_doubled_modes = []
        for i in range(len(path)):
            path_w_doubled_modes.append(path[i])
            if i + 1 < len(path) and path[i].mode != path[i + 1].mode:
                path_w_doubled_modes.append(State(path[i].q, path[i + 1].mode))
        path = path_w_doubled_modes

    interpolated_path = interpolate_path(path, 0.05, kind="euclidean")

    path_valid = env.is_valid_plan(path)
    interpolated_valid = env.is_valid_plan(interpolated_path)
    print(
        f"[PLAN] {'VALID' if path_valid and interpolated_valid else 'INVALID'} "
        f"(original={path_valid}, interpolated={interpolated_valid})"
    )

    print("cost", info["costs"])
    print("comp_time", info["times"])

    noisy_paths = []
    noisy_labels = []
    if args.num_rollouts > 0:
        print(f"[ROLLOUTS] Simulating {args.num_rollouts} rollouts...")

        n_valid = 0
        n_endpoint_mismatch = 0
        n_time_exceeded = 0
        noisy_valid = []
        for i in range(args.num_rollouts):
            np.random.seed(_execution_seed(args.seed, i))
            random.seed(_execution_seed(args.seed, i))
            result = replay_stochastic_path(env, interpolated_path)
            status = "OK" if result.valid else f"FAILED ({result.break_reason})"
            print(f"[ROLLOUTS] rollout {i} -> {status}")
            n_valid += int(result.valid)
            noisy_valid.append(result.valid)
            if result.break_reason is not None:
                if result.break_reason.startswith("endpoint_mismatch"):
                    n_endpoint_mismatch += 1
                elif result.break_reason.startswith("time_exceeded"):
                    n_time_exceeded += 1
            noisy_paths.append(result.path)

            if result.valid:
                outcome = "OK"
            else:
                parts = result.break_reason.split(":")
                outcome = f"[FAILED] {parts[0]}"
            noisy_labels.append(f"rollout {i}: {outcome} ({len(result.path)} steps)")

        print(
            f"[ROLLOUTS] success rate: {n_valid}/{args.num_rollouts} "
            f"(endpoint mismatch: {n_endpoint_mismatch}, time exceeded: {n_time_exceeded})"
        )

    planner_paths = list(info["paths"])

    if args.save:
        info["paths"].append(interpolated_path)
        info["paths"].extend(noisy_paths)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"./out/{timestamp}_{args.env}/"
        if not os.path.isdir(experiment_folder):
            os.makedirs(experiment_folder)

        planner_folder = experiment_folder + "rrt_stochastic_skills/"
        export_planner_data(planner_folder, 0, info)

        # Export executions for plotting ECDF properly
        executions = []
        for path, valid in zip(noisy_paths, noisy_valid):
            if not valid:
                executions.append({"cost": None, "success": False})
            else:
                c = 0.0
                for a, b in zip(path, path[1:]):
                    c += float(env.batch_config_cost(a.q, np.asarray([b.q.state()]))[0])
                executions.append({"cost": c, "success": True})

        run_dir = os.path.join(planner_folder, "0")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "executions.json"), "w") as f:
            import json
            json.dump(executions, f)

    if args.viser:
        viser_paths = planner_paths + [interpolated_path] + noisy_paths
        viser_labels = (
            [f"planner path {i} ({len(p)} steps)" for i, p in enumerate(planner_paths)]
            + [f"nominal plan ({len(interpolated_path)} steps)"]
            + noisy_labels
        )
        env.display_path_viser(
            paths=viser_paths,
            path_labels=viser_labels,
            primitives_only=True,
        )

    if hasattr(env, "close"):
        env.close()

if __name__ == "__main__":
    main()
