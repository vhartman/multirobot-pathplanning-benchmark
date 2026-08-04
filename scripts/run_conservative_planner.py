"""
run_conservative_planner.py
===========================
The stochastic counterpart of run_planner.py

Runs the conservative open-loop stochastic planner RRTSkillsConservative on a given environment

Extras over run_planner.py:
  - No --planner selector: always RRTSkillsConservative
  - New --num_rollouts: number of fresh noisy skill realizations to simulate after planning
  - No post-hoc shortcutting in the script (handled internally by the planner's _shortcut)
"""
from simple_parsing import ArgumentParser
import numpy as np
import random
import datetime
import os
import copy
from dataclasses import dataclass
from typing import Optional, List

from run_experiment import export_planner_data
from multi_robot_multi_goal_planning.problems import get_env_by_name
from multi_robot_multi_goal_planning.problems.planning_env import State
from multi_robot_multi_goal_planning.problems.util import interpolate_path
from multi_robot_multi_goal_planning.problems.skills import (
    BaseStochasticTimedSkill,
    StochasticBaseSkill
)
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    IterationTerminationCondition,
    RuntimeTerminationCondition,
)
from multi_robot_multi_goal_planning.planners import (
    RRTSkillsConservative,
    RRTSkillsConservativeConfig,
)

# =====================================================================
# Open-Loop Stochastic Replay Helpers
# =====================================================================
def stochastic_task_indices(env, task) -> np.ndarray:
    """
    Computes which indices of the combined state array belong to the robots executing the given task
    Used as during replay we only want to overwrite the active robot's joints with fresh rollout 
    """
    idx = []
    end = 0
    for robot in env.robots:
        dim = env.robot_dims[robot]
        if robot in task.robots:
            idx.extend(range(end, end + dim))
        end += dim
    return np.array(idx)

def _rollout_fresh(env, task, q_init: np.ndarray) -> np.ndarray:
    """
    Executes one fresh noisy realization of task.skill from q_init
    Replay only checks the realized endpoint against the plan (see replay_stochastic_path),
    so no branch targeting is needed here - any realization is accepted as-is
    """
    skill = task.skill
    result = skill.rollout(q_init, task, env.get_joint_names(), env, t0=0.0)
    return result.trajectory

@dataclass
class ReplayResult:
    path: List[State]
    valid: bool = True
    break_reason: Optional[str] = None

# Max joint-space distance between realized and planned skill endpoints
_ENDPOINT_TOL = 1e-2

def replay_stochastic_path(env, path: List[State]) -> ReplayResult:
    """
    Simulates real-world execution of a conservative open-loop plan under one
    fresh noisy realization of each stochastic skill

    Two failure modes are detected:
      - time_exceeded: the realized rollout needed more steps than the budgeted segment
      - endpoint_mismatch: the active robot landed more than _ENDPOINT_TOL away from
        where the plan expects it, meaning the rest of the plan is not valid

    The conservative planner already guaranteed collision safety for any intermediate
    path (it checked all MC rollouts during planning), so a different intermediate
    trajectory is always safe, only the final configuration matters
    """
    stoc_tasks = [
        t for t in env.tasks
        if getattr(t, "skill", None) is not None
        and isinstance(t.skill, (BaseStochasticTimedSkill, StochasticBaseSkill))
    ]
    replayed = copy.deepcopy(path)

    if not stoc_tasks:
        return ReplayResult(path=replayed, valid=True, break_reason=None)

    for task in stoc_tasks:
        idx = stochastic_task_indices(env, task)
        seg_indices = [
            i for i, s in enumerate(replayed)
            if getattr(s, "is_skill_waypoint", False)
            and task.name in getattr(s, "skill_steps", {})
        ]
        if not seg_indices:
            continue

        seg_states = [replayed[i] for i in seg_indices]
        init_state = replayed[seg_indices[0] - 1] if seg_indices[0] > 0 else seg_states[0]
        q_init = np.asarray(init_state.q.state(), dtype=np.float64)[idx]

        fresh_traj = _rollout_fresh(env, task, q_init)

        fresh_segment = []
        for s_plan in seg_states:
            step_idx = s_plan.skill_steps.get(task.name, 0)
            # If rollout completed earlier than budgeted, hold at the final configuration
            active_q = fresh_traj[min(step_idx, len(fresh_traj) - 1)]
            q = np.asarray(s_plan.q.state(), dtype=np.float64).copy()
            q[idx] = active_q
            fresh_segment.append(
                State(
                    env.get_start_pos().from_flat(q),
                    s_plan.mode,
                    is_skill_waypoint=True,
                    skill_steps=dict(s_plan.skill_steps),
                )
            )

        prefix = replayed[:seg_indices[0]]
        suffix = replayed[seg_indices[-1] + 1:]

        # fresh_traj[0] = q_init, so number of actual steps = len - 1
        time_exceeded = (len(fresh_traj) - 1) > len(seg_states)

        # Check if the active robot landed at the expected endpoint
        # The conservative planner already guaranteed safety for any intermediate path
        # (checked all MC rollouts during planning), so only the final config matters
        committed_endpoint = np.asarray(seg_states[-1].q.state(), dtype=np.float64)[idx]
        endpoint_dist = np.linalg.norm(fresh_traj[-1] - committed_endpoint)
        endpoint_mismatch = endpoint_dist > _ENDPOINT_TOL

        if time_exceeded or endpoint_mismatch:
            reason = (
                f"time_exceeded:{task.name}" if time_exceeded
                else f"endpoint_mismatch:{task.name}:dist={endpoint_dist:.4f}"
            )
            print(f"[REPLAY] '{task.name}' open-loop plan failed: {reason}. Halting execution.")
            return ReplayResult(path=prefix + fresh_segment, valid=False, break_reason=reason)

        replayed = prefix + fresh_segment + suffix if suffix else prefix + fresh_segment

    # Final collision check on the full spliced trajectory
    if not env.is_valid_plan(replayed):
        print("[REPLAY] Collision detected in physical execution realization.")
        return ReplayResult(path=replayed, valid=False, break_reason="collision")

    return ReplayResult(path=replayed, valid=True, break_reason=None)

# =====================================================================
# Main Runner
# =====================================================================
def main():
    parser = ArgumentParser(description="RRTSkillsConservative runner")

    parser.add_argument("env", nargs="?", default="default", help="env to run")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable optimization if the planner supports it. (default: False)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed")
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
        for i in range(args.num_rollouts):
            result = replay_stochastic_path(env, interpolated_path)
            status = "OK" if result.valid else f"FAILED ({result.break_reason})"
            print(f"[ROLLOUTS] rollout {i} -> {status}")
            n_valid += int(result.valid)
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
