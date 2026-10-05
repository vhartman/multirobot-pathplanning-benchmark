import dataclasses
import random
from typing import Any, Dict, List, Tuple

import numpy as np

from multi_robot_multi_goal_planning.problems.util import path_cost, skill_edge_seconds
from multi_robot_multi_goal_planning.planners.rrt_skills_conservative import replay_stochastic_path
from multi_robot_multi_goal_planning.planners.rrt_skills_reactive import ReactiveRoadmapConfig
from multi_robot_multi_goal_planning.planners.reactive_mdp import ReactiveMDPConfig
from multi_robot_multi_goal_planning.planners.reactive_policy import ReactiveExecutor
from multi_robot_multi_goal_planning.planners.policy_shortcut import PolicyShortcutConfig

@dataclasses.dataclass
class ReactiveExperimentConfig:
    roadmap: Dict[str, Any]
    mdp: Dict[str, Any]
    shortcut: Dict[str, Any]
    batched: Dict[str, Any]
    num_executions: int

def draw_execution_base_seed() -> int:
    """Draw a base seed for the executions of one planner run"""
    return int(np.random.randint(0, 2 ** 31 - 1))

def _execution_seed(base_seed: int, index: int) -> int:
    """Derive a repeatable seed for one execution index"""
    return int(np.random.SeedSequence([int(base_seed), int(index)]).generate_state(1)[0])

def _execution_record(env, path, index, reached_goal, break_reason, extra=None) -> Dict:
    def _entry(cost, real_time, mode, step):
        ids = list(mode.task_ids)
        task_types = []
        for r_idx, task_id in enumerate(ids):
            if 0 <= task_id < len(env.tasks):
                task = env.tasks[task_id]
                if getattr(task, "skill", None) is not None:
                    robot_name = env.robots[r_idx]
                    if robot_name in getattr(task, "robots", []):
                        task_types.append("active_skill")
                    else:
                        task_types.append("inactive_skill")
                else:
                    task_types.append("transit")
            else:
                task_types.append("transit")

        return {"time": cost, "real_time": real_time, "step": step, "tasks": ids,
                "task_names": [env.tasks[i].name if 0 <= i < len(env.tasks) else str(i)
                               for i in ids],
                "task_types": task_types}

    timeline = []
    state_timestamps = []
    if path and len(path) > 1:
        current_cost = 0.0
        current_real = 0.0
        timeline.append(_entry(0.0, 0.0, path[0].mode, 0))
        state_timestamps.append(0.0)
        v_ref = getattr(env, "v_ref", 1.0)
        for i in range(1, len(path)):
            cost_step = float(env.batch_config_cost([path[i-1]], [path[i]])[0])
            current_cost += cost_step
            edge_seconds = skill_edge_seconds(env, path[i-1], path[i])
            current_real += edge_seconds if edge_seconds is not None else cost_step / v_ref
            state_timestamps.append(current_real)
            if path[i].mode != path[i-1].mode:
                timeline.append(_entry(current_cost, current_real, path[i].mode, i))
        timeline.append(_entry(current_cost, current_real, path[-1].mode, len(path)-1))

    valid = bool(path is not None and len(path) > 1 and env.is_valid_plan(path))
    if break_reason is None and reached_goal and not valid:
        break_reason = "post_validation_invalid"

    record = {
        "index": index,
        "reached_goal": bool(reached_goal),
        "valid": valid,
        "cost": float(path_cost(path, env.batch_config_cost, env=env)) if path and len(path) > 1 else None,
        "break_reason": break_reason,
        "timeline": timeline,
        "state_timestamps": state_timestamps,
    }
    record["success"] = record["reached_goal"] and record["valid"]
    record.update(extra or {})
    return record

def _skill_seconds_by_task(env, steps_by_task: Dict[str, int]) -> Dict[str, float]:
    """Convert realized skill step counts to seconds, preserving each task separately"""
    dt = {t.name: getattr(getattr(t, "skill", None), "dt", 0.0) for t in env.tasks}
    return {name: n * dt.get(name, 0.0) for name, n in (steps_by_task or {}).items()}

def _skill_seconds(env, steps_by_task: Dict[str, int]) -> float:
    """Longest realized skill duration in SECONDS. The max-reduction of _skill_seconds_by_task"""
    return max(_skill_seconds_by_task(env, steps_by_task).values(), default=0.0)

def evaluate_open_loop(env, path, n_executions: int, base_seed: int) -> Tuple[List[Dict], List]:
    """Evaluate one open-loop plan over fresh stochastic realizations"""
    records, paths = [], []
    for i in range(n_executions):
        np.random.seed(_execution_seed(base_seed, i))
        random.seed(_execution_seed(base_seed, i))
        result = replay_stochastic_path(env, path)
        records.append(_execution_record(
            env, result.path, i, reached_goal=result.valid, break_reason=result.break_reason,
            extra={"skill_duration": _skill_seconds(env, result.skill_steps),
                   "skill_seconds_by_task": _skill_seconds_by_task(env, result.skill_steps)},
        ))
        paths.append(result.path)
    return records, paths

def evaluate_reactive_policy(env, mdp, n_executions: int, base_seed: int) -> Tuple[List[Dict], List]:
    """Executes the reactive policy closed-loop against n_executions fresh realizations"""
    executor = ReactiveExecutor(mdp)
    records, paths = [], []
    for i in range(n_executions):
        np.random.seed(_execution_seed(base_seed, i))
        random.seed(_execution_seed(base_seed, i))
        run = executor.run()
        records.append(_execution_record(
            env, run.path, i, reached_goal=run.reached_goal, break_reason=run.failure,
            extra={"skill_duration": float(run.skill_seconds),
                   "skill_seconds_by_task": {k: float(v)
                                             for k, v in run.skill_seconds_by_task.items()},
                   "skill_epochs": int(run.skill_epochs),
                   "planner_cost": float(run.cost) if np.isfinite(run.cost) else None},
        ))
        paths.append(run.path)

    return records, paths

def build_reactive_configs(options, runtime, cost_model, problem_env=None):
    """Build the roadmap, MDP, shortcut, and batched reactive configurations"""
    from multi_robot_multi_goal_planning.planners.reactive_batched import (
        BatchedConfig,
    )

    roadmap_fields = {f.name for f in dataclasses.fields(ReactiveRoadmapConfig)}
    mdp_fields = {f.name for f in dataclasses.fields(ReactiveMDPConfig)}
    shortcut_fields = {f.name for f in dataclasses.fields(PolicyShortcutConfig)}
    batched_fields = {f.name for f in dataclasses.fields(BatchedConfig)}
    shortcut_options = {
        key[len("shortcut_"):]: value
        for key, value in options.items()
        if key.startswith("shortcut_") and key[len("shortcut_"):] in shortcut_fields
    }
    consumed = set(roadmap_fields) | set(mdp_fields) | set(batched_fields)
    consumed |= {"shortcut_" + key for key in shortcut_options}
    unknown = set(options) - consumed
    if unknown:
        raise ValueError(f"Unknown reactive planner options: {sorted(unknown)}")

    roadmap_config = ReactiveRoadmapConfig(
        **{key: value for key, value in options.items() if key in roadmap_fields}
    )
    if "rrg_runtime" not in options:
        roadmap_config.rrg_runtime = runtime

    mdp_kwargs = {key: value for key, value in options.items() if key in mdp_fields}
    mdp_kwargs.setdefault("objective", "time" if cost_model == "time" else "geometric")
    if mdp_kwargs.get("objective") == "time":
        mdp_kwargs.setdefault("validate_exit_edges", True)

    mdp_config = ReactiveMDPConfig(**mdp_kwargs)
    batched_config = BatchedConfig(
        **{key: value for key, value in options.items() if key in batched_fields}
    )
    roadmap_config.build_shortcut = batched_config.growth_shortcutting
    if problem_env is not None:
        from multi_robot_multi_goal_planning.planners.reactive_batched import (
            prepare_batched_mdp_config,
        )
        mdp_config = prepare_batched_mdp_config(
            problem_env, mdp_config, batched_config
        )

    return roadmap_config, mdp_config, PolicyShortcutConfig(**shortcut_options), batched_config
