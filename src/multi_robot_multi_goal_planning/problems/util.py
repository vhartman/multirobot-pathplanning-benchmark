import random
from typing import List, Optional

import numpy as np

from .planning_env import BaseProblem, Mode, State
from .core.configuration import config_dist


def compute_reachable_modes(env: BaseProblem, max_iter: int = 500) -> tuple[Mode, ...]:
    """Sample reachable modes by repeatedly trying to transition from known modes."""
    conf_type = type(env.get_start_pos())

    def _try_transition(mode: Mode):
        if env.is_terminal_mode(mode):
            return None
        failed = 0
        while True:
            if failed > 1000:
                return None
            combos = env.get_valid_next_task_combinations(mode)
            if combos:
                active_task = env.get_active_task(mode, combos[random.randint(0, len(combos) - 1)])
            else:
                active_task = env.get_active_task(mode, None)

            if getattr(active_task, "skill", None) is not None:
                q_init = active_task.initiation_goal.sample(mode)
                
                active_task.skill.joints = []
                for r in active_task.robots:
                    active_task.skill.joints.extend(env.robot_joints[r])
                all_joints = []
                for r in env.robots:
                    all_joints.extend(env.robot_joints[r])
                    
                env.C.selectJoints(active_task.skill.joints)
                skill_result = active_task.skill.rollout(q_init, active_task, all_joints, env, 0)
                env.C.selectJoints(all_joints)
                goal_sample = skill_result.trajectory[-1]
                completed_task_ids = [env.tasks.index(active_task)]
            else:
                goal_sample = active_task.goal.sample(mode)
                completed_task_ids = None

            q = env.sample_config_uniform_in_limits()

            for i, r in enumerate(env.robots):
                if r in active_task.robots:
                    offset = 0
                    for task_robot in active_task.robots:
                        if task_robot == r:
                            q[i] = goal_sample[offset: offset + env.robot_dims[task_robot]]
                            break
                        offset += env.robot_dims[task_robot]

            if env.is_collision_free(q, mode):
                return env.get_next_modes(q, mode, completed_task_ids=completed_task_ids)
            failed += 1

    reachable = {env.get_start_mode()}
    for _ in range(max_iter):
        next_modes = _try_transition(random.choice(tuple(reachable)))
        if next_modes is not None:
            reachable.update(next_modes)
    return tuple(reachable)


def skill_edge_seconds(env: BaseProblem, s_from: State, s_to: State) -> Optional[float]:
    """
    Return the physical duration represented by a skill edge, or ``None`` for transit edges.
    Explicit skill-step deltas avoid double-counting duplicate boundary states; unnumbered
    skill waypoints charge one skill timestep when they move.
    """
    if not (getattr(s_from, "is_skill_waypoint", False) and getattr(s_to, "is_skill_waypoint", False)):
        return None
    if s_from.mode != s_to.mode:
        return None
    active_tasks = [env.tasks[t] for t in dict.fromkeys(s_to.mode.task_ids)
                    if getattr(env.tasks[t], "skill", None) is not None]
    if not active_tasks:
        return None

    moved = not np.allclose(s_from.q.state(), s_to.q.state())
    charges = []
    for task in active_tasks:
        dt = task.skill.dt
        steps_from = s_from.skill_steps.get(task.name)
        steps_to = s_to.skill_steps.get(task.name)
        if steps_from is not None and steps_to is not None:
            delta = steps_to - steps_from
            charges.append(delta * dt if delta != 0 else (dt if moved else 0.0))
        else:
            charges.append(dt if moved else 0.0)
    return max(charges)


def path_cost(path: List[State], batch_cost_fun, agent_slices=None, env: Optional[BaseProblem] = None) -> float:
    """
    Sum path costs, replacing skill-edge costs with measured duration for time-based environments.
    """
    if isinstance(path[0], State):
        pts = [start.q.state() for start in path]
        agent_slices = path[0].q._array_slice
        batch_costs = np.asarray(batch_cost_fun(pts, None, tmp_agent_slice=agent_slices), dtype=float)
        if env is not None and getattr(env, "cost_model", "geometric") == "time":
            for i, (a, b) in enumerate(zip(path, path[1:])):
                seconds = skill_edge_seconds(env, a, b)
                if seconds is not None:
                    batch_costs[i] = seconds * env.v_ref
    elif isinstance(path[0], np.ndarray) and agent_slices is not None:
        batch_costs = batch_cost_fun(path, None, tmp_agent_slice=agent_slices)
    else:
        raise ValueError("Arguments to path cost seem to be wrong.")
        
    # batch_costs = batch_cost_fun(path, None)
    # assert np.allclose(batch_costs, batch_costs_tmp)

    return np.sum(batch_costs)


def interpolate_path(path: List[State], resolution: float = 0.1, kind="max") -> List[State]:
    """
    Takes a path and interpolates it at the given resolution.
    Uses the euclidean distance between states to do the resolution.
    """
    new_path = []

    # Discretize path
    for i in range(len(path) - 1):
        q0 = path[i].q
        q1 = path[i + 1].q

        
        is_skill = getattr(path[i], 'is_skill_waypoint', False)
        next_is_skill = getattr(path[i + 1], 'is_skill_waypoint', False)
        skill_steps = dict(getattr(path[i], 'skill_steps', {}))

        # Edge inside skill mode -> keep the waypoint as is, no interpolation
        if is_skill and next_is_skill:
            new_path.append(State(q0.from_flat(q0.state()), path[i].mode, is_skill_waypoint=True, skill_steps=skill_steps))
        else:
            # Edge outside skill mode (free-space AND skill-exit) -> must interpolate
            dist = config_dist(q0, q1, kind)
            N = int(dist / resolution)
            N = max(1, N)

            q0_state = q0.state()
            q1_state = q1.state()
            dir = (q1_state - q0_state) / N

            for j in range(N):
                q = q0_state + dir * j
                # Mark exit config (j==0) as skill-waypoint, but not the interpolated points
                first_is_skill = is_skill and j == 0
                new_path.append(State(q0.from_flat(q), path[i].mode,
                                      is_skill_waypoint=first_is_skill,
                                      skill_steps=(skill_steps if first_is_skill else {})))

    # Add the final state (which is not added in the interpolation before)
    final_is_skill = getattr(path[-1], 'is_skill_waypoint', False)
    final_skill_steps = dict(getattr(path[-1], 'skill_steps', {}))
    final_q = path[-1].q.from_flat(path[-1].q.state())
    new_path.append(State(final_q, path[-1].mode, final_is_skill, skill_steps=final_skill_steps))
    
    # TODO DEBUG (remove)
    counter = sum(1 for s in new_path if getattr(s, 'is_skill_waypoint', False))
    print(f"[DEBUG INTERPOLATE] There are {counter} skill points in the new_path")

    return new_path