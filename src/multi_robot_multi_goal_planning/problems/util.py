import random
from typing import List

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

            goal_sample = active_task.goal.sample(mode)
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
                return env.get_next_modes(q, mode)
            failed += 1

    reachable = {env.get_start_mode()}
    for _ in range(max_iter):
        next_modes = _try_transition(random.choice(tuple(reachable)))
        if next_modes is not None:
            reachable.update(next_modes)
    return tuple(reachable)


def path_cost(path: List[State], batch_cost_fun, agent_slices=None) -> float:
    """
    Computes the path cost via the batch cost function and summing it up.
    """
    if isinstance(path[0], State):
        pts = [start.q.state() for start in path]
        agent_slices = path[0].q._array_slice
        batch_costs = batch_cost_fun(pts, None, tmp_agent_slice=agent_slices)
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

    # discretize path
    for i in range(len(path) - 1):
        q0 = path[i].q
        q1 = path[i + 1].q

        # if path[i].mode != path[i + 1].mode:
        #     new_path.append(State(config_type.from_list(q), path[i].mode))
        #     continue

        dist = config_dist(q0, q1, kind)
        N = int(dist / resolution)
        N = max(1, N)

        q0_state = q0.state()
        q1_state = q1.state()
        dir = (q1_state - q0_state) / N

        for j in range(N):
            q = q0_state + dir * j
            new_path.append(State(q0.from_flat(q), path[i].mode))

    # add the final state (which is not added in the interpolation before)
    new_path.append(State(path[-1].q, path[-1].mode))

    return new_path