import numpy as np
from typing import Any, List, Optional

from .planning_env import State
from .configuration import config_dist, Configuration



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


def interpolate_path(path: List[State], resolution: float = 0.1) -> List[State]:
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

        dist = config_dist(q0, q1, "euclidean")
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


# def interpolate_path_nonlinear(
#     self,
#     path: List[State],
#     resolution: float = 0.1,
# ) -> List[State]:
#     """
#     Interpolates the path at the given resolution and PROJECTS every
#     intermediate configuration onto the active constraints of the segment's mode.
#     Assumes each segment [i, i+1] is traversed in path[i].mode.
#     """
#     new_path: List[State] = []
#     if len(path) < 2:
#         return path.copy()

#     for i in range(len(path) - 1):
#         q0 = path[i].q
#         q1 = path[i + 1].q
#         mode = path[i].mode  # segment mode = left endpoint's mode

#         # linear interpolation count
#         dist = config_dist(q0, q1, "euclidean")
#         N = max(1, int(dist / max(resolution, 1e-12)))

#         q0_state = q0.state().astype(float)
#         q1_state = q1.state().astype(float)
#         direction = (q1_state - q0_state) / N

#         # collect constraints once per segment
#         eq_env, ineq_env, eq_task, ineq_task = self.collect_constraints(mode)
#         c_eq = eq_env + eq_task
#         c_ineq = ineq_env + ineq_task

#         for j in range(N):
#             q_lin = q0_state + direction * j
#             # project
#             q_proj = self.project_nonlinear_dispatch(
#                 q0.from_flat(q_lin), c_eq, c_ineq, mode
#             )
#             q_proj_flat = np.asarray(q_proj.state(), dtype=float)
#             new_path.append(State(q0.from_flat(q_proj_flat), mode))

#     new_path.append(State(path[-1].q, path[-1].mode))

#     return new_path

def resample_on_manifold(path, step_size, planner, env):
    """
    Densify a path by interpolating ON the constraint manifold.
    This is NOT linear interpolation. It uses tangent-direction
    stepping and projects each new point back onto the manifold.

    Inputs:
        path       - List[State], sparse path after planning/shortcutting
        step_size  - maximum distance between resampled points
        planner    - must provide collect_constraints() and project_nonlinear_dispatch()
        env        - for collision checking

    Output:
        dense_path - List[State], path with small steps, valid and constraint-satisfying
    """

    if len(path) < 2:
        return path

    dense_path = [path[0]]
    q_template = path[0].q  # used to reconstruct configs from flat arrays

    for k in range(len(path) - 1):
        q0 = path[k].q
        q1 = path[k + 1].q
        mode = path[k].mode

        # Compute direction toward the next node
        v = q1.state() - q0.state()
        dist = np.linalg.norm(v)
        if dist < 1e-9:
            continue

        v /= dist  # normalize

        # Number of tangent steps
        n_steps = int(np.ceil(dist / step_size))

        q_current = q0

        # Tangent-step loop
        for s in range(1, n_steps):
            # Take a small step in the direction of q1
            q_guess = q_current.state() + v * step_size

            # === PROJECT to constraint manifold ===
            eq_aff, ineq_aff, eq_nl, ineq_nl = planner.collect_constraints(mode)
            eq = eq_aff + eq_nl
            ineq = ineq_aff + ineq_nl

            q_proj = planner.project_nonlinear_dispatch(
                q_template.from_flat(q_guess),
                eq, ineq, mode
            )

            if q_proj is None:
                # Cannot project = high curvature / bad step → stop interpolation
                break

            # Continuous collision check for this tiny step
            if not env.is_edge_collision_free(
                q_current, q_proj, mode,
                resolution=env.collision_resolution,
                tolerance=env.collision_tolerance
            ):
                break

            # Accept projected point
            dense_path.append(State(q_proj, mode))
            q_current = q_proj

        # Finally add the next original node
        dense_path.append(path[k + 1])

    return dense_path
