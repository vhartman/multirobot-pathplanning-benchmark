import numpy as np
from typing import List, Optional

from .planning_env import State
from .configuration import config_dist, Configuration
from .constraints import AffineConfigurationSpaceEqualityConstraint

from numpy.linalg import pinv, norm


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


def project_to_manifold(
    q: Configuration,
    constraints: List,
    eps: float = 1e-6,
    max_iters: int = 50
) -> Optional[Configuration]:
    """
    Projects a configuration onto the constraint manifold defined by
    the given constraints, using an iterative Gauss-Newton style update.

    Args:
        q: Configuration to project
        constraints: list of constraint objects, each providing .F(q_vec) and .J(q_vec)
        eps: tolerance for constraint satisfaction
        max_iters: maximum number of iterations

    Returns:
        A new Configuration lying on the manifold within tolerance,
        or None if projection failed to converge.
    """
    q_vec = q.state().copy()
    n = len(q_vec)

    for _ in range(max_iters):
        # Collect residuals and Jacobians from all constraints
        Fs = []
        Js = []
        for c in constraints:
            f = c.F(q_vec)       # residual vector, shape (k,)
            J = c.J(q_vec)       # Jacobian matrix, shape (k, n)
            Fs.append(f)
            Js.append(J)

        if not Fs:  # no constraints -> nothing to project
            return q

        F_all = np.concatenate(Fs)
        J_all = np.vstack(Js)

        # Check if constraints are already satisfied
        if norm(F_all) <= eps:
            return q.from_flat(q_vec)

        try:
            dq = pinv(J_all) @ F_all
        except np.linalg.LinAlgError:
            return None

        q_vec = q_vec - dq

    # If we exit the loop, projection did not converge
    return None
    
   
def project_affine_only(q: Configuration, constraints: List[AffineConfigurationSpaceEqualityConstraint]) -> Configuration:
    q_vec = q.state().copy()
    A = np.vstack([c.mat for c in constraints])
    b = np.concatenate([c.constraint_pose.ravel() for c in constraints])
    dq = np.linalg.pinv(A) @ (A @ q_vec - b)
    q_proj = q_vec - dq
    return q.from_flat(q_proj)
