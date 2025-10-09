import numpy as np
from typing import List, Optional
from .planning_env import State
from .configuration import config_dist, Configuration
from .constraints import AffineConfigurationSpaceEqualityConstraint

from numpy.linalg import pinv, norm

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
