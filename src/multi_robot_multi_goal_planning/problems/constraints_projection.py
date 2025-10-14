from __future__ import annotations
import numpy as np
from numpy.linalg import norm
from typing import List, Optional, Protocol, Any
from scipy.optimize import minimize

class EqualityConstraint(Protocol):
    def F(self, x: np.ndarray) -> np.ndarray: ...
    def J(self, x: np.ndarray) -> np.ndarray: ...

class RegionConstraint(Protocol):
    # g(x) <= 0 elementwise defines the feasible region
    def G(self, x: np.ndarray) -> np.ndarray: ...
    def dG(self, x: np.ndarray) -> np.ndarray: ...


def project_affine_only(
    q: Any,
    constraints: List[Any]
) -> Any:
    """
    Exact orthogonal projection onto affine subspace A q = b.
    """
    q_vec = q.state().astype(float).copy()
    A = np.vstack([c.mat for c in constraints])
    b = np.concatenate([c.constraint_pose.ravel() for c in constraints])

    # Orthogonal projection formula: q* = q - Aᵀ (A Aᵀ)^⁻¹ (A q - b)
    At = A.T
    try:
        delta = At @ np.linalg.solve(A @ At, A @ q_vec - b)
    except np.linalg.LinAlgError:
        # fallback to pseudoinverse if singular
        delta = At @ np.linalg.pinv(A @ At) @ (A @ q_vec - b)

    q_proj = q_vec - delta
    return q.from_flat(q_proj)


def project_to_manifold(
    q: Any,
    constraints: List[Any],
    eps: float = 1e-6,
    max_iters: int = 50,
    damping: float = 1e-6,
    verbose: bool = False
) -> Optional[Any]:
    """
    Projects a configuration onto the nonlinear constraint manifold defined by F_i(q)=0.
    Uses damped Gauss–Newton with adaptive step size.
    """
    q_vec = q.state().astype(float).copy()
    n = len(q_vec)

    for it in range(max_iters):
        Fs, Js = [], []
        for c in constraints:
            f = c.F(q_vec)
            J = c.J(q_vec)
            Fs.append(f)
            Js.append(J)

        if not Fs:
            return q  # no constraints

        F_all = np.concatenate(Fs)
        J_all = np.vstack(Js)

        res_norm = norm(F_all, ord=2)
        if res_norm <= eps:
            if verbose:
                print(f"[proj] converged at iter {it}, ‖F‖={res_norm:.2e}")
            return q.from_flat(q_vec)

        # Solve (JᵀJ + λI) dq = Jᵀ F  → Levenberg damping
        JTJ = J_all.T @ J_all
        rhs = J_all.T @ F_all
        try:
            dq = np.linalg.solve(JTJ + damping * np.eye(n), rhs)
        except np.linalg.LinAlgError:
            return None

        # Backtracking line search
        alpha = 1.0
        while alpha > 1e-6:
            q_trial = q_vec - alpha * dq
            F_trial = np.concatenate([c.F(q_trial) for c in constraints])
            if norm(F_trial) < res_norm:
                q_vec = q_trial
                break
            alpha *= 0.5

        if alpha <= 1e-6:
            # no progress, terminate
            if verbose:
                print(f"[proj] stalled at iter {it}, ‖F‖={res_norm:.2e}")
            return None

    if verbose:
        print(f"[proj] failed after {max_iters} iters, ‖F‖={res_norm:.2e}")
    return None


def project_nlp_sqp(
    q: Any,
    eq_constraints: List[EqualityConstraint] | None = None,
    ineq_constraints: List[RegionConstraint] | None = None,
    tol: float = 1e-8,
    max_iters: int = 200,
    verbose: bool = False,
) -> Optional[Any]:
    """
    Euclidean projection by solving:
        minimize 0.5 * ||x - x0||^2
        s.t. F_i(x) = 0  (equalities)
             G_j(x) <= 0 (inequalities)
    Uses SciPy SLSQP. Works for regions and/or manifolds. If SciPy is missing, returns None.
    """

    x0 = q.state().astype(float).copy()

    def obj(x):
        d = x - x0
        return 0.5 * float(d @ d)

    def grad(x):
        return (x - x0)

    cons = []

    if eq_constraints:
        for c in eq_constraints:
            cons.append({
                "type": "eq",
                "fun": (lambda x, c=c: c.F(x).reshape(-1)),
                "jac": (lambda x, c=c: c.J(x))
            })

    if ineq_constraints:
        for c in ineq_constraints:
            # SLSQP expects c(x) >= 0; we provide -G(x) so that G(x) <= 0 ⇒ -G(x) >= 0
            cons.append({
                "type": "ineq",
                "fun": (lambda x, c=c: -c.G(x).reshape(-1)),
                "jac": (lambda x, c=c: -c.dG(x))
            })

    res = minimize(
        obj, x0, jac=grad, method="SLSQP",
        constraints=cons,
        options={"maxiter": max_iters, "ftol": tol, "disp": verbose}
    )

    if not res.success:
        if verbose:
            print(f"[sqp] failed: {res.message}")
        return None

    return q.from_flat(res.x)
