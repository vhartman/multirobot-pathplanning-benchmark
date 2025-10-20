from __future__ import annotations
import numpy as np
from numpy.linalg import norm, LinAlgError
from typing import List, Optional, Protocol, Any
from scipy.optimize import minimize
from .constraints import AffineConfigurationSpaceInequalityConstraint, AffineConfigurationSpaceEqualityConstraint

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


def project_affine_cspace(
    q,
    eq_constraints=None,
    ineq_constraints=None,
    eps: float = 1e-8,
    max_iters: int = 50,
):
    """
    Euclidean projection of configuration q onto the feasible affine set:
        A_eq x = b_eq
        A_in x <= b_in

    Handles four cases automatically:
        1) No constraints
        2) Only affine equalities
        3) Only affine inequalities
        4) Both affine equalities and inequalities

    Args:
        q: configuration object with .state() and .from_flat()
        eq_constraints: list of affine equality constraints (must have .mat, .rhs)
        ineq_constraints: list of affine inequality constraints (must have .mat, .rhs)
        eps: tolerance for feasibility
        max_iters: maximum active-set iterations
        verbose: print debug info

    Returns:
        Projected configuration q_proj or None if failed.
    """

    x = q.state().astype(float).copy()
    n = len(x)

    # Collect matrices
    if not ineq_constraints:
        if not eq_constraints:
            # case 1: no constraints
            return q  # no constraints
        
        else:
            # case 2: only equalities
            q_proj = project_affine_only(q, eq_constraints)
            return q_proj
    
    else:
        Aineq = np.vstack([c.mat for c in ineq_constraints])
        bineq = np.concatenate([c.rhs.ravel() for c in ineq_constraints])

        if not eq_constraints:
            # case 3: only inequalities
            # Active-set Euclidean projection
            active = np.where(Aineq @ x - bineq > eps)[0].tolist()
            for _ in range(max_iters):
                if not active:
                    return q.from_flat(x)
                Aact = Aineq[active]
                bact = bineq[active]
                try:
                    # Project onto active set
                    KKT = np.block([[np.eye(n), Aact.T],
                                    [Aact, np.zeros((len(active), len(active)))]])
                    rhs = np.concatenate([x.ravel(), bact.ravel()])
                    sol = np.linalg.solve(KKT, rhs)
                    x_proj = sol[:n]
                except LinAlgError:
                    return None

                # Check feasibility
                viol = np.where(Aineq @ x_proj - bineq > eps)[0].tolist()
                if not viol:
                    return q.from_flat(x_proj)

                # Add new active constraints
                active = sorted(list(set(active + viol)))
                x = x_proj

        else:
            # case 4: both equalities and inequalities

            Aeq = np.vstack([c.mat for c in eq_constraints])
            beq = np.concatenate([c.rhs.ravel() for c in eq_constraints])
            m_eq = Aeq.shape[0]

            active = np.where(Aineq @ x - bineq > eps)[0].tolist()

            for _ in range(max_iters):
                Aact = Aineq[active] if active else np.zeros((0, n))
                bact = bineq[active] if active else np.zeros(0)
                try:
                    # Build KKT system
                    KKT = np.block([
                        [np.eye(n), Aeq.T, Aact.T],
                        [Aeq, np.zeros((m_eq, m_eq + len(active)))],
                        [Aact, np.zeros((len(active), m_eq + len(active)))]
                    ])
                    rhs = np.concatenate([x.ravel(), beq.ravel(), bact.ravel()])
                    sol = np.linalg.solve(KKT, rhs)
                    x_proj = sol[:n]
                except LinAlgError:
                    return None

                # Check feasibility
                resid_eq = norm(Aeq @ x_proj - beq)
                viol = np.where(Aineq @ x_proj - bineq > eps)[0].tolist()

                if resid_eq < eps and not viol:
                    return q.from_flat(x_proj)

                # Update active set
                active = sorted(list(set(active + viol)))
                x = x_proj

    return None


def project_affine_cspace_interior(
    q,
    eq_constraints=None,
    ineq_constraints=None,
    eps: float = 1e-8,
    max_iters: int = 50,
    alpha_range=(0.01, 0.5),
):
    """
    Stochastic interior projection: same as project_affine_cspace,
    but after projecting, adds a random inward displacement so that
    the final point lies *strictly inside* the feasible region.

    Args:
        q: configuration object with .state() and .from_flat()
        eq_constraints: list of affine equality constraints (with .mat, .rhs)
        ineq_constraints: list of affine inequality constraints (with .mat, .rhs)
        eps: feasibility tolerance
        max_iters: max active-set iterations for the main projection
        alpha_range: range of random inward step magnitude (fraction of slack distance)

    Returns:
        Projected interior configuration q_interior or None if projection fails.
    """

    x = q.state().astype(float).copy()
    n = len(x)

    # === 1) Collect constraints ===
    if not ineq_constraints:
        if not eq_constraints:
            return q  # unconstrained
        else:
            # Only equalities: pure orthogonal projection
            q_proj = project_affine_only(q, eq_constraints)
            return q_proj

    # Stack inequalities
    Aineq = np.vstack([c.mat for c in ineq_constraints])
    bineq = np.concatenate([c.rhs.ravel() for c in ineq_constraints])

    # === 2) Standard projection (same as project_affine_cspace) ===
    if not eq_constraints:
        # Only inequalities
        active = np.where(Aineq @ x - bineq > eps)[0].tolist() # detects active (violated) constraints
        for _ in range(max_iters):
            if not active:
                x_proj = x
                break
            Aact = Aineq[active]
            bact = bineq[active]
            try: # we find the projection onto the active set
                KKT = np.block([[np.eye(n), Aact.T],
                                [Aact, np.zeros((len(active), len(active)))]])
                rhs = np.concatenate([x.ravel(), bact.ravel()])
                sol = np.linalg.solve(KKT, rhs)
                x_proj = sol[:n]
            except LinAlgError:
                return None

            viol = np.where(Aineq @ x_proj - bineq > eps)[0].tolist() # check if we are still violating sth
            if not viol:
                break
            active = sorted(list(set(active + viol)))
            x = x_proj
    else:
        # Both equalities and inequalities
        Aeq = np.vstack([c.mat for c in eq_constraints])
        beq = np.concatenate([c.rhs.ravel() for c in eq_constraints])
        m_eq = Aeq.shape[0]
        active = np.where(Aineq @ x - bineq > eps)[0].tolist()

        for _ in range(max_iters):
            Aact = Aineq[active] if active else np.zeros((0, n))
            bact = bineq[active] if active else np.zeros(0)
            try:
                KKT = np.block([
                    [np.eye(n), Aeq.T, Aact.T],
                    [Aeq, np.zeros((m_eq, m_eq + len(active)))],
                    [Aact, np.zeros((len(active), m_eq + len(active)))]
                ])
                rhs = np.concatenate([x.ravel(), beq.ravel(), bact.ravel()])
                sol = np.linalg.solve(KKT, rhs)
                x_proj = sol[:n]
            except LinAlgError:
                return None

            resid_eq = norm(Aeq @ x_proj - beq)
            viol = np.where(Aineq @ x_proj - bineq > eps)[0].tolist()
            if resid_eq < eps and not viol:
                break
            active = sorted(list(set(active + viol)))
            x = x_proj

    # If projection failed
    if x_proj is None:
        return None

    # === 3) Compute random inward displacement ===
    slack = bineq - Aineq @ x_proj  # positive inside, 0 = active
    active = np.where(slack <= eps)[0]
    inactive = np.where(slack > eps)[0]

    if len(ineq_constraints) == 0 or (slack > eps).all():
        # Already interior, nothing to do
        return q.from_flat(x_proj)

    # Build an inward direction (combination of active constraint normals)
    if len(active) > 0:
        n_inward = -np.sum(Aineq[active], axis=0)
        if norm(n_inward) > 0:
            n_inward /= norm(n_inward)
        else:
            n_inward = np.random.randn(n)
            n_inward /= norm(n_inward)
    else:
        return q.from_flat(x_proj)

    # Random step magnitude proportional to minimum slack of inactive constraints
    min_slack = np.min(slack[inactive]) if len(inactive) > 0 else eps
    alpha = np.random.uniform(*alpha_range) * min_slack

    x_interior = x_proj + alpha * n_inward

    # === 4) Checking if all constraints are satisfied after inward displacement , and reprojection if needed ===
    viol_ineq = np.where(Aineq @ x_interior - bineq > eps)[0]
    viol_eq = np.where(np.abs(Aeq @ x_interior - beq) > eps)[0] if eq_constraints else []

    if len(viol_ineq) > 0 or len(viol_eq) > 0:
        # treat violated constraints as temporary equalities and reproject
        Aact = []
        bact = []
        if len(viol_eq) > 0:
            Aact.append(Aeq[viol_eq])
            bact.append(beq[viol_eq])
        if len(viol_ineq) > 0:
            Aact.append(Aineq[viol_ineq])
            bact.append(bineq[viol_ineq])
        Aact = np.vstack(Aact)
        bact = np.concatenate(bact)

        # orthogonal projection onto the violated constraints
        KKT = np.block([[np.eye(n), Aact.T],
                        [Aact, np.zeros((Aact.shape[0], Aact.shape[0]))]])
        rhs = np.concatenate([x_interior, bact])
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
        x_interior = sol[:n]

    # === 5) Return new configuration ===
    return q.from_flat(x_interior)


def project_affine_cspace_explore(
    q,
    eq_constraints=None,
    ineq_constraints=None,
    eps: float = 1e-8,
    max_iters: int = 100,
    n_explore: int = 20,
    explore_sigma: float = 0.3,
):
    """
    Projection + local stochastic exploration inside the feasible region.

    This behaves like project_affine_cspace:
        - Projects q onto the feasible set defined by A_eq x = b_eq, A_in x <= b_in
    But then:
        - Samples a few small random perturbations around the projected point
        - Keeps the first perturbation that remains feasible (strictly inside)
        - Returns that interior point

    Args:
        q: configuration object with .state() and .from_flat()
        eq_constraints: list of affine equality constraints (must have .mat, .rhs)
        ineq_constraints: list of affine inequality constraints (must have .mat, .rhs)
        eps: feasibility tolerance
        max_iters: maximum active-set iterations
        n_explore: number of random interior samples to try
        explore_sigma: perturbation magnitude (relative to configuration scale)

    Returns:
        Projected configuration q_proj (possibly moved slightly inside)
        or None if projection failed.
    """

    x = q.state().astype(float).copy()
    n = len(x)

    # === 1) Collect constraints ===
    if not ineq_constraints:
        if not eq_constraints:
            return q  # no constraints
        else:
            return project_affine_only(q, eq_constraints)

    Aineq = np.vstack([c.mat for c in ineq_constraints])
    bineq = np.concatenate([c.rhs.ravel() for c in ineq_constraints])

    # === 2) Perform standard projection first (same as project_affine_cspace) ===
    if not eq_constraints:
        # only inequalities
        active = np.where(Aineq @ x - bineq > eps)[0].tolist()
        for _ in range(max_iters):
            if not active:
                x_proj = x
                break
            Aact = Aineq[active]
            bact = bineq[active]
            try:
                KKT = np.block([[np.eye(n), Aact.T],
                                [Aact, np.zeros((len(active), len(active)))]])
                rhs = np.concatenate([x.ravel(), bact.ravel()])
                sol = np.linalg.solve(KKT, rhs)
                x_proj = sol[:n]
            except LinAlgError:
                return None

            viol = np.where(Aineq @ x_proj - bineq > eps)[0].tolist()
            if not viol:
                break
            active = sorted(list(set(active + viol)))
            x = x_proj
    else:
        # equalities + inequalities
        Aeq = np.vstack([c.mat for c in eq_constraints])
        beq = np.concatenate([c.rhs.ravel() for c in eq_constraints])
        m_eq = Aeq.shape[0]
        active = np.where(Aineq @ x - bineq > eps)[0].tolist()

        for _ in range(max_iters):
            Aact = Aineq[active] if active else np.zeros((0, n))
            bact = bineq[active] if active else np.zeros(0)
            try:
                KKT = np.block([
                    [np.eye(n), Aeq.T, Aact.T],
                    [Aeq, np.zeros((m_eq, m_eq + len(active)))],
                    [Aact, np.zeros((len(active), m_eq + len(active)))]
                ])
                rhs = np.concatenate([x.ravel(), beq.ravel(), bact.ravel()])
                sol = np.linalg.solve(KKT, rhs)
                x_proj = sol[:n]
            except LinAlgError:
                return None

            resid_eq = norm(Aeq @ x_proj - beq)
            viol = np.where(Aineq @ x_proj - bineq > eps)[0].tolist()
            if resid_eq < eps and not viol:
                break
            active = sorted(list(set(active + viol)))
            x = x_proj

    # If projection failed
    if x_proj is None:
        return None

    # === 3) Local exploration around projected point ===
    # Try random perturbations and pick the first that stays feasible
    scale = norm(x_proj) if norm(x_proj) > 1e-8 else 1.0
    for _ in range(n_explore):
        delta = np.random.randn(n)
        delta /= norm(delta)
        delta *= explore_sigma * scale
        x_try = x_proj + delta

        # Check inequality feasibility (strictly inside)
        if np.all(Aineq @ x_try <= bineq - eps):
            # Optional equality re-projection (if any)
            if eq_constraints:
                try:
                    Aeq = np.vstack([c.mat for c in eq_constraints])
                    beq = np.concatenate([c.rhs.ravel() for c in eq_constraints])
                    KKT = np.block([[np.eye(n), Aeq.T],
                                    [Aeq, np.zeros((Aeq.shape[0], Aeq.shape[0]))]])
                    rhs = np.concatenate([x_try.ravel(), beq.ravel()])
                    sol = np.linalg.solve(KKT, rhs)
                    x_try = sol[:n]
                except LinAlgError:
                    pass
            return q.from_flat(x_try)

    # If no interior perturbation found, just return boundary projection
    return q.from_flat(x_proj)


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
