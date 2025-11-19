from __future__ import annotations
import numpy as np
from copy import deepcopy
from numpy.linalg import norm, LinAlgError
from typing import List, Optional, Protocol, Any
from scipy.optimize import minimize

class EqualityConstraint(Protocol):
    def F(self, x: np.ndarray) -> np.ndarray: ...
    def J(self, x: np.ndarray) -> np.ndarray: ...

class RegionConstraint(Protocol):
    # g(x) <= 0 elementwise defines the feasible region
    def G(self, x: np.ndarray) -> np.ndarray: ...
    def dG(self, x: np.ndarray) -> np.ndarray: ...

# ============================================================
# AFFINE PROJECTORS (robust, rank-agnostic versions)
# ============================================================

# ------------------------------------------------------------
# --- helpers ---
# ------------------------------------------------------------

def _orth_proj_eq_general(x: np.ndarray, A_list: List[np.ndarray], b_list: List[np.ndarray]) -> np.ndarray:
    """
    Exact orthogonal projection onto the stacked affine equalities A x = b.
    Robust to redundancy and overdetermined stacks via normal equations solve:
        x* = x - Aᵀ y,   with   (A Aᵀ) y = A x - b
    Using lstsq on (A Aᵀ) avoids explicit pseudoinverse construction.
    """
    if not A_list:
        return x
    A = np.vstack(A_list)
    b = np.concatenate(b_list)
    # compute y from (A Aᵀ) y = A x - b
    Ax_minus_b = A @ x - b
    # For rank-deficient, lstsq gives the minimum-norm y (same effect as pinv on (A Aᵀ))
    y, *_ = np.linalg.lstsq(A @ A.T, Ax_minus_b, rcond=None)
    return x - A.T @ y


def _nullspace(A: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Return an orthonormal basis Z for Null(A)."""
    if A.size == 0:
        # No equalities → nullspace is full space
        return np.eye(A.shape[1], dtype=float)
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    if S.size == 0:
        return np.eye(A.shape[1], dtype=float)
    tol = rcond * S[0]
    r = int((S > tol).sum())
    return Vt[r:].T  # shape (n, n-r)


def _ineq_feasible(Ain: np.ndarray, bin: np.ndarray, x: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if x satisfies all inequalities A_in x <= b_in (within tolerance)."""
    return (Ain.size == 0) or np.all(Ain @ x <= bin + tol)


def _nullspace_min_norm_ineq(
    Ain: np.ndarray,
    bin: np.ndarray,
    x_eq: np.ndarray,
    Z: np.ndarray,
    tol: float = 1e-8,
    max_iters: int = 50,
) -> Optional[np.ndarray]:
    """
    Solve in the equality nullspace:
        min 0.5||z||²  s.t. Ain(x_eq + Z z) <= bin
    Simple active-set in the reduced space. Returns feasible x or None.
    """
    if (Ain.size == 0) or (Z.size == 0):
        return x_eq if _ineq_feasible(Ain, bin, x_eq, tol) else None

    # Precompute reduced data once
    G = Ain @ Z                    # (m, k)
    h = bin - Ain @ x_eq           # (m,)
    k = Z.shape[1]
    z = np.zeros(k, dtype=float)
    active: List[int] = []

    # Prebuild identity used in KKT
    Ik = np.eye(k, dtype=float)

    for _ in range(max_iters):
        # Identify violated constraints
        viol = np.nonzero(G @ z - h > tol)[0]
        if viol.size:
            # Add newly violated to active set
            aset = set(active)
            for i in viol:
                if i not in aset:
                    active.append(i)

        # Project onto active set via least-norm correction
        if active:
            Gact = G[active, :]              # (p, k)
            p = Gact.shape[0]
            KKT = np.block([[Ik, Gact.T],
                            [Gact, np.zeros((p, p), dtype=float)]])
            rhs = np.concatenate([np.zeros(k, dtype=float), h[active]])
            try:
                sol = np.linalg.solve(KKT, rhs)
            except LinAlgError:
                sol, *_ = np.linalg.lstsq(KKT, rhs, rcond=None)
            z = sol[:k]
        else:
            # No active constraints; z stays at 0 (least-norm)
            pass

        # Check feasibility of reduced inequalities
        if np.all(G @ z - h <= tol):
            return x_eq + Z @ z

        # Active-set cleanup: drop constraints with negative multipliers
        if active:
            Gact = G[active, :]
            try:
                lam, *_ = np.linalg.lstsq(Gact @ Gact.T + 1e-12*np.eye(Gact.shape[0]), Gact @ z - h[active], rcond=None)
            except LinAlgError:
                lam = np.zeros(Gact.shape[0], dtype=float)
            # keep only non-negative multipliers (within small tolerance)
            active = [idx for idx, l in zip(active, lam) if l > -1e-12]

    x_try = x_eq + Z @ z
    return x_try if _ineq_feasible(Ain, bin, x_try, tol) else None

# ------------------------------------------------------------
# --- projection ---
# ------------------------------------------------------------

def project_affine_only(q: Any, mode, env, constraints: List[Any]) -> Any:
    """
    Exact orthogonal projection onto affine subspace A q = b.
    Uses robust normal-equations solve; logic identical to original.
    """
    x = q.state().astype(float, copy=True)
    # single-pass stack (avoid attribute checks in a loop afterward)
    A = np.vstack([c.J(x, mode, env) for c in constraints])
    b = np.concatenate([(c.rhs if hasattr(c, "rhs") else c.constraint_pose).ravel() for c in constraints])
    x_proj = _orth_proj_eq_general(x, [A], [b])
    return q.from_flat(x_proj)


def project_affine_cspace(
    q: Any,
    constraints: List[Any],
    mode=None,
    env=None,
    max_iter: int = 100,
    tol: float = 1e-5,
    step_size: float = 0.6,
    verbose: bool = False,
) -> Optional[Any]:

    x = q.state().astype(float, copy=True)
    n = x.size

    # split constraints
    aff_eq = [c for c in constraints if c.type == "affine_equality"]
    aff_in = [c for c in constraints if c.type == "affine_inequality"]

    # 0) no constraints
    if not (aff_eq or aff_in):
        return q

    # 1) exact equality projection + nullspace
    if aff_eq:
        Aeq = np.vstack([c.J(x, mode, env) for c in aff_eq])
        beq = np.concatenate([c.constraint_pose.ravel() for c in aff_eq])
        x = _orth_proj_eq_general(x, [Aeq], [beq])
        Z = _nullspace(Aeq)
    else:
        Z = np.eye(n)

    # 2) inequality correction in tangent
    if aff_in:
        Ain = np.vstack([c.dG(x, mode, env) for c in aff_in])
        bin = np.concatenate([c.constraint_pose.ravel() for c in aff_in])
        x_try = _nullspace_min_norm_ineq(Ain, bin, x, Z, tol=tol)
        if x_try is not None:
            x = x_try

    return q.from_flat(x)


def project_affine_cspace_interior(
    q: Any,
    eq_constraints: Optional[List[Any]] = None,
    ineq_constraints: Optional[List[Any]] = None,
    mode=None,
    env=None,
    eps: float = 1e-8,
    alpha_range=(0.05, 0.3),
) -> Optional[Any]:
    """
    Projects q onto the affine feasible set and then nudges strictly inside
    inequalities (stochastic inward step). Logic preserved.
    """
    q_proj = project_affine_cspace(q, eq_constraints + ineq_constraints, eps)
    if q_proj is None:
        return None

    x_proj = q_proj.state().astype(float, copy=True)
    if not ineq_constraints:
        return q_proj

    # Stack inequalities once
    Aineq = np.vstack([c.dG(x_proj, mode, env) for c in ineq_constraints])
    bineq = np.concatenate([c.rhs.ravel() for c in ineq_constraints])
    slack = bineq - Aineq @ x_proj

    # Already strictly inside
    if np.all(slack > eps):
        return q_proj

    n = x_proj.size
    # Inward direction: sum of outward normals of (nearly) active constraints
    active = np.nonzero(slack <= eps)[0]
    if active.size > 0:
        n_inward = -np.sum(Aineq[active, :], axis=0)
        if not np.any(n_inward):
            n_inward = np.random.randn(n)
    else:
        n_inward = np.random.randn(n)
    nrm = norm(n_inward)
    if nrm == 0.0:
        n_inward = np.random.randn(n)
        nrm = norm(n_inward)
    n_inward /= nrm

    # Step size proportional to smallest positive slack
    pos = slack[slack > eps]
    min_slack = float(np.min(pos)) if pos.size else eps
    alpha = np.random.uniform(*alpha_range) * min_slack
    x_interior = x_proj + alpha * n_inward

    # Verify and (if needed) reproject
    if np.any(Aineq @ x_interior - bineq > eps):
        reproj = project_affine_cspace(q.from_flat(x_interior), eq_constraints + ineq_constraints, eps)
        if reproj is None:
            return q_proj
        x_interior = reproj.state().astype(float, copy=False)

    return q.from_flat(x_interior)


def project_affine_cspace_explore(
    q: Any,
    eq_constraints: Optional[List[Any]] = None,
    ineq_constraints: Optional[List[Any]] = None,
    mode=None,
    env=None,
    eps: float = 1e-8,
    n_explore: int = 10,
    explore_sigma: float = 0.05,
) -> Optional[Any]:
    """
    Robust projection + local stochastic exploration inside feasible region.

    Steps:
      1) Exact equality projection
      2) Inequality correction in equality nullspace
      3) Random exploration in tangent nullspace (keeps feasibility)
    """
    x0 = q.state().astype(float, copy=True)
    n = x0.size

    # Equalities
    if eq_constraints:
        Aeq = np.vstack([c.J(x0, mode, env) for c in eq_constraints])
        beq = np.concatenate([c.rhs.ravel() for c in eq_constraints])
        Aeq_list = [Aeq]
        beq_list = [beq]
    else:
        Aeq = np.zeros((0, n), dtype=float)
        Aeq_list = []
        beq_list = []

    # Inequalities
    if ineq_constraints:
        Ain = np.vstack([c.dG(x0, mode, env) for c in ineq_constraints])
        bin = np.concatenate([c.rhs.ravel() for c in ineq_constraints])
    else:
        Ain = np.zeros((0, n), dtype=float)
        bin = np.zeros(0, dtype=float)

    # Equality projection
    x_eq = _orth_proj_eq_general(x0, Aeq_list, beq_list)
    if Aeq.size and norm(Aeq @ x_eq - beq) > 1e-6:
        return None

    # Inequality correction
    Z = _nullspace(Aeq)
    x_proj = _nullspace_min_norm_ineq(Ain, bin, x_eq, Z, tol=eps)
    if x_proj is None:
        return None

    # Exploration in nullspace
    if n_explore > 0 and Z.size > 0:
        k = Z.shape[1]
        scale = max(norm(x_proj), 1.0)
        for _ in range(n_explore):
            dz = np.random.randn(k)
            dz /= (norm(dz) + 1e-12)
            dz *= explore_sigma * scale
            x_try = x_proj + Z @ dz
            # Snap to equalities again (cheap)
            x_try = _orth_proj_eq_general(x_try, Aeq_list, beq_list)
            if _ineq_feasible(Ain, bin, x_try, tol=eps):
                return q.from_flat(x_try)

    return q.from_flat(x_proj)


# ============================================================
# NONLINEAR PROJECTORS
# ============================================================

# def project_gauss_newton(
#     q: Any,
#     constraints: List[Any],
#     mode=None,
#     env=None,
#     tol: float = 1e-8,
#     max_iters: int = 100,
#     step_size: float = 1.0,
#     damping: float = 1e-6,
#     verbose: bool = False,
# ) -> Optional[Any]:
#     """
#     Trivial Gauss-Newton projection.
#     No exact projection, no nullspace handling.
#     Just pure least-squares correction of current violations.
#     """

#     x = q.state().astype(float, copy=True)
#     n = x.size

#     for it in range(max_iters):
#         residuals = []
#         jacobians = []

#         for c in constraints:
#             # --- equality-type constraint ---
#             if hasattr(c, "F") and hasattr(c, "J"):
#                 F = np.asarray(c.F(x), dtype=float).reshape(-1)
#                 if F.size:
#                     J = np.asarray(c.J(x), dtype=float).reshape(-1, n)
#                     residuals.append(F)
#                     jacobians.append(J)

#             # --- inequality-type constraint ---
#             if hasattr(c, "G") and hasattr(c, "dG"):
#                 G = np.asarray(c.G(x), dtype=float).reshape(-1)
#                 if G.size:
#                     dG = np.asarray(c.dG(x), dtype=float).reshape(-1, n)
#                     active = G > 0.0  # only violated constraints
#                     if np.any(active):
#                         residuals.append(G[active])
#                         jacobians.append(dG[active])

#         # nothing to fix
#         if not residuals:
#             if verbose:
#                 print(f"[gn_trivial] converged: no residuals at iter {it}")
#             break

#         r = np.concatenate(residuals)
#         J = np.vstack(jacobians)

#         # damped normal equations (Gauss–Newton step)
#         H = J.T @ J + damping * np.eye(n)
#         g = J.T @ r

#         try:
#             dq = np.linalg.solve(H, g)
#         except np.linalg.LinAlgError:
#             dq, *_ = np.linalg.lstsq(H, g, rcond=None)

#         x -= step_size * dq

#         if norm(r) < tol or norm(dq) < 1e-10:
#             if verbose:
#                 print(f"[gn_trivial] converged at iter {it}")
#             break

#     return q.from_flat(x)


def project_gauss_newton(
    q: Any,
    constraints: List[Any],
    mode=None,
    env=None,
    tol: float = 1e-2,
    max_iters: int = 500,
    step_size: float = 0.5,
    damping: float = 1e-6,
    verbose: bool = False,
) -> Optional[Any]:
    """
    Local Gauss-Newton projection, strict about affine equalities:
        1) exact affine projection (Aeq x = beq)
        2) iterate in nullspace for nonlinear equalities and active inequalities
        3) affine inequality correction
    """

    x = q.state().astype(float, copy=True)
    n = x.size

    # remove constraints that are here twice
    constraints = list(set(constraints))

    # split constraints
    aff_eq = [c for c in constraints if c.type == "affine_equality"]
    aff_in = [c for c in constraints if c.type == "affine_inequality"]
    nl_eq = [c for c in constraints if c.type == "nonlinear_equality"]
    nl_in = [c for c in constraints if c.type == "nonlinear_inequality"]
    

    if aff_eq:
        Aeq = np.vstack([c.J(x, mode, env) for c in aff_eq])
        beq = np.concatenate([c.constraint_pose.ravel() for c in aff_eq])
        x = _orth_proj_eq_general(x, [Aeq], [beq])
        Z = _nullspace(Aeq)
    else:
        Z = np.eye(n)

    if Z.shape[1] == 0:
        if verbose:
            print(f"[project_gauss_newton] no nullspace to explore")
        return q.from_flat(x)

    for it in range(max_iters):
        residuals = []
        jacobians = []

        # nonlinear equalities
        for c in nl_eq:
            F = np.asarray(c.F(x, mode, env), dtype=float).reshape(-1)
            J = np.asarray(c.J(x, mode, env), dtype=float).reshape(-1, n)
            residuals.append(F)
            jacobians.append(J)

        # nonlinear inequalities (active only)
        for c in nl_in:
            g = np.asarray(c.G(x, mode, env), dtype=float).reshape(-1)
            dG = np.asarray(c.dG(x, mode, env), dtype=float).reshape(-1, n)
            active = g > 0.0
            if np.any(active):
                residuals.append(g[active])
                jacobians.append(dG[active])

        # affine inequalities (active only)
        for c in aff_in:
            g = c.G(x, mode, env).ravel()
            active = g > 0.0
            if np.any(active):
                residuals.append(g[active])
                jacobians.append(c.dG(x, mode, env).reshape(-1, n)[active])
        # stop if nothing to fix
        if not residuals:
            if verbose:
                print(f"no constraints")
            break

        r = np.concatenate(residuals)
        J = np.vstack(jacobians)
        Jt = J if Z.shape[1] == n else J @ Z # reduced Jacobian in nullspace

        # damped normal equations
        H = Jt.T @ Jt + damping * np.eye(Jt.shape[1])
        g = Jt.T @ r

        try:
            dq_t = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            dq_t, *_ = np.linalg.lstsq(H, g, rcond=None)

        dq = Z @ dq_t if Z.shape[1] != n else dq_t
        x -= step_size * dq

        # # reproject to affine equalities
        # if aff_eq:
        #     x = _orth_proj_eq_general(x, [Aeq], [beq])
        #     Z = _nullspace(Aeq)

        if norm(r) < tol: # or norm(dq) < 1e-10
            if verbose:
                print(f"[project_gauss_newton] converged at iter {it}")
                satisfied = all(c.is_fulfilled(q.from_flat(x), mode, env) for c in nl_eq)
                print(f"all nonlinear constraints satisfied = {satisfied}")
                print(f"residual norm = {norm(r)}")

                for c in nl_eq:
                    res_c = c.F(q.from_flat(x).state(), mode, env)
                    print(f"  CONSTR {c}: RES = {res_c}")
                
            break

    if aff_in:
        Ain = np.vstack([c.dG(x, mode, env) for c in aff_in])
        bin_ = np.concatenate([c.constraint_pose.ravel() for c in aff_in])
        x_try = _nullspace_min_norm_ineq(Ain, bin_, x, Z, tol=tol)
        if x_try is not None:
            x = x_try
    
    return q.from_flat(x)


def project_nlp_sqp(
    q: Any,
    constraints: List[Any],
    mode=None,
    env=None,
    tol: float = 1e-8,
    max_iters: int = 200,
    verbose: bool = False,
) -> Optional[Any]:
    """
    Euclidean projection via SLSQP, strict about affine equalities:
        1) exact affine projection (Aeq x = beq)
        2) nonlinear equality and inequality constraints via SLSQP
        3) affine inequality correction
    """

    x = q.state().astype(float, copy=True)
    n = x.size

    # split constraints
    aff_eq = [c for c in constraints if c.type == "affine_equality"]
    aff_in = [c for c in constraints if c.type == "affine_inequality"]
    nl_eq = [c for c in constraints if c.type == "nonlinear_equality"]
    nl_in = [c for c in constraints if c.type == "nonlinear_inequality"]

    if aff_eq:
        Aeq = np.vstack([c.J(x, mode, env) for c in aff_eq])
        beq = np.concatenate([c.constraint_pose.ravel() for c in aff_eq])
        x = _orth_proj_eq_general(x, [Aeq], [beq])

    def obj(xv):
        d = xv - x
        return 0.5 * float(d @ d)

    def grad(xv):
        return xv - x

    cons = []

    # affine inequalities -> simple numeric
    for c in aff_in:
        A, b = c.dG(x, mode, env), c.constraint_pose
        cons.append({
            "type": "ineq",
            "fun": lambda xv, A=A, b=b: (b - A @ xv).ravel(),
            "jac": lambda xv, A=A: -A,
        })

    # nonlinear equalities
    for c in nl_eq:
        cons.append({
            "type": "eq",
            "fun": lambda xv, c=c: np.asarray(c.F(xv, mode, env), dtype=float).ravel(),
            "jac": lambda xv, c=c: np.asarray(c.J(xv, mode, env), dtype=float),
        })

    # nonlinear inequalities
    for c in nl_in:
        cons.append({
            "type": "ineq",
            "fun": lambda xv, c=c: -np.asarray(c.G(xv, mode, env), dtype=float).ravel(),
            "jac": lambda xv, c=c: -np.asarray(c.dG(xv, mode, env), dtype=float),
        })

    res = minimize(
        obj,
        x,
        jac=grad,
        method="SLSQP",
        constraints=cons,
        options={"maxiter": max_iters, "ftol": tol, "disp": verbose},
    )

    if not res.success:
        if verbose:
            print(f"[project_nlp_sqp] failed: {res.message}")
        x_proj = x
    else:
        x_proj = res.x

    if aff_in:
        Ain = np.vstack([c.dG(x_proj, mode, env) for c in aff_in])
        bin_ = np.concatenate([c.constraint_pose.ravel() for c in aff_in])
        x_try = _nullspace_min_norm_ineq(Ain, bin_, x_proj, np.eye(n), tol=tol)
        if x_try is not None:
            x_proj = x_try

    return q.from_flat(x_proj)


def project_cspace_cnkz(
    q: Any,
    constraints: List[Any],
    mode=None,
    env=None,
    max_iter: int = 100,
    tol: float = 1e-5,
    step_size: float = 0.6,
    verbose: bool = False,
) -> Optional[Any]:
    """
    Fast Constrained Nonlinear Kaczmarz (block / vectorized):
      1) exact affine-equality projection
      2) nullspace (tangent) computation
      3) stacked update  dq = Z (JZ)^T F / ||JZ||_F^2
      4) reduced-space inequality correction
    """
    x = q.state().astype(float, copy=True)
    n = x.size

    # split constraints
    aff_eq = [c for c in constraints if c.type == "affine_equality"]
    aff_in = [c for c in constraints if c.type == "affine_inequality"]
    nl_eq = [c for c in constraints if c.type == "nonlinear_equality"]
    nl_in = [c for c in constraints if c.type == "nonlinear_inequality"]
    nonlinear = nl_eq + nl_in

    # 0) no constraints
    if not (aff_eq or aff_in or nonlinear):
        return q

    # 1) exact equality projection + nullspace
    if aff_eq:
        Aeq = np.vstack([c.J(x, mode, env) for c in aff_eq])
        beq = np.concatenate([c.constraint_pose.ravel() for c in aff_eq])
        x = _orth_proj_eq_general(x, [Aeq], [beq])
        Z = _nullspace(Aeq)
    else:
        Z = np.eye(n)

    # 2) nonlinear stacked iterations
    if nonlinear:
        for it in range(max_iter):
            F_all = np.concatenate([c.F(x, mode, env) for c in nonlinear])
            J_all = np.vstack([c.J(x, mode, env) for c in nonlinear])

            Jt = J_all if Z.shape[1] == n else (J_all @ Z)
            res_norm = norm(F_all)
            if res_norm < tol:
                break

            denom = float(np.sum(Jt * Jt))
            if denom < 1e-12:
                break

            dq_t = (Jt.T @ F_all) / denom
            dq = step_size * (Z @ dq_t) if Z.shape[1] != n else step_size * dq_t
            x -= dq

            if norm(dq) < 1e-8:
                break

    # 3) inequality correction in tangent
    if aff_in:
        Ain = np.vstack([c.dG(x, mode, env) for c in aff_in])
        bin = np.concatenate([c.constraint_pose.ravel() for c in aff_in])
        x_try = _nullspace_min_norm_ineq(Ain, bin, x, Z, tol=tol)
        if x_try is not None:
            x = x_try

    return q.from_flat(x)
