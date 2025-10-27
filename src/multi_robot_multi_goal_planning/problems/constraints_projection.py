from __future__ import annotations
import numpy as np
from copy import deepcopy
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

# ============================================================
# AFFINE PROJECTORS (robust, rank-agnostic versions)
# ============================================================

# ------------------------------------------------------------
# --- helpers ---
# ------------------------------------------------------------

def _orth_proj_eq_general(x, A_list, b_list):
    """
    Exact orthogonal projection onto the stacked affine equalities A x = b.
    Robust to redundancy and overdetermined stacks via pseudoinverse:
        x* = x - Aᵀ (A Aᵀ)^+ (A x - b)
    """
    if not A_list:
        return x
    A = np.vstack(A_list)
    b = np.concatenate(b_list)
    At = A.T
    M = np.linalg.pinv(A @ At)
    delta = At @ M @ (A @ x - b)
    return x - delta


def _nullspace(A, rcond=1e-12):
    """Return an orthonormal basis Z for Null(A)."""
    if A.size == 0:
        return np.eye(A.shape[1])
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    tol = rcond * (S[0] if S.size else 1.0)
    r = int((S > tol).sum())
    return Vt[r:].T  # shape (n, n-r)


def _ineq_feasible(Ain, bin, x, tol=1e-8):
    """Check if x satisfies all inequalities A_in x <= b_in (within tolerance)."""
    return (Ain.size == 0) or np.all(Ain @ x <= bin + tol)


def _nullspace_min_norm_ineq(Ain, bin, x_eq, Z, tol=1e-8, max_iters=50):
    """
    Solve in the equality nullspace:
        min 0.5||z||²  s.t. Ain(x_eq + Z z) <= bin
    Simple active-set in the reduced space. Returns feasible x or None.
    """
    if (Ain.size == 0) or (Z.size == 0):
        return x_eq if _ineq_feasible(Ain, bin, x_eq, tol) else None

    G = Ain @ Z
    h = bin - Ain @ x_eq
    z = np.zeros(Z.shape[1])
    active = []

    for _ in range(max_iters):
        # Find violated constraints
        viol = np.where(G @ z - h > tol)[0]
        for i in viol:
            if i not in active:
                active.append(i)

        # Project onto active set (least-norm correction)
        if active:
            Gact = G[active]
            hact = h[active]
            k = Z.shape[1]
            p = len(active)
            KKT = np.block([[np.eye(k), Gact.T],
                            [Gact, np.zeros((p, p))]])
            rhs = np.concatenate([np.zeros(k), hact])
            try:
                sol = np.linalg.solve(KKT, rhs)
            except LinAlgError:
                sol, *_ = np.linalg.lstsq(KKT, rhs, rcond=None)
            z = sol[:k]
        else:
            z = np.zeros_like(z)

        # Check feasibility
        if np.all(G @ z - h <= tol):
            return x_eq + Z @ z

        # Drop negative multipliers (simple active-set maintenance)
        if active:
            Gact = G[active]
            try:
                lam = np.linalg.lstsq(Gact @ Gact.T + 1e-12*np.eye(len(active)),
                                      Gact @ z - h[active], rcond=None)[0]
            except LinAlgError:
                lam = np.zeros(len(active))
            active = [idx for idx, l in zip(active, lam) if l > -1e-12]

    x_try = x_eq + Z @ z
    return x_try if _ineq_feasible(Ain, bin, x_try, tol) else None


# ============================================================
# 1. project_affine_only
# ============================================================

def project_affine_only(q: Any, constraints: List[Any]) -> Any:
    """
    Exact orthogonal projection onto affine subspace A q = b.
    Uses pseudoinverse, robust to redundant or overdetermined constraint sets.
    """
    q_vec = q.state().astype(float).copy()

    A = np.vstack([c.mat for c in constraints])
    b = np.concatenate([
        c.rhs.ravel() if hasattr(c, "rhs") else c.constraint_pose.ravel()
        for c in constraints
    ])

    q_proj = _orth_proj_eq_general(q_vec, [A], [b])
    return q.from_flat(q_proj)


# ============================================================
# 2. project_affine_cspace
# ============================================================

def project_affine_cspace(
    q: Any,
    eq_constraints: Optional[List[Any]] = None,
    ineq_constraints: Optional[List[Any]] = None,
    eps: float = 1e-8,
) -> Optional[Any]:
    """
    Euclidean projection of configuration q onto the feasible affine set:
        A_eq x = b_eq
        A_in x <= b_in

    Rank-robust formulation:
      1) Project exactly onto stacked equalities (using pseudoinverse)
      2) Compute equality nullspace Z
      3) Min-norm correction in Z to satisfy inequalities

    Returns:
        Projected configuration q_proj or None if infeasible.
    """
    x = q.state().astype(float).copy()
    n = len(x)

    # --- collect equality constraints ---
    Aeq_list, beq_list = [], []
    if eq_constraints:
        Aeq_list.append(np.vstack([c.mat for c in eq_constraints]))
        beq_list.append(np.concatenate([c.rhs.ravel() for c in eq_constraints]))
    Aeq = Aeq_list[0] if Aeq_list else np.zeros((0, n))

    # --- collect inequalities ---
    Ain = (np.vstack([c.mat for c in ineq_constraints])
           if ineq_constraints else np.zeros((0, n)))
    bin = (np.concatenate([c.rhs.ravel() for c in ineq_constraints])
           if ineq_constraints else np.zeros(0))

    # --- 1) equality projection ---
    x_eq = _orth_proj_eq_general(x, Aeq_list, beq_list)
    if Aeq.size > 0 and norm(Aeq @ x_eq - np.concatenate(beq_list)) > 1e-6:
        return None  # inconsistent equalities

    # --- 2) inequality projection in equality tangent ---
    Z = _nullspace(Aeq)
    x_proj = _nullspace_min_norm_ineq(Ain, bin, x_eq, Z, tol=eps)

    if x_proj is None:
        return None

    return q.from_flat(x_proj)


# ============================================================
# 3. project_affine_cspace_interior
# ============================================================

def project_affine_cspace_interior(
    q: Any,
    eq_constraints: Optional[List[Any]] = None,
    ineq_constraints: Optional[List[Any]] = None,
    eps: float = 1e-8,
    alpha_range=(0.05, 0.3),
) -> Optional[Any]:
    """
    Projects q onto the affine feasible set (A_eq, A_in) and then
    adds a small stochastic inward step to ensure strict inequality feasibility.
    """
    q_proj = project_affine_cspace(q, eq_constraints, ineq_constraints, eps)
    if q_proj is None:
        return None

    x_proj = q_proj.state().astype(float)
    n = len(x_proj)

    if not ineq_constraints:
        return q_proj

    # Stack inequalities
    Aineq = np.vstack([c.mat for c in ineq_constraints])
    bineq = np.concatenate([c.rhs.ravel() for c in ineq_constraints])
    slack = bineq - Aineq @ x_proj

    # If already interior → done
    if np.all(slack > eps):
        return q_proj

    # Compute random inward direction
    active = np.where(slack <= eps)[0]
    if len(active) > 0:
        n_inward = -np.sum(Aineq[active], axis=0)
    else:
        n_inward = np.random.randn(n)
    if norm(n_inward) == 0:
        n_inward = np.random.randn(n)
    n_inward /= norm(n_inward)

    min_slack = np.min(slack[slack > eps]) if np.any(slack > eps) else eps
    alpha = np.random.uniform(*alpha_range) * min_slack
    x_interior = x_proj + alpha * n_inward

    # Verify constraints after move
    if np.any(Aineq @ x_interior - bineq > eps):
        # Reproject if slightly violated
        x_interior = project_affine_cspace(q.from_flat(x_interior),
                                           eq_constraints, ineq_constraints, eps).state()

    return q.from_flat(x_interior)


# ============================================================
# 4. project_affine_cspace_explore
# ============================================================

def project_affine_cspace_explore(
    q: Any,
    eq_constraints: Optional[List[Any]] = None,
    ineq_constraints: Optional[List[Any]] = None,
    eps: float = 1e-8,
    n_explore: int = 10,
    explore_sigma: float = 0.05,
) -> Optional[Any]:
    """
    Robust projection + local stochastic exploration inside feasible region.

    Steps:
      1. Exact equality projection (rank-robust)
      2. Inequality correction in equality nullspace
      3. Random exploration in tangent nullspace (keeps feasibility)
    """
    x0 = q.state().astype(float).copy()
    n = len(x0)

    # --- stack equalities ---
    Aeq_list, beq_list = [], []
    if eq_constraints:
        Aeq_list.append(np.vstack([c.mat for c in eq_constraints]))
        beq_list.append(np.concatenate([c.rhs.ravel() for c in eq_constraints]))
    Aeq = Aeq_list[0] if Aeq_list else np.zeros((0, n))

    # --- stack inequalities ---
    Ain = (np.vstack([c.mat for c in ineq_constraints])
           if ineq_constraints else np.zeros((0, n)))
    bin = (np.concatenate([c.rhs.ravel() for c in ineq_constraints])
           if ineq_constraints else np.zeros(0))

    # --- equality projection ---
    x_eq = _orth_proj_eq_general(x0, Aeq_list, beq_list)
    if Aeq.size > 0 and norm(Aeq @ x_eq - np.concatenate(beq_list)) > 1e-6:
        return None

    # --- inequality correction in tangent ---
    Z = _nullspace(Aeq)
    x_proj = _nullspace_min_norm_ineq(Ain, bin, x_eq, Z, tol=eps)
    if x_proj is None:
        return None

    # --- local exploration (optional) ---
    if n_explore > 0 and Z.size > 0:
        k = Z.shape[1]
        scale = max(norm(x_proj), 1.0)
        for _ in range(n_explore):
            dz = np.random.randn(k)
            dz /= (norm(dz) + 1e-12)
            dz *= explore_sigma * scale
            x_try = x_proj + Z @ dz
            # snap to equalities again
            x_try = _orth_proj_eq_general(x_try, Aeq_list, beq_list)
            if _ineq_feasible(Ain, bin, x_try, tol=eps):
                return q.from_flat(x_try)

    return q.from_flat(x_proj)

# ============================================================
# NONLINEAR PROJECTORS
# ============================================================

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
    eq_constraints: List[Any] | None = None,
    ineq_constraints: List[Any] | None = None,
    mode=None,
    env=None,
    tol: float = 1e-8,
    max_iters: int = 200,
    verbose: bool = False,
) -> Optional[Any]:
    """
    Euclidean projection by solving:
        minimize 0.5 * ||x - x0||^2
        s.t. F_i(x[, mode, env]) = 0       (equalities)
             G_j(x[, mode, env]) <= 0      (inequalities)

    Uses SciPy's SLSQP solver. Supports both configuration-space and
    environment-dependent (task-space) constraints.

    Args:
        q: configuration object supporting .state() and .from_flat()
        eq_constraints: list of equality constraints
        ineq_constraints: list of inequality (region) constraints
        mode: mode to pass to env if required
        env: environment, needed for task-space constraints
        tol: convergence tolerance for SLSQP
        max_iters: maximum number of SLSQP iterations
        verbose: if True, print diagnostics

    Returns:
        Projected configuration (same type as q) or None if optimization fails.
    """

    x0 = q.state().astype(float).copy()

    def obj(x):
        d = x - x0
        return 0.5 * float(d @ d)

    def grad(x):
        return (x - x0)

    cons = []

    def _call_with_env(fun, x, c):
        """Helper: call F/J/G/dG with or without env/mode."""
        varnames = fun.__code__.co_varnames
        if "mode" in varnames or "env" in varnames:
            return fun(x, mode, env)
        else:
            return fun(x)

    # equality constraints
    if eq_constraints:
        for c in eq_constraints:
            cons.append({
                "type": "eq",
                "fun": lambda x, c=c: np.asarray(
                    _call_with_env(c.F, x, c), dtype=float
                ).reshape(-1),
                "jac": lambda x, c=c: np.asarray(
                    _call_with_env(c.J, x, c), dtype=float
                ).reshape(-1, x.size)
            })

    # inequality constraints
    if ineq_constraints:
        for c in ineq_constraints:
            cons.append({
                "type": "ineq",
                "fun": lambda x, c=c: -np.asarray(
                    _call_with_env(c.G, x, c), dtype=float
                ).reshape(-1),
                "jac": lambda x, c=c: -np.asarray(
                    _call_with_env(c.dG, x, c), dtype=float
                ).reshape(-1, x.size)
            })

    res = minimize(
        obj,
        x0,
        jac=grad,
        method="SLSQP",
        constraints=cons,
        options={"maxiter": max_iters, "ftol": tol, "disp": verbose},
    )

    if not res.success:
        if verbose:
            print(f"[project_nlp_sqp] failed: {res.message}")
        return None

    return q.from_flat(res.x)


def project_cspace_cnkz(
    q,
    constraints,
    mode=None,
    env=None,
    max_iter: int = 100,
    tol: float = 1e-8,
    step_size: float = 0.6,
    verbose: bool = False,
):
    """
    Fast Constrained Nonlinear Kaczmarz projection (cNKZ) using vectorized updates.

    Ensures all state vectors remain flat (n,).
    """

    from copy import deepcopy

    # --- ensure 1D float copy ---
    q_vec = np.asarray(q.state(), dtype=float).ravel()
    n = q_vec.size

    # --- classify constraints ---
    aff_eq = [c for c in constraints if isinstance(c, AffineConfigurationSpaceEqualityConstraint)]
    aff_ineq = [c for c in constraints if isinstance(c, AffineConfigurationSpaceInequalityConstraint)]
    nonlinear = [c for c in constraints
                 if not isinstance(c, (AffineConfigurationSpaceEqualityConstraint,
                                       AffineConfigurationSpaceInequalityConstraint))]

    # --- 1. robust affine equality projection ---
    if aff_eq:
        Aeq = np.vstack([c.mat for c in aff_eq])
        beq = np.concatenate([c.constraint_pose.ravel() for c in aff_eq])
        q_vec = np.asarray(_orth_proj_eq_general(q_vec, [Aeq], [beq])).ravel()
        Z = _nullspace(Aeq)
    else:
        Z = np.eye(n)

    # --- 2. iterative nonlinear refinement (vectorized) ---
    for it in range(max_iter):
        if not nonlinear:
            break

        # stack all residuals and Jacobians
        F_all = np.concatenate([np.asarray(c.F(q_vec, mode, env)).ravel() for c in nonlinear])
        J_all = np.vstack([np.asarray(c.J(q_vec, mode, env)) for c in nonlinear])

        # restrict to equality tangent
        Jt = J_all @ Z
        res_norm = np.linalg.norm(F_all)
        if res_norm < tol:
            break

        denom = np.sum(Jt ** 2)
        if denom < 1e-12:
            break

        dq_tangent = (Jt.T @ F_all) / denom
        dq = step_size * (Z @ dq_tangent)
        q_vec = (q_vec - dq).ravel()  # ensure stays flat

        if np.linalg.norm(dq) < 1e-8:
            break

    # --- 3. inequality feasibility correction ---
    if aff_ineq:
        Ain = np.vstack([c.mat for c in aff_ineq])
        bin = np.concatenate([c.constraint_pose.ravel() for c in aff_ineq])
        q_try = _nullspace_min_norm_ineq(Ain, bin, q_vec, Z, tol=tol)
        if q_try is not None:
            q_vec = np.asarray(q_try).ravel()

    # --- 4. return projected configuration ---
    q_new = deepcopy(q)
    if hasattr(q_new, "set_state"):
        q_new.set_state(q_vec.ravel())
    else:
        q_new = q.from_flat(np.asarray(q_vec).ravel())

    if verbose:
        print(f"[cNKZ] done after {it} iters, residual={res_norm:.2e}, step={np.linalg.norm(dq):.2e}")

    return q_new






# # FASTER VERSION:
# from __future__ import annotations
# import numpy as np
# from numpy.linalg import norm, LinAlgError
# from typing import List, Optional, Protocol, Any, Tuple, Dict
# from scipy.optimize import minimize

# from .constraints import (
#     AffineConfigurationSpaceInequalityConstraint,
#     AffineConfigurationSpaceEqualityConstraint,
# )

# # ============================================================
# # Protocols
# # ============================================================

# class EqualityConstraint(Protocol):
#     def F(self, x: np.ndarray) -> np.ndarray: ...
#     def J(self, x: np.ndarray) -> np.ndarray: ...

# class RegionConstraint(Protocol):
#     # G(x) <= 0 elementwise defines the feasible region
#     def G(self, x: np.ndarray) -> np.ndarray: ...
#     def dG(self, x: np.ndarray) -> np.ndarray: ...

# # ============================================================
# # Lightweight caches (speedups when the same A shows up repeatedly)
# # ============================================================

# # cache for nullspaces Z = Null(A)
# _Z_CACHE: Dict[int, np.ndarray] = {}
# # cache for H = A^T (A A^T)^+  (used in equality orthogonal projection)
# _H_CACHE: Dict[int, np.ndarray] = {}

# def _mat_key(A: np.ndarray) -> int:
#     """Fast, simple key for caching: hash of the array buffer + shape + dtype."""
#     # Python's hash on bytes is randomized per process; XOR with shape/dtype for stability.
#     # This is not cryptographic—just a cheap cache key.
#     return hash((A.dtype.str, A.shape, A.tobytes()))

# # ============================================================
# # Helpers (optimized)
# # ============================================================

# def _orth_proj_eq_general(x: np.ndarray, A_list: List[np.ndarray], b_list: List[np.ndarray]) -> np.ndarray:
#     """
#     Exact orthogonal projection onto stacked equalities A x = b.
#     Optimized: prefer solves/lstsq over pseudoinverse; cache H = A^T (A A^T)^+.
#     """
#     if not A_list:
#         return x

#     A = np.vstack(A_list)
#     b = np.concatenate(b_list)
#     key = _mat_key(A)

#     # Try to reuse H if available; otherwise build it fast.
#     H = _H_CACHE.get(key)
#     if H is None:
#         At = A.T
#         AAt = A @ At
#         # Prefer solve/lstsq to pinv
#         try:
#             # If AAt is (near) full rank, solve is fastest.
#             H = At @ np.linalg.solve(AAt, np.eye(AAt.shape[0]))
#         except LinAlgError:
#             # Fallback to least squares (robust to rank deficiency)
#             H = At @ np.linalg.lstsq(AAt, np.eye(AAt.shape[0]), rcond=None)[0]
#         _H_CACHE[key] = H

#     # delta = H @ (A x - b)
#     delta = H @ (A @ x - b)
#     return x - delta


# def _nullspace(A: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
#     """
#     Orthonormal basis Z for Null(A).
#     Optimized: use QR on A^T (faster than SVD) + cache.
#     """
#     if A.size == 0:
#         # no equalities: full space is the tangent
#         return np.eye(A.shape[1])

#     key = _mat_key(A)
#     Z = _Z_CACHE.get(key)
#     if Z is not None:
#         return Z

#     # QR on A^T with 'complete' gives an orthonormal basis; nullspace columns are the tail.
#     # rank from matrix_rank keeps tolerance handling.
#     Q, _ = np.linalg.qr(A.T, mode='complete')
#     rank = np.linalg.matrix_rank(A, tol=rcond)
#     Z = Q[:, rank:]  # shape (n, n-r)
#     _Z_CACHE[key] = Z
#     return Z


# def _ineq_feasible(Ain: np.ndarray, bin: np.ndarray, x: np.ndarray, tol: float = 1e-8) -> bool:
#     """Check A_in x <= b_in within tolerance."""
#     return (Ain.size == 0) or np.all(Ain @ x <= bin + tol)


# def _nullspace_min_norm_ineq(
#     Ain: np.ndarray,
#     bin: np.ndarray,
#     x_eq: np.ndarray,
#     Z: np.ndarray,
#     tol: float = 1e-8,
#     max_iters: int = 50
# ) -> Optional[np.ndarray]:
#     """
#     Solve in equality tangent: min 0.5||z||^2 s.t. Ain(x_eq + Z z) <= bin.
#     Optimized: simpler active-set using least-squares on the active block; avoids assembling KKT.
#     """
#     if (Ain.size == 0) or (Z.size == 0):
#         return x_eq if _ineq_feasible(Ain, bin, x_eq, tol) else None

#     G = Ain @ Z          # (m, k)
#     h = bin - Ain @ x_eq # (m,)
#     z = np.zeros(Z.shape[1])
#     active: List[int] = []

#     for _ in range(max_iters):
#         # add violated constraints
#         viol = np.where(G @ z - h > tol)[0]
#         for i in viol:
#             if i not in active:
#                 active.append(i)

#         if not active:
#             # feasible already in reduced space
#             return x_eq + Z @ z

#         # solve least-norm correction on active set: minimize ||z|| subj. G_act z = h_act
#         Gact = G[active]          # (p,k)
#         hact = h[active]          # (p,)
#         # use LS; more robust than forming KKT
#         try:
#             z = np.linalg.lstsq(Gact, hact, rcond=None)[0]
#         except LinAlgError:
#             return None

#         # feasibility check (all constraints)
#         if np.all(G @ z - h <= tol):
#             return x_eq + Z @ z

#         # drop constraints with negative multipliers (cheap surrogate)
#         # approximate multipliers via LS on Gact Gact^T
#         try:
#             lam = np.linalg.lstsq(Gact @ Gact.T + 1e-12 * np.eye(Gact.shape[0]),
#                                   Gact @ z - hact, rcond=None)[0]
#         except LinAlgError:
#             lam = np.zeros(len(active))
#         active = [idx for idx, l in zip(active, lam) if l > -1e-12]

#     x_try = x_eq + Z @ z
#     return x_try if _ineq_feasible(Ain, bin, x_try, tol) else None


# # ============================================================
# # 1) Affine: project_affine_only
# # ============================================================

# def project_affine_only(q: Any, constraints: List[Any]) -> Any:
#     """
#     Exact orthogonal projection onto {A q = b}.
#     Uses cached H = A^T (A A^T)^+ to avoid repeated pinv calls.
#     """
#     q_vec = q.state().astype(float, copy=True)

#     A = np.vstack([c.mat for c in constraints])
#     b = np.concatenate([
#         (c.rhs.ravel() if hasattr(c, "rhs") else c.constraint_pose.ravel())
#         for c in constraints
#     ])

#     q_proj = _orth_proj_eq_general(q_vec, [A], [b])
#     return q.from_flat(q_proj)


# # ============================================================
# # 2) Affine: project_affine_cspace
# # ============================================================

# def project_affine_cspace(
#     q: Any,
#     eq_constraints: Optional[List[Any]] = None,
#     ineq_constraints: Optional[List[Any]] = None,
#     eps: float = 1e-8,
# ) -> Optional[Any]:
#     """
#     Euclidean projection onto {A_eq x = b_eq, A_in x <= b_in}.
#     Rank-robust via exact equality projection + reduced-space inequality correction.
#     """
#     x = q.state().astype(float, copy=True)
#     n = x.size

#     # stack equalities
#     if eq_constraints:
#         Aeq = np.vstack([c.mat for c in eq_constraints])
#         beq = np.concatenate([c.rhs.ravel() for c in eq_constraints])
#         x_eq = _orth_proj_eq_general(x, [Aeq], [beq])
#         # quick consistency check (cheap)
#         if norm(Aeq @ x_eq - beq) > 1e-6:
#             return None
#         Z = _nullspace(Aeq)
#     else:
#         Aeq = np.zeros((0, n))
#         beq = np.zeros(0)
#         x_eq = x
#         Z = np.eye(n)

#     # stack inequalities
#     Ain = (np.vstack([c.mat for c in ineq_constraints]) if ineq_constraints else np.zeros((0, n)))
#     bin = (np.concatenate([c.rhs.ravel() for c in ineq_constraints]) if ineq_constraints else np.zeros(0))

#     # reduced-space inequality correction
#     x_proj = _nullspace_min_norm_ineq(Ain, bin, x_eq, Z, tol=eps)
#     if x_proj is None:
#         return None
#     return q.from_flat(x_proj)


# # ============================================================
# # 3) Affine: project_affine_cspace_interior
# # ============================================================

# def project_affine_cspace_interior(
#     q: Any,
#     eq_constraints: Optional[List[Any]] = None,
#     ineq_constraints: Optional[List[Any]] = None,
#     eps: float = 1e-8,
#     alpha_range: Tuple[float, float] = (0.05, 0.3),
# ) -> Optional[Any]:
#     """
#     Projection onto affine set, then a small inward step to ensure strict inequality feasibility.
#     Optimized: avoid unnecessary re-projection when moving strictly in nullspace.
#     """
#     q_proj = project_affine_cspace(q, eq_constraints, ineq_constraints, eps)
#     if q_proj is None:
#         return None

#     if not ineq_constraints:
#         return q_proj

#     x_proj = q_proj.state().astype(float, copy=True)
#     Aineq = np.vstack([c.mat for c in ineq_constraints])
#     bineq = np.concatenate([c.rhs.ravel() for c in ineq_constraints])
#     slack = bineq - Aineq @ x_proj

#     if np.all(slack > eps):
#         return q_proj

#     # inward direction ~ sum of active normals
#     active = np.where(slack <= eps)[0]
#     n = x_proj.size
#     inward = (-np.sum(Aineq[active], axis=0) if active.size else np.random.randn(n))
#     nrm = norm(inward)
#     inward = inward / (nrm + 1e-12)

#     # conservative step toward interior (using available positive slack)
#     pos_slack = slack[slack > eps]
#     min_slack = np.min(pos_slack) if pos_slack.size else eps
#     alpha = np.random.uniform(*alpha_range) * min_slack
#     x_interior = x_proj + alpha * inward

#     # if slight violation due to numerical noise, snap with one fast pass
#     if np.any(Aineq @ x_interior - bineq > eps):
#         q_fix = project_affine_cspace(q.from_flat(x_interior), eq_constraints, ineq_constraints, eps)
#         if q_fix is None:
#             return q_proj  # fallback to original feasible point
#         return q_fix

#     return q.from_flat(x_interior)


# # ============================================================
# # 4) Affine: project_affine_cspace_explore
# # ============================================================

# def project_affine_cspace_explore(
#     q: Any,
#     eq_constraints: Optional[List[Any]] = None,
#     ineq_constraints: Optional[List[Any]] = None,
#     eps: float = 1e-8,
#     n_explore: int = 10,
#     explore_sigma: float = 0.05,
# ) -> Optional[Any]:
#     """
#     Projection + local stochastic exploration inside feasible region.
#     Optimized: if perturbing in the equality nullspace, re-projection to equalities is optional.
#     """
#     x0 = q.state().astype(float, copy=True)
#     n = x0.size

#     # equalities
#     if eq_constraints:
#         Aeq = np.vstack([c.mat for c in eq_constraints])
#         beq = np.concatenate([c.rhs.ravel() for c in eq_constraints])
#         x_eq = _orth_proj_eq_general(x0, [Aeq], [beq])
#         if norm(Aeq @ x_eq - beq) > 1e-6:
#             return None
#         Z = _nullspace(Aeq)
#     else:
#         Aeq = np.zeros((0, n))
#         beq = np.zeros(0)
#         x_eq = x0
#         Z = np.eye(n)

#     # inequalities
#     Ain = (np.vstack([c.mat for c in ineq_constraints]) if ineq_constraints else np.zeros((0, n)))
#     bin = (np.concatenate([c.rhs.ravel() for c in ineq_constraints]) if ineq_constraints else np.zeros(0))

#     # inequality correction
#     x_proj = _nullspace_min_norm_ineq(Ain, bin, x_eq, Z, tol=eps)
#     if x_proj is None:
#         return None

#     # exploration in tangent nullspace (keeps equalities automatically)
#     if n_explore > 0 and Z.size > 0:
#         k = Z.shape[1]
#         scale = max(norm(x_proj), 1.0)
#         for _ in range(n_explore):
#             dz = np.random.randn(k)
#             dz *= (explore_sigma * scale) / (norm(dz) + 1e-12)
#             x_try = x_proj + Z @ dz
#             # optional snap to equalities in 30% of trials (cheap robustness)
#             if Aeq.size and np.random.rand() < 0.3:
#                 x_try = _orth_proj_eq_general(x_try, [Aeq], [beq])
#             if _ineq_feasible(Ain, bin, x_try, tol=eps):
#                 return q.from_flat(x_try)

#     return q.from_flat(x_proj)


# # ============================================================
# # 5) Nonlinear: project_to_manifold (damped GN with light LS)
# # ============================================================

# def project_to_manifold(
#     q: Any,
#     constraints: List[Any],
#     eps: float = 1e-6,
#     max_iters: int = 50,
#     damping: float = 1e-6,
#     verbose: bool = False
# ) -> Optional[Any]:
#     """
#     Damped Gauss–Newton with discrete backtracking.
#     Optimized: use lstsq (robust) and avoid recomputing J during line search.
#     """
#     q_vec = q.state().astype(float, copy=True)
#     n = q_vec.size

#     for it in range(max_iters):
#         Fs = [c.F(q_vec) for c in constraints]
#         Js = [c.J(q_vec) for c in constraints]

#         if not Fs:
#             return q

#         F_all = np.concatenate(Fs)
#         J_all = np.vstack(Js)
#         res_norm = norm(F_all)

#         if res_norm <= eps:
#             if verbose:
#                 print(f"[GN] converged at it={it}, ||F||={res_norm:.2e}")
#             return q.from_flat(q_vec)

#         # (JᵀJ + λI) dq = Jᵀ F  -> via least squares on augmented system
#         # Equiv to solving [J; sqrt(λ)I] dq = [F; 0] — better conditioned
#         aug_A = np.vstack([J_all, np.sqrt(max(damping, 1e-12)) * np.eye(n)])
#         aug_b = np.concatenate([F_all, np.zeros(n)])
#         try:
#             dq = np.linalg.lstsq(aug_A, aug_b, rcond=None)[0]
#         except LinAlgError:
#             return None

#         # discrete backtracking (few evaluations)
#         improved = False
#         for alpha in (1.0, 0.5, 0.25, 0.1):
#             q_trial = q_vec - alpha * dq
#             F_trial = np.concatenate([c.F(q_trial) for c in constraints])
#             if norm(F_trial) < res_norm:
#                 q_vec = q_trial
#                 improved = True
#                 break

#         if not improved:
#             if verbose:
#                 print(f"[GN] stalled at it={it}, ||F||={res_norm:.2e}")
#             return None

#     if verbose:
#         print(f"[GN] max iters reached, last ||F||={res_norm:.2e}")
#     return None


# # ============================================================
# # 6) Nonlinear: project_nlp_sqp (SciPy frontend)
# # ============================================================

# def project_nlp_sqp(
#     q: Any,
#     eq_constraints: List[EqualityConstraint] | None = None,
#     ineq_constraints: List[RegionConstraint] | None = None,
#     tol: float = 1e-8,
#     max_iters: int = 200,
#     verbose: bool = False,
# ) -> Optional[Any]:
#     """
#     Euclidean projection:
#         minimize 0.5 * ||x - x0||^2
#         s.t. F_i(x) = 0,  G_j(x) <= 0
#     Uses SLSQP. (You can switch to 'trust-constr' if it benchmarks faster for your cases.)
#     """
#     x0 = q.state().astype(float, copy=True)

#     def obj(x):
#         d = x - x0
#         return 0.5 * float(d @ d)

#     def grad(x):
#         return (x - x0)

#     cons = []
#     if eq_constraints:
#         for c in eq_constraints:
#             cons.append({
#                 "type": "eq",
#                 "fun": (lambda x, c=c: c.F(x).reshape(-1)),
#                 "jac": (lambda x, c=c: c.J(x)),
#             })
#     if ineq_constraints:
#         # SLSQP expects c(x) >= 0; provide -G(x) so G(x) <= 0 ⇒ -G >= 0
#         for c in ineq_constraints:
#             cons.append({
#                 "type": "ineq",
#                 "fun": (lambda x, c=c: -c.G(x).reshape(-1)),
#                 "jac": (lambda x, c=c: -c.dG(x)),
#             })

#     res = minimize(
#         obj, x0, jac=grad, method="SLSQP",
#         constraints=cons,
#         options={"maxiter": max_iters, "ftol": tol, "disp": verbose},
#     )

#     if not res.success:
#         if verbose:
#             print(f"[sqp] failed: {res.message}")
#         return None

#     return q.from_flat(res.x)


# # ============================================================
# # 7) Nonlinear: project_cspace_cnkz (vectorized, tangent-restricted)
# # ============================================================

# def project_cspace_cnkz(
#     q: Any,
#     constraints: List[Any],
#     mode=None,
#     env=None,
#     max_iter: int = 100,
#     tol: float = 1e-5,
#     step_size: float = 0.6,
#     verbose: bool = False,
# ) -> Optional[Any]:
#     """
#     Fast Constrained Nonlinear Kaczmarz (block / vectorized):
#       1) exact affine-equality projection
#       2) nullspace (tangent) computation
#       3) stacked update  dq = Z (JZ)^T F / ||JZ||_F^2
#       4) reduced-space inequality correction

#     Optimizations:
#       - reuse fast equality projector + QR nullspace
#       - single stacked residual/Jacobian per iteration
#       - early stop on residual and step size
#     """
#     x = q.state().astype(float, copy=True)
#     n = x.size

#     # split constraints
#     aff_eq = [c for c in constraints if isinstance(c, AffineConfigurationSpaceEqualityConstraint)]
#     aff_in = [c for c in constraints if isinstance(c, AffineConfigurationSpaceInequalityConstraint)]
#     nonlinear = [c for c in constraints if not isinstance(c, (AffineConfigurationSpaceEqualityConstraint,
#                                                               AffineConfigurationSpaceInequalityConstraint))]

#     # 1) exact equality projection + nullspace
#     if aff_eq:
#         Aeq = np.vstack([c.mat for c in aff_eq])
#         beq = np.concatenate([c.constraint_pose.ravel() for c in aff_eq])
#         x = _orth_proj_eq_general(x, [Aeq], [beq])
#         Z = _nullspace(Aeq)
#     else:
#         Z = np.eye(n)

#     # 2) nonlinear stacked iterations
#     if nonlinear:
#         for it in range(max_iter):
#             F_all = np.concatenate([c.F(x, mode, env) for c in nonlinear])
#             J_all = np.vstack([c.J(x, mode, env) for c in nonlinear])

#             Jt = J_all if Z.shape[1] == n else (J_all @ Z)
#             res_norm = norm(F_all)
#             if res_norm < tol:
#                 break

#             denom = float(np.sum(Jt * Jt))
#             if denom < 1e-12:
#                 break

#             dq_t = (Jt.T @ F_all) / denom
#             dq = step_size * (Z @ dq_t) if Z.shape[1] != n else step_size * dq_t
#             x -= dq

#             if norm(dq) < 1e-8:
#                 break

#     # 3) inequality correction in tangent
#     if aff_in:
#         Ain = np.vstack([c.mat for c in aff_in])
#         bin = np.concatenate([c.constraint_pose.ravel() for c in aff_in])
#         x_try = _nullspace_min_norm_ineq(Ain, bin, x, Z, tol=tol)
#         if x_try is not None:
#             x = x_try

#     return q.from_flat(x)
