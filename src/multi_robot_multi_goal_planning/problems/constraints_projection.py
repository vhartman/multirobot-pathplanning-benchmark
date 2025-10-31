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
            # (simple append; duplicates avoided)
            aset = set(active)
            for i in viol:
                if i not in aset:
                    active.append(i)

        # Project onto active set via least-norm correction
        if active:
            Gact = G[active, :]              # (p, k)
            p = Gact.shape[0]
            # Build KKT once per change; here we rebuild every iter (simple and robust)
            # [ I  G^T ] [ z ] = [ 0 ]
            # [ G   0 ] [ λ ]   [ h ]
            # Solve KKT * [z, λ] = rhs
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
            # λ ≈ (Gact Gactᵀ)^+ (Gact z - h_active)
            try:
                lam, *_ = np.linalg.lstsq(Gact @ Gact.T + 1e-12*np.eye(Gact.shape[0]), Gact @ z - h[active], rcond=None)
            except LinAlgError:
                lam = np.zeros(Gact.shape[0], dtype=float)
            # keep only non-negative multipliers (within small tolerance)
            active = [idx for idx, l in zip(active, lam) if l > -1e-12]

    x_try = x_eq + Z @ z
    return x_try if _ineq_feasible(Ain, bin, x_try, tol) else None


# ============================================================
# 1. project_affine_only
# ============================================================

def project_affine_only(q: Any, constraints: List[Any]) -> Any:
    """
    Exact orthogonal projection onto affine subspace A q = b.
    Uses robust normal-equations solve; logic identical to original.
    """
    x = q.state().astype(float, copy=True)
    # single-pass stack (avoid attribute checks in a loop afterward)
    A = np.vstack([c.mat for c in constraints])
    b = np.concatenate([(c.rhs if hasattr(c, "rhs") else c.constraint_pose).ravel() for c in constraints])
    x_proj = _orth_proj_eq_general(x, [A], [b])
    return q.from_flat(x_proj)


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

    Rank-robust:
      (1) Exact projection to equalities
      (2) Nullspace computation
      (3) Min-norm inequality correction in the nullspace
    """
    x = q.state().astype(float, copy=True)
    n = x.size

    # Stack equalities (if any)
    if eq_constraints:
        Aeq = np.vstack([c.mat for c in eq_constraints])
        beq = np.concatenate([c.rhs.ravel() for c in eq_constraints])
        Aeq_list = [Aeq]
        beq_list = [beq]
    else:
        Aeq = np.zeros((0, n), dtype=float)
        Aeq_list = []
        beq_list = []

    # Stack inequalities (if any)
    if ineq_constraints:
        Ain = np.vstack([c.mat for c in ineq_constraints])
        bin = np.concatenate([c.rhs.ravel() for c in ineq_constraints])
    else:
        Ain = np.zeros((0, n), dtype=float)
        bin = np.zeros(0, dtype=float)

    # (1) exact equality projection
    x_eq = _orth_proj_eq_general(x, Aeq_list, beq_list)
    if Aeq.size and norm(Aeq @ x_eq - beq) > 1e-6:
        return None  # inconsistent equalities

    # (2)-(3) inequality correction in equality tangent
    Z = _nullspace(Aeq)
    x_proj = _nullspace_min_norm_ineq(Ain, bin, x_eq, Z, tol=eps)
    if x_proj is None:
        return None
    return q.from_flat(x_proj)


# ============================================================
# 2. project_affine_cspace
# ============================================================

def project_affine_cspace_interior(
    q: Any,
    eq_constraints: Optional[List[Any]] = None,
    ineq_constraints: Optional[List[Any]] = None,
    eps: float = 1e-8,
    alpha_range=(0.05, 0.3),
) -> Optional[Any]:
    """
    Projects q onto the affine feasible set and then nudges strictly inside
    inequalities (stochastic inward step). Logic preserved.
    """
    q_proj = project_affine_cspace(q, eq_constraints, ineq_constraints, eps)
    if q_proj is None:
        return None

    x_proj = q_proj.state().astype(float, copy=True)
    if not ineq_constraints:
        return q_proj

    # Stack inequalities once
    Aineq = np.vstack([c.mat for c in ineq_constraints])
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
        reproj = project_affine_cspace(q.from_flat(x_interior), eq_constraints, ineq_constraints, eps)
        if reproj is None:
            return q_proj
        x_interior = reproj.state().astype(float, copy=False)

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
      1) Exact equality projection
      2) Inequality correction in equality nullspace
      3) Random exploration in tangent nullspace (keeps feasibility)
    """
    x0 = q.state().astype(float, copy=True)
    n = x0.size

    # Equalities
    if eq_constraints:
        Aeq = np.vstack([c.mat for c in eq_constraints])
        beq = np.concatenate([c.rhs.ravel() for c in eq_constraints])
        Aeq_list = [Aeq]
        beq_list = [beq]
    else:
        Aeq = np.zeros((0, n), dtype=float)
        Aeq_list = []
        beq_list = []

    # Inequalities
    if ineq_constraints:
        Ain = np.vstack([c.mat for c in ineq_constraints])
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
    Damped Gauss–Newton with backtracking. Identical logic, leaner allocations.
    """
    x = q.state().astype(float, copy=True)
    n = x.size

    for it in range(max_iters):
        # Stack residuals/Jacobians
        # (no extra ravel; assume constraints return compatible shapes)
        Fs = [c.F(x) for c in constraints]
        if not Fs:
            return q  # no constraints
        Js = [c.J(x) for c in constraints]
        F_all = np.concatenate(Fs)
        J_all = np.vstack(Js)

        res_norm = norm(F_all)
        if res_norm <= eps:
            if verbose:
                print(f"[proj] converged at iter {it}, ‖F‖={res_norm:.2e}")
            return q.from_flat(x)

        # Solve (JᵀJ + λI) dq = Jᵀ F (Levenberg)
        JT = J_all.T
        JTJ = JT @ J_all
        rhs = JT @ F_all
        try:
            dq = np.linalg.solve(JTJ + damping * np.eye(n, dtype=float), rhs)
        except LinAlgError:
            return None

        # Backtracking
        alpha = 1.0
        while alpha > 1e-6:
            x_trial = x - alpha * dq
            F_trial = np.concatenate([c.F(x_trial) for c in constraints])
            if norm(F_trial) < res_norm:
                x = x_trial
                break
            alpha *= 0.5

        if alpha <= 1e-6:
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
    Euclidean projection via SLSQP:
        minimize 0.5 * ||x - x0||^2
        s.t. F_i(x[, mode, env]) = 0
             G_j(x[, mode, env]) <= 0

    Same logic, but fewer allocations and zero per-call introspection in hot path.
    """
    x0 = q.state().astype(float, copy=True)

    # Objective and gradient are trivial; keep as simple callables
    def obj(x):
        d = x - x0
        return 0.5 * float(d @ d)

    def grad(x):
        return x - x0  # returns a view-compatible array; SLSQP copies if needed

    cons = []

    # Prepare fast wrappers that *do not* inspect signatures on every evaluation.
    # We resolve the "does it need env/mode?" once per constraint.
    def _wrap_eq(c):
        F = getattr(c, "F")
        J = getattr(c, "J")
        needs_env_F = ("mode" in F.__code__.co_varnames) or ("env" in F.__code__.co_varnames)
        needs_env_J = ("mode" in J.__code__.co_varnames) or ("env" in J.__code__.co_varnames)

        if needs_env_F:
            fun = lambda x, _c=c: np.asarray(_c.F(x, mode, env), dtype=float).reshape(-1)
        else:
            fun = lambda x, _c=c: np.asarray(_c.F(x), dtype=float).reshape(-1)

        if needs_env_J:
            jac = lambda x, _c=c: np.asarray(_c.J(x, mode, env), dtype=float).reshape(-1, x.size)
        else:
            jac = lambda x, _c=c: np.asarray(_c.J(x), dtype=float).reshape(-1, x.size)

        return {"type": "eq", "fun": fun, "jac": jac}

    def _wrap_in(c):
        G = getattr(c, "G")
        dG = getattr(c, "dG")
        needs_env_G = ("mode" in G.__code__.co_varnames) or ("env" in G.__code__.co_varnames)
        needs_env_dG = ("mode" in dG.__code__.co_varnames) or ("env" in dG.__code__.co_varnames)

        if needs_env_G:
            fun = lambda x, _c=c: -np.asarray(_c.G(x, mode, env), dtype=float).reshape(-1)
        else:
            fun = lambda x, _c=c: -np.asarray(_c.G(x), dtype=float).reshape(-1)

        if needs_env_dG:
            jac = lambda x, _c=c: -np.asarray(_c.dG(x, mode, env), dtype=float).reshape(-1, x.size)
        else:
            jac = lambda x, _c=c: -np.asarray(_c.dG(x), dtype=float).reshape(-1, x.size)

        return {"type": "ineq", "fun": fun, "jac": jac}

    if eq_constraints:
        cons.extend(_wrap_eq(c) for c in eq_constraints)
    if ineq_constraints:
        cons.extend(_wrap_in(c) for c in ineq_constraints)

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

    Optimizations:
      - reuse fast equality projector + QR nullspace
      - single stacked residual/Jacobian per iteration
      - early stop on residual and step size
    """
    x = q.state().astype(float, copy=True)
    n = x.size

    # split constraints
    aff_eq = [c for c in constraints if isinstance(c, AffineConfigurationSpaceEqualityConstraint)]
    aff_in = [c for c in constraints if isinstance(c, AffineConfigurationSpaceInequalityConstraint)]
    nonlinear = [c for c in constraints if not isinstance(c, (AffineConfigurationSpaceEqualityConstraint,
                                                              AffineConfigurationSpaceInequalityConstraint))]

    # 1) exact equality projection + nullspace
    if aff_eq:
        Aeq = np.vstack([c.mat for c in aff_eq])
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
        Ain = np.vstack([c.mat for c in aff_in])
        bin = np.concatenate([c.constraint_pose.ravel() for c in aff_in])
        x_try = _nullspace_min_norm_ineq(Ain, bin, x, Z, tol=tol)
        if x_try is not None:
            x = x_try

    return q.from_flat(x)

def project_and_visualize(
        q: Any,
        constraints: List[Any],
        mode: None,
        env: None,
        projector: str = "cnkz",
):
    if not constraints:
        print("No constraints provided.")
        return
    
    if projector == "cnkz":
        q_proj = project_cspace_cnkz(q, constraints, mode, env)
    elif projector == "sqp":
        q_proj = project_nlp_sqp(q, constraints, mode, env)
    else:
        raise ValueError(f"Unknown projector: {projector}")
    
    if q_proj is None:
        print(f"Projection failed for {projector}.")
    else:
        print(f"Original Q:")
        env.show_config(q, blocking=True)
        print(f"Projected Q ({projector}):")
        env.show_config(q_proj, blocking=True)
