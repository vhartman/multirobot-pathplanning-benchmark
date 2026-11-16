"""Nonlinear projectors with exact affine elimination.

This finds a locally feasible point, not necessarily the nearest projection.
Returned points preserve affine constraints, joint bounds and pinned coordinates.
"""
import numpy as np

from .affine_system import AffineSystem


def project_slsqp_bands(q, constraints, mode=None, env=None, max_iters=200,
                        *, affine_system=None):
    """Local nearest-point SQP in affine coordinates with tolerance bands.

    Nonlinear equalities already have finite feasibility tolerances. Expressing
    those bands as inequalities avoids SLSQP's singular/overdetermined equality
    subproblems (e.g. three components of a unit orientation vector). This is
    still a local solve; neither global optimality nor exact-manifold output is
    claimed. A tiny inward offset avoids strict-boundary roundoff failures.
    A supplied affine system must include this request's bounds and pins.
    """
    from scipy.optimize import minimize
    constraints = tuple({id(c): c for c in constraints}.values())
    eq = [c for c in constraints if c.type == 'affine_equality']
    ineq = [c for c in constraints if c.type == 'affine_inequality']
    nonlinear = [c for c in constraints if c.type.startswith('nonlinear')]
    n = len(q.state())
    system = affine_system
    if system is None:
        system = AffineSystem(n, [(c.mat, c.constraint_pose) for c in eq],
                              [(c.mat, c.constraint_pose) for c in ineq],
                              getattr(env, "limits", None))
    initial = system.project_feasible(q.state())
    if initial is None:
        return None

    def valid(x):
        candidate = q.from_flat(x)
        return system.satisfied(x) and all(c.is_fulfilled(candidate, mode, env) for c in constraints)

    if valid(initial):
        return q.from_flat(initial)
    if system.Z.shape[1] == 0:
        return None
    target = system.Z.T @ (q.state()-system.origin)
    z0 = system.Z.T @ (initial-system.origin)
    reduced_G = system.G @ system.Z
    reduced_h = system.h - system.G @ system.origin
    cached = [None, None, None]

    def evaluate(z):
        if cached[0] is not None and np.array_equal(z, cached[0]):
            return cached[1], cached[2]
        x = system.origin + system.Z @ z
        residuals = [reduced_h-reduced_G @ z]
        jacobians = [-reduced_G]
        for c in nonlinear:
            if c.type == 'nonlinear_equality':
                f, j = (c.FJ(x, mode, env) if hasattr(c, 'FJ')
                        else (c.F(x, mode, env), c.J(x, mode, env)))
                f = np.asarray(f).reshape(-1)
                j = np.asarray(j).reshape(-1, n) @ system.Z
                tolerance = np.asarray(getattr(c, 'eps', getattr(c, 'epsilon', 1e-8))).reshape(-1) * (1-1e-6)
                residuals.extend((tolerance-f, tolerance+f))
                jacobians.extend((-j, j))
            else:
                g, j = (c.GJ(x, mode, env) if hasattr(c, 'GJ')
                        else (c.G(x, mode, env), c.dG(x, mode, env)))
                g = np.asarray(g).reshape(-1)
                j = np.asarray(j).reshape(-1, n) @ system.Z
                tolerance = np.asarray(getattr(c, 'eps', 0.)).reshape(-1) * (1-1e-6)
                residuals.append(tolerance-g)
                jacobians.append(-j)
        values, derivatives = np.concatenate(residuals), np.vstack(jacobians)
        cached[:] = [z.copy(), values, derivatives]
        return values, derivatives

    result = minimize(lambda z: .5*np.dot(z-target, z-target), z0,
                      jac=lambda z: z-target, method='SLSQP',
                      constraints={'type':'ineq', 'fun':lambda z:evaluate(z)[0],
                                   'jac':lambda z:evaluate(z)[1]},
                      options={'maxiter':max_iters, 'ftol':1e-11})
    x = system.origin + system.Z @ result.x
    return q.from_flat(x) if result.success and valid(x) else None


def project_backtracking(q, constraints, mode=None, env=None, max_iters=80,
                         damping=1e-6, trace=None, *, affine_system=None):
    """Damped projection with backtracking and hard affine constraints.

    A supplied affine system must include this request's bounds and pins.
    """
    constraints = tuple({id(c): c for c in constraints}.values())
    eq = [c for c in constraints if c.type == 'affine_equality']
    ineq = [c for c in constraints if c.type == 'affine_inequality']
    nonlinear = [c for c in constraints if c.type.startswith('nonlinear')]
    n = len(q.state())
    system = affine_system
    if system is None:
        system = AffineSystem(n, [(c.mat, c.constraint_pose) for c in eq],
                              [(c.mat, c.constraint_pose) for c in ineq],
                              getattr(env, "limits", None))
    x = system.project_feasible(q.state())
    if x is None:
        return None

    def evaluate(x, jacobian):
        residuals, jacobians = [], []
        for c in nonlinear:
            if c.type == 'nonlinear_equality':
                if jacobian and hasattr(c, 'FJ'):
                    value, derivative = c.FJ(x, mode, env)
                else:
                    value = c.F(x, mode, env)
                    if jacobian:
                        derivative = c.J(x, mode, env)
                value = np.asarray(value, dtype=float).reshape(-1)
            else:
                if jacobian and hasattr(c, 'GJ'):
                    value, derivative = c.GJ(x, mode, env)
                else:
                    value = c.G(x, mode, env)
                    if jacobian:
                        derivative = c.dG(x, mode, env)
                value = np.asarray(value, dtype=float).reshape(-1)
                if jacobian:
                    derivative = np.asarray(derivative).reshape(-1, n).copy()
                    derivative[value <= 0] = 0
                value = np.maximum(value, 0)
            residuals.append(value)
            if jacobian:
                jacobians.append(np.asarray(derivative, dtype=float).reshape(-1, n))
        r = np.concatenate(residuals) if residuals else np.empty(0)
        J = np.vstack(jacobians) if jacobians else np.empty((0, n))
        return r, J

    for iteration in range(max_iters+1):
        candidate = q.from_flat(x)
        r, J = evaluate(x, True)
        merit = float(r @ r)
        if trace is not None:
            trace.append({'iteration': iteration, 'residual_norm': float(np.sqrt(merit))})
        if not np.isfinite(merit) or not np.all(np.isfinite(J)):
            return None
        if system.satisfied(x) and all(c.is_fulfilled(candidate, mode, env) for c in nonlinear):
            return candidate
        if iteration == max_iters or system.Z.shape[1] == 0:
            return None
        reduced = J @ system.Z
        m, k = reduced.shape
        if m < k:
            step = -reduced.T @ np.linalg.solve(reduced @ reduced.T + damping*np.eye(m), r)
        else:
            step = -np.linalg.solve(reduced.T @ reduced + damping*np.eye(k), reduced.T @ r)
        direction = system.Z @ step
        accepted = False
        for power in range(12):
            trial = system.project_feasible(x + (0.5**power)*direction)
            if trial is None:
                continue
            residual, _ = evaluate(trial, False)
            trial_merit = float(residual @ residual)
            if np.isfinite(trial_merit) and trial_merit < merit:
                x = trial
                accepted = True
                break
        if not accepted:
            return None
    return None
