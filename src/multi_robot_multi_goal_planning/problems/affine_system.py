"""Equality projection with rejection against inequalities and bounds."""
import numpy as np
from copy import copy


class AffineSystem:
    """A closed convex set, with a cached equality nullspace decomposition."""

    def __init__(self, dimension, equalities=(), inequalities=(), bounds=None):
        self.dimension = dimension
        self.A, self.b = self._stack(equalities)
        self.G, self.h = self._stack(inequalities)
        self._append_bounds(bounds)
        self._prepare_equalities()

    def _append_bounds(self, bounds):
        dimension = self.dimension
        if bounds is not None:
            bounds = np.asarray(bounds, dtype=float)
            if bounds.shape != (2, dimension) or np.any(np.isnan(bounds)):
                raise ValueError("Invalid configuration bounds")
            lower, upper = bounds
            # The environment represents unbounded joints by reversed limits.
            bounded = lower <= upper
            eye = np.eye(dimension)
            lo = bounded & np.isfinite(lower)
            hi = bounded & np.isfinite(upper)
            self.G = np.vstack((self.G, -eye[lo], eye[hi]))
            self.h = np.concatenate((self.h, -lower[lo], upper[hi]))

    def _prepare_equalities(self):
        dimension = self.dimension
        if len(self.A):
            u, s, vt = np.linalg.svd(self.A, full_matrices=True)
            threshold = max(self.A.shape) * np.finfo(float).eps * (s[0] if len(s) else 0)
            rank = int(np.sum(s > threshold))
            self.pinv = (vt[:rank].T / s[:rank]) @ u[:, :rank].T
            self.origin = self.pinv @ self.b
            self.Z = vt[rank:].T
        else:
            self.pinv = np.zeros((dimension, 0))
            self.origin = np.zeros(dimension)
            self.Z = np.eye(dimension)
        self.consistent = bool(np.all(np.abs(self.A @ self.origin - self.b) <= 1e-8))

    def with_bounds(self, bounds):
        """Add coordinate bounds without recomputing immutable equality geometry."""
        result = copy(self)
        result._append_bounds(bounds)
        return result

    def _stack(self, blocks):
        blocks = list(blocks)
        if not blocks:
            return np.empty((0, self.dimension)), np.empty(0)
        matrices, rhs = zip(*blocks)
        A = np.vstack(matrices).astype(float)
        b = np.concatenate([np.asarray(v, dtype=float).reshape(-1) for v in rhs])
        if A.shape != (len(b), self.dimension) or not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
            raise ValueError("Invalid affine matrix/RHS dimensions or nonfinite coefficients")
        return A, b

    def satisfied(self, x, tolerance=1e-8):
        x = np.asarray(x, dtype=float).reshape(-1)
        return bool(x.shape == (self.dimension,) and np.all(np.isfinite(x))
                    and np.all(np.abs(self.A @ x - self.b) <= tolerance)
                    and np.all(self.G @ x - self.h <= tolerance))

    def project(self, x):
        """Project onto equalities; reject if inequalities or bounds are violated."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if not self.consistent or x.shape != (self.dimension,) or not np.all(np.isfinite(x)):
            return None
        z0 = self.Z.T @ (x - self.origin)
        candidate = self.origin + self.Z @ z0
        return candidate if self.satisfied(candidate) else None

    def project_feasible(self, x):
        """Repair a seed by Euclidean projection onto equalities and inequalities.

        This is not a random sampler. Return None on infeasibility or failure.
        """
        from scipy.optimize import linprog, minimize

        x = np.asarray(x, dtype=float).reshape(-1)
        if not self.consistent or x.shape != (self.dimension,) or not np.all(np.isfinite(x)):
            return None
        z0 = self.Z.T @ (x - self.origin)
        candidate = self.origin + self.Z @ z0
        if self.satisfied(candidate):
            return candidate
        if not len(z0):
            return None
        G, h = self.G @ self.Z, self.h - self.G @ self.origin
        constraints = {"type": "ineq", "fun": lambda z: h - G @ z,
                       "jac": lambda z: -G}

        def solve(start):
            return minimize(lambda z: 0.5 * np.dot(z - z0, z - z0), start,
                            jac=lambda z: z - z0, constraints=constraints,
                            method="SLSQP", options={"maxiter": 100, "ftol": 1e-11})

        result = solve(z0)
        if not result.success:
            feasible = linprog(np.zeros(len(z0)), A_ub=G, b_ub=h,
                               bounds=[(None, None)] * len(z0), method="highs",
                               options={"time_limit": 2.0})
            if not feasible.success:
                return None
            result = solve(feasible.x)
        candidate = self.origin + self.Z @ result.x
        return candidate if result.success and self.satisfied(candidate) else None
