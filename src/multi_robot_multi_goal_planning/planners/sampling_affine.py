"""Sample bounded affine polytopes, factoring independent coordinates.

Small components use direct volume-uniform sampling. Larger components use
hit-and-run: uniform is its stationary distribution, not a finite-chain guarantee.
Collision rejection must not roll back the walk.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

from multi_robot_multi_goal_planning.problems.affine_system import AffineSystem


class _PolytopeComponent:
    # Bound triangulation work; larger components use a walk instead.
    direct_dimension_limit = 5
    direct_row_limit = 32
    warmup_sweeps = 10

    def __init__(self, system):
        self.system = system
        self.dimension = system.Z.shape[1]
        self.simplices = None
        if not system.consistent:
            raise ValueError("Inconsistent equalities")

        G = system.G @ system.Z
        h = system.h - system.G @ system.origin
        norms = np.linalg.norm(G, axis=1)
        # Nullspace cancellation can leave tiny slopes on fixed coordinates.
        tolerance = (32 * np.finfo(float).eps * system.dimension
                     * np.linalg.norm(system.G, axis=1))
        active = norms > tolerance
        if np.any(h[~active] < -1e-8):
            raise ValueError("Infeasible constant inequality")
        self.G = G[active] / norms[active, None]
        self.h = h[active] / norms[active]
        if self.dimension == 0:
            return
        if self.dimension == 1:
            slopes = self.G[:, 0]
            self.lower = np.max(self.h[slopes < 0] / slopes[slopes < 0], initial=-np.inf)
            self.upper = np.min(self.h[slopes > 0] / slopes[slopes > 0], initial=np.inf)
            if (not np.isfinite(self.lower) or not np.isfinite(self.upper)
                    or self.lower > self.upper):
                raise ValueError("Empty or unbounded interval")
            return

        self._check_bounded()
        # Maximize the inscribed ball radius to initialize strictly inside.
        result = linprog(
            np.r_[np.zeros(self.dimension), -1.0],
            A_ub=np.c_[self.G, np.ones(len(self.G))], b_ub=self.h,
            bounds=[(None, None)] * self.dimension + [(0, None)],
            method="highs", options={"time_limit": 2.0},
        )
        if not result.success or result.x[-1] <= 1e-10:
            raise ValueError("No numerically usable interior")
        self.z = result.x[:-1].copy()
        self.warmup_remaining = self.warmup_sweeps * self.dimension
        if (self.dimension <= self.direct_dimension_limit
                and len(self.G) <= self.direct_row_limit):
            self._triangulate()

    def _check_bounded(self):
        # Finite coordinate bounds (including pins) are a cheap certificate.
        system = self.system
        lower = np.zeros(system.dimension, dtype=bool)
        upper = lower.copy()
        for matrix, equality in ((system.A, True), (system.G, False)):
            for row in matrix:
                indices = np.flatnonzero(row)
                if len(indices) == 1:
                    i = indices[0]
                    lower[i] |= equality or row[i] < 0
                    upper[i] |= equality or row[i] > 0
        if np.all(lower & upper):
            return
        # Coupled inequalities can bound a region without coordinate bounds.
        for direction in np.eye(self.dimension):
            for sign in (-1, 1):
                result = linprog(
                    sign * direction, A_ub=self.G, b_ub=self.h,
                    bounds=[(None, None)] * self.dimension,
                    method="highs", options={"time_limit": 2.0},
                )
                if not result.success:
                    raise ValueError("Could not certify a bounded polytope")

    def _triangulate(self):
        try:
            vertices = HalfspaceIntersection(np.c_[self.G, -self.h], self.z).intersections
            if (not np.all(np.isfinite(vertices))
                    or np.any(vertices @ self.G.T - self.h > 1e-8)):
                return
            if len(vertices) == self.dimension + 1:
                simplices = vertices[None, :, :]
            else:
                facets = vertices[ConvexHull(vertices).simplices]
                centers = np.broadcast_to(self.z, (len(facets), 1, self.dimension))
                simplices = np.concatenate((centers, facets), axis=1)
            signs, logs = np.linalg.slogdet(simplices[:, 1:] - simplices[:, :1])
            keep = (signs != 0) & np.isfinite(logs)
            if not np.any(keep):
                return
            weights = np.exp(logs[keep] - np.max(logs[keep]))
            self.cdf = np.cumsum(weights / weights.sum())
            self.cdf[-1] = 1.0
            self.simplices = simplices[keep]
        except (QhullError, ValueError, np.linalg.LinAlgError):
            # No geometry perturbation: retain the walk on the original region.
            return

    def draw(self):
        if self.dimension == 0:
            return self.system.origin.copy()
        if self.dimension == 1:
            z = np.array([np.random.uniform(self.lower, self.upper)])
        elif self.simplices is not None:
            index = np.searchsorted(self.cdf, np.random.random(), side="right")
            weights = np.random.dirichlet(np.ones(self.dimension + 1))
            z = weights @ self.simplices[index]
        else:
            # One dimension-sized sweep per draw, plus initial warmup.
            for _ in range(self.dimension + self.warmup_remaining):
                direction = np.random.normal(size=self.dimension)
                direction /= np.linalg.norm(direction)
                slopes = self.G @ direction
                slack = self.h - self.G @ self.z
                upper = np.min(slack[slopes > 0] / slopes[slopes > 0], initial=np.inf)
                lower = np.max(slack[slopes < 0] / slopes[slopes < 0], initial=-np.inf)
                if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
                    return None
                candidate = self.z + np.random.uniform(lower, upper) * direction
                if np.any(self.G @ candidate - self.h > 1e-8):
                    return None
                self.z = candidate
                if self.warmup_remaining:
                    self.warmup_remaining -= 1
            z = self.z
        return self.system.origin + self.system.Z @ z


class AffinePolytopeSampler:
    """Independent constraint components, each sampled in its equality nullspace.

    Construction raises ValueError when preparation is unavailable, including
    implicit lower-dimensional faces; this is not an infeasibility certificate.
    """

    def __init__(self, system):
        self.dimension = system.dimension
        self.parts = []
        parent = list(range(system.dimension))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        # Factor before SVD, which can rotate otherwise independent coordinates.
        for matrix, rhs, equality in ((system.A, system.b, True),
                                      (system.G, system.h, False)):
            for row, value in zip(matrix, rhs):
                indices = np.flatnonzero(row)
                if not len(indices):
                    if (abs(value) > 1e-8 if equality else value < -1e-8):
                        raise ValueError("Infeasible constant constraint")
                    continue
                for index in indices[1:]:
                    parent[root(int(index))] = root(int(indices[0]))
        groups = {}
        for i in range(system.dimension):
            groups.setdefault(root(i), []).append(i)
        for indices in groups.values():
            indices = np.asarray(indices, dtype=int)
            eq_rows = np.any(system.A[:, indices] != 0, axis=1)
            ineq_rows = np.any(system.G[:, indices] != 0, axis=1)
            part = AffineSystem(
                len(indices),
                [(system.A[eq_rows][:, indices], system.b[eq_rows])],
                [(system.G[ineq_rows][:, indices], system.h[ineq_rows])],
            )
            self.parts.append((indices, _PolytopeComponent(part)))

        self._scalar_product = all(component.dimension <= 1 for _, component in self.parts)
        if self._scalar_product:
            self._origin = np.empty(self.dimension)
            indices, factors, directions, lower, upper = [], [], [], [], []
            for coordinates, component in self.parts:
                self._origin[coordinates] = component.system.origin
                if component.dimension == 1:
                    indices.extend(coordinates)
                    factors.extend([len(lower)] * len(coordinates))
                    directions.extend(component.system.Z[:, 0])
                    lower.append(component.lower)
                    upper.append(component.upper)
            self._interval_indices = np.asarray(indices, dtype=int)
            self._interval_factors = np.asarray(factors, dtype=int)
            self._interval_directions = np.asarray(directions)
            self._interval_lower = np.asarray(lower)
            self._interval_upper = np.asarray(upper)

    def _draw_scalar_product(self, count):
        result = np.broadcast_to(self._origin, (count, self.dimension)).copy()
        if len(self._interval_lower):
            # Sample-major order matches successive scalar draws.
            values = np.random.uniform(self._interval_lower, self._interval_upper,
                                       size=(count, len(self._interval_lower)))
            indices = self._interval_indices
            result[:, indices] = (self._origin[indices]
                + self._interval_directions * values[:, self._interval_factors])
        return result

    def draw_many(self, count):
        """Batch scalar factors, retaining draw order for coupled components."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Sample count must be a nonnegative integer")
        if self._scalar_product:
            return self._draw_scalar_product(count)
        rows = [self.draw() for _ in range(count)]
        return np.asarray([row for row in rows if row is not None]).reshape((-1, self.dimension))

    def draw(self):
        if self._scalar_product:
            return self._draw_scalar_product(1)[0]
        result = np.empty(self.dimension)
        for indices, component in self.parts:
            x = component.draw()
            if x is None:
                return None
            result[indices] = x
        return result
