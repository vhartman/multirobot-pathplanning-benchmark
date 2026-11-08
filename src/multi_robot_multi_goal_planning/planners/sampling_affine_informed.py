"""Affine-feasible informed proposals with exact cost rejection or hit-and-run.

Geometry and walk state belong to one endpoint pair, cost bound and set of pins.
Finite walks are correlated and initialization-biased; collisions are checked
by the caller and never roll back a walk.
"""

import numpy as np
from numba import njit

from multi_robot_multi_goal_planning.problems.affine_system import AffineSystem
from .sampling_affine import AffinePolytopeSampler
from .sampling_joint_cost import joint_cost


def informed_coordinate_bounds(endpoints, budget, slices, euclidean, maximum):
    """Necessary bounds for L2/L-infinity, with sum or max+0.01*sum reduction."""
    a, b = np.asarray(endpoints, dtype=float)
    if not np.all(np.isfinite(endpoints)) or not np.isfinite(budget) or budget <= 0:
        raise ValueError("Endpoints and a positive cost bound must be finite")
    delta = b - a
    distances = np.array([
        np.linalg.norm(delta[s:e]) if euclidean else np.max(np.abs(delta[s:e]))
        for s, e in slices
    ])
    bounds = np.empty((2, len(a)))
    for i, (start, end) in enumerate(slices):
        # Other robots must at least cover their endpoint distance. For max,
        # dropping the nonnegative sum penalty only enlarges the proposal box.
        allowed = budget if maximum else budget - (distances.sum() - distances[i])
        if allowed < distances[i]:
            raise ValueError("Cost bound below endpoint distance")
        radius = (np.sqrt(np.maximum(0., (allowed - distances[i]) * (allowed + distances[i])
                                    + delta[start:end] ** 2)) / 2
                  if euclidean else np.full(end - start, allowed / 2))
        center = a[start:end] / 2 + b[start:end] / 2
        margin = 1e-10 * (1 + np.abs(center) + radius)
        bounds[:, start:end] = [center - radius - margin, center + radius + margin]
    return bounds


@njit(cache=True)
def affine_cost_walk(z, origin, basis, G, h, a, b, slices, budget,
                     directions, uniforms, steps, euclidean, maximum):
    z = z.copy()
    output = np.empty((len(directions) // steps, len(origin)))
    for step, direction in enumerate(directions):
        x = origin + basis @ z
        dx = basis @ direction
        slopes, slack = G @ direction, h - G @ z
        lo, hi = -np.inf, np.inf
        for row in range(len(h)):
            if slopes[row] > 0:
                hi = min(hi, slack[row] / slopes[row])
            elif slopes[row] < 0:
                lo = max(lo, slack[row] / slopes[row])
        if np.isfinite(lo) and np.isfinite(hi) and lo <= 0 <= hi:
            # Convex costs intersect each line in a single interval. Keep the
            # feasible side of each bisection, as in the bounds-only walker.
            if joint_cost(x + lo * dx, a, b, slices, euclidean, maximum) > budget:
                invalid, valid = lo, 0.
                for _ in range(30):
                    mid = (invalid + valid) * .5
                    if joint_cost(x + mid * dx, a, b, slices, euclidean, maximum) <= budget:
                        valid = mid
                    else:
                        invalid = mid
                lo = valid
            if joint_cost(x + hi * dx, a, b, slices, euclidean, maximum) > budget:
                valid, invalid = 0., hi
                for _ in range(30):
                    mid = (valid + invalid) * .5
                    if joint_cost(x + mid * dx, a, b, slices, euclidean, maximum) <= budget:
                        valid = mid
                    else:
                        invalid = mid
                hi = valid
            z += (lo + (hi - lo) * uniforms[step]) * direction
        if (step + 1) % steps == 0:
            output[step // steps] = origin + basis @ z
    return output, z


class AffineInformedSampler:
    """Sample a bounded affine proposal, optionally walking in its cost subset.

    Direct polytope components are IID; large components use stationary-uniform
    walks. If no strict cost-feasible seed is found, keep proposing from the
    polytope instead of declaring the informed region empty.
    """

    warmup_sweeps = 10

    def __init__(self, env, active, fixed, endpoints, budget, settings, steps=1, walk=False):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("Walk steps must be a positive integer")
        self.a, self.b = np.array(endpoints, dtype=float, copy=True)
        self.slices = np.array(env.start_pos._array_slice, dtype=np.int64, order="C", copy=True)
        self.budget, self.settings, self.steps = budget, settings, steps
        bounds = informed_coordinate_bounds(endpoints, budget, self.slices, *settings)
        affine = active._affine_system(fixed) if fixed else active.affine
        self.system = AffineSystem(
            affine.dimension, [(affine.A, affine.b)], [(affine.G, affine.h)], bounds)
        self.polytope = AffinePolytopeSampler(self.system)
        self.use_walk = walk and self.system.Z.shape[1] > 0
        self.z = None
        if not self.use_walk:
            return
        self.origin = np.ascontiguousarray(self.system.origin)
        self.basis = np.ascontiguousarray(self.system.Z)
        G = self.system.G @ self.basis
        h = self.system.h - self.system.G @ self.origin
        norms = np.linalg.norm(G, axis=1)
        tolerance = (32 * np.finfo(float).eps * affine.dimension
                     * np.linalg.norm(self.system.G, axis=1))
        nonconstant = norms > tolerance
        self.G = np.ascontiguousarray(G[nonconstant] / norms[nonconstant, None])
        self.h = np.ascontiguousarray(h[nonconstant] / norms[nonconstant])
        # A fixed metric in the equality tangent space; never adapt it to the
        # current state, which would change the stationary distribution.
        weighted = self.basis / (bounds[1] - bounds[0])[:, None]
        metric = weighted.T @ weighted
        try:
            self.transform = np.linalg.solve(np.linalg.cholesky(metric).T,
                                             np.eye(self.basis.shape[1]))
        except np.linalg.LinAlgError:
            self.transform = np.eye(self.basis.shape[1])

    def sample(self, n):
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("Sample count must be a nonnegative integer")
        if n == 0:
            return np.empty((0, len(self.a)))
        warmup = 0
        if self.z is None:
            rows = [self.polytope.draw() for _ in range(n)]
            rows = np.asarray([row for row in rows if row is not None]).reshape((-1, len(self.a)))
            if not self.use_walk:
                return rows
            for row in rows:
                z = self.basis.T @ (row - self.origin)
                if (np.all(self.G @ z < self.h)
                        and joint_cost(row, self.a, self.b, self.slices, *self.settings) < self.budget):
                    self.z = z
                    break
            if self.z is None and len(rows):
                # A thin cost region may reject every polytope draw. A feasible
                # projected midpoint can seed the walk; move off any faces
                # toward a polytope draw without losing strict cost feasibility.
                midpoint = self.system.project(self.a / 2 + self.b / 2)
                if (midpoint is not None and joint_cost(
                        midpoint, self.a, self.b, self.slices, *self.settings) < self.budget):
                    fraction = .5
                    for _ in range(40):
                        candidate = (1 - fraction) * midpoint + fraction * rows[0]
                        z = self.basis.T @ (candidate - self.origin)
                        if (np.all(self.G @ z < self.h) and joint_cost(
                                candidate, self.a, self.b, self.slices, *self.settings) < self.budget):
                            self.z = z
                            break
                        fraction *= .5
            if self.z is None:
                return rows
            warmup = int(np.ceil(self.warmup_sweeps * len(self.z) / self.steps))
        count = (n + warmup) * self.steps
        directions = np.ascontiguousarray(
            np.random.normal(size=(count, len(self.z))) @ self.transform.T)
        rows, self.z = affine_cost_walk(
            self.z, self.origin, self.basis, self.G, self.h, self.a, self.b,
            self.slices, self.budget, directions, np.random.random(count), self.steps, *self.settings)
        return rows[warmup:]


def warm_affine_cost_kernel():
    """Compile without consuming randomness, before planner timing."""
    affine_cost_walk(np.zeros(2), np.zeros(2), np.eye(2),
                     np.r_[np.eye(2), -np.eye(2)], np.ones(4),
                     np.zeros(2), np.zeros(2), np.array([[0, 2]], dtype=np.int64),
                     3., np.ones((2, 2)), np.full(2, .5), 1, True, True)
