"""Hit-and-run inside joint bounds and a two-endpoint cost sublevel set.

Uniform volume is stationary, but finite chains started at the midpoint are
correlated and have initialization bias. Directions use a fixed per-interval
preconditioner; collisions are rejected by the caller, not inside the chain.
"""

import numpy as np
from numba import njit

from .sampling_bounded_phs import PreparedBoundedPHS, joint_bounds


@njit(cache=True)
def joint_cost(x, a, b, slices, euclidean, maximum):
    total = 0.
    for endpoint in (a, b):
        largest = 0.
        summed = 0.
        for start, end in slices:
            value = 0.
            for k in range(start, end):
                diff = abs(x[k] - endpoint[k])
                value = value + diff * diff if euclidean else max(value, diff)
            if euclidean:
                value = np.sqrt(value)
            summed += value
            largest = max(largest, value)
        total += largest + .01 * summed if maximum else summed
    return total


@njit(cache=True)
def walk(center, a, b, lower, upper, slices, budget, directions, uniforms,
         steps, euclidean, maximum):
    x = center.copy()
    output = np.empty((len(directions) // steps, len(x)))
    for step, direction in enumerate(directions):
        lo, hi = -np.inf, np.inf
        for k in range(len(x)):
            if abs(direction[k]) > 1e-300:
                left = (lower[k] - x[k]) / direction[k]
                right = (upper[k] - x[k]) / direction[k]
                lo, hi = max(lo, min(left, right)), min(hi, max(left, right))
        if np.isfinite(lo) and np.isfinite(hi) and lo <= 0 <= hi:
            # Convex costs give one feasible chord; retain its feasible ends.
            if joint_cost(x + lo * direction, a, b, slices, euclidean, maximum) > budget:
                invalid, valid = lo, 0.
                for _ in range(30):
                    mid = (invalid + valid) * .5
                    if joint_cost(x + mid * direction, a, b, slices, euclidean, maximum) <= budget:
                        valid = mid
                    else:
                        invalid = mid
                lo = valid
            if joint_cost(x + hi * direction, a, b, slices, euclidean, maximum) > budget:
                valid, invalid = 0., hi
                for _ in range(30):
                    mid = (invalid + valid) * .5
                    if joint_cost(x + mid * direction, a, b, slices, euclidean, maximum) <= budget:
                        valid = mid
                    else:
                        invalid = mid
                hi = valid
            x = x + (lo + (hi - lo) * uniforms[step]) * direction
        if (step + 1) % steps == 0:
            output[step // steps] = x
    return output


class JointCostSampler:
    """One bounds-only informed interval, supporting L2/L-infinity and sum/max."""

    def __init__(self, env, endpoints, budget, settings, steps=1):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("Walk steps must be a positive integer")
        if not np.all(np.isfinite(endpoints)) or not np.isfinite(budget) or budget <= 0:
            raise ValueError("Endpoints and a positive budget must be finite")
        self.a, self.b = np.array(endpoints, dtype=float, copy=True)
        self.slices = np.array(env.start_pos._array_slice, copy=True)
        self.settings = settings
        self.budget, self.steps = budget, steps
        self.center = self.a / 2 + self.b / 2
        lower, upper = joint_bounds(env)
        self.lower = np.maximum(lower, np.maximum(self.a - budget, self.b - budget))
        self.upper = np.minimum(upper, np.minimum(self.a + budget, self.b + budget))
        if np.any(self.center < self.lower) or np.any(self.center > self.upper):
            raise ValueError("Midpoint outside bounds")
        if joint_cost(self.center, self.a, self.b, self.slices, *settings) >= budget:
            raise ValueError("Empty or degenerate informed set")
        self.blocks = []
        for start, end in self.slices:
            if settings[0]:
                phs = PreparedBoundedPHS.from_endpoints(
                    self.a[start:end], self.b[start:end], budget,
                    self.lower[start:end], self.upper[start:end])
                self.blocks.append(phs.u * phs.singular)
            else:
                self.blocks.append(np.diag(self.upper[start:end] - self.lower[start:end]))

    def sample(self, n):
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("Sample count must be a nonnegative integer")
        if n == 0:
            return np.empty((0, len(self.a)))
        directions = np.random.normal(size=(n * self.steps, len(self.a)))
        for (start, end), basis in zip(self.slices, self.blocks):
            directions[:, start:end] = directions[:, start:end] @ basis.T
        directions[:, self.lower == self.upper] = 0.
        rows = walk(self.center, self.a, self.b, self.lower, self.upper, self.slices,
                    self.budget, directions, np.random.random(len(directions)),
                    self.steps, *self.settings)
        self.center = rows[-1].copy()
        return rows


def warm_joint_cost_kernel():
    """Compile without consuming randomness, before planner timing."""
    a, b = np.zeros(2), np.ones(2)
    walk((a + b) * .5, a, b, a - 2, b + 2, np.array([[0, 1], [1, 2]]),
         3., np.ones((2, 2)), np.ones(2) * .5, 1, True, True)
