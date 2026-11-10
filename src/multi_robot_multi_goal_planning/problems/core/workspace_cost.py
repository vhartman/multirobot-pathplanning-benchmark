"""Sampled position travel of robot-fixed workspace frames."""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .cost import MotionCost, coordinates


@dataclass(frozen=True)
class WorkspaceLength(MotionCost):
    """Integrate frame travel along joint-space motion, not its workspace chord.

    positions(q) returns (robots, 3), one fixed frame position per robot.
    resolution bounds each joint-coordinate increment used for quadrature;
    it is independent of collision resolution. Results approximate arc length.
    The FK provider must depend only on q, not the planner's mutable mode/scene.
    """

    positions: Callable
    reduction: str = "max"
    sum_weight: float = 0.01
    resolution: float = 0.05
    cache_size: int = 4096
    batch_positions: Callable | None = None
    _cache: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.reduction not in ("sum", "max"):
            raise ValueError("Robot reduction must be 'sum' or 'max'")
        if not np.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError("Cost integration resolution must be finite and positive")
        if not np.isfinite(self.sum_weight) or self.sum_weight < 0:
            raise ValueError("Sum weight must be finite and nonnegative")
        if isinstance(self.cache_size, bool) or not isinstance(self.cache_size, int) or self.cache_size < 0:
            raise ValueError("Cache size must be a nonnegative integer")

    def _positions(self, rows):
        keys = [row.tobytes() for row in rows]
        found, missing = {}, {}
        for key, row in zip(keys, rows):
            if key in self._cache:
                self._cache.move_to_end(key)
                found[key] = self._cache[key]
            else:
                missing[key] = row
        if missing:
            unique = np.array(list(missing.values()))
            values = (self.batch_positions(unique) if self.batch_positions is not None
                      else [self.positions(row) for row in unique])
            values = np.asarray(values, dtype=float)
            if values.ndim != 3 or values.shape[0] != len(unique) or values.shape[1] == 0 or values.shape[2] != 3:
                raise ValueError("FK must return one 3D position per robot and configuration")
            if not np.all(np.isfinite(values)):
                raise ValueError("FK returned non-finite positions")
            for key, value in zip(missing, values):
                found[key] = value.copy()
                if self.cache_size:
                    self._cache[key] = found[key]
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
        return np.array([found[key] for key in keys])

    def _reduce(self, distances):
        total = distances.sum(axis=-1)
        return total if self.reduction == "sum" else distances.max(axis=-1) + self.sum_weight * total

    def _endpoints(self, start, end):
        a, b = coordinates(start), coordinates(end)
        if a.ndim != 1 or a.shape != b.shape or not np.all(np.isfinite([a, b])):
            raise ValueError("Edge endpoints must be matching finite configuration vectors")
        return a, b

    def edge_cost(self, start, end):
        a, b = self._endpoints(start, end)
        if np.array_equal(a, b):
            return 0.0  # includes duplicated mode-switch states
        steps = max(1, int(np.ceil(np.max(np.abs(b - a)) / self.resolution)))
        # Canonical direction gives identical quadrature for an edge and its reverse.
        if a.tobytes() > b.tobytes():
            a, b = b, a
        points = self._positions(np.linspace(a, b, steps + 1))
        increments = np.linalg.norm(np.diff(points, axis=0), axis=-1)
        return float(np.sum(self._reduce(increments)))

    def lower_bound(self, start, end):
        a, b = self._endpoints(start, end)
        points = self._positions(np.array([a, b]))
        return float(self._reduce(np.linalg.norm(points[1] - points[0], axis=-1)))

