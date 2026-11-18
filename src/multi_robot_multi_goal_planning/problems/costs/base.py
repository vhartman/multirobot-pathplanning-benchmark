"""Motion objectives, separate from steering distances and search lower bounds."""

from abc import ABC, abstractmethod

import numpy as np

from ..core.configuration import Configuration


def configuration(value):
    """Accept either a configuration or a mode-labelled State."""
    return value.q if not isinstance(value, Configuration) and hasattr(value, "q") else value


def coordinates(value):
    value = configuration(value)
    return np.asarray(value.state() if isinstance(value, Configuration) else value, dtype=float)


def endpoint_pairs(starts, ends):
    """Pair batches, broadcasting a single configuration/state or a flat array."""
    def rows(values):
        if isinstance(values, Configuration) or hasattr(values, "mode"):
            return [values]
        if isinstance(values, np.ndarray) and values.ndim == 1:
            return [values]
        return list(values)

    left, right = rows(starts), rows(ends)
    if not left or not right:
        return []
    if len(left) == 1:
        left *= len(right)
    if len(right) == 1:
        right *= len(left)
    if len(left) != len(right):
        raise ValueError("Endpoint batches must have equal length or one singleton")
    return zip(left, right)




class MotionCost(ABC):
    """Costs of straight joint-space edges and piecewise-linear motions.

    Lower bounds refer to ANY connecting motion, not just the straight edge.
    A cost without a useful bound can safely return zero. Derived costs may
    override motion_cost when the motion representation includes dynamics.
    """

    joint_norm_geometry = None
    endpoint_exact = False

    @abstractmethod
    def edge_cost(self, start, end):
        pass

    def batch_edge_cost(self, starts, ends, tmp_agent_slice=None):
        if ends is None:
            return self.segment_costs(starts, tmp_agent_slice)
        return np.array([self.edge_cost(a, b) for a, b in endpoint_pairs(starts, ends)])

    def lower_bound(self, start, end):
        return 0.0

    def batch_lower_bound(self, starts, ends):
        return np.array([self.lower_bound(a, b) for a, b in endpoint_pairs(starts, ends)])

    def segment_costs(self, path, agent_slices=None):
        if len(path) < 2:
            return np.empty(0)
        return self.batch_edge_cost(path[:-1], path[1:], tmp_agent_slice=agent_slices)

    def motion_cost(self, points, agent_slices=None):
        """Evaluate every segment of the supplied motion, preserving its waypoints."""
        return float(np.sum(self.segment_costs(points, agent_slices)))

    def path_cost(self, path, agent_slices=None):
        return self.motion_cost(path, agent_slices)
