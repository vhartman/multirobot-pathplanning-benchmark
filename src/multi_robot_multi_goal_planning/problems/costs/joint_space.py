from dataclasses import dataclass
import numpy as np
from multi_robot_multi_goal_planning.problems.core.configuration import (
    Configuration,
    _batch_config_cost_impl,
    batch_config_cost,
)
from multi_robot_multi_goal_planning.problems.costs.base import (
    MotionCost,
    coordinates,
    configuration,
    endpoint_pairs,
)

@dataclass(frozen=True)
class JointNormGeometry:
    """Geometry available to joint-space informed proposal generators."""

    metric: str
    reduction: str
    sum_weight: float


@dataclass(frozen=True)
class JointSpaceLength(MotionCost):
    """Sum of edge lengths, using L2/L-infinity per robot and sum or max+sum."""

    metric: str = "euclidean"
    reduction: str = "max"
    sum_weight: float = 0.01
    endpoint_exact = True

    def __post_init__(self):
        if self.metric not in ("euclidean", "max"):
            raise ValueError("Joint metric must be 'euclidean' or 'max'")
        if self.reduction not in ("sum", "max"):
            raise ValueError("Robot reduction must be 'sum' or 'max'")
        if not np.isfinite(self.sum_weight) or self.sum_weight < 0:
            raise ValueError("Sum weight must be finite and nonnegative")

    @property
    def joint_norm_geometry(self):
        return JointNormGeometry(self.metric, self.reduction, self.sum_weight)

    def edge_cost(self, start, end):
        if not isinstance(start, Configuration):
            start = start.q
        if not isinstance(end, Configuration):
            end = end.q
        diff = (start.state() - end.state())[None, :]
        return float(_batch_config_cost_impl(
            diff, start._array_slice, self.metric, self.reduction, self.sum_weight
        )[0])

    def batch_edge_cost(self, starts, ends, tmp_agent_slice=None):
        if ends is None:
            return self.segment_costs(starts, tmp_agent_slice)
        if len(ends) == 0:
            return np.empty(0)
        if isinstance(starts, Configuration):
            if not isinstance(ends, np.ndarray):
                ends = np.array([coordinates(end) for end in ends])
            return _batch_config_cost_impl(
                starts.state() - ends, starts._array_slice,
                self.metric, self.reduction, self.sum_weight
            )
        if isinstance(starts, np.ndarray):
            if len(starts) == 0:
                return np.empty(0)
            return _batch_config_cost_impl(
                starts - ends, tmp_agent_slice,
                self.metric, self.reduction, self.sum_weight
            )
        if isinstance(starts, list):
            if not starts:
                return np.empty(0)
            # Configuration/State batches already have a vectorized kernel route.
            if isinstance(starts[0], np.ndarray):
                starts, ends = np.asarray(starts), np.asarray(ends)
        else:
            return self.batch_edge_cost(starts.q, ends, tmp_agent_slice)
        return batch_config_cost(starts, ends, self.metric, self.reduction,
                                 self.sum_weight, tmp_agent_slice=tmp_agent_slice)

    def segment_costs(self, path, agent_slices=None):
        if len(path) < 2:
            return np.empty(0)
        first = path[0]
        if isinstance(first, Configuration):
            agent_slices = first._array_slice
            points = np.array([q.state() for q in path])
        elif hasattr(first, "q"):
            agent_slices = first.q._array_slice
            points = np.array([s.q.state() for s in path])
        else:
            if agent_slices is None:
                raise ValueError("Array paths require robot slices")
            points = np.asarray(path)
        return _batch_config_cost_impl(
            points[1:] - points[:-1], agent_slices,
            self.metric, self.reduction, self.sum_weight
        )

    lower_bound = edge_cost
    batch_lower_bound = batch_edge_cost
