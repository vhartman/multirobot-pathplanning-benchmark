import numpy as np

from typing import List, Tuple, Union, Any, Optional
from numpy.typing import NDArray
import numba

from abc import ABC, abstractmethod

# TODO: make batch dist support everything, not just np config
# TODO: This is all very inconsistent atm


class Configuration(ABC):
    """
    This is the base class for representing a multirobot configuration.
    A multirobot configuration needs to provide all robots' states
    How this is done is up to the actual implementations.
    """

    @abstractmethod
    def num_agents(self) -> int:
        pass

    def __getitem__(self, ind):
        return self.robot_state(ind)

    @abstractmethod
    def __setitem__(self, ind: int, data: NDArray):
        pass

    @abstractmethod
    def robot_state(self, ind: int) -> NDArray:
        pass

    @abstractmethod
    def state(self) -> NDArray:
        pass

    @classmethod
    @abstractmethod
    def from_list(cls, q_list: List[NDArray]) -> "Configuration":
        pass

    @abstractmethod
    def from_flat(cls, q: NDArray) -> "Configuration":
        pass

    @classmethod
    def _dist(cls, pt, other, metric: str = "euclidean") -> float:
        num_agents = pt.num_agents()
        dists = np.zeros(num_agents)

        for robot_index in range(num_agents):
            diff = pt.robot_state(robot_index) - other.robot_state(robot_index)
            if metric == "euclidean":
                d = 0
                for j in range(len(diff)):
                    d += (diff[j]) ** 2
                dists[robot_index] = d**0.5
            else:
                dists[robot_index] = np.max(np.abs(diff))

        return float(np.max(dists))

    @classmethod
    def _batch_dist(cls, pt, batch_other, metric: str = "euclidean") -> NDArray:
        return np.array([cls._dist(pt, o, metric) for o in batch_other])


# Uses a list to represent all the agents states:
# q = [q1, q2, ..., qN]
class ListConfiguration(Configuration):
    def __init__(self, q_list: List[NDArray]):
        self.q = q_list

    def __getitem__(self, ind: int) -> NDArray:
        return self.robot_state(ind)

    def __setitem__(self, ind: int, data: NDArray):
        self.q[ind] = data

    @classmethod
    def from_list(cls, q_list: List[NDArray]) -> "ListConfiguration":
        return cls(q_list)

    def from_flat(self, q: NDArray) -> "ListConfiguration":
        raise NotImplementedError

    def robot_state(self, ind: int) -> NDArray:
        return self.q[ind]

    def state(self) -> NDArray:
        return np.concatenate(self.q)

    def num_agents(self) -> int:
        return len(self.q)


@numba.jit(
    (numba.float64[:, :], numba.int64[:, :]),
    nopython=True,
    fastmath=True,
    boundscheck=False,
)
def compute_sliced_euclidean_dists(diff: NDArray, slices: NDArray) -> NDArray:
    """Compute Euclidean distances for sliced configurations with optimizations."""
    num_slices = len(slices)
    num_samples = diff.shape[0]
    dists = np.empty((num_slices, num_samples), dtype=np.float64)

    # Process each slice independently
    for i in range(num_slices):
        s, e = slices[i]

        # Optimize the inner loop for better vectorization and cache usage
        for j in range(num_samples):
            sum_squared = 0.0
            # For larger slices, use a regular loop which Numba can vectorize
            for k in range(s, e):
                sum_squared += diff[j, k] * diff[j, k]

            dists[i, j] = np.sqrt(sum_squared)

    return dists


@numba.jit(
    numba.float64[:](numba.float64[:, :]),
    nopython=True,
    fastmath=True,
    boundscheck=False,
)
def compute_sum_reduction(dists: NDArray) -> NDArray:
    """Compute sum reduction across robot distances."""
    num_slices, num_samples = dists.shape
    result = np.empty(num_samples, dtype=np.float64)

    # Manually compute sum along axis 0
    for j in range(num_samples):
        sum_val = 0.0
        for i in range(num_slices):
            sum_val += dists[i, j]
        result[j] = sum_val

    return result


@numba.jit(
    numba.float64[:](numba.float64[:, :], numba.float64),
    nopython=True,
    fastmath=True,
    boundscheck=False,
)
def compute_max_sum_reduction(dists: NDArray, w: float) -> NDArray:
    """Compute max + w*sum reduction across robot distances."""
    num_slices, num_samples = dists.shape
    result = np.empty(num_samples, dtype=np.float64)

    # Manually compute max along axis 0
    for j in range(num_samples):
        max_val = dists[0, j]
        sum_val = dists[0, j]
        for i in range(1, num_slices):
            if dists[i, j] > max_val:
                max_val = dists[i, j]
            sum_val += dists[i, j]
        result[j] = max_val + w * sum_val

    return result


@numba.jit(
    numba.float64[:](numba.float64[:, :]),
    nopython=True,
    fastmath=True,
    boundscheck=False,
)
def compute_abs_max_reduction(dists: NDArray) -> NDArray:
    """Compute the maximum absolute value along axis 1 for each row."""
    num_rows, num_cols = dists.shape
    result = np.empty(num_rows, dtype=np.float64)

    for i in range(num_rows):
        # Start with first element
        max_val = abs(dists[i, 0])

        # Find maximum absolute value in the row
        for j in range(1, num_cols):
            abs_val = abs(dists[i, j])
            if abs_val > max_val:
                max_val = abs_val

        result[i] = max_val

    return result


@numba.jit(
    numba.float64[:](numba.float64[:, :]),
    nopython=True,
    fastmath=True,
    boundscheck=False,
)
def compute_max_reduction(dists: NDArray) -> NDArray:
    """Compute max + w*sum reduction across robot distances."""
    num_slices, num_samples = dists.shape
    result = np.empty(num_samples, dtype=np.float64)

    # Manually compute max along axis 0
    for j in range(num_samples):
        max_val = dists[0, j]
        for i in range(1, num_slices):
            if dists[i, j] > max_val:
                max_val = dists[i, j]
        result[j] = max_val

    return result


class NpConfiguration(Configuration):
    """
    Uses a numpy array to store all the separate agents states:
    We use one single array, and a list of indices to store the starts and ends of the separate agents
    Since different dimensions are possible for all agents, we can not easily just store the
    configuration in a single 2d array (and ragged arrays are not supported nicely)
    """

    __slots__ = (
        "_array_slice",
        "q",
        "_num_agents",
        "_robot_state_optimized",
    )  # , "robot_views"

    _array_slice: NDArray
    q: NDArray
    _num_agents: int

    def __init__(self, q: NDArray, _slice: List[Tuple[int, int]] | NDArray):
        self._array_slice = np.array(_slice)
        self.q = q.astype(np.float64)

        self._num_agents = len(self._array_slice)

        if self._num_agents == 1:
            self._robot_state_optimized = self._robot_state_single
        else:
            self._robot_state_optimized = self._robot_state_multi

    def num_agents(self) -> int:
        return self._num_agents

    def __setitem__(self, ind, data):
        s, e = self._array_slice[ind]
        self.q[s:e] = data

    @classmethod
    # @profile # run with kernprof -l examples/run_planner.py [your environment]
    def from_list(cls, q_list: List[NDArray]) -> "NpConfiguration":
        if len(q_list) == 1:
            return cls(q_list[0], [(0, len(q_list[0]))])

        slices = [(0, 0) for _ in range(len(q_list))]
        s = 0
        for i in range(len(q_list)):
            dim = len(q_list[i])
            slices[i] = (s, s + dim)
            s += dim

        return cls(np.concatenate(q_list), slices)

    def from_flat(self, q: NDArray) -> "NpConfiguration":
        assert q.shape == self.q.shape, "Shape mismatch"
        return NpConfiguration(q, self._array_slice.copy())

    @classmethod
    def from_numpy(cls, arr: NDArray) -> "NpConfiguration":
        return cls(arr, [(0, len(arr))])

    def _robot_state_single(self, ind: int) -> NDArray:
        return self.q

    def _robot_state_multi(self, ind: int) -> NDArray:
        start, end = self._array_slice[ind]
        return self.q[start:end]

    def robot_state(self, ind: int) -> NDArray:
        return self._robot_state_optimized(ind)

    def state(self) -> NDArray:
        return self.q

    @classmethod
    def _dist(cls, pt, other, metric: str = "euclidean") -> float:
        return cls._batch_dist(pt, [other], metric)[0]

    # TODO: change into specific function for one against many and many against many
    @classmethod
    # @profile # run with kernprof -l examples/run_planner.py [your environment]
    def _batch_dist(
        cls, pt: "NpConfiguration", batch_other, metric: str = "euclidean"
    ) -> NDArray:
        if isinstance(batch_other, np.ndarray):
            diff = pt.q - batch_other
            # print(pt.q.shape)
            # print(batch_other.shape)
        else:
            diff = pt.q - np.array([other.q for other in batch_other], dtype=np.float64)

        if metric == "euclidean":
            # squared_diff = diff * diff
            return compute_sliced_euclidean_dists(
                diff, np.array([[0, pt._array_slice[-1][-1]]])
            )[0]
            # return np.linalg.norm(diff, axis=1)
        elif metric == "sum_euclidean" or metric == "max_euclidean":
            dists = compute_sliced_euclidean_dists(diff, pt._array_slice)

            if metric == "sum_euclidean":
                return compute_sum_reduction(dists)
            elif metric == "max_euclidean":
                return compute_max_reduction(dists)
        else:
            return compute_abs_max_reduction(diff)


def config_dist(
    q_start: Configuration, q_end: Configuration, metric: str = "max"
) -> float:
    """
    Computes the distance between two configurations. Calls the class implementation.
    - Possible values for the metric are [euclidean, sum_euclidean, max_euclidean, max]
    """
    return type(q_start)._dist(q_start, q_end, metric)


def batch_config_dist(
    pt: Configuration, batch_pts: List[Configuration], metric: str = "max"
) -> NDArray:
    """
    Computes the distance between two lists of configurations. Calls the class implementation.
    - Possible values for the metric are [euclidean, sum_euclidean, max_euclidean, max]
    """
    return type(pt)._batch_dist(pt, batch_pts, metric)


def config_cost(
    q_start: Configuration,
    q_end: Configuration,
    metric: str = "max",
    reduction: str = "max",
) -> float:
    """
    Computes the cost between two configurations. calls the batch function.
    - Possible values for the metric are ['max', 'euclidean']
    - Possible values for the reduction are ['max', 'sum']
    """
    return batch_config_cost(q_start, q_end.state()[None, :], metric, reduction)[0]


# def one_to_many_batch_config_cost(
#     starts: Configuration,
#     batch_other: List[Configuration],
#     metric: str = "max",
#     reduction: str = "max",
#     w: float = 0.01,
# ) -> NDArray:
#     diff = starts.state() - batch_other
#     agent_slices = starts._array_slice

#     if metric == "euclidean":
#         all_robot_dists = compute_sliced_euclidean_dists(diff, agent_slices)
#     else:
#         for i, (s, e) in enumerate(agent_slices):
#             all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

#     if reduction == "max":
#         return compute_max_sum_reduction(all_robot_dists, w)
#     elif reduction == "sum":
#         return compute_sum_reduction(all_robot_dists)

#     raise ValueError


# def many_to_many_batch_config_cost(
#     starts: NDArray,
#     ends: NDArray,
#     agent_slices: NDArray,
#     metric: str = "max",
#     reduction: str = "max",
#     w: float = 0.01,
# ) -> NDArray:
#     diff = starts - ends

#     if metric == "euclidean":
#         all_robot_dists = compute_sliced_euclidean_dists(diff, agent_slices)
#     else:
#         for i, (s, e) in enumerate(agent_slices):
#             all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

#     if reduction == "max":
#         return compute_max_sum_reduction(all_robot_dists, w)
#     elif reduction == "sum":
#         return compute_sum_reduction(all_robot_dists)

#     raise ValueError


# def sequential_batch_config_cost(
#     points: NDArray,
#     agent_slices: NDArray,
#     metric: str = "max",
#     reduction: str = "max",
#     w: float = 0.01,
# ) -> NDArray:
#     diff = points[1:, :] - points[:-1, :]

#     if metric == "euclidean":
#         all_robot_dists = compute_sliced_euclidean_dists(diff, agent_slices)
#     else:
#         for i, (s, e) in enumerate(agent_slices):
#             all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

#     if reduction == "max":
#         return compute_max_sum_reduction(all_robot_dists, w)
#     elif reduction == "sum":
#         return compute_sum_reduction(all_robot_dists)

#     raise ValueError


def batch_config_cost(
    starts: Union[Configuration, List[Any]],
    batch_other: Union[NDArray, List[Any]],
    metric: str = "max",
    reduction: str = "max",
    w: float = 0.01,
    tmp_agent_slice: Optional[NDArray] = None
) -> NDArray:
    """
    Computes the cost between two lists of configurations.
    - Possible values for the metric are ['max', 'euclidean']
    - Possible values for the reduction are ['max', 'sum']
    """

    if isinstance(starts, Configuration) and isinstance(batch_other, np.ndarray):
        diff = starts.state() - batch_other
        agent_slices = starts._array_slice

    # special case for a path cost computation: We want to compute the cost between the pairs, shifted by one
    elif batch_other is None and tmp_agent_slice is not None:
        # TODO: get rid of this
        # if isinstance(starts[0], State):
        # arr = np.array([start.q.state() for start in starts])
        # diff = arr[1:, :] - arr[:-1, :]
        # agent_slices = starts[0].q._array_slice

        if isinstance(starts[0], np.ndarray):
            arr = np.array(starts)
            diff = arr[1:, :] - arr[:-1, :]
            agent_slices = tmp_agent_slice
        else:
            raise ValueError

    elif isinstance(starts, list) and isinstance(batch_other, list):
        if isinstance(starts[0], Configuration):
            diff = np.array([start.state() for start in starts]) - np.array(
                [other.state() for other in batch_other], dtype=np.float64
            )
            agent_slices = starts[0]._array_slice

        else:
            diff = np.array([start.q.state() for start in starts]) - np.array(
                [other.q.state() for other in batch_other], dtype=np.float64
            )
            agent_slices = starts[0].q._array_slice
    else:
        raise ValueError

    return _batch_config_cost_impl(diff, agent_slices, metric, reduction, w)

def _batch_config_cost_impl(
    diff: NDArray,
    agent_slices: NDArray,
    metric: str = "max",
    reduction: str = "max",
    w: float = 0.01,
):
    if metric == "euclidean":
        all_robot_dists = compute_sliced_euclidean_dists(diff, agent_slices)
    else:
        all_robot_dists = np.zeros((len(agent_slices), diff.shape[0]))
        for i, (s, e) in enumerate(agent_slices):
            all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

    if reduction == "max":
        return compute_max_sum_reduction(all_robot_dists, w)
    elif reduction == "sum":
        return compute_sum_reduction(all_robot_dists)

    raise ValueError
