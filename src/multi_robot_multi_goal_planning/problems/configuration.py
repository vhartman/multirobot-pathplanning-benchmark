import numpy as np

from typing import List, Tuple
from numpy.typing import NDArray
import numba

from abc import ABC, abstractmethod


class Configuration(ABC):
    @abstractmethod
    def num_agents(self) -> int:
        pass

    def __getitem__(self, ind):
        return self.robot_state(ind)

    @abstractmethod
    def robot_state(self, ind: int) -> NDArray:
        pass

    @abstractmethod
    def state(self) -> NDArray:
        pass

    @abstractmethod
    def from_list(cls, q_list: List[NDArray]):
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


class ListConfiguration(Configuration):
    def __init__(self, q_list):
        self.q = q_list

    def __getitem__(self, ind):
        return self.robot_state(ind)

    def __setitem__(self, ind, data):
        self.q[ind] = data

    @classmethod
    def from_list(cls, q_list: List[NDArray]) -> "ListConfiguration":
        return cls(q_list)

    def robot_state(self, ind: int) -> NDArray:
        return self.q[ind]

    def state(self) -> NDArray:
        return np.concatenate(self.q)

    def num_agents(self):
        return len(self.q)


@numba.jit(nopython=True, parallel=True)
def numba_parallelized_sum(squared_diff, slices, num_agents):
    dists = np.zeros((num_agents, squared_diff.shape[0]))

    for i in numba.prange(num_agents):
        s, e = slices[i]
        # slice_data = diff[:, s:e]
        # Manual norm computation
        squared = squared_diff[:, s:e]
        dists[i, :] = np.sqrt(np.sum(squared, axis=1))

    return dists


@numba.jit((numba.float64[:, :], numba.int64[:, :]), nopython=True)
def compute_sliced_dists(squared_diff, slices):
    num_slices = len(slices)
    num_samples = squared_diff.shape[0]
    dists = np.empty((num_slices, num_samples), dtype=np.float32)

    for i in range(num_slices):
        s, e = slices[i]
        dists[i, :] = np.sqrt(np.sum(squared_diff[:, s:e], axis=1))

    return dists


@numba.jit((numba.float64[:, :], numba.int64[:, :]), nopython=True)
def compute_sliced_dists_transpose(squared_diff, slices):
    num_slices = len(slices)
    num_samples = squared_diff.shape[1]
    dists = np.empty((num_slices, num_samples), dtype=np.float32)

    for i in range(num_slices):
        s, e = slices[i]
        dists[i, :] = np.sqrt(np.sum(squared_diff[s:e, :], axis=0)).T

    return dists


class NpConfiguration(Configuration):
    __slots__ = "slice", "q", "_num_agents"

    def __init__(self, q: NDArray, slice: List[Tuple[int, int]]):
        self.slice = slice
        self.q = q

        self._num_agents = len(slice)

    def num_agents(self):
        return self._num_agents

    def __setitem__(self, ind, data):
        self.q[self.slice[ind][0] : self.slice[ind][1]] = data

    @classmethod
    # @profile # run with kernprof -l examples/run_planner.py [your environment]
    def from_list(cls, q_list: List[NDArray]) -> "NpConfiguration":
        if len(q_list) == 1:
            return cls(q_list[0], [(0, len(q_list[0]))])

        slices = [None for _ in range(len(q_list))]
        s = 0
        for i in range(len(q_list)):
            dim = len(q_list[i])
            slices[i] = (s, s + dim)
            s += dim

        return cls(np.concatenate(q_list), slices)

    @classmethod
    def from_numpy(cls, arr: NDArray):
        return cls(arr, [(0, len(arr))])

    def robot_state(self, ind: int) -> NDArray:
        if self._num_agents == 1:
            return self.q

        start, end = self.slice[ind]
        return self.q[start:end]

    def state(self) -> NDArray:
        return self.q

    @classmethod
    def _dist(cls, pt, other, metric: str = "euclidean") -> float:
        num_agents = pt._num_agents
        dists = np.zeros(num_agents)

        diff = pt.q - other.q

        if metric == "euclidean":
            for i, (s, e) in enumerate(pt.slice):
                d = 0
                for j in range(s, e):
                    d += (diff[j]) ** 2
                dists[i] = d**0.5
            return float(np.max(dists))
        else:
            return float(np.max(np.abs(diff)))

    # _preallocated_q = None
    # @classmethod
    # def _initialize_memory(cls, max_size, q_dim):
    #     if cls._preallocated_q is None or cls._preallocated_q.shape != (max_size, q_dim):
    #         cls._preallocated_q = np.empty((max_size, q_dim))  # Preallocate

    @classmethod
    # @profile # run with kernprof -l examples/run_planner.py [your environment]
    def _batch_dist(cls, pt, batch_other, metric: str = "max") -> NDArray:
        if isinstance(batch_other, np.ndarray):
            diff = pt.q - batch_other
            # print(pt.q.shape)
            # print(batch_other.shape)
        else:
            diff = pt.q - np.array([other.q for other in batch_other])

        if metric == "euclidean":
            return np.linalg.norm(diff, axis=1)
        elif metric == "sum_euclidean" or metric == "max_euclidean":
            squared_diff = diff * diff
            # dists = np.zeros((pt._num_agents, diff.shape[0]))

            # dists = numba_parallelized_sum(squared_diff, pt.slice, pt._num_agents)

            # for i, (s, e) in enumerate(pt.slice):
            #     # Use sqrt(sum(x^2)) instead of np.linalg.norm
            #     # and pre-computed squared differences
            #     dists[i, :] = np.sqrt(np.sum(squared_diff[:, s:e], axis=1))

            # print([e for _, e in pt.slice[:-1]])
            # dists = np.sqrt(
            #     np.add.reduceat(
            #         squared_diff.T, [0] + [e for _, e in pt.slice[:-1]], axis=0
            #     )
            # )

            dists = compute_sliced_dists(squared_diff, np.array(pt.slice))
            # dists = compute_sliced_dists_transpose(squared_diff.T, np.array(pt.slice))

            # print(tmp - dists)

            # for i, (s, e) in enumerate(pt.slice):
            #     dists[i, :] = np.linalg.norm(diff[:, s:e], axis=1)
            # dists = np.array([np.linalg.norm(diff[:, s:e], axis=1) for s, e in pt.slice])
            if metric == "sum_euclidean":
                return np.sum(dists, axis=0)
            elif metric == "max_euclidean":
                return np.max(dists, axis=0) + 0.01 * np.sum(dists, axis=0)
        else:
            return np.max(np.abs(diff), axis=1)


def config_dist(
    q_start: Configuration, q_end: Configuration, metric: str = "max"
) -> float:
    return type(q_start)._dist(q_start, q_end, metric)


def batch_config_dist(
    pt: Configuration, batch_pts: List[Configuration], metric: str = "max"
) -> NDArray:
    return type(pt)._batch_dist(pt, batch_pts, metric)


def config_cost(
    q_start: Configuration,
    q_end: Configuration,
    metric: str = "max",
    reduction: str = "max",
) -> float:
    # return batch_config_cost([q_start], [q_end], metric, reduction)
    num_agents = q_start.num_agents()
    dists = np.zeros(num_agents)

    for robot_index in range(num_agents):
        # print(robot_index)
        # print(q_start)
        # print(q_end)
        # d = np.linalg.norm(q_start[robot_index] - q_end[robot_index])
        diff = q_start.robot_state(robot_index) - q_end.robot_state(robot_index)
        if metric == "euclidean":
            d = 0
            for j in range(len(diff)):
                d += (diff[j]) ** 2
            dists[robot_index] = d**0.5
        else:
            dists[robot_index] = np.max(np.abs(diff))

    # dists = np.linalg.norm(np.array(q_start) - np.array(q_end), axis=1)
    if reduction == "max":
        return max(dists) + 0.01 * sum(dists)
    elif reduction == "sum":
        return np.sum(dists)


def batch_config_cost(
    starts: List[Configuration],
    batch_other: List[Configuration],
    metric: str = "max",
    reduction: str = "max",
) -> float:
    if isinstance(starts, Configuration) and isinstance(batch_other, np.ndarray):
        diff = starts.state() - batch_other
        all_robot_dists = np.zeros((starts._num_agents, diff.shape[0]))
        agent_slices = starts.slice
    else:
        diff = np.array([start.q.state() for start in starts]) - np.array(
            [other.q.state() for other in batch_other]
        )
        all_robot_dists = np.zeros((starts[0].q._num_agents, diff.shape[0]))
        agent_slices = starts[0].q.slice

    # return np.linalg.norm(diff, axis=1)

    if metric == "euclidean":
        squared_diff = diff * diff
        all_robot_dists = compute_sliced_dists(squared_diff, np.array(agent_slices))
    else:
        for i, (s, e) in enumerate(agent_slices):
            all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

    # for i, (s, e) in enumerate(agent_slices):
    #     if metric == "euclidean":
    #         all_robot_dists[i, :] = np.linalg.norm(diff[:, s:e], axis=1)
    #     else:
    #         all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

    # print(tmp - all_robot_dists)

        # print(all_robot_dists)

    if reduction == "max":
        return np.max(all_robot_dists, axis=0) + 0.01 * np.sum(all_robot_dists, axis=0)
    elif reduction == "sum":
        return np.sum(all_robot_dists, axis=0)
