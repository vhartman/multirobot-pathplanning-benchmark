import numpy as np
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from multi_robot_multi_goal_planning.problems.memory_util import *
from typing import List, Tuple
from numpy.typing import NDArray

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
    def _dists(cls, pt, other, metric: str = "euclidean") -> float:
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
        return dists

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


class NpConfiguration(Configuration):
    __slots__ = 'slice', 'q', '_num_agents'
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
    def _batch_dist(cls, pt, batch_other, metric: str = "euclidean") -> NDArray:
        # batch_q = np.empty((len(batch_other), pt.q.size))  # Preallocate memory
        # for i, other in enumerate(batch_other):
        #     batch_q[i, :] = other.q  # Fill in directly without overhead
        # diff = pt.q - batch_q

        # num_items = len(batch_other)
        # q_dim = pt.q.size

        # # if num_items > cls._batch_size:
        # #   cls._batch_size += 5000

        # # Ensure memory is initialized
        # cls._initialize_memory(max_size=10000, q_dim=q_dim)

        # # Populate preallocated memory (only up to num_items)
        # for i, other in enumerate(batch_other):
        #     cls._preallocated_q[i, :] = other.state.q.q
        # # cls._preallocated_q[:num_items, :] = np.array([other.state.q.q for other in batch_other])

        # # Use only the relevant part of the array
        # batch_q = cls._preallocated_q[:num_items, :]
        # diff = pt.q - batch_q

        diff = pt.q - np.array([other.q for other in batch_other])

        if metric == "euclidean":
        # if True:
            # return np.linalg.norm(diff, axis=1)
            dists = np.zeros((pt._num_agents, diff.shape[0]))
            for i, (s, e) in enumerate(pt.slice):
                dists[i, :] = np.linalg.norm(diff[:, s:e], axis=1)
            # dists = np.array([np.linalg.norm(diff[:, s:e], axis=1) for s, e in pt.slice])
            return np.max(dists, axis=0)
        else:
            return np.max(np.abs(diff), axis=1)

    @classmethod
    def _batch_dist_torch(cls, q_tensor, pt, batch_other, metric: str = "euclidean", batch_size: int = 32768) -> torch.Tensor:
        # Dynamically initialize shared tensors for the fixed batch size
        if not hasattr(cls, 'diff_batch_dist') or cls.diff_batch_dist.shape[0] < batch_size:
            cls.diff_batch_dist = torch.empty(
                (batch_size, *batch_other.shape[1:]),
                device=device,
                dtype=torch.float32,
            )

        num_slices = len(pt.slice)
        if not hasattr(cls, 'dists_batch_dist') or cls.dists_batch_dist.shape[0] < batch_size or cls.dists_batch_dist.shape[1] != num_slices:
            cls.dists_batch_dist = torch.empty(
                (batch_size, num_slices),
                device=device,
                dtype=torch.float32,
            )
        # print(batch_other.size())
        # Dynamically initialize or resize result tensor for the entire dataset
        if not hasattr(cls, 'result_dists') or batch_other.shape[0] > cls.result_dists.shape[0]:
            cls.result_dists = torch.empty(
                (int(batch_other.shape[0]),),
                device=device,
                dtype=torch.float32,
            )

        with torch.no_grad():
            num_batches = (batch_other.shape[0] + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, batch_other.shape[0])

                # Determine the actual size of the current batch
                current_batch_size = end_idx - start_idx

                # Slice preallocated tensors for current batch
                current_diff = cls.diff_batch_dist[:current_batch_size]
                current_dists = cls.dists_batch_dist[:current_batch_size]

                # Compute differences
                torch.sub(q_tensor, batch_other[start_idx:end_idx], out=current_diff)

                if metric == "euclidean":
                    # Compute Euclidean norms for each slice and find max distance for the batch
                    for i, (s, e) in enumerate(pt.slice):
                        torch.linalg.norm(current_diff[:, s:e], dim=1, out=current_dists[:, i])

                    cls.result_dists[start_idx:end_idx] = current_dists.max(dim=1).values

                elif metric == "chebyshev":
                    # Compute Chebyshev distances for the batch
                    cls.result_dists[start_idx:end_idx] = torch.max(torch.abs(current_diff), dim=1).values

        return cls.result_dists[:batch_other.shape[0]]

    @classmethod
    def _batch_torch(cls, q_tensor, pt, batch_other, metric: str = "euclidean") -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute costs and distances efficiently for _batch_torch."""
        # Dynamically check and initialize or resize shared tensors for _batch_torch
        if not hasattr(cls, 'diff_batch_torch') or batch_other.shape[0] > cls.diff_batch_torch.shape[0]:
            cls.diff_batch_torch = torch.empty(
                (batch_other.shape[0], *batch_other.shape[1:]),
                device=device,
                dtype=torch.float32,
            )

        num_slices = len(pt.slice)
        if not hasattr(cls, 'dists_batch_torch') or batch_other.shape[0] > cls.dists_batch_torch.shape[0]:
            cls.dists_batch_torch = torch.empty(
                (batch_other.shape[0], num_slices),
                device=device,
                dtype=torch.float32,
            )

        # Current slices for active computation
        current_diff = cls.diff_batch_torch[:batch_other.shape[0]]
        current_dists = cls.dists_batch_torch[:batch_other.shape[0], :]

        with torch.no_grad():
            # Compute differences
            torch.sub(q_tensor, batch_other, out=current_diff)

            # Iterate through slices and compute distances
            for i, (s, e) in enumerate(pt.slice):
                if metric == "euclidean":
                    torch.linalg.norm(current_diff[:, s:e], dim=1, out=current_dists[:, i])
                elif metric == "chebyshev":
                    current_dists[:, i] = torch.max(torch.abs(current_diff[:, s:e]), dim=1).values
                else:
                    raise ValueError(f"Unsupported metric: {metric}")

            # Compute costs
            max_dists = current_dists.max(dim=1).values  # Max distances
            sum_dists = current_dists.sum(dim=1)         # Sum of distances
            costs = max_dists + 0.01 * sum_dists         # Combine max and sum distances

            return costs, current_dists

    @classmethod
    def _batches_torch(cls, q_tensor, pt, batch_other, metric: str = "euclidean") -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure q_tensor is 2D.
        if q_tensor.dim() == 1:
            q_tensor = q_tensor.unsqueeze(0)  # Now shape becomes (1, D)

        B, D = q_tensor.shape
        N = batch_other.shape[0]
        num_slices = len(pt.slice)
        device = batch_other.device

        # Preallocate (or reuse) the differences buffer.
        if (not hasattr(cls, 'diff_batches_torch') or 
            cls.diff_batches_torch.shape[0] < B or 
            cls.diff_batches_torch.shape[1] < N or 
            cls.diff_batches_torch.shape[2] < D):
            cls.diff_batches_torch = torch.empty((B, N, D), device=device, dtype=torch.float32)
        current_diff = cls.diff_batches_torch[:B, :N, :D]

        if (not hasattr(cls, 'dists_batches_torch') or 
            cls.dists_batches_torch.shape[0] < B or 
            cls.dists_batches_torch.shape[1] < N or 
            cls.dists_batches_torch.shape[2] < num_slices):
            cls.dists_batches_torch = torch.empty((B, N, num_slices), device=device, dtype=torch.float32)
        current_dists = cls.dists_batches_torch[:B, :N, :num_slices]

        with torch.no_grad():
            torch.sub(batch_other.unsqueeze(0), q_tensor.unsqueeze(1), out=current_diff)
            for i, (s, e) in enumerate(pt.slice):
                if metric == "euclidean":
                    torch.linalg.norm(current_diff[..., s:e], dim=2, out=current_dists[..., i])
                elif metric == "chebyshev":
                    current_dists[..., i].copy_(torch.amax(torch.abs(current_diff[..., s:e]), dim=2))
                else:
                    raise ValueError(f"Unsupported metric: {metric}")

            # Compute cost as: cost = max(distance slices) + 0.01 * sum(distance slices)
            max_dists = current_dists.max(dim=2).values  # shape: (B, N)
            sum_dists = current_dists.sum(dim=2)           # shape: (B, N)
            costs = max_dists + 0.01 * sum_dists

        return costs, current_dists

def config_dist(
    q_start: Configuration, q_end: Configuration, metric: str = "."
) -> float:
    return type(q_start)._dist(q_start, q_end, metric)
def config_dists(
    q_start: Configuration, q_end: Configuration, metric: str = "."
) -> NDArray:
    return type(q_start)._dists(q_start, q_end, metric)

def batch_config_dist(
    pt: Configuration, batch_pts: List[Configuration], metric: str = "."
) -> NDArray:
    return type(pt)._batch_dist(pt, batch_pts, metric)

def batch_config_dist_torch(q_tensor:torch.tensor,
    pt: Configuration, batch_pts: torch.tensor, metric: str = "euclidean"
) -> NDArray:
    return type(pt)._batch_dist_torch(q_tensor, pt, batch_pts, metric)

def batch_config_torch(q_tensor:torch.tensor,
    pt: Configuration, batch_pts: torch.tensor,metric: str = "euclidean"
) -> torch.Tensor:
    return type(pt)._batch_torch(q_tensor, pt, batch_pts, metric)

def batches_config_torch(q_tensor:torch.tensor,
    pt: Configuration, batch_pts: torch.tensor,metric: str = "euclidean"
) -> torch.Tensor:
    return type(pt)._batches_torch(q_tensor, pt, batch_pts, metric)

def config_cost(
    q_start: Configuration, q_end: Configuration, metric: str = "."
) -> float:
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
    return max(dists) + 0.01 * sum(dists)
    # return np.sum(dists)

# TODO: this is only applicable to NpConfiguration atm.
def batch_config_cost(
    starts: List[Configuration],
    batch_other: List[Configuration],
    metric: str = ".",
) -> float:
    diff = np.array([start.q.state() for start in starts]) - np.array(
        [other.q.state() for other in batch_other]
    )
    all_robot_dists = np.zeros((starts[0].q._num_agents, diff.shape[0]))

    for i, (s, e) in enumerate(starts[0].q.slice):
        if metric == "euclidean":
            all_robot_dists[i, :] = np.linalg.norm(diff[:, s:e], axis=1)
        else:
            all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

        # print(all_robot_dists)

    return np.max(all_robot_dists, axis=0) + 0.01 * np.sum(all_robot_dists, axis=0)
