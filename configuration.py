import numpy as np

from typing import List
from numpy.typing import NDArray

from abc import ABC, abstractmethod
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch


class Configuration(ABC):
    @abstractmethod
    def num_agents(self) -> int:
        pass

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

        return np.max(dists)
    
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
    def _batch_dist(cls, pt, batch_other, metric: str = "euclidean") -> float:
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
    # __slots__ = 'slice', 'q', '_num_agents'
    def __init__(self, q: NDArray, slice: List[int]):
        self.slice = slice
        self.q = q

        self._num_agents = len(slice)

    def num_agents(self):
        return self._num_agents

    def __getitem__(self, ind):
        return self.robot_state(ind)

    def __setitem__(self, ind, data):
        self.q[self.slice[ind][0] : self.slice[ind][1]] = data

    @classmethod
    def from_list(cls, q_list: List[NDArray]) -> "NpConfiguration":
        slices = []
        s = 0
        for i in range(len(q_list)):
            slices.append((s, s + len(q_list[i])))
            s += len(q_list[i])

        return cls(np.concatenate(q_list), slices)

    def robot_state(self, ind: int) -> NDArray:
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
            return np.max(dists)
        else:
            return np.max(np.abs(diff))

    # _preallocated_q = None
    # @classmethod
    # def _initialize_memory(cls, max_size, q_dim):
    #     if cls._preallocated_q is None or cls._preallocated_q.shape != (max_size, q_dim):
    #         cls._preallocated_q = np.empty((max_size, q_dim))  # Preallocate

    @classmethod
    def _batch_dist(cls, pt, batch_other, metric: str = "euclidean") -> float:
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
            dists = np.zeros((pt._num_agents, diff.shape[0]))
            for i, (s, e) in enumerate(pt.slice):
                dists[i, :] = np.linalg.norm(diff[:, s:e], axis=1)
            # dists = np.array([np.linalg.norm(diff[:, s:e], axis=1) for s, e in pt.slice])
            return np.max(dists, axis=0)
        else:
            return np.max(np.abs(diff), axis=1)
    
    @classmethod
    def _batch_dist_torch(cls, pt, batch_other, metric: str = "euclidean") -> torch.Tensor:
        torch.cuda.empty_cache()
        q_tensor = torch.as_tensor(pt.q, device='cuda').unsqueeze(0)
        # batch_other = batch_other.to(dtype=torch.float16)
    
        q_tensor = torch.as_tensor(pt.q, device='cuda')
        with torch.no_grad():
            diff = q_tensor - batch_other

        if metric == "euclidean":
            dists = [
            torch.linalg.norm(diff[:, s:e], dim=1, keepdim=True) for s, e in pt.slice
            ] 
            dists = torch.cat(dists, dim=1)
            return dists.max(dim=1).values 
        # elif metric == "chebyshev":
        else:
            return torch.max(torch.abs(diff), dim=1).values  # Shape: (batch_size,)
        
   

    # @classmethod
    # def _batch_dist_torch(cls, pt, batch_other, metric: str = "euclidean") -> torch.Tensor:
    #     torch.cuda.empty_cache()
    #     q_tensor = torch.as_tensor(pt.q, device='cuda', dtype=torch.float16)
    #     batch_other = batch_other.to(dtype=torch.float16)

    #     with torch.no_grad():
    #         diff = q_tensor - batch_other

    #     if metric == "euclidean":
    #         dists = [
    #             torch.linalg.norm(diff[:, s:e], dim=1, keepdim=True) for s, e in pt.slice
    #         ]
    #         dists = torch.cat(dists, dim=1)
    #         del diff
    #         torch.cuda.empty_cache()
    #         return dists.max(dim=1).values
    #     else:
    #         return torch.max(torch.abs(diff), dim=1).values

    # @classmethod
    # def _batch_dist_torch(cls, pt, batch_other, metric: str = "euclidean") -> torch.Tensor:
    #     q_tensor = torch.as_tensor(pt.q, device='cuda', dtype=torch.float16).unsqueeze(0)
    #     batch_other = batch_other.to(dtype=torch.float16)
    #     slice_indices = torch.tensor(pt.slice, device='cuda')
    #     slice_starts, slice_ends = slice_indices[:, 0], slice_indices[:, 1]

    #     max_batch_size = 4000
    #     total_size = batch_other.size(0)
    #     num_batches = (total_size + max_batch_size - 1) // max_batch_size

    #     results = []

    #     with torch.no_grad():
    #         for batch_idx in range(num_batches):
    #             # Determine batch range
    #             start_idx = batch_idx * max_batch_size
    #             end_idx = min(start_idx + max_batch_size, total_size)
    #             batch_part = batch_other[start_idx:end_idx]

    #             # Compute differences in a single operation
    #             batch_diff = q_tensor - batch_part.unsqueeze(1)  # Shape: (batch_size, 1, dim)

    #             if metric == "euclidean":
    #                 dists = torch.linalg.norm(
    #                     torch.stack([batch_diff[:, :, start:end] for start, end in zip(slice_starts, slice_ends)], dim=2),
    #                     dim=3
    #                 )  # Shape: (batch_size, num_slices, dim_slices)
    #                 dists_max = dists.max(dim=2).values  # Max distance across slices
    #             else:
    #                 # Compute max absolute differences in a single operation
    #                 dists_max = torch.max(
    #                     torch.abs(torch.stack([batch_diff[:, :, start:end] for start, end in zip(slice_starts, slice_ends)], dim=2)),
    #                     dim=3
    #                 ).values
    #             # Append batch results
    #             results.append(dists_max)

    #     # Concatenate results for all batches
    #     return torch.cat(results, dim=0)

    # @classmethod
    # def _batch_dist_torch(cls, pt, batch_other, metric: str = "euclidean") -> torch.Tensor:
    #     import torch
    #     q_tensor = torch.as_tensor(pt.q, device='cuda').unsqueeze(0)

    #     max_batch_size = 1000  # Adjust based on available GPU memory
    #     total_size = batch_other.size(0)
    #     num_batches = (total_size + max_batch_size - 1) // max_batch_size

    #     slice_indices = torch.tensor(pt.slice, device='cuda')
    #     slice_starts, slice_ends = slice_indices[:, 0], slice_indices[:, 1]

    #     results = []

    #     with torch.no_grad():
    #         for batch_idx in range(num_batches):
    #             # Process one batch
    #             start_idx = batch_idx * max_batch_size
    #             end_idx = min(start_idx + max_batch_size, total_size)
    #             batch_part = batch_other[start_idx:end_idx]

    #             max_dists_per_batch = []

    #             # Compute distances slice by slice to avoid large intermediate tensors
    #             for start, end in zip(slice_starts, slice_ends):
    #                 batch_diff = q_tensor[..., start:end] - batch_part[..., start:end]

    #                 if metric == "euclidean":
    #                     dists = torch.linalg.norm(batch_diff, dim=-1)  # Shape: (batch_size, num_points)
    #                 else:  # Chebyshev metric
    #                     dists = torch.max(torch.abs(batch_diff), dim=-1).values  # Shape: (batch_size, num_points)

    #                 max_dists_per_batch.append(dists)  # Collect max distances for this slice

    #             # Take maximum across slices for the current batch
    #             max_dists_per_batch = torch.stack(max_dists_per_batch, dim=-1).max(dim=-1).values  # Shape: (batch_size,)
    #             results.append(max_dists_per_batch)

    #             # Clear intermediate tensors to free memory
    #             del batch_part, batch_diff, dists, max_dists_per_batch
    #             torch.cuda.empty_cache()

    #     # Concatenate results across all batches
        # return torch.cat(results, dim=0)


            
    @classmethod
    def _batch_torch(cls, pt, batch_other, metric: str = "euclidean") -> torch.Tensor:
        torch.cuda.empty_cache()
        q_tensor = torch.as_tensor(pt.q, device='cuda')
        with torch.no_grad():
            diff = q_tensor - batch_other  


        if metric == "euclidean":
            dists = [
                torch.linalg.norm(diff[:, s:e], dim=1, keepdim=True) for s, e in pt.slice
            ]  
        else:  
            dists = [
                torch.max(torch.abs(diff[:, s:e]), dim=1, keepdim=True).values for s, e in pt.slice
            ] 
        dists = torch.cat(dists, dim=1)  
        max_dists = dists.max(dim=1).values 
        sum_dists = dists.sum(dim=1)  
        costs = max_dists + 0.01 * sum_dists
        return costs, dists
    
def config_dist(
    q_start: Configuration, q_end: Configuration, metric: str = "euclidean"
) -> float:
    return type(q_start)._dist(q_start, q_end, metric)

def config_dists(
    q_start: Configuration, q_end: Configuration, metric: str = "."
) -> NDArray:
    return type(q_start)._dists(q_start, q_end, metric)

def batch_config_dist(
    pt: Configuration, batch_pts: List[Configuration], metric: str = "euclidean"
) -> NDArray:
    return type(pt)._batch_dist(pt, batch_pts, metric)

def batch_config_dist_torch(
    pt: Configuration, batch_pts: torch.tensor, metric: str = "euclidean"
) -> NDArray:
    return type(pt)._batch_dist_torch(pt, batch_pts, metric)

def batch_config_torch(
    pt: Configuration, batch_pts: torch.tensor,metric: str = "euclidean"
) -> torch.Tensor:
    return type(pt)._batch_torch(pt, batch_pts, metric)

def config_cost(
    q_start: Configuration, q_end: Configuration, metric: str = "euclidean"
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

def batch_config_cost(
    starts: List[Configuration],
    batch_other: List[Configuration],
    metric: str = "euclidean",
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
