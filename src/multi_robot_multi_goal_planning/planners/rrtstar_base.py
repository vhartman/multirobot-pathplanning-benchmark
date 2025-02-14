import numpy as np
import logging
from datetime import datetime
import yaml
import json
import pickle

from multi_robot_multi_goal_planning.problems.configuration import *
from multi_robot_multi_goal_planning.problems.planning_env import *
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
)
from multi_robot_multi_goal_planning.planners import shortcutting
import time as time
import math as math
from typing import Tuple, Optional, Union
from numba import njit
from scipy.stats.qmc import Halton

class Operation:
    """ Planner operation variables"""
    def __init__(self):
        
        self.path = []
        self.path_modes = []
        self.path_nodes = None
        self.cost = np.inf
        self.cost_change = np.inf
        self.init_sol = False
        self.costs = np.empty(10000000, dtype=np.float64)
        self.paths_inter = []
    
    def get_cost(self, idx):
        """Return cost of node with specific id"""
        return self.costs[idx]

class Node:
    id_counter = 0

    def __init__(self, state:State, operation: Operation):
        self.state = state   
        self.parent = None  
        self.children = []    
        self.transition = False
        self.cost_to_parent = None
        self.operation = operation
        self.id = Node.id_counter
        Node.id_counter += 1
        self.neighbors = {}
        self.hash = None


    @property
    def cost(self):
        return self.operation.get_cost(self.id)
    
    @cost.setter
    def cost(self, value):
        """Set the cost in the shared operation costs tensor."""
        self.operation.costs[self.id] = value

    def __repr__(self):
        return f"<N- {self.state.q.state()}, c: {self.cost}>"
    
    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        if self.hash is None:
            self.hash = hash((self.state.q.state().data.tobytes(), tuple(self.state.mode.task_ids)))
        return self.hash

class BaseTree(ABC):
    def __init__(self):
        pass

    def _resize_array(self, array: np.ndarray, current_capacity: int, new_capacity: int) -> np.ndarray:
        """
        Dynamically resizes the given NumPy array to the specified new capacity.

        Args:
            array (np.ndarray): The array to resize.
            current_capacity (int): The current capacity of the array.
            new_capacity (int): The new capacity to allocate.

        Returns:
            np.ndarray: The resized array.
        """
        new_array = np.empty((new_capacity, *array.shape[1:]), dtype=array.dtype)
        new_array[:current_capacity] = array  # Copy existing data
        del array  # Free old array (Python garbage collector will handle memory)
        return new_array

    def ensure_capacity(self, array: np.ndarray, required_capacity: int) -> np.ndarray:
        """
        Ensures that the NumPy array has enough capacity to add new elements. Resizes if necessary.

        Args:
            array (np.ndarray): The array to check and potentially resize.
            required_capacity (int): The required capacity.

        Returns:
            np.ndarray: The array with ensured capacity.
        """
        current_size = array.shape[0]

        if required_capacity == current_size:
            return self._resize_array(array, current_size, required_capacity * 2)  # Double the size

        return array
    
    @abstractmethod
    def add_node(self, n:Node, tree:str) -> None:
        pass
    @abstractmethod
    def remove_node(self, n:Node, tree:str = '') -> None:
        pass
    @abstractmethod
    def get_batch_subtree(self, tree:str = '') -> NDArray:
        pass
    @abstractmethod
    def get_node_idx_subtree(self, tree:str = '') -> NDArray:
        pass
    @abstractmethod
    def add_transition_node_as_start_node(self, n:Node, tree:str = '') -> None:
        pass
    @abstractmethod
    def get_node(self, id:int, tree:str = '') -> Node:
        pass
    @abstractmethod
    def get_number_of_nodes_in_tree(self) -> int:
        pass

class SingleTree(BaseTree):
    """Single tree description"""
    def __init__(self, env):
        self.order = 1
        # self.informed = Informed()
        robot_dims = sum(env.robot_dims.values())
        self.subtree = {}
        self.initial_capacity = 100000
        self.batch_subtree = np.empty((self.initial_capacity, robot_dims), dtype=np.float64)
        self.node_idx_subtree = np.empty(self.initial_capacity, dtype=np.int64)
    def add_node(self, n:Node, tree:str = '') -> None:
        self.subtree[n.id] = n               
        position = len(self.subtree) -1
        self.batch_subtree = self.ensure_capacity(self.batch_subtree, position)
        self.batch_subtree[position,:] = n.state.q.state()
        self.node_idx_subtree = self.ensure_capacity(self.node_idx_subtree, position)
        self.node_idx_subtree[position] = n.id
    
    def remove_node(self, n:Node, tree:str = '') -> None:
        mask = self.node_idx_subtree != n.id
        self.node_idx_subtree = self.node_idx_subtree[mask] 
        self.batch_subtree = self.batch_subtree[mask]
        del self.subtree[n.id]


    def get_batch_subtree(self, tree:str = '') -> NDArray:
        return self.batch_subtree[:len(self.subtree)]
    
    def get_node_idx_subtree(self, tree:str = '') -> NDArray:
        return self.node_idx_subtree[:len(self.subtree)]

    def add_transition_node_as_start_node(self, n:Node, tree:str = '') -> None:
        if n.id not in self.subtree:
            self.add_node(n)
    
    def get_node(self, id:int, tree:str = '') -> Node:
        return self.subtree.get(id)
    def get_number_of_nodes_in_tree(self) -> int:
        return len(self.subtree)

class BidirectionalTree(BaseTree):
    """Bidirectional tree description"""
    def __init__(self, env):
        self.order = 1
        # self.informed = Informed()
        robot_dims = sum(env.robot_dims.values())
        self.subtree = {}
        self.initial_capacity = 100000
        self.batch_subtree = np.empty((self.initial_capacity, robot_dims), dtype=np.float64)
        self.node_idx_subtree = np.empty(self.initial_capacity, dtype=np.int64)
        self.subtree_b = {} 
        self.batch_subtree_b = np.empty((self.initial_capacity, robot_dims), dtype=np.float64)
        self.node_idx_subtree_b = np.empty(self.initial_capacity, dtype=np.int64)
        self.connected = False
    
    def add_node(self, n:Node, tree:str = '') -> None:
        if tree == 'A' or tree == '':
            self.subtree[n.id] = n               
            position = len(self.subtree) -1
            self.batch_subtree = self.ensure_capacity(self.batch_subtree, position)
            self.batch_subtree[position,:] = n.state.q.state()
            self.node_idx_subtree = self.ensure_capacity(self.node_idx_subtree, position)
            self.node_idx_subtree[position] = n.id
        if tree == 'B':
            self.subtree_b[n.id] = n               
            position = len(self.subtree_b) -1
            self.batch_subtree_b = self.ensure_capacity(self.batch_subtree_b, position)
            self.batch_subtree_b[position,:] = n.state.q.state()
            self.node_idx_subtree_b = self.ensure_capacity(self.node_idx_subtree_b, position)
            self.node_idx_subtree_b[position] = n.id
    
    def remove_node(self, n:Node, tree:str = '') -> None:
        if tree == 'A' or tree == '':
            mask = self.node_idx_subtree != n.id
            self.node_idx_subtree = self.node_idx_subtree[mask] 
            self.batch_subtree = self.batch_subtree[mask]
            del self.subtree[n.id]

        if tree == 'B':
            mask = self.node_idx_subtree_b != n.id
            self.node_idx_subtree_b = self.node_idx_subtree_b[mask] 
            self.batch_subtree_b = self.batch_subtree_b[mask]
            del self.subtree_b[n.id]

    def swap_tree(self) -> None:
        if not self.connected:
            self.subtree, self.subtree_b = self.subtree_b, self.subtree
            self.batch_subtree, self.batch_subtree_b = self.batch_subtree_b, self.batch_subtree
            self.node_idx_subtree, self.node_idx_subtree_b = self.node_idx_subtree_b, self.node_idx_subtree
            self.order *= -1


    def get_batch_subtree(self, tree:str = '') -> NDArray:
        if tree == 'A' or tree == '':
            return self.batch_subtree[:len(self.subtree)]
        if tree == 'B':
            return self.batch_subtree_b[:len(self.subtree_b)]
    
    def get_node_idx_subtree(self, tree:str = '') -> NDArray:
        if tree == 'A' or tree == '':
            return self.node_idx_subtree[:len(self.subtree)]
        if tree == 'B':
            return self.node_idx_subtree_b[:len(self.subtree_b)]

    def add_transition_node_as_start_node(self, n:Node, tree:str = '') -> None:
        if self.order == 1:
            if n.id not in self.subtree:
                self.add_node(n)
        else:
            if n.id not in self.subtree_b:
                self.add_node(n, 'B')

    def get_node(self, id:int, tree:str = '') -> Node:
        if tree == 'A' or tree == '':
            return self.subtree.get(id)
        if tree == 'B':
            return self.subtree_b.get(id)
        
    def swap(self) -> None:
        if not self.connected:
            self.subtree, self.subtree_b = self.subtree_b, self.subtree
            self.batch_subtree, self.batch_subtree_b = self.batch_subtree_b, self.batch_subtree
            self.node_idx_subtree, self.node_idx_subtree_b = self.node_idx_subtree_b, self.node_idx_subtree 
            self.order *= (-1)
    def get_number_of_nodes_in_tree(self) -> int:
        return len(self.subtree) + len(self.subtree_b)


class Informed(ABC):
    def __init__(self):
        pass
    
    def rotation_to_world_frame(self, start:Configuration, goal:Configuration, n:int) -> Tuple[float, NDArray, NDArray]:
        """ Returns: 
                Norm and rotation matrix C from the hyperellipsoid-aligned frame to the world frame."""
        diff = goal.state() - start.state()
        norm = np.linalg.norm(diff)
        if norm < 1e-3 :
            return None, None
        a1 = diff / norm 

        # Create first column of the identity matrix
        e1 = np.zeros(n)
        e1[0] = 1

        # Compute SVD directly on the outer product
        U, _, Vt = np.linalg.svd(np.outer(a1, e1))
        V = Vt.T

        # Construct the rotation matrix C
        det_factor = np.linalg.det(U) * np.linalg.det(V)
        C = U @ np.diag([1] * (n - 1) + [det_factor]) @ Vt

        return norm, C
    
    def initialize(self):
        for robot in self.env.robots:
            r_idx = self.env.robots.index(robot)
            self.n[r_idx] = self.env.robot_dims[robot]
            
    def cholesky_decomposition(self, r_indices:List[int], r_idx:int, path_nodes:List[Node]):
        cmax = self.get_right_side_of_eq(r_indices, r_idx, path_nodes)
        if not cmax or self.cmin[r_idx] < 0:
            return
        r1 = cmax / 2 
        r2 = np.sqrt(cmax**2 - self.cmin[r_idx]**2) / 2
        return np.diag(np.concatenate([[r1], np.repeat(r2, self.n[r_idx] - 1)])) 
    
    @abstractmethod
    def get_right_side_of_eq():
        pass

    
class InformedVersion0(Informed):
    """Local informed sampling: Only consider agent itself nothing else"""

    def __init__(self, env, cost_fct):
        self.env = env
        self.mode_task_ids_task = {}
        self.cmin = {}
        self.C = {}
        self.start = {}
        self.goal = {}
        self.mode_task_ids_home_poses = {}
        self.center = {}
        self.cost_fct = cost_fct
        self.n = {}
        self.L = {}
        self.cost = np.inf
    
    def get_right_side_of_eq(self, r_indices:List[int], r_idx:int, path_nodes:List[Node]):
        #need to calculate the cost only cnsidering one agent (euclidean distance)
        n1 = NpConfiguration.from_numpy(path_nodes[0].state.q.state()[r_indices])
        cost = 0
        if self.mode_task_ids_home_poses[r_idx] == [-1]:
            start_cost = 0
            self.start[r_idx] = n1
        for node in path_nodes[1:]:
            n2 = NpConfiguration.from_numpy(node.state.q.state()[r_indices])
            cost += self.cost_fct(n1, n2, self.env.cost_metric, self.env.cost_reduction)
            #need to work with task ids as same mode can have different configs
            if node.transition and node.state.mode.task_ids == self.mode_task_ids_home_poses[r_idx]:
                start_cost = cost
                self.start[r_idx] = n2
            if node.transition and node.state.mode.task_ids == self.mode_task_ids_task[r_idx]:
                goal_cost = cost
                self.goal[r_idx] = n2
                break 
            n1 = n2
        try:
            norm, self.C[r_idx] = self.rotation_to_world_frame(self.start[r_idx], self.goal[r_idx], self.n[r_idx])
        except:
            pass
        if norm is None:
            return
        self.cmin[r_idx]  = norm-2*self.env.collision_tolerance
        self.center[r_idx] = (self.goal[r_idx].state() + self.start[r_idx].state())/2
        return goal_cost-start_cost    
class InformedVersion1(Informed):
    """Global informed sampling: Considering whole configuration"""

    def __init__(self, env, cost_fct):
        self.env = env
        self.mode_task_ids_task = []
        self.cmin = None
        self.C = None
        self.start = None
        self.goal = None
        self.mode_task_ids_home_poses = []
        self.center = None
        self.cost_fct = cost_fct
        self.n = None
        self.L = None
        self.cost = np.inf
        self.active_robots_idx = []
    
    def get_right_side_of_eq(self, path_nodes:List[Node]):
        if len(self.active_robots_idx) == 1:
            idx = self.active_robots_idx[0]
        else:
            idx = random.choice(self.active_robots_idx)

        mode_task_ids_home_pose = self.mode_task_ids_home_poses[idx]
        mode_task_ids_task = self.mode_task_ids_task[idx]

        if mode_task_ids_home_pose == [-1]:
            self.start = path_nodes[0]

        for node in path_nodes:
            if node.transition and node.state.mode.task_ids == mode_task_ids_home_pose:
                self.start = node
            if node.transition and node.state.mode.task_ids == mode_task_ids_task:
                self.goal = node
                break 
        lb_goal = self.cost_fct(self.goal.state.q, path_nodes[-1].state.q, self.env.cost_metric, self.env.cost_reduction)
        lb_start = self.cost_fct( path_nodes[0].state.q, self.start.state.q, self.env.cost_metric, self.env.cost_reduction)
        norm, self.C = self.rotation_to_world_frame(self.start.state.q, self.goal.state.q, self.n)
        self.cmin = norm-2*self.env.collision_tolerance
        self.center = (self.goal.state.q.state() + self.start.state.q.state())/2
        return path_nodes[-1].cost - lb_goal - lb_start
    
    def initialize(self, mode:Mode, next_ids:List[int]): 
        active_robots = self.env.get_active_task(mode, next_ids).robots 
        for robot in self.env.robots:
            if robot in active_robots:
                self.active_robots_idx.append(self.env.robots.index(robot))
        self.n = sum(self.env.robot_dims.values())

    def cholesky_decomposition(self, path_nodes:List[Node]):
        cmax = self.get_right_side_of_eq( path_nodes)
        r1 = cmax / 2
        r2 = np.sqrt(cmax**2 - self.cmin**2) / 2
        return np.diag(np.concatenate([np.repeat(r1, 1), np.repeat(r2, self.n - 1)]))   
class InformedVersion2(Informed):
    """Global informed smapling: Each agent separately (Similar to version 1)"""

    def __init__(self, env, cost_fct):
        self.env = env
        self.mode_task_ids_task = {}
        self.cmin = {}
        self.C = {}
        self.start = {}
        self.goal = {}
        self.mode_task_ids_home_poses = {}
        self.center = {}
        self.cost_fct = cost_fct
        self.n = {}
        self.L = {}
        self.cost = np.inf
    
    def get_right_side_of_eq(self, r_indices:List[int], r_idx:int, path_nodes:List[Node]):
        if self.mode_task_ids_home_poses[r_idx] == [-1]:
            start_node = path_nodes[0]
            self.start[r_idx] = NpConfiguration.from_numpy(start_node.state.q.state()[r_indices])
        for node in path_nodes[1:]:
            #need to work with task ids as same mode can have different configs
            if node.transition and node.state.mode.task_ids == self.mode_task_ids_home_poses[r_idx]:
                start_node = node
                self.start[r_idx] = NpConfiguration.from_numpy(node.state.q.state()[r_indices])
            if node.transition and node.state.mode.task_ids == self.mode_task_ids_task[r_idx]:
                goal_node = node
                self.goal[r_idx] = NpConfiguration.from_numpy(node.state.q.state()[r_indices])
                break 

        lb_goal = self.cost_fct(goal_node.state.q, path_nodes[-1].state.q, self.env.cost_metric, self.env.cost_reduction)
        lb_start = self.cost_fct( path_nodes[0].state.q, start_node.state.q, self.env.cost_metric, self.env.cost_reduction)
        norm, self.C[r_idx] = self.rotation_to_world_frame(self.start[r_idx], self.goal[r_idx], self.n[r_idx])
        if norm is None:
            return
        self.cmin[r_idx]  = norm-2*self.env.collision_tolerance
        self.center[r_idx] = (self.goal[r_idx].state() + self.start[r_idx].state())/2
        return path_nodes[-1].cost - lb_goal - lb_start    
class InformedVersion3(Informed):
    """Local informed sampling: Each agent separately (Similar to version 1)"""

    def __init__(self, env, cost_fct):
        self.env = env
        self.mode_task_ids_task = {}
        self.cmin = {}
        self.C = {}
        self.start = {}
        self.goal = {}
        self.mode_task_ids_home_poses = {}
        self.center = {}
        self.cost_fct = cost_fct
        self.n = {}
        self.L = {}
        self.cost = np.inf
    
    def get_right_side_of_eq(self, r_indices:List[int], r_idx:int, path_nodes:List[Node]):
        if self.mode_task_ids_home_poses[r_idx] == [-1]:
            start_node = path_nodes[0]
            self.start[r_idx] = NpConfiguration.from_numpy(start_node.state.q.state()[r_indices])
            start_idx = 0
        for idx , node in enumerate(path_nodes[1:]):
            #need to work with task ids as same mode can have different configs
            if node.transition and node.state.mode.task_ids == self.mode_task_ids_home_poses[r_idx]:
                start_node = node
                self.start[r_idx] = NpConfiguration.from_numpy(node.state.q.state()[r_indices])
                start_idx = idx
            if node.transition and node.state.mode.task_ids == self.mode_task_ids_task[r_idx]:
                goal_node = node
                self.goal[r_idx] = NpConfiguration.from_numpy(node.state.q.state()[r_indices])
                goal_idx = idx
                break 
        idx1 = np.random.randint(0, start_idx+1)
        idx2 = np.random.randint(goal_idx, len(path_nodes))


        lb_goal = self.cost_fct(goal_node.state.q, path_nodes[idx2].state.q, self.env.cost_metric, self.env.cost_reduction)
        lb_start = self.cost_fct( path_nodes[idx1].state.q, start_node.state.q, self.env.cost_metric, self.env.cost_reduction)
        norm, self.C[r_idx] = self.rotation_to_world_frame(self.start[r_idx], self.goal[r_idx], self.n[r_idx])
        if norm is None:
            return
        self.cmin[r_idx]  = norm-2*self.env.collision_tolerance
        self.center[r_idx] = (self.goal[r_idx].state() + self.start[r_idx].state())/2
        
        return path_nodes[idx2].cost - path_nodes[idx1].cost - lb_goal - lb_start   
class InformedVersion4(Informed):
    """Global informed sampling (random): Each agent separately (my version)"""

    def __init__(self, env, cost_fct):
        self.env = env
        self.mode_task_ids_task = {}
        self.cmin = {}
        self.C = {}
        self.start = {}
        self.goal = {}
        self.mode_task_ids_home_poses = {}
        self.center = {}
        self.cost_fct = cost_fct
        self.n = {}
        self.L = {}
        self.cost = np.inf
    
    def get_right_side_of_eq(self, r_indices:List[int], r_idx:int, path_nodes:List[Node]):
        while True:
            idx1 = np.random.randint(0, len(path_nodes))
            idx2 = np.random.randint(0, len(path_nodes))
            if idx2 < idx1:
                idx1, idx2 = idx2, idx1
            if idx2 -idx1 > 2:
                break
        goal = path_nodes[idx2]
        start = path_nodes[idx1]
        self.start[r_idx] = NpConfiguration.from_numpy(start.state.q.state()[r_indices])
        self.goal[r_idx] = NpConfiguration.from_numpy(goal.state.q.state()[r_indices])

        lb_goal = self.cost_fct(goal.state.q, path_nodes[-1].state.q, self.env.cost_metric, self.env.cost_reduction)
        lb_start = self.cost_fct( path_nodes[0].state.q, start.state.q, self.env.cost_metric, self.env.cost_reduction)
        norm, self.C[r_idx] = self.rotation_to_world_frame(self.start[r_idx], self.goal[r_idx], self.n[r_idx])
        if norm is None:
            return
        self.cmin[r_idx]  = norm-2*self.env.collision_tolerance
        self.center[r_idx] = (self.goal[r_idx].state() + self.start[r_idx].state())/2
        return path_nodes[-1].cost - lb_goal - lb_start
class InformedVersion5(Informed):
    """Global informed smapling: Each agent separately (Similar to version 1)"""

    def __init__(self, env, cost_fct):
        self.env = env
        self.mode_task_ids_task = {}
        self.cmin = {}
        self.C = {}
        self.start = {}
        self.goal = {}
        self.mode_task_ids_home_poses = {}
        self.center = {}
        self.cost_fct = cost_fct
        self.n = {}
        self.L = {}
        self.cost = np.inf
    
    def get_right_side_of_eq(self, r_indices:List[int], r_idx:int, path_nodes:List[Node]):
        if self.mode_task_ids_home_poses[r_idx] == [-1]:
            start_node = path_nodes[0]
            self.start[r_idx] = NpConfiguration.from_numpy(start_node.state.q.state()[r_indices])
        for node in path_nodes[1:]:
            #need to work with task ids as same mode can have different configs
            if node.transition and node.state.mode.task_ids == self.mode_task_ids_home_poses[r_idx]:
                start_node = node
                self.start[r_idx] = NpConfiguration.from_numpy(node.state.q.state()[r_indices])
            if node.transition and node.state.mode.task_ids == self.mode_task_ids_task[r_idx]:
                goal_node = node
                self.goal[r_idx] = NpConfiguration.from_numpy(node.state.q.state()[r_indices])
                break 

        norm, self.C[r_idx] = self.rotation_to_world_frame(self.start[r_idx], self.goal[r_idx], self.n[r_idx])
        if norm is None:
            return
        self.cmin[r_idx]  = norm-2*self.env.collision_tolerance
        self.center[r_idx] = (self.goal[r_idx].state() + self.start[r_idx].state())/2
        return goal_node.cost - start_node.cost

@njit(fastmath=True, cache=True)
def find_nearest_indices(set_dists, r):
    r += 1e-10 #float issues
    return np.nonzero(set_dists <= r)[0]
@njit
def cumulative_sum(batch_cost):
    cost = np.empty(len(batch_cost), dtype=np.float64) 
    for idx in range(0, len(batch_cost)):
        cost[idx] = np.sum(batch_cost[:idx+1])
    return cost
@njit
def get_mode_task_ids_of_active_task_in_path(path_modes, task_id:Task, r_idx:int):
    last_index = 0 
    for i in range(len(path_modes)):
        if path_modes[i][r_idx] == task_id:  
            last_index = i 
    return path_modes[last_index]
@njit
def get_mode_task_ids_of_home_pose_in_path(path_modes, task_id:Task, r_idx:int):
    for i in range(len(path_modes)):
        if path_modes[i][r_idx] == task_id:  
            if i == 0:
                return np.array([-1])
            return path_modes[i-1]


class BaseRRTstar(ABC):
    def __init__(self, 
                 env:BaseProblem,
                 ptc: PlannerTerminationCondition, 
                 general_goal_sampling: bool = False, 
                 informed_sampling: bool = False, 
                 informed_sampling_version: int = 0, 
                 distance_metric: str = 'max_euclidean',
                 p_goal: float = 0.9, 
                 p_stay: float = 0.3,
                 p_uniform: float = 0.8, 
                 shortcutting: bool = True, 
                 mode_sampling: Optional[Union[int, float]] = None, 
                 gaussian: bool = True,
                 shortcutting_dim_version = 2, 
                 shortcutting_robot_version = 1 
                 ):
        self.env = env
        self.ptc = ptc
        self.general_goal_sampling = general_goal_sampling
        self.informed_sampling = informed_sampling
        self.informed_sampling_version = informed_sampling_version
        self.distance_metric = distance_metric
        self.p_goal = p_goal
        self.shortcutting = shortcutting
        self.mode_sampling = mode_sampling
        self.p_uniform = p_uniform
        self.gaussian = gaussian
        self.p_stay = p_stay
        self.shortcutting_dim_version = shortcutting_dim_version
        self.shortcutting_robot_version = shortcutting_robot_version
        self.eta = np.sqrt(sum(self.env.robot_dims.values())/len(self.env.robots))
        self.operation = Operation()
        self.start_single_goal= SingleGoal(self.env.start_pos.q)
        self.modes = [] 
        self.trees = {}
        self.transition_node_ids = {}
        self.informed = {}
        self.start_time = time.time()
        self.costs = [] 
        self.times = []
        self.all_paths = []


    def add_tree(self, mode: Mode, tree_instance: Optional[Union["SingleTree", "BidirectionalTree"]] = None) -> None:
        """Initilaizes a new tree instance"""
        if tree_instance is None:
            raise ValueError("You must provide a tree instance type: SingleTree or BidirectionalTree.")
        
        # Check type and initialize the tree
        if tree_instance == SingleTree:
            self.trees[mode] = SingleTree(self.env)
        elif tree_instance == BidirectionalTree:
            self.trees[mode] = BidirectionalTree(self.env)
        else:
            raise TypeError("tree_instance must be SingleTree or BidirectionalTree.")
           
    def add_new_mode(self, q:Optional[Configuration]=None, mode:Mode=None, tree_instance: Optional[Union["SingleTree", "BidirectionalTree"]] = None) -> None: #TODO entry_configuration needs to be specified
        """Initializes a new mode"""
        if mode is None: 
            new_mode = self.env.make_start_mode()
            new_mode.prev_mode = None
        else:
            new_mode = self.env.get_next_mode(q, mode)
            new_mode.prev_mode = mode
        if new_mode in self.modes:
            return 
        self.modes.append(new_mode)
        self.add_tree(new_mode, tree_instance)
        self.InformedInitialization(new_mode)

    def mark_node_as_transition(self, mode:Mode, n:Node) -> None:
        """Mark node as potential transition node in mode"""
        n.transition = True
        if mode not in self.transition_node_ids:
            self.transition_node_ids[mode] = []
        if n.id not in self.transition_node_ids[mode]:
            self.transition_node_ids[mode].append(n.id)

    def convert_node_to_transition_node(self, mode:Mode, n:Node) -> None:
        """Mark node as potential transition node in mode and and add it as start node to next mode """
        self.mark_node_as_transition(mode,n)
        for next_mode in mode.next_modes:
            self.trees[next_mode].add_transition_node_as_start_node(n)

    def get_lb_transition_node_id(self, modes:List[Mode]) -> Tuple[float, int]:
        """Returns lb cost and index of transition nodes of the mode"""
        indices, costs = [], []
        for mode in modes:
            i = np.argmin(self.operation.costs[self.transition_node_ids[mode]], axis=0) 
            indices.append(i)
            costs.append(self.operation.costs[self.transition_node_ids[mode]][i])
        idx = np.argmin(costs, axis=0) 
        m = modes[idx]
        node_id = self.transition_node_ids[m][indices[idx]]
        return (costs[idx], node_id), m
        # sorted_indices = costs.argsort()
        # for idx in sorted_indices:       
            
        #     if self.trees[mode].order == 1 and node_id in self.trees[mode].subtree:
                
        #     elif self.trees[mode].order == -1 and node_id in self.trees[mode].subtree_b:
        #         return (costs[idx], node_id)
           
    def get_transition_node(self, mode:Mode, id:int)-> Node:
        """Returns transition node corresponding to id in main subtree A"""
        if self.trees[mode].order == 1:
            return self.trees[mode].subtree.get(id)
        else:
            return self.trees[mode].subtree_b.get(id)

    def get_lebesgue_measure_of_free_configuration_space(self, num_samples=10000):
        """Estimate the Lebesgue measure of C_free using Halton sequence sampling."""
        total_volume = 1.0
        limits = []
        
        for robot in self.env.robots:
            r_indices = self.env.robot_idx[robot]  # Get joint indices for the robot
            lims = self.env.limits[:, r_indices]  # Extract joint limits
            limits.append(lims)
            total_volume *= np.prod(lims[1] - lims[0])  # Compute volume product

        # Generate Halton sequence samples
        halton_sampler = Halton(self.d, scramble=False)
        halton_samples = halton_sampler.random(num_samples)  # Scaled [0,1] samples

        # Map Halton samples to configuration space
        free_samples = 0
        q_robots = np.array([
            lims[0] + halton_samples[:, self.env.robot_idx[robot]] * (lims[1] - lims[0])
            for robot, lims in zip(self.env.robots, limits)
        ])
        # idx = 0
        # q_ellipse = []
        for i in range(num_samples):
            q = q_robots[:,i]              
            q = type(self.env.get_start_pos()).from_list(q)
            # if idx < 800:
            #     q_ellipse.append(q.state())
            #     idx+=1

            # Check if sample is collision-free
            if self.env.is_collision_free_without_mode(q):
                free_samples += 1
        # self.SaveData(None, time.time()-self.start_time, ellipse=q_ellipse)
        # Estimate C_free measure
        self.c_free = (free_samples / num_samples) * total_volume
        
    def set_gamma_rrtstar(self):
        self.d = sum(self.env.robot_dims.values())
        unit_ball_volume = math.pi ** (self.d/ 2) / math.gamma((self.d / 2) + 1)
        self.get_lebesgue_measure_of_free_configuration_space()
        self.gamma_rrtstar = ((2 *(1 + 1/self.d))**(1/self.d) * (self.c_free/unit_ball_volume)**(1/self.d))*self.eta
    
    def get_next_ids(self, mode:Mode):
        possible_next_task_combinations = self.env.get_valid_next_task_combinations(mode)
        if len(possible_next_task_combinations) == 0:
            return None
        return random.choice(possible_next_task_combinations)

    # @cache
    def get_home_poses(self, mode:Mode) -> List[NDArray]:
        """Returns home pose (most recent completed task) of agents in current mode """
        # Start mode
        if mode.prev_mode is None: 
            q = self.env.start_pos
            q_new = []
            for r_idx in range(len(self.env.robots)):
                q_new.append(q.robot_state(r_idx))
        # all other modes
        else:
            previous_task = self.env.get_active_task(mode.prev_mode, mode.task_ids) 
            goal = previous_task.goal.sample(mode.prev_mode)
            q = self.get_home_poses(mode.prev_mode)
            q_new = []
            end_idx = 0
            for robot in self.env.robots:
                r_idx = self.env.robots.index(robot)
                if robot in previous_task.robots:
                    dim = self.env.robot_dims[robot]
                    indices = list(range(end_idx, end_idx + dim))
                    q_new.append(goal[indices])
                    end_idx += dim 
                    continue
                q_new.append(q[r_idx])
        return q_new
    # @cache
    def get_task_goal_of_agent(self, mode:Mode, r:str):
        """Returns task goal of agent in current mode"""
        # task = self.env.get_active_task(mode, self.get_next_ids(mode)) 
        # if r not in task.robots:
        r_idx = self.env.robots.index(r)
        goal = self.env.tasks[mode.task_ids[r_idx]].goal.sample(mode)
        if len(goal) == self.env.robot_dims[r]:
            return goal
        else:
            constrained_robot = self.env.get_active_task(mode, self.get_next_ids(mode)).robots
            end_idx = 0
            for robot in constrained_robot:
                dim = self.env.robot_dims[r]
                if robot == r:
                    indices = list(range(end_idx, end_idx + dim))
                    return goal[indices]
                end_idx += dim 
        # goal = task.goal.sample(mode)
        # if len(goal) == self.env.robot_dims[r]:
        #    return goal
        # else:
        #     return goal[self.env.robot_idx[r]]
    # @cache
    def sample_transition_configuration(self, mode)-> Configuration:
        """Returns transition node of mode"""
        constrained_robot = self.env.get_active_task(mode, self.get_next_ids(mode)).robots
        while True:
            goal = self.env.get_active_task(mode, self.get_next_ids(mode)).goal.sample(mode)
            q = []
            end_idx = 0
            for robot in self.env.robots:
                if robot in constrained_robot:
                    dim = self.env.robot_dims[robot]
                    indices = list(range(end_idx, end_idx + dim))
                    q.append(goal[indices])
                    end_idx += dim 
                    continue
                lims = self.env.limits[:, self.env.robot_idx[robot]]
                q.append(np.random.uniform(lims[0], lims[1]))
            q = type(self.env.get_start_pos()).from_list(q)   
            if self.env.is_collision_free(q, mode):
                return q
    
    def sample_configuration(self, mode:Mode, sampling_type: int, transition_node_ids:Dict[Mode, List[int]] = None, tree_order:int = 1) -> Configuration:
        is_goal_sampling = sampling_type == 2
        is_informed_sampling = sampling_type == 1
        is_home_pose_sampling = sampling_type == 3
        is_gaussian_sampling = sampling_type == 4
        constrained_robots = self.env.get_active_task(mode, self.get_next_ids(mode)).robots
        attemps = 0  # if home poses are in collision

        while True:
            #goal sampling
            #TODO only needed for parallized rrtstar
            # if is_goal_sampling and not self.operation.init_sol and self.planner == "rrtstar_par":
            #     if self.env.is_terminal_mode(mode):
            #        return self.sample_transition_configuration(mode) 
            #     transition_nodes_id = transition_node_ids[mode]                  
            #     node_id = np.random.choice(transition_nodes_id)
            #     node = self.trees[self.modes[mode.id +1]].subtree.get(node_id)#TODO not applicable when several configurations are possible
            #     return node.state.q
            #goal sampling
            if is_goal_sampling and tree_order == -1:
                if mode.prev_mode is None: 
                    return self.env.start_pos
                else: 
                    transition_nodes_id = transition_node_ids[mode.prev_mode]
                    if transition_nodes_id == []:
                        return self.sample_transition_configuration(mode.prev_mode)
                        
                    else:
                        node_id = np.random.choice(transition_nodes_id)
                        node = self.trees[mode.prev_mode].subtree.get(node_id)
                        if node is None:
                            node = self.trees[mode.prev_mode].subtree_b.get(node_id)
                        return node.state.q
            if is_goal_sampling:
                goal_sample = []
                q = self.sample_transition_configuration(mode)
                goal_sample.append(q)
                while True:
                    q_noise = []
                    for r in range(len(self.env.robots)):
                        q_robot = q.robot_state(r)
                        noise = np.random.normal(0, 0.1, q_robot.shape)
                        q_noise.append(q_robot + noise)
                    q = type(self.env.get_start_pos()).from_list(q_noise)
                    if self.env.is_collision_free(q, mode):
                        goal_sample.append(q)
                        break    
                return random.choice(goal_sample)
                
            
            if not is_informed_sampling and not is_gaussian_sampling:
                q = []
                if is_home_pose_sampling:
                    attemps += 1
                    q_home = self.get_home_poses(mode)
                for robot in self.env.robots:
                    #home pose sampling
                    if is_home_pose_sampling:
                        r_idx = self.env.robots.index(robot)
                        if robot not in constrained_robots: # can cause problems if several robots are not constrained and their home poses are in collision
                            q.append(q_home[r_idx])
                            continue
                        if np.array_equal(self.get_task_goal_of_agent(mode, robot), q_home[r_idx]):
                            if np.random.uniform(0, 1) > self.p_goal: # goal sampling
                                q.append(q_home[r_idx])
                                continue
                    #uniform sampling
                    lims = self.env.limits[:, self.env.robot_idx[robot]]
                    q.append(np.random.uniform(lims[0], lims[1]))

            #informed sampling
            if is_informed_sampling:
                q = self.sample_informed(mode)
            #gaussian noise
            if is_gaussian_sampling: 
                path_state = np.random.choice(self.operation.path)
                standar_deviation = np.random.uniform(0, 5.0)
                # standar_deviation = 0.5
                noise = np.random.normal(0, standar_deviation, path_state.q.state().shape)
                q = (path_state.q.state() + noise).tolist()

            q = type(self.env.get_start_pos()).from_list(q)
            if self.env.is_collision_free(q, mode):
                return q
            if attemps > 100: # if home pose causes failed attemps
                is_home_pose_sampling = False
    
    def sample_informed(self, mode:Mode) -> None:
        """Returns: 
                Samples a point from the ellipsoidal subset defined by the start and goal positions and c_best.
        """
        if not self.informed_sampling_version == 1:
            q_rand = []
            update = False
            if self.operation.cost != self.informed[mode].cost:
                self.informed[mode].cost = self.operation.cost
                update = True
            for robot in self.env.robots:
                r_idx = self.env.robots.index(robot)
                r_indices = self.env.robot_idx[robot]
                if update:
                    self.informed[mode].mode_task_ids_home_poses[r_idx] = get_mode_task_ids_of_home_pose_in_path(np.array(self.operation.path_modes), mode.task_ids[r_idx], r_idx).tolist()
                    self.informed[mode].mode_task_ids_task[r_idx] = get_mode_task_ids_of_active_task_in_path(np.array(self.operation.path_modes), mode.task_ids[r_idx], r_idx).tolist()  
                    if not self.informed_sampling_version == 4 and not self.informed_sampling_version == 3:
                        self.informed[mode].L[r_idx] = self.informed[mode].cholesky_decomposition(r_indices, r_idx ,self.operation.path_nodes)
                if self.informed_sampling_version == 4 or self.informed_sampling_version == 3:
                    self.informed[mode].L[r_idx] = self.informed[mode].cholesky_decomposition(r_indices, r_idx ,self.operation.path_nodes)
                if self.informed[mode].L[r_idx] is None:
                    lims = self.env.limits[:, self.env.robot_idx[robot]]
                    q_rand.append(np.random.uniform(lims[0], lims[1]))
                    continue
                    
                # amount_of_failed_attemps = 0
                while True:
                    x_ball = self.sample_unit_n_ball(self.informed[mode].n[r_idx])  
                    x_rand = self.informed[mode].C[r_idx] @ (self.informed[mode].L[r_idx] @ x_ball) + self.informed[mode].center[r_idx]
                    # Check if x_rand is within limits
                    lims = self.env.limits[:, r_indices]
                    if np.all((lims[0] <= x_rand) & (x_rand <= lims[1])):  
                        q_rand.append(x_rand)
                        # print(amount_of_failed_attemps)
                        break
            #         # amount_of_failed_attemps += 1
            # i = 0
            # q_ellipse = []
            
            # while i < 800:
            #     q_ellipse_ = np.empty((1, sum(self.env.robot_dims.values())))
            #     for robot in self.env.robots:
            #         r_idx = self.env.robots.index(robot)
            #         r_indices = self.env.robot_idx[robot]
            #         try:
            #             x_ball = self.sample_unit_n_ball(self.informed[mode].n[r_idx])  
            #             x_rand = self.informed[mode].C[r_idx] @ (self.informed[mode].L[r_idx] @ x_ball) + self.informed[mode].center[r_idx]
            #             q_ellipse_[0, r_indices] = x_rand
            #         except:
            #             q_ellipse_[0, r_indices] = self.get_task_goal_of_agent(mode, robot)
            #     q_ellipse.append(q_ellipse_[0])
            #     i+=1 
            # self.SaveData(mode, time.time()-self.start_time, ellipse=q_ellipse)
            return q_rand
        else: 
            next_ids = self.get_next_ids(mode)
            self.informed[mode].initialize(mode, next_ids)
            q_rand = []
            update = False
            if self.operation.cost != self.informed[mode].cost:
                self.informed[mode].cost = self.operation.cost
                update = True
            if update:
                for robot in self.env.robots:
                    r_idx = self.env.robots.index(robot)
                    self.informed[mode].mode_task_ids_home_poses.append(get_mode_task_ids_of_home_pose_in_path(np.array(self.operation.path_modes), mode.task_ids[r_idx], r_idx).tolist())
                    self.informed[mode].mode_task_ids_task.append(get_mode_task_ids_of_active_task_in_path(np.array(self.operation.path_modes), mode.task_ids[r_idx], r_idx).tolist()) 
                #repeat every time because of randomness 
                self.informed[mode].L = self.informed[mode].cholesky_decomposition(self.operation.path_nodes)
            while True:
                x_ball = self.sample_unit_n_ball(self.informed[mode].n)  
                x_rand = self.informed[mode].C @ (self.informed[mode].L @ x_ball) + self.informed[mode].center
                # Check if x_rand is within limits
                for robot in self.env.robots:
                    r_indices = self.env.robot_idx[robot]
                    lims = self.env.limits[:, r_indices]
                    #rejection sampling -> is criteria met?
                    if np.all((lims[0] <= x_rand[r_indices]) & (x_rand[r_indices] <= lims[1])):  
                        q_rand.append(x_rand[r_indices])
                    else:
                        q_rand = []
                        break
                if q_rand != []:
                    # i = 0
                    # q_ellipse = []
                    # while i < 800:
                    #     x_ball = self.sample_unit_n_ball(self.informed[mode].n)  
                    #     x_rand = self.informed[mode].C @ (self.informed[mode].L @ x_ball) + self.informed[mode].center
                    #     q_ellipse.append(x_rand)
                    #     i+=1 
                    # self.SaveData(mode, time.time()-self.start_time, ellipse=q_ellipse)
                    return q_rand
             
    def sample_unit_n_ball(self, n:int) -> NDArray:
        """Returns:
                Uniform sample from the volume of an n-ball of unit radius centred at origin
        """
        x_ball = np.random.normal(0, 1, n) 
        x_ball /= np.linalg.norm(x_ball, ord=2)     # Normalize with L2 norm
        radius = np.random.rand()**(1 / n)          # Generate a random radius and apply power
        return x_ball * radius
    
    def get_termination_modes(self) -> List[Mode]:
        termination_modes = []
        for mode in self.modes:
            if self.env.is_terminal_mode(mode):
                termination_modes.append(mode)
        return termination_modes

    def Nearest(self, mode:Mode, q_rand: Configuration, tree: str = '') -> Node:
        set_dists = batch_config_dist(q_rand, self.trees[mode].get_batch_subtree(tree), self.distance_metric)
        idx = np.argmin(set_dists)
        node_id = self.trees[mode].get_node_idx_subtree(tree)[idx]
        # print([float(set_dists[idx])])
        return  self.trees[mode].get_node(node_id, tree), set_dists[idx], set_dists
    
    def Steer(self, mode:Mode, n_nearest: Node, q_rand: Configuration, dist: NDArray, i=1) -> State: 
        if np.equal(n_nearest.state.q.state(), q_rand.state()).all():
            return None
        q_nearest = n_nearest.state.q.state()
        q_rand = q_rand.state()
        direction = q_rand - q_nearest
        N = float((dist / self.eta)) # to have exactly the step size

        if N <= 1 or int(N) == i-1:#for bidirectional or drrt
            q_new = q_rand
        else:
            q_new = q_nearest + (direction * (i /N))
        state_new = State(type(self.env.get_start_pos())(q_new, n_nearest.state.q.array_slice), mode)
        return state_new
    
    def Near(self, mode:Mode, n_new: Node, set_dists=None):      
        batch_subtree = self.trees[mode].get_batch_subtree()
        if set_dists is None:
            set_dists = batch_config_dist(n_new.state.q, batch_subtree, self.distance_metric)
        vertices = self.trees[mode].get_number_of_nodes_in_tree()
        r = np.minimum(self.gamma_rrtstar*(np.log(vertices)/vertices)**(1/self.d), self.eta)
        if r == 0 and set_dists.size == 1: # when only one vertex is in the tree
            r = set_dists[0] 
        indices = find_nearest_indices(set_dists, r) # indices of batch_subtree
        # if indices.size == 0:
        #     print("-")
        node_indices = self.trees[mode].node_idx_subtree[indices]
        n_near_costs = self.operation.costs[node_indices]
        N_near_batch = batch_subtree[indices]
        return N_near_batch, n_near_costs, node_indices
    
    def FindParent(self, mode:Mode, node_indices: NDArray, n_new: Node, n_nearest: Node, batch_cost: NDArray, n_near_costs: NDArray) -> None:
        idx =  np.where(np.array(node_indices) == n_nearest.id)[0][0]
        c_new_tensor = n_near_costs + batch_cost
        c_min = c_new_tensor[idx]
        c_min_to_parent = batch_cost[idx]
        n_min = n_nearest
        valid_mask = c_new_tensor < c_min
        if np.any(valid_mask):
            sorted_indices = np.where(valid_mask)[0][np.argsort(c_new_tensor[valid_mask])]
            for idx in sorted_indices:
                node = self.trees[mode].subtree.get(node_indices[idx].item())
                if self.env.is_edge_collision_free(node.state.q, n_new.state.q, mode):
                    c_min = c_new_tensor[idx]
                    c_min_to_parent = batch_cost[idx]      # Update minimum cost
                    n_min = node                            # Update parent node
                    break
        n_new.parent = n_min
        n_new.cost_to_parent = c_min_to_parent
        n_min.children.append(n_new) #Set child
        self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, n_new.id) 
        n_new.cost = c_min
        self.trees[mode].add_node(n_new)
    
    def Rewire(self, mode:Mode,  node_indices: NDArray, n_new: Node, batch_cost: NDArray, 
               n_near_costs: NDArray, n_rand = None, n_nearest = None) -> bool:
        rewired = False
        c_potential_tensor = n_new.cost + batch_cost

        improvement_mask = c_potential_tensor < n_near_costs
        
        if np.any(improvement_mask):
            improved_indices = np.nonzero(improvement_mask)[0]

            for idx in improved_indices:
                n_near = self.trees[mode].subtree.get(node_indices[idx].item())
                if n_near == n_new.parent or n_near.cost == np.inf or n_near == n_new:
                    continue

                if self.env.is_edge_collision_free(n_new.state.q, n_near.state.q, mode):
                    if n_near.parent is not None:
                        n_near.parent.children.remove(n_near)
                    n_near.parent = n_new                    
                    n_new.children.append(n_near)

                    n_near.cost = c_potential_tensor[idx]
                    n_near.cost_to_parent = batch_cost[idx]
                    rewired = True
        return rewired
      
    def GeneratePath(self, mode:Mode, n: Node, shortcutting_bool:bool = True) -> None:
        path_nodes, path, path_modes, path_shortcutting = [], [], [], []
        while n:
            path_nodes.append(n)
            path_modes.append(n.state.mode.task_ids)
            path.append(n.state)
            if shortcutting_bool:
                path_shortcutting.append(n.state)
                if n.parent is not None and n.parent.state.mode != n.state.mode:
                    new_state = State(n.parent.state.q, n.state.mode)
                    path_shortcutting.append(new_state)
            n = n.parent
        path_in_order = path[::-1]
        self.operation.path_modes = path_modes[::-1]
        self.operation.path = path_in_order  
        self.operation.path_nodes = path_nodes[::-1]
        self.operation.cost = self.operation.path_nodes[-1].cost
        self.costs.append(self.operation.cost)
        self.times.append(time.time() - self.start_time)
        self.all_paths.append(self.operation.path)
        if self.operation.init_sol and self.shortcutting and shortcutting_bool:
            path_shortcutting_in_order = path_shortcutting[::-1]
            # print(f"-- M", mode.task_ids, "Cost: ", self.operation.cost.item())
            shortcut_path, _ = shortcutting.robot_mode_shortcut(
                                self.env,
                                path_shortcutting_in_order,
                                500,
                                resolution=self.env.collision_resolution,
                                tolerance=self.env.collision_tolerance,
                            )

            batch_cost = batch_config_cost(shortcut_path[:-1], shortcut_path[1:], self.env.cost_metric, self.env.cost_reduction)
            shortcut_path_costs = cumulative_sum(batch_cost)
            shortcut_path_costs = np.insert(shortcut_path_costs, 0, 0.0)
            if shortcut_path_costs[-1] < self.operation.cost:
                self.TreeExtension(mode, shortcut_path, shortcut_path_costs)
            
    def RandomMode(self) -> List[float]:
        num_modes = len(self.modes)
        if num_modes == 1:
            return np.random.choice(self.modes)
        # if self.operation.task_sequence == [] and self.mode_sampling != 0:
        elif self.operation.init_sol and self.mode_sampling != 0:
                p = [1/num_modes] * num_modes
        
        elif self.mode_sampling == None:
            # equally (= mode uniformly)
            return np.random.choice(self.modes)

        elif self.mode_sampling == 1: #can cause some problem ...
            # greedy (only latest mode is selected until initial paths are found and then it continues with equally)
            probability = [0] * (num_modes)
            probability[-1] = 1
            p =  probability

        elif self.mode_sampling == 0:#TODO not working properly for bidirectional
            # Uniformly
            total_transition_nodes = sum(len(mode) for mode in self.transition_node_ids.values())
            total_nodes = Node.id_counter -1 + total_transition_nodes
            # Calculate probabilities inversely proportional to node counts
            inverse_probabilities = [
                1 - (len(self.trees[mode].subtree) / total_nodes)
                for mode in self.modes
            ]
            # Normalize the probabilities to sum to 1
            total_inverse = sum(inverse_probabilities)
            p =   [
                inv_prob / total_inverse for inv_prob in inverse_probabilities
            ]

        else:
            # manually set
            total_transition_nodes = sum(len(mode) for mode in self.transition_node_ids.values())
            total_nodes = sum(len(self.trees[mode].subtree) for mode in self.modes[:-1]) - total_transition_nodes
            # Calculate probabilities inversely proportional to node counts
            inverse_probabilities = [
                1 - (len(self.trees[mode].subtree) / total_nodes)
                for mode in self.modes[:-1]
            ]

            # Normalize the probabilities of all modes except the last one
            remaining_probability = 1-self.mode_sampling  
            total_inverse = sum(inverse_probabilities)
            p =  [
                (inv_prob / total_inverse) * remaining_probability
                for inv_prob in inverse_probabilities
            ] + [self.mode_sampling]

        return np.random.choice(self.modes, p = p)

    def InformedInitialization(self, mode: Mode) -> None: 
        if not self.informed_sampling:
            return
        if self.informed_sampling_version == 0:
            self.informed[mode] = InformedVersion0(self.env, config_cost)
            self.informed[mode].initialize()
        if self.informed_sampling_version == 1:
            self.informed[mode] = InformedVersion1(self.env, config_cost)
        if self.informed_sampling_version == 2:
            self.informed[mode] = InformedVersion2(self.env, config_cost)
            self.informed[mode].initialize()
        if self.informed_sampling_version == 3:
            self.informed[mode] = InformedVersion3(self.env, config_cost)
            self.informed[mode].initialize()
        if self.informed_sampling_version == 4:
            self.informed[mode] = InformedVersion4(self.env, config_cost)
            self.informed[mode].initialize()
        if self.informed_sampling_version == 5:
            self.informed[mode] = InformedVersion5(self.env, config_cost)
            self.informed[mode].initialize()

    def SampleNodeManifold(self, mode:Mode) -> Configuration:
        if np.random.uniform(0, 1) <= self.p_goal:
            #informed_sampling
            if self.informed_sampling and self.operation.init_sol: 
                if self.informed_sampling_version == 0 and np.random.uniform(0, 1) > self.p_uniform or self.informed_sampling_version == 5 and np.random.uniform(0, 1) > self.p_uniform:
                    #uniform sampling
                    return self.sample_configuration(mode, 0)
                return self.sample_configuration(mode, 1)
            # gaussian sampling
            if self.gaussian and self.operation.init_sol: 
                return self.sample_configuration(mode, 4)
            # house pose sampling
            if self.p_stay != 0 and np.random.uniform(0, 1) < self.p_stay: 
                return self.sample_configuration(mode, 3)
            #uniform sampling
            return self.sample_configuration(mode, 0)
        # goal sampling
        return self.sample_configuration(mode, 2, self.transition_node_ids, self.trees[mode].order)
        
    def FindLBTransitionNode(self, iter: int) -> None:
        if self.operation.init_sol: 
            modes = self.get_termination_modes()     
            result, mode = self.get_lb_transition_node_id(modes) 
            if not result:
                return
            valid_mask = result[0] < self.operation.cost
            if valid_mask.any():
                lb_transition_node = self.get_transition_node(mode, result[1])
                self.GeneratePath(mode, lb_transition_node)
                # print(f"{iter} M", mode.task_ids, "Cost: ", self.operation.cost.item())

    def UpdateDiscretizedCost(self, path , cost, idx):
        path_a, path_b = path[idx-1: -1], path[idx: ]
        batch_cost = batch_config_cost(path_a, path_b, self.env.cost_metric, self.env.cost_reduction)
        cost[idx:] = cumulative_sum(cost[idx-1], batch_cost)

    def Shortcutting(self,active_mode:Mode, version = None, choice = None, deterministic = False):
        indices  = [self.env.robot_idx[r] for r in self.env.robots]

        discretized_path, discretized_modes, discretized_costs = self.Discretization(self.operation.path_nodes, indices)  

        termination_cost = discretized_costs[-1]
        termination_iter = 0
        dim = None
        a = 0

        if not deterministic:
            range1 = 1
            range2 = 2000
        else:
            range1 = len(discretized_path)
            range2 = range1

        for i in range(range1):
            for j in range(range2):
                if not deterministic:
                    i1 = np.random.choice(len(discretized_path)-1)
                    i2 = np.random.choice(len(discretized_path)-1)
                else:
                    i1 = i
                    i2 = j
                if np.abs(i1-i2) < 2:
                    continue
                idx1 = min(i1, i2)
                idx2 = max(i1, i2)
                m1 = discretized_modes[idx1]    
                m2 = discretized_modes[idx2]    
                if m1 == m2 and self.shortcutting_robot_version == 0: #take all possible robots
                    robot = None
                    if self.shortcutting_dim_version == 2:
                        dim = [np.random.choice(indices[r_idx]) for r_idx in range(len(self.env.robots))]
                    
                    if self.shortcutting_dim_version == 3:
                        dim = []
                        for r_idx in range(len(self.env.robots)):
                            all_indices = [i for i in indices[r_idx]]
                            num_indices = np.random.choice(range(len(indices[r_idx])))
                            random.shuffle(all_indices)
                            dim.append(all_indices[:num_indices])
                else:
                    robot = np.random.choice(len(self.env.robots)) 
                    #robot just needs to pursue the same task across the modes
                    if len(m1) > 1:
                        task_agent = m1[1][robot]
                    else:
                        task_agent = m1[0][robot]

                    if m2[0][robot] != task_agent:
                        continue

                    if self.shortcutting_dim_version == 2:
                        dim = [np.random.choice(indices[robot])]
                    if self.shortcutting_dim_version == 3:
                        all_indices = [i for i in indices[robot]]
                        num_indices = np.random.choice(range(len(indices[robot])))

                        random.shuffle(all_indices)
                        dim = all_indices[:num_indices]


                edge, edge_cost =  self.EdgeInterpolation(discretized_path[idx1:idx2+1].copy(), 
                                                            discretized_costs[idx1], indices, dim, self.shortcutting_dim_version, robot, self.env.robots)
        
                if edge_cost[-1] < discretized_costs[idx2] and self.env.is_path_collision_free(edge, resolution=0.001, tolerance=0.001): #need to make path_collision_free
                    discretized_path[idx1:idx2+1] = edge
                    discretized_costs[idx1:idx2+1] = edge_cost
                    self.UpdateDiscretizedCost(discretized_path, discretized_costs, idx2)
                    self.operation.path = discretized_path
                    if not deterministic:
                        if np.abs(discretized_costs[-1] - termination_cost) > 0.001:
                            termination_cost = discretized_costs[-1]
                            termination_iter = j
                        elif np.abs(termination_iter -j) > 25000:
                            break
                    
        self.TreeExtension(active_mode, discretized_path, discretized_costs, discretized_modes)

    def TreeExtension(self, active_mode, discretized_path, discretized_costs):
        mode = discretized_path[0].mode
        parent = self.operation.path_nodes[0]
        for i in range(1, len(discretized_path) - 1):
            state = discretized_path[i]
            node = Node(state, self.operation)
            node.parent = parent
            self.operation.costs = self.trees[discretized_path[i].mode].ensure_capacity(self.operation.costs, node.id)
            node.cost = discretized_costs[i]
            node.cost_to_parent = node.cost - node.parent.cost
            parent.children.append(node)
            if self.trees[discretized_path[i].mode].order == 1:
                self.trees[discretized_path[i].mode].add_node(node)
            else:
                self.trees[discretized_path[i].mode].add_node(node, 'B')
            parent = node
            if mode != discretized_path[i].mode:
                self.convert_node_to_transition_node(mode, node.parent)
                mode = discretized_path[i].mode
            mode = discretized_path[i].mode
        #Reuse terminal node (don't want to add a new one)
        self.operation.path_nodes[-1].parent.children.remove(self.operation.path_nodes[-1])
        parent.children.append(self.operation.path_nodes[-1])
        self.operation.path_nodes[-1].parent = parent
        self.operation.path_nodes[-1].cost = discretized_costs[-1]
        self.operation.path_nodes[-1].cost_to_parent = self.operation.path_nodes[-1].cost - self.operation.path_nodes[-1].parent.cost
        
        self.GeneratePath(active_mode, self.operation.path_nodes[-1], shortcutting_bool=False)

    def EdgeInterpolation(self, path, cost, indices, dim, version, r = None, robots =None):
        q0 = path[0].q.state()
        q1 = path[-1].q.state()
        edge  = []
        # edge_cost = [cost]
        segment_vector = q1 - q0
        # dim_indices = [indices[i][dim] for i in range(len(indices))]
        N = len(path) -1
        for i in range(len(path)):
            if version == 0 :
                q = q0 +  (segment_vector * (i / N))

            elif version == 1: #shortcutting all indices of agent
                q = path[i].q.state().copy()
                for robot in range(len(robots)):
                    if r is not None and r == robot:
                        q[indices[robot]] = q0[indices[robot]] +  (segment_vector[indices[robot]] * (i /N))
                        break
                    if r is None:
                        q[indices[robot]] = q0[indices[robot]] +  (segment_vector[indices[robot]] * (i / N))

            elif version == 2: #partial shortcutting agent single dim 
                q = path[i].q.state().copy()
                for robot in range(len(robots)):
                    if r is not None and r == robot:
                        q[dim] = q0[dim] +  (segment_vector[dim] * (i / N))
                        break
                    if r is None:
                        q[dim[robot]] = q0[dim[robot]] +  (segment_vector[dim[robot]] * (i / N))
            
            elif version == 3: #partial shortcutting agent random set of dim 
                q = path[i].q.state().copy()
                for robot in range(len(robots)):
                    if r is not None and r == robot:
                        for idx in dim:
                            q[idx] = q0[idx] + ((q1[idx] - q0[idx])* (i / N))
                        break
                    if r is None:
                        for idx in dim[robot]:
                            q[idx] = q0[idx] + ((q1[idx] - q0[idx])* (i / N))

            q_list = [q[indices[i]] for i in range(len(indices))]
            edge.append(State(NpConfiguration.from_list(q_list),path[i].mode))
            if i == 0:
                continue  
            edge_a, edge_b = edge[:-1], edge[1:]
            batch_cost = batch_config_cost(edge_a, edge_b, self.env.cost_metric, self.env.cost_reduction)
            batch_cost = np.insert(batch_cost, 0, 0.0)
            edge_cost = cumulative_sum(cost, batch_cost)
        return edge, edge_cost

    def Discretization(self, path, indices, resolution=0.1):
        discretized_path, discretized_modes = [], []
        for i in range(len(path) - 1):
            start = path[i].state.q
            end = path[i+1].state.q

            # Calculate the vector difference and the length of the segment
            segment_vector = end.state() - start.state()
            # Determine the number of points needed for the current segment
            if resolution is None: 
                num_points = 2 
            else:
                N = config_dist(start, end) / resolution
                N = max(2, N)
                num_points = int(N)            # num_points = 
            
            # Create the points along the segment
            s = None
            mode = [path[i].state.mode.task_ids]
            for j in range(num_points):
                if path[i].transition and j == 0:
                    if mode[0] != path[i+1].state.mode.task_ids:
                        mode.append(path[i+1].state.mode.task_ids)
                if j == 0:
                    original_mode = path[i].state.mode
                    discretized_modes.append(mode)
                    discretized_path.append(path[i].state)
                    if i == 0:
                        continue

                else:
                    original_mode = path[i+1].state.mode
                    if j != num_points-1:
                        interpolated_point = start.state() + (segment_vector * (j / (num_points -1)))
                        q_list = [interpolated_point[indices[i]] for i in range(len(indices))]
                        discretized_path.append(State(NpConfiguration.from_list(q_list), original_mode))
                        discretized_modes.append([mode[-1]])
                        
                
        discretized_modes.append([path[-1].state.mode.task_ids])
        discretized_path.append(path[-1].state)
        path_a, path_b = discretized_path[:-1], discretized_path[1:]
        batch_cost = batch_config_cost(path_a, path_b, self.env.cost_metric, self.env.cost_reduction)
        batch_cost = np.insert(batch_cost, 0, 0.0)
        discretized_costs = cumulative_sum(0.0, batch_cost)
        return discretized_path, discretized_modes, discretized_costs

    # def PTC(self, iter:int):
    #     if iter% 1000 == 0:
    #         if check_gpu_memory_usage():
    #             return True
    #     if iter% 100000 == 0:
    #         print(iter)
    #     if time.time()-self.start_time >= self.ptc_time:
    #         print('Finished after ', np.round(time.time()-self.start_time, 1),'s' )
    #         return True

    def SaveFinalData(self) -> None:
        self.costs.append(self.operation.cost)
        self.times.append(time.time()-self.start_time)
        if self.operation.path == []:
            return
        #path and cost data
        all_init_path = False
        for idx, t in enumerate(self.times): 
            cost = self.costs[idx]
            if t == self.times[-1]:
                path_data = [state.q.state() for state in self.operation.path]

                intermediate_tot = [node.cost for node in self.operation.path_nodes]
                # intermediate_agent_dists = torch.cat([node.agent_dists for node in self.operation.path_nodes])
                result = {
                    "path": path_data,
                    "total": self.operation.path_nodes[-1].cost,
                    "agent_dists": None,
                    "intermediate_tot": intermediate_tot,
                    "is_transition": [node.transition for node in self.operation.path_nodes],
                    "modes": [node.state.mode.task_ids for node in self.operation.path_nodes],
                }
            else:
                result = {
                    "path": None,
                    "total": cost,
                    "agent_dists": None,
                    "intermediate_tot": None,
                    "is_transition": None,
                    "modes": None,
                }

            # Data Assembly
            data = {
                "result": result,
                "all_init_path": all_init_path,
                "time": t,
            }
            all_init_path = True
            # Directory Handling: Ensure directory exists
            frames_directory = os.path.join(self.output_dir, 'FramesFinalData')
            os.makedirs(frames_directory, exist_ok=True)

            # Determine Next File Number: Use generator expressions for efficiency
            next_file_number = max(
                (int(file.split('.')[0]) for file in os.listdir(frames_directory)
                if file.endswith('.pkl') and file.split('.')[0].isdigit()),
                default=-1
            ) + 1

            # Save Data as Pickle File
            filename = os.path.join(frames_directory, f"{next_file_number:04d}.pkl")
            with open(filename, 'wb') as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    def SaveData(self, mode:Mode, passed_time: time, n_new:NDArray=None, 
                 N_near:NDArray=None, r:float=None, n_rand:NDArray=None, 
                 n_nearest:NDArray = None, N_parent:NDArray=None, N_near_:List[NDArray]=None, ellipse:List[NDArray]=None) -> None:
        if self.debug_mode:
            return
        #Tree data
        tree_data = []
        for m in self.modes:
            task_ids = m.task_ids
            subtree_data = [
                {
                    "state": node.state.q.state(),
                    "parent": node.parent.state.q.state() if node.parent else None,
                    "mode": task_ids,
                }
                for node in self.trees[mode].subtree.values()
            ]
            tree_data.extend(subtree_data)
        try: 
            self.trees[mode].subtree_b
            for m in self.modes:
                task_ids = m.task_ids
                subtree_data = [
                    {
                        "state": node.state.q.state(),
                        "parent": node.parent.state.q.state() if node.parent else None,
                        "mode": task_ids,
                    }
                    for node in self.trees[mode].subtree_b.values()
                ]
                tree_data.extend(subtree_data)
        except:
            pass
        #graph
        try:
            graph_data_robot = [q.q for lst in self.g.robot_nodes.values() for q in lst]
            # graph_data_robot = [q.q for q in self.g.robot_nodes['a1_0']]
            graph_data_transition = [q.q for lst in self.g.transition_nodes.values() for q in lst]
        except:
            graph_data_robot = None
            graph_data_transition = None



        #path and cost data
        if self.operation.path_nodes is not None:
            transition_node = self.operation.path_nodes[-1]
            path_data = [state.q.state() for state in self.operation.path]

            intermediate_tot = [node.cost for node in self.operation.path_nodes]

            result = {
                "path": path_data,
                "total": transition_node.cost,
                "intermediate_tot": intermediate_tot,
                "is_transition": [node.transition for node in self.operation.path_nodes],
                "modes": [node.state.mode.task_ids for node in self.operation.path_nodes],
            }
        else:
            # If no path avaialable yet
            result = {
                "path": None,
                "total": None,
                "intermediate_tot": None,
                "is_transition": None,
                "modes": None,
            }
        
        if self.operation.paths_inter != []:
            inter_result = [{
                "path": [node.state.q.state() for node in path],
                "modes": [node.state.mode.task_ids for node in path],
            } for path in self.operation.paths_inter]
        else:
            # If no path nodes, set all result values to None
            inter_result = [{
                "path": None,
                "modes": None,
            }]

        # Informed Sampling Data
        if self.operation.init_sol and self.informed_sampling:
            try:
                
                informed_sampling = [
                    {
                        "C": self.informed[mode].C,
                        "L": self.informed[mode].L,
                        "center": self.informed[mode].center,
                        "start": self.informed[mode].start,
                        "goal": self.informed[mode].goal,
                        "mode": mode.task_ids
                    }
                ]
            except: 
                informed_sampling = None
        else:
            informed_sampling = None


        # Nearby Nodes
        if N_near is None:
            N_near_list = []
        else:
            N_near_list = [N for N in N_near] if N_near.size(0) > 0 else []

        # Nearby Nodes
        if N_near_ is None:
            N_near_list = []
        else:
            N_near_list = [N.q for N in N_near_] if len(N_near_) > 0 else []

        if N_parent is None:
            N_parent_list = []
        else:
            N_parent_list = [N for N in N_parent] if N_parent.size(0) > 0 else []

        if mode is not None:
            m = mode.task_ids
        else:
            m = None
        # Data Assembly
        data = {
            "tree": tree_data,
            "result": result,
            "inter_result":inter_result,
            "all_init_path": self.operation.init_sol,
            "time": passed_time,
            "informed_sampling": informed_sampling,
            "n_rand": n_rand,
            "N_near": N_near_list,
            "N_parent": N_parent_list,
            "rewire_r": r,
            "n_new": n_new,
            "n_nearest": n_nearest,
            "active_mode": m,
            "graph": graph_data_robot,
            "graph_transition": graph_data_transition,
            'ellipse': ellipse, 
            'mode': m
        }

        # Directory Handling: Ensure directory exists
        frames_directory = os.path.join(self.output_dir, 'FramesData')
        os.makedirs(frames_directory, exist_ok=True)

        # Determine Next File Number: Use generator expressions for efficiency
        next_file_number = max(
            (int(file.split('.')[0]) for file in os.listdir(frames_directory)
            if file.endswith('.pkl') and file.split('.')[0].isdigit()),
            default=-1
        ) + 1

        # Save Data as Pickle File
        filename = os.path.join(frames_directory, f"{next_file_number:04d}.pkl")
        with open(filename, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    @abstractmethod
    def UpdateCost(self, n: Node)-> None:
        pass
    @abstractmethod
    def PlannerInitialization(self)-> None:
        pass
    @abstractmethod
    def ManageTransition(self, n_new: Node, iter: int)-> None:
        pass
    @abstractmethod
    def Plan(self) -> dict:
        pass
