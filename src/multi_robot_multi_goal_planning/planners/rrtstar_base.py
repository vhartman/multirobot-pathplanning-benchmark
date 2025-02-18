import numpy as np
import time as time
import math as math
import random
import multi_robot_multi_goal_planning.planners as mrmgp
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, List, Dict, Callable
from numpy.typing import NDArray
from numba import njit
from scipy.stats.qmc import Halton

from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
    Mode,
    SingleGoal
)
from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
    NpConfiguration,
    config_dist,
    config_cost,
    batch_config_dist,
    batch_config_cost
)

from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
)


class Operation:
    """Represents an operation instance responsible for managing variables related to path planning and cost optimization. """
    def __init__(self):
        
        self.path = []
        self.path_modes = []
        self.path_nodes = None
        self.cost = np.inf
        self.cost_change = np.inf
        self.init_sol = False
        self.costs = np.empty(10000000, dtype=np.float64)
        self.paths_inter = []
    
    def get_cost(self, idx:int) -> float:
        """
        Returns cost of node with the specified index.

        Args: 
            idx (int): Index of node whose cost is to be retrieved.

        Returns: 
            float: Cost associated with the specified node."""
        return self.costs[idx]
class Node:
    """Represents a node in the planning structure"""
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
    def cost(self, value) -> None:
        """Set the cost in the shared operation costs array.

        Args:
            value (float): Cost value to assign to the current node.

        Returns: 
            None: This method does not return any value."""
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
    """
    Represents base structure for different tree implementations.
    """
    def __init__(self):
        pass

    def _resize_array(self, array: NDArray, current_capacity: int, new_capacity: int) -> NDArray:
        """
        Dynamically resizes a NumPy array to a new capacity.

        Args: 
        array (NDArray): Array to be resized. 
        current_capacity (int): Current capacity of array.
        new_capacity (int):Target capacity for the array.

        Returns: 
        NDArray: The resized array. 
        """
        new_array = np.empty((new_capacity, *array.shape[1:]), dtype=array.dtype)
        new_array[:current_capacity] = array  # Copy existing data
        del array  # Free old array (Python garbage collector will handle memory)
        return new_array

    def ensure_capacity(self, array: NDArray, required_capacity: int) -> NDArray:
        """ 
        Ensures that a NumPy array has sufficient capacity to accommodate new elements and resizes it if necessary.

        Args: 
        array (NDArray): The array to be checked and potentially resized. 
        required_capacity (int): The minimum required capacity for the array.

        Returns: 
        NDArray: The array with ensured capacity. """
        current_size = array.shape[0]

        if required_capacity == current_size:
            return self._resize_array(array, current_size, required_capacity * 2)  # Double the size

        return array
    
    @abstractmethod
    def add_node(self, n:Node, tree:str = '') -> None:
        """
        Adds a node to the specified subtree.

        Args:
            n (Node): Node to be added.
            tree (str, optional): Identifier of subtree where the node will be added.

        Returns:
            None: This method does not return any value.
        """

        pass
    @abstractmethod
    def remove_node(self, n:Node, tree:str = '') -> None:
        """
        Removes a node from the specified subtree.

        Args:
            n (Node): The node to be removed.
            tree (str, optional): Identifier of subtree from which the node will be removed.

        Returns:
            None: This method does not return any value.
        """

        pass
    @abstractmethod
    def get_batch_subtree(self, tree:str = '') -> NDArray:
        """
        Retrieves batch representation of specified subtree.

        Args:
            tree (str, optional): Identifier of subtree to be retrieved.

        Returns:
            NDArray: A NumPy array representing the batch data of the subtree.
        """

        pass
    @abstractmethod
    def get_node_ids_subtree(self, tree:str = '') -> NDArray:
        """
        Retrieves node IDs of specified subtree.

        Args:
            tree (str, optional): Identifier of subtree.

        Returns:
            NDArray: A NumPy array containing node IDs of the subtree.
        """

        pass
    @abstractmethod
    def add_transition_node_as_start_node(self, n:Node, tree:str = '') -> None:
        """
        Adds transition node as a start node in the specified subtree.

        Args:
            n (Node): The transition node to be added as the start node.
            tree (str, optional):  Identifier of subtree.

        Returns:
            None: This method does not return any value.
        """

        pass
    @abstractmethod
    def get_node(self, id:int, tree:str = '') -> Node:
        """
        Retrieves node from the specified subtree by its id.

        Args:
            id (int): The unique ID of the node to be retrieved.
            tree (str, optional):  Identifier of subtree.

        Returns:
            Node: The node with the desired id.
        """

        pass
    @abstractmethod
    def get_number_of_nodes_in_tree(self) -> int:
        """
        Returns total number of nodes in the tree.

        Args:
            None

        Returns:
            int: Number of nodes present in the tree.
        """

        pass
class SingleTree(BaseTree):
    """
    Represents single tree structure.
    """

    def __init__(self, env:BaseProblem):
        self.order = 1
        # self.informed = Informed()
        robot_dims = sum(env.robot_dims.values())
        self.subtree = {}
        self.initial_capacity = 100000
        self.batch_subtree = np.empty((self.initial_capacity, robot_dims), dtype=np.float64)
        self.node_ids_subtree = np.empty(self.initial_capacity, dtype=np.int64)
    def add_node(self, n:Node, tree:str = '') -> None:
        self.subtree[n.id] = n               
        position = len(self.subtree) -1
        self.batch_subtree = self.ensure_capacity(self.batch_subtree, position)
        self.batch_subtree[position,:] = n.state.q.state()
        self.node_ids_subtree = self.ensure_capacity(self.node_ids_subtree, position)
        self.node_ids_subtree[position] = n.id
    
    def remove_node(self, n:Node, tree:str = '') -> None:
        mask = self.node_ids_subtree != n.id
        self.node_ids_subtree = self.node_ids_subtree[mask] 
        self.batch_subtree = self.batch_subtree[mask]
        del self.subtree[n.id]


    def get_batch_subtree(self, tree:str = '') -> NDArray:
        return self.batch_subtree[:len(self.subtree)]
    
    def get_node_ids_subtree(self, tree:str = '') -> NDArray:
        return self.node_ids_subtree[:len(self.subtree)]

    def add_transition_node_as_start_node(self, n:Node, tree:str = '') -> None:
        if n.id not in self.subtree:
            self.add_node(n)
    
    def get_node(self, id:int, tree:str = '') -> Node:
        return self.subtree.get(id)
    def get_number_of_nodes_in_tree(self) -> int:
        return len(self.subtree)
class BidirectionalTree(BaseTree):
    """
    Represents bidirectional tree structure. 
    """

    def __init__(self, env:BaseProblem):
        self.order = 1
        # self.informed = Informed()
        robot_dims = sum(env.robot_dims.values())
        self.subtree = {}
        self.initial_capacity = 100000
        self.batch_subtree = np.empty((self.initial_capacity, robot_dims), dtype=np.float64)
        self.node_ids_subtree = np.empty(self.initial_capacity, dtype=np.int64)
        self.subtree_b = {} 
        self.batch_subtree_b = np.empty((self.initial_capacity, robot_dims), dtype=np.float64)
        self.node_ids_subtree_b = np.empty(self.initial_capacity, dtype=np.int64)
        self.connected = False
    
    def add_node(self, n:Node, tree:str = '') -> None:
        if tree == 'A' or tree == '':
            self.subtree[n.id] = n               
            position = len(self.subtree) -1
            self.batch_subtree = self.ensure_capacity(self.batch_subtree, position)
            self.batch_subtree[position,:] = n.state.q.state()
            self.node_ids_subtree = self.ensure_capacity(self.node_ids_subtree, position)
            self.node_ids_subtree[position] = n.id
        if tree == 'B':
            self.subtree_b[n.id] = n               
            position = len(self.subtree_b) -1
            self.batch_subtree_b = self.ensure_capacity(self.batch_subtree_b, position)
            self.batch_subtree_b[position,:] = n.state.q.state()
            self.node_ids_subtree_b = self.ensure_capacity(self.node_ids_subtree_b, position)
            self.node_ids_subtree_b[position] = n.id
    
    def remove_node(self, n:Node, tree:str = '') -> None:
        if tree == 'A' or tree == '':
            mask = self.node_ids_subtree != n.id
            self.node_ids_subtree = self.node_ids_subtree[mask] 
            self.batch_subtree = self.batch_subtree[mask]
            del self.subtree[n.id]

        if tree == 'B':
            mask = self.node_ids_subtree_b != n.id
            self.node_ids_subtree_b = self.node_ids_subtree_b[mask] 
            self.batch_subtree_b = self.batch_subtree_b[mask]
            del self.subtree_b[n.id]

    def get_batch_subtree(self, tree:str = '') -> NDArray:
        if tree == 'A' or tree == '':
            return self.batch_subtree[:len(self.subtree)]
        if tree == 'B':
            return self.batch_subtree_b[:len(self.subtree_b)]
    
    def get_node_ids_subtree(self, tree:str = '') -> NDArray:
        if tree == 'A' or tree == '':
            return self.node_ids_subtree[:len(self.subtree)]
        if tree == 'B':
            return self.node_ids_subtree_b[:len(self.subtree_b)]

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
        """ 
        Swaps the forward (primary) and reversed growing subtree representations if not already connected.

        Args: 
        None

        Returns: 
        None: This method does not return any value. """
        if not self.connected:
            self.subtree, self.subtree_b = self.subtree_b, self.subtree
            self.batch_subtree, self.batch_subtree_b = self.batch_subtree_b, self.batch_subtree
            self.node_ids_subtree, self.node_ids_subtree_b = self.node_ids_subtree_b, self.node_ids_subtree 
            self.order *= (-1)
    def get_number_of_nodes_in_tree(self) -> int:
        return len(self.subtree) + len(self.subtree_b)


class Informed(ABC):
    """
    Represents an informed sampling instance, providing methods for sampling within an informed set.
    """
    def __init__(self):
        pass
    
    def rotation_to_world_frame(self, start:Configuration, goal:Configuration, n:int) -> Tuple[float, NDArray]:
        """
        Computes rotation from the hyperellipsoid-aligned to the world frame based on the difference between the start and goal configs.

        Args: 
        start (Configuration): Start configuration 
        goal (Configuration): Goal configuration
        n (int): Dimensionality of the state space.

        Returns: 
        Tuple[float, NDArray]:  
            - Norm of difference between goal and start configs 
            - NDArray representing the rotation matrix from the hyperellipsoid-aligned frame to the world frame."""
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
    
    def initialize(self) -> None:
        """ 
        Initializes dimensions of each robot

        Args: 
        None

        Returns: 
        None: This method does not return any value. """
        for robot in self.env.robots:
            r_idx = self.env.robots.index(robot)
            self.n[r_idx] = self.env.robot_dims[robot]
            
    def cholesky_decomposition(self, r_indices:List[int], r_idx:int, path_nodes:List[Node]) -> Optional[NDArray]:
        """ 
        Computes cholesky decomposition of hyperellipsoid matrix

        Args: 
        r_indices (List[int]): List of indices of robot with idx r_idx. 
        r_idx (int): Idx of robot 
        path_nodes (List[Node]): Path consisting of nodes used to compute right-hand side of informed set description.

        Returns: 
        NDArray: Cholesky decomposition (=diagonal matrix) if conditions are met, otherwise it returns None. """
        cmax = self.get_right_side_of_eq(r_indices, r_idx, path_nodes)
        if not cmax or self.cmin[r_idx] < 0:
            return
        r1 = cmax / 2 
        r2 = np.sqrt(cmax**2 - self.cmin[r_idx]**2) / 2
        return np.diag(np.concatenate([[r1], np.repeat(r2, self.n[r_idx] - 1)])) 
    
    @abstractmethod
    def get_right_side_of_eq(self, r_indices:List[int], r_idx:int, path_nodes:List[Node]) -> float:
        """
        Computes right-hand side of informed set description

        Args: 
        r_indices (List[int]): List of indices of robot with idx r_idx. 
        r_idx (int): Idx of robot 
        path_nodes (List[Node]): Path consisting of nodes used to compute right-hand side of informed set description.

        Returns:
            float: Value of right-hand side (= cost)
        """
        pass  
class InformedVersion0(Informed):
    """
    Locally informed sampling for each robot separately (start and goal config are equal to home pose and task).
    Upper bound cost is determined by cost of one robot.
    
    """

    def __init__(self, 
                 env:BaseProblem, 
                 cost_fct: Callable):
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
    
    def get_right_side_of_eq(self, r_indices:List[int], r_idx:int, path_nodes:List[Node]) -> float:
        #need to calculate the cost only cnsidering one agent (euclidean distance)
        n1 = NpConfiguration.from_numpy(path_nodes[0].state.q.state()[r_indices])
        cost = 0
        if self.mode_task_ids_home_poses[r_idx] == [-1]:
            start_cost = 0
            self.start[r_idx] = n1
        for node in path_nodes[1:]:
            n2 = NpConfiguration.from_numpy(node.state.q.state()[r_indices])
            cost += self.cost_fct(n1, n2, self.env.cost_metric, self.env.cost_reduction)
            #need to work with task IDs as same mode can have different configs
            if node.transition and node.state.mode.task_ids == self.mode_task_ids_home_poses[r_idx]:
                start_cost = cost
                self.start[r_idx] = n2
            if node.transition and node.state.mode.task_ids == self.mode_task_ids_task[r_idx]:
                goal_cost = cost
                self.goal[r_idx] = n2
                break 
            n1 = n2
        norm, self.C[r_idx] = self.rotation_to_world_frame(self.start[r_idx], self.goal[r_idx], self.n[r_idx])

        if norm is None:
            return
        self.cmin[r_idx]  = norm-2*self.env.collision_tolerance
        self.center[r_idx] = (self.goal[r_idx].state() + self.start[r_idx].state())/2
        return goal_cost-start_cost    
class InformedVersion1(Informed):
    """
    Globally informed sampling considering whole state space.
    Upper bound cost is determined by subtracting a lower bound from the path cost.

    """

    def __init__(self, 
                 env:BaseProblem, 
                 cost_fct: Callable):
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
    
    def get_right_side_of_eq(self, path_nodes:List[Node]) -> float:
        """
        Computes right-hand side of informed set description

        Args: 
        path_nodes (List[Node]): Path consisting of nodes used to compute right-hand side of informed set description.

        Returns:
            float: Value of right-hand side (= cost)
        """
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
    
    def initialize(self, mode:Mode, next_ids:List[int])-> None: 
        """ 
        Initializes dimensions of each robot

        Args: 
        mode (Mode): Current active mode used to determine active tasks. 
        next_ids (List[int]): List containing next task IDs 

        Returns: 
        None: This method does not return any value. """
        active_robots = self.env.get_active_task(mode, next_ids).robots 
        for robot in self.env.robots:
            if robot in active_robots:
                self.active_robots_idx.append(self.env.robots.index(robot))
        self.n = sum(self.env.robot_dims.values())

    def cholesky_decomposition(self, path_nodes:List[Node]) -> NDArray:
        """ 
        Computes cholesky decomposition of hyperellipsoid matrix

        Args: 
        path_nodes (List[Node]): Path consisting of nodes used to compute right-hand side of informed set description.

        Returns: 
        NDArray: Cholesky decomposition (=diagonal matrix) if conditions are met, otherwise it returns None. """
        cmax = self.get_right_side_of_eq( path_nodes)
        r1 = cmax / 2
        r2 = np.sqrt(cmax**2 - self.cmin**2) / 2
        return np.diag(np.concatenate([np.repeat(r1, 1), np.repeat(r2, self.n - 1)]))   
class InformedVersion2(Informed):
    """
    Globally informed sampling for each agent separately (start and goal config are equal to home pose and task).
    Upper bound cost is determined by subtracting a lower bound from the path cost.
    """

    def __init__(self, 
                 env:BaseProblem, 
                 cost_fct: Callable):
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
            #need to work with task IDs as same mode can have different configs
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
    """
    Locally informed sampling for each agent seperately (randomly select start and goal config between home pose and task).
    Upper bound cost is determined by subtracting a lower bound from the path cost between start and goal config.
    """

    def __init__(self, 
                 env:BaseProblem, 
                 cost_fct: Callable):
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
            #need to work with task IDs as same mode can have different configs
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
    """
    Globally informed sampling for each agent separately (randomly select start and end config on whole path).
    Upper bound cost is determined by subtracting a lower bound from the path cost.
    """

    def __init__(self, 
                 env:BaseProblem, 
                 cost_fct: Callable):
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
    """
    Globally informed sampling for each agent separately (start and goal config are equal to home pose and task).
    Upper bound cost is determined by the path cost between start and goal node.
    """

    def __init__(self, 
                 env:BaseProblem, 
                 cost_fct: Callable):
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
    
    def get_right_side_of_eq(self, r_indices:List[int], r_idx:int, path_nodes:List[Node]) -> float:
        if self.mode_task_ids_home_poses[r_idx] == [-1]:
            start_node = path_nodes[0]
            self.start[r_idx] = NpConfiguration.from_numpy(start_node.state.q.state()[r_indices])
        for node in path_nodes[1:]:
            #need to work with task IDs as same mode can have different configs
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
class InformedVersion6():
    """Locally and globally informed sampling"""
    def __init__(self, 
                 env: BaseProblem, 
                 modes: List[Mode], 
                 locally_informed_sampling: bool):
        self.env = env
        self.modes = modes
        self.conf_type = type(self.env.get_start_pos())
        self.locally_informed_sampling = locally_informed_sampling
        
    def sample_unit_ball(self, dim:int) -> NDArray:
        """ 
        Samples a point uniformly from a n-dimensional unit ball centered at the origin.

        Args: 
            n (int): Dimension of the ball.

        Returns: 
            NDArray: Sampled point from the unit ball. """
        # u = np.random.uniform(-1, 1, dim)
        # norm = np.linalg.norm(u)
        # r = np.random.random() ** (1.0 / dim)
        # return r * u / norm
        u = np.random.normal(0, 1, dim)
        norm = np.linalg.norm(u)
        # Generate radius with correct distribution
        r = np.random.random() ** (1.0 / dim)
        return (r / norm) * u

    def compute_PHS_matrices(self, a:NDArray, b:NDArray, c:float) -> Tuple[NDArray, NDArray]:
        """
        Computes transformation matrix and center for a Prolate Hyperspheroid (PHS) defined by endpoints a and b and cost parameter c.

        Args:
            a (NDArray): Start point of the PHS.
            b (NDArray): End point of the PHS.
            c (float): Cost parameter defining scaling of the PHS.

        Returns:
            Tuple:    
                - NDArray: Transformation matrix (rotation and scaling) of the PHS.
                - NDArray: Center of the PHS, calculated as the midpoint between a and b.
               
        """

        dim = len(a)
        diff = b - a

        # Calculate the center of the PHS.
        center = (a + b) / 2
        # The transverse axis in the world frame.
        c_min = np.linalg.norm(diff)

        # The first column of the identity matrix.
        # one_1 = np.eye(a1.shape[0])[:, 0]
        a1 = diff / c_min
        e1 = np.zeros(dim)
        e1[0] = 1.0

        # Optimized rotation matrix calculation
        U, S, Vt = np.linalg.svd(np.outer(a1, e1))
        # Sigma = np.diag(S)
        # lam = np.eye(Sigma.shape[0])
        lam = np.eye(dim)
        lam[-1, -1] = np.linalg.det(U) * np.linalg.det(Vt.T)
        # Calculate the rotation matrix.
        # cwe = np.matmul(U, np.matmul(lam, Vt))
        cwe = U @ lam @ Vt
        # Get the radius of the first axis of the PHS.
        # r1 = c / 2
        # Get the radius of the other axes of the PHS.
        # rn = [np.sqrt(c**2 - c_min**2) / 2] * (dim - 1)
        # Create a vector of the radii of the PHS.
        # r = np.diag(np.array([r1] + rn))
        # sqrt_term = c**2 - c_min**2

        # if sqrt_term < 0 or np.isnan(sqrt_term):
        #     print("hallo")
        r = np.diag([c * 0.5] + [np.sqrt(c**2 - c_min**2) * 0.5] * (dim - 1))


        return cwe @ r, center

    def sample_phs_with_given_matrices(self, rot:NDArray, center:NDArray) -> NDArray:
        """
        Samples point from a prolate hyperspheroid (PHS) defined by the given rotation matrix and center.

        Args:
            rot (NDArray): Transformation matrix (rotation and scaling) for the PHS.
            center (NDArray): Center point of the PHS.

        Returns:
            NDArray: Sampled point from the PHS.
        """

        dim = len(center)
        x_ball = self.sample_unit_ball(dim)
        return rot @ x_ball + center

    def sample_mode(self,
        mode_sampling_type: str = "uniform_reached"
    ) -> Mode:
        """
        Selects a mode based on the specified sampling strategy.

        Args:
            mode_sampling_type (str): Mode sampling strategy to use.

        Returns:
            Mode: Sampled mode according to the specified strategy.
        """

        if mode_sampling_type == "uniform_reached":
            m_rnd = random.choice(self.modes)
        return m_rnd
    
    def can_improve(self, 
        rnd_state: State, path: List[State], start_index:int, end_index:int, path_segment_costs:NDArray
    ) -> bool:
        """
        Determines if a segment of the path can be improved by comparing its cost to a lower-bound estimate.

        Args:
            rnd_state (State): Reference state used for computing lower-bound costs.
            path (List[State]): Sequence of states representing current path.
            start_index (int): Index marking the start of the segment to evaluate.
            end_index (int): Index marking the end of the segment to evaluate.
            path_segment_costs (NDArray):Costs associated with each segment between consecutive states in the path.

        Returns:
            bool: True if the segment's actual cost exceeds lower-bound estimate (indicating potential improvement); otherwise, False.
        """

        # path_segment_costs = env.batch_config_cost(path[:-1], path[1:])

        # compute the local cost
        path_cost_from_start_to_index = np.sum(path_segment_costs[:start_index])
        path_cost_from_goal_to_index = np.sum(path_segment_costs[end_index:])
        path_cost = np.sum(path_segment_costs)

        if start_index == 0:
            assert path_cost_from_start_to_index == 0
        if end_index == len(path) - 1:
            assert path_cost_from_goal_to_index == 0

        path_cost_from_index_to_index = (
            path_cost - path_cost_from_goal_to_index - path_cost_from_start_to_index
        )

        # print(path_cost_from_index_to_index)

        lb_cost_from_start_index_to_state = self.env.config_cost(
            rnd_state.q, path[start_index].q
        )
        # if path[start_index].mode != rnd_state.mode:
        #     start_state = path[start_index]
        #     lb_cost_from_start_to_state = lb_cost_from_start(rnd_state)
        #     lb_cost_from_start_to_index = lb_cost_from_start(start_state)

        #     lb_cost_from_start_index_to_state = max(
        #         (lb_cost_from_start_to_state - lb_cost_from_start_to_index),
        #         lb_cost_from_start_index_to_state,
        #     )

        lb_cost_from_state_to_end_index = self.env.config_cost(
            rnd_state.q, path[end_index].q
        )
        # if path[end_index].mode != rnd_state.mode:
        #     goal_state = path[end_index]
        #     lb_cost_from_goal_to_state = lb_cost_from_goal(rnd_state)
        #     lb_cost_from_goal_to_index = lb_cost_from_goal(goal_state)

        #     lb_cost_from_state_to_end_index = max(
        #         (lb_cost_from_goal_to_state - lb_cost_from_goal_to_index),
        #         lb_cost_from_state_to_end_index,
        #     )

        # print("can_imrpove")

        # print("start", lb_cost_from_start_index_to_state)
        # print("end", lb_cost_from_state_to_end_index)

        # print('start index', start_index)
        # print('end_index', end_index)

        # assert(lb_cost_from_start_index_to_state >= 0)
        # assert(lb_cost_from_state_to_end_index >= 0)

        # print("segment cost", path_cost_from_index_to_index)
        # print(
        #     "lb cost",
        #     lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index,
        # )

        if (
            path_cost_from_index_to_index
            > lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index
        ):
            return True

        return False

    def generate_informed_samples(self,
        batch_size:int,
        path:List[State],
        mode:Mode,
        max_attempts_per_sample:int =200,
        locally_informed_sampling:bool =True,
        try_direct_sampling:bool =True,
    ) -> Configuration:
        """ 
        Samples configuration from informed set for the given mode.

        Args: 
            batch_size (int): Number of samples to generate in a batch.
            path (List[State]): Current path used to guide the informed sampling.
            mode (Mode): Current operational mode for which the configuration is to be sampled.
            max_attempts_per_sample (int, optional): Maximum number of attempts per sample.
            locally_informed_sampling (bool, optional): If True, applies locally informed sampling; otherwise globally.
            try_direct_sampling (bool, optional): If True, attempts direct sampling from the informed set.

        Returns: 
            Configuration: Configuration within the informed set that satisfies the specified limits for the robots. 
        """
        path = mrmgp.joint_prm_planner.interpolate_path(path)
        new_samples = []
        path_segment_costs = self.env.batch_config_cost(path[:-1], path[1:])

        num_attempts = 0
        while True:
            if num_attempts > batch_size:
                break

            num_attempts += 1
            # print(len(new_samples))
            # sample mode
            if locally_informed_sampling:
                for _ in range(500):
                    start_ind = random.randint(0, len(path) - 1)
                    end_ind = random.randint(0, len(path) - 1)

                    if end_ind - start_ind > 2:
                        # if end_ind - start_ind > 2 and end_ind - start_ind < 50:
                        current_cost = sum(path_segment_costs[start_ind:end_ind])
                        lb_cost = self.env.config_cost(path[start_ind].q, path[end_ind].q)

                        if lb_cost < current_cost:
                            break

                # TODO: we need to sample from the set of all reachable modes here
                # not only from the modes on the path
                k = random.randint(start_ind, end_ind)
                m = path[k].mode
            else:
                start_ind = 0
                end_ind = len(path) - 1
                m = self.sample_mode("uniform_reached")

            # print(m)

            current_cost = sum(path_segment_costs[start_ind:end_ind])

            # tmp = 0
            # for i in range(start_ind, end_ind):
            #     tmp += env.config_cost(path[i].q, path[i+1].q)

            # print(current_cost, tmp)

            # plt.figure()
            # samples = []
            # for _ in range(500):
            #     sample = samplePHS(np.array([-1, 1, 0]), np.array([1, -1, 0]), 3)
            #     # sample = samplePHS(np.array([[-1], [0]]), np.array([[1], [0]]), 3)
            #     samples.append(sample[:2])
            #     print("sample", sample)

            # plt.scatter([a[0] for a in samples], [a[1] for a in samples])
            # plt.show()

            focal_points = np.array(
                [path[start_ind].q.state(), path[end_ind].q.state()], dtype=np.float64
            )

            precomputed_phs_matrices = {}
            precomputed_robot_cost_bounds = {}

            obv_inv_attempts = 0
            sample_in_collision = 0

            for k in range(max_attempts_per_sample):
                had_to_be_clipped = False
                if not try_direct_sampling or self.env.cost_metric != "euclidean":
                    # completely random sample configuration from the (valid) domain robot by robot
                    q = []
                    for i in range(len(self.env.robots)):
                        r = self.env.robots[i]
                        lims = self.env.limits[:, self.env.robot_idx[r]]
                        if lims[0, 0] < lims[1, 0]:
                            qr = (
                                np.random.rand(self.env.robot_dims[r])
                                * (lims[1, :] - lims[0, :])
                                + lims[0, :]
                            )
                        else:
                            qr = np.random.rand(self.env.robot_dims[r]) * 6 - 3

                        q.append(qr)
                else:
                    # sample by sampling each agent separately
                    q = []
                    for i in range(len(self.env.robots)):
                        r = self.env.robots[i]
                        lims = self.env.limits[:, self.env.robot_idx[r]]
                        if lims[0, 0] < lims[1, 0]:
                            if i not in precomputed_robot_cost_bounds:
                                if self.env.cost_reduction == "sum":
                                    precomputed_robot_cost_bounds[i] = (
                                        current_cost
                                        - sum(
                                            [
                                                np.linalg.norm(
                                                    path[start_ind].q[j]
                                                    - path[end_ind].q[j]
                                                )
                                                for j in range(len(self.env.robots))
                                                if j != i
                                            ]
                                        )
                                    )
                                else:
                                    precomputed_robot_cost_bounds[i] = current_cost

                            norm = np.linalg.norm(
                                    path[start_ind].q[i] - path[end_ind].q[i]
                                )
                            if (norm < 1e-3 or norm > precomputed_robot_cost_bounds[i]
                            ):
                                qr = (
                                    np.random.rand(self.env.robot_dims[r])
                                    * (lims[1, :] - lims[0, :])
                                    + lims[0, :]
                                )
                            else:
                                # print("cost", current_cost)
                                # print("robot cst", c_robot_bound)
                                # print(
                                #     np.linalg.norm(
                                #         path[start_ind].q[i] - path[end_ind].q[i]
                                #     )
                                # )

                                if i not in precomputed_phs_matrices:
                                    precomputed_phs_matrices[i] =self.compute_PHS_matrices(
                                        path[start_ind].q[i],
                                        path[end_ind].q[i],
                                        precomputed_robot_cost_bounds[i],
                                    )

                                qr = self.sample_phs_with_given_matrices(
                                    *precomputed_phs_matrices[i]
                                )
                                # plt.figure()
                                # samples = []
                                # for _ in range(500):
                                #     sample = self.sample_phs_with_given_matrices(
                                #         *precomputed_phs_matrices[i]
                                #     )
                                #     # sample = samplePHS(np.array([[-1], [0]]), np.array([[1], [0]]), 3)
                                #     samples.append(sample[:2])
                                #     print("sample", sample)

                                # plt.scatter(
                                #     [a[0] for a in samples], [a[1] for a in samples]
                                # )
                                # plt.show()

                                # qr = samplePHS(path[start_ind].q[i], path[end_ind].q[i], c_robot_bound)
                                # qr = rejection_sample_from_ellipsoid(
                                #     path[start_ind].q[i], path[end_ind].q[i], c_robot_bound
                                # )

                                # if np.linalg.norm(path[start_ind].q[i] - qr) + np.linalg.norm(path[end_ind].q[i] - qr) > c_robot_bound:
                                #     print("AAAAAAAAA")
                                #     print(np.linalg.norm(path[start_ind].q[i] - qr) + np.linalg.norm(path[end_ind].q[i] - qr), c_robot_bound)

                                clipped = np.clip(qr, lims[0, :], lims[1, :])
                                if not np.array_equal(clipped, qr):
                                    had_to_be_clipped = True
                                    break
                                    # print("AAA")

                        q.append(qr)

                if had_to_be_clipped:
                    continue

                q = self.conf_type.from_list(q)

                if sum(self.env.batch_config_cost(q, focal_points)) > current_cost:
                    # print(path[start_ind].mode, path[end_ind].mode, m)
                    # print(
                    #     current_cost,
                    #     env.config_cost(path[start_ind].q, q)
                    #     + env.config_cost(path[end_ind].q, q),
                    # )
                    # if can_improve(State(q, m), path, start_ind, end_ind):
                    #     assert False

                    obv_inv_attempts += 1

                    continue

                # if can_improve(State(q, m), path, 0, len(path)-1):
                # if can_improve(State(q, m), path, start_ind, end_ind):
                if not self.env.is_collision_free(q, m):
                    sample_in_collision += 1
                    continue

                if self.can_improve(
                    State(q, m), path, start_ind, end_ind, path_segment_costs
                ):
                    # if env.is_collision_free(q, m) and can_improve(State(q, m), path, 0, len(path)-1):
                    if m == mode:
                        return q
                    new_samples.append(State(q, m))
                    break

            # print("inv attempt", obv_inv_attempts)
            # print("coll", sample_in_collision)

        # print(len(new_samples) / num_attempts)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter([a.q[0][0] for a in new_samples], [a.q[0][1] for a in new_samples], [a.q[0][2] for a in new_samples])
        # ax.scatter([a.q[1][0] for a in new_samples], [a.q[1][1] for a in new_samples], [a.q[1][2] for a in new_samples])
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.scatter([a.q[0][0] for a in new_samples], [a.q[0][1] for a in new_samples])
        # ax.scatter([a.q[1][0] for a in new_samples], [a.q[1][1] for a in new_samples])
        # plt.show()

        return 

    def can_transition_improve(self, transition: Tuple[Configuration, Mode, Mode], path:List[State], start_index:int, end_index:int) -> bool:
        """
        Determines if current path segment can be improved by comparing its actual cost to a lower-bound estimate.

        Args:
            transition (Tuple[Configuration, Mode, Mode]): Tuple containing a configuration and two mode identifiers, used to construct reference states for lower-bound cost calculations.
            path (List[State]): A sequence of states representing current path.
            start_index (int): Index marking the beginning of the segment to evaluate.
            end_index (int): Index marking the end of the segment to evaluate.

        Returns:
            bool: True if the segment's actual cost exceeds lower-bound estimate (indicating potential improvement); otherwise, False.
        """

        path_segment_costs = self.env.batch_config_cost(path[:-1], path[1:])

        # compute the local cost
        path_cost_from_start_to_index = np.sum(path_segment_costs[:start_index])
        path_cost_from_goal_to_index = np.sum(path_segment_costs[end_index:])
        path_cost = np.sum(path_segment_costs)

        if start_index == 0:
            assert path_cost_from_start_to_index == 0
        if end_index == len(path) - 1:
            assert path_cost_from_goal_to_index == 0

        path_cost_from_index_to_index = (
            path_cost - path_cost_from_goal_to_index - path_cost_from_start_to_index
        )

        # print(path_cost_from_index_to_index)

        rnd_state_mode_1 = State(transition[0], transition[1])
        rnd_state_mode_2 = State(transition[0], transition[2])

        lb_cost_from_start_index_to_state = self.env.config_cost(
            rnd_state_mode_1.q, path[start_index].q
        )
        # if path[start_index].mode != rnd_state_mode_1.mode:
        #     start_state = path[start_index]
        #     lb_cost_from_start_to_state = lb_cost_from_start(rnd_state_mode_1)
        #     lb_cost_from_start_to_index = lb_cost_from_start(start_state)

        #     lb_cost_from_start_index_to_state = max(
        #         (lb_cost_from_start_to_state - lb_cost_from_start_to_index),
        #         lb_cost_from_start_index_to_state,
        #     )

        lb_cost_from_state_to_end_index = self.env.config_cost(
            rnd_state_mode_2.q, path[end_index].q
        )
        # if path[end_index].mode != rnd_state_mode_2.mode:
        #     goal_state = path[end_index]
        #     lb_cost_from_goal_to_state = lb_cost_from_goal(rnd_state_mode_2)
        #     lb_cost_from_goal_to_index = lb_cost_from_goal(goal_state)

        #     lb_cost_from_state_to_end_index = max(
        #         (lb_cost_from_goal_to_state - lb_cost_from_goal_to_index),
        #         lb_cost_from_state_to_end_index,
        #     )

        # print("can_imrpove")

        # print("start", lb_cost_from_start_index_to_state)
        # print("end", lb_cost_from_state_to_end_index)

        # print('start index', start_index)
        # print('end_index', end_index)

        # assert(lb_cost_from_start_index_to_state >= 0)
        # assert(lb_cost_from_state_to_end_index >= 0)

        # print("segment cost", path_cost_from_index_to_index)
        # print(
        #     "lb cost",
        #     lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index,
        # )

        if (
            path_cost_from_index_to_index
            > lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index
        ):
            return True

        return False

    def generate_informed_transitions(self,
        batch_size:int, path:List[State], active_mode:Mode, locally_informed_sampling:bool =True, max_attempts_per_sample:int =100
    ) -> Configuration:
        """ 
        Samples transition configuration from informed set for the given mode.

        Args: 
            batch_size (int): Number of samples to generate in a batch.
            path (List[State]): Current path used to guide the informed sampling.
            active_mode (Mode): Current operational mode for which the configuration is to be sampled.
            locally_informed_sampling (bool, optional): If True, applies locally informed sampling; otherwise globally.
            max_attempts_per_sample (int, optional): Maximum number of attempts per sample.

        Returns: 
            Configuration: Transiiton configuration within the informed set that satisfies the specified limits for the robots. 
        """
        
        path =  mrmgp.joint_prm_planner.interpolate_path(path)
        if len(self.env.tasks) == 1:
            return []

        new_transitions = []
        num_attempts = 0
        path_segment_costs = self.env.batch_config_cost(path[:-1], path[1:])

        while True:
            num_attempts += 1

            if num_attempts > batch_size:
                break

            # print(len(new_samples))
            # sample mode
            if locally_informed_sampling:
                while True:
                    start_ind = random.randint(0, len(path) - 1)
                    end_ind = random.randint(0, len(path) - 1)

                    if (
                        path[end_ind].mode != path[start_ind].mode
                        and end_ind - start_ind > 2
                    ):
                        # if end_ind - start_ind > 2 and end_ind - start_ind < 50:
                        current_cost = sum(path_segment_costs[start_ind:end_ind])
                        lb_cost = self.env.config_cost(path[start_ind].q, path[end_ind].q)

                        if lb_cost < current_cost:
                            break

                k = random.randint(start_ind, end_ind)
                mode = path[k].mode
            else:
                start_ind = 0
                end_ind = len(path) - 1
                mode = self.sample_mode("uniform_reached")

            # print(m)

            current_cost = sum(path_segment_costs[start_ind:end_ind])

            # sample transition at the end of this mode
            possible_next_task_combinations = self.env.get_valid_next_task_combinations(mode)
            if len(possible_next_task_combinations) > 0:
                ind = random.randint(0, len(possible_next_task_combinations) - 1)
                active_task = self.env.get_active_task(
                    mode, possible_next_task_combinations[ind]
                )
            else:
                continue

            goals_to_sample = active_task.robots

            goal_sample = active_task.goal.sample(mode)

            for k in range(max_attempts_per_sample):
                # completely random sample configuration from the (valid) domain robot by robot
                q = []
                for i in range(len(self.env.robots)):
                    r = self.env.robots[i]
                    if r in goals_to_sample:
                        offset = 0
                        for _, task_robot in enumerate(active_task.robots):
                            if task_robot == r:
                                q.append(
                                    goal_sample[
                                        offset : offset + self.env.robot_dims[task_robot]
                                    ]
                                )
                                break
                            offset += self.env.robot_dims[task_robot]
                    else:  # uniform sample
                        lims = self.env.limits[:, self.env.robot_idx[r]]
                        if lims[0, 0] < lims[1, 0]:
                            qr = (
                                np.random.rand(self.env.robot_dims[r])
                                * (lims[1, :] - lims[0, :])
                                + lims[0, :]
                            )
                        else:
                            qr = np.random.rand(self.env.robot_dims[r]) * 6 - 3

                        q.append(qr)

                q = self.conf_type.from_list(q)

                if (
                    self.env.config_cost(path[start_ind].q, q)
                    + self.env.config_cost(path[end_ind].q, q)
                    > current_cost
                ):
                    continue

                if self.env.is_terminal_mode(mode):
                    assert False
                else:
                    next_mode = self.env.get_next_mode(q, mode)

                if self.can_transition_improve(
                    (q, mode, next_mode), path, start_ind, end_ind
                ) and self.env.is_collision_free(q, mode):
                    if mode == active_mode:
                        return q
                    new_transitions.append((q, mode, next_mode))
                    break

        print(len(new_transitions) / num_attempts)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter([a.q[0][0] for a in new_samples], [a.q[0][1] for a in new_samples], [a.q[0][2] for a in new_samples])
        # ax.scatter([a.q[1][0] for a in new_samples], [a.q[1][1] for a in new_samples], [a.q[1][2] for a in new_samples])
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.scatter([a[0][0][0] for a in new_transitions], [a[0][0][1] for a in new_transitions])
        # ax.scatter([a[0][1][0] for a in new_transitions], [a[0][1][1] for a in new_transitions])
        # plt.show()

        return 

    






@njit(fastmath=True, cache=True)
def find_nearest_indices(set_dists:NDArray, r:float) -> NDArray:
    """
    Finds the indices of elements in the distance array that are less than or equal to the specified threshold r.

    Args:
        set_dists (NDArray): Array of distance values.
        r (float): Threshold value for comparison (.

    Returns:
        NDArray: Array of indices where the distance values are less than or equal to the threshold.
    """

    r += 1e-10 # a small epsilon is added to mitigate floating point issues
    return np.nonzero(set_dists <= r)[0]
@njit
def cumulative_sum(batch_cost:NDArray) -> NDArray:
    """
    Computes the cumulative sum of the input cost array, where each element is the sum of all previous elements including itself.

    Args:
        batch_cost (NDArray): Array of cost values.

    Returns:
        NDArray: Array containing the cumulative sums.
    """
    cost = np.empty(len(batch_cost), dtype=np.float64) 
    for idx in range(0, len(batch_cost)):
        cost[idx] = np.sum(batch_cost[:idx+1])
    return cost
@njit
def get_mode_task_ids_of_active_task_in_path(path_modes, task_id:List[int], r_idx:int) -> List[int]:
    """
    Retrieves mode task ID for the active task in a given path by selecting the last occurrence that matches the specified task id.

    Args:
        path_modes (NDArray): Sequence of mode task IDs along the path.
        task_id (List[int]): Task IDs to search for in the path.
        r_idx (int): Corresponding idx of robot.

    Returns:
        NDArray: Mode task ID associated with the last occurrence of the specified active task for the desired robot.
    """

    last_index = 0 
    for i in range(len(path_modes)):
        if path_modes[i][r_idx] == task_id:  
            last_index = i 
    return path_modes[last_index]
@njit
def get_mode_task_ids_of_home_pose_in_path(path_modes, task_id:List[int], r_idx:int) -> List[int]:
    """
    Retrieves mode task ID for the active task in a given path by selecting the first occurrence that matches the specified task id.

    Args:
        path_modes (NDArray): Sequence of mode task IDs along the path.
        task_id (List[int]): Task IDs to search for in the path.
        r_idx (int): Corresponding idx of robot.

    Returns:
        NDArray: Mode task ID associated with the first occurrence of the specified active task for the desired robot ( = home pose). 
    """

    for i in range(len(path_modes)):
        if path_modes[i][r_idx] == task_id:  
            if i == 0:
                return np.array([-1]) # if its at the start pose
            return path_modes[i-1]


class BaseRRTstar(ABC):
    """
    Represents the base class for RRT*-based algorithms, providing core functionalities for motion planning.
    """
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
                 shortcutting_robot_version = 1, 
                 locally_informed_sampling = True

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
        self.locally_informed_sampling = locally_informed_sampling
        self.eta = np.sqrt(sum(self.env.robot_dims.values()))
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

    def add_tree(self, 
                 mode: Mode, 
                 tree_instance: Optional[Union["SingleTree", "BidirectionalTree"]] = None
                 ) -> None:
        """
        Initializes new tree instance for specified mode.

        Args:
            mode (Mode): Mode for which tto add the tree.
            tree_instance (Optional[Union["SingleTree", "BidirectionalTree"]]): Type of tree instance to initialize. Must be either SingleTree or BidirectionalTree.

        Returns:
            None: This method does not return any value.
        """

        if tree_instance is None:
            raise ValueError("You must provide a tree instance type: SingleTree or BidirectionalTree.")
        
        # Check type and initialize the tree
        if tree_instance == SingleTree:
            self.trees[mode] = SingleTree(self.env)
        elif tree_instance == BidirectionalTree:
            self.trees[mode] = BidirectionalTree(self.env)
        else:
            raise TypeError("tree_instance must be SingleTree or BidirectionalTree.")
           
    def add_new_mode(self, 
                     q:Configuration=None, 
                     mode:Mode=None, 
                     tree_instance: Optional[Union["SingleTree", "BidirectionalTree"]] = None
                     ) -> None:
        """
        Initializes a new mode (including its corresponding tree instance and performs informed initialization).

        Args:
            q (Configuration): Configuration used to determine the new mode. 
            mode (Mode): The current mode from which to get the next mode. 
            tree_instance (Optional[Union["SingleTree", "BidirectionalTree"]]): Type of tree instance to initialize for the next mode. Must be either SingleTree or BidirectionalTree.

        Returns:
            None: This method does not return any value.
        """

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
        """
        Marks node as a potential transition node for the specified mode.

        Args:
            mode (Mode): Mode in which the node is marked as a transition node.
            n (Node): Node to be marked as a transition node.

        Returns:
            None: This method does not return any value.
        """

        n.transition = True
        if mode not in self.transition_node_ids:
            self.transition_node_ids[mode] = []
        if n.id not in self.transition_node_ids[mode]:
            self.transition_node_ids[mode].append(n.id)

    def convert_node_to_transition_node(self, mode:Mode, n:Node) -> None:
        """
        Marks a node as potential transition node in the specified mode and adds it as a start node for each subsequent mode. 
        
        Args: 
            mode (Mode): Mode in which the node is converted to a transition node.
            n (Node): Node to convert into a transition node. 
        
        Returns: 
            None: This method does not return any value. 
        """

        self.mark_node_as_transition(mode,n)
        for next_mode in mode.next_modes:
            self.trees[next_mode].add_transition_node_as_start_node(n)

    def get_lb_transition_node_id(self, modes:List[Mode]) -> Tuple[Tuple[float, int], Mode]:
        """
        Retrieves the lower bound cost and corresponding transition node ID from a list of modes. 
        
        Args:
            modes (List[Mode]): List of modes to search for the transition node with the minimum cost.
        
        Returns:
            Tuple: 
                - float: A tuple with the lower bound cost.
                - int: Corresponding transition node id.
                - Mode: Mode from which the transition node was selected.
        """

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
           
    def get_transition_node(self, mode:Mode, id:int) -> Node:
        """
        Retrieves transition node from the primary subtree of the given mode by its id.

        Args:
            mode (Mode): Mode from which the transition node is to be retrieved.
            id (int): The unique ID of the transition node.

        Returns:
            Node: Transition node from the primary subtree (i.e. 'subtree' if order is 1; otherwise 'subtree_b' as trees are swapped).
        """

        if self.trees[mode].order == 1:
            return self.trees[mode].subtree.get(id)
        else:
            return self.trees[mode].subtree_b.get(id)

    def get_lebesgue_measure_of_free_configuration_space(self, num_samples:int=10000) -> None:
        """
        Sets the free configuration space parameter by estimating its Lebesgue measure using Halton sequence sampling.

        Args:
            num_samples (int): Number of samples to generate the estimation.

        Returns:
            None: This method does not return any value. 
        """

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
        # Estimate C_free measure
        self.c_free = (free_samples / num_samples) * total_volume
        
    def set_gamma_rrtstar(self) -> None:
        """
        Sets the constant gamma parameter used to define the search radius in RRT* and the dimension of state space.

        Args:
            None

        Returns:
            None: This method does not return any value. 
        """

        self.d = sum(self.env.robot_dims.values())
        unit_ball_volume = math.pi ** (self.d/ 2) / math.gamma((self.d / 2) + 1)
        self.get_lebesgue_measure_of_free_configuration_space()
        self.gamma_rrtstar = ((2 *(1 + 1/self.d))**(1/self.d) * (self.c_free/unit_ball_volume)**(1/self.d))*self.eta
    
    def get_next_ids(self, mode:Mode) -> List[int]:
        """
        Retrieves valid combination of next task IDs for given mode.

        Args:
            mode (Mode): Mode for which to retrieve valid next task ID combinations.

        Returns:
            Optional[List[int]]: Randomly selected valid combination of all next task ID combinations if available
        """

        possible_next_task_combinations = self.env.get_valid_next_task_combinations(mode)
        if len(possible_next_task_combinations) == 0:
            return None
        return random.choice(possible_next_task_combinations)

    def get_home_poses(self, mode:Mode) -> List[NDArray]:
        """
        Retrieves home poses (i.e., the most recent completed task configurations) for all agents of given mode.

        Args:
            mode (Mode): Mode for which to retrieve home poses.

        Returns:
            List[NDArray]: Representing home poses for each agent.
        """

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

    def get_task_goal_of_agent(self, mode:Mode, r:str) -> NDArray:
        """Returns task goal of agent in current mode"""
        """
        Retrieves task goal configuration of given mode for the specified robot.

        Args:
            mode (Mode): Mode for which to retrieve task goal configuration.
            r (str): The identifier of the robot whose task goal is to be retrieved.

        Returns:
            NDArray: Goal configuration for the specified robot. 
        """

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

    def sample_transition_configuration(self, mode) -> Configuration:
        """
        Samples a collision-free transition configuration for the given mode.

        Args:
            mode (Mode): Mode for which a transition configuration is to be sampled.

        Returns:
            Configuration: Collision-free configuration constructed by combining goal samples (active robots) with random samples (non-active robots).
        """

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
    
    def sample_configuration(self, 
                             mode:Mode, 
                             sampling_type: str, 
                             transition_node_ids:Dict[Mode, List[int]] = None, 
                             tree_order:int = 1
                             ) -> Configuration:
        """
        Samples a collision-free configuration for the given mode using the specified sampling strategy.
        
        Args: 
            mode (Mode): Mode for which a configuration is to be sampled. 
            sampling_type (str): String type specifying the sampling strategy to use. 
            transition_node_ids (Optional[Dict[Mode, List[int]]]): Dictionary mapping modes to lists of transition node IDs. 
            tree_order (int): Order of the subtrees (i.e. value of 1 indicates sampling from 'subtree' (primary); otherwise from 'subtree_b')

        Returns: 
            Configuration:Collision-free configuration within specified limits for the robots based on the sampling strategy. 
        """
        is_goal_sampling = sampling_type == "goal"
        is_informed_sampling = sampling_type == "informed"
        is_home_pose_sampling = sampling_type == "home_pose"
        is_gaussian_sampling = sampling_type == "gaussian"
        constrained_robots = self.env.get_active_task(mode, self.get_next_ids(mode)).robots
        attemps = 0  # needed if home poses are in collision

        while True:
            #goal sampling
            if is_goal_sampling:
                if tree_order == -1:
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
                        
                if self.operation.init_sol and self.informed: 
                    if self.informed_sampling_version == 6 and not self.env.is_terminal_mode(mode):
                        q = self.informed[mode].generate_informed_transitions(
                            200,
                            self.operation.path, mode, locally_informed_sampling = self.locally_informed_sampling
                        )
                        if q is None:
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
                        if self.env.is_collision_free(q, mode):
                            return q
                        continue
                    elif not self.informed_sampling_version == 1 and not self.informed_sampling_version == 6:
                        q = self.sample_informed(mode, True)
                        q = type(self.env.get_start_pos()).from_list(q)
                        if self.env.is_collision_free(q, mode):
                            return q
                        continue
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
            #informed sampling       
            if is_informed_sampling: 
                if self.informed_sampling_version == 6:
                    q = self.informed[mode].generate_informed_samples(
                            200,
                            self.operation.path, mode, locally_informed_sampling = self.locally_informed_sampling
                        )
                    if q is None:
                        is_informed_sampling = False
                        continue
                    if self.env.is_collision_free(q, mode):
                        return q
                elif not self.informed_sampling_version == 6:
                    q = self.sample_informed(mode)
            #gaussian noise
            if is_gaussian_sampling: 
                path_state = np.random.choice(self.operation.path)
                standar_deviation = np.random.uniform(0, 5.0)
                # standar_deviation = 0.5
                noise = np.random.normal(0, standar_deviation, path_state.q.state().shape)
                q = (path_state.q.state() + noise).tolist()
            #home pose sampling or uniform sampling
            else: 
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
            q = type(self.env.get_start_pos()).from_list(q)
            if self.env.is_collision_free(q, mode):
                return q
            if attemps > 100: # if home pose causes failed attemps
                is_home_pose_sampling = False
    
    def sample_informed(self, mode:Mode, goal_sampling:bool = False) -> None:
        """ 
        Samples configuration from informed set for the given mode.

        Args: 
            mode (Mode): Mode for which the configuration is to be sampled. 
            goal_sampling (bool): Flag indicating whether to perform standard or goal-directed sampling within the informed set.

        Returns: 
            Configuration: Configuration within the informed set that satisfies the specified limits for the robots. """

        if not self.informed_sampling_version == 1:
            q_rand = []
            update = False
            if self.operation.cost != self.informed[mode].cost:
                self.informed[mode].cost = self.operation.cost
                update = True
            if goal_sampling:
                constrained_robot = self.env.get_active_task(mode, self.get_next_ids(mode)).robots
                goal = self.env.get_active_task(mode, self.get_next_ids(mode)).goal.sample(mode)
                q_rand = []
                end_idx = 0
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
                    if robot in constrained_robot:
                        dim = self.env.robot_dims[robot]
                        indices = list(range(end_idx, end_idx + dim))
                        q_rand.append(goal[indices])
                        end_idx += dim 
                        continue
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
                return q_rand 
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
                    return q_rand
             
    def sample_unit_n_ball(self, n:int) -> NDArray:
        """ 
        Samples a point uniformly from a n-dimensional unit ball centered at the origin.

        Args: 
            n (int): Dimension of the ball.

        Returns: 
            NDArray: Sampled point from the unit ball. """
        x_ball = np.random.normal(0, 1, n) 
        x_ball /= np.linalg.norm(x_ball, ord=2)     # Normalize with L2 norm
        radius = np.random.rand()**(1 / n)          # Generate a random radius and apply power
        return x_ball * radius
    
    def get_termination_modes(self) -> List[Mode]:
        """ 
        Retrieves a list of modes that are considered terminal based on the environment's criteria.

        Args: 
            None

        Returns: 
            List[Mode]:List containing all terminal modes. """

        termination_modes = []
        for mode in self.modes:
            if self.env.is_terminal_mode(mode):
                termination_modes.append(mode)
        return termination_modes

    def Nearest(self, 
                mode:Mode, 
                q_rand: Configuration, 
                tree: str = ''
                ) -> Tuple[Node, float, NDArray, int]:
        """
        Retrieves nearest node to a configuration in the specified subree for a given mode.

        Args:
            mode (Mode): Current operational mode.
            q_rand (Configuration): Random configuration for which the nearest node is sought.
            tree (str): Identifier of subtree in which the nearest node is searched for
        Returns:
            Tuple:   
                - Node: Nearest node.
                - float: Distance from q_rand to nearest node.
                - NDArray: Array of distances from q_rand to all nodes in the specified subtree.
                - int: Index of the nearest node in the distance array.
        """

        set_dists = batch_config_dist(q_rand, self.trees[mode].get_batch_subtree(tree), self.distance_metric)
        idx = np.argmin(set_dists)
        node_id = self.trees[mode].get_node_ids_subtree(tree)[idx]
        # print([float(set_dists[idx])])
        return  self.trees[mode].get_node(node_id, tree), set_dists[idx], set_dists, idx
    
    def Steer(self, 
              mode:Mode, 
              n_nearest: Node, 
              q_rand: Configuration, 
              dist: NDArray, 
              i:int=1
              ) -> State: 
        """
        Steers from the nearest node toward the target configuration by taking an incremental step.

        Args:
            mode (Mode): Current operational mode.
            n_nearest (Node): Nearest node from which steering begins.
            q_rand (Configuration): Target random configuration for steering.
            dist (NDArray): Distance between the nearest node and q_rand.
            i (int): Step index for incremental movement. 

        Returns:
            State: New obtained state by steering from n_nearest towards q_rand (i.e. q_rand, if within the allowed step size; otherwise, advances by one incremental step).
        """

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
    
    def Near(self, 
             mode:Mode, 
             n_new: Node, 
             n_nearest_idx:int, 
             set_dists:NDArray=None
             ) -> Tuple[NDArray, NDArray, NDArray]:      
        """
        Retrieves neighbors of a node within a calculated radius for the given mode.

        Args:
            mode (Mode):  Current operational mode.
            n_new (Node): New node for which neighbors are being identified.
            n_nearest_idx (int): Index of the nearest node to n_new.
            set_dists (Optional[NDArray]): Precomputed distances from n_new to all nodes in the specified subtree.

        Returns:
            Tuple:   
                - NDArray: Batch of neighbors near n_new.
                - NDArray: Corresponding cost values of these nodes.
                - NDArray: Corresponding IDs of these nodes.
        """

        batch_subtree = self.trees[mode].get_batch_subtree()
        if set_dists is None:
            set_dists = batch_config_dist(n_new.state.q, batch_subtree, self.distance_metric)
        vertices = self.trees[mode].get_number_of_nodes_in_tree()
        r = np.minimum(self.gamma_rrtstar*(np.log(vertices)/vertices)**(1/self.d), self.eta)
        indices = find_nearest_indices(set_dists, r) # indices of batch_subtree
        if n_nearest_idx not in indices:
            indices = np.insert(indices, 0, n_nearest_idx)
        node_indices = self.trees[mode].node_ids_subtree[indices]
        n_near_costs = self.operation.costs[node_indices]
        N_near_batch = batch_subtree[indices]
        return N_near_batch, n_near_costs, node_indices
    
    def FindParent(self, 
                   mode:Mode, 
                   node_indices: NDArray, 
                   n_new: Node, 
                   n_nearest: Node, 
                   batch_cost: NDArray, 
                   n_near_costs: NDArray
                   ) -> None:
        """
        Sets the optimal parent for a new node by evaluating connection costs among candidate nodes.

        Args:
            mode (Mode): Current operational mode.
            node_indices (NDArray): Array of IDs representing candidate neighboring nodes.
            n_new (Node): New node that needs a parent connection.
            n_nearest (Node): Nearest candidate node to n_new.
            batch_cost (NDArray): Costs associated from n_new to all candidate neighboring nodes.
            n_near_costs (NDArray): Cost values for all candidate neighboring nodes.

        Returns:
            None: This method does not return any value.
        """

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
    
    def Rewire(self, 
               mode:Mode,  
               node_indices: NDArray, 
               n_new: Node, 
               batch_cost: NDArray, 
               n_near_costs: NDArray
               ) -> bool:
        """
        Rewires neighboring nodes by updating their parent connection to n_new if a lower-cost path is established.

        Args:
            mode (Mode): Current operational mode.
            node_indices (NDArray): Array of IDs representing candidate neighboring nodes.
            n_new (Node): New node as potential parent for neighboring nodes.
            batch_cost (NDArray): Costs associated from n_new to all candidate neighboring nodes.
            n_near_costs (NDArray): Cost values for all candidate neighboring nodes.

        Returns:
            bool: True if any neighbor's parent connection is updated to n_new; otherwise, False.
        """

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
      
    def GeneratePath(self, 
                     mode:Mode, 
                     n: Node, 
                     shortcutting_bool:bool = True
                     ) -> None:
        """
        Sets path from the specified node back to the root by following parent links, with optional shortcutting.

        Args:
            mode (Mode): Current operational Mode.
            n (Node): Starting node from which the path is generated.
            shortcutting_bool (bool): If True, applies shortcutting to the generated path.

        Returns:
            None: This method does not return any value.
        """

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
            shortcut_path, _ = mrmgp.shortcutting.robot_mode_shortcut(
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
            
    def RandomMode(self) -> Mode:
        """
        Randomly selects a mode based on the current mode sampling strategy.

        Args:
            None

        Returns:
            Mode: Sampled Mode based on the mode sampling strategy.
        """

        num_modes = len(self.modes)
        if num_modes == 1:
            return np.random.choice(self.modes)
        # if self.operation.task_sequence == [] and self.mode_sampling != 0:
        elif self.operation.init_sol and self.mode_sampling != 0:
                p = [1/num_modes] * num_modes
        
        elif self.mode_sampling is None:
            # equally (= mode uniformly)
            return np.random.choice(self.modes)

        elif self.mode_sampling == 1: 
            # greedy (only latest mode is selected until initial paths are found and then it continues with equally)
            probability = [0] * (num_modes)
            probability[-1] = 1
            p =  probability

        elif self.mode_sampling == 0:
            # Uniformly
            total_nodes = sum(self.trees[mode].get_number_of_nodes_in_tree() for mode in self.modes)
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
            total_nodes = sum(self.trees[mode].get_number_of_nodes_in_tree() for mode in self.modes[:-1]) 
            # Calculate probabilities inversely proportional to node counts
            inverse_probabilities = [
                1 - (len(self.trees[mode].subtree) / total_nodes)
                for mode in self.modes[:-1]
            ]

            # Normalize the probabilities of all modes except the last one
            remaining_probability = 1-self.mode_sampling  
            total_inverse = sum(inverse_probabilities)
            if total_inverse == 0:
                p = [1-self.mode_sampling, self.mode_sampling]
            else:
                p =  [
                    (inv_prob / total_inverse) * remaining_probability
                    for inv_prob in inverse_probabilities
                ] + [self.mode_sampling]

        return np.random.choice(self.modes, p = p)

    def InformedInitialization(self, mode: Mode) -> None: 
        """
        Initializes the informed sampling module for the given mode based on the specified version.

        Args:
            mode (Mode): Current operational mode for which informed sampling is initialized.

        Returns:
            None: This method does not return any value.
        """

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
        if self.informed_sampling_version == 6:
            self.informed[mode] = InformedVersion6(self.env, self.modes, self.locally_informed_sampling)

    def SampleNodeManifold(self, mode:Mode) -> Configuration:
        """
        Samples a node configuration from the manifold based on various probabilistic strategies.

        Args:
            mode (Mode): Current operational mode for which the configuration is sampled.

        Returns:
            Configuration: Configuration obtained by a sampling strategy based on preset probabilities and operational conditions.
        """

        if  np.random.uniform(0, 1) < self.p_goal:
            # goal sampling
            return self.sample_configuration(mode, "goal", self.transition_node_ids, self.trees[mode].order)
        else:       
            if self.informed_sampling and self.operation.init_sol: 
                if self.informed_sampling_version == 0 and np.random.uniform(0, 1) < self.p_uniform or self.informed_sampling_version == 5 and np.random.uniform(0, 1) < self.p_uniform:
                    #uniform sampling
                    return self.sample_configuration(mode, "uniform")
                #informed_sampling
                return self.sample_configuration(mode, "informed")
            # gaussian sampling
            if self.gaussian and self.operation.init_sol: 
                return self.sample_configuration(mode, "gaussian")
            # home pose sampling
            if self.p_stay != 0 and np.random.uniform(0, 1) < self.p_stay: 
                return self.sample_configuration(mode, "home_pose")
            #uniform sampling
            return self.sample_configuration(mode, "uniform")
               
    def FindLBTransitionNode(self) -> None:
        """
        Searches lower-bound transition node and generates its corresponding path if a valid candidate is found.

        Args:
            None

        Returns:
            None: This method does not return any value.
        """

        if self.operation.init_sol: 
            modes = self.get_termination_modes()     
            result, mode = self.get_lb_transition_node_id(modes) 
            if not result:
                return
            valid_mask = result[0] < self.operation.cost
            if valid_mask.any():
                lb_transition_node = self.get_transition_node(mode, result[1])
                self.GeneratePath(mode, lb_transition_node)

    def UpdateDiscretizedCost(self, path:List[State], cost:NDArray, idx:int) -> None:
        """
        Updates discretized cost along a given path by recalculating segment costs from a specified index onward.

        Args:
            path (List[State]): The sequence of states representing the path.
            cost (NDArray): The array storing cumulative cost values along the path.
            idx (int): The index from which cost updates should begin.

        Returns:
            None: This method does not return any value.
        """

        path_a, path_b = path[idx-1: -1], path[idx: ]
        batch_cost = batch_config_cost(path_a, path_b, self.env.cost_metric, self.env.cost_reduction)
        cost[idx:] = cumulative_sum(batch_cost) + cost[idx-1]

    def Shortcutting(self,
                     active_mode:Mode, 
                     deterministic:bool = False
                     ) -> None:
        """
        Applies shortcutting logic based on the given mode.

        Args:
            active_mode (Mode): Current operational mode.
            deterministic (bool): If True, applies a deterministic approach. 

        Returns:
            None: This method does not return any value.
        """
        
        indices  = [self.env.robot_idx[r] for r in self.env.robots]

        discretized_path, discretized_modes, discretized_costs = self.Discretization(self.operation.path_nodes, indices)  

        termination_cost = discretized_costs[-1]
        termination_iter = 0
        dim = None

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
                                                            discretized_costs[idx1], indices, dim, self.shortcutting_dim_version, robot)
        
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

    def TreeExtension(self, 
                      active_mode:Mode, 
                      discretized_path:List[State], 
                      discretized_costs:List[float]
                      ) -> None:
        """
        Extends the tree by adding path states as nodes and updating parent-child relationships.

        Args:
            active_mode (Mode): Current mode for tree extension.
            discretized_path (List[State]):Sequence of states forming the discretized path.
            discretized_costs (List[float]): Associated costs for each state in the discretized path.

        Returns:
            None: This method does not return any value.
        """

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

    def EdgeInterpolation(self, 
                          path:List[State], 
                          cost:float, 
                          indices:List[List[int]], 
                          dim:Union[int, List[int]], 
                          version:int, 
                          r:Optional[int]
                          ) -> Tuple[List[State], List[float]]:
        """
            Interpolates an edge between two configurations along a path using different shortcutting strategies.

            Args:
                path (List[State]): Sequence of states representing path.
                cost (float): Initial cost associated with path segment.
                indices (List[List[int]]): Indices of relevant dimensions for each robot.
                dim (Union[int, List[int]]): Specific dimensions to interpolate.
                version (int): The interpolation strategy to use:
                    - 0: Uniform interpolation across all dimensions.
                    - 1: Shortcutting across all indices of a robot.
                    - 2: Partial shortcutting for a single dimension of a robot.
                    - 3: Partial shortcutting for a random set of dimensions.
                r (Optional[int]): Specific robot to apply shortcutting to.

            Returns:
                Tuple: 
                    - List[State]: Interpolated states forming the edge.
                    - List[float]: Computed costs for each segment the interpolated path.
            """

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
                for robot in range(len(self.env.robots)):
                    if r is not None and r == robot:
                        q[indices[robot]] = q0[indices[robot]] +  (segment_vector[indices[robot]] * (i /N))
                        break
                    if r is None:
                        q[indices[robot]] = q0[indices[robot]] +  (segment_vector[indices[robot]] * (i / N))

            elif version == 2: #partial shortcutting agent single dim 
                q = path[i].q.state().copy()
                for robot in range(len(self.env.robots)):
                    if r is not None and r == robot:
                        q[dim] = q0[dim] +  (segment_vector[dim] * (i / N))
                        break
                    if r is None:
                        q[dim[robot]] = q0[dim[robot]] +  (segment_vector[dim[robot]] * (i / N))
            
            elif version == 3: #partial shortcutting agent random set of dim 
                q = path[i].q.state().copy()
                for robot in range(len(self.env.robots)):
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
            edge_cost = cumulative_sum(batch_cost) + cost
        return edge, edge_cost

    def Discretization(self, 
                       path:List[State], 
                       indices: List[List[int]], 
                       resolution:float=0.1
                       ) -> Tuple[List[State], List[List[int]], List[float]]:
        """
        Discretizes a given path into intermediate states based on specified resolution.

        Args:
            path (List[State]): Sequence of states representing original path.
            indices (List[List[int]]): Indices of relevant dimensions for interpolation.
            resolution (float, optional): Step size used to determine the number of discretized points.

        Returns:
            Tuple:
                - List[State]: List of discretized states forming the refined path.
                - List[List[int]: List of mode transitions associated with each discretized state.
                - List[float]: List of cumulative costs along the discretized path.
        """

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
        discretized_costs = cumulative_sum(batch_cost)
        return discretized_path, discretized_modes, discretized_costs

    @abstractmethod
    def UpdateCost(self, n: Node) -> None:
        """
        Updates cost for a given node and all its descendants by propagating the cost change down the tree.

        Args:
            n (Node): Root node from which the cost update begins.

        Returns:
            None: This method does not return any value.
        """
        pass
    @abstractmethod
    def PlannerInitialization(self) -> None:
        """
        Initializes planner by setting parameters, creating the initial mode, and adding start node.

        Args:
            None

        Returns:
            None: None: This method does not return any value.
        """
        pass
    @abstractmethod
    def ManageTransition(self, mode:Mode, n_new: Node) -> None:
        """
        Checks if new node qualifies as a transition or termination node. 
        If it does, node is converted into transition node, mode is updated accordingly, 
        and a new path is generated if node is the lower-bound transition node. 

        Args:
            mode (Mode): The current operational mode.
            n_new (Node): The newly added node to evaluate for triggering a transition or termination.

        Returns:
            None: This method does not return any value.
        """
        pass
    @abstractmethod
    def Plan(self) -> Tuple[List[State], Dict[str, List[Union[float, float, List[State]]]]]:
        """
        Executes planning process using an RRT* framework.

        Args:
            None

        Returns:
            Tuple:
                - List[State]: The planned path as a list of states.
                - Dict[str, List]: A dictionary containing:
                    - "costs" (List[float]): Recorded path costs.
                    - "times" (List[float]): Recorded execution times.
                    - "paths" (List[List[State]]): All explored paths.


        """
        pass
