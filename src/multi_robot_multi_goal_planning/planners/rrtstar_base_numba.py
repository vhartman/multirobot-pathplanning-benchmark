import numpy as np
import logging
from datetime import datetime
import yaml
import json
import pickle

from multi_robot_multi_goal_planning.problems.configuration import *
from multi_robot_multi_goal_planning.problems.planning_env import *
from multi_robot_multi_goal_planning.problems.memory_util import *
import argparse
import time as time
import math as math
from typing import Tuple, Optional, Union
from functools import cache
from numba import njit

class ConfigManager:
    def __init__(self, config_file:str):
        # Load configuration
        config_file_path = os.path.join(os.path.dirname(__file__), "../../../examples/")+ config_file
        print(config_file_path)
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Set defaults and dynamically assign attributes
        defaults = {
            'planner': None, 'p_goal': 0.95, 'p_stay' : 0.95,'general_goal_sampling': True, 'gaussian' :True,
            'step_size': 0.3, 'cost_metric' : 'euclidean', 'cost_reduction' : 'max',
            'ptc_time': 600, 'mode_probability': 0.4, 'p_uniform': 0.3,
            'informed_sampling': True, 'informed_sampling_version': 0, 
            'cprofiler': False, 'cost_type': 'euclidean', 'dist_type': 'euclidean', 
            'debug_mode': False, 'transition_nodes': 100, 'birrtstar_version' :1, 
            'amount_of_runs' : 1, 'use_seed' : True, 'seed': 1, 'depth': 1, 'batch_size': 2000, 'expand_iter': 10, 'shortcutting': True
        }
        for key, value in defaults.items():
            setattr(self, key, config.get(key, value))
        
        # Output directory
        self.timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        self.output_dir = os.path.join(os.path.expanduser("~"), f'output/{self.timestamp}/')

    def _setup_logger(self) -> None:
        logging.basicConfig(
            filename=os.path.join(self.output_dir, 'general.log'), 
            level=logging.INFO, 
            format='%(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def log_params(self, args: argparse) -> None:
        # Set up logging
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_logger()
        """Logs configuration parameters along with environment and planner details."""
        config_dict = {k: v for k, v in self.__dict__.items() if k != "logger"}
        self.logger.info('Environment: %s', json.dumps(args.env_name, indent=2))
        self.logger.info('Planner: %s', json.dumps(args.planner, indent=2))
        self.logger.info('Configuration Parameters: %s', json.dumps(config_dict, indent=4))
    
    def reset_logger(self):
        """Reset the logger by removing all handlers."""
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

class Operation:
    """ Planner operation variables"""
    def __init__(self):
        
        self.path = []
        self.path_nodes = None
        self.cost = np.inf
        self.cost_change = np.inf
        self.init_sol = False
        self.costs = np.empty(10000000, dtype=np.float32)
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

class SingleTree(BaseTree):
    """Single tree description"""
    def __init__(self, env):
        self.order = 1
        # self.informed = Informed()
        robot_dims = sum(env.robot_dims.values())
        self.subtree = {}
        self.initial_capacity = 100000
        self.batch_subtree = np.empty((self.initial_capacity, robot_dims), dtype=np.float32)
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

class BidirectionalTree(BaseTree):
    """Bidirectional tree description"""
    def __init__(self, env):
        self.order = 1
        # self.informed = Informed()
        robot_dims = sum(env.robot_dims.values())
        self.subtree = {}
        self.initial_capacity = 100000
        self.batch_subtree = np.empty((self.initial_capacity, robot_dims), dtype=np.float32)
        self.node_idx_subtree = np.empty(self.initial_capacity, dtype=np.int64)
        self.subtree_b = {} 
        self.batch_subtree_b = np.empty((self.initial_capacity, robot_dims), dtype=np.float32)
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


class Informed(ABC):
    def __init__(self):
        pass
    
    def rotation_to_world_frame(self, start:Configuration, goal:Configuration, n:int) -> Tuple[float, NDArray, NDArray]:
        """ Returns: 
                Norm and rotation matrix C from the hyperellipsoid-aligned frame to the world frame."""
        diff = goal.state() - start.state()
        norm = config_dist(start, goal, 'euclidean')
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
    
    @abstractmethod
    def get_right_side_of_eq():
        pass
    @abstractmethod
    def initialize():
        pass
    @abstractmethod
    def cholesky_decomposition(self, cmin, path_nodes, modes):
        pass
    
class InformedVersion0(Informed):

    def __init__(self, env, cost_fct, cost_metric, cost_reduction):
        self.env = env
        self.cost_metric = cost_metric
        self.cost_reduction = cost_reduction
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
        n1 = NpConfiguration.from_numpy(path_nodes[0].state.q.state()[r_indices])
        cost = 0
        if self.mode_task_ids_home_poses[r_idx] is None:
            start_cost = 0
        for node in path_nodes[1:]:
            n2 = NpConfiguration.from_numpy(node.state.q.state()[r_indices])
            cost += self.cost_fct(n1, n2, self.cost_metric, self.cost_reduction)
            #need to work with task ids as same mode can have different configs
            if node.transition and node.state.mode.task_ids == self.mode_task_ids_home_poses[r_idx]:
                start_cost = cost
            if node.transition and node.state.mode.task_ids == self.mode_task_ids_task[r_idx]:
                goal_cost = cost
                break 
            n1 = n2
        return goal_cost-start_cost   

    def initialize(self, start_config:Configuration, goal_config:Configuration, mode_task_ids_home_poses:List[List[int]]):
        for robot in self.env.robots:
            r_idx = self.env.robots.index(robot)
            if np.equal(goal_config[r_idx].state(), start_config[r_idx].state()).all():
                continue
            self.mode_task_ids_home_poses[r_idx] = mode_task_ids_home_poses[r_idx]
            self.start[r_idx] = start_config[r_idx]
            self.goal[r_idx] = goal_config[r_idx]
            self.n[r_idx] = self.env.robot_dims[robot]
            norm, self.C[r_idx] = self.rotation_to_world_frame(start_config[r_idx], goal_config[r_idx], self.n[r_idx])
            self.cmin[r_idx] = norm-2*self.env.collision_tolerance
            self.center[r_idx] = (goal_config[r_idx].state() + start_config[r_idx].state())/2
            
    def cholesky_decomposition(self, r_indices:List[int], r_idx:int, path_nodes:List[Node]):
        cmax = self.get_right_side_of_eq(r_indices, r_idx, path_nodes)
        r1 = cmax / 2
        r2 = np.sqrt(cmax**2 - self.cmin[r_idx]**2) / 2
        return np.diag(np.concatenate([[r1], np.repeat(r2, self.n[r_idx] - 1)]))   
class InformedVersion1(Informed):

    def __init__(self, env, cost_fct, cost_metric, cost_reduction):
        self.env = env
        self.cost_metric = cost_metric
        self.cost_reduction = cost_reduction
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

        if mode_task_ids_home_pose is None:
            self.start = path_nodes[0]

        for node in path_nodes:
            if node.transition and node.state.mode.task_ids == mode_task_ids_home_pose:
                self.start = node
            if node.transition and node.state.mode.task_ids == mode_task_ids_task:
                self.goal = node
                break 
        lb_goal = self.cost_fct(self.goal.state.q, path_nodes[-1].state.q,  self.cost_metric, self.cost_reduction)
        lb_start = self.cost_fct( path_nodes[0].state.q, self.start.state.q,  self.cost_metric, self.cost_reduction)
        norm, self.C = self.rotation_to_world_frame(self.start.state.q, self.goal.state.q, self.n)
        self.cmin = norm-2*self.env.collision_tolerance
        self.center = (self.goal.state.q.state() + self.start.state.q.state())/2
        return path_nodes[-1].cost - lb_goal - lb_start
    
    def initialize(self, mode:Mode, mode_task_ids_home_poses:List[List[int]]):
        active_robots = self.env.get_active_task(mode, None).robots
        for robot in self.env.robots:
            if robot in active_robots:
                self.active_robots_idx.append(self.env.robots.index(robot))
        self.mode_task_ids_home_poses = mode_task_ids_home_poses
        self.n = sum(self.env.robot_dims.values())

    def cholesky_decomposition(self, path_nodes:List[Node]):
        cmax = self.get_right_side_of_eq( path_nodes)
        r1 = cmax / 2
        r2 = np.sqrt(cmax**2 - self.cmin**2) / 2
        return np.diag(np.concatenate([np.repeat(r1, 1), np.repeat(r2, self.n - 1)]))   
class InformedVersion2(Informed):
    """Similar to InformedVersion1 but instead of using the actual start and end path nodes use random samples that are just ouside of the range of start and goal"""

    def __init__(self, env, cost_fct, cost_metric, cost_reduction):
        self.env = env
        self.cost_metric = cost_metric
        self.cost_reduction = cost_reduction
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
            i = self.active_robots_idx[0]
        else:
            i = random.choice(self.active_robots_idx)

        mode_task_ids_home_pose = self.mode_task_ids_home_poses[i]
        mode_task_ids_task = self.mode_task_ids_task[i]
        if mode_task_ids_home_pose is None:
            self.start = path_nodes[0]
            start = 0

        for idx, node in enumerate(path_nodes):
            if node.transition and node.state.mode.task_ids == mode_task_ids_home_pose:
                self.start = node
                start = idx
            if node.transition and node.state.mode.task_ids == mode_task_ids_task:
                self.goal = node
                goal = idx
                break 

        idx1 = np.random.randint(0, start+1)
        idx2 = np.random.randint(goal, len(path_nodes))


        lb_goal = self.cost_fct(self.goal.state.q, path_nodes[idx2].state.q,  self.cost_metric, self.cost_reduction)
        lb_start = self.cost_fct( path_nodes[idx1].state.q, self.start.state.q,  self.cost_metric, self.cost_reduction)
        norm, self.C = self.rotation_to_world_frame(self.start.state.q, self.goal.state.q, self.n)
        self.cmin = norm-2*self.env.collision_tolerance
        self.center = (self.goal.state.q.state() + self.start.state.q.state())/2
        return path_nodes[idx2].cost - lb_goal - lb_start
    
    def initialize(self, mode:Mode, mode_task_ids_home_poses:List[List[int]]):
        active_robots = self.env.get_active_task(mode, None).robots
        for robot in self.env.robots:
            if robot in active_robots:
                self.active_robots_idx.append(self.env.robots.index(robot))
        self.mode_task_ids_home_poses = mode_task_ids_home_poses
        self.n = sum(self.env.robot_dims.values())

    def cholesky_decomposition(self, path_nodes:List[Node]):
        cmax = self.get_right_side_of_eq(path_nodes)
        r1 = cmax / 2
        r2 = np.sqrt(cmax**2 - self.cmin**2) / 2
        return np.diag(np.concatenate([np.repeat(r1, 1), np.repeat(r2, self.n - 1)]))
class InformedVersion4(Informed):
    """Just select 2 random points as start and goal, nothing to do with mode (by Valentin)"""
    def __init__(self, env, cost_fct, cost_metric, cost_reduction):
        self.env = env
        self.cost_metric = cost_metric
        self.cost_reduction = cost_reduction
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
        idx1 = np.random.randint(0, len(path_nodes))
        idx2 = np.random.randint(0, len(path_nodes))
        if idx2 < idx1:
            idx1, idx2 = idx2, idx1
        goal = path_nodes[idx2]
        start = path_nodes[idx1]

        lb_goal = self.cost_fct(goal.state.q, path_nodes[-1].state.q,  self.cost_metric, self.cost_reduction)
        lb_start = self.cost_fct( path_nodes[0].state.q, start.state.q,  self.cost_metric, self.cost_reduction)
        norm, self.C = self.rotation_to_world_frame(start.state.q, goal.state.q, self.n)
        self.cmin = norm-2*self.env.collision_tolerance
        self.center = (goal.state.q.state() + start.state.q.state())/2
        return path_nodes[-1].cost - lb_goal - lb_start
    
    def initialize(self,mode):
        active_robots = self.env.get_active_task(mode, None).robots
        for robot in self.env.robots:
            if robot in active_robots:
                self.active_robots_idx.append(self.env.robots.index(robot))
        self.n = sum(self.env.robot_dims.values())

    def cholesky_decomposition(self, path_nodes:List[Node]):
        cmax = self.get_right_side_of_eq(path_nodes)
        r1 = cmax / 2
        r2 = np.sqrt(cmax**2 - self.cmin**2) / 2
        return np.diag(np.concatenate([np.repeat(r1, 1), np.repeat(r2, self.n - 1)]))
class InformedVersion5(Informed):
    """Adaption of InformedVersion2 -> choose between start and goal and new start an goal and choose based on that the actual start and end path nodes use random samples that are just ouside of the range of start and goal"""
    def __init__(self, env, cost_fct, cost_metric, cost_reduction):
        self.env = env
        self.cost_metric = cost_metric
        self.cost_reduction = cost_reduction
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
            i = self.active_robots_idx[0]
        else:
            i = random.choice(self.active_robots_idx)

        mode_task_ids_home_pose = self.mode_task_ids_home_poses[i]
        mode_task_ids_task = self.mode_task_ids_task[i]
        if mode_task_ids_home_pose is None:
            self.start = path_nodes[0]
            start = 0

        for idx, node in enumerate(path_nodes):
            if node.transition and node.state.mode.task_ids == mode_task_ids_home_pose:
                self.start = node
                start = idx
            if node.transition and node.state.mode.task_ids == mode_task_ids_task:
                self.goal = node
                goal = idx
                break 
            
        idxstart = np.random.randint(start, goal+1)
        idxgoal = np.random.randint(start, goal+1)
        if idxstart > idxgoal:
            idxstart, idxgoal = idxgoal, idxstart
        

        idx1 = np.random.randint(0, idxstart+1)
        idx2 = np.random.randint(idxgoal, len(path_nodes))


        lb_goal = self.cost_fct(path_nodes[idxgoal].state.q, path_nodes[idx2].state.q,  self.cost_metric, self.cost_reduction)
        lb_start = self.cost_fct( path_nodes[idx1].state.q, path_nodes[idxstart].state.q,  self.cost_metric, self.cost_reduction)
        norm, self.C = self.rotation_to_world_frame(self.start.state.q, self.goal.state.q, self.n)
        self.cmin = norm-2*self.env.collision_tolerance
        self.center = (self.goal.state.q.state() + self.start.state.q.state())/2
        return path_nodes[idx2].cost - lb_goal - lb_start
    
    def initialize(self, mode:Mode, mode_task_ids_home_poses:List[List[int]]):
        active_robots = self.env.get_active_task(mode, None).robots
        for robot in self.env.robots:
            if robot in active_robots:
                self.active_robots_idx.append(self.env.robots.index(robot))
        self.mode_task_ids_home_poses = mode_task_ids_home_poses
        self.n = sum(self.env.robot_dims.values())

    def cholesky_decomposition(self, path_nodes:List[Node]):
        cmax = self.get_right_side_of_eq(path_nodes)
        r1 = cmax / 2
        r2 = np.sqrt(cmax**2 - self.cmin**2) / 2
        return np.diag(np.concatenate([np.repeat(r1, 1), np.repeat(r2, self.n - 1)]))    
class InformedVersion6(Informed):
    """Adaption of InformedVersion2 -> choose between start and goal and new start an goal and choose based on that the actual start and end path nodes use random samples that are just ouside of the range of start and goal"""
    def __init__(self, env, cost_fct, cost_metric, cost_reduction):
        self.env = env
        self.cost_metric = cost_metric
        self.cost_reduction = cost_reduction
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
            i = self.active_robots_idx[0]
        else:
            i = random.choice(self.active_robots_idx)

        mode_task_ids_home_pose = self.mode_task_ids_home_poses[i]
        mode_task_ids_task = self.mode_task_ids_task[i]
        if mode_task_ids_home_pose is None:
            self.start = path_nodes[0]
            start = 0

        for idx, node in enumerate(path_nodes):
            if node.transition and node.state.mode.task_ids == mode_task_ids_home_pose:
                self.start = node
                start = idx
            if node.transition and node.state.mode.task_ids == mode_task_ids_task:
                self.goal = node
                goal = idx
                break 
            
        idxstart = np.random.randint(start, goal+1)
        idxgoal = np.random.randint(start, goal+1)
        if idxstart > idxgoal:
            idxstart, idxgoal = idxgoal, idxstart
        
        
        idx1 = np.random.randint(0, start+1)
        idx2 = np.random.randint(goal, len(path_nodes))


        lb_goal = self.cost_fct(path_nodes[idxgoal].state.q, path_nodes[idx2].state.q,  self.cost_metric, self.cost_reduction)
        lb_start = self.cost_fct( path_nodes[idx1].state.q, path_nodes[idxstart].state.q,  self.cost_metric, self.cost_reduction)
        norm, self.C = self.rotation_to_world_frame(self.start.state.q, self.goal.state.q, self.n)
        self.cmin = norm-2*self.env.collision_tolerance
        self.center = (self.goal.state.q.state() + self.start.state.q.state())/2
        return path_nodes[idx2].cost - lb_goal - lb_start
    
    def initialize(self, mode:Mode, mode_task_ids_home_poses:List[List[int]]):
        active_robots = self.env.get_active_task(mode, None).robots
        for robot in self.env.robots:
            if robot in active_robots:
                self.active_robots_idx.append(self.env.robots.index(robot))
        self.mode_task_ids_home_poses = mode_task_ids_home_poses
        self.n = sum(self.env.robot_dims.values())

    def cholesky_decomposition(self, path_nodes:List[Node]):
        cmax = self.get_right_side_of_eq(path_nodes)
        r1 = cmax / 2
        r2 = np.sqrt(cmax**2 - self.cmin**2) / 2
        return np.diag(np.concatenate([np.repeat(r1, 1), np.repeat(r2, self.n - 1)]))   
class InformedVersion7(Informed):
    """Similar to InformedVersion2 but using actual start and end and random nodes in between start and goal mode"""

    def __init__(self, env, cost_fct, cost_metric, cost_reduction):
        self.env = env
        self.cost_metric = cost_metric
        self.cost_reduction = cost_reduction
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
            i = self.active_robots_idx[0]
        else:
            i = random.choice(self.active_robots_idx)

        mode_task_ids_home_pose = self.mode_task_ids_home_poses[i]
        mode_task_ids_task = self.mode_task_ids_task[i]
        if mode_task_ids_home_pose is None:
            self.start = path_nodes[0]
            start = 0

        for idx, node in enumerate(path_nodes):
            if node.transition and node.state.mode.task_ids == mode_task_ids_home_pose:
                self.start = node
                start = idx
            if node.transition and node.state.mode.task_ids == mode_task_ids_task:
                self.goal = node
                goal = idx
                break 

        idx1 = np.random.randint(start, goal+1)
        idx2 = np.random.randint(start, goal+1)
        if idx1 > idx2: 
            idx1 , idx2 = idx2, idx1



        lb_goal = self.cost_fct(path_nodes[idx1].state.q ,path_nodes[-1].state.q,  self.cost_metric, self.cost_reduction)
        lb_start = self.cost_fct( path_nodes[0].state.q, path_nodes[idx2].state.q,  self.cost_metric, self.cost_reduction)
        norm, self.C = self.rotation_to_world_frame(self.start.state.q, self.goal.state.q, self.n)
        self.cmin = norm-2*self.env.collision_tolerance
        self.center = (self.goal.state.q.state() + self.start.state.q.state())/2
        return path_nodes[-1].cost - lb_goal - lb_start
    
    def initialize(self, mode:Mode, mode_task_ids_home_poses:List[List[int]]):
        active_robots = self.env.get_active_task(mode, None).robots
        for robot in self.env.robots:
            if robot in active_robots:
                self.active_robots_idx.append(self.env.robots.index(robot))
        self.mode_task_ids_home_poses = mode_task_ids_home_poses
        self.n = sum(self.env.robot_dims.values())

    def cholesky_decomposition(self, path_nodes:List[Node]):
        cmax = self.get_right_side_of_eq(path_nodes)
        r1 = cmax / 2
        r2 = np.sqrt(cmax**2 - self.cmin**2) / 2
        return np.diag(np.concatenate([np.repeat(r1, 1), np.repeat(r2, self.n - 1)]))


@njit(fastmath=True, cache=True)
def find_nearest_indices(set_dists, r):
    return np.nonzero(set_dists < r)[0]


class BaseRRTstar(ABC):
    def __init__(self, env, config: ConfigManager):
        self.env = env
        # self.gamma = ((2 *(1 + 1/self.dim))**(1/self.dim) * (self.FreeSpace()/self.UnitBallVolume())**(1/self.dim))*1.1 
        self.config = config
        self.r = self.config.step_size * sum(self.env.robot_dims.values())
        self.operation = Operation()
        self.start = time.time()
        self.start_single_goal= SingleGoal(self.env.start_pos.q)
        self.modes = [] 
        self.trees = {}
        self.transition_node_ids = {}
        self.informed = {}
        self.costs = [1e10] # cost of infinity
        self.times = [time.time()-self.start]

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
        
    def get_tree(self, mode:Mode) -> None:
        """Returns tree of mode"""
        return self.trees.get(mode, None)
    
    def add_new_mode(self, q:Optional[Configuration]=None, mode:Mode=None, tree_instance: Optional[Union["SingleTree", "BidirectionalTree"]] = None) -> None: #TODO entry_configuration needs to be specified
        """Initializes a new mode"""
        if mode is None: 
            new_mode = self.env.start_mode
            new_mode.prev_mode = None
        else:
            new_mode = self.env.get_next_mode(q, mode)
            new_mode.prev_mode = mode
        self.modes.append(new_mode)
        self.add_tree(new_mode, tree_instance)

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
        if not self.env.is_terminal_mode(mode):
            next_mode = self.modes[mode.id +1]
            self.trees[next_mode].add_transition_node_as_start_node(n)

    def get_lb_transition_node_id(self, mode:Mode) -> Tuple[float, int]:
        """Returns lb cost and index of transition nodes of the mode"""
        idx = np.argmin(self.operation.costs[self.transition_node_ids[mode]], axis=0) 
        cost = self.operation.costs[self.transition_node_ids[mode]][idx]
        node_id = self.transition_node_ids[mode][idx]
        return (cost, node_id)
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

    # @cache
    def get_home_poses(self, mode:Mode) -> List[NDArray]:
        """Returns home pose (most recent completed task) of agents in current mode """
        # Start mode
        if mode.id == 0:
            q = self.env.start_pos
            q_new = []
            for r_idx in range(len(self.env.robots)):
                q_new.append(q.robot_state(r_idx))
        # all other modes
        else:
            previous_task = self.env.get_active_task(mode.prev_mode, None) 
            q = self.get_home_poses(mode.prev_mode)
            q_new = []
            for r in self.env.robots:
                r_idx = self.env.robots.index(r)
                if r in previous_task.robots:
                    #if several goals are available get random one
                    goal = previous_task.goal.sample(mode.prev_mode)
                    if len(goal) == self.env.robot_dims[r]:
                        q_new.append(goal)
                    else:
                        q_new.append(goal[r_idx])
                else:
                    q_new.append(q[r_idx])
        return q_new
    # @cache
    def get_task_goal_of_agent(self, mode:Mode, r:str):
        """Returns task goal of agent in current mode"""
        task = self.env.get_active_task(mode, None)
        if r not in task.robots:
            r_idx = self.env.robots.index(r)
            goal = self.env.tasks[mode.task_ids[r_idx]].goal.sample(mode)
            if len(goal) == self.env.robot_dims[r]:
                return goal
            else:
                return goal[self.env.robot_idx[r]]
        goal = task.goal.sample(mode)
        if len(goal) == self.env.robot_dims[r]:
           return goal
        else:
            return goal[self.env.robot_idx[r]]
    @cache
    def get_mode_task_ids_of_home_pose_of_agent(self, mode:Mode):
        # Start mode
        if mode.id == 0:
            return [None, None]
        # all other modes
        else:
            previous_task = self.env.get_active_task(mode.prev_mode, None) 
            modes = self.get_mode_task_ids_of_home_pose_of_agent(mode.prev_mode)
            modes_new = []
            for r in self.env.robots:
                r_idx = self.env.robots.index(r)
                if r in previous_task.robots:
                    modes_new.append(mode.prev_mode.task_ids)
                else:
                    modes_new.append(modes[r_idx])
        return modes_new
    @cache
    def get_start_and_goal_config(self, mode:Mode):
        goal_config = []
        for robot in self.env.robots:
            goal_config.append(NpConfiguration.from_numpy(self.get_task_goal_of_agent(mode, robot)))
        start = self.get_home_poses(mode)
        start_config = []
        for s in start:
            start_config.append(NpConfiguration.from_numpy(s))
        return start_config, goal_config, self.get_mode_task_ids_of_home_pose_of_agent(mode)
    @cache
    def get_mode_task_ids_of_active_task(self, mode:Mode, task:Task):
        active_task = self.env.get_active_task(mode, None)
        if active_task == task:
            return mode.task_ids #as only task id is important and not conf as well
        for mode in self.modes:
            active_seq_idx = self.env.get_current_seq_index(mode)
            active_task = self.env.tasks[self.env.sequence[active_seq_idx]]
            if task == active_task:
               return mode.task_ids 

    def sample_transition_configuration(self, mode)-> Configuration:
        """Returns transition node of mode"""
        constrained_robot = self.env.get_active_task(mode, None).robots
        while True:
            q = []
            for robot in self.env.robots:
                if robot in constrained_robot:
                    q.append(self.get_task_goal_of_agent(mode, robot))
                else:
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
        constrained_robots = self.env.get_active_task(mode, None).robots
          
        while True:
            #goal sampling
            if is_goal_sampling and not self.operation.init_sol and self.config.planner == "rrtstar_par":
                if self.env.is_terminal_mode(mode):
                   return self.sample_transition_configuration(mode) 
                transition_nodes_id = transition_node_ids[mode]                  
                node_id = np.random.choice(transition_nodes_id)
                node = self.trees[self.modes[mode.id +1]].subtree.get(node_id)
                return node.state.q
            #goal sampling
            if is_goal_sampling and tree_order == -1: #TODO
                if mode.id== 0:
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
            
            if not is_informed_sampling and not is_gaussian_sampling:
                q = []
                if is_home_pose_sampling:
                    q_home = self.get_home_poses(mode)
                for robot in self.env.robots:
                    #home pose sampling
                    if is_home_pose_sampling:
                        r_idx = self.env.robots.index(robot)
                        if robot not in constrained_robots:
                            q.append(q_home[r_idx])
                            continue
                        if np.array_equal(self.get_task_goal_of_agent(mode, robot), q_home[r_idx]):
                            q.append(q_home[r_idx])
                            continue
                    #goal sampling
                    if is_goal_sampling: 
                        if self.config.general_goal_sampling or robot in constrained_robots:
                            q.append(self.get_task_goal_of_agent(mode, robot))
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
          
    def sample_informed(self, mode:Mode) -> None:
        """Returns: 
                Samples a point from the ellipsoidal subset defined by the start and goal positions and c_best.
        """
        q_rand = []
        update = False
        if self.operation.cost != self.informed[mode].cost:
            self.informed[mode].cost = self.operation.cost
            update = True
        if self.config.informed_sampling_version == 0 or self.config.informed_sampling_version == 3:
            for robot in self.env.robots:
                r_idx = self.env.robots.index(robot)
                r_indices = self.env.robot_idx[robot]
                if r_idx not in self.informed[mode].mode_task_ids_home_poses:
                    # if goal and start of task of robot is the same
                    q_rand.append(self.get_task_goal_of_agent(mode, robot))
                    continue
                if update:
                    if r_idx not in  self.informed[mode].mode_task_ids_task:
                        task = self.env.tasks[mode.task_ids[r_idx]]
                        self.informed[mode].mode_task_ids_task[r_idx] = self.get_mode_task_ids_of_active_task(mode, task)  
                    self.informed[mode].L[r_idx] = self.informed[mode].cholesky_decomposition(r_indices, r_idx ,self.operation.path_nodes)
                while True:
                    x_ball = self.sample_unit_n_ball(self.informed[mode].n[r_idx])  
                    x_rand = self.informed[mode].C[r_idx] @ (self.informed[mode].L[r_idx] @ x_ball) + self.informed[mode].center[r_idx]
                    # Check if x_rand is within limits
                    lims = self.env.limits[:, r_indices]
                    if np.all((lims[0] <= x_rand) & (x_rand <= lims[1])):  
                        q_rand.append(x_rand)
                        break
            return q_rand
        if self.config.informed_sampling_version == 1 : 
            if update:
                if self.informed[mode].mode_task_ids_task == []:
                    for robot in self.env.robots:
                        r_idx = self.env.robots.index(robot)
                        task = self.env.tasks[mode.task_ids[r_idx]]
                        self.informed[mode].mode_task_ids_task.append(self.get_mode_task_ids_of_active_task(mode, task))  
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
                    # self.SaveData(mode, time.time()-self.start, ellipse=q_ellipse)
                    return q_rand
        if self.config.informed_sampling_version == 2 or self.config.informed_sampling_version == 4 or self.config.informed_sampling_version == 5 or self.config.informed_sampling_version == 6 or self.config.informed_sampling_version == 7:
            if update:
                if self.informed[mode].mode_task_ids_task == []:
                    for robot in self.env.robots:
                        r_idx = self.env.robots.index(robot)
                        task = self.env.tasks[mode.task_ids[r_idx]]
                        self.informed[mode].mode_task_ids_task.append(self.get_mode_task_ids_of_active_task(mode, task))  
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
                    i= 0
                    q_ellipse = []
                    while i < 800:
                        x_ball = self.sample_unit_n_ball(self.informed[mode].n)  
                        x_rand = self.informed[mode].C @ (self.informed[mode].L @ x_ball) + self.informed[mode].center
                        q_ellipse.append(x_rand)
                        i+=1 
                    self.SaveData(mode, time.time()-self.start, ellipse=q_ellipse)
                    return q_rand
                
    def sample_unit_n_ball(self, n:int) -> torch.tensor:
        """Returns:
                Uniform sample from the volume of an n-ball of unit radius centred at origin
        """
        x_ball = np.random.normal(0, 1, n) 
        x_ball /= np.linalg.norm(x_ball, ord=2)     # Normalize with L2 norm
        radius = np.random.rand()**(1 / n)          # Generate a random radius and apply power
        return x_ball * radius

    def SaveData(self, mode:Mode, passed_time: time, n_new:NDArray=None, 
                 N_near:NDArray=None, r:float=None, n_rand:NDArray=None, 
                 n_nearest:NDArray = None, N_parent:NDArray=None, N_near_:List[NDArray]=None, ellipse:List[NDArray]=None) -> None:
        if self.config.debug_mode:
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
        if self.operation.init_sol and self.config.informed_sampling:
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
            "active_mode": mode.task_ids,
            "graph": graph_data_robot,
            "graph_transition": graph_data_transition,
            'ellipse': ellipse, 
            'mode': mode.task_ids
        }

        # Directory Handling: Ensure directory exists
        frames_directory = os.path.join(self.config.output_dir, 'FramesData')
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

    def Nearest(self, mode:Mode, q_rand: Configuration, tree: str = '') -> Node:
        set_dists = batch_config_dist(q_rand, self.trees[mode].get_batch_subtree(tree), self.config.dist_type)
        idx = np.argmin(set_dists)
        node_id = self.trees[mode].get_node_idx_subtree(tree)[idx]
        # print([float(set_dists[idx])])
        return  self.trees[mode].get_node(node_id, tree), set_dists[idx]
    
    def Steer(self, mode:Mode, n_nearest: Node, q_rand: Configuration, dist: torch.Tensor, i=1) -> State: 
        if np.equal(n_nearest.state.q.state(), q_rand.state()).all():
            return None
        q_nearest = n_nearest.state.q.state()
        q_rand = q_rand.state()
        direction = q_rand - q_nearest
        N = float((dist / self.config.step_size)) # to have exactly the step size

        if N <= 1 or int(N) == i-1:#for bidirectional or drrt
            q_new = q_rand
        else:
            q_new = q_nearest + (direction * (i /N))
        state_new = State(type(self.env.get_start_pos())(q_new, n_nearest.state.q.array_slice), mode)
        return state_new
    
    def Near(self, mode:Mode, n_new: Node):      

        #TODO generalize rewiring radius
        # n_nodes = sum(1 for _ in self.operation.current_mode.subtree.inorder()) + 1
        # r = min((7)*self.step_size, 3 + self.gamma * ((math.log(n_nodes) / n_nodes) ** (1 / self.dim)))
        batch_subtree = self.trees[mode].get_batch_subtree()
        set_dists = batch_config_dist(n_new.state.q, batch_subtree, self.config.dist_type)
        # set_dists = batch_dist_torch(n_new.q_tensor, n_new.state.q, batch_subtree, self.config.dist_type)
        indices = find_nearest_indices(set_dists, self.r) # indices of batch_subtree
        node_indices = self.trees[mode].node_idx_subtree[indices]
        n_near_costs = self.operation.costs[node_indices]
        N_near_batch = batch_subtree[indices]
        return N_near_batch, n_near_costs, node_indices
        # if not self.config.informed_sampling:
            
        # return self.fit_to_informed_subset(indices, N_near_batch, n_near_costs, node_indices) #TODO
    
    def FindParent(self, mode:Mode, node_indices: NDArray, n_new: Node, n_nearest: Node, batch_cost: NDArray, n_near_costs: torch.tensor) -> None:
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

    def UnitBallVolume(self) -> float:
        return math.pi ** (self.dim / 2) / math.gamma((self.dim / 2) + 1)
    
    def Rewire(self, mode:Mode,  node_indices: torch.Tensor, n_new: Node, batch_cost: torch.tensor, 
               n_near_costs: torch.tensor, n_rand = None, n_nearest = None) -> bool:
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
      
    def GeneratePath(self, mode:Mode, n: Node, shortcutting:bool = True) -> None:
        path_nodes, path = [], []
        while n:
            path_nodes.append(n)
            path.append(n.state)
            n = n.parent
        path_in_order = path[::-1]
        self.operation.path = path_in_order  
        self.operation.path_nodes = path_nodes[::-1]
        self.operation.cost = self.operation.path_nodes[-1].cost
        self.SaveData(mode, time.time()-self.start)
        self.costs.append(self.operation.cost)
        self.times.append(time.time()-self.start)
        if self.operation.init_sol and self.config.shortcutting and shortcutting:
            print(f"-- M", mode.task_ids, "Cost: ", self.operation.cost.item())
            self.Shortcutting(mode)
            
    def RandomMode(self, num_modes) -> List[float]:
        if num_modes == 1:
            return np.random.choice(self.modes)
        # if self.operation.task_sequence == [] and self.config.mode_probability != 0:
        elif self.operation.init_sol and self.config.mode_probability != 0:
                p = [1/num_modes] * num_modes
        
        elif self.config.mode_probability == 'None':
            # equally (= mode uniformly)
            return np.random.choice(self.modes)

        elif self.config.mode_probability == 1:
            # greedy (only latest mode is selected until initial paths are found and then it continues with equally)
            probability = [0] * (num_modes)
            probability[-1] = 1
            p =  probability

        elif self.config.mode_probability == 0:#TODO fix it for bidirectional
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
            total_nodes = Node.id_counter -1 + total_transition_nodes
            # Calculate probabilities inversely proportional to node counts
            inverse_probabilities = [
                1 - (len(mode.subtree) / total_nodes)
                for mode in self.modes[:-1]  # Exclude the last mode
            ]

            # Normalize the probabilities of all modes except the last one
            remaining_probability = 1-self.config.mode_probability  
            total_inverse = sum(inverse_probabilities)
            p =  [
                (inv_prob / total_inverse) * remaining_probability
                for inv_prob in inverse_probabilities
            ] + [self.config.mode_probability]

        return np.random.choice(self.modes, p = p)

    def InformedInitialization(self, mode: Mode) -> None: 
        if not self.config.informed_sampling:
            return
        if self.config.informed_sampling_version == 0 or self.config.informed_sampling_version == 3:
            self.informed[mode] = InformedVersion0(self.env, config_cost, self.config.cost_metric, self.config.cost_reduction)
            start_config, goal_config, mode_task_ids_home_poses = self.get_start_and_goal_config(mode)
            self.informed[mode].initialize(start_config, goal_config, mode_task_ids_home_poses)
        if self.config.informed_sampling_version == 1:
            self.informed[mode] = InformedVersion1(self.env, config_cost, self.config.cost_metric, self.config.cost_reduction)
            mode_task_ids_home_poses = self.get_mode_task_ids_of_home_pose_of_agent(mode)
            self.informed[mode].initialize(mode, mode_task_ids_home_poses)
        if self.config.informed_sampling_version == 2:
            self.informed[mode] = InformedVersion2(self.env, config_cost, self.config.cost_metric, self.config.cost_reduction)
            mode_task_ids_home_poses = self.get_mode_task_ids_of_home_pose_of_agent(mode)
            self.informed[mode].initialize(mode, mode_task_ids_home_poses)
        if self.config.informed_sampling_version == 4:
            self.informed[mode] = InformedVersion4(self.env, config_cost, self.config.cost_metric, self.config.cost_reduction)
            self.informed[mode].initialize(mode)
        if self.config.informed_sampling_version == 5:
            self.informed[mode] = InformedVersion5(self.env, config_cost, self.config.cost_metric, self.config.cost_reduction)
            mode_task_ids_home_poses = self.get_mode_task_ids_of_home_pose_of_agent(mode)
            self.informed[mode].initialize(mode, mode_task_ids_home_poses)
        if self.config.informed_sampling_version == 6:
            self.informed[mode] = InformedVersion6(self.env, config_cost, self.config.cost_metric, self.config.cost_reduction)
            mode_task_ids_home_poses = self.get_mode_task_ids_of_home_pose_of_agent(mode)
            self.informed[mode].initialize(mode, mode_task_ids_home_poses)
        if self.config.informed_sampling_version == 7:
            self.informed[mode] = InformedVersion7(self.env, config_cost, self.config.cost_metric, self.config.cost_reduction)
            mode_task_ids_home_poses = self.get_mode_task_ids_of_home_pose_of_agent(mode)
            self.informed[mode].initialize(mode, mode_task_ids_home_poses)

    def SampleNodeManifold(self, mode:Mode) -> Configuration:
        if np.random.uniform(0, 1) <= self.config.p_goal:
            #informed_sampling
            if self.config.informed_sampling and self.operation.init_sol: 
                if self.config.informed_sampling_version == 3 and np.random.uniform(0, 1) > self.config.p_uniform:
                    #uniform sampling
                    return self.sample_configuration(mode, 0)
                return self.sample_configuration(mode, 1)
            # gaussian sampling
            if self.config.gaussian and self.operation.init_sol: 
                return self.sample_configuration(mode, 4)
            # house pose sampling
            if self.config.p_stay != 1 and np.random.uniform(0, 1) > self.config.p_stay: 
                return self.sample_configuration(mode, 3)
            #uniform sampling
            return self.sample_configuration(mode, 0)
        # goal sampling
        return self.sample_configuration(mode, 2, self.transition_node_ids, self.trees[mode].order)
        
    def FindLBTransitionNode(self, iter: int) -> None:
        if self.operation.init_sol:      
            mode = self.modes[-1]
            result = self.get_lb_transition_node_id(mode) 
            if not result:
                return
            valid_mask = result[0] < self.operation.cost
            if valid_mask.any():
                lb_transition_node = self.get_transition_node(mode, result[1])
                self.GeneratePath(mode, lb_transition_node)
                print(f"{iter} M", mode.task_ids, "Cost: ", self.operation.cost.item())

    def UpdateDiscretizedCost(self, path , cost, idx):
        while True: 
            cost[idx] = cost[idx -1] + batch_config_cost([path[idx-1]], [path[idx]],metric = self.config.cost_metric, reduction=self.config.cost_reduction)
            if idx == len(path)-1:
                break
            idx+=1

    def Shortcutting(self,active_mode:Mode, version = 4, choice = 1, deterministic = False):
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
                    # dim = np.random.choice(range(env.robot_dims[env.robots[0]]))# TODO only feasible for same dimension across all robots
                if np.abs(i1-i2) < 2:
                    continue
                idx1 = min(i1, i2)
                idx2 = max(i1, i2)
                m1 = discretized_modes[idx1]    
                m2 = discretized_modes[idx2]    
                if m1 == m2 and choice == 0: #take all possible robots
                    robot = None
                    if version == 4:
                        dim = [np.random.choice(indices[r_idx]) for r_idx in range(len(self.env.robots))]
                    
                    if version == 5:
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

                    if version == 4:
                        dim = [np.random.choice(indices[robot])]
                    if version == 5:
                        all_indices = [i for i in indices[robot]]
                        num_indices = np.random.choice(range(len(indices[robot])))

                        random.shuffle(all_indices)
                        dim = all_indices[:num_indices]


                edge, edge_cost =  self.EdgeInterpolation(discretized_path[idx1:idx2+1].copy(), 
                                                            discretized_costs[idx1], discretized_modes[idx1:idx2+1], indices, dim, version, robot, self.env.robots)

                if edge_cost[-1] < discretized_costs[idx2] and self.env.is_path_collision_free(edge): #what when two different modes??? (possible for one task) 
                    discretized_path[idx1:idx2+1] = edge
                    discretized_costs[idx1:idx2+1] = edge_cost
                    self.UpdateDiscretizedCost(discretized_path, discretized_costs, idx2)

                    if not deterministic:
                        if np.abs(discretized_costs[-1] - termination_cost) > 0.001:
                            termination_cost = discretized_costs[-1]
                            termination_iter = j
                        elif np.abs(termination_iter -j) > 25000:
                            break
                    
        self.TreeExtension(active_mode, discretized_path, discretized_costs, discretized_modes)

    def TreeExtension(self, active_mode, discretized_path, discretized_costs, discretized_modes):
        mode_idx = 0
        mode = self.modes[mode_idx]
        parent = self.operation.path_nodes[0]
        for i in range(1, len(discretized_path) - 1):
            state = discretized_path[i]
            node = Node(state, self.operation)
            node.parent = parent
            self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, node.id)
            node.cost = discretized_costs[i][0]
            node.cost_to_parent = node.cost - node.parent.cost
            # agent_dist = batch_dist.to(dtype=torch.float16).cpu()
            # node.agent_dists = node.parent.agent_dists + agent_dist
            # node.agent_dists_to_parent = agent_dist
            parent.children.append(node)
            if self.trees[mode].order == 1:
                self.trees[mode].add_node(node)
            else:
                self.trees[mode].add_node(node, 'B')
            if len(discretized_modes[i]) > 1:
                self.convert_node_to_transition_node(mode, node)
                mode_idx += 1
                mode = self.modes[mode_idx]
            parent = node
        #Reuse terminal node (don't want to add a new one)
        self.operation.path_nodes[-1].parent.children.remove(self.operation.path_nodes[-1])
        parent.children.append(self.operation.path_nodes[-1])
        self.operation.path_nodes[-1].parent = parent
        self.operation.path_nodes[-1].cost = discretized_costs[-1][0]
        self.operation.path_nodes[-1].cost_to_parent = self.operation.path_nodes[-1].cost - self.operation.path_nodes[-1].parent.cost
        
        self.GeneratePath(active_mode, self.operation.path_nodes[-1], shortcutting =False)

    def EdgeInterpolation(self, path, cost, modes, indices, dim, version, r = None, robots =None):
        q0 = path[0].q.state()
        q1 = path[-1].q.state()
        edge  = []
        edge_cost = [cost]
        segment_vector = q1 - q0
        # dim_indices = [indices[i][dim] for i in range(len(indices))]
        N = len(path) -1
        for i in range(len(path)):
            mode = modes[i][0]
            if version == 0 :
                q = q0 +  (segment_vector * (i / N))

            elif version == 3: #shortcutting agent
                q = path[i].q.state()
                for robot in range(len(robots)):
                    if r is not None and r == robot:
                        q[indices[robot]] = q0[indices[robot]] +  (segment_vector[indices[robot]] * (i /N))
                        break
                    if r is None:
                        q[indices[robot]] = q0[indices[robot]] +  (segment_vector[indices[robot]] * (i / N))
                    
            elif version == 1:
                q = path[i].q.state()
                q[dim] = q0[dim] + ((q1[dim] - q0[dim])* (i / N))

            elif version == 4: #partial shortcutting agent single dim 
                q = path[i].q.state().copy()
                for robot in range(len(robots)):
                    if r is not None and r == robot:
                        q[dim] = q0[dim] +  (segment_vector[dim] * (i / N))
                        break
                    if r is None:
                        q[dim[robot]] = q0[dim[robot]] +  (segment_vector[dim[robot]] * (i / N))

            elif version == 2:
                q = path[i].q.state()
                for idx in dim:
                    q[idx] = q0[idx] + ((q1[idx] - q0[idx])* (i / N))
            
            elif version == 5: #partial shortcutting agent random set of dim 
                q = path[i].q.state()
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
            edge_cost.append(edge_cost[-1] + batch_config_cost([edge[-2]], [edge[-1]], metric = self.config.cost_metric, reduction=self.config.cost_reduction))

        return edge, edge_cost

    def Discretization(self, path, indices, resolution=0.1):
        discretized_path, discretized_modes, discretized_costs = [], [], []
        discretized_costs.append(path[0].cost.item())
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

                    discretized_costs.append(discretized_costs[-1] + batch_config_cost([discretized_path[-2]], [discretized_path[-1]], metric = self.config.cost_metric, reduction=self.config.cost_reduction))
                else:
                    original_mode = path[i+1].state.mode
                    if j != num_points-1:
                        interpolated_point = start.state() + (segment_vector * (j / (num_points -1)))
                        q_list = [interpolated_point[indices[i]] for i in range(len(indices))]
                        discretized_path.append(State(NpConfiguration.from_list(q_list), original_mode))
                        discretized_modes.append([mode[-1]])
                        discretized_costs.append(discretized_costs[-1] + batch_config_cost([discretized_path[-2]], [discretized_path[-1]], metric = self.config.cost_metric, reduction=self.config.cost_reduction))
                
        discretized_modes.append([path[-1].state.mode.task_ids])
        discretized_path.append(path[-1].state)
        discretized_costs.append(discretized_costs[-1] + batch_config_cost([discretized_path[-2]], [discretized_path[-1]], metric = self.config.cost_metric, reduction=self.config.cost_reduction))
        
        return discretized_path, discretized_modes, discretized_costs
         
    def PTC(self, iter:int):
        if iter% 1000 == 0:
            if check_gpu_memory_usage():
                return True
        if iter% 100000 == 0:
            print(iter)
        if time.time()-self.start >= self.config.ptc_time:
            print('Finished after ', np.round(time.time()-self.start, 1),'s' )
            return True

    def SaveFinalData(self) -> None:
        self.costs.append(self.operation.cost)
        self.times.append(time.time()-self.start)
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
            frames_directory = os.path.join(self.config.output_dir, 'FramesFinalData')
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
