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
import copy
import time as time
import math as math
from typing import Tuple, Optional, Union
from functools import cache


class ConfigManager:
    def __init__(self, config_file:str):
        # Load configuration
        config_file_path = os.path.join(os.path.dirname(__file__), "../../../examples/")+ config_file
        print(config_file_path)
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Set defaults and dynamically assign attributes
        defaults = {
            'planner': None, 'p_goal': 0.95, 'p_stay' : 0.95,'general_goal_sampling': True, 
            'step_size': 0.3, 
            'ptc_time': 600, 'mode_sampling': 0.4, 
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
        self.costs = torch.empty(
            1000000, device=device, dtype=torch.float32
        )
        self.paths_inter = []
    
    def get_cost(self, idx):
        """Return cost of node with specific id"""
        return self.costs[idx]

class Node:
    id_counter = 0

    def __init__(self, state:State, operation: Operation):
        self.state = state   
        self.q_tensor = torch.tensor(state.q.state(), device=device, dtype=torch.float64)
        self.parent = None  
        self.children = []    
        self.transition = False
        self.num_agents = state.q.num_agents()
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

    def _resize_tensor(self, tensor: torch.Tensor, current_capacity: int, new_capacity: int) -> None:
        """
        Dynamically resizes the given tensor to the specified new capacity.

        Args:
            tensor (torch.Tensor): The tensor to resize.
            current_capacity (int): The current capacity of the tensor.
            new_capacity (int): The new capacity to allocate.

        Returns:
            torch.Tensor: The resized tensor.
        """
        new_tensor = torch.empty(
            (new_capacity, *tensor.shape[1:]),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        new_tensor[:current_capacity] = tensor
        del tensor  # Free old tensor memory
        torch.cuda.empty_cache() 
        return new_tensor

    def ensure_capacity(self, tensor: torch.Tensor, required_capacity:int) -> None:
        """
        Ensures that the tensor has enough capacity to add new elements. Resizes if necessary.

        Args:
            tensor (torch.Tensor): The tensor to check and potentially resize.
            capacity_attr (str): The name of the capacity attribute associated with the tensor.
            new_elements (int): The number of new elements to accommodate.

        Returns:
            torch.Tensor: The tensor with ensured capacity.
        """
        current_size = tensor.shape[0]

        if required_capacity == current_size:
            required_capacity
            return self._resize_tensor(tensor, current_size, required_capacity*2)
        return tensor
    
    @abstractmethod
    def add_node(self, n:Node, tree:str) -> None:
        pass
    @abstractmethod
    def remove_node(self, n:Node, tree:str = '') -> None:
        pass
    @abstractmethod
    def get_batch_subtree(self, tree:str = '') -> torch.Tensor:
        pass
    @abstractmethod
    def get_node_idx_subtree(self, tree:str = '') -> torch.Tensor:
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
        self.informed = Informed()
        robot_dims = sum(env.robot_dims.values())
        self.subtree = {}
        self.initial_capacity = 100000
        self.batch_subtree = torch.empty(
            (self.initial_capacity, robot_dims), device=device, dtype=torch.float32
        )
        self.node_idx_subtree = torch.empty(
            self.initial_capacity, device=device, dtype=torch.long
        )
    def add_node(self, n:Node, tree:str = '') -> None:
        self.subtree[n.id] = n               
        position = len(self.subtree) -1
        self.batch_subtree = self.ensure_capacity(self.batch_subtree, position)
        self.batch_subtree[position,:] = n.q_tensor
        self.node_idx_subtree = self.ensure_capacity(self.node_idx_subtree, position)
        self.node_idx_subtree[position] = n.id
    
    def remove_node(self, n:Node, tree:str = '') -> None:
        mask = self.node_idx_subtree != n.id
        self.node_idx_subtree = self.node_idx_subtree[mask] 
        self.batch_subtree = self.batch_subtree[mask]
        del self.subtree[n.id]

    def get_batch_subtree(self, tree:str = '') -> torch.Tensor:
        return self.batch_subtree[:len(self.subtree)]
    
    def get_node_idx_subtree(self, tree:str = '') -> torch.Tensor:
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
        self.informed = Informed()
        robot_dims = sum(env.robot_dims.values())
        self.subtree = {}
        self.initial_capacity = 100000
        self.batch_subtree = torch.empty(
            (self.initial_capacity, robot_dims), device=device, dtype=torch.float32
        )
        self.node_idx_subtree = torch.empty(
            self.initial_capacity, device=device, dtype=torch.long
        )
        self.subtree_b = {}  # This remains the same
        self.batch_subtree_b = torch.empty(
            (self.initial_capacity, robot_dims), device=device, dtype=torch.float32
        )
        self.node_idx_subtree_b = torch.empty(
            self.initial_capacity, device=device, dtype=torch.long
        )
        self.connected = False
    
    def add_node(self, n:Node, tree:str = '') -> None:
        if tree == 'A' or tree == '':
            self.subtree[n.id] = n               
            position = len(self.subtree) -1
            self.batch_subtree = self.ensure_capacity(self.batch_subtree, position)
            self.batch_subtree[position,:] = n.q_tensor
            self.node_idx_subtree = self.ensure_capacity(self.node_idx_subtree, position)
            self.node_idx_subtree[position] = n.id
        if tree == 'B':
            self.subtree_b[n.id] = n              
            position = len(self.subtree_b) -1
            self.batch_subtree_b = self.ensure_capacity(self.batch_subtree_b, position)
            self.batch_subtree_b[position,:] = n.q_tensor
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


    def get_batch_subtree(self, tree:str = '') -> torch.Tensor:
        if tree == 'A' or tree == '':
            return self.batch_subtree[:len(self.subtree)]
        if tree == 'B':
            return self.batch_subtree_b[:len(self.subtree_b)]
    
    def get_node_idx_subtree(self, tree:str = '') -> torch.Tensor:
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

class Informed:

    def __init__(self):
        self.C = {}
        self.inv_C = {}
        self.L = {}
        self.cmin = {}
        self.state_centre = {}
        self.start = {}
        self.goal = {}

    def cmax(self, transition_nodes:List[Node], path_nodes:List[Node], 
             goal_radius:float, r_idx:int, start:NDArray) -> Tuple[torch.tensor,torch.tensor, torch.tensor]:
        if transition_nodes == []:
            return
        else:
            for node in transition_nodes:
                if node in path_nodes:
                    c_agent = node.agent_dists[:,r_idx]
                    c_tot = node.cost
                    break
            c_start_agent, c_start_tot = self.get_start_node(path_nodes, goal_radius, r_idx, start)
            return c_tot, c_start_tot, c_agent - c_start_agent

    def get_start_node(self, path_nodes:List[Node], goal_radius:float, r_idx:int, start:NDArray) -> Tuple[torch.tensor, torch.tensor]:
        for node in path_nodes:
            if np.array_equal(node.state.q[r_idx], start) or np.linalg.norm(start - node.state.q[r_idx]) < goal_radius:
                return node.agent_dists[:,r_idx], node.cost 

class BaseRRTstar(ABC):
    def __init__(self, env, config: ConfigManager):
        self.env = env
        # self.gamma = ((2 *(1 + 1/self.dim))**(1/self.dim) * (self.FreeSpace()/self.UnitBallVolume())**(1/self.dim))*1.1 
        self.config = config
        self.r = self.config.step_size * sum(self.env.robot_dims.values())
        # self.r = self.config.step_size
        self.operation = Operation()
        self.start = time.time()
        self.start_single_goal= SingleGoal(self.env.start_pos.q)
        self.modes = [] 
        self.trees = {}
        self.transition_node_ids = {}
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

    def get_lb_transition_node_id(self, mode:Mode) -> Tuple[torch.Tensor, int]:
        """Returns lb cost and index of transition nodes of the mode"""
        cost, idx = self.operation.costs[self.transition_node_ids[mode]].min(0)
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

    @cache
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
                    goal = previous_task.goal.goal
                    if len(goal) == self.env.robot_dims[r]:
                        q_new.append(goal)
                    else:
                        q_new.append(goal[r_idx])
                else:
                    q_new.append(q[r_idx])
        return q_new
    
    @cache 
    def get_task_goal_of_agent(self, mode:Mode, r:str):
        """Returns task goal of agent in current mode"""
        task = self.env.get_active_task(mode, None)
        if r not in task.robots:
            r_idx = self.env.robots.index(r)
            return self.env.tasks[mode.task_ids[r_idx]].goal.sample(mode)
        goal = task.goal.goal
        if len(goal) == self.env.robot_dims[r]:
           return goal
        else:
            return goal[self.env.robot_idx[r]]    
        
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
            else:
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
                    #informed sampling
                    if is_informed_sampling: 
                        r_idx = self.env.robots.index(robot)
                        qr = self.sample_informed(r_idx, self.env.collision_tolerance)
                        if qr is not None:
                            q.append(qr)
                        else:
                            lims = self.operation.limits[robot]
                            q.append(np.random.uniform(lims[0], lims[1]))
                        continue

                    #uniform sampling
                    lims = self.env.limits[:, self.env.robot_idx[robot]]
                    q.append(np.random.uniform(lims[0], lims[1]))
            q = type(self.env.get_start_pos()).from_list(q)
            if self.env.is_collision_free(q, mode):
                return q

    def mode_selection(self, mode:Mode, r_idx:int): #TODO 
        if mode.__eq__(self.env.terminal_mode):
        # if self.operation.active_mode.label == self.env.terminal_mode:
            if not self.operation.init_sol:
                return
            # return self.operation.active_mode
            return mode
        # if self.operation.active_mode == self.operation.modes[-1]:
        if mode.__eq__(self.modes[-1]):
            return
        # task = self.operation.active_mode.label[r_idx]
        task = mode.task_ids[r_idx]
        for mode in self.modes:
            active_seq_idx = self.env.get_current_seq_index(mode.label)
            active_task = self.env.sequence[active_seq_idx]
            if task == active_task:
                if mode.label != self.operation.modes[-1].label:
                    return mode
                return
        return

    def sample_informed(self, r_idx:int, goal_radius:float) -> None:
        """Samples a point from the ellipsoidal subset defined by the start and goal positions and c_best."""
        mode = self.mode_selection(r_idx)
        agent_name = self.env.robots[r_idx]
        if not mode:
            return  # Initial path not yet found
        if not self.operation.init_sol and (self.config.informed_sampling_version == 1 or self.config.informed_sampling_version == 2):
            return
        c_tot, c_start_tot, cmax =  mode.informed.cmax(mode.transition_nodes, self.operation.path_nodes, goal_radius, r_idx, mode.informed.start[r_idx]) # reach task for all agents, reach goal of task for specific agent
        if self.config.informed_sampling_version == 0:
            cmax = cmax
            cmin = mode.informed.cmin[r_idx]
        elif self.config.informed_sampling_version == 1:
            cmax = self.operation.path_nodes[-1].cost
            cmin = mode.informed.cmin[r_idx] + self.operation.path_nodes[-1].cost -c_tot + c_start_tot
        elif self.config.informed_sampling_version == 2:
            cmax = self.operation.path_nodes[-1].cost
            indices = self.env.robot_idx[agent_name]
            start = copy.deepcopy(self.operation.path_nodes[0])
            cmin = 0
            m = self.env.start_mode
            active_robots = self.env.get_active_task(m).robots
            idx = []
            for r in active_robots:
                idx.extend( self.env.robot_idx[r])
            
            for node in self.operation.path_nodes:
                if node.transition:
                    if np.equal(start.state.q.state()[indices], mode.informed.start[r_idx]).all() and np.equal(node.state.q.state()[indices], mode.informed.goal[r_idx]).all():
                        cmin += mode.informed.cmin[r_idx]
                        start.state.q.state()[indices] = node.state.q.state()[indices]
                        m = self.env.get_next_mode(None, m)
                        active_robots = self.env.get_active_task(m).robots
                        idx = []
                        for r in active_robots:
                            idx.extend( self.env.robot_idx[r])

                    else:
                        cmin += config_cost(start.state.q, node.state.q, self.config.cost_type)
                        start.state.q.state()[idx] = node.state.q.state()[idx]
                        m = self.env.get_next_mode(None, m)
                        active_robots = self.env.get_active_task(m).robots
                        idx = []
                        for r in active_robots:
                            idx.extend( self.env.robot_idx[r])

        r1 = cmax / 2
        r2 = torch.sqrt(cmax**2 - cmin**2) / 2

        
        n = self.env.robot_dims[agent_name]
        mode.informed.L[r_idx] = torch.diag(torch.cat([r1.repeat(1), r2.repeat(n - 1)]))

        lims = torch.as_tensor(
            self.operation.limits[agent_name],
            device=cmax.device,
            dtype=cmax.dtype
        )
        L = mode.informed.L[r_idx]
        C = mode.informed.C[r_idx]
        centre = mode.informed.state_centre[r_idx]
        while True:
            x_ball = self.sample_unit_n_ball(n)  # Assumes this returns a tensor on the correct device
            x_rand = C @ (L @ x_ball) + centre
            # Check if x_rand is within limits
            if torch.all((lims[0] <= x_rand) & (x_rand <= lims[1])):
                return x_rand.cpu().numpy()

    def rotation_to_world_frame(self, x_start:NDArray, x_goal:NDArray, robot:str) -> Tuple[float, NDArray]:
        """Returns the norm and rotation matrix C from the hyperellipsoid-aligned frame to the world frame."""
        diff = x_goal - x_start
        norm = np.linalg.norm(diff)
        a1 = diff / norm  # Unit vector along the transverse axis

        # Create first column of the identity matrix
        n = self.env.robot_dims[robot]
        e1 = np.zeros(n)
        e1[0] = 1

        # Compute SVD directly on the outer product
        U, _, Vt = np.linalg.svd(np.outer(a1, e1))
        V = Vt.T

        # Construct the rotation matrix C
        det_factor = np.linalg.det(U) * np.linalg.det(V)
        C = U @ np.diag([1] * (n - 1) + [det_factor]) @ Vt

        return norm, C

    def sample_unit_n_ball(self, n:int) -> torch.tensor:
        """Returns uniform sample from the volume of an n-ball of unit radius centred at origin"""
        x_ball = torch.randn(n, device=device, dtype=torch.float64)
        x_ball /= torch.linalg.norm(x_ball, ord=2)
        radius = torch.rand(1, device=device, dtype=torch.float64).pow(1 / n)
        return x_ball * radius 

    def fit_to_informed_subset(self, N_near_indices: torch.Tensor, N_near_batch: torch.Tensor, n_near_costs: torch.Tensor, node_indices:torch.Tensor) -> List[Node]:
        N_near_size = N_near_batch.size(0)
        if  N_near_size == 0:
            return N_near_indices, N_near_batch, n_near_costs, node_indices

        final_valid_mask = torch.ones(N_near_size, device=N_near_batch.device, dtype=torch.bool)  # Start with all True

        for r_idx, r in enumerate(self.env.robots):
            mode = self.mode_selection(r_idx)
            if not mode or r_idx not in mode.informed.goal or np.array_equal(mode.informed.goal[r_idx], mode.informed.start[r_idx]):
                continue
            if mode.informed.L:
                indices = self.env.robot_idx[r]
                centre = mode.informed.state_centre[r_idx]
                L = mode.informed.L[r_idx]
                inv_C = mode.informed.inv_C[r_idx]

                inv_L = torch.linalg.inv(L)
                combined_transform = torch.matmul(inv_L, inv_C)

                diff = N_near_batch[:, indices] - centre  # Shape: (num_nodes, len(indices))
                scaled_nodes = torch.matmul(diff, combined_transform.T)  # Shape: (num_nodes, len(indices))

                squared_norms = torch.sum(scaled_nodes**2, dim=1)  # Shape: (num_nodes,)
                valid_mask = squared_norms <= 1

                final_valid_mask &= valid_mask

        if final_valid_mask.all():
            return N_near_indices, N_near_batch, n_near_costs

        valid_indices = final_valid_mask.nonzero(as_tuple=True)[0]
        return N_near_indices[valid_indices], N_near_batch[valid_indices], n_near_costs[valid_indices], node_indices[valid_indices]

    def SaveData(self, mode:Mode, passed_time: time, n_new:NDArray=None, 
                 N_near:torch.tensor=None, r:float=None, n_rand:NDArray=None, 
                 n_nearest:NDArray = None, N_parent:torch.tensor=None, N_near_:List[NDArray]=None) -> None:
        if self.config.debug_mode:
            return
        #Tree data
        tree_data = []
        for mode in self.modes:
            m = m = mode.task_ids
            subtree_data = [
                {
                    "state": node.state.q.state(),
                    "parent": node.parent.state.q.state() if node.parent else None,
                    "mode": m,
                }
                for node in self.trees[mode].subtree.values()
            ]
            tree_data.extend(subtree_data)
        try: 
            self.trees[mode].subtree_b
            for mode in self.modes:
                m = mode.task_ids
                subtree_data = [
                    {
                        "state": node.state.q.state(),
                        "parent": node.parent.state.q.state() if node.parent else None,
                        "mode": m,
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

            intermediate_tot = torch.tensor(
                [node.cost for node in self.operation.path_nodes],
                device=device,
                dtype=torch.float32
            )
            # intermediate_agent_dists = torch.cat([node.agent_dists for node in self.operation.path_nodes])]
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
        # informed_sampling = [
        #     {
        #         "C": mode.informed.C,
        #         "L": mode.informed.L,
        #         "center": mode.informed.state_centre,
        #         "start": mode.informed.start,
        #         "goal": mode.informed.goal,
        #     }
        #     for mode in self.operation.modes if mode.informed.L
        # ]

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
            "informed_sampling": None,
            "n_rand": n_rand,
            "N_near": N_near_list,
            "N_parent": N_parent_list,
            "rewire_r": r,
            "n_new": n_new,
            "n_nearest": n_nearest,
            "active_mode": mode.task_ids,
            "graph": graph_data_robot,
            "graph_transition": graph_data_transition
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
        # print([float(q) for q in q_rand.state()])
        q_tensor = torch.as_tensor(q_rand.state(), device=device, dtype=torch.float64).unsqueeze(0)
        # print([q.item() for q in q_tensor[0]])
        set_dists = batch_dist_torch(q_tensor, q_rand, self.trees[mode].get_batch_subtree(tree), self.config.dist_type).clone()
        # print([n for n in self.trees[mode].get_batch_subtree(tree)])
        # print([d.item() for d in set_dists])
        idx = torch.argmin(set_dists).item()
        node_id = self.trees[mode].get_node_idx_subtree(tree)[idx]
        # print([set_dists[idx].item()])
        return  self.trees[mode].get_node(node_id.item(), tree), set_dists[idx]
   
    def Steer(self, mode:Mode, n_nearest: Node, q_rand: Configuration, dist: torch.Tensor, i=1) -> State: 
        if np.equal(n_nearest.state.q.state(), q_rand.state()).all():
            return None
        q_nearest = n_nearest.state.q.state()
        q_rand = q_rand.state()
        direction = q_rand - q_nearest
        N = (dist / self.config.step_size).item() # to have exactly the step size

        if N <= 1 or int(N) == i-1:#for bidirectional or drrt
            q_new = q_rand
        else:
            q_new = q_nearest + (direction * (i /N))
        state_new = State(type(self.env.get_start_pos())(q_new, n_nearest.state.q.array_slice), mode)
        # print(dist.item())
        # print([float(s) for s in state_new.q.state()])
        return state_new
  
    def Near(self, mode:Mode, n_new: Node) -> Tuple[List[Node], torch.tensor]:
        #TODO generalize rewiring radius
        # n_nodes = sum(1 for _ in self.operation.current_mode.subtree.inorder()) + 1
        # r = min((7)*self.step_size, 3 + self.gamma * ((math.log(n_nodes) / n_nodes) ** (1 / self.dim)))
        batch_subtree = self.trees[mode].get_batch_subtree()
        set_dists = batch_dist_torch(n_new.q_tensor, n_new.state.q, batch_subtree, self.config.dist_type).clone()
        indices = torch.where(set_dists < self.r)[0] # indices of batch_subtree
        N_near_batch = batch_subtree.index_select(0, indices)
        node_indices = self.trees[mode].node_idx_subtree.index_select(0,indices) # actual node indices (node.id)
        n_near_costs = self.operation.costs.index_select(0,node_indices)
        return N_near_batch, n_near_costs, node_indices
        
    def FindParent(self, mode:Mode, node_indices: torch.tensor, n_new: Node, n_nearest: Node, batch_cost: torch.tensor,n_near_costs: torch.tensor) -> None:
        idx = torch.where(node_indices == n_nearest.id)[0] 
        c_new_tensor = n_near_costs + batch_cost
        c_min = c_new_tensor[idx]
        c_min_to_parent = batch_cost[idx]
        n_min = n_nearest
        valid_mask = c_new_tensor < c_min[0]
        if torch.any(valid_mask):
            sorted_indices = torch.nonzero(valid_mask, as_tuple=True)[0][torch.sort(c_new_tensor[valid_mask], stable=True)[1]]
            for idx in sorted_indices:
                node = self.trees[mode].subtree.get(node_indices[idx].item())
                if self.env.is_edge_collision_free(node.state.q, n_new.state.q, mode):
                    c_min = c_new_tensor[idx]
                    c_min_to_parent = batch_cost[idx]       # Update minimum cost
                    n_min = node                            # Update nearest node
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
        
        if torch.any(improvement_mask):
            # self.SaveData(time.time()-self.start, n_new = n_new.state.q.state(), N_near = N_near, 
            #                   r =self.r, n_rand = n_rand, n_nearest = n_nearest) 
            improved_indices = improvement_mask.nonzero(as_tuple=True)[0]

            for idx in improved_indices:
                n_near = self.trees[mode].subtree.get(node_indices[idx].item())
                # n_near = N_near[idx]
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
        self.operation.cost = self.operation.path_nodes[-1].cost.clone()
        self.SaveData(mode, time.time()-self.start)
        self.costs.append(self.operation.cost)
        self.times.append(time.time()-self.start)
        if self.operation.init_sol and self.config.shortcutting and shortcutting:
            print(f"-- M", mode.task_ids, "Cost: ", self.operation.cost.item())
            self.Shortcutting(mode)
            
    def RandomMode(self, num_modes) -> List[float]:
        if num_modes == 1:
            return np.random.choice(self.modes)
        # if self.operation.task_sequence == [] and self.config.mode_sampling != 0:
        elif self.operation.init_sol and self.config.mode_sampling != 0:
                p = [1/num_modes] * num_modes
        
        elif self.config.mode_sampling == 'None':
            # equally (= mode uniformly)
            return np.random.choice(self.modes)

        elif self.config.mode_sampling == 1:
            # greedy (only latest mode is selected until all initial paths are found and then it continues with equally)
            probability = [0] * (num_modes)
            probability[-1] = 1
            p =  probability

        elif self.config.mode_sampling == 0:#TODO fix it for bidirectional
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
            remaining_probability = 1-self.config.mode_sampling  
            total_inverse = sum(inverse_probabilities)
            p =  [
                (inv_prob / total_inverse) * remaining_probability
                for inv_prob in inverse_probabilities
            ] + [self.config.mode_sampling]

        return np.random.choice(self.modes, p = p)

    def InformedInitialization(self, new_mode: Mode) -> None:
        if not self.config.informed_sampling:
            return
        task = self.env.get_active_task(new_mode, None)
        #Only looking at constrained robots
        for robot in task.robots: 
            r = self.env.robots.index(robot)
            q_home = self.get_home_poses(new_mode)
            goal = self.get_task_goal_of_agent(new_mode, robot)
            if not np.equal(goal, q_home[r]).all():
                cmin, C = self.rotation_to_world_frame(q_home[r], goal ,robot)
                C = torch.tensor(C, device = device, dtype=torch.float32)
                self.informed[new_mode].C[r] = C
                self.informed[new_mode].inv_C[r] = torch.linalg.inv(C)
                self.informed[new_mode].cmin[r] = torch.tensor(cmin-2*self.env.collision_tolerance, device= device, dtype=torch.float32)
                self.informed[new_mode].state_centre[r] = torch.tensor(((q_home[r] + goal)/2), device=device, dtype=torch.float32)

    def SampleNodeManifold(self, mode:Mode) -> Configuration:
        if np.random.uniform(0, 1) <= self.config.p_goal:
            # house pose sampling
            if self.config.p_stay != 1 and np.random.uniform(0, 1) > self.config.p_stay: 
                return self.sample_configuration(mode, 3)
            # informed sampling
            if self.config.informed_sampling: 
                # informed sampling cannot be applied as there hasn't been found an initial solution in this mode
                if not self.operation.init_sol and mode.__eq__(self.modes[-1]): 
                    #uniform sampling
                    return self.sample_configuration(mode, 0)
                return self.sample_configuration(mode, 1)
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
            cost[idx] = cost[idx -1] + batch_config_cost([path[idx-1]], [path[idx]], metric = "euclidean", reduction="max")
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
            node.cost = discretized_costs[i][0].item()
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
        self.operation.path_nodes[-1].cost = discretized_costs[-1][0].item()
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
            edge_cost.append(edge_cost[-1] + batch_config_cost([edge[-2]], [edge[-1]], metric = "euclidean", reduction="max"))

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

                    discretized_costs.append(discretized_costs[-1] + batch_config_cost([discretized_path[-2]], [discretized_path[-1]], metric = "euclidean", reduction="max"))
                else:
                    original_mode = path[i+1].state.mode
                    if j != num_points-1:
                        interpolated_point = start.state() + (segment_vector * (j / (num_points -1)))
                        q_list = [interpolated_point[indices[i]] for i in range(len(indices))]
                        discretized_path.append(State(NpConfiguration.from_list(q_list), original_mode))
                        discretized_modes.append([mode[-1]])
                        discretized_costs.append(discretized_costs[-1] + batch_config_cost([discretized_path[-2]], [discretized_path[-1]], metric = "euclidean", reduction="max"))
                
        discretized_modes.append([path[-1].state.mode.task_ids])
        discretized_path.append(path[-1].state)
        discretized_costs.append(discretized_costs[-1] + batch_config_cost([discretized_path[-2]], [discretized_path[-1]], metric = "euclidean", reduction="max"))
        
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

                intermediate_tot = torch.tensor(
                    [node.cost for node in self.operation.path_nodes],
                    device=device,
                    dtype=torch.float32
                )
                # intermediate_agent_dists = torch.cat([node.agent_dists for node in self.operation.path_nodes])
                intermediate_agent_dists = None
                result = {
                    "path": path_data,
                    "total": self.operation.path_nodes[-1].cost,
                    "intermediate_tot": intermediate_tot,
                    "is_transition": [node.transition for node in self.operation.path_nodes],
                    "modes": [node.state.mode.task_ids for node in self.operation.path_nodes],
                }
            else:
                result = {
                    "path": None,
                    "total": cost,
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
