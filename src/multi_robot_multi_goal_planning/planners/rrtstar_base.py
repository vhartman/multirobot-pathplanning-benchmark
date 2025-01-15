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
from typing import Tuple, Optional

torch.cuda.manual_seed_all(10)

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
            'step_size': 0.3, 'ptc_threshold': 0.1, 
            'ptc_max_iter': 3000, 'mode_probability': 0.4, 
            'informed_sampling': True, 'informed_sampling_version': 0, 
            'cprofiler': False, 'cost_type': 'euclidean', 'dist_type': 'euclidean', 
            'debug_mode': False, 'transition_nodes': 100, 'birrtstar_version' :1, 
            'amount_of_runs' : 1, 'use_seed' : True, 'seed': 1, 'depth': 1,  
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

class Mode:
    """Mode description"""
    def __init__(self, mode, env):
        self.label = mode
        self.transition_nodes = []
        robot_dims = sum(env.robot_dims.values())
        self.subtree = []
        self.initial_capacity = 100000
        self.batch_subtree = torch.empty(
            (self.initial_capacity, robot_dims), device=device, dtype=torch.float32
        )
        self.node_idx_subtree = torch.empty(
            self.initial_capacity, device=device, dtype=torch.long
        )
        self.subtree_b = []  # This remains the same
        self.batch_subtree_b = torch.empty(
            (self.initial_capacity, robot_dims), device=device, dtype=torch.float32
        )
        self.node_idx_subtree_b = torch.empty(
            self.initial_capacity, device=device, dtype=torch.long
        )
        self.informed = Informed()
        self.order = 1
        self.connected = False
        self.constrained_robots = env.get_active_task(self.label).robots
        self.indices = [idx for r in self.constrained_robots for idx in env.robot_idx[r]]
        self.starts = {}

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

class Operation:
    """ Planner operation variables"""
    def __init__(self, env):
        self.modes = [Mode(env.start_mode, env)] # list of visited modes
        self.active_mode = None
        self.path = []
        self.path_nodes = None
        self.cost = np.inf
        self.ptc_cost = 0 # needed for termination
        self.ptc_iter = None # needed for termination
        self.init_sol = False
        self.costs = torch.empty(
            1000000, device=device, dtype=torch.float32
        )
        # self.costs = torch.tensor([0], device=device, dtype=torch.float32)
        self.limits = {robot: env.limits[:, env.robot_idx[robot]] for robot in env.robots}
        self.paths_inter = []
        self.start_node = None
    
    def get_cost(self, idx):
        """
        Retrieve the cost for a specific index.
        """
        return self.costs[idx]
    
class Node:
    id_counter = 0

    def __init__(self, state:State, operation: Operation):
        self.state = state   
        self.q_tensor = torch.tensor(state.q.state(), device=device, dtype=torch.float32)
        self.parent = None  
        self.children = []    
        self.transition = False
        self.num_agents = state.q.num_agents()
        self.agent_dists = torch.zeros(1, self.num_agents, device = 'cpu', dtype=torch.float32)
        self.cost_to_parent = None
        self.agent_dists_to_parent = torch.zeros(1, self.num_agents, device = 'cpu', dtype=torch.float32)
        self.operation = operation
        self.idx = Node.id_counter
        Node.id_counter += 1

    @property
    def cost(self):
        return self.operation.get_cost(self.idx)
    
    @cost.setter
    def cost(self, value):
        """Set the cost in the shared operation costs tensor."""
        self.operation.costs[self.idx] = value

    def __repr__(self):
        return f"<N- {self.state.q.state()}, c: {self.cost}>"

class Sampling:
    def __init__(self, env: base_env, operation: Operation, config: ConfigManager):
        self.env = env
        self.operation = operation
        self.config = config
    
    def get_goal_config_of_mode(self, mode:Mode)-> Configuration:
        q = np.zeros(sum(self.env.robot_dims.values()))
        for robot in self.env.robots:
            indices = self.env.robot_idx[robot]
            if indices[0] not in mode.indices:
                lims = self.operation.limits[robot]
                q[indices] = np.random.uniform(lims[0], lims[1])
        q[mode.indices] = self.env.get_active_task(mode.label).goal.goal 
        return q   
    
    def sample_state(self, mode: Mode, sampling_type: int) -> Configuration:
        m = mode.label
        is_goal_sampling = sampling_type == 2
        is_informed_sampling = sampling_type == 1
        is_home_pose_sampling = sampling_type == 3
          
        while True:
            if is_goal_sampling and not self.operation.init_sol and self.config.planner == "rrtstar_par":
                transition_nodes = self.operation.active_mode.transition_nodes
                node = np.random.choice(transition_nodes)
                return node.state.q

            if is_goal_sampling and mode.order == -1:
                mode_idx = self.operation.modes.index(self.operation.active_mode)
                if mode_idx == 0:
                    return self.env.start_pos
                else: 
                    transition_nodes = self.operation.modes[mode_idx - 1].transition_nodes
                    if transition_nodes == []:
                        q = self.get_goal_config_of_mode(self.operation.modes[mode_idx - 1])
                        return type(self.env.get_start_pos())(q, self.operation.start_node.state.q.slice)

                    else:
                        node = np.random.choice(transition_nodes)
                        return node.state.q
            else:
                q = []
                for i, robot in enumerate(self.env.robots):
                    informed_goal = mode.informed.goal.get(i)
                    informed_start = mode.informed.start.get(i)
                    if is_home_pose_sampling and robot not in mode.constrained_robots:
                        q.append(mode.starts[i])
                        continue
                    if is_home_pose_sampling and informed_goal is not None and np.array_equal(informed_goal, informed_start): #mode restriction 
                        q.append(informed_goal)
                        continue

                    if is_goal_sampling:  # Goal sampling
                        task = m[i]
                        if self.config.general_goal_sampling or robot in mode.constrained_robots:
                            goal = self.env.tasks[task].goal.sample()
                            if len(goal) == self.env.robot_dims[robot]:
                                q.append(goal)
                            else:
                                q.append(goal[self.env.robot_idx[robot]])
                            continue

                    if is_informed_sampling:  # Informed sampling
                        qr = self.sample_informed(i, self.env.tolerance)
                        if qr is not None:
                            q.append(qr)
                        else:
                            lims = self.operation.limits[robot]
                            q.append(np.random.uniform(lims[0], lims[1]))
                        continue

                    # Uniform sampling (default)
                    lims = self.operation.limits[robot]
                    q.append(np.random.uniform(lims[0], lims[1]))

                q_config = type(self.env.get_start_pos()).from_list(q)

            if self.env.is_collision_free(q_config.state(), m):
                return q_config

    def mode_selection(self, r_idx:int) -> Optional[Mode]:
        if self.operation.active_mode.label == self.env.terminal_mode:
            if not self.operation.init_sol:
                return
            return self.operation.active_mode
        if self.operation.active_mode == self.operation.modes[-1]:
            return
        
        task = self.operation.active_mode.label[r_idx]
        for mode in self.operation.modes:
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
        self.operation = Operation(env)
        self.sampling = Sampling(env, self.operation, self.config)
        self.start = time.time()
        self.start_single_goal= SingleGoal(self.env.start_pos.q)

    def SaveData(self, passed_time: time, n_new:NDArray=None, 
                 N_near:torch.tensor=None, r:float=None, n_rand:NDArray=None, n_nearest:NDArray = None, N_parent:torch.tensor=None) -> None:
        if self.config.debug_mode:
            return
        tree_data = []
        for mode in self.operation.modes:
            m = mode.label
            subtree_data = [
                {
                    "state": node.state.q.state(),
                    "parent": node.parent.state.q.state() if node.parent else None,
                    "mode": m,
                }
                for node in mode.subtree
            ]
            tree_data.extend(subtree_data)

        for mode in self.operation.modes:
            m = mode.label
            subtree_data = [
                {
                    "state": node.state.q.state(),
                    "parent": node.parent.state.q.state() if node.parent else None,
                    "mode": m,
                }
                for node in mode.subtree_b
            ]
            tree_data.extend(subtree_data)


        # Path and Cost Data
        if self.operation.path_nodes:
            transition_node = self.operation.path_nodes[-1]
            path_data = [state.q.state() for state in self.operation.path]

            intermediate_tot = torch.tensor(
                [node.cost for node in self.operation.path_nodes],
                device=device,
                dtype=torch.float32
            )
            # intermediate_agent_dists = torch.cat([node.agent_dists for node in self.operation.path_nodes])
            intermediate_agent_dists =[node.agent_dists for node in self.operation.path_nodes]
            result = {
                "path": path_data,
                "total": transition_node.cost,
                "agent_dists": transition_node.agent_dists,
                "intermediate_tot": intermediate_tot,
                "intermediate_agent_dists": intermediate_agent_dists,
                "is_transition": [node.transition for node in self.operation.path_nodes],
                "modes": [node.state.mode for node in self.operation.path_nodes],
            }
        else:
            # If no path nodes, set all result values to None
            result = {
                "path": None,
                "total": None,
                "agent_dists": None,
                "intermediate_tot": None,
                "intermediate_agent_dists": None,
                "is_transition": None,
                "modes": None,
            }
        
        if self.operation.paths_inter != []:
            inter_result = [{
                "path": [node.state.q.state() for node in path],
                "modes": [node.state.mode for node in path],
            } for path in self.operation.paths_inter]
        else:
            # If no path nodes, set all result values to None
            inter_result = [{
                "path": None,
                "modes": None,
            }]

        # Informed Sampling Data
        informed_sampling = [
            {
                "C": mode.informed.C,
                "L": mode.informed.L,
                "center": mode.informed.state_centre,
                "start": mode.informed.start,
                "goal": mode.informed.goal,
            }
            for mode in self.operation.modes if mode.informed.L
        ]

        # Nearby Nodes
        if N_near is None:
            N_near_list = []
        else:
            N_near_list = [N for N in N_near] if N_near.size(0) > 0 else []

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
            "active_mode": self.operation.active_mode.label,
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

    def Nearest(self, q_rand: Configuration, subtree: List[Node], subtree_set: torch.tensor) -> Node:
        q_tensor = torch.as_tensor(q_rand.state(), device=device, dtype=torch.float32).unsqueeze(0)
        set_dists = batch_config_dist_torch(q_tensor, q_rand, subtree_set, self.config.dist_type)
        idx = torch.argmin(set_dists).item()
        return  subtree[idx]

    def Steer(self, n_nearest: Node, q_rand: Configuration, m_label: List[int]) -> State: 
        if np.equal(n_nearest.state.q.state(), q_rand.state()).all():
            return None
        dists = config_dists(n_nearest.state.q, q_rand, self.config.dist_type)
        q_nearest = n_nearest.state.q.state()
        q_rand = q_rand.state()
        direction = q_rand - q_nearest

        q_new = np.empty_like(q_nearest)
        for idx, robot in enumerate(self.env.robots):
            indices = self.env.robot_idx[robot]
            robot_dist = dists[idx]
            if robot_dist < self.config.step_size:
                q_new[indices] = q_rand[indices]
            else:
                t = min(1, self.config.step_size / robot_dist) if robot_dist != 0 else 0
                q_new[indices] = q_nearest[indices] + t * direction[indices]

        q_new = np.clip(q_new, self.env.limits[0], self.env.limits[1]) 
        state_new = State(type(self.env.get_start_pos())(q_new, n_nearest.state.q.slice), m_label)
        return state_new
  
    def Near(self, n_new: Node, subtree_set: torch.tensor) -> Tuple[List[Node], torch.tensor]:
        #TODO generalize rewiring radius
        # n_nodes = sum(1 for _ in self.operation.current_mode.subtree.inorder()) + 1
        # r = min((7)*self.step_size, 3 + self.gamma * ((math.log(n_nodes) / n_nodes) ** (1 / self.dim)))
        set_dists = batch_config_dist_torch(n_new.q_tensor, n_new.state.q, subtree_set, self.config.dist_type)
        indices = torch.where(set_dists < self.r)[0] # indices of batch_subtree
        # N_near_batch = self.operation.active_mode.batch_subtree[indices, :]
        N_near_batch = self.operation.active_mode.batch_subtree.index_select(0, indices)
        # node_indices = self.operation.active_mode.node_idx_subtree[indices] # actual node indices (node.idx)
        node_indices = self.operation.active_mode.node_idx_subtree.index_select(0,indices) # actual node indices (node.idx)
        if indices.size(0)== 1:
            self.operation.active_mode.node_idx_subtree[indices]
        n_near_costs = self.operation.costs.index_select(0,node_indices)
        if not self.config.informed_sampling:
            return indices, N_near_batch, n_near_costs, node_indices
        return self.sampling.fit_to_informed_subset(indices, N_near_batch, n_near_costs, node_indices)
    
    def UpdateMode(self, mode:Mode, n:Node, tree:str) -> None:
        if tree == 'A':
            mode.subtree.append(n)                
            position = len(mode.subtree) -1
            mode.batch_subtree = mode.ensure_capacity(mode.batch_subtree, position)
            mode.batch_subtree[position,:] = n.q_tensor
            mode.node_idx_subtree = mode.ensure_capacity(mode.node_idx_subtree, position)
            mode.node_idx_subtree[position] = n.idx
        if tree == 'B':
            mode.subtree_b.append(n)
            position = len(mode.subtree_b) -1
            mode.batch_subtree_b = mode.ensure_capacity(mode.batch_subtree_b, position)
            mode.batch_subtree_b[position,:] = n.q_tensor
            mode.node_idx_subtree_b = mode.ensure_capacity(mode.node_idx_subtree_b, position)
            mode.node_idx_subtree_b[position] = n.idx

    def FindParent(self, N_near_indices: torch.tensor, idx:int, n_new: Node, n_nearest: Node, batch_cost: torch.tensor, batch_dist: torch.tensor, n_near_costs: torch.tensor) -> None:
        c_min = n_nearest.cost + batch_cost[idx]
        c_min_to_parent = batch_cost[idx]
        n_min = n_nearest
        c_new_tensor = n_near_costs + batch_cost
        valid_mask = c_new_tensor < c_min
        if torch.any(valid_mask):
            # valid_indices = torch.arange(c_new_tensor.size(0), device=device)[valid_mask]
            # sorted_indices = valid_indices[c_new_tensor[valid_indices].argsort()]
            sorted_indices = torch.arange(c_new_tensor.size(0), device=device)[valid_mask][c_new_tensor[valid_mask].argsort()]
            for idx in sorted_indices:
                node = self.operation.active_mode.subtree[N_near_indices[idx]]
                if self.env.is_edge_collision_free(
                    node.state.q, n_new.state.q, self.operation.active_mode.label
                ):
                    c_min = c_new_tensor[idx]
                    c_min_to_parent = batch_cost[idx]       # Update minimum cost
                    n_min = node                            # Update nearest node
                    break
        n_new.parent = n_min
        n_new.cost_to_parent = c_min_to_parent
        n_min.children.append(n_new) #Set child
        agent_dist = batch_dist[idx].unsqueeze(0).to(dtype=torch.float16).cpu()
        n_new.agent_dists = n_new.parent.agent_dists + agent_dist
        n_new.agent_dists_to_parent = agent_dist
        # self.operation.costs = torch.cat((self.operation.costs, c_min.unsqueeze(0)), dim=0) 
        # self.operation.costs.append(c_min.unsqueeze(0))
        self.operation.costs = self.operation.active_mode.ensure_capacity(self.operation.costs, n_new.idx) 
        n_new.cost = c_min
        
        self.UpdateMode(self.operation.active_mode, n_new, 'A') 

    def UnitBallVolume(self) -> float:
        return math.pi ** (self.dim / 2) / math.gamma((self.dim / 2) + 1)
    
    def Rewire(self, N_near_indices: torch.Tensor, n_new: Node, batch_cost: torch.tensor, 
               batch_dist: torch.tensor, n_near_costs: torch.tensor, n_rand = None, n_nearest = None) -> bool:
        rewired = False
        c_potential_tensor = n_new.cost + batch_cost
        # c_agent_tensor = batch_dist + n_new.agent_dists

        improvement_mask = c_potential_tensor < n_near_costs
        
        if torch.any(improvement_mask):
            # self.SaveData(time.time()-self.start, n_new = n_new.state.q.state(), N_near = N_near, 
            #                   r =self.r, n_rand = n_rand, n_nearest = n_nearest) 
            improved_indices = torch.arange(c_potential_tensor.size(0), device=device)[improvement_mask]
            for idx in improved_indices:
                n_near = self.operation.active_mode.subtree[N_near_indices[idx].item()]
                # n_near = N_near[idx]
                if n_near == n_new.parent or n_near.cost == np.inf or n_near == n_new:
                    continue

                if self.env.is_edge_collision_free(n_near.state.q, n_new.state.q, self.operation.active_mode.label):
                    if n_near.parent is not None:
                        n_near.parent.children.remove(n_near)
                    n_near.parent = n_new                    
                    n_new.children.append(n_near)

                    n_near.cost = c_potential_tensor[idx].item()
                    # n_near.agent_dists = c_agent_tensor[idx].unsqueeze(0) 
                    agents = batch_dist[idx].unsqueeze(0).to(dtype=torch.float16).cpu()
                    n_near.agent_dists =  agents + n_new.agent_dists
                    n_near.cost_to_parent = batch_cost[idx]
                    n_near.agent_dists_to_parent = agents
                    # self.SaveData(time.time()-self.start, n_new = n_new.state.q.state(), 
                    #               r =self.r, n_rand = n_near.state.q.state())
                    rewired = True
        return rewired
      
    def GeneratePath(self, n: Node) -> None:
        path_nodes, path = [], []
        while n:
            path_nodes.append(n)
            path.append(n.state)
            n = n.parent
        path_in_order = path[::-1]
        self.operation.path = path_in_order  
        self.operation.path_nodes = path_nodes[::-1]
        self.operation.cost = self.operation.path_nodes[-1].cost.clone()
        # if not self.env.is_path_collision_free(self.operation.path):
        #     print('hallo')
        self.SaveData(time.time()-self.start)

    def AddTransitionNode(self, n: Node) -> None:
            """Need to add transition node n as a start node in the next mode"""
            idx = self.operation.modes.index(self.operation.active_mode)
            if idx != len(self.operation.modes) - 1:
                mode = self.operation.modes[idx + 1]
                if self.operation.modes[idx + 1].order == 1:
                    mode.subtree.append(n)                
                    position = len(mode.subtree) -1
                    mode.batch_subtree = mode.ensure_capacity(mode.batch_subtree, position)
                    mode.batch_subtree[position,:] = n.q_tensor
                    mode.node_idx_subtree = mode.ensure_capacity(mode.node_idx_subtree, position)
                    mode.node_idx_subtree[position] = n.idx

                else:
                    mode.subtree_b.append(n)
                    position = len(mode.subtree_b) -1
                    mode.batch_subtree_b = mode.ensure_capacity(mode.batch_subtree_b, position)
                    mode.batch_subtree_b[position,:] = n.q_tensor
                    mode.node_idx_subtree_b = mode.ensure_capacity(mode.node_idx_subtree_b, position)
                    mode.node_idx_subtree_b[position] = n.idx

    def SetModePorbability(self) -> List[float]:
        num_modes = len(self.operation.modes)
        if num_modes == 1:
            return [1] 
        # if self.operation.task_sequence == [] and self.config.mode_probability != 0:
        if self.operation.init_sol and self.config.mode_probability != 0:
                return [1/num_modes] * num_modes
        
        elif self.config.mode_probability == 'None':
            # equally
            return [1 / (num_modes)] * (num_modes)

        elif self.config.mode_probability == 1:
            # greedy (only latest mode is selected until all initial paths are found)
            probability = [0] * (num_modes)
            probability[-1] = 1
            return probability

        elif self.config.mode_probability == 0:
            # Uniformly
            total_transition_nodes = sum(len(mode.transition_nodes) for mode in self.operation.modes)
            total_nodes = Node.id_counter -1 + total_transition_nodes
            # Calculate probabilities inversely proportional to node counts
            inverse_probabilities = [
                1 - (len(mode.subtree) / total_nodes)
                for mode in self.operation.modes
            ]
            # Normalize the probabilities to sum to 1
            total_inverse = sum(inverse_probabilities)
            return [
                inv_prob / total_inverse for inv_prob in inverse_probabilities
            ]

        else:
            # manually set
            total_transition_nodes = sum(len(mode.transition_nodes) for mode in self.operation.modes)
            total_nodes = Node.id_counter -1 + total_transition_nodes
            # Calculate probabilities inversely proportional to node counts
            inverse_probabilities = [
                1 - (len(mode.subtree) / total_nodes)
                for mode in self.operation.modes[:-1]  # Exclude the last mode
            ]

            # Normalize the probabilities of all modes except the last one
            remaining_probability = 1-self.config.mode_probability  
            total_inverse = sum(inverse_probabilities)
            return [
                (inv_prob / total_inverse) * remaining_probability
                for inv_prob in inverse_probabilities
            ] + [self.config.mode_probability]

    def ModeInitialization(self,new_mode: Mode) -> None: #TODO imporve?
        mode = self.env.start_mode
        previous_mode = mode
        robots = []
        goal = {}
        state_start = {}
        while True:
            task = self.env.get_active_task(mode)
            constrained_robtos = task.robots
            for robot in self.env.robots: #only for constrained robots
                indices = self.env.robot_idx[robot]
                r = self.env.robots.index(robot)
                if r not in robots: 
                    if robot in constrained_robtos:
                        robots.append(r)
                    state_start[r] = self.env.start_pos.q[indices]
                else:
                    if previous_mode[r] != mode[r]:
                        state_start[r] = goal[r]
                if robot in constrained_robtos:
                    if len(constrained_robtos) > 1:
                        goal[r] = task.goal.sample()[indices]
                    else:
                        goal[r] = task.goal.sample()
                if new_mode.label == mode:
                    if robot in constrained_robtos:
                        new_mode.informed.start[r] = state_start[r]
                        new_mode.informed.goal[r] = goal[r]
                        if self.config.informed_sampling and not np.equal(goal[r], state_start[r]).all():
                            cmin, C = self.sampling.rotation_to_world_frame(state_start[r], goal[r] ,robot)
                            C = torch.tensor(C, device = device, dtype=torch.float32)
                            new_mode.informed.C[r] = C
                            new_mode.informed.inv_C[r] = torch.linalg.inv(C)
                            new_mode.informed.cmin[r] = torch.tensor(cmin-2*self.env.tolerance, device= device, dtype=torch.float32)
                            new_mode.informed.state_centre[r] = torch.tensor(((state_start[r] + goal[r])/2), device=device, dtype=torch.float32)
                    if robot == self.env.robots[-1]:
                        new_mode.starts = state_start
                        return
            if mode != previous_mode:
                previous_mode = mode
            mode = self.env.get_next_mode(None,mode)

    def SampleNodeManifold(self, operation: Operation) -> Configuration:
        if np.random.uniform(0, 1) <= self.config.p_goal:
            if self.config.p_stay != 1 and np.random.uniform(0, 1) > self.config.p_stay: # sample pose of task before?
                if self.operation.init_sol:
                    return self.sampling.sample_state(operation.active_mode, 3)
                return self.sampling.sample_state(operation.active_mode, 0)
                
            elif self.env.terminal_mode != operation.modes[-1].label and operation.active_mode.label == operation.modes[-1].label:
                return self.sampling.sample_state(operation.active_mode, 0)
            else:  
                if self.config.informed_sampling: #have found initial sol for informed smapling
                    return self.sampling.sample_state(operation.active_mode, 1)
                return self.sampling.sample_state(operation.active_mode, 0)
        return self.sampling.sample_state(operation.active_mode, 2) #goal sampling
    
    def FindOptimalTransitionNode(self, iter: int, mode_init_sol: bool = False) -> None:
        if len(self.operation.modes) < 2:
            return
        if self.operation.init_sol:
            transition_nodes = self.operation.modes[-1].transition_nodes
        else:
            transition_nodes = self.operation.modes[-2].transition_nodes 
        if len(transition_nodes) > 1:
            costs = torch.tensor([node.cost for node in transition_nodes], device=device, dtype=torch.float32)
        else:
            costs = transition_nodes[0].cost.view(-1)
        if mode_init_sol:
            valid_mask = costs < np.inf
        else:
            valid_mask = costs < self.operation.cost

        if valid_mask.any():
            min_cost_idx = torch.masked_select(torch.arange(len(costs), device=costs.device), valid_mask)[
                costs[valid_mask].argmin()
            ].item()
            lowest_cost_node = transition_nodes[min_cost_idx]
            self.GeneratePath(lowest_cost_node)
            print(f"{iter} Cost: ", self.operation.cost, " Mode: ", self.operation.active_mode.label)

            if (self.operation.ptc_cost - self.operation.cost) > self.config.ptc_threshold:
                self.operation.ptc_cost = self.operation.cost
                self.operation.ptc_iter = iter

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
