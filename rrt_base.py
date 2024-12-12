import numpy as np
import logging
from datetime import datetime
import yaml
import json
import pickle
import os
from configuration import *
from planning_env import *
from util import *
import argparse
import copy
import torch
import time as time
import math as math
from typing import Tuple, Optional
from operator import itemgetter

torch.cuda.manual_seed_all(10)

class ConfigManager:
    def __init__(self, config_file:str):
        # Load configuration
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        
        # Set defaults and dynamically assign attributes
        defaults = {
            'goal_radius': 0.1, 'p_goal': 0.95, 'general_goal_sampling': True, 
            'step_size': 0.3, 'cost_function': 2, 'ptc_threshold': 0.1, 
            'ptc_max_iter': 3000, 'mode_probability': 0.4, 
            'informed_sampling': True, 'informed_sampling_version': 0, 
            'cprofiler': False, 'cost': 'euclidean', 'show_path': False 
        }
        for key, value in defaults.items():
            setattr(self, key, config.get(key, value))
        
        # Output directory
        self.timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        self.output_dir = os.path.join(os.path.expanduser("~"), f'output/{self.timestamp}/')
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up logging
        self._setup_logger()

    def _setup_logger(self) -> None:
        logging.basicConfig(
            filename=os.path.join(self.output_dir, 'general.log'), 
            level=logging.INFO, 
            format='%(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def log_params(self, args: argparse) -> None:
        """Logs configuration parameters along with environment and planner details."""
        config_dict = {k: v for k, v in self.__dict__.items() if k != "logger"}
        self.logger.info('Environment: %s', json.dumps(args.env_name, indent=2))
        self.logger.info('Planner: %s', json.dumps(args.planner, indent=2))
        self.logger.info('Configuration Parameters: %s', json.dumps(config_dict, indent=4))

class Mode:
    """Mode description"""
    def __init__(self, mode, env):
        self.label = mode
        self.transition_nodes = []
        robot_dims = sum(env.robot_dims.values())
        self.subtree = {}
        self.batch_subtree = torch.empty((0, robot_dims), device='cuda')
        self.node_idx_subtree = torch.empty(0, device='cuda', dtype=torch.long)
        self.subtree_b = {}
        self.batch_subtree_b = torch.empty((0, robot_dims), device='cuda')
        self.node_idx_subtree_b =  torch.empty(0, device='cuda', dtype=torch.long)
        self.informed = Informed()
        self.order = 1
        self.connected = False

class Operation:
    """ Planner operation variables"""
    def __init__(self, env):
        self.modes = [Mode(env.start_mode, env)] # list of visited modes
        self.active_mode = None
        self.path = []
        self.path_nodes = None
        self.cost = np.inf
        self.tree = 1
        self.ptc_cost = 0 # needed for termination
        self.ptc_iter = None # needed for termination
        self.init_sol = False
        self.costs = torch.tensor([0], device='cuda')
    
    def get_cost(self, idx):
        """
        Retrieve the cost for a specific index.
        """
        return self.costs[idx]
    


class Node:
    def __init__(self, state:State, idx: int, operation: Operation):
        self.idx = idx
        self.state = state   
        self.q_tensor = torch.tensor(state.q.state(), device='cuda')
        self.parent = None  
        self.children = []    
        self.transition = False
        num_agents = state.q.num_agents()
        # self.cost = torch.tensor(0, device='cuda')
        self.agent_cost = torch.zeros(1, num_agents, device = 'cuda')
        self.cost_to_parent = torch.tensor(0, device='cuda')
        self.agent_cost_to_parent = torch.zeros(1, num_agents, device = 'cuda')
        self.operation = operation

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
        pass
       
    def sample_state(self, mode: Mode, sampling_type: int) -> Configuration:
        m = mode.label
        active_task = self.env.get_active_task(m)
        constrained_robots = active_task.robots
        limits = {robot: self.env.limits[:, self.env.robot_idx[robot]] for robot in self.env.robots}
        is_goal_sampling = sampling_type == 2
        is_informed_sampling = sampling_type == 1

        while True:
            if sampling_type == 2 and mode.order == -1:
                mode_idx = self.operation.modes.index(self.operation.active_mode)
                if mode_idx == 0:
                    q_config = self.env.start_pos
                else:
                    transition_nodes = self.operation.modes[mode_idx - 1].transition_nodes
                    node = np.random.choice(transition_nodes)
                    q_config = node.state.q
            else:
                q = []
                for i, robot in enumerate(self.env.robots):
                    informed_goal = mode.informed.goal.get(i)
                    informed_start = mode.informed.start.get(i)


                    if informed_goal is not None and np.array_equal(informed_goal, informed_start):
                        q.append(informed_goal)
                        continue

                    if is_goal_sampling:  # Goal sampling
                        task = m[i]
                        if self.config.general_goal_sampling or robot in constrained_robots:
                            goal = self.env.tasks[task].goal.sample()
                            if len(goal) == self.env.robot_dims[robot]:
                                q.append(goal)
                            else:
                                q.append(goal[self.env.robot_idx[robot]])
                            continue

                    if is_informed_sampling:  # Informed sampling
                        qr = self.sample_informed(i, self.config.goal_radius)
                        if qr is not None:
                            q.append(qr)
                        else:
                            lims = limits[robot]
                            q.append(np.random.uniform(lims[0], lims[1]))
                        continue

                    # Uniform sampling (default)
                    lims = limits[robot]
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
        c_tot, c_start_tot, cmax =  mode.informed.cmax(mode.transition_nodes, self.operation.path_nodes, goal_radius, r_idx, mode.informed.start[agent]) # reach task for all agents, reach goal of task for specific agent
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
                    if np.equal(start.state.q.state()[indices], mode.informed.start[r_idx]).all() and np.equal(node.state.q.state()[indices], mode.informed.goal[agent]).all():
                        cmin += mode.informed.cmin[r_idx]
                        start.state.q.state()[indices] = node.state.q.state()[indices]
                        m = self.env.get_next_mode(None, m)
                        active_robots = self.env.get_active_task(m).robots
                        idx = []
                        for r in active_robots:
                            idx.extend( self.env.robot_idx[r])

                    else:
                        cmin += config_cost(start.state.q, node.state.q, "euclidean")
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
            self.env.limits[:, self.env.robot_idx[agent_name]],
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
        x_ball = torch.randn(n, device='cuda', dtype=torch.float64)
        x_ball /= torch.linalg.norm(x_ball, ord=2)
        radius = torch.rand(1, device='cuda', dtype=torch.float64).pow(1 / n)
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
                    c_agent = node.agent_cost[:,r_idx]
                    c_tot = node.cost
                    break
            c_start_agent, c_start_tot = self.get_start_node(path_nodes, goal_radius, r_idx, start)
            return c_tot, c_start_tot, c_agent - c_start_agent

    def get_start_node(self, path_nodes:List[Node], goal_radius:float, r_idx:int, start:NDArray) -> Tuple[torch.tensor, torch.tensor]:
        for node in path_nodes:
            if np.array_equal(node.state.q[r_idx], start) or np.linalg.norm(start - node.state.q[r_idx]) < goal_radius:
                return node.agent_cost[:,r_idx], node.cost 

class BaseRRTstar(ABC):
    def __init__(self, env, config: ConfigManager):
        self.env = env
        # self.gamma = ((2 *(1 + 1/self.dim))**(1/self.dim) * (self.FreeSpace()/self.UnitBallVolume())**(1/self.dim))*1.1 
        self.config = config
        self.r = self.config.step_size * sum(self.env.robot_dims.values())
        self.operation = Operation(env)
        self.sampling = Sampling(env, self.operation, self.config)
        self.start = time.time()

    def SaveData(self, passed_time: time, n_new:NDArray=None, 
                 N_near:List[Node]=None, r:float=None, n_rand:NDArray=None, n_nearest:NDArray = None) -> None:
        tree_data = []
        for mode in self.operation.modes:
            m = mode.label
            subtree_data = [
                {
                    "state": node.state.q.state(),
                    "parent": node.parent.state.q.state() if node.parent else None,
                    "mode": m,
                }
                for node in mode.subtree.values()
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
                for node in mode.subtree_b.values()
            ]
            tree_data.extend(subtree_data)


        # Path and Cost Data
        if self.operation.path_nodes:
            transition_node = self.operation.path_nodes[-1]
            path_data = [state.q.state() for state in self.operation.path]
            intermediate_tot = torch.cat([node.cost.unsqueeze(0) for node in self.operation.path_nodes])
            intermediate_agent_cost = torch.cat([node.agent_cost for node in self.operation.path_nodes])
            result = {
                "path": path_data,
                "total": transition_node.cost,
                "agent_cost": transition_node.agent_cost,
                "intermediate_tot": intermediate_tot,
                "intermediate_agent_cost": intermediate_agent_cost,
                "is_transition": [node.transition for node in self.operation.path_nodes],
                "modes": [node.state.mode for node in self.operation.path_nodes],
            }
        else:
            # If no path nodes, set all result values to None
            result = {
                "path": None,
                "total": None,
                "agent_cost": None,
                "intermediate_tot": None,
                "intermediate_agent_cost": None,
                "is_transition": None,
                "modes": None,
            }

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
        N_near_list = [N.state.q.state() for N in N_near] if N_near else []

        # Data Assembly
        data = {
            "tree": tree_data,
            "result": result,
            "all_init_path": self.operation.init_sol,
            "time": passed_time,
            "informed_sampling": informed_sampling,
            "n_rand": n_rand,
            "N_near": N_near_list,
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
        set_dists = batch_config_dist_torch(q_rand, subtree_set, "euclidean")
        idx = torch.argmin(set_dists).item()
        return  subtree[idx], idx

    def Steer(self, n_nearest: Node, q_rand: Configuration, m_label: List[int]) -> Node: 
        dists = config_dists(n_nearest.state.q, q_rand, "euclidean")
        if np.max(dists) < self.config.step_size:
            state_new = State(q_rand, m_label)
            n_new = Node(state_new, self.operation.tree, self.operation)
        q_new = []
        q_nearest = n_nearest.state.q.state()
        q_rand = q_rand.state()
        direction = q_rand - q_nearest
        for idx, robot in enumerate(self.env.robots):
            indices = self.env.robot_idx[robot]
            robot_dist = dists[idx]
            if robot_dist < self.config.step_size:
                q_new.append(q_rand[indices])
            else:
                if robot_dist == 0:
                    t = 0
                else:
                    t = min(1, self.config.step_size / robot_dist) 
                q_new.append(q_nearest[indices] + t * direction[indices])
        q_new = np.concatenate(q_new, axis=0)
        q_new = np.clip(q_new, self.env.limits[0], self.env.limits[1]) 
        state_new = State(type(self.env.get_start_pos())(q_new, n_nearest.state.q.slice), m_label)
        n_new = Node(state_new, self.operation.tree, self.operation)
        return n_new
  
    def Near(self, n_new: Node, subtree_set: torch.tensor) -> Tuple[List[Node], torch.tensor]:
        #TODO generalize rewiring radius
        # n_nodes = sum(1 for _ in self.operation.current_mode.subtree.inorder()) + 1
        # r = min((7)*self.step_size, 3 + self.gamma * ((math.log(n_nodes) / n_nodes) ** (1 / self.dim)))
        set_dists = batch_config_dist_torch(n_new.state.q, subtree_set, "euclidean")
        indices = torch.where(set_dists < self.r)[0]
        N_near_batch = self.operation.active_mode.batch_subtree[indices, :]
        node_indices = self.operation.active_mode.node_idx_subtree[indices]
        n_near_costs = self.operation.costs[node_indices]
        if not self.config.informed_sampling:
            return indices, N_near_batch, n_near_costs, node_indices
        return self.sampling.fit_to_informed_subset(indices, N_near_batch, n_near_costs, node_indices)
                
    def FindParent(self, N_near_indices: torch.tensor, idx:int, n_new: Node, n_nearest: Node, batch_cost: torch.tensor, batch_dist: torch.tensor, n_near_costs: torch.tensor) -> None:
        c_min = n_near_costs[idx] + batch_cost[idx]
        c_min_to_parent = batch_cost[idx]
        n_min = n_nearest
        c_new_tensor = n_near_costs + batch_cost
        valid_indices = torch.nonzero(c_new_tensor < c_min, as_tuple=False).view(-1)

        if valid_indices.numel():
            sorted_indices = valid_indices[c_new_tensor[valid_indices].argsort()]
            for idx in sorted_indices:
                node_idx = N_near_indices[idx].item()
                node = self.operation.active_mode.subtree[node_idx]
                if self.env.is_edge_collision_free(
                    node.state.q, n_new.state.q, self.operation.active_mode.label
                ):
                    c_min = c_new_tensor[idx]
                    c_min_to_parent = batch_cost[idx]       # Update minimum cost
                    n_min = node             # Update nearest node
                    break

        n_new.parent = n_min
        n_new.cost_to_parent = c_min_to_parent
        n_min.children.append(n_new) #Set child
        n_new.agent_cost = n_new.parent.agent_cost + batch_dist[idx].unsqueeze(0) 
        n_new.agent_cost_to_parent = batch_dist[idx].unsqueeze(0)
        self.operation.tree +=1
        self.operation.active_mode.subtree[len(self.operation.active_mode.subtree)] = n_new
        self.operation.active_mode.batch_subtree = torch.cat((self.operation.active_mode.batch_subtree, n_new.q_tensor.unsqueeze(0)), dim=0)
        self.operation.costs = torch.cat((self.operation.costs, c_min.unsqueeze(0)), dim=0) #set cost of n_new
        self.operation.active_mode.node_idx_subtree = torch.cat((self.operation.active_mode.node_idx_subtree, torch.tensor([n_new.idx], device='cuda')),dim=0)

    def UnitBallVolume(self) -> float:
        return math.pi ** (self.dim / 2) / math.gamma((self.dim / 2) + 1)
    
    def Rewire(self, N_near_indices: torch.Tensor, n_new: Node, batch_cost: torch.tensor, 
               batch_dist: torch.tensor, n_near_costs: torch.tensor, n_rand = None, n_nearest = None) -> bool:
        rewired = False
        c_potential_tensor = n_new.cost + batch_cost
        c_agent_tensor = batch_dist + n_new.agent_cost

        improvement_mask = c_potential_tensor < n_near_costs
        improved_indices = torch.nonzero(improvement_mask, as_tuple=False).view(-1)
        if improved_indices.numel():
            # self.SaveData(time.time()-self.start, n_new = n_new.state.q.state(), N_near = N_near, 
            #                   r =self.r, n_rand = n_rand, n_nearest = n_nearest) 
            for idx in improved_indices:
                n_near = self.operation.active_mode.subtree[N_near_indices[idx].item()]
                # n_near = N_near[idx]
                if n_near == n_new.parent:
                    continue

                if self.env.is_edge_collision_free(n_near.state.q, n_new.state.q, self.operation.active_mode.label):
                    if n_near.parent is not None:
                        n_near.parent.children.remove(n_near)

                    n_near.parent = n_new
                    if n_new != n_near:
                        n_new.children.append(n_near)

                    n_near.cost = c_potential_tensor[idx].item()
                    n_near.agent_cost = c_agent_tensor[idx].unsqueeze(0) 
                    n_near.cost_to_parent = batch_cost[idx]
                    n_near.agent_cost_to_parent = batch_dist[idx].unsqueeze(0)
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
        self.operation.cost = self.operation.path_nodes[-1].cost
        self.SaveData(time.time()-self.start)

    def AddTransitionNode(self, n: Node) -> None:
            idx = self.operation.modes.index(self.operation.active_mode)
            if idx != len(self.operation.modes) - 1:
                if self.operation.modes[idx + 1].order == 1:
                    self.operation.modes[idx + 1].subtree[len(self.operation.modes[idx + 1].subtree)] = n
                    self.operation.modes[idx + 1].batch_subtree = torch.cat((self.operation.modes[idx + 1].batch_subtree, n.q_tensor.unsqueeze(0)), dim=0)
                    self.operation.modes[idx + 1].node_idx_subtree = torch.cat((self.operation.modes[idx + 1].node_idx_subtree, torch.tensor([n.idx], device='cuda')),dim=0)
                else:
                    self.operation.modes[idx + 1].subtree_b[len(self.operation.modes[idx + 1].subtree_b)] = n
                    self.operation.modes[idx + 1].batch_subtree_b = torch.cat((self.operation.modes[idx + 1].batch_subtree_b, n.q_tensor.unsqueeze(0)), dim=0)
                    self.operation.modes[idx + 1].node_idx_subtree_b = torch.cat((self.operation.modes[idx + 1].node_idx_subtree_b, torch.tensor([n.idx], device='cuda')),dim=0)

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
            total_nodes = self.operation.tree + total_transition_nodes
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
            total_nodes = self.operation.tree_size + total_transition_nodes
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

    def InitializationMode(self, added_mode: Mode) -> None:
        mode = self.env.start_mode
        robots = []
        goal = {}
        state_start = {}
        while True:
            task = self.env.get_active_task(mode)
            constrained_robtos = task.robots
            for robot in constrained_robtos:
                indices = self.env.robot_idx[robot]
                r = self.env.robots.index(robot)
                if r not in robots: 
                    robots.append(r)
                    state_start[r] = self.env.start_pos.q[indices]
                else:
                    state_start[r] = goal[r]
                if len(constrained_robtos) > 1:
                    goal[r] = task.goal.sample()[indices]
                else:
                    goal[r] = task.goal.sample()
                
                if added_mode.label == mode:
                    added_mode.informed.start[r] = state_start[r]
                    added_mode.informed.goal[r] = goal[r]
                    if self.config.informed_sampling and not np.equal(added_mode.informed.goal[r], added_mode.informed.start[r]).all():
                        cmin, C = self.sampling.rotation_to_world_frame(state_start[r], goal[r] ,robot)
                        C = torch.tensor(C, device = 'cuda')
                        added_mode.informed.C[r] = C
                        added_mode.informed.inv_C[r] = torch.linalg.inv(C)
                        added_mode.informed.cmin[r] = torch.tensor(cmin-2*self.config.goal_radius, device= 'cuda')
                        added_mode.informed.state_centre[r] = torch.tensor(((state_start[r] + goal[r])/2), device='cuda')

                    if robot == constrained_robtos[-1]:
                        return
            mode = self.env.get_next_mode(None,mode)

    def SampleNodeManifold(self, operation: Operation) -> Configuration:
        if np.random.uniform(0, 1) <= self.config.p_goal:
            if self.env.terminal_mode != operation.modes[-1].label and operation.active_mode.label == operation.modes[-1].label:
                return self.sampling.sample_state(operation.active_mode, 0)
            else:  
                if self.config.informed_sampling: 
                    return self.sampling.sample_state(operation.active_mode, 1)
                return self.sampling.sample_state(operation.active_mode, 0)
        return self.sampling.sample_state(operation.active_mode, 2)
    
    def FindOptimalTransitionNode(self, iter: int, mode_init_sol: bool = False) -> None:
        if len(self.operation.modes) < 2:
            return
        if self.operation.init_sol:
            transition_nodes = self.operation.modes[-1].transition_nodes
        else:
            transition_nodes = self.operation.modes[-2].transition_nodes 
        if len(transition_nodes) > 1:
            costs = torch.cat([node.cost.unsqueeze(0) for node in transition_nodes])
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
            if not mode_init_sol:
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
