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

torch.cuda.manual_seed_all(10)

def save_data(config, operation, passed_time, n_new=None, N_near=None, r=None, n_rand=None, n_rand_label=None, init_path=False):
    tree_data = []
    for mode in operation.modes:
        subtree_data = [
            {
                "state": node.state.q.state(),
                "parent": node.parent.state.q.state() if node.parent else None,
                "mode": node.state.mode,
            }
            for node in mode.subtree.values()
        ]
        tree_data.extend(subtree_data)
    for mode in operation.modes:
        subtree_data = [
            {
                "state": node.state.q.state(),
                "parent": node.parent.state.q.state() if node.parent else None,
                "mode": node.state.mode,
            }
            for node in mode.subtree_b.values()
        ]
        tree_data.extend(subtree_data)


    # Path and Cost Data
    if operation.path_nodes:
        transition_node = operation.path_nodes[-1]
        path_data = [state.q.state() for state in operation.path]
        intermediate_tot = torch.cat([node.cost.unsqueeze(0) for node in operation.path_nodes])
        intermediate_agent_cost = torch.cat([node.agent_cost for node in operation.path_nodes])
        result = {
            "path": path_data,
            "total": transition_node.cost,
            "agent_cost": transition_node.agent_cost,
            "intermediate_tot": intermediate_tot,
            "intermediate_agent_cost": intermediate_agent_cost,
            "is_transition": [node.transition for node in operation.path_nodes],
            "modes": [node.state.mode for node in operation.path_nodes],
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
        for mode in operation.modes if mode.informed.L
    ]

    # Nearby Nodes
    N_near_list = [N.state.q.state() for N in N_near] if N_near else []

    # Data Assembly
    data = {
        "tree": tree_data,
        "result": result,
        "all_init_path": init_path,
        "time": passed_time,
        "informed_sampling": informed_sampling,
        "n_rand": n_rand,
        "n_rand_label": n_rand_label,
        "N_near": N_near_list,
        "rewire_r": r,
        "n_new": n_new.state.q.state() if n_new else None,
        "active_mode": operation.active_mode.label,
    }

    # Directory Handling: Ensure directory exists
    frames_directory = os.path.join(config.output_dir, 'FramesData')
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

class ConfigManager:
    def __init__(self, config_file):
        # Load configuration
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        
        # Set defaults and dynamically assign attributes
        defaults = {
            'goal_radius': 0.1, 'p_goal': 0.95, 'general_goal_sampling': True, 
            'step_size': 0.3, 'cost_function': 2, 'ptc_threshold': 0.1, 
            'ptc_max_iter': 3000, 'mode_probability': 0.4, 
            'informed_sampling': True, 'informed_sampling_version': 0, 
            'cprofiler': False, 'cost': 'euclidean' 
        }
        for key, value in defaults.items():
            setattr(self, key, config.get(key, value))
        
        # Output directory
        self.timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        self.output_dir = os.path.join(os.path.expanduser("~"), f'output/{self.timestamp}/')
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up logging
        self._setup_logger()

    def _setup_logger(self):
        logging.basicConfig(
            filename=os.path.join(self.output_dir, 'general.log'), 
            level=logging.INFO, 
            format='%(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def log_params(self, args: argparse):
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
        # self.subtree = kdtree.create(dimensions=sum(env.robot_dims.values()))
        self.subtree = {}
        self.batch_subtree = torch.empty((0, sum(env.robot_dims.values())), device='cuda')
        self.subtree_b = {}
        self.batch_subtree_b = torch.empty((0, sum(env.robot_dims.values())), device='cuda')
        self.informed = Informed()
        self.order = 1
        self.connected = False

class Node:
    def __init__(self, state:State):
        self.state = state   
        self.q_tensor = torch.tensor(state.q.state(), device='cuda')
        self.parent = None  
        self.children = []    
        self.transition = False
        self.cost = torch.tensor(0, device='cuda')
        self.agent_cost = torch.zeros(1, state.q.num_agents(), device = 'cuda')
        self.cost_to_parent = torch.tensor(0, device='cuda')
        self.agent_cost_to_parent = torch.zeros(1, state.q.num_agents(), device = 'cuda')

    def __repr__(self):
        return f"<N- {self.state.q.state()}, c: {self.cost}>"
    
class Operation:
    """ Planner operation variables"""
    def __init__(self, env):
        self.modes = [Mode(env.start_mode, env)] # list of visited modes
        self.active_mode = None
        self.path = []
        self.path_nodes = None
        self.cost = 0
        self.tree = 1
        self.ptc_cost = 0 # needed for termination
        self.ptc_iter = None # needed for termination
        self.init_path = False

class Sampling:
    def __init__(self, env, operation, config):
        self.env = env
        self.operation = operation
        self.config = config
        pass

            
    def sample_state(self, mode: Mode, sampling_type: int, config: ConfigManager, operation:Operation = None) -> State:
        m = mode.label
        active_task = self.env.get_active_task(m)
        constrained_robots = active_task.robots
        limits = {robot: self.env.limits[:, self.env.robot_idx[robot]] for robot in self.env.robots}
        

        def sample_robot_state(i, robot):
            lims = limits[robot]

            if i in mode.informed.goal and np.array_equal(mode.informed.goal[i], mode.informed.start[i]):
                return mode.informed.goal[i]

            if sampling_type == 2:  # Goal sampling
                task = m[i]
                if config.general_goal_sampling or robot in constrained_robots:
                    goal = self.env.tasks[task].goal.sample()
                    return goal[self.env.robot_idx[robot]] if len(goal) != self.env.robot_dims[robot] else goal #TODO not working for handover

            if sampling_type == 1:  # Informed sampling
                qr = self.sample_informed(i, config.goal_radius)
                return qr if qr is not None else np.random.uniform(lims[0], lims[1])

            return np.random.uniform(lims[0], lims[1])  # Uniform sampling (default)

        while True:
            if sampling_type == 2 and mode.order == -1:
                mode_idx = self.operation.modes.index(self.operation.active_mode)
                if mode_idx == 0:
                     state_candidate = self.env.start_pos
                else:
                    transition_nodes = self.operation.modes[mode_idx-1].transition_nodes
                    node = np.random.choice(transition_nodes)
                    state_candidate = node.state.q

            else:
                q = [sample_robot_state(i, robot) for i, robot in enumerate(self.env.robots)]
                state_candidate = type(self.env.get_start_pos()).from_list(q)
            if self.env.is_collision_free(state_candidate.state(), m):
                return State(state_candidate, m)


    def mode_selection(self, r):
        if self.operation.active_mode.label == self.env.terminal_mode:
            if not self.operation.init_path:
                return
            return self.operation.active_mode
        if self.operation.active_mode == self.operation.modes[-1]:
            return
        
        task = self.operation.active_mode.label[r]
        for mode in self.operation.modes:
            active_seq_idx = self.env.get_current_seq_index(mode.label)
            active_task = self.env.sequence[active_seq_idx]
            if task == active_task:
                if mode.label != self.operation.modes[-1].label:
                    return mode
                return
        return

    def sample_informed(self, agent:int, goal_radius:float):
        """Samples a point from the ellipsoidal subset defined by the start and goal positions and c_best."""
        mode = self.mode_selection(agent)
        agent_name = self.env.robots[agent]
        if not mode:
            return  # Initial path not yet found
        if not self.operation.init_path and (self.config.informed_sampling_version == 1 or self.config.informed_sampling_version == 2):
            return
        c_tot, c_start_tot, cmax =  mode.informed.cmax(mode.transition_nodes, self.operation.path_nodes, goal_radius, agent, mode.informed.start[agent]) # reach task for all agents, reach goal of task for specific agent
        if self.config.informed_sampling_version == 0:
            cmax = cmax
            cmin = mode.informed.cmin[agent]
        elif self.config.informed_sampling_version == 1:
            cmax = self.operation.path_nodes[-1].cost
            cmin = mode.informed.cmin[agent] + self.operation.path_nodes[-1].cost -c_tot + c_start_tot
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
                    if np.equal(start.state.q.state()[indices], mode.informed.start[agent]).all() and np.equal(node.state.q.state()[indices], mode.informed.goal[agent]).all():
                        cmin += mode.informed.cmin[agent]
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

            pass

        r1 = cmax / 2
        r2 = torch.sqrt(cmax**2 - cmin**2) / 2

        
        n = self.env.robot_dims[agent_name]
        mode.informed.L[agent] = torch.diag(torch.cat([r1.repeat(1), r2.repeat(n - 1)]))

        lims = torch.as_tensor(
            self.env.limits[:, self.env.robot_idx[agent_name]],
            device=cmax.device,
            dtype=cmax.dtype
        )
        L = mode.informed.L[agent]
        C = mode.informed.C[agent]
        centre = mode.informed.state_centre[agent]
        while True:
            x_ball = self.sample_unit_n_ball(n)  # Assumes this returns a tensor on the correct device
            x_rand = C @ (L @ x_ball) + centre
            # Check if x_rand is within limits
            if torch.all((lims[0] <= x_rand) & (x_rand <= lims[1])):
                return x_rand.cpu().numpy()

    def rotation_to_world_frame(self, x_start, x_goal, r):
        """Returns the norm and rotation matrix C from the hyperellipsoid-aligned frame to the world frame."""
        diff = x_goal - x_start
        norm = np.linalg.norm(diff)
        a1 = diff / norm  # Unit vector along the transverse axis

        # Create first column of the identity matrix
        n = self.env.robot_dims[r]
        e1 = np.zeros(n)
        e1[0] = 1

        # Compute SVD directly on the outer product
        U, _, Vt = np.linalg.svd(np.outer(a1, e1))
        V = Vt.T

        # Construct the rotation matrix C
        det_factor = np.linalg.det(U) * np.linalg.det(V)
        C = U @ np.diag([1] * (n - 1) + [det_factor]) @ Vt

        return norm, C

    def sample_unit_n_ball(self, n:int):
        """Returns uniform sample from the volume of an n-ball of unit radius centred at origin"""
        x_ball = torch.randn(n, device='cuda', dtype=torch.float64)
        x_ball /= torch.linalg.norm(x_ball, ord=2)
        radius = torch.rand(1, device='cuda', dtype=torch.float64).pow(1 / n)
        return x_ball * radius 

    def fit_to_informed_subset(self, N_near: List[Node], N_near_batch: torch.Tensor) -> List[Node]:
        if not N_near or N_near_batch.size(0) == 0:
            return N_near, N_near_batch

        final_valid_mask = torch.ones(len(N_near), device=N_near_batch.device, dtype=torch.bool)  # Start with all True

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
            return N_near, N_near_batch 

        valid_indices = final_valid_mask.nonzero(as_tuple=True)[0]
        N_near = [N_near[i] for i in valid_indices.tolist()]
        N_near_batch = N_near_batch[valid_indices]

        return N_near, N_near_batch

class Informed:

    def __init__(self):
        self.inv_C = {}
        self.C = {}
        self.L = {}
        self.cmin = {}
        self.state_centre = {}
        self.start = {}
        self.goal = {}

    def cmax(self, transition_nodes, path_nodes, goal_radius, agent, start):
        if transition_nodes == []:
            return
        else:
            for node in transition_nodes:
                if node in path_nodes:
                    c_agent = node.agent_cost[:,agent]
                    c_tot = node.cost
                    break
            c_start_agent, c_start_tot = self.get_start_node(path_nodes, goal_radius, agent, start)
            return c_tot, c_start_tot, c_agent - c_start_agent

    def get_start_node(self, path_nodes, goal_radius, agent, start):
        for node in path_nodes:
            if np.array_equal(node.state.q[agent], start) or np.linalg.norm(start - node.state.q[agent]) < goal_radius:
                return node.agent_cost[:,agent], node.cost 

