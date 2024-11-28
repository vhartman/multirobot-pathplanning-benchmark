import numpy as np
import math
from kdtree import kdtree
import time
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

def save_data(config, env, operation, passed_time, n_new=None, N_near=None, r=None, n_rand=None, n_rand_label=None, init_path=False):
    # Initialize data dictionary
    data = {}

    # Tree Data
    tree_data = [
        {
            "state": node.data.state.q.state(),
            "parent": node.data.parent.data.state.q.state() if node.data.parent else None,
            "mode": node.data.state.mode,
        }
        for node in operation.tree.inorder()
    ]

    # Path and Cost Data
    if operation.path_nodes:
        transition_node = operation.path_nodes[-1]
        path_data = [state.q.state() for state in operation.path]
        intermediate_tot = [node.data.cost for node in operation.path_nodes]
        intermediate_agent_cost = [node.data.agent_cost for node in operation.path_nodes]
        tot_cost = transition_node.data.cost
        agent_cost_data = transition_node.data.agent_cost
        is_transition = [node.data.transition for node in operation.path_nodes]
        modes = [node.data.state.mode for node in operation.path_nodes]
    else:
        path_data = None
        tot_cost = None
        agent_cost_data = None
        intermediate_tot = None
        intermediate_agent_cost = None
        is_transition = None
        modes = None


    result = {
        "path": path_data,
        "total": tot_cost,
        "agent_cost": agent_cost_data,
        "intermediate_tot": intermediate_tot,
        "intermediate_agent_cost": intermediate_agent_cost,
        "is_transition": is_transition,
        "modes": modes
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
        for mode in operation.modes
        if mode.informed.L != []
    ]

    # Nearby Nodes
    N_near_list = [N.state.q.state() for N in N_near] if N_near else []

    data.update({
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
    })

    # Directory 
    frames_directory = os.path.join(config.output_dir, 'FramesData')
    os.makedirs(frames_directory, exist_ok=True)

    # Find next available file number
    existing_files = (int(file.split('.')[0]) for file in os.listdir(frames_directory) if file.endswith('.pkl') and file.split('.')[0].isdigit())
    next_file_number = max(existing_files, default=-1) + 1

    # Save data as pickle file
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
            'informed_sampling': True, 'informed_sampling_version': None, 
            'cprofiler': False
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

class Node:
    def __init__(self, state:State):
        self.state = state   
        self.parent = None  
        self.children = []    
        self.transition = False
        self.cost = 0 
        self.agent_cost = [0] *state.q.num_agents()

    @property
    def coords(self):
        return tuple(self.state.q.state())
    
    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        return self.coords[i]

    def __repr__(self):
        return f"<Node - {self.coords}, Cost: {self.cost}>"
    
class Operation:
    """ Planner operation variables"""
    def __init__(self, env):
        self.modes = [Mode(env.start_mode, env)] # list of visited modes
        self.active_mode = None
        self.path = []
        self.path_nodes = None
        self.cost = 0
        self.tree = kdtree.create(dimensions=sum(env.robot_dims.values())) 
        self.tree_size = 0
        self.ptc_cost = 0 # needed for termination
        self.ptc_iter = None # needed for termination
        self.init_path = False

class Mode:
    """Mode description"""
    def __init__(self, mode, env):
        self.label = mode
        self.transition_nodes = []
        self.subtree = kdtree.create(dimensions=sum(env.robot_dims.values()))
        self.subtree_size = 0
        self.informed = Informed()

class Sampling:
    def __init__(self, env, operation):
        self.env = env
        self.operation = operation
        pass

    def sample_state(self, mode:Mode, sampling_type:int, config:ConfigManager) -> State: # or in rai_env?
        m = mode.label
        active_task = self.env.get_active_task(m)
        constrained_robtos = active_task.robots
        while True:
            q = []
            for i , robot in enumerate(self.env.robots):
                lims = self.env.limits[:, self.env.robot_idx[robot]]
                if i in mode.informed.goal.keys() and np.equal(mode.informed.goal[i], mode.informed.start[i]).all():
                    qr = mode.informed.goal[i] #mode restriction (agent stays at same place as in previous mode)
                elif sampling_type == 2:  #goal sampling
                    task = m[i]
                    if config.general_goal_sampling or not config.general_goal_sampling and robot in constrained_robtos:
                        goal = self.env.tasks[task].goal.sample()
                        qr = goal[self.env.robot_idx[robot]] if len(goal) != self.env.robot_dims[robot] else goal
                    else:
                        qr = np.random.uniform(lims[0], lims[1])
                elif sampling_type == 1: #informed sampling
                    qr = self.sample_informed(i, config.goal_radius)
                    if qr is None: 
                        qr = np.random.uniform(lims[0], lims[1])
                elif sampling_type == 0: #uniform sampling
                    qr = np.random.uniform(lims[0], lims[1]) 
                q.append(qr)
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
        if not mode:
            return  # Initial path not yet found

        cmax = mode.informed.cmax(mode.transition_nodes, self.operation.path_nodes, goal_radius, agent, mode.informed.start[agent])
        r1 = cmax / 2
        r2 = np.sqrt(cmax**2 - mode.informed.cmin[agent]**2) / 2
        
        # Debugging condition
        if abs(r1 - mode.informed.cmin[agent]) <= 8e-10:
            print(mode.informed.cmin[agent], cmax)

        agent_name = self.env.robots[agent]
        n = self.env.robot_dims[agent_name]
        mode.informed.L[agent] = np.diag([r1] + [r2] * (n - 1))

        lims = self.env.limits[:, self.env.robot_idx[agent_name]]
        L, C, centre = mode.informed.L[agent], mode.informed.C[agent], mode.informed.state_centre[agent]

        while True:
            x_ball = self.sample_unit_n_ball(n)
            x_rand = C @ (L @ x_ball) + centre
            if np.all((lims[0] <= x_rand) & (x_rand <= lims[1])):
                return x_rand

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
        # Sample a point from the normal distribution
        x_ball = np.random.normal(0, 1, n)
        # Normalize it to lie on the surface of the unit sphere
        x_ball /= np.linalg.norm(x_ball)
        # Scale it to a random radius within the unit ball
        radius = np.random.rand() ** (1 / n)
        return x_ball * radius

    def fit_to_informed_subset(self, N_near:List[Node])-> List[Node]:
        for r_idx, r in enumerate(self.env.robots):
            mode = self.mode_selection(r_idx)
            if not mode or r_idx not in mode.informed.goal or np.array_equal(mode.informed.goal[r_idx], mode.informed.start[r_idx]):
                continue
            if  mode.informed.L:
                indices = self.env.robot_idx[r]
                centre = mode.informed.state_centre[r_idx]
                L, C = mode.informed.L[r_idx], mode.informed.C[r_idx] 
                inv_C = np.linalg.inv(C)
                inv_L = np.linalg.inv(L)
                N_near_filtered = []
                for node in N_near:
                    scaled_node = inv_L @ (inv_C @ (node.state.q.state()[indices] - centre).reshape(-1))
                    if np.dot(scaled_node, scaled_node) <= 1:
                        N_near_filtered.append(node)
                N_near = N_near_filtered
        return N_near

class Informed:

    def __init__(self):
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
                    c = node.data.agent_cost[agent]
                    break
            return c-self.get_start_node(path_nodes, goal_radius, agent, start)

    def get_start_node(self, path_nodes, goal_radius, agent, start):
        for node in path_nodes:
            if np.array_equal(node.data.state.q[agent], start) or np.linalg.norm(start - node.data.state.q[agent])< goal_radius:
                return node.data.agent_cost[agent]
 
class RRTstar:
    def __init__(self, env, config: ConfigManager):
        self.env = env
        # self.gamma = ((2 *(1 + 1/self.dim))**(1/self.dim) * (self.FreeSpace()/self.UnitBallVolume())**(1/self.dim))*1.1 
        self.config = config
        self.r = self.config.step_size * sum(self.env.robot_dims.values()) # TODO
        self.operation = Operation(env)
        self.sampling = Sampling(env, self.operation)
        self.optimization_mode_idx = 0
        self.start = time.time()
        
    def Cost(self, n1, n2):
        n1_state = n1.data.state if hasattr(n1, 'data') else n1.state
        n2_state = n2.data.state if hasattr(n2, 'data') else n2.state
        # print(config_dist(n1_state.q, n2_state.q, "euclidean"))
        # print(self.env.state_cost(n1_state, n2_state))
        # return self.env.state_cost(n1_state, n2_state)
        return config_dist(n1_state.q, n2_state.q, "euclidean")

    def AgentDist(self, n1, n2): 
        n1_state = n1.data.state if hasattr(n1, 'data') else n1.state
        n2_state = n2.data.state if hasattr(n2, 'data') else n2.state
        return config_agent_dist(n1_state.q, n2_state.q, "euclidean")

    def StateDist(self, n1, n2):
        n1_state = n1.data.state.q.state() if hasattr(n1, 'data') else n1.state.q.state()
        n2_state = n2.data.state.q.state() if hasattr(n2, 'data') else n2.state.q.state()
        direction = n2_state - n1_state
        # d = config_dist(n1.data.state.q.state(), n2.data.state.q.state(), "euclidean")
        return np.linalg.norm(direction)
        # return state_dist(n1_state, n2_state)
    
    def Nearest(self, n_rand): 
        nearest_node, _ = self.operation.active_mode.subtree.search_nn(n_rand.data, self.StateDist)
        return  nearest_node
      
    def Steer(self,
        n_nearest: kdtree.KDNode,
        n_rand: kdtree.KDNode,
        m: List[int]
    ) -> bool:        
        dist = config_dist(n_nearest.data.state.q, n_rand.data.state.q, "euclidean") #TODO
        if dist <=self.config.step_size:
            return n_rand
        
        q_nearest = n_nearest.data.state.q.state()
        q_rand = n_rand.data.state.q.state()
        direction = q_rand - q_nearest
        norm = np.linalg.norm(direction)
        N = int(np.ceil(norm / self.config.step_size))
        t = 1 / N
        q = q_nearest + t * direction
        q = np.clip(q, self.env.limits[0], self.env.limits[1]) 
        state_new = State(type(self.env.get_start_pos())(q, n_nearest.data.state.q.slice), m)
        n_new = Node(state_new)
        return kdtree.KDNode(data=n_new, dimensions=sum(self.env.robot_dims.values()))
  
    def Near(self, n_new):
        # n_nodes = sum(1 for _ in self.operation.current_mode.subtree.inorder()) + 1
        #TODO generalize the radius!!!!!
        # r = min((7)*self.step_size, 3 + self.gamma * ((math.log(n_nodes) / n_nodes) ** (1 / self.dim)))
        N_near = self.operation.active_mode.subtree.search_nn_dist(n_new.data, self.r, dist =  self.StateDist) 
        if not self.config.informed_sampling:
            return N_near
        # else:
        #     if self.operation.active_mode.label == self.operation.modes[-1].label and not self.operation.init_path:
        #     # if self.operation.task_sequence != [] and self.operation.task_sequence[0] == self.operation.current_mode.constraint.label:
        #         return N_near
        #     #N_near needs to be preselected
        return self.sampling.fit_to_informed_subset(N_near)        
    
    def FindParent(self, N_near, n_new, n_nearest):
        c_min = n_nearest.data.cost + self.Cost(n_nearest, n_new)
        n_min = n_nearest.data
        for n_near in N_near:
            c_new = n_near.cost + self.Cost(n_near, n_new.data)
            if c_new < c_min :
                if self.env.is_edge_collision_free(n_near.state.q, n_new.data.state.q, self.operation.active_mode.label):
                    c_min = c_new
                    n_min = n_near
        n_new.data.cost = c_min
        n_new.data.parent = kdtree.KDNode(data=n_min, dimensions=len(n_min.coords))
        n_min.children.append(n_new.data) #Set child
        n_new.data.agent_cost = n_new.data.parent.data.agent_cost + self.AgentDist(n_new.data.parent.data, n_new.data)
        self.operation.tree.add(n_new.data)
        self.operation.active_mode.subtree.add(n_new.data)
        self.operation.tree_size+=1
        self.operation.active_mode.subtree_size +=1

    def UnitBallVolume(self):
        return math.pi ** (self.dim / 2) / math.gamma((self.dim / 2) + 1)

    def Rewire(self, N_near, n_new, costs_before):
        rewired = False
        for n_near in N_near:
            costs_before.append(n_near.cost)
            if n_near in {n_new.data, n_new.data.parent.data}:
                continue       
            c_pot = n_new.data.cost + self.Cost(n_new.data, n_near)
            if c_pot < n_near.cost:
                if  self.env.is_edge_collision_free(n_near.state.q, n_new.data.state.q, self.operation.active_mode.label):
                    c_agent = n_new.data.agent_cost + self.AgentDist(n_new.data, n_near)
                    #reset children
                    n_near.parent.data.children.remove(n_near)
                    #set parent
                    n_near.parent = n_new
                    if n_new.data != n_near:
                        n_new.data.children.append(n_near)
                    n_near.cost = c_pot
                    n_near.agent_cost = c_agent
                    rewired = True
        return rewired
    
    def GeneratePath(self, node):
        path_nodes, path = [], []
        while node:
            path_nodes.append(node)
            path.append(node.data.state)
            node = node.data.parent
        path_in_order = path[::-1]
        # if self.env.is_valid_plan(path_in_order): 
        self.operation.path = path_in_order  
        # print([state.q.state() for state in path_in_order])
        self.operation.path_nodes = path_nodes[::-1]
        self.operation.cost = self.operation.path_nodes[-1].data.cost

    def UpdateCost(self, n):
        stack = [n]
        while stack:
            current_node = stack.pop()
            n_agent_cost = current_node.agent_cost
            for child in current_node.children:
                child.cost = current_node.cost + self.Cost(current_node, child)
                child.agent_cost = n_agent_cost + self.AgentDist(current_node, child)
                stack.append(child)
   
    def FindOptimalTransitionNode(self, iteration):
        transition_nodes = self.operation.modes[-1].transition_nodes
        lowest_cost = np.inf
        lowest_cost_idx = None
        for idx, node in enumerate(transition_nodes):
            if node.data.cost < lowest_cost and node.data.cost < self.operation.cost:
                lowest_cost = node.data.cost
                lowest_cost_idx = idx
        if lowest_cost_idx is not None: 
            self.GeneratePath(transition_nodes[lowest_cost_idx]) 
            print(f"iter  {iteration}: Changed cost to ", self.operation.cost, " Mode ", self.operation.active_mode.label)
            if (self.operation.ptc_cost - self.operation.cost) > self.config.ptc_threshold: #TODO
                self.operation.ptc_cost = self.operation.cost
                self.operation.ptc_iter = iteration

            end_iteration = time.time()
            passed_time = end_iteration - self.start
            save_data(self.config, self.env, self.operation, passed_time) 

    def AddTransitionNode(self, n):
            idx = self.operation.modes.index(self.operation.active_mode)
            if idx != len(self.operation.modes) - 1:
                self.operation.modes[idx + 1].subtree.add(n.data)
                self.operation.modes[idx+1].subtree_size +=1
                n.transition = True
    
    def ManageTransition(self, n_new, iteration):
        constrained_robots = self.env.get_active_task(self.operation.active_mode.label).robots
        indices = []
        radius = 0
        for r in constrained_robots:
            indices.extend(self.env.robot_idx[r])
            radius += self.config.goal_radius
        if self.env.get_active_task(self.operation.active_mode.label).goal.satisfies_constraints(n_new.data.state.q.state()[indices], self.config.goal_radius):
            self.operation.active_mode.transition_nodes.append(n_new)
            n_new.data.transition = True
            # Check if initial transition node of current mode is found
            if self.operation.active_mode.label == self.operation.modes[-1].label and not self.operation.init_path:
            # if self.operation.task_sequence != [] and self.operation.active_mode.constraint.label == self.operation.task_sequence[0]:
                print(f"iter  {iteration}: {constrained_robots} found T{self.env.get_current_seq_index(self.operation.active_mode.label)}: Cost: ", n_new.data.cost)
                # if self.operation.task_sequence != []:
                self.InitializationMode(self.operation.modes[-1])
                if self.env.terminal_mode != self.operation.modes[-1].label:
                    self.operation.modes.append(Mode(self.env.get_next_mode(n_new.data.state.q,self.operation.active_mode.label), self.env))
                elif self.operation.active_mode.label == self.env.terminal_mode:
                    self.operation.ptc_iter = iteration
                    self.operation.ptc_cost = n_new.data.cost
                    self.operation.init_path = True
                
                self.GeneratePath(n_new)
                end_iteration = time.time()
                passed_time = end_iteration - self.start
                save_data(self.config, self.env, self.operation, passed_time, init_path=self.operation.init_path)
            self.AddTransitionNode(n_new)
        self.FindOptimalTransitionNode(iteration)

    def SetModePorbability(self):
        num_modes = len(self.operation.modes)
        if num_modes == 1:
            return [1] 
        # if self.operation.task_sequence == [] and self.config.mode_probability != 0:
        if self.env.terminal_mode == self.operation.modes[-1].label and self.config.mode_probability != 0:
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
            total_nodes = self.operation.tree_size + total_transition_nodes
            # Calculate probabilities inversely proportional to node counts
            inverse_probabilities = [
                1 - (mode.subtree_size / total_nodes)
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
                1 - (mode.subtree_size / total_nodes)
                for mode in self.operation.modes[:-1]  # Exclude the last mode
            ]

            # Normalize the probabilities of all modes except the last one
            remaining_probability = 1-self.config.mode_probability  
            total_inverse = sum(inverse_probabilities)
            return [
                (inv_prob / total_inverse) * remaining_probability
                for inv_prob in inverse_probabilities
            ] + [self.config.mode_probability]

    def InitializationMode(self, m:Mode):
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
                
                if m.label == mode:
                    m.informed.start[r] = state_start[r]
                    m.informed.goal[r] = goal[r]
                    if self.config.informed_sampling and not np.equal(m.informed.goal[r], m.informed.start[r]).all():
                        cmin, C = self.sampling.rotation_to_world_frame(state_start[r], goal[r] ,robot)
                        m.informed.C[r] = C
                        m.informed.cmin[r] = cmin-2*self.config.goal_radius
                        m.informed.state_centre[r] = ((state_start[r] + goal[r])/2)

                    if robot == constrained_robtos[-1]:
                        return
            mode = self.env.get_next_mode(None,mode)
                
    def SampleNodeManifold(self, operation: Operation):
        # if np.random.uniform(0, 1) <= self.p_goal: #TODO
        if np.random.uniform(0, 1) <= 0.95:
            if self.env.terminal_mode != operation.modes[-1].label and operation.active_mode.label == operation.modes[-1].label:
            # if operation.task_sequence != [] and operation.task_sequence[0] == operation.current_mode.constraint.label: # initial path not yet found -> sample uniformly
                state = self.sampling.sample_state(operation.active_mode, 0, self.config) 
            else:   
                state = self.sampling.sample_state(operation.active_mode, 1, self.config)
        else:
            state = self.sampling.sample_state(operation.active_mode, 2, self.config)

        n_rand = Node(state)
        return kdtree.KDNode(data=n_rand, dimensions=len(n_rand.coords))

    def Plan(self) -> dict:
        i = 0
        while True:
            # Mode selection
            self.operation.active_mode  = (np.random.choice(self.operation.modes, p= self.SetModePorbability()))
            # InitializationMode of active mode
            if  self.operation.tree.data is None:
                start_state = State(self.env.start_pos, self.operation.active_mode.label)
                start_node = Node(start_state)
                self.operation.tree.add(start_node) 
                self.operation.active_mode.subtree.add(start_node) 
                self.operation.tree_size+=1
                self.operation.active_mode.subtree_size +=1

            # RRT* core
            n_rand = self.SampleNodeManifold(self.operation)
            n_nearest = self.Nearest(n_rand)    
            # self.env.C.setJointState(n_rand.data.state.q.state())
            # self.env.show()  
            n_new = self.Steer(n_nearest, n_rand, self.operation.active_mode.label)
            # self.env.C.setJointState(n_new.data.state.q.state())
            # self.env.show()

            if self.env.is_collision_free(n_new.data.state.q.state(), self.operation.active_mode.label) and self.env.is_edge_collision_free(n_nearest.data.state.q, n_new.data.state.q, self.operation.active_mode.label):
                N_near = self.Near(n_new)
                self.FindParent(N_near, n_new, n_nearest)
                costs_before = []

                if self.Rewire(N_near, n_new, costs_before):
                    self.UpdateCost(n_new.data)
                     
                self.ManageTransition(n_new, i)
                if i%200 == 0:
                    if i%400 == 0:
                        print("iter ",i)
                    end_iteration = time.time()
                    passed_time = end_iteration - self.start
                    save_data(self.config, self.env, self.operation, passed_time, n_new = n_new.data, N_near = N_near, r =self.r, n_rand = n_rand.data.state.q)
            if self.operation.init_path and i != self.operation.ptc_iter and (i- self.operation.ptc_iter)%self.config.ptc_max_iter == 0:
                diff = self.operation.ptc_cost - self.operation.cost
                self.operation.ptc_cost = self.operation.cost
                if diff < self.config.ptc_threshold:
                    break
            i += 1

        end_iteration = time.time()
        passed_time = end_iteration - self.start
        save_data(self.config, self.env, self.operation, passed_time, n_new = n_new.data, N_near = N_near, r =self.r, n_rand = n_rand.data.state.q)
        return self.operation.path    




