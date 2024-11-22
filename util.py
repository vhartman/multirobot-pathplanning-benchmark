import numpy as np
from numpy.typing import NDArray
from planning_env import *
from rai_envs import *

from typing import List

from jax import jit
import pickle
import os

def state_dist(start: State, end: State) -> float:
    if start.mode != end.mode:
        return np.inf

    return config_dist(start.q, end.q)


def state_cost(start: State, end: State) -> float:
    if start.mode != end.mode:
        return np.inf

    return config_cost(start.q, end.q)


def path_cost(path: List[State]) -> float:
    cost = 0

    batch_costs = batch_config_cost(path[:-1], path[1:])
    return np.sum(batch_costs)
    # print("A")
    # print(batch_costs)

    # costs = []

    for i in range(len(path) - 1):
        # costs.append(float(config_cost(path[i].q, path[i + 1].q)))
        cost += config_cost(path[i].q, path[i + 1].q)

    # print(costs)

    return cost

class Analysis:
    def __init__(self, env, config, directory, cost_function):
        self.env = env
        self.config = config
        self.debugging = self.config.get('debugging')
        self.directory = directory
        self.log_file_path = None
        self.frame_count = 0 
        self.agent_cost_data = {i: [] for i in range(len(self.env.agents))}
        self.tot_cost = []
        self.iter = []
        self.cost_function = cost_function

    def SavePklFIle(self, tree, operation, colors, passed_time, num_DOF, n_new= None, N_near = None, r=None, n_rand = None, n_rand_label = None, init_path = False):
        data = {}

        path = operation.path_nodes
        
        #Environment
        if self.frame_count == 0:
            mode_sequence = operation.mode_sequence
            env_data = {
                "collision_possibility": self.env.collision_possibility,
                "grid_padding": self.env.grid_padding,
                "grid_size": self.env.grid_size,
                "colors": colors,
                "agents": [],  
                "obstacles": [],  
                "num_DOF": num_DOF
            }
            #Agents
            for agent in self.env.agents:
                agent_dict = {
                    "start": agent.start,
                    "tasks": {k: v for k, v in agent.tasks.items()},
                    "color": agent.color,
                    "radius": agent.dimension,
                    "name": agent.name
                }
                env_data["agents"].append(agent_dict)
            #Modes
            modesequence = []
            for mode in mode_sequence:
                modesequence.append(mode.label)
            #Obstacles
            for obstacle in self.env.obstacles:
                obstacle_dict = {"coords": list(obstacle.geometry.exterior.coords)} 
                env_data["obstacles"].append(obstacle_dict)
            data.update({
                "env": env_data,  
                "colors": colors,
                "modes" : modesequence,
                "cost_function": self.cost_function
            })

        #Tree
        tree_data = []
        for node in tree.inorder():
            node_dict = {
                "state": node.data.state, 
                "parent": node.data.parent.data.state if node.data.parent else None,
                "mode": node.data.mode.label
            }
            tree_data.append(node_dict)
        #Path
        if operation.path is not None and any(v is not None for v in operation.path.values()):
            path_data = {k: [list(node) for node in v] for k, v in operation.path.items() if v is not None}
        else:
            path_data = {}  
        #Cost
        agent_cost_data = []
        if path is not None: 
            transition_node = path[-1]
            cost = transition_node.data.cost
            for agent in self.env.agents:
                agent_cost_dict = {
                    "cost": transition_node.data.agent_cost[agent.name]
                }
                agent_cost_data.append(agent_cost_dict)
        else:
            cost = None
            for agent in self.env.agents:
                agent_cost_dict = {
                    "cost": None
                }
                agent_cost_data.append(agent_cost_dict)

        # For each mode there only exists one informed sampling ...
        informed_sampling = []
        for mode in operation.modes:
            if mode.constraint.L is not None: 
                informed = {
                    "C": mode.constraint.C,
                    "L": mode.constraint.L,
                    "center": mode.constraint.state_centre,
                    "start": mode.constraint.start.state[next(iter(mode.constraint.label))],
                    "goal": mode.constraint.transition.state[next(iter(mode.constraint.label))]
                }
                informed_sampling.append(informed)
                
        N_near_list = []
        if N_near is not None:
            n_new = n_new.state
            for i, N in enumerate(N_near):
                N_near_dict = {}
                for agent in self.env.agents:
                    N_near_dict[agent.name] = N.state[agent.name]
                N_near_list.append(N_near_dict)
        # Organize data into a dictionary for pickling
        data.update({
            "tree": tree_data,
            "path": path_data,
            "agent_cost": agent_cost_data,
            "cost": cost,
            "all_init_path": init_path,
            "time": passed_time,
            "informed_sampling": informed_sampling, 
            "n_rand": n_rand,
            "n_rand_label": n_rand_label, 
            "N_near": N_near_list, 
            "rewire_r": r,
            "n_new": n_new, 
            "current_mode": operation.current_mode.label
        })

        os.makedirs(self.directory + 'FramesData/', exist_ok=True)
        filename = f"{self.directory}FramesData/{self.frame_count:04d}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        self.frame_count += 1
