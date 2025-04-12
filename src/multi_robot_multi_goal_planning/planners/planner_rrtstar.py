import numpy as np
import time as time
import math as math
from typing import Tuple, Optional, Union, List, Dict
from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
    Mode
)
from multi_robot_multi_goal_planning.planners.rrtstar_base import (
    BaseRRTstar, 
    Node, 
    SingleTree

)
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
)

class RRTstar(BaseRRTstar):
    """Represents the class for the RRT* based planner"""
    def __init__(self, 
                 env: BaseProblem,
                 ptc: PlannerTerminationCondition,  
                 general_goal_sampling: bool = False, 
                 informed_sampling: bool = False, 
                 informed_sampling_version: int = 6, 
                 distance_metric: str = 'max_euclidean',
                 p_goal: float = 0.1, 
                 p_stay: float = 0.0,
                 p_uniform: float = 0.2, 
                 shortcutting: bool = False, 
                 mode_sampling: Optional[Union[int, float]] = None, 
                 sample_near_path: bool = False,
                 locally_informed_sampling:bool = True, 
                 remove_redundant_nodes:bool = True,
                 informed_batch_size: int = 500,
                 test_mode_sampling:bool = False
                 
                ):
        super().__init__(env = env, ptc = ptc, general_goal_sampling = general_goal_sampling, informed_sampling = informed_sampling, 
                         informed_sampling_version = informed_sampling_version, distance_metric = distance_metric,
                         p_goal = p_goal, p_stay = p_stay, p_uniform = p_uniform, shortcutting = shortcutting, mode_sampling = mode_sampling, 
                         sample_near_path = sample_near_path, locally_informed_sampling = locally_informed_sampling, remove_redundant_nodes = remove_redundant_nodes, 
                         informed_batch_size = informed_batch_size, test_mode_sampling = test_mode_sampling )
     
    def UpdateCost(self, mode:Mode, n:Node) -> None:
        stack = [n]
        while stack:
            current_node = stack.pop()
            children = current_node.children
            if children:
                for _, child in enumerate(children):
                    child.cost = current_node.cost + child.cost_to_parent
                    # child.agent_dists = current_node.agent_dists + child.agent_dists_to_parent
                stack.extend(children)
   
    def ManageTransition(self, mode:Mode, n_new: Node) -> None:
        #check if transition is reached
        if self.env.is_transition(n_new.state.q, mode):
            self.add_new_mode(n_new.state.q, mode, SingleTree)
            self.convert_node_to_transition_node(mode, n_new)
        #check if termination is reached
        if self.env.done(n_new.state.q, mode):
            self.convert_node_to_transition_node(mode, n_new)
            if not self.operation.init_sol:
                self.operation.init_sol = True
        self.FindLBTransitionNode()
 
    def PlannerInitialization(self) -> None:
        self.set_gamma_rrtstar()
        # Initilaize first Mode
        self.add_new_mode(tree_instance=SingleTree)
        active_mode = self.modes[-1]
        # Create start node
        start_state = State(self.env.start_pos, active_mode)
        start_node = Node(start_state, self.operation)
        self.trees[active_mode].add_node(start_node)
        start_node.cost = 0.0
        start_node.cost_to_parent = 0.0
    
    def Plan(self, optimize:bool=True) ->  Tuple[List[State], Dict[str, List[Union[float, float, List[State]]]]]:
        i = 0
        self.PlannerInitialization()
        while True:
            i += 1
            # Mode selectiom       
            active_mode  = self.RandomMode()
            # RRT* core
            q_rand = self.SampleNodeManifold(active_mode)
            if not q_rand:
                continue
            n_nearest, dist, set_dists, n_nearest_idx = self.Nearest(active_mode, q_rand)    
            state_new = self.Steer(active_mode, n_nearest, q_rand, dist)
            if not state_new:
                continue
            if self.env.is_collision_free(state_new.q, active_mode) and self.env.is_edge_collision_free(n_nearest.state.q, state_new.q, active_mode):
                n_new = Node(state_new, self.operation)
                if np.equal(n_new.state.q.state(), q_rand.state()).all():
                    N_near_batch, n_near_costs, node_indices = self.Near(active_mode, n_new, n_nearest_idx, set_dists)
                else:
                    N_near_batch, n_near_costs, node_indices = self.Near(active_mode, n_new, n_nearest_idx)
                batch_cost = self.env.batch_config_cost(n_new.state.q, N_near_batch)
                self.FindParent(active_mode, node_indices, n_new, n_nearest, batch_cost, n_near_costs)
                if self.Rewire(active_mode, node_indices, n_new, batch_cost, n_near_costs):
                    self.UpdateCost(active_mode, n_new) 
                self.ManageTransition(active_mode, n_new)

            if not optimize and self.operation.init_sol:
                break

            if self.ptc.should_terminate(i, time.time() - self.start_time):
                print('Number of iterations: ', i)
                break
        self.costs.append(self.operation.cost)
        self.times.append(time.time() - self.start_time)
        self.all_paths.append(self.operation.path)
        info = {"costs": self.costs, "times": self.times, "paths": self.all_paths}
        # print('Path is collision free:', self.env.is_path_collision_free(self.operation.path))
        return self.operation.path, info    