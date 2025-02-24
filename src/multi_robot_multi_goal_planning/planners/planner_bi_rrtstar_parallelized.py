import numpy as np
import time as time
import math as math
from typing import Optional, Union
from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
    Mode
)
from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration
)
from multi_robot_multi_goal_planning.planners.planner_birrtstar import (
    BidirectionalRRTstar
)

from multi_robot_multi_goal_planning.planners.rrtstar_base import (
    Node, 
    BidirectionalTree,
    SingleTree

)
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
)


class ParallelizedBidirectionalRRTstar(BidirectionalRRTstar):
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
                 shortcutting: bool = False, 
                 mode_sampling: Optional[Union[int, float]] = None, 
                 gaussian: bool = False, 
                 transition_nodes: int = 50, 
                 birrtstar_version: int = 2 
                ):
        super().__init__(env, ptc, general_goal_sampling, informed_sampling, informed_sampling_version, distance_metric,
                    p_goal, p_stay, p_uniform, shortcutting, mode_sampling, 
                    gaussian)
        self.transition_nodes = transition_nodes 
        self.birrtstar_version = birrtstar_version

    def add_new_mode(self, q:Optional[Configuration]=None, mode:Mode=None, tree_instance: Optional[Union["SingleTree", "BidirectionalTree"]] = None) -> None: 
        if mode is None: 
            new_mode = self.env.make_start_mode()
            new_mode.prev_mode = None
            self.modes.append(new_mode)
            self.add_tree(new_mode, tree_instance)
            self.InformedInitialization(new_mode)
            return
        else:
            new_mode = self.env.get_next_mode(q, mode)
            new_mode.prev_mode = mode
        if new_mode in self.modes:
            return 
        self.modes.append(new_mode)
        self.add_tree(new_mode, tree_instance)
        self.InformedInitialization(new_mode)
        i = 0
        while i < self.transition_nodes:               
            q = self.sample_transition_configuration(mode)
            node = Node(State(q, mode), self.operation)
            if node in self.trees[mode].subtree.values(): #prevent to have several similar nodes
                continue
            node.cost_to_parent = 0.0
            self.convert_node_to_transition_node(mode, node)
            self.trees[mode].add_node(node, 'B')
            self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, node.id) 
            node.cost = np.inf
            i += 1


    def PlannerInitialization(self) -> None:
        # Initilaize first Mode
        self.add_new_mode(tree_instance=BidirectionalTree)
        mode = self.modes[-1]
        # Create start node
        start_state = State(self.env.start_pos, mode)
        start_node = Node(start_state, self.operation)
        self.trees[mode].add_node(start_node)
        start_node.cost = 0
        start_node.cost_to_parent = 0.0
        #Initialize other modes:
        while True:
            self.InformedInitialization(mode)
            if not self.env.is_terminal_mode(mode): 
                self.add_new_mode(mode=mode, tree_instance=BidirectionalTree) 
            else:
                break 
          
    def Connect(self, mode:Mode, n_new:Node, iter:int) -> None:
        if not self.trees[mode].subtree_b:
            return
        n_nearest_b, dist = self.Nearest(mode, n_new.state.q, 'B')

        if self.birrtstar_version == 1 or self.birrtstar_version == 2 and self.trees[mode].connected: #Based on paper Bi-RRT* by B. Wang 
            #TODO only check dist of active robots to connect (cost can be extremly high)? or the smartest way to just connect when possible?
          
          
            cost =  self.env.batch_config_cost([n_new.state], [n_nearest_b.state])
            # relevant_dists = []
            # for r_idx, r in enumerate(self.env.robots):
            #     if r in constrained_robots:
            #         relevant_dists.append(dists[0][r_idx].item())
            # if np.max(relevant_dists) > self.step_size:
            #     return

            if np.max(dist) > self.step_size:
                return

            if not self.env.is_edge_collision_free(n_new.state.q, n_nearest_b.state.q, mode): #ORder rigth? TODO
                return
          

        elif self.birrtstar_version == 2 and not self.trees[mode].connected: #Based on paper RRT-Connect by JJ. Kuffner/ RRT*-Connect by S.Klemm
            n_nearest_b = self.Extend(mode, n_nearest_b, n_new, dist)
            if not n_nearest_b:
                return
            cost =  self.env.batch_config_cost([n_new.state], [n_nearest_b.state])
            
 
        if self.trees[mode].order == -1:
            #switch such that subtree is beginning from start and subtree_b from goal
            self.trees[mode].swap()
            self.UpdateTree(mode, n_new, n_nearest_b, cost[0]) 
        else:
            self.UpdateTree(mode, n_nearest_b, n_new, cost[0])
    
    def GeneratePath(self, mode:Mode, n: Node, iter:int = None, inter:bool = False, shortcutting:bool = True) -> None:
        path_nodes, path = [], []
        while n:
            path_nodes.append(n)
            path.append(n.state)
            n = n.parent
        path_in_order = path[::-1]
        self.operation.path = path_in_order  
        self.operation.path_nodes = path_nodes[::-1]
        # if inter:
        #     #check if termination is reached
        #     if self.env.done(path_nodes_[-1].state.q, self.modes[-1]):
        #         if path_nodes_[-1].id not in self.transition_node_ids[self.modes[-1]]:
        #             self.mark_node_as_transition(self.modes[-1], path_nodes_[-1])
        #     #intermediate solutions
        #     if not self.operation.init_sol and path_nodes_ not in self.operation.paths_inter:
        #         self.operation.paths_inter.append(path_nodes_) 
        #     return
            
        #check if terminal node has been reached

        if shortcutting and self.start_single_goal.satisfies_constraints(self.operation.path_nodes[0].state.q.state(), mode, self.env.collision_tolerance):
            if not self.operation.init_sol:
                print(time.time()-self.start_time)
                self.operation.init_sol = True
            self.operation.cost = self.operation.path_nodes[-1].cost
            self.costs.append(self.operation.cost)
            self.times.append(time.time()-self.start_time)
            if self.shortcutting:
                print("-- M", mode.task_ids, "Cost: ", self.operation.cost)
                self.Shortcutting(mode)
            print(f"{iter} M", mode.task_ids, "Cost: ", self.operation.cost)
        if self.shortcutting and not shortcutting:
            self.operation.cost = self.operation.path_nodes[-1].cost
            self.costs.append(self.operation.cost)
            self.times.append(time.time()-self.start_time)
                 
    def ManageTransition(self, mode:Mode, n_new:Node, iter:int) -> None:
        #if tree is in the right order
        if self.trees[mode].order == 1: 
            #Check if transition is reached
            if self.env.is_transition(n_new.state.q, mode):
                self.convert_node_to_transition_node(mode, n_new)
            #Check if termination is reached
            if self.env.done(n_new.state.q, mode):
                self.convert_node_to_transition_node(mode, n_new)

        else:
            if mode.prev_mode and self.env.is_transition(n_new.state.q, mode.prev_mode):
                n_new.state.mode = mode.prev_mode
                n_new_parent = n_new.parent
                n_new.parent.children.remove(n_new)
                cost = n_new.cost_to_parent
                n_new.parent = None
                n_new.cost_to_parent = 0.0
                n_new.cost = np.inf
                self.trees[mode].swap()
                self.convert_node_to_transition_node(mode.prev_mode, n_new)
                self.trees[mode].remove_node(n_new, 'B')
                self.UpdateTree(mode, n_new_parent, n_new, cost)
                #need to add node to previous mode
                if self.trees[mode.prev_mode].order == 1:
                    self.trees[mode.prev_mode].add_node(n_new, 'B')
                else:
                    self.trees[mode.prev_mode].add_node(n_new, 'A')
            #need to handle first mode separately
            elif self.start_single_goal.satisfies_constraints(n_new.state.q.state(),mode, self.env.collision_tolerance):
                    n_new_parent = n_new.parent
                    n_new.parent.children.remove(n_new)
                    cost = n_new.cost_to_parent
                    n_new.parent = None
                    n_new.cost_to_parent = 0.0
                    n_new.cost = 0
                    self.trees[mode].swap()
                    self.convert_node_to_transition_node(mode.prev_mode, n_new)
                    self.trees[mode].remove_node(n_new, 'B')
                    self.UpdateTree(mode, n_new_parent, n_new, cost)
        self.FindLBTransitionNode(mode, iter)
            
    def FindLBTransitionNode(self, mode:Mode, iter: int) -> None:
        #check if termination node is reached (cost < np.inf)
        result = self.get_lb_transition_node_id(self.modes[-1])
        if not result:
            return 
        valid_mask = result[0] < self.operation.cost
        if valid_mask.any():
            lb_transition_node = self.get_transition_node(self.modes[-1], result[1])
            self.GeneratePath(mode, lb_transition_node, iter, shortcutting=True) 

    def Plan(self) -> dict:
        i = 0
        self.PlannerInitialization()
        while True:
            i += 1
            # Mode selection
            active_mode  = self.RandomMode(Mode.id_counter)
            
            # Bi-RRT* parallelized core
            q_rand = self.SampleNodeManifold(active_mode)
            if not self.trees[active_mode].subtree or  self.operation.init_sol and self.trees[active_mode].order == -1:
                self.trees[active_mode].swap()
                self.trees[active_mode].connected = True
                    
            n_nearest, dist = self.Nearest(active_mode, q_rand)        
            state_new = self.Steer(active_mode, n_nearest, q_rand, dist)
            if not state_new: # meaning n_new is exact the same as one of the nodes in the tree
                continue
            
            if self.env.is_collision_free(state_new.q, active_mode) and self.env.is_edge_collision_free(n_nearest.state.q, state_new.q, active_mode):
                n_new = Node(state_new, self.operation)
                N_near_batch, n_near_costs, node_indices = self.Near(active_mode, n_new)
                if n_nearest.id not in node_indices:
                    continue

              
                batch_cost = self.env.batch_config_cost(n_new.state.q, N_near_batch)
                self.FindParent(active_mode, node_indices, n_new, n_nearest, batch_cost, n_near_costs)
                # if self.operation.init_sol:
                if self.operation.init_sol:
                    if self.Rewire(active_mode, node_indices, n_new, batch_cost, n_near_costs):
                        self.UpdateCost(active_mode,n_new)
                self.Connect(active_mode, n_new, i)
                self.ManageTransition(active_mode, n_new, i)
            self.trees[active_mode].swap()
            if self.ptc.should_terminate(i, time.time() - self.start_time):
                break
        self.costs.append(self.operation.cost)
        self.times.append(time.time() - self.start_time)
        self.all_paths.append(self.operation.path)
        info = {"costs": self.costs, "times": self.times, "paths": self.all_paths}
        return self.operation.path, info      
         



