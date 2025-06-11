import numpy as np
import time as time
import math as math
from typing import Tuple, Optional, Union, List, Dict, Any
from numpy.typing import NDArray
from numba import njit

from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
    Mode
)
from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration
)

from multi_robot_multi_goal_planning.planners.rrtstar_base import (
    BaseRRTConfig,
    BaseRRTstar, 
    Node, 
    BidirectionalTree,
    save_data

)
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
)

@njit
def compute_child_costs(parent_cost, cost_to_parents):
    return parent_cost + cost_to_parents
class BidirectionalRRTstar(BaseRRTstar):
    """Represents the class for the Bidirectional RRT* based planner"""

    def __init__(self, 
                 env: BaseProblem,
                 config: BaseRRTConfig
                ):
        
        super().__init__(env=env, config=config)
        self.swap = True
       
    def update_cost(self, mode:Mode, n:Node, connection:bool = False) -> None:
        stack = [n]
        while stack:
            current_node = stack.pop()
            if connection:
                #add node to main tree and delte it from subtree b
                if mode.__eq__(current_node.state.mode):
                    if current_node.id in self.trees[mode].get_node_ids_subtree('B'):
                        self.trees[mode].add_node(current_node)
                        self.trees[mode].remove_node(current_node, 'B')

            children = current_node.children
            if children:
                for _, child in enumerate(children):
                    child.cost = current_node.cost + child.cost_to_parent                    
                stack.extend(children)

    def add_new_mode(self, 
                     q:Optional[Configuration]=None, 
                     mode:Mode=None, 
                     tree_instance: BidirectionalTree = None
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
            new_modes = [self.env.get_start_mode()]
        else:
            new_modes = self.env.get_next_modes(q, mode)
            new_modes = self.mode_validation.get_valid_modes(mode, tuple(new_modes))
            if new_modes == []:
                self.modes = self.mode_validation.track_invalid_modes(mode, self.modes)
        for new_mode in new_modes:
            if new_mode in self.modes:
                continue 
            if new_mode in self.blacklist_mode:
                continue
            self.modes.append(new_mode)
            self.add_tree(new_mode, tree_instance)
            if self.config.informed_sampling_version != 6:
                self.initialize_informed_sampling(new_mode)
            #Initialize transition nodes
            node = None
            for i in range(self.config.transition_nodes):    
                q = self.sample_transition_configuration(new_mode)
                if q is None:
                    if new_mode in self.modes:
                        self.modes.remove(new_mode)
                    break
                if i > 0 and np.equal(q.state(), node.state.q.state()).all():
                    break
                node = Node(State(q, new_mode), self.operation)
                node.cost_to_parent = 0.0
                self.mark_node_as_transition(new_mode, node)
                self.trees[new_mode].add_node(node, 'B')
                self.operation.costs = self.trees[new_mode].ensure_capacity(self.operation.costs, node.id) 
                node.cost = np.inf

    def manage_transition(self, mode:Mode, n_new: Node) -> None:
        if mode not in self.modes:
            return
        #check if transition is reached
        if self.trees[mode].order == 1:
            if n_new.id in self.trees[mode].subtree:
                if self.env.is_transition(n_new.state.q, mode):
                    self.trees[mode].connected = True
                    self.save_tree_data()
                    self.add_new_mode(n_new.state.q, mode, BidirectionalTree)
                    self.save_tree_data()
                    self.convert_node_to_transition_node(mode, n_new)
                #check if termination is reached
                if self.env.done(n_new.state.q, mode):
                    self.trees[mode].connected = True
                    self.convert_node_to_transition_node(mode, n_new)
                    if not self.operation.init_sol:
                        # print(time.time()-self.start_time)
                        self.operation.init_sol = True
        self.find_lb_transition_node()

    def update_tree(self, 
                   mode:Mode,
                   n: Node, 
                   n_parent:Node, 
                   cost_to_parent:NDArray
                   ) -> None: 
        """
        Updates tree by transferring nodes from the reversed growing subtree (subtree_b) to the primary subtree, adjusting parent-child relationships and propagating cost updates.

        Args:
            mode (Mode): Current operational mode for which the tree update is performed.
            n (Node): Node whose parent and cost values are being updated.
            n_parent (Node): New parent node for n.
            cost_to_parent (NDArray): Cost of the edge connecting n_parent to n.

        Returns:
            None: This method does not return any value.
        """

        while True:
            n.cost = n_parent.cost +  cost_to_parent
            # dist_cpu= dists.clone().to(dtype=torch.float16).cpu()
            # n.agent_dists = n_parent.agent_dists + dist_cpu
            self.update_cost(mode, n, True)
            n_parent_inter = n.parent
            cost_to_parent_inter = n.cost_to_parent
            # agent_dists_to_parent_inter = n.agent_dists_to_parent

            n.parent = n_parent
            n.cost_to_parent = cost_to_parent
            # n.agent_dists_to_parent = dist_cpu
            n_parent.children.append(n) #Set child
            
            cost_to_parent = cost_to_parent_inter
            # dists = agent_dists_to_parent_inter
            n_parent = n
            n = n_parent_inter
            if not n:
                #need to have this transition node as the last one in the queue
                if n_parent.transition:
                    self.transition_node_ids[mode].remove(n_parent.id)
                    self.mark_node_as_transition(mode, n_parent)
                return 
            # print(n.state.mode)
            n.children.remove(n_parent)
  
    def connect(self, mode:Mode, n_new:Node) -> None:
        """
        Attempts to connect the bidirectional trees (forward and reversed growing subtrees) in the given mode by extending and linking nodes.

        Args:
            mode (Mode): Current operational mode.
            n_new (Node): Node to be connected.

        Returns:
            None: This method does not return any value.
        """

        if not self.trees[mode].subtree_b:
            return
        n_nearest_b, dist, _, _= self.nearest(mode, n_new.state.q, 'B')

        if self.config.birrtstar_version == 1 or self.config.birrtstar_version == 2 and self.trees[mode].connected: #Based on paper Bi-RRT* by B. Wang 
            #TODO only check dist of active robots to connect (cost can be extremly high)? or the smartest way to just connect when possible?
            # relevant_dists = []
            # for r_idx, r in enumerate(self.env.robots):
            #     if r in constrained_robots:
            #         relevant_dists.append(dists[0][r_idx].item())
            # if np.max(relevant_dists) > self.step_size:
            #     return

            if dist > self.eta:
                return

            if not self.env.is_edge_collision_free(n_new.state.q, n_nearest_b.state.q, mode): #ORder rigth? TODO
                return
          
        elif self.config.birrtstar_version == 2 and not self.trees[mode].connected: #Based on paper RRT-Connect by JJ. Kuffner/ RRT*-Connect by S.Klemm
            n_nearest_b = self.extend(mode, n_nearest_b, n_new, dist)
            if not n_nearest_b:
                return
           
        cost =  self.env.batch_config_cost([n_new.state],  [n_nearest_b.state])
        if self.trees[mode].order == -1:
            #switch such that subtree is beginning from start and subtree_b from goal
            self.trees[mode].swap()
            self.swap = False
            if not self.env.is_edge_collision_free(n_nearest_b.state.q, n_new.state.q, mode):
                return
            self.update_tree(mode, n_new, n_nearest_b, cost[0]) 
            
        else:
            if not self.env.is_edge_collision_free(n_new.state.q, n_nearest_b.state.q, mode):
                return
            self.update_tree(mode, n_nearest_b, n_new, cost[0])

        transition_node = self.get_transition_node(mode, self.transition_node_ids[mode][-1]) 
        #initial solution has been found for the frist time in this mode
        if not self.trees[mode].connected: 
            self.trees[mode].connected = True
            #check if terminal mode was already reached
            if not self.env.is_terminal_mode(mode):
                self.add_new_mode(transition_node.state.q, mode, BidirectionalTree)
            elif transition_node.cost != np.inf:
                self.operation.init_sol = True
                # print(time.time()-self.start_time)
        #need to do that after the next mode was initialized
        self.convert_node_to_transition_node(mode, transition_node)

    def extend(self, 
               mode:Mode, 
               n_nearest_b:Node, 
               n_new:Node, dist
               )-> Optional[Node]:
        """
        Extends subtree by incrementally steering from the nearest node toward the target configuration.

        Args:
            mode (Mode): Current operational mode for the extension.
            n_nearest_b (Node): Nearest node from which extension begins.
            n_new (Node): Target node toward which the extension proceeds.
            dist: Distance between n_nearest_b and n_new.

        Returns:
            Optional[Node]: Last valid (collision-free) node reached during extension.
        """

        q = n_new.state.q
        #RRT not RRT*
        i = 1
        while True:
            state_new = self.steer(mode, n_nearest_b, q, dist, i)
            if not state_new or np.equal(state_new.q.state(), q.state()).all(): # Reached
                return n_nearest_b
            if self.env.is_collision_free(state_new.q, mode) and self.env.is_edge_collision_free(n_nearest_b.state.q, state_new.q, mode):
                # Add n_new to tree
                n_new = Node(state_new,self.operation)
               
                cost =  self.env.batch_config_cost([n_new.state], [n_nearest_b.state])
                c_min = n_nearest_b.cost + cost

                n_new.parent = n_nearest_b
                n_new.cost_to_parent = cost
                n_nearest_b.children.append(n_new) #Set child
                self.trees[mode].add_node(n_new, 'B') 
                self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, n_new.id) 
                n_new.cost = c_min
                n_nearest_b = n_new
                i +=1
            else:
                return 
            
    def save_tree_data(self) -> None:
        if not self.config.with_tree_visualization:
            return
        data = {}
        data['all_nodes'] = [self.trees[m].subtree[id].state.q.state() for m in self.modes for id in self.trees[m].get_node_ids_subtree('A')]
        
        try:
            data['all_transition_nodes'] = [self.trees[m].subtree[id].state.q.state() for m in self.modes for id in self.transition_node_ids[m]]
            data['all_transition_nodes_mode'] = [self.trees[m].subtree[id].state.mode.task_ids for m in self.modes for id in self.transition_node_ids[m]]
        except Exception:
            data['all_transition_nodes'] = []
            data['all_transition_nodes_mode'] = []
        data['all_nodes_mode'] = [self.trees[m].subtree[id].state.mode.task_ids for m in self.modes for id in self.trees[m].get_node_ids_subtree()]
        
        for i, type in enumerate(['forward', 'reverse']): 
            data[type] = {}
            data[type]['nodes'] = []
            data[type]['parents'] = []
            data[type]['modes'] = []
            for m in self.modes:
                if type == 'forward':
                    ids = self.trees[m].get_node_ids_subtree('A')
                else:
                    ids = self.trees[m].get_node_ids_subtree('B')
                for id in ids:
                    if type == 'forward':
                        node = self.trees[m].subtree[id]
                    else:
                        node = self.trees[m].subtree_b[id]
                    data[type]["nodes"].append(node.state.q.state())
                    data[type]['modes'].append(node.state.mode.task_ids)
                    parent = node.parent
                    if parent is not None:
                        data[type]["parents"].append(parent.state.q.state())
                    else:
                        data[type]["parents"].append(None)
            
        data['pathnodes'] = []
        data['pathparents'] = []
        if self.operation.path_nodes is not None:
            for node in self.operation.path_nodes: 
                data['pathnodes'].append(node.state.q.state())
                parent = node.parent
                if parent is not None:
                    data['pathparents'].append(parent.state.q.state())
                else:
                    data['pathparents'].append(None)

        save_data(data, True)

    def initialize_planner(self) -> None:
        # Initilaize first Mode
        self.set_gamma_rrtstar()
        self.add_new_mode(tree_instance=BidirectionalTree)
        mode = self.modes[-1]
        # Create start node
        start_state = State(self.env.start_pos, mode)
        start_node = Node(start_state, self.operation)
        self.trees[mode].add_node(start_node)
        start_node.cost = 0.0
        start_node.cost_to_parent = 0.0
        self.manage_transition(mode, start_node)

    def plan(
        self,
        ptc: PlannerTerminationCondition,
        optimize: bool = True,
    ) -> Tuple[List[State] | None, Dict[str, Any]]:
        i = 0
        self.initialize_planner()
        while True:
            self.swap = True
            i += 1
            # Mode selection
            active_mode  = self.random_mode()
            # Bi RRT* core
            q_rand = self.sample_node_manifold(active_mode)
            if not q_rand:
                continue
            n_nearest, dist, set_dists, n_nearest_idx = self.nearest(active_mode, q_rand)        
            state_new = self.steer(active_mode, n_nearest, q_rand, dist)
            # q_rand == n_nearest
            if not state_new: 
                continue

            if self.env.is_collision_free(state_new.q, active_mode) and self.env.is_edge_collision_free(n_nearest.state.q, state_new.q, active_mode):
                n_new = Node(state_new, self.operation)
                if np.equal(n_new.state.q.state(), q_rand.state()).all():
                    N_near_batch, n_near_costs, node_indices = self.near(active_mode, n_new, n_nearest_idx, set_dists)
                else:
                    N_near_batch, n_near_costs, node_indices = self.near(active_mode, n_new, n_nearest_idx)

                batch_cost = self.env.batch_config_cost(n_new.state.q, N_near_batch)
                self.find_parent(active_mode, node_indices, n_new, n_nearest, batch_cost, n_near_costs)
                if self.rewire(active_mode, node_indices, n_new, batch_cost, n_near_costs):
                    self.update_cost(active_mode,n_new)
                self.connect(active_mode, n_new)
                self.manage_transition(active_mode, n_new)
            if self.swap:
                self.trees[active_mode].swap()
            
            if not optimize and self.operation.init_sol:
                self.save_tree_data()
                break

            if ptc.should_terminate(i, time.time() - self.start_time):
                print('Number of iterations: ', i)
                break
            
        self.update_results_tracking(self.operation.cost, self.operation.path)
        info = {"costs": self.costs, "times": self.times, "paths": self.all_paths}
        print(self.mode_validation.counter)
        return self.operation.path, info      




