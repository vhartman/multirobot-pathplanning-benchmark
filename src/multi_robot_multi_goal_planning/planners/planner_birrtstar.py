import numpy as np
import time as time
import math as math
from typing import Tuple, Optional, Union, List, Dict
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
    BaseRRTstar, 
    Node, 
    BidirectionalTree

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
                 env:BaseProblem, 
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
                 gaussian: bool = False, 
                 transition_nodes: int = 50, 
                 birrtstar_version: int = 2,
                 locally_informed_sampling: bool = True, 
                 remove_redundant_nodes: bool = True, 
                 informed_batch_size: int = 500, 
                ):
        super().__init__(env, ptc, general_goal_sampling, informed_sampling, informed_sampling_version, distance_metric,
                    p_goal, p_stay, p_uniform, shortcutting, mode_sampling, 
                    gaussian = gaussian, locally_informed_sampling = locally_informed_sampling, remove_redundant_nodes = remove_redundant_nodes, informed_batch_size = informed_batch_size )
        self.transition_nodes = transition_nodes 
        self.birrtstar_version = birrtstar_version
       
    def UpdateCost(self, mode:Mode, n:Node, connection:bool = False) -> None:
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
            new_mode = self.env.make_start_mode()
            new_mode.prev_mode = None
        else:
            new_mode = self.env.get_next_mode(q, mode)
            new_mode.prev_mode = mode
        if new_mode in self.modes:
            return 
    
        self.modes.append(new_mode)
        self.add_tree(new_mode, tree_instance)
        self.InformedInitialization(new_mode)
        #Initialize transition nodes
        node = None
        for i in range(self.transition_nodes):                 
            q = self.sample_transition_configuration(new_mode)
            if i > 0 and np.equal(q.state(), node.state.q.state()).all():
                break
            node = Node(State(q, new_mode), self.operation)
            if node in self.trees[new_mode].subtree.values():
                continue
            node.cost_to_parent = 0.0
            self.mark_node_as_transition(new_mode, node)
            self.trees[new_mode].add_node(node, 'B')
            self.operation.costs = self.trees[new_mode].ensure_capacity(self.operation.costs, node.id) 
            node.cost = 0.0

    def ManageTransition(self, mode:Mode, n_new: Node) -> None:
        #check if transition is reached
        if self.trees[mode].order == 1 and self.env.is_transition(n_new.state.q, mode):
            self.trees[mode].connected = True
            self.add_new_mode(n_new.state.q, mode, BidirectionalTree)
            self.convert_node_to_transition_node(mode, n_new)
        #check if termination is reached
        if self.trees[mode].order == 1 and self.env.done(n_new.state.q, mode):
            self.trees[mode].connected = True
            self.convert_node_to_transition_node(mode, n_new)
            if not self.operation.init_sol:
                # print(time.time()-self.start_time)
                self.operation.init_sol = True
        self.FindLBTransitionNode()

    def UpdateTree(self, 
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
            self.UpdateCost(mode, n, True)
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
  
    def Connect(self, mode:Mode, n_new:Node) -> None:
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
        n_nearest_b, dist, _, _= self.Nearest(mode, n_new.state.q, 'B')

        if self.birrtstar_version == 1 or self.birrtstar_version == 2 and self.trees[mode].connected: #Based on paper Bi-RRT* by B. Wang 
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
          
        elif self.birrtstar_version == 2 and not self.trees[mode].connected: #Based on paper RRT-Connect by JJ. Kuffner/ RRT*-Connect by S.Klemm
            n_nearest_b = self.Extend(mode, n_nearest_b, n_new, dist)
            if not n_nearest_b:
                return
           
        cost =  self.env.batch_config_cost([n_new.state],  [n_nearest_b.state])
        if self.trees[mode].order == -1:
            #switch such that subtree is beginning from start and subtree_b from goal
            self.trees[mode].swap()
            if not self.env.is_edge_collision_free(n_nearest_b.state.q, n_new.state.q, mode):
                return
            self.UpdateTree(mode, n_new, n_nearest_b, cost[0]) 
            
        else:
            if not self.env.is_edge_collision_free(n_new.state.q, n_nearest_b.state.q, mode):
                return
            self.UpdateTree(mode, n_nearest_b, n_new, cost[0])

        transition_node = self.get_transition_node(mode, self.transition_node_ids[mode][-1]) 
        #initial solution has been found for the frist time in this mode
        if not self.trees[mode].connected: 
            self.trees[mode].connected = True
            #check if terminal mode was already reached
            if not self.env.is_terminal_mode(mode):
                self.add_new_mode(transition_node.state.q, mode, BidirectionalTree)
            else:
                self.operation.init_sol = True
                # print(time.time()-self.start_time)
        #need to do that after the next mode was initialized
        self.convert_node_to_transition_node(mode, transition_node)

    def Extend(self, 
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
            state_new = self.Steer(mode, n_nearest_b, q, dist, i)
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

    def PlannerInitialization(self) -> None:
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

    def Plan(self) ->  Tuple[List[State], Dict[str, List[Union[float, float, List[State]]]]]:
        i = 0
        self.PlannerInitialization()
        while True:
            i += 1
            # Mode selection
            active_mode  = self.RandomMode()
            # Bi RRT* core
            q_rand = self.SampleNodeManifold(active_mode)
            if not q_rand:
                continue
            n_nearest, dist, set_dists, n_nearest_idx = self.Nearest(active_mode, q_rand)        
            state_new = self.Steer(active_mode, n_nearest, q_rand, dist)
            # q_rand == n_nearest
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
                    self.UpdateCost(active_mode,n_new)
                self.Connect(active_mode, n_new)
                self.ManageTransition(active_mode, n_new)
            self.trees[active_mode].swap()

            if self.ptc.should_terminate(i, time.time() - self.start_time):
                break
        self.costs.append(self.operation.cost)
        self.times.append(time.time() - self.start_time)
        self.all_paths.append(self.operation.path)
        info = {"costs": self.costs, "times": self.times, "paths": self.all_paths}
        return self.operation.path, info      




