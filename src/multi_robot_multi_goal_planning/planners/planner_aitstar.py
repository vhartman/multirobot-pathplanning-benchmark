import numpy as np
import random

from typing import (
    List,
    Dict,
    Tuple,
    Optional,
    Set,
    ClassVar,
    Any,
    Union,
    Generic,
    TypeVar,
)
from numpy.typing import NDArray

import heapq
import time
import math
from abc import ABC, abstractmethod
from itertools import chain
from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
    Mode,
)
from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
    batch_config_dist,
)
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
)
from multi_robot_multi_goal_planning.planners.itstar_base import Informed
from multi_robot_multi_goal_planning.planners.rrtstar_base import find_nearest_indices
from multi_robot_multi_goal_planning.planners.rrtstar_base import save_data
from multi_robot_multi_goal_planning.planners.itstar_base import (
    BaseITstar,
    Node,
    Graph,
    DictIndexHeap
)

# taken from https://github.com/marleyshan21/Batch-informed-trees/blob/master/python/BIT_Star.py
# needed adaption to work.

class ReverseQueue(DictIndexHeap[Node]):
    def __init__(self):
        super().__init__()

    def key(self, node: Node) -> float:
        min_lb = min(node.lb_cost_to_go, node.lb_cost_to_go_expanded) 
        return ( min_lb + node.lb_cost_to_go,min_lb)

    def heappop(self) -> Node:
        """Pop the item with the smallest priority from the heap."""
        if not self.queue:
            raise IndexError("pop from an empty heap")

         # Remove from dictionary (Lazy approach)
        while self.current_entries:
            priority, idx = heapq.heappop(self.queue)
            item = self.items.pop(idx)
            if item in self.current_entries:
                current_priority, current_idx = self.current_entries[item]
                if current_priority == priority and current_idx == idx:
                    del self.current_entries[item]
                    return item
            else:
                continue
                
        raise IndexError("pop from an empty queue")
    
class AITstar(BaseITstar):
    def __init__(
        self,
        env: BaseProblem,
        ptc: PlannerTerminationCondition,
        optimize: bool = True,
        mode_sampling_type: str = "greedy",
        distance_metric: str = "euclidean",
        try_sampling_around_path: bool = True,
        use_k_nearest: bool = True,
        try_informed_sampling: bool = True,
        uniform_batch_size: int = 200,
        uniform_transition_batch_size: int = 500,
        informed_batch_size: int = 500,
        informed_transition_batch_size: int = 500,
        path_batch_size: int = 500,
        locally_informed_sampling: bool = True,
        try_informed_transitions: bool = True,
        try_shortcutting: bool = True,
        try_direct_informed_sampling: bool = True,
        informed_with_lb:bool = True
        ):
        super().__init__(
            env, ptc, optimize, mode_sampling_type, distance_metric, 
            try_sampling_around_path, use_k_nearest, try_informed_sampling, 
            uniform_batch_size, uniform_transition_batch_size, informed_batch_size, 
            informed_transition_batch_size, path_batch_size, locally_informed_sampling, 
            try_informed_transitions, try_shortcutting, try_direct_informed_sampling, 
            informed_with_lb
        )

        self.alpha = 4.5
        self.start_transition_arrays = {}
        self.end_transition_arrays = {}
        self.remove_nodes = False
        
    def inconcistency_check(self, node: Node):
        self.g.reverse_queue.remove(node)
        if node.lb_cost_to_go != node.lb_cost_to_go_expanded:
            self.g.reverse_queue.heappush(node)
          
    def update_heuristic(self, edge: Optional[Tuple[Node, Node]] = None) -> float:
        self.reversed_closed_set = set() #do it correctl
        self.reversed_updated_set =set()
        if edge is None:
            self.g.reverse_queue = ReverseQueue()
            self.g.reset_reverse_tree()
            self.g.reset_all_goal_nodes_lb_costs_to_go()
        else:
            self.update_edge_collision_cache(edge[0], edge[1], False)

            if edge[0].reverse_parent == edge[1]:
                self.invalidate_rev_branch(edge[0])
            elif edge[1].reverse_parent == edge[0]:
                self.invalidate_rev_branch(edge[1])
            else: #nothing changed
                return

        # Process the reverse queue until stopping conditions are met.
        num_iter = 0
        while len(self.g.reverse_queue) > 0 and (
            self.g.reverse_queue.is_smallest_priority_less_than_root_priority(self.g.root)
            or self.g.root.lb_cost_to_go_expanded < self.g.root.lb_cost_to_go
            or len(self.g.forward_queue) > 0
        ): 
            if not self.optimize and self.current_best_cost is not None:
                break

            if self.ptc.should_terminate(self.cnt, time.time() - self.start_time):
                break
            n = self.g.reverse_queue.heappop()
            self.reversed_closed_set.add(n.id)
            num_iter += 1
            if num_iter % 100000 == 0:
                print(num_iter, ": Reverse Queue: ", len(self.g.reverse_queue))
            # if the connected cost is lower than the expanded cost
            if n.lb_cost_to_go < n.lb_cost_to_go_expanded:
                n.lb_cost_to_go_expanded = n.lb_cost_to_go
            else:
                n.lb_cost_to_go_expanded = np.inf
                self.update_state(n)
            neighbors = self.g.get_neighbors(n, self.approximate_space_extent)
            for id in neighbors:  #node itself is not included in neighbors
                nb = self.g.nodes[id]
                self.update_state(nb)
                self.reversed_updated_set.add(nb.id)
     
    def update_state(self, node: Node) -> None:
        if node.id == self.g.root.id or node in self.g.goal_nodes:
            return
        # node was already expanded in current heuristic call
        if node.lb_cost_to_go == node.lb_cost_to_go_expanded and node.id in self.reversed_closed_set:
            self.inconcistency_check(node)
            return
        in_updated_set = node.id in self.reversed_updated_set
        neighbors = list(self.g.get_neighbors(node, self.approximate_space_extent, in_closed_set= in_updated_set))
        if not neighbors:
            self.update_node_without_available_reverse_parent(node)
            self.inconcistency_check(node)
            return
                        
        batch_cost = self.g.tot_neighbors_batch_cost_cache[node.id]
        
        if node.is_transition:
            idx = neighbors.index(node.transition.id)
            #only want to consider nodes as reverse parent with mode of node or its next modes
            if node.transition.state.mode == node.state.mode.prev_mode:
                neighbors.pop(idx)
                batch_cost = np.delete(batch_cost, idx)
            else:
                best_parent = self.g.nodes[neighbors[idx]]
                best_edge_cost = batch_cost[idx]
                # if best_parent.reverse_parent == node:
                #     print("drgdgf")
                # if best_parent.reverse_parent is not None and best_parent.reverse_parent.id in node.reverse_children:
                #     print("drgdgf")
                # if best_parent.id in node.reverse_children: 
                #     print("drgdgf")
                self.g.update_connectivity(
                    best_parent, node, best_edge_cost, "reverse"
                ) 
                self.inconcistency_check(node)
                return
                
        lb_costs_to_go_expanded = self.operation.lb_costs_to_go_expanded[neighbors]
        candidates =  lb_costs_to_go_expanded + batch_cost
        # if all neighbors are infinity, no need to update node
        if np.all(np.isinf(candidates)):
            self.inconcistency_check(node)
            return
        sorted_candidates_indices = np.argsort(candidates)
        best_idx = sorted_candidates_indices[0]
        
        # still returns same parent with same lb_cost_to_go as the best one
        if candidates[best_idx] == node.lb_cost_to_go:
            if node.reverse_parent is not None and node.lb_cost_to_go == node.reverse_parent.lb_cost_to_go_expanded + node.reverse_cost_to_parent: 
                self.inconcistency_check(node)
                return

        best_parent = None
        for idx in sorted_candidates_indices:
            n = self.g.nodes[neighbors[idx]]
            assert (n.state.mode != node.state.mode.prev_mode or n.id != node.id), (
                "ohhhhhhhhhhhhhhh"
            )
            assert (n.reverse_parent == node) == (n.id in node.reverse_children), (
                f"Parent and children don't coincide (reverse): parent {node.id} of {n.id}"
            )
            #to avoid inner cycles in reverse tree (can happen when a parent appears on blacklist at some point):
            if n.reverse_parent == node:
                continue
            if n.reverse_parent is not None and n.reverse_parent.id in node.reverse_children:
                continue
            if n.id in node.reverse_children: 
                continue

            best_parent = n
            best_edge_cost = batch_cost[idx]
            break

        if best_parent is not None:
            
            if best_parent.reverse_parent is not None:
                assert best_parent.reverse_parent.id not in node.reverse_children, (
                    "Parent of potential parent of node is one of its children"
                )
            assert best_parent.id not in node.reverse_children, (
                "Potential parent of node is one of its children"
            )
            assert best_parent.reverse_parent != node, (
                "Parent of potential parent of node is node itself"
            )
            # if best_value < node.lb_cost_to_go:
            if node.reverse_parent is not None:
                assert node.id in node.reverse_parent.reverse_children, (
                    f"Parent and children don't coincide (reverse): parent {node.id} of {n.id}"
                )

            self.g.update_connectivity(
                best_parent, node, best_edge_cost, "reverse"
            ) 
            if node.reverse_parent is not None:
                assert (node.id in best_parent.reverse_children) and (
                    node.reverse_parent == best_parent
                ), "Not correct connected"
            if node.reverse_parent is not None:
                assert node.id in node.reverse_parent.reverse_children, (
                    f"Parent and children don't coincide (reverse): parent {node.id} of {n.id}"
                )
        else:
            self.update_node_without_available_reverse_parent(node)                
        self.inconcistency_check(node)
     
    def invalidate_rev_branch(self, node:Node):
        if node not in self.g.goal_nodes:
            self.update_node_without_available_reverse_parent(node)
        node.lb_cost_to_go_expanded
        self.g.reverse_queue.remove(node)
        for id in node.reverse_children:
            rev_child = self.g.nodes[id]
            self.invalidate_rev_branch(rev_child)
        self.update_state(node)

    def update_node_without_available_reverse_parent(self, node:Node):

        node.lb_cost_to_go = np.inf
        node.lb_cost_to_go_expanded = np.inf
        if node.reverse_parent is not None:
            node.reverse_parent.reverse_children.remove(node.id)
            node.reverse_parent.reverse_fam.remove(node.id)
            node.reverse_fam.remove(node.reverse_parent.id) 
        node.reverse_parent = None
        node.reverse_cost_to_parent = np.inf

    def initialize_samples_and_forward_queue(self):
        self.sample_manifold()
        self.expand()
        self.wasted_pops = 0
        self.processed_edges = 0

    def PlannerInitialization(self) -> None:
        q0 = self.env.get_start_pos()
        m0 = self.env.get_start_mode()

        self.reached_modes.append(m0)

        self.g = Graph(self.operation,
            State(q0, m0), 
            lambda a, b: batch_config_dist(a, b, self.distance_metric),
            lambda a, b: self.env.batch_config_cost(a, b)
        )
        self.initialize_samples_and_forward_queue()

    def Plan(
        self,
    ) -> Tuple[List[State], Dict[str, List[Union[float, float, List[State]]]]]:
        self.PlannerInitialization()
        num_iter = 0
        while True:
            num_iter += 1
            if num_iter % 100000 == 0:
                print("Forward Queue: ", len(self.g.forward_queue))
            if len(self.g.forward_queue) < 1:
                self.initialize_samples_and_forward_queue()
                continue
            edge_cost, (n0, n1) = self.g.forward_queue.heappop()
            self.forward_closed_set.add((n0, n1))
            if (
                self.current_best_cost is None
                or n0.forward_cost + edge_cost + n1.lb_cost_to_go
                < self.current_best_cost
            ):
                if n1.forward_parent == n0:  # if its already the parent
                    self.expand(n1)
                elif (
                    n0.forward_cost + edge_cost < n1.forward_cost
                ):  # parent can improve the cost
                    assert n0.id not in n1.forward_children, (
                        "Potential parent is already a child (forward)"
                    )
                    # if n1.id in self.closed_set:
                    #     wasted_pops += 1
                    #     continue
                    # check edge sparsely now. if it is not valid, blacklist it, and continue with the next edge
                    if n0.id not in n1.whitelist:
                        if n1.id in n0.blacklist:
                            self.update_heuristic((n0, n1))
                            continue 

                        collision_free = self.env.is_edge_collision_free(
                            n0.state.q,
                            n1.state.q,
                            n0.state.mode,
                            self.env.collision_resolution,
                        )
                        self.update_edge_collision_cache(n0, n1, collision_free)
                        if not collision_free:
                            self.update_heuristic((n0, n1))
                            continue
                    self.processed_edges += 1
                    self.g.update_connectivity(n0, n1, edge_cost, "forward")
                    self.expand(n1)
                    path = self.generate_path()
                    if len(path) > 0:
                        self.process_valid_path(path)
                            
            else:
                self.initialize_samples_and_forward_queue()

            if not self.optimize and self.current_best_cost is not None:
                break

            if self.ptc.should_terminate(self.cnt, time.time() - self.start_time):
                break
        if self.costs != []:
            self.update_results_tracking(self.costs[-1], self.current_best_path)
        info = {"costs": self.costs, "times": self.times, "paths": self.all_paths}
        return self.current_best_path, info
