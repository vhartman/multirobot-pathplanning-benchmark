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
    BaseOperation,
    BaseGraph,
    DictIndexHeap,
    BaseNode
)

# taken from https://github.com/marleyshan21/Batch-informed-trees/blob/master/python/BIT_Star.py
# needed adaption to work.

class Operation(BaseOperation):
    """Represents an operation instance responsible for managing variables related to path planning and cost optimization. """
    
    def __init__(self) -> None:
        super().__init__()
        self.lb_costs_to_go_expanded = np.empty(10000000, dtype=np.float64)
    def get_lb_cost_to_go_expanded(self, idx:int) -> float:
        """
        Returns cost of node with the specified index.

        Args: 
            idx (int): Index of node whose cost is to be retrieved.

        Returns: 
            float: Cost associated with the specified node."""
        return self.lb_costs_to_go_expanded[idx]

    def update(self, node:"BaseNode", lb_cost_to_go:float = np.inf, cost:float = np.inf):
        self.lb_costs_to_go_expanded = self.ensure_capacity(self.lb_costs_to_go_expanded, node.id) 
        node.lb_cost_to_go_expanded = lb_cost_to_go
        node.lb_cost_to_go = lb_cost_to_go
        self.lb_costs_to_come = self.ensure_capacity(self.lb_costs_to_come, node.id) 
        node.lb_cost_to_come = np.inf
        self.costs = self.ensure_capacity(self.costs, node.id) 
        node.cost = cost

class Graph(BaseGraph):
    def __init__(self, root, operation, batch_dist_fun, batch_cost_fun, node_cls):
        super().__init__(root, operation, batch_dist_fun, batch_cost_fun, node_cls)
    
    def add_path_states(self, path:List[State], current_best_cost:float):
        self.current_best_path_nodes = []
        self.current_best_path = []
        batch_edge_cost = self.batch_cost_fun(path[:-1], path[1:])
        parent = self.root
        self.current_best_path_nodes.append(parent)
        self.current_best_path.append(parent.state)
        parent.lb_cost_to_go = current_best_cost
        parent.lb_cost_to_go_expanded = current_best_cost
        for i in range(len(path)):
            if i == 0:
                continue
            is_transition = False
            next_mode = None
            edge_cost = batch_edge_cost[i-1]

            if (
                i < len(path) - 1
                and path[i].mode
                != path[i + 1].mode
            ):
                is_transition = True
                next_mode = path[i+1].mode
            if i == len(path)-1:
                is_transition = True
            node = Node(self.operation, path[i], is_transition)
            self.current_best_path_nodes.append(node)
            self.current_best_path.append(node.state)
            self.add_path_node(node, parent, edge_cost, is_transition, next_mode, current_best_cost)
            parent = self.current_best_path_nodes[-1]
        return self.current_best_path, self.current_best_path_nodes

    def update_connectivity(self, n0: BaseNode, n1: BaseNode, edge_cost, tree: str = "forward"):
        if tree == "forward":
            if n1.forward.parent is not None and n1.forward.parent.id != n0.id:
                n1.forward.parent.forward.children.remove(n1.id)
                n1.forward.parent.forward.fam.remove(n1.id)
                n1.forward.fam.remove(n1.forward.parent.id)
            n1.forward.parent = n0
            updated_cost = n0.cost + edge_cost
            n1.forward.cost_to_parent = edge_cost
            n0.forward.children.append(n1.id)
            if updated_cost != n1.cost:
                n1.cost = updated_cost
                if len(n1.forward.children) > 0:
                    self.update_forward_cost_of_children(n1)
            else:
                print("uhhh")
            self.add_vertex(n1)
            self.add_vertex(n0)
            n1.forward.fam.add(n0.id)
            n0.forward.fam.add(n1.id)
        else:
            n1.lb_cost_to_go = n0.lb_cost_to_go_expanded + edge_cost
            if n1.rev.parent is not None:
                if n1.rev.parent.id == n0.id:
                    return
                if n1.rev.parent.id != n0.id:
                    assert [
                                (self.nodes[child].rev.parent, child)
                                for child in n1.rev.parent.rev.children
                                if self.nodes[child].rev.parent is None
                                or self.nodes[child].rev.parent.id != n1.rev.parent.id
                            ] == [], "parent and children not correct"

                    n1.rev.parent.rev.children.remove(n1.id)
                    n1.rev.parent.rev.fam.remove(n1.id)
                    n1.rev.fam.remove(n1.rev.parent.id)


            n1.rev.parent = n0
            assert n1.id not in n0.rev.children, (
                "already a child")
            n0.rev.children.append(n1.id)
            n1.rev.cost_to_parent = edge_cost
            assert [
                        (self.nodes[child].rev.parent, child)
                        for child in n1.rev.parent.rev.children
                        if self.nodes[child].rev.parent is None
                        or self.nodes[child].rev.parent.id != n1.rev.parent.id
                    ] == [], (
                        "new parent and new children not correct")
            n1.rev.fam.add(n0.id)
            n0.rev.fam.add(n1.id)

    def reset_reverse_tree(self):
        [
            (
                setattr(self.nodes[node_id], "lb_cost_to_go", math.inf),
                setattr(self.nodes[node_id], "lb_cost_to_go_expanded", math.inf),
                self.nodes[node_id].rev.reset()
            )
            for node_id in list(chain.from_iterable(self.node_ids.values()))
        ]
        [
            (
                setattr(self.nodes[node_id], "lb_cost_to_go", math.inf),
                setattr(self.nodes[node_id], "lb_cost_to_go_expanded", math.inf),
                self.nodes[node_id].rev.reset()
            )
            for node_id in list(chain.from_iterable(self.transition_node_ids.values()))
        ]  # also includes goal nodes
        [
            (
                setattr(self.nodes[node_id], "lb_cost_to_go", math.inf),
                setattr(self.nodes[node_id], "lb_cost_to_go_expanded", math.inf),
                self.nodes[node_id].rev.reset()
            )
              for node_id in list(chain.from_iterable(self.reverse_transition_node_ids.values()))
        ]
    
    def reset_all_goal_nodes_lb_costs_to_go(self):
        for node in self.goal_nodes:
            node.lb_cost_to_go = 0
            node.rev.cost_to_parent = 0
            self.reverse_queue.heappush(node)

    def compute_transition_lb_cost_to_go(self):
        # run a reverse search on the transition nodes without any collision checking
        costs = {}
        closed_set = set()

        queue = []
        for g in self.goal_nodes:
            heapq.heappush(queue, (0, g))
            g.lb_cost_to_go = 0.0
            costs[g.id] = 0
            # parents[hash(g)] = None

        while len(queue) > 0:
            # node = queue.pop(0)
            _, node = heapq.heappop(queue)
            # print(node)

            # error happens at start node
            if node.state.mode == self.root.state.mode:
                continue

            if node.id in closed_set:
                continue
            if node.state.mode not in self.reverse_transition_node_ids:
                continue
            neighbors = [
                self.nodes[id].transition for id in self.reverse_transition_node_ids[node.state.mode]
            ]

            if len(neighbors) == 0:
                continue

            self.update_cache(node.state.mode)

            closed_set.add(node.id)
            # add neighbors to open_queue
            edge_costs = self.batch_cost_fun(
                node.state.q,
                self.reverse_transition_node_array_cache[node.state.mode],
            )

            parent_cost = costs[node.id]
            for edge_cost, n in zip(edge_costs, neighbors):
                cost = parent_cost + edge_cost
                id = n.id
                if id not in costs or cost < costs[id]:
                    costs[id] = cost
                    n.lb_cost_to_go = cost
                    if n.transition is not None:
                        n.transition.lb_cost_to_go = cost
                    heapq.heappush(queue, (cost, n))

    def compute_node_lb_cost_to_go(self):
        processed = 0
        transition_node_lb_cache = {}
        for mode in self.node_ids:
            for id in self.node_ids[mode]:
                n = self.nodes[id]
                mode = n.state.mode
                if mode not in self.transition_node_array_cache:
                    continue

                if mode not in transition_node_lb_cache:
                    transition_node_lb_cache[mode] = np.array(
                        [
                            self.nodes[id].lb_cost_to_go
                            for id in self.transition_node_ids[mode]
                        ],
                        dtype=np.float64,
                    )

                costs_to_transitions = self.batch_cost_fun(
                    n.state.q,
                    self.transition_node_array_cache[mode],
                )

                min_cost = np.min(
                    transition_node_lb_cache[mode] + costs_to_transitions
                )
                n.lb_cost_to_go = min_cost
                processed +=1
        print(processed)



class Node(BaseNode):
    def __init__(self, operation: "BaseOperation", state: "State", is_transition: bool = False) -> None:
        super().__init__(operation, state, is_transition)

    @property
    def lb_cost_to_go_expanded(self):
        return self.operation.get_lb_cost_to_go_expanded(self.id)
    
    @lb_cost_to_go_expanded.setter
    def lb_cost_to_go_expanded(self, value) -> None:
        """Set the cost in the shared operation costs array.

        Args:
            value (float): Cost value to assign to the current node.

        Returns: 
            None: This method does not return any value."""
        self.operation.lb_costs_to_go_expanded[self.id] = value

class ForwardQueue(DictIndexHeap[Tuple[Any]]):
    def __init__(self, alpha = 1.0):
        super().__init__()
        self.alpha = alpha

    def key(self, item: Tuple[Any]) -> float:
        # item[0]: edge_cost from n0 to n1
        # item[1]: edge (n0, n1)
        return (
            item[1][0].cost + item[0] + item[1][1].lb_cost_to_go*self.alpha,
            item[1][0].cost + item[0],
            item[1][0].cost,
        )
    def add_node(self, item):
        node = item[1][1]
        if np.isinf(node.lb_cost_to_go) and np.isinf(node.lb_cost_to_go_expanded):
            self.nodes.add(node.id)
        elif node.lb_cost_to_go != node.lb_cost_to_go_expanded:
            self.nodes.add(node.id)

    def remove_node(self, item):
        node = item[1][1]
        if np.isinf(node.lb_cost_to_go) and np.isinf(node.lb_cost_to_go_expanded):
            return
        if node.lb_cost_to_go == node.lb_cost_to_go_expanded:
            self.nodes.discard(node.id)

class ReverseQueue(DictIndexHeap[Node]):
    def __init__(self):
        super().__init__()

    def key(self, node: Node) -> float:
        min_lb = min(node.lb_cost_to_go, node.lb_cost_to_go_expanded) 
        return ( min_lb + node.lb_cost_to_come, min_lb)
    
    def add_node(self, node):
        self.nodes.add(node.id)

    def remove_node(self, node):
        self.nodes.discard(node.id)

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
        informed_with_lb:bool = True,
        remove_based_on_modes:bool = False
        ):
        super().__init__(
            env, ptc, optimize, mode_sampling_type, distance_metric, 
            try_sampling_around_path, use_k_nearest, try_informed_sampling, 
            uniform_batch_size, uniform_transition_batch_size, informed_batch_size, 
            informed_transition_batch_size, path_batch_size, locally_informed_sampling, 
            try_informed_transitions, try_shortcutting, try_direct_informed_sampling, 
            informed_with_lb,remove_based_on_modes
        )

        self.alpha = 4.5
        self.start_transition_arrays = {}
        self.end_transition_arrays = {}
        self.remove_nodes = False
        self.inconsistent_target_nodes = set()

    def _create_operation(self) -> BaseOperation:
        return Operation()
        
    def inconcistency_check(self, node: Node):
        self.g.reverse_queue.remove(node)
        if node.lb_cost_to_go != node.lb_cost_to_go_expanded:
            self.g.reverse_queue.heappush(node)
    
    def continue_reverse_search(self) -> bool:
        if len(self.g.reverse_queue) == 0 or len(self.g.forward_queue) == 0:
            return False
        forward_key, item = self.g.forward_queue.peek_first_element()
        reverse_key, node = self.g.reverse_queue.peek_first_element()
        reverse_key_forward_edge_target = self.g.reverse_queue.key(item[1][1])
        if item[1][1].lb_cost_to_go == item[1][1].lb_cost_to_go_expanded:
            if forward_key[0] <= reverse_key[0]:
                return False
        if not self.g.forward_queue.nodes:
            return False
        
            #need to check that for all in forward queue
            # if reverse_key_forward_edge_target <= reverse_key:
            #     return False
        # if  <= reverse_key:elf.g.reverse_queue.key(self.g.root):
        #     return False
        
        # if self.g.root.lb_cost_to_go_expanded == self.g.root.lb_cost_to_go:
        #     return False
        return True
        
    def is_forward_search_tc_met(self) -> bool:
        pass
          
    def update_heuristic(self, edge: Optional[Tuple[Node, Node]] = None) -> float:
        self.reversed_closed_set = set()
        self.reversed_updated_set =set()
        if edge is None:
            self.g.reverse_queue = ReverseQueue()
            self.g.reset_reverse_tree()
            self.g.reset_all_goal_nodes_lb_costs_to_go()
        else:
            self.update_edge_collision_cache(edge[0], edge[1], False)
            # assert(edge[1].rev.parent != edge[0]), (
            #     "wrong "
            # )
            if edge[0].rev.parent == edge[1]:
                # print("Collision effects reverse search")
                self.invalidate_rev_branch(edge[1])
            elif edge[1].rev.parent == edge[0]:
                # print("Collision effects reverse search")
                self.invalidate_rev_branch(edge[0])
            return

        # Process the reverse queue until stopping conditions are met.
        num_iter = 0
        while self.continue_reverse_search(): 
            if not self.optimize and self.current_best_cost is not None:
                break

            if self.ptc.should_terminate(self.cnt, time.time() - self.start_time):
                break
            n = self.g.reverse_queue.heappop()
            # if edge is not None:
            #     print(n.id)
            self.reversed_closed_set.add(n.id)
            num_iter += 1
            if num_iter % 100000 == 0:
                print(num_iter, ": Reverse Queue: ", len(self.g.reverse_queue))
            # if the connected cost is lower than the expanded cost
            if n.lb_cost_to_go < n.lb_cost_to_go_expanded:
                n.lb_cost_to_go_expanded = n.lb_cost_to_go
            else:
                n.lb_cost_to_go_expanded = np.inf
                self.update_state(n, True)
            neighbors = self.g.get_neighbors(n, self.approximate_space_extent)
            # if len(neighbors) == 0:
            #     self.invalidate_rev_branch(n)
            #     continue
            for id in neighbors:  #node itself is not included in neighbors
                nb = self.g.nodes[id]
                self.update_state(nb, True)
                self.reversed_updated_set.add(nb.id)
     
    def update_state(self, node: Node, flag:bool = False) -> None:
        if node.id == self.g.root.id or node in self.g.goal_nodes:
            return
        # node was already expanded in current heuristic call
        if node.lb_cost_to_go == node.lb_cost_to_go_expanded and node.id in self.reversed_closed_set:
            self.inconcistency_check(node)
            return
        in_updated_set = node.id in self.reversed_updated_set
        neighbors = list(self.g.get_neighbors(node, self.approximate_space_extent, in_closed_set= in_updated_set))
        if len(neighbors) == 0:
            # if flag:
            #     self.invalidate_rev_branch(node)
            #     return
            self.update_node_without_available_reverse_parent(node)
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
                # if best_parent.rev.parent == node:
                #     print("drgdgf")
                # if best_parent.rev.parent is not None and best_parent.rev.parent.id in node.rev.children:
                #     print("drgdgf")
                # if best_parent.id in node.rev.children: 
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
            if node.rev.parent is not None and node.lb_cost_to_go == node.rev.parent.lb_cost_to_go_expanded + node.rev.cost_to_parent: 
                self.inconcistency_check(node)
                return

        best_parent = None
        for idx in sorted_candidates_indices:
            n = self.g.nodes[neighbors[idx]]
            assert (n.state.mode != node.state.mode.prev_mode or n.id != node.id), (
                "ohhhhhhhhhhhhhhh"
            )
            assert (n.rev.parent == node) == (n.id in node.rev.children), (
                f"Parent and children don't coincide (reverse): parent {node.id} of {n.id}"
            )
            if node.is_transition:
                if n.is_transition and n.transition is not None and n.transition.state.mode == node.state.mode.prev_mode:
                    continue

            #to avoid inner cycles in reverse tree (can happen when a parent appears on blacklist at some point):
            if n.rev.parent == node:
                continue
            if n.rev.parent is not None and n.rev.parent.id in node.rev.children:
                continue
            if n.id in node.rev.children: 
                continue

            best_parent = n
            best_edge_cost = batch_cost[idx]
            break

        if best_parent is not None:
            
            if best_parent.rev.parent is not None:
                assert best_parent.rev.parent.id not in node.rev.children, (
                    "Parent of potential parent of node is one of its children"
                )
            assert best_parent.id not in node.rev.children, (
                "Potential parent of node is one of its children"
            )
            assert best_parent.rev.parent != node, (
                "Parent of potential parent of node is node itself"
            )
            # if best_value < node.lb_cost_to_go:
            if node.rev.parent is not None:
                assert node.id in node.rev.parent.rev.children, (
                    f"Parent and children don't coincide (reverse): parent {node.id} of {n.id}"
                )

            self.g.update_connectivity(
                best_parent, node, best_edge_cost, "reverse"
            ) 
            if node.rev.parent is not None:
                assert (node.id in best_parent.rev.children) and (
                    node.rev.parent == best_parent
                ), "Not correct connected"
            if node.rev.parent is not None:
                assert node.id in node.rev.parent.rev.children, (
                    f"Parent and children don't coincide (reverse): parent {node.id} of {n.id}"
                )
        else:
            # if flag:
            #     self.invalidate_rev_branch(node)
            #     return
            self.update_node_without_available_reverse_parent(node)              
        self.inconcistency_check(node)
     
    def invalidate_rev_branch(self, node:Node):
        self.update_node_without_available_reverse_parent(node)
        # print("i",node.id)
        for id in node.rev.children:
            self.invalidate_rev_branch(self.g.nodes[id])
        self.update_state(node)
        self.reversed_updated_set.add(node.id)

    def update_node_without_available_reverse_parent(self, node:Node):
        if node not in self.g.goal_nodes:
            node.lb_cost_to_go = np.inf
        else:
            node.lb_cost_to_go = 0.0
        node.lb_cost_to_go_expanded = np.inf
        if node.rev.parent is not None:
            node.rev.parent.rev.children.remove(node.id)
            node.rev.parent.rev.fam.remove(node.id)
            node.rev.fam.remove(node.rev.parent.id) 
        node.rev.parent = None
        node.rev.cost_to_parent = np.inf
        self.g.reverse_queue.remove(node)

    def update_node_without_available_neighbors(self, node:Node):

        if node.forward.parent is not None:
            node.forward.parent.forward.children.remove(node.id)
            node.forward.parent.forward.fam.remove(node.id)
            node.forward.fam.remove(node.forward.parent.id) 
        if node.id != self.g.node.id:
            node.cost = np.inf
        else:
            node.cost = 0
        node.forward.cost_to_parent = np.inf    

    def expand_node_forward(self, node: Optional[Node] = None) -> None:
        start_reverse_search = False
        if node is None:
            start_reverse_search = True
            node = self.g.root
            self.g.forward_queue = ForwardQueue(self.alpha)
            self.forward_closed_set = set()
            assert node.cost == 0 ,(
                " root node wrong"
            )
            
        neighbors = self.g.get_neighbors(node, space_extent=self.approximate_space_extent)
        # if neighbors.size == 0:
        #     self.update_node_without_available_neighbors()
        #     return
        # add neighbors to open_queue
        edge_costs = self.g.tot_neighbors_batch_cost_cache[node.id]
        for id, edge_cost in zip(neighbors, edge_costs):
            n = self.g.nodes[id]
            assert (n.forward.parent == node) == (n.id in node.forward.children), (
                    f"Parent and children don't coincide (reverse): parent: {node.id}, child: {n.id}"
                    )
            if n.id in node.blacklist:
                continue
            if node.state.mode in n.state.mode.next_modes:
                continue
            if node.is_transition:
                if node.transition is not None and node.transition.state.mode in node.state.mode.next_modes:
                    if node.transition.id != n.id:
                        continue
            if n.is_transition:
                if n.transition is not None and n.id in self.g.reverse_transition_node_ids[node.state.mode]: 
                   continue
            edge = (node, n)
            self.g.forward_queue.heappush((edge_cost, edge))
               
        # if start_reverse_search:
        #     print("Restart reverse search ...")
        #     self.update_heuristic()
        #     print("... finished")
            # self.expand_node_forward(self.g.root)
    
    def initialize_search(self):
        self.sample_manifold()
        self.g.compute_transition_lb_cost_to_come(self.env)
        self.g.compute_node_lb_cost_to_come(self.env)
        self.g.compute_transition_lb_cost_to_go()
        self.g.compute_node_lb_cost_to_go()
        self.expand_node_forward()
        self.wasted_pops = 0
        self.processed_edges = 0

    def PlannerInitialization(self) -> None:
        q0 = self.env.get_start_pos()
        m0 = self.env.get_start_mode()

        self.reached_modes.append(m0)
        root = Node(self.operation, State(q0, m0))

        self.g = Graph(
            root=root,
            operation=self.operation,
            batch_dist_fun=lambda a, b: batch_config_dist(a, b, self.distance_metric),
            batch_cost_fun= lambda a, b: self.env.batch_config_cost(a, b),
            node_cls=Node
            )
        # initialize all queues (edge-queues)    
        self.initialize_search()
    
    def Plan(
        self,
    ) -> Tuple[List[State], Dict[str, List[Union[float, float, List[State]]]]]:
        self.PlannerInitialization()
        num_iter = 0
        n1 = None
        while True:
            num_iter += 1
            if num_iter % 100000 == 0:
                print("Forward Queue: ", len(self.g.forward_queue))
            if len(self.g.forward_queue) < 1:
                print("------------------------",n1.state.mode.task_ids)
                self.initialize_search()
                continue
            edge_cost, (n0, n1) = self.g.forward_queue.heappop()
            # if n1.state.mode.task_ids == [3,2]:
            #     pass
            # if n1.is_transition and n1.transition is not None: 
            #     if n1.transition.state.mode.task_ids == [3,2]:
            #         pass
            # if (n0,n1) in self.forward_closed_set and self.current_best_cost is None:
            #     continue
            
            if (
                self.current_best_cost is None
                or n0.cost + edge_cost + n1.lb_cost_to_go
                < self.current_best_cost
            ):
                if n1.forward.parent == n0:  # if its already the parent
                    if n1 not in self.forward_closed_set:
                        self.expand_node_forward(n1)
                        self.forward_closed_set.add(n1)
                elif (
                    n0.cost + edge_cost < n1.cost
                ):  # parent can improve the cost
                    assert n0.id not in n1.forward.children, (
                        "Potential parent is already a child (forward)"
                    )
                    # if n1.id in self.closed_set:
                    #     wasted_pops += 1
                    #     continue
                    # check edge sparsely now. if it is not valid, blacklist it, and continue with the next edge
                    if n0.id not in n1.whitelist:
                        assert(n1.id not in n0.blacklist), (
                            "sudafjklaöo"
                        )
                        # if n1.id in n0.blacklist:
                            
                        #     self.update_heuristic((n0, n1))
                        #     continue 

                        collision_free = self.env.is_edge_collision_free(
                            n0.state.q,
                            n1.state.q,
                            n0.state.mode,
                            self.env.collision_resolution,
                        )
                        self.update_edge_collision_cache(n0, n1, collision_free)
                        if not collision_free:
                            continue
                    self.processed_edges += 1
                    self.g.update_connectivity(n0, n1, edge_cost, "forward")
                    if n1 not in self.forward_closed_set:
                        self.expand_node_forward(n1)
                        self.forward_closed_set.add(n1)
                    path = self.generate_path()
                    if len(path) > 0:
                        self.process_valid_path(path)
                            
            else:
                self.initialize_search()

            if not self.optimize and self.current_best_cost is not None:
                break

            if self.ptc.should_terminate(self.cnt, time.time() - self.start_time):
                break
        if self.costs != []:
            self.update_results_tracking(self.costs[-1], self.current_best_path)
        info = {"costs": self.costs, "times": self.times, "paths": self.all_paths}
        return self.current_best_path, info
