import numpy as np

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

import heapq
import time
import math
from itertools import chain
from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
)
from multi_robot_multi_goal_planning.problems.configuration import (
    batch_config_dist,
)
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
)
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

    def update(self, node:"Node", lb_cost_to_go:float = np.inf, cost:float = np.inf, lb_cost_to_come:float = np.inf):
        self.lb_costs_to_go_expanded = self.ensure_capacity(self.lb_costs_to_go_expanded, node.id) 
        node.lb_cost_to_go_expanded = lb_cost_to_go
        node.lb_cost_to_go = lb_cost_to_go
        self.lb_costs_to_come = self.ensure_capacity(self.lb_costs_to_come, node.id) 
        node.lb_cost_to_come = lb_cost_to_come
        self.costs = self.ensure_capacity(self.costs, node.id) 
        node.cost = cost
class Graph(BaseGraph):
    def __init__(self, 
                 root, 
                 operation, 
                 batch_dist_fun, 
                 batch_cost_fun, 
                 is_edge_collision_free, 
                 node_cls):
        super().__init__(root, operation, batch_dist_fun, batch_cost_fun, is_edge_collision_free, node_cls)
    
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
            g.test = 0.0
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
                    n.test = cost
                    if n.transition is not None:
                        n.transition.test = cost
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
                            self.nodes[id].test
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
                n.test = min_cost
                processed +=1
        print(processed)

    def update_forward_queue_keys(self, type:str ,node_ids:Optional[Set[BaseNode]] = None):
        self.forward_queue.update(node_ids, type)

class Node(BaseNode):
    def __init__(self, operation: "BaseOperation", state: "State", is_transition: bool = False) -> None:
        super().__init__(operation, state, is_transition)
        self.test = np.inf

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

    def close(self, resolution):
        self.lb_cost_to_go_expanded = self.lb_cost_to_go

    def set_to_goal_node(self):
        self.lb_cost_to_go = 0.0
        self.lb_cost_to_go_expanded = 0.0
class ForwardQueue(DictIndexHeap[Tuple[Any]]):
    def __init__(self, alpha = 1.0):
        super().__init__()
        self.alpha = alpha
        self.target_nodes = set()
        self.start_nodes = set()
        self.target_nodes_with_item = dict()
        self.start_nodes_with_item = dict()

    def key(self, item: Tuple[Any]) -> float:
        # item[0]: edge_cost from n0 to n1
        # item[1]: edge (n0, n1)

        return (
            item[1][0].cost + item[0] + item[1][1].lb_cost_to_go*self.alpha,
            item[1][0].cost + item[0],
            item[1][0].cost,
        )

    def push_and_sync(self, item):
        self.items[DictIndexHeap.idx] = item  # Store only valid items
        priority = self.key(item)
        self.current_entries[item] = (priority, DictIndexHeap.idx) # always up to date with the newest one!
        self.add_and_sync(item)
        return priority
    
    def add_and_sync(self, item):
        node = item[1][1]
        start_node = item[1][0]
        self.target_nodes.add(node.id)
        self.start_nodes.add(start_node.id)

        if node.id not in self.target_nodes_with_item:
            self.target_nodes_with_item[node.id] = set()
        self.target_nodes_with_item[node.id].add(item)

        if start_node.id not in self.start_nodes_with_item:
            self.start_nodes_with_item[start_node.id] = set()
        self.start_nodes_with_item[start_node.id].add(item)
    
    def remove(self, item, in_current_entries:bool = False):
        if not in_current_entries and item not in self.current_entries:
           return
        node = item[1][1]
        start_node = item[1][0]
        self.target_nodes_with_item[node.id].remove(item)
        self.start_nodes_with_item[start_node.id].remove(item)
        del self.current_entries[item]
        if not self.target_nodes_with_item[node.id]:
            self.target_nodes.remove(node.id)
        if not self.start_nodes_with_item[start_node.id]:
            self.start_nodes.remove(start_node.id)

    def update(self, node_ids:Optional[Set[BaseNode]], type:str):
        if node_ids is None:
            if type == 'target':
                node_ids = self.target_nodes
            elif type == 'start':
                node_ids = self.start_nodes
        if len(node_ids) == 0:
            return
        cnt = 0
        before = (len(self.current_entries))
        if type == 'start':
            for id in self.start_nodes:
                 if id not in node_ids:
                     continue
                 items = set(self.start_nodes_with_item[id])
                 for item in items: 
                    assert item in self.current_entries, (
                    "ijfdlk")
                    self.heappush(item)
                    cnt +=1
        if type == 'target':
            for id in self.target_nodes:
                if id not in node_ids:
                    continue
                items = set(self.target_nodes_with_item[id])
                for item in items: 
                    assert item in self.current_entries, (
                    "ijfdlk")
                    self.heappush(item)
                    cnt +=1
        assert before == len(self.current_entries), (
        "hjk,l")
class ReverseQueue(DictIndexHeap[Node]):
    def __init__(self):
        super().__init__()
        self.nodes = set()

    def key(self, node: Node) -> float:
        min_lb = min(node.lb_cost_to_go, node.lb_cost_to_go_expanded) 
        return (min_lb + node.lb_cost_to_come, min_lb)

class AITstar(BaseITstar):
    def __init__(
        self,
        env: BaseProblem,
        ptc: PlannerTerminationCondition,
        mode_sampling_type: str = "greedy",
        distance_metric: str = "euclidean",
        try_sampling_around_path: bool = True,
        try_informed_sampling: bool = True,
        first_uniform_batch_size: int = 100,
        first_transition_batch_size:int = 100,
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
        remove_based_on_modes:bool = False,
        with_tree_visualization:bool = False
        ):
        super().__init__(
            env = env, ptc=ptc, mode_sampling_type = mode_sampling_type, distance_metric = distance_metric, 
            try_sampling_around_path = try_sampling_around_path,try_informed_sampling = try_informed_sampling, 
            first_uniform_batch_size=first_uniform_batch_size, first_transition_batch_size=first_transition_batch_size,
            uniform_batch_size = uniform_batch_size, uniform_transition_batch_size = uniform_transition_batch_size, informed_batch_size = informed_batch_size, 
            informed_transition_batch_size = informed_transition_batch_size, path_batch_size = path_batch_size, locally_informed_sampling = locally_informed_sampling, 
            try_informed_transitions = try_informed_transitions, try_shortcutting = try_shortcutting, try_direct_informed_sampling = try_direct_informed_sampling, 
            informed_with_lb = informed_with_lb,remove_based_on_modes = remove_based_on_modes, with_tree_visualization = with_tree_visualization)

        self.alpha = 2.5
        self.start_transition_arrays = {}
        self.end_transition_arrays = {}
        self.remove_nodes = False
        self.dynamic_reverse_search_update = False
        
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
        if item[1][1].lb_cost_to_go == item[1][1].lb_cost_to_go_expanded:
            if forward_key[0] <= reverse_key[0]:
                return False
        inconsistent_nodes_in_fq = self.g.forward_queue.target_nodes - self.consistent_nodes
        if len(inconsistent_nodes_in_fq) == 0:
            for id in self.g.forward_queue.target_nodes:
                node = self.g.nodes[id]
                reverse_key_forward_edge_target = self.g.reverse_queue.key(node)
                if reverse_key_forward_edge_target > reverse_key:
                    return True
            return False
        return True
        
    def invalidate_rev_branch(self, node:Node):
        if node.is_transition and not node.is_reverse_transition:
            pass
        self.update_node_without_available_reverse_parent(node)
        for id in node.rev.children:
            self.invalidate_rev_branch(self.g.nodes[id])
        self.update_state(node)
      
    def reverse_search(self, edge: Optional[Tuple[Node, Node]] = None) -> float:
        self.reversed_closed_set = set()
        self.reversed_updated_set =set()
        self.reverse_udpated_lb_cost_to_go_set = set()
        if edge is not None:
            if edge[0].rev.parent == edge[1]:
                self.invalidate_rev_branch(edge[0])
            elif edge[1].rev.parent == edge[0]:
                self.invalidate_rev_branch(edge[1])
            else: #doesn't effect reverse search
                return
            self.g.forward_queue.update(self.reverse_udpated_lb_cost_to_go_set, 'target')
            self.reverse_udpated_lb_cost_to_go_set = set()
            if self.current_best_cost is None:
                return
            if not self.dynamic_reverse_search_update:
                return

        # Process the reverse queue until stopping conditions are met.
        num_iter = 0
        while self.continue_reverse_search(): 
            if self.ptc.should_terminate(self.cnt, time.time() - self.start_time):
                break
            n = self.g.reverse_queue.heappop()
            self.reversed_closed_set.add(n.id)
            num_iter += 1
            if num_iter % 100000 == 0:
                print(num_iter, ": Reverse Queue: ", len(self.g.reverse_queue))
            # if the connected cost is lower than the expanded cost
            # if n.id == 1162:
            #     pass
            is_rev_transition = False
            if n.lb_cost_to_go < n.lb_cost_to_go_expanded:
                self.consistent_nodes.add(n.id)
                n.lb_cost_to_go_expanded = n.lb_cost_to_go
                if n.is_transition and n.is_reverse_transition:
                    is_rev_transition = True
                    n.transition.lb_cost_to_go_expanded = n.lb_cost_to_go_expanded
                    self.consistent_nodes.add(n.transition.id)
                if n.is_transition and not n.is_reverse_transition:
                    pass
            else:
                self.consistent_nodes.discard(n.id)
                n.lb_cost_to_go_expanded = np.inf 
                if n.is_transition and n.is_reverse_transition:
                    n.transition.lb_cost_to_go_expanded = n.lb_cost_to_go_expanded
                    self.consistent_nodes.discard(n.transition.id)
                self.update_state(n)


            assert  n.lb_cost_to_go_expanded == n.lb_cost_to_go or (np.isinf(n.lb_cost_to_go_expanded)), (
                "lb_cost_to_go_expanded should not be finite and different from lb_cost_to_go"
            )

            self.update_heuristic_of_neihgbors(n)
            if is_rev_transition:
                assert(n.transition.lb_cost_to_go == n.lb_cost_to_go), ("ohhh")
                assert(n.transition.lb_cost_to_go_expanded == n.lb_cost_to_go_expanded), ("ohhh")
                assert(n.lb_cost_to_go == n.lb_cost_to_go_expanded), ("ohhh")
                # print(n.state.mode, n.transition.state.mode)
                self.update_heuristic_of_neihgbors(n.transition)
        self.g.update_forward_queue_keys('target', self.reverse_udpated_lb_cost_to_go_set)

    def update_heuristic_of_neihgbors(self, n):
        in_updated_set = n.id in self.reversed_updated_set 
        neighbors = self.g.get_neighbors(n, self.approximate_space_extent, in_updated_set)
        self.reversed_updated_set.add(n.id)
        for id in neighbors:  #node itself is not included in neighbors
            nb = self.g.nodes[id]
            if nb in self.g.goal_nodes:
                continue
            if nb.id in self.reversed_closed_set and nb.id in self.consistent_nodes:
                continue
                # if nb.rev.parent is None:
                #     pass
                # parent_before = nb.rev.parent.id
                # lb_cost_to_go_before = nb.lb_cost_to_go
                # self.update_state(nb)
                # assert(nb.rev.parent.id == parent_before and lb_cost_to_go_before == nb.lb_cost_to_go), (
                # "asdklföä")
            else:
                self.update_state(nb)
                        
    def update_state(self, node: Node) -> None:
        in_updated_set = node.id in self.reversed_updated_set
        self.reversed_updated_set.add(node.id)
        
        if node.id == self.g.root.id or node in self.g.goal_nodes:
            return
        if node.is_transition and not node.is_reverse_transition:
            return
        # node was already expanded in current heuristic call
        if node.lb_cost_to_go == node.lb_cost_to_go_expanded and node.id in self.reversed_closed_set:
            self.g.reverse_queue.remove(node)
            return
    
        
        neighbors = list(self.g.get_neighbors(node, self.approximate_space_extent, in_closed_set= in_updated_set))
        if len(neighbors) == 0:
            self.update_node_without_available_reverse_parent(node)
            return
        
        batch_cost = self.g.tot_neighbors_batch_cost_cache[node.id]
        is_rev_transition = False
        if node.is_transition and node.is_reverse_transition:
            #only want to consider nodes as reverse parent with mode of node or its next modes
            is_rev_transition = True
            idx = neighbors.index(node.transition.id)
            neighbors.pop(idx)
            batch_cost = np.delete(batch_cost, idx)

                
        lb_costs_to_go_expanded = self.operation.lb_costs_to_go_expanded[neighbors]
        candidates =  lb_costs_to_go_expanded + batch_cost
        # if all neighbors are infinity, no need to update node
        if np.all(np.isinf(candidates)):
            self.update_node_without_available_reverse_parent(node)
            return
        # still returns same parent with same lb_cost_to_go as the best one
        sorted_candidates = np.argsort(candidates)
        best_idx = sorted_candidates[0]
        best_lb_cost_to_go = candidates[best_idx]
        best_parent = self.g.nodes[neighbors[best_idx]]
        if  best_lb_cost_to_go == node.lb_cost_to_go:
            #check if its the right parent that leads to this lb cost to go (can have several nodes that lead to same result)
            if node.rev.parent.lb_cost_to_go_expanded + node.rev.cost_to_parent == best_lb_cost_to_go:
                self.inconcistency_check(node)
                return          
        if node.rev.parent is not None and best_parent.id == node.rev.parent.id:
            if best_lb_cost_to_go != node.lb_cost_to_go:
                self.reverse_udpated_lb_cost_to_go_set.add(node.id)
                node.lb_cost_to_go = best_lb_cost_to_go
                if node.is_reverse_transition: 
                    self.reverse_udpated_lb_cost_to_go_set.add(node.transition.id)
                    node.transition.lb_cost_to_go = best_lb_cost_to_go
            self.inconcistency_check(node)
            return
        best_parent = None
        for idx in sorted_candidates:
            if np.isinf(candidates[idx]):
                break
            n = self.g.nodes[neighbors[idx]]
            assert(n.id not in node.blacklist), (
            "neighbors are wrong")
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

        if best_parent is None: 
            self.update_node_without_available_reverse_parent(node)
            return
        
        assert(not np.isinf(best_parent.lb_cost_to_go_expanded)), (
            "fghjksl"
        )
        
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
        if candidates[idx] != node.lb_cost_to_go:
            self.reverse_udpated_lb_cost_to_go_set.add(node.id)
            if is_rev_transition:
                self.reverse_udpated_lb_cost_to_go_set.add(node.transition.id)
        self.g.update_connectivity(
            best_parent, node, best_edge_cost, best_parent.lb_cost_to_go_expanded + best_edge_cost , "reverse", is_rev_transition
        ) 
        self.inconcistency_check(node)

        if node.rev.parent is not None:
            assert (node.id in best_parent.rev.children) and (
                node.rev.parent == best_parent
            ), "Not correct connected"
        if node.rev.parent is not None:
            assert node.id in node.rev.parent.rev.children, (
                f"Parent and children don't coincide (reverse): parent {node.id} of {n.id}"
            )
        assert  n.lb_cost_to_go_expanded == n.lb_cost_to_go or not (np.isinf(n.lb_cost_to_go_expanded)), (
            "lb_cost_to_go_expanded should not be finite and different from lb_cost_to_go"
        )

    def update_node_without_available_reverse_parent(self, node:Node):
        self.consistent_nodes.discard(node.id)
        if not np.isinf(node.lb_cost_to_go):
            self.reverse_udpated_lb_cost_to_go_set.add(node.id)

        if node not in self.g.goal_nodes:
            node.lb_cost_to_go = np.inf
            node.lb_cost_to_go_expanded = np.inf
            node.rev.cost_to_parent = np.inf
        if node.rev.parent is not None:
            node.rev.parent.rev.children.remove(node.id)
            node.rev.parent.rev.fam.remove(node.id)
            node.rev.fam.remove(node.rev.parent.id) 
        node.rev.parent = None
        self.g.reverse_queue.remove(node)
        if node.is_transition and node.is_reverse_transition:
            self.update_node_without_available_reverse_parent(node.transition)
        
    def update_forward_queue(self, edge_cost, edge):
        self.g.forward_queue.heappush((edge_cost, edge))
               
    def initialize_search(self):
        # if self.g.get_num_samples() >200:
        #     q_samples = []
        #     modes = []
        #     for mode in self.reached_modes:
        #         q_samples.extend([self.g.nodes[id].state.q.state() for id in self.g.node_ids[mode]])
        #         modes.extend([self.g.nodes[id].state.mode.task_ids for id in self.g.node_ids[mode]])
        #     for mode in self.reached_modes:
        #         q_samples.extend([self.g.nodes[id].state.q.state() for id in self.g.transition_node_ids[mode]])
        #         modes.extend([self.g.nodes[id].state.mode.task_ids for id in self.g.transition_node_ids[mode]])
        #     data = {
        #         "q_samples": q_samples,
        #         "modes": modes,
        #         "path": self.current_best_path
        #     }
        #     save_data(data)
        #     print()
        #     q_samples = []
        #     modes = []
        #     vertices = list(self.g.vertices)
        #     q_samples.extend([self.g.nodes[id].state.q.state() for id in vertices])
        #     modes.extend([self.g.nodes[id].state.mode.task_ids for id in vertices])
        #     data = {
        #         "q_samples": q_samples,
        #         "modes": modes,
        #         "path": self.current_best_path
        #     }
        #     save_data(data)
        #     print()
        # self.sample_manifold()
        # q_samples = []
        # modes = []
        # for mode in self.reached_modes:
        #     q_samples.extend([self.g.nodes[id].state.q.state() for id in self.g.node_ids[mode]])
        #     modes.extend([self.g.nodes[id].state.mode.task_ids for id in self.g.node_ids[mode]])
        # for mode in self.reached_modes:
        #     q_samples.extend([self.g.nodes[id].state.q.state() for id in self.g.transition_node_ids[mode]])
        #     modes.extend([self.g.nodes[id].state.mode.task_ids for id in self.g.transition_node_ids[mode]])
        # data = {
        #     "q_samples": q_samples,
        #     "modes": modes,
        #     "path": self.current_best_path
        # }
        # save_data(data)
        # print()
        # if self.current_best_cost is not None:
        #     path = self.generate_path(True)
        #     if len(path) > 0:
        #         self.process_valid_path(path, force_update = True, update_queues=False )
        if self.current_best_cost is not None:
            self.current_best_path_nodes = self.generate_path(True)
            self.process_valid_path(self.current_best_path_nodes, False, True, True)

        if self.current_best_cost is not None:
            self.update_removal_conditions()        
        self.sample_manifold()
        self.g.compute_transition_lb_cost_to_come()
        self.g.compute_node_lb_cost_to_come()
        #just for comparison #TODO
        self.g.compute_transition_lb_cost_to_go()
        self.g.compute_node_lb_cost_to_go()

        self.initialze_forward_search()
        self.initialize_reverse_search()
        self.dynamic_reverse_search_update = False

    def initialze_forward_search(self):
        self.g.forward_queue = ForwardQueue(self.alpha)
        self.forward_closed_set = set()
        self.expand_node_forward(self.g.root)

    def initialize_reverse_search(self):
        self.g.reverse_queue = ReverseQueue()
        # if len(self.consistent_nodes) > 0:
        #     consistent_nodes = list(self.consistent_nodes)
        #     q_samples = []
        #     modes = []
        #     q_samples.extend([self.g.nodes[id].state.q.state() for id in consistent_nodes])
        #     modes.extend([self.g.nodes[id].state.mode.task_ids for id in consistent_nodes])

        #     data = {
        #         "q_samples": q_samples,
        #         "modes": modes,
        #         "path": self.current_best_path
        #     }
        #     save_data(data)
        #     print()
            

        self.consistent_nodes = set()
        self.g.reset_reverse_tree()
        self.g.reset_all_goal_nodes_lb_costs_to_go()
        print("Restart reverse search ...")
        self.reverse_search()
        print("... finished")
    
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
            is_edge_collision_free = self.env.is_edge_collision_free,
            node_cls=Node
            )
        # initialize all queues (edge-queues)   
        self.g.add_vertex_to_tree(self.g.root) 
        self.initialize_search()

    def Plan(
        self, optimize:bool = True
    ) -> Tuple[List[State], Dict[str, List[Union[float, float, List[State]]]]]:
        self.collision_cache = set()
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
            assert n0.id not in n1.blacklist, (
                "askdflö"
            )
            is_transition = False
            if n1.is_transition and not n1.is_reverse_transition and n1.transition is not None:
                is_transition = True
            # if n0.id != self.g.root.id:
            #     assert not np.isinf(n0.lb_cost_to_go), (
            #         "hjklö"
            #     )
            if np.isinf(n1.lb_cost_to_go):
                assert not [self.g.forward_queue.key(item)[0] for item in self.g.forward_queue.current_entries if not np.isinf(self.g.forward_queue.key(item)[0])], (
                "Forward queue contains non-infinite keys!")
            # already found best possible parent
            if n1.id in self.forward_closed_set:
                continue
            if (not np.isinf(n1.lb_cost_to_go) and (self.current_best_cost is None 
                or n0.cost + edge_cost + n1.lb_cost_to_go
                < self.current_best_cost)
                ):
                if n1.forward.parent == n0:  # if its already the parent
                    if is_transition:
                        self.expand_node_forward(n1.transition)
                    else:
                        self.expand_node_forward(n1)
                elif (
                    n0.cost + edge_cost < n1.cost
                ):  # parent can improve the cost
            
                    assert n0.id not in n1.forward.children, (
                        "Potential parent is already a child (forward)"
                    )
                    # check edge sparsely now. if it is not valid, blacklist it, and continue with the next edge
                    if n0.id not in n1.whitelist:
                        collision_free = False
                        if n0.id not in n1.blacklist:
                            collision_free = self.env.is_edge_collision_free(
                                n0.state.q,
                                n1.state.q,
                                n0.state.mode,
                                self.env.collision_resolution,
                            )
                            self.g.update_edge_collision_cache(n0, n1, collision_free)
                        else:
                            pass
                        if not collision_free:
                            self.g.forward_queue.remove((edge_cost, (n1, n0)))
                            assert (n0.id, n1.id) not in self.collision_cache, (
                                "kl,dö.fäghjk"
                            )
                            self.collision_cache.add((n0.id, n1.id))
                            self.collision_cache.add((n1.id, n0.id))
                            self.reverse_search((n0, n1))
                            continue
                    self.g.update_connectivity(n0, n1, edge_cost, n0.cost + edge_cost ,"forward", is_transition)
                    if self.current_best_cost is not None: 
                        assert (n1.cost + n1.lb_cost_to_go <= self.current_best_cost), (
                                "hjklö"
                            )
                    if is_transition:
                        print(n1.state.mode, n1.transition.state.mode)
                        self.expand_node_forward(n1.transition)
                    else:
                        self.expand_node_forward(n1)
                    with_shortcutting = False
                    if n1 in self.g.goal_nodes:
                        with_shortcutting = True
                    if self.dynamic_reverse_search_update or n1 in self.g.goal_nodes:
                        path = self.generate_path()
                        if len(path) > 0:
                            self.process_valid_path(path, with_shortcutting= with_shortcutting)
                            
            else:
                if np.isinf(n1.lb_cost_to_go):
                    assert (len(self.g.forward_queue.target_nodes - self.consistent_nodes) == len(self.g.forward_queue.target_nodes)), (
                        "jkslöadfägh"
                    ) # still can have inconsitent ones with a non inf lb cost to go 
                print("------------------------",n1.lb_cost_to_go)
                print("------------------------",n1.state.mode.task_ids)
                self.initialize_search()
                

            if not optimize and self.current_best_cost is not None:
                break

            if self.ptc.should_terminate(self.cnt, time.time() - self.start_time):
                break
        if self.costs != []:
            self.update_results_tracking(self.costs[-1], self.current_best_path)
        info = {"costs": self.costs, "times": self.times, "paths": self.all_paths}
        return self.current_best_path, info
