import numpy as np

from typing import (
    List,
    Dict,
    Tuple,
    Optional,
    Set,
    Any,
    Union,
)

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
from multi_robot_multi_goal_planning.planners.rrtstar_base_old import save_data
from multi_robot_multi_goal_planning.planners.itstar_base import (
    BaseITstar,
    BaseOperation,
    BaseGraph,
    DictIndexHeap,
    BaseNode, 
    BaseTree
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
                 root_state, 
                 operation, 
                 distance_metric,
                 batch_dist_fun, 
                 batch_cost_fun, 
                 is_edge_collision_free,
                 get_next_modes,
                 collision_resolution, 
                 node_cls):
        super().__init__(root_state=root_state, operation=operation, distance_metric=distance_metric, 
                         batch_dist_fun=batch_dist_fun, batch_cost_fun=batch_cost_fun, is_edge_collision_free=is_edge_collision_free, 
                         get_next_modes=get_next_modes, collision_resolution=collision_resolution, 
                         node_cls=node_cls, including_effort=False)
        self.reverse_queue = None
        self.forward_queue = None

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

    def update_forward_queue(self, edge_cost, edge):
        self.forward_queue.heappush((edge_cost, edge))
           
    def update_forward_queue_keys(self, type:str ,node_ids:Optional[Set[BaseNode]] = None):
        self.forward_queue.update(node_ids, type)
    
    def update_reverse_queue_keys(self, type:str, node_ids:Optional[Set[BaseNode]] = None):
        if type == "start":
            return
        self.reverse_queue.update(node_ids, type)

    def remove_forward_queue(self, edge_cost, n0, n1):
        self.forward_queue.remove((edge_cost, (n1, n0))) 
        self.forward_queue.remove((edge_cost, (n0, n1))) 

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
class EdgeQueue(DictIndexHeap[Tuple[Any]]):
    def __init__(self, alpha = 1.0, collision_resolution: Optional[float] = None):
        super().__init__(collision_resolution)
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
   
    def add_and_sync(self, item):
        start_node, target_node = item[1]

        self.target_nodes.add(target_node.id)
        self.start_nodes.add(start_node.id)

        self.target_nodes_with_item.setdefault(target_node.id, set()).add(item)
        self.start_nodes_with_item.setdefault(start_node.id, set()).add(item)
    
    def remove(self, item, in_current_entries:bool = False):
        if not in_current_entries and item not in self.current_entries:
           return
        start_node_id, target_node_id = item[1][0].id, item[1][1].id
        self.target_nodes_with_item[target_node_id].remove(item)
        self.start_nodes_with_item[start_node_id].remove(item)
        del self.current_entries[item]

        if not self.target_nodes_with_item[target_node_id]:
            self.target_nodes.discard(target_node_id)

        if not self.start_nodes_with_item[start_node_id]:
            self.start_nodes.discard(start_node_id)

    def update(self, node_ids: Optional[Set[BaseNode]], type: str):
        if node_ids is not None and not node_ids:
            return

        before = len(self.current_entries)
        cnt = 0

        if type == 'start':
            node_set = self.start_nodes
            nodes_with_items = self.start_nodes_with_item
        elif type == 'target':
            node_set = self.target_nodes
            nodes_with_items = self.target_nodes_with_item
        else:
            raise ValueError(f"Unknown update type: {type}")

        if not node_set:
            return

        update_nodes = node_set if node_ids is None else node_ids & node_set
        if not update_nodes:
            return

        heappush = self.heappush
        current_entries = self.current_entries

        for nid in update_nodes:
            for item in nodes_with_items[nid]:
                assert item in current_entries, "ijfdlk"
                heappush(item)
                cnt += 1

        assert before == len(current_entries), "hjk,l"
    
class VertexQueue(DictIndexHeap[Node]):
    def __init__(self):
        super().__init__()
        self.nodes = set()
        self.nodes_with_item = dict()
    
    def add_and_sync(self, item: Node):
        self.nodes.add(item.id)
        self.nodes_with_item[item.id] = item
    
    def remove(self, item, in_current_entries:bool = False):
        if not in_current_entries and item not in self.current_entries:
           return
        del self.current_entries[item]
        self.nodes.remove(item.id)
        # del self.nodes_with_item[item.id]

    def key(self, node: Node) -> float:
        min_lb = min(node.lb_cost_to_go, node.lb_cost_to_go_expanded) 
        return (min_lb + node.lb_cost_to_come, min_lb)

    def update(self, node_ids:Optional[Set[BaseNode]], type:str):
        if node_ids is None:
            node_ids = self.nodes
        if len(node_ids) == 0:
            return
        cnt = 0
        before = (len(self.current_entries))
        for id in self.nodes:
            if id not in node_ids:
                continue
            item = self.nodes_with_item[id]
            self.heappush(item)
            cnt +=1
        assert before == len(self.current_entries), (
        "hjk,l")

class AITstar(BaseITstar):
    def __init__(
        self,
        env: BaseProblem,
        ptc: PlannerTerminationCondition,
        init_mode_sampling_type: str = "greedy",
        distance_metric: str = "euclidean",
        try_sampling_around_path: bool = True,
        try_informed_sampling: bool = True,
        init_uniform_batch_size: int = 100,
        init_transition_batch_size:int = 100,
        uniform_batch_size: int = 200,
        uniform_transition_batch_size: int = 500,
        informed_batch_size: int = 500,
        informed_transition_batch_size: int = 500,
        path_batch_size: int = 500,
        locally_informed_sampling: bool = True,
        try_informed_transitions: bool = True,
        try_shortcutting: bool = True,
        try_direct_informed_sampling: bool = True,
        inlcude_lb_in_informed_sampling:bool = True,
        remove_based_on_modes:bool = False,
        with_tree_visualization:bool = False,
        apply_long_horizon:bool = False,
        frontier_mode_sampling_probability:float = 1.0,
        horizon_length: int = 1,
        ):
        super().__init__(
            env = env, ptc=ptc, init_mode_sampling_type = init_mode_sampling_type, distance_metric = distance_metric, 
            try_sampling_around_path = try_sampling_around_path,try_informed_sampling = try_informed_sampling, 
            init_uniform_batch_size=init_uniform_batch_size, init_transition_batch_size=init_transition_batch_size,
            uniform_batch_size = uniform_batch_size, 
            uniform_transition_batch_size = uniform_transition_batch_size, informed_batch_size = informed_batch_size, 
            informed_transition_batch_size = informed_transition_batch_size, 
            path_batch_size = path_batch_size, locally_informed_sampling = locally_informed_sampling, 
            try_informed_transitions = try_informed_transitions, try_shortcutting = try_shortcutting, 
            try_direct_informed_sampling = try_direct_informed_sampling, 
            inlcude_lb_in_informed_sampling = inlcude_lb_in_informed_sampling,
            remove_based_on_modes = remove_based_on_modes, with_tree_visualization = with_tree_visualization,
            apply_long_horizon = apply_long_horizon, frontier_mode_sampling_probability=frontier_mode_sampling_probability,
            horizon_length = horizon_length)

        self.alpha = 3.0
        self.consistent_nodes = set() #lb_cost_to_go_expanded == lb_cost_to_go
        self.no_available_parent_in_this_batch = set() #nodes that have no available parent in this batch
        self.init_rev_search = True
        self.reduce_neighbors = True

    def _create_operation(self) -> BaseOperation:
        return Operation()
    
    def _create_graph(self,root_state) -> BaseGraph:
        return Graph(
            root_state=root_state,
            operation=self.operation,
            distance_metric=self.distance_metric,
            batch_dist_fun=lambda a, b, c=None: batch_config_dist(a, b, c or self.distance_metric),
            batch_cost_fun= lambda a, b: self.env.batch_config_cost(a, b),
            is_edge_collision_free = self.env.is_edge_collision_free,
            get_next_modes = self.env.get_next_modes,
            collision_resolution = self.env.collision_resolution,
            node_cls=Node
            )
        
    def inconcistency_check(self, node: Node):  
        self.g.reverse_queue.remove(node)
        if node.lb_cost_to_go != node.lb_cost_to_go_expanded:
            self.g.reverse_queue.heappush(node)
        else:
            pass
   
    def continue_reverse_search(self, iter) -> bool:
        if len(self.g.reverse_queue) == 0 or len(self.g.forward_queue) == 0:
            return False
        if iter > 0 and len(self.updated_target_nodes) == 0:
            return True
        self.g.update_forward_queue_keys('target', self.updated_target_nodes)
        self.reverse_tree_set.update(self.updated_target_nodes)
        self.updated_target_nodes = set()

        forward_key, item = self.g.forward_queue.peek_first_element()
        # if not self.dynamic_reverse_search_update :
        #     if not np.isinf(item[1][1].lb_cost_to_go):
        #         return False
        if not self.dynamic_reverse_search_update and not self.init_rev_search:
            target_nodes_in_reverse_tree = self.g.forward_queue.target_nodes - self.reverse_tree_set
            if len(target_nodes_in_reverse_tree) != len(self.g.forward_queue.target_nodes):
                return False

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
        
    def invalidate_rev_branch(self, n: Node, nodes_to_update) -> set[int]:
        # Only reset if not a goal node
        if n not in self.g.goal_nodes:
            # Reset cost values
            n.lb_cost_to_go = np.inf
            n.lb_cost_to_go_expanded = np.inf
        else:
            n.lb_cost_to_go = 0.0
            n.lb_cost_to_go_expanded = 0.0

        # Remove from various tracking sets
        self.reverse_closed_set.discard(n.id)
        self.reverse_tree_set.discard(n.id)
        self.consistent_nodes.discard(n.id)
        self.g.reverse_queue.remove(n)

        # Disconnect from parent's children/family (if parent exists)
        if n.rev.parent:
            if n.id in n.rev.parent.rev.children:
                n.rev.parent.rev.children.remove(n.id)
            n.rev.parent.rev.fam.discard(n.id)

        nodes_to_update.add(n.id)

        # Recurse on children
        for child_id in list(n.rev.children):
            child = self.g.nodes[child_id]
            child_nodes = self.invalidate_rev_branch(child, nodes_to_update)
            nodes_to_update.update(child_nodes)

        # Reset current node's reverse state (after children handled)
        n.rev.reset()

        # Update current node
        if self.dynamic_reverse_search_update:
            self.update_state(n)

        return nodes_to_update

    def reverse_search(self, edge: Optional[Tuple[Node, Node]] = None) -> float:
        self.reverse_closed_set = set() # node was visited in reverse search
        self.reverse_tree_set.update(self.updated_target_nodes)
        self.updated_target_nodes = set()
        if edge is not None:
            nodes_to_update = set()
            if self.with_tree_visualization:
                self.save_tree_data((BaseTree.all_vertices, self.reverse_tree_set))
            if edge[0].rev.parent == edge[1]:
                nodes_to_update = self.invalidate_rev_branch(edge[0], nodes_to_update)
            elif edge[1].rev.parent == edge[0]:
                nodes_to_update = self.invalidate_rev_branch(edge[1], nodes_to_update)
            else: #doesn't effect reverse search
                return
            self.g.update_forward_queue_keys('target', nodes_to_update) 
            self.reverse_tree_set.update(nodes_to_update)
            if self.with_tree_visualization:
                self.save_tree_data((BaseTree.all_vertices, self.reverse_tree_set))
            # if not self.dynamic_reverse_search_update:
            #     return

        # Process the reverse queue until stopping conditions are met.
        num_iter = 0
        while self.continue_reverse_search(num_iter): 
            if self.ptc.should_terminate(self.cnt, time.time() - self.start_time):
                break
            n = self.g.reverse_queue.heappop()
            self.reverse_closed_set.add(n.id)
            num_iter += 1
            if num_iter % 100000 == 0:
                print(num_iter, ": Reverse Queue: ", len(self.g.reverse_queue))
            is_rev_transition = False
            if n.is_transition and n.is_reverse_transition:
                is_rev_transition = True
                # if n.transition_neighbors[0].lb_cost_to_go < n.lb_cost_to_go_expanded:
                #     #don't change the parent
                #     self.update_heuristic_of_neihgbors(n.transition_neighbors[0])
                #     continue
            
            if n.lb_cost_to_go < n.lb_cost_to_go_expanded:
                self.consistent_nodes.add(n.id)
                n.lb_cost_to_go_expanded = n.lb_cost_to_go
                if is_rev_transition:
                    assert len(n.transition_neighbors) == 1, (
                        "Transition neighbor should be only one"
                    )
                    n.transition_neighbors[0].lb_cost_to_go_expanded = n.lb_cost_to_go_expanded
                    self.consistent_nodes.add(n.transition_neighbors[0].id)
                if n.is_transition and not n.is_reverse_transition:
                    pass
            else:
                self.consistent_nodes.discard(n.id)
                n.lb_cost_to_go_expanded = np.inf 
                if is_rev_transition:
                    n.transition_neighbors[0].lb_cost_to_go_expanded = n.lb_cost_to_go_expanded
                    self.consistent_nodes.discard(n.transition_neighbors[0].id)
                self.update_state(n)

            assert  n.lb_cost_to_go_expanded == n.lb_cost_to_go or (np.isinf(n.lb_cost_to_go_expanded)), (
                "lb_cost_to_go_expanded should not be finite and different from lb_cost_to_go"
            )

            self.update_heuristic_of_neihgbors(n)
            if is_rev_transition:
                # assert math.isclose(n.transition_neighbors[0].lb_cost_to_go, n.lb_cost_to_go, rel_tol=1e-10, abs_tol=1e-5), (
                # "ohhh")
                # assert math.isclose(n.transition_neighbors[0].lb_cost_to_go_expanded, n.lb_cost_to_go_expanded, rel_tol=1e-10, abs_tol=1e-5), (
                # "ohhh")

                # assert(n.lb_cost_to_go == n.lb_cost_to_go_expanded), ("ohhh")
                self.update_heuristic_of_neihgbors(n.transition_neighbors[0])
        if self.with_tree_visualization and num_iter > 0:
            self.save_tree_data((BaseTree.all_vertices, self.reverse_tree_set))
        self.init_rev_search = False
        
    def update_heuristic_of_neihgbors(self, n):
        neighbors = self.g.get_neighbors(n, self.approximate_space_extent)
        if 1 not in neighbors:
            pass
        # batch_cost = self.g.tot_neighbors_batch_cost_cache[n.id]
        for idx, id in enumerate(neighbors):  #node itself is not included in neighbors
            nb = self.g.nodes[id]
            if nb in self.g.goal_nodes:
                continue
            if nb.id in self.reverse_closed_set and nb.id in self.consistent_nodes:
                continue
            if nb in self.no_available_parent_in_this_batch:
                continue
            self.update_state(nb)
                        
    def update_state(self, node: Node) -> None:
        if node.id == self.g.root.id or node in self.g.goal_nodes:
            return
        if node.is_transition and not node.is_reverse_transition:
            return
        is_rev_transition = False
        if node.is_transition and node.is_reverse_transition:
            #only want to consider nodes as reverse parent with mode of node or its next modes
            is_rev_transition = True
        # node was already expanded in current heuristic call
        if node.lb_cost_to_go == node.lb_cost_to_go_expanded and node.id in self.reverse_closed_set:
            self.g.reverse_queue.remove(node)
            return
    
        # in_updated_set = True
        neighbors = list(self.g.get_neighbors(node, self.approximate_space_extent))
        if len(neighbors) == 0:
            self.update_node_without_available_reverse_parent(node)
            return
        
        batch_cost = self.g.tot_neighbors_batch_cost_cache[node.id]
        if is_rev_transition:
            #only want to consider nodes as reverse parent with mode of node or its next modes
            idx = neighbors.index(node.transition_neighbors[0].id)
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
                self.updated_target_nodes.add(node.id)
                node.lb_cost_to_go = best_lb_cost_to_go
                if node.is_reverse_transition: 
                    self.updated_target_nodes.add(node.transition_neighbors[0].id)
                    node.transition_neighbors[0].lb_cost_to_go = best_lb_cost_to_go
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
            if is_rev_transition:
                if n.is_transition:
                    if node.state.mode != n.state.mode:
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
            self.no_available_parent_in_this_batch.add(node.id) 
            self.update_node_without_available_reverse_parent(node)
            return
        if is_rev_transition and node.transition_neighbors[0].rev.parent is not None:
            if node.transition_neighbors[0].rev.parent.id != node.id:
                potential_cost_to_go = best_parent.lb_cost_to_go_expanded + best_edge_cost
                if node.transition_neighbors[0].lb_cost_to_go < potential_cost_to_go:
                    # nodes_to_update = set()
                    # nodes_to_update = self.invalidate_rev_branch(best_parent, nodes_to_update)
                    # self.g.update_forward_queue_keys('target', nodes_to_update) 
                    # self.reverse_tree_set.update(nodes_to_update)
                    self.update_node_without_available_reverse_parent(node, update_transition=False)
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
            self.updated_target_nodes.add(node.id)
            if is_rev_transition:
                self.updated_target_nodes.add(node.transition_neighbors[0].id)
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

    def update_node_without_available_reverse_parent(self, node:Node, update_transition:bool = True):
        if len(node.rev.fam) == 0 and np.isinf(node.lb_cost_to_go) and node.id not in self.consistent_nodes:
            return
        self.consistent_nodes.discard(node.id)
        if not np.isinf(node.lb_cost_to_go):
            self.updated_target_nodes.add(node.id)
        if node not in self.g.goal_nodes:
            node.lb_cost_to_go = np.inf
            node.lb_cost_to_go_expanded = np.inf
            node.rev.cost_to_parent = None
        if node.rev.parent is not None:
            node.rev.parent.rev.children.remove(node.id)
            node.rev.parent.rev.fam.remove(node.id)
            node.rev.fam.remove(node.rev.parent.id) 
        node.rev.parent = None
        self.g.reverse_queue.remove(node)
        if node.is_transition and node.is_reverse_transition and update_transition:
            self.update_node_without_available_reverse_parent(node.transition_neighbors[0])
         
    def initialize_lb(self):    
        self.g.compute_transition_lb_cost_to_come()
        self.g.compute_node_lb_to_come()

    def initialze_forward_search(self):
        # if self.current_best_cost is not None:
        #     self.g.weight = 0.5
        # else:
        #     self.g.weight = 1
        self.init_rev_search = True
        self.no_available_parent_in_this_batch = set()
        self.g.forward_queue = EdgeQueue(self.alpha)
        self.expand_node_forward(self.g.root, regardless_forward_closed_set = True, first_search=self.first_search)

    def initialize_reverse_search(self, reset:bool=True):
        self.g.reverse_queue = VertexQueue()
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
        self.reverse_tree_set = set()
        self.g.reset_reverse_tree()
        self.g.reset_all_goal_nodes_lb_costs_to_go()
        print("Restart reverse search ...")
        self.reverse_search()
        print("... finished")
        self.g.update_forward_queue_keys('target') 
    
    def update_reverse_sets(self, node):
        self.reverse_closed_set.add(node.id)
        self.consistent_nodes.add(node.id)
        self.reverse_tree_set.add(node.id)

    def Plan(
        self, optimize:bool = True
    ) -> Tuple[List[State], Dict[str, List[Union[float, float, List[State]]]]]:
        self.collision_cache = set()
        self.PlannerInitialization()
        num_iter = 0
        n1 = None
        while True:
            num_iter += 1
            # self.reverse_search()
            if num_iter % 100000 == 0:
                print("Forward Queue: ", len(self.g.forward_queue))
            if len(self.g.forward_queue) < 1:
                self.initialize_search()
                continue
            edge_cost, (n0, n1) = self.g.forward_queue.heappop()
            assert n0.id not in n1.blacklist, (
                "askdflö"
            )
            is_transition = False
            if n1.is_transition and not n1.is_reverse_transition and n1.transition_neighbors:
                is_transition = True
            if np.isinf(n1.lb_cost_to_go):
                assert not [self.g.forward_queue.key(item)[0] for item in self.g.forward_queue.current_entries if not np.isinf(self.g.forward_queue.key(item)[0])], (
                "Forward queue contains non-infinite keys!")
            # already found best possible parent
            if n1 in BaseTree.all_vertices:
                    continue
            if (not np.isinf(n1.lb_cost_to_go) and (self.current_best_cost is None 
                or n0.cost + edge_cost + n1.lb_cost_to_go
                < self.current_best_cost)
                ):
                if n1.forward.parent == n0:  # if its already the parent
                    if is_transition:
                        for transition in n1.transition_neighbors:
                            self.expand_node_forward(transition)
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
                                n0.state.mode
                            )
                            self.g.update_edge_collision_cache(n0, n1, collision_free)
                        else:
                            pass
                        if not collision_free:
                            self.g.remove_forward_queue(edge_cost, n0, n1)
                            self.reverse_search((n0, n1))
                            # self.manage_edge_in_collision(n0, n1)
                            continue
                    self.g.update_connectivity(n0, n1, edge_cost, n0.cost + edge_cost ,"forward", is_transition)
                    if self.current_best_cost is not None: 
                        assert (n1.cost + n1.lb_cost_to_go <= self.current_best_cost), (
                                "hjklö"
                            )
                    if is_transition:
                        for transition in n1.transition_neighbors:
                            self.expand_node_forward(transition)
                    else:
                        self.expand_node_forward(n1)
                    update = False
                    if n1 in self.g.goal_nodes:
                        update = True
                    if self.dynamic_reverse_search_update or n1 in self.g.goal_nodes:
                        path = self.generate_path()
                        if len(path) > 0:
                            if self.with_tree_visualization and (BaseTree.all_vertices or self.reverse_tree_set):
                                self.save_tree_data((BaseTree.all_vertices, self.reverse_tree_set))
                            self.process_valid_path(path, with_shortcutting= update, with_queue_update=update)
                            if self.with_tree_visualization and (BaseTree.all_vertices or self.reverse_tree_set):
                                self.save_tree_data((BaseTree.all_vertices, self.reverse_tree_set))
           
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
