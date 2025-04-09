import numpy as np

from typing import (
    List,
    Dict,
    Tuple,
    Optional,
    Union, 
    Any, 
    Set
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
from multi_robot_multi_goal_planning.planners.itstar_base import (
    BaseITstar,
    BaseGraph,
    BaseOperation, 
    BaseNode,
    BaseTree
    )
from functools import cache
from multi_robot_multi_goal_planning.planners.planner_aitstar import EdgeQueue
# taken from https://github.com/marleyshan21/Batch-informed-trees/blob/master/python/BIT_Star.py
# needed adaption to work.

class InadmissibleHeuristics():
    __slots__ = [
        "lb_cost_to_go",
        "effort",    
    ]
    
    lb_cost_to_go: float
    effort: float

    def __init__(self):
        self.lb_cost_to_go = np.inf
        self.effort = np.inf

    def update(self, pot_parent:"Node", edge_cost:float, resolution:float):
        self.lb_cost_to_go = min(self.lb_cost_to_go, pot_parent.inad.lb_cost_to_go + edge_cost)
        if np.isinf(self.lb_cost_to_go):
            pass
        new_effort = pot_parent.inad.effort + edge_cost/resolution
        if new_effort < self.effort:
            self.effort = new_effort

    def reset_goal_nodes(self):
       self.lb_cost_to_go = 0.0
       self.effort = 0.0
    
    def reset(self):
        self.lb_cost_to_go = np.inf
        self.effort = np.inf

class Graph(BaseGraph):
    def __init__(self, 
                 root_state, 
                 operation, 
                 batch_dist_fun, 
                 batch_cost_fun, 
                 is_edge_collision_free,
                 collision_resolution, 
                 node_cls):
        super().__init__(root_state, operation, batch_dist_fun, batch_cost_fun, is_edge_collision_free, collision_resolution, node_cls)
    
        self.reverse_queue = None
        self.effort_estimate_queue = None
        self.cost_bound_queue = None
        self.epsilon = np.inf
        # self.g.cost_estimate_queue = None

    def reset_reverse_tree(self):
        [
            (
                setattr(self.nodes[node_id], "lb_cost_to_go", math.inf),
                self.nodes[node_id].rev.reset(),
                self.nodes[node_id].inad.reset()
            )
            for node_id in list(chain.from_iterable(self.node_ids.values()))
        ]
        [
            (
                setattr(self.nodes[node_id], "lb_cost_to_go", math.inf),
                self.nodes[node_id].rev.reset(),
                self.nodes[node_id].inad.reset()
            )
            for node_id in list(chain.from_iterable(self.transition_node_ids.values()))
        ]  # also includes goal nodes
        [
            (
                setattr(self.nodes[node_id], "lb_cost_to_go", math.inf),
                self.nodes[node_id].rev.reset(),
                self.nodes[node_id].inad.reset()
            )
              for node_id in list(chain.from_iterable(self.reverse_transition_node_ids.values()))
        ]

    def reset_all_goal_nodes_lb_costs_to_go(self):
        for node in self.goal_nodes:
            node.lb_cost_to_go = 0
            node.rev.cost_to_parent = 0
            node.inad.reset_goal_nodes()

    def update_forward_queue(self, edge_cost, edge):
        self.effort_estimate_queue.heappush((edge_cost, edge))
        self.cost_bound_queue.heappush((edge_cost, edge))
        # self.g.cost_estimate_queue.heappush((edge_cost, edge)) 

    def update_forward_queue_keys(self, type:str, node_ids:Optional[Set[BaseNode]] = None):
        if node_ids is not None and len(node_ids) == 0:
            return
        # if node_ids is not None:
        #     self.effort_estimate_queue.update(node_ids, type)
        # self.cost_estimate_queue.update(node_ids, type)
        self.cost_bound_queue.update(node_ids, type)
    
    def update_reverse_queue_keys(self, type:str, node_ids:Optional[Set[BaseNode]] = None):
        if node_ids is not None and len(node_ids) == 0:
            return
        self.reverse_queue.update(node_ids, type)

    def remove_forward_queue(self, edge_cost, n0, n1):
        self.cost_bound_queue.remove((edge_cost, (n1, n0))) 
        self.effort_estimate_queue.remove((edge_cost, (n1, n0)))
        # self.g.cost_estimate_queue.remove((edge_cost, (n1, n0)))
        self.cost_bound_queue.remove((edge_cost, (n0, n1))) 
        self.effort_estimate_queue.remove((edge_cost, (n0, n1)))
        # self.g.cost_estimate_queue.remove((edge_cost, (n0, n1)))

    def get_best_forward_edge(self):
        key, (edge_cost_ee,(ee0, ee1)) = self.effort_estimate_queue.peek_first_element()
        # key_ce, (edge_cost_ce,(ce0, ce1)) = self.cost_estimate_queue.peek_first_element()
        key_cb, (edge_cost_cb,(cb0, cb1)) = self.cost_bound_queue.peek_first_element()
        if np.isinf(cb1.lb_cost_to_go): 
            # if not np.isinf(key_cb):
            #     pass
            assert(np.isinf(key_cb)), (
                "key_cb is not inf"
            )
        item = (edge_cost_ee, (ee0, ee1))
        key_ee = self.cost_bound_queue.key(item)
        if key_ee < self.epsilon*key_cb:
            item = (edge_cost_ee, (ee0, ee1))
        # elif key_ce < key_cb:
        #     item = (edge_cost_ce, (ce0, ce1))
        else:
            item = (edge_cost_cb, (cb0, cb1))

        self.effort_estimate_queue.remove(item)
        # self.g.cost_estimate_queue.remove(item)
        self.cost_bound_queue.remove(item)
        # assert(key_ce == key_cb), (
        #     "ghjklö"
        # )
        return item

class Operation(BaseOperation):
    """Represents an operation instance responsible for managing variables related to path planning and cost optimization. """
    
    def __init__(self) -> None:
        super().__init__()

    def update(self, node:"Node", lb_cost_to_go:float = np.inf, cost:float = np.inf, lb_cost_to_come:float = np.inf):
        node.lb_cost_to_go = lb_cost_to_go
        self.lb_costs_to_come = self.ensure_capacity(self.lb_costs_to_come, node.id) 
        node.lb_cost_to_come = lb_cost_to_come
        self.costs = self.ensure_capacity(self.costs, node.id) 
        node.cost = cost

class Node(BaseNode):
    __slots__ = [
        "inad",
        "lb_effort_to_come"
    ]

    inad: InadmissibleHeuristics
    lb_effort_to_come: float

    def __init__(self, operation: "Operation", state: "State", is_transition: bool = False) -> None:
        super().__init__(operation, state, is_transition)
        self.inad = InadmissibleHeuristics()
        self.lb_effort_to_come = np.inf
    
    def close(self, resolution):
        self.inad.lb_cost_to_go = self.lb_cost_to_go
        self.inad.effort = self.rev.parent.inad.effort +  self.rev.cost_to_parent/resolution

    def set_to_goal_node(self):
        self.lb_cost_to_go = 0.0
        self.inad.reset_goal_nodes()
    
class ReverseQueue(EdgeQueue):
    def __init__(self, alpha =1.0, collision_resolution: Optional[float] = None):
        super().__init__(alpha, collision_resolution)
    
    def key(self, item: Tuple[Any]) -> float:
        # item[0]: edge_cost from n0 to n1
        # item[1]: edge (n0, n1)
        start_node, target_node = item[1] 
        return (start_node.inad.lb_cost_to_go + item[0] + target_node.lb_cost_to_come,
                start_node.inad.effort + item[0]/self.collision_resolution + target_node.lb_effort_to_come)
          
class EffortEstimateQueue(EdgeQueue):
    def __init__(self, alpha =1.0, collision_resolution: Optional[float] = None):
        super().__init__(alpha, collision_resolution)
    #remaining effort through an edge
    def key(self, item: Tuple[Any]) -> float:
        # item[0]: cost from n0 to n1
        # item[1]: edge (n0, n1)
        return item[0]/self.collision_resolution
    
class CostBoundQueue(EdgeQueue):
    def key(self, item: Tuple[Any]) -> float:
        # item[0]: edge_cost from n0 to n1
        # item[1]: edge (n0, n1)
        start_node, target_node = item[1] 
        return start_node.cost + item[0] + target_node.lb_cost_to_go

# class CostEstimateQueue(EdgeQueue):
#     def key(self, item: Tuple[Any]) -> float:
#         # item[0]: edge_cost from n0 to n1
#         # item[1]: edge (n0, n1)
#         return (item[1][0].cost + item[0] + item[1][1].inad.lb_cost_to_go)

class EITstar(BaseITstar):
    
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
        with_tree_visualization:bool = False,
        use_max_distance_metric_effort:bool = False,
        ):
        super().__init__(
            env = env, ptc=ptc, mode_sampling_type = mode_sampling_type, distance_metric = distance_metric, 
            try_sampling_around_path = try_sampling_around_path,try_informed_sampling = try_informed_sampling, 
            first_uniform_batch_size=first_uniform_batch_size, first_transition_batch_size=first_transition_batch_size,
            uniform_batch_size = uniform_batch_size, uniform_transition_batch_size = uniform_transition_batch_size, informed_batch_size = informed_batch_size, 
            informed_transition_batch_size = informed_transition_batch_size, path_batch_size = path_batch_size, locally_informed_sampling = locally_informed_sampling, 
            try_informed_transitions = try_informed_transitions, try_shortcutting = try_shortcutting, try_direct_informed_sampling = try_direct_informed_sampling, 
            informed_with_lb = informed_with_lb,remove_based_on_modes = remove_based_on_modes, with_tree_visualization = with_tree_visualization)

        self.sparse_number_of_points = 3
        self.use_max_distance_metric_effort = use_max_distance_metric_effort
        self.reverse_tree_set = set()
        self.reduce_neighbors = False
        self.sparesly_checked_edges = {}
        self.check = set()

    def _create_operation(self) -> BaseOperation:
        return Operation()
    
    def _create_graph(self, root_state) -> BaseGraph:
        return Graph(
            root_state=root_state,
            operation=self.operation,
            batch_dist_fun=lambda a, b, c=None: batch_config_dist(a, b, c or self.distance_metric),
            batch_cost_fun= lambda a, b: self.env.batch_config_cost(a, b),
            is_edge_collision_free = self.env.is_edge_collision_free,
            collision_resolution = self.env.collision_resolution,
            node_cls=Node
            )
           
    def continue_reverse_search(self, iter) -> bool:
        if len(self.g.reverse_queue) == 0 or len(self.g.cost_bound_queue) == 0:
            return False
        if iter > 0 and len(self.updated_target_nodes) == 0:
            return True
        #update forward queue
        self.g.update_forward_queue_keys('target', self.updated_target_nodes)
        self.updated_target_nodes = set()
        if len(self.g.reverse_queue) == 0 or len(self.g.cost_bound_queue) == 0:
            return False
        if np.isinf(self.g.epsilon):
            target_nodes_in_reverse_tree = self.g.cost_bound_queue.target_nodes - self.reverse_tree_set
            if len(target_nodes_in_reverse_tree) != len(self.g.cost_bound_queue.target_nodes):
                return False
        not_closed_nodes = self.g.cost_bound_queue.target_nodes - self.reverse_closed_set
        if len(not_closed_nodes) == 0:
            return False
        forward_key, forward_item = self.g.cost_bound_queue.peek_first_element()
        reverse_key, _ = self.g.reverse_queue.peek_first_element()
        if forward_item[1][1].id in self.reverse_closed_set and forward_key <= reverse_key[0]:
            return False
        return True
    
    def update_reverse_sets(self, node):
        self.reverse_closed_set.add(node.id)
        self.reverse_tree_set.add(node.id)
           
    def expand_node_reverse(self, nodes: List[Node]) -> None:
        for node in nodes:
            if node.id == self.g.root.id:
                return  
            # if node.id in self.reverse_tree_set:
            #     self.g.reverse_queue.update({node.id}, 'start')
            #     return
            self.reverse_tree_set.add(node.id)
            self.updated_target_nodes.add(node.id)
            neighbors = self.g.get_neighbors(node, space_extent=self.approximate_space_extent)
            if neighbors.size == 0:
                return          
            edge_costs = self.g.tot_neighbors_batch_cost_cache[node.id]
            for id, edge_cost in zip(neighbors, edge_costs):
                n = self.g.nodes[id]
                assert (n.forward.parent == node) == (n.id in node.forward.children), (
                        f"Parent and children don't coincide (reverse): parent: {node.id}, child: {n.id}"
                        )
                if n in self.g.goal_nodes or n.id == self.g.root.id:
                    continue
                assert(n.id not in node.blacklist), (
                "neighbors are wrong")
                if node.rev.parent is not None and node.rev.parent.id == n.id:
                    continue
                if n.is_transition and not n.is_reverse_transition:
                    continue

                if n.id in self.reverse_closed_set and not self.dynamic_reverse_search_update:
                    continue
                
                if n.id in self.reverse_tree_set:
                    if not self.dynamic_reverse_search_update:
                        continue
                    if n.lb_cost_to_go < node.lb_cost_to_go + edge_cost:
                        assert(n.lb_cost_to_go- (node.lb_cost_to_go + edge_cost) < 1e-5), (
                                    f"{n.id}, {node.id}, qwdfertzj"
                                )
                        continue
                if self.current_best_cost is not None:
                    if node.lb_cost_to_go + edge_cost + n.lb_cost_to_come > self.current_best_cost:
                        continue
                edge = (node, n)
                # print(node.id, n.id)
                if edge in self.check:
                    pass
                self.check.add(edge)

                self.g.reverse_queue.heappush((edge_cost, edge))
                 
    def update_inflation_factor(self):
        # if self.current_best_cost is None:
        if not self.dynamic_reverse_search_update:
            self.g.epsilon = np.inf
        else:
            self.g.epsilon = 1

    @cache
    def get_sparse_num_of_collision_checks(self, edge_cost, N) -> float:
        r = edge_cost/N
        if r < self.env.collision_resolution:
            return None
        return N

    def reverse_search(self):
        self.updated_target_nodes = set()
        iter = 0
        while self.continue_reverse_search(iter):
            # if self.reverse_tree_set and self.with_tree_visualization:
            #     self.save_tree_data((BaseTerree.all_vertices, self.reverse_tree_set))
            # print(iter, self.reverse_tree_set)
            iter +=1
            edge_cost, edge = self.g.reverse_queue.heappop()
            n0, n1 = edge
            # if not self.dynamic_reverse_search_update and n1.id in self.reverse_tree_set:
            #     continue
            self.reverse_closed_set.add(n0.id)
            is_transition = False
            if n1.is_transition:
                pass
            if n1.is_transition and n1.is_reverse_transition:
                is_transition = True
            if n0.id not in n1.whitelist:
                sparsely_collision_free = False
                if n0.id not in n1.blacklist:
                    sparse_num_of_collision_checks = self.get_sparse_num_of_collision_checks(edge_cost, self.sparse_number_of_points)
                    sparsely_collision_free = self.env.is_edge_collision_free(
                                n1.state.q,
                                n0.state.q,
                                n0.state.mode,
                                N =sparse_num_of_collision_checks
                            )
                    #TODO if already checked with env resolution add to whitelist!
                    if edge not in self.sparesly_checked_edges:
                        self.sparesly_checked_edges[edge] = 0
                    self.sparesly_checked_edges[edge] += 1
                if not sparsely_collision_free:
                    self.g.remove_forward_queue(edge_cost, n0, n1)
                    self.g.update_edge_collision_cache(n0, n1, False)
                    continue
            n1.inad.update(n0, edge_cost, self.env.collision_resolution)
            if is_transition:
                n1.transition.inad.update(n1, 0.0, self.env.collision_resolution)
            potential_lb_cost_to_go = n0.lb_cost_to_go + edge_cost
            if n1.lb_cost_to_go > potential_lb_cost_to_go:
                self.g.update_connectivity(n0, n1, edge_cost, potential_lb_cost_to_go,"reverse", is_transition)

                assert (n1.lb_cost_to_go == n1.inad.lb_cost_to_go), (
                    "ghjklö"
                )
                if is_transition:
                    self.reverse_tree_set.add(n1.id)
                    self.expand_node_reverse([n1.transition])
                    continue
                self.expand_node_reverse([n1])

        if self.with_tree_visualization and iter > 0:
            self.save_tree_data((BaseTree.all_vertices, self.reverse_tree_set))
                    
    def clear_reverse_edge_in_collision(self, n0, n1):
        if self.with_tree_visualization and (BaseTree.all_vertices or self.reverse_tree_set):
            self.save_tree_data((BaseTree.all_vertices, self.reverse_tree_set))
        if n1.rev.parent is not None and n1.rev.parent.id == n0.id:
            nodes_to_update = self.update_reverse_node_of_children(n1)
        elif n0.rev.parent is not None and n0.rev.parent.id == n1.id:
            nodes_to_update = self.update_reverse_node_of_children(n0)
        self.g.update_forward_queue_keys('target', nodes_to_update)
        self.g.update_reverse_queue_keys('start', nodes_to_update)

        if self.with_tree_visualization and (BaseTree.all_vertices or self.reverse_tree_set):
            self.save_tree_data((BaseTree.all_vertices, self.reverse_tree_set))
    
    def update_reverse_node_of_children(self, n: BaseNode) -> set[int]:
        nodes_to_update = set()
        n.lb_cost_to_go = np.inf
        n.inad.reset()
        self.reverse_closed_set.discard(n.id)
        self.reverse_tree_set.discard(n.id)
        n.rev.parent.rev.children.remove(n.id)
        n.rev.parent.rev.fam.remove(n.id)
        if n.is_transition and n.transition is not None:
            nodes_to_update.add(n.transition.id)
        nodes_to_update.add(n.id)
        if len(n.rev.children) == 0:
            n.rev.reset()
            return nodes_to_update
        stack = [n.id]
        while stack:
            current_id = stack.pop()
            current_node = self.g.nodes[current_id]
            children = current_node.rev.children
            current_node.rev.reset()
            if children:
                for _, id in enumerate(children):
                    child = self.g.nodes[id]
                    child.lb_cost_to_go = np.inf
                    if child in self.g.goal_nodes:
                        child.lb_cost_to_go = 0.0
                    child.inad.reset()
                    self.reverse_closed_set.discard(child.id)
                    self.reverse_tree_set.discard(child.id)
                    nodes_to_update.add(child.id)
                stack.extend(children)
        return nodes_to_update

    def manage_edge_in_collision(self, n0, n1):
        if n0.rev.parent == n1 or n1.rev.parent == n0:
            if not self.dynamic_reverse_search_update:
                self.clear_reverse_edge_in_collision(n0, n1)
                return
            self.sparse_number_of_points +=1
            # self.clear_reverse_edge_in_collision(n0, n1)
            self.initialize_reverse_search(False)    

    def initialize_lb(self):
        calculate_effort = True
        if self.use_max_distance_metric_effort:
            calculate_effort = False
            self.g.compute_transition_lb_effort_to_come()
            self.g.compute_node_lb_effort_to_come()
        self.g.compute_transition_lb_to_come(calculate_effort)
        self.g.compute_node_lb_to_come(calculate_effort)

    def initialze_forward_search(self):
        if self.current_best_cost is not None:
            self.g.weight = 0.5
        self.update_inflation_factor()
        self.g.effort_estimate_queue = EffortEstimateQueue(collision_resolution=self.env.collision_resolution)
        self.g.cost_bound_queue = CostBoundQueue()
        # self.g.cost_estimate_queue = CostEstimateQueue()
        self.expand_node_forward(self.g.root)

    def initialize_reverse_search(self, reset:bool = True):
        if len(BaseTree.all_vertices) > 1:
            if self.with_tree_visualization and (BaseTree.all_vertices or self.reverse_tree_set):
                self.save_tree_data((BaseTree.all_vertices, self.reverse_tree_set))
        self.reverse_closed_set = set() #node was as start node visited in the reverse search
        self.reverse_tree_set = set() #node was extended in the reverse search
        self.g.reverse_queue = ReverseQueue(collision_resolution=self.env.collision_resolution)
        self.g.reset_reverse_tree()
        self.g.reset_all_goal_nodes_lb_costs_to_go()
        self.updated_target_nodes = set() #lb_cost_to_go was updated in reverse search
        if reset:
            self.sparse_number_of_points = 3
        self.expand_node_reverse(self.g.goal_nodes) 
        self.g.update_forward_queue_keys('target') 
 
    def Plan(
        self,optimize:bool = True
    ) -> Tuple[List[State], Dict[str, List[Union[float, float, List[State]]]]]:
        self.collision_cache = set()
        self.PlannerInitialization()
        num_iter = 0
        n1 = None
        while True:
            num_iter += 1
            self.reverse_search()
            if num_iter % 100000 == 0:
                print("Forward Queue: ", len(self.g.cost_bound_queue))
            if len(self.g.cost_bound_queue) < 1:
                print("------------------------",n1.state.mode.task_ids)
                self.initialize_search()
                continue
            
            key, _ = self.g.cost_bound_queue.peek_first_element()
            if self.current_best_cost is None or key < self.current_best_cost:
                edge_cost, (n0, n1) = self.g.get_best_forward_edge()
                assert n0.id not in n1.blacklist, (
                    "askdflö"
                )
                is_transition = False
                if n1.is_transition and not n1.is_reverse_transition and n1.transition is not None:
                    is_transition = True
                
                if n1 in BaseTree.all_vertices and not self.dynamic_reverse_search_update:
                    continue

                # if np.isinf(n1.lb_cost_to_go):
                #     assert not [self.g.cost_bound_queue.key(item) for item in self.g.cost_bound_queue.current_entries if not np.isinf(self.g.cost_bound_queue.key(item))], (
                #     "Forward queue contains non-infinite keys!")
                # already found best possible parent
           
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
                        if not collision_free:
                            self.g.remove_forward_queue(edge_cost, n0, n1)
                            self.manage_edge_in_collision(n0, n1)
                            continue
                    self.g.update_connectivity(n0, n1, edge_cost, n0.cost + edge_cost,"forward", is_transition)
                    if self.current_best_cost is not None: 
                        assert (n1.cost + n1.lb_cost_to_go <= self.current_best_cost), (
                                "hjklö"
                            )
                    if is_transition:
                        print(n1.state.mode, n1.transition.state.mode)
                        self.expand_node_forward(n1.transition)
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
                            self.update_inflation_factor()
                            if self.with_tree_visualization and (BaseTree.all_vertices or self.reverse_tree_set):
                                self.save_tree_data((BaseTree.all_vertices, self.reverse_tree_set))

                            
      
            else:
                self.initialize_search()
                
            if not optimize and self.current_best_cost is not None:
                break

            if self.ptc.should_terminate(self.cnt, time.time() - self.start_time):
                break
        if self.costs != []:
            self.update_results_tracking(self.costs[-1], self.current_best_path)
        info = {"costs": self.costs, "times": self.times, "paths": self.all_paths}
        return self.current_best_path, info
