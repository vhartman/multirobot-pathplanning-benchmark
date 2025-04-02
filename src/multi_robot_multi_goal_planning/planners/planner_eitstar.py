import numpy as np

from typing import (
    List,
    Dict,
    Tuple,
    Optional,
    Union, 
    Any, 
    ClassVar, 
    Set
)
import time
import heapq
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
    DictIndexHeap, 
    BaseOperation, 
    BaseNode
)
from functools import lru_cache
from multi_robot_multi_goal_planning.planners.planner_aitstar import ForwardQueue
# taken from https://github.com/marleyshan21/Batch-informed-trees/blob/master/python/BIT_Star.py
# needed adaption to work.

class InadmissibleHeuristics():
    __slots__ = [
        "lb_cost_to_go",
        "effort",
        "effort_to_parent"     
    ]
    
    lb_cost_to_go: float
    effort: float
    effort_to_parent: float

    def __init__(self):
        self.lb_cost_to_go = np.inf
        self.effort = np.inf
        self.effort_to_parent = np.inf

    def update(self, pot_parent:"Node", edge_cost:float, resolution:float):
        self.lb_cost_to_go = min(self.lb_cost_to_go, pot_parent.inad.lb_cost_to_go + edge_cost)
        if np.isinf(self.lb_cost_to_go):
            pass
        effort_to_parent = edge_cost/resolution
        new_effort = pot_parent.inad.effort + effort_to_parent
        if new_effort < self.effort:
            self.effort = new_effort
            self.effort_to_parent = effort_to_parent

    def reset_goal_nodes(self):
       self.lb_cost_to_go = 0.0
       self.effort = 0.0
       self.effort_to_parent = 0.0
    
    def reset(self):
        self.lb_cost_to_go = np.inf
        self.effort = np.inf
        self.effort_to_parent = np.inf

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

    def update_forward_queue_keys(self, type:str, node_ids:Optional[Set[BaseNode]] = None):
        self.effort_estimate_queue.update(node_ids, type)
        self.cost_estimate_queue.update(node_ids, type)
        self.cost_bound_queue.update(node_ids, type)

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
        self.inad.effort_to_parent = self.rev.cost_to_parent/resolution
        self.inad.effort = self.rev.parent.inad.effort +  self.inad.effort_to_parent

    def set_to_goal_node(self):
        self.lb_cost_to_go = 0.0
        self.inad.reset_goal_nodes()
    
class ReverseQueue(DictIndexHeap[Tuple[Any]]):

    def key(self, item: Tuple[Any]) -> float:
        # item[0]: edge_cost from n0 to n1
        # item[1]: edge (n0, n1)
        first_entry = item[1][0].lb_cost_to_go + item[0] + item[1][1].lb_cost_to_come
        second_entry = item[1][0].inad.effort +  item[1][1].inad.effort_to_parent + item[1][1].lb_effort_to_come
        return (first_entry, second_entry)
           
class EffortEstimateQueue(ForwardQueue):
    #remaining effort through an edge
    def key(self, item: Tuple[Any]) -> float:
        # item[0]: cost from n0 to n1
        # item[1]: edge (n0, n1)
        return (item[1][1].inad.effort_to_parent + item[1][1].inad.effort)
    
class CostBoundQueue(ForwardQueue):
    def key(self, item: Tuple[Any]) -> float:
        # item[0]: edge_cost from n0 to n1
        # item[1]: edge (n0, n1)
        return (item[1][0].cost + item[0] + item[1][1].lb_cost_to_go)

class CostEstimateQueue(ForwardQueue):
    def key(self, item: Tuple[Any]) -> float:
        # item[0]: edge_cost from n0 to n1
        # item[1]: edge (n0, n1)
        return (item[1][0].cost + item[0] + item[1][1].inad.lb_cost_to_go)

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

        self.start_transition_arrays = {}
        self.end_transition_arrays = {}
        self.remove_nodes = False
        self.inconsistent_target_nodes = set()
        self.sparse_number_of_points = 2
        self.epsilon = np.inf
        self.forward_closed_set = None
        self.reverse_closed_set = None

    def _create_operation(self) -> BaseOperation:
        return Operation()
           
    def continue_reverse_search(self) -> bool:
        if len(self.g.reverse_queue) == 0 or len(self.g.cost_bound_queue) == 0:
            return False
        forward_key, _ = self.g.cost_bound_queue.peek_first_element()
        reverse_key, _ = self.g.reverse_queue.peek_first_element()
        if forward_key <= reverse_key[0]:
            return False
        not_closed_nodes = self.g.cost_bound_queue.target_nodes - self.reverse_closed_set
        if len(not_closed_nodes) == 0:
            return False
        return True
             
    def expand_node_reverse(self, nodes: List[Node]) -> None:
        for node in nodes:
            self.reverse_closed_set.add(node.id)
            self.updated_target_nodes.add(node.id)
            if node.id == self.g.root.id:
                return   
            neighbors = self.g.get_neighbors(node, space_extent=self.approximate_space_extent)
            if neighbors.size == 0:
                return
            
            edge_costs = self.g.tot_neighbors_batch_cost_cache[node.id]
            for id, edge_cost in zip(neighbors, edge_costs):
                n = self.g.nodes[id]
                if n.id in self.reverse_closed_set:
                    continue
                assert (n.forward.parent == node) == (n.id in node.forward.children), (
                        f"Parent and children don't coincide (reverse): parent: {node.id}, child: {n.id}"
                        )
                if n in self.g.goal_nodes:
                    continue
                assert(n.id not in node.blacklist), (
                "neighbors are wrong")
                if node.rev.parent is not None and node.rev.parent.id == n.id:
                    continue
                if n.is_transition and not n.is_reverse_transition:
                    continue
                edge = (node, n)
                self.g.reverse_queue.heappush((edge_cost, edge))
                
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

        self.sparse_number_of_points = 2
        self.update_inflation_factor()
        self.sample_manifold()
        self.g.compute_transition_lb_cost_to_come()
        self.g.compute_node_lb_cost_to_come()
        self.compute_transition_lb_effort_to_come(self.env)
        self.compute_node_lb_effort_to_come(self.env)

        self.initialze_forward_search()
        self.initialize_reverse_search()
        self.dynamic_reverse_search_update = False

    def initialze_forward_search(self):
        self.g.effort_estimate_queue = EffortEstimateQueue()
        self.g.cost_bound_queue = CostBoundQueue()
        self.g.cost_estimate_queue = CostEstimateQueue()
        self.forward_closed_set = set()
        self.expand_node_forward(self.g.root)

    def initialize_reverse_search(self, reset:bool = True):
        if self.reverse_closed_set is not None and self.with_tree_visualization:
            self.save_tree_data(self.reverse_closed_set, 'reverse')
        self.forward_closed_set = set()
        self.reverse_closed_set = set()
        self.g.reverse_queue = ReverseQueue()
        if reset:
            self.g.reset_reverse_tree()
        self.g.reset_all_goal_nodes_lb_costs_to_go()
        self.updated_target_nodes = set()
        self.expand_node_reverse(self.g.goal_nodes)
        self.g.update_forward_queue_keys('target', self.updated_target_nodes)
        # print("Restart reverse search ...")
        # self.reverse_search()
        # print("... finished")

    def get_best_forward_edge(self):
        _, (edge_cost_ee,(ee0, ee1)) = self.g.effort_estimate_queue.peek_first_element()
        key_ce, (edge_cost_ce,(ce0, ce1)) = self.g.cost_estimate_queue.peek_first_element()
        key_cb, (edge_cost_cb,(cb0, cb1)) = self.g.cost_bound_queue.peek_first_element()

        item = (edge_cost_ee, (ee0, ee1))
        key_ee = self.g.cost_bound_queue.key(item)
        if key_ee < self.epsilon*key_cb:
            item = (edge_cost_ee, (ee0, ee1))
        elif key_ce < key_cb:
            item = (edge_cost_ce, (ce0, ce1))
        else:
            item = (edge_cost_cb, (cb0, cb1))

        self.g.effort_estimate_queue.remove(item)
        self.g.cost_estimate_queue.remove(item)
        self.g.cost_bound_queue.remove(item)
        assert(key_ce == key_cb), (
            "ghjklö"
        )

        return item
        
    def update_inflation_factor(self):
        if self.current_best_cost is None:
            self.epsilon = np.inf
        else:
            self.epsilon = 1

    @lru_cache(maxsize=None)  
    def get_sparse_cd_res(self, edge_cost, N) -> float:
        r = edge_cost/N
        if r < self.env.collision_resolution:
            return self.env.collision_resolution
        return r

    def reverse_search(self):
        self.updated_target_nodes = set()
        while self.continue_reverse_search():
            edge_cost, edge = self.g.reverse_queue.heappop()
            n0, n1 = edge
            if n1.id in self.reverse_closed_set:
                continue
            is_transition = False
            if n1.is_transition and n1.is_reverse_transition:
                is_transition = True
            if n0.id not in n1.whitelist:
                sparsely_collision_free = False
                if n0.id not in n1.blacklist:
                    sparse_resolution = self.get_sparse_cd_res(edge_cost, self.sparse_number_of_points)
                    sparsely_collision_free = self.env.is_edge_collision_free(
                                n0.state.q,
                                n1.state.q,
                                n0.state.mode,
                                resolution=sparse_resolution
                            )
                if not sparsely_collision_free:
                    self.remove_forward_queue(edge_cost, n0, n1)
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
                    self.reverse_closed_set.add(n1.id)
                    self.expand_node_reverse([n1.transition])
                    continue
                self.expand_node_reverse([n1])

        self.g.update_forward_queue_keys('target', self.updated_target_nodes)
        if self.reverse_closed_set is not None and self.with_tree_visualization:
            self.save_tree_data(self.reverse_closed_set, 'reverse')
                
    def compute_transition_lb_effort_to_come(self, env:BaseProblem):
        # run a reverse search on the transition nodes without any collision checking
        efforts = {}
        transition_nodes = {}
        processed = 0

        closed_set = set()

        queue = []
        heapq.heappush(queue, (0, self.g.root))
        efforts[self.g.root.id] = 0
        self.g.root.lb_effort_to_come = 0

        while len(queue) > 0:
            _, node = heapq.heappop(queue)
            mode = node.state.mode
            if node.id in closed_set:
                continue
            closed_set.add(node.id)
            if mode not in self.g.transition_node_ids:
                continue

            if mode not in transition_nodes:
                if mode.task_ids == self.g.goal_nodes[0].state.mode.task_ids:
                    transition_nodes[mode] = self.g.goal_nodes
                else:
                    transition_nodes[mode] = [self.g.nodes[id].transition for id in self.g.transition_node_ids[mode]]

            if len(transition_nodes[mode]) == 0:
                continue
            self.g.update_cache(mode)
            
            if mode not in self.g.transition_node_array_cache:
                continue           

            edge_costs = self.env.batch_config_cost(
                node.state.q,
                self.g.transition_node_array_cache[mode],
            )
            # add neighbors to open_queue
            parent_effort = efforts[node.id]
            for edge_cost, n in zip(edge_costs, transition_nodes[mode]):
                effort = parent_effort + edge_cost/self.env.collision_resolution
                if n.state.mode not in efforts:
                    efforts[n.id] = effort
                if n.lb_effort_to_come > effort:
                    n.lb_effort_to_come = effort
                    if n.transition is not None:
                        n.transition.lb_effort_to_come = effort
                    processed += 1
                    heapq.heappush(queue, (effort, n))
        print(processed)
        
    def compute_node_lb_effort_to_come(self, env:BaseProblem):
        processed = 0
        reverse_transition_node_lb_cache = {}
        for mode in self.g.node_ids:
            for id in self.g.node_ids[mode]:
                n = self.g.nodes[id]
                mode = n.state.mode
                if mode not in self.g.reverse_transition_node_array_cache:
                    continue

                if mode not in reverse_transition_node_lb_cache:
                    reverse_transition_node_lb_cache[mode] = np.array(
                        [
                            self.g.nodes[id].lb_cost_to_come
                            for id in self.g.reverse_transition_node_ids[mode]
                        ],
                        dtype=np.float64,
                    )

                costs_to_transitions = self.env.batch_config_cost(
                    n.state.q,
                    self.g.reverse_transition_node_array_cache[mode],
                )

                min_effort = np.min(
                    reverse_transition_node_lb_cache[mode] 
                    + costs_to_transitions/self.env.collision_resolution
                )

                n.lb_effort_to_come = min_effort
                processed +=1
        print(processed)

    def update_forward_queue(self, edge_cost, edge):
        self.g.effort_estimate_queue.heappush((edge_cost, edge))
        self.g.cost_bound_queue.heappush((edge_cost, edge))
        self.g.cost_estimate_queue.heappush((edge_cost, edge)) 

    def PlannerInitialization(self) -> None:
        q0 = self.env.get_start_pos()
        m0 = self.env.get_start_mode()
        self.reached_modes.append(m0)
        #initilaize graph
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

    def manage_edge_in_collision(self, edge, already_detected:bool = False):
        if not self.dynamic_reverse_search_update:
            return
        if edge[0].rev.parent == edge[1] or edge[1].rev.parent == edge[0]:
            if not already_detected: #menaing collision was detected with sparse collision checking but was already in forward queue
                self.sparse_number_of_points *=1.2
            self.g.initialize_cache()
            self.initialize_reverse_search()

    def remove_forward_queue(self, edge_cost, n0, n1):
        self.g.cost_bound_queue.remove((edge_cost, (n1, n0))) 
        self.g.effort_estimate_queue.remove((edge_cost, (n1, n0)))
        self.g.cost_estimate_queue.remove((edge_cost, (n1, n0)))
        self.g.cost_bound_queue.remove((edge_cost, (n0, n1))) 
        self.g.effort_estimate_queue.remove((edge_cost, (n0, n1)))
        self.g.cost_estimate_queue.remove((edge_cost, (n0, n1)))

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
                edge_cost, (n0, n1) = self.get_best_forward_edge()
                if n1.id in self.forward_closed_set:
                    continue
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
                    assert not [self.g.cost_bound_queue.key(item) for item in self.g.cost_bound_queue.current_entries if not np.isinf(self.g.cost_bound_queue.key(item))], (
                    "Forward queue contains non-infinite keys!")
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
                        else:
                            pass
                        if not collision_free:
                            self.remove_forward_queue(edge_cost, n0, n1)
                            assert (n0.id, n1.id) not in self.collision_cache, (
                                "kl,dö.fäghjk"
                            )
                            self.collision_cache.add((n0.id, n1.id))
                            self.collision_cache.add((n1.id, n0.id))
                            self.manage_edge_in_collision((n0, n1))
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
                    with_shortcutting = False
                    if n1 in self.g.goal_nodes:
                        with_shortcutting = True
                    if self.dynamic_reverse_search_update or n1 in self.g.goal_nodes:
                        path = self.generate_path()
                        if len(path) > 0:
                            self.process_valid_path(path, with_shortcutting= with_shortcutting)
                            self.update_inflation_factor()
                            
            else:
                if np.isinf(n1.lb_cost_to_go):
                    assert (len(self.g.cost_bound_queue.target_nodes - self.consistent_nodes) == len(self.g.cost_bound_queue.target_nodes)), (
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
