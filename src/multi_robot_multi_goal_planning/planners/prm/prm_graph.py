import heapq
import math
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
    batch_config_dist,
)
from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseProblem,
    Mode,
    State,
)

from .prm_queues import DictIndexHeap

class Node:
    __slots__ = [
        "state",
        "lb_cost_to_goal",
        "lb_cost_from_start",
        "is_transition",
        "neighbors",
        "whitelist",
        "blacklist",
        "id",
    ]

    # Class attribute
    id_counter: ClassVar[int] = 0

    # Instance attributes
    state: State
    lb_cost_to_goal: Optional[float]
    lb_cost_from_start: Optional[float]
    is_transition: bool
    neighbors: List["Node"]
    whitelist: Set[int]
    blacklist: Set[int]
    id: int

    def __init__(self, state: State, is_transition: bool = False) -> None:
        self.state = state
        self.lb_cost_to_goal = np.inf
        self.lb_cost_from_start = np.inf

        self.is_transition = is_transition

        self.neighbors = []

        self.whitelist = set()
        self.blacklist = set()

        self.id = Node.id_counter
        Node.id_counter += 1

    def __lt__(self, other: "Node") -> bool:
        return self.id < other.id

    def __hash__(self) -> int:
        return self.id


class MultimodalGraph:
    """ "
    The graph that we will construct and refine and search on for the prm
    planner. Maintains all the search/construction and the functions necessary for it.

    Consists effectively of a list of nodes that is split into transition nodes
    and general mode-nodes for efficiency reasons.
    """

    root: Node
    nodes: Dict[Mode, List[Node]]

    node_array_cache: Dict[Mode, NDArray]

    transition_node_array_cache: Dict[Mode, NDArray]
    reverse_transition_node_array_cache: Dict[Mode, NDArray]

    # batch_dist_fun

    def __init__(self, start: State, batch_dist_fun, use_k_nearest: bool = True):
        self.root = Node(start)
        self.root.lb_cost_from_start = 0
        # self.nodes = [self.root]

        self.batch_dist_fun = batch_dist_fun

        self.use_k_nearest = use_k_nearest

        self.nodes = {}
        self.nodes[self.root.state.mode] = [self.root]

        self.transition_nodes: Dict[Mode, List[Node]] = {}  # contains the transitions at the end of the mode
        self.reverse_transition_nodes = {}
        self.reverse_transition_nodes[self.root.state.mode] = [self.root]

        self.goal_nodes = []

        self.mode_to_goal_lb_cost = {}

        self.node_array_cache = {}

        self.transition_node_array_cache = {}
        self.reverse_transition_node_array_cache = {}

        self.transition_node_lb_cache = {}
        self.rev_transition_node_lb_cache = {}

    def get_num_samples(self) -> int:
        num_samples = 0
        for k, v in self.nodes.items():
            num_samples += len(v)

        num_transition_samples = 0
        for k, v in self.transition_nodes.items():
            num_transition_samples += len(v)

        return num_samples + num_transition_samples

    def get_num_samples_in_mode(self, mode: Mode) -> int:
        num_samples = 0
        if mode in self.nodes:
            num_samples += len(self.nodes[mode])
        if mode in self.transition_nodes:
            num_samples += len(self.transition_nodes[mode])
        return num_samples

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def compute_lower_bound_to_goal(self, batch_cost, best_found_cost):
        """
        Computes the lower bound on the cost to reach to goal from any configuration by
        running a reverse search on the transition nodes without any collision checking.
        """
        costs = {}
        closed_set = set()

        for mode in self.nodes.keys():
            for i in range(len(self.nodes[mode])):
                self.nodes[mode][i].lb_cost_to_goal = np.inf

        if best_found_cost is None:
            best_found_cost = np.inf

        queue = []
        for g in self.goal_nodes:
            heapq.heappush(queue, (0, g))

            costs[g.id] = 0
            # parents[hash(g)] = None

        while queue:
            # node = queue.pop(0)
            _, node = heapq.heappop(queue)
            # print(node)

            # error happens at start node
            if node.state.mode == self.root.state.mode:
                continue

            if node.id in closed_set:
                continue

            closed_set.add(node.id)

            # neighbors = []

            # this is the same code as below, but slightly more legible
            # for n in self.reverse_transition_nodes[node.state.mode]:
            #     for q in n.neighbors:
            #         neighbors.append(q)
            neighbors: List[Node] = [
                q
                for n in self.reverse_transition_nodes[node.state.mode]
                for q in n.neighbors
            ]

            # neighbors = [
            #     neighbor for n in self.reverse_transition_nodes[node.state.mode] for neighbor in n.neighbors
            # ]

            if not neighbors:
                continue

            if node.state.mode not in self.reverse_transition_node_array_cache:
                self.reverse_transition_node_array_cache[node.state.mode] = np.array(
                    [n.state.q.state() for n in neighbors], dtype=np.float64
                )

            # add neighbors to open_queue
            edge_costs = batch_cost(
                node.state.q,
                self.reverse_transition_node_array_cache[node.state.mode],
            )
            parent_cost = costs[node.id]
            for edge_cost, n in zip(edge_costs, neighbors):
                cost = parent_cost + edge_cost

                if cost > best_found_cost:
                    continue

                id = n.id
                if id not in costs or cost < costs[id]:
                    costs[id] = cost
                    n.lb_cost_to_goal = cost

                    heapq.heappush(queue, (cost, n))

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def compute_lower_bound_from_start(self, batch_cost):
        """
        compute the lower bound to reach a configuration from the start.
        run a reverse search on the transition nodes without any collision checking
        """
        costs = {}

        closed_set = set()

        queue = []
        heapq.heappush(queue, (0, self.root))
        costs[self.root.id] = 0

        while len(queue) > 0:
            _, node = heapq.heappop(queue)

            if node.id in closed_set:
                continue

            if node.state.mode.task_ids == self.goal_nodes[0].state.mode.task_ids:
                continue

            if node.state.mode not in self.transition_nodes:
                continue

            neighbors = [n.neighbors[0] for n in self.transition_nodes[node.state.mode]]

            if not neighbors:
                continue

            if node.state.mode not in self.transition_node_array_cache:
                self.transition_node_array_cache[node.state.mode] = np.array(
                    [n.state.q.state() for n in neighbors], dtype=np.float64
                )

            closed_set.add(node.id)

            # add neighbors to open_queue
            edge_costs = batch_cost(
                node.state.q,
                self.transition_node_array_cache[node.state.mode],
            )

            parent_cost = costs[node.id]
            for edge_cost, n in zip(edge_costs, neighbors):
                cost = parent_cost + edge_cost
                id = n.id
                if id not in costs or cost < costs[id]:
                    costs[id] = cost
                    n.lb_cost_from_start = cost

                    heapq.heappush(queue, (cost, n))

    def add_node(self, new_node: Node) -> None:
        self.node_array_cache = {}

        key = new_node.state.mode
        if key not in self.nodes:
            self.nodes[key] = []
        node_list = self.nodes[key]
        node_list.append(new_node)

    def add_states(self, states: List[State]):
        for s in states:
            self.add_node(Node(s))

    def add_nodes(self, nodes: List[Node]):
        for n in nodes:
            self.add_node(n)

    def add_transition_nodes(
        self, transitions: List[Tuple[Configuration, Mode, List[Mode] | None]]
    ):
        """
        Adds transition nodes.

        A transition node consists of a configuration, the mode it is in, and the modes it is a transition to.
        The configuration is added as node to the current mode, and to all the following modes.

        Also adds/updates the caches and ensures that transitions nodes are not added
        multiple times.
        """

        self.transition_node_array_cache = {}
        self.reverse_transition_node_array_cache = {}

        self.transition_node_lb_cache = {}
        self.rev_transition_node_lb_cache = {}

        for q, this_mode, next_modes in transitions:
            node_this_mode = Node(State(q, this_mode), True)

            if (
                this_mode in self.transition_nodes
                and len(self.transition_nodes[this_mode]) > 0
            ):
                # print("A", this_mode, len(self.transition_nodes[this_mode]))
                dists = self.batch_dist_fun(
                    node_this_mode.state.q,
                    [n.state.q for n in self.transition_nodes[this_mode]],
                )
                # print("B")

                # if the transition node is very close to another one
                # we treat it as already added and do not add it again
                if min(dists) < 1e-6:
                    continue

            if next_modes is None:
                # the current mode is a terminal node. deal with it accordingly
                # print("attempting goal node")
                is_in_goal_nodes_already = False
                for g in self.goal_nodes:
                    if (
                        np.linalg.norm(
                            g.state.q.state() - node_this_mode.state.q.state()
                        )
                        < 1e-3
                    ):
                        is_in_goal_nodes_already = True
                        break

                if not is_in_goal_nodes_already:
                    self.goal_nodes.append(node_this_mode)
                    node_this_mode.lb_cost_to_goal = 0

                    if this_mode in self.transition_nodes:
                        self.transition_nodes[this_mode].append(node_this_mode)
                    else:
                        self.transition_nodes[this_mode] = [node_this_mode]
            else:
                if not isinstance(next_modes, list):
                    next_modes = [next_modes]

                # print(next_modes)
                if len(next_modes) == 0:
                    continue

                next_nodes = []
                for next_mode in next_modes:
                    node_next_mode = Node(State(q, next_mode), True)
                    next_nodes.append(node_next_mode)

                node_this_mode.neighbors = next_nodes

                for node_next_mode, next_mode in zip(next_nodes, next_modes):
                    node_next_mode.neighbors = [node_this_mode]

                    assert this_mode.task_ids != next_mode.task_ids, "ghj"

                # if this_mode in self.transition_nodes:
                # print(len(self.transition_nodes[this_mode]))

                if this_mode in self.transition_nodes:
                    self.transition_nodes[this_mode].append(node_this_mode)
                else:
                    self.transition_nodes[this_mode] = [node_this_mode]

                # add the same things to the rev transition nodes
                for next_mode, next_node in zip(next_modes, next_nodes):
                    if next_mode in self.reverse_transition_nodes:
                        self.reverse_transition_nodes[next_mode].append(next_node)
                    else:
                        self.reverse_transition_nodes[next_mode] = [next_node]

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def get_neighbors(
        self, node: Node, space_extent: Optional[float] = None
    ) -> Tuple[List[Node], NDArray | None]:
        """
        Computes all neighbours to the current mode, either k_nearest, or according to a radius.
        """
        key = node.state.mode
        if key in self.nodes:
            node_list = self.nodes[key]

            if key not in self.node_array_cache:
                self.node_array_cache[key] = np.array(
                    [n.state.q.q for n in node_list], dtype=np.float64
                )

            if len(self.node_array_cache[key]) == 0:
                return [], None

            dists = self.batch_dist_fun(
                node.state.q, self.node_array_cache[key]
            )  # this, and the list copm below are the slowest parts

        if key in self.transition_nodes:
            transition_node_list = self.transition_nodes[key]

            if key not in self.transition_node_array_cache:
                self.transition_node_array_cache[key] = np.array(
                    [n.state.q.q for n in transition_node_list], dtype=np.float64
                )

            if len(self.transition_node_array_cache[key]) == 0:
                return [], None

            transition_dists = self.batch_dist_fun(
                node.state.q, self.transition_node_array_cache[key]
            )

        # plt.plot(dists)
        # plt.show()

        dim = len(node.state.q.state())

        best_nodes_arr = np.zeros((0, dim))
        best_transitions_arr = np.zeros((0, dim))

        if self.use_k_nearest:
            best_nodes = []
            if key in self.nodes:
                assert node_list is not None
                assert dists is not None

                k_star = int(np.e * (1 + 1 / dim) * np.log(len(node_list))) + 1
                # # print(k_star)
                # k = k_star
                k_normal_nodes = k_star

                k_clip = min(k_normal_nodes, len(node_list))
                topk = np.argpartition(dists, k_clip - 1)[:k_clip]
                topk = topk[np.argsort(dists[topk])]

                best_nodes = [node_list[i] for i in topk]
                best_nodes_arr = self.node_array_cache[key][topk, :]

            best_transition_nodes = []
            if key in self.transition_nodes:
                k_star = (
                    int(np.e * (1 + 1 / dim) * np.log(len(transition_node_list))) + 1
                )
                # # print(k_star)
                # k_transition_nodes = k
                k_transition_nodes = k_star

                transition_k_clip = min(k_transition_nodes, len(transition_node_list))
                transition_topk = np.argpartition(
                    transition_dists, transition_k_clip - 1
                )[:transition_k_clip]
                transition_topk = transition_topk[
                    np.argsort(transition_dists[transition_topk])
                ]

                best_transition_nodes = [
                    transition_node_list[i] for i in transition_topk
                ]
                best_transitions_arr = self.transition_node_array_cache[key][
                    transition_topk
                ]

            best_nodes = best_nodes + best_transition_nodes

        else:
            unit_n_ball_measure = ((np.pi**0.5) ** dim) / math.gamma(dim / 2 + 1)
            informed_measure = 1
            if space_extent is not None:
                informed_measure = space_extent
                # informed_measure = space_extent / 2

            best_nodes = []
            if key in self.nodes:
                # r_star = 2 * 1 / (len(node_list)**(1/dim))
                r_star = (
                    1.001
                    * 2
                    * (
                        informed_measure
                        / unit_n_ball_measure
                        * (np.log(len(node_list)) / len(node_list))
                        * (1 + 1 / dim)
                    )
                    ** (1 / dim)
                )

                best_nodes = [node_list[i] for i in np.where(dists < r_star)[0]]
                best_nodes_arr = self.node_array_cache[key][
                    np.where(dists < r_star)[0], :
                ]

                # print("fraction of nodes in mode", len(best_nodes)/len(dists))
                # print(r_star)
                # print(len(best_nodes))

            best_transition_nodes = []
            if key in self.transition_nodes:
                # r_star = 2 * 1 / (len(node_list)**(1/dim))

                r_star = (
                    1.001
                    * 2
                    * (
                        (1 + 1 / dim)
                        * informed_measure
                        / unit_n_ball_measure
                        * (
                            np.log(len(transition_node_list))
                            / len(transition_node_list)
                        )
                    )
                    ** (1 / dim)
                )
                # print(node.state.mode, r_star)

                if len(transition_node_list) == 1:
                    r_star = 1e6

                best_transition_nodes = [
                    transition_node_list[i]
                    for i in np.where(transition_dists < r_star)[0]
                ]
                best_transitions_arr = self.transition_node_array_cache[key][
                    np.where(transition_dists < r_star)[0]
                ]

            best_nodes = best_nodes + best_transition_nodes

        arr = np.vstack([best_nodes_arr, best_transitions_arr], dtype=np.float64)

        if node.is_transition:
            tmp = np.vstack([n.state.q.state() for n in node.neighbors])
            arr = np.vstack([arr, tmp])
            return best_nodes + node.neighbors, arr

        return best_nodes, arr

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def search(
        self,
        start_node: Node,
        goal_nodes: List[Node],
        env: BaseProblem,
        best_cost: Optional[float] = None,
        resolution: float = 0.1,
        approximate_space_extent: float | None = None,
    ) -> List[Node]:
        """
        Entry point for the search.
        """
        if approximate_space_extent is None:
            approximate_space_extent = float(np.prod(np.diff(env.limits, axis=0)))

        goal = None
        h_cache = {}

        if best_cost is None:
            best_cost = np.inf

        def h(node):
            # return 0
            if node.id in h_cache:
                return h_cache[node.id]

            if node.state.mode not in self.transition_nodes:
                return np.inf

            if node.state.mode not in self.transition_node_array_cache:
                self.transition_node_array_cache[node.state.mode] = np.array(
                    [o.state.q.q for o in self.transition_nodes[node.state.mode]],
                    dtype=np.float64,
                )

            if node.state.mode not in self.transition_node_lb_cache:
                self.transition_node_lb_cache[node.state.mode] = np.array(
                    [o.lb_cost_to_goal for o in self.transition_nodes[node.state.mode]],
                    dtype=np.float64,
                )

            if len(self.transition_node_array_cache[node.state.mode]) == 0:
                return np.inf

            costs_to_transitions = env.batch_config_cost(
                node.state.q,
                self.transition_node_array_cache[node.state.mode],
            )

            min_cost = np.min(
                self.transition_node_lb_cache[node.state.mode] + costs_to_transitions
            )

            h_cache[node.id] = min_cost
            return min_cost

        def d(n0, n1):
            # return 1.0
            cost = env.config_cost(n0.state.q, n1.state.q)
            return cost

        # reached_modes = []

        parents = {start_node: None}
        gs = {start_node.id: 0}  # best cost to get to a node

        start_neighbors, _ = self.get_neighbors(
            start_node, space_extent=approximate_space_extent
        )

        # populate open_queue and fs
        start_edges = [(start_node, n) for n in start_neighbors]

        # queue = HeapQueue()
        # queue = BucketHeapQueue()
        # queue = BucketIndexHeap()
        # queue = DiscreteBucketIndexHeap()
        # queue = IndexHeap()
        queue = DictIndexHeap()
        # queue = SortedQueue()
        # queue = EfficientEdgeQueue()

        # fs = {}  # total cost of a node (f = g + h)
        for e in start_edges:
            if e[0] != e[1]:
                # open_queue.append(e)
                edge_cost = d(e[0], e[1])
                cost = gs[start_node.id] + edge_cost + h(e[1])
                # fs[(e[0].id, e[1].id)] = cost
                # heapq.heappush(open_queue, (cost, edge_cost, e))
                queue.heappush((cost, edge_cost, e))
                # open_queue.append((cost, edge_cost, e))

        wasted_pops = 0
        processed_edges = 0

        queue_pop = queue.heappop
        queue_push = queue.heappush

        num_iter = 0
        while queue:
            num_iter += 1

            if num_iter % 100000 == 0:
                print(len(queue))

            f_pred, edge_cost, (n0, n1) = queue_pop()

            if n1.id in gs:
                wasted_pops += 1
                continue

            # check edge now. if it is not valid, blacklist it, and continue with the next edge
            collision_free = False

            if n0.id in n1.whitelist:
                collision_free = True
            else:
                if n1.id in n0.blacklist:
                    continue

                q0 = n0.state.q
                q1 = n1.state.q
                collision_free = env.is_edge_collision_free(
                    q0, q1, n0.state.mode, resolution
                )

                if not collision_free:
                    n1.blacklist.add(n0.id)
                    n0.blacklist.add(n1.id)
                    continue
                else:
                    n1.whitelist.add(n0.id)
                    n0.whitelist.add(n1.id)

            processed_edges += 1

            g_tentative = gs[n0.id] + edge_cost
            gs[n1.id] = g_tentative
            parents[n1] = n0

            if n1 in goal_nodes:
                goal = n1
                break

            # get_neighbors
            neighbors, tmp = self.get_neighbors(
                n1, space_extent=approximate_space_extent
            )

            if not neighbors:
                continue

            # add neighbors to open_queue
            edge_costs = env.batch_config_cost(n1.state.q, tmp)
            for n, edge_cost in zip(neighbors, edge_costs):
                # if n == n0:
                #     continue

                if n.id in n1.blacklist:
                    continue

                # edge_cost = edge_costs[i]
                # g_new = g_tentative + edge_cost

                # if n.id in gs:
                #     print(n.id)

                if n.id not in gs:
                    g_new = g_tentative + edge_cost
                    f_node = g_new + h(n)

                    if f_node > best_cost:
                        continue

                    queue_push((f_node, edge_cost, (n1, n)))

        path = []

        # if we found a path to a goal, we reconstruct the path
        if goal is not None:
            path.append(goal)

            n = goal

            while n is not None and parents[n] is not None:
                path.append(parents[n])
                n = parents[n]

            path.append(n)
            path = path[::-1]

        print("Wasted pops", wasted_pops)
        print("Processed edges", processed_edges)

        return path