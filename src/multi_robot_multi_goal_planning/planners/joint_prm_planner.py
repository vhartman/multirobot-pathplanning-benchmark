import numpy as np
import random
from matplotlib import pyplot as plt

from typing import List, Dict, Tuple, Optional, Set, ClassVar, Any
from numpy.typing import NDArray

import heapq
from sortedcontainers import SortedList
from collections import defaultdict
import time
import math

from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
    Mode,
)
from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
    NpConfiguration,
    config_dist,
    batch_config_dist,
)
from multi_robot_multi_goal_planning.problems.util import path_cost, interpolate_path

from multi_robot_multi_goal_planning.planners import shortcutting

from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
)
from multi_robot_multi_goal_planning.planners.sampling_phs import (
    sample_phs_with_given_matrices, compute_PHS_matrices,
)

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
    neighbors: List[int]
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


class HeapQueue:
    __slots__ = ["queue"]

    # Class attribute type hints
    queue: List[Any]

    def __init__(self) -> None:
        self.queue = []
        heapq.heapify(self.queue)

    def __len__(self) -> int:
        return len(self.queue)

    def heappush(self, item: Any) -> None:
        heapq.heappush(self.queue, item)

    def heappop(self) -> Any:
        # if len(self.queue) == 0:
        #     raise IndexError("pop from an empty heap")
        return heapq.heappop(self.queue)

    def remove(self, node: Any) -> None:
        """
        Removes a node from the heap.
        Note: This is a placeholder implementation.
        """
        # (cost, edge_cost, e)
        # To implement removal, you would need to:
        # 1. Find the node in the heap.
        # 2. Mark it as removed (e.g., using a flag or a separate set).
        # 3. Re-heapify the queue if necessary.
        pass


class SortedQueue:
    """
    Edge queue implementation using SortedList for efficient removal operations.
    """

    def __init__(self):
        # SortedList maintains elements in sorted order
        # Elements are tuples: (cost, edge_cost, (node1, node2))
        self.queue = SortedList()

    def heappush(self, item):
        """Add new edge to queue"""
        self.queue.add(item)

    def heappop(self):
        """Remove and return lowest cost edge"""
        if self.queue:
            return self.queue.pop(0)
        return None

    # def remove_by_node(self, node):
    #     """Remove all edges where node2 == node"""
    #     # Create list of indices to remove (in reverse order)
    #     to_remove = [
    #         i for i, (_, _, (_, node2)) in enumerate(self.queue)
    #         if node2 == node
    #     ]

    #     # Remove from highest index to lowest to maintain valid indices
    #     for idx in reversed(to_remove):
    #         del self.queue[idx]

    def remove_by_node(self, node):
        """Remove all edges where node2 == node"""
        i = 0
        while i < len(self.queue):
            _, _, (_, node2) = self.queue[i]
            if node2 == node:
                del self.queue[
                    i
                ]  # Deleting an item shifts elements left, so don't increment i
            else:
                i += 1  # Only increment if an element was not removed

    def __len__(self):
        return len(self.queue)


class EfficientEdgeQueue:
    """
    Edge queue using a min-heap for efficient pops and a dictionary for fast removals.
    """

    def __init__(self):
        # Min-heap of (cost, edge_cost, (node1, node2))
        self.heap = []
        # Dictionary mapping node2 to a set of edges for quick removal
        self.edges_by_node = defaultdict(set)

    def heappush(self, item):
        """Add a new edge to the queue."""
        cost, edge_cost, nodes = item

        heapq.heappush(self.heap, item)
        self.edges_by_node[nodes[1]].add(item)

    def heappop(self):
        """Remove and return the lowest cost edge."""
        while self.heap:
            item = heapq.heappop(self.heap)
            nodes = item[2]
            if item in self.edges_by_node[nodes[1]]:  # Ensure the edge is still valid
                self.edges_by_node[nodes[1]].remove(item)
                return item

        return None

    def remove_by_node(self, node):
        """Remove all edges where node2 == node."""
        if node in self.edges_by_node:
            for item in self.edges_by_node[node]:
                # Lazy deletion: mark edge as removed by excluding it from the dict
                self.heap.remove(item)  # O(n), but happens rarely
            heapq.heapify(self.heap)  # Restore heap property, O(n)
            del self.edges_by_node[node]  # Remove the entry

    def __len__(self):
        return len(self.heap)


class BucketHeapQueue:
    def __init__(self):
        self.queues = {}
        self.priority_lookup = []

        self.len = 0

    def __len__(self):
        return self.len

    def heappush(self, item):
        self.len += 1
        priority = int(item[0] * 10000)

        if priority not in self.queues:
            self.queues[priority] = []
            heapq.heappush(self.priority_lookup, priority)

        heapq.heappush(self.queues[priority], item)

    def heappop(self):
        self.len -= 1

        min_priority = self.priority_lookup[0]
        value = heapq.heappop(self.queues[min_priority])

        if not self.queues[min_priority]:
            del self.queues[min_priority]
            heapq.heappop(self.priority_lookup)

        return value


class IndexHeap:
    __slots__ = ["queue", "items"]

    # Class attribute type hints
    queue: List[Tuple[float, int]]  # (priority, index)
    items: List[Any]  # The actual items

    def __init__(self) -> None:
        self.queue = []
        self.items = []
        heapq.heapify(self.queue)

    def __len__(self) -> int:
        return len(self.queue)

    def heappush_list(self, items: List[Tuple[float, Any]]) -> None:
        for item in items:
            idx = len(self.items)
            self.items.append(item)
            self.queue.append((item[0], idx))

        heapq.heapify(self.queue)

    def heappush(self, item: Tuple[float, Any]) -> None:
        idx = len(self.items)
        self.items.append(item)
        heapq.heappush(self.queue, (item[0], idx))

    def heappop(self) -> Any:
        # if len(self.queue) == 0:
        #     raise IndexError("pop from an empty heap")

        _, idx = heapq.heappop(self.queue)
        return self.items[idx]


class DictIndexHeap:
    __slots__ = ["queue", "items"]

    queue: List[Tuple[float, int]]  # (priority, index)
    items: Dict[int, Any]  # Dictionary for storing active items

    idx = 0

    def __init__(self) -> None:
        self.queue = []
        self.items = {}
        heapq.heapify(self.queue)

    def __len__(self) -> int:
        return len(self.queue)

    def __bool__(self):
        return bool(self.queue)

    # def heappush_list(self, items: List[Tuple[float, Any]]) -> None:
    #     """Push a list of items into the heap."""
    #     for priority, value in items:
    #         idx = len(self.items)
    #         self.items[idx] = value  # Store only valid items
    #         self.queue.append((priority, idx))

    #     heapq.heapify(self.queue)

    def heappush(self, item: Tuple[float, Any]) -> None:
        """Push a single item into the heap."""
        # idx = len(self.items)
        self.items[DictIndexHeap.idx] = item  # Store only valid items
        heapq.heappush(self.queue, (item[0], DictIndexHeap.idx))
        DictIndexHeap.idx += 1

    def heappop(self) -> Any:
        """Pop the item with the smallest priority from the heap."""
        if not self.queue:
            raise IndexError("pop from an empty heap")

        _, idx = heapq.heappop(self.queue)
        value = self.items.pop(idx)  # Remove from dictionary
        return value


class BucketIndexHeap:
    __slots__ = ["granularity", "queues", "priority_lookup", "items", "len"]

    # Class attribute type hints
    granularity: int
    queues: Dict[int, List[Tuple[float, int]]]
    priority_lookup: List[int]
    items: List[Any]
    len: int

    def __init__(self, granularity: int = 100) -> None:
        self.granularity = granularity
        self.len = 0

        self.queues = {}
        self.priority_lookup = []
        self.items = []

    def __len__(self) -> int:
        return self.len

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def heappush(self, item: Tuple[float, Any]) -> None:
        self.len += 1
        priority: int = int(item[0] * self.granularity)

        idx: int = len(self.items)
        self.items.append(item)

        if priority not in self.queues:
            self.queues[priority] = []
            heapq.heappush(self.priority_lookup, priority)

        heapq.heappush(self.queues[priority], (item[0], idx))

    # def heappush_list(self, items: List[Tuple[float, Any]]) -> None:
    #     for item in items:
    #         self.heappush(item)

    def heappop(self) -> Any:
        # I do not want the possible performance penalty
        # if self.len == 0:
        #     raise IndexError("pop from an empty heap")

        self.len -= 1
        min_priority: int = self.priority_lookup[0]
        _, idx = heapq.heappop(self.queues[min_priority])

        if not self.queues[min_priority]:
            del self.queues[min_priority]
            heapq.heappop(self.priority_lookup)

        value: Any = self.items[idx]
        return value


# class DiscreteBucketIndexHeap:
#     def __init__(self, granularity=1000):
#         self.granularity = granularity

#         self.queues = {}
#         self.priority_lookup = []

#         self.items = []

#         self.len = 0

#     def __len__(self):
#         # num_elements = 0
#         # for k, v in self.queues.items():
#         #     num_elements += len(v)

#         # return num_elements
#         return self.len

#     # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
#     def heappush(self, item):
#         self.len += 1
#         priority = int(item[0] * self.granularity)

#         idx = len(self.items)
#         self.items.append(item)

#         if priority not in self.queues:
#             self.queues[priority] = []
#             heapq.heappush(self.priority_lookup, priority)

#         self.queues[priority].append((item[0], idx))

#     # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
#     def heappop(self):
#         self.len -= 1

#         min_priority = self.priority_lookup[0]
#         _, idx = self.queues[min_priority].pop()

#         if not self.queues[min_priority]:
#             del self.queues[min_priority]
#             heapq.heappop(self.priority_lookup)

#         value = self.items[idx]

#         return value


class DiscreteBucketIndexHeap:
    __slots__ = ["granularity", "queues", "priority_lookup", "items", "len"]

    # Class attribute type hints
    granularity: int
    queues: Dict[int, List[Tuple[float, int]]]
    priority_lookup: List[int]
    items: List[Any]
    len: int

    def __init__(self, granularity: int = 1000) -> None:
        self.granularity = granularity
        self.queues = {}
        self.priority_lookup = []
        self.items = []
        self.len = 0

    def __len__(self) -> int:
        return self.len

    def heappush(self, item: Tuple[float, Any]) -> None:
        self.len += 1
        priority: int = int(item[0] * self.granularity)

        idx: int = len(self.items)
        self.items.append(item)

        if priority not in self.queues:
            self.queues[priority] = []
            heapq.heappush(self.priority_lookup, priority)

        self.queues[priority].append((item[0], idx))

    def heappop(self) -> Any:
        if self.len == 0:
            raise IndexError("pop from an empty heap")

        self.len -= 1

        min_priority: int = self.priority_lookup[0]
        _, idx = self.queues[min_priority].pop()

        if not self.queues[min_priority]:
            del self.queues[min_priority]
            heapq.heappop(self.priority_lookup)

        value: Any = self.items[idx]
        return value


class Graph:
    root: State
    nodes: Dict

    # batch_dist_fun

    def __init__(self, start: State, batch_dist_fun, use_k_nearest: bool = True):
        self.root = Node(start)
        self.root.lb_cost_from_start = 0
        # self.nodes = [self.root]

        self.batch_dist_fun = batch_dist_fun

        self.use_k_nearest = use_k_nearest

        self.nodes = {}
        self.nodes[self.root.state.mode] = [self.root]

        self.transition_nodes = {}  # contains the transitions at the end of the mode
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

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def compute_lower_bound_to_goal(self, batch_cost):
        # run a reverse search on the transition nodes without any collision checking
        costs = {}
        closed_set = set()

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

            neighbors = [
                n.neighbors[0] for n in self.reverse_transition_nodes[node.state.mode]
            ]

            if not neighbors:
                continue

            if node.state.mode not in self.reverse_transition_node_array_cache:
                self.reverse_transition_node_array_cache[node.state.mode] = np.array(
                    [n.state.q.state() for n in neighbors], dtype=np.float64
                )

            closed_set.add(node.id)

            # add neighbors to open_queue
            edge_costs = batch_cost(
                node.state.q,
                self.reverse_transition_node_array_cache[node.state.mode],
            )

            parent_cost = costs[node.id]
            for edge_cost, n in zip(edge_costs, neighbors):
                cost = parent_cost + edge_cost
                id = n.id
                # current_cost = costs.get(id, float('inf'))
                # if cost < current_cost:
                #     costs[id] = cost
                #     n.lb_cost_to_goal = cost
                #     n.neighbors[0].lb_cost_to_goal = cost

                # print(cost)
                # parents[n] = node
                if id not in costs or cost < costs[id]:
                    # if n.neighbors[0].lb_cost_to_goal is not None:
                    #     print(n.neighbors[0].lb_cost_to_goal)
                    #     print(cost)
                    #     print()

                    costs[id] = cost
                    n.lb_cost_to_goal = cost

                    # if n.neighbors[0].lb_cost_to_goal is not None and n.neighbors[0].lb_cost_to_goal > cost:
                    #     print("AAAA")
                    #     n.neighbors[0].lb_cost_to_goal = cost

                    #     # parents[n] = node

                    # queue.append(n)
                    heapq.heappush(queue, (cost, n))

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def compute_lower_bound_from_start(self, batch_cost):
        # run a reverse search on the transition nodes without any collision checking
        costs = {}

        closed_set = set()

        queue = []
        heapq.heappush(queue, (0, self.root))
        costs[self.root.id] = 0

        while len(queue) > 0:
            # node = queue.pop(0)
            _, node = heapq.heappop(queue)
            # print(node)

            # if node.id in costs and:
            # continue

            if node.id in closed_set:
                continue

            if node.state.mode.task_ids == self.goal_nodes[0].state.mode.task_ids:
                continue

            # print(node.state.mode)
            # # print(node.neighbors[0].state.mode)
            # print(len(self.goal_nodes))
            # print(self.goal_nodes[0].state.mode.task_ids)

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
                # current_cost = costs.get(id, float('inf'))
                # if cost < current_cost:
                #     costs[id] = cost
                #     n.lb_cost_to_goal = cost
                #     n.neighbors[0].lb_cost_to_goal = cost

                # print(cost)
                # parents[n] = node
                if id not in costs or cost < costs[id]:
                    # if n.neighbors[0].lb_cost_to_goal is not None:
                    #     print(n.neighbors[0].lb_cost_to_goal)
                    #     print(cost)
                    #     print()

                    costs[id] = cost
                    n.lb_cost_from_start = cost

                    # if n.neighbors[0].lb_cost_to_goal is not None and n.neighbors[0].lb_cost_to_goal > cost:
                    #     print("AAAA")
                    #     n.neighbors[0].lb_cost_to_goal = cost

                    #     # parents[n] = node

                    # queue.append(n)
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

    def add_transition_nodes(self, transitions: Tuple[Configuration, Mode, Mode]):
        self.transition_node_array_cache = {}
        self.reverse_transition_node_array_cache = {}

        self.transition_node_lb_cache = {}
        self.rev_transition_node_lb_cache = {}

        for q, this_mode, next_mode in transitions:
            node_this_mode = Node(State(q, this_mode), True)
            node_next_mode = Node(State(q, next_mode), True)

            if next_mode is not None:
                node_next_mode.neighbors = [node_this_mode]
                node_this_mode.neighbors = [node_next_mode]

                assert this_mode.task_ids != next_mode.task_ids

            if this_mode in self.transition_nodes:
                is_in_transition_nodes_already = False
                dists = self.batch_dist_fun(
                    node_this_mode.state.q,
                    [n.state.q for n in self.transition_nodes[this_mode]],
                )
                if min(dists) < 1e-6:
                    is_in_goal_nodes_already = True
                # for n in self.transition_nodes[this_mode]:
                #     if (
                #         np.linalg.norm(
                #             n.state.q.state() - node_this_mode.state.q.state()
                #         )
                #         < 1e-3
                #     ):
                #         is_in_transition_nodes_already = True
                #         break

                if is_in_transition_nodes_already:
                    continue

            if next_mode is not None:
                if this_mode in self.transition_nodes:
                    self.transition_nodes[this_mode].append(node_this_mode)
                else:
                    self.transition_nodes[this_mode] = [node_this_mode]
            else:
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

            # add the same things to the rev transition nodes
            if next_mode is not None:
                if next_mode in self.reverse_transition_nodes:
                    self.reverse_transition_nodes[next_mode].append(node_next_mode)
                else:
                    self.reverse_transition_nodes[next_mode] = [node_next_mode]

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def get_neighbors(
        self, node: Node, space_extent: float = None
    ) -> Tuple[List[Node], NDArray]:
        key = node.state.mode
        if key in self.nodes:
            node_list = self.nodes[key]

            if key not in self.node_array_cache:
                self.node_array_cache[key] = np.array(
                    [n.state.q.q for n in node_list], dtype=np.float64
                )

            dists = self.batch_dist_fun(
                node.state.q, self.node_array_cache[key]
            )  # this, and the list copm below are the slowest parts

        if key in self.transition_nodes:
            transition_node_list = self.transition_nodes[key]

            if key not in self.transition_node_array_cache:
                self.transition_node_array_cache[key] = np.array(
                    [n.state.q.q for n in transition_node_list], dtype=np.float64
                )

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
        approximate_space_extent: float = None,
    ) -> List[Node]:
        if approximate_space_extent is None:
            approximate_space_extent = np.prod(np.diff(env.limits, axis=0))

        goal = None
        h_cache = {}

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

        # open_queue.sort(reverse=True)

        # for k, v in self.transition_nodes.items():
        #     for n in v:
        #         if len(n.neighbors) > 0:
        #             assert(n.lb_cost_to_goal == n.neighbors[0].lb_cost_to_goal)

        wasted_pops = 0
        processed_edges = 0

        queue_pop = queue.heappop
        queue_push = queue.heappush

        num_iter = 0
        while queue:
            # while len(open_queue) > 0:
            num_iter += 1

            # print(len(queue))

            if num_iter % 100000 == 0:
                # print(len(open_queue))
                print(len(queue))

            # f_pred, edge_cost, (n0, n1) = heapq.heappop(open_queue)
            f_pred, edge_cost, (n0, n1) = queue_pop()
            # print(open_queue[-1])
            # print(open_queue[-2])
            # f_pred, edge_cost, edge = open_queue.pop()
            # n0, n1 = edge

            # g_tentative = gs[n0.id] + edge_cost

            # if we found a better way to get there before, do not expand this edge
            # if n1.id in gs:
            #     print(f_pred)
            #     print(gs[n0.id])
            #     print('new_cost/oldcost', g_tentative, gs[n1.id])
            #     print('n0id/n1id', n0.id, n1.id)
            #     print('n0mode/n1mode', n0.state.mode, n1.state.mode)
            #     print('n0trans/n1trans', n0.is_transition, n1.is_transition)

            #     print("root", self.root.state.q.state())

            #     if (n1.is_transition):
            #         print('n1 is transition')
            #         print(n1.neighbors[0].id)
            #         print(n1.neighbors[0].state.mode)
            #         print(n1.neighbors[0].state.q.state())

            #     if (n0.is_transition):
            #         print('n0 is transition')
            #         print(n0.neighbors[0].id)

            #     print(edge_cost)

            #     print(gs[n0.id] + h(n0))
            #     print(gs[n1.id] + h(n1))

            #     assert(g_tentative >= gs[n1.id])

            # if n1.id in gs and g_tentative >= gs[n1.id]:
            if n1.id in gs:
                wasted_pops += 1
                continue

            # check edge sparsely now. if it is not valid, blacklist it, and continue with the next edge
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

            # remove all other edges with this goal from the queue

            # if n0.state.mode not in reached_modes:
            #     reached_modes.append(n0.state.mode)

            # print('reached modes', reached_modes)

            # print('adding', n1.id)
            g_tentative = gs[n0.id] + edge_cost
            gs[n1.id] = g_tentative
            parents[n1] = n0

            # if len(queue) > 1e6:
            #     for node in parents:
            #         queue.remove_by_node(n1)

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
                    # if n.id not in gs or g_new < gs[n.id]:
                    # sparsely check only when expanding
                    # cost to get to neighbor:
                    # q0 = n1.state.q
                    # q1 = n.state.q
                    # collision_free = env.is_edge_collision_free(
                    #     q0, q1, n1.state.mode, 5
                    # )
                    # if not collision_free:
                    #     n.blacklist.add(n1.id)
                    #     n1.blacklist.add(n.id)

                    #     continue

                    g_new = g_tentative + edge_cost
                    f_node = g_new + h(n)
                    # fs[(n1, n)] = f_node

                    if best_cost is not None and f_node > best_cost:
                        continue

                    # if n not in closed_list:
                    # heapq.heappush(open_queue, (f_node, edge_cost, (n1, n)))
                    queue_push((f_node, edge_cost, (n1, n)))
                    # new_edges.append((f_node, edge_cost, (n1, n)))

                    # added_edge = True
                    # queue.append((f_node, edge_cost, (n1, n)))
            # heapq.heapify(open_queue)
            # open_queue.sort(reverse=True)

            # if len(new_edges) > 0:
            #     queue.heappush_list(new_edges)

        path = []

        if goal is not None:
            path.append(goal)

            n = goal

            while parents[n] is not None:
                path.append(parents[n])
                n = parents[n]

                # print(gs[n.id] + h(n))
                # print('\t\t', h(n))

            path.append(n)
            path = path[::-1]

        print("Wasted pops", wasted_pops)
        print("Processed edges", processed_edges)

        return path

    def search_with_vertex_queue(
        self,
        start_node: Node,
        goal_nodes: List[Node],
        env: BaseProblem,
        best_cost: Optional[float] = None,
        resolution: float = 0.1,
        approximate_space_extent: float = None,
    ) -> List[Node]:
        open_queue = []

        goal = None

        h_cache = {}

        def h(node):
            # return 0
            if node.id in h_cache:
                return h_cache[node.id]

            if node.state.mode not in self.transition_node_array_cache:
                self.transition_node_array_cache[node.state.mode] = np.array(
                    [n.state.q.q for n in self.transition_nodes[node.state.mode]],
                    dtype=np.float64,
                )

            if node.state.mode not in self.transition_node_lb_cache:
                self.transition_node_lb_cache[node.state.mode] = np.array(
                    [n.lb_cost_to_goal for n in self.transition_nodes[node.state.mode]],
                    dtype=np.float64,
                )

            costs_to_transitions = env.batch_config_cost(
                node.state.q,
                self.transition_node_array_cache[node.state.mode],
            )

            min_cost = np.min(
                self.transition_node_lb_cache[node.state.mode] - costs_to_transitions
            )

            h_cache[node.id] = min_cost
            return min_cost

        def d(n0, n1):
            # return 1.0
            cost = env.config_cost(n0.state.q, n1.state.q)
            return cost

        parents = {start_node: None}
        gs = {start_node: 0}  # best cost to get to a node

        # populate open_queue and fs

        # fs = {start_node: h(start_node)}  # total cost of a node (f = g + h)
        heapq.heappush(open_queue, (0, start_node))

        num_iter = 0
        while open_queue:
            num_iter += 1

            if num_iter % 1000 == 0:
                print(len(open_queue))

            f_val, node = heapq.heappop(open_queue)
            # print('g:', v)

            # print(num_iter, len(open_queue))

            # if n0.state.mode == [0, 3]:
            #     env.show(True)

            if node in goal_nodes:
                goal = node
                break

            # get_neighbors
            neighbors, tmp = self.get_neighbors(
                node, space_extent=approximate_space_extent
            )

            edge_costs = env.batch_config_cost(node.state.q, tmp)
            # add neighbors to open_queue
            for i, n in enumerate(neighbors):
                if n == node:
                    continue

                if node.id in n.blacklist:
                    continue

                g_new = gs[node] + edge_costs[i]

                if n not in gs or g_new < gs[n]:
                    # collision check

                    collision_free = False
                    if n.id in node.whitelist:
                        collision_free = True
                    else:
                        if n.id in node.blacklist:
                            continue

                        collision_free = env.is_edge_collision_free(
                            node.state.q, n.state.q, n.state.mode, resolution=0.1
                        )

                        if not collision_free:
                            node.blacklist.add(n.id)
                            n.blacklist.add(node.id)
                            continue
                        # else:
                        #     node.whitelist.add(n.id)
                        #     n.whitelist.add(node.id)

                    # cost to get to neighbor:
                    gs[n] = g_new
                    cost = g_new + h(n)
                    parents[n] = node

                    if best_cost is not None and cost > best_cost:
                        continue

                    # if n not in open_queue:
                    heapq.heappush(open_queue, (cost, n))

        path = []

        if goal is not None:
            path.append(goal)

            n = goal

            while parents[n] is not None:
                path.append(parents[n])
                n = parents[n]

            path.append(n)

            path = path[::-1]

        return path


# @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
def joint_prm_planner(
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
) -> Optional[Tuple[List[State], List]]:
    q0 = env.get_start_pos()
    m0 = env.get_start_mode()

    reached_modes = [m0]

    conf_type = type(env.get_start_pos())

    def sample_mode(
        mode_sampling_type: str = "uniform_reached", found_solution: bool = False
    ) -> Mode:
        if mode_sampling_type == "uniform_reached":
            m_rnd = random.choice(reached_modes)
        elif mode_sampling_type == "weighted":
            # sample such that we tend to get similar number of pts in each mode
            w = []
            for m in reached_modes:
                num_nodes = 0
                if m in g.nodes:
                    num_nodes += len(g.nodes[m])
                if m in g.transition_nodes:
                    num_nodes += len(g.transition_nodes[m])
                w.append(1 / max(1, num_nodes))
            m_rnd = random.choices(reached_modes, weights=w)[0]
        # elif mode_sampling_type == "greedy_until_first_sol":
        #     if found_solution:
        #         m_rnd = reached_modes[-1]
        #     else:
        #         w = []
        #         for m in reached_modes:
        #             w.append(1 / len(g.nodes[tuple(m)]))
        #         m_rnd = random.choices(reached_modes, weights=w)[0]
        # else:
        #     # sample very greedily and only expand the newest mode
        #     m_rnd = reached_modes[-1]

        return m_rnd

    # we are checking here if a sample can imrpove a part of the path
    # the condition to do so is that the
    def can_improve(
        rnd_state: State, path: List[State], start_index, end_index, path_segment_costs
    ) -> bool:
        # path_segment_costs = env.batch_config_cost(path[:-1], path[1:])

        # compute the local cost
        path_cost_from_start_to_index = np.sum(path_segment_costs[:start_index])
        path_cost_from_goal_to_index = np.sum(path_segment_costs[end_index:])
        path_cost = np.sum(path_segment_costs)

        if start_index == 0:
            assert path_cost_from_start_to_index == 0
        if end_index == len(path) - 1:
            assert path_cost_from_goal_to_index == 0

        path_cost_from_index_to_index = (
            path_cost - path_cost_from_goal_to_index - path_cost_from_start_to_index
        )

        # print(path_cost_from_index_to_index)

        def lb_cost_from_start(state):
            if state.mode not in g.reverse_transition_node_array_cache:
                g.reverse_transition_node_array_cache[state.mode] = np.array(
                    [o.state.q.q for o in g.reverse_transition_nodes[state.mode]],
                    dtype=np.float64,
                )

            if state.mode not in g.rev_transition_node_lb_cache:
                g.rev_transition_node_lb_cache[state.mode] = np.array(
                    [
                        o.lb_cost_from_start
                        for o in g.reverse_transition_nodes[state.mode]
                    ],
                    dtype=np.float64,
                )

            costs_to_transitions = env.batch_config_cost(
                state.q,
                g.reverse_transition_node_array_cache[state.mode],
            )

            min_cost = np.min(
                g.rev_transition_node_lb_cache[state.mode] + costs_to_transitions
            )

            return min_cost

        def lb_cost_from_goal(state):
            if state.mode not in g.transition_nodes:
                return np.inf

            if state.mode not in g.transition_node_array_cache:
                g.transition_node_array_cache[state.mode] = np.array(
                    [o.state.q.q for o in g.transition_nodes[state.mode]],
                    dtype=np.float64,
                )

            if state.mode not in g.transition_node_lb_cache:
                g.transition_node_lb_cache[state.mode] = np.array(
                    [o.lb_cost_to_goal for o in g.transition_nodes[state.mode]],
                    dtype=np.float64,
                )

            costs_to_transitions = env.batch_config_cost(
                state.q,
                g.transition_node_array_cache[state.mode],
            )

            min_cost = np.min(
                g.transition_node_lb_cache[state.mode] + costs_to_transitions
            )

            return min_cost

        lb_cost_from_start_index_to_state = env.config_cost(
            rnd_state.q, path[start_index].q
        )
        # if path[start_index].mode != rnd_state.mode:
        #     start_state = path[start_index]
        #     lb_cost_from_start_to_state = lb_cost_from_start(rnd_state)
        #     lb_cost_from_start_to_index = lb_cost_from_start(start_state)

        #     lb_cost_from_start_index_to_state = max(
        #         (lb_cost_from_start_to_state - lb_cost_from_start_to_index),
        #         lb_cost_from_start_index_to_state,
        #     )

        lb_cost_from_state_to_end_index = env.config_cost(
            rnd_state.q, path[end_index].q
        )
        # if path[end_index].mode != rnd_state.mode:
        #     goal_state = path[end_index]
        #     lb_cost_from_goal_to_state = lb_cost_from_goal(rnd_state)
        #     lb_cost_from_goal_to_index = lb_cost_from_goal(goal_state)

        #     lb_cost_from_state_to_end_index = max(
        #         (lb_cost_from_goal_to_state - lb_cost_from_goal_to_index),
        #         lb_cost_from_state_to_end_index,
        #     )

        # print("can_imrpove")

        # print("start", lb_cost_from_start_index_to_state)
        # print("end", lb_cost_from_state_to_end_index)

        # print('start index', start_index)
        # print('end_index', end_index)

        # assert(lb_cost_from_start_index_to_state >= 0)
        # assert(lb_cost_from_state_to_end_index >= 0)

        # print("segment cost", path_cost_from_index_to_index)
        # print(
        #     "lb cost",
        #     lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index,
        # )

        if (
            path_cost_from_index_to_index
            > lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index
        ):
            return True

        return False

    def get_inbetween_modes(start_mode, end_mode):
        """
        Find all possible paths from start_mode to end_mode.

        Args:
            start_mode: The starting mode object
            end_mode: The ending mode object

        Returns:
            A list of lists, where each inner list represents a valid path
            from start_mode to end_mode (inclusive of both).
        """
        # Store all found paths
        open_paths = [[start_mode]]

        in_between_modes = set()
        in_between_modes.add(start_mode)
        in_between_modes.add(end_mode)

        while open_paths:
            p = open_paths.pop()
            last_mode = p[-1]

            if last_mode == end_mode:
                for m in p:
                    in_between_modes.add(m)
                continue

            if last_mode.next_modes:
                for mode in last_mode.next_modes:
                    new_path = p.copy()
                    new_path.append(mode)
                    open_paths.append(new_path)

        return list(in_between_modes)

    def generate_informed_samples(
        batch_size,
        path,
        max_attempts_per_sample=200,
        locally_informed_sampling=True,
        try_direct_sampling=True,
    ):
        new_samples = []
        path_segment_costs = env.batch_config_cost(path[:-1], path[1:])

        in_between_mode_cache = {}

        num_attempts = 0
        while len(new_samples) < batch_size:
            if num_attempts > batch_size:
                break

            num_attempts += 1
            # print(len(new_samples))
            # sample mode
            if locally_informed_sampling:
                for _ in range(500):
                    start_ind = random.randint(0, len(path) - 1)
                    end_ind = random.randint(0, len(path) - 1)

                    if end_ind - start_ind > 2:
                        # if end_ind - start_ind > 2 and end_ind - start_ind < 50:
                        current_cost = sum(path_segment_costs[start_ind:end_ind])
                        lb_cost = env.config_cost(path[start_ind].q, path[end_ind].q)

                        if lb_cost < current_cost:
                            break

                # TODO: we need to sample from the set of all reachable modes here
                # not only from the modes on the path
                if (
                    path[start_ind].mode,
                    path[end_ind].mode,
                ) not in in_between_mode_cache:
                    in_between_modes = get_inbetween_modes(
                        path[start_ind].mode, path[end_ind].mode
                    )
                    in_between_mode_cache[
                        (path[start_ind].mode, path[end_ind].mode)
                    ] = in_between_modes

                # print(in_between_mode_cache[(path[start_ind].mode, path[end_ind].mode)])

                m = random.choice(
                    in_between_mode_cache[(path[start_ind].mode, path[end_ind].mode)]
                )

                # k = random.randint(start_ind, end_ind)
                # m = path[k].mode
            else:
                start_ind = 0
                end_ind = len(path) - 1
                m = sample_mode("uniform_reached", True)

            # print(m)

            current_cost = sum(path_segment_costs[start_ind:end_ind])

            # tmp = 0
            # for i in range(start_ind, end_ind):
            #     tmp += env.config_cost(path[i].q, path[i+1].q)

            # print(current_cost, tmp)

            # plt.figure()
            # samples = []
            # for _ in range(500):
            #     sample = samplePHS(np.array([-1, 1, 0]), np.array([1, -1, 0]), 3)
            #     # sample = samplePHS(np.array([[-1], [0]]), np.array([[1], [0]]), 3)
            #     samples.append(sample[:2])
            #     print("sample", sample)

            # plt.scatter([a[0] for a in samples], [a[1] for a in samples])
            # plt.show()

            focal_points = np.array(
                [path[start_ind].q.state(), path[end_ind].q.state()], dtype=np.float64
            )

            precomputed_phs_matrices = {}
            precomputed_robot_cost_bounds = {}

            is_almost_the_same = {}

            obv_inv_attempts = 0
            sample_in_collision = 0

            num_samples_at_a_time = 10

            for k in range(max_attempts_per_sample // num_samples_at_a_time):
                if not try_direct_sampling or env.cost_metric != "euclidean":
                    # completely random sample configuration from the (valid) domain robot by robot
                    q = env.sample_config_uniform_in_limits()
                else:
                    # sample by sampling each agent separately
                    q = []
                    for i, r in enumerate(env.robots):
                        lims = env.limits[:, env.robot_idx[r]]
                        if lims[0, 0] < lims[1, 0]:
                            if i not in precomputed_robot_cost_bounds:
                                if env.cost_reduction == "sum":
                                    precomputed_robot_cost_bounds[i] = (
                                        current_cost
                                        - sum(
                                            [
                                                np.linalg.norm(
                                                    path[start_ind].q[j]
                                                    - path[end_ind].q[j]
                                                )
                                                for j in range(len(env.robots))
                                                if j != i
                                            ]
                                        )
                                    )
                                else:
                                    precomputed_robot_cost_bounds[i] = current_cost

                            if i not in is_almost_the_same:
                                is_almost_the_same[i] = (
                                    np.linalg.norm(
                                        path[start_ind].q[i] - path[end_ind].q[i]
                                    )
                                    < 1e-3
                                )

                            if is_almost_the_same[i]:
                                qr = (
                                    np.random.uniform(size=(num_samples_at_a_time, env.robot_dims[r]), low=lims[0, :], high = lims[1, :]).T
                                )
                            else:
                                # print("cost", current_cost)
                                # print("robot cst", c_robot_bound)
                                # print(
                                #     np.linalg.norm(
                                #         path[start_ind].q[i] - path[end_ind].q[i]
                                #     )
                                # )

                                if i not in precomputed_phs_matrices:
                                    precomputed_phs_matrices[i] = compute_PHS_matrices(
                                        path[start_ind].q[i],
                                        path[end_ind].q[i],
                                        precomputed_robot_cost_bounds[i],
                                    )

                                qr = sample_phs_with_given_matrices(
                                    *precomputed_phs_matrices[i], n=num_samples_at_a_time
                                )

                                # plt.figure()
                                # samples = []
                                # for _ in range(500):
                                #     sample = sample_phs_with_given_matrices(
                                #         *precomputed_phs_matrices[i]
                                #     )
                                #     # sample = samplePHS(np.array([[-1], [0]]), np.array([[1], [0]]), 3)
                                #     samples.append(sample[:2])
                                #     print("sample", sample)

                                # plt.scatter(
                                #     [a[0] for a in samples], [a[1] for a in samples]
                                # )
                                # plt.show()

                                # qr = samplePHS(path[start_ind].q[i], path[end_ind].q[i], c_robot_bound)
                                # qr = rejection_sample_from_ellipsoid(
                                #     path[start_ind].q[i], path[end_ind].q[i], c_robot_bound
                                # )

                                # if np.linalg.norm(path[start_ind].q[i] - qr) + np.linalg.norm(path[end_ind].q[i] - qr) > c_robot_bound:
                                #     print("AAAAAAAAA")
                                #     print(np.linalg.norm(path[start_ind].q[i] - qr) + np.linalg.norm(path[end_ind].q[i] - qr), c_robot_bound)

                                # clipped = np.clip(qr, lims[0, :], lims[1, :])
                                # if not np.array_equal(clipped, qr):
                                # if np.any((qr < lims[0, :]) | (qr > lims[1, :])):
                                #     had_to_be_clipped = True
                                #     break
                                # print("AAA")

                        q.append(qr)

                if isinstance(q, list):
                    qs = []
                    for i in range(num_samples_at_a_time):
                        q_config = []
                        for j in range(len(env.robots)):
                            q_config.append(q[j][:, i])

                        qnp = np.concatenate(q_config)
                        qs.append(conf_type(qnp, env.start_pos.array_slice))
                else:
                    qs = [q]

                found_a_sample = False
                for q in qs:
                    if not isinstance(q, Configuration):
                        # q = conf_type.from_list(q)
                        qnp = np.concatenate(q)
                        if np.any((qnp < env.limits[0, :]) | (qnp > env.limits[1, :])):
                            continue
                        q = conf_type(qnp, env.start_pos.array_slice)

                    if sum(env.batch_config_cost(q, focal_points)) > current_cost:
                        # print(path[start_ind].mode, path[end_ind].mode, m)
                        # print(
                        #     current_cost,
                        #     env.config_cost(path[start_ind].q, q)
                        #     + env.config_cost(path[end_ind].q, q),
                        # )
                        # if can_improve(State(q, m), path, start_ind, end_ind):
                        #     assert False

                        obv_inv_attempts += 1

                        continue

                    # if can_improve(State(q, m), path, 0, len(path)-1):
                    # if can_improve(State(q, m), path, start_ind, end_ind):
                    # if not env.is_collision_free(q, m):
                    #     sample_in_collision += 1
                    #     continue

                    if can_improve(
                        State(q, m), path, start_ind, end_ind, path_segment_costs
                    ) and env.is_collision_free(q, m):
                        # if env.is_collision_free(q, m) and can_improve(State(q, m), path, 0, len(path)-1):
                        new_samples.append(State(q, m))
                        found_a_sample = True
                        # break

                if found_a_sample:
                    break

            # print("inv attempt", obv_inv_attempts)
            # print("coll", sample_in_collision)

        print(len(new_samples) / num_attempts)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter([a.q[0][0] for a in new_samples], [a.q[0][1] for a in new_samples], [a.q[0][2] for a in new_samples])
        # ax.scatter([a.q[1][0] for a in new_samples], [a.q[1][1] for a in new_samples], [a.q[1][2] for a in new_samples])
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.scatter([a.q[0][0] for a in new_samples], [a.q[0][1] for a in new_samples])
        # ax.scatter([a.q[1][0] for a in new_samples], [a.q[1][1] for a in new_samples])
        # plt.show()

        return new_samples

    def can_transition_improve(transition, path, start_index, end_index):
        path_segment_costs = env.batch_config_cost(path[:-1], path[1:])

        # compute the local cost
        path_cost_from_start_to_index = np.sum(path_segment_costs[:start_index])
        path_cost_from_goal_to_index = np.sum(path_segment_costs[end_index:])
        path_cost = np.sum(path_segment_costs)

        if start_index == 0:
            assert path_cost_from_start_to_index == 0
        if end_index == len(path) - 1:
            assert path_cost_from_goal_to_index == 0

        path_cost_from_index_to_index = (
            path_cost - path_cost_from_goal_to_index - path_cost_from_start_to_index
        )

        # print(path_cost_from_index_to_index)

        def lb_cost_from_start(state):
            if state.mode not in g.reverse_transition_node_array_cache:
                g.reverse_transition_node_array_cache[state.mode] = np.array(
                    [o.state.q.q for o in g.reverse_transition_nodes[state.mode]],
                    dtype=np.float64,
                )

            if state.mode not in g.rev_transition_node_lb_cache:
                g.rev_transition_node_lb_cache[state.mode] = np.array(
                    [
                        o.lb_cost_from_start
                        for o in g.reverse_transition_nodes[state.mode]
                    ],
                    dtype=np.float64,
                )

            costs_to_transitions = env.batch_config_cost(
                state.q,
                g.reverse_transition_node_array_cache[state.mode],
            )

            min_cost = np.min(
                g.rev_transition_node_lb_cache[state.mode] + costs_to_transitions
            )

            return min_cost

        def lb_cost_from_goal(state):
            if state.mode not in g.transition_node_array_cache:
                g.transition_node_array_cache[state.mode] = np.array(
                    [o.state.q.q for o in g.transition_nodes[state.mode]],
                    dtype=np.float64,
                )

            if state.mode not in g.transition_node_lb_cache:
                g.transition_node_lb_cache[state.mode] = np.array(
                    [o.lb_cost_to_goal for o in g.transition_nodes[state.mode]],
                    dtype=np.float64,
                )

            costs_to_transitions = env.batch_config_cost(
                state.q,
                g.transition_node_array_cache[state.mode],
            )

            min_cost = np.min(
                g.transition_node_lb_cache[state.mode] + costs_to_transitions
            )

            return min_cost

        rnd_state_mode_1 = State(transition[0], transition[1])
        rnd_state_mode_2 = State(transition[0], transition[2])

        lb_cost_from_start_index_to_state = env.config_cost(
            rnd_state_mode_1.q, path[start_index].q
        )
        # if path[start_index].mode != rnd_state_mode_1.mode:
        #     start_state = path[start_index]
        #     lb_cost_from_start_to_state = lb_cost_from_start(rnd_state_mode_1)
        #     lb_cost_from_start_to_index = lb_cost_from_start(start_state)

        #     lb_cost_from_start_index_to_state = max(
        #         (lb_cost_from_start_to_state - lb_cost_from_start_to_index),
        #         lb_cost_from_start_index_to_state,
        #     )

        lb_cost_from_state_to_end_index = env.config_cost(
            rnd_state_mode_2.q, path[end_index].q
        )
        # if path[end_index].mode != rnd_state_mode_2.mode:
        #     goal_state = path[end_index]
        #     lb_cost_from_goal_to_state = lb_cost_from_goal(rnd_state_mode_2)
        #     lb_cost_from_goal_to_index = lb_cost_from_goal(goal_state)

        #     lb_cost_from_state_to_end_index = max(
        #         (lb_cost_from_goal_to_state - lb_cost_from_goal_to_index),
        #         lb_cost_from_state_to_end_index,
        #     )

        # print("can_imrpove")

        # print("start", lb_cost_from_start_index_to_state)
        # print("end", lb_cost_from_state_to_end_index)

        # print('start index', start_index)
        # print('end_index', end_index)

        # assert(lb_cost_from_start_index_to_state >= 0)
        # assert(lb_cost_from_state_to_end_index >= 0)

        # print("segment cost", path_cost_from_index_to_index)
        # print(
        #     "lb cost",
        #     lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index,
        # )

        if (
            path_cost_from_index_to_index
            > lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index
        ):
            return True

        return False

    def generate_informed_transitions(
        batch_size, path, locally_informed_sampling=False, max_attempts_per_sample=100
    ):
        if len(env.tasks) == 1:
            return []

        new_transitions = []
        num_attempts = 0
        path_segment_costs = env.batch_config_cost(path[:-1], path[1:])

        in_between_mode_cache = {}

        while len(new_transitions) < batch_size:
            num_attempts += 1

            if num_attempts > batch_size:
                break

            # print(len(new_samples))
            # sample mode
            if locally_informed_sampling:
                while True:
                    start_ind = random.randint(0, len(path) - 1)
                    end_ind = random.randint(0, len(path) - 1)

                    if (
                        path[end_ind].mode != path[start_ind].mode
                        and end_ind - start_ind > 2
                    ):
                        # if end_ind - start_ind > 2 and end_ind - start_ind < 50:
                        current_cost = sum(path_segment_costs[start_ind:end_ind])
                        lb_cost = env.config_cost(path[start_ind].q, path[end_ind].q)

                        if lb_cost < current_cost:
                            break

                if (
                    path[start_ind].mode,
                    path[end_ind].mode,
                ) not in in_between_mode_cache:
                    in_between_modes = get_inbetween_modes(
                        path[start_ind].mode, path[end_ind].mode
                    )
                    in_between_mode_cache[
                        (path[start_ind].mode, path[end_ind].mode)
                    ] = in_between_modes

                # print(in_between_mode_cache[(path[start_ind].mode, path[end_ind].mode)])

                mode = random.choice(
                    in_between_mode_cache[(path[start_ind].mode, path[end_ind].mode)]
                )

                # k = random.randint(start_ind, end_ind)
                # mode = path[k].mode
            else:
                start_ind = 0
                end_ind = len(path) - 1
                mode = sample_mode("uniform_reached", True)

            # print(m)

            current_cost = sum(path_segment_costs[start_ind:end_ind])

            # sample transition at the end of this mode
            possible_next_task_combinations = env.get_valid_next_task_combinations(mode)
            if len(possible_next_task_combinations) > 0:
                ind = random.randint(0, len(possible_next_task_combinations) - 1)
                active_task = env.get_active_task(
                    mode, possible_next_task_combinations[ind]
                )
            else:
                continue

            goals_to_sample = active_task.robots

            goal_sample = active_task.goal.sample(mode)

            focal_points = np.array(
                [path[start_ind].q.state(), path[end_ind].q.state()], dtype=np.float64
            )

            for k in range(max_attempts_per_sample):
                # completely random sample configuration from the (valid) domain robot by robot
                q = []
                for i in range(len(env.robots)):
                    r = env.robots[i]
                    if r in goals_to_sample:
                        offset = 0
                        for _, task_robot in enumerate(active_task.robots):
                            if task_robot == r:
                                q.append(
                                    goal_sample[
                                        offset : offset + env.robot_dims[task_robot]
                                    ]
                                )
                                break
                            offset += env.robot_dims[task_robot]
                    else:  # uniform sample
                        lims = env.limits[:, env.robot_idx[r]]
                        if lims[0, 0] < lims[1, 0]:
                            qr = (
                                np.random.rand(env.robot_dims[r])
                                * (lims[1, :] - lims[0, :])
                                + lims[0, :]
                            )
                        else:
                            qr = np.random.rand(env.robot_dims[r]) * 6 - 3

                        q.append(qr)

                q = conf_type(np.concatenate(q), env.start_pos.array_slice)

                if sum(env.batch_config_cost(q, focal_points)) > current_cost:
                    continue

                if env.is_terminal_mode(mode):
                    assert False
                else:
                    next_mode = env.get_next_mode(q, mode)

                if can_transition_improve(
                    (q, mode, next_mode), path, start_ind, end_ind
                ) and env.is_collision_free(q, mode):
                    new_transitions.append((q, mode, next_mode))
                    break

        print(len(new_transitions) / num_attempts)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter([a.q[0][0] for a in new_samples], [a.q[0][1] for a in new_samples], [a.q[0][2] for a in new_samples])
        # ax.scatter([a.q[1][0] for a in new_samples], [a.q[1][1] for a in new_samples], [a.q[1][2] for a in new_samples])
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.scatter([a[0][0][0] for a in new_transitions], [a[0][0][1] for a in new_transitions])
        # ax.scatter([a[0][1][0] for a in new_transitions], [a[0][1][1] for a in new_transitions])
        # plt.show()

        return new_transitions

    def sample_around_path(path):
        # sample index
        interpolated_path = interpolate_path(path)
        # interpolated_path = current_best_path
        new_states_from_path_sampling = []
        new_transitions_from_path_sampling = []
        for _ in range(path_batch_size):
            idx = random.randint(0, len(interpolated_path) - 2)
            state = interpolated_path[idx]

            # this is a transition. we would need to figure out which robots are active and not sample those
            q = []
            if (
                state.mode != interpolated_path[idx + 1].mode
                and np.linalg.norm(
                    state.q.state() - interpolated_path[idx + 1].q.state()
                )
                < 1e-5
            ):
                next_task_ids = interpolated_path[idx + 1].mode.task_ids

                # TODO: this seems to move transitions around
                task = env.get_active_task(state.mode, next_task_ids)
                involved_robots = task.robots
                for i in range(len(env.robots)):
                    r = env.robots[i]
                    if r in involved_robots:
                        qr = state.q[i] * 1.0
                    else:
                        qr_mean = state.q[i] * 1.0

                        qr = np.random.rand(len(qr_mean)) * 0.5 + qr_mean

                        lims = env.limits[:, env.robot_idx[r]]
                        if lims[0, 0] < lims[1, 0]:
                            qr = np.clip(qr, lims[0, :], lims[1, :])

                    q.append(qr)

                q = conf_type.from_list(q)

                if env.is_collision_free(q, state.mode):
                    new_transitions_from_path_sampling.append(
                        (q, state.mode, interpolated_path[idx + 1].mode)
                    )

            else:
                for i in range(len(env.robots)):
                    r = env.robots[i]
                    qr_mean = state.q[i]

                    qr = np.random.rand(len(qr_mean)) * 0.5 + qr_mean

                    lims = env.limits[:, env.robot_idx[r]]
                    if lims[0, 0] < lims[1, 0]:
                        qr = np.clip(qr, lims[0, :], lims[1, :])

                    q.append(qr)

                q = conf_type.from_list(q)

                if env.is_collision_free(q, state.mode):
                    rnd_state = State(q, state.mode)
                    new_states_from_path_sampling.append(rnd_state)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter([a.q[0][0] for a in new_states_from_path_sampling], [a.q[0][1] for a in new_states_from_path_sampling], [a.q[0][2] for a in new_states_from_path_sampling])
        # ax.scatter([a.q[1][0] for a in new_states_from_path_sampling], [a.q[1][1] for a in new_states_from_path_sampling], [a.q[1][2] for a in new_states_from_path_sampling])
        # ax.scatter([a.q[2][0] for a in new_states_from_path_sampling], [a.q[2][1] for a in new_states_from_path_sampling], [a.q[1][2] for a in new_states_from_path_sampling])
        # plt.show()

        return new_states_from_path_sampling, new_transitions_from_path_sampling

    def sample_valid_uniform_batch(batch_size: int, cost: float) -> List[State]:
        new_samples = []
        num_attempts = 0
        num_valid = 0

        if g.goal_nodes:
            focal_points = np.array(
                [g.root.state.q.state(), g.goal_nodes[0].state.q.state()],
                dtype=np.float64,
            )

        while len(new_samples) < batch_size:
            num_attempts += 1
            # print(len(new_samples))
            # sample mode
            m = sample_mode("uniform_reached", cost is not None)

            # print(m)

            # sample configuration
            q = env.sample_config_uniform_in_limits()

            if cost is not None:
                if sum(env.batch_config_cost(q, focal_points)) > cost:
                    continue

            if env.is_collision_free(q, m):
                new_samples.append(State(q, m))
                num_valid += 1

        print("Percentage of succ. attempts", num_valid / num_attempts)

        return new_samples, num_attempts

    def sample_valid_uniform_transitions(transistion_batch_size, cost):
        transitions = []

        if g.goal_nodes:
            focal_points = np.array(
                [g.root.state.q.state(), g.goal_nodes[0].state.q.state()],
                dtype=np.float64,
            )

        while len(transitions) < transistion_batch_size:
            # sample mode
            mode = sample_mode("uniform_reached", None)

            # sample transition at the end of this mode
            possible_next_task_combinations = env.get_valid_next_task_combinations(mode)
            # print(mode, possible_next_task_combinations)

            if len(possible_next_task_combinations) > 0:
                ind = random.randint(0, len(possible_next_task_combinations) - 1)
                active_task = env.get_active_task(
                    mode, possible_next_task_combinations[ind]
                )
            else:
                active_task = env.get_active_task(mode, None)

            goals_to_sample = active_task.robots
            goal_sample = active_task.goal.sample(mode)

            # if mode.task_ids == [3, 8]:
            #     print(active_task.name)

            # q = env.sample_config_uniform_in_limits()

            # for i in range(len(env.robots)):
            #     r = env.robots[i]
            #     if r in goals_to_sample:
            #         offset = 0
            #         for _, task_robot in enumerate(active_task.robots):
            #             if task_robot == r:
            #                 q[i] = goal_sample[offset : offset + env.robot_dims[task_robot]]

            q = []
            for i in range(len(env.robots)):
                r = env.robots[i]
                if r in goals_to_sample:
                    offset = 0
                    for _, task_robot in enumerate(active_task.robots):
                        if task_robot == r:
                            q.append(
                                goal_sample[
                                    offset : offset + env.robot_dims[task_robot]
                                ]
                            )
                            break
                        offset += env.robot_dims[task_robot]
                else:  # uniform sample
                    lims = env.limits[:, env.robot_idx[r]]
                    if lims[0, 0] < lims[1, 0]:
                        qr = (
                            np.random.rand(env.robot_dims[r])
                            * (lims[1, :] - lims[0, :])
                            + lims[0, :]
                        )
                    else:
                        qr = np.random.rand(env.robot_dims[r]) * 6 - 3

                    q.append(qr)

            q = conf_type(np.concatenate(q), env.start_pos.array_slice)

            if cost is not None:
                if sum(env.batch_config_cost(q, focal_points)) > cost:
                    continue

            if env.is_collision_free(q, mode):
                if env.is_terminal_mode(mode):
                    next_mode = None
                else:
                    next_mode = env.get_next_mode(q, mode)

                transitions.append((q, mode, next_mode))

                # print(mode, mode.next_modes)

                if next_mode not in reached_modes and next_mode is not None:
                    reached_modes.append(next_mode)
            # else:
            #     if mode.task_ids == [3, 8]:
            #         env.show(True)

        return transitions

    g = Graph(
        State(q0, m0),
        lambda a, b: batch_config_dist(a, b, distance_metric),
        use_k_nearest=use_k_nearest,
    )

    current_best_cost = None
    current_best_path = None

    costs = []
    times = []

    add_new_batch = True

    start_time = time.time()

    resolution = env.collision_resolution

    all_paths = []

    approximate_space_extent = np.prod(np.diff(env.limits, axis=0))

    cnt = 0
    while True:
        if current_best_path is not None:
            # prune
            num_pts_for_removal = 0
            focal_points = np.array(
                [g.root.state.q.state(), g.goal_nodes[0].state.q.state()],
                dtype=np.float64,
            )
            # for mode, nodes in g.nodes.items():
            #     for n in nodes:
            #         if sum(env.batch_config_cost(n.state.q, focal_points)) > current_best_cost:
            #             num_pts_for_removal += 1

            # for mode, nodes in g.transition_nodes.items():
            #     for n in nodes:
            #         if sum(env.batch_config_cost(n.state.q, focal_points)) > current_best_cost:
            #             num_pts_for_removal += 1
            # Remove elements from g.nodes
            for mode in list(g.nodes.keys()):  # Avoid modifying dict while iterating
                original_count = len(g.nodes[mode])
                g.nodes[mode] = [
                    n
                    for n in g.nodes[mode]
                    if sum(env.batch_config_cost(n.state.q, focal_points))
                    <= current_best_cost
                ]
                num_pts_for_removal += original_count - len(g.nodes[mode])

            # Remove elements from g.transition_nodes
            for mode in list(g.transition_nodes.keys()):
                original_count = len(g.transition_nodes[mode])
                g.transition_nodes[mode] = [
                    n
                    for n in g.transition_nodes[mode]
                    if sum(env.batch_config_cost(n.state.q, focal_points))
                    <= current_best_cost
                ]
                num_pts_for_removal += original_count - len(g.transition_nodes[mode])

            print(f"Removed {num_pts_for_removal} nodes")

        print()
        print(f"Samples: {cnt}; time: {time.time() - start_time:.2f}s; {ptc}")

        samples_in_graph_before = g.get_num_samples()

        if add_new_batch:
            # add new batch of nodes
            effective_uniform_batch_size = (
                uniform_batch_size if current_best_cost is not None else 500
            )
            effective_uniform_transition_batch_size = (
                uniform_transition_batch_size if current_best_cost is not None else 500
            )

            # if env.terminal_mode not in reached_modes:
            print("Sampling transitions")
            new_transitions = sample_valid_uniform_transitions(
                transistion_batch_size=effective_uniform_transition_batch_size,
                cost=current_best_cost,
            )
            g.add_transition_nodes(new_transitions)
            print(f"Adding {len(new_transitions)} transitions")

            print("Sampling uniform")
            new_states, required_attempts_this_batch = sample_valid_uniform_batch(
                batch_size=effective_uniform_batch_size, cost=current_best_cost
            )
            g.add_states(new_states)
            print(f"Adding {len(new_states)} new states")

            # nodes_per_state = []
            # for m in reached_modes:
            #     num_nodes = 0
            #     for n in new_states:
            #         if n.mode == m:
            #             num_nodes += 1

            #     nodes_per_state.append(num_nodes)

            # plt.figure("Uniform states")
            # plt.bar([str(mode) for mode in reached_modes], nodes_per_state)
            # plt.show()

            approximate_space_extent = (
                np.prod(np.diff(env.limits, axis=0))
                * len(new_states)
                / required_attempts_this_batch
            )

            # print(reached_modes)

            if not g.goal_nodes:
                continue

            # g.compute_lower_bound_to_goal(env.batch_config_cost)
            # g.compute_lower_bound_from_start(env.batch_config_cost)

            if current_best_cost is not None and (
                try_informed_sampling or try_informed_transitions
            ):
                interpolated_path = interpolate_path(current_best_path)
                # interpolated_path = current_best_path

                if try_informed_sampling:
                    print("Generating informed samples")
                    new_informed_states = generate_informed_samples(
                        informed_batch_size,
                        interpolated_path,
                        locally_informed_sampling=locally_informed_sampling,
                        try_direct_sampling=try_direct_informed_sampling,
                    )
                    g.add_states(new_informed_states)

                    print(f"Adding {len(new_informed_states)} informed samples")

                if try_informed_transitions:
                    print("Generating informed transitions")
                    new_informed_transitions = generate_informed_transitions(
                        informed_transition_batch_size,
                        interpolated_path,
                        locally_informed_sampling=locally_informed_sampling,
                    )
                    g.add_transition_nodes(new_informed_transitions)
                    print(
                        f"Adding {len(new_informed_transitions)} informed transitions"
                    )

                    # g.compute_lower_bound_to_goal(env.batch_config_cost)
                    # g.compute_lower_bound_from_start(env.batch_config_cost)

            if try_sampling_around_path and current_best_path is not None:
                print("Sampling around path")
                path_samples, path_transitions = sample_around_path(current_best_path)

                g.add_states(path_samples)
                print(f"Adding {len(path_samples)} path samples")

                g.add_transition_nodes(path_transitions)
                print(f"Adding {len(path_transitions)} path transitions")

            g.compute_lower_bound_to_goal(env.batch_config_cost)

        samples_in_graph_after = g.get_num_samples()
        cnt += samples_in_graph_after - samples_in_graph_before

        # search over nodes:
        # 1. search from goal state with sparse check
        reached_terminal_mode = False
        for m in reached_modes:
            if env.is_terminal_mode(m):
                reached_terminal_mode = True

        if not reached_terminal_mode:
            continue

        # for m in reached_modes:
        #     plt.figure()
        #     plt.scatter([a.state.q.state()[0] for a in g.nodes[m]], [a.state.q.state()[1] for a in g.nodes[m]])
        #     plt.scatter([a.state.q.state()[2] for a in g.nodes[m]], [a.state.q.state()[3] for a in g.nodes[m]])
        #     # plt.scatter()

        # plt.show()

        # pts_per_mode = []
        # transitions_per_mode = []
        # for m in reached_modes:
        #     num_transitions = 0
        #     if m in g.transition_nodes:
        #         num_transitions += len(g.transition_nodes[m])

        #     num_pts = 0

        #     if m in g.nodes:
        #         num_pts += len(g.nodes[m])

        #     pts_per_mode.append(num_pts)
        #     transitions_per_mode.append(num_transitions)

        # plt.figure()
        # plt.bar([str(mode) for mode in reached_modes], pts_per_mode)

        # plt.figure()
        # plt.bar([str(mode) for mode in reached_modes], transitions_per_mode)

        # plt.show()

        while True:
            # print([node.neighbors[0].state.mode for node in g.reverse_transition_nodes[g.goal_nodes[0].state.mode]])
            # print([node.neighbors[0].state.mode for node in g.transition_nodes[g.root.state.mode]])

            sparsely_checked_path = g.search(
                g.root,
                g.goal_nodes,
                env,
                current_best_cost,
                resolution,
                approximate_space_extent,
            )

            # sparsely_checked_path = g.search_with_vertex_queue(
            #     g.root, g.goal_nodes, env, current_best_cost, resolution, approximate_space_extent
            # )

            # 2. in case this found a path, search with dense check from the other side
            if sparsely_checked_path:
                add_new_batch = False

                is_valid_path = True
                for i in range(len(sparsely_checked_path) - 1):
                    n0 = sparsely_checked_path[i]
                    n1 = sparsely_checked_path[i + 1]

                    s0 = n0.state
                    s1 = n1.state

                    if n0.id in n1.whitelist:
                        continue

                    if not env.is_edge_collision_free(
                        s0.q, s1.q, s0.mode, resolution=0.001
                    ):
                        print("Path is in collision")
                        is_valid_path = False
                        # env.show(True)
                        n0.blacklist.add(n1.id)
                        n1.blacklist.add(n0.id)
                        break
                    else:
                        n1.whitelist.add(n0.id)
                        n0.whitelist.add(n1.id)

                if is_valid_path:
                    path = [node.state for node in sparsely_checked_path]
                    new_path_cost = path_cost(path, env.batch_config_cost)
                    if current_best_cost is None or new_path_cost < current_best_cost:
                        current_best_path = path
                        current_best_cost = new_path_cost

                        # extract modes
                        modes = [path[0].mode]
                        for p in path:
                            if p.mode != modes[-1]:
                                modes.append(p.mode)

                        print("Modes of new path")
                        print([m.task_ids for m in modes])

                        print(
                            f"New cost: {new_path_cost} at time {time.time() - start_time}"
                        )
                        costs.append(new_path_cost)
                        times.append(time.time() - start_time)

                        all_paths.append(path)

                        if try_shortcutting:
                            print("Shortcutting path")
                            shortcut_path, _ = shortcutting.robot_mode_shortcut(
                                env,
                                path,
                                250,
                                resolution=env.collision_resolution,
                                tolerance=env.collision_tolerance,
                            )

                            shortcut_path = shortcutting.remove_interpolated_nodes(
                                shortcut_path
                            )

                            shortcut_path_cost = path_cost(
                                shortcut_path, env.batch_config_cost
                            )

                            if shortcut_path_cost < current_best_cost:
                                print("New cost: ", shortcut_path_cost)
                                costs.append(shortcut_path_cost)
                                times.append(time.time() - start_time)

                                all_paths.append(shortcut_path)

                                current_best_path = shortcut_path
                                current_best_cost = shortcut_path_cost

                                interpolated_path = shortcut_path

                                for i in range(len(interpolated_path)):
                                    s = interpolated_path[i]
                                    if not env.is_collision_free(s.q, s.mode):
                                        continue

                                    if (
                                        i < len(interpolated_path) - 1
                                        and interpolated_path[i].mode
                                        != interpolated_path[i + 1].mode
                                    ):
                                        # add as transition
                                        g.add_transition_nodes(
                                            [
                                                (
                                                    s.q,
                                                    s.mode,
                                                    interpolated_path[i + 1].mode,
                                                )
                                            ]
                                        )
                                        pass
                                    else:
                                        g.add_states([s])

                    add_new_batch = True

                    # plt.figure()

                    # plt.plot([pt.q.state()[0] for pt in current_best_path], [pt.q.state()[1] for pt in current_best_path], 'o-')
                    # plt.plot([pt.q.state()[2] for pt in current_best_path], [pt.q.state()[3] for pt in current_best_path], 'o-')

                    # plt.show()

                    break

            else:
                print("Did not find a solution")
                add_new_batch = True
                break

        if not optimize and current_best_cost is not None:
            break

        if ptc.should_terminate(cnt, time.time() - start_time):
            break

    costs.append(costs[-1])
    times.append(time.time() - start_time)

    info = {"costs": costs, "times": times, "paths": all_paths}
    return current_best_path, info
