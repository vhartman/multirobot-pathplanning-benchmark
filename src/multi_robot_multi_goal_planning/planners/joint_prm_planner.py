import numpy as np
import random

from matplotlib import pyplot as plt

from typing import List, Dict, Tuple, Optional
from numpy.typing import NDArray
import heapq
# import _heapq as heapq

import time
import math

from multi_robot_multi_goal_planning.problems.planning_env import State, BaseProblem
from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
    NpConfiguration,
    config_dist,
    batch_config_dist,
)
from multi_robot_multi_goal_planning.problems.util import path_cost, interpolate_path

from multi_robot_multi_goal_planning.planners import shortcutting


def edge_tuple(n0, n1):
    if n0.id < n1.id:
        return (n0, n1)
    else:
        return (n1, n0)


class Node:
    # __slots__ = 'id'

    state: State
    id_counter = 0

    def __init__(self, state, is_transition=False):
        self.state = state
        self.lb_cost_to_goal = None
        self.lb_cost_from_start = None

        self.is_transition = is_transition

        self.neighbors = []

        self.whitelist = set()
        self.blacklist = set()

        self.id = Node.id_counter
        Node.id_counter += 1

    def __lt__(self, other):
        return self.id < other.id

    def __hash__(self):
        return self.id


class HeapQueue:
    def __init__(self):
        self.queue = []
        heapq.heapify(self.queue)

    def __len__(self):
        return len(self.queue)

    def heappush(self, item):
        heapq.heappush(self.queue, item)

    def heappop(self):
        return heapq.heappop(self.queue)

    def remove(self, node):
        # (cost, edge_cost, e)
        pass


from sortedcontainers import SortedList


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


from collections import defaultdict


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
    def __init__(self):
        self.queue = []
        self.items = []
        heapq.heapify(self.queue)

    def __len__(self):
        return len(self.queue)

    def heappush(self, item):
        idx = len(self.items)
        self.items.append(item)

        heapq.heappush(self.queue, (item[0], idx))

    def heappop(self):
        _, idx = heapq.heappop(self.queue)
        return self.items[idx]


class BucketIndexHeap:
    def __init__(self, granularity=100):
        self.granularity = granularity

        self.queues = {}
        self.priority_lookup = []

        self.items = []

        self.len = 0

    def __len__(self):
        return self.len

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def heappush(self, item):
        self.len += 1
        priority = int(item[0] * self.granularity)

        idx = len(self.items)
        self.items.append(item)

        if priority not in self.queues:
            self.queues[priority] = []
            heapq.heappush(self.priority_lookup, priority)

        heapq.heappush(self.queues[priority], (item[0], idx))

    def heappop(self):
        self.len -= 1
        min_priority = self.priority_lookup[0]
        _, idx = heapq.heappop(self.queues[min_priority])

        if not self.queues[min_priority]:
            del self.queues[min_priority]
            heapq.heappop(self.priority_lookup)

        value = self.items[idx]

        return value


class DiscreteBucketIndexHeap:
    def __init__(self, granularity=1000):
        self.granularity = granularity

        self.queues = {}
        self.priority_lookup = []

        self.items = []

        self.len = 0

    def __len__(self):
        # num_elements = 0
        # for k, v in self.queues.items():
        #     num_elements += len(v)

        # return num_elements
        return self.len

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def heappush(self, item):
        self.len += 1
        priority = int(item[0] * self.granularity)

        idx = len(self.items)
        self.items.append(item)

        if priority not in self.queues:
            self.queues[priority] = []
            heapq.heappush(self.priority_lookup, priority)

        self.queues[priority].append((item[0], idx))

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def heappop(self):
        self.len -= 1

        min_priority = self.priority_lookup[0]
        _, idx = self.queues[min_priority].pop()

        if not self.queues[min_priority]:
            del self.queues[min_priority]
            heapq.heappop(self.priority_lookup)

        value = self.items[idx]

        return value


class Graph:
    root: State
    nodes: Dict

    # batch_dist_fun

    def __init__(self, start: State, batch_dist_fun, use_k_nearest=True):
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

    def get_num_samples(self):
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

        while len(queue) > 0:
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

            if len(neighbors) == 0:
                continue

            if node.state.mode not in self.reverse_transition_node_array_cache:
                self.reverse_transition_node_array_cache[node.state.mode] = np.array(
                    [n.state.q.state() for n in neighbors]
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

            neighbors = [n.neighbors[0] for n in self.transition_nodes[node.state.mode]]

            if len(neighbors) == 0:
                continue

            if node.state.mode not in self.transition_node_array_cache:
                self.transition_node_array_cache[node.state.mode] = np.array(
                    [n.state.q.state() for n in neighbors]
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

    def add_states(self, states):
        for s in states:
            self.add_node(Node(s))

    def add_nodes(self, nodes):
        for n in nodes:
            self.add_node(n)

    def add_transition_nodes(self, transitions):
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
    def get_neighbors(self, node, k=20, space_extent=None):
        key = node.state.mode
        if key in self.nodes:
            node_list = self.nodes[key]

            if key not in self.node_array_cache:
                self.node_array_cache[key] = np.array([n.state.q.q for n in node_list])

            dists = self.batch_dist_fun(
                node.state.q, self.node_array_cache[key]
            )  # this, and the list copm below are the slowest parts

        if key in self.transition_nodes:
            transition_node_list = self.transition_nodes[key]

            if key not in self.transition_node_array_cache:
                self.transition_node_array_cache[key] = np.array(
                    [n.state.q.q for n in transition_node_list]
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
            r = 3

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
                r = r_star

                best_nodes = [node_list[i] for i in np.where(dists < r)[0]]
                best_nodes_arr = self.node_array_cache[key][np.where(dists < r)[0], :]

                # print("fraction of nodes in mode", len(best_nodes)/len(dists))

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
                r = r_star

                if len(transition_node_list) == 1:
                    r = 1e6

                best_transition_nodes = [
                    transition_node_list[i] for i in np.where(transition_dists < r)[0]
                ]
                best_transitions_arr = self.transition_node_array_cache[key][
                    np.where(transition_dists < r)[0]
                ]

            best_nodes = best_nodes + best_transition_nodes

        arr = np.vstack([best_nodes_arr, best_transitions_arr])

        if node.is_transition:
            tmp = np.vstack([n.state.q.state() for n in node.neighbors])
            arr = np.vstack([arr, tmp])
            return best_nodes + node.neighbors, arr

        return best_nodes, arr

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def search(
        self,
        start_node,
        goal_nodes: List,
        env: BaseProblem,
        best_cost=None,
        resolution=0.1,
    ):
        open_queue = []
        closed_list = set()

        goal = None

        h_cache = {}

        # TODO: decent heuristic makes everything better but is computationally not amazing
        def h(node):
            # return 0
            if node in h_cache:
                return h_cache[node]

            if node.state.mode not in self.transition_node_array_cache:
                self.transition_node_array_cache[node.state.mode] = np.array(
                    [o.state.q.q for o in self.transition_nodes[node.state.mode]]
                )

            if node.state.mode not in self.transition_node_lb_cache:
                self.transition_node_lb_cache[node.state.mode] = np.array(
                    [o.lb_cost_to_goal for o in self.transition_nodes[node.state.mode]]
                )

            costs_to_transitions = env.batch_config_cost(
                node.state.q,
                self.transition_node_array_cache[node.state.mode],
            )

            min_cost = np.min(
                self.transition_node_lb_cache[node.state.mode] + costs_to_transitions
            )

            h_cache[node] = min_cost
            return min_cost

        def d(n0, n1):
            # return 1.0
            cost = env.config_cost(n0.state.q, n1.state.q)
            return cost

        # reached_modes = []

        parents = {start_node: None}
        gs = {start_node.id: 0}  # best cost to get to a node

        start_neighbors, _ = self.get_neighbors(
            start_node, space_extent=np.prod(np.diff(env.limits, axis=0))
        )

        # populate open_queue and fs
        start_edges = [(start_node, n) for n in start_neighbors]
        # start_edges = [
        #     (start_node, n)
        #     for n, _ in self.get_neighbors(
        #         start_node, space_extent=np.prod(np.diff(env.limits, axis=0))
        #     )
        # ]

        # queue = HeapQueue()
        # queue = BucketHeapQueue()
        queue = BucketIndexHeap()
        # queue = DiscreteBucketIndexHeap()
        # queue = IndexHeap()
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

        num_iter = 0
        while len(queue) > 0:
            # while len(open_queue) > 0:
            num_iter += 1

            # print(len(queue))

            if num_iter % 100000 == 0:
                # print(len(open_queue))
                print(len(queue))

            # f_pred, edge_cost, (n0, n1) = heapq.heappop(open_queue)
            f_pred, edge_cost, (n0, n1) = queue.heappop()
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
            # et = edge_tuple(n0, n1)

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

            # remove all other edges with this goal from the queue

            # if n0.state.mode not in reached_modes:
            #     reached_modes.append(n0.state.mode)

            # print('reached modes', reached_modes)

            # print('adding', n1.id)
            g_tentative = gs[n0.id] + edge_cost
            gs[n1.id] = g_tentative
            parents[n1] = n0

            # queue.remove_by_node(n1)

            if n1 in goal_nodes:
                goal = n1
                break

            # get_neighbors
            neighbors, tmp = self.get_neighbors(
                n1, space_extent=np.prod(np.diff(env.limits, axis=0))
            )

            if len(neighbors) == 0:
                continue

            # add neighbors to open_queue
            # edge_costs = env.batch_config_cost(
            #     n1.state.q, np.array([n.state.q.state() for n in neighbors])
            # )
            edge_costs = env.batch_config_cost(n1.state.q, tmp)
            for n, edge_cost in zip(neighbors, edge_costs):
                if n == n0:
                    continue

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
                    queue.heappush((f_node, edge_cost, (n1, n)))

                    # open_queue.append((f_node, edge_cost, (n1, n)))
            # heapq.heapify(open_queue)
            # open_queue.sort(reverse=True)

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

        return path

    def search_with_vertex_queue(
        self,
        start_node,
        goal_nodes: List,
        env: BaseProblem,
        best_cost=None,
        resolution=0.1,
    ):
        open_queue = []

        goal = None

        h_cache = {}

        def h(node):
            # return 0
            if node.id in h_cache:
                return h_cache[node.id]

            if node.state.mode not in self.transition_node_array_cache:
                self.transition_node_array_cache[node.state.mode] = np.array(
                    [n.state.q.q for n in self.transition_nodes[node.state.mode]]
                )

            if node.state.mode not in self.transition_node_lb_cache:
                self.transition_node_lb_cache[node.state.mode] = np.array(
                    [n.lb_cost_to_goal for n in self.transition_nodes[node.state.mode]]
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
        while len(open_queue) > 0:
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
            neighbors = self.get_neighbors(
                node, space_extent=np.prod(np.diff(env.limits, axis=0))
            )

            edge_costs = env.batch_config_cost(
                node.state.q, np.array([n.state.q.state() for n in neighbors])
            )
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
                            node.state.q, n.state.q, n.state.mode
                        )

                        if not collision_free:
                            node.blacklist.add(n.id)
                            n.blacklist.add(node.id)
                            continue
                        else:
                            node.whitelist.add(n.id)
                            n.whitelist.add(node.id)

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


def joint_prm_planner(
    env: BaseProblem,
    optimize: bool = True,
    mode_sampling_type: str = "greedy",
    max_iter: int = 2000,
    distance_metric="euclidean",
    try_sampling_around_path=True,
    use_k_nearest=True,
    try_informed_sampling=True,
    uniform_batch_size=200,
    uniform_transition_batch_size=500,
    informed_batch_size=1000,
    informed_transition_batch_size=1000,
    path_batch_size=500,
    locally_informed_sampling=False,
    try_informed_transitions=True,
    try_shortcutting=True,
) -> Optional[Tuple[List[State], List]]:
    q0 = env.get_start_pos()
    m0 = env.get_start_mode()

    reached_modes = [m0]

    conf_type = type(env.get_start_pos())

    def sample_mode(mode_sampling_type="weighted", found_solution=False):
        return random.choice(reached_modes)
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
    def can_improve(rnd_state, path, start_index, end_index):
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
                    [o.state.q.q for o in g.reverse_transition_nodes[state.mode]]
                )

            if state.mode not in g.rev_transition_node_lb_cache:
                g.rev_transition_node_lb_cache[state.mode] = np.array(
                    [
                        o.lb_cost_from_start
                        for o in g.reverse_transition_nodes[state.mode]
                    ]
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
                    [o.state.q.q for o in g.transition_nodes[state.mode]]
                )

            if state.mode not in g.transition_node_lb_cache:
                g.transition_node_lb_cache[state.mode] = np.array(
                    [o.lb_cost_to_goal for o in g.transition_nodes[state.mode]]
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
        if path[start_index].mode != rnd_state.mode:
            start_state = path[start_index]
            lb_cost_from_start_to_state = lb_cost_from_start(rnd_state)
            lb_cost_from_start_to_index = lb_cost_from_start(start_state)

            lb_cost_from_start_index_to_state = max(
                (lb_cost_from_start_to_state - lb_cost_from_start_to_index),
                lb_cost_from_start_index_to_state,
            )

        lb_cost_from_state_to_end_index = env.config_cost(
            rnd_state.q, path[end_index].q
        )
        if path[end_index].mode != rnd_state.mode:
            goal_state = path[end_index]
            lb_cost_from_goal_to_state = lb_cost_from_goal(rnd_state)
            lb_cost_from_goal_to_index = lb_cost_from_goal(goal_state)

            lb_cost_from_state_to_end_index = max(
                (lb_cost_from_goal_to_state - lb_cost_from_goal_to_index),
                lb_cost_from_state_to_end_index,
            )

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

    # taken from https://github.com/marleyshan21/Batch-informed-trees/blob/master/python/BIT_Star.py
    def sample_unit_ball(dim) -> np.array:
        """Samples a point uniformly from the unit ball. This is used to sample points from the Prolate HyperSpheroid (PHS).

        Returns:
            Sampled Point (np.array): The sampled point from the unit ball.
        """
        u = np.random.uniform(-1, 1, dim)
        norm = np.linalg.norm(u)
        r = np.random.random() ** (1.0 / dim)
        return r * u / norm

    def samplePHS(a, b, c) -> np.array:
        """Samples a point from the Prolate HyperSpheroid (PHS) defined by the start and goal nodes.

        Returns:
            Node: The sampled node from the PHS.
        """
        dim = len(a)
        # Calculate the center of the PHS.
        center = (a + b) / 2
        # The transverse axis in the world frame.
        c_min = np.linalg.norm(a - b)
        a1 = (b - a) / c_min
        # The first column of the identity matrix.
        one_1 = np.eye(a1.shape[0])[:, 0]
        U, S, Vt = np.linalg.svd(np.outer(a1, one_1.T))
        Sigma = np.diag(S)
        lam = np.eye(Sigma.shape[0])
        lam[-1, -1] = np.linalg.det(U) * np.linalg.det(Vt.T)
        # Calculate the rotation matrix.
        cwe = np.matmul(U, np.matmul(lam, Vt))
        # Get the radius of the first axis of the PHS.
        r1 = c / 2
        # Get the radius of the other axes of the PHS.
        rn = [np.sqrt(c**2 - c_min**2) / 2] * (dim - 1)
        # Create a vector of the radii of the PHS.
        r = np.diag(np.array([r1] + rn))

        # Sample a point from the PHS.
        # Sample a point from the unit ball.
        x_ball = sample_unit_ball(dim)
        # Transform the point from the unit ball to the PHS.
        op = np.matmul(np.matmul(cwe, r), x_ball) + center
        # Round the point to 7 decimal places.
        op = np.around(op, 7)
        # Check if the point is in the PHS.

        return op

    def rejection_sample_from_ellipsoid(a, b, c):
        m = (a + b) / 2
        n = len(a)
        cnt = 0
        while True:
            cnt += 1
            x = np.random.uniform(m - c / 2, m + c / 2, n)
            if np.linalg.norm(x - a) + np.linalg.norm(x - b) < c:
                return x

    def generate_informed_samples(
        batch_size, path, max_attempts_per_sample=200, locally_informed_sampling=True
    ):
        new_samples = []
        path_segment_costs = env.batch_config_cost(path[:-1], path[1:])

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

                k = random.randint(start_ind, end_ind)
                m = path[k].mode
            else:
                start_ind = 0
                end_ind = len(path) - 1
                m = sample_mode("weighted", True)

            # print(m)

            current_cost = sum(path_segment_costs[start_ind:end_ind])

            # tmp = 0
            # for i in range(start_ind, end_ind):
            #     tmp += env.config_cost(path[i].q, path[i+1].q)

            # print(current_cost, tmp)

            # plt.figure()
            # samples = []
            # for _ in range(200):
            #     sample = samplePHS(np.array([-1, 0]), np.array([1, 0]), 3)
            #     # sample = samplePHS(np.array([[-1], [0]]), np.array([[1], [0]]), 3)
            #     samples.append(sample)
            #     print("sample", sample)

            # plt.scatter([a[0] for a in samples], [a[1] for a in samples])
            # plt.show()

            focal_points = np.array(
                [path[start_ind].q.state(), path[end_ind].q.state()]
            )
            try_direct_sampling = False
            for k in range(max_attempts_per_sample):
                if not try_direct_sampling or env.cost_metric != "euclidean":
                    # completely random sample configuration from the (valid) domain robot by robot
                    q = []
                    for i in range(len(env.robots)):
                        r = env.robots[i]
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
                else:
                    # sample by sampling each agent separately
                    q = []
                    for i in range(len(env.robots)):
                        r = env.robots[i]
                        lims = env.limits[:, env.robot_idx[r]]
                        if lims[0, 0] < lims[1, 0]:
                            if env.cost_reduction == "sum":
                                c_robot_bound = current_cost - sum(
                                    [
                                        np.linalg.norm(
                                            path[start_ind].q[j] - path[end_ind].q[j]
                                        )
                                        for j in range(len(env.robots))
                                        if j != i
                                    ]
                                )
                            else:
                                c_robot_bound = current_cost

                            if np.linalg.norm(
                                    path[start_ind].q[i] - path[end_ind].q[i]
                                ) < 1e-3:
                                qr = (
                                    np.random.rand(env.robot_dims[r])
                                    * (lims[1, :] - lims[0, :])
                                    + lims[0, :]
                                )
                            else:

                                # print("cost", current_cost)
                                # print("robot cst", c_robot_bound)
                                # print(
                                #     np.linalg.norm(
                                #         path[start_ind].q[i] - path[end_ind].q[i]
                                #     )
                                # )

                                qr = samplePHS(path[start_ind].q[i], path[end_ind].q[i], c_robot_bound)
                                # qr = rejection_sample_from_ellipsoid(
                                #     path[start_ind].q[i], path[end_ind].q[i], c_robot_bound
                                # )

                                # if np.linalg.norm(path[start_ind].q[i] - qr) + np.linalg.norm(path[end_ind].q[i] - qr) > c_robot_bound:
                                #     print("AAAAAAAAA")
                                #     print(np.linalg.norm(path[start_ind].q[i] - qr) + np.linalg.norm(path[end_ind].q[i] - qr), c_robot_bound)

                                qr = np.clip(qr, lims[0, :], lims[1, :])

                        q.append(qr)

                q = conf_type.from_list(q)

                if sum(env.batch_config_cost(q, focal_points)) > current_cost:
                    # print(path[start_ind].mode, path[end_ind].mode, m)
                    # print(
                    #     current_cost,
                    #     env.config_cost(path[start_ind].q, q)
                    #     + env.config_cost(path[end_ind].q, q),
                    # )
                    # if can_improve(State(q, m), path, start_ind, end_ind):
                    #     assert False

                    continue

                # if can_improve(State(q, m), path, 0, len(path)-1):
                # if can_improve(State(q, m), path, start_ind, end_ind):
                if can_improve(
                    State(q, m), path, start_ind, end_ind
                ) and env.is_collision_free(q, m):
                    # if env.is_collision_free(q, m) and can_improve(State(q, m), path, 0, len(path)-1):
                    new_samples.append(State(q, m))
                    break

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
                    [o.state.q.q for o in g.reverse_transition_nodes[state.mode]]
                )

            if state.mode not in g.rev_transition_node_lb_cache:
                g.rev_transition_node_lb_cache[state.mode] = np.array(
                    [
                        o.lb_cost_from_start
                        for o in g.reverse_transition_nodes[state.mode]
                    ]
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
                    [o.state.q.q for o in g.transition_nodes[state.mode]]
                )

            if state.mode not in g.transition_node_lb_cache:
                g.transition_node_lb_cache[state.mode] = np.array(
                    [o.lb_cost_to_goal for o in g.transition_nodes[state.mode]]
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
        if path[start_index].mode != rnd_state_mode_1.mode:
            start_state = path[start_index]
            lb_cost_from_start_to_state = lb_cost_from_start(rnd_state_mode_1)
            lb_cost_from_start_to_index = lb_cost_from_start(start_state)

            lb_cost_from_start_index_to_state = max(
                (lb_cost_from_start_to_state - lb_cost_from_start_to_index),
                lb_cost_from_start_index_to_state,
            )

        lb_cost_from_state_to_end_index = env.config_cost(
            rnd_state_mode_2.q, path[end_index].q
        )
        if path[end_index].mode != rnd_state_mode_2.mode:
            goal_state = path[end_index]
            lb_cost_from_goal_to_state = lb_cost_from_goal(rnd_state_mode_2)
            lb_cost_from_goal_to_index = lb_cost_from_goal(goal_state)

            lb_cost_from_state_to_end_index = max(
                (lb_cost_from_goal_to_state - lb_cost_from_goal_to_index),
                lb_cost_from_state_to_end_index,
            )

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
        batch_size, path, naive_informed_sampling=False, max_attempts_per_sample=100
    ):
        new_transitions = []
        num_attempts = 0
        path_segment_costs = env.batch_config_cost(path[:-1], path[1:])

        while len(new_transitions) < batch_size:
            num_attempts += 1

            if num_attempts > batch_size:
                break

            # print(len(new_samples))
            # sample mode
            if naive_informed_sampling:
                start_ind = 0
                end_ind = len(path) - 1
                mode = sample_mode("weighted", True)
            else:
                while True:
                    start_ind = random.randint(0, len(path) - 1)
                    end_ind = random.randint(0, len(path) - 1)

                    if (
                        path[end_ind].mode != path[start_ind].mode
                        and end_ind - start_ind > 2  # and end_ind - start_ind < 50
                    ):
                        # if end_ind - start_ind > 2 and end_ind - start_ind < 50:
                        current_cost = sum(path_segment_costs[start_ind:end_ind])
                        lb_cost = env.config_cost(path[start_ind].q, path[end_ind].q)

                        if lb_cost < current_cost:
                            break

                k = random.randint(start_ind, end_ind)
                mode = path[k].mode

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

                q = conf_type.from_list(q)

                if (
                    env.config_cost(path[start_ind].q, q)
                    + env.config_cost(path[end_ind].q, q)
                    > current_cost
                ):
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

    def sample_valid_uniform_batch(batch_size, cost):
        new_samples = []
        num_attempts = 0
        num_valid = 0
        if True:
            while len(new_samples) < batch_size:
                num_attempts += 1
                # print(len(new_samples))
                # sample mode
                m = sample_mode("weighted", cost is not None)

                # print(m)

                # sample configuration
                q = []
                for i in range(len(env.robots)):
                    r = env.robots[i]
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

                q = conf_type.from_list(q)

                if env.is_collision_free(q, m):
                    new_samples.append(State(q, m))
                    num_valid += 1

            print(num_valid / num_attempts)
        elif True:
            # grid sampling

            m = sample_mode("weighted", cost is not None)
            q = []
            for i in range(len(env.robots)):
                r = env.robots[i]
                lims = env.limits[:, env.robot_idx[r]]
                if lims[0, 0] < lims[1, 0]:
                    qr = (
                        np.random.rand(env.robot_dims[r]) * (lims[1, :] - lims[0, :])
                        + lims[0, :]
                    )
                else:
                    qr = np.random.rand(env.robot_dims[r]) * 6 - 3

                q.append(qr)

            q = conf_type.from_list(q)

            if env.is_collision_free(q, m):
                new_samples.append(State(q, m))
        else:
            # we found a solution, and can do informed sampling:
            # cost = (cost to get to mode) + cost in mode + (cost_to_goal)
            # maybe better formulation:
            # cost =

            # NOTES:
            # minimum cost to reach a mode is the lower bound of all the mode transitions we have so far
            # admissible possibilities to compute the minimum cost:
            # - lb through each mode, sum them up?
            # -- disregards continuity, might underestimate massively
            # - use dependency graph and compute lb per task?
            #####
            # inadmissible possibilities:
            # - take only cost in current mode
            pass

        return new_samples

    def sample_valid_uniform_transitions(transistion_batch_size):
        transitions = []

        while len(transitions) < transistion_batch_size:
            # sample mode
            mode = sample_mode("uniform_reached", None)

            # sample transition at the end of this mode
            possible_next_task_combinations = env.get_valid_next_task_combinations(mode)
            if len(possible_next_task_combinations) > 0:
                ind = random.randint(0, len(possible_next_task_combinations) - 1)
                active_task = env.get_active_task(
                    mode, possible_next_task_combinations[ind]
                )
            else:
                active_task = env.get_active_task(mode, None)

            goals_to_sample = active_task.robots

            goal_sample = active_task.goal.sample(mode)

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

            q = conf_type.from_list(q)

            if env.is_collision_free(q, mode):
                if env.is_terminal_mode(mode):
                    next_mode = None
                else:
                    next_mode = env.get_next_mode(q, mode)

                transitions.append((q, mode, next_mode))

                if next_mode not in reached_modes and next_mode is not None:
                    reached_modes.append(next_mode)

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

    resolution = 0.02

    all_paths = []

    cnt = 0
    while True:
        cnt = g.get_num_samples()

        print()
        print("Count:", cnt, "max_iter:", max_iter)

        if add_new_batch:
            if try_sampling_around_path and current_best_path is not None:
                # sample index
                interpolated_path = interpolate_path(current_best_path)
                # interpolated_path = current_best_path
                new_states_from_path_sampling = []
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
                        current_task_ids = state.mode.task_ids
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
                            g.add_transition_nodes(
                                [(q, state.mode, interpolated_path[idx + 1].mode)]
                            )

                    else:
                        for i in range(len(env.robots)):
                            r = env.robots[i]
                            qr_mean = state.q[i]

                            qr = np.random.rand(len(qr_mean)) * 0.5 + qr_mean

                            lims = env.limits[:, env.robot_idx[r]]
                            if lims[0, 0] < lims[1, 0]:
                                qr = np.clip(qr, lims[0, :], lims[1, :])
                                # qr = (
                                #     np.random.rand(env.robot_dims[r]) * (lims[1, :] - lims[0, :])
                                #     + lims[0, :]
                                # )

                            q.append(qr)

                        q = conf_type.from_list(q)

                        if env.is_collision_free(q, state.mode):
                            rnd_state = State(q, state.mode)
                            new_states_from_path_sampling.append(rnd_state)

                            g.add_states([rnd_state])

                # fig = plt.figure()
                # ax = fig.add_subplot(projection='3d')
                # ax.scatter([a.q[0][0] for a in new_states_from_path_sampling], [a.q[0][1] for a in new_states_from_path_sampling], [a.q[0][2] for a in new_states_from_path_sampling])
                # ax.scatter([a.q[1][0] for a in new_states_from_path_sampling], [a.q[1][1] for a in new_states_from_path_sampling], [a.q[1][2] for a in new_states_from_path_sampling])
                # ax.scatter([a.q[2][0] for a in new_states_from_path_sampling], [a.q[2][1] for a in new_states_from_path_sampling], [a.q[1][2] for a in new_states_from_path_sampling])
                # plt.show()

            # add new batch of nodes
            effective_uniform_batch_size = (
                uniform_batch_size if current_best_cost is not None else 500
            )
            effective_uniform_transition_batch_size = (
                uniform_transition_batch_size if current_best_cost is not None else 500
            )

            print("Sampling uniform")
            new_states = sample_valid_uniform_batch(
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

            # if env.terminal_mode not in reached_modes:
            print("Sampling transitions")
            new_transitions = sample_valid_uniform_transitions(
                transistion_batch_size=effective_uniform_transition_batch_size
            )
            g.add_transition_nodes(new_transitions)
            print(f"Adding {len(new_transitions)} transitions")

            g.compute_lower_bound_to_goal(env.batch_config_cost)
            g.compute_lower_bound_from_start(env.batch_config_cost)

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
                    )
                    g.add_states(new_informed_states)

                    # nodes_per_state = []
                    # for m in reached_modes:
                    #     num_nodes = 0
                    #     for n in new_informed_states:
                    #         if n.mode == m:
                    #             num_nodes += 1

                    #     nodes_per_state.append(num_nodes)

                    # plt.figure("Informed states")
                    # plt.bar([str(mode) for mode in reached_modes], nodes_per_state)

                    print(f"Adding {len(new_informed_states)} informed samples")

                if try_informed_transitions:
                    print("Generating informed transitions")
                    new_informed_transitions = generate_informed_transitions(
                        informed_transition_batch_size, interpolated_path
                    )
                    g.add_transition_nodes(new_informed_transitions)
                    print(
                        f"Adding {len(new_informed_transitions)} informed transitions"
                    )

                    g.compute_lower_bound_to_goal(env.batch_config_cost)
                    g.compute_lower_bound_from_start(env.batch_config_cost)

        # search over nodes:
        # 1. search from goal state with sparse check
        reached_terminal_mode = False
        for m in reached_modes:
            if env.is_terminal_mode(m):
                reached_terminal_mode = True

        if not reached_terminal_mode:
            continue

        # print(reached_modes)

        # for m in reached_modes:
        #     plt.figure()
        #     plt.scatter([a.state.q.state()[0] for a in g.nodes[m]], [a.state.q.state()[1] for a in g.nodes[m]])
        #     plt.scatter([a.state.q.state()[2] for a in g.nodes[m]], [a.state.q.state()[3] for a in g.nodes[m]])
        #     # plt.scatter()

        # plt.show()

        pts_per_mode = []
        transitions_per_mode = []
        for m in reached_modes:
            num_transitions = 0
            if m in g.transition_nodes:
                num_transitions += len(g.transition_nodes[m])

            num_pts = 0

            if m in g.nodes:
                num_pts += len(g.nodes[m])

            pts_per_mode.append(num_pts)
            transitions_per_mode.append(num_transitions)

        # plt.figure()
        # plt.bar([str(mode) for mode in reached_modes], pts_per_mode)

        # plt.figure()
        # plt.bar([str(mode) for mode in reached_modes], transitions_per_mode)

        # plt.show()

        while True:
            sparsely_checked_path = g.search(
                g.root, g.goal_nodes, env, current_best_cost, resolution
            )
            # sparsely_checked_path = g.search_with_vertex_queue(
            #     g.root, g.goal_nodes, env, current_best_cost, resolution
            # )

            # 2. in case this found a path, search with dense check from the other side
            if len(sparsely_checked_path) > 0:
                add_new_batch = False

                is_valid_path = True
                for i in range(len(sparsely_checked_path) - 1):
                    n0 = sparsely_checked_path[i]
                    n1 = sparsely_checked_path[i + 1]

                    s0 = n0.state
                    s1 = n1.state

                    # this is a transition, we do not need to collision check this
                    # if s0.mode != s1.mode:
                    #     continue

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

                        print("New cost: ", new_path_cost)
                        costs.append(new_path_cost)
                        times.append(time.time() - start_time)

                        all_paths.append(path)

                        if try_shortcutting:
                            print("Shortcutting path")
                            shortcut_path, _ = shortcutting.robot_mode_shortcut(
                                env, path, 500
                            )
                            # path = [node.state for node in sparsely_checked_path]

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

                                # interpolated_path = interpolate_path(shortcut_path)
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

        if cnt >= max_iter:
            break

    costs.append(costs[-1])
    times.append(time.time() - start_time)

    info = {"costs": costs, "times": times, "paths": all_paths}
    return current_best_path, info
