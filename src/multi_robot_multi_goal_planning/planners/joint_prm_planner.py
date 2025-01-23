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

from concurrent.futures import ThreadPoolExecutor


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

            neighbors = [
                n.neighbors[0] for n in self.reverse_transition_nodes[node.state.mode]
            ]

            if len(neighbors) == 0:
                continue

            if node.state.mode not in self.reverse_transition_node_array_cache:
                self.reverse_transition_node_array_cache[node.state.mode] = np.array(
                    [n.state.q.state() for n in neighbors]
                )

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

        queue = []
        heapq.heappush(queue, (0, self.root))
        costs[self.root.id] = 0

        while len(queue) > 0:
            # node = queue.pop(0)
            _, node = heapq.heappop(queue)
            # print(node)

            if node.state.mode.task_ids == self.goal_nodes[0].state.mode.task_ids:
                continue

            neighbors = [n.neighbors[0] for n in self.transition_nodes[node.state.mode]]

            if len(neighbors) == 0:
                continue

            if node.state.mode not in self.transition_node_array_cache:
                self.transition_node_array_cache[node.state.mode] = np.array(
                    [n.state.q.state() for n in neighbors]
                )

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

            best_transition_nodes = []
            if key in self.transition_nodes:
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

            if n1 in goal_nodes:
                goal = n1
                break

            # get_neighbors
            neighbors, tmp = self.get_neighbors(
                n1, space_extent=np.prod(np.diff(env.limits, axis=0))
            )

            if len(neighbors) != 0:
                # add neighbors to open_queue
                # edge_costs = env.batch_config_cost(
                #     n1.state.q, np.array([n.state.q.state() for n in neighbors])
                # )
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
    try_informed_sampling=True
) -> Optional[Tuple[List[State], List]]:
    q0 = env.get_start_pos()
    m0 = env.get_start_mode()

    reached_modes = [m0]

    conf_type = type(env.get_start_pos())

    def sample_mode(mode_sampling_type="weighted", found_solution=False):
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

        start_state = path[start_index]
        lb_cost_from_start_to_state = lb_cost_from_start(rnd_state)
        lb_cost_from_start_to_index = lb_cost_from_start(start_state)

        goal_state = path[end_index]
        lb_cost_from_goal_to_state = lb_cost_from_goal(rnd_state)
        lb_cost_from_goal_to_index = lb_cost_from_goal(goal_state)

        if path[start_index].mode == rnd_state.mode:
            lb_cost_from_start_index_to_state = env.config_cost(rnd_state.q, path[start_index].q)
        else:
            lb_cost_from_start_index_to_state = (
                lb_cost_from_start_to_state - lb_cost_from_start_to_index
            )

        if path[end_index].mode == rnd_state.mode:
            lb_cost_from_state_to_end_index = env.config_cost(rnd_state.q, path[end_index].q)
        else:
            lb_cost_from_state_to_end_index = (
                lb_cost_from_goal_to_state - lb_cost_from_goal_to_index
            )

        # print('start', lb_cost_from_start_index_to_state)
        # print('end', lb_cost_from_state_to_end_index)

        # print('start index', start_index)
        # print('end_index', end_index)

        # assert(lb_cost_from_start_index_to_state >= 0)
        # assert(lb_cost_from_state_to_end_index >= 0)

        # print(path_cost_from_index_to_index)
        # print(lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index)

        if (
            path_cost_from_index_to_index
            > lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index
        ):
            return True

        return False

    def generate_informed_samples(batch_size, path):
        new_samples = []

        num_attempts = 0

        path_segment_costs = env.batch_config_cost(path[:-1], path[1:])

        while len(new_samples) < batch_size:
            num_attempts += 1
            # print(len(new_samples))
            # sample mode
            if True:
                while True:
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
                end_ind = len(path)-1
                m = sample_mode("weighted", True)

            # print(m)

            for _ in range(100):
                # sample configuration
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

                # if can_improve(State(q, m), path, 0, len(path)-1):
                # if can_improve(State(q, m), path, start_ind, end_ind):
                if can_improve(
                    State(q, m), path, start_ind, end_ind
                ) and env.is_collision_free(q, m):
                    # if env.is_collision_free(q, m) and can_improve(State(q, m), path, 0, len(path)-1):
                    new_samples.append(State(q, m))
                    break

            if num_attempts > batch_size:
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

    def generate_somewhat_informed_samples(batch_size, path):
        def cost_to_mode_on_path(mode, path):
            for i in range(len(path)):
                curr_state = path[i]

                if curr_state.mode == mode:
                    if i == 0:
                        return 0, curr_state.q

                    cost = path_cost(path[: i + 1], env.batch_config_cost)
                    # env.show_config(curr_state.q)

                    return cost, curr_state.q

            return None, None

        def cost_to_next_mode_on_path(mode, path):
            for i in range(1, len(path)):
                curr_state = path[i - 1]
                next_state = path[i]

                if curr_state.mode == mode and (
                    curr_state.mode != next_state.mode or i == len(path) - 1
                ):
                    cost = path_cost(path[: i + 1], env.batch_config_cost)
                    # env.show_config(next_state.q)
                    return cost, next_state.q

            return None, None

        somewhat_informed_samples = []
        for _ in range(batch_size):
            mode = sample_mode("uniform_reached", None)
            # print(mode.task_ids)

            cost_to_reach_mode, mode_start_config = cost_to_mode_on_path(
                mode, current_best_path
            )
            next_mode_cost, mode_end_config = cost_to_next_mode_on_path(
                mode, current_best_path
            )

            # print("next mode cost:", next_mode_cost)
            # print("path_cost", current_best_cost)

            if cost_to_reach_mode is None:
                continue

            this_mode_cost = next_mode_cost - cost_to_reach_mode
            # print("this mode cost:", this_mode_cost)

            # print(mode_start_config.state())
            # print(mode_end_config.state())

            # print(
            #     "min cost", env.config_cost(mode_start_config, mode_end_config)
            # )

            assert (
                this_mode_cost - env.config_cost(mode_start_config, mode_end_config)
                > -1e-3
            )
            if (
                this_mode_cost - env.config_cost(mode_start_config, mode_end_config)
                < 1e-3
            ):
                continue

            tmp = []
            for _ in range(1000):
                for _ in range(10):
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
                    heuristic_cost = env.config_cost(
                        mode_start_config, q
                    ) + env.config_cost(mode_end_config, q)
                    # print(heuristic_cost)
                    if heuristic_cost < this_mode_cost:
                        break

                    q = None

                if q is None:
                    continue

                tmp.append(q)

            plt.figure()
            plt.scatter([a[0][0] for a in tmp], [a[0][1] for a in tmp])
            plt.scatter([a[1][0] for a in tmp], [a[1][1] for a in tmp])
            plt.show()

            if env.is_collision_free(q, mode):
                rnd_state = State(q, mode)
                somewhat_informed_samples.append(rnd_state)

        g.add_states(somewhat_informed_samples)

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

    batch_size = 200
    transition_batch_size = 500

    costs = []
    times = []

    add_new_batch = True

    start_time = time.time()

    # TODO: add this again
    # mode_sequence = [m0]
    # while True:
    #     if env.is_terminal_mode(mode_sequence[-1]):
    #         break

    #     mode_sequence.append(env.get_next_mode(None, mode_sequence[-1]))

    # resolution 0.2 is too big
    resolution = 0.1

    cnt = 0
    while True:
        cnt = g.get_num_samples()

        print()
        print("Count:", cnt, "max_iter:", max_iter)

        if add_new_batch:
            sample_locally_informed = False
            if sample_locally_informed and current_best_path is not None:
                generate_somewhat_informed_samples(current_best_path)

            if try_sampling_around_path and current_best_path is not None:
                # sample index
                interpolated_path = interpolate_path(current_best_path)
                # interpolated_path = current_best_path
                new_states_from_path_sampling = []
                for _ in range(500):
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
                        print("AAAAA")
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

            # add new batch of nodes to
            print("Sampling uniform")
            new_states = sample_valid_uniform_batch(
                batch_size=batch_size, cost=current_best_cost
            )
            g.add_states(new_states)

            # if env.terminal_mode not in reached_modes:
            print("Sampling transitions")
            new_transitions = sample_valid_uniform_transitions(
                transistion_batch_size=transition_batch_size
            )
            g.add_transition_nodes(new_transitions)
            print("Done adding transitions")

            g.compute_lower_bound_to_goal(env.batch_config_cost)
            g.compute_lower_bound_from_start(env.batch_config_cost)

            if try_informed_sampling and current_best_cost is not None:
                print("Visualizing informed samples")
                interpolated_path = interpolate_path(current_best_path)
                # interpolated_path = current_best_path

                # for _ in range(5):
                #     generate_informed_samples(1000, interpolated_path)
                new_informed_states = generate_informed_samples(500, interpolated_path)

                g.add_states(new_informed_states)

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

        # pts_per_mode = []
        # for m in reached_modes:
        #     num_pts = 0
        #     if m in g.transition_nodes:
        #         num_pts += len(g.transition_nodes[m])

        #     if m in g.nodes:
        #         num_pts += len(g.nodes[m])

        #     pts_per_mode.append(num_pts)

        # plt.figure()
        # plt.bar([str(mode) for mode in reached_modes], pts_per_mode)
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
                    if s0.mode != s1.mode:
                        continue

                    if n0.id in n1.whitelist:
                        continue

                    if not env.is_edge_collision_free(s0.q, s1.q, s0.mode, resolution):
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

    info = {"costs": costs, "times": times}
    return current_best_path, info
