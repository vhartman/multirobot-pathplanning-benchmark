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


class Graph:
    root: State
    nodes: Dict

    # batch_dist_fun

    def __init__(self, start: State, batch_dist_fun, use_k_nearest=True):
        self.root = Node(start)
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
        self.transition_node_lb_cache = {}

        self.reverse_transition_node_array_cache = {}

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def compute_lb_mode_transisitons(self, batch_cost):
        if True:
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

                neighbors_next_mode = [
                    n.neighbors[0]
                    for n in self.reverse_transition_nodes[node.state.mode]
                ]

                neighbors_this_mode = [
                    n for n in self.reverse_transition_nodes[node.state.mode]
                ]

                neighbors = neighbors_this_mode + neighbors_next_mode

                if len(neighbors) == 0:
                    continue

                if node.state.mode not in self.reverse_transition_node_array_cache:
                    self.reverse_transition_node_array_cache[node.state.mode] = (
                        np.array([n.state.q.state() for n in neighbors])
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
                        costs[id] = cost
                        n.lb_cost_to_goal = cost
                        n.neighbors[0].lb_cost_to_goal = cost
                        #     # parents[n] = node

                        # queue.append(n)
                        heapq.heappush(queue, (cost, n))
        elif True:
            # search that computes the minimum cot to reach a mode per robot
            # -> node is only one config
            # -> cost in mode/node is min(robot)
            #    this is not amazing, and could likely be better via taking more information (configs?)
            #    into account, but we leave this for future work
            for r in range(self.root.state.q.num_agents()):
                costs = {}

                queue = []
                for g in self.goal_nodes:
                    heapq.heappush(queue, (0, g))

                    costs[(g.state.mode, g.state.q[r].tobytes())] = 0
                    # parents[hash(g)] = None

                while len(queue) > 0:
                    # node = queue.pop(0)
                    _, node = heapq.heappop(queue)
                    # print(node)

                    # error happens at start node
                    if node.state.mode == self.root.state.mode:
                        continue

                    neighbors = [
                        n.neighbors[0]
                        for n in self.reverse_transition_nodes[node.state.mode]
                    ]

                    if len(neighbors) == 0:
                        continue

                    # if node.state.mode not in self.reverse_transition_node_array_cache:
                    # self.reverse_transition_node_array_cache[node.state.mode] = np.array(
                    #     [n.state.q.state() for n in neighbors]
                    # )
                    tmp = np.array([n.state.q[r] for n in neighbors])

                    tmp_config = NpConfiguration.from_list([node.state.q[r]])

                    # add neighbors to open_queue
                    edge_costs = batch_cost(tmp_config, tmp)
                    parent_cost = costs[(node.state.mode, node.state.q[r].tobytes())]
                    for i, n in enumerate(neighbors):
                        cost = parent_cost + edge_costs[i]
                        id = (n.state.mode, n.state.q[r].tobytes())
                        current_cost = costs.get(id, float("inf"))
                        if cost < current_cost:
                            costs[id] = cost
                            if n.lb_cost_to_goal is None or n.lb_cost_to_goal < cost:
                                n.lb_cost_to_goal = cost
                                n.neighbors[0].lb_cost_to_goal = cost

                            # parents[n] = node
                            # if id not in costs or cost < costs[id]:
                            #     costs[id] = cost
                            #     n.lb_cost_to_goal = cost
                            #     n.neighbors[0].lb_cost_to_goal = cost
                            #     # parents[n] = node

                            # queue.append(n)
                            heapq.heappush(queue, (cost, n))

        else:
            pass
            # search that lumps modes to be faster: I do not think this works.
            # Counterexample -> low cost path from g to t3 which has overall larger cost to teget from g to t2
            # Ex below: want to take t3_v2
            #
            # (t3_v1)    g              (t3_v2)         t2
            #
            # -> would need to maintain multiple q's per mode: would be possible, but makes complexity bigger again
            # costs = {}
            # queue = []

            # for g in self.goal_nodes:
            #     heapq.heappush(queue, (0, g.state.mode, g.state.q))
            #     costs[(g.state.mode, g.state.q.state().tobytes())] = 0

            # while len(queue) > 0:
            #     _, mode, byte_array = heapq.heappop(queue)

            #     if mode == self.root.state.mode:
            #         continue

            #     config = np.frombuffer(byte_array)

            #     neighbors = [
            #         n.neighbors[0] for n in self.reverse_transition_nodes[mode]
            #     ]

            #     if len(neighbors) == 0:
            #         continue

            #     for n in neighbors:
            #         active_task = env.get_active_task(n.mode, mode.task_ids)
            #         active_robots = active_task.robots
            #         cost =

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

            # with ThreadPoolExecutor() as executor:
            #     result = list(executor.map(lambda node: node.state.q, node_list))

            # dists = self.batch_dist_fun(node.state.q, result) # this, and the list copm below are the slowest parts
            # result = list(map(lambda n: n.state.q, node_list))
            # dists = self.batch_dist_fun(node.state.q, result) # this, and the list copm below are the slowest parts
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

            best_nodes = best_nodes + best_transition_nodes
        else:
            r = 3

            unit_n_ball_measure = ((np.pi**0.5) ** dim) / math.gamma(dim / 2 + 1)
            informed_measure = 1
            if space_extent is not None:
                informed_measure = space_extent

            best_nodes = []
            if key in self.nodes:
                r_star = 2 * (
                    informed_measure
                    / unit_n_ball_measure
                    * (np.log(len(node_list)) / len(node_list))
                    * (1 + 1 / dim)
                ) ** (1 / dim)
                r = r_star

                best_nodes = [n for i, n in enumerate(node_list) if dists[i] < r]

            best_transition_nodes = []
            if key in self.transition_nodes:
                r_star = 2 * (
                    (1 + 1 / dim)
                    * informed_measure
                    / unit_n_ball_measure
                    * (np.log(len(transition_node_list)) / len(transition_node_list))
                ) ** (1 / dim)
                # print(node.state.mode, r_star)
                r = r_star

                if len(transition_node_list) == 1:
                    r = 1e6

                best_transition_nodes = [
                    n
                    for i, n in enumerate(transition_node_list)
                    if transition_dists[i] < r
                ]

            best_nodes = best_nodes + best_transition_nodes

        if node.is_transition:
            # we do not want to have other transition nodes as neigbors
            # filtered_neighbors = [n for n in best_nodes if not n.is_transition]

            return best_nodes + node.neighbors

        return best_nodes[1:]

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

            h_cache[node] = min_cost
            return min_cost

        def d(n0, n1):
            # return 1.0
            cost = env.config_cost(n0.state.q, n1.state.q)
            return cost

        # reached_modes = []

        parents = {start_node: None}
        gs = {start_node.id: 0}  # best cost to get to a node

        # populate open_queue and fs
        start_edges = [
            (start_node, n)
            for n in self.get_neighbors(
                start_node, space_extent=np.prod(np.diff(env.limits, axis=0))
            )
        ]

        # fs = {}  # total cost of a node (f = g + h)
        for e in start_edges:
            if e[0] != e[1]:
                # open_queue.append(e)
                edge_cost = d(e[0], e[1])
                cost = gs[start_node.id] + edge_cost + h(e[1])
                # fs[(e[0].id, e[1].id)] = cost
                heapq.heappush(open_queue, (cost, edge_cost, e))
                # open_queue.append((cost, edge_cost, e))

        # open_queue.sort(reverse=True)

        num_iter = 0
        while len(open_queue) > 0:
            num_iter += 1

            if num_iter % 10000 == 0:
                print(len(open_queue))

            f_pred, edge_cost, edge = heapq.heappop(open_queue)
            # print(open_queue[-1])
            # print(open_queue[-2])
            # f_pred, edge_cost, edge = open_queue.pop()
            n0, n1 = edge

            g_tentative = gs[n0.id] + edge_cost

            # if we found a better way to get there before, do not expand this edge
            if n1.id in gs and g_tentative >= gs[n1.id]:
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
            gs[n1.id] = g_tentative
            parents[n1] = n0

            if n1 in goal_nodes:
                goal = n1
                break

            # get_neighbors
            neighbors = self.get_neighbors(
                n1, space_extent=np.prod(np.diff(env.limits, axis=0))
            )

            if len(neighbors) != 0:
                # add neighbors to open_queue
                # edge_costs = env.batch_config_cost(
                #     [n1.state] * len(neighbors), [n.state for n in neighbors]
                # )
                edge_costs = env.batch_config_cost(
                    n1.state.q, np.array([n.state.q.state() for n in neighbors])
                )
                for i, n in enumerate(neighbors):
                    # if n == n0:
                    #     continue

                    if n.id in n1.blacklist:
                        continue

                    edge_cost = edge_costs[i]
                    g_new = g_tentative + edge_cost

                    # if n.id in gs:
                    #     print(n.id)

                    if n.id not in gs or g_new < gs[n.id]:
                        # sparsely check only when expanding
                        # cost to get to neighbor:
                        f_node = g_new + h(n)
                        # fs[(n1, n)] = f_node

                        if best_cost is not None and f_node > best_cost:
                            continue

                        # if n not in closed_list:
                        heapq.heappush(open_queue, (f_node, edge_cost, (n1, n)))
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

            path.append(n)
            path = path[::-1]

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
            if node in h_cache:
                return h_cache[node]

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

            h_cache[node] = min_cost
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

    def sample_valid_uniform_batch(batch_size, cost):
        new_samples = []

        if True:
            while len(new_samples) < batch_size:
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

    batch_size = 500
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
        print("Count:", cnt, "max_iter:", max_iter)

        if add_new_batch:
            # if current_best_path is not None:
            # if cnt > 5000 and current_best_path is not None:
            if False and current_best_path is not None:
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
                for _ in range(200):
                    mode = sample_mode("uniform_reached", None)
                    print(mode.task_ids)

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
                        this_mode_cost
                        - env.config_cost(mode_start_config, mode_end_config)
                        > -1e-3
                    )
                    if (
                        this_mode_cost
                        - env.config_cost(mode_start_config, mode_end_config)
                        < 1e-3
                    ):
                        continue

                    # tmp = []
                    # for _ in range(1000):
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

                    # tmp.append(q)

                    # plt.figure()
                    # plt.scatter([a[0][0] for a in tmp], [a[0][1] for a in tmp])
                    # plt.scatter([a[1][0] for a in tmp], [a[1][1] for a in tmp])
                    # plt.show()

                    if env.is_collision_free(q, mode):
                        rnd_state = State(q, mode)
                        somewhat_informed_samples.append(rnd_state)

                g.add_states(somewhat_informed_samples)

            if try_sampling_around_path and current_best_path is not None:
                # sample index
                interpolated_path = interpolate_path(current_best_path)
                # interpolated_path = current_best_path
                for _ in range(200):
                    idx = random.randint(0, len(interpolated_path) - 2)
                    state = interpolated_path[idx]

                    # this is a transition. we would need to figure out which robots are active and not sample those
                    q = []
                    if state.mode != interpolated_path[idx + 1].mode:
                        current_task_ids = state.mode.task_ids
                        next_task_ids = interpolated_path[idx + 1].mode.task_ids

                        task = env.get_active_task(state.mode, next_task_ids)
                        involved_robots = task.robots
                        for i in range(len(env.robots)):
                            r = env.robots[i]
                            if r in involved_robots:
                                qr = state.q[i]
                            else:
                                qr_mean = state.q[i]

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

                            g.add_states([rnd_state])

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

            g.compute_lb_mode_transisitons(env.batch_config_cost)

        # search over nodes:
        # 1. search from goal state with sparse check
        reached_terminal_mode = False
        for m in reached_modes:
            if env.is_terminal_mode(m):
                reached_terminal_mode = True

        if not reached_terminal_mode:
            continue

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
                    break

            else:
                print("Did not find a solution")
                add_new_batch = True
                break

        if not optimize and current_best_cost is not None:
            break

        if cnt >= max_iter:
            break

        cnt += batch_size + transition_batch_size

    costs.append(costs[-1])
    times.append(time.time() - start_time)

    info = {"costs": costs, "times": times}
    return current_best_path, info
