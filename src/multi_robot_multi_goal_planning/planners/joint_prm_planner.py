import numpy as np
import random

from matplotlib import pyplot as plt

from typing import List, Dict, Tuple, Optional
from numpy.typing import NDArray
import heapq

import time

from multi_robot_multi_goal_planning.problems.planning_env import State, BaseProblem
from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
    NpConfiguration,
    config_dist,
    batch_config_dist,
)
from multi_robot_multi_goal_planning.problems.util import path_cost


class Node:
    state: State
    id_counter = 0

    def __init__(self, state, is_transition=False):
        self.state = state
        self.cost = None

        self.is_transition = is_transition

        self.neighbors = []

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

    def __init__(self, start: State, dist_fun):
        self.root = Node(start)
        # self.nodes = [self.root]

        self.batch_dist_fun = batch_config_dist

        self.nodes = {}
        self.nodes[self.root.state.mode] = [self.root]

        self.transition_nodes = {}  # contains the transitions at the end of the mode
        self.goal_nodes = []

        self.blacklist = set()
        self.whitelist = set()

        self.dist = dist_fun

        self.mode_to_goal_lb_cost = {}

    def compute_lb_mode_transisitons(self, cost, mode_sequence):
        # this assumes that we deal with a sequence
        cheapest_transition = []
        for i, current_mode in enumerate(mode_sequence[:-1]):
            next_mode = mode_sequence[i + 1]

            # find transition nodes in current mode
            current_mode_transitions = self.transition_nodes[current_mode]

            next_mode_transitions = []
            if i == len(mode_sequence) - 2:
                next_mode_transitions = self.goal_nodes
            else:
                next_mode_transitions = self.transition_nodes[next_mode]

            min_cost = 1e6
            for cmt in current_mode_transitions:
                for nmt in next_mode_transitions:
                    min_cost = min(cost(cmt.state.q, nmt.state.q), min_cost)

            cheapest_transition.append(min_cost)

        for i, current_mode in enumerate(mode_sequence[:-1]):
            self.mode_to_goal_lb_cost[current_mode] = sum(
                cheapest_transition[i:]
            )

        self.mode_to_goal_lb_cost[self.goal_nodes[0].state.mode] = 0

        # print(cheapest_transition)
        # print(self.mode_to_goal_lb_cost)

    def add_node(self, new_node: Node) -> None:
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
        nodes = []
        for q, this_mode, next_mode in transitions:
            node_this_mode = Node(State(q, this_mode), True)
            node_next_mode = Node(State(q, next_mode), True)

            if next_mode is not None:
                node_next_mode.neighbors = [node_this_mode]
                node_this_mode.neighbors = [node_next_mode]

            nodes.append(node_this_mode)

            if next_mode is not None:
                nodes.append(node_next_mode)
            else:
                self.goal_nodes.append(node_this_mode)

            if this_mode in self.transition_nodes:
                self.transition_nodes[this_mode].append(node_this_mode)
            else:
                self.transition_nodes[this_mode] = [node_this_mode]

        # self.add_nodes(nodes)

    def get_neighbors(self, node, k=20):
        key = node.state.mode
        if key in self.nodes:
            node_list = self.nodes[key]
            dists = self.batch_dist_fun(node.state.q, [n.state.q for n in node_list])

        transition_node_list = self.transition_nodes[key]
        transition_dists = self.batch_dist_fun(node.state.q, [n.state.q for n in transition_node_list])

        # plt.plot(dists)
        # plt.show()

        dim = len(node.state.q.state())

        if True:
            # if key in self.nodes:
            #     k_star = int(np.e * (1 + 1 / dim) * np.log(len(node_list))) + 1
            #     # # print(k_star)
            #     k = k_star
            best_nodes = []

            if key in self.nodes:
                k_clip = min(k, len(node_list) - 1)
                topk = np.argpartition(dists, k_clip)[:k_clip+1]
                topk = topk[np.argsort(dists[topk])]

                best_nodes = [node_list[i] for i in topk]

            transition_k_clip = min(k, len(transition_node_list) - 1)
            transition_topk = np.argpartition(transition_dists, transition_k_clip)[:transition_k_clip+1]
            transition_topk = transition_topk[np.argsort(transition_dists[transition_topk])]

            best_transition_nodes = [transition_node_list[i] for i in transition_topk]

            best_nodes = best_nodes + best_transition_nodes
        else:
            r = 10
            # r_star = (np.log(len(node_list)) / len(node_list)) ** (1/dim) * 2 * (1 + 1/dim) ** (1/dim)
            # r = r_star

            best_nodes = [n for i, n in enumerate(node_list) if dists[i] < r]

        if node.is_transition:
            # we do not want to have other transition nodes as neigbors
            # filtered_neighbors = [n for n in best_nodes if not n.is_transition]

            return best_nodes + node.neighbors

        return best_nodes[1:]

    def search(self, start_node, goal_nodes: List, env: BaseProblem, best_cost=None):
        open_queue = []
        closed_list = set()

        goal = None

        h_cache = {}

        def h(node):
            if node in h_cache:
                return h_cache[node]

            lb_to_goal_through_rest_of_modes = 0
            if len(self.mode_to_goal_lb_cost) > 0:
                lb_to_goal_through_rest_of_modes = self.mode_to_goal_lb_cost[
                    node.state.mode
                ]

            # return lb_to_goal_through_rest_of_modes

            # compute lowest cost to get to the goal:
            current_mode = node.state.mode
            if current_mode.task_ids == env.terminal_mode:
                mode_cost = None
                for g in self.goal_nodes:
                    cost_to_transition = env.config_cost(node.state.q, g.state.q)
                    if mode_cost is None or cost_to_transition < mode_cost:
                        mode_cost = cost_to_transition

                h_cache[node] = mode_cost
                return mode_cost

            mode_cost = min(
                env.batch_config_cost(
                    [node.state] * len(self.transition_nodes[node.state.mode]),
                    [n.state for n in self.transition_nodes[node.state.mode]],
                )
            )

            h_cache[node] = mode_cost + lb_to_goal_through_rest_of_modes
            return mode_cost + lb_to_goal_through_rest_of_modes

        def d(n0, n1):
            # return 1.0
            cost = env.config_cost(n0.state.q, n1.state.q)
            return cost

        reached_modes = []

        parents = {start_node: None}
        gs = {start_node: 0}  # best cost to get to a node

        # populate open_queue and fs
        start_edges = [(start_node, n) for n in self.get_neighbors(start_node)]

        fs = {}  # total cost of a node (f = g + h)
        for e in start_edges:
            if e[0] != e[1]:
                # open_queue.append(e)
                edge_cost = d(e[0], e[1])
                fs[e] = gs[start_node] + edge_cost + h(e[1])
                heapq.heappush(open_queue, (fs[e], edge_cost, e))

        num_iter = 0
        while len(open_queue) > 0:
            num_iter += 1

            if num_iter % 10000 == 0:
                print(len(open_queue))

            f_pred, edge_cost, edge = heapq.heappop(open_queue)
            n0, n1 = edge

            g_tentative = gs[n0] + edge_cost

            # if we found a better way to get there before, do not expand this edge
            if n1 in gs and g_tentative >= gs[n1]:
                continue

            # check edge sparsely now. if it is not valid, blacklist it, and continue with the next edge
            collision_free = False
            if edge in self.whitelist or (edge[1], edge[0]) in self.whitelist:
                collision_free = True
            else:
                if edge in self.blacklist or (edge[1], edge[0]) in self.blacklist:
                    continue
            
                q0 = n0.state.q
                q1 = n1.state.q
                collision_free = env.is_edge_collision_free(q0, q1, n0.state.mode)

                if not collision_free:
                    self.blacklist.add(edge)
                    continue
                else:
                    self.whitelist.add(edge)

            if n0.state.mode not in reached_modes:
                reached_modes.append(n0.state.mode)

            # print('reached modes', reached_modes)

            gs[n1] = g_tentative
            parents[n1] = n0

            if n1 in goal_nodes:
                goal = n1
                break

            # get_neighbors
            neighbors = self.get_neighbors(n1, 10)

            # add neighbors to open_queue
            edge_costs = env.batch_config_cost(
                [n1.state] * len(neighbors), [n.state for n in neighbors]
            )
            for i, n in enumerate(neighbors):
                if n == n1 or n == n0:
                    continue

                if (n, n1) in self.blacklist or (n1, n) in self.blacklist:
                    continue

                g_new = g_tentative + edge_costs[i]

                if n not in gs or g_new < gs[n]:
                    # sparsely check only when expanding
                    # cost to get to neighbor:
                    fs[(n1, n)] = g_new + h(n)

                    if best_cost is not None and fs[(n1, n)] > best_cost:
                        continue

                    if n not in closed_list:
                        heapq.heappush(
                            open_queue, (fs[(n1, n)], edge_costs[i], (n1, n))
                        )

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

    def search_with_vertex_queue(self, start_node, goal_nodes: List, env: BaseProblem):
        open_queue = []

        goal = None

        def h(node):
            return 0
            # compute lowest cost to get to the next mode:
            total_cost = 0

            current_mode = node.state.mode
            next_mode = None
            if current_mode == env.terminal_mode:
                mode_cost = None
                for g in self.goal_nodes:
                    cost_to_transition = batch_config_dist(node.state.q, [g.state.q])
                    if mode_cost is None or cost_to_transition < mode_cost:
                        mode_cost = cost_to_transition

                return mode_cost
            else:
                next_mode = env.get_next_mode(None, current_mode)

            mode_cost = None
            for transition in self.transition_nodes:
                if next_mode == transition.state.mode:
                    cost_to_transition = batch_config_dist(
                        node.state.q, [transition.state.q]
                    )
                    if mode_cost is None or cost_to_transition < mode_cost:
                        mode_cost = cost_to_transition

            # print(mode_cost)

            return mode_cost

        def d(n0, n1):
            # return 1.0
            dist = batch_config_dist(n0.state.q, [n1.state.q])[0]
            return dist

        parents = {start_node: None}
        gs = {start_node: 0}  # best cost to get to a node

        # populate open_queue and fs

        fs = {start_node: h(start_node)}  # total cost of a node (f = g + h)
        heapq.heappush(open_queue, (fs[start_node], start_node))

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

            print('blacklist',len(self.blacklist))
            print('whitelist',len(self.whitelist))

            # get_neighbors
            neighbors = self.get_neighbors(node)

            edge_costs = env.batch_config_cost(
                [node.state] * len(neighbors), [n.state for n in neighbors]
            )
            # add neighbors to open_queue
            for i, n in enumerate(neighbors):
                if n == node:
                    continue

                if (node, n) in self.blacklist or (n, node) in self.blacklist:
                    continue

                g_new = gs[node] + edge_costs[i]

                if n not in gs or g_new < gs[n]:
                    # collision check

                    collision_free = False
                    if (node, n) in self.whitelist or (n, node) in self.whitelist:
                        collision_free = True
                    else:
                        collision_free = env.is_edge_collision_free(
                            node.state.q, n.state.q, n.state.mode
                        )

                        if not collision_free:
                            self.blacklist.add((node, n))
                            continue
                        else:
                            self.whitelist.add((node, n))

                    # cost to get to neighbor:
                    gs[n] = g_new
                    fs[n] = g_new + h(n)
                    parents[n] = node

                    if n not in open_queue:
                        heapq.heappush(open_queue, (fs[n], n))

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
) -> Optional[Tuple[List[State], List]]:
    q0 = env.get_start_pos()
    m0 = env.get_start_mode()

    reached_modes = [m0]

    def sample_mode(mode_sampling_type="weighted", found_solution=False):
        if mode_sampling_type == "uniform_reached":
            m_rnd = random.choice(reached_modes)
        elif mode_sampling_type == "weighted":
            # sample such that we tend to get similar number of pts in each mode
            w = []
            for m in reached_modes:
                if m in g.nodes:
                    w.append(1 / (1+len(g.nodes[m])))
                else:
                    w.append(1)
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

    def sample_valid_uniform_batch(batch_size, found_solution):
        new_samples = []

        while len(new_samples) < batch_size:
            # print(len(new_samples))
            # sample mode
            m = sample_mode("weighted", found_solution)

            # print(m)

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

            q = NpConfiguration.from_list(q)

            if env.is_collision_free(q.state(), m):
                new_samples.append(State(q, m))

        return new_samples

    def sample_valid_uniform_transitions(transistion_batch_size):
        transitions = []

        while len(transitions) < transistion_batch_size:
            # sample mode
            mode = sample_mode("uniform_reached", None)

            # sample transition at the end of this mode
            goals_to_sample = env.get_goal_constrained_robots(mode)
            active_task = env.get_active_task(mode)
            goal_sample = active_task.goal.sample()

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

            if mode.task_ids == env.terminal_mode:
                next_mode = None
            else:
                next_mode = env.get_next_mode(q, mode)

            q = NpConfiguration.from_list(q)

            if env.is_collision_free(q.state(), mode):
                transitions.append((q, mode, next_mode))

                if next_mode not in reached_modes and next_mode is not None:
                    reached_modes.append(next_mode)

        return transitions

    g = Graph(State(q0, m0), batch_config_dist)

    current_best_cost = None
    current_best_path = None

    batch_size = 500
    transition_batch_size = 500

    costs = []
    times = []

    add_new_batch = True

    start_time = time.time()

    mode_sequence = [m0]
    while True:
        if mode_sequence[-1].task_ids == env.terminal_mode:
            break

        mode_sequence.append(env.get_next_mode(None, mode_sequence[-1]))

    cnt = 0
    while True:
        print("Count:", cnt, "max_iter:", max_iter)

        if add_new_batch:
            # add new batch of nodes to
            print("Sampling uniform")
            new_states = sample_valid_uniform_batch(
                batch_size=batch_size, found_solution=(current_best_cost is not None)
            )
            g.add_states(new_states)

            # if env.terminal_mode not in reached_modes:
            print("Sampling transitions")
            new_transitions = sample_valid_uniform_transitions(
                transistion_batch_size=transition_batch_size
            )
            g.add_transition_nodes(new_transitions)

            g.compute_lb_mode_transisitons(env.config_cost, mode_sequence)

        # search over nodes:
        # 1. search from goal state with sparse check
        reached_terminal_mode = False
        for m in reached_modes:
            if m.task_ids == env.terminal_mode:
                reached_terminal_mode = True

        if not reached_terminal_mode:
            continue

        while True:
            sparsely_checked_path = g.search(g.root, g.goal_nodes, env, current_best_cost)

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

                    if (n0, n1) in g.whitelist:
                        continue

                    if not env.is_edge_collision_free(s0.q, s1.q, s0.mode):
                        print("Path is in collision")
                        is_valid_path = False
                        # env.show(True)
                        g.blacklist.add((n0, n1))
                        break
                    else:
                        g.whitelist.add((n0, n1))

                if is_valid_path:
                    path = [node.state for node in sparsely_checked_path]
                    new_path_cost = path_cost(path, env.batch_config_cost)
                    if current_best_cost is None or new_path_cost < current_best_cost:
                        current_best_path = path
                        current_best_cost = new_path_cost

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

        cnt += batch_size

    costs.append(costs[-1])
    times.append(time.time() - start_time)

    info = {"costs": costs, "times": times}
    return current_best_path, info
