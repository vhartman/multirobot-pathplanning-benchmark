import numpy as np
import random

from typing import List, Dict, Tuple, Optional
from numpy.typing import NDArray
import heapq

import time

from multi_robot_multi_goal_planning.problems.planning_env import State, base_env
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
        self.nodes[tuple(self.root.state.mode)] = [self.root]

        self.transition_nodes = []
        self.goal_nodes = []

        self.blacklist = set()
        self.whitelist = set()

        self.dist = dist_fun

        self.cheapest_mode_transitions = {}

    def add_node(self, new_node: Node) -> None:
        key = tuple(new_node.state.mode)
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

            self.transition_nodes.append(node_this_mode)

        self.add_nodes(nodes)

    def get_neighbors(self, node, k=20):
        key = tuple(node.state.mode)
        node_list = self.nodes[key]
        dists = self.batch_dist_fun(node.state.q, [n.state.q for n in node_list])

        # k_clip = min(k, len(node_list) - 1)
        # topk = np.argpartition(dists, k_clip)[:k_clip]
        # topk[np.argsort(dists[topk])]

        # best_nodes = [node_list[i] for i in topk]

        best_nodes = [n for i, n in enumerate(node_list) if dists[i] < 4]

        if node.is_transition:
            # we do not want to have other transition nodes as neigbors
            # filtered_neighbors = [n for n in best_nodes if not n.is_transition]

            return node.neighbors + best_nodes

        return best_nodes

    def search(self, start_node, goal_nodes: List, env: base_env):
        open_queue = []
        closed_list = set()

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
                    cost_to_transition = batch_config_dist(node.state.q, [transition.state.q])
                    if mode_cost is None or cost_to_transition < mode_cost:
                        mode_cost = cost_to_transition
                        
            # print(mode_cost)

            return mode_cost

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

            if num_iter % 1000 == 0:
                print(len(open_queue))

            # get next node to look at
            # edge = None
            # min_f_cost = None
            # for i, e in enumerate(open_queue):
            #     f_cost = fs[e]
            #     if min_f_cost is None or f_cost < min_f_cost:
            #         min_f_cost = f_cost
            #         edge = e
            #         best_idx = i

            # print(best_idx)
            # open_queue.remove(edge)

            f_pred, edge_cost, edge = heapq.heappop(open_queue)

            # print('g:', v)

            # print(num_iter, len(open_queue))

            n0, n1 = edge

            # closed_list.add(n0)

            # if n0.state.mode == [0, 3]:
            #     env.show(True)

            g_tentative = gs[n0] + edge_cost

            # if we found a better way to get there before, do not expand this edge
            if n1 in gs and g_tentative >= gs[n1]:
                continue

            # check edge sparsely now. if it is not valid, blacklist it, and continue with the next edge
            q0 = n0.state.q
            q1 = n1.state.q

            # if edge in self.blacklist or (edge[1], edge[0]) in self.blacklist:
            #     continue

            # if env.is_edge_collision_free(q0, q1, n0.state.mode, 1):
            #     self.blacklist.add(edge)

            collision_free = False
            if edge in self.whitelist or (edge[1], edge[0]) in self.whitelist:
                collision_free = True
            else:
                if edge in self.blacklist or (edge[1], edge[0]) in self.blacklist:
                    continue

                collision_free = env.is_edge_collision_free(q0, q1, n0.state.mode)

            if not collision_free:
                self.blacklist.add(edge)
                continue
            else:
                self.whitelist.add(edge)

            # env.show(False)

            # print(q0.state())
            # print(q1.state())

            # print(n0.state.mode)
            # print(n1.state.mode)
            # env.show(False)

            if n0.state.mode not in reached_modes:
                reached_modes.append(n0.state.mode)

            # print('reached modes', reached_modes)

            gs[n1] = g_tentative
            parents[n1] = n0

            if n1 in goal_nodes:
                goal = n1
                break

            # get_neighbors
            neighbors = self.get_neighbors(n1, 500)

            # add neighbors to open_queue
            edge_costs = env.batch_config_cost([n1.state]*len(neighbors), [n.state for n in neighbors])
            for i, n in enumerate(neighbors):
                if n == n1:
                    continue

                if (n, n1) in self.blacklist or (n1, n) in self.blacklist:
                    continue

                g_new = g_tentative + edge_costs[i]

                if n not in gs or g_new < gs[n]:
                    # sparsely check only when expanding
                    # open_queue.append((n1, n))

                    # cost to get to neighbor:
                    fs[(n1, n)] = g_new + h(n)

                    if n not in closed_list:
                        heapq.heappush(open_queue, (fs[(n1, n)], edge_costs[i], (n1, n)))


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

    
    def search_with_vertex_queue(self, start_node, goal_nodes: List, env: base_env):
        open_queue = []

        goal = None

        def h(node):
            # return 0
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
                    cost_to_transition = batch_config_dist(node.state.q, [transition.state.q])
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

            # get_neighbors
            neighbors = self.get_neighbors(node)

            # add neighbors to open_queue
            for n in neighbors:
                if n == node:
                    continue

                if (node, n) in self.blacklist or (n, node) in self.blacklist:
                    continue

                g_new = gs[node] + d(node, n)

                if n not in gs or g_new < gs[n]:
                    # collision check

                    collision_free = False
                    if (node, n) in self.whitelist or (n, node) in self.whitelist:
                        collision_free = True
                    else:
                        if (node, n) in self.blacklist or (n, node) in self.blacklist:
                            continue

                        collision_free = env.is_edge_collision_free(node.state.q, n.state.q, n.state.mode)

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
    env: base_env,
    optimize: bool = True,
    mode_sampling_type: str = "greedy",
    max_iter: int = 2000,
) -> Optional[Tuple[List[State], List]]:
    q0 = env.get_start_pos()
    m0 = env.get_start_mode()

    reached_modes = [m0]

    def sample_mode(mode_sampling_type='weighted', found_solution=False):
        if mode_sampling_type == "uniform_reached":
            m_rnd = random.choice(reached_modes)
        elif mode_sampling_type == "weighted":
            # sample such that we tend to get similar number of pts in each mode
            w = []
            for m in reached_modes:
                w.append(1 / len(g.nodes[tuple(m)]))
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

            if mode == env.terminal_mode:
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

    batch_size = 200

    costs = []
    times = []

    add_new_batch = True

    start_time = time.time()

    cnt = 0
    while True:
        if add_new_batch:
            # add new batch of nodes to
            print("Sampling uniform")
            new_states = sample_valid_uniform_batch(
                batch_size=batch_size, found_solution=(current_best_cost is not None)
            )
            g.add_states(new_states)

            # if env.terminal_mode not in reached_modes:
            print("Sampling transitions")
            new_transitions = sample_valid_uniform_transitions(transistion_batch_size=100)
            g.add_transition_nodes(new_transitions)


        # search over nodes:
        # 1. search from goal state with sparse check
        if env.terminal_mode not in reached_modes:
            continue

        sparsely_checked_path = g.search(g.root, g.goal_nodes, env)

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
                    print('Path is in collision')
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

            if not optimize and current_best_cost is not None:
                break
        else:
            print('Did not find solutio')
            add_new_batch = True

        if cnt > max_iter:
            break

        cnt += batch_size

    info = {'costs': costs, 'times': times}
    return current_best_path, info
