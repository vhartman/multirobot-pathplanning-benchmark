import numpy as np
import random

from matplotlib import pyplot as plt

from typing import List, Dict, Tuple, Optional
from numpy.typing import NDArray
import heapq
from itertools import product

import time

from multi_robot_multi_goal_planning.problems.planning_env import State, base_env
from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
    NpConfiguration,
    config_dist,
    batch_config_dist,
)
from multi_robot_multi_goal_planning.problems.util import path_cost


class RobotNode:
    def __init__(self, rs, q, task):
        self.rs = rs
        self.q = q
        self.task = task


class Node:
    state: State
    id_counter = 0

    def __init__(self, state):
        self.state = state
        self.id = Node.id_counter

        Node.id_counter += 1


class ImplicitTensorGraph:
    robots: List[str]

    def __init__(self, start: State, robots, dist_fun):
        self.robots = robots
        self.root = start

        self.batch_dist_fun = batch_config_dist

        self.robot_nodes = {}
        self.transition_nodes = {}

        self.goal_nodes = []

        self.blacklist = set()
        self.whitelist = set()

    def get_key(self, rs, t):
        robot_key = "_".join(rs)
        return robot_key + str(t)

    def add_robot_node(self, rs, q, t, is_transition):
        key = self.get_key(rs, t)
        if key not in self.robot_nodes:
            self.robot_nodes[key] = []

        if is_transition:
            self.transition_nodes[key].append(RobotNode(rs, q, t))
        else:
            self.robot_nodes[key].append(RobotNode(rs, q, t))

    def get_robot_neighbors(self, rs, q, t, get_transitions, k=10):
        key = self.get_key(rs, t)

        if get_transitions:
            nodes = self.transition_nodes[key]
        else:
            nodes = self.robot_nodes[key]

        dists = self.batch_dist_fun(q, [n.q for n in nodes])

        k_clip = min(k, len(nodes) - 1)
        topk = np.argpartition(dists, k_clip)[:k_clip]
        topk = topk[np.argsort(dists[topk])]

        best_nodes = [nodes[i] for i in topk]

        return best_nodes

    def get_joint_groups(self, mode):
        groups = {}
        for i, r in enumerate(self.robots):
            if mode[i] in groups:
                groups[mode[i]].append(r)
            else:
                groups[mode[i]] = [r]

        return [v for k, v in groups.items()]

    # TODO: how do we deal with joint goals??
    # might be doable by introducing separationi between goals and rest
    # and then add flag to determine of we need to sample from joint graph
    # or not
    def get_neighbors_from_implicit_graph(self, node, k=10):
        mode = node.state.mode

        # get separate nn for each robot-group
        per_group_nn = {}
        for i, r in enumerate(self.robots):
            task = mode[i]
            per_group_nn[r] = self.get_robot_neighbors(r, node.state.q, task, False, k)

        # TODO: for the transitioning robots, add the transition neighbors
        active_robots = env.get_goal_constrained_robots()
        transitions = self.get_robot_neighbors(active_robots, node.state.q, task, True, k)

        tmp = [per_group_nn[r] for r in self.robots if r not in active_robots]
        tmp.append(transitions)
        transition_combinations = product(tmp)
        transition_combination_states = transition_combinations
        
        # out of those, get the closest n
        combinations = product([per_group_nn[r] for r in self.robots])
        combination_states = combinations

        all_states = combination_states + transition_combination_states
        dists = self.batch_dist_fun(node.state.q, all_states)

        k_clip = min(k, len(all_states) - 1)
        topk = np.argpartition(dists, k_clip)[:k_clip]
        topk = topk[np.argsort(dists[topk])]

        best_nodes = [all_states[i] for i in topk]

        return best_nodes

    def get_neighbors(self, node, k=10):
        neighbors = self.get_neighbors_from_implicit_graph(node, k)
        return neighbors

    # this is a copy of the search from the normal prm
    def search(self, start_node, goal_nodes: List, env: base_env, best_cost=None):
        open_queue = []
        closed_list = set()

        goal = None

        h_cache = {}

        def h(node):
            return 0
            # if node in h_cache:
            #     return h_cache[node]

            # lb_to_goal_through_rest_of_modes = 0
            # if len(self.mode_to_goal_lb_cost) > 0:
            #     lb_to_goal_through_rest_of_modes = self.mode_to_goal_lb_cost[
            #         tuple(node.state.mode)
            #     ]

            # # return lb_to_goal_through_rest_of_modes

            # # compute lowest cost to get to the goal:
            # current_mode = node.state.mode
            # if current_mode == env.terminal_mode:
            #     mode_cost = None
            #     for g in self.goal_nodes:
            #         cost_to_transition = env.config_cost(node.state.q, g.state.q)
            #         if mode_cost is None or cost_to_transition < mode_cost:
            #             mode_cost = cost_to_transition

            #     h_cache[node] = mode_cost
            #     return mode_cost

            # mode_cost = min(
            #     env.batch_config_cost(
            #         [node.state] * len(self.transition_nodes[tuple(node.state.mode)]),
            #         [n.state for n in self.transition_nodes[tuple(node.state.mode)]],
            #     )
            # )

            # h_cache[node] = mode_cost + lb_to_goal_through_rest_of_modes
            # return mode_cost + lb_to_goal_through_rest_of_modes

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

# this does implicit search over a graph
# nearest neighbors are found by checking the nearest neighbors in each separate graph, and then
# taking the nearest of those
def tensor_prm_planner(
    env: base_env,
    optimize: bool = True,
    mode_sampling_type: str = "greedy",
    max_iter: int = 2000,
) -> Optional[Tuple[List[State], List]]:
    q0 = env.get_start_pos()
    m0 = env.get_start_mode()

    reached_modes = [m0]

    # env.is_collision_free()

    g = ImplicitTensorGraph(State(q0, m0), env.robots, batch_config_dist)

    current_best_cost = None
    current_best_path = None

    batch_size = 500
    transition_batch_size = 50

    costs = []
    times = []

    add_new_batch = True

    start_time = time.time()

    cnt = 0

    def sample_uniform_valid_per_robot():
        pass

    def sample_uniform_valid_transition_for_active_robots():
        pass

    while True:
        if add_new_batch:
            # sample uniform

            new_states = sample_uniform_valid_per_robot()
            g.add_robot_node()

            # sample transition
            transitions = sample_uniform_valid_transition_for_active_robots()
            g.add_robot_node()


        while True:
            potential_path = g.search()

            if len(potential_path) > 0:
                add_new_batch = False

                is_valid_path = True
                for i in range(len(potential_path) - 1):
                    n0 = potential_path[i]
                    n1 = potential_path[i + 1]

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
                    path = [node.state for node in potential_path]
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
