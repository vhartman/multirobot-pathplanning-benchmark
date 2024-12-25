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

        self.hash = None

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        if self.hash is None:
            self.hash = hash((self.state.q.state().data.tobytes(), tuple(self.state.mode)))
        return self.hash
    
        # return hash((tuple(np.round(self.state.q.state(), 3)), tuple(self.state.mode)))
        # return self.id


class ImplicitTensorGraph:
    robots: List[str]

    def __init__(self, start: State, robots, dist_fun):
        self.robots = robots
        self.root = Node(start)

        self.batch_dist_fun = batch_config_dist

        self.robot_nodes = {}
        self.transition_nodes = {}

        self.goal_nodes = []

        self.blacklist = set()
        self.whitelist = set()

        self.mode_sequence = None

        self.per_robot_task_to_goal_lb_cost = {}

    def compute_lb_mode_transisitons(self, cost, mode_sequence):
        # cost_per_robot = None
        self.mode_sequence = mode_sequence

        for i, r in enumerate(self.robots):
            cheapest_transition = []
            for j, m in enumerate(mode_sequence[:-1]):
                next_mode = mode_sequence[j+1]

                if m[i] == next_mode[i]:
                    continue
                
                nodes = self.transition_nodes

                for k in self.transition_nodes.keys():
                    task = int(k.split("_")[-1])
                    if m[i] == task:
                        key = k

                    if next_mode[i] == task:
                        next_key = k

                # key = self.get_key([r], t)
                # if key not in self.transition_nodes:
                #     continue 

                nodes = self.transition_nodes[key]
                next_nodes = self.transition_nodes[next_key]

                min_cost = None
                for n in nodes:
                    q = n.q[n.rs.index(r)]
                    for nn in next_nodes:
                        qn = nn.q[nn.rs.index(r)]

                        c = cost(NpConfiguration.from_numpy(q), NpConfiguration.from_numpy(qn))
                        if min_cost is None or c < min_cost:
                            min_cost = c
                
                cheapest_transition.append(min_cost)

            cnt = 0
            for j, current_mode in enumerate(mode_sequence[:-1]):
                if current_mode[i] == mode_sequence[j+1][i]:
                    continue

                self.per_robot_task_to_goal_lb_cost[tuple([r, current_mode[i]])] = sum(
                    cheapest_transition[cnt:]
                )
                cnt += 1

    def get_key(self, rs, t):
        robot_key = "_".join(rs)
        return robot_key + "_" +  str(t)

    def add_robot_node(self, rs, q, t, is_transition):
        key = self.get_key(rs, t)

        if is_transition:
            # print(f"adding transition with key {key}")
            if key not in self.transition_nodes:
                self.transition_nodes[key] = []

            does_already_exist = False
            for n in self.transition_nodes[key]:
                if rs == n.rs and t == n.task and np.linalg.norm(np.concatenate(q) - np.concatenate(n.q)) < 1e-5:
                    does_already_exist = True
                    break

            if not does_already_exist:
                self.transition_nodes[key].append(RobotNode(rs, q, t))
        else:
            # print(f"adding node with key {key}")
            if key not in self.robot_nodes:
                self.robot_nodes[key] = []
            self.robot_nodes[key].append(RobotNode(rs, q, t))

    def get_robot_neighbors(self, rs, q, t, get_transitions, k=10):
        # print(f"finding robot neighbors for {rs}")
        key = self.get_key(rs, t)

        if get_transitions:
            nodes = self.transition_nodes[key]
        else:
            nodes = self.robot_nodes[key]

        # print(nodes)
        # print(len(nodes))

        # print(f"computing distances for {rs}")
        # TODO: this makes the thing slow
        # TODO: introduce possibility to compute distances without having to convert to configuration
        dists = self.batch_dist_fun(q, [NpConfiguration.from_list(n.q) for n in nodes])

        if True:
            k_clip = min(k, len(nodes) - 1)
            topk = np.argpartition(dists, k_clip)[:k_clip+1]
            topk = topk[np.argsort(dists[topk])]

            best_nodes = [nodes[i] for i in topk]
        else:
            r = 5
            best_nodes = [n for i, n in enumerate(nodes) if dists[i] < r]

        # print(q.state())
        # print(best_nodes[0].q)

        return best_nodes

    def get_joint_groups(self, mode):
        groups = {}
        for i, r in enumerate(self.robots):
            if mode[i] in groups:
                groups[mode[i]].append(r)
            else:
                groups[mode[i]] = [r]

        return [v for k, v in groups.items()]

    def get_neighbors_from_implicit_graph(self, node, active_robots, k=10):
        mode = node.state.mode

        # get separate nn for each robot-group
        per_group_nn = {}
        for i, r in enumerate(self.robots):
            task = mode[i]
            # extract the correct state here
            q = node.state.q[i]
            per_group_nn[r] = self.get_robot_neighbors([r], NpConfiguration.from_numpy(q), task, False, k)

        # print(f"finding neighbors from transitions for {active_robots}")
        qs = []
        for r in active_robots:
            i = self.robots.index(r)
            q = node.state.q[i]
            qs.append(q)
            task = mode[i]

        active_robot_config = NpConfiguration.from_list(qs)
        transitions = self.get_robot_neighbors(active_robots, active_robot_config, task, True, k)

        # print("making transition tensor product")
        tmp = [per_group_nn[r] for r in self.robots if r not in active_robots]
        tmp.append(transitions)

        transition_combinations = list(product(*tmp))
        transition_combination_states = []
        for combination in transition_combinations:
            d = {}
            for robot_node in combination:
                for i, r in enumerate(robot_node.rs):
                    d[r] = robot_node.q[i]
            q = [d[r] for r in self.robots]
            transition_combination_states.append(NpConfiguration.from_list(q))
        
        # print("making other tensor product")
        combinations = list(product(*[per_group_nn[r] for r in self.robots]))

        combination_states = []
        for combination in combinations:
            d = {}
            for robot_node in combination:
                for i, r in enumerate(robot_node.rs):
                    d[r] = robot_node.q[i]
            q = [d[r] for r in self.robots]
            combination_states.append(NpConfiguration.from_list(q))

        # out of those, get the closest n
        dists = self.batch_dist_fun(node.state.q, combination_states)

        if True:
            k_clip = min(k, len(combination_states) - 1)
            topk = np.argpartition(dists, k_clip)[:k_clip+1]
            topk = topk[np.argsort(dists[topk])]

            best_normal_nodes = [Node(State(combination_states[i], mode)) for i in topk]
        else:
            best_normal_nodes = [Node(State(s, mode)) for i, s in enumerate(combination_states) if dists[i] < 1]

        transition_dists = self.batch_dist_fun(node.state.q, transition_combination_states)

        k_clip = min(k, len(transition_combination_states) - 1)
        topk = np.argpartition(transition_dists, k_clip)[:k_clip+1]
        topk = topk[np.argsort(transition_dists[topk])]

        best_transition_nodes = [Node(State(transition_combination_states[i], mode)) for i in topk]

        best_nodes = best_transition_nodes + best_normal_nodes

        return best_nodes

    def get_neighbors(self, node, active_robots, k=10):
        neighbors = self.get_neighbors_from_implicit_graph(node, active_robots, k)
        return neighbors

    # this is a copy of the search from the normal prm
    def search(self, start_node, goal_nodes: List, env: base_env, best_cost=None):
        open_queue = []
        closed_list = set()

        goal = None

        h_cache = {}

        def h(node):
            if node in h_cache:
                return h_cache[node]

            per_robot_min_to_goal = []
            
            for i, r in enumerate(self.robots):
                t = node.state.mode[i]

                for k in self.transition_nodes.keys():
                    task = int(k.split("_")[-1])
                    if t == task:
                        key = k
                        break

                # key = self.get_key([r], t)
                # if key not in self.transition_nodes:
                #     continue 

                nodes = self.transition_nodes[key]

                q = node.state.q[self.robots.index(r)]

                min_dist = 1e6
                for n in nodes:
                    transition_q = n.q[n.rs.index(r)]
                    d = np.max(np.abs(q - transition_q))
                    if d < min_dist:
                        min_dist = d
                
                from_transition_to_goal = 0
                if t != env.terminal_mode[i]:
                    from_transition_to_goal = self.per_robot_task_to_goal_lb_cost[tuple([r, t])]

                per_robot_min_to_goal.append(min_dist + from_transition_to_goal)

            return max(per_robot_min_to_goal)

        def d(n0, n1):
            # return 1.0
            cost = env.config_cost(n0.state.q, n1.state.q)
            return cost

        reached_modes = []

        parents = {start_node: None}
        gs = {start_node: 0}  # best cost to get to a node

        # populate open_queue and fs
        active_robots = env.get_goal_constrained_robots(start_node.state.mode)
        next_mode = env.get_next_mode(None, start_node.state.mode)

        start_edges = [(start_node, n) for n in self.get_neighbors(start_node, active_robots)]

        # for e in start_edges:
        #     # print(e[0].state.q.state())
        #     print(e[1].state.q.state())
        # raise ValueError

        fs = {}  # total cost of a node (f = g + h)
        for e in start_edges:
            if e[0] != e[1]:
                # open_queue.append(e)
                edge_cost = d(e[0], e[1])
                fs[e] = gs[start_node] + edge_cost + h(e[1])
                heapq.heappush(open_queue, (fs[e], edge_cost, e))

        # for o in open_queue:
        #     print(o[0])
        #     print(o[1])
        #     print(o[2][0].state.q.state())
        #     print(o[2][1].state.q.state())

        # raise ValueError

        num_iter = 0
        while len(open_queue) > 0:
            num_iter += 1

            if num_iter % 1000 == 0:
                print(len(open_queue))

            f_pred, edge_cost, edge = heapq.heappop(open_queue)
            n0, n1 = edge

            if num_iter % 1000 == 0:
                print('reached modes', reached_modes)
                print(f_pred)

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

            # env.show(False)

            gs[n1] = g_tentative
            parents[n1] = n0

            if env.done(n1.state.q, n1.state.mode):
                goal = n1
                break

            # get_neighbors
            if env.is_transition(n1.state.q, n1.state.mode):
                next_mode = env.get_next_mode(None, n1.state.mode)
                neighbors = [Node(State(n1.state.q, next_mode))]
            else:
                active_robots = env.get_goal_constrained_robots(n1.state.mode)
                # print(active_robots)
                neighbors = self.get_neighbors(n1, active_robots, 50)

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

    batch_size = 200
    transition_batch_size = 50

    costs = []
    times = []

    add_new_batch = True

    start_time = time.time()

    cnt = 0

    mode_sequence = [m0]
    while True:
        if mode_sequence[-1] == env.terminal_mode:
            break

        mode_sequence.append(env.get_next_mode(None, mode_sequence[-1]))

    def sample_mode(mode_sampling_type="weighted", found_solution=False):
        # if mode_sampling_type == "uniform_reached":
        m_rnd = random.choice(reached_modes)
        # elif mode_sampling_type == "weighted":
        #     # sample such that we tend to get similar number of pts in each mode
        #     w = []
        #     for m in reached_modes:
        #         if tuple(m) in g.nodes:
        #             w.append(1 / (1+len(g.nodes[tuple(m)])))
        #         else:
        #             w.append(1)
        #     m_rnd = random.choices(reached_modes, weights=w)[0]

        return m_rnd

    def sample_uniform_valid_per_robot(num_pts = 500):
        pts = []
        added_pts_dict = {}

        for i, r in enumerate(env.robots):
            robot_pts = 0
            while robot_pts < num_pts:
                lims = env.limits[:, env.robot_idx[r]]

                if lims[0, 0] < lims[1, 0]:
                    q = (
                        np.random.rand(env.robot_dims[r]) * (lims[1, :] - lims[0, :])
                        + lims[0, :]
                    )
                else:
                    q = np.random.rand(env.robot_dims[r]) * 6 - 3
                
                m = sample_mode("weighted")
                t = m[i]

                if env.is_collision_free_for_robot([r], q, m):
                    pt = (r, q, t)
                    pts.append(pt)
                    robot_pts += 1

                    if tuple([r, t]) not in added_pts_dict:
                        added_pts_dict[tuple([r, t])] = 0

                    added_pts_dict[tuple([r, t])] += 1

        print(added_pts_dict)

        return pts

    def sample_uniform_valid_transition_for_active_robots(num_pts=50):
        transitions = []
        for _ in range(num_pts):
            m = sample_mode("weighted")
            goal_constrainted_robots = env.get_goal_constrained_robots(m)
            active_task = env.get_active_task(m)

            goal_sample = active_task.goal.sample()

            q = goal_sample
            t = m[env.robots.index(goal_constrainted_robots[0])]

            q_list = []
            offset = 0
            for r in goal_constrainted_robots:
                dim = env.robot_dims[r]
                q_list.append(q[offset:offset+dim])
                offset += dim
            
            # print(f"checking colls for {goal_constrainted_robots}")
            if env.is_collision_free_for_robot(goal_constrainted_robots, q, m):
                transition = (goal_constrainted_robots, q_list, t)
                transitions.append(transition)

                if m == env.terminal_mode:
                    next_mode = None
                else:
                    next_mode = env.get_next_mode(q, m)

                if next_mode not in reached_modes and next_mode is not None:
                    reached_modes.append(next_mode)
            # else:
                # print("transition infeasible")
                # env.show(True)

        print(reached_modes)

        return transitions

    while True:
        if add_new_batch:
            # sample transition
            transitions = sample_uniform_valid_transition_for_active_robots(transition_batch_size)
            for transition in transitions:
                robots = transition[0]
                q = transition[1]
                task = transition[2]

                g.add_robot_node(robots, q, task, True)
            
            print('num transitions: ', len(transitions))

            # sample uniform
            new_states = sample_uniform_valid_per_robot(batch_size)
            for s in new_states:
                r = s[0]
                q = s[1]
                task = s[2]
                
                g.add_robot_node([r], [q], task, False)

            print('num new_states: ', len(new_states))

            g.compute_lb_mode_transisitons(env.config_cost, mode_sequence)
        
        if env.terminal_mode not in reached_modes:
            continue

        while True:
            # TODO: goal nodes are always empty here
            potential_path = g.search(g.root, g.goal_nodes, env, current_best_cost)

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
