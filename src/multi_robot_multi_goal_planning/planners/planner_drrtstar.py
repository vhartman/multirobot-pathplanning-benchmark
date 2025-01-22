from multi_robot_multi_goal_planning.planners.rrtstar_base import *

"""This file contains the dRRT based on the paper 'Finding a needle in an exponential haystack:
Discrete RRT for exploration of implicit
roadmaps in multi-robot motion planning' by K. Solovey et al."""

import numpy as np
import random

from typing import List, Dict, Tuple, Optional
from numpy.typing import NDArray
import heapq
from itertools import product

import time

from multi_robot_multi_goal_planning.problems.planning_env import State, base_env
from multi_robot_multi_goal_planning.problems.configuration import (
    NpConfiguration,
    batch_config_dist,
)
from multi_robot_multi_goal_planning.problems.util import path_cost

class Node:
    id_counter = 0

    def __init__(self, state:State, operation: Operation):
        self.state = state   
        self.q_tensor = torch.tensor(state.q.state(), device=device, dtype=torch.float32)
        self.parent = None  
        self.children = []    
        self.transition = False
        self.num_agents = state.q.num_agents()
        self.agent_dists = torch.zeros(1, self.num_agents, device = 'cpu', dtype=torch.float32)
        self.cost_to_parent = None
        self.agent_dists_to_parent = torch.zeros(1, self.num_agents, device = 'cpu', dtype=torch.float32)
        self.operation = operation
        self.idx = Node.id_counter
        Node.id_counter += 1
        self.neighbors = {}
        self.hash = None


    @property
    def cost(self):
        return self.operation.get_cost(self.idx)
    
    @cost.setter
    def cost(self, value):
        """Set the cost in the shared operation costs tensor."""
        self.operation.costs[self.idx] = value

    def __repr__(self):
        return f"<N- {self.state.q.state()}, c: {self.cost}>"
    
    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        if self.hash is None:
            self.hash = hash((self.state.q.state().data.tobytes(), tuple(self.state.mode)))
        return self.hash
    
class ImplicitTensorGraph:
    robots: List[str]

    def __init__(self, start: State, robots, operation:Operation):
        self.robots = robots
        self.root = Node(start, operation)

        self.batch_dist_fun = batch_config_dist

        self.robot_nodes = {}
        self.transition_nodes = {}
        self.blacklist = set() #Compositions that are in collisions
        self.whitelist = set() #Compositions that are collision free and were added to the tree

        self.goal_nodes = []
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
            if key not in self.robot_nodes:
                self.robot_nodes[key] = []
            self.robot_nodes[key].append(NpConfiguration.from_list(q))

            # print(f"adding transition with key {key}")
            if key not in self.transition_nodes:
                self.transition_nodes[key] = []

            does_already_exist = False
            for n in self.transition_nodes[key]:
                if rs == n.rs and t == n.task and np.linalg.norm(np.concatenate(q) - n.q.state()) < 1e-5:
                    does_already_exist = True
                    break

            if not does_already_exist:
                self.transition_nodes[key].append(NpConfiguration.from_list(q))
        else:
            # print(f"adding node with key {key}")
            if key not in self.robot_nodes:
                self.robot_nodes[key] = []
            self.robot_nodes[key].append(NpConfiguration.from_list(q))

    def get_robot_neighbors(self, rs, q, t, k=10):
        # print(f"finding robot neighbors for {rs}")
        key = self.get_key(rs, t)
        nodes = self.robot_nodes[key]
        dists = self.batch_dist_fun(q, nodes, "euclidean")
        k_clip = min(k, len(nodes) - 1)
        topk = np.argpartition(dists, k_clip)[:k_clip+1]
        topk = topk[np.argsort(dists[topk])]
        best_nodes = [nodes[i] for i in topk]
        return best_nodes

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

        # @profile # run with kernprof -l examples/run_planner.py [your environment]
        def get_combinations(to_be_combined):
            combo = ([a[0] for a in to_be_combined])

            mapping = {}
            dims = {}
            for j, robot_node in enumerate(combo):
                for i, r in enumerate(robot_node.rs):
                    mapping[r] = (j, i)
                    dims[r] = len(robot_node.q[i])
            flat_mapping = [mapping[r] for r in self.robots]
            
            slices = []
            s = 0
            for r in self.robots:
                slices.append((s, s+dims[r]))
                s += dims[r]

            combined = []        
            for combination in product(*to_be_combined):
                q = [combination[j].q[i] for (j, i) in flat_mapping]
                concat = np.concat(q)
                combined.append(NpConfiguration(concat, slices))

            return combined
            
        transition_combination_states = get_combinations(tmp)
        combination_states = get_combinations([per_group_nn[r] for r in self.robots])

        # out of those, get the closest n
        dists = self.batch_dist_fun(node.state.q, combination_states)
        k_clip = min(k, len(combination_states) - 1)
        topk = np.argpartition(dists, k_clip)[:k_clip+1]
        topk = topk[np.argsort(dists[topk])]

        best_normal_nodes = [Node(State(combination_states[i], mode)) for i in topk]


        transition_dists = self.batch_dist_fun(node.state.q, transition_combination_states)

        k = k*2

        k_clip = min(k, len(transition_combination_states) - 1)
        topk = np.argpartition(transition_dists, k_clip)[:k_clip+1]
        topk = topk[np.argsort(transition_dists[topk])]

        best_transition_nodes = [Node(State(transition_combination_states[i], mode)) for i in topk]

        best_nodes = best_transition_nodes + best_normal_nodes

        # plt.figure()
        # for n in best_normal_nodes:
        #     c = ['blue', 'red']
        #     for i in range(n.state.q.num_agents()):
        #         q = n.state.q[i]
        #         plt.scatter(q[0], q[1], color=c[i])

        #     for i in range(node.state.q.num_agents()):
        #         q = node.state.q[i]
        #         plt.scatter(q[0], q[1], color='orange')

        # plt.gca().set_xlim(-1, 1)
        # plt.gca().set_ylim(-1, 1)

        # plt.show()

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


class dRRTstar(BaseRRTstar):
    def __init__(self, env, config: ConfigManager):
        super().__init__(env, config)
        self.g = None
        self.tree = []

    def UpdateCost(self, n:Node) -> None:
        stack = [n]
        while stack:
            current_node = stack.pop()
            children = current_node.children
            if children:
                for _, child in enumerate(children):
                    child.cost = current_node.cost + child.cost_to_parent
                    child.agent_dists = current_node.agent_dists + child.agent_dists_to_parent
                stack.extend(children)
   
    def ManageTransition(self, n_new: Node, iter: int, new_node:bool = True) -> None:
        if self.env.get_active_task(self.operation.active_mode.label).goal.satisfies_constraints(n_new.state.q.state()[self.operation.active_mode.indices], self.env.tolerance):
            if new_node:
                self.operation.active_mode.transition_nodes.append(n_new)
                n_new.transition = True
            # Check if initial transition node of current mode is found
            if self.operation.active_mode.label == self.operation.modes[-1].label and not self.operation.init_sol:
                print(time.time()-self.start)
                print(f"{iter} {self.operation.active_mode.constrained_robots} found T{self.env.get_current_seq_index(self.operation.active_mode.label)}")
                if self.env.terminal_mode != self.operation.modes[-1].label:
                    self.operation.modes.append(Mode(self.env.get_next_mode(n_new.state.q,self.operation.active_mode.label), self.env))
                    self.ModeInitialization(self.operation.modes[-1])
                elif self.operation.active_mode.label == self.env.terminal_mode:
                    self.operation.ptc_iter = iter
                    self.operation.ptc_cost = n_new.cost
                    self.operation.init_sol = True
                    print(time.time()-self.start)
                if new_node:
                    self.AddTransitionNode(n_new)    
                self.FindLBTransitionNode(iter, True)
                return
            if new_node:
                self.AddTransitionNode(n_new)
        self.FindLBTransitionNode(iter)

    def KNearest(self, n_new_q_tensor:torch.tensor, n_new: Configuration, subtree_set: torch.tensor, k:int = 20) -> Tuple[List[Node], torch.tensor]:
        set_dists = batch_config_dist_torch(n_new_q_tensor, n_new, subtree_set, self.config.dist_type)
        _, indices = torch.sort(set_dists)
        indices = indices[:k] # indices of batch_subtree # select the nearest 10 neighbors
        N_near_batch = self.operation.active_mode.batch_subtree.index_select(0, indices)
        node_indices = self.operation.active_mode.node_idx_subtree.index_select(0,indices) # actual node indices (node.idx)
        if indices.size(0)== 1:
            self.operation.active_mode.node_idx_subtree[indices]
        n_near_costs = self.operation.costs.index_select(0,node_indices)
        return indices, N_near_batch, n_near_costs, node_indices

    def CreateTerminationCompositionNode(self, mode: List[int]):
        active_robots = self.env.get_goal_constrained_robots(mode)
        while True:
            terminal = []
            for i, r in enumerate(self.env.robots):
                key = self.g.get_key([r], mode[i])
                if r in active_robots:
                    terminal.append(self.g.transition_nodes[key][0].q)
                else:
                    terminal.append(np.random.choice(self.g.robot_nodes[key]).q)
            terminal = NpConfiguration.from_list(terminal)
            if self.env.is_collision_free(terminal.q, self.operation.active_mode.label):
                return terminal

    def FindParent(self, n_near: Node, n_new: Node, batch_cost: torch.tensor, batch_dist: torch.tensor) -> None:
        potential_cost = n_near.cost + batch_cost
        if n_new.cost > potential_cost:
            if self.env.is_edge_collision_free(
                n_near.state.q, n_new.state.q, self.operation.active_mode.label
            ):
                    
                if n_new.parent is not None:
                    n_new.parent.children.remove(n_new)
                n_new.parent = n_near
                n_new.cost_to_parent = batch_cost
                n_near.children.append(n_new) #Set child
                agent_dist = batch_dist.unsqueeze(0).to(dtype=torch.float16).cpu()
                n_new.agent_dists = n_new.parent.agent_dists + agent_dist
                n_new.agent_dists_to_parent = agent_dist
                self.operation.costs = self.operation.active_mode.ensure_capacity(self.operation.costs, n_new.idx) 
                n_new.cost = potential_cost
                return True
        return False
   
    def PlannerInitialization(self):
        #Initialization of graph
        q0 = self.env.get_start_pos()
        m0 = self.env.get_start_mode()
        mode_sequence = [m0]
        while True:
            if mode_sequence[-1] == self.env.terminal_mode:
                break
            mode_sequence.append(self.env.get_next_mode(None, mode_sequence[-1]))

        def sample_mode():
            m_rnd = random.choice(mode_sequence)
            return m_rnd

        def sample_uniform_valid_per_robot(num_pts):
            pts = []
            added_pts_dict = {}

            for i, r in enumerate(self.env.robots):
                robot_pts = 0
                while robot_pts < num_pts:
                    lims = self.env.limits[:, self.env.robot_idx[r]]

                    if lims[0, 0] < lims[1, 0]:
                        q = (
                            np.random.rand(self.env.robot_dims[r]) * (lims[1, :] - lims[0, :])
                            + lims[0, :]
                        )
                    else:
                        q = np.random.rand(self.env.robot_dims[r]) * 6 - 3
                    
                    m = sample_mode()
                    t = m[i]

                    if self.env.is_robot_env_collision_free([r], q, m):
                        pt = (r, q, t)
                        pts.append(pt) #TODO need to make sure its a general pt (not the same point for the same seperate graph)
                        robot_pts += 1

                        if tuple([r, t]) not in added_pts_dict:
                            added_pts_dict[tuple([r, t])] = 0

                        added_pts_dict[tuple([r, t])] += 1
            
            print(added_pts_dict)
            return pts

        def sample_uniform_valid_transition_for_active_robots():
            """Sample all transitions for active robots as vertices for the separate graph of each robot"""
            transitions = []
            for m in mode_sequence:
                goal_constrainted_robots = self.env.get_goal_constrained_robots(m)
                active_task = self.env.get_active_task(m)

                goal_sample = active_task.goal.sample()

                q = goal_sample
                t = m[self.env.robots.index(goal_constrainted_robots[0])]
                
                # print(f"checking colls for {goal_constrainted_robots}")
                if self.env.is_collision_free_for_robot(goal_constrainted_robots, q, m):
                    offset = 0
                    for r in goal_constrainted_robots:
                        dim = self.env.robot_dims[r]
                        q_transition = q[offset:offset+dim]
                        offset += dim
                        transition = (r, q_transition, t)
                        transitions.append(transition)
            return transitions

        self.g = ImplicitTensorGraph(State(q0, m0), self.env.robots, self.operation) # vertices of graph are in general position (= one position only appears once)

        # What is better: graph for each robot or graph for each robot per task(now) #TODO

        # sample transition
        transitions = sample_uniform_valid_transition_for_active_robots()
        for transition in transitions:
            r = transition[0]
            q = transition[1]
            task = transition[2]

            self.g.add_robot_node([r], [q], task, True)
        
        print('num transitions: ', len(transitions))

        # sample uniform
        new_states = sample_uniform_valid_per_robot(self.config.batch_size)
        for s in new_states:
            r = s[0]
            q = s[1]
            task = s[2]
            
            self.g.add_robot_node([r], [q], task, False)
       
        # Similar to RRT*
        active_mode = self.operation.modes[0]
        start_node = self.g.root
        active_mode.subtree.append(start_node)
        active_mode.batch_subtree[len(active_mode.subtree)-1, :] = start_node.q_tensor
        active_mode.node_idx_subtree[len(active_mode.subtree)-1] = start_node.idx
        start_node.cost = 0
        start_node.cost_to_parent = torch.tensor(0, device=device, dtype=torch.float32)
        self.operation.active_mode  = (np.random.choice(self.operation.modes, p= self.SetModePorbability()))
        self.g.whitelist.add(start_node)
        
    def RandomSample(self) -> Configuration: 
        q = []
        for i, robot in enumerate(self.env.robots):
            lims = self.operation.limits[robot]
            q.append(np.random.uniform(lims[0], lims[1]))
        # if np.random.uniform(0, 1) <= self.config.p_goal:
        #     for i, robot in enumerate(self.env.robots):
        #         lims = self.operation.limits[robot]
        #         q.append(np.random.uniform(lims[0], lims[1]))
        # else:
        #     for i, robot in enumerate(self.env.robots):
        #         task = self.operation.active_mode.label[i]
        #         if self.config.general_goal_sampling or robot in self.operation.active_mode.constrained_robots:
        #             goal = self.env.tasks[task].goal.sample()
        #             if len(goal) == self.env.robot_dims[robot]:
        #                 q.append(goal)
        #             else:
        #                 q.append(goal[self.env.robot_idx[robot]])
        #             continue
        #         lims = self.operation.limits[robot]
        #         q.append(np.random.uniform(lims[0], lims[1]))
        return  type(self.env.get_start_pos()).from_list(q)

    def Expand(self, iter:int):
        i = 0
        while i < self.config.expand_iter:
            i += 1
            self.operation.active_mode  = (np.random.choice(self.operation.modes, p= self.SetModePorbability()))
            q_rand = self.RandomSample()
            # get nearest node in tree
            n_nearest = self.Nearest(q_rand, self.operation.active_mode.subtree, self.operation.active_mode.batch_subtree[:len(self.operation.active_mode.subtree)])        
            self.DirectionOracle(q_rand, n_nearest, iter) 
    
    def DirectionOracle(self, q_rand, n_near, iter):
            candidate = []
            for i, r in enumerate(self.env.robots):
                q_near = NpConfiguration.from_numpy(n_near.state.q.state()[self.env.robot_idx[r]]) 
                if r not in n_near.neighbors:
                    n_near.neighbors[r] = self.g.get_robot_neighbors([r], q_near, self.operation.active_mode.label[i])
                
               
                best_neighbor = None
                min_angle = float('inf')
                vector_to_rand = q_rand.q[self.env.robot_idx[r]] - q_near.q
                if np.all(vector_to_rand == 0):# when its q_near itslef
                    candidate.append(q_near.q)
                    continue
                vector_to_rand = vector_to_rand / np.linalg.norm(vector_to_rand)

                for neighbor in n_near.neighbors[r]:
                    vector_to_neighbor = neighbor.q - q_near.q
                    if np.all(vector_to_neighbor == 0):# q_near = neighbor
                        continue
                    vector_to_neighbor = vector_to_neighbor / np.linalg.norm(vector_to_neighbor)
                    angle = np.arccos(np.clip(np.dot(vector_to_rand, vector_to_neighbor), -1.0, 1.0))
                    if angle < min_angle:
                        min_angle = angle
                        best_neighbor = neighbor.q
                candidate.append(best_neighbor)
            # self.SaveData(time.time()-self.start, n_rand= q_rand.q, n_nearest=n_near.state.q.state(), N_near_ = n_near.neighbors[r]) 
            candidate = NpConfiguration.from_list(candidate)
            # self.SaveData(time.time()-self.start, n_nearest = candidate.q) 
            n_candidate = Node(State(candidate, self.operation.active_mode.label), self.operation)
                # if n_candidate not in self.g.blacklist: TODO
                #     break
                # self.g.blacklist.add(n_candidate)
            existing_node = next((node for node in self.g.whitelist if node == n_candidate), None)
            if not existing_node:        
                if not self.env.is_edge_collision_free(n_near.state.q, candidate, self.operation.active_mode.label):
                    return
                self.g.whitelist.add(n_candidate)
                batch_cost, batch_dist =  batch_config_torch(n_candidate.q_tensor, n_candidate.state.q, n_near.q_tensor.unsqueeze(0), metric = self.config.cost_type)
                n_candidate.parent = n_near
                n_candidate.cost_to_parent = batch_cost
                n_near.children.append(n_candidate)
                self.operation.costs = self.operation.active_mode.ensure_capacity(self.operation.costs, n_candidate.idx) 
                n_candidate.cost = n_near.cost + batch_cost
                agent_dist = batch_dist[0].unsqueeze(0).to(dtype=torch.float16).cpu()
                n_candidate.agent_dists = n_candidate.parent.agent_dists + agent_dist
                n_candidate.agent_dists_to_parent = agent_dist
                self.UpdateMode(self.operation.active_mode, n_candidate, 'A')  
                self.ManageTransition(n_candidate, iter) #Check if we have reached a goal
            else:
                # #use existing node
                # N_near_indices, N_near_batch, n_near_costs, _ = self.Near(existing_node, self.operation.active_mode.batch_subtree[:len(self.operation.active_mode.subtree)])
                # batch_cost, batch_dist =  batch_config_torch(existing_node.q_tensor, existing_node.state.q, N_near_batch, metric = self.config.cost_type)
                batch_cost, batch_dist =  batch_config_torch(existing_node.q_tensor, existing_node.state.q, n_near.q_tensor.unsqueeze(0), metric = self.config.cost_type)
                if self.FindParent(n_near, existing_node, batch_cost, batch_dist):
                    self.UpdateCost(existing_node)
                # if self.Rewire(N_near_indices, existing_node, batch_cost, batch_dist, n_near_costs):
                #     self.UpdateCost(existing_node)
                self.ManageTransition(existing_node, iter, False) #Check if we have reached a goal
            
    def ConnectToTarget(self, iter:int):
        #TODO try to connect to termination node -> didn't do it with selected order like in paper
        # # select random termination node of created ones and try to connect
        new_node = True
        if self.operation.init_sol and self.operation.active_mode.label == self.env.terminal_mode:# WHen the termination node is restricted for all agents
            termination_node = np.random.choice(self.operation.active_mode.transition_nodes) # could be that we have several transition nodes
            terminal_q = termination_node.state.q
            terminal_q_tensor = termination_node.q_tensor
            new_node = False
        else:
            terminal_q = self.CreateTerminationCompositionNode(self.operation.active_mode.label)
            terminal_q_tensor = torch.as_tensor(terminal_q.q, device=device, dtype=torch.float32).unsqueeze(0)
            termination_node = Node(State(terminal_q, self.operation.active_mode.label), self.operation)
        
        N_near_indices, N_near_batch, n_near_costs, _ = self.KNearest(terminal_q_tensor, terminal_q, self.operation.active_mode.batch_subtree[:len(self.operation.active_mode.subtree)])
        batch_cost, batch_dist =  batch_config_torch(terminal_q_tensor, terminal_q, N_near_batch, metric = self.config.cost_type)
        c_terminal_costs = n_near_costs + batch_cost       
        _, sorted_mask = torch.sort(c_terminal_costs)
        for idx in sorted_mask:
            if termination_node.parent is None or termination_node.cost > c_terminal_costs[idx]:
                node = self.operation.active_mode.subtree[N_near_indices[idx]]
                if self.env.is_edge_collision_free(terminal_q, node.state.q, self.operation.active_mode.label):
                    if new_node:
                        termination_node = Node(State(terminal_q, self.operation.active_mode.label), self.operation)
                        self.g.whitelist.add(termination_node)
                    
                    termination_node.parent = node
                    termination_node.cost_to_parent = batch_cost[idx]
                    node.children.append(termination_node) #Set child
                    self.operation.costs = self.operation.active_mode.ensure_capacity(self.operation.costs, termination_node.idx) 
                    termination_node.cost = c_terminal_costs[idx]
                    agent_dist = batch_dist[idx].unsqueeze(0).to(dtype=torch.float16).cpu()
                    termination_node.agent_dists = termination_node.parent.agent_dists + agent_dist
                    termination_node.agent_dists_to_parent = agent_dist
                    if new_node:
                        self.UpdateMode(self.operation.active_mode, termination_node, 'A') 
                    self.ManageTransition(termination_node, iter, new_node)
                    return 

    def Plan(self) -> List[State]:
        i = 0
        self.PlannerInitialization()
        

        while True:
            i += 1
            # print(i)
            self.Expand(i)
            self.operation.active_mode  = (np.random.choice(self.operation.modes, p= self.SetModePorbability()))
            # if self.operation.init_sol:
            #     self.operation.active_mode = self.operation.modes[-1]
            self.ConnectToTarget(i)


            #Connect to target
            if self.operation.init_sol and i != self.operation.ptc_iter and (i- self.operation.ptc_iter)%self.config.ptc_max_iter == 0:
                diff = self.operation.ptc_cost - self.operation.cost
                self.operation.ptc_cost = self.operation.cost
                if diff < self.config.ptc_threshold:
                    break
            
            if i% 1000 == 0:
                if check_gpu_memory_usage():
                    break
            if i% 100 == 0:
                print(i)

            
         

        self.SaveData(time.time()-self.start)
        print(time.time()-self.start)
        # print('i', i)
        # print('tree', self.operation.tree)
        # print(torch.cuda.memory_summary())
        return self.operation.path    

