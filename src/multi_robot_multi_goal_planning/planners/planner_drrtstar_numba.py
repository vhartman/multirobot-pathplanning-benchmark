from multi_robot_multi_goal_planning.planners.rrtstar_base_numba import *

"""This file contains the dRRT based on the paper 'Finding a needle in an exponential haystack:
Discrete RRT for exploration of implicit
roadmaps in multi-robot motion planning' by K. Solovey et al."""

import numpy as np

from typing import List, Dict, Tuple, Optional
from numpy.typing import NDArray

import time

from multi_robot_multi_goal_planning.problems.planning_env import State
from multi_robot_multi_goal_planning.problems.configuration import (
    NpConfiguration,
    batch_config_dist,
)

class ImplicitTensorGraph:
    robots: List[str]

    def __init__(self, robots):
        self.robots = robots
        self.batch_dist_fun = batch_config_dist

        self.robot_nodes = {}
        self.transition_nodes = {}

        self.goal_nodes = []
        self.per_robot_task_to_goal_lb_cost = {}

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

class dRRTstar(BaseRRTstar):
    def __init__(self, env, config: ConfigManager):
        super().__init__(env, config)
        self.g = None
        self.mode_sequence = []
        

    def initialize_mode(self, q:Optional[Configuration]=None, mode:Mode=None) -> None: #TODO entry_configuration needs to be specified
        """Initializes a new mode"""
        if mode is None: 
            new_mode = self.env.start_mode
            new_mode.prev_mode = None
        else:
            new_mode = self.env.get_next_mode(q, mode)
            new_mode.prev_mode = mode
        self.mode_sequence.append(new_mode)
    
    def add_new_mode(self, mode:Mode=None, tree_instance: Optional[Union["SingleTree", "BidirectionalTree"]] = None) -> None: #TODO entry_configuration needs to be specified
        """Initializes a new mode"""
        if mode is None:
            new_mode = self.mode_sequence[0]
        else:
            new_mode = self.mode_sequence[mode.id +1]
        self.modes.append(new_mode)
        self.add_tree(new_mode, tree_instance)

    def UpdateCost(self, n:Node) -> None:
        stack = [n]
        while stack:
            current_node = stack.pop()
            children = current_node.children
            if children:
                for _, child in enumerate(children):
                    child.cost = current_node.cost + child.cost_to_parent
                stack.extend(children)
   
    def ManageTransition(self, mode:Mode, n_new: Node, iter: int, new_node:bool = True) -> None:
        #check if transition is reached
        if self.env.is_transition(n_new.state.q, mode):
            if not self.operation.init_sol and mode.__eq__(self.modes[-1]):
                    self.add_new_mode(mode, SingleTree)
                    if new_node:
                        self.convert_node_to_transition_node(mode, n_new)
                    self.InformedInitialization(self.modes[-1])
                    return
            if new_node:
                self.convert_node_to_transition_node(mode, n_new)
        #check if termination is reached
        if self.env.done(n_new.state.q, mode):
            if new_node:
                self.convert_node_to_transition_node(mode, n_new)
            if not self.operation.init_sol:
                print(time.time()-self.start)
                self.operation.init_sol = True
        self.FindLBTransitionNode(iter)

    def KNearest(self, mode:Mode, n_new_q_tensor:torch.tensor, n_new: Configuration, k:int = 20) -> Tuple[List[Node], torch.tensor]:
        batch_subtree = self.trees[mode].get_batch_subtree()
        set_dists = batch_config_dist(n_new.state.q, batch_subtree, self.config.dist_type)
        indices = np.argsort(set_dists)
        indices = indices[:k]
        N_near_batch = batch_subtree.index_select(0, indices)
        node_indices = self.trees[mode].node_idx_subtree.index_select(0,indices) # actual node indices (node.id)
        n_near_costs = self.operation.costs.index_select(0,node_indices)
        return N_near_batch, n_near_costs, node_indices   

    def sample_transition_composition_node(self, mode)-> Configuration:
        """Returns transition node of mode"""
        constrained_robot = self.env.get_active_task(mode, None).robots
        while True:
            q = []
            for i, r in enumerate(self.env.robots):
                key = self.g.get_key([r], mode.task_ids[i])
                if r in constrained_robot:
                    q.append(self.g.transition_nodes[key][0].q)
                else:
                    q.append(np.random.choice(self.g.robot_nodes[key]).q)
            q = NpConfiguration.from_list(q)
            if self.env.is_collision_free(q, mode):
                return q     

    def FindParent(self, mode:Mode, n_near: Node, n_new: Node, batch_cost: torch.tensor) -> None:
        potential_cost = n_near.cost + batch_cost
        if n_new.cost > potential_cost:
            if self.env.is_edge_collision_free(
                n_near.state.q, n_new.state.q, mode
            ):
                    
                if n_new.parent is not None:
                    n_new.parent.children.remove(n_new)
                n_new.parent = n_near
                n_new.cost_to_parent = batch_cost
                n_near.children.append(n_new) #Set child
                self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, n_new.id) 
                n_new.cost = potential_cost
                return True
        return False
   
    def PlannerInitialization(self):
        #initialization of graph
        #all possible modes for this environment
        self.initialize_mode()
        while True:
            mode = self.mode_sequence[-1]
            if self.env.is_terminal_mode(mode):
                break
            self.initialize_mode(None, mode)
            
        def sample_uniform_valid_per_robot(num_pts:int):
            pts = []
            added_pts_dict = {}
            for t, task in enumerate(self.env.tasks):
                for r in task.robots:
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

                        if self.env.is_robot_env_collision_free([r], q, mode):
                            pt = (r, q, t)
                            pts.append(pt) #TODO need to make sure its a general pt (not the same point for the same seperate graph)
                            robot_pts += 1

                            if tuple([r, t]) not in added_pts_dict:
                                added_pts_dict[tuple([r, t])] = 0

                            added_pts_dict[tuple([r, t])] += 1
            
            print(added_pts_dict)
            for s in pts:
                r = s[0]
                q = s[1]
                task = s[2]
                
                self.g.add_robot_node([r], [q], task, False)

        def sample_goal_for_active_robots():
            """Sample goals for active robots as vertices for corresponding separate graph"""
            transitions = []
            for m in self.mode_sequence:
                active_task = self.env.get_active_task(m, None)
                constrained_robot = active_task.robots

                q = active_task.goal.sample(m)
                t = m.task_ids[self.env.robots.index(constrained_robot[0])]
                
                # print(f"checking colls for {goal_constrainted_robots}")
                if self.env.is_collision_free_for_robot(constrained_robot, q, m):
                    offset = 0
                    for r in constrained_robot:
                        dim = self.env.robot_dims[r]
                        q_transition = q[offset:offset+dim]
                        offset += dim
                        transition = (r, q_transition, t)
                        transitions.append(transition)
            for s in transitions:
                r = s[0]
                q = s[1]
                task = s[2]
                
                self.g.add_robot_node([r], [q], task, True)

        self.g = ImplicitTensorGraph(self.env.robots) # vertices of graph are in general position (= one position only appears once)
        #sample task goal
        sample_goal_for_active_robots()
        # sample uniform
        sample_uniform_valid_per_robot(self.config.batch_size)
        
        # Similar to RRT*
        # Initilaize first Mode
        self.add_new_mode(tree_instance=SingleTree)
        active_mode = self.modes[-1]
        self.InformedInitialization(active_mode)
        # Create start node
        start_state = State(self.env.get_start_pos(), active_mode)
        start_node = Node(start_state, self.operation)
        self.trees[active_mode].add_node(start_node)
        start_node.cost = 0
        start_node.cost_to_parent = np.float32(0)
        
    def Expand(self, iter:int):
        i = 0
        while i < self.config.expand_iter:
            i += 1
            active_mode  = self.RandomMode(Mode.id_counter)
            q_rand = self.sample_configuration(active_mode, 0)
            #get nearest node in tree8:
            n_nearest, _ = self.Nearest(active_mode, q_rand)
            self.DirectionOracle(active_mode, q_rand, n_nearest, iter) 

    def CheckForExistingNode(self, mode:Mode, n: Node, tree: str = ''):
        # q_tensor = torch.as_tensor(q_rand.state(), device=device, dtype=torch.float32).unsqueeze(0)
        set_dists = batch_config_dist(n.state.q, self.trees[mode].get_batch_subtree(tree), 'euclidean')
        # set_dists = batch_dist_torch(n.q_tensor.unsqueeze(0), n.state.q, self.trees[mode].get_batch_subtree(tree), self.config.dist_type)
        idx = np.argmin(set_dists)
        if set_dists[idx] < 1e-100:
            node_id = self.trees[mode].get_node_idx_subtree(tree)[idx]
            return  self.trees[mode].get_node(node_id, tree)
        return None
    
    def Extend(self, mode:Mode, n_nearest_b:Node, n_new:Node, dist )-> Optional[Node]:
        q = n_new.state.q
        #RRT not RRT*
        i = 1
        while True:
            state_new = self.Steer(mode, n_nearest_b, q, dist, i)
            if not state_new or np.equal(state_new.q.state(), q.state()).all(): # Reached
                # self.SaveData(mode, time.time()-self.start, n_nearest = n_nearest_b.state.q.state(), n_rand = q.state(), n_new = n_new.state.q.state())
                return n_nearest_b
            # self.SaveData(mode, time.time()-self.start, n_nearest = n_nearest_b.state.q.state(), n_rand = q.state(), n_new = state_new.q.state())
            if self.env.is_collision_free(state_new.q, mode) and self.env.is_edge_collision_free(n_nearest_b.state.q, state_new.q, mode):
                # Add n_new to tree
        
                n_new = Node(state_new,self.operation)
                
                cost =  batch_config_cost([n_new.state], [n_nearest_b.state], metric = "euclidean", reduction="max")
                c_min = n_nearest_b.cost + cost

                n_new.parent = n_nearest_b
                n_new.cost_to_parent = cost
                n_nearest_b.children.append(n_new) #Set child
                self.trees[mode].add_node(n_new) 
                self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, n_new.id) 
                n_new.cost = c_min
                n_nearest_b = n_new
                i +=1
            else:
                return 
             
    def DirectionOracle(self, mode:Mode, q_rand:Configuration, n_near:Node, iter:int) -> None:
            candidate = []
            for i, r in enumerate(self.env.robots):
                # select configuration of robot r
                q_near = NpConfiguration.from_numpy(n_near.state.q.robot_state(i)) 
                if r not in n_near.neighbors:
                    n_near.neighbors[r] = self.g.get_robot_neighbors([r], q_near, mode.task_ids[i])
                
                best_neighbor = None
                min_angle = np.inf
                vector_to_rand = q_rand.q[self.env.robot_idx[r]] - q_near.q
                if np.all(vector_to_rand == 0):# when its q_near itslef
                    candidate.append(q_near.q)
                    continue
                vector_to_rand = vector_to_rand / np.linalg.norm(vector_to_rand)

                for _ , neighbor in enumerate(n_near.neighbors[r]):
                    vector_to_neighbor = neighbor.q - q_near.q
                    if np.all(vector_to_neighbor == 0):# q_near = neighbor
                        continue
                    vector_to_neighbor = vector_to_neighbor / np.linalg.norm(vector_to_neighbor)
                    angle = np.arccos(np.clip(np.dot(vector_to_rand, vector_to_neighbor), -1.0, 1.0))
                    # print(float(angle))
                    if angle < min_angle:
                        min_angle = angle
                        best_neighbor = neighbor.q
                candidate.append(best_neighbor)
            # self.SaveData(time.time()-self.start, n_rand= q_rand.q, n_nearest=n_near.state.q.state(), N_near_ = n_near.neighbors[r]) 
            candidate = NpConfiguration.from_list(candidate)
            # self.SaveData(time.time()-self.start, n_nearest = candidate.q) 
            n_candidate = Node(State(candidate, mode), self.operation)
                # if n_candidate not in self.g.blacklist: TODO
                #     break
                # self.g.blacklist.add(n_candidate)
            existing_node = self.CheckForExistingNode(mode, n_candidate)
            
            #node doesn't exist yet
            # existing_node = False 
            if not existing_node:        
                if not self.env.is_edge_collision_free(n_near.state.q, candidate, mode):
                    return
                batch_cost = batch_config_cost([n_candidate.state], [n_near.state], metric = "euclidean", reduction="max")
                n_candidate.parent = n_near
                n_candidate.cost_to_parent = batch_cost
                n_near.children.append(n_candidate)
                self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, n_candidate.id) 
                n_candidate.cost = n_near.cost + batch_cost
                self.trees[mode].add_node(n_candidate)
                self.ManageTransition(mode, n_candidate, iter) #Check if we have reached a goal
            else:
                #reuse existing node
                # existing_node_ = self.CheckForExistingNode(mode, n_candidate)
                batch_cost = batch_config_cost([existing_node.state], [n_near.state], metric = "euclidean", reduction="max")
                if self.FindParent(mode, n_near, existing_node, batch_cost):
                    self.UpdateCost(existing_node)
                # N_near_batch, n_near_costs, node_indices = self.Near(mode, existing_node)
                # batch_cost, batch_dist =  batch_cost_torch(existing_node.q_tensor, existing_node.state.q, N_near_batch, metric = self.config.cost_type)
                # if self.Rewire(mode, node_indices, existing_node, batch_cost, batch_dist, n_near_costs):
                #     self.UpdateCost(existing_node)
                # if self.Rewire(N_near_indices, existing_node, batch_cost, batch_dist, n_near_costs):
                #     self.UpdateCost(existing_node)
                #need to get trasntion node?
                self.FindLBTransitionNode(iter)
                # self.ManageTransition(mode, existing_node, iter, False) #Check if we have reached a goal
            
    def ConnectToTarget(self, mode:Mode, iter:int):
        """Local connector: Tries to connect to a termination node in mode"""
        #Not implemented as described in paper which uses a selected order
        # # select random termination node of created ones and try to connect
        new_node = True
        if self.operation.init_sol and self.env.is_terminal_mode(mode):
            #when termination node is restricted for all agents -> don't create a new transition node            
            node_id = np.random.choice(self.transition_node_ids[mode] )
            termination_node = self.trees[mode].subtree.get(node_id)
            terminal_q = termination_node.state.q
            new_node = False
        else:
            terminal_q = self.sample_transition_composition_node(mode)
            termination_node = Node(State(terminal_q, mode), self.operation)
        
        # N_near_batch, n_near_costs, node_indices = self.KNearest(mode, terminal_q_tensor.unsqueeze(0), terminal_q) #TODO
        N_near_batch, n_near_costs, node_indices = self.Near(mode, termination_node)
        batch_cost = batch_config_cost(termination_node.state.q, N_near_batch, metric = "euclidean", reduction="max")
        c_terminal_costs = n_near_costs + batch_cost       
        sorted_mask = np.argsort(c_terminal_costs)
        for idx in sorted_mask:
            if termination_node.parent is None or termination_node.cost > c_terminal_costs[idx]:
                node = self.trees[mode].subtree.get(node_indices[idx].item())
                dist = batch_config_dist(node.state.q, [termination_node.state.q], self.config.dist_type)
                # dist = batch_dist_torch(node.q_tensor, node.state.q, termination_node.q_tensor.unsqueeze(0), self.config.dist_type)
                n_nearest = self.Extend(mode, node, termination_node, dist)
                if n_nearest is not None:
                    if self.env.is_edge_collision_free(n_nearest.state.q, terminal_q,  mode):
                        cost = batch_config_cost([n_nearest.state], [termination_node.state], metric = "euclidean", reduction="max")
                        if termination_node.parent is not None:
                            termination_node.parent.children.remove(termination_node)
                        termination_node.parent = n_nearest
                        termination_node.cost_to_parent = cost
                        n_nearest.children.append(termination_node) #Set child
                        if new_node:
                            self.trees[mode].add_node(termination_node)
                            self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, termination_node.id) 
                        termination_node.cost = n_nearest.cost + cost
                        self.ManageTransition(mode, termination_node, iter, new_node)
                        return 

    def Plan(self) -> List[State]:
        i = 0
        self.PlannerInitialization()
        

        while True:
            i += 1
            self.Expand(i)
            active_mode  = self.RandomMode(Mode.id_counter)
            self.ConnectToTarget(active_mode, i)
            
            if self.PTC(i):
                "PTC applied"
                self.SaveFinalData()
                break



            
         

        self.SaveData(active_mode, time.time()-self.start)
        print(time.time()-self.start)
        return self.operation.path    

