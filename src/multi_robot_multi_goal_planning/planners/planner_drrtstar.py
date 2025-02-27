import numpy as np
import time as time
import math as math
from typing import Tuple, Optional, Union, List
from numpy.typing import NDArray
from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
    Mode
)
from multi_robot_multi_goal_planning.planners.rrtstar_base import (
    BaseRRTstar, 
    Node, 
    SingleTree,
    BidirectionalTree

)
from multi_robot_multi_goal_planning.planners.planner_rrtstar import (
    RRTstar

)
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
)
from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
    NpConfiguration,
    batch_config_dist,  
    
)
from multi_robot_multi_goal_planning.problems.planning_env import (
    SingleGoal
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
        if key not in self.robot_nodes:
            self.robot_nodes[key] = []
        if is_transition:
            if key not in self.transition_nodes:
                self.transition_nodes[key] = []

            does_already_exist = False
            for n in self.transition_nodes[key]:
                if np.linalg.norm(np.concatenate(q) - n.state()) < 1e-5:
                    does_already_exist = True
                    break

            if not does_already_exist:
                q = NpConfiguration.from_list(q)
                self.transition_nodes[key].append(q)
                self.robot_nodes[key].append(q)
        else:
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
    """Represents the class for the dRRT* based planner"""
    def __init__(self, 
                 env: BaseProblem,
                 ptc: PlannerTerminationCondition,  
                 general_goal_sampling: bool = False, 
                 informed_sampling: bool = False, 
                 informed_sampling_version: int = 6, 
                 distance_metric: str = 'max_euclidean',
                 p_goal: float = 0.1, 
                 p_stay: float = 0.0,
                 p_uniform: float = 0.2, 
                 shortcutting: bool = False, 
                 mode_sampling: Optional[Union[int, float]] = None, 
                 gaussian: bool = False,
                 locally_informed_sampling:bool = True, 
                 remove_redundant_nodes:bool = True,
                 sample_batch_size_per_task:int = 500,
                 transistion_batch_size_per_mode:int = 200, 
                 expand_iter:int = 10
                 
                ):
        super().__init__(env, ptc, general_goal_sampling, informed_sampling, informed_sampling_version, distance_metric,
                    p_goal, p_stay, p_uniform, shortcutting, mode_sampling, 
                    gaussian, locally_informed_sampling = locally_informed_sampling, remove_redundant_nodes = remove_redundant_nodes)
        self.g = None
        self.sample_batch_size_per_task = sample_batch_size_per_task
        self.transistion_batch_size_per_mode = transistion_batch_size_per_mode
        self.expand_iter = expand_iter
        self.conf_type = type(env.get_start_pos())
    
    def sample_uniform_valid_per_robot(self, modes:List[Mode]):
            pts = []
            added_pts_dict = {}
            for mode in modes:
                for i, r in enumerate(self.env.robots):
                    t = mode.task_ids[i]
                    key = tuple([r, t])
                    if key not in added_pts_dict:
                        added_pts_dict[key] = 0
                    task = self.env.tasks[t]
                    lims = self.env.limits[:, self.env.robot_idx[r]]
                    while added_pts_dict[key] < self.sample_batch_size_per_task:
                        q = np.random.uniform(lims[0], lims[1])
                        if self.env.is_robot_env_collision_free([r], q, mode):
                            pt = (r, q, t)
                            pts.append(pt) #TODO need to make sure its a general pt (not the same point for the same seperate graph)
                            added_pts_dict[key] += 1
            
            print(added_pts_dict)
            
            for s in pts:
                r = s[0]
                q = s[1]
                task = s[2]
                
                self.g.add_robot_node([r], [q], task, False)

    def sample_goal_for_active_robots(self, modes:List[Mode]):
        """Sample goals for active robots as vertices for corresponding separate graph"""
        
        for m in modes:
            transitions = []
            while  len(transitions) < self.transistion_batch_size_per_mode:
                next_ids = self.get_next_ids(m)
                active_task = self.env.get_active_task(m, next_ids)
                if len(transitions) > 0 and len(self.env.get_valid_next_task_combinations(m)) <= 1 and type(active_task.goal) is SingleGoal:
                    break
                constrained_robots = active_task.robots
                q = active_task.goal.sample(m)
                t = m.task_ids[self.env.robots.index(constrained_robots[0])]
                if self.env.is_collision_free_for_robot(constrained_robots, q, m):
                    offset = 0
                    for r in constrained_robots:
                        dim = self.env.robot_dims[r]
                        q_transition = q[offset:offset+dim]
                        offset += dim
                        transition = (r, q_transition, t)
                        transitions.append(transition)
            for t in transitions:
                r = t[0]
                q = t[1]
                task = t[2]
                
                self.g.add_robot_node([r], [q], task, True)

    def add_samples_to_graph(self, modes:Optional[List[Mode]]=None):  
        if modes is None:
            modes = self.modes
        #sample task goal
        self.sample_goal_for_active_robots(modes)
        # sample uniform
        self.sample_uniform_valid_per_robot(modes)

    def add_new_mode(self, 
                     q:Configuration=None, 
                     mode:Mode=None, 
                     tree_instance: Optional[Union["SingleTree", "BidirectionalTree"]] = None
                     ) -> None:
        """
        Initializes a new mode (including its corresponding tree instance and performs informed initialization).

        Args:
            q (Configuration): Configuration used to determine the new mode. 
            mode (Mode): The current mode from which to get the next mode. 
            tree_instance (Optional[Union["SingleTree", "BidirectionalTree"]]): Type of tree instance to initialize for the next mode. Must be either SingleTree or BidirectionalTree.

        Returns:
            None: This method does not return any value.
        """
        if mode is None: 
            new_mode = self.env.make_start_mode()
            new_mode.prev_mode = None
        else:
            new_mode = self.env.get_next_mode(q, mode)
            new_mode.prev_mode = mode
        if new_mode in self.modes:
            return 
        self.modes.append(new_mode)
        self.add_tree(new_mode, tree_instance)
        self.InformedInitialization(new_mode)
        self.add_samples_to_graph([new_mode])

    def UpdateCost(self, mode:Mode, n:Node) -> None:
       return RRTstar.UpdateCost(self, mode, n)
      
    def ManageTransition(self, mode:Mode, n_new: Node) -> None:
        RRTstar.ManageTransition(self, mode, n_new)

    def KNearest(self, mode:Mode, n_new: Configuration, k:int = 20) -> Tuple[List[Node], NDArray]:
        batch_subtree = self.trees[mode].get_batch_subtree()
        set_dists = batch_config_dist(n_new.state.q, batch_subtree, self.distance_metric)
        indices = np.argsort(set_dists)
        indices = indices[:k]
        N_near_batch = batch_subtree.index_select(0, indices)
        node_indices = self.trees[mode].node_idx_subtree.index_select(0,indices) # actual node indices (node.id)
        n_near_costs = self.operation.costs.index_select(0,node_indices)
        return N_near_batch, n_near_costs, node_indices   

    # def ChangeParent(self, mode:Mode, n_near: Node, n_new: Node, batch_cost: NDArray) -> None:
    #     potential_cost = n_near.cost + batch_cost
    #     if n_new.cost > potential_cost:
    #         if self.env.is_edge_collision_free(
    #             n_near.state.q, n_new.state.q, mode
    #         ):
                    
    #             if n_new.parent is not None:
    #                 n_new.parent.children.remove(n_new)
    #             n_new.parent = n_near
    #             n_new.cost_to_parent = batch_cost
    #             n_near.children.append(n_new) #Set child
    #             self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, n_new.id) 
    #             n_new.cost = potential_cost
    #             return True
    #     return False
    
    def ChangeParent(self, 
                   mode:Mode, 
                   node_indices: NDArray, 
                   n_new: Node, 
                   batch_cost: NDArray, 
                   n_near_costs: NDArray
                   ) -> None:
        """
        Sets the optimal parent for a new node by evaluating connection costs among candidate nodes.

        Args:
            mode (Mode): Current operational mode.
            node_indices (NDArray): Array of IDs representing candidate neighboring nodes.
            n_new (Node): New node that needs a parent connection.
            n_nearest (Node): Nearest candidate node to n_new.
            batch_cost (NDArray): Costs associated from n_new to all candidate neighboring nodes.
            n_near_costs (NDArray): Cost values for all candidate neighboring nodes.

        Returns:
            None: This method does not return any value.
        """
        c_new_tensor = n_near_costs + batch_cost
        valid_mask = c_new_tensor < n_new.cost
        if np.any(valid_mask):
            sorted_indices = np.where(valid_mask)[0][np.argsort(c_new_tensor[valid_mask])]
            for idx in sorted_indices:
                node = self.trees[mode].subtree.get(node_indices[idx].item())
                if self.env.is_edge_collision_free(node.state.q, n_new.state.q, mode):
                    self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, n_new.id) 
                    n_new.cost = c_new_tensor[idx]
                    n_new.cost_to_parent = batch_cost[idx]   
                    if n_new.parent is not None:
                        n_new.parent.children.remove(n_new)
                    n_new.parent = node     
                    node.children.append(n_new)
                    self.UpdateCost(mode, n_new)                       
                    return
   
    def PlannerInitialization(self):   
        self.g = ImplicitTensorGraph(self.env.robots) # vertices of graph are in general position (= one position only appears once)
        # Similar to RRT*
        # Initilaize first Mode
        self.set_gamma_rrtstar()
        self.add_new_mode(tree_instance=SingleTree)
        active_mode = self.modes[-1]
        self.InformedInitialization(active_mode)
        # Create start node
        start_state = State(self.env.get_start_pos(), active_mode)
        start_node = Node(start_state, self.operation)
        self.trees[active_mode].add_node(start_node)
        start_node.cost = 0
        start_node.cost_to_parent = 0.0
        
    def Expand(self, iter:int):
        i = 0
        while i < self.expand_iter:
            i += 1
            active_mode  = self.RandomMode()
            q_rand = self.sample_configuration(active_mode, 0)
            #get nearest node in tree8:
            n_nearest, _ , _, _= self.Nearest(active_mode, q_rand)
            self.DirectionOracle(active_mode, q_rand, n_nearest, iter) 

    def CheckForExistingNode(self, mode:Mode, n: Node, tree: str = ''):
        # q_tensor = torch.as_tensor(q_rand.state(), device=device, dtype=torch.float32).unsqueeze(0)
        set_dists = batch_config_dist(n.state.q, self.trees[mode].get_batch_subtree(tree), 'euclidean')
        # set_dists = batch_dist_torch(n.q_tensor.unsqueeze(0), n.state.q, self.trees[mode].get_batch_subtree(tree), self.distance_metric)
        idx = np.argmin(set_dists)
        if set_dists[idx] < 1e-100:
            node_id = self.trees[mode].get_node_ids_subtree(tree)[idx]
            return  self.trees[mode].get_node(node_id, tree)
        return None
    
    def Extend(self, mode:Mode, n_nearest_b:Node, n_new:Node, dist )-> Optional[Node]:
        q = n_new.state.q
        #RRT not RRT*
        i = 1
        while True:
            state_new = self.Steer(mode, n_nearest_b, q, dist, i)
            if not state_new or np.equal(state_new.q.state(), q.state()).all(): # Reached
                # self.SaveData(mode, time.time()-self.start_time, n_nearest = n_nearest_b.state.q.state(), n_rand = q.state(), n_new = n_new.state.q.state())
                return n_nearest_b
            # self.SaveData(mode, time.time()-self.start_time, n_nearest = n_nearest_b.state.q.state(), n_rand = q.state(), n_new = state_new.q.state())
            if self.env.is_collision_free(state_new.q, mode) and self.env.is_edge_collision_free(n_nearest_b.state.q, state_new.q, mode):
                # Add n_new to tree
        
                n_new = Node(state_new,self.operation)
                
                cost =  self.env.batch_config_cost([n_new.state], [n_nearest_b.state])
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
            # self.SaveData(time.time()-self.start_time, n_rand= q_rand.q, n_nearest=n_near.state.q.state(), N_near_ = n_near.neighbors[r]) 
            candidate = NpConfiguration.from_list(candidate)
            # self.SaveData(time.time()-self.start_time, n_nearest = candidate.q) 
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
                batch_cost = self.env.batch_config_cost([n_candidate.state], [n_near.state])
                n_candidate.parent = n_near
                n_candidate.cost_to_parent = batch_cost
                n_near.children.append(n_candidate)
                self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, n_candidate.id) 
                n_candidate.cost = n_near.cost + batch_cost
                self.trees[mode].add_node(n_candidate)
                _, _, set_dists, n_nearest_idx = self.Nearest(mode, n_candidate.state.q) 
                N_near_batch, n_near_costs, node_indices = self.Near(mode, n_candidate, n_nearest_idx, set_dists )
                batch_cost = self.env.batch_config_cost(n_candidate.state.q, N_near_batch)
                if self.Rewire(mode, node_indices, n_candidate, batch_cost, n_near_costs):
                    self.UpdateCost(mode,n_candidate)
                self.ManageTransition(mode, n_candidate) #Check if we have reached a goal
            else:
                #reuse existing node
                # existing_node_ = self.CheckForExistingNode(mode, n_candidate) # TODO make sure n_near is in neighbors
                # _, _, set_dists, n_nearest_idx = self.Nearest(mode, existing_node.state.q)  #TODO not needed?
                idx =  np.where(self.trees[mode].get_node_ids_subtree() == n_near.id)[0][0] 
                N_near_batch, n_near_costs, node_indices = self.Near(mode, existing_node, idx)
                batch_cost = self.env.batch_config_cost(existing_node.state.q, N_near_batch)
                # batch_cost = self.env.batch_config_cost([existing_node.state], [n_near.state])
                self.ChangeParent(mode, node_indices, existing_node, batch_cost, n_near_costs)
                if self.Rewire(mode, node_indices, existing_node, batch_cost, n_near_costs):
                    self.UpdateCost(mode, existing_node)
            self.FindLBTransitionNode()
                       
    def ConnectToTarget(self, mode:Mode, iter:int):
        """Local connector: Tries to connect to a transition node in mode"""
        #Not implemented as described in paper which uses a selected order
        # # select random termination node of created ones and try to connect
        new_node = True
        if self.operation.init_sol and self.env.is_terminal_mode(mode):
            #when termination node is restricted for all agents -> don't create a new transition node            
            node_id = np.random.choice(self.transition_node_ids[mode])
            termination_node = self.trees[mode].subtree.get(node_id)
            terminal_q = termination_node.state.q
            new_node = False
        else:
            terminal_q = self.sample_transition_configuration(mode)
            termination_node = Node(State(terminal_q, mode), self.operation)
            self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, termination_node.id)
            termination_node.cost = np.inf

        
        # N_near_batch, n_near_costs, node_indices = self.KNearest(mode, terminal_q_tensor.unsqueeze(0), terminal_q) #TODO
        _, _, set_dists, n_nearest_idx = self.Nearest(mode, termination_node.state.q) 
        N_near_batch, n_near_costs, node_indices = self.Near(mode, termination_node, n_nearest_idx, set_dists)
        batch_cost = self.env.batch_config_cost(termination_node.state.q, N_near_batch)
        c_terminal_costs = n_near_costs + batch_cost       
        sorted_mask = np.argsort(c_terminal_costs)
        for idx in sorted_mask:
            if termination_node.cost > c_terminal_costs[idx]:
                node = self.trees[mode].get_node(node_indices[idx].item())
                dist = batch_config_dist(node.state.q, [termination_node.state.q], self.distance_metric)
                # dist = batch_dist_torch(node.q_tensor, node.state.q, termination_node.q_tensor.unsqueeze(0), self.distance_metric)
                n_nearest = self.Extend(mode, node, termination_node, dist)
                if n_nearest is not None:
                    if self.env.is_edge_collision_free(n_nearest.state.q, terminal_q,  mode):
                        cost = self.env.batch_config_cost([n_nearest.state], [termination_node.state])
                        if termination_node.parent is not None:
                            termination_node.parent.children.remove(termination_node)
                        termination_node.parent = n_nearest
                        termination_node.cost_to_parent = cost
                        n_nearest.children.append(termination_node) #Set child
                        if new_node:
                            self.trees[mode].add_node(termination_node)
                        termination_node.cost = n_nearest.cost + cost
                        _, _, set_dists, n_nearest_idx = self.Nearest(mode, termination_node.state.q)  
                        N_near_batch, n_near_costs, node_indices = self.Near(mode, termination_node, n_nearest_idx, set_dists)
                        batch_cost = self.env.batch_config_cost(termination_node.state.q, N_near_batch)
                        if self.Rewire(mode, node_indices, termination_node, batch_cost, n_near_costs):
                            self.UpdateCost(mode, termination_node)
                        self.ManageTransition(mode, termination_node)
                        return 

    def Plan(self) -> List[State]:
        i = 0
        self.PlannerInitialization()
        
        while True:
            i += 1
            self.Expand(i)
            active_mode  = self.RandomMode()
            self.ConnectToTarget(active_mode, i)
            if self.operation.init_sol and i %100 == 0: # make it better!
                self.add_samples_to_graph() 
            if self.ptc.should_terminate(i, time.time() - self.start_time):
                break

        self.costs.append(self.operation.cost)
        self.times.append(time.time() - self.start_time)
        self.all_paths.append(self.operation.path)
        info = {"costs": self.costs, "times": self.times, "paths": self.all_paths}
        return self.operation.path, info    

