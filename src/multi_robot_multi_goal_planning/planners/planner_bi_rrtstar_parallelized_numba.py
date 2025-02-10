from multi_robot_multi_goal_planning.planners.planner_bi_rrtstar_numba import *

"""
This file contains the most important changes for the parallelized Bi-RRT* algorithm compared to the original RRT*. There are 2 versions possible: 
Version 1 is inspired by the paper 'Bi-RRT*: An Improved Bidirectional RRT* Path Planner for Robot in Two-Dimensional Space' by B. Wang et al. 
and version 2 is inspired by the paper 'RRT-Connect: An Efficient Approach to Single-Query Path Planning' by J.J Kuffner et al. and 
by the paper 'RRT*-Connect: Faster, Asymptotically Optimal Motion Planning' by S. Klemm et al. 
"""


class ParallelizedBidirectionalRRTstar(BidirectionalRRTstar):
    def __init__(self, env, config: ConfigManager):
        super().__init__(env, config)

    def PlannerInitialization(self) -> None:
        # Initilaize first Mode
        self.add_new_mode(tree_instance=BidirectionalTree)
        mode = self.modes[-1]
        # Create start node
        start_state = State(self.env.start_pos, mode)
        start_node = Node(start_state, self.operation)
        self.trees[mode].add_node(start_node)
        start_node.cost = 0
        start_node.cost_to_parent = np.float32(0)
        #Initialize other modes:
        while True:
            self.InformedInitialization(mode)
            if not self.env.is_terminal_mode(mode): 
                self.add_new_mode(mode=mode, tree_instance=BidirectionalTree) 
            for _ in range(self.config.transition_nodes):                 
                q = self.sample_transition_configuration(mode)
                node = Node(State(q, mode), self.operation)
                node.cost_to_parent = np.float32(0)
                self.mark_node_as_transition(mode, node)
                self.trees[mode].add_node(node, 'B')
                self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, node.id) 
                node.cost = np.inf
                if self.env.is_terminal_mode(mode): 
                    return
                self.trees[self.modes[-1]].add_transition_node_as_start_node(node)
            mode = self.modes[-1]
          
    def Connect(self, mode:Mode, n_new:Node, iter:int) -> None:
        if not self.trees[mode].subtree_b:
            return
        n_nearest_b, dist = self.Nearest(mode, n_new.state.q, 'B')

        if self.config.birrtstar_version == 1 or self.config.birrtstar_version == 2 and self.trees[mode].connected: #Based on paper Bi-RRT* by B. Wang 
            #TODO only check dist of active robots to connect (cost can be extremly high)? or the smartest way to just connect when possible?
          
          
            cost =  batch_config_cost([n_new.state], [n_nearest_b.state], metric = self.config.cost_metric, reduction=self.config.cost_reduction)
            # relevant_dists = []
            # for r_idx, r in enumerate(self.env.robots):
            #     if r in constrained_robots:
            #         relevant_dists.append(dists[0][r_idx].item())
            # if np.max(relevant_dists) > self.config.step_size:
            #     return

            if np.max(dist) > self.config.step_size:
                return

            if not self.env.is_edge_collision_free(n_new.state.q, n_nearest_b.state.q, mode): #ORder rigth? TODO
                return
          

        elif self.config.birrtstar_version == 2 and not self.trees[mode].connected: #Based on paper RRT-Connect by JJ. Kuffner/ RRT*-Connect by S.Klemm
            n_nearest_b = self.Extend(mode, n_nearest_b, n_new, dist)
            if not n_nearest_b:
                return
            cost =  batch_config_cost([n_new.state], [n_nearest_b.state], metric = self.config.cost_metric, reduction=self.config.cost_reduction)
            
 
        if self.trees[mode].order == -1:
            #switch such that subtree is beginning from start and subtree_b from goal
            self.trees[mode].swap()
            self.UpdateTree(mode, n_new, n_nearest_b, cost[0]) 
        else:
            self.UpdateTree(mode, n_nearest_b, n_new, cost[0])
    
    def GeneratePath(self, mode:Mode, n: Node, iter:int = None, inter:bool = False, shortcutting:bool = True) -> None:
        path_nodes, path = [], []
        while n:
            path_nodes.append(n)
            path.append(n.state)
            n = n.parent
        path_in_order = path[::-1]
        self.operation.path = path_in_order  
        self.operation.path_nodes = path_nodes[::-1]
        # if inter:
        #     #check if termination is reached
        #     if self.env.done(path_nodes_[-1].state.q, self.modes[-1]):
        #         if path_nodes_[-1].id not in self.transition_node_ids[self.modes[-1]]:
        #             self.mark_node_as_transition(self.modes[-1], path_nodes_[-1])
        #     #intermediate solutions
        #     if not self.operation.init_sol and path_nodes_ not in self.operation.paths_inter:
        #         self.operation.paths_inter.append(path_nodes_) 
        #     return
            
        #check if terminal node has been reached

        if shortcutting and self.start_single_goal.satisfies_constraints(self.operation.path_nodes[0].state.q.state(), mode, self.env.collision_tolerance):
            if not self.operation.init_sol:
                print(time.time()-self.start)
                self.operation.init_sol = True
            self.operation.cost = self.operation.path_nodes[-1].cost
            self.SaveData(mode, time.time()-self.start)
            self.costs.append(self.operation.cost)
            self.times.append(time.time()-self.start)
            if self.config.shortcutting:
                print(f"-- M", mode.task_ids, "Cost: ", self.operation.cost)
                self.Shortcutting(mode)
            print(f"{iter} M", mode.task_ids, "Cost: ", self.operation.cost)
        if self.config.shortcutting and not shortcutting:
            self.operation.cost = self.operation.path_nodes[-1].cost
            self.SaveData(mode, time.time()-self.start)
            self.costs.append(self.operation.cost)
            self.times.append(time.time()-self.start)
                 
    def ManageTransition(self, mode:Mode, n_new:Node, iter:int) -> None:
        #if tree is in the right order
        if self.trees[mode].order == 1: 
            #Check if transition is reached
            if self.env.is_transition(n_new.state.q, mode):
                self.convert_node_to_transition_node(mode, n_new)
            #Check if termination is reached
            if self.env.done(n_new.state.q, mode):
                self.convert_node_to_transition_node(mode, n_new)

        else:
            if mode.prev_mode and self.env.is_transition(n_new.state.q, mode.prev_mode):
                n_new.state.mode = mode.prev_mode
                n_new_parent = n_new.parent
                n_new.parent.children.remove(n_new)
                cost = n_new.cost_to_parent
                n_new.parent = None
                n_new.cost_to_parent = np.float32(0)
                n_new.cost = np.inf
                self.trees[mode].swap()
                self.convert_node_to_transition_node(mode.prev_mode, n_new)
                self.trees[mode].remove_node(n_new, 'B')
                self.UpdateTree(mode, n_new_parent, n_new, cost)
                #need to add node to previous mode
                if self.trees[mode.prev_mode].order == 1:
                    self.trees[mode.prev_mode].add_node(n_new, 'B')
                else:
                    self.trees[mode.prev_mode].add_node(n_new, 'A')
            #need to handle first mode separately
            elif self.start_single_goal.satisfies_constraints(n_new.state.q.state(),mode, self.env.collision_tolerance):
                    n_new_parent = n_new.parent
                    n_new.parent.children.remove(n_new)
                    cost = n_new.cost_to_parent
                    n_new.parent = None
                    n_new.cost_to_parent = np.float32(0)
                    n_new.cost = 0
                    self.trees[mode].swap()
                    self.convert_node_to_transition_node(mode.prev_mode, n_new)
                    self.trees[mode].remove_node(n_new, 'B')
                    self.UpdateTree(mode, n_new_parent, n_new, cost)
        self.FindLBTransitionNode(mode, iter)
            
    def FindLBTransitionNode(self, mode:Mode, iter: int) -> None:
        #check if termination node is reached (cost < np.inf)
        result = self.get_lb_transition_node_id(self.modes[-1])
        if not result:
            return 
        valid_mask = result[0] < self.operation.cost
        if valid_mask.any():
            lb_transition_node = self.get_transition_node(self.modes[-1], result[1])
            self.GeneratePath(mode, lb_transition_node, iter, shortcutting=True) 

    def Plan(self) -> dict:
        i = 0
        self.PlannerInitialization()
        while True:
            i += 1
            # Mode selection
            active_mode  = self.RandomMode(Mode.id_counter)
            
            # Bi-RRT* parallelized core
            q_rand = self.SampleNodeManifold(active_mode)
            if not self.trees[active_mode].subtree or  self.operation.init_sol and self.trees[active_mode].order == -1:
                self.trees[active_mode].swap()
                self.trees[active_mode].connected = True
                    
            n_nearest, dist = self.Nearest(active_mode, q_rand)        
            state_new = self.Steer(active_mode, n_nearest, q_rand, dist)
            if not state_new: # meaning n_new is exact the same as one of the nodes in the tree
                continue
            
            if self.env.is_collision_free(state_new.q, active_mode) and self.env.is_edge_collision_free(n_nearest.state.q, state_new.q, active_mode):
                n_new = Node(state_new, self.operation)
                N_near_batch, n_near_costs, node_indices = self.Near(active_mode, n_new)
                if n_nearest.id not in node_indices:
                    continue

              
                batch_cost = batch_config_cost(n_new.state.q, N_near_batch, metric = self.config.cost_metric, reduction=self.config.cost_reduction)
                self.FindParent(active_mode, node_indices, n_new, n_nearest, batch_cost, n_near_costs)
                # if self.operation.init_sol:
                if self.operation.init_sol:
                    if self.Rewire(active_mode, node_indices, n_new, batch_cost, n_near_costs):
                        self.UpdateCost(active_mode,n_new)
                self.Connect(active_mode, n_new, i)
                self.ManageTransition(active_mode, n_new, i)
            self.trees[active_mode].swap()
           
            if self.PTC(i):
                "PTC applied"
                self.SaveFinalData()
                break
            

        self.SaveData(active_mode, time.time()-self.start, n_new = n_new.state.q)
        print(time.time()-self.start)
        return self.operation.path    




