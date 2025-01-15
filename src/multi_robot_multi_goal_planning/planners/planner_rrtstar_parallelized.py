from multi_robot_multi_goal_planning.planners.planner_bi_rrtstar_parallelized import *


"""This file contains the original RRT* based on the paper 'Sampling-based Algorithms for Optimal Motion Planning' by E. Frazolli et al. but parallelized over all the modes."""

class ParallelizedRRTstar(ParallelizedBidirectionalRRTstar):
    def __init__(self, env, config: ConfigManager):
        super().__init__(env, config)
 
    def PlannerInitialization(self) -> None:
        # Initialization of frist mode (handled separately as start condition is given for all robots)
        active_mode = self.operation.modes[0]
        start_state = State(self.env.start_pos, active_mode.label)
        start_node = Node(start_state, self.operation)
        self.operation.start_node = start_node
        active_mode.subtree.append(start_node)
        active_mode.batch_subtree[len(active_mode.subtree)-1, :] = start_node.q_tensor
        active_mode.node_idx_subtree[len(active_mode.subtree)-1] =start_node.idx
        start_node.cost = 0
        start_node.cost_to_parent = torch.tensor(0, device=device, dtype=torch.float32)
        #Initialize all other modes:

        while True:
            self.ModeInitialization(self.operation.modes[-1])     
            if active_mode.label == self.env.terminal_mode:
                nr = 1
            else:
                self.operation.modes.append( Mode(self.env.get_next_mode(None,active_mode.label), self.env))
                nr = self.config.transition_nodes
            for _ in range(nr): 
                
                end_q = self.sampling.get_goal_config_of_mode(active_mode)
                end_state = State(type(self.env.get_start_pos())(end_q, start_node.state.q.slice), active_mode.label)
                end_node = Node(end_state, self.operation)
                end_node.cost_to_parent = torch.tensor(0, device=device, dtype=torch.float32)
                end_node.transition = True

                self.operation.costs = active_mode.ensure_capacity(self.operation.costs, end_node.idx) 
                end_node.cost = np.inf
                active_mode.transition_nodes.append(end_node)
                if active_mode.label == self.env.terminal_mode:
                    # self.ModeInitialization() 
                    return
                self.UpdateMode(self.operation.modes[-1], end_node, 'A')
    
            active_mode =  self.operation.modes[-1]

    def ManageTransition(self, n_new:Node, iter:int) -> None:
        mode_idx = self.operation.modes.index(self.operation.active_mode)
        if self.env.get_active_task(self.operation.active_mode.label).goal.satisfies_constraints(n_new.state.q.state()[self.operation.active_mode.indices], self.env.tolerance):
            self.operation.active_mode.transition_nodes.append(n_new)
            # self.UpdateCost(n_new)
            n_new.transition = True
            
            #Need to add potential start/end node to other subtree
            if mode_idx != len(self.operation.modes)-1:
                next_mode = self.operation.modes[mode_idx +1]
                self.UpdateMode(next_mode, n_new, 'A')
        self.FindOptimalTransitionNode(iter)

    def Plan(self) -> dict:
        i = 0
        self.PlannerInitialization()
        while True:
            # Mode selection
            self.operation.active_mode  = (np.random.choice(self.operation.modes, p= self.SetModePorbability()))
            
            # RRT* parallelized core
            q_rand = self.SampleNodeManifold(self.operation)  
            n_nearest= self.Nearest(q_rand, self.operation.active_mode.subtree, self.operation.active_mode.batch_subtree[:len(self.operation.active_mode.subtree)])      
            state_new = self.Steer(n_nearest, q_rand, self.operation.active_mode.label)
            if not state_new: # meaning n_new is exactly the same as one of the nodes in the tree
                continue
            if self.env.is_collision_free(state_new.q.state(), self.operation.active_mode.label) and self.env.is_edge_collision_free(n_nearest.state.q, state_new.q, self.operation.active_mode.label):
                n_new = Node(state_new, self.operation)
                N_near_indices, N_near_batch, n_near_costs, node_indices = self.Near(n_new, self.operation.active_mode.batch_subtree[:len(self.operation.active_mode.subtree)])
                if n_nearest.idx not in node_indices:
                    continue
                n_nearest_index = torch.where(node_indices == n_nearest.idx)[0].item() 
                batch_cost, batch_dist =  batch_config_torch(n_new.q_tensor, n_new.state.q, N_near_batch, metric = self.config.cost_type)
                self.FindParent(N_near_indices, n_nearest_index, n_new, n_nearest, batch_cost, batch_dist, n_near_costs)
                if self.operation.init_sol:
                    if self.Rewire(N_near_indices, n_new, batch_cost, batch_dist, n_near_costs, n_rand = q_rand.state(), n_nearest = n_nearest.state.q.state() ):
                        self.UpdateCost(n_new)
                self.ManageTransition(n_new, i)
            if self.operation.init_sol and i != self.operation.ptc_iter and (i- self.operation.ptc_iter)%self.config.ptc_max_iter == 0:
                diff = self.operation.ptc_cost - self.operation.cost
                self.operation.ptc_cost = self.operation.cost
                if diff < self.config.ptc_threshold:
                    break
            i += 1
            if i%1000 == 0:
                print(i)

        self.SaveData(time.time()-self.start, n_new = n_new.state.q)
        return self.operation.path    




