from multi_robot_multi_goal_planning.planners.planner_bi_rrtstar_parallelized import *


"""This file contains the original RRT* based on the paper 'Sampling-based Algorithms for Optimal Motion Planning' by E. Frazolli et al. but parallelized over all the modes."""

class ParallelizedRRTstar(ParallelizedBidirectionalRRTstar):
    def __init__(self, env, config: ConfigManager):
        super().__init__(env, config)
 
    def PlannerInitialization(self) -> None:
        # Initilaize first Mode
        self.add_new_mode(tree_instance=SingleTree)
        mode = self.modes[-1]
        # Create start node
        start_state = State(self.env.start_pos, mode)
        start_node = Node(start_state, self.operation)
        self.trees[mode].add_node(start_node)
        start_node.cost = 0
        start_node.cost_to_parent = torch.tensor(0, device=device, dtype=torch.float32)
        #Initialize other modes:
        while True:
            self.ModeInitialization(mode)
            if not self.env.is_terminal_mode(mode): 
                self.add_new_mode(mode=mode, tree_instance=SingleTree) 
            for _ in range(self.config.transition_nodes):                 
                q = self.sample_transition_configuration(mode)
                node = Node(State(q, mode), self.operation)
                node.cost_to_parent = torch.tensor(0, device=device, dtype=torch.float32)
                node.transition = True
                self.mark_node_as_transition(mode, node)
                self.trees[self.modes[-1]].add_node(node)
                self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, node.id) 
                node.cost = np.inf
            mode = self.modes[-1]
            if self.env.is_terminal_mode(mode): 
                return
   
    def ManageTransition(self, mode:Mode, n_new: Node, iter: int) -> None:
        #check if transition is reached
        if self.env.is_transition(n_new.state.q, mode):
            self.convert_node_to_transition_node(mode, n_new)
        #check if termination is reached
        if self.env.done(n_new.state.q, mode):
            self.convert_node_to_transition_node(mode, n_new)
        if self.modes[-1] not in self.transition_node_ids:
            return
        self.FindLBTransitionNode(mode, iter)

    def Plan(self) -> List[State]:
        i = 0
        self.PlannerInitialization()
        while True:
            i += 1
            # Mode selection
            active_mode  = self.RandomMode(Mode.id_counter)
            # RRT* core
            q_rand = self.SampleNodeManifold(active_mode)
            n_nearest, dist = self.Nearest(active_mode, q_rand)        
            state_new = self.Steer(active_mode, n_nearest, q_rand, dist)
            if not state_new: # meaning n_new is exact the same as one of the nodes in the tree
                continue
            # if i == 1538 :
            #     print("hal")
            if self.env.is_collision_free(state_new.q, active_mode) and self.env.is_edge_collision_free(n_nearest.state.q, state_new.q, active_mode):
                n_new = Node(state_new, self.operation)
                N_near_batch, n_near_costs, node_indices = self.Near(active_mode, n_new)
                if n_nearest.id not in node_indices:
                    continue
                batch_cost =  batch_cost_torch(n_new.q_tensor, n_new.state.q, N_near_batch, metric = self.config.cost_type).clone()
                self.FindParent(active_mode, node_indices,n_new, n_nearest, batch_cost, n_near_costs)
                # if self.operation.init_sol:
                if self.Rewire(active_mode, node_indices, n_new, batch_cost, n_near_costs):
                    self.UpdateCost(active_mode, n_new)
                    # self.SaveData(time.time()-self.start, n_new = n_new.state.q.state(), N_near = N_near, 
                    #           r =self.r, n_rand = n_rand.state.q.state(), n_nearest = n_nearest.state.q.state()) 
                
                self.ManageTransition(active_mode, n_new, i)

            if self.PTC(i):
                "PTC applied"
                self.SaveFinalData()
                break

            
         

        self.SaveData(active_mode, time.time()-self.start)
        print(time.time()-self.start)
        return self.operation.path    



