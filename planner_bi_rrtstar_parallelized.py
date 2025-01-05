from planner_bi_rrtstar import *

"""
This file contains the most important changes for the parallelized Bi-RRT* algorithm compared to the original RRT*. There are 2 versions possible: 
Version 1 is inspired by the paper 'Bi-RRT*: An Improved Bidirectional RRT* Path Planner for Robot in Two-Dimensional Space' by B. Wang et al. 
and version 2 is inspired by the paper 'RRT-Connect: An Efficient Approach to Single-Query Path Planning' by J.J Kuffner et al. and 
by the paper 'RRT*-Connect: Faster, Asymptotically Optimal Motion Planning' by S. Klemm et al. 
"""


class ParallelizedBidirectionalRRTstar(BidirectionalRRTstar):
    def __init__(self, env, config: ConfigManager):
        super().__init__(env, config)

    def UpdateMode(self, mode:Mode, n:Node, tree:str) -> None:
        if tree == 'A':
            if n.idx not in mode.node_idx_subtree:
                mode.batch_subtree = torch.cat((mode.batch_subtree, n.q_tensor.unsqueeze(0)), dim=0)
                mode.subtree.append(n)
                mode.node_idx_subtree = torch.cat((mode.node_idx_subtree, torch.tensor([n.idx], device='cuda')),dim=0)
        if tree == 'B':
            if n.idx not in mode.node_idx_subtree_b :
                mode.batch_subtree_b = torch.cat((mode.batch_subtree_b, n.q_tensor.unsqueeze(0)), dim=0)
                mode.subtree_b.append(n)
                mode.node_idx_subtree_b = torch.cat((mode.node_idx_subtree_b, torch.tensor([n.idx], device='cuda')),dim=0)
        
    def PlannerInitialization(self) -> None:
        # Initialization of frist mode (handled separately as start condition is given for all robots)
        active_mode = self.operation.modes[0]
        start_state = State(self.env.start_pos, active_mode.label)
        start_node = Node(start_state, (self.operation.tree -1), self.operation)
        self.operation.start_node = start_node
        self.UpdateMode(active_mode, start_node, 'A')
        #Initialize all other modes:
        while True:
            self.InitializationMode(self.operation.modes[-1])            
            if active_mode.label == self.env.terminal_mode:
                nr = 1
            else:
                self.operation.modes.append( Mode(self.env.get_next_mode(None,active_mode.label), self.env))
                nr = self.config.transition_nodes
            for _ in range(nr): 
                end_q = self.sampling.get_goal_config_of_mode(active_mode)
                end_state = State(type(self.env.get_start_pos())(end_q, start_node.state.q.slice), active_mode.label)
                end_node = Node(end_state, self.operation.tree, self.operation)
                self.operation.tree += 1
                #Potentila goal nodes
                end_node.transition = True
                self.UpdateMode(active_mode, end_node, 'B')
                self.operation.costs = torch.cat((self.operation.costs, torch.tensor([np.inf], device='cuda')), dim=0)
                active_mode.transition_nodes.append(end_node)
                if active_mode.label == self.env.terminal_mode:
                    return
                # end_node.state.mode = self.operation.modes[-1].label #-> not needed
                #Each potential goal node is a potential start node for the next mode
                # self.operation.costs = torch.cat((self.operation.costs, torch.tensor([0], device='cuda')), dim=0)
                self.UpdateMode(self.operation.modes[-1], end_node, 'A')
                
            active_mode =  self.operation.modes[-1]

    def Connect(self, n_new:Node, iter:int) -> None: 
        if self.operation.active_mode.subtree_b == []:
            return
        n_nearest_b =  self.Nearest(n_new.state.q, self.operation.active_mode.subtree_b, self.operation.active_mode.batch_subtree_b)

        if self.config.birrtstar_version == 1 or self.config.birrtstar_version == 2 and self.operation.init_sol: #Based on paper Bi-RRT* by B. Wang 
            #TODO rewire everything?
            #TODO only check dist of active robots to connect (cost can be extremly high)? or the smartest way to just connect when possible?
          
            cost, dists = batch_config_torch(n_new.state.q, n_nearest_b.q_tensor.unsqueeze(0), "euclidean")
            # relevant_dists = []
            # for r_idx, r in enumerate(self.env.robots):
            #     if r in constrained_robots:
            #         relevant_dists.append(dists[0][r_idx].item())
            # if np.max(relevant_dists) > self.config.step_size:
            #     return

            if torch.max(dists).item() > self.config.step_size:
                return

            if not self.env.is_edge_collision_free(n_new.state.q, n_nearest_b.state.q, self.operation.active_mode.label): 
                return
          

        elif self.config.birrtstar_version == 2 and not self.operation.init_sol: #Based on paper RRT-Connect by JJ. Kuffner/ RRT*-Connect by S.Klemm
            n_nearest_b = self.Extend(n_nearest_b, n_new.state.q)
            if not n_nearest_b:
                return
            cost, dists = batch_config_torch(n_new.state.q, n_nearest_b.q_tensor.unsqueeze(0), "euclidean")
        print(iter)
        if self.operation.active_mode.order == -1: #Switch such that subtree is beginning from start and subtree_b from goal
            self.SWAP() 
            self.UpdateTree(n_new, n_nearest_b, cost[0], dists) 
        else:
            self.UpdateTree(n_nearest_b, n_new, cost[0], dists)
        # if self.operation.active_mode.connected:
        # if self.operation.init_sol:
        #     self.AddTransitionNode(self.operation.active_mode.transition_nodes[-1])
        
        # if not self.operation.active_mode.connected: #initial sol of this mode hasn't been found yet
        if not self.operation.init_sol:
            self.SaveData(time.time()-self.start)
            return 
        
        self.FindOptimalTransitionNode(iter)
    
    def UpdateTree(self, n: Node, n_parent:Node , cost_to_parent:torch.Tensor, dists:torch.Tensor) -> None: #TODO need to update it
        while True:
            n.cost = n_parent.cost +  cost_to_parent
            n.agent_dists = n_parent.agent_dists + dists
            self.UpdateCost(n, True)
            n_parent_inter = n.parent
            cost_to_parent_inter = n.cost_to_parent
            agent_dists_to_parent_inter = n.agent_dists_to_parent
            n.parent = n_parent
            n.cost_to_parent = cost_to_parent
            n.agent_dists_to_parent = dists
            n_parent.children.append(n) #Set child
            cost_to_parent = cost_to_parent_inter
            dists = agent_dists_to_parent_inter
            n_parent = n
            n = n_parent_inter
            if not n: # this if statement differs to the not parallelized rrtstar
                self.GeneratePath(n_parent, True)
                return 
            n.children.remove(n_parent)

    def GeneratePath(self, n: Node, inter:bool = False) -> None:
        path_nodes, path = [], []
        while n:
            path_nodes.append(n)
            path.append(n.state)
            n = n.parent
        path_nodes_ = path_nodes[::-1]
        if inter:
        #    print([node.state.mode for node in path_nodes_]) 
            if self.env.get_active_task(self.operation.modes[-1].label).goal.satisfies_constraints(path_nodes_[-1].state.q.state()[self.operation.modes[-1].indices], self.config.goal_radius):
                if path_nodes_[-1] not in self.operation.modes[-1].transition_nodes:
                    self.operation.modes[-1].transition_nodes.append(path_nodes_[-1])
            if not self.operation.init_sol and path_nodes_ not in self.operation.paths_inter:
                self.operation.paths_inter.append(path_nodes_) 
            return
            
        path_in_order = path[::-1]
        self.operation.path = path_in_order  
        self.operation.path_nodes = path_nodes_
        #Check if the last node of path is the termination node 
        if self.operation.init_sol:
            self.operation.cost = self.operation.path_nodes[-1].cost
            self.SaveData(time.time()-self.start)
                
    def ManageTransition(self, n_new:Node, iter:int) -> None:
        mode_idx = self.operation.modes.index(self.operation.active_mode)
        if self.operation.active_mode.order == 1: 
            if self.env.get_active_task(self.operation.active_mode.label).goal.satisfies_constraints(n_new.state.q.state()[self.operation.active_mode.indices], self.config.goal_radius):
                self.operation.active_mode.transition_nodes.append(n_new)
                # self.UpdateCost(n_new)
                n_new.transition = True
                
                #Need to add potential start/end node to other subtree
                if mode_idx != len(self.operation.modes)-1:
                    next_mode = self.operation.modes[mode_idx +1]
                    if next_mode.order == 1:
                        self.UpdateMode(next_mode, n_new, 'A')
                    else:
                        self.UpdateMode(next_mode, n_new, 'B')
        if self.operation.active_mode.order == -1:
            if mode_idx != 0:
                previous_mode = self.operation.modes[mode_idx -1] #previous mode
                if self.env.get_active_task(previous_mode.label).goal.satisfies_constraints(n_new.state.q.state()[previous_mode.indices], self.config.goal_radius):
                    # Need to transform it back to a transition node of the previous mode -> Need to change some data of n_new
                    n_new.state.mode = previous_mode.label
                    n_new_parent = n_new.parent
                    n_new.parent.children.remove(n_new)
                    dists = n_new.agent_dists_to_parent
                    cost = n_new.cost_to_parent
                    n_new.parent = None
                    n_new.agent_dists = torch.zeros(1, n_new.state.q.num_agents(), device = 'cuda')
                    n_new.cost_to_parent = torch.tensor(0, device='cuda')
                    n_new.agent_dists_to_parent = torch.zeros(1, n_new.state.q.num_agents(), device = 'cuda')
                    n_new.cost = torch.tensor([np.inf], device='cuda')
                    previous_mode.transition_nodes.append(n_new)
                    n_new.transition = True
                    self.SWAP()
                    self.UpdateMode(self.operation.active_mode, n_new, 'A') 
                    self.operation.active_mode.node_idx_subtree_b = self.operation.active_mode.node_idx_subtree_b[self.operation.active_mode.node_idx_subtree_b != n_new.idx]
                    self.operation.active_mode.batch_subtree_b = self.operation.active_mode.batch_subtree_b[~torch.all(self.operation.active_mode.batch_subtree_b == n_new.q_tensor.unsqueeze(0), dim=1)]
                    self.operation.active_mode.subtree_b.remove(n_new) 
                    self.UpdateTree(n_new_parent, n_new, cost, dists)
                     
                    if previous_mode.order == 1:
                        self.UpdateMode(previous_mode, n_new, 'B')
                    else:
                        self.UpdateMode(previous_mode, n_new, 'A')
                    
            else: #TODO really needed?
                if self.start_single_goal.satisfies_constraints(n_new.state.q.state(), self.config.goal_radius):
                    n_new_parent = n_new.parent
                    n_new.parent.children.remove(n_new)
                    dists = n_new.agent_dists_to_parent
                    cost = n_new.cost_to_parent
                    n_new.parent = None
                    n_new.agent_dists = torch.zeros(1, n_new.state.q.num_agents(), device = 'cuda')
                    n_new.cost_to_parent = torch.tensor(0, device='cuda')
                    n_new.agent_dists_to_parent = torch.zeros(1, n_new.state.q.num_agents(), device = 'cuda')
                    n_new.cost = torch.tensor([0], device='cuda')
                    n_new.transition = True
                    self.SWAP()
                    self.UpdateMode(self.operation.active_mode, n_new, 'A') 
                    self.operation.active_mode.node_idx_subtree_b = self.operation.active_mode.node_idx_subtree_b[self.operation.active_mode.node_idx_subtree_b != n_new.idx]
                    self.operation.active_mode.batch_subtree_b = self.operation.active_mode.batch_subtree_b[~torch.all(self.operation.active_mode.batch_subtree_b == n_new.q_tensor.unsqueeze(0), dim=1)]
                    self.operation.active_mode.subtree_b.remove(n_new) 
                    self.UpdateTree(n_new_parent, n_new, cost, dists)
                    
        self.FindOptimalTransitionNode(iter)
            
    def FindOptimalTransitionNode(self, iter: int) -> None:
        transition_nodes = self.operation.modes[-1].transition_nodes #transition nodes only of last mode
        costs = torch.cat([node.cost.unsqueeze(0) for node in transition_nodes])
        valid_mask = costs < self.operation.cost
        if valid_mask.any():
            min_cost_idx = torch.masked_select(torch.arange(len(costs), device=costs.device), valid_mask)[
                costs[valid_mask].argmin()
            ].item()
            lowest_cost_node = transition_nodes[min_cost_idx]
            self.GeneratePath(lowest_cost_node) 
            if self.operation.init_sol:
                print(f"{iter} Cost: ", self.operation.cost, " Mode: ", self.operation.active_mode.label) 
            elif not self.operation.init_sol and self.start_single_goal.satisfies_constraints(self.operation.path_nodes[0].state.q.state(), self.config.goal_radius):
                self.operation.cost = lowest_cost_node.cost
                self.operation.ptc_iter = iter
                self.operation.ptc_cost = self.operation.cost
                self.operation.init_sol = True
                self.operation.paths_inter = []
                print(f"{iter} Cost: ", self.operation.cost, " Mode: ", self.operation.active_mode.label) 
            
            if (self.operation.ptc_cost - self.operation.cost) > self.config.ptc_threshold:
                self.operation.ptc_cost = self.operation.cost
                self.operation.ptc_iter = iter

    def Plan(self) -> dict:
        i = 0
        self.PlannerInitialization()
        while True:
            # Mode selection
            self.operation.active_mode  = (np.random.choice(self.operation.modes, p= self.SetModePorbability()))
            
            # Bi-RRT* parallelized core
            q_rand = self.SampleNodeManifold(self.operation)
            if self.operation.active_mode.subtree == [] or self.operation.init_sol and self.operation.active_mode.order == -1:
                self.SWAP()
                self.operation.active_mode.connected = True
                    
            n_nearest= self.Nearest(q_rand, self.operation.active_mode.subtree, self.operation.active_mode.batch_subtree)    
            n_new = self.Steer(n_nearest, q_rand, self.operation.active_mode.label)
            if not n_new: # meaning n_new is exactly the same as one of the nodes in the tree
                continue

            if self.env.is_collision_free(n_new.state.q.state(), self.operation.active_mode.label) and self.env.is_edge_collision_free(n_nearest.state.q, n_new.state.q, self.operation.active_mode.label):
                N_near_indices, N_near_batch, n_near_costs, node_indices = self.Near(n_new, self.operation.active_mode.batch_subtree)
                if n_nearest.idx not in node_indices:
                    continue
                n_nearest_index = torch.where(node_indices == n_nearest.idx)[0].item() 
                batch_cost, batch_dist =  batch_config_torch(n_new.state.q, N_near_batch, metric = self.config.cost_type)
                self.FindParent(N_near_indices, n_nearest_index, n_new, n_nearest, batch_cost, batch_dist, n_near_costs)
                if self.operation.init_sol:
                    if self.Rewire(N_near_indices, n_new, batch_cost, batch_dist, n_near_costs, n_rand = q_rand.state(), n_nearest = n_nearest.state.q.state() ):
                        self.UpdateCost(n_new)
                self.Connect(n_new, i)
                self.ManageTransition(n_new, i)
            if self.operation.init_sol and i != self.operation.ptc_iter and (i- self.operation.ptc_iter)%self.config.ptc_max_iter == 0:
                diff = self.operation.ptc_cost - self.operation.cost
                self.operation.ptc_cost = self.operation.cost
                if diff < self.config.ptc_threshold:
                    break
            self.SWAP()
            i += 1
            

        self.SaveData(time.time()-self.start, n_new = n_new.state.q)
        return self.operation.path    




