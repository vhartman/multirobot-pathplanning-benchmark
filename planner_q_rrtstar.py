from planner_rrtstar import *

"""
This file contains the most important changes for the Quick-RRT* algorithm based on the paper 
'Quick-RRT*: Triangular inequality-based implementation of RRT* with improved inital solution and convergence rate' by IB Jeong et al. 
compared to the original RRT*. 
"""

class QRRTstar(RRTstar):
    def __init__(self, env, config: ConfigManager):
        super().__init__(env, config)
    
    def Near(self, n_new: Node, subtree_set: torch.tensor) -> Tuple[List[Node], torch.tensor]:
        #TODO generalize rewiring radius
        # n_nodes = sum(1 for _ in self.operation.current_mode.subtree.inorder()) + 1
        # r = min((7)*self.step_size, 3 + self.gamma * ((math.log(n_nodes) / n_nodes) ** (1 / self.dim)))
        set_dists = batch_config_dist_torch(n_new.state.q, subtree_set, self.config.dist_type)
        indices = torch.where(set_dists < self.r)[0] # of batch_subtree
        node_indices = self.operation.active_mode.node_idx_subtree[indices] # actual node indices (node.idx)
        return indices, node_indices
        # if not self.config.informed_sampling:
        #     return indices, N_near_batch, n_near_costs, node_indices
        # return self.sampling.fit_to_informed_subset(indices, N_near_batch, n_near_costs, node_indices)

    def Ancestry(self, N_near_indices, node_indices, node_idx_subtree_set):
        active_mode_subtree = self.operation.active_mode.subtree
        node_union_list = []

        for node_idx in N_near_indices.tolist():
            node = active_mode_subtree[node_idx]
            depth = 0
            while node.parent is not None and depth < self.config.depth:
                parent_idx = node.parent.idx
                if parent_idx not in node_idx_subtree_set:
                    break
                node_union_list.append(parent_idx)
                node = node.parent
                depth += 1

        if not node_union_list:
            node_union_indices = node_indices
            indices_union = N_near_indices
        else:
            # Convert list to tensor
            node_union_tensor = torch.tensor(node_union_list, device='cuda', dtype=torch.long).unique(sorted=False)
            # Use set-based difference to improve performance
            node_indices_set = set(node_indices.tolist())
            mask_new = torch.tensor([idx not in node_indices_set for idx in node_union_tensor.tolist()], device='cuda')
            node_union_indices = torch.cat([node_indices, node_union_tensor[mask_new]])
            indices_union = torch.searchsorted(self.operation.active_mode.node_idx_subtree, node_union_indices.view(-1))

        # Retrieve costs and batch
        n_union_costs = self.operation.costs[node_union_indices]
        N_union_batch = self.operation.active_mode.batch_subtree[indices_union, :]

        return indices_union, N_union_batch, n_union_costs, node_union_indices 

    def ancestor(self, node, node_idx_subtree_set):
        # Collect ancestor indices up to the maximum depth
        node_ancestor_list = []
        for _ in range(self.config.depth):
            if node.parent is None or node.parent.idx not in node_idx_subtree_set:
                break
            node_ancestor_list.append(node.parent.idx)
            node = node.parent

        if node_ancestor_list == []:
            return None
        node_ancestor_indices = torch.tensor(node_ancestor_list, device='cuda', dtype=torch.long).unique(sorted=False)
        indices_ancestor = torch.searchsorted(self.operation.active_mode.node_idx_subtree, node_ancestor_indices.view(-1))
        # indices_ancestor = torch.nonzero(torch.isin(active_mode_node_idx_subtree, node_ancestor_indices), as_tuple=False).squeeze(-1)
        # n_ancestor_costs = self.operation.costs[node_ancestor_indices]
        # N_ancestor_batch = self.operation.active_mode.batch_subtree[indices_ancestor, :]
        return indices_ancestor

    def Plan(self) -> List[State]:
        i = 0
        self.PlannerInitialization()
        while True:
            # Mode selection
            self.operation.active_mode  = (np.random.choice(self.operation.modes, p= self.SetModePorbability()))
            # RRT* core
            q_rand = self.SampleNodeManifold(self.operation)
            n_nearest= self.Nearest(q_rand, self.operation.active_mode.subtree, self.operation.active_mode.batch_subtree)        
            n_new = self.Steer(n_nearest, q_rand, self.operation.active_mode.label)
            if not n_new: # meaning n_new is exact the same as one of the nodes in the tree
                continue
            if self.env.is_collision_free(n_new.state.q.state(), self.operation.active_mode.label) and self.env.is_edge_collision_free(n_nearest.state.q, n_new.state.q, self.operation.active_mode.label):
                N_near_indices, node_indices = self.Near(n_new, self.operation.active_mode.batch_subtree)
                # N_near_indices, node_indices = self.Near(n_new, self.operation.active_mode.batch_subtree)
                # if n_nearest.idx not in node_indices:
                #     continue
                node_idx_subtree_set = set(self.operation.active_mode.node_idx_subtree.tolist())
                
                N_union_indices, N_union_batch, n_union_costs, node_union_indices = self.Ancestry(N_near_indices, node_indices, node_idx_subtree_set)
                n_nearest_index = torch.where(node_union_indices == n_nearest.idx)[0].item()
                batch_cost, batch_dist = batch_config_torch(n_new.state.q, N_union_batch, metric = self.config.cost_type)
                # Input differs to original RRT* 
                self.FindParent(N_union_indices, n_nearest_index, n_new, n_nearest, batch_cost, batch_dist, n_union_costs)

                indices_ancestor = self.ancestor(n_new, node_idx_subtree_set)
                batch_cost_, batch_dist_ = batch_cost[:len(N_near_indices)],  batch_dist[:len(N_near_indices)]
                n_near_costs = self.operation.costs[node_indices] #costs of all n in N_near
                if self.Rewire(N_near_indices, n_new, batch_cost_, batch_dist_, n_near_costs, n_rand = q_rand.state(), n_nearest = n_nearest.state.q.state() ):
                    self.UpdateCost(n_new)
                    
                
                if indices_ancestor is not None:
                    N_near_batch = N_union_batch[:len(node_indices)]
                    for ancestor_idx in indices_ancestor:                    
                        node = self.operation.active_mode.subtree[ancestor_idx]
                        n_near_costs = self.operation.costs[node_indices] #costs of all n in N_near
                        batch_cost, batch_dist =  batch_config_torch(node.state.q, N_near_batch, metric = self.config.cost_type)
                        if self.Rewire(N_near_indices, node, batch_cost, batch_dist, n_near_costs):
                            self.UpdateCost(node)
                            # self.SaveData(time.time()-self.start, n_new = n_new.state.q.state(), N_near = N_near_batch, 
                            #       r =self.r, n_rand = node.state.q.state(), n_nearest = n_nearest.state.q.state())

                self.ManageTransition(n_new, i)
            if self.operation.init_sol and i != self.operation.ptc_iter and (i- self.operation.ptc_iter)%self.config.ptc_max_iter == 0:
                diff = self.operation.ptc_cost - self.operation.cost
                self.operation.ptc_cost = self.operation.cost
                if diff < self.config.ptc_threshold:
                    break
            if i%1000 == 0:
                print(i)
            
            i += 1
         

        self.SaveData(time.time()-self.start)
        print(time.time()-self.start)
        return self.operation.path    




