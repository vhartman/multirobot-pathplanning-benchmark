from multi_robot_multi_goal_planning.planners.planner_rrtstar import *
from collections import OrderedDict

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
        set_dists = batch_config_dist_torch(n_new.q_tensor, n_new.state.q, subtree_set, self.config.dist_type)
        indices = torch.where(set_dists < self.r)[0] # of batch_subtree
        node_indices = self.operation.active_mode.node_idx_subtree.index_select(0, indices) # actual node indices (node.idx)
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
            node_union_tensor = torch.tensor(node_union_list, device=device, dtype=torch.long).unique(sorted=False)
            node_indices_set = set(node_indices.tolist())
            mask_new = torch.tensor([idx not in node_indices_set for idx in node_union_tensor.tolist()], device=device)
            node_union_indices = torch.cat([node_indices, node_union_tensor[mask_new]])
            indices_union = torch.searchsorted(self.operation.active_mode.node_idx_subtree[:len(self.operation.active_mode.subtree)], node_union_indices.view(-1))

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

        ordered_unique = list(OrderedDict.fromkeys(node_ancestor_list))
        node_ancestor_indices = torch.tensor(ordered_unique, device=device, dtype=torch.long)
        indices_ancestor = torch.searchsorted(self.operation.active_mode.node_idx_subtree[:len(self.operation.active_mode.subtree)], node_ancestor_indices.view(-1))
        return torch.flip(indices_ancestor, dims=[0])

    def Rewire(self, N_near_indices: torch.Tensor, N_near_batch, n_new: Node, ancestors: List[Node],
            batch_cost: torch.Tensor, batch_dist: torch.Tensor, node_indices) -> bool:
        rewired = False
        num_near = N_near_indices.shape[0]

        # Compute the potential cost for connection via n_new.
        c_potential_tensor = n_new.cost + batch_cost[:num_near]  # shape: (num_near,)

        if ancestors:
            ancestor_q_tensors = torch.stack([a.q_tensor for a in ancestors], dim=0)
            ancestor_costs = torch.tensor([a.cost for a in ancestors], device=batch_cost.device)

            ancestor_batch_costs, ancestor_batch_dists = batches_config_torch(
                ancestor_q_tensors, ancestors[0].state.q, N_near_batch, metric=self.config.cost_type
            )
            ancestor_total_costs = ancestor_batch_costs + ancestor_costs.unsqueeze(1)
            all_costs = torch.cat((ancestor_total_costs, c_potential_tensor.unsqueeze(0)), dim=0)  # shape: (num_candidates, num_near)
        else:
            all_costs = c_potential_tensor.unsqueeze(0)  # shape: (1, num_near)

        n_near_costs = self.operation.costs[node_indices]  # shape: (num_near,)
        mask = all_costs < n_near_costs.unsqueeze(0)
        if not mask.any():
            return False

        candidate_costs = torch.where(mask, all_costs, torch.tensor(float("inf"), device=all_costs.device))
        # best_candidate_costs, _ = candidate_costs.min(dim=0)
        # if not torch.equal(candidate_costs[0], best_candidate_costs):
        #     print("hallo")
        valid_near = candidate_costs[0] < n_near_costs
        if not valid_near.any():
            return False

        valid_near_indices = torch.nonzero(valid_near, as_tuple=False).squeeze(1)

        for near_idx in valid_near_indices.tolist():
            n_near = self.operation.active_mode.subtree[N_near_indices[near_idx].item()]
            valid_candidates = torch.nonzero(mask[:, near_idx], as_tuple=False).squeeze(1)
            candidate_list = valid_candidates.tolist()

            for cand in candidate_list:
                if ancestors:
                    if cand == len(ancestors):
                        parent = n_new
                        best_dist = batch_dist[near_idx]
                        cost_parent = batch_cost[near_idx]
                    else:
                        parent = ancestors[cand]
                        best_dist = ancestor_batch_dists[cand, near_idx]
                        cost_parent = ancestor_batch_costs[cand, near_idx]
                else:
                    parent = n_new
                    best_dist = batch_dist[near_idx]
                    cost_parent = batch_cost[near_idx]
                if n_near == parent or (n_near.parent is not None and n_near == n_near.parent) or n_near.cost == float("inf"):
                    continue

                if not self.env.is_edge_collision_free(parent.state.q, n_near.state.q,  self.operation.active_mode.label):
                    continue
                if n_near.parent is not None:
                    n_near.parent.children.remove(n_near)
                n_near.parent = parent
                parent.children.append(n_near)

                n_near.cost = all_costs[cand, near_idx].item()
                agents = best_dist.unsqueeze(0).to(dtype=torch.float16).cpu()
                n_near.agent_dists = agents + parent.agent_dists
                n_near.cost_to_parent = cost_parent
                n_near.agent_dists_to_parent = agents
                # if cand == 0 and len(ancestors) > 1 and parent != n_new:
                #     print(cand)

                rewired = True
                break  # Only the best valid candidate is used per near node.

        return rewired

    def Plan(self) -> List[State]:
        i = 0
        self.PlannerInitialization()
        while True:
            i += 1
            # Mode selection
            self.operation.active_mode  = (np.random.choice(self.operation.modes, p= self.SetModePorbability()))
            # RRT* core
            q_rand = self.SampleNodeManifold(self.operation)
            n_nearest= self.Nearest(q_rand, self.operation.active_mode.subtree, self.operation.active_mode.batch_subtree[:len(self.operation.active_mode.subtree)])        
            state_new = self.Steer(n_nearest, q_rand, self.operation.active_mode.label)
            if not state_new: # meaning n_new is exact the same as one of the nodes in the tree
                continue
            if self.env.is_collision_free(state_new.q.state(), self.operation.active_mode.label) and self.env.is_edge_collision_free(n_nearest.state.q, state_new.q, self.operation.active_mode.label):
                n_new = Node(state_new, self.operation)
                N_near_indices, node_indices = self.Near(n_new, self.operation.active_mode.batch_subtree[:len(self.operation.active_mode.subtree)])
                # N_near_indices, node_indices = self.Near(n_new, self.operation.active_mode.batch_subtree)
                # if n_nearest.idx not in node_indices:
                #     continue
                node_idx_subtree_set = set(self.operation.active_mode.node_idx_subtree[:len(self.operation.active_mode.subtree)].tolist())
                # if i == 1175:
                #     print("hallo")
                N_union_indices, N_union_batch, n_union_costs, node_union_indices = self.Ancestry(N_near_indices, node_indices, node_idx_subtree_set)
                n_nearest_index = torch.where(node_union_indices == n_nearest.idx)[0].item()
                batch_cost, batch_dist = batch_config_torch(n_new.q_tensor, n_new.state.q, N_union_batch, metric = self.config.cost_type)
                # Input differs to original RRT* 
                self.FindParent(N_union_indices, n_nearest_index, n_new, n_nearest, batch_cost, batch_dist, n_union_costs)
                indices_ancestor = self.ancestor(n_new, node_idx_subtree_set)
                if indices_ancestor is not None:
                    ancestors = [self.operation.active_mode.subtree[i] for i in indices_ancestor]
                else:
                    ancestors = []
                N_near_batch = N_union_batch[:len(node_indices)]
                if self.Rewire(N_near_indices, N_near_batch, n_new, ancestors, batch_cost, batch_dist, node_indices):
                    self.UpdateCost(n_new)
                
                self.ManageTransition(n_new, i)
            if self.operation.init_sol and i != self.operation.ptc_iter and (i- self.operation.ptc_iter)%self.config.ptc_max_iter == 0:
                diff = self.operation.ptc_cost - self.operation.cost
                self.operation.ptc_cost = self.operation.cost
                if diff < self.config.ptc_threshold:
                    break
            if i%1000 == 0:
                print(i)
        
            
            
         

        self.SaveData(time.time()-self.start)
        print(time.time()-self.start)
        return self.operation.path    




