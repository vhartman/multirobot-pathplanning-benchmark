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
    
    def Near(self, mode:Mode, n_new: Node) -> Tuple[List[Node], torch.tensor]:
        #TODO generalize rewiring radius
        # n_nodes = sum(1 for _ in self.operation.current_mode.subtree.inorder()) + 1
        # r = min((7)*self.step_size, 3 + self.gamma * ((math.log(n_nodes) / n_nodes) ** (1 / self.dim)))
        batch_subtree = self.trees[mode].get_batch_subtree()
        set_dists = batch_dist_torch(n_new.q_tensor, n_new.state.q, batch_subtree, self.config.dist_type).clone()
        indices = torch.where(set_dists < self.r)[0] # indices of batch_subtree
        node_indices = self.trees[mode].node_idx_subtree.index_select(0,indices) # actual node indices (node.id)
        return indices, node_indices

    def Ancestry(self, mode:Mode, N_near_indices, node_indices, node_idx_subtree_set):
        active_mode_subtree =  self.trees[mode].subtree
        node_union_list = []

        for node_idx in node_indices.tolist():
            node = active_mode_subtree[node_idx]
            depth = 0
            while node.parent is not None and depth < self.config.depth:
                parent_idx = node.parent.id
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
            indices_union = torch.searchsorted(self.trees[mode].get_node_idx_subtree(), node_union_indices.view(-1))

        # Retrieve costs and batch
        n_union_costs = self.operation.costs[node_union_indices]
        N_union_batch = self.trees[mode].get_batch_subtree()[indices_union, :]

        return N_union_batch, n_union_costs, node_union_indices 

    def ancestor(self, mode:Mode, node, node_idx_subtree_set):
        # Collect ancestor indices up to the maximum depth
        node_ancestor_list = []
        for _ in range(self.config.depth):
            if node.parent is None or node.parent.id not in node_idx_subtree_set:
                break
            node_ancestor_list.append(node.parent.id)
            node = node.parent

        if node_ancestor_list == []:
            return None

        ordered_unique = list(OrderedDict.fromkeys(node_ancestor_list))
        node_ancestor_indices = torch.tensor(ordered_unique, device=device, dtype=torch.long)
        # indices_ancestor = torch.searchsorted(self.trees[mode].get_node_idx_subtree(), node_ancestor_indices.view(-1))
        return torch.flip(node_ancestor_indices, dims=[0])

    def Rewire(self, mode:Mode, N_near_indices: torch.Tensor, N_near_batch, n_new: Node, ancestors: List[Node],
            batch_cost: torch.Tensor, node_indices) -> bool:
        rewired = False
        num_near = N_near_indices.shape[0]

        # Compute the potential cost for connection via n_new.
        c_potential_tensor = n_new.cost + batch_cost[:num_near]  # shape: (num_near,)

        if ancestors:
            ancestor_q_tensors = torch.stack([a.q_tensor for a in ancestors], dim=0)
            ancestor_costs = torch.tensor([a.cost for a in ancestors], device=batch_cost.device)

            ancestor_batch_costs = batches_config_torch(
                ancestor_q_tensors, ancestors[0].state.q, N_near_batch, metric=self.config.cost_type
            ).clone()
            ancestor_total_costs = ancestor_batch_costs + ancestor_costs.unsqueeze(1)
            all_costs = torch.cat((ancestor_total_costs, c_potential_tensor.unsqueeze(0)), dim=0)  # shape: (num_candidates, num_near)
        else:
            all_costs = c_potential_tensor.unsqueeze(0)  # shape: (1, num_near)

        n_near_costs = self.operation.costs[node_indices]  # shape: (num_near,)
        mask = all_costs < n_near_costs.unsqueeze(0)
        if not mask.any():
            return False

        candidate_costs = torch.where(mask, all_costs, torch.tensor(float("inf"), device=all_costs.device))
        valid_near = candidate_costs[0] < n_near_costs
        if not valid_near.any():
            return False

        valid_near_indices = torch.nonzero(valid_near, as_tuple=False).squeeze(1)

        for near_idx in valid_near_indices.tolist():
            n_near = self.trees[mode].subtree.get(node_indices[near_idx].item())
            valid_candidates = torch.nonzero(mask[:, near_idx], as_tuple=False).squeeze(1)
            candidate_list = valid_candidates.tolist()

            for cand in candidate_list:
                if ancestors:
                    if cand == len(ancestors):
                        parent = n_new
                        cost_parent = batch_cost[near_idx]
                    else:
                        parent = ancestors[cand]
                        cost_parent = ancestor_batch_costs[cand, near_idx]
                else:
                    parent = n_new
                    cost_parent = batch_cost[near_idx]
                if n_near == parent or (n_near.parent is not None and n_near == n_near.parent) or n_near.cost == float("inf"):
                    continue

                if self.env.is_edge_collision_free(parent.state.q, n_near.state.q, mode):
                    continue
                if n_near.parent is not None:
                    n_near.parent.children.remove(n_near)
                n_near.parent = parent
                parent.children.append(n_near)

                n_near.cost = all_costs[cand, near_idx].item()
                n_near.cost_to_parent = cost_parent

                rewired = True
                break  # Only the best valid candidate is used per near node.

        return rewired

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
            if self.env.is_collision_free(state_new.q, active_mode) and self.env.is_edge_collision_free(n_nearest.state.q, state_new.q, active_mode):
                n_new = Node(state_new, self.operation)
                N_near_indices, node_indices = self.Near(active_mode, n_new)
                # N_near_indices, node_indices = self.Near(n_new, self.operation.active_mode.batch_subtree)
                # if n_nearest.id not in node_indices:
                #     continue
                
                node_idx_subtree_set = set(self.trees[active_mode].get_node_idx_subtree().tolist())
                # if i == 1175:
                #     print("hallo")
                N_union_batch, n_union_costs, node_union_indices = self.Ancestry(active_mode, N_near_indices, node_indices, node_idx_subtree_set)
                batch_cost = batch_cost_torch(n_new.q_tensor, n_new.state.q, N_union_batch, metric = self.config.cost_type).clone()
                # Input differs to original RRT* 
                self.FindParent(active_mode, node_union_indices, n_new, n_nearest, batch_cost, n_union_costs)
                node_ancestor_indices = self.ancestor(active_mode, n_new, node_idx_subtree_set)
                if node_ancestor_indices is not None:
                    ancestors = [self.trees[active_mode].subtree[i.item()] for i in node_ancestor_indices]
                else:
                    ancestors = []
                N_near_batch = N_union_batch[:len(node_indices)]
                if self.Rewire(active_mode, N_near_indices, N_near_batch, n_new, ancestors, batch_cost, node_indices):
                    self.UpdateCost(n_new)
                
                self.ManageTransition(active_mode, n_new, i)

            if self.PTC(i):
                "PTC applied"
                self.SaveFinalData()
                break
        
            
            
         

        self.SaveData(time.time()-self.start)
        print(time.time()-self.start)
        return self.operation.path    




