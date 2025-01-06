from multi_robot_multi_goal_planning.planners.rrtstar_base import *

"""
This file contains the most important changes for the Bi-RRT* algorithm compared to the original RRT*. There are 2 versions possible: 
Version 1 is inspired by the paper 'Bi-RRT*: An Improved Bidirectional RRT* Path Planner for Robot inTwo-Dimensional Space' by B. Wang et al. 
and version 2 is inspired by the paper 'RRT-Connect: An Efficient Approach to Single-Query Path Planning' by J.J Kuffner et al. and 
by the paper 'RRT*-Connect: Faster, Asymptotically Optimal Motion Planning' by S. Klemm et al. 
"""



class BidirectionalRRTstar(BaseRRTstar):
    def __init__(self, env, config: ConfigManager):
        super().__init__(env, config)
    
    def UpdateCost(self, n:Node, connection:bool = False) -> None:
        stack = [n]
        while stack:
            current_node = stack.pop()
            if connection:
                #add node to main subtree and delte it from subtree_b
                if current_node.state.mode == self.operation.active_mode.label:
                    if current_node.idx not in self.operation.active_mode.node_idx_subtree:
                        self.operation.active_mode.subtree.append(current_node)
                        self.operation.active_mode.batch_subtree = torch.cat((self.operation.active_mode.batch_subtree, current_node.q_tensor.unsqueeze(0)), dim=0)
                        self.operation.active_mode.node_idx_subtree = torch.cat((self.operation.active_mode.node_idx_subtree, torch.tensor([current_node.idx], device='cuda')),dim=0)
                        self.operation.active_mode.node_idx_subtree_b = self.operation.active_mode.node_idx_subtree_b[self.operation.active_mode.node_idx_subtree_b != current_node.idx]
                        self.operation.active_mode.batch_subtree_b = self.operation.active_mode.batch_subtree_b[~torch.all(self.operation.active_mode.batch_subtree_b == current_node.q_tensor.unsqueeze(0), dim=1)]
                        self.operation.active_mode.subtree_b.remove(current_node)

            children = current_node.children
            if children:
                for _, child in enumerate(children):
                    child.cost = current_node.cost + child.cost_to_parent
                    child.agent_dists = current_node.agent_dists + child.agent_dists_to_parent
                    
                stack.extend(children)
    
    def ManageTransition(self, n_new:Node, iter:int) -> None:
        if self.env.get_active_task(self.operation.active_mode.label).goal.satisfies_constraints(n_new.state.q.state()[self.operation.active_mode.indices], self.config.goal_radius):
            self.operation.active_mode.transition_nodes.append(n_new)
            n_new.transition = True
            self.AddTransitionNode(n_new)
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
            if not n:
                self.operation.active_mode.transition_nodes.append(n_parent) 
                return 
            print(n.state.mode)
            n.children.remove(n_parent)
  
    def SWAP(self) -> None:
        if not self.operation.active_mode.connected:
            self.operation.active_mode.subtree, self.operation.active_mode.subtree_b = self.operation.active_mode.subtree_b, self.operation.active_mode.subtree
            self.operation.active_mode.batch_subtree, self.operation.active_mode.batch_subtree_b = self.operation.active_mode.batch_subtree_b, self.operation.active_mode.batch_subtree
            self.operation.active_mode.node_idx_subtree, self.operation.active_mode.node_idx_subtree_b = self.operation.active_mode.node_idx_subtree_b, self.operation.active_mode.node_idx_subtree 
            self.operation.active_mode.order *= (-1)
 
    def Connect(self, n_new:Node, iter:int) -> None:
        if self.operation.active_mode.subtree_b == []:
            return
        n_nearest_b=  self.Nearest(n_new.state.q, self.operation.active_mode.subtree_b, self.operation.active_mode.batch_subtree_b)
        if self.config.birrtstar_version == 1 or self.config.birrtstar_version == 2 and self.operation.active_mode.connected: #Based on paper Bi-RRT* by B. Wang 
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
          

        elif self.config.birrtstar_version == 2 and not self.operation.active_mode.connected: #Based on paper RRT-Connect by JJ. Kuffner/ RRT*-Connect by S.Klemm
            n_nearest_b = self.Extend(n_nearest_b, n_new.state.q)
            if not n_nearest_b:
                return
            cost, dists = batch_config_torch(n_new.state.q, n_nearest_b.q_tensor.unsqueeze(0), "euclidean")
 
        if self.operation.active_mode.order == -1: #Switch such that subtree is beginning from start and subtree_b from goal
            self.SWAP() 
            self.UpdateTree(n_new, n_nearest_b, cost[0], dists) 
        else:
            self.UpdateTree(n_nearest_b, n_new, cost[0], dists)
        if self.operation.active_mode.connected:
            self.AddTransitionNode(self.operation.active_mode.transition_nodes[-1])
        
        if not self.operation.active_mode.connected: #initial sol of this mode hasn't been found yet
            self.operation.active_mode.connected = True
            self.InitializationMode(self.operation.modes[-1])
            if self.env.terminal_mode != self.operation.modes[-1].label:
                self.operation.modes.append(Mode(self.env.get_next_mode(n_new.state.q,self.operation.active_mode.label), self.env))
                # Initialization of new goal tree T_b
                if self.operation.modes[-1].label == self.env.terminal_mode:
                    nr = 1
                else:
                    nr = self.config.transition_nodes
                for _ in range(nr):
                    end_q = self.sampling.get_goal_config_of_mode(self.operation.modes[-1])
                    end_state = State(type(self.env.get_start_pos())(end_q, n_new.state.q.slice), self.operation.modes[-1].label)
                    end_node = Node(end_state, self.operation.tree, self.operation)
                    self.operation.tree += 1
                    end_node.transition = True
                    self.operation.modes[-1].batch_subtree_b = torch.cat((self.operation.modes[-1].batch_subtree_b, end_node.q_tensor.unsqueeze(0)), dim=0)
                    self.operation.modes[-1].subtree_b.append(end_node)
                    self.operation.costs = torch.cat((self.operation.costs, torch.tensor([0], device='cuda')), dim=0)
                    self.operation.modes[-1].node_idx_subtree_b = torch.cat((self.operation.modes[-1].node_idx_subtree_b, torch.tensor([end_node.idx], device='cuda')),dim=0)
            elif self.operation.active_mode.label == self.env.terminal_mode:
                self.operation.ptc_iter = iter
                self.operation.ptc_cost = self.operation.cost
                self.operation.init_sol = True
                print(time.time()-self.start)
            print(f"{iter}: {self.operation.active_mode.constrained_robots} found T{self.env.get_current_seq_index(self.operation.active_mode.label)}")
            self.SaveData(time.time()-self.start)
            self.FindOptimalTransitionNode(iter, True)
            self.AddTransitionNode(self.operation.active_mode.transition_nodes[-1])
            return 
        
        self.FindOptimalTransitionNode(iter)
       
    def Extend(self, n_nearest_b:Node, q:Configuration )-> Optional[Node]:
        #RRT not RRT*
        while True:
            n_new = self.Steer(n_nearest_b, q, self.operation.active_mode.label)
            if not n_new or np.equal(n_new.state.q.state(), q.state()).all(): # Reached
                # self.SaveData(time.time()-self.start, n_nearest = n_nearest_b.state.q.state(), n_rand = q.state(), n_new = n_new.state.q.state())
                return n_nearest_b
            # self.SaveData(time.time()-self.start, n_nearest = n_nearest_b.state.q.state(), n_rand = q.state(), n_new = n_new.state.q.state())
            if self.env.is_collision_free(n_new.state.q.state(), self.operation.active_mode.label) and self.env.is_edge_collision_free(n_nearest_b.state.q, n_new.state.q, self.operation.active_mode.label):
                # Add n_new to tree
                cost, dists = batch_config_torch(n_new.state.q, n_nearest_b.q_tensor.unsqueeze(0), "euclidean")
                c_min = n_nearest_b.cost + cost.view(1)[0]

                n_new.parent = n_nearest_b
                n_new.cost_to_parent = cost.view(1)[0]
                n_nearest_b.children.append(n_new) #Set child
                n_new.agent_dists = n_new.parent.agent_dists + dists
                n_new.agent_dists_to_parent = dists   
                self.operation.tree +=1
                self.operation.active_mode.subtree_b.append(n_new)
                self.operation.active_mode.batch_subtree_b = torch.cat((self.operation.active_mode.batch_subtree_b, n_new.q_tensor.unsqueeze(0)), dim=0)
                self.operation.costs = torch.cat((self.operation.costs, c_min.unsqueeze(0)), dim=0) #set cost of n_new
                self.operation.active_mode.node_idx_subtree_b = torch.cat((self.operation.active_mode.node_idx_subtree_b, torch.tensor([n_new.idx], device='cuda')),dim=0)
                # self.SaveData(time.time()-self.start, n_nearest = n_nearest_b.state.q.state(), n_rand = q.state(), n_new = n_new.state.q.state())
                n_nearest_b = n_new
            else:
                return 

    def PlannerInitialization(self) -> None:
        active_mode = self.operation.modes[0]
        start_state = State(self.env.start_pos, active_mode.label)
        start_node = Node(start_state, (self.operation.tree -1), self.operation)
        active_mode.batch_subtree = torch.cat((active_mode.batch_subtree, start_node.q_tensor.unsqueeze(0)), dim=0)
        active_mode.subtree.append(start_node)
        active_mode.node_idx_subtree = torch.cat((active_mode.node_idx_subtree, torch.tensor([start_node.idx], device='cuda')),dim=0)
        for _ in range(self.config.transition_nodes): 
            end_q = self.sampling.get_goal_config_of_mode(active_mode)
            end_state = State(type(self.env.get_start_pos())(end_q, start_node.state.q.slice), active_mode.label)
            end_node = Node(end_state, self.operation.tree, self.operation)
            self.operation.tree += 1
            end_node.transition = True
            active_mode.batch_subtree_b = torch.cat((active_mode.batch_subtree_b, end_node.q_tensor.unsqueeze(0)), dim=0)
            active_mode.subtree_b.append(end_node)
            active_mode.node_idx_subtree_b = torch.cat((active_mode.node_idx_subtree_b, torch.tensor([end_node.idx], device='cuda')),dim=0)
            self.operation.costs = torch.cat((self.operation.costs, torch.tensor([0], device='cuda')), dim=0)
                  
    def Plan(self) -> dict:
        i = 0
        self.PlannerInitialization()
        while True:
            # Mode selection
            self.operation.active_mode  = (np.random.choice(self.operation.modes, p= self.SetModePorbability()))
            
            # Bi-RRT* core
            q_rand = self.SampleNodeManifold(self.operation)
            
            n_nearest= self.Nearest(q_rand, self.operation.active_mode.subtree, self.operation.active_mode.batch_subtree)    
            n_new = self.Steer(n_nearest, q_rand, self.operation.active_mode.label)
            if not n_new: # meaning n_new is exact the same as one of the nodes in the tree
                continue
            
            if self.env.is_collision_free(n_new.state.q.state(), self.operation.active_mode.label) and self.env.is_edge_collision_free(n_nearest.state.q, n_new.state.q, self.operation.active_mode.label):
                N_near_indices, N_near_batch, n_near_costs, node_indices = self.Near(n_new, self.operation.active_mode.batch_subtree)
                if n_nearest.idx not in node_indices:
                    continue
                n_nearest_index = torch.where(node_indices == n_nearest.idx)[0].item() 
                batch_cost, batch_dist =  batch_config_torch(n_new.state.q, N_near_batch, metric = self.config.cost_type)
                self.FindParent(N_near_indices, n_nearest_index, n_new, n_nearest, batch_cost, batch_dist, n_near_costs)
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
        print(time.time()-self.start)
        return self.operation.path    




