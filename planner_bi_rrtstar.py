from configuration import *
from planning_env import *
from util import *
from planner_base import *
import time as time
import math

class BidirectionalRRTstar:
    def __init__(self, env, config: ConfigManager):
        self.env = env
        # self.gamma = ((2 *(1 + 1/self.dim))**(1/self.dim) * (self.FreeSpace()/self.UnitBallVolume())**(1/self.dim))*1.1 
        self.config = config
        self.r = self.config.goal_radius * sum(self.env.robot_dims.values())
        self.operation = Operation(env)
        self.sampling = Sampling(env, self.operation, self.config)
        self.start = time.time()
     
    def Nearest(self, n_rand, subtree, batch_subtree): 
        set_dists = batch_config_dist_torch(n_rand.state.q, batch_subtree, "euclidean")
        idx = torch.argmin(set_dists).item()
        return  subtree[idx]
      
    def Steer(self,
        n_nearest: Node,
        n_rand: Node,
        m: List[int]
    ) -> bool:    

        dists = config_dists(n_nearest.state.q, n_rand.state.q, "euclidean")
        q_new = []
        q_nearest = n_nearest.state.q.state()
        q_rand = n_rand.state.q.state()
        direction = q_rand - q_nearest
        for idx, robot in enumerate(self.env.robots):
            indices = self.env.robot_idx[robot]
            robot_dist = dists[idx]
            if robot_dist < self.config.step_size:
                q_new.append(q_rand[indices])
            else:
                if robot_dist == 0:
                    t = 0
                else:
                    t = min(1, self.config.step_size / robot_dist) 
                q_new.append(q_nearest[indices] + t * direction[indices])
        q_new = np.concatenate(q_new, axis=0)
        q_new = np.clip(q_new, self.env.limits[0], self.env.limits[1]) 
        state_new = State(type(self.env.get_start_pos())(q_new, n_nearest.state.q.slice), m)
        n_new = Node(state_new)
        return n_new
  
    def Near(self, n_new, subtree_set):
        #TODO generalize rewiring radius
        # n_nodes = sum(1 for _ in self.operation.current_mode.subtree.inorder()) + 1
        # r = min((7)*self.step_size, 3 + self.gamma * ((math.log(n_nodes) / n_nodes) ** (1 / self.dim)))
        set_dists = batch_config_dist_torch(n_new.state.q, subtree_set, "euclidean")
        indices = torch.where(set_dists < self.r)[0]
        N_near = [self.operation.active_mode.subtree[idx] for idx in indices.tolist()]
        N_near_batch = self.operation.active_mode.batch_subtree[indices, :]
        if not self.config.informed_sampling:
            return N_near, N_near_batch
        return self.sampling.fit_to_informed_subset(N_near, N_near_batch) 
                
    def FindParent(self, N_near, n_new, n_nearest, batch_cost, batch_dist, n_near_costs):
        n_nearest_index = N_near.index(n_nearest)
        c_min = n_near_costs[n_nearest_index] + batch_cost[n_nearest_index]
        c_min_to_parent = batch_cost[n_nearest_index]
        n_min = n_nearest
        c_new_tensor = n_near_costs + batch_cost
        valid_indices = torch.nonzero(c_new_tensor < c_min, as_tuple=False).squeeze()
        valid_indices = torch.unique(valid_indices)
        if valid_indices.numel() > 0:
            sorted_indices = valid_indices[torch.argsort(c_new_tensor[valid_indices])]
            for idx in sorted_indices:
                idx = idx.item() 
                if self.env.is_edge_collision_free(
                    N_near[idx].state.q, n_new.state.q, self.operation.active_mode.label
                ):
                    c_min = c_new_tensor[idx]        # Update minimum cost
                    c_min_to_parent = batch_cost[idx]
                    n_min = N_near[idx]              # Update nearest node
                    break

        n_new.cost = c_min
        n_new.parent = n_min
        n_min.children.append(n_new) #Set child
        n_new.agent_cost = n_new.parent.agent_cost + batch_dist[N_near.index(n_min)].unsqueeze(0) 
        n_new.agent_cost_to_parent = batch_dist[N_near.index(n_min)].unsqueeze(0)
        n_new.cost_to_parent = c_min_to_parent
        self.operation.tree +=1
        self.operation.active_mode.subtree[len(self.operation.active_mode.subtree)] = n_new
        self.operation.active_mode.batch_subtree = torch.cat((self.operation.active_mode.batch_subtree, n_new.q_tensor.unsqueeze(0)), dim=0)

    def UnitBallVolume(self):
        return math.pi ** (self.dim / 2) / math.gamma((self.dim / 2) + 1)
    
    def Rewire(self, N_near, n_new, batch_cost, batch_dist, n_near_costs):
        rewired = False
        c_potential_tensor = n_new.cost + batch_cost
        c_agent_tensor = batch_dist + n_new.agent_cost

        all_indices = torch.arange(len(N_near), device=n_near_costs.device)
        # Find indices where rewiring improves cost
        improvement_mask = c_potential_tensor < n_near_costs
        improved_indices = all_indices[improvement_mask].tolist()  
        for idx in improved_indices:
            n_near = N_near[idx]
            if n_near == n_new.parent:
                continue

            if self.env.is_edge_collision_free(n_near.state.q, n_new.state.q, self.operation.active_mode.label):
                if n_near.parent is not None:
                    n_near.parent.children.remove(n_near)

                n_near.parent = n_new
                if n_new != n_near:
                    n_new.children.append(n_near)

                n_near.cost = c_potential_tensor[idx].item()
                n_near.agent_cost = c_agent_tensor[idx].unsqueeze(0) 
                n_near.cost_to_parent = batch_cost[idx]
                n_near.agent_cost_to_parent = batch_dist[idx].unsqueeze(0)
                rewired = True

        return rewired
      
    def GeneratePath(self, node):
        path_nodes, path = [], []
        while node:
            path_nodes.append(node)
            path.append(node.state)
            node = node.parent
        path_in_order = path[::-1]
        # if self.env.is_valid_plan(path_in_order): 
        self.operation.path = path_in_order  
        # print([state.q.state() for state in path_in_order])
        self.operation.path_nodes = path_nodes[::-1]
        self.operation.cost = self.operation.path_nodes[-1].cost

    def UpdateCost(self, n, connection = False):
        stack = [n]
        while stack:
            current_node = stack.pop()
            if connection:
                        if current_node not in self.operation.active_mode.subtree.values():
                            self.operation.active_mode.subtree[len(self.operation.active_mode.subtree)] = current_node
                            self.operation.active_mode.batch_subtree = torch.cat((self.operation.active_mode.batch_subtree, current_node.q_tensor.unsqueeze(0)), dim=0)
            children = current_node.children
            if children:
                for _, child in enumerate(children):
                    child.cost = current_node.cost + child.cost_to_parent
                    child.agent_cost = current_node.agent_cost + child.agent_cost_to_parent
                    
                stack.extend(children)

    def UpdateTree(self, n, n_parent, cost_to_parent, dists): #TODO need to update it
        while True:
            n.cost = n_parent.cost +  cost_to_parent
            n.agent_cost = n_parent.parent.agent_cost + dists
            self.UpdateCost(n, True)
            n_parent_inter = n.parent
            cost_to_parent_inter = n.cost_to_parent
            agent_cost_to_parent_inter = n.agent_cost_to_parent

            n.parent = n_parent
            n.cost_to_parent = cost_to_parent
            n.agent_cost_to_parent = dists
            n_parent.children.append(n) #Set child
            
            cost_to_parent = cost_to_parent_inter
            dists = agent_cost_to_parent_inter
            n_parent = n
            n = n_parent_inter
            if not n:
                self.operation.active_mode.transition_nodes.append(n_parent)
                return n_parent
            n.children.remove(n_parent)
        
    def FindOptimalTransitionNode(self, iteration, connection = False):
        transition_nodes = self.operation.modes[-1].transition_nodes
        if not transition_nodes:
            return
        costs = torch.cat([node.cost.unsqueeze(0) for node in transition_nodes])
        if connection:
            valid_mask = costs < np.inf

        else:
            valid_mask = costs < self.operation.cost

        if valid_mask.any(): 
            # Find the index of the node with the lowest cost among valid nodes
            valid_costs = costs[valid_mask]
            min_cost_idx = torch.argmin(valid_costs).item()

            # Map the valid index back to the original node index
            lowest_cost_node = transition_nodes[torch.nonzero(valid_mask)[min_cost_idx].item()]

            self.GeneratePath(lowest_cost_node)
            if not connection:
                print(f"iter  {iteration}: Changed cost to ", self.operation.cost, " Mode ", self.operation.active_mode.label)

            if (self.operation.ptc_cost - self.operation.cost) > self.config.ptc_threshold:
                self.operation.ptc_cost = self.operation.cost
                self.operation.ptc_iter = iteration

            save_data(self.config, self.operation, time.time()-self.start)

    def AddTransitionNode(self, n): #Don't need both trees
            idx = self.operation.modes.index(self.operation.active_mode)
            if idx != len(self.operation.modes) - 1:
                self.operation.modes[idx + 1].subtree[len(self.operation.modes[idx + 1].subtree)] = n
                self.operation.modes[idx + 1].batch_subtree = torch.cat((self.operation.modes[idx + 1].batch_subtree, n.q_tensor.unsqueeze(0)), dim=0)

    def ManageTransition(self, n_new, iteration):
            constrained_robots = self.env.get_active_task(self.operation.active_mode.label).robots
            indices = []
            radius = 0
            for r in constrained_robots:
                indices.extend(self.env.robot_idx[r])
                radius += self.config.goal_radius
            if self.env.get_active_task(self.operation.active_mode.label).goal.satisfies_constraints(n_new.state.q.state()[indices], self.config.goal_radius):
                if not self.operation.active_mode.connected:
                    n_new.transition = True
                    return
                self.operation.active_mode.transition_nodes.append(n_new)
                n_new.transition = True
                self.AddTransitionNode(n_new)
            self.FindOptimalTransitionNode(iteration)

    def SetModePorbability(self):
        num_modes = len(self.operation.modes)
        if num_modes == 1:
            return [1] 
        # if self.operation.task_sequence == [] and self.config.mode_probability != 0:
        if self.operation.init_path and self.config.mode_probability != 0:
                return [1/num_modes] * num_modes
        
        elif self.config.mode_probability == 'None':
            # equally
            return [1 / (num_modes)] * (num_modes)

        elif self.config.mode_probability == 1:
            # greedy (only latest mode is selected until all initial paths are found)
            probability = [0] * (num_modes)
            probability[-1] = 1
            return probability

        elif self.config.mode_probability == 0:
            # Uniformly
            total_transition_nodes = sum(len(mode.transition_nodes) for mode in self.operation.modes)
            total_nodes = self.operation.tree + total_transition_nodes
            # Calculate probabilities inversely proportional to node counts
            inverse_probabilities = [
                1 - (len(mode.subtree) / total_nodes)
                for mode in self.operation.modes
            ]
            # Normalize the probabilities to sum to 1
            total_inverse = sum(inverse_probabilities)
            return [
                inv_prob / total_inverse for inv_prob in inverse_probabilities
            ]

        else:
            # manually set
            total_transition_nodes = sum(len(mode.transition_nodes) for mode in self.operation.modes)
            total_nodes = self.operation.tree_size + total_transition_nodes
            # Calculate probabilities inversely proportional to node counts
            inverse_probabilities = [
                1 - (len(mode.subtree) / total_nodes)
                for mode in self.operation.modes[:-1]  # Exclude the last mode
            ]

            # Normalize the probabilities of all modes except the last one
            remaining_probability = 1-self.config.mode_probability  
            total_inverse = sum(inverse_probabilities)
            return [
                (inv_prob / total_inverse) * remaining_probability
                for inv_prob in inverse_probabilities
            ] + [self.config.mode_probability]

    def InitializationMode(self, m:Mode):
        mode = self.env.start_mode
        robots = []
        goal = {}
        state_start = {}
        while True:
            task = self.env.get_active_task(mode)
            constrained_robtos = task.robots
            for robot in constrained_robtos:
                indices = self.env.robot_idx[robot]
                r = self.env.robots.index(robot)
                if r not in robots: 
                    robots.append(r)
                    state_start[r] = self.env.start_pos.q[indices]
                else:
                    state_start[r] = goal[r]
                if len(constrained_robtos) > 1:
                    goal[r] = task.goal.sample()[indices]
                else:
                    goal[r] = task.goal.sample()
                
                if m.label == mode:
                    m.informed.start[r] = state_start[r]
                    m.informed.goal[r] = goal[r]
                    if self.config.informed_sampling and not np.equal(m.informed.goal[r], m.informed.start[r]).all():
                        cmin, C = self.sampling.rotation_to_world_frame(state_start[r], goal[r] ,robot)
                        C = torch.tensor(C, device = 'cuda')
                        m.informed.C[r] = C
                        m.informed.inv_C[r] = torch.linalg.inv(C)
                        m.informed.cmin[r] = torch.tensor(cmin-2*self.config.goal_radius, device= 'cuda')
                        m.informed.state_centre[r] = torch.tensor(((state_start[r] + goal[r])/2), device='cuda')

                    if robot == constrained_robtos[-1]:
                        return
            mode = self.env.get_next_mode(None,mode)

    def Initialization(self):
        active_mode = self.operation.modes[0]
        start_state = State(self.env.start_pos, active_mode.label)
        start_node = Node(start_state)
        active_mode.batch_subtree = torch.cat((active_mode.batch_subtree, start_node.q_tensor.unsqueeze(0)), dim=0)
        active_mode.subtree[len(active_mode.subtree)] = start_node
        # Bidirectional
        for _ in range(100): 
            end_state = self.GetGoalStateOfMode(active_mode.label, start_node)
            first_end_node = Node(end_state)
            self.operation.tree += 1
            # active_mode.transition_nodes.append(first_end_node)
            first_end_node.transition = True
            active_mode.batch_subtree_b = torch.cat((active_mode.batch_subtree_b, first_end_node.q_tensor.unsqueeze(0)), dim=0)
            active_mode.subtree_b[len(active_mode.subtree_b)] = first_end_node
       
    def SampleNodeManifold(self, operation: Operation):
        if np.random.uniform(0, 1) <= self.config.p_goal:
            if self.env.terminal_mode != operation.modes[-1].label and operation.active_mode.label == operation.modes[-1].label:
                return Node(self.sampling.sample_state(operation.active_mode, 0, self.config)) 
            else:  
                if self.config.informed_sampling: 
                    return Node(self.sampling.sample_state(operation.active_mode, 1, self.config))
                return Node(self.sampling.sample_state(operation.active_mode, 0, self.config))
        return Node(self.sampling.sample_state(operation.active_mode, 2, self.config))
    
    #New
    def GetGoalStateOfMode(self, mode:List[int], n:Node)-> State:
        constrained_robots = self.env.get_active_task(mode).robots
        q = np.zeros(len(n.state.q.state()))
        i = 0
        for robot in self.env.robots:
            indices = self.env.robot_idx[robot]
            if robot in constrained_robots:
                if len(constrained_robots) > 1:
                    dim = self.env.robot_dims[robot]
                    q[indices] = self.env.get_active_task(mode).goal.goal[i*dim:(i+1)*dim]
                    i+=1
                else:
                    q[indices] = self.env.get_active_task(mode).goal.goal
            else:
                lims = self.env.limits[:, self.env.robot_idx[robot]]
                q[indices] = np.random.uniform(lims[0], lims[1])

        return State(type(self.env.get_start_pos())(q, n.state.q.slice), mode)
    #New
    def SWAP(self):
        if not self.operation.active_mode.connected:
            self.operation.active_mode.subtree, self.operation.active_mode.subtree_b = self.operation.active_mode.subtree_b, self.operation.active_mode.subtree
            self.operation.active_mode.batch_subtree, self.operation.active_mode.batch_subtree_b = self.operation.active_mode.batch_subtree_b, self.operation.active_mode.batch_subtree
            self.operation.active_mode.order *= (-1)
    #New
    def Connect(self, n_new, n_nearest_b, iteration):
        #Only need to connect if initial path not found for this mode
        if not self.operation.active_mode.connected:
            # dists = config_dists(n_new.state.q, n_nearest_b.state.q, "euclidean")
            cost, dists = batch_config_torch(n_new.state.q, torch.tensor([n_nearest_b.state.q.state()], device = 'cuda'), "euclidean")
            constrained_robots = self.env.get_active_task(self.operation.active_mode.label).robots
            relevant_dists = []
            for r_idx, r in enumerate(self.env.robots):
                if r in constrained_robots:
                    relevant_dists.append(dists[0][r_idx].item())

            # active_robot = 0 #TODO only connect if path for active mode is found
            if np.max(relevant_dists) < self.config.step_size:
                if self.env.is_edge_collision_free(n_new.state.q, n_nearest_b.state.q, self.operation.active_mode.label): 
                    if self.operation.active_mode.order == -1: # Meaning we switch again such that subtree is beginning from start and subtree_b from goal
                        self.SWAP() 
                        n = self.UpdateTree(n_new, n_nearest_b, cost[0], dists[0].unsqueeze(0)) #Need to make all the costs right of the other tre #TODO not working yet
                    else:
                        n = self.UpdateTree(n_nearest_b, n_new, cost[0], dists[0].unsqueeze(0)) #Need to make all the costs right of the other tre #TODO not working yet
                     
                    self.FindOptimalTransitionNode(iteration, True)
                    self.operation.active_mode.connected = True
                    print(f"iter  {iteration}: {constrained_robots} found T{self.env.get_current_seq_index(self.operation.active_mode.label)}: Cost: ", self.operation.cost)
                    # if self.operation.task_sequence != []:
                    self.InitializationMode(self.operation.modes[-1])
                    if self.env.terminal_mode != self.operation.modes[-1].label:
                        self.operation.modes.append(Mode(self.env.get_next_mode(n_new.state.q,self.operation.active_mode.label), self.env))
                        # Need to initialize goal tree
                        if self.operation.modes[-1].label == self.env.terminal_mode:
                            nr = 1
                        else:
                            nr = 100
                        for _ in range(nr):
                            end_state = self.GetGoalStateOfMode(self.operation.modes[-1].label, n_new)
                            first_end_node = Node(end_state)
                            self.operation.tree += 1
                            # self.operation.modes[-1].transition_nodes.append(first_end_node)
                            first_end_node.transition = True
                            self.operation.modes[-1].batch_subtree_b = torch.cat((self.operation.modes[-1].batch_subtree_b, first_end_node.q_tensor.unsqueeze(0)), dim=0)
                            self.operation.modes[-1].subtree_b[len(self.operation.modes[-1].subtree_b)] = first_end_node
                        self.AddTransitionNode(n)

                    elif self.operation.active_mode.label == self.env.terminal_mode:
                        self.operation.ptc_iter = iteration
                        self.operation.ptc_cost = self.operation.cost
                        self.operation.init_path = True
                    save_data(self.config, self.operation, time.time()-self.start, init_path=self.operation.init_path)
                                   
                #TODO rewire everything?
                #TODO only check dist of active robots to connect?

    def Plan(self) -> dict:
        i = 0
        self.Initialization()
        while True:
            # Mode selection
            self.operation.active_mode  = (np.random.choice(self.operation.modes, p= self.SetModePorbability()))
            # InitializationMode of active mode

            # RRT* core
            n_rand = self.SampleNodeManifold(self.operation)
            n_nearest = self.Nearest(n_rand, self.operation.active_mode.subtree, self.operation.active_mode.batch_subtree)    
            n_new = self.Steer(n_nearest, n_rand, self.operation.active_mode.label)

            if self.env.is_collision_free(n_new.state.q.state(), self.operation.active_mode.label) and self.env.is_edge_collision_free(n_nearest.state.q, n_new.state.q, self.operation.active_mode.label):
                N_near, N_near_batch = self.Near(n_new, self.operation.active_mode.batch_subtree)
                if n_nearest not in N_near:
                    continue
                batch_cost, batch_dist =  batch_config_torch(n_new.state.q, N_near_batch, "euclidean")
                n_near_costs = torch.stack([n.cost for n in N_near])
                self.FindParent(N_near, n_new, n_nearest, batch_cost, batch_dist, n_near_costs)
                if self.Rewire(N_near, n_new, batch_cost, batch_dist, n_near_costs):
                    self.UpdateCost(n_new)
                n_nearest_b =  self.Nearest(n_new, self.operation.active_mode.subtree_b, self.operation.active_mode.batch_subtree_b)
                self.Connect(n_new, n_nearest_b, i) 
                self.ManageTransition(n_new, i)
            if self.operation.init_path and i != self.operation.ptc_iter and (i- self.operation.ptc_iter)%self.config.ptc_max_iter == 0:
                diff = self.operation.ptc_cost - self.operation.cost
                self.operation.ptc_cost = self.operation.cost
                if diff < self.config.ptc_threshold:
                    break
            self.SWAP()
            i += 1
            

        save_data(self.config, self.operation, time.time()-self.start, n_new = n_new, N_near = N_near, r =self.r, n_rand = n_rand.state.q)
        return self.operation.path    




