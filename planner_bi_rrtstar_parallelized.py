from configuration import *
from planning_env import *
from util import *
from rrt_base import *
import time as time

class ParallelizedBidirectionalRRTstar(BaseRRTstar):
    def __init__(self, env, config: ConfigManager):
        super().__init__(env, config)
    
    def UpdateCost(self, n:Node, connection:bool = False) -> None:
        stack = [n]
        while stack:
            current_node = stack.pop()
            if connection:
                #add node to main subtree
                if current_node.idx not in self.operation.active_mode.node_idx_subtree:
                    self.operation.active_mode.subtree[len(self.operation.active_mode.subtree)] = current_node
                    self.operation.active_mode.batch_subtree = torch.cat((self.operation.active_mode.batch_subtree, current_node.q_tensor.unsqueeze(0)), dim=0)
                    self.operation.active_mode.node_idx_subtree = torch.cat((self.operation.active_mode.node_idx_subtree, torch.tensor([current_node.idx], device='cuda')),dim=0)
                else:
                    continue
                #remove node from subtree b ->

            
            children = current_node.children
            if children:
                for _, child in enumerate(children):
                    child.cost = current_node.cost + child.cost_to_parent
                    child.agent_cost = current_node.agent_cost + child.agent_cost_to_parent
                    
                stack.extend(children)
    
    def ManageTransition(self, n_new:Node, iter:int) -> None:
            constrained_robots = self.env.get_active_task(self.operation.active_mode.label).robots
            indices = []
            radius = 0
            for r in constrained_robots:
                indices.extend(self.env.robot_idx[r])
                radius += self.config.goal_radius
            if self.env.get_active_task(self.operation.active_mode.label).goal.satisfies_constraints(n_new.state.q.state()[indices], self.config.goal_radius):
                # if not self.operation.active_mode.connected:
                #     n_new.transition = True
                #     return
                self.operation.active_mode.transition_nodes.append(n_new)
                n_new.transition = True
                self.AddTransitionNode(n_new)
            self.FindOptimalTransitionNode(iter)
 
    def UpdateTree(self, n: Node, n_parent:Node , cost_to_parent:torch.Tensor, dists:torch.Tensor) -> None: #TODO need to update it
        while True:
            n.cost = n_parent.cost +  cost_to_parent
            n.agent_cost = n_parent.agent_cost + dists
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
                return 
            n.children.remove(n_parent)
  
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
  
    def SWAP(self) -> None:
        if not self.operation.active_mode.connected:
            self.operation.active_mode.subtree, self.operation.active_mode.subtree_b = self.operation.active_mode.subtree_b, self.operation.active_mode.subtree
            self.operation.active_mode.batch_subtree, self.operation.active_mode.batch_subtree_b = self.operation.active_mode.batch_subtree_b, self.operation.active_mode.batch_subtree
            self.operation.active_mode.node_idx_subtree, self.operation.active_mode.node_idx_subtree_b = self.operation.active_mode.node_idx_subtree_b, self.operation.active_mode.node_idx_subtree 
            self.operation.active_mode.order *= (-1)
 
    def Connect(self, n_new:Node, iter:int) -> None: 
        n_nearest_b=  self.Nearest(n_new.state.q, self.operation.active_mode.subtree_b, self.operation.active_mode.batch_subtree_b)
        if self.operation.active_mode.connected: # First connection already exists
                if n_nearest_b in self.operation.active_mode.subtree.values(): # Connection not possible as already in main subtree
                    return

        if self.config.birrtstar_version == 1 or self.config.birrtstar_version == 2 and self.operation.active_mode.connected: #Based on paper Bi-RRT* by B. Wang 
            #TODO rewire everything?
            #TODO only check dist of active robots to connect (cost can be extremly high)? or the smartest way to just connect when possible?
          
            # constrained_robots = self.env.get_active_task(self.operation.active_mode.label).robots
            # cost, dists = batch_config_torch(n_new.state.q, n_nearest_b.q_tensor.unsqueeze(0), "euclidean")
            # relevant_dists = []
            # for r_idx, r in enumerate(self.env.robots):
            #     if r in constrained_robots:
            #         relevant_dists.append(dists[0][r_idx].item())
            # if np.max(relevant_dists) > self.config.step_size:
            #     return

            # if torch.max(dists).item() > self.config.step_size:
            #     return

            if not self.env.is_edge_collision_free(n_new.state.q, n_nearest_b.state.q, self.operation.active_mode.label): 
                return

            

        elif self.config.birrtstar_version == 2 and not self.operation.active_mode.connected: #Based on paper RRT-Connect by JJ. Kuffner/ RRT*-Connect by S.Klemm
            n_nearest_b = self.Extend(n_nearest_b, n_new.state.q)
            if not n_nearest_b:
                return

        cost, dists = batch_config_torch(n_new.state.q, n_nearest_b.q_tensor.unsqueeze(0), "euclidean")
        if self.operation.active_mode.order == -1: # Meaning we switch again such that subtree is beginning from start and subtree_b from goal
            self.SWAP() 
            self.UpdateTree(n_new, n_nearest_b, cost[0], dists) 
        else:
            self.UpdateTree(n_nearest_b, n_new, cost[0], dists)
        if self.operation.active_mode.connected:
            self.AddTransitionNode(self.operation.active_mode.transition_nodes[-1])
        
        if not self.operation.active_mode.connected: #initial sol of mode hasn't been found yet -> connect trees for the frist time
            self.operation.active_mode.connected = True
            if self.operation.active_mode.label == self.env.terminal_mode:
                self.operation.ptc_iter = iter
                self.operation.ptc_cost = self.operation.cost
                self.operation.init_sol = True
                print(time.time()-self.start)

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
                return n_nearest_b
            self.SaveData(time.time()-self.start, n_nearest = n_nearest_b.state.q.state(), n_rand = q.state(), n_new = n_new.state.q.state())
            if self.env.is_collision_free(n_new.state.q.state(), self.operation.active_mode.label) and self.env.is_edge_collision_free(n_nearest_b.state.q, n_new.state.q, self.operation.active_mode.label):
                # Add n_new to tree
                cost, dists = batch_config_torch(n_new.state.q, n_nearest_b.q_tensor.unsqueeze(0), "euclidean")
                c_min = n_nearest_b.cost + cost.view(1)[0]

                n_new.parent = n_nearest_b
                n_new.cost_to_parent = cost.view(1)[0]
                n_nearest_b.children.append(n_new) #Set child
                n_new.agent_cost = n_new.parent.agent_cost + dists
                n_new.agent_cost_to_parent = dists   
                self.operation.tree +=1
                self.operation.active_mode.subtree_b[len(self.operation.active_mode.subtree_b)] = n_new
                self.operation.active_mode.batch_subtree_b = torch.cat((self.operation.active_mode.batch_subtree_b, n_new.q_tensor.unsqueeze(0)), dim=0)
                self.operation.costs = torch.cat((self.operation.costs, c_min.unsqueeze(0)), dim=0) #set cost of n_new
                self.operation.active_mode.node_idx_subtree_b = torch.cat((self.operation.active_mode.node_idx_subtree_b, torch.tensor([n_new.idx], device='cuda')),dim=0)
                self.SaveData(time.time()-self.start, n_nearest = n_nearest_b.state.q.state(), n_rand = q.state(), n_new = n_new.state.q.state())
                n_nearest_b = n_new

            else:
                return 
                                 
    def PlannerInitialization(self) -> None:
        # Initialize all nodes:
        # Initialization of frist mode (handled separately as start condition is given for all robots)
        active_mode = self.operation.modes[0]
        start_state = State(self.env.start_pos, active_mode.label)
        start_node = Node(start_state, (self.operation.tree -1), self.operation)
        active_mode.batch_subtree = torch.cat((active_mode.batch_subtree, start_node.q_tensor.unsqueeze(0)), dim=0)
        active_mode.subtree[len(active_mode.subtree)] = start_node
        active_mode.node_idx_subtree = torch.cat((active_mode.node_idx_subtree, torch.tensor([start_node.idx], device='cuda')),dim=0)

        while True:
            #Add mode to mode list
            self.InitializationMode(self.operation.modes[-1])            
            if active_mode.label == self.env.terminal_mode:
                nr = 1
            else:
                self.operation.modes.append( Mode(self.env.get_next_mode(None,active_mode.label), self.env))
                nr = self.config.transition_nodes
            for _ in range(nr): 
                end_state = self.GetGoalStateOfMode(active_mode.label, start_node)
                end_node = Node(end_state, self.operation.tree, self.operation)
                self.operation.tree += 1
                # possible goal nodes for this mode #TODO also need to add them as start nodes for next mode
                end_node.transition = True
                active_mode.batch_subtree_b = torch.cat((active_mode.batch_subtree_b, end_node.q_tensor.unsqueeze(0)), dim=0)
                active_mode.subtree_b[len(active_mode.subtree_b)] = end_node
                active_mode.node_idx_subtree_b = torch.cat((active_mode.node_idx_subtree_b, torch.tensor([end_node.idx], device='cuda')),dim=0)
                self.operation.costs = torch.cat((self.operation.costs, torch.tensor([0], device='cuda')), dim=0)

                if active_mode.label == self.env.terminal_mode:
                    return
                self.operation.modes[-1].batch_subtree = torch.cat((self.operation.modes[-1].batch_subtree, end_node.q_tensor.unsqueeze(0)), dim=0)
                self.operation.modes[-1].subtree[len(self.operation.modes[-1].subtree)] = end_node
                self.operation.costs = torch.cat((self.operation.costs, torch.tensor([0], device='cuda')), dim=0)
                self.operation.modes[-1].node_idx_subtree = torch.cat((self.operation.modes[-1].node_idx_subtree, torch.tensor([end_node.idx], device='cuda')),dim=0)
                
            
            #TODO need to change the costs if subtree of next mode as its not right yet ...
            # Update active_mode
            active_mode =  self.operation.modes[-1]
             
    def Plan(self) -> dict:
        i = 0
        self.PlannerInitialization()
        while True:
            # Mode selection
            self.operation.active_mode  = (np.random.choice(self.operation.modes, p= self.SetModePorbability()))
            
            # Bi-RRT* core
            q_rand, sampling_type = self.SampleNodeManifold(self.operation)
            if sampling_type == 2 and self.operation.active_mode.order == -1: # A start of this node was sampled and need to add it as goal node to previous mode
                pass #TODO
            if sampling_type == 2 and self.operation.active_mode.order == 1: # A goal of this node was sampled and need to add it as start node to next mode
                pass #TODO

            n_nearest= self.Nearest(q_rand, self.operation.active_mode.subtree, self.operation.active_mode.batch_subtree)    
            n_new = self.Steer(n_nearest, q_rand, self.operation.active_mode.label)
            if not n_new: # meaning n_new is exactly the same as one of the nodes in the tree
                continue

            if self.env.is_collision_free(n_new.state.q.state(), self.operation.active_mode.label) and self.env.is_edge_collision_free(n_nearest.state.q, n_new.state.q, self.operation.active_mode.label):
                N_near_indices, N_near_batch, n_near_costs, node_indices = self.Near(n_new, self.operation.active_mode.batch_subtree)
                if n_nearest.idx not in node_indices:
                    continue
                n_nearest_index = torch.where(node_indices == n_nearest.idx)[0].item() 
                batch_cost, batch_dist =  batch_config_torch(n_new.state.q, N_near_batch, "euclidean")
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
        return self.operation.path    




