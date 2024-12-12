from configuration import *
from planning_env import *
from util import *
from rrt_base import *
import time as time

class BidirectionalRRTstar(BaseRRTstar):
    def __init__(self, env, config: ConfigManager):
        super().__init__(env, config)
    
    def UpdateCost(self, n, connection = False):
        stack = [n]
        while stack:
            current_node = stack.pop()
            if connection:
                        # if current_node not in self.operation.active_mode.subtree.values():
                        self.operation.active_mode.subtree[len(self.operation.active_mode.subtree)] = current_node
                        self.operation.active_mode.batch_subtree = torch.cat((self.operation.active_mode.batch_subtree, current_node.q_tensor.unsqueeze(0)), dim=0)
                        self.operation.active_mode.node_idx_subtree = torch.cat((self.operation.active_mode.node_idx_subtree, torch.tensor([current_node.idx], device='cuda')),dim=0)
            children = current_node.children
            if children:
                for _, child in enumerate(children):
                    child.cost = current_node.cost + child.cost_to_parent
                    child.agent_cost = current_node.agent_cost + child.agent_cost_to_parent
                    
                stack.extend(children)
    
    def ManageTransition(self, n_new, iter):
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
            self.FindOptimalTransitionNode(iter)
 
    def UpdateTree(self, n, n_parent, cost_to_parent, dists): #TODO need to update it
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
  
    def SWAP(self):
        if not self.operation.active_mode.connected:
            self.operation.active_mode.subtree, self.operation.active_mode.subtree_b = self.operation.active_mode.subtree_b, self.operation.active_mode.subtree
            self.operation.active_mode.batch_subtree, self.operation.active_mode.batch_subtree_b = self.operation.active_mode.batch_subtree_b, self.operation.active_mode.batch_subtree
            self.operation.active_mode.node_idx_subtree, self.operation.active_mode.node_idx_subtree_b = self.operation.active_mode.node_idx_subtree_b, self.operation.active_mode.node_idx_subtree 
            self.operation.active_mode.order *= (-1)
 
    def Connect(self, n_new:Node, n_nearest_b:Node, iteration:int, n_nearest_b_q:torch.tensor):
        # dists = config_dists(n_new.state.q, n_nearest_b.state.q, "euclidean")
        cost, dists = batch_config_torch(n_new.state.q, n_nearest_b_q.unsqueeze(0), "euclidean")
        # constrained_robots = self.env.get_active_task(self.operation.active_mode.label).robots
        # relevant_dists = []
        # for r_idx, r in enumerate(self.env.robots):
        #     if r in constrained_robots:
        #         relevant_dists.append(dists[0][r_idx].item())

        #active_robot = 0 #TODO only connect if path for active mode is found
        #TODO if doing so the cost is high...
        #TODO rewire everything?
        #TODO only check dist of active robots to connect?
        #TODO the smartest way to just connect when possible?
        
        # if np.max(relevant_dists) < self.config.step_size:
        if self.env.is_edge_collision_free(n_new.state.q, n_nearest_b.state.q, self.operation.active_mode.label): 
            if self.operation.active_mode.order == -1: # Meaning we switch again such that subtree is beginning from start and subtree_b from goal
                self.SWAP() 
                self.UpdateTree(n_new, n_nearest_b, cost[0], dists[0].unsqueeze(0)) 
            else:
                self.UpdateTree(n_nearest_b, n_new, cost[0], dists[0].unsqueeze(0)) 
            
            self.operation.active_mode.connected = True
            self.InitializationMode(self.operation.modes[-1])
            print(time.time()-self.start)
            self.SaveData(time.time()-self.start)
            if self.env.terminal_mode != self.operation.modes[-1].label:
                self.operation.modes.append(Mode(self.env.get_next_mode(n_new.state.q,self.operation.active_mode.label), self.env))
                self.operation.active_mode.subtree_b = {}
                self.operation.active_mode.batch_subtree_b = None
                self.operation.active_mode.node_idx_subtree_b =  None
                # Initialization of new goal tree T_b
                if self.operation.modes[-1].label == self.env.terminal_mode:
                    nr = 1
                else:
                    nr = 100
                for _ in range(nr):
                    end_state = self.GetGoalStateOfMode(self.operation.modes[-1].label, n_new)
                    end_node = Node(end_state, self.operation.tree, self.operation)
                    self.operation.tree += 1
                    # self.operation.modes[-1].transition_nodes.append(first_end_node)
                    end_node.transition = True
                    self.operation.modes[-1].batch_subtree_b = torch.cat((self.operation.modes[-1].batch_subtree_b, end_node.q_tensor.unsqueeze(0)), dim=0)
                    self.operation.modes[-1].subtree_b[len(self.operation.modes[-1].subtree_b)] = end_node
                    self.operation.costs = torch.cat((self.operation.costs, torch.tensor([0], device='cuda')), dim=0)
                    self.operation.modes[-1].node_idx_subtree_b = torch.cat((self.operation.modes[-1].node_idx_subtree_b, torch.tensor([end_node.idx], device='cuda')),dim=0)
            elif self.operation.active_mode.label == self.env.terminal_mode:
                self.operation.ptc_iter = iteration
                self.operation.ptc_cost = self.operation.cost
                self.operation.init_sol = True
                print(time.time()-self.start)
            self.SaveData(time.time()-self.start)
            self.FindOptimalTransitionNode(iteration, True)
            self.AddTransitionNode(self.operation.active_mode.transition_nodes[-1])
            constrained_robots = self.env.get_active_task(self.operation.active_mode.label).robots
            print(f"{iteration}: {constrained_robots} found T{self.env.get_current_seq_index(self.operation.active_mode.label)}: Cost: ", self.operation.cost)
                              
    def PlannerInitialization(self):
        active_mode = self.operation.modes[0]
        start_state = State(self.env.start_pos, active_mode.label)
        start_node = Node(start_state, (self.operation.tree -1), self.operation)
        active_mode.batch_subtree = torch.cat((active_mode.batch_subtree, start_node.q_tensor.unsqueeze(0)), dim=0)
        active_mode.subtree[len(active_mode.subtree)] = start_node
        active_mode.node_idx_subtree = torch.cat((active_mode.node_idx_subtree, torch.tensor([start_node.idx], device='cuda')),dim=0)
        for _ in range(100): 
            end_state = self.GetGoalStateOfMode(active_mode.label, start_node)
            end_node = Node(end_state, self.operation.tree, self.operation)
            self.operation.tree += 1
            # active_mode.transition_nodes.append(first_end_node)
            end_node.transition = True
            active_mode.batch_subtree_b = torch.cat((active_mode.batch_subtree_b, end_node.q_tensor.unsqueeze(0)), dim=0)
            active_mode.subtree_b[len(active_mode.subtree_b)] = end_node
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
            n_nearest, _ = self.Nearest(q_rand, self.operation.active_mode.subtree, self.operation.active_mode.batch_subtree)    
            n_new = self.Steer(n_nearest, q_rand, self.operation.active_mode.label)

            if self.env.is_collision_free(n_new.state.q.state(), self.operation.active_mode.label) and self.env.is_edge_collision_free(n_nearest.state.q, n_new.state.q, self.operation.active_mode.label):
                N_near_indices, N_near_batch, n_near_costs, node_indices = self.Near(n_new, self.operation.active_mode.batch_subtree)
                if n_nearest.idx not in node_indices:
                    continue
                n_nearest_index = torch.where(node_indices == n_nearest.idx)[0].item() 
                batch_cost, batch_dist =  batch_config_torch(n_new.state.q, N_near_batch, "euclidean")
                self.FindParent(N_near_indices, n_nearest_index, n_new, n_nearest, batch_cost, batch_dist, n_near_costs)
                if self.operation.init_sol:
                    if self.Rewire(N_near_indices, n_new, batch_cost, batch_dist, n_near_costs, n_rand = q_rand.state(), n_nearest = n_nearest.state.q.state() ):
                        self.UpdateCost(n_new)
                if not self.operation.active_mode.connected:
                    n_nearest_b, idx =  self.Nearest(n_new.state.q, self.operation.active_mode.subtree_b, self.operation.active_mode.batch_subtree_b)
                    n_nearest_b_q = self.operation.active_mode.batch_subtree_b[idx] 
                    self.Connect(n_new, n_nearest_b, i, n_nearest_b_q) 
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




