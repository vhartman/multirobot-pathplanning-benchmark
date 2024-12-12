from configuration import *
from planning_env import *
from util import *
from rrt_base import *
import time as time

class RRTstar(BaseRRTstar):
    def __init__(self, env, config: ConfigManager):
        super().__init__(env, config)
     
    def UpdateCost(self, n:Node) -> None:
        stack = [n]
        while stack:
            current_node = stack.pop()
            children = current_node.children
            if children:
                for _, child in enumerate(children):
                    child.cost = current_node.cost + child.cost_to_parent
                    child.agent_cost = current_node.agent_cost + child.agent_cost_to_parent
                stack.extend(children)
   
    def ManageTransition(self, n_new: Node, iter: int) -> None:
        constrained_robots = self.env.get_active_task(self.operation.active_mode.label).robots
        indices = []
        radius = 0
        for r in constrained_robots:
            indices.extend(self.env.robot_idx[r])
            radius += self.config.goal_radius
        if self.env.get_active_task(self.operation.active_mode.label).goal.satisfies_constraints(n_new.state.q.state()[indices], self.config.goal_radius):
            self.operation.active_mode.transition_nodes.append(n_new)
            n_new.transition = True
            # Check if initial transition node of current mode is found
            if self.operation.active_mode.label == self.operation.modes[-1].label and not self.operation.init_sol:
                print(time.time()-self.start)
                print(f"{iter} {constrained_robots} found T{self.env.get_current_seq_index(self.operation.active_mode.label)}: Cost: ", n_new.cost)
                self.InitializationMode(self.operation.modes[-1])
                if self.env.terminal_mode != self.operation.modes[-1].label:
                    self.operation.modes.append(Mode(self.env.get_next_mode(n_new.state.q,self.operation.active_mode.label), self.env))
                elif self.operation.active_mode.label == self.env.terminal_mode:
                    self.operation.ptc_iter = iter
                    self.operation.ptc_cost = n_new.cost
                    self.operation.init_sol = True
                    print(time.time()-self.start)
                self.SaveData(time.time()-self.start)
                self.FindOptimalTransitionNode(iter, True)
                self.AddTransitionNode(n_new)
                return
            self.AddTransitionNode(n_new)
        self.FindOptimalTransitionNode(iter)
 
    def PlannerInitialization(self) -> None:
        active_mode = self.operation.modes[0]
        start_state = State(self.env.start_pos, active_mode.label)
        start_node = Node(start_state, (self.operation.tree -1), self.operation)
        active_mode.batch_subtree = torch.cat((active_mode.batch_subtree, start_node.q_tensor.unsqueeze(0)), dim=0)
        active_mode.subtree[len(active_mode.subtree)] = start_node
        active_mode.node_idx_subtree = torch.cat((active_mode.node_idx_subtree, torch.tensor([start_node.idx], device='cuda')),dim=0)

    def Plan(self) -> List[State]:
        i = 0
        self.PlannerInitialization()
        while True:
            # Mode selection
            self.operation.active_mode  = (np.random.choice(self.operation.modes, p= self.SetModePorbability()))
            # RRT* core
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
                # if self.operation.init_sol:
                if self.Rewire(N_near_indices, n_new, batch_cost, batch_dist, n_near_costs, n_rand = q_rand.state(), n_nearest = n_nearest.state.q.state() ):
                    self.UpdateCost(n_new)
                    # self.SaveData(time.time()-self.start, n_new = n_new.state.q.state(), N_near = N_near, 
                    #           r =self.r, n_rand = n_rand.state.q.state(), n_nearest = n_nearest.state.q.state()) 
                     
                self.ManageTransition(n_new, i)
            if self.operation.init_sol and i != self.operation.ptc_iter and (i- self.operation.ptc_iter)%self.config.ptc_max_iter == 0:
                diff = self.operation.ptc_cost - self.operation.cost
                self.operation.ptc_cost = self.operation.cost
                if diff < self.config.ptc_threshold:
                    break
            
            i += 1

        self.SaveData(time.time()-self.start)
        return self.operation.path    




