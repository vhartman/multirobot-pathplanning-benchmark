from multi_robot_multi_goal_planning.planners.rrtstar_base import *

"""This file contains the original RRT* based on the paper 'Sampling-based Algorithms for Optimal Motion Planning' by E. Frazolli et al."""

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
                stack.extend(children)
   
    def ManageTransition(self, mode:Mode, n_new: Node, iter: int) -> None:
        #check if transition is reached
        if self.env.is_transition(n_new.state.q, mode):
            if not self.operation.init_sol and mode.__eq__(self.modes[-1]):
                    self.add_new_mode(n_new.state.q, mode, SingleTree)
                    self.ModeInitialization(self.modes[-1])
            self.convert_node_to_transition_node(mode, n_new)
        #check if termination is reached
        if self.env.done(n_new.state.q, mode):
            self.convert_node_to_transition_node(mode, n_new)
            if not self.operation.init_sol:
                print(time.time()-self.start)
                self.operation.init_sol = True
        self.FindLBTransitionNode(iter)
 
    def PlannerInitialization(self) -> None:
        # Initilaize first Mode
        self.add_new_mode(tree_instance=SingleTree)
        active_mode = self.modes[-1]
        self.ModeInitialization(active_mode)
        # Create start node
        start_state = State(self.env.start_pos, active_mode)
        start_node = Node(start_state, self.operation)
        self.trees[active_mode].add_node(start_node)
        start_node.cost = 0
        start_node.cost_to_parent = torch.tensor(0, device=device, dtype=torch.float32)
    
    def Plan(self) -> List[State]:
        i = 0
        self.PlannerInitialization()
        while True:
            i += 1
            # Mode selection
            active_mode  = self.RandomMode(Mode.id_counter)
            
            # RRT* core
            q_rand = self.SampleNodeManifold(active_mode)
            
            # print(i, active_mode,  [s for s in q_rand.state()])
            n_nearest, dist = self.Nearest(active_mode, q_rand)   
            state_new = self.Steer(active_mode, n_nearest, q_rand, dist)
            # print([float(s) for s in state_new.q.state()]) 
             # q_rand == n_nearest
            if not state_new:
                continue
            if self.env.is_collision_free(state_new.q, active_mode) and self.env.is_edge_collision_free(n_nearest.state.q, state_new.q, active_mode):
                n_new = Node(state_new, self.operation)
                # print(i, [float(s) for s in n_new.state.q.state()])
                N_near_batch, n_near_costs, node_indices = self.Near(active_mode, n_new)

                if n_nearest.id not in node_indices:
                    continue
                batch_cost =  batch_cost_torch(n_new.q_tensor, n_new.state.q, N_near_batch, metric = self.config.cost_type).clone()
                self.FindParent(active_mode, node_indices, n_new, n_nearest, batch_cost, n_near_costs)
                if self.Rewire(active_mode, node_indices, n_new, batch_cost, n_near_costs):
                    self.UpdateCost(n_new)
                    
                self.ManageTransition(active_mode, n_new, i)

            if self.PTC(i):
                self.SaveFinalData()
                "PTC applied"
                print(i)
                break

        self.SaveData(active_mode, time.time()-self.start)
        return self.operation.path    




