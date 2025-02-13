from multi_robot_multi_goal_planning.planners.rrtstar_base import *

"""This file contains the original RRT* based on the paper 'Sampling-based Algorithms for Optimal Motion Planning' by E. Frazolli et al."""

class RRTstar(BaseRRTstar):
    def __init__(self, 
                 env: BaseProblem,
                 ptc: PlannerTerminationCondition,  
                 general_goal_sampling: bool = False, 
                 informed_sampling: bool = False, 
                 informed_sampling_version: int = 0, 
                 distance_metric: str = 'max_euclidean',
                 p_goal: float = 0.9, 
                 p_stay: float = 0.3,
                 p_uniform: float = 0.8, 
                 shortcutting: bool = False, 
                 mode_sampling: Optional[Union[int, float]] = None, 
                 gaussian: bool = False, 
                ):
        super().__init__(env, ptc, general_goal_sampling, informed_sampling, informed_sampling_version, distance_metric,
                    p_goal, p_stay, p_uniform, shortcutting, mode_sampling, 
                    gaussian)
     
    def UpdateCost(self, n:Node) -> None:
        stack = [n]
        while stack:
            current_node = stack.pop()
            children = current_node.children
            if children:
                for _, child in enumerate(children):
                    child.cost = current_node.cost + child.cost_to_parent
                    # child.agent_dists = current_node.agent_dists + child.agent_dists_to_parent
                stack.extend(children)
   
    def ManageTransition(self, mode:Mode, n_new: Node, iter: int) -> None:
        #check if transition is reached
        if self.env.is_transition(n_new.state.q, mode):
            self.add_new_mode(n_new.state.q, mode, SingleTree)
            self.convert_node_to_transition_node(mode, n_new)
        #check if termination is reached
        if self.env.done(n_new.state.q, mode):
            self.convert_node_to_transition_node(mode, n_new)
            if not self.operation.init_sol:
                self.operation.init_sol = True
        self.FindLBTransitionNode(iter)
 
    def PlannerInitialization(self) -> None:
        self.set_gamma_rrtstar()
        # Initilaize first Mode
        self.add_new_mode(tree_instance=SingleTree)
        active_mode = self.modes[-1]
        # Create start node
        start_state = State(self.env.start_pos, active_mode)
        start_node = Node(start_state, self.operation)
        self.trees[active_mode].add_node(start_node)
        start_node.cost = 0.0
        start_node.cost_to_parent = 0.0
    
    def Plan(self) -> List[State]:
        i = 0
        self.PlannerInitialization()
        while True:
            i += 1
            # Mode selectiom       
            active_mode  = self.RandomMode()

            # RRT* core
            q_rand = self.SampleNodeManifold(active_mode)
            n_nearest, dist, set_dists = self.Nearest(active_mode, q_rand)    
            state_new = self.Steer(active_mode, n_nearest, q_rand, dist)
            if not state_new:
                continue
            if self.env.is_collision_free(state_new.q, active_mode) and self.env.is_edge_collision_free(n_nearest.state.q, state_new.q, active_mode):
                n_new = Node(state_new, self.operation)
                if np.equal(n_new.state.q.state(), q_rand.state()).all():
                    N_near_batch, n_near_costs, node_indices = self.Near(active_mode, n_new, set_dists)
                else:
                    N_near_batch, n_near_costs, node_indices = self.Near(active_mode, n_new)
                if n_nearest.id not in node_indices:
                    continue
                batch_cost = batch_config_cost(n_new.state.q, N_near_batch, self.env.cost_metric, self.env.cost_reduction)
                self.FindParent(active_mode, node_indices, n_new, n_nearest, batch_cost, n_near_costs)
                if self.Rewire(active_mode, node_indices, n_new, batch_cost, n_near_costs):
                    self.UpdateCost(n_new) 
                self.ManageTransition(active_mode, n_new, i)

            if self.ptc.should_terminate(i, time.time() - self.start_time):
                break
            if self.operation.init_sol:
                break
        self.costs.append(self.operation.cost)
        self.times.append(time.time() - self.start_time)
        self.all_paths.append(self.operation.path)
        # self.SaveData(active_mode, time.time()-self.start_time)
        info = {"costs": self.costs, "times": self.times, "paths": self.all_paths}
        return self.operation.path, info    




