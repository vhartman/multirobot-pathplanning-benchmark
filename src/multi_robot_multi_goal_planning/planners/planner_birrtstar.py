from multi_robot_multi_goal_planning.planners.rrtstar_base import *

"""
This file contains the most important changes for the Bi-RRT* algorithm compared to the original RRT*. There are 2 versions possible: 
Version 1 is inspired by the paper 'Bi-RRT*: An Improved Bidirectional RRT* Path Planner for Robot inTwo-Dimensional Space' by B. Wang et al. 
and version 2 is inspired by the paper 'RRT-Connect: An Efficient Approach to Single-Query Path Planning' by J.J Kuffner et al. and 
by the paper 'RRT*-Connect: Faster, Asymptotically Optimal Motion Planning' by S. Klemm et al. 
"""

@njit
def compute_child_costs(parent_cost, cost_to_parents):
    return parent_cost + cost_to_parents


class BidirectionalRRTstar(BaseRRTstar):
    def __init__(self, 
                 env:BaseProblem, 
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
                 transition_nodes: int = 50, 
                 birrtstar_version: int = 2 
                ):
        super().__init__(env, ptc, general_goal_sampling, informed_sampling, informed_sampling_version, distance_metric,
                    p_goal, p_stay, p_uniform, shortcutting, mode_sampling, 
                    gaussian)
        self.transition_nodes = transition_nodes 
        self.birrtstar_version = birrtstar_version
    
    
    def UpdateCost(self, mode:Mode, n:Node, connection:bool = False) -> None:
        stack = [n]
        while stack:
            current_node = stack.pop()
            if connection:
                #add node to main tree and delte it from subtree b
                if mode.__eq__(current_node.state.mode):
                    if current_node.id in self.trees[mode].get_node_idx_subtree('B'):
                        self.trees[mode].add_node(current_node)
                        self.trees[mode].remove_node(current_node, 'B')

            children = current_node.children
            if children:
                for _, child in enumerate(children):
                    child.cost = current_node.cost + child.cost_to_parent                    
                stack.extend(children)

    def add_new_mode(self, q:Optional[Configuration]=None, mode:Mode=None, tree_instance: Optional[Union["SingleTree", "BidirectionalTree"]] = None) -> None: #TODO entry_configuration needs to be specified
        """Initializes a new mode"""
        if mode is None: 
            new_mode = self.env.make_start_mode()
            new_mode.prev_mode = None
        else:
            new_mode = self.env.get_next_mode(q, mode)
            new_mode.prev_mode = mode
        if new_mode in self.modes:
            return 
    
        self.modes.append(new_mode)
        self.add_tree(new_mode, tree_instance)
        self.InformedInitialization(new_mode)
        #Initialize transition nodes
        for i in range(self.transition_nodes):                 
            q = self.sample_transition_configuration(new_mode)
            if i > 0 and np.equal(q.state(), node.state.q.state()).all():
                break
            node = Node(State(q, new_mode), self.operation)
            node.cost_to_parent = 0.0
            self.mark_node_as_transition(new_mode, node)
            self.trees[new_mode].add_node(node, 'B')
            self.operation.costs = self.trees[new_mode].ensure_capacity(self.operation.costs, node.id) 
            node.cost = np.inf

    def ManageTransition(self, mode:Mode, n_new: Node, iter: int) -> None:
        #check if transition is reached
        if self.trees[mode].order == 1 and self.env.is_transition(n_new.state.q, mode):
            self.add_new_mode(n_new.state.q, mode, BidirectionalTree)
            self.convert_node_to_transition_node(mode, n_new)
        #check if termination is reached
        if self.trees[mode].order == 1 and self.env.done(n_new.state.q, mode):
            self.convert_node_to_transition_node(mode, n_new)
            if not self.operation.init_sol:
                # print(time.time()-self.start_time)
                self.operation.init_sol = True
        self.FindLBTransitionNode(iter)

    def UpdateTree(self, mode:Mode ,n: Node, n_parent:Node , cost_to_parent:NDArray) -> None: #TODO need to update it
        while True:
            n.cost = n_parent.cost +  cost_to_parent
            # dist_cpu= dists.clone().to(dtype=torch.float16).cpu()
            # n.agent_dists = n_parent.agent_dists + dist_cpu
            self.UpdateCost(mode, n, True)
            n_parent_inter = n.parent
            cost_to_parent_inter = n.cost_to_parent
            # agent_dists_to_parent_inter = n.agent_dists_to_parent

            n.parent = n_parent
            n.cost_to_parent = cost_to_parent
            # n.agent_dists_to_parent = dist_cpu
            n_parent.children.append(n) #Set child
            
            cost_to_parent = cost_to_parent_inter
            # dists = agent_dists_to_parent_inter
            n_parent = n
            n = n_parent_inter
            if not n:
                #need to have this transition node as the last one in the queue
                if n_parent.transition:
                    self.transition_node_ids[mode].remove(n_parent.id)
                    self.mark_node_as_transition(mode, n_parent)
                return 
            # print(n.state.mode)
            n.children.remove(n_parent)
  
    def Connect(self, mode:Mode, n_new:Node, iter:int) -> None:
        if not self.trees[mode].subtree_b:
            return
        n_nearest_b, dist, _ = self.Nearest(mode, n_new.state.q, 'B')

        if self.birrtstar_version == 1 or self.birrtstar_version == 2 and self.trees[mode].connected: #Based on paper Bi-RRT* by B. Wang 
            #TODO only check dist of active robots to connect (cost can be extremly high)? or the smartest way to just connect when possible?
            # relevant_dists = []
            # for r_idx, r in enumerate(self.env.robots):
            #     if r in constrained_robots:
            #         relevant_dists.append(dists[0][r_idx].item())
            # if np.max(relevant_dists) > self.step_size:
            #     return

            if dist > self.eta:
                return

            if not self.env.is_edge_collision_free(n_new.state.q, n_nearest_b.state.q, mode): #ORder rigth? TODO
                return
          
        elif self.birrtstar_version == 2 and not self.trees[mode].connected: #Based on paper RRT-Connect by JJ. Kuffner/ RRT*-Connect by S.Klemm
            n_nearest_b = self.Extend(mode, n_nearest_b, n_new, dist)
            if not n_nearest_b:
                return
           
        cost =  batch_config_cost([n_new.state],  [n_nearest_b.state], metric = self.env.cost_metric, reduction=self.env.cost_reduction)
        if self.trees[mode].order == -1:
            #switch such that subtree is beginning from start and subtree_b from goal
            self.trees[mode].swap()
            if not self.env.is_edge_collision_free(n_nearest_b.state.q, n_new.state.q, mode):
                return
            self.UpdateTree(mode, n_new, n_nearest_b, cost[0]) 
            
        else:
            if not self.env.is_edge_collision_free(n_new.state.q, n_nearest_b.state.q, mode):
                return
            self.UpdateTree(mode, n_nearest_b, n_new, cost[0])

        transition_node = self.get_transition_node(mode, self.transition_node_ids[mode][-1]) 
        #initial solution has been found for the frist time in this mode
        if not self.trees[mode].connected: 
            self.trees[mode].connected = True
            #check if terminal mode was already reached
            if not self.env.is_terminal_mode(mode):
                self.add_new_mode(transition_node.state.q, mode, BidirectionalTree)
            else:
                self.operation.init_sol = True
                # print(time.time()-self.start_time)
        #need to do that after the next mode was initialized
        self.convert_node_to_transition_node(mode, transition_node)


    def Extend(self, mode:Mode, n_nearest_b:Node, n_new:Node, dist )-> Optional[Node]:
        q = n_new.state.q
        #RRT not RRT*
        i = 1
        while True:
            state_new = self.Steer(mode, n_nearest_b, q, dist, i)
            if not state_new or np.equal(state_new.q.state(), q.state()).all(): # Reached
                # self.SaveData(mode, time.time()-self.start_time, n_nearest = n_nearest_b.state.q.state(), n_rand = q.state(), n_new = n_new.state.q.state())
                return n_nearest_b
            # self.SaveData(mode, time.time()-self.start_time, n_nearest = n_nearest_b.state.q.state(), n_rand = q.state(), n_new = state_new.q.state())
            if self.env.is_collision_free(state_new.q, mode) and self.env.is_edge_collision_free(n_nearest_b.state.q, state_new.q, mode):
                # Add n_new to tree
        
                n_new = Node(state_new,self.operation)
               
                cost =  batch_config_cost([n_new.state], [n_nearest_b.state], metric = self.env.cost_metric, reduction=self.env.cost_reduction)
                c_min = n_nearest_b.cost + cost

                n_new.parent = n_nearest_b
                n_new.cost_to_parent = cost
                n_nearest_b.children.append(n_new) #Set child
                self.trees[mode].add_node(n_new, 'B') 
                self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, n_new.id) 
                n_new.cost = c_min
                n_nearest_b = n_new
                i +=1
            else:
                return 

    def PlannerInitialization(self) -> None:
        # Initilaize first Mode
        self.set_gamma_rrtstar()
        self.add_new_mode(tree_instance=BidirectionalTree)
        mode = self.modes[-1]
        # Create start node
        start_state = State(self.env.start_pos, mode)
        start_node = Node(start_state, self.operation)
        self.trees[mode].add_node(start_node)
        start_node.cost = 0.0
        start_node.cost_to_parent = 0.0

    def Plan(self) -> dict:
        i = 0
        self.PlannerInitialization()
        while True:
            i += 1
            # Mode selection
            active_mode  = self.RandomMode()
            # Bi RRT* core
            q_rand = self.SampleNodeManifold(active_mode)
            n_nearest, dist, set_dists = self.Nearest(active_mode, q_rand)        
            state_new = self.Steer(active_mode, n_nearest, q_rand, dist)
            # q_rand == n_nearest
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
                batch_cost = batch_config_cost(n_new.state.q, N_near_batch, metric = self.env.cost_metric, reduction=self.env.cost_reduction)
                self.FindParent(active_mode, node_indices, n_new, n_nearest, batch_cost, n_near_costs)
                if self.Rewire(active_mode, node_indices, n_new, batch_cost, n_near_costs):
                    self.UpdateCost(active_mode,n_new)
                self.Connect(active_mode, n_new, i)
                self.ManageTransition(active_mode, n_new, i)
            self.trees[active_mode].swap()

            if self.ptc.should_terminate(i, time.time() - self.start_time):
                break
        self.costs.append(self.operation.cost)
        self.times.append(time.time() - self.start_time)
        self.all_paths.append(self.operation.path)
        # self.SaveData(active_mode, time.time()-self.start_time)
        info = {"costs": self.costs, "times": self.times, "paths": self.all_paths}
        return self.operation.path, info      




