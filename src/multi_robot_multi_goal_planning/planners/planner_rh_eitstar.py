import numpy as np

from typing import (
    List,
    Dict,
    Tuple,
    Optional,
    Union, 
    Any, 
    Set
)
import time
import math
from itertools import chain
import heapq
import random
from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
    Mode
)
from multi_robot_multi_goal_planning.problems.configuration import (
    batch_config_dist,
    Configuration
)
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
)
from multi_robot_multi_goal_planning.planners.itstar_base import (
    BaseITstar,
    BaseGraph,
    BaseOperation, 
    BaseNode,
    BaseTree
    )
from functools import cache
from multi_robot_multi_goal_planning.planners.planner_aitstar import EdgeQueue
from multi_robot_multi_goal_planning.planners.planner_eitstar import (
    EITstar,
    Node,
    Graph
)
from numba import njit
from multi_robot_multi_goal_planning.problems.util import path_cost, interpolate_path
# taken from https://github.com/marleyshan21/Batch-informed-trees/blob/master/python/BIT_Star.py
# needed adaption to work.

def get_index_of_last_mode_appereance(path_modes:List[List[int]], m:List[int]) -> int:
    """
    Retrieves mode task ID for the active task in a given path by selecting the last occurrence that matches the specified task id.

    Args:
        path_modes (NDArray): Sequence of mode task IDs along the path.
        task_id (List[int]): Task IDs to search for in the path.
        r_idx (int): Corresponding idx of robot.

    Returns:
        NDArray: Mode task ID associated with the last occurrence of the specified active task for the desired robot.
    """

    last_index = 0 
    for i in range(len(path_modes)):
        if path_modes[i] == m:  
            last_index = i 
    return last_index



class RHEITstar(EITstar):
    
    def __init__(
        self,
        env: BaseProblem,
        ptc: PlannerTerminationCondition,
        mode_sampling_type: str = "greedy",
        distance_metric: str = "euclidean",
        try_sampling_around_path: bool = True,
        try_informed_sampling: bool = True,
        init_uniform_batch_size: int = 100,
        init_transition_batch_size:int = 100,
        uniform_batch_size: int = 200,
        uniform_transition_batch_size: int = 500,
        informed_batch_size: int = 500,
        informed_transition_batch_size: int = 500,
        path_batch_size: int = 500,
        locally_informed_sampling: bool = True,
        try_informed_transitions: bool = True,
        try_shortcutting: bool = True,
        try_direct_informed_sampling: bool = True,
        inlcude_lb_in_informed_sampling:bool = True,
        remove_based_on_modes:bool = False,
        with_tree_visualization:bool = False,
        use_max_distance_metric_effort:bool = False,
        ):
        super().__init__(
            env = env, ptc=ptc, mode_sampling_type = mode_sampling_type, distance_metric = distance_metric, 
            try_sampling_around_path = try_sampling_around_path,try_informed_sampling = try_informed_sampling, 
            init_uniform_batch_size=init_uniform_batch_size, init_transition_batch_size=init_transition_batch_size,
            uniform_batch_size = uniform_batch_size, uniform_transition_batch_size = uniform_transition_batch_size, informed_batch_size = informed_batch_size, 
            informed_transition_batch_size = informed_transition_batch_size, path_batch_size = path_batch_size, locally_informed_sampling = locally_informed_sampling, 
            try_informed_transitions = try_informed_transitions, try_shortcutting = try_shortcutting, try_direct_informed_sampling = try_direct_informed_sampling, 
            inlcude_lb_in_informed_sampling = inlcude_lb_in_informed_sampling,remove_based_on_modes = remove_based_on_modes, with_tree_visualization = with_tree_visualization)

        self.sparse_number_of_points = 1
        self.use_max_distance_metric_effort = use_max_distance_metric_effort
        self.reverse_tree_set = set()
        self.reduce_neighbors = False
        self.sparesly_checked_edges = {}
        self.check = set()
        self.horizon = 2 #the next task and one after
        self.step = 0
        self.global_path = None
        self.global_path_nodes = None
        self.global_cost = None
        self.global_reached_modes = []
        self.global_reached_modes_task_ids = None
        self.horizon_goal_mode = None
        self.actual_path = []
        self.current_horizon_path = None
        self.actual_path_cost = 0
    
    def sample_valid_uniform_transitions(self, transistion_batch_size, cost):
        transitions = 0

        if len(self.g.goal_nodes) > 0:
            focal_points = np.array(
                [self.g.root.state.q.state(), self.g.goal_nodes[0].state.q.state()],
                dtype=np.float64,
            )
        # while len(transitions) < transistion_batch_size:
        failed_attemps = 0
        while transitions < transistion_batch_size:
            if failed_attemps > 2* transistion_batch_size:
                break 
            if len(self.g.goal_node_ids) == 0:
                mode_sampling_type = "greedy"
            else:
                mode_sampling_type = "uniform_reached"
            # sample mode
            mode = self.sample_mode(self.reached_modes, mode_sampling_type, None)

            # sample transition at the end of this mode
            possible_next_task_combinations = self.env.get_valid_next_task_combinations(
                mode
            )
            # print(mode, possible_next_task_combinations)

            if len(possible_next_task_combinations) > 0:
                ind = random.randint(0, len(possible_next_task_combinations) - 1)
                active_task = self.env.get_active_task(
                    mode, possible_next_task_combinations[ind]
                )
            else:
                active_task = self.env.get_active_task(mode, None)

            goals_to_sample = active_task.robots

            goal_sample = active_task.goal.sample(mode)

            # if mode.task_ids == [3, 8]:
            #     print(active_task.name)

            q = []
            for i in range(len(self.env.robots)):
                r = self.env.robots[i]
                if r in goals_to_sample:
                    offset = 0
                    for _, task_robot in enumerate(active_task.robots):
                        if task_robot == r:
                            q.append(
                                goal_sample[
                                    offset : offset + self.env.robot_dims[task_robot]
                                ]
                            )
                            break
                        offset += self.env.robot_dims[task_robot]
                else:  # uniform sample
                    lims = self.env.limits[:, self.env.robot_idx[r]]
                    if lims[0, 0] < lims[1, 0]:
                        qr = (
                            np.random.rand(self.env.robot_dims[r])
                            * (lims[1, :] - lims[0, :])
                            + lims[0, :]
                        )
                    else:
                        qr = np.random.rand(self.env.robot_dims[r]) * 6 - 3

                    q.append(qr)

            q = self.conf_type.from_list(q)

            if cost is not None:
                if sum(self.env.batch_config_cost(q, focal_points)) > cost:
                    continue
            horizon_goal = False        
            if self.env.is_collision_free(q, mode):
                if self.env.is_terminal_mode(mode): # need to treat horzion goal as actual goal
                    next_mode = None
                    horizon_goal = True
                else:
                    next_mode = self.env.get_next_mode(q, mode)
                    if mode == self.horizon_goal_mode:
                        horizon_goal = True


                
                self.g.add_transition_nodes([(q, mode, next_mode)])
                if len(list(chain.from_iterable(self.g.transition_node_ids.values()))) > transitions:
                    transitions +=1
                else:
                    failed_attemps +=1
                # print(mode, mode.next_modes)

                if next_mode not in self.reached_modes and next_mode is not None:
                    if not horizon_goal:
                        self.reached_modes.append(next_mode)
                        self.global_reached_modes.append(next_mode)

                    # for dependency graphs
                    elif next_mode not in self.global_reached_modes:
                        self.global_reached_modes.append(next_mode)
            
        print(f"Adding {transitions} transitions")
        return
    
    def add_sample_horizon_batch(self):
        # add new batch of nodes
        while True:
 
            effective_uniform_batch_size = (
                self.uniform_batch_size if self.current_best_cost is not None or not self.first_search
                else self.init_uniform_batch_size
            )
            effective_uniform_transition_batch_size = (
                self.uniform_transition_batch_size
                if self.current_best_cost is not None or not self.first_search
                else self.init_transition_batch_size
            )

            # nodes_per_state = []
            # for m in reached_modes:
            #     num_nodes = 0
            #     for n in new_states:
            #         if n.mode == m:
            #             num_nodes += 1

            #     nodes_per_state.append(num_nodes)

            # plt.figure("Uniform states")
            # plt.bar([str(mode) for mode in reached_modes], nodes_per_state)

            # if self.env.terminal_mode not in reached_modes:   
            print("--------------------")
            print("Sampling transitions")
            self.sample_valid_uniform_transitions(
                transistion_batch_size=effective_uniform_transition_batch_size,
                cost=self.current_best_cost,
            )
            # new_transitions = self.sample_valid_uniform_transitions(
            #     transistion_batch_size=effective_uniform_transition_batch_size,
            #     cost=self.current_best_cost,
            # )
            if self.horizon_goal_mode in self.g.transition_node_ids:
                self.g.goal_node_ids = self.g.transition_node_ids[self.horizon_goal_mode].copy()
                self.g.goal_nodes = [self.g.nodes[id] for id in self.g.goal_node_ids]
            if len(self.g.goal_nodes) == 0:
                continue


            # self.g.add_transition_nodes(new_transitions)
            # print(f"Adding {len(new_transitions)} transitions")
            
            print("Sampling uniform")
            new_states, required_attempts_this_batch = self.sample_valid_uniform_batch(
                    batch_size=effective_uniform_batch_size, cost=self.current_best_cost
                )
            self.g.add_states(new_states)
            print(f"Adding {len(new_states)} new states")

            self.approximate_space_extent = (
                np.prod(np.diff(self.env.limits, axis=0))
                * len(new_states)
                / required_attempts_this_batch
            )

            # print(reached_modes)

            

        # g.compute_lb_cost_to_go(self.env.batch_config_cost)
        # g.compute_lower_bound_from_start(self.env.batch_config_cost)

            if self.current_best_cost is not None and (
                self.try_informed_sampling or self.try_informed_transitions
            ):
                # adjust this
                interpolated_path = interpolate_path(self.current_horizon_path)
                # interpolated_path = current_best_path

                if self.try_informed_sampling:
                    print("Generating informed samples")
                    new_informed_states = self.informed.generate_samples(
                                            self.reached_modes,
                                            self.informed_batch_size,
                                            interpolated_path,
                                            try_direct_sampling=self.try_direct_informed_sampling,
                                            g=self.g
                                        )
                    
                    self.g.add_states(new_informed_states)

                    print(f"Adding {len(new_informed_states)} informed samples")

                if self.try_informed_transitions:
                    print("Generating informed transitions")
                    new_informed_transitions = self.informed.generate_transitions(
                                                self.reached_modes,
                                                self.informed_transition_batch_size,
                                                interpolated_path,
                                                g=self.g
                                            )

                    
                    self.g.add_transition_nodes(new_informed_transitions)
                    print(f"Adding {len(new_informed_transitions)} informed transitions")
                    if self.horizon_goal_mode in self.g.transition_node_ids:
                        self.g.goal_node_ids = self.g.transition_node_ids[self.horizon_goal_mode].copy()
                        self.g.goal_nodes = [self.g.nodes[id] for id in self.g.goal_node_ids]

                    # g.compute_lb_cost_to_go(self.env.batch_config_cost)
                    # g.compute_lower_bound_from_start(self.env.batch_config_cost)

            if self.try_sampling_around_path and self.current_best_path is not None:
                print("Sampling around path")
                path_samples, path_transitions = self.sample_around_path(
                    self.current_best_path
                )

                self.g.add_states(path_samples)
                print(f"Adding {len(path_samples)} path samples")

                self.g.add_transition_nodes(path_transitions)
                print(f"Adding {len(path_transitions)} path transitions")
            
            if len(self.g.goal_nodes) != 0:
                break
        # for mode in self.reached_modes:
        #     q_samples = [self.g.nodes[id].state.q.state() for id in self.g.node_ids[mode]]
        #     modes = [self.g.nodes[id].state.mode.task_ids for id in self.g.node_ids[mode]]
        #     data = {
        #         "q_samples": q_samples,
        #         "modes": modes,
        #         "path": self.current_best_path
        #     }
        #     save_data(data)
        #     print()
        

        # self.g.compute_lb_reverse_cost_to_come(self.env.batch_config_cost)
        # self.g.compute_lb_cost_to_come(self.env.batch_config_cost)

    def sample_manifold(self) -> None:
        print("====================")
        while True:
            self.g.initialize_cache()
            if self.current_best_path is not None:
                # prune
                # for mode in self.reached_modes:
                #     q_samples = [self.g.nodes[id].state.q.state() for id in self.g.node_ids[mode]]
                #     modes = [self.g.nodes[id].state.mode.task_ids for id in self.g.node_ids[mode]]
                #     data = {
                #         "q_samples": q_samples,
                #         "modes": modes,
                #         "path": self.current_best_path
                #     }
                #     save_data(data)
                #     print()
                # for mode in self.reached_modes:
                #     q_samples = [self.g.nodes[id].state.q.state() for id in self.g.transition_node_ids[mode]]
                #     modes = [self.g.nodes[id].state.mode.task_ids for id in self.g.transition_node_ids[mode]]
                #     data = {
                #         "q_samples": q_samples,
                #         "modes": modes,
                #         "path": self.current_best_path
                #     }
                #     save_data(data)
                #     print()

                self.remove_nodes_in_graph()

                # for mode in self.reached_modes:
                #     q_samples = [self.g.nodes[id].state.q.state() for id in self.g.node_ids[mode]]
                #     modes = [self.g.nodes[id].state.mode.task_ids for id in self.g.node_ids[mode]]
                #     data = {
                #         "q_samples": q_samples,
                #         "modes": modes,
                #         "path": self.current_best_path
                #     }
                #     save_data(data)
                #     print()
                # for mode in self.reached_modes:
                #     q_samples = [self.g.nodes[id].state.q.state() for id in self.g.transition_node_ids[mode]]
                #     modes = [self.g.nodes[id].state.mode.task_ids for id in self.g.transition_node_ids[mode]]
                #     data = {
                #         "q_samples": q_samples,
                #         "modes": modes,
                #         "path": self.current_best_path
                #     }
                #     save_data(data)
                #     print()


            print(f"Samples: {self.cnt}; {self.ptc}")

            samples_in_graph_before = self.g.get_num_samples()
            if self.first_search:
                self.global_reached_modes = [self.g.root.state.mode]
                self.add_sample_batch()
            else:
                self.add_sample_horizon_batch()#TODO need to chagne the global goals to local goals
            # self.global_reached_modes = self.reached_modes
            self.global_reached_modes_task_ids = [mode.task_ids for mode in self.global_reached_modes]
            self.g.initialize_cache()

            samples_in_graph_after = self.g.get_num_samples()
            self.cnt += samples_in_graph_after - samples_in_graph_before

            # search over nodes:
            # 1. search from goal state with sparse check
            if self.first_search:
                reached_terminal_mode = False
                for m in self.reached_modes:
                    if self.env.is_terminal_mode(m):
                        reached_terminal_mode = True
                        break
                if reached_terminal_mode:
                    print("====================")
                    break
            else:
                break

            # if reached_terminal_mode:
            #     print("====================")
            #     break
    
    def initialize_search(self):
        #only long horizon path
        if self.current_best_cost is not None:
            print()
            print("Shortcutting before new batch")
            self.current_best_path_nodes = self.generate_path(True)
            self.process_valid_path(self.current_best_path_nodes, False, True, True)
            self.update_removal_conditions() 
        self.adjust_to_horizon()
        self.sample_manifold()
        self.initialize_lb()
        self.initialze_forward_search()
        self.initialize_reverse_search()
        self.dynamic_reverse_search_update = False
        self.first_search = False

    def adjust_to_horizon(self):
        
        if self.current_best_cost is None:
            return
        # adjust reached modes to horizon
        self.reached_modes = []
        mode = self.env.get_start_mode()
        self.reached_modes.append(mode)
        while len(self.reached_modes) < self.horizon:
            possible_next_task_combinations = self.env.get_valid_next_task_combinations(mode)
            if not possible_next_task_combinations:
                continue
            next_ids = random.choice(possible_next_task_combinations)
            index = self.global_reached_modes_task_ids.index(next_ids)
            mode = self.global_reached_modes[index]
            self.reached_modes.append(mode)
            if self.env.is_terminal_mode(mode):
                break
        self.horizon_goal_mode = self.reached_modes[-1]
        self.adjust_global_path()
        
        # adjust goal_nodes to horizon
    
    def adjust_global_path(self):
        # the first time
        if self.global_cost is None and self.current_best_cost is not None:
            self.global_path = self.current_best_path
            self.global_path_nodes = self.current_best_path_nodes
            self.global_cost = self.current_best_cost
            self.global_path_mode = [s.mode.task_ids for s in self.global_path]
            index = get_index_of_last_mode_appereance(self.global_path_mode, self.horizon_goal_mode.task_ids)
            self.current_horizon_path = self.global_path[0:index+1]
        else:
            self.current_horizon_path = self.current_best_path
    
    def update_results_tracking(self, cost, path):
        # if self.current_best_cost is None or cost < self.current_best_co
        self.costs.append(cost)
        self.times.append(time.time() - self.start_time)
        self.all_paths.append(path)

    def PlannerInitializationNewHorizon(self) -> None:
        self.g.root = self.current_best_path_nodes[1]
        self.actual_path_cost += self.current_best_path_nodes[1].cost
        self.step += 1
        if self.step%10 == 0:
            self.horizon +=1
        #initilaize graph
        # self.g = self._create_graph(State(q0, m0))
        # # initialize all queues (edge-queues)    
        # self.g.add_vertex_to_tree(self.g.root)
        self.initialize_search()
        
        
    def Plan(
        self,optimize:bool = True
    ) -> Tuple[List[State], Dict[str, List[Union[float, float, List[State]]]]]:
        self.collision_cache = set()
        self.PlannerInitialization()
        num_iter = 0
        n1 = None
        while True:
            num_iter += 1
            self.reverse_search()
            if num_iter % 100000 == 0:
                print("Forward Queue: ", len(self.g.cost_bound_queue))
            if len(self.g.cost_bound_queue) < 1:
                print("------------------------",n1.state.mode.task_ids)
                self.PlannerInitializationNewHorizon()
                continue
            
            key, _ = self.g.cost_bound_queue.peek_first_element()
            if self.current_best_cost is None or key < self.current_best_cost:
                edge_cost, (n0, n1), edge_effort = self.g.get_best_forward_edge()
                if n0.id in n1.blacklist:
                    #needed because of rewiring
                    continue
                assert n0.id not in n1.blacklist, (
                    "askdflö"
                )
                is_transition = False
                if n1.is_transition and not n1.is_reverse_transition and n1.transition is not None:
                    is_transition = True
                
                if n1 in BaseTree.all_vertices and not self.dynamic_reverse_search_update:
                    continue

                # if np.isinf(n1.lb_cost_to_go):
                #     assert not [self.g.cost_bound_queue.key(item) for item in self.g.cost_bound_queue.current_entries if not np.isinf(self.g.cost_bound_queue.key(item))], (
                #     "Forward queue contains non-infinite keys!")
                # already found best possible parent
           
                if n1.forward.parent == n0:  # if its already the parent
                    if is_transition:
                        self.expand_node_forward(n1.transition)
                    else:
                        self.expand_node_forward(n1)
                elif (
                    n0.cost + edge_cost < n1.cost
                ):  # parent can improve the cost
            
                    assert n0.id not in n1.forward.children, (
                        "Potential parent is already a child (forward)"
                    )
                    # check edge sparsely now. if it is not valid, blacklist it, and continue with the next edge
                    if n0.id not in n1.whitelist:
                        collision_free = False
                        if n0.id not in n1.blacklist:
                            N = max(2, int(edge_effort))
                            collision_free = self.env.is_edge_collision_free(
                                n0.state.q,
                                n1.state.q,
                                n0.state.mode,
                                self.env.collision_resolution,
                                N=N,
                            )
                            
                        if not collision_free:
                            self.g.update_edge_collision_cache(n0, n1, collision_free)
                            self.g.remove_forward_queue(edge_cost, n0, n1, edge_effort)
                            self.manage_edge_in_collision(n0, n1)
                            continue
                    self.g.update_connectivity(n0, n1, edge_cost, n0.cost + edge_cost,"forward", is_transition)
                    if self.current_best_cost is not None: 
                        assert (n1.cost + n1.lb_cost_to_go <= self.current_best_cost), (
                                "hjklö"
                            )
                    if is_transition:
                        print(n1.state.mode, n1.transition.state.mode)
                        self.expand_node_forward(n1.transition)
                    else:
                        self.expand_node_forward(n1)
                    update = False
                    if n1 in self.g.goal_nodes:
                        update = True
                    if self.dynamic_reverse_search_update or n1 in self.g.goal_nodes:
                        path = self.generate_path()
                        if len(path) > 0:
                            if self.with_tree_visualization and (BaseTree.all_vertices or self.reverse_tree_set):
                                self.save_tree_data((BaseTree.all_vertices, self.reverse_tree_set))
                            self.process_valid_path(path, with_shortcutting= update, with_queue_update=update)
                            self.update_inflation_factor()
                            if self.with_tree_visualization and (BaseTree.all_vertices or self.reverse_tree_set):
                                self.save_tree_data((BaseTree.all_vertices, self.reverse_tree_set))
                            # current_state = self.env.execute_path(self.current_best_path)
                            self.PlannerInitializationNewHorizon()
                                
                            

                            
      
            else:
                self.PlannerInitializationNewHorizon()
                
            if not optimize and self.current_best_cost is not None:
                break

            if self.ptc.should_terminate(self.cnt, time.time() - self.start_time):
                break
        if self.costs != []:
            self.update_results_tracking(self.costs[-1], self.current_best_path)
        info = {"costs": self.costs, "times": self.times, "paths": self.all_paths}
        return self.current_best_path, info


