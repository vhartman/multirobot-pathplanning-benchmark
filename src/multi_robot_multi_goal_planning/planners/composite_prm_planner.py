import random
import time
from typing import (
    Any,
    Dict,
    List,
    Set,
    Tuple,
)
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from itertools import chain

from multi_robot_multi_goal_planning.problems.configuration import (
    batch_config_dist,
)
from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseProblem,
    Mode,
    State,
)
from multi_robot_multi_goal_planning.problems.util import interpolate_path, path_cost

from multi_robot_multi_goal_planning.planners import shortcutting
from .baseplanner import BasePlanner
from .sampling_informed import InformedSampling
from .mode_validation import ModeValidation
from .termination_conditions import (
    PlannerTerminationCondition,
)
from .prm.prm_graph import MultimodalGraph


@dataclass
class CompositePRMConfig:
    mode_sampling_type: str = "uniform_reached"
    distance_metric: str = "max_euclidean"
    use_k_nearest: bool = False
    try_informed_sampling: bool = True
    uniform_batch_size: int = 200
    uniform_transition_batch_size: int = 500
    informed_batch_size: int = 500
    informed_transition_batch_size: int = 500
    locally_informed_sampling: bool = True
    try_informed_transitions: bool = True
    try_shortcutting: bool = True
    shortcutting_mode: str = "round_robin"
    shortcutting_iters: int = 250
    shortcutting_interpolation_resolution: float = 0.1
    try_direct_informed_sampling: bool = True
    inlcude_lb_in_informed_sampling: bool = False
    init_mode_sampling_type: str = "greedy"
    frontier_mode_sampling_probability: float = 0.98
    init_uniform_batch_size: int = 150
    init_transition_batch_size: int = 90
    with_mode_validation: bool = False
    with_noise: bool = False


class CompositePRM(BasePlanner):
    def __init__(self, env: BaseProblem, config: CompositePRMConfig | None = None):
        self.env = env
        self.config = config if config is not None else CompositePRMConfig()
        self.mode_validation = ModeValidation(
            self.env,
            self.config.with_mode_validation,
            with_noise=self.config.with_noise,
        )
        self.init_next_modes, self.init_next_ids = {}, {}
        self.found_init_mode_sequence = False
        self.first_search = True
        self.dummy_start_mode = False
        self.sorted_reached_modes = []

    def _sample_mode(
        self,
        reached_modes: List[Mode],
        graph: MultimodalGraph,
        mode_sampling_type: str = "uniform_reached",
        found_solution: bool = False,
    ) -> Mode:
        """
        Sample a mode from the previously reached modes.
        """

        if mode_sampling_type == "uniform_reached":
            return random.choice(reached_modes)
        elif mode_sampling_type == "frontier":
            if len(reached_modes) == 1:
                return reached_modes[0]

            total_nodes = graph.get_num_samples()
            p_frontier = self.config.frontier_mode_sampling_probability
            p_remaining = 1 - p_frontier

            frontier_modes = []
            remaining_modes = []
            sample_counts = {}
            inv_prob = []

            for m in reached_modes:
                sample_count = graph.get_num_samples_in_mode(m)
                sample_counts[m] = sample_count
                if not m.next_modes:
                    frontier_modes.append(m)
                else:
                    remaining_modes.append(m)
                    inv_prob.append(1 - (sample_count / total_nodes))

            if self.config.frontier_mode_sampling_probability == 1:
                if not frontier_modes:
                    frontier_modes = reached_modes
                if len(frontier_modes) > 0:
                    p = [1 / len(frontier_modes)] * len(frontier_modes)
                    return random.choices(frontier_modes, weights=p, k=1)[0]
                else:
                    return random.choice(reached_modes)

            if not remaining_modes or not frontier_modes:
                return random.choice(reached_modes)

            total_inverse = sum(
                1 - (sample_counts[m] / total_nodes) for m in remaining_modes
            )
            if total_inverse == 0:
                return random.choice(reached_modes)

            sorted_reached_modes = frontier_modes + remaining_modes
            p = [p_frontier / len(frontier_modes)] * len(frontier_modes)
            inv_prob = np.array(inv_prob)
            p.extend((inv_prob / total_inverse) * p_remaining)

            return random.choices(sorted_reached_modes, weights=p, k=1)[0]
        elif mode_sampling_type == "greedy":
            return reached_modes[-1]
        elif mode_sampling_type == "weighted":
            # sample such that we tend to get similar number of pts in each mode
            w = []
            for m in reached_modes:
                num_nodes = 0
                if m in graph.nodes:
                    num_nodes += len(graph.nodes[m])
                if m in graph.transition_nodes:
                    num_nodes += len(graph.transition_nodes[m])
                w.append(1 / max(1, num_nodes))
            return random.choices(tuple(reached_modes), weights=w)[0]

        return random.choice(reached_modes)

    def _sample_valid_uniform_batch(
        self, graph: MultimodalGraph, batch_size: int, cost: float | None
    ) -> Tuple[List[State], int]:
        new_samples = []
        num_attempts = 0
        num_valid = 0

        if graph.goal_nodes:
            focal_points = np.array(
                [graph.root.state.q.state(), graph.goal_nodes[0].state.q.state()],
                dtype=np.float64,
            )

        while len(new_samples) < batch_size:
            num_attempts += 1
            if num_attempts > 100 * batch_size:
                break

            # print(len(new_samples))
            # sample mode
            m = self._sample_mode(
                self.sorted_reached_modes,
                graph,
                self.config.mode_sampling_type,
                cost is not None,
            )

            # print(m)

            # sample configuration
            q = self.env.sample_config_uniform_in_limits()

            if (
                cost is not None
                and sum(self.env.batch_config_cost(q, focal_points)) > cost
            ):
                continue

            if self.env.is_collision_free(q, m):
                new_samples.append(State(q, m))
                num_valid += 1

            # self.env.show(False)

        print("Percentage of succ. attempts", num_valid / num_attempts)

        return new_samples, num_attempts

    def _sample_uniform_transition_configuration(self, mode, reached_terminal_mode):
        # sample transition at the end of this mode
        if reached_terminal_mode:
            # init next ids: caches version of next ids
            next_ids = self.init_next_ids[mode]
        else:
            next_ids = self.mode_validation.get_valid_next_ids(mode)

        active_task = self.env.get_active_task(mode, next_ids)
        constrained_robot = active_task.robots
        goal_sample = active_task.goal.sample(mode)

        # sample a configuration
        q = []
        end_idx = 0
        for robot in self.env.robots:
            if robot in constrained_robot:
                dim = self.env.robot_dims[robot]
                q.append(goal_sample[end_idx : end_idx + dim])
                end_idx += dim
            else:
                r_idx = self.env.robot_idx[robot]
                lims = self.env.limits[:, r_idx]
                q.append(np.random.uniform(lims[0], lims[1]))
        q = self.env.start_pos.from_list(q)

        return q

    # TODO:
    # - Introduce mode_subset_to_sample
    # - Fix function below:
    # -- reduce side-effects
    def sample_valid_uniform_transitions(
        self,
        g,
        transistion_batch_size: int,
        cost: float | None,
        reached_modes: Set[Mode],
    ) -> Set[Mode]:
        transitions, failed_attemps = 0, 0
        reached_terminal_mode = False

        # if we did not yet reach the goal mode, sample using the specified initial sampling strategy
        if len(g.goal_nodes) == 0:
            mode_sampling_type = self.config.init_mode_sampling_type
        else:
            mode_sampling_type = self.config.mode_sampling_type

        # if we already found goal nodes, we construct the focal points of our ellipse
        if len(g.goal_nodes) > 0:
            focal_points = np.array(
                [g.root.state.q.state(), g.goal_nodes[0].state.q.state()],
                dtype=np.float64,
            )

        # if we reached the goal, but we have not found a path yet, we set reached_terminal_mode to True
        # reason: only sample the mode sequence that lead us to the terminal mode
        if cost is None and len(g.goal_nodes) > 0 and self.config.with_mode_validation:
            reached_terminal_mode = True

        # If sorted_reached_modes is not up to date, update it
        # I am not sure if this should be happening here -> symptom for other problems?
        if len(reached_modes) != len(self.sorted_reached_modes):
            if not reached_terminal_mode:
                self.sorted_reached_modes = sorted(reached_modes, key=lambda m: m.id)

        # sorted reached modes is mainly for debugging and reproducability
        mode_subset_to_sample = self.sorted_reached_modes

        while (
            transitions < transistion_batch_size
            and failed_attemps < 5 * transistion_batch_size
        ):
            # sample mode
            mode = self._sample_mode(
                mode_subset_to_sample, g, mode_sampling_type, cost is None
            )

            q = self._sample_uniform_transition_configuration(
                mode, reached_terminal_mode
            )

            # could this transition possibly improve the path?
            if (
                cost is not None
                and sum(self.env.batch_config_cost(q, focal_points)) > cost
            ):
                failed_attemps += 1
                continue

            # check if the transition is collision free
            if self.env.is_collision_free(q, mode):
                if self.env.is_terminal_mode(mode):
                    valid_next_modes = None
                else:
                    # we only cache the ones that are in the valid sequence
                    if reached_terminal_mode:
                        # we cache the next modes only if they are on the mode path
                        if mode not in self.init_next_modes:
                            next_modes = self.env.get_next_modes(q, mode)
                            valid_next_modes = self.mode_validation.get_valid_modes(
                                mode, list(next_modes)
                            )
                            self.init_next_modes[mode] = valid_next_modes

                        valid_next_modes = self.init_next_modes[mode]
                    else:
                        next_modes = self.env.get_next_modes(q, mode)
                        valid_next_modes = self.mode_validation.get_valid_modes(
                            mode, list(next_modes)
                        )

                        assert not (
                            set(valid_next_modes)
                            & self.mode_validation.invalid_next_ids.get(mode, set())
                        ), "There are invalid modes in the 'next_modes'."

                        # if there are no valid next modes, we add this mode to the invalid modes (and remove them from the reached modes)
                        if valid_next_modes == []:
                            reached_modes = self.mode_validation.track_invalid_modes(
                                mode, reached_modes
                            )

                # if the mode is not (anymore) in the reachable modes, do not add this to the transitions
                if mode not in reached_modes:
                    if not reached_terminal_mode:
                        self.sorted_reached_modes = list(
                            sorted(reached_modes, key=lambda m: m.id)
                        )
                        mode_subset_to_sample = self.sorted_reached_modes
                    continue

                # add the transition to the graph
                g.add_transition_nodes([(q, mode, valid_next_modes)])

                # this seems to be a very strange way of checking if the transition was added?
                # but this seems wrong
                if (
                    len(list(chain.from_iterable(g.transition_nodes.values())))
                    > transitions
                ):
                    transitions += 1

                    # if the mode that we added is the root mode with the state being equal to the root state, do not add it
                    if (
                        mode == g.root.state.mode
                        and np.equal(q.state(), g.root.state.q.state()).all()
                    ):
                        reached_modes.discard(mode)
                        self.dummy_start_mode = True

                else:
                    failed_attemps += 1
                    continue
            else:
                # self.env.show(False)
                failed_attemps += 1
                continue

            if valid_next_modes is not None and len(valid_next_modes) > 0:
                reached_modes.update(valid_next_modes)

            def get_init_mode_sequence(mode: Mode, current_best_cost):
                if self.found_init_mode_sequence:
                    return []

                mode_seq = []
                if current_best_cost is None and len(g.goal_nodes) > 0:
                    assert self.env.is_terminal_mode(mode)

                    self.found_init_mode_sequence = True
                    mode_seq = create_initial_mode_sequence(mode)

                return mode_seq

            def create_initial_mode_sequence(mode: Mode):
                init_search_modes = [mode]
                self.init_next_ids[mode] = None

                # go through the chain of modes that lead us to this mode.
                while True:
                    prev_mode = mode.prev_mode
                    if prev_mode is not None:
                        init_search_modes.append(prev_mode)
                        self.init_next_ids[prev_mode] = mode.task_ids
                        mode = prev_mode
                    else:
                        break

                init_search_modes = init_search_modes[::-1]

                if self.dummy_start_mode and init_search_modes[0] == g.root.state.mode:
                    init_search_modes = init_search_modes[1:]

                return init_search_modes

            # This is called exactly once: when we reach the terminal mode
            init_mode_seq = get_init_mode_sequence(mode, cost)
            if init_mode_seq and self.config.with_mode_validation:
                mode_subset_to_sample = init_mode_seq

                # We override sorted_reached modes for the moment, since this is used as the set we sample from
                self.sorted_reached_modes = mode_subset_to_sample

                reached_terminal_mode = True
                mode_sampling_type = self.config.mode_sampling_type
            elif len(reached_modes) != len(self.sorted_reached_modes):
                if not reached_terminal_mode:
                    self.sorted_reached_modes = list(
                        sorted(reached_modes, key=lambda m: m.id)
                    )
                    mode_subset_to_sample = self.sorted_reached_modes

        print(f"Adding {transitions} transitions")
        print(self.mode_validation.counter)

        return reached_modes

    def _prune(self, g, current_best_cost):
        num_pts_for_removal = 0
        focal_points = np.array(
            [g.root.state.q.state(), g.goal_nodes[0].state.q.state()],
            dtype=np.float64,
        )
        # Remove elements from g.nodes
        for mode in list(g.nodes.keys()):  # Avoid modifying dict while iterating
            original_count = len(g.nodes[mode])
            g.nodes[mode] = [
                n
                for n in g.nodes[mode]
                if sum(self.env.batch_config_cost(n.state.q, focal_points))
                <= current_best_cost
            ]
            num_pts_for_removal += original_count - len(g.nodes[mode])

        # Remove elements from g.transition_nodes
        for mode in list(g.transition_nodes.keys()):
            original_count = len(g.transition_nodes[mode])
            g.transition_nodes[mode] = [
                n
                for n in g.transition_nodes[mode]
                if sum(self.env.batch_config_cost(n.state.q, focal_points))
                <= current_best_cost
            ]
            num_pts_for_removal += original_count - len(g.transition_nodes[mode])

        for mode in list(g.reverse_transition_nodes.keys()):
            original_count = len(g.reverse_transition_nodes[mode])
            g.reverse_transition_nodes[mode] = [
                n
                for n in g.reverse_transition_nodes[mode]
                if sum(self.env.batch_config_cost(n.state.q, focal_points))
                <= current_best_cost
            ]
            # num_pts_for_removal += original_count - len(g.reverse_transition_nodes[mode])

        print(f"Removed {num_pts_for_removal} nodes")

    def _refine_approximation(
        self, g, informed_sampler, reached_modes, current_best_path, current_best_cost
    ):
        # add new batch of nodes
        effective_uniform_batch_size = (
            self.config.uniform_batch_size
            if not self.first_search
            else self.config.init_uniform_batch_size
        )
        effective_uniform_transition_batch_size = (
            self.config.uniform_transition_batch_size
            if not self.first_search
            else self.config.init_transition_batch_size
        )
        self.first_search = False

        # if self.env.terminal_mode not in reached_modes:
        print("Sampling transitions")
        reached_modes = self.sample_valid_uniform_transitions(
            g,
            transistion_batch_size=effective_uniform_transition_batch_size,
            cost=current_best_cost,
            reached_modes=reached_modes,
        )

        # g.add_transition_nodes(new_transitions)
        # print(f"Adding {len(new_transitions)} transitions")

        print("Sampling uniform")
        new_states, required_attempts_this_batch = self._sample_valid_uniform_batch(
            g,
            batch_size=effective_uniform_batch_size,
            cost=current_best_cost,
        )
        g.add_states(new_states)
        print(f"Adding {len(new_states)} new states")

        # nodes_per_state = []
        # for m in reached_modes:
        #     num_nodes = 0
        #     for n in new_states:
        #         if n.mode == m:
        #             num_nodes += 1

        #     nodes_per_state.append(num_nodes)

        # plt.figure("Uniform states")
        # plt.bar([str(mode) for mode in reached_modes], nodes_per_state)
        # plt.show()

        approximate_space_extent = float(
            np.prod(np.diff(self.env.limits, axis=0))
            * len(new_states)
            / required_attempts_this_batch
        )

        # print(reached_modes)

        if not g.goal_nodes:
            return None

        # g.compute_lower_bound_to_goal(self.env.batch_config_cost)
        # g.compute_lower_bound_from_start(self.env.batch_config_cost)

        if (
            current_best_cost is not None
            and current_best_path is not None
            and (
                self.config.try_informed_sampling
                or self.config.try_informed_transitions
            )
        ):
            interpolated_path = interpolate_path(current_best_path)
            # interpolated_path = current_best_path

            if self.config.try_informed_sampling:
                print("Generating informed samples")
                new_informed_states = informed_sampler.generate_samples(
                    list(reached_modes),
                    self.config.informed_batch_size,
                    interpolated_path,
                    try_direct_sampling=self.config.try_direct_informed_sampling,
                    g=g,
                )
                g.add_states(new_informed_states)

                print(f"Adding {len(new_informed_states)} informed samples")

            if self.config.try_informed_transitions:
                print("Generating informed transitions")
                new_informed_transitions = informed_sampler.generate_transitions(
                    list(reached_modes),
                    self.config.informed_transition_batch_size,
                    interpolated_path,
                    g=g,
                )
                g.add_transition_nodes(new_informed_transitions)
                print(f"Adding {len(new_informed_transitions)} informed transitions")

                # g.compute_lower_bound_to_goal(self.env.batch_config_cost)
                # g.compute_lower_bound_from_start(self.env.batch_config_cost)

        return approximate_space_extent

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def plan(
        self,
        ptc: PlannerTerminationCondition,
        optimize: bool = True,
    ) -> Tuple[List[State] | None, Dict[str, Any]]:
        """
        Main entry point for the PRM planner in composite space.
        """
        q0 = self.env.get_start_pos()
        m0 = self.env.get_start_mode()

        assert self.env.is_collision_free(q0, m0)

        reached_modes = set([m0])
        self.sorted_reached_modes = list(sorted(reached_modes, key=lambda m: m.id))

        informed_sampler = InformedSampling(
            self.env,
            "graph_based",
            self.config.locally_informed_sampling,
            include_lb=self.config.inlcude_lb_in_informed_sampling,
        )

        graph = MultimodalGraph(
            State(q0, m0),
            lambda a, b: batch_config_dist(a, b, self.config.distance_metric),
            use_k_nearest=self.config.use_k_nearest,
        )

        current_best_cost = None
        current_best_path = None

        costs = []
        times = []

        add_new_batch = True

        start_time = time.time()

        resolution = self.env.collision_resolution

        all_paths = []

        approximate_space_extent = float(np.prod(np.diff(self.env.limits, axis=0)))

        cnt = 0
        while True:
            if ptc.should_terminate(cnt, time.time() - start_time):
                break
            
            # prune
            if current_best_path is not None and current_best_cost is not None:
                self._prune(graph, current_best_cost)

            print()
            print(f"Samples: {cnt}; time: {time.time() - start_time:.2f}s; {ptc}")
            print(f"Currently {len(reached_modes)} modes")

            samples_in_graph_before = graph.get_num_samples()

            if add_new_batch:
                approximate_space_extent = self._refine_approximation(
                    graph, informed_sampler, reached_modes, current_best_path, current_best_cost
                )

                # update the lower bound to goal cost estimation of nodes.
                graph.compute_lower_bound_to_goal(
                    self.env.batch_config_cost, current_best_cost
                )

            samples_in_graph_after = graph.get_num_samples()
            cnt += samples_in_graph_after - samples_in_graph_before

            # we need to keep adding samples until we have reached a terminal mode
            # with our transitions before we can run a search.
            reached_terminal_mode = False
            for m in reached_modes:
                if self.env.is_terminal_mode(m):
                    reached_terminal_mode = True

            if not reached_terminal_mode:
                continue

            # for m in reached_modes:
            #     plt.figure()
            #     plt.scatter([a.state.q.state()[0] for a in g.nodes[m]], [a.state.q.state()[1] for a in g.nodes[m]])
            #     plt.scatter([a.state.q.state()[2] for a in g.nodes[m]], [a.state.q.state()[3] for a in g.nodes[m]])
            #     # plt.scatter()

            # plt.show()

            # pts_per_mode = []
            # transitions_per_mode = []
            # for m in reached_modes:
            #     num_transitions = 0
            #     if m in g.transition_nodes:
            #         num_transitions += len(g.transition_nodes[m])

            #     num_pts = 0

            #     if m in g.nodes:
            #         num_pts += len(g.nodes[m])

            #     pts_per_mode.append(num_pts)
            #     transitions_per_mode.append(num_transitions)

            # plt.figure()
            # plt.bar([str(mode) for mode in reached_modes], pts_per_mode)

            # plt.figure()
            # plt.bar([str(mode) for mode in reached_modes], transitions_per_mode)

            # plt.show()

            # search over nodes:
            # 1. search from goal state with sparse check
            while True:
                sparsely_checked_path = graph.search(
                    graph.root,
                    graph.goal_nodes,
                    self.env,
                    current_best_cost,
                    resolution,
                    approximate_space_extent,
                )

                # in case this found a path, search with dense check from the other side
                if sparsely_checked_path:
                    add_new_batch = False

                    is_valid_path = True
                    for i in range(len(sparsely_checked_path) - 1):
                        n0 = sparsely_checked_path[i]
                        n1 = sparsely_checked_path[i + 1]

                        s0 = n0.state
                        s1 = n1.state

                        if n0.id in n1.whitelist:
                            continue

                        if not self.env.is_edge_collision_free(
                            s0.q,
                            s1.q,
                            s0.mode,
                            resolution=self.env.collision_resolution,
                            tolerance=self.env.collision_tolerance,
                        ):
                            print("Path is in collision")
                            is_valid_path = False
                            # self.env.show(True)
                            n0.blacklist.add(n1.id)
                            n1.blacklist.add(n0.id)
                            break
                        else:
                            n1.whitelist.add(n0.id)
                            n0.whitelist.add(n1.id)

                    if is_valid_path:
                        path = [node.state for node in sparsely_checked_path]
                        new_path_cost = path_cost(path, self.env.batch_config_cost)
                        if (
                            current_best_cost is None
                            or new_path_cost < current_best_cost
                        ):
                            current_best_path = path
                            current_best_cost = new_path_cost

                            # extract modes
                            modes = [path[0].mode]
                            for p in path:
                                if p.mode != modes[-1]:
                                    modes.append(p.mode)

                            print("Modes of new path")
                            print([m.task_ids for m in modes])
                            # print([(m, m.additional_hash_info) for m in modes])

                            # prev_mode = modes[-1].prev_mode
                            # while prev_mode:
                            #     print(prev_mode, prev_mode.additional_hash_info)
                            #     prev_mode = prev_mode.prev_mode

                            print(
                                f"New cost: {new_path_cost} at time {time.time() - start_time}"
                            )
                            costs.append(new_path_cost)
                            times.append(time.time() - start_time)

                            all_paths.append(path)

                            if self.config.try_shortcutting:
                                print("Shortcutting path")
                                shortcut_path, _ = shortcutting.robot_mode_shortcut(
                                    self.env,
                                    path,
                                    max_iter=self.config.shortcutting_iters,
                                    resolution=self.env.collision_resolution,
                                    tolerance=self.env.collision_tolerance,
                                    robot_choice=self.config.shortcutting_mode,
                                    interpolation_resolution=self.config.shortcutting_interpolation_resolution,
                                )

                                shortcut_path = shortcutting.remove_interpolated_nodes(
                                    shortcut_path
                                )

                                shortcut_path_cost = path_cost(
                                    shortcut_path, self.env.batch_config_cost
                                )

                                if shortcut_path_cost < current_best_cost:
                                    print("New cost: ", shortcut_path_cost)
                                    costs.append(shortcut_path_cost)
                                    times.append(time.time() - start_time)

                                    all_paths.append(shortcut_path)

                                    current_best_path = shortcut_path
                                    current_best_cost = shortcut_path_cost

                                    interpolated_path = shortcut_path

                                    for i in range(len(interpolated_path)):
                                        s = interpolated_path[i]
                                        if not self.env.is_collision_free(s.q, s.mode):
                                            continue

                                        if (
                                            i < len(interpolated_path) - 1
                                            and interpolated_path[i].mode
                                            != interpolated_path[i + 1].mode
                                        ):
                                            # add as transition
                                            graph.add_transition_nodes(
                                                [
                                                    (
                                                        s.q,
                                                        s.mode,
                                                        [interpolated_path[i + 1].mode],
                                                    )
                                                ]
                                            )
                                            pass
                                        else:
                                            graph.add_states([s])

                        add_new_batch = True

                        # plt.figure()

                        # plt.plot([pt.q.state()[0] for pt in current_best_path], [pt.q.state()[1] for pt in current_best_path], 'o-')
                        # plt.plot([pt.q.state()[2] for pt in current_best_path], [pt.q.state()[3] for pt in current_best_path], 'o-')

                        # plt.show()

                        break

                else:
                    print("Did not find a solution")
                    add_new_batch = True
                    break

            if current_best_cost is not None:
                # check if we might have reached the optimal cost? Straightline connection
                if (
                    np.linalg.norm(
                        current_best_cost
                        - self.env.config_cost(q0, graph.goal_nodes[0].state.q)
                    )
                    < 1e-6
                ):
                    break

            if not optimize and current_best_cost is not None:
                break

        if len(costs) > 0:
            costs.append(costs[-1])
            times.append(time.time() - start_time)

        info = {"costs": costs, "times": times, "paths": all_paths}

        # pts_per_mode = []
        # transitions_per_mode = []
        # for m in reached_modes:
        #     num_transitions = 0
        #     if m in g.transition_nodes:
        #         num_transitions += len(g.transition_nodes[m])

        #     num_pts = 0

        #     if m in g.nodes:
        #         num_pts += len(g.nodes[m])

        #     pts_per_mode.append(num_pts)
        #     transitions_per_mode.append(num_transitions)

        # plt.figure()
        # plt.title("pts per mode")
        # plt.bar([str(mode) for mode in reached_modes], pts_per_mode)
        # plt.xticks(rotation=90)

        # plt.figure()
        # plt.title("transitions per mode")
        # plt.bar([str(mode) for mode in reached_modes], transitions_per_mode)
        # plt.xticks(rotation=90)

        # plt.show()

        return current_best_path, info
