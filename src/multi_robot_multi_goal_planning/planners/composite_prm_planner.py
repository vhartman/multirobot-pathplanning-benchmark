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
    # Mode exploration
    mode_sampling_type: str = "uniform_reached"
    init_mode_sampling_type: str = "greedy"
    frontier_mode_sampling_probability: float = 0.98

    # Sampling budget
    uniform_batch_size: int = 200
    uniform_transition_batch_size: int = 500
    informed_batch_size: int = 500
    informed_transition_batch_size: int = 500
    init_uniform_batch_size: int = 150
    init_transition_batch_size: int = 90

    # Informed sampling
    try_informed_sampling: bool = True
    locally_informed_sampling: bool = True
    try_informed_transitions: bool = True
    try_direct_informed_sampling: bool = True
    inlcude_lb_in_informed_sampling: bool = False
    
    # Optimization
    try_shortcutting: bool = True
    shortcutting_mode: str = "round_robin"
    shortcutting_iters: int = 250
    shortcutting_interpolation_resolution: float = 0.1

    distance_metric: str = "max_euclidean"
    use_k_nearest: bool = False
    with_mode_validation: bool = False
    with_noise: bool = False

    # TODO (Liam) new
    skill_phase: int = 1
    max_skill_rollouts: int = 1
    informed_entry_selection: bool = False

"""
NOTES (Liam):
Remove later also within code, keep for now for better understanding 
=============================================================================================
Composite PRM Planner

Core idea:
- Composite C-space: all robots are planned together in one joint configuration space
- Build PRM graph across multiple modes
- Alternates uniform sampling (exploration) + informed sampling (exploitation)

Key points:
1. Sample configurations in reached modes + transitions between modes
2. Search graph for path
3. If found, add informed samples near path, prune far nodes, shortcut
4. Repeat

Differences to prioritized planner:
- Shorter, because:
    1. Reusable components: uses MultimodelGraph, InformedSampling, ModeValidation and shortcutting
    2. No time dimension: plans in configuration space only, doesn't track time
    3. No per-robot decomposition: treats all robots as a single composite configuration
- ...
=============================================================================================
"""

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
        Selecty which mode to sample a configuration in. There are 4 different strategies for the mode sampling:
        - uniformed_reached: Uniformly choose a mode from the list, can pick already expanded modes and find alternative transitions 
        - frontier: Pick randomly a mode that wasn't explored yet, balance exploration
        - greedy: Pick the newest mode from frontier list (mode that wasn't explored yet)
        - weighted: Pick randomly (weighted choice) from the list, giving more probability to less sampled modes
        """
        # Strategy A: Equal probability for each mode
        # Can waste effort on already well-explored modes  
        if mode_sampling_type == "uniform_reached":
            return random.choice(reached_modes)
        
        # Strategy B: Modes with fewer samples get higher probability among remaining modes
        # Biased toward sampling frontier modes (modes that haven't been explored) to balance exploration
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

            # Check for frontier modes in the list of so far discovered modes
            for m in reached_modes:
                sample_count = graph.get_num_samples_in_mode(m)
                sample_counts[m] = sample_count
                if not m.next_modes: 
                    frontier_modes.append(m) # No transition to any successor ever found -> frontier mode
                else:
                    remaining_modes.append(m) # Modes already having known successors
                    inv_prob.append(1 - (sample_count / total_nodes)) # Higher inverse fraction for modes with fewer samples

            # Special case: only frontier mode should be sampled
            if self.config.frontier_mode_sampling_probability == 1:
                if not frontier_modes:
                    frontier_modes = reached_modes
                if len(frontier_modes) > 0:
                    p = [1 / len(frontier_modes)] * len(frontier_modes)
                    return random.choices(frontier_modes, weights=p, k=1)[0]
                else:
                    return random.choice(reached_modes)
            
            # Fallback to uniform if either partition is empty
            if not remaining_modes or not frontier_modes:
                return random.choice(reached_modes)

            # Build probability distribution
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
        
        # Strategy C: Always pick most recently reached mode
        # Most direct progression possible towards the terminal mode
        elif mode_sampling_type == "greedy":
            return reached_modes[-1] # Newest frontier
        
        # Strategy D: Inversely proportional to node count (balances samples across modes)
        # Doesn't care about frontier, purely cares about equal sample count
        elif mode_sampling_type == "weighted":
            # sample such that we tend to get similar number of pts in each mode
            w = []
            for m in reached_modes:
                num_nodes = 0
                if m in graph.nodes:
                    num_nodes += len(graph.nodes[m])
                if m in graph.transition_nodes:
                    num_nodes += len(graph.transition_nodes[m])
                w.append(1 / max(1, num_nodes)) # Weight each mode by inverse
            return random.choices(tuple(reached_modes), weights=w)[0]

        return random.choice(reached_modes)

    def _sample_valid_uniform_batch(
        self, 
        graph: MultimodalGraph, 
        batch_size: int, 
        cost: float | None,
        non_skill_modes: List[Mode] | None = None,
    ) -> Tuple[List[State], int]:
        """
        Generate batch of collision-free free-space configurations (general roadmap nodes),
        not transition nodes. They fill in the interior of modes, so graph search can find paths 
        within a mode between transition points. 

        Rejection sampling:
        - Only keep samples within ellipse with foci at start and goal (in composite C-space it's
          a hyperellipsoid?)
        - Only keep samples that are collision-free
        """
        new_samples = []
        num_attempts = 0
        num_valid = 0

        # If graph already discovered goal nodes, build foci of ellipsoid in config space
        # Foci: start node and (first) goal node 
        if graph.goal_nodes:
            focal_points = np.array(
                [graph.root.state.q.state(), graph.goal_nodes[0].state.q.state()],
                dtype=np.float64,
            )

        # SAMPLING: repeat until batch_size valid samples are found
        while len(new_samples) < batch_size:
            num_attempts += 1
            if num_attempts > 100 * batch_size:
                break

            # Step 1: sample mode
            m = self._sample_mode(
                non_skill_modes if non_skill_modes is not None else self.sorted_reached_modes,
                # self.sorted_reached_modes, 
                graph,
                self.config.mode_sampling_type,
                cost is not None,
            )

            # Step 2: sample configuration (uniformly within joint limits)
            # TODO: QUESTION Doesn't use mode information (unlike sample_valid_uniform_transitions). Why doesn't use mode?
            # TODO: because purpose simply is to densify roadmap within free C-space? need mode only for collision check
            q = self.env.sample_config_uniform_in_limits()

            # Step 3: rejection (ellipsoid) and collision check 
            if (
                cost is not None
                and sum(self.env.batch_config_cost(q, focal_points)) > cost # cost = length of current best path
            ):
                continue
            
            if self.env.is_collision_free(q, m):
                new_samples.append(State(q, m))
                num_valid += 1

        print("Percentage of succ. attempts", num_valid / num_attempts)

        return new_samples, num_attempts

    def _sample_uniform_transition_configuration(self, mode, reached_terminal_mode):
        """
        Sample a single composite configuration that satisfies a mode transition constraint.
        Robots responsible for the active task are placed at a goal configuration, while
        other robots are placed randomly. 

        These nodes enable switching between modes. 
        
        Example: Mode A: task_ids=[0,1,2] -> Mode B: task_ids=[3,1,2]
                 Robot 0 completed task 0
            For this transition to be geometrically valid, robot 0 must be at a configuration
            that satisfies task 0's goal! 
        """
        # Step 1: Choose which transition to attempt
        if reached_terminal_mode:
            # If path to goal already exists, looks at cached sequence leading to goal (exploits good known transitions)
            next_ids = self.init_next_ids[mode]
        else:
            # Planner still is exploring, get from oravle any lofically valid task that could be completed next 
            next_ids = self.mode_validation.get_valid_next_ids(mode)

        # Step 2: Identify the active task (being completed) and its constraints
        active_task = self.env.get_active_task(mode, next_ids) # By comparing elementwise which task ID changed -> finished task
        constrained_robot = active_task.robots # Need to satisfy goal constraint for transition to happen
        goal_sample = active_task.goal.sample(mode) # Draw sample from goal region

        # Step 3: Build the composite configuration
        q = []
        end_idx = 0
        for robot in self.env.robots:
            if robot in constrained_robot: # Constrained robots
                dim = self.env.robot_dims[robot]
                q.append(goal_sample[end_idx : end_idx + dim]) # Constrained to be at sampled goal config
                end_idx += dim
            else: # Unconstrained robots
                r_idx = self.env.robot_idx[robot]
                lims = self.env.limits[:, r_idx]
                q.append(np.random.uniform(lims[0], lims[1])) # Free to be anywhere (within limits)
        q = self.env.start_pos.from_list(q)

        return q
    
    def _skill_rollout(self, g, mode, active_task, used_entry_ids):
        """
        Rollout skill from an entry node and add as skill chain into PRM graph

        Phase 1: frozen inactive robots, random entry (from reverse_transition_nodes)
        Phase 2: interpolated inactive robots, informed entry
        """        
        # 1. Pick entry node for skill
        candidate_entry_nodes = g.reverse_transition_nodes.get(mode, [])
        if not candidate_entry_nodes:
            candidate_entry_nodes = g.nodes.get(mode, [])

        if not candidate_entry_nodes:
            return False, None

        # Filter already used entries
        if used_entry_ids:
            candidate_entry_nodes = [n for n in candidate_entry_nodes if n.id not in used_entry_ids]
        if not candidate_entry_nodes:
            return False, None

        # 2. Select entry node (informed or random)
        if self.config.informed_entry_selection:
            entry_node = self._select_informed_entry(candidate_entry_nodes, active_task, mode) # TODO not implemented
        else:
            entry_node = random.choice(candidate_entry_nodes)
        
        q_entry = entry_node.state.q.state()
        
        # 3. Identify active joints for this task & extract starting configuration for active joints 
        active_joints = []
        for r in active_task.robots:
            active_joints.extend(self.env.robot_joints[r])
        active_task.skill.joints = active_joints
        
        all_joints = []
        for r in self.env.robots:
            all_joints.extend(self.env.robot_joints[r])
            
        parts = []
        offset = 0
        for r in self.env.robots:
            dim = self.env.robot_dims[r]
            if r in active_task.robots:
                parts.append(q_entry[offset:offset+dim])
            offset += dim            
        q_init = np.concatenate(parts)
        
        # 4. Rollout skill
        skill_result = active_task.skill.rollout(q_init, active_task, all_joints, self.env, t0=0.0)
        skill_traj = skill_result.trajectory
                
        # 5. Reconstruct composite trajectory: active robots (skill rollout) + inactive robots (phase 1: frozen, phase 2: interpolated)
        if self.config.skill_phase == 1:
            composite_traj = self._build_composite_traj_frozen(
                q_entry, skill_traj, active_task
            )
        else:
            composite_traj = self._build_composite_traj_sampled(
                # TODO not implemented
            ) 

        # 6. Collision check # TODO only waypoint check for now. 
        for step_idx in range(len(composite_traj)):
            q_check = self.env.start_pos.from_flat(composite_traj[step_idx])
            if not self.env.is_collision_free(q_check, mode):
                # self.env.C.view(True)
                return False, None 
            
        # TODO fallback strategy
        # Add some fallback for phase 2 instead of throwing traj away in step 6..? 
        # Otherwise could it be that phase 2 would perform worse than phase 1..
        # - If traj not valid -> try frozen inactive robots (phase 1)?
        # -  
            
        # DEBUG
        dist_moved = np.linalg.norm(composite_traj[0] - composite_traj[-1])
        print(f"[DEBUG ROLLOUT] Mode {mode.id} Rollout | Steps: {len(composite_traj)} | Distance: {dist_moved:.4f}")
        # return True

        # 7. Compute valid next modes
        q_final = self.env.start_pos.from_flat(composite_traj[-1])
        if self.env.is_terminal_mode(mode):
            valid_next_modes = []
        else:
            next_modes = self.env.get_next_modes(q_final, mode)
            valid_next_modes = self.mode_validation.get_valid_modes(mode, list(next_modes))

            if not valid_next_modes:
                return False, None 

        # 8. Convert to PRM States, mark them as skill waypoints and add to graph
        states = [State(self.env.start_pos.from_flat(q), mode, is_skill_waypoint=True) for q in composite_traj]
        g.add_skill_path(entry_node, states, valid_next_modes)
        
        # print(f"[DEBUG ROLLOUT] Successfully added {len(states)} protected nodes into Mode {mode.id}")
        # print(f"[DEBUG LINKED SEQUENCE] Linked {len(states)} nodes to entry_node {entry_node.id} in Mode {mode.id}")        
        return True, valid_next_modes
        
    def _build_composite_traj_frozen(self, q_entry, skill_traj, active_task):
        """
        Phase 1: inactive robots stay at q_entry
        """
        composite_traj = []

        for step_q_active in skill_traj:
            full_q = q_entry.copy()
            active_offset = 0
            full_offset = 0
            for r in self.env.robots:
                dim = self.env.robot_dims[r]
                if r in active_task.robots:
                    full_q[full_offset:full_offset+dim] = step_q_active[active_offset:active_offset+dim]
                    active_offset += dim
                full_offset += dim
            composite_traj.append(full_q)
        
        return composite_traj
    
    def _build_composite_traj_sampled(self):
        """
        Phase 2: inactive robots get interpolated toward ???
        """
        # 
        raise NotImplementedError
    
    def _select_informed_entry(self, candidate_entry_nodes, active_task, mode):
        """
        Pick entry node with active-robot config closest to skill's goal
        """
        # TODO not needed for now, as mode switch happens at SingleGoals -> returns every time same config -> only ever creates 1 entry_node -> 1 candidate
        goal_sample = active_task.goal.sample(mode)
        # - Loop through candidate nodes
        # -- Loop through active robots
        # --- Compute distance between active robots & goal_sample
        # -- Update best distance and best node
        # - Return best node  
        raise NotImplementedError

    def _mode_has_skill(self, mode: Mode) -> bool:
        """Check if any robot's task in this mode has a skill"""
        for task_id in mode.task_ids:
            if task_id is not None and getattr(self.env.tasks[task_id], 'skill', None) is not None:
                return True
        return False

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
        """
        Sample and add transition nodes to the graph to expand mode exploration.
        -> Update reached_modes

        Transition node: configuration satisfying goal constraint for moving from one mode to another. 
        Basically connections between different modes in the graph.

        Flow:
        - Sample modes using configured strategy (uniform, frontier, greedy, weighted)
        - For each mode, sample configurations satisfying transition goals
        - Validate those nodes and determine valid successor modes
        - Add transitions to graph 
        """
        transitions, failed_attemps = 0, 0
        reached_terminal_mode = False
        
        # PHASE 1: setup
        # Sampling strategy selection: 
        if len(g.goal_nodes) == 0:
            # No terminal mode has been reached (goal_nodes are nodes whose state is in the terminal mode)
            # Use initial mode sampling strategy (exploratory)
            mode_sampling_type = self.config.init_mode_sampling_type
        else:
            # Once goal mode has been reached
            # Use mode sampling strategy (more focused)
            mode_sampling_type = self.config.mode_sampling_type

        # Focal points of ellipse for informed sampling, used later for sampling rejection 
        if len(g.goal_nodes) > 0:
            # Goal nodes already found
            focal_points = np.array( # Start: joint configs (all robots) @ t=0, Goal: joint configs (all robots) at first found terminal node
                [g.root.state.q.state(), g.goal_nodes[0].state.q.state()],
                dtype=np.float64,
            )

        # Terminal mode flag
        # When True: restrict sampling to only the mode sequence leading toward the terminal mode 
        # Sort of exploitation-over-exploration switch 
        if cost is None and len(g.goal_nodes) > 0 and self.config.with_mode_validation:
            # No valid path found yet & sampled a transition into a terminal mode & mode validation enabled (for pruning)
            reached_terminal_mode = True # From now on, 

        # Sorted reached modes
        if len(reached_modes) != len(self.sorted_reached_modes):
            # Update sorted_reached_modes if not up to date?
            # TODO: I am not sure if this should be happening here -> symptom for other problems? (Valentin)
            if not reached_terminal_mode:
                self.sorted_reached_modes = sorted(reached_modes, key=lambda m: m.id)

        # Mainly for debugging and reproducability
        mode_subset_to_sample = self.sorted_reached_modes

        # PHASE 2: main sampling loop
        while (
            transitions < transistion_batch_size
            and failed_attemps < 5 * transistion_batch_size
        ):
            # Step 1: pick a mode to sample a transition for, using chosen strategy
            mode = self._sample_mode(
                mode_subset_to_sample, g, mode_sampling_type, cost is None
            )

            # TODO (Liam) new
            # Step 2A: intercept if task is a skill
            # 1. Get task for this mode (just like in _sample_uniform_transition_configuration)
            if reached_terminal_mode:
                next_ids = self.init_next_ids.get(mode)
            else:
                next_ids = self.mode_validation.get_valid_next_ids(mode)                
            active_task = self.env.get_active_task(mode, next_ids)
                
            # 2. Intercept if task is a skill
            if getattr(active_task, 'skill', None) is not None:
                # print(f"[DEBUG SKILL] Intercepted skill for mode {mode.id}")

                # Skip rollout if there already is a skill trajectory for this mode
                existing_skill_start_nodes = [
                    n for n in g.nodes.get(mode, [])
                    if n.state.is_skill_waypoint and n.skill_step == 0
                ]
                if len(existing_skill_start_nodes) >= self.config.max_skill_rollouts: # Caps at max_skill_rollouts skill trajectories per mode
                    failed_attemps += 1
                    continue

                # Track which entry nodes already have a rollout (avoid duplicates)
                used_entry_ids = set()
                if self.config.skill_phase == 1:
                    for sn in existing_skill_start_nodes:
                        for candidate_entry_node in g.reverse_transition_nodes.get(mode, []):
                            if candidate_entry_node.id in sn.whitelist:
                                used_entry_ids.add(candidate_entry_node.id)

                # Run the rollout
                success, valid_next_modes = self._skill_rollout(g, mode, active_task, used_entry_ids)
                
                if success:
                    transitions += 1 
                    if valid_next_modes:
                        reached_modes.update(valid_next_modes)
                else:
                    failed_attemps += 1
                    
                # Skip the normal random geometric sampling for this iteration
                # That way skill modes get exactly 1 skill-based transition and 0 geometric transitions
                continue 

            # TODO (Liam) rest unchanged
            # Step 2B: if no skill, sample a transition configuration in that mode
            # Generates config q with active robots constrained to goal positions & other robots free
            # Uses mode information (unlike _sample_valid_uniform_batch), because transition nodes need to satisfy specific goal/task constraints
            q = self._sample_uniform_transition_configuration(
                mode, reached_terminal_mode
            )

            # Step 3: informed (cost-based) rejection if path already exists
            if (
                cost is not None
                and sum(self.env.batch_config_cost(q, focal_points)) > cost
            ): # If sum of distances from sample to focal points > current best cost
                failed_attemps += 1 
                continue # Reject samples outside the informed ellipsoid

            # Step 4-5: collision check & compute valid next modes
            # TODO: QUESTION get_next_modes() can be seen as the oracle (returns which modes are reachable next)?
            if self.env.is_collision_free(q, mode):
                # Current mode is a terminal mode
                if self.env.is_terminal_mode(mode):
                    valid_next_modes = None # No successors (all tasks for all robots completed)
                else:
                    # Current mode is not a terminal mode BUT we already know a path to the goal
                    if reached_terminal_mode:
                        if mode not in self.init_next_modes: # First encounter of mode
                            # Compute and cache valid next modes on the successful path
                            next_modes = self.env.get_next_modes(q, mode)
                            valid_next_modes = self.mode_validation.get_valid_modes(
                                mode, list(next_modes)
                            )
                            self.init_next_modes[mode] = valid_next_modes
                        # Use cached init_next_modes instead of recomputing
                        valid_next_modes = self.init_next_modes[mode]
                    # Current mode is not a terminal mode AND we are still exploring
                    else:
                        # Compute next modes & validate them (no caching)
                        next_modes = self.env.get_next_modes(q, mode)
                        valid_next_modes = self.mode_validation.get_valid_modes(
                            mode, list(next_modes)
                        )

                        assert not (
                            set(valid_next_modes)
                            & self.mode_validation.invalid_next_ids.get(mode, set())
                        ), "There are invalid modes in the 'next_modes'."

                        # Current mode is DEAD end
                        if valid_next_modes == []:
                            # If no valid next modes exist, mark this mode as invalid & remove from reached modes
                            reached_modes = self.mode_validation.track_invalid_modes(
                                mode, reached_modes
                            )

                # Step 6: handle pruned modes
                # If mode validation removed this mode, skip adding the transition and update sampling set
                if mode not in reached_modes:
                    if not reached_terminal_mode:
                        self.sorted_reached_modes = list( # TODO Actual sampling pool, QUESTION sorted for determinism/reproducibility?
                            sorted(reached_modes, key=lambda m: m.id)
                        )
                        mode_subset_to_sample = self.sorted_reached_modes
                    continue # Skip rest of loop as mode is no longer valid

                # Step 7: add the transition config with its valid next modes to the graph
                g.add_transition_nodes([(q, mode, valid_next_modes)])

                # Step 8: verify it was actually added?
                # This seems to be a very strange way of checking if the transition was added? (Valentin)
                # but this seems wrong (Valentin)
                if (
                    len(list(chain.from_iterable(g.transition_nodes.values()))) # Total transition nodes across all modes
                    > transitions
                ):
                    transitions += 1

                    # If the mode that we added is the root mode with the state being equal to the root state, do not add it
                    if (
                        mode == g.root.state.mode
                        and np.equal(q.state(), g.root.state.q.state()).all()
                    ):
                        reached_modes.discard(mode)
                        self.dummy_start_mode = True

                else: # Graph rejected transition node (duplicate)
                    failed_attemps += 1
                    continue
            else: # Graph rejected transition node (collision)
                # self.env.C.view(True)
                failed_attemps += 1
                continue
            
            # Step 9: Update reached modes with valid new successor modes (grow set of modes we can sample in)
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

            # Trace graph backward from current mode to start mode
            # Returns sequence of modes first time planner reached terminal mode
            # Returns empty list if planner already found a path previously 
            init_mode_seq = get_init_mode_sequence(mode, cost)

            # Planner connected start to goal for very first time
            # (Exploitation) 
            if init_mode_seq and self.config.with_mode_validation:
                mode_subset_to_sample = init_mode_seq
                self.sorted_reached_modes = mode_subset_to_sample
                reached_terminal_mode = True
                mode_sampling_type = self.config.mode_sampling_type
            
            # (Exploration) -> only update sorted_reached_modes with new valid_next_modes if we are still exploring
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
        """
        Discards nodes that don't improve shorter path.
        """
        num_pts_for_removal = 0
        focal_points = np.array( # Foci are start and goal nodes
            [g.root.state.q.state(), g.goal_nodes[0].state.q.state()],
            dtype=np.float64,
        )

        # Remove elements from g.nodes 
        for mode in list(g.nodes.keys()): # Avoid modifying dict while iterating
            original_count = len(g.nodes[mode])
            g.nodes[mode] = [
                n for n in g.nodes[mode]
                if n.state.is_skill_waypoint or
                    sum(self.env.batch_config_cost(n.state.q, focal_points)) <= current_best_cost
            ]
            num_pts_for_removal += original_count - len(g.nodes[mode])

        # Remove elements from g.transition_nodes
        for mode in list(g.transition_nodes.keys()):
            original_count = len(g.transition_nodes[mode])
            g.transition_nodes[mode] = [
                n for n in g.transition_nodes[mode]
                if n.state.is_skill_waypoint or
                    sum(self.env.batch_config_cost(n.state.q, focal_points)) <= current_best_cost
            ]
            num_pts_for_removal += original_count - len(g.transition_nodes[mode])
        
        # Remove elements from g.reverse_transition_nodes
        for mode in list(g.reverse_transition_nodes.keys()):
            original_count = len(g.reverse_transition_nodes[mode])
            g.reverse_transition_nodes[mode] = [
                n for n in g.reverse_transition_nodes[mode]
                if n.state.is_skill_waypoint or 
                    sum(self.env.batch_config_cost(n.state.q, focal_points)) <= current_best_cost
            ]

        print(f"Removed {num_pts_for_removal} nodes")

    def _refine_approximation(
        self, g, informed_sampler, reached_modes, current_best_path, current_best_cost
    ):
        """
        Densify roadmap by adding new samples. After initial path, add more nodes to the graph
        in strategic locations to find better (shorter) paths. 

        - Uniform transitions -> explore more connectivity
        - Uniform states -> explore configuration space
        - Informed samples (if path exists) -> improve current solution
        """
        # Step 1: determine batch sizes 
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

        # Step 2: add transition nodes to connect modes
        print("Sampling transitions")
        reached_modes = self.sample_valid_uniform_transitions(
            g,
            transistion_batch_size=effective_uniform_transition_batch_size,
            cost=current_best_cost,
            reached_modes=reached_modes,
        )

        # Filter modes (no uniform & informed sampling in skill modes -> should only contain skill transitions and skill nodes!) # TODO check if my reasoning is correct
        non_skill_modes = [m for m in reached_modes if not self._mode_has_skill(m)]

        # Step 3: add regular nodes (not transitions) within modes
        print("Sampling uniform")
        new_states, required_attempts_this_batch = self._sample_valid_uniform_batch(
            g,
            batch_size=effective_uniform_batch_size,
            cost=current_best_cost,
            non_skill_modes=non_skill_modes,
        )

        g.add_states(new_states) # Adding new valid samples (regular nodes) to graph
        print(f"Adding {len(new_states)} new states")

        # Step 4: active estimation of total volume of C_free
        approximate_space_extent = float(
            np.prod(np.diff(self.env.limits, axis=0))   # total config space volume
            * len(new_states)                           # successful samples
            / required_attempts_this_batch              # total attempts
        )

        if not g.goal_nodes:
            return None # Early exit

        # Step 5: informed refinement (biased towards informed set around best path -> ellipsoid)
        if (
            current_best_cost is not None and
            current_best_path is not None and # If we have a path
            (self.config.try_informed_sampling or self.config.try_informed_transitions) and
            non_skill_modes
        ):
            # Interpolate the current best path
            interpolated_path = interpolate_path(current_best_path)

            # Sample configurations near current best path 
            if self.config.try_informed_sampling:
                print("Generating informed samples")
                new_informed_states = informed_sampler.generate_samples(
                    non_skill_modes, #list(reached_modes),
                    self.config.informed_batch_size,
                    interpolated_path,
                    try_direct_sampling=self.config.try_direct_informed_sampling,
                    g=g,
                )
                # new_informed_states = [ # TODO (Liam) remove -> taken care of in sampling_informed.py
                #     s for s in new_informed_states if not self._mode_has_skill(s.mode)
                # ]
                # Add those informed new states to the graph 
                g.add_states(new_informed_states)

                print(f"Adding {len(new_informed_states)} informed samples")

            # Sample transition nodes near mode transitions in current path
            if self.config.try_informed_transitions:
                print("Generating informed transitions")
                new_informed_transitions = informed_sampler.generate_transitions(
                    non_skill_modes, #list(reached_modes),
                    self.config.informed_transition_batch_size,
                    interpolated_path,
                    g=g,
                )
                # new_informed_transitions = [ # TODO (Liam) remove -> taken care of in sampling_informed.py
                #     (q, m, nm) for q, m, nm in new_informed_transitions if not self._mode_has_skill(m)
                # ]
                # Add those informed new transition nodes to the graph
                g.add_transition_nodes(new_informed_transitions)
                print(f"Adding {len(new_informed_transitions)} informed transitions")

        return approximate_space_extent

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def plan(
        self,
        ptc: PlannerTerminationCondition,
        optimize: bool = True,
    ) -> Tuple[List[State] | None, Dict[str, Any]]:
        """
        Main entry point for the PRM planner in composite space.
        1. Init graph
        2. Loop: 
            - Prune nodes outside ellipse (if solution exists)
            - Add new samples: uniform transitions and states / informed samples (if solution exists)
            - If terminal mode reached -> search graph (sparse collision check)
            - Validate path (dense collision check): invlid -> blacklist, valid -> update best path
            - Shortcut path and add to graph
            - CHeck early termination
        3. Return best path found
        """
        # SETUP
        # Get start configuration and mode + initialize reached modes
        q0 = self.env.get_start_pos()
        m0 = self.env.get_start_mode()

        assert self.env.is_collision_free(q0, m0)

        reached_modes = set([m0]) # Set of modes M that we reached so far
        self.sorted_reached_modes = list(sorted(reached_modes, key=lambda m: m.id))

        # Create informed sampler
        informed_sampler = InformedSampling(
            self.env,
            "graph_based",
            self.config.locally_informed_sampling,
            include_lb=self.config.inlcude_lb_in_informed_sampling,
        )

        # Initialize graph with start state as root
        graph = MultimodalGraph(
            State(q0, m0), # Root: init config and init mode
            lambda a, b: batch_config_dist(a, b, self.config.distance_metric),
            use_k_nearest=self.config.use_k_nearest,
        )

        # Tracking variables
        current_best_cost = None
        current_best_path = None
        costs = []
        times = []
        add_new_batch = True
        start_time = time.time()
        resolution = self.env.collision_resolution
        all_paths = []
        approximate_space_extent = float(np.prod(np.diff(self.env.limits, axis=0)))
        
        # MAIN PLANNING LOOP
        cnt = 0
        while True:
            if ptc.should_terminate(cnt, time.time() - start_time):
                break # Loop until termination condition met
            
            # If we have solution, remove nodes outside informed ellipsoid (prune)
            if current_best_path is not None and current_best_cost is not None:
                self._prune(graph, current_best_cost)

            print()
            print(f"Samples: {cnt}; time: {time.time() - start_time:.2f}s; {ptc}")
            print(f"Currently {len(reached_modes)} modes")

            # Count samples before and after to track progress
            samples_in_graph_before = graph.get_num_samples()

            # PART 1: SAMPLING REFINEMENT
            if add_new_batch: # Adds uniform + informed samples
                approximate_space_extent = self._refine_approximation( 
                    graph, informed_sampler, reached_modes, current_best_path, current_best_cost
                )

                # Update lower bound to goal cost estimation of nodes (for A* heuristic in graph search)
                graph.compute_lower_bound_to_goal(
                    self.env.batch_config_cost, current_best_cost
                )

            samples_in_graph_after = graph.get_num_samples()
            cnt += samples_in_graph_after - samples_in_graph_before # Update count for ptc

            # Keep sampling until we have reached a terminal mode
            # Can't search for a solution else
            reached_terminal_mode = False
            for m in reached_modes:
                if self.env.is_terminal_mode(m):
                    reached_terminal_mode = True
            if not reached_terminal_mode:
                continue # Stop current iteration -> stay in outer loop (sampling)

            # PART 2: GRAPH SEARCH (search over nodes)
            while True:

                # # TODO (Liam) DEBUG remove later -> print skill mode contents before every search
                # for m in reached_modes:
                #     if self._mode_has_skill(m):
                #         nodes_in_mode = graph.nodes.get(m, [])
                #         trans_in_mode = graph.transition_nodes.get(m, [])
                #         rev_trans_in_mode = graph.reverse_transition_nodes.get(m, [])
                #         skill_chain = graph.skill_chain_nodes.get(m, [])
                #         non_skill_trans = [n for n in trans_in_mode if not n.state.is_skill_waypoint]
                #         non_skill_nodes = [n for n in nodes_in_mode if not n.state.is_skill_waypoint]
                #         print()
                #         print(f"[DEBUG GRAPH] Skill mode {m.id}:")
                #         print(f"Nodes: {len(nodes_in_mode)} (non-skill: {len(non_skill_nodes)})")
                #         print(f"Transition_nodes: {len(trans_in_mode)} (non-skill: {len(non_skill_trans)})")
                #         print(f"Reverse_transition_nodes: {len(rev_trans_in_mode)}")
                #         print(f"Skill_chain_nodes: {len(skill_chain)}")
                #         print()

                # Search: A* from root to goal nodes to get candidate path
                # Too expensive to check every single edge.. 
                sparsely_checked_path = graph.search(
                    graph.root,
                    graph.goal_nodes,
                    self.env,
                    current_best_cost,
                    resolution,
                    approximate_space_extent,
                )

                # PART 3: DENSE VALIDATION
                # Dense collision check of candidate path (high resolution)
                if sparsely_checked_path:
                    add_new_batch = False # Found candidate, don't add more samples
                    is_valid_path = True

                    # Validate the path with dense edge checks between consecutive pair of nodes
                    for i in range(len(sparsely_checked_path) - 1):
                        n0 = sparsely_checked_path[i]
                        n1 = sparsely_checked_path[i + 1]

                        s0 = n0.state
                        s1 = n1.state

                        if n0.id in n1.whitelist:
                            continue # Skip, already validated this edge (whitelist)

                        if not self.env.is_edge_collision_free(
                            s0.q,
                            s1.q,
                            s0.mode,
                            resolution=self.env.collision_resolution,
                            tolerance=self.env.collision_tolerance,
                        ):
                            print("Path is in collision")
                            is_valid_path = False
                            n0.blacklist.add(n1.id) # Remember invalid edge (blacklist)
                            n1.blacklist.add(n0.id)
                            break # Continune search loop
                        else:
                            n1.whitelist.add(n0.id) # Cache valid edge (whitelist)
                            n0.whitelist.add(n1.id)

                    if is_valid_path: # Path passed all collision checks
                        path = [node.state for node in sparsely_checked_path]
                        new_path_cost = path_cost(path, self.env.batch_config_cost)

                        # Update tracking variables if valid and better than current best                        
                        if (
                            current_best_cost is None
                            or new_path_cost < current_best_cost
                        ):
                            # Update current best path and cost
                            current_best_path = path
                            current_best_cost = new_path_cost

                            # Extract mode sequence from the path (for reporting)
                            modes = [path[0].mode]
                            for p in path:
                                if p.mode != modes[-1]:
                                    modes.append(p.mode)

                            print("Modes of new path")
                            print([m.task_ids for m in modes])

                            print(
                                f"New cost: {new_path_cost} at time {time.time() - start_time}"
                            )

                            # Append costs, times and path
                            costs.append(new_path_cost)
                            times.append(time.time() - start_time)
                            all_paths.append(path)

                            # PART 4: SHORTCUTTING
                            # Try to connect non-adjacent waypoints to remove unnecessary intermediate points
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

                                # TODO (Liam) remove later -> check but the shortcutter already handles this internally
                                # # Remove interpolated points (just used for collision check)
                                # shortcut_path = shortcutting.remove_interpolated_nodes(
                                #     shortcut_path
                                # )

                                shortcut_path_cost = path_cost(
                                    shortcut_path, self.env.batch_config_cost
                                )

                                # Update current best path and cost
                                if shortcut_path_cost < current_best_cost:
                                    print("New cost: ", shortcut_path_cost)
                                    costs.append(shortcut_path_cost)
                                    times.append(time.time() - start_time)

                                    all_paths.append(shortcut_path)

                                    current_best_path = shortcut_path
                                    current_best_cost = shortcut_path_cost

                                    interpolated_path = shortcut_path

                                    # Check shortcutted path (mode changes?)
                                    # Add path to graph
                                    debug_skipped_skills = 0
                                    debug_added_regular = 0
                                    debug_added_trans = 0

                                    for i in range(len(interpolated_path)):
                                        s = interpolated_path[i]

                                        if self._mode_has_skill(s.mode): # Avoid skill-mode states from being re-added to graph
                                            debug_skipped_skills += 1
                                            continue

                                        if not self.env.is_collision_free(s.q, s.mode):
                                            continue

                                        if (
                                            i < len(interpolated_path) - 1
                                            and interpolated_path[i].mode
                                            != interpolated_path[i + 1].mode
                                        ):
                                            # Mode CHANGES -> add as transition
                                            graph.add_transition_nodes([(s.q, s.mode, [interpolated_path[i + 1].mode])])
                                            debug_added_trans += 1
                                            pass
                                        else:
                                            # Mode SAME -> add as regular node
                                            debug_added_regular += 1
                                            graph.add_states([s])

                                    print(f"[DEBUG SHORTCUTTER] Graph updated! Skipped {debug_skipped_skills} skill waypoints. Added {debug_added_regular} regular nodes and {debug_added_trans} transition nodes.")

                        add_new_batch = True # To add more samples in next iteration
                        break # Exit inner loop

                else: # No path found
                    print("Did not find a solution")
                    add_new_batch = True
                    break # Go back to sampling

            if current_best_cost is not None:
                # Check if we've might reached the optimal cost (straight-line distance)
                if (
                    np.linalg.norm(
                        current_best_cost
                        - self.env.config_cost(q0, graph.goal_nodes[0].state.q)
                    )
                    < 1e-6
                ):
                    break

            if not optimize and current_best_cost is not None:
                break # Return first feasible path (if goal is not to optimize path..)

        if len(costs) > 0:
            costs.append(costs[-1])
            times.append(time.time() - start_time)

        info = {"costs": costs, "times": times, "paths": all_paths}

        return current_best_path, info
