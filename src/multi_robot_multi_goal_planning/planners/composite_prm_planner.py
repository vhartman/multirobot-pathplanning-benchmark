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
from .prm.prm_graph import MultimodalGraph, Node

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
    shortcutting_iters: int = 250 #1000
    shortcutting_interpolation_resolution: float = 0.1

    distance_metric: str = "max_euclidean"
    use_k_nearest: bool = False
    with_mode_validation: bool = False
    with_noise: bool = False

    # TODO (Liam) new
    skill_phase: int = 2                    # 1 (frozen), 2 (lanes), 3 (incremental)
    max_skill_rollouts: int | None = None     # 1 (single), None (unlimited), N (capped)
    skill_corridor_width: int = 50          # phase 2
    skill_k_neighbors: int = 20             # phase 2&3
    skill_batch_size: int = 50              # phase 3

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

        # TODO (Liam) new
        self._used_skill_entry_ids: Dict[Mode, Set[int]] = {}       # mode -> set of successful entry node IDs
        self._failed_skill_entry_ids: Dict[Mode, Set[int]] = {}     # mode -> set of failed entry node IDs
        self._exhausted_skill_modes: Set[Mode] = set()              # no candidate_entry_nodes
        self._saturated_skill_modes: Set[Mode] = set()              # hit max_skill_rollouts

        self._skill_rollout_counts: Dict[Mode, int] = {}            # unified rollout counter for ALL phases
        self._skill_traj_cache: Dict[Mode, np.ndarray] = {}         # mode -> skill trajectory (rollout once, reuse)
        self._skill_valid_next_modes: Dict[Mode, List[Mode]] = {}   # mode -> valid next modes (computed once)
        self._skill_entry_node: Dict[Mode, Node] = {}               # mode -> entry node used for initial rollout

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
                # Check if this robot's task in the current mode has a skill
                robot_idx = self.env.robots.index(robot)
                robot_task_id = mode.task_ids[robot_idx]

                if robot_task_id is not None:
                    robot_task = self.env.tasks[robot_task_id]

                    # Robot has upcoming skill: sample from task goal
                    # (= skill initiation config, e.g. pre_pick pose)
                    if getattr(robot_task, 'skill', None) is not None:
                        skill_init_sample = robot_task.goal.sample(mode)
                        q.append(skill_init_sample)
                        continue

                # Default: random within limits
                r_idx = self.env.robot_idx[robot]
                lims = self.env.limits[:, r_idx]
                q.append(np.random.uniform(lims[0], lims[1])) # Free to be anywhere (within limits)
        q = self.env.start_pos.from_list(q)

        return q
    
    def _skill_rollout(self, g, mode, active_task):
        """
        Rollout skill from an entry node and add as skill chain into PRM graph

        PHASE 1: frozen inactive robots, random entry (from reverse_transition_nodes)
        PHASE 2: interpolated inactive robots, informed entry
        """
        # Phase 3: if trajectory already cached, skip rollout and just add a batch
        if self.config.skill_phase == 3 and mode in self._skill_traj_cache:
            return self._add_skill_batch(g, mode, active_task)
        
        # 1. Get candidate entry nodes for skill
        candidate_entry_nodes = g.reverse_transition_nodes.get(mode, [])
        if not candidate_entry_nodes:
            print(f"[DEBUG SKILL ENTRY] Mode: {mode.id} | No entry nodes yet")
            return False, None # No entry nodes yet, wait

        # 2. Filter out failed entry nodes (both phases) + used entries (phase 1 only)
        # - PHASE 1: must be a new entry node (deterministic skill + frozen -> identical rollout)
        # - PHASE 2: preger unused, but allow reuse up (different inactive configs each time)
        failed_ids = self._failed_skill_entry_ids.get(mode, set())
        used_ids = self._used_skill_entry_ids.get(mode, set())
        candidate_entry_nodes = [n for n in candidate_entry_nodes if n.id not in failed_ids]
        unused_candidates = [n for n in candidate_entry_nodes if n.id not in used_ids]

        # 3. Select entry node from candidates
        if self.config.skill_phase == 1:
            if not unused_candidates:
                self._exhausted_skill_modes.add(mode) # All entry nodes tried
                print(f"[DEBUG ROLLOUT SKILL] Mode: {mode.id} | All entry nodes tried (added mode {mode.id} to exhausted)")
                return False, None
            entry_node = random.choice(unused_candidates)
        else:
            if unused_candidates:
                entry_node = random.choice(unused_candidates)
            elif candidate_entry_nodes:
                entry_node = random.choice(candidate_entry_nodes)
            else:
                self._exhausted_skill_modes.add(mode) # All entry nodes tried
                print(f"[DEBUG ROLLOUT SKILL] Mode: {mode.id} | All entry nodes tried (added mode {mode.id} to exhausted)")
                return False, None
            
        print(f"[DEBUG SKILL ENTRY] Mode: {mode.id} | Phase: {self.config.skill_phase} | Candidates: {len(candidate_entry_nodes)}")        
        q_entry = entry_node.state.q.state()

        # 4. Rollout skill
        # Identify active joints for this task & extract starting configuration for active joints 
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

        # TODO implement caching of skill trajectory for phase 2

        # Rollout skill
        skill_result = active_task.skill.rollout(q_init, active_task, all_joints, self.env, t0=0.0)
        skill_traj = skill_result.trajectory

        # TODO check degenerate skill rollouts... need to define cases for failed rollouts (maybe only the collision..)
        dist_moved = np.linalg.norm(skill_traj[0] - skill_traj[-1])
        if dist_moved < 1e-4:
            self._log_entry_to_dict(mode, entry_node.id, self._failed_skill_entry_ids)
            print(f"[DEBUG SKILL WARNING] Problem with rollout in mode {mode.id}: steps={len(skill_traj)}, dist={dist_moved:.4f}")
            return False, None
            
        # 5. Build composite trajectory (phase specific)
        # - PHASE 1: inactive robots (frozen)
        if self.config.skill_phase == 1:
            composite_traj = self._build_composite_traj_frozen(q_entry, skill_traj, active_task)

            # TODO DEBUG
            print(f"[DEBUG SKILL COLL CHECK] Mode: {mode.id} | a2 frozen at: {q_entry[2:]}")
            print(f"[DEBUG SKILL COLL CHECK] Skill traj length: {len(composite_traj)}")
            print(f"[DEBUG SKILL COLL CHECK] Checking {len(composite_traj)} waypoints...")

            # 6. Collision check (waypoints + edges)
            for step_idx in range(len(composite_traj)):
                q_check = self.env.start_pos.from_flat(composite_traj[step_idx])
                if not self.env.is_collision_free(q_check, mode):
                    # self.env.C.view(True)
                    
                    # TODO DEBUG
                    print(f"[DEBUG SKILL COLLISION] Mode: {mode.id} | Collision at step {step_idx}/{len(composite_traj)}")
                    print(f"[DEBUG SKILL COLLISION] a1 pos: {composite_traj[step_idx][:2]}, a2 pos: {composite_traj[step_idx][2:]}")
                    
                    self._log_entry_to_dict(mode, entry_node.id, self._failed_skill_entry_ids)
                    return False, None 
                # Edge check between consecutive steps # TODO usually not done during graph build.. can be removed (but in phase 1 won't be computationally too important (single chain vs. roadmap..)?)
                if step_idx > 0: 
                    q_prev = self.env.start_pos.from_flat(composite_traj[step_idx - 1])
                    if not self.env.is_edge_collision_free( # TODO check if needed (skill resolution vs. collision resolution)
                        q_prev, q_check, mode, 
                        resolution=self.env.collision_resolution,
                        tolerance=self.env.collision_tolerance,
                    ):
                        self._log_entry_to_dict(mode, entry_node.id, self._failed_skill_entry_ids)
                        print(f"[DEBUG SKILL COLLISION] Mode: {mode.id} | Collision (edge)!)")
                        return False, None

            # 7. Compute valid next modes
            q_final = self.env.start_pos.from_flat(composite_traj[-1])
            if self.env.is_terminal_mode(mode):
                valid_next_modes = []
            else:
                next_modes = self.env.get_next_modes(q_final, mode)
                valid_next_modes = self.mode_validation.get_valid_modes(mode, list(next_modes))

                if not valid_next_modes:
                    self._log_entry_to_dict(mode, entry_node.id, self._failed_skill_entry_ids)
                    return False, None 

            # 8. Add skill points to graph
            states = [
                State(self.env.start_pos.from_flat(q), mode, is_skill_waypoint=True) 
                for q in composite_traj
            ]
            g.add_skill_path(states, valid_next_modes, entry_node)
        
        # 5. Build composite trajectory (phase specific)
        # - PHASE 2: inactive robots (build roadmap: pool of same M inactive configs x N skill steps)
        elif self.config.skill_phase == 2:
            roadmap = self._build_skill_roadmap(q_entry, skill_traj, active_task, mode)
            
            if roadmap is None:
                self._log_entry_to_dict(mode, entry_node.id, self._failed_skill_entry_ids)
                print(f"[DEBUG ROADMAP] Failed to build roadmap in mode {mode.id}")
                return False, None

            # 7. Compute valid next modes (all exit configs share same active robot position)
            q_final = self.env.start_pos.from_flat(roadmap[-1][0][1]) # TODO (Liam) 
            if self.env.is_terminal_mode(mode):
                valid_next_modes = []
            else:
                next_modes = self.env.get_next_modes(q_final, mode)
                valid_next_modes = self.mode_validation.get_valid_modes(mode, list(next_modes))
                if not valid_next_modes:
                    self._log_entry_to_dict(mode, entry_node.id, self._failed_skill_entry_ids)
                    return False, None
            
            # TODO (Liam) pool_idx not needed anymore -> remove
            # Convert to (pool_idx, State) and add to graph
            roadmap_states = [
                [(pool_idx, State(self.env.start_pos.from_flat(q), mode, is_skill_waypoint=True)) 
                 for pool_idx, q in step_entries]
                 for step_entries in roadmap
            ]
            g.add_skill_roadmap(roadmap_states, valid_next_modes, entry_node,
                                k_neighbors=self.config.skill_k_neighbors)

        # 5. Build composite trajectory (phase specific)
        # - PHASE 3: inactive robots (random and incremental)
        elif self.config.skill_phase == 3:
            # TODO (Liam)
            # pass

            # First-time setup: cache trajectory and valid next modes, then add first batch
            
            # Compute valid next modes using entry's inactive config
            # (only the active robot's final position matters for mode transition)
            full_q_final = q_entry.copy()
            active_offset_tmp = 0
            full_offset_tmp = 0
            for r in self.env.robots:
                dim = self.env.robot_dims[r]
                if r in active_task.robots:
                    full_q_final[full_offset_tmp:full_offset_tmp+dim] = skill_traj[-1][active_offset_tmp:active_offset_tmp+dim]
                    active_offset_tmp += dim
                full_offset_tmp += dim

            q_final = self.env.start_pos.from_flat(full_q_final)
            if self.env.is_terminal_mode(mode):
                valid_next_modes = []
            else:
                next_modes = self.env.get_next_modes(q_final, mode)
                valid_next_modes = self.mode_validation.get_valid_modes(mode, list(next_modes))
                if not valid_next_modes:
                    self._log_entry_to_dict(mode, entry_node.id, self._failed_skill_entry_ids)
                    return False, None

            # Cache for all future batches
            self._skill_traj_cache[mode] = skill_traj
            self._skill_valid_next_modes[mode] = valid_next_modes
            self._skill_entry_node[mode] = entry_node

            # Add first batch (and return its result)
            return self._add_skill_batch(g, mode, active_task)

        # print(f"[DEBUG ROLLOUT] Successfully added {len(states)} protected nodes into Mode {mode.id}")
        # print(f"[DEBUG LINKED SEQUENCE] Linked {len(states)} nodes to entry_node {entry_node.id} in Mode {mode.id}")        
        self._skill_rollout_counts[mode] = self._skill_rollout_counts.get(mode, 0) + 1
        self._log_entry_to_dict(mode, entry_node.id, self._used_skill_entry_ids)
        return True, valid_next_modes

    def _build_composite_traj_frozen(self, q_entry, skill_traj, active_task):
        """
        Phase 1: Builds a composite trajectory where active robots follow the skill exactly, 
        and inactive robots stay frozen at q_entry
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
    
    def _build_skill_roadmap(self, q_entry, skill_traj, active_task, mode):
        """
        Phase 2: build a roadmap for the inactive robot through the skill segment.

        """
        # Step 1: Sample a  pool of I collision free inactive robot configs
        # - Only sampling inactive configs (within joint limits)
        # - Not paired yet with any skill steps, just densifying the inactive config space
        #  

        # Step 2: For each skill step k:
        # - For active config: take skill_traj[k]
        # - For inactive config: loop through pool I
        # -- Combine skill_traj[k] and pool[i]
        # -- Collision check the composite config
        # -- Valid: store as valid node (step k, pool index i) 
        # -- Invalid: mark as blocked at this step

        # Step 3: Return valid composite configs by step
        # -  

        # raise NotImplementedError
        
        # TODO: Fix for single agent (didn't work because couldn't sample inactive config pool..) need to find other way to get to the inactive config
        # Single agent guard
        inactive_robots = [r for r in self.env.robots if r not in active_task.robots]
        if not inactive_robots:
            roadmap = []
            for k, step_q_active in enumerate(skill_traj):
                # Only 1 lane -> full_q is the step_q_active
                roadmap.append([(0, step_q_active)])
            return roadmap

        # Multi agent pool generation
        M = self.config.skill_corridor_width
        pool = []

        # 1. Initialize the pool of inactive configs
        # Pool[0] = entry node's inactive config
        entry_inactive = []
        offset = 0
        for r in self.env.robots:
            dim = self.env.robot_dims[r]
            if r not in active_task.robots:
                entry_inactive.append(q_entry[offset:offset+dim])
            offset += dim
        pool.append(np.concatenate(entry_inactive)) 

        # TODO (Liam) find better way to deal with this
        # Check if any inactive robot has a skill in this mode
        # If yes: freeze them at entry config (pool[0] only), don't allow random sampling 
        inactive_robot_has_skill = any(
            getattr(self.env.tasks[mode.task_ids[self.env.robots.index(r)]], 'skill', None) is not None
            for r in self.env.robots
            if r not in active_task.robots
            and mode.task_ids[self.env.robots.index(r)] is not None
        )

        # assert inactive_robot_has_skill == False

        if not inactive_robot_has_skill:

            # 2. Sampling the random pool of inactive configs 
            # Pool[1..M-1] = random inactive configs
            attempts = 0
            while len(pool) < M and attempts < 20 * M:
                attempts += 1
                parts = []
                for r in self.env.robots:
                    if r not in active_task.robots:
                        r_idx = self.env.robot_idx[r]
                        lims = self.env.limits[:, r_idx]
                        parts.append(np.random.uniform(lims[0], lims[1]))
                pool.append(np.concatenate(parts))

        # 3. Build the grid
        # For each step pair each pool config with active config and collision check
        roadmap = [] # roadmap[k] = list of (pool_idx, full_composite_flat)
        for k, step_q_active in enumerate(skill_traj):
            step_valid = []
            for pool_idx, inactive_q in enumerate(pool):
                # Compose full config: active from skill_traj[k], inactive from pool[pool_idx]
                full_q = np.zeros_like(q_entry)
                active_offset = 0
                inactive_offset = 0
                full_offset = 0

                for r in self.env.robots:
                    dim = self.env.robot_dims[r]
                    if r in active_task.robots:
                        full_q[full_offset:full_offset+dim] = step_q_active[active_offset:active_offset+dim]
                        active_offset += dim
                    else:
                        full_q[full_offset:full_offset+dim] = inactive_q[inactive_offset:inactive_offset+dim]
                        inactive_offset += dim
                    full_offset += dim

                # Collision check of active/inactive combination
                q_check = self.env.start_pos.from_flat(full_q)
                if self.env.is_collision_free(q_check, mode):
                    step_valid.append((pool_idx, full_q))

            if len(step_valid) == 0:
                print(f"[DEBUG ROADMAP] Mode {mode.id} | Step {k}: 0 valid configs -> aborting")
                return None
            
            roadmap.append(step_valid)

        return roadmap

    def _add_skill_batch(self, g, mode, active_task):
        """
        Phase 3: Sample a batch of random inactive configs, collision-check against
        each skill step, and add valid nodes to the graph incrementally.
        """
        skill_traj = self._skill_traj_cache[mode]
        valid_next_modes = self._skill_valid_next_modes[mode]
        entry_node = self._skill_entry_node[mode]
        B = self.config.skill_batch_size
        N = len(skill_traj)

        # Identify inactive robots
        inactive_robots = [r for r in self.env.robots if r not in active_task.robots]

        # Extract the entry inactive config (needed for the freeze guard)
        q_entry = entry_node.state.q.state()
        entry_inactive_parts = []
        offset = 0
        for r in self.env.robots:
            dim = self.env.robot_dims[r]
            if r not in active_task.robots:
                entry_inactive_parts.append(q_entry[offset:offset+dim])
            offset += dim
        entry_inactive_q = np.concatenate(entry_inactive_parts) if entry_inactive_parts else np.array([])

        # TODO (Liam) find better way to deal with this
        # Check if any inactive robot has a skill in this mode
        # If yes: freeze them at entry config (pool[0] only), don't allow random sampling 
        inactive_robot_has_skill = False
        if inactive_robots:
            inactive_robot_has_skill = any(
                getattr(self.env.tasks[mode.task_ids[self.env.robots.index(r)]], 'skill', None) is not None
                for r in inactive_robots
                if mode.task_ids[self.env.robots.index(r)] is not None
            )

        # Single agent and freeze guard saturation
        if not inactive_robots:
            self._saturated_skill_modes.add(mode)
            print(f"[DEBUG P3 BATCH] Single-agent detected. Saturating mode {mode.id} after 1 batch")
            B = 1 # No inactive robots -> we need only 1 batch and that's it
        elif inactive_robot_has_skill:
            self._saturated_skill_modes.add(mode)
            print(f"[DEBUG P3 BATCH] Freeze Guard active. Saturating mode {mode.id} after 1 batch")
            B = 1

        # TODO (Liam) instead of sampling B then go through k, sampling B while going through k?

        # TODO approach with B inside loop
        # # Sampling loop
        # batch_states = []  # batch_states[k] = list of valid State objects
        
        # for k in range(N):
        #     step_valid = []
            
        #     # Sample B configs INSIDE the time step loop
        #     for _ in range(B):
                
        #         # Determine inactive_q based on guards vs true randomness
        #         if not inactive_robots:
        #             inactive_q = np.array([])
        #         elif inactive_robot_has_skill:
        #             # Freeze guard: robot is queued for a skill # TODO
        #             inactive_q = entry_inactive_q
        #         else:
        #             # Sample a brand new random config at this specific time step
        #             parts = []
        #             for r in inactive_robots:
        #                 r_idx = self.env.robot_idx[r]
        #                 lims = self.env.limits[:, r_idx]
        #                 parts.append(np.random.uniform(lims[0], lims[1]))
        #             inactive_q = np.concatenate(parts)

        #         # Compose full config
        #         full_q = np.zeros(sum(self.env.robot_dims[r] for r in self.env.robots))
        #         active_offset = 0
        #         inactive_offset = 0
        #         full_offset = 0
        #         for r in self.env.robots:
        #             dim = self.env.robot_dims[r]
        #             if r in active_task.robots:
        #                 full_q[full_offset:full_offset+dim] = skill_traj[k][active_offset:active_offset+dim]
        #                 active_offset += dim
        #             else:
        #                 full_q[full_offset:full_offset+dim] = inactive_q[inactive_offset:inactive_offset+dim]
        #                 inactive_offset += dim
        #             full_offset += dim

        #         # Collision check
        #         q_check = self.env.start_pos.from_flat(full_q)
        #         if self.env.is_collision_free(q_check, mode):
        #             step_valid.append(State(q_check, mode, is_skill_waypoint=True))

        #     batch_states.append(step_valid)

        # TODO approach with B outside
        # 1. Pre-sample the B inactive configurations outside the loop
        pre_sampled_inactive = []
        if not inactive_robots:
            pre_sampled_inactive.append(np.array([]))
        elif inactive_robot_has_skill:
            pre_sampled_inactive.append(entry_inactive_q)
        else:
            for _ in range(B):
                parts = []
                for r in inactive_robots:
                    r_idx = self.env.robot_idx[r]
                    lims = self.env.limits[:, r_idx]
                    parts.append(np.random.uniform(lims[0], lims[1]))
                pre_sampled_inactive.append(np.concatenate(parts))

        # 2. Apply these exact same B configurations across all steps
        batch_states = []
        for k in range(N):
            step_valid = []
            for inactive_q in pre_sampled_inactive:
                
                # Compose full config
                full_q = np.zeros(sum(self.env.robot_dims[r] for r in self.env.robots))
                active_offset = 0
                inactive_offset = 0
                full_offset = 0
                for r in self.env.robots:
                    dim = self.env.robot_dims[r]
                    if r in active_task.robots:
                        full_q[full_offset:full_offset+dim] = skill_traj[k][active_offset:active_offset+dim]
                        active_offset += dim
                    else:
                        full_q[full_offset:full_offset+dim] = inactive_q[inactive_offset:inactive_offset+dim]
                        inactive_offset += dim
                    full_offset += dim

                # Collision check
                q_check = self.env.start_pos.from_flat(full_q)
                if self.env.is_collision_free(q_check, mode):
                    step_valid.append(State(q_check, mode, is_skill_waypoint=True))

            batch_states.append(step_valid)

        # Need at least 1 valid config at step 0 and step N-1 for this batch to be useful
        total_valid = sum(len(s) for s in batch_states)
        if not batch_states[0] or not batch_states[-1]:
            print(f"[DEBUG P3 BATCH] Mode {mode.id} | Batch empty at entry/exit (valid={total_valid}) — skipping")
            return False, None

        # Add to graph
        g.add_skill_batch(
            mode, batch_states, valid_next_modes, entry_node,
            k_neighbors=self.config.skill_k_neighbors,
        )

        print(f"[DEBUG P3 BATCH] Mode {mode.id} | Added {total_valid} nodes across {N} steps")
        return True, valid_next_modes

    def _mode_has_skill(self, mode: Mode) -> bool:
        """Check if any robot's task in this mode has a skill"""
        for task_id in mode.task_ids:
            if task_id is not None and getattr(self.env.tasks[task_id], 'skill', None) is not None:
                return True
        return False

    def _log_entry_to_dict(self, mode, entry_node_id, target_dict):
        """
        Generalized logger:
        - Updates the failed dict if an entry node fails to do a skill rollout.
        - Updates the used dict if an entry node has already been used (phase 1 only)
        """
        if mode not in target_dict:
            target_dict[mode] = set()
        target_dict[mode].add(entry_node_id)

    def _get_valid_modes_to_sample(self, exclude_modes: Set[Mode] = None):
        """
        Filters the reached modes to exclude modes that are:
        1. Exhausted: no candidate entry nodes left 
        2. Saturated: hit the max_skill_rollouts
        3. Rolled out this batch: already rolled out in the current batch (optional)
        """
        exclude = exclude_modes if exclude_modes is not None else set()
        return [
            m for m in self.sorted_reached_modes
            if m not in self._exhausted_skill_modes
            and m not in self._saturated_skill_modes
            and m not in exclude
        ]

    # TODO:
    # - Introduce mode_subset_to_sample
    # - Fix function below:
    # -- reduce side-effects

    def sample_valid_uniform_transitions(
        self,
        g,
        transition_batch_size: int,
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
        transitions, failed_attempts = 0, 0
        modes_rolled_out_this_batch = set()
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

        # PHASE 2: main sampling loop
        while (
            transitions < transition_batch_size
            and failed_attempts < 5 * transition_batch_size
        ):
            # Step 0: Check exhausted skill modes for new entry candidates
            for mode in list(self._exhausted_skill_modes):
                total_candidates = len(g.reverse_transition_nodes.get(mode, []))
                failed_count = len(self._failed_skill_entry_ids.get(mode, set()))

                if self.config.skill_phase == 1:
                    # Un-exhaust if there are more candidates than (failed + used) combined
                    used_count = len(self._used_skill_entry_ids.get(mode, set()))
                    if total_candidates > (failed_count + used_count):
                        self._exhausted_skill_modes.discard(mode)
                        print(f"[DEBUG SKILL ENTRY] Mode: {mode.id} | Exhausted entry node restored")
                else:
                    # Un-exhaust if there are more candidates thatn failed nodes (can reuse successful nodes)
                    if total_candidates > failed_count:
                        self._exhausted_skill_modes.discard(mode)
                        print(f"[DEBUG SKILL ENTRY] Mode: {mode.id} | Exhausted entry node restored")

            # Exclude exhausted/saturated AND "skill modes rolled out in this batch" from sampling pool
            mode_subset_to_sample = self._get_valid_modes_to_sample(exclude_modes=modes_rolled_out_this_batch)
            
            if not mode_subset_to_sample:
                # No modes left to sample in this batch (e.g. all skill modes rolled out once)
                break

            # Step 1: pick a mode to sample a transition for, using chosen strategy
            mode = self._sample_mode(
                mode_subset_to_sample, g, mode_sampling_type, cost is None
            )

            # Step 2A: Identify Tasks
            if reached_terminal_mode:
                next_ids = self.init_next_ids.get(mode)
            else:
                next_ids = self.mode_validation.get_valid_next_ids(mode)       
            active_task = self.env.get_active_task(mode, next_ids)         
      
            # Step 2B: Skill Intercept
            if getattr(active_task, 'skill', None) is not None:
                print(f"[DEBUG SKILL] Intercepted skill for mode {mode.id}")   

                # Run the rollout
                success, valid_next_modes = self._skill_rollout(g, mode, active_task)
                
                if success:
                    # Mark for the rest of THIS batch
                    modes_rolled_out_this_batch.add(mode)

                    if self.config.skill_phase == 3:
                        transitions += self.config.skill_batch_size 
                    else:
                        transitions += 1

                    # Immediate saturation check
                    if self.config.max_skill_rollouts is not None:
                        if self._skill_rollout_counts.get(mode, 0) >= self.config.max_skill_rollouts:
                            print(f"[DEBUG SKILL] Reached max. rollout ({self.config.max_skill_rollouts}) in mode {mode.id} --> Saturate")
                            self._saturated_skill_modes.add(mode)

                    if valid_next_modes:
                        reached_modes.update(valid_next_modes)
                        
                        # Sync sampling pool!
                        if len(reached_modes) != len(self.sorted_reached_modes):
                            if not reached_terminal_mode:
                                self.sorted_reached_modes = list(
                                    sorted(reached_modes, key=lambda m: m.id)
                                )
                else:
                    failed_attempts += 1
                    print(f"[DEBUG SKILL] Failed attempt ({failed_attempts})")
                
                # Refresh pool to exclude newly exhausted skill modes
                mode_subset_to_sample = self._get_valid_modes_to_sample(exclude_modes=modes_rolled_out_this_batch)
                
                # Skip the normal random geometric sampling for this iteration
                # That way skill modes get exactly 1 skill-based transition and 0 geometric transitions
                continue 

            # Step 2C: if no skill, sample a transition configuration in that mode
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
                failed_attempts += 1 
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
                        mode_subset_to_sample = self._get_valid_modes_to_sample()
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
                    failed_attempts += 1
                    continue
            else: # Graph rejected transition node (collision)
                # self.env.C.view(True)
                failed_attempts += 1
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
                    mode_subset_to_sample = self._get_valid_modes_to_sample()

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
            transition_batch_size=effective_uniform_transition_batch_size,
            cost=current_best_cost,
            reached_modes=reached_modes,
        )

        # Filter modes (no uniform & informed sampling in skill modes -> should only contain skill transitions and skill nodes!) 
        non_skill_modes = [m for m in reached_modes if not self._mode_has_skill(m)]

        # Step 3: add regular nodes (not transitions) within modes
        print("Sampling uniform")
        new_states, required_attempts_this_batch = self._sample_valid_uniform_batch(
            g,
            batch_size=effective_uniform_batch_size,
            cost=current_best_cost,
            # non_skill_modes=None,
            non_skill_modes=non_skill_modes, # Don't sample in skill modes
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
            (self.config.try_informed_sampling or self.config.try_informed_transitions) #and
            # non_skill_modes # TODO check
        ):
            # Interpolate the current best path
            interpolated_path = interpolate_path(current_best_path)

            # Sample configurations near current best path 
            if self.config.try_informed_sampling:
                print("Generating informed samples")
                new_informed_states = informed_sampler.generate_samples(
                    non_skill_modes, # list(reached_modes), # Don't sample in skill modes
                    self.config.informed_batch_size,
                    interpolated_path,
                    try_direct_sampling=self.config.try_direct_informed_sampling,
                    g=g,
                )
                # Add those informed new states to the graph 
                g.add_states(new_informed_states)

                print(f"Adding {len(new_informed_states)} informed samples")

            # Sample transition nodes near mode transitions in current path
            if self.config.try_informed_transitions:
                print("Generating informed transitions")
                new_informed_transitions = informed_sampler.generate_transitions(
                    non_skill_modes, #list(reached_modes), Transition nodes are NOT fine in skill nodes
                    self.config.informed_transition_batch_size,
                    interpolated_path,
                    g=g,
                )
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

        # TODO DEBUG remove
        # self.env.C.view(True)
        print("[DEBUG ENV] Limits shape:", self.env.limits.shape)
        print("[DEBUG ENV] Limits:\n", self.env.limits)

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

                # # TODO DEBUG remove
                # skill_states = [s for s in path if s.is_skill_waypoint]
                # print(f"[DEBUG OPT] New path has {len(skill_states)} skill waypoints")
                
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

                # TODO DBUG
                for mode_obj in reached_modes:
                    if self._mode_has_skill(mode_obj):
                        trans = graph.transition_nodes.get(mode_obj, [])
                        skill_trans = [t for t in trans if t.state.is_skill_waypoint]
                        non_skill_trans = [t for t in trans if not t.state.is_skill_waypoint]
                        print(f"[DEBUG PRE-A*] Skill mode {mode_obj.id}: {len(trans)} transitions ({len(skill_trans)} skill, {len(non_skill_trans)} NON-skill)")

                # TODO DEBUG
                print(f"[DEBUG SEARCH] Modes in graph: {[m.task_ids for m in graph.nodes.keys()]}")
                print(f"[DEBUG SEARCH] Goal nodes: {len(graph.goal_nodes)}")
                print(f"[DEBUG SEARCH] Skill chains: {[(m.task_ids, len(v)) for m,v in graph.skill_chain_nodes.items()]}")

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
                            # TODO DEBUG
                            skill_states = [s for s in path if s.is_skill_waypoint]
                            skill_modes_in_path = set(s.mode.id for s in path if self._mode_has_skill(s.mode))
                            print()
                            print(f"[DEBUG A*] New path found")
                            print(f"[DEBUG A*] Path length: {len(path)}, cost: {new_path_cost}")
                            print(f"[DEBUG A*] Skill waypoints in path: {len(skill_states)}")
                            print(f"[DEBUG A*] Skill modes traversed: {skill_modes_in_path}")
                            for m_id in skill_modes_in_path:
                                states_in_mode = [s for s in path if s.mode.id == m_id]
                                skill_states_in_mode = [s for s in states_in_mode if s.is_skill_waypoint]
                                print(f"[DEBUG A*] -- Mode {m_id}: {len(states_in_mode)} total states, {len(skill_states_in_mode)} skill wp")
                            print()

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
                                
                                # TODO DEBUG
                                print()
                                print(f"[DEBUG PRE-SHORTCUT] Before shortcutting")
                                for m_id in skill_modes_in_path:
                                    states_in_mode = [s for s in path if s.mode.id == m_id]
                                    sw = [s for s in states_in_mode if s.is_skill_waypoint]
                                    print(f"[DEBUG PRE-SHORTCUT] Mode {m_id}: {len(states_in_mode)} states, {len(sw)} skill wp")
                                print()

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

                                print(f"[DEBUG SHORTCUTPATH]: length before = {len(shortcut_path)}")

                                # TODO DEBUG
                                print()
                                print(f"[DEBUG POST-SHORTCUT] After shortcutting, before remove_interpolated")
                                print(f"[DEBUG POST-SHORTCUT] Path length: {len(shortcut_path)}")
                                for m_id in skill_modes_in_path:
                                    states_in_mode = [s for s in shortcut_path if s.mode.id == m_id]
                                    sw = [s for s in states_in_mode if s.is_skill_waypoint]
                                    non_sw = [s for s in states_in_mode if not s.is_skill_waypoint]
                                    print(f"[DEBUG POST-SHORTCUT] Mode {m_id}: {len(states_in_mode)} states, {len(sw)} skill wp, {len(non_sw)} non-skill")
                                print()

                                # # TODO (Liam) DON'T COMMENT OUT! "Needed", because it removes interpolated points used during shortcutting
                                # Remove interpolated points (just used for collision check)
                                shortcut_path = shortcutting.remove_interpolated_nodes(
                                    shortcut_path
                                )

                                print(f"[DEBUG SHORTCUTPATH]: length after = {len(shortcut_path)}")
                                
                                print()
                                print(f"[DEBUG POST-REMOVAL] After remove_interpolated_nodes")
                                print(f"[DEBUG POST-REMOVAL] Path length: {len(shortcut_path)}")
                                for m_id in skill_modes_in_path:
                                    states_in_mode = [s for s in shortcut_path if s.mode.id == m_id]
                                    sw = [s for s in states_in_mode if s.is_skill_waypoint]
                                    non_sw = [s for s in states_in_mode if not s.is_skill_waypoint]
                                    print(f"[DEBUG POST-REMOVAL] Mode {m_id}: {len(states_in_mode)} states, {len(sw)} skill wp, {len(non_sw)} non-skill")
                                print()

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

                                    # Check shortcutted path (& handle mode changes)
                                    # Update graph
                                    current_skill_states = []

                                    for i in range(len(interpolated_path)):
                                        s = interpolated_path[i]

                                        if not self.env.is_collision_free(s.q, s.mode):
                                            print(f"[DEBUG COLLISION] Index {i}, mode {s.mode.id} tasks: {s.mode.task_ids}, is_skill_wp: {s.is_skill_waypoint}")
                                            print(f"[DEBUG COLLISION] q = {s.q.state()}")
                                            continue

                                        is_mode_boundary = (
                                            i < len(interpolated_path) - 1
                                            and interpolated_path[i].mode != interpolated_path[i + 1].mode
                                        )

                                        if self._mode_has_skill(s.mode): 
                                        # if s.is_skill_waypoint:
                                            # SKILL MODE: don't add nodes directly to graph, use add:skill_path!
                                            # Collect the state
                                            current_skill_states.append(s)

                                            # When the skill mode ends, add the collected current_skill_states  
                                            if is_mode_boundary:
                                                next_mode = interpolated_path[i + 1].mode
                                                
                                                # Handles hiding, neighbors, and whitelisting for skill nodes
                                                graph.add_skill_path(states=current_skill_states, valid_next_modes=[next_mode])
                                                skill_mode = current_skill_states[0].mode
                                                print(f"[DEBUG GRAPH] Skill Mode {skill_mode} | Main Nodes: {len(graph.nodes.get(skill_mode, []))} | Hidden Skill Nodes: {len(graph.skill_chain_nodes.get(skill_mode, []))}")
                                                
                                                # TODO (Liam) consider removing old skill chain & transition node?
                                                
                                                # Reset for the next potential skill
                                                current_skill_states = []

                                        else:
                                            # NON-SKILL MODE: standard prm behaviour
                                            s.is_skill_waypoint = False # To be sure # TODO (Liam) probably not necessary..
                                            
                                            if is_mode_boundary:
                                                graph.add_transition_nodes([(s.q, s.mode, [interpolated_path[i + 1].mode])])
                                            else:
                                                graph.add_states([s])

                                    # Edge case: if a shortcutted skill path ends exactly on the goal without a boundary trigger
                                    if current_skill_states:
                                        graph.add_skill_path(states=current_skill_states, valid_next_modes=None)

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
                
                # TODO DEBUG remove
                skill_states = [s for s in path if s.is_skill_waypoint]
                print(f"[DEBUG OPT] New path has {len(skill_states)} skill waypoints")

                break # Return first feasible path (if goal is not to optimize path..)

        if len(costs) > 0:
            costs.append(costs[-1])
            times.append(time.time() - start_time)

        info = {"costs": costs, "times": times, "paths": all_paths}

        return current_best_path, info
