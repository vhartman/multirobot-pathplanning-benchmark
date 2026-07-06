import numpy as np
import random
import math
import time
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass

from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseProblem,
    Mode,
    State, 
    Task
)
from multi_robot_multi_goal_planning.problems.core.configuration import (
    Configuration,
    batch_config_dist,
)
from multi_robot_multi_goal_planning.problems.util import interpolate_path, path_cost
from multi_robot_multi_goal_planning.planners import shortcutting
from .baseplanner import BasePlanner
from .mode_validation import ModeValidation
from .sampling_informed import InformedSampling
from .termination_conditions import PlannerTerminationCondition

from multi_robot_multi_goal_planning.problems.skills import (
    BaseDeterministicTimedSkill,
    BaseStochasticTimedSkill
)

# =====================================================================
# Config and Data Structures
# =====================================================================
@dataclass
class RRTStochasticSkillsConfig:
    """
    Hyperparameters for the multi-modal RRT with skills
    """
    # -----------------------------------------------------------------
    # RRT* CORE PARAMETERS
    # -----------------------------------------------------------------
    
    extension_strategy: str = "connect"                 # "linear" | "connect"
    p_goal: float = 0.1
    is_bidirectional: bool = False
    distance_metric: str = "max_euclidean"
    with_noise: bool = False

    # -----------------------------------------------------------------
    # EXTEND NON-SKILL MODES PARAMETERS
    # -----------------------------------------------------------------
    
    # LINEAR
    step_size_strategy: str = "sqrt_d"                  # "constant" | "scaled" | "sqrt_d_scaled" | "sqrt_d" | "sqrt_d_robots"
    step_size: float = 1                                # Constant
    step_size_factor: float = 0.1                       # Dynamic step size tuning factor

    # CONNECT
    eta_step: float = 0.1
    connect_max_steps: int = 30
    init_connect_target_policy: str = "transition"      # "transition" | "all"
    opt_connect_target_policy: str = "all"              # "transition" | "all"
    init_connect_add_all_nodes: bool = False
    opt_connect_add_all_nodes: bool = True              # For rewiring
    connect_target_policy: Optional[str] = None         # override
    connect_add_all_nodes: Optional[bool] = None        # override

    # MODE SAMPLING
    mode_sampling_type: str = "uniform"                 # "uniform" | "greedy" | "frontier"
    init_mode_sampling_type: str = "frontier"
    p_greedy: float = 0.98
    p_frontier: float = 0.98
    with_mode_validation: bool = False                  # Geometric pre-check on mode (blacklist_modes) # TODO
    
    # -----------------------------------------------------------------
    # EXTEND SKILL MODES PARAMETERS
    # -----------------------------------------------------------------
    
    skill_expansion_strategy: str = "kinodynamic"       # "single_step" | "kinodynamic"
    kinodynamic_steps: int = 5                          # Only for kinodynamic strategy 
    inactive_steering_mode: str = "concurrent"          # "freeze" | "concurrent"
    inactive_max_vel: float = 2.0                       # TODO define value, units,...
    inactive_transition_source: str = "uniform_random"  # "uniform_random" | "random_tree"

    # -----------------------------------------------------------------
    # STOCHASTIC SKILL PARAMETERS (conservative nominal + inflated tube)
    # -----------------------------------------------------------------

    tube_rollouts: int = 200                            # MC rollouts to estimate the uncertainty tube
    tube_quantile: float = 0.95                         # Per-step deviation quantile for the tube radius (1.0 = worst case)
    tube_margin_scale: float = 1.0                      # Scales the tube radii (0.0 disables inflation)

    # -----------------------------------------------------------------
    # RRT* OPTIMIZATION PARAMETERS
    # -----------------------------------------------------------------
    
    # REWIRING
    use_rrt_star: bool = True
    rewire_after_first_solution: bool = True 
    rewire_neighbor_strategy: str = "radius"            # "radius" | "k_nearest"
    rewire_radius_max: float = 1.0
    rewire_k_constant: Optional[float] = 10             # float | None uses the sufficient k-nearest RRT* paper constant
    gamma_rrtstar: float = 0.0

    # INFORMED SAMPLING
    try_informed_sampling: bool = True
    locally_informed_sampling: bool = True
    informed_batch_size: int = 300                      # Used in batch-mode sampling
    informed_transition_batch_size: int = 100

    # PATH POST-PROCESSING (SHORTCUTTING)
    try_shortcutting: bool = True
    shortcutting_mode: str = "round_robin"
    periodic_shortcutting_iters: int = 500
    final_shortcutting_iters: int = 1000
    shortcutting_interpolation_resolution: float = 0.1
    shortcut_period_iters: int = 500
    sync_shortcut_to_tree: bool = True    

@dataclass
class SkillEdge:
    """
    Stores the intermediate waypoints of a "kinodynamic" skill edge
    
    """
    waypoints: np.ndarray
    t_norms: np.ndarray

class Node:
    """
    Represents a single state in the multi-modal tree
    """
    def __init__(self, state: State, parent: Optional['Node'] = None):
        self.state = state
        self.parent = parent
        self.children: List['Node'] = []
        self.cost: float = 0.0
        self.cost_to_parent: float = 0.0

        # Flags for skills and transitions
        self.is_skill_waypoint: bool = getattr(state, "is_skill_waypoint", False)
        self.skill_steps: Dict[str, int] = dict(getattr(state, "skill_steps", {}))
        self.skill_edge: Optional['SkillEdge'] = None # Kinodynamic only

class Subtree:
    """
    Manages nodes and vectorized data for a specific mode
    """
    def __init__(self, mode: Mode, robot_dims: int, initial_capacity: int = 10000): 
        self.mode = mode
        self.nodes: List[Node] = []
        self.node_to_idx: Dict[int, int] = {}
        self.batch_q: np.ndarray = np.zeros((initial_capacity, robot_dims))
        self.size = 0

    def add_node(self, node: Node):
        """
        Adds a node to the subtree and updates the vectorized batch for NN search 
        """
        # If full, double capacity
        if self.size >= self.batch_q.shape[0]:
            new_batch = np.zeros((2*self.batch_q.shape[0], self.batch_q.shape[1]))
            new_batch[:self.size] = self.batch_q
            self.batch_q = new_batch

        # Add node to batch
        self.node_to_idx[id(node)] = self.size
        self.batch_q[self.size] = node.state.q.state()
        self.nodes.append(node)
        self.size += 1

    def get_near(self, q: Configuration, radius: float, metric: str = "max_euclidean") -> List[Tuple[int, float]]:
        """
        Returns (index, dist) for all nodes within radius
        """
        if self.size == 0:
            return []
        
        dists = batch_config_dist(q, self.batch_q[:self.size], metric)
        indices = np.where(dists < radius)[0]
        return [(int(i), float(dists[i])) for i in indices]

    def get_nearest(self, q_target: Configuration, metric: str = "max_euclidean") -> Tuple[Node, float]:
        """
        Finds the nearest node in this subtree to the target configuration
        """
        if self.size == 0:
            return None, float('inf')
        
        dists = batch_config_dist(q_target, self.batch_q[:self.size], metric)
        idx = np.argmin(dists)
        return self.nodes[idx], float(dists[idx])
    
class MultiModalTree:
    """
    Collection of subtrees, one per mode
    """
    def __init__(self, env: BaseProblem):
        self.env = env
        self.subtrees: Dict[Mode, Subtree] = {} 
        self.root: Node = None # TODO optional
        self.robot_dims = sum(env.robot_dims.values())

    def add_subtree(self, mode: Mode):
        """
        Adds a new subtree to the multi-modal tree
        """
        if mode not in self.subtrees:
            self.subtrees[mode] = Subtree(mode, self.robot_dims)


# =====================================================================
# CURRENT TODOS
# =====================================================================
"""
CURRENT TODOS
# RRT
# TODO [o] in _sample_mode add different mode sampling strategies like PRM (for now uniform)
# TODO [ ] in _sample_transition_config add reached_terminal_mode like PRM?
# TODO [ ] differentiate between goal bias and transition bias?
# TODO [ ] in _sample_transition_config use a smarter approach than random config sampling for inactive robots

# Improvements
# TODO [ ] blacklisting
# TODO [ ] exploit transition nodes already found, just like _sample_goal is doing
# TODO [ ] informed sampling in skill modes (inactive-DOF-only sampler. otherwise sampler wastes effort computing and validating a FULL-config sample when only inactive part matters..)
# TODO [ ] detect cost improvement from rewiring without waiting for periodic check (I think that might be what makes planner_rrtstar good..)
# TODO [ ] tune shortcutting_iters (too frequent -> tree small changes, waste of time / too infrequent -> misses improvements from rewiring)
# TODO [ ] track best transition nodes (lowest cost) in some transition registry so cheaper terminal candidate can be found via another transition or rewiring
# TODO [ ] tune hyperparams
# TODO [ ] shortcuts self.best_path but not a freshly extracted path from self.solution:node after rewiring..
# TODO [ ] add node pruning on sync sc-path to tree (e.g., snapping to existing node if distance < threshold?)
# TODO [ ] add node pruning to rrt* in general (e.g., some cost-based branch pruning: once rrt* finds initial path, use c_best as upper bound, and prune node + children if c(stars->x)+h(x->goal) > c_best

# RRT*
# TODO [ ] (later) rewiring in skill modes (inactive parts)
# TODO [ ] in _find_best_parent, seeding best_parent = n_near needs edge collision check? 
# TODO [ ] add bidirectional (BRRT*) in non skill modes (check first if old BIRRT* really is faster)

# GENERAL
# TODO [ ] p_transition for mode specific goal AND p_goal for terminal goal?
# TODO [ ] in subtree.get_near() should we limit to k_nearest?
"""

# =====================================================================
# Main Planner Class
# =====================================================================
class RRTStochasticSkills(BasePlanner):
    """
    Core multi-modal RRT* planner supporting deterministic skill integration,
    concurrent inactive robot steering, and optimization strategies. Expansion 
    is handled via customizable strategies in both non-skill (e.g., connect, linear)
    and skill modes (e.g., kinodynamic, single_step)
    """

    # =====================================================================
    # Initialization
    # =====================================================================

    def __init__(self, env: BaseProblem, config: RRTStochasticSkillsConfig):
        self.env = env
        self.config = config
        self.tree = MultiModalTree(env)

        self.mode_validation = ModeValidation(self.env, self.config.with_mode_validation, self.config.with_noise)
        self.reached_modes: List[Mode] = []

        self.start_time = 0.0
        self.solution_node: Node = None

        # Post-shortcut best path
        self.best_path: List[State] = None
        self.best_cost: float = float("inf")
        
        # Informed sampling (init in _initialize_planner)
        self.informed_sampler: InformedSampling = None
        self.informed_path: List[State] = None 
        self.improvement_count: int = 0 # For periodic shortcutting
        self._refresh_phase_params()

        # Uncertainty tubes for stochastic skills (task name -> (nominal traj, per-step radii))
        self._skill_tubes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def _refresh_phase_params(self):
        """
        Applies the active phase settings switching between initialization and
        optimization phases once a solution is found
        """
        c = self.config
        is_init = self.solution_node is None

        self._active_mode_sampling_type = c.init_mode_sampling_type if is_init else c.mode_sampling_type

        self._active_connect_target_policy = c.connect_target_policy if c.connect_target_policy is not None else (
            c.init_connect_target_policy if is_init else c.opt_connect_target_policy
        )

        self._active_connect_add_all_nodes = c.connect_add_all_nodes if c.connect_add_all_nodes is not None else (
            c.init_connect_add_all_nodes if is_init else c.opt_connect_add_all_nodes
        )
    
    def _initialize_planner(self):
        """
        Sets up the start node and initial mode. Computes dynamic step sizes, initializes
        RRT* volume estimates, prepares the informed sampler, and more..
        """
        if self.tree.root is not None:
            return # Already initialized

        # Dynamic step size
        if self.config.extension_strategy == "linear":
            self.eta = self._compute_dynamic_eta()
        elif self.config.extension_strategy == "connect":
            self.eta = self.config.eta_step
        else:
            raise ValueError(f"Unknown extension_strategy: {self.config.extension_strategy}")

        # Mode
        start_mode = self.env.get_start_mode()
        self.reached_modes.append(start_mode)
        self.tree.add_subtree(start_mode)

        # Node
        start_node = Node(State(self.env.get_start_pos(), start_mode))
        start_node.cost = 0.0
        self.tree.root = start_node
        self.tree.subtrees[start_mode].add_node(start_node)

        # Registry of all discovered terminal candidates
        self.terminal_nodes: List[Node] = [] # For _periodic_improve to re-extract from cheapest

        # RRT* gamma
        self.valid_samples = 0
        self.total_samples = 0
        if self.config.use_rrt_star:
            self.mu_X_total = float(np.prod(self.env.limits[1] - self.env.limits[0]))
            self._set_gamma_rrt_star(mu_X_free=self.mu_X_total) # Initial approximation (will get updated)

        # Informed sampler (used after first solution)
        if self.config.try_informed_sampling:
            self.informed_sampler = InformedSampling(self.env, "sampling_based", self.config.locally_informed_sampling)

    def _init_debug_counters(self):
        """
        # NOTE: GENERATED WITH GEMINI
        Initializes all debug counters
        """
        self._dbg_goal_bias_attempt = 0
        self._dbg_goal_bias_success = 0
        self._dbg_informed_attempt = 0
        self._dbg_informed_success = 0
        self._dbg_informed_trans_attempt = 0
        self._dbg_informed_trans_success = 0
        self._dbg_snap_events = 0
        self._dbg_validate_fail = 0
        self._dbg_is_trans_true = 0
        self._dbg_get_next_empty = 0
        self._dbg_seed_coll_fail = 0
        self._dbg_seed_added = 0
        self._dbg_min_nn_dist = float("inf")

        self._dbg_w_rewires = 0
        self._dbg_w_best_parent_swaps = 0
        self._dbg_w_near_size_sum = 0
        self._dbg_w_near_size_count = 0
        self._dbg_w_shortcut_hits = 0
        self._dbg_last_r_n = 0.0

        self._dbg_kino_edges = 0

    # =====================================================================
    # Main Planning Loop
    # =====================================================================

    def plan(self, ptc: PlannerTerminationCondition, optimize: bool = False):
        """
        Main planning loop that iteratively samples targets, extends the multi-modal tree, 
        processes mode transitions, and applies RRT* rewiring and shortcutting optimizations
        """
        self.start_time = time.time()
        self._initialize_planner()

        iterations = 0
        costs = []
        times = []

        # DEBUG prints (init)
        self._init_debug_counters()

        while not ptc.should_terminate(iterations, time.time() - self.start_time):
            iterations += 1

            # DEBUG prints
            if iterations % 500 == 0:
                self._print_debug(iterations)
                
            # 1. Sample mode
            mode = self._sample_mode()

            # 2. Sample target
            q_target, is_uniform = self._sample_target(mode)

            # 3. Nearest neighbor
            n_near, dist = self.tree.subtrees[mode].get_nearest(q_target, self.config.distance_metric)
            if dist < self._dbg_min_nn_dist:
                  self._dbg_min_nn_dist = dist

            # 4. Steer (linear or skill based)
            skill_tasks = self._get_active_skill_tasks(mode)
            new_nodes = self._expand(n_near, q_target, mode, skill_tasks, is_uniform)
            
            if not new_nodes:
                continue

            # 5. Handle transitions and rewiring
            terminal_node = None
            for n_new in new_nodes:
                next_mode_seeds = self._check_transitions(n_new)

                # RRT* rewire
                if self._should_rewire() and not n_new.is_skill_waypoint and not skill_tasks:
                    self._rewire(n_new, mode)

                terminal_node = self._get_terminal_node(n_new, next_mode_seeds)
                if terminal_node is not None:
                    break
            
            # 6. Check if we reached the terminal goal
            if terminal_node is not None:
                print("[RRT DONE] self.env.done() is TRUE")
                self._record_solution(costs, times, node=terminal_node)

                if not optimize:
                    break # Stop after first solution
            
            # 7. Periodic re-extraction (only in optimize mode, after first solution)
            if optimize and self.solution_node is not None and iterations % self.config.shortcut_period_iters == 0:
                self._periodic_improve(costs, times)

        # Final rescan first to capture any unrecorded rewiring gains
        if optimize and self.solution_node is not None:
            best_term = self._get_best_terminal()
            if best_term is not None:
                tree_path = self._extract_path(best_term)
                tree_cost = path_cost(tree_path, self.env.batch_config_cost)
                self._record_solution(costs, times, path=tree_path, node=best_term, cost=tree_cost)

        # Final shortcut
        if self.config.try_shortcutting and self.best_path is not None:
            sc_path = self._shortcut(self.best_path, self.config.final_shortcutting_iters)
            sc_cost = path_cost(sc_path, self.env.batch_config_cost) if sc_path and len(sc_path) > 1 else float("inf")
            shortcut_improved = self._record_solution(costs, times, path=sc_path, cost=sc_cost)
            if shortcut_improved:
                print(f"[RRT FINAL SHORTCUT] Improved cost to {self.best_cost:.3f}")

                if self.config.sync_shortcut_to_tree:
                    self._sync_shortcut_to_tree(sc_path)
                    best_term = self._get_best_terminal()
                    if best_term is not None:
                        self._record_solution(costs, times, node=best_term)

        # Return
        path = self.best_path
        info = {
            "costs": costs, 
            "times": times, 
            "paths": [path] if path else [],
            "skill_tubes": self._skill_tubes
        }
        return path, info

    def _record_solution(self, costs: List[float], times: List[float], 
                         path: List[State] = None, node: Node = None, cost: Optional[float] = None) -> bool:
        """
        Tracks best_cost and best_path from a candidate path or node
        Returns True when best cost got updated
        """
        if path is None and node is not None:
            path = self._extract_path(node)
        if path is None or len(path) < 2:
            return False

        new_cost = cost if cost is not None else path_cost(path, self.env.batch_config_cost)
        if new_cost >= self.best_cost - 1e-8:
            return False

        # Update tree pointer ONLY when improvement came from a tree node
        if node is not None:
            self._set_solution_node(node)

        self.best_cost = new_cost
        self.best_path = list(path)
        self._update_informed_path()
        costs.append(self.best_cost)
        times.append(time.time() - self.start_time)
        return True

    def _periodic_improve(self, costs: List[float], times: List[float]):
        """
        Periodic improvement:
        1. Always re-extract the cheapest terminal tree path so rewiring-only
            improvements are visible.
        2. If shortcutting is enabled, shortcut that fresh tree path, not the
            previous best_path.
        """
        # 1. Re-extract from cheapest terminal
        best_terminal = self._get_best_terminal()
        if best_terminal is None:
            return

        tree_path = self._extract_path(best_terminal)
        if self._record_solution(costs, times, path=tree_path, node=best_terminal):
            print(f"[RRT TREE REWIRE] Improved cost to {self.best_cost:.3f}")

        if not self.config.try_shortcutting or self.best_path is None or len(self.best_path) < 2:
            return

        # 2. Shortcut 
        sc_path = self._shortcut(self.best_path, self.config.periodic_shortcutting_iters)

        if self._record_solution(costs, times, path=sc_path):
            self.improvement_count += 1
            print(f"[RRT SHORTCUT #{self.improvement_count}] cost={self.best_cost:.3f}")

            if self.config.sync_shortcut_to_tree:
                self._sync_shortcut_to_tree(sc_path)
                best_terminal = self._get_best_terminal()
                if best_terminal is not None:
                    self._record_solution(costs, times, node=best_terminal)

    def _print_debug(self, iterations: int):
        """
        # NOTE: GENERATED WITH GEMINI
        Prints periodic performance telemetry and resets window counters.
        """
        nodes = sum(s.size for s in self.tree.subtrees.values())
        tag = "RRT*" if self.config.use_rrt_star else "RRT"
        sol_cost = self.solution_node.cost if self.solution_node is not None else float("inf")
        near_avg = (self._dbg_w_near_size_sum / self._dbg_w_near_size_count 
                    if self._dbg_w_near_size_count > 0 else 0.0)

        rewire_status = "ON" if self._should_rewire() else "OFF"
        informed_status = "ON" if (self.informed_path is not None and self.config.try_informed_sampling) else "OFF"
        connect_policy = self._active_connect_target_policy
        connect_add_all = self._active_connect_add_all_nodes

        print(
            f"[{tag}] it={iterations} nodes={nodes} modes={len(self.reached_modes)} "
            f"best={self.best_cost:.3f} sol={sol_cost:.3f} "
            f"etaMacro={self.eta:.2f} etaStep={self.config.eta_step:.2f} "
            f"rMax={self.config.rewire_radius_max:.2f} "
            f"rewire={rewire_status} informed={informed_status}\n"
            f"connect={connect_policy}/addAll={connect_add_all}\n"
            f"       | w: rewires={self._dbg_w_rewires} "
            f"r_n={self._dbg_last_r_n} "
            f"bestP={self._dbg_w_best_parent_swaps} "
            f"nearAvg={near_avg:.1f} "
            f"sCut={self._dbg_w_shortcut_hits}\n"
            f"       | c: snap={self._dbg_snap_events} "
            f"vfail={self._dbg_validate_fail} "
            f"gb={self._dbg_goal_bias_success}/{self._dbg_goal_bias_attempt} "
            f"inf={self._dbg_informed_success}/{self._dbg_informed_attempt} "                        
            f"infT={self._dbg_informed_trans_success}/{self._dbg_informed_trans_attempt} "  
            f"isT={self._dbg_is_trans_true} "
            f"nextE={self._dbg_get_next_empty} "
            f"sCF={self._dbg_seed_coll_fail} sAdd={self._dbg_seed_added} "
            f"kinoEdges={self._dbg_kino_edges} "
            f"impr={self.improvement_count}"
        )
        # Reset window counters
        self._dbg_w_rewires = 0
        self._dbg_w_best_parent_swaps = 0
        self._dbg_w_near_size_sum = 0
        self._dbg_w_near_size_count = 0
        self._dbg_w_rewire_extracts = 0
        self._dbg_w_shortcut_hits = 0
        self._dbg_kino_edges = 0

    # =====================================================================
    # Sampling
    # =====================================================================

    def _sample_mode(self) -> Mode:
        """
        Selects which mode to expand next based on the selected strategy
        - "uniform": pick uniformly from reached_modes
        - "greedy": p_greedy to newest, else uniform
        - "frontier": p_frontier split across modes without outgoing transitions, 
                      remainder split across others by inverse node count 
        
        NOTE: frontier implementation from composite_prm_planner
        """
        if len(self.reached_modes) == 1:
            return self.reached_modes[0]
        
        strategy = self._active_mode_sampling_type
        
        # Greedy strategy
        if strategy == "greedy":
            if random.random() < self.config.p_greedy:
                return self.reached_modes[-1]
            return random.choice(self.reached_modes)
            
        # Frontier strategy (from prm implementation)
        if strategy == "frontier":
            total_nodes = sum(self.tree.subtrees[m].size for m in self.reached_modes)
            p_frontier = self.config.p_frontier
            p_remaining = 1.0 - p_frontier

            frontier_modes = []
            remaining_modes = []
            sample_counts = {}
            inv_prob = []

            # Check for frontier modes in the list of so far discovered modes
            for m in self.reached_modes:
                sample_count = self.tree.subtrees[m].size
                sample_counts[m] = sample_count
                if not m.next_modes:
                    frontier_modes.append(m)
                else:
                    remaining_modes.append(m)
                    inv_prob.append(1 - (sample_count / total_nodes))

            # Special case: only frontier mode should be sampled
            if p_frontier == 1.0:
                if not frontier_modes:
                    frontier_modes = self.reached_modes
                if len(frontier_modes) > 0:
                    p = [1 / len(frontier_modes)] * len(frontier_modes)
                    return random.choices(frontier_modes, weights=p, k=1)[0]
                else:
                    return random.choice(self.reached_modes)

            # Fallback to uniform if either partition is empty
            if not remaining_modes or not frontier_modes:
                return random.choice(self.reached_modes)
            if total_nodes == 0:
                return random.choice(self.reached_modes)

            # Build probability distribution
            total_inverse = sum(
                1 - (sample_counts[m] / total_nodes) for m in remaining_modes
            )
            if total_inverse == 0:
                return random.choice(self.reached_modes)

            sorted_reached_modes = frontier_modes + remaining_modes
            p = [p_frontier / len(frontier_modes)] * len(frontier_modes)
            inv_prob = np.array(inv_prob)
            p.extend((inv_prob / total_inverse) * p_remaining)

            return random.choices(sorted_reached_modes, weights=p, k=1)[0]

        # Uniform strategy (also fallback)
        return random.choice(self.reached_modes)

    def _sample_target(self, mode: Mode) -> tuple[Configuration, bool]:
        """
        Samples a random configuration or goal bias / transition
        """
        # Goal/transition bias
        if random.random() < self.config.p_goal:
            self._dbg_goal_bias_attempt += 1
            q_target = self._sample_transition_config(mode)
            if q_target is not None:
                self._dbg_goal_bias_success += 1
                return q_target, False # Not uniform
        
        # Informed sampling (after first solution, for non-skill AND skill-modes)
        if (self.config.try_informed_sampling
            and self.informed_path is not None):
            q_informed = self._sample_informed(mode)
            if q_informed is not None:
                self._dbg_informed_success += 1
                # NOTE: In skill modes, only the INACTIVE parts of q_informed will be used
                # (active overridden by skill.step). Basically provides informed inactive sampling
                # in skill mode.
                return q_informed, False
        
        # Uniform sampling (fallback)
        return self.env.sample_config_uniform_in_limits(), True # Is uniform

    def _sample_transition_config(self, mode: Mode) -> Configuration:
        """
        Samples a configuration satisfying the current mode's transition requirements,
        using either the informed transition sampler or a bounded rejection sampling approach
        """
        # Informed transition sampling (after first solution, non-skill mode, not terminal)
        if (self.config.try_informed_sampling
            and self.informed_sampler is not None
            and self.informed_path is not None
            and not self._get_active_skill_tasks(mode)
            and not self.env.is_terminal_mode(mode)):
            self._dbg_informed_trans_attempt += 1
            q = self.informed_sampler.generate_transitions(
                self._non_skill_reached_modes(),
                self.config.informed_transition_batch_size,
                self.informed_path,
                active_mode=mode
            )
            if q is not None and q != []:
                self._dbg_informed_trans_success += 1
                return q # Else uniform below

        # Transition sampling with selectable inactive-DOF source
        max_attempts = 1000
        iters = 0
        for _ in range(max_attempts):
            iters += 1
            # Get task that needs to be completed to switch mode
            next_task_ids = self.mode_validation.get_valid_next_ids(mode)
            if not next_task_ids and not self.env.is_terminal_mode(mode):
                return None

            # Sample the goal for the robots finishing their task
            active_task = self.env.get_active_task(mode, next_task_ids)
            
            # We cannot geometrically sample the end state of a skill or a task without a goal
            if getattr(active_task, "skill", None) is not None or active_task.goal is None:
                continue

            constrained_robots = active_task.robots
            goal = active_task.goal.sample(mode)

            # Build transition candidate with constrained robots at goal
            q = self.env.sample_config_uniform_in_limits()

            # Apply constraints
            end_idx = 0
            for i, robot in enumerate(self.env.robots):
                if robot in constrained_robots:
                    # Overwrite with goal
                    dim = self.env.robot_dims[robot]
                    q[i] = goal[end_idx : end_idx + dim]
                    end_idx += dim

            active_indices = np.array(self._get_active_subspace_indices([active_task]), dtype=int)
            source_node = self._select_inactive_source_node(mode, q, active_indices)
            if source_node is not None:
                constrained_set = set(constrained_robots)
                for i, robot in enumerate(self.env.robots):
                    if robot in constrained_set:
                        continue
                    q[i] = source_node.state.q.robot_state(i).copy()

            # Validate that the constrained config is collision-free in current mode
            if self.env.is_collision_free(q, mode):
                return q

        return None

    def _select_inactive_source_node(
        self,
        mode: Mode,
        q_active_goal: Configuration,
        active_indices: np.ndarray,
    ) -> Optional[Node]:
        """
        Selects a source node for inactive robot DOFs used during transition sampling
        """
        source = self.config.inactive_transition_source
        if source == "uniform_random":
            return None

        subtree = self.tree.subtrees.get(mode)
        if subtree is None or subtree.size == 0:
            return None

        if source == "random_tree":
            idx = random.randint(0, subtree.size - 1)
            return subtree.nodes[idx]

        return None

    def _sample_informed(self, mode: Mode) -> Optional[Configuration]:
        """
        Gets single informed sample for the given mode using the InformedSampler
        """
        if self.informed_sampler is None or self.informed_path is None:
            return None

        self._dbg_informed_attempt += 1

        q = self.informed_sampler.generate_samples( # Returns one config in "sampling_based" mode
            self.reached_modes,
            self.config.informed_batch_size,
            self.informed_path,
            active_mode=mode,
            # try_direct_sampling=???, # TODO check what that is (used in prm)
        )

        if q is None or q == []:
            return None
        return q

    def _update_informed_path(self):
        """
        Builds the interpolated path that the informed sampler uses as reference
        (focal points for the PHS ellipsoid). Called after every cost improvement.
        """
        if self.best_path is not None and len(self.best_path) > 1:
            self.informed_path = interpolate_path(self.best_path)
        else:
            self.informed_path = None

    # =====================================================================
    # Tree Expansion & Steering
    # =====================================================================

    def _expand(self, n_near: Node, q_target: Configuration, mode: Mode, skill_tasks, is_uniform: bool = True) -> List[Node]:
        """
        Routes the expansion and steering logic based on the mode type (skill vs. non-skill) 
        and the configured expansion strategy (e.g., linear, connect, kinodynamic)
        """
        # Strategy for expanding/steering in NON-skill-modes
        if not skill_tasks:
            if self.config.extension_strategy == "linear":
                return self._expand_linear(n_near, q_target, mode, is_uniform)
            if self.config.extension_strategy == "connect":
                return self._expand_connect(n_near, q_target, mode, is_uniform)
            raise ValueError(f"Unknown extension_strategy: {self.config.extension_strategy}")
        
        # Strategies for expanding/steering in skill-modes
        strategy = self.config.skill_expansion_strategy
        if strategy == "single_step":
            return self._expand_single_step(n_near, q_target, mode, skill_tasks)
        if strategy == "kinodynamic":
            return self._expand_kinodynamic(n_near, q_target, mode, skill_tasks)
        
        raise ValueError(f"Unknown skill_explansion_strategy: {strategy}")

    def _expand_linear(self, n_near: Node, q_target: Configuration, mode: Mode, is_uniform: bool) -> List[Node]:
        """
        Standard one-step RRT expansion for non-skill modes.
        """
        state_new = self._linear_steer(n_near, q_target, mode)
        if state_new is None:
            return []

        if not self._validate(state_new, n_near.state.q, is_skill=False, is_uniform=is_uniform):
            return []

        return [self._create_and_add_node(state_new, n_near, mode, is_skill=False)]

    def _linear_steer(self, n_near: Node, q_target: Configuration, mode: Mode):
        """
        Standard linear interpolation towards q_target
        """
        q_near_vec = n_near.state.q.state() # .state() returns NDArray
        q_target_vec = q_target.state()

        dist_array = batch_config_dist(n_near.state.q, [q_target], self.config.distance_metric)
        dist = dist_array.item()
        if dist < 1e-6:
            return None
        
        # Snap to target
        if dist <= self.eta: # TODO snapping to target (probably tries it all the time with big eta and fails in single_agent_bin_picking -> probably tries to connect all the time but edge in collision)
            q_new = q_target
            self._dbg_snap_events += 1
        else:
            # print(f"[DEBUG LINEAR STEER] linear")
            # step = min(dist, self.eta)
            q_new_vec = q_near_vec + self.eta * (q_target_vec - q_near_vec) / dist
            q_new = self.env.get_start_pos().from_flat(q_new_vec)

        # DEBUG
        # dbg_dists = batch_config_dist(n_near.state.q, [q_target, q_new], self.config.distance_metric)
        # for i, dist in enumerate(dbg_dists):
        #     print(f"[DEBUG LINEAR STEER DIST] distance {i} = {dbg_dists[i]}")

        # print(f"[DEBUG LINEAR STEER] dist = {dist:.4f}, eta = {self.eta:.4f}")
        return State(q_new, mode)

    def _steer_inactive(self, q_full: np.ndarray, q_target_vec: np.ndarray, active_indices: np.ndarray, dt: float) -> np.ndarray:
        """
        Concurrent inactive robot steering, bounded by inactive_max_vel * dt per robot.
        Maintains a straight-line path in the full C-space by scaling the entire direction
        vector based on the bottleneck robot
        """
        if self.config.inactive_steering_mode != "concurrent":
            return q_full.copy()

        direction = q_target_vec - q_full
        direction[active_indices] = 0.0

        # Find the maximum velocity required by any single robot
        max_robot_vel = 0.0
        end_idx = 0
        for robot in self.env.robots:
            dim = self.env.robot_dims[robot]
            robot_dir = direction[end_idx : end_idx + dim]
            robot_vel = np.linalg.norm(robot_dir) / dt
            if robot_vel > max_robot_vel:
                max_robot_vel = robot_vel
            end_idx += dim

        if max_robot_vel <= 1e-8:
            return q_full.copy()

        # Scale the full vector so the fastest robot moves exactly at inactive_max_vel
        scale = min(1.0, self.config.inactive_max_vel / max_robot_vel)
        
        return q_full + scale * direction

    def _expand_connect(self, n_near: Node, q_target: Configuration, mode: Mode, is_uniform: bool) -> List[Node]:
        """
        RRT-Connect-style extension: goes from n_near towards q_target in steps of eta_step

        Node insertion is phase-specific:
        1. add_all_nodes=False: only final node enters the tree
        - Fast for finding initial solutions but breaks RRT*-rewiring (long edges + small rewire radius).

        2. add_all_nodes=True: every intermediate step enters the tree
        - Required for asymptotic optimality with use_rrt_star=True.
        """
        q_target_vec = q_target.state().copy()
        q_curr_vec = n_near.state.q.state().copy()
        q_curr_cfg = n_near.state.q

        eta_step = self.eta

        target_policy = self._active_connect_target_policy
        if target_policy == "transition":
            is_transition_target = self.env.is_transition(q_target, mode)
            max_steps = self.config.connect_max_steps if is_transition_target else 1
        elif target_policy == "all":
            max_steps = self.config.connect_max_steps
        else:
            raise ValueError(f"Unknown connect_target_policy: {target_policy}")

        add_all = self._active_connect_add_all_nodes

        new_nodes: List[Node] = []
        n_parent = n_near # Parent for the next step (becomes previous step's node when add_all)
        progress = False
        reached_snap = False

        # 
        for _ in range(max_steps):
            dist = batch_config_dist(q_curr_cfg, [q_target], self.config.distance_metric).item()
            if dist < 1e-6:
                reached_snap = True
                break

            # Steer
            step = min(eta_step, dist)
            snap = step >= dist - 1e-9
            q_next_vec = q_target_vec.copy() if snap else q_curr_vec + step * (q_target_vec - q_curr_vec) / dist
            q_next_cfg = self.env.get_start_pos().from_flat(q_next_vec)

            # Validate
            if not self.env.is_collision_free(q_next_cfg, mode):
                if self.config.use_rrt_star:
                    self._update_cfree_estimate(was_valid=False, was_uniform=is_uniform)
                break

            if not self.env.is_edge_collision_free(q_curr_cfg, q_next_cfg, mode):
                break

            # Step accepted
            q_curr_vec = q_next_vec
            q_curr_cfg = q_next_cfg
            progress = True
            if self.config.use_rrt_star:
                self._update_cfree_estimate(was_valid=True, was_uniform=is_uniform)

            if add_all:
                # RRT*-connect: each step is a tree node with full ChooseParent treatment
                # _rewire is then applied per node in plan()'s post-expand loop
                state_step = State(q_curr_cfg, mode)
                n_step = self._create_and_add_node(state_step, n_parent, mode, is_skill=False)
                new_nodes.append(n_step)
                n_parent = n_step

            if snap:
                reached_snap = True
                break

        if reached_snap:
            self._dbg_snap_events += 1

        if not progress:
            return []

        if not add_all:
            # Satisficing connect: single node at the end of the chain
            state_new = State(q_curr_cfg, mode)
            new_nodes = [self._create_and_add_node(state_new, n_near, mode, is_skill=False)]


        # state_new = State(q_curr_cfg, mode)
        # return [self._create_and_add_node(state_new, n_near, mode, is_skill=False)]

        return new_nodes

    # =====================================================================
    # Skill Expansion
    # =====================================================================

    def _get_active_skill_tasks(self, mode: Mode):
        """
        Returns a list of all tasks with a skill that are currently active in this mode
        """
        if self.env.is_terminal_mode(mode):
            return []

        active_tasks = []
        seen = set()
        for task_id in mode.task_ids:
            if task_id in seen:
                continue
            seen.add(task_id)
            task = self.env.tasks[task_id]
            if getattr(task, "skill", None) is not None:
                active_tasks.append(task)
        return active_tasks

    def _get_active_subspace_indices(self, active_tasks) -> List[int]:
        """
        Returns indices for the robots involved in the active tasks
        """
        active_indices = []
        end_idx = 0
        active_robots = set()
        for task in active_tasks:
            active_robots.update(task.robots)
            
        for robot in self.env.robots:
            dim = self.env.robot_dims[robot]
            if robot in active_robots:
                active_indices.extend(range(end_idx, end_idx + dim))
            end_idx += dim
        return active_indices

    def _use_nominal_tube(self, skill) -> bool:
        """
        Determines if a skill should be executed using the robust Tube-RRT strategy
        """
        return (
            self.config.tube_margin_scale > 0.0
            and isinstance(skill, BaseStochasticTimedSkill)
        )
        
    def _get_skill_tube(self, skill_task, q_subspace: np.ndarray, skill_step: int):
        """
        Evaluates the skill execution starting at q_init and estimates an uncertainty tube 
        that is inflating (radially) the nominal trajectory based on Monte Carlo rollouts
        """
        key = skill_task.name

        # 1. Check cache if tube already calculated for this task
        if key in self._skill_tubes:
            return self._skill_tubes[key][1]
        
        skill = skill_task.skill

        # 2. Starting point
        if skill_step == 0:
            q_init = np.asarray(q_subspace)
        else: # TODO double check my logic...
            # If first request not at skill start (e.g., resumed mid-skill across a mode 
            # boundary before any step-0 expansion) -> use initiation config to compute full 
            # tube from step 0
            q_init = np.asarray(skill_task.initation_goal.sample(None))

        all_joints = self.env.get_joint_names()

        # Calculate exactly how many steps this skill takes
        n_steps = max(1, round(skill.duration / skill.dt))

        def _pad(traj: np.ndarray) -> np.ndarray:
            """
            If skill terminates early, pad trajectory repearing last config, so all rollouts have 
            same length for stacking and numpy operations
            # TODO: suggested by claude -> double check
            """
            if len(traj) < n_steps + 1:
                traj = np.vstack([traj, np.repeat(traj[-1:], n_steps + 1 - len(traj), axis=0)])
            return traj

        # 3. Monte Carlo rollouts: run N noisy executions
        rollouts = np.stack([
            _pad(skill.rollout(q_init, skill_task, all_joints, self.env, t0=0.0).trajectory)
            for _ in range(self.config.tube_rollouts)
        ], axis=0)

        # 4. Tube
        nominal = np.mean(rollouts, axis=0)
        deviations = np.linalg.norm(rollouts - nominal, axis=2)
        radii = np.quantile(deviations, self.config.tube_quantile, axis=0)

        # DEBUG
        print(f"[TUBE] task '{key}': {self.config.tube_rollouts} rollouts, radii min/max = {radii.min():.3f}/{radii.max():.3f}")
        
        # 5. Cache it
        self._skill_tubes[key] = (nominal, radii)

        return radii
    
    def _tube_margins_free(self, tube_tasks: List, tube_radii: Dict[str, np.ndarray], steps: Dict[str, int],
                           q_flat: np.ndarray, mode: Mode) -> bool:
        """
        Checks the inflated clearance for one composite skill waypoint, ensuring the inactive 
        robots don't pass through this active robot's inflated tube
        """
        for t in tube_tasks:
            # 1. Get pre calculated radius for this skill at current step
            radii = tube_radii[t.name]
            r = float(radii[min(steps[t.name], len(radii) - 1)])

            # 2. Add uncertainty from other concurrently active stochastic skills for each step
            r_other = sum(
                float(tube_radii[o.name][min(steps[o.name], len(tube_radii[o.name]) - 1)])
                for o in tube_tasks if o is not t
            )
            
            # 3. Scale by config parameter
            margin = (r + r_other) * self.config.tube_margin_scale
            
            # 4. Collision check with inflated geometry
            if not self.env.is_collision_free_with_margin(q_flat, mode, t.robots, margin):
                return False

        return True

    def _expand_single_step(self, n_near: Node, q_target: Configuration, mode: Mode, skill_tasks: List) -> List[Node]:
        """
        Rolls out multiple concurrent skills by one step, with optional concurrent steering for the inactive robots.
        Inactive robots motions are bounded by max_vel*dt
        """
        if not skill_tasks:
            return []
        dt = skill_tasks[0].skill.dt # TODO Assume same dt for all skills

        # 1. Get positions from all robots
        q_full = n_near.state.q.state().copy()
        q_target_vec = q_target.state().copy()

        # 2. Inactive robots
        active_indices = self._get_active_subspace_indices(skill_tasks)
        q_base = self._steer_inactive(q_full, q_target_vec, active_indices, dt)
        
        all_joints = self.env.get_joint_names()

        # 3. Stochastic tube preparation 
        tube_tasks = [t for t in skill_tasks if self._use_nominal_tube(t.skill)]
        tube_radii = {
            t.name: self._get_skill_tube(
                t, q_full[self._get_active_subspace_indices([t])],
                n_near.skill_steps.get(t.name, 0), q_full_flat=q_full, mode=mode,
            ) for t in tube_tasks
        }

        # 4. Active robots
        for skill_task in skill_tasks:
            skill = skill_task.skill
            task_name = skill_task.name
            task_indices = self._get_active_subspace_indices([skill_task])

            # Isolate just the active robot's joints
            q_subspace = q_full[task_indices]
            self.env.C.selectJoints(skill.joints)

            base_step = n_near.skill_steps.get(task_name, 0)

            if isinstance(skill, (BaseDeterministicTimedSkill, BaseStochasticTimedSkill)):
                n_steps = max(1, round(skill.duration / dt))
                
                # Avoid step past horizon
                if base_step >= n_steps:
                    self.env.C.selectJoints(all_joints)
                    return []
                t_norm = min((base_step + 1) / n_steps, 1.0)

                # Stochastic: don't call skill.step() -> adds random noise
                # Instead force robot to follow center of uncertainty tube
                if self._use_nominal_tube(skill):
                    nominal_traj, _ = self._skill_tubes[task_name]
                    q_subspace = nominal_traj[min(base_step + 1, len(nominal_traj) - 1)].copy()
                else:
                    # Deterministic skills just step normally
                    q_subspace_new = skill.step(t_norm, q_subspace, self.env)
            else: 
                q_subspace_new = skill.step(q_subspace, self.env)

            # Insert the active robot's new position back into the main array
            q_base[task_indices] = q_subspace_new

        self.env.C.selectJoints(all_joints)

        # 5. Assemble the new state
        q_new = self.env.get_start_pos().from_flat(q_base)
        
        # Keep track of how many steps each skill has taken so far
        new_skill_steps = dict(n_near.skill_steps)
        for skill_task in skill_tasks:
            new_skill_steps[skill_task.name] = new_skill_steps.get(skill_task.name, 0) + 1
            
        state_new = State(q_new, mode, is_skill_waypoint=True, skill_steps=new_skill_steps)
     
        # 6. Collision checking (checking if nominal center path hit anything)
        if not self._validate(state_new, n_near.state.q, is_skill=True):
            return []
        
        # If nominal safe, check inflated path
        if tube_tasks:
            steps = {t.name: n_near.skill_steps.get(t.name, 0) + 1 for t in tube_tasks}
            if not self._tube_margins_free(tube_tasks, tube_radii, steps, q_base, mode):
                return []

        # 7. Add to tree
        n_new = self._create_and_add_node(state_new, n_near, mode, is_skill=True)
        n_new.skill_steps = new_skill_steps

        return [n_new]
 
    def _expand_kinodynamic(self, n_near: Node, q_target: Configuration, mode: Mode, skill_tasks: List) -> List[Node]:
        """
        Rolls out multiple concurrent skills as one kinodynamic edge
        - All intermediate steps are collision checked during construction
        - Only the end node enters the subtree (for NN search)
        - Intermediate waypoints are stored in a SkillEdge on the end node
        """
        if not skill_tasks:
            return []
            
        dt = skill_tasks[0].skill.dt
        n_kino = self.config.kinodynamic_steps
        active_indices = self._get_active_subspace_indices(skill_tasks)

        q_curr = n_near.state.q.state().copy()
        q_target_vec = q_target.state().copy()

        waypoints = [q_curr.copy()]
        rollout_steps = n_kino
        skill_infos = []
        
        # 1. Setup: precompute bounds to make sure we don't try to unroll past the skill's end
        for task in skill_tasks:
            skill = task.skill
            base_step = n_near.skill_steps.get(task.name, 0)
            is_timed = isinstance(skill, (BaseDeterministicTimedSkill, BaseStochasticTimedSkill))
            n_total = max(1, round(skill.duration / dt)) if is_timed else 0
            
            if is_timed:
                if n_total - base_step <= 0: return []
                rollout_steps = min(rollout_steps, n_total - base_step)
                
            skill_infos.append({
                'skill': skill,
                'indices': self._get_active_subspace_indices([task]),
                'base_step': base_step,
                'is_timed': is_timed,
                'n_total': n_total
            })

        t_norms_list = [0.0]

        # Fetch stochastic tubes for all active stochastic tasks
        tube_tasks = [t for t in skill_tasks if self._use_nominal_tube(t.skill)]
        tube_radii = {
            t.name: self._get_skill_tube(
                t, q_curr[self._get_active_subspace_indices([t])],
                n_near.skill_steps.get(t.name, 0), q_full_flat=q_curr, mode=mode,
            ) for t in tube_tasks
        }

        skill_done = False
        actual_steps = 0
        all_joints = self.env.get_joint_names()

        # 2. The unrolling loop
        for i in range(1, rollout_steps + 1):

            # Move inactive robots
            q_next = self._steer_inactive(q_curr, q_target_vec, active_indices, dt)
            
            # Move active robots
            for info in skill_infos:
                skill = info['skill']
                q_sub = q_curr[info['indices']]
                self.env.C.selectJoints(skill.joints)
                
                if info['is_timed']:
                    t_norm = min((info['base_step'] + i) / info['n_total'], 1.0)
                    
                    # Force stochastic skill to follow nominal baseline
                    if self._use_nominal_tube(skill):
                        nominal_traj, _ = self._skill_tubes[info['name']]
                        q_sub_new = nominal_traj[min(info['base_step'] + i, len(nominal_traj) - 1)].copy()
                    else:
                        q_sub_new = skill.step(t_norm, q_sub, self.env)
                    if skill.done(t_norm, q_sub_new, self.env): skill_done = True
                else:
                    q_sub_new = skill.step(q_sub, self.env)
                    if skill.done(q_sub_new, self.env): skill_done = True
                
                q_next[info['indices']] = q_sub_new
            
            self.env.C.selectJoints(all_joints)

            # 3. Intermediate collision checking
            q_next_cfg = self.env.get_start_pos().from_flat(q_next)
            q_curr_cfg = self.env.get_start_pos().from_flat(q_curr)
            
            state_next = State(q_next_cfg, mode)
            if not self._validate(state_next, q_curr_cfg, is_skill=True):
                break

            # Inflated stochastic collision check
            if tube_tasks:
                steps = {t.name: n_near.skill_steps.get(t.name, 0) + i for t in tube_tasks}
                if not self._tube_margins_free(tube_tasks, tube_radii, steps, q_next, mode):
                    break
            
            # Passed collision check -> record waypoint
            actual_steps = i
            waypoints.append(q_next.copy())
            t_norms_list.append(float(i))
            q_curr = q_next

            if skill_done:
                break # Skill finished early
        
        # 4. Final node creation
        if len(waypoints) < 2:
            return [] 
        
        q_end_cfg = self.env.get_start_pos().from_flat(waypoints[-1])
        
        # Dict for step tracking
        end_step_dict = dict(n_near.skill_steps)
        for skill_task in skill_tasks:
            end_step_dict[skill_task.name] = end_step_dict.get(skill_task.name, 0) + actual_steps
        state_new = State(q_end_cfg, mode, is_skill_waypoint=True, skill_steps=end_step_dict)

        # Put all itermediate waypoints into an Edge and compute true distance
        skill_edge = SkillEdge(waypoints=np.array(waypoints), t_norms=np.array(t_norms_list))
        edge_cost = self._skill_edge_cost(np.asarray(waypoints), mode)

        # Add single node to tree
        n_new = self._create_and_add_node(state_new, n_near, mode, is_skill=True, edge_cost_override=edge_cost)
        n_new.skill_steps = end_step_dict
        n_new.skill_edge = skill_edge
        
        return [n_new]

    def _skill_edge_cost(self, waypoints: np.ndarray, mode: Mode) -> float:
        """
        Computes true cost for kinodynamic edges instead of using straight-line parent-to-end-costs
        """ 
        if len(waypoints) < 2:
            return 0.0

        q_from_flat = self.env.get_start_pos().from_flat
        configs = [q_from_flat(q) for q in waypoints]
        return float(np.sum(self.env.batch_config_cost(configs[:-1], configs[1:])))

    # =====================================================================
    # Node Management & Transitions
    # =====================================================================

    def _validate(self, state_new: State, q_near: Configuration, is_skill: bool, is_uniform: bool = True) -> bool:
        """
        Performs geometric collision checks for both the node configuration and the edge connecting 
        it to its parent. Also updates the online c_free volume estimate for RRT*
        """
        # 1. Config check
        is_state_free = self.env.is_collision_free(state_new.q, state_new.mode)

        # 2. Update c_free self._update_cfree_estimate
        if self.config.use_rrt_star and not is_skill:
            self._update_cfree_estimate(was_valid=is_state_free, was_uniform=is_uniform)

        # 3. Failure based on config check 
        if not is_state_free:
            self._dbg_validate_fail += 1
            return False

        # 4. Edge check
        if not self.env.is_edge_collision_free(state_new.q, q_near, state_new.mode):
            self._dbg_validate_fail += 1
            return False

        return True

    def _create_and_add_node(self, state_new: State, n_near: Node, mode: Mode, is_skill: bool = False, edge_cost_override: Optional[float] = None) -> Node:
        """
        Handles node creation and addition, including RRT* parent optimization
        Optional edge_cost_override for kinodynamic skilledge to pass the true cost
        """
        if is_skill or not self._should_rewire():
            # Skill node or RRT* inactive -> always attach to n_near
            parent = n_near
            if edge_cost_override is not None:
                cost_to_parent = edge_cost_override
            else:
                cost_to_parent = self.env.config_cost(n_near.state.q, state_new.q)
            cost = n_near.cost + cost_to_parent
        else:
            # Non-skill node + RRT* active -> run RRT* choose parent
            parent, cost, cost_to_parent = self._find_best_parent(n_near, state_new.q, mode)

        # Create
        n_new = Node(state_new, parent=parent)
        n_new.cost = cost
        n_new.cost_to_parent = cost_to_parent

        # Skill node bookkeeping
        if is_skill:
            n_new.is_skill_waypoint = True
            n_new.state.is_skill_waypoint = True # TODO (for shortcutter.. change and only keep on node..?)

        # Add
        self.tree.subtrees[mode].add_node(n_new)
        n_new.parent.children.append(n_new)
        return n_new

    def _check_transitions(self, n_new: Node) -> List[Node]:
        """
        Generates the successor mode nodes (seeds) when a transition is detected. While "_is_mode_transition()"
        just checks if something finished, this function identifies exactly which tasks finished. It then asks
        the environment for the valid next modes and creates the starting nodes (seeds) for those modes
        """
        created_seeds: List[Node] = []

        # Broad check, if anything finished
        if not self._is_mode_transition(n_new):
            return created_seeds

        mode = n_new.state.mode
        self._dbg_is_trans_true += 1
        completed_task_ids = []
        skill_tasks = self._get_active_skill_tasks(mode)

        # Step 1: Identigy exactly which tasks have completed (who triggered the transition)
        # We loop through every robot's current task to figure out exactly what finished at this timestep
        for i, task_id in enumerate(mode.task_ids):
            task = self.env.tasks[task_id]
            q_concat = np.concatenate([n_new.state.q.robot_state(self.env.robots.index(r)) for r in task.robots])
            
            # Evaluate completion based on the task type (skill vs. geometric)
            if task in skill_tasks:
                # Skills don't have geometric goals, we rely on skill.done()
                if self._is_skill_done(n_new, task):
                    completed_task_ids.append(task_id)
            # Geometric tasks have goals (check if goal reached)
            elif task.goal is not None and task.goal.satisfies_constraints(q_concat, mode=mode, tolerance=1e-8):
                completed_task_ids.append(task_id)

        # Step 2: Request the valid next modes from the environment
        try:
            # Pass completed_task_ids since skills lack geometric goal
            next_modes = self.env.get_next_modes(n_new.state.q, mode, completed_task_ids=completed_task_ids)
        except ValueError:
            return created_seeds
        
        # Filter to allowed modes
        valid_next_modes = self.mode_validation.get_valid_modes(mode, list(next_modes))
        if not valid_next_modes:
            self._dbg_get_next_empty += 1
            return created_seeds

        # Step 3: Create the seed nodes for the newly reached modes
        for next_mode in valid_next_modes:
            # Check if configuration is actually collision-free in new mode's context
            if not self.env.is_collision_free(n_new.state.q, next_mode):
                self._dbg_seed_coll_fail += 1
                continue

            # Create the successor subtree treating transition nodes as start nodes of the next mode
            if next_mode not in self.reached_modes:
                self.reached_modes.append(next_mode)
                self.tree.add_subtree(next_mode)

            seed_state = State(n_new.state.q, next_mode)
            seed_node = Node(seed_state, parent=n_new)

            # Costs inherited as they are, across the mode boundary
            seed_node.cost = n_new.cost
            seed_node.cost_to_parent = 0.0

            # Pass skill states if skills continue (mid-skill switch)
            current_active_tasks = self._get_active_skill_tasks(mode)
            next_active_tasks = self._get_active_skill_tasks(next_mode)
            
            next_task_names = set(t.name for t in next_active_tasks)
            
            continuing_skills = False
            for task in current_active_tasks:
                if task.name in next_task_names:
                    seed_node.skill_steps[task.name] = n_new.skill_steps.get(task.name, 0)
                    continuing_skills = True
            
            if continuing_skills:
                seed_node.is_skill_waypoint = True

            self.tree.subtrees[next_mode].add_node(seed_node)
            n_new.children.append(seed_node)
            created_seeds.append(seed_node)
            
            # RRT* optimization
            if self._should_rewire() and not seed_node.is_skill_waypoint and not next_active_tasks:
                self._rewire(seed_node, next_mode)

            self._dbg_seed_added += 1

        return created_seeds

    def _is_mode_transition(self, node: Node) -> bool:
        """
        Determines if a node has successfully reached the transition criteria for its current mode.
        It verifies if ANY active robot has finished its skill, or if any inactive robot has 
        reached its geometric goal
        """
        mode = node.state.mode

        if self.env.is_terminal_mode(mode):
            return False
        
        skill_tasks = self._get_active_skill_tasks(mode)

        # 1. Did ANY active skill finish executing?
        for skill_task in skill_tasks:
            if self._is_skill_done(node, skill_task):
                return True
        
        # 2. Did any inactive robot geometrically reach its goal?
        return self.env.is_transition(node.state.q, mode)

    def _is_skill_done(self, node: Node, task: Task) -> bool:
        """
        Evaluates if an active skill has finished its execution
        """
        skill = task.skill
        q_subspace = node.state.q.state()[self._get_active_subspace_indices([task])]

        # Check if the skill is timed vs. untimed
        if isinstance(skill, (BaseDeterministicTimedSkill, BaseStochasticTimedSkill)):
            n_steps = max(1, round(skill.duration / skill.dt))
            base_step = node.skill_steps.get(task.name, 0)
            t_norm = min(base_step / n_steps, 1.0)
            return skill.done(t_norm, q_subspace, self.env)
        return skill.done(q_subspace, self.env)

    def _non_skill_reached_modes(self) -> List[Mode]:
        """
        Returns a list of all currently reached modes that do not involve an active skill task
        """
        return [m for m in self.reached_modes if not self._get_active_skill_tasks(m)]

    def _get_terminal_node(self, n_new: Node, next_mode_seeds: List[Node]) -> Optional[Node]:
        """
        Registers newly-discovered terminal candidates in self.terminal_nodes and returns one
        (if any) for immediate solution recording
        """
        found: Optional[Node] = None

        if self.env.done(n_new.state.q, n_new.state.mode):
            if n_new not in self.terminal_nodes:
                self.terminal_nodes.append(n_new)
                found = n_new

        for seed in next_mode_seeds:
            if self.env.done(seed.state.q, seed.state.mode):
                if seed not in self.terminal_nodes:
                    self.terminal_nodes.append(seed)
                if found is None:
                    found = seed

        return found

    def _get_best_terminal(self) -> Optional[Node]:
        """
        Returns the current lowest-cost terminal candidate (None if not discovered yet)
        """
        if not self.terminal_nodes:
            return None
        return min(self.terminal_nodes, key=lambda n: n.cost)

    def _set_solution_node(self, node: Node):
        """
        Records a tree-backed solution and switches phase settings once
        """
        if self.solution_node is None:
            self.solution_node = node
            self._refresh_phase_params()
        else:
            self.solution_node = node

    # =====================================================================
    # Path Extraction & Shortcutting
    # =====================================================================

    def _extract_path(self, node: Node) -> List[State]:
        """
        Traces back from the given node to the root, resolving kinodynamic
        skill edges into individual waypoints along the way
        """
        nodes = []
        curr = node
        while curr: 
            nodes.append(curr)
            curr = curr.parent
        nodes.reverse()

        # Build path (inserting SkillEdge intermediates where present)
        path = []
        for n in nodes:
            if n.skill_edge is not None:
                for idx, wp in enumerate(n.skill_edge.waypoints[1:]):
                    q_wp = self.env.get_start_pos().from_flat(wp)
                    wp_skill_steps = dict(n.parent.skill_steps) if n.parent else {}
                    for skill_task_name in n.skill_steps:
                        wp_skill_steps[skill_task_name] = wp_skill_steps.get(skill_task_name, 0) + idx + 1
                    path.append(State(q_wp, n.state.mode, is_skill_waypoint=True, skill_steps=wp_skill_steps))
            else:
                path.append(n.state)
        return path

    def _tube_state_validator(self, state: State) -> bool:
        """
        Tube clearance check for shortcutting: inactive robots may be moved by the shortcutter 
        through skill modes, but must stay outside the stochastic skills uncertainty tubes
        """
        # NOTE: changed in shortcutting.py to have skill_steps as State attribute
        # Figure out which active skills need tube validation in this mode (tube_tasks)
        # Extract radii and steps from tube_tasks
        # We can tehn collision check the inactive robots against the inflated rubes by calling tube_margins_free
        raise NotImplementedError

    def _shortcut(self, path: List[State], shortcutting_iters: int) -> List[State]:
        """
        Post-processes a path with robot_mode_shortcut
        Skill segments are protected
        """
        # TODO (stochastic) changes for shortcutting
        # Add state validator (or callable) to arguments
        # In shortcutting.py make sure proposed_shortcut has skill_steps + when doing CC for path, 
        # check that new shortcut states are not in tube (call state_validator)
        shortcut_path, _ = shortcutting.robot_mode_shortcut(
            self.env, path, shortcutting_iters,
            resolution=self.env.collision_resolution,
            tolerance=self.env.collision_tolerance,
            robot_choice=self.config.shortcutting_mode,
            interpolation_resolution=self.config.shortcutting_interpolation_resolution
        )

        # Remove interpolated points used in shortcutting (collision check)
        return shortcutting.remove_interpolated_nodes(shortcut_path)

    def _sync_shortcut_to_tree(self, shortcut_path: List[State]) -> Optional[Node]:
        """
        NOTE (old): Inserts a shortcutted path back into the tree as a fresh connected chain.
        Near-duplicate snapping to avoid exploding tree 
        
        NOTE (new): Inserts a shortcutted path as an RRT*-aware chain (like in rrtstar_base)
        - Same mode steps: _find_best_parent + _rewire
        - Mode transition steps: force natural parent + _check_transitions on parent + _rewire
        - Skill waypoints / skill modes: force natural parent, no rewire 
        """
        if not shortcut_path or len(shortcut_path) < 2:
            return None

        parent_node = self.tree.root
        rewire_on = self._should_rewire()

        for state in shortcut_path[1:]:
            is_skill = getattr(state, "is_skill_waypoint", False)
            new_mode = state.mode
            mode_changed = (new_mode != parent_node.state.mode)
            is_skill_mode = bool(self._get_active_skill_tasks(new_mode))

            # 1) Parent selection (only find_best_parent if not mode change, not skill mode, parent already in subtree)
            subtree = self.tree.subtrees[new_mode]
            can_use_rrt_star = (rewire_on and not mode_changed and not is_skill_mode and id(parent_node) in subtree.node_to_idx)

            if can_use_rrt_star:
                best_parent, best_cost, best_cost_to_parent = self._find_best_parent(parent_node, state.q, new_mode)
            else:
                best_parent = parent_node
                best_cost_to_parent = self.env.config_cost(parent_node.state.q, state.q)
                best_cost = parent_node.cost + best_cost_to_parent

            # 2) Create and attach
            new_node = Node(state, parent=best_parent)
            new_node.cost_to_parent = best_cost_to_parent
            new_node.cost = best_cost
            
            # Preserve skill flags if this state was marked as one
            if getattr(state, "is_skill_waypoint", False):
                new_node.is_skill_waypoint = True
   
            best_parent.children.append(new_node)
            subtree.add_node(new_node)

            # 3) Mode-boundary logic
            if mode_changed:
                self._check_transitions(new_node)

            # 4) Global improvement (rewire around new node)
            if rewire_on and not is_skill_mode:
                self._rewire(new_node, new_mode)

            parent_node = new_node

        # Terminal handling 
        if self.env.done(parent_node.state.q, parent_node.state.mode):
            if parent_node not in self.terminal_nodes:
                self.terminal_nodes.append(parent_node)
            self._set_solution_node(parent_node)
            return parent_node
        return None

    # =====================================================================
    # RRT* / Optimization (Rewiring)
    # =====================================================================

    def _set_gamma_rrt_star(self, mu_X_free: float = None):
        """
        RRT*: asymptotic optimality constant
        
        gamma_rrtstar = (2(1+1/d))^(1/d) * (mu(X_free)/zeta_d)^(1/d)
        d: dimensionality of state space
        mu(X_free): Lebesque measure (volume) of obstacle-free search space
        zeta: volume of unit ball in d-dimensional space
        """
        self.d = sum(self.env.robot_dims.values())
        zeta_d = math.pi ** (self.d / 2) / (math.gamma(self.d / 2 + 1))
        self.gamma_rrt_star = (2 * (1 + 1 / self.d)) ** (1 / self.d) * (mu_X_free / zeta_d) ** (1 / self.d)

    def _compute_dynamic_eta(self):
        """
        Dynamically compute step size eta based based on environment boundaries and chosen strategy
        """
        strategy = self.config.step_size_strategy

        if strategy == "constant":
            return self.config.step_size

        elif strategy == "sqrt_d":
            d = sum(self.env.robot_dims.values())
            return math.sqrt(d)

        elif strategy == "sqrt_d_robots":
            d = sum(self.env.robot_dims.values())
            num_robots = len(self.env.robots)
            return math.sqrt(d / num_robots)

        robot_diameters = []
        offset = 0
        for robot in self.env.robots:
            dim = self.env.robot_dims[robot]
            lo = self.env.limits[0, offset : offset + dim]
            hi = self.env.limits[1, offset : offset + dim]
            robot_diameters.append(np.linalg.norm(hi - lo))
            offset += dim

        workspace_diameter = max(robot_diameters)

        if strategy == "scaled":
            return self.config.step_size_factor * workspace_diameter

        elif strategy == "sqrt_d_scaled":
            d = sum(self.env.robot_dims.values())
            return workspace_diameter / math.sqrt(d)

        else:
            raise ValueError(f"Unknown step_size_strategy: {strategy}")

    def _update_cfree_estimate(self, was_valid: bool, was_uniform: bool = True): # TODO (to be tested)
        """
        Approximates c_free by tracking current sample validity (online) 
        """
        # Only track samples from a global uniform distribution
        if not was_uniform:
            return
        
        # Track valid and total samples
        self.total_samples += 1
        if was_valid:
            self.valid_samples += 1

        # Compute mu_X_free and gamma after 100 samples & update periodically
        if self.total_samples % 200 == 0 and self.total_samples > 100:
            frac_free = self.valid_samples / self.total_samples
            mu_X_free = frac_free * self.mu_X_total
            self._set_gamma_rrt_star(mu_X_free)

    def _compute_rewiring_radius(self, n: int):
        """
        RRT*: shrinking ball radius
        r_n = min(gamma_rrtstar * (logn/n)^(1/d), rewire_radius_max)
        n: number of nodes in tree
        d: dimensionality of state space
        gamma_rrtstar: asymptotic optimality constant
        rewire_radius_max: early-iteration cap, intentionally independent of eta_step?
        """
        if n <= 1:
            return self.config.rewire_radius_max
        r_n = self.gamma_rrt_star * (math.log(n) / n) ** (1.0 / self.d)
        return min(r_n, self.config.rewire_radius_max)

    def _compute_rewiring_k(self, n: int) -> int:
        """
        k-nearest RRT*: k(n) = ceil(k_RRT* log(n))
        If no constant is configured, use the sufficient bound from the paper
        """
        if n <= 1:
            return 1

        k_constant = self.config.rewire_k_constant
        if k_constant is None:
            k_constant = (2 ** (self.d + 1)) * math.e * (1.0 + 1.0 / self.d)

        return max(1, min(n, int(math.ceil(k_constant * math.log(n)))))

    def _near_indices(self, dists: np.ndarray, radius: float) -> np.ndarray:
        """
        Select RRT* neighbors either by radius or by k-nearest candidate count
        """
        strategy = self.config.rewire_neighbor_strategy
        # Select neighbors based on radius
        if strategy == "radius":
            return np.nonzero(dists < radius)[0]

        # Select neighbors based on k-nearest
        if strategy == "k_nearest":
            k = self._compute_rewiring_k(len(dists))
            if k >= len(dists):
                return np.arange(len(dists), dtype=np.int64)
            return np.argpartition(dists, k - 1)[:k]

        raise ValueError(f"Unknown rewire_neighbor_strategy: {strategy}")

    def _near_batch_costs(
        self,
        q: Configuration,
        mode: Mode,
        radius: float,
        force_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns near-set indices, costs from q to each near node, and current node costs
        Cost computation is batched to avoid one config_cost() call per neighbor
        """
        subtree = self.tree.subtrees[mode] 

        # Compute distances to all nodes in subtree (vectorized) and get indices within radius
        dists = batch_config_dist(q, subtree.batch_q[:subtree.size], self.config.distance_metric)
        near_indices = self._near_indices(dists, radius)

        # Ensure n_near is included
        if force_idx is not None and not np.any(near_indices == force_idx):
            near_indices = np.insert(near_indices, 0, force_idx)
        if len(near_indices) == 0:
            empty = np.empty(0, dtype=np.float64)
            return near_indices, empty, empty

        # Compute costs for all near edges in one batch
        near_batch = subtree.batch_q[near_indices]
        edge_costs = np.asarray(self.env.batch_config_cost(q, near_batch), dtype=np.float64)
        
        # Retrieve current costs of near nodes
        near_costs = np.array(
            [subtree.nodes[int(idx)].cost for idx in near_indices],
            dtype=np.float64,
        )
        return near_indices, edge_costs, near_costs

    def _find_best_parent(self, n_near: Node, q_new: Configuration, mode: Mode):
        """
        RRT*: find the lowest-cost parent from the near set
        Returns (best_parent, best_cost, cost_to_parent)
        """
        subtree = self.tree.subtrees[mode]
        r_n = self._compute_rewiring_radius(subtree.size)
        n_near_idx = subtree.node_to_idx[id(n_near)]
        
        # Get candidates and batch costs
        near_indices, edge_costs, near_costs = self._near_batch_costs(
            q_new, mode, r_n, force_idx=n_near_idx,
        )

        # DEBUG
        self._dbg_last_r_n = r_n
        self._dbg_w_near_size_sum += len(near_indices)
        self._dbg_w_near_size_count += 1

        # Initialize with n_near
        best_parent = n_near # TODO needs collision check?
        fallback_pos = int(np.where(near_indices == n_near_idx)[0][0])
        best_cost_to_parent = float(edge_costs[fallback_pos])
        best_cost = n_near.cost + best_cost_to_parent

        # Check if any neighbor provides a lower cost parent
        potential_costs = near_costs + edge_costs
        improvement_mask = potential_costs < best_cost # Vectorized filtering

        if np.any(improvement_mask):
            # Sort by cost for efficient search
            sorted_positions = np.where(improvement_mask)[0][
                np.argsort(potential_costs[improvement_mask])
            ]
            for pos in sorted_positions:
                candidate = subtree.nodes[int(near_indices[pos])]
                if candidate is n_near or candidate.is_skill_waypoint:
                    continue
                if self.env.is_edge_collision_free(candidate.state.q, q_new, mode):
                    best_parent = candidate
                    best_cost_to_parent = float(edge_costs[pos])
                    best_cost = float(potential_costs[pos])
                    break

        if best_parent is not n_near:
            self._dbg_w_best_parent_swaps += 1

        return best_parent, best_cost, best_cost_to_parent

    def _rewire(self, n_new: Node, mode: Mode):
        """
        RRT*: rewire neighbors if n_new provides cheaper path
        """
        subtree = self.tree.subtrees[mode]
        r_n = self._compute_rewiring_radius(subtree.size)
        # Get candidates for rewiring
        near_indices, edge_costs, near_costs = self._near_batch_costs(n_new.state.q, mode, r_n)

        # DEBUG
        self._dbg_last_r_n = r_n
        self._dbg_w_near_size_sum += len(near_indices)
        self._dbg_w_near_size_count += 1

        # Check if n_new improves cost for any neighbor
        potential_costs = n_new.cost + edge_costs
        improvement_mask = potential_costs < near_costs

        for pos in np.nonzero(improvement_mask)[0]:
            n_near = subtree.nodes[int(near_indices[pos])]
            if n_near is n_new or n_near is n_new.parent or n_near.is_skill_waypoint:
                continue
            # rewire if edge is collision free
            if self.env.is_edge_collision_free(n_new.state.q, n_near.state.q, mode):
                # Detach from old parent
                old_parent = n_near.parent
                if old_parent is not None:
                    old_parent.children.remove(n_near)

                # Connect to n_new
                n_near.parent = n_new
                n_new.children.append(n_near)
                n_near.cost_to_parent = float(edge_costs[pos])
                n_near.cost = float(potential_costs[pos])                
                self._propagate_cost_improvement(n_near)
                self._dbg_w_rewires += 1

    def _should_rewire(self) -> bool:
        """
        Determines if RRT* should rewire or not:
        - If use_rrt_star = False: never rewire
        - If rewire_after_first_solution = True: only rewire after first solution found
        - Otherwise: always rewire when use_rrt_star = True
        """
        if not self.config.use_rrt_star:
            return False
        if self.config.rewire_after_first_solution:
            return self.solution_node is not None
        return True

    def _propagate_cost_improvement(self, node: Node):
        """
        RRT*: propagate cost changes down the tree after rewiring
        """
        stack = list(node.children)
        while stack:
            child = stack.pop()
            child.cost = child.parent.cost + child.cost_to_parent
            stack.extend(child.children)
