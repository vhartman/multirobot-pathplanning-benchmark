"""
Multi-modal RRT(*) with skills

(OUTDATED NOTES..)
# TODO update notes

--------------------------------------
DATA STRUCTURES:
--------------------------------------
Node: 
- id
- state State (q, mode)
- parent Node, children list[Node]
- cost (cumulative cost from root)
- is_skill_waypoint, is_transition, skill_step, 

MultiModalTree: 
- root Node
- subtrees (dict map between mode -> subtree)
- batch configs (vectorized distance computation)
- transition nodes, entry nodes, skill chain nodes

KD-Trees (why not?):
- Becomes less efficient -> curse of dimensionality
- RRT* (rewiring) -> tree constantly changing -> rebuild KD-tree every iteration -> expensive
- Vectorized NumPy -> very fast -> vectorized brute-force search faster than complex data structure? 

--------------------------------------
PLANNER LOOP 
--------------------------------------
A) Sample (mode, q_rand):
    - sample mode from reached_modes
    - sample config q_rand (uniform in C-spacem, goal bias, informed after first solution?)

B) Find nearest:
    - efficient search to get closest node in subtree
    - RRT* shrinking radius?

C) Steer:
    - normal mode: standard linear interpolation from n_nearest towards q_rand by step_size -> q_new
    - skill mode: calls skill' skill.step() -> q_new (naturally incorporate skill's dynamics and stochasticity into the tree) 

D) Validate
    - collision check q_new and edge n_nearest -> q_new
    -- no lazy check like PRM as would break tree otherwise..

E) Add
    - add n_new to subtree[mode]
    - check if n_new satisfies env.is_transition(q_new) or if skill.done(q_new) 
    -- mark n_new as is_transition, compute valid next_modes, create successor node in next mode (n_new -> entry_node)

F) Rewire (RRT*)
    - check for each node (in radius) if going through q_new is cheaper -> change parent + update/propagate cost 
    - no rewiring in skill chains

G) Extract path
    - no A*
    - trace back (parent) from goal to start

H) 

--------------------------------------
CHALLENGES / OPEN QUESTIONS
--------------------------------------
Mode transition logic:
    - need to check for each new node if it satisifies (by chance) the transition conditions..
    - no direct transition node sampling
    - okay when using goal bias?
    -- sample transition configs (like in PRM), upfront or in loop, and steer towards them with p_goal? 
    -- probably even better with RRT connect? 

Skill integration:
    - skill chain of nodes -> sequence by forcing parents.. 
    - 

Collision checking:
    - no way around, needs to be done every step
    - config check before edge check  
    - how come it's that faster than PRM with ?

RRT* cost propagation:
    - after rewiring, cost propagation with all the skill chains, crossing modes.. expensive
    - shrinking radius -> fallback with k-nearest?

Bidirectional tree:
    - how to handle skill rollouts in backward tree.. stop at skill mode boundary?
    - precompute skill and lookup?
    - how to merge in multi-modal setting


--------------------------------------
ROADMAP
--------------------------------------
A) RRT basic (no skills): MultiModalTree, geometric transitions (normal mode -> normal mode), basic linear steering
B) RRT basic (with skills): update steer to handle skill.step(), transition with skill.done()  
C) RRT* (rewiring): with near() and rewire(), cost propagation
D) BRRT*: 
E) 
F) 

--------------------------------------
COMPARISON TO PRM
--------------------------------------
- simpler: tree grows naturally, no graph connectivity "issues", no A* search, (no need to sample explicitly transitions?), path extraction trivial
- harder: 

--------------------------------------
REUSABLE?
--------------------------------------
- Implement from scratch
- 

"""

import numpy as np
import random
import math
import time
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass

from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseProblem,
    Mode,
    State
)
from multi_robot_multi_goal_planning.problems.configuration import (
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
    BaseDeterministicTimedSkill
)

@dataclass
class RRTSkillsConfig:
    """
    Hyperparameters for the multi-modal RRT with skills
    """
    # RRT
    step_size_strategy: str = "sqrt_d" # "constant" | "scaled" | "sqrt_d_scaled" | "sqrt_d" | "sqrt_d_robots"
    step_size: float = 1 # Constant
    step_size_factor: float = 0.1 # Dynamic step size tuning factor

    p_goal: float = 0.3
    # p_terminal_goal: float = 0.1 

    # Mode sampling # TODO frontier implementation (from prm)
    mode_sampling_type: str = "frontier" # "uniform" | "greedy" | "frontier"
    p_newest_mode: float = 0.8
    p_frontier: float = 0.98
    
    distance_metric: str = "max_euclidean"
    with_mode_validation: bool = False # Geometric pre-check on mode (blacklist_modes) # TODO
    with_noise: bool = False

    # Skills # TODO full_rollout and kinodynamic
    skill_expansion_strategy: str = "single_step" # "single_step" | "kinodynamic" | "full_rollout" 
    kinodynamic_steps: int = 5 # Only for kinodynamic strategy 
    inactive_steering_mode: str = "concurrent" # "freeze" | "concurrent"
    inactive_max_vel: float = 2.0 # TODO define value, units,...

    # RRT*
    use_rrt_star: bool = True
    rewire_after_first_solution: bool = False 
    rewire_radius: float = 0.5 # TODO not used
    gamma_rrtstar: float = 0.0

    # BRRT*
    is_bidirectional: bool = False

    # Informed sampling
    try_informed_sampling: bool = True
    locally_informed_sampling: bool = False
    informed_batch_size: int = 300 # In "sampling_based" informed sampling, this is the budget, not the batch size
    informed_transition_batch_site: int = 100

    # Shortcutting (post-processing)
    try_shortcutting: bool = True
    shortcutting_mode: str = "round_robin"
    shortcutting_iters: int = 1000 #250
    shortcutting_interpolation_resolution: float = 0.1

@dataclass
class SkillEdge:
    """
    Stores the intermediate waypoints of a "kinodynamic" skill edge
    
    """
    waypoints: np.ndarray
    t_norms: np.ndarray
    # TODO

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
        self.is_skill_waypoint: bool = False
        self.skill_step: int = 0
        self.is_transition: bool = False
        self.skill_edge: Optional['SkillEdge'] = None # Kinodynamic only

class Subtree:
    """
    Manages nodes and vectorized data for a specific mode
    """
    def __init__(self, mode: Mode, robot_dims: int, initial_capacity: int = 10000): 
        self.mode = mode
        self.nodes: List[Node] = []
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

# OLD TODOS
"""
OLD TODOS:
# General 
# TODO [x] init the multi modal tree
# TODO [x] vectorized batch_q for fast nearest-neighbor lookups
# TODO [x] pre-allocate so append(node) -> O(1) instead of O(N)

# RRT
# TODO [x] add root to multi modal tree? optional?
# TODO [x] import batch_config_cost batch_config_dist or access with self.env?
# TODO [x] check in PRM transition cost=0.0?
# TODO [x] in _steer use config_cost(), restpecting distance_metric 
# TODO [x] in _check_transitions always create and insert seed node (even if mode already reached)
# TODO [x] in _initialize_planner add early return if self.tree.root is not None (not duplicated start mode/node) if plan() called again?
# TODO [x] global shortcutting

# SKILLS
# TODO [x] update _add_node to set skill flag
# TODO [x] update _check_transitions when skill is done
# TODO [x] differentiate between linear steer and skill steer
# TODO [x] in _steer add skills -> call skill.step()

"""

# CURRENT TODOS
"""
CURRENT TODOS
# RRT
# TODO [o] in _sample_mode add different mode sampling strategies like PRM (for now uniform)
# TODO [ ] in _sample_transition_config add reached_terminal_mode like PRM?
# TODO [ ] differentiate between goal bias and transition bias?
# TODO [x] add debug prints in the planning loop
# TODO [ ] in _steer add clippling of q_new to self.env.limits?
# TODO [ ] in plan when creating/adding a new node, we only update parents, what about children? 
# TODO [ ] in _sample_transition_config consider taking random node from tree for inactive instead of random sampling? (seems like tree struggles to grow with random sampling in certain envs -> yes, random sampling is the idea, but doesn't seem to be efficient -> maybe something else is the problem..)
# TODO [ ] in _check_transitions, config check really needed or if config ok in modeA -> ok in modeB?
# TODO [ ] in _sample_transition_config use a smarter approach than random config for inactive robots
# TODO [o] in _linear_steer fix the dist computation (accidentally used cost function..)
# TODO [o] in _initialize_planner compare using eta=sqrt(d) and eta=sqrt(d/#robots)

# Improvements
# TODO [ ] blacklisting
# TODO [ ] exploit transition nodes already found, just like _sample_goal is doing
# TODO [o] add informed sampling

# RRT-connect
# TODO [ ] in _steer instead of taking single step towards target, use "connect" approach by taking multiple steps until collision or target reached

# SKILLS
# TODO [o] add unified structure with 3 expansion modes for skill rollouts (full, single, kino)
# TODO [x] check self.dt with chosen dt in skills..
# TODO [o] implement kindodynamic with SkillEdge 
# TODO [ ] check deviation between actual skill x-steps and interpolation between ends of the SkillEdge
# TODO [o] skill edge cost correct computation 

# RRT*
# TODO [o] add rewiring (RRT*)
# TODO [o] compute gamma (RRT*) -> okay to approximate mu(Xfree)?
# TODO [o] dynamically define step size eta
# TODO [ ] in _find_best_parent, seeding best_parent = n_near needs edge collision check? 
# TODO [ ] add bidirectional (BRRT*)

# GENERAL
# TODO [o] how to do preallocation? allocate and then double?
# TODO [o] figure out how to add a new subtree and transition seeding at mode boundary
# TODO [x] add all relevant hyperparams to RRTSkillsConfig
# TODO [ ] p_transition for mode specific goal AND p_goal for terminal goal?
"""

class RRTSkills(BasePlanner):
    """
    RRT planner:
    - Step-based skill rollouts
    - RRT* rewiring (optional)
    - Bidirectional search (optional)
    """
    def __init__(self, env: BaseProblem, config: RRTSkillsConfig):
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

    def plan(self, ptc: PlannerTerminationCondition, optimize: bool = False):
        """
        Main planning loop
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
            skill_task = self._get_active_skill_task(mode)
            new_nodes = self._expand(n_near, q_target, mode, skill_task, is_uniform)
            
            if not new_nodes:
                continue

            # 5. Collision check 
            for n_new in new_nodes:
                self._check_transitions(n_new)

                # RRT* rewire                                                                                                
                if self._should_rewire() and not n_new.is_skill_waypoint:                                
                    self._rewire(n_new, mode)                                                                                

            # 6. Check if we reached the terminal goal 
            final_state = new_nodes[-1].state                                                                                                                                     
            if self.env.done(final_state.q, final_state.mode):
                print(f"[RRT DONE] self.env.done() is TRUE")
                improved = self._record_solution(costs, times, node=new_nodes[-1])

                if self.config.try_shortcutting:
                    sc = self._shortcut(self._extract_path(new_nodes[-1]))
                    improved = True                           
                    self.improvement_count += 1                                                  
                    print(f"[RRT SHORTCUT #{self.improvement_count}] Improved cost to {self.best_cost:.3f}") 

                if improved:
                    self._update_informed_path()

                if not optimize:
                    break # Stop after first solution
            
            # 7. Periodic re-extraction from tree (every 10 iterations to avoid overhead)
            # Improvement sources:
            # - Rewiring dropping solution_node.cost
            # - Informed samples enabling better shortcut 
            if optimize and self.solution_node is not None and iterations % 10 == 0:
                improved = self._record_solution(costs, times, node=self.solution_node)
                
                if self.config.try_shortcutting:
                    sc = self._shortcut(self._extract_path(self.solution_node))
                    if self._record_solution(costs, times, path=sc):
                        improved = True
                        self.improvement_count += 1
                        print(f"[RRT SHORTCUT #{self.improvement_count}] Improved cost to {self.best_cost:.3f}") 
                
                if improved:
                    self._update_informed_path()

        # Final shortcut
        if self.config.try_shortcutting and self.best_path is not None:
            sc = self._shortcut(self.best_path)
            if self._record_solution(costs, times, path=sc):
                self.improvement_count += 1
                print(f"[RRT FINAL SHORTCUT] Improved cost to {self.best_cost:.3f}")

        # Extract path and info
        if self.best_path is not None:
            path = self.best_path
        elif self.solution_node is not None:
            path = self._extract_path(self.solution_node)
        else: 
            path = None

        info = {
            "costs": costs, 
            "times": times, 
            "paths": [path] if path else []
        }
        return path, info

    def _initialize_planner(self):
        """
        Setup start node and initial mode
        """
        if self.tree.root is not None:
            return # Already initialized
        
        # Dynamic step size
        self.eta = self._compute_dynamic_eta()
        
        # Mode
        start_mode = self.env.get_start_mode()
        self.reached_modes.append(start_mode)
        self.tree.add_subtree(start_mode)
        
        # Node
        start_node = Node(State(self.env.get_start_pos(), start_mode))
        start_node.cost = 0.0
        self.tree.root = start_node
        self.tree.subtrees[start_mode].add_node(start_node)

        # RRT* gamma
        self.valid_samples = 0
        self.total_samples = 0
        if self.config.use_rrt_star:
            self.mu_X_total = float(np.prod(self.env.limits[1] - self.env.limits[0]))
            self._set_gamma_rrt_star(mu_X_free=self.mu_X_total) # Initial approximation (will get updated)

        # Informed sampler (used after first solution)
        if self.config.try_informed_sampling:
            self.informed_sampler = InformedSampling(self.env, "sampling_based", self.config.locally_informed_sampling)
    
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
            return math.sqrt(d) / d * workspace_diameter
            
        else:
            raise ValueError(f"Unknown step_size_strategy: {strategy}")

    def _sample_mode(self) -> Mode:
        """
        Selects which mode to expand next based on the selected strategy
        - "uniform": pick uniformly from reached_modes
        - "greedy": p_newest_mode to newest, else uniform
        - "frontier": p_frontier split across modes without outgoing transitions, 
                      remainder split across others by inverse node count 
        
        NOTE: frontier implementation from composite_prm_planner
        """
        if len(self.reached_modes) == 1:
            return self.reached_modes[0]
        
        # Greedy strategy
        if self.config.mode_sampling_type == "greedy":
            if random.random() < self.config.p_newest_mode:
                return self.reached_modes[-1]
            return random.choice(self.reached_modes)
            
        # Frontier strategy (from prm implementation) # TODO check
        if self.config.mode_sampling_type == "frontier":
            total_nodes = 0
            p_frontier = self.config.p_frontier
            p_remaining = 1.0 - p_frontier

            frontier_modes = []
            remaining_modes = []
            sample_counts = {}
            inv_prob = []

            # Check for frontier modes in the list of so far discovered modes
            for m in self.reached_modes:
                sample_count = self.tree.subtrees[m].size
                total_nodes += sample_count
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
        is_skill_mode = self._get_active_skill_task(mode) is not None

        # Goal/transition bias
        if random.random() < self.config.p_goal:
            self._dbg_goal_bias_attempt += 1
            q_target = self._sample_transition_config(mode)
            if q_target is not None:
                self._dbg_goal_bias_success += 1
                return q_target, False # Not uniform
        
        # Informed sampling (after first solution, non-skill modes only) # TODO non skill modes only, correct reasoning?
        if (self.solution_node is not None 
            and self.config.try_informed_sampling
            and self.informed_path is not None
            and not is_skill_mode):
            q_informed = self._sample_informed(mode)
            if q_informed is not None:
                self._dbg_informed_success += 1
                return q_informed, False
        
        # Uniform sampling (fallback)
        return self.env.sample_config_uniform_in_limits(), True # Is uniform

    def _sample_transition_config(self, mode: Mode) -> Configuration:
        """
        Samples a configuration that satisfies the transition of the current mode 
        """
        # Informed transition sampling (after first solution, non-skill mode, not terminal)
        if (self.config.try_informed_sampling
            and self.informed_sampler is not None
            and self.informed_path is not None
            and self._get_active_skill_task(mode) is None
            and not self.env.is_terminal_mode(mode)):
            self._dbg_informed_trans_attempt += 1
            q = self.informed_sampler.generate_transitions(
                self._non_skill_reached_modes(),
                self.config.informed_batch_size,
                self.informed_path,
                active_mode=mode
            )
            if q is not None and q != []:
                self._dbg_informed_trans_success += 1
                return q # Else uniform below

        # Uniform transition sampling
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
            constrained_robots = active_task.robots
            goal = active_task.goal.sample(mode)

            # Build configuration 
            q = self.env.sample_config_uniform_in_limits()
            
            # TODO should remain random, or maybe not.. why have inactive random if we can do something smarter..?
            # # NOTE: instead of uniform sampling, use config from random node in subtree
            # subtree = self.tree.subtrees.get(mode)
            # idx = random.randint(0, subtree.size - 1)
            # q_src = subtree.nodes[idx].state.q
            # q = self.env.get_start_pos().from_flat(q_src.state().copy())

            # Apply constraints
            end_idx = 0
            for i, robot in enumerate(self.env.robots):
                if robot in constrained_robots:
                    # Overwrite with goal
                    dim = self.env.robot_dims[robot]
                    q[i] = goal[end_idx : end_idx + dim]
                    end_idx += dim
                else:
                    # If robot has upcoming skill
                    # ...
                    pass

            # Validate that the constrained config is collision-free in current mode
            if self.env.is_collision_free(q, mode):
                # print(f"[TRANSITION SAMPLING] done at {iters} iterations")
                return q
            
        print(f"[TRANSITION SAMPLING] failed at {iters} iterations")
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
        Builds the interpolated path that the informed sampled uses as reference
        (focal points for the PHS ellipsoid). Called after every cost improvement
        """
        if self.best_path is not None and len(self.best_path) > 1:
            self.informed_path = interpolate_path(self.best_path)
        else:
            self.informed_path = None

    def _expand(self, n_near: Node, q_target: Configuration, mode: Mode, skill_task, is_uniform: bool = True) -> List[Node]:
        """
        
        """
        # Strategy for expanding/steering in NON-skill-modes
        if skill_task is None:
            state_new = self._linear_steer(n_near, q_target, mode)
            if state_new is None or not self._validate(state_new, n_near, is_skill=False, is_uniform=is_uniform):
                self._dbg_validate_fail += 1
                return []
            return [self._create_and_add_node(state_new, n_near, mode, is_skill=False)]
        
        # Strategies for expanding/steering in skill-modes
        strategy = self.config.skill_expansion_strategy
        if strategy == "single_step":
            return self._expand_single_step(n_near, q_target, mode, skill_task)
        if strategy == "full_rollout":
            return self._expand_full_rollout() # TODO remove (won't need it)
        if strategy == "kinodynamic":
            return self._expand_kinodynamic(n_near, q_target, mode, skill_task)
        
        raise ValueError(f"Unknown skill_explansion_strategy: {strategy}")

    def _steer_inactive(self, q_full: np.ndarray, q_target_vec: np.ndarray, active_indices: np.ndarray, dt: float) -> np.ndarray:
        """
        Concurrent inactive robot steering, bounded by inactive_max_vel * dt
        """
        if self.config.inactive_steering_mode != "concurrent":
            return q_full.copy()
        
        direction = q_target_vec - q_full
        direction[active_indices] = 0.0
        inactive_dist = np.linalg.norm(direction)

        if inactive_dist <= 1e-8:
            return q_full.copy()
        eta_inactive = min(self.config.inactive_max_vel * dt, inactive_dist)

        return q_full + eta_inactive * (direction / inactive_dist)

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
        if dist <= self.eta: # TODO change, snaps all the time.. (or fix eta)
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

    def _validate(self, state_new: State, n_near: Node, is_skill: bool, is_uniform: bool = True) -> bool:
        """
        Collision checking for configurations and edges
        """
        # 1. Config check
        is_state_free = self.env.is_collision_free(state_new.q, state_new.mode)

        # 2. Update c_free self._update_cfree_estimate
        if self.config.use_rrt_star and not is_skill:
            self._update_cfree_estimate(was_valid=is_state_free, was_uniform=is_uniform)

        # 3. Failure based on config check 
        if not is_state_free:
            if is_skill and self._dbg_validate_fail % 100 == 0:  # Print every 100th fail to avoid spam
                print(f"[DEBUG SKILL] State collision at t_norm = {n_near.skill_step/100:.2f} (approx)")
            return False
        
        # 4. Edge check
        if not self.env.is_edge_collision_free(state_new.q, n_near.state.q, state_new.mode):
            if is_skill and self._dbg_validate_fail % 100 == 0:
                print(f"[DEBUG SKILL] Edge collision from step {n_near.skill_step} to {n_near.skill_step + 1}")
            return False
        
        return True

    def _create_and_add_node(self, state_new: State, n_near: Node, mode: Mode, is_skill: bool = False) -> Node:
        """
        Handles node creation and addition, including RRT* parent optimization

        NOTE: find_best_parent always runs for non-skill nodes when use_rrt_star=True,
        regardless of rewire_after_first_solution flag as it is cheap and should always
        be beneficial..
        """
        if is_skill:
            parent = n_near
            cost_to_parent = self.env.config_cost(n_near.state.q, state_new.q)
            cost = parent.cost + cost_to_parent
        elif self.config.use_rrt_star:
            parent, cost, cost_to_parent = self._find_best_parent(n_near, state_new.q, mode) # Always
        else:
            parent = n_near
            cost_to_parent = self.env.config_cost(n_near.state.q, state_new.q)
            cost = n_near.cost + cost_to_parent

        # Create
        n_new = Node(state_new, parent=parent)
        n_new.cost = cost
        n_new.cost_to_parent = cost_to_parent

        # Skill node bookkeeping
        if is_skill:
            n_new.is_skill_waypoint = True
            n_new.state.is_skill_waypoint = True # TODO (for shortcutter.. change and only keep on node..?)
            n_new.skill_step = n_near.skill_step + 1

        # Add
        self.tree.subtrees[mode].add_node(n_new)
        n_new.parent.children.append(n_new)
        return n_new

    def _check_transitions(self, n_new: Node):
        """
        Checks if n_new triggers a mode switch 
        - Normal mode: is_transition()?
        - Skill mode: skill.done()?
        If transition node -> get the next valid modes, add and seed new subtree
        """
        mode = n_new.state.mode
        skill_task = self._get_active_skill_task(mode)

        # Check transition for skill mode
        if skill_task is not None:
            skill = skill_task.skill

            # Extract subspace
            q_full = n_new.state.q.state()
            active_indices = self._get_active_subspace_indices(skill_task)
            q_subspace = q_full[active_indices]

            is_timed = isinstance(skill, BaseDeterministicTimedSkill)
            if is_timed:
                dt = skill.dt
                n_steps = max(1, round(skill.duration / dt))
                t_norm = n_new.skill_step / n_steps
                is_transition = skill.done(t_norm, q_subspace, self.env)
            else:
                is_transition = skill.done(q_subspace, self.env)
        
        # Check transition for normal (geometric) mode
        else:
            # Standard geometric transition check
            is_transition = self.env.is_transition(n_new.state.q, n_new.state.mode)

        # Transition node -> get next modes, add and seed new subtree
        if is_transition:
            self._dbg_is_trans_true += 1
            n_new.is_transition = True

            # Next modes
            next_modes = self.env.get_next_modes(n_new.state.q, n_new.state.mode)
            valid_next_modes = self.mode_validation.get_valid_modes(n_new.state.mode, list(next_modes))

            if not valid_next_modes:
                self._dbg_get_next_empty += 1
                return

            for next_mode in valid_next_modes:
                # Skip if same configuration is not valid in new mode
                if not self.env.is_collision_free(n_new.state.q, next_mode):
                    self._dbg_seed_coll_fail += 1
                    continue

                # Register mode if new
                if next_mode not in self.reached_modes:
                    self.reached_modes.append(next_mode)
                    self.tree.add_subtree(next_mode)

                # Seed next mode with a transition node (always -> alternative entries provide different cost paths)
                seed_state = State(n_new.state.q, next_mode)
                seed_node = Node(seed_state, parent=n_new)
                seed_node.cost = n_new.cost 
                seed_node.is_transition = True

                self.tree.subtrees[next_mode].add_node(seed_node)
                n_new.children.append(seed_node)
                self._dbg_seed_added += 1

    def _shortcut(self, path: List[State]) -> List[State]:
        """
        Post-processes a path with robot_mode_shortcut
        Skill segments are protected
        """
        shortcut_path, _ = shortcutting.robot_mode_shortcut(
            self.env, path, self.config.shortcutting_iters,
            resolution=self.env.collision_resolution,
            tolerance=self.env.collision_tolerance,
            robot_choice=self.config.shortcutting_mode,
            interpolation_resolution=self.config.shortcutting_interpolation_resolution
        )

        # Remove interpolated points used in shortcutting (collision check)
        return shortcutting.remove_interpolated_nodes(shortcut_path)

    def _extract_path(self, node: Node) -> List[State]:
        """
        Traces back from the giben node to the root
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
                for wp in n.skill_edge.waypoints[1:]:
                    q_wp = self.env.get_start_pos().from_flat(wp)
                    path.append(State(q_wp, n.state.mode, is_skill_waypoint=True))
            else:
                path.append(n.state)
        return path

    # TODO ADDITIONAL HELPER FUNCTIONS (plan)
    def _init_debug_counters(self):
        """
        # NOTE: GENERATED WITH GEMINI

        Initializes all debug counters.
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
        self._dbg_w_rewire_extracts = 0
        self._dbg_w_shortcut_hits = 0
        self._dbg_last_r_n = 0.0

        self._dbg_kino_edges = 0

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
        informed_status = "ON" if (self.solution_node and self.config.try_informed_sampling) else "OFF"
        
        print(
            f"[{tag}] it={iterations} nodes={nodes} modes={len(self.reached_modes)} "
            f"best={self.best_cost:.3f} sol={sol_cost:.3f} "
            f"eta={self.eta:.2f} r_n={self._dbg_last_r_n:.3f}\n"
            f"rewire={rewire_status} informed={informed_status}\n"
            f"       | w: rewires={self._dbg_w_rewires} "
            f"bestP={self._dbg_w_best_parent_swaps} "
            f"nearAvg={near_avg:.1f} "
            f"reExt={self._dbg_w_rewire_extracts} "
            f"sCut={self._dbg_w_shortcut_hits}\n"
            f"       | c: snap={self._dbg_snap_events} "
            f"vfail={self._dbg_validate_fail} "
            f"gb={self._dbg_goal_bias_success}/{self._dbg_goal_bias_attempt} "
            f"inf={self._dbg_informed_success}/{self._dbg_informed_attempt} "                        
            f"infT={self._dbg_informed_trans_success}/{self._dbg_informed_trans_attempt} "  
            f"isT={self._dbg_is_trans_true} "
            f"nextE={self._dbg_get_next_empty} "
            f"sCF={self._dbg_seed_coll_fail} sAdd={self._dbg_seed_added} "
            f"minNN={self._dbg_min_nn_dist:.3f}\n"
            f"       | kino: edges={self._dbg_kino_edges} " 
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

    def _record_solution(self, costs: List[float], times: List[float], node: Optional[Node] = None, path: Optional[List[State]] = None):
        """
        Tracks best_cost, best_path, and solution_node
        Returns True when best cost got updated
        """
        updated = False
        now = time.time() - self.start_time
        
        if node is not None:
            # 1. Update solution_node if this node is better or it's the first solution
            if self.solution_node is None or node.cost < self.solution_node.cost - 1e-9:
                self.solution_node = node
            
            # 2. Update best_cost/best_path if this node provides an improvement over ANY best so far
            if node.cost < self.best_cost - 1e-9:
                self.best_cost = node.cost
                self.best_path = self._extract_path(node)
                print(f"[SOLUTION] New best cost: {self.best_cost:.3f}")
                updated = True
                
        elif path is not None:
            # 3. Path improved via post-processing (shortcutting)
            new_cost = path_cost(path, self.env.batch_config_cost)
            if new_cost < self.best_cost - 1e-9:
                self.best_cost = new_cost
                self.best_path = path
                updated = True
        
        if updated:
            costs.append(self.best_cost)
            times.append(now)
            return True
        return False

    # TODO SKILLS
    def _get_active_skill_task(self, mode: Mode):
        """
        Returns the task with a skill in this mode or None 
        """
        next_ids = self.mode_validation.get_valid_next_ids(mode)
        if not next_ids:
            return None
        
        task = self.env.get_active_task(mode, next_ids)
        if hasattr(task, 'skill') and task.skill is not None:
            return task
        return None
    
    def _non_skill_reached_modes(self) -> List[Mode]:
        """
        Filter skill modes
        """
        return [m for m in self.reached_modes if self._get_active_skill_task(m) is None]

    def _get_active_subspace_indices(self, active_task) -> List[int]:
      """
      Returns indices for the robots involved in the active task
      """
      active_indices = []
      end_idx = 0
      for robot in self.env.robots:
          dim = self.env.robot_dims[robot]
          if robot in active_task.robots:
              active_indices.extend(range(end_idx, end_idx + dim))
          end_idx += dim
      return active_indices
    
    def _expand_single_step(self, n_near: Node, q_target: Configuration, mode: Mode, skill_task) -> List[Node]:
        """
        Rolls the skill out by one step, with optional concurrent steering for the inactive robots.
        Inactive robots motions are bounded by max_vel*dt
        """
        skill = skill_task.skill
        dt = skill.dt

        q_full = n_near.state.q.state().copy()
        active_indices = self._get_active_subspace_indices(skill_task)
        q_subspace = q_full[active_indices]

        # 1. Skill step for active robots
        all_joints = self.env.get_joint_names()
        self.env.C.selectJoints(skill.joints)

        if isinstance(skill, BaseDeterministicTimedSkill):
            n_steps = max(1, round(skill.duration / dt))
            if n_near.skill_step >= n_steps: # Avoid step past horizon
                self.env.C.selectJoints(all_joints)
                return []
            t_norm = min((n_near.skill_step + 1) / n_steps, 1.0)
            q_subspace_new = skill.step(t_norm, q_subspace, self.env)
        else: 
            q_subspace_new = skill.step(q_subspace, self.env)

        self.env.C.selectJoints(all_joints)

        # 2. Steer inactive robots with bounded step 
        q_base = self._steer_inactive(q_full, q_target.state(), active_indices, dt)

        # 3. Overwrite the active subspace in the base configuration with the skill result
        q_base[active_indices] = q_subspace_new
        q_new = self.env.get_start_pos().from_flat(q_base)
        state_new = State(q_new, mode, is_skill_waypoint=True)
     
        # 4. Validate and add node
        if not self._validate(state_new, n_near, is_skill=True):
            return []
        
        return [self._create_and_add_node(state_new, n_near, mode, is_skill=True)]
    
    def _expand_full_rollout(self, n_near: Node, q_target: Configuration, mode: Mode, skill_task) -> List[Node]: # TODO
        """
        Full skill rollout from n_near
        """
        raise NotImplementedError
    
    def _expand_kinodynamic(self, n_near: Node, q_target: Configuration, mode: Mode, skill_task) -> List[Node]:
        """
        Rolls out skill steps as one skill edge
        - All intermediate steps are collision checked during construction
        - Only the end node enters the subtree (for NN sesarch)
        - Intermediate waypoints are stored in a SkillEdge on the end node
        - 
        """
        skill = skill_task.skill
        dt = skill.dt
        n_kino = self.config.kinodynamic_steps
        active_indices = self._get_active_subspace_indices(skill_task)

        is_timed = isinstance(skill, BaseDeterministicTimedSkill)
        n_total_steps = max(1, round(skill.duration / dt)) if is_timed else None

        base_step = n_near.skill_step
        q_curr = n_near.state.q.state().copy()
        q_target_vec = q_target.state()

        waypoints = [q_curr.copy()] # waypoints[0] = parent config
        if is_timed:
            remaining_steps = n_total_steps - base_step
            if remaining_steps <= 0:
                return []
            rollout_steps = min(n_kino, remaining_steps)
            t_norms_list = [base_step / n_total_steps]
        else: 
            rollout_steps = n_kino
            t_norms_list = [0.0]

        skill_done = False
        for i in range(1, rollout_steps + 1):
            q_subspace = q_curr[active_indices]

            all_joints = self.env.get_joint_names()
            self.env.C.selectJoints(skill.joints)
            
            # Skill step
            if is_timed:
                t_norm = min((base_step + i) / n_total_steps, 1.0)
                q_subspace_new = skill.step(t_norm, q_subspace, self.env)
            else:
                q_subspace_new = skill.step(q_subspace, self.env)
            self.env.C.selectJoints(all_joints)

            # Inactive steering (same bounded logic as single_step)
            q_next = self._steer_inactive(q_curr, q_target_vec, active_indices, dt)
            q_next[active_indices] = q_subspace_new

            # Collision check this step
            q_next_cfg = self.env.get_start_pos().from_flat(q_next)
            q_curr_cfg = self.env.get_start_pos().from_flat(q_curr)
            prev_node = Node(State(q_curr_cfg, mode, is_skill_waypoint=True))
            prev_node.skill_step = base_step + i - 1
            state_next = State(q_next_cfg, mode, is_skill_waypoint=True)

            if not self._validate(state_next, prev_node, is_skill=True):
                break

            waypoints.append(q_next.copy())
            t_norms_list.append(t_norm if is_timed else float(i))

            # Check skill completion
            if is_timed:
                skill_done = skill.done(t_norm, q_subspace_new, self.env)
            else:
                skill_done = skill.done(q_subspace_new, self.env)

            q_curr = q_next

            if skill_done:
                break

        if len(waypoints) < 2:
            return [] # No valid steps at all

        # Create end node (only this enters the subtree)
        actual_steps = len(waypoints) - 1
        q_end_cfg = self.env.get_start_pos().from_flat(waypoints[-1])
        state_new = State(q_end_cfg, mode, is_skill_waypoint=True)

        skill_edge = SkillEdge(
            waypoints=np.array(waypoints),
            t_norms=np.array(t_norms_list)
        )

        n_new = self._create_and_add_node(state_new, n_near, mode, is_skill=True)
        n_new.skill_step = base_step + actual_steps
        n_new.skill_edge = skill_edge

        # TODO (okay for now) but careful, that's the straight line cost.. not sum of intermediate step cost.. would be problem only in RRT* tho? but won't break optimality as in rewiring we are skipping skill nodes..? 
        edge_cost = self._skill_edge_cost(np.asarray(waypoints), mode)
        n_new.cost_to_parent = edge_cost
        n_new.cost = n_near.cost + edge_cost 

        # Debug
        self._dbg_kino_edges += 1

        return [n_new]

    def _skill_edge_cost(self, waypoints: np.ndarray, mode: Mode) -> float:
        """
        Computes true cost for kinodynamic edges instead of using straight-line parent-to-end-costs
        """ 
        total = 0.0
        q_from_flat = self.env.get_start_pos().from_flat
        for i in range(len(waypoints) - 1):
            total += self.env.config_cost(q_from_flat(waypoints[i]), q_from_flat(waypoints[i + 1]))
        return total 
    
    # TODO RRT*
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

        r_n = min(gamma_rrtstar * (logn/n)^(1/d), eta)
        n: number of nodes in tree
        d: dimensionality of state space
        gamma_rrtstar: asymptotic optimality constant
        eta: step_size
        """
        if n <= 1:
            return self.eta
        r_n = self.gamma_rrt_star * (math.log(n) / n) ** (1 / self.d)
        return min(r_n, self.eta) # TODO set self.rewire_radius instead

    def _find_best_parent(self, n_near: Node, q_new: Configuration, mode: Mode):
        """
        RRT*: find the lowest-cost parent from the near set
        Returns (best_parent, best_cost, cost_to_parent)
        """
        subtree = self.tree.subtrees[mode]
        r_n = self._compute_rewiring_radius(subtree.size)
        near_set = subtree.get_near(q_new, r_n, self.config.distance_metric)

        # DEBUG
        self._dbg_last_r_n = r_n
        self._dbg_w_near_size_sum += len(near_set)
        self._dbg_w_near_size_count += 1

        best_parent = n_near # TODO needs collision check?
        best_cost_to_parent = self.env.config_cost(n_near.state.q, q_new)
        best_cost = n_near.cost + best_cost_to_parent

        for idx, _ in near_set:
            candidate = subtree.nodes[idx]
            if candidate is n_near or candidate.is_skill_waypoint:
                continue
            edge_cost = self.env.config_cost(candidate.state.q, q_new)
            potential_cost = candidate.cost + edge_cost
            if potential_cost < best_cost:
                if self.env.is_edge_collision_free(candidate.state.q, q_new, mode):
                    best_parent = candidate
                    best_cost_to_parent = edge_cost
                    best_cost = potential_cost

        if best_parent is not n_near:
            self._dbg_w_best_parent_swaps += 1

        return best_parent, best_cost, best_cost_to_parent

    def _rewire(self, n_new: Node, mode: Mode):
        """
        RRT*: rewire neighbors if n_new provides cheaper path
        """
        subtree = self.tree.subtrees[mode]
        r_n = self._compute_rewiring_radius(subtree.size)
        near_set = subtree.get_near(n_new.state.q, r_n, self.config.distance_metric)

        # DEBUG
        self._dbg_last_r_n = r_n
        self._dbg_w_near_size_sum += len(near_set)
        self._dbg_w_near_size_count += 1

        for idx, _ in near_set:
            n_near = subtree.nodes[idx]
            if n_near is n_new or n_near is n_new.parent or n_near.is_skill_waypoint:
                continue
            edge_cost = self.env.config_cost(n_new.state.q, n_near.state.q)
            potential_cost = n_new.cost + edge_cost
            if potential_cost < n_near.cost:
                if self.env.is_edge_collision_free(n_new.state.q, n_near.state.q, mode):
                    # Detach old parent
                    old_parent = n_near.parent
                    if old_parent is not None:
                        old_parent.children.remove(n_near)

                    # Rewire
                    n_near.parent = n_new
                    n_new.children.append(n_near)
                    n_near.cost_to_parent = edge_cost
                    n_near.cost = potential_cost
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

    # TODO BRRT*
    # ...
