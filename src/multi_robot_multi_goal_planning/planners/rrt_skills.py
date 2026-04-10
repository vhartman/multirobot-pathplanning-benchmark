"""
Multi-modal RRT(*) with skills

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
from .baseplanner import BasePlanner
from .mode_validation import ModeValidation
from .termination_conditions import PlannerTerminationCondition

# TODO [ ] add all relevant hyperparams
# TODO [ ] p_transition for mode specific goal AND p_goal for terminal goal?
@dataclass
class RRTSkillsConfig:
    """
    Hyperparameters for the multi-modal RRT with skills
    """
    # RRT
    step_size: float = 0.1
    p_goal: float = 0.1 
    p_transition: float = 0.2 # TODO
    
    distance_metric: str = "max_euclidean"
    with_mode_validation: bool = False
    with_noise: bool = False

    # RRT*
    use_rrt_star: bool = False
    rewire_radius: float = 0.5

    # BRRT*
    is_bidirectional: bool = False

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

# TODO [ ] vectorized batch_q for fast nearest-neighbor lookups
# TODO [x] pre-allocate so append(node) -> O(1) instead of O(N)
# TODO [o] how to do preallocation? allocate and then double?
# TODO [x] init the multi modal tree
# TODO [ ] figure out how to add a new subtree and transition seeding at mode boundary
# TODO [x] add root to multi modal tree? optional?
# TODO [x] import batch_config_cost batch_config_dist or access with self.env?

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

    def get_near(self, q: Configuration, radius: float, metric: str = "max_euclidean"):
        """
        Returns (index, dist) for all nodes within radius
        """
        if self.size == 0:
            return []
        
        dists = batch_config_dist(q, self.batch_q[:self.size], metric)
        indices = np.where(dists < radius)[0]
        return [(int(i), float(dists[i])) for i in indices]

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

    def get_nearest(self, mode: Mode, q_target: Configuration, metric: str = "max_euclidean") -> Tuple[Node, float]:
        """
        Finds the nearest node in the specified mode's subtree
        """
        subtree = self.subtrees.get(mode)
        dists = self.env.batch_config_cost(q_target, subtree.batch_q[:subtree.size], metric)
        idx = np.argmin(dists)
        return subtree.nodes[idx], float(dists[idx])

# TODO [ ] in _sample_mode add different mode sampling strategies like PRM (for now uniform)
# TODO [ ] in _sample_transition_config add reached_terminal_mode like PRM?
# TODO [ ] differentiate between goal bias and transition bias?
# TODO [x] check in PRM transition cost=0.0?
# TODO [ ] in _check_transitions add skill completion check  
# TODO [x] in _steer use config_cost(), restpecting distance_metric 
# TODO [x] in _check_transitions always create and insert seed node (even if mode already reached)
# TODO [x] in _initialize_planner add early return if self.tree.root is not None (not duplicated start mode/node) if plan() called again?
# TODO [ ] add debug prints in the planning loop
# TODO [ ] shortcutting

# TODO [ ] update _add_node to set skill flag
# TODO [ ] update _check_transitions when skill is done
# TODO [ ] differentiate between linear steer and skill steer
# TODO [ ] 

# TODO [ ] in _steer add skills -> call skill.step()
# TODO [o] add rewiring (RRT*)
# TODO [ ] compute gamma (RRT*)
# TODO [ ] add bidirectional (BRRT*)

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

    def plan(self, ptc: PlannerTerminationCondition, optimize: bool = False):
        """
        Main planning loop
        """
        self.start_time = time.time()
        self._initialize_planner()

        iterations = 0
        costs = []
        times = []

        while not ptc.should_terminate(iterations, time.time() - self.start_time):
            iterations += 1

            # 1. Sample mode
            mode = self._sample_mode()

            # 2. Sample target
            q_target = self._sample_target(mode)

            # 3. Nearest neighbor
            n_near, dist = self.tree.get_nearest(mode, q_target, self.config.distance_metric)

            # 4. Steer (linear or skill based)
            state_new = self._steer(n_near, q_target, mode)

            if state_new and self._validate(state_new, n_near):
                if self.config.use_rrt_star:
                    # RRT*: find best parent from near set
                    parent, cost, cost_to_parent = self._find_best_parent()
                else:
                    # RRT: parent is n_near
                    parent = n_near
                    cost_to_parent = self.env.config_cost(n_near.state.q, state_new.q)
                    cost = n_near.cost + cost_to_parent

                # Create and add new node
                n_new = Node(state_new, parent=parent)
                n_new.cost = cost
                n_new.cost_to_parent = cost_to_parent
                self._add_node(n_new, mode)

                # 5. Check transitions and termination
                if self.env.done(n_new.state.q, mode):
                    if self.solution_node is None or n_new.cost < self.solution_node.cost:
                        # Better solution
                        self.solution_node = n_new
                        costs.append(self.solution_node.cost)
                        times.append(time.time() - self.start_time)
                    
                    if not optimize:
                        break

                self._check_transitions(n_new)

                # RRT*: rewire
                if self.config.use_rrt_star:
                    self._rewire(n_new, mode) # TODO
                
                # # BRRT*
                # if self.config.is_bidirectional:
                #     pass # TODO

        # Extract path and info
        path = self._extract_path(self.solution_node) if self.solution_node else None
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
        
        # Mode
        start_mode = self.env.get_start_mode()
        self.reached_modes.append(start_mode)
        self.tree.add_subtree(start_mode)
        
        # Node
        start_node = Node(State(self.env.get_start_pos(), start_mode))
        start_node.cost = 0.0
        self.tree.root = start_node
        self.tree.subtrees[start_mode].add_node(start_node)

        # RRT*
        if self.config.use_rrt_star:
            self._set_gamma_rrt_star()

    def _sample_mode(self) -> Mode:
        """
        Selects which mode to expand next based on the selected strategy
        """
        return np.random.choice(self.reached_modes)

    def _sample_target(self, mode: Mode) -> Configuration:
        """
        Samples a random configuration or goal bias / transition
        """
        r = random.random()

        # Goal bias
        if r < self.config.p_goal:
            return self._sample_transition_config(mode)
        
        # TODO transition bias
        
        # Uniform sampling
        return self.env.sample_config_uniform_in_limits()

    def _sample_transition_config(self, mode: Mode) -> Configuration:
        """
        Samples a configuration that satisfies the transition of the current mode 
        """
        next_task_ids = self.mode_validation.get_valid_next_ids(mode)

        active_task = self.env.get_active_task(mode, next_task_ids)
        constrained_robots = active_task.robots
        goal_sample = active_task.goal.sample(mode)

        q = self.env.sample_config_uniform_in_limits()
        offset = 0

        for i, robot in enumerate(self.env.robots):
            dim = self.env.robot_dims[robot]
            if robot in constrained_robots:
                q[i] = goal_sample[offset : offset + dim]
                offset += dim
        return q
    
    def _steer(self, n_near: Node, q_target: Configuration, mode: Mode) -> Optional[State]:
        """
        Mode dependent steering:
        - Normal mode: linear interpolation towards q_target
        - Skill mode: call skill.step() # TODO
        """
        q_near_vec = n_near.state.q.state() # .state() returns NDArray
        q_target_vec = q_target.state()

        # dist = np.linalg.norm(q_target_vec - q_near_vec)
        dist = self.env.config_cost(n_near.state.q, q_target)
        if dist < 1e-6:
            return None
        
        step = min(dist, self.config.step_size)
        q_new_vec = q_near_vec + step * (q_target_vec - q_near_vec) / dist
        
        q_new = self.env.get_start_pos().from_flat(q_new_vec)
        return State(q_new, mode)

    def _linear_steer(self): # TODO
        """
        Standard linear interpolation towards q_target
        """
        raise NotImplementedError

    def _validate(self, state_new: State, n_near: Node) -> bool:
        """
        Collision checking for configurations and edges
        """
        if not self.env.is_collision_free(state_new.q, state_new.mode):
            return False
        
        if not self.env.is_edge_collision_free(state_new.q, n_near.state.q, state_new.mode):
            return False
        
        return True

    def _add_node(self, n_new: Node, mode: Mode):
        """
        Inserts node into tree and update bookkeeping
        """
        self.tree.subtrees[mode].add_node(n_new)
        n_new.parent.children.append(n_new)

    def _check_transitions(self, n_new: Node):
        """
        Checks if n_new triggers a mode switch 
        """
        if self.env.is_transition(n_new.state.q, n_new.state.mode):
            n_new.is_transition = True

            # Next modes
            next_modes = self.env.get_next_modes(n_new.state.q, n_new.state.mode)
            valid_next_modes = self.mode_validation.get_valid_modes(n_new.state.mode, list(next_modes))
        
            for next_mode in valid_next_modes:
                # Register mode if new
                if next_mode not in self.reached_modes:
                    self.reached_modes.append(next_mode)
                    self.tree.add_subtree(next_mode)

                # Seed next mode with a transition node (always -> alternative entries provide different cost paths)
                seed_state = State(n_new.state.q, next_mode)
                seed_node = Node(seed_state, parent=n_new)
                seed_node.cost = n_new.cost 

                self.tree.subtrees[next_mode].add_node(seed_node)
                n_new.children.append(seed_node)

    def _extract_path(self, node: Node) -> List[State]:
        """
        Traces back from the giben node to the root
        """
        path = []
        curr = node
        while curr: 
            path.append(curr.state)
            curr = curr.parent
        return path[::-1]

    # TODO SKILLS
    def _skill_steer(self): # TODO
        """
        Advance skill by one step?
        """
        raise NotImplementedError

    def _mode_has_skill(self, mode: Mode) -> bool: # TODO
        """
        Check if any robot's taks in this mode has a skill
        """
        raise NotImplementedError
    
    def _get_active_skill_task(self, mode: Mode): # TODO
        """
        Returns the task with a skill in this mode or None 
        """
        raise NotImplementedError
    
    # TODO RRT*
    def _set_gamma_rrt_star(self): # TODO
        """
        RRT*: asymptotic optimality constant
        
        gamma_rrtstar = (2(1+1/d))^(1/d) * (mu(X_free)/zeta_d)^(1/d)
        d: dimensionality of state space
        mu(X_free): Lebesque measure (volume) of obstacle-free search space
        zeta: volume of unit ball in d-dimensional space
        """
        # TODO where to best define self.d?
        self.d = sum(self.env.robot_dims.values())

        zeta_d = math.pi ** (self.d / 2) / (math.gamma(self.d / 2 + 1))
        mu_Xfree = ... # TODO

        gamma_rrt_star = (2 * (1 + 1 / self.d)) ** (1 / self.d) * (mu_Xfree / zeta_d) ** (1 / self.d)
        raise gamma_rrt_star

    def _compute_rewiring_radius(self, n: int): # TODO
        """
        RRT*: shrinking ball radius

        r_n = min(gamma_rrtstar * (logn/n)^(1/d), eta)
        n: number of nodes in tree
        d: dimensionality of state space
        gamma_rrtstar: asymptotic optimality constant
        eta: step size
        """
        # TODO where to best define self.d?
        self.d = sum(self.env.robot_dims.values())

        r_n = min(
            self._set_gamma_rrt_star * (np.log(n) / n) ** (1 / self.d), 
            self.config.step_size
        )
        return r_n

    def _find_best_parent(self): # TODO
        """
        RRT*: find the lowest-cost parent from the near set
        """
        # Init best_variables
        # Get set of near nodes from subtree
        # Iterate through set
        # - Get/compute cost for candidate (candidate.cost + dist) 
        # - If cost lower -> collision check & update best_variables
        # Return best_variables
        
        # return best_parent, best_cost, best_cost_to_parent
        raise NotImplementedError

    def _rewire(self): # TODO
        """
        RRT*: rewire neighbors if n_new provides cheaper path
        """
        # _compute_rewiring_radius()
        # Get set of newar nodes from subtree
        # Iterate through set
        # - Skip (continue) if n_near is n_new.parent or if n_near is n_new or if n_near is skill waypoint
        # - cost to check: n_new.cost + distance(n_new -> n_near)
        # if that cost < n_near.cost AND collision free edge -> rewire:
        # - Detach old parent: remove n_near from n_near.parent.children 
        # - Set new parent: n_near.parent = n_new
        # - Update also children of n_new, costs.. AND propagate cost!
        #  
        raise NotImplementedError
    
    def _propagate_cost_improvement(self): # TODO
        """
        RRT*: propagate cost changes down the tree after rewiring
        """
        # Get children of the node -> stack them in list
        # Iterate through list -> better while loop
        # while stack:
        # - pop -> child -> update cost of child (parent cost + cost to parent)
        # - extend stack with children of that child
        #
        raise NotImplementedError

    # TODO BRRT*
    # ...