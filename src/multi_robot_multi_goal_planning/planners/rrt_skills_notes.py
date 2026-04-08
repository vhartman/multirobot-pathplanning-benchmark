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
from .termination_conditions import PlannerTerminationCondition

@dataclass
class RRTSkillsConfig: # TODO
    """
    Hyperparameters for the multi-modal RRT with skills
    """
    # RRT
    step_size: float = 0.1
    p_goal: float = 0.2

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

        # Flags for skills and transitions
        self.is_skill_waypoint: bool = False
        self.skill_step: int = 0
        self.is_transition: bool = False

class Subtree:
    """
    Manages nodes and vectorized data for a specific mode
    """
    def __init__(self): # TODO
        pass

    def add_node(self):
        raise NotImplementedError

class MultiModalTree:
    """
    Collection of subtrees, one per mode
    """
    def __init__(self): # TODO
        pass

    def get_nearest(self): # TODO
        raise NotImplementedError

class RRTSkills(BasePlanner):
    """
    RRT planner:
    - Step-based skill rollouts
    - RRT* rewiring (optional)
    - Bidirectional search (optional)
    """
    def __init__(self, env: BaseProblem, config: RRTSkillsConfig): # TODO
        # super().__init__(env)
        self.env = env
        self.config = config
        self.tree = MultiModalTree(env)
        self.reached_modes: List[Mode] = []

    def plan(self, ptc: PlannerTerminationCondition): # TODO
        """
        Main planning loop
        """
        while not ptc.should_terminate():
            # Sample mode
            mode = self._sample_mode()

            # Sample target
            q_target = self._sample_target()

            # Nearest neighbor
            n_near, dist = self.tree.get_nearest(mode, q_target)

            # Steer (linear or skill based)
            # - Check transitions
            # - RRT* rewire
            # - Bidirectional connection
            state_new = self._steer(n_near, q_target, mode)

            if state_new and self._validate(state_new, n_near):
                n_new = Node(state_new, parent=n_near)
                self._add_node(n_new, mode)

                # Check transitions
                self._check_transitions(n_new)

                # RRT* rewire
                if self.config.use_rrt_star:
                    self._rewire(n_new, mode)
                
                # BRRT*
                if self.config.is_bidirectional:
                    # ...
                    raise NotImplementedError

            # Check solution
            # ...
        raise NotImplementedError

    def _sample_mode(self) -> Mode:
        """
        Selects which mode to expand next based on the selected strategy
        # TODO for now simple -> uniform
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
        
        # Uniform sampling
        return self.env.sample_config_uniform_in_limits()

    def _sample_transition_config(self, mode): # TODO
        """
        
        """
        raise NotImplementedError

    def _steer(self): # TODO
        """
        Mode dependent steering:
        - Normal mode: linear interpolation towards q_target
        - Skill mode: call skill.step()
        """
        raise NotImplementedError

    def _validate(self): # TODO
        """
        Collision checking for configurations and edges
        """
        raise NotImplementedError

    def _add_node(self): # TODO
        """
        Inserts node into tree
        """
        raise NotImplementedError

    def _check_transitions(self): # TODO
        """
        Checks if n_new triggers a mode switch or skill completion
        """
        raise NotImplementedError

    def _rewire(self): # TODO
        """
        RRT* rewire neighbors if n_new provides cheaper path
        """
        raise NotImplementedError
    
    def _update_costs(self): # TODO
        """
        RRT* propagate cost changes down the tree after rewiring
        """
        raise NotImplementedError