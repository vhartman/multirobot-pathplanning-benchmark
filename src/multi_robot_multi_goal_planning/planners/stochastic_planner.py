"""
Stochastic path planning

1. CL reactive planner (MDP solved via DP)
2. OL conservative planner (worst-case envelope avoidance)

Flowchart: 
1. Collect active robot rollouts, simulating the stochastic skill
2. Bin the active configuration (spatial binning) and calculate transitions
3. Build a local grid for the inactive robot and find adjacent neighbors
4. (TBD) collision check between active bins and inactive grid nodes
5. Run Backward Induction to compute optimal value and policy tables
6. Execute CL or OL policies on fresh rollouts


IMPLEMENTATION:

Imports
Configuration (dataclass)

Core part / algorithms (ipad notes)
- bin_rollouts
- transition_probs
- build_grid
- dijkstra
- 
- backward_induction

Stochastic planner class
- init
- setup?
- plan (reactive vs. conservatice)
- evaluate
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class Config:
    """
    Planner parameters
    """

    # n_rollouts
    # 
    # params for grid and bin (discretization active and inactive continuous configuration space)
    #
    # idle_cost
    # fail_cost 
    # 
    # mc_trials
    # rng_seed
    #  
    pass


# =====================================================================
# Policy Object
# =====================================================================

class StochasticPolicy:
    """
    # TODO check (maybe easier to have this class..)
    Encapsulates a computed policy. Used by global planner to do fast continuous-to-discrete
    state mapping and retrieve the next collision-free action for the inactive robot at each step
    """

    def __init__(self):
        """
        
        """
        pass

    def get_next_state(self):
        """
        Calculates the next target joint configuration for the inactive robot
        Could be useful?
        """
        #
        # return q_inactive_next 
        raise NotImplementedError


# =====================================================================
# Core algorithm
# =====================================================================

def bin_rollouts():
    """
    Groups continuous active robot trajectories into discrete representative bins at each
    phase. Used to discretize the continuous C-space into a finite state space for MDP planning
    """
    # Init lists
    # Loop phase k from 0 to K 
    # - Extract configs of all rollouts at phase k   
    # - Discretize -> cell coordinates
    # - Map each continuous config to unique cell ID
    # - For each unique cell ID compute mean of all continuous configs mapped to it 
    # - Append means to respective lists
    #

    # return   
    raise NotImplementedError

def transition_probs():
    """
    Estimates the transition probabilities between active robot bins from phase k to k+1. Used
    to construct the stochastic propagation model of the active robot's noisy skill
    """
    # Init list (of dict for each phase k)
    # For each phase k:
    # - Loop through each rollout index
    # -- Get bin index of the rollout at phase k (a) and phase k+1 (b)
    # -- Increment transition count from bin a to bin b
    # - Convert counts into probability distributions
    # -- For each bin a divide transition counts to b by the total transitions out of a
    # -- Store in dict
    # - Append list of dicts for phase k to main transition list 
    #
    #  

    # return transition probability list 
    raise NotImplementedError

def build_grid():
    """
    Generates dense coordinates covering the inactive robot's workspace to define search node 
    positions. Used to bound and structure the discrete grid space
    """
    # Not sure how, not sure if actually needed..
    # 
    # 

    # return grid array 
    raise NotImplementedError

def backward_induction():
    """
    Runs DP backwards from the terminal goal state to the start state. Used to find the optimal
    expected cost-to-go value function and compile the optimal feedback control policy 
    """
    # Check iPad notes for implementation details
    # 
    # 
    # 
    # 
    # 

    # return value table and policy table 
    raise NotImplementedError


# =====================================================================
# Planner
# =====================================================================

class StochasticSkillPlanner:
    """
    
    Used as high-level "wrapper" to setup the pipeline and plan paths/policies 
    """

    def __init__(self):
        """
        Initializes the planner wrapper
        """
        #
        # 
        # 
        # 
        #  
        pass

    def setup(self):
        """
        Prepares all required data structures before the solving. Used to perform rollouts, 
        construct the search grid and precompute...
        """
        # ...?
        # Run MC simulations to collect n_rollouts trajectories of the active skill
        # Run bin rollouts to cluster rollout trajectories into spatial bins per phase
        # Run transition probs to build transition probability model between those active bins
        # Build the grid for the inactive robot?
        # ...? 
        #  
        raise NotImplementedError

    def plan(self):
        """
        Computes and simulates a path using backward induction with either the reactive or 
        conservative strategy. Used to run the evaluation and return paths and metrics?
        """
        # Run setup
        # Select strategy (reactive, conservative)
        # - If reactive: use all bins and transition probabilities directly
        # - If conservative: collapse all active bins into sindle worst-case envelope and transition deterministically
        # Run backward induction to compute optimal policy table 
        # Run policy evaluation -> cost, other metrics
        # Reconstruct path
        #
        # 

        # return path and info (metrics..)  
        raise NotImplementedError
    
    def get_policy(self):
        """
        Computes and returns the feedback control policy (StochasticPolicy) object. Used by 
        global planners to query local steering actions
        """
        # Setup
        # Run backward induction to obtain policy table
        # Instatiate and return a StochasticPolicy object using policy, grid, ...
        # 
        # 
        #  
        raise NotImplementedError

    # TODO add more methods while coding..

    def _rollout(self):
        """
        Runs a forward simulation (rollout) of the active robot's skill 
        """
        # Get active robot start configuration
        # Init trajectory sequence list with start config
        # Loop step index from 0 to n_steps-1
        # - Run, record
        # - If skill done -> break
        
        # return full trajectory array
        raise NotImplementedError

    def _execute_once(self):
        """
        Executes the policy control loop once on a new active rollout
        """
        #
        # 
        # 
        # 
        # 
        # 

        # return ?
        raise NotImplementedError 

    def _reconstruct(self):
        """
        Builds shortest path from ?? to the goal once the skill finishes 
        """
        #
        # 
        # 
        # 
        #  
        raise NotImplementedError

    def _evaluate(self):
        """
        Evaluates policy metrics (expected control cost, collision rates, goal-reaching rates) 
        across multiple randomized simulation runs
        """
        # Loop for mc_trials simulation runs:
        # - Run policy execution 
        # - Accumulate outcomes
        # Compute means, other metrics
        # 

        # Return dictionary summarizing metrics 
        raise NotImplementedError
    
    def _representative_path(self):
        """
        Executes the policy once and returns the sequence of state objects. Used to generate the 
        final path for visualization / execution..
        """
        # Execute policy once
        # Iterate through active and inactive trajectory configurations
        # Merge to get state object
        # 
        #  Return list of state objects 
        raise NotImplementedError