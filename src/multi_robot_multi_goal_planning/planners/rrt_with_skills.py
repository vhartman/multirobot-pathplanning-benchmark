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

from multi_robot_multi_goal_planning.problems.configuration import (
    batch_config_dist,
)
from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseProblem,
    Mode,
    State,
)

from .baseplanner import BasePlanner
from .termination_conditions import (
    PlannerTerminationCondition,
)


@dataclass
class RRTWithSkillsConfig:
    pass

class Node:
    def __init__(self):
        pass

class ModeTree:
    def __init__(self):
        pass

    def get_nearest(self, node):
        pass

    def add_node(self, node, parent):
        pass

class Tree:
    def __init__(self, n0):
        self.mode_trees = []

    def get_nearest(self, node):
        pass

    def add_node(self, node, parent):
        pass

# possible to solve with two aproachees:
# - formulate as big problem with time as part of the planning space
# -- fully time annotated: also means that we can formulate this properly as makespan cost as well.
# - formulate as geometric problem, and deal with the timing in the rollout.
# -- this means that we return a path that is partially annotated with time-info

class RRTWithSkills(BasePlanner):
    def __init__(self, env: BaseProblem, config: RRTWithSkillsConfig | None = None):
        self.env = env
        self.config = config if config is not None else RRTWithSkillsConfig()
        
    def plan(
        self, planner_termination_criterion: PlannerTerminationCondition
    ) -> Tuple[List[State] | None, Dict[str, Any]]:
        # initialize tree with start pose
        t = Tree()

        while True:
            ####################
            # possibility where we do time in the planning space:
            ####################

            # sample rnd node + time
            # - goal bias
            # - possibly informed sampling
            # - possibly project for constrained planning

            # steer towards node
            # - return an edge-type to support non-linear interpolation: should likely consist of a sequence of nodes (with associated time).
            # - steering behaviour depending on the mode we are in: there are modes that are skills
            # -- if a mode corresponds to a skill: rollout;
            # --- rollout of a skill: api should be ~step(configuration, delta t)
            # --- step until we reach the sampled time or the goal is reached
            # -- else: just linear interpolation, constrained planning/whatever
            # - analogy: this is somewhat like kinodynamic planning, where the system dynamics are
            #            given by the skill.

            # check timed edge for collision
            # - take timed edge as input, not points

            # add node if no collision

            # rewiring?
            # should work as normally, with the caveat that we can not connect to everywhere

            # if path found: shortcutting
            # -- shortcutting: should work as is.
            # -- do not shortcut over skills

            ####################
            # possibility with stuff in rollout
            ####################

            # sample rnd node + mode
            # same as above

            # steer:
            # - do similar to kinodynamic motion planning:
            # - sample duration -> rollout steps that much
            # - steer the linear interpolation to that location over the sampled time

            # rest is the same

            pass