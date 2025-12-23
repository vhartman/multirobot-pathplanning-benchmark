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
            # sample rnd node + time
            # - possibly informed sampling
            # - possibly goal bias
            # - possibly project

            # steer towards node
            # - return an edge-type to support non-linear interpolation

            # check edge for collision
            # - take edge as input, not to points

            # add node if no collision