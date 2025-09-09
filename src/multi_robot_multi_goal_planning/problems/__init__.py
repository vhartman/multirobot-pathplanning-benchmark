import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from typing import Dict

from multi_robot_multi_goal_planning.problems.planning_env import BaseProblem

from .registry import get_env_by_name as re_get_env_by_name
from .registry import get_all_environments as re_get_all_environments

def get_all_environments() -> Dict:
    return re_get_all_environments()


def get_env_by_name(name: str) -> BaseProblem:
    return re_get_env_by_name(name)
