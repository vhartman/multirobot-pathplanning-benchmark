import os
import sys
import importlib.util

from . import rai_envs
from . import rai_single_goal_envs
from . import rai_unordered_envs
from . import rai_free_envs
from . import abstract_env
from . import rai_envs_constrained


if importlib.util.find_spec("pinocchio") is not None:
    from . import pinocchio_env

if importlib.util.find_spec("mujoco") is not None:
    from . import mujoco_env

from .registry import get_env_by_name, get_all_environments

__all__ = ["get_env_by_name", "get_all_environments"]
