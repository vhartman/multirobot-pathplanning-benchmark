import os
import sys
import importlib.util

from . import abstract_env

from .rai import rai_envs
from .rai import rai_single_goal_envs
from .rai import rai_unordered_envs
from .rai import rai_free_envs
from .rai import rai_envs_constrained
from .rai import rai_skill_envs

import sys
sys.path.append("/usr/local/lib/python3.10/dist-packages")  # TODO: install mr_planner_core into venv
if importlib.util.find_spec("mr_planner_core") is not None:
    from . import mr_vamp_env
sys.path.pop()


if importlib.util.find_spec("pinocchio") is not None:
    from . import pinocchio_env

if importlib.util.find_spec("mujoco") is not None:
    from . import mujoco_env

from .registry import get_env_by_name, get_all_environments

__all__ = ["get_env_by_name", "get_all_environments"]
