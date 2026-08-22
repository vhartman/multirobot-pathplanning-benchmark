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
# find_spec is NOT a reliable "importable" test: the mr_planner_core .so on disk is built for
# a different Python (3.10) and fails to LOAD in this venv. Guard the actual import instead.
try:
    sys.path.append("/usr/local/lib/python3.10/dist-packages")  # TODO: install mr_planner_core into venv
    from . import mr_vamp_env
except ImportError:
    pass
finally:
    if sys.path and sys.path[-1] == "/usr/local/lib/python3.10/dist-packages":
        sys.path.pop()


try:
    from . import pinocchio_env
except ImportError:
    pass

try:
    from . import mujoco_env
except ImportError:
    pass

from .core.registry import get_env_by_name, get_all_environments

__all__ = ["get_env_by_name", "get_all_environments"]
