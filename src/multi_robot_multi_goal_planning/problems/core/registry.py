from typing import Callable, Dict
from multi_robot_multi_goal_planning.problems.planning_env import BaseProblem

EnvFactory = Callable[[], BaseProblem]
REGISTRY: Dict[str, EnvFactory] = {}


def register(names):
    """
    names can be:
    - a single string (simple case)
    - a list of (name, kwargs) pairs
    """

    def decorator(cls):
        if isinstance(names, str):
            REGISTRY[names] = lambda: cls()
        else:
            for name, kwargs in names:
                if name in REGISTRY:
                    raise ValueError(f"Duplicate env name: {name}")
                REGISTRY[name] = lambda cls=cls, kwargs=kwargs: cls(**kwargs)
        return cls

    return decorator

def get_all_environments():
    return REGISTRY

def get_env_by_name(name: str) -> BaseProblem:
    try:
        return REGISTRY[name]()
    except KeyError:
        raise ValueError(f"Unknown environment: {name}")


def list_envs(prefix: str = ""):
    return sorted([k for k in REGISTRY if k.startswith(prefix)])
