from typing import Callable, Dict
from importlib import import_module
import warnings
from multi_robot_multi_goal_planning.problems.planning_env import BaseProblem

EnvFactory = Callable[[], BaseProblem]
REGISTRY: Dict[str, EnvFactory] = {}

# Environment prefix -> module and installation hint.
_OPTIONAL_BACKENDS = {
    "vampmr": ("mr_vamp_env", "Install mr_planner_core and meshcat for the active Python interpreter."),
    "pinocchio": ("pinocchio_env", "Install the project's [pinocchio] extra."),
    "mujoco": ("mujoco_env", "Install the project's [mujoco] extra."),
}
_BACKEND_ERRORS: Dict[str, ImportError] = {}


def _load_optional_backend(prefix, *, required):
    module, hint = _OPTIONAL_BACKENDS[prefix]
    error = _BACKEND_ERRORS.get(prefix)
    if error is None:
        registered = REGISTRY.copy()
        try:
            import_module(f"multi_robot_multi_goal_planning.problems.{module}")
            return
        except ImportError as exc:
            # Do not expose registrations from a partially imported backend.
            REGISTRY.clear()
            REGISTRY.update(registered)
            error = _BACKEND_ERRORS[prefix] = exc
            if not required:
                warnings.warn(
                    f"Backend {prefix!r} is unavailable: {exc}. {hint}",
                    RuntimeWarning, stacklevel=3,
                )
    if required:
        raise ImportError(f"Cannot load backend {prefix!r}: {error}. {hint}") from error


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
    """Return registered environments, loading available optional backends."""
    for prefix in _OPTIONAL_BACKENDS:
        _load_optional_backend(prefix, required=False)
    return REGISTRY

def get_env_by_name(name: str) -> BaseProblem:
    prefix = name.partition(".")[0]
    if name not in REGISTRY and prefix in _OPTIONAL_BACKENDS:
        _load_optional_backend(prefix, required=True)
    try:
        factory = REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown environment: {name}") from None
    return factory()


def list_envs(prefix: str = ""):
    return sorted(k for k in get_all_environments() if k.startswith(prefix))
