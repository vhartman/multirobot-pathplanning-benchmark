from typing import Callable, Dict
from importlib import import_module
from functools import partial, wraps
import random
import warnings

import numpy as np
from multi_robot_multi_goal_planning.problems.planning_env import BaseProblem

EnvFactory = Callable[..., BaseProblem]
REGISTRY: Dict[str, EnvFactory] = {}

# Environment prefix -> module and installation hint.
_OPTIONAL_BACKENDS = {
    "comotion": ("comotion.envs", "Install the CoMotion collision package; see the [comotion] extra."),
    "rai": ("rai.envs", "Install the robotic package for RAI environments."),
    "vampmr": ("vampmr.envs", "Install mr_planner_core and meshcat for the active Python interpreter."),
    "pinocchio": ("pinocchio.envs", "Install the project's [pinocchio] extra."),
    "mujoco": ("mujoco.envs", "Install the project's [mujoco] extra."),
    "roboballet": ("mujoco.envs.roboballet.welding", "Install [mujoco-scenes] and pass the external release directory as assets."),
    "cuarms": ("mujoco.envs.cuarms.packing", "Install [mujoco-scenes] and pass the RoboBallet asset directory as assets."),
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


def _deterministic(factory, seed=0):
    """Build under a fixed seed of the global random streams and restore the caller's states afterwards.

    Builders draw from random / np.random (task assignment, KOMO restarts). Without this, an
    environment depended on the caller's RNG state, so each run seed planned a different instance.
    """
    @wraps(factory)
    def build(**kwargs):
        py, npy = random.getstate(), np.random.get_state()
        random.seed(seed)
        np.random.seed(seed)
        try:
            return factory(**kwargs)
        finally:
            random.setstate(py)
            np.random.set_state(npy)
    return build


def register(names):
    """
    names can be:
    - a single string (simple case)
    - a list of (name, kwargs) pairs
    """

    def decorator(factory):
        entries = [(names, {})] if isinstance(names, str) else names
        for name, defaults in entries:
            if name in REGISTRY:
                raise ValueError(f"Duplicate env name: {name}")
            REGISTRY[name] = _deterministic(partial(factory, **defaults))
        return factory

    return decorator

def get_all_environments():
    """Return registered environments, loading available optional backends."""
    for prefix in _OPTIONAL_BACKENDS:
        _load_optional_backend(prefix, required=False)
    return REGISTRY

def get_env_by_name(name: str, **kwargs) -> BaseProblem:
    """Construct a named environment; keyword arguments override preset defaults."""
    prefix = name.partition(".")[0]
    if name not in REGISTRY and prefix in _OPTIONAL_BACKENDS:
        _load_optional_backend(prefix, required=True)
    try:
        factory = REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown environment: {name}") from None
    return factory(**kwargs)


def list_envs(prefix: str = ""):
    return sorted(k for k in get_all_environments() if k.startswith(prefix))
