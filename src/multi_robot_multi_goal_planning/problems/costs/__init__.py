"""Construct an objective from experiment configuration."""

from .joint_space import JointSpaceLength, JointNormGeometry
from .base import MotionCost, configuration, coordinates, endpoint_pairs


def make_cost(env, settings):
    options = dict(settings)
    kind = options.pop("type", "joint")
    if kind == "joint":
        return JointSpaceLength(**options)
    if kind == "workspace":
        from .workspace import workspace_cost
        frames = options.pop("frames", None)
        if frames is None:
            raise ValueError("Workspace cost requires explicit frame names")
        return workspace_cost(env, frames, **options)
    raise ValueError(f"Unknown motion cost: {kind}")
