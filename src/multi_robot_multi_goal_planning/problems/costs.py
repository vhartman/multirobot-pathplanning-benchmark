"""Construct an objective from experiment configuration."""

from .core.cost import JointSpaceLength


def make_cost(env, settings):
    options = dict(settings)
    kind = options.pop("type", "joint")
    if kind == "joint":
        return JointSpaceLength(**options)
    if kind == "workspace":
        from .rai.workspace_cost import workspace_cost
        if not hasattr(env, "C_orig") or not hasattr(env, "robot_joints"):
            raise ValueError("The configured workspace FK adapter requires a RAI environment")
        frames = options.pop("frames", None)
        if frames is None:
            raise ValueError("Workspace cost requires explicit frame names")
        return workspace_cost(env, frames, **options)
    raise ValueError(f"Unknown motion cost: {kind}")
