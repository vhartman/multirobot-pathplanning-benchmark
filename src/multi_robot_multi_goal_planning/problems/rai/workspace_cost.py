"""Workspace objectives using a private RAI forward-kinematics configuration."""

import numpy as np
import robotic as ry

from ..core.workspace_cost import WorkspaceLength


class RaiFramePositions:
    """One frame per logical robot, in env.robots order; no attachment changes."""

    def __init__(self, env, frame_names):
        self.frame_names = tuple(frame_names)
        if len(self.frame_names) != len(env.robots):
            raise ValueError("Specify one robot-fixed frame per robot, in env.robots order")
        self.joint_names = tuple(j for r in env.robots for j in env.robot_joints[r])
        self.config = ry.Config()
        self.config.addConfigurationCopy(env.C_orig)
        changing = {task.frames[1] for task in env.tasks
                    if task.frames and len(task.frames) > 1 and (task.type or task.side_effect)}
        for name in self.frame_names:
            frame = self.config.getFrame(name)
            if frame is None:
                raise ValueError(f"Unknown workspace frame: {name}")
            while frame is not None:
                if frame.name in changing:
                    raise ValueError(f"Workspace frame {name} depends on a task-modified frame")
                frame = frame.getParent()

    def __call__(self, q):
        self.config.setJointState(q, list(self.joint_names))
        return np.array([self.config.getFrame(name).getPosition() for name in self.frame_names])

    def __deepcopy__(self, memo):
        other = type(self).__new__(type(self))
        memo[id(self)] = other
        other.frame_names, other.joint_names = self.frame_names, self.joint_names
        other.config = ry.Config()
        other.config.addConfigurationCopy(self.config)
        return other


def workspace_cost(env, frame_names, **kwargs):
    """Build a position-travel objective without changing env's live collision scene."""
    return WorkspaceLength(RaiFramePositions(env, frame_names), **kwargs)
