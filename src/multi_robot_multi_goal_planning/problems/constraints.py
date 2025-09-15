import numpy as np
from scipy.spatial.transform import Rotation as R

from abc import ABC, abstractmethod

from typing import List, Dict, Optional, Any, Tuple
from numpy.typing import NDArray

from .configuration import Configuration

class Constraint(ABC):
    @abstractmethod
    def is_fulfilled(self, q: Configuration, env) -> bool:
        pass

    # @abstractmethod
    # def get_gradient(self, q, env):
    #     pass

    # @abstractmethod
    # def project_to_manifold(self, q, env):
    #     pass


def get_axes_from_quaternion(quat):
    """
    Returns the x, y, z unit vectors of the frame defined by the quaternion.
    Each axis is a 3D vector in world coordinates.
    """
    rot = R.from_quat(quat)
    rot_matrix = rot.as_matrix()  # 3x3 rotation matrix
    x_axis = rot_matrix[:, 0]
    y_axis = rot_matrix[:, 1]
    z_axis = rot_matrix[:, 2]
    return x_axis, y_axis, z_axis


class TaskSpaceEqualityConstraint(Constraint):
    def __init__(self, frame_name, multipliers, pose, eps=1e-3):
        self.multipliers = multipliers
        self.frame_name = frame_name
        self.constraint_pose = pose
        self.eps = eps

        assert len(self.multipliers) == len(self.constraint_pose)

    def is_fulfilled(self, q: Configuration, env) -> bool:
        frame_pose = env.get_frame_pose(self.frame_name)

        return np.isclose(frame_pose * self.multipliers, self.constraint_pose * self.multipliers, self.eps)


class ConfigurationSpaceEqualityConstraint(Constraint):
    def __init__(self, multipliers, pose, eps=1e-3):
        self.multipliers = multipliers
        self.constraint_pose = pose
        self.eps = eps

        assert len(self.constraint_pose) == len(self.multipliers)

    def is_fulfilled(self, q: Configuration, env) -> bool:
        return np.isclose(q.state() * self.multipliers, self.constraint_pose * self.multipliers, self.eps)


class ConfigurationSpaceRelativeConstraint(Constraint):
    def __init__(self, multipliers, pose, eps=1e-3):
        self.multipliers = multipliers
        self.constraint_pose = pose
        self.eps = eps

        assert len(self.constraint_pose) == len(self.multipliers)

    def is_fulfilled(self, q: Configuration, env) -> bool:
        return np.isclose(q.state() * self.multipliers, self.constraint_pose * self.multipliers, self.eps)


class FrameOrientationConstraint(Constraint):
    """
    Pose of a single frame.
    """

    def __init__(self, frame_name, multipliers, desired_orientation_vector, epsilon):
        self.frame_name = frame_name
        self.multipliers = multipliers
        self.desired_orientation_vector = desired_orientation_vector
        self.epsilon = epsilon

    def is_fulfilled(self, q, env):
        frame_pose = env.get_frame_pose(self.frame_name)

        # get vector from quaternion
        x_axis, y_axis, z_axis = get_axes_from_quaternion(frame_pose[3:])

        return np.dot(self.desired_orientation_vector, z_axis) >= 1 - self.epsilon


class PathConstraint(Constraint):
    """
    Describes a constraint that imposes that we move a long a specified path.
    This is a proxy for a skill.

    Ideally this would have timing/phase information as well, but this is currently not supported by the rest of the framework.
    """

    def __init__(self, frame_name, path, epsilon):
        self.frame_name = frame_name
        self.path = path
        self.epsilon = epsilon

    def is_fulfilled(self, q, env):
        frame_pose = env.get_frame_pose(self.frame_name)
        
        # find closest element on path
        for p in self.path:
            dist = np.linalg.norm(frame_pose - p)


            if dist < self.epsilon:
                return True


def relative_pose(a, b):
    """
    a, b: 7D poses [x, y, z, qw, qx, qy, qz]
    returns: relative pose b_in_a as [x, y, z, qw, qx, qy, qz]
    """
    pa, qa = np.array(a[:3]), np.array(a[3:])
    pb, qb = np.array(b[:3]), np.array(b[3:])

    # rotation matrices
    Ra = R.from_quat([qa[1], qa[2], qa[3], qa[0]])  # scipy expects (x,y,z,w)
    Rb = R.from_quat([qb[1], qb[2], qb[3], qb[0]])

    # relative position
    prel = Ra.inv().apply(pb - pa)

    # relative orientation
    qrel = Ra.inv() * Rb
    qrel = qrel.as_quat()  # (x,y,z,w)
    qrel = np.array([qrel[3], qrel[0], qrel[1], qrel[2]])  # back to (w,x,y,z)

    return np.concatenate([prel, qrel])


class FrameRelativePoseConstraint(Constraint):
    def __init__(self, frame_names: List[str], multipliers, rel_pose, eps: float = 1e-3):
        self.frames = frame_names
        self.multipliers = multipliers
        self.desired_relative_pose = rel_pose
        self.eps = eps

    def is_fulfilled(self, q, env):
        frame_1_pose = env.get_frame_pose(self.frames[0])
        frame_2_pose = env.get_frame_pose(self.frames[1])

        rel_pose = relative_pose(frame_1_pose, frame_2_pose)

        return np.isclose(rel_pose, self.desired_relative_pose, self.eps)