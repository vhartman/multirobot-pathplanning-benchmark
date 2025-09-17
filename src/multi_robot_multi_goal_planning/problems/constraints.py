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


# constraint of the form 
# A * frame_pose = b
# can be used to e.g. constrain the end effector to a certain pose
class AffineTaskSpaceEqualityConstraint(Constraint):
    def __init__(self, frame_name, projection_matrix, pose, eps=1e-3):
        self.frame_name = frame_name

        self.mat = projection_matrix
        self.constraint_pose = pose
        self.eps = eps

        assert self.mat.shape[0] == len(self.constraint_pose)
        assert self.mat.shape[1] == 7

    def is_fulfilled(self, q: Configuration, env) -> bool:
        frame_pose = env.get_frame_pose(self.frame_name)

        return np.isclose(self.mat @ frame_pose, self.constraint_pose, self.eps)


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


# constraint of the form
# A * (frame_pose_1 - frame_pose_2) = b
class RelativeAffineTaskSpaceEqualityConstraint(Constraint):
    def __init__(self, frame_names: List[str], mat, rel_pose, eps: float = 1e-3):
        self.frames = frame_names
        self.mat = mat
        self.desired_relative_pose = rel_pose
        self.eps = eps

        assert self.mat.shape[0] == len(self.desired_relative_pose)
        assert self.mat.shape[1] == 7

    def is_fulfilled(self, q, env):
        frame_1_pose = env.get_frame_pose(self.frames[0])
        frame_2_pose = env.get_frame_pose(self.frames[1])

        rel_pose = relative_pose(frame_1_pose, frame_2_pose)

        return np.isclose(self.mat @ rel_pose, self.desired_relative_pose, self.eps)

# constraint of the form 
# A * q = b
# can b eused to e.g. constrain the configurtion space pose to a certain value
# or to ensure that two values in the pose are the same
class AffineConfigurationSpaceEqualityConstraint(Constraint):
    def __init__(self, projection_matrix, pose, eps=1e-3):
        self.mat = projection_matrix
        self.constraint_pose = pose
        self.eps = eps

        assert self.mat.shape[0] == len(self.constraint_pose)

    def is_fulfilled(self, q: Configuration, env) -> bool:
        return np.isclose(self.mat @ q.state(), self.constraint_pose, self.eps)


class AffineFrameOrientationConstraint(Constraint):
    def __init__(self, frame_name, mat, desired_orientation_vector, epsilon):
        self.frame_name = frame_name
        self.mat = mat
        self.desired_orientation_vector = desired_orientation_vector
        self.epsilon = epsilon

        raise NotImplementedError

    def is_fulfilled(self, q, env):
        frame_pose = env.get_frame_pose(self.frame_name)

        # get vector from quaternion
        x_axis, y_axis, z_axis = get_axes_from_quaternion(frame_pose[3:])

        raise NotImplementedError
    

# projects the pose of a frame to a path 
class TaskSpacePathConstraint(Constraint):
    """
    Describes a constraint that imposes that we move a long a specified path.
    This is a proxy for a skill.

    Ideally this would have timing/phase information as well, but this is currently not supported by the rest of the framework.
    """

    def __init__(self, frame_name, path, mat, epsilon):
        self.frame_name = frame_name
        self.path = path
        self.mat = mat
        self.epsilon = epsilon

    def is_fulfilled(self, q, env):
        frame_pose = env.get_frame_pose(self.frame_name)
        
        # find closest element on path
        for p in self.path:
            dist = np.linalg.norm(self.mat @ frame_pose - p)

            if dist < self.epsilon:
                return True
            

# projects the pose of a frame to a path 
class ConfigurationSpacePathConstraint(Constraint):
    """
    Describes a constraint that imposes that we move a long a specified path.
    This is a proxy for a skill.

    Ideally this would have timing/phase information as well, but this is currently not supported by the rest of the framework.
    """

    def __init__(self, frame_name, path, mat, epsilon):
        self.path = path
        self.mat = mat
        self.epsilon = epsilon

    def is_fulfilled(self, q, env):
        # find closest element on path
        for p in self.path:
            dist = np.linalg.norm(self.mat @ q - p)

            if dist < self.epsilon:
                return True