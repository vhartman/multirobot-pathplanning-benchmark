import numpy as np
from scipy.spatial.transform import Rotation as R

from abc import ABC, abstractmethod

from typing import List, Dict, Optional, Any, Tuple
from numpy.typing import NDArray

from .configuration import Configuration

# from .rai_base_env import rai_env
import robotic

class Constraint(ABC):
    @abstractmethod
    def is_fulfilled(self, q: Configuration, mode, env) -> bool:
        """
        Method to check if a constraint is fulfilled.
        Returns boolean only.
        """
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
    rot = R.from_quat(quat, scalar_first=True)
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

    def is_fulfilled(self, q: Configuration, mode, env) -> bool:
        # frame_pose = env.get_frame_pose(self.frame_name)
        if mode is not None:
            env.set_to_mode(mode)
        
        env.C.setJointState(q.state())
        frame_pose = env.C.getFrame(self.frame_name).getPose()

        return all(np.isclose(self.mat @ frame_pose[:, None], self.constraint_pose, self.eps))


    def F(self, q_vec, mode, env):
        if mode is not None:
            env.set_to_mode(mode)
        
        env.C.setJointState(q_vec)

        [y, J] = env.C.eval(robotic.FS.pose, [self.frame_name], self.mat, self.constraint_pose)
        return y
    

    def J(self, q_vec, mode, env):
        if mode is not None:
            env.set_to_mode(mode)
        env.C.setJointState(q_vec)
        
        [y, J] = env.C.eval(robotic.FS.pose, [self.frame_name], self.mat, self.constraint_pose)

        return J

# constraint of the form 
# A * frame_pose <= b
# can be used to e.g. constrain the end effector to a certain pose
class AffineTaskSpaceInequalityConstraint(Constraint):
    def __init__(self, frame_name, projection_matrix, pose):
        self.frame_name = frame_name

        self.mat = projection_matrix
        self.constraint_pose = pose

        assert self.mat.shape[0] == len(self.constraint_pose)
        assert self.mat.shape[1] == 7

    def is_fulfilled(self, q: Configuration, mode, env) -> bool:
        # frame_pose = env.get_frame_pose(self.frame_name)
        if mode is not None:
            env.set_to_mode(mode)
        env.C.setJointState(q.state())
        frame_pose = env.C.getFrame(self.frame_name).getPose()

        return all(self.mat @ frame_pose[:, None] <  self.constraint_pose)


def relative_pose(a, b):
    """
    a, b: 7D poses [x, y, z, qw, qx, qy, qz]
    returns: relative pose b_in_a as [x, y, z, qw, qx, qy, qz]
    """
    pa, qa = np.array(a[:3]), np.array(a[3:])
    pb, qb = np.array(b[:3]), np.array(b[3:])

    # rotation matrices
    Ra = R.from_quat(qa, scalar_first=True)
    Rb = R.from_quat(qb, scalar_first=True)

    # relative position
    prel = Ra.inv().apply(pb - pa)

    # relative orientation
    qrel = Ra.inv() * Rb
    qrel = qrel.as_quat(scalar_first=True)  # (x,y,z,w)

    return np.concatenate([prel, qrel])


# constraint of the form
# A * (frame_pose_1 - frame_pose_2) = b
# TODO: possibly change to A * frame_pose_1 - A_2 * frame_pose_2 = b
class RelativeAffineTaskSpaceEqualityConstraint(Constraint):
    def __init__(self, frame_names: List[str], mat, rel_pose, eps: float = 1e-3):
        self.frames = frame_names
        self.mat = mat
        self.desired_relative_pose = rel_pose
        self.eps = eps

        assert self.mat.shape[0] == len(self.desired_relative_pose)
        assert self.mat.shape[1] == 7

    def is_fulfilled(self, q, mode, env):
        # frame_1_pose = env.get_frame_pose(q, self.frames[0])
        # frame_2_pose = env.get_frame_pose(q, self.frames[1])

        if mode is not None:
            env.set_to_mode(mode)
        env.C.setJointState(q.state())
        frame_1_pose = env.C.getFrame(self.frames[0]).getPose()

        env.C.setJointState(q.state())
        frame_2_pose = env.C.getFrame(self.frames[1]).getPose()

        rel_pose = relative_pose(frame_1_pose, frame_2_pose)

        return all(np.isclose(self.mat @ rel_pose[:, None], self.desired_relative_pose, self.eps))


# constraint of the form
# A * (frame_pose_1 - frame_pose_2) = b
# TODO: possibly change to A * frame_pose_1 - A_2 * frame_pose_2 = b
class RelativeAffineTaskSpaceInequalityConstraint(Constraint):
    def __init__(self, frame_names: List[str], mat, rel_pose):
        self.frames = frame_names
        self.mat = mat
        self.desired_relative_pose = rel_pose

        # could possibly change in the future.
        assert len(frame_names) == 2

        assert self.mat.shape[0] == len(self.desired_relative_pose)
        assert self.mat.shape[1] == 7

    def is_fulfilled(self, q, mode, env):
        # frame_1_pose = env.get_frame_pose(q, self.frames[0])
        # frame_2_pose = env.get_frame_pose(q, self.frames[1])

        if mode is not None:
            env.set_to_mode(mode)
        
        env.C.setJointState(q.state())
        frame_1_pose = env.C.getFrame(self.frames[0]).getPose()

        env.C.setJointState(q.state())
        frame_2_pose = env.C.getFrame(self.frames[1]).getPose()

        rel_pose = relative_pose(frame_1_pose, frame_2_pose)

        return all(self.mat @ rel_pose[:, None] <= self.desired_relative_pose)


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

    def is_fulfilled(self, q: Configuration, mode, env) -> bool:
        return all(np.isclose(self.mat @ q.state()[:, None], self.constraint_pose, self.eps))

# constraint of the form 
# A * q <= b
# can b eused to e.g. constrain the configurtion space pose to a certain value
# or to ensure that two values in the pose are the same
class AffineConfigurationSpaceInequalityConstraint(Constraint):
    def __init__(self, projection_matrix, pose):
        self.mat = projection_matrix
        self.constraint_pose = pose

        assert self.mat.shape[0] == len(self.constraint_pose)

    def is_fulfilled(self, q: Configuration, mode, env) -> bool:
        return all(self.mat @ q.state()[:, None] < self.constraint_pose)


# This might currently still be a bit overcomplicated?
class AffineFrameOrientationConstraint(Constraint):
    def __init__(self, frame_name, vector, desired_orientation_vector, epsilon):
        self.frame_name = frame_name
        self.desired_orientation_vector = desired_orientation_vector
        self.epsilon = epsilon
        self.vector = vector

        assert self.vector in ["x", "y", "z"]

    def is_fulfilled(self, q, mode, env):
        # TODO: make applicable to all envs
        if mode is not None:
            env.set_to_mode(mode)
        env.C.setJointState(q.state())
        frame_pose = env.C.getFrame(self.frame_name).getPose()

        # get vector from quaternion
        x_axis, y_axis, z_axis = get_axes_from_quaternion(frame_pose[3:])

        if self.vector == "x":
            axis_to_check = x_axis
        elif self.vector == "y":
            axis_to_check = y_axis
        elif self.vector == "z":
            axis_to_check = z_axis
        else:
            raise ValueError
        
        print(axis_to_check)
        
        return all(np.isclose(axis_to_check, self.desired_orientation_vector, atol=self.epsilon))


    def J(self, q, mode, env):
        if mode is not None:
            env.set_to_mode(mode)
        
        env.C.setJointState(q)
        
        fs = robotic.FS.vectorX

        if self.vector == "y":
            fs = robotic.FS.vectorY
        elif self.vector == "z":
            fs = robotic.FS.vectorZ
        else:
            raise ValueError
        
        [y, J] = env.C.eval(fs, [self.frame_name], 1, self.desired_orientation_vector)
        return J

    def F(self, q, mode, env):
        if mode is not None:
            env.set_to_mode(mode)
        
        env.C.setJointState(q)

        fs = robotic.FS.vectorX
        if self.vector == "y":
            fs = robotic.FS.vectorY
        elif self.vector == "z":
            fs = robotic.FS.vectorZ
        else:
            raise ValueError
        
        [y, J] = env.C.eval(fs, [self.frame_name], 1, self.desired_orientation_vector)
        return y

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

    def is_fulfilled(self, q, mode, env):
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

    def is_fulfilled(self, q, mode, env):
        # find closest element on path
        for p in self.path:
            dist = np.linalg.norm(self.mat @ q - p)

            if dist < self.epsilon:
                return True