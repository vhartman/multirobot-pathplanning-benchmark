import pytest

import numpy as np

from multi_robot_multi_goal_planning.problems.configuration import NpConfiguration

from multi_robot_multi_goal_planning.problems.constraints import (
    AffineConfigurationSpaceEqualityConstraint,
    AffineConfigurationSpaceInequalityConstraint,
    AffineFrameOrientationConstraint,
    RelativeAffineTaskSpaceEqualityConstraint,
    AffineTaskSpaceEqualityConstraint,
    relative_pose,
)

from multi_robot_multi_goal_planning.problems import get_env_by_name


def test_conf_space_eq_constraint():
    A = np.zeros((1, 10))
    A[0, 0] = 1
    A[0, 5] = -1
    b = np.zeros((1, 1))
    constraint = AffineConfigurationSpaceEqualityConstraint(A, b)

    q = np.zeros((10))
    q[0] = 5
    q[5] = 5
    assert constraint.is_fulfilled(NpConfiguration.from_list([q]), None)

    q2 = np.zeros((10))
    q2[0] = 5
    assert not constraint.is_fulfilled(NpConfiguration.from_list([q2]), None)


def test_conf_space_ineq_constraint():
    A = np.zeros((1, 10))
    A[0, 0] = 1
    b = np.zeros((1, 1))
    constraint = AffineConfigurationSpaceInequalityConstraint(A, b)

    q = np.zeros((10))
    q[0] = 5
    assert not constraint.is_fulfilled(NpConfiguration.from_list([q]), None)

    q = np.zeros((10))
    q[0] = -1
    assert constraint.is_fulfilled(NpConfiguration.from_list([q]), None)


def test_task_space_eq_constraint():
    env = get_env_by_name("rai.piano")

    A = np.zeros((1, 7))
    A[0] = 1  # constraining x

    b = np.zeros((1, 1))

    constraint = AffineTaskSpaceEqualityConstraint("a1", A, b, 1e-3)

    q = env.get_start_pos()
    q[0][0] = 0
    assert not constraint.is_fulfilled(q, env)

    q = env.get_start_pos()
    q[0][0] = 1
    assert not constraint.is_fulfilled(q, env)


def test_task_space_relative_eq_constraint():
    env = get_env_by_name("rai.piano")

    p1 = env.C.getFrame("a1").getPose()
    p2 = env.C.getFrame("a2").getPose()

    A = np.eye(7)
    b = relative_pose(p1, p2)[:, None]

    constraint = RelativeAffineTaskSpaceEqualityConstraint(["a1", "a2"], A, b, 1e-3)

    q = env.get_start_pos()
    assert constraint.is_fulfilled(q, env)

    q1 = env.get_start_pos()
    q1[0][0] += 1
    q1[1][0] += 1
    assert constraint.is_fulfilled(q1, env)

    q2 = env.get_start_pos()
    q2[0][1] -= 1
    q2[1][1] -= 1
    assert constraint.is_fulfilled(q2, env)

    q3 = env.get_start_pos()
    q3[0][1] -= 1
    q3[1][1] += 1
    assert not constraint.is_fulfilled(q2, env)


def test_affine_frame_orientation_constraint():
    env = get_env_by_name("rai.piano")

    constraint = AffineFrameOrientationConstraint("a1", "z", np.array([0, 0, 1]), 1e-3)

    q = env.get_start_pos()
    assert constraint.is_fulfilled(q, env)


# def test_task_space_path_constraint():
#     assert False


# def test_conf_space_path_constraint():
#     assert False
