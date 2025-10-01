import pytest

import numpy as np

from multi_robot_multi_goal_planning.problems.configuration import NpConfiguration

from multi_robot_multi_goal_planning.problems.constraints import (
    AffineConfigurationSpaceEqualityConstraint,
    AffineConfigurationSpaceInequalityConstraint,
)


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
    assert False


def test_task_space_relative_eq_constraint():
    assert False


def test_affine_frame_orientation_constraint():
    assert False


def test_task_space_path_constraint():
    assert False


def test_conf_space_path_constraint():
    assert False
