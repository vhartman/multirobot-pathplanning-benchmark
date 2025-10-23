import pytest

import numpy as np
import copy

from multi_robot_multi_goal_planning.problems.configuration import NpConfiguration

from multi_robot_multi_goal_planning.problems.constraints import (
    AffineConfigurationSpaceEqualityConstraint,
    AffineConfigurationSpaceInequalityConstraint,
    AffineFrameOrientationConstraint,
    RelativeAffineTaskSpaceEqualityConstraint,
    RelativeAffineTaskSpaceInequalityConstraint,
    AffineTaskSpaceEqualityConstraint,
    AffineTaskSpaceInequalityConstraint,
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
    assert constraint.is_fulfilled(NpConfiguration.from_list([q]), None, None)

    q2 = np.zeros((10))
    q2[0] = 5
    assert not constraint.is_fulfilled(NpConfiguration.from_list([q2]), None, None)


def test_conf_space_ineq_constraint():
    A = np.zeros((1, 10))
    A[0, 0] = 1
    b = np.zeros((1, 1))
    constraint = AffineConfigurationSpaceInequalityConstraint(A, b)

    q = np.zeros((10))
    q[0] = 5
    assert not constraint.is_fulfilled(NpConfiguration.from_list([q]), None, None)

    q = np.zeros((10))
    q[0] = -1
    assert constraint.is_fulfilled(NpConfiguration.from_list([q]), None, None)


def test_task_space_eq_constraint():
    env = get_env_by_name("rai.piano")

    A = np.zeros((1, 7))
    A[0, 0] = 1  # constraining x

    b = np.zeros((1, 1))

    constraint = AffineTaskSpaceEqualityConstraint("a1", A, b, 1e-3)

    q = env.get_start_pos()
    q[0][0] = 0
    assert constraint.is_fulfilled(q, None, env)

    q = env.get_start_pos()
    q[0][0] = 1
    assert not constraint.is_fulfilled(q, None, env)

    residual = constraint.F(q.state(), None, env)

    assert residual[0] == 1

    jac = constraint.J(q.state(), None, env)

    jac_analytical = np.zeros((1, 6))
    jac_analytical[0, 0] = 1

    assert np.isclose(jac, jac_analytical).all()

    # nonzero version
    b = np.ones((1, 1))

    non_zero_constraint = AffineTaskSpaceEqualityConstraint("a1", A, b, 1e-3)

    q = env.get_start_pos()
    q[0][0] = 0
    assert not non_zero_constraint.is_fulfilled(q, None, env)

    q = env.get_start_pos()
    q[0][0] = 1
    assert non_zero_constraint.is_fulfilled(q, None, env)

    residual = non_zero_constraint.F(q.state(), None, env)

    assert residual[0] == 0

    jac = non_zero_constraint.J(q.state(), None, env)

    jac_analytical = np.zeros((1, 6))
    jac_analytical[0, 0] = 1

    assert np.isclose(jac, jac_analytical).all()


def test_task_space_ineq_constraint():
    env = get_env_by_name("rai.piano")

    A = np.zeros((2, 7))
    A[0, 0] = 1  # constraining x
    A[0, 1] = 1  # constraining y

    b = np.zeros((2, 1))

    constraint = AffineTaskSpaceInequalityConstraint("a1", A, b)

    q = env.get_start_pos()
    q[0][0] = -1
    q[0][1] = 0
    assert not constraint.is_fulfilled(q, None, env)

    q = env.get_start_pos()
    q[0][0] = 1
    q[0][1] = -1
    assert not constraint.is_fulfilled(q, None, env)


def test_task_space_relative_eq_constraint():
    env = get_env_by_name("rai.piano")

    p1 = env.C.getFrame("a1").getPose()
    p2 = env.C.getFrame("a2").getPose()

    A = np.eye(7)
    b = relative_pose(p1, p2)[:, None]

    constraint = RelativeAffineTaskSpaceEqualityConstraint(["a1", "a2"], A, b, 1e-3)

    q = env.get_start_pos()
    assert constraint.is_fulfilled(q, None, env)

    q1 = env.get_start_pos()
    q1[0][0] += 1
    q1[1][0] += 1
    assert constraint.is_fulfilled(q1, None, env)

    q2 = env.get_start_pos()
    q2[0][1] -= 1
    q2[1][1] -= 1
    assert constraint.is_fulfilled(q2, None, env)

    q3 = env.get_start_pos()
    q3[0][1] -= 1
    q3[1][1] += 1
    assert not constraint.is_fulfilled(q2, None, env)


def test_task_space_relative_ineq_constraint():
    env = get_env_by_name("rai.piano")

    p1 = env.C.getFrame("a1").getPose()
    p2 = env.C.getFrame("a2").getPose()

    A = np.eye(7)
    b = relative_pose(p1, p2)[:, None]

    constraint = RelativeAffineTaskSpaceInequalityConstraint(["a1", "a2"], A, b)

    q = env.get_start_pos()
    assert constraint.is_fulfilled(q, None, env)

    q1 = env.get_start_pos()
    q1[0][0] += 1
    q1[1][0] += 0
    assert constraint.is_fulfilled(q1, None, env)

    q2 = env.get_start_pos()
    q2[0][1] -= 0
    q2[1][1] -= 1
    assert constraint.is_fulfilled(q2, None, env)

    q3 = env.get_start_pos()
    q3[0][1] -= 1
    q3[1][1] += 1
    assert not constraint.is_fulfilled(q2, None, env)


def test_affine_frame_orientation_constraint_simple():
    env = get_env_by_name("rai.piano")

    constraint = AffineFrameOrientationConstraint("a1", "z", np.array([0, 0, 1]), 1e-3)

    q = env.get_start_pos()
    assert constraint.is_fulfilled(q, None, env)

    residual = constraint.F(q.state(), None, env)

    assert residual[0] == 0.

    jac_analytical = np.zeros((1, 6))    
    jac = constraint.J(q.state(), None, env)

    assert np.isclose(jac, jac_analytical).all()


def test_affine_frame_orientation_constraint_bottle():
    env = get_env_by_name("rai.arm_ee_pose")

    r1_constraint = env.tasks[1].constraints[0]
    r1_pick_pose = env.tasks[0].goal.sample(None)
    r1_place_pose = env.tasks[1].goal.sample(None)

    r2_constraint = env.tasks[3].constraints[0]
    r2_pick_pose = env.tasks[2].goal.sample(None)
    r2_place_pose = env.tasks[3].goal.sample(None)

    ##############
    # robot 1 pick
    ##############
    q = env.get_start_pos()
    assert r1_constraint.is_fulfilled(q, None, env)

    residual = r1_constraint.F(q.state(), None, env)
    assert residual[2] == 0
    
    r1_complete_pick_pose = copy.deepcopy(env.get_start_pos())
    r1_complete_pick_pose[0] = r1_pick_pose

    r1_pick_mode = env.get_next_modes(r1_complete_pick_pose, env.get_start_mode())[0]

    assert r1_constraint.is_fulfilled(r1_complete_pick_pose, r1_pick_mode, env)

    residual = r1_constraint.F(r1_complete_pick_pose.state(), r1_pick_mode, env)
    assert residual[2] == 0

    # ensure that this constraint is not fulfilled at the start pose afte rhaving picked
    assert not r1_constraint.is_fulfilled(env.get_start_pos(), r1_pick_mode, env)

    jac = r1_constraint.J(env.get_start_pos().state(), r1_pick_mode, env)
    residual = r1_constraint.F(env.get_start_pos().state(), r1_pick_mode, env)

    # ensure that the jacobian for robot 1 is nonzero, and it is zero for robot 2
    assert np.linalg.norm(jac[:, :6]) > 0
    assert np.linalg.norm(jac[:, 6:]) == 0

    assert np.linalg.norm(residual) > 0

    ##############
    # robot 2 pick
    ##############
    r2_complete_pick_pose = copy.deepcopy(env.get_start_pos())
    r2_complete_pick_pose[1] = r2_pick_pose

    r2_pick_mode = env.get_next_modes(r2_complete_pick_pose, r1_pick_mode)[0]

    r2_constraint.is_fulfilled(r2_complete_pick_pose, r2_pick_mode, env)

    assert r2_constraint.is_fulfilled(r2_complete_pick_pose, r2_pick_mode, env)

    residual = r2_constraint.F(r2_complete_pick_pose.state(), r2_pick_mode, env)
    assert residual[2] == 0

    assert not r1_constraint.is_fulfilled(env.get_start_pos(), r2_pick_mode, env)

    jac = r2_constraint.J(env.get_start_pos().state(), r2_pick_mode, env)
    residual = r2_constraint.F(env.get_start_pos().state(), r2_pick_mode, env)

    # ensure that the jacobian for robot 1 is nonzero, and it is zero for robot 2
    assert np.linalg.norm(jac[:, :6]) == 0
    assert np.linalg.norm(jac[:, 6:]) > 0

    assert np.linalg.norm(residual) > 0

    ##############
    # R1 place
    ##############
    r1_complete_place_pose = copy.deepcopy(env.get_start_pos())
    r1_complete_place_pose[0] = r1_place_pose

    r1_place_mode = env.get_next_modes(r1_complete_place_pose, r2_pick_mode)[0]
    r1_constraint.is_fulfilled(r1_complete_place_pose, r1_place_mode, env)

    # test that constraint is fulfilled
    assert r1_constraint.is_fulfilled(r1_complete_place_pose, r1_place_mode, env)

    residual = r1_constraint.F(r1_complete_place_pose.state(), r1_place_mode, env)
    assert abs(residual[2]) < 1e-6

    # test that jac should be zero after relinking
    jac = r1_constraint.J(r1_complete_place_pose.state(), r1_place_mode, env)
    assert np.linalg.norm(jac) == 0

    ##############
    # R2 place
    ##############
    r2_complete_place_pose = copy.deepcopy(env.get_start_pos())
    r2_complete_place_pose[1] = r2_place_pose
    r2_place_mode = env.get_next_modes(r2_complete_place_pose, r1_place_mode)[0]

    r2_constraint.is_fulfilled(r2_complete_place_pose, r2_place_mode, env)

    assert r2_constraint.is_fulfilled(r2_complete_place_pose, r2_place_mode, env)

    residual = r2_constraint.F(r2_complete_place_pose.state(), r2_place_mode, env)
    assert abs(residual[2]) < 1e-6

    jac = r2_constraint.J(r2_complete_place_pose.state(), r2_place_mode, env)

    assert np.linalg.norm(jac) == 0


# def test_task_space_path_constraint():
#     assert False


# def test_conf_space_path_constraint():
#     assert False
