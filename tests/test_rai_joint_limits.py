"""rai's getJointLimits() reshapes multi-dof joint limits row-wise; rai_base_env.get_joint_limits undoes that."""
import numpy as np
import pytest

ry = pytest.importorskip("robotic")
from multi_robot_multi_goal_planning.problems.rai_base_env import get_joint_limits


def test_multi_dof_joint_limits():
    C = ry.Config()
    C.addFrame("world")
    C.addFrame("a", "world").setJoint(ry.JT.transXY, limits=[-1, 1, -2, 2])
    C.addFrame("b", "world").setJoint(ry.JT.transXYPhi, limits=[-3, 3, -4, 4, -5, 5])
    C.addFrame("c", "world").setJoint(ry.JT.hingeZ, limits=[-6, 6])
    lim = get_joint_limits(C)
    assert np.allclose(lim[0], [-1, -2, -3, -4, -5, -6])
    assert np.allclose(lim[1], [1, 2, 3, 4, 5, 6])


def test_planar_env_samples_inside_limits():
    from multi_robot_multi_goal_planning.problems import get_env_by_name
    env = get_env_by_name("rai.random_2d_no_rot")
    samples = np.array([env.sample_config_uniform_in_limits().state() for _ in range(500)])
    assert np.all(samples >= env.limits[0] - 1e-12) and np.all(samples <= env.limits[1] + 1e-12)
    assert np.all(samples.max(axis=0) - samples.min(axis=0) > 0.5)  # not degenerate
