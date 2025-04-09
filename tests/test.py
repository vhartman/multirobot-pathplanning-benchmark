import pytest

import numpy as np

from multi_robot_multi_goal_planning.problems.planning_env import (
    generate_binary_search_indices,
)
from multi_robot_multi_goal_planning.problems import get_env_by_name
from multi_robot_multi_goal_planning.problems.configuration import NpConfiguration
from multi_robot_multi_goal_planning.problems.planning_env import State

from multi_robot_multi_goal_planning.planners.joint_prm_planner import joint_prm_planner
from multi_robot_multi_goal_planning.planners.planner_rrtstar import RRTstar

from multi_robot_multi_goal_planning.planners.termination_conditions import (
    RuntimeTerminationCondition,
)


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, (0,)),
        (2, (0, 1)),
        (3, (1, 0, 2)),
        (4, (1, 0, 2, 3)),
        (5, (2, 0, 3, 1, 4)),
    ],
)
def test_binary_indices(n, expected):
    assert generate_binary_search_indices(n) == expected


def test_edge_checking():
    env = get_env_by_name("abstract_test")

    q1 = NpConfiguration(np.array([-1, 0, 1, 1]), env.start_pos.array_slice)
    q2 = NpConfiguration(np.array([-1, 1, 1, 0]), env.start_pos.array_slice)

    is_collision_free = env.is_edge_collision_free(q1, q2, env.start_mode)

    assert is_collision_free


def test_edge_checking_resolution(mocker):
    env = get_env_by_name("abstract_test")

    q1 = NpConfiguration(np.array([-1, 0, 1, 1]), env.start_pos.array_slice)
    q2 = NpConfiguration(np.array([-1, 1, 1, 0]), env.start_pos.array_slice)

    mock = mocker.patch.object(env, "is_collision_free", return_value=True)

    env.is_edge_collision_free(
        q1, q2, env.start_mode, resolution=0.5, include_endpoints=True
    )
    assert mock.call_count == 3

    mock.reset_mock()
    env.is_edge_collision_free(
        q1, q2, env.start_mode, resolution=0.5, include_endpoints=False
    )
    assert mock.call_count == 1

    mock.reset_mock()
    env.is_edge_collision_free(
        q1, q2, env.start_mode, resolution=0.1, include_endpoints=False
    )
    assert mock.call_count == 9

    mock.reset_mock()
    env.is_edge_collision_free(
        q1, q2, env.start_mode, resolution=0.1, include_endpoints=True
    )
    assert mock.call_count == 11


def test_path_collision_checking(mocker):
    env = get_env_by_name("abstract_test")

    q1 = NpConfiguration(np.array([-1, 0, 1, 1]), env.start_pos.array_slice)
    q2 = NpConfiguration(np.array([-1, 1, 1, 0]), env.start_pos.array_slice)
    q3 = NpConfiguration(np.array([-1, 2, 1, -1]), env.start_pos.array_slice)

    s1 = State(q1, env.start_mode)
    s2 = State(q2, env.start_mode)
    s3 = State(q3, env.start_mode)

    is_collision_free = env.is_path_collision_free([s1, s2, s3], resolution=0.5)
    assert is_collision_free

    mock = mocker.patch.object(env, "is_collision_free", return_value=True)

    env.is_path_collision_free([s1, s2, s3], resolution=0.5, check_edges_in_order=True)
    assert mock.call_count == 5

    mock.reset_mock()
    env.is_path_collision_free([s1, s2, s3], resolution=0.5)
    assert mock.call_count == 5

    mock.reset_mock()
    env.is_path_collision_free([s1, s2, s3], resolution=0.1, check_edges_in_order=True)
    assert mock.call_count == 21

    mock.reset_mock()
    env.is_path_collision_free([s1, s2, s3], resolution=0.1)
    assert mock.call_count == 21


@pytest.mark.parametrize(
    "planner_fn",
    [
        lambda env, ptc: joint_prm_planner(env, ptc=ptc, optimize=False),
        lambda env, ptc: RRTstar(env, ptc=ptc).Plan(False),
    ],
)
def test_planner_on_abstract_env(planner_fn):
    env = get_env_by_name("abstract_test")
    ptc = RuntimeTerminationCondition(10)

    path, _ = planner_fn(env, ptc)

    assert path is not None

    assert np.array_equal(path[0].q.state(), env.start_pos.state())
    assert env.is_terminal_mode(path[-1].mode)
    assert env.is_valid_plan(path)


@pytest.mark.parametrize(
    "planner_fn",
    [
        lambda env, ptc: joint_prm_planner(env, ptc=ptc, optimize=False),
        lambda env, ptc: RRTstar(env, ptc=ptc).Plan(False),
    ],
)
def test_planner_on_hallway_dependency_env(planner_fn):
    env = get_env_by_name("other_hallway_dep")
    ptc = RuntimeTerminationCondition(10)

    path, _ = planner_fn(env, ptc)

    assert path is not None

    assert np.array_equal(path[0].q.state(), env.start_pos.state())
    assert env.is_terminal_mode(path[-1].mode)
    assert env.is_valid_plan(path)
