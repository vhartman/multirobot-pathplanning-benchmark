import pytest
import numpy as np

from multi_robot_multi_goal_planning.problems.planning_env import generate_binary_search_indices
from multi_robot_multi_goal_planning.problems import get_env_by_name

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