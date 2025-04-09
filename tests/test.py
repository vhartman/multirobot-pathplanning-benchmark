import pytest
import numpy as np
import random

from multi_robot_multi_goal_planning.problems.util import generate_binary_search_indices
from multi_robot_multi_goal_planning.problems import get_env_by_name

from multi_robot_multi_goal_planning.planners.joint_prm_planner import joint_prm_planner
from multi_robot_multi_goal_planning.planners.planner_rrtstar import RRTstar

from multi_robot_multi_goal_planning.planners.termination_conditions import (
    RuntimeTerminationCondition,
)
from multi_robot_multi_goal_planning.planners.shortcutting import robot_mode_shortcut


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
    "planner_fn_no_shurtcutting",
    [
        lambda env, ptc: RRTstar(env, ptc=ptc, shortcutting=False).Plan(optimize = False),
    ],
)

@pytest.mark.parametrize("run_idx", range(3))
def test_shortcutting(planner_fn_no_shurtcutting, run_idx):
    env = get_env_by_name("hallway")
    ptc = RuntimeTerminationCondition(10)

    path, _ = planner_fn_no_shurtcutting(env, ptc)
    assert path is not None

    print(run_idx)
    seed = run_idx

    shortcut_path_checked_edge_in_order, (cost1, time1) = robot_mode_shortcut(
        env,
        path,
        550,
        resolution=env.collision_resolution,
        tolerance=env.collision_tolerance,
        check_edges_in_order=True, 
        seed=seed
    )

    shortcut_path_checked_edge_not_in_order, (cost2, time2) = robot_mode_shortcut(
        env,
        path,
        550,
        resolution=env.collision_resolution,
        tolerance=env.collision_tolerance,
        check_edges_in_order=False,
        seed=seed
    )

    list1 = [s.q.state() for s in shortcut_path_checked_edge_in_order]
    list2 = [s.q.state() for s in shortcut_path_checked_edge_not_in_order]

    assert all(np.allclose(a1, a2) for a1, a2 in zip(list1, list2))
    assert cost1 == cost2
    assert len(shortcut_path_checked_edge_in_order) == len(shortcut_path_checked_edge_not_in_order)
    # assert time2[-1] < time1[-1]


