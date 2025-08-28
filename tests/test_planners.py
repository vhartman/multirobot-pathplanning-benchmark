import pytest

import numpy as np

from multi_robot_multi_goal_planning.problems import get_env_by_name

from multi_robot_multi_goal_planning.planners.composite_prm_planner import CompositePRM, CompositePRMConfig
from multi_robot_multi_goal_planning.planners.planner_rrtstar import RRTstar, BaseRRTConfig
from multi_robot_multi_goal_planning.planners.planner_birrtstar import (
    BidirectionalRRTstar,
)
from multi_robot_multi_goal_planning.planners.planner_aitstar import (
    AITstar, BaseITConfig
)
from multi_robot_multi_goal_planning.planners.planner_eitstar import (
    EITstar
)

from multi_robot_multi_goal_planning.planners.termination_conditions import (
    RuntimeTerminationCondition,
)


@pytest.mark.parametrize(
    "planner_fn",
    [
        lambda env, ptc: CompositePRM(env, CompositePRMConfig()).plan(ptc=ptc, optimize=False),
        lambda env, ptc: RRTstar(env, BaseRRTConfig()).plan(ptc=ptc, optimize=False),
        lambda env, ptc: BidirectionalRRTstar(env, BaseRRTConfig()).plan(ptc=ptc, optimize=False),
        lambda env, ptc: AITstar(env, BaseITConfig()).plan(ptc=ptc, optimize=False),
        lambda env, ptc: EITstar(env, BaseITConfig()).plan(ptc=ptc, optimize=False),
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
        lambda env, ptc: CompositePRM(env, CompositePRMConfig()).plan(ptc=ptc, optimize=False),
        lambda env, ptc: RRTstar(env, BaseRRTConfig()).plan(ptc=ptc, optimize=False),
        lambda env, ptc: BidirectionalRRTstar(env, BaseRRTConfig()).plan(ptc=ptc, optimize=False),
        lambda env, ptc: AITstar(env, BaseITConfig()).plan(ptc=ptc, optimize=False),
        lambda env, ptc: EITstar(env, BaseITConfig()).plan(ptc=ptc, optimize=False),
    ],
)
def test_planner_on_rai_manip_env(planner_fn):
    env = get_env_by_name("piano")
    ptc = RuntimeTerminationCondition(10)

    path, _ = planner_fn(env, ptc)

    assert path is not None

    assert np.array_equal(path[0].q.state(), env.start_pos.state())
    assert env.is_terminal_mode(path[-1].mode)
    assert env.is_valid_plan(path)

@pytest.mark.parametrize(
    "planner_fn",
    [
        lambda env, ptc: CompositePRM(env, CompositePRMConfig()).plan(ptc=ptc, optimize=False),
        lambda env, ptc: RRTstar(env, BaseRRTConfig()).plan(ptc=ptc, optimize=False),
        lambda env, ptc: BidirectionalRRTstar(env, BaseRRTConfig()).plan(ptc=ptc, optimize=False),
        lambda env, ptc: AITstar(env, BaseITConfig()).plan(ptc=ptc, optimize=False),
        lambda env, ptc: EITstar(env, BaseITConfig()).plan(ptc=ptc, optimize=False),
    ],
)
def test_planner_on_pinocchio_manip_env(planner_fn):
    env = get_env_by_name("pin_piano")
    ptc = RuntimeTerminationCondition(10)

    path, _ = planner_fn(env, ptc)

    assert path is not None

    assert np.array_equal(path[0].q.state(), env.start_pos.state())
    assert env.is_terminal_mode(path[-1].mode)
    assert env.is_valid_plan(path)


@pytest.mark.parametrize(
    "planner_fn",
    [
        lambda env, ptc: CompositePRM(env, CompositePRMConfig()).plan(ptc=ptc, optimize=False),
        lambda env, ptc: RRTstar(env, BaseRRTConfig()).plan(ptc=ptc, optimize=False),
        lambda env, ptc: BidirectionalRRTstar(env, BaseRRTConfig()).plan(ptc=ptc, optimize=False),
        lambda env, ptc: AITstar(env, BaseITConfig()).plan(ptc=ptc, optimize=False),
        lambda env, ptc: EITstar(env, BaseITConfig()).plan(ptc=ptc, optimize=False),
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
