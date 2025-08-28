import pytest

from multi_robot_multi_goal_planning.problems import get_all_environments

from multi_robot_multi_goal_planning.planners.composite_prm_planner import CompositePRM, CompositePRMConfig
from multi_robot_multi_goal_planning.planners.prioritized_planner import PrioritizedPlanner, PrioritizedPlannerConfig

from multi_robot_multi_goal_planning.planners.termination_conditions import (
    RuntimeTerminationCondition,
)

@pytest.mark.parametrize("env_name,env_fun_call", get_all_environments().items())
def test_env_can_be_constructed(env_name, env_fun_call):
    env = env_fun_call()
    assert env is not None

@pytest.mark.parametrize("env_name,env_fun_call", get_all_environments().items())
def test_planner_ingests_env(env_name, env_fun_call):
    env = env_fun_call()
    assert env is not None

    planner_fn = lambda env, ptc: CompositePRM(env, CompositePRMConfig()).plan(ptc=ptc, optimize=False)
    ptc = RuntimeTerminationCondition(1)

    path, _ = planner_fn(env, ptc)

# @pytest.mark.parametrize("env_name,env_fun_call", get_all_environments().items())
# def test_prio_planner_ingests_env(env_name, env_fun_call):
#     env = env_fun_call()
#     assert env is not None

#     planner_fn = lambda env, ptc: PrioritizedPlanner(env, PrioritizedPlannerConfig()).plan(ptc=ptc, optimize=False)
#     ptc = RuntimeTerminationCondition(1)

#     path, _ = planner_fn(env, ptc)