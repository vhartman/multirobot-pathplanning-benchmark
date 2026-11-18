import pytest

from multi_robot_multi_goal_planning.problems import get_all_environments

from multi_robot_multi_goal_planning.planners.composite_prm_planner import CompositePRM, CompositePRMConfig
from multi_robot_multi_goal_planning.planners.prioritized_planner import PrioritizedPlanner, PrioritizedPlannerConfig

from multi_robot_multi_goal_planning.planners.termination_conditions import (
    RuntimeTerminationCondition,
)

import os
import numpy as np
import random

# Environments that need inputs the repository does not contain.
EXTERNAL_INPUTS = {
    "comotion.converted": "needs a saved CoMotion scene (path=...)",
    "rai.strut_assembly": "reads exported_boxes.json from a private path",
}


def construct(env_name, env_fun_call):
    if env_name in EXTERNAL_INPUTS:
        pytest.skip(EXTERNAL_INPUTS[env_name])
    kwargs = {}
    if env_name.startswith(("roboballet.", "cuarms.")):
        assets = os.environ.get("MRMG_ROBOBALLET_ASSETS")
        try:
            from multi_robot_multi_goal_planning.problems.mujoco.envs.roboballet.instances import default_assets
            found = assets is not None or (default_assets() / "data/mujoco_world").is_dir()
        except ImportError:
            found = assets is not None
        if not found:
            pytest.skip("RoboBallet assets not found (scripts/fetch_roboballet_assets.py or MRMG_ROBOBALLET_ASSETS)")
        if assets is not None:
            kwargs["assets"] = assets
    return env_fun_call(**kwargs)


@pytest.mark.parametrize("env_name,env_fun_call", get_all_environments().items())
def test_env_can_be_constructed(env_name, env_fun_call):
    np.random.seed(0)
    random.seed(0)
    
    env = construct(env_name, env_fun_call)
    assert env is not None

@pytest.mark.parametrize("env_name,env_fun_call", get_all_environments().items())
def test_planner_ingests_env(env_name, env_fun_call):
    np.random.seed(0)
    random.seed(0)

    env = construct(env_name, env_fun_call)
    assert env is not None

    planner_fn = lambda env, ptc: CompositePRM(env, CompositePRMConfig()).plan(ptc=ptc, optimize=False)
    ptc = RuntimeTerminationCondition(1)

    try:
        path, _ = planner_fn(env, ptc)
    except ValueError as e:
        if "does not support path constraints" in str(e):
            pytest.skip(str(e))
        raise

# @pytest.mark.parametrize("env_name,env_fun_call", get_all_environments().items())
# def test_prio_planner_ingests_env(env_name, env_fun_call):
#     env = env_fun_call()
#     assert env is not None

#     planner_fn = lambda env, ptc: PrioritizedPlanner(env, PrioritizedPlannerConfig()).plan(ptc=ptc, optimize=False)
#     ptc = RuntimeTerminationCondition(1)

#     path, _ = planner_fn(env, ptc)