# src/multi_robot_multi_goal_planning/planners/solve.py
from copy import deepcopy

from .composite_prm_planner import CompositePRM, CompositePRMConfig
from .planner_rrtstar import RRTstar
from .planner_birrtstar import BidirectionalRRTstar
from .planner_aitstar import AITstar
from .planner_eitstar import EITstar
from .rrtstar_base import BaseRRTConfig
from .itstar_base import BaseITConfig
from .shortcutting import robot_mode_shortcut
from .termination_conditions import RuntimeTerminationCondition
from ..problems.util import path_cost

# name -> (planner class, default-config factory). Adding a planner = one line here.
PLANNERS = {
    "composite_prm": (CompositePRM,          CompositePRMConfig),
    "rrt_star":      (RRTstar,               BaseRRTConfig),
    "birrt":         (BidirectionalRRTstar,  BaseRRTConfig),
    "aitstar":       (AITstar,               BaseITConfig),
    "eitstar":       (EITstar,               BaseITConfig),
}

def solve(env, planner="birrt", runtime=60.0, optimize=False,
          shortcut=True, shortcut_iters=1000, shortcut_resolution=None,
          config=None, seed=None, verbose=True):
    """One-call motion planning for a mode env. Returns (path, info).

    `runtime` bounds the planner call, including its internal shortcutting.
    optimize=True keeps improving until `runtime`; otherwise planning stops at
    the first solution. The supplied config is copied and its settings retained;
    without one, the planner's defaults apply.

    `shortcut` controls additional postprocessing, outside the runtime budget.
    `shortcut_iters` bounds collision-checked proposals, not accepted shortcuts.
    Postprocessing uses the environment's collision tolerance and resolution
    (unless shortcut_resolution is supplied), and the config's shortcut policy.
    To disable all shortcutting, also disable it in the planner config.
    """
    if planner not in PLANNERS:
        raise ValueError(f"unknown planner {planner!r}; have {sorted(PLANNERS)}")
    cls, default_cfg = PLANNERS[planner]
    cfg = deepcopy(config) if config is not None else default_cfg()

    if seed is not None:
        import numpy as np, random
        np.random.seed(seed); random.seed(seed)

    path, info = cls(env, config=cfg).plan(
        ptc=RuntimeTerminationCondition(runtime), optimize=optimize)
    if verbose:
        print(f"[solve/{planner}] {'%d states' % len(path) if path else 'NO path'}")

    if path and shortcut:
        res = (shortcut_resolution if shortcut_resolution is not None
               else getattr(env, "collision_resolution", 0.01))   # don't check finer than the planner
        before = path_cost(path, env.cost)
        path, sc_info = robot_mode_shortcut(
            env, path, max_iter=shortcut_iters, resolution=res,
            tolerance=env.collision_tolerance,
            run_choice=getattr(cfg, "shortcutting_run_choice", "tree"),
            robot_choice=getattr(cfg, "shortcutting_mode", "round_robin"),
        )
        info["shortcut"] = sc_info
        if verbose:
            after = path_cost(path, env.cost)
            print(f"[solve/{planner}] shortcut: {len(path)} states, {before:.3f} -> {after:.3f}")
    return path, info
