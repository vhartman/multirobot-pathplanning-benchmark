# src/multi_robot_multi_goal_planning/planners/solve.py
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

    `runtime` bounds the TREE SEARCH only; shortcutting is a separate, un-timed
    post-process (`shortcut_iters` successes). optimize=True keeps improving cost
    until `runtime`; optimize=False stops at the first solution.
    """
    if planner not in PLANNERS:
        raise ValueError(f"unknown planner {planner!r}; have {sorted(PLANNERS)}")
    cls, default_cfg = PLANNERS[planner]
    cfg = config if config is not None else default_cfg()

    # RRT-family knobs (harmless no-ops on configs that lack them).
    if hasattr(cfg, "shortcutting"):
        cfg.shortcutting = optimize            # internal rewiring only meaningful when optimizing
    if hasattr(cfg, "with_mode_validation"):
        cfg.with_mode_validation = False       # pre-validation blacklists whole SEQUENCE chains

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
        before = path_cost(path, env.batch_config_cost)
        path, sc_info = robot_mode_shortcut(env, path, max_iter=shortcut_iters, resolution=res)
        info["shortcut"] = sc_info
        if verbose:
            after = path_cost(path, env.batch_config_cost)
            print(f"[solve/{planner}] shortcut: {len(path)} states, {before:.3f} -> {after:.3f}")
    return path, info
