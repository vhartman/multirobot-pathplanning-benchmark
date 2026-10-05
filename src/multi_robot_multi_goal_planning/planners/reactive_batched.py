import time
from dataclasses import dataclass, replace

import numpy as np
from typing import Dict, List, Optional, Tuple

from .policy_shortcut import PolicyShortcutConfig, iterative_policy_shortcut
from .reactive_mdp import ReactiveMDP, ReactiveMDPConfig
from .rrt_skills_reactive import ReactiveRoadmap

@dataclass
class BatchedConfig:
    first_batch_seconds: float = 90.0
    batch_seconds: float = 60.0
    keep_batching_after_coverage: bool = True
    patience: int = 3
    min_absorb_headroom: float = 1.25
    absorb_seconds: float = 45.0
    stop_at_first_solution: bool = True
    growth_shortcutting: bool = True
    exact_planar_edges: bool = True
    inactive_batch_fraction: float = 0.25

def prepare_batched_mdp_config(env, config: Optional[ReactiveMDPConfig],
                               cfg: BatchedConfig) -> ReactiveMDPConfig:
    """
    Resolve the one environment-dependent safety default used by the batched driver
    """
    config = config or ReactiveMDPConfig()
    dims = getattr(env, "robot_dims", {})
    planar = bool(dims) and max(dims.values()) <= 2
    if cfg.exact_planar_edges and planar:

        return replace(
            config,
            det_edge_validation="solve",
            exit_repair_rounds=max(64, config.exit_repair_rounds),
        )
    return config

def batched_reactive_plan(
    roadmap: ReactiveRoadmap,
    mdp_config: Optional[ReactiveMDPConfig] = None,
    shortcut_config: Optional[PolicyShortcutConfig] = None,
    cfg: Optional[BatchedConfig] = None,
    deadline: Optional[float] = None,
) -> Tuple[ReactiveMDP, List[Dict]]:
    """
    Grow the roadmap and the policy together until `deadline`. Returns (final MDP, history)
    """
    cfg = cfg or BatchedConfig()
    mdp_config = prepare_batched_mdp_config(roadmap.env, mdp_config, cfg)
    roadmap.config.build_shortcut = bool(cfg.growth_shortcutting)
    history: List[Dict] = []
    mdp: Optional[ReactiveMDP] = None
    flat = batch = 0
    absorb = 0.0
    last_solve = 0.0
    best_v = float("inf")

    def left() -> float:
        return float("inf") if deadline is None else deadline - time.time()

    while True:

        if left() <= 0:
            break
        covered = mdp is not None
        if covered and not cfg.keep_batching_after_coverage:
            break
        if flat >= cfg.patience:
            break

        seconds = cfg.first_batch_seconds if batch == 0 else cfg.batch_seconds
        seconds = min(seconds, max(0.0, left() - cfg.min_absorb_headroom * absorb))
        if seconds <= 0:
            break

        t0 = time.time()
        added = roadmap.grow(
            seconds,
            stop_at_first_solution=(batch == 0 and cfg.stop_at_first_solution),
            inactive_fraction=cfg.inactive_batch_fraction,
        )
        grow_time = time.time() - t0
        prev_v = best_v
        t1 = time.time()
        batch_deadline = t1 + cfg.absorb_seconds

        if deadline is not None:
            batch_deadline = min(batch_deadline, deadline)

        mdp, rounds = iterative_policy_shortcut(
            roadmap, mdp_config, shortcut_config,
            deadline=batch_deadline, previous=mdp)
        
        absorb = time.time() - t1
        last_solve = float(rounds[-1].get("solve_time", 0.0)) if rounds else 0.0
        v = mdp.get_start_cost_to_go()
        grow_v = float(rounds[0].get("V_start", v)) if rounds else v
        grow_helped = bool(np.isfinite(prev_v) and grow_v < prev_v * (1.0 - 1e-4))
        productive = bool(added["modes"] or added.get("inactive", 0)) or grow_helped
        flat = 0 if productive else flat + 1
        best_v = min(best_v, v)

        if rounds:
            rounds[0] = dict(rounds[0])
            rounds[0]["solve_time"] = float(rounds[0].get("solve_time", 0.0)) + grow_time
            history.extend(rounds)
        batch += 1

    if mdp is None:

        raise RuntimeError("batched planner absorbed no batch before the deadline; "
                           "first_batch_seconds is larger than the budget")

    if left() > 2.0 * last_solve:
        mdp, rounds = iterative_policy_shortcut(
            roadmap, mdp_config, shortcut_config,
            deadline=deadline, previous=mdp)
        history.extend(rounds)

    return mdp, history
