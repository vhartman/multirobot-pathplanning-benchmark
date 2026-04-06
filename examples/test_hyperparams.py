"""
Hyperparam ablation runner.

Structure
---------
Each planner has:
  - a config dataclass (defaults come from asdict(ConfigClass()))
  - a list of sweep axes, where each axis is a list of override dicts.

For an *independent* sweep (one-at-a-time), all axes are simply concatenated:
  variants = [default] + [default+override for axis in axes for override in axis]

To run a *coupled / 2D grid* sweep over two axes A and B, pass a single merged
axis built with itertools.product before calling run_ablation:
  coupled = [dict(**a, **b) for a, b in itertools.product(axis_A, axis_B)]
  axes = [..., coupled, ...]
"""

import argparse
import copy
import datetime
import itertools
import os
import random
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_experiment import (
    export_config,
    run_experiment,
    run_experiment_in_parallel,
    setup_planner,
)
from multi_robot_multi_goal_planning.problems import get_env_by_name
from multi_robot_multi_goal_planning.planners import (
    CompositePRMConfig,
    BaseRRTConfig,
    BaseITConfig,
    PrioritizedPlannerConfig,
)


# ---------------------------------------------------------------------------
# Sweep definitions
# Base options are derived from the dataclasses — no manual copying needed.
# ---------------------------------------------------------------------------

BIRRT_SWEEPS = [
    # informed_sampling
    [{"informed_sampling": True}, {"informed_sampling": False}],
    # locally_informed_sampling
    [{"locally_informed_sampling": True}, {"locally_informed_sampling": False}],
    # informed_batch_size
    [{"informed_batch_size": 100}, {"informed_batch_size": 300}, {"informed_batch_size": 600}],
    # p_goal
    [{"p_goal": 0.1}, {"p_goal": 0.2}, {"p_goal": 0.4}, {"p_goal": 0.6}],
    # init_mode_sampling_type
    [{"init_mode_sampling_type": "frontier"}, {"init_mode_sampling_type": "uniform_reached"}],
    # frontier_mode_sampling_probability
    [{"frontier_mode_sampling_probability": 0.8}, {"frontier_mode_sampling_probability": 0.9},
     {"frontier_mode_sampling_probability": 0.98}, {"frontier_mode_sampling_probability": 1.0}],
    # transition_nodes
    [{"transition_nodes": 10}, {"transition_nodes": 50}, {"transition_nodes": 100}],
    # balanced_trees
    [{"balanced_trees": True}, {"balanced_trees": False}],
    # with_mode_validation
    [{"with_mode_validation": True}, {"with_mode_validation": False}],
]

# ---------------------------------------------------------------------------

PRM_SWEEPS = [
    # mode_sampling_type
    [{"mode_sampling_type": "uniform_reached"}, {"mode_sampling_type": "frontier"}],
    # try_informed_sampling
    [{"try_informed_sampling": True}, {"try_informed_sampling": False}],
    # locally_informed_sampling
    [{"locally_informed_sampling": True}, {"locally_informed_sampling": False}],
    # uniform_batch_size
    [{"uniform_batch_size": 50}, {"uniform_batch_size": 200}, {"uniform_batch_size": 500}],
    # informed_batch_size
    [{"informed_batch_size": 100}, {"informed_batch_size": 500}, {"informed_batch_size": 1000}],
    # init_mode_sampling_type
    [{"init_mode_sampling_type": "greedy"}, {"init_mode_sampling_type": "frontier"},
     {"init_mode_sampling_type": "uniform_reached"}],
    # try_direct_informed_sampling
    [{"try_direct_informed_sampling": True}, {"try_direct_informed_sampling": False}],
    # use_k_nearest
    [{"use_k_nearest": True}, {"use_k_nearest": False}],
    # with_mode_validation
    [{"with_mode_validation": True}, {"with_mode_validation": False}],
]

# ---------------------------------------------------------------------------

PRIORITIZED_SWEEPS = [
    # shortcut_iters
    [{"shortcut_iters": 0}, {"shortcut_iters": 50}, {"shortcut_iters": 100}, {"shortcut_iters": 300}],
    # multirobot_shortcut_iters
    [{"multirobot_shortcut_iters": 0}, {"multirobot_shortcut_iters": 50},
     {"multirobot_shortcut_iters": 100}, {"multirobot_shortcut_iters": 300}],
]

# ---------------------------------------------------------------------------

AIT_SWEEPS = [
    # try_informed_sampling
    [{"try_informed_sampling": True}, {"try_informed_sampling": False}],
    # locally_informed_sampling
    [{"locally_informed_sampling": True}, {"locally_informed_sampling": False}],
    # uniform_batch_size
    [{"uniform_batch_size": 50}, {"uniform_batch_size": 100}, {"uniform_batch_size": 300}],
    # informed_batch_size
    [{"informed_batch_size": 100}, {"informed_batch_size": 350}, {"informed_batch_size": 700}],
    # try_direct_informed_sampling
    [{"try_direct_informed_sampling": True}, {"try_direct_informed_sampling": False}],
    # with_rewiring
    [{"with_rewiring": True}, {"with_rewiring": False}],
    # init_mode_sampling_type
    [{"init_mode_sampling_type": "frontier"}, {"init_mode_sampling_type": "uniform_reached"}],
    # frontier_mode_sampling_probability
    [{"frontier_mode_sampling_probability": 0.8}, {"frontier_mode_sampling_probability": 0.9},
     {"frontier_mode_sampling_probability": 0.98}, {"frontier_mode_sampling_probability": 1.0}],
    # with_mode_validation
    [{"with_mode_validation": True}, {"with_mode_validation": False}],
    # remove_based_on_modes
    [{"remove_based_on_modes": True}, {"remove_based_on_modes": False}],
]


# ---------------------------------------------------------------------------
# Variant expansion
# ---------------------------------------------------------------------------

def override_to_name(override: dict) -> str:
    if not override:
        return "default"
    return "_".join(f"{k}={v}" for k, v in sorted(override.items()))


def expand_independent_sweeps(config_cls, axes: list) -> list:
    """
    Returns [(name, options_dict), ...]: the default variant plus one variant
    per override value per axis (one-at-a-time ablation).

    config_cls is a dataclass class (e.g. BaseRRTConfig). Base options are
    derived from a default instance so they stay in sync with the code.

    Each axis is a list of override dicts.
    To sweep a 2D grid over axes A and B, pass a single merged axis:
        coupled = [dict(**a, **b) for a, b in itertools.product(axis_A, axis_B)]
        expand_independent_sweeps(config_cls, [..., coupled, ...])
    """
    base_options = asdict(config_cls())
    variants = [("default", base_options)]
    for axis in axes:
        for override in axis:
            name = override_to_name(override)
            variants.append((name, {**base_options, **override}))
    return variants


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

PLANNER_SPECS = {
    "birrt":      ("birrtstar",   BaseRRTConfig,           BIRRT_SWEEPS),
    "prm":        ("prm",         CompositePRMConfig,      PRM_SWEEPS),
    "prioritized":("prioritized", PrioritizedPlannerConfig,PRIORITIZED_SWEEPS),
    "ait":        ("aitstar",     BaseITConfig,            AIT_SWEEPS),
}


def run_ablation(
    planner_key: str,
    env_name: str,
    base_config: dict,
    parallel: bool,
    num_processes: int,
):
    planner_type, config_cls, sweep_axes = PLANNER_SPECS[planner_key]

    variants = expand_independent_sweeps(config_cls, sweep_axes)
    print(f"\n=== {planner_key}: {len(variants)} variants ===")

    np.random.seed(base_config["seed"])
    random.seed(base_config["seed"])

    env = get_env_by_name(env_name)
    env.cost_reduction = base_config["cost_reduction"]
    env.cost_metric = base_config["per_agent_cost"]

    config = copy.deepcopy(base_config)
    config["experiment_name"] = f"ablation_{planner_key}"
    config["environment"] = env_name
    config["planners"] = [
        {"name": name, "type": planner_type, "options": opts}
        for name, opts in variants
    ]

    planners = []
    for planner_config in config["planners"]:
        name, planner_fn, resolved_config = setup_planner(
            planner_config, config["max_planning_time"], config["optimize"]
        )
        planners.append((name, planner_fn))
        planner_config["options"] = asdict(resolved_config)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = (
        f"./out/{timestamp}_{config['experiment_name']}_{env_name}/"
    )
    os.makedirs(experiment_folder, exist_ok=True)
    export_config(experiment_folder, config)

    if parallel:
        run_experiment_in_parallel(
            env, planners, config, experiment_folder, max_parallel=num_processes
        )
    else:
        run_experiment(env, planners, config, experiment_folder)


DEFAULT_CONFIG = {
    "seed": 2,
    "num_runs": 2,
    "optimize": False,
    "max_planning_time": 100,
    "per_agent_cost": "euclidean",
    "cost_reduction": "max",
}


def main():
    parser = argparse.ArgumentParser(description="Hyperparam ablation runner")
    parser.add_argument("env", help="Environment name (e.g. rai.box_stacking_two_robots)")
    
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--num_processes", type=int, default=2)

    args = parser.parse_args()

    base_config = copy.deepcopy(DEFAULT_CONFIG)

    for planner_key in ["birrt"]:
        run_ablation(
            planner_key=planner_key,
            env_name=args.env,
            base_config=base_config,
            parallel=args.parallel,
            num_processes=args.num_processes,
        )


if __name__ == "__main__":
    main()
