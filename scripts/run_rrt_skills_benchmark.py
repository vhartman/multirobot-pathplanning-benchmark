import argparse
import ast
import datetime
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from copy import deepcopy
from typing import Any

# =====================================================================
# 1. Environments - comment in the ones you want to run
# =====================================================================

# Environments without skills, from problems/rai/rai_envs.py
NON_SKILL_ENVS = [
    # "rai.simple",
    # "rai.other_hallway",
    # "rai.piano",
    # "rai.random_2d",
    # "rai.2d_handover",
    # "rai.handover",
    # "rai.box_sorting",
    # "rai.box_stacking_three_robots",
]

# Environments with deterministic skills, from problems/rai/rai_skill_envs.py
DETERMINISTIC_SKILL_ENVS = [    
    "rai.multi_agent_bin_picking",
    "rai.multi_agent_bin_packing",
    "rai.dual_arm_transport",    
    "rai.skill_handover",
]

# Environments (2D) with stochastic skills for intuitive analysis, from problems/rai/rai_skill_envs.py
STOCHASTIC_2D_SKILL_ENVS = [
    "rai.dep_stochastic_square_island",
    "rai.dep_stochastic_rectangle_island",
    "rai.dep_bimodal_shared_point",
    
    # "rai.dep_stochastic_bimodal_switch",
    # "rai.dep_reconverging_bimodal_switch",
]

# Environments (3D) with stochastic skills for scalability analysis, from problems/rai/rai_skill_envs.py
SCALABILITY_ENVS = [
    "rai.stochastic_sequence_spread_stacking_4r_8b",
    "rai.stochastic_sequence_spread_stacking_3r_8b",
    "rai.stochastic_sequence_spread_stacking_2r_8b",
    # "rai.dep_stochastic_spread_stacking_4r_8b",
    # "rai.dep_stochastic_spread_stacking_3r_8b",
    # "rai.dep_stochastic_spread_stacking_2r_8b",
]

# =====================================================================
# 2. Planners - tweak the options here
# =====================================================================

# Params applied to every environment, both overridable from the CLI
SEED = 0 
MAX_TIME = 300
NUM_EXECUTIONS = 100
MC_ROLLOUTS = 500
TUBE_ROLLOUTS = 200
SHORTCUT_ROUNDS = 12
RRG_BUILD_SECONDS = 30.0
RRG_BUILD_FRACTION = 0.5
RRG_BUILD_CAP = 500.0

def reactive_build_seconds(max_time: float) -> float:
    """Seconds of `max_time` the reactive planner spends building its roadmap (phase A)"""
    return min(RRG_BUILD_CAP, max(RRG_BUILD_SECONDS, RRG_BUILD_FRACTION * max_time))

RRG_FIRST_BATCH_FRACTION = 0.08
RRG_FIRST_BATCH_MIN = 15.0
RRG_FIRST_BATCH_MAX = 60.0

def reactive_first_batch_seconds(max_time: float) -> float:
    """Sampling seconds in the FIRST batch, i.e. how long until a policy can exist at all"""
    return min(RRG_FIRST_BATCH_MAX, max(RRG_FIRST_BATCH_MIN, RRG_FIRST_BATCH_FRACTION * max_time))

def reactive_absorb_seconds(max_time: float) -> float:
    """Wall clock one batch may spend folding itself in (one solve + one improve round)"""
    return min(90.0, max(20.0, 0.05 * max_time))

RRT_SKILLS_OPTIONS = {
    "extension_strategy": "connect",
    "connect_target_policy": "transition",
    "connect_add_all_nodes": True,
    "skill_expansion_strategy": "kinodynamic",
    "init_mode_sampling_type": "frontier",
    "use_rrt_star": True,
    "try_informed_sampling": True,
    "try_shortcutting": True,
    "sync_shortcut_to_tree": True,
    "shortcut_max_iters": 500,
    "shortcut_period_improve": 100,
    "inactive_max_vel": 2.0,
    "inactive_transition_source": "random_tree",
}

RRT_SKILLS = {
    "name": "rrt_skills",
    "type": "rrt_skills",
    "options": dict(RRT_SKILLS_OPTIONS),
}

RRG_SKILLS = {
    "name": "rrg_skills",
    "type": "rrt_skills",
    "options": {**RRT_SKILLS_OPTIONS, "build_mode": "rrg"},
}

PRM = {
    "name": "prm_incremental_frozen_lane",
    "type": "prm",
    "options": {
        "skill_lane_strategy": "incremental_frozen_lane",
        "try_informed_sampling": True,
        "try_shortcutting": True,
    },
}

PP = {
    "name": "pp",
    "type": "prioritized",
    "options": {},
}

RRTSTAR_OLD = {
    "name": "rrtstar_old",
    "type": "rrtstar",
    "options": {
        "informed_sampling": True,
        "shortcutting": True,
        "with_mode_validation": False,
    },
}

BIRRTSTAR_OLD = {
    "name": "birrtstar_old",
    "type": "birrtstar",
    "options": {
        "informed_sampling": True,
        "shortcutting": True,
        "with_mode_validation": False,
    },
}

RRT_CONSERVATIVE = {
    "name": "rrt_conservative",
    "type": "rrt_skills_conservative",
    "options": {
        **RRT_SKILLS_OPTIONS,
        "tube_rollouts": TUBE_ROLLOUTS,
        "kinodynamic_steps": 5,
        "num_executions": NUM_EXECUTIONS,

        "inactive_transition_source": "random_tree",
    },
}

RRT_DETERMINISTIC_EXECUTED = {
    "name": "rrt_deterministic",
    "type": "rrt_skills",
    "options": {**RRT_SKILLS_OPTIONS, "num_executions": NUM_EXECUTIONS},
}

RRG_REACTIVE = {
    "name": "rrg_reactive",
    "type": "reactive",
    "options": {
        "inactive_k_max": 10,
        "inactive_max_vel": 2.0,
        "n_mc": MC_ROLLOUTS,
        "decision_frequency": 10,
        "nature_state": "position_time",
        "rrg_inactive_transition_source": "random_tree",
        "bin_tol_floor": 0.1,
        "vi_max_iters": 100,
        "num_executions": NUM_EXECUTIONS,
        "rrg_runtime": reactive_build_seconds(MAX_TIME),
        "batched": True,
        "first_batch_seconds": reactive_first_batch_seconds(MAX_TIME),
        "batch_seconds": 0.75 * reactive_first_batch_seconds(MAX_TIME),
        "absorb_seconds": reactive_absorb_seconds(MAX_TIME),
        "shortcut_modes_per_round": 8,
        "inactive_n_samples": 500,
        "growth_shortcutting": True,
        "skill_epoch_distance_weight": 0.01,
    },
}

# =====================================================================
# 3. Test groups
# =====================================================================

TEST_GROUPS = {
    "non_skill": {
        "description": "RRTSkills against the existing planners on environments without skills.",
        "optimize": True,
        "cost_model": "geometric",
        "envs": NON_SKILL_ENVS,
        "planners": [RRTSTAR_OLD, BIRRTSTAR_OLD, PRM, RRT_SKILLS],
    },
    "deterministic_skill": {
        "description": "The three planners that integrate deterministic skills: PP, PRM*, RRTSkills, and RRGSkills.",
        "optimize": True,
        "cost_model": "geometric",
        "envs": DETERMINISTIC_SKILL_ENVS,
        "planners": [PP, PRM, RRT_SKILLS],
    },
    "stochastic_skill": {
        "description": "Stochastic skill environments: open-loop conservative tube planner "
                       "vs. the closed-loop reactive planner, with the prioritized planner as "
                       "baseline.",
        "optimize": True,
        "cost_model": "time",
        "envs": STOCHASTIC_2D_SKILL_ENVS,
        "planners": [RRT_CONSERVATIVE, RRG_REACTIVE],
    },
    "scalability": {
        "description": "Cost convergence at 2, 3 and 4 robots on the same 8-box task. One "
                       "experiment folder per environment, so the figure can be drawn for any "
                       "subset of them afterwards.",
        "optimize": True,
        "cost_model": "time",
        "envs": SCALABILITY_ENVS,
        "planners": [RRT_CONSERVATIVE, RRG_REACTIVE],
    },
    "deterministic_on_stochastic": {
        "description": "Test deterministic RRT planner on stochastic environments to show poor robustness.",
        "optimize": True,
        "cost_model": "geometric",
        "envs": [
            "rai.dep_stochastic_square_island",
            "rai.dep_stochastic_rectangle_island",
            "rai.dep_bimodal_shared_point",
            "rai.dep_stochastic_stacking_2r_4b",
            "rai.dep_stochastic_stacking_3r_3b",
            "rai.dep_stochastic_stacking_4r_4b",
        ],
        "planners": [RRT_DETERMINISTIC_EXECUTED],
    },
}

# =====================================================================
# 4. CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark the skill planners")
    parser.add_argument(
        "--group",
        choices=list(TEST_GROUPS.keys()),
        default="stochastic_skill",
        help="which test group to run",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="override the group's environment list, e.g. --envs rai.simple rai.piano",
    )
    parser.add_argument(
        "--planners",
        nargs="+",
        default=None,
        help="only run these planner names out of the group, e.g. --planners rrt_skills",
    )
    parser.add_argument(
        "--cost_reduction",
        choices=["max", "sum"],
        default="max",
        help="how per-robot costs are combined. This is an ENVIRONMENT property, so it changes "
             "the cost every planner is measured in -- use a distinct --experiment_name when "
             "changing it, or the two are plotted on one axis as if comparable.",
    )
    parser.add_argument(
        "--cost_model",
        choices=["geometric", "time"],
        default=None,
        help="'geometric': cost is config-space distance everywhere, so a skill's "
             "DURATION is invisible to the planner -- waiting inside a skill window is free. "
             "'time': skill edges are charged in seconds (micro-steps * skill.dt), transit edges "
             "unchanged. Defaults to the group's setting (geometric for deterministic groups, "
             "time for stochastic groups). Reaches RRT/conservative/reactive (via their shared "
             "RRTSkills base and ReactiveMDPConfig.objective); PP and PRM stay geometric regardless. "
             "Use a distinct --experiment_name when changing it.",
    )
    parser.add_argument(
        "--v_ref", type=float, default=2.0,
        help="Reference velocity the 'time' cost model divides seconds by; 1.0 (default) makes "
             "'time' a byte-identical no-op on any env without skills. An approximation -- see "
             "docs/final_plan/6_final_changes/00_plan_2026-09-01_post_meeting_26.md §4.4.",
    )

    parser.add_argument("--num_runs", type=int, default=5,
                        help="R: planning runs per environment (different planner seeds). This is "
                             "what the cost figures' median and confidence band repeat over, and "
                             "it is the axis that buys statistical power. 5 to iterate, 10 final.")
    parser.add_argument("--num_executions", type=int, default=NUM_EXECUTIONS,
                        help="E: fresh skill realizations each plan/policy is executed against. "
                             "Near-free, so keep it at 10 -- the ECDF plateau and the outcome-"
                             "branch figures are computed from it.")
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--experiment_name", default="skill_benchmark")
    parser.add_argument(
        "--profile_dir",
        default=None,
        help="directory for --group profile Speedscope JSON. Default: out/_profiles/<timestamp>_<experiment_name>",
    )
    parser.add_argument(
        "--profile_rate",
        type=int,
        default=100,
        help="py-spy sampling rate in Hz for --group profile",
    )
    parser.add_argument(
        "--profile_native",
        action="store_true",
        help="include native stack frames in --group profile output; useful for Numba-heavy VI",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--max_time",
        type=float,
        default=MAX_TIME,
        help="planning time budget per run, in seconds",
    )
    parser.add_argument(
        "--planner_opt",
        action="append",
        default=None,
        metavar="NAME.KEY=VALUE",
        help="override one planner option without editing this file, e.g. "
             "--planner_opt rrg_reactive.allow_wait=False. VALUE is parsed as a Python "
             "literal, falling back to a string. Repeatable. The key must already be a field of "
             "that planner's config -- ablations are A/B knobs, so a typo must fail loudly "
             "rather than be silently ignored (the reactive branch of setup_planner rejects "
             "unknown options anyway, but the other planners do not).",
    )
    parser.add_argument("--no_parallel", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--pdf", action="store_true", help="save plots as pdf instead of png")
    parser.add_argument("--no_legend", action="store_true")
    return parser.parse_args()

# =====================================================================
# 5. Running
# =====================================================================

def list_matching_experiment_folders(
    experiment_name: str, env_name: str
) -> set[pathlib.Path]:
    out = pathlib.Path(os.environ.get("MRMG_OUTPUT_DIR", "out"))
    if not out.exists():
        return set()

    suffix = f"_{experiment_name}_{env_name}"
    return {p for p in out.iterdir() if p.is_dir() and p.name.endswith(suffix)}

STOCHASTIC_PLANNER_TYPES = {"rrt_skills_conservative", "reactive"}

def any_stochastic(planners: list[dict]) -> bool:
    """True if any selected planner executes its solution against fresh skill realizations"""
    return any(p["type"] in STOCHASTIC_PLANNER_TYPES for p in planners)

def uses_stochastic_plots(planners: list[dict]) -> bool:
    """True if the execution-level plots are the ones that describe this run"""
    return any_stochastic(planners)

def select_planners(group_config: dict[str, Any], names: list[str] | None,
                    num_executions: int | None = None,
                    max_time: float | None = None,
                    overrides: list[tuple[str, str, Any]] | None = None) -> list[dict]:
    planners = group_config["planners"]
    if names is None:
        selected = deepcopy(planners)
    else:
        selected = deepcopy([p for p in planners if p["name"] in names])
        unknown = set(names) - {p["name"] for p in planners}
        if unknown:
            available = ", ".join(p["name"] for p in planners)
            raise SystemExit(f"Unknown planner(s) {sorted(unknown)}. Available: {available}")

    if num_executions is not None:

        for p in selected:
            if "num_executions" in p["options"]:
                p["options"]["num_executions"] = num_executions

    if max_time is not None:
        for p in selected:
            if "rrg_runtime" in p["options"]:
                p["options"]["rrg_runtime"] = reactive_build_seconds(max_time)

            if "first_batch_seconds" in p["options"]:
                p["options"]["first_batch_seconds"] = reactive_first_batch_seconds(max_time)
                p["options"]["batch_seconds"] = 0.75 * reactive_first_batch_seconds(max_time)
                p["options"]["absorb_seconds"] = reactive_absorb_seconds(max_time)

    for name, key, value in overrides or []:
        targets = [p for p in selected if p["name"] == name]
        if not targets:
            raise SystemExit(f"--planner_opt {name}.{key}: planner {name!r} is not in this run. "
                             f"Selected: {', '.join(p['name'] for p in selected)}")
        for p in targets:
            p["options"][key] = value
            print(f"[OVERRIDE] {name}.{key} = {value!r}")
    return selected

def parse_planner_opts(raw: list[str] | None) -> list[tuple[str, str, Any]]:
    """Parses --planner_opt NAME.KEY=VALUE into (planner_name, option_key, value)"""
    out = []
    for item in raw or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise SystemExit(f"--planner_opt expects NAME.KEY=VALUE, got {item!r}")
        target, value = item.split("=", 1)
        name, key = target.split(".", 1)
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = value
        out.append((name, key, parsed))
    return out

def make_plots(experiment_folder: pathlib.Path, max_time: float, args) -> None:
    cmd = [
        sys.executable,
        "./scripts/analysis/make_plots_deterministic_skill.py",
        str(experiment_folder),
        "--save",
        "--no_display",
        "--limited_max_time",
        str(max_time),
    ]

    if not args.pdf:
        cmd.append("--png")

    if not args.no_legend:
        cmd.append("--legend")

    subprocess.run(cmd, check=True)

def make_stochastic_plots(experiment_folder: pathlib.Path, args) -> None:
    """Execution-level plots for the stochastic group"""
    cmd = [
        sys.executable,
        "./scripts/analysis/make_plots_stochastic_skill.py",
        str(experiment_folder),
        "--out", str(experiment_folder / "plots_stochastic"),
    ]
    if args.pdf:
        cmd.append("--pdf")
    if args.no_legend:
        cmd.append("--no_legend")

    subprocess.run(cmd, check=True)

    if args.group in ("stochastic_skill", "stochastic_ablation"):
        cmd_top_down = [
            sys.executable,
            "./scripts/analysis/make_plots_stochastic_topdown.py",
            str(experiment_folder),
            "--out", str(experiment_folder / "plots_stochastic"),
        ]
        if args.pdf:
            cmd_top_down.append("--pdf")
        if args.no_legend:
            cmd_top_down.append("--no_legend")
        subprocess.run(cmd_top_down, check=False)

def run_environment(
    env_name: str,
    args,
    test_group: str,
    group_config: dict[str, Any],
    planners: list[dict],
) -> pathlib.Path | None:
    print(
        f"Running {test_group}:{env_name} with seed {args.seed}, "
        f"runtime {args.max_time}s, {args.num_runs} runs"
    )
    before = list_matching_experiment_folders(args.experiment_name, env_name)
    config = {
        "experiment_name": args.experiment_name,
        "environment": env_name,
        "per_agent_cost": "euclidean",
        "cost_reduction": args.cost_reduction,
        "cost_model": args.cost_model,
        "v_ref": args.v_ref,
        "max_planning_time": args.max_time,
        "num_runs": args.num_runs,
        "seed": args.seed,
        "optimize": group_config["optimize"],
        "planners": planners,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmpfile:
        json.dump(config, tmpfile, indent=2)
        tmpfile_path = tmpfile.name

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_experiment",
        tmpfile_path,
        "--num_processes",
        str(args.num_processes),
    ]

    if test_group == "profile":
        profile_dir = pathlib.Path(args.profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        safe_env = env_name.replace("/", "_").replace(":", "_")
        safe_planners = "_".join(p["name"] for p in planners)
        profile_out = profile_dir / f"{safe_env}__{safe_planners}.speedscope.json"
        profile_config = profile_dir / f"{safe_env}__{safe_planners}.config.json"
        profile_config.write_text(json.dumps(config, indent=2) + "\n")
        pyspy_cmd = [
            "py-spy", "record",
            "--subprocesses",
        ]
        if args.profile_native:
            pyspy_cmd.append("--native")
        cmd = pyspy_cmd + [
            "--format", "speedscope",
            "--rate", str(args.profile_rate),
            "-o", str(profile_out),
            "--",
        ] + cmd
        print(f"[PROFILE] Speedscope output: {profile_out}")
        print(f"[PROFILE] Copied benchmark config: {profile_config}")

    if not args.no_parallel and not any_stochastic(planners):
        cmd.append("--parallel_execution")

    try:
        subprocess.run(cmd, check=True, cwd=repo_root)
    finally:
        os.remove(tmpfile_path)

    after = list_matching_experiment_folders(args.experiment_name, env_name)
    new_folders = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if not new_folders:
        print(f"Could not identify output folder for {env_name}")
        return None

    return new_folders[-1]

def _prefer_self_as_oom_victim() -> None:
    """Make THIS process tree the kernel's first choice when the machine runs out of memory"""
    if os.environ.get("MRMG_NO_OOM_ADJ"):
        return
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("800")
    except Exception:
        pass

def main():
    args = parse_args()
    _prefer_self_as_oom_victim()

    if args.group == "profile" and args.profile_dir is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.profile_dir = str(pathlib.Path("out") / "_profiles" / f"{stamp}_{args.experiment_name}")

    group_config = TEST_GROUPS[args.group]
    if args.cost_model is None:
        args.cost_model = group_config.get("cost_model", "time")

    envs = args.envs if args.envs is not None else group_config["envs"]
    planners = select_planners(group_config, args.planners, args.num_executions, args.max_time,
                               parse_planner_opts(args.planner_opt))

    if not envs:
        raise SystemExit(
            f"No environments selected for group '{args.group}'. "
            f"Comment one in at the top of this file or pass --envs."
        )

    print(f"\n=== {args.group}: {group_config['description']} ===")
    print(f"Cost model:   {args.cost_model}")
    print(f"Environments: {', '.join(envs)}")
    print(f"Planners:     {', '.join(p['name'] for p in planners)}\n")

    for env_name in envs:
        experiment_folder = run_environment(
            env_name, args, args.group, group_config, deepcopy(planners)
        )
        if experiment_folder is None:
            continue

        if not args.no_plots:

            if uses_stochastic_plots(planners):
                make_stochastic_plots(experiment_folder, args)
            else:
                make_plots(experiment_folder, args.max_time, args)

        print(f"Finished {env_name}. Results: {experiment_folder}")

if __name__ == "__main__":
    main()
