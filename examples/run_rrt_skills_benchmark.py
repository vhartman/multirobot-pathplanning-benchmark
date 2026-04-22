import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


ENVS = [
    # ("rai.skill_handover", 2, 500),
    ("rai.single_agent_drawing_square", 2, 500),
    ("rai.single_agent_screw", 2, 500),
    ("rai.single_agent_pick_and_place", 2, 500),
    ("rai.single_agent_bin_picking", 2, 500),
    ("rai.single_agent_bin_packing", 2, 500),
    ("rai.single_agent_scripted_insert", 2, 500),
    ("rai.skill_handover", 2, 500),
    ("rai.dual_arm_transport", 2, 500),
    ("rai.bimanual_sorting", 2, 500),
    ("rai.multi_agent_bin_picking", 2, 500),
    ("rai.multi_agent_bin_packing", 2, 500),
    ("rai.multi_agent_scripted_insert", 2, 500),
    ("rai.skill_box_stacking_two_robots", 2, 500),
    ("rai.hallway_counterexample_wide_sweep", 2, 500),
    ("rai.hallway_counterexample_wide_full_sweep", 2, 500),
    ("rai.single_agent_bin_packing_goal_set", 2, 500),
    ("rai.single_agent_scripted_insert_goal_set", 2, 500),
    ("rai.dep_multi_agent_bin_picking", 2, 500),
]


PLANNERS = [
    {
        "name": "rrt_single_step",
        "type": "rrt_skills",
        "options": {
            "use_rrt_star": False,
            "skill_expansion_strategy": "single_step",
            "try_shortcutting": True,
        },
    },
    {
        "name": "rrtstar_single_step",
        "type": "rrt_skills",
        "options": {
            "use_rrt_star": True,
            "skill_expansion_strategy": "single_step",
            "try_shortcutting": True,
        },
    },
    {
        "name": "rrt_kinodynamic",
        "type": "rrt_skills",
        "options": {
            "use_rrt_star": False,
            "skill_expansion_strategy": "kinodynamic",
            "kinodynamic_steps": 5,
            "try_shortcutting": True,
        },
    },
    {
        "name": "rrtstar_kinodynamic",
        "type": "rrt_skills",
        "options": {
            "use_rrt_star": True,
            "skill_expansion_strategy": "kinodynamic",
            "kinodynamic_steps": 5,
            "try_shortcutting": True,
        },
    },
    # {
    #     "name": "prm_phase3_outside",
    #     "type": "prm",
    #     "options": {
    #         "skill_phase": 3,
    #         "skill_batch_strategy": "outside",
    #         "try_shortcutting": True,
    #     },
    # },
    # {
    #     "name": "prioritized",
    #     "type": "prioritized",
    #     "options": {},
    # },
    # {
    #     "name": "rrtstar_old",
    #     "type": "rrtstar",
    #     "options": {},
    # },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RRTSkills and PRM phase-3 benchmarks across skill environments."
    )
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--experiment_name", default="rrt_skills_step4")
    parser.add_argument(
        "--max_time",
        type=float,
        default=None,
        help="Override the per-environment max planning time in seconds.",
    )
    parser.add_argument(
        "--max_envs",
        type=int,
        default=None,
        help="Only run the first N environments from ENVS.",
    )
    parser.add_argument(
        "--no_parallel",
        action="store_true",
        help="Run planner variants sequentially instead of using run_experiment parallel mode.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip automatic cost/success plot generation.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Save plots as PDF instead of PNG.",
    )
    parser.add_argument(
        "--no_legend",
        action="store_true",
        help="Do not include legends in generated plots.",
    )
    return parser.parse_args()


def list_matching_experiment_folders(
    experiment_name: str, env_name: str
) -> set[pathlib.Path]:
    out = pathlib.Path("out")
    if not out.exists():
        return set()

    suffix = f"_{experiment_name}_{env_name}"
    return {p for p in out.iterdir() if p.is_dir() and p.name.endswith(suffix)}


def read_trace_file(path: pathlib.Path) -> list[list[float]]:
    if not path.exists():
        return []

    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                rows.append([])
                continue
            rows.append([float(x) for x in line.split(",") if x])
    return rows


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(experiment_folder: pathlib.Path) -> list[dict[str, Any]]:
    per_run_rows = []
    aggregate_rows = []

    with (experiment_folder / "config.json").open() as f:
        config = json.load(f)

    expected_runs = int(config["num_runs"])
    planner_dirs = [
        p for p in experiment_folder.iterdir() if p.is_dir() and p.name != "plots"
    ]

    for planner_dir in sorted(planner_dirs):
        times_by_run = read_trace_file(planner_dir / "timestamps.txt")
        costs_by_run = read_trace_file(planner_dir / "costs.txt")

        initial_times = []
        initial_costs = []
        final_times = []
        final_costs = []
        improvements = []

        for run_id in range(expected_runs):
            times = times_by_run[run_id] if run_id < len(times_by_run) else []
            costs = costs_by_run[run_id] if run_id < len(costs_by_run) else []
            success = bool(times) and bool(costs)

            row = {
                "environment": config["environment"],
                "planner": planner_dir.name,
                "run_id": run_id,
                "success": int(success),
                "initial_time": "",
                "initial_cost": "",
                "final_time": "",
                "final_cost": "",
                "num_improvements": 0,
            }

            if success:
                row["initial_time"] = times[0]
                row["initial_cost"] = costs[0]
                row["final_time"] = times[-1]
                row["final_cost"] = costs[-1]
                row["num_improvements"] = len(costs)

                initial_times.append(times[0])
                initial_costs.append(costs[0])
                final_times.append(times[-1])
                final_costs.append(costs[-1])
                improvements.append(len(costs))

            per_run_rows.append(row)

        success_count = len(final_costs)
        aggregate_rows.append(
            {
                "environment": config["environment"],
                "planner": planner_dir.name,
                "num_runs": expected_runs,
                "success_count": success_count,
                "success_rate": success_count / expected_runs if expected_runs else 0.0,
                "median_initial_time": np.median(initial_times)
                if initial_times
                else "",
                "median_initial_cost": np.median(initial_costs)
                if initial_costs
                else "",
                "median_final_time": np.median(final_times) if final_times else "",
                "median_final_cost": np.median(final_costs) if final_costs else "",
                "median_num_improvements": np.median(improvements)
                if improvements
                else "",
            }
        )

    write_csv(experiment_folder / "summary_per_run.csv", per_run_rows)
    write_csv(experiment_folder / "summary_by_planner.csv", aggregate_rows)
    return aggregate_rows


def make_plots(experiment_folder: pathlib.Path, max_time: float, args) -> None:
    cmd = [
        sys.executable,
        "./examples/make_plots.py",
        str(experiment_folder),
        "--save",
        "--no_display",
        "--limited_max_time",
        str(max_time),
    ]

    if not args.pdf:
        cmd.append("--png")

    if not args.no_legend:
        cmd.extend(["--legend", "--info"])

    subprocess.run(cmd, check=True)


def run_environment(env_name: str, seed: int, runtime: float, args) -> pathlib.Path | None:
    print(f"Running {env_name} with seed {seed}, runtime {runtime}s")
    before = list_matching_experiment_folders(args.experiment_name, env_name)

    config = {
        "experiment_name": args.experiment_name,
        "environment": env_name,
        "per_agent_cost": "euclidean",
        "cost_reduction": "max",
        "max_planning_time": runtime,
        "num_runs": args.num_runs,
        "seed": seed,
        "optimize": True,
        "planners": PLANNERS,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmpfile:
        json.dump(config, tmpfile, indent=2)
        tmpfile_path = tmpfile.name

    cmd = [
        sys.executable,
        "./examples/run_experiment.py",
        tmpfile_path,
        "--num_processes",
        str(args.num_processes),
    ]

    if not args.no_parallel:
        cmd.append("--parallel_execution")

    try:
        subprocess.run(cmd, check=True)
    finally:
        os.remove(tmpfile_path)

    after = list_matching_experiment_folders(args.experiment_name, env_name)
    new_folders = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if not new_folders:
        print(f"Could not identify output folder for {env_name}")
        return None

    return new_folders[-1]


def main():
    args = parse_args()

    envs = ENVS
    if args.max_envs is not None:
        envs = envs[: args.max_envs]

    all_aggregate_rows = []

    for env_name, seed, default_runtime in envs:
        runtime = args.max_time if args.max_time is not None else default_runtime
        experiment_folder = run_environment(env_name, seed, runtime, args)
        if experiment_folder is None:
            continue

        aggregate_rows = write_summary(experiment_folder)
        all_aggregate_rows.extend(aggregate_rows)

        if not args.no_plots:
            make_plots(experiment_folder, runtime, args)

        print(f"Finished {env_name}. Results: {experiment_folder}")

    summary_path = pathlib.Path("out") / f"{args.experiment_name}_summary_all_envs.csv"
    write_csv(summary_path, all_aggregate_rows)
    print(f"Wrote aggregate summary: {summary_path}")


if __name__ == "__main__":
    main()
