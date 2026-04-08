import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_plots import load_data_from_folder, load_config_from_folder, planner_name_to_color
from compute_confidence_intervals import computeConfidenceInterval


def load_scaling_data(path: str, mode: str = "stacking") -> dict:
    """
    Scans `path` for experiment folders and loads the latest run per (num_robots, secondary_key).

    mode="stacking": folders like <ts>_scaling_stacking_r<N>_b<M>
        secondary key = num_boxes
    mode="mobile": folders like <ts>_scaling_mobile_r<N>_x<X>_z<Z>
        secondary key = x * z  (wall area)

    Returns:
        data[planner_name][(num_robots, secondary_key)] = list of initial solution times
    """
    if mode == "stacking":
        pattern = re.compile(r"(\d{8}_\d{6})_scaling_stacking_r(\d+)_b(\d+)")
        def extract_key(m):
            return int(m.group(2)), int(m.group(3)) * 2
    elif mode == "mobile":
        pattern = re.compile(r"(\d{8}_\d{6})_scaling_mobile_r(\d+)_x(\d+)_z(\d+)")
        def extract_key(m):
            return int(m.group(2)), int(m.group(3)) * int(m.group(4)) * 2
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    latest: dict = {}
    for entry in os.listdir(path):
        m = pattern.match(entry)
        if m:
            ts = m.group(1)
            key = extract_key(m)
            if key not in latest or ts > latest[key][0]:
                latest[key] = (ts, entry)

    data = defaultdict(lambda: defaultdict(list))

    for key, (_, folder) in sorted(latest.items()):
        exp_dir = os.path.join(path, folder, "")
        exp_data = load_data_from_folder(exp_dir)

        for planner_name, runs in exp_data.items():
            for run in runs:
                if run.get("times"):
                    data[planner_name][key].append(run["times"][0])

    return dict(data)


def _median_and_ci(times):
    """Returns (median, lower_bound, upper_bound) using computeConfidenceInterval."""
    if not times:
        return None, None, None
    n = len(times)
    med = np.median(times)
    lb_idx, ub_idx, _ = computeConfidenceInterval(n, 0.95)
    sorted_times = np.sort(times)
    lb = sorted_times[lb_idx]
    ub = sorted_times[ub_idx - 1]
    return med, lb, ub


def make_scaling_plots(data: dict, secondary_label: str = "boxes"):
    planners = sorted(data.keys())
    all_robots = sorted({r for p in data.values() for (r, _) in p})
    all_secondary = sorted({s for p in data.values() for (_, s) in p})

    fallback_colors = plt.cm.tab10.colors
    fallback_idx = 0
    colors = {}
    for p in planners:
        if p in planner_name_to_color:
            colors[p] = planner_name_to_color[p]
        else:
            colors[p] = fallback_colors[fallback_idx % len(fallback_colors)]
            fallback_idx += 1

    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "p"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Plot 1: initial solution time vs secondary (boxes / wall area) ---
    ax = axes[0]
    for planner in planners:
        alphas = np.linspace(0.4, 0.9, max(len(all_robots), 1))
        for i, r in enumerate(all_robots):
            xs, meds, lbs, ubs = [], [], [], []
            for s in all_secondary:
                times = data[planner].get((r, s), [])
                med, lb, ub = _median_and_ci(times)
                if med is None:
                    continue
                xs.append(s)
                meds.append(med)
                lbs.append(med - lb)
                ubs.append(ub - med)
            if not xs:
                continue
            ax.errorbar(
                xs, meds,
                yerr=[lbs, ubs],
                label=f"{planner}, r={r}",
                color=colors[planner],
                linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)],
                capsize=3,
                alpha=alphas[i],
            )
    ax.set_xlabel(secondary_label)
    ax.set_ylabel("Median initial solution time [s]")
    ax.set_title(f"Scaling with {secondary_label.lower()}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: initial solution time vs num_robots ---
    ax = axes[1]
    for planner in planners:
        alphas = np.linspace(0.4, 0.9, max(len(all_secondary), 1))
        for i, s in enumerate(all_secondary):
            xs, meds, lbs, ubs = [], [], [], []
            for r in all_robots:
                times = data[planner].get((r, s), [])
                med, lb, ub = _median_and_ci(times)
                if med is None:
                    continue
                xs.append(r)
                meds.append(med)
                lbs.append(med - lb)
                ubs.append(ub - med)
            if not xs:
                continue
            ax.errorbar(
                xs, meds,
                yerr=[lbs, ubs],
                label=f"{planner}, {secondary_label.lower()[:1]}={s}",
                color=colors[planner],
                linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)],
                capsize=3,
                alpha=alphas[i],
            )
    ax.set_xlabel("Number of robots")
    ax.set_ylabel("Median initial solution time [s]")
    ax.set_title("Scaling with number of robots")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def make_per_planner_plot(data: dict, secondary_label: str = "boxes"):
    """One subplot per planner: x = secondary key, one colored line per num_robots."""
    planners = sorted(data.keys())
    all_robots = sorted({r for p in data.values() for (r, _) in p})
    all_secondary = sorted({s for p in data.values() for (_, s) in p})

    robot_colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(all_robots), 1)))

    fig, axes = plt.subplots(1, len(planners), figsize=(6 * len(planners), 5), sharey=False)
    if len(planners) == 1:
        axes = [axes]

    for ax, planner in zip(axes, planners):
        for i, r in enumerate(all_robots):
            xs, meds, lbs, ubs = [], [], [], []
            for s in all_secondary:
                times = data[planner].get((r, s), [])
                med, lb, ub = _median_and_ci(times)
                if med is None:
                    continue
                xs.append(s)
                meds.append(med)
                lbs.append(med - lb)
                ubs.append(ub - med)
            if not xs:
                continue
            ax.errorbar(
                xs, meds,
                yerr=[lbs, ubs],
                label=f"{r} robot{'s' if r > 1 else ''}",
                color=robot_colors[i],
                linestyle="-",
                marker="o",
                capsize=3,
            )
        ax.set_xlabel(secondary_label)
        ax.set_ylabel("Median initial solution time [s]")
        ax.set_title(planner)
        ax.legend(title="Robots", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Scaling with {secondary_label.lower()} (per planner)", y=1.02)
    fig.tight_layout()
    return fig


def make_per_planner_robots_plot(data: dict, secondary_label: str = "tasks"):
    """One subplot per planner: x = num_robots, one colored line per secondary key (tasks)."""
    planners = sorted(data.keys())
    all_robots = sorted({r for p in data.values() for (r, _) in p})
    all_secondary = sorted({s for p in data.values() for (_, s) in p})

    task_colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(all_secondary), 1)))

    fig, axes = plt.subplots(1, len(planners), figsize=(6 * len(planners), 5), sharey=False)
    if len(planners) == 1:
        axes = [axes]

    for ax, planner in zip(axes, planners):
        for i, s in enumerate(all_secondary):
            xs, meds, lbs, ubs = [], [], [], []
            for r in all_robots:
                times = data[planner].get((r, s), [])
                med, lb, ub = _median_and_ci(times)
                if med is None:
                    continue
                xs.append(r)
                meds.append(med)
                lbs.append(med - lb)
                ubs.append(ub - med)
            if not xs:
                continue
            ax.errorbar(
                xs, meds,
                yerr=[lbs, ubs],
                label=f"{s} {secondary_label.lower().split()[-1]}",
                color=task_colors[i],
                linestyle="-",
                marker="o",
                capsize=3,
            )
        ax.set_xlabel("Number of robots")
        ax.set_ylabel("Median initial solution time [s]")
        ax.set_title(planner)
        ax.legend(title=secondary_label, fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Scaling with number of robots (per planner)", y=1.02)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stacking", "mobile"], default="mobile")
    parser.add_argument("--path", type=str, default=None)
    args = parser.parse_args()

    if args.path is not None:
        path = args.path
    elif args.mode == "stacking":
        path = "out/stacking_scaling"
    else:
        path = "out/mobile_scaling"

    secondary_label = "Number of tasks"

    data = load_scaling_data(path, mode=args.mode)

    fig = make_scaling_plots(data, secondary_label=secondary_label)
    fig.savefig(os.path.join(path, "scaling_plots.pdf"), bbox_inches="tight")

    fig2 = make_per_planner_plot(data, secondary_label=secondary_label)
    fig2.savefig(os.path.join(path, "scaling_plots_per_planner.pdf"), bbox_inches="tight")

    fig3 = make_per_planner_robots_plot(data, secondary_label=secondary_label)
    fig3.savefig(os.path.join(path, "scaling_plots_per_planner_robots.pdf"), bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
