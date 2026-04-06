import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_plots import load_data_from_folder, load_config_from_folder, planner_name_to_color
from compute_confidence_intervals import computeConfidenceInterval


def load_scaling_data(path: str) -> dict:
    """
    Scans `path` for experiment folders named like
      <timestamp>_scaling_stacking_r<N>_b<M>/
    and loads the latest run per (num_robots, num_boxes).

    Returns:
      data[planner_name][(num_robots, num_boxes)] = list of initial solution times
    """
    pattern = re.compile(r"(\d{8}_\d{6})_scaling_stacking_r(\d+)_b(\d+)")

    latest: dict = {}
    for entry in os.listdir(path):
        m = pattern.match(entry)
        if m:
            ts, r, b = m.group(1), int(m.group(2)), int(m.group(3))
            key = (r, b)
            if key not in latest or ts > latest[key][0]:
                latest[key] = (ts, entry)

    data = defaultdict(lambda: defaultdict(list))

    for (r, b), (_, folder) in sorted(latest.items()):
        exp_dir = os.path.join(path, folder, "")
        exp_data = load_data_from_folder(exp_dir)

        for planner_name, runs in exp_data.items():
            for run in runs:
                if run.get("times"):
                    data[planner_name][(r, b)].append(run["times"][0])

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


def make_scaling_plots(data: dict):
    planners = sorted(data.keys())
    all_robots = sorted({r for p in data.values() for (r, _) in p})
    all_boxes = sorted({b for p in data.values() for (_, b) in p})

    # Assign colors: use existing map if available, else cycle tab10
    fallback_colors = plt.cm.tab10.colors
    fallback_idx = 0
    colors = {}
    for p in planners:
        if p in planner_name_to_color:
            colors[p] = planner_name_to_color[p]
        else:
            colors[p] = fallback_colors[fallback_idx % len(fallback_colors)]
            fallback_idx += 1

    # Line styles cycle for the secondary grouping variable
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "p"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Plot 1: initial solution time vs num_boxes ---
    ax = axes[0]
    for planner in planners:
        for i, r in enumerate(all_robots):
            xs, meds, lbs, ubs = [], [], [], []
            for b in all_boxes:
                times = data[planner].get((r, b), [])
                med, lb, ub = _median_and_ci(times)
                if med is None:
                    continue
                xs.append(b)
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
                alpha=0.5 + 0.15 * i,
            )
    ax.set_xlabel("Number of boxes")
    ax.set_ylabel("Median initial solution time [s]")
    ax.set_title("Scaling with number of boxes")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: initial solution time vs num_robots ---
    ax = axes[1]
    for planner in planners:
        for i, b in enumerate(all_boxes):
            xs, meds, lbs, ubs = [], [], [], []
            for r in all_robots:
                times = data[planner].get((r, b), [])
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
                label=f"{planner}, b={b}",
                color=colors[planner],
                linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)],
                capsize=3,
                alpha=0.4 + 0.07 * i,
            )
    ax.set_xlabel("Number of robots")
    ax.set_ylabel("Median initial solution time [s]")
    ax.set_title("Scaling with number of robots")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def make_per_planner_boxes_plot(data: dict):
    """One subplot per planner: x = num_boxes, one colored line per num_robots."""
    planners = sorted(data.keys())
    all_robots = sorted({r for p in data.values() for (r, _) in p})
    all_boxes = sorted({b for p in data.values() for (_, b) in p})

    robot_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_robots)))

    fig, axes = plt.subplots(1, len(planners), figsize=(6 * len(planners), 5), sharey=False)
    if len(planners) == 1:
        axes = [axes]

    for ax, planner in zip(axes, planners):
        for i, r in enumerate(all_robots):
            xs, meds, lbs, ubs = [], [], [], []
            for b in all_boxes:
                times = data[planner].get((r, b), [])
                med, lb, ub = _median_and_ci(times)
                if med is None:
                    continue
                xs.append(b)
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
        ax.set_xlabel("Number of boxes")
        ax.set_ylabel("Median initial solution time [s]")
        ax.set_title(planner)
        ax.legend(title="Robots", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Scaling with number of boxes (per planner)", y=1.02)
    fig.tight_layout()
    return fig


def main():
    path = "out/stacking_scaling/"

    data = load_scaling_data(path)
    fig = make_scaling_plots(data)
    fig.savefig(os.path.join(path, "scaling_plots.pdf"), bbox_inches="tight")

    fig2 = make_per_planner_boxes_plot(data)
    fig2.savefig(os.path.join(path, "scaling_plots_per_planner.pdf"), bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
