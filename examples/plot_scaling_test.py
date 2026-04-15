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


def make_scaling_plots(data: dict, secondary_label: str = "boxes", log_y: bool = False):
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

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

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

    if log_y:
        axes[0].set_yscale("log")
    fig.tight_layout()
    return fig


def make_per_planner_plot(data: dict, secondary_label: str = "boxes", log_y: bool = False):
    """One subplot per planner: x = secondary key, one colored line per num_robots."""
    planners = sorted(data.keys())
    all_robots = sorted({r for p in data.values() for (r, _) in p})
    all_secondary = sorted({s for p in data.values() for (_, s) in p})

    robot_colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(all_robots), 1)))

    fig, axes = plt.subplots(1, len(planners), figsize=(6 * len(planners), 5), sharey=True)
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
        if log_y:
            ax.set_yscale("log")

    fig.suptitle(f"Scaling with {secondary_label.lower()} (per planner)", y=1.02)
    fig.tight_layout()
    return fig


def make_per_planner_robots_plot(data: dict, secondary_label: str = "tasks", log_y: bool = False):
    """One subplot per planner: x = num_robots, one colored line per secondary key (tasks)."""
    planners = sorted(data.keys())
    all_robots = sorted({r for p in data.values() for (r, _) in p})
    all_secondary = sorted({s for p in data.values() for (_, s) in p})

    task_colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(all_secondary), 1)))

    fig, axes = plt.subplots(1, len(planners), figsize=(6 * len(planners), 5), sharey=True)
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
        if log_y:
            ax.set_yscale("log")

    fig.suptitle(f"Scaling with number of robots (per planner)", y=1.02)
    fig.tight_layout()
    return fig


def load_per_robot_path_lengths(path: str, mode: str = "stacking") -> dict:
    """
    For each scaling experiment folder, load path_0.json (first found solution) from
    every run and compute per-robot path lengths.

    Returns:
        data[(num_robots, secondary_key)][planner_name] = list of per-robot length arrays
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
            ts, key, num_robots = m.group(1), extract_key(m), int(m.group(2))
            if key not in latest or ts > latest[key][0]:
                latest[key] = (ts, entry, num_robots)

    # data[key][planner] = list of (per_robot_lengths, total_max_l1_sum)
    data = defaultdict(lambda: defaultdict(list))

    for key, (_, folder, num_robots) in sorted(latest.items()):
        exp_dir = os.path.join(path, folder, "")
        for planner_name, runs in load_data_from_folder(exp_dir, load_paths=1).items():
            for run in runs:
                if not run.get("paths"):
                    continue
                path_data = run["paths"][0]
                if len(path_data) < 2:
                    continue
                qs = np.array([s["q"] for s in path_data])  # (T, D)
                diffs = np.diff(qs, axis=0)                  # (T-1, D)
                dof = qs.shape[1] // num_robots
                lengths = np.array([
                    np.linalg.norm(diffs[:, r*dof:(r+1)*dof], axis=1).sum()
                    for r in range(num_robots)
                ])
                # sum of max-abs across all joints per edge — proxy for total collision checks
                max_l1_sum = np.max(np.abs(diffs), axis=1).sum()
                data[key][planner_name].append((lengths, max_l1_sum))

    return dict(data)


def make_per_robot_path_length_plot(
    data: dict, secondary_label: str = "tasks", log_y: bool = False
):
    """
    Two rows of subplots, one column per planner.
    Top row: mean per-robot path length (first solution).
    Bottom row: sum of max-L1 distances along path (proxy for total collision checks).
    """
    planners = sorted({p for runs in data.values() for p in runs})
    all_robots = sorted({r for (r, _) in data})
    all_secondary = sorted({s for (_, s) in data})

    task_colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(all_secondary), 1)))

    fig, axes = plt.subplots(
        2, len(planners), figsize=(6 * len(planners), 9),
        sharex=True, squeeze=False,
    )

    for col, planner in enumerate(planners):
        ax_len = axes[0][col]
        ax_l1  = axes[1][col]

        for i, s in enumerate(all_secondary):
            xs_len, meds_len, lbs_len, ubs_len = [], [], [], []
            xs_l1,  meds_l1,  lbs_l1,  ubs_l1  = [], [], [], []

            for r in all_robots:
                runs = data.get((r, s), {}).get(planner, [])
                if not runs:
                    continue

                run_means = [lengths.mean() for lengths, _ in runs]
                med, lb, ub = _median_and_ci(run_means)
                if med is not None:
                    xs_len.append(r)
                    meds_len.append(med)
                    lbs_len.append(med - lb)
                    ubs_len.append(ub - med)

                l1_vals = [max_l1 for _, max_l1 in runs]
                med, lb, ub = _median_and_ci(l1_vals)
                if med is not None:
                    xs_l1.append(r)
                    meds_l1.append(med)
                    lbs_l1.append(med - lb)
                    ubs_l1.append(ub - med)

            label = f"{s} {secondary_label.lower().split()[-1]}"
            color = task_colors[i]

            if xs_len:
                ax_len.errorbar(
                    xs_len, meds_len, yerr=[lbs_len, ubs_len],
                    label=label, color=color, linestyle="-", marker="o", capsize=3,
                )
            if xs_l1:
                ax_l1.errorbar(
                    xs_l1, meds_l1, yerr=[lbs_l1, ubs_l1],
                    label=label, color=color, linestyle="-", marker="o", capsize=3,
                )

        ax_len.set_title(planner)
        ax_len.set_ylabel("Median mean per-robot path length")
        ax_len.legend(title=secondary_label, fontsize=8)
        ax_len.grid(True, alpha=0.3)

        ax_l1.set_xlabel("Number of robots")
        ax_l1.set_ylabel("Median sum of max-L1 dist (collision check proxy)")
        ax_l1.legend(title=secondary_label, fontsize=8)
        ax_l1.grid(True, alpha=0.3)

        if log_y:
            ax_len.set_yscale("log")
            ax_l1.set_yscale("log")

    fig.suptitle("Path length and collision check proxy of first solution", y=1.02)
    fig.tight_layout()
    return fig


def load_timing_data(path: str, mode: str = "stacking") -> dict:
    """
    Loads timing.csv files written by run_experiment for RRT-based planners.

    Returns:
        data[planner_name][(num_robots, secondary_key)] = list of (sampling_time, edge_success_time, edge_failure_time)
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
        exp_dir = os.path.join(path, folder)
        if not os.path.isdir(exp_dir):
            continue
        for planner_name in os.listdir(exp_dir):
            timing_file = os.path.join(exp_dir, planner_name, "timing.csv")
            if not os.path.isfile(timing_file):
                continue

            planner_data = load_data_from_folder(exp_dir + "/")
            first_sol_times = {}
            for run_idx, run in enumerate(planner_data.get(planner_name, [])):
                times = run.get("times", [])
                if times:
                    first_sol_times[run_idx] = times[0]

            with open(timing_file) as f:
                f.readline()  # skip header
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 5:
                        run_id, s, es, ef, cc = parts
                    elif len(parts) == 4:
                        run_id, s, es, ef = parts
                        cc = "0"
                    else:
                        continue
                    t_first = first_sol_times.get(int(run_id))
                    if t_first is None:
                        continue
                    data[planner_name][key].append((float(s), float(es), float(ef), float(cc), t_first))

    return dict(data)


def _timing_plot(
    timing_data: dict,
    secondary_label: str,
    absolute: bool,
    log_y: bool,
):
    """Shared implementation for fraction and absolute timing plots.
    Returns one figure per planner. Layout: 2-column grid over secondary keys.
    Each subplot: x = num_robots, one errorbar line per metric with 95% CI.
    """
    planners = sorted(timing_data.keys())
    if not planners:
        return []

    all_robots = sorted({r for p in timing_data.values() for (r, _) in p})
    all_secondary = sorted({s for p in timing_data.values() for (_, s) in p})

    metric_colors = ["steelblue", "seagreen", "tomato", "darkorange"]
    metric_styles = ["-", "--", ":", "-."]
    metric_markers = ["o", "s", "^", "D"]
    metric_labels = ["sampling", "edge (free)", "edge (blocked)", "single config"]

    ncols = 2
    nrows = int(np.ceil(len(all_secondary) / ncols))

    figs = []
    for planner in planners:
        pdata = timing_data[planner]
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 4 * nrows),
            sharey=True, sharex=True,
            squeeze=False,
        )

        for si, s in enumerate(all_secondary):
            ax = axes[si // ncols][si % ncols]
            for mi in range(4):
                xs, meds, lbs, ubs = [], [], [], []
                for r in all_robots:
                    triples = pdata.get((r, s), [])
                    if not triples:
                        continue
                    if absolute:
                        vals = [t[mi] for t in triples]
                    else:
                        vals = [
                            t[mi] / t[4] if t[4] > 0 else 0.0
                            for t in triples
                        ]
                    med, lb, ub = _median_and_ci(vals)
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
                    color=metric_colors[mi],
                    linestyle=metric_styles[mi],
                    marker=metric_markers[mi],
                    label=metric_labels[mi],
                    capsize=3,
                    markersize=4,
                )

            ax.set_title(f"{secondary_label.split()[-1]}={s}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if not absolute:
                ax.set_ylim(0, 1)
            if log_y:
                ax.set_yscale("log")
            if si // ncols == nrows - 1:
                ax.set_xlabel("Number of robots")
            if si % ncols == 0:
                ax.set_ylabel("Median time [s]" if absolute else "Fraction of tracked time")
            if si == 0:
                ax.legend(fontsize=7)

        # hide unused subplots
        for si in range(len(all_secondary), nrows * ncols):
            axes[si // ncols][si % ncols].set_visible(False)

        title = (
            f"{planner} — absolute time: sampling vs edge collision"
            if absolute
            else f"{planner} — fraction of time: sampling vs edge collision"
        )
        fig.suptitle(title)
        fig.tight_layout()
        figs.append((planner, fig))

    return figs


def make_timing_breakdown_plot(
    timing_data: dict, secondary_label: str = "tasks", log_y: bool = False
):
    """Fraction of tracked time per component, split by number of tasks."""
    return _timing_plot(timing_data, secondary_label, absolute=False, log_y=log_y)


def make_timing_absolute_plot(
    timing_data: dict, secondary_label: str = "tasks", log_y: bool = False
):
    """Absolute median time per component, split by number of tasks."""
    return _timing_plot(timing_data, secondary_label, absolute=True, log_y=log_y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stacking", "mobile"], default="mobile")
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--log_y", action="store_true", help="Use logarithmic y-axis")
    args = parser.parse_args()

    if args.path is not None:
        path = args.path
    elif args.mode == "stacking":
        path = "out/stacking_scaling"
    else:
        path = "out/mobile_scaling"

    secondary_label = "Number of tasks"

    data = load_scaling_data(path, mode=args.mode)

    fig = make_scaling_plots(data, secondary_label=secondary_label, log_y=args.log_y)
    fig.savefig(os.path.join(path, "scaling_plots.pdf"), bbox_inches="tight")

    fig2 = make_per_planner_plot(data, secondary_label=secondary_label, log_y=args.log_y)
    fig2.savefig(os.path.join(path, "scaling_plots_per_planner.pdf"), bbox_inches="tight")

    fig3 = make_per_planner_robots_plot(data, secondary_label=secondary_label, log_y=args.log_y)
    fig3.savefig(os.path.join(path, "scaling_plots_per_planner_robots.pdf"), bbox_inches="tight")

    path_length_data = load_per_robot_path_lengths(path, mode=args.mode)
    fig4 = make_per_robot_path_length_plot(path_length_data, secondary_label=secondary_label, log_y=args.log_y)
    fig4.savefig(os.path.join(path, "scaling_plots_per_robot_path_lengths.pdf"), bbox_inches="tight")

    timing_data = load_timing_data(path, mode=args.mode)
    if timing_data:
        for planner, fig in make_timing_breakdown_plot(timing_data, secondary_label=secondary_label, log_y=args.log_y):
            fig.savefig(os.path.join(path, f"scaling_plots_timing_fraction_{planner}.pdf"), bbox_inches="tight")

        for planner, fig in make_timing_absolute_plot(timing_data, secondary_label=secondary_label, log_y=args.log_y):
            fig.savefig(os.path.join(path, f"scaling_plots_timing_absolute_{planner}.pdf"), bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
