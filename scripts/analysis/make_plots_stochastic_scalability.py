import argparse
import pathlib
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from make_plots_stochastic_skill import (
    FLAT,
    PLANNER_STYLE,
    apply_style,
    interpolate_costs,
    load_experiment,
    median_and_band,
    planners_in_order,
    save_legend,
)


def robot_count(env_name: str):
    match = re.search(r"_(\d+)r_\d+b", env_name)
    return int(match.group(1)) if match else None


def load_folders(folders):
    out = []
    for folder in folders:
        folder = pathlib.Path(folder)
        if not folder.is_dir():
            print(f"  [skip] {folder} is not a directory")
            continue
        experiment = load_experiment(folder)
        if experiment["planners"]:
            out.append((experiment["env"], experiment))
        else:
            print(f"  [skip] {folder}: no planner data")
    return out


def _infer_robot_count(experiment):
    for _, _, _, _, data in planners_in_order(experiment):
        for run_execs in data.get("per_run") or [data.get("executions", [])]:
            for r in run_execs:
                timeline = r.get("timeline")
                if timeline:
                    return len(timeline[0]["tasks"])
    return None


def _robots(env_name, experiment):
    return robot_count(env_name) or _infer_robot_count(experiment)


def plot_scalability_overlay(experiments, out_path, legend=True):
    ordered = sorted(experiments, key=lambda item: (_robots(*item) or 0, item[0]))
    if not ordered:
        print("No experiments to draw.")
        return

    LINESTYLES = ["-", "--", ":", "-."]
    if len(ordered) > len(LINESTYLES):
        print(f"  [warn] {len(ordered)} environments but only {len(LINESTYLES)} linestyles -- "
              f"they will repeat.")

    series = []
    for i, (env_name, experiment) in enumerate(ordered):
        ls = LINESTYLES[i % len(LINESTYLES)]
        robots = _robots(env_name, experiment)
        suffix = f"{robots}R" if robots else env_name
        for name, kind, label, color, data in planners_in_order(experiment):
            runs = [(np.asarray(t), np.asarray(c)) for t, c in data["curves"] if len(t) > 1]
            if runs:
                series.append((f"{label} {suffix}", color, ls, runs))

    if not series:
        print("No anytime curves in any folder.")
        return

    t_max = max(t[-1] for _, _, _, runs in series for t, _ in runs)
    grid = np.arange(0.0, t_max, 1e-2)

    fig, ax = plt.subplots(figsize=(FLAT[0], FLAT[1] * 1.3))
    for label, color, ls, runs in series:
        stacked = [interpolate_costs(grid, t, c) for t, c in runs]
        median, lb, ub = median_and_band(stacked)
        drawn = np.isfinite(median)
        ax.plot(grid[drawn], median[drawn], color=color, ls=ls, label=label)
        if lb is not None:
            band = drawn & np.isfinite(lb) & np.isfinite(ub)
            ax.fill_between(grid[band], lb[band], ub[band], color=color, alpha=0.18, lw=0)

    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    decades = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
    candidates = [m * 10.0 ** k for k in decades for m in (1, 2, 3, 4, 5, 6, 8)]
    ticks = [v for v in candidates if lo <= v <= hi]
    ax.set_yticks(ticks or [t for t in MaxNLocator(nbins=6).tick_values(lo, hi) if lo <= t <= hi])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Planning time [s]")
    ax.set_ylabel("Cost")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        save_legend(handles, labels, out_path.parent / f"legend_{out_path.stem}{out_path.suffix}",
                   ncol=2)
    fig.savefig(out_path)
    plt.close(fig)


def plot_scalability(experiments, out_path, legend=True):
    ordered = sorted(experiments, key=lambda item: (_robots(*item) or 0, item[0]))
    if not ordered:
        print("No experiments to draw.")
        return

    fig, axes = plt.subplots(len(ordered), 1, sharex=True,
                             figsize=(FLAT[0], FLAT[1] * 0.85 * len(ordered)),
                             gridspec_kw={"hspace": 0.25})
    axes = np.atleast_1d(axes)

    drew_any = False
    for ax, (env_name, experiment) in zip(axes, ordered):
        series = []
        for name, kind, label, color, data in planners_in_order(experiment):
            runs = [(np.asarray(t), np.asarray(c)) for t, c in data["curves"] if len(t) > 1]
            if runs:
                series.append((label, color, runs))
        if not series:
            ax.text(0.5, 0.5, f"no curves for {env_name}", transform=ax.transAxes,
                    ha="center", va="center", color="0.5")
            continue

        t_max = max(t[-1] for _, _, runs in series for t, _ in runs)
        grid = np.arange(0.0, t_max, 1e-2)
        for label, color, runs in series:
            stacked = [interpolate_costs(grid, t, c) for t, c in runs]
            median, lb, ub = median_and_band(stacked)
            drawn = np.isfinite(median)
            ax.plot(grid[drawn], median[drawn], color=color, label=label)
            if lb is not None:
                band = drawn & np.isfinite(lb) & np.isfinite(ub)
                ax.fill_between(grid[band], lb[band], ub[band], color=color, alpha=0.25, lw=0)
        drew_any = True

        robots = _robots(env_name, experiment)
        ax.text(0.985, 0.88, f"{robots} robots" if robots else env_name,
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.35")
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ax.set_yticks([t for t in MaxNLocator(nbins=3).tick_values(lo, hi) if lo <= t <= hi])
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        ax.get_yaxis().set_minor_formatter(plt.NullFormatter())
        ax.set_ylabel("Cost")

    if not drew_any:
        print("No anytime curves in any folder.")
        plt.close(fig)
        return

    axes[-1].set_xlabel("Planning time [s]")
    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=4))
    if legend:
        handles, labels = axes[-1].get_legend_handles_labels()
        save_legend(handles, labels, out_path.parent / f"legend_{out_path.stem}{out_path.suffix}")
    fig.savefig(out_path)
    plt.close(fig)


def execution_times(data):
    if data.get("nominal"):
        return []
    times = []
    for run_execs in data.get("per_run") or [data.get("executions", [])]:
        for r in run_execs:
            if r.get("success") and r.get("timeline"):
                times.append(r["timeline"][-1].get("real_time", r["timeline"][-1]["time"]))
    return times


def plot_execution_boxplot(experiments, out_path):
    ordered = sorted(experiments, key=lambda item: (_robots(*item) or 0, item[0]))
    if not ordered:
        print("No experiments to draw.")
        return

    kinds_present = []
    for _, experiment in ordered:
        for _, kind, _, _, _ in planners_in_order(experiment):
            if kind not in kinds_present:
                kinds_present.append(kind)
    kinds_present = [k for k in PLANNER_STYLE if k in kinds_present]
    if not kinds_present:
        print("No planner data in any folder.")
        return

    x_labels = [f"{_robots(env, experiment) or '?'}R" for env, experiment in ordered]

    positions = np.arange(len(ordered))
    n_kinds = len(kinds_present)
    width = 0.8 / n_kinds

    fig, ax = plt.subplots(figsize=(1.6 * len(ordered) + 2, 4))
    for k, kind in enumerate(kinds_present):
        plabel, color = PLANNER_STYLE[kind]
        offset = (k - (n_kinds - 1) / 2) * width

        data_per_env = []
        for env, experiment in ordered:
            times, planner_data = [], None
            for _, ekind, _, _, data in planners_in_order(experiment):
                if ekind == kind:
                    times, planner_data = execution_times(data), data
                    break
            if not times:
                why = ("never executed" if planner_data and planner_data.get("nominal")
                       else "no successful executions" if planner_data is not None
                       else "no data")
                print(f"  [empty] {plabel} on {env}: {why}")
            data_per_env.append(times if times else [np.nan])

        box = ax.boxplot(data_per_env, positions=positions + offset, widths=width * 0.85,
                         patch_artist=True, showfliers=True)
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        for median in box["medians"]:
            median.set(color="black", linewidth=1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Elapsed execution time [s]")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(1.2)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("folder_paths", nargs="+",
                        help="one experiment folder per robot count, in the same family "
                             "(e.g. dep_stochastic_stacking_2r_4b / _3r_4b / _4r_4b)")
    parser.add_argument("--out", default=None,
                        help="output directory; default is <first folder>/plots_stochastic, "
                             "matching where make_plots_stochastic_skill.py already writes "
                             "that folder's other figures")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--no_legend", action="store_true")
    parser.add_argument("--paper", action="store_true",
                        help="Use paper_2.mplstyle (requires a LaTeX install)")
    args = parser.parse_args()

    experiments = load_folders(args.folder_paths)
    if not experiments:
        print("No usable experiment folders.")
        return

    out_dir = (pathlib.Path(args.out) if args.out
              else pathlib.Path(args.folder_paths[0]) / "plots_stochastic")
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "pdf" if args.pdf else "png"
    apply_style(args.paper)
    legend = not args.no_legend

    print(f"Read {len(experiments)} environment(s): "
          f"{', '.join(env for env, _ in experiments)}")
    plot_scalability(experiments, out_dir / f"scalability.{ext}", legend)
    plot_scalability_overlay(experiments, out_dir / f"scalability_overlay.{ext}", legend)
    plot_execution_boxplot(experiments, out_dir / f"execution_boxplot.{ext}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
