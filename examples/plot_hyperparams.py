"""
Plot hyperparam ablation results.

For each ablation experiment folder, produces two figures:
  - initial solution time  vs each swept parameter
  - initial solution cost  vs each swept parameter

Each figure has one subplot per swept axis (parameter), showing all variant
values as points with 95% CI error bars. The default variant is highlighted.
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_plots import load_data_from_folder, load_config_from_folder, interpolate_costs
from compute_confidence_intervals import computeConfidenceInterval


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _median_and_ci(values):
    """Returns (median, lower_bound, upper_bound) using 95% CI."""
    if not values:
        return None, None, None
    n = len(values)
    med = np.median(values)
    lb_idx, ub_idx, _ = computeConfidenceInterval(n, 0.95)
    sv = np.sort(values)
    return med, sv[lb_idx], sv[ub_idx - 1]


def load_ablation_data(folder: str) -> dict:
    """
    Loads all variant subfolders in an ablation experiment folder.

    Returns:
      variant_name -> {
        "times":  [initial_solution_time, ...]   one per run
        "costs":  [initial_solution_cost, ...]   one per run
        "runs":   [{"times": [...], "costs": [...]}, ...]  full series per run
      }
    """
    all_data = load_data_from_folder(folder)
    result = {}
    for variant_name, runs in all_data.items():
        times, costs = [], []
        for run in runs:
            if run.get("times"):
                times.append(run["times"][0])
                costs.append(run["costs"][0])
        result[variant_name] = {"times": times, "costs": costs, "runs": runs}
    return result


def axis_name_from_variant(variant_name: str) -> str:
    """
    Derive the swept axis name from a variant name string.
    'default'                    -> 'default'
    'informed_sampling=True'     -> 'informed_sampling'
    'p_goal=0.4'                 -> 'p_goal'
    'key1=v1_key2=v2' (coupled)  -> 'key1+key2'
    """
    if variant_name == "default":
        return "default"
    # Split on _ but only at key=value boundaries
    # Each token looks like "word...=value" where value has no '='
    parts = variant_name.split("=")
    # parts[0] is first key; parts[-1] is last value; middle parts are "value_nextkey"
    keys = [parts[0]]
    for p in parts[1:-1]:
        # last word after _ is the next key
        keys.append(p.rsplit("_", 1)[-1])
    return "+".join(keys)


def group_variants_by_axis(variant_names: list) -> dict:
    """
    Groups variant names by their swept axis.
    'default' is placed in every axis group as the reference point.

    Returns: axis_name -> [variant_names]
    """
    groups = {}
    defaults = [n for n in variant_names if n == "default"]

    for name in variant_names:
        if name == "default":
            continue
        ax = axis_name_from_variant(name)
        groups.setdefault(ax, list(defaults))
        groups[ax].append(name)

    return groups


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _short_label(variant_name: str, axis_name: str) -> str:
    """Strip the repeated key prefix from a variant label for a cleaner axis tick."""
    if variant_name == "default":
        return "default"
    label = variant_name
    for key in axis_name.split("+"):
        label = label.replace(f"{key}=", "")
    return label


def _parse_numeric(label: str):
    """Try to parse a label string as float. Returns float or None."""
    try:
        return float(label)
    except ValueError:
        return None


def _plot_axis(ax, variants_in_axis, axis_name, data, metric, ylabel):
    """
    Plot one swept axis into `ax`.
    Numeric axes: x = actual parameter value, with a line connecting points.
    Categorical axes: x = integer index with string tick labels.
    The default variant is highlighted in orange.
    """
    points = []  # (x_or_idx, med, lo_err, hi_err, label, is_default)
    for vname in variants_in_axis:
        vals = data.get(vname, {}).get(metric, [])
        med, lb, ub = _median_and_ci(vals)
        if med is None:
            continue
        label = _short_label(vname, axis_name)
        points.append((label, med, med - lb, ub - med, vname == "default"))

    if not points:
        ax.set_visible(False)
        return

    labels = [p[0] for p in points]
    numeric_vals = [_parse_numeric(l) for l in labels]
    is_numeric = all(
        v is not None for l, v in zip(labels, numeric_vals) if l != "default"
    )

    if is_numeric and any(l != "default" for l in labels):
        # Build (x, med, lo, hi, is_default) sorted by numeric x for non-default points
        non_default = [(nv, p[1], p[2], p[3]) for nv, p in zip(numeric_vals, points) if not p[4]]
        non_default.sort(key=lambda t: t[0])
        xs_nd, meds_nd, lo_nd, hi_nd = zip(*non_default) if non_default else ([], [], [], [])

        ax.plot(xs_nd, meds_nd, "-", color="tab:blue", zorder=2)
        ax.errorbar(xs_nd, meds_nd, yerr=[lo_nd, hi_nd],
                    fmt="o", capsize=4, color="tab:blue", zorder=3)

        # Default: mark as vertical line + point if it has a numeric value
        for nv, p in zip(numeric_vals, points):
            if p[4] and nv is not None:
                ax.axvline(nv, color="tab:orange", linestyle="--", alpha=0.7, zorder=1)
                ax.errorbar([nv], [p[1]], yerr=[[p[2]], [p[3]]],
                            fmt="o", capsize=4, color="tab:orange", zorder=5,
                            label="default")

        ax.set_xlabel(axis_name, fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        # Categorical
        xs = list(range(len(points)))
        meds = [p[1] for p in points]
        lo_errs = [p[2] for p in points]
        hi_errs = [p[3] for p in points]
        colors = ["tab:orange" if p[4] else "tab:blue" for p in points]

        ax.errorbar(xs, meds, yerr=[lo_errs, hi_errs], fmt="o", capsize=4, color="tab:blue")
        for x, med, lo, hi, c in zip(xs, meds, lo_errs, hi_errs, colors):
            if c != "tab:blue":
                ax.errorbar([x], [med], yerr=[[lo], [hi]], fmt="o", capsize=4,
                            color=c, zorder=5)

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    ax.set_title(axis_name, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)


def make_ablation_figure(data: dict, metric: str, title: str) -> plt.Figure:
    """
    One subplot per swept axis.
    Numeric axes are plotted with a continuous x-axis and connecting line.
    Categorical/boolean axes use index-based x with string tick labels.

    metric: "times" or "costs"
    """
    variant_names = list(data.keys())
    groups = group_variants_by_axis(variant_names)
    axes_names = sorted(groups.keys())

    n_axes = len(axes_names)
    if n_axes == 0:
        return None

    ncols = min(3, n_axes)
    nrows = math.ceil(n_axes / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    ylabel = "Median initial solution time [s]" if metric == "times" else "Median initial solution cost"

    for ax, axis_name in zip(axes, axes_names):
        _plot_axis(ax, groups[axis_name], axis_name, data, metric, ylabel)

    # Hide unused subplots
    for ax in axes[n_axes:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convergence plots
# ---------------------------------------------------------------------------

# Colour cycle for variants within a subplot
_VARIANT_COLORS = plt.cm.tab10.colors


def make_convergence_curves_figure(data: dict, max_time: float, title: str) -> plt.Figure:
    """
    Option A: one subplot per swept axis, one median cost-over-time curve per variant.
    The default variant is drawn in orange with a dashed line.
    """
    variant_names = list(data.keys())
    groups = group_variants_by_axis(variant_names)
    axes_names = sorted(groups.keys())
    n_axes = len(axes_names)
    if n_axes == 0:
        return None

    time_grid = np.arange(0, max_time, max_time / 500)

    ncols = min(3, n_axes)
    nrows = math.ceil(n_axes / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False)
    axes = np.array(axes).flatten()

    for ax, axis_name in zip(axes, axes_names):
        variants_in_axis = groups[axis_name]
        for ci, vname in enumerate(variants_in_axis):
            runs = data.get(vname, {}).get("runs", [])
            if not runs:
                continue

            all_costs = []
            for run in runs:
                rtimes, rcosts = run.get("times", []), run.get("costs", [])
                if not rtimes:
                    continue
                all_costs.append(interpolate_costs(time_grid, rtimes, rcosts))

            if not all_costs:
                continue

            all_costs = np.array(all_costs)
            median_costs = np.median(all_costs, axis=0)
            lb_idx, ub_idx, _ = computeConfidenceInterval(len(all_costs), 0.95)
            sorted_costs = np.sort(all_costs, axis=0)
            lb_costs = sorted_costs[lb_idx]
            ub_costs = sorted_costs[ub_idx - 1]

            is_default = vname == "default"
            color = "tab:orange" if is_default else _VARIANT_COLORS[ci % len(_VARIANT_COLORS)]
            ls = "--" if is_default else "-"
            lw = 2 if is_default else 1
            label = _short_label(vname, axis_name)

            valid = np.isfinite(median_costs)
            ax.plot(time_grid[valid], median_costs[valid], color=color, ls=ls, lw=lw, label=label)
            ub_plot = np.where(np.isfinite(ub_costs), ub_costs, np.nan)
            ax.fill_between(time_grid, lb_costs, ub_plot, alpha=0.15, color=color)

        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_ylabel("Cost", fontsize=8)
        ax.set_title(axis_name, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes[n_axes:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def make_time_to_fraction_figure(
    data: dict, max_time: float, fractions: list, title: str
) -> plt.Figure:
    """
    Option B: for each variant compute the time to reach (fraction * best_final_cost).
    One subplot per fraction threshold, one point+CI per variant, grouped by axis.

    fractions: list of floats, e.g. [1.0, 1.1, 1.5]  (1.0 = best final cost found)
    """
    variant_names = list(data.keys())
    groups = group_variants_by_axis(variant_names)
    axes_names = sorted(groups.keys())
    n_axes = len(axes_names)
    if n_axes == 0:
        return None

    time_grid = np.arange(0, max_time, max_time / 500)

    # Compute best final cost across all variants (lower envelope of all runs)
    best_final = np.inf
    for vdata in data.values():
        for run in vdata.get("runs", []):
            rcosts = run.get("costs", [])
            if rcosts:
                best_final = min(best_final, rcosts[-1])

    n_fracs = len(fractions)
    ncols = min(3, n_axes)
    nrows = math.ceil(n_axes / ncols)
    fig, axes = plt.subplots(
        nrows * n_fracs, ncols,
        figsize=(5 * ncols, 3 * nrows * n_fracs),
        sharey=False,
        squeeze=False,
    )

    for fi, frac in enumerate(fractions):
        target = frac * best_final
        row_offset = fi * nrows

        for col_i, axis_name in enumerate(axes_names):
            row_i = col_i // ncols + row_offset
            col_j = col_i % ncols
            ax = axes[row_i][col_j]

            variants_in_axis = groups[axis_name]
            points = []
            for vname in variants_in_axis:
                runs = data.get(vname, {}).get("runs", [])
                ttf_list = []
                for run in runs:
                    rtimes, rcosts = run.get("times", []), run.get("costs", [])
                    if not rtimes:
                        continue
                    interp = interpolate_costs(time_grid, rtimes, rcosts)
                    reached = np.where(interp <= target)[0]
                    ttf_list.append(time_grid[reached[0]] if len(reached) else max_time)
                med, lb, ub = _median_and_ci(ttf_list)
                if med is None:
                    continue
                label = _short_label(vname, axis_name)
                points.append((label, med, med - lb, ub - med, vname == "default"))

            if not points:
                ax.set_visible(False)
                continue

            labels = [p[0] for p in points]
            numeric_vals = [_parse_numeric(l) for l in labels]
            is_numeric = all(
                v is not None for l, v in zip(labels, numeric_vals) if l != "default"
            )

            if is_numeric and any(l != "default" for l in labels):
                non_default = [
                    (nv, p[1], p[2], p[3])
                    for nv, p in zip(numeric_vals, points) if not p[4]
                ]
                non_default.sort(key=lambda t: t[0])
                xs_nd, meds_nd, lo_nd, hi_nd = zip(*non_default)
                ax.plot(xs_nd, meds_nd, "-", color="tab:blue", zorder=2)
                ax.errorbar(xs_nd, meds_nd, yerr=[lo_nd, hi_nd],
                            fmt="o", capsize=4, color="tab:blue", zorder=3)
                for nv, p in zip(numeric_vals, points):
                    if p[4] and nv is not None:
                        ax.axvline(nv, color="tab:orange", ls="--", alpha=0.7)
                        ax.errorbar([nv], [p[1]], yerr=[[p[2]], [p[3]]],
                                    fmt="o", capsize=4, color="tab:orange", zorder=5)
                ax.set_xlabel(axis_name, fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                xs = list(range(len(points)))
                meds = [p[1] for p in points]
                lo_errs = [p[2] for p in points]
                hi_errs = [p[3] for p in points]
                colors = ["tab:orange" if p[4] else "tab:blue" for p in points]
                ax.errorbar(xs, meds, yerr=[lo_errs, hi_errs], fmt="o", capsize=4, color="tab:blue")
                for x, med, lo, hi, c in zip(xs, meds, lo_errs, hi_errs, colors):
                    if c != "tab:blue":
                        ax.errorbar([x], [med], yerr=[[lo], [hi]], fmt="o", capsize=4,
                                    color=c, zorder=5)
                ax.set_xticks(xs)
                ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
                ax.grid(True, axis="y", alpha=0.3)

            ax.set_title(f"{axis_name}  (≤ {frac:.1f}× best)", fontsize=9)
            ax.set_ylabel("Time to reach target [s]", fontsize=8)

    # Hide trailing empty cells
    for row in axes:
        for ax in row:
            if not ax.get_visible():
                pass  # already hidden
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot hyperparam ablation results")
    parser.add_argument("folder", help="Path to ablation experiment folder")
    parser.add_argument("--save", action="store_true", help="Save figures to PDF")
    
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[1.0, 1.1, 1.5],
        help="Cost fraction thresholds for time-to-fraction plot (default: 1.0 1.1 1.5)",
    )
    args = parser.parse_args()

    folder = args.folder.rstrip("/") + "/"
    config = load_config_from_folder(folder)
    data = load_ablation_data(folder)

    env = config.get("environment", "")
    exp = config.get("experiment_name", "ablation")
    base_title = f"{exp} — {env}"
    max_time = config.get("max_planning_time", 100)

    fig_time  = make_ablation_figure(data, "times", f"{base_title}\nInitial solution time")
    fig_cost  = make_ablation_figure(data, "costs", f"{base_title}\nInitial solution cost")
    fig_curves = make_convergence_curves_figure(data, max_time, f"{base_title}\nConvergence curves")
    fig_ttf   = make_time_to_fraction_figure(
        data, max_time, args.fractions, f"{base_title}\nTime to fraction of best cost"
    )

    if args.save:
        plots_dir = os.path.join(folder, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        for fig, name in [
            (fig_time,   "ablation_init_time.pdf"),
            (fig_cost,   "ablation_init_cost.pdf"),
            (fig_curves, "ablation_convergence_curves.pdf"),
            (fig_ttf,    "ablation_time_to_fraction.pdf"),
        ]:
            if fig:
                fig.savefig(os.path.join(plots_dir, name), bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
