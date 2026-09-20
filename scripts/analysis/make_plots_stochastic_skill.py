import argparse
import subprocess
import sys
import json
import os
import pathlib

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_plots import interpolate_costs
from compute_confidence_intervals import computeConfidenceInterval

PLANNER_STYLE = {
    "rrt": ("RRT*", "#A01CBB"),
    "rrg": ("RRG", "#1F77B4"),
    "prm": ("PRM*", "#E21616"),
    "deterministic": ("Deterministic", "#1F77B4"),
    "conservative": ("Conservative", "#ff7f0e"),
    "reactive": ("Reactive", "#2ca02c"),
    "prioritized": ("PP", "#FFD61F"),
}


def style(name: str):
    n = name.lower()
    if "reactive" in n:
        kind = "reactive"
    elif "conservative" in n or "rrt_stochastic" in n:
        kind = "conservative"
    elif "prioritized" in n or n.startswith("pp"):
        kind = "prioritized"
    elif "prm" in n:
        kind = "prm"
    elif "rrg" in n:
        kind = "rrg"
    elif "rrt" in n:
        kind = "rrt"
    else:
        kind = "deterministic"
    return (kind,) + PLANNER_STYLE[kind]


FLAT = (6.6, 2.1)


def apply_style(paper: bool = False):
    if paper:
        plt.style.use("./scripts/analysis/paper_2.mplstyle")
        plt.rcParams.update({"savefig.bbox": "tight", "savefig.dpi": 300,
                             "figure.figsize": FLAT,
                             "axes.labelsize": 9, "xtick.labelsize": 8.5,
                             "ytick.labelsize": 8.5, "legend.fontsize": 11,
                             "figure.autolayout": False})
        return
    plt.rcParams.update({
        "figure.figsize": FLAT,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.color": "0.5",
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "legend.fontsize": 12,
        "lines.linewidth": 2.0,
    })


def save_legend(handles, labels, out_path, ncol: int | None = None, handler_map=None):
    if not handles:
        return
    fig = plt.figure(figsize=(FLAT[0], 0.6))
    leg = fig.legend(handles, labels, loc="center", ncol=ncol or len(handles),
                     frameon=True, fancybox=False, borderpad=0.6, fontsize=14,
                     handlelength=1.6, columnspacing=1.8, handler_map=handler_map)
    leg.get_frame().set_edgecolor("0.75")
    leg.get_frame().set_linewidth(0.8)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_planner_legend(experiment, out_path):
    items = _pooled_series(experiment)
    handles = [Line2D([0], [0], color=color, lw=2.5) for _, color, *_ in items]
    labels = [distribution_label(label, data, len(costs), n_total)
             for label, color, costs, n_total, data in items]
    save_legend(handles, labels, out_path)


def save_convergence_legend(experiment, out_path):
    handles, labels = [], []
    for _, _, label, color, data in planners_in_order(experiment):
        if not data.get("curves"):
            continue
        handles.append(Line2D([0], [0], color=color, lw=2.5))
        labels.append(run_label(label, data))
    save_legend(handles, labels, out_path, ncol=2)


def sparse_ticks(ax, nx: int = 4, ny: int | None = 3):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nx))
    if ny is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=ny))


def _read_rows(path: pathlib.Path):
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                rows.append([float(x) for x in line.split(",")])
            except ValueError:
                continue
    return rows


def load_anytime_curves(planner_dir: pathlib.Path, stem: str = ""):
    curves, batches = [], []
    for times, costs in zip(_read_rows(planner_dir / f"{stem}timestamps.txt"),
                            _read_rows(planner_dir / f"{stem}costs.txt")):
        n = min(len(times), len(costs))
        t = np.asarray(times[:n], dtype=float)
        c = np.asarray(costs[:n], dtype=float)
        finite = np.isfinite(t) & np.isfinite(c)
        if not finite.any():
            continue
        t, c = t[finite], c[finite]
        if len(t) > 1 and np.all(t == t[0]):
            batches.append(c)
        else:
            curves.append((t, c))
    return curves, batches


def curve_from_history(path: pathlib.Path):
    if not path.exists():
        return None
    with open(path) as f:
        hist = json.load(f)

    t = float(hist.get("build_time") or 0.0)
    times, costs = [], []
    for h in hist.get("history", []):
        t += h.get("improve_time", 0.0) + h.get("solve_time", 0.0)
        v = float(h.get("V_start", np.inf))
        if np.isfinite(v):
            times.append(t)
            costs.append(v)

    if not times:
        return None
    return np.asarray(times), np.asarray(costs)


def load_experiment(folder: pathlib.Path, exec_run: int | None = None):
    config_path = folder / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    env_name = config.get("environment", folder.name)

    planners = {}
    for planner_dir in sorted(folder.iterdir()):
        if not planner_dir.is_dir() or "plots" in planner_dir.name:
            continue

        per_run = []
        for run_dir in sorted(planner_dir.iterdir()):
            exec_file = run_dir / "executions.json"
            if exec_run is not None and run_dir.name != str(exec_run):
                continue
            if run_dir.is_dir() and exec_file.exists():
                with open(exec_file) as f:
                    run_execs = json.load(f)
                if run_execs:
                    for e in run_execs:
                        e["_run_dir"] = str(run_dir)
                    per_run.append(run_execs)
        executions = [e for run_execs in per_run for e in run_execs]

        curves, batches = load_anytime_curves(planner_dir)
        realized_curves, _ = load_anytime_curves(planner_dir, stem="realized_")

        phase_a_points, infinite_runs = [], 0
        build_times, open_loop_success = [], []
        for hist_path in sorted(planner_dir.glob("*/shortcut_history.json"),
                                key=lambda q: int(q.parent.name) if q.parent.name.isdigit() else 0):
            with open(hist_path) as f:
                history = json.load(f)
            phase_a_points.append(int(history.get("phase_a_points") or 0))
            if history.get("build_time") is not None:
                build_times.append(float(history["build_time"]))
            if history.get("open_loop_success") is not None:
                open_loop_success.append(float(history["open_loop_success"]))
            rounds = history.get("history") or []
            if rounds and not np.isfinite(float(rounds[-1].get("V_start", np.inf))):
                infinite_runs += 1
        n_total = len(executions)

        if not executions and batches:
            costs = np.concatenate(batches)
            executions = [{"cost": float(c), "success": True} for c in costs]
            n_total = max(int(config.get("num_runs", len(costs))), len(costs))
            per_run = [executions]

        nominal = not executions and bool(curves)
        if nominal:
            executions = [{"cost": float(c[-1]), "success": True} for _, c in curves]
            n_total = len(executions)
            per_run = [executions]

        if not curves:
            candidates = sorted(planner_dir.glob("*/shortcut_history.json"))
            if style(planner_dir.name)[0] == "reactive":
                candidates.append(folder / "shortcut_history.json")
            for hist_path in candidates:
                curve = curve_from_history(hist_path)
                if curve is not None:
                    curves.append(curve)

        runs_planned = int(config.get("num_runs", len(per_run)) or len(per_run))
        runs_solved = len(per_run)
        missing = runs_planned - len(per_run)
        if exec_run is None and missing > 0:
            per_execs = int(np.median([len(e) for e in per_run])) if per_run else 0
            per_run = per_run + [[] for _ in range(missing)]
            n_total += per_execs * missing

        if executions or curves:
            planners[planner_dir.name] = {
                "executions": executions, "per_run": per_run, "curves": curves,
                "realized_curves": realized_curves, "phase_a_points": phase_a_points,
                "build_times": build_times, "open_loop_success": open_loop_success,
                "nominal": nominal, "n_total": n_total,
                "runs_planned": runs_planned, "runs_solved": runs_solved,
            }

    return {"env": env_name, "planners": planners}


def planners_in_order(experiment):
    order = list(PLANNER_STYLE)
    items = sorted(((name,) + style(name) + (data,)
                    for name, data in experiment["planners"].items()),
                   key=lambda it: (order.index(it[1]), it[0]))

    kinds = [it[1] for it in items]
    seen = {}
    out = []
    for name, kind, label, color, data in items:
        total = kinds.count(kind)
        index = seen.get(kind, 0)
        seen[kind] = index + 1
        if total > 1:
            label = name
            rgb = mcolors.to_rgb(color)
            f = 0.55 * index / (total - 1)
            color = tuple(c + (1.0 - c) * f for c in rgb)
        out.append((name, kind, label, color, data))
    return out


def _successful(executions):
    costs = [r["cost"] for r in executions
             if r.get("success") and r.get("cost") is not None and np.isfinite(r["cost"])]
    return np.sort(np.asarray(costs, dtype=float))


def realized(data):
    runs = data["executions"]
    return _successful(runs), max(data.get("n_total", 0), len(runs))


def realized_per_run(data):
    per_run = data.get("per_run") or []
    if not per_run:
        costs, n_total = realized(data)
        return [(costs, n_total)] if len(costs) else []
    widths = [len(e) for e in per_run if e]
    default = int(np.median(widths)) if widths else 0
    return [(_successful(execs), len(execs) or default) for execs in per_run]


def median_and_band(stacked, band: str = "ci"):
    stacked = np.asarray(stacked, dtype=float)
    finite_count = np.sum(np.isfinite(stacked), axis=0)
    min_runs = max(1, int(np.ceil(len(stacked) / 2)))
    median = np.nanmedian(stacked, axis=0)
    median[finite_count < min_runs] = np.nan
    if len(stacked) < 2:
        return median, None, None
    lb = np.full(stacked.shape[1], np.nan)
    ub = np.full(stacked.shape[1], np.nan)
    enough = finite_count == len(stacked)
    lb[enough] = np.nanpercentile(stacked[:, enough], 25, axis=0)
    ub[enough] = np.nanpercentile(stacked[:, enough], 75, axis=0)
    return median, lb, ub


def distribution_label(label, data, n_ok, n_total):
    if data["nominal"]:
        return f"{label} (not executed)"
    return f"{label} ({n_ok}/{n_total})"


def run_label(label, data):
    planned = int(data.get("runs_planned") or 0)
    solved = int(data.get("runs_solved") or 0)
    if planned > 1:
        return f"{label} ({solved}/{planned} runs)"
    return label


def plot_cost_convergence(experiment, out_path):
    series = []
    for name, kind, label, color, data in planners_in_order(experiment):
        runs = [(np.asarray(t), np.asarray(c)) for t, c in (data["curves"] or []) if len(t) > 1]
        if runs:
            series.append((label, color, runs, data.get("phase_a_points") or [], data))

    if not series:
        print("No anytime curves found for the convergence plot.")
        return

    t_max = max(t[-1] for _, _, runs, _, _ in series for t, _ in runs)
    grid = np.arange(0.0, t_max, 1e-2)

    fig, ax = plt.subplots(figsize=(FLAT[0] * 2/3, FLAT[1]))
    min_cost, max_cost = np.inf, -np.inf

    for label, color, runs, phase_a_points, data in series:
        trimmed = [(t[n:], c[n:]) for (t, c), n in
                   zip(runs, list(phase_a_points) + [0] * len(runs)) if len(t) - n > 1]
        if not trimmed:
            continue

        stacked = np.array([interpolate_costs(grid, t, c, before_value=np.nan)
                            for t, c in trimmed])
        finite_count = np.sum(np.isfinite(stacked), axis=0)
        min_runs = max(1, int(np.ceil(len(trimmed) / 2)))
        median = np.nanmedian(stacked, axis=0)
        median[finite_count < min_runs] = np.nan
        lb = ub = None

        mask = np.isfinite(median)
        finite_vals = stacked[np.isfinite(stacked)]
        if len(finite_vals) > 0:
            min_cost = min(min_cost, float(np.min(finite_vals)))
            max_cost = max(max_cost, float(np.max(finite_vals)))

        ax.plot(grid[mask], median[mask], color=color, label=run_label(label, data))

        if len(trimmed) > 1:
            lb = np.full(stacked.shape[1], np.nan)
            ub = np.full(stacked.shape[1], np.nan)
            enough = finite_count == len(trimmed)
            lb[enough] = np.nanpercentile(stacked[:, enough], 25, axis=0)
            ub[enough] = np.nanpercentile(stacked[:, enough], 75, axis=0)
            band = mask & enough & np.isfinite(lb) & np.isfinite(ub)
            if band.any():
                ax.fill_between(grid[band], lb[band], ub[band], color=color, alpha=0.2, lw=0)

    ax.set_yscale("log")
    if min_cost < np.inf and max_cost > -np.inf:
        ax.set_ylim(min_cost * 0.9, max_cost * 1.1)

    lo, hi = ax.get_ylim()
    decades = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
    ticks = [t for t in MaxNLocator(nbins=3).tick_values(lo, hi) if lo <= t <= hi]
    if not ticks:
        ticks = [lo, (lo+hi)/2.0, hi]
    ax.set_yticks(ticks)
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Planning time [s]", fontsize=18)
    ax.set_ylabel("Cost", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    sparse_ticks(ax, nx=4, ny=None)
    fig.savefig(out_path)
    plt.close(fig)


def plot_cost_epdf(experiment, out_path, bins: int | None = None):
    series = []
    for name, kind, label, color, data in planners_in_order(experiment):
        runs = realized_per_run(data)
        if runs:
            series.append((label, color, runs, data))

    valid_costs = [c for _, _, runs, _ in series for c, _ in runs if len(c)]
    if not valid_costs:
        print("No execution data found for the EPDF plot.")
        return

    all_costs = np.concatenate(valid_costs)
    lo, hi = float(all_costs.min()), float(all_costs.max())
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    if bins is None:
        bins = 100
    pad = (hi - lo) * 0.03
    edges = np.linspace(lo - pad, hi + pad, bins + 1)
    widths = np.diff(edges)

    fig, ax = plt.subplots(figsize=(FLAT[0] * 2/3, FLAT[1]))
    max_density = 0.0
    for label, color, runs, data in series:
        n_ok = sum(len(c) for c, _ in runs)
        n_all = sum(n for _, n in runs)
        pooled = np.concatenate([c for c, _ in runs if len(c)]) if n_ok else np.array([])
        counts, _ = np.histogram(pooled, bins=edges)
        density = counts / (max(n_all, 1) * widths)
        max_density = max(max_density, float(density.max()))
        ax.stairs(density, edges, fill=True, color=color, alpha=0.25, lw=0)
        ax.stairs(density, edges, color=color, linewidth=2.0,
                  label=distribution_label(label, data, n_ok, n_all))

    ax.set_xlabel("Realized cost", fontsize=18)
    ax.set_ylabel("ePDF", fontsize=18)
    if max_density > 0:
        ax.set_ylim(bottom=0, top=max_density * 1.1)
    ax.set_xlim(edges[0], edges[-1])
    sparse_ticks(ax, nx=4, ny=3)
    ax.tick_params(axis='both', which='major', labelsize=16)
    fig.savefig(out_path)
    plt.close(fig)


def plot_cost_ecdf(experiment, out_path, generate_epdf: bool = True, bins: int | None = None):
    if generate_epdf:
        out_p = pathlib.Path(out_path)
        if "ecdf" in out_p.name:
            epdf_path = out_p.with_name(out_p.name.replace("ecdf", "epdf"))
        else:
            epdf_path = out_p.parent / f"epdf_{out_p.name}"
        plot_cost_epdf(experiment, epdf_path, bins=bins)

    series = []
    for name, kind, label, color, data in planners_in_order(experiment):
        runs = realized_per_run(data)
        if runs:
            series.append((label, color, runs, data))

    if not series:
        print("No execution data found for the ECDF plot.")
        return

    valid_costs = [c for _, _, runs, _ in series for c, _ in runs if len(c)]
    if not valid_costs:
        print("No execution data found for the ECDF plot.")
        return

    all_costs = np.concatenate(valid_costs)
    pad = max((all_costs.max() - all_costs.min()) * 0.03, 1e-9)
    grid = np.linspace(all_costs.min() - pad, all_costs.max() + pad, 1000)

    fig, ax = plt.subplots(figsize=(FLAT[0] * 2/3, FLAT[1]))
    for label, color, runs, data in series:
        stacked = [np.searchsorted(costs, grid, side="right") / max(n_total, 1)
                   for costs, n_total in runs]
        _, lb, ub = median_and_band(stacked)

        n_ok = sum(len(c) for c, _ in runs)
        n_all = sum(n for _, n in runs)
        pooled = np.concatenate([c for c, _ in runs if len(c)]) if n_ok else np.array([])
        curve = np.searchsorted(np.sort(pooled), grid, side="right") / max(n_all, 1)
        ax.step(grid, curve, where="post", color=color, linewidth=2.0,
                label=distribution_label(label, data, n_ok, n_all))
        if lb is not None:
            ax.fill_between(grid, lb, ub, step="post", color=color, alpha=0.22, lw=0)

    ax.set_xlabel("Realized cost", fontsize=18)
    ax.set_ylabel("eCDF", fontsize=18)
    ax.set_ylim(0, 1.03)
    ax.set_xlim(grid[0], grid[-1])
    sparse_ticks(ax, nx=4, ny=None)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis='both', which='major', labelsize=16)
    fig.savefig(out_path)
    plt.close(fig)


def _distribution_items(experiment, what: str):
    items = []
    for name, kind, label, color, data in planners_in_order(experiment):
        costs, n_total = realized(data)
        if len(costs) and n_total:
            items.append((label, color, costs, n_total, data))
    if not items:
        print(f"No execution data found for the {what} plot.")
    return items


def outcome_branches(costs, min_size: int = 3, gap_factor: float = 6.0):
    s = np.sort(np.asarray(costs, dtype=float))
    gaps = np.diff(s)
    if s.size < 2 * min_size or not len(gaps) or not np.any(gaps > 0):
        return [s]

    i = int(np.argmax(gaps))
    if min(i + 1, s.size - i - 1) < min_size:
        return [s]

    material = max(abs(float(s.mean())), 1e-12) * 1e-6
    if gaps[i] <= material:
        return [s]

    others = np.delete(gaps, i)
    others = others[others > 0]
    if not len(others):
        return (outcome_branches(s[:i + 1], min_size, gap_factor)
                + outcome_branches(s[i + 1:], min_size, gap_factor))
    typical = float(np.median(others))
    if typical <= 0 or gaps[i] <= gap_factor * typical:
        return [s]
    return (outcome_branches(s[:i + 1], min_size, gap_factor)
            + outcome_branches(s[i + 1:], min_size, gap_factor))


def _pooled_series(experiment):
    out = []
    for name, kind, label, color, data in planners_in_order(experiment):
        runs = realized_per_run(data)
        if not runs:
            continue
        costs = np.concatenate([c for c, _ in runs if len(c)]) if any(len(c) for c, _ in runs) \
            else np.array([])
        n_total = sum(n for _, n in runs)
        if len(costs) and n_total:
            out.append((label, color, np.sort(costs), n_total, data))
    if not out:
        print("No execution data found.")
    return out


def plot_cost_strip(experiment, out_path):
    items = _distribution_items(experiment, "strip")
    if not items:
        return

    fig, ax = plt.subplots(figsize=(FLAT[0], 1.5))
    rng = np.random.default_rng(0)
    rows = list(reversed(items))
    for y, (label, color, costs, n_total, data) in enumerate(rows):
        ax.scatter(costs, y + rng.normal(0, 0.055, costs.size), s=22, color=color,
                   alpha=0.85, linewidths=0.7, edgecolors="white",
                   label=distribution_label(label, data, len(costs), n_total))
        ax.vlines(costs.mean(), y - 0.28, y + 0.28, color=color, lw=2.2)

    ax.set_yticks([])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Realized cost")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.savefig(out_path)
    plt.close(fig)


def report_branches(experiment):
    for name, kind, label, color, data in planners_in_order(experiment):
        runs = realized_per_run(data)
        if not runs:
            continue
        n_ok = sum(len(c) for c, _ in runs)
        n_all = sum(n for _, n in runs)
        print(f"  {label}: {n_ok}/{n_all} executed over {len(runs)} planning run(s)")
        for i, (costs, n_total) in enumerate(runs):
            if not len(costs):
                print(f"    run {i}: 0/{n_total}, every execution failed")
                continue
            branches = outcome_branches(costs)
            print(f"    run {i}: {len(costs)}/{n_total}, mean {costs.mean():.4f}, "
                  f"{len(branches)} outcome branch(es)")
            for branch in branches:
                sd = float(branch.std(ddof=1)) if branch.size > 1 else 0.0
                print(f"        n={branch.size:>3}  w={branch.size / n_total:.3f}  "
                      f"mean={branch.mean():.4f}  sd={sd:.4f}")


def plot_gantt_chart_comparative(experiment, out_path, legend_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import re
    planners = experiment["planners"]
    env_name = str(experiment.get("env", "")).lower()

    reactive_name = next((p for p in planners if "reactive" in p.lower()), None)
    conservative_name = next((p for p in planners if "conservative" in p.lower()), None)

    if not reactive_name or not conservative_name:
        return

    def runs_by_index(planner_data):
        per_run = planner_data.get("per_run") or [planner_data.get("executions", [])]
        return [{r["index"]: r for r in run_execs
                 if r.get("success") and r.get("timeline")} for run_execs in per_run]

    reac_runs = runs_by_index(planners[reactive_name])
    cons_runs = runs_by_index(planners[conservative_name])

    def _median_cost(records):
        costs = [r["cost"] for r in records if r.get("cost") is not None and np.isfinite(r["cost"])]
        return float(np.median(costs)) if costs else np.inf

    reac_execs = cons_execs = {}
    shared_runs = [i for i in range(min(len(reac_runs), len(cons_runs)))
                   if set(reac_runs[i]) & set(cons_runs[i])]
    if shared_runs:
        ranked = sorted(shared_runs, key=lambda i: _median_cost(reac_runs[i].values()))
        best = ranked[(len(ranked) - 1) // 2]
        print(f"[GANTT] representative planning run {best}: median reactive cost "
              f"{_median_cost(reac_runs[best].values()):.3f} ({len(ranked)} shared run(s))")
        reac_execs, cons_execs = reac_runs[best], cons_runs[best]
    if not reac_execs:
        reac_execs = next((r for r in reac_runs if r), {})
        cons_execs = next((r for r in cons_runs if r), {})
        if reac_execs and cons_execs:
            print("[GANTT] no single planning run has a drawable execution for both planners; "
                  "the two panels are NOT the same skill realization.")

    valid_indices = set(cons_execs.keys()).intersection(set(reac_execs.keys()))
    if not valid_indices:
        return

    def clean_task_name(name):
        name = str(name).lower()
        name = re.sub(r'^(a|r|robot\s*)\d+\s+', '', name)

        if "terminal" in name:
            return "terminal"
        if "home" in name:
            return "home"
        if "pre" in name and "pick" in name:
            return "pre pick"
        if "place" in name:
            return "place"
        if "pick" in name:
            return "pick"
        if "skill" in name:
            return "skill"
        if "visit" in name:
            return "visit"
        if "return" in name:
            return "return"
        if "forward" in name:
            return "forward"
        return re.sub(r'^[a-z]+\d+[_\s]+', '', name).replace("_", " ").strip() or name


    def t_of(entry):
        return entry.get("real_time", entry["time"])

    is_complex_env = any(word in env_name for word in ["stacking", "packing", "picking", "transport"])

    ROW_INCHES, CHROME_INCHES = 0.16, 0.8
    LABEL_INCHES_LARGE = 0.35
    LABEL_INCHES_SMALL = 0.05

    def _sized_grouped(n_robots, is_complex):
        axes_h = ROW_INCHES * n_robots
        if is_complex:
            fig = plt.figure(figsize=(9, CHROME_INCHES + 2 * axes_h + LABEL_INCHES_LARGE * 2))
            gs = fig.add_gridspec(2, 1, hspace=LABEL_INCHES_LARGE / axes_h)
            axes = [fig.add_subplot(gs[i]) for i in range(2)]
            for ax in axes[:-1]:
                ax.tick_params(labelbottom=False)
            return fig, axes
        else:
            fig = plt.figure(figsize=(9, CHROME_INCHES + 4 * axes_h + LABEL_INCHES_LARGE * 2 + LABEL_INCHES_SMALL * 2))
            gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1])
            axes = [fig.add_subplot(gs[i]) for i in range(4)]
            fig.subplots_adjust(hspace=0)
            return None, None # placeholder

    n_robots = len(next(iter(cons_execs.values()))["timeline"][0]["tasks"])
    axes_h = ROW_INCHES * n_robots

    if is_complex_env:
        run_median = _median_cost(reac_execs.values())
        target_idx = min(sorted(valid_indices),
                         key=lambda i: abs(reac_execs[i]["cost"] - run_median))
        fig = plt.figure(figsize=(9, CHROME_INCHES + 2 * axes_h + LABEL_INCHES_LARGE * 2))
        gs = fig.add_gridspec(2, 1, hspace=LABEL_INCHES_LARGE / axes_h)
        axes = [fig.add_subplot(gs[i]) for i in range(2)]
        axes[0].sharex(axes[1])
        plot_configs = [
            (axes[0], cons_execs[target_idx], "Conservative", target_idx),
            (axes[1], reac_execs[target_idx], "Reactive", target_idx),
        ]
    else:
        sorted_indices = sorted(valid_indices, key=lambda idx: reac_execs[idx].get("skill_duration", 0.0))
        short_idx = sorted_indices[0]
        long_idx = sorted_indices[-1]

        fig = plt.figure(figsize=(9, CHROME_INCHES + 4 * axes_h + LABEL_INCHES_LARGE * 2 + LABEL_INCHES_SMALL * 2))
        gs1 = fig.add_gridspec(2, 1, top=0.95, bottom=0.55, hspace=LABEL_INCHES_SMALL / axes_h)
        gs2 = fig.add_gridspec(2, 1, top=0.45, bottom=0.05, hspace=LABEL_INCHES_SMALL / axes_h)
        axes = [fig.add_subplot(gs1[0]), fig.add_subplot(gs1[1]),
                fig.add_subplot(gs2[0]), fig.add_subplot(gs2[1])]

        axes[0].sharex(axes[3])
        axes[1].sharex(axes[3])
        axes[2].sharex(axes[3])

        plot_configs = [
            (axes[0], cons_execs[short_idx], "Conservative", short_idx),
            (axes[1], cons_execs[long_idx], "", long_idx),
            (axes[2], reac_execs[short_idx], "Reactive", short_idx),
            (axes[3], reac_execs[long_idx], "", long_idx),
        ]

    global_max_end = 0
    for _, run, _, _ in plot_configs:
        timeline = run["timeline"]
        if timeline:
            global_max_end = max(global_max_end, t_of(timeline[-1]))


    FIXED_COLORS = {
        "pick": ("dimgray", "red", ""),
        "skill": ("dimgray", "red", ""),
        "wait": ("dimgray", "black", "////"),
        "terminal": ("#f4f4f4", "#999999", ""),
    }
    PALETTE = ["#c5b0e5", "#add8e6", "#ffb6c1", "#ffd8a8", "#90ee90", "#a8e6cf", "#f5cba7"]

    present = []
    for _, run, _, _ in plot_configs:
        for entry in run["timeline"]:
            for raw in (entry.get("task_names") or []):
                label = clean_task_name(raw)
                if label not in FIXED_COLORS and label not in present:
                    present.append(label)
    TASK_COLORS = dict(FIXED_COLORS)
    for i, label in enumerate(present):
        TASK_COLORS[label] = (PALETTE[i % len(PALETTE)], "black", "")
    TASK_COLORS["default"] = ("#bdbdbd", "black", "")

    drawn_labels = set()

    for ax, run, planner_type, seed_idx in plot_configs:
        timeline = run["timeline"]
        if not timeline:
            continue

        num_robots = len(timeline[0]["tasks"])
        max_end_step = 0

        by_task = run.get("skill_seconds_by_task") or {}
        fallback_dur = float(run.get("skill_duration") or 0.0) if not by_task else 0.0

        _epochs = run.get("skill_epochs")
        _epoch = (float(run["skill_duration"]) / float(_epochs)
                  if _epochs and run.get("skill_duration") else 0.05)
        wait_tol = 3.0 * _epoch

        for r in range(num_robots):
            blocks = []

            runs_r = []
            for i in range(len(timeline) - 1):
                seg_start = t_of(timeline[i])
                seg_end = t_of(timeline[i+1])
                max_end_step = max(max_end_step, seg_end)

                names = timeline[i].get("task_names")
                if names and r < len(names):
                    seg_name = names[r]
                else:
                    seg_name = f"T{timeline[i]['tasks'][r]}"

                if runs_r and runs_r[-1][0] == seg_name and abs(runs_r[-1][2] - seg_start) < 1e-9:
                    runs_r[-1][2] = seg_end
                else:
                    runs_r.append([seg_name, seg_start, seg_end])

            for raw_name, start_step, end_step in runs_r:
                clean_name = clean_task_name(raw_name)

                if clean_name in ["pick", "skill"]:
                    true_dur = float(by_task.get(raw_name, fallback_dur))
                    window = end_step - start_step
                    if true_dur > 1e-3 and window > true_dur + wait_tol:
                        blocks.append({"start": start_step, "end": start_step + true_dur,
                                       "name": clean_name})
                        blocks.append({"start": start_step + true_dur, "end": end_step,
                                       "name": "wait"})
                        continue
                    if true_dur <= 1e-3 and window > 1e-3:
                        blocks.append({"start": start_step, "end": end_step, "name": "wait"})
                        continue

                blocks.append({"start": start_step, "end": end_step, "name": clean_name})

            merged = []
            for b in blocks:
                if (merged and merged[-1]["name"] == b["name"]
                        and b["name"] not in ("wait",)
                        and abs(merged[-1]["end"] - b["start"]) < 1e-9):
                    merged[-1]["end"] = b["end"]
                else:
                    merged.append(dict(b))
            blocks = merged

            for b in blocks:
                duration = b["end"] - b["start"]
                if duration <= 0: continue
                c_name = b["name"]
                color, edge, hatch = TASK_COLORS.get(c_name, TASK_COLORS["default"])
                drawn_labels.add(c_name)

                lw = 1.5 if edge == "red" else 0.8
                zorder = 3 if edge == "red" else 2
                alpha = 1.0 if c_name in ["pick", "skill", "wait", "terminal"] else 0.6
                ax.barh(r, duration, left=b["start"], height=0.4, color=color, edgecolor=edge,
                         linewidth=lw, hatch=hatch, zorder=zorder, alpha=alpha)

        for r in range(num_robots):
            ax.vlines(max_end_step, r - 0.25, r + 0.25, color='black', linewidth=2)

        ax.set_ylim(num_robots - 0.5, -0.5)
        ax.set_yticks(range(num_robots))
        ax.set_yticklabels([f"R{r+1}" for r in range(num_robots)], fontsize=16)
        ax.text(0.0, 1.04, planner_type, transform=ax.transAxes, ha='left', va='bottom',
                fontsize=16, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        if ax != axes[-1]:
            ax.tick_params(labelbottom=False)

    skill_label = "Pick / Skill" if "pick" in drawn_labels else "Skill"
    legend_elements = [
        patches.Patch(facecolor='dimgray', edgecolor='red', linewidth=1.5, label=skill_label),
        patches.Patch(facecolor='dimgray', edgecolor='black', hatch='////', label='Wait'),
    ]
    for label in present:
        if label not in drawn_labels:
            continue
        face, edge, hatch = TASK_COLORS[label]
        legend_elements.append(
            patches.Patch(facecolor=face, edgecolor=edge, hatch=hatch,
                          alpha=0.6, label=label.replace("_", " ").title()))
    if "terminal" in drawn_labels:
        legend_elements.append(
            patches.Patch(facecolor='#f4f4f4', edgecolor='#999999', label='Terminal'))
    if legend_path is not None:
        save_legend(legend_elements, [h.get_label() for h in legend_elements], legend_path,
                    ncol=len(legend_elements))

    padding = global_max_end * 0.05
    axes[-1].set_xlim(left=0, right=global_max_end + padding)
    axes[-1].set_xlabel("Elapsed time [s]", fontsize=16)
    axes[-1].tick_params(axis='x', labelsize=16)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_paths", nargs='+', help="Experiment folder(s) (e.g. out/TIMESTAMP_env)")
    parser.add_argument("--out", default=None, help="Output directory for plots")
    parser.add_argument("--pdf", action="store_true", help="Save as PDF")
    parser.add_argument("--no_legend", action="store_true", help="Hide legend")
    parser.add_argument("--paper", action="store_true",
                        help="Use paper_2.mplstyle (requires a LaTeX install)")
    parser.add_argument("--exec_run", type=int, default=None,
                        help="Restrict the ECDF/strip to ONE planning run's executions "
                             "(e.g. 0). Without it every run is pooled, which mixes nature's "
                             "spread with the planner's and multiplies the visible modes.")
    parser.add_argument("--bins", type=int, default=100,
                        help="Number of bins for the ePDF histogram (default: 100)")
    args = parser.parse_args()

    combined_experiment = {"env": None, "planners": {}}

    for path_str in args.folder_paths:
        folder = pathlib.Path(path_str)
        if not folder.exists() or not folder.is_dir():
            print(f"Warning: {folder} is not a valid directory. Skipping.")
            continue

        experiment = load_experiment(folder, args.exec_run)
        if combined_experiment["env"] is None:
            combined_experiment["env"] = experiment["env"]

        if not experiment.get("planners"):
            print(f"No planner data found in {folder}.")
        else:
            combined_experiment["planners"].update(experiment["planners"])

    if not combined_experiment["planners"]:
        print("No planner data found in any of the provided folders.")
        return

    if args.out:
        out_dir = pathlib.Path(args.out)
    else:
        out_dir = pathlib.Path(args.folder_paths[0]) / "plots_stochastic"

    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "pdf" if args.pdf else "png"
    legend = not args.no_legend
    apply_style(args.paper)

    print(f"Generating combined plots for {combined_experiment['env']}...")
    plot_cost_convergence(combined_experiment, out_dir / f"cost.{ext}")
    plot_cost_ecdf(combined_experiment, out_dir / f"ecdf.{ext}", bins=args.bins)
    plot_cost_strip(combined_experiment, out_dir / f"strip.{ext}")
    plot_gantt_chart_comparative(
        combined_experiment, out_dir / f"gantt.{ext}",
        legend_path=(out_dir / f"legend_gantt.{ext}") if legend else None)
    if legend:
        save_planner_legend(combined_experiment, out_dir / f"legend_stochastic.{ext}")
        save_convergence_legend(combined_experiment, out_dir / f"legend_cost.{ext}")
    report_branches(combined_experiment)

    env = combined_experiment["env"] or ""
    if any(tag in env for tag in ("square_island", "rectangle_island", "shared_point")):
        top_down = pathlib.Path(__file__).with_name("make_plots_stochastic_topdown.py")
        cmd = [sys.executable, str(top_down), args.folder_paths[0], "--out", str(out_dir)]
        if args.pdf:
            cmd.append("--pdf")
        if args.no_legend:
            cmd.append("--no_legend")
        if args.paper:
            cmd.append("--paper")
        subprocess.run(cmd, check=False)

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
