import argparse
import glob
import json
import os
import pathlib
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_plots_stochastic_skill import PLANNER_STYLE, FLAT, apply_style, save_legend, style

GEOMETRY = {
    "rectangle_island": (1.2, 0.9, 0.3, 2.0),
    "square_island": (1.2, 1.2, 0.0, 2.0),
    "shared_point": (None, None, None, 2.0),
}
DEFAULT_GEOMETRY = (None, None, None, 2.0)


def failed_execution_paths(run_dir: pathlib.Path):
    exec_file = os.path.join(run_dir, "executions.json")
    if not os.path.exists(exec_file):
        return set()
    try:
        with open(exec_file) as fin:
            records = json.load(fin)
    except (ValueError, OSError):
        return set()
    return {i + 1 for i, r in enumerate(records)
            if not (r.get("reached_goal") and r.get("valid"))}


def load_paths(run_dir: pathlib.Path, include_failed: bool = False):
    skip = set() if include_failed else failed_execution_paths(run_dir)
    paths, dropped = [], 0
    def get_num(f):
        match = re.search(r"path_(\d+)\.json$", f)
        return int(match.group(1)) if match else -1

    for f in sorted(glob.glob(os.path.join(run_dir, "path_*.json")), key=get_num):
        idx = get_num(f)
        if idx in skip:
            dropped += 1
            continue
        with open(f) as fin:
            data = json.load(fin)
        if not data:
            continue
        qs = np.array([state["q"] for state in data])
        flags = [bool(state.get("is_skill_waypoint", False)) for state in data]
        paths.append((qs, flags))
    if dropped:
        print(f"[TOP-DOWN] {run_dir}: {dropped} failed execution(s) not drawn "
              f"(--include_failed to show them)")
    return paths


def side_of(qs, skill_flags):
    skill_qs = qs[np.asarray(skill_flags)] if np.any(skill_flags) else qs
    a1_y = skill_qs[:, 1]
    return "UP" if a1_y[np.argmax(np.abs(a1_y))] > 0 else "DOWN"


def skill_steps(qs, skill_flags):
    return int(np.count_nonzero(skill_flags))


def classify(paths):
    lengths = [skill_steps(*p) for p in paths]
    sides = [side_of(*p) for p in paths]
    lo, hi = min(lengths), max(lengths)
    if lo > 0 and hi >= 2 * lo:
        cut = 0.5 * (lo + hi)
        return (["SHORT SKILL", "LONG SKILL"],
                ["SHORT SKILL" if n <= cut else "LONG SKILL" for n in lengths])
    if len(set(sides)) > 1:
        return ["UP", "DOWN"], sides
    return [sides[0]], sides


def segments(skill_flags, qs=None):
    raw, current = [], [0]
    for i in range(1, len(skill_flags)):
        current.append(i)
        if skill_flags[i] != skill_flags[i - 1]:
            raw.append((skill_flags[i - 1], current))
            current = [i]
    if current:
        raw.append((skill_flags[-1], current))

    if qs is None:
        return raw

    out = []
    for is_skill, seg in raw:
        if not is_skill and len(seg) <= 2 and qs is not None:
            disp = np.max(np.abs(qs[seg[-1]] - qs[seg[0]]))
            if disp < 0.08:
                if out and out[-1][0]:
                    out[-1] = (True, out[-1][1] + seg[1:])
                    continue
        out.append((is_skill, seg))
    return out


def adjust_lightness(color, amount=1.0):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def draw_pointy_arrow(ax, px, py, dx, dy, color, size=0.18, alpha=1.0):
    angle = np.arctan2(dy, dx)
    L = size
    W = size * 0.35

    pts = np.array([
        [L*0.5, 0],
        [-L*0.5, W],
        [-L*0.5, -W]
    ])

    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    pts = pts @ rot.T + np.array([px, py])

    poly = plt.Polygon(pts, facecolor=color, edgecolor='none', lw=0.5, alpha=alpha, zorder=6)
    ax.add_patch(poly)

def _place_arrow_on_leg(ax, leg_x, leg_y, color, alpha, fraction=0.5):
    if len(leg_x) < 2:
        return

    dxs = np.diff(leg_x)
    dys = np.diff(leg_y)
    step_dists = np.hypot(dxs, dys)
    cum_dists = np.concatenate(([0], np.cumsum(step_dists)))
    total_dist = cum_dists[-1]

    if total_dist < 0.2:
        return

    target_dist = total_dist * fraction
    mid_idx = np.searchsorted(cum_dists, target_dist)
    mid_idx = max(1, min(len(leg_x) - 2, mid_idx))

    idx1 = mid_idx
    while idx1 > 0 and cum_dists[mid_idx] - cum_dists[idx1] < 0.05:
        idx1 -= 1

    idx2 = mid_idx
    while idx2 < len(leg_x) - 1 and cum_dists[idx2] - cum_dists[mid_idx] < 0.05:
        idx2 += 1

    dx = leg_x[idx2] - leg_x[idx1]
    dy = leg_y[idx2] - leg_y[idx1]

    if np.hypot(dx, dy) < 1e-4:
        return

    draw_pointy_arrow(ax, leg_x[mid_idx], leg_y[mid_idx], dx, dy, color, size=0.42, alpha=alpha)


def _add_all_arrows(ax, qs, dark_color, light_color, active_skill_color='#606060'):
    def place(x, y, color, alpha, fraction=0.5):
        _place_arrow_on_leg(ax, x, y, color, alpha, fraction=fraction)

    x_in, y_in = qs[:, 2], qs[:, 3]
    f_in = np.argmax(np.hypot(x_in - x_in[0], y_in - y_in[0]))
    if f_in > 5 and np.hypot(x_in[f_in] - x_in[0], y_in[f_in] - y_in[0]) > 0.5:
        place(x_in[:f_in+1], y_in[:f_in+1], dark_color, 1.0, fraction=0.5)
        place(x_in[f_in:], y_in[f_in:], dark_color, 1.0, fraction=0.5)
    else:
        place(x_in, y_in, dark_color, 1.0, fraction=0.5)

    x_act, y_act = qs[:, 0], qs[:, 1]
    f_act = np.argmax(np.hypot(x_act - x_act[0], y_act - y_act[0]))
    if f_act > 5 and np.hypot(x_act[f_act] - x_act[0], y_act[f_act] - y_act[0]) > 0.5:
        place(x_act[:f_act+1], y_act[:f_act+1], active_skill_color, 0.8, fraction=0.5)
        place(x_act[f_act:], y_act[f_act:], light_color, 1.0, fraction=0.5)
    else:
        place(x_act, y_act, active_skill_color, 0.8, fraction=0.5)

_START_SIZE = 120
_GOAL_SIZE = 120


def _shades(planner_color):
    if planner_color.lower() in ("#ff7f0e", "tab:orange"):
        return "#FFA500", "#E65C00"
    return adjust_lightness(planner_color, 1.4), adjust_lightness(planner_color, 0.7)


def _half_split_marker(ax, x, y, marker, size, left_color, right_color):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    for side, color in (("left", left_color), ("right", right_color)):
        sc = ax.scatter([x], [y], color=color, marker=marker, s=size,
                        edgecolors="black", lw=0.5, zorder=5)
        left_edge = x0 if side == "left" else x
        width = (x - x0) if side == "left" else (x1 - x)
        sc.set_clip_path(plt.Rectangle((left_edge, y0), width, y1 - y0, transform=ax.transData))


def draw_panel(ax, paths, geometry, planner_color, max_paths=5, thin=False):
    half_x, half_y, centre_y, arena = geometry
    ax.set_xlim(-arena, arena)
    ax.set_ylim(-arena, arena)
    ax.set_aspect("equal")
    if half_x is not None:
        ax.add_patch(plt.Rectangle((-half_x, -half_y + centre_y), 2 * half_x, 2 * half_y,
                                   color="gray", alpha=0.3, lw=0, zorder=0))
    ax.add_patch(plt.Rectangle((-arena, -arena), 2 * arena, 2 * arena, fill=False,
                               color="gray", lw=1.2, zorder=0))
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if not paths:
        ax.text(0, 0, "no executions\nin this branch", ha="center", va="center",
                fontsize=8, color="0.5")
        return


    import matplotlib.patheffects as pe

    light_color, dark_color = _shades(planner_color)

    lw_solid = 1.0 if thin else 3.5
    lw_dash = 1.2 if thin else 4.0
    lw_stroke = 2.5 if thin else 6.0

    qs, flags = paths[0]
    for is_skill, seg in segments(flags, qs):
        ls_active = "-" if is_skill else (0, (2, 1.5))
        ls_inactive = "-" if is_skill else (0, (1, 1.2))
        alpha = 0.9 if is_skill else 0.8
        if is_skill:
            ax.plot(qs[seg, 2], qs[seg, 3], color=dark_color, lw=lw_solid, ls=ls_active, alpha=alpha, zorder=3)
        else:
            ax.plot(qs[seg, 2], qs[seg, 3], color=dark_color, lw=lw_dash, ls=ls_inactive, alpha=alpha, zorder=4,
                    path_effects=[pe.withStroke(linewidth=lw_stroke, foreground='white')])
            
            ax.plot(qs[seg, 0], qs[seg, 1], color=light_color, lw=lw_dash, ls=ls_active, alpha=alpha, zorder=2)

    def get_marker(x_coord):
        return "o" if x_coord < 0 else "D"

    m_r2_start = get_marker(qs[0, 2])
    m_r2_goal = get_marker(qs[-1, 2])
    m_r1_start = get_marker(qs[0, 0])
    m_r1_goal = get_marker(qs[-1, 0])

    def get_size(m):
        return 180 if m == "D" else 250

    ax.scatter(qs[0, 2], qs[0, 3], color=dark_color, marker=m_r2_start, s=get_size(m_r2_start),
               edgecolors="black", lw=0.5, zorder=6)
    ax.scatter(qs[-1, 2], qs[-1, 3], color=dark_color, marker=m_r2_goal, s=get_size(m_r2_goal),
               edgecolors="black", lw=0.5, zorder=6)
    _half_split_marker(ax, qs[0, 0], qs[0, 1], m_r1_start, get_size(m_r1_start), '#606060', light_color)
    _half_split_marker(ax, qs[-1, 0], qs[-1, 1], m_r1_goal, get_size(m_r1_goal), '#606060', light_color)

    for index, (qs_bundle, flags) in enumerate(paths[:max_paths]):
        for is_skill, seg in segments(flags, qs_bundle):
            if is_skill:
                ax.plot(qs_bundle[seg, 0], qs_bundle[seg, 1], color='#606060', lw=2.5, ls='-', alpha=0.4, zorder=1)

    _add_all_arrows(ax, paths[0][0], dark_color, light_color)

class _TwoLineIcon:
    def __init__(self, top_color, bottom_color):
        self.top_color = top_color
        self.bottom_color = bottom_color


class _TwoLineIconHandler(HandlerBase):

    def create_artists(self, legend, handle, x0, y0, width, height, fontsize, trans):
        y_top = y0 + height * 0.72
        y_bot = y0 + height * 0.24
        top = Line2D([x0, x0 + width], [y_top, y_top], color=handle.top_color, lw=1.8, ls="-")
        bot = Line2D([x0, x0 + width], [y_bot, y_bot], color=handle.bottom_color, lw=1.8,
                    ls=(0, (2, 1.3)))
        top.set_transform(trans)
        bot.set_transform(trans)
        return [top, bot]


def _legend_handles(kinds_present):
    handles, labels = [], []
    for kind in [k for k in PLANNER_STYLE if k in kinds_present]:
        _, label, color = style(kind)
        light, dark = _shades(color)
        handles.append(_TwoLineIcon("#606060", light))
        labels.append(f"{label} R1")
        handles.append(_TwoLineIcon(dark, dark))
        labels.append(f"{label} R2")
    return handles, labels, {_TwoLineIcon: _TwoLineIconHandler()}


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("folder", help="one experiment folder")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--run", type=int, default=0, help="which planning run's paths to draw")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--no_legend", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--include_failed", action="store_true",
                        help="also draw executions that did not reach the goal validly; "
                             "they stop mid-air and read as a geometry defect")
    parser.add_argument("--thin", action="store_true", help="use thinner line widths for top-down trajectories")
    args = parser.parse_args()

    folder = pathlib.Path(args.folder)
    config = {}
    if (folder / "config.json").exists():
        with open(folder / "config.json") as f:
            config = json.load(f)
    env_name = config.get("environment", folder.name)
    geometry = next((g for tag, g in GEOMETRY.items() if tag in env_name), DEFAULT_GEOMETRY)

    planners = []
    for planner_dir in sorted(folder.iterdir()):
        if not planner_dir.is_dir() or "plots" in planner_dir.name:
            continue
        paths = load_paths(planner_dir / str(args.run), include_failed=args.include_failed)
        if paths:
            kind = style(planner_dir.name)[0]
            planners.append((kind, paths))

    if not planners:
        print(f"No path_*.json found in {folder}/*/{args.run}. Nothing to draw.")
        return

    apply_style(args.paper)
    out_dir = pathlib.Path(args.out) if args.out else folder / "plots_stochastic"
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "pdf" if args.pdf else "png"

    for kind, paths in planners:
        branches, labels = classify(paths)
        for branch in branches:
            picked = [p for p, lab in zip(paths, labels) if lab == branch]
            fig, ax = plt.subplots(1, 1, figsize=(2.45, 2.45))
            _, _, p_color = style(kind)
            draw_panel(ax, picked, geometry, p_color, thin=args.thin)

            suffix = branch.lower().replace(" ", "_")
            kind_short = {"conservative": "cons", "reactive": "react"}.get(kind, kind)
            out_path = out_dir / f"topdown_{kind_short}_{suffix}.{ext}"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_path}")


    if not args.no_legend:
        kinds_present = {kind for kind, _ in planners}
        legend_handles, legend_labels, handler_map = _legend_handles(kinds_present)
        ncol = len(legend_handles)
        fig = plt.figure(figsize=(FLAT[0], 0.6))
        leg = fig.legend(legend_handles, legend_labels, loc="center", ncol=ncol,
                         frameon=True, fancybox=False, borderpad=0.6, fontsize=14,
                         handlelength=1.6, columnspacing=1.8,
                         handler_map=handler_map)
        leg.get_frame().set_edgecolor("0.75")
        leg.get_frame().set_linewidth(0.8)
        legend_path = out_dir / f"legend_topdown.{ext}"
        fig.savefig(legend_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {legend_path}")


if __name__ == "__main__":
    main()
