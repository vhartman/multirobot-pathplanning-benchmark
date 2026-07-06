"""
Runner for the kinodynamic RRT.

Kept separate from run_planner.py because that script assumes straight-line edges
(shortcutting, linear interpolation, is_valid_plan). Kinodynamic paths are already
dense, dynamically-feasible State sequences, so here we only validate collisions /
mode transitions and (optionally) plot the trajectory.

Examples:
  python3 scripts/run_kino_planner.py kino.point_mass_2d_swap --max_time 20 --plot
  python3 scripts/run_kino_planner.py kino.car_2d_swap --max_time 30 --expansion steer --plot
"""

import argparse
import random

import numpy as np

from multi_robot_multi_goal_planning.problems import get_env_by_name
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    IterationTerminationCondition,
    RuntimeTerminationCondition,
)
from multi_robot_multi_goal_planning.planners.planner_kinodynamic_rrt import (
    KinodynamicRRT,
    KinodynamicRRTConfig,
    kinodynamic_shortcut,
    kinodynamic_robot_shortcut,
)


def validate_kino_path(env, path) -> bool:
    """Collision-free everywhere, contiguous modes, and the last state is `done`."""
    ok = True
    for i, s in enumerate(path):
        if not env.is_collision_free(s.q, s.mode):
            print(f"  collision at index {i}, mode {s.mode.task_ids}")
            ok = False
    for i in range(len(path) - 1):
        a, b = path[i].mode, path[i + 1].mode
        if a != b and b not in a.next_modes:
            print(f"  invalid mode transition at {i}: {a.task_ids} -> {b.task_ids}")
            ok = False
    if not env.done(path[-1].q, path[-1].mode):
        print("  final state does not satisfy done()")
        ok = False
    return ok


def plot_path(env, path, fname):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if env.position_dim_per_robot != 2:
        print("  (skipping plot: position dim != 2)")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    for o in env.obstacles:
        ax.add_artist(o.get_artists())

    colors = plt.cm.tab10.colors
    n = env.start_pos.num_agents()
    has_bodies = hasattr(env, "robot_segments")
    for i in range(n):
        pts = np.array([env.robot_position(s.q, i) for s in path])
        ax.plot(pts[:, 0], pts[:, 1], "-", color=colors[i % 10], label=f"robot {i}")
        ax.plot(pts[0, 0], pts[0, 1], "o", color=colors[i % 10])
        ax.plot(pts[-1, 0], pts[-1, 1], "*", color=colors[i % 10], markersize=14)
        # draw the body (e.g. the trailer link) at a few sampled poses along the path
        if has_bodies:
            for k in np.linspace(0, len(path) - 1, 9).astype(int):
                for p0, p1 in env.robot_segments(path[k].q, i):
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "-", color=colors[i % 10], lw=2, alpha=0.35)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("kinodynamic RRT path")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    print(f"  saved plot to {fname}")


def animate_path(env, path, fname, fps=20, max_frames=400):
    """Animate the projected (2D) kinodynamic motion: each robot as a moving disc with a
    velocity arrow and a growing trail. Works for any env whose positions project to 2D
    (the abstract point-mass / car envs, and the 2-DOF rai wrappers). Saved as a GIF so it
    is viewable without an interactive display."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if getattr(env, "position_dim_per_robot", None) != 2:
        print("  (skipping animation: position dim != 2)")
        return

    n = env.start_pos.num_agents()
    radii = getattr(env, "agent_radii", [0.1] * n)
    colors = plt.cm.tab10.colors
    has_bodies = hasattr(env, "robot_segments")

    # project once; subsample to keep the gif light
    P = np.array([[env.robot_position(s.q, i) for i in range(n)] for s in path])  # (T, n, 2)
    T = len(path)
    step = max(1, T // max_frames)
    frames = list(range(0, T, step))
    if frames[-1] != T - 1:
        frames.append(T - 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_title("kinodynamic trajectory")

    def draw(fi):
        ax.clear()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.set_title("kinodynamic trajectory")
        for o in env.obstacles:
            ax.add_artist(o.get_artists())
        for i in range(n):
            c = colors[i % 10]
            ax.plot(P[: fi + 1, i, 0], P[: fi + 1, i, 1], "-", color=c, lw=1, alpha=0.6)
            # body (e.g. the trailer link) at the current pose
            if has_bodies:
                for p0, p1 in env.robot_segments(path[fi].q, i):
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "-", color=c, lw=4, solid_capstyle="round")
                    ax.add_artist(plt.Circle(p1, radii[i] * 0.9, color=c))
            ax.add_artist(plt.Circle(P[fi, i], radii[i], color=c))
            # velocity arrow from finite difference of the projected position
            if fi + 1 < T:
                v = P[min(fi + step, T - 1), i] - P[fi, i]
                nrm = np.linalg.norm(v)
                if nrm > 1e-6:
                    d = v / nrm * (radii[i] + 0.15)
                    ax.arrow(P[fi, i, 0], P[fi, i, 1], d[0], d[1],
                             head_width=0.06, color=c, length_includes_head=True)
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False)
    anim.save(fname, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  saved animation to {fname} ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Kinodynamic planner runner")
    parser.add_argument("env", help="environment name, e.g. kino.point_mass_2d_swap")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_time", type=float, default=None)
    parser.add_argument("--num_iters", type=int, default=None)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--planner", choices=["rrt", "sst"], default="rrt",
                        help="rrt/rrt* (steering) or SST (propagation-only, for non-steerable systems)")
    parser.add_argument("--no_loop_shortcut", action="store_true",
                        help="disable in-RRT*-loop shortcutting + re-injection (on by default)")
    parser.add_argument("--goal_bias", type=float, default=0.15)
    parser.add_argument("--max_extension_steps", type=int, default=10)
    parser.add_argument("--expansion", choices=["steer", "propagate"], default="steer")
    parser.add_argument("--cost_metric", choices=["time", "length"], default="time",
                        help="what the planner minimises: trajectory time or spatial path length")
    parser.add_argument("--shortcut", action="store_true", help="post-process with exact-steering shortcutting")
    parser.add_argument("--shortcut_iters", type=int, default=2000)
    parser.add_argument("--shortcut_rounds", type=int, default=2, help="alternating composite/per-robot passes")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_file", default="kino_path.png")
    parser.add_argument("--animate", action="store_true", help="save a GIF animation of the projected 2D motion")
    parser.add_argument("--animate_file", default="kino_path.gif")
    parser.add_argument("--display", action="store_true", help="animate in the env viewer (matplotlib / rai)")
    parser.add_argument("--viser", action="store_true", help="interactive viser web viewer (compares planned vs shortcut)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    env = get_env_by_name(args.env)

    if args.num_iters is not None:
        ptc = IterationTerminationCondition(args.num_iters)
    else:
        ptc = RuntimeTerminationCondition(args.max_time if args.max_time else 20.0)

    if args.planner == "sst":
        from multi_robot_multi_goal_planning.planners.planner_kinodynamic_sst import (
            KinodynamicSST,
            KinodynamicSSTConfig,
        )
        planner = KinodynamicSST(env, KinodynamicSSTConfig(goal_bias=args.goal_bias, cost_metric=args.cost_metric))
    else:
        config = KinodynamicRRTConfig(
            goal_bias=args.goal_bias,
            max_extension_steps=args.max_extension_steps,
            expansion=args.expansion,
            shortcut_in_loop=not args.no_loop_shortcut,
            cost_metric=args.cost_metric,
        )
        planner = KinodynamicRRT(env, config)

    path, info = planner.plan(ptc=ptc, optimize=args.optimize)

    if path is None:
        print("No path found.")
        return

    print(f"path length: {len(path)} states")
    print("costs:", info["costs"])
    print("times:", info["times"])
    print("validating path...")
    print("valid:", validate_kino_path(env, path))

    if args.plot:
        plot_path(env, path, args.plot_file)

    planned_path = path
    display_paths = [("planned", planned_path)]

    if args.shortcut:
        sc_path = path
        if getattr(env, "has_exact_steering", False):
            print("shortcutting (alternating full-composite min-time and per-robot passes)...")
            for _ in range(args.shortcut_rounds):
                sc_path, _ = kinodynamic_shortcut(env, sc_path, max_iter=args.shortcut_iters)
                sc_path, _ = kinodynamic_robot_shortcut(env, sc_path, max_iter=args.shortcut_iters)
        else:
            # no exact steering (e.g. the car): use the iLQR two-point-BVP shortcut instead
            print("shortcutting (iLQR, non-exact steering: composite + per-robot passes)...")
            from multi_robot_multi_goal_planning.planners.kino_ilqr import (
                kinodynamic_ilqr_shortcut,
                kinodynamic_ilqr_robot_shortcut,
            )
            for _ in range(args.shortcut_rounds):
                sc_path, _ = kinodynamic_ilqr_shortcut(env, sc_path, max_iter=args.shortcut_iters)
                sc_path, _ = kinodynamic_ilqr_robot_shortcut(env, sc_path, max_iter=args.shortcut_iters)
        print("shortcut valid:", validate_kino_path(env, sc_path))
        if args.plot:
            plot_path(env, sc_path, args.plot_file.replace(".png", "_shortcut.png"))
        path = sc_path
        display_paths.append(("shortcut", sc_path))

    if args.animate:
        # animate the planned path, and (if shortcutting) the shortcut path too
        animate_path(env, planned_path, args.animate_file.replace(".gif", "_planned.gif"))
        if args.shortcut:
            animate_path(env, path, args.animate_file.replace(".gif", "_shortcut.gif"))

    if args.viser:
        if hasattr(env, "display_path_viser"):
            # interactive web viewer: compare planned vs shortcut (real geometry for rai envs)
            env.display_path_viser(
                [p for _, p in display_paths],
                path_labels=[name for name, _ in display_paths],
                primitives_only=True,
            )
        else:
            print("(--viser not supported by this env; use --display for the matplotlib viewer)")

    if args.display:
        env.show(blocking=True)
        env.display_path(path, stop=False, stop_at_end=True, adapt_to_max_distance=True)


if __name__ == "__main__":
    main()
