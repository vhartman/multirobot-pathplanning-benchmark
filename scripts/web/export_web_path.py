#!/usr/bin/env python3
"""Export a planned path and the scene geometry for the static web viewer.

Runs a planner on an environment (same options as scripts/run_planner.py),
then writes a self-contained data bundle for web/index.html:

    <out>/
      data.json    manifest: frames, path metadata, per-step annotations
      scene.bin    concatenated mesh buffers (positions f32, indices u32, colors u8)
      poses.bin    per-path per-step world poses of the dynamic frames (f32)

Example (matches the user-facing CLI of run_planner.py):

    python3 scripts/web/export_web_path.py rai.box_stacking_two_robots \
        --distance_metric=max_euclidean --max_time=100 --seed=12 \
        --planner birrt_star --cost_reduction=max

Requires the `robotic` (rai) backend and `numpy`. Meshes are exported at full
resolution by default; passing `--target_tris N` or `--budget_tris N` enables
adaptive, colour-region aware decimation, which additionally requires `scipy`,
`trimesh` and `fast_simplification` (the rai meshes are STL-style and must be
welded with trimesh before quadric decimation). Decimation runs per
uniform-colour region with a budget split by local curvature, so large smooth
cylinders lose most of their triangles while joints and knuckles keep detail,
and no decimated triangle ever crosses a colour boundary.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from math import ceil
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _import_deps():
    """Import the package + optional decimation deps (keeps argparse fast)."""
    import robotic as ry  # noqa: F401

    from multi_robot_multi_goal_planning.problems import get_env_by_name
    from multi_robot_multi_goal_planning.problems.planning_env import State
    from multi_robot_multi_goal_planning.problems.util import (
        interpolate_path,
        path_cost,
    )
    from multi_robot_multi_goal_planning.problems.rai.rai_config import (
        get_robot_joints,
    )
    from multi_robot_multi_goal_planning.planners import (
        AITstar,
        BaseITConfig,
        BaseRRTConfig,
        BidirectionalRRTstar,
        CompositePRM,
        CompositePRMConfig,
        RRTstar,
        EITstar,
    )
    from multi_robot_multi_goal_planning.planners.termination_conditions import (
        IterationTerminationCondition,
        RuntimeTerminationCondition,
    )

    try:
        import trimesh
        from fast_simplification import simplify
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover
        trimesh = None
        simplify = None
        cKDTree = None

    return {
        "ry": ry,
        "get_env_by_name": get_env_by_name,
        "State": State,
        "interpolate_path": interpolate_path,
        "path_cost": path_cost,
        "get_robot_joints": get_robot_joints,
        "AITstar": AITstar,
        "BaseITConfig": BaseITConfig,
        "BaseRRTConfig": BaseRRTConfig,
        "BidirectionalRRTstar": BidirectionalRRTstar,
        "CompositePRM": CompositePRM,
        "CompositePRMConfig": CompositePRMConfig,
        "RRTstar": RRTstar,
        "EITstar": EITstar,
        "IterationTerminationCondition": IterationTerminationCondition,
        "RuntimeTerminationCondition": RuntimeTerminationCondition,
        "simplify": simplify,
        "cKDTree": cKDTree,
        "trimesh": trimesh,
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("env", help="Environment name (e.g. rai.box_stacking_two_robots)")
    p.add_argument("--out", default=None, help="Output folder (default: web/<env>)")
    p.add_argument(
        "--planner",
        choices=["composite_prm", "rrt_star", "birrt_star", "aitstar", "eitstar"],
        default="birrt_star",
        help="Planner to use (default: birrt_star)",
    )
    p.add_argument(
        "--distance_metric",
        choices=["euclidean", "sum_euclidean", "max", "max_euclidean"],
        default="max_euclidean",
    )
    p.add_argument(
        "--per_agent_cost_function",
        choices=["euclidean", "max"],
        default="euclidean",
    )
    p.add_argument(
        "--cost_reduction",
        choices=["sum", "max"],
        default="max",
    )
    p.add_argument("--max_time", type=float, default=None, help="Planning budget (s)")
    p.add_argument("--num_iters", type=int, default=None, help="Iteration budget")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--optimize", action="store_true")
    p.add_argument(
        "--interpolation_resolution",
        type=float,
        default=0.05,
        help="Euclidean interpolation resolution for playback paths (default: 0.05)",
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=5000,
        help="Cap the number of exported playback steps per path (default: 5000, "
        "effectively no downsampling for typical paths)",
    )
    p.add_argument(
        "--target_tris",
        type=int,
        default=0,
        help="Per-mesh cap: decimate any mesh larger than this many triangles "
        "(default: 0 = full resolution, no decimation). Distribution of the "
        "triangles within a mesh is automatic (colour-region + curvature aware).",
    )
    p.add_argument(
        "--budget_tris",
        type=int,
        default=0,
        help="Scene-wide triangle budget (adaptive). Instead of capping every "
        "mesh at --target_tris, this distributes a single total across all "
        "frames proportional to each mesh's curvature/detail, so big smooth "
        "cylinders get few triangles and detailed joints keep more. "
        "Mutually exclusive with --target_tris (when set, --target_tris is ignored).",
    )
    p.add_argument(
        "--no_decimate",
        action="store_true",
        help="Export all meshes at full resolution (large scene.bin; useful for debugging)",
    )
    return p


def plan_paths(dep, env, args):
    """Plan, then interpolate the first and last incumbent solutions.

    The env is constructed in main() with args.seed already applied (the seed
    also drives env construction - some seeds produce infeasible scenes).
    Here we set up the planner and reseed exactly like run_planner.py does
    before calling plan().
    """
    env.cost_reduction = args.cost_reduction
    env.cost_metric = args.per_agent_cost_function

    if args.planner == "composite_prm":
        config = dep["CompositePRMConfig"]()
        config.distance_metric = args.distance_metric
        planner = dep["CompositePRM"](env, config)
    elif args.planner == "rrt_star":
        config = dep["BaseRRTConfig"]()
        config.distance_metric = args.distance_metric
        planner = dep["RRTstar"](env, config=config)
    elif args.planner == "birrt_star":
        config = dep["BaseRRTConfig"]()
        config.distance_metric = args.distance_metric
        planner = dep["BidirectionalRRTstar"](env, config=config)
    elif args.planner == "aitstar":
        config = dep["BaseITConfig"]()
        config.distance_metric = args.distance_metric
        planner = dep["AITstar"](env, config=config)
    elif args.planner == "eitstar":
        config = dep["BaseITConfig"]()
        config.distance_metric = args.distance_metric
        planner = dep["EITstar"](env, config=config)
    else:  # pragma: no cover
        raise ValueError(args.planner)

    if args.num_iters is not None and args.max_time is not None:
        raise ValueError("Cannot specify both num_iters and max_time.")
    if args.num_iters is not None:
        ptc = dep["IterationTerminationCondition"](args.num_iters)
    elif args.max_time is not None:
        ptc = dep["RuntimeTerminationCondition"](args.max_time)
    else:
        ptc = dep["RuntimeTerminationCondition"](30.0)

    t0 = time.time()

    # same seeding order as run_planner.py: reseed after the planner is set up
    np.random.seed(args.seed)
    random.seed(args.seed)

    try:
        path, info = planner.plan(ptc=ptc, optimize=args.optimize)
    except AssertionError as e:
        print(
            f"[export] planning failed: {e}\n"
            "This usually means the scene sampled with this --seed is "
            "infeasible. Try a different seed.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"[export] planning finished in {time.time() - t0:.1f}s")
    assert path is not None, "planner returned no path"

    print("[export] validity of planned path:", env.is_valid_plan(path))

    paths: List[Tuple[str, List]] = []

    # incumbent solutions recorded during planning: first found and final/best
    incumbents = list(info.get("paths") or [])
    if not incumbents:
        incumbents = [path]

    def insert_transition_nodes(p):
        """Double the mode-transition nodes.

        The config after a mode switch in the raw tree path is a transition
        SEED of the new mode's tree, which can be far away from the grasp
        config.  Doubling the node switches the mode at the same config the
        previous task ended at, so attachments happen at the grasp position
        instead of teleporting boxes into the gripper.
        """
        out = []
        for i, st in enumerate(p):
            out.append(st)
            if i + 1 < len(p) and st.mode != p[i + 1].mode:
                out.append(dep["State"](st.q, p[i + 1].mode))
        return out

    for label, incumbent in [
        ("first solution (interpolated)", incumbents[0]),
    ] + (
        [("final solution (interpolated)", incumbents[-1])]
        if len(incumbents) > 1
        else []
    ):
        paths.append(
            (
                label,
                dep["interpolate_path"](
                    insert_transition_nodes(incumbent),
                    args.interpolation_resolution,
                    kind="euclidean",
                ),
            )
        )

    costs = {}
    for label, p in paths:
        costs[label] = float(dep["path_cost"](p, env.batch_config_cost))
        print(f"[export] {label}: {len(p)} states, cost {costs[label]:.3f}")

    return paths, info, costs


# quality/speed trade-off for the underlying quadric simplifier (0 = slow+good)
_SIMPLIFY_AGG = 4.0


def _weld(dep, verts, tris):
    """Repair the STL-style (unwelded) rai mesh into a watertight indexed mesh."""
    return dep["trimesh"].Trimesh(vertices=verts, faces=tris, process=True)


def _mesh_importance(dep, verts, tris):
    """Total curvature-energy of a whole frame (for global budget allocation).

    Large smooth meshes (cylinder bodies, table tops) score low and therefore
    get a small share of a scene-wide triangle budget, while small-but-detailed
    parts (fingers, knuckles) score high.  Degenerate/empty meshes score 0.
    """
    if verts.ndim < 2 or tris.ndim < 2 or tris.shape[0] == 0:
        return 0.0
    try:
        mesh = _weld(dep, verts, tris)
    except Exception:  # pragma: no cover - defensive
        return 0.0
    pts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if pts.shape[0] < 3 or faces.shape[0] == 0:
        return 0.0
    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
    fn = np.cross(
        pts[faces[:, 1]] - pts[faces[:, 0]],
        pts[faces[:, 2]] - pts[faces[:, 0]],
    )
    nrm = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)
    if adj.shape[0]:
        dih = np.arccos(
            np.clip((nrm[adj[:, 0]] * nrm[adj[:, 1]]).sum(axis=1), -1.0, 1.0)
        )
        return float(dih.sum())
    return 0.0


def _color_regions(dep, mesh, colors, orig_verts):
    """Split the welded mesh into connected uniform-colour face regions.

    Faces that share an *edge* and have the same (rounded) colour end up in
    the same region; meshes without per-vertex colours are a single region.
    The regions are only used for *colouring* the decimated output (flat, one
    colour per triangle), not for decimating the geometry.

    Returns (regions, welded_vertex_colors) where ``regions`` is a list of
    face-index arrays and ``welded_vertex_colors`` is the per-vertex colour
    of the welded mesh (None when the mesh has no per-vertex colours).
    """
    pts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_f = faces.shape[0]

    if colors is not None:
        tree = dep["cKDTree"](np.ascontiguousarray(orig_verts))
        _, nn = tree.query(pts, k=1)
        wvc = colors[nn]
        fkey = np.round(wvc[faces].mean(axis=1)[:, :3]).astype(np.int64)
    else:
        wvc = None
        fkey = np.zeros(n_f, dtype=np.int64)

    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
    keep = (fkey[adj[:, 0]] == fkey[adj[:, 1]]).all(axis=1)
    edges = adj[keep]
    if edges.shape[0] == 0:
        return [np.arange(n_f, dtype=np.int64)], wvc

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.int64), (rows, cols)),
        shape=(n_f, n_f),
    ).tocsr()
    ncomp, labels = connected_components(graph, directed=False)
    sels = [
        np.flatnonzero(labels == lab) for lab in range(ncomp) if np.any(labels == lab)
    ]
    return sels, wvc


def _smooth_normals(pts, faces):
    """Per-vertex area-weighted normals, with a stable fallback."""
    fn = np.cross(
        pts[faces[:, 1]] - pts[faces[:, 0]],
        pts[faces[:, 2]] - pts[faces[:, 0]],
    )
    acc = np.zeros_like(pts)
    np.add.at(acc, faces.reshape(-1), np.repeat(fn, 3, axis=0))
    lens = np.linalg.norm(acc, axis=1, keepdims=True)

    bad = (lens.ravel() <= 1e-12).nonzero()[0]
    if bad.size:
        face_of_vertex = np.full(len(pts), -1, dtype=np.int64)
        face_of_vertex[faces.reshape(-1)] = np.repeat(np.arange(len(faces)), 3)
        for v in bad:
            n = fn[face_of_vertex[v]] if face_of_vertex[v] >= 0 else np.zeros(3)
            if np.linalg.norm(n) <= 1e-12:
                n = np.array([0.0, 0.0, 1.0])
            acc[v] = n
        lens = np.linalg.norm(acc, axis=1, keepdims=True)

    return acc / np.maximum(lens, 1e-12)


def decimate(
    dep,
    verts,
    tris,
    colors,
    target_tris,
    no_decimate=False,
):
    """Colour-region aware decimation to <= target_tris triangles.

    The rai meshes are STL-style (every face owns its own 3 vertices), so the
    mesh is first welded/repaired with trimesh and then decimated **as a
    whole** with ``preserve_border`` — every boundary edge of the welded mesh
    is kept exactly, so decimation never opens new holes (verified: boundary
    edge count is identical before/after).

    Colours come from the welded (split) mesh, not re-sampled from the
    original: the welded mesh is split into connected uniform-colour face
    regions, each region gets one representative colour, and every output
    triangle is painted with the colour of the region whose face lies nearest
    to its centroid.  The mesh is unwelded (one vertex per face corner) so
    colours cannot interpolate across colour boundaries -> no banding/bleeding.

    Smooth vertex normals are computed on the indexed mesh before unwelding so
    the geometric shading stays smooth despite the flat colours.

    Returns (verts, tris, colors|None, normals|None).
    """
    if (
        no_decimate
        or tris.shape[0] <= target_tris
        or dep["simplify"] is None
        or dep["trimesh"] is None
        or dep["cKDTree"] is None
    ):
        return verts, tris, colors, None

    mesh = _weld(dep, verts, tris)
    pts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    if mesh.faces.shape[0] <= target_tris and pts.shape[0] == verts.shape[0]:
        # nothing to gain after welding; keep geometry untouched
        return verts, tris, colors, None

    sels, wvc = _color_regions(dep, mesh, colors, verts)
    if not sels:
        return verts, tris, colors, None

    # flat colour per welded face = colour of its region (from the split mesh)
    face_color = None
    if wvc is not None:
        face_color = np.zeros((len(faces), wvc.shape[1]), dtype=np.float64)
        for region in sels:
            face_color[region] = np.round(
                wvc[faces[region].reshape(-1)].mean(axis=0)
            )

    # whole-mesh decimation; preserve_border keeps every boundary edge intact
    p2, f2 = dep["simplify"](
        np.ascontiguousarray(pts),
        np.ascontiguousarray(faces, dtype=np.int32),
        target_count=int(target_tris),
        agg=_SIMPLIFY_AGG,
        preserve_border=True,
    )
    p2 = np.asarray(p2, dtype=np.float64)
    f2 = np.asarray(f2, dtype=np.int64)

    normals = _smooth_normals(p2, f2)

    if face_color is not None:
        # paint each output triangle with the colour of the welded face that
        # lies nearest to its centroid (that face belongs to exactly one
        # colour region, so the assignment can never cross a boundary)
        tree = dep["cKDTree"](np.ascontiguousarray(pts[faces].mean(axis=1)))
        centroids = p2[f2].mean(axis=1)
        _, nn = tree.query(centroids, k=1)
        tri_colors = face_color[nn]

        # unweld: one vertex per face corner so colours can't interpolate
        pts_out = p2[f2].reshape(-1, 3)
        colors_out = np.repeat(tri_colors, 3, axis=0)
        normals_out = normals[f2].reshape(-1, 3)
        faces_out = np.arange(len(pts_out), dtype=np.int64).reshape(-1, 3)
        return pts_out, faces_out, colors_out, normals_out

    return p2, f2, None, normals


def export_scene(dep, env, args):
    """Extract + decimate all frame meshes of the visual config C_orig.

    Decimation is a two-pass, curvature/colour aware operation:

    * Pass 1 collects every eligible frame (raw verts/tris/colors) and, when a
      scene-wide ``--budget_tris`` is given, measures each mesh's curvature
      energy so a single global triangle budget can be split adaptively.
    * Pass 2 calls :func:`decimate` per frame (colour-region aware) either with
      the per-mesh ``--target_tris`` cap or with the allocated share.
    """
    C = env.C_orig
    frames = []
    v_list, t_list, c_list, n_list = [], [], [], []
    robot_prefixes = tuple(env.robots)

    raw = []  # (frame, verts, tris, colors, flat)
    for f in C.getFrames():
        # collision-only frames (e.g. a1_ur_coll0) are skipped entirely
        if "_coll" in f.name:
            continue
        # simplified gripper proxy shapes (translucent in rai) are not exported
        if f.name.endswith(("_palm", "_palm_2", "_finger1", "_finger2")):
            continue

        verts = np.asarray(f.getMeshPoints(), dtype=np.float64)
        tris = np.asarray(f.getMeshTriangles(), dtype=np.int64)
        if verts.ndim < 2 or tris.ndim < 2 or tris.shape[0] == 0:
            continue

        mesh_colors = np.asarray(f.getMeshColors())
        if mesh_colors.ndim == 2 and mesh_colors.shape[0] == verts.shape[0]:
            colors = mesh_colors.astype(np.float64)
            flat = False
        else:
            colors = None
            flat = True

        raw.append((f, verts, tris, colors, flat))

    # ---- adaptive global budget: split --budget_tris across frames ----
    if args.budget_tris and args.budget_tris > 0 and not args.no_decimate:
        importances = [_mesh_importance(dep, v, t) for (_, v, t, _, _) in raw]
        total_imp = sum(importances) or 1.0
        per_frame = []
        for (_, v, t, _, _), imp in zip(raw, importances):
            share = int(round(args.budget_tris * imp / total_imp))
            # never ask to decimate a mesh below what it already has resolved
            share = min(share if share > 0 else 0, int(t.shape[0]))
            per_frame.append(share)
    else:
        per_frame = [args.target_tris] * len(raw)

    for (f, verts, tris, colors, flat), target in zip(raw, per_frame):
        use_no_decimate = args.no_decimate or target <= 0
        verts, tris, colors, normals = decimate(
            dep, verts, tris, colors, target if target > 0 else args.target_tris,
            no_decimate=use_no_decimate,
        )

        if colors is None:
            info = f.info()
            rawc = info.get("color", [0.7, 0.7, 0.7])
            rgba = [
                float(rawc[0]) * 255.0,
                float(rawc[1]) * 255.0,
                float(rawc[2]) * 255.0,
                float(rawc[3]) * 255.0 if len(rawc) > 3 else 255.0,
            ]
            colors = np.tile(np.array(rgba, dtype=np.float64), (verts.shape[0], 1))

        colors = np.clip(colors, 0, 255).astype(np.uint8)

        # robot meshes (palm/fingers etc.) are rendered fully opaque
        if f.name.startswith(robot_prefixes):
            colors[:, 3] = 255

        frames.append(
            {
                "name": f.name,
                "vCount": int(verts.shape[0]),
                "tCount": int(tris.shape[0]),
                "vOffset": 0,  # filled below (absolute offset in scene.bin)
                "tOffset": 0,
                "cOffset": 0,
                "nOffset": -1,  # smooth normals section, -1 if computed client-side
                "flat": bool(flat),
                "color": [int(x) for x in colors[0].tolist()],
            }
        )
        v_list.append(np.ascontiguousarray(verts, dtype=np.float32))
        t_list.append(np.ascontiguousarray(tris, dtype=np.uint32))
        c_list.append(np.ascontiguousarray(colors, dtype=np.uint8))
        n_list.append(
            np.ascontiguousarray(normals, dtype=np.float32)
            if normals is not None
            else None
        )

    # scene.bin layout: [all vertex arrays][all index arrays][all color arrays][all normal arrays]
    v_base = 0
    t_base = v_base + sum(arr.nbytes for arr in v_list)
    c_base = t_base + sum(arr.nbytes for arr in t_list)
    n_base = c_base + sum(arr.nbytes for arr in c_list)

    v_off = t_off = c_off = n_off = 0
    for rec, v, t, c, n in zip(frames, v_list, t_list, c_list, n_list):
        rec["vOffset"] = v_base + v_off
        rec["tOffset"] = t_base + t_off
        rec["cOffset"] = c_base + c_off
        if n is not None:
            rec["nOffset"] = n_base + n_off
            n_off += n.nbytes
        v_off += v.nbytes
        t_off += t.nbytes
        c_off += c.nbytes

    total_bytes = n_base + sum(arr.nbytes for arr in n_list if arr is not None)
    print(
        f"[export] scene: {len(frames)} frames, "
        f"{sum(f['tCount'] for f in frames)} triangles, "
        f"scene.bin {total_bytes / 1e6:.1f} MB"
    )

    return (
        frames,
        np.concatenate(v_list).tobytes(),
        np.concatenate(t_list).tobytes(),
        np.concatenate(c_list).tobytes(),
        b"".join(arr.tobytes() for arr in n_list if arr is not None),
    )


def export_poses(dep, env, frames, paths, max_steps):
    """Record world poses of every geometric frame at every playback step."""
    ry = dep["ry"]
    get_robot_joints = dep["get_robot_joints"]

    frame_names = [f["name"] for f in frames]
    name_to_idx = {n: i for i, n in enumerate(frame_names)}

    C_display_base = ry.Config()
    C_display_base.addConfigurationCopy(env.C_orig)

    joints_cache = {r: get_robot_joints(env.C_orig, r) for r in env.robots}

    mode_cache: Dict = {}

    def mode_config(mode):
        if mode not in mode_cache:
            tmp = ry.Config()
            tmp.addConfigurationCopy(C_display_base)
            env.set_to_mode(mode, config=tmp, use_cached=False, place_in_cache=False)
            mode_cache[mode] = tmp
        return mode_cache[mode]

    C_display = ry.Config()
    C_display.addConfigurationCopy(C_display_base)

    out_paths = []
    pose_arrays = []

    for label, path in paths:
        t0 = time.time()
        stride = max(1, int(ceil(len(path) / max_steps)))
        idxs = list(range(0, len(path), stride))
        # always include the exact mode-transition boundaries: stride
        # downsampling would otherwise skip them and shift the timing of
        # mode switches (attachments etc.) to the nearest sampled step
        for i in range(1, len(path)):
            if path[i - 1].mode != path[i].mode:
                idxs.append(i)
        idxs = sorted(set(idxs))
        if idxs[-1] != len(path) - 1:
            idxs.append(len(path) - 1)

        rows = np.zeros((len(idxs), len(frame_names), 7), dtype=np.float32)
        mode_ids = np.zeros((len(idxs), len(env.robots)), dtype=np.int64)
        transition = np.zeros(len(idxs), dtype=bool)

        for si, pi in enumerate(idxs):
            st = path[pi]
            C_display.clear()
            C_display.addConfigurationCopy(mode_config(st.mode))
            for k, robot in enumerate(env.robots):
                C_display.setJointState(st.q[k], joints_cache[robot])

            for fi, name in enumerate(frame_names):
                fr = C_display.getFrame(name)
                p = fr.getPosition()
                q = fr.getQuaternion()  # ry returns [w, x, y, z]
                rows[si, fi, :3] = p
                rows[si, fi, 3:] = q

            mode_ids[si] = st.mode.task_ids
            if pi > 0 and path[pi - 1].mode != path[pi].mode:
                transition[si] = True

        # static frames: world pose never changes along this path
        dev = np.max(np.abs(rows - rows[0][None, ...]), axis=0).max(axis=1)
        static_mask = dev < 1e-6
        dyn_mask = ~static_mask
        dyn_indices = np.where(dyn_mask)[0].tolist()
        static_poses = {}
        for fi in np.where(static_mask)[0]:
            static_poses[int(fi)] = rows[0, fi].tolist()

        pose_arr = np.ascontiguousarray(rows[:, dyn_indices, :].reshape(-1), dtype=np.float32)

        out_paths.append(
            {
                "label": label,
                "steps": len(idxs),
                "stride": stride,
                "dynFrames": dyn_indices,
                "static": static_poses,
                "poseOffset": None,  # filled below
                "poseCount": int(pose_arr.shape[0]),
                "modeIds": mode_ids.reshape(-1).tolist(),
                "transition": transition.tolist(),
            }
        )
        pose_arrays.append(pose_arr)
        print(
            f"[export] {label}: {len(idxs)} steps (stride {stride}), "
            f"{len(dyn_indices)} dynamic frames"
        )

    poses_bin = np.concatenate(pose_arrays).tobytes()
    byte_offset = 0
    for rec, arr in zip(out_paths, pose_arrays):
        rec["poseOffset"] = byte_offset
        byte_offset += arr.nbytes

    print(f"[export] poses.bin: {byte_offset / 1e6:.1f} MB")
    return out_paths, poses_bin


def compute_camera(env, frames, first_path_poses):
    """Camera framing from the world positions of the frames at step 0.

    Aim at the table area and stand off at a distance proportional to the
    spread of the scene, so the full scene is framed on load.
    """
    pts = np.array(first_path_poses, dtype=np.float64)  # [nframes, 7]
    pos = pts[:, :3]
    center = pos.mean(axis=0)
    radius = float(np.linalg.norm(pos - center, axis=1).max())
    radius = max(radius, 0.5)
    dist = radius * 2.2
    direction = np.array([1.0, -1.8, 1.5])
    direction /= np.linalg.norm(direction)
    target = [float(center[0]), float(center[1]), 0.3]
    return {
        "position": (np.array(target) + direction * dist).tolist(),
        "target": target,
        "up": [0.0, 0.0, 1.0],
    }


def main():
    args = build_parser().parse_args()
    dep = _import_deps()

    if dep["simplify"] is None:
        print(
            "warning: fast_simplification/trimesh/scipy not available; "
            "meshes will be exported at full resolution.",
            file=sys.stderr,
        )

    out = args.out or f"web/{args.env}"
    os.makedirs(out, exist_ok=True)

    # The seed drives env construction too (box shuffling etc.) - apply it
    # before get_env_by_name, exactly like scripts/run_planner.py.
    np.random.seed(args.seed)
    random.seed(args.seed)
    env = dep["get_env_by_name"](args.env)
    paths, info, costs = plan_paths(dep, env, args)

    frames, verts_bin, tris_bin, colors_bin, normals_bin = export_scene(dep, env, args)

    path_records, poses_bin = export_poses(dep, env, frames, paths, args.max_steps)

    # camera from step 0 of the first (raw) path
    first_poses = [
        [0.0] * 7 for _ in frames
    ]
    # grab actual step-0 poses from the first exported path record
    if path_records:
        rec = path_records[0]
        step0 = np.frombuffer(
            poses_bin,
            dtype=np.float32,
            count=len(rec["dynFrames"]) * 7,
            offset=rec["poseOffset"],
        ).reshape(-1, 7)
        for di, fi in enumerate(rec["dynFrames"]):
            first_poses[fi] = step0[di].tolist()
        for fi, pose in rec["static"].items():
            first_poses[int(fi)] = pose
    camera = compute_camera(env, frames, first_poses)

    task_info = []
    for tid, task in enumerate(env.tasks):
        task_info.append(
            {"id": tid, "name": task.name, "type": task.type if task.type is not None else "goto"}
        )

    conv_times = np.asarray(info.get("times", []), dtype=float).ravel().tolist()
    conv_costs = np.asarray(info.get("costs", []), dtype=float).ravel().tolist()

    manifest = {
        "env": args.env,
        "meta": {
            "planner": args.planner,
            "seed": args.seed,
            "max_time": args.max_time,
            "num_iters": args.num_iters,
            "distance_metric": args.distance_metric,
            "per_agent_cost_function": args.per_agent_cost_function,
            "cost_reduction": args.cost_reduction,
            "interpolation_resolution": args.interpolation_resolution,
            "num_robots": len(env.robots),
        },
        "robots": list(env.robots),
        "frames": frames,
        "taskInfo": task_info,
        "convergence": {"times": conv_times, "costs": conv_costs},
        "paths": path_records,
        "pathCosts": costs,
        "camera": camera,
    }

    with open(os.path.join(out, "data.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(out, "scene.bin"), "wb") as f:
        f.write(verts_bin)
        f.write(tris_bin)
        f.write(colors_bin)
        f.write(normals_bin)
    with open(os.path.join(out, "poses.bin"), "wb") as f:
        f.write(poses_bin)

    print(f"[export] wrote {out}/data.json, scene.bin, poses.bin")
    print(f"[export] open web/index.html?data={os.path.basename(out.rstrip('/'))} to view")


if __name__ == "__main__":
    main()
