# Web path viewer

Static, self-contained 3D viewer for planned paths produced by the benchmark
planners. One data bundle per scene lives in its own folder (e.g.
`box_stacking_two_robots/`), and `index.html` renders whichever bundle is given
via the `?data=` query parameter (default: `box_stacking_two_robots`).

## Viewing locally

```bash
cd mr_bench
python3 -m http.server 8000
# open http://localhost:8000/web/index.html?data=box_stacking_two_robots
```

A plain `file://` open will not work because the browser blocks `fetch` on the
data files — serve over HTTP (as above) or use GitHub Pages.

## Regenerating the data

```bash
python3 scripts/web/export_web_path.py rai.box_stacking_two_robots \
    --distance_metric=max_euclidean --max_time=100 --seed=12 \
    --planner birrt_star --cost_reduction=max \
    --out web/box_stacking_two_robots
```

The exporter takes the same arguments as `scripts/run_planner.py`
(`--planner`, `--max_time`, `--seed`, `--distance_metric`, `--cost_reduction`,
`--optimize`, ...) and writes three files into `--out`:

- `data.json` — frame/mesh manifest, path metadata, per-step annotations,
  convergence data, camera framing
- `scene.bin` — decimated mesh buffers (positions f32, indices u32, colors u8)
- `poses.bin` — per-step world poses of the moving frames (f32)

Notes:

- Meshes are exported at **full resolution by default**. Decimation is opt-in:
  - `--target_tris N` caps any mesh larger than `N` triangles.
  - `--budget_tris N` (mutually exclusive) sets a **single scene-wide** budget
    and distributes it across all frames proportional to each mesh's
    curvature/detail.
  Because the rai meshes are STL-style, decimation first welds them with
  `trimesh` and then quadric-simplifies the mesh **as a whole** with
  `preserve_border`, so no new holes are introduced (boundary edges are kept
  exactly). Colours are assigned from the welded mesh's uniform-colour face
  regions: each output triangle gets the flat colour of the region whose face
  lies nearest to its centroid, so colours never bleed across material
  boundaries. Requires `trimesh`, `fast_simplification` and `scipy` when
  decimating.
- The viewer shows two playback paths: the planner's **first** incumbent
  solution and its **final (best)** solution, both interpolated. Mode
  transition nodes are doubled (as with `run_planner.py
  --insert_transition_nodes`) so attachments happen at the grasp config
  instead of teleporting boxes into the gripper.
- Collision shapes (`*_coll*`) and the translucent gripper proxy shapes
  (`*_palm`, `*_finger1`, `*_finger2`) are skipped; the remaining robot meshes
  are exported fully opaque.
- The seed drives the environment construction as well (box colors, target
  shuffle, ...); some seeds produce infeasible scenes. The exporter reports a
  clear error in that case — try another seed.
- As in `run_planner.py`, without `--optimize` the RRT planners stop at the
  first solution found. Pass `--optimize` to use the full time budget.

## Multiple scenes

The viewer reads `web/scenes.json` to populate the **scene** dropdown in the
header; each entry maps a data folder to a display label:

```json
[
  { "folder": "box_stacking_two_robots", "label": "Box stacking, two robots (rai)" }
]
```

Export additional environments with `--out web/<folder>` and add an entry for
each. The `?data=<folder>` URL parameter still selects the initial scene.

## GitHub Pages

1. Commit and push the `web/` folder.
2. In the repository settings enable Pages (deploy from branch).
3. The viewer is then available at
   `https://<user>.github.io/<repo>/web/index.html`
   (it defaults to the `box_stacking_two_robots` bundle).
