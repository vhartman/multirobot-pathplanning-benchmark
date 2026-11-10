# Motion costs

Set the objective before constructing a planner:

```python
from multi_robot_multi_goal_planning.problems.core.cost import JointSpaceLength

env.cost = JointSpaceLength(metric="euclidean", reduction="max", sum_weight=0.01)
```

`sum` adds robot distances. `max` takes their maximum plus `sum_weight` times
their sum. Path cost adds these values over motion segments. Steering and
nearest-neighbor distances remain separate planner settings.

`MotionCost` separates evaluated `edge_cost` / `batch_edge_cost` from admissible
`lower_bound` / `batch_lower_bound`. An edge means the straight joint-space
connection. `motion_cost(points)` and `path_cost(path)` retain all supplied
waypoints and sum segment costs; they do not replace a motion by its endpoint
chord. State inputs retain mode information for objectives that need it.

`JointSpaceLength` uses the existing batched kernels. Its endpoint distances are
both exact straight-edge costs and lower bounds. Its `joint_norm_geometry`
describes the geometry supported by informed samplers. Costs without this
geometry use uniform feasible proposals and lower-bound rejection instead.

The environment's `cost_metric`, `cost_reduction`, `config_cost`, and
`batch_config_cost` remain compatibility adapters for existing experiment
scripts. New code should use `env.cost`. Keep the objective fixed throughout a
planning run: tree costs, heuristic caches, and informed budgets depend on it.

## Workspace position travel

```python
from multi_robot_multi_goal_planning.problems.rai.workspace_cost import workspace_cost

# Explicit frame names, one per robot in env.robots order.
env.cost = workspace_cost(env, frame_names, reduction="max", resolution=0.05)
```

The FK adapter uses a private configuration and rejects frames affected by task
attachment changes. It never changes the live collision scene. Frame positions
are cached in a bounded cache. A non-RAI backend can instead construct
`WorkspaceLength(positions=callback, batch_positions=optional_batch_callback)`.

Each joint-space edge is subdivided so no coordinate increment exceeds
`resolution`. Frame displacement is computed between successive samples,
aggregated over robots at each step, then summed along the motion. This is an
arc-length approximation; halve the resolution to check sensitivity. Subdividing
an existing path differently can change quadrature error. This setting does not
change collision checking. Position-only cost deliberately ignores orientation
and motion in the selected frames' positional nullspace.

Endpoint frame distance supplies a lower bound, never the evaluated edge cost.
Joint-space ellipsoids and convex informed walks are disabled for this objective;
ordinary proposals use lower-bound filtering. Steering distances are unchanged.
The prioritized planner retains joint-space timing assumptions and explicitly
rejects workspace objectives.

The planner CLI accepts `--cost_space workspace --workspace_frames FRAME1 FRAME2
--cost_resolution 0.05`. The experiment runner accepts an optional `motion_cost`
object, which overrides the legacy joint-cost fields:

```json
{"motion_cost": {"type": "workspace", "frames": ["FRAME1", "FRAME2"],
                 "reduction": "max", "resolution": 0.05}}
```
