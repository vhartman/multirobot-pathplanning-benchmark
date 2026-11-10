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
