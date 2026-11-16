"""Collect mode-local path constraints independently of sampling and projection."""

import numpy as np


class ConstraintSet:
    """Constraint references grouped by type, deduplicated by object identity."""

    def __init__(self, env, mode, constraints):
        self.env, self.mode = env, mode
        unique = {id(c): c for c in constraints}
        self.constraints = tuple(unique.values())
        groups = {kind: [] for kind in (
            "affine_equality", "affine_inequality",
            "nonlinear_equality", "nonlinear_inequality",
        )}
        for constraint in self.constraints:
            kind = getattr(constraint, "type", None)
            if kind not in groups:
                raise ValueError(f"Unsupported constraint kind: {kind!r}")
            groups[kind].append(constraint)
        self.eq_aff = tuple(groups["affine_equality"])
        self.ineq_aff = tuple(groups["affine_inequality"])
        self.eq_nl = tuple(groups["nonlinear_equality"])
        self.ineq_nl = tuple(groups["nonlinear_inequality"])
        self.kinds = frozenset(kind for kind, values in groups.items() if values)
        self._affine = None

    @property
    def has_nonlinear(self):
        return bool(self.eq_nl or self.ineq_nl)

    def _affine_system(self, fixed=None):
        from .affine_system import AffineSystem
        dimension = len(self.env.get_start_pos().state())
        equalities = [(c.mat, c.constraint_pose) for c in self.eq_aff]
        if fixed:
            indices = np.asarray(list(fixed), dtype=int)
            if np.any(indices < 0) or np.any(indices >= dimension):
                raise ValueError("Pinned coordinate outside configuration")
            equalities.append((np.eye(dimension)[indices], list(fixed.values())))
        return AffineSystem(dimension, equalities,
                            [(c.mat, c.constraint_pose) for c in self.ineq_aff],
                            getattr(self.env, "limits", None))

    @property
    def affine(self):
        if self._affine is None:
            self._affine = self._affine_system()
        return self._affine

    def satisfied(self, q, fixed=None):
        x = q.state()
        if fixed and any(i < 0 or i >= len(x) for i in fixed):
            raise ValueError("Pinned coordinate outside configuration")
        return (self.affine.satisfied(x)
                and all(abs(x[i] - value) <= 1e-8 for i, value in (fixed or {}).items())
                and all(c.is_fulfilled(q, self.mode, self.env) for c in self.constraints))

    def edge_satisfied(self, q1, q2, resolution=None, should_stop=None):
        """Check the straight segment; nonlinear feasibility is discretized.

        Affine feasibility follows from the endpoints. Nonlinear checks also
        include interior samples, with at least a midpoint for short edges.
        """
        if should_stop is not None and should_stop():
            return False
        if not self.satisfied(q1) or not self.satisfied(q2):
            return False
        if not self.has_nonlinear:
            return True
        resolution = self.env.collision_resolution if resolution is None else resolution
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("Constraint edge resolution must be positive and finite")
        origin = q1.state()
        delta = q2.state() - origin
        steps = max(2, int(np.ceil(np.max(np.abs(delta)) / resolution)))
        intervals = [(1, steps - 1)]
        nonlinear = self.eq_nl + self.ineq_nl
        while intervals:
            lo, hi = intervals.pop()
            if lo > hi:
                continue
            if should_stop is not None and should_stop():
                return False
            mid = (lo + hi) // 2
            q = q1.from_flat(origin + delta * (mid / steps))
            if not all(c.is_fulfilled(q, self.mode, self.env) for c in nonlinear):
                return False
            intervals.extend(((mid + 1, hi), (lo, mid - 1)))
        return True

    def project(self, q, fixed=None, *, nonlinear_method="backtracking"):
        """Find a feasible configuration, preserving bounds and pinned coordinates.

        Affine-only modes project equalities and reject violated inequalities.
        Nonlinear modes use a local solve; this is not a uniform sampler.
        """
        if nonlinear_method not in ("backtracking", "slsqp_bands"):
            raise ValueError("Unknown nonlinear projection method")
        if self.satisfied(q, fixed):
            return q.from_flat(q.state().copy())
        system = self._affine_system(fixed) if fixed else self.affine
        if self.has_nonlinear:
            from .nonlinear_projection import project_backtracking, project_slsqp_bands
            solver = (project_backtracking if nonlinear_method == "backtracking"
                      else project_slsqp_bands)
            candidate = solver(q, self.constraints, mode=self.mode, env=self.env,
                               affine_system=system)
        else:
            x = system.project(q.state())
            candidate = None if x is None else q.from_flat(x)
        if candidate is None or not self.satisfied(candidate, fixed):
            return None
        return candidate


class ConstraintProvider:
    """Cache definitions by mode identity; call clear() after editing constraints."""

    def __init__(self, env):
        self.env = env
        self._cache = {}

    def clear(self):
        self._cache.clear()

    def _task_constraints(self, task_ids):
        for task_id in dict.fromkeys(task_ids):
            yield from self.env.tasks[task_id].constraints

    def required_types(self):
        """Include all tasks, not just those active in the initial mode."""
        constraints = list(getattr(self.env, "constraints", ()))
        for task in self.env.tasks:
            constraints.extend(task.constraints)
        return ConstraintSet(self.env, None, constraints).kinds

    def for_mode(self, mode):
        key = id(mode)
        if key not in self._cache:
            constraints = list(getattr(self.env, "constraints", ()))
            if mode is not None:
                constraints.extend(self._task_constraints(mode.task_ids))
            self._cache[key] = ConstraintSet(self.env, mode, constraints)
        return self._cache[key]

    def fixed_robots(self, pinned):
        fixed = {}
        for robot, values in (pinned or {}).items():
            indices = np.asarray(self.env.robot_idx[robot], dtype=int).reshape(-1)
            values = np.asarray(values, dtype=float).reshape(-1)
            if len(indices) != len(values) or not np.all(np.isfinite(values)):
                raise ValueError("Invalid pinned robot configuration")
            for index, value in zip(indices, values):
                if index in fixed and fixed[index] != value:
                    raise ValueError("Conflicting pinned coordinates")
                fixed[int(index)] = float(value)
        return fixed

    def path_satisfied(self, path):
        """Check vertices once, and outgoing constraints at mode boundaries."""
        previous = None
        for state in path:
            if not self.for_mode(state.mode).satisfied(state.q):
                return False
            if (previous is not None and previous.mode is not state.mode
                    and not self.for_mode(previous.mode).satisfied(state.q)):
                return False
            previous = state
        return True

    def for_transition(self, mode, next_task_ids):
        """Combine outgoing and incoming affine constraints at a transition."""
        next_task_ids = () if next_task_ids is None else tuple(next_task_ids)
        if not next_task_ids:
            return self.for_mode(mode)
        key = (id(mode), next_task_ids)
        if key not in self._cache:
            constraints = list(self.for_mode(mode).constraints)
            constraints.extend(self._task_constraints(next_task_ids))
            active = ConstraintSet(self.env, mode, constraints)
            if active.has_nonlinear:
                raise NotImplementedError(
                    "Transition dispatch supports only affine constraints; "
                    "nonlinear constraints require separate incoming/outgoing scene checks"
                )
            self._cache[key] = active
        return self._cache[key]
