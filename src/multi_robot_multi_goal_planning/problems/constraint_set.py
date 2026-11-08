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
        if self.has_nonlinear:
            raise NotImplementedError("Only affine feasibility is supported")
        x = q.state()
        if fixed and any(i < 0 or i >= len(x) for i in fixed):
            raise ValueError("Pinned coordinate outside configuration")
        return (self.affine.satisfied(x)
                and all(abs(x[i] - value) <= 1e-8 for i, value in (fixed or {}).items())
                and all(c.is_fulfilled(q, self.mode, self.env) for c in self.constraints))

    def edge_satisfied(self, q1, q2):
        """Affine feasibility of a straight segment follows from its endpoints."""
        return self.satisfied(q1) and self.satisfied(q2)

    def project(self, q, fixed=None):
        """Project equalities and pins, rejecting violated inequalities or bounds."""
        if self.has_nonlinear:
            raise NotImplementedError("Only affine projection is supported")
        if self.satisfied(q, fixed):
            return q.from_flat(q.state().copy())
        system = self._affine_system(fixed) if fixed else self.affine
        x = system.project(q.state())
        if x is None:
            return None
        candidate = q.from_flat(x)
        return candidate if self.satisfied(candidate, fixed) else None


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
