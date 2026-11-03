"""Collect mode-local path constraints independently of sampling and projection."""


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

    @property
    def has_nonlinear(self):
        return bool(self.eq_nl or self.ineq_nl)


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
