"""Projection and edge validation for nonlinear shortcut polylines."""

import numpy as np

from multi_robot_multi_goal_planning.problems.planning_env import State


def segment_satisfied(provider, segment, resolution, should_stop=None):
    """Check vertices, discretized edges and identical mode-switch states."""
    for state in segment:
        if should_stop is not None and should_stop():
            return False
        if not provider.for_mode(state.mode).satisfied(state.q):
            return False
    for left, right in zip(segment, segment[1:]):
        if left.mode != right.mode:
            if not np.array_equal(left.q.state(), right.q.state()):
                return False
        elif not provider.for_mode(left.mode).edge_satisfied(
                left.q, right.q, resolution, should_stop=should_stop):
            return False
    return True


def project_segment(provider, segment, active_indices, method, should_stop=None):
    """Project interior points with unselected coordinates pinned.

    Mode-switch copies share one projected configuration, checked in every
    involved scene. Endpoints remain exact to preserve adjacent edges.
    """
    inactive = sorted(set(range(len(segment[0].q.state()))) - set(active_indices))
    out = []
    first = 0
    while first < len(segment):
        if should_stop is not None and should_stop():
            return None
        last = first
        while last + 1 < len(segment) and segment[last].mode != segment[last + 1].mode:
            if not np.array_equal(segment[last].q.state(), segment[last + 1].q.state()):
                return None
            last += 1
        states = segment[first:last + 1]
        constraints = [provider.for_mode(state.mode) for state in states]
        q = states[0].q
        fixed = {i: q.state()[i] for i in inactive}
        if first != 0 and last != len(segment) - 1:
            active = next((c for c in constraints if c.has_nonlinear), constraints[0])
            q = active.project(q, fixed, nonlinear_method=method)
            if q is None:
                return None
            # Restore exact pins before verifying the combined constraints.
            values = q.state().copy()
            values[inactive] = states[0].q.state()[inactive]
            q = q.from_flat(values)
        if not all(c.satisfied(q, fixed) for c in constraints):
            return None
        out.extend(State(q, state.mode) for state in states)
        first = last + 1
    return out
