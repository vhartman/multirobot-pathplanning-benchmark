import numpy as np
import random
from collections import OrderedDict
from copy import deepcopy
from typing import List, Tuple, Optional
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
    Mode,
)
from multi_robot_multi_goal_planning.problems.core.configuration import (
    Configuration, NpConfiguration, _batch_config_cost_impl)
from multi_robot_multi_goal_planning.problems.costs.joint_space import JointSpaceLength
from multi_robot_multi_goal_planning.problems.constraint_set import ConstraintProvider
from .sampling_bounded_phs import PreparedBoundedPHS, joint_bounds
from .sampling_joint_cost import JointCostSampler, joint_cost, warm_joint_cost_kernel
from .sampling_affine_informed import (
    AffineInformedSampler,
    AffineInformedSetup,
    warm_affine_cost_kernel,
)


# Bound the work spent looking for an improvable path interval.
MAX_CONFIGURATION_INTERVAL_ATTEMPTS = 500
MAX_TRANSITION_INTERVAL_ATTEMPTS = 1000
LOCAL_INTERVAL_ATTEMPTS = 30  # index pairs tried for a local interval with slack
LOCAL_INTERVAL_MIN_SLACK = 1e-3  # relative to the interval's cost: below this the informed set is degenerate


def sample_unit_ball(dim, n=1):
    """Return uniform unit-ball samples as columns, shape (dim, n)."""
    unit = np.random.normal(size=(dim, n))
    norms = np.linalg.norm(unit, axis=0)
    while np.any(norms == 0):
        zero = norms == 0
        unit[:, zero] = np.random.normal(size=(dim, int(zero.sum())))
        norms = np.linalg.norm(unit, axis=0)
    return unit * (np.random.random(n) ** (1. / dim) / norms)


def compute_PHS_matrices(a, b, c):
    """Return an ellipsoid transform and center, including coincident foci."""
    phs = PreparedBoundedPHS.from_endpoints(
        a, b, c, np.full(len(a), -np.inf), np.full(len(a), np.inf))
    return phs.u * phs.singular, phs.center


def sample_phs_with_given_matrices(rot, center, n=1):
    return rot @ sample_unit_ball(len(center), n) + center[:, None]


# Visualize a two-dimensional PHS sample cloud.
# import matplotlib.pyplot as plt
# transform, center = compute_PHS_matrices(np.array([-1., 0.]), np.array([1., 0.]), 3.)
# points = sample_phs_with_given_matrices(transform, center, n=500)
# plt.scatter(points[0], points[1])
# plt.axis("equal")
# plt.show()


class InformedSampling:
    """Locally and globally informed sampling with batched bounds/cost checks.

    The environment's declared metric/reduction determine proposal geometry;
    candidate costs are evaluated through its batch_config_cost method.
    L2 sum/max costs use bounded per-robot ellipsoids, otherwise proposals
    use the environment's uniform sampler. Custom costs without this geometry
    must disable try_direct_sampling. Collision checking is scalar and stops
    when the requested number of samples is reached.

    Affine proposals include constraints, bounds and transition goal pins before
    cost rejection. Hit-and-run can walk inside their joint cost region, retaining
    polytope proposals in 10% of intervals. Unconstrained transitions remain
    independent. Short chains are correlated, not IID uniform samples.
    """

    def __init__(
        self,
        env: BaseProblem,
        locally_informed_sampling: bool = False,
        include_lb: bool = False,
        proposal: str = "independent",
        walk_steps: int = 1,
        proposal_batch_size: int = 10,
        affine_sampling: bool = True,
        cache_setup: bool = True,
        reuse_affine_geometry: bool = False,
        informed_transitions: bool = False,
        mode_aware_intervals: bool = True,
        local_share: float = 0.0,
    ):
        if (isinstance(proposal_batch_size, bool)
                or not isinstance(proposal_batch_size, int) or proposal_batch_size < 1):
            raise ValueError("proposal_batch_size must be a positive integer")
        if proposal not in ("independent", "hit_and_run"):
            raise ValueError("proposal must be 'independent' or 'hit_and_run'")
        if isinstance(walk_steps, bool) or not isinstance(walk_steps, int) or walk_steps < 1:
            raise ValueError("walk_steps must be a positive integer")
        self.env = env
        self.proposal = proposal
        self.walk_steps = walk_steps
        self.affine_sampling = affine_sampling
        self.cache_setup = cache_setup
        self._setup_cache = OrderedDict()
        self._costs_path, self._costs_len, self._costs = None, -1, None  # edge costs of the current path
        self._inbetween_cache = {}  # get_inbetween_modes results, valid for one Mode.graph_version
        self._inbetween_version = None
        self.reuse_affine_geometry = reuse_affine_geometry
        self._affine_geometry = OrderedDict()
        self.affine_geometry_stats = dict(hits=0, misses=0)
        self.constraint_provider = ConstraintProvider(env)
        self.proposal_batch_size = proposal_batch_size
        self.lower, self.upper = joint_bounds(env)
        geometry = env.cost.joint_norm_geometry
        self.cost_settings = ((geometry.metric == "euclidean", geometry.reduction == "max")
                              if geometry is not None and geometry.sum_weight == 0.01
                              else None)
        if proposal == "hit_and_run" and self.cost_settings is not None:
            warm_joint_cost_kernel()
            if affine_sampling:
                warm_affine_cost_kernel()
        self.locally_informed_sampling = locally_informed_sampling
        self.include_lb = include_lb
        # Unconstrained transitions: draw from the interval's informed proposal (per-robot PHS, or hit-and-run)
        # instead of the whole joint space. The proposal covers the cost region, so after the exact cost filter the
        # samples have the same distribution; far fewer candidates are rejected.
        self.informed_transitions = informed_transitions
        # Locally informed configuration samples for a requested mode: draw the interval among those whose
        # in-between modes contain the mode, instead of drawing blindly and rejecting. Same distribution of the
        # accepted interval (see _select_interval_for_mode).
        self.mode_aware_intervals = mode_aware_intervals
        # Share of the informed samples whose interval lies inside the sample's own mode (transitions: inside the
        # two modes at the switch); see _local_interval. 0: every interval is drawn over the whole path.
        if not 0.0 <= local_share <= 1.0:
            raise ValueError("local_share must be in [0, 1]")
        self.local_share = local_share
        self._transition_mode = None  # the mode a transition sample is requested for
        self._local_path, self._local_len = None, -1  # the path the mode segments below belong to
        self._local_first, self._local_modes, self._local_index = None, None, {}
        self._requested_mode = None
        self._pairs_path = None  # the path the segment tables belong to (kept alive, so `is` is a safe identity test)
        self._pairs_len = -1
        self._pairs_version = None
        self._segment_first = None  # first path index of each mode segment, plus len(path)
        self._segment_modes = None  # the mode of each segment
        self._segment_pairs = {}  # requested mode id -> (segment i, segment j, cumulative number of index pairs)
        self._inbetween_hashes = {}  # (start mode id, end mode id) -> hashes of the in-between modes
        self._segment_ids = ()  # ids of the path's segment modes, in path order
        self._contains_cache = {}  # (segment mode ids, requested mode id) -> segment x segment containment table

    def sample_mode(
        self, reached_modes: List[Mode], mode_sampling_type: str = "uniform_reached"
    ) -> Mode:
        """
        Selects a mode based on the specified sampling strategy.

        Args:
            reached_modes (List[Mode]): List of modes that have been reached.
            mode_sampling_type (str): Mode sampling strategy to use.

        Returns:
            Mode: Sampled mode according to the specified strategy.
        """

        assert mode_sampling_type == "uniform_reached"
        return random.choice(reached_modes)

    def get_inbetween_modes(self, start_mode: Mode, end_mode: Mode) -> List[Mode]:
        """
        Find all possible paths from start_mode to end_mode.

        Args:
            start_mode (Mode): The starting mode object
            end_mode (Mode): The ending mode object

        Returns:
            Modes on paths between the endpoints, sorted by mode id.
        """
        open_paths = [[start_mode]]

        in_between_modes = set()
        in_between_modes.add(start_mode)
        in_between_modes.add(end_mode)

        while open_paths:
            p = open_paths.pop()
            last_mode = p[-1]

            if last_mode == end_mode:
                for m in p:
                    in_between_modes.add(m)
                continue

            if last_mode.next_modes:
                for mode in last_mode.next_modes:
                    new_path = p.copy()
                    new_path.append(mode)
                    open_paths.append(new_path)

        return list(sorted(in_between_modes, key=lambda m: m.id))

    def _cached_inbetween_modes(self, start_mode: Mode, end_mode: Mode) -> List[Mode]:
        """get_inbetween_modes(start_mode, end_mode), reused across calls while the mode tree has not grown.

        The result depends only on the next_modes links below start_mode and on the modes' equality, which does not
        change after a mode is attached; Mode.graph_version counts attachments. Keyed by mode id: equal but distinct
        modes keep their own entries, exactly as recomputing would. Callers only read the returned list.
        """
        if self._inbetween_version != Mode.graph_version:
            self._inbetween_cache.clear()
            self._inbetween_version = Mode.graph_version
        key = (start_mode.id, end_mode.id)
        result = self._inbetween_cache.get(key)
        if result is None:
            result = self._inbetween_cache[key] = self.get_inbetween_modes(start_mode, end_mode)
        return result

    def lb_cost_from_start(
        self, state: State, g, lb_attribute_name="lb_cost_from_start"
    ):
        if state.mode not in g.reverse_transition_node_array_cache:
            g.reverse_transition_node_array_cache[state.mode] = np.array(
                [o.state.q.q for o in g.reverse_transition_nodes[state.mode]],
                dtype=np.float64,
            )

        if state.mode not in g.rev_transition_node_lb_cache:
            g.rev_transition_node_lb_cache[state.mode] = np.array(
                [
                    getattr(o, lb_attribute_name)
                    for o in g.reverse_transition_nodes[state.mode]
                ],
                dtype=np.float64,
            )

        costs_to_transitions = self.env.cost.batch_lower_bound(
            state.q,
            g.reverse_transition_node_array_cache[state.mode],
        )

        min_cost = np.min(
            g.rev_transition_node_lb_cache[state.mode] + costs_to_transitions
        )

        return min_cost

    def lb_cost_from_goal(self, state: State, g, lb_attribute_name="lb_cost_to_goal"):
        if state.mode not in g.transition_nodes:
            return np.inf

        if state.mode not in g.transition_node_array_cache:
            g.transition_node_array_cache[state.mode] = np.array(
                [o.state.q.q for o in g.transition_nodes[state.mode]],
                dtype=np.float64,
            )

        if state.mode not in g.transition_node_lb_cache:
            g.transition_node_lb_cache[state.mode] = np.array(
                [getattr(o, lb_attribute_name) for o in g.transition_nodes[state.mode]],
                dtype=np.float64,
            )

        costs_to_transitions = self.env.cost.batch_lower_bound(
            state.q,
            g.transition_node_array_cache[state.mode],
        )

        min_cost = np.min(g.transition_node_lb_cache[state.mode] + costs_to_transitions)

        return min_cost

    def can_improve(
        self,
        rnd_state: State,
        path: List[State],
        start_index: int,
        end_index: int,
        path_segment_costs: NDArray,
        g=None,
    ) -> bool:
        """
        Determines if a segment of the path can be improved by comparing its cost to a lower-bound estimate.

        Args:
            rnd_state (State): Reference state used for computing lower-bound costs.
            path (List[State]): Sequence of states representing current path.
            start_index (int): Index marking the start of the segment to evaluate.
            end_index (int): Index marking the end of the segment to evaluate.
            path_segment_costs (NDArray):Costs associated with each segment between consecutive states in the path.

        Returns:
            bool: True if the segment's actual cost exceeds lower-bound estimate (indicating potential improvement); otherwise, False.
        """


        path_cost_cumsum = np.cumsum(path_segment_costs)
        path_cost = path_cost_cumsum[-1]

        path_cost_from_start_to_index = path_cost_cumsum[start_index-1]
        if start_index == 0:
            path_cost_from_start_to_index = 0

        path_cost_from_goal_to_index = path_cost - path_cost_cumsum[end_index-1]

        if start_index == 0:
            assert path_cost_from_start_to_index == 0
        if end_index == len(path) - 1:
            assert path_cost_from_goal_to_index == 0

        path_cost_from_index_to_index = (
            path_cost - path_cost_from_goal_to_index - path_cost_from_start_to_index
        )


        tmp = self.env.cost.batch_lower_bound(
            rnd_state.q, np.array([path[start_index].q.state(), path[end_index].q.state()])
        )
        lb_cost_from_start_index_to_state = tmp[0]
        lb_cost_from_state_to_end_index = tmp[1]

        if self.include_lb:
            if path[start_index].mode != rnd_state.mode:
                start_state = path[start_index]
                lb_cost_from_start_to_state = self.lb_cost_from_start(rnd_state, g)
                if not np.isinf(lb_cost_from_start_to_state):
                    lb_cost_from_start_to_index = self.lb_cost_from_start(start_state, g)
                    if not np.isinf(lb_cost_from_start_to_index):
                        lb_cost_from_start_index_to_state = max(
                            (lb_cost_from_start_to_state - lb_cost_from_start_to_index),
                            lb_cost_from_start_index_to_state,
                        )


        if self.include_lb:
            if path[end_index].mode != rnd_state.mode:
                goal_state = path[end_index]
                lb_cost_from_goal_to_state = self.lb_cost_from_goal(rnd_state, g)
                if not np.isinf(lb_cost_from_goal_to_state):
                    lb_cost_from_goal_to_index = self.lb_cost_from_goal(goal_state, g)
                    if not np.isinf(lb_cost_from_goal_to_index):
                        lb_cost_from_state_to_end_index = max(
                            (lb_cost_from_goal_to_state - lb_cost_from_goal_to_index),
                            lb_cost_from_state_to_end_index,
                        )


        if (
            path_cost_from_index_to_index
            > lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index
        ):
            return True

        return False

    def can_transition_improve(
        self,
        transition: Tuple[Configuration, Mode, Mode],
        path: List[State],
        start_index: int,
        end_index: int,
        g=None,
    ):
        """
        Determines if current path segment can be improved by comparing its actual cost to a lower-bound estimate.

        Args:
            transition (Tuple[Configuration, Mode, Mode]): Tuple containing a configuration and two mode identifiers, used to construct reference states for lower-bound cost calculations.
            path (List[State]): A sequence of states representing current path.
            start_index (int): Index marking the beginning of the segment to evaluate.
            end_index (int): Index marking the end of the segment to evaluate.

        Returns:
            bool: True if the segment's actual cost exceeds lower-bound estimate (indicating potential improvement); otherwise, False.
        """
        path_segment_costs = self.env.cost.batch_edge_cost(path[:-1], path[1:])

        path_cost_from_start_to_index = np.sum(path_segment_costs[:start_index])
        path_cost_from_goal_to_index = np.sum(path_segment_costs[end_index:])
        path_cost = np.sum(path_segment_costs)

        if start_index == 0:
            assert path_cost_from_start_to_index == 0
        if end_index == len(path) - 1:
            assert path_cost_from_goal_to_index == 0

        path_cost_from_index_to_index = (
            path_cost - path_cost_from_goal_to_index - path_cost_from_start_to_index
        )

        rnd_state_mode_1 = State(transition[0], transition[1])
        rnd_state_mode_2 = State(transition[0], transition[2])

        lb_cost_from_start_index_to_state = self.env.cost.lower_bound(
            rnd_state_mode_1.q, path[start_index].q
        )
        if self.include_lb:
            if path[start_index].mode != rnd_state_mode_1.mode:
                start_state = path[start_index]
                lb_cost_from_start_to_state = self.lb_cost_from_start(
                    rnd_state_mode_1, g
                )
                if not np.isinf(lb_cost_from_start_to_state):
                    lb_cost_from_start_to_index = self.lb_cost_from_start(start_state, g)
                    if not np.isinf(lb_cost_from_start_to_index):
                        lb_cost_from_start_index_to_state = max(
                            (lb_cost_from_start_to_state - lb_cost_from_start_to_index),
                            lb_cost_from_start_index_to_state,
                        )

        lb_cost_from_state_to_end_index = self.env.cost.lower_bound(
            rnd_state_mode_2.q, path[end_index].q
        )
        if self.include_lb:
            if path[end_index].mode != rnd_state_mode_2.mode:
                goal_state = path[end_index]
                lb_cost_from_goal_to_state = self.lb_cost_from_goal(rnd_state_mode_2, g)
                if not np.isinf(lb_cost_from_goal_to_state):
                    lb_cost_from_goal_to_index = self.lb_cost_from_goal(goal_state, g)
                    if not np.isinf(lb_cost_from_goal_to_index):
                        lb_cost_from_state_to_end_index = max(
                            (lb_cost_from_goal_to_state - lb_cost_from_goal_to_index),
                            lb_cost_from_state_to_end_index,
                        )


        if (
            path_cost_from_index_to_index
            > lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index
        ):
            return True

        return False

    def _select_interval(self, path, prefix, transitions=False):
        if not self.locally_informed_sampling:
            return 0, len(path) - 1
        if len(path) < 4:
            return None
        mode = self._transition_mode if transitions else self._requested_mode
        if self.local_share > 0 and mode is not None and random.random() < self.local_share:
            interval = self._local_interval(path, prefix, mode, transitions)
            if interval is not None:
                return interval  # otherwise (mode not on the path, or no slack): the rule below
        if not transitions and self._requested_mode is not None:
            return self._select_interval_for_mode(path, prefix, self._requested_mode)
        max_attempts = (MAX_TRANSITION_INTERVAL_ATTEMPTS if transitions
                        else MAX_CONFIGURATION_INTERVAL_ATTEMPTS)
        for _ in range(max_attempts):
            start = random.randrange(len(path))
            end = random.randrange(len(path))
            if end - start <= 2:
                continue
            if transitions and path[start].mode == path[end].mode:
                continue
            if self.env.cost.lower_bound(path[start].q, path[end].q) < prefix[end] - prefix[start]:
                return start, end
        return None

    def _local_interval(self, path, prefix, mode, transitions):
        """An interval inside the sample's own mode, or None.

        The informed set of an interval is defined with the configuration distance as lower bound. Across mode
        switches that distance ignores the goals the robots have to visit in between, so for an interval that
        spans several modes the set is close to the whole space, or at least far larger than what can pay: 97-99%
        of the configuration samples drawn that way cannot shorten the path within their mode, and the other 1-3%
        produce nearly all of the tree's improvements. Inside one mode the bound is exact up to collisions:
          - configuration sample for `mode`: both ends in the mode's segment of the path;
          - transition sample out of `mode`: one end in the mode's segment, the other in its successor's.
        Intervals whose slack is below LOCAL_INTERVAL_MIN_SLACK of their cost are skipped.

        Local samples refine the route the path already takes. Only intervals over the whole path can find another
        one, so local_share should stay below 1 (see BaseRRTConfig.informed_local_share).
        """
        if path is not self._local_path or len(path) != self._local_len:
            first, modes = [], []
            for i, s in enumerate(path):
                if not modes or s.mode is not modes[-1]:
                    modes.append(s.mode)
                    first.append(i)
            self._local_path, self._local_len = path, len(path)  # keeps the path alive: `is` is a safe test
            self._local_first, self._local_modes = first + [len(path)], modes
            self._local_index = {}
            for k, m in enumerate(modes):
                self._local_index.setdefault(m.id, k)
        modes = self._local_modes
        k = self._local_index.get(mode.id)
        if k is None:
            k = next((i for i, m in enumerate(modes) if m == mode), None)
        if k is None:
            return None
        if not transitions:
            return self._draw_local_interval(path, prefix, k, k)
        for k2 in (k - 1, k + 1):  # the informed path may run goal-to-start
            if 0 <= k2 < len(modes) and modes[k2].prev_mode is not None and modes[k2].prev_mode == mode:
                return self._draw_local_interval(path, prefix, min(k, k2), max(k, k2))
        return None

    def _draw_local_interval(self, path, prefix, lo, hi):
        """An index pair (i in segment lo, j in segment hi, j - i > 2) with slack, or None."""
        first = self._local_first
        a1, n1 = first[lo], first[lo + 1] - first[lo]
        a2, n2 = first[hi], first[hi + 1] - first[hi]
        lower_bound = self.env.cost.lower_bound
        for _ in range(LOCAL_INTERVAL_ATTEMPTS):
            i, j = a1 + random.randrange(n1), a2 + random.randrange(n2)
            if j < i and lo == hi:
                i, j = j, i
            if j - i <= 2:
                continue
            budget = prefix[j] - prefix[i]
            if budget - lower_bound(path[i].q, path[j].q) > LOCAL_INTERVAL_MIN_SLACK * budget:
                return i, j
        return None

    def _select_interval_for_mode(self, path, prefix, mode):
        """A configuration interval whose in-between modes contain `mode`.

        _select_interval draws index pairs blindly and generate_samples rejects those whose in-between modes do not
        contain the requested mode (most of them: on a path with K mode segments about 2 / K pass). Here the pair
        is drawn uniformly among the pairs that pass, and only the lower-bound test is left to rejection. The
        accepted interval has the same distribution; the random stream differs.

        The pairs are not enumerated (their number is quadratic in the path length, and shortcut paths have several
        hundred to about 1000 states). A pair of mode segments is drawn with probability proportional to its number
        of index pairs, then one index in each segment; pairs with end - start <= 2 are redrawn. That is uniform
        over the same set, with tables of size (segments)^2.
        """
        first = self._segment_first_indices(path)
        I, J, cumulative = self._mode_segment_pairs(mode)
        if not len(cumulative):
            return None  # no interval contains the mode: generate_samples stops, as after exhausted attempts
        lower_bound = self.env.cost.lower_bound
        tests = redraws = 0
        while tests < MAX_CONFIGURATION_INTERVAL_ATTEMPTS and redraws < 20 * MAX_CONFIGURATION_INTERVAL_ATTEMPTS:
            k = min(int(np.searchsorted(cumulative, random.random() * cumulative[-1], side="right")),
                    len(cumulative) - 1)
            i, j = int(I[k]), int(J[k])
            if i == j:  # two distinct states of one segment
                size = int(first[i + 1] - first[i])
                a = random.randrange(size)
                b = random.randrange(size - 1)
                b += b >= a
                start, end = int(first[i]) + min(a, b), int(first[i]) + max(a, b)
            else:
                start = int(first[i]) + random.randrange(int(first[i + 1] - first[i]))
                end = int(first[j]) + random.randrange(int(first[j + 1] - first[j]))
            if end - start <= 2:
                redraws += 1
                continue
            tests += 1
            if lower_bound(path[start].q, path[end].q) < prefix[end] - prefix[start]:
                return start, end
        return None

    def _segment_first_indices(self, path):
        """First path index of each mode segment, plus len(path) (cached per path and mode-tree version)."""
        if (path is not self._pairs_path or len(path) != self._pairs_len
                or self._pairs_version != Mode.graph_version):
            if self._pairs_version != Mode.graph_version:
                self._inbetween_hashes.clear()
                self._contains_cache.clear()
            self._pairs_path, self._pairs_len, self._pairs_version = path, len(path), Mode.graph_version
            first, modes = [], []
            for i, s in enumerate(path):
                if not modes or s.mode is not modes[-1]:
                    modes.append(s.mode)
                    first.append(i)
            self._segment_first = np.array(first + [len(path)], dtype=np.int64)
            self._segment_modes = modes
            self._segment_ids = tuple(m.id for m in modes)
            self._segment_pairs = {}
        return self._segment_first

    def _mode_segment_pairs(self, mode):
        """Segment pairs (i <= j) whose in-between modes contain `mode`, with their cumulative index-pair counts."""
        table = self._segment_pairs.get(mode.id)
        if table is None:
            contains = self._contains_table(self._segment_modes, mode)
            n = np.diff(self._segment_first)
            I, J = np.nonzero(contains)
            weight = np.where(I == J, n[I] * (n[I] - 1) // 2, n[I] * n[J]).astype(np.float64)
            keep = weight > 0
            table = self._segment_pairs[mode.id] = (I[keep], J[keep], np.cumsum(weight[keep]))
        return table

    def _contains_table(self, seg_modes, mode):
        """contains[i, j] for i <= j: is `mode` among the in-between modes of path segments i and j?

        The table depends only on the segment modes, the requested mode and the mode tree, not on the path's
        states. A new path usually keeps its mode sequence, so the table is kept per (segment mode ids, mode id)
        while the mode tree does not grow, instead of being rebuilt with K * (K + 1) / 2 lookups for every path.
        """
        key = (self._segment_ids, mode.id)
        contains = self._contains_cache.get(key)
        if contains is None:
            if len(self._contains_cache) >= 4096:
                self._contains_cache.clear()
            contains = np.zeros((len(seg_modes), len(seg_modes)), dtype=bool)
            for i in range(len(seg_modes)):
                for j in range(i, len(seg_modes)):
                    contains[i, j] = self._inbetween_contains(seg_modes[i], seg_modes[j], mode)
            self._contains_cache[key] = contains
        return contains

    def _inbetween_contains(self, start_mode, end_mode, mode):
        """`mode in get_inbetween_modes(start_mode, end_mode)`, with Mode.__eq__ semantics, via a hash prefilter."""
        key = start_mode.id, end_mode.id
        hashes = self._inbetween_hashes.get(key)
        if hashes is None:
            hashes = self._inbetween_hashes[key] = frozenset(
                hash(m) for m in self._cached_inbetween_modes(start_mode, end_mode))
        if hash(mode) not in hashes:
            # Mode.__eq__ is id equality or (equal task ids and equal hashes); neither holds without a hash match
            return any(m.id == mode.id for m in self._cached_inbetween_modes(start_mode, end_mode))
        return mode in self._cached_inbetween_modes(start_mode, end_mode)

    def _interval_mode(self, reached_modes, path, start, end, cache, requested_mode=None):
        if not self.locally_informed_sampling:
            modes = reached_modes
        else:
            key = path[start].mode, path[end].mode
            if key not in cache:
                cache[key] = self._cached_inbetween_modes(*key)
            modes = cache[key]
        if requested_mode is not None:
            return requested_mode if requested_mode in modes else None
        return random.choice(modes) if modes else None

    def _project_candidates(self, rows, project):
        """Project before checking costs; rejected proposals are discarded."""
        projected = []
        for row in rows:
            q = project(self.env.start_pos.from_flat(row.copy()))
            if q is not None:
                projected.append(q.state())
        return np.asarray(projected).reshape((-1, rows.shape[1]))

    def _filter_by_bounds_and_cost(self, rows, endpoints, budget):
        """Reject invalid bounds and costs before constructing collision queries."""
        rows = np.asarray(rows, dtype=float)
        valid = (np.all(np.isfinite(rows), axis=1)
                 & np.all(rows >= self.lower, axis=1)
                 & np.all(rows <= self.upper, axis=1))
        rows = rows[valid]
        if not len(rows):
            return rows
        cost = self.env.cost
        if (type(cost).batch_lower_bound is JointSpaceLength.batch_lower_bound
                and isinstance(self.env.start_pos, NpConfiguration)):
            # JointSpaceLength's own batch bound, on the arrays it would build from Configuration objects (bit-identical).
            slices = self.env.start_pos._array_slice
            costs = sum(_batch_config_cost_impl(rows - np.asarray(endpoint, dtype=np.float64), slices,
                                                cost.metric, cost.reduction, cost.sum_weight)
                        for endpoint in endpoints)
            return rows[np.isfinite(costs) & (costs < budget)]
        candidates = [self.env.start_pos.from_flat(row) for row in rows]
        costs = sum(np.asarray(self.env.cost.batch_lower_bound(
            candidates, [self.env.start_pos.from_flat(endpoint)] * len(candidates)))
                    for endpoint in endpoints)
        return rows[np.isfinite(costs) & (costs < budget)]

    def _reuse_setup(self, key, create, stateful=False):
        """Bounded geometry cache; stateful proposals restart from pristine setup."""
        if not self.cache_setup:
            return create()
        if key in self._setup_cache:
            value = self._setup_cache.pop(key)
            self._setup_cache[key] = value
            return deepcopy(value) if stateful else value
        value = create()
        self._setup_cache[key] = deepcopy(value) if stateful else value
        if len(self._setup_cache) > 64:
            self._setup_cache.popitem(last=False)
        return value

    def _proposal_geometry_key(self, endpoints, budget):
        return (endpoints.shape, endpoints.tobytes(), float(budget), self.cost_settings,
                self.env.start_pos._array_slice.tobytes(),
                self.lower.tobytes(), self.upper.tobytes())

    def _prepare_proposal(self, endpoints, budget, mode, direct):
        if not direct or self.cost_settings is None:
            return None
        if self.affine_sampling:
            affine = self._prepare_affine_proposal(
                endpoints, budget, self.constraint_provider.for_mode(mode))
            if affine is not None:
                return affine
        if self.proposal == "hit_and_run" and not self.constraint_provider.for_mode(mode).constraints:
            if np.random.random() >= 0.1:
                try:
                    return JointCostSampler(self.env, endpoints, budget,
                                            self.cost_settings, self.walk_steps)
                except ValueError:
                    pass
        if not self.cost_settings[0] or np.any(self.lower == self.upper):
            return None
        key = ("phs", self._proposal_geometry_key(endpoints, budget))
        return self._reuse_setup(key, lambda: self._prepare_phs(endpoints, budget))

    def _prepare_phs(self, endpoints, budget):
        distances = np.array([np.linalg.norm(endpoints[1, s:e] - endpoints[0, s:e])
                              for s, e in self.env.start_pos._array_slice])
        proposals = []
        for i, (start, end) in enumerate(self.env.start_pos._array_slice):
            bound = budget if self.cost_settings[1] else budget - (distances.sum() - distances[i])
            try:
                proposals.append(PreparedBoundedPHS.from_endpoints(
                    endpoints[0, start:end], endpoints[1, start:end], bound,
                    self.lower[start:end], self.upper[start:end]))
            except ValueError:
                return []
        return proposals

    def _prepare_affine_proposal(self, endpoints, budget, active, fixed=None):
        if (self.local_share > 0 and fixed and self.affine_sampling and self.cost_settings is not None
                and not active.constraints and not active.has_nonlinear):
            return self._prepare_pinned_proposal(endpoints, budget, active, fixed)
        if (not self.affine_sampling or self.cost_settings is None
                or not active.constraints or active.has_nonlinear):
            return None
        walk = self.proposal == "hit_and_run" and np.random.random() >= 0.1
        fixed = fixed or {}
        # Constraint sets are immutable; provider.clear() supplies new identities.
        key = ("affine", active, tuple(fixed.items()), self.walk_steps, walk,
               self._proposal_geometry_key(endpoints, budget))
        try:
            return self._reuse_setup(key, lambda: self._make_affine_proposal(
                endpoints, budget, active, fixed, walk), stateful=True)
        except (ValueError, np.linalg.LinAlgError):
            # Preparation failure is not proof that the region is empty.
            return None

    def _prepare_pinned_proposal(self, endpoints, budget, active, fixed):
        """Proposal for an unconstrained transition: the informed set with the switching robots at their goal.

        A transition sample has the coordinates of the robots that switch task pinned to the goal. Drawing from
        the informed set of all coordinates and pinning afterwards leaves almost nothing that passes the cost
        filter once intervals are local (0.35% on rai.cube_four). Here the candidates come from a hit-and-run walk
        in the set restricted to the pinned coordinates.

        The walk needs a feasible start, and the cheapest configuration of the pinned set is known (_pinned_seed).
        Without slack against it the set is empty or degenerate: [] is returned, nothing is drawn, and the caller
        moves on to the next interval. None: no proposal could be built; the caller falls back.
        """
        seed = self._pinned_seed(endpoints, fixed)
        slices = np.array(self.env.start_pos._array_slice, dtype=np.int64, order="C")
        a, b = np.array(endpoints, dtype=float, copy=True)
        if budget - joint_cost(seed, a, b, slices, *self.cost_settings) <= LOCAL_INTERVAL_MIN_SLACK * budget:
            return []
        try:
            sampler = self._make_affine_proposal(endpoints, budget, active, fixed, True, reuse=True)
        except (ValueError, np.linalg.LinAlgError):
            return None
        if sampler.use_walk and sampler.z is None:
            z = sampler.basis.T @ (seed - sampler.origin)
            if np.all(sampler.G @ z < sampler.h):
                sampler.z = z
                sampler.sample(int(np.ceil(sampler.warmup_sweeps * len(z) / sampler.steps)))  # mix; discarded
        return sampler

    def _pinned_seed(self, endpoints, fixed):
        """The cheapest configuration between the interval's ends that has the pinned coordinates at the goal.

        The pinned robots are at the goal. Every other robot is on its straight line between the ends, at the
        fraction of the way that the pinned robots' first leg takes of their whole way (if another robot's line
        is longer than both legs together: the middle of the fractions that keep both legs below it). Its cost is
        the larger of the pinned legs and the longest straight line, plus the sum term of straight lines: the
        minimum over the set, for the max and for the sum reduction.
        """
        a, b = np.asarray(endpoints[0], dtype=float), np.asarray(endpoints[1], dtype=float)
        idx = np.fromiter(fixed, dtype=int); goal = np.fromiter(fixed.values(), dtype=float)
        first = last = longest = 0.0
        for s_, e_ in self.env.start_pos._array_slice:
            mask = (idx >= s_) & (idx < e_)
            if mask.any():
                via = a[s_:e_].copy(); via[idx[mask] - s_] = goal[mask]
                first = max(first, float(np.linalg.norm(via - a[s_:e_])))
                last = max(last, float(np.linalg.norm(b[s_:e_] - via)))
            else:
                longest = max(longest, float(np.linalg.norm(b[s_:e_] - a[s_:e_])))
        if first + last <= 0.0:
            fraction = 0.5
        elif longest <= first + last:
            fraction = first / (first + last)
        else:
            fraction = 0.5 * (first / longest + 1.0 - last / longest)
        seed = a + fraction * (b - a)
        seed[idx] = goal
        return seed

    def _make_affine_proposal(self, endpoints, budget, active, fixed, walk, reuse=None):
        setup = None
        if self.reuse_affine_geometry if reuse is None else reuse:
            key = (active, tuple(fixed.items()))
            if key in self._affine_geometry:
                self.affine_geometry_stats['hits'] += 1
                self._affine_geometry.move_to_end(key)
            else:
                self.affine_geometry_stats['misses'] += 1
                self._affine_geometry[key] = AffineInformedSetup(active, fixed)
                if len(self._affine_geometry) > 64:
                    self._affine_geometry.popitem(last=False)
            setup = self._affine_geometry[key]
        return AffineInformedSampler(self.env, active, fixed, endpoints, budget,
                                     self.cost_settings, self.walk_steps, walk, setup=setup)

    def _draw_candidates(self, proposal, count):
        if proposal is None:
            return np.array([self.env.sample_config_uniform_in_limits().state().copy()
                             for _ in range(count)])
        if isinstance(proposal, (JointCostSampler, AffineInformedSampler)):
            return proposal.sample(count)
        if not proposal:
            return np.empty((0, len(self.lower)))
        parts = [p.sample(count, max_draws=4 * count)[0] for p in proposal]
        accepted = min(part.shape[1] for part in parts)
        return np.concatenate([part[:, :accepted] for part in parts], axis=0).T

    def _path_costs(self, path):
        """(edge costs, prefix sums) of the path, computed once per path object instead of on every call."""
        if path is not self._costs_path or len(path) != self._costs_len:
            segment_costs = self.env.cost.batch_edge_cost(path[:-1], path[1:])
            self._costs_path, self._costs_len = path, len(path)  # keeps the path alive: `is` is a safe test
            self._costs = (segment_costs, np.r_[0., np.cumsum(segment_costs)])
        return self._costs

    def generate_samples(
        self, reached_modes: List[Mode], batch_size: int, path: List[State],
        max_attempts_per_sample: int = 200, try_direct_sampling: bool = True,
        g=None, requested_mode: Optional[Mode] = None, seed: Optional[int] = None,
        *, max_samples: Optional[int] = None, project_candidate=None, should_stop=None,
    ) -> List[State]:
        """Return states from at most batch_size informed intervals.

        max_samples limits returned states (default: batch_size). Use the requested
        mode directly, skipping incompatible intervals. Without a requested mode,
        draw one from the interval.
        Bounds/costs are filtered in batches; collision queries retain draw order.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        sample_limit = batch_size if max_samples is None else max_samples
        if len(path) < 2 or not reached_modes or sample_limit <= 0:
            return []
        self._requested_mode = (requested_mode if self.mode_aware_intervals and self.locally_informed_sampling
                                else None)
        segment_costs, prefix = self._path_costs(path)
        mode_cache = {}
        samples = []
        for _ in range(batch_size):
            if should_stop is not None and should_stop():
                return samples
            interval = self._select_interval(path, prefix)
            if interval is None:
                break
            start, end = interval
            mode = self._interval_mode(
                reached_modes, path, start, end, mode_cache, requested_mode)
            if mode is None:
                continue
            endpoints = np.array([path[start].q.state(), path[end].q.state()])
            budget = prefix[end] - prefix[start]
            proposal = self._prepare_proposal(endpoints, budget, mode, try_direct_sampling)
            found = False
            for offset in range(0, max_attempts_per_sample, self.proposal_batch_size):
                count = min(self.proposal_batch_size, max_attempts_per_sample - offset)
                if should_stop is not None and should_stop():
                    return samples
                rows = self._draw_candidates(proposal, count)
                if project_candidate is not None:
                    rows = self._project_candidates(rows, lambda q: project_candidate(q, mode))
                for row in self._filter_by_bounds_and_cost(rows, endpoints, budget):
                    q = self.env.start_pos.from_flat(row.copy())
                    state = State(q, mode)
                    if (self.include_lb
                            and not self.can_improve(state, path, start, end, segment_costs, g)):
                        continue
                    if not self.env.is_collision_free(q, mode):
                        continue
                    samples.append(state)
                    found = True
                    if len(samples) == sample_limit:
                        return samples
                if found:
                    break
        # Visualize graph samples per robot (first two or three coordinates).
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # for robot in range(len(self.env.robots)):
        #     points = np.array([state.q[robot] for state in samples])
        #     ax.scatter(points[:, 0], points[:, 1], label=str(robot))
        # ax.legend()
        # plt.show()
        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # robot = 0
        # points = np.array([state.q[robot] for state in samples])
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        # plt.show()
        return samples

    def generate_transitions(
        self, reached_modes: List[Mode], batch_size: int, path: List[State],
        max_attempts_per_sample: int = 100, g=None,
        requested_mode: Optional[Mode] = None, seed: Optional[int] = None,
        *, max_samples: Optional[int] = None, project_transition=None, should_stop=None,
    ) -> List[Tuple[Configuration, Mode, List[Mode]]]:
        """Return (configuration, mode, successor modes) transition samples.

        Search at most batch_size intervals; return at most max_samples results
        (default: batch_size). Use the requested mode, or sample an interval mode.
        Free coordinates share a goal; joint limits are checked after insertion.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        sample_limit = batch_size if max_samples is None else max_samples
        if len(self.env.tasks) == 1 or len(path) < 2 or not reached_modes or sample_limit <= 0:
            return []
        prefix = self._path_costs(path)[1]
        self._transition_mode = requested_mode
        mode_cache = {}
        transitions = []
        for _ in range(batch_size):
            if should_stop is not None and should_stop():
                return transitions
            interval = self._select_interval(path, prefix, transitions=True)
            if interval is None:
                break
            start, end = interval
            if requested_mode is not None:
                mode = requested_mode
            else:
                mode = self._interval_mode(reached_modes, path, start, end, mode_cache)
            if mode is None or self.env.is_terminal_mode(mode):
                continue
            combinations = self.env.get_valid_next_task_combinations(mode)
            if len(combinations) == 0:
                continue
            next_ids = random.choice(combinations)
            task = self.env.get_active_task(mode, next_ids)
            goal = np.asarray(task.goal.sample(mode)).reshape(-1)
            indices = np.concatenate([self.env.robot_idx[r] for r in task.robots]).astype(int)
            if (not np.all(np.isfinite(goal)) or np.any(goal < self.lower[indices])
                    or np.any(goal > self.upper[indices])):
                continue
            endpoints = np.array([path[start].q.state(), path[end].q.state()])
            budget = prefix[end] - prefix[start]
            if (project_transition is None
                    and type(self.env.cost).batch_lower_bound is JointSpaceLength.batch_lower_bound):
                # Every candidate has the goal at `indices`. Its lower bound to an endpoint is at least that of
                # the endpoint with only those coordinates moved to the goal (the bound does not decrease when a
                # robot moves further). If these two legs already use up the budget, no candidate passes the cost
                # filter: skip the interval instead of drawing max_attempts_per_sample candidates for nothing.
                pinned = endpoints.copy()
                pinned[:, indices] = goal
                if (self.env.cost.lower_bound(path[start].q, self.env.start_pos.from_flat(pinned[0]))
                        + self.env.cost.lower_bound(self.env.start_pos.from_flat(pinned[1]), path[end].q)
                        >= budget):
                    continue
            proposal = None
            if self.affine_sampling and self.cost_settings is not None:
                active = self.constraint_provider.for_transition(mode, next_ids)
                proposal = self._prepare_affine_proposal(
                    endpoints, budget, active, dict(zip(indices, goal)))
            if (proposal is None and self.informed_transitions
                    and not self.constraint_provider.for_transition(mode, next_ids).constraints):
                proposal = self._prepare_proposal(endpoints, budget, mode, True)
            found = False
            for offset in range(0, max_attempts_per_sample, self.proposal_batch_size):
                count = min(self.proposal_batch_size, max_attempts_per_sample - offset)
                if should_stop is not None and should_stop():
                    return transitions
                rows = self._draw_candidates(proposal, count)
                rows[:, indices] = goal
                if project_transition is not None:
                    rows = self._project_candidates(
                        rows, lambda q: project_transition(q, mode, next_ids, task))
                for row in self._filter_by_bounds_and_cost(rows, endpoints, budget):
                    q = self.env.start_pos.from_flat(row.copy())
                    if not self.env.is_collision_free(q, mode):
                        continue
                    next_modes = self.env.get_next_modes(q, mode)
                    improving = [m for m in next_modes if not self.include_lb
                                 or self.can_transition_improve((q, mode, m), path, start, end, g)]
                    if not improving:
                        continue
                    transitions.append((q, mode, improving))
                    if len(transitions) == sample_limit:
                        return transitions
                    found = True
                    break
                if found:
                    break
        return transitions
