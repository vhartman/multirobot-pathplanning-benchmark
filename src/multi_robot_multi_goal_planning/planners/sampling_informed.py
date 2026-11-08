import numpy as np
import random
from typing import List, Tuple, Optional
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
    Mode,
)
from multi_robot_multi_goal_planning.problems.core.configuration import Configuration
from multi_robot_multi_goal_planning.problems.constraint_set import ConstraintProvider
from .sampling_bounded_phs import PreparedBoundedPHS, joint_bounds
from .sampling_joint_cost import JointCostSampler, warm_joint_cost_kernel
from .sampling_affine_informed import AffineInformedSampler, warm_affine_cost_kernel


# Bound the work spent looking for an improvable path interval.
MAX_CONFIGURATION_INTERVAL_ATTEMPTS = 500
MAX_TRANSITION_INTERVAL_ATTEMPTS = 1000


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
        self.constraint_provider = ConstraintProvider(env)
        if proposal == "hit_and_run":
            warm_joint_cost_kernel()
            if affine_sampling:
                warm_affine_cost_kernel()
        self.proposal_batch_size = proposal_batch_size
        self.lower, self.upper = joint_bounds(env)
        metric, reduction = env.cost_metric, env.cost_reduction
        self.cost_settings = ((metric == "euclidean", reduction == "max")
                              if metric in ("euclidean", "max") and reduction in ("sum", "max")
                              else None)
        self.locally_informed_sampling = locally_informed_sampling
        self.include_lb = include_lb

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

        costs_to_transitions = self.env.batch_config_cost(
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

        costs_to_transitions = self.env.batch_config_cost(
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


        tmp = self.env.batch_config_cost(
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
        path_segment_costs = self.env.batch_config_cost(path[:-1], path[1:])

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

        lb_cost_from_start_index_to_state = self.env.config_cost(
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

        lb_cost_from_state_to_end_index = self.env.config_cost(
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
        max_attempts = (MAX_TRANSITION_INTERVAL_ATTEMPTS if transitions
                        else MAX_CONFIGURATION_INTERVAL_ATTEMPTS)
        for _ in range(max_attempts):
            start = random.randrange(len(path))
            end = random.randrange(len(path))
            if end - start <= 2:
                continue
            if transitions and path[start].mode == path[end].mode:
                continue
            if self.env.config_cost(path[start].q, path[end].q) < prefix[end] - prefix[start]:
                return start, end
        return None

    def _interval_mode(self, reached_modes, path, start, end, cache, requested_mode=None):
        if not self.locally_informed_sampling:
            modes = reached_modes
        else:
            key = path[start].mode, path[end].mode
            if key not in cache:
                cache[key] = self.get_inbetween_modes(*key)
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
        candidates = [self.env.start_pos.from_flat(row) for row in rows]
        costs = sum(np.asarray(self.env.batch_config_cost(
            candidates, [self.env.start_pos.from_flat(endpoint)] * len(candidates)))
                    for endpoint in endpoints)
        return rows[np.isfinite(costs) & (costs < budget)]

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
        if (not self.affine_sampling or self.cost_settings is None
                or not active.constraints or active.has_nonlinear):
            return None
        walk = self.proposal == "hit_and_run" and np.random.random() >= 0.1
        try:
            return AffineInformedSampler(self.env, active, fixed or {}, endpoints,
                                         budget, self.cost_settings, self.walk_steps, walk)
        except (ValueError, np.linalg.LinAlgError):
            # Preparation failure is not proof that the region is empty.
            return None

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
        segment_costs = self.env.batch_config_cost(path[:-1], path[1:])
        prefix = np.r_[0., np.cumsum(segment_costs)]
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
        prefix = np.r_[0., np.cumsum(self.env.batch_config_cost(path[:-1], path[1:]))]
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
            proposal = None
            if self.affine_sampling and self.cost_settings is not None:
                active = self.constraint_provider.for_transition(mode, next_ids)
                proposal = self._prepare_affine_proposal(
                    endpoints, budget, active, dict(zip(indices, goal)))
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
                    next_modes = self.env.get_next_modes(q, mode)
                    if not self.env.is_collision_free(q, mode):
                        continue
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
