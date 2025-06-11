import numpy as np
import random
from typing import List, Tuple, Optional, Union
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
    Mode,
)
from multi_robot_multi_goal_planning.problems.configuration import Configuration


# taken from https://github.com/marleyshan21/Batch-informed-trees/blob/master/python/BIT_Star.py
# needed adaption to work.
def sample_unit_ball(dim, n=1) -> np.ndarray:
    """Samples n points uniformly from the unit ball. This is used to sample points from the Prolate HyperSpheroid (PHS).

    Args:
        dim (int): Dimension of the unit ball.
        n (int): Number of points to sample. Default is 1.

    Returns:
        np.ndarray: An array of shape (n, dim) containing the sampled points.
    """
    u = np.random.normal(0, 1, (dim, n))
    norms = np.linalg.norm(u, axis=0, keepdims=True)
    # Generate radii with correct distribution
    r = np.random.random(n) ** (1.0 / dim)
    return (r[None, :] / norms) * u


def compute_PHS_matrices(a, b, c):
    dim = len(a)
    diff = b - a

    # Calculate the center of the PHS.
    center = (a + b) / 2
    # The transverse axis in the world frame.
    c_min = np.linalg.norm(diff)

    # The first column of the identity matrix.
    # one_1 = np.eye(a1.shape[0])[:, 0]
    a1 = diff / c_min
    e1 = np.zeros(dim)
    e1[0] = 1.0

    # Optimized rotation matrix calculation
    U, S, Vt = np.linalg.svd(np.outer(a1, e1))
    # Sigma = np.diag(S)
    # lam = np.eye(Sigma.shape[0])
    lam = np.eye(dim)
    lam[-1, -1] = np.linalg.det(U) * np.linalg.det(Vt.T)
    # Calculate the rotation matrix.
    # cwe = np.matmul(U, np.matmul(lam, Vt))
    cwe = U @ lam @ Vt
    # Get the radius of the first axis of the PHS.
    # r1 = c / 2
    # Get the radius of the other axes of the PHS.
    # rn = [np.sqrt(c**2 - c_min**2) / 2] * (dim - 1)
    # Create a vector of the radii of the PHS.
    # r = np.diag(np.array([r1] + rn))
    r = np.diag([c * 0.5] + [np.sqrt(c**2 - c_min**2) * 0.5] * (dim - 1))

    return cwe @ r, center


def sample_phs_with_given_matrices(rot, center, n=1):
    dim = len(center)
    x_ball = sample_unit_ball(dim, n)
    # Transform the point from the unit ball to the PHS.
    # op = np.matmul(np.matmul(cwe, r), x_ball) + center
    return rot @ x_ball + center[:, None]


class InformedSampling:
    """Locally and globally informed sampling"""

    def __init__(
        self,
        env: BaseProblem,
        planning_approach: str,
        locally_informed_sampling: bool = False,
        include_lb: bool = False,
    ):
        self.env = env
        self.planning_approach = planning_approach
        self.locally_informed_sampling = locally_informed_sampling
        self.include_lb = include_lb
        self.conf_type = type(self.env.get_start_pos())

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
            A list of lists, where each inner list represents a valid path
            from start_mode to end_mode (inclusive of both).
        """
        # Store all found paths
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

        # return list(in_between_modes)
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
        # path_segment_costs = self.env.batch_config_cost(path[:-1], path[1:])

        # compute the local cost
        # path_cost_from_start_to_index = np.sum(path_segment_costs[:start_index])
        # path_cost_from_goal_to_index = np.sum(path_segment_costs[end_index:])
        # path_cost = np.sum(path_segment_costs)

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

        # print(path_cost_from_index_to_index)

        tmp = self.env.batch_config_cost(
            rnd_state.q, np.array([path[start_index].q.state(), path[end_index].q.state()])
        )
        lb_cost_from_start_index_to_state = tmp[0]
        lb_cost_from_state_to_end_index = tmp[1]

        # lb_cost_from_start_index_to_state = self.env.config_cost(
        #     rnd_state.q, path[start_index].q
        # )
        if self.planning_approach == "graph_based" and self.include_lb:
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

        # lb_cost_from_state_to_end_index = self.env.config_cost(
        #     rnd_state.q, path[end_index].q
        # )

        if self.planning_approach == "graph_based" and self.include_lb:
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

        # print("can_imrpove")

        # print("start", lb_cost_from_start_index_to_state)
        # print("end", lb_cost_from_state_to_end_index)

        # print('start index', start_index)
        # print('end_index', end_index)

        # assert(lb_cost_from_start_index_to_state >= 0)
        # assert(lb_cost_from_state_to_end_index >= 0)

        # print("segment cost", path_cost_from_index_to_index)
        # print(
        #     "lb cost",
        #     lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index,
        # )

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

        # compute the local cost
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
        if self.planning_approach == "graph_based" and self.include_lb:
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
        if self.planning_approach == "graph_based" and self.include_lb:
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

        # print("can_imrpove")

        # print("start", lb_cost_from_start_index_to_state)
        # print("end", lb_cost_from_state_to_end_index)

        # print('start index', start_index)
        # print('end_index', end_index)

        # assert(lb_cost_from_start_index_to_state >= 0)
        # assert(lb_cost_from_state_to_end_index >= 0)

        # print("segment cost", path_cost_from_index_to_index)
        # print(
        #     "lb cost",
        #     lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index,
        # )

        if (
            path_cost_from_index_to_index
            > lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index
        ):
            return True

        return False

    def generate_samples(
        self,
        reached_modes: List[Mode],
        batch_size: int,
        path: List[State],
        max_attempts_per_sample: int = 200,
        try_direct_sampling: bool = True,
        g=None,
        active_mode: Optional[Mode] = None,
        seed: Optional[int] = None,
    ) -> Optional[Union[Configuration, List[Configuration]]]:
        """
        Samples configuration from informed set for given mode.

        Args:
            reached_modes (List[Mode]): List of modes that have been reached.
            batch_size (int): Number of samples to generate in a batch.
            path (List[State]): Current path used to guide the informed sampling.
            max_attempts_per_sample (int, optional): Maximum number of attempts per sample.
            try_direct_sampling (bool, optional): If True, attempts direct sampling from the informed set.
            g (Optional[Graph]): Graph object used for lower-bound cost calculations.
            active_mode (Optional[Mode]): Current operational mode.
            seed (Optional[int]): Random seed for reproducibility.

        Returns:
            Configuration, List[Configuration]: One or several configurations within the informed set that satisfies the specified limits for the robots.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        new_samples = []
        path_segment_costs = self.env.batch_config_cost(path[:-1], path[1:])

        in_between_mode_cache = {}

        num_attempts = 0
        while len(new_samples) < batch_size:
            if num_attempts > batch_size:
                break

            num_attempts += 1
            # print(len(new_samples))
            # sample mode
            if self.locally_informed_sampling:
                path_segment_costs_cumsum = np.cumsum(path_segment_costs)

                for _ in range(500):
                    start_ind = random.randint(0, len(path) - 1)
                    end_ind = random.randint(0, len(path) - 1)

                    if end_ind - start_ind > 2:
                        # if end_ind - start_ind > 2 and end_ind - start_ind < 50:
                        current_cost = path_segment_costs_cumsum[end_ind - 1] - (path_segment_costs_cumsum[start_ind - 1] if start_ind > 0 else 0)
                        # current_cost = sum(path_segment_costs[start_ind:end_ind])
                        lb_cost = self.env.config_cost(
                            path[start_ind].q, path[end_ind].q
                        )

                        if lb_cost < current_cost:
                            break

                # TODO: we need to sample from the set of all reachable modes here
                # not only from the modes on the path
                if (
                    path[start_ind].mode,
                    path[end_ind].mode,
                ) not in in_between_mode_cache:
                    in_between_modes = self.get_inbetween_modes(
                        path[start_ind].mode, path[end_ind].mode
                    )
                    in_between_mode_cache[
                        (path[start_ind].mode, path[end_ind].mode)
                    ] = in_between_modes

                # print(in_between_mode_cache[(path[start_ind].mode, path[end_ind].mode)])

                m = random.choice(
                    in_between_mode_cache[(path[start_ind].mode, path[end_ind].mode)]
                )

                # k = random.randint(start_ind, end_ind)
                # m = path[k].mode
            else:
                start_ind = 0
                end_ind = len(path) - 1
                m = self.sample_mode(reached_modes=reached_modes)
            if self.planning_approach == "sampling_based" and active_mode != m:
                continue

            # print(m)

            current_cost = sum(path_segment_costs[start_ind:end_ind])

            # tmp = 0
            # for i in range(start_ind, end_ind):
            #     tmp += self.env.config_cost(path[i].q, path[i+1].q)

            # print(current_cost, tmp)

            # plt.figure()
            # samples = []
            # for _ in range(500):
            #     sample = samplePHS(np.array([-1, 1, 0]), np.array([1, -1, 0]), 3)
            #     # sample = samplePHS(np.array([[-1], [0]]), np.array([[1], [0]]), 3)
            #     samples.append(sample[:2])
            #     print("sample", sample)

            # plt.scatter([a[0] for a in samples], [a[1] for a in samples])
            # plt.show()

            focal_points = np.array(
                [path[start_ind].q.state(), path[end_ind].q.state()], dtype=np.float64
            )

            precomputed_phs_matrices = {}
            precomputed_robot_cost_bounds = {}

            is_almost_the_same = {}

            obv_inv_attempts = 0
            sample_in_collision = 0

            num_samples_at_a_time = 10

            for k in range(max_attempts_per_sample // num_samples_at_a_time):
                if not try_direct_sampling or self.env.cost_metric != "euclidean":
                    # completely random sample configuration from the (valid) domain robot by robot
                    q = self.env.sample_config_uniform_in_limits()
                else:
                    # sample by sampling each agent separately
                    q = []
                    for i, r in enumerate(self.env.robots):
                        lims = self.env.limits[:, self.env.robot_idx[r]]
                        if lims[0, 0] < lims[1, 0]:
                            if i not in precomputed_robot_cost_bounds:
                                if self.env.cost_reduction == "sum":
                                    precomputed_robot_cost_bounds[i] = (
                                        current_cost
                                        - sum(
                                            [
                                                np.linalg.norm(
                                                    path[start_ind].q[j]
                                                    - path[end_ind].q[j]
                                                )
                                                for j in range(len(self.env.robots))
                                                if j != i
                                            ]
                                        )
                                    )
                                else:
                                    precomputed_robot_cost_bounds[i] = current_cost

                            if i not in is_almost_the_same:
                                is_almost_the_same[i] = (
                                    np.linalg.norm(
                                        path[start_ind].q[i] - path[end_ind].q[i]
                                    )
                                    < 1e-3
                                )

                            if is_almost_the_same[i]:
                                qr = np.random.uniform(
                                    size=(
                                        num_samples_at_a_time,
                                        self.env.robot_dims[r],
                                    ),
                                    low=lims[0, :],
                                    high=lims[1, :],
                                ).T
                            else:
                                # print("cost", current_cost)
                                # print("robot cst", c_robot_bound)
                                # print(
                                #     np.linalg.norm(
                                #         path[start_ind].q[i] - path[end_ind].q[i]
                                #     )
                                # )

                                if i not in precomputed_phs_matrices:
                                    precomputed_phs_matrices[i] = compute_PHS_matrices(
                                        path[start_ind].q[i],
                                        path[end_ind].q[i],
                                        precomputed_robot_cost_bounds[i],
                                    )

                                qr = sample_phs_with_given_matrices(
                                    *precomputed_phs_matrices[i],
                                    n=num_samples_at_a_time,
                                )

                                # plt.figure()
                                # samples = []
                                # for _ in range(500):
                                #     sample = sample_phs_with_given_matrices(
                                #         *precomputed_phs_matrices[i]
                                #     )
                                #     # sample = samplePHS(np.array([[-1], [0]]), np.array([[1], [0]]), 3)
                                #     samples.append(sample[:2])
                                #     print("sample", sample)

                                # plt.scatter(
                                #     [a[0] for a in samples], [a[1] for a in samples]
                                # )
                                # plt.show()

                                # qr = samplePHS(path[start_ind].q[i], path[end_ind].q[i], c_robot_bound)
                                # qr = rejection_sample_from_ellipsoid(
                                #     path[start_ind].q[i], path[end_ind].q[i], c_robot_bound
                                # )

                                # if np.linalg.norm(path[start_ind].q[i] - qr) + np.linalg.norm(path[end_ind].q[i] - qr) > c_robot_bound:
                                #     print("AAAAAAAAA")
                                #     print(np.linalg.norm(path[start_ind].q[i] - qr) + np.linalg.norm(path[end_ind].q[i] - qr), c_robot_bound)

                                # clipped = np.clip(qr, lims[0, :], lims[1, :])
                                # if not np.array_equal(clipped, qr):
                                # if np.any((qr < lims[0, :]) | (qr > lims[1, :])):
                                #     had_to_be_clipped = True
                                #     break
                                # print("AAA")

                        q.append(qr)

                if isinstance(q, list):
                    qs = []
                    for i in range(num_samples_at_a_time):
                        q_config = []
                        for j in range(len(self.env.robots)):
                            q_config.append(q[j][:, i])

                        qnp = np.concatenate(q_config)
                        qs.append(self.env.start_pos.from_flat(qnp))
                else:
                    qs = [q]

                found_a_sample = False
                for q in qs:
                    if not isinstance(q, Configuration):
                        # q = conf_type.from_list(q)
                        qnp = np.concatenate(q)
                        if np.any(
                            (qnp < self.env.limits[0, :])
                            | (qnp > self.env.limits[1, :])
                        ):
                            continue
                        q = self.env.start_pos.from_flat(qnp)

                    if sum(self.env.batch_config_cost(q, focal_points)) > current_cost:
                        # print(path[start_ind].mode, path[end_ind].mode, m)
                        # print(
                        #     current_cost,
                        #     self.env.config_cost(path[start_ind].q, q)
                        #     + self.env.config_cost(path[end_ind].q, q),
                        # )
                        # if can_improve(State(q, m), path, start_ind, end_ind):
                        #     assert False

                        obv_inv_attempts += 1

                        continue

                    # if can_improve(State(q, m), path, 0, len(path)-1):
                    # if can_improve(State(q, m), path, start_ind, end_ind):
                    # if not self.env.is_collision_free(q, m):
                    #     sample_in_collision += 1
                    #     continue

                    if self.can_improve(
                        State(q, m), path, start_ind, end_ind, path_segment_costs, g
                    ) and self.env.is_collision_free(q, m):
                        # if self.env.is_collision_free(q, m) and can_improve(State(q, m), path, 0, len(path)-1):
                        if self.planning_approach == "sampling_based":
                            return q
                        new_samples.append(State(q, m))
                        found_a_sample = True
                        # break

                if found_a_sample:
                    break

            # print("inv attempt", obv_inv_attempts)
            # print("coll", sample_in_collision)

        print(len(new_samples) / num_attempts)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter([a.q[0][0] for a in new_samples], [a.q[0][1] for a in new_samples], [a.q[0][2] for a in new_samples])
        # ax.scatter([a.q[1][0] for a in new_samples], [a.q[1][1] for a in new_samples], [a.q[1][2] for a in new_samples])
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.scatter([a.q[0][0] for a in new_samples], [a.q[0][1] for a in new_samples])
        # ax.scatter([a.q[1][0] for a in new_samples], [a.q[1][1] for a in new_samples])
        # plt.show()

        return new_samples

    def generate_transitions(
        self,
        reached_modes: List[Mode],
        batch_size: int,
        path: List[State],
        max_attempts_per_sample: int = 100,
        g=None,
        active_mode: Optional[Mode] = None,
        seed: Optional[int] = None,
    ) -> Optional[Union[Configuration, List[Configuration]]]:
        """
        Samples transition configuration from informed set for the given mode.

        Args:
            reached_modes (List[Mode]): List of modes that have been reached.
            batch_size (int): Number of samples to generate in a batch.
            path (List[State]): Current path used to guide the informed sampling.
            max_attempts_per_sample (int, optional): Maximum number of attempts per sample.
            g (Optional[Graph]): Graph object used for lower-bound cost calculations.
            active_mode (Optional[Mode]): Current operational mode.
            seed (Optional[int]): Random seed for reproducibility.

        Returns:
            Configuration, List[Configuration]: One or several configurations within the informed set that satisfies the specified limits for the robots.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        if len(self.env.tasks) == 1:
            return []

        new_transitions = []
        num_attempts = 0
        path_segment_costs = self.env.batch_config_cost(path[:-1], path[1:])

        in_between_mode_cache = {}

        while len(new_transitions) < batch_size:
            num_attempts += 1

            if num_attempts > batch_size:
                break

            # print(len(new_samples))
            # sample mode
            if self.locally_informed_sampling:
                path_segment_costs_cumsum = np.cumsum(path_segment_costs)

                while True:
                    start_ind = random.randint(0, len(path) - 1)
                    end_ind = random.randint(0, len(path) - 1)

                    if (
                        path[end_ind].mode != path[start_ind].mode
                        and end_ind - start_ind > 2
                    ):
                        # if end_ind - start_ind > 2 and end_ind - start_ind < 50:
                        # current_cost = sum(path_segment_costs[start_ind:end_ind])
                        current_cost = path_segment_costs_cumsum[end_ind - 1] - (path_segment_costs_cumsum[start_ind - 1] if start_ind > 0 else 0)

                        lb_cost = self.env.config_cost(
                            path[start_ind].q, path[end_ind].q
                        )

                        if lb_cost < current_cost:
                            break

                if (
                    path[start_ind].mode,
                    path[end_ind].mode,
                ) not in in_between_mode_cache:
                    in_between_modes = self.get_inbetween_modes(
                        path[start_ind].mode, path[end_ind].mode
                    )
                    in_between_mode_cache[
                        (path[start_ind].mode, path[end_ind].mode)
                    ] = in_between_modes

                # print(in_between_mode_cache[(path[start_ind].mode, path[end_ind].mode)])

                mode = random.choice(
                    in_between_mode_cache[(path[start_ind].mode, path[end_ind].mode)]
                )

                # k = random.randint(start_ind, end_ind)
                # mode = path[k].mode
            else:
                start_ind = 0
                end_ind = len(path) - 1
                mode = self.sample_mode(reached_modes=reached_modes)
            if self.planning_approach == "sampling_based" and active_mode != mode:
                continue

            # print(m)

            # sample transition at the end of this mode
            possible_next_task_combinations = self.env.get_valid_next_task_combinations(
                mode
            )
            if len(possible_next_task_combinations) > 0:
                ind = random.randint(0, len(possible_next_task_combinations) - 1)
                active_task = self.env.get_active_task(
                    mode, possible_next_task_combinations[ind]
                )
            else:
                continue

            goals_to_sample = active_task.robots

            goal_sample = active_task.goal.sample(mode)

            focal_points = np.array(
                [path[start_ind].q.state(), path[end_ind].q.state()], dtype=np.float64
            )

            current_cost = sum(path_segment_costs[start_ind:end_ind])

            for k in range(max_attempts_per_sample):
                # completely random sample configuration from the (valid) domain robot by robot
                q = []
                for i in range(len(self.env.robots)):
                    r = self.env.robots[i]
                    if r in goals_to_sample:
                        offset = 0
                        for _, task_robot in enumerate(active_task.robots):
                            if task_robot == r:
                                q.append(
                                    goal_sample[
                                        offset : offset
                                        + self.env.robot_dims[task_robot]
                                    ]
                                )
                                break
                            offset += self.env.robot_dims[task_robot]
                    else:  # uniform sample
                        lims = self.env.limits[:, self.env.robot_idx[r]]
                        if lims[0, 0] < lims[1, 0]:
                            qr = (
                                np.random.rand(self.env.robot_dims[r])
                                * (lims[1, :] - lims[0, :])
                                + lims[0, :]
                            )
                        else:
                            qr = np.random.rand(self.env.robot_dims[r]) * 6 - 3

                        q.append(qr)

                q = self.env.start_pos.from_flat(np.concatenate(q))

                to_q_cost = self.env.batch_config_cost(q, focal_points)
                if to_q_cost[0] + to_q_cost[1] > current_cost:
                    continue

                if self.env.is_terminal_mode(mode):
                    assert False
                else:
                    next_modes = self.env.get_next_modes(q, mode)

                if not self.env.is_collision_free(q, mode):
                    continue
                    
                improving_modes = []
                for next_mode in next_modes:
                    if self.can_transition_improve(
                        (q, mode, next_mode), path, start_ind, end_ind, g
                    ):
                        if self.planning_approach == "sampling_based":
                            return q
                        
                        improving_modes.append(next_mode)
                        
                if len(improving_modes) > 0:
                    new_transitions.append((q, mode, improving_modes))
                    break

        print(len(new_transitions) / num_attempts)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter([a.q[0][0] for a in new_samples], [a.q[0][1] for a in new_samples], [a.q[0][2] for a in new_samples])
        # ax.scatter([a.q[1][0] for a in new_samples], [a.q[1][1] for a in new_samples], [a.q[1][2] for a in new_samples])
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.scatter([a[0][0][0] for a in new_transitions], [a[0][0][1] for a in new_transitions])
        # ax.scatter([a[0][1][0] for a in new_transitions], [a[0][1][1] for a in new_transitions])
        # plt.show()

        return new_transitions
