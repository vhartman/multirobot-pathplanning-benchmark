import numpy as np
import time as time
import math as math
from typing import Tuple, Optional, Union, List, Dict
from numpy.typing import NDArray
import random
from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
    Mode
)
from multi_robot_multi_goal_planning.planners.rrtstar_base import (
    BaseRRTstar, 
    Node, 
    SingleTree,
    BidirectionalTree

)
from multi_robot_multi_goal_planning.planners.planner_rrtstar import (
    RRTstar

)
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
)
from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
    NpConfiguration,
    batch_config_dist,  
    
)
from multi_robot_multi_goal_planning.problems.planning_env import (
    SingleGoal
)

import multi_robot_multi_goal_planning.planners as mrmgp
class InformedSamplingPerRobot():
    def __init__(self, 
                 env: BaseProblem, 
                 modes: List[Mode], 
                 locally_informed_sampling: bool):
        self.env = env
        self.modes = modes
        self.conf_type = type(self.env.get_start_pos())
        self.locally_informed_sampling = locally_informed_sampling

    def sample_unit_ball(self, dim) -> np.array:
        """Samples a point uniformly from the unit ball. This is used to sample points from the Prolate HyperSpheroid (PHS).

        Returns:
            Sampled Point (np.array): The sampled point from the unit ball.
        """
        # u = np.random.uniform(-1, 1, dim)
        # norm = np.linalg.norm(u)
        # r = np.random.random() ** (1.0 / dim)
        # return r * u / norm
        u = np.random.normal(0, 1, dim)
        norm = np.linalg.norm(u)
        # Generate radius with correct distribution
        r = np.random.random() ** (1.0 / dim)
        return (r / norm) * u
    
    def compute_PHS_matrices(self, a, b, c):
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

    def sample_phs_with_given_matrices(self, rot, center):
        dim = len(center)
        x_ball = self.sample_unit_ball(dim)
        # Transform the point from the unit ball to the PHS.
        # op = np.matmul(np.matmul(cwe, r), x_ball) + center
        return rot @ x_ball + center

    def get_inbetween_modes(start_mode, end_mode):
        """
        Find all possible paths from start_mode to end_mode.

        Args:
            start_mode: The starting mode object
            end_mode: The ending mode object

        Returns:
            A list of lists, where each inner list represents a valid path
            from start_mode to end_mode (inclusive of both).
        """
        # Store all found paths
        open_paths = [[start_mode]]

        in_between_modes = set()
        in_between_modes.add(start_mode)
        in_between_modes.add(end_mode)

        while len(open_paths) > 0:
            p = open_paths.pop()
            last_mode = p[-1]

            if last_mode == end_mode:
                for m in p:
                    in_between_modes.add(m)
                continue

            if len(last_mode.next_modes) > 0:
                for mode in last_mode.next_modes:
                    new_path = p.copy()
                    new_path.append(mode)
                    open_paths.append(new_path)

        return list(in_between_modes)

    def sample_mode(
        modes:List[Mode], mode_sampling_type: str = "uniform_reached", found_solution: bool = False
    ) -> Mode:
        if mode_sampling_type == "uniform_reached":
            m_rnd = random.choice(modes)
        return m_rnd

    def can_improve(
        self, rnd_state: State, path: List[State], start_index, end_index, path_segment_costs
    ) -> bool:
        # path_segment_costs = env.batch_config_cost(path[:-1], path[1:])

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

        # print(path_cost_from_index_to_index)

        lb_cost_from_start_index_to_state = self.env.config_cost(
            rnd_state.q, path[start_index].q
        )
        # if path[start_index].mode != rnd_state.mode:
        #     start_state = path[start_index]
        #     lb_cost_from_start_to_state = lb_cost_from_start(rnd_state)
        #     lb_cost_from_start_to_index = lb_cost_from_start(start_state)

        #     lb_cost_from_start_index_to_state = max(
        #         (lb_cost_from_start_to_state - lb_cost_from_start_to_index),
        #         lb_cost_from_start_index_to_state,
        #     )

        lb_cost_from_state_to_end_index = self.env.config_cost(
            rnd_state.q, path[end_index].q
        )
        # if path[end_index].mode != rnd_state.mode:
        #     goal_state = path[end_index]
        #     lb_cost_from_goal_to_state = lb_cost_from_goal(rnd_state)
        #     lb_cost_from_goal_to_index = lb_cost_from_goal(goal_state)

        #     lb_cost_from_state_to_end_index = max(
        #         (lb_cost_from_goal_to_state - lb_cost_from_goal_to_index),
        #         lb_cost_from_state_to_end_index,
        #     )

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
    
    def generate_informed_samples(
        self, 
        modes,
        batch_size,
        path,
        max_attempts_per_sample=200,
        locally_informed_sampling=True,
        try_direct_sampling=True,
    ):
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
            if locally_informed_sampling:
                for _ in range(500):
                    start_ind = random.randint(0, len(path) - 1)
                    end_ind = random.randint(0, len(path) - 1)

                    if end_ind - start_ind > 2:
                        # if end_ind - start_ind > 2 and end_ind - start_ind < 50:
                        current_cost = sum(path_segment_costs[start_ind:end_ind])
                        lb_cost = self.env.config_cost(path[start_ind].q, path[end_ind].q)

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
                m = self.sample_mode(modes, "uniform_reached", True)

            # print(m)

            current_cost = sum(path_segment_costs[start_ind:end_ind])

            # tmp = 0
            # for i in range(start_ind, end_ind):
            #     tmp += env.config_cost(path[i].q, path[i+1].q)

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

            obv_inv_attempts = 0
            sample_in_collision = 0

            for k in range(max_attempts_per_sample):
                had_to_be_clipped = False
                if not try_direct_sampling or self.env.cost_metric != "euclidean":
                    # completely random sample configuration from the (valid) domain robot by robot
                    q = []
                    for i in range(len(self.env.robots)):
                        t = m.task_ids[i]
                        r = self.env.robots[i]
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
                else:
                    # sample by sampling each agent separately
                    q = []
                    for i in range(len(self.env.robots)):
                        r = self.env.robots[i]
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

                            if (
                                np.linalg.norm(
                                    path[start_ind].q[i] - path[end_ind].q[i]
                                )
                                < 1e-3
                            ):
                                qr = (
                                    np.random.rand(self.env.robot_dims[r])
                                    * (lims[1, :] - lims[0, :])
                                    + lims[0, :]
                                )
                            else:
                                # print("cost", current_cost)
                                # print("robot cst", c_robot_bound)
                                # print(
                                #     np.linalg.norm(
                                #         path[start_ind].q[i] - path[end_ind].q[i]
                                #     )
                                # )

                                if i not in precomputed_phs_matrices:
                                    precomputed_phs_matrices[i] = self.compute_PHS_matrices(
                                        path[start_ind].q[i],
                                        path[end_ind].q[i],
                                        precomputed_robot_cost_bounds[i],
                                    )

                                qr = self.sample_phs_with_given_matrices(
                                    *precomputed_phs_matrices[i]
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

                                clipped = np.clip(qr, lims[0, :], lims[1, :])
                                if not np.array_equal(clipped, qr):
                                    had_to_be_clipped = True
                                    break
                                    # print("AAA")

                        if self.env.is_robot_env_collision_free([r], q, m):
                            pt = (r, q, t)
                        q.append(qr)

                if had_to_be_clipped:
                    continue

                q = self.conf_type.from_list(q)

                if sum(self.env.batch_config_cost(q, focal_points)) > current_cost:
                    # print(path[start_ind].mode, path[end_ind].mode, m)
                    # print(
                    #     current_cost,
                    #     env.config_cost(path[start_ind].q, q)
                    #     + env.config_cost(path[end_ind].q, q),
                    # )
                    # if can_improve(State(q, m), path, start_ind, end_ind):
                    #     assert False

                    obv_inv_attempts += 1

                    continue

                # if can_improve(State(q, m), path, 0, len(path)-1):
                # if can_improve(State(q, m), path, start_ind, end_ind):
                if not self.env.is_collision_free(q, m):
                    sample_in_collision += 1
                    continue

                if self.can_improve(
                    State(q, m), path, start_ind, end_ind, path_segment_costs
                ):
                    # if env.is_collision_free(q, m) and can_improve(State(q, m), path, 0, len(path)-1):
                    new_samples.append(State(q, m))
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

    def can_transition_improve(self, transition, path, start_index, end_index):
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

        # print(path_cost_from_index_to_index)

        rnd_state_mode_1 = State(transition[0], transition[1])
        rnd_state_mode_2 = State(transition[0], transition[2])

        lb_cost_from_start_index_to_state = self.env.config_cost(
            rnd_state_mode_1.q, path[start_index].q
        )
        # if path[start_index].mode != rnd_state_mode_1.mode:
        #     start_state = path[start_index]
        #     lb_cost_from_start_to_state = lb_cost_from_start(rnd_state_mode_1)
        #     lb_cost_from_start_to_index = lb_cost_from_start(start_state)

        #     lb_cost_from_start_index_to_state = max(
        #         (lb_cost_from_start_to_state - lb_cost_from_start_to_index),
        #         lb_cost_from_start_index_to_state,
        #     )

        lb_cost_from_state_to_end_index = self.env.config_cost(
            rnd_state_mode_2.q, path[end_index].q
        )
        # if path[end_index].mode != rnd_state_mode_2.mode:
        #     goal_state = path[end_index]
        #     lb_cost_from_goal_to_state = lb_cost_from_goal(rnd_state_mode_2)
        #     lb_cost_from_goal_to_index = lb_cost_from_goal(goal_state)

        #     lb_cost_from_state_to_end_index = max(
        #         (lb_cost_from_goal_to_state - lb_cost_from_goal_to_index),
        #         lb_cost_from_state_to_end_index,
        #     )

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

    def generate_informed_transitions(
        self, modes, batch_size, path, locally_informed_sampling=False, max_attempts_per_sample=100
    ):
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
            if locally_informed_sampling:
                while True:
                    start_ind = random.randint(0, len(path) - 1)
                    end_ind = random.randint(0, len(path) - 1)

                    if (
                        path[end_ind].mode != path[start_ind].mode
                        and end_ind - start_ind > 2
                    ):
                        # if end_ind - start_ind > 2 and end_ind - start_ind < 50:
                        current_cost = sum(path_segment_costs[start_ind:end_ind])
                        lb_cost = self.env.config_cost(path[start_ind].q, path[end_ind].q)

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
                mode = self.sample_mode( modes, "uniform_reached", True)

            # print(m)

            current_cost = sum(path_segment_costs[start_ind:end_ind])

            # sample transition at the end of this mode
            possible_next_task_combinations = self.env.get_valid_next_task_combinations(mode)
            if len(possible_next_task_combinations) > 0:
                ind = random.randint(0, len(possible_next_task_combinations) - 1)
                active_task = self.env.get_active_task(
                    mode, possible_next_task_combinations[ind]
                )
            else:
                continue

            goals_to_sample = active_task.robots

            goal_sample = active_task.goal.sample(mode)

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
                                        offset : offset + self.env.robot_dims[task_robot]
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

                q = self.conf_type.from_list(q)

                if (
                    self.env.config_cost(path[start_ind].q, q)
                    + self.env.config_cost(path[end_ind].q, q)
                    > current_cost
                ):
                    continue

                if self.env.is_terminal_mode(mode):
                    assert False
                else:
                    next_mode = self.env.get_next_mode(q, mode)

                if self.can_transition_improve(
                    (q, mode, next_mode), path, start_ind, end_ind
                ) and self.env.is_collision_free(q, mode):
                    new_transitions.append((q, mode, next_mode))
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

class InformedVersion6():
    """Locally and globally informed sampling"""
    def __init__(self, 
                 env: BaseProblem, 
                 modes: List[Mode], 
                 locally_informed_sampling: bool):
        self.env = env
        self.modes = modes
        self.conf_type = type(self.env.get_start_pos())
        self.locally_informed_sampling = locally_informed_sampling
        
    def sample_unit_ball(self, dim:int) -> NDArray:
        """ 
        Samples a point uniformly from a n-dimensional unit ball centered at the origin.

        Args: 
            n (int): Dimension of the ball.

        Returns: 
            NDArray: Sampled point from the unit ball. """
        # u = np.random.uniform(-1, 1, dim)
        # norm = np.linalg.norm(u)
        # r = np.random.random() ** (1.0 / dim)
        # return r * u / norm
        u = np.random.normal(0, 1, dim)
        norm = np.linalg.norm(u)
        # Generate radius with correct distribution
        r = np.random.random() ** (1.0 / dim)
        return (r / norm) * u

    def compute_PHS_matrices(self, a:NDArray, b:NDArray, c:float) -> Tuple[NDArray, NDArray]:
        """
        Computes transformation matrix and center for a Prolate Hyperspheroid (PHS) defined by endpoints a and b and cost parameter c.

        Args:
            a (NDArray): Start point of the PHS.
            b (NDArray): End point of the PHS.
            c (float): Cost parameter defining scaling of the PHS.

        Returns:
            Tuple:    
                - NDArray: Transformation matrix (rotation and scaling) of the PHS.
                - NDArray: Center of the PHS, calculated as the midpoint between a and b.
               
        """

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
        # sqrt_term = c**2 - c_min**2

        # if sqrt_term < 0 or np.isnan(sqrt_term):
        #     print("hallo")
        r = np.diag([c * 0.5] + [np.sqrt(c**2 - c_min**2) * 0.5] * (dim - 1))


        return cwe @ r, center

    def sample_phs_with_given_matrices(self, rot:NDArray, center:NDArray) -> NDArray:
        """
        Samples point from a prolate hyperspheroid (PHS) defined by the given rotation matrix and center.

        Args:
            rot (NDArray): Transformation matrix (rotation and scaling) for the PHS.
            center (NDArray): Center point of the PHS.

        Returns:
            NDArray: Sampled point from the PHS.
        """

        dim = len(center)
        x_ball = self.sample_unit_ball(dim)
        return rot @ x_ball + center

    def sample_mode(self,
        mode_sampling_type: str = "uniform_reached"
    ) -> Mode:
        """
        Selects a mode based on the specified sampling strategy.

        Args:
            mode_sampling_type (str): Mode sampling strategy to use.

        Returns:
            Mode: Sampled mode according to the specified strategy.
        """

        if mode_sampling_type == "uniform_reached":
            m_rnd = random.choice(self.modes)
        return m_rnd
    
    def can_improve(self, 
        rnd_state: State, path: List[State], start_index:int, end_index:int, path_segment_costs:NDArray, current_cost
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

        # path_segment_costs = env.batch_config_cost(path[:-1], path[1:])

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

        # print(path_cost_from_index_to_index)

        lb_cost_from_start_index_to_state = self.env.config_cost(
            rnd_state.q, path[start_index].q
        )
        # if path[start_index].mode != rnd_state.mode:
        #     start_state = path[start_index]
        #     lb_cost_from_start_to_state = lb_cost_from_start(rnd_state)
        #     lb_cost_from_start_to_index = lb_cost_from_start(start_state)

        #     lb_cost_from_start_index_to_state = max(
        #         (lb_cost_from_start_to_state - lb_cost_from_start_to_index),
        #         lb_cost_from_start_index_to_state,
        #     )

        lb_cost_from_state_to_end_index = self.env.config_cost(
            rnd_state.q, path[end_index].q
        )
        # if path[end_index].mode != rnd_state.mode:
        #     goal_state = path[end_index]
        #     lb_cost_from_goal_to_state = lb_cost_from_goal(rnd_state)
        #     lb_cost_from_goal_to_index = lb_cost_from_goal(goal_state)

        #     lb_cost_from_state_to_end_index = max(
        #         (lb_cost_from_goal_to_state - lb_cost_from_goal_to_index),
        #         lb_cost_from_state_to_end_index,
        #     )

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
        # focal_points = np.array(
        #         [path[start_index].q.state(), path[end_index].q.state()], dtype=np.float64
        #     )
        # c = sum(self.env.batch_config_cost(rnd_state.q, focal_points))
        # if current_cost != path_cost_from_index_to_index:
        #     print("")
        # if c != lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index:
        #     print("")

        if (
            path_cost_from_index_to_index
            > lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index
        ):
            return True

        return False

    def get_inbetween_modes(self, start_mode, end_mode):
        """
        Find all possible paths from start_mode to end_mode.

        Args:
            start_mode: The starting mode object
            end_mode: The ending mode object

        Returns:
            A list of lists, where each inner list represents a valid path
            from start_mode to end_mode (inclusive of both).
        """
        # Store all found paths
        open_paths = [[start_mode]]

        in_between_modes = set()
        in_between_modes.add(start_mode)
        in_between_modes.add(end_mode)

        while len(open_paths) > 0:
            p = open_paths.pop()
            last_mode = p[-1]

            if last_mode == end_mode:
                for m in p:
                    in_between_modes.add(m)
                continue

            if len(last_mode.next_modes) > 0:
                for mode in last_mode.next_modes:
                    new_path = p.copy()
                    new_path.append(mode)
                    open_paths.append(new_path)

        return list(in_between_modes)

    def generate_informed_samples(self,
        batch_size:int,
        path:List[State],
        mode:Mode,
        max_attempts_per_sample:int =200,
        locally_informed_sampling:bool =True,
        try_direct_sampling:bool =True,
    ) -> Configuration:
        """ 
        Samples configuration from informed set for given mode.

        Args: 
            batch_size (int): Number of samples to generate in a batch.
            path (List[State]): Current path used to guide the informed sampling.
            mode (Mode): Current operational mode.
            max_attempts_per_sample (int, optional): Maximum number of attempts per sample.
            locally_informed_sampling (bool, optional): If True, applies locally informed sampling; otherwise globally.
            try_direct_sampling (bool, optional): If True, attempts direct sampling from the informed set.

        Returns: 
            Configuration: Configuration within the informed set that satisfies the specified limits for the robots. 
        """
        # path = mrmgp.shortcutting.remove_interpolated_nodes(path)
        path = mrmgp.joint_prm_planner.interpolate_path(path)
        new_samples = []
        path_segment_costs = self.env.batch_config_cost(path[:-1], path[1:])
        
        in_between_mode_cache = {}

        num_attempts = 0
        while True:
            if num_attempts > batch_size:
                break

            num_attempts += 1
            # print(len(new_samples))
            # sample mode
            if locally_informed_sampling:
                for _ in range(500):
                    start_ind = random.randint(0, len(path) - 1)
                    end_ind = random.randint(0, len(path) - 1)

                    if end_ind - start_ind > 2:
                        # if end_ind - start_ind > 2 and end_ind - start_ind < 50:
                        current_cost = sum(path_segment_costs[start_ind:end_ind])
                        lb_cost = self.env.config_cost(path[start_ind].q, path[end_ind].q)

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

                m = random.choice(
                    in_between_mode_cache[(path[start_ind].mode, path[end_ind].mode)]
                )

                # k = random.randint(start_ind, end_ind)
                # m = path[k].mode
                # k = random.randint(start_ind, end_ind)
                # m = path[k].mode
            else:
                start_ind = 0
                end_ind = len(path) - 1
                m = self.sample_mode("uniform_reached")
            if mode != m:
                continue

            # print(m)

            current_cost = sum(path_segment_costs[start_ind:end_ind])

            # tmp = 0
            # for i in range(start_ind, end_ind):
            #     tmp += env.config_cost(path[i].q, path[i+1].q)

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

            obv_inv_attempts = 0
            sample_in_collision = 0

            for k in range(max_attempts_per_sample):
                had_to_be_clipped = False
                if not try_direct_sampling or self.env.cost_metric != "euclidean":
                    # completely random sample configuration from the (valid) domain robot by robot
                    q = []
                    for i in range(len(self.env.robots)):
                        r = self.env.robots[i]
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
                else:
                    # sample by sampling each agent separately
                    q = []
                    for i in range(len(self.env.robots)):
                        r = self.env.robots[i]
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

                            norm = np.linalg.norm(
                                    path[start_ind].q[i] - path[end_ind].q[i]
                                )
                            if (norm < 1e-3 or norm > precomputed_robot_cost_bounds[i]
                            ):
                                qr = (
                                    np.random.rand(self.env.robot_dims[r])
                                    * (lims[1, :] - lims[0, :])
                                    + lims[0, :]
                                )
                            else:
                                # print("cost", current_cost)
                                # print("robot cst", c_robot_bound)
                                # print(
                                #     np.linalg.norm(
                                #         path[start_ind].q[i] - path[end_ind].q[i]
                                #     )
                                # )

                                if i not in precomputed_phs_matrices:
                                    precomputed_phs_matrices[i] =self.compute_PHS_matrices(
                                        path[start_ind].q[i],
                                        path[end_ind].q[i],
                                        precomputed_robot_cost_bounds[i],
                                    )

                                qr = self.sample_phs_with_given_matrices(
                                    *precomputed_phs_matrices[i]
                                )
                                # plt.figure()
                                # samples = []
                                # for _ in range(500):
                                #     sample = self.sample_phs_with_given_matrices(
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

                                clipped = np.clip(qr, lims[0, :], lims[1, :])
                                if not np.array_equal(clipped, qr):
                                    had_to_be_clipped = True
                                    break
                                    # print("AAA")

                        q.append(qr)

                if had_to_be_clipped:
                    continue

                q = self.conf_type.from_list(q)

                if sum(self.env.batch_config_cost(q, focal_points)) > current_cost: #gives the same as can_improve...
                    # print(path[start_ind].mode, path[end_ind].mode, m)
                    # print(
                    #     current_cost,
                    #     env.config_cost(path[start_ind].q, q)
                    #     + env.config_cost(path[end_ind].q, q),
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
                    State(q, m), path, start_ind, end_ind, path_segment_costs, current_cost
                ):
                    # if env.is_collision_free(q, m) and can_improve(State(q, m), path, 0, len(path)-1):
                    if m == mode:
                        return q
                    new_samples.append(State(q, m))
                    break

            # print("inv attempt", obv_inv_attempts)
            # print("coll", sample_in_collision)

        # print(len(new_samples) / num_attempts)

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

        return 

    def can_transition_improve(self, transition: Tuple[Configuration, Mode, Mode], path:List[State], start_index:int, end_index:int) -> bool:
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

        # print(path_cost_from_index_to_index)

        rnd_state_mode_1 = State(transition[0], transition[1])
        rnd_state_mode_2 = State(transition[0], transition[2])

        lb_cost_from_start_index_to_state = self.env.config_cost(
            rnd_state_mode_1.q, path[start_index].q
        )
        # if path[start_index].mode != rnd_state_mode_1.mode:
        #     start_state = path[start_index]
        #     lb_cost_from_start_to_state = lb_cost_from_start(rnd_state_mode_1)
        #     lb_cost_from_start_to_index = lb_cost_from_start(start_state)

        #     lb_cost_from_start_index_to_state = max(
        #         (lb_cost_from_start_to_state - lb_cost_from_start_to_index),
        #         lb_cost_from_start_index_to_state,
        #     )

        lb_cost_from_state_to_end_index = self.env.config_cost(
            rnd_state_mode_2.q, path[end_index].q
        )
        # if path[end_index].mode != rnd_state_mode_2.mode:
        #     goal_state = path[end_index]
        #     lb_cost_from_goal_to_state = lb_cost_from_goal(rnd_state_mode_2)
        #     lb_cost_from_goal_to_index = lb_cost_from_goal(goal_state)

        #     lb_cost_from_state_to_end_index = max(
        #         (lb_cost_from_goal_to_state - lb_cost_from_goal_to_index),
        #         lb_cost_from_state_to_end_index,
        #     )

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

    def generate_informed_transitions(self,
        batch_size:int, path:List[State], active_mode:Mode, locally_informed_sampling:bool =True, max_attempts_per_sample:int =100
    ) -> Configuration:
        """ 
        Samples transition configuration from informed set for the given mode.

        Args: 
            batch_size (int): Number of samples to generate in a batch.
            path (List[State]): Current path used to guide the informed sampling.
            active_mode (Mode): Current operational mode.
            locally_informed_sampling (bool, optional): If True, applies locally informed sampling; otherwise globally.
            max_attempts_per_sample (int, optional): Maximum number of attempts per sample.

        Returns: 
            Configuration: Transiiton configuration within the informed set that satisfies the specified limits for the robots. 
        """
        # path = mrmgp.shortcutting.remove_interpolated_nodes(path)
        path =  mrmgp.joint_prm_planner.interpolate_path(path)
        if len(self.env.tasks) == 1:
            return []

        new_transitions = []
        num_attempts = 0
        path_segment_costs = self.env.batch_config_cost(path[:-1], path[1:])

        in_between_mode_cache = {}


        while True:
            num_attempts += 1

            if num_attempts > batch_size:
                break

            # print(len(new_samples))
            # sample mode
            if locally_informed_sampling:
                while True:
                    start_ind = random.randint(0, len(path) - 1)
                    end_ind = random.randint(0, len(path) - 1)

                    if (
                        path[end_ind].mode != path[start_ind].mode
                        and end_ind - start_ind > 2
                    ):
                        # if end_ind - start_ind > 2 and end_ind - start_ind < 50:
                        current_cost = sum(path_segment_costs[start_ind:end_ind])
                        lb_cost = self.env.config_cost(path[start_ind].q, path[end_ind].q)

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
                mode = self.sample_mode("uniform_reached")
            if mode != active_mode:
                continue

            # print(m)

            current_cost = sum(path_segment_costs[start_ind:end_ind])

            # sample transition at the end of this mode
            possible_next_task_combinations = self.env.get_valid_next_task_combinations(mode)
            if len(possible_next_task_combinations) > 0:
                ind = random.randint(0, len(possible_next_task_combinations) - 1)
                active_task = self.env.get_active_task(
                    mode, possible_next_task_combinations[ind]
                )
            else:
                continue

            goals_to_sample = active_task.robots

            goal_sample = active_task.goal.sample(mode)

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
                                        offset : offset + self.env.robot_dims[task_robot]
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

                q = self.conf_type.from_list(q)

                if (
                    self.env.config_cost(path[start_ind].q, q)
                    + self.env.config_cost(path[end_ind].q, q)
                    > current_cost
                ):
                    continue

                if self.env.is_terminal_mode(mode):
                    assert False
                else:
                    next_mode = self.env.get_next_mode(q, mode)

                if self.can_transition_improve(
                    (q, mode, next_mode), path, start_ind, end_ind
                ):# and self.env.is_collision_free(q, mode):
                    if mode == active_mode:
                        return q
                    new_transitions.append((q, mode, next_mode))
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

        return 

class ImplicitTensorGraph:
    robots: List[str]

    def __init__(self, robots):
        self.robots = robots
        self.batch_dist_fun = batch_config_dist

        self.robot_nodes = {}
        self.transition_nodes = {}

        self.goal_nodes = []
        self.per_robot_task_to_goal_lb_cost = {}

    def get_key(self, rs, t):
        robot_key = "_".join(rs)
        return robot_key + "_" +  str(t)

    def add_robot_node(self, rs, q, t, is_transition):
        key = self.get_key(rs, t)
        if key not in self.robot_nodes:
            self.robot_nodes[key] = []
        if is_transition:
            if key not in self.transition_nodes:
                self.transition_nodes[key] = []

            does_already_exist = False
            for n in self.transition_nodes[key]:
                if np.linalg.norm(np.concatenate(q) - n.state()) < 1e-5:
                    does_already_exist = True
                    break

            if not does_already_exist:
                q = NpConfiguration.from_list(q)
                self.transition_nodes[key].append(q)
                self.robot_nodes[key].append(q)
        else:
            self.robot_nodes[key].append(NpConfiguration.from_list(q))

    def get_robot_neighbors(self, rs, q, t, k=50):
        # print(f"finding robot neighbors for {rs}")
        key = self.get_key(rs, t)
        nodes = self.robot_nodes[key]
        dists = self.batch_dist_fun(q, nodes, "euclidean")
        k_clip = min(k, len(nodes) - 1)
        topk = np.argpartition(dists, k_clip)[:k_clip+1]
        topk = topk[np.argsort(dists[topk])]
        best_nodes = [nodes[i] for i in topk]
        return best_nodes

class dRRTstar(BaseRRTstar):
    """Represents the class for the dRRT* based planner"""
    def __init__(self, 
                 env: BaseProblem,
                 ptc: PlannerTerminationCondition,  
                 general_goal_sampling: bool = False, 
                 informed_sampling: bool = False, 
                 informed_sampling_version: int = 6, 
                 distance_metric: str = 'max_euclidean',
                 p_goal: float = 0.1, 
                 p_stay: float = 0.0,
                 p_uniform: float = 0.2, 
                 shortcutting: bool = False, 
                 mode_sampling: Optional[Union[int, float]] = None, 
                 gaussian: bool = False,
                 locally_informed_sampling:bool = True, 
                 remove_redundant_nodes:bool = True,
                 sample_batch_size_per_task:int = 200,
                 transistion_batch_size_per_mode:int = 200, 
                 expand_iter:int = 200
                 
                ):
        super().__init__(env, ptc, general_goal_sampling, informed_sampling, informed_sampling_version, distance_metric,
                    p_goal, p_stay, p_uniform, shortcutting, mode_sampling, 
                    gaussian, locally_informed_sampling = locally_informed_sampling, remove_redundant_nodes = remove_redundant_nodes)
        self.g = None
        self.sample_batch_size_per_task = sample_batch_size_per_task
        self.transistion_batch_size_per_mode = transistion_batch_size_per_mode
        self.expand_iter = expand_iter
        self.conf_type = type(env.get_start_pos())
    
    def sample_uniform_valid_per_robot(self, modes:List[Mode]):
            pts = []
            added_pts_dict = {}
            for mode in modes:
                for i, r in enumerate(self.env.robots):
                    t = mode.task_ids[i]
                    key = tuple([r, t])
                    if key not in added_pts_dict:
                        added_pts_dict[key] = 0
                    task = self.env.tasks[t]
                    lims = self.env.limits[:, self.env.robot_idx[r]]
                    while added_pts_dict[key] < self.sample_batch_size_per_task:
                        q = np.random.uniform(lims[0], lims[1])
                        if self.env.is_robot_env_collision_free([r], q, mode):
                            pt = (r, q, t)
                            pts.append(pt) #TODO need to make sure its a general pt (not the same point for the same seperate graph)
                            added_pts_dict[key] += 1
            
            print(added_pts_dict)
            
            for s in pts:
                r = s[0]
                q = s[1]
                task = s[2]
                
                self.g.add_robot_node([r], [q], task, False)

    def sample_goal_for_active_robots(self, modes:List[Mode]):
        """Sample goals for active robots as vertices for corresponding separate graph"""
        
        for m in modes:
            transitions = []
            while  len(transitions) < self.transistion_batch_size_per_mode:
                next_ids = self.get_next_ids(m)
                active_task = self.env.get_active_task(m, next_ids)
                if len(transitions) > 0 and len(self.env.get_valid_next_task_combinations(m)) <= 1 and type(active_task.goal) is SingleGoal:
                    break
                constrained_robots = active_task.robots
                q = active_task.goal.sample(m)
                t = m.task_ids[self.env.robots.index(constrained_robots[0])]
                if self.env.is_collision_free_for_robot(constrained_robots, q, m):
                    offset = 0
                    for r in constrained_robots:
                        dim = self.env.robot_dims[r]
                        q_transition = q[offset:offset+dim]
                        offset += dim
                        transition = (r, q_transition, t)
                        transitions.append(transition)
            for t in transitions:
                r = t[0]
                q = t[1]
                task = t[2]
                
                self.g.add_robot_node([r], [q], task, True)

    def add_samples_to_graph(self, modes:Optional[List[Mode]]=None):  
        if modes is None:
            modes = self.modes
        
        if self.informed_sampling and self.operation.init_sol: 
            pass
        #sample task goal
        self.sample_goal_for_active_robots(modes)
        # sample uniform
        self.sample_uniform_valid_per_robot(modes)

    def add_new_mode(self, 
                     q:Configuration=None, 
                     mode:Mode=None, 
                     tree_instance: Optional[Union["SingleTree", "BidirectionalTree"]] = None
                     ) -> None:
        """
        Initializes a new mode (including its corresponding tree instance and performs informed initialization).

        Args:
            q (Configuration): Configuration used to determine the new mode. 
            mode (Mode): The current mode from which to get the next mode. 
            tree_instance (Optional[Union["SingleTree", "BidirectionalTree"]]): Type of tree instance to initialize for the next mode. Must be either SingleTree or BidirectionalTree.

        Returns:
            None: This method does not return any value.
        """
        if mode is None: 
            new_mode = self.env.make_start_mode()
            new_mode.prev_mode = None
        else:
            new_mode = self.env.get_next_mode(q, mode)
            new_mode.prev_mode = mode
        if new_mode in self.modes:
            return 
        self.modes.append(new_mode)
        self.add_tree(new_mode, tree_instance)
        self.InformedInitialization(new_mode)
        self.add_samples_to_graph([new_mode])

    def UpdateCost(self, mode:Mode, n:Node) -> None:
       return RRTstar.UpdateCost(self, mode, n)
      
    def ManageTransition(self, mode:Mode, n_new: Node) -> None:
        RRTstar.ManageTransition(self, mode, n_new)

    def KNearest(self, mode:Mode, n_new: Configuration, k:int = 20) -> Tuple[List[Node], NDArray]:
        batch_subtree = self.trees[mode].get_batch_subtree()
        set_dists = batch_config_dist(n_new.state.q, batch_subtree, self.distance_metric)
        indices = np.argsort(set_dists)
        indices = indices[:k]
        N_near_batch = batch_subtree.index_select(0, indices)
        node_indices = self.trees[mode].node_idx_subtree.index_select(0,indices) # actual node indices (node.id)
        n_near_costs = self.operation.costs.index_select(0,node_indices)
        return N_near_batch, n_near_costs, node_indices   
    
    def ChangeParent(self, 
                   mode:Mode, 
                   node_indices: NDArray, 
                   n_new: Node, 
                   batch_cost: NDArray, 
                   n_near_costs: NDArray
                   ) -> None:
        """
        Sets the optimal parent for a new node by evaluating connection costs among candidate nodes.

        Args:
            mode (Mode): Current operational mode.
            node_indices (NDArray): Array of IDs representing candidate neighboring nodes.
            n_new (Node): New node that needs a parent connection.
            n_nearest (Node): Nearest candidate node to n_new.
            batch_cost (NDArray): Costs associated from n_new to all candidate neighboring nodes.
            n_near_costs (NDArray): Cost values for all candidate neighboring nodes.

        Returns:
            None: This method does not return any value.
        """
        c_new_tensor = n_near_costs + batch_cost
        valid_mask = c_new_tensor < n_new.cost
        if np.any(valid_mask):
            sorted_indices = np.where(valid_mask)[0][np.argsort(c_new_tensor[valid_mask])]
            for idx in sorted_indices:
                node = self.trees[mode].subtree.get(node_indices[idx].item())
                if self.env.is_edge_collision_free(node.state.q, n_new.state.q, mode):
                    self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, n_new.id) 
                    n_new.cost = c_new_tensor[idx]
                    n_new.cost_to_parent = batch_cost[idx]   
                    if n_new.parent is not None:
                        n_new.parent.children.remove(n_new)
                    n_new.parent = node     
                    node.children.append(n_new)
                    self.UpdateCost(mode, n_new)                       
                    return
   
    def PlannerInitialization(self) -> None:
        self.set_gamma_rrtstar()
        self.g = ImplicitTensorGraph(self.env.robots)
        # Initilaize first Mode
        self.add_new_mode(tree_instance=SingleTree)
        active_mode = self.modes[-1]
        # Create start node
        start_state = State(self.env.start_pos, active_mode)
        start_node = Node(start_state, self.operation)
        self.trees[active_mode].add_node(start_node)
        start_node.cost = 0.0
        start_node.cost_to_parent = 0.0
          
    def Expand(self, iter:int):
        i = 0
        while i < self.expand_iter:
            i += 1
            active_mode  = self.RandomMode()
            q_rand = self.SampleNodeManifold(active_mode)
            if q_rand is None:
                continue
            #get nearest node in tree8:
            n_nearest, _ , _, _= self.Nearest(active_mode, q_rand)
            self.DirectionOracle(active_mode, q_rand, n_nearest, iter) 

    def CheckForExistingNode(self, mode:Mode, n: Node, tree: str = ''):
        # q_tensor = torch.as_tensor(q_rand.state(), device=device, dtype=torch.float32).unsqueeze(0)
        set_dists = batch_config_dist(n.state.q, self.trees[mode].get_batch_subtree(tree), 'euclidean')
        # set_dists = batch_dist_torch(n.q_tensor.unsqueeze(0), n.state.q, self.trees[mode].get_batch_subtree(tree), self.distance_metric)
        idx = np.argmin(set_dists)
        if set_dists[idx] < 1e-100:
            node_id = self.trees[mode].get_node_ids_subtree(tree)[idx]
            return  self.trees[mode].get_node(node_id, tree)
        return None
    
    def Extend(self, mode:Mode, n_nearest_b:Node, n_new:Node, dist )-> Optional[Node]:
        q = n_new.state.q
        #RRT not RRT*
        i = 1
        while True:
            state_new = self.Steer(mode, n_nearest_b, q, dist, i)
            if not state_new or np.equal(state_new.q.state(), q.state()).all(): # Reached
                # self.SaveData(mode, time.time()-self.start_time, n_nearest = n_nearest_b.state.q.state(), n_rand = q.state(), n_new = n_new.state.q.state())
                return n_nearest_b
            # self.SaveData(mode, time.time()-self.start_time, n_nearest = n_nearest_b.state.q.state(), n_rand = q.state(), n_new = state_new.q.state())
            if self.env.is_collision_free(state_new.q, mode) and self.env.is_edge_collision_free(n_nearest_b.state.q, state_new.q, mode):
                # Add n_new to tree
        
                n_new = Node(state_new,self.operation)
                
                cost =  self.env.batch_config_cost([n_new.state], [n_nearest_b.state])
                c_min = n_nearest_b.cost + cost

                n_new.parent = n_nearest_b
                n_new.cost_to_parent = cost
                n_nearest_b.children.append(n_new) #Set child
                self.trees[mode].add_node(n_new) 
                self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, n_new.id) 
                n_new.cost = c_min
                n_nearest_b = n_new
                i +=1
            else:
                return 
             
    def DirectionOracle(self, mode:Mode, q_rand:Configuration, n_near:Node, iter:int) -> None:
            candidate = []
            for i, r in enumerate(self.env.robots):
                # select configuration of robot r
                q_near = NpConfiguration.from_numpy(n_near.state.q.robot_state(i)) 
                if r not in n_near.neighbors:
                    n_near.neighbors[r] = self.g.get_robot_neighbors([r], q_near, mode.task_ids[i])
                
                best_neighbor = None
                min_angle = np.inf
                vector_to_rand = q_rand.q[self.env.robot_idx[r]] - q_near.q
                if np.all(vector_to_rand == 0):# when its q_near itslef
                    candidate.append(q_near.q)
                    continue
                vector_to_rand = vector_to_rand / np.linalg.norm(vector_to_rand)

                for _ , neighbor in enumerate(n_near.neighbors[r]):
                    vector_to_neighbor = neighbor.q - q_near.q
                    if np.all(vector_to_neighbor == 0):# q_near = neighbor
                        continue
                    vector_to_neighbor = vector_to_neighbor / np.linalg.norm(vector_to_neighbor)
                    angle = np.arccos(np.clip(np.dot(vector_to_rand, vector_to_neighbor), -1.0, 1.0))
                    # print(float(angle))
                    if angle < min_angle:
                        min_angle = angle
                        best_neighbor = neighbor.q
                candidate.append(best_neighbor)
            # self.SaveData(time.time()-self.start_time, n_rand= q_rand.q, n_nearest=n_near.state.q.state(), N_near_ = n_near.neighbors[r]) 
            candidate = NpConfiguration.from_list(candidate)
            # self.SaveData(time.time()-self.start_time, n_nearest = candidate.q) 
            n_candidate = Node(State(candidate, mode), self.operation)
                # if n_candidate not in self.g.blacklist: TODO
                #     break
                # self.g.blacklist.add(n_candidate)
            existing_node = self.CheckForExistingNode(mode, n_candidate)
            
            #node doesn't exist yet
            # existing_node = False 
            if not existing_node:        
                if not self.env.is_edge_collision_free(n_near.state.q, candidate, mode):
                    return
                batch_cost = self.env.batch_config_cost([n_candidate.state], [n_near.state])
                n_candidate.parent = n_near
                n_candidate.cost_to_parent = batch_cost
                n_near.children.append(n_candidate)
                self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, n_candidate.id) 
                n_candidate.cost = n_near.cost + batch_cost
                self.trees[mode].add_node(n_candidate)
                # if self.operation.init_sol:
                #     _, _, set_dists, n_nearest_idx = self.Nearest(mode, n_candidate.state.q) 
                #     N_near_batch, n_near_costs, node_indices = self.Near(mode, n_candidate, n_nearest_idx, set_dists )
                #     batch_cost = self.env.batch_config_cost(n_candidate.state.q, N_near_batch)
                #     if self.Rewire(mode, node_indices, n_candidate, batch_cost, n_near_costs):
                #         self.UpdateCost(mode,n_candidate)
                self.ManageTransition(mode, n_candidate) #Check if we have reached a goal
            else:
                #reuse existing node
                # existing_node_ = self.CheckForExistingNode(mode, n_candidate) # TODO make sure n_near is in neighbors
                # _, _, set_dists, n_nearest_idx = self.Nearest(mode, existing_node.state.q)  #TODO not needed?
                idx =  np.where(self.trees[mode].get_node_ids_subtree() == n_near.id)[0][0] 
                N_near_batch, n_near_costs, node_indices = self.Near(mode, existing_node, idx)
                batch_cost = self.env.batch_config_cost(existing_node.state.q, N_near_batch)
                # batch_cost = self.env.batch_config_cost([existing_node.state], [n_near.state])
                self.ChangeParent(mode, node_indices, existing_node, batch_cost, n_near_costs)
                if self.Rewire(mode, node_indices, existing_node, batch_cost, n_near_costs):
                    self.UpdateCost(mode, existing_node)
            self.FindLBTransitionNode()
                       
    def ConnectToTarget(self, mode:Mode, iter:int):
        """Local connector: Tries to connect to a transition node in mode"""
        #Not implemented as described in paper which uses a selected order
        # # select random termination node of created ones and try to connect
        new_node = True
        if self.operation.init_sol and self.env.is_terminal_mode(mode):
            #when termination node is restricted for all agents -> don't create a new transition node            
            node_id = np.random.choice(self.transition_node_ids[mode])
            termination_node = self.trees[mode].subtree.get(node_id)
            terminal_q = termination_node.state.q
            new_node = False
        else:
            terminal_q = self.sample_transition_configuration(mode)
            termination_node = Node(State(terminal_q, mode), self.operation)
            self.operation.costs = self.trees[mode].ensure_capacity(self.operation.costs, termination_node.id)
            termination_node.cost = np.inf

        
        # N_near_batch, n_near_costs, node_indices = self.KNearest(mode, terminal_q_tensor.unsqueeze(0), terminal_q) #TODO
        _, _, set_dists, n_nearest_idx = self.Nearest(mode, termination_node.state.q) 
        N_near_batch, n_near_costs, node_indices = self.Near(mode, termination_node, n_nearest_idx, set_dists)
        batch_cost = self.env.batch_config_cost(termination_node.state.q, N_near_batch)
        c_terminal_costs = n_near_costs + batch_cost       
        sorted_mask = np.argsort(c_terminal_costs)
        for idx in sorted_mask:
            if termination_node.cost > c_terminal_costs[idx]:
                node = self.trees[mode].get_node(node_indices[idx].item())
                dist = batch_config_dist(node.state.q, [termination_node.state.q], self.distance_metric)
                # dist = batch_dist_torch(node.q_tensor, node.state.q, termination_node.q_tensor.unsqueeze(0), self.distance_metric)
                n_nearest = self.Extend(mode, node, termination_node, dist)
                if n_nearest is not None:
                    if self.env.is_edge_collision_free(n_nearest.state.q, terminal_q,  mode):
                        cost = self.env.batch_config_cost([n_nearest.state], [termination_node.state])
                        if termination_node.parent is not None:
                            termination_node.parent.children.remove(termination_node)
                        termination_node.parent = n_nearest
                        termination_node.cost_to_parent = cost
                        n_nearest.children.append(termination_node) #Set child
                        if new_node:
                            self.trees[mode].add_node(termination_node)
                        termination_node.cost = n_nearest.cost + cost
                        if self.operation.init_sol:
                            _, _, set_dists, n_nearest_idx = self.Nearest(mode, termination_node.state.q)  
                            N_near_batch, n_near_costs, node_indices = self.Near(mode, termination_node, n_nearest_idx, set_dists)
                            batch_cost = self.env.batch_config_cost(termination_node.state.q, N_near_batch)
                            if self.Rewire(mode, node_indices, termination_node, batch_cost, n_near_costs):
                                self.UpdateCost(mode, termination_node)
                        self.ManageTransition(mode, termination_node)
                        return 

    def SampleNodeManifold(self, mode:Mode) -> Configuration:
        """
        Samples a node configuration from the manifold based on various probabilistic strategies.

        Args:
            mode (Mode): Current operational mode.

        Returns:
            Configuration: Configuration obtained by a sampling strategy based on preset probabilities and operational conditions.
        """

        if  np.random.uniform(0, 1) < self.p_goal:
            # goal sampling
            return self.sample_configuration(mode, "goal", self.transition_node_ids, self.trees[mode].order)
        else:       
            if self.informed_sampling and self.operation.init_sol: 
                if self.informed_sampling_version == 0 and np.random.uniform(0, 1) < self.p_uniform or self.informed_sampling_version == 5 and np.random.uniform(0, 1) < self.p_uniform:
                    #uniform sampling
                    return self.sample_configuration(mode, "uniform")
                #informed_sampling
                return self.sample_configuration(mode, "informed")
            # gaussian sampling
            if self.gaussian and self.operation.init_sol: 
                return self.sample_configuration(mode, "gaussian")
            # home pose sampling
            if np.random.uniform(0, 1) < self.p_stay: 
                return self.sample_configuration(mode, "home_pose")
            #uniform sampling
            return self.sample_configuration(mode, "uniform")

    def sample_transition_configuration(self, mode) -> Configuration:
        """
        Samples a collision-free transition configuration for the given mode.

        Args:
            mode (Mode): Current operational mode.

        Returns:
            Configuration: Collision-free configuration constructed by combining goal samples (active robots) with random samples (non-active robots).
        """
        next_ids = self.get_next_ids(mode)
        constrained_robot = self.env.get_active_task(mode, next_ids).robots
        goal = self.env.get_active_task(mode, next_ids).goal.sample(mode)
        q = []
        end_idx = 0
        for robot in self.env.robots:
            if robot in constrained_robot:
                dim = self.env.robot_dims[robot]
                indices = list(range(end_idx, end_idx + dim))
                q.append(goal[indices])
                end_idx += dim 
                continue
            lims = self.env.limits[:, self.env.robot_idx[robot]]
            q.append(np.random.uniform(lims[0], lims[1]))
        q = type(self.env.get_start_pos()).from_list(q)  
        return q 
            # if self.env.is_collision_free(q, mode):
            #     return q

    def sample_configuration(self, 
                             mode:Mode, 
                             sampling_type: str, 
                             transition_node_ids:Dict[Mode, List[int]] = None, 
                             tree_order:int = 1
                             ) -> Configuration:
        """
        Samples a collision-free configuration for the given mode using the specified sampling strategy.
        
        Args: 
            mode (Mode): Current operational mode. 
            sampling_type (str): String type specifying the sampling strategy to use. 
            transition_node_ids (Optional[Dict[Mode, List[int]]]): Dictionary mapping modes to lists of transition node IDs. 
            tree_order (int): Order of the subtrees (i.e. value of 1 indicates sampling from 'subtree' (primary); otherwise from 'subtree_b')

        Returns: 
            Configuration:Collision-free configuration within specified limits for the robots based on the sampling strategy. 
        """
        is_goal_sampling = sampling_type == "goal"
        is_informed_sampling = sampling_type == "informed"
        is_home_pose_sampling = sampling_type == "home_pose"
        is_gaussian_sampling = sampling_type == "gaussian"
        constrained_robots = self.env.get_active_task(mode, self.get_next_ids(mode)).robots
        attemps = 0  # needed if home poses are in collision

        #goal sampling
        if is_goal_sampling:
            if tree_order == -1:
                if mode.prev_mode is None: 
                    return self.env.start_pos
                else: 
                    transition_nodes_id = transition_node_ids[mode.prev_mode]
                    if transition_nodes_id == []:
                        return self.sample_transition_configuration(mode.prev_mode)
                        
                    else:
                        node_id = np.random.choice(transition_nodes_id)
                        node = self.trees[mode.prev_mode].subtree.get(node_id)
                        if node is None:
                            node = self.trees[mode.prev_mode].subtree_b.get(node_id)
                        return node.state.q
                    
            if self.operation.init_sol and self.informed: 
                if self.informed_sampling_version == 6 and not self.env.is_terminal_mode(mode):
                    q = self.informed[mode].generate_informed_transitions(
                        self.informed_batch_size,
                        self.operation.path_shortcutting, mode, locally_informed_sampling = self.locally_informed_sampling
                    )
                    if q is None:
                        return
                    return q
                elif not self.informed_sampling_version == 6:
                    q = self.sample_informed(mode, True)
                    if q is None:
                        return
                        # q = self.sample_transition_configuration(mode)
                        # if random.choice([0,1]) == 0:
                        #     return q
                        # while True:
                        #     q_noise = []
                        #     for r in range(len(self.env.robots)):
                        #         q_robot = q.robot_state(r)
                        #         noise = np.random.normal(0, 0.1, q_robot.shape)
                        #         q_noise.append(q_robot + noise)
                        #     q = type(self.env.get_start_pos()).from_list(q_noise)
                        #     if self.env.is_collision_free(q, mode):
                        #         return q
                    q = type(self.env.get_start_pos()).from_list(q)
                    return q
            q = self.sample_transition_configuration(mode)
            if random.choice([0,1]) == 0:
                return q
            q_noise = []
            for r in range(len(self.env.robots)):
                q_robot = q.robot_state(r)
                noise = np.random.normal(0, 0.1, q_robot.shape)
                q_noise.append(q_robot + noise)
            q = type(self.env.get_start_pos()).from_list(q_noise)
            return q
        #informed sampling       
        if is_informed_sampling:
            if self.informed_sampling_version == 6:
                q = self.informed[mode].generate_informed_samples(
                    self.informed_batch_size,
                    self.operation.path_shortcutting, mode, locally_informed_sampling = self.locally_informed_sampling
                    )
                if q is None:
                    return
                    is_informed_sampling = False
                return q
            elif not self.informed_sampling_version == 6:
                q = self.sample_informed(mode)
                if q is None:
                    # is_informed_sampling = False
                    # continue
                    return
        #gaussian noise
        if is_gaussian_sampling: 
            path_state = np.random.choice(self.operation.path)
            standar_deviation = np.random.uniform(0, 5.0)
            # standar_deviation = 0.5
            noise = np.random.normal(0, standar_deviation, path_state.q.state().shape)
            q = (path_state.q.state() + noise).tolist()
        #home pose sampling or uniform sampling
        else: 
            q = []
            if is_home_pose_sampling:
                attemps += 1
                q_home = self.get_home_poses(mode)
            for robot in self.env.robots:
                #home pose sampling
                if is_home_pose_sampling:
                    r_idx = self.env.robots.index(robot)
                    if robot not in constrained_robots: # can cause problems if several robots are not constrained and their home poses are in collision
                        q.append(q_home[r_idx])
                        continue
                    if np.array_equal(self.get_task_goal_of_agent(mode, robot), q_home[r_idx]):
                        if np.random.uniform(0, 1) < self.p_goal: # goal sampling
                            q.append(q_home[r_idx])
                            continue
                #uniform sampling
                lims = self.env.limits[:, self.env.robot_idx[robot]]
                q.append(np.random.uniform(lims[0], lims[1]))
        q = type(self.env.get_start_pos()).from_list(q)
        return q

    def Plan(self) -> List[State]:
        i = 0
        self.PlannerInitialization()
        
        while True:
            i += 1
            self.Expand(i)
            active_mode  = self.RandomMode()
            self.ConnectToTarget(active_mode, i)
            if self.operation.init_sol and i %100 == 0: # make it better!
                self.add_samples_to_graph() 
            if self.ptc.should_terminate(i, time.time() - self.start_time):
                break

        self.costs.append(self.operation.cost)
        self.times.append(time.time() - self.start_time)
        self.all_paths.append(self.operation.path)
        info = {"costs": self.costs, "times": self.times, "paths": self.all_paths}
        return self.operation.path, info    

