import numpy as np

import time
import random

from typing import List

from multi_robot_multi_goal_planning.problems.constraints import AffineConfigurationSpaceEqualityConstraint, AffineConfigurationSpaceInequalityConstraint
from multi_robot_multi_goal_planning.problems.constraints_projection import project_affine_cspace_interior
from multi_robot_multi_goal_planning.problems.planning_env import State, BaseProblem

# from multi_robot_multi_goal_planning.problems.configuration import config_dist
from multi_robot_multi_goal_planning.problems.util import interpolate_path, path_cost


def single_mode_shortcut(env: BaseProblem, path: List[State], max_iter: int = 1000):
    """
    Shortcutting the composite path a single mode at a time.
    I.e. we never shortcut over mode transitions, even if it would be possible.

    Works by randomly sampling indices of the path, and attempting to do a shortcut if it is in the same mode.
    """
    new_path = interpolate_path(path, 0.05)

    costs = [path_cost(new_path, env.batch_config_cost)]
    times = [0.0]

    start_time = time.time()

    cnt = 0

    for _ in range(max_iter):
        i = np.random.randint(0, len(new_path))
        j = np.random.randint(0, len(new_path))

        if i > j:
            tmp = i
            i = j
            j = tmp

        if abs(j - i) < 2:
            continue

        if new_path[i].mode != new_path[j].mode:
            continue

        q0 = new_path[i].q
        q1 = new_path[j].q
        mode = new_path[i].mode

        # check if the shortcut improves cost
        if path_cost([new_path[i], new_path[j]], env.batch_config_cost) >= path_cost(
            new_path[i:j], env.batch_config_cost
        ):
            continue

        cnt += 1

        robots_to_shortcut = [r for r in range(len(env.robots))]
        if False:
            random.shuffle(robots_to_shortcut)
            num_robots = np.random.randint(0, len(robots_to_shortcut))
            robots_to_shortcut = robots_to_shortcut[:num_robots]

        # this is wrong for partial shortcuts atm.
        if env.is_edge_collision_free(
            q0,
            q1,
            mode,
            resolution=env.collision_resolution,
            tolerance=env.collision_tolerance,
        ):
            for k in range(j - i):
                for r in robots_to_shortcut:
                    q = q0[r] + (q1[r] - q0[r]) / (j - i) * k
                    new_path[i + k].q[r] = q

        current_time = time.time()
        times.append(current_time - start_time)
        costs.append(path_cost(new_path, env.batch_config_cost))

    print("original cost:", path_cost(path, env.batch_config_cost))
    print("Attempted shortcuts: ", cnt)
    print("new cost:", path_cost(new_path, env.batch_config_cost))

    return new_path, [costs, times]


def robot_mode_shortcut(
    env: BaseProblem,
    path: List[State],
    max_iter: int = 1000,
    resolution=0.001,
    tolerance=0.01,
    robot_choice = "round_robin",
    interpolation_resolution: float=0.1
):
    """
    Shortcutting the composite path one robot at a time, but allowing shortcutting over the modes as well if the
    robot we are shortcutting is not active.

    Works by randomly sampling indices, then randomly choosing a robot, and then checking if the direct interpolation is
    collision free.
    """
    non_redundant_path = remove_interpolated_nodes(path)
    new_path = interpolate_path(non_redundant_path, interpolation_resolution)

    costs = [path_cost(new_path, env.batch_config_cost)]
    times = [0.0]

    # for p in new_path:
    #     if not env.is_collision_free(p.q, p.mode):
    #         print("startpath is in collision")

    start_time = time.time()

    cnt = 0
    # for iter in range(max_iter):
    max_attempts = 250 * 10
    iter = 0

    rr_robot = 0

    def collect_affine_constraints_for_env():
        eq = [c for c in env.constraints if isinstance(c, AffineConfigurationSpaceEqualityConstraint)]
        ineq = [c for c in env.constraints if isinstance(c, AffineConfigurationSpaceInequalityConstraint)]
        return eq, ineq
    
    env_eq_constraints, env_ineq_constraints = collect_affine_constraints_for_env()

    while True:
        iter += 1
        if cnt >= max_iter or iter >= max_attempts:
            break

        i = np.random.randint(0, len(new_path))
        j = np.random.randint(0, len(new_path))

        if i > j:
            q = i
            i = j
            j = q

        if abs(j - i) < 2:
            continue

        # robots_to_shortcut = [r for r in range(len(env.robots))]
        # random.shuffle(robots_to_shortcut)
        # # num_robots = np.random.randint(0, len(robots_to_shortcut))
        # num_robots = 1
        # robots_to_shortcut = robots_to_shortcut[:num_robots]
        if robot_choice == "round_robin":
            robots_to_shortcut = [rr_robot % len(env.robots)]
            rr_robot += 1
        else:
            robots_to_shortcut = [np.random.randint(0, len(env.robots))]

        can_shortcut_this = True
        for r in robots_to_shortcut:
            if new_path[i].mode.task_ids[r] != new_path[j].mode.task_ids[r]:
                can_shortcut_this = False
                break

        if not can_shortcut_this:
            continue

        # if not env.is_path_collision_free(new_path[i:j], resolution=0.01, tolerance=0.01):
        #     print("path is not collision free")
        #     env.show(True)

        q0 = new_path[i].q
        q1 = new_path[j].q

        # precopmute all the differences
        q0_tmp = {}
        q1_tmp = {}
        diff_tmp = {}
        for r in robots_to_shortcut:
            q0_tmp[r] = q0[r] * 1
            q1_tmp[r] = q1[r] * 1
            diff_tmp[r] = (q1_tmp[r] - q0_tmp[r]) / (j - i)

        # constuct pth element for the shortcut
        path_element = []
        for k in range(j - i + 1):
            q = new_path[i + k].q.state() * 1.0

            r_cnt = 0
            for r in range(len(env.robots)):
                # print(r, i, j, k)
                dim = env.robot_dims[env.robots[r]]
                if r in robots_to_shortcut:
                    q_interp = q0_tmp[r] + diff_tmp[r] * k
                    q[r_cnt : r_cnt + dim] = q_interp
                # else:
                #     q[r_cnt : r_cnt + dim] = new_path[i + k].q[r]

                r_cnt += dim

            # print(tmp)
            # print(q)

            # print(q)
            path_element.append(
                State(q0.from_flat(q), new_path[i + k].mode)
            )

        # path_element = []
        # for k in range(j - i + 1):
        #     q_flat = new_path[i + k].q.state().astype(float, copy=True)

        #     r_cnt = 0
        #     for r in range(len(env.robots)):
        #         dim = env.robot_dims[env.robots[r]]
        #         if r in robots_to_shortcut:
        #             q_interp = q0_tmp[r] + diff_tmp[r] * k
        #             q_flat[r_cnt : r_cnt + dim] = q_interp
        #         r_cnt += dim

        #     q_config = q0.from_flat(q_flat)

        #     if 0 < k < (j - i):
        #         q_proj_config = project_affine_cspace_interior(
        #             q_config,
        #             eq_constraints=env_eq_constraints,
        #             ineq_constraints=env_ineq_constraints,
        #         )
        #         if q_proj_config is None:
        #             feasible = False
        #             break
        #         q_use_config = q_proj_config
        #     else:
        #         q_use_config = q_config
        #     path_element.append(State(q_use_config, new_path[i + k].mode))


        # check if the shortcut improves cost
        if path_cost(path_element, env.batch_config_cost) >= path_cost(
            new_path[i : j + 1], env.batch_config_cost
        ):
            # print(f"{cnt} does not improve cost")
            continue

        assert np.linalg.norm(path_element[0].q.state() - q0.state()) < 1e-6
        assert np.linalg.norm(path_element[-1].q.state() - q1.state()) < 1e-6

        cnt += 1

        constraints_ok = True
        for state in path_element:
            # Check all active equality and inequality constraints
            for c in env_eq_constraints:
                if not c.is_fulfilled(state.q, env):
                    constraints_ok = False
                    break

            for c in env_ineq_constraints:
                if not c.is_fulfilled(state.q, env):
                    constraints_ok = False
                    break

            if not constraints_ok:
                break

        if not constraints_ok:
            continue  # reject this shortcut and try another

        if env.is_path_collision_free(
            path_element, resolution=resolution, tolerance=tolerance, check_start_and_end=False
        ):
            for k in range(j - i + 1):
                new_path[i + k].q = path_element[k].q

                # if not np.array_equal(new_path[i+k].mode, path_element[k].mode):
                # print('fucked up')
        # else:
        #     print("in colllision")
        # env.show(True)

        # print(i, j, len(path_element))

        current_time = time.time()
        times.append(current_time - start_time)
        costs.append(path_cost(new_path, env.batch_config_cost))

    assert new_path[-1].mode == path[-1].mode
    assert np.linalg.norm(new_path[-1].q.state() - path[-1].q.state()) < 1e-6
    assert np.linalg.norm(new_path[0].q.state() - path[0].q.state()) < 1e-6

    print("original cost:", path_cost(path, env.batch_config_cost))
    print("Attempted shortcuts", cnt)
    print("new cost:", path_cost(new_path, env.batch_config_cost))

    return new_path, [costs, times]


def remove_interpolated_nodes(path: List[State], tolerance=1e-15) -> List[State]:
    """
    Removes interpolated points from a given path, retaining only key nodes where direction changes or new mode begins.

    Args:
        path (List[Object]): Sequence of states representing original path.
        tolerance (float, optional): Threshold for detecting collinearity between segments.

    Returns:
        List[Object]: Sequence of states representing a path without redundant nodes.
    """

    if len(path) < 3:
        return path

    simplified_path = [path[0]]

    for i in range(1, len(path) - 1):
        A = simplified_path[-1]
        B = path[i]
        C = path[i + 1]

        AB = B.q.state() - A.q.state()
        AC = C.q.state() - A.q.state()

        # If A and C are almost the same, skip B.
        if np.linalg.norm(AC) < tolerance:
            continue
        lam = np.dot(AB, AC) / np.dot(AC, AC)

        # Check if AB is collinear to AC (AB = lambda * AC)
        if np.linalg.norm(AB - lam * AC) > tolerance or A.mode != C.mode:
            simplified_path.append(B)

    simplified_path.append(path[-1])

    return simplified_path
