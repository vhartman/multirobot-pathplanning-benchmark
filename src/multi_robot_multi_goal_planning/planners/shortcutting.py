import numpy as np
from multi_robot_multi_goal_planning.problems.rai_base_env import rai_env

import time
import random

from typing import List

from multi_robot_multi_goal_planning.problems.planning_env import State

# from multi_robot_multi_goal_planning.problems.configuration import config_dist
from multi_robot_multi_goal_planning.problems.util import interpolate_path, path_cost


def single_mode_shortcut(env: rai_env, path: List[State], max_iter: int = 1000):
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
    env: rai_env,
    path: List[State],
    max_iter: int = 1000,
    resolution=0.001,
    tolerance=0.01,
    check_edges_in_order:bool = False,
    seed:int = None,

):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    non_redundant_path = remove_interpolated_nodes(path)
    new_path = interpolate_path(non_redundant_path, 0.1)
    # if not all([
    #         env.is_edge_collision_free(path[i].q, path[i+1].q, path[i].mode) 
    #         for i in range(len(path) - 1)
    #     ]):
    #         for i in range(len(path)-1):
    #             if not env.is_edge_collision_free(path[i].q, path[i+1].q, path[i].mode):
    #                 env.is_edge_collision_free(path[i].q, path[i+1].q, path[i].mode)
    
    # if not all([
    #         env.is_edge_collision_free(non_redundant_path[i].q, non_redundant_path[i+1].q, non_redundant_path[i].mode) 
    #         for i in range(len(non_redundant_path) - 1)
    #     ]):
    #         for i in range(len(non_redundant_path)-1):
    #             if not env.is_edge_collision_free(non_redundant_path[i].q, non_redundant_path[i+1].q, non_redundant_path[i].mode):
    #                 env.is_edge_collision_free(non_redundant_path[i].q, non_redundant_path[i+1].q, non_redundant_path[i].mode)

    # if not all([
    #         env.is_edge_collision_free(new_path[i].q, new_path[i+1].q, new_path[i].mode) 
    #         for i in range(len(new_path) - 1)
    #     ]):
    #         for i in range(len(new_path)-1):
    #             if not env.is_edge_collision_free(new_path[i].q, new_path[i+1].q, new_path[i].mode):
    #                 env.is_edge_collision_free(new_path[i].q, new_path[i+1].q, new_path[i].mode)

    costs = [path_cost(new_path, env.batch_config_cost)]
    times = [0.0]

    # for p in new_path:
    #     if not env.is_collision_free(p.q, p.mode):
    #         print("startpath is in collision")

    start_time = time.time()

    config_type = type(env.get_start_pos())

    config_dim = len(env.get_start_pos().state())


    cnt = 0
    # for iter in range(max_iter):
    max_attempts = max_iter * 10
    iter = 0
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

        robots_to_shortcut = [r for r in range(len(env.robots))]
        random.shuffle(robots_to_shortcut)
        # num_robots = np.random.randint(0, len(robots_to_shortcut))
        num_robots = 1
        robots_to_shortcut = robots_to_shortcut[:num_robots]

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
            q = new_path[i + k].q.state() * 1.

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
                State(config_type(q, q0.array_slice), new_path[i + k].mode)
            )
            # path_element.append(State(config_type.from_list(q), new_path[i + k].mode))

        # check if the shortcut improves cost
        if path_cost(path_element, env.batch_config_cost) >= path_cost(
            new_path[i : j + 1], env.batch_config_cost
        ):
            # print(f"{cnt} does not improve cost")
            continue

        assert np.linalg.norm(path_element[0].q.state() - q0.state()) < 1e-6
        assert np.linalg.norm(path_element[-1].q.state() - q1.state()) < 1e-6

        cnt += 1

        # this is wrong for partial shortcuts atm.
        if env.is_path_collision_free(
            path_element, resolution=resolution, tolerance=tolerance, check_edges_in_order=check_edges_in_order
        ):
            for k in range(j - i + 1):
                new_path[i + k].q = path_element[k].q

                # if not np.array_equal(new_path[i+k].mode, path_element[k].mode):
                # print('fucked up')
        # else:
        #     print("in colllision")
        # env.show(True)

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
