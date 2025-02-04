import numpy as np
from src.multi_robot_multi_goal_planning.problems.rai_base_env import rai_env

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
        if env.is_edge_collision_free(q0, q1, mode, resolution= 0.001):
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


def robot_mode_shortcut(env: rai_env, path: List[State], max_iter: int = 1000):
    new_path = interpolate_path(path, 0.05)
    costs = [path_cost(new_path, env.batch_config_cost)]
    times = [0.0]

    start_time = time.time()

    config_type = type(env.get_start_pos())

    cnt = 0
    for iter in range(max_iter):
        i = np.random.randint(0, len(new_path))
        j = np.random.randint(0, len(new_path))

        if i > j:
            tmp = i
            i = j
            j = tmp

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

        if not can_shortcut_this:
            continue

        cnt += 1

        q0 = new_path[i].q
        q1 = new_path[j].q

        # constuct pth element for the shortcut
        path_element = []
        for k in range(j - i + 1):
            q = []
            for r in range(len(env.robots)):
                # print(r, i, j, k)
                if r in robots_to_shortcut:
                    q_interp = q0[r] + (q1[r] - q0[r]) / (j - i) * k
                    q.append(q_interp)
                else:
                    q.append(new_path[i + k].q[r])

            # print(q)
            path_element.append(State(config_type.from_list(q), new_path[i + k].mode))

        # check if the shortcut improves cost
        if path_cost(path_element, env.batch_config_cost) >= path_cost(
            new_path[i : j + 1], env.batch_config_cost
        ):
            continue

        # this is wrong for partial shortcuts atm.
        if env.is_path_collision_free(path_element, resolution=0.001, tolerance=0.001):
            for k in range(j - i + 1):
                new_path[i + k].q = path_element[k].q

                # if not np.array_equal(new_path[i+k].mode, path_element[k].mode):
                # print('fucked up')

        current_time = time.time()
        times.append(current_time - start_time)
        costs.append(path_cost(new_path, env.batch_config_cost))

    assert new_path[-1].mode == path[-1].mode
    assert np.linalg.norm(new_path[-1].q.state() - path[-1].q.state()) < 1e-6
    assert np.linalg.norm(new_path[0].q.state() - path[0].q.state()) < 1e-6

    print("original cost:", path_cost(path, env.batch_config_cost))
    print("Attempted shortcuts", cnt)
    print("new cost:", path_cost(new_path, env.batch_config_cost))

    # plt.plot(times, costs)
    # plt.show()

    return new_path, [costs, times]
