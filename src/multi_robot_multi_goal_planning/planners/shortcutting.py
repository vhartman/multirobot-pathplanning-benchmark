import numpy as np

import time
import random

from typing import List

from multi_robot_multi_goal_planning.problems.planning_env import State, BaseProblem

# from multi_robot_multi_goal_planning.problems.configuration import config_dist
from multi_robot_multi_goal_planning.problems.util import interpolate_path, path_cost

# TODO (Liam) added
from multi_robot_multi_goal_planning.problems.planning_env import Mode 


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
    start_time = time.time()

    cnt = 0
    max_attempts = 250 * 10
    iter = 0
    rr_robot = 0

    # TODO (Liam) Helper function to check if mode contains skill task for given robot
    def mode_contains_skill_for_robot(env: BaseProblem, mode: Mode, robot_index: int) -> bool:
        """
        Check if the mode corresponds to a skill task for the robot_index
        Need this helper function because global shortcutter has no other way knowing
        about skill segments
        """
        task_id = mode.task_ids[robot_index]
        if task_id is None:
            return False
        task = env.tasks[task_id]
        contains_skill = getattr(task, "skill", None) is not None
        return contains_skill

    while True:
        iter += 1
        if cnt >= max_iter or iter >= max_attempts:
            break

        i = np.random.randint(0, len(new_path))
        j = np.random.randint(0, len(new_path))

        if i > j:
            i, j = j, i

        if abs(j - i) < 2:
            continue

        # Choose (one) robot to shortcut
        if robot_choice == "round_robin":
            robots_to_shortcut = [rr_robot % len(env.robots)]
            rr_robot += 1
        else:
            robots_to_shortcut = [np.random.randint(0, len(env.robots))]

        can_shortcut_this = True
        for r in robots_to_shortcut: # Only one robot in the list!
            # Check 1: must be same task id at endpoints
            if new_path[i].mode.task_ids[r] != new_path[j].mode.task_ids[r]:
                can_shortcut_this = False
                break
            # TODO (Liam) Check if [i,j] are part of skill segment (remove)
            # Check 2: do not touch any part of a skill trajectory  
            # for k in range(i, j+1):
            #     if mode_contains_skill_for_robot(env, new_path[k].mode, r):
            #         can_shortcut_this = False
            #         break
            if mode_contains_skill_for_robot(env, new_path[i].mode, r): # Either whole segment is or not a skill
                can_shortcut_this = False
            if not can_shortcut_this:
                # TODO (Liam) Stop checking further robots? (if multiple robots in list)
                # Because we are doing joint shortcut (simultaneously for every robot in list)
                break 
        if not can_shortcut_this:
            continue # Skip to next [i,j] if can't shortcut

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
                dim = env.robot_dims[env.robots[r]]
                if r in robots_to_shortcut:
                    # we assume that we double the mode switch configurations
                    if k != 0 and i+k != j and new_path[i+k].mode != new_path[i+k-1].mode:
                        q_interp = q0_tmp[r] + diff_tmp[r] * (k-1)
                    else:
                        q_interp = q0_tmp[r] + diff_tmp[r] * k
                    q[r_cnt : r_cnt + dim] = q_interp

                r_cnt += dim

            path_element.append(
                State(q0.from_flat(q), new_path[i + k].mode)
            )

        # check if the shortcut improves cost
        if path_cost(path_element, env.batch_config_cost) >= path_cost(
            new_path[i : j + 1], env.batch_config_cost
        ):
            continue

        assert np.linalg.norm(path_element[0].q.state() - q0.state()) < 1e-6
        assert np.linalg.norm(path_element[-1].q.state() - q1.state()) < 1e-6

        cnt += 1

        if env.is_path_collision_free(
            path_element, resolution=resolution, tolerance=tolerance, check_start_and_end=False
        ):
            for k in range(j - i + 1):
                new_path[i + k].q = path_element[k].q

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

        # Preserve skill waypoints
        is_skill = B.is_skill_waypoint

        # Check if AB is collinear to AC (AB = lambda * AC)
        if np.linalg.norm(AB - lam * AC) > tolerance or A.mode != C.mode or is_skill: 
            simplified_path.append(B)

    simplified_path.append(path[-1])

    return simplified_path
