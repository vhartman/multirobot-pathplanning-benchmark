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
    working_path = interpolate_path(non_redundant_path, interpolation_resolution)
    
    costs = [path_cost(working_path, env.batch_config_cost)]
    times = [0.0]
    start_time = time.time()

    attempted_shortcuts = 0
    max_attempts = 250 * 10
    iter_count = 0
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
        iter_count += 1
        if attempted_shortcuts >= max_iter or iter_count >= max_attempts:
            break

        start_idx = np.random.randint(0, len(working_path))
        end_idx = np.random.randint(0, len(working_path))

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        if abs(end_idx - start_idx) < 2:
            continue

        # 1, Choose (one) robot to shortcut
        if robot_choice == "round_robin":
            robots_to_shortcut = [rr_robot % len(env.robots)]
            rr_robot += 1
        else:
            robots_to_shortcut = [np.random.randint(0, len(env.robots))]

        # 2. Check if this specific robot can be shortcutted
        can_shortcut_this = True
        for r in robots_to_shortcut:
            
            # Check 1: must be same task id at endpoints
            if working_path[start_idx].mode.task_ids[r] != working_path[end_idx].mode.task_ids[r]:
                can_shortcut_this = False
                break

            # Check 2: do not touch any part of a skill trajectory 
            for k in range(start_idx, end_idx + 1):
                if mode_contains_skill_for_robot(env, working_path[k].mode, r):
                    can_shortcut_this = False
                    break

            # if mode_contains_skill_for_robot(env, working_path[start_idx].mode, r): # Either whole segment is or not a skill
            #     can_shortcut_this = False # TODO use instead of check 2 cause after check 1 we can be sure whole segment is fully or not in a skill mode
            
            if not can_shortcut_this:
                # TODO (Liam) Stop checking further robots? (if multiple robots in list)
                # Because we are doing joint shortcut (simultaneously for every robot in list)
                break 

        if not can_shortcut_this:
            continue # Skip to next [i,j] if can't shortcut

        start_q = working_path[start_idx].q
        end_q = working_path[end_idx].q

        # Precompute all the differences for the active robots
        start_q_subset = {}
        end_q_subset = {}
        q_step_size = {}
        for r in robots_to_shortcut:
            start_q_subset[r] = start_q[r] * 1
            end_q_subset[r] = end_q[r] * 1
            q_step_size[r] = (end_q_subset[r] - start_q_subset[r]) / (end_idx - start_idx)

        # Construct the proposed shortcut segment
        proposed_shortcut = []
        for k in range(end_idx - start_idx + 1):
            q_flat = working_path[start_idx + k].q.state() * 1.0

            r_cnt = 0
            for r in range(len(env.robots)):
                dim = env.robot_dims[env.robots[r]]
                if r in robots_to_shortcut:
                    # we assume that we double the mode switch configurations
                    if k != 0 and start_idx + k != end_idx and working_path[start_idx + k].mode != working_path[start_idx + k - 1].mode:
                        q_interp = start_q_subset[r] + q_step_size[r] * (k - 1)
                    else:
                        q_interp = start_q_subset[r] + q_step_size[r] * k
                    q_flat[r_cnt : r_cnt + dim] = q_interp

                r_cnt += dim

            proposed_shortcut.append(
                State(start_q.from_flat(q_flat), working_path[start_idx + k].mode)
            )

        current_segment = working_path[start_idx : end_idx + 1]

        # # check if the shortcut improves cost
        # if path_cost(path_element, env.batch_config_cost) >= path_cost(
        #     current_segment, env.batch_config_cost
        # ):
        #     continue

        # TODO (Liam) check my reasoning (max cost can be before and after 10, e.g., dominated by robot A, but robot B could improve cost from 5 -> 3, but with ">=", that improvement would be discarded..)
        if path_cost(proposed_shortcut, env.batch_config_cost) > path_cost( 
            current_segment, env.batch_config_cost
        ) + 1e-8:
            continue

        assert np.linalg.norm(proposed_shortcut[0].q.state() - start_q.state()) < 1e-6
        assert np.linalg.norm(proposed_shortcut[-1].q.state() - end_q.state()) < 1e-6

        attempted_shortcuts += 1

        if env.is_path_collision_free(
            proposed_shortcut, resolution=resolution, tolerance=tolerance, check_start_and_end=False
        ):
            # Apply the successful shortcut to the working trajectory
            for k in range(end_idx - start_idx + 1):
                working_path[start_idx + k].q = proposed_shortcut[k].q

        current_time = time.time()
        times.append(current_time - start_time)
        costs.append(path_cost(working_path, env.batch_config_cost))

    # TODO (Liam)
    # Restore skill waypoint flags lost by interpolate_path
    for s in working_path:
        for task_id in s.mode.task_ids:
            if getattr(env.tasks[task_id], 'skill', None) is not None:
                s.is_skill_waypoint = True
                break

    assert working_path[-1].mode == path[-1].mode
    assert np.linalg.norm(working_path[-1].q.state() - path[-1].q.state()) < 1e-6
    assert np.linalg.norm(working_path[0].q.state() - path[0].q.state()) < 1e-6

    print("Original cost:", path_cost(path, env.batch_config_cost))
    print("Attempted shortcuts", attempted_shortcuts)
    print("New cost:", path_cost(working_path, env.batch_config_cost))

    return working_path, [costs, times]


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
