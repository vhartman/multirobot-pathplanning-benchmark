import numpy as np

import time
import random

from typing import List, Optional

from multi_robot_multi_goal_planning.planners.baseplanner import BasePlanner
from multi_robot_multi_goal_planning.problems.constraints import AffineConfigurationSpaceEqualityConstraint, AffineConfigurationSpaceInequalityConstraint
from multi_robot_multi_goal_planning.problems.constraints_projection import project_affine_cspace_interior, project_gauss_newton
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


def robot_mode_shortcut_no_constr(
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
                    # we assume that we double the mode switch configurations
                    if k != 0 and i+k != j and new_path[i+k].mode != new_path[i+k-1].mode:
                        q_interp = q0_tmp[r] + diff_tmp[r] * (k-1)
                    else:
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

        # check if the shortcut improves cost
        if path_cost(path_element, env.batch_config_cost) >= path_cost(
            new_path[i : j + 1], env.batch_config_cost
        ):
            # print(f"{cnt} does not improve cost")
            continue

        assert np.linalg.norm(path_element[0].q.state() - q0.state()) < 1e-6
        assert np.linalg.norm(path_element[-1].q.state() - q1.state()) < 1e-6

        cnt += 1

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


def robot_mode_shortcut(
    env: BaseProblem,
    path: List[State],
    max_iter: int = 1000,
    resolution=0.001,
    tolerance=0.01,
    robot_choice="round_robin",
    interpolation_resolution: float = 0.1,
):
    """
    Shortcutting the composite path one robot at a time, but allowing shortcutting
    over the modes as well if the robot we are shortcutting is not active.
    Works by randomly sampling indices, then randomly choosing a robot or constraint,
    and then checking if the direct interpolation is collision free.
    """
    non_redundant_path = remove_interpolated_nodes(path)
    new_path = interpolate_path(non_redundant_path, interpolation_resolution)
    costs = [path_cost(new_path, env.batch_config_cost)]
    times = [0.0]

    start_time = time.time()
    cnt = 0
    max_attempts = 250 * 10
    iter = 0
    rr = 0
    shortcutting_robot = False
    shortcutting_constr = False

    totsc = 0
    skipsc = 0

    # Collect affine constraints
    def collect_affine_constraints_for_env():
        eq = [c for c in env.constraints if isinstance(c, AffineConfigurationSpaceEqualityConstraint)]
        ineq = [c for c in env.constraints if isinstance(c, AffineConfigurationSpaceInequalityConstraint)]
        return eq, ineq

    env_eq_constraints, env_ineq_constraints = collect_affine_constraints_for_env()
    constraints = env_eq_constraints + env_ineq_constraints
    n_constr = len(constraints)
    # Break down constraints in rows
    c_row = []
    for c in constraints:
        A = np.asarray(c.mat)
        for row_idx in range(A.shape[0]):
            c_row.append(A[row_idx, :])
    n_of_rows = len(c_row)

    # Build a boolean mask of free DOFs (True = free)
    total_dofs = len(new_path[0].q.state())
    constrained = np.zeros(total_dofs, dtype=bool)
    for c in env_eq_constraints + env_ineq_constraints:
        A = c.mat
        constrained |= np.any(np.abs(A) > 1e-12, axis=0)
    free_mask = ~constrained

    # Main loop
    while True:
        totsc += 1
        iter += 1
        if cnt >= max_iter or iter >= max_attempts:
            break

        i = np.random.randint(0, len(new_path))
        j = np.random.randint(0, len(new_path))
        if i > j:
            i, j = j, i
        if abs(j - i) < 2:
            continue

        if robot_choice == "round_robin":
            dofs_to_shortcut = [rr % (len(env.robots) + n_of_rows)]
            if dofs_to_shortcut[0] < len(env.robots):
                # shortcutting on free DOFs of a robot
                shortcutting_robot = True
                shortcutting_constr = False
                robots_to_shortcut = [dofs_to_shortcut[0]]
            else:
                # shortcutting on constrained DOFs
                shortcutting_robot = False
                shortcutting_constr = True
                row_to_shortcut = [dofs_to_shortcut[0] - len(env.robots)]
            rr += 1
        else:
            robots_to_shortcut = [np.random.randint(0, len(env.robots))]
            shortcutting_robot = True
            shortcutting_constr = False

        path_element = None
        can_shortcut_this = True

        if shortcutting_robot:
            for r in robots_to_shortcut:
                if new_path[i].mode.task_ids[r] != new_path[j].mode.task_ids[r]:
                    can_shortcut_this = False
                    break
            if not can_shortcut_this:
                continue

            q0 = new_path[i].q
            q1 = new_path[j].q

            # Precompute differences
            q0_tmp, q1_tmp, diff_tmp = {}, {}, {}
            for r in robots_to_shortcut:
                q0_tmp[r] = q0[r] * 1
                q1_tmp[r] = q1[r] * 1
                diff_tmp[r] = (q1_tmp[r] - q0_tmp[r]) / (j - i)

            # Construct candidate shortcut
            path_element = []
            for k in range(j - i + 1):
                q = new_path[i + k].q.state().copy()
                r_cnt = 0
                for r in range(len(env.robots)):
                    dim = env.robot_dims[env.robots[r]]
                    sl = slice(r_cnt, r_cnt + dim)
                    if r in robots_to_shortcut:
                        free_sl = free_mask[sl]
                        if np.any(free_sl):
                            q_interp = q0_tmp[r] + diff_tmp[r] * k
                            q_slice = q[sl].copy()
                            q_slice[free_sl] = q_interp[free_sl]
                            q[sl] = q_slice
                    r_cnt += dim
                path_element.append(State(q0.from_flat(q), new_path[i + k].mode))

        elif shortcutting_constr:
            robot_slices = []
            offset = 0
            for r in env.robots:
                dim = env.robot_dims[r]
                robot_slices.append(slice(offset, offset + dim))
                offset += dim

            q0 = new_path[i].q
            q1 = new_path[j].q
            q0_vec = np.asarray(q0.state(), dtype=float)
            q1_vec = np.asarray(q1.state(), dtype=float)
            steps = j - i
            if steps <= 0:
                continue

            diff = (q1_vec - q0_vec) / float(steps)

            constr_row = c_row[row_to_shortcut[0]]
            
            # identify active DOFs in this row
            active_dofs = np.where(np.abs(constr_row) > 1e-12)[0]
            if len(active_dofs) == 0:
                continue

            # identify robots affected by these DOFs
            active_robots = []
            active_sl = []
            for r, sl in enumerate(robot_slices):
                if np.any((active_dofs >= sl.start) & (active_dofs < sl.stop)):
                    active_robots.append(r)
                    active_sl.append(robot_slices[r])
            active_idxs = np.concatenate([np.arange(sli.start, sli.stop) for sli in active_sl])

            # ensure same task for all those robots
            can_shortcut_this = True
            for r in active_robots:
                if new_path[i].mode.task_ids[r] != new_path[j].mode.task_ids[r]:
                    can_shortcut_this = False
                    break
            if not can_shortcut_this:
                continue

            # construct the candidate shortcut
            path_element = []
            for k in range(j - i + 1):
                q = new_path[i + k].q.state().astype(float).copy()

                # interpolate only the active DOFs of this row
                q_interp = q0_vec.copy()
                q_interp[active_idxs] = q0_vec[active_idxs] + diff[active_idxs] * k

                # apply interpolated values to q
                q[active_idxs] = q_interp[active_idxs]

                path_element.append(State(q0.from_flat(q), new_path[i + k].mode))

            # safety: must match segment length and endpoints
            if len(path_element) != (j - i + 1):
                print("ERROR: length mismatch")
                continue
            if not np.allclose(path_element[0].q.state(), new_path[i].q.state(), atol=1e-8):
                print("ERROR: start state mismatch")
                continue
            if not np.allclose(path_element[-1].q.state(), new_path[j].q.state(), atol=1e-8):
                print("ERROR: end state mismatch")
                continue


        if path_element is None or not can_shortcut_this:
            continue

        old_cost = path_cost(new_path[i:j + 1], env.batch_config_cost)
        new_cost = path_cost(path_element, env.batch_config_cost)
        if new_cost >= old_cost:
            continue

        constraints_ok = True
        for state in path_element:
            mode = state.mode
            # Check all active equality and inequality constraints
            for c in env_eq_constraints:
                if not c.is_fulfilled(state.q, mode, env):
                    constraints_ok = False
                    break

            for c in env_ineq_constraints:
                if not c.is_fulfilled(state.q, mode, env):
                    constraints_ok = False
                    break

            if not constraints_ok:
                break

        if not constraints_ok:
            skipsc += 1
            continue  # reject this shortcut and try another

        assert np.linalg.norm(path_element[0].q.state() - new_path[i].q.state()) < 1e-6
        assert np.linalg.norm(path_element[-1].q.state() - new_path[j].q.state()) < 1e-6

        cnt += 1
        if env.is_path_collision_free(path_element, resolution=resolution, tolerance=tolerance, check_start_and_end=False):
            for k in range(j - i + 1):
                new_path[i + k].q = path_element[k].q

        current_time = time.time()
        times.append(current_time - start_time)
        costs.append(path_cost(new_path, env.batch_config_cost))

    # Final checks
    assert new_path[-1].mode == path[-1].mode
    assert np.linalg.norm(new_path[-1].q.state() - path[-1].q.state()) < 1e-6
    assert np.linalg.norm(new_path[0].q.state() - path[0].q.state()) < 1e-6

    print("original cost:", path_cost(path, env.batch_config_cost))
    print("Attempted shortcuts:", cnt)
    print("new cost:", path_cost(new_path, env.batch_config_cost))
    print("Total shortcuts tried:", totsc)
    print("Shortcuts skipped due to constraints:", skipsc)

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


def robot_mode_shortcut_nl(
    env: BaseProblem,
    path: List[State],
    max_iter: int = 1000,
    resolution=0.001,
    tolerance=0.01,
    robot_choice = "round_robin",
    interpolation_resolution: float=0.1,
    planner: Optional[BasePlanner] = None,
):
    """
    Shortcutting the composite path one robot at a time, but allowing shortcutting over the modes as well if the
    robot we are shortcutting is not active.

    Works by randomly sampling indices, then randomly choosing a robot, and then checking if the direct interpolation is
    collision free.
    """

    if planner is None:
        non_redundant_path = remove_interpolated_nodes(path)
        new_path = interpolate_path(non_redundant_path, interpolation_resolution)
    else:
        new_path = planner.interpolate_path_nonlinear(path, interpolation_resolution)


    costs = [path_cost(new_path, env.batch_config_cost)]
    times = [0.0]

    start_time = time.time()

    cnt = 0
    # for iter in range(max_iter):
    max_attempts = 250 * 10
    iter = 0

    rr_robot = 0

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
                    # we assume that we double the mode switch configurations
                    if k != 0 and i+k != j and new_path[i+k].mode != new_path[i+k-1].mode:
                        q_interp = q0_tmp[r] + diff_tmp[r] * (k-1)
                    else:
                        q_interp = q0_tmp[r] + diff_tmp[r] * k
                    q[r_cnt : r_cnt + dim] = q_interp

                r_cnt += dim
            
            if planner is not None and k not in (0, j - i):
                eq_aff, ineq_aff, eq_nl, ineq_nl = planner.collect_constraints(new_path[k].mode)
                eq = eq_aff + eq_nl
                ineq = ineq_aff + ineq_nl
                q_proj = planner.project_nonlinear_dispatch(
                    q0.from_flat(q),
                    eq,
                    ineq,
                    new_path[k].mode,
                )

                path_element.append(
                    State(q_proj, new_path[i + k].mode)
                )
            else:
                path_element.append(State(q0.from_flat(q), new_path[i + k].mode))

        # check if the shortcut improves cost
        if path_cost(path_element, env.batch_config_cost) >= path_cost(
            new_path[i : j + 1], env.batch_config_cost
        ):
            # print(f"{cnt} does not improve cost")
            continue

        assert np.linalg.norm(path_element[0].q.state() - q0.state()) < 1e-6
        assert np.linalg.norm(path_element[-1].q.state() - q1.state()) < 1e-6

        cnt += 1

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


def _opt_shortcut_segment_nl(
    new_path: List[State],
    i: int,
    j: int,
    robots_to_shortcut: List[int],
    env: BaseProblem,
    planner: BasePlanner,
    max_inner_iters: int = 30,
    step_size: float = 0.2,
    tol: float = 1e-6,
) -> Optional[List[State]]:
    """
    Nonlinear constrained optimization based shortcut between new_path[i] and new_path[j].

    - Keeps exactly the same number of nodes (j - i + 1).
    - Endpoints are fixed.
    - Inner nodes are optimized to reduce squared path length.
    - After each gradient step, inner nodes are projected onto the constraint manifold
      via planner.project_nonlinear_dispatch.

    Returns:
        A list of State with length (j - i + 1) if optimization succeeded,
        or None on numerical failure.
    """
    assert planner is not None, "Nonlinear optimization based shortcutting requires a planner with projection."

    segment_len = j - i + 1
    if segment_len < 3:
        # Nothing to optimize, need at least one inner node
        return None

    # Flatten robot DOFs structure
    total_dofs = len(new_path[i].q.state())
    robot_slices = []
    offset = 0
    for r_name in env.robots:
        dim = env.robot_dims[r_name]
        robot_slices.append(slice(offset, offset + dim))
        offset += dim
    assert offset == total_dofs

    # Active DOFs: all DOFs of the selected robots
    active_slices = [robot_slices[r] for r in robots_to_shortcut]
    active_idx = np.concatenate([np.arange(sli.start, sli.stop) for sli in active_slices])

    # Extract the segment states into an array Q of shape (segment_len, total_dofs)
    Q = np.zeros((segment_len, total_dofs), dtype=float)
    modes = []
    for k in range(segment_len):
        state = new_path[i + k]
        Q[k, :] = np.asarray(state.q.state(), dtype=float).copy()
        modes.append(state.mode)

    # Template configuration object to rebuild configs from flat vectors
    q_template = new_path[i].q

    # Make sure endpoints are exactly as original (they should be already)
    Q[0, :] = new_path[i].q.state()
    Q[-1, :] = new_path[j].q.state()

    # Simple elastic-band style smoothing with projection onto constraints
    for it in range(max_inner_iters):
        # Discrete Laplacian gradient for path length: sum ||q_{k+1} - q_k||^2
        grad = np.zeros_like(Q)

        for k in range(1, segment_len - 1):
            # Gradient only on active DOFs
            grad[k, active_idx] = 2.0 * (
                2.0 * Q[k, active_idx]
                - Q[k - 1, active_idx]
                - Q[k + 1, active_idx]
            )

        # Compute max step for convergence check
        max_step = np.max(np.abs(step_size * grad[1:-1, :][:, active_idx]))
        if max_step < tol:
            break

        # Gradient descent step on inner nodes (endpoints fixed)
        Q[1:-1, active_idx] -= step_size * grad[1:-1, active_idx]

        # Project inner nodes back onto the manifold
        for k in range(1, segment_len - 1):
            mode_k = modes[k]
            q_flat = Q[k, :]

            # Collect constraints for this mode
            eq_aff, ineq_aff, eq_nl, ineq_nl = planner.collect_constraints(mode_k)
            eq = eq_aff + eq_nl
            ineq = ineq_aff + ineq_nl

            # Projection; q_proj is a configuration object
            q_proj = planner.project_nonlinear_dispatch(
                q_template.from_flat(q_flat),
                eq,
                ineq,
                mode_k,
            )

            if q_proj is None:
                # Projection failed; bail out on this shortcut
                return None

            Q[k, :] = np.asarray(q_proj.state(), dtype=float)

    # Rebuild the path segment as State objects
    path_element: List[State] = []
    for k in range(segment_len):
        q_cfg = q_template.from_flat(Q[k, :])
        path_element.append(State(q_cfg, modes[k]))

    # Sanity: endpoints must match original segment to numerical tolerance
    if not np.allclose(path_element[0].q.state(), new_path[i].q.state(), atol=1e-6):
        # If this happens, you screwed up boundary handling
        return None
    if not np.allclose(path_element[-1].q.state(), new_path[j].q.state(), atol=1e-6):
        return None

    return path_element


def robot_mode_shortcut_nl_opt(
    env: BaseProblem,
    path: List[State],
    max_iter: int = 1000,
    resolution: float = 0.001,
    tolerance: float = 0.01,
    robot_choice: str = "round_robin",
    interpolation_resolution: float = 0.1,
    planner: Optional[BasePlanner] = None,
    inner_opt_iters: int = 30,
    inner_step_size: float = 0.2,
):
    """
    Nonlinear constraint-aware shortcutting using a local optimizator.

    - For each candidate pair (i, j), we:
        1. Check that the selected robots have the same task_ids at i and j.
        2. Run a small trajectory optimization over the segment new_path[i:j+1],
           keeping the same number of nodes (no re-discretization).
        3. Optimization minimizes squared path length over the DOFs of the
           selected robots, while projecting inner nodes back to the nonlinear
           constraint manifold via planner.project_nonlinear_dispatch.
        4. If the optimized segment has lower cost and is collision free,
           we accept it and overwrite new_path[i:j+1].

    - The number of nodes in each segment is preserved (j - i + 1).
    """

    assert planner is not None, "robot_mode_shortcut_nl_opt requires a planner with nonlinear projection."

    # Initial path densification
    if planner is None:
        non_redundant_path = remove_interpolated_nodes(path)
        new_path = interpolate_path(non_redundant_path, interpolation_resolution)
    else:
        new_path = planner.interpolate_path_nonlinear(path, interpolation_resolution)

    costs = [path_cost(new_path, env.batch_config_cost)]
    times = [0.0]
    start_time = time.time()

    cnt = 0
    max_attempts = 250 * 10
    iter_attempts = 0

    rr_robot = 0

    while True:
        iter_attempts += 1
        if cnt >= max_iter or iter_attempts >= max_attempts:
            break

        # Randomly sample a pair of indices
        i = np.random.randint(0, len(new_path))
        j = np.random.randint(0, len(new_path))

        if i > j:
            i, j = j, i

        if abs(j - i) < 2:
            # Need at least one inner node to optimize
            continue

        # Choose which robot(s) we try to shortcut
        if robot_choice == "round_robin":
            robots_to_shortcut = [rr_robot % len(env.robots)]
            rr_robot += 1
        else:
            robots_to_shortcut = [np.random.randint(0, len(env.robots))]

        # Check tasks compatibility for selected robots
        can_shortcut_this = True
        for r in robots_to_shortcut:
            if new_path[i].mode.task_ids[r] != new_path[j].mode.task_ids[r]:
                can_shortcut_this = False
                break

        if not can_shortcut_this:
            continue

        # Original segment and its cost
        old_segment = new_path[i : j + 1]
        old_cost = path_cost(old_segment, env.batch_config_cost)

        # Run constrained geodesic shortcut on this segment
        path_element = _opt_shortcut_segment_nl(
            new_path=new_path,
            i=i,
            j=j,
            robots_to_shortcut=robots_to_shortcut,
            env=env,
            planner=planner,
            max_inner_iters=inner_opt_iters,
            step_size=inner_step_size,
        )

        if path_element is None:
            # Optimization or projection failed; skip this attempt
            continue

        # Check cost improvement for this segment
        new_seg_cost = path_cost(path_element, env.batch_config_cost)
        if new_seg_cost >= old_cost:
            continue

        # Sanity on endpoints
        q0 = new_path[i].q
        q1 = new_path[j].q
        assert np.linalg.norm(path_element[0].q.state() - q0.state()) < 1e-6
        assert np.linalg.norm(path_element[-1].q.state() - q1.state()) < 1e-6

        # Full path cost after hypothetical replacement (cheap, but global)
        # You can skip this and trust local cost if you want speed.
        new_full_path = new_path[:i] + path_element + new_path[j + 1 :]
        new_full_cost = path_cost(new_full_path, env.batch_config_cost)

        if new_full_cost >= costs[-1]:
            # Does not improve global cost; skip
            continue

        # Collision check on the candidate segment only
        if env.is_path_collision_free(
            path_element,
            resolution=resolution,
            tolerance=tolerance,
            check_start_and_end=False,
        ):
            # Accept the shortcut: overwrite the configurations
            for k in range(j - i + 1):
                new_path[i + k].q = path_element[k].q

            cnt += 1
            # Update time/cost logs
            current_time = time.time()
            times.append(current_time - start_time)
            costs.append(path_cost(new_path, env.batch_config_cost))

    # Final consistency checks
    assert new_path[-1].mode == path[-1].mode
    assert np.linalg.norm(new_path[-1].q.state() - path[-1].q.state()) < 1e-6
    assert np.linalg.norm(new_path[0].q.state() - path[0].q.state()) < 1e-6

    print("original cost:", path_cost(path, env.batch_config_cost))
    print("Accepted shortcuts:", cnt)
    print("final cost:", path_cost(new_path, env.batch_config_cost))

    return new_path, [costs, times]
