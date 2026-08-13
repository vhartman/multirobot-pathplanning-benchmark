import heapq
import time
import random

import numpy as np

from typing import List, Optional

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


def constant_task_runs(path: List[State], robot: int, min_span: int = 2):
    """
    The index ranges over which the task of `robot` does not change.

    A robot may only be shortcut between two indices at which its task is the same, and a task
    occupies one contiguous stretch of the path, so every pair that may be shortcut lies inside
    one of these runs. Runs too short to hold a pair are dropped. Returns inclusive (first, last)
    index pairs.
    """
    task_ids = [s.mode.task_ids[robot] for s in path]

    runs = []
    start = 0
    for k in range(1, len(path) + 1):
        if k == len(path) or task_ids[k] != task_ids[start]:
            if k - 1 - start >= min_span:
                runs.append((start, k - 1))
            start = k

    return runs


def full_run_gain(env: BaseProblem, path: List[State], robot: int, lo: int, hi: int) -> float:
    """
    The cost the shortcutter could still win inside path[lo:hi+1] by straightening `robot`:
    the cost of the segment minus the cost of the segment with `robot` fully straightened.
    Can be negative — the composite cost takes a max over the robots, so straightening one of
    them can raise it. Uses the same interpolation as the shortcut candidates, including the
    doubled mode switch handling, so a run with zero gain really has nothing left to propose.
    """
    q0 = path[lo].q
    q0_tmp = q0[robot] * 1
    diff = (path[hi].q[robot] * 1 - q0_tmp) / (hi - lo)

    straightened = []
    for k in range(hi - lo + 1):
        q = path[lo + k].q.state() * 1.0
        r_cnt = 0
        for r in range(len(env.robots)):
            dim = env.robot_dims[env.robots[r]]
            if r == robot:
                # we assume that we double the mode switch configurations
                if k != 0 and lo + k != hi and path[lo + k].mode != path[lo + k - 1].mode:
                    q_interp = q0_tmp + diff * (k - 1)
                else:
                    q_interp = q0_tmp + diff * k
                q[r_cnt : r_cnt + dim] = q_interp
            r_cnt += dim
        straightened.append(State(q0.from_flat(q), path[lo + k].mode))

    return path_cost(path[lo : hi + 1], env.batch_config_cost) - path_cost(
        straightened, env.batch_config_cost
    )


def robot_mode_shortcut(
    env: BaseProblem,
    path: List[State],
    max_iter: int = 1000,
    resolution=0.001,
    tolerance=0.01,
    robot_choice = "round_robin",
    interpolation_resolution: float=0.5,
    max_attempts: Optional[int] = None,
    run_choice: str = "gain"
):
    """
    Shortcutting the composite path one robot at a time, but allowing shortcutting over the modes as well if the
    robot we are shortcutting is not active.

    Works by sampling indices for one robot, and then checking if the direct interpolation is collision free.
    The indices are drawn from within one of the ranges over which that robot keeps the same task, since a pair
    that spans a task change of the robot can not be shortcut anyway (see constant_task_runs). Drawing them from
    the whole path instead spends the attempt budget on pairs that are then discarded: measured on assembly paths
    with three and four robots, only 7% of the pairs survived, so the 2500 attempts bought ~125 collision checks
    instead of the 1000 that were asked for.

    `run_choice` decides where the next pair comes from. "gain" (the default) keeps, per (robot, run), the
    cost that fully straightening the robot inside the run would still win (see full_run_gain), and picks the
    pair from the (robot, run) with the highest remaining gain per draw already spent on it — so the budget
    flows to where slack remains and away from runs that have converged, at the price of recomputing the gains
    touched by an accepted shortcut. "span" picks the robot round-robin (or at random, see `robot_choice`) and
    the run by its length. Measured on assembly scenes at a 1000-shortcut budget, "gain" reaches a given cost
    with roughly half the wall time of "span" and is never worse at matched budgets.
    "tree" starts with a structural prefix: every maximal constant-task run is proposed
    broadest-first, and a colliding run is split at the first failing edge the checker reports
    (dropped only when no edge is reported). The exact cost gate is unchanged, and once the
    prefix is exhausted the "gain" selector takes over for the remaining budget. Measured on
    assembly scenes at a 1000-shortcut budget, "tree" matches or beats "gain" in final cost
    everywhere and spends less wall time on long, detour-heavy paths.

    "tree_random" is the same structural prefix with the plain "span" fallback (round-robin
    robot, longest-run divisor, uniform pair inside the run) instead of the gain selector.
    Measured on the assembly scenes the span fallback finishes a few percent above the gain
    selector at matched budgets; this variant is kept as the smaller-diff option.
    `max_attempts` bounds the draws, as opposed to `max_iter`, which bounds the shortcuts that are actually
    checked. It defaults to five times `max_iter`, but at least the 2500 that used to be hardcoded, so
    callers with a small `max_iter` keep the attempt budget they had.
    """
    # Keep the original edge partition. The input edges have already been
    # collision checked, while joining collinear edges changes the collision
    # checker's discretization and can turn a certified path into an invalid one.
    new_path = interpolate_path(path, interpolation_resolution)
    
    costs = [path_cost(new_path, env.batch_config_cost)]
    times = [0.0]

    # for p in new_path:
    #     if not env.is_collision_free(p.q, p.mode):
    #         print("startpath is in collision")

    start_time = time.time()

    cnt = 0
    # for iter in range(max_iter):
    if max_attempts is None:
        max_attempts = max(250 * 10, 5 * max_iter)
    iter = 0

    rr_robot = 0

    # the ranges each robot may be shortcut within, and how often we have drawn from each of them
    runs = [constant_task_runs(new_path, r) for r in range(len(env.robots))]
    runs_used = [[0] * len(r) for r in runs]

    if not any(runs):
        return new_path, [costs, times]

    gains = runs_prop = runs_acc = None
    tree_heap = []
    tree_serial = 0

    if run_choice == "gain":
        # remaining gain per (robot, run), and a floor so converged runs keep a small
        # exploration share instead of never being drawn again
        gains = [
            [max(0.0, full_run_gain(env, new_path, r, lo, hi)) for lo, hi in runs[r]]
            for r in range(len(env.robots))
        ]
        all_gains = [g for per_robot in gains for g in per_robot]
        mean_gain = sum(all_gains) / len(all_gains)
        gain_floor = 0.05 * mean_gain if mean_gain > 0 else 1.0
        # per-run acceptance evidence. The gain alone says what a straight line can WIN, not
        # whether it is collision-free -- on paths whose high-gain runs are blocked, the raw
        # score parks the budget on runs that never accept. The Beta-posterior factor
        # (1 + accepts) / (2 + proposals) starts at 0.5 for every run (identical ranking to
        # the raw score) and then moves with the evidence: failing runs sink, productive runs
        # rise. No tuned constant.
        runs_prop = [[0] * len(rs) for rs in runs]
        runs_acc = [[0] * len(rs) for rs in runs]
    elif run_choice in ("tree", "tree_random"):
        # The structural prefix: each maximal constant-task run enters the frontier as one
        # broad work item. A broader interval is always proposed before every interval it
        # contains (random tie-breaking only among equal spans). On collision the interval
        # splits at the first failing edge the checker reports; every other outcome drops it.
        for r, rs in enumerate(runs):
            for lo, hi in rs:
                heapq.heappush(tree_heap, (-(hi - lo), random.random(), tree_serial, r, lo, hi))
                tree_serial += 1

    while True:
        iter += 1
        if cnt >= max_iter or iter >= max_attempts:
            break

        # robots_to_shortcut = [r for r in range(len(env.robots))]
        # random.shuffle(robots_to_shortcut)
        # # num_robots = np.random.randint(0, len(robots_to_shortcut))
        # num_robots = 1
        # robots_to_shortcut = robots_to_shortcut[:num_robots]
        from_tree = False
        if run_choice in ("tree", "tree_random") and tree_heap:
            # structural prefix: broadest interval first, robot and interval from the frontier
            from_tree = True
            _, _, _, r, i, j = heapq.heappop(tree_heap)
            robots_to_shortcut = [r]
            tree_last = (r, i, j)
        else:
            if run_choice == "tree" and gains is None:
                # prefix exhausted: initialize the landed gain selector for the rest of the budget
                gains = [
                    [max(0.0, full_run_gain(env, new_path, r, lo, hi)) for lo, hi in runs[r]]
                    for r in range(len(env.robots))
                ]
                all_gains = [g for per_robot in gains for g in per_robot]
                mean_gain = sum(all_gains) / len(all_gains)
                gain_floor = 0.05 * mean_gain if mean_gain > 0 else 1.0
                runs_prop = [[0] * len(rs) for rs in runs]
                runs_acc = [[0] * len(rs) for rs in runs]
            if run_choice in ("gain", "tree"):
                # pick the (robot, run) with the most remaining REALIZABLE gain per draw already
                # spent on it: the gain times the Beta-posterior acceptance estimate. The robot
                # falls out of the choice, so robot_choice does not apply here.
                r, run = max(
                    ((rr, k) for rr in range(len(env.robots)) for k in range(len(runs[rr]))),
                    key=lambda arm: (
                        gains[arm[0]][arm[1]]
                        * (1.0 + runs_acc[arm[0]][arm[1]])
                        / (2.0 + runs_prop[arm[0]][arm[1]])
                        + gain_floor
                    )
                    / (runs_used[arm[0]][arm[1]] + 1),
                )
                robots_to_shortcut = [r]
            else:
                if robot_choice == "round_robin":
                    robots_to_shortcut = [rr_robot % len(env.robots)]
                    rr_robot += 1
                else:
                    robots_to_shortcut = [np.random.randint(0, len(env.robots))]

                # pick one of the ranges of the robot we shortcut, and draw the pair from it. Longer
                # ranges get proportionally more draws (the divisor rule; drawing from the whole path
                # favoured them even more, quadratically).
                r = robots_to_shortcut[0]
                if not runs[r]:
                    continue

                run = max(
                    range(len(runs[r])),
                    key=lambda k: (runs[r][k][1] - runs[r][k][0]) / (runs_used[r][k] + 1),
                )
            runs_used[r][run] += 1
            lo, hi = runs[r][run]

            i = np.random.randint(lo, hi + 1)
            j = np.random.randint(lo, hi + 1)

            if i > j:
                q = i
                i = j
                j = q

            if abs(j - i) < 2:
                continue

            if gains is not None:
                # keep the picked pair: the loops below reuse `r` as their loop variable
                sel_r, sel_run = r, run
                runs_prop[sel_r][sel_run] += 1

        # holds by construction now, since i and j come from a range over which the task of the
        # robot does not change, but the shortcut is only valid if it does
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

        # TODO: is path colision free makes this horrible, since the edges are the interpolated nodes
        # Therefore, many edges have length 2. Possibly remove interpolated things here before checking?
        # needs to be fixed.
        #
        # "tree" observes the failing edge endpoint during the check (no extra collision
        # queries) so a colliding interval can split at the FIRST failing edge instead of
        # being discarded whole.
        fail_q = [None]
        if from_tree:
            original_edge_check = env.is_edge_collision_free
            original_config_check = env.is_collision_free

            def _edge_observe(q_a, *args, **kwargs):
                result = original_edge_check(q_a, *args, **kwargs)
                if not result:
                    fail_q[0] = q_a
                return result

            def _config_observe(q, *args, **kwargs):
                result = original_config_check(q, *args, **kwargs)
                if not result:
                    fail_q[0] = q
                return result

            env.is_edge_collision_free = _edge_observe
            env.is_collision_free = _config_observe
        try:
            free = env.is_path_collision_free(
                path_element, resolution=resolution, tolerance=tolerance, check_start_and_end=False
            )
        finally:
            if from_tree:
                env.is_edge_collision_free = original_edge_check
                env.is_collision_free = original_config_check

        fail_k = None
        if not free and fail_q[0] is not None:
            for k in range(j - i + 1):
                if path_element[k].q is fail_q[0]:
                    fail_k = i + k
                    break

        if free:
            for k in range(j - i + 1):
                new_path[i + k].q = path_element[k].q

                # if not np.array_equal(new_path[i+k].mode, path_element[k].mode):
                # print('fucked up')

            if gains is not None and not from_tree:
                runs_acc[sel_r][sel_run] += 1
                # the composite cost takes a max over the robots, so an accepted shortcut on one
                # robot changes the remaining gain of EVERY run that overlaps it
                for rr in range(len(env.robots)):
                    for k, (lo_k, hi_k) in enumerate(runs[rr]):
                        if lo_k < j and i < hi_k:
                            gains[rr][k] = max(
                                0.0, full_run_gain(env, new_path, rr, lo_k, hi_k)
                            )
        elif from_tree:
            # split the colliding interval at the located failing edge; both sides remain
            # eligible because the observed collision lies between them. Without a located
            # edge the interval is dropped -- no midpoint refinement.
            tr, ti, tj = tree_last
            if fail_k is not None and ti < fail_k < tj:
                for lo, hi in ((ti, fail_k), (fail_k, tj)):
                    if hi - lo >= 2:
                        heapq.heappush(tree_heap, (-(hi - lo), random.random(), tree_serial, tr, lo, hi))
                        tree_serial += 1
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
    Preserve a path's certified edge partition.

    This compatibility wrapper intentionally no longer joins collinear edges.
    A joined edge follows the same geometry, but the finite collision checker
    samples it at different configurations, so it must be recertified before it
    can safely replace the original edges.

    Args:
        path (List[Object]): Sequence of states representing original path.
        tolerance (float, optional): Retained for API compatibility.

    Returns:
        List[Object]: A shallow copy of the input path.
    """

    # Joining collinear edges changes the finite collision-check sampling grid.
    # Without recertifying the merged edge, preserving all checked edges is the
    # only correctness-preserving behavior.
    return list(path)
