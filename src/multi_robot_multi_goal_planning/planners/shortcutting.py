import heapq
import time
import random

import numpy as np

from typing import List, Optional

from multi_robot_multi_goal_planning.problems.planning_env import State, BaseProblem

from multi_robot_multi_goal_planning.problems.core.configuration import config_dist
from .shortcutting_compatible import CompatibleSubsetProposer
from .shortcutting_affine import AffineIndexGroups, CoordinateProposals, shortcut_atoms
from multi_robot_multi_goal_planning.problems.constraint_set import ConstraintProvider
from multi_robot_multi_goal_planning.problems.util import interpolate_path, path_cost


def _path_constraint_provider(env, path):
    provider = ConstraintProvider(env)
    required = provider.required_types()
    if not required:
        return None
    if required - {"affine_equality", "affine_inequality"}:
        raise ValueError("Shortcutting supports only affine path constraints")
    if not provider.path_satisfied(path):
        raise ValueError("Input path violates affine constraints or bounds")
    return provider


def single_mode_shortcut(env: BaseProblem, path: List[State], max_iter: int = 1000):
    """
    Shortcutting the composite path a single mode at a time.
    I.e. we never shortcut over mode transitions, even if it would be possible.

    Works by randomly sampling indices of the path, and attempting to do a shortcut if it is in the same mode.
    """
    constraints = _path_constraint_provider(env, path)
    new_path = interpolate_path(path, 0.05)

    costs = [path_cost(new_path, env.cost)]
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

        candidate = _shortcut_candidate(env, new_path, i, j, range(len(env.robots)))
        # Evaluate the representation that will actually be retained.
        if path_cost(candidate, env.cost) >= path_cost(
            new_path[i:j + 1], env.cost
        ):
            continue

        cnt += 1

        robots_to_shortcut = [r for r in range(len(env.robots))]
        if False:
            random.shuffle(robots_to_shortcut)
            num_robots = np.random.randint(0, len(robots_to_shortcut))
            robots_to_shortcut = robots_to_shortcut[:num_robots]

        # this is wrong for partial shortcuts atm.
        if constraints is not None and not constraints.for_mode(mode).edge_satisfied(q0, q1):
            continue
        if env.is_edge_collision_free(
            q0,
            q1,
            mode,
            resolution=env.collision_resolution,
            tolerance=env.collision_tolerance,
        ):
            new_path[i:j + 1] = candidate

        current_time = time.time()
        times.append(current_time - start_time)
        costs.append(path_cost(new_path, env.cost))

    return new_path, [costs, times]


def constant_task_runs(path: List[State], robot: int, min_span: int = 2):
    """
    The index ranges over which the task of `robot` does not change.

    Returns maximal contiguous runs as inclusive (first, last) index pairs.
    A run must span at least min_span edges. Equal task IDs separated by a
    different task belong to separate runs.
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


def _shortcut_candidate(env, path, first, last, robots=(), *, indices=None):
    """Straighten selected coordinates without moving mode-switch duplicates."""
    start, end = path[first].q, path[last].q
    if indices is None:
        indices = [idx for r in robots for idx in env.robot_idx[env.robots[r]]]
    indices = np.asarray(indices, dtype=int)
    # A mode switch is not a motion step, including at either interval end.
    progress = np.r_[0, np.cumsum([
        path[k].mode == path[k - 1].mode for k in range(first + 1, last + 1)
    ])]
    selected_start = start.state()[indices]
    selected_end = end.state()[indices]
    delta = selected_end - selected_start
    candidate = []
    for offset, step in enumerate(progress):
        flat = path[first + offset].q.state().copy()
        if progress[-1]:
            if step == 0:
                flat[indices] = selected_start
            elif step == progress[-1]:
                flat[indices] = selected_end
            else:
                flat[indices] = selected_start + delta * (step / progress[-1])
        candidate.append(State(start.from_flat(flat), path[first + offset].mode))
    return candidate


def full_run_gain(env: BaseProblem, path: List[State], robot: int, lo: int, hi: int,
                  *, indices=None) -> float:
    """Composite-cost decrease from straightening a robot or selected coordinates.

    This evaluates the candidate's cost without collision checking. The result
    can be negative, even when that robot's individual path becomes shorter.
    """
    candidate = _shortcut_candidate(env, path, lo, hi, (robot,), indices=indices)
    return (path_cost(path[lo:hi + 1], env.cost)
            - path_cost(candidate, env.cost))


def _check_shortcut(env, candidate, resolution, tolerance, observe_failure, ordered=False):
    """Check a candidate and optionally locate the first reported failure.

    The failure offset identifies a failing state or the start of a failing
    edge in the candidate. It is None if the checker cannot identify one.
    Observation wrappers are removed even when collision checking raises.
    """
    failing_q = [None]
    originals = {}
    missing = object()
    if observe_failure:
        for name in ("is_edge_collision_free", "is_collision_free"):
            original = getattr(env, name)
            originals[name] = env.__dict__.get(name, missing)
            def observe(q, *args, _check=original, **kwargs):
                free = _check(q, *args, **kwargs)
                if not free:
                    failing_q[0] = q
                return free
            setattr(env, name, observe)
    try:
        options = dict(check_edges_in_order=True, binary_order=False) if ordered else {}
        free = env.is_path_collision_free(
            candidate, resolution=resolution, tolerance=tolerance,
            check_start_and_end=False, **options,
        )
    finally:
        for name, original in originals.items():
            if original is missing:
                delattr(env, name)
            else:
                setattr(env, name, original)
    failure = next((k for k, state in enumerate(candidate)
                    if state.q is failing_q[0]), None) if not free else None
    return free, failure


def _interpolate_compatible_path(env, path, spacing, resolution, tolerance):
    """Keep mode boundaries intact; retain raw edges if subdivision is invalid."""
    dense = []
    for start, end in zip(path, path[1:]):
        steps = (max(1, int(config_dist(start.q, end.q, "max") / spacing))
                 if start.mode == end.mode else 1)
        delta = (end.q.state() - start.q.state()) / steps
        for k in range(steps):
            dense.append(State(start.q.from_flat(start.q.state() + delta * k), start.mode))
    dense.append(State(path[-1].q.from_flat(path[-1].q.state().copy()), path[-1].mode))
    options = dict(resolution=resolution, tolerance=tolerance,
                   check_start_and_end=True, check_edges_in_order=True, binary_order=False)
    if env.is_path_collision_free(dense, **options):
        return dense
    if not env.is_path_collision_free(path, **options):
        raise ValueError("Input path failed collision validation at the configured resolution")
    return [State(state.q.from_flat(state.q.state().copy()), state.mode) for state in path]


def robot_mode_shortcut(
    env: BaseProblem,
    path: List[State],
    max_iter: int = 1000,
    resolution=0.001,
    tolerance=0.01,
    robot_choice = "round_robin",
    interpolation_resolution: float=0.5,
    max_attempts: Optional[int] = None,
    run_choice: str = "tree",
    *,
    deadline: Optional[float] = None,
    statistics: Optional[dict] = None,
):
    """Reduce path cost by straightening robots within constant-task runs.

    Selected robots are interpolated between two path indices; other robots
    retain their coordinates. Candidates must reduce env.batch_config_cost
    and pass collision checking at resolution/tolerance before being applied.
    Input paths must represent mode switches with duplicate configurations.

    The tree proposes entire single-robot runs, longest first, and splits
    colliding intervals at observed failure indices. Compatible-subset sampling
    chooses a subset size, robot subset, compatible run, and endpoint indices
    uniformly at each level. A compatible run contains no task change for any
    selected robot; subsets are neither enumerated nor cached.

    run_choice selects the proposal schedule:
      - tree_compatible: alternate tree and compatible-subset draws, then use
        subsets alone when the tree is exhausted.
      - tree_then_compatible: exhaust the tree, then sample compatible subsets.
      - compatible: sample compatible subsets only.
      - tree (default): exhaust the tree, then use single-robot gain scheduling.
      - atom_tree: the same tree/gain schedule over equality-coupled coordinate
        groups sharing a robot set, keeping private arm coordinates separate.
      - gain: use single-robot gain scheduling only.
      - tree_random: exhaust the tree, then use span scheduling.
      - span: use span scheduling only.
      - index_endpoints: random coordinate subset and path endpoints; close
        the subset over equality supports active within that interval.
      - index_compatible: random equality-closed coordinate subset and run.
      - index_tree: alternate full/block tree intervals and coordinate subsets.
      - index_full: alternate full configurations and coordinate subsets.
      - mechanism_compatible: sample groups of equality blocks sharing the
        same owning robots, such as a shared base or a private arm.
      - mechanism_full: alternate full configurations and mechanism groups.
      - full / full_compatible: full configurations with endpoints / runs.

    Index/full policies ignore robot_choice. Compatible index policies close
    groups over all path modes; every touched robot must keep its task. Tree
    roots are full configurations and equality-connected blocks; failed index
    tree intervals split at their midpoint. Inequality violations are rejected.
    No projection is used.

    Gain scheduling ranks runs by potential cost decrease, acceptance evidence,
    and draw count; overlapping gains are updated after accepted shortcuts.
    Span scheduling selects robots according to robot_choice (round_robin or
    random), then maximizes run span divided by one plus its draw count.

    max_iter bounds collision-checked proposals. max_attempts bounds draws,
    including rejected pairs; its default is max(2500, 5 * max_iter).
    All policies require a strictly positive cost improvement.
    deadline is an optional absolute time.monotonic() limit, checked per draw.
    statistics, if supplied, receives draw/check/acceptance counts.
    Returns the copied path and [costs, elapsed_times].
    """
    start_time = time.monotonic()
    use_atoms = run_choice == 'atom_tree'
    if use_atoms:
        run_choice = 'tree'
    index_policies = {"index_compatible": ("compatible", "index_groups"),
                      "index_tree": ("tree_group", "index_groups"),
                      "index_full": ("full_group", "index_groups"),
                      "full_compatible": ("compatible", "full"),
                      "mechanism_compatible": ("mechanism_group", "index_groups"),
                      "mechanism_full": ("full_mechanism", "index_groups")}
    coordinate_policy = run_choice in (*index_policies, "index_endpoints", "full")
    stats = dict(draws=0, compatible=0, checked=0, accepted=0, constraint=0, cost=0)
    if statistics is not None:
        statistics.update(stats)
    schedules = {"tree_compatible": "interleaved", "tree_then_compatible": "sequenced",
                 "compatible": "subset"}
    if run_choice not in ("tree", "tree_random", "gain", "span", *schedules) and not coordinate_policy:
        raise ValueError(f"Unknown run_choice: {run_choice}")
    if not path:
        return [], [[0.0], [0.0]]
    if interpolation_resolution <= 0 or not np.isfinite(interpolation_resolution):
        raise ValueError("interpolation_resolution must be finite and positive")
    constraints = _path_constraint_provider(env, path)
    scalable = run_choice in schedules
    # Keep the original edge partition. The input edges have already been
    # collision checked, while joining collinear edges changes the collision
    # checker's discretization and can turn a certified path into an invalid one.
    new_path = (_interpolate_compatible_path(
        env, path, interpolation_resolution, resolution, tolerance,
    ) if scalable or coordinate_policy else interpolate_path(path, interpolation_resolution))
    
    costs = [path_cost(new_path, env.cost)]
    times = [0.0]

    cnt = 0
    if max_attempts is None:
        max_attempts = max(250 * 10, 5 * max_iter)
    iter = 0

    rr_robot = 0

    # the ranges each robot may be shortcut within, and how often we have drawn from each of them
    atoms = atom_robots = None
    if use_atoms:
        atoms, atom_robots = shortcut_atoms(env, new_path)
        runs = []
        for owners in atom_robots:
            signatures = [tuple(s.mode.task_ids[r] for r in owners) for s in new_path]
            boundaries = [0] + [i for i in range(1, len(signatures))
                                if signatures[i] != signatures[i-1]] + [len(signatures)]
            runs.append([(a, b-1) for a, b in zip(boundaries[:-1], boundaries[1:]) if b-a >= 3])
    else:
        runs = [constant_task_runs(new_path, r) for r in range(len(env.robots))]

    def run_gain(unit, first, last):
        return full_run_gain(env, new_path, unit, first, last,
                             indices=atoms[unit] if atoms is not None else None)

    runs_used = [[0] * len(r) for r in runs]

    if not any(runs) and not coordinate_policy:
        return new_path, [costs, times]

    proposer = (CompatibleSubsetProposer(
        new_path, runs, np.random.default_rng(np.random.randint(0, 2**32)), schedules[run_choice],
    ) if scalable else None)
    coordinate_proposer = (CoordinateProposals(env, new_path, *index_policies[run_choice])
                           if run_choice in index_policies else None)
    index_groups = (AffineIndexGroups(env, new_path)
                    if run_choice == "index_endpoints" else None)
    gains = runs_prop = runs_acc = None
    tree_heap = []
    tree_serial = 0

    if run_choice == "gain":
        # remaining gain per (robot, run), and a floor so converged runs keep a small
        # exploration share instead of never being drawn again
        gains = [
            [max(0.0, run_gain(r, lo, hi)) for lo, hi in runs[r]]
            for r in range(len(runs))
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
        if (cnt >= max_iter or iter >= max_attempts
                or (deadline is not None and time.monotonic() >= deadline)):
            break
        stats['draws'] += 1
        active_indices = None
        if len(new_path) < 3:
            break

        from_tree = False
        if coordinate_policy:
            if coordinate_proposer is not None:
                proposal = coordinate_proposer()
                if proposal is None:
                    continue
                i, j, active_indices = proposal
            else:
                i, j = sorted(np.random.randint(0, len(new_path), size=2))
                if j - i < 2:
                    continue
                dimension = len(new_path[0].q.state())
                if index_groups is None:
                    active_indices = list(range(dimension))
                else:
                    size = np.random.randint(1, dimension + 1)
                    selected = np.random.choice(dimension, size=size, replace=False)
                    active_indices = index_groups.close(i, j, selected)
            selected = set(active_indices)
            robots_to_shortcut = [r for r, name in enumerate(env.robots)
                                 if selected.intersection(env.robot_idx[name])]
        elif proposer is not None:
            proposal = proposer()
            if proposal is None:
                continue
            i, j, robots_to_shortcut = proposal.first, proposal.last, proposal.robots
            from_tree = proposal.from_tree
        elif run_choice in ("tree", "tree_random") and tree_heap:
            # structural prefix: broadest interval first, robot and interval from the frontier
            from_tree = True
            _, _, _, r, i, j = heapq.heappop(tree_heap)
            robots_to_shortcut = list(atom_robots[r]) if atoms is not None else [r]
            active_indices = atoms[r] if atoms is not None else None
            tree_last = (r, i, j)
        else:
            if run_choice == "tree" and gains is None:
                # Initialize gain scheduling once the tree is exhausted.
                gains = [
                    [max(0.0, run_gain(r, lo, hi)) for lo, hi in runs[r]]
                    for r in range(len(runs))
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
                    ((rr, k) for rr in range(len(runs)) for k in range(len(runs[rr]))),
                    key=lambda arm: (
                        gains[arm[0]][arm[1]]
                        * (1.0 + runs_acc[arm[0]][arm[1]])
                        / (2.0 + runs_prop[arm[0]][arm[1]])
                        + gain_floor
                    )
                    / (runs_used[arm[0]][arm[1]] + 1),
                )
                robots_to_shortcut = list(atom_robots[r]) if atoms is not None else [r]
                active_indices = atoms[r] if atoms is not None else None
            else:
                if robot_choice == "round_robin":
                    robots_to_shortcut = [rr_robot % len(env.robots)]
                    rr_robot += 1
                else:
                    robots_to_shortcut = [np.random.randint(0, len(env.robots))]

                # Prioritize runs by span divided by one plus their draw count.
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

        # Selected robots must retain their tasks across the shortcut interval.
        can_shortcut_this = True
        for r in robots_to_shortcut:
            if any(s.mode.task_ids[r] != new_path[i].mode.task_ids[r] for s in new_path[i:j + 1]):
                can_shortcut_this = False
                break

        if not can_shortcut_this:
            continue

        stats['compatible'] += 1
        q0 = new_path[i].q
        q1 = new_path[j].q

        path_element = _shortcut_candidate(env, new_path, i, j, robots_to_shortcut,
                                          indices=active_indices)
        # check if the shortcut improves cost
        improvement = (path_cost(new_path[i:j + 1], env.cost)
                       - path_cost(path_element, env.cost))
        if improvement <= 0.0:
            stats['cost'] += 1
            if coordinate_proposer is not None:
                coordinate_proposer.feedback('cost')
            continue

        if constraints is not None and not constraints.path_satisfied(path_element):
            stats['constraint'] += 1
            if coordinate_proposer is not None:
                coordinate_proposer.feedback('constraint')
            continue

        assert np.linalg.norm(path_element[0].q.state() - q0.state()) < 1e-6
        assert np.linalg.norm(path_element[-1].q.state() - q1.state()) < 1e-6

        cnt += 1
        stats["checked"] += 1

        free, fail_offset = _check_shortcut(
            env, path_element, resolution, tolerance, from_tree, ordered=scalable,
        )
        fail_k = i + fail_offset if fail_offset is not None else None

        if coordinate_proposer is not None:
            coordinate_proposer.feedback('accepted' if free else 'collision')
        if free:
            stats['accepted'] += 1
            for k in range(j - i + 1):
                new_path[i + k].q = path_element[k].q

            if gains is not None and not from_tree:
                runs_acc[sel_r][sel_run] += 1
                # the composite cost takes a max over the robots, so an accepted shortcut on one
                # robot changes the remaining gain of EVERY run that overlaps it
                for rr in range(len(runs)):
                    for k, (lo_k, hi_k) in enumerate(runs[rr]):
                        if lo_k < j and i < hi_k:
                            gains[rr][k] = max(
                                0.0, run_gain(rr, lo_k, hi_k)
                            )
        elif from_tree and proposer is not None:
            proposer.collision(proposal, fail_k)
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
        current_time = time.monotonic()
        times.append(current_time - start_time)
        costs.append(path_cost(new_path, env.cost))

    if statistics is not None:
        statistics.update(stats)
        if coordinate_proposer is not None:
            statistics['proposer'] = dict(coordinate_proposer.stats)

    assert new_path[-1].mode == path[-1].mode
    assert np.linalg.norm(new_path[-1].q.state() - path[-1].q.state()) < 1e-6
    assert np.linalg.norm(new_path[0].q.state() - path[0].q.state()) < 1e-6

    return new_path, [costs, times]


def remove_interpolated_nodes(path: List[State], tolerance=1e-15) -> List[State]:
    """
    Preserve a path's certified edge partition.

    Returns a shallow copy without joining collinear edges. A joined edge
    follows the same geometry, but changes the finite collision checker's
    sample locations and therefore requires a separate collision check.

    Args:
        path (List[Object]): Sequence of states representing original path.
        tolerance (float, optional): Unused; no nodes are removed.

    Returns:
        List[Object]: A shallow copy of the input path.
    """

    # Joining collinear edges changes the finite collision-check sampling grid.
    # Without recertifying the merged edge, preserving all checked edges is the
    # only correctness-preserving behavior.
    return list(path)
