import contextlib
import io
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from multi_robot_multi_goal_planning.problems.planning_env import Mode, State
from multi_robot_multi_goal_planning.problems.util import skill_edge_seconds

from . import shortcutting
from .reactive_mdp import ReactiveMDP, ReactiveMDPConfig
from .rrt_skills import Node
from .rrt_skills_reactive import (
    ReactiveRoadmap,
    add_inactive_vertices,
    inactive_feasibility,
    sample_kernel_trajectory,
)

_MARGIN = 1e-3
_SHORTCUT_MAX_ITER = 100
_ABS_MARGIN = 1e-6
_COLD_SOLVE_DISCOUNT = 0.6

@dataclass
class PolicyShortcutConfig:
    enabled: bool = True
    n_iters: int = 12
    n_starts: int = 2
    n_rollouts: int = 2
    patience: int = 3
    do_det_modes: bool = True
    do_skill_modes: bool = True
    do_det_chains: bool = True
    chain_window: Optional[int] = 2
    modes_per_round: Optional[int] = None
    chain_heads_per_round: Optional[int] = 1
    seed: int = 0

def _copy_state(st: State, mode: Mode) -> State:
    q = np.asarray(st.q.state(), dtype=np.float64)
    return State(st.q.from_flat(q), mode)

def _edge_exists(a: Node, b: Node) -> bool:
    """
    The _add_roadmap_edge silently skips duplicates, so ask BEFORE calling it, otherwise a round that
    only re-asserts existing edges reports work it did not do, and `added` stops being a convergence
    signal
    """
    return any(nb is b for nb, _, _ in a.edges)

def _config_path_cost(env, qs: Sequence) -> float:
    """
    Cost of a configuration sequence measured EXACTLY as the roadmap measures it: the env's
    config_cost summed over consecutive pairs
    """
    return float(sum(env.config_cost(a, b) for a, b in zip(qs, qs[1:])))

def _leg_cost(env, states: Sequence[State]) -> float:
    """
    Cost of one proposal, measured so that the two sides of an accept test are comparable. max +
    w*sum is NOT partition-invariant across a turn, and the two sides arrive partitioned
    differently: `shortcut` has been through robot_mode_shortcut's interpolate_path, the policy
    chain has not
    """
    collapsed = shortcutting.remove_interpolated_nodes(list(states), tolerance=1e-5)
    if getattr(env, "cost_model", "geometric") == "time":
        total = 0.0
        for a, b in zip(collapsed, collapsed[1:]):
            seconds = skill_edge_seconds(env, a, b)
            total += seconds * env.v_ref if seconds is not None else env.config_cost(a.q, b.q)
        return total
    return _config_path_cost(env, [s.q for s in collapsed])

def _path_is_collision_free(env, qs: Sequence, mode: Mode) -> bool:
    """Validate shortcut waypoints as well as the open intervals between them"""
    if not all(env.is_collision_free(q, mode) for q in qs):
        return False
    return all(env.is_edge_collision_free(a, b, mode) for a, b in zip(qs, qs[1:]))

def _polyline_length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))

def _rollout_starts(entries: List[int], n_available: int, n_starts: int, rng) -> List[int]:
    """Where this round's rollouts begin: the mode's known entries first, then uniform samples"""
    if n_available == 0:
        return []
    entries = list(dict.fromkeys(int(v) for v in entries))

    if len(entries) > n_starts:
        entries = [int(v) for v in rng.choice(entries, size=n_starts, replace=False)]
    extra = max(0, n_starts - len(entries))
    uniform = rng.integers(0, n_available, size=extra) if extra else []
    return list(dict.fromkeys(entries + [int(v) for v in uniform]))[:n_starts]

def _rotate_starts(roadmap: ReactiveRoadmap, key, starts: List[int], cap: Optional[int]) -> List[int]:
    """Rotate a capped set of shortcut starts across rounds"""
    if cap is None or len(starts) <= cap:
        return starts
    cursor = getattr(roadmap, "_shortcut_entry_cursor", None)
    if cursor is None:
        cursor = {}
        roadmap._shortcut_entry_cursor = cursor
    i0 = cursor.get(key, 0) % len(starts)
    out = [starts[(i0 + i) % len(starts)] for i in range(cap)]
    cursor[key] = (i0 + cap) % len(starts)
    return out

def _rotating_window(roadmap: ReactiveRoadmap, attr: str, items: List,
                     cap: Optional[int]) -> List:
    """Return a persistent round-robin window over ``items``"""
    if cap is None or len(items) <= cap:
        return items
    i0 = int(getattr(roadmap, attr, 0)) % len(items)
    out = [items[(i0 + i) % len(items)] for i in range(max(1, cap))]
    setattr(roadmap, attr, (i0 + len(out)) % len(items))
    return out

def _persistent_rng(roadmap: ReactiveRoadmap, attr: str, seed: int):
    """Keep a shortcut sampling stream alive across batched policy-loop calls"""
    rng = getattr(roadmap, attr, None)
    if rng is None:
        rng = np.random.default_rng(seed)
        setattr(roadmap, attr, rng)
    return rng

def policy_chain(mdp: ReactiveMDP, mode: Mode, start: int) -> List[int]:
    """
    Node indices along pi_det from `start` until the policy leaves the mode. This is the leg
    _execute_deterministic_mode drives
    """
    pi = mdp.pi_det[mode]
    chain, seen, i = [], set(), start
    while i >= 0 and i not in seen:
        seen.add(i)
        chain.append(i)
        i = int(pi[i])
    return chain

def det_entry_nodes(env, roadmap: ReactiveRoadmap, mdp: ReactiveMDP, mode: Mode) -> List[int]:
    """Roadmap nodes for the seam configurations this mode is entered at, plus the global start"""
    configs = [np.asarray(q, dtype=np.float64) for q in roadmap.get_mode_entry_configs(mode)]
    if mode == env.get_start_mode():
        configs.append(np.asarray(env.get_start_pos().state(), dtype=np.float64))

    out: List[int] = []
    for q in configs:
        _, node = mdp.evaluate_connection_cost(q, mode)
        if node >= 0 and node not in out:
            out.append(int(node))
    return out

def _interp_resolution(planner) -> float:
    """
    The build-time pass's working-path spacing, so the offline operator samples the same geometry
    RRTSkills._shortcut does
    """
    return float(getattr(planner.config, "shortcutting_interpolation_resolution", 0.1))

def shortcut_states(env, states: List[State],
                    interpolation_resolution: float = 0.1) -> List[State]:
    """Shortcut one policy leg, with the shortcutter the rest of the codebase already uses"""
    with contextlib.redirect_stdout(io.StringIO()):
        shortcut, _ = shortcutting.robot_mode_shortcut(
            env, states,

            max_iter=_SHORTCUT_MAX_ITER,
            resolution=env.collision_resolution,
            tolerance=env.collision_tolerance,
            interpolation_resolution=interpolation_resolution,
        )
        return shortcutting.remove_interpolated_nodes(shortcut, tolerance=1e-5)

def insert_chain(planner, mode: Mode, first: Node, last: Node,
                 states: Sequence[State]) -> int:
    """Inserts an accepted leg as first -> s1 -> ... -> sk -> last"""
    env = planner.env
    qs = [first.state.q] + [s.q for s in states[1:-1]] + [last.state.q]

    if not _path_is_collision_free(env, qs, mode):
        return 0

    subtree = planner.tree.subtrees[mode]
    prev, added = first, 0
    for st in states[1:-1]:
        node = Node(_copy_state(st, mode), parent=prev)
        node.cost_to_parent = env.config_cost(prev.state.q, node.state.q)
        node.cost = prev.cost + node.cost_to_parent
        prev.children.append(node)
        subtree.add_node(node)
        planner._roadmap_add_node(node, mode, is_skill=False)
        prev, added = node, added + 1

    if prev is not last and not _edge_exists(prev, last):
        planner._add_roadmap_edge(
            prev, last, env.config_cost(prev.state.q, last.state.q), kind="geometric")
        added += 1
    return added

def improve_det_mode(roadmap: ReactiveRoadmap, mdp: ReactiveMDP, mode: Mode,
                     cfg: PolicyShortcutConfig, rng, deadline: Optional[float] = None) -> int:
    """
    One round for a single non-skill mode. deadline: optional absolute time.time() after which the
    start loop stops
    """
    if mode not in mdp.V_det:
        return 0
    env, planner, nodes = roadmap.env, roadmap.planner, mdp.det_nodes[mode]
    entries = det_entry_nodes(env, roadmap, mdp, mode)
    reachable = np.nonzero(np.isfinite(mdp.V_det[mode]))[0]
    if len(reachable) == 0:
        return 0

    entries = list(dict.fromkeys(int(v) for v in entries))
    extra = max(0, cfg.n_starts - len(entries))
    uniform = rng.choice(reachable, size=min(extra, len(reachable)), replace=False)
    starts = list(dict.fromkeys(entries + [int(v) for v in uniform]))
    added = 0
    for start in starts:
        if deadline is not None and time.time() > deadline:
            break
        chain = policy_chain(mdp, mode, start)
        if len(chain) < 3:
            continue
        states = [nodes[i].state for i in chain]
        shortcut = shortcut_states(env, states, _interp_resolution(planner))
        if len(shortcut) < 2:
            continue

        old = _leg_cost(env, states)
        new = _leg_cost(env, shortcut)
        if new >= old - _ABS_MARGIN:
            continue

        added += insert_chain(planner, mode, nodes[chain[0]], nodes[chain[-1]], shortcut)
    return added

def det_chain_heads(roadmap: ReactiveRoadmap, mdp: ReactiveMDP) -> List[Mode]:
    """
    Deterministic modes a cross-seam rollout can START in, i.e. those with at least one det -> det
    seam leaving them
    """
    skill = set(roadmap.skill_modes)
    heads = []
    for src, dst in roadmap.seams:
        if src in skill or dst in skill:
            continue
        if src in mdp.V_det and dst in mdp.V_det and src not in heads:
            heads.append(src)
    return heads

class ChainRollout:
    """One forward rollout of pi_det across a run of deterministic modes"""
    __slots__ = ("states", "first", "last", "seams")

    def __init__(self, states: List[State], first: Node, last: Node,
                 seams: List[Tuple[Node, Node]]):
        self.states, self.first, self.last, self.seams = states, first, last, seams

def policy_rollout_chain(mdp: ReactiveMDP, skill_modes, mode: Mode, start: int,
                         window: Optional[int] = 2) -> Optional[ChainRollout]:
    """
    Walks pi_det from `start`, crossing every seam the POLICY chooses until it would leave the
    deterministic world or the window is full
    """
    states: List[State] = []
    seams: List[Tuple[Node, Node]] = []
    node_idx, first_node, last_node = start, None, None
    visited_modes = set()

    while True:
        nodes, pi = mdp.det_nodes[mode], mdp.pi_det[mode]
        if not (0 <= node_idx < len(nodes)):
            return None
        visited_modes.add(mode)

        i, last_i, seen = node_idx, node_idx, set()
        while i >= 0 and i not in seen:
            seen.add(i)
            states.append(nodes[i].state)
            last_i = i
            i = int(pi[i])
        if first_node is None:
            first_node = nodes[node_idx]
        last_node = nodes[last_i]

        if window is not None and len(visited_modes) >= window:
            break
        child = mdp.det_exit[mode].get(last_i)
        if child is None:
            break
        nxt_mode = child.state.mode

        if (nxt_mode in skill_modes or nxt_mode not in mdp.V_det
                or nxt_mode in visited_modes):
            break
        nxt = mdp.det_pos[nxt_mode].get(id(child))
        if nxt is None:
            break
        states.append(child.state)
        seams.append((nodes[last_i], child))
        mode, node_idx, last_node = nxt_mode, nxt, child

    if first_node is None or last_node is None or len(states) < 3:
        return None
    return ChainRollout(states, first_node, last_node, seams)

def split_on_mode_change(states: Sequence[State]) -> List[List[State]]:
    """
    Contiguous runs of equal mode. With the doubled seam convention every run after the first begins
    with its own copy of the seam configuration, so the runs tile the trajectory and each one is a
    complete leg through a single mode
    """
    legs: List[List[State]] = []
    for st in states:
        if legs and legs[-1][0].mode == st.mode:
            legs[-1].append(st)
        else:
            legs.append([st])
    return legs

def insert_chain_open(planner, mode: Mode, first: Node,
                      states: Sequence[State]) -> Tuple[int, Optional[Node]]:
    """
    insert_chain with the LAST waypoint materialised as a real node instead of being merged into a
    pinned endpoint
    """
    env = planner.env
    qs = [first.state.q] + [s.q for s in states[1:]]
    if not _path_is_collision_free(env, qs, mode):
        return 0, None

    subtree = planner.tree.subtrees[mode]
    prev, added = first, 0
    for st in states[1:]:
        node = Node(_copy_state(st, mode), parent=prev)
        node.cost_to_parent = env.config_cost(prev.state.q, node.state.q)
        node.cost = prev.cost + node.cost_to_parent
        prev.children.append(node)
        subtree.add_node(node)
        planner._roadmap_add_node(node, mode, is_skill=False)
        prev, added = node, added + 1
    return added, prev

def synthesize_seam(planner, exit_node: Node, mode_next: Mode) -> Optional[Node]:
    """Hangs a transition twin in `mode_next` off a freshly created exit node. Returns the twin"""
    env = planner.env
    q = exit_node.state.q
    if not env.is_collision_free(q, mode_next):
        return None

    seed = Node(State(q.from_flat(np.asarray(q.state(), dtype=np.float64)), mode_next),
                parent=exit_node)
    seed.cost_to_parent = 0.0
    seed.cost = exit_node.cost
    exit_node.children.append(seed)
    planner.tree.subtrees[mode_next].add_node(seed)
    planner._roadmap_add_node(seed, mode_next, is_skill=False)
    return seed

def inject_chain(planner, env, rollout: ChainRollout, states: Sequence[State]) -> int:
    """Puts a shortcut multi-mode trajectory back into the roadmap. Returns elements added"""
    legs = split_on_mode_change(states)
    if len(legs) != len(rollout.seams) + 1:
        return 0

    added, first = 0, rollout.first
    for k, leg in enumerate(legs):
        mode = leg[0].mode
        if k + 1 == len(legs):
            return added + insert_chain(planner, mode, first, rollout.last, leg)

        exit_a, seed_b = rollout.seams[k]

        if not np.allclose(np.asarray(leg[-1].q.state(), dtype=np.float64),
                           np.asarray(legs[k + 1][0].q.state(), dtype=np.float64), atol=1e-9):
            return added

        if np.allclose(np.asarray(leg[-1].q.state(), dtype=np.float64),
                       np.asarray(exit_a.state.q.state(), dtype=np.float64), atol=1e-9):
            added += insert_chain(planner, mode, first, exit_a, leg)
            first = seed_b
            continue

        assert env.is_transition(leg[-1].q, mode), "shortcut moved a goal-constrained robot"
        n, new_exit = insert_chain_open(planner, mode, first, leg)
        added += n
        if new_exit is None:
            return added
        new_seed = synthesize_seam(planner, new_exit, legs[k + 1][0].mode)
        if new_seed is None:
            return added
        added += 1
        first = new_seed
    return added

def improve_det_chain(roadmap: ReactiveRoadmap, mdp: ReactiveMDP, head: Mode,
                      cfg: PolicyShortcutConfig, rng,
                      deadline: Optional[float] = None) -> int:
    """
    One round for rollouts STARTING in a single deterministic mode and crossing whatever seams the
    policy takes them across
    """
    env, planner = roadmap.env, roadmap.planner
    if head not in mdp.V_det or head not in mdp.pi_det:
        return 0
    skill_modes = set(roadmap.skill_modes)
    reachable = np.nonzero(np.isfinite(mdp.V_det[head]))[0]
    if len(reachable) == 0:
        return 0

    entries = [v for v in det_entry_nodes(env, roadmap, mdp, head) if np.isfinite(mdp.V_det[head][v])]
    entries = list(dict.fromkeys(int(v) for v in entries))
    extra = max(0, cfg.n_starts - len(entries))
    uniform = rng.choice(reachable, size=min(extra, len(reachable)), replace=False)
    starts = list(dict.fromkeys(entries + [int(v) for v in uniform]))
    starts = _rotate_starts(roadmap, ("chain", head), starts, cfg.n_starts)
    added = 0
    for start in starts:
        if deadline is not None and time.time() > deadline:
            break
        rollout = policy_rollout_chain(mdp, skill_modes, head, start, cfg.chain_window)

        if rollout is None or not rollout.seams:
            continue
        shortcut = shortcut_states(env, rollout.states, _interp_resolution(planner))
        if len(shortcut) < 2:
            continue

        old = _leg_cost(env, rollout.states)
        new = _leg_cost(env, shortcut)
        if new >= old - _ABS_MARGIN:
            continue
        added += inject_chain(planner, env, rollout, shortcut)
    return added

def rollout_skill_indices(mdp: ReactiveMDP, mode: Mode, v0: int, rng) -> List[int]:
    """One (nature realization, policy response) pair, as the VERTEX INDICES the policy visits"""
    _, nature, _ = mdp._mode_nature(mode)
    pi = mdp.pi_skill[mode]
    v = int(v0)
    visited = [v]
    for nu in sample_kernel_trajectory(nature, rng):
        nxt = int(pi[v, nu])
        if nxt < 0:
            break
        v = nxt
        visited.append(v)
    return visited

def rollout_skill(mdp: ReactiveMDP, mode: Mode, v0: int, rng) -> np.ndarray:
    """The same rollout as an inactive-subspace polyline, which is what straighten() consumes"""
    graph = mdp.roadmap.inactive_roadmaps[mode]
    visited = rollout_skill_indices(mdp, mode, v0, rng)
    return np.asarray(graph.vertices[visited], dtype=np.float64)

def straighten(env, graph, path: np.ndarray, rng) -> np.ndarray:
    """Straightens an inactive-subspace polyline while KEEPING THE NUMBER OF WAYPOINTS FIXED"""
    n = len(path)
    if n < 3:
        return path
    free, _ = inactive_feasibility(env, graph.mode, graph.robots, graph.indices)
    path = np.array(path, dtype=np.float64, copy=True)

    for _ in range(4 * n):
        i, j = (int(x) for x in np.sort(rng.integers(0, n, size=2)))
        if j - i < 2:
            continue
        if np.linalg.norm(path[j] - path[i]) >= _polyline_length(path[i: j + 1]) - 1e-9:
            continue
        chord = path[i] + np.linspace(0.0, 1.0, j - i + 1)[:, None] * (path[j] - path[i])
        if all(free(q) for q in chord[1:-1]):
            path[i: j + 1] = chord
    return path

def improve_skill_mode(roadmap: ReactiveRoadmap, mdp: ReactiveMDP, mode: Mode,
                       cfg: PolicyShortcutConfig, rng) -> int:
    """
    One round for a single skill mode. Several realizations per start because the response IS
    realization-dependent: one rollout would straighten one branch of a bimodal skill and ignore the
    other
    """
    graph = roadmap.inactive_roadmaps.get(mode)
    if graph is None or graph.n_vertices == 0 or len(graph.indices) == 0:
        return 0
    if mode not in mdp.pi_skill:
        return 0

    entries = list(dict.fromkeys(int(v) for v in graph.seeds.get("entry", [])))
    starts = _rotate_starts(roadmap, ("skill", mode), entries, cfg.n_starts)
    extra = max(0, cfg.n_starts - len(starts))
    if extra:
        uniform = rng.integers(0, graph.n_vertices, size=extra)
        starts = list(dict.fromkeys(starts + [int(v) for v in uniform]))[:cfg.n_starts]
    proposals: List[np.ndarray] = []
    for v0 in starts:
        for _ in range(cfg.n_rollouts):
            path = rollout_skill(mdp, mode, v0, rng)
            if len(path) < 3:
                continue
            straightened = straighten(roadmap.env, graph, path, rng)

            if _polyline_length(straightened) >= _polyline_length(path) * (1.0 - _MARGIN):
                continue
            proposals.extend(straightened[1:-1])

    if not proposals:
        return 0

    inserted = int(add_inactive_vertices(
        roadmap.env, graph, proposals, roadmap.config,
        k_connect=roadmap.config.inactive_k_max))
    return inserted

def iterative_policy_shortcut(
    roadmap: ReactiveRoadmap,
    mdp_config: Optional[ReactiveMDPConfig] = None,
    cfg: Optional[PolicyShortcutConfig] = None,
    deadline: Optional[float] = None,
    previous: Optional[ReactiveMDP] = None,
) -> Tuple[ReactiveMDP, List[Dict]]:
    """Returns (final MDP, per-round history)"""
    cfg = cfg or PolicyShortcutConfig()
    mdp_config = mdp_config or ReactiveMDPConfig()
    rng = _persistent_rng(roadmap, "_shortcut_rng", cfg.seed)

    def solve(previous: Optional[ReactiveMDP]) -> ReactiveMDP:
        mdp = ReactiveMDP(roadmap, mdp_config)
        if previous is not None:
            mdp._edge_cache = previous._edge_cache
            mdp._mask_cache = previous._mask_cache
            mdp._exit_edge_cache = previous._exit_edge_cache
            mdp._exit_choice = previous._exit_choice
            mdp._state_cc_cache = previous._state_cc_cache
        return mdp.solve()

    history: List[Dict] = []
    mdp = solve(previous)
    history.append(_snapshot(0, mdp, added=0, improve_time=0.0))
    if not cfg.enabled:
        return mdp, history

    chain_rng = _persistent_rng(roadmap, "_shortcut_chain_rng", cfg.seed + 7717)
    flat = 0
    barren_modes = 0
    carried_improve = 0.0
    last_round = 0.0
    sweep_start = int(getattr(roadmap, "_shortcut_sweep_start", 0))
    for k in range(1, cfg.n_iters + 1):

        if deadline is not None and time.time() + last_round > deadline:
            break
        t0 = time.time()
        added = 0
        improve_deadline = None
        if deadline is not None:
            reserve = float(mdp.stats.get("solve_time", 0.0))
            improve_deadline = deadline - reserve * (_COLD_SOLVE_DISCOUNT if k == 1 else 1.0)

        if cfg.do_det_chains:
            _np_state = np.random.get_state()
            heads = _rotating_window(
                roadmap, "_shortcut_chain_head_cursor", det_chain_heads(roadmap, mdp),
                cfg.chain_heads_per_round)
            for head in heads:
                if improve_deadline is not None and time.time() > improve_deadline:
                    break
                added += improve_det_chain(roadmap, mdp, head, cfg, chain_rng,
                                          deadline=improve_deadline)
            np.random.set_state(_np_state)

        order = list(roadmap.modes)
        swept = 0
        n_sweep = (len(order) if cfg.modes_per_round is None
                   else min(len(order), max(1, cfg.modes_per_round)))
        for j in range(n_sweep):
            if improve_deadline is not None and time.time() > improve_deadline:
                break
            mode = order[(sweep_start + j) % len(order)]
            if mode in roadmap.skill_modes:
                if cfg.do_skill_modes:
                    added += improve_skill_mode(roadmap, mdp, mode, cfg, rng)
            elif cfg.do_det_modes:
                added += improve_det_mode(roadmap, mdp, mode, cfg, rng, deadline=improve_deadline)
            swept += 1
        sweep_start = (sweep_start + swept) % max(1, len(order))
        roadmap._shortcut_sweep_start = sweep_start
        improve_time = time.time() - t0

        if added == 0 and swept == 0:
            break
        barren_modes = barren_modes + swept if added == 0 else 0
        if added == 0 and barren_modes >= len(order):
            break
        if added == 0:
            last_round = time.time() - t0
            carried_improve += last_round
            continue

        v_old = mdp.get_start_cost_to_go()
        if cfg.do_det_chains:
            roadmap._extract_mode_transitions()
        mdp = solve(mdp)
        history.append(_snapshot(k, mdp, added, improve_time + carried_improve))
        carried_improve = 0.0

        v_new = mdp.get_start_cost_to_go()
        gain = (v_old - v_new) / abs(v_old) if np.isfinite(v_old) and v_old else np.inf

        last_round = time.time() - t0
        flat = flat + 1 if gain < _MARGIN else 0
        if flat >= cfg.patience:
            break

    return mdp, history

def _snapshot(k: int, mdp: ReactiveMDP, added: int, improve_time: float) -> Dict:
    return {
        "iter": k,
        "V_start": float(mdp.get_start_cost_to_go()),
        "added": int(added),
        "improve_time": float(improve_time),
        "solve_time": float(mdp.stats.get("solve_time", 0.0)),
    }
