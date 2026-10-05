import heapq
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from multi_robot_multi_goal_planning.problems.core.configuration import (
    batch_config_cost,
    _batch_config_cost_impl,
)
from multi_robot_multi_goal_planning.problems.planning_env import Mode

from .rrt_skills_reactive import (
    DONE,
    ReactiveRoadmap,
    active_skill_tasks,
    frame_positions,
    is_legal_mode_successor,
    quantize_to_nature_key,
    robot_frame_spheres,
    subspace_indices,
)

def kernel_is_layered(nature) -> bool:
    """
    True when every nature transition moves strictly forward in decision epoch, i.e. the kernel is a
    layered DAG
    """
    if not nature.time_indexed:
        return False
    epochs = nature.epochs
    for i, successors in enumerate(nature.kernel):
        for j, _ in successors:
            if j != DONE and epochs[j] <= epochs[i]:
                return False
    return True

@dataclass
class ReactiveMDPConfig:
    """Configuration parameters for MDP construction and Bellman solving"""
    mask_bin_samples: int = 8
    connect_k: int = 25
    validate_exit_edges: bool = False
    det_edge_validation: str = "solve"
    exit_repair_rounds: int = 6
    validate_edges: bool = True
    lazy_edge_validation: bool = True
    objective: str = "geometric"
    skill_epoch_distance_weight: float = 0.01
    block_unmodelled_early_goals: bool = True
    allow_wait: bool = True
    vi_max_iters: int = 100
    vi_tie_tol: float = 1e-9
    vi_tol: float = 1e-4
    max_sum_weight: Optional[float] = None

class ReactiveMDP:
    _EXIT_EDGE_CACHE_MAX = 2_000_000
    _ENTRY_MARGIN_HOPS = 3
    _WIDE_CANDIDATE_CAP = 200

    def __init__(self, roadmap: ReactiveRoadmap, config: Optional[ReactiveMDPConfig] = None):
        """Initialize an MDP over the reactive roadmap"""
        self.roadmap = roadmap
        self.env = roadmap.env
        self.config = config or ReactiveMDPConfig()
        self.template = np.asarray(self.env.get_start_pos().state(), dtype=np.float64)
        self.slice = self.env.get_start_pos()._array_slice
        self.det_nodes: Dict[Mode, list] = {}
        self.det_pos: Dict[Mode, dict] = {}
        self.V_det: Dict[Mode, np.ndarray] = {}
        self.pi_det: Dict[Mode, np.ndarray] = {}
        self.det_boundary: Dict[Mode, np.ndarray] = {}
        self.V_skill: Dict[Mode, np.ndarray] = {}
        self.pi_skill: Dict[Mode, np.ndarray] = {}
        self.blocked: Dict[Mode, np.ndarray] = {}
        self.exit_values: Dict[Mode, np.ndarray] = {}
        self.inherited: Dict[Mode, Dict[int, np.ndarray]] = {}
        self.det_exit: Dict[Mode, Dict[int, object]] = {}
        self._connect_cache: Dict[Mode, tuple] = {}
        self.stats: Dict[str, float] = {}
        self._edge_cache: Dict[Tuple[int, int], bool] = {}
        self._edge_cache_keepalive: set = set()
        planner = getattr(roadmap, "planner", None)
        for attr, ref in (("_roadmap_validated", "_roadmap_validated_nodes"),
                          ("_tree_edges_validated", "_tree_edges_validated_nodes")):
            for key in getattr(planner, attr, ()) or ():
                self._edge_cache[key] = True
            self._edge_cache_keepalive |= set(getattr(planner, ref, ()) or ())
        self.stats["edges_inherited"] = len(self._edge_cache)
        self._mask_cache: Dict[Mode, np.ndarray] = {}
        self._exit_edge_cache: Dict[Tuple[Mode, bytes, int], bool] = {}
        self._exit_choice: Dict[Tuple[Mode, bytes], int] = {}
        self._state_cc_cache: Dict[Tuple[Mode, bytes], bool] = {}

    def _cost(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluates the environment's exact multi-robot cost between batch configuration pairs"""
        if self.config.max_sum_weight is None:
            return batch_config_cost(starts, ends, self.env.cost_metric,
                                     self.env.cost_reduction, tmp_agent_slice=self.slice)
        return batch_config_cost(starts, ends, self.env.cost_metric, self.env.cost_reduction,
                                 w=self.config.max_sum_weight, tmp_agent_slice=self.slice)

    def _cost_from_diff(self, diff: np.ndarray) -> np.ndarray:
        """_cost for pairs whose DIFFERENCE is already known"""
        w = 0.01 if self.config.max_sum_weight is None else self.config.max_sum_weight
        return _batch_config_cost_impl(diff, self.slice, self.env.cost_metric,
                                       self.env.cost_reduction, w)

    def _compose_full_config(self, graph, vertices: np.ndarray, active_idx, active_q) -> np.ndarray:
        """Assembles full multi-robot configuration vectors from active and inactive components"""
        q = np.tile(self.template, (len(vertices), 1))
        q[:, graph.indices] = graph.vertices[vertices]
        q[:, active_idx] = active_q
        return q

    def _is_state_collision_free(self, q: np.ndarray, mode: Mode) -> bool:
        """Collision check for a raw configuration vector"""
        return bool(self.env.is_collision_free_np(q, mode))

    def _is_edge_collision_free(self, a, b, mode: Mode) -> bool:
        """Validate deterministic roadmap edges at a fine resolution"""
        key = (id(a), id(b)) if id(a) < id(b) else (id(b), id(a))
        if key not in self._edge_cache:
            _t = time.perf_counter()
            free = bool(self.env.is_edge_collision_free(
                a.state.q,
                b.state.q,
                mode,
            ))
            self._edge_cache[key] = free

            self._edge_cache_keepalive.add(a)
            self._edge_cache_keepalive.add(b)
            self.stats["n_det_edge_cc"] = self.stats.get("n_det_edge_cc", 0) + 1
            self.stats["t_det_edge_cc"] = self.stats.get("t_det_edge_cc", 0.0) + (time.perf_counter() - _t)
            if not free:
                self.stats["n_det_edge_blocked"] = self.stats.get("n_det_edge_blocked", 0) + 1
                tree = (getattr(a, "parent", None) is b) or (getattr(b, "parent", None) is a)
                if tree:
                    self.stats["n_det_tree_edge_blocked"] = (
                        self.stats.get("n_det_tree_edge_blocked", 0) + 1)
            else:
                tree = (getattr(a, "parent", None) is b) or (getattr(b, "parent", None) is a)
                if tree:
                    self.stats["n_det_tree_edge_cc"] = self.stats.get("n_det_tree_edge_cc", 0) + 1
        return self._edge_cache[key]

    def _edge_known_blocked(self, a, b) -> bool:
        """True only when this edge has ALREADY been checked and found in collision"""
        key = (id(a), id(b)) if id(a) < id(b) else (id(b), id(a))
        return self._edge_cache.get(key) is False

    def _mode_nature(self, mode: Mode):
        """Everything the solver needs to treat a skill mode PER ROBOT rather than per "the" skill"""
        tasks = self.roadmap.mode_active_tasks(mode)
        nature = self.roadmap.joint_nature_sets.get(mode)
        if nature is None:
            nature = self.roadmap.nature_sets[tasks[0].name]
        active_idx = subspace_indices(self.env, [r for t in tasks for r in t.robots])
        return tasks, nature, active_idx

    def _get_successor_mode(self, mode: Mode, task, still_active: bool) -> Optional[Mode]:
        """Finds successor mode where the skill task either continues running or has terminated"""
        for (src, dst) in self.roadmap.seams:
            if src != mode:
                continue
            running = any(t.name == task.name for t in active_skill_tasks(self.env, dst))
            if running == still_active:
                return dst
        return None

    def _get_successor_mode_for(self, mode: Mode, finished: Tuple[str, ...]) -> Optional[Mode]:
        """The mode reached when EXACTLY the skills named in `finished` complete and the rest keep running"""
        want = {t.name for t in self.roadmap.mode_active_tasks(mode)} - set(finished)
        for (src, dst) in self.roadmap.seams:
            if src != mode:
                continue
            running = {t.name for t in active_skill_tasks(self.env, dst)}
            if running & (want | set(finished)) == want:
                return dst
        return None

    def solve(self) -> "ReactiveMDP":
        """
        Solves the global MDP backwards from terminal goals to the start mode. Sweeps modes in reverse
        topological order, calling exact backward Dijkstra for deterministic modes and backward
        expectimax for skill modes
        """
        t0 = time.time()
        self._check_memory_budget()
        useful = set(self.roadmap.modes)
        self.stats["n_modes_total"] = len(useful)
        self.stats["n_modes_solved"] = len(useful)
        for mode in reversed(self._topological_mode_order()):
            if mode not in useful:

                continue
            if mode in self.roadmap.skill_modes:

                for _ in range(max(0, self.config.exit_repair_rounds) + 1):
                    self._solve_skill(mode)
                    if not self.config.exit_repair_rounds or not self.config.validate_exit_edges:
                        break
                    _tr = time.perf_counter()
                    tasks, nature, active_idx = self._mode_nature(mode)
                    graph = self.roadmap.inactive_roadmaps[mode]
                    changed = self._repair_policy_exits(mode, tasks, graph, nature, active_idx)
                    self.stats["t_exit_repair"] = (
                        self.stats.get("t_exit_repair", 0.0) + (time.perf_counter() - _tr))
                    if not changed:
                        break
                    self.stats["n_exit_repair_rounds"] = (
                        self.stats.get("n_exit_repair_rounds", 0) + 1)
            else:
                _td = time.perf_counter()
                self._solve_deterministic(mode)
                self.stats["t_det"] = self.stats.get("t_det", 0.0) + (time.perf_counter() - _td)
        self.stats["solve_time"] = time.time() - t0
        self.stats["edges_validated"] = len(self._edge_cache)
        return self

    def _check_memory_budget(self) -> None:
        """
        Predicts the value-table footprint BEFORE allocating it, and refuses rather than letting the OOM
        killer take the machine down
        """
        budget = self._available_memory_bytes()
        rows = []
        for mode in self.roadmap.skill_modes:
            graph = self.roadmap.inactive_roadmaps.get(mode)
            if graph is None or not active_skill_tasks(self.env, mode):
                continue
            try:
                _, nature, _ = self._mode_nature(mode)
            except KeyError:
                continue
            rows.append((mode, graph.n_vertices, nature.n_states,
                         graph.n_vertices * nature.n_states * 24))
        if not rows:
            return
        peak = max(r[3] for r in rows)
        self.stats["predicted_table_bytes"] = float(peak)
        if budget is None or peak < 0.35 * budget:
            return
        if peak > 0.8 * budget:
            raise MemoryError(
                f"reactive MDP needs ~{peak / 1e9:.1f} GB for one skill mode but only "
                f"{budget / 1e9:.1f} GB is available; refusing to start rather than "
                f"triggering the OOM killer. See the [MEMORY] report above.")

    @staticmethod
    def _available_memory_bytes() -> Optional[float]:
        """MemAvailable from /proc/meminfo, or None where that is not readable"""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return float(line.split()[1]) * 1024.0
        except Exception:
            pass
        return None

    def _topological_mode_order(self) -> List[Mode]:
        """Computes a topological ordering of all modes from start to terminal goals"""
        successors = {m: [] for m in self.roadmap.modes}
        indegree = {m: 0 for m in self.roadmap.modes}
        for src, dst in self.roadmap.seams:
            if src in successors and dst in indegree:
                successors[src].append(dst)
                indegree[dst] += 1
        queue = deque([m for m in self.roadmap.modes if indegree[m] == 0])
        order = []
        while queue:
            mode = queue.popleft()
            order.append(mode)
            for nxt in successors[mode]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        if len(order) != len(self.roadmap.modes):

            stuck = [m for m in self.roadmap.modes if m not in set(order)]
            residual = [(a.task_ids, b.task_ids) for a, b in self.roadmap.seams
                        if a in stuck and b in stuck]
            raise RuntimeError(
                f"mode graph is not a DAG; the backward sweep needs one. "
                f"{len(stuck)} of {len(self.roadmap.modes)} modes are in the cycle: "
                f"{[m.task_ids for m in stuck[:12]]}; seams among them: {residual[:20]}")
        return order

    def _get_successor_node_value(self, child) -> float:
        """
        Returns the optimal cost-to-go of a node located in an already-solved successor mode. Acts as a
        bridge between modes during backward induction, providing exact terminal costs
        """
        mode = child.state.mode
        if mode in self.V_det:
            i = self.det_pos[mode].get(id(child))
            return float(self.V_det[mode][i]) if i is not None else np.inf
        if mode in self.V_skill:
            graph = self.roadmap.inactive_roadmaps[mode]
            _, nature, _ = self._mode_nature(mode)
            v = self._find_nearest_vertex(graph, np.asarray(child.state.q.state())[graph.indices])
            return float(self.V_skill[mode][v, nature.initial])
        return np.inf

    @staticmethod
    def _find_nearest_vertex(graph, q_sub: np.ndarray) -> int:
        """Returns the index of the closest vertex in graph.vertices to q_sub"""
        return int(np.argmin(np.linalg.norm(graph.vertices - q_sub, axis=1)))

    @staticmethod
    def _value_delta(v_new: np.ndarray, v_old: np.ndarray) -> float:
        """Max |V - V_prev| between two Gauss-Seidel sweeps, restricted to entries finite in both"""
        finite_new, finite_old = np.isfinite(v_new), np.isfinite(v_old)
        if not np.array_equal(finite_new, finite_old):
            return np.inf
        if not finite_new.any():
            return 0.0
        return float(np.max(np.abs(v_new[finite_new] - v_old[finite_new])))

    def evaluate_connection_cost(self, q: np.ndarray, mode: Mode, validate: bool = True,
                                 wide: bool = False) -> Tuple[float, int]:
        """
        Evaluates cost-to-go from an off-roadmap (continuous) configuration q by connecting it into
        mode's (deterministic) roadmap
        """
        _t0 = time.perf_counter()
        self.stats["n_connect_calls"] = self.stats.get("n_connect_calls", 0) + 1
        if mode not in self.V_det:
            self.stats["n_connect_state_blocked"] = self.stats.get("n_connect_state_blocked", 0) + 1
            return np.inf, -1

        qb = q.tobytes()
        free = self._state_cc_cache.get((mode, qb))
        if free is None:
            free = self._is_state_collision_free(q, mode)
            self._state_cc_cache[(mode, qb)] = free
        else:
            self.stats["n_connect_state_cc_cached"] = self.stats.get("n_connect_state_cc_cached", 0) + 1
        self.stats["t_connect_state_cc"] = self.stats.get("t_connect_state_cc", 0.0) + (time.perf_counter() - _t0)
        if not free:
            self.stats["n_connect_state_blocked"] = self.stats.get("n_connect_state_blocked", 0) + 1
            return np.inf, -1
        node_q, values, reachable = self._connect_table(mode)
        if len(reachable) == 0:
            return np.inf, -1

        proved = self._exit_choice.get((mode, qb))
        if proved is not None:
            if proved < 0:
                if not wide:
                    return np.inf, -1

            else:
                where = np.nonzero(reachable == proved)[0]
                if len(where):
                    i = int(where[0])
                    return float(self._cost_from_diff(q[None, :] - node_q[i:i + 1])[0] + values[i]), proved

        geo_cost = self._cost_from_diff(q[None, :] - node_q)
        total_cost = geo_cost + values
        candidates = self._connect_candidates(geo_cost[None, :], total_cost[None, :])[0]
        cost, node = self._first_free_candidate(q, mode, candidates, total_cost, reachable, validate)
        if node < 0 and wide and validate and len(reachable) > len(candidates):

            allc = self._connect_candidates(geo_cost[None, :], total_cost[None, :],
                                            k=min(len(reachable), self._WIDE_CANDIDATE_CAP))[0]
            cost, node = self._first_free_candidate(q, mode, allc, total_cost, reachable, validate)
            key = "n_connect_widened_rescue" if node >= 0 else "n_connect_widened_failed"
            self.stats[key] = self.stats.get(key, 0) + 1
        if validate:

            self._exit_choice[(mode, q.tobytes())] = int(node)
        return cost, node

    def _connect_table(self, mode: Mode):
        """(node_q, V, index) for the reachable vertices of a deterministic mode's roadmap, memoized"""
        if mode not in self._connect_cache:
            V = self.V_det[mode]
            reachable = np.nonzero(np.isfinite(V))[0]
            node_q = (np.stack([np.asarray(self.det_nodes[mode][i].state.q.state()) for i in reachable])
                      if len(reachable) else np.zeros((0, len(self.template))))
            self._connect_cache[mode] = (node_q, V[reachable], reachable)
        return self._connect_cache[mode]

    def _connect_candidates(self, geo_cost: np.ndarray, total_cost: np.ndarray,
                            k: Optional[int] = None) -> np.ndarray:
        """The connect_k spatially nearest roadmap vertices per row, ordered by cost-to-go"""
        n_nodes = geo_cost.shape[1]
        k = min(n_nodes, self.config.connect_k if k is None else k)
        if k < n_nodes:
            nearest = np.argpartition(geo_cost, k - 1, axis=1)[:, :k]
        else:
            nearest = np.broadcast_to(np.arange(n_nodes), (geo_cost.shape[0], n_nodes))
        rows = np.arange(geo_cost.shape[0])[:, None]
        return np.take_along_axis(nearest, np.argsort(total_cost[rows, nearest], axis=1), axis=1)

    def _first_free_candidate(self, q, mode, candidates, total_cost, reachable, validate=True):
        """
        Walks the ordered candidates and returns the first whose connecting edge is free. With
        validate=False the first candidate is taken unchecked -- see
        ReactiveMDPConfig.validate_exit_edges for when that is sound and what it costs
        """
        if not validate:
            if len(candidates) == 0:
                return np.inf, -1
            c = candidates[0]
            return float(total_cost[c]), int(reachable[c])
        config = self.env.get_start_pos().from_flat(q)
        _t = time.perf_counter()
        qb = q.tobytes()
        cache = self._exit_edge_cache
        for c in candidates:
            target = self.det_nodes[mode][reachable[c]]
            self.stats["n_connect_edge_cc"] = self.stats.get("n_connect_edge_cc", 0) + 1
            ck = (mode, qb, id(target))
            free = cache.get(ck)
            if free is None:
                free = bool(self.env.is_edge_collision_free(
                    config,
                    target.state.q,
                    mode,
                ))

                if len(cache) < self._EXIT_EDGE_CACHE_MAX:
                    cache[ck] = free
            else:
                self.stats["n_connect_edge_cc_cached"] = (
                    self.stats.get("n_connect_edge_cc_cached", 0) + 1)
            if free:
                self.stats["t_connect_edge_cc"] = self.stats.get("t_connect_edge_cc", 0.0) + (time.perf_counter() - _t)
                return float(total_cost[c]), int(reachable[c])
        self.stats["t_connect_edge_cc"] = self.stats.get("t_connect_edge_cc", 0.0) + (time.perf_counter() - _t)
        self.stats["n_connect_all_blocked"] = self.stats.get("n_connect_all_blocked", 0) + 1
        return np.inf, -1

    def evaluate_connection_costs(self, configs: np.ndarray, mode: Mode,
                                  validate: bool = True) -> np.ndarray:
        """evaluate_connection_cost over a whole set of configurations. Deliberately still a loop"""
        return np.array([self.evaluate_connection_cost(q, mode, validate)[0] for q in configs])

    def _solve_deterministic(self, mode: Mode):
        """Computes exact cost-to-go and policy for a deterministic mode using backward Dijkstra"""
        subtree = self.roadmap.planner.tree.subtrees[mode]
        nodes = [n for n in subtree.nodes[: subtree.size] if not n.state.is_skill_waypoint]
        position = {id(n): i for i, n in enumerate(nodes)}
        V = np.full(len(nodes), np.inf)
        pi = np.full(len(nodes), -1, dtype=int)
        reweight = None
        if self.config.max_sum_weight is not None:
            starts, ends, keys = [], [], []
            for i, node in enumerate(nodes):
                for neighbour, _, kind in node.edges:
                    j = position.get(id(neighbour))
                    if kind == "geometric" and j is not None:
                        starts.append(node.state.q.state())
                        ends.append(neighbour.state.q.state())
                        keys.append((i, j))
            if keys:
                values = self._cost(np.asarray(starts, dtype=np.float64),
                                    np.asarray(ends, dtype=np.float64))
                reweight = dict(zip(keys, (float(v) for v in values)))

        boundary = np.full(len(nodes), np.inf)
        heap, exits = [], {}
        for i, node in enumerate(nodes):
            best, via = (0.0 if self.env.done(node.state.q, mode) else np.inf), None
            for child in node.children:
                if child.state.mode != mode:

                    if not is_legal_mode_successor(self.env, mode, child.state.mode):
                        continue
                    value = self._get_successor_node_value(child)
                    if value < best:
                        best, via = value, child
            if np.isfinite(best):
                V[i] = boundary[i] = best
                if via is not None:
                    exits[i] = via
                heapq.heappush(heap, (best, i))

        lazy = self.config.validate_edges and self.config.lazy_edge_validation
        check_at_pop = self.config.det_edge_validation != "execution"
        lazy = lazy or not check_at_pop
        settled = np.zeros(len(nodes), dtype=bool)
        while heap:
            d, i = heapq.heappop(heap)

            if settled[i] or abs(d - V[i]) > 1e-12:
                continue
            if (lazy and check_at_pop and pi[i] >= 0
                    and not self._is_edge_collision_free(nodes[i], nodes[pi[i]], mode)):

                V[i], pi[i] = boundary[i], -1
                for neighbour, weight, kind in nodes[i].edges:
                    j = position.get(id(neighbour))
                    if kind != "geometric" or j is None or not settled[j]:
                        continue
                    if reweight is not None:

                        weight = reweight.get((j, i), weight)
                    candidate = V[j] + weight
                    if candidate < V[i] - 1e-12 and not self._edge_known_blocked(nodes[i], nodes[j]):
                        V[i], pi[i] = candidate, j
                if np.isfinite(V[i]):
                    heapq.heappush(heap, (V[i], i))
                continue

            settled[i] = True
            for neighbour, weight, kind in nodes[i].edges:
                j = position.get(id(neighbour))
                if reweight is not None and j is not None:
                    weight = reweight.get((i, j), weight)
                if kind != "geometric" or j is None or d + weight >= V[j] - 1e-12:
                    continue
                if lazy:

                    if self._edge_known_blocked(nodes[j], nodes[i]):
                        continue
                elif self.config.validate_edges and not self._is_edge_collision_free(nodes[j], nodes[i], mode):
                    continue
                V[j], pi[j] = d + weight, i
                heapq.heappush(heap, (V[j], j))

        self.det_nodes[mode], self.det_pos[mode] = nodes, position
        self.V_det[mode], self.pi_det[mode] = V, pi
        self.det_exit[mode] = exits
        self.det_boundary[mode] = boundary

    @staticmethod
    def _position_groups(nature) -> np.ndarray:
        """Maps each nature state to the index of its active-robot POSITION bin"""
        groups, group_of = {}, np.empty(nature.n_states, dtype=int)
        for nu in range(nature.n_states):
            key = quantize_to_nature_key(nature.configs[nu], nature.bin_tol)
            group_of[nu] = groups.setdefault(key, len(groups))
        return group_of

    def _compute_robot_collision_mask(self, mode, graph, nature, active_idx, group_of) -> np.ndarray:
        """Precomputes robot-robot collision mask of shape (n_vertices, n_nature)"""
        n_groups = int(group_of.max()) + 1
        cached = self._mask_cache.get(mode)

        if cached is not None and (cached.shape[1] != n_groups
                                  or cached.shape[0] > graph.n_vertices):
            cached = None
        n_old = 0 if cached is None else cached.shape[0]

        if n_old < graph.n_vertices:
            new_idx = np.arange(n_old, graph.n_vertices)
            new = self._collision_free_rows(mode, graph, nature, active_idx, group_of,
                                            new_idx, n_groups)
            cached = new if cached is None else np.vstack([cached, new])
            self._mask_cache[mode] = cached

        return ~cached[:, group_of]

    def _collision_free_rows(self, mode, graph, nature, active_idx, group_of,
                             new_idx, n_groups) -> np.ndarray:
        """The (new vertices x position bins) FREE mask, evaluated without checking every pair"""
        env = self.env
        free = np.ones((len(new_idx), n_groups), dtype=bool)
        if len(new_idx) == 0:
            return free

        active_robots = [r for t in self.roadmap.mode_active_tasks(mode) for r in t.robots]
        inactive_robots_ = list(graph.robots)
        env.set_to_mode(mode)

        rep = np.full(n_groups, -1, dtype=int)
        for nu in range(nature.n_states):
            g = int(group_of[nu])
            if rep[g] < 0:
                rep[g] = nu

        ref = self._compose_full_config(graph, new_idx, active_idx, nature.configs[int(rep[0])])
        blocked_v = np.asarray(
            [not env.is_collision_free_for_robot(inactive_robots_, q, mode, set_mode=False)
             for q in ref])
        free[blocked_v, :] = False
        n_checks = len(new_idx)

        try:
            a_frames, a_radii = robot_frame_spheres(env, active_robots, ref[0])
            i_frames, i_radii = robot_frame_spheres(env, inactive_robots_, ref[0])
        except Exception:
            a_frames, i_frames = [], []
            a_radii = i_radii = np.zeros(0)
        use_broadphase = len(a_frames) > 0 and len(i_frames) > 0
        self.stats["broadphase_modes"] = self.stats.get("broadphase_modes", 0) + 1
        self.stats["broadphase_on"] = self.stats.get("broadphase_on", 0) + int(use_broadphase)

        if use_broadphase:

            i_pos = np.empty((len(new_idx), len(i_frames), 3))
            for k, q in enumerate(ref):
                i_pos[k] = frame_positions(env, i_frames, q)
            a_pos = np.empty((n_groups, len(a_frames), 3))
            probe = ref[0].copy()
            for g in range(n_groups):
                probe[active_idx] = nature.configs[int(rep[g])]
                a_pos[g] = frame_positions(env, a_frames, probe)

            pair_reach2 = (i_radii[:, None] + a_radii[None, :]) ** 2

        alive = np.nonzero(~blocked_v)[0]
        if use_broadphase:

            alive_flat = np.ascontiguousarray(i_pos[alive].reshape(-1, 3))
            alive_sq = np.einsum("ij,ij->i", alive_flat, alive_flat)
            n_i = len(i_frames)
            pair_reach2_flat = np.tile(pair_reach2, (len(alive), 1)) if len(alive) else pair_reach2
        probe = ref[0].copy()
        n_pruned = 0
        for g in range(n_groups):
            active_q = nature.configs[int(rep[g])]
            probe[active_idx] = active_q
            n_checks += 1
            if not env.is_collision_free_for_robot(active_robots, probe, mode, set_mode=False):
                free[:, g] = False
                continue
            if use_broadphase:

                ap = a_pos[g]
                d2 = (alive_sq[:, None]
                      + np.einsum("ij,ij->i", ap, ap)[None, :]
                      - 2.0 * (alive_flat @ ap.T))
                touching = (d2 <= pair_reach2_flat).reshape(len(alive), n_i * len(ap))
                candidates = alive[touching.any(axis=1)]
                n_pruned += len(alive) - len(candidates)
            else:
                candidates = alive
            if len(candidates) == 0:
                continue

            probes = [active_q]
            if self.config.mask_bin_samples > 1 and nature.samples is not None:
                extra = np.asarray(nature.samples[int(rep[g])], dtype=np.float64)
                probes += [extra[j] for j in range(1, min(len(extra),
                                                         self.config.mask_bin_samples))]

            survivors = candidates
            for probe_q in probes:
                if len(survivors) == 0:
                    break
                configs = self._compose_full_config(graph, new_idx[survivors], active_idx, probe_q)
                still = []
                for i, k in enumerate(survivors):
                    if env.is_collision_free_np(configs[i], mode, set_mode=False):
                        still.append(k)
                    else:
                        free[k, g] = False
                n_checks += len(survivors)
                survivors = np.asarray(still, dtype=int)
            free[survivors, g] = True

        self.stats["n_collision_checks"] = self.stats.get("n_collision_checks", 0) + n_checks
        self.stats["n_broadphase_pruned"] = self.stats.get("n_broadphase_pruned", 0) + n_pruned
        return free

    def _mode_entry_vertices(self, mode: Mode, graph) -> List[int]:
        """The inactive-roadmap vertices this skill mode can be ENTERED at, plus a margin"""
        entries = set()
        for (_, dst), qs in self.roadmap.seams.items():
            if dst != mode:
                continue
            for q in qs:
                entries.add(self._find_nearest_vertex(graph, np.asarray(q)[graph.indices]))
        if not entries and mode == self.env.get_start_mode():
            q = np.asarray(self.env.get_start_pos().state())
            entries.add(self._find_nearest_vertex(graph, q[graph.indices]))

        frontier, seen = deque((v, 0) for v in entries), set(entries)
        while frontier:
            v, d = frontier.popleft()
            if d >= self._ENTRY_MARGIN_HOPS:
                continue
            for j, _ in graph.adjacency[v]:
                if j not in seen:
                    seen.add(j)
                    frontier.append((j, d + 1))
        return sorted(seen)

    def _policy_exit_states(self, mode, graph, nature) -> List[Tuple[int, int]]:
        """
        The (vertex, nature state) pairs at which the CURRENT policy actually leaves this mode. A
        forward sweep of pi over the counted nature kernel from the mode's entry states
        """
        pi = self.pi_skill.get(mode)
        if pi is None:
            return []

        reached = {x for nu in range(nature.n_states) for x, _ in nature.kernel[nu] if x != DONE}
        sources = [nu for nu in range(nature.n_states) if nu not in reached] or [nature.initial]
        exits, seen = set(), set()
        stack = [(v, nu) for v in self._mode_entry_vertices(mode, graph) for nu in sources]
        while stack:
            state = stack.pop()
            if state in seen:
                continue
            seen.add(state)
            v, nu = state
            nxt_v = int(pi[v, nu])
            if nxt_v < 0:
                continue
            for x, _ in nature.kernel[nu]:
                if x == DONE:
                    exits.add((nxt_v, nu))
                else:
                    stack.append((nxt_v, x))
        self.stats["n_policy_exit_states"] = (
            self.stats.get("n_policy_exit_states", 0) + len(exits))
        return sorted(exits)

    def _repair_policy_exits(self, mode, tasks, graph, nature, active_idx) -> int:
        """
        Collision-checks the exit edges the policy actually uses, and returns how many of them turned
        out to be blocked at their optimistic choice. The whole point of lazy exit validation: the
        optimistic pass priced every exit off its NEAREST candidate without checking it, and the memory
        `project_validate_exit_edges_time` records what that costs -- under --cost_model time the
        inactive robots' in-window motion is free, so VI parks them wherever it likes and then believes
        an exit exists from those parked states
        """
        repaired = 0
        for v, nu in self._policy_exit_states(mode, graph, nature):
            for nxt, active_config in self._exit_targets(mode, tasks, nature, nu):
                if nxt is None or nxt not in self.V_det:
                    continue
                q = self._compose_full_config(graph, [v], active_idx, active_config)[0]
                if (nxt, q.tobytes()) in self._exit_choice:
                    continue
                _, node = self.evaluate_connection_cost(q, nxt, validate=True, wide=True)

                if node != self._optimistic_choice(q, nxt):
                    repaired += 1
        self.stats["n_exit_repairs"] = self.stats.get("n_exit_repairs", 0) + repaired
        return repaired

    def _optimistic_choice(self, q: np.ndarray, mode: Mode) -> int:
        """
        The roadmap node an UNVALIDATED evaluate_connection_cost would have picked for `q`: the cheapest
        (edge cost + V) candidate, taken without a collision check
        """
        if mode not in self.V_det:
            return -1
        node_q, values, reachable = self._connect_table(mode)
        if len(reachable) == 0:
            return -1
        geo_cost = self._cost_from_diff(q[None, :] - node_q)
        candidates = self._connect_candidates(geo_cost[None, :], (geo_cost + values)[None, :])[0]
        return int(reachable[candidates[0]]) if len(candidates) else -1

    def _exit_targets(self, mode, tasks, nature, nu):
        """(successor mode, active-block configuration) for every way the mode can end at nature state `nu`"""
        if not nature.is_joint:
            if nu not in nature.exits:
                return []
            return [(self._get_successor_mode(mode, tasks[0], still_active=False),
                     nature.exits[nu])]
        return [(self._get_successor_mode_for(mode, b.finished), b.active_config)
                for b in (nature.exit_branches or {}).get(nu, [])]

    def _compute_skill_exit_values(self, mode, tasks, graph, nature, active_idx) -> np.ndarray:
        """Evaluates cost-to-go when the mode ENDS, i.e. when a skill running in it finishes"""
        out = np.full((graph.n_vertices, nature.n_states), np.inf)
        all_vertices = np.arange(graph.n_vertices)

        if not nature.is_joint:
            nxt = self._get_successor_mode(mode, tasks[0], still_active=False)
            if nxt is None:
                return out

            cache = {}
            for nu in nature.exits:
                key = quantize_to_nature_key(nature.exits[nu], nature.bin_tol)
                if key not in cache:
                    cache[key] = self._boundary_values(
                        nxt, graph, all_vertices, active_idx, nature.exits[nu], None)
                out[:, nu] = cache[key]
            return out

        cache = {}
        for nu, branches in nature.exit_branches.items():
            column = np.zeros(graph.n_vertices)
            for branch in branches:
                nxt = self._get_successor_mode_for(mode, branch.finished)
                key = (nxt, branch.finished, branch.remaining,
                       quantize_to_nature_key(branch.active_config, nature.bin_tol))
                if key not in cache:
                    cache[key] = self._boundary_values(
                        nxt, graph, all_vertices, active_idx, branch.active_config,
                        branch, nature.components)
                column = column + branch.prob * cache[key]
            out[:, nu] = column
        return out

    def _boundary_values(self, nxt, graph, vertices, active_idx, active_config,
                         branch=None, components=None) -> np.ndarray:
        """Cost-to-go just after the mode ends, for every inactive vertex, on ONE exit branch"""
        n = len(vertices)
        if nxt is None:
            return np.full(n, np.inf)
        configs = self._compose_full_config(graph, vertices, active_idx, active_config)

        if nxt in self.V_det:

            eager = self.config.validate_exit_edges and not self.config.exit_repair_rounds
            return self.evaluate_connection_costs(configs, nxt, eager)

        if nxt not in self.V_skill:
            return np.full(n, np.inf)

        other = self.roadmap.inactive_roadmaps[nxt]
        nu_next = self._successor_nature_state(nxt, branch, components)
        if nu_next is None:
            return np.full(n, np.inf)

        _, nxt_nature, nxt_active_idx = self._mode_nature(nxt)
        projected = np.ascontiguousarray(configs[:, other.indices])
        verts = other.vertices
        sq_verts = np.einsum("ij,ij->i", verts, verts)
        nearest = np.empty(len(projected), dtype=np.intp)
        chunk = max(1, 32_000_000 // max(1, len(verts)))
        for start in range(0, len(projected), chunk):
            block = projected[start:start + chunk]
            d2 = sq_verts[None, :] - 2.0 * (block @ verts.T)
            nearest[start:start + len(block)] = np.argmin(d2, axis=1)
        targets = self._compose_full_config(
            other, nearest, nxt_active_idx, nxt_nature.configs[nu_next])
        values = self.V_skill[nxt][nearest, nu_next]
        return self._cost(configs, targets) + values

    def _successor_nature_state(self, nxt, branch, components) -> Optional[int]:
        """The nature state `nxt` is entered at, once the previous mode ended"""
        _, nxt_nature, _ = self._mode_nature(nxt)
        survivors = ({} if branch is None or components is None else
                     {name: state for name, state in zip(components, branch.remaining)
                      if state != DONE})

        def state_of(name: str) -> Optional[int]:
            if name in survivors:
                return int(survivors[name])
            fresh = self.roadmap.nature_sets.get(name)
            return None if fresh is None else int(fresh.initial)

        if nxt_nature.is_joint:
            key = tuple(state_of(name) for name in nxt_nature.components)
            if any(k is None for k in key):
                return None
            return nxt_nature.index.get(key)
        return state_of(self.roadmap.mode_active_tasks(nxt)[0].name)

    def _compute_early_goal_values(self, mode, graph, nature) -> Dict[int, np.ndarray]:
        """
        Extracts downstream cost-to-go columns for vertices where an inactive robot completes its task
        early while the mode's skills are all still running
        """
        nxt = self._get_successor_mode_for(mode, finished=())
        if nxt is None or nxt not in self.V_skill:
            return {}
        other = self.roadmap.inactive_roadmaps[nxt]
        _, nxt_nature, _ = self._mode_nature(nxt)
        goal_vertices = graph.seeds.get("task_goal", [])
        if (np.array_equal(graph.indices, other.indices)
                and nxt_nature.n_states == nature.n_states):
            return {
                v: self.V_skill[nxt][self._find_nearest_vertex(other, graph.vertices[v])]
                for v in goal_vertices
            }
        return {}

    def _unmodelled_early_goal_vertices(self, mode, graph, nature, active_idx) -> List[int]:
        """
        Inactive-roadmap vertices where the EXECUTOR would leave this mode into a mode the model does
        not contain
        """
        seeds = graph.seeds.get("task_goal", [])
        if not seeds or self.config.block_unmodelled_early_goals is False:
            return []
        out: List[int] = []
        for v in seeds:
            q = self._compose_full_config(graph, [v], active_idx, nature.configs[nature.initial])[0]
            config = self.env.get_start_pos().from_flat(q)
            try:
                if not self.env.is_transition(config, mode):
                    continue
                nxt = self.env.get_next_modes(config, mode)
            except ValueError:
                continue

            if nxt and not any(m in self.V_skill or m in self.det_nodes for m in nxt):
                out.append(int(v))
        return out

    @staticmethod
    def _topological_nature_order(nature) -> List[int]:
        """Computes topological ordering of discrete nature states from start to finish"""
        indegree = [0] * nature.n_states
        for i, successors in enumerate(nature.kernel):
            for j, _ in successors:
                if j != DONE and j != i:
                    indegree[j] += 1
        queue = deque([i for i in range(nature.n_states) if indegree[i] == 0])
        order = []
        while queue:
            i = queue.popleft()
            order.append(i)
            for j, _ in nature.kernel[i]:
                if j != DONE and j != i:
                    indegree[j] -= 1
                    if indegree[j] == 0:
                        queue.append(j)
        if len(order) != nature.n_states:
            remaining = [i for i in range(nature.n_states) if indegree[i] > 0]
            order.extend(remaining)
        return order

    def _solve_skill(self, mode: Mode):
        """Solves optimal reactive policy in a skill mode via backward expectimax Bellman updates"""
        graph = self.roadmap.inactive_roadmaps[mode]
        tasks, nature, active_idx = self._mode_nature(mode)
        n_vertices, n_nature = graph.n_vertices, nature.n_states
        src, dst = [], []
        for i, adjacent in enumerate(graph.adjacency):
            if self.config.allow_wait:
                src.append(i)
                dst.append(i)
            for j, _ in adjacent:
                src.append(i)
                dst.append(j)
        src, dst = np.asarray(src), np.asarray(dst)
        self.stats["n_actions"] = self.stats.get("n_actions", 0) + len(src)
        self.stats["n_nature"] = self.stats.get("n_nature", 0) + n_nature
        self.stats["n_skill_vertices"] = self.stats.get("n_skill_vertices", 0) + n_vertices
        group_of = self._position_groups(nature)
        self.stats["n_position_groups"] = (
            self.stats.get("n_position_groups", 0) + int(group_of.max()) + 1)
        _t0 = time.perf_counter()
        blocked = self._compute_robot_collision_mask(mode, graph, nature, active_idx, group_of)
        _t1 = time.perf_counter()
        self.stats["t_mask"] = self.stats.get("t_mask", 0.0) + (_t1 - _t0)
        exit_values = self._compute_skill_exit_values(mode, tasks, graph, nature, active_idx)
        _t2 = time.perf_counter()
        self.stats["t_exit"] = self.stats.get("t_exit", 0.0) + (_t2 - _t1)
        inherited = self._compute_early_goal_values(mode, graph, nature)

        for v in self._unmodelled_early_goal_vertices(mode, graph, nature, active_idx):
            blocked[v, :] = True
        self.exit_values[mode], self.inherited[mode] = exit_values, inherited

        order = self._topological_nature_order(nature)
        no_choice = graph.n_vertices <= 1
        exit_finite = np.isfinite(exit_values).any(axis=0)
        dead = blocked.all(axis=0).copy()
        if no_choice:

            for nu in reversed(order):
                if dead[nu]:
                    continue
                alive = False
                for x, _ in nature.kernel[nu]:
                    if (exit_finite[nu] if x == DONE else not dead[x]):
                        alive = True
                        break
                dead[nu] = not alive
        kernel = nature.kernel
        if dead.any():
            kernel = []
            for nu in range(nature.n_states):
                keep = [(x, p) for x, p in nature.kernel[nu] if x == DONE or not dead[x]]
                total_p = sum(p for _, p in keep)

                kernel.append([(x, p / total_p) for x, p in keep] if total_p > 0.0
                              else list(nature.kernel[nu]))

        w = self.config.max_sum_weight if self.config.max_sum_weight is not None else 0.01
        active_q_dummy = nature.configs[0]
        starts_I = self._compose_full_config(graph, src, active_idx, active_q_dummy)
        ends_I = self._compose_full_config(graph, dst, active_idx, active_q_dummy)
        s_I = batch_config_cost(starts_I, ends_I, self.env.cost_metric, "sum", tmp_agent_slice=self.slice)
        m_I = batch_config_cost(starts_I, ends_I, self.env.cost_metric, "max", w=0.0, tmp_agent_slice=self.slice)
        chunk_costs = {}
        geometry_keys_seen = set()

        for nu in range(n_nature):
            for x, _ in kernel[nu]:
                if x == DONE:
                    end_config = nature.exits[nu]
                    key = (int(group_of[nu]), quantize_to_nature_key(end_config, nature.bin_tol))
                else:
                    end_config = nature.configs[x]
                    key = (int(group_of[nu]), int(group_of[x]))

                start_A = self._compose_full_config(graph, [0], active_idx, nature.configs[nu])
                end_A = self._compose_full_config(graph, [0], active_idx, end_config)
                s_A = batch_config_cost(start_A, end_A, self.env.cost_metric, "sum", tmp_agent_slice=self.slice)[0]
                m_A = batch_config_cost(start_A, end_A, self.env.cost_metric, "max", w=0.0, tmp_agent_slice=self.slice)[0]
                chunk_costs[(nu, x)] = (m_A, s_A)
                geometry_keys_seen.add(key)

        mode_bytes = 5 * len(src) * 8 + 2 * len(chunk_costs) * 8
        self.stats["n_transitions"] = self.stats.get("n_transitions", 0) + len(chunk_costs)
        self.stats["n_geometry_keys"] = self.stats.get("n_geometry_keys", 0) + len(geometry_keys_seen)
        self.stats["n_skill_modes"] = self.stats.get("n_skill_modes", 0) + 1
        self.stats["geometry_peak_bytes"] = max(
            self.stats.get("geometry_peak_bytes", 0), mode_bytes)
        self.stats["geometry_sum_bytes"] = (
            self.stats.get("geometry_sum_bytes", 0) + mode_bytes)
        _t3 = time.perf_counter()
        self.stats["t_chunk"] = self.stats.get("t_chunk", 0.0) + (_t3 - _t2)
        n_iters = 0
        is_sum = self.env.cost_reduction == "sum"
        is_time = self.config.objective == "time"
        duration = self.env.v_ref / self.roadmap.config.decision_frequency if is_time else 0.0
        dist_wt = self.config.skill_epoch_distance_weight if is_time else 1.0
        w_sI = w * s_I
        geo_buf = np.empty(len(src))
        term_buf = np.empty(len(src))
        layered = kernel_is_layered(nature)
        max_sweeps = 1 if layered else self.config.vi_max_iters
        self.stats["vi_layered_modes"] = self.stats.get("vi_layered_modes", 0) + int(layered)
        V = np.zeros((n_vertices, n_nature))
        pi = np.full((n_vertices, n_nature), -1, dtype=int)
        for _ in range(max_sweeps):
            delta = 0.0
            for nu in reversed(order):
                previous_column = V[:, nu].copy() if max_sweeps > 1 else None
                total = np.zeros(len(src))

                for x, p in kernel[nu]:
                    if x == DONE:
                        onward = exit_values[:, nu]
                    else:
                        onward = V[:, x]
                    m_A, s_A = chunk_costs[(nu, x)]
                    if is_sum:
                        np.add(s_I, s_A, out=geo_buf)
                    else:
                        np.maximum(m_I, m_A, out=geo_buf)
                        geo_buf += w_sI
                        geo_buf += w * s_A
                    if is_time:
                        geo_buf *= dist_wt
                        geo_buf += duration

                    np.take(onward, dst, out=term_buf)
                    term_buf += geo_buf
                    term_buf *= p
                    total += term_buf

                column = np.full(n_vertices, np.inf)
                np.minimum.at(column, src, total)

                tol = 1e-12 + self.config.vi_tie_tol * np.abs(column[src])
                tie = np.nonzero(total <= column[src] + tol)[0]
                gi = s_I if is_sum else m_I
                tie = tie[np.argsort(-gi[tie], kind="stable")]
                choice = np.full(n_vertices, -1, dtype=int)
                choice[src[tie]] = dst[tie]

                V[:, nu], pi[:, nu] = column, choice
                V[blocked[:, nu], nu] = np.inf

                for v, values in inherited.items():
                    if blocked[v, nu]:
                        continue
                    V[v, nu], pi[v, nu] = values[nu], v
                pi[~np.isfinite(V[:, nu]), nu] = -1

                if previous_column is not None:
                    delta = max(delta, self._value_delta(V[:, nu], previous_column))

            n_iters += 1
            if delta < self.config.vi_tol:
                break

        _t4 = time.perf_counter()
        self.stats["t_vi"] = self.stats.get("t_vi", 0.0) + (_t4 - _t3)
        self.stats["vi_iterations"] = max(self.stats.get("vi_iterations", 0), n_iters)
        self.V_skill[mode], self.pi_skill[mode], self.blocked[mode] = V, pi, blocked

    def validated_chain(self, mode: Mode, node: int, max_repairs: int = 8) -> List[int]:
        """
        The pi_det chain out of `node`, with every edge on it proved collision-free. This is where
        det_edge_validation="execution" pays its bill
        """
        for _ in range(max_repairs + 1):
            nodes, policy = self.det_nodes[mode], self.pi_det[mode]
            if node >= len(nodes):
                return []
            chain, i, seen, blocked = [], int(node), set(), False
            while i >= 0 and i not in seen:
                seen.add(i)
                chain.append(i)
                j = int(policy[i])
                if j < 0:
                    break
                if not self._is_edge_collision_free(nodes[i], nodes[j], mode):
                    self.stats["n_chain_edges_blocked"] = (
                        self.stats.get("n_chain_edges_blocked", 0) + 1)
                    blocked = True
                    break
                i = j
            if not blocked:
                return chain
            self.stats["n_chain_repairs"] = self.stats.get("n_chain_repairs", 0) + 1
            self._connect_cache.pop(mode, None)
            self._solve_deterministic(mode)
        return chain

    def get_start_cost_to_go(self) -> float:
        """
        Returns the expected total cost-to-go from the initial start state. Queries the nearest roadmap
        node in the starting mode (deterministic or skill)
        """
        mode = self.env.get_start_mode()
        q = np.asarray(self.env.get_start_pos().state())
        if mode in self.V_skill:
            graph = self.roadmap.inactive_roadmaps[mode]
            _, nature, _ = self._mode_nature(mode)
            return float(self.V_skill[mode][self._find_nearest_vertex(graph, q[graph.indices]), nature.initial])
        nodes = self.det_nodes[mode]
        distances = [float(np.linalg.norm(np.asarray(n.state.q.state()) - q)) for n in nodes]
        return float(self.V_det[mode][int(np.argmin(distances))])
