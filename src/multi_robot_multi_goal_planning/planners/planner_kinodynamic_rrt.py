"""
Kinodynamic RRT / RRT* for multi-robot multi-goal problems.

Reuses the existing mode / mode-switch machinery (BaseModeLogic: is_transition / done /
get_next_modes) exactly like the geometric planners, but grows the tree by *propagating
dynamics* instead of straight-line interpolation.

Two regimes, selected by `optimize`:

  * optimize=False  -> feasible RRT: first solution wins, no rewiring.
  * optimize=True   -> RRT* with rewiring, *if* the environment provides an exact
    steering function (env.has_exact_steering). Rewiring needs to connect two given
    states exactly and know the cost; the double-integrator envs do (analytic steering),
    the car does not (feedback steering), so for the car optimize just keeps the best
    feasible solution.

Design notes (see kinodynamic_planning_env.py / kino_env.py):
  * Edges are curved, so every node stores the discretized trajectory that connects it to
    its parent (the geometric stack stores none).
  * Expansion prefers env.steer; if the env has none it falls back to random control
    propagation (env.propagate).
  * Mode switches happen wherever env.is_transition holds. Because mode-switch goals encode
    zero velocity (goal target has v = 0), reaching a transition means coming to rest. The
    transition configuration is duplicated into the next mode's tree as a root, mirroring
    the geometric RRT*. Cross-mode cost updates flow through the parent/child graph:
    transition nodes are children (in the next mode) of the node that reached them, so a
    rewire that lowers a node's cost propagates into every downstream mode automatically.
"""

import time
import random
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    Mode,
    BaseProblem,
    generate_binary_search_indices,
)
from multi_robot_multi_goal_planning.problems.core.goals import SingleGoal
from .termination_conditions import PlannerTerminationCondition


from functools import lru_cache


@lru_cache(maxsize=256)
def _binary_indices(n: int) -> Tuple[int, ...]:
    """Cached midpoint-first traversal order of range(n) (edge lengths repeat)."""
    return generate_binary_search_indices(n)


@dataclass
class KinodynamicRRTConfig:
    goal_bias: float = 0.15
    """Probability of sampling a mode-switch/terminal goal instead of a uniform state."""
    goal_region_bias: float = 0.15
    """Probability of *approaching* (trimmed) a sampled collision-free transition config, to
    densify the tree near it so a full goal-connection can then reach it."""
    goal_region_sigma: float = 0.25
    """Std-dev (position units) of the jitter applied to the active robots around their goal
    during goal-region sampling."""
    transition_sample_tries: int = 200
    """Attempts to find a collision-free transition config (active pinned, others free)."""
    transition_cache_size: int = 20
    """Number of collision-free transition configs cached per mode."""
    transition_resample_p: float = 0.2
    """Probability of sampling a fresh transition config instead of reusing a cached one."""
    pending_goal_bias: float = 0.6
    """When building a transition config, probability that each NON-active robot is placed at
    its own pending goal (where its next mode switch needs it) instead of a random collision-
    free pose. Makes idle robots pre-position toward their next task rather than wander, which
    removes random meandering and shortens the makespan."""
    max_extension_steps: int = 10
    """Max number of integration steps kept for a (non-goal) tree extension."""
    goal_steer_steps: int = 400
    """Max steps allowed when steering onto a goal (feedback controllers may need many)."""
    goal_snap_tol: float = 1e-6
    """If a goal-directed edge lands this close to a SingleGoal, snap onto it exactly."""
    expansion: str = "steer"
    """'steer' -> use env.steer with propagation fallback; 'propagate' -> force random controls."""
    num_control_samples: int = 12
    """Random controls tried per extension when propagating."""
    cost_metric: str = "time"
    """What the planner minimises: 'time' (trajectory duration) or 'length' (total spatial
    path length, summed over robots). Length directly penalises spatial meandering."""
    distance_metric: str = "position"
    """'position' (Euclidean over robot positions) or 'state' (full extended state)."""
    frontier_bias: float = 0.4
    """Probability of sampling the deepest-progress mode(s) instead of a uniform mode.
    Concentrates growth on advancing the task sequence -- important for long horizons."""
    # RRT* (optimize=True, exact-steering envs only)
    neighbor_radius: float = 1.0
    """(unused) the near radius is now the Karaman-Frazzoli shrinking ball; see _init_rrtstar_radius."""
    max_near: int = 20
    """Hard cap on the near-set size (keeps the near set bounded when the shrinking ball is
    still large at low node counts)."""
    shortcut_in_loop: bool = True
    """Like the geometric RRT*: whenever the solution improves, shortcut it (bounded) and
    re-inject the result into the tree, so shortcutting and rewiring compound."""
    loop_shortcut_iters: int = 300
    """Shortcut iterations per in-loop call. Shortcutting is far more cost-effective per
    second than rewiring (RRT* rewiring plateaus while shortcutting keeps cutting), so a
    healthy in-loop budget speeds convergence."""
    loop_shortcut_stride: int = 5
    """Node spacing used when re-injecting the shortcut path into the tree (avoids bloat)."""


class _Node:
    __slots__ = ("x", "mode", "parent", "children", "traj", "cost", "cost_to_parent", "pos", "is_transition")

    def __init__(self, x, mode, parent, traj, cost, cost_to_parent, pos, is_transition=False):
        self.x = x
        self.mode = mode
        self.parent = parent
        self.children: List["_Node"] = []
        self.traj = traj  # states from parent.x (exclusive) to x (inclusive); [] for a pure mode switch
        self.cost = cost
        self.cost_to_parent = cost_to_parent
        self.pos = pos
        self.is_transition = is_transition


class KinodynamicRRT:
    def __init__(self, env: BaseProblem, config: Optional[KinodynamicRRTConfig] = None):
        self.env = env
        self.config = config or KinodynamicRRTConfig()

        self.dt = getattr(env, "integration_dt", 0.05)
        self.state_bounds = env.state_bounds()
        self.control_bounds = env.control_bounds()
        self.exact_steering = bool(getattr(env, "has_exact_steering", False))
        self.cost_metric = self.config.cost_metric

        self._init_rrtstar_radius()

        # nodes bucketed by mode; nearest-neighbour is restricted to the same mode
        self.nodes_by_mode: Dict[Mode, List[_Node]] = {}
        # append-only contiguous position matrix per mode (node.pos is fixed once
        # added; rewiring only changes parent/cost). Avoids rebuilding
        # np.array([n.pos for n in nodes]) on every nearest/near query, which was
        # O(n) python-level work per iteration => O(n^2) over a run.
        self._pos_by_mode: Dict[Mode, NDArray] = {}
        self._pos_n: Dict[Mode, int] = {}
        self.modes: List[Mode] = []
        self.goal_nodes: List[_Node] = []
        self._transition_configs: Dict[Mode, List[NDArray]] = {}

        self.start_time = 0.0
        self.costs: List[float] = []
        self.times: List[float] = []
        self.all_paths: List[List[State]] = []

    # ------------------------------------------------------------------ helpers
    def _robot_positions(self, x: NDArray) -> NDArray:
        q = self.env.start_pos.from_flat(x)
        return np.concatenate(
            [np.asarray(self.env.robot_position(q, i)) for i in range(q.num_agents())]
        )

    def _add_node(self, node: _Node) -> None:
        m = node.mode
        if m not in self.nodes_by_mode:
            self.nodes_by_mode[m] = []
            self.modes.append(m)
            self._pos_by_mode[m] = np.empty((8, node.pos.shape[0]), dtype=np.float64)
            self._pos_n[m] = 0
        self.nodes_by_mode[m].append(node)
        buf = self._pos_by_mode[m]
        n = self._pos_n[m]
        if n >= buf.shape[0]:  # amortized doubling
            newbuf = np.empty((buf.shape[0] * 2, buf.shape[1]), dtype=np.float64)
            newbuf[:n] = buf[:n]
            self._pos_by_mode[m] = buf = newbuf
        buf[n] = node.pos
        self._pos_n[m] = n + 1

    def _pos_matrix(self, mode: Mode) -> NDArray:
        """Contiguous view of all node positions in `mode` (append-only cache)."""
        return self._pos_by_mode[mode][: self._pos_n[mode]]

    def _mode_progress(self, mode: Mode) -> int:
        if self.env.is_terminal_mode(mode):
            return 1 << 30
        get_idx = getattr(self.env, "get_current_seq_index", None)
        if get_idx is None:
            return 0
        try:
            return get_idx(mode)
        except Exception:
            return 0

    def _select_mode(self) -> Mode:
        # Frontier bias drives the search DOWN to a first solution. Once we have one, drop it
        # so RRT* densifies/rewires every mode on the path (otherwise only the deepest mode is
        # optimised and the earlier segments -- most of the cost -- never improve).
        fb = self.config.frontier_bias if not self.goal_nodes else 0.0
        if len(self.modes) > 1 and random.random() < fb:
            progresses = [self._mode_progress(m) for m in self.modes]
            mx = max(progresses)
            frontier = [m for m, p in zip(self.modes, progresses) if p == mx]
            return random.choice(frontier)
        return random.choice(self.modes)

    def _sample_state(self) -> NDArray:
        lo, hi = self.state_bounds[0], self.state_bounds[1]
        return np.random.uniform(lo, hi)

    def _active_goal(self, mode: Mode):
        task = self.env.get_active_task(mode, None)
        return task.robots, task.goal

    def _rest_state(self, x: NDArray) -> NDArray:
        """Zero all velocities (bring the system to rest). Only meaningful for the
        [pos, vel] double-integrator layout; a no-op otherwise."""
        pv = getattr(self.env, "_pos_vel_indices", None)
        if pv is not None:
            _, vel_idx = pv()
            x[vel_idx] = 0.0
        return x

    def _sample_transition_config(self, mode: Mode) -> Optional[NDArray]:
        """A collision-free full configuration with the active robots pinned at their goal
        (and at rest) and the NON-active robots sampled to be collision-free -- i.e. out of
        the active robot's way. This mirrors the geometric planner's
        sample_transition_configuration: the tree then grows toward these validated targets,
        instead of toward random configs (which forces pointless coupled motion) or the
        nearest node's config (which leaves other robots blocking)."""
        cache = self._transition_configs.setdefault(mode, [])
        if cache and random.random() > self.config.transition_resample_p:
            return cache[random.randrange(len(cache))]

        robots, goal = self._active_goal(mode)
        active_idx = {self.env.robots.index(r) for r in robots}
        goal_arr = np.asarray(goal.sample(mode), dtype=float)
        d = self.env.position_dim_per_robot
        for _ in range(self.config.transition_sample_tries):
            x = self._rest_state(self._sample_state())
            # non-active robots: pre-position at their own pending goal (bias) instead of random
            for r_idx in range(self.env.start_pos.num_agents()):
                if r_idx in active_idx:
                    continue
                if random.random() < self.config.pending_goal_bias:
                    pg = self._robot_pending_goal_pos(mode, r_idx)
                    if pg is not None:
                        s, _e = self.env.start_pos._array_slice[r_idx]
                        x[s : s + d] = pg  # velocity already zeroed by _rest_state
            # active robots -> their goal (overwrite)
            off = 0
            for rr in robots:
                r_idx = self.env.robots.index(rr)
                s, e = self.env.start_pos._array_slice[r_idx]
                dim = e - s
                x[s:e] = goal_arr[off : off + dim]
                off += dim
            if self.env.is_collision_free(self.env.start_pos.from_flat(x), mode):
                cache.append(x)
                if len(cache) > self.config.transition_cache_size:
                    cache.pop(0)
                return x
        return cache[random.randrange(len(cache))] if cache else None

    def _robot_pending_goal_pos(self, mode: Mode, r_idx: int) -> Optional[NDArray]:
        """Position (dim = position_dim_per_robot) of robot r's currently-assigned task goal --
        where its next mode switch will need it. Works whether the env's task goals are
        position-only (manipulation wrapper) or extended [pos, vel] (double integrator)."""
        r = self.env.robots[r_idx]
        task = self.env.tasks[mode.task_ids[r_idx]]
        if r not in task.robots:
            return None
        try:
            goal_arr = np.asarray(task.goal.sample(mode), dtype=float)
        except Exception:
            return None
        per = len(goal_arr) // len(task.robots)
        j = task.robots.index(r)
        return goal_arr[j * per : j * per + self.env.position_dim_per_robot]

    def _sample_move(self, mode: Mode) -> Tuple[NDArray, bool, _Node]:
        """Pick (target, reach_goal, nearest_node) for one iteration: uniform exploration,
        or grow toward a sampled collision-free transition config (goal / goal-region)."""
        r = random.random()
        if r >= self.config.goal_bias + self.config.goal_region_bias:
            target = self._sample_state()  # uniform exploration
            return target, False, self._nearest(mode, target)

        tc = self._sample_transition_config(mode)
        if tc is None:
            target = self._sample_state()
            return target, False, self._nearest(mode, target)

        if r < self.config.goal_bias:
            return tc.copy(), True, self._nearest(mode, tc)  # reach the transition config exactly

        # goal-region: keep the (collision-free) non-active robots from tc, but JITTER the
        # active robots around their goal. This densifies the goal's neighbourhood so a
        # free-side node next to an obstacle-shadowed goal is found (the exact steer can
        # then reach the goal from there). Both parts matter: non-active out of the way AND
        # active-neighbourhood exploration.
        target = tc.copy()
        d = self.env.position_dim_per_robot
        robots, _ = self._active_goal(mode)
        for rr in robots:
            r_idx = self.env.robots.index(rr)
            s, _e = self.env.start_pos._array_slice[r_idx]
            target[s : s + d] += np.random.normal(0.0, self.config.goal_region_sigma, size=d)
        return target, False, self._nearest(mode, target)

    def _dist_key(self, x):
        return x if self.config.distance_metric == "state" else self._robot_positions(x)

    def _nearest(self, mode: Mode, target: NDArray) -> _Node:
        nodes = self.nodes_by_mode[mode]
        if self.config.distance_metric == "state":
            arr = np.array([n.x for n in nodes])
            key = target
        else:
            arr = self._pos_matrix(mode)
            key = self._robot_positions(target)
        dists = np.linalg.norm(arr - key, axis=1)
        return nodes[int(np.argmin(dists))]

    def _init_rrtstar_radius(self):
        """Karaman & Frazzoli RRT* shrinking-ball constant (same as the geometric planner's
        set_gamma_rrtstar), measured in the position space that nearest/near operate on. A
        fixed radius is wrong: too small => nothing to rewire, too large => the near set (and
        collision cost) blows up. The gamma below makes r(n) = min(gamma*(log n/n)^(1/d), eta)
        shrink correctly with the node count and adapt to the dimension / C-space volume."""
        d = self.env.start_pos.num_agents() * self.env.position_dim_per_robot
        c_free = 1.0
        dp = self.env.position_dim_per_robot
        for i in range(self.env.start_pos.num_agents()):
            s, _e = self.env.start_pos._array_slice[i]
            for k in range(dp):
                c_free *= float(self.state_bounds[1][s + k] - self.state_bounds[0][s + k])
        unit_ball = math.pi ** (d / 2.0) / math.gamma(d / 2.0 + 1.0)
        self._rrt_d = d
        self._rrt_gamma = (2 * (1 + 1.0 / d)) ** (1.0 / d) * (c_free / unit_ball) ** (1.0 / d)
        self._rrt_eta = math.sqrt(d)  # cap, like the geometric planner's stepsize fallback

    def _neighbor_radius(self, n: int) -> float:
        n = max(n, 2)
        return min(self._rrt_gamma * (math.log(n) / n) ** (1.0 / self._rrt_d), self._rrt_eta)

    def _near(self, mode: Mode, pos: NDArray, nearest: _Node) -> List[_Node]:
        nodes = self.nodes_by_mode[mode]
        arr = self._pos_matrix(mode)
        dists = np.linalg.norm(arr - pos, axis=1)
        radius = self._neighbor_radius(len(nodes))
        k = self.config.max_near
        within = np.flatnonzero(dists <= radius)
        if within.size == 0:
            # nothing inside the ball => keep just the single closest node
            idxs = [int(np.argmin(dists))]
        elif within.size > k:
            # k closest among the in-radius set (argpartition avoids a full sort)
            part = within[np.argpartition(dists[within], k)[:k]]
            idxs = part[np.argsort(dists[part])].tolist()
        else:
            idxs = within[np.argsort(dists[within])].tolist()
        out = [nodes[i] for i in idxs]
        if nearest not in out:
            out.append(nearest)
        return out

    def _propagate_toward(self, x_near: NDArray, target: NDArray) -> Optional[List[NDArray]]:
        lo, hi = self.control_bounds[0], self.control_bounds[1]
        target_pos = self._robot_positions(target)
        best, best_d = None, math.inf
        for _ in range(self.config.num_control_samples):
            u = np.random.uniform(lo, hi)
            steps = random.randint(1, self.config.max_extension_steps)
            roll = self.env.propagate(x_near, u, num_steps=steps)
            d = np.linalg.norm(self._robot_positions(roll[-1]) - target_pos)
            if d < best_d:
                best_d, best = d, roll
        return best

    def _expand(self, x_near: NDArray, target: NDArray, reach_goal: bool) -> Optional[List[NDArray]]:
        traj = None
        if self.config.expansion == "steer":
            max_steps = self.config.goal_steer_steps if reach_goal else self.config.max_extension_steps
            traj = self.env.steer(x_near, target, max_steps=max_steps)
        if traj is None:
            traj = self._propagate_toward(x_near, target)
        if traj is None or len(traj) == 0:
            return None
        if not reach_goal and len(traj) > self.config.max_extension_steps:
            traj = traj[: self.config.max_extension_steps]
        return traj

    def _within_bounds(self, x: NDArray) -> bool:
        lo, hi = self.state_bounds[0], self.state_bounds[1]
        return bool(np.all(x >= lo - 1e-9) and np.all(x <= hi + 1e-9))

    def _traj_valid(self, traj: List[NDArray], mode: Mode) -> bool:
        n = len(traj)
        if n == 0:
            return True
        # bounds for the whole trajectory in one vectorized pass (two np.all calls
        # total instead of two per waypoint)
        arr = np.asarray(traj)
        lo, hi = self.state_bounds[0], self.state_bounds[1]
        if not (np.all(arr >= lo - 1e-9) and np.all(arr <= hi + 1e-9)):
            return False
        from_flat = self.env.start_pos.from_flat
        is_free = self.env.is_collision_free
        # Midpoint-first (binary) order, matching the geometric planner's
        # is_path_collision_free(binary_order=True): a colliding edge usually
        # collides in its *middle/end* (its start is a validated tree node), so a
        # sequential scan wastes checks on the free prefix before hitting it.
        # Binary order samples the whole edge coarsely first -> earlier exit on
        # rejects. Same set of points, just reordered => no missed collisions.
        for i in _binary_indices(n):
            if not is_free(from_flat(traj[i]), mode):
                return False
        return True


    def _traj_length(self, x_from: NDArray, traj: List[NDArray]) -> float:
        """Total spatial path length of an edge: sum over robots of each robot's position
        travel along [x_from, *traj]."""
        n = self.env.start_pos.num_agents()
        d = self.env.position_dim_per_robot
        prev = self._robot_positions(x_from).reshape(n, d)
        total = 0.0
        for x in traj:
            cur = self._robot_positions(x).reshape(n, d)
            total += float(np.sum(np.linalg.norm(cur - prev, axis=1)))
            prev = cur
        return total

    def _edge_cost(self, x_from: NDArray, traj: List[NDArray]) -> float:
        """Cost of a (collision-free) edge under the chosen metric."""
        if self.cost_metric == "length":
            return self._traj_length(x_from, traj)
        return len(traj) * self.dt

    def _path_cost(self, path: List[State]) -> float:
        if self.cost_metric == "length":
            return _path_pos_length(self.env, path)
        return _path_duration(path, self.dt)

    def _straight_line_length(self, x_from: NDArray, x_to: NDArray) -> float:
        """Sum over robots of the straight-line position distance -- a cheap LOWER BOUND on
        the steered edge length, used only for RRT* ranking (exact length is computed on the
        accepted edge)."""
        n = self.env.start_pos.num_agents()
        d = self.env.position_dim_per_robot
        a = self._robot_positions(x_from).reshape(n, d)
        b = self._robot_positions(x_to).reshape(n, d)
        return float(np.sum(np.linalg.norm(b - a, axis=1)))

    def _steer_cost(self, x_from: NDArray, x_to: NDArray) -> Optional[float]:
        """Cheap RANKING cost (no trajectory sampling / no collision). For 'time' it is the
        exact edge time (horizon); for 'length' it is the straight-line lower bound. Both are
        lower bounds on / equal to the true edge cost, so ranking by them and computing the
        exact cost on accepted edges stays correct."""
        if self.cost_metric == "length":
            return self._straight_line_length(x_from, x_to)
        sc = getattr(self.env, "steer_cost", None)
        if sc is not None:
            return sc(x_from, x_to)
        traj = self.env.steer(x_from, x_to, max_steps=None)
        return None if not traj else self._edge_cost(x_from, traj)

    def _collision_free_edge(self, x_from: NDArray, x_to: NDArray, mode: Mode) -> Optional[List[NDArray]]:
        """Generate the steering trajectory and collision-check it (the expensive step,
        only run on edges we actually want to accept). Returns the trajectory or None."""
        traj = self.env.steer(x_from, x_to, max_steps=None)
        if traj is None or len(traj) == 0 or np.linalg.norm(traj[-1] - x_to) > 1e-6:
            return None
        if not self._traj_valid(traj, mode):
            return None
        return traj

    def _maybe_snap_goal(self, x: NDArray, mode: Mode) -> NDArray:
        robots, goal = self._active_goal(mode)
        if not isinstance(goal, SingleGoal):
            return x
        goal_arr = np.asarray(goal.goal, dtype=float)
        x = x.copy()
        offset = 0
        for r in robots:
            r_idx = self.env.robots.index(r)
            s, e = self.env.start_pos._array_slice[r_idx]
            dim = e - s
            if np.linalg.norm(x[s:e] - goal_arr[offset : offset + dim]) < self.config.goal_snap_tol:
                x[s:e] = goal_arr[offset : offset + dim]
            offset += dim
        return x

    # ------------------------------------------------------------------ cost / rewire
    def _update_cost(self, node: _Node) -> None:
        stack = [node]
        while stack:
            cur = stack.pop()
            for ch in cur.children:
                ch.cost = cur.cost + ch.cost_to_parent
                stack.append(ch)

    def _is_ancestor(self, a: _Node, b: _Node) -> bool:
        """Is `a` an ancestor of `b` (following parent links, across modes)?"""
        n = b
        while n is not None:
            if n is a:
                return True
            n = n.parent
        return False

    def _handle_mode_events(self, node: _Node) -> None:
        q = self.env.start_pos.from_flat(node.x)
        if self.env.done(q, node.mode):
            self.goal_nodes.append(node)
        elif self.env.is_transition(q, node.mode):
            for nm in self.env.get_next_modes(q, node.mode):
                trans = _Node(node.x, nm, node, [], node.cost, 0.0, node.pos.copy(), is_transition=True)
                node.children.append(trans)
                self._add_node(trans)

    # ------------------------------------------------------------------ path build
    def _build_path(self, goal_node: _Node) -> List[State]:
        chain: List[_Node] = []
        n = goal_node
        while n is not None:
            chain.append(n)
            n = n.parent
        chain.reverse()

        path: List[State] = []
        start = chain[0]
        path.append(State(self.env.start_pos.from_flat(start.x), start.mode))
        for node in chain[1:]:
            for x in node.traj:
                path.append(State(self.env.start_pos.from_flat(x), node.mode))
            if not node.traj:  # pure mode-switch: double the state into the new mode
                path.append(State(self.env.start_pos.from_flat(node.x), node.mode))
        return path

    def _best_goal(self) -> Optional[_Node]:
        if not self.goal_nodes:
            return None
        return min(self.goal_nodes, key=lambda g: g.cost)

    def _record(self, cost: float, path: List[State]) -> None:
        self.costs.append(cost)
        self.times.append(time.time() - self.start_time)
        self.all_paths.append(path)

    # ------------------------------------------------------------------ main loop
    def plan(
        self,
        ptc: PlannerTerminationCondition,
        optimize: bool = False,
    ) -> Tuple[Optional[List[State]], Dict[str, Any]]:
        self.start_time = time.time()
        do_rewire = optimize and self.exact_steering

        start_mode = self.env.get_start_mode()
        x_start = np.asarray(self.env.start_pos.state(), dtype=float)
        start_node = _Node(x_start, start_mode, None, [], 0.0, 0.0, self._robot_positions(x_start))
        self._add_node(start_node)
        self._start_node = start_node

        best_reported = math.inf
        first_solution_seen = False
        i = 0
        while True:
            i += 1
            if ptc.should_terminate(i, time.time() - self.start_time):
                break

            mode = self._select_mode()
            target, reach_goal, n_near = self._sample_move(mode)

            traj = self._expand(n_near.x, target, reach_goal)
            if traj is None:
                continue
            if reach_goal:
                traj = list(traj)
                traj[-1] = self._maybe_snap_goal(traj[-1], mode)
            if not self._traj_valid(traj, mode):
                continue

            x_new = traj[-1]
            pos_new = self._robot_positions(x_new)

            # RRT* find_parent/rewire is for OPTIMISATION; it is not needed to find a first
            # solution and its (rai) collision cost slows the search. So expand feasibly until
            # a solution exists, then switch on rewiring to drive the cost down. This keeps the
            # first-solution as reliable/fast as the feasible planner while still converging.
            if do_rewire and self.goal_nodes:
                new_node = self._add_rrtstar(mode, x_new, pos_new, n_near, traj)
            else:
                edge = self._edge_cost(n_near.x, traj)
                new_node = _Node(x_new, mode, n_near, traj, n_near.cost + edge, edge, pos_new)
                n_near.children.append(new_node)
                self._add_node(new_node)

            if new_node is None:
                continue

            self._handle_mode_events(new_node)

            best = self._best_goal()
            if best is not None and best.cost < best_reported - 1e-9:
                best_reported = best.cost
                path = self._build_path(best)
                self._record(best.cost, path)
                print(f"[kino-rrt{'*' if do_rewire else ''}] solution at iter {i}: cost={best.cost:.3f}")
                if not optimize:
                    break
                # geometric-style: shortcut the improved path and re-inject it into the
                # tree. Deferred until AFTER the first solution: the first solution is
                # reached by pure feasible search (no rewire, no shortcut) so time-to-
                # first-solution is not spent on the (expensive) shortcut; shortcutting
                # then engages on every subsequent improvement to drive convergence.
                if do_rewire and self.config.shortcut_in_loop and first_solution_seen:
                    best_reported = self._shortcut_and_reinject(path, best_reported)
                first_solution_seen = True

        best = self._best_goal()
        if best is None:
            print(f"[kino-rrt] no solution found ({i} iterations)")
            return None, {"costs": self.costs, "times": self.times, "paths": self.all_paths}

        best_path = self._build_path(best)
        info = {"costs": self.costs, "times": self.times, "paths": self.all_paths}
        return best_path, info

    def _add_rrtstar(self, mode, x_new, pos_new, n_near, traj) -> Optional[_Node]:
        """RRT* insertion, structured like the geometric planner's find_parent/rewire:
        rank candidates by (cheap, collision-free) cost, and collision-check LAZILY -- for
        the parent, in cost order until the first collision-free edge; for rewiring, only the
        cost-improving edges. This keeps the expensive (rai) collision checks to ~1-3 per
        iteration instead of ~2*max_near."""
        near = self._near(mode, pos_new, n_near)

        # the extension edge from the nearest node is already collision-free -> the fallback
        fb_edge = self._edge_cost(n_near.x, traj)
        best_parent, best_traj, best_edge, best_cost = n_near, traj, fb_edge, n_near.cost + fb_edge

        # choose parent: rank by cost (cheap), collision-check in cost order, take first free
        cands = []
        for nb in near:
            if nb is n_near:
                continue
            c = self._steer_cost(nb.x, x_new)
            if c is not None:
                cands.append((nb.cost + c, nb, c))
        cands.sort(key=lambda t: t[0])
        for total, nb, edge_cost in cands:
            # `total` uses the ranking cost, a lower bound on the true cost, so once it exceeds
            # the best exact cost no remaining (sorted) candidate can beat the fallback.
            if total >= best_cost:
                break
            edge = self._collision_free_edge(nb.x, x_new, mode)
            if edge is not None:
                exact = self._edge_cost(nb.x, edge)  # exact (curved) edge cost
                if nb.cost + exact < best_cost:
                    best_parent, best_traj, best_edge, best_cost = nb, edge, exact, nb.cost + exact
                break  # first collision-free in cost order (like the geometric find_parent)

        new_node = _Node(x_new, mode, best_parent, best_traj, best_cost, best_edge, pos_new)
        best_parent.children.append(new_node)
        self._add_node(new_node)

        # rewire: cheap cost-improvement filter first, collision-check only the winners
        for nb in near:
            if nb is best_parent or nb.is_transition or nb.parent is None:
                continue
            if self._is_ancestor(nb, new_node):
                continue
            c = self._steer_cost(x_new, nb.x)  # lower bound
            if c is None or new_node.cost + c >= nb.cost - 1e-9:
                continue
            edge = self._collision_free_edge(x_new, nb.x, mode)
            if edge is None:
                continue
            exact = self._edge_cost(x_new, edge)
            if new_node.cost + exact >= nb.cost - 1e-9:
                continue
            nb.parent.children.remove(nb)
            nb.parent = new_node
            new_node.children.append(nb)
            nb.traj = edge
            nb.cost_to_parent = exact
            nb.cost = new_node.cost + exact
            self._update_cost(nb)

        return new_node

    # ------------------------------------------------------------------ in-loop shortcut
    def _shortcut_and_reinject(self, path: List[State], best_reported: float) -> float:
        """Per-robot shortcut the improved solution (bounded) and re-inject it into the tree
        as a node chain -- mirrors the geometric planner's robot_mode_shortcut + tree_extension.
        The per-robot (fixed-time) shortcut keeps the trajectory TIME unchanged but straightens
        individual robots, so re-injecting it gives rewiring better-placed nodes to reconnect;
        the time-cost reductions then come from the subsequent min-time rewiring. Returns the
        (possibly lower) best cost."""
        sc_path, _ = kinodynamic_robot_shortcut(
            self.env, path, max_iter=self.config.loop_shortcut_iters, verbose=False
        )
        # only bother re-injecting if the per-robot pass actually straightened something
        if _path_pos_length(self.env, sc_path) >= _path_pos_length(self.env, path) - 1e-9:
            return best_reported
        goal = self._reinject_path(sc_path)
        if goal is None:
            return best_reported
        sc_cost = self._path_cost(sc_path)
        if sc_cost < best_reported - 1e-9:
            self._record(sc_cost, sc_path)
            print(f"[kino-rrt* + shortcut] cost={sc_cost:.3f}")
            return sc_cost
        return best_reported

    def _reinject_path(self, path: List[State]) -> Optional[_Node]:
        """Add `path` back into the tree as a chain of nodes (with the exact sub-trajectories
        as edges), rooted at the start node. The chain's goal becomes a (cheaper) goal node
        and its nodes become rewire anchors for later iterations."""
        dt = self.dt
        stride = max(1, self.config.loop_shortcut_stride)
        prev = self._start_node
        seg: List[NDArray] = []  # states accumulated since the last placed node

        def place(mode: Mode, is_trans: bool = False) -> _Node:
            nonlocal prev, seg
            if is_trans:
                traj: List[NDArray] = []
                x = prev.x
            else:
                traj = seg
                x = seg[-1]
            edge = 0.0 if is_trans else self._edge_cost(prev.x, traj)
            node = _Node(
                np.asarray(x, dtype=float), mode, prev, list(traj),
                prev.cost + edge, edge, self._robot_positions(np.asarray(x, dtype=float)),
                is_transition=is_trans,
            )
            prev.children.append(node)
            self._add_node(node)
            prev = node
            seg = []
            return node

        for i in range(1, len(path)):
            if path[i].mode == path[i - 1].mode:
                seg.append(path[i].q.state())
                if len(seg) >= stride:
                    place(path[i].mode)
            else:  # doubled mode-switch state: flush remainder, then a transition node
                if seg:
                    place(path[i - 1].mode)
                place(path[i].mode, is_trans=True)
        if seg:
            place(path[-1].mode)

        if prev is self._start_node:
            return None
        if self.env.done(self.env.start_pos.from_flat(prev.x), prev.mode):
            self.goal_nodes.append(prev)
        return prev


# ---------------------------------------------------------------------------
# Kinodynamic shortcutting (post-processing)
#
# Inspiration is taken from planners/shortcutting.py::single_mode_shortcut: sample a
# random index pair (i, j) in the same mode, and if a direct reconnection is cheaper and
# collision-free, replace the sub-path. The only difference is the reconnection: instead of
# straight-line interpolation (meaningless under dynamics) we use the env's EXACT steering
# function. Because steering starts at x_i and ends at x_j exactly -- including velocity --
# the shortcut splices back into the path with full state (and hence velocity) continuity,
# and the whole path stays dynamically feasible. The cost being reduced is trajectory time
# (number of integration steps), which is exactly what the kinodynamic RRT/RRT* minimises.
# ---------------------------------------------------------------------------


def _path_duration(path: List[State], dt: float) -> float:
    return dt * sum(1 for a, b in zip(path[:-1], path[1:]) if a.mode == b.mode)


def kinodynamic_shortcut(
    env: BaseProblem,
    path: List[State],
    max_iter: int = 2000,
    time_budget: Optional[float] = None,
    verbose: bool = True,
):
    """Same-mode exact-steering shortcutting. Requires env.has_exact_steering (the
    double-integrator envs); returns (new_path, [costs, times]).

    All robots are reconnected simultaneously in (near) minimum time, which straightens
    the meandering that a feasible/limited-time RRT* leaves behind."""
    if not getattr(env, "has_exact_steering", False):
        if verbose:
            print("[kino-shortcut] env has no exact steering; skipping")
        return path, [[_path_duration(path, getattr(env, "integration_dt", 0.05))], [0.0]]

    dt = getattr(env, "integration_dt", 0.05)
    new_path = list(path)

    costs = [_path_duration(new_path, dt)]
    times = [0.0]
    start_time = time.time()
    cnt = 0

    for _ in range(max_iter):
        if time_budget is not None and time.time() - start_time > time_budget:
            break

        n = len(new_path)
        if n < 4:
            break

        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i > j:
            i, j = j, i
        if j - i < 2:
            continue

        mode = new_path[i].mode
        if new_path[j].mode != mode:
            continue  # never shortcut across a mode switch (steering is per-mode)

        x_i = np.asarray(new_path[i].q.state(), dtype=float)
        x_j = np.asarray(new_path[j].q.state(), dtype=float)

        traj = env.steer(x_i, x_j, max_steps=None)
        if traj is None or len(traj) == 0:
            continue
        if np.linalg.norm(traj[-1] - x_j) > 1e-6:
            continue
        if len(traj) >= (j - i):  # not shorter in time
            continue

        # collision + bounds along the shortcut (endpoints are already on the valid path)
        lo, hi = env.state_bounds()[0], env.state_bounds()[1]
        ok = True
        interior = traj[:-1]
        for idx in _binary_indices(len(interior)):  # midpoint-first for early reject
            x = interior[idx]
            if not (np.all(x >= lo - 1e-9) and np.all(x <= hi + 1e-9)):
                ok = False
                break
            if not env.is_collision_free(env.start_pos.from_flat(x), mode):
                ok = False
                break
        if not ok:
            continue

        new_seg = [new_path[i]] + [State(env.start_pos.from_flat(x), mode) for x in traj]
        new_path[i : j + 1] = new_seg
        cnt += 1

        costs.append(_path_duration(new_path, dt))
        times.append(time.time() - start_time)

    if verbose:
        print(
            f"[kino-shortcut] accepted {cnt} shortcuts, "
            f"cost {costs[0]:.3f} -> {costs[-1]:.3f}"
        )
    return new_path, [costs, times]


def _robot_seg_pos_length(env, seg: List[State], r: int) -> float:
    total = 0.0
    for a, b in zip(seg[:-1], seg[1:]):
        if a.mode != b.mode:
            continue
        pa = np.asarray(env.robot_position(a.q, r))
        pb = np.asarray(env.robot_position(b.q, r))
        total += float(np.linalg.norm(pb - pa))
    return total


def _path_pos_length(env, path: List[State]) -> float:
    return sum(_robot_seg_pos_length(env, path, r) for r in range(len(env.robots)))


def kinodynamic_robot_shortcut(
    env: BaseProblem,
    path: List[State],
    max_iter: int = 2000,
    time_budget: Optional[float] = None,
    verbose: bool = True,
):
    """Partial (per-robot) exact-steering shortcutting -- the kinodynamic analogue of
    planners/shortcutting.py::robot_mode_shortcut.

    One robot is re-routed at a time; the others keep their trajectory. Because a single
    robot's timeline must stay aligned with the rest, the robot is re-steered over the
    *same* time span with a FIXED-time connection (env.steer_fixed_time), not a minimum-time
    one. This lets the shortcut cross mode boundaries as long as the chosen robot is doing
    the *same task* throughout the span (so its own zero-velocity mode switch is never
    violated). The cost reduced here is per-robot spatial path length / control effort
    (the time cost is fixed by the mode-switch schedule), which removes the single-robot
    excursions that full-composite shortcutting cannot touch.

    Doubled mode-switch states (same configuration, adjacent modes, zero elapsed time) are
    handled by mapping path indices to real integration-step indices, so both halves of a
    doubled state receive the same re-steered slice and the invariant is preserved."""
    if not getattr(env, "has_exact_steering", False) or not hasattr(env, "steer_fixed_time"):
        if verbose:
            print("[kino-robot-shortcut] env has no fixed-time exact steering; skipping")
        return path, [[_path_pos_length(env, path)], [0.0]]

    d = env.position_dim_per_robot
    u_max = env.u_max
    n_robots = len(env.robots)
    new_path = list(path)
    n = len(new_path)

    # ---- per-robot position / length cache (built once, updated on the rare ACCEPT) ----
    # This function dominated the whole-planner profile because every ATTEMPT recomputed
    # the original span length (O(span) robot_position calls + norms) and built the full
    # State segment before knowing whether it improved anything. Instead:
    #   pos[r, k]      robot r's position at path state k (position-first slice layout,
    #                  guaranteed for every exact-steering env: robot_position == q[r][:d])
    #   edge_len[r, k] robot r's length of edge (k, k+1); 0 across doubled mode switches
    #   cum[r]         prefix sums, so span length over [i, j] is cum[r, j] - cum[r, i]
    # A shortcut replaces states strictly inside [i, j], so n and all indices are stable.
    slices = [tuple(env.start_pos._array_slice[r]) for r in range(n_robots)]
    pos = np.empty((n_robots, n, d))
    for k, st in enumerate(new_path):
        xk = st.q.state()
        for r in range(n_robots):
            s, _e = slices[r]
            pos[r, k] = xk[s : s + d]
    same_mode = np.array(
        [new_path[k].mode == new_path[k + 1].mode for k in range(n - 1)], dtype=bool
    )
    edge_len = np.linalg.norm(np.diff(pos, axis=1), axis=2)  # (n_robots, n-1)
    edge_len[:, ~same_mode] = 0.0
    cum = np.concatenate([np.zeros((n_robots, 1)), np.cumsum(edge_len, axis=1)], axis=1)
    total_len = float(cum[:, -1].sum())

    costs = [total_len]
    times = [0.0]
    start_time = time.time()
    cnt = 0
    rr = 0

    for _ in range(max_iter):
        if time_budget is not None and time.time() - start_time > time_budget:
            break
        if n < 4:
            break

        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i > j:
            i, j = j, i
        if j - i < 2:
            continue

        r = rr % n_robots
        rr += 1

        # cheap lower bound BEFORE any steering/collision work: no re-steered segment can
        # be shorter than the straight line between its endpoints, so if even that does
        # not beat the current span length the attempt is hopeless. On mostly-converged
        # paths this rejects the vast majority of attempts at O(1) cost.
        orig_len = float(cum[r, j] - cum[r, i])
        if float(np.linalg.norm(pos[r, j] - pos[r, i])) >= orig_len - 1e-9:
            continue

        # the chosen robot must be doing the same task across the whole span, so its own
        # zero-velocity mode switch is never inside the re-steered segment
        task_r = new_path[i].mode.task_ids[r]
        if any(new_path[i + k].mode.task_ids[r] != task_r for k in range(j - i + 1)):
            continue

        # map each path index to a real integration-step index (doubled switches take 0 time)
        t_index = [0]
        for k in range(1, j - i + 1):
            step = 0 if new_path[i + k - 1].mode != new_path[i + k].mode else 1
            t_index.append(t_index[-1] + step)
        num_real_steps = t_index[-1]
        if num_real_steps < 1:
            continue

        x_i = np.asarray(new_path[i].q.state(), dtype=float)
        x_j = np.asarray(new_path[j].q.state(), dtype=float)

        sub_states, peak = env.steer_fixed_time(x_i, x_j, num_real_steps)
        # feasibility only for the chosen robot's control block
        if np.any(peak[r * d : (r + 1) * d] > u_max + 1e-6):
            continue

        s, e = slices[r]

        # candidate robot-r positions straight from the raw steered states -- the State
        # segment (from_flat + copies) is only built AFTER the improvement test passes
        ti = np.asarray(t_index)
        sub_arr = np.asarray(sub_states)
        cand = np.empty((ti.size, d))
        zm = ti == 0  # doubled-switch prefix keeps the original (identical) configs
        if zm.any():
            cand[zm] = pos[r, i + np.flatnonzero(zm)]
        cand[~zm] = sub_arr[ti[~zm] - 1, s : s + d]
        # doubled switches inside the span repeat the same steered state => diff 0, so no
        # mode-boundary masking is needed here
        cand_edges = np.linalg.norm(np.diff(cand, axis=0), axis=1)
        cand_len = float(cand_edges.sum())
        if cand_len >= orig_len - 1e-9:
            continue

        # bounds: only robot r's slice changed (the rest of each state is already on the
        # valid path), and the steered states are stacked -> one vectorized check
        lo, hi = env.state_bounds()[0], env.state_bounds()[1]
        block = sub_arr[:, s:e]
        if not (np.all(block >= lo[s:e] - 1e-9) and np.all(block <= hi[s:e] + 1e-9)):
            continue

        seg = []
        for k in range(j - i + 1):
            base = new_path[i + k]
            q = base.q.state().copy()
            tk = t_index[k]
            if tk > 0:
                q[s:e] = sub_states[tk - 1][s:e]
            # tk == 0 keeps x_i's slice (already in q)
            seg.append(State(env.start_pos.from_flat(q), base.mode))

        # collision-check only the states we actually changed (interior points),
        # midpoint-first for early reject (bounds already vectorized above)
        ok = True
        for off in _binary_indices(max(j - i - 1, 0)):
            k = 1 + off
            if not env.is_collision_free(seg[k].q, seg[k].mode):
                ok = False
                break
        if not ok:
            continue

        new_path[i : j + 1] = seg
        # incremental cache update: only robot r's positions inside (i, j) changed, so
        # only its edges i..j-1 changed; endpoints (hence edges i-1 and j) are untouched
        pos[r, i : j + 1] = cand
        edge_len[r, i:j] = cand_edges
        cum[r, 1:] = np.cumsum(edge_len[r])
        total_len += cand_len - orig_len
        cnt += 1
        costs.append(total_len)
        times.append(time.time() - start_time)

    if verbose:
        print(
            f"[kino-robot-shortcut] accepted {cnt} shortcuts, "
            f"pos-length {costs[0]:.3f} -> {costs[-1]:.3f}"
        )
    return new_path, [costs, times]
