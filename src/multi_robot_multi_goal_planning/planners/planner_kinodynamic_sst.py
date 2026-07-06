"""
SST (Stable Sparse RRT) for kinodynamic multi-robot multi-goal problems.

For systems WITHOUT an exact steering function (the car, the trailer, anything you can only
propagate), plain RRT is neither optimal nor -- with a goal-seeking feedback controller --
even probabilistically complete over the full state space (that controller parks at every
node, so the tree never explores velocity). SST fixes both using ONLY random control
propagation:

  * BestNear selection: among tree nodes within delta_bn of a sampled state, expand the
    LOWEST-cost one -> pulls growth toward cheap paths (asymptotic near-optimality).
  * Monte-Carlo propagation: apply a random control for a random duration (no steering).
  * Witness domination + pruning: keep at most one node per delta_s ball (its cheapest
    representative), pruning dominated leaves -> a sparse, monotonically-improving tree.

It reuses the same mode / mode-switch machinery as the RRT planner (is_transition / done /
get_next_modes): a node that satisfies is_transition seeds the next mode's SST tree as a
protected root. Anytime: the reported best-goal cost decreases over time.

Reference: Li, Littlefield, Bekris, "Asymptotically Optimal Sampling-based Kinodynamic
Planning" (IJRR 2016).
"""

import time
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.planning_env import State, Mode, BaseProblem
from .termination_conditions import PlannerTerminationCondition


@dataclass
class KinodynamicSSTConfig:
    goal_bias: float = 0.1
    """Probability of sampling around the active-task goal (drives selection toward it)."""
    delta_bn: float = 0.5
    """BestNear selection radius (full-state metric, angles wrapped)."""
    delta_s: float = 0.25
    """Witness / pruning radius. Smaller -> denser tree, better cost, slower."""
    min_prop_steps: int = 1
    max_prop_steps: int = 12
    """Random propagation duration is uniform in [min, max] integration steps. Shorter
    durations = finer control = better cost (at the cost of more iterations)."""
    frontier_bias: float = 0.4
    """Bias mode selection toward the deepest-progress mode(s) (advances the task sequence)."""
    hold_prob: float = 0.3
    """Per-robot probability of applying ZERO control (hold) in a propagation. A robot at
    rest with zero control stays at rest, so robots parked at their goals can hold while
    another manoeuvres -- needed to reach the shared zero-velocity terminal. Unlike strict
    one-robot-at-a-time actuation, every robot may still move together (prob (1-hold_prob)^n),
    so coordinated/simultaneous motion is sampled and completeness+optimality are preserved."""
    cost_metric: str = "time"
    """'time' (duration) or 'length' (summed spatial path length)."""


class _Node:
    __slots__ = ("x", "mode", "parent", "children", "cost", "is_rep", "protected")

    def __init__(self, x, mode, parent, cost, protected=False):
        self.x = x
        self.mode = mode
        self.parent = parent
        self.children: List["_Node"] = []
        self.cost = cost
        self.is_rep = True       # every added node starts as its witness's representative
        self.protected = protected  # start / transition nodes are never pruned


class _Witness:
    __slots__ = ("x", "rep")

    def __init__(self, x, rep):
        self.x = x
        self.rep = rep


class KinodynamicSST:
    def __init__(self, env: BaseProblem, config: Optional[KinodynamicSSTConfig] = None):
        self.env = env
        self.config = config or KinodynamicSSTConfig()
        self.dt = getattr(env, "integration_dt", 0.05)
        self.state_bounds = env.state_bounds()
        self.control_bounds = env.control_bounds()
        self.cost_metric = self.config.cost_metric
        self.angular_dims = list(getattr(env, "angular_dims", []) or [])

        self.nodes_by_mode: Dict[Mode, List[_Node]] = {}
        self.witnesses_by_mode: Dict[Mode, List[_Witness]] = {}
        self.modes: List[Mode] = []
        self.goal_nodes: List[_Node] = []

        self.start_time = 0.0
        self.costs: List[float] = []
        self.times: List[float] = []
        self.all_paths: List[List[State]] = []

    # ------------------------------------------------------------------ metrics / helpers
    def _wrap(self, x: NDArray) -> NDArray:
        if not self.angular_dims:
            return x
        x = x.copy()
        for i in self.angular_dims:
            x[i] = (x[i] + np.pi) % (2 * np.pi) - np.pi
        return x

    def _state_dists(self, arr: NDArray, x: NDArray) -> NDArray:
        d = arr - x
        for i in self.angular_dims:
            d[:, i] = (d[:, i] + np.pi) % (2 * np.pi) - np.pi
        return np.linalg.norm(d, axis=1)

    def _robot_positions(self, x: NDArray) -> NDArray:
        q = self.env.start_pos.from_flat(x)
        return np.concatenate(
            [np.asarray(self.env.robot_position(q, i)) for i in range(q.num_agents())]
        )

    def _edge_cost(self, x_from: NDArray, traj: List[NDArray]) -> float:
        if self.cost_metric == "length":
            n = self.env.start_pos.num_agents()
            d = self.env.position_dim_per_robot
            prev = self._robot_positions(x_from).reshape(n, d)
            total = 0.0
            for x in traj:
                cur = self._robot_positions(x).reshape(n, d)
                total += float(np.sum(np.linalg.norm(cur - prev, axis=1)))
                prev = cur
            return total
        return len(traj) * self.dt

    def _add_node(self, node: _Node):
        if node.mode not in self.nodes_by_mode:
            self.nodes_by_mode[node.mode] = []
            self.witnesses_by_mode[node.mode] = []
            self.modes.append(node.mode)
        self.nodes_by_mode[node.mode].append(node)

    def _remove_node(self, node: _Node):
        self.nodes_by_mode[node.mode].remove(node)

    # ------------------------------------------------------------------ mode selection / sampling
    def _mode_progress(self, mode: Mode) -> int:
        if self.env.is_terminal_mode(mode):
            return 1 << 30
        get_idx = getattr(self.env, "get_current_seq_index", None)
        try:
            return get_idx(mode) if get_idx else 0
        except Exception:
            return 0

    def _select_mode(self) -> Mode:
        fb = self.config.frontier_bias if not self.goal_nodes else 0.0
        if len(self.modes) > 1 and random.random() < fb:
            progresses = [self._mode_progress(m) for m in self.modes]
            mx = max(progresses)
            return random.choice([m for m, p in zip(self.modes, progresses) if p == mx])
        return random.choice(self.modes)

    def _sample(self, mode: Mode) -> NDArray:
        lo, hi = self.state_bounds[0], self.state_bounds[1]
        x = np.random.uniform(lo, hi)
        if random.random() < self.config.goal_bias:
            task = self.env.get_active_task(mode, None)
            goal = np.asarray(task.goal.sample(mode), dtype=float)
            offset = 0
            for r in task.robots:
                r_idx = self.env.robots.index(r)
                s, e = self.env.start_pos._array_slice[r_idx]
                x[s:e] = goal[offset : offset + (e - s)]
                offset += e - s
        return x

    def _best_near(self, mode: Mode, x_rand: NDArray) -> _Node:
        nodes = self.nodes_by_mode[mode]
        arr = np.array([n.x for n in nodes])
        dists = self._state_dists(arr, x_rand)
        within = np.where(dists <= self.config.delta_bn)[0]
        if len(within) > 0:
            costs = np.array([nodes[i].cost for i in within])
            return nodes[within[int(np.argmin(costs))]]
        return nodes[int(np.argmin(dists))]

    def _nearest_witness(self, mode: Mode, x: NDArray) -> Tuple[Optional[_Witness], float]:
        ws = self.witnesses_by_mode[mode]
        if not ws:
            return None, math.inf
        arr = np.array([w.x for w in ws])
        dists = self._state_dists(arr, x)
        i = int(np.argmin(dists))
        return ws[i], float(dists[i])

    # ------------------------------------------------------------------ propagation / validity
    def _propagate(self, x0: NDArray, mode: Mode) -> Optional[List[NDArray]]:
        lo, hi = self.control_bounds[0], self.control_bounds[1]
        u = np.random.uniform(lo, hi)
        # each robot independently either moves (random control) or holds (zero control), so
        # parked robots can stay put while others manoeuvre, WITHOUT forbidding simultaneous
        # coordinated motion (all robots move with prob (1-hold_prob)^n).
        n = self.env.start_pos.num_agents()
        if n > 1 and self.config.hold_prob > 0.0:
            m_per = u.shape[0] // n
            for r in range(n):
                if random.random() < self.config.hold_prob:
                    u[r * m_per : (r + 1) * m_per] = 0.0
        steps = random.randint(self.config.min_prop_steps, self.config.max_prop_steps)
        raw = self.env.propagate(x0, u, num_steps=steps)
        traj = []
        slo, shi = self.state_bounds[0], self.state_bounds[1]
        for x in raw:
            xw = self._wrap(x)
            if not (np.all(xw >= slo - 1e-9) and np.all(xw <= shi + 1e-9)):
                return None
            if not self.env.is_collision_free(self.env.start_pos.from_flat(xw), mode):
                return None
            traj.append(xw)
        return traj

    # ------------------------------------------------------------------ pruning
    def _prune(self, node: _Node):
        while node is not None and not node.children and not node.is_rep and not node.protected:
            self._remove_node(node)
            parent = node.parent
            if parent is not None:
                parent.children.remove(node)
            node = parent

    # ------------------------------------------------------------------ mode events
    def _make_witness_root(self, node: _Node):
        self.witnesses_by_mode.setdefault(node.mode, [])
        self.witnesses_by_mode[node.mode].append(_Witness(node.x, node))

    def _handle_mode_events(self, node: _Node):
        q = self.env.start_pos.from_flat(node.x)
        if self.env.done(q, node.mode):
            self.goal_nodes.append(node)
        elif self.env.is_transition(q, node.mode):
            for nm in self.env.get_next_modes(q, node.mode):
                trans = _Node(node.x.copy(), nm, node, node.cost, protected=True)
                node.children.append(trans)
                self._add_node(trans)
                self._make_witness_root(trans)

    # ------------------------------------------------------------------ path build
    def _build_path(self, goal: _Node) -> List[State]:
        chain, n = [], goal
        while n is not None:
            chain.append(n)
            n = n.parent
        chain.reverse()
        path = [State(self.env.start_pos.from_flat(chain[0].x), chain[0].mode)]
        for node in chain[1:]:
            # transition nodes share the parent's state (mode switch); double the state
            path.append(State(self.env.start_pos.from_flat(node.x), node.mode))
        return path

    def _best_goal(self) -> Optional[_Node]:
        return min(self.goal_nodes, key=lambda g: g.cost) if self.goal_nodes else None

    # ------------------------------------------------------------------ main loop
    def plan(self, ptc: PlannerTerminationCondition, optimize: bool = True) -> Tuple[Optional[List[State]], Dict[str, Any]]:
        self.start_time = time.time()
        start_mode = self.env.get_start_mode()
        x_start = np.asarray(self.env.start_pos.state(), dtype=float)
        start = _Node(x_start, start_mode, None, 0.0, protected=True)
        self._add_node(start)
        self._make_witness_root(start)

        best_reported = math.inf
        i = 0
        while True:
            i += 1
            if ptc.should_terminate(i, time.time() - self.start_time):
                break

            mode = self._select_mode()
            x_rand = self._sample(mode)
            selected = self._best_near(mode, x_rand)

            traj = self._propagate(selected.x, mode)
            if traj is None:
                continue
            x_new = traj[-1]
            new_cost = selected.cost + self._edge_cost(selected.x, traj)

            witness, wdist = self._nearest_witness(mode, x_new)
            if witness is None or wdist > self.config.delta_s:
                witness = _Witness(x_new, None)
                self.witnesses_by_mode[mode].append(witness)

            if witness.rep is not None and new_cost >= witness.rep.cost - 1e-12:
                continue  # dominated -> discard

            node = _Node(x_new, mode, selected, new_cost)
            selected.children.append(node)
            self._add_node(node)

            old = witness.rep
            witness.rep = node
            if old is not None:
                old.is_rep = False
                self._prune(old)

            self._handle_mode_events(node)

            best = self._best_goal()
            if best is not None and best.cost < best_reported - 1e-9:
                best_reported = best.cost
                path = self._build_path(best)
                self.costs.append(best.cost)
                self.times.append(time.time() - self.start_time)
                self.all_paths.append(path)
                print(f"[kino-sst] solution at iter {i}: cost={best.cost:.3f}")
                if not optimize:
                    break

        best = self._best_goal()
        info = {"costs": self.costs, "times": self.times, "paths": self.all_paths}
        if best is None:
            print(f"[kino-sst] no solution found ({i} iterations)")
            return None, info
        return self._build_path(best), info
