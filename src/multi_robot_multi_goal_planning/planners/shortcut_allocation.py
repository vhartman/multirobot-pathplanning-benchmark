"""How shortcutting shares the planning time with the tree search (BaseRRTConfig.shortcut_allocation="marginal").

With "fixed", BaseRRTstar.generate_path runs one shortcut pass of 100 collision-checked proposals whenever the tree
finds a better path, and never otherwise. Here shortcutting runs in chunks, with several operators, and after each
chunk a rule decides whether the next chunk or the tree gets the time. There is no fixed time share.

Operators, coarse to fine: the policies in ``shortcut_policies`` (run_choice values of robot_mode_shortcut; by
default whole robots, then subsets of coordinates), then damped shortcuts (``shortcut_damped``; only on problems
without constraints).

The rule compares cost reduction per second.
  - The tree's rate is an exponentially discounted average of its own improvements over ``shortcut_tree_horizon``
    seconds of tree time. Before the tree has shown anything it gets the benefit of the doubt: its rate starts at
    ``shortcut_tree_prior`` times the rate of the very first chunk.
  - The expected rate of an operator is the rate of its last chunk. After the tree has changed the path, it is the
    rate the operator achieved the last time it ran first on a changed path (at most halved per measurement). An
    operator that has never run gets a probe of ``shortcut_probe_chunk`` proposals, so that its rate is measured.
  - An operator runs while its expected rate is above the tree's rate, and above ``shortcut_rate_floor`` x cost.
    No operator is run just because the tree improved the path: with informed samples that pay, the tree does
    that 100-300 times per run. An operator that was skipped on ``shortcut_probe_every`` tree improvements in a
    row gets one chunk regardless.
  - Operators that were skipped come back before a tree iteration once the tree's rate has decayed below their
    expected rate (resume()).
Neither side can be starved: chunks on an unchanged path stop paying, so their rate falls below the floor and the
tree runs; every operator gets a chunk at the latest after ``shortcut_probe_every`` skipped tree improvements;
and stopped operators resume when the tree goes quiet. So right after the first solution only the coarse operator runs, and the fine ones start when the tree
slows down.

A shortcut result is reported after every improving chunk and reinserted into the tree once per session.

The rates are measured in wall-clock time, so runs with "marginal" are not reproducible from the seed alone.
"fixed" is.
"""
import math
import time

import numpy as np

from multi_robot_multi_goal_planning.problems.util import path_cost

from .shortcutting import damped_shortcut_chunk, remove_interpolated_nodes, robot_mode_shortcut

DAMPED = "damped"


class ShortcutAllocator:
    def __init__(self, planner):
        config = planner.config
        self.planner = planner
        self.ops = ([name.strip() for name in config.shortcut_policies.split(",") if name.strip()]
                    or [config.shortcutting_run_choice])
        if config.shortcut_damped and not planner.has_path_constraints:
            self.ops.append(DAMPED)  # damped candidates are only collision checked
        n = len(self.ops)
        self._expected = [None] * n  # expected rate of each operator; None: never run
        self._fresh = [None] * n  # its rate when it last ran first on a changed path
        self._on_new_path = [False] * n  # the tree changed the path since the operator last ran
        self._skipped = [0] * n  # tree improvements in a row without a chunk of the operator
        self.best_cost = np.inf  # best cost reported so far, by the tree or by shortcutting
        self._tree_rate, self._tree_rate_time = 0.0, 0.0
        self._first_chunk_pending = False
        self._start = time.perf_counter()
        self.stats = dict(sessions=0, resumes=0, tree_gain=0.0, reinsert_s=0.0,
                          chunks={op: 0 for op in self.ops}, seconds={op: 0.0 for op in self.ops},
                          gain={op: 0.0 for op in self.ops})

    # --- the two entry points ------------------------------------------------------------------------------------
    def tree_improved(self):
        """The tree found a better path (BaseRRTstar.generate_path): credit the tree, then shortcut the path."""
        cost, t = self.planner.operation.cost, self._tree_time()
        self._first_chunk_pending = not np.isfinite(self.best_cost)
        if not self._first_chunk_pending and cost < self.best_cost:
            gain = self.best_cost - cost
            self._add_tree_gain(t, gain)
            self.stats["tree_gain"] += float(gain)
        self.best_cost = min(self.best_cost, cost)
        self._session(changed_path=True)

    def resume(self):
        """Before a tree iteration: run the operators whose expected rate now beats the tree's."""
        if (np.isfinite(self.best_cost)
                and max(self._expected_rates()) > self._threshold(self.planner.operation.cost)):
            self.stats["resumes"] += 1
            self._session(changed_path=False)

    # --- rates ---------------------------------------------------------------------------------------------------
    def _tree_time(self):
        """Seconds the tree search has had: wall time without the shortcut chunks and their reinsertion."""
        return (time.perf_counter() - self._start - sum(self.stats["seconds"].values())
                - self.stats["reinsert_s"])

    def _tree_rate_now(self, t=None):
        t = self._tree_time() if t is None else t
        horizon = self.planner.config.shortcut_tree_horizon
        return self._tree_rate * math.exp(-max(0.0, t - self._tree_rate_time) / horizon)

    def _add_tree_gain(self, t, gain):
        """An improvement of `gain` at tree time t: an impulse gain / horizon that decays with the horizon."""
        self._tree_rate = self._tree_rate_now(t) + gain / self.planner.config.shortcut_tree_horizon
        self._tree_rate_time = t

    def _threshold(self, cost):
        return max(self._tree_rate_now(), self.planner.config.shortcut_rate_floor * cost)

    def _expected_rates(self):
        """Expected rate per operator; one that has never run inherits the one before it (the first: infinity)."""
        rates, previous = [], np.inf
        for rate in self._expected:
            previous = previous if rate is None else rate
            rates.append(previous)
        return rates

    # --- sessions ------------------------------------------------------------------------------------------------
    def _run(self, op, path, cost, proposals=None):
        """One chunk of `op`. Returns (path, cost, seconds); the path is None if the chunk did not lower the cost."""
        planner, config = self.planner, self.planner.config
        env = planner.env
        t0 = time.perf_counter()
        if op == DAMPED:
            new_path, gain, _ = damped_shortcut_chunk(
                env, path, proposals or config.shortcut_damped_chunk, config.shortcut_damped_steps,
                env.collision_resolution, env.collision_tolerance)
            new_cost = path_cost(new_path, env.cost) if gain > 0 else cost
        else:
            new_path, result = robot_mode_shortcut(
                env, path, proposals or config.shortcut_chunk, run_choice=op,
                nonlinear_projection=config.nl_projection, should_stop=planner._planning_expired,
                interpolation_resolution=((config.nl_steer_step_size or 0.05)
                                          if planner.has_nonlinear_constraints else 0.5),
                resolution=env.collision_resolution, tolerance=env.collision_tolerance)
            new_cost = result[0][-1]
        seconds = time.perf_counter() - t0
        self.stats["seconds"][op] += seconds
        self.stats["chunks"][op] += 1
        return (new_path if new_cost < cost else None), min(new_cost, cost), seconds

    def _final_path(self, path):
        return remove_interpolated_nodes(path) if self.planner.config.remove_redundant_nodes else path

    def _session(self, changed_path):
        planner, config = self.planner, self.planner.config
        path, cost, improved = planner.operation.path_shortcutting, planner.operation.cost, None
        self.stats["sessions"] += 1
        if changed_path:  # every operator has a fresh chance on the tree's new path
            for k in range(len(self.ops)):
                self._on_new_path[k] = True
                if self._fresh[k] is not None:
                    self._expected[k] = self._fresh[k]
        for k, op in enumerate(self.ops):
            forced = changed_path and self._skipped[k] >= config.shortcut_probe_every
            probe = config.shortcut_probe_chunk if (k > 0 and self._expected[k] is None) else 0
            ran = False
            while not planner._planning_expired():
                if not (forced or probe) and self._expected_rates()[k] <= self._threshold(cost):
                    break
                forced = False
                new_path, new_cost, seconds = self._run(op, path, cost, proposals=probe or None)
                probe = 0
                rate = (cost - new_cost) / max(seconds, 1e-9)
                ran = True
                if new_path is not None:
                    path, cost, improved = new_path, new_cost, new_path
                    if new_cost < self.best_cost:  # usable from now on: report it
                        self.stats["gain"][op] += float(self.best_cost - new_cost)
                        self.best_cost = new_cost
                        planner.update_results_tracking(new_cost, self._final_path(new_path))
                self._expected[k] = rate
                if self._on_new_path[k]:  # its first chunk on a changed path
                    self._on_new_path[k] = False
                    self._fresh[k] = max(rate, 0.5 * (self._fresh[k] or 0.0))
                if self._first_chunk_pending:  # the tree's optimistic prior, from the first chunk ever
                    self._first_chunk_pending = False
                    self._add_tree_gain(self._tree_time(),
                                        config.shortcut_tree_prior * rate * config.shortcut_tree_horizon)
            if changed_path:
                self._skipped[k] = 0 if ran else self._skipped[k] + 1
        if improved is not None:
            t0 = time.perf_counter()
            planner.tree_extension(self._final_path(improved))  # once per session
            self.stats["reinsert_s"] += time.perf_counter() - t0
