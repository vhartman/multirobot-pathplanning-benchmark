"""Affine coordinate groups and compatible-run shortcut proposals.

Equality-connected coordinates move together. Cost, inequality feasibility,
collision checking and path updates remain in the shortcutting loop.
"""
import heapq
from bisect import bisect_right

import numpy as np

from multi_robot_multi_goal_planning.problems.constraint_set import ConstraintProvider


class AffineIndexGroups:
    """Index immutable mode supports once; close only over the queried interval.

    Coordinates change during shortcutting, but modes and affine supports do
    not. Integer masks avoid reconstructing Python/NumPy support sets for
    every rejected endpoint draw. No union over unrelated modes is allowed.
    """

    def __init__(self, env, path):
        self.dimension = len(path[0].q.state())
        provider = ConstraintProvider(env)
        self.starts, self.supports = [], []
        by_mode = {}
        previous = None
        for index, state in enumerate(path):
            key = id(state.mode)
            if key == previous:
                continue
            previous = key
            if key not in by_mode:
                equalities = provider.for_mode(state.mode).eq_aff
                by_mode[key] = frozenset(
                    sum(1 << int(i) for i in np.flatnonzero(row))
                    for constraint in equalities for row in constraint.mat)
            supports = by_mode[key]
            if not self.supports or supports != self.supports[-1]:
                self.starts.append(index)
                self.supports.append(supports)

    def close(self, first, last, indices):
        lo = bisect_right(self.starts, first)-1
        hi = bisect_right(self.starts, last)
        supports = set().union(*self.supports[lo:hi])
        group = sum(1 << int(i) for i in set(indices))
        previous = None
        while group != previous:
            previous = group
            for support in supports:
                if group & support:
                    group |= support
        return [i for i in range(self.dimension) if group & (1 << i)]


class CoordinateProposals:
    """Uniform coordinate subsets, closed over equality-connected blocks.

    Compatible runs keep the tasks of every touched robot fixed. Subset size,
    subset, and eligible run are drawn uniformly; endpoints are drawn without
    replacement. Blocks are joined across all path modes, conservatively.

    full_group alternates full and random-group proposals. tree_group alternates
    longest-first full/block intervals and random groups; failed tree intervals
    split at their midpoint. Once the tree is empty, only random groups remain.
    Mechanism policies group blocks with the same owning-robot set, then draw
    subsets of these groups. full_mechanism alternates with full configurations.
    Coordinates may change, but path length and modes must remain fixed.
    """

    POLICIES = ('compatible', 'full_group', 'tree_group', 'mechanism_group', 'full_mechanism')

    def __init__(self, env, path, policy, robot_choice='index_groups'):
        if policy not in self.POLICIES:
            raise ValueError(f'Unknown proposal policy: {policy}')
        if robot_choice not in ('index_groups', 'full'):
            raise ValueError('Compatible policies require index_groups or full selection')
        self.policy, self.robot_choice = policy, robot_choice
        self.dimension = len(path[0].q.state())
        provider = ConstraintProvider(env)
        self.robot_indices = [set(env.robot_idx[r]) for r in env.robots]
        self.tasks = np.asarray([s.mode.task_ids for s in path])
        # Equality-connected coordinates form atomic blocks. Union across modes
        # is conservative for mode-local constraints, exact for shared joints.
        parent = list(range(self.dimension))
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        seen = set()
        for state in path:
            if id(state.mode) in seen:
                continue
            seen.add(id(state.mode))
            equalities = provider.for_mode(state.mode).eq_aff
            for c in equalities:
                for row in c.mat:
                    support = np.flatnonzero(row)
                    for idx in support[1:]:
                        parent[root(int(idx))] = root(int(support[0]))
        components = {}
        for idx in range(self.dimension):
            components.setdefault(root(idx), []).append(idx)
        self.blocks = list(components.values())
        self.block_of = {idx: block for block in self.blocks for idx in block}
        mechanisms = {}
        for block in self.blocks:
            mechanisms.setdefault(tuple(self.robots(block)), []).extend(block)
        self.mechanisms = list(mechanisms.values())
        self.full = list(range(self.dimension))
        self.turn = 0
        self.heap, self.serial = [], 0
        self.pending = None
        self.stats = dict(draws=0, tree_proposals=0, full_proposals=0,
                          group_proposals=0, accepted=0, collision=0,
                          cost=0, projection=0, constraint=0, splits=0)
        if policy == 'tree_group':
            # Full and equality-block roots; no subset enumeration.
            for indices in [self.full] + self.blocks:
                for first, last in self.runs(indices):
                    self.push(first, last, indices)

    def robots(self, indices):
        selected = set(indices)
        return [r for r, owned in enumerate(self.robot_indices) if selected & owned]

    def runs(self, indices):
        signatures = self.tasks[:, self.robots(indices)]
        changes = np.flatnonzero(np.any(signatures[1:] != signatures[:-1], axis=1)) + 1
        boundaries = [0, *changes.tolist(), len(self.tasks)]
        return [(a, b-1) for a, b in zip(boundaries[:-1], boundaries[1:]) if b-a >= 3]

    def push(self, first, last, indices):
        if last-first < 2:
            return
        heapq.heappush(self.heap, (-(last-first), self.serial, first, last, indices))
        self.serial += 1

    def __call__(self):
        self.stats['draws'] += 1
        self.pending = None
        turn = self.turn
        self.turn += 1
        if self.policy == 'tree_group' and self.heap and turn % 2 == 0:
            _, _, first, last, indices = heapq.heappop(self.heap)
            self.pending = (first, last, indices)
            self.stats['tree_proposals'] += 1
            return first, last, indices
        if self.robot_choice == 'full' or (self.policy in ('full_group', 'full_mechanism') and turn % 2 == 0):
            indices = self.full
            self.stats['full_proposals'] += 1
        else:
            if self.policy in ('mechanism_group', 'full_mechanism'):
                size = np.random.randint(1, len(self.mechanisms)+1)
                sampled = np.random.choice(len(self.mechanisms), size=size, replace=False)
                indices = sorted({j for idx in sampled for j in self.mechanisms[idx]})
            else:
                size = np.random.randint(1, self.dimension+1)
                sampled = np.random.choice(self.dimension, size=size, replace=False)
                indices = sorted({j for idx in sampled for j in self.block_of[idx]})
            self.stats['group_proposals'] += 1
        runs = self.runs(indices)
        if not runs:
            return None
        first, last = runs[np.random.randint(len(runs))]
        i, j = sorted(np.random.choice(np.arange(first, last+1), size=2, replace=False))
        if j-i < 2:
            return None
        return int(i), int(j), indices

    def feedback(self, outcome):
        self.stats[outcome] += 1
        if self.pending is not None and outcome in ('collision', 'projection', 'constraint'):
            first, last, indices = self.pending
            mid = (first+last)//2
            self.push(first, mid, indices)
            self.push(mid, last, indices)
            self.stats['splits'] += 1
        self.pending = None
