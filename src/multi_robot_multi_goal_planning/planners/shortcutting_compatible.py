"""Tree and random-subset proposals for multi-robot path shortcutting.

Only proposals live here; the common shortcutting loop owns interpolation,
cost evaluation, collision checking, and path updates.
"""
from dataclasses import dataclass
import heapq

import numpy as np


@dataclass(frozen=True)
class ShortcutProposal:
    """Inclusive path endpoints and indices of the robots to straighten."""

    first: int
    last: int
    robots: tuple[int, ...]
    from_tree: bool = False


class CompatibleSubsetProposer:
    """Propose shortcuts within intervals of constant task IDs.

    The tree starts with each robot's maximal constant-task runs, considering
    longer intervals first and breaking equal-length ties randomly. A colliding
    interval splits at the reported failure index; children spanning fewer than
    two edges are discarded. Accepted or non-improving intervals are removed.

    Random proposals choose a subset size uniformly from 1 through the robot
    count, then a subset uniformly among subsets of that size. A compatible run
    is a maximal interval where none of those robots changes task. Choose one
    eligible run uniformly, then draw two indices independently and uniformly
    within it. Return None if no run exists or the indices span fewer than two
    edges. Runs are computed per draw; robot subsets are not enumerated/cached.

    ``interleaved`` alternates tree and random draws while tree work remains.
    ``sequenced`` exhausts the tree before drawing random subsets. ``subset``
    uses only random draws. Both tree schedules continue with random subsets
    after the tree is exhausted. Path length and task IDs must remain fixed
    while this proposer is in use; coordinates may change after acceptance.
    """

    def __init__(self, path, runs, rng, schedule="interleaved"):
        if schedule not in ("interleaved", "sequenced", "subset"):
            raise ValueError(f"Unknown compatible-subset schedule: {schedule}")
        self.tasks = np.asarray([state.mode.task_ids for state in path], dtype=int)
        self.robot_count = self.tasks.shape[1]
        self.rng = rng
        self.schedule = schedule
        self.tree_turn = True
        self.heap = []
        self.serial = 0
        if schedule != "subset":
            for robot, intervals in enumerate(runs):
                for first, last in intervals:
                    self._push(robot, first, last)

    def _push(self, robot, first, last):
        if last - first >= 2:
            heapq.heappush(self.heap, (
                -(last - first), float(self.rng.random()), self.serial,
                robot, first, last,
            ))
            self.serial += 1

    def __call__(self):
        if self.heap and (self.schedule == "sequenced" or self.tree_turn):
            self.tree_turn = False
            _, _, _, robot, first, last = heapq.heappop(self.heap)
            return ShortcutProposal(first, last, (robot,), from_tree=True)

        self.tree_turn = True
        size = int(self.rng.integers(1, self.robot_count + 1))
        robots = tuple(sorted(int(r) for r in self.rng.choice(
            self.robot_count, size=size, replace=False,
        )))
        signatures = self.tasks[:, robots]
        changes = np.flatnonzero(np.any(signatures[1:] != signatures[:-1], axis=1)) + 1
        boundaries = [0, *changes.tolist(), len(self.tasks)]
        runs = [(a, b - 1) for a, b in zip(boundaries, boundaries[1:]) if b - a >= 3]
        if not runs:
            return None
        first, last = runs[int(self.rng.integers(len(runs)))]
        i, j = sorted((int(self.rng.integers(first, last + 1)),
                       int(self.rng.integers(first, last + 1))))
        return ShortcutProposal(i, j, robots) if j - i >= 2 else None

    def collision(self, proposal, fail_index):
        if (proposal.from_tree and fail_index is not None
                and proposal.first < fail_index < proposal.last):
            robot, = proposal.robots
            self._push(robot, proposal.first, fail_index)
            self._push(robot, fail_index, proposal.last)
