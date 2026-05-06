from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.core.configuration import Configuration
from multi_robot_multi_goal_planning.problems.planning_env import BaseProblem, Mode


PinnedRobots = Dict[str, NDArray]
SeedFn = Callable[[Mode], Configuration]


@dataclass
class CollisionFreeSamplerConfig:
    sampler: str = "auto"
    sampler_gibbs_sweeps: int = 1
    sampler_auto_warmup: int = 20
    max_per_robot_attempts: int = 100

    @classmethod
    def from_planner_config(cls, config: Any) -> "CollisionFreeSamplerConfig":
        return cls(
            sampler=getattr(config, "sampler", cls.sampler),
            sampler_gibbs_sweeps=getattr(
                config, "sampler_gibbs_sweeps", cls.sampler_gibbs_sweeps
            ),
            sampler_auto_warmup=getattr(
                config, "sampler_auto_warmup", cls.sampler_auto_warmup
            ),
            max_per_robot_attempts=getattr(
                config, "max_per_robot_attempts", cls.max_per_robot_attempts
            ),
        )


@dataclass
class SamplingResult:
    q: Configuration | None
    collision_checks: int = 0
    sampler_name: str = ""


class BaseCollisionFreeSampler:
    def __init__(self, env: BaseProblem, config: CollisionFreeSamplerConfig):
        self.env = env
        self.config = config

    def sample(
        self, mode: Mode, pinned: PinnedRobots | None = None
    ) -> Configuration | None:
        return self._sample_result(mode, pinned or {}).q

    def sample_batch(
        self,
        mode: Mode,
        batch_size: int,
        pinned: PinnedRobots | None = None,
    ) -> List[Configuration]:
        samples = []
        attempts = 0
        max_attempts = max(1, 100 * batch_size)
        while len(samples) < batch_size and attempts < max_attempts:
            attempts += 1
            q = self.sample(mode, pinned)
            if q is not None:
                samples.append(self._copy_config(q))
        return samples

    def _sample_result(self, mode: Mode, pinned: PinnedRobots) -> SamplingResult:
        raise NotImplementedError

    def selected_sampler_name(self, mode: Mode | None = None) -> str:
        return self.config.sampler

    def _apply_pinned(self, q: Configuration, pinned: PinnedRobots) -> None:
        for robot, values in pinned.items():
            i = self.env.robots.index(robot)
            q[i] = values

    def _copy_config(self, q: Configuration) -> Configuration:
        return q.from_flat(q.state().copy())


class JointRejectionSampler(BaseCollisionFreeSampler):
    def _sample_result(self, mode: Mode, pinned: PinnedRobots) -> SamplingResult:
        q = self.env.sample_config_uniform_in_limits()
        self._apply_pinned(q, pinned)
        if self.env.is_collision_free(q, mode):
            return SamplingResult(q, collision_checks=1, sampler_name="joint")
        return SamplingResult(None, collision_checks=1, sampler_name="joint")


class PerRobotRejectionSampler(BaseCollisionFreeSampler):
    def _sample_result(self, mode: Mode, pinned: PinnedRobots) -> SamplingResult:
        q = self.env.sample_config_uniform_in_limits()
        self._apply_pinned(q, pinned)
        collision_checks = 0

        for i, robot in enumerate(self.env.robots):
            if robot in pinned:
                continue

            lims = self.env.limits[:, self.env.robot_idx[robot]]
            for _ in range(self.config.max_per_robot_attempts):
                collision_checks += 1
                if self.env.is_collision_free_for_robot([robot], q.state(), mode):
                    break
                q[i] = np.random.uniform(lims[0], lims[1])
            else:
                return SamplingResult(
                    None, collision_checks=collision_checks, sampler_name="per_robot"
                )

        collision_checks += 1
        if self.env.is_collision_free(q, mode):
            return SamplingResult(
                q, collision_checks=collision_checks, sampler_name="per_robot"
            )
        return SamplingResult(
            None, collision_checks=collision_checks, sampler_name="per_robot"
        )


class GibbsSampler(BaseCollisionFreeSampler):
    """
    Approximate Gibbs sampler over robots.

    Intermediate states in the first sweep can be invalid; each robot gets a
    bounded number of proposals and the final full configuration is validated.
    This mirrors the previous RRT* implementation while making the sampler
    reusable by graph-based planners later.
    """

    def __init__(
        self,
        env: BaseProblem,
        config: CollisionFreeSamplerConfig,
        seed_fn: SeedFn | None = None,
    ):
        super().__init__(env, config)
        self.seed_fn = seed_fn

    def _sample_result(self, mode: Mode, pinned: PinnedRobots) -> SamplingResult:
        seed = (
            self.seed_fn(mode)
            if self.seed_fn is not None
            else self.env.get_start_pos()
        )
        q = self.env.sample_config_uniform_in_limits()
        for i in range(len(self.env.robots)):
            q[i] = seed[i].copy()
        self._apply_pinned(q, pinned)

        collision_checks = 0
        for _ in range(self.config.sampler_gibbs_sweeps):
            for i, robot in enumerate(self.env.robots):
                if robot in pinned:
                    continue
                lims = self.env.limits[:, self.env.robot_idx[robot]]
                for _ in range(self.config.max_per_robot_attempts):
                    q[i] = np.random.uniform(lims[0], lims[1])
                    collision_checks += 1
                    if self.env.is_collision_free(q, mode):
                        break

        collision_checks += 1
        if self.env.is_collision_free(q, mode):
            return SamplingResult(
                q, collision_checks=collision_checks, sampler_name="gibbs"
            )
        return SamplingResult(
            None, collision_checks=collision_checks, sampler_name="gibbs"
        )


class AutoCollisionFreeSampler(BaseCollisionFreeSampler):
    def __init__(
        self,
        env: BaseProblem,
        config: CollisionFreeSamplerConfig,
        joint_sampler: BaseCollisionFreeSampler,
        gibbs_sampler: BaseCollisionFreeSampler,
    ):
        super().__init__(env, config)
        self.joint_sampler = joint_sampler
        self.gibbs_sampler = gibbs_sampler
        self._joint_sampling_stats: Dict[Mode, List[int]] = {}
        self._gibbs_sampling_stats: Dict[Mode, List[int]] = {}

    def selected_sampler_name(self, mode: Mode | None = None) -> str:
        if mode is None:
            return "joint"

        joint_stats = self._joint_sampling_stats.get(mode)
        if joint_stats is None or joint_stats[0] < self.config.sampler_auto_warmup:
            return "joint"

        joint_attempts, joint_successes = joint_stats
        expected_joint = joint_attempts / max(joint_successes, 1e-9)
        gibbs_stats = self._gibbs_sampling_stats.get(mode)
        if gibbs_stats is None:
            return "gibbs" if joint_successes == 0 else "joint"
        if gibbs_stats[1] == 0:
            return "joint"

        expected_gibbs = gibbs_stats[0] / gibbs_stats[1]
        return "gibbs" if expected_gibbs < expected_joint else "joint"

    def _sample_result(self, mode: Mode, pinned: PinnedRobots) -> SamplingResult:
        sampler_name = self.selected_sampler_name(mode)
        if sampler_name == "gibbs":
            result = self.gibbs_sampler._sample_result(mode, pinned)
            stats = self._gibbs_sampling_stats.setdefault(mode, [0, 0])
        else:
            result = self.joint_sampler._sample_result(mode, pinned)
            stats = self._joint_sampling_stats.setdefault(mode, [0, 0])
        stats[0] += result.collision_checks
        stats[1] += result.q is not None
        return result


def make_collision_free_sampler(
    env: BaseProblem,
    planner_config: Any,
    seed_fn: SeedFn | None = None,
) -> BaseCollisionFreeSampler:
    config = CollisionFreeSamplerConfig.from_planner_config(planner_config)
    joint = JointRejectionSampler(env, config)
    per_robot = PerRobotRejectionSampler(env, config)
    gibbs = GibbsSampler(env, config, seed_fn)

    if config.sampler == "auto":
        return AutoCollisionFreeSampler(env, config, joint, gibbs)
    if config.sampler == "gibbs":
        return gibbs
    if config.sampler == "per_robot":
        return per_robot
    return joint
