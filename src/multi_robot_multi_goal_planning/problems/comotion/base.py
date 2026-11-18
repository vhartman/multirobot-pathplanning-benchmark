"""Collision-query infrastructure for static CoMotion planning environments."""
import copy
import numpy as np

from ..planning_env import BaseProblem
from ..core.configuration import config_dist


class CoMotionEnvironment(BaseProblem):

    def __deepcopy__(self, memo):
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key not in ("_uniform_sampler", "C"):
                setattr(result, key, copy.deepcopy(value, memo))
        if result.backend == "vamp":
            result._collision_scene.kernel.set_optimized(result._collision_optimized)
        result._uniform_sampler = result._make_uniform_sampler()
        return result

    @staticmethod
    def _flat(q):
        return np.asarray(q.state() if hasattr(q, "state") else q, dtype=np.float64)

    def get_scenegraph_info_for_mode(self, mode, is_start_mode=False):
        return {}

    def set_to_mode(self, mode):
        # All modes have the same geometry; only their goal constraints change.
        pass

    def is_collision_free(self, q, mode=None, collision_tolerance=None):
        if collision_tolerance is None:
            return self._collision_scene.kernel.state_valid(self._flat(q))
        return self._collision_scene.kernel.state_valid(
            self._flat(q), self._collision_scene._initial, (),
            0. if collision_tolerance is None else collision_tolerance)

    def is_collision_free_for_robot(self, r, q, m=None, collision_tolerance=None, set_mode=True):
        group = [r] if isinstance(r, str) else list(r)
        if not group:
            return True
        return self._collision_scene.kernel.state_valid(
            self._flat(q), self._collision_scene._initial,
            [self.robots.index(name) for name in group],
            0. if collision_tolerance is None else collision_tolerance)

    def batch_is_collision_free(self, qs, mode=None, collision_tolerance=None):
        if not isinstance(qs, np.ndarray):
            qs = np.asarray([self._flat(q) for q in qs]).reshape(-1, len(self.start_pos.state()))
        return self._collision_scene.kernel.batch_valid(
            qs, self._collision_scene._initial, (), True,
            0. if collision_tolerance is None else collision_tolerance)

    def set_collision_optimizations(self, enabled=True, *, edge_kernel=None):
        if self.backend != "vamp":
            raise ValueError("Cached SIMD validation is available for the VAMP backend only")
        if edge_kernel is not None:
            raise ValueError("Explicit-grid edge checks use the common native kernel; legacy motion kernels are available through RobotScene")
        self._collision_scene.kernel.set_optimized(bool(enabled))
        self._collision_optimized = bool(enabled)

    def is_edge_collision_free(self, q1, q2, mode=None, resolution=None, tolerance=None,
                               include_endpoints=False, N_start=0, N_max=None, N=None):
        resolution = self.collision_resolution if resolution is None else resolution
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError('Invalid collision resolution')
        n = max(2, int(config_dist(q1, q2, 'max') / resolution) + 1) if N is None else N
        stop = n if N_max is None else min(n, N_max)
        if n < 2 or not 0 <= N_start <= stop <= n:
            raise ValueError('Invalid edge sample range')
        return self._collision_scene.edge_valid(
            self._flat(q1), self._flat(q2), mode, n, N_start, stop,
            include_endpoints, tolerance)

    def display_path(self, path, **kwargs):
        raise NotImplementedError('No viewer is installed for CoMotion robot scenes')

    def show(self, blocking=False):
        return self.show_config(self.start_pos, blocking=blocking)

    def show_config(self, q, blocking=False):
        raise NotImplementedError('No viewer is installed for CoMotion robot scenes')
