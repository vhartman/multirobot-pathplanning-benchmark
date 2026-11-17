"""Static CoMotion scenes with our costs, goals, mode logic and planners.

The optional native package provides cached FK and SIMD collision queries.
It does not import their planners, and does not support carried objects.
"""
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .core.configuration import NpConfiguration, config_dist
from .core.cost import JointSpaceLength
from comotion import resource_directory, load_builtin
from .core.dependency_graph import DependencyGraph
from .core.goals import SingleGoal
from .core.registry import register
from .planning_env import (
    BaseProblem, BaseModeLogic, SequenceMixin, DependencyGraphMixin, Task,
    ProblemSpec, AgentType, ConstraintType, ManipulationType, DependencyType,
    DynamicsType, GoalType, SafePoseType, generate_binary_search_indices,
)


class CoMotionGeometry(BaseProblem):
    def _init_scene(self, family, num_robots, task_index, visits, task_style,
                    backend, scene_file):
        from comotion import RobotScene
        BaseProblem.__init__(self)
        self.scene = (json.loads(Path(scene_file).read_text()) if scene_file else
                      load_builtin(family, num_robots))
        self.family = family
        self.backend = backend
        records = self.scene["tasks"]
        if not 0 <= task_index < len(records) or not 1 <= visits <= len(records):
            raise ValueError("Invalid task_index or visits")
        self.task_index, self.visits, self.task_style = task_index, visits, task_style
        bases = np.repeat(np.eye(4)[None], len(self.scene["robot_bases"]), axis=0)
        for matrix, base in zip(bases, self.scene["robot_bases"]):
            matrix[:3, 3] = base["position"]
            matrix[:3, :3] = Rotation.from_quat(base.get("quaternion_xyzw", [0, 0, 0, 1])).as_matrix()
        spheres, cylinders = [], []
        for obstacle in self.scene.get("obstacles", []):
            if obstacle["type"] == "sphere":
                spheres.append([*obstacle["center"], obstacle["radius"]])
            elif obstacle["type"] == "cylinder":
                cylinders.append([*obstacle["center"], *obstacle["axis"],
                                  obstacle["radius"], obstacle["half_height"]])
            else:
                raise ValueError(f"Unsupported obstacle type: {obstacle['type']}")
        resources = resource_directory() / "panda"
        self._native_args = (str(resources / "panda_spherized.urdf"),
                             str(resources / "panda.srdf"), bases,
                             np.asarray(spheres, dtype=float).reshape(-1, 4),
                             np.asarray(cylinders, dtype=float).reshape(-1, 8), backend)
        self._scene = RobotScene(*self._native_args)
        self._collision_optimized = backend == "vamp"
        if self._collision_optimized:
            self._scene.set_optimized(True)
        self._edge_kernel = None  # native default, or an explicit replay/ablation override
        self.limits = self._scene.limits()
        self.robots = [f"a{i+1}" for i in range(len(bases))]
        self.robot_dims = {r: 7 for r in self.robots}
        self.robot_idx = {r: list(range(7*i, 7*(i+1))) for i, r in enumerate(self.robots)}
        self.start_pos = NpConfiguration.from_list([np.array(q, dtype=float) for q in records[task_index]["starts"]])
        self.collision_resolution = 0.01
        self.collision_tolerance = 0.0
        self.cost = JointSpaceLength()
        self.manipulating_env = False
        self.tasks = []
        self.graph = DependencyGraph()
        previous = {}
        for visit in range(visits):
            goals = np.asarray(records[(task_index + visit) % len(records)]["goals"], dtype=float)
            if goals.shape != (len(self.robots), 7):
                raise ValueError("Goal dimensions do not match the robot scene")
            if not self._scene.valid(goals.ravel()):
                raise ValueError(f"Upstream goal {visit} is invalid under backend {backend}")
            if task_style == "simultaneous":
                self.tasks.append(Task(f"reach_{visit}", self.robots, SingleGoal(goals.ravel().copy())))
            else:
                for i, r in enumerate(self.robots):
                    name = f"{r}_reach_{visit}"
                    self.tasks.append(Task(name, [r], SingleGoal(goals[i].copy())))
                    self.graph.add_node(name)
                    if r in previous:
                        self.graph.add_dependency(name, previous[r])
                    previous[r] = name
        if task_style != "simultaneous":
            self.tasks.append(Task("terminal", self.robots, SingleGoal(goals.ravel().copy())))
            for name in previous.values():
                self.graph.add_dependency("terminal", name)
        self.sequence = list(range(len(self.tasks)))
        self.spec = ProblemSpec(
            AgentType.MULTI_AGENT, ConstraintType.UNCONSTRAINED, ManipulationType.STATIC,
            DependencyType.UNORDERED if task_style == "independent" else DependencyType.FULLY_ORDERED,
            DynamicsType.GEOMETRIC, GoalType.SINGLE_GOAL if len(self.tasks) == 1 else GoalType.MULTI_GOAL,
            SafePoseType.HAS_NO_SAFE_HOME_POSE)
        BaseModeLogic.__init__(self)
        if not self.is_collision_free(self.start_pos, self.start_mode):
            raise ValueError(f"Upstream start is invalid under backend {backend}")

    def __deepcopy__(self, memo):
        from comotion import RobotScene
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key not in ("_scene", "_uniform_sampler", "C"):
                setattr(result, key, copy.deepcopy(value, memo))
        result._scene = RobotScene(*result._native_args)
        if result._collision_optimized:
            result._scene.set_optimized(True)
        if result._edge_kernel is not None:
            result._scene.set_edge_kernel(result._edge_kernel)
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
        if collision_tolerance not in (None, 0):
            raise ValueError('RobotScene supports zero collision tolerance only')
        return self._scene.valid(self._flat(q))

    def is_collision_free_for_robot(self, r, q, m=None, collision_tolerance=None, set_mode=True):
        if collision_tolerance not in (None, 0):
            raise ValueError('RobotScene supports zero collision tolerance only')
        group = [r] if isinstance(r, str) else list(r)
        # Existing planner contract: isolated feasibility of these robots, not
        # feasibility against the arbitrary poses of nonparticipating robots.
        # ModeValidation uses this as a necessary condition for a partial goal.
        # All accepted states/edges still undergo full robot-robot checks.
        return self._scene.group_valid(self._flat(q), [self.robots.index(x) for x in group], False)

    def batch_is_collision_free(self, qs, mode=None, collision_tolerance=None):
        if collision_tolerance not in (None, 0):
            raise ValueError('RobotScene supports zero collision tolerance only')
        return self._scene.batch_valid(np.asarray([self._flat(q) for q in qs]).reshape(-1, len(self.start_pos.state())))

    def set_collision_optimizations(self, enabled=True, *, edge_kernel=None):
        """Select cached SIMD queries or the reference collision implementation."""
        if self.backend != "vamp":
            raise ValueError("Cached SIMD validation is available for the VAMP backend only")
        self._scene.set_optimized(bool(enabled))
        self._collision_optimized = bool(enabled)
        if edge_kernel is not None:
            self._scene.set_edge_kernel(edge_kernel)
            self._edge_kernel = edge_kernel

    def is_edge_collision_free(self, q1, q2, mode=None, resolution=None, tolerance=None,
                               include_endpoints=False, N_start=0, N_max=None, N=None):
        resolution = self.collision_resolution if resolution is None else resolution
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError('Invalid collision resolution')
        n = max(2, int(config_dist(q1, q2, 'max') / resolution) + 1) if N is None else N
        stop = n if N_max is None else min(n, N_max)
        if n < 2 or not 0 <= N_start <= stop <= n:
            raise ValueError('Invalid edge sample range')
        indices = [i for i in generate_binary_search_indices(n)[N_start:stop]
                   if include_endpoints or i not in (0, n - 1)]
        a, b = self._flat(q1), self._flat(q2)
        for offset in range(0, len(indices), 16):
            t = np.asarray(indices[offset:offset + 16]) / (n - 1)
            if not self.batch_is_collision_free(a + t[:, None] * (b - a), mode, tolerance).all():
                return False
        return True

    def display_path(self, path, **kwargs):
        raise NotImplementedError('No viewer is installed for CoMotion robot scenes')

    def show(self, blocking=False):
        return self.show_config(self.start_pos, blocking=blocking)

    def show_config(self, q, blocking=False):
        raise NotImplementedError('No viewer is installed for CoMotion robot scenes')


@register([(f"{prefix}.panda_{family}_{n}{suffix}", dict(family=family, num_robots=n, task_style=style, backend=backend))
           for prefix, backend in [("comotion", "vamp")]
           for family, counts in [("cage", (2, 4, 8, 16)), ("flat", (4, 8, 16))]
           for n in counts for suffix, style in [("", "simultaneous"), ("_sequence", "ordered")]])
class CoMotionEnv(SequenceMixin, CoMotionGeometry):
    def __init__(self, family="cage", num_robots=4, task_index=0, visits=1,
                 task_style="simultaneous", backend="vamp", scene_file=None):
        if task_style not in ("simultaneous", "ordered"):
            raise ValueError("Use CoMotionIndependentEnv for independent goal chains")
        self._init_scene(family, num_robots, task_index, visits, task_style, backend, scene_file)


@register([(f"{prefix}.panda_{family}_{n}_independent", dict(family=family, num_robots=n, backend=backend))
           for prefix, backend in [("comotion", "vamp")]
           for family, counts in [("cage", (2, 4, 8, 16)), ("flat", (4, 8, 16))] for n in counts])
class CoMotionIndependentEnv(DependencyGraphMixin, CoMotionGeometry):
    """Each robot has a goal chain; inter-robot task order is free to be planned."""
    def __init__(self, family="cage", num_robots=4, task_index=0, visits=1,
                 backend="vamp", scene_file=None):
        self._init_scene(family, num_robots, task_index, visits, "independent", backend, scene_file)

# Register frozen converted problems under the same optional backend.
from .comotion_primitive_env import load_comotion_problem, to_comotion
