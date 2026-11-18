"""CoMotion collision checking with our costs, tasks and planners.

Install the MRMG extra; it builds CoMotion's Python bindings from the
vhartman/comotion branch vh/experimental/mrmg (pinned commit; not on PyPI)::

    python -m pip install -e '.[comotion]'  # From the MRMG repository.

The build needs a C++17 compiler, CMake, Eigen3, FCL (with libccd and octomap)
and the Boost headers. Alternatively build a checkout yourself::

    git clone --recurse-submodules -b vh/experimental/mrmg https://github.com/vhartman/comotion.git
    python -m pip install ./comotion -Cbuild.tool-args=-j8

Load a built-in scene and pass it to a planner like any other environment::

    from multi_robot_multi_goal_planning.problems import get_env_by_name

    env = get_env_by_name("comotion.panda_cage_4_sequence", visits=3)

The sequence variant fixes the task order. Use ``_independent`` instead of
``_sequence`` for independent per-robot goal chains. These built-in scenes
have static obstacles and no carried objects; only collision checking uses
CoMotion, not its planners. Built-in scenes use the generated Panda kernels;
``backend="generic"`` runs the identical geometry through the generic primitive
checker. Both use the common CoMotion scene definition and query API. Unsupported
geometry or an incompatible generated model raises an error; no approximation
or backend fallback occurs.

An existing compatible RAI environment can also be converted once, including
rigid object attachments. Given an initialized ``rai_env``::

    from multi_robot_multi_goal_planning.problems.comotion_env import to_comotion

    env = to_comotion(rai_env)
    env.save("scene.json")
    restored = get_env_by_name("comotion.converted", path="scene.json")

Converted queries use native FK and collision checking without updating RAI.
The conversion supports spheres, capsules, cylinders and sharp boxes, fixed
joint-space goals/goal sets, and sequence/dependency-graph tasks. Meshes,
rounded boxes, path constraints, kinodynamics and custom costs are rejected;
geometry approximations must be explicit. Collision decisions near a nonzero
penetration tolerance can differ between RAI and CoMotion.
"""
import copy
import json
from pathlib import Path

import numpy as np

from .core.configuration import NpConfiguration, config_dist
from .core.cost import JointSpaceLength
from comotion import resource_directory, load_builtin, import_robot_scene
from .comotion_scene import CoMotionScene
from .core.dependency_graph import DependencyGraph
from .core.goals import SingleGoal
from .core.registry import register
from .planning_env import (
    BaseProblem, BaseModeLogic, SequenceMixin, DependencyGraphMixin, Task,
    ProblemSpec, AgentType, ConstraintType, ManipulationType, DependencyType,
    DynamicsType, GoalType, SafePoseType,
)


class CoMotionGeometry(BaseProblem):
    def _init_scene(self, family, num_robots, task_index, visits, task_style,
                    backend, scene_file):
        BaseProblem.__init__(self)
        self.scene = (json.loads(Path(scene_file).read_text()) if scene_file else
                      load_builtin(family, num_robots))
        self.family = family
        self.backend = backend
        records = self.scene["tasks"]
        if not 0 <= task_index < len(records) or not 1 <= visits <= len(records):
            raise ValueError("Invalid task_index or visits")
        self.task_index, self.visits, self.task_style = task_index, visits, task_style
        resources = resource_directory() / "panda"
        definition = import_robot_scene(
            resources / "panda_spherized.urdf", resources / "panda.srdf",
            self.scene["robot_bases"], self.scene.get("obstacles", []),
            start_q=np.asarray(records[task_index]["starts"]).ravel())
        self.robots = [f"a{i+1}" for i in range(len(self.scene["robot_bases"]))]
        definition["robot_names"] = self.robots
        implementation = "generated" if backend == "vamp" else backend
        self._collision_scene = CoMotionScene(definition, implementation=implementation)
        self._collision_optimized = backend == "vamp"
        self.limits = np.asarray(definition["limits"]).T
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
            if not self._collision_scene.valid(goals.ravel()):
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
