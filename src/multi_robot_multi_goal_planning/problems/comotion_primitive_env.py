"""Independent fixed-goal planning on exported articulated primitive scenes."""
import copy
import json
from dataclasses import asdict, fields
from pathlib import Path
import numpy as np
from .core.configuration import config_dist, NpConfiguration
from .core.cost import JointSpaceLength
from .core.goals import SingleGoal, GoalSet
from .comotion_scene import CoMotionScene
from .core.registry import register
from .planning_env import BaseProblem, BaseModeLogic, SequenceMixin, DependencyGraphMixin, DynamicsType, generate_binary_search_indices


class ConvertedPrimitiveEnvironment(BaseProblem):
    def __init__(self, source):
        if getattr(source, 'constraints', []) or any(t.constraints for t in source.tasks):
            raise NotImplementedError('Path constraints need a native constraint implementation')
        if source.spec.dynamics != DynamicsType.GEOMETRIC:
            raise NotImplementedError('Only geometric problems are converted')
        if type(source.cost) is not JointSpaceLength:
            raise NotImplementedError('Only joint-space costs are currently exported')
        if any(type(t.goal) not in (SingleGoal, GoalSet) for t in source.tasks):
            raise NotImplementedError('Only fixed goals/goal sets are exported')
        from .rai_comotion import export_rai_scene
        self.scene = CoMotionScene(export_rai_scene(source))
        for name in ['robots','robot_idx','robot_dims','robot_joints','limits','start_pos','tasks','spec','cost',
                     'collision_resolution','collision_tolerance','manipulating_env','sequence','graph','safe_pose']:
            if hasattr(source, name):
                setattr(self, name, copy.deepcopy(getattr(source, name)))
        self.constraints = []
        BaseProblem.__init__(self)
        BaseModeLogic.__init__(self)

    def __deepcopy__(self, memo):
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        for name, value in self.__dict__.items():
            if name != '_uniform_sampler':
                setattr(result, name, copy.deepcopy(value, memo))
        result._uniform_sampler = result._make_uniform_sampler()
        return result

    def save(self, path):
        """Freeze the geometry, fixed goals and planning metadata as portable JSON."""
        data = dict(format='comotion-primitive-problem-v1', scene=self.scene.definition,
                    robots=self.robots, robot_idx={r: np.asarray(v).tolist() for r,v in self.robot_idx.items()},
                    robot_dims=self.robot_dims, robot_joints=self.robot_joints,
                    limits=self.limits.tolist(), start=self.start_pos.state().tolist(),
                    slices=self.start_pos._array_slice.tolist(), cost=asdict(self.cost),
                    spec={f.name: getattr(self.spec, f.name).value for f in fields(self.spec)},
                    collision_resolution=self.collision_resolution,
                    collision_tolerance=self.collision_tolerance, manipulating_env=self.manipulating_env,
                    tasks=[dict(name=t.name, robots=t.robots, goal_type=type(t.goal).__name__,
                                goal=t.goal.serialize(), type=t.type, frames=t.frames, side_effect=t.side_effect,
                                side_effect_data=None if t.side_effect_data is None else np.asarray(t.side_effect_data).tolist())
                           for t in self.tasks])
        if isinstance(self, SequenceMixin):
            data['sequence'] = self.sequence
        elif isinstance(self, DependencyGraphMixin):
            data['dependencies'] = {k: sorted(v) for k,v in self.graph.dependencies.items()}
        else:
            raise NotImplementedError('Unsupported mode logic')
        if hasattr(self, 'safe_pose'):
            data['safe_pose'] = {r: np.asarray(v).tolist() for r,v in self.safe_pose.items()}
        Path(path).write_text(json.dumps(data, indent=2) + '\n')

    @classmethod
    def load(cls, path):
        """Load a frozen problem without constructing a RAI scene or recomputing goals."""
        from .planning_env import Task, ProblemSpec
        from .core.dependency_graph import DependencyGraph
        data = json.loads(Path(path).read_text())
        if data.get('format') != 'comotion-primitive-problem-v1':
            raise ValueError('Unsupported primitive problem format')
        if ('sequence' in data) == ('dependencies' in data):
            raise ValueError('Expected one mode-logic specification')
        typ = ConvertedSequenceEnvironment if 'sequence' in data else ConvertedDependencyEnvironment
        env = typ.__new__(typ)
        env.scene = CoMotionScene(data['scene'])
        for name in ('robots', 'robot_dims', 'robot_joints', 'collision_resolution',
                     'collision_tolerance', 'manipulating_env'):
            setattr(env, name, data[name])
        env.robot_idx = {r: np.array(v, dtype=int) for r,v in data['robot_idx'].items()}
        env.limits = np.array(data['limits'])
        env.start_pos = NpConfiguration(np.array(data['start']), data['slices'])
        env.cost = JointSpaceLength(**data['cost'])
        env.spec = ProblemSpec(**{f.name: f.type(data['spec'][f.name]) for f in fields(ProblemSpec)})
        goal_types = {'SingleGoal': SingleGoal, 'GoalSet': GoalSet}
        env.tasks = []
        for saved in data['tasks']:
            record = saved.copy()
            record['goal'] = goal_types[record.pop('goal_type')].from_data(record['goal'])
            if record['side_effect_data'] is not None:
                record['side_effect_data'] = np.asarray(record['side_effect_data'])
            env.tasks.append(Task(**record))
        if 'sequence' in data:
            env.sequence = data['sequence']
        else:
            env.graph = DependencyGraph()
            for name in data['dependencies']:
                env.graph.add_node(name)
            for name, dependencies in data['dependencies'].items():
                for dependency in dependencies:
                    env.graph.add_dependency(name, dependency)
        if 'safe_pose' in data:
            env.safe_pose = {r: np.asarray(v) for r,v in data['safe_pose'].items()}
        env.constraints = []
        BaseProblem.__init__(env)
        BaseModeLogic.__init__(env)
        return env

    def get_scenegraph_info_for_mode(self, mode, is_start_mode=False):
        if not self.manipulating_env:
            return {}
        return self.scene.scenegraph_info(mode)

    def set_to_mode(self, mode):
        self.scene.set_mode(mode)
        self.prev_mode = mode

    def is_collision_free(self, q=None, mode=None, collision_tolerance=None):
        return self.scene.valid(q, mode, collision_tolerance)

    def is_collision_free_np(self, q=None, m=None, collision_tolerance=None, set_mode=True):
        return self.scene.valid(q, m, collision_tolerance, set_mode=set_mode)

    def is_collision_free_for_robot(self, r, q, m=None, collision_tolerance=None, set_mode=True):
        return self.scene.valid(q, m, collision_tolerance, set_mode=set_mode, robots=r)

    def is_edge_collision_free(self, q1, q2, m, resolution=None, tolerance=None,
                               include_endpoints=False, N_start=0, N_max=None, N=None):
        resolution = self.collision_resolution if resolution is None else resolution
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError('Invalid collision resolution')
        N = max(2, int(config_dist(q1, q2, 'max') / resolution) + 1) if N is None else N
        N_max = N if N_max is None else min(N, N_max)
        if N < 2 or not 0 <= N_start <= N_max <= N:
            raise ValueError('Invalid edge sample range')
        return self.scene.edge_valid(q1.state(), q2.state(), m, N, N_start,
                                     N_max, include_endpoints, tolerance)

    def batch_is_collision_free(self, qs, mode=None, collision_tolerance=None):
        return self.scene.batch_valid(qs, mode, collision_tolerance)

    def batch_is_edge_collision_free(self, edges, *, include_endpoints=False):
        """Batch same-mode edge samples on the scalar checker's exact lattice.

        Consecutive mode groups retain attachment-update order. Each wave checks
        at most sixteen binary-ordered samples per surviving edge.
        """
        resolution = self.collision_resolution
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError('Invalid collision resolution')
        free = np.ones(len(edges), dtype=bool)
        begin = 0
        while begin < len(edges):
            mode = edges[begin][2]
            end = begin + 1
            while end < len(edges) and edges[end][2] is mode:
                end += 1
            work = []
            for a, b, _ in edges[begin:end]:
                n = max(2, int(config_dist(a, b, 'max') / resolution) + 1)
                indices = [i for i in generate_binary_search_indices(n)
                           if include_endpoints or i not in (0, n - 1)]
                work.append((a.state(), b.state() - a.state(), np.asarray(indices) / (n - 1)))
            for offset in range(0, max((len(w[2]) for w in work), default=0), 16):
                rows, owners = [], []
                for i, (a, delta, ts) in enumerate(work, begin):
                    if free[i] and offset < len(ts):
                        chunk = ts[offset:offset + 16]
                        rows.append(a + chunk[:, None] * delta)
                        owners.extend([i] * len(chunk))
                if not rows:
                    break
                valid = self.batch_is_collision_free(np.concatenate(rows), mode)
                np.logical_and.at(free, np.asarray(owners), valid)
            begin = end
        return free

    def show(self, blocking=True):
        raise NotImplementedError('Replay saved paths with the original scene viewer')

    def show_config(self, q, blocking=True):
        raise NotImplementedError('Replay saved paths with the original scene viewer')

    def display_path(self, path, **kwargs):
        raise NotImplementedError('Replay saved paths with the original scene viewer')


class ConvertedSequenceEnvironment(SequenceMixin, ConvertedPrimitiveEnvironment):
    def __init__(self, source):
        ConvertedPrimitiveEnvironment.__init__(self, source)


class ConvertedDependencyEnvironment(DependencyGraphMixin, ConvertedPrimitiveEnvironment):
    def __init__(self, source):
        ConvertedPrimitiveEnvironment.__init__(self, source)


def to_comotion(source):
    """Snapshot a supported geometric RAI problem as an independent CoMotion problem.

    Spheres, capsules, cylinders and sharp boxes retain their geometry. Rounded
    boxes, meshes, constraints and dynamic models require explicit conversion.
    Nonzero collision tolerances use the native solver's penetration depths,
    which can differ from RAI near the tolerance boundary.
    """
    if isinstance(source, SequenceMixin):
        return ConvertedSequenceEnvironment(source)
    if isinstance(source, DependencyGraphMixin):
        return ConvertedDependencyEnvironment(source)
    raise NotImplementedError('Only sequence/dependency-graph mode logic is supported')


@register('comotion.converted')
def load_comotion_problem(path):
    return ConvertedPrimitiveEnvironment.load(path)
