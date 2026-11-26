"""Load frozen RAI and MuJoCo problems with archived goals.

Workers load frozen geometry/tasks and never call an IK scene builder.
RAI's text writer rounds numbers, so exact zero-state poses/sizes are restored
from JSON before loading the archived start configuration.
"""
import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import robotic as ry
from multi_robot_multi_goal_planning.problems.rai.base import rai_env
from multi_robot_multi_goal_planning.problems.planning_env import SequenceMixin, DependencyGraphMixin, BaseModeLogic, Task, ProblemSpec
from multi_robot_multi_goal_planning.problems.core.goals import SingleGoal, GoalSet
from multi_robot_multi_goal_planning.problems.costs.joint_space import JointSpaceLength
from multi_robot_multi_goal_planning.problems.core.dependency_graph import DependencyGraph


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenRai(rai_env):
    def __init__(self, directory):
        directory = Path(directory)
        meta = json.loads((directory/'problem.json').read_text())
        if sha(directory/'scene.g') != meta['geometry_sha256']:
            raise ValueError('Frozen geometry changed')
        if len({r['name'] for r in meta['exact_frames']}) != len(meta['exact_frames']):
            raise ValueError('Frozen frame names must be unique')
        self.C = ry.Config()
        self.C.addFile(str(directory/'scene.g'))
        if list(self.C.getJointNames()) != meta['joint_names']:
            raise ValueError('Frozen RAI joint order changed')
        self.C.setJointState(np.zeros(len(meta['start'])))
        for record in meta['exact_frames']:
            f = self.C.getFrame(record['name'])
            if f.getParent() is None:
                f.setPosition(record['pose'][:3]); f.setQuaternion(record['pose'][3:])
            else:
                f.setRelativePosition(record['pose'][:3]); f.setRelativeQuaternion(record['pose'][3:])
            if record['shape'] is not None:
                f.setShape(getattr(ry.ST, record['shape']), record['size'])
            f.setContact(record['contact'])
        self.C.setJointState(np.asarray(meta['start']))
        self.robots = meta['robots']
        rai_env.__init__(self, delete_visual_only=False)
        for r in self.robots:
            if list(self.robot_idx[r]) != meta['robot_idx'][r]:
                raise ValueError('Frozen robot coordinate mapping changed')
        self.limits = np.asarray(meta['limits'])
        self.collision_resolution = meta['collision_resolution']
        self.collision_tolerance = meta['collision_tolerance']
        self.manipulating_env = meta['manipulating_env']
        self.cost = JointSpaceLength(**meta['cost'])
        self.spec = ProblemSpec(**{f.name: f.type(meta['spec'][f.name]) for f in dataclasses.fields(ProblemSpec)})
        self.constraints = []
        self.tasks = []
        for task in meta['tasks']:
            record = copy.deepcopy(task)
            cls = {'SingleGoal': SingleGoal, 'GoalSet': GoalSet}[record.pop('goal_type')]
            record['goal'] = cls.from_data(record['goal'])
            if record['side_effect_data'] is not None:
                record['side_effect_data'] = np.asarray(record['side_effect_data'])
            self.tasks.append(Task(**record))
        if meta.get('dependencies') is not None:
            self.graph = DependencyGraph()
            for name in meta['dependencies']:
                self.graph.add_node(name)
            for name, dependencies in meta['dependencies'].items():
                for dependency in dependencies:
                    self.graph.add_dependency(name, dependency)
        else:
            self.sequence = meta['sequence']
        if meta.get('safe_pose') is not None:
            self.safe_pose = {r: np.asarray(v) for r,v in meta['safe_pose'].items()}
        if meta.get('collision_exclusions'):
            self.collision_exclusions = frozenset(frozenset(pair) for pair in meta['collision_exclusions'])
        BaseModeLogic.__init__(self)
        self.prev_mode = self.start_mode


class FrozenSequence(SequenceMixin, FrozenRai):
    __init__ = FrozenRai.__init__


class FrozenDependencies(DependencyGraphMixin, FrozenRai):
    __init__ = FrozenRai.__init__


def checked_file(directory, name):
    path = (directory / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(directory.resolve()):
        raise ValueError('Input filename escapes the bundle: ' + name)
    return path


def build_env(input_dir, scene):
    directory = Path(input_dir)/scene
    manifest = json.loads((directory/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if sha(checked_file(directory, name)) != expected:
            raise ValueError(f'Frozen input changed: {scene}/{name}')
    if manifest.get('source_backend', 'rai') == 'mujoco':
        from multi_robot_multi_goal_planning.problems.mujoco.envs.roboballet.env import RoboBalletEnvironment
        from multi_robot_multi_goal_planning.assets import ASSET_ROOT
        for name, expected in manifest['package_assets'].items():
            if sha(checked_file(ASSET_ROOT, name)) != expected:
                raise ValueError(f'Package asset differs from frozen benchmark: {name}')
        # Reuse tracked model assets; the explicit instance bypasses IK.
        env = RoboBalletEnvironment(assets=ASSET_ROOT/'roboballet', instance=directory/'instance.json')
        goals = json.loads((directory/'goals.json').read_text())
        if len(goals) != len(env.tasks):
            raise ValueError('Frozen goal count differs from task count')
        for task, goal in zip(env.tasks, goals):
            cls = {'SingleGoal': SingleGoal, 'GoalSet': GoalSet}[goal['type']]
            task.goal = cls.from_data(goal['data'])
        return env, manifest
    meta = json.loads((directory/'problem.json').read_text())
    cls = FrozenDependencies if meta.get('dependencies') is not None else FrozenSequence
    return cls(directory), manifest


def convert_env(env, backend, manifest):
    source = manifest.get('source_backend', 'rai')
    if backend == source:
        return copy.deepcopy(env)
    if source == 'mujoco':
        if backend != 'rai':
            raise ValueError('Native packing inputs support only MuJoCo and RAI')
        from multi_robot_multi_goal_planning.problems.rai.from_mujoco import to_rai
        result = to_rai(env)
        if manifest.get('sharp_boxes'):
            for config in (result.C, result.C_base, result.C_orig):
                for frame in config.getFrames():
                    if frame.info().get('shape') in ('box', 'ssBox'):
                        frame.setShape(ry.ST.box, [*frame.getSize()[:3], 0.])
            result.C_cache.clear()
            result.prev_mode = None
        return result
    if backend == 'comotion':
        from multi_robot_multi_goal_planning.problems.comotion.from_rai import to_comotion
        return to_comotion(env)
    if backend == 'mujoco':
        from multi_robot_multi_goal_planning.problems.mujoco.from_rai import to_mujoco
        return to_mujoco(env)
    raise ValueError('Unsupported backend: ' + backend)

