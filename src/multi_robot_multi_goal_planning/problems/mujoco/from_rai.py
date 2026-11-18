"""Snapshot geometric RAI planning problems as independent MuJoCo problems."""
import copy
import time
import numpy as np
import mujoco

from ..costs.joint_space import JointSpaceLength
from ..core.goals import SingleGoal, GoalSet, GoalRegion
from .base import MujocoEnvironment
from ..planning_env import (
    BaseProblem,
    BaseModeLogic,
    SequenceMixin,
    DependencyGraphMixin,
    DynamicsType,
)
from .rai_scene import RaiMujocoScene


class ConvertedMujocoEnvironment(MujocoEnvironment):
    """One collision model plus copied task data, with no retained RAI objects.

    Path constraints and callback goals need explicit MuJoCo implementations and
    are rejected here. Scene-only conversion supports their geometry separately.
    """

    def __init__(self, source, *, plain_boxes=False):
        if getattr(source, 'constraints', []) or any(t.constraints for t in source.tasks):
            raise NotImplementedError('Path constraints are not converted')
        if source.spec.dynamics != DynamicsType.GEOMETRIC:
            raise NotImplementedError('Only geometric planning is converted')
        if type(source.cost) is not JointSpaceLength:
            raise NotImplementedError('Only joint-space length costs are converted')
        if any(type(t.goal) not in (SingleGoal, GoalSet, GoalRegion) for t in source.tasks):
            raise NotImplementedError('Only fixed goals, goal sets, and joint-space goal regions are converted')
        self.scene = RaiMujocoScene(source, plain_boxes=plain_boxes)
        for name in ['robots', 'robot_idx', 'robot_dims', 'limits',
                     'start_pos', 'tasks', 'spec', 'cost', 'collision_resolution',
                     'collision_tolerance',
                     'safe_pose']:
            if hasattr(source, name):
                setattr(self, name, copy.deepcopy(getattr(source, name)))
        self._copy_task_order(source)
        self.constraints = []
        self.manipulating_env = bool(self.scene.movable)
        self.model, self.data = self.scene.model, self.scene.data
        self._all_robot_idx = self.scene.qadr
        joints = {int(self.model.jnt_qposadr[j]): self.model.joint(j).name
                  for j in range(self.model.njnt)}
        self.robot_joints = {r: [joints[int(self._all_robot_idx[i])]
                                for i in self.robot_idx[r]] for r in self.robots}
        self._collision_batch = None
        self._batch_collision_edges = False
        self.viewer = None
        self._enter_pressed = False
        BaseProblem.__init__(self)
        BaseModeLogic.__init__(self)

    def __deepcopy__(self, memo):
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        memo[id(self.scene.spec)] = self.scene.spec.copy()
        model = copy.deepcopy(self.model, memo)
        data = mujoco.MjData(model)
        state_type = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.model, state_type))
        mujoco.mj_getState(self.model, self.data, state, state_type)
        mujoco.mj_setState(model, data, state, state_type)
        memo[id(self.data)] = data
        for name, value in self.__dict__.items():
            if name not in ('viewer', '_collision_batch', '_uniform_sampler'):
                setattr(result, name, copy.deepcopy(value, memo))
        result.viewer = result._collision_batch = None
        result._batch_collision_edges = False
        result._enter_pressed = False
        # Mode-cache keys are object IDs and must not survive copying.
        result.scene._mode_cache.clear()
        result.scene.update(data.qpos[result._all_robot_idx].copy())
        result._uniform_sampler = result._make_uniform_sampler()
        return result

    def get_conversion_metadata(self):
        raise NotImplementedError(
            'Converting RAI-derived contact policies back to RAI is not supported')

    def get_scenegraph_info_for_mode(self, mode, is_start_mode=False):
        if not self.manipulating_env:
            return {}
        self.scene.set_mode(mode)
        state = self.scene._active_state
        result = {}
        for name, record in state['attachments'].items():
            # Include all manipulated roots, irrespective of frame naming.
            pose = np.round(record['pose'], 3)
            pose[np.abs(pose) < 1e-6] = 0.0
            result[name] = (record['parent'], pose.tobytes(),
                            int(state['contacts'][self.scene.frame_index[name]]))
        return result

    def set_to_mode(self, mode):
        self.scene.set_mode(mode)
        self.prev_mode = mode

    def is_collision_free(self, q=None, mode=None, collision_tolerance=None):
        return self.scene.valid(q, mode, collision_tolerance)

    def is_collision_free_np(self, q=None, m=None, collision_tolerance=None, set_mode=True):
        return self.scene.valid(q, m, collision_tolerance, set_mode=set_mode)

    def is_collision_free_for_robot(self, r, q, m=None, collision_tolerance=None, set_mode=True):
        return self.scene.valid(q, m, collision_tolerance, set_mode=set_mode, robots=r)

    def _update_collision_state(self, q, mode):
        self.scene.set_mode(mode)
        self.scene.update(q)

    def _prepare_collision_state(self, q, mode):
        self.scene.set_mode(mode)
        self.scene.update(q, collision=False)

    def _contacts_are_valid(self, tolerance):
        return self.scene.contacts_valid(tolerance)

    def show_config(self, q, blocking=True):
        self.scene.update(q.state() if hasattr(q, 'state') else np.asarray(q))
        self.show(blocking)

    def display_path(self, path, stop=True, pause_time=0.01, stop_at_end=False, **kwargs):
        for state in path:
            self.set_to_mode(state.mode)
            self.show_config(state.q, blocking=stop)
            time.sleep(pause_time)
        if stop_at_end:
            self.show(blocking=True)


class ConvertedSequenceEnvironment(SequenceMixin, ConvertedMujocoEnvironment):
    def __init__(self, source, *, plain_boxes=False):
        ConvertedMujocoEnvironment.__init__(self, source, plain_boxes=plain_boxes)

    def _copy_task_order(self, source):
        self.sequence = list(source.sequence)


class ConvertedDependencyEnvironment(DependencyGraphMixin, ConvertedMujocoEnvironment):
    def __init__(self, source, *, plain_boxes=False):
        ConvertedMujocoEnvironment.__init__(self, source, plain_boxes=plain_boxes)

    def _copy_task_order(self, source):
        self.graph = copy.deepcopy(source.graph)


def to_mujoco(source, *, plain_boxes=False):
    """Snapshot a RAI problem, including tasks, cost, and task ordering.

    Reconvert after editing C_base or the problem definition. plain_boxes
    explicitly replaces rounded boxes with sharp ones without editing the source.
    The result retains no RAI objects and uses the shared MuJoCo checker/batcher.
    """
    from ..rai.base import rai_env
    if not isinstance(source, rai_env):
        raise TypeError('Expected a RAI environment')
    if isinstance(source, SequenceMixin):
        return ConvertedSequenceEnvironment(source, plain_boxes=plain_boxes)
    if isinstance(source, DependencyGraphMixin):
        return ConvertedDependencyEnvironment(source, plain_boxes=plain_boxes)
    raise NotImplementedError('Only sequence and dependency-graph task logic is converted')
