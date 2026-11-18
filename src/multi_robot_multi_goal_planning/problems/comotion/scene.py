"""Native SIMD collision checking for frozen primitive manipulation scenes.

The simulator is needed only for export. Prepared attachment modes share immutable
pair lists and switch by handle; full and partial queries use the same kernels.
"""
from collections import OrderedDict
import numpy as np
from comotion import Scene


class _ActiveQueries:
    """Inspection/query interface for the currently selected native mode."""
    def __init__(self, scene):
        self.scene = scene

    def poses(self, q):
        return self.scene.kernel.poses(np.asarray(q), self.scene._handle)

    def batch_valid(self, qs, tolerance=0, group=(), simd=True):
        return self.scene.kernel.batch_valid(np.asarray(qs), self.scene._handle, list(group), simd, tolerance)

    def valid(self, q, tolerance=0, group=()):
        return self.scene.kernel.state_valid(np.asarray(q), self.scene._handle, list(group), tolerance)


class CoMotionScene:
    def __init__(self, definition, *, implementation="generic"):
        if not np.isfinite(definition['tolerance']) or definition['tolerance']<0:
            raise ValueError('Invalid collision tolerance')
        self.edge_packing = "binary"
        self.edge_early_exit = True
        self.implementation = implementation
        self.definition = definition
        self.frame_names = definition['frame_names']
        self.frame_index = {name: i for i, name in enumerate(self.frame_names)}
        self.joint_names = definition['joint_names']
        self.robot_names = definition['robot_names']
        self.shape_frames = np.asarray(definition['shape_frames'], dtype=np.int32)
        self.movable = definition['movable']
        self.actions = definition['actions']
        self.tolerance = float(definition['tolerance'])
        self.kernel = Scene(definition, implementation=implementation)
        self._initial = self.kernel.initial()
        self._handle = self._initial
        self._mode = None
        self._mode_cache = OrderedDict()
        self._info_cache = OrderedDict()
        self.native = _ActiveQueries(self)
        self.last_q = np.asarray(definition['start_q'], dtype=float)
        self.stats = dict(queries=0, mode_updates=0)

    def __deepcopy__(self, memo):
        import copy
        result = type(self)(copy.deepcopy(self.definition, memo), implementation=self.implementation)
        memo[id(self)] = result
        result.edge_packing = self.edge_packing
        result.edge_early_exit = self.edge_early_exit
        result.set_mode(copy.deepcopy(self._mode, memo))
        result.last_q = self.last_q.copy()
        return result

    def save(self, path):
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(self.definition, indent=2) + '\n')

    def _transition(self, previous, mode):
        changed = {a for a, b in zip(mode.prev_mode.task_ids, mode.task_ids) if a != b}
        if len(changed) != 1:
            raise NotImplementedError('Expected one task per transition')
        action = self.actions[changed.pop()]
        parent = obj = -1
        effect = 0
        appear = []
        if action['type'] not in (None, 'goto'):
            parent, obj = (self.frame_index[n] for n in action['frames'][:2])
            if self.frame_names[obj] not in self.movable:
                raise NotImplementedError('Unknown manipulated object')
            if action['side_effect'] == 'make_appear':
                effect, appear = 1, action['side_effect_data']
            elif action['side_effect'] == 'remove':
                effect = 2
            elif action['side_effect'] is not None:
                raise NotImplementedError('Unsupported manipulation side effect')
        elif action['side_effect'] is not None:
            raise NotImplementedError('Unsupported side effect on non-manipulation task')
        self.stats['mode_updates'] += 1
        return self.kernel.transition(previous, action['indices'],
                                      mode.entry_configuration.state(), obj, parent, effect, appear)

    def set_mode(self, mode):
        if mode is self._mode:
            return
        if not self.actions:  # Static multi-goal scenes share one geometry mode.
            self._mode = mode
            return
        missing, ancestor = [], mode
        while ancestor is not None and id(ancestor) not in self._mode_cache:
            missing.append(ancestor)
            ancestor = ancestor.prev_mode
        handle = self._initial if ancestor is None else self._mode_cache[id(ancestor)][1]
        for node in reversed(missing):
            if node.prev_mode is not None:
                handle = self._transition(handle, node)
            self._mode_cache[id(node)] = (node, handle)
            if len(self._mode_cache) > 128:
                self._mode_cache.popitem(last=False)
        self._handle, self._mode = handle, mode

    def scenegraph_info(self, mode):
        self.set_mode(mode)
        key = id(self._handle)
        if key not in self._info_cache:
            frames = self.kernel.info(self._handle)
            poses = np.round(np.asarray([values for _, _, values in frames]), 3)
            poses[np.abs(poses) < 1e-6] = 0.
            result = {
                self.frame_names[frame]: (
                    None if parent < 0 else self.frame_names[parent], pose.tobytes())
                for (frame, parent, _), pose in zip(frames, poses)
            }
            self._info_cache[key] = self._handle, result
            if len(self._info_cache) > 128:
                self._info_cache.popitem(last=False)
        return self._info_cache[key][1]

    def valid(self, q=None, mode=None, tolerance=None, *, set_mode=True, robots=None):
        if set_mode:
            self.set_mode(mode)
        q = self.last_q if q is None else np.asarray(q.state() if hasattr(q, 'state') else q, dtype=float)
        group = [] if robots is None else [self.robot_names.index(r) for r in ([robots] if isinstance(robots, str) else robots)]
        if robots is not None and not group:
            return True
        self.last_q = q.copy()
        self.stats['queries'] += 1
        return self.native.valid(q, self.tolerance if tolerance is None else tolerance, group=group)

    def edge_valid(self, start, end, mode, n, begin, limit, endpoints, tolerance=None):
        self.set_mode(mode)
        valid, last, checked = self.kernel.edge_valid(
            start, end, self._handle, n, begin, limit, endpoints,
            self.tolerance if tolerance is None else tolerance,
            self.edge_packing, self.edge_early_exit)
        self.stats['queries'] += checked
        if last >= 0:
            self.last_q = start + (last / (n - 1)) * (end - start)
        return valid

    def batch_valid(self, qs, mode=None, tolerance=None):
        self.set_mode(mode)
        if not isinstance(qs, np.ndarray):
            qs = [q.state() if hasattr(q, 'state') else q for q in qs]
        qs = np.asarray(qs, dtype=float)
        if qs.size == 0:
            qs = np.empty((0, len(self.last_q)))
        if len(qs):
            self.last_q = qs[-1].copy()
        self.stats['queries'] += len(qs)
        return self.native.batch_valid(qs, self.tolerance if tolerance is None else tolerance)


# Compatibility for clients importing the old adapter name.
SimdPrimitiveScene = CoMotionScene
