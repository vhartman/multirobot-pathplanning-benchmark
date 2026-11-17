"""Convert MuJoCo planning environments to independent RAI environments.

Geometry, scalar joints, tasks and ordering are copied once. Collision masks
and attachment exclusions are retained; contact distances still depend on RAI.
Planes use analytic support distances, so no finite floor approximation enters
collision checking. Constraints, callback goals and dynamics are not converted.
"""
import copy
import time
from itertools import combinations

import mujoco
import numpy as np
import robotic as ry
from scipy.spatial.transform import Rotation

from .core.cost import JointSpaceLength
from .core.goals import SingleGoal, GoalSet, GoalRegion
from .mujoco_env import MujocoEnvironment
from .planning_env import (
    BaseProblem, BaseModeLogic, SequenceMixin, DependencyGraphMixin, DynamicsType,
)
from .rai_base_env import rai_env


def _quaternion(rotation):
    return np.roll(rotation.as_quat(), 1)


def _rotation(quaternion):
    return Rotation.from_quat(np.roll(quaternion, -1))


def _collision_pairs(model):
    """MuJoCo's static pair selection, excluding unsupported contact margins."""
    if np.any(model.geom_margin) or np.any(model.geom_gap) or np.any(model.pair_margin) or np.any(model.pair_gap):
        raise NotImplementedError("Nonzero contact margins/gaps are not converted")
    excluded = set(map(int, model.exclude_signature))
    filter_parent = not (model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
    pairs = set()
    for a, b in combinations(range(model.ngeom), 2):
        ba, bb = map(int, model.geom_bodyid[[a, b]])
        wa, wb = map(int, model.body_weldid[[ba, bb]])
        if wa == wb or ((min(ba, bb) << 16) + max(ba, bb)) in excluded:
            continue
        if filter_parent and wa and wb:
            pa, pb = model.body_weldid[model.body_parentid[[wa, wb]]]
            if pa == wb or pb == wa:
                continue
        if ((model.geom_contype[a] & model.geom_conaffinity[b]) or
                (model.geom_contype[b] & model.geom_conaffinity[a])):
            pairs.add((a, b))
    pairs.update(tuple(sorted((int(a), int(b)))) for a, b in zip(model.pair_geom1, model.pair_geom2))
    return pairs


def _import_model(source, metadata):
    model = source.model
    if model.nflex or model.nplugin:
        raise NotImplementedError("Flex and plugin models are not converted")
    if mujoco.get_mjcb_contactfilter() is not None:
        raise NotImplementedError("Custom MuJoCo contact callbacks are not converted")
    data = mujoco.MjData(model)
    data.qpos[:] = source.data.qpos
    data.mocap_pos[:] = source.data.mocap_pos
    data.mocap_quat[:] = source.data.mocap_quat
    data.qpos[source._all_robot_idx] = source.start_pos.state()
    mujoco.mj_kinematics(model, data)
    for kind, count in [(mujoco.mjtObj.mjOBJ_BODY, model.nbody),
                        (mujoco.mjtObj.mjOBJ_SITE, model.nsite)]:
        for i in range(count):
            if "__mj_" in (mujoco.mj_id2name(model, kind, i) or ""):
                raise ValueError("Source frame names must not contain the reserved prefix __mj_")
    C = ry.Config()
    body_names = {i: model.body(i).name or f"__mj_body_{i}" for i in range(model.nbody)}
    if len(set(body_names.values())) != model.nbody:
        raise ValueError("Body frame names must be unique")
    C.addFrame(body_names[0])
    movable = set(metadata['attachments'])
    joint_map, robot_joints = {}, {}
    ordered_names = [None] * len(source.start_pos.state())
    for robot in source.robots:
        names = []
        if len(source.robot_joints[robot]) != len(source.robot_idx[robot]):
            raise NotImplementedError("Robot coordinates must be scalar hinge/slide joints")
        for index, name in zip(source.robot_idx[robot], source.robot_joints[robot]):
            j = model.joint(name).id
            if j in joint_map:
                raise NotImplementedError("Shared robot joints are not converted")
            if model.jnt_type[j] not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                raise NotImplementedError("Active ball/free joints are not converted")
            if int(source._all_robot_idx[index]) != int(model.jnt_qposadr[j]):
                raise ValueError("Robot joint names and configuration indices disagree")
            exported = f"{robot}/__mj_joint_{j}"
            joint_map[j] = exported
            names.append(exported)
            ordered_names[index] = exported
        robot_joints[robot] = names
    if any(n is None for n in ordered_names):
        raise ValueError("Robot coordinates must partition the configuration")
    for i in range(model.neq):
        if (model.eq_type[i] != int(mujoco.mjtEq.mjEQ_JOINT) or
                int(model.eq_obj1id[i]) in joint_map or int(model.eq_obj2id[i]) in joint_map):
            raise NotImplementedError("Only equality couplings between frozen passive joints are supported")
    for b in range(1, model.nbody):
        name, parent = body_names[b], body_names[int(model.body_parentid[b])]
        joints = list(range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]))
        floating = any(model.jnt_type[j] == int(mujoco.mjtJoint.mjJNT_FREE) for j in joints)
        if floating or model.body_mocapid[b] >= 0:
            if any(j in joint_map for j in joints):
                raise NotImplementedError("Active floating bodies are not converted")
            if floating and name not in movable:
                raise NotImplementedError(f"Free body {name!r} has no attachment metadata")
            C.addFrame(name, body_names[0]).setRelativePosition(data.xpos[b]).setRelativeQuaternion(data.xquat[b])
            continue
        origin = f"__mj_origin_{b}"
        C.addFrame(origin, parent).setRelativePosition(model.body_pos[b]).setRelativeQuaternion(model.body_quat[b])
        tail = origin
        for j in joints:
            typ = int(model.jnt_type[j])
            if typ not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                raise NotImplementedError(f"Unsupported joint type on {name!r}")
            axis = model.jnt_axis[j]
            alignment = Rotation.align_vectors([axis], [[0., 0., 1.]])[0]
            pivot = model.jnt_pos[j] if typ == int(mujoco.mjtJoint.mjJNT_HINGE) else np.zeros(3)
            reference = float(model.qpos0[model.jnt_qposadr[j]])
            frame_name = joint_map.get(j, f"__mj_fixed_joint_{j}")
            anchor_name = f"__mj_before_joint_{j}"
            frame = C.addFrame(anchor_name, tail)
            if typ == int(mujoco.mjtJoint.mjJNT_HINGE):
                frame.setRelativePosition(pivot).setRelativeQuaternion(
                    _quaternion(alignment * Rotation.from_rotvec([0, 0, -reference])))
                joint_type = ry.JT.hingeZ
            else:
                frame.setRelativePosition(-axis * reference).setRelativeQuaternion(_quaternion(alignment))
                joint_type = ry.JT.transZ
            C.addFrame(frame_name, anchor_name).setJoint(
                joint_type, model.jnt_range[j] if model.jnt_limited[j] else [-1e10, 1e10])
            C.setJointState([data.qpos[model.jnt_qposadr[j]]], [frame_name])
            tail = f"__mj_after_joint_{j}"
            C.addFrame(tail, frame_name).setRelativePosition(-alignment.inv().apply(pivot)).setRelativeQuaternion(_quaternion(alignment.inv()))
        C.addFrame(name, tail)
    C.selectJoints(ordered_names)
    C.setJointState(source.start_pos.state(), ordered_names)
    for i in range(model.nsite):
        name = model.site(i).name
        if not name:
            continue
        if name in body_names.values():
            raise NotImplementedError(f"Ambiguous site/body frame name: {name}")
        C.addFrame(name, body_names[int(model.site_bodyid[i])]).setRelativePosition(model.site_pos[i]).setRelativeQuaternion(model.site_quat[i])
    pairs = _collision_pairs(model)
    active_geoms = {g for pair in pairs for g in pair}
    geom_names, supports, planes = {}, {}, []
    for g in range(model.ngeom):
        name = f"__mj_geom_{g}"
        geom_names[g] = name
        frame = C.addFrame(name, body_names[int(model.geom_bodyid[g])])
        frame.setJoint(ry.JT.rigid)
        frame.setRelativePosition(model.geom_pos[g]).setRelativeQuaternion(model.geom_quat[g])
        kind, size = int(model.geom_type[g]), model.geom_size[g].copy()
        rgba = model.geom_rgba[g].copy()
        if model.geom_matid[g] >= 0:
            rgba = model.mat_rgba[model.geom_matid[g]].copy()
        if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = int(model.geom_dataid[g])
            a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[a:a+n].astype(float)
            if g in active_geoms:
                frame.setConvexMesh(vertices)
            else:
                a, n = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
                frame.setMesh(vertices, model.mesh_face[a:a+n].astype(np.uint32))
            supports[g] = ('mesh', vertices)
        elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            frame.setShape(ry.ST.box, 2*size)
            supports[g] = ('box', size)
        elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            frame.setShape(ry.ST.sphere, [size[0]])
            supports[g] = ('sphere', size)
        elif kind in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            capsule = kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
            frame.setShape(ry.ST.capsule if capsule else ry.ST.cylinder, [2*size[1], size[0]])
            supports[g] = ('capsule' if capsule else 'cylinder', size)
        elif kind == int(mujoco.mjtGeom.mjGEOM_PLANE):
            planes.append(g)
            # Display only. Collision uses the infinite plane equation below.
            frame.setShape(ry.ST.box, [2*max(size[0], 3), 2*max(size[1], 3), .001])
        else:
            raise NotImplementedError(f"Unsupported geom type {kind} on {model.geom(g).name!r}")
        frame.setColor(rgba)
        # Keep RAI's automatic rigid-neighborhood filtering from removing source pairs.
        frame.setContact(int(g in active_geoms and g not in planes))
    for a, b in pairs:
        if a in planes and b in planes:
            raise NotImplementedError("Plane-plane contacts are not supported")
    initial_attachments = copy.deepcopy(metadata['attachments'])
    for name, (parent, position, quaternion) in initial_attachments.items():
        if name not in body_names.values():
            raise ValueError(f"Unknown attachment body: {name}")
        # Attach cuts at the nearest joint; give the object its own boundary.
        frame = C.getFrame(name)
        position_world, quaternion_world = frame.getPosition(), frame.getQuaternion()
        frame.setJoint(ry.JT.rigid).setPosition(position_world).setQuaternion(quaternion_world)
        C.attach(parent, name)
        C.getFrame(name).setRelativePosition(position).setRelativeQuaternion(quaternion)
    for task in source.tasks:
        for name in task.frames or []:
            if name not in set(body_names.values()) | {model.site(i).name for i in range(model.nsite)}:
                raise ValueError(f"Unknown task frame: {name}")
    # Names map source geoms to their original bodies, independent of attachments.
    geom_bodies = [body_names[int(b)] for b in model.geom_bodyid]
    body_robot = {}
    for b, name in body_names.items():
        current = b
        while current:
            if body_names[current] in source.robots:
                body_robot[name] = body_names[current]
                break
            current = int(model.body_parentid[current])
    for i in range(model.nsite):
        if model.site(i).name:
            owner = body_robot.get(body_names[int(model.site_bodyid[i])])
            if owner is not None:
                body_robot[model.site(i).name] = owner
    return C, dict(joint_names=ordered_names, robot_joints=robot_joints,
                   geom_names=geom_names, geom_bodies=geom_bodies, body_robot=body_robot,
                   collision_pairs=pairs, supports=supports, planes=planes,
                   initial_attachments=initial_attachments,
                   attachment_exclusions=copy.deepcopy(metadata['attachment_exclusions']),
                   tolerance_mode=metadata['tolerance_mode'],
                   check_joint_limits=metadata['check_joint_limits'])


class _ConvertedRaiEnvironment(rai_env):
    def __init__(self, source):
        if source.constraints or any(t.constraints for t in source.tasks):
            raise NotImplementedError("Path constraints require a RAI-specific implementation")
        if source.spec.dynamics != DynamicsType.GEOMETRIC or not isinstance(source.cost, JointSpaceLength):
            raise NotImplementedError("Only geometric planning with joint-space costs is converted")
        for task in source.tasks:
            if task.type in ('pick', 'place') and (not task.frames or len(task.frames) != 2):
                raise ValueError('Pick/place tasks need [parent frame, object frame]')
            if type(task.goal) not in (SingleGoal, GoalSet, GoalRegion):
                raise NotImplementedError("Only fixed goals, goal sets and joint-space goal regions are converted")
            if task.type not in (None, 'goto', 'pick', 'place') or task.side_effect is not None:
                raise NotImplementedError("Unsupported task action or side effect")
        self.C, metadata = _import_model(source, source.get_conversion_metadata())
        self.__dict__.update(metadata)
        for name in ['robots', 'robot_idx', 'robot_dims', 'limits', 'start_pos', 'tasks',
                     'spec', 'cost', 'collision_resolution', 'collision_tolerance', 'safe_pose']:
            if hasattr(source, name):
                setattr(self, name, copy.deepcopy(getattr(source, name)))
        self.constraints = []
        self._copy_task_order(source)
        self.manipulating_env = bool(self.initial_attachments)
        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)
        self.C_orig = ry.Config()
        self.C_orig.addConfigurationCopy(self.C)
        self.C_cache = {}
        self.prev_mode = None
        self._geom_index = {name: g for g, name in self.geom_names.items()}
        BaseProblem.__init__(self)
        BaseModeLogic.__init__(self)
        self.set_to_mode(self.start_mode)
        self.C.setJointState(self.start_pos.state(), self.joint_names)

    def __deepcopy__(self, memo):
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key == '_uniform_sampler':
                continue
            if isinstance(value, ry.Config):
                copied = ry.Config()
                copied.addConfigurationCopy(value)
                setattr(result, key, copied)
            else:
                setattr(result, key, copy.deepcopy(value, memo))
        result._uniform_sampler = result._make_uniform_sampler()
        return result

    def _apply_attachments(self, config, attachments):
        for name, (parent, position, quaternion) in attachments.items():
            config.attach(parent, name)
            config.getFrame(name).setRelativePosition(position).setRelativeQuaternion(quaternion)

    def get_scenegraph_info_for_mode(self, mode, is_start_mode=False):
        if is_start_mode or mode.prev_mode is None:
            return copy.deepcopy(self.initial_attachments)
        tmp = ry.Config()
        tmp.addConfigurationCopy(self.C_base)
        self._apply_attachments(tmp, mode.prev_mode.sg)
        tmp.setJointState(mode.entry_configuration.state(), self.joint_names)
        changed = {a for a, b in zip(mode.prev_mode.task_ids, mode.task_ids) if a != b}
        for index in sorted(changed):
            task = self.tasks[index]
            if task.type in ('pick', 'place'):
                tmp.attach(*task.frames)
        return {name: (tmp.getFrame(name).getParent().name,
                       tuple(tmp.getFrame(name).getRelativePosition()),
                       tuple(tmp.getFrame(name).getRelativeQuaternion()))
                for name in self.initial_attachments}

    def set_to_mode(self, mode, config=None, **kwargs):
        mode = self.start_mode if mode is None else mode
        target = self.C if config is None else config
        if config is None and self.prev_mode is mode:
            return
        self._apply_attachments(target, mode.sg)
        if config is None:
            self.prev_mode = mode
            self._active_attachments = mode.sg

    def _pair_is_relevant(self, a, b, robots=None):
        if tuple(sorted((a, b))) not in self.collision_pairs:
            return False
        for obj, (parent, _, _) in self._active_attachments.items():
            ignored = self.attachment_exclusions.get(parent, ())
            # Grasp exclusions refer to geom IDs; descendants of held bodies count too.
            if ((self._belongs_to(a, obj) and b in ignored) or
                    (self._belongs_to(b, obj) and a in ignored)):
                return False
        if robots is not None:
            return any(self._geom_owner(g) in robots for g in (a, b))
        return True

    def _belongs_to(self, geom, ancestor):
        frame = self.C.getFrame(self.geom_names[geom])
        while frame is not None:
            if frame.name == ancestor:
                return True
            frame = frame.getParent()
        return False

    def _geom_owner(self, geom):
        frame = self.C.getFrame(self.geom_names[geom])
        while frame is not None:
            if frame.name in self.body_robot:
                return self.body_robot[frame.name]
            frame = frame.getParent()
        return None

    def _plane_distance(self, plane, geom):
        p, g = self.C.getFrame(self.geom_names[plane]), self.C.getFrame(self.geom_names[geom])
        normal = _rotation(p.getQuaternion()).apply([0, 0, 1])
        local = _rotation(g.getQuaternion()).inv().apply(normal)
        kind, size = self.supports[geom]
        if kind == 'mesh':
            extent = -float(np.min(size @ local))
        elif kind == 'box':
            extent = float(np.abs(local) @ size)
        elif kind == 'sphere':
            extent = size[0]
        elif kind == 'capsule':
            extent = size[0] + abs(local[2])*size[1]
        else:
            extent = size[0]*np.linalg.norm(local[:2]) + abs(local[2])*size[1]
        return float(normal @ (g.getPosition()-p.getPosition()) - extent)

    def _contacts_valid(self, tolerance, robots=None):
        threshold = -abs(tolerance) if self.tolerance_mode == 'penetration' else tolerance
        self.C.computeCollisions()
        for first, second, distance in self.C.getCollisions(0.):
            a, b = self._geom_index[first], self._geom_index[second]
            if distance < threshold and self._pair_is_relevant(a, b, robots):
                return False
        for plane in self.planes:
            for a, b in self.collision_pairs:
                if plane not in (a, b):
                    continue
                other = b if a == plane else a
                if self._pair_is_relevant(plane, other, robots):
                    distance = self._plane_distance(plane, other)
                    if distance <= 0 and distance < threshold:
                        return False
        return True

    def is_collision_free(self, q=None, m=None, collision_tolerance=None):
        return self.is_collision_free_np(q, m, collision_tolerance)

    def is_collision_free_np(self, q, m, collision_tolerance=None, set_mode=True):
        if q is not None:
            q = np.asarray(q.state() if hasattr(q, 'state') else q)
            if q.shape != self.limits[0].shape or not np.isfinite(q).all():
                return False
            if self.check_joint_limits and (np.any(q < self.limits[0]-1e-10) or np.any(q > self.limits[1]+1e-10)):
                return False
        if set_mode:
            self.set_to_mode(m)
        if q is not None:
            self.C.setJointState(q, self.joint_names)
        return self._contacts_valid(self.collision_tolerance if collision_tolerance is None else collision_tolerance)

    def is_collision_free_for_robot(self, r, q, m=None, collision_tolerance=None, set_mode=True):
        selected = {r} if isinstance(r, str) else set(r)
        if not selected.issubset(self.robots):
            raise ValueError('Unknown robot')
        q = np.asarray(q.state() if hasattr(q, 'state') else q)
        if q.shape != self.limits[0].shape or not np.isfinite(q).all():
            return False
        if self.check_joint_limits and (np.any(q < self.limits[0]-1e-10) or np.any(q > self.limits[1]+1e-10)):
            return False
        if set_mode:
            self.set_to_mode(m)
        self.C.setJointState(q, self.joint_names)
        return self._contacts_valid(self.collision_tolerance if collision_tolerance is None else collision_tolerance, selected)

    def show_config(self, q, blocking=True):
        self.C.setJointState(q.state() if hasattr(q, 'state') else q, self.joint_names)
        self.show(blocking)

    def display_path(self, path, stop=True, pause_time=.01, stop_at_end=False, **kwargs):
        for state in path:
            self.set_to_mode(state.mode)
            self.show_config(state.q, blocking=stop)
            time.sleep(pause_time)
        if stop_at_end:
            self.show(True)


class ConvertedSequenceEnvironment(SequenceMixin, _ConvertedRaiEnvironment):
    def __init__(self, source):
        _ConvertedRaiEnvironment.__init__(self, source)

    def _copy_task_order(self, source):
        self.sequence = list(source.sequence)


class ConvertedDependencyEnvironment(DependencyGraphMixin, _ConvertedRaiEnvironment):
    def __init__(self, source):
        _ConvertedRaiEnvironment.__init__(self, source)

    def _copy_task_order(self, source):
        self.graph = copy.deepcopy(source.graph)


def to_rai(source):
    """Snapshot a MuJoCo environment; later source edits require reconversion.

    Supports scalar hinge/slide joints, rigid pick/place, primitive/convex mesh
    collision geometry, fixed goals/goal sets/regions, and sequence/dependency
    ordering. The result retains no source environment or MuJoCo model/data.
    """
    if not isinstance(source, MujocoEnvironment):
        raise TypeError('Expected a MujocoEnvironment')
    if isinstance(source, SequenceMixin):
        return ConvertedSequenceEnvironment(source)
    if isinstance(source, DependencyGraphMixin):
        return ConvertedDependencyEnvironment(source)
    raise NotImplementedError('Only sequence and dependency-graph task logic is converted')
