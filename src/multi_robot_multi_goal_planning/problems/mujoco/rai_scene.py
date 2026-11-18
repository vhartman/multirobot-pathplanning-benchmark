"""One-time RAI scene conversion to an independent MuJoCo collision model.

The exported scene retains no RAI environment or configuration. Supported:
scalar hinges/translations, planar/trans3 joints, and primitive geometry.
Rounded boxes require explicit ``plain_boxes=True`` conversion. Other
unsupported shapes/joints fail explicitly; meshes are not silently convexified. No physics is simulated.
"""

from itertools import product, combinations
from collections import OrderedDict
from pathlib import Path
import json
import copy
import numpy as np
import mujoco

_AXES = dict(X=[1.0, 0.0, 0.0], Y=[0.0, 1.0, 0.0], Z=[0.0, 0.0, 1.0])
_JOINTS = {
    **{f"hinge{a}": [("hinge", a)] for a in "XYZ"},
    **{f"trans{a}": [("slide", a)] for a in "XYZ"},
    "transXY": [("slide", "X"), ("slide", "Y")],
    "trans3": [("slide", "X"), ("slide", "Y"), ("slide", "Z")],
    "transXYPhi": [("slide", "X"), ("slide", "Y"), ("hinge", "Z")],
    "phiTransXY": [("hinge", "Z"), ("slide", "X"), ("slide", "Y")],
}


def _parts(frame, plain_boxes=False):
    """Return primitive geometry in frame coordinates."""
    kind = frame.info().get("shape")
    size = np.asarray(frame.getSize())

    def part(kind, size, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
        return dict(
            type=getattr(mujoco.mjtGeom, "mjGEOM_" + kind.upper()),
            size=size,
            pos=pos,
            quat=quat,
        )

    if kind == "box":
        return [part("box", size[:3] / 2)]
    if kind == "sphere":
        return [part("sphere", [size[-1], 0, 0])]
    if kind in ("capsule", "cylinder"):
        return [part(kind, [size[-1], size[-2] / 2, 0])]
    if kind == "ssBox" and (size[3] == 0 or plain_boxes):
        return [part("box", size[:3] / 2)]
    raise NotImplementedError(f"{frame.name}: unsupported collision shape {kind!r}")


class RaiMujocoScene:
    def __init__(self, env, *, plain_boxes=False):
        """Snapshot C_base and task actions; reconvert after editing the source.

        plain_boxes explicitly replaces rounded boxes by sharp boxes of the
        same outer dimensions. It does not modify the RAI environment.
        """
        from ..rai.base import rai_env
        if isinstance(env, rai_env):
            for method in ("set_to_mode", "is_collision_free", "is_collision_free_np",
                           "is_collision_free_for_robot"):
                if getattr(type(env), method) is not getattr(rai_env, method):
                    raise NotImplementedError("Custom RAI attachment/collision policies need an explicit converter")
        for task in env.tasks:
            if task.type not in (None, "goto"):
                if task.frames is None or len(task.frames) != 2:
                    raise ValueError("Attachment tasks require [parent frame, object frame]")
            if task.side_effect not in (None, "make_appear", "remove"):
                raise NotImplementedError(f"Unsupported task side effect: {task.side_effect}")
            if task.side_effect is not None and task.type in (None, "goto"):
                raise NotImplementedError("Side effects require an attachment task")
            if task.side_effect == "make_appear":
                pose = np.asarray(task.side_effect_data, dtype=float)
                if pose.shape != (7,) or not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]), 1):
                    raise ValueError("make_appear requires a finite position and unit quaternion")
        # RAI is consulted only here, while exporting the scene and task actions.
        self.collision_tolerance = float(env.collision_tolerance)
        self.stats = dict(queries=0, mode_updates=0, compilations=2)
        self._compile(env, plain_boxes)
        self.start_q = env.start_pos.state().copy()
        self.robot_names = list(env.robots)
        self.frame_robot = {
            self.coordinate_frames[int(index)]: self.robot_names.index(robot)
            for robot in self.robot_names for index in env.robot_idx[robot]
        }
        self.actions = []
        for task in env.tasks:
            self.actions.append(dict(
                type=task.type, frames=list(task.frames or []),
                indices=np.concatenate([env.robot_idx[r] for r in task.robots]).tolist(),
                side_effect=task.side_effect,
                side_effect_data=(None if task.side_effect_data is None
                                  else np.asarray(task.side_effect_data).tolist()),
            ))
        self.initial_state = dict(
            attachments=self.initial_attachments,
            contacts=self.contacts.tolist(), joints=self.has_joint.tolist(),
            parents=self.parents.tolist(), q=self.start_q.tolist(),
        )
        self._mode_cache = OrderedDict()
        self._pair_mask_cache = OrderedDict()
        self._mode = None
        self._active_state = None
        mujoco.mj_kinematics(self.model, self.data)
        self.set_state(self.initial_state)
        self.update(self.start_q)
        for i, j in combinations(np.unique(self.geom_frames), 2):
            expected = frozenset((self.frame_names[i], self.frame_names[j])) in self._initial_pairs
            if self.allowed[i, j] != expected:
                raise NotImplementedError("RAI contact filtering is not represented by this exporter")
        del self._initial_pairs

    def _compile(self, env, plain_boxes):
        import robotic as ry
        C = ry.Config()
        C.addConfigurationCopy(env.C_base)
        active_names = set(C.getJointNames())
        frame_coordinates = {}
        for frame in C.getFrames():
            axes = _JOINTS.get(frame.info().get("joint"), ())
            keys = [frame.name if len(axes) == 1 else f"{frame.name}:{i}"
                    for i in range(len(axes))]
            frame_coordinates[frame.name] = [key for key in keys if key in active_names]
        self.joint_names = [None] * len(env.start_pos.state())
        for robot in env.robots:
            names = []
            for name in env.robot_joints[robot]:
                names.extend(frame_coordinates.get(name, [name]))
            if len(names) != len(env.robot_idx[robot]):
                raise ValueError("Robot joint names and coordinate indices disagree")
            for index, name in zip(env.robot_idx[robot], names):
                if not 0 <= index < len(self.joint_names) or self.joint_names[index] is not None:
                    raise ValueError("Robot indices must partition the configuration")
                self.joint_names[index] = name
        if (None in self.joint_names or len(set(self.joint_names)) != len(self.joint_names)
                or set(self.joint_names) != set(C.getJointNames())):
            raise NotImplementedError("Robot coordinates must cover the active RAI joints exactly")
        coordinate_index = {name: i for i, name in enumerate(self.joint_names)}
        self.coordinate_frames = [None] * len(self.joint_names)
        C.setJointState(np.zeros(len(self.joint_names)))
        frames = C.getFrames()
        self.frame_names = [f.name for f in frames]
        self.frame_index = {name: i for i, name in enumerate(self.frame_names)}
        self.movable = {
            task.frames[1]
            for task in env.tasks
            if getattr(task, "type", None) not in (None, "goto")
            and len(task.frames) > 1
        }
        unknown = {name for task in env.tasks for name in (task.frames or [])} - set(self.frame_names)
        if unknown:
            raise NotImplementedError(f"Unknown manipulated frames: {unknown}")
        for name in self.movable:
            frame = C.getFrame(name)
            if frame.info().get("joint") not in (None, "rigid"):
                raise NotImplementedError("Manipulated roots must be rigid objects")
        spec = mujoco.MjSpec()
        spec.compiler.fusestatic = False
        spec.compiler.degree = False
        spec.option.gravity = [0, 0, 0]
        spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
        bodies = {}
        joint_mapping = {}
        geom_frames = {}
        geoms_by_frame = {}
        box_meshes = {}
        approximated_boxes = []

        self.parents = np.array([
            self.frame_index[f.getParent().name] if f.getParent() is not None else -1
            for f in frames], dtype=int)
        self.has_joint = np.array([f.info().get("joint") is not None for f in frames])
        self.contacts = np.array([int(f.info().get("contact", 0)) for f in frames])
        self.initial_attachments = {
            name: dict(parent=(C.getFrame(name).getParent().name
                               if C.getFrame(name).getParent() is not None else None),
                       pose=np.r_[C.getFrame(name).getRelativePosition(), C.getFrame(name).getRelativeQuaternion()].tolist())
            for name in self.movable
        }
        # Add ancestors recursively: reparenting can make frame IDs non-topological.
        def body_for(frame):
            if frame.name in bodies:
                return bodies[frame.name]
            parent = frame.getParent()
            floating = frame.name in self.movable
            parent_body = (
                spec.worldbody if parent is None or floating else body_for(parent)
            )
            position = (
                frame.getPosition()
                if parent is None or floating
                else frame.getRelativePosition()
            )
            quaternion = (
                frame.getQuaternion()
                if parent is None or floating
                else frame.getRelativeQuaternion()
            )
            body = parent_body.add_body(name=frame.name, pos=position, quat=quaternion)
            bodies[frame.name] = body
            if floating:
                body.add_freejoint(name="object/" + frame.name)
            else:
                info = frame.info()
                kind = info.get("joint")
                if kind not in (None, "rigid"):
                    if (
                        kind not in _JOINTS
                        or info.get("mimic")
                        or info.get("scale", 1.0) != 1.0
                    ):
                        raise NotImplementedError(
                            f"{frame.name}: unsupported joint {info}"
                        )
                    axes = _JOINTS[kind]
                    keys = [frame.name if len(axes) == 1 else f"{frame.name}:{k}"
                            for k in range(len(axes))]
                    active = [key in coordinate_index for key in keys]
                    if not any(active):
                        return body  # Passive joint pose is already in the fixed body transform.
                    if not all(active):
                        raise NotImplementedError(f"Partially active joint: {frame.name}")
                    for k, (typ, axis) in enumerate(axes):
                        key = frame.name if len(axes) == 1 else f"{frame.name}:{k}"
                        if key not in self.joint_names:
                            raise NotImplementedError(
                                f"Inactive/mimic joint not supported: {key}"
                            )
                        jname = f"joint/{key}"
                        limits = np.asarray(env.limits)[:, coordinate_index[key]]
                        bounded = bool(np.isfinite(limits).all() and limits[0] < limits[1])
                        body.add_joint(
                            name=jname,
                            type=getattr(mujoco.mjtJoint, "mjJNT_" + typ.upper()),
                            axis=_AXES[axis],
                            limited=bounded,
                            range=limits if bounded else [0, 0],
                        )
                        joint_mapping[coordinate_index[key]] = jname
                        self.coordinate_frames[coordinate_index[key]] = frame.name
                    # Collision-only FK still needs valid inertial data to compile.
                    body.mass = 1.0
                    body.inertia = [1.0, 1.0, 1.0]
            return body

        for f in frames:
            body_for(f)
        for f in frames:
            info = f.info()
            if (not info.get("contact") and f.name not in self.movable) or info.get(
                "shape"
            ) in (None, "none", "marker"):
                continue
            geoms_by_frame[f.name] = []
            if info.get("shape") == "ssBox" and f.getSize()[3] != 0 and plain_boxes:
                approximated_boxes.append(f.name)
            for k, part in enumerate(_parts(f, plain_boxes)):
                name = f"collision/{f.name}/{k}"
                # Exact box mesh avoids the capsule/box endcap-only crossing case.
                if part["type"] == mujoco.mjtGeom.mjGEOM_BOX:
                    half = tuple(part.pop("size"))
                    if half not in box_meshes:
                        meshname = f"box/{len(box_meshes)}"
                        spec.add_mesh(
                            name=meshname,
                            uservert=np.array(
                                list(product(*[(-h, h) for h in half]))
                            ).ravel(),
                        )
                        box_meshes[half] = meshname
                    part["type"] = mujoco.mjtGeom.mjGEOM_MESH
                    part["meshname"] = box_meshes[half]
                colors = np.asarray(f.getMeshColors())
                color = (colors[0, :3] / 255.0 if colors.ndim == 2 and len(colors)
                         else np.asarray(info.get("color", [.65, .65, .7]))[:3])
                bodies[f.name].add_geom(
                    rgba=[*color, 1.0],
                    name=name,
                    contype=1,
                    conaffinity=1,
                    margin=0.0,
                    **part,
                )
                geoms_by_frame[f.name].append(name)
                geom_frames[name] = self.frame_index[f.name]
        initial_pairs = {
            frozenset(pair)
            for pair in np.asarray(C.getCollidablePairs()).reshape(-1, 2)
        }
        self.excluded_pairs = set()
        for pair in getattr(env, "collision_exclusions", ()):
            if len(pair) != 2 or not set(pair).issubset(geoms_by_frame):
                raise ValueError(f"Unknown collision shape in excluded pair: {pair}")
            a, b = pair
            self.excluded_pairs.add(tuple(sorted((self.frame_index[a], self.frame_index[b]))))
            initial_pairs.discard(frozenset(pair))
        self._initial_pairs = initial_pairs
        mutable_shapes = set()
        for name in geoms_by_frame:
            frame = C.getFrame(name)
            while frame is not None:
                if frame.name in self.movable:
                    mutable_shapes.add(name)
                    break
                frame = frame.getParent()
        for first, second in combinations(geoms_by_frame, 2):
            if (
                not ({first, second} & mutable_shapes)
                and frozenset((first, second)) not in initial_pairs
            ):
                spec.add_exclude(bodyname1=first, bodyname2=second)
        # Explicit pairs recover checks suppressed for welded/static bodies.
        probe = spec.compile()
        for first, second in combinations(geoms_by_frame, 2):
            a, b = probe.body(first).id, probe.body(second).id
            if (
                probe.body_weldid[a] == probe.body_weldid[b]
                and frozenset((first, second)) in initial_pairs
            ):
                for ga, gb in product(geoms_by_frame[first], geoms_by_frame[second]):
                    spec.add_pair(
                        geomname1=ga,
                        geomname2=gb,
                        condim=1,
                        margin=0.0,
                    )
        spec.worldbody.add_light(name="key", pos=[2, -2, 4], dir=[-1, 1, -2],
                                 castshadow=True)
        self.spec = spec
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        if len(joint_mapping) != len(self.joint_names):
            raise NotImplementedError("Not all active RAI coordinates were exported")
        self.qadr = np.array(
            [
                int(self.model.joint(joint_mapping[i]).qposadr[0])
                for i in range(len(self.joint_names))
            ]
        )
        self.body_ids = np.array(
            [self.model.body(name).id for name in self.frame_names]
        )
        self.geom_frames = np.array(
            [geom_frames[self.model.geom(g).name] for g in range(self.model.ngeom)]
        )
        self.object_qadr = {
            name: int(self.model.joint("object/" + name).qposadr[0])
            for name in self.movable
        }
        self.report = dict(
            frames=len(frames),
            geoms=self.model.ngeom,
            coordinates=len(self.joint_names),
            movable_roots=sorted(self.movable),
            explicit_pairs=self.model.npair,
            plain_box_frames=approximated_boxes,
        )

    def _pair_mask(self, parents, joints, contacts):
        """RAI's rigid-link/contact-depth exclusions, evaluated on plain arrays."""
        key = (parents.tobytes(), joints.tobytes(), contacts.tobytes())
        cached = self._pair_mask_cache.get(key)
        if cached is not None:
            self._pair_mask_cache.move_to_end(key)
            return cached

        def link(i):
            while parents[i] >= 0 and not joints[i]:
                i = parents[i]
            return i

        def child_of(a, b, depth):
            p = parents[a]
            while p >= 0:
                depth -= int(joints[p])
                if depth < 0:
                    return False
                if p == b:
                    return True
                p = parents[p]
            return False

        n = len(self.frame_names)
        allowed = np.zeros((n, n), dtype=bool)
        for i, j in combinations(np.unique(self.geom_frames), 2):
            if self.excluded_pairs and (int(i), int(j)) in self.excluded_pairs:
                continue
            if not contacts[i] or not contacts[j]:
                continue
            a, b = link(i), link(j)
            if a == b:
                continue
            if contacts[i] < 0 and child_of(a, b, -contacts[i]):
                continue
            if contacts[j] < 0 and child_of(b, a, -contacts[j]):
                continue
            allowed[i, j] = allowed[j, i] = True
        allowed.flags.writeable = False
        self._pair_mask_cache[key] = allowed
        if len(self._pair_mask_cache) > 128:
            self._pair_mask_cache.popitem(last=False)
        return allowed

    def set_state(self, state):
        """Install attachment/exclusion metadata without rebuilding the model."""
        if state is self._active_state:
            return
        parents = np.asarray(state["parents"], dtype=int)
        joints = np.asarray(state["joints"], dtype=bool)
        contacts = np.asarray(state["contacts"], dtype=int)
        self.allowed = self._pair_mask(parents, joints, contacts)
        self.owners = np.full(len(self.frame_names), -1, dtype=int)
        for i, name in enumerate(self.frame_names):
            ancestor = i
            while ancestor >= 0:
                robot = self.frame_robot.get(self.frame_names[ancestor])
                if robot is not None:
                    self.owners[i] = robot
                    break
                ancestor = parents[ancestor]
        self.object_levels = []
        remaining = set(state["attachments"])
        resolved = set()
        while remaining:
            level = []
            for name in sorted(remaining):
                record = state["attachments"][name]
                parent = record["parent"]
                ancestor = -1 if parent is None else self.frame_index[parent]
                blocked = False
                visited = {self.frame_index[name]}
                while ancestor >= 0:
                    if ancestor in visited:
                        raise ValueError("Cyclic attachment hierarchy")
                    visited.add(ancestor)
                    if self.frame_names[ancestor] in remaining:
                        blocked = True
                        break
                    ancestor = parents[ancestor]
                if blocked:
                    continue
                pose = np.asarray(record["pose"])
                parent_id = 0 if parent is None else int(self.model.body(parent).id)
                # World-fixed objects need no extra FK pass.
                ancestor = -1 if parent is None else self.frame_index[parent]
                moving = False
                while ancestor >= 0:
                    if (self.frame_names[ancestor] in self.movable
                            or self.frame_names[ancestor] in self.joint_names
                            or any(j.startswith(self.frame_names[ancestor] + ":")
                                   for j in self.joint_names)):
                        moving = True
                        break
                    ancestor = parents[ancestor]
                if moving:
                    level.append((self.object_qadr[name], parent_id, pose[:3], pose[3:]))
                else:
                    # Static parent pose is already available after compilation/FK.
                    adr = self.object_qadr[name]
                    R = self.data.xmat[parent_id].reshape(3, 3)
                    self.data.qpos[adr:adr+3] = self.data.xpos[parent_id] + R @ pose[:3]
                    mujoco.mju_mulQuat(self.data.qpos[adr+3:adr+7],
                                      self.data.xquat[parent_id], pose[3:])
                resolved.add(name)
            if not resolved.intersection(remaining):
                raise ValueError("Cyclic attachment hierarchy")
            remaining -= resolved
            if level:
                self.object_levels.append(level)
        self._active_state = state
        self.stats["mode_updates"] += 1

    def _transition(self, previous, mode):
        changed = {a for a, b in zip(mode.prev_mode.task_ids, mode.task_ids) if a != b}
        if len(changed) != 1:
            raise NotImplementedError("Expected one task per mode transition")
        action = self.actions[changed.pop()]
        state = copy.deepcopy(previous)
        q = np.asarray(state["q"], dtype=float)
        idx = action["indices"]
        q[idx] = mode.entry_configuration.state()[idx]
        state["q"] = q.tolist()
        if action["type"] in (None, "goto"):
            if action["side_effect"] is not None:
                raise NotImplementedError("Side effects on non-manipulation tasks")
            return state
        parent, obj = action["frames"][:2]
        ancestor = self.frame_index[parent]
        while ancestor >= 0:
            if ancestor == self.frame_index[obj]:
                raise ValueError("Task would create a cyclic attachment hierarchy")
            ancestor = state["parents"][ancestor]
        if obj not in state["attachments"]:
            raise NotImplementedError(f"Unknown manipulated root {obj}")
        if action["side_effect"] == "make_appear":
            state["attachments"][obj]["pose"] = action["side_effect_data"]
            state["contacts"][self.frame_index[obj]] = 1
        elif action["side_effect"] not in (None, "remove"):
            raise NotImplementedError(f"Unknown side effect {action['side_effect']}")
        self.set_state(state)
        self.update(q, collision=False)
        child_id = self.model.body(obj).id
        parent_id = self.model.body(parent).id
        relative_pos = self.data.xmat[parent_id].reshape(3, 3).T @ (
            self.data.xpos[child_id] - self.data.xpos[parent_id])
        inverse = self.data.xquat[parent_id].copy()
        inverse[1:] *= -1
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, inverse, self.data.xquat[child_id])
        state["attachments"][obj] = dict(parent=parent, pose=np.r_[relative_pos, quat].tolist())
        fi = self.frame_index[obj]
        state["parents"][fi] = self.frame_index[parent]
        state["joints"][fi] = True  # RAI attach creates a rigid joint boundary.
        if state["contacts"][fi]:
            state["contacts"][fi] = -1
        if action["side_effect"] == "remove":
            for i in range(len(self.frame_names)):
                ancestor = i
                while ancestor >= 0:
                    if ancestor == fi:
                        state["contacts"][i] = 0
                        break
                    ancestor = state["parents"][ancestor]
        self._active_state = None  # state changed after its provisional FK update
        return state

    def set_mode(self, mode):
        if mode is self._mode:
            return
        missing = []
        ancestor = mode
        while ancestor is not None and id(ancestor) not in self._mode_cache:
            missing.append(ancestor)
            ancestor = ancestor.prev_mode
        state = self.initial_state if ancestor is None else self._mode_cache[id(ancestor)][1]
        for node in reversed(missing):
            if node.prev_mode is not None:
                state = self._transition(state, node)
            self._mode_cache[id(node)] = (node, state)
            # Keep attachment records bounded; never cache a compiled model per mode.
            if len(self._mode_cache) > 128:
                self._mode_cache.popitem(last=False)
        self.set_state(state)
        self._mode = mode

    def update(self, q, collision=True):
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        for level in self.object_levels:
            for adr, parent, pos, quat in level:
                R = self.data.xmat[parent].reshape(3, 3)
                self.data.qpos[adr:adr+3] = self.data.xpos[parent] + R @ pos
                mujoco.mju_mulQuat(self.data.qpos[adr+3:adr+7], self.data.xquat[parent], quat)
            mujoco.mj_kinematics(self.model, self.data)
        if collision:
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_collision(self.model, self.data)

    def valid(self, q=None, mode=None, tolerance=None, set_mode=True, robots=None):
        tolerance = self.collision_tolerance if tolerance is None else float(tolerance)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Penetration tolerance must be finite and nonnegative")
        if q is None:
            q = self.data.qpos[self.qadr].copy()
        else:
            q = np.asarray(q.state() if hasattr(q, "state") else q, dtype=float)
        if q.shape != (len(self.joint_names),) or not np.isfinite(q).all():
            return False
        if set_mode:
            self.set_mode(mode)
        self.update(q)
        self.stats["queries"] += 1
        return self.contacts_valid(tolerance, robots)

    def contacts_valid(self, tolerance=None, robots=None):
        tolerance = self.collision_tolerance if tolerance is None else tolerance
        contacts = self.data.contact
        negative = contacts.dist < 0
        if not np.any(negative):
            return True
        pairs = self.geom_frames[contacts.geom[negative]]
        depth = -contacts.dist[negative]
        relevant = self.allowed[pairs[:, 0], pairs[:, 1]]
        if robots is not None:
            names = [robots] if isinstance(robots, str) else list(robots)
            selected = [self.robot_names.index(name) for name in names]
            owners = self.owners[pairs]
            involved = np.isin(owners, selected)
            relevant &= involved.any(axis=1) & (involved | (owners < 0)).all(axis=1)
        if not np.any(relevant):
            return True
        depth, pairs = depth[relevant], pairs[relevant]
        if depth.max() > tolerance:
            return False
        # Count a shape pair once, rather than summing its manifold points.
        keys = np.minimum(pairs[:, 0], pairs[:, 1]) * len(self.frame_names) + np.maximum(pairs[:, 0], pairs[:, 1])
        _, inverse = np.unique(keys, return_inverse=True)
        penetration = np.zeros(inverse.max() + 1)
        np.maximum.at(penetration, inverse, depth)
        return bool(penetration.sum() <= tolerance)

    def save(self, path):
        """Write standalone MJCF plus attachment metadata; neither needs RAI."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The keyframe makes the exported initial pose available in native viewers.
        spec = self.spec.copy()
        key = spec.key("exported")
        if key is None:
            spec.add_key(name="exported", qpos=self.data.qpos)
        else:
            key.qpos = self.data.qpos
        spec.compile()
        path.write_text(spec.to_xml())
        fields = ["joint_names", "frame_names", "frame_index", "object_qadr", "report",
                  "collision_tolerance", "actions", "initial_state", "robot_names", "frame_robot",
                  "coordinate_frames"]
        metadata = {key: getattr(self, key) for key in fields}
        metadata.update(qadr=self.qadr.tolist(), body_ids=self.body_ids.tolist(),
                        geom_frames=self.geom_frames.tolist(), movable=sorted(self.movable),
                        start_q=self.data.qpos[self.qadr].tolist(), state=self._active_state)
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        return path

    @classmethod
    def load(cls, path):
        """Load an exported collision scene without importing or constructing RAI."""
        path = Path(path)
        obj = cls.__new__(cls)
        metadata = json.loads(path.with_suffix(".json").read_text())
        state = metadata.pop("state")
        for key, value in metadata.items():
            setattr(obj, key, value)
        for key in ["qadr", "body_ids", "geom_frames", "start_q"]:
            setattr(obj, key, np.asarray(getattr(obj, key)))
        obj.movable = set(obj.movable)
        obj.spec = mujoco.MjSpec.from_file(str(path))
        obj.model = obj.spec.compile()
        obj.data = mujoco.MjData(obj.model)
        mujoco.mj_resetDataKeyframe(obj.model, obj.data, obj.model.key("exported").id)
        mujoco.mj_kinematics(obj.model, obj.data)
        obj.stats = dict(queries=0, mode_updates=0, compilations=1)
        obj._active_state = obj._mode = None
        obj._mode_cache = OrderedDict()
        obj._pair_mask_cache = OrderedDict()
        obj.set_state(state)
        obj.update(obj.start_q)
        return obj
