"""RoboBallet welding scenes and cuArms-inspired Panda packing experiments.

Shared Panda assets come from the separately supplied RoboBallet release.
Packing layouts are recreations, not released cuArms instances. Prepared JSON
instances freeze geometry, goals and assignments. No RL dependencies are needed.
The objective is geometric joint-space length, not execution time with dwell.
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .planning_env import (
    BaseProblem,
    BaseModeLogic,
    DependencyGraphMixin,
    SequenceMixin,
    Task,
    SingleGoal,
    ProblemSpec,
    AgentType,
    GoalType,
    ConstraintType,
    DynamicsType,
    ManipulationType,
    DependencyType,
    SafePoseType,
)
from .core.configuration import NpConfiguration
from .core.registry import register
from .core.dependency_graph import DependencyGraph
from .core.goals import GoalSet
from .mujoco_env import MujocoEnvironment


from .roboballet_models import HOME, make_welding_model, make_packing_model, quaternion
from .roboballet_instances import load_instance, resolve_assets


class _RoboBalletEnvironment(MujocoEnvironment):
    """Static welding or pick/place tasks with fixed grasp frames.

    ``assets`` is the external release directory. Without ``instance``, scene
    options (kind, robots, targets, episode, objects, seed) construct goals via IK.
    A prepared JSON path or mapping bypasses preparation and preserves its goals.
    Source meshes are the default. Mocap objects use this class's pose hooks;
    the shared checker also supports the existing free-joint environments.
    """

    def __init__(self, assets, instance=None, geometry="source", sequence_seed=0, **scene_options):
        if instance is None:
            from .roboballet_setup import make_instance
            instance = make_instance(assets, geometry=geometry, **scene_options)
        elif scene_options:
            raise ValueError("Pass either an instance or scene construction options")
        instance = load_instance(instance)
        self.instance = instance
        self.assets = resolve_assets(assets, instance)
        if geometry not in ("source", "primitives", "panda_capsules", "panda_spheres"):
            raise ValueError(f"Unknown collision geometry: {geometry!r}")
        self.geometry = geometry
        self.robots = [f"panda{i+1}" for i in range(instance["robots"])]
        self.packing = instance["kind"] in ("pack", "unpack")
        self.model = (
            make_packing_model(self.assets, instance, geometry)
            if self.packing
            else make_welding_model(
                self.assets,
                len(self.robots),
                instance.get("boxes", []),
                instance.get("targets", []),
                geometry,
                instance.get("hulls"),
                instance.get("sphere_model"),
                instance.get("workpiece_geometry", "auto"),
            )
        )
        self.data = mujoco.MjData(self.model)
        if self.packing:
            for r in self.robots:
                for j in [1, 2]:
                    self.data.qpos[self.model.joint(f"{r}/finger_joint{j}").qposadr] = (
                        0.035
                    )
        self.robot_idx = {
            r: np.arange(7 * i, 7 * (i + 1)) for i, r in enumerate(self.robots)
        }
        self.robot_dims = {r: 7 for r in self.robots}
        self.robot_joints = {
            r: [f"{r}/joint{j+1}" for j in range(7)] for r in self.robots
        }
        joints = [
            self.model.joint(j).id for r in self.robots for j in self.robot_joints[r]
        ]
        self.qadr = self.model.jnt_qposadr[joints]
        self._all_robot_idx = self.qadr
        self.dofadr = self.model.jnt_dofadr[joints]
        self.limits = self.model.jnt_range[joints].T.copy()
        self.start_pos = NpConfiguration.from_list([HOME.copy() for r in self.robots])
        self.safe_pose = {r: HOME.copy() for r in self.robots}
        self.constraints = []
        self.collision_resolution = float(instance.get("collision_resolution", 0.05))
        self.collision_tolerance = 0.0
        self.manipulating_env = self.packing
        self.viewer = None
        suffix = "grasp_tcp" if self.packing else "tool/tool_tip_site"
        self.tcp = [self.model.site(f"{r}/{suffix}").id for r in self.robots]
        self.object_mocap = {
            f"block_{i}": int(self.model.body(f"block_{i}").mocapid[0])
            for i in range(len(instance.get("objects", [])))
        }
        self.geom_object = {
            self.model.geom(f"block_geom_{i}").id: f"block_{i}"
            for i in range(len(instance.get("objects", [])))
        }
        self.initial_objects = {
            f"block_{i}": ("world", tuple(obj["start"]), (1.0, 0.0, 0.0, 0.0))
            for i, obj in enumerate(instance.get("objects", []))
        }
        self.active_objects = self.initial_objects
        self.geom_robot = np.full(self.model.ngeom, -1, dtype=int)
        for g in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[g])
            while body:
                name = self.model.body(body).name
                if name in self.robots:
                    self.geom_robot[g] = self.robots.index(name)
                    break
                body = int(self.model.body_parentid[body])
        self._geom_is_object = np.zeros(self.model.ngeom, dtype=bool)
        self._geom_is_object[list(self.geom_object)] = True
        self._grasp_geom_robot = np.full(self.model.ngeom, -1, dtype=int)
        for r, name in enumerate(self.robots):
            for part in ("hand", "left_finger", "right_finger"):
                body = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"{name}/{part}"
                )
                if body >= 0:
                    self._grasp_geom_robot[self.model.geom_bodyid == body] = r
        self._robot_number = {name: i for i, name in enumerate(self.robots)}
        self.tasks = []
        for task in instance.get("tasks", []):
            r = self.robots[task["robot"]]
            goal = (
                GoalSet([np.asarray(q, dtype=float) for q in task["q_candidates"]])
                if "q_candidates" in task
                else SingleGoal(np.asarray(task["q"], dtype=float))
            )
            kwargs = {}
            if "action" in task:
                parent = (self.model.site(self.tcp[task["robot"]]).name
                          if task["action"] == "pick" else "world")
                kwargs = dict(type=task["action"], frames=[parent, f"block_{task['object']}"])
            self.tasks.append(Task(task["name"], [r], goal, **kwargs))

        self.tasks.append(
            Task("terminal", self.robots, SingleGoal(self.start_pos.state().copy()))
        )
        dependency = self._initialize_task_order(sequence_seed)
        self.spec = ProblemSpec(
            agent_type=AgentType.MULTI_AGENT,
            constraints=ConstraintType.UNCONSTRAINED,
            manipulation=(
                ManipulationType.MANIPULATION
                if self.packing
                else ManipulationType.STATIC
            ),
            dependency=dependency,
            dynamics=DynamicsType.GEOMETRIC,
            goals=GoalType.MULTI_GOAL,
            home_pose=SafePoseType.HAS_SAFE_HOME_POSE,
        )
        BaseProblem.__init__(self)
        BaseModeLogic.__init__(self)
        self.set_configuration(self.start_pos.state())

    def get_scenegraph_info_for_mode(self, mode, is_start_mode=False):
        if not self.packing:
            return {}
        if is_start_mode or mode.prev_mode is None:
            return self.initial_objects.copy()
        sg = mode.prev_mode.sg.copy()
        changed = {a for a, b in zip(mode.prev_mode.task_ids, mode.task_ids) if a != b}
        self.set_configuration(
            mode.entry_configuration.state(), mode.prev_mode, collision=False
        )
        for task_id in changed:
            task = self.tasks[task_id]
            if task.type not in ("pick", "place"):
                continue
            action = task.type
            robot = self.robots.index(task.robots[0])
            key = task.frames[1]
            parent, pos, quat = sg[key]
            R = Rotation.from_quat((quat[1], quat[2], quat[3], quat[0])).as_matrix()
            tcp_p = self.data.site_xpos[self.tcp[robot]].copy()
            tcp_R = self.data.site_xmat[self.tcp[robot]].reshape(3, 3).copy()
            if action == "pick":
                if parent != "world":
                    raise ValueError("Cannot pick an attached object")
                sg[key] = (
                    self.robots[robot],
                    tuple(tcp_R.T @ (np.asarray(pos) - tcp_p)),
                    tuple(quaternion(tcp_R.T @ R)),
                )
            else:
                if parent != self.robots[robot]:
                    raise ValueError("Placing an object not held by this robot")
                sg[key] = (
                    "world",
                    tuple(tcp_p + tcp_R @ pos),
                    tuple(quaternion(tcp_R @ R)),
                )
        return sg

    def get_conversion_metadata(self):
        attachments = {}
        for name, (parent, position, quaternion) in self.start_mode.sg.items():
            if parent in self.robots:
                parent = self.model.site(self.tcp[self.robots.index(parent)]).name
            attachments[name] = (parent, position, quaternion)
        exclusions = {
            self.model.site(self.tcp[i]).name:
                tuple(map(int, np.flatnonzero(self._grasp_geom_robot == i)))
            for i in range(len(self.robots))
        }
        return dict(attachments=attachments, attachment_exclusions=exclusions,
                    tolerance_mode="penetration", check_joint_limits=True)

    def set_to_mode(self, mode):
        self.prev_mode = mode
        self.set_configuration(self.data.qpos[self.qadr].copy(), mode)

    def set_configuration(self, x, mode=None, collision=True):
        self.data.qpos[self.qadr] = x
        attached = []
        if self.packing:
            self.active_objects = self.initial_objects if mode is None else mode.sg
            for name, (parent, pos, quat) in self.active_objects.items():
                mid = self.object_mocap[name]
                if parent == "world":
                    self.data.mocap_pos[mid], self.data.mocap_quat[mid] = pos, quat
                else:
                    attached.append((mid, parent, pos, quat))
        mujoco.mj_kinematics(self.model, self.data)
        for mid, parent, pos, quat in attached:
            r = self.robots.index(parent)
            tcp_R = self.data.site_xmat[self.tcp[r]].reshape(3, 3)
            R = Rotation.from_quat((quat[1], quat[2], quat[3], quat[0])).as_matrix()
            self.data.mocap_pos[mid] = self.data.site_xpos[self.tcp[r]] + tcp_R @ pos
            self.data.mocap_quat[mid] = quaternion(tcp_R @ R)
        if attached:
            mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        if collision:
            mujoco.mj_collision(self.model, self.data)

    def colliding_pairs(self, robot=None, tolerance=None):
        threshold = self.collision_tolerance if tolerance is None else -abs(tolerance)
        pairs = []
        for c in self.data.contact[: self.data.ncon]:
            a, b = map(int, c.geom)
            ra, rb = self.geom_robot[[a, b]]
            oa, ob = self.geom_object.get(a), self.geom_object.get(b)
            if ra < 0 and rb < 0 and oa is None and ob is None:
                continue
            # Only the grasping hand may contact its attached block. Other arm
            # links, other blocks and the table remain collision checked.
            allowed = False
            for obj, other in [(oa, b), (ob, a)]:
                if obj is not None:
                    parent = self.active_objects[obj][0]
                    body_name = self.model.body(self.model.geom_bodyid[other]).name
                    if parent != "world" and body_name in [
                        f"{parent}/hand",
                        f"{parent}/left_finger",
                        f"{parent}/right_finger",
                    ]:
                        allowed = True
            if allowed:
                continue
            owners = {ra, rb}
            for obj in (oa, ob):
                if obj is not None and self.active_objects[obj][0] in self.robots:
                    owners.add(self.robots.index(self.active_objects[obj][0]))
            if robot is not None and robot not in owners:
                continue
            if c.dist < threshold:
                pairs.append((a, b, float(c.dist)))
        return pairs

    def _configuration_is_valid(self, q):
        # Inclusive finite joint bounds also reject NaN/inf without a second pass.
        return q.shape == self.limits[0].shape and bool(np.all(
            (q >= self.limits[0] - 1e-10) & (q <= self.limits[1] + 1e-10)
        ))

    def _prepare_collision_state(self, q, mode):
        """Position mocap attachments using the robots' current TCP poses."""
        self.set_configuration(q, mode, collision=False)

    def _update_collision_state(self, q, mode):
        self.set_configuration(q, mode)

    def _contact_distance_threshold(self, tolerance):
        return -abs(tolerance)

    def _relevant_contact_pairs(self, pairs):
        a, b = pairs.T
        relevant = (
            (self.geom_robot[a] >= 0)
            | (self.geom_robot[b] >= 0)
            | self._geom_is_object[a]
            | self._geom_is_object[b]
        )
        for geom, obj in self.geom_object.items():
            owner = self._robot_number.get(self.active_objects[obj][0])
            if owner is not None:
                relevant &= ~(
                    ((a == geom) & (self._grasp_geom_robot[b] == owner))
                    | ((b == geom) & (self._grasp_geom_robot[a] == owner))
                )
        return relevant

    def is_collision_free_for_robot(
        self, r, q, m=None, collision_tolerance=None, **kwargs
    ):
        values = q.state() if hasattr(q, "state") else np.asarray(q)
        if not super()._configuration_is_valid(values):
            return False
        names = [r] if isinstance(r, str) else list(r)
        indices = (
            np.concatenate([self.robot_idx[name] for name in names])
            if names
            else np.array([], dtype=int)
        )
        if np.any(values[indices] < self.limits[0, indices] - 1e-10) or np.any(
            values[indices] > self.limits[1, indices] + 1e-10
        ):
            return False
        self.set_configuration(values, m)
        return all(
            not self.colliding_pairs(self.robots.index(name), collision_tolerance)
            for name in names
        )

    def show_config(self, q, blocking=True):
        self.set_configuration(q.state() if hasattr(q, "state") else np.asarray(q),
                               self.prev_mode)
        self.show(blocking)

    def object_poses(self):
        return {
            name: (self.data.mocap_pos[mid].copy(), self.data.mocap_quat[mid].copy())
            for name, mid in self.object_mocap.items()
        }

    def render(self, filename, q=None, width=1400, height=1000, mode=None):
        from PIL import Image

        self.set_configuration(self.start_pos.state() if q is None else q, mode)
        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height
        self.model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
        self.model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, 0.85 if len(self.robots) == 8 else 0.35]
        camera.distance = 3.2 if len(self.robots) == 8 else 2.5
        camera.azimuth, camera.elevation = 135, -28
        if self.packing:
            camera.lookat[:] = [0, 0, 0.25]
            camera.distance, camera.azimuth, camera.elevation = 2.1, 90, -35
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        option.sitegroup[:] = 0
        option.sitegroup[2] = 1
        with mujoco.Renderer(self.model, height=height, width=width) as renderer:
            renderer.update_scene(self.data, camera=camera, scene_option=option)
            Image.fromarray(renderer.render()).save(filename)


class RoboBalletEnvironment(DependencyGraphMixin, _RoboBalletEnvironment):
    """Fixed per-robot task orders with freely chosen interleavings."""

    def __init__(self, assets, instance=None, geometry="source", **scene_options):
        _RoboBalletEnvironment.__init__(self, assets, instance, geometry, **scene_options)

    def _initialize_task_order(self, sequence_seed):
        self.graph = DependencyGraph()
        last = {}
        for task in self.instance.get("tasks", []):
            self.graph.add_node(task["name"])
            robot = task["robot"]
            if robot in last:
                self.graph.add_dependency(task["name"], last[robot])
            last[robot] = task["name"]
        self.graph.add_node("terminal")
        for name in last.values():
            self.graph.add_dependency("terminal", name)
        return DependencyType.UNORDERED


class SequentialRoboBalletEnvironment(SequenceMixin, _RoboBalletEnvironment):
    """Seeded random interleaving, fixed for the entire planning problem.

    Choose uniformly among robots with remaining tasks, taking each robot's
    next task. This preserves pick/place order; terminal home is always last.
    The local generator does not consume the planner's random state.
    """

    def __init__(self, assets, instance=None, geometry="source", sequence_seed=0, **scene_options):
        _RoboBalletEnvironment.__init__(self, assets, instance, geometry, sequence_seed, **scene_options)

    def _initialize_task_order(self, sequence_seed):
        rng = np.random.default_rng(sequence_seed)
        queues = [[i for i, task in enumerate(self.instance.get("tasks", []))
                   if task["robot"] == robot] for robot in range(len(self.robots))]
        queues = [queue for queue in queues if queue]
        self.sequence = []
        while queues:
            index = int(rng.integers(len(queues)))
            self.sequence.append(queues[index].pop(0))
            if not queues[index]:
                queues.pop(index)
        self.sequence.append(len(self.tasks) - 1)
        return DependencyType.FULLY_ORDERED


@register([
    ("roboballet.welding", {"kind": "welding"}),
    ("cuarms.pack", {"kind": "pack"}),
    ("cuarms.unpack", {"kind": "unpack"}),
    ("roboballet.welding_sequence", {"kind": "welding", "task_order": "random-sequence"}),
    ("cuarms.pack_sequence", {"kind": "pack", "task_order": "random-sequence"}),
    ("cuarms.unpack_sequence", {"kind": "unpack", "task_order": "random-sequence"}),
])
def make_roboballet_environment(assets, instance=None, geometry="source", *,
                               kind=None, task_order="dependency", sequence_seed=0,
                               **scene_options):
    """Build a named preset or a prepared instance with explicit external assets.

    Welding defaults to eight robots and 40 targets; pack/unpack to four blocks.
    Scene options override construction defaults. A prepared instance replaces
    construction options but must match the named preset's kind.
    """
    if kind is not None:
        if instance is None:
            scene_options["kind"] = kind
        else:
            instance = load_instance(instance)
            if instance["kind"] != kind:
                raise ValueError(f"Preset {kind!r} does not match instance kind {instance['kind']!r}")
    if task_order == "dependency":
        return RoboBalletEnvironment(assets, instance, geometry, **scene_options)
    if task_order == "random-sequence":
        return SequentialRoboBalletEnvironment(assets, instance, geometry, sequence_seed, **scene_options)
    raise ValueError("task_order must be dependency or random-sequence")
