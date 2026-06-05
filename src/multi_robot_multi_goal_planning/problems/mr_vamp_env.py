import copy
import numpy as np
import time

from typing import List, Optional
from numpy.typing import NDArray

import matplotlib.pyplot as plt

from .planning_env import (
    generate_binary_search_indices,
)

from .core.configuration import (
    Configuration,
    NpConfiguration,
    config_dist,
    config_cost,
    batch_config_cost,
)

from .planning_env import (
    BaseModeLogic,
    SequenceMixin,
    DependencyGraphMixin,
    State,
    Mode,
    Task,
    BaseProblem,
    ProblemSpec,
    AgentType,
    GoalType,
    ConstraintType,
    DynamicsType,
    ManipulationType,
    DependencyType,
    SafePoseType,
)
from .core.goals import (
    SingleGoal,
    GoalSet,
    GoalRegion,
    ConditionalGoal,
)

import sys
sys.path.append("/usr/local/lib/python3.10/dist-packages")  # TODO: install mr_planner_core into venv
import mr_planner_core
sys.path.pop()

from .core.registry import register

import meshcat
import copy

class VampEnv(BaseProblem):
    """
    Simple environment, only supporting rectangle and sphere obstacles, and spherical agents.
    """

    def __init__(self):
        super().__init__()

        self.limits = None
        self.start_pos = None

        self.env = None

        self.cost_metric = "euclidean"
        self.cost_reduction = "max"

        self.manipulating_env = False

        # sg format: {obj_name: ("attached", robot_id, joints_list) |
        #                       ("world",    robot_id_was, joints_at_detach)}
        self.initial_sg: dict = {}
        self._current_sg: dict = {}
        # Stores mr_planner_core.Object instances so objects can be reset to their
        # initial world pose (e.g. when scrubbing backward across a pick).
        self._initial_objects: dict = {}

        # Cache world poses for each mode to support direct mode jumps and avoid
        # stale pose propagation during world->world transitions.
        self._mode_object_poses: dict = {}

        self.current_mode = None

        self.spec = ProblemSpec(
            agent_type=AgentType.MULTI_AGENT,
            constraints=ConstraintType.UNCONSTRAINED,
            manipulation=ManipulationType.MANIPULATION,
            dependency=DependencyType.FULLY_ORDERED,
            dynamics=DynamicsType.GEOMETRIC,
            goals=GoalType.MULTI_GOAL,
            home_pose=SafePoseType.HAS_NO_SAFE_HOME_POSE,
        )

    def _set_zero_safe_pose(self, dim: int):
        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE
        self.safe_pose = {r: np.zeros(dim) for r in self.robots}

    def __deepcopy__(self, memo):
        # Save the C attribute
        env = self.env

        # Temporarily remove C to allow deepcopy of other attributes
        self.env = None
        
        # Create a deep copy of self without C
        new_env = copy.deepcopy(super(), memo)

        # Restore C in both objects
        self.env = env
        new_env.env = env

        return new_env

    def display_path(
        self,
        path: List[State],
        stop: bool = True,
        export: bool = False,
        pause_time: float = 0.01,
        stop_at_end=False,
        adapt_to_max_distance: bool = False,
        stop_at_mode: bool = False,
    ) -> None:
        self.env.enable_meshcat()

        for i in range(len(path)):
            self.set_to_mode(path[i].mode)
            self.show_config(path[i].q)

            time.sleep(0.1)
        
        # self.env.enable_meshcat()
        # num_robots = path[0].q.num_agents()

        # path_as_list = []
        # for i in range(num_robots):
        #     per_agent_path = []
        #     for p in path:
        #         per_agent_path.append(p.q[i])

        #     path_as_list.append(per_agent_path)

        # # self.env.play_trajectory(path_as_list)

    def _viser_build_scene(self, server) -> None:
        """Add static environment objects (boxes, spheres, cylinders) to a viser server."""
        objects = self.env.get_scene_objects()
        _ENV_COLOR = (180, 180, 180)

        for obj in objects:
            name = obj["name"]
            pos = obj["position"]        # [x, y, z]
            q = obj["quaternion"]        # [qx, qy, qz, qw] (xyzw)
            wxyz = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
            position = (float(pos[0]), float(pos[1]), float(pos[2]))
            shape = obj.get("type", obj.get("shape", ""))

            if shape == "box":
                sz = obj["size"]
                server.scene.add_box(
                    name=f"env/{name}",
                    color=_ENV_COLOR,
                    dimensions=(float(sz[0]), float(sz[1]), float(sz[2])),
                    wxyz=wxyz,
                    position=position,
                )
            elif shape == "sphere":
                server.scene.add_icosphere(
                    name=f"env/{name}",
                    radius=float(obj["radius"]),
                    color=_ENV_COLOR,
                    wxyz=wxyz,
                    position=position,
                )
            elif shape == "cylinder":
                server.scene.add_cylinder(
                    name=f"env/{name}",
                    radius=float(obj["radius"]),
                    height=float(obj["length"]),
                    color=_ENV_COLOR,
                    wxyz=wxyz,
                    position=position,
                )

    def display_path_viser(
        self,
        paths: "List[List[State]] | List[State]",
        pause_time: float = 0.05,
        port: int = 8080,
        path_labels: Optional[List[str]] = None,
        primitives_only: bool = False,
        step_annotations: "Optional[List[str]]" = None,
    ) -> None:
        """Display one or more planned paths interactively in viser.

        Renders the robot sphere decomposition using get_sphere_poses() and
        static environment objects using get_scene_objects().

        Args:
            paths: A single path (List[State]) or a list of paths to compare.
            pause_time: Initial time between frames during playback (seconds).
            port: Port for the viser HTTP server.
            path_labels: Optional display names for each path.
            primitives_only: Unused (kept for API parity with rai version).
            step_annotations: Optional per-step markdown strings appended to the
                mode label panel (one entry per step in the single/first path).
        """
        if paths and not isinstance(paths[0], list):
            paths = [paths]
        try:
            import viser
        except ImportError:
            raise ImportError("viser is required. Install with: pip install viser")

        _ROBOT_COLOR = (100, 160, 220)

        server = viser.ViserServer(port=port)
        server.scene.set_up_direction("+z")
        server.scene.world_axes.visible = False

        first_state = paths[0][0]
        # Apply initial mode scenegraph so attached objects start in the right place
        self.set_to_mode(first_state.mode)
        self.env.set_joint_positions(first_state.q.as_list())

        # Add static environment objects
        self._viser_build_scene(server)

        # Find maximum sphere count across all states (attached spheres at the end)
        max_spheres = 0
        for path in paths:
            for state in path:
                self.set_to_mode(state.mode)
                n = len(self.env.get_sphere_poses(state.q.as_list()))
                max_spheres = max(max_spheres, n)

        # Create handles for all possible spheres
        sphere_handles = []
        for i in range(max_spheres):
            handle = server.scene.add_icosphere(
                name=f"spheres/sphere_{i}",
                radius=0.1,  # Default radius, will be updated
                color=_ROBOT_COLOR,
                position=(0.0, 0.0, 0.0),
                subdivisions=2,
                visible=False,
            )
            sphere_handles.append(handle)

        # Initialize with first state
        self.set_to_mode(first_state.mode)
        self.env.set_joint_positions(first_state.q.as_list())
        spheres = self.env.get_sphere_poses(first_state.q.as_list())

        # Set initial visibility and properties (attached spheres are already at end)
        for i, (cx, cy, cz, r) in enumerate(spheres):
            sphere_handles[i].radius = float(r)
            sphere_handles[i].position = np.array([cx, cy, cz], dtype=np.float32)
            sphere_handles[i].visible = True

        current_visible_count = len(spheres)

        # Pre-create handles for manipulable objects so their poses can be updated
        _OBJ_COLOR = (220, 160, 80)
        object_handles: dict = {}
        if self.manipulating_env and self.initial_sg:
            scene_objs = {obj["name"]: obj for obj in self.env.get_scene_objects()}
            for obj_name in self.initial_sg:
                if obj_name not in scene_objs:
                    continue
                obj = scene_objs[obj_name]
                pos = obj["position"]
                q_o = obj["quaternion"]
                shape = obj.get("type", obj.get("shape", ""))
                position = (float(pos[0]), float(pos[1]), float(pos[2]))
                wxyz = (float(q_o[3]), float(q_o[0]), float(q_o[1]), float(q_o[2]))
                if shape == "box":
                    sz = obj["size"]
                    h = server.scene.add_box(
                        name=f"env/{obj_name}",
                        color=_OBJ_COLOR,
                        dimensions=(float(sz[0]), float(sz[1]), float(sz[2])),
                        wxyz=wxyz,
                        position=position,
                    )
                elif shape == "sphere":
                    h = server.scene.add_icosphere(
                        name=f"env/{obj_name}",
                        radius=float(obj["radius"]),
                        color=_OBJ_COLOR,
                        wxyz=wxyz,
                        position=position,
                    )
                elif shape == "cylinder":
                    h = server.scene.add_cylinder(
                        name=f"env/{obj_name}",
                        radius=float(obj["radius"]),
                        height=float(obj["length"]),
                        color=_OBJ_COLOR,
                        wxyz=wxyz,
                        position=position,
                    )
                else:
                    continue
                object_handles[obj_name] = h

        # Track current number of visible spheres to avoid unnecessary visibility changes
        current_visible_count = len(spheres)

        def _set_step(step: int, path: "List[State]", lbl) -> None:
            nonlocal current_visible_count
            state = path[min(step, len(path) - 1)]
            # Apply mode scenegraph (attach/detach objects) then set joint positions
            self.set_to_mode(state.mode)
            self.env.set_joint_positions(state.q.as_list())
            
            # Get current spheres (robot spheres first, attached spheres at end)
            spheres = self.env.get_sphere_poses(state.q.as_list())
            total_spheres = len(spheres)
            
            def _set_visibility(count: int) -> None:
                nonlocal current_visible_count
                if count == current_visible_count:
                    return
                for i in range(max_spheres):
                    sphere_handles[i].visible = (i < count)
                current_visible_count = count

            # Only update visibility when count changes
            _set_visibility(total_spheres)

            # Update positions/radii for all spheres that should be visible
            for i, (cx, cy, cz, r) in enumerate(spheres):
                sphere_handles[i].position = np.array([cx, cy, cz], dtype=np.float32)
                sphere_handles[i].radius = float(r)
            
            # Update manipulable object poses
            if object_handles:
                scene_objs = {obj["name"]: obj for obj in self.env.get_scene_objects()}
                for obj_name, handle in object_handles.items():
                    if obj_name in scene_objs:
                        p = scene_objs[obj_name]["position"]
                        q_o = scene_objs[obj_name]["quaternion"]
                        handle.position = (float(p[0]), float(p[1]), float(p[2]))
                        handle.wxyz = (float(q_o[3]), float(q_o[0]), float(q_o[1]), float(q_o[2]))
                        # handle.visible = False

            # Update mode label
            m = state.mode
            task_names = [self.tasks[tid].name for tid in m.task_ids]
            task_names_str = "  \n".join(task_names)
            annotation = ""
            if step_annotations is not None and step < len(step_annotations):
                annotation = f"  \n{step_annotations[step]}"
            lbl.content = (
                f"**Step:** {step} / {len(path) - 1}  \n"
                f"**Mode:** `{m.task_ids}`  \n"
                f"**Tasks:**   \n {task_names_str}"
                f"{annotation}"
            )

        if path_labels is None:
            path_labels = [f"Path {i} ({len(p)} steps)" for i, p in enumerate(paths)]

        # --- GUI controls ---
        path_dropdown = server.gui.add_dropdown(
            label="Path",
            options=path_labels,
            initial_value=path_labels[-1],
        )
        max_steps = max(len(p) for p in paths)
        step_slider = server.gui.add_slider(
            label="Step", min=0, max=max_steps - 1, step=1, initial_value=0,
        )
        play_checkbox = server.gui.add_checkbox("Play", initial_value=False)
        pause_time_field = server.gui.add_number(
            "Pause time (s)", initial_value=pause_time, min=0.0, step=0.001,
        )
        step_size_field = server.gui.add_number(
            "Step size", initial_value=1, min=1, step=1,
        )
        prev_btn = server.gui.add_button("◀ Prev")
        next_btn = server.gui.add_button("Next ▶")
        stop_btn = server.gui.add_button("Stop")
        mode_label = server.gui.add_markdown("")

        _stopped = False

        def current_path() -> "List[State]":
            idx = path_labels.index(path_dropdown.value)
            return paths[idx]

        def clamped_step() -> int:
            return min(step_slider.value, len(current_path()) - 1)

        @stop_btn.on_click
        def _(_):
            nonlocal _stopped
            _stopped = True

        @prev_btn.on_click
        def _(_):
            play_checkbox.value = False
            step = max(clamped_step() - int(step_size_field.value), 0)
            step_slider.value = step
            _set_step(step, current_path(), mode_label)

        @next_btn.on_click
        def _(_):
            play_checkbox.value = False
            step = min(clamped_step() + int(step_size_field.value), len(current_path()) - 1)
            step_slider.value = step
            _set_step(step, current_path(), mode_label)

        @path_dropdown.on_update
        def _(_):
            play_checkbox.value = False
            step_slider.value = 0
            _set_step(0, current_path(), mode_label)

        @step_slider.on_update
        def _(event):
            if not play_checkbox.value:
                step = min(event.target.value, len(current_path()) - 1)
                _set_step(step, current_path(), mode_label)

        _set_step(0, current_path(), mode_label)

        print(f"[viser] Open http://localhost:{port} to view the path.")
        print("[viser] Press Stop in the GUI or Ctrl-C to exit.")

        try:
            while not _stopped:
                if play_checkbox.value:
                    path = current_path()
                    next_step = (clamped_step() + int(step_size_field.value)) % len(path)
                    step_slider.value = next_step
                    _set_step(next_step, path, mode_label)
                time.sleep(pause_time_field.value)
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()

    def get_scenegraph_info_for_mode(self, mode: Mode, is_start_mode: bool = False):
        if not self.manipulating_env:
            return {}

        prev_mode = mode.prev_mode
        if prev_mode is None:
            return self.initial_sg.copy()

        sg = prev_mode.sg.copy()

        active_task = self.get_active_task(prev_mode, mode.task_ids)

        if active_task.type is None or active_task.type == "goto":
            return sg

        obj_name = active_task.frames[1]
        new_parent = active_task.frames[0]

        if new_parent in self.robots:
            # pick: robot grabs the object
            robot_id = self.robots.index(new_parent)
            joints = tuple(mode.entry_configuration[robot_id])
            sg[obj_name] = ("attached", robot_id, tuple(np.round(joints, 3)), joints, None)
        else:
            # place / handover to world: robot releases the object
            prev_entry = prev_mode.sg.get(obj_name)
            if prev_entry is not None and prev_entry[0] == "attached":
                prev_robot_id = prev_entry[1]
                joints = tuple(mode.entry_configuration[prev_robot_id])
                pick_joints = prev_entry[3] if len(prev_entry) > 3 else None
                sg[obj_name] = ("world", prev_robot_id, tuple(np.round(joints, 3)), joints, pick_joints)
            else:
                # print("AAAAAA")
                # print("AAAAAA")
                # print("AAAAAA")
                # print("AAAAAA")
                sg[obj_name] = ("world", None, None, None, None)

        return sg

    def _reposition_object_to_robot_state(self, obj_name: str, robot_id: int, place_joints, pick_joints=None):
        """Put object at the world pose corresponding to placing at (robot_id, place_joints).

        pick_joints is the grasp configuration used when the object was picked up.
        It determines T_rel (the relative transform from the end-effector to the object).
        If pick_joints is None (unknown), this falls back to a no-op — callers should
        provide pick_joints whenever possible.
        """
        if pick_joints is None:
            return

        if obj_name in self._initial_objects:
            self.env.move_object(self._initial_objects[obj_name])

        self.env.attach_object(obj_name, robot_id, list(pick_joints))
        self.env.detach_object(obj_name, robot_id, list(place_joints))

    def _apply_scenegraph(self, sg: dict) -> None:
        """Sync mr_planner_core attachment state with *sg*."""
        for obj_name, new_state in sg.items():
            cur_state = self._current_sg.get(obj_name)
            if cur_state == new_state:
                continue

            state_type = new_state[0]
            robot_id = new_state[1] if len(new_state) > 1 else None
            joints = new_state[3] if len(new_state) > 3 else None

            if state_type == "attached":
                # If the object is currently at a placed (non-initial) world position,
                # we must reset it to its initial world pose before attaching so that
                # the relative transform T_rel is computed correctly.
                if cur_state is not None and cur_state[0] == "world" and cur_state[1] is not None:
                    if obj_name in self._initial_objects:
                        self.env.move_object(self._initial_objects[obj_name])

                # If previously attached to a different robot, detach first.
                if cur_state is not None and cur_state[0] == "attached" and cur_state[1] != robot_id:
                    prev_robot_id = cur_state[1]
                    prev_joints = cur_state[3] if len(cur_state) > 3 else None
                    if prev_joints is not None:
                        self.env.detach_object(obj_name, prev_robot_id, list(prev_joints))

                self.env.attach_object(obj_name, robot_id, list(joints))

            else:  # world
                if robot_id is None:
                    # Target is the initial world state — reset to initial pose.
                    if obj_name in self._initial_objects:
                        if cur_state is not None and cur_state[0] == "attached":
                            _, attach_robot_id, _, attach_joints, _ = cur_state
                            self.env.detach_object(obj_name, attach_robot_id, list(attach_joints))

                        self.env.move_object(self._initial_objects[obj_name])

                else:
                    # Target is a specific placed world position.
                    # If we came from attached state, detach through normal place.
                    if cur_state is not None and cur_state[0] == "attached":
                        # print(f"pre detach {obj_name}")
                        # print(self.env.get_object(obj_name).z)

                        self.env.detach_object(obj_name, robot_id, list(joints))

                        # print(f"pre detach {obj_name}")
                        # print(self.env.get_object(obj_name).x, self.env.get_object(obj_name).y, self.env.get_object(obj_name).z)
                    else:
                        # If we came from a previous world state, compute placement exactly.
                        pick_joints = new_state[4] if len(new_state) > 4 else None
                        self._reposition_object_to_robot_state(obj_name, robot_id, joints, pick_joints)

        self._current_sg = sg

        # self.env.reset_scene()
        # self.env.update_scene()

    
    def _cache_mode_object_poses(self, mode: Mode) -> None:
        poses = {}
        for obj in self.env.get_scene_objects():
            name = obj.get("name")
            if name not in self._initial_objects:
                continue

            pos = obj.get("position")
            quat = obj.get("quaternion")
            poses[name] = {
                "position": tuple(float(x) for x in pos),
                "quaternion": tuple(float(x) for x in quat),
            }

            # print(f"caching {name}, mode {mode}")
            # print(poses[name])

        self._mode_object_poses[mode.id] = poses

        # print()
        # print(f"cacheing {mode}")
        # for k, v in poses.items():
        #     print(k, v)

    def _apply_cached_mode(self, mode: Mode) -> None:
        # First apply the (fast) scenegraph transitions to ensure attachment/detach state is correct.
        self._apply_scenegraph(mode.sg)

        # Then override world-placed object poses with the cached absolute poses.
        mode_cache = self._mode_object_poses.get(mode.id, {})
        for obj_name, state in mode.sg.items():
            if state[0] != "world":
                # print(f"skipping cache for {obj_name}")
                continue

            robot_id = state[1] if len(state) > 1 else None
            if robot_id is None:
                continue

            cached = mode_cache.get(obj_name)
            if cached is None or obj_name not in self._initial_objects:
                continue

            # mr_planner_core.Object is not deepcopy-able via Python pickle, so mutate
            # the initial template object temporarily while moving, then restore.
            template_obj = self.env.get_object(obj_name)

            # print(f"pre cache application {obj_name}")
            # print(template_obj.x, template_obj.y, template_obj.z)

            template_obj.x, template_obj.y, template_obj.z = cached["position"]
            qw, qx, qy, qz = cached["quaternion"]
            template_obj.qw, template_obj.qx, template_obj.qy, template_obj.qz = qw, qx, qy, qz

            self.env.move_object(template_obj)

            # print("post cache application")
            # print(self.env.get_object(obj_name).x, self.env.get_object(obj_name).y, self.env.get_object(obj_name).z)

        self._current_sg = mode.sg.copy()

    def set_to_mode(self, m: Mode):
        if not self.manipulating_env:
            return

        if self.current_mode == m:
            return

        if m.id in self._mode_object_poses:
            self._apply_cached_mode(m)
            self.current_mode = m
            # print(f"returning after application of cached mode {m}")
            return

        # TODO: this is still doing something wrong
        # print(m.sg)
        
        self._apply_scenegraph(m.sg)
        self._cache_mode_object_poses(m)

        self.current_mode = m

        # print()


    def show(self, blocking=True):
        self.env.update_scene()

        for obj in self.env.get_scene_objects():
            print(obj)

        time.sleep(10)

    def show_config(self, q, blocking=True):
        self.env.set_joint_positions(q.as_list()) 
        # self.env.update_scene() # no need for this as it happens automatically in the set_joint_posisitons call

        time.sleep(0.001)

    def config_cost(self, start: Configuration, end: Configuration) -> float:
        return config_cost(start, end, self.cost_metric, self.cost_reduction)

    def batch_config_cost(
        self,
        starts: List[Configuration],
        ends: List[Configuration],
        tmp_agent_slice=None,
    ) -> NDArray:
        return batch_config_cost(
            starts,
            ends,
            self.cost_metric,
            self.cost_reduction,
            tmp_agent_slice=tmp_agent_slice,
        )

    def is_collision_free(self, q: Optional[Configuration], mode: Optional[Mode]):
        if q is None:
            raise ValueError

        if mode is not None:
            self.set_to_mode(mode)

        return not self.env.in_collision(q.as_list(), self_only=False)

    def is_collision_free_for_robot(
        self,
        r: List[str] | str,
        q: NDArray,
        m: Mode | None = None,
        collision_tolerance: float | None = None,
        set_mode: bool = True,
    ) -> bool:
        if collision_tolerance is None:
            collision_tolerance = self.collision_tolerance

        if isinstance(r, str):
            r = [r]

        # if q is not None:
        #     self.set_to_mode(m)
        #     for robot in r:
        #         robot_indices = self.robot_idx[robot]
        #         self.C.setJointState(q[robot_indices], self.robot_joints[robot])
        if set_mode:
            self.set_to_mode(m)

        print("need to add prooper treatmet of robot id")

        robot_id = 0

        q_config = NpConfiguration.from_list([q[self.robot_idx[r]] for r in self.robots])
        q_as_list = q_config.as_list()
        self.env.set_joint_positions(q_as_list)
        return not self.env.in_collision_robot(robot_id, q_as_list[robot_id], self_only=False)

    def _batch_is_collision_free(self, qs: List[Configuration], mode: List[int]):
        raise NotImplementedError

    def is_edge_collision_free(
        self,
        q1: Configuration,
        q2: Configuration,
        mode: Mode,
        resolution: Optional[float] = None,
        tolerance: Optional[float] = None,
        include_endpoints: bool = False,
        N_start: int = 0,
        N_max: Optional[int] = None,
        N: Optional[int] = None,
    ) -> bool:
        if resolution is None:
            resolution = self.collision_resolution

        if tolerance is None:
            tolerance = self.collision_tolerance

        # print('q1', q1)
        # print('q2', q2)
        if N is None:
            N = int(config_dist(q1, q2, "max") / resolution) + 1
            N = max(2, N)

        if N_start > N:
            assert False

        if N_max is None:
            N_max = N

        N_max = min(N, N_max)

        # print(N_max)

        # for a distance < resolution * 2, we do not do collision checking
        # if N == 0:
        #     return True

        if mode is not None:
            self.set_to_mode(mode)

        return not self.env.motion_in_collision(q1.as_list(), q2.as_list(), step_size=resolution, self_only=False)

    def is_path_collision_free(
        self,
        path: List[State],
        binary_order: bool = True,
        resolution: Optional[float] = None,
        tolerance: Optional[float] = None,
        check_edges_in_order: bool = False,
        check_start_and_end: bool = True,
    ) -> bool:
        if tolerance is None:
            tolerance = self.collision_tolerance

        if resolution is None:
            resolution = self.collision_resolution

        # print('end', path[-1].q.state())
        # if check_start_and_end and not self.is_collision_free(path[-1].q, path[-1].mode):
        #     return False

        # check whole edge
        for i in range(len(path)-1):
            q1 = path[i].q
            q2 = path[i + 1].q
            mode = path[i].mode

            if not self.is_edge_collision_free(
                q1,
                q2,
                mode,
                resolution=resolution,
                tolerance=tolerance,
                include_endpoints=False,
            ):
                return False

            # valid_edges += 1

        if not self.is_collision_free(path[-1].q, path[-1].mode):
            return False

        return True

@register("vampmr.quad_panda")
class vamp_quad_panda_env(SequenceMixin, VampEnv):
    def __init__(self, agents_can_rotate=True):
        VampEnv.__init__(self)
        self.env = mr_planner_core.VampEnvironment("panda_four")

        info = self.env.info()
        num_robots = int(info["num_robots"])

        self.robots = ["a1", "a2", "a3", "a4"]

        # self.env.enable_meshcat()
        self.env.reset_scene()

        panda_dim = 7
        self.limits = np.array([[-2] * 4 * panda_dim, [2] * 4 * panda_dim])

        self.start_pos = NpConfiguration.from_list([[0] * panda_dim, [0] * panda_dim, [0] * panda_dim, [0] * panda_dim])

        self.robot_dims = {"a1": panda_dim, "a2": panda_dim, "a3": panda_dim, "a4": panda_dim}
        self.robot_idx = {f"a{i+1}": list(range(i * panda_dim, (i + 1) * panda_dim)) for i in range(num_robots)}

        goal_pos = self.start_pos.q

        goal_tf = np.eye(4)
        goal_tf[2, 3] = 0.5

        print(goal_tf)

        ik_poses = {}
        for i, r in enumerate(self.robots):
            current_transform = self.env.end_effector_transform(i, [0]*7) 
            current_transform = np.array(current_transform)    
            current_transform[0, 3] = 0
            current_transform[1, 3] = 0
            current_transform[2, 3] = 0.2
            pose = self.env.inverse_kinematics(i, current_transform, max_restarts=100, tol_pos=0.01, tol_ang=1)
            print(pose)
            if pose:
                ik_poses[i] = pose

        self.tasks = [
            # r1
            Task("a1_goal", ["a1"], SingleGoal(np.array(ik_poses[0]))),
            # r2
            Task("a2_goal", ["a2"], SingleGoal(np.array(ik_poses[1]))),
            Task("a3_goal", ["a3"], SingleGoal(np.array(ik_poses[2]))),
            Task("a4_goal", ["a4"], SingleGoal(np.array(ik_poses[3]))),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(goal_pos),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a1_goal", "a2_goal", "a3_goal", "a4_goal", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self._set_zero_safe_pose(panda_dim)

@register("vampmr.hex_panda")
class vamp_hex_panda_env(SequenceMixin, VampEnv):
    def __init__(self, agents_can_rotate=True):
        VampEnv.__init__(self)
        self.env = mr_planner_core.VampEnvironment("panda_six")

        objects = self.env.get_scene_objects()
        print(objects)

        # move all the table segments down a bit
        for i in range(12):
            name = "panda_six_table_segment_" + str(i)
            self.env.remove_object(name)    

        # self.env.remove_object("panda_six_plank")
        
        info = self.env.info()
        num_robots = int(info["num_robots"])

        self.robots = ["a1", "a2", "a3", "a4", "a5", "a6"]

        # self.env.enable_meshcat()
        self.env.update_scene()

        panda_dim = 7
        self.limits = np.array([[-2] * num_robots * panda_dim, [2] * num_robots * panda_dim])

        self.start_pos = NpConfiguration.from_list([[0] * panda_dim, [0] * panda_dim, [0] * panda_dim, [0] * panda_dim, [0] * panda_dim, [0] * panda_dim])

        self.robot_dims = {"a1": panda_dim, "a2": panda_dim, "a3": panda_dim, "a4": panda_dim, "a5": panda_dim, "a6": panda_dim}
        self.robot_idx = {f"a{i+1}": list(range(i * panda_dim, (i + 1) * panda_dim)) for i in range(num_robots)}
        # self.C.view(True)

        goal_pos = self.start_pos.q

        self.tasks = [
            # r1
            Task("a1_goal", ["a1"], SingleGoal(np.array([0.0, 0.5, -1, 0, -0, 0, 0]))),
            # r2
            Task("a2_goal", ["a2"], SingleGoal(np.array([-0.0, 0.5, -1, 0, 0, 0, 0]))),
            Task("a3_goal", ["a3"], SingleGoal(np.array([-0.0, 0.5, -1, 0, 0, 0, 0]))),
            Task("a4_goal", ["a4"], SingleGoal(np.array([-0.0, 0.5, -1, 0, 0, 0, 0]))),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(goal_pos),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a1_goal", "a2_goal", "a3_goal", "a4_goal", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self._set_zero_safe_pose(panda_dim)


@register("vampmr.dual_ur5")
class vamp_hex_panda_env(SequenceMixin, VampEnv):
    def __init__(self, num_repetitions: int = 2):
        VampEnv.__init__(self)
        self.env = mr_planner_core.VampEnvironment("dual_ur5")

        floor = mr_planner_core.Object()
        floor.name = "floor"
        floor.shape = mr_planner_core.Object.Box
        floor.state = mr_planner_core.Object.Static
        floor.x = 0.
        floor.y = 0.0
        floor.z = -0.0
        floor.qw = 1.0
        floor.qx = 0.0
        floor.qy = 0.0
        floor.qz = 0.0
        floor.width = 10   # x
        floor.height = 0.01  # y
        floor.length = 10  # z
        self.env.add_object(floor)

        self.env.set_allowed_collision("floor", "ur5_0_arm_base_link", True)
        self.env.set_allowed_collision("floor", "ur5_0_arm_shoulder_link", True)

        self.env.set_allowed_collision("floor", "ur5_1_arm_base_link", True)
        self.env.set_allowed_collision("floor", "ur5_1_arm_shoulder_link", True)
        
        def make_transform(angle_z, tx, ty, tz):
            c, s = np.cos(angle_z), np.sin(angle_z)
            return [
                [c, -s, 0, tx],
                [s,  c, 0, ty],
                [0,  0, 1, tz],
                [0,  0, 0,  1],
            ]

        q_offset = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        def rai_to_vamp_config(q_rai):
            return q_rai + q_offset

        self.env.set_robot_base_transform(0, make_transform(-np.pi/2,        0.0,        0.0,        -0.9144))
        self.env.set_robot_base_transform(1, make_transform(np.pi/2, 0.88128092, -0.01226491, -0.9144))

        info = self.env.info()
        num_robots = int(info["num_robots"])

        self.robots = ["a1", "a2"]

        # self.env.enable_meshcat()
        # self.env.update_scene()

        ur_dim = 6
        self.limits = np.array([[-4] * num_robots * ur_dim, [4] * num_robots * ur_dim])

        # start_config_rai = np.array([0, 0, 0, 0, 0, 0])
        start_config_rai = np.array([0, -.5, 1, .5, -1.57, 0])

        start_config = rai_to_vamp_config(start_config_rai)

        self.start_pos = NpConfiguration.from_list([start_config, start_config])

        self.robot_dims = {"a1": ur_dim, "a2": ur_dim}
        self.robot_idx = {f"a{i+1}": list(range(i * ur_dim, (i + 1) * ur_dim)) for i in range(num_robots)}
        # self.C.view(True)

        goal_pos = self.start_pos.q

        p1 = [ 6.05587010e-01, -1.82889078e-03, -1.86183192e+00, -7.57176381e-01, 8.65312213e-01, -8.24393655e-04]
        p2 = [ 0.52888098, -0.14924623, -1.7284803,  -0.64827086,  0.72253539,  0.01270607]

        goal_tasks = []
        sequence_names = []
        for rep in range(num_repetitions):
            suffix = f"_{rep + 1}" if num_repetitions > 1 else ""
            a1_name = f"a1_goal{suffix}"
            a2_name = f"a2_goal{suffix}"
            goal_tasks.append(Task(a1_name, ["a1"], SingleGoal(rai_to_vamp_config(np.array(p1)))))
            goal_tasks.append(Task(a2_name, ["a2"], SingleGoal(rai_to_vamp_config(np.array(p2)))))
            sequence_names += [a1_name, a2_name]

        self.tasks = goal_tasks + [
            Task("terminal", self.robots, SingleGoal(goal_pos)),
        ]

        self.sequence = self._make_sequence_from_names(sequence_names + ["terminal"])

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self._set_zero_safe_pose(ur_dim)


@register("vampmr.dual_ur5_with_box")
class vamp_hex_panda_env(SequenceMixin, VampEnv):
    def __init__(self, agents_can_rotate=True):
        VampEnv.__init__(self)
        self.env = mr_planner_core.VampEnvironment("dual_ur5")

        self.manipulating_env = True

        info = self.env.info()
        num_robots = int(info["num_robots"])

        self.robots = ["a1", "a2"]

        box = mr_planner_core.Object()
        box.name = "box1"
        box.shape = mr_planner_core.Object.Box
        box.state = mr_planner_core.Object.Static
        box.x = 0.5
        box.y = 0.0
        box.z = 0.5
        box.qw = 1.0
        box.qx = 0.0
        box.qy = 0.0
        box.qz = 0.0
        box.width = 0.3   # x
        box.height = 0.1  # y
        box.length = 0.1  # z
        self.env.add_object(box)
        self._initial_objects[box.name] = box

        floor = mr_planner_core.Object()
        floor.name = "floor"
        floor.shape = mr_planner_core.Object.Box
        floor.state = mr_planner_core.Object.Static
        floor.x = 0.
        floor.y = 0.0
        floor.z = -0.0
        floor.qw = 1.0
        floor.qx = 0.0
        floor.qy = 0.0
        floor.qz = 0.0
        floor.width = 10   # x
        floor.height = 0.01  # y
        floor.length = 10  # z
        self.env.add_object(floor)

        self.env.set_allowed_collision("floor", "ur5_0_arm_base_link", True)
        self.env.set_allowed_collision("floor", "ur5_0_arm_shoulder_link", True)

        self.env.set_allowed_collision("floor", "ur5_1_arm_base_link", True)
        self.env.set_allowed_collision("floor", "ur5_1_arm_shoulder_link", True)

        def make_transform(angle_z, tx, ty, tz):
            c, s = np.cos(angle_z), np.sin(angle_z)
            return [
                [c, -s, 0, tx],
                [s,  c, 0, ty],
                [0,  0, 1, tz],
                [0,  0, 0,  1],
            ]

        q_offset = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        def rai_to_vamp_config(q_rai):
            return q_rai + q_offset

        self.env.set_robot_base_transform(0, make_transform(-np.pi/2,        0.0,        0.0,        -0.9144))
        self.env.set_robot_base_transform(1, make_transform(np.pi/2, 0.88128092, -0.01226491, -0.9144))

        # self.env.enable_meshcat()
        self.env.update_scene()

        ur_dim = 6
        self.limits = np.array([[-4] * num_robots * ur_dim, [4] * num_robots * ur_dim])

        start_config_rai = np.array([0, -.5, 1, .5, -1.57, 0])
        start_config = rai_to_vamp_config(start_config_rai)

        self.start_pos = NpConfiguration.from_list([start_config, start_config])

        self.robot_dims = {"a1": ur_dim, "a2": ur_dim}
        self.robot_idx = {f"a{i+1}": list(range(i * ur_dim, (i + 1) * ur_dim)) for i in range(num_robots)}
        # self.C.view(True)

        goal_pos = self.start_pos.q

        self.initial_sg = {"box1": ("world", None, None, None, None)}

        self.tasks = [
            # r1
            Task("a1_goal", ["a1"], SingleGoal(start_config + np.array([0, 0.0, 0., 0, 0, 0])), type="pick", frames=["a1", "box1"]),
            # r2
            Task("a2_goal", ["a2"], SingleGoal(np.array([-0.0, -0.5, -1, 0, 0, 0]))),
            # Task("a1_place", ["a1"], SingleGoal(start_config + np.array([0, 0, 0, 0, 1, 0]))),
            Task("a1_place", ["a1"], SingleGoal(start_config + np.array([0, 0, 0, 0, 1, 0])), type="place", frames=["world", "box1"]),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(goal_pos),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a1_goal", "a2_goal", "a1_place", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self._set_zero_safe_pose(ur_dim)

@register("vampmr.quad_ur5")
class vamp_hex_panda_env(SequenceMixin, VampEnv):
    def __init__(self, num_repetitions=2):
        VampEnv.__init__(self)
        self.env = mr_planner_core.VampEnvironment("quad_ur5")

        floor = mr_planner_core.Object()
        floor.name = "floor"
        floor.shape = mr_planner_core.Object.Box
        floor.state = mr_planner_core.Object.Static
        floor.x = 0.
        floor.y = 0.0
        floor.z = -0.0
        floor.qw = 1.0
        floor.qx = 0.0
        floor.qy = 0.0
        floor.qz = 0.0
        floor.width = 10   # x
        floor.height = 0.01  # y
        floor.length = 10  # z
        self.env.add_object(floor)

        for i in range(4):
            self.env.set_allowed_collision("floor", f"ur5_{i}_arm_base_link", True)
            self.env.set_allowed_collision("floor", f"ur5_{i}_arm_shoulder_link", True)
        
        def make_transform(angle_z, tx, ty, tz):
            c, s = np.cos(angle_z), np.sin(angle_z)
            return [
                [c, -s, 0, tx],
                [s,  c, 0, ty],
                [0,  0, 1, tz],
                [0,  0, 0,  1],
            ]

        q_offset = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        def rai_to_vamp_config(q_rai):
            return q_rai + q_offset

        self.env.set_robot_base_transform(0, make_transform(np.pi/2,   0.36, -0.36, -0.9144))
        self.env.set_robot_base_transform(1, make_transform(-np.pi/2, -0.36, -0.36, -0.9144))
        self.env.set_robot_base_transform(2, make_transform(np.pi/2, 0.36,  0.36, -0.9144))
        self.env.set_robot_base_transform(3, make_transform(-np.pi/2, -0.36,  0.36, -0.9144))

        info = self.env.info()
        num_robots = int(info["num_robots"])

        self.robots = ["a1", "a2", "a3", "a4"]

        # self.env.enable_meshcat()
        # self.env.reset_scene()
        # self.env.update_scene()

        ur_dim = 6
        lower = rai_to_vamp_config(np.array([-np.pi, -2, -np.pi, -np.pi, -np.pi, -np.pi])).tolist()
        upper = rai_to_vamp_config(np.array([np.pi, 0.75, np.pi, np.pi, np.pi, np.pi])).tolist()
        self.limits = np.array([lower * num_robots, 
                                upper * num_robots])
        
        start_config_rai = np.array([0, -.5, 1, .5, -1.57, 0])
        start_config = rai_to_vamp_config(start_config_rai)

        self.start_pos = NpConfiguration.from_list([start_config, start_config, start_config, start_config])
        
        self.env.set_joint_positions(self.start_pos.as_list())

        self.robot_dims = {"a1": ur_dim, "a2": ur_dim, "a3": ur_dim, "a4": ur_dim}
        self.robot_idx = {f"a{i+1}": list(range(i * ur_dim, (i + 1) * ur_dim)) for i in range(num_robots)}
        # self.C.view(True)

        goal_pos = self.start_pos.q

        p1 = np.array([-1.08443716e+00,  2.65812454e-01,  1.19333815e+00,4.95651623e-01, -1.51123655e+00, -6.12287039e-04])
        p2 = np.array([ 0.25372525,  0.15432183,  1.37953107,  0.4719497 , -1.11911701, 0.01556415])
        p3 = np.array([ 0.25500657,  0.15616105,  1.37700978,  0.47377805, -1.12129939, 0.00521846])
        p4 = np.array([-1.08587427,  0.27071415,  1.18725584,  0.50407107, -1.5087889, 0.00212636])

        goal_tasks = []
        sequence_names = []
        for rep in range(num_repetitions):
            suffix = f"_{rep + 1}" if num_repetitions > 1 else ""
            a1_name = f"a1_goal{suffix}"
            a2_name = f"a2_goal{suffix}"
            a3_name = f"a3_goal{suffix}"
            a4_name = f"a4_goal{suffix}"
            goal_tasks.append(Task(a1_name, ["a1"], SingleGoal(rai_to_vamp_config(p1))))
            goal_tasks.append(Task(a2_name, ["a2"], SingleGoal(rai_to_vamp_config(p2))))
            goal_tasks.append(Task(a3_name, ["a3"], SingleGoal(rai_to_vamp_config(p3))))
            goal_tasks.append(Task(a4_name, ["a4"], SingleGoal(rai_to_vamp_config(p4))))
            sequence_names += [a1_name, a2_name, a3_name, a4_name] 

        self.tasks = goal_tasks + [
            Task("terminal", self.robots, SingleGoal(goal_pos)),
        ]

        self.sequence = self._make_sequence_from_names(sequence_names + ["terminal"])

        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self._set_zero_safe_pose(ur_dim)


@register("vampmr.ur5_box_stacking")
class vamp_ur5_box_stacking_env(SequenceMixin, VampEnv):
    def __init__(self, num_robots: int = 4, num_boxes: int = 8):
        VampEnv.__init__(self)
        self.env = mr_planner_core.VampEnvironment("quad_ur5")
        self.manipulating_env = True

        from . import rai_config
        C, keyframes, all_robots = rai_config.make_box_stacking_env(
            num_robots=num_robots, num_boxes=num_boxes, robot_types="ur5"
        )

        self.robots = [f"a{i+1}" for i in range(num_robots)]
        ur_dim = 6

        def make_transform(angle_z, tx, ty, tz):
            c, s = np.cos(angle_z), np.sin(angle_z)
            return [
                [c, -s, 0, tx],
                [s,  c, 0, ty],
                [0,  0, 1, tz],
                [0,  0, 0,  1],
            ]

        q_offset = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        def rai_to_vamp_config(q_rai):
            return np.array(q_rai) + q_offset * 0

        # Robot base transforms — same layout as vampmr.quad_ur5; adjust offsets as needed
        individual_offsets = [
            [-0.1, 0.1, 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0.0, 0.02, 0.],
        ]
        for i in range(4):
            pos = C.getFrame(f"a{i+1}_ur_base").getPosition() + np.array(individual_offsets[i]) * 0
            z_offset = 0.9144 - 0.
            rot = np.pi/2
            if i>=2:
                rot = -np.pi/2
            self.env.set_robot_base_transform(i, make_transform(rot + np.pi/2, pos[0], pos[1], pos[2] - z_offset))

        for i in range(num_robots):
            self.env.set_allowed_collision("floor", f"ur5_{i}_arm_base_link", True)
            self.env.set_allowed_collision("floor", f"ur5_{i}_arm_shoulder_link", True)

        # Floor
        floor = mr_planner_core.Object()
        floor.name = "floor"
        floor.shape = mr_planner_core.Object.Box
        floor.state = mr_planner_core.Object.Static
        floor.x = 0.; floor.y = 0.; floor.z = 0.
        floor.qw = 1.; floor.qx = 0.; floor.qy = 0.; floor.qz = 0.
        floor.width = 10; floor.height = 0.01; floor.length = 10
        self.env.add_object(floor)

        # Table — get world pose from rai config
        table_pos  = C.getFrame("table").getPosition()   # [x, y, z]
        table_quat = C.getFrame("table").getQuaternion() # [w, x, y, z]

        table_obj = mr_planner_core.Object()
        table_obj.name = "table"
        table_obj.shape = mr_planner_core.Object.Box
        table_obj.state = mr_planner_core.Object.Static
        table_obj.x = float(table_pos[0]); table_obj.y = float(table_pos[1]); table_obj.z = float(table_pos[2])
        table_obj.qw = float(table_quat[3]); table_obj.qx = float(table_quat[0])
        table_obj.qy = float(table_quat[1]); table_obj.qz = float(table_quat[2])
        table_obj.width = 3.0; table_obj.height = 0.06; table_obj.length = 3.0
        self.env.add_object(table_obj)

        for i in range(num_robots):
            self.env.set_allowed_collision("table", f"ur5_{i}_arm_base_link", True)
            self.env.set_allowed_collision("table", f"ur5_{i}_arm_shoulder_link", True)

        # Boxes — extract world positions from rai config
        boxes_seen = []
        for r_rai, b, qs, g in keyframes:
            if b not in boxes_seen:
                boxes_seen.append(b)

        for box_name in boxes_seen:
            pos  = C.getFrame(box_name).getPosition()   # [x, y, z]
            quat = C.getFrame(box_name).getQuaternion() # [w, x, y, z]

            box_obj = mr_planner_core.Object()
            box_obj.name = box_name
            box_obj.shape = mr_planner_core.Object.Box
            box_obj.state = mr_planner_core.Object.Static
            box_obj.x = float(pos[0]); box_obj.y = float(pos[1]); box_obj.z = float(pos[2])
            box_obj.qw = float(quat[0]); box_obj.qx = float(quat[1])
            box_obj.qy = float(quat[2]); box_obj.qz = float(quat[3])
            box_obj.width = 0.05; box_obj.height = 0.05; box_obj.length = 0.05

            self.env.add_object(box_obj)
            self._initial_objects[box_name] = box_obj
            self.initial_sg[box_name] = ("world", None, None, None, None)

        # Joint limits and start config
        # lower = rai_to_vamp_config(np.array([-np.pi, -3, -np.pi, -np.pi, -np.pi, -np.pi])).tolist()
        # upper = rai_to_vamp_config(np.array([np.pi, 0.0, np.pi, np.pi, np.pi, np.pi])).tolist()
        # self.limits = np.array([lower * num_robots, upper * num_robots])

        self.robot_dims = {f"a{i+1}": ur_dim for i in range(num_robots)}
        self.robot_idx = {f"a{i+1}": list(range(i * ur_dim, (i + 1) * ur_dim)) for i in range(num_robots)}
        
        # start_config = rai_to_vamp_config(np.array([0, -.5, 1, .5, -1.57, 0]))
        # start_config = rai_to_vamp_config(np.array([0, -2., 1, -1, -1.57, 0]))
        self.start_pos = NpConfiguration.from_list([C.getJointState()[self.robot_idx[r]] for r in self.robots])
        self.env.set_joint_positions(self.start_pos.as_list())

        self.limits = C.getJointLimits()

        # Build pick/place tasks from rai keyframes
        # keyframe format: (robot_prefix, box_name, [pick_q, place_q], goal_name)
        # robot_prefix is like "a1_ur_"; split on "_" to get "a1"
        self.tasks = []
        task_names_seq = []

        for r_rai, b, qs, _g in keyframes[:num_boxes*2]:
            r = r_rai.split("_")[0]  # "a1_ur_" -> "a1"
            pick_q  = rai_to_vamp_config(qs[0])
            place_q = rai_to_vamp_config(qs[1])

            pick_name  = f"{r}_pick_{b}"
            place_name = f"{r}_place_{b}"

            self.tasks.append(Task(pick_name,  [r], SingleGoal(pick_q),  "pick",  frames=[r, b]))
            self.tasks.append(Task(place_name, [r], SingleGoal(place_q), "place", frames=["world", b]))
            task_names_seq += [pick_name, place_name]

            # print(place_name)
            # print(place_q)

        self.tasks.append(Task("terminal", self.robots, SingleGoal(self.start_pos.q)))
        task_names_seq.append("terminal")

        self.sequence = self._make_sequence_from_names(task_names_seq)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.00

        self._set_zero_safe_pose(ur_dim)
