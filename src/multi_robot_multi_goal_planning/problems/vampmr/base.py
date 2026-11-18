import copy
import numpy as np
import time

from typing import List, Optional
from numpy.typing import NDArray

import matplotlib.pyplot as plt

from ..planning_env import generate_binary_search_indices

from ..core.configuration import Configuration, NpConfiguration, config_dist

from ..planning_env import (
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
from ..core.goals import SingleGoal, GoalSet, GoalRegion, ConditionalGoal

import sys
sys.path.append("/usr/local/lib/python3.10/dist-packages")  # TODO: install mr_planner_core into venv
import mr_planner_core
sys.path.pop()

from ..core.registry import register

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
