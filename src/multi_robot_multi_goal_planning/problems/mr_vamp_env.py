import numpy as np
import time

from typing import List, Optional
from numpy.typing import NDArray

import matplotlib.pyplot as plt

from .planning_env import (
    generate_binary_search_indices,
)

from .configuration import (
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
from .goals import (
    SingleGoal,
    GoalSet,
    GoalRegion,
    ConditionalGoal,
)

import sys
sys.path.append("/usr/local/lib/python3.10/dist-packages")  # TODO: install mr_planner_core into venv
import mr_planner_core
sys.path.pop()

from .registry import register

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

        self.spec = ProblemSpec(
            agent_type=AgentType.MULTI_AGENT,
            constraints=ConstraintType.UNCONSTRAINED,
            manipulation=ManipulationType.MANIPULATION,
            dependency=DependencyType.FULLY_ORDERED,
            dynamics=DynamicsType.GEOMETRIC,
            goals=GoalType.MULTI_GOAL,
            home_pose=SafePoseType.HAS_NO_SAFE_HOME_POSE,
        )

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
        num_robots = path[0].q.num_agents()

        path_as_list = []
        for i in range(num_robots):
            per_agent_path = []
            for p in path:
                per_agent_path.append(p.q[i])

            path_as_list.append(per_agent_path)

        self.env.play_trajectory(path_as_list)

    def _viser_build_scene(self, server) -> None:
        """Add static environment objects (boxes, spheres, cylinders) to a viser server."""
        objects = self.env.get_scene_objects()
        _ENV_COLOR = (180, 180, 180)

        for obj in objects:
            name = obj["name"]
            pos = obj["position"]        # [x, y, z]
            q = obj["quaternion"]        # [qw, qx, qy, qz]
            wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
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
        primitives_only=False
    ) -> None:
        """Display one or more planned paths interactively in viser.

        Renders the robot sphere decomposition using get_sphere_poses() and
        static environment objects using get_scene_objects().

        Args:
            paths: A single path (List[State]) or a list of paths to compare.
            pause_time: Initial time between frames during playback (seconds).
            port: Port for the viser HTTP server.
            path_labels: Optional display names for each path.
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

        # Add static environment objects
        self._viser_build_scene(server)

        # Pre-create handles from initial config
        first_state = paths[0][0]
        initial_spheres = self.env.get_sphere_poses(first_state.q.as_list())

        # Pre-create one icosphere handle per robot sphere
        sphere_handles = []
        for i, (cx, cy, cz, r) in enumerate(initial_spheres):
            handle = server.scene.add_icosphere(
                name=f"robots/sphere_{i}",
                radius=float(r),
                color=_ROBOT_COLOR,
                position=(float(cx), float(cy), float(cz)),
                subdivisions=2,
            )
            sphere_handles.append(handle)

        def _set_step(step: int, path: "List[State]") -> None:
            state = path[min(step, len(path) - 1)]
            spheres = self.env.get_sphere_poses(state.q.as_list())
            for handle, (cx, cy, cz, _r) in zip(sphere_handles, spheres):
                handle.position = np.array([cx, cy, cz], dtype=np.float32)

        if path_labels is None:
            path_labels = [f"Path {i} ({len(p)} steps)" for i, p in enumerate(paths)]

        # --- GUI controls ---
        path_dropdown = server.gui.add_dropdown(
            label="Path",
            options=path_labels,
            initial_value=path_labels[0],
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

        @path_dropdown.on_update
        def _(_):
            play_checkbox.value = False
            step_slider.value = 0
            _set_step(0, current_path())

        @step_slider.on_update
        def _(event):
            if not play_checkbox.value:
                _set_step(event.target.value, current_path())

        _set_step(0, current_path())

        print(f"[viser] Open http://localhost:{port} to view the path.")
        print("[viser] Press Stop in the GUI or Ctrl-C to exit.")

        try:
            while not _stopped:
                if play_checkbox.value:
                    path = current_path()
                    next_step = (clamped_step() + int(step_size_field.value)) % len(path)
                    step_slider.value = next_step
                    _set_step(next_step, path)
                    m = path[next_step].mode
                    task_names_str = "  \n ".join(
                        f"- {t.name}" for t in self.tasks if t.name != "terminal"
                    )
                    mode_label.content = (
                        f"**Step:** {next_step} / {len(path) - 1}  \n"
                        f"**Mode:** `{m.task_ids}`  \n"
                        f"**Tasks:**   \n {task_names_str}"
                    )
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
            joints = list(mode.entry_configuration[robot_id])
            sg[obj_name] = ("attached", robot_id, joints)
        else:
            # place / handover to world: robot releases the object
            prev_entry = prev_mode.sg.get(obj_name)
            if prev_entry is not None and prev_entry[0] == "attached":
                prev_robot_id = prev_entry[1]
                joints = list(mode.entry_configuration[prev_robot_id])
                sg[obj_name] = ("world", prev_robot_id, joints)
            else:
                sg[obj_name] = ("world", None, None)

        return sg

    def _apply_scenegraph(self, sg: dict) -> None:
        """Sync mr_planner_core attachment state with *sg*, making minimal API calls."""
        for obj_name, state in sg.items():
            if self._current_sg.get(obj_name) == state:
                continue
            if state[0] == "attached":
                _, robot_id, joints = state
                self.env.attach_object(obj_name, robot_id, list(joints))
            else:  # "world"
                _, robot_id, joints = state
                if robot_id is not None and joints is not None:
                    self.env.detach_object(obj_name, robot_id, list(joints))
            self._current_sg[obj_name] = state

    def show(self, blocking=True):
        self.env.update_scene()

    def show_config(self, q, blocking=True):
        self.env.set_joint_positions(q.as_list()) 
        self.env.update_scene() # no need for this as it happens automatically in the set_joint_posisitons call

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

        # for a distance < resolution * 2, we do not do collision checking
        # if N == 0:
        #     return True

        if mode is not None:
            self.set_to_mode(mode)

        return not self.env.motion_in_collision(q1.as_list(), q2.as_list(), step_size=resolution, self_only=False)

    def set_to_mode(self, m: Mode):
        if not self.manipulating_env:
            return
        self._apply_scenegraph(m.sg)

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

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array([0] * panda_dim)

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

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array([0] * panda_dim)


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

        self.env.set_robot_base_transform(0, make_transform(np.pi/2,        0.0,        0.0,        -0.9144))
        self.env.set_robot_base_transform(1, make_transform(-np.pi/2, 0.88128092, -0.01226491, -0.9144))

        info = self.env.info()
        num_robots = int(info["num_robots"])

        self.robots = ["a1", "a2"]

        # self.env.enable_meshcat()
        # self.env.update_scene()

        ur_dim = 6
        self.limits = np.array([[-4] * num_robots * ur_dim, [4] * num_robots * ur_dim])

        start_config_rai = np.array([0, 0, 0, 0, 0, 0])
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

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array([0] * ur_dim)


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
        box.width = 0.1   # x
        box.height = 0.1  # y
        box.length = 0.1  # z
        self.env.add_object(box)

        # self.env.enable_meshcat()
        self.env.update_scene()

        ur_dim = 6
        self.limits = np.array([[-4] * num_robots * ur_dim, [4] * num_robots * ur_dim])

        self.start_pos = NpConfiguration.from_list([[0] * ur_dim, [0] * ur_dim])

        self.robot_dims = {"a1": ur_dim, "a2": ur_dim}
        self.robot_idx = {f"a{i+1}": list(range(i * ur_dim, (i + 1) * ur_dim)) for i in range(num_robots)}
        # self.C.view(True)

        goal_pos = self.start_pos.q

        self.initial_sg = {"box1": ("world", None, None)}

        self.tasks = [
            # r1
            Task("a1_goal", ["a1"], SingleGoal(np.array([0.0, 0.5, -1, 0, -0, 0])), type="pick", frames=["a1", "box1"]),
            # r2
            Task("a2_goal", ["a2"], SingleGoal(np.array([-0.0, 0.5, -1, 0, 0, 0]))),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(goal_pos),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a1_goal", "a2_goal", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array([0] * ur_dim)

@register("vampmr.quad_ur5")
class vamp_hex_panda_env(SequenceMixin, VampEnv):
    def __init__(self, agents_can_rotate=True):
        VampEnv.__init__(self)
        self.env = mr_planner_core.VampEnvironment("quad_ur5")

        info = self.env.info()
        num_robots = int(info["num_robots"])

        self.robots = ["a1", "a2", "a3", "a4"]

        # self.env.enable_meshcat()
        # self.env.reset_scene()
        # self.env.update_scene()

        ur_dim = 6
        self.limits = np.array([[-4] * num_robots * ur_dim, [4] * num_robots * ur_dim])

        self.start_pos = NpConfiguration.from_list([[0] * ur_dim, [0] * ur_dim, [0] * ur_dim, [0] * ur_dim])
        
        for i in range(4):
            self.start_pos[i][0] = 0
            self.start_pos[i][1] = 0

        self.robot_dims = {"a1": ur_dim, "a2": ur_dim, "a3": ur_dim, "a4": ur_dim}
        self.robot_idx = {f"a{i+1}": list(range(i * ur_dim, (i + 1) * ur_dim)) for i in range(num_robots)}
        # self.C.view(True)

        goal_pos = self.start_pos.q

        self.tasks = [
            # r1
            Task("a1_goal", ["a1"], SingleGoal(np.array([0.0, 0, -1, 0, -0, 0]))),
            # r2
            Task("a2_goal", ["a2"], SingleGoal(np.array([-0.0, 0, -1, 0, 0, 0]))),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(goal_pos),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a1_goal", "a2_goal", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array([0] * ur_dim)