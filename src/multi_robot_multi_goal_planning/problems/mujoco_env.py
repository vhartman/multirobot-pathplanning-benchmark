import numpy as np

import time
import sys
import select

from typing import List, Optional
from numpy.typing import NDArray

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
    BaseProblem,
    BaseModeLogic,
    SequenceMixin,
    Mode,
    State,
    Task,
    ProblemSpec,
    AgentType,
    GoalType,
    ConstraintType,
    DynamicsType,
    ManipulationType,
    DependencyType,
    SafePoseType,
    SingleGoal,
)

import mujoco
import mujoco.viewer

from concurrent.futures import ProcessPoolExecutor
import threading
from queue import Queue

class MujocoEnvironment(BaseProblem):
    """
    Simple environment, only supporting rectangle and sphere obstacles, and spherical agents.
    """

    def get_body_ids(self, root_name):
        # Build parent->children mapping
        parent2children = {}
        for i in range(self.model.nbody):
            pid = int(self.model.body(i).parentid)
            parent2children.setdefault(pid, []).append(i)

        # Recursively collect all body IDs in subtree
        def subtree_body_ids(body_id):
            ids = [body_id]
            for child in parent2children.get(body_id, []):
                ids.extend(subtree_body_ids(child))
            return ids

        root_id = self.model.body(root_name).id
        robot_body_ids = np.array(subtree_body_ids(root_id))
        return robot_body_ids

    def collect_joints(self, root_name):
        robot_body_ids = self.get_body_ids(root_name)

        joint_names = [
            self.model.joint(j).name
            for j in range(self.model.njnt)
            if self.model.jnt_bodyid[j] in robot_body_ids
        ]

        return joint_names

    def collect_joint_ids(self, root_name):
        robot_body_ids = self.get_body_ids(root_name)

        joint_ids = [
            j
            for j in range(self.model.njnt)
            if self.model.jnt_bodyid[j] in robot_body_ids
        ]

        return joint_ids

    def __init__(self, xml_path, n_data_pool:int = 1):
        self.limits = None

        self.cost_metric = "euclidean"
        self.cost_reduction = "max"

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.viewer = None
        self._enter_pressed = False

        self.limits = np.zeros((2, self.model.njnt))

        # Preallocated pool for parallel collision checking
        self._data_pool = [mujoco.MjData(self.model) for _ in range(n_data_pool)]

        for i in range(self.model.njnt):
            self.limits[0, i] = self.model.jnt_range[i, 0]  # lower limit
            self.limits[1, i] = self.model.jnt_range[i, 1]  # upper limit

        self.robot_idx = {}
        self.robot_dims = {}
        self.robot_joints = {}

        for r in self.robots:
            self.robot_joints[r] = self.collect_joints(r)
            self.robot_idx[r] = self.collect_joint_ids(r)
            self.robot_dims[r] = len(self.robot_joints[r])

        self.spec = ProblemSpec(
            agent_type=AgentType.MULTI_AGENT,
            constraints=ConstraintType.UNCONSTRAINED,
            manipulation=ManipulationType.MANIPULATION,
            dependency=DependencyType.FULLY_ORDERED,
            dynamics=DynamicsType.GEOMETRIC,
            goals=GoalType.MULTI_GOAL,
            home_pose=SafePoseType.HAS_NO_SAFE_HOME_POSE,
        )

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

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
        for i in range(len(path)):
            self.show_config(path[i].q, stop)

            # if export:
            #     self.C.view_savePng("./z.vid/")

            dt = pause_time
            if adapt_to_max_distance:
                if i < len(path) - 1:
                    v = 5
                    diff = config_dist(path[i].q, path[i + 1].q, "max_euclidean")
                    dt = diff / v

            time.sleep(dt)

        if stop_at_end:
            self.show_config(path[-1].q, True)

    def sample_config_uniform_in_limits(self):
        rnd = np.random.uniform(low=self.limits[0, :], high=self.limits[1, :])
        q = self.start_pos.from_flat(rnd)

        return q

    def get_scenegraph_info_for_mode(self, mode: Mode, is_start_mode: bool = False):
        return {}

    def _key_callback(self, key):
        # Enter key toggles pause
        if chr(key) == "ā":  # Enter key
            self._enter_pressed = True

    def show(self, blocking=True):
        """Open viewer at current state."""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self._key_callback
            )
        self.viewer.sync()

        if blocking:
            self._enter_pressed = False
            while self.viewer.is_running() and not self._enter_pressed:
                self.viewer.sync()

    def show_config(self, q, blocking=True):
        """Display a configuration `q` in the viewer."""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self._key_callback
            )

        self.data.qpos[:] = q.state()
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        self.viewer.sync()

        if blocking:
            self._enter_pressed = False
            while self.viewer.is_running() and not self._enter_pressed:
                self.viewer.sync()

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
        # data = mujoco.MjData(self.model)
        self.data.qpos[:] = q.state()
        mujoco.mj_forward(self.model, self.data)

        # If any contact distance < 0, collision
        for i in range(self.data.ncon):
            if self.data.contact[i].dist < self.collision_tolerance:
                return False

        return True

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

        idx = generate_binary_search_indices(N)

        q1_state = q1.state()
        q2_state = q2.state()
        dir = (q2_state - q1_state) / (N - 1)

        for i in idx[N_start:N_max]:
            if not include_endpoints and (i == 0 or i == N - 1):
                continue

            # print(i / (N-1))
            q = q1_state + dir * (i)
            q = q1.from_flat(q)

            if not self.is_collision_free(q, mode):
                return False

        return True

    def set_to_mode(self, m: List[int]):
        if not self.manipulating_env:
            return
        else:
            raise NotImplementedError("This is not supported for this environment.")


class four_arm_mujoco_env(SequenceMixin, MujocoEnvironment):
    def __init__(self, agents_can_rotate=True):
        path = (
            "/home/valentin/Downloads/roboballet/data/mujoco_world/4_pandas_world.xml"
        )

        self.robots = [
            "panda1",
            "panda2",
            "panda3",
            "panda4",
        ]

        self.start_pos = NpConfiguration.from_list(
            [np.array([0, -0.5, 0, -2, 0, 2, -0.5]) for r in self.robots]
        )

        MujocoEnvironment.__init__(self, path)

        self.tasks = [
            Task(["panda1"], SingleGoal(np.array([1, -0.5, 0, -2, 0, 2, -0.5]))),
            Task(["panda2"], SingleGoal(np.array([1, -0.5, 0, -2, 0, 2, -0.5]))),
            Task(["panda3"], SingleGoal(np.array([1, -0.5, 0, -2, 0, 2, -0.5]))),
            Task(["panda4"], SingleGoal(np.array([1, -0.5, 0, -2, 0, 2, -0.5]))),
            # terminal mode
            Task(
                self.robots,
                SingleGoal(self.start_pos.state()),
            ),
        ]

        self.tasks[0].name = "p1_goal"
        self.tasks[1].name = "p2_goal"
        self.tasks[2].name = "p3_goal"
        self.tasks[3].name = "p4_goal"
        self.tasks[4].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            ["p1_goal", "p2_goal", "p3_goal", "p4_goal", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01
