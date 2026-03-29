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
        self.limits = None
        self.start_pos = None

        self.env = None

        self.cost_metric = "euclidean"
        self.cost_reduction = "max"

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
        # for i in range(len(path)):
        #     self.show_config(path[i].q, stop)

        #     # if export:
        #     #     self.C.view_savePng("./z.vid/")

        #     dt = pause_time
        #     if adapt_to_max_distance:
        #         if i < len(path) - 1:
        #             v = 5
        #             diff = config_dist(path[i].q, path[i + 1].q, "max_euclidean")
        #             dt = diff / v

        #     time.sleep(dt)

        # if stop_at_end:
        #     self.show_config(path[-1].q, True)

    def sample_config_uniform_in_limits(self):
        rnd = np.random.uniform(low=self.limits[0, :], high=self.limits[1, :])
        q = self.start_pos.from_flat(rnd)

        return q

    def get_scenegraph_info_for_mode(self, mode: Mode, is_start_mode: bool = False):
        return {}

    def show(self, blocking=True):
        self.env.update_scene()

    def show_config(self, q, blocking=True):
        self.env.set_joint_positions(q.as_list()) 
        # self.env.update_scene() # no need for this as it happens automatically in the set_joint_posisitons call

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

        self.env.set_joint_positions(q.as_list()) 
        return not self.env.in_collision_robot(robot_id, q[robot_id], self_only=False)

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

        return not self.env.motion_in_collision(q1.as_list(), q2.as_list(), step_size=resolution, self_only=False)

    def set_to_mode(self, m: List[int]):
        if not self.manipulating_env:
            return
        else:
            raise NotImplementedError("This is not supported for this environment.")


def make_middle_obstacle_n_dim_env(dim=2):
    num_agents = 2

    joint_limits = np.ones((2, num_agents * dim)) * 2
    joint_limits[0, :] = -2

    start_poses = np.zeros(num_agents * dim)
    start_poses[0] = -0.8
    start_poses[dim] = 0.8

    sphere_obs = Sphere(np.zeros(dim), 0.2)
    rect_obs = Rectangle(np.array([0, 0.4]), np.array([0.5, 0.5]))

    obstacles = [sphere_obs, rect_obs]

    return start_poses, joint_limits, obstacles


@register("vampmr.test")
class vamp_test_env(SequenceMixin, VampEnv):
    def __init__(self, agents_can_rotate=True):
        pass

@register("vampmr.quad_panda")
class vamp_quad_panda_env(SequenceMixin, VampEnv):
    def __init__(self, agents_can_rotate=True):
        VampEnv.__init__(self)

        self.manipulating_env = False

        self.env = mr_planner_core.VampEnvironment("panda_four")
        info = self.env.info()
        num_robots = int(info["num_robots"])

        # self.env.enable_meshcat()
        # self.env.reset_scene()

        panda_dim = 7
        self.limits = np.array([[-2] * 4 * panda_dim, [2] * 4 * panda_dim])

        self.start_pos = NpConfiguration.from_list([[0] * panda_dim, [0] * panda_dim, [0] * panda_dim, [0] * panda_dim])

        self.robot_dims = {"a1": panda_dim, "a2": panda_dim, "a3": panda_dim, "a4": panda_dim}
        self.robot_idx = {f"a{i+1}": list(range(i * panda_dim, (i + 1) * panda_dim)) for i in range(num_robots)}
        # self.C.view(True)

        self.robots = ["a1", "a2", "a3", "a4"]

        goal_pos = self.start_pos.q

        self.tasks = [
            # r1
            Task("a1_goal", ["a1"], SingleGoal(np.array([0.0, 1, 1, 0, -2, 0, 0]))),
            # r2
            Task("a2_goal", ["a2"], SingleGoal(np.array([-0.0, 0, 1, 0, 2, 0, 0]))),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(goal_pos),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a2_goal", "a1_goal", "terminal"]
        )

        self.collision_tolerance = 0.01

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        for r in self.robots:
            self.safe_pose[r] = np.array([0] * panda_dim)