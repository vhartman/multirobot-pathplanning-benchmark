import numpy as np

from typing import List, Dict, Optional
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.util import generate_binary_search_indices

from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
    NpConfiguration,
    config_dist,
    config_cost,
    batch_config_cost,
)

from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseModeLogic,
    SequenceMixin,
    DependencyGraphMixin,
    State,
    Mode,
    Task,
    SingleGoal,
    GoalSet,
    GoalRegion,
    ConditionalGoal,
    BaseProblem,
)

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

import time
import sys


class PinocchioEnvironment(BaseProblem):
    def __init__(self):
        self.limits = None
        self.start_pos = None

        self.model = None
        self.collision_model = None
        self.visual_model = None

        self.data = None
        self.geom_data = None

        self.viz = None

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

            if export:
                print("Exporting is not supported in pinocchio envs.")

            dt = pause_time
            if adapt_to_max_distance:
                if i < len(path) - 1:
                    v = 5
                    diff = config_dist(path[i].q, path[i + 1].q, "max_euclidean")
                    dt = diff / v

            time.sleep(dt)

            if stop:
                input("Press Enter to continue...")

        if stop_at_end:
            self.show_config(path[-1].q, True)

    def get_scenegraph_info_for_mode(self, mode: Mode):
        return {}

    def show(self, blocking=True):
        self.viz.display(self.start_pos.state())

        if blocking:
            input("Press Enter to continue...")

    def show_config(self, q, blocking=True):
        self.viz.display(q.state())

        if blocking:
            input("Press Enter to continue...")

    def config_cost(self, start: Configuration, end: Configuration) -> float:
        return config_cost(start, end, self.cost_metric, self.cost_reduction)

    def batch_config_cost(
        self, starts: List[Configuration], ends: List[Configuration]
    ) -> NDArray:
        return batch_config_cost(starts, ends, self.cost_metric, self.cost_reduction)

    def is_collision_free(self, q: Optional[Configuration], mode: List[int]):
        if q is None:
            raise ValueError

        in_collision = pin.computeCollisions(
            self.model, self.data, self.collision_model, self.geom_data, q.state(), True
        )

        if not in_collision:
            return True

        return False

    def is_edge_collision_free(
        self,
        q1: Configuration,
        q2: Configuration,
        mode: Mode,
        resolution: float = None,
        randomize_order: bool = True,
        tolerance: float = None,
    ):
        N = config_dist(q1, q2) / resolution
        N = max(5, N)

        idx = list(range(int(N)))
        if randomize_order:
            # np.random.shuffle(idx)
            idx = generate_binary_search_indices(int(N)).copy()

        qs = []

        for i in idx:
            # print(i / (N-1))
            q = q1.state() + (q2.state() - q1.state()) * (i) / (N - 1)
            q = NpConfiguration(q, q1.array_slice)
            qs.append(q)

        # is_in_collision = self.batch_is_collision_free(qs, mode)
        is_collision_free = True

        for q in qs:
            if not self.is_collision_free(q, mode):
                is_collision_free = False
                break

        if is_collision_free:
            # print('coll')
            return True

        return False

    def is_path_collision_free(
        self, path: List[State], randomize_order=True, resolution=None, tolerance=None
    ) -> bool:
        if tolerance is None:
            tolerance = self.collision_tolerance

        if resolution is None:
            resolution = self.collision_resolution

        idx = list(range(len(path) - 1))
        if randomize_order:
            np.random.shuffle(idx)

        for i in idx:
            # skip transition nodes
            # if path[i].mode != path[i + 1].mode:
            #     continue

            q1 = path[i].q
            q2 = path[i + 1].q
            mode = path[i].mode

            if not self.is_edge_collision_free(q1, q2, mode, resolution=resolution):
                return False

        return True

    def set_to_mode(self, m: List[int]):
        return

        raise NotImplementedError("This is not supported for this environment.")


def make_pinocchio_random_env(dim=2):
    num_agents = 2

    joint_limits = np.ones((2, num_agents * dim)) * 2
    joint_limits[0, :] = -2

    print(joint_limits)

    start_poses = np.zeros(num_agents * dim)
    start_poses[0] = -0.8
    start_poses[dim] = 0.8

    return None


def make_pin_middle_obstacle_two_dim_env():
    filename = "./middle_obstacle_two_agents.urdf"
    # filename = "./2d_handover.urdf"
    # filename = "./random_2d.urdf"
    # filename = "/home/valentin/git/postdoc/pin-venv/hello-world/ur5e_2nd_try/ur5e.urdf"
    model, collision_model, visual_model = pin.buildModelsFromUrdf(filename)

    collision_model.addAllCollisionPairs()

    return model, collision_model, visual_model


class pinocchio_middle_obs(SequenceMixin, PinocchioEnvironment):
    def __init__(self, agents_can_rotate=True):
        PinocchioEnvironment.__init__(self)

        self.model, self.collision_model, self.visual_model = (
            make_pin_middle_obstacle_two_dim_env()
        )

        self.limits = np.ones((2, 2 * 3)) * 1
        self.limits[0, :] = -1

        self.data = self.model.createData()
        self.geom_data = pin.GeometryData(self.collision_model)

        self.viz = MeshcatVisualizer(
            self.model, self.collision_model, self.visual_model
        )

        try:
            self.viz.initViewer(open=True)
        except ImportError as err:
            print(
                "Error while initializing the viewer. It seems you should install Python meshcat"
            )
            print(err)
            sys.exit(0)

        # Load the robot in the viewer.
        self.viz.loadViewerModel()

        # Display a robot configuration.
        self.viz.displayVisuals(True)
        self.viz.displayCollisions(True)

        self.start_pos = NpConfiguration.from_list([[0, -0.8, 0], [0, 0.8, 0]])

        self.robot_idx = {"a1": [0, 1, 2], "a2": [3, 4, 5]}
        self.robot_dims = {"a1": 3, "a2": 3}
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        self.tasks = [
            # r1
            Task(["a1"], SingleGoal(np.array([0, 0.8, 0]))),
            # r2
            Task(["a2"], SingleGoal(np.array([0, -0.8, 0]))),
            # terminal mode
            Task(
                ["a1", "a2"],
                SingleGoal(np.array([0, -0.8, 0, 0, 0.8, 0])),
            ),
        ]

        self.tasks[0].name = "a1_goal"
        self.tasks[1].name = "a2_goal"
        self.tasks[2].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            ["a2_goal", "a1_goal", "terminal"]
        )

        self.collision_tolerance = 0.01

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01


def make_wall_gap_two_dim():
    pass


def make_pinocchio_hallway():
    pass
