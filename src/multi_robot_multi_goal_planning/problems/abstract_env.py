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

import matplotlib.pyplot as plt


class Sphere:
    def __init__(self, pos, radius):
        self.pos = pos
        self.radius = radius

    def collides_with_sphere(self, center, radius):
        if np.linalg.norm(self.pos - center) < self.radius + radius:
            return True
        return False

    # TODO: check if this is actually true
    def batch_collides_with_sphere(self, centers, radius):
        if (np.linalg.norm(self.pos - centers, axis=1) < self.radius + radius).any():
            return True
        return False


class AbstractEnvironment(BaseProblem):
    def __init__(self):
        self.limits = None
        self.agent_radii = None
        self.start_pos = None

        self.obstacles = []

    def get_scenegraph_info_for_mode(self, mode: Mode):
        return {}

    def show(self):
        plt.figure()
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.axis("equal")

        for o in self.obstacles:
            circle = plt.Circle(o.pos, o.radius, color="black")
            plt.gca().add_artist(circle)

        # get the tabular color rotation
        colors = plt.cm.tab20.colors
        for i in range(self.start_pos.num_agents()):
            circle = plt.Circle(self.start_pos[i], self.agent_radii[i], color=colors[i%len(colors)])
            plt.gca().add_artist(circle)

        plt.show()

    def show_config(self, q):
        plt.figure()
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.axis("equal")

        for o in self.obstacles:
            circle = plt.Circle(o.pos, o.radius, color="black")
            plt.gca().add_artist(circle)

        # get the tabular color rotation
        colors = plt.cm.tab20.colors
        for i in range(self.start_pos.num_agents()):
            circle = plt.Circle(self.start_pos[i], self.agent_radii[i], color=colors[i%len(colors)])
            plt.gca().add_artist(circle)

        plt.show()

    def config_cost(self, start: Configuration, end: Configuration) -> float:
        return config_cost(start, end, self.cost_metric, self.cost_reduction)

    def batch_config_cost(
        self, starts: List[Configuration], ends: List[Configuration]
    ) -> NDArray:
        return batch_config_cost(starts, ends, self.cost_metric, self.cost_reduction)

    def is_collision_free(self, q: Optional[Configuration], mode: List[int]):
        if q is None:
            raise ValueError

        num_agents = q.num_agents()

        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                # check collision of the two agents againt each other
                if (
                    np.linalg.norm(q[i] - q[j])
                    < self.agent_radii[i] + self.agent_radii[j]
                ):
                    return False

        for i in range(num_agents):
            pos = q[i]
            for o in self.obstacles:
                if o.collides_with_sphere(pos, self.agent_radii[i]):
                    return False

        return True

    def batch_is_collision_free(self, qs: List[Configuration], mode: List[int]):
        num_agents = qs[0].num_agents()
        for i in range(num_agents):
            positions = [qs[j][i] for j in range(len(qs))]
            for o in self.obstacles:
                if o.batch_collides_with_sphere(
                    np.array(positions), self.agent_radii[i]
                ):
                    return False

        # TODO: this can probably be batched better
        for k in range(len(qs)):
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    # check collision of the two spheres
                    if (
                        np.linalg.norm(qs[k][i] - qs[k][j])
                        < self.agent_radii[i] + self.agent_radii[j]
                    ):
                        return False

        return True

    def is_edge_collision_free(
        self,
        q1: Configuration,
        q2: Configuration,
        mode: List[int],
        resolution: float = 0.1,
        randomize_order: bool = True,
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

        is_in_collision = self.batch_is_collision_free(qs, mode)

        if not is_in_collision:
            # print('coll')
            return False

        return True

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

            if not self.is_edge_collision_free(
                q1, q2, mode, resolution=resolution
            ):
                return False

        return True

    def set_to_mode(self, m: List[int]):
        return

        raise NotImplementedError("This is not supported for this environment.")


def make_middle_obstacle_n_dim_env(dim=2):
    num_agents = 2

    joint_limits = np.ones((2, num_agents * dim)) * 2
    joint_limits[0, :] = -2

    print(joint_limits)

    start_poses = np.zeros(num_agents * dim)
    start_poses[0] = -0.8
    start_poses[dim] = 0.8

    sphere_obs = Sphere(np.zeros(dim), 0.2)
    obstacles = [sphere_obs]
    # obstacles = []

    return start_poses, joint_limits, obstacles


class abstract_env_two_dim_middle_obs(SequenceMixin, AbstractEnvironment):
    def __init__(self, agents_can_rotate=True):
        AbstractEnvironment.__init__(self)

        _, self.limits, self.obstacles = make_middle_obstacle_n_dim_env()

        self.start_pos = NpConfiguration.from_list([[-0.8, 0], [0.8, 0]])

        self.agent_radii = [0.1, 0.1]

        self.robot_idx = {"a1": [0, 1], "a2": [2, 3]}
        self.robot_dims = {"a1": 2, "a2": 2}
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        self.tasks = [
            # r1
            Task(["a1"], SingleGoal(np.array([-0.8, 0]))),
            # r2
            Task(["a2"], SingleGoal(np.array([0.8, 0]))),
            # terminal mode
            Task(
                ["a1", "a2"],
                SingleGoal(np.array([0.8, 0, -0.8, 0])),
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


def make_random_rectangles_nd(dims, num_agents, num_goals, num_obstacles):
    pass
