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

import time


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

    def get_artists(self):
        return plt.Circle(self.pos, self.radius, color="black")


class Rectangle:
    def __init__(self, center, bounds):
        self.center = center
        self.bounds = bounds

        self.min_bounds = self.center - self.bounds / 2
        self.max_bounds = self.center + self.bounds / 2

    def collides_with_sphere(self, center, radius):
        center = np.array(center, dtype=float)

        # Verify the sphere center has the same dimensions
        if center.shape != self.center.shape:
            raise ValueError(
                "Sphere center must have the same dimensionality as the rectangle"
            )

        # Calculate the closest point using vectorized operations
        closest_point = np.clip(center, self.min_bounds, self.max_bounds)

        # Calculate squared distance efficiently
        distance_squared = np.sum((closest_point - center) ** 2)

        return distance_squared <= radius**2

    def batch_collides_with_sphere(self, centers, radius):
        centers = np.array(centers, dtype=float)

        # Handle single center case
        if centers.ndim == 1:
            centers = centers.reshape(1, -1)

        # Verify dimensions match
        if centers.shape[1] != len(self.center):
            raise ValueError(f"Sphere centers must have {len(self.center)} dimensions")

        # Broadcasting to find closest points for all centers at once
        closest_points = np.clip(centers, self.min_bounds, self.max_bounds)

        # Calculate all distances squared at once
        distances_squared = np.sum((closest_points - centers) ** 2, axis=1)

        # Compare with radius squared
        return (distances_squared <= radius**2).any()

    def get_artists(self):
        return plt.Rectangle(
            self.min_bounds, self.bounds[0], self.bounds[1], color="black"
        )


class AbstractEnvironment(BaseProblem):
    def __init__(self):
        self.limits = None
        self.agent_radii = None
        self.start_pos = None

        self.obstacles = []

        self.fig = None
        self.ax = None
        self.key_pressed = False

        self.cost_metric = "euclidean"
        self.cost_reduction = "max"

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
        q = NpConfiguration(rnd, self.start_pos.array_slice)

        return q

    def get_scenegraph_info_for_mode(self, mode: Mode):
        return {}

    def show(self, blocking=True):
        if len(self.start_pos[0]) > 2:
            return

        plt.figure()
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.axis("equal")

        for o in self.obstacles:
            artist = o.get_artists()
            plt.gca().add_artist(artist)

        # get the tabular color rotation
        colors = plt.cm.tab20.colors
        for i in range(self.start_pos.num_agents()):
            circle = plt.Circle(
                self.start_pos[i], self.agent_radii[i], color=colors[i % len(colors)]
            )
            plt.gca().add_artist(circle)

        plt.show()

    def show_config(self, q, blocking=True):
        if len(self.start_pos[0]) > 2:
            return

        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()
            # Set up the key press event handler only once
            self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Clear the previous plot but keep the figure
        self.ax.clear()
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect("equal")

        for o in self.obstacles:
            artist = o.get_artists()
            plt.gca().add_artist(artist)

        # get the tabular color rotation
        colors = plt.cm.tab20.colors
        for i in range(self.start_pos.num_agents()):
            circle = plt.Circle(
                q[i], self.agent_radii[i], color=colors[i % len(colors)]
            )
            plt.gca().add_artist(circle)

        # Add instruction text
        if len(self.fig.texts) > 0:
            self.fig.texts[0].set_text("Press any key to continue")
        else:
            self.fig.text(0.5, 0.01, "Press any key to continue", ha="center")

        # Reset the key_pressed flag
        self.key_pressed = False

        if not blocking:
            self.key_pressed = True

        # Show the plot and make it interactive
        plt.draw()
        plt.pause(0.001)  # Small pause to ensure the window appears

        # Block until a key is pressed
        while not self.key_pressed:
            plt.pause(0.1)  # This keeps the GUI responsive

    def _on_key(self, event):
        if event.key is not None:
            self.key_pressed = True

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

    def _batch_is_collision_free(self, qs: List[Configuration], mode: List[int]):
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
        mode: Mode,
        resolution: float = None,
        tolerance: float = None,
        include_endpoints: bool = False,
        N_start: int = 0,
        N_max: int = None,
    ) -> bool:
        if resolution is None:
            resolution = self.collision_resolution

        if tolerance is None:
            tolerance = self.collision_tolerance

        # print('q1', q1)
        # print('q2', q2)
        N = int(config_dist(q1, q2, "max") / resolution)
        N = max(2, N)

        if N_start > N:
            return None
        
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
            q = NpConfiguration(q, q1.array_slice)

        
            if not self.is_collision_free(q, mode):
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

            if not self.is_edge_collision_free(q1, q2, mode, resolution=resolution):
                return False

        return True

    def set_to_mode(self, m: List[int]):
        return

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
            Task(["a1"], SingleGoal(np.array([0.8, 0]))),
            # r2
            Task(["a2"], SingleGoal(np.array([-0.8, 0]))),
            # terminal mode
            Task(
                ["a1", "a2"],
                SingleGoal(np.array([-0.8, 0, 0.8, 0])),
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


def make_center_rectangle_nd(dim, num_agents=2):
    joint_limits = np.ones((2, num_agents * dim)) * 2
    joint_limits[0, :] = -2

    start_poses = np.zeros(num_agents * dim)
    start_poses[0] = -0.8
    start_poses[dim] = 0.8

    rect_obs = Rectangle(np.zeros(dim), np.ones(dim) * 0.5)

    obstacles = [rect_obs]

    return start_poses, joint_limits, obstacles


class abstract_env_center_rect_nd(SequenceMixin, AbstractEnvironment):
    def __init__(self, n=2):
        AbstractEnvironment.__init__(self)

        _, self.limits, self.obstacles = make_center_rectangle_nd(n)

        r1_start = np.zeros(n)
        r1_start[0] = -0.8
        r2_start = np.zeros(n)
        r2_start[0] = 0.8

        self.start_pos = NpConfiguration.from_list([r1_start, r2_start])

        self.agent_radii = [0.1, 0.1]

        self.robot_idx = {"a1": [i for i in range(n)], "a2": [n + i for i in range(n)]}
        self.robot_dims = {"a1": n, "a2": n}
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        self.tasks = [
            # r1
            Task(["a1"], SingleGoal(np.array(r2_start))),
            # r2
            Task(["a2"], SingleGoal(np.array(r1_start))),
            # terminal mode
            Task(
                ["a1", "a2"],
                SingleGoal(self.start_pos.state()),
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
