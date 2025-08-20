import robotic as ry
import numpy as np
import random

from typing import List, Dict, Optional
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.dependency_graph import DependencyGraph

import multi_robot_multi_goal_planning.problems.rai_config as rai_config
from multi_robot_multi_goal_planning.problems.configuration import config_dist

# from multi_robot_multi_goal_planning.problems.rai_config import *
from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseModeLogic,
    SequenceMixin,
    DependencyGraphMixin,
    State,
    Task,
    SingleGoal,
    GoalSet,
    GoalRegion,
    ConditionalGoal,
    ProblemSpec,
    AgentType,
    GoalType,
    ConstraintType,
    DynamicsType,
    ManipulationType,
    DependencyType,
    SafePoseType,
)
from multi_robot_multi_goal_planning.problems.rai_base_env import rai_env

# In the follwoing, we want to test planners on a variety of tings
# In particular, we first want to establish a few simple problems
# that can be used for debugging, and can be visualized easily
# and later increasingly more complex problems up until we arrive at the
# full manipulation planning problem
#
# The things that we can influence to make a problem more complex (or easier)
# are
# - complexity of the environment
# - number of agents
# - dimensionality of agents
# - manipulation of the environment/no manipulation
# - number of tasks/goals/horizon length
# - single goal state vs goal-neighbourhood vs constraint
# - kinodynamic planning vs not vs constraint on the motion
# - fully specified sequence vs per robot sequence vs dependency graph
#
# We will vary these things like so
# 2D Envs
# - environment
# -- Empty env
# -- dividing wall(s)
# -- goal enclosure
# -- repeating rectangles
# - agents
# -- xy
# -- xyphi
# -- heterogenous agents/sizes etc
# - goals
# -- rearrangement of agents
# -- agents going through hole and back
# - manip/no manip
#
# robot arm envs
# - single/dual/triple arm
# - waypoint scene
# - inspection scene
# - spot welding setting
# - pick and place setting
#
# mobile manipulation setting
# - wall stacking
# - cards?

# 2d inspection?


##############################
# 2 dimensional environments #
##############################


class rai_two_dim_env(SequenceMixin, rai_env):
    def __init__(self, agents_can_rotate=True):
        self.C, keyframes = rai_config.make_2d_rai_env(
            agents_can_rotate=agents_can_rotate
        )
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        self.tasks = [
            # r1
            Task(["a1"], SingleGoal(keyframes[0][self.robot_idx["a1"]])),
            # r2
            Task(["a2"], SingleGoal(keyframes[1][self.robot_idx["a2"]])),
            # terminal mode
            Task(
                ["a1", "a2"],
                SingleGoal(
                    np.concatenate(
                        [
                            keyframes[0][self.robot_idx["a1"]],
                            keyframes[2][self.robot_idx["a2"]],
                        ]
                    )
                ),
            ),
        ]

        self.tasks[0].name = "a1_goal"
        self.tasks[1].name = "a2_goal"
        self.tasks[2].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            ["a2_goal", "a1_goal", "terminal"]
        )

        self.collision_tolerance = 0.01

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_NO_SAFE_HOME_POSE
        self.spec.manipulation = ManipulationType.STATIC


# very simple task:
# make the robots go back and forth.
# should be trivial for decoupled methods, hard for joint methods that sample partial goals
# Optimal cost is be: 5.1 (no matter if rotationis enabled or not)
class rai_two_dim_env_no_obs(SequenceMixin, rai_env):
    def __init__(self, agents_can_rotate=True):
        self.C = rai_config.make_2d_rai_env_no_obs(agents_can_rotate=agents_can_rotate)
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        # r1 starts at both negative
        r1_state = self.C.getJointState()[self.robot_idx["a1"]]
        # r2 starts at both positive
        r2_state = self.C.getJointState()[self.robot_idx["a2"]]

        r1_goal = r1_state * 1.0
        r1_goal[:2] = [-0.5, 0.5]

        r2_goal_1 = r2_state * 1.0
        r2_goal_1[:2] = [0.5, -0.5]
        r2_goal_2 = r2_state * 1.0
        r2_goal_2[:2] = [0.5, 0.5]

        self.tasks = [
            # r1
            Task(["a1"], SingleGoal(r1_goal)),
            # r2
            Task(["a2"], SingleGoal(r2_goal_1)),
            Task(["a2"], SingleGoal(r2_goal_2)),
            Task(["a2"], SingleGoal(r2_goal_1)),
            Task(["a2"], SingleGoal(r2_goal_2)),
            # terminal mode
            Task(
                ["a1", "a2"],
                SingleGoal(self.C.getJointState()),
            ),
        ]

        self.tasks[0].name = "a1_goal"
        self.tasks[1].name = "a2_goal_0"
        self.tasks[2].name = "a2_goal_1"
        self.tasks[3].name = "a2_goal_2"
        self.tasks[4].name = "a2_goal_3"
        self.tasks[5].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            ["a2_goal_0", "a2_goal_1", "a2_goal_2", "a2_goal_3", "a1_goal", "terminal"]
        )

        self.collision_tolerance = 0.001

        BaseModeLogic.__init__(self)

        self.spec.manipulation = ManipulationType.STATIC
        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE


# for the case of the dependency graph, the optimal solution should be 4.1
class rai_two_dim_env_no_obs_dep_graph(DependencyGraphMixin, rai_env):
    def __init__(self, agents_can_rotate=True):
        self.C = rai_config.make_2d_rai_env_no_obs(agents_can_rotate=agents_can_rotate)
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        # r1 starts at both negative
        r1_state = self.C.getJointState()[self.robot_idx["a1"]]
        # r2 starts at both positive
        r2_state = self.C.getJointState()[self.robot_idx["a2"]]

        r1_goal = r1_state * 1.0
        r1_goal[:2] = [-0.5, 0.5]

        r2_goal_1 = r2_state * 1.0
        r2_goal_1[:2] = [0.5, -0.5]
        r2_goal_2 = r2_state * 1.0
        r2_goal_2[:2] = [0.5, 0.5]

        self.tasks = [
            # r1
            Task(["a1"], SingleGoal(r1_goal)),
            # r2
            Task(["a2"], SingleGoal(r2_goal_1)),
            Task(["a2"], SingleGoal(r2_goal_2)),
            Task(["a2"], SingleGoal(r2_goal_1)),
            Task(["a2"], SingleGoal(r2_goal_2)),
            # terminal mode
            Task(
                ["a1", "a2"],
                SingleGoal(self.C.getJointState()),
            ),
        ]

        self.tasks[0].name = "a1_goal"
        self.tasks[1].name = "a2_goal_0"
        self.tasks[2].name = "a2_goal_1"
        self.tasks[3].name = "a2_goal_2"
        self.tasks[4].name = "a2_goal_3"
        self.tasks[5].name = "terminal"

        self.graph = DependencyGraph()
        self.graph.add_dependency("a2_goal_1", "a2_goal_0")
        self.graph.add_dependency("a2_goal_2", "a2_goal_1")
        self.graph.add_dependency("a2_goal_3", "a2_goal_2")

        self.graph.add_dependency("terminal", "a1_goal")
        self.graph.add_dependency("terminal", "a2_goal_3")

        print(self.graph)

        self.collision_tolerance = 0.001

        BaseModeLogic.__init__(self)

        self.spec.dependency = DependencyType.UNORDERED
        self.spec.manipulation = ManipulationType.STATIC


# trivial environment for planing
# challenging to get the optimal solution dpeending on the approach
# optimal solution is 5.15 (independent of rotation or not)
class rai_two_dim_env_no_obs_three_agents(SequenceMixin, rai_env):
    def __init__(self, agents_can_rotate=True):
        self.C = rai_config.make_2d_rai_env_no_obs_three_agents(
            agents_can_rotate=agents_can_rotate
        )
        # self.C.view(True)

        self.robots = ["a1", "a2", "a3"]

        rai_env.__init__(self)

        # r1 starts at both negative
        r1_state = self.C.getJointState()[self.robot_idx["a1"]]

        # r2 starts at both positive
        r2_state = self.C.getJointState()[self.robot_idx["a2"]]

        r3_state = self.C.getJointState()[self.robot_idx["a3"]]

        r1_goal = r1_state * 1.0
        r1_goal[:2] = [-0.5, 0.5]

        r2_goal_1 = r2_state * 1.0
        r2_goal_1[:2] = [0.5, -0.5]
        r2_goal_2 = r2_state * 1.0
        r2_goal_2[:2] = [0.5, 0.5]

        r3_goal = r3_state * 1.0
        r3_goal[:2] = [0.0, -0.5]

        self.tasks = [
            # r1
            Task(["a1"], SingleGoal(r1_goal)),
            # r2
            Task(["a2"], SingleGoal(r2_goal_1)),
            Task(["a2"], SingleGoal(r2_goal_2)),
            Task(["a2"], SingleGoal(r2_goal_1)),
            Task(["a2"], SingleGoal(r2_goal_2)),
            # r3
            Task(["a3"], SingleGoal(r3_goal)),
            # terminal mode
            Task(
                ["a1", "a2", "a3"],
                SingleGoal(self.C.getJointState()),
            ),
        ]

        self.tasks[0].name = "a1_goal"
        self.tasks[1].name = "a2_goal_0"
        self.tasks[2].name = "a2_goal_1"
        self.tasks[3].name = "a2_goal_2"
        self.tasks[4].name = "a2_goal_3"
        self.tasks[5].name = "a3_goal"
        self.tasks[6].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            [
                "a2_goal_0",
                "a2_goal_1",
                "a2_goal_2",
                "a2_goal_3",
                "a1_goal",
                "a3_goal",
                "terminal",
            ]
        )

        self.collision_tolerance = 0.01
        BaseModeLogic.__init__(self)

        self.spec.manipulation = ManipulationType.STATIC


class rai_two_dim_single_agent_neighbourhood(SequenceMixin, rai_env):
    def __init__(self):
        self.C, keyframes = rai_config.make_single_agent_mover_env(
            num_goals=50, view=False
        )
        # self.C.view(True)

        self.robots = ["a1"]

        rai_env.__init__(self)

        self.manipulating_env = True

        pick_task = Task(
            ["a1"],
            GoalSet([k[0] for k in keyframes]),
            type="pick",
            frames=["a1", "obj1"],
        )

        place_task = Task(
            ["a1"],
            ConditionalGoal([k[0] for k in keyframes], [k[1] for k in keyframes]),
            type="place",
            frames=["table", "obj1"],
        )

        terminal_task = Task(
            ["a1"],
            SingleGoal(
                keyframes[0][2],
            ),
        )

        self.tasks = [pick_task, place_task, terminal_task]

        self.tasks[0].name = "a1_pick"
        self.tasks[1].name = "a1_place"
        self.tasks[2].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            ["a1_pick", "a1_place", "terminal"]
        )

        self.collision_tolerance = 0.01

        BaseModeLogic.__init__(self)

        self.prev_mode = self.start_mode

        self.spec.manipulation = ManipulationType.STATIC


# best max-cost sol: 3.41
# best sum-cost sol: 5.922
class rai_two_dim_simple_manip(SequenceMixin, rai_env):
    def __init__(self):
        self.C, keyframes = rai_config.make_piano_mover_env()
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        self.manipulating_env = True

        self.tasks = [
            # a1
            Task(
                ["a1"],
                SingleGoal(keyframes[0][self.robot_idx["a1"]]),
                type="pick",
                frames=["a1", "obj1"],
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[1][self.robot_idx["a1"]]),
                type="place",
                frames=["table", "obj1"],
            ),
            # a2
            Task(
                ["a2"],
                SingleGoal(keyframes[0][self.robot_idx["a2"]]),
                type="pick",
                frames=["a2", "obj2"],
            ),
            Task(
                ["a2"],
                SingleGoal(keyframes[1][self.robot_idx["a2"]]),
                type="place",
                frames=["table", "obj2"],
            ),
            # terminal
            Task(
                ["a1", "a2"],
                SingleGoal(
                    np.concatenate(
                        [
                            keyframes[2][self.robot_idx["a1"]],
                            keyframes[2][self.robot_idx["a2"]],
                        ]
                    )
                ),
            ),
        ]

        # for t in self.tasks:
        #     arr = t.goal.sample(None)
        #     print(np.array2string(np.array(arr), separator=", "))

        self.tasks[0].name = "a1_pick"
        self.tasks[1].name = "a1_place"
        self.tasks[2].name = "a2_pick"
        self.tasks[3].name = "a2_place"
        self.tasks[4].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            ["a2_pick", "a1_pick", "a2_place", "a1_place", "terminal"]
        )
        # self.sequence = [2, 0, 3, 1, 4]

        self.collision_tolerance = 0.01

        BaseModeLogic.__init__(self)

        self.prev_mode = self.start_mode

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {
            "a1": np.array(keyframes[0][self.robot_idx["a1"]]),
            "a2": np.array(keyframes[1][self.robot_idx["a2"]])
        }


class rai_two_dim_simple_manip_dependency_graph(DependencyGraphMixin, rai_env):
    def __init__(self):
        self.C, keyframes = rai_config.make_piano_mover_env()
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        self.manipulating_env = True

        self.tasks = [
            # a1
            Task(
                ["a1"],
                SingleGoal(keyframes[0][self.robot_idx["a1"]]),
                type="pick",
                frames=["a1", "obj1"],
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[1][self.robot_idx["a1"]]),
                type="place",
                frames=["table", "obj1"],
            ),
            # a2
            Task(
                ["a2"],
                SingleGoal(keyframes[0][self.robot_idx["a2"]]),
                type="pick",
                frames=["a2", "obj2"],
            ),
            Task(
                ["a2"],
                SingleGoal(keyframes[1][self.robot_idx["a2"]]),
                type="place",
                frames=["table", "obj2"],
            ),
            # terminal
            Task(
                ["a1", "a2"],
                SingleGoal(
                    np.concatenate(
                        [
                            keyframes[2][self.robot_idx["a1"]],
                            keyframes[2][self.robot_idx["a2"]],
                        ]
                    )
                ),
            ),
        ]

        self.tasks[0].name = "a1_pick"
        self.tasks[1].name = "a1_place"
        self.tasks[2].name = "a2_pick"
        self.tasks[3].name = "a2_place"
        self.tasks[4].name = "terminal"

        self.graph = DependencyGraph()
        self.graph.add_dependency("a1_place", "a1_pick")
        self.graph.add_dependency("a2_place", "a2_pick")

        self.graph.add_dependency("terminal", "a1_place")
        self.graph.add_dependency("terminal", "a2_place")

        print(self.graph)

        BaseModeLogic.__init__(self)

        self.collision_tolerance = 0.01

        self.prev_mode = self.start_mode

        self.spec.dependency = DependencyType.UNORDERED
        
        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {
            "a1": np.array(keyframes[0][self.robot_idx["a1"]]),
            "a2": np.array(keyframes[1][self.robot_idx["a2"]])
        }


# best cost found for max-cost is 17.64
# best cost found for sum-cost is 25.28
class rai_two_dim_handover(SequenceMixin, rai_env):
    def __init__(self):
        self.C, keyframes = rai_config.make_two_dim_handover()
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        self.manipulating_env = True

        translated_handover_poses = []
        for _ in range(100):
            new_pose = keyframes[1] * 1.0
            translation = np.random.rand(2) * 1 - 0.5
            new_pose[0:2] += translation
            new_pose[3:5] += translation

            translated_handover_poses.append(new_pose)

        translated_handover_poses.append(keyframes[1])

        # generate set of random translations of the original keyframe
        rotated_terminal_poses = []
        for _ in range(100):
            new_pose = keyframes[3] * 1.0
            rot = np.random.rand(2) * 6 - 3
            new_pose[2] = rot[0]
            new_pose[5] = rot[1]

            rotated_terminal_poses.append(new_pose)

        rotated_terminal_poses.append(keyframes[3])

        # self.import_tasks("2d_handover_tasks.txt")

        self.tasks = [
            # a1
            Task(
                ["a1"],
                SingleGoal(keyframes[0][self.robot_idx["a1"]]),
                type="pick",
                frames=["a1", "obj1"],
            ),
            Task(
                ["a1", "a2"],
                GoalSet(translated_handover_poses),
                # SingleGoal(keyframes[1]),
                type="hanover",
                frames=["a2", "obj1"],
            ),
            Task(
                ["a2"],
                SingleGoal(keyframes[2][self.robot_idx["a2"]]),
                type="place",
                frames=["table", "obj1"],
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[4][self.robot_idx["a1"]]),
                type="pick",
                frames=["a1", "obj2"],
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[5][self.robot_idx["a1"]]),
                type="place",
                frames=["table", "obj2"],
            ),
            # terminal
            # Task(["a1", "a2"], SingleGoal(keyframes[3])),
            Task(["a1", "a2"], GoalSet(rotated_terminal_poses)),
        ]

        # for t in self.tasks:
            # print(t.goal.sample(None))

        self.tasks[0].name = "a1_pick_obj1"
        self.tasks[1].name = "handover"
        self.tasks[2].name = "a2_place"
        self.tasks[3].name = "a1_pick_obj2"
        self.tasks[4].name = "a1_place_obj2"
        self.tasks[5].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            [
                "a1_pick_obj1",
                "handover",
                "a1_pick_obj2",
                "a1_place_obj2",
                "a2_place",
                "terminal",
            ]
        )

        # self.export_tasks("2d_handover_tasks.txt")

        BaseModeLogic.__init__(self)

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.01

        self.prev_mode = self.start_mode


class rai_two_dim_handover_dependency_graph(DependencyGraphMixin, rai_env):
    def __init__(self):
        self.C, keyframes = rai_config.make_two_dim_handover()
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        self.manipulating_env = True

        translated_handover_poses = []
        for _ in range(100):
            new_pose = keyframes[1] * 1.0
            translation = np.random.rand(2) * 1 - 0.5
            new_pose[0:2] += translation
            new_pose[3:5] += translation

            translated_handover_poses.append(new_pose)

        translated_handover_poses.append(keyframes[1])

        # generate set of random translations of the original keyframe
        rotated_terminal_poses = []
        for _ in range(100):
            new_pose = keyframes[3] * 1.0
            rot = np.random.rand(2) * 6 - 3
            new_pose[2] = rot[0]
            new_pose[5] = rot[1]

            rotated_terminal_poses.append(new_pose)

        rotated_terminal_poses.append(keyframes[3])

        self.tasks = [
            # a1
            Task(
                ["a1"],
                SingleGoal(keyframes[0][self.robot_idx["a1"]]),
                type="pick",
                frames=["a1", "obj1"],
            ),
            Task(
                ["a1", "a2"],
                GoalSet(translated_handover_poses),
                # SingleGoal(keyframes[1]),
                type="hanover",
                frames=["a2", "obj1"],
            ),
            Task(
                ["a2"],
                SingleGoal(keyframes[2][self.robot_idx["a2"]]),
                type="place",
                frames=["table", "obj1"],
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[4][self.robot_idx["a1"]]),
                type="pick",
                frames=["a1", "obj2"],
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[5][self.robot_idx["a1"]]),
                type="place",
                frames=["table", "obj2"],
            ),
            # terminal
            # Task(["a1", "a2"], SingleGoal(keyframes[3])),
            Task(["a1", "a2"], GoalSet(rotated_terminal_poses)),
        ]

        self.tasks[0].name = "a1_pick_obj1"
        self.tasks[1].name = "a2_handover"
        self.tasks[2].name = "a2_place"
        self.tasks[3].name = "a1_pick_obj2"
        self.tasks[4].name = "a1_place_obj2"
        self.tasks[5].name = "terminal"

        self.graph = DependencyGraph()
        self.graph.add_dependency("a2_handover", "a1_pick_obj1")
        self.graph.add_dependency("a1_pick_obj2", "a2_handover")
        self.graph.add_dependency("a1_place_obj2", "a1_pick_obj2")
        self.graph.add_dependency("terminal", "a1_place_obj2")
        self.graph.add_dependency("a2_place", "a2_handover")
        self.graph.add_dependency("terminal", "a2_place")

        print(self.graph)
        # self.graph.visualize()

        BaseModeLogic.__init__(self)

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.01

        self.prev_mode = self.start_mode

        self.spec.dependency = DependencyType.UNORDERED


# best solution found with sum-cost: 49.48
# best solution found with max-cost: xx
class rai_random_two_dim(SequenceMixin, rai_env):
    def __init__(
        self, num_robots=3, num_goals=4, num_obstacles=10, agents_can_rotate=True
    ):
        self.C, keyframes = rai_config.make_random_two_dim(
            num_agents=num_robots,
            num_goals=num_goals,
            num_obstacles=num_obstacles,
            agents_can_rotate=agents_can_rotate,
        )
        # self.C.view(True)

        self.robots = [f"a{i}" for i in range(num_robots)]

        rai_env.__init__(self)

        self.tasks = []
        self.sequence = []

        print(keyframes)

        cnt = 0
        for r in self.robots:
            for i in range(num_goals):
                self.tasks.append(Task([r], SingleGoal(keyframes[cnt])))
                self.tasks[-1].name = f"goal_{r}_{i}"
                self.sequence.append(cnt)

                cnt += 1

        q_home = self.C.getJointState()
        # self.tasks.append(Task(self.robots, GoalRegion(self.limits)))
        self.tasks.append(Task(self.robots, SingleGoal(q_home)))
        self.tasks[-1].name = "terminal"

        random.shuffle(self.sequence)
        self.sequence.append(len(self.tasks) - 1)

        BaseModeLogic.__init__(self)

        self.collision_tolerance = 0.01

        self.prev_mode = self.start_mode

        self.spec.manipulation = ManipulationType.STATIC


# best solution found with sum-cost: 14.08 (independent of rotation)
# best solution found with max-cost: 9.7 (independent of rotation)
class rai_hallway_two_dim(SequenceMixin, rai_env):
    def __init__(self, agents_can_rotate=True):
        self.C, keyframes = rai_config.make_two_dim_tunnel_env(
            agents_can_rotate=agents_can_rotate
        )
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        self.tasks = []
        self.sequence = []

        self.tasks = [
            Task(["a1"], SingleGoal(keyframes[0])),
            Task(["a2"], SingleGoal(keyframes[1])),
            Task(["a1", "a2"], SingleGoal(keyframes[2])),
        ]

        self.tasks[0].name = "a1_goal_1"
        self.tasks[1].name = "a2_goal_1"
        self.tasks[2].name = "terminal"

        self.sequence = [0, 1, 2]

        BaseModeLogic.__init__(self)

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.005

        self.spec.manipulation = ManipulationType.STATIC


# best solution found with sum-cost: xx (independent of rotation)
# best solution found with max-cost: xx (independent of rotation)
class rai_alternative_hallway_two_dim(SequenceMixin, rai_env):
    def __init__(self, agents_can_rotate=True):
        self.C, keyframes = rai_config.make_two_dim_short_tunnel_env(
            agents_can_rotate=agents_can_rotate
        )
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        self.tasks = []
        self.sequence = []

        self.tasks = [
            Task(["a1"], SingleGoal(keyframes[0])),
            Task(["a2"], SingleGoal(keyframes[1])),
            Task(["a1", "a2"], SingleGoal(keyframes[2])),
        ]

        self.tasks[0].name = "a1_goal_1"
        self.tasks[1].name = "a2_goal_1"
        self.tasks[2].name = "terminal"

        self.sequence = [0, 1, 2]

        BaseModeLogic.__init__(self)

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.1

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE
        self.spec.manipulation = ManipulationType.STATIC

        self.safe_pose = {
            "a1": np.array(keyframes[0]),
            "a2": np.array(keyframes[1])
        }


class rai_hallway_two_dim_dependency_graph(DependencyGraphMixin, rai_env):
    def __init__(self, agents_can_rotate=True):
        self.C, keyframes = rai_config.make_two_dim_tunnel_env(agents_can_rotate)
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        self.tasks = []
        self.sequence = []

        self.tasks = [
            Task(["a1"], SingleGoal(keyframes[0])),
            Task(["a2"], SingleGoal(keyframes[1])),
            Task(["a1", "a2"], SingleGoal(keyframes[2])),
        ]

        self.tasks[0].name = "a1_goal_1"
        self.tasks[1].name = "a2_goal_1"
        self.tasks[2].name = "terminal"

        self.graph = DependencyGraph()
        self.graph.add_dependency("terminal", "a1_goal_1")
        self.graph.add_dependency("terminal", "a2_goal_1")

        print(self.graph)

        BaseModeLogic.__init__(self)

        self.collision_tolerance = 0.01

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE
        self.spec.manipulation = ManipulationType.STATIC
        self.spec.dependency = DependencyType.UNORDERED


class rai_alternative_hallway_two_dim_dependency_graph(DependencyGraphMixin, rai_env):
    def __init__(self, agents_can_rotate=True):
        self.C, keyframes = rai_config.make_two_dim_short_tunnel_env(
            agents_can_rotate=agents_can_rotate
        )
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        self.tasks = []
        self.sequence = []

        self.tasks = [
            Task(["a1"], SingleGoal(keyframes[0])),
            Task(["a2"], SingleGoal(keyframes[1])),
            Task(["a1", "a2"], SingleGoal(keyframes[2])),
        ]

        self.tasks[0].name = "a1_goal_1"
        self.tasks[1].name = "a2_goal_1"
        self.tasks[2].name = "terminal"

        self.graph = DependencyGraph()
        self.graph.add_dependency("terminal", "a1_goal_1")
        self.graph.add_dependency("terminal", "a2_goal_1")

        print(self.graph)

        BaseModeLogic.__init__(self)

        self.collision_tolerance = 0.01

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE
        self.spec.manipulation = ManipulationType.STATIC
        self.spec.dependency = DependencyType.UNORDERED

        self.safe_pose = {
            "a1": np.array(keyframes[0]),
            "a2": np.array(keyframes[1])
        }


# best sum-cost: 12.9
# best max-cost: 6.56
class rai_two_dim_three_agent_env(SequenceMixin, rai_env):
    def __init__(self):
        self.C, keyframes = rai_config.make_2d_rai_env_3_agents()
        # self.C.view(True)

        self.robots = ["a1", "a2", "a3"]

        rai_env.__init__(self)

        self.tasks = [
            # a1
            Task(["a1"], SingleGoal(keyframes[0][self.robot_idx["a1"]])),
            Task(["a1"], SingleGoal(keyframes[4][self.robot_idx["a1"]])),
            # a2
            Task(["a2"], SingleGoal(keyframes[1][self.robot_idx["a2"]])),
            Task(["a2"], SingleGoal(keyframes[3][self.robot_idx["a2"]])),
            # a3
            Task(["a3"], SingleGoal(keyframes[2][self.robot_idx["a3"]])),
            # terminal
            Task(
                ["a1", "a2", "a3"],
                SingleGoal(
                    np.concatenate(
                        [
                            keyframes[5][self.robot_idx["a1"]],
                            keyframes[5][self.robot_idx["a2"]],
                            keyframes[5][self.robot_idx["a3"]],
                        ]
                    )
                ),
            ),
        ]

        self.tasks[0].name = "a1_goal_1"
        self.tasks[1].name = "a1_goal_2"
        self.tasks[2].name = "a2_goal_1"
        self.tasks[3].name = "a2_goal_2"
        self.tasks[4].name = "a3_goal_1"
        self.tasks[5].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            [
                "a1_goal_1",
                "a1_goal_2",
                "a2_goal_1",
                "a2_goal_2",
                "a3_goal_1",
                "terminal",
            ]
        )

        BaseModeLogic.__init__(self)

        self.collision_tolerance = 0.01

        self.spec.manipulation = ManipulationType.STATIC


class rai_two_dim_three_agent_env_dependency_graph(DependencyGraphMixin, rai_env):
    def __init__(self):
        self.C, keyframes = rai_config.make_2d_rai_env_3_agents()
        # self.C.view(True)

        self.robots = ["a1", "a2", "a3"]

        rai_env.__init__(self)

        self.tasks = [
            # a1
            Task(["a1"], SingleGoal(keyframes[0][self.robot_idx["a1"]])),
            Task(["a1"], SingleGoal(keyframes[4][self.robot_idx["a1"]])),
            # a2
            Task(["a2"], SingleGoal(keyframes[1][self.robot_idx["a2"]])),
            Task(["a2"], SingleGoal(keyframes[3][self.robot_idx["a2"]])),
            # a3
            Task(["a3"], SingleGoal(keyframes[2][self.robot_idx["a3"]])),
            # terminal
            Task(
                ["a1", "a2", "a3"],
                SingleGoal(
                    np.concatenate(
                        [
                            keyframes[5][self.robot_idx["a1"]],
                            keyframes[5][self.robot_idx["a2"]],
                            keyframes[5][self.robot_idx["a3"]],
                        ]
                    )
                ),
            ),
        ]

        self.tasks[0].name = "a1_goal_1"
        self.tasks[1].name = "a1_goal_2"
        self.tasks[2].name = "a2_goal_1"
        self.tasks[3].name = "a2_goal_2"
        self.tasks[4].name = "a3_goal_1"
        self.tasks[5].name = "terminal"

        self.graph = DependencyGraph()
        self.graph.add_dependency("a1_goal_2", "a1_goal_1")
        self.graph.add_dependency("terminal", "a1_goal_2")
        self.graph.add_dependency("a2_goal_2", "a2_goal_1")
        self.graph.add_dependency("terminal", "a2_goal_2")
        self.graph.add_dependency("terminal", "a3_goal_1")

        print(self.graph)

        BaseModeLogic.__init__(self)

        print(self.start_mode)
        print(self._terminal_task_ids)

        self.collision_tolerance = 0.01

        self.spec.dependency = DependencyType.UNORDERED
        self.spec.manipulation = ManipulationType.STATIC


##############################
# 3 dimensional environments #
##############################


# single robot neighbourhood
class rai_single_ur10_arm_waypoint_env:
    pass


# sample poses -> difficult since pick influences place pose
class rai_single_ur10_arm_pick_and_place_env:
    pass


# goals are neighbourhoods
class rai_dual_ur10_arm_inspection_env:
    pass


# TODO: this is messy (currrently pick/place without actual manip)
class rai_dual_ur10_arm_env(SequenceMixin, rai_env):
    def __init__(self):
        self.C, self.keyframes = rai_config.make_box_sorting_env()

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        self.tasks = [
            # a1
            Task(["a1"], SingleGoal(self.keyframes[0][self.robot_idx["a1"]])),
            Task(["a1"], SingleGoal(self.keyframes[1][self.robot_idx["a1"]])),
            # a2
            Task(["a2"], SingleGoal(self.keyframes[3][self.robot_idx["a2"]])),
            Task(["a2"], SingleGoal(self.keyframes[4][self.robot_idx["a2"]])),
            # terminal
            Task(
                ["a1", "a2"],
                SingleGoal(
                    np.concatenate(
                        [
                            self.keyframes[2][self.robot_idx["a1"]],
                            self.keyframes[5][self.robot_idx["a2"]],
                        ]
                    )
                ),
            ),
        ]

        self.tasks[0].name = "a1_goal_1"
        self.tasks[1].name = "a1_goal_2"
        self.tasks[2].name = "a2_goal_1"
        self.tasks[3].name = "a2_goal_2"
        self.tasks[4].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            ["a1_goal_1", "a2_goal_1", "a1_goal_2", "a2_goal_2", "terminal"]
        )

        BaseModeLogic.__init__(self)

        self.collision_tolerance = 0.01

        self.spec.manipulation = ManipulationType.STATIC


class rai_multi_panda_arm_waypoint_env(SequenceMixin, rai_env):
    def __init__(
        self, num_robots: int = 3, num_waypoints: int = 6, shuffle_goals: bool = False
    ):
        self.C, keyframes = rai_config.make_panda_waypoint_env(
            num_robots=num_robots, num_waypoints=num_waypoints
        )

        self.robots = ["a0", "a1", "a2"]
        self.robots = self.robots[:num_robots]

        rai_env.__init__(self)

        self.tasks = []

        cnt = 0
        for r in self.robots:
            self.tasks.extend(
                [
                    Task([r], SingleGoal(keyframes[cnt + i][self.robot_idx[r]]))
                    for i in range(num_waypoints)
                ]
            )
            cnt += num_waypoints + 1

        q_home = self.C.getJointState()
        self.tasks.append(Task(["a0", "a1", "a2"], SingleGoal(q_home)))

        self.sequence = []

        for i in range(num_waypoints):
            for j in range(num_robots):
                self.sequence.append(i + j * num_waypoints)

        # permute goals, but only the ones that ware waypoints, not the final configuration
        if shuffle_goals:
            random.shuffle(self.sequence)

        # append terminal task
        self.sequence.append(len(self.tasks) - 1)

        BaseModeLogic.__init__(self)

        self.collision_tolerance = 0.01

        self.spec.manipulation = ManipulationType.STATIC


# goals are poses
class rai_quadruple_ur10_arm_spot_welding_env(SequenceMixin, rai_env):
    def __init__(self, num_robots=4, num_pts: int = 6, shuffle_goals: bool = False):
        self.C, keyframes = rai_config.make_welding_env(
            num_robots=num_robots, view=False, num_pts=num_pts
        )

        self.robots = ["a1", "a2", "a3", "a4"][:num_robots]

        rai_env.__init__(self)

        self.tasks = []

        cnt = 0
        for r in self.robots:
            for _ in range(num_pts):
                self.tasks.append(
                    Task([r], SingleGoal(keyframes[cnt][self.robot_idx[r]]))
                )
                # self.robot_goals[-1].name =
                cnt += 1
            cnt += 1

        q_home = self.C.getJointState()
        self.tasks.append(Task(self.robots, SingleGoal(q_home)))

        self.sequence = []
        for i in range(num_pts):
            for j, r in enumerate(self.robots):
                self.sequence.append(i + j * num_pts)

        # permute goals, but only the ones that ware waypoints, not the final configuration
        if shuffle_goals:
            random.shuffle(self.sequence)

        self.sequence.append(len(self.tasks) - 1)

        BaseModeLogic.__init__(self)

        self.collision_tolerance = 0.01

        self.spec.manipulation = ManipulationType.STATIC

class rai_ur10_arm_egg_carton_env(SequenceMixin, rai_env):
    def __init__(self, num_boxes: int = 9):
        self.C, keyframes = rai_config.make_egg_carton_env(num_boxes)

        self.robots = ["a1_", "a2_"]

        rai_env.__init__(self)

        self.manipulating_env = True

        self.tasks = []

        obj_tasks = []

        for robot_name, v in keyframes.items():
            for t in v:
                for i, task_name in enumerate(["pick", "place"]):
                    sideeffect = None
                    if task_name == "place":
                        frames = ["table", t[0]]
                        sideeffect = "remove"
                    else:
                        frames = [robot_name + "ur_vacuum", t[0]]

                    self.tasks.append(
                        Task(
                            [robot_name],
                            SingleGoal(t[1][i][self.robot_idx[robot_name]]),
                            type=task_name,
                            frames=frames,
                            side_effect=sideeffect,
                        )
                    )
                    name = robot_name + "_" + task_name + t[0]

                    self.tasks[-1].name = name

                obj_tasks.append((self.tasks[-2].name, self.tasks[-1].name))

        self.tasks.append(
            Task(
                ["a1_", "a2_"],
                SingleGoal(self.C.getJointState()),
            ),
        )
        self.tasks[-1].name = "terminal"

        named_sequence = []
        random.shuffle(obj_tasks)

        for t1, t2 in obj_tasks:
            named_sequence.append(t1)
            named_sequence.append(t2)

        named_sequence.append("terminal")

        self.sequence = self._make_sequence_from_names(named_sequence)

        BaseModeLogic.__init__(self)

        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.01

        # q = self.C_base.getJointState()
        # print(self.is_collision_free(q, [0, 0]))

        # self.C_base.view(True)


class rai_ur10_arm_pick_and_place_env(rai_dual_ur10_arm_env):
    def __init__(self):
        super().__init__()

        self.manipulating_env = True

        self.tasks = [
            Task(
                ["a1"],
                SingleGoal(self.keyframes[0][self.robot_idx["a1"]]),
                type="pick",
                frames=["a1_ur_vacuum", "obj100"],
            ),
            Task(
                ["a1"],
                SingleGoal(self.keyframes[1][self.robot_idx["a1"]]),
                type="place",
                frames=["table", "obj100"],
                side_effect="remove",
            ),
            Task(
                ["a2"],
                SingleGoal(self.keyframes[3][self.robot_idx["a2"]]),
                type="pick",
                frames=["a2_ur_vacuum", "obj101"],
            ),
            Task(
                ["a2"],
                SingleGoal(self.keyframes[4][self.robot_idx["a2"]]),
                type="place",
                frames=["table", "obj101"],
                side_effect="remove",
            ),
            Task(
                ["a1", "a2"],
                SingleGoal(
                    np.concatenate(
                        [
                            self.keyframes[2][self.robot_idx["a1"]],
                            self.keyframes[5][self.robot_idx["a2"]],
                        ]
                    )
                ),
            ),
        ]

        self.tasks[0].name = "a1_goal_1"
        self.tasks[1].name = "a1_goal_2"
        self.tasks[2].name = "a2_goal_1"
        self.tasks[3].name = "a2_goal_2"
        self.tasks[4].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            ["a1_goal_1", "a2_goal_1", "a1_goal_2", "a2_goal_2", "terminal"]
        )

        BaseModeLogic.__init__(self)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode


# moving objects from a rolling cage to a 'conveyor'
class rai_ur10_box_sort_env:
    pass


# moving objects from a 'conveyor' to a rolling cage
class rai_ur10_palletizing_env:
    pass


class rai_ur10_strut_env:
    pass


class rai_ur10_arm_shelf_env:
    pass


class rai_ur10_arm_conveyor_env:
    pass


# best max cost: 9.24
class rai_ur10_handover_env(SequenceMixin, rai_env):
    def __init__(self):
        self.C, keyframes = rai_config.make_handover_env()

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        print(self.start_pos.state())

        self.manipulating_env = True

        self.tasks = [
            Task(
                ["a1"],
                SingleGoal(keyframes[0][self.robot_idx["a1"]]),
                type="pick",
                frames=["a1_ur_vacuum", "obj1"],
            ),
            Task(
                ["a1", "a2"],
                SingleGoal(keyframes[1]),
                type="handover",
                frames=["a2_ur_vacuum", "obj1"],
            ),
            Task(
                ["a2"],
                SingleGoal(keyframes[2][self.robot_idx["a2"]]),
                type="place",
                frames=["table", "obj1"],
            ),
            Task(["a1", "a2"], SingleGoal(keyframes[3])),
        ]

        self.tasks[0].name = "a1_pick"
        self.tasks[1].name = "handover"
        self.tasks[2].name = "a2_place"
        self.tasks[3].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            ["a1_pick", "handover", "a2_place", "terminal"]
        )

        BaseModeLogic.__init__(self)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.05

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        dim = 6
        for i, r in enumerate(self.robots):
            print(self.C.getJointState()[0:6])
            self.safe_pose[r] = np.array(self.C.getJointState()[dim*i:dim*(i+1)])


class rai_ur10_arm_bottle_env(SequenceMixin, rai_env):
    def __init__(self, num_bottles=2):
        assert num_bottles in [1,2]

        self.C, keyframes = rai_config.make_bottle_insertion()

        self.robots = ["a0", "a1"]

        rai_env.__init__(self)

        self.manipulating_env = True

        self.tasks = [
            Task(
                ["a0"],
                SingleGoal(keyframes[0][self.robot_idx["a0"]]),
                type="pick",
                frames=["a0_ur_vacuum", "bottle_1"],
            ),
            Task(
                ["a0"],
                SingleGoal(keyframes[1][self.robot_idx["a0"]]),
                type="place",
                frames=["table", "bottle_1"],
            ),
            # SingleGoal(keyframes[2][self.robot_idx["a1"]]),
            Task(
                ["a0"],
                SingleGoal(keyframes[3][self.robot_idx["a0"]]),
                type="pick",
                frames=["a0_ur_vacuum", "bottle_12"],
            ),
            Task(
                ["a0"],
                SingleGoal(keyframes[4][self.robot_idx["a0"]]),
                type="place",
                frames=["table", "bottle_12"],
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[6][self.robot_idx["a1"]]),
                type="pick",
                frames=["a1_ur_vacuum", "bottle_3"],
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[7][self.robot_idx["a1"]]),
                type="place",
                frames=["table", "bottle_3"],
            ),
            # SingleGoal(keyframes[8][self.robot_idx["a1"]]),
            Task(
                ["a1"],
                SingleGoal(keyframes[9][self.robot_idx["a1"]]),
                type="pick",
                frames=["a1_ur_vacuum", "bottle_5"],
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[10][self.robot_idx["a1"]]),
                type="place",
                frames=["table", "bottle_5"],
            ),
            Task(
                ["a0", "a1"],
                SingleGoal(
                    np.concatenate(
                        [
                            keyframes[5][self.robot_idx["a0"]],
                            keyframes[11][self.robot_idx["a1"]],
                        ]
                    )
                ),
            ),
        ]

        self.tasks[0].name = "a1_pick_b1"
        self.tasks[1].name = "a1_place_b1"
        self.tasks[2].name = "a1_pick_b2"
        self.tasks[3].name = "a1_place_b2"
        self.tasks[4].name = "a2_pick_b1"
        self.tasks[5].name = "a2_place_b1"
        self.tasks[6].name = "a2_pick_b2"
        self.tasks[7].name = "a2_place_b2"
        self.tasks[8].name = "terminal"

        if num_bottles == 2:
            self.sequence = self._make_sequence_from_names(
                [
                    "a1_pick_b1",
                    "a2_pick_b1",
                    "a1_place_b1",
                    "a2_place_b1",
                    "a1_pick_b2",
                    "a2_pick_b2",
                    "a1_place_b2",
                    "a2_place_b2",
                    "terminal",
                ]
            )
        else:
            self.sequence = self._make_sequence_from_names(
                [
                    "a1_pick_b1",
                    "a2_pick_b1",
                    "a1_place_b1",
                    "a2_place_b1",
                    "terminal",
                ]
            )

        BaseModeLogic.__init__(self)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.0000
        self.collision_resolution = 0.005

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        dim = 6
        for i, r in enumerate(self.robots):
            print(self.C.getJointState()[0:6])
            self.safe_pose[r] = np.array(self.C.getJointState()[dim*i:dim*(i+1)])


class rai_ur10_arm_bottle_dep_env(DependencyGraphMixin, rai_env):
    def __init__(self, num_bottles=2):
        assert num_bottles in [1,2]

        self.C, keyframes = rai_config.make_bottle_insertion()

        self.robots = ["a0", "a1"]

        rai_env.__init__(self)

        self.manipulating_env = True

        self.tasks = [
            Task(
                ["a0"],
                SingleGoal(keyframes[0][self.robot_idx["a0"]]),
                type="pick",
                frames=["a0_ur_vacuum", "bottle_1"],
            ),
            Task(
                ["a0"],
                SingleGoal(keyframes[1][self.robot_idx["a0"]]),
                type="place",
                frames=["table", "bottle_1"],
            ),
            # SingleGoal(keyframes[2][self.robot_idx["a1"]]),
            Task(
                ["a0"],
                SingleGoal(keyframes[3][self.robot_idx["a0"]]),
                type="pick",
                frames=["a0_ur_vacuum", "bottle_12"],
            ),
            Task(
                ["a0"],
                SingleGoal(keyframes[4][self.robot_idx["a0"]]),
                type="place",
                frames=["table", "bottle_12"],
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[6][self.robot_idx["a1"]]),
                type="pick",
                frames=["a1_ur_vacuum", "bottle_3"],
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[7][self.robot_idx["a1"]]),
                type="place",
                frames=["table", "bottle_3"],
            ),
            # SingleGoal(keyframes[8][self.robot_idx["a1"]]),
            Task(
                ["a1"],
                SingleGoal(keyframes[9][self.robot_idx["a1"]]),
                type="pick",
                frames=["a1_ur_vacuum", "bottle_5"],
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[10][self.robot_idx["a1"]]),
                type="place",
                frames=["table", "bottle_5"],
            ),
            Task(
                ["a0", "a1"],
                SingleGoal(
                    np.concatenate(
                        [
                            keyframes[5][self.robot_idx["a0"]],
                            keyframes[11][self.robot_idx["a1"]],
                        ]
                    )
                ),
            ),
        ]

        self.tasks[0].name = "a1_pick_b1"
        self.tasks[1].name = "a1_place_b1"
        self.tasks[2].name = "a1_pick_b2"
        self.tasks[3].name = "a1_place_b2"
        self.tasks[4].name = "a2_pick_b1"
        self.tasks[5].name = "a2_place_b1"
        self.tasks[6].name = "a2_pick_b2"
        self.tasks[7].name = "a2_place_b2"
        self.tasks[8].name = "terminal"
        
        if num_bottles == 2:
            self.graph = DependencyGraph()
            self.graph.add_dependency("a1_place_b1", "a1_pick_b1")
            self.graph.add_dependency("a1_pick_b2", "a1_place_b1")
            self.graph.add_dependency("a1_place_b2", "a1_pick_b2")

            self.graph.add_dependency("a2_place_b1", "a2_pick_b1")
            self.graph.add_dependency("a2_pick_b2", "a2_place_b1")
            self.graph.add_dependency("a2_place_b2", "a2_pick_b2")

            self.graph.add_dependency("terminal", "a2_place_b2")
            self.graph.add_dependency("terminal", "a1_place_b2")
        else:
            self.graph = DependencyGraph()
            self.graph.add_dependency("a1_place_b1", "a1_pick_b1")

            self.graph.add_dependency("a2_place_b1", "a2_pick_b1")

            self.graph.add_dependency("terminal", "a2_place_b1")
            self.graph.add_dependency("terminal", "a1_place_b1")

        BaseModeLogic.__init__(self)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.0000
        self.collision_resolution = 0.005

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        dim = 6
        for i, r in enumerate(self.robots):
            print(self.C.getJointState()[0:6])
            self.safe_pose[r] = np.array(self.C.getJointState()[dim*i:dim*(i+1)])



class rai_ur10_arm_box_rearrangement_env(SequenceMixin, rai_env):
    def __init__(self, num_robots=2, num_boxes=9, logo: bool = False):
        if not logo:
            self.C, actions, self.robots = rai_config.make_box_rearrangement_env(
                num_boxes=num_boxes, num_robots=num_robots
            )
        else:
            self.C, actions, self.robots = rai_config.make_crl_logo_rearrangement_env(
                num_robots=num_robots
            )

        rai_env.__init__(self)

        self.manipulating_env = True

        self.tasks = []

        direct_place_actions = ["pick", "place"]
        indirect_place_actions = ["pick", "place", "pick", "place"]

        action_names = {}

        obj_goal = {}

        for a in actions:
            robot = a[0]
            obj = a[1]
            keyframes = a[2]
            obj_goal[obj] = a[3]

            task_names = None
            if len(keyframes) == 2:
                task_names = direct_place_actions
            else:
                task_names = indirect_place_actions

            cnt = 0
            for t, k in zip(task_names, keyframes):
                if t == "pick":
                    ee_name = robot + "ur_vacuum"
                    self.tasks.append(
                        Task([robot], SingleGoal(k), t, frames=[ee_name, obj])
                    )
                else:
                    self.tasks.append(
                        Task([robot], SingleGoal(k), t, frames=["table", obj])
                    )

                self.tasks[-1].name = robot + t + "_" + obj + "_" + str(cnt)
                cnt += 1

                if obj in action_names:
                    action_names[obj].append(self.tasks[-1].name)
                else:
                    action_names[obj] = [self.tasks[-1].name]

        self.tasks.append(Task(self.robots, SingleGoal(self.C.getJointState())))
        self.tasks[-1].name = "terminal"

        # initialize the sequence with picking the first object
        named_sequence = [self.tasks[0].name]
        # remove the first task from the first action sequence
        action_names[actions[0][1]].pop(0)

        # initialize the available action sequences with the first two objects
        available_action_sequences = [actions[0][1], actions[1][1]]

        robot_gripper_free = {}
        for r in self.robots:
            robot_gripper_free[r] = True

        robot_gripper_free[self.tasks[0].name[:3]] = False
        location_is_free = {}

        for k, v in obj_goal.items():
            location_is_free[k[-2:]] = False

        location_is_free[available_action_sequences[0][-2:]] = True
        print(location_is_free)

        while True:
            # choose an action thingy from the available action sequences at random
            obj = random.choice(available_action_sequences)

            if len(action_names[obj]) == 0:
                continue

            potential_next_task = action_names[obj][0]
            r = potential_next_task[:3]

            if len(action_names[obj]) == 2 and not location_is_free[obj_goal[obj][-2:]]:
                continue

            if "pick" in potential_next_task and not robot_gripper_free[r]:
                continue

            if "place" in potential_next_task:
                robot_gripper_free[r] = True

            if "pick" in potential_next_task:
                robot_gripper_free[r] = False

            next_task = action_names[obj].pop(0)

            print(available_action_sequences)
            print(robot_gripper_free)
            print(next_task)

            if next_task[-1] == "0":
                location_is_free[obj[-2:]] = True
                if len(available_action_sequences) < len(actions):
                    available_action_sequences.append(
                        actions[len(available_action_sequences)][1]
                    )

            print(location_is_free)

            named_sequence.append(next_task)

            no_more_actions_available = True

            for obj, a in action_names.items():
                if len(a) > 0:
                    no_more_actions_available = False
                    break

            if no_more_actions_available:
                break

        named_sequence.append("terminal")

        self.sequence = self._make_sequence_from_names(named_sequence)

        print(self.sequence)

        BaseModeLogic.__init__(self)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.01

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        dim = 6
        for i, r in enumerate(self.robots):
            print(self.C.getJointState()[0:6])
            self.safe_pose[r] = np.array(self.C.getJointState()[dim*i:dim*(i+1)])


class rai_ur10_box_pile_cleanup_env(SequenceMixin, rai_env):
    def __init__(self, num_boxes=9, make_many_handover_poses: bool = False):
        if not make_many_handover_poses:
            self.C, keyframes = rai_config.make_box_pile_env(num_boxes=num_boxes)
        else:
            self.C, keyframes = rai_config.make_box_pile_env(
                num_boxes=num_boxes, compute_multiple_keyframes=make_many_handover_poses
            )

        self.robots = ["a1_", "a2_"]

        rai_env.__init__(self)

        self.manipulating_env = True

        self.tasks = []
        pick_task_names = ["pick", "place"]
        handover_task_names = ["pick", "handover", "place"]

        cnt = 0
        for primitive_type, robots, box_index, qs in keyframes:
            box_name = "obj" + str(box_index)
            print(primitive_type)
            if primitive_type == "pick":
                for t, k in zip(pick_task_names, qs[0]):
                    print(robots)
                    print(k)
                    if t == "pick":
                        ee_name = robots[0] + "ur_vacuum"
                        self.tasks.append(
                            Task(robots, SingleGoal(k), t, frames=[ee_name, box_name])
                        )
                    else:
                        self.tasks.append(
                            Task(robots, SingleGoal(k), t, frames=["tray", box_name])
                        )

                    self.tasks[-1].name = (
                        robots[0] + t + "_" + box_name + "_" + str(cnt)
                    )
                    cnt += 1
            else:
                for t, k in zip(handover_task_names, qs[0]):
                    if t == "pick":
                        ee_name = robots[0] + "ur_vacuum"
                        self.tasks.append(
                            Task(
                                [robots[0]],
                                SingleGoal(k[self.robot_idx[robots[0]]]),
                                t,
                                frames=[ee_name, box_name],
                            )
                        )
                    elif t == "handover":
                        ee_name = robots[1] + "ur_vacuum"
                        self.tasks.append(
                            # Task(
                            #     self.robots,
                            #     SingleGoal(k),
                            #     t,
                            #     frames=[ee_name, box_name],
                            # )
                            Task(
                                self.robots,
                                GoalSet([q[1] for q in qs]),
                                t,
                                frames=[ee_name, box_name],
                            )
                        )
                    else:
                        self.tasks.append(
                            Task(
                                [robots[1]],
                                SingleGoal(k[self.robot_idx[robots[1]]]),
                                t,
                                frames=["tray", box_name],
                            )
                        )

                    self.tasks[-1].name = (
                        robots[0] + t + "_" + box_name + "_" + str(cnt)
                    )
                    cnt += 1

        self.tasks.append(Task(self.robots, SingleGoal(self.C.getJointState())))
        self.tasks[-1].name = "terminal"

        # for t in self.tasks:
        #     arr = t.goal.sample(None)
        #     print(t.robots)
        #     print(np.array2string(np.array(arr), separator=", "))

        # self.export_tasks("tmp.txt")

        # print([t.name for t in self.tasks])

        self.sequence = self._make_sequence_from_names([t.name for t in self.tasks])

        BaseModeLogic.__init__(self)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.02

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        dim = 6
        for i, r in enumerate(self.robots):
            print(self.C.getJointState()[0:6])
            self.safe_pose[r] = np.array(self.C.getJointState()[dim*i:dim*(i+1)])



class rai_ur10_box_pile_cleanup_env_dep(DependencyGraphMixin, rai_env):
    def __init__(self, num_boxes=9, make_many_handover_poses: bool = False):
        if not make_many_handover_poses:
            self.C, keyframes = rai_config.make_box_pile_env(num_boxes=num_boxes)
        else:
            self.C, keyframes = rai_config.make_box_pile_env(
                num_boxes=num_boxes, compute_multiple_keyframes=make_many_handover_poses
            )

        self.robots = ["a1_", "a2_"]

        rai_env.__init__(self)

        self.manipulating_env = True

        self.tasks = []
        pick_task_names = ["pick", "place"]
        handover_task_names = ["pick", "handover", "place"]

        self.graph = DependencyGraph()

        last_robot_task = {}
        for r in self.robots:
            last_robot_task[r] = None

        cnt = 0
        for primitive_type, robots, box_index, qs in keyframes:
            box_name = "obj" + str(box_index)
            print(primitive_type)
            prev_task = None

            print(last_robot_task)

            if primitive_type == "pick":
                for t, k in zip(pick_task_names, qs[0]):
                    print(robots)
                    print(k)
                    task_name = robots[0] + t + "_" + box_name + "_" + str(cnt)

                    if (
                        last_robot_task[robots[0]] is not None
                        and task_name != last_robot_task[robots[0]]
                    ):
                        self.graph.add_dependency(task_name, last_robot_task[robots[0]])

                    if t == "pick":
                        ee_name = robots[0] + "ur_vacuum"
                        self.tasks.append(
                            Task(robots, SingleGoal(k), t, frames=[ee_name, box_name])
                        )
                        last_robot_task[robots[0]] = task_name
                    else:
                        self.tasks.append(
                            Task(robots, SingleGoal(k), t, frames=["tray", box_name])
                        )
                        last_robot_task[robots[0]] = task_name

                    self.tasks[-1].name = task_name

                    if prev_task is not None:
                        self.graph.add_dependency(task_name, prev_task)

                    prev_task = task_name

                    cnt += 1
            else:
                for t, k in zip(handover_task_names, qs[0]):
                    task_name = robots[0] + t + "_" + box_name + "_" + str(cnt)

                    if t == "pick":
                        ee_name = robots[0] + "ur_vacuum"
                        self.tasks.append(
                            Task(
                                [robots[0]],
                                SingleGoal(k[self.robot_idx[robots[0]]]),
                                t,
                                frames=[ee_name, box_name],
                            )
                        )

                        if (
                            last_robot_task[robots[0]] != task_name
                            and last_robot_task[robots[0]] is not None
                        ):
                            print("A")
                            self.graph.add_dependency(
                                task_name, last_robot_task[robots[0]]
                            )

                        last_robot_task[robots[0]] = task_name

                    elif t == "handover":
                        ee_name = robots[1] + "ur_vacuum"
                        self.tasks.append(
                            # Task(
                            #     robots,
                            #     SingleGoal(k),
                            #     t,
                            #     frames=[ee_name, box_name],
                            # )
                            Task(
                                self.robots,
                                GoalSet([q[1] for q in qs]),
                                t,
                                frames=[ee_name, box_name],
                            )
                        )

                        if (
                            last_robot_task[robots[1]] != task_name
                            and last_robot_task[robots[1]] is not None
                        ):
                            self.graph.add_dependency(
                                task_name, last_robot_task[robots[1]]
                            )

                        last_robot_task[robots[0]] = task_name
                        last_robot_task[robots[1]] = task_name

                    else:
                        task_name = robots[1] + t + "_" + box_name + "_" + str(cnt)

                        self.tasks.append(
                            Task(
                                [robots[1]],
                                SingleGoal(k[self.robot_idx[robots[1]]]),
                                t,
                                frames=["tray", box_name],
                            )
                        )

                        if (
                            last_robot_task[robots[1]] != task_name
                            and last_robot_task[robots[1]] is not None
                        ):
                            self.graph.add_dependency(
                                task_name, last_robot_task[robots[1]]
                            )

                        last_robot_task[robots[1]] = task_name

                    self.tasks[-1].name = task_name

                    if prev_task is not None:
                        self.graph.add_dependency(task_name, prev_task)

                    prev_task = task_name

                    cnt += 1

        self.tasks.append(Task(self.robots, SingleGoal(self.C.getJointState())))
        self.tasks[-1].name = "terminal"

        for r in self.robots:
            self.graph.add_dependency("terminal", last_robot_task[r])

        BaseModeLogic.__init__(self)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.01
        # self.collision_resolution = 0.1

        self.spec.dependency = DependencyType.UNORDERED

class rai_ur10_arm_box_pyramid_appearing_parts(SequenceMixin, rai_env):
    def __init__(self, num_robots=4, num_boxes: int = 6):
        self.C, keyframes, self.robots = rai_config.make_pyramid_env(
            num_robots, num_boxes, view=True
        )

        rai_env.__init__(self)

        self.manipulating_env = True

        self.tasks = []
        task_names = ["pick", "place"]
        for r, b, qs, g in keyframes:
            cnt = 0
            for t, k in zip(task_names, qs):
                if t == "pick":
                    ee_name = r + "gripper_center"
                    appearance_pose = np.array([0, 0, 0.6, 1, 0, 0, 0])
                    task = Task([r], SingleGoal(k), t, frames=[ee_name, b], side_effect="make_appear", side_effect_data=appearance_pose)
                    self.tasks.append(task)
                else:
                    self.tasks.append(Task([r], SingleGoal(k), t, frames=["table", b]))

                self.tasks[-1].name = r + t + "_" + b + "_" + str(cnt)
                cnt += 1

                # if b in action_names:
                #     action_names[b].append(self.tasks[-1].name)
                # else:
                #     action_names[b] = [self.tasks[-1].name]

        self.tasks.append(Task(self.robots, SingleGoal(self.C.getJointState())))
        self.tasks[-1].name = "terminal"

        self.sequence = self._make_sequence_from_names([t.name for t in self.tasks])

        BaseModeLogic.__init__(self)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.005
        # self.collision_resolution = 0.005
        self.collision_resolution = 0.01

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        for r in self.robots:
            print(self.C.getJointState()[0:6])
            self.safe_pose[r] = np.array(self.C.getJointState()[0:6])


# best cost found (max): 21.45
class rai_ur10_arm_box_stack_env(SequenceMixin, rai_env):
    def __init__(self, num_robots=4, num_boxes: int = 8):
        self.C, keyframes, self.robots = rai_config.make_box_stacking_env(
            num_robots, num_boxes
        )

        rai_env.__init__(self)

        self.manipulating_env = True

        self.tasks = []
        task_names = ["pick", "place"]
        for r, b, qs, g in keyframes:
            cnt = 0
            for t, k in zip(task_names, qs):
                if t == "pick":
                    ee_name = r + "gripper_center"
                    self.tasks.append(Task([r], SingleGoal(k), t, frames=[ee_name, b]))
                else:
                    self.tasks.append(Task([r], SingleGoal(k), t, frames=["table", b]))

                self.tasks[-1].name = r + t + "_" + b + "_" + str(cnt)
                cnt += 1

                # if b in action_names:
                #     action_names[b].append(self.tasks[-1].name)
                # else:
                #     action_names[b] = [self.tasks[-1].name]

        self.tasks.append(Task(self.robots, SingleGoal(self.C.getJointState())))
        self.tasks[-1].name = "terminal"

        self.sequence = self._make_sequence_from_names([t.name for t in self.tasks])

        BaseModeLogic.__init__(self)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.005
        # self.collision_resolution = 0.005
        self.collision_resolution = 0.01

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        for r in self.robots:
            print(self.C.getJointState()[0:6])
            self.safe_pose[r] = np.array(self.C.getJointState()[0:6])


class rai_ur10_arm_box_stack_env_dep(DependencyGraphMixin, rai_env):
    def __init__(self, num_robots=4, num_boxes: int = 8):
        np.random.seed(1)
        random.seed(2)

        self.C, keyframes, self.robots = rai_config.make_box_stacking_env(
            num_robots, num_boxes
        )

        rai_env.__init__(self)

        self.manipulating_env = True

        self.graph = DependencyGraph()

        prev_task_names = {}
        for robot in self.robots:
            prev_task_names[robot] = None

        prev_place = None

        self.tasks = []
        task_names = ["pick", "place"]
        for r, b, qs, g in keyframes:
            cnt = 0
            for t, k in zip(task_names, qs):
                task_name = r + t + "_" + b + "_" + str(cnt)

                if t == "pick":
                    ee_name = r + "gripper_center"
                    self.tasks.append(Task([r], SingleGoal(k), t, frames=[ee_name, b]))
                else:
                    self.tasks.append(Task([r], SingleGoal(k), t, frames=["table", b]))

                self.tasks[-1].name = task_name
                cnt += 1

                if prev_task_names[r] is not None:
                    self.graph.add_dependency(task_name, prev_task_names[r])

                if t == "place" and prev_place is not None:
                    self.graph.add_dependency(task_name, prev_place)

                if t == "place":
                    prev_place = task_name

                prev_task_names[r] = task_name

        self.tasks.append(Task(self.robots, SingleGoal(self.C.getJointState())))
        self.tasks[-1].name = "terminal"

        for r in self.robots:
            self.graph.add_dependency("terminal", prev_task_names[r])

        BaseModeLogic.__init__(self)

        # self.graph.visualize()

        # buffer for faster collision checking
        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.005
        self.collision_resolution = 0.01

        self.spec.dependency = DependencyType.UNORDERED
        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        for r in self.robots:
            print(self.C.getJointState()[0:6])
            self.safe_pose[r] = np.array(self.C.getJointState()[0:6])


# mobile manip
class rai_mobile_manip_wall(SequenceMixin, rai_env):
    def __init__(self, num_robots=4):
        self.C, keyframes = rai_config.make_mobile_manip_env(num_robots)

        self.robots = [k for k in keyframes]

        rai_env.__init__(self)

        self.manipulating_env = True

        self.tasks = []
        task_names = ["pick", "place"]
        for robot_prefix, robot_tasks in keyframes.items():
            for box, poses in robot_tasks:
                cnt = 0
                for t, k in zip(task_names, poses):
                    if t == "pick":
                        ee_name = robot_prefix + "gripper"
                        self.tasks.append(
                            Task(
                                [robot_prefix], SingleGoal(k), t, frames=[ee_name, box]
                            )
                        )
                    else:
                        self.tasks.append(
                            Task(
                                [robot_prefix], SingleGoal(k), t, frames=["table", box]
                            )
                        )

                    self.tasks[-1].name = robot_prefix + t + "_" + box + "_" + str(cnt)
                    cnt += 1

                    # if b in action_names:
                    #     action_names[b].append(self.tasks[-1].name)
                    # else:
                    #     action_names[b] = [self.tasks[-1].name]

        self.tasks.append(Task(self.robots, SingleGoal(self.C.getJointState())))
        self.tasks[-1].name = "terminal"

        self.sequence = self._make_sequence_from_names([t.name for t in self.tasks])

        BaseModeLogic.__init__(self)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.005
        self.collision_resolution = 0.02

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        dim = 6
        for i, r in enumerate(self.robots):
            print(self.C.getJointState()[0:6])
            self.safe_pose[r] = np.array(self.C.getJointState()[dim*i:dim*(i+1)])


class rai_mobile_manip_wall_dep(DependencyGraphMixin, rai_env):
    def __init__(self, num_robots=3):
        self.C, keyframes = rai_config.make_mobile_manip_env(num_robots)

        self.robots = [k for k in keyframes]

        rai_env.__init__(self)

        self.manipulating_env = True

        self.graph = DependencyGraph()

        self.tasks = []
        task_names = ["pick", "place"]
        cnt = 0
        for robot_prefix, robot_tasks in keyframes.items():
            prev_task_name = None
            for box, poses in robot_tasks:
                for t, k in zip(task_names, poses):
                    task_name = robot_prefix + t + "_" + box + "_" + str(cnt)

                    if t == "pick":
                        ee_name = robot_prefix + "gripper"
                        self.tasks.append(
                            Task(
                                [robot_prefix], SingleGoal(k), t, frames=[ee_name, box]
                            )
                        )
                    else:
                        self.tasks.append(
                            Task(
                                [robot_prefix], SingleGoal(k), t, frames=["table", box]
                            )
                        )

                    if prev_task_name is not None:
                        self.graph.add_dependency(task_name, prev_task_name)

                    self.tasks[-1].name = task_name
                    prev_task_name = task_name
                    cnt += 1

            self.graph.add_dependency("terminal", prev_task_name)

        self.tasks.append(Task(self.robots, SingleGoal(self.C.getJointState())))
        self.tasks[-1].name = "terminal"

        # random.shuffle(self.tasks)

        print(self.graph)

        # self.graph.visualize()

        # for t in self.tasks:
        #     print(t.name)

        BaseModeLogic.__init__(self)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.005
        self.collision_resolution = 0.02

        self.spec.dependency = DependencyType.UNORDERED
        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        dim = 6
        for i, r in enumerate(self.robots):
            print(self.C.getJointState()[0:6])
            self.safe_pose[r] = np.array(self.C.getJointState()[dim*i:dim*(i+1)])


class rai_coop_tamp_architecture(SequenceMixin, rai_env):
    def __init__(self, assembly_name):
        self.C, self.robots, keyframes = rai_config.coop_tamp_architecture_env(assembly_name=assembly_name)

        rai_env.__init__(self)

        self.manipulating_env = True

        self.tasks = []
        task_names = ["pick", "place"]
        for r, ee_name, b, qs, appearance_pose in keyframes:
            cnt = 0
            for t, k in zip(task_names, qs):
                if t == "pick":
                    task = Task([r], SingleGoal(k), t, frames=[ee_name, b], side_effect="make_appear", side_effect_data=appearance_pose)
                    self.tasks.append(task)
                else:
                    self.tasks.append(Task([r], SingleGoal(k), t, frames=["table", b]))

                self.tasks[-1].name = r + t + "_" + b + "_" + str(cnt)
                cnt += 1

                # if b in action_names:
                #     action_names[b].append(self.tasks[-1].name)
                # else:
                #     action_names[b] = [self.tasks[-1].name]

        self.tasks.append(Task(self.robots, SingleGoal(self.C.getJointState())))
        self.tasks[-1].name = "terminal"

        self.sequence = self._make_sequence_from_names([t.name for t in self.tasks])

        BaseModeLogic.__init__(self)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.005
        self.collision_resolution = 0.005

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        dim = 6
        for i, r in enumerate(self.robots):
            print(self.C.getJointState()[0:6])
            self.safe_pose[r] = np.array(self.C.getJointState()[dim*i:dim*(i+1)])


def check_all_modes():
    all_envs = [
        "piano",
        "simple_2d",
        "three_agents",
        "box_sorting",
        "eggs",
        "triple_waypoints",
        "welding",
        "bottles",
    ]

    for env_name in all_envs:
        print(env_name)
        env = rai_config.get_env_by_name(env_name)
        q_home = env.start_pos
        m = env.start_mode
        for i in range(len(env.sequence)):
            if env.is_terminal_mode(m):
                switching_robots = [r for r in env.robots]
            else:
                # find the robot(s) that needs to switch the mode
                switching_robots = env.get_goal_constrained_robots(m)

            q = []
            task = env.get_active_task(m)
            goal_sample = task.goal.sample(m)

            print("switching robots: ", switching_robots)

            # env.show()

            for j, r in enumerate(env.robots):
                if r in switching_robots:
                    # TODO: need to check all goals here
                    # figure out where robot r is in the goal description
                    offset = 0
                    for _, task_robot in enumerate(task.robots):
                        if task_robot == r:
                            q.append(
                                goal_sample[
                                    offset : offset + env.robot_dims[task_robot]
                                ]
                            )
                            break
                        offset += env.robot_dims[task_robot]
                    # q.append(goal_sample)
                else:
                    q.append(q_home.robot_state(j))

            is_collision_free = env.is_collision_free(
                type(env.get_start_pos()).from_list(q), m
            )
            print(f"mode {m} is collision free: ", is_collision_free)

            env.show()

            colls = env.C.getCollisions()
            for c in colls:
                if c[2] < 0:
                    print(c)

            if not is_collision_free:
                raise ValueError()

            ms = env.get_next_modes(None, m)
            assert len(ms) == 0
            m = ms[0]



def export_env(env: rai_env):
    # export scene
    rai_env.C.writeURDF()

    # export dependendency graph/sequence
    ## export computed exact keyframes
    pass


def load_env_from_file(filepath):
    pass
