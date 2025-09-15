import robotic as ry
import numpy as np
from .planning_env import (
    BaseModeLogic,
    UnorderedButAssignedMixin,
    Task,
    SafePoseType,
    ManipulationType
)
from .goals import (
    SingleGoal,
    GoalSet,
    GoalRegion,
    ConditionalGoal,
)
from .rai_base_env import rai_env
import multi_robot_multi_goal_planning.problems.rai_config as rai_config

from .registry import register

@register("rai.unordered")
class rai_two_dim_env(UnorderedButAssignedMixin, rai_env):
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
            Task(
                ["a1", "a2"],
                # GoalRegion(self.C.getJointLimits()),
                SingleGoal(self.C.getJointState())
            ),
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

        self.tasks[0].name = "dummy_start"
        self.tasks[1].name = "a1_goal"
        self.tasks[2].name = "a2_goal_0"
        self.tasks[3].name = "a2_goal_1"
        self.tasks[4].name = "a2_goal_2"
        self.tasks[5].name = "a2_goal_3"
        self.tasks[6].name = "terminal"

        self.per_robot_tasks = [[1], [2, 3, 4, 5]]
        self.terminal_task = 6
        self.task_dependencies = {}

        self.collision_tolerance = 0.01

        BaseModeLogic.__init__(self)

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE
        self.spec.manipulation = ManipulationType.STATIC

        self.safe_pose = {
            "a1": r1_goal,
            "a2": r2_goal_1
        }

@register("rai.unordered_square")
class rai_two_dim_square_env(UnorderedButAssignedMixin, rai_env):
    def __init__(self, agents_can_rotate=True):
        self.C = rai_config.make_2d_rai_env_no_obs(agents_can_rotate=agents_can_rotate)
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        # r1 starts at both negative (-.5, -.5)
        r1_state = self.C.getJointState()[self.robot_idx["a1"]]
        # r2 starts at both positive (.5, .5)
        r2_state = self.C.getJointState()[self.robot_idx["a2"]]

        r1_goal = r1_state * 1.0
        r1_goal[:2] = [-0.5, 0.5]

        r2_goal_1 = r2_state * 1.0
        r2_goal_1[:2] = [0.5, -0.5]
        r2_goal_2 = r2_state * 1.0
        r2_goal_2[:2] = [-0.5, -0.5]
        r2_goal_3 = r2_state * 1.0
        r2_goal_3[:2] = [-0.5, 0.5]

        self.tasks = [
            Task(
                ["a1", "a2"],
                # GoalRegion(self.C.getJointLimits()),
                SingleGoal(self.C.getJointState())
            ),
            # r1
            Task(["a1"], SingleGoal(r1_goal)),
            # r2
            Task(["a2"], SingleGoal(r2_goal_1)),
            Task(["a2"], SingleGoal(r2_goal_2)),
            Task(["a2"], SingleGoal(r2_goal_3)),
            # Task(["a2"], SingleGoal(r2_goal_3)),
            # terminal mode
            Task(
                ["a1", "a2"],
                SingleGoal(self.C.getJointState()),
            ),
        ]

        self.tasks[0].name = "dummy_start"
        self.tasks[1].name = "a1_goal"
        self.tasks[2].name = "a2_goal_0"
        self.tasks[3].name = "a2_goal_1"
        self.tasks[4].name = "a2_goal_2"
        self.tasks[5].name = "terminal"

        self.per_robot_tasks = [[1], [2, 3, 4]]
        self.terminal_task = 5

        self.collision_tolerance = 0.01
        self.task_dependencies = {}

        BaseModeLogic.__init__(self)

@register("rai.unordered_circle")
class rai_two_dim_circle_env(UnorderedButAssignedMixin, rai_env):
    def __init__(self, agents_can_rotate=False):
        self.C = rai_config.make_2d_rai_env_no_obs(agents_can_rotate=agents_can_rotate)
        # self.C.view(True)

        self.C.addFrame("obs0").setParent(self.C.getFrame("table")).setPosition(
            self.C.getFrame("table").getPosition() + [0.0, 0, 0.07]
        ).setShape(ry.ST.box, size=[0.4, 0.4, 0.06, 0.005]).setContact(1).setColor(
            [0, 0, 0]
        ).setJoint(ry.JT.rigid)

        # self.C.view(True)

        self.robots = ["a1", "a2"]

        rai_env.__init__(self)

        # r1 starts at both negative (-.5, -.5)
        r1_state = self.C.getJointState()[self.robot_idx["a1"]]
        # r2 starts at both positive (.5, .5)
        r2_state = self.C.getJointState()[self.robot_idx["a2"]]

        r1_goal = r1_state * 1.0
        r1_goal[:2] = [-0.5, 0.5]

        r2_goals = []

        N = 6
        r = 0.5

        for i in range(N):
            goal = r2_state * 1.0
            goal[:2] = [
                np.sin(1.0 * i / N * np.pi * 2) * r,
                np.cos(1.0 * i / N * np.pi * 2) * r,
            ]
            r2_goals.append(goal)

        self.tasks = [
            Task(
                ["a1", "a2"],
                # GoalRegion(self.C.getJointLimits()),
                SingleGoal(self.C.getJointState())
            ),
            # r1
            Task(["a1"], SingleGoal(r1_goal)),
        ]

        for i, g in enumerate(r2_goals):
            self.tasks.append(
                Task(["a2"], SingleGoal(g)),
            )
            self.tasks[-1].name = f"a2_goal_{i}"

        self.tasks.append(  # terminal mode
            Task(
                ["a1", "a2"],
                SingleGoal(self.C.getJointState()),
            ),
        )
        self.tasks[-1].name = "terminal"

        self.tasks[0].name = "dummy_start"
        self.tasks[1].name = "a1_goal"

        self.per_robot_tasks = [[1], [2 + i for i in range(len(r2_goals))]]
        self.terminal_task = len(self.tasks) - 1
        self.task_dependencies = {}

        self.collision_tolerance = 0.01

        BaseModeLogic.__init__(self)


@register("rai.unordered_single_agent_circle")
class rai_two_dim_circle_single_agent(UnorderedButAssignedMixin, rai_env):
    def __init__(self, agents_can_rotate=False):
        self.C, keyframes = rai_config.make_random_two_dim_single_goal(
            num_agents=1,
            num_obstacles=0,
            agents_can_rotate=agents_can_rotate,
        )
        # self.C.view(True)

        self.robots = [f"a{i}" for i in range(1)]

        rai_env.__init__(self)

        state = self.C.getJointState()[self.robot_idx["a0"]]

        goals = []

        N = 5
        r = 1.0

        for i in range(N):
            goal = state * 1.0
            goal[:2] = [
                np.sin(1.0 * i / N * np.pi * 2) * r,
                np.cos(1.0 * i / N * np.pi * 2) * r,
            ]
            goals.append(goal)

        self.tasks = [
            Task(
                ["a0"],
                # GoalRegion(self.C.getJointLimits()),
                SingleGoal(self.C.getJointState())
            ),
        ]

        for i, g in enumerate(goals):
            self.tasks.append(
                Task(["a0"], SingleGoal(g)),
            )
            self.tasks[-1].name = f"a2_goal_{i}"

        self.tasks.append(  # terminal mode
            Task(
                ["a0"],
                SingleGoal(self.C.getJointState()),
            ),
        )
        self.tasks[-1].name = "terminal"

        self.tasks[0].name = "dummy_start"

        self.per_robot_tasks = [[1 + i for i in range(len(goals))]]
        self.terminal_task = len(self.tasks) - 1
        self.task_dependencies = {}

        self.collision_tolerance = 0.01

        BaseModeLogic.__init__(self)


@register("rai.unordered_box_reorientation")
class rai_unordered_ur10_box_pile_cleanup_env(UnorderedButAssignedMixin, rai_env):
    def __init__(self, num_boxes=6):
        self.C, keyframes = rai_config.make_box_pile_env(
            num_boxes=num_boxes, random_orientation=False
        )

        self.robots = ["a1_", "a2_"]

        rai_env.__init__(self)

        self.manipulating_env = True

        self.tasks = [
            Task(
                ["a1_", "a2_"],
                # GoalRegion(self.limits),
                SingleGoal(self.C.getJointState())
            ),
        ]
        self.tasks[-1].name = "dummy_start"

        pick_task_names = ["pick", "place"]

        self.per_robot_tasks = [[], []]
        self.task_dependencies = {}

        cnt = 0
        for primitive_type, robots, box_index, qs in keyframes:
            box_name = "obj" + str(box_index)

            robot_index = 0
            if robots[0] == "a2_":
                robot_index = 1
            
            # print("robot index", robot_index)

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
                        self.task_dependencies[len(self.tasks) - 1] = [
                            len(self.tasks) - 2
                        ]

                    self.per_robot_tasks[robot_index].append(len(self.tasks) - 1)

                    self.tasks[-1].name = (
                        robots[0] + t + "_" + box_name + "_" + str(cnt)
                    )
                    cnt += 1
            else:
                assert False

        self.tasks.append(Task(self.robots, SingleGoal(self.C.getJointState())))
        self.tasks[-1].name = "terminal"

        self.terminal_task = len(self.tasks) - 1

        BaseModeLogic.__init__(self)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.01

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {}
        dim = 6
        for i, r in enumerate(self.robots):
            print(self.C.getJointState()[0:6])
            self.safe_pose[r] = np.array(self.C.getJointState()[dim*i:dim*(i+1)])
