import robotic as ry
import numpy as np
import random
import argparse
import time

from typing import List, Dict
from numpy.typing import NDArray

# from dependency_graph import DependencyGraph

from rai_config import *
from planning_env import base_env, State, Task, SingleGoal, GoalSet, GoalRegion
from configuration import Configuration, NpConfiguration, config_dist, config_cost, batch_config_cost

from util import generate_binary_search_indices

def get_joint_indices(C: ry.Config, prefix: str) -> List[int]:
    all_joints_weird = C.getJointNames()

    indices = []
    for idx, j in enumerate(all_joints_weird):
        if prefix in j:
            indices.append(idx)

    return indices


def get_robot_state(C: ry.Config, robot_prefix: str) -> NDArray:
    idx = get_joint_indices(C, robot_prefix)
    q = C.getJointState()[idx]

    return q


def set_robot_active(C: ry.Config, robot_prefix: str) -> None:
    robot_joints = get_robot_joints(C, robot_prefix)
    C.selectJoints(robot_joints)


class rai_env(base_env):
    # robot things
    C: ry.Config
    robots: List[str]
    robot_dims: Dict[str, NDArray]
    robot_idx: Dict[str, NDArray]
    start_pos: Configuration
    limits: NDArray

    # sequence things
    sequence: List[int]
    tasks: List[Task]
    start_mode: List[int]
    terminal_mode: List[int]

    # misc
    tolerance: float

    def __init__(self):
        self.robot_idx = {}
        self.robot_dims = {}

        for r in self.robots:
            self.robot_idx[r] = get_joint_indices(self.C, r)
            self.robot_dims[r] = len(get_joint_indices(self.C, r))

        self.start_pos = NpConfiguration.from_list(
            [get_robot_state(self.C, r) for r in self.robots]
        )

        self.manipulating_env = False

        self.limits = self.C.getJointLimits()

        self.tolerance = 0.1

    def config_cost(self, start, end):
        return config_cost(start, end, "max")

    def batch_config_cost(self, starts, ends):
        return batch_config_cost(starts, ends, "max")

    def _make_sequence_from_names(self, names):
        sequence = []

        for name in names:
            no_task_with_name_found = True
            for idx, task in enumerate(self.tasks):
                if name == task.name:
                    sequence.append(idx)
                    no_task_with_name_found = False

            if no_task_with_name_found:
                raise ValueError(f"Task with name {name} not found.")

        return sequence

    def _make_start_mode_from_sequence(self):
        mode_dict = {}

        for task_index in self.sequence:
            task_robots = self.tasks[task_index].robots

            for r in task_robots:
                if r not in mode_dict:
                    mode_dict[r] = task_index

        mode = []
        for r in self.robots:
            mode.append(mode_dict[r])

        return mode

    def _make_terminal_mode_from_sequence(self):
        mode_dict = {}

        for task_index in self.sequence:
            task_robots = self.tasks[task_index].robots

            # difference to above: we do not check if the robot already has a task assigned
            for r in task_robots:
                mode_dict[r] = task_index

        mode = []
        for r in self.robots:
            mode.append(mode_dict[r])

        return mode

    # TODO: is that really a good way to sample a mode?
    def sample_random_mode(self) -> List[int]:
        m = self.start_mode
        rnd = random.randint(0, len(self.sequence))

        for _ in range(rnd):
            m = self.get_next_mode(_, m)

        return m

    # TODO: this only works for the sequence, i.e. a single task being active
    def get_current_seq_index(self, mode: List[int]) -> int:
        # Approach: iterate throug all indices, find them in the sequence, and check which is the one
        # that has to be fulfilled first
        min_sequence_pos = len(self.sequence) - 1
        for i, m in enumerate(mode):
            # print("robots in task:", self.tasks[m].robots, self.sequence.index(m))
            if m != self.terminal_mode[i]:
                min_sequence_pos = min(self.sequence.index(m), min_sequence_pos)

        return min_sequence_pos

    def get_goal_constrained_robots(self, mode: List[int]) -> List[str]:
        seq_index = self.get_current_seq_index(mode)
        task = self.tasks[self.sequence[seq_index]]
        return task.robots

    def done(self, q: Configuration, m: List[int]) -> bool:
        if m != self.terminal_mode:
            return False

        # TODO: this is not necessarily true!
        terminal_task_idx = self.sequence[-1]
        terminal_task = self.tasks[terminal_task_idx]
        involved_robots = terminal_task.robots

        q_concat = []
        for r in involved_robots:
            r_idx = self.robots.index(r)
            q_concat.append(q.robot_state(r_idx))

        q_concat = np.concatenate(q_concat)

        if terminal_task.goal.satisfies_constraints(q_concat, self.tolerance):
            return True

        return False

    def show_config(self, q):
        self.C.setJointState(q)
        self.C.view(True)

    def show(self):
        self.C.view(True)

    def is_transition(self, q: Configuration, m: List[int]) -> bool:
        if m == self.terminal_mode:
            return False

        robots_with_constraints_in_current_mode = self.get_goal_constrained_robots(m)
        task = self.get_active_task(m)

        q_concat = []
        for r in robots_with_constraints_in_current_mode:
            r_idx = self.robots.index(r)
            q_concat.append(q.robot_state(r_idx))

        q_concat = np.concatenate(q_concat)

        if task.goal.satisfies_constraints(q_concat, self.tolerance):
            return True

        return False

    def get_next_mode(self, q: Configuration, mode: List[int]) -> List[int]:
        seq_idx = self.get_current_seq_index(mode)

        # print('seq_idx', seq_idx)

        # find the next mode for the currently constrained one(s)
        task_idx = self.sequence[seq_idx]
        rs = self.tasks[task_idx].robots

        # next_robot_mode_ind = None

        m_next = mode.copy()

        # print(rs)

        # find next occurrence of the robot in the sequence/dep graph
        for r in rs:
            for idx in self.sequence[seq_idx + 1 :]:
                if r in self.tasks[idx].robots:
                    r_idx = self.robots.index(r)
                    m_next[r_idx] = idx
                    break

        return m_next

    def get_active_task(self, mode: List[int]) -> Task:
        seq_idx = self.get_current_seq_index(mode)
        return self.tasks[self.sequence[seq_idx]]

    def get_tasks_for_mode(self, mode: List[int]) -> List[Task]:
        tasks = []
        for _, j in enumerate(mode):
            tasks.append(self.tasks[j])

        return tasks

    def is_collision_free(
        self, q: Configuration, m: List[int], collision_tolerance: float = 0.01
    ) -> bool:
        # print(q)
        # self.C.setJointState(q)

        self.set_to_mode(m)
        self.C.setJointState(q)

        binary_collision_free = self.C.getCollisionFree()
        if binary_collision_free:
            return True

        col = self.C.getCollisionsTotalPenetration()
        # print(col)
        # self.C.view(False)
        if col > collision_tolerance:
            # self.C.view(False)
            return False

        return True

    def is_edge_collision_free(
        self,
        q1: Configuration,
        q2: Configuration,
        m: List[int],
        resolution=0.1,
        randomize_order=True,
    ) -> bool:
        # print('q1', q1)
        # print('q2', q2)
        N = config_dist(q1, q2) / resolution
        N = max(5, N)

        idx = list(range(int(N)))
        if randomize_order:
            # np.random.shuffle(idx)
            idx = generate_binary_search_indices(int(N)).copy()

        for i in idx:
            # print(i / (N-1))
            q = q1.state() + (q2.state() - q1.state()) * (i) / (N - 1)
            if not self.is_collision_free(q, m):
                # print('coll')
                return False

        return True

    def is_path_collision_free(self, path: List[State], randomize_order=True) -> bool:
        idx = list(range(len(path) - 1))
        if randomize_order:
            np.random.shuffle(idx)

        for i in idx:
            # skip transition nodes
            if path[i].mode != path[i + 1].mode:
                continue

            q1 = path[i].q
            q2 = path[i + 1].q
            mode = path[i].mode

            if not self.is_edge_collision_free(q1, q2, mode):
                return False

        return True

    def set_to_mode(self, m: List[int]):
        if not self.manipulating_env:
            return

        # do not remake the config if we are in the same mode as we have been before
        if m == self.prev_mode:
            return

        self.prev_mode = m

        # self.C.view(True)
        self.C.clear()
        self.C.addConfigurationCopy(self.C_base)

        # find current mode
        current_mode = self.get_current_seq_index(m)

        if current_mode != self.get_current_seq_index(m):
            print(current_mode, self.get_current_seq_index(m))
            raise ValueError

        if current_mode == 0:
            return

        mode = self.start_mode
        for i in range(len(self.sequence) + 1):
            if i == 0:
                continue

            m_prev = mode.copy()
            mode = self.get_next_mode(None, mode)

            mode_switching_robots = self.get_goal_constrained_robots(m_prev)

            # set robot to config
            prev_mode_index = m_prev[self.robots.index(mode_switching_robots[0])]
            # robot = self.robots[mode_switching_robots]

            # TODO: ensure that se are choosing the correct goal here
            q = self.tasks[prev_mode_index].goal.sample()
            joint_names = []
            for r in mode_switching_robots:
                joint_names.extend(get_robot_joints(self.C, r))

            self.C.setJointState(q, joint_names)

            if self.tasks[prev_mode_index].type == "goto":
                pass
            else:
                self.C.attach(
                    self.tasks[prev_mode_index].frames[0],
                    self.tasks[prev_mode_index].frames[1],
                )

            # postcondition
            if self.tasks[prev_mode_index].side_effect is not None:
                box = self.tasks[prev_mode_index].frames[1]
                self.C.delFrame(box)

            if i == current_mode:
                break


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


class rai_two_dim_env(rai_env):
    def __init__(self):
        self.C, keyframes = make_2d_rai_env()
        # self.C.view(True)

        print("keyframes")
        print(keyframes)

        self.robots = ["a1", "a2"]

        super().__init__()

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

        # # TODO: this should eventually be replaced by a dependency graph
        self.sequence = self._make_sequence_from_names(
            ["a2_goal", "a1_goal", "terminal"]
        )

        self.start_mode = self._make_start_mode_from_sequence()
        self.terminal_mode = self._make_terminal_mode_from_sequence()

        self.tolerance = 0.1


class rai_two_dim_single_agent_neighbourhood(rai_env):
    pass


class rai_two_dim_two_agents_long_horizon(rai_env):
    pass


class rai_two_dim_many_narrow_passage(rai_env):
    pass


class rai_two_dim_simple_manip(rai_env):
    def __init__(self):
        self.C, keyframes = make_piano_mover_env()
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        super().__init__()

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

        self.sequence = self._make_sequence_from_names(
            ["a2_pick", "a1_pick", "a2_place", "a1_place", "terminal"]
        )
        # self.sequence = [2, 0, 3, 1, 4]

        self.start_mode = self._make_start_mode_from_sequence()
        self.terminal_mode = self._make_terminal_mode_from_sequence()

        self.tolerance = 0.05

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        self.prev_mode = [0, 0]


class rai_two_dim_handover(rai_env):
    def __init__(self):
        self.C, keyframes = make_two_dim_handover()
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        super().__init__()

        self.manipulating_env = True

        translated_handover_poses = []
        for _ in range(100):
            new_pose = keyframes[1] * 1.0
            translation = np.random.rand(2) * 1 - 0.5
            new_pose[0:2] += translation
            new_pose[3:5] += translation

            translated_handover_poses.append(new_pose)

        translated_handover_poses.append(keyframes[1])

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

        self.start_mode = self._make_start_mode_from_sequence()
        self.terminal_mode = self._make_terminal_mode_from_sequence()

        self.tolerance = 0.05

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        self.prev_mode = [0, 0]


class rai_random_two_dim(rai_env):
    def __init__(self):
        num_robots = 3
        num_goals = 4
        self.C, keyframes = make_random_two_dim(
            num_agents=num_robots, num_goals=num_goals, num_obstacles=10
        )
        # self.C.view(True)

        self.robots = [f"a{i}" for i in range(num_robots)]

        super().__init__()

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

        # TODO: there seems to be a bug somewhere if we do not have a terminal mode!
        q_home = self.C.getJointState()
        # self.tasks.append(Task(self.robots, GoalRegion(self.limits)))
        self.tasks.append(Task(self.robots, SingleGoal(q_home)))
        self.tasks[-1].name = "terminal"

        random.shuffle(self.sequence)
        self.sequence.append(len(self.tasks) - 1)

        self.start_mode = self._make_start_mode_from_sequence()
        self.terminal_mode = self._make_terminal_mode_from_sequence()

        print("seq", self.sequence)
        print("terminal", self.terminal_mode)

        self.tolerance = 0.05

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        self.prev_mode = [0, 0]


class rai_hallway_two_dim(rai_env):
    def __init__(self):
        self.C, keyframes = make_two_dim_tunnel_env()
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        super().__init__()

        self.tasks = []
        self.sequence = []

        print(keyframes)

        self.tasks = [
            Task(["a1"], SingleGoal(keyframes[0])),
            Task(["a2"], SingleGoal(keyframes[1])),
            Task(["a1", "a2"], SingleGoal(keyframes[2])),
        ]

        self.tasks[0].name = "a1_goal_1"
        self.tasks[1].name = "a2_goal_1"
        self.tasks[2].name = "terminal"

        self.sequence = [0, 1, 2]

        self.start_mode = self._make_start_mode_from_sequence()
        self.terminal_mode = self._make_terminal_mode_from_sequence()

        self.tolerance = 0.05

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        self.prev_mode = [0, 0]


class rai_two_dim_piano_mover(rai_env):
    pass


class rai_two_dim_three_agent_env(rai_env):
    def __init__(self):
        self.C, keyframes = make_2d_rai_env_3_agents()
        # self.C.view(True)

        self.robots = ["a1", "a2", "a3"]

        super().__init__()

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

        self.start_mode = self._make_start_mode_from_sequence()
        self.terminal_mode = self._make_terminal_mode_from_sequence()

        self.tolerance = 0.1


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
class rai_dual_ur10_arm_env(rai_env):
    def __init__(self):
        self.C, self.keyframes = make_box_sorting_env()

        # more efficient collision scene that only has the collidabe shapes (and the links)
        self.C_coll = ry.Config()
        self.C_coll.addConfigurationCopy(self.C)

        # go through all frames, and delete the ones that are only visual
        # that is, the frames that do not have a child, and are not
        # contact frames
        for f in self.C_coll.frames():
            info = f.info()
            if "shape" in info and info["shape"] == "mesh":
                self.C_coll.delFrame(f.name)

        # self.C_coll.view(True)
        # self.C.view(True)

        # self.C.clear()
        # self.C.addConfigurationCopy(self.C_coll)

        self.robots = ["a1", "a2"]

        super().__init__()

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

        self.start_mode = self._make_start_mode_from_sequence()
        self.terminal_mode = self._make_terminal_mode_from_sequence()

        self.tolerance = 0.1


# goals are poses, more complex sequencing
class rai_dual_ur10_arm_handover_env:
    pass


class rai_multi_panda_arm_waypoint_env(rai_env):
    def __init__(
        self, num_robots: int = 3, num_waypoints: int = 6, shuffle_goals: bool = False
    ):
        self.C, keyframes = make_panda_waypoint_env(
            num_robots=num_robots, num_waypoints=num_waypoints
        )

        # more efficient collision scene that only has the collidabe shapes (and the links)
        self.C_coll = ry.Config()
        self.C_coll.addConfigurationCopy(self.C)

        # go through all frames, and delete the ones that are only visual
        # that is, the frames that do not have a child, and are not
        # contact frames
        for f in self.C_coll.frames():
            info = f.info()
            if "shape" in info and info["shape"] == "mesh":
                self.C_coll.delFrame(f.name)

        self.C.clear()
        self.C.addConfigurationCopy(self.C_coll)

        self.robots = ["a0", "a1", "a2"]
        self.robots = self.robots[:num_robots]

        super().__init__()

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

        self.start_mode = self._make_start_mode_from_sequence()
        self.terminal_mode = self._make_terminal_mode_from_sequence()

        self.tolerance = 0.1


# goals are poses
class rai_quadruple_ur10_arm_spot_welding_env(rai_env):
    def __init__(self, num_pts: int = 2, shuffle_goals: bool = False):
        self.C, keyframes = make_welding_env(view=False, num_pts=num_pts)

        self.C_coll = ry.Config()
        self.C_coll.addConfigurationCopy(self.C)

        # go through all frames, and delete the ones that are only visual
        # that is, the frames that do not have a child, and are not
        # contact frames
        for f in self.C_coll.frames():
            info = f.info()
            if "shape" in info and info["shape"] == "mesh":
                self.C_coll.delFrame(f.name)

        # self.C_coll.view(True)
        # self.C.view(True)

        self.C.clear()
        self.C.addConfigurationCopy(self.C_coll)

        self.robots = ["a1", "a2", "a3", "a4"]

        super().__init__()

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

        self.start_mode = self._make_start_mode_from_sequence()
        self.terminal_mode = self._make_terminal_mode_from_sequence()

        self.tolerance = 0.1


# TODO: enable making this a simpler environment where one can set the number of boxes
class rai_ur10_arm_egg_carton_env(rai_env):
    def __init__(self, num_boxes: int = 9):
        self.C, keyframes = make_egg_carton_env()

        # more efficient collision scene that only has the collidabe shapes (and the links)
        self.C_coll = ry.Config()
        self.C_coll.addConfigurationCopy(self.C)

        # go through all frames, and delete the ones that are only visual
        # that is, the frames that do not have a child, and are not
        # contact frames
        for f in self.C_coll.frames():
            info = f.info()
            if "shape" in info and info["shape"] == "mesh":
                self.C_coll.delFrame(f.name)

        # self.C_coll.view(True)
        # self.C.view(True)

        self.C.clear()
        self.C.addConfigurationCopy(self.C_coll)

        self.robots = ["a1", "a2"]

        super().__init__()

        self.manipulating_env = True

        self.tasks = [
            # a1
            Task(
                ["a1"],
                SingleGoal(keyframes[0][self.robot_idx["a1"]]),
                type="pick",
                frames=["a1_ur_vacuum", "box000"],
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[1][self.robot_idx["a1"]]),
                type="place",
                frames=["table", "box000"],
                side_effect="remove",
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[3][self.robot_idx["a1"]]),
                type="pick",
                frames=["a1_ur_vacuum", "box001"],
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[4][self.robot_idx["a1"]]),
                type="place",
                frames=["table", "box001"],
                side_effect="remove",
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[6][self.robot_idx["a1"]]),
                type="pick",
                frames=["a1_ur_vacuum", "box011"],
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[7][self.robot_idx["a1"]]),
                type="place",
                frames=["table", "box011"],
                side_effect="remove",
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[9][self.robot_idx["a1"]]),
                type="pick",
                frames=["a1_ur_vacuum", "box002"],
            ),
            Task(
                ["a1"],
                SingleGoal(keyframes[10][self.robot_idx["a1"]]),
                type="place",
                frames=["table", "box002"],
                side_effect="remove",
            ),
            # a2
            Task(
                ["a2"],
                SingleGoal(keyframes[12][self.robot_idx["a2"]]),
                type="pick",
                frames=["a2_ur_vacuum", "box021"],
            ),
            Task(
                ["a2"],
                SingleGoal(keyframes[13][self.robot_idx["a2"]]),
                type="place",
                frames=["table", "box021"],
                side_effect="remove",
            ),
            Task(
                ["a2"],
                SingleGoal(keyframes[15][self.robot_idx["a2"]]),
                type="pick",
                frames=["a2_ur_vacuum", "box010"],
            ),
            Task(
                ["a2"],
                SingleGoal(keyframes[16][self.robot_idx["a2"]]),
                type="place",
                frames=["table", "box010"],
                side_effect="remove",
            ),
            Task(
                ["a2"],
                SingleGoal(keyframes[18][self.robot_idx["a2"]]),
                type="pick",
                frames=["a2_ur_vacuum", "box012"],
            ),
            Task(
                ["a2"],
                SingleGoal(keyframes[19][self.robot_idx["a2"]]),
                type="place",
                frames=["table", "box012"],
                side_effect="remove",
            ),
            Task(
                ["a2"],
                SingleGoal(keyframes[21][self.robot_idx["a2"]]),
                type="pick",
                frames=["a2_ur_vacuum", "box022"],
            ),
            Task(
                ["a2"],
                SingleGoal(keyframes[22][self.robot_idx["a2"]]),
                type="place",
                frames=["table", "box022"],
                side_effect="remove",
            ),
            Task(
                ["a2"],
                SingleGoal(keyframes[24][self.robot_idx["a2"]]),
                type="pick",
                frames=["a2_ur_vacuum", "box020"],
            ),
            Task(
                ["a2"],
                SingleGoal(keyframes[25][self.robot_idx["a2"]]),
                type="place",
                frames=["table", "box020"],
                side_effect="remove",
            ),
            # terminal
            Task(
                ["a1", "a2"],
                SingleGoal(
                    np.concatenate(
                        [
                            keyframes[11][self.robot_idx["a1"]],
                            keyframes[26][self.robot_idx["a2"]],
                        ]
                    )
                ),
            ),
        ]

        # really ugly way to construct this
        num_a1_tasks = 8
        self.sequence = []
        for i in range(8):
            self.sequence.append(i)
            self.sequence.append(i + num_a1_tasks)

        self.sequence.append(16)
        self.sequence.append(17)
        self.sequence.append(18)

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        self.start_mode = self._make_start_mode_from_sequence()
        self.terminal_mode = self._make_terminal_mode_from_sequence()

        self.prev_mode = self.start_mode.copy()

        self.tolerance = 0.1

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
                frames=["a1_ur_vacuum", "box100"],
            ),
            Task(
                ["a1"],
                SingleGoal(self.keyframes[1][self.robot_idx["a1"]]),
                type="place",
                frames=["table", "box100"],
                side_effect="remove",
            ),
            Task(
                ["a2"],
                SingleGoal(self.keyframes[3][self.robot_idx["a2"]]),
                type="pick",
                frames=["a2_ur_vacuum", "box101"],
            ),
            Task(
                ["a2"],
                SingleGoal(self.keyframes[4][self.robot_idx["a2"]]),
                type="place",
                frames=["table", "box101"],
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

        self.start_mode = self._make_start_mode_from_sequence()
        self.terminal_mode = self._make_terminal_mode_from_sequence()

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        # buffer for faster collision checking
        self.prev_mode = [0, 0]


class rai_ur10_box_sort_env:
    pass


class rai_ur10_palletizing_env:
    pass


class rai_ur10_arm_shelf_env:
    pass


class rai_ur10_arm_conveyor_env:
    pass


class rai_ur10_handover_env(rai_env):
    def __init__(self):
        self.C, keyframes = make_handover_env()

        # more efficient collision scene that only has the collidabe shapes (and the links)
        self.C_coll = ry.Config()
        self.C_coll.addConfigurationCopy(self.C)

        # go through all frames, and delete the ones that are only visual
        # that is, the frames that do not have a child, and are not
        # contact frames
        for f in self.C_coll.frames():
            info = f.info()
            if "shape" in info and info["shape"] == "mesh":
                self.C_coll.delFrame(f.name)

        # self.C_coll.view(True)
        # self.C.view(True)

        self.C.clear()
        self.C.addConfigurationCopy(self.C_coll)

        self.robots = ["a1", "a2"]

        super().__init__()

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

        self.start_mode = self._make_start_mode_from_sequence()
        self.terminal_mode = self._make_terminal_mode_from_sequence()

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode.copy()

        self.tolerance = 0.1

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)


class rai_ur10_arm_bottle_env(rai_env):
    def __init__(self):
        self.C, keyframes = make_bottle_insertion()

        # more efficient collision scene that only has the collidabe shapes (and the links)
        self.C_coll = ry.Config()
        self.C_coll.addConfigurationCopy(self.C)

        # go through all frames, and delete the ones that are only visual
        # that is, the frames that do not have a child, and are not
        # contact frames
        for f in self.C_coll.frames():
            info = f.info()
            if "shape" in info and info["shape"] == "mesh":
                self.C_coll.delFrame(f.name)

        # self.C_coll.view(True)
        # self.C.view(True)

        self.C.clear()
        self.C.addConfigurationCopy(self.C_coll)

        self.robots = ["a0", "a1"]

        super().__init__()

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

        self.start_mode = self._make_start_mode_from_sequence()
        self.terminal_mode = self._make_terminal_mode_from_sequence()

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        # buffer for faster collision checking
        self.prev_mode = self.start_mode.copy()

        self.tolerance = 0.1

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)


# mobile manip
class rai_mobile_manip_wall:
    pass


# TODO: make rai-independent
def visualize_modes(env: rai_env):
    env.show()

    q_home = env.start_pos

    m = env.start_mode
    for i in range(len(env.sequence)):
        print("mode", m)
        switching_robots = env.get_goal_constrained_robots(m)

        q = []
        task = env.get_active_task(m)
        goal_sample = task.goal.sample()

        print(task.name)
        print(goal_sample)

        print("switching robots: ", switching_robots)

        for j, r in enumerate(env.robots):
            if r in switching_robots:
                # TODO: need to check all goals here
                # figure out where robot r is in the goal description
                offset = 0
                for _, task_robot in enumerate(task.robots):
                    if task_robot == r:
                        q.append(
                            goal_sample[offset : offset + env.robot_dims[task_robot]]
                        )
                        break
                    offset += env.robot_dims[task_robot]
                # q.append(goal_sample)
            else:
                q.append(q_home.robot_state(j))

        print(q)

        print(
            "is collision free: ",
            env.is_collision_free(type(env.get_start_pos()).from_list(q).state(), m),
        )

        # colls = env.C.getCollisions()
        # for c in colls:
        #     if c[2] < 0:
        #         print(c)

        env.show()

        if m == env.terminal_mode:
            break

        m = env.get_next_mode(None, m)


def display_path(
    env,
    path: List[State],
    stop: bool = True,
    export: bool = False,
    pause_time: float = 0.01,
) -> None:
    for i in range(len(path)):
        env.set_to_mode(path[i].mode)
        for k in range(len(env.robots)):
            q = path[i].q[k]
            env.C.setJointState(q, get_robot_joints(env.C, env.robots[k]))

        env.C.view(stop)

        if export:
            env.C.view_savePng("./z.vid/")

        time.sleep(pause_time)


def benchmark_collision_checking(env: rai_env, N=5000):
    start = time.time()
    for _ in range(N):
        q = []
        for i in range(len(env.robots)):
            lims = env.limits[:, env.robot_idx[env.robots[i]]]
            if lims[0, 0] < lims[1, 0]:
                qr = (
                    np.random.rand(env.robot_dims[env.robots[i]])
                    * (lims[1, :] - lims[0, :])
                    + lims[0, :]
                )
            else:
                qr = np.random.rand(env.robot_dims[env.robots[i]]) * 6 - 3
            q.append(qr)

        m = env.sample_random_mode()

        env.is_collision_free(type(env.get_start_pos()).from_list(q).state(), m)

    end = time.time()

    print(f"Took on avg. {(end-start)/N * 1000} ms for a collision check.")


def get_env_by_name(name):
    if name == "piano":
        env = rai_two_dim_simple_manip()
    elif name == "simple_2d":
        env = rai_two_dim_env()
    elif name == "hallway":
        env = rai_hallway_two_dim()
    elif name == "random_2d":
        env = rai_random_two_dim()
    elif name == "2d_handover":
        env = rai_two_dim_handover()
    elif name == "three_agents":
        env = rai_two_dim_three_agent_env()
    elif name == "box_sorting":
        env = rai_ur10_arm_pick_and_place_env()
    elif name == "eggs":
        env = rai_ur10_arm_egg_carton_env()
    elif name == "triple_waypoints":
        env = rai_multi_panda_arm_waypoint_env(num_robots=3, num_waypoints=5)
    elif name == "welding":
        env = rai_quadruple_ur10_arm_spot_welding_env()
    elif name == "bottles":
        env = rai_ur10_arm_bottle_env()
    elif name == "handover":
        env = rai_ur10_handover_env()
    else:
        raise NotImplementedError("Name does not exist")

    return env


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
        env = get_env_by_name(env_name)
        q_home = env.start_pos
        m = env.start_mode
        for i in range(len(env.sequence)):
            if m == env.terminal_mode:
                switching_robots = [r for r in env.robots]
            else:
                # find the robot(s) that needs to switch the mode
                switching_robots = env.get_goal_constrained_robots(m)

            q = []
            task = env.get_active_task(m)
            goal_sample = task.goal.sample()

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
                type(env.get_start_pos()).from_list(q).state(), m
            )
            print(f"mode {m} is collision free: ", is_collision_free)

            env.show()

            colls = env.C.getCollisions()
            for c in colls:
                if c[2] < 0:
                    print(c)

            if not is_collision_free:
                raise ValueError()

            m = env.get_next_mode(None, m)


def export_env(env):
    pass


def load_env_from_file(filepath):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Env shower")
    parser.add_argument("env_name", nargs="?", default="default", help="env to show")
    parser.add_argument(
        "--mode",
        choices=["benchmark", "show", "modes"],
        required=True,
        help="Select the mode of operation",
    )
    args = parser.parse_args()

    # check_all_modes()

    env = get_env_by_name(args.env_name)

    if args.mode == "show":
        print("Environment starting position")
        env.show()
    elif args.mode == "benchmark":
        benchmark_collision_checking(env)
    elif args.mode == "modes":
        print("Environment modes/goals")
        visualize_modes(env)
