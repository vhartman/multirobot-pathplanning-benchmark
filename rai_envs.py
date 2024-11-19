import robotic as ry
import numpy as np
import random
import argparse

from typing import List
from numpy.typing import NDArray

# from dependency_graph import DependencyGraph

from rai_config import *
from planning_env import *
from util import *


def get_robot_joints(C: ry.Config, prefix: str) -> List[str]:
    links = []

    for name in C.getJointNames():
        if prefix in name:
            name = name.split(":")[0]

            if name not in links:
                links.append(name)

    return links


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
    C: ry.Config
    goals: dict[str]
    robots: List[str]
    robot_dims: dict[str]
    start_pos: Configuration
    bounds: NDArray
    sequence: List
    mode_sequence: List
    start_mode: NDArray
    terminal_mode: NDArray
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

        self.modes = None

        self.limits = self.C.getJointLimits()

        self.tolerance = 0.1

    def done(self, q: Configuration, m):
        if not np.array_equal(m, self.terminal_mode):
            return False

        for i, r in enumerate(self.robots):
            if not self.goals[r][-1].satisfies_constraints(
                q.robot_state(i), self.tolerance
            ):
                return False

        return True

    def show_config(self, q):
        self.C.setJointState(q)
        self.C.view(True)

    def show(self):
        self.C.view(True)

    def is_transition(self, q: Configuration, m):
        if np.array_equal(m, self.terminal_mode):
            return False

        # find the next thing that has to be satisfied from the goal sequence
        for i, mode in enumerate(self.mode_sequence):
            if np.array_equal(m, mode):
                next_mode = self.mode_sequence[i + 1]

                for i in range(len(self.robots)):
                    if next_mode[i] != m[i]:
                        # robot 1 needs to do smth
                        if self.goals[self.robots[i]][m[i]].satisfies_constraints(
                            q.robot_state(i), self.tolerance
                        ):
                            return True

        return False

    def get_next_mode(self, q: Configuration, m):
        for i, mode in enumerate(self.mode_sequence):
            if np.array_equal(m, mode):
                return np.array(self.mode_sequence[i + 1])

        raise ValueError("No next mode found, this might be the terminal mode.")

    def is_collision_free(self, q: Configuration, m, collision_tolerance: float = 0.01):
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
        m,
        resolution=0.1,
        randomize_order=True,
    ):
        # print('q1', q1)
        # print('q2', q2)
        N = config_dist(q1, q2) / resolution
        N = max(5, N)

        idx = list(range(int(N)))
        if randomize_order:
            random.shuffle(idx)

        for i in idx:
            # print(i / (N-1))
            q = q1.state() + (q2.state() - q1.state()) * (i) / (N - 1)
            if not self.is_collision_free(q, m):
                # print('coll')
                return False

        return True

    def is_path_collision_free(self, path: List[State], randomize_order=True):
        idx = list(range(len(path) - 1))
        if randomize_order:
            random.shuffle(idx)

        for i in idx:
            # skip transition nodes
            if not np.array_equal(path[i].mode, path[i + 1].mode):
                continue

            q1 = path[i].q
            q2 = path[i + 1].q
            mode = path[i].mode

            if not self.is_edge_collision_free(q1, q2, mode):
                return False

        return True

    def set_to_mode(self, m):
        if self.modes is None:
            return

        # do not remake the config if we are in the same mode as we have been before
        if np.array_equal(m, self.prev_mode):
            return

        self.prev_mode = m

        # self.C.view(True)
        self.C.clear()
        self.C.addConfigurationCopy(self.C_base)

        # find current mode
        current_mode = 0
        for i, mode in enumerate(self.mode_sequence):
            if np.array_equal(m, mode):
                current_mode = i
                break

        if current_mode == 0:
            return

        for i, mode in enumerate(self.mode_sequence):
            if i == 0:
                continue

            # figure out which robot switched from the previous mode to this mode
            # TODO: this does currently not work for multiple robots switching at the same time
            mode_switching_robot = 0
            for r in range(len(self.robots)):
                if mode[r] != self.mode_sequence[i - 1][r]:
                    # this mode switched
                    mode_switching_robot = r
                    break

            # set robot to config
            mode_index = mode[mode_switching_robot]
            robot = self.robots[mode_switching_robot]
            q = self.goals[robot][mode_index - 1].goal * 1.0

            # self.C.view(True)

            self.C.setJointState(q, get_robot_joints(self.C, robot))

            # self.C.view(True)

            # print(self.modes[robot][i-1])
            if self.modes[robot][mode_index - 1][0] == "goto":
                pass
            else:
                self.C.attach(
                    self.modes[robot][mode_index - 1][1],
                    self.modes[robot][mode_index - 1][2],
                )

            # postcondition
            # TODO: I am not sure if this does not lead to issues later on
            if self.modes[robot][mode_index - 1][3] is not None:
                box = self.modes[robot][mode_index - 1][2]
                self.C.delFrame(box)

            # print(mode_switching_robot, current_mode)
            # self.C.view(True)

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

        self.robots = ["a1", "a2"]

        super().__init__()

        self.goals = {
            "a1": [SingleGoal(keyframes[0][self.robot_idx["a1"]]), 
                   SingleGoal(keyframes[0][self.robot_idx["a1"]])],
            "a2": [
                SingleGoal(keyframes[1][self.robot_idx["a2"]]),
                SingleGoal(keyframes[2][self.robot_idx["a2"]]),
            ],
        }

        # corresponds to
        # a1  0
        # a2 0 1
        self.sequence = [("a2", 0), ("a1", 0)]
        self.mode_sequence = make_mode_sequence_from_sequence(
            self.robots, self.sequence
        )

        self.start_mode = np.array([0, 0])
        self.terminal_mode = np.array([0, 1])

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

        self.goals = {
            "a1": [
                SingleGoal(keyframes[0][self.robot_idx["a1"]]),
                SingleGoal(keyframes[1][self.robot_idx["a1"]]),
                SingleGoal(keyframes[2][self.robot_idx["a1"]]),
            ],
            "a2": [
                SingleGoal(keyframes[0][self.robot_idx["a2"]]),
                SingleGoal(keyframes[1][self.robot_idx["a2"]]),
                SingleGoal(keyframes[2][self.robot_idx["a2"]]),
            ],
        }

        self.modes = {
            "a1": [
                ("pick", "a1", "obj1", None),
                ("place", "table", "obj1", None),
                ("goto", None, None),
            ],
            "a2": [
                ("pick", "a2", "obj2", None),
                ("place", "table", "obj2", None),
                ("goto", 0, 0, None),
            ],
        }

        # corresponds to
        # a1  0
        # a2 0 1
        self.sequence = [("a2", 0), ("a1", 0), ("a2", 1), ("a1", 1)]
        # self.sequence = [("a2", 0), ("a2", 1), ("a1", 0), ("a1", 1)]
        # self.sequence = [("a1", 0), ("a2", 0), ("a2", 1), ("a2", 1)]
        self.mode_sequence = make_mode_sequence_from_sequence(
            self.robots, self.sequence
        )

        print(self.mode_sequence)
        # self.C.view(True)

        self.start_mode = np.array(self.mode_sequence[0])
        self.terminal_mode = np.array(self.mode_sequence[-1])

        self.tolerance = 0.1

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        self.prev_mode = np.array([0, 0])


class rai_two_dim_piano_mover(rai_env):
    pass


class rai_two_dim_three_agent_env(rai_env):
    def __init__(self):
        self.C, keyframes = make_2d_rai_env_3_agents()
        # self.C.view(True)

        self.robots = ["a1", "a2", "a3"]

        super().__init__()

        self.goals = {
            "a1": [
                SingleGoal(keyframes[0][self.robot_idx["a1"]]),
                SingleGoal(keyframes[4][self.robot_idx["a1"]]),
                SingleGoal(keyframes[5][self.robot_idx["a1"]]),
            ],
            "a2": [
                SingleGoal(keyframes[1][self.robot_idx["a2"]]),
                SingleGoal(keyframes[3][self.robot_idx["a2"]]),
                SingleGoal(keyframes[5][self.robot_idx["a2"]]),
            ],
            "a3": [
                SingleGoal(keyframes[2][self.robot_idx["a3"]]),
                SingleGoal(keyframes[5][self.robot_idx["a3"]]),
            ],
        }

        self.sequence = [("a1", 0), ("a2", 0), ("a3", 0), ("a2", 1), ("a1", 1)]
        self.mode_sequence = make_mode_sequence_from_sequence(
            self.robots, self.sequence
        )

        # self.mode_sequence = [
        #     [0, 0, 0],
        #     [1, 0, 0],
        #     [1, 1, 0],
        #     [1, 1, 1],
        #     [1, 2, 1],
        #     [2, 2, 1],
        # ]

        self.start_mode = np.array(self.mode_sequence[0])
        self.terminal_mode = np.array(self.mode_sequence[-1])

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
        self.C, keyframes = make_box_sorting_env()

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

        self.goals = {
            "a1": [
                SingleGoal(keyframes[0][self.robot_idx["a1"]]),
                SingleGoal(keyframes[1][self.robot_idx["a1"]]),
                SingleGoal(keyframes[2][self.robot_idx["a1"]]),
            ],
            "a2": [
                SingleGoal(keyframes[3][self.robot_idx["a2"]]),
                SingleGoal(keyframes[4][self.robot_idx["a2"]]),
                SingleGoal(keyframes[5][self.robot_idx["a2"]]),
            ],
        }

        self.sequence = [("a1", 0), ("a2", 0), ("a1", 1), ("a2", 1)]
        self.mode_sequence = make_mode_sequence_from_sequence(
            self.robots, self.sequence
        )

        # self.mode_sequence = [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]]

        self.start_mode = np.array(self.mode_sequence[0])
        self.terminal_mode = np.array(self.mode_sequence[-1])

        self.tolerance = 0.1

        # print(self.goals["a1"][0].goal)
        # self.C.setJointState(self.goals["a1"][0].goal, get_robot_joints(self.C, "a1"))
        # self.C.view(True)


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

        # self.C_coll.view(True)
        # self.C.view(True)

        self.C.clear()
        self.C.addConfigurationCopy(self.C_coll)

        self.robots = ["a0", "a1", "a2"]
        self.robots = self.robots[:num_robots]

        super().__init__()

        self.goals = {}
        for r in self.robots:
            self.goals[r] = []

        cnt = 0
        for r in self.robots:
            self.goals[r] = [
                SingleGoal(keyframes[cnt + i][self.robot_idx[r]])
                for i in range(num_waypoints + 1)
            ]
            cnt += num_waypoints + 1

        # permute goals, but only the ones that ware waypoints, not the final configuration
        if shuffle_goals:
            for r in self.robots:
                sublist = self.goals[r][:num_waypoints]
                random.shuffle(sublist)
                self.goals[r][:num_waypoints] = sublist

        self.sequence = []
        for i in range(num_waypoints):
            for r in self.robots:
                self.sequence.append((r, i))

        self.mode_sequence = make_mode_sequence_from_sequence(
            self.robots, self.sequence
        )

        self.start_mode = np.array(self.mode_sequence[0])
        self.terminal_mode = np.array(self.mode_sequence[-1])

        self.tolerance = 0.1


# goals are poses
class rai_quadruple_ur10_arm_spot_welding_env(rai_env):
    def __init__(self, num_pts: int = 4):
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

        self.goals = {}
        for r in self.robots:
            self.goals[r] = []

        cnt = 0
        for r in self.robots:
            for _ in range(num_pts + 1):
                self.goals[r].append(SingleGoal(keyframes[cnt][self.robot_idx[r]]))
                cnt += 1

        # self.goals = {
        #     "a1": [
        #         SingleGoal(keyframes[0][self.robot_idx["a1"]]),
        #         SingleGoal(keyframes[1][self.robot_idx["a1"]]),
        #         SingleGoal(keyframes[2][self.robot_idx["a1"]]),
        #         SingleGoal(keyframes[3][self.robot_idx["a1"]]),
        #         SingleGoal(keyframes[4][self.robot_idx["a1"]]),
        #     ],
        #     "a2": [
        #         SingleGoal(keyframes[5][self.robot_idx["a2"]]),
        #         SingleGoal(keyframes[6][self.robot_idx["a2"]]),
        #         SingleGoal(keyframes[7][self.robot_idx["a2"]]),
        #         SingleGoal(keyframes[8][self.robot_idx["a2"]]),
        #         SingleGoal(keyframes[9][self.robot_idx["a2"]]),
        #     ],
        #     "a3": [
        #         SingleGoal(keyframes[10][self.robot_idx["a3"]]),
        #         SingleGoal(keyframes[11][self.robot_idx["a3"]]),
        #         SingleGoal(keyframes[12][self.robot_idx["a3"]]),
        #         SingleGoal(keyframes[13][self.robot_idx["a3"]]),
        #         SingleGoal(keyframes[14][self.robot_idx["a3"]]),
        #     ],
        #     "a4": [
        #         SingleGoal(keyframes[15][self.robot_idx["a4"]]),
        #         SingleGoal(keyframes[16][self.robot_idx["a4"]]),
        #         SingleGoal(keyframes[17][self.robot_idx["a4"]]),
        #         SingleGoal(keyframes[18][self.robot_idx["a4"]]),
        #         SingleGoal(keyframes[19][self.robot_idx["a4"]]),
        #     ],
        # }

        self.sequence = []
        for i in range(num_pts):
            for r in self.robots:
                self.sequence.append((r, i))

        self.mode_sequence = make_mode_sequence_from_sequence(
            self.robots, self.sequence
        )

        self.start_mode = np.array(self.mode_sequence[0])
        self.terminal_mode = np.array(self.mode_sequence[-1])

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

        self.goals = {
            "a1": [
                SingleGoal(keyframes[0][self.robot_idx["a1"]]),
                SingleGoal(keyframes[1][self.robot_idx["a1"]]),
                # SingleGoal(keyframes[2][self.robot_idx["a1"]]),
                SingleGoal(keyframes[3][self.robot_idx["a1"]]),
                SingleGoal(keyframes[4][self.robot_idx["a1"]]),
                # SingleGoal(keyframes[5][self.robot_idx["a1"]]),
                SingleGoal(keyframes[6][self.robot_idx["a1"]]),
                SingleGoal(keyframes[7][self.robot_idx["a1"]]),
                # SingleGoal(keyframes[8][self.robot_idx["a1"]]),
                SingleGoal(keyframes[9][self.robot_idx["a1"]]),
                SingleGoal(keyframes[10][self.robot_idx["a1"]]),
                SingleGoal(keyframes[11][self.robot_idx["a1"]]),
            ],
            "a2": [
                SingleGoal(keyframes[12][self.robot_idx["a2"]]),
                SingleGoal(keyframes[13][self.robot_idx["a2"]]),
                # SingleGoal(keyframes[14][self.robot_idx["a2"]]),
                SingleGoal(keyframes[15][self.robot_idx["a2"]]),
                SingleGoal(keyframes[16][self.robot_idx["a2"]]),
                # SingleGoal(keyframes[17][self.robot_idx["a2"]]),
                SingleGoal(keyframes[18][self.robot_idx["a2"]]),
                SingleGoal(keyframes[19][self.robot_idx["a2"]]),
                # SingleGoal(keyframes[20][self.robot_idx["a2"]]),
                SingleGoal(keyframes[21][self.robot_idx["a2"]]),
                SingleGoal(keyframes[22][self.robot_idx["a2"]]),
                # SingleGoal(keyframes[23][self.robot_idx["a2"]]),
                SingleGoal(keyframes[24][self.robot_idx["a2"]]),
                SingleGoal(keyframes[25][self.robot_idx["a2"]]),
                SingleGoal(keyframes[26][self.robot_idx["a2"]]),
            ],
        }

        self.sequence = [
            ("a1", 0),
            ("a2", 0),
            ("a1", 1),
            ("a2", 1),
            ("a1", 2),
            ("a2", 2),
            ("a1", 3),
            ("a2", 3),
            ("a1", 4),
            ("a2", 4),
            ("a1", 5),
            ("a2", 5),
            ("a1", 6),
            ("a2", 6),
            ("a1", 7),
            ("a2", 7),
            ("a2", 8),
            ("a2", 9),
        ]
        # self.sequence = self.sequence[:5]
        self.mode_sequence = make_mode_sequence_from_sequence(
            self.robots, self.sequence
        )

        # a1_boxes = ["box000", "box001", "box011", "box002"]
        # a2_boxes = ["box021", "box010", "box012", "box022", "box020"]

        self.modes = {
            "a1": [
                ("pick", "a1_ur_vacuum", "box000", None),
                ("place", "table", "box000", "remove"),
                ("pick", "a1_ur_vacuum", "box001", None),
                ("place", "table", "box001", "remove"),
                ("pick", "a1_ur_vacuum", "box011", None),
                ("place", "table", "box011", "remove"),
                ("pick", "a1_ur_vacuum", "box002", None),
                ("place", "table", "box002", "remove"),
                ("goto", None, None),
            ],
            "a2": [
                ("pick", "a2_ur_vacuum", "box021", None),
                ("place", "table", "box021", "remove"),
                ("pick", "a2_ur_vacuum", "box010", None),
                ("place", "table", "box010", "remove"),
                ("pick", "a2_ur_vacuum", "box012", None),
                ("place", "table", "box012", "remove"),
                ("pick", "a2_ur_vacuum", "box022", None),
                ("place", "table", "box022", "remove"),
                ("pick", "a2_ur_vacuum", "box020", None),
                ("place", "table", "box020", "remove"),
                ("goto", 0, 0, None),
            ],
        }

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        # buffer for faster collision checking
        self.prev_mode = np.array([0, 0])
        self.start_mode = np.array(self.mode_sequence[0])
        self.terminal_mode = np.array(self.mode_sequence[-1])

        self.tolerance = 0.1

        # q = self.C_base.getJointState()
        # print(self.is_collision_free(q, [0, 0]))

        # for m in self.mode_sequence:
        #     print(self.is_collision_free(q, m))
        #     self.show()

        # self.C_base.view(True)


class rai_ur10_arm_pick_and_place_env(rai_dual_ur10_arm_env):
    def __init__(self):
        super().__init__()

        self.modes = {
            "a1": [
                ("pick", "a1_ur_vacuum", "box100", None),
                ("place", "table", "box100", "remove"),
                ("goto", None, None),
            ],
            "a2": [
                ("pick", "a2_ur_vacuum", "box101", None),
                ("place", "table", "box101", "remove"),
                ("goto", 0, 0, None),
            ],
        }

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        self.sequence = [("a1", 0), ("a2", 0), ("a1", 1), ("a2", 1)]
        self.mode_sequence = make_mode_sequence_from_sequence(
            self.robots, self.sequence
        )

        # buffer for faster collision checking
        self.prev_mode = np.array([0, 0])


class rai_ur10_box_sort_env:
    pass


class rai_ur10_palletizing_env:
    pass


class rai_ur10_arm_shelf_env:
    pass


class rai_ur10_arm_conveyor_env:
    pass


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

        self.goals = {
            "a0": [
                SingleGoal(keyframes[0][self.robot_idx["a0"]]),
                SingleGoal(keyframes[1][self.robot_idx["a0"]]),
                # SingleGoal(keyframes[2][self.robot_idx["a1"]]),
                SingleGoal(keyframes[3][self.robot_idx["a0"]]),
                SingleGoal(keyframes[4][self.robot_idx["a0"]]),
                SingleGoal(keyframes[5][self.robot_idx["a0"]]),
            ],
            "a1": [
                SingleGoal(keyframes[6][self.robot_idx["a1"]]),
                SingleGoal(keyframes[7][self.robot_idx["a1"]]),
                # SingleGoal(keyframes[8][self.robot_idx["a1"]]),
                SingleGoal(keyframes[9][self.robot_idx["a1"]]),
                SingleGoal(keyframes[10][self.robot_idx["a1"]]),
                SingleGoal(keyframes[11][self.robot_idx["a1"]]),
            ],
        }

        for k, v in self.goals.items():
            for g in v:
                print(g.goal)

        self.sequence = [
            ("a0", 0),
            ("a1", 0),
            ("a0", 1),
            ("a1", 1),
            ("a0", 2),
            ("a1", 2),
            ("a0", 3),
            ("a1", 3),
        ]
        # self.sequence = self.sequence[:5]
        self.mode_sequence = make_mode_sequence_from_sequence(
            self.robots, self.sequence
        )

        # a1_boxes = ["box000", "box001", "box011", "box002"]
        # a2_boxes = ["box021", "box010", "box012", "box022", "box020"]

        self.modes = {
            "a0": [
                ("pick", "a0_ur_vacuum", "bottle_1", None),
                ("place", "table", "bottle_1", None),
                ("pick", "a0_ur_vacuum", "bottle_12", None),
                ("place", "table", "bottle_12", None),
                ("goto", None, None),
            ],
            "a1": [
                ("pick", "a1_ur_vacuum", "bottle_3", None),
                ("place", "table", "bottle_3", None),
                ("pick", "a1_ur_vacuum", "bottle_5", None),
                ("place", "table", "bottle_5", None),
                ("goto", 0, 0, None),
            ],
        }

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        # buffer for faster collision checking
        self.prev_mode = np.array([0, 0])
        self.start_mode = np.array(self.mode_sequence[0])
        self.terminal_mode = np.array(self.mode_sequence[-1])

        self.tolerance = 0.1

        # q = self.C_base.getJointState()
        # print(self.is_collision_free(q, [0, 0]))

        # for m in self.mode_sequence:
        #     print(self.is_collision_free(q, m))
        #     self.show()

        # self.C_base.view(True)

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        # buffer for faster collision checking
        self.prev_mode = np.array([0, 0])


# mobile manip
class rai_mobile_manip_wall:
    pass


# TODO: make rai-independent
def visualize_modes(env):
    q_home = env.start_pos

    for i, m in enumerate(env.mode_sequence):
        # figure out which are the robots that change their mode
        if i < len(env.mode_sequence) - 1:
            m_next = env.mode_sequence[i + 1]

            switching_robots = []
            for j, r in enumerate(env.robots):
                if m[j] != m_next[j]:
                    switching_robots.append(r)
        else:
            switching_robots = env.robots

        q = []
        for j, r in enumerate(env.robots):
            if r in switching_robots:
                q.append(env.goals[r][m[j]].goal)
            else:
                q.append(q_home.robot_state(j))

        print("is collision free: ", env.is_collision_free(type(env.get_start_pos()).from_list(q).state(), m))

        # colls = env.C.getCollisions()
        # for c in colls:
        #     if c[2] < 0:
        #         print(c)

        env.show()

    env.C.view(True)


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

        m = random.choice(env.mode_sequence)

        env.is_collision_free(type(env.get_start_pos()).from_list(q).state(), m)

    end = time.time()

    print(f"Took on avg. {(end-start)/N * 1000} ms for a collision check.")


def get_env_by_name(name):
    if name == "piano":
        env = rai_two_dim_simple_manip()
    elif name == "simple_2d":
        env = rai_two_dim_env()
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
    else:
        raise NotImplementedError("Name does not exist")

    return env


def check_all_modes():
    all_envs = ["piano", "simple_2d", "three_agents", "box_sorting", "eggs", "triple_waypoints", "welding", "bottles"]

    for env_name in all_envs:
        print(env_name)
        env = get_env_by_name(env_name)
        q_home = env.start_pos
        for i, m in enumerate(env.mode_sequence):
            # figure out which are the robots that change their mode
            if i < len(env.mode_sequence) - 1:
                m_next = env.mode_sequence[i + 1]

                switching_robots = []
                for j, r in enumerate(env.robots):

                    if m[j] != m_next[j]:
                        switching_robots.append(r)
            else:
                switching_robots = env.robots

            q = []
            for j, r in enumerate(env.robots):
                if r in switching_robots:
                    q.append(env.goals[r][m[j]].goal)
                else:
                    q.append(q_home.robot_state(j))

            is_collision_free = env.is_collision_free(type(env.get_start_pos()).from_list(q).state(), m)
            if not is_collision_free:
                raise ValueError()
            
            print(f"mode {m} is collision free: ", is_collision_free)
    
            env.show()

            # colls = env.C.getCollisions()
            # for c in colls:
            #     if c[2] < 0:
            #         print(c)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Env shower")
    parser.add_argument("env_name", nargs="?", default="default", help="env to show")
    parser.add_argument(
        '--mode', 
        choices=['benchmark', 'show', 'modes'], 
        required=True, 
        help="Select the mode of operation"
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
