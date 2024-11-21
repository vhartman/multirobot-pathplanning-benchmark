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
    robots: List[str]
    robot_dims: dict[str]
    start_pos: Configuration
    bounds: NDArray
    sequence: List
    start_mode: List[int]
    terminal_mode: List[int]
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

    def sample_random_mode(self):
        m = self.start_mode
        rnd = random.randint(0, len(self.sequence))

        for _ in range(rnd):
            m = self.get_next_mode(_, m)

        return m

    def get_current_seq_index(self, mode):
        # min_sequence_pos = 0
        # for i, m in enumerate(mode):
        #     min_sequence_pos = min(self.sequence.index(m), min_sequence_pos))

        # involved_robots = self.modes[min_sequence_pos].robots
        # return involved_robots

        translated_mode = [(self.robots[i], int(ind)) for i, ind in enumerate(mode)]
        min_sequence_pos = len(self.sequence)
        for i, m in enumerate(translated_mode):
            if m in self.sequence:
                ind = self.sequence.index(m)
                # print(m, ind)
                min_sequence_pos = min(ind, min_sequence_pos)

            # else:
            #     print('element not in list')

        return min_sequence_pos

    def get_goal_constrained_robots(self, mode):
        min_sequence_pos = self.get_current_seq_index(mode)
        involved_robots = [self.sequence[min_sequence_pos][0]]

        # print(mode, involved_robots)
        return involved_robots

    def done(self, q: Configuration, m: List[int]):
        if not np.array_equal(m, self.terminal_mode):
            return False

        for i, r in enumerate(self.robots):
            if not self.robot_goals[r][-1].goal.satisfies_constraints(
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

        robots_with_constraints_in_current_mode = self.get_goal_constrained_robots(m)

        for r in robots_with_constraints_in_current_mode:
            robot_idx = self.robots.index(r)
            if not self.robot_goals[r][m[robot_idx]].goal.satisfies_constraints(
                q.robot_state(robot_idx), self.tolerance
            ):
                return False
        return True

    def get_next_mode(self, q: Configuration, mode):
        seq_pos = self.get_current_seq_index(mode)

        # find the next mode for the currently constrained one(s)
        r = self.sequence[seq_pos][0]
        r_idx = self.robots.index(r)

        m_next = mode.copy()
        m_next[r_idx] += 1

        return m_next
        # raise ValueError("No next mode found, this might be the terminal mode.")

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

            # figure out which robot switched from the previous mode to this mode
            # TODO: this does currently not work for multiple robots switching at the same time
            # robot_new = self.get_goal_constrained_robots(mode)[0]
            # robot_idx = self.robots.index(robot_new)

            mode_switching_robot = 0
            for r in range(len(self.robots)):
                if mode[r] != m_prev[r]:
                    # this mode switched
                    mode_switching_robot = r
                    break

            # print(robot_idx, mode_switching_robot)

            # set robot to config
            mode_index = mode[mode_switching_robot]
            robot = self.robots[mode_switching_robot]

            # TODO: ensure that se are choosing the correct goal here
            q = self.robot_goals[robot][mode_index - 1].goal.goal

            # self.C.view(True)

            self.C.setJointState(q, get_robot_joints(self.C, robot))

            # self.C.view(True)

            if self.robot_goals[robot][mode_index - 1].type == "goto":
                pass
            else:
                self.C.attach(
                    self.robot_goals[robot][mode_index - 1].frames[0],
                    self.robot_goals[robot][mode_index - 1].frames[1],
                )

            # postcondition
            if self.robot_goals[robot][mode_index - 1].side_effect is not None:
                box = self.robot_goals[robot][mode_index - 1].frames[1]
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

        self.robot_goals = {
            "a1": [
                Mode(["a1"], SingleGoal(keyframes[0][self.robot_idx["a1"]])),
                Mode(["a1"], SingleGoal(keyframes[0][self.robot_idx["a1"]])),
            ],
            "a2": [
                Mode(["a2"], SingleGoal(keyframes[1][self.robot_idx["a2"]])),
                Mode(["a2"], SingleGoal(keyframes[2][self.robot_idx["a2"]])),
            ],
        }

        # self.modes = [
        #     # r1
        #     Mode(["a1"], SingleGoal(keyframes[0][self.robot_idx["a1"]])),
        #     # r2
        #     Mode(["a2"], SingleGoal(keyframes[1][self.robot_idx["a2"]])),
        #     # Mode(["a2"], SingleGoal(keyframes[2][self.robot_idx["a2"]])),
        #     # terminal mode
        #     Mode(["a1", "a2"], SingleGoal(np.concatenate([keyframes[0][self.robot_idx["a1"]],
        #                                                   keyframes[2][self.robot_idx["a2"]]])))
        # ]

        # # TODO: this should eventually be replaced by a dependency graph
        # self.sequence = [1, 0, 2]

        # corresponds to
        # a1  0
        # a2 0 1
        self.sequence = [("a2", 0), ("a1", 0)]

        self.start_mode = [0, 0]
        self.terminal_mode = [0, 1]

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

        self.robot_goals = {
            "a1": [
                Mode(
                    ["a1"],
                    SingleGoal(keyframes[0][self.robot_idx["a1"]]),
                    type="pick",
                    frames=["a1", "obj1"],
                    side_effect=None,
                ),
                Mode(
                    ["a1"],
                    SingleGoal(keyframes[1][self.robot_idx["a1"]]),
                    type="place",
                    frames=["table", "obj1"],
                    side_effect=None,
                ),
                Mode(["a1"], SingleGoal(keyframes[2][self.robot_idx["a1"]])),
            ],
            "a2": [
                Mode(
                    ["a2"],
                    SingleGoal(keyframes[0][self.robot_idx["a2"]]),
                    type="pick",
                    frames=["a2", "obj2"],
                    side_effect=None,
                ),
                Mode(
                    ["a2"],
                    SingleGoal(keyframes[1][self.robot_idx["a2"]]),
                    type="place",
                    frames=["table", "obj2"],
                    side_effect=None,
                ),
                Mode(["a2"], SingleGoal(keyframes[2][self.robot_idx["a2"]])),
            ],
        }

        # corresponds to
        # a1  0
        # a2 0 1
        self.sequence = [("a2", 0), ("a1", 0), ("a2", 1), ("a1", 1)]
        # self.sequence = [("a2", 0), ("a2", 1), ("a1", 0), ("a1", 1)]
        # self.sequence = [("a1", 0), ("a2", 0), ("a2", 1), ("a2", 1)]
        # self.C.view(True)

        self.start_mode = [0 for _ in self.robots]
        self.terminal_mode = \
            [len(self.robot_goals[r]) - 1 for r in self.robots]
        

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

        self.robot_goals = {
            "a1": [
                Mode(["a1"], SingleGoal(keyframes[0][self.robot_idx["a1"]])),
                Mode(["a1"], SingleGoal(keyframes[4][self.robot_idx["a1"]])),
                Mode(["a1"], SingleGoal(keyframes[5][self.robot_idx["a1"]])),
            ],
            "a2": [
                Mode(["a2"], SingleGoal(keyframes[1][self.robot_idx["a2"]])),
                Mode(["a2"], SingleGoal(keyframes[3][self.robot_idx["a2"]])),
                Mode(["a2"], SingleGoal(keyframes[5][self.robot_idx["a2"]])),
            ],
            "a3": [
                Mode(["a3"], SingleGoal(keyframes[2][self.robot_idx["a3"]])),
                Mode(["a3"], SingleGoal(keyframes[5][self.robot_idx["a3"]])),
            ],
        }

        self.sequence = [("a1", 0), ("a2", 0), ("a3", 0), ("a2", 1), ("a1", 1)]

        self.start_mode = [0 for _ in self.robots]
        self.terminal_mode = \
            [len(self.robot_goals[r]) - 1 for r in self.robots]

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

        self.robot_goals = {
            "a1": [
                Mode(["a1"], SingleGoal(keyframes[0][self.robot_idx["a1"]])),
                Mode(["a1"], SingleGoal(keyframes[1][self.robot_idx["a1"]])),
                Mode(["a1"], SingleGoal(keyframes[2][self.robot_idx["a1"]])),
            ],
            "a2": [
                Mode(["a2"], SingleGoal(keyframes[3][self.robot_idx["a2"]])),
                Mode(["a2"], SingleGoal(keyframes[4][self.robot_idx["a2"]])),
                Mode(["a2"], SingleGoal(keyframes[5][self.robot_idx["a2"]])),
            ],
        }

        self.sequence = [("a1", 0), ("a2", 0), ("a1", 1), ("a2", 1)]

        self.start_mode = [0 for _ in self.robots]
        self.terminal_mode = \
            [len(self.robot_goals[r]) - 1 for r in self.robots]

        self.tolerance = 0.1

        # print(self.robot_goals["a1"][0].goal)
        # self.C.setJointState(self.robot_goals["a1"][0].goal, get_robot_joints(self.C, "a1"))
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

        self.robot_goals = {}
        for r in self.robots:
            self.robot_goals[r] = []

        cnt = 0
        for r in self.robots:
            self.robot_goals[r] = [
                Mode([r], SingleGoal(keyframes[cnt + i][self.robot_idx[r]]))
                for i in range(num_waypoints + 1)
            ]
            cnt += num_waypoints + 1

        # permute goals, but only the ones that ware waypoints, not the final configuration
        if shuffle_goals:
            for r in self.robots:
                sublist = self.robot_goals[r][:num_waypoints]
                random.shuffle(sublist)
                self.robot_goals[r][:num_waypoints] = sublist

        self.sequence = []
        for i in range(num_waypoints):
            for r in self.robots:
                self.sequence.append((r, i))

        self.start_mode = [0 for _ in self.robots]
        self.terminal_mode = \
            [len(self.robot_goals[r]) - 1 for r in self.robots]

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

        self.robot_goals = {}
        for r in self.robots:
            self.robot_goals[r] = []

        cnt = 0
        for r in self.robots:
            for _ in range(num_pts + 1):
                self.robot_goals[r].append(
                    Mode([r], SingleGoal(keyframes[cnt][self.robot_idx[r]]))
                )
                cnt += 1

        self.sequence = []
        for i in range(num_pts):
            for r in self.robots:
                self.sequence.append((r, i))

        self.start_mode = [0 for _ in self.robots]
        self.terminal_mode = \
            [len(self.robot_goals[r]) - 1 for r in self.robots]

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

        self.robot_goals = {
            "a1": [
                Mode(
                    ["a1"],
                    SingleGoal(keyframes[0][self.robot_idx["a1"]]),
                    type="pick",
                    frames=["a1_ur_vacuum", "box000"],
                    side_effect=None,
                ),
                Mode(
                    ["a1"],
                    SingleGoal(keyframes[1][self.robot_idx["a1"]]),
                    type="place",
                    frames=["table", "box000"],
                    side_effect="remove",
                ),
                # SingleGoal(keyframes[2][self.robot_idx["a1"]]),
                Mode(
                    ["a1"],
                    SingleGoal(keyframes[3][self.robot_idx["a1"]]),
                    type="pick",
                    frames=["a1_ur_vacuum", "box001"],
                    side_effect=None,
                ),
                Mode(
                    ["a1"],
                    SingleGoal(keyframes[4][self.robot_idx["a1"]]),
                    type="place",
                    frames=["table", "box001"],
                    side_effect="remove",
                ),
                # SingleGoal(keyframes[5][self.robot_idx["a1"]]),
                Mode(
                    ["a1"],
                    SingleGoal(keyframes[6][self.robot_idx["a1"]]),
                    type="pick",
                    frames=["a1_ur_vacuum", "box011"],
                    side_effect=None,
                ),
                Mode(
                    ["a1"],
                    SingleGoal(keyframes[7][self.robot_idx["a1"]]),
                    type="place",
                    frames=["table", "box011"],
                    side_effect="remove",
                ),
                # SingleGoal(keyframes[8][self.robot_idx["a1"]]),
                Mode(
                    ["a1"],
                    SingleGoal(keyframes[9][self.robot_idx["a1"]]),
                    type="pick",
                    frames=["a1_ur_vacuum", "box002"],
                    side_effect=None,
                ),
                Mode(
                    ["a1"],
                    SingleGoal(keyframes[10][self.robot_idx["a1"]]),
                    type="place",
                    frames=["table", "box002"],
                    side_effect="remove",
                ),
                Mode(["a1"], SingleGoal(keyframes[11][self.robot_idx["a1"]])),
            ],
            "a2": [
                Mode(
                    ["a2"],
                    SingleGoal(keyframes[12][self.robot_idx["a2"]]),
                    type="pick",
                    frames=["a2_ur_vacuum", "box021"],
                    side_effect=None,
                ),
                Mode(
                    ["a2"],
                    SingleGoal(keyframes[13][self.robot_idx["a2"]]),
                    type="place",
                    frames=["table", "box021"],
                    side_effect="remove",
                ),
                # SingleGoal(keyframes[14][self.robot_idx["a2"]]),
                Mode(
                    ["a2"],
                    SingleGoal(keyframes[15][self.robot_idx["a2"]]),
                    type="pick",
                    frames=["a2_ur_vacuum", "box010"],
                    side_effect=None,
                ),
                Mode(
                    ["a2"],
                    SingleGoal(keyframes[16][self.robot_idx["a2"]]),
                    type="place",
                    frames=["table", "box010"],
                    side_effect="remove",
                ),
                # SingleGoal(keyframes[17][self.robot_idx["a2"]]),
                Mode(
                    ["a2"],
                    SingleGoal(keyframes[18][self.robot_idx["a2"]]),
                    type="pick",
                    frames=["a2_ur_vacuum", "box012"],
                    side_effect=None,
                ),
                Mode(
                    ["a2"],
                    SingleGoal(keyframes[19][self.robot_idx["a2"]]),
                    type="place",
                    frames=["table", "box012"],
                    side_effect="remove",
                ),
                # SingleGoal(keyframes[20][self.robot_idx["a2"]]),
                Mode(
                    ["a2"],
                    SingleGoal(keyframes[21][self.robot_idx["a2"]]),
                    type="pick",
                    frames=["a2_ur_vacuum", "box022"],
                    side_effect=None,
                ),
                Mode(
                    ["a2"],
                    SingleGoal(keyframes[22][self.robot_idx["a2"]]),
                    type="place",
                    frames=["table", "box022"],
                    side_effect="remove",
                ),
                # SingleGoal(keyframes[23][self.robot_idx["a2"]]),
                Mode(
                    ["a2"],
                    SingleGoal(keyframes[24][self.robot_idx["a2"]]),
                    type="pick",
                    frames=["a2_ur_vacuum", "box020"],
                    side_effect=None,
                ),
                Mode(
                    ["a2"],
                    SingleGoal(keyframes[25][self.robot_idx["a2"]]),
                    type="place",
                    frames=["table", "box020"],
                    side_effect="remove",
                ),
                Mode(["a2"], SingleGoal(keyframes[26][self.robot_idx["a2"]])),
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

        # a1_boxes = ["box000", "box001", "box011", "box002"]
        # a2_boxes = ["box021", "box010", "box012", "box022", "box020"]

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        # buffer for faster collision checking
        self.prev_mode = [0, 0]

        self.start_mode = [0 for _ in self.robots]
        self.terminal_mode = \
            [len(self.robot_goals[r]) - 1 for r in self.robots]

        self.tolerance = 0.1

        # q = self.C_base.getJointState()
        # print(self.is_collision_free(q, [0, 0]))

        # self.C_base.view(True)


class rai_ur10_arm_pick_and_place_env(rai_dual_ur10_arm_env):
    def __init__(self):
        super().__init__()

        self.manipulating_env = True

        self.robot_goals = {
            "a1": [
                Mode(
                    ["a1"],
                    SingleGoal(keyframes[0][self.robot_idx["a1"]]),
                    type="pick",
                    frames=["a1_ur_vacuum", "box100"],
                    side_effect=None,
                ),
                Mode(
                    ["a1"],
                    SingleGoal(keyframes[1][self.robot_idx["a1"]]),
                    type="place",
                    frames=["table", "box100"],
                    side_effect="remove",
                ),
                Mode(["a1"], SingleGoal(keyframes[2][self.robot_idx["a1"]])),
            ],
            "a2": [
                Mode(
                    ["a2"],
                    SingleGoal(keyframes[3][self.robot_idx["a2"]]),
                    type="pick",
                    frames=["a2_ur_vacuum", "box101"],
                    side_effect=None,
                ),
                Mode(
                    ["a2"],
                    SingleGoal(keyframes[4][self.robot_idx["a2"]]),
                    type="place",
                    frames=["table", "box101"],
                    side_effect="remove",
                ),
                Mode(["a2"], SingleGoal(keyframes[5][self.robot_idx["a2"]])),
            ],
        }

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        self.sequence = [("a1", 0), ("a2", 0), ("a1", 1), ("a2", 1)]

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

        self.robot_goals = {
            "a0": [
                Mode(
                    ["a0"],
                    SingleGoal(keyframes[0][self.robot_idx["a0"]]),
                    type="pick",
                    frames=["a0_ur_vacuum", "bottle_1"],
                    side_effect=None,
                ),
                Mode(
                    ["a0"],
                    SingleGoal(keyframes[1][self.robot_idx["a0"]]),
                    type="place",
                    frames=["table", "bottle_1"],
                    side_effect=None,
                ),
                # SingleGoal(keyframes[2][self.robot_idx["a1"]]),
                Mode(
                    ["a0"],
                    SingleGoal(keyframes[3][self.robot_idx["a0"]]),
                    type="pick",
                    frames=["a0_ur_vacuum", "bottle_12"],
                    side_effect=None,
                ),
                Mode(
                    ["a0"],
                    SingleGoal(keyframes[4][self.robot_idx["a0"]]),
                    type="place",
                    frames=["table", "bottle_12"],
                    side_effect=None,
                ),
                Mode(["a0"], SingleGoal(keyframes[5][self.robot_idx["a0"]])),
            ],
            "a1": [
                Mode(
                    ["a1"],
                    SingleGoal(keyframes[6][self.robot_idx["a1"]]),
                    type="pick",
                    frames=["a1_ur_vacuum", "bottle_3"],
                    side_effect=None,
                ),
                Mode(
                    ["a1"],
                    SingleGoal(keyframes[7][self.robot_idx["a1"]]),
                    type="place",
                    frames=["table", "bottle_3"],
                    side_effect=None,
                ),
                # SingleGoal(keyframes[8][self.robot_idx["a1"]]),
                Mode(
                    ["a1"],
                    SingleGoal(keyframes[9][self.robot_idx["a1"]]),
                    type="pick",
                    frames=["a1_ur_vacuum", "bottle_5"],
                    side_effect=None,
                ),
                Mode(
                    ["a1"],
                    SingleGoal(keyframes[10][self.robot_idx["a1"]]),
                    type="place",
                    frames=["table", "bottle_5"],
                    side_effect=None,
                ),
                Mode(["a1"], SingleGoal(keyframes[11][self.robot_idx["a1"]])),
            ],
        }

        # for k, v in self.robot_goals.items():
        #     for g in v:
        #         print(g.goal)

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

        # a1_boxes = ["box000", "box001", "box011", "box002"]
        # a2_boxes = ["box021", "box010", "box012", "box022", "box020"]

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        # buffer for faster collision checking
        self.prev_mode = [0, 0]

        self.start_mode = [0 for _ in self.robots]
        self.terminal_mode = \
            [len(self.robot_goals[r]) - 1 for r in self.robots]

        self.tolerance = 0.1

        # q = self.C_base.getJointState()
        # print(self.is_collision_free(q, [0, 0]))

        # self.C_base.view(True)

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        # buffer for faster collision checking
        self.prev_mode = [0, 0]


# mobile manip
class rai_mobile_manip_wall:
    pass


# TODO: make rai-independent
def visualize_modes(env: rai_env):
    q_home = env.start_pos

    m = env.start_mode
    for i in range(len(env.sequence)):
        if np.array_equal(m, env.terminal_mode):
            switching_robots = [r for r in env.robots]
        else:
            # find the robot(s) that needs to switch the mode
            switching_robots = env.get_goal_constrained_robots(m)

        q = []
        for j, r in enumerate(env.robots):
            if r in switching_robots:
                # TODO: need to check all goals here
                q.append(env.robot_goals[r][m[j]].goal.goal)
            else:
                q.append(q_home.robot_state(j))

        print(
            "is collision free: ",
            env.is_collision_free(type(env.get_start_pos()).from_list(q).state(), m),
        )

        m = env.get_next_mode(None, m)

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

        m = env.sample_random_mode()

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
            if np.array_equal(m, env.terminal_mode):
                switching_robots = [r for r in env.robots]
            else:
                # find the robot(s) that needs to switch the mode
                switching_robots = env.get_goal_constrained_robots(m)

            q = []
            for j, r in enumerate(env.robots):
                if r in switching_robots:
                    # TODO: need to check all goals here
                    q.append(env.goals[r][m[j]].goal.goal)
                else:
                    q.append(q_home.robot_state(j))

            is_collision_free = env.is_collision_free(
                type(env.get_start_pos()).from_list(q).state(), m
            )
            if not is_collision_free:
                raise ValueError()

            print(f"mode {m} is collision free: ", is_collision_free)

            env.show()

            # colls = env.C.getCollisions()
            # for c in colls:
            #     if c[2] < 0:
            #         print(c)
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
