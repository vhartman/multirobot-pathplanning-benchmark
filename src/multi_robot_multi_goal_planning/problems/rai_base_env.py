import robotic as ry
import numpy as np

from typing import List, Dict, Optional
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.rai_config import get_robot_joints
from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseProblem,
    SequenceMixin,
    Mode,
    State,
    Task,
)
from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
    NpConfiguration,
    config_dist,
    config_cost,
    batch_config_cost,
)

from multi_robot_multi_goal_planning.problems.util import generate_binary_search_indices


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


# def set_robot_active(C: ry.Config, robot_prefix: str) -> None:
#     robot_joints = get_robot_joints(C, robot_prefix)
#     C.selectJoints(robot_joints)


class rai_env(BaseProblem):
    # robot things
    C: ry.Config
    limits: NDArray

    # sequence things
    sequence: List[int]
    tasks: List[Task]
    start_mode: Mode
    _terminal_task_ids: List[int]

    # misc
    tolerance: float

    def __init__(self):
        self.robot_idx = {}
        self.robot_dims = {}
        self.robot_joints = {}

        for r in self.robots:
            self.robot_joints[r] = get_robot_joints(self.C, r)
            self.robot_idx[r] = get_joint_indices(self.C, r)
            self.robot_dims[r] = len(get_joint_indices(self.C, r))

        self.start_pos = NpConfiguration.from_list(
            [get_robot_state(self.C, r) for r in self.robots]
        )

        self.manipulating_env = False

        self.limits = self.C.getJointLimits()

        self.tolerance = 0.1

        self.cost_metric = "max"
        self.cost_reduction = "max"

        self.C_cache = {}

    def config_cost(self, start: Configuration, end: Configuration) -> float:
        return config_cost(start, end, self.cost_metric, self.cost_reduction)

    def batch_config_cost(
        self, starts: List[Configuration], ends: List[Configuration]
    ) -> NDArray:
        return batch_config_cost(starts, ends, self.cost_metric, self.cost_reduction)

    def show_config(self, q: NDArray, blocking: bool = True):
        self.C.setJointState(q)
        self.C.view(blocking)

    def show(self, blocking: bool = True):
        self.C.view(blocking)

    # Environment functions: collision checking
    def is_collision_free(
        self,
        q: Optional[Configuration],
        m: Mode,
        collision_tolerance: float = 0.01,
    ) -> bool:
        # print(q)
        # self.C.setJointState(q)

        if q is not None:
            self.set_to_mode(m)
            self.C.setJointState(q.state())

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

    def is_collision_free_for_robot(
        self, r: str, q: NDArray, m: Mode, collision_tolerance=0.01
    ) -> bool:
        if isinstance(r, str):
            r = [r]

        if q is not None:
            self.set_to_mode(m)
            offset = 0
            for robot in r:
                dim = self.robot_dims[robot]
                self.C.setJointState(q[offset:offset+dim], self.robot_joints[robot])
                offset += dim

        binary_collision_free = self.C.getCollisionFree()
        if binary_collision_free:
            return True

        col = self.C.getCollisionsTotalPenetration()
        # print(col)
        # self.C.view(False)
        if col > collision_tolerance:
            # self.C.view(False)
            colls = self.C.getCollisions()
            for c in colls:
                # ignore minor collisions
                if c[2] > -collision_tolerance / 10:
                    continue

                # print(c)
                involves_relevant_robot = False
                for robot in r:
                    if c[2] < 0 and (robot in c[0] or robot in c[1]):
                        involves_relevant_robot = True
                        break
                if not involves_relevant_robot:
                    # print("A")
                    # print(c)
                    continue
                # else:
                #     print("B")
                #     print(c)

                is_collision_with_other_robot = False
                for other_robot in self.robots:
                    if other_robot in r:
                        continue
                    if other_robot in c[0] or other_robot in c[1]:
                        is_collision_with_other_robot = True
                        break

                if not is_collision_with_other_robot:
                    # print(c)
                    return False

        return True

    def is_edge_collision_free(
        self,
        q1: Configuration,
        q2: Configuration,
        m: Mode,
        resolution=0.1,
        randomize_order=True,
    ) -> bool:
        # print('q1', q1)
        # print('q2', q2)
        N = config_dist(q1, q2) / resolution
        N = max(2, N)

        idx = list(range(int(N)))
        if randomize_order:
            # np.random.shuffle(idx)
            idx = generate_binary_search_indices(int(N)).copy()

        for i in idx:
            # print(i / (N-1))
            q = q1.state() + (q2.state() - q1.state()) * (i) / (N - 1)
            q_conf = NpConfiguration(q, q1.slice)
            if not self.is_collision_free(q_conf, m):
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

    def get_scenegraph_info_for_mode(self, mode: Mode):
        if not self.manipulating_env:
            return {}
        
        self.set_to_mode(mode)

        # TODO: should we simply list the movable objects manually?
        # collect all movable objects and colect parents and relative transformations
        movable_objects = []

        sg = {}
        for frame in self.C.frames():
            if "obj" in frame.name:
                movable_objects.append(frame.name)
                sg[frame.name] = (frame.getParent().name, np.round(frame.getRelativeTransform(), 3).tobytes())

        return sg

    def set_to_mode(self, m: Mode):
        if not self.manipulating_env:
            return

        # do not remake the config if we are in the same mode as we have been before
        if m == self.prev_mode:
            return

        self.prev_mode = m

        if m in self.C_cache:
            self.C.clear()
            self.C.addConfigurationCopy(self.C_cache[m])
            return

        # TODO: we might want to cache different modes
        self.C.clear()
        self.C.addConfigurationCopy(self.C_base)

        mode_sequence = []

        current_mode = m
        while True:
            if current_mode.prev_mode is None:
                break

            mode_sequence.append(current_mode)
            current_mode = current_mode.prev_mode
        
        mode_sequence.append(current_mode)
        mode_sequence = mode_sequence[::-1]

        # print(mode_sequence)

        for i, mode in enumerate(mode_sequence[:-1]):
            next_mode = mode_sequence[i+1]

            active_task = self.get_active_task(mode, next_mode.task_ids)

            # mode_switching_robots = self.get_goal_constrained_robots(mode)
            mode_switching_robots = active_task.robots

            # set robot to config
            prev_mode_index = mode.task_ids[self.robots.index(mode_switching_robots[0])]
            # robot = self.robots[mode_switching_robots]

            q_new = []
            joint_names = []
            for r in mode_switching_robots:
                joint_names.extend(self.robot_joints[r])
                q_new.append(next_mode.entry_configuration[self.robots.index(r)])

            assert(mode is not None)
            assert(mode.entry_configuration is not None)

            q = np.concat(q_new)
            self.C.setJointState(q, joint_names)

            if self.tasks[prev_mode_index].type is not None:
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

        self.C_cache[m] = ry.Config()
        self.C_cache[m].addConfigurationCopy(self.C)