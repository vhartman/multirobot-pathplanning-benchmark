import robotic as ry
import numpy as np

from typing import List, Dict, Optional
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.rai_config import *
from multi_robot_multi_goal_planning.problems.planning_env import (
    base_env,
    SequenceMixin,
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


class rai_env(SequenceMixin, base_env):
    # robot things
    C: ry.Config
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

    def config_cost(self, start: Configuration, end: Configuration) -> float:
        return config_cost(start, end, "max")

    def batch_config_cost(
        self, starts: List[Configuration], ends: List[Configuration]
    ) -> NDArray:
        return batch_config_cost(starts, ends, "max")

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

    def show_config(self, q: NDArray, blocking: bool = True):
        self.C.setJointState(q)
        self.C.view(blocking)

    def show(self, blocking: bool = True):
        self.C.view(blocking)

    def is_transition(self, q: Configuration, m: List[int]) -> bool:
        if m == self.terminal_mode:
            return False

        # robots_with_constraints_in_current_mode = self.get_goal_constrained_robots(m)
        task = self.get_active_task(m)

        q_concat = []
        for r in task.robots:
            r_idx = self.robots.index(r)
            q_concat.append(q.robot_state(r_idx))

        q_concat = np.concatenate(q_concat)

        if task.goal.satisfies_constraints(q_concat, self.tolerance):
            return True

        return False

    def get_next_mode(self, q: Optional[Configuration], mode: List[int]) -> List[int]:
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

    # Environment functions: collision checking
    def is_collision_free(
        self,
        q: Optional[Configuration],
        m: List[int],
        collision_tolerance: float = 0.01,
    ) -> bool:
        # print(q)
        # self.C.setJointState(q)

        if q is not None:
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

    def is_collision_free_for_robot(
        self, r: str, q: NDArray, m: List[int], collision_tolerance=0.01
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
    
    def is_robot_env_collision_free(
        self, r: str, q: NDArray, m: List[int], collision_tolerance=0.01
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
                involves_relevant_robot = False
                relevant_robot = None
                for robot in r:
                    if robot in c[0] or robot in c[1]:
                        involves_relevant_robot = True
                        relevant_robot = robot
                        break
                if not involves_relevant_robot:
                    continue
                involves_objects = True
                for other_robot in self.robots:
                    if other_robot != relevant_robot:
                        if other_robot in c[0] or other_robot in c[1]:
                            involves_objects = False
                            break
                if involves_objects:
                    return False
            return True
        return False

    def is_edge_collision_free(
        self,
        q1: Configuration,
        q2: Configuration,
        m: List[int],
        resolution=0.01,
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

    # def is_path_collision_free(self, path: List[State], randomize_order=True) -> bool:
    #     idx = list(range(len(path) - 1))
    #     if randomize_order:
    #         np.random.shuffle(idx)

    #     for i in idx:
    #         # skip transition nodes
    #         if path[i].mode != path[i + 1].mode:
    #             continue

    #         q1 = path[i].q
    #         q2 = path[i + 1].q
    #         mode = path[i].mode

    #         if not self.is_edge_collision_free(q1, q2, mode):
    #             return False

    #     return True
    
    def is_path_collision_free(self, path: List[State], randomize_order=True) -> bool:
        idx = list(range(len(path) - 1))
        if randomize_order:
            np.random.shuffle(idx)

        for i in idx:
            if path[i].mode != path[i + 1].mode:
                mode = path[i+1].mode
            else:
                mode = path[i].mode

            q1 = path[i].q
            q2 = path[i + 1].q

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

        # TODO: we might want to cache different modes
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
                joint_names.extend(self.robot_joints[r])

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
