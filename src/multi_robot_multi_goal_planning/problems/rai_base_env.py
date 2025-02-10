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


import os
import sys
import io
from contextlib import contextmanager
from functools import wraps
import tempfile

DEVNULL = os.open(os.devnull, os.O_WRONLY)


def close_devnull():
    """Closes the global /dev/null file descriptor."""
    global DEVNULL
    if DEVNULL is not None:
        os.close(DEVNULL)
        DEVNULL = None


@contextmanager
def silence_output():
    """
    Context manager to silence all output (stdout and stderr) using a persistent /dev/null.
    """
    # Backup original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    # Redirect stdout and stderr to the persistent /dev/null
    os.dup2(DEVNULL, stdout_fd)
    os.dup2(DEVNULL, stderr_fd)

    try:
        yield
    finally:
        # Restore original file descriptors
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


# Decorator version
def silence_function(func):
    """
    Decorator to silence all output from a function.
    """

    def wrapper(*args, **kwargs):
        with silence_output():
            return func(*args, **kwargs)

    return wrapper


class FilterManager:
    def __init__(self, filter_func):
        self.filter_func = filter_func
        self.read_fd = None
        self.write_fd = None
        self.saved_stdout_fd = None
        self.saved_stderr_fd = None

    def start(self):
        if self.read_fd is None or self.write_fd is None:
            self.read_fd, self.write_fd = os.pipe()
        self.saved_stdout_fd = os.dup(sys.stdout.fileno())
        self.saved_stderr_fd = os.dup(sys.stderr.fileno())
        os.dup2(self.write_fd, sys.stdout.fileno())
        os.dup2(self.write_fd, sys.stderr.fileno())

    def stop(self):
        if self.saved_stdout_fd is None or self.saved_stderr_fd is None:
            return  # Avoid double stopping
        os.dup2(self.saved_stdout_fd, sys.stdout.fileno())
        os.dup2(self.saved_stderr_fd, sys.stderr.fileno())
        os.close(self.saved_stdout_fd)
        os.close(self.saved_stderr_fd)
        self.saved_stdout_fd = None
        self.saved_stderr_fd = None

        if self.write_fd is not None:
            os.close(self.write_fd)
            self.write_fd = None

        # Read from the pipe and apply the filter
        if self.read_fd is not None:
            with os.fdopen(self.read_fd, "r") as pipe:
                for line in pipe:
                    if self.filter_func(line):
                        sys.stdout.write(line)
                        sys.stdout.flush()
            self.read_fd = None


# Decorator version
def reusable_filter_output(filter_func):
    """
    Decorator to filter output using a reusable FilterManager.
    """
    manager = FilterManager(filter_func)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager.start()
            try:
                return func(*args, **kwargs)
            finally:
                manager.stop()

        return wrapper

    return decorator


@contextmanager
def fast_capture_and_filter_output(filter_func):
    """
    Captures and filters output directly from file descriptors, bypassing os.fdopen.
    """
    # Backup original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    # Create a pipe for capturing output
    read_fd, write_fd = os.pipe()

    # Redirect stdout and stderr to the write end of the pipe
    os.dup2(write_fd, stdout_fd)
    os.dup2(write_fd, stderr_fd)

    try:
        yield
    finally:
        # Close the write end to signal EOF
        os.close(write_fd)

        # Restore original file descriptors
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)

        # Read directly from the pipe and filter the output
        while True:
            chunk = os.read(read_fd, 4096)  # Read in chunks
            if not chunk:
                break
            for line in chunk.decode().splitlines():
                if filter_func(line):
                    sys.stdout.write(line + "\n")
                    sys.stdout.flush()

        os.close(read_fd)


def filter_output(filter_func):
    """
    Decorator that filters output based on the provided function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with fast_capture_and_filter_output(filter_func):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def hide_pair_collision(line):
    """Filter out lines containing 'debug'"""
    return "pairCollision.cpp:libccd:" not in line


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
    collision_tolerance: float
    collision_resolution: float

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

        self.collision_tolerance = 0.1
        self.collision_resolution = 0.05

        self.cost_metric = "max"
        self.cost_reduction = "max"

        self.C_cache = {}

    def config_cost(self, start: Configuration, end: Configuration) -> float:
        return config_cost(start, end, self.cost_metric, self.cost_reduction)

    def batch_config_cost(
        self, starts: List[Configuration], ends: List[Configuration]
    ) -> NDArray:
        return batch_config_cost(starts, ends, self.cost_metric, self.cost_reduction)

    def show_config(self, q: Configuration, blocking: bool = True):
        self.C.setJointState(q.state())
        self.C.view(blocking)

    def show(self, blocking: bool = True):
        self.C.view(blocking)

    # Environment functions: collision checking
    # @filter_output(hide_pair_collision)
    # @silence_function
    # @reusable_filter_output(hide_pair_collision)
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

        # self.C.view()

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
                self.C.setJointState(q[offset : offset + dim], self.robot_joints[robot])
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

    def is_collision_free_without_mode(
        self,
        q: Optional[Configuration],
        collision_tolerance: float = 0.01,
    ) -> bool:
        # print(q)
        # self.C.setJointState(q)

        if q is not None:
            self.C.setJointState(q.state())

        # self.C.view()

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
        m: Mode,
        resolution=0.01,
        randomize_order=True,
        tolerance=0.01,
    ) -> bool:
        # print('q1', q1)
        # print('q2', q2)
        N = int(config_dist(q1, q2) / resolution)
        N = max(2, N)

        # for a distance < resolution * 2, we do not do collision checking
        # if N == 0:
        #     return True

        idx = list(range(N))
        if randomize_order:
            # np.random.shuffle(idx)
            idx = generate_binary_search_indices(N).copy()

        for i in idx:
            # print(i / (N-1))
            q = q1.state() + (q2.state() - q1.state()) * (i) / (N - 1)
            q_conf = NpConfiguration(q, q1.array_slice)
            if not self.is_collision_free(q_conf, m, collision_tolerance=tolerance):
                # print('coll')
                return False

        return True

    def is_path_collision_free(
        self, path: List[State], randomize_order=True, resolution=0.1, tolerance=0.01
    ) -> bool:
        idx = list(range(len(path) - 1))
        if randomize_order:
            np.random.shuffle(idx)

        for i in idx:
            # skip transition nodes
            # if path[i].mode != path[i + 1].mode:
            #     continue

            q1 = path[i].q
            q2 = path[i + 1].q
            # mode = path[i].mode
            for i in idx:
                if path[i].mode != path[i + 1].mode:
                    mode = path[i+1].mode
                else:
                    mode = path[i].mode



            if not self.is_edge_collision_free(
                q1, q2, mode, resolution=resolution, tolerance=tolerance
            ):
                return False

        return True
    
    def get_scenegraph_info_for_mode(self, mode: Mode):
        if not self.manipulating_env:
            return {}

        self.set_to_mode(mode)

        # TODO: should we simply list the movable objects manually?
        # collect all movable objects and collect parents and relative transformations
        movable_objects = []

        sg = {}
        for frame in self.C.getFrames():
            if "obj" in frame.name:
                movable_objects.append(frame.name)
                sg[frame.name] = (
                    frame.getParent().name,
                    np.round(frame.getRelativeTransform(), 3).tobytes(),
                )

        return sg

    def set_to_mode(self, m: Mode, config=None):
        if not self.manipulating_env:
            return

        # do not remake the config if we are in the same mode as we have been before
        if m == self.prev_mode:
            return

        self.prev_mode = m

        if m in self.C_cache:
            if config is not None:
                config.clear()
                config.addConfigurationCopy(self.C_cache[m])
            else:
                # for k, v in self.C_cache.items():
                #     print(k)
                #     v.setJointState(v.getJointState()*0)
                #     # v.view(False)

                self.C = self.C_cache[m]

            # self.C.view(True)
            return

        tmp = ry.Config()

        # self.C.clear()
        tmp.addConfigurationCopy(self.C_base)

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
            next_mode = mode_sequence[i + 1]

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

            assert mode is not None
            assert mode.entry_configuration is not None

            q = np.concat(q_new)
            tmp.setJointState(q, joint_names)

            if self.tasks[prev_mode_index].type is not None:
                if self.tasks[prev_mode_index].type == "goto":
                    pass
                else:
                    tmp.attach(
                        self.tasks[prev_mode_index].frames[0],
                        self.tasks[prev_mode_index].frames[1],
                    )
                    tmp.getFrame(self.tasks[prev_mode_index].frames[1]).setContact(-1)

                # postcondition
                if self.tasks[prev_mode_index].side_effect is not None:
                    box = self.tasks[prev_mode_index].frames[1]
                    tmp.delFrame(box)

        self.C_cache[m] = ry.Config()
        self.C_cache[m].addConfigurationCopy(tmp)

        # self.C = None
        viewer = self.C.get_viewer()
        self.C_cache[m].set_viewer(viewer)
        self.C = self.C_cache[m]
