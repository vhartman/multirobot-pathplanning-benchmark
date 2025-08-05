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
    ProblemSpec,
    AgentType,
    GoalType,
    ConstraintType,
    DynamicsType,
    ManipulationType,
    DependencyType,
    SafePoseType,
)
from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
    NpConfiguration,
    config_dist,
    config_cost,
    batch_config_cost,
)

from multi_robot_multi_goal_planning.problems.planning_env import (
    generate_binary_search_indices,
)

import time

import os
import sys
from contextlib import contextmanager
from functools import wraps
import copy

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


def delete_visual_only_frames(C):
    C_coll = ry.Config()
    C_coll.addConfigurationCopy(C)

    # go through all frames, and delete the ones that are only visual
    # that is, the frames that do not have a child, and are not
    # contact frames
    for f in C_coll.getFrames():
        info = f.info()
        if "shape" in info and info["shape"] == "mesh":
            C_coll.delFrame(f.name)

    return C_coll


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

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.01

        self.cost_metric = "euclidean"
        self.cost_reduction = "max"

        self.C_cache = {}

        self.C_orig = ry.Config()
        self.C_orig.addConfigurationCopy(self.C)

        C_coll = delete_visual_only_frames(self.C)

        self.C.clear()
        self.C.addConfigurationCopy(C_coll)

        self.C_base = ry.Config()
        self.C_base.addConfigurationCopy(self.C)

        self.spec = ProblemSpec(
            agent_type=AgentType.MULTI_AGENT,
            constraints=ConstraintType.UNCONSTRAINED,
            manipulation=ManipulationType.MANIPULATION,
            dependency=DependencyType.FULLY_ORDERED,
            dynamics=DynamicsType.GEOMETRIC,
            goals=GoalType.MULTI_GOAL,
            home_pose=SafePoseType.HAS_NO_SAFE_HOME_POSE,
        )

    def __deepcopy__(self, memo):
        # Save the C attribute
        C = self.C
        C_base = self.C_base
        C_orig = self.C_orig

        # Temporarily remove C to allow deepcopy of other attributes
        self.C = None
        self.C_base = None
        self.C_orig = None

        # Create a deep copy of self without C
        new_env = copy.deepcopy(super(), memo)

        # Restore C in both objects
        self.C = C
        new_env.C = ry.Config()
        new_env.C.addConfigurationCopy(self.C)

        self.C_base = C_base
        new_env.C_base = ry.Config()
        new_env.C_base.addConfigurationCopy(self.C_base)

        self.C_orig = C_orig
        new_env.C_orig = ry.Config()
        new_env.C_orig.addConfigurationCopy(self.C_orig)

        return new_env

    def sample_config_uniform_in_limits(self) -> NpConfiguration:
        rnd = np.random.uniform(low=self.limits[0, :], high=self.limits[1, :])
        q = self.start_pos.from_flat(rnd)

        return q

    def sample_goal_configuration(self, mode, task):
        goals_to_sample = task.robots
        goal_sample = task.goal.sample(mode)

        q = []
        for i in range(len(self.robots)):
            r = self.robots[i]
            if r in goals_to_sample:
                offset = 0
                for _, task_robot in enumerate(task.robots):
                    if task_robot == r:
                        q.append(
                            goal_sample[offset : offset + self.robot_dims[task_robot]]
                        )
                        break
                    offset += self.robot_dims[task_robot]
            else:  # uniform sample
                lims = self.limits[:, self.robot_idx[r]]
                if lims[0, 0] < lims[1, 0]:
                    qr = (
                        np.random.rand(self.robot_dims[r]) * (lims[1, :] - lims[0, :])
                        + lims[0, :]
                    )
                else:
                    qr = np.random.rand(self.robot_dims[r]) * 6 - 3

                q.append(qr)

        q = NpConfiguration.from_list(q)

        return q

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
        m: Optional[Mode],
        collision_tolerance: float | None = None,
    ) -> bool:
        if collision_tolerance is None:
            collision_tolerance = self.collision_tolerance

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

    def is_collision_free_np(
        self,
        q: Optional[Configuration],
        m: Mode,
        collision_tolerance: float = None,
        set_mode: bool = True,
    ) -> bool:
        if collision_tolerance is None:
            collision_tolerance = self.collision_tolerance

        # print(q)
        # self.C.setJointState(q)

        if q is not None:
            if set_mode:
                self.set_to_mode(m)
            self.C.setJointState(q)

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
        self,
        r: str,
        q: NDArray,
        m: Mode | None = None,
        collision_tolerance: float | None = None,
        set_mode: bool = True,
    ) -> bool:
        if collision_tolerance is None:
            collision_tolerance = self.collision_tolerance

        if isinstance(r, str):
            r = [r]

        # if q is not None:
        #     self.set_to_mode(m)
        #     for robot in r:
        #         robot_indices = self.robot_idx[robot]
        #         self.C.setJointState(q[robot_indices], self.robot_joints[robot])
        if set_mode:
            self.set_to_mode(m)
        self.C.setJointState(q)

        # self.C.view(True)

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
                involves_relevant_object = False
                for robot in r:
                    task_idx = m.task_ids[self.robots.index(robot)]
                    task = self.tasks[task_idx]
                    if c[2] < 0 and (robot in c[0] or robot in c[1]):
                        involves_relevant_robot = True
                    elif c[2] < 0 and (c[0] in task.frames or c[1] in task.frames):
                        involves_relevant_object = True

                if not involves_relevant_robot and not involves_relevant_object:
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

    # @silence_function
    def is_edge_collision_free(
        self,
        q1: Configuration,
        q2: Configuration,
        m: Mode,
        resolution: float | None = None,
        tolerance: float | None = None,
        include_endpoints: bool = False,
        N_start: int = 0,
        N_max: int | None = None,
        N: int | None = None,
    ) -> bool:
        if resolution is None:
            resolution = self.collision_resolution

        if tolerance is None:
            tolerance = self.collision_tolerance

        # print('q1', q1)
        # print('q2', q2)
        if N is None:
            N = int(config_dist(q1, q2, "max") / resolution) + 1
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

        is_collision_free = self.is_collision_free_np

        q1_state = q1.state()
        q2_state = q2.state()
        dir = (q2_state - q1_state) / (N - 1)
        set_mode = True
        for i in idx[N_start:N_max]:
            if not include_endpoints and (i == 0 or i == N - 1):
                continue

            # print(i / (N-1))
            q = q1_state + dir * (i)
            # q_conf = NpConfiguration(q, q1.array_slice)
            if not is_collision_free(
                q, m, collision_tolerance=tolerance, set_mode=set_mode
            ):
                # print('coll')
                return False
            set_mode = False

        return True

    def get_scenegraph_info_for_mode(self, mode: Mode, is_start_mode: bool = False):
        if not self.manipulating_env:
            return {}

        self.set_to_mode(mode, place_in_cache=False, use_cached=False)

        # TODO: should we simply list the movable objects manually?
        # collect all movable objects and collect parents and relative transformations
        movable_objects = []

        sg = {}
        for frame in self.C.getFrames():
            if "obj" in frame.name:
                movable_objects.append(frame.name)

                relative_pose = np.round(frame.getRelativeTransform(), 3)
                relative_pose[abs(relative_pose) < 1e-6] = 0.0
                relative_pose.flags.writeable = False

                sg[frame.name] = (
                    frame.getParent().name,
                    relative_pose.tobytes(),
                )
        mode._cached_hash = None
        return sg

    def set_to_mode(
        self, m: Mode, config=None, use_cached: bool = True, place_in_cache: bool = True
    ):
        if not self.manipulating_env:
            return

        # do not remake the config if we are in the same mode as we have been before
        if m == self.prev_mode:
            return

        self.prev_mode = m

        if use_cached and m in self.C_cache:
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

        # if m.task_ids == [9,9]:
        #     print(m.sg)
        #     print(hash(m))

        show_stuff = False

        for i, mode in enumerate(mode_sequence[:-1]):
            next_mode = mode_sequence[i + 1]

            # for j in range(2):
            #     if (
            #         mode.task_ids[j] in [1, 3, 5, 7, 11, 9]
            #         and next_mode.task_ids[j] == 13
            #         and m.task_ids == [13, 13]
            #     ):
            #         show_stuff = True

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

            q = np.concatenate(q_new)
            tmp.setJointState(q, joint_names)

            # if m.task_ids == [9, 9]:
            #     print(mode, next_mode, active_task.name, mode_switching_robots, [self.tasks[i].name for i in next_mode.task_ids])
            #     tmp.view(True)

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

            if show_stuff:
                tmp.view(True)

        if place_in_cache:
            self.C_cache[m] = ry.Config()
            self.C_cache[m].addConfigurationCopy(tmp)

            # self.C = None
            viewer = self.C.get_viewer()
            self.C_cache[m].set_viewer(viewer)
            self.C = self.C_cache[m]

        else:
            viewer = self.C.get_viewer()
            tmp.set_viewer(viewer)
            self.C = tmp

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
            self.set_to_mode(path[i].mode)
            for k in range(len(self.robots)):
                q = path[i].q[k]
                self.C.setJointState(q, get_robot_joints(self.C, self.robots[k]))

                # print(q)

            if stop_at_mode and i < len(path) - 1:
                if path[i].mode != path[i + 1].mode:
                    print(i)
                    print("Current mode:", path[i].mode)
                    print("Next mode:", path[i + 1].mode)
                    self.C.view(True)

            self.C.view(stop)

            if export:
                self.C.view_savePng("./z.vid/")

            dt = pause_time
            if adapt_to_max_distance:
                if i < len(path) - 1:
                    v = 5
                    diff = config_dist(path[i].q, path[i + 1].q, "max_euclidean")
                    dt = diff / v

            time.sleep(dt)

        if stop_at_end:
            self.C.view(True)
