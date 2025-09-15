import numpy as np
import random
import time

import bisect

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)
from numpy.typing import NDArray

from collections import namedtuple
import copy

from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseProblem,
    Mode,
    State,
    SafePoseType,
    DependencyType,
    generate_binary_search_indices,
)
from multi_robot_multi_goal_planning.problems.goals import SingleGoal
from .termination_conditions import (
    PlannerTerminationCondition,
    RuntimeTerminationCondition,
)
from .shortcutting import robot_mode_shortcut
from multi_robot_multi_goal_planning.problems.rai_envs import rai_env
from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
    batch_config_dist,
    config_dist,
    batch_config_cost,
)
from .baseplanner import BasePlanner

from multi_robot_multi_goal_planning.problems.util import path_cost

import logging

logger = logging.getLogger(__name__)

TimedPath = namedtuple("TimedPath", ["time", "path"])
Path = namedtuple("Path", ["path", "task_index"])


class MultiRobotPath:
    __slots__ = [
        "robots",
        "paths",
        "q0",
        "m0",
        "q0_split",
        "robot_ids",
        "timed_mode_sequence",
        "times",
    ]

    def __init__(self, q0: Configuration, m0: Mode, robots: List[str]):
        self.robots = robots
        self.paths = {}
        self.q0 = q0
        self.m0 = m0

        self.q0_split = {}
        for i, r in enumerate(robots):
            self.q0_split[i] = self.q0[i]

        self.robot_ids = {}
        for i, r in enumerate(robots):
            self.robot_ids[r] = i

        # indicates which mode is active after a certain time (until the next one)
        self.timed_mode_sequence = [(0, self.m0)]
        self.times = [0]

        for r in robots:
            self.paths[r] = []

    def get_mode_at_time(self, t: float) -> Mode:
        """
        Iterate over all the stored modes and give the one back that we are in at a certain time.
        """

        if t >= self.times[-1]:
            return self.timed_mode_sequence[-1][1]

        idx = bisect.bisect_right(self.times, t)

        return self.timed_mode_sequence[idx - 1][1]

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def get_robot_poses_at_time(self, robots: List[str], t: float):
        poses = []

        for r in robots:
            i = self.robot_ids[r]

            if not self.paths[r]:
                poses.append(self.q0_split[i])
            else:
                if t >= self.paths[r][-1].path.time[-1]:
                    pose = self.paths[r][-1].path.path[-1]
                elif t <= self.paths[r][0].path.time[0]:
                    pose = self.q0_split[i]
                else:
                    path_end_times = [rp.path.time[-1] for rp in self.paths[r]]
                    segment_idx = bisect.bisect_left(path_end_times, t)

                    rp = self.paths[r][segment_idx]

                    if t == rp.path.time[-1] and segment_idx < len(self.paths[r]) - 1:
                        # If 't' is exactly the end time of a segment that is not the very last segment,
                        # we take the last pose of *that* segment.
                        pose = rp.path.path[-1]
                    elif rp.path.time[0] <= t < rp.path.time[-1]:
                        # 't' is within the current segment (exclusive of the end time).
                        p = rp.path.path
                        time_points = rp.path.time

                        # Find the specific time interval within the segment
                        k = bisect.bisect_right(time_points, t) - 1

                        time_k = time_points[k]
                        time_kn = time_points[k + 1]

                        # Linear interpolation
                        td = time_kn - time_k
                        q0 = p[k]
                        pose = q0 + (t - time_k) / td * (p[k + 1] - q0)
                    else:
                        if (
                            segment_idx > 0
                            and t >= self.paths[r][segment_idx - 1].path.time[-1]
                        ):
                            # If t is past the previous segment's end, and not in current, take last of previous.
                            pose = self.paths[r][segment_idx - 1].path.path[-1]

                if pose is None:
                    logger.debug("pose is none")
                    logger.debug(t, r)

                poses.append(pose)

        return poses

    def get_end_times(self, robots: List[str]):
        end_times = {}

        for r in robots:
            if len(self.paths[r]) > 0:
                end_times[r] = self.paths[r][-1].path.time[-1]
            else:
                end_times[r] = 0

        return end_times

    def get_non_escape_end_times(self, robots: List[str]):
        end_times = {}

        for r in robots:
            if len(self.paths[r]) > 0:
                if self.paths[r][-1].task_index == -1:
                    end_times[r] = self.paths[r][-2].path.time[-1]
                else:
                    end_times[r] = self.paths[r][-1].path.time[-1]
            else:
                end_times[r] = 0

        return end_times

    def add_path(
        self,
        robots: List[str],
        path: Path,
        next_mode: Optional[Mode],
        is_escape_path: bool = False,
    ):
        logger.debug("adding path to multi-robot-path")
        for r in robots:
            # get robot-path from the original path
            subpath = Path(
                task_index=path.task_index,
                path=path.path[r],
            )
            self.paths[r].append(subpath)

            final_time = path.path[r].time[-1]
            logger.debug("max_time of path:", final_time)

        # constructing the mode-sequence:
        # this is done simply by adding the next mode to the sequence
        if not is_escape_path:
            self.timed_mode_sequence.append((final_time, next_mode))
            self.times.append(final_time)

    def remove_final_escape_path(self, robots: List[str]):
        for r in robots:
            if not self.paths[r]:
                continue
            if self.paths[r][-1].task_index == -1:
                self.paths[r] = self.paths[r][:-1]

    def get_final_time(self):
        T = 0
        for k, v in self.paths.items():
            if len(v) > 0:
                T = max(T, v[-1].path.time[-1])

        return T

    def get_final_non_escape_time(self):
        T = 0
        for k, v in self.paths.items():
            if len(v) > 0:
                if v[-1].task_index != -1:
                    T = max(T, v[-1].path.time[-1])
                else:
                    T = max(T, v[-2].path.time[-1])

        return T


def display_multi_robot_path(env: rai_env, path: MultiRobotPath, blocking=True):
    T = path.get_final_time()
    N = 5 * int(T)

    for i in range(N):
        t = i * T / (N - 1)
        poses = path.get_robot_poses_at_time(env.robots, t)
        mode = path.get_mode_at_time(t)
        env.set_to_mode(mode)
        print(t)
        env.show_config(env.start_pos.from_flat(np.concatenate(poses)), blocking=blocking)
        env.C.setJointState(np.concatenate(poses))

        if not env.is_collision_free(None, None):
            logger.info(f"Collision at time {t}")
            env.show(blocking=True)

        # env.show(blocking=False)

        time.sleep(0.01)


class Node:
    def __init__(self, t: float, q: Configuration):
        self.t = t
        self.q = q

        self.parent = None
        self.children = []


class Tree:
    def __init__(self, start: Optional[Node], reverse=False):
        self.reverse = reverse

        if start is not None:
            self.nodes = [start]
            self.configs = [[start.q[i] for i in range(start.q.num_agents())]]
        else:
            self.nodes = []
            self.configs = []

        self.batch_config_dist_fun = batch_config_dist

        self.prev_plans = None
        self.robots = None

        self.gamma = 0.7

    def batch_dist_fun(self, n1: Node, n2: List[Node], v_max):
        ts = np.array([1.0 * n.t for n in n2])

        if self.reverse:
            t_diff = ts - n1.t

        else:
            t_diff = n1.t - ts

        q_dist_for_vel = self.batch_config_dist_fun(n1.q, [n.q for n in n2])
        v = q_dist_for_vel / t_diff

        mask = (t_diff < 0) | (abs(v) > v_max)

        if len(self.robots) > 1:
            end_times = self.prev_plans.get_end_times(self.robots)

            intermediate_poses_arr = np.zeros((len(n2), len(self.nodes[0].q.state())))
            for i in range(len(n2)):
                n = n2[i]
                offset = 0
                for j, r in enumerate(self.robots):
                    if n1.t > end_times[r] and n.t < end_times[r]:
                        p = self.prev_plans.get_robot_poses_at_time([r], n1.t)[0]
                    else:
                        # p = n.q[j]
                        p = self.configs[i][j]

                    dim = len(p)
                    intermediate_poses_arr[i, offset : offset + dim] = 1.0 * p
                    offset += dim

            q_dist_to_inter = self.batch_config_dist_fun(
                n1.q, intermediate_poses_arr, "max"
            )
            q_dist_from_inter = batch_config_cost(
                intermediate_poses_arr,
                np.array([n.q.state() for n in n2]),
                "max",
                tmp_agent_slice=n.q._array_slice,
            )

            dist = (q_dist_from_inter + q_dist_to_inter) * self.gamma + (
                1 - self.gamma
            ) * t_diff

        else:
            q_dist = self.batch_config_dist_fun(n1.q, [n.q for n in n2], "max")
            dist = q_dist * self.gamma + (1 - self.gamma) * t_diff

        dist[mask] = np.inf

        return dist

        # print(q_diff)

    def get_nearest_neighbor(self, node: Node, v_max) -> Optional[Node]:
        batch_dists = self.batch_dist_fun(node, self.nodes, v_max)
        batch_idx = np.argmin(batch_dists)

        if np.isinf(batch_dists[batch_idx]):
            return None

        # print(batch_dists[batch_idx], config_dist(node.q, self.nodes[batch_idx].q), node.t - self.nodes[batch_idx].t)

        # print('time', node.t)

        # if len(self.nodes) % 50 == 0 and len(self.nodes) > 0:
        # if len(self.nodes[0].q.state()) == 6:
        #     to_plt = [n.q.state()[3:5] for n in self.nodes]
        # else:
        #     to_plt = [n.q.state()[:2] for n in self.nodes]

        # fig = plt.figure(figsize=(12, 12))
        # ax = fig.add_subplot(projection='3d')

        # ax.scatter([a[0] for a in to_plt], [a[1] for a in to_plt], [a.t for a in self.nodes], c=batch_dists, cmap=mpl.colormaps["binary"])
        # ax.scatter(node.q.state()[3], node.q.state()[4], node.t, marker='x')
        # ax.scatter(self.nodes[batch_idx].q.state()[3], self.nodes[batch_idx].q.state()[4], self.nodes[batch_idx].t, color='red')

        # for i, n in enumerate(self.nodes):
        #     k = 2
        #     node_dists = self.batch_dist_fun(n, self.nodes, v_max)
        #     ind = np.argpartition(node_dists, k)[:k]

        #     for j in ind:
        #         if not np.isinf(node_dists[j]):
        #             ax.plot([n.q.state()[3], self.nodes[j].q.state()[3]], [n.q.state()[4], self.nodes[j].q.state()[4]], [n.t, self.nodes[j].t], color='black')

        # ax.set_xlim([-2, 2])
        # ax.set_ylim([-2, 2])

        # plt.show()

        # fig = plt.figure(figsize=(12, 12))
        # ax = fig.add_subplot(projection='3d')
        # for i, n in enumerate(self.nodes):
        #     if n.parent is not None:
        #         ax.plot([n.q.state()[0], n.parent.q.state()[0]], [n.q.state()[1], n.parent.q.state()[1]], [n.t, n.parent.t], color='black')

        # ax.set_xlim([-2, 2])
        # ax.set_ylim([-2, 2])

        # ax.set_xlabel('x')

        # ax.scatter(self.nodes[0].q.state()[0], self.nodes[0].q.state()[1], self.nodes[0].t, color='red')

        # plt.show()

        return self.nodes[batch_idx]

        # best_node = None
        # best_dist = None
        # for n in self.nodes:
        #     qd = config_dist(node.q, n.q)
        #     td = node.t - n.t

        #     if td < 0:
        #         continue

        #     if qd / td > v_max:
        #         continue

        #     if best_dist is None or qd < best_dist:
        #         best_node = n
        #         best_dist = qd

        # return best_node

    def get_near_neighbors(self, node: Node, k: int, v_max) -> List[Node]:
        node_list = self.nodes
        dists = self.batch_dist_fun(node, self.nodes, v_max)

        k_clip = min(k, len(node_list) - 1)
        topk = np.argpartition(dists, k_clip)[:k_clip]
        topk[np.argsort(dists[topk])]

        best_nodes = [node_list[i] for i in topk]
        return best_nodes

    def add_node(self, new_node: Node, parent: Optional[Node]) -> None:
        node_list = self.nodes
        node_list.append(new_node)

        self.configs.append([new_node.q[i] for i in range(new_node.q.num_agents())])

        if parent is not None:
            new_node.parent = parent
            parent.children.append(new_node)


# @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
def collision_free_with_moving_obs(
    env: BaseProblem,
    t,
    q,
    q_buffer,
    prev_plans,
    end_times,
    robots,
    other_robots,
):
    logger.debug(f"coll check for {robots} at time {t}")
    logger.debug("Config %s", q)

    mode = prev_plans.get_mode_at_time(t)
    # env.set_to_mode(mode)

    robot_poses = prev_plans.get_robot_poses_at_time(other_robots, t)
    for i, r in enumerate(other_robots):
        q_buffer[env.robot_idx[r]] = robot_poses[i]
    # if other_robots:
    # q_buffer[np.array([env.robot_idx[r] for r in other_robots])] = np.array(robot_poses).flatten()

    offset = 0
    for r in robots:
        dim = env.robot_dims[r]
        q_buffer[env.robot_idx[r]] = q[offset : offset + dim]
        offset += dim

    # env.C.setJointState(q_buffer)

    # if env.is_collision_free_np(q_buffer, mode):
    if env.is_collision_free(env.start_pos.from_flat(q_buffer), mode):
        # if t > 85:
        #     env.show(blocking=False)
        return True
    
    # if t > 85:
    #     env.show(blocking=False)
        
    # # involves_robot_we_plan_for = False
    # colls = env.C.getCollisions()
    # for c in colls:
    #     for r in robots:
    #         if (r in c[1] or r in c[0]) or ("obj" in c[0] or "obj" in c[1]):
    #             # involves_robot_we_plan_for = True
    #             return False
    #         # else:
    #         #     env.C.view(True)
    #         #     print(c)
    # env.C.view(True)
    # # print(c)

    # return True

    # if c[2] < 0:
    #     print(c)

    # colls = env.C.getCollisions()
    # for c in colls:
    #     for r in robots:
    #         if t < end_times[r] and (r in c[0] or r in c[1]) and c[2] < 0:
    #             print(c)
    # if global_collision_counter > 50000 and t > max([t_e for r, t_e in end_times.items()]):
    #     colls = env.C.getCollisions()
    #     for c in colls:
    #         if c[2] < 0:
    #             print(c)
    #     print(t)
    #     env.C.view(False)

    # print('col')

    return False


# @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
def edge_collision_free_with_moving_obs(
    env: BaseProblem,
    qs,
    qe,
    ts,
    te,
    prev_plans: MultiRobotPath,
    robots,
    end_times,
    resolution=0.1,
):
    logger.debug(f"edge check for {robots}")
    logger.debug(f"start/end time: {ts}, {te}")

    if te < ts:
        te, ts = ts * 1.0, te * 1.0
        qs, qe = copy.deepcopy(qe), copy.deepcopy(qs)

    # # compute discretizatoin step
    N = config_dist(qe, qs) / resolution
    tdiff = te - ts
    N = max(int(tdiff / 1), N)
    N = max(int(N), 10)

    # logger.info(f"{N}, {config_dist(qe, qs) / resolution}")

    # print(N)

    # N = sum(part_indices)

    # rel = False

    # if ts > 74 and ts < 75 and te > 84 and te < 85:
    #     print("AAAAAAAAA")
    #     print("AAAAAAAAA")
    #     print("AAAAAAAAA")

    #     print(N)

    #     rel = True

    times = [ts + tdiff * idx / (N - 1) for idx in range(N)]

    # in addition to the things above, we check the times at which we might start planning again
    non_escape_end_times = prev_plans.get_non_escape_end_times(env.robots)
    for r, additional_time_to_check in non_escape_end_times.items():
        if ts < additional_time_to_check < te or te < additional_time_to_check < ts:
            times.append(additional_time_to_check)

    times.sort()

    indices = [i for i in range(len(times))]

    start_interpolation_at_index = {}

    for r in robots:
        if ts < end_times[r] and te > end_times[r]:
            for idx in indices:
                t = times[idx]
                if t <= end_times[r]:
                    start_interpolation_at_index[r] = idx

        elif end_times[r] <= ts:
            start_interpolation_at_index[r] = 0
        elif end_times[r] > te:
            start_interpolation_at_index[r] = indices[-1]
        else:
            start_interpolation_at_index[r] = None

    q0s = {}
    q1s = {}
    qdiff = {}

    for i, r in enumerate(robots):
        q0s[i] = qs[i]
        q1s[i] = qe[i]
        qdiff[i] = qe[i] - qs[i]

    indices = generate_binary_search_indices(len(times))

    other_robots = []
    for r in env.robots:
        if r not in robots:
            other_robots.append(r)

    q_buffer = env.start_pos.state() * 1.0

    all_poses = [None] * len(times)

    for idx in indices:
        ql = []
        t = times[idx]

        default_interp = (t - ts) / tdiff

        for i, r in enumerate(robots):
            # robot_poses[r] = []
            if start_interpolation_at_index[r] == 0:
                q0 = q0s[i]
                # q1 = q1s[i]
                # qdiff = q1 - q0

                p = q0 + qdiff[i] * default_interp
                # robot_poses[r].append(p)
            else:
                logger.debug(f"interpolating {r}")
                if start_interpolation_at_index[r] >= idx:
                    p = prev_plans.get_robot_poses_at_time([r], t)[0] * 1.0
                else:
                    robot_start_interp_pose = prev_plans.get_robot_poses_at_time(
                        [r], times[start_interpolation_at_index[r]]
                    )[0]
                    time_for_interp_traversal = (
                        times[-1] - times[start_interpolation_at_index[r]]
                    )
                    interp = (
                        t - times[start_interpolation_at_index[r]]
                    ) / time_for_interp_traversal
                    q1 = q1s[i]
                    p = (
                        robot_start_interp_pose
                        + (q1 - robot_start_interp_pose) * interp
                    )

            ql.append(p)

        q = np.concatenate(ql)
        # all_poses.append(q)
        all_poses[idx] = q

        if not collision_free_with_moving_obs(
            env,
            t,
            q,
            q_buffer,
            prev_plans,
            end_times,
            robots,
            other_robots,
        ):
            # if rel:
            #     env.show(True)
            return False
        # if rel:
        #     env.show(True)
    
    # TODO: this is an ugly hack and should be done differently
    if len(robots) > 1:
        for i in range(len(all_poses)-1):
            if config_dist(qs.from_flat(all_poses[i]), qs.from_flat(all_poses[i+1])) > resolution * 2:
                return False

    return True


def plan_in_time_space(
    ptc,
    env: BaseProblem,
    t0,
    prev_plans: MultiRobotPath,
    robots,
    end_times,
    goal,
    t_lb,
    v_max=0.5,
) -> TimedPath:
    computation_start_time = time.time()

    max_iter = 50000

    conf_type = type(env.get_start_pos())

    logger.info("start_time", t0)
    logger.info("robots", robots)
    logger.info("earliest end time", t_lb)

    start_configuration = prev_plans.get_robot_poses_at_time(robots, t0)
    q0 = conf_type.from_list(start_configuration)

    logger.info("start state", q0.state())

    tree = Tree(Node(t0, q0))
    tree.prev_plans = prev_plans
    tree.robots = robots

    def steer(close_node: Node, rnd_node: Node, max_stepsize=30):
        if close_node.t > rnd_node.t:
            logger.warn("Time of rnd node is smaller than close node:")
            logger.warn(f"close node time {close_node.t}")
            logger.warn(f"rnd node time {rnd_node.t}")
            assert False

        t_diff = rnd_node.t - close_node.t
        q_diff = rnd_node.q.state() - close_node.q.state()
        length = config_dist(rnd_node.q, close_node.q)

        v = length / t_diff

        # if length < 1e-3:
        #     return None, None

        logger.debug(f"length {length}")

        if v > v_max:
            return None, None
            assert False

        # max_q = 1

        # if length < max_q:
        #     return rnd_node.t, rnd_node.q

        # t_req = 1/length * t_diff
        # t = close_node.t + t_req
        # q = close_node.q.state() + t_req * v * q_diff / length

        if t_diff < max_stepsize:
            # print('reached', rnd_node.q.state())
            return rnd_node.t, rnd_node.q

        # t_m = min(max_q / v_max, max_stepsize)
        t_m = min(max_stepsize * 1.0, t_diff)

        if length < 1e-3:
            return None, None
        # # print('not')

        t = close_node.t + t_m
        q = close_node.q.state() + t_m * v * q_diff / length

        # print('v', v)
        # print('scale', max_stepsize / length)
        # print('close in steer', close_node.q.state())
        # print('rnd_node in steer', rnd_node.q.state())
        # print('q_diff', q_diff)
        # print('q in steer', q)
        # if q[0] > 1:
        #     raise ValueError

        q_list = []
        offset = 0
        for r in robots:
            dim = env.robot_dims[r]
            q_list.append(q[offset : dim + offset])
            offset += dim

        return t, conf_type.from_list(q_list)

    sampled_goals = []

    def sample_goal(t_ub):
        t_rnd = np.random.rand() * (t_ub - t_lb) + t_lb
        q_goal = goal.sample(None)

        q_goal_as_list = []
        offset = 0
        for r in robots:
            dim = env.robot_dims[r]
            q_goal_as_list.append(q_goal[offset : offset + dim])
            offset += dim

        return t_rnd, conf_type.from_list(q_goal_as_list)

    def sample_uniform(t_ub, goal_sampling_probability=0.1):
        informed_sampling = True
        if informed_sampling:
            # sample from box
            q0_state = q0.state()
            qg_state = sampled_goals[0][1].state()

            mid = (q0_state + qg_state) / 2

            max_goal_time = max([g[0] for g in sampled_goals])

            c_max = (max_goal_time - t0) * v_max
            # c_min = config_dist(q0, sampled_goals[0][1])

            s = []
            i = 0
            for r in robots:
                idx = env.robot_idx[r]
                lims = env.limits[:, idx]

                for j in range(env.robot_dims[r]):
                    lo = mid[i] - c_max * 0.5
                    hi = mid[i] + c_max * 0.5

                    lo = max(lims[0, j], lo)
                    hi = min(lims[1, j], hi)

                    s.append(np.random.uniform(lo, hi))

                    i += 1

            conf = q0.from_flat(np.array(s))

            min_dt_from_start = config_dist(q0, conf) / v_max

            min_t_sample = t0 + min_dt_from_start
            max_t_sample = (
                max_goal_time - config_dist(conf, sampled_goals[0][1]) / v_max
            )

            if min_t_sample > max_t_sample:
                return max_goal_time + 1, q0.from_flat(np.random.rand(len(q0.state())))

            t_rnd = np.random.uniform(min_t_sample, max_t_sample)
        else:
            max_goal_time = max([g[0] for g in sampled_goals])

            t_rnd = np.random.rand() * (max_goal_time - t0) + t0

            if t_rnd < t0:
                raise ValueError

            q_rnd = []
            # rnd_config = env.sample_config_uniform_in_limits()
            # print(rnd_config.state())
            # for i, r in enumerate(env.robots):
            #     if r not in robots:
            #         continue
            for r in robots:
                idx = env.robot_idx[r]

                lims = env.limits[:, idx]

                # print(lims)

                dim = env.robot_dims[r]

                rnd_uni_0_1 = np.random.rand(dim)
                q = rnd_uni_0_1 * (lims[1, :] - lims[0, :]) + lims[0, :]

                q_rnd.append(q * 1.0)

            conf = conf_type.from_list(q_rnd)

        return t_rnd, conf

    def project_sample_to_preplanned_path(t, q):
        logger.debug("projecting")
        q_new = q

        for i, r in enumerate(robots):
            if end_times[r] >= t:
                pose = prev_plans.get_robot_poses_at_time([r], t)[0]
                q_new[i] = pose * 1.0

        return q_new

    latest_end_time = max([end_times[r] for r in robots])
    t_lb = max(latest_end_time + 1, t_lb)

    logger.debug(f"end_time: {t_lb}")

    # estimate distance
    start_poses = prev_plans.get_robot_poses_at_time(robots, t_lb)
    goal_pose = goal.sample(None)
    goal_config = []
    offset = 0
    for r in robots:
        dim = env.robot_dims[r]
        goal_config.append(goal_pose[offset : offset + dim])
        offset += dim
    d = config_dist(conf_type.from_list(goal_config), conf_type.from_list(start_poses))

    logger.debug("Goal pose", conf_type.from_list(goal_config).state())
    logger.debug("start/goal dist", d)

    # compute max time from it
    max_t = t_lb + 1 + (d / v_max) * 10

    escape_path_end_time = prev_plans.get_final_time()
    max_t = max(max_t, escape_path_end_time)

    logger.debug("start_times", end_times)
    logger.debug("max time", max_t)

    curr_t_ub = max([end_times[r] for r in robots]) + (d / v_max) * 3

    curr_t_ub = max(curr_t_ub, t_lb)

    logger.debug("times for lb ", t0, d / v_max)

    configurations = None

    sampled_times = []
    sampled_pts = []

    other_robots = []
    for r in env.robots:
        if r not in robots:
            other_robots.append(r)

    # print("Goal pose")
    # res = collision_free_with_moving_obs(
    #     env,
    #     t_lb,
    #     conf_type.from_list(goal_config).state(),
    #     env.start_pos.state() * 1.0,
    #     prev_plans,
    #     end_times,
    #     robots,
    #     other_robots,
    #     robot_joints,)
    # print(res)
    # env.C.view(True)

    # if not collision_free_with_moving_obs(
    #     env,
    #     t0,
    #     q0.state(),
    #     env.start_pos.state() * 1.0,
    #     prev_plans,
    #     end_times,
    #     robots,
    #     other_robots,
    # ):
    #     print("Start pose not feasible")
    #     env.C.view(True)
    #     assert False

    iter = 0
    while True:
        iter += 1
        if ptc.should_terminate(0, time.time() - computation_start_time):
            break

        # increase upper bound that we are sampling
        if iter % 50:
            curr_t_ub += 1
            curr_t_ub = min(curr_t_ub, max_t)

        # sample pt
        # sample time and position
        rnd = random.random()

        goal_sampling_probability = 0.1
        if len(sampled_goals) == 0 or rnd < goal_sampling_probability:
            t_rnd, q_sampled = sample_goal(curr_t_ub)
            sampled_goals.append((t_rnd, q_sampled))
            print(f"Adding goal at {t_rnd}")
        else:
            t_rnd, q_sampled = sample_uniform(curr_t_ub)

        q_rnd = project_sample_to_preplanned_path(t_rnd, q_sampled)

        logger.debug("rnd state,", q_rnd.state())

        # check if there is a chance that we can reach the goal (or the start)
        time_from_start = t_rnd - t0
        d_from_start = config_dist(q0, q_rnd, "max")
        if d_from_start / time_from_start > v_max:
            # print("pos not reachable from start")
            continue

        reachable_goal = False
        for tg, qg in sampled_goals:
            time_from_goal = tg - t_rnd
            # we do not need to plan to a time which does not yet have a goal that we can reach.
            if tg < t_rnd:
                continue

            if np.linalg.norm(qg.state() - q_rnd.state()) < 1e-3:
                reachable_goal = True
                break

            d_from_goal = config_dist(qg, q_rnd, "max")
            if d_from_goal / time_from_goal <= v_max:
                reachable_goal = True
                break

            # print("goal dists", d_from_goal, time_from_goal, d_from_goal/time_from_goal)

        if not reachable_goal:
            logger.debug("No reachable goal for the sampled node.")
            logger.debug("times", tg, t_rnd)

            continue

        # check if sample is valid
        if not collision_free_with_moving_obs(
            env,
            t_rnd,
            q_rnd.state(),
            env.start_pos.state() * 1.0,
            prev_plans,
            end_times,
            robots,
            other_robots,
        ):
            continue

        if t_rnd < t0:
            raise ValueError

        # find closest pt in tree
        n_close = tree.get_nearest_neighbor(Node(t_rnd, q_rnd), v_max)

        # if len(tree.nodes) > 200 and loopy_bois % 5 == 0:
        #     plt.figure()
        #     for n in tree.nodes:
        #         if n.parent is None:
        #             continue

        #         x0 = n.parent.q.state()[0]
        #         y0 = n.parent.q.state()[1]

        #         x1 = n.q.state()[0]
        #         y1 = n.q.state()[1]

        #         # print(x0, x1, y0, y1)

        #         plt.plot([x0, x1], [y0, y1])

        #     plt.figure()
        #     for n in tree.nodes:
        #         if n.parent is None:
        #             continue

        #         x0 = n.parent.q.state()[3]
        #         y0 = n.parent.q.state()[4]

        #         x1 = n.q.state()[3]
        #         y1 = n.q.state()[4]

        #         # print(x0, x1, y0, y1)

        #         plt.plot([x0, x1], [y0, y1])

        #     plt.show()

        # if len(q_rnd.state()) > 3:
        #     rnd_nodes = []
        #     for _ in range(5000):
        #         t_rnd, q_uni_rnd = uniform_sample(curr_t_ub)
        #         q_rnd = project_sample_to_preplanned_path(t_rnd, q_uni_rnd)

        #         if collision_free_with_moving_obs(t_rnd, q_rnd.state()):
        #             rnd_nodes.append(Node(t_rnd, q_rnd))

        #     fig = plt.figure(figsize=(12, 12))
        #     ax = fig.add_subplot(projection='3d')

        #     for i, n in enumerate(rnd_nodes):
        #         k = 10
        #         node_dists = tree.batch_dist_fun(n, rnd_nodes, v_max)
        #         ind = np.argpartition(node_dists, k)[:k]

        #         for j in ind[1:]:
        #             # print(n.t, rnd_nodes[j].t)
        #             if abs(n.t - rnd_nodes[j].t) < 1e-3:
        #                 continue
        #             if not np.isinf(node_dists[j]) and edge_collision_free_with_moving_obs(rnd_nodes[j].q, n.q, rnd_nodes[j].t, n.t):
        #                 ax.plot([n.q.state()[0], rnd_nodes[j].q.state()[0]], [n.q.state()[1], rnd_nodes[j].q.state()[1]], [n.t, rnd_nodes[j].t], color='black')

        #     ax.set_xlim([-2, 2])
        #     ax.set_ylim([-2, 2])

        #     ax.set_xlabel('x')

        #     plt.show()

        if n_close is None:
            continue

        added_pt = False
        q_new = None
        t_new = None

        extend = False
        if extend:
            t_goal = t_rnd
            q_goal = q_rnd

            n_prev = n_close
            steps = 0
            while True:
                steps += 1

                t_next, q_next = steer(n_prev, Node(t_goal, q_goal), 5)
                if t_next is None or q_next is None:
                    break

                q_next = project_sample_to_preplanned_path(t_next, q_next)

                if not collision_free_with_moving_obs(
                    env,
                    t_next,
                    q_next.state(),
                    env.start_pos.state() * 1.0,
                    prev_plans,
                    end_times,
                    robots,
                    other_robots,
                ):
                    break

                if edge_collision_free_with_moving_obs(
                    env,
                    n_prev.q,
                    q_next,
                    n_prev.t,
                    t_next,
                    prev_plans,
                    robots,
                    end_times,
                    resolution=env.collision_resolution,
                ):
                    # add to tree
                    tree.add_node(Node(t_next, q_next), n_prev)

                    n_prev = tree.nodes[-1]

                    added_pt = True
                    t_new = t_next
                    q_new = q_next

                    if np.linalg.norm(q_goal.state() - q_new.state()) < 1e-3:
                        break
                else:
                    break
        else:
            # steer towards pt
            t_new, q_new = steer(n_close, Node(t_rnd, q_rnd), max_stepsize=30)

            if t_new is None or q_new is None:
                continue

            q_new = project_sample_to_preplanned_path(t_new, q_new)

            # check if edge is collision-free
            if edge_collision_free_with_moving_obs(
                env,
                n_close.q,
                q_new,
                n_close.t,
                t_new,
                prev_plans,
                robots,
                end_times,
                resolution=env.collision_resolution,
            ):
                # add to tree
                tree.add_node(Node(t_new, q_new), n_close)

                sampled_times.append(t_new)
                sampled_pts.append(q_new.state())

                # if cnt % 100 == 0 and cnt > 0:
                #     plt.figure()
                #     plt.hist(sampled_times, bins=100)

                #     plt.figure()
                #     plt.scatter([s[0] for s in sampled_pts], [s[1] for s in sampled_pts])

                #     plt.figure()
                #     plt.scatter([s[3] for s in sampled_pts], [s[4] for s in sampled_pts])

                #     plt.show()

                added_pt = True
                logger.debug(f"succ at time {t_new}")
            else:
                # plt.plot([n_close.q.state()[0], q_new.state()[0]], [n_close.q.state()[1], q_new.state()[1]])
                logger.debug(f"failed at time {t_new}")
                logger.debug(f"Tree size {len(tree.nodes)}")

                # env.C.view(False)

        if (
            added_pt
            and goal.satisfies_constraints(q_new.state(), mode=None, tolerance=1e-5)
            and t_new > t_lb
        ):
            # plt.show()
            configurations = [q_new.state()]
            times = [t_new]

            p = n_close

            while p.parent is not None:
                configurations.append(p.q.state())
                times.append(p.t)

                p = p.parent

            configurations.append(p.q.state())
            times.append(p.t)

            # print(times[::-1])

            # print(configurations)
            # print(times)

            computation_end_time = time.time()
            logger.debug(f"Took {computation_end_time - computation_start_time}s")

            # for k in range(len(robots)):
            #     # plt.figure()
            #     # for n in tree.nodes:
            #     #     if n.parent is None:
            #     #         continue

            #     #     x0 = n.parent.q.state()[0 + k*2]
            #     #     y0 = n.parent.q.state()[1 + k*2]

            #     #     x1 = n.q.state()[0 + k*2]
            #     #     y1 = n.q.state()[1 + k*2]

            #     #     # print(x0, x1, y0, y1)

            #     #     plt.plot([x0, x1], [y0, y1])
            #     fig = plt.figure(figsize=(10, 8)) # Create a new figure for each robot
            #     # Add a 3D subplot to the figure
            #     ax = fig.add_subplot(111, projection='3d')
            #     ax.set_title(f"Robot {k} Path in 3D (t-component as Z)")
            #     ax.set_xlabel("X-coordinate")
            #     ax.set_ylabel("Y-coordinate")
            #     ax.set_zlabel("T-component")

            #     # Iterate through the nodes in the tree to plot the path segments
            #     for n in tree.nodes:
            #         if n.parent is None:
            #             continue # Skip the root node as it has no parent to draw a line from

            #         # Parent's coordinates and t-component
            #         dim = 3
            #         x0 = n.parent.q.state()[0 + k*dim] # X-coordinate of parent
            #         y0 = n.parent.q.state()[1 + k*dim] # Y-coordinate of parent
            #         z0 = n.parent.t             # T-component of parent (now Z)

            #         # Current node's coordinates and t-component
            #         x1 = n.q.state()[0 + k*dim] # X-coordinate of current node
            #         y1 = n.q.state()[1 + k*dim] # Y-coordinate of current node
            #         z1 = n.t             # T-component of current node (now Z)

            #         # Plot the line segment in 3D
            #         # The 'k*2' from your original code was likely for indexing into a flattened state
            #         # array that contained multiple robot states. Here, we assume n.q.state()
            #         # directly gives the x, y for the current robot, and n.t is its time.
            #         ax.plot([x0, x1], [y0, y1], [z0, z1], marker='o', linestyle='-')

            # plt.show()

            return TimedPath(time=times[::-1], path=configurations[::-1])

        if iter > max_iter:
            break

    if configurations is None:
        return None

    return None


def plan_in_time_space_bidirectional(
    ptc,
    env: BaseProblem,
    t0,
    prev_plans: MultiRobotPath,
    robots,
    end_times,
    goal,
    t_lb,
    v_max=0.5,
) -> TimedPath:
    computation_start_time = time.time()
    conf_type = type(env.get_start_pos())

    # print("start_time", t0)
    # print("robots", robots)
    # print("earliest end time", t_lb)

    start_configuration = prev_plans.get_robot_poses_at_time(robots, t0)
    q0 = conf_type.from_list(start_configuration)

    logger.info(f"start state {q0.state()}")

    t_fwd = Tree(Node(t0, q0))
    t_rev = Tree(None, reverse=True)

    t_fwd.prev_plans = prev_plans
    t_fwd.robots = robots

    t_rev.prev_plans = prev_plans
    t_rev.robots = robots

    t_a = t_fwd
    t_b = t_rev

    sampled_goals = []

    def steer(close_node: Node, rnd_node: Node, max_stepsize=30):
        if close_node.t > rnd_node.t:
            print(close_node.t)
            print(rnd_node.t)
            assert False

        t_diff = rnd_node.t - close_node.t
        q_diff = rnd_node.q.state() - close_node.q.state()
        length = config_dist(rnd_node.q, close_node.q)

        v = length / t_diff

        # if length < 1e-3:
        #     return None, None

        if v > v_max:
            return None, None
            assert False

        # max_q = 1

        # if length < max_q:
        #     return rnd_node.t, rnd_node.q

        # t_req = 1/length * t_diff
        # t = close_node.t + t_req
        # q = close_node.q.state() + t_req * v * q_diff / length

        if t_diff < max_stepsize:
            # print('reached', rnd_node.q.state())
            return rnd_node.t, rnd_node.q

        # t_m = min(max_q / v_max, max_stepsize)
        t_m = min(max_stepsize * 1.0, t_diff)

        if length < 1e-3:
            return None, None
        # # print('not')

        t = close_node.t + t_m
        q = close_node.q.state() + t_m * v * q_diff / length

        q_list = []
        offset = 0
        for r in robots:
            dim = env.robot_dims[r]
            q_list.append(q[offset : dim + offset])
            offset += dim

        return t, conf_type.from_list(q_list)

    # we still steer from close to rnd.
    # But we now assume that close node has a higher time.
    def reverse_steer(close_node: Node, rnd_node: Node, max_stepsize=30):
        if close_node.t < rnd_node.t:
            logger.warn("close", close_node.t)
            logger.warn("goal", rnd_node.t)
            assert False

        t_diff = close_node.t - rnd_node.t
        q_diff = rnd_node.q.state() - close_node.q.state()
        length = config_dist(rnd_node.q, close_node.q)

        v = length / t_diff

        if v > v_max:
            return None, None

        if t_diff < max_stepsize:
            return rnd_node.t, rnd_node.q

        t_m = min(max_stepsize * 1.0, t_diff)

        if length < 1e-3:
            return None, None

        t = close_node.t - t_m
        q = close_node.q.state() + t_m * v * q_diff / length

        q_list = []
        offset = 0
        for r in robots:
            dim = env.robot_dims[r]
            q_list.append(q[offset : dim + offset])
            offset += dim

        return t, conf_type.from_list(q_list)

    def sample_goal(t_ub):
        t_rnd = np.random.rand() * (t_ub - t_lb) + t_lb
        q_goal = goal.sample(None)

        q_goal_as_list = []
        offset = 0
        for r in robots:
            dim = env.robot_dims[r]
            q_goal_as_list.append(q_goal[offset : offset + dim])
            offset += dim

        return t_rnd, conf_type.from_list(q_goal_as_list)

    def sample_uniform():
        informed_sampling = True
        if informed_sampling:
            # sample from box
            q0_state = q0.state()
            qg_state = sampled_goals[0][1].state()

            mid = (q0_state + qg_state) / 2

            max_goal_time = max([g[0] for g in sampled_goals])

            c_max = (max_goal_time - t0) * v_max
            # c_min = config_dist(q0, sampled_goals[0][1])

            s = []
            i = 0
            for r in robots:
                idx = env.robot_idx[r]
                lims = env.limits[:, idx]

                for j in range(env.robot_dims[r]):
                    lo = mid[i] - c_max * 0.5
                    hi = mid[i] + c_max * 0.5

                    lo = max(lims[0, j], lo)
                    hi = min(lims[1, j], hi)

                    s.append(np.random.uniform(lo, hi))

                    i += 1

            conf = q0.from_flat(np.array(s))

            min_dt_from_start = config_dist(q0, conf) / v_max

            min_t_sample = t0 + min_dt_from_start
            max_t_sample = (
                max_goal_time - config_dist(conf, sampled_goals[0][1]) / v_max
            )

            if min_t_sample > max_t_sample:
                return max_goal_time + 1, q0.from_flat(np.random.rand(len(q0.state())))

            t_rnd = np.random.uniform(min_t_sample, max_t_sample)
        else:
            max_goal_time = max([g[0] for g in sampled_goals])

            t_rnd = np.random.rand() * (max_goal_time - t0) + t0

            if t_rnd < t0:
                raise ValueError

            q_rnd = []
            # rnd_config = env.sample_config_uniform_in_limits()
            # print(rnd_config.state())
            # for i, r in enumerate(env.robots):
            #     if r not in robots:
            #         continue
            for r in robots:
                idx = env.robot_idx[r]

                lims = env.limits[:, idx]

                # print(lims)

                dim = env.robot_dims[r]

                rnd_uni_0_1 = np.random.rand(dim)
                q = rnd_uni_0_1 * (lims[1, :] - lims[0, :]) + lims[0, :]

                # print('rnd val', rnd_uni_0_1)
                # print((lims[1, :] - lims[0, :]))
                # print(lims[0, :])
                # print('q', q)

                q_rnd.append(q * 1.0)
                # q_rnd.append(rnd_config[i])

            conf = conf_type.from_list(q_rnd)

            # print(q_rnd)

        return t_rnd, conf

    def project_sample_to_preplanned_path(t: float, q: Configuration) -> Configuration:
        q_new = q

        for i, r in enumerate(robots):
            if end_times[r] >= t:
                pose = prev_plans.get_robot_poses_at_time([r], t)[0]
                q_new[i] = pose * 1.0

        return q_new

    latest_end_time = max([end_times[r] for r in robots])
    t_lb = max(latest_end_time + 1, t_lb)

    logger.info(f"end_time: {t_lb}")

    # estimate distance
    start_poses = prev_plans.get_robot_poses_at_time(robots, t_lb)
    goal_pose = goal.sample(None)
    goal_config = []
    offset = 0
    for r in robots:
        dim = env.robot_dims[r]
        goal_config.append(goal_pose[offset : offset + dim])
        offset += dim
    d = config_dist(conf_type.from_list(goal_config), conf_type.from_list(start_poses))

    logger.debug(f"Goal pose {conf_type.from_list(goal_config).state()}")
    logger.debug(f"start/goal dist {d}")

    # compute max time from it
    max_t = t_lb + 1 + (d / v_max) * 10

    escape_path_end_time = prev_plans.get_final_time()
    max_t = max(max_t, escape_path_end_time)

    logger.debug(f"start_times{end_times}")
    logger.debug(f"max time{max_t}")

    # curr_t_ub = t_lb + (d / v_max) * 3
    curr_t_ub = max([end_times[r] for r in robots]) + (d / v_max) * 3
    # curr_t_ub = max_t

    curr_t_ub = max(curr_t_ub, t_lb)

    configurations = None

    other_robots = []
    for r in env.robots:
        if r not in robots:
            other_robots.append(r)

    # print("Goal pose")
    # res = collision_free_with_moving_obs(
    #     env,
    #     t_lb,
    #     conf_type.from_list(goal_config).state(),
    #     env.start_pos.state() * 1.0,
    #     prev_plans,
    #     end_times,
    #     robots,
    #     other_robots,
    #     robot_joints,)
    # print(res)
    # env.C.view(True)

    iter = 0
    while True:
        iter += 1

        t_rnd, q_rnd = sample_goal(max_t)
        if iter == 1:
            t_rnd = max_t

        res = collision_free_with_moving_obs(
            env,
            t_rnd,
            q_rnd.state(),
            env.start_pos.state() * 1.0,
            prev_plans,
            end_times,
            robots,
            other_robots,
        )

        if res:
            t_rev.add_node(Node(t_rnd, q_rnd), None)
            sampled_goals.append((t_rnd, q_rnd))

        if len(sampled_goals) > 0 and iter > 50:
            break

    if not collision_free_with_moving_obs(
        env,
        t0,
        q0.state(),
        env.start_pos.state() * 1.0,
        prev_plans,
        end_times,
        robots,
        other_robots,
    ):
        logger.warn("Start pose not feasible")
        # env.C.view(True)
        return None

    max_iters = 10000
    for iter in range(max_iters):
        tmp = t_a
        t_a = t_b
        t_b = tmp

        if ptc.should_terminate(0, time.time() - computation_start_time):
            break

        # increase upper bound that we are sampling
        if iter % 50:
            curr_t_ub += 1
            curr_t_ub = min(curr_t_ub, max_t)

        if iter % 500 == 0:
            logger.debug(f"iteration {iter}")
            logger.debug(f"num nodes in trees: {len(t_a.nodes)} {len(t_b.nodes)}")
            logger.debug(f"Current t_ub {curr_t_ub}")

        t_rnd, q_sampled = sample_uniform()

        q_rnd = project_sample_to_preplanned_path(t_rnd, q_sampled)

        # check if there is a chance that we can reach the goal (or the start)
        time_from_start = t_rnd - t0
        d_from_start = config_dist(q0, q_rnd, "max")
        if d_from_start / time_from_start > v_max:
            continue

        reachable_goal = False
        for tg, qg in sampled_goals:
            time_from_goal = tg - t_rnd
            # we do not need to plan to a time which does not yet have a goal that we can reach.
            if tg < t_rnd:
                continue

            if np.linalg.norm(qg.state() - q_rnd.state()) < 1e-3:
                reachable_goal = True
                break

            d_from_goal = config_dist(qg, q_rnd, "max")
            if d_from_goal / time_from_goal <= v_max:
                reachable_goal = True
                break

            # print("goal dists", d_from_goal, time_from_goal, d_from_goal/time_from_goal)

        if not reachable_goal:
            logger.warn("No reachable goal for the sampled node.")
            # print("times", tg, t_rnd)
            continue

        # check if sample is valid
        if not collision_free_with_moving_obs(
            env,
            t_rnd,
            q_rnd.state(),
            env.start_pos.state() * 1.0,
            prev_plans,
            end_times,
            robots,
            other_robots,
        ):
            # print('invalid sample')
            # env.show(True)
            # if iter >= 1000 and iter % 500 == 0:
            #     # for some reason there is some bug that seems fixed when displaying a conf once.
            #     # I do not know what the reason is. I suspect that I am somehow mishandling modes.
            #     # Not sure how that would be fixed by doing this though.
            #     # env.show(False)
            #     if isinstance(env, rai_env):
            #         env.C.view_recopyMeshes()
            # env.C.view_close()
            continue

        if t_rnd < t0:
            raise ValueError

        # find closest pt in tree
        n_close = t_a.get_nearest_neighbor(Node(t_rnd, q_rnd), v_max)

        assert n_close is not None

        # go in dir and check collision
        if t_a.reverse:
            t_new, q_new = reverse_steer(n_close, Node(t_rnd, q_rnd), max_stepsize=10)
        else:
            t_new, q_new = steer(n_close, Node(t_rnd, q_rnd), max_stepsize=10)

        if t_new is None or q_new is None:
            continue

        q_new = project_sample_to_preplanned_path(t_new, q_new)

        if edge_collision_free_with_moving_obs(
            env,
            n_close.q,
            q_new,
            n_close.t,
            t_new,
            prev_plans,
            robots,
            end_times,
            resolution=env.collision_resolution,
        ):
            # add to tree
            t_a.add_node(Node(t_new, q_new), n_close)  # add to tree

            n_close_opposite = t_b.get_nearest_neighbor(Node(t_rnd, q_rnd), v_max)

            assert n_close_opposite is not None

            # should we steer here, instead of attempting to connect?

            if edge_collision_free_with_moving_obs(
                env,
                q_new,
                n_close_opposite.q,
                t_new,
                n_close_opposite.t,
                prev_plans,
                robots,
                end_times,
                resolution=env.collision_resolution,
            ):
                logger.info("found a path")

                # extract path from first tree
                configurations = [q_new.state()]
                times = [t_new]

                p = n_close

                while p.parent is not None:
                    configurations.append(p.q.state())
                    times.append(p.t)

                    p = p.parent

                configurations.append(p.q.state())
                times.append(p.t)

                # extract path from other tree
                other_configurations = []
                other_times = []

                p = n_close_opposite
                while p.parent is not None:
                    other_configurations.append(p.q.state())
                    other_times.append(p.t)

                    p = p.parent

                other_configurations.append(p.q.state())
                other_times.append(p.t)

                # print("BBBBBBBB")
                # print(times[-1])
                # print(other_times[-1])
                # print(times[0])
                # print(other_times[0])

                configurations = configurations[::-1]
                times = times[::-1]

                path = configurations + other_configurations
                times = times + other_times

                if t_a.reverse:
                    path = path[::-1]
                    times = times[::-1]

                # insert endtimes of robots
                if len(robots) > 1:
                    for r in robots:
                        t_robot_end = end_times[r]
                        if t_robot_end > t0:
                            # find time interval that contains t_robot_end
                            for k in range(len(times)-1):
                                if times[k] <= t_robot_end <= times[k+1]:
                                    # linearly interpolate
                                    alpha = (t_robot_end - times[k]) / (times[k+1] - times[k])
                                    interpolated_pose = path[k] + alpha * (path[k+1] - path[k])
                                    p = interpolated_pose
                                    break

                            p_conf = q0.from_flat(p)
                            p_proj = project_sample_to_preplanned_path(t_robot_end, p_conf).state()

                            path.append(p_proj)
                            times.append(t_robot_end)
                    
                    # sort according to increasing times
                    sorted_indices = np.argsort(times)
                    path = [path[i] for i in sorted_indices]
                    times = [times[i] for i in sorted_indices]
                    
                # print(times)

                # return path
                return TimedPath(time=times, path=path)

    return None


def shortcut_with_dynamic_obstacles(
    env: BaseProblem, other_paths: MultiRobotPath, robots, path, max_iter=500
):
    logger.info("shortcutting")

    # print((path.time[0]))
    # print((path.time[-1]))

    ql = []
    offset = 0
    for r in env.robots:
        dim = env.robot_dims[r]
        if r in robots:
            ql.append(env.get_start_pos().state()[offset : offset + dim])
        offset += dim

    conf_type = type(env.get_start_pos())
    tmp_conf = conf_type.from_list(ql)

    def arr_to_config(q):
        return tmp_conf.from_flat(q)

    # new_path = copy.copy(path)

    discretized_path = []
    discretized_time = []

    # discretize path
    resolution = 0.1
    for i in range(len(path.path) - 1):
        # print('interpolating at index', i)
        q0 = arr_to_config(path.path[i])
        q1 = arr_to_config(path.path[i + 1])

        t0 = path.time[i]
        t1 = path.time[i + 1]

        dist = config_dist(q0, q1)
        N = int(dist / resolution)
        N = max(1, N)

        # if len(robots) > 1:
        #     print(t0)
        #     coll_free = env.is_collision_free_np(path.path[i], None)
        #     env.show()

        for j in range(N):
            q = []
            for k in range(q0.num_agents()):
                qr = q0[k] + (q1[k] - q0[k]) / N * j
                q.append(qr)

            discretized_path.append(np.concatenate(q))

            t = t0 + (t1 - t0) * j / N
            discretized_time.append(t)

            # if len(robots) > 1:
            #     print(t)
            #     coll_free = env.is_collision_free_np(np.concatenate(q), None)
            #     env.show()

    discretized_path.append(path.path[-1])
    discretized_time.append(path.time[-1])

    new_path = TimedPath(time=discretized_time, path=discretized_path)

    num_indices = len(new_path.path)
    end_times = other_paths.get_end_times(robots)

    # print(end_times)

    # start_time = time.time()

    # if len(robots) > 1:
    #     for k in range(len(new_path.path)):
    #         print(new_path.time[k])
    #         pose = env.start_pos.from_flat(new_path.path[k])
    #         env.is_collision_free(pose, env.start_mode)
    #         env.show(new_path.time[k] > 70)
    #         time.sleep(0.01)

    indices = {}
    offset = 0
    for r in robots:
        dim = env.robot_dims[r]
        indices[r] = np.arange(offset, offset + dim)
        offset += dim

    attempted_shortcuts = 0
    max_attempts = max_iter * 10
    for _ in range(max_attempts):
        if attempted_shortcuts > max_iter:
            break

        i = np.random.randint(0, num_indices)
        j = np.random.randint(0, num_indices)

        if i > j:
            i, j = j, i

        if abs(j - i) < 2:
            continue

        robot_idx_to_shortcut = np.random.randint(0, len(robots))
        robot_name_to_shortcut = robots[robot_idx_to_shortcut]

        # if robot_name_to_shortcut == "a1":
        #     continue

        # we skip this attempt of shortcutting if the attempt tries to shortcut a path before the end time of this robot
        if (
            new_path.time[i] < end_times[robot_name_to_shortcut]
            or new_path.time[j] < end_times[robot_name_to_shortcut]
        ):
            continue

        assert new_path.time[i] < new_path.time[j]

        q0 = arr_to_config(new_path.path[i])
        q1 = arr_to_config(new_path.path[j])

        t0 = new_path.time[i]
        t1 = new_path.time[j]

        # check if the shortcut improves cost
        if path_cost(
            [q0.state(), q1.state()],
            env.batch_config_cost,
            agent_slices=q0._array_slice,
        ) >= path_cost(
            new_path.path[i:j], env.batch_config_cost, agent_slices=q0._array_slice
        ):
            continue

        attempted_shortcuts += 1

        q0 = conf_type.from_list([q0[robot_idx_to_shortcut]])
        q1 = conf_type.from_list([q1[robot_idx_to_shortcut]])

        # append paths that are not involved
        if len(robots) > 1:
            tmp_other_paths = other_paths

            # tmp_other_paths = copy.deepcopy(other_paths)
            tmp_paths = {}
            for r in robots:
                if r != robot_name_to_shortcut:
                    ind = indices[r]
                    tmp_paths[r] = TimedPath(
                        path=[pt[ind] * 1.0 for pt in new_path.path],
                        time=copy.deepcopy(discretized_time),
                    )

            # uninvolved_indices = np.array([ind for r, ind in indices.items() if r != robots[robot_to_shortcut]]).flatten()
            # tmp_path = TimedPath(time=discretized_time, path=[pt[uninvolved_indices] for pt in new_path.path])
            tmp_other_paths.add_path(
                [r for r in robots if r != robot_name_to_shortcut],
                Path(path=tmp_paths, task_index=3),
                None,
                is_escape_path=True,
            )
        else:
            tmp_other_paths = other_paths

        # this is wrong for partial shortcuts atm.
        if edge_collision_free_with_moving_obs(
            env,
            q0,
            q1,
            t0,
            t1,
            tmp_other_paths,
            [robot_name_to_shortcut],
            end_times,
            resolution=env.collision_resolution,
        ):
            # prev_time = new_path.time[i]
            # prev_time_2 = new_path.time[j]
            # print(robot_name_to_shortcut)
            # print(prev_time, prev_time_2)
            for k in range(j - i):
                ql = []
                for r_idx, r in enumerate(robots):
                    if r_idx == robot_idx_to_shortcut:
                        q = q0[0] + (q1[0] - q0[0]) / (j - i) * k
                    else:
                        q = new_path.path[i + k][indices[r]] * 1.0
                    ql.append(q)

                new_path.path[i + k] = np.concatenate(ql)
                new_path.time[i + k] = new_path.time[i] + k / (j - i) * (
                    new_path.time[j] - new_path.time[i]
                )

                # if len(robots) > 1:
                #     res = env.is_collision_free_np(np.concatenate(ql), None)
                #     if res == False:
                #         print("AAAAAAAA")
                #     env.show(True)
                    # other_robots = [r for r in robots if r != robot_name_to_shortcut]
                    # other_poses = tmp_other_paths.get_robot_poses_at_time(other_robots, new_path.time[i+k])

                    # pose = []
                    # for r in robots:
                    #     if r == robot_name_to_shortcut:
                    #         pose.append()


            # assert new_path.time[i] == prev_time
            # assert new_path.time[j] == prev_time_2

            # if len(robots) > 1:
            #     for k in range(len(new_path.path)):
            #             env.show_config(env.start_pos.from_flat(new_path.path[k]), blocking=False)

        if len(robots) > 1:
            tmp_other_paths.remove_final_escape_path(
                [r for r in robots if r != robot_name_to_shortcut]
            )

    # print(arr_to_config(path.path[0]).state())
    # print(arr_to_config(new_path.path[0]).state())

    # print(arr_to_config(path.path[-1]).state())
    # print(arr_to_config(new_path.path[-1]).state())


    # print((path.time[0]))
    # print((new_path.time[0]))

    # print((path.time[-1]))
    # print((new_path.time[-1]))

    # assert np.isclose(arr_to_config(path.path[0]).state(),arr_to_config(new_path.path[0]).state())[0]
    # assert np.isclose(arr_to_config(path.path[-1]).state(),arr_to_config(new_path.path[-1]).state())[0]

    logger.info(
        f"original cost: {path_cost(path.path, env.batch_config_cost, agent_slices=q0._array_slice)}"
    )
    logger.info(f"Attempted shortcuts: {attempted_shortcuts}")
    logger.info(
        f"new cost: {path_cost(new_path.path, env.batch_config_cost, agent_slices=q0._array_slice)}",
    )

    info = {}

    return new_path, info


def plan_robots_in_dyn_env(
    config,
    ptc,
    env,
    t0,
    other_paths,
    robots,
    q0,
    end_times,
    goal,
    t_lb=-1,
    use_bidirectional_planner=True,
) -> Tuple[Optional[Dict[str, TimedPath]], Optional[NDArray]]:
    # plan
    if use_bidirectional_planner:
        path = plan_in_time_space_bidirectional(
            ptc, env, t0, other_paths, robots, end_times, goal, t_lb
        )
    else:
        path = plan_in_time_space(
            ptc, env, t0, other_paths, robots, end_times, goal, t_lb
        )

    if path is None:
        return None, None

    # if len(robots) == 1:
    #     plt.figure()

    #     x = [pt[0] for pt in path.path]
    #     y = [pt[1] for pt in path.path]

    #     plt.plot(x, y)

    # if len(robots) == 2:
    #     plt.figure()

    #     x = [pt[0] for pt in path.path]
    #     y = [pt[1] for pt in path.path]

    #     plt.plot(x, y)

    #     x = [pt[3] for pt in path.path]
    #     y = [pt[4] for pt in path.path]

    #     plt.plot(x, y)

    # postprocess
    if config.shortcut_iters > 0:
        postprocessed_path, info = shortcut_with_dynamic_obstacles(
            env, other_paths, robots, path, max_iter=config.shortcut_iters
        )
        path = postprocessed_path
        # display_multi_robot_path(env, path)

    # take the separate paths apart
    separate_paths = {}
    offset = 0
    logger.debug(f"end times {end_times}")
    for r in robots:
        dim = env.robot_dims[r]
        c_n = []
        per_robot_times = []
        for i in range(len(path.path)):
            if path.time[i] >= end_times[r]:
                per_robot_times.append(path.time[i])
                c_n.append(path.path[i][offset : offset + dim])

        offset += dim

        separate_paths[r] = TimedPath(time=per_robot_times, path=c_n)

    #     if len(robots) == 1:
    #         x = [pt[0] for pt in separate_paths[r].path]
    #         y = [pt[1] for pt in separate_paths[r].path]

    #         plt.plot(x, y)
    #         # plt.show()

    # if len(robots) == 2:
    #     for r in robots:
    #         x = [pt[0] for pt in separate_paths[r].path]
    #         y = [pt[1] for pt in separate_paths[r].path]

    #         plt.plot(x, y)
    # plt.show()

    return separate_paths, path.path[-1]


@dataclass
class PrioritizedPlannerConfig:
    # gamma: float = 0.7
    # distance_metric: str = "euclidean"
    use_bidirectional_planner: bool = True
    shortcut_iters: int = 500
    multirobot_shortcut_iters: int = 0


class PrioritizedPlanner(BasePlanner):
    def __init__(
        self,
        env: BaseProblem,
        config: PrioritizedPlannerConfig = PrioritizedPlannerConfig(),
    ):
        self.env = env
        self.config = config

    def plan(
        self,
        ptc: PlannerTerminationCondition,
        optimize: bool = True,
    ) -> Tuple[List[State] | None, Dict[str, Any]]:
        q0 = self.env.get_start_pos()
        m0 = self.env.get_start_mode()

        conf_type = type(self.env.get_start_pos())

        robots = self.env.robots

        # get a sequence for the tasks from the environment
        # this is a constraint that this planner has, we need to have a sequence for planning

        if self.env.spec.home_pose != SafePoseType.HAS_SAFE_HOME_POSE:
            raise ValueError("No safe home pose")

        # life is easier if we just assume that we get a sequence-env for now
        if (
            self.env.spec.dependency != DependencyType.FULLY_ORDERED
            and self.env.spec.dependency != DependencyType.UNORDERED
        ):
            raise ValueError

        sequence_cache = []
        skipped_sequences = 0

        info = {"costs": [], "times": [], "paths": []}

        best_path = None

        computation_start_time = time.time()
        while True:
            if ptc.should_terminate(0, time.time() - computation_start_time):
                break

            robot_paths = MultiRobotPath(q0, m0, robots)

            seq_index = 0

            success = False
            env = copy.deepcopy(self.env)
            env.C_cache = {}

            # plan for a single sequence
            sequence = env.get_sequence()
            logger.info(f"Planning for sequence \n{sequence}")

            # check if we planned for this sequence before
            if sequence in sequence_cache:
                logger.debug("Planned for this sequence before, skipping.")
                skipped_sequences += 1

                if skipped_sequences > 10:
                    break

                continue

            skipped_sequences = 0
            sequence_cache.append(sequence)

            # construct task id sequence
            task_id_sequence = [env.start_mode.task_ids]

            for s in range(len(sequence) - 1):
                ids = copy.deepcopy(task_id_sequence[-1])
                task_idx = sequence[s]
                task = env.tasks[task_idx]

                for robot_idx, r in enumerate(env.robots):
                    if r not in task.robots:
                        continue

                    for i in range(s + 1, len(sequence)):
                        next_task_idx = sequence[i]
                        next_task = env.tasks[next_task_idx]

                        if r in next_task.robots:
                            ids[robot_idx] = next_task_idx
                            break

                task_id_sequence.append(ids)

            while True:
                if ptc.should_terminate(0, time.time() - computation_start_time):
                    break

                # get next active task
                task_index = sequence[seq_index]
                task = env.tasks[task_index]
                involved_robots = task.robots

                logger.info(f"task name {task.name}")
                logger.info(f"task_index {task_index}")
                logger.info(f"robots: {involved_robots}")
                logger.info(f"sequence index {seq_index}")

                # figure out when this task can end at the earliest
                earliest_end_time = robot_paths.get_final_non_escape_time()
                logger.debug(f"earliest end time {earliest_end_time}")

                # remove escape path from plan

                robot_paths.remove_final_escape_path(involved_robots)

                end_times = robot_paths.get_end_times(involved_robots)
                t0 = min([v for k, v in end_times.items()])

                # get current robot position and last planned time
                logger.debug("Collecting start times")
                start_time = min([t for _, t in end_times.items()])
                start_pose = robot_paths.get_robot_poses_at_time(
                    involved_robots, start_time
                )

                # sample goal from the task
                task_goal = task.goal

                # plan actual task
                current_time = time.time()
                planning_ptc = RuntimeTerminationCondition(
                    ptc.max_runtime_in_s - (current_time - computation_start_time)
                )

                # display_multi_robot_path(env, robot_paths)

                path, final_pose = plan_robots_in_dyn_env(
                    self.config,
                    planning_ptc,
                    env,
                    t0,
                    robot_paths,
                    involved_robots,
                    start_pose,
                    end_times,
                    task_goal,
                    earliest_end_time,
                    use_bidirectional_planner=self.config.use_bidirectional_planner,
                )

                logger.debug(f"final_pose {final_pose}")

                if path is None:
                    logger.warn("Failed planning")
                    break

                # add plan to overall path
                prev_mode = robot_paths.get_mode_at_time(
                    robot_paths.get_final_non_escape_time()
                )
                curr_mode = prev_mode
                if not env.is_terminal_mode(curr_mode):
                    final_time = path[involved_robots[0]].time[-1]
                    logger.debug(f"start_time {path[involved_robots[0]].time[0]}")
                    logger.debug(f"final_time: {final_time}")
                    q = []
                    for r in env.robots:
                        if r in involved_robots:
                            q.append(path[r].path[-1])
                        else:
                            q.append(
                                robot_paths.get_robot_poses_at_time([r], final_time)[0]
                            )

                    logger.debug(f"curr mode {curr_mode}")
                    logger.debug(f"final pose in path {final_pose}")
                    next_modes = env.get_next_modes(
                        env.start_pos.from_list(q), curr_mode
                    )
                    if len(next_modes) > 1:
                        for next_mode in next_modes:
                            if next_mode.task_ids == task_id_sequence[seq_index + 1]:
                                break
                            # for this_id, next_id in zip(curr_mode.task_ids, next_mode.task_ids):
                            #     if this_id != next_id and next_id == sequence[seq_index+1]:
                            #         break
                    else:
                        assert len(next_modes) == 1
                        next_mode = next_modes[0]
                else:
                    next_mode = curr_mode

                robot_paths.add_path(
                    involved_robots, Path(path=path, task_index=task_index), next_mode
                )

                # if len(involved_robots) > 1:
                # display_multi_robot_path(env, robot_paths, blocking=False)


                # for r in env.robots:
                #     for p in robot_paths.paths[r]:
                #         print(p.path)

                # print("Drawing path")
                # final_time = robot_paths.get_final_time()
                # # plt.figure()
                # for r in robots:
                #     x = []
                #     y = []
                #     for t in np.arange(0, int(final_time), 0.1):
                #         p = robot_paths.get_robot_poses_at_time([r], t)[0]
                #         x.append(p[0])
                #         y.append(p[1])

                #     plt.plot(x, y)

                # plt.show()

                # for r in robots:
                #     print("___________________")
                #     print("R", r)
                #     tprev = 0
                #     for segment in robot_paths.paths[r]:
                #         print("next segment")
                #         print(segment.task_index)
                #         for t in segment.path.time:
                #             # if t < tprev:
                #             print(t)

                #             # tprev = t

                # print("A")
                # print(seq_index)
                # print(env.sequence)
                if seq_index + 1 >= len(sequence):
                    success = True
                    logger.info("Found a solution.")
                    break

                # plan escape path

                logger.info("planning escape path")
                escape_start_time = path[involved_robots[0]].time[-1]
                end_times = robot_paths.get_end_times(involved_robots)
                failed_escape_planning = False

                for r in involved_robots:
                    q_non_blocking = env.safe_pose[r]
                    escape_goal = SingleGoal(q_non_blocking)

                    escape_start_pose = robot_paths.get_robot_poses_at_time(
                        [r], escape_start_time
                    )

                    t0 = min([v for k, v in end_times.items()])

                    current_time = time.time()
                    escape_planning_ptc = RuntimeTerminationCondition(
                        ptc.max_runtime_in_s - (current_time - computation_start_time)
                    )

                    escape_path, _ = plan_robots_in_dyn_env(
                        self.config,
                        escape_planning_ptc,
                        env,
                        t0,
                        robot_paths,
                        [r],
                        escape_start_pose,
                        end_times,
                        escape_goal,
                        use_bidirectional_planner=self.config.use_bidirectional_planner,
                    )

                    if escape_path is None:
                        logger.warn("Failed escape path planning.")
                        failed_escape_planning = True
                        break

                    # add plan to overall path
                    robot_paths.add_path(
                        [r],
                        Path(path=escape_path, task_index=-1),
                        next_mode=None,
                        is_escape_path=True,
                    )

                if failed_escape_planning:
                    break

                # check if there is a task left to do
                # print("escape path")
                # display_multi_robot_path(env, robot_paths)

                seq_index += 1

            if False:
                for k, v in robot_paths.paths.items():
                    for p in v:
                        print(p.path.time)
                        print(p.path.path)

                display_multi_robot_path(env, robot_paths)

            # re-organize the data such that it is in the same format as before
            if success:
                path = []

                T = robot_paths.get_final_time()
                N = 5 * int(np.ceil(T))
                for i in range(N):
                    t = i * T / (N - 1)

                    q = robot_paths.get_robot_poses_at_time(env.robots, t)
                    config = conf_type.from_list(q)
                    mode = robot_paths.get_mode_at_time(t)

                    state = State(config, mode)

                    path.append(state)

                end_time = time.time()
                cost = path_cost(path, env.batch_config_cost)

                if len(info["costs"]) == 0 or info["costs"][-1] > cost:
                    info["costs"].append(cost)
                    info["times"].append(end_time - computation_start_time)
                    info["paths"].append(path)

                    best_path = copy.deepcopy(path)

                    logger.info(f"Added path with cost {cost}.")

                if self.config.multirobot_shortcut_iters > 0:
                    path_w_doubled_modes = []
                    for i in range(len(path)):
                        path_w_doubled_modes.append(path[i])

                        if i + 1 < len(path) and path[i].mode != path[i + 1].mode:
                            path_w_doubled_modes.append(
                                State(path[i].q, path[i + 1].mode)
                            )

                    path = path_w_doubled_modes

                    shortcut_path, info_shortcut = robot_mode_shortcut(
                        env,
                        path,
                        self.config.multirobot_shortcut_iters,
                        tolerance=env.collision_tolerance,
                        resolution=env.collision_resolution,
                    )
                    path = shortcut_path
                    cost = path_cost(path, env.batch_config_cost)

                end_time = time.time()

                if len(info["costs"]) == 0 or info["costs"][-1] > cost:
                    info["costs"].append(cost)
                    info["times"].append(end_time - computation_start_time)
                    info["paths"].append(path)

                    best_path = copy.deepcopy(path)

                    logger.info(f"Added path with cost {cost}.")

                    if not optimize:
                        break
                else:
                    logger.info(f"Not adding this sequence, cost to high: {cost}")

            if isinstance(env, rai_env):
                del env.C

        return best_path, info
