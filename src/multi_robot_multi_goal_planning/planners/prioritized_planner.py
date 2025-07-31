import numpy as np
import random
import time

from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)
from numpy.typing import NDArray

from collections import namedtuple
import copy

from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseProblem,
    Mode,
    State,
)
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
)
from multi_robot_multi_goal_planning.problems.rai_envs import rai_env
from multi_robot_multi_goal_planning.problems.rai_config import get_robot_joints
from multi_robot_multi_goal_planning.problems.configuration import (
    NpConfiguration,
    batch_config_dist,
    config_dist,
)
from multi_robot_multi_goal_planning.planners.baseplanner import BasePlanner

from multi_robot_multi_goal_planning.problems.util import *

import matplotlib.pyplot as plt
import matplotlib as mpl


TimedPath = namedtuple("Path", ["time", "path"])
Path = namedtuple("Path", ["path", "task_index", "next_task_index"])


class MultiRobotPath:
    def __init__(self, q0, m0, robots):
        self.robots = robots
        self.paths = {}
        self.q0 = q0
        self.m0 = m0

        # indicates which mode is active after a certain time
        self.timed_mode_sequence = [(0, self.m0)]

        for r in robots:
            self.paths[r] = []

    def get_mode_at_time(self, t):
        for i in range(len(self.timed_mode_sequence) - 1):
            mode_time = self.timed_mode_sequence[i]
            mode = self.timed_mode_sequence[i]

            if t >= mode_time and t < self.timed_mode_sequence[i + 1]:
                return mode

        return self.timed_mode_sequence[-1][1]
        # m = []
        # for i, r in enumerate(self.robots):
        #     if len(self.paths[r]) == 0:
        #         m.append(self.m0[i])
        #     else:
        #         robot_mode = self.m0[i]
        #         for robot_path in self.paths[r]:
        #             if t >= robot_path.path.time[0] and t <= robot_path.path.time[-1]:
        #                 robot_mode = robot_path.task_index
        #             elif t > robot_path.path.time[-1] and robot_path.next_task_index is not None:
        #                 robot_mode = robot_path.next_task_index

        #         m.append(robot_mode)

    def get_robot_poses_at_time(self, robots, t):
        poses = []

        # TODO: map tp correct index
        for i, r in enumerate(self.robots):
            if r not in robots:
                continue

            if len(self.paths[r]) == 0:
                poses.append(self.q0[i])
            else:
                pose = self.q0[i]
                for robot_path in self.paths[r]:
                    p = robot_path.path
                    if p.time[-1] <= t:
                        pose = p.path[-1]

                    if p.time[0] <= t and t <= p.time[-1]:
                        for k in range(len(p.time) - 1):
                            if p.time[k] <= t and t <= p.time[k + 1]:
                                td = p.time[k + 1] - p.time[k]
                                pose = p.path[k] + (t - p.time[k]) / td * (
                                    p.path[k + 1] - p.path[k]
                                )

                if pose is None:
                    print("pose is none")
                    print(t, r)

                poses.append(pose)

        return poses

    def get_end_times(self, robots):
        end_times = {}

        for r in robots:
            if len(self.paths[r]) > 0:
                end_times[r] = self.paths[r][-1].path.time[-1]
            else:
                end_times[r] = 0

        # print(end_times)

        return end_times

    def add_path(self, env, robots, path, next_task_indices):
        print("adding path to multi-robot-path")
        for r in robots:
            # get robot-path from the original path
            subpath = Path(
                task_index=path.task_index,
                path=path.path[r],
                next_task_index=next_task_indices[r],
            )
            self.paths[r].append(subpath)

        # constructing the mode-sequence:

    def add_next_task_to_final_paths(self, mode):
        for i, r in enumerate(self.robots):
            self.paths[r][-1].next_task_index = mode[i]

    def remove_final_escape_path(self, robots):
        for r in robots:
            if len(self.paths[r]) == 0:
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


def display_multi_robot_path(env: rai_env, path: MultiRobotPath):
    T = path.get_final_time()
    N = 5 * int(T)

    for i in range(N):
        t = i * T / (N - 1)
        poses = path.get_robot_poses_at_time(env.robots, t)
        mode = path.get_mode_at_time(t)
        env.set_to_mode(mode)
        env.show_config(np.concatenate(poses))

        time.sleep(0.01)


class Node:
    def __init__(self, t, q):
        self.t = t
        self.q = q

        self.parent = None
        self.children = []


class Tree:
    def __init__(self, start: Node):
        self.root = start
        # self.nodes = [self.root]

        print("root:")
        print(self.root)

        self.nodes = [self.root]
        self.batch_config_dist_fun = batch_config_dist

    def batch_dist_fun(self, n1: Node, n2: List[Node], v_max):
        ts = np.array([1.0 * n.t for n in n2])
        t_diff = n1.t - ts

        # print("time diffs")
        # print(n1.t)
        # print(np.array([n.t for n in n2]))
        # print(tdiff)

        q_dist = self.batch_config_dist_fun(n1.q, [n.q for n in n2])

        v = q_dist / t_diff

        mask = (t_diff < 0) | (abs(v) > v_max)

        # print(q_diff)

        q_dist[mask] = np.inf
        t_diff[mask] = np.inf

        # speed_filter[np.isinf(tdiff)] = -np.inf

        # return -speed_filter

        l = 0.7
        # dist = q_dist
        dist = q_dist * l + (1 - l) * t_diff
        return dist

        # t_diff[q_dist > 2] = np.inf
        # q_dist[t_diff > 10] = np.inf
        # return q_dist

        # print('t_diff')
        # print(t_diff)
        # print('q')
        # print(q_dist)
        # print('v')
        # print(v)

        # print('dist')
        # print(dist)

        return dist

        # print(q_diff)

    def get_nearest_neighbor(self, node: Node, v_max) -> Node:
        # print("node")
        # print(node)
        # print('nodes')
        # print(self.nodes)
        batch_dists = self.batch_dist_fun(node, self.nodes, v_max)
        batch_idx = np.argmin(batch_dists)

        if np.isinf(batch_dists[batch_idx]):
            return None

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

    def add_node(self, new_node: Node, parent: Node) -> None:
        node_list = self.nodes
        node_list.append(new_node)

        new_node.parent = parent
        parent.children.append(new_node)


def collision_free_with_moving_obs(
    env, t, q, prev_plans, end_times, robots, robot_joints, task_idx
):
    mode = prev_plans.get_mode_at_time(t)

    for r in robots:
        if end_times[r] <= t:
            idx = env.robots.index(r)
            mode[idx] = task_idx

    env.set_to_mode(mode)

    # set robots to their planned position for time
    robot_poses = prev_plans.get_robot_poses_at_time(env.robots, t)
    for i, r in enumerate(env.robots):
        if r not in robots:
            pose = robot_poses[i]
            joint_names = robot_joints[r]
            env.C.setJointState(pose, joint_names)

            # print('setting to prev pos')

    # actually check if it is collision free
    curr_robot_joint_names = []
    for r in robots:
        curr_robot_joint_names.extend(robot_joints[r])

    env.C.setJointState(q, curr_robot_joint_names)

    # if mode == [4, 4] or mode == [1, 4]:
    #     env.show(True)

    # if t > 10.0:
    #     env.show(True)

    # if mode == [4, 4]:
    #     env.show(True)

    # if t < t0:
    #     raise ValueError

    # env.show(False)

    if env.is_collision_free(None, mode):
        return True

    # colls = env.C.getCollisions()
    # for c in colls:
    #     if c[2] < 0:
    #         print(c)

    # print('col')

    return False


def edge_collision_free_with_moving_obs(
    env: BaseProblem,
    qs,
    qe,
    ts,
    te,
    prev_plans,
    robots,
    end_times,
    joint_names,
    task_idx,
    resolution=0.1,
):
    conf_type = type(env.get_start_pos())

    # print("A", ts, te)
    # print('edge check')
    assert ts < te

    # # compute discretizatoin step
    N = config_dist(qe, qs) / resolution
    tdiff = te - ts
    N = max(int(tdiff / 1), N)
    N = max(int(N), 10)

    # N = sum(part_indices)

    indices = [i for i in range(N)]
    times = [ts + (te - ts) * idx / (N - 1) for idx in range(N)]

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

    # print(times)
    # print("interp start", start_interpolation_at_index)

    robot_poses = {}
    for i, r in enumerate(robots):
        robot_poses[r] = []
        for idx in indices:
            t = times[idx]
            if start_interpolation_at_index[r] == 0:
                p = qs[i] + (qe[i] - qs[i]) * (t - ts) / (te - ts)
                # robot_poses[r].append(p)
            else:
                # print(f'interpolating {r}')
                if start_interpolation_at_index[r] >= idx:
                    p = prev_plans.get_robot_poses_at_time(r, t)[0] * 1.0
                else:
                    robot_start_interp_pose = robot_poses[r][
                        start_interpolation_at_index[r]
                    ]
                    time_for_interp_traversal = (
                        times[-1] - times[start_interpolation_at_index[r]]
                    )
                    interp = (
                        t - times[start_interpolation_at_index[r]]
                    ) / time_for_interp_traversal
                    # print('interp scle', interp)
                    p = (
                        robot_start_interp_pose
                        + (qe[i] - robot_start_interp_pose) * interp
                    )

                    # if config_dist(
                    #     NpConfiguration.from_list([qe[i]]), NpConfiguration.from_list([robot_start_interp_pose])) / (time_for_interp_traversal) > v_max:
                    #     return False

            robot_poses[r].append(p)

    indices = generate_binary_search_indices(len(indices)).copy()

    for idx in indices:
        # todo this interpolatoin is wrong if we need to project to the manifold
        t = times[idx]

        ql = []
        for i, r in enumerate(robots):
            ql.append(robot_poses[r][idx])

        q = np.concatenate(ql)

        if not collision_free_with_moving_obs(
            env, t, q, prev_plans, end_times, robots, joint_names, task_idx
        ):
            return False

    return True


def plan_in_time_space(
    env: BaseProblem,
    prev_plans: MultiRobotPath,
    robots,
    task_idx,
    end_times,
    goal,
    t_lb,
):
    max_iter = 50000
    t0 = min([v for k, v in end_times.items()])
    # t0 = max([v for k, v in end_times.items()])
    # t0 = prev_plans.get_final_time()

    conf_type = type(env.get_start_pos())

    start_configuration = prev_plans.get_robot_poses_at_time(robots, t0)
    q0 = conf_type.from_list(start_configuration)

    print("start state", q0.state())

    # for i, r in enumerate(robots):
    #     pose = q0[i]
    #     joint_names = get_robot_joints(env.C, r)
    #     env.C.setJointState(pose, joint_names)
    # env.show()

    print("start_time", t0)

    tree = Tree(Node(t0, q0))

    v_max = 0.5

    print(q0)

    robot_joints = {}
    for r in env.robots:
        robot_joints[r] = get_robot_joints(env.C, r)

    def steer(close_node: Node, rnd_node: Node, max_stepsize=30):
        if close_node.t > rnd_node.t:
            print(close_node.t)
            print(rnd_node.t)
            assert False

        t_diff = rnd_node.t - close_node.t
        q_diff = rnd_node.q.state() - close_node.q.state()
        length = config_dist(rnd_node.q, close_node.q)

        v = length / t_diff

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

        # # print('not')

        t = close_node.t + max_stepsize
        q = close_node.q.state() + max_stepsize * v * q_diff / length

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

    def uniform_sample(t_ub, goal_sampling_probability=0.05):
        rnd = random.random()

        if rnd < goal_sampling_probability:
            t_rnd = np.random.rand() * (t_ub - t_lb) + t_lb
            q_goal = goal.sample(None)

            q_goal_as_list = []
            offset = 0
            for r in robots:
                dim = env.robot_dims[r]
                q_goal_as_list.append(q_goal[offset : offset + dim])
                offset += dim

            sampled_goals.append((t_rnd, conf_type.from_list(q_goal_as_list)))

            return t_rnd, conf_type.from_list(q_goal_as_list)

        t_rnd = np.random.rand() * (t_ub - t0) + t0

        if t_rnd < t0:
            raise ValueError

        q_rnd = []
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

        # print(q_rnd)

        return t_rnd, conf_type.from_list(q_rnd)

    def project_sample_to_preplanned_path(t, q):
        q_new = q

        for i, r in enumerate(robots):
            if end_times[r] >= t:
                pose = prev_plans.get_robot_poses_at_time(r, t)[0]
                q_new[i] = pose * 1.0
                # print("projecting")

        return q_new

    latest_end_time = max([end_times[r] for r in robots])
    t_lb = max(latest_end_time, t_lb)

    print("end_time: ", t_lb)

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

    # compute max time from it
    max_t = t_lb + (d / v_max) * 50

    print("start_times", end_times)
    print("max time", max_t)

    curr_t_ub = t_lb + (d / v_max) * 3
    # curr_t_ub = max_t

    configurations = None

    sampled_times = []
    sampled_pts = []

    cnt = 0
    loopy_bois = 0
    while True:
        loopy_bois += 1
        # increase upper bound that we are sampling
        if cnt % 10:
            curr_t_ub += 1
            curr_t_ub = min(curr_t_ub, max_t)

        if cnt % 500 == 0:
            print("cnt", cnt)
            print(len(tree.nodes))
            print(loopy_bois)

        # sample pt
        # sample time and position
        t_rnd, q_uni_rnd = uniform_sample(curr_t_ub)
        q_rnd = project_sample_to_preplanned_path(t_rnd, q_uni_rnd)

        # check if there is a chance that we can reach the goal (or the start)
        time_from_start = t_rnd - t0
        d_from_start = config_dist(q0, q_rnd)
        if d_from_start / time_from_start > v_max:
            continue

        reachable_goal = False
        for tg, qg in sampled_goals:
            if np.linalg.norm(qg.state() - q_rnd.state()) < 1e-3:
                reachable_goal = True
                break

            time_from_goal = tg - t_rnd
            if tg < t_rnd:
                continue

            d_from_goal = config_dist(qg, q_rnd)
            if d_from_goal / time_from_goal < v_max:
                reachable_goal = True
                break

        if not reachable_goal:
            continue

        # check if sample is valid
        if not collision_free_with_moving_obs(
            env,
            t_rnd,
            q_rnd.state(),
            prev_plans,
            end_times,
            robots,
            robot_joints,
            task_idx,
        ):
            # print('invalid sample')
            # env.show(True)
            continue

        if t_rnd < t0:
            raise ValueError

        # find closest pt in tree
        n_close = tree.get_nearest_neighbor(Node(t_rnd, q_rnd), v_max)

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
            # print('no close node')
            continue

        # for i, r in enumerate(robots):
        #     pose = n_close.q[i]
        #     joint_names = get_robot_joints(env.C, r)
        #     env.C.setJointState(pose, joint_names)

        # env.show(False)

        # print(t_rnd - n_close.t)

        added_pt = False
        q_new = None
        t_new = None

        extend = True
        if extend:
            t_goal = t_rnd
            q_goal = q_rnd

            n_prev = n_close
            steps = 0
            while True:
                steps += 1

                t_next, q_next = steer(n_prev, Node(t_goal, q_goal), 10)
                if t_next is None:
                    break

                q_next = project_sample_to_preplanned_path(t_next, q_next)

                if not collision_free_with_moving_obs(
                    env,
                    t_next,
                    q_next.state(),
                    prev_plans,
                    end_times,
                    robots,
                    robot_joints,
                    task_idx,
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
                    robot_joints,
                    task_idx,
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
            t_new, q_new = steer(n_close, Node(t_rnd, q_rnd))

            # # check if sample is valid
            # if not collision_free_with_moving_obs(env, t_new, q_new.state(), prev_plans, end_times, robots, robot_joints, task_idx):
            #     # print('invalid sample')
            #     # env.show(True)
            #     continue

            # if t_new < t0:
            #     raise ValueError

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
                robot_joints,
                task_idx,
            ):
                # if len(q_new.state()) > 3 and q_new[0] > 0.1:
                #     for i, r in enumerate(robots):
                #         pose = q_new[i]
                #         joint_names = get_robot_joints(env.C, r)
                #         env.C.setJointState(pose, joint_names)

                #     env.show(True)

                # print(np.linalg.norm(goal.sample() - q_rnd.state()))

                # neighbors = t.get_near_neighbors()

                # for n in neighbors:
                #     # check if there is a cheaper path
                #     pass

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

                # if goal.satisfies_constraints(q_new.state(), 0.1):
                #     print("A")
                #     if t_new < t_lb:
                #         print("B")

                added_pt = True

        if (
            added_pt
            and goal.satisfies_constraints(q_new.state(), mode=None, tolerance=0.1)
            and t_new > t_lb
        ):
            configurations = [q_new.state()]
            times = [t_new]

            p = n_close

            while p.parent is not None:
                configurations.append(p.q.state())
                times.append(p.t)

                p = p.parent

            configurations.append(p.q.state())
            times.append(p.t)

            # print(configurations)
            # print(times)

            return TimedPath(time=times[::-1], path=configurations[::-1])

        if cnt > max_iter:
            break

        cnt += 1

    if configurations is None:
        return None

    times = 0
    configurations = 0
    return TimedPath(time=times, path=configurations)


def interpolate_in_time_space(
    env: rai_env, robots, paths: MultiRobotPath, q0, end_times, goal, t_lb
):
    print("interpolating")
    N = 5
    v_max = 1

    # q0 = np.concatenate(q0)
    # t0 = min([t for _, t in end_times.items()])

    print("start_pose", q0)
    qg = goal.sample(None)
    print("goal pose", qg)

    offset = 0
    t_max_goal = t_lb
    for i, r in enumerate(robots):
        pose = paths.get_robot_poses_at_time(r, end_times[r])
        dim = env.robot_dims[r]

        print("dim", dim)

        qg_robot = qg[offset : offset + dim]
        offset += dim

        pose_diff_robot = qg_robot - pose

        robot_dist = np.linalg.norm(pose_diff_robot)

        robot_t0 = end_times[r]
        t_goal = max(t_lb, t_max_goal)

        t_diff = t_goal - robot_t0

        if t_diff <= 0:
            t_diff = 1e-6

        v = robot_dist / t_diff

        if v > v_max:
            v = v_max
            t_goal = robot_t0 + robot_dist / v

        t_max_goal = max(t_goal, t_max_goal)

    new_paths = {}
    offset = 0
    for i, r in enumerate(robots):
        pose = paths.get_robot_poses_at_time(r, end_times[r])[0]
        dim = env.robot_dims[r]

        qg_robot = qg[offset : offset + dim]

        pose_diff_robot = qg_robot - pose

        robot_dist = np.linalg.norm(pose_diff_robot)

        robot_t0 = end_times[r]

        configurations = [pose + pose_diff_robot * i / (N - 1) for i in range(N)]
        times = [robot_t0 + (t_max_goal - robot_t0) * i / (N - 1) for i in range(N)]

        new_paths[r] = TimedPath(times, configurations)

        offset += dim

    # # split paths for robots
    # paths = {}
    # offset = 0
    # for r in robots:
    #     dim = env.robot_dims[r]
    #     paths[r] = TimedPath(times, [c[offset : offset + dim] for c in configurations])
    #     offset += dim

    # remove parts of paths where they were not available yet

    return new_paths


def plan_in_time_space_bidirectional(
    env: rai_env,
    prev_plans: MultiRobotPath,
    robots,
    task_idx,
    q0,
    end_times,
    goal,
    t_lb,
):
    pass


def shortcut_with_dynamic_obstacles(
    env: BaseProblem, other_paths: MultiRobotPath, robots, path, task_idx, max_iter=500
):
    # costs = [path_cost(new_path, env.batch_config_cost)]
    # times = [0.0]

    conf_type = type(env.get_start_pos())

    def arr_to_config(q):
        offset = 0
        ql = []
        for r in robots:
            dim = env.robot_dims[r]
            ql.append(q[offset : offset + dim])
            offset += dim
        return conf_type.from_list(ql)

    def arr_to_state(q):
        return State(arr_to_config(q), None)

    def arrs_to_states(qs):
        configs = [arr_to_state(q) for q in qs]
        return configs

    robot_joints = {}
    for r in env.robots:
        robot_joints[r] = get_robot_joints(env.C, r)

    new_path = copy.copy(path)

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

        for j in range(N):
            q = []
            for k in range(q0.num_agents()):
                qr = q0[k] + (q1[k] - q0[k]) / N * j
                q.append(qr)

                # env.C.setJointState(qr, get_robot_joints(env.C, env.robots[k]))

                # env.C.setJointState(qr, [env.robots[k]])

            # env.C.view(True)

            discretized_path.append(np.concatenate(q))

            t = t0 + (t1 - t0) * j / N
            discretized_time.append(t)

    discretized_path.append(path.path[-1])
    discretized_time.append(path.time[-1])

    new_path = TimedPath(time=discretized_time, path=discretized_path)

    num_indices = len(new_path.path)
    end_times = other_paths.get_end_times(robots)

    # start_time = time.time()

    cnt = 0
    for _ in range(max_iter):
        i = np.random.randint(0, num_indices)
        j = np.random.randint(0, num_indices)

        if i > j:
            tmp = i
            i = j
            j = tmp

        if abs(j - i) < 2:
            continue

        skip = False
        for r in robots:
            if np.sign(new_path.time[i] - end_times[r]) != np.sign(
                new_path.time[j] - end_times[r]
            ):
                skip = True
                break

        if skip:
            continue

        q0 = arr_to_config(new_path.path[i])
        q1 = arr_to_config(new_path.path[j])

        t0 = new_path.time[i]
        t1 = new_path.time[j]

        # check if the shortcut improves cost
        if path_cost(
            arrs_to_states([q0.state(), q1.state()]), env.batch_config_cost
        ) >= path_cost(arrs_to_states(new_path.path[i:j]), env.batch_config_cost):
            continue

        cnt += 1

        robots_to_shortcut = [r for r in range(len(robots))]
        if False:
            random.shuffle(robots_to_shortcut)
            num_robots = np.random.randint(0, len(robots_to_shortcut))
            robots_to_shortcut = robots_to_shortcut[:num_robots]

        # this is wrong for partial shortcuts atm.
        if edge_collision_free_with_moving_obs(
            env, q0, q1, t0, t1, other_paths, robots, end_times, robot_joints, task_idx
        ):
            for k in range(j - i):
                ql = []
                for r in robots_to_shortcut:
                    q = q0[r] + (q1[r] - q0[r]) / (j - i) * k
                    ql.append(q)
                new_path.path[i + k] = np.concatenate(ql)
                new_path.time[i + k] = new_path.time[i] + k / (j - i) * (
                    new_path.time[j] - new_path.time[i]
                )

        # env.show(True)

        # current_time = time.time()
        # times.append(current_time - start_time)
        # costs.append(path_cost(new_path, env.batch_config_cost))

    print("original cost:", path_cost(arrs_to_states(path.path), env.batch_config_cost))
    print("Attempted shortcuts: ", cnt)
    print("new cost:", path_cost(arrs_to_states(new_path.path), env.batch_config_cost))

    info = {}

    return new_path, info


def plan_robots_in_dyn_env(
    env, other_paths, robots, task_idx, q0, end_times, goal, t_lb=-1
):
    # plan
    path = plan_in_time_space(env, other_paths, robots, task_idx, end_times, goal, t_lb)

    # print("planning in dyn env")
    # print("start time", t0)
    # print("start pose", q0)
    # path = interpolate_in_time_space(env, robots, other_paths, q0, t0, goal, t_lb)

    # postprocess
    postprocessed_path, info = shortcut_with_dynamic_obstacles(
        env, other_paths, robots, path, task_idx
    )
    path = postprocessed_path

    # take the separate paths apart
    separate_paths = {}
    offset = 0
    print(end_times)
    for r in robots:
        dim = env.robot_dims[r]
        c_n = []
        per_robot_times = []
        for i in range(len(path.path)):
            if path.time[i] >= end_times[r]:
                per_robot_times.append(path.time[i])
                c_n.append(path.path[i][offset : offset + dim])

        if end_times[r] > path.time[0]:
            c_n.insert(0, other_paths.get_robot_poses_at_time(r, end_times[r])[0])
            per_robot_times.insert(0, end_times[r])

        offset += dim

        separate_paths[r] = TimedPath(time=per_robot_times, path=c_n)

    return separate_paths


class PrioritizedPlanner(BasePlanner):
    def __init__(
        self,
        env: BaseProblem,
    ):
        self.env = env

    def plan(
        self,
        ptc: PlannerTerminationCondition,
        optimize: bool = True,
    ) -> Tuple[List[State] | None, Dict[str, Any]]:
        env = self.env

        q0 = env.get_start_pos()
        m0 = env.get_start_mode()

        conf_type = type(env.get_start_pos())

        robots = env.robots

        robot_paths = MultiRobotPath(q0, m0, robots)

        # def sample_non_blocking(current_robots, next_robots, next_goal, mode):
        #     next_robots_joint_names = []
        #     for r in next_robots:
        #         next_robots_joint_names.extend(get_robot_joints(env.C, r))

        #     curr_robots_joint_names = []
        #     for r in current_robots:
        #         curr_robots_joint_names.extend(get_robot_joints(env.C, r))

        #     lims = env.limits[
        #         :, env.robot_idx[env.robots[env.robots.index(next_robots[0])]]
        #     ]

        #     while True:
        #         qg = next_goal.sample(mode)
        #         env.C.setJointState(qg, next_robots_joint_names)

        #         q = (
        #             np.random.rand(env.robot_dims[next_robots[0]])
        #             * (lims[1, :] - lims[0, :])
        #             + lims[0, :]
        #         )
        #         env.C.setJointState(q, curr_robots_joint_names)

        #         if env.is_collision_free(None, mode):
        #             return q

        #         # env.C.view(True)

        seq_index = 0

        # get a sequence for the tasks from the environment
        # this is a constraint that this planner has, we need to have a sequence for planning
        sequence = env.get_task_sequence()

        computation_start_time = time.time()

        while True:
            # get next active task
            task_index = sequence[seq_index]
            task = env.tasks[task_index]
            involved_robots = task.robots

            print("task name", task.name)
            print("robots:", involved_robots)

            # remove escape path from plan
            robot_paths.remove_final_escape_path(involved_robots)

            # for k,v in robot_paths.paths.items():
            #     for p in v:
            #         print(p.path.time)
            #         print(p.path.path)

            # get current robot position and last planned time
            print("Collecting start times")
            end_times = robot_paths.get_end_times(involved_robots)
            start_time = min([t for _, t in end_times.items()])
            print("Collecting start poses")
            start_pose = robot_paths.get_robot_poses_at_time(
                involved_robots, start_time
            )

            # sample goal from the task
            task_goal = task.goal

            # figure out when this task can end at the earliest
            earliest_end_time = robot_paths.get_final_non_escape_time()

            # plan actual task
            path = plan_robots_in_dyn_env(
                env,
                robot_paths,
                involved_robots,
                task_index,
                start_pose,
                end_times,
                task_goal,
                earliest_end_time,
            )

            if path is None:
                print("Failed")
                return

            # add plan to overall path
            prev_mode = robot_paths.get_mode_at_time(
                robot_paths.get_final_non_escape_time()
            )
            curr_mode = prev_mode.copy()
            for i, r in enumerate(involved_robots):
                curr_mode[i] = task_index
            next_modes = env.get_next_modes(None, curr_mode)
            assert len(next_modes) == 1
            next_mode = next_modes[0]

            next_task_indices = {}
            for i, r in enumerate(involved_robots):
                next_task_indices[r] = next_mode[i]

            robot_paths.add_path(
                env,
                involved_robots,
                Path(path=path, task_index=task_index, next_task_index=None),
                next_task_indices=next_task_indices,
            )

            robot_done = False
            if robot_done:
                continue

            # print("A")
            # print(seq_index)
            # print(env.sequence)
            if seq_index + 1 >= len(sequence):
                break

            # plan escape path

            # TODO: escape paths always need to be planned separately for each agent
            if False:
                print("planning escape path")

                # sample a position that we feel like is not gonna be in the way to reach the next goal
                next_task_index = sequence[seq_index + 1]
                next_task = env.tasks[next_task_index]
                next_task_robots = next_task.robots
                next_task_goal = next_task.goal
                mode = []
                q_non_blocking = sample_non_blocking(
                    involved_robots, next_task_robots, next_task_goal, mode
                )
                escape_goal = SingleGoal(q_non_blocking)

                escape_start_time = path[involved_robots[0]].time[-1]
                escape_start_pose = robot_paths.get_robot_poses_at_time(
                    involved_robots, escape_start_time
                )

                escape_path = plan_robots_in_dyn_env(
                    env,
                    robot_paths,
                    involved_robots,
                    task_index,
                    escape_start_pose,
                    escape_start_time,
                    escape_goal,
                )

                print("escape path")
                print(escape_path)

                if escape_path is None:
                    print("Failed")
                    return

                # add plan to overall path
                robot_paths.add_path(
                    involved_robots, Path(path=escape_path, task_index=-1)
                )

            # check if there is a task left to do

            seq_index += 1

        if False:
            print("displaying")
            for k, v in robot_paths.paths.items():
                for p in v:
                    print(p.path.time)
                    print(p.path.path)

            display_multi_robot_path(env, robot_paths)

        # re-organize the data such that it is in the same format as before
        path = []

        T = robot_paths.get_final_time()
        N = 5 * int(T)
        for i in range(N):
            t = i * T / N

            q = robot_paths.get_robot_poses_at_time(env.robots, t)
            config = conf_type.from_list(q)
            mode = robot_paths.get_mode_at_time(t)

            state = State(config, mode)

            path.append(state)

        end_time = time.time()
        cost = path_cost(path, env.batch_config_cost)

        info = {"costs": [cost], "times": [end_time - computation_start_time]}

        return path, info