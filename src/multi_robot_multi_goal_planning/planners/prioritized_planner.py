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
from multi_robot_multi_goal_planning.problems.core.goals import SingleGoal
from .termination_conditions import (
    PlannerTerminationCondition,
    RuntimeTerminationCondition,
)
from .shortcutting import robot_mode_shortcut
from multi_robot_multi_goal_planning.problems.rai.rai_envs import rai_env
from multi_robot_multi_goal_planning.problems.core.configuration import (
    Configuration,
    batch_config_dist,
    config_dist,
    batch_config_cost,
)
from .baseplanner import BasePlanner

from multi_robot_multi_goal_planning.problems.util import path_cost

import logging

"""
NOTES (Liam):
=============================================================================================
SUMMARY

Core idea: 
- Plan one task at a time instead of all simultaneously (computationally expensive)
- Robot involved in the current task are planned jointly 
- All other robots (with already planned paths) are treated as moving obstacles

Key steps:
1. Get task sequence: Determine which task to do in which order (from oracle/task planner)
2. For each task in sequence:
    - Identify which robots are involved
    - Plan their paths while avoiding robots that already have plans
    - Add an "escape path" to move robots to safe poses afterward
3. Iterate: Try different task sequences to find better solutions
=============================================================================================
"""
logger = logging.getLogger(__name__)

TimedPath = namedtuple("TimedPath", ["time", "path"])
Path = namedtuple("Path", ["path", "task_index"])

"""
NOTES (Liam):
Remove later also comments within code, keep for now for better understanding
=============================================================================================
MULTIROBOTPATH

- Stores the complete planned paths for all robots over time, built incrementally as robots are
  planned one by one in prioritized planning
- Answers questions: where robot X at time t? what mode is active at time t?
=============================================================================================
"""
class MultiRobotPath:
    # Allocate space for fixed set of attributes (unlike __dict__)
    # Memory saving & faster attribute access 
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

        # Indicates which mode is active after a certain time (until the next one)
        self.timed_mode_sequence = [(0, self.m0)]
        self.times = [0]

        for r in robots:
            self.paths[r] = []
    
    def get_mode_at_time(self, t: float) -> Mode:
        """Return active mode at time t"""
        if t >= self.times[-1]:
            return self.timed_mode_sequence[-1][1]

        idx = bisect.bisect_right(self.times, t) # ([0,10,20,30],15) -> 2

        return self.timed_mode_sequence[idx - 1][1] # 2 = mode_B

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def get_robot_poses_at_time(self, robots: List[str], t: float):
        """
        Calculates the exact configuration of a given list of robots at a specific time
        """
        poses = []

        for r in robots:
            i = self.robot_ids[r]

            # CASE 1: robot has NO path (not planned yet) -> stay at initial position
            if not self.paths[r]:
                poses.append(self.q0_split[i])
            else:
                # CASE 2: query time AFTER all paths END -> return last pose of last path segment
                # Why? Robot stays stationary after completing all tasks
                if t >= self.paths[r][-1].path.time[-1]:
                    pose = self.paths[r][-1].path.path[-1]
                
                # CASE 3: query time BEFORE all paths START -> return initial pose
                elif t <= self.paths[r][0].path.time[0]:
                    pose = self.q0_split[i]
                
                # CASE 4: query time WITHIN the path -> find which segment + interpolate within
                else:
                    path_end_times = [rp.path.time[-1] for rp in self.paths[r]]
                    segment_idx = bisect.bisect_left(path_end_times, t) # Path segment time falls into
                    rp = self.paths[r][segment_idx]
                    
                    # 4A. query time == segment end time -> return last pose of that segment
                    if t == rp.path.time[-1] and segment_idx < len(self.paths[r]) - 1:
                        # If 't' is exactly the end time of a segment that is not the very last segment,
                        # we take the last pose of *that* segment.
                        pose = rp.path.path[-1]
                    
                    # 4B. query time IN segment -> linear interpolation within segment
                    elif rp.path.time[0] <= t < rp.path.time[-1]:
                        # 't' is within the current segment (exclusive of the end time)
                        p = rp.path.path
                        time_points = rp.path.time

                        # Find the SPECIFIC time interval within the segment
                        k = bisect.bisect_right(time_points, t) - 1

                        time_k = time_points[k]
                        time_kn = time_points[k + 1]

                        # Linear interpolation
                        td = time_kn - time_k
                        q0 = p[k]
                        pose = q0 + (t - time_k) / td * (p[k + 1] - q0)
                    
                    # 4C. edge case handling
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
        """Returns dict of when each robot's last path ends"""
        end_times = {}

        for r in robots:
            if len(self.paths[r]) > 0:
                # Get the last time of the last path segment of robot r
                end_times[r] = self.paths[r][-1].path.time[-1]
            else:
                end_times[r] = 0

        return end_times

    def get_non_escape_end_times(self, robots: List[str]):
        """Get end times excluding escape plans (task_index == -1)"""
        end_times = {}

        for r in robots:
            if len(self.paths[r]) > 0:
                if self.paths[r][-1].task_index == -1: # Escape path
                    end_times[r] = self.paths[r][-2].path.time[-1] # [-2] is last non-escape path segment
                else:
                    end_times[r] = self.paths[r][-1].path.time[-1] # [-1] is last when there's no escape path segment
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
        """
        Add a new path segment for one or more robots
        
        Example: 
        - Robot 1 finishes task, picks up object -> mode changes
          path = plan_to_goal...
          multi_robot_path.add_path(robot1, path, next_mode=mode_with_obj, is_escape=False)
        - Robot 2 does temporary escape motion -> mode unchanged
          escape_path = plan_escape...
          multi_robot_path.add_path(robot2, escape_path, next_mode=None, is_escape=True)
        """
        logger.debug("adding path to multi-robot-path")
        for r in robots:
            # Get robot-path from the original path
            subpath = Path(
                task_index=path.task_index,
                path=path.path[r],
            )
            self.paths[r].append(subpath)

            # All the paths always have the same end time if multiple robots are involved
            final_time = path.path[r].time[-1]
            logger.debug("max_time of path:", final_time)

        # Constructing the mode-sequence:
        # This is done simply by adding the next mode to the sequence
        if not is_escape_path:
            self.timed_mode_sequence.append((final_time, next_mode))
            self.times.append(final_time)

    def remove_final_escape_path(self, robots: List[str]):
        """Remove escape path that are no longer needed (if better plan is found)"""
        for r in robots:
            if not self.paths[r]:
                continue
            if self.paths[r][-1].task_index == -1:
                self.paths[r] = self.paths[r][:-1] # Keep all but last (remove escape)

    def get_final_time(self):
        """Get the makespan (when the last robot finishes)"""
        T = 0
        for k, v in self.paths.items():
            if len(v) > 0:
                T = max(T, v[-1].path.time[-1])

        return T

    def get_final_non_escape_time(self):
        """Get the makespan excluding escape paths"""
        T = 0
        for k, v in self.paths.items():
            if len(v) > 0:
                if v[-1].task_index != -1: # No escape paths
                    T = max(T, v[-1].path.time[-1])
                else: # Don't consider escape paths
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

        time.sleep(0.01)

"""
NOTES (Liam):
Remove later also comments within code, keep for now for better understanding
=============================================================================================
NODE and TREE

Node:
- Single space-time (q,t) point in search tree
- Tree links: parent node + list of children nodes

Tree:
- Manages nodes with space-time metric that, 1) respects temporal causality, 2) enforces velocity
  limits, and 3) accounts for multi-robot coordination timing
=============================================================================================
"""
class Node:
    def __init__(self, t: float, q: Configuration):
        self.t = t
        self.q = q
        self.parent = None
        self.children = []

class Tree:
    def __init__(self, start: Optional[Node], reverse=False):
        self.reverse = reverse # For bidirectional search

        if start is not None: # Checks if tree initialized with root
            self.nodes = [start]
            self.configs = [[start.q[i] for i in range(start.q.num_agents())]]
        else:
            self.nodes = []
            self.configs = []

        self.batch_config_dist_fun = batch_config_dist

        self.prev_plans = None
        self.robots = None # Depending on task, contains 1 or more robots ()

        self.gamma = 0.7

    def batch_dist_fun(self, n1: Node, n2: List[Node], v_max):
        """
        Space time distance metric
        Calculates the "distance" from a query node n1 (random sample) to a list of existing
        tree nodes n2 (n). Filters out physically impossible connections
        """
        # Step 1: Time direction check
        ts = np.array([1.0 * n.t for n in n2])
        if self.reverse:
            t_diff = ts - n1.t # Reverse (backward) tree (parent.t > child.t) -> t_diff < 0
        else:
            t_diff = n1.t - ts # Forward tree (parent.t < child.t) -> t_diff > 0

        # Step 2: Velocity feasibility check
        q_dist_for_vel = self.batch_config_dist_fun(n1.q, [n.q for n in n2])
        v = q_dist_for_vel / t_diff # Required velocities to connect n1 with n2's

        # Mask for infeasible connections (can't violate physics and causality)
        mask = (t_diff < 0) | (abs(v) > v_max)

        """
        1. Get release times for each robot being planned (time when it finishes previously assigned path)
        2. Prepare intermediate waypoints for each (n1,n2) pair 
        3. Iterate through nearest nodes and robots and compute distance between nodes
        3A. If segment crosses moment where robot ends prior task -> p = "release point"
        3B. Else -> p = new proposed point
        4. Compute distances
        """
        # Step 3A: Planning for multiple robots with different end times
        if len(self.robots) > 1:
            end_times = self.prev_plans.get_end_times(self.robots)
            intermediate_poses_arr = np.zeros((len(n2), len(self.nodes[0].q.state())))
            
            # Iterate through every node of tree n2
            # Check if the connection n to n1 crosses the robot's end_time
            for i in range(len(n2)):
                n = n2[i]
                offset = 0
                for j, r in enumerate(self.robots):
                    if n1.t > end_times[r] and n.t < end_times[r]:
                        # Robot locked to previous path (segment n-n1 crosses robot's release time)
                        # Timeline: n --(old path)-- end_times[r] --(new path)-- n1 
                        # Set intermediate point p to release pose (very last pose from robot's old path)
                        p = self.prev_plans.get_robot_poses_at_time([r], n1.t)[0]
                    else:
                        # Robot is already free (finished previous task)
                        # OR robot still busy (for entire duration) 
                        # Set intermediate point p to start point n.q 
                        p = self.configs[i][j]

                    dim = len(p)
                    intermediate_poses_arr[i, offset : offset + dim] = 1.0 * p
                    offset += dim
            
            # Distance from sample to intermediate points 
            q_dist_to_inter = self.batch_config_dist_fun(
                n1.q, intermediate_poses_arr, "max"
            )

            # Distance from intermediate points back to actual tree nodes n2
            q_dist_from_inter = batch_config_cost(
                intermediate_poses_arr,
                np.array([n.q.state() for n in n2]),
                "max",
                tmp_agent_slice=n.q._array_slice,
            )

            # Final weighted space-time distance 
            dist = (q_dist_from_inter + q_dist_to_inter) * self.gamma + (
                1 - self.gamma) * t_diff

        # Step 3B: Planning for single robot
        else:
            # Weighted straight space-time metric (distance) from n1 to each node in n2
            q_dist = self.batch_config_dist_fun(n1.q, [n.q for n in n2], "max")
            dist = q_dist * self.gamma + (1 - self.gamma) * t_diff

        # Infeasible connections -> mark with infinite distance
        dist[mask] = np.inf

        return dist

    def get_nearest_neighbor(self, node: Node, v_max) -> Optional[Node]:
        """Find closest node in the tree to connect a new sample to"""
        batch_dists = self.batch_dist_fun(node, self.nodes, v_max)
        batch_idx = np.argmin(batch_dists)

        if np.isinf(batch_dists[batch_idx]):
            return None
        
        return self.nodes[batch_idx]

    def add_node(self, new_node: Node, parent: Optional[Node]) -> None:
        """Add a node to the tree and establish parent-child relationship"""
        node_list = self.nodes
        node_list.append(new_node)

        self.configs.append([new_node.q[i] for i in range(new_node.q.num_agents())])

        if parent is not None:
            new_node.parent = parent
            parent.children.append(new_node)

"""
NOTES (Liam):
Remove later also comments within code, keep for now for better understanding
=============================================================================================
COLLISION CHECK

Collision free with moving obstacles:
- Check if the configuration is collision-free at time t, consider other robots as moving obstacles

Edge collision free with moving obstacles:
- Check if an entire edge (path segment) is collision free over time 
=============================================================================================
"""
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
    """
    Collision check with robots (active and others)
    """
    logger.debug(f"coll check for {robots} at time {t}")
    logger.debug("Config %s", q)

    # Step 1: Get mode (what objects are attached, etc.) at time t
    mode = prev_plans.get_mode_at_time(t)

    # Step 2: Get positions of other robots at time t from their existing plans 
    robot_poses = prev_plans.get_robot_poses_at_time(other_robots, t)
    
    # Step 3: Build complete configuration: other robots + robots being planned
    for i, r in enumerate(other_robots):
        q_buffer[env.robot_idx[r]] = robot_poses[i] # Other robots

    offset = 0
    for r in robots:
        dim = env.robot_dims[r]
        q_buffer[env.robot_idx[r]] = q[offset : offset + dim] # Robots being planned
        offset += dim

    # Step 4: Check collision
    if env.is_collision_free(env.start_pos.from_flat(q_buffer), mode):
        return True
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
    """
    Collision check of a straight-line path for a set of robots between a start and end configuration and time
    """
    logger.debug(f"edge check for {robots}")
    logger.debug(f"start/end time: {ts}, {te}")

    if te < ts: # Handles edges going backwards in time (for bidirectional planning)
        te, ts = ts * 1.0, te * 1.0
        qs, qe = copy.deepcopy(qe), copy.deepcopy(qs)

    # Step 1: Discretize the edge into N checkpoints
    N = config_dist(qe, qs) / resolution # A check for every resolution unit of distance in config space
    tdiff = te - ts
    N = max(int(tdiff / 1), N) # At least one check per time unit
    N = max(int(N), 10) # Minimum 10 checkpoints

    # Step 2: Generate uniformly spaced time points from ts to te
    times = [ts + tdiff * idx / (N - 1) for idx in range(N)]
        
    # Step 3: Critical time points (when robot finishes tasks -> mode transitions happen)
    # Why? Crucial because a mode transition (e.g., robot picks up object) can change 
    # the collision geometry (discontinuously). Need to check exactly then
    non_escape_end_times = prev_plans.get_non_escape_end_times(env.robots)
    for r, additional_time_to_check in non_escape_end_times.items():
        if ts < additional_time_to_check < te or te < additional_time_to_check < ts:
            times.append(additional_time_to_check)

    times.sort()

    indices = [i for i in range(len(times))]

    start_interpolation_at_index = {}

    # Step 4: Determine the interpolation strategy 
    # At which time index we should start interpolating toward qe?
    for r in robots:
        if ts < end_times[r] and te >= end_times[r]:
            # CASE 1: Robot r finishes DURING this edge
            # Start interpolating from where the robot finished toward qe
            for idx in indices:
                t = times[idx]
                if t <= end_times[r]:
                    start_interpolation_at_index[r] = idx

        elif end_times[r] <= ts:
            # CASE 2: robot r finishes BEFORE this edge started
            # Start interpolating from the beginning of the new planned edge toward qe
            start_interpolation_at_index[r] = 0
        elif end_times[r] > te:
            # CASE 3: robot r finishes AFTER this edge is started 
            # Don't interpolate, simply follow previous plan for all time samples
            start_interpolation_at_index[r] = indices[-1]
        else:
            start_interpolation_at_index[r] = None

    # Step 5: Store start/end configurations
    q0s = {}
    q1s = {}
    qdiff = {}

    for i, r in enumerate(robots):
        q0s[i] = qs[i]
        q1s[i] = qe[i]
        qdiff[i] = qe[i] - qs[i]

    # Step 6: Check middle points first for early collision detection
    # Instead of checking [0, 1, 2... 10] we check [5, 2, 8, 1, ...]
    indices = generate_binary_search_indices(len(times))

    # Identify other robots (robots: being planned, other_robots: moving obstacles)
    other_robots = []
    for r in env.robots:
        if r not in robots:
            other_robots.append(r)

    # Buffer to build the full configuration of all robots for the env's check
    q_buffer = env.start_pos.state() * 1.0
    all_poses = [None] * len(times)

    # Step 8-9: Main collision check loop  
    for idx in indices:
        ql = []
        t = times[idx]

        default_interp = (t - ts) / tdiff # Interpolation param [0,1]

        # Step 9A: Compute each robot position at time t 
        for i, r in enumerate(robots):
            # CASE 1: Robot hasn't finished (normal case) -> interpolate from qs to qe
            if start_interpolation_at_index[r] == 0:
                q0 = q0s[i]
                p = q0 + qdiff[i] * default_interp
            # CASE 2: Robot finished during or before the edge    
            else:
                logger.debug(f"interpolating {r}")
                if start_interpolation_at_index[r] >= idx:
                    # Before robot finished -> use its previous plan
                    # Robot was free from the start -> simple linear interpolation 
                    p = prev_plans.get_robot_poses_at_time([r], t)[0] * 1.0
                else:
                    # Robot finished -> interpolate from finish pose to final edge pose
                    # Robot has a mixed strategy
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

        # Step 10: Collision check
        q = np.concatenate(ql) # Combine all robot poses into one config
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
            return False # Collision found

    # Final sanity check: if generated path doesn't have large discontinuous jumps
    # TODO: this is an ugly hack and should be done differently
    if len(robots) > 1:
        for i in range(len(all_poses)-1):
            if config_dist(qs.from_flat(all_poses[i]), qs.from_flat(all_poses[i+1])) > resolution * 2:
                return False

    return True

"""
NOTES (Liam):
Remove later also comments within code, keep for now for better understanding
=============================================================================================
PLANNING FUNCTIONS

plan_in_time_space:
- Unidirectional RRT in joint config-time space. Grows single tree from start, samples random (q,t)
  nodes, steers under velocity constraints, checks collisions against static env + dyn obstacles 
  (other_paths), and returns TimedPath in joint space.

plan_in_time_space_bidirectional:
- Bidirectional RRT in joint config-time space. Grows forward tree from start and reverse tree from
  sampled goals, alternating expansion and attempting to connect them. Same velocity/collision checks.
  Generally faster than unidirectional. Returns TimedPath in joint space. 

shortcut_with_dynamic_obstacles:
- Post-processing optimizer. Discretizes RRT path, then repeatedly picks two random waypoints and 
  tries to replace the segment between them with straight-line interpolation. Accepts only if 
  collision-free against static env + dyn obstacles. Smooths and shortens the path.

plan_robots_in_dyn_env:
- Wrapper functions managing the planning for one task, choosing appropriate planner, optionally
  post-processes with shortcutting, splits the planned joint path into per-robot paths, and returns
  per-robot separated paths and final joint configuration  
=============================================================================================
"""
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
    """
    Unidirectional RRT planner operating in joint space-time configuration
    Plans for a specific subset of robots (active) while treating other robots as dynamic obstacles
    """
    # Setup and initialization
    computation_start_time = time.time()
    max_iter = 50000
    conf_type = type(env.get_start_pos())

    logger.info("start_time", t0)
    logger.info("robots", robots)
    logger.info("earliest end time", t_lb)

    start_configuration = prev_plans.get_robot_poses_at_time(robots, t0)
    q0 = conf_type.from_list(start_configuration)

    logger.info("start state", q0.state())

    # Tree initialization
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
        q_diff = rnd_node.q.state() - close_node.q.state() # Vector form (direction + magnitude)
        length = config_dist(rnd_node.q, close_node.q) # Scalar distance

        v = length / t_diff # Required velocity

        logger.debug(f"length {length}")

        if v > v_max:
            return None, None # Physically infeasible (enforces kinodynamic constraints)

        # Either, can reach sample directly
        if t_diff < max_stepsize:
            return rnd_node.t, rnd_node.q 

        # Otherwise, take a step of size max_stepsize toward sample
        t_m = min(max_stepsize * 1.0, t_diff)

        if length < 1e-3:
            return None, None

        t = close_node.t + t_m
        q = close_node.q.state() + t_m * v * q_diff / length # t_m*v (displacement magnitude), q_diff/length (unit direction vector)

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
        """Informed sampling: narrow the sampling region to a box around start-goal corridor"""
        informed_sampling = True
        if informed_sampling:
            # Compute midpoint between q0 and qg
            q0_state = q0.state()
            qg_state = sampled_goals[0][1].state()
            mid = (q0_state + qg_state) / 2

            # Compute box half-width
            max_goal_time = max([g[0] for g in sampled_goals])
            c_max = (max_goal_time - t0) * v_max # Max distance robot can travel in available time

            # Sample in box: [mid-c_max/2, mid+c_max/2]
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

                    # For each DOF sample uniformly within box
                    s.append(np.random.uniform(lo, hi))

                    i += 1

            # Compute feasible time window for this sample
            conf = q0.from_flat(np.array(s))
            min_dt_from_start = config_dist(q0, conf) / v_max
            min_t_sample = t0 + min_dt_from_start # Earliest robot can reach this config
            max_t_sample = ( # Latest time s.t. robot can still reach the goal
                max_goal_time - config_dist(conf, sampled_goals[0][1]) / v_max
            )

            if min_t_sample > max_t_sample:
                return max_goal_time + 1, q0.from_flat(np.random.rand(len(q0.state())))

            # Sample time uniformly within bounds
            t_rnd = np.random.uniform(min_t_sample, max_t_sample)
        else:
            max_goal_time = max([g[0] for g in sampled_goals])

            t_rnd = np.random.rand() * (max_goal_time - t0) + t0

            if t_rnd < t0:
                raise ValueError

            q_rnd = []
    
            for r in robots:
                idx = env.robot_idx[r]
                lims = env.limits[:, idx]
                dim = env.robot_dims[r]

                rnd_uni_0_1 = np.random.rand(dim)
                q = rnd_uni_0_1 * (lims[1, :] - lims[0, :]) + lims[0, :]

                q_rnd.append(q * 1.0)

            conf = conf_type.from_list(q_rnd)

        return t_rnd, conf

    def project_sample_to_preplanned_path(t, q):
        """Enforce that robots follow their previous plan when they're still busy"""
        logger.debug("projecting")
        q_new = q

        for i, r in enumerate(robots):
            if end_times[r] >= t:
                pose = prev_plans.get_robot_poses_at_time([r], t)[0]
                q_new[i] = pose * 1.0

        return q_new

    # Time bounds computation
    latest_end_time = max([end_times[r] for r in robots])
    t_lb = max(latest_end_time + 1, t_lb) # New plan can only start at t_lb (after all robots are free + buffer)

    logger.debug(f"end_time: {t_lb}")

    # Estimate start-goal distance
    start_poses = prev_plans.get_robot_poses_at_time(robots, t_lb)
    goal_pose = goal.sample(None)
    goal_config = []
    offset = 0
    for r in robots:
        dim = env.robot_dims[r]
        goal_config.append(goal_pose[offset : offset + dim])
        offset += dim

    # Configuration space distance between start and goal
    d = config_dist(conf_type.from_list(goal_config), conf_type.from_list(start_poses))

    logger.debug("Goal pose", conf_type.from_list(goal_config).state())
    logger.debug("start/goal dist", d)

    # Compute time bounds (with calculated config space distance and max velocity)
    max_t = t_lb + 1 + (d / v_max) * 10 # Conservative upper bound (allows for detours around obstacles)

    escape_path_end_time = prev_plans.get_final_time()
    # Needs to span at least as long as any existing plan to ensure correct collision checking
    max_t = max(max_t, escape_path_end_time) # Consider other robot's escape paths

    logger.debug("start_times", end_times)
    logger.debug("max time", max_t)

    curr_t_ub = max([end_times[r] for r in robots]) + (d / v_max) * 3 # Start with more reasonable initial bound
    curr_t_ub = max(curr_t_ub, t_lb)

    logger.debug("times for lb ", t0, d / v_max)

    configurations = None

    sampled_times = []
    sampled_pts = []

    other_robots = [] # All robots not being planned -> dynamic obstacles
    for r in env.robots:
        if r not in robots:
            other_robots.append(r)

    # MAIN PLANNING LOOP 
    iter = 0
    while True:
        iter += 1

        # Termination checks
        if ptc.should_terminate(0, time.time() - computation_start_time):
            break

        if iter > max_iter:
            print("Max iter reached, stopping")
            break

        # Gradually expand time horizon (upper bound that we are sampling)
        if iter % 50:
            curr_t_ub += 1
            curr_t_ub = min(curr_t_ub, max_t)

        # Step 1: sampling phase
        # sample time and position
        rnd = random.random()
        goal_sampling_probability = 0.1 # Balance between exploration/exploitation

        # First always samples goal because need at least one goal to define informed sampling region  
        if len(sampled_goals) == 0 or rnd < goal_sampling_probability:
            t_rnd, q_sampled = sample_goal(curr_t_ub)
            sampled_goals.append((t_rnd, q_sampled))
            print(f"Adding goal at {t_rnd}")
        else:
            t_rnd, q_sampled = sample_uniform(curr_t_ub)

        # Projection after sampling
        # Why? sample_uniform doesn't know about multi-robot constraints (e.g., prev_plans), if robot 
        # still executing -> override sample with exact pose at time t along prev_path (through interpolation) 
        # Simpler to sample and project, than baking constraint into sampling itself 
        q_rnd = project_sample_to_preplanned_path(t_rnd, q_sampled)

        logger.debug("rnd state,", q_rnd.state())

        # Step 2: feasibility checks
        # Checks are ordered by computational cost (cheap to expensive)
        # Check 1: Reachable from start?
        time_from_start = t_rnd - t0
        d_from_start = config_dist(q0, q_rnd, "max")
        if d_from_start / time_from_start > v_max:
            continue
        
        # Check 2: Can reach some goal?
        reachable_goal = False
        for tg, qg in sampled_goals:
            time_from_goal = tg - t_rnd
            # We do not need to plan to a time which does not yet have a goal that we can reach.
            if tg < t_rnd:
                continue # Skip goals from past (can't go backward in time)

            if np.linalg.norm(qg.state() - q_rnd.state()) < 1e-3:
                reachable_goal = True # Already at goal
                break
            
            # Use max norm -> conservative, uses worst-case velocity
            d_from_goal = config_dist(qg, q_rnd, "max")
            if d_from_goal / time_from_goal <= v_max:
                reachable_goal = True
                break

        if not reachable_goal:
            logger.debug("No reachable goal for the sampled node.")
            logger.debug("times", tg, t_rnd)
            continue

        # Check 3: Collision-free (valid) sample?
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

        # Step 3: tree extension
        # Find nearest neighbour (closest pt in existing tree)
        n_close = tree.get_nearest_neighbor(Node(t_rnd, q_rnd), v_max)

        if n_close is None:
            continue # No feasible connection exists

        added_pt = False
        q_new = None
        t_new = None
        extend = False

        # Takes multiple small steps toward the sample
        # Greedily extends the tree as far as possible toward the sample in one iteration
        if extend:
            t_goal = t_rnd
            q_goal = q_rnd

            n_prev = n_close
            steps = 0
            while True:
                steps += 1

                t_next, q_next = steer(n_prev, Node(t_goal, q_goal), 5) # Steer small step (5) from n_prev to sample
                if t_next is None or q_next is None:
                    break # Stop if steer fails
                
                # Second projection on steered sample
                q_next = project_sample_to_preplanned_path(t_next, q_next) # Busy robots follow their plans

                # Collision check (point and edge)
                # Why check point if after we either way check edge?
                # If endpoint collides, no reason to then check whole edge (makes sense here  with many small steps)
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
                    # Add node to tree
                    tree.add_node(Node(t_next, q_next), n_prev)

                    n_prev = tree.nodes[-1]

                    added_pt = True
                    t_new = t_next
                    q_new = q_next

                    if np.linalg.norm(q_goal.state() - q_new.state()) < 1e-3:
                        break # Stop when reaching sample 
                else:
                    break # Stop when hitting a collision
        # Take one big step toward the sample
        else:
            t_new, q_new = steer(n_close, Node(t_rnd, q_rnd), max_stepsize=30) # Steer one big step

            # Same 
            if t_new is None or q_new is None:
                continue
            
            # Second projection (on steered sample)
            q_new = project_sample_to_preplanned_path(t_new, q_new) 

            # Collision check (edge only)
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
                # Add node to tree
                tree.add_node(Node(t_new, q_new), n_close)

                sampled_times.append(t_new)
                sampled_pts.append(q_new.state())

                added_pt = True
                logger.debug(f"succ at time {t_new}")
            else:
                logger.debug(f"failed at time {t_new}")
                logger.debug(f"Tree size {len(tree.nodes)}")

        # Step 4: goal check 
        # Condition 1: successfully added to tree
        # Condition 2: in goal region
        # Condition 3: finished after earliest allowed time
        if (added_pt and 
            goal.satisfies_constraints(q_new.state(), mode=None, tolerance=1e-5) and 
            t_new > t_lb
        ):
            # Extract path by backtracking through parents (gives goal -> start)
            configurations = [q_new.state()]
            times = [t_new]
            p = n_close

            while p.parent is not None:
                configurations.append(p.q.state())
                times.append(p.t)
                p = p.parent

            configurations.append(p.q.state()) # Add root
            times.append(p.t)

            computation_end_time = time.time()
            logger.debug(f"Took {computation_end_time - computation_start_time}s")

            # Reverse to get (start -> goal)
            return TimedPath(time=times[::-1], path=configurations[::-1])

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
    max_stepsize = 5,
) -> TimedPath:
    """
    Bidirectional RRT planner operating in joint space-time configuration
    Plans for a specific subset of robots (active) while treating other robots as dynamic obstacles
    """
    computation_start_time = time.time()
    conf_type = type(env.get_start_pos())

    start_configuration = prev_plans.get_robot_poses_at_time(robots, t0)
    q0 = conf_type.from_list(start_configuration)

    logger.info(f"start state {q0.state()}")

    # Tree initialization (forward and reverse)
    t_fwd = Tree(Node(t0, q0))
    t_rev = Tree(None, reverse=True) # Empty because don't know goal configuration yet

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
        length = config_dist(rnd_node.q, close_node.q, "max")

        v = length / t_diff # Required velocity

        if v > v_max:
            return None, None # Physically infeasible (enforces kinodynamic constraints)
            assert False

        # Either, can reach sample directly
        if t_diff < max_stepsize:
            return rnd_node.t, rnd_node.q

        # Otherwise, take a step of size max_stepsize toward sample
        t_m = min(max_stepsize * 1.0, t_diff)

        if length < 1e-3:
            return None, None

        t = close_node.t + t_m
        q = close_node.q.state() + t_m * v * q_diff / length # t_m*v -> displacement magnitude, q_diff/length -> unit direction vector

        q_list = []
        offset = 0
        for r in robots:
            dim = env.robot_dims[r]
            q_list.append(q[offset : dim + offset])
            offset += dim

        return t, conf_type.from_list(q_list)

    # We still steer from close to rnd.
    # But we now assume that close node has a higher time.
    def reverse_steer(close_node: Node, rnd_node: Node, max_stepsize=30):
        if close_node.t < rnd_node.t: # DIFFERENCE IS HERE!
            logger.warn("close", close_node.t)
            logger.warn("goal", rnd_node.t)
            assert False

        t_diff = close_node.t - rnd_node.t
        q_diff = rnd_node.q.state() - close_node.q.state()
        length = config_dist(rnd_node.q, close_node.q, "max")

        v = length / t_diff # Required velocity

        if v > v_max:
            return None, None # Physically infeasible (enforces kinodynamic constraints)

        # Either, can reach sample directly
        if t_diff < max_stepsize:
            return rnd_node.t, rnd_node.q

        # Otherwise, take a step of size max_stepsize toward sample
        t_m = min(max_stepsize * 1.0, t_diff)

        if length < 1e-3:
            return None, None

        t = close_node.t - t_m # DIFFERENCE IS HERE!
        q = close_node.q.state() + t_m * v * q_diff / length # t_m*v -> displacement magnitude, q_diff/length -> unit direction vector

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
        """Informed sampling: narrow the sampling region to a box around start-goal corridor"""
        informed_sampling = True
        if informed_sampling:
            # Compute midpoint between q0 and qg
            q0_state = q0.state()
            qg_state = sampled_goals[0][1].state()
            mid = (q0_state + qg_state) / 2

            # Compute box half-width
            max_goal_time = max([g[0] for g in sampled_goals])
            c_max = (max_goal_time - t0) * v_max

            # Sample in box: [mid-c_max/2, mid+c_max/2]
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

                    # For each DOF sample uniformly within box
                    s.append(np.random.uniform(lo, hi))

                    i += 1

            # Compute feasible time window for this sample
            conf = q0.from_flat(np.array(s))
            min_dt_from_start = config_dist(q0, conf) / v_max
            min_t_sample = t0 + min_dt_from_start # Earliest robot can reach this config
            max_t_sample = ( # Latest time s.t. robot can still reach the goal
                max_goal_time - config_dist(conf, sampled_goals[0][1]) / v_max
            )

            if min_t_sample > max_t_sample:
                return max_goal_time + 1, q0.from_flat(np.random.rand(len(q0.state())))
            # Sample time uniformly within bounds
            t_rnd = np.random.uniform(min_t_sample, max_t_sample)
        else:
            max_goal_time = max([g[0] for g in sampled_goals])

            t_rnd = np.random.rand() * (max_goal_time - t0) + t0

            if t_rnd < t0:
                raise ValueError

            q_rnd = []
   
            for r in robots:
                idx = env.robot_idx[r]
                lims = env.limits[:, idx]
                dim = env.robot_dims[r]

                rnd_uni_0_1 = np.random.rand(dim)
                q = rnd_uni_0_1 * (lims[1, :] - lims[0, :]) + lims[0, :]

                q_rnd.append(q * 1.0)

            conf = conf_type.from_list(q_rnd)

        return t_rnd, conf

    def project_sample_to_preplanned_path(t: float, q: Configuration) -> Configuration:
        """Enforce that robots follow their previous plan when they're still busy"""
        q_new = q

        for i, r in enumerate(robots):
            if end_times[r] >= t:
                pose = prev_plans.get_robot_poses_at_time([r], t)[0]
                q_new[i] = pose * 1.0

        return q_new

    # Time bounds computation
    latest_end_time = max([end_times[r] for r in robots])
    t_lb = max(latest_end_time + 1, t_lb) # New plan can only start at t_lb (after all robots are free + buffer)

    logger.info(f"end_time: {t_lb}")

    # Estimate start-goal distance
    start_poses = prev_plans.get_robot_poses_at_time(robots, t_lb)
    goal_pose = goal.sample(None)
    goal_config = []
    offset = 0
    for r in robots:
        dim = env.robot_dims[r]
        goal_config.append(goal_pose[offset : offset + dim])
        offset += dim
    # Configuration space distance between start and goal
    d = config_dist(conf_type.from_list(goal_config), conf_type.from_list(start_poses))

    logger.debug(f"Goal pose {conf_type.from_list(goal_config).state()}")
    logger.debug(f"start/goal dist {d}")

    # Compute time bounds (with calculated config space distance and max velocity)
    max_t = t_lb + 1 + (d / v_max) * 10 # Conservative upper bound (allows for detours around obstacles)

    escape_path_end_time = prev_plans.get_final_time()
    # Needs to span at least as long as any existing plan to ensure correct collision checking
    max_t = max(max_t, escape_path_end_time) # Consider other robot's escape paths

    logger.debug(f"start_times{end_times}")
    logger.debug(f"max time{max_t}")

    # curr_t_ub = t_lb + (d / v_max) * 3
    curr_t_ub = max([end_times[r] for r in robots]) + (d / v_max) * 3 # Start with more reasonable initial bound
    # curr_t_ub = max_t

    print("Upper bound time", max_t)

    curr_t_ub = max(curr_t_ub, t_lb)

    configurations = None

    other_robots = [] # All robots not being planned -> dynamic obstacles
    for r in env.robots:
        if r not in robots:
            other_robots.append(r)

    # Phase 1: goal sampling
    # The reverse tree needs at least one collision-free goal node to start growing backward from 
    iter = 0
    while True:
        iter += 1

        t_rnd, q_rnd = sample_goal(max_t)
        if iter == 1:
            t_rnd = max_t # First sample at latest time (conservative upper bound computed earlier)

        # Collision check of goal with env at time t_rnd
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
            t_rev.add_node(Node(t_rnd, q_rnd), None) # Add to reverse tree (root nodes of reverse tree, no parent)
            sampled_goals.append((t_rnd, q_rnd))

        if len(sampled_goals) > 0 and iter > 50:
            break # Got at least one valid goal

        if iter > 10000:
            print("Max iters in goal sampling.")
            # env.C.view(True)
            return None

    # Start pose validation (check before expensive tree growth)
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
        return None

    # Phase 2: MAIN PLANNING LOOP 
    max_iters = 50000
    for iter in range(max_iters):
        tmp = t_a
        t_a = t_b
        t_b = tmp # Swap active and passive trees at every iteration (balanced iterations)

        # Termination checks
        if ptc.should_terminate(0, time.time() - computation_start_time):
            print("Stopping due to time")
            break

        # Gradually expand time horizon (upper bound that we are sampling)
        if iter % 50:
            curr_t_ub += 1
            curr_t_ub = min(curr_t_ub, max_t)

        if iter % 500 == 0:
            logger.debug(f"iteration {iter}")
            logger.debug(f"num nodes in trees: {len(t_a.nodes)} {len(t_b.nodes)}")
            logger.debug(f"Current t_ub {curr_t_ub}")

         # Step 1: sampling phase
         # No goal sampling, already sampled in PHASE 1
        t_rnd, q_sampled = sample_uniform()
        q_rnd = project_sample_to_preplanned_path(t_rnd, q_sampled) # Only free robot get randomly sampled positions

        # Step 2: feasibility checks
        # Checks are ordered by computational cost (cheap to expensive)
        # Check 1: Reachable from start?
        time_from_start = t_rnd - t0
        d_from_start = config_dist(q0, q_rnd, "max")
        if d_from_start / time_from_start > v_max:
            continue

        # Check 2: Can reach some goal?
        reachable_goal = False
        for tg, qg in sampled_goals:
            time_from_goal = tg - t_rnd
            # We do not need to plan to a time which does not yet have a goal that we can reach.
            if tg < t_rnd:
                continue # Skip goals from past (can't go backward in time)

            if np.linalg.norm(qg.state() - q_rnd.state()) < 1e-3:
                reachable_goal = True # Already at goal
                break
            
            # Use max norm -> conservative, use worst-case velocity
            d_from_goal = config_dist(qg, q_rnd, "max")
            if d_from_goal / time_from_goal <= v_max:
                reachable_goal = True
                break

        if not reachable_goal:
            logger.warn("No reachable goal for the sampled node.")
            # print("times", tg, t_rnd)
            continue

        # Check 3: Collision-free (valid) sample?
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

        # Step 3: tree extension
        # Find nearest neighbour (closest pt in existing tree)
        n_close = t_a.get_nearest_neighbor(Node(t_rnd, q_rnd), v_max)

        assert n_close is not None # Feasible connections should always exists

        # Steering with direction awareness
        if t_a.reverse:
            # Time decreases (backward tree)
            t_new, q_new = reverse_steer(n_close, Node(t_rnd, q_rnd), max_stepsize=max_stepsize)
        else:
            # Time increases (forward tree)
            t_new, q_new = steer(n_close, Node(t_rnd, q_rnd), max_stepsize=max_stepsize)

        if t_new is None or q_new is None:
            continue
        
        # Project steered sample to enforce robot constraints at this new time 
        q_new = project_sample_to_preplanned_path(t_new, q_new)

        # Collision check 
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
            # Add node (sample) to tree A
            t_a.add_node(Node(t_new, q_new), n_close)

            # Check if opposite tree (B) has node near sample -> might connect?
            n_close_opposite = t_b.get_nearest_neighbor(Node(t_rnd, q_rnd), v_max)
            assert n_close_opposite is not None

            # TODO: should we steer here, instead of attempting to connect?
            # Try to link both trees in one shot might fail. Steer more conservative,
            # but more likely to succeed in cluttered spaces.

            # Check if edge connecting tree A (node q_new) and B (node n_close_opposite) is collision-free 
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

                # Extract path from first tree A (q_new -> parent -> ... -> root_of_t_a)
                configurations = [q_new.state()]
                times = [t_new]

                p = n_close
                while p.parent is not None:
                    configurations.append(p.q.state()) 
                    times.append(p.t)
                    p = p.parent

                configurations.append(p.q.state()) # Add root
                times.append(p.t)

                # Extract path from other tree B (n_close_opposite -> parent -> ... -> root_of_t_b)
                other_configurations = []
                other_times = []

                p = n_close_opposite
                while p.parent is not None:
                    other_configurations.append(p.q.state())
                    other_times.append(p.t)
                    p = p.parent

                other_configurations.append(p.q.state())
                other_times.append(p.t)

                configurations = configurations[::-1] # Reverse first path
                times = times[::-1]

                path = configurations + other_configurations # Concatenate
                times = times + other_times

                # If t_a was forward_tree -> path already correct order
                # If t_a was backward_tree -> path backward -> needs flip
                if t_a.reverse:
                    path = path[::-1]
                    times = times[::-1]

                # Handle multi-robot different end times
                # Sampled times may not include these robot-specific end times -> add them!
                if len(robots) > 1:
                    for r in robots:
                        t_robot_end = end_times[r]
                        if t_robot_end > t0:
                            # Find time interval that contains t_robot_end
                            for k in range(len(times)-1):
                                if times[k] <= t_robot_end <= times[k+1]:
                                    # Found interval, linearly interpolate
                                    alpha = (t_robot_end - times[k]) / (times[k+1] - times[k])
                                    interpolated_pose = path[k] + alpha * (path[k+1] - path[k])
                                    p = interpolated_pose
                                    break
                            
                            # Projects that interpolated configuration back onto robot's preplanned path
                            p_conf = q0.from_flat(p)
                            p_proj = project_sample_to_preplanned_path(t_robot_end, p_conf).state()
                            # Append new point and time to the path
                            path.append(p_proj)
                            times.append(t_robot_end)
                    
                    # Sort according to increasing times
                    sorted_indices = np.argsort(times)
                    path = [path[i] for i in sorted_indices]
                    times = [times[i] for i in sorted_indices]
                    
                # Return path
                return TimedPath(time=times, path=path)
        else:
            # print("not coll free")
            pass

    print(f"Did not find a path in {max_iters} iters.")
    return None

def shortcut_with_dynamic_obstacles(
    env: BaseProblem, 
    other_paths: MultiRobotPath, 
    robots, 
    path, 
    max_iter=500
):
    """
    
    """
    # Step 1: setup 
    logger.info("shortcutting")

    ql = []
    offset = 0
    for r in env.robots:
        dim = env.robot_dims[r]
        if r in robots: # Only robots involved in this task
            ql.append(env.get_start_pos().state()[offset : offset + dim])
        offset += dim

    conf_type = type(env.get_start_pos())
    tmp_conf = conf_type.from_list(ql)

    def arr_to_config(q):
        return tmp_conf.from_flat(q)

    # Step 2: discretization 
    discretized_path = []
    discretized_time = []

    # Discretize path: add many intermediate points to TimedPath
    resolution = 0.1
    for i in range(len(path.path) - 1): # For each segment
        q0 = arr_to_config(path.path[i])
        q1 = arr_to_config(path.path[i + 1])

        t0 = path.time[i]
        t1 = path.time[i + 1]

        dist = config_dist(q0, q1)
        N = int(dist / resolution) # Number of intermediate points
        N = max(1, N)

        for j in range(N): # Interpolate the N points (configurations and time)
            q = []
            for k in range(q0.num_agents()): # For each robot (independently because #DOFs may vary)
                qr = q0[k] + (q1[k] - q0[k]) / N * j # Linear interpolation
                q.append(qr)

            discretized_path.append(np.concatenate(q))

            t = t0 + (t1 - t0) * j / N # Linear time interpolation
            discretized_time.append(t)

    discretized_path.append(path.path[-1]) # Add final waypoint (loop only goes to N-1)
    discretized_time.append(path.time[-1])

    # Discretized new path (dense)
    new_path = TimedPath(time=discretized_time, path=discretized_path)

    # Step 3: setup for shortcutting
    num_indices = len(new_path.path) # Number of discretized waypoints
    end_times = other_paths.get_end_times(robots) # When each robot finished prev task

    indices = {}
    offset = 0
    # Build index map (which inidices in joint config belong to which robot?)
    for r in robots:
        dim = env.robot_dims[r]
        indices[r] = np.arange(offset, offset + dim) # 'rA': [0,1,2], 'rB': [3,4]
        offset += dim

    # Step 4: MAIN SHORTCUTTING LOOP
    attempted_shortcuts = 0
    max_attempts = max_iter * 10
    for _ in range(max_attempts):
        if attempted_shortcuts > max_iter:
            break
        
        # 1: Shortcut selection (sample random 2 waypoints)
        i = np.random.randint(0, num_indices) # Start of shortcut
        j = np.random.randint(0, num_indices) # End of shortcut

        if i > j:
            i, j = j, i

        if abs(j - i) < 2:
            continue # Skip if i,j too close, e.g., when i=10 and j=11
        
        # Select random robot to optimize (one at a time)
        robot_idx_to_shortcut = np.random.randint(0, len(robots))
        robot_name_to_shortcut = robots[robot_idx_to_shortcut]

        # 2: End time (time at which robot finished prev task) filter
        # Skip attempt of shortcutting if it tries shortcutting path before the end time of this robot
        # Example:
        #   rA end_time=5 rB end_time=10
        #   t0=min(5,10)=5
        #   Joint path waypoints: t=5   t=6  t=7   ...   t=10   ...   t=20
        #                         rA moves freely        both moving
        #                         rB holds position      toward task goal
        #   Can't shortcut rB before t=10 because previous task not done!
        if (
            new_path.time[i] < end_times[robot_name_to_shortcut]
            or new_path.time[j] < end_times[robot_name_to_shortcut]
        ):
            continue

        # 3: Cost improvement check
        assert new_path.time[i] < new_path.time[j]

        q0 = arr_to_config(new_path.path[i])
        q1 = arr_to_config(new_path.path[j])

        t0 = new_path.time[i]
        t1 = new_path.time[j]

        # Check if the shortcut improves cost
        if path_cost(
            [q0.state(), q1.state()], env.batch_config_cost, agent_slices=q0._array_slice,
        ) >= path_cost(
            new_path.path[i:j], env.batch_config_cost, agent_slices=q0._array_slice
        ):
            continue

        attempted_shortcuts += 1 # Count it as valid attempt

        # 4: Extract single-robot configurations (only check robot being shortcut)
        q0 = conf_type.from_list([q0[robot_idx_to_shortcut]])
        q1 = conf_type.from_list([q1[robot_idx_to_shortcut]])

        # 5: Multi-robot coordination setup
        # Create paths for OTHER robots (not being shortcut)
        if len(robots) > 1: # Multiple robots involved in this task
            tmp_other_paths = other_paths

            tmp_paths = {}
            for r in robots:
                if r != robot_name_to_shortcut: # OTHER robot
                    ind = indices[r] # Indices of robot r in joint config
                    tmp_paths[r] = TimedPath( # Extract robot r's configs from ALL waypoints
                        path=[pt[ind] * 1.0 for pt in new_path.path],
                        time=copy.deepcopy(discretized_time),
                    )

            # Add OTHER robot (its existing path from previous tasks)
            # Temporary (just for collision check) 
            tmp_other_paths.add_path(
                [r for r in robots if r != robot_name_to_shortcut],
                Path(path=tmp_paths, task_index=3),
                None,
                is_escape_path=True,
            )
        else: # Single robot case 
            tmp_other_paths = other_paths # No need to add anything

        # TODO: This is wrong for partial shortcuts atm.
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
            for k in range(j - i): # k = 0,1,2,.. (waypoints 1,2,3,..)
                ql = []
                for r_idx, r in enumerate(robots):
                    if r_idx == robot_idx_to_shortcut:
                        # Interpolate shortcut (keep structure intact -> same #waypoints)
                        q = q0[0] + (q1[0] - q0[0]) / (j - i) * k
                    else:
                        # Keep original path for OTHER
                        q = new_path.path[i + k][indices[r]] * 1.0
                    ql.append(q)

                new_path.path[i + k] = np.concatenate(ql)
                new_path.time[i + k] = new_path.time[i] + k / (j - i) * (
                    new_path.time[j] - new_path.time[i]
                )

        # Cleanup: remove OTHER robot's path (was just temp added for collision check)
        if len(robots) > 1:
            tmp_other_paths.remove_final_escape_path(
                [r for r in robots if r != robot_name_to_shortcut]
            )

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
    """
    
    """
    # Call i) bidirectional or ii) unidirectional time-space RRT planner 
    # Both consider the other_paths as dynamic obstacles
    if use_bidirectional_planner:
        path = plan_in_time_space_bidirectional(
            ptc, env, t0, other_paths, robots, end_times, goal, t_lb, max_stepsize = config.stepsize,
        )
    else:
        path = plan_in_time_space(
            ptc, env, t0, other_paths, robots, end_times, goal, t_lb
        )

    # Failure check (no collision free path, time budget, too many iterations)
    if path is None:
        print("Did not find a path in dyn env.")
        return None, None

    # Post-processing (shortcutting) of TimedPath (tuple with .time and .path)
    if config.shortcut_iters > 0:
        postprocessed_path, info = shortcut_with_dynamic_obstacles(
            env, other_paths, robots, path, max_iter=config.shortcut_iters
        )
        path = postprocessed_path

    # Split joint path into per-robot paths
    # Example: rA with DOF=3 and rB with DOF=2
    # - Without split: 1 TimedPath with 5-nunmber configs
    # - With split: 2 TimedPath with 3-number and 2-number configs
    # Why? MultiRobotPath.add_path() expects per-robot paths.   
    separate_paths = {}
    offset = 0
    logger.debug(f"end times {end_times}")
    for r in robots:
        dim = env.robot_dims[r]
        c_n = []
        per_robot_times = []
        for i in range(len(path.path)):
            if path.time[i] >= end_times[r]:
                per_robot_times.append(path.time[i]) # Extract per_robot time
                c_n.append(path.path[i][offset : offset + dim]) # Extract per_robot path

        offset += dim # Tracks where in joint config each robot's DOFs start

        separate_paths[r] = TimedPath(time=per_robot_times, path=c_n)

        # Example
        # separate_paths = {
        #   'rA': TimedPath(
        #       time=[t1,t2,...],
        #       path = [[x1,y1,theta1],[x2,y2,theta2],...]
        #   ),
        #   'rB': TimedPath(
        #       time=[t1,t2,...],
        #       path = [[x1,y1],[x2,y2],...]
        #   ),
        # } First 

    # Return value 1: sepatarte_paths 
    # - Used by main planner to build robot_path
    # - Task planning: added with actual task_index and next_mode
    # - Escape planning: added with task_index=-1 and next_mode=None

    # Return value 2: path.path[-1] 
    # - Final joint config used by main planner to compute mode transition
    
    return separate_paths, path.path[-1]

"""
NOTES (Liam):
Remove later also comments within code, keep for now for better understanding 
=============================================================================================
PRIORITIZED PLANNER

Plans for multiple robots by solving one task at a time in a fixed sequence order. 

OUTER LOOP:
- Iterates over task-sequence, keeping best solution found

INNER LOOP:
- Removes prior escape path from involved robots
- Plans task path, treating all other committed robot paths as dynamic obstacles
- Resolves the mode transition via a precomputed task_id_sequence
- Commits the path, then plans escape paths to send involved robots to safe poses to not
  block future tasks
=============================================================================================
"""
@dataclass
class PrioritizedPlannerConfig:
    # gamma: float = 0.7
    # distance_metric: str = "euclidean"
    use_bidirectional_planner: bool = True
    shortcut_iters: int = 100
    multirobot_shortcut_iters: int = 100

    stepsize:int=10

    max_start_to_earliest_end_diff: int = 50
    remove_escapes_based_on_diff: bool = True


class PrioritizedPlanner(BasePlanner):
    def __init__(
        self,
        env: BaseProblem,
        config: PrioritizedPlannerConfig = PrioritizedPlannerConfig(),
    ):
        self.env = env
        self.config = config

    def _execute_skill_task(
            self, 
            task, 
            env: BaseProblem,  
            start_pose, 
            t0,
            prev_plans: MultiRobotPath,
            end_times: Dict[str, float],
    ):
        """
        Rollout skill
        Return path in same format as plan_robots_in_dyn_env()
        """
        conf_type = type(env.get_start_pos()) 
        q_init = conf_type.from_list(start_pose).state() # Only involved robots

        # Collect all joints (q_full)
        all_joints = []
        for r in env.robots:
            all_joints.extend(env.robot_joints[r])

        skill_result = task.skill.rollout(q_init, task, all_joints, env, t0)
        traj, times = skill_result.trajectory, skill_result.times # Single flat arrays

        # Check each consecutive edge for collision 
        for k in range(len(times)-1):
            parts_s, parts_e = [], []
            offset = 0
            for r in task.robots:
                dim = env.robot_dims[r]
                parts_s.append(traj[k][offset : offset + dim])
                parts_e.append(traj[k+1][offset : offset + dim])
                offset += dim     
            qs_conf = conf_type.from_list(parts_s)
            qe_conf = conf_type.from_list(parts_e)
            
            if not edge_collision_free_with_moving_obs(
                env,
                qs_conf, # NpConfiguration objects 
                qe_conf, 
                times[k],
                times[k+1],
                prev_plans,
                task.robots,    
                end_times,
                env.collision_resolution,
            ):
                # env.C.view(True)

                return None, None # Failure makes outer loop try different sequence

        # Avoid shortcutting: taken care of in robot_mode_shortcut()    

        # Path splitting into per-robot sub-trajectories from concatenated rollout
        # Rest of planner expects dict[robot -> Timedpath], one traj per robot 
        path = {}
        offset = 0
        for r in task.robots:
            dim = env.robot_dims[r]
            sub_traj = [q[offset : offset + dim] for q in traj]
            path[r] = TimedPath(time=times.tolist(), path=sub_traj)
            offset += dim

        final_pose = traj[-1]
        return path, final_pose
    
    def plan(
        self,
        ptc: PlannerTerminationCondition,
        optimize: bool = True,
    ) -> Tuple[List[State] | None, Dict[str, Any]]:
        """
        Try multiple task orderings / sequences (OUTTER LOOP), and for each one, plan tasks
        one at a time, treating already-planned robots as dynamic obstacles (INNER LOOP).

        Stop OUTTER LOOP if time runs out or sequences are exhausted.

        Returns: 
        - Best solution found (sequence of states with configs + modes)
        - Info dictionary with costs, times, paths for each sequence tried?
        """
        # SETUP
        q0 = self.env.get_start_pos() # Starting config and mode
        m0 = self.env.get_start_mode()

        conf_type = type(self.env.get_start_pos())
        robots = self.env.robots

        # Requirement check: prio-planner requires safe HOME poses for escape paths
        if self.env.spec.home_pose != SafePoseType.HAS_SAFE_HOME_POSE:
            raise ValueError("No safe home pose")

        # Dependency check: assume that we get a sequence-env for now
        # Option 1: Fixed sequence (returns [0,1,2,..])
        # Option 2: Oracle provides sequence (tries to find best sequence)
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

        # OUTTER LOOP: Try different task sequences
        while True:
            if ptc.should_terminate(0, time.time() - computation_start_time):
                break # Time budget exhausted
            
            # Step 1: create a fresh plan 
            robot_paths = MultiRobotPath(q0, m0, robots)  # Stores individual path segments for every robot
            seq_index = 0
            success = False
            env = copy.deepcopy(self.env)
            env.C_cache = {} # Collision cache

            # Step 2: get task ordering from oracle/env (plan for single sequence)
            sequence = env.get_sequence() 
            logger.info(f"Planning for sequence \n{sequence}")

            # Skip duplicates (if we planned for this sequence before)
            if sequence in sequence_cache:
                logger.debug("Planned for this sequence before, skipping.")
                skipped_sequences += 1

                if skipped_sequences > 10:
                    break # Oracle is repeating

                continue # Try next sequence

            skipped_sequences = 0
            sequence_cache.append(sequence)

            # Step 3: construct task ID sequence 
            task_id_sequence = [env.start_mode.task_ids]

            # Iterate through each task in the sequence
            for s in range(len(sequence) - 1):
                ids = copy.deepcopy(task_id_sequence[-1]) # Copy last known task assignments
                task_idx = sequence[s]
                task = env.tasks[task_idx]

                for robot_idx, r in enumerate(env.robots):
                    if r not in task.robots:
                        continue # Skip robots not involved in this task
                    
                    # Find next task for this robot
                    for i in range(s + 1, len(sequence)):
                        next_task_idx = sequence[i]
                        next_task = env.tasks[next_task_idx]

                        # Update their ID
                        if r in next_task.robots:
                            ids[robot_idx] = next_task_idx
                            break # Found next task for this robot

                task_id_sequence.append(ids)

                # Example:
                # T0 for R0, T1 for R1, T2 for R0
                # sequence = [0,1,2] 
                # task_id_sequence = [(0,1), (2,1), (2,1)]
                # Each entry is the next task each robot will execute 

            # INNER LOOP: Plan tasks in sequence
            while True:
                if ptc.should_terminate(0, time.time() - computation_start_time):
                    print("Terminating due to max time")
                    break

                # Step 1: prepare current task to be planned from sequence
                task_index = sequence[seq_index]
                task = env.tasks[task_index]
                involved_robots = task.robots

                logger.info(f"task name {task.name}")
                logger.info(f"task_index {task_index}")
                logger.info(f"robots: {involved_robots}")
                logger.info(f"sequence index {seq_index}")

                # Figure out when this task can end at the earliest
                earliest_end_time = robot_paths.get_final_non_escape_time()
                logger.debug(f"earliest end time {earliest_end_time}")

                # remove escape path from plan

                if self.config.remove_escapes_based_on_diff:
                    no_escape_path_end_times = robot_paths.get_non_escape_end_times(involved_robots)
                    with_escape_end_times = robot_paths.get_end_times(involved_robots)
                    
                    remove_escape_paths = copy.copy(involved_robots)

                    for r, t in no_escape_path_end_times.items():
                        earliest_robot_start_time = max(0, earliest_end_time - self.config.max_start_to_earliest_end_diff)
                        if t < earliest_robot_start_time and with_escape_end_times[r] < earliest_end_time:
                            # keep escape path if the start time without escape path
                            # is too far back.

                            # TODO: double check this
                            remove_escape_paths.remove(r)

                            # if len(robot_paths.paths[r]) > 0:
                            #     robot_paths.paths[r][-1].task_index = 0

                    robot_paths.remove_final_escape_path(remove_escape_paths)
                
                    # overwrite the end times with the earliest end times we like for the robots
                    # that still have an escape path.
                    end_times = robot_paths.get_end_times(involved_robots)
                    for r, t in end_times.items():
                        if r not in remove_escape_paths or len(robot_paths.paths[r]) == 0:
                            end_times[r] = max(0, earliest_end_time - self.config.max_start_to_earliest_end_diff)

                    t0 = min([v for k, v in end_times.items()])
                else:
                    robot_paths.remove_final_escape_path(involved_robots)

                    end_times = robot_paths.get_end_times(involved_robots)
                    for r, t in end_times.items():
                        if len(robot_paths.paths[r]) == 0:
                            end_times[r] = max(0, earliest_end_time - self.config.max_start_to_earliest_end_diff)
                    
                    t0 = min([v for k, v in end_times.items()])

                logger.info(f"Robot end times: {end_times}")
                logger.info(f"Robot start time: {t0}")

                # Get exact time (t0) and config (start_pose) from which new plan for involved robots should start
                logger.debug("Collecting start times")
                start_time = min([t for _, t in end_times.items()])
                start_pose = robot_paths.get_robot_poses_at_time(
                    involved_robots, start_time
                )

                # Goal specification for this task
                task_goal = task.goal

                # Step 2: motion planning OR skill execution (rollout)
                if task.skill is not None:
                    # EXECUTE SKILL (instead of planning)
                    logger.info(">> This is a SKILL task")
                    path, final_pose = self._execute_skill_task(
                        task, env, start_pose, t0,
                        prev_plans=robot_paths,
                        end_times=end_times,
                    )
                    if path is None:
                        logger.warn('Skill trajectory in collision, trying next sequence.')
                        # env.C.view(True)
                        break
                        
                else:
                    # Planning with time aware RRT
                    logger.info(">> This is a PLANNING task")
                    current_time = time.time()
                    planning_ptc = RuntimeTerminationCondition( # Create time-limited termination condition
                        ptc.max_runtime_in_s - (current_time - computation_start_time)
                    )

                    # PLAN THE TASK
                    path, final_pose = plan_robots_in_dyn_env(
                        self.config,
                        planning_ptc,
                        env,
                        t0,
                        robot_paths, # Previous paths (moving obstacles)
                        involved_robots,
                        start_pose,
                        end_times,
                        task_goal,
                        earliest_end_time, # t_lb: lower bound for planning (task can't finish before)
                        use_bidirectional_planner=self.config.use_bidirectional_planner,
                    )

                logger.debug(f"final_pose {final_pose}")

                if path is None:
                    logger.warn("Failed planning.")
                    break # Sequence failed, try next sequence

                # Step 3: determine mode transition (next mode for task)
                prev_mode = robot_paths.get_mode_at_time(
                    robot_paths.get_final_non_escape_time()
                )
                curr_mode = prev_mode

                # Not reached terminal mode yet
                if not env.is_terminal_mode(curr_mode):
                    final_time = path[involved_robots[0]].time[-1]
                    logger.debug(f"start_time {path[involved_robots[0]].time[0]}")
                    logger.debug(f"final_time: {final_time}")
                    
                    q = []
                    for r in env.robots:
                        if r in involved_robots:
                            q.append(path[r].path[-1]) # Final pose of just-planned robot path
                        else:
                            q.append( # Where the other robots are
                                robot_paths.get_robot_poses_at_time([r], final_time)[0]
                            )

                    logger.debug(f"curr mode {curr_mode}")
                    logger.debug(f"final pose in path {final_pose}")

                    # Get next modes that can be reached from current mode and config
                    next_modes = env.get_next_modes(
                        env.start_pos.from_list(q), curr_mode
                    )
                    # If ambiguity -> use precomputed task_id_sequence
                    if len(next_modes) > 1:
                        for next_mode in next_modes:
                            if next_mode.task_ids == task_id_sequence[seq_index + 1]:
                                break # The mode matching the planned sequence
                    else:
                        assert len(next_modes) == 1
                        next_mode = next_modes[0]
                else:
                    next_mode = curr_mode

                # Commit path to robot_paths
                robot_paths.add_path(
                    involved_robots,
                    Path(path=path, task_index=task_index),
                    next_mode,
                )

                # Step 4: check if done
                if seq_index + 1 >= len(sequence): # Check if this was the last task in the given sequence
                    success = True
                    logger.info("Found a solution.")
                    break

                # Step 5: plan escape path
                logger.info("planning escape path")
                escape_start_time = path[involved_robots[0]].time[-1] # End of main task path
                end_times = robot_paths.get_end_times(involved_robots)
                failed_escape_planning = False

                for r in involved_robots:
                    q_non_blocking = env.safe_pose[r] # Goal for escape path
                    escape_goal = SingleGoal(q_non_blocking)

                    escape_start_pose = robot_paths.get_robot_poses_at_time(
                        [r], escape_start_time
                    )

                    t0 = min([v for k, v in end_times.items()])

                    current_time = time.time()
                    escape_planning_ptc = RuntimeTerminationCondition(
                        ptc.max_runtime_in_s - (current_time - computation_start_time)
                    )

                    print("start time", t0)
                    escape_path, _ = plan_robots_in_dyn_env(
                        self.config,
                        escape_planning_ptc,
                        env,
                        t0,
                        robot_paths,
                        [r], # Simpler single-robot planning problem
                        escape_start_pose,
                        end_times,
                        escape_goal,
                        use_bidirectional_planner=self.config.use_bidirectional_planner,
                    )

                    if escape_path is None:
                        logger.warn("Failed escape path planning.")
                        failed_escape_planning = True
                        break

                    # Commit escape path to involved robots
                    # No mode change, just temporary repositioning move 
                    robot_paths.add_path(
                        [r],
                        Path(path=escape_path, task_index=-1),
                        next_mode=None,
                        is_escape_path=True,
                    )

                if failed_escape_planning:
                    break # Consider entire task sequence as failure

                seq_index += 1

            # Step 6: finalizing a successful sequence
            # Re-organize the data such that it is in the same format as before
            # - Internal use: prioritized planner works with robot_paths (per-robot, separated paths) 
            # - External use: unified path as list of State(config, mode) objects
            if success:
                path = []

                T = robot_paths.get_final_time() # Total makespan of the plan
                N = 5 * int(np.ceil(T)) # Sample plan at N evenly spaced time points (uniform)

                # At each sample time, query configs and current modes from robot_paths
                for i in range(N):
                    t = i * T / (N - 1) # dt constant
                    q = robot_paths.get_robot_poses_at_time(env.robots, t)
                    config = conf_type.from_list(q)
                    mode = robot_paths.get_mode_at_time(t)

                    state = State(config, mode) # Pack information into State
                    path.append(state)

                end_time = time.time()
                cost = path_cost(path, env.batch_config_cost)

                # Check and save if this path is the best
                if len(info["costs"]) == 0 or info["costs"][-1] > cost:
                    info["costs"].append(cost)
                    info["times"].append(end_time - computation_start_time)
                    info["paths"].append(path)

                    best_path = copy.deepcopy(path)

                    logger.info(f"Added path with cost {cost}.")

                # Step 7: global shortcutting / post-optimization (on ENTIRE assembled path)
                if self.config.multirobot_shortcut_iters > 0:
                    path_w_doubled_modes = []

                    # Shortcutter needs to know exactly when a mode change happens
                    # Example: State(qA,m1) -> State(qB,m2) ambiguous as we don't know where (which config) exactly change happens 
                    for i in range(len(path)):
                        path_w_doubled_modes.append(path[i])

                        # At every mode transition, insert extra state State(old_config, new_mode)
                        # Example: State(qA,m1) -> State(qA,m2) -> State(qB,m2)
                        if i + 1 < len(path) and path[i].mode != path[i + 1].mode:
                            path_w_doubled_modes.append(
                                State(path[i].q, path[i + 1].mode)
                            )
                        
                        # Explanation why shortcutting over modes is not allowed for active robots:
                        # If we would shortcut, would create straight-line interpolation that skips exect config where, e.g., grasping happens.
                        # Robot would hold object at configs where it hasn't actually grasped yet (or vice-versa). 

                    path = path_w_doubled_modes

                    # Iteratively tries shorcutting segments 
                    # Respecting mode constraints (no mode shortcutting over modes, except for inactive (other) robots) 
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
