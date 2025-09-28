import numpy as np

import time

from typing import List, Optional
from numpy.typing import NDArray

from numba import jit, float64, int64  # ty: ignore[possibly-unbound-import]

from .planning_env import (
    generate_binary_search_indices,
)

from .configuration import (
    Configuration,
    NpConfiguration,
    config_dist,
    config_cost,
    batch_config_cost,
)

from .planning_env import (
    BaseProblem,
    BaseModeLogic,
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
    SingleGoal,
)

from .registry import register

import mujoco
from mujoco import mjx
import jax

import mujoco.viewer

import threading
from concurrent.futures import ThreadPoolExecutor

import copy
import os.path


@jit(
    (float64[:], float64[:], float64[:]),
    nopython=True,
    fastmath=True,
    boundscheck=False,
)
def mul_quat(out, q1, q2):
    """Quaternion multiplication q1 ⊗ q2 -> out"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    out[0] = w1*w2 - x1*x2 - y1*y2 - z1*z2
    out[1] = w1*x2 + x1*w2 + y1*z2 - z1*y2
    out[2] = w1*y2 - x1*z2 + y1*w2 + z1*x2
    out[3] = w1*z2 + x1*y2 - y1*x2 + z1*w2

@jit((float64[:], float64[:], float64[:]), nopython=True, fastmath=True, boundscheck=False)
def rot_vec_quat(out, vec, quat):
    """Rotate vector by quaternion using efficient formula: out = quat * vec * quat_conj
    
    This uses the efficient formula: v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    where q = [w, x, y, z] and v is the 3D vector
    """
    w, x, y, z = quat
    vx, vy, vz = vec
    
    # First cross product: cross(q.xyz, v) + q.w * v
    cx = y * vz - z * vy + w * vx
    cy = z * vx - x * vz + w * vy  
    cz = x * vy - y * vx + w * vz
    
    # Second cross product: cross(q.xyz, first_cross)
    out[0] = vx + 2.0 * (y * cz - z * cy)
    out[1] = vy + 2.0 * (z * cx - x * cz)
    out[2] = vz + 2.0 * (x * cy - y * cx)


class MujocoEnvironment(BaseProblem):
    """
    Simple environment, only supporting rectangle and sphere obstacles, and spherical agents.
    """

    def get_body_ids(self, root_name):
        # Build parent->children mapping
        parent2children = {}
        for i in range(self.model.nbody):
            pid = int(self.model.body(i).parentid)
            parent2children.setdefault(pid, []).append(i)

        # Recursively collect all body IDs in subtree
        def subtree_body_ids(body_id):
            ids = [body_id]
            for child in parent2children.get(body_id, []):
                ids.extend(subtree_body_ids(child))
            return ids

        root_id = self.model.body(root_name).id
        robot_body_ids = np.array(subtree_body_ids(root_id))
        return robot_body_ids

    def collect_joints(self, root_name):
        robot_body_ids = self.get_body_ids(root_name)

        joint_names = [
            self.model.joint(j).name
            for j in range(self.model.njnt)
            if self.model.jnt_bodyid[j] in robot_body_ids
        ]

        return joint_names

    def collect_joint_ids(self, root_name):
        robot_body_ids = self.get_body_ids(root_name)

        joint_ids = [
            j
            for j in range(self.model.njnt)
            if self.model.jnt_bodyid[j] in robot_body_ids
        ]

        return joint_ids

    def collect_adr(self, root_name):
        robot_body_ids = self.get_body_ids(root_name)

        joint_addr = [
            self.model.jnt_qposadr[j]
            for j in range(self.model.njnt)
            if self.model.jnt_bodyid[j] in robot_body_ids
        ]

        return joint_addr

    def __init__(self, xml_path, n_data_pool: int = 1):
        self.limits = None

        self.cost_metric = "euclidean"
        self.cost_reduction = "max"

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.manipulating_env = False

        self.viewer = None
        self._enter_pressed = False

        # Preallocated pool for parallel collision checking
        self._data_pool = [mujoco.MjData(self.model) for _ in range(n_data_pool)]

        self.robot_idx = {}
        self.robot_dims = {}
        self.robot_joints = {}
        self._mujoco_joint_adr = {}
        self._mujoco_q_adr = {}

        self.body_id_q_adr = {}

        offset = 0
        for r in self.robots:
            self.robot_joints[r] = self.collect_joints(r)
            self.robot_idx[r] = np.arange(offset, offset + len(self.robot_joints[r]))
            self.robot_dims[r] = len(self.robot_joints[r])

            self._mujoco_joint_adr[r] = self.collect_adr(r)
            self._mujoco_q_adr[r] = self.collect_joint_ids(r)

            offset += self.robot_dims[r]

        self._all_robot_idx = np.array(
            [self._mujoco_joint_adr[r] for r in self.robots]
        ).flatten()

        self._mujoco_joint_id_mapping = np.array(
            [self._mujoco_q_adr[r] for r in self.robots]
        ).flatten()

        self.limits = np.zeros((2, len(self._mujoco_joint_id_mapping)))

        for idx, i in enumerate(self._mujoco_joint_id_mapping):
            self.limits[0, idx] = self.model.jnt_range[i, 0]  # lower limit
            self.limits[1, idx] = self.model.jnt_range[i, 1]  # upper limit

        self.initial_sg = {}

        self.root_name = "floor"

        mujoco.mj_forward(self.model, self.data)

        for i in range(self.model.nbody):
            if i not in self.body_id_q_adr:
                self.body_id_q_adr[i] = self.model.jnt_qposadr[
                    self.model.body_jntadr[i]
                ]

            # obj_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            obj_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            # obj_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)

            pos = self.data.geom_xpos[i]  # runtime info
            # print(obj_name, pos)
            if obj_name is None:
                continue

            if obj_name[:3] == "obj" or obj_name[:3] == "box":
                parent_id = self.model.body_parentid[i]
                parent_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, parent_id
                )

                pos_w = self.data.xpos[i]
                quat_w = self.data.xquat[i]

                parent_pos_w = self.data.xpos[parent_id]
                parent_quat_w = self.data.xquat[parent_id]

                # relative translation
                dp = pos_w - parent_pos_w
                inv_parent = np.zeros(4)
                mujoco.mju_negQuat(inv_parent, parent_quat_w)

                rel_pos = np.zeros(3)
                mujoco.mju_rotVecQuat(rel_pos, dp, inv_parent)

                # relative quaternion: q_rel = q_parent^-1 * q_child
                rel_quat = np.zeros(4)
                mujoco.mju_mulQuat(rel_quat, inv_parent, quat_w)

                # print(rel_pos)
                # print(rel_quat)

                self.initial_sg[i] = (
                    parent_name,
                    parent_id,
                    np.round(rel_pos, 3).tobytes(),
                    np.round(rel_quat, 3).tobytes(),
                )
    
        # self.qpos_body_views = {}
        self.qpos_body_views_pos = {}
        self.qpos_body_views_quat = {}
    
        for body_id in self.body_id_q_adr.keys():
            jadr = self.body_id_q_adr[body_id]
            # self.qpos_body_views[body_id] = self.data.qpos[jadr:jadr+7]
            self.qpos_body_views_pos[body_id] = self.data.qpos[jadr:jadr+3]
            self.qpos_body_views_quat[body_id] = self.data.qpos[jadr+3:jadr+7]

        #     parent = collision_model.geometryObjects[id_1].parentJoint
        #     placement = collision_model.geometryObjects[id_1].placement
        #     # print(obj_name)
        #     # print(placement)
        #     self.initial_sg[id_1] = (
        #         self.root_name,
        #         parent,
        #         np.round(placement, 3).tobytes(),
        #         pin.SE3(placement),
        #     )

        self.current_scenegraph = self.initial_sg.copy()

        self.spec = ProblemSpec(
            agent_type=AgentType.MULTI_AGENT,
            constraints=ConstraintType.UNCONSTRAINED,
            manipulation=ManipulationType.MANIPULATION,
            dependency=DependencyType.FULLY_ORDERED,
            dynamics=DynamicsType.GEOMETRIC,
            goals=GoalType.MULTI_GOAL,
            home_pose=SafePoseType.HAS_NO_SAFE_HOME_POSE,
        )

        self.child_xquat_buf = np.empty(4)
        self.rotated_buf = np.empty(3)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

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
            self.show_config(path[i].q, stop)
            self._set_to_scenegraph(path[i].mode.sg)

            # if export:
            #     self.C.view_savePng("./z.vid/")

            dt = pause_time
            if adapt_to_max_distance:
                if i < len(path) - 1:
                    v = 5
                    diff = config_dist(path[i].q, path[i + 1].q, "max_euclidean")
                    dt = diff / v

            time.sleep(dt)

        if stop_at_end:
            self.show_config(path[-1].q, True)

    def sample_config_uniform_in_limits(self):
        rnd = np.random.uniform(low=self.limits[0, :], high=self.limits[1, :])
        q = self.start_pos.from_flat(rnd)

        return q

    def _key_callback(self, key):
        # Enter key toggles pause
        if chr(key) == "ā":  # Enter key
            self._enter_pressed = True

    def show(self, blocking=True):
        """Open viewer at current state."""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self._key_callback
            )
        self.viewer.sync()

        if blocking:
            self._enter_pressed = False
            while self.viewer.is_running() and not self._enter_pressed:
                self.viewer.sync()
                time.sleep(0.01)

    def show_config(self, q, blocking=True):
        """Display a configuration `q` in the viewer."""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self._key_callback
            )

        self.data.qpos[self._all_robot_idx] = q.state()
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        self.viewer.sync()

        if blocking:
            self._enter_pressed = False
            while self.viewer.is_running() and not self._enter_pressed:
                self.viewer.sync()
                time.sleep(0.01)

    def config_cost(self, start: Configuration, end: Configuration) -> float:
        return config_cost(start, end, self.cost_metric, self.cost_reduction)

    def batch_config_cost(
        self,
        starts: List[Configuration],
        ends: List[Configuration],
        tmp_agent_slice=None,
    ) -> NDArray:
        return batch_config_cost(
            starts,
            ends,
            self.cost_metric,
            self.cost_reduction,
            tmp_agent_slice=tmp_agent_slice,
        )

    def is_collision_free(self, q: Optional[Configuration], mode: Optional[Mode]):
        # data = mujoco.MjData(self.model)
        # self.show(blocking=False)

        # if mode:
        #     self._set_to_scenegraph(mode.sg)

        # self.data.qpos[self._all_robot_idx] = q.state()
        # mujoco.mj_forward(self.model, self.data)

        # mujoco.mj_kinematics(self.model, self.data)

        if mode:
            self.data.qpos[self._all_robot_idx] = q.state()
            mujoco.mj_kinematics(self.model, self.data)

            self._set_to_scenegraph(mode.sg)
            mujoco.mj_forward(self.model, self.data)
            # mujoco.mj_collision(self.model, self.data)
        else:
            self.data.qpos[self._all_robot_idx] = q.state()
            mujoco.mj_forward(self.model, self.data)
            # mujoco.mj_collision(self.model, self.data)

        # If any contact distance < 0, collision
        for i in range(self.data.ncon):
            if self.data.contact[i].dist < self.collision_tolerance:
                return False

        return True

    def is_collision_free_for_robot(
        self,
        r: List[str] | str,
        q: NDArray,
        m: Optional[Mode] = None,
        collision_tolerance: Optional[float] = None,
        set_mode: bool = True,
    ) -> bool:
        raise NotImplementedError

    def is_edge_collision_free(
        self,
        q1: Configuration,
        q2: Configuration,
        mode: Mode,
        resolution: Optional[float] = None,
        tolerance: Optional[float] = None,
        include_endpoints: bool = False,
        N_start: int = 0,
        N_max: Optional[int] = None,
        N: Optional[int] = None,
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
            assert False

        if N_max is None:
            N_max = N

        N_max = min(N, N_max)

        # for a distance < resolution * 2, we do not do collision checking
        # if N == 0:
        #     return True

        idx = generate_binary_search_indices(N)

        q1_state = q1.state()
        q2_state = q2.state()
        dir = (q2_state - q1_state) / (N - 1)

        for i in idx[N_start:N_max]:
            if not include_endpoints and (i == 0 or i == N - 1):
                continue

            # print(i / (N-1))
            q = q1_state + dir * (i)
            q = q1.from_flat(q)

            if not self.is_collision_free(q, mode):
                return False

        return True

    def get_scenegraph_info_for_mode(self, mode: Mode, is_start_mode: bool = False):
        if not self.manipulating_env:
            return {}

        # self.set_to_mode(mode)
        prev_mode = mode.prev_mode
        if prev_mode is None:
            return self.initial_sg
        sg = prev_mode.sg.copy()

        active_task = self.get_active_task(prev_mode, mode.task_ids)

        # mode_switching_robots = self.get_goal_constrained_robots(mode)
        mode_switching_robots = active_task.robots

        # set robot to config
        prev_mode_index = prev_mode.task_ids[
            self.robots.index(mode_switching_robots[0])
        ]
        # robot = self.robots[mode_switching_robots]

        q_new = []
        for r in self.robots:
            if r in mode_switching_robots:
                q_new.append(mode.entry_configuration[self.robots.index(r)])
            else:
                q_new.append(np.zeros(self.robot_dims[r]))

        assert mode is not None
        assert mode.entry_configuration is not None

        q = np.concatenate(q_new)

        # TODO: set world to q
        self.data.qpos[self._all_robot_idx] = q
        # mujoco.mj_forward(self.model, self.data)
        mujoco.mj_kinematics(self.model, self.data)

        # print("BBBB")
        # self.show(blocking=True)

        self._set_to_scenegraph(sg)
        mujoco.mj_forward(self.model, self.data)

        # print("PPPPPP")
        # self.show(blocking=True)

        last_task = self.tasks[prev_mode_index]

        if last_task.type is not None:
            if last_task.type == "goto":
                pass
            else:
                pass
                # get id from frame name
                # obj_id = self.collision_model.getGeometryId(last_task.frames[1])
                # new_parent_id = self.collision_model.getGeometryId(last_task.frames[0])
                obj_bid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY.value, last_task.frames[1]
                )
                new_parent_bid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY.value, last_task.frames[0]
                )

                # print("obj", last_task.frames[1])
                # print("new parent", last_task.frames[0])

                obj_xpos = self.data.xpos[obj_bid].copy()
                obj_xquat = self.data.xquat[obj_bid].copy()
                parent_xpos = self.data.xpos[new_parent_bid].copy()
                parent_xquat = self.data.xquat[new_parent_bid].copy()

                # print("new parent pose", parent_xpos)
                # print("obj_pose", obj_xpos)

                # compute relative orientation
                inv_parent = np.array(
                    [
                        parent_xquat[0],
                        -parent_xquat[1],
                        -parent_xquat[2],
                        -parent_xquat[3],
                    ]
                )
                relative_quat = np.empty(4)
                mujoco.mju_mulQuat(relative_quat, inv_parent, obj_xquat)

                # compute relative position
                diff = obj_xpos - parent_xpos
                relative_pose = np.empty(3)
                mujoco.mju_rotVecQuat(relative_pose, diff, inv_parent)

                # print()
                # print("AAAAAAAAAAA")
                # print(relative_pose)
                # print(relative_quat)

                # update scenegraph
                sg[obj_bid] = (
                    last_task.frames[0],
                    new_parent_bid,
                    np.round(relative_pose, 3).tobytes(),
                    np.round(relative_quat, 3).tobytes(),
                )

        # print("NNNNN")
        # self.show(blocking=True)

        return sg

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def _set_to_scenegraph(self, sg):
        # child_xquat = np.empty(4)
        # rotated = np.empty(3)
        # pose_buf = np.empty(7)

        # qpos_cpy = np.array(self.data.qpos)

        for body_id, (
            parent_name,
            parent_bid,
            position_binary,
            rotation_binary,
        ) in sg.items():
            # if the body is already at the location it is supposed to be,
            # dont do anything
            if (
                parent_name == self.root_name
                and body_id in self.current_scenegraph
                and parent_name == self.current_scenegraph[body_id][0]
                and self.current_scenegraph[body_id][2] == position_binary
                and self.current_scenegraph[body_id][3] == rotation_binary
            ):
                continue

            position = np.array(np.frombuffer(position_binary))
            rotation = np.array(np.frombuffer(rotation_binary))

            # position = np.frombuffer(position_binary)
            # rotation = np.frombuffer(rotation_binary)

            parent_xpos = self.data.xpos[parent_bid]
            parent_xquat = self.data.xquat[parent_bid]

            # child world orientation: parent ⊗ child_rel
            # mujoco.mju_mulQuat(self.child_xquat_buf, parent_xquat, rotation)
            mul_quat(self.child_xquat_buf, parent_xquat, rotation)

            # child world position: parent + R_parent * pos_rel
            # mujoco.mju_rotVecQuat(self.rotated_buf, position, parent_xquat)
            rot_vec_quat(self.rotated_buf, position, parent_xquat)
            
            pos_view = self.qpos_body_views_pos[body_id]
            quat_view = self.qpos_body_views_quat[body_id]
            
            # Direct operations on exact-sized views
            # pos_view[:] = parent_xpos + self.rotated_buf
            np.add(parent_xpos, self.rotated_buf, out=pos_view)
            quat_view[:] = self.child_xquat_buf

            self.current_scenegraph[body_id] = sg[body_id]

        # self.data.qpos[:] = qpos_cpy

    def set_to_mode(
        self,
        mode: Mode,
        config=None,
        use_cached: bool = True,
        place_in_cache: bool = True,
    ):
        pass


class OptimizedMujocoEnvironment(MujocoEnvironment):
    """
    Optimized version with better parallel collision checking
    """

    def __init__(self, xml_path, n_data_pool: int = 4):
        super().__init__(xml_path, n_data_pool)

        # Create thread pool executor for reuse
        self._executor = ThreadPoolExecutor(
            max_workers=n_data_pool, thread_name_prefix="collision_checker"
        )
        self._pool_lock = threading.Lock()
        self._available_data = list(range(len(self._data_pool)))

    def close(self):
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)
        super().close()

    def _get_data_object(self):
        """Thread-safe way to get a data object from pool"""
        with self._pool_lock:
            if self._available_data:
                idx = self._available_data.pop()
                return idx, self._data_pool[idx]
            return None, None

    def _return_data_object(self, idx):
        """Thread-safe way to return a data object to pool"""
        with self._pool_lock:
            self._available_data.append(idx)

    def _check_collision_batch(self, qs_batch, collision_found):
        """Check collision for a batch of configurations"""
        data_idx, data = self._get_data_object()
        if data is None:
            return True  # Assume collision if can't get data object

        try:
            for q in qs_batch:
                # Early termination if collision already found by another thread
                if collision_found.is_set():
                    return False  # Another thread found collision, this batch is irrelevant

                data.qpos[self._all_robot_idx] = q
                data.qvel[:] = 0
                # TODO: deal with set to scenegraph
                assert not self.manipulating_env

                mujoco.mj_forward(self.model, data)

                # Check for collision
                for c_idx in range(data.ncon):
                    if data.contact[c_idx].dist < self.collision_tolerance:
                        collision_found.set()  # Signal other threads to stop
                        return True  # This batch found collision

            return False  # No collision found in this batch
        finally:
            self._return_data_object(data_idx)

    def _batch_is_collision_free_optimized(self, qs: List[np.ndarray]) -> bool:
        """
        Optimized batch collision checking with CORRECT batching
        """
        if not qs:
            return True

        n = len(qs)

        # For small batches, sequential is often faster due to overhead
        if n < 10:
            return self._sequential_collision_check(qs)

        num_threads = min(len(self._data_pool), n)
        collision_found = threading.Event()

        # Create batches ensuring ALL indices are covered - THIS IS THE FIX!
        batches = []
        for i in range(num_threads):
            # Calculate start and end indices for this thread
            start_idx = i * n // num_threads
            end_idx = (i + 1) * n // num_threads

            # For the last thread, ensure we go to the very end
            if i == num_threads - 1:
                end_idx = n

            if start_idx < end_idx:  # Only create batch if there's work to do
                batch = qs[start_idx:end_idx]
                batches.append(batch)

        # Submit all batch jobs
        futures = []
        for batch in batches:
            future = self._executor.submit(
                self._check_collision_batch, batch, collision_found
            )
            futures.append(future)

        # Wait for results - any collision means edge is not collision-free
        collision_detected = False

        for future in futures:
            try:
                has_collision = future.result(timeout=10.0)
                if has_collision:
                    collision_detected = True
                    collision_found.set()  # Signal other threads to stop
                    break
            except Exception as e:
                print(f"Error in collision checking: {e}")
                collision_detected = True  # Assume collision on error
                break

        # Cancel remaining futures if collision found
        if collision_detected:
            for future in futures:
                future.cancel()

        return not collision_detected

    def _sequential_collision_check(self, qs: List[np.ndarray]) -> bool:
        """Fallback sequential collision checking"""
        data = self._data_pool[0]  # Use first data object for sequential

        for q in qs:
            data.qpos[self._all_robot_idx] = q
            data.qvel[:] = 0
            # TODO: deal with set to scenegraph
            assert not self.manipulating_env

            mujoco.mj_forward(self.model, data)

            for c_idx in range(data.ncon):
                if data.contact[c_idx].dist < self.collision_tolerance:
                    return False
        return True

    def is_edge_collision_free(
        self,
        q1: Configuration,
        q2: Configuration,
        mode: Mode,
        resolution: Optional[float] = None,
        tolerance: Optional[float] = None,
        include_endpoints: bool = False,
        N_start: int = 0,
        N_max: Optional[int] = None,
        N: Optional[int] = None,
        force_parallel: bool = False,
    ) -> bool:
        if resolution is None:
            resolution = self.collision_resolution

        if tolerance is None:
            tolerance = self.collision_tolerance

        if N is None:
            N = int(config_dist(q1, q2, "max") / resolution) + 1
            N = max(2, N)

        if N_start > N:
            return True

        if N_max is None:
            N_max = N

        N_max = min(N, N_max)

        # Generate indices using your existing binary search method
        idx = generate_binary_search_indices(N)

        q1_state = q1.state()
        q2_state = q2.state()
        dir = (q2_state - q1_state) / (N - 1)

        # Prepare configurations to check
        qs = []
        for i in idx[N_start:N_max]:
            if not include_endpoints and (i == 0 or i == N - 1):
                continue
            q = q1_state + dir * i
            qs.append(q)

        if not qs:
            return True

        # Decide whether to use parallel or sequential based on problem size
        # use_parallel = force_parallel or (len(qs) >= 20 and len(self._data_pool) > 1)
        use_parallel = False

        if use_parallel:
            return self._batch_is_collision_free_optimized(qs)
        else:
            return self._sequential_collision_check(qs)


class MjxEnv(MujocoEnvironment):
    def __init__(self, xml_path):
        super().__init__(xml_path)

        self.mjx_model = mjx.put_model(self.model)

        self.mjx_data = mjx.make_data(self.mjx_model)
        
        # n_batch = 4096
        self.n_batch = 512
        self.mjx_batch = jax.tree.map(lambda x: jax.numpy.broadcast_to(x, (self.n_batch,) + x.shape), self.mjx_data)
        
        # _ = self.mjx_data.qpos.at[0].set(0)
        # _ = self.mjx_data.qvel.at[0].set(0)
        # mjx.forward(self.mjx_model, self.mjx_data)

        # for i in range(self.data.ncon):
        #     if self.data.contact[i].dist < self.collision_tolerance:
        #         return False

        self.jit_fwd = mjx.forward
        self.jit_step = mjx.step

    def _batch_is_collision_free_optimized(self, qs):
        qs = jax.numpy.stack(qs)  # shape (batch_size, n_dof)
        n_qs = qs.shape[0]

        # If qs smaller than pre-batched, slice
        if n_qs <= self.n_batch:
            batch_data = jax.tree.map(lambda x: x[:n_qs], self.mjx_batch)
        else:
            # Otherwise, replicate base data to create a new batch
            batch_data = jax.tree.map(lambda x: jax.numpy.broadcast_to(x, (n_qs,) + x.shape[1:]), self.mjx_data)

        # Replace qpos
        batch_data = batch_data.replace(qpos=qs)

        # Forward and collision check
        batch_forward = jax.vmap(lambda d: mjx.forward(self.mjx_model, d))
        batch_data = batch_forward(batch_data)

        def is_free(d):
            return jax.numpy.all(d.contact.dist >= self.collision_tolerance)

        batch_is_free = jax.vmap(is_free)(batch_data)
        return batch_is_free
    
    def _sequential_collision_check(self, qs):
        for q in qs:
            print(q)
            # TODO: deal with set to scenegraph
            assert not self.manipulating_env
    
            self.mjx_data = self.mjx_data.replace(
                qpos = self.mjx_data.qpos.at[self._all_robot_idx].set(q)
            )
            # _ = self.mjx_data.qvel.at[0].set(0)
            
            self.jit_step(self.mjx_model, self.mjx_data)
            # self.jit_fwd(self.mjx_model, self.mjx_data)

            if self.mjx_data.ncon > 0:  # optional, avoid empty contact array
                if jax.numpy.any(self.mjx_data.contact.dist < self.collision_tolerance):
                    return False
        return True

    def is_edge_collision_free(
        self,
        q1: Configuration,
        q2: Configuration,
        mode: Mode,
        resolution: Optional[float] = None,
        tolerance: Optional[float] = None,
        include_endpoints: bool = False,
        N_start: int = 0,
        N_max: Optional[int] = None,
        N: Optional[int] = None,
        force_parallel: bool = False,
    ) -> bool:
        if resolution is None:
            resolution = self.collision_resolution

        if tolerance is None:
            tolerance = self.collision_tolerance

        if N is None:
            N = int(config_dist(q1, q2, "max") / resolution) + 1
            N = max(2, N)

        if N_start > N:
            return True

        if N_max is None:
            N_max = N

        N_max = min(N, N_max)

        # Generate indices using your existing binary search method
        idx = generate_binary_search_indices(N)

        q1_state = q1.state()
        q2_state = q2.state()
        dir = (q2_state - q1_state) / (N - 1)

        # Prepare configurations to check
        qs = []
        for i in idx[N_start:N_max]:
            if not include_endpoints and (i == 0 or i == N - 1):
                continue
            q = q1_state + dir * i
            qs.append(q)

        if not qs:
            return True

        # Decide whether to use parallel or sequential based on problem size
        # use_parallel = force_parallel or (len(qs) >= 20 and len(self._data_pool) > 1)
        use_parallel = False

        if use_parallel:
            return self._batch_is_collision_free_optimized(qs)
        else:
            return self._sequential_collision_check(qs)


@register("mujoco.swap")
class simple_mujoco_env(SequenceMixin, OptimizedMujocoEnvironment):
    def __init__(self):
        path = os.path.join(
            os.path.dirname(__file__), "../assets/models/mujoco/mj_two_dim.xml"
        )
        self.robots = [
            "a1",
            "a2",
        ]

        self.start_pos = NpConfiguration.from_list(
            [
                np.array([0, -1, 0]),
                np.array([0, 1, 0]),
            ]
        )

        OptimizedMujocoEnvironment.__init__(self, path)

        self.tasks = [
            Task("a1_goal", ["a1"], SingleGoal(np.array([0, 1, 0.0]))),
            Task("a2_goal", ["a2"], SingleGoal(np.array([0.0, -1, 0.0]))),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(self.start_pos.state()),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a1_goal", "a2_goal", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.00

        # self.show_config(self.start_pos)


@register("mujoco.hallway")
class simple_mujoco_env(SequenceMixin, MujocoEnvironment):
    def __init__(self):
        path = os.path.join(
            os.path.dirname(__file__), "../assets/models/mujoco/mj_hallway.xml"
        )
        self.robots = [
            "a1",
            "a2",
        ]

        self.start_pos = NpConfiguration.from_list(
            [
                np.array([1.5, 0.0, 0]),
                np.array([-1.5, 0.0, 0]),
            ]
        )

        MujocoEnvironment.__init__(self, path)

        self.tasks = [
            Task("a1_goal", ["a1"], SingleGoal(np.array([-1.5, 1, np.pi / 2]))),
            Task("a2_goal", ["a2"], SingleGoal(np.array([1.5, 1, 0.0]))),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(self.start_pos.state()),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a1_goal", "a2_goal", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.00

        # self.show_config(self.start_pos)


@register("mujoco.manip")
class manip_mujoco_env(SequenceMixin, MujocoEnvironment):
    def __init__(self):
        path = os.path.join(os.path.dirname(__file__), "../assets/models/mujoco/mj_manip.xml")
        self.robots = [
            "a1",
            "a2",
        ]

        self.start_pos = NpConfiguration.from_list(
            [
                np.array([1.5, 0.0, 0]),
                np.array([-1.5, 0.0, 0]),
            ]
        )

        MujocoEnvironment.__init__(self, path)
        self.manipulating_env = True

        self.tasks = [
            Task(
                "a2_pick",
                ["a2"],
                SingleGoal(np.array([1.0, 0.6, 0])),
                type="pick",
                frames=["a2", "obj1"],
            ),
            Task("a1_goal", ["a1"], SingleGoal(np.array([-1.5, 1, 0.0]))),
            Task(
                "a2_place",
                ["a2"],
                SingleGoal(np.array([-1.0, -1.1, 0])),
                type="place",
                frames=["floor", "obj1"],
            ),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(self.start_pos.state()),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a2_pick", "a1_goal", "a2_place", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.00

        # self.show_config(self.start_pos)


@register("mujoco.piano")
class piano_mujoco_env(SequenceMixin, MujocoEnvironment):
    def __init__(self):
        path = os.path.join(os.path.dirname(__file__), "../assets/models/mujoco/mj_piano.xml")
        self.robots = [
            "a1",
            "a2",
        ]

        self.start_pos = NpConfiguration.from_list(
            [
                np.array([1.5, 0.0, 0]),
                np.array([-1.5, 0.0, 0]),
            ]
        )

        MujocoEnvironment.__init__(self, path)
        self.manipulating_env = True

        self.tasks = [
            Task(
                "a2_pick",
                ["a2"],
                SingleGoal(np.array([1.0, 0.6, 0])),
                type="pick",
                frames=["a2", "obj1"],
            ),
            Task(
                "a1_pick",
                ["a1"],
                SingleGoal(np.array([-1.0, -0.6, 0.0])),
                type="pick",
                frames=["a1", "obj2"],
            ),
            Task(
                "a2_place",
                ["a2"],
                SingleGoal(np.array([-1.0, -1.1, 0])),
                type="place",
                frames=["floor", "obj1"],
            ),
            Task(
                "a1_place",
                ["a1"],
                SingleGoal(np.array([1.0, 1.1, 0])),
                type="place",
                frames=["floor", "obj2"],
            ),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(self.start_pos.state()),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a2_pick", "a1_pick", "a2_place", "a1_place", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.00

        # self.show_config(self.start_pos)


# @register("mujoco.four_panda")
# class four_arm_mujoco_env(SequenceMixin, OptimizedMujocoEnvironment):
#     def __init__(self, agents_can_rotate=True):
#         path = "/home/valentin/Downloads/roboballet/data/mujoco_world/4_pandas_world_closer.xml"

#         self.robots = [
#             "panda1",
#             "panda2",
#             "panda3",
#             "panda4",
#         ]

#         self.start_pos = NpConfiguration.from_list(
#             [np.array([0, -0.5, 0, -2, 0, 2, -0.5]) for r in self.robots]
#         )

#         OptimizedMujocoEnvironment.__init__(self, path)

#         self.tasks = [
#             Task(
#                 ["panda1"], SingleGoal(np.array([-1, 0.05, 0.4, -2, 0.17, 2.5, -1.5]))
#             ),
#             Task(
#                 ["panda2"], SingleGoal(np.array([0.2, 0.05, 0.4, -2, 0.17, 2.5, -1.5]))
#             ),
#             Task(
#                 ["panda3"], SingleGoal(np.array([-1, 0.05, 0.4, -2, 0.17, 2.5, -1.5]))
#             ),
#             Task(
#                 ["panda4"], SingleGoal(np.array([0.2, 0.05, 0.4, -2, 0.17, 2.5, -1.5]))
#             ),
#             # terminal mode
#             Task(
#                 self.robots,
#                 SingleGoal(self.start_pos.state()),
#             ),
#         ]

#         self.tasks[0].name = "p1_goal"
#         self.tasks[1].name = "p2_goal"
#         self.tasks[2].name = "p3_goal"
#         self.tasks[3].name = "p4_goal"
#         self.tasks[4].name = "terminal"

#         self.sequence = self._make_sequence_from_names(
#             ["p1_goal", "p2_goal", "p3_goal", "p4_goal", "terminal"]
#         )

#         # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
#         BaseModeLogic.__init__(self)

#         self.collision_resolution = 0.01
#         self.collision_tolerance = 0.00


@register("mujoco.four_ur10")
class four_arm_ur10_mujoco_env(SequenceMixin, MjxEnv):
    def __init__(self, agents_can_rotate=True):
        path = os.path.join(
            os.path.dirname(__file__), "../assets/models/mujoco/mujoco_4_ur10_world_closer.xml"
        )

        self.robots = [
            "ur10_1",
            "ur10_2",
            "ur10_3",
            "ur10_4",
        ]

        self.start_pos = NpConfiguration.from_list(
            [np.array([0, -2, 1.0, -1.0, -1.57, 1.0]) for r in self.robots]
        )

        MjxEnv.__init__(self, path)

        self.tasks = [
            Task("p1_goal", ["ur10_1"], SingleGoal(np.array([-1, -1, 1.3, -1.0, -1.57, 1.0]))),
            Task("p2_goal", ["ur10_2"], SingleGoal(np.array([1, -1, 1.3, -1.0, -1.57, 1.0]))),
            Task("p3_goal", ["ur10_3"], SingleGoal(np.array([-1, -1, 1.3, -1.0, -1.57, 1.0]))),
            Task("p4_goal", ["ur10_4"], SingleGoal(np.array([1, -1, 1.3, -1.0, -1.57, 1.0]))),
            # terminal mode
            Task(
                "terminal",
                self.robots,
                SingleGoal(self.start_pos.state()),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["p1_goal", "p2_goal", "p3_goal", "p4_goal", "terminal"]
        )

        BaseModeLogic.__init__(self)

        self.collision_resolution = 0.01
        self.collision_tolerance = 0.01

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE

        self.safe_pose = {
            "ur10_1": np.array([0, -2, 1.0, -1.0, -1.57, 1.0]),
            "ur10_2": np.array([0, -2, 1.0, -1.0, -1.57, 1.0]),
            "ur10_3": np.array([0, -2, 1.0, -1.0, -1.57, 1.0]),
            "ur10_4": np.array([0, -2, 1.0, -1.0, -1.57, 1.0]),
        }



# @register("mujoco.rfl")
# class rfl_mujoco(SequenceMixin, OptimizedMujocoEnvironment):
