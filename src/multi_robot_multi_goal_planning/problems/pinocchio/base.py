import numpy as np

import time
import sys
from numba import jit, float64  # ty: ignore[possibly-unbound-import]
import json

from typing import List, Optional
from numpy.typing import NDArray

from ..planning_env import generate_binary_search_indices

from ..core.configuration import Configuration, NpConfiguration, config_dist

from ..planning_env import (
    BaseModeLogic,
    SequenceMixin,
    DependencyGraphMixin,
    State,
    Mode,
    Task,
    BaseProblem,
    ProblemSpec,
    AgentType,
    GoalType,
    ConstraintType,
    DynamicsType,
    ManipulationType,
    DependencyType,
    SafePoseType,
)
from ..core.goals import SingleGoal, GoalSet, GoalRegion, ConditionalGoal

import pinocchio as pin
import hppfcl as fcl
from pathlib import Path

from pinocchio.visualize import MeshcatVisualizer

from ..core.registry import register


@jit((float64[:, :], float64[:, :]), nopython=True)
def multiply_4x4_matrices(mat1, mat2):
    result = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i, j] += mat1[i, k] * mat2[k, j]
    return result


class PinocchioEnvironment(BaseProblem):
    # misc
    collision_tolerance: float
    collision_resolution: float

    def __init__(
        self, model, collision_model, visual_model, start_pos, robots, root_name="table"
    ):
        super().__init__()

        self.limits = np.vstack([model.lowerPositionLimit, model.upperPositionLimit])

        self.model = model
        self.collision_model = collision_model
        self.visual_model = visual_model

        self.data = self.model.createData()
        self.geom_data = pin.GeometryData(self.collision_model)

        self.start_pos = start_pos
        self.robots = robots

        offset = 0
        self.robot_idx = {}
        self.robot_dims = {}
        for i, r in enumerate(self.robots):
            dim = len(self.start_pos[i])
            self.robot_idx[r] = [i + offset for i in range(dim)]
            self.robot_dims[r] = dim
            offset += dim

        self.viz = None

        if not hasattr(self, "collision_tolerance"):
            self.collision_tolerance = 0.00
        if not hasattr(self, "collision_resolution"):
            self.collision_resolution = 0.01

        self.cost_metric = "euclidean"
        self.cost_reduction = "max"

        self.manipulating_env = False

        self.initial_sg = {}

        ids = range(len(collision_model.geometryObjects))

        self.root_name = root_name

        for i, id_1 in enumerate(ids):
            obj_name = collision_model.geometryObjects[id_1].name
            if obj_name[:3] == "obj" or obj_name[:3] == "box":
                parent = collision_model.geometryObjects[id_1].parentJoint
                placement = collision_model.geometryObjects[id_1].placement
                # print(obj_name)
                # print(placement)
                self.initial_sg[id_1] = (
                    self.root_name,
                    parent,
                    np.round(placement, 3).tobytes(),
                    pin.SE3(placement),
                )

        self.current_scenegraph = self.initial_sg.copy()

        n = len(self.collision_model.geometryObjects)
        mat = np.zeros((n, n)) - self.collision_tolerance

        self.geom_data.setSecurityMargins(self.collision_model, mat)

        self.spec = ProblemSpec(
            agent_type=AgentType.MULTI_AGENT,
            constraints=ConstraintType.UNCONSTRAINED,
            manipulation=ManipulationType.MANIPULATION,
            dependency=DependencyType.FULLY_ORDERED,
            dynamics=DynamicsType.GEOMETRIC,
            goals=GoalType.MULTI_GOAL,
            home_pose=SafePoseType.HAS_NO_SAFE_HOME_POSE,
        )

    def setup_visualization(self):
        self.viz = MeshcatVisualizer(
            self.model, self.collision_model, self.visual_model
        )

        try:
            self.viz.initViewer(open=False)
            self.viz.viewer["/Background"].set_property("top_color", [0.9, 0.9, 0.9])
            self.viz.viewer["/Background"].set_property("bottom_color", [0.9, 0.9, 0.9])

            self.viz.viewer["/Grid"].set_property("color", [1, 0, 0, 0.2])
        except ImportError as err:
            print(
                "Error while initializing the viewer. It seems you should install Python meshcat"
            )
            print(err)
            sys.exit(0)

        # Load the robot in the viewer.
        self.viz.loadViewerModel()

        # Display a robot configuration.
        self.viz.displayVisuals(True)
        self.viz.displayCollisions(True)

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
            q = path[i].q.state()
            pin.forwardKinematics(self.model, self.data, q)

            pin.updateGeometryPlacements(
                self.model, self.data, self.collision_model, self.geom_data, q
            )
            self._set_to_scenegraph(path[i].mode.sg)

            self.show_config(path[i].q, stop)

            if export:
                print("Exporting is not supported in pinocchio envs.")
                # with viz.create_video_ctx("../leap.mp4"):
                #   viz.play(qs, dt, callback=my_callback)

            dt = pause_time
            if adapt_to_max_distance:
                if i < len(path) - 1:
                    v = 5
                    diff = config_dist(path[i].q, path[i + 1].q, "max_euclidean")
                    dt = diff / v

            time.sleep(dt)

            if stop:
                input("Press Enter to continue...")

        if stop_at_end:
            self.show_config(path[-1].q, True)

    def set_to_mode(
        self,
        mode: Mode,
        config=None,
        use_cached: bool = True,
        place_in_cache: bool = True,
    ):
        pass

    # @profile # run with kernprof -l scripts/run_planner.py [your environment] [your flags]
    def _set_to_scenegraph(
        self, sg, update_visual: bool = False, update_collision_pairs=False
    ):
        # update object positions
        # placement = pin.SE3.Identity()
        for frame_id, (parent, parent_joint, pose, placement) in sg.items():
            # placement = pin.SE3(np.frombuffer(pose).reshape(4, 4))

            # print(frame_id)
            if (
                frame_id in self.current_scenegraph
                and parent == self.current_scenegraph[frame_id][0]
                and parent == self.root_name
                and self.current_scenegraph[frame_id][2] == pose
            ):
                #     # print("A")
                continue

            frame_pose = self.data.oMi[parent_joint].act(placement)
            # frame_pose = multiply_4x4_matrices(self.data.oMi[parent_joint].homogeneous, placement.homogeneous)
            # frame_pose = pin.SE3(frame_pose)
            self.collision_model.geometryObjects[frame_id].placement = frame_pose

            if update_visual:
                frame_name = self.collision_model.geometryObjects[frame_id].name
                vis_frame_id = self.visual_model.getGeometryId(frame_name)
                self.visual_model.geometryObjects[vis_frame_id].placement = frame_pose

            if (
                update_collision_pairs
                and self.current_scenegraph[frame_id][0] != sg[frame_id][0]
            ):
                # disable collisions for the objects that are linked together
                new_parent_id = self.collision_model.getGeometryId(parent)
                old_parent_id = self.collision_model.getGeometryId(
                    self.current_scenegraph[frame_id][0]
                )

                # print(self.collision_model.geometryObjects[frame_id].name, parent)
                # print("removing", frame_id, new_parent_id)

                # print(self.collision_model.geometryObjects[frame_id].name, self.current_scenegraph[frame_id][0])
                # print("adding", frame_id, old_parent_id)

                if self.collision_model.existCollisionPair(
                    pin.CollisionPair(frame_id, new_parent_id)
                ):
                    new_pair_id = self.collision_model.findCollisionPair(
                        pin.CollisionPair(frame_id, new_parent_id)
                    )
                    self.geom_data.deactivateCollisionPair(new_pair_id)

                # re-enable them for the objects that are not linked together anymore
                if self.collision_model.existCollisionPair(
                    pin.CollisionPair(frame_id, old_parent_id)
                ):
                    old_pair_id = self.collision_model.findCollisionPair(
                        pin.CollisionPair(frame_id, old_parent_id)
                    )
                    self.geom_data.activateCollisionPair(old_pair_id)

            self.current_scenegraph[frame_id] = sg[frame_id]

        # self.viz.display()

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
        pin.forwardKinematics(self.model, self.data, q)

        pin.updateGeometryPlacements(
            self.model, self.data, self.collision_model, self.geom_data, q
        )
        self._set_to_scenegraph(sg)

        last_task = self.tasks[prev_mode_index]

        if last_task.type is not None:
            if last_task.type == "goto":
                pass
            else:
                # print(f"frame 0: {last_task.frames[0]}")
                # print(f"frame 1: {last_task.frames[1]}")

                # update scene graph by changing the links as below
                obj_id = self.collision_model.getGeometryId(last_task.frames[1])
                new_parent_id = self.collision_model.getGeometryId(last_task.frames[0])

                # print(f"frame id: {frame_id}")
                # print(f"new parent id: {new_parent_id}")
                new_parent_joint = self.collision_model.geometryObjects[
                    new_parent_id
                ].parentJoint
                obj_parent_joint = self.collision_model.geometryObjects[
                    obj_id
                ].parentJoint

                new_parent_frame_abs_pose = (
                    self.data.oMi[new_parent_joint]
                    # * self.collision_model.geometryObjects[new_parent_id].placement
                )
                old_parent_frame_abs_pose = self.data.oMi[obj_parent_joint]
                # obj_parent_joint_id = self.collision_model.geometryObjects[new_parent_id].parentJoint

                obj_frame_abs_pose = (
                    old_parent_frame_abs_pose
                    * self.collision_model.geometryObjects[obj_id].placement
                )

                obj_pose_in_new_frame = (
                    obj_frame_abs_pose.inverse() * new_parent_frame_abs_pose
                )

                sg[obj_id] = (
                    # sg[last_task.frames[1]] = (
                    last_task.frames[0],
                    new_parent_joint,
                    np.round(obj_pose_in_new_frame.inverse(), 3).tobytes(),
                    pin.SE3(obj_pose_in_new_frame.inverse()),
                )

        return sg

    def show(self, blocking=True):
        if self.viz is None:
            self.setup_visualization()

        self.viz.display()

        if blocking:
            input("Press Enter to continue...")

    def show_config(self, q, blocking=True):
        if self.viz is None:
            self.setup_visualization()

        self.viz.display(q.state())

        if blocking:
            input("Press Enter to continue...")



    # @profile # run with kernprof -l scripts/run_planner.py [your environment] [your flags]
    def is_collision_free(self, q: Optional[Configuration], mode: Optional[Mode]):
        if q is None:
            raise ValueError

        # self.show_config(q, blocking=False)

        # q_orig = q

        # print(mode)

        if isinstance(q, Configuration):
            q = q.state()

        # self.show_config(q_orig, blocking=False)

        # pin.forwardKinematics(self.model, self.data, q)
        pin.updateGeometryPlacements(
            self.model, self.data, self.collision_model, self.geom_data, q
        )

        # update object positions
        if mode:
            self._set_to_scenegraph(mode.sg)

        pin.updateGeometryPlacements(
            self.model, self.data, self.collision_model, self.geom_data
        )

        # self.viz.display(q)
        # print(mode)
        # input("AA")

        # self.viz.updatePlacements(pin.GeometryType.COLLISION)
        # self.viz.updatePlacements(pin.GeometryType.VISUAL)

        if False:
            for i in range(len(self.collision_model.collisionPairs)):
                in_collision = pin.computeCollision(
                    self.collision_model, self.geom_data, i
                )
                if in_collision:
                    cr = self.geom_data.collisionResults[i]
                    cp = self.collision_model.collisionPairs[i]
                    if cr.isCollision():
                        print(
                            self.collision_model.geometryObjects[cp.first].name,
                            self.collision_model.geometryObjects[cp.second].name,
                        )
                        print(
                            "collision pair:",
                            cp.first,
                            ",",
                            cp.second,
                            "- collision:",
                            "Yes" if cr.isCollision() else "No",
                        )
            # print(q)
            # print('colliding')
            # input("A")

            #         return False

            # return True

        in_collision = pin.computeCollisions(self.collision_model, self.geom_data, True)
        # in_collision = pin.computeCollisionsparallel(self.collision_model, self.geom_data, True)

        if in_collision:
            # pin.computeDistances(self.collision_model, self.geom_data)

            # total_penetration = 0
            # for k in range(len(self.collision_model.collisionPairs)):
            #     cr = self.geom_data.distanceResults[k]
            #     # cp = self.collision_model.collisionPairs[k]
            #     if cr.min_distance < 0:
            #         total_penetration += -cr.min_distance
            #     # print(cr.min_distance)
            #     # if cr.isCollision():
            #     #     print(self.collision_model.geometryObjects[cp.first].name, self.collision_model.geometryObjects[cp.second].name)
            #     #     print("collision pair:", cp.first,",",cp.second,"- collision:","Yes" if cr.isCollision() else "No")
            # # print(q)
            # # print('colliding')
            # self.show_config(q_orig, blocking=True)

            # if total_penetration > self.collision_tolerance:
            return False

        # print(mode)
        # self.show_config(q_orig, blocking=False)

        return True

    def is_collision_free_for_robot(
        self,
        r: List[str] | str,
        q: NDArray,
        m: Optional[Mode] = None,
        collision_tolerance: Optional[float] = None,
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
        pin.updateGeometryPlacements(
            self.model, self.data, self.collision_model, self.geom_data, q
        )

        # update object positions
        if m:
            self._set_to_scenegraph(m.sg)

        pin.updateGeometryPlacements(
            self.model, self.data, self.collision_model, self.geom_data
        )
        in_collision = pin.computeCollisions(self.collision_model, self.geom_data, True)
        if not in_collision:
            return True
        pin.computeDistances(self.collision_model, self.geom_data)

        total_penetration = 0
        min_dist = {}
        for k in range(len(self.collision_model.collisionPairs)):
            cr = self.geom_data.distanceResults[k]
            # cp = self.collision_model.collisionPairs[k]
            if cr.min_distance < 0:
                total_penetration += -cr.min_distance
            min_dist[k] = -cr.min_distance
            # print(cr.min_distance)
            # if cr.isCollision():
            #     print(self.collision_model.geometryObjects[cp.first].name, self.collision_model.geometryObjects[cp.second].name)
            #     print("collision pair:", cp.first,",",cp.second,"- collision:","Yes" if cr.isCollision() else "No")
        # print(q)
        # print('colliding')
        # self.show_config(q, blocking=True)
        # TODO need to be tested properly
        collisionPairs = self.collision_model.collisionPairs.tolist()
        if total_penetration > self.collision_tolerance:
            for k in range(len(self.collision_model.collisionPairs)):
                dist = min_dist[k]
                # ignore minor collisions
                if min_dist[k] > -collision_tolerance / 10:
                    continue
                object1 = self.collision_model.geometryObjects[
                    collisionPairs[k].first
                ].name
                object2 = self.collision_model.geometryObjects[
                    collisionPairs[k].second
                ].name
                # print(c)
                involves_relevant_robot = False
                involves_relevant_object = False
                for robot in r:
                    task_idx = m.task_ids[self.robots.index(robot)]
                    task = self.tasks[task_idx]
                    if dist < 0 and (robot in object1 or robot in object2):
                        involves_relevant_robot = True
                    elif dist < 0 and (
                        object1 in task.frames or object2 in task.frames
                    ):
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
                    if other_robot in object1 or other_robot in object2:
                        is_collision_with_other_robot = True
                        break

                if not is_collision_with_other_robot:
                    # print(c)
                    return False

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
            q = NpConfiguration(q, q1._array_slice)

            if not self.is_collision_free(q, mode):
                # print(q)
                # print("crash ", mode)
                # self.viz.display(q)
                return False

        # print(mode)

        return True
