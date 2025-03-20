import numpy as np

from typing import List, Dict, Optional
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.util import generate_binary_search_indices

from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
    NpConfiguration,
    config_dist,
    config_cost,
    batch_config_cost,
)

from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseModeLogic,
    SequenceMixin,
    DependencyGraphMixin,
    State,
    Mode,
    Task,
    SingleGoal,
    GoalSet,
    GoalRegion,
    ConditionalGoal,
    BaseProblem,
)

import pinocchio as pin
import hppfcl as fcl
from pathlib import Path

import json

from pinocchio.visualize import MeshcatVisualizer

import time
import sys
import numba

@numba.jit((numba.float64[:, :], numba.float64[:, :]), nopython=True)
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

    def __init__(self, model, collision_model, visual_model):
        self.limits = np.vstack([model.lowerPositionLimit, model.upperPositionLimit])
        self.start_pos = None

        self.model = model
        self.collision_model = collision_model
        self.visual_model = visual_model

        self.data = self.model.createData()
        self.geom_data = pin.GeometryData(self.collision_model)

        self.viz = None

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.01

        self.manipulating_env = False

        self.initial_sg = {}

        ids = range(len(collision_model.geometryObjects))

        self.root_name = "table"

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

        # self.geom_data.setSecurityMargins(self.collision_model, mat)

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

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def _set_to_scenegraph(self, sg, update_visual: bool = False, update_collision_pairs = False):
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

            if update_collision_pairs and self.current_scenegraph[frame_id][0] != sg[frame_id][0]:
                # disable collisions for the objects that are linked together
                new_parent_id = self.collision_model.getGeometryId(parent)
                old_parent_id = self.collision_model.getGeometryId(self.current_scenegraph[frame_id][0])

                # print(self.collision_model.geometryObjects[frame_id].name, parent)
                # print("removing", frame_id, new_parent_id)

                # print(self.collision_model.geometryObjects[frame_id].name, self.current_scenegraph[frame_id][0])
                # print("adding", frame_id, old_parent_id)

                if self.collision_model.existCollisionPair(pin.CollisionPair(frame_id, new_parent_id)):
                    new_pair_id = self.collision_model.findCollisionPair(pin.CollisionPair(frame_id, new_parent_id))
                    self.geom_data.deactivateCollisionPair(new_pair_id)

                # re-enable them for the objects that are not linked together anymore
                if self.collision_model.existCollisionPair(pin.CollisionPair(frame_id, old_parent_id)):
                    old_pair_id = self.collision_model.findCollisionPair(pin.CollisionPair(frame_id, old_parent_id))
                    self.geom_data.activateCollisionPair(old_pair_id)

            self.current_scenegraph[frame_id] = sg[frame_id]

        # self.viz.display()

    def get_scenegraph_info_for_mode(self, mode: Mode):
        if not self.manipulating_env:
            return {}

        # self.set_to_mode(mode)
        prev_mode = mode.prev_mode
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

        # self.viz.updatePlacements(pin.GeometryType.COLLISION)
        # self.viz.updatePlacements(pin.GeometryType.VISUAL)

        # self.viz.display()
        # input("BB")

        # self.viz.display(q)

        last_task = self.tasks[prev_mode_index]

        # print("prev mode")
        # print(mode_switching_robots)
        # print(q)
        # print(prev_mode)

        # print(last_task.name)
        # print(active_task.name)

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

                # print("new paren tpose", last_task.frames[0])
                # print(new_parent_frame_abs_pose)

                # print("obj pose in new frame", last_task.frames[1])
                # print(obj_pose_in_new_frame)

                # self.viz.display()
                # input("BB")

                # rel_pose_in_world_frame = obj_frame_abs_pose.actInv(
                #     new_parent_frame_abs_pose
                # )
                # rel_pose_in_joint_frame = (
                #     new_parent_frame_abs_pose * rel_pose_in_world_frame
                # )

                # print(last_task.frames[0])
                # print(last_task.frames[1])
                # print(rel_pose_in_joint_frame)

                sg[obj_id] = (
                    # sg[last_task.frames[1]] = (
                    last_task.frames[0],
                    new_parent_joint,
                    np.round(obj_pose_in_new_frame.inverse(), 3).tobytes(),
                    pin.SE3(obj_pose_in_new_frame.inverse()),
                )

                # if last_task.name == "a1_pick_obj1":
                #     print(last_task.frames[1])
                #     print(old_parent_frame_abs_pose)
                #     print(last_task.frames[0])
                #     print(new_parent_frame_abs_pose)
                #     print("obj abs pose")
                #     print(obj_frame_abs_pose)
                #     print("obj rel pose")
                #     print(rel_pose_in_world_frame)
                #     print("obj rel pose in new joint frame")
                #     print(rel_pose_in_joint_frame)

                #     print(sg)
                # tmp.attach(
                #     self.tasks[prev_mode_index].frames[0],
                #     self.tasks[prev_mode_index].frames[1],
                # )
                # tmp.getFrame(self.tasks[prev_mode_index].frames[1]).setContact(-1)

            # postcondition
            # if self.tasks[prev_mode_index].side_effect is not None:
            #     box = self.tasks[prev_mode_index].frames[1]
            #     tmp.delFrame(box)

        # print(last_task.name)
        # print(sg)
        # print(rel_pose_in_joint_frame)
        # self.show()

        # if last_task.name == "a1_pick_obj1":
        #     print(mode)
        #     print(last_task.name)
        #     print(sg)

        #     print(rel_pose_in_joint_frame)

        # self.show()

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

    def config_cost(self, start: Configuration, end: Configuration) -> float:
        return config_cost(start, end, self.cost_metric, self.cost_reduction)

    def batch_config_cost(
        self, starts: List[Configuration], ends: List[Configuration]
    ) -> NDArray:
        return batch_config_cost(starts, ends, self.cost_metric, self.cost_reduction)

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def is_collision_free(self, q: Optional[Configuration], mode: Mode):
        if q is None:
            raise ValueError

        # self.show_config(q, blocking=False)

        q_orig = q

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
                in_collision = pin.computeCollision(self.collision_model, self.geom_data, i)
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

    def is_edge_collision_free(
        self,
        q1: Configuration,
        q2: Configuration,
        mode: Mode,
        resolution: float = None,
        randomize_order: bool = True,
        tolerance: float = None,
    ) -> bool:
        if resolution is None:
            resolution = self.collision_resolution

        if tolerance is None:
            tolerance = self.collision_tolerance

        # print('q1', q1)
        # print('q2', q2)
        N = int(config_dist(q1, q2, "max") / resolution)
        N = max(2, N)

        idx = list(range(int(N)))
        if randomize_order:
            # np.random.shuffle(idx)
            idx = generate_binary_search_indices(int(N)).copy()

        qs = []

        for i in idx:
            # print(i / (N-1))
            q = q1.state() + (q2.state() - q1.state()) * (i) / (N - 1)
            q = NpConfiguration(q, q1.array_slice)

            if not self.is_collision_free(q, mode):
                # print(q)
                # print("crash ", mode)
                # self.viz.display(q)
                return False

        # print(mode)

        return True

    def is_path_collision_free(
        self, path: List[State], randomize_order=True, resolution=None, tolerance=None
    ) -> bool:
        if tolerance is None:
            tolerance = self.collision_tolerance

        if resolution is None:
            resolution = self.collision_resolution

        idx = list(range(len(path) - 1))
        if randomize_order:
            np.random.shuffle(idx)

        for i in idx:
            # skip transition nodes
            # if path[i].mode != path[i + 1].mode:
            #     continue

            q1 = path[i].q
            q2 = path[i + 1].q
            mode = path[i].mode

            if not self.is_edge_collision_free(q1, q2, mode, resolution=resolution):
                return False

        return True


def make_pin_middle_obstacle_two_dim_env():
    filename = "./src/multi_robot_multi_goal_planning/problems/urdfs/middle_obstacle_two_agents.urdf"
    model, collision_model, visual_model = pin.buildModelsFromUrdf(filename)

    collision_model.addAllCollisionPairs()

    return model, collision_model, visual_model


class pinocchio_middle_obs(SequenceMixin, PinocchioEnvironment):
    def __init__(self, agents_can_rotate=True):
        model, collision_model, visual_model = make_pin_middle_obstacle_two_dim_env()
        PinocchioEnvironment.__init__(self, model, collision_model, visual_model)

        self.start_pos = NpConfiguration.from_list([[0, -0.8, 0], [0, 0.8, 0]])

        self.robot_idx = {"a1": [0, 1, 2], "a2": [3, 4, 5]}
        self.robot_dims = {"a1": 3, "a2": 3}
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        self.tasks = [
            # r1
            Task(["a1"], SingleGoal(np.array([0, 0.8, 0]))),
            # r2
            Task(["a2"], SingleGoal(np.array([0, -0.8, 0]))),
            # terminal mode
            Task(
                ["a1", "a2"],
                SingleGoal(np.array([0, -0.8, 0, 0, 0.8, 0])),
            ),
        ]

        self.tasks[0].name = "a1_goal"
        self.tasks[1].name = "a2_goal"
        self.tasks[2].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            ["a2_goal", "a1_goal", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        # self.collision_resolution = 0.01
        # self.collision_tolerance = 0.01


def make_pin_other_hallway_two_dim_env():
    filename = "./src/multi_robot_multi_goal_planning/problems/urdfs/other_hallway.urdf"
    model, collision_model, visual_model = pin.buildModelsFromUrdf(filename)

    ids = range(len(collision_model.geometryObjects))

    env_names = [
        "table_0",
        "wall1_0",
        "wall2_0",
        "wall3_0",
        "wall4_0",
        "obs3_0",
        "obs4_0",
    ]

    for i, id_1 in enumerate(ids):
        for id_2 in ids[i + 1 :]:
            #  = geomModel.getGeometryId(geomModelB.geometryObjects[cp.second].name);
            if (
                collision_model.geometryObjects[id_1].name in env_names
                and collision_model.geometryObjects[id_2].name in env_names
            ):
                continue
            # print(
            #     "adding ",
            #     id_1,
            #     id_2,
            #     collision_model.geometryObjects[id_1].name,
            #     collision_model.geometryObjects[id_2].name,
            # )

            collision_model.addCollisionPair(pin.CollisionPair(id_1, id_2))

    # collision_model.addAllCollisionPairs()

    return model, collision_model, visual_model


class pinocchio_other_hallway(SequenceMixin, PinocchioEnvironment):
    def __init__(self):
        model, collision_model, visual_model = make_pin_other_hallway_two_dim_env()
        PinocchioEnvironment.__init__(self, model, collision_model, visual_model)

        self.root_name = "table_0"

        self.start_pos = NpConfiguration.from_list([[1.5, 0.0, 0], [-1.5, 0.0, 0]])

        self.robot_idx = {"a1": [0, 1, 2], "a2": [3, 4, 5]}
        self.robot_dims = {"a1": 3, "a2": 3}
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        self.tasks = [
            # r1
            Task(["a1"], SingleGoal(np.array([-1.5, 1, np.pi / 2]))),
            # r2
            Task(["a2"], SingleGoal(np.array([1.5, 1, 0]))),
            # terminal mode
            Task(
                ["a1", "a2"],
                SingleGoal(np.array([1.5, 0.0, 0, -1.5, 0.0, 0])),
            ),
        ]

        self.tasks[0].name = "a1_goal"
        self.tasks[1].name = "a2_goal"
        self.tasks[2].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            ["a2_goal", "a1_goal", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        # self.collision_resolution = 0.01
        # self.collision_tolerance = 0.01


def make_2d_handover():
    filename = "./src/multi_robot_multi_goal_planning/problems/urdfs/2d_handover.urdf"
    model, collision_model, visual_model = pin.buildModelsFromUrdf(filename)

    # collision_model.addAllCollisionPairs()

    ids = range(len(collision_model.geometryObjects))

    env_names = [
        "table_0",
        "wall1_0",
        "wall2_0",
        "wall3_0",
        "wall4_0",
        "obs1_0",
        "obs2_0",
        "obs3_0",
        "obs4_0",
    ]

    for i, id_1 in enumerate(ids):
        for id_2 in ids[i + 1 :]:
            #  = geomModel.getGeometryId(geomModelB.geometryObjects[cp.second].name);
            if (
                collision_model.geometryObjects[id_1].name in env_names
                and collision_model.geometryObjects[id_2].name in env_names
            ):
                continue

            # print(
            #     "adding ",
            #     id_1,
            #     id_2,
            #     collision_model.geometryObjects[id_1].name,
            #     collision_model.geometryObjects[id_2].name,
            # )
            collision_model.addCollisionPair(pin.CollisionPair(id_1, id_2))

    return model, collision_model, visual_model


class pinocchio_handover_two_dim(SequenceMixin, PinocchioEnvironment):
    def __init__(self):
        model, collision_model, visual_model = make_2d_handover()
        PinocchioEnvironment.__init__(self, model, collision_model, visual_model)

        self.root_name = "table_0"

        self.manipulating_env = True

        self.start_pos = NpConfiguration.from_list([[-0.5, 0.8, 0], [0.0, -0.5, 0]])

        self.robot_idx = {"a1": [0, 1, 2], "a2": [3, 4, 5]}
        self.robot_dims = {"a1": 3, "a2": 3}
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        for obj in self.collision_model.geometryObjects:
            if obj.name[-2:] == "_0":
                obj.name = obj.name[:-2]

        self.import_tasks("2d_handover_tasks.txt")

        self.sequence = self._make_sequence_from_names(
            [
                "a1_pick_obj1",
                "handover",
                "a1_pick_obj2",
                "a1_place_obj2",
                "a2_place",
                "terminal",
            ]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.start_mode.sg = self.initial_sg

    def make_start_mode(self):
        start_mode = super().make_start_mode()
        start_mode.sg = self.initial_sg
        return start_mode


def make_piano():
    filename = "./src/multi_robot_multi_goal_planning/problems/urdfs/piano.urdf"
    model, collision_model, visual_model = pin.buildModelsFromUrdf(filename)

    # collision_model.addAllCollisionPairs()

    ids = range(len(collision_model.geometryObjects))

    env_names = [
        "table_0",
        "wall1_0",
        "wall2_0",
        "wall3_0",
        "wall4_0",
        "obs1_0",
        "obs2_0",
    ]

    for i, id_1 in enumerate(ids):
        for id_2 in ids[i + 1 :]:
            #  = geomModel.getGeometryId(geomModelB.geometryObjects[cp.second].name);
            if (
                collision_model.geometryObjects[id_1].name in env_names
                and collision_model.geometryObjects[id_2].name in env_names
            ):
                continue

            # print(
            #     "adding ",
            #     id_1,
            #     id_2,
            #     collision_model.geometryObjects[id_1].name,
            #     collision_model.geometryObjects[id_2].name,
            # )
            collision_model.addCollisionPair(pin.CollisionPair(id_1, id_2))

    return model, collision_model, visual_model


class pinocchio_piano_two_dim(SequenceMixin, PinocchioEnvironment):
    def __init__(self):
        model, collision_model, visual_model = make_piano()
        PinocchioEnvironment.__init__(self, model, collision_model, visual_model)

        self.manipulating_env = True

        self.start_pos = NpConfiguration.from_list([[-0.5, 0.8, 0], [0.0, -0.5, 0]])

        self.robot_idx = {"a1": [0, 1, 2], "a2": [3, 4, 5]}
        self.robot_dims = {"a1": 3, "a2": 3}
        # self.C.view(True)

        self.root_name = "table_0"

        self.robots = ["a1", "a2"]

        task_1_pose = np.array([0.47685958, 0.22499656, 0.00066826])
        task_2_pose = np.array([-5.23035087e-01, -7.74943558e-01, 7.45490689e-04])
        task_3_pose = np.array([4.93481978e-01, -2.25359882e-01, -4.49533203e-04])
        task_4_pose = np.array([-5.06410855e-01, 7.74578852e-01, -5.39715393e-04])
        task_5_pose = np.array(
            [
                7.64079390e-12,
                -4.95512533e-01,
                8.62113109e-12,
                -2.05527456e-12,
                4.95512533e-01,
                -7.33881002e-12,
            ]
        )

        self.tasks = [
            # a1
            Task(
                ["a1"],
                SingleGoal(task_1_pose),
                type="pick",
                frames=["a1_0", "obj1_0"],
            ),
            Task(
                ["a1"],
                SingleGoal(task_2_pose),
                type="place",
                frames=["table_0", "obj1_0"],
            ),
            # a2
            Task(
                ["a2"],
                SingleGoal(task_3_pose),
                type="pick",
                frames=["a2_0", "obj2_0"],
            ),
            Task(
                ["a2"],
                SingleGoal(task_4_pose),
                type="place",
                frames=["table_0", "obj2_0"],
            ),
            # terminal
            Task(
                ["a1", "a2"],
                SingleGoal(task_5_pose),
            ),
        ]

        self.tasks[0].name = "a1_pick"
        self.tasks[1].name = "a1_place"
        self.tasks[2].name = "a2_pick"
        self.tasks[3].name = "a2_place"
        self.tasks[4].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            ["a2_pick", "a1_pick", "a2_place", "a1_place", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.start_mode.sg = self.initial_sg

    def make_start_mode(self):
        start_mode = super().make_start_mode()
        start_mode.sg = self.initial_sg
        return start_mode


def add_namespace_prefix_to_models(model, collision_model, visual_model, namespace):
    # Rename geometry objects in collision model:
    for geom in collision_model.geometryObjects:
        geom.name = f"{namespace}/{geom.name}"

    # Rename geometry objects in visual model:
    for geom in visual_model.geometryObjects:
        geom.name = f"{namespace}/{geom.name}"

    # Rename frames in model:
    for f in model.frames:
        f.name = f"{namespace}/{f.name}"

    # Rename joints in model:
    for k in range(len(model.names)):
        model.names[k] = f"{namespace}/{model.names[k]}"


def make_dual_ur5_waypoint_env():
    # urdf_path = "./src/multi_robot_multi_goal_planning/problems/urdfs/ur5e/ur5e.urdf"
    # urdf_path = "./src/multi_robot_multi_goal_planning/problems/urdfs/ur5e/ur5e_constrained_coll_primitives.urdf"
    urdf_path = "./src/multi_robot_multi_goal_planning/problems/urdfs/ur5e/ur5e_constrained.urdf"

    coll_mesh_dir = Path(urdf_path).resolve().parent / "ur_description/meshes/ur5e/collision/"
    visual_mesh_dir = Path(urdf_path).resolve().parent / "ur_description/meshes/ur5e/visual/"

    # mesh_dir = "../src/multi_robot_multi_goal_planning/problems/urdfs/ur10e/ur_description/meshes/ur10e/visual/"

    robot_1 = pin.buildModelFromUrdf(urdf_path)
    r1_coll = pin.buildGeomFromUrdf(robot_1, urdf_path, pin.GeometryType.COLLISION, coll_mesh_dir)
    r1_viz = pin.buildGeomFromUrdf(robot_1, urdf_path, pin.GeometryType.VISUAL, visual_mesh_dir)

    robot_2 = pin.buildModelFromUrdf(urdf_path)
    r2_coll = pin.buildGeomFromUrdf(robot_1, urdf_path, pin.GeometryType.COLLISION, coll_mesh_dir)
    r2_viz = pin.buildGeomFromUrdf(robot_1, urdf_path, pin.GeometryType.VISUAL, visual_mesh_dir)

    # robot_1, r1_coll, r1_viz = pin.buildModelsFromUrdf(urdf_path)
    # robot_2, r2_coll, r2_viz = pin.buildModelsFromUrdf(urdf_path)

    # robot_3, r3_coll, r3_viz = pin.buildModelsFromUrdf(urdf_path)
    # robot_4, r4_coll, r4_viz = pin.buildModelsFromUrdf(urdf_path)

    add_namespace_prefix_to_models(robot_1, r1_coll, r1_viz, "ur5_1")
    add_namespace_prefix_to_models(robot_2, r2_coll, r2_viz, "ur5_2")
    # add_namespace_prefix_to_models(robot_3, r3_coll, r3_viz, "ur5_3")
    # add_namespace_prefix_to_models(robot_4, r4_coll, r4_viz, "ur5_4")

    # Create a composite model
    world = pin.Model()
    geom_model = pin.GeometryModel()

    geom1_name = "table"
    shape1 = fcl.Box(4, 4, 0.01)
    geom1_obj = pin.GeometryObject(geom1_name, 0, pin.SE3.Identity(), shape1)
    geom1_obj.meshColor = np.ones(4)
    geom_model.addGeometryObject(geom1_obj)

    # rotation matrix around the z-axis
    theta = np.pi / 2  # Example: 45 degrees rotation around z-axis
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    joint_placement1 = pin.SE3(R, np.array([0.5, -0.5, 0.01]))  # Base of first UR5
    joint_placement2 = pin.SE3(R, np.array([-0.5, -0.5, 0.01]))  # Base of second UR5
    # joint_placement3 = pin.SE3(
    #     np.eye(3), np.array([0.0, 1.0, 0.01])
    # )  # Base of second UR5
    # joint_placement4 = pin.SE3(
    #     np.eye(3), np.array([1.0, 1.0, 0.01])
    # )  # Base of second UR5

    # Add UR5 models to the world model
    model, visual_model = pin.appendModel(
        world,
        robot_1,
        geom_model,
        r1_viz,
        0,
        joint_placement1,
    )
    model, collision_model = pin.appendModel(
        world,
        robot_1,
        geom_model,
        r1_coll,
        0,
        joint_placement1,
    )

    tmp = model.copy()

    model, visual_model = pin.appendModel(
        tmp,
        robot_2,
        visual_model,
        r2_viz,
        0,
        joint_placement2,
    )
    model, collision_model = pin.appendModel(
        tmp,
        robot_2,
        collision_model,
        r2_coll,
        0,
        joint_placement2,
    )

    collision_model.addAllCollisionPairs()

    ids = range(len(collision_model.geometryObjects))
    for i, id_1 in enumerate(ids):
        for id_2 in ids[i + 1 :]:
            #  = geomModel.getGeometryId(geomModelB.geometryObjects[cp.second].name);
            # if (
            #     "ur5_1" in collision_model.geometryObjects[id_1].name
            #     and "ur5_1" in collision_model.geometryObjects[id_2].name
            #     and abs(id_1 - id_2) < 4
            # ) or (
            #     "ur5_2" in collision_model.geometryObjects[id_1].name
            #     and "ur5_2" in collision_model.geometryObjects[id_2].name
            #     and abs(id_1 - id_2) < 4
            # ):
            if (
                (
                    "base_link" in collision_model.geometryObjects[id_1].name
                    or id_1 + 1 == id_2
                )
                or (
                    collision_model.geometryObjects[id_1].parentJoint
                    == collision_model.geometryObjects[id_2].parentJoint
                )
                or (
                    collision_model.geometryObjects[id_1].parentJoint
                    == model.parents[collision_model.geometryObjects[id_2].parentJoint]
                )
            ):
                print(
                    "removing",
                    id_1,
                    id_2,
                    collision_model.geometryObjects[id_1].name,
                    collision_model.geometryObjects[id_2].name,
                    model.names[collision_model.geometryObjects[id_1].parentJoint],
                    model.names[
                        model.parents[collision_model.geometryObjects[id_2].parentJoint]
                    ],
                )
                collision_model.removeCollisionPair(pin.CollisionPair(id_1, id_2))

    model.lowerPositionLimit[0] = -3.14
    model.upperPositionLimit[0] = 3.14

    model.lowerPositionLimit[1] = -3.14
    model.upperPositionLimit[1] = 0

    model.lowerPositionLimit[6] = -3.14
    model.upperPositionLimit[6] = 3.14

    model.lowerPositionLimit[7] = -3.14
    model.upperPositionLimit[7] = 0

    return model, collision_model, visual_model


class pin_random_dual_ur5_env(SequenceMixin, PinocchioEnvironment):
    def sample_random_valid_state(self):
        while True:
            q = np.random.uniform(low=self.limits[0, :], high=self.limits[1, :])
            q = NpConfiguration.from_list([q[:6], q[6:]])
            if self.is_collision_free(q, None):
                return q

    def __init__(self):
        model, collision_model, visual_model = make_dual_ur5_waypoint_env()
        PinocchioEnvironment.__init__(self, model, collision_model, visual_model)

        print(self.limits)

        q_rnd_start = self.sample_random_valid_state()
        self.start_pos = q_rnd_start

        self.robots = ["a1", "a2"]

        self.robot_idx = {"a1": [i for i in range(6)], "a2": [i + 6 for i in range(6)]}
        self.robot_dims = {"a1": 6, "a2": 6}

        self.tasks = []
        self.sequence = []

        q_inter = self.sample_random_valid_state()
        q_goal = self.sample_random_valid_state()

        print(q_rnd_start.state())
        print(q_inter.state())
        print(q_goal.state())

        self.tasks = [
            # r1
            Task(
                ["a1"],
                SingleGoal(q_inter[0]),
            ),
            # r2
            Task(
                ["a2"],
                SingleGoal(q_inter[1]),
            ),
            # terminal mode
            Task(
                ["a1", "a2"],
                SingleGoal(q_goal.state()),
            ),
        ]

        self.tasks[0].name = "a1_goal"
        self.tasks[1].name = "a2_goal"
        self.tasks[2].name = "terminal"

        self.sequence = self._make_sequence_from_names(
            ["a2_goal", "a1_goal", "terminal"]
        )

        # permute goals, but only the ones that ware waypoints, not the final configuration

        # append terminal task

        BaseModeLogic.__init__(self)

        self.collision_tolerance = 0.01

        q = np.array(
            [
                0.19570329,
                -1.32450056,
                1.46959289,
                -0.73982317,
                -0.24690964,
                1.87430268,
                -0.85429199,
                -1.37153235,
                1.39270263,
                -1.65086109,
                -1.59343625,
                0.62897099,
            ]
        )

        q = NpConfiguration.from_list([q[:6], q[6:]])
        self.show_config(q)

        # for j in range(6):
        #     low = model.lowerPositionLimit
        #     up = model.upperPositionLimit

        #     for i in range(100):
        #         q = np.zeros(6)
        #         q[j] = low[j] + (up[j] - low[j]) / 100 * i
        #         q = NpConfiguration.from_list([q[:6], np.zeros(6)])
        #         self.show_config(q)


def make_dual_ur5_reorientation_env():
    urdf_path = "./src/multi_robot_multi_goal_planning/problems/urdfs/ur10e/ur10_spherized.urdf"
    # urdf_path = "./src/multi_robot_multi_goal_planning/problems/urdfs/ur10e/ur10e_meshes.urdf"
    # urdf_path = "./src/multi_robot_multi_goal_planning/problems/urdfs/ur10e/ur10e_primitives.urdf"

    coll_mesh_dir = Path(urdf_path).resolve().parent / "ur_description/meshes/ur10e/collision/"
    visual_mesh_dir = Path(urdf_path).resolve().parent / "ur_description/meshes/ur10e/visual/"

    robot_1 = pin.buildModelFromUrdf(urdf_path)
    r1_coll = pin.buildGeomFromUrdf(robot_1, urdf_path, pin.GeometryType.COLLISION, coll_mesh_dir)
    r1_viz = pin.buildGeomFromUrdf(robot_1, urdf_path, pin.GeometryType.VISUAL, visual_mesh_dir)

    robot_2 = pin.buildModelFromUrdf(urdf_path)
    r2_coll = pin.buildGeomFromUrdf(robot_1, urdf_path, pin.GeometryType.COLLISION, coll_mesh_dir)
    r2_viz = pin.buildGeomFromUrdf(robot_1, urdf_path, pin.GeometryType.VISUAL, visual_mesh_dir)

    # robot_3, r3_coll, r3_viz = pin.buildModelsFromUrdf(urdf_path)
    # robot_4, r4_coll, r4_viz = pin.buildModelsFromUrdf(urdf_path)

    add_namespace_prefix_to_models(robot_1, r1_coll, r1_viz, "ur5_1")
    add_namespace_prefix_to_models(robot_2, r2_coll, r2_viz, "ur5_2")
    # add_namespace_prefix_to_models(robot_3, r3_coll, r3_viz, "ur5_3")
    # add_namespace_prefix_to_models(robot_4, r4_coll, r4_viz, "ur5_4")

    # Create a composite model
    world = pin.Model()
    geom_model = pin.GeometryModel()

    geom1_name = "table"
    shape1 = fcl.Box(4, 4, 0.01)
    geom1_obj = pin.GeometryObject(geom1_name, 0, pin.SE3.Identity(), shape1)
    geom1_obj.meshColor = np.ones(4)
    geom_model.addGeometryObject(geom1_obj)

    # rotation matrix around the z-axis
    theta = np.pi / 2  # Example: 45 degrees rotation around z-axis
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    joint_placement1 = pin.SE3(R, np.array([0.5, -0.5, 0.0]))  # Base of first UR5
    joint_placement2 = pin.SE3(R, np.array([-0.5, -0.5, 0.0]))  # Base of second UR5
    # joint_placement3 = pin.SE3(
    #     np.eye(3), np.array([0.0, 1.0, 0.01])
    # )  # Base of second UR5
    # joint_placement4 = pin.SE3(
    #     np.eye(3), np.array([1.0, 1.0, 0.01])
    # )  # Base of second UR5

    # Add UR5 models to the world model
    model, visual_model = pin.appendModel(
        world,
        robot_1,
        geom_model,
        r1_viz,
        0,
        joint_placement1,
    )
    model, collision_model = pin.appendModel(
        world,
        robot_1,
        geom_model,
        r1_coll,
        0,
        joint_placement1,
    )

    tmp = model.copy()

    model, visual_model = pin.appendModel(
        tmp,
        robot_2,
        visual_model,
        r2_viz,
        0,
        joint_placement2,
    )
    model, collision_model = pin.appendModel(
        tmp,
        robot_2,
        collision_model,
        r2_coll,
        0,
        joint_placement2,
    )

    # Read the file back and print the data

    def quaternion_to_rotation_matrix(q):
        """
        Convert a quaternion to a 3x3 rotation matrix.

        Parameters:
        q : array-like, shape (4,)
            Quaternion in form [w, x, y, z] where w is the real part
            and x, y, z are the imaginary parts.

        Returns:
        R : ndarray, shape (3, 3)
            Rotation matrix.
        """
        # Normalize the quaternion
        q = np.array(q)
        q = q / np.linalg.norm(q)

        w, x, y, z = q

        # Calculate rotation matrix elements
        R = np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                ],
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * x * w,
                ],
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ]
        )

        return R

    # add boxes
    transformation_z_180 = pin.SE3(
        np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    )

    with open("src/multi_robot_multi_goal_planning/problems/desc/box_poses.json", "r") as f:
        loaded_box_data = json.load(f)
        print("Loaded box data:")
        for box in loaded_box_data:
            print(f"Position: {box['position']}, Quaternion: {box['quaternion']}")

            geom1_name = box['name']
            shape1 = fcl.Box(0.1, 0.1, 0.1)
            placement = pin.SE3.Identity()
            placement.translation = np.array(box["position"])
            placement.translation[2] += 0.048
            placement.rotation = quaternion_to_rotation_matrix(box["quaternion"])
            geom1_obj = pin.GeometryObject(geom1_name, 0, transformation_z_180 * placement, shape1)
            color = np.random.rand(4)
            color[3] = 1
            geom1_obj.meshColor = color
            collision_model.addGeometryObject(geom1_obj)

    collision_model.addAllCollisionPairs()

    ids = range(len(collision_model.geometryObjects))
    for i, id_1 in enumerate(ids):
        for id_2 in ids[i + 1 :]:
            #  = geomModel.getGeometryId(geomModelB.geometryObjects[cp.second].name);
            # if (
            #     "ur5_1" in collision_model.geometryObjects[id_1].name
            #     and "ur5_1" in collision_model.geometryObjects[id_2].name
            #     and abs(id_1 - id_2) < 4
            # ) or (
            #     "ur5_2" in collision_model.geometryObjects[id_1].name
            #     and "ur5_2" in collision_model.geometryObjects[id_2].name
            #     and abs(id_1 - id_2) < 4
            # ):
            if (
                "box" in collision_model.geometryObjects[id_1].name
                or "box" in collision_model.geometryObjects[id_2].name
            ):
                continue
            if (
                (
                    "base_link" in collision_model.geometryObjects[id_1].name and ((
                    "ur5_1" in collision_model.geometryObjects[id_1].name
                    and "ur5_1" in collision_model.geometryObjects[id_2].name) or (
                    "ur5_2" in collision_model.geometryObjects[id_1].name
                    and "ur5_2" in collision_model.geometryObjects[id_2].name))
                )
                or (
                    collision_model.geometryObjects[id_1].parentJoint
                    == collision_model.geometryObjects[id_2].parentJoint
                )
                or (
                    collision_model.geometryObjects[id_1].parentJoint
                    == model.parents[collision_model.geometryObjects[id_2].parentJoint]
                )
            ):
                print(
                    "removing",
                    id_1,
                    id_2,
                    collision_model.geometryObjects[id_1].name,
                    collision_model.geometryObjects[id_2].name,
                    model.names[collision_model.geometryObjects[id_1].parentJoint],
                    model.names[
                        model.parents[collision_model.geometryObjects[id_2].parentJoint]
                    ],
                )
                collision_model.removeCollisionPair(pin.CollisionPair(id_1, id_2))

    model.lowerPositionLimit[0] = -3.14
    model.upperPositionLimit[0] = 3.14

    model.lowerPositionLimit[1] = -3.14
    model.upperPositionLimit[1] = 0

    model.lowerPositionLimit[6] = -3.14
    model.upperPositionLimit[6] = 3.14

    model.lowerPositionLimit[7] = -3.14
    model.upperPositionLimit[7] = 0

    return model, collision_model, visual_model


class pin_reorientation_dual_ur5_env(SequenceMixin, PinocchioEnvironment):
    def sample_random_valid_state(self):
        while True:
            q = np.random.uniform(low=self.limits[0, :], high=self.limits[1, :])
            q = NpConfiguration.from_list([q[:6], q[6:]])
            if self.is_collision_free(q, None):
                return q

    def __init__(self):
        model, collision_model, visual_model = make_dual_ur5_reorientation_env()
        PinocchioEnvironment.__init__(self, model, collision_model, visual_model)

        # self.show(blocking=True)

        print(self.limits)

        # q_rnd_start = self.sample_random_valid_state()
        # self.start_pos = q_rnd_start
        q = np.array([0, -2, 1.0, -1, -1.57, 1])

        self.start_pos = NpConfiguration.from_list([q, q])

        self.robots = ["a1_", "a2_"]

        self.robot_idx = {
            "a1_": [i for i in range(6)],
            "a2_": [i + 6 for i in range(6)],
        }
        self.robot_dims = {"a1_": 6, "a2_": 6}

        self.tasks = []
        self.sequence = []


        self.import_tasks("src/multi_robot_multi_goal_planning/problems/desc/box_reorientation_tasks.txt")
        self.sequence = self._make_sequence_from_names(
            [
                "a2_pick_box0_0",
                "a2_handover_box0_1",
                "a2_place_box0_2",
                "a1_pick_box1_3",
                "a1_handover_box1_4",
                "a1_place_box1_5",
                "a1_pick_box2_6",
                "a1_handover_box2_7",
                "a1_place_box2_8",
                "a2_pick_box3_9",
                "a2_handover_box3_10",
                "a2_place_box3_11",
                "a2_pick_box4_12",
                "a2_handover_box4_13",
                "a2_place_box4_14",
                "a2_pick_box5_15",
                "a2_place_box5_16",
                # "a1_pick_box6_17",
                # "a1_handover_box6_18",
                # "a1_place_box6_19",
                "a1_pick_box7_20",
                "a1_handover_box7_21",
                "a1_place_box7_22",
                "a2_pick_box8_23",
                "a2_handover_box8_24",
                "a2_place_box8_25",
                "terminal",
            ]
        )

        BaseModeLogic.__init__(self)

        self.collision_tolerance = 0.01

        self.manipulating_env = True

    def make_start_mode(self):
        start_mode = super().make_start_mode()
        start_mode.sg = self.initial_sg
        return start_mode
