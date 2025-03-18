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
from pinocchio.visualize import MeshcatVisualizer

import time
import sys


class PinocchioEnvironment(BaseProblem):
    # misc
    collision_tolerance: float
    collision_resolution: float

    def __init__(self, model, collision_model, visual_model):
        self.limits = None
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

        for i, id_1 in enumerate(ids):
            obj_name = collision_model.geometryObjects[id_1].name
            if obj_name[:3] == "obj":
                parent = collision_model.geometryObjects[id_1].parentJoint
                placement = collision_model.geometryObjects[id_1].placement
                self.initial_sg[id_1] = (
                    "table_0",
                    parent,
                    np.round(placement, 3).tobytes(),
                    pin.SE3(placement),
                )

        self.current_scenegraph = self.initial_sg.copy()

        n = len(self.collision_model.geometryObjects)
        mat = np.zeros((n, n)) - self.collision_tolerance

        self.geom_data.setSecurityMargins(self.collision_model, mat)

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
    def _set_to_scenegraph(self, sg, update_visual: bool = False):
        # update object positions
        # placement = pin.SE3.Identity()
        for frame_id, (parent, parent_joint, pose, placement) in sg.items():
            # placement = pin.SE3(np.frombuffer(pose).reshape(4, 4))

            if (
                frame_id in self.current_scenegraph
                and parent == self.current_scenegraph[frame_id][0]
                and parent == "table_0"
                and self.current_scenegraph[frame_id][2] == pose
            ):
                # print("A")
                continue

            frame_pose = self.data.oMi[parent_joint].act(placement)
            self.collision_model.geometryObjects[frame_id].placement = frame_pose

            if update_visual:
                frame_name = self.collision_model.geometryObjects[frame_id].name
                vis_frame_id = self.visual_model.getGeometryId(frame_name)
                self.visual_model.geometryObjects[vis_frame_id].placement = frame_pose

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
                    * self.collision_model.geometryObjects[new_parent_id].placement
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
        self._set_to_scenegraph(mode.sg)

        pin.updateGeometryPlacements(
            self.model, self.data, self.collision_model, self.geom_data
        )

        # self.viz.display(q)
        # print(mode)
        # input("AA")

        # self.viz.updatePlacements(pin.GeometryType.COLLISION)
        # self.viz.updatePlacements(pin.GeometryType.VISUAL)

        # for i in range(len(self.collision_model.collisionPairs)):
        #     in_collision = pin.computeCollision(self.collision_model, self.geom_data, i)
        #     if in_collision:
        #         # cr = self.geom_data.collisionResults[i]
        #         # cp = self.collision_model.collisionPairs[i]
        #         # if cr.isCollision():
        #         #     print(self.collision_model.geometryObjects[cp.first].name, self.collision_model.geometryObjects[cp.second].name)
        #         #     print("collision pair:", cp.first,",",cp.second,"- collision:","Yes" if cr.isCollision() else "No")
        #         # print(q)
        #         # print('colliding')
        #         # input("A")

        #         return False

        # return True

        in_collision = pin.computeCollisions(self.collision_model, self.geom_data, True)

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
            # # self.show_config(q_orig, blocking=True)

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


def make_pinocchio_random_env(dim=2):
    num_agents = 2

    joint_limits = np.ones((2, num_agents * dim)) * 2
    joint_limits[0, :] = -2

    print(joint_limits)

    start_poses = np.zeros(num_agents * dim)
    start_poses[0] = -0.8
    start_poses[dim] = 0.8

    return None


def make_pin_middle_obstacle_two_dim_env():
    filename = "./src/multi_robot_multi_goal_planning/problems/urdfs/middle_obstacle_two_agents.urdf"
    model, collision_model, visual_model = pin.buildModelsFromUrdf(filename)

    collision_model.addAllCollisionPairs()

    return model, collision_model, visual_model


class pinocchio_middle_obs(SequenceMixin, PinocchioEnvironment):
    def __init__(self, agents_can_rotate=True):
        model, collision_model, visual_model = make_pin_middle_obstacle_two_dim_env()

        PinocchioEnvironment.__init__(self, model, collision_model, visual_model)

        self.limits = np.ones((2, 2 * 3)) * 1
        self.limits[0, :] = -1

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

    env_names = ["wall1", "wall2", "wall3", "wall4", "obs3", "obs4"]

    for i, id_1 in enumerate(ids):
        for id_2 in ids[i + 1 :]:
            #  = geomModel.getGeometryId(geomModelB.geometryObjects[cp.second].name);
            if (
                collision_model.geometryObjects[id_1].name in env_names
                and collision_model.geometryObjects[id_2].name in env_names
            ):
                continue

            collision_model.addCollisionPair(pin.CollisionPair(id_1, id_2))

    # collision_model.addAllCollisionPairs()

    return model, collision_model, visual_model


class pinocchio_other_hallway(SequenceMixin, PinocchioEnvironment):
    def __init__(self):
        model, collision_model, visual_model = make_pin_other_hallway_two_dim_env()

        PinocchioEnvironment.__init__(self, model, collision_model, visual_model)

        self.limits = np.ones((2, 2 * 3)) * 2
        self.limits[0, :] = -2
        self.limits[0, 2] = -3.1415
        self.limits[1, 2] = 3.1415
        self.limits[0, 5] = -3.1415
        self.limits[1, 5] = 3.1415

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
        "",
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

        self.manipulating_env = True

        self.limits = np.ones((2, 2 * 3)) * 2
        self.limits[0, :] = -2
        self.limits[0, 2] = -3.1415
        self.limits[1, 2] = 3.1415
        self.limits[0, 5] = -3.1415
        self.limits[1, 5] = 3.1415

        self.start_pos = NpConfiguration.from_list([[-0.5, 0.8, 0], [0.0, -0.5, 0]])

        self.robot_idx = {"a1": [0, 1, 2], "a2": [3, 4, 5]}
        self.robot_dims = {"a1": 3, "a2": 3}
        # self.C.view(True)

        self.robots = ["a1", "a2"]

        task_1_pose = np.array([-0.00353716, 0.76632651, 0.58110193])
        # task_1_pose = np.array([-0.00353716, 0.76632651, 0.])
        handover_pose = np.array(
            [
                0.98929765,
                -0.89385425,
                1.02667076,
                1.49362046 + 0.01,
                -1.02441501 + 0.01,
                1.42582986,
            ]
        )
        task_2_pose = np.array([1.19521468, 0.41126415, 0.98020169])
        task_3_pose = np.array([0.50360456, -1.10537035, 0.97861029])
        task_4_pose = np.array([0.80358028, 1.54452128, 0.97864346])
        task_5_pose = np.array(
            [
                -4.99289680e-01,
                7.98808085e-01,
                7.11081401e-01,
                -1.40245030e-04,
                -4.99267552e-01,
                -2.93781443e00,
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
                ["a1", "a2"],
                GoalSet([handover_pose]),
                # SingleGoal(keyframes[1]),
                type="handover",
                frames=["a2_0", "obj1_0"],
            ),
            Task(
                ["a2"],
                SingleGoal(task_2_pose),
                type="place",
                frames=["table_0", "obj1_0"],
            ),
            Task(
                ["a1"],
                SingleGoal(task_3_pose),
                type="pick",
                frames=["a1_0", "obj2_0"],
            ),
            Task(
                ["a1"],
                SingleGoal(task_4_pose),
                type="place",
                frames=["table_0", "obj2_0"],
            ),
            # terminal
            # Task(["a1", "a2"], SingleGoal(keyframes[3])),
            Task(["a1", "a2"], GoalSet([task_5_pose])),
        ]

        self.tasks[0].name = "a1_pick_obj1"
        self.tasks[1].name = "handover"
        self.tasks[2].name = "a2_place"
        self.tasks[3].name = "a1_pick_obj2"
        self.tasks[4].name = "a1_place_obj2"
        self.tasks[5].name = "terminal"

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
        "",
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

        self.limits = np.ones((2, 2 * 3)) * 1
        self.limits[0, :] = -1
        self.limits[0, 2] = -3.1415
        self.limits[1, 2] = 3.1415
        self.limits[0, 5] = -3.1415
        self.limits[1, 5] = 3.1415

        self.start_pos = NpConfiguration.from_list([[-0.5, 0.8, 0], [0.0, -0.5, 0]])

        self.robot_idx = {"a1": [0, 1, 2], "a2": [3, 4, 5]}
        self.robot_dims = {"a1": 3, "a2": 3}
        # self.C.view(True)

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
