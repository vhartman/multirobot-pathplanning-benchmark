from multi_robot_multi_goal_planning.assets import ASSET_ROOT
import numpy as np
import time
import sys
from numba import jit, float64  # ty: ignore[possibly-unbound-import]
import json
from typing import List, Optional
from numpy.typing import NDArray
from ...planning_env import generate_binary_search_indices
from ...core.configuration import Configuration, NpConfiguration, config_dist
from ...planning_env import (
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
from ...core.goals import SingleGoal, GoalSet, GoalRegion, ConditionalGoal
import pinocchio as pin
import hppfcl as fcl
from pathlib import Path
from pinocchio.visualize import MeshcatVisualizer
from ...core.registry import register

from multi_robot_multi_goal_planning.problems.pinocchio.base import (
    PinocchioEnvironment,
    multiply_4x4_matrices,
)

def make_pin_middle_obstacle_two_dim_env():
    filename = str(ASSET_ROOT / 'models/pinocchio/middle_obstacle_two_agents.urdf')
    model, collision_model, visual_model = pin.buildModelsFromUrdf(filename)

    collision_model.addAllCollisionPairs()

    return model, collision_model, visual_model


@register("pinocchio.middle_obs")
class pinocchio_middle_obs(SequenceMixin, PinocchioEnvironment):
    def __init__(self, agents_can_rotate=True):
        model, collision_model, visual_model = make_pin_middle_obstacle_two_dim_env()

        start_pos = NpConfiguration.from_list([[0, -0.8, 0], [0, 0.8, 0]])
        robots = ["a1", "a2"]

        PinocchioEnvironment.__init__(
            self, model, collision_model, visual_model, start_pos, robots
        )

        self.tasks = [
            # r1
            Task("a1_goal", ["a1"], SingleGoal(np.array([0, 0.8, 0]))),
            # r2
            Task("a2_goal", ["a2"], SingleGoal(np.array([0, -0.8, 0]))),
            # terminal mode
            Task(
                "terminal",
                ["a1", "a2"],
                SingleGoal(np.array([0, -0.8, 0, 0, 0.8, 0])),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a2_goal", "a1_goal", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        # self.collision_resolution = 0.01
        # self.collision_tolerance = 0.01


def make_pin_other_hallway_two_dim_env():
    filename = str(ASSET_ROOT / 'models/pinocchio/other_hallway.urdf')
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


@register("pinocchio.other_hallway")
class pinocchio_other_hallway(SequenceMixin, PinocchioEnvironment):
    def __init__(self):
        model, collision_model, visual_model = make_pin_other_hallway_two_dim_env()

        root_name = "table_0"

        start_pos = NpConfiguration.from_list([[1.5, 0.0, 0], [-1.5, 0.0, 0]])
        robots = ["a1", "a2"]

        PinocchioEnvironment.__init__(
            self, model, collision_model, visual_model, start_pos, robots, root_name
        )

        # self.C.view(True)
        self.tasks = [
            # r1
            Task("a1_goal", ["a1"], SingleGoal(np.array([-1.5, 1, np.pi / 2]))),
            # r2
            Task("a2_goal", ["a2"], SingleGoal(np.array([1.5, 1, 0]))),
            # terminal mode
            Task(
                "terminal",
                ["a1", "a2"],
                SingleGoal(np.array([1.5, 0.0, 0, -1.5, 0.0, 0])),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a2_goal", "a1_goal", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        self.collision_tolerance = 0.01
        self.collision_resolution = 0.02

        self.spec.home_pose = SafePoseType.HAS_SAFE_HOME_POSE
        self.spec.manipulation = ManipulationType.STATIC

        self.safe_pose = {
            "a1": np.array([-1.5, 1, np.pi / 2]),
            "a2": np.array([1.5, 1, 0]),
        }


def make_2d_handover():
    filename = str(ASSET_ROOT / 'models/pinocchio/2d_handover.urdf')
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


@register("pinocchio.2d_handover")
class pinocchio_handover_two_dim(SequenceMixin, PinocchioEnvironment):
    def __init__(self):
        model, collision_model, visual_model = make_2d_handover()

        start_pos = NpConfiguration.from_list([[-0.5, 0.8, 0], [0.0, -0.5, 0]])

        robots = ["a1", "a2"]
        root_name = "table_0"

        self.collision_tolerance = 0.01

        PinocchioEnvironment.__init__(
            self, model, collision_model, visual_model, start_pos, robots, root_name
        )

        self.manipulating_env = True

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

        # self.start_mode.sg = self.initial_sg

    # def make_start_mode(self):
    #     start_mode = super().make_start_mode()
    #     start_mode.sg = self.initial_sg
    #     return start_mode


def make_piano():
    filename = str(ASSET_ROOT / 'models/pinocchio/piano.urdf')
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


@register("pinocchio.piano")
class pinocchio_piano_two_dim(SequenceMixin, PinocchioEnvironment):
    def __init__(self):
        model, collision_model, visual_model = make_piano()

        start_pos = NpConfiguration.from_list([[-0.5, 0.8, 0], [0.0, -0.5, 0]])
        root_name = "table_0"
        robots = ["a1", "a2"]

        self.collision_tolerance = 0.01

        PinocchioEnvironment.__init__(
            self, model, collision_model, visual_model, start_pos, robots, root_name
        )

        self.manipulating_env = True

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
                "a1_pick",
                ["a1"],
                SingleGoal(task_1_pose),
                type="pick",
                frames=["a1_0", "obj1_0"],
            ),
            Task(
                "a1_place",
                ["a1"],
                SingleGoal(task_2_pose),
                type="place",
                frames=["table_0", "obj1_0"],
            ),
            # a2
            Task(
                "a2_pick",
                ["a2"],
                SingleGoal(task_3_pose),
                type="pick",
                frames=["a2_0", "obj2_0"],
            ),
            Task(
                "a2_place",
                ["a2"],
                SingleGoal(task_4_pose),
                type="place",
                frames=["table_0", "obj2_0"],
            ),
            # terminal
            Task(
                "terminal",
                ["a1", "a2"],
                SingleGoal(task_5_pose),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a2_pick", "a1_pick", "a2_place", "a1_place", "terminal"]
        )

        # AbstractEnvironment.__init__(self, 2, env.start_pos, env.limits)
        BaseModeLogic.__init__(self)

        # self.start_mode.sg = self.initial_sg

    # def make_start_mode(self):
    #     start_mode = super().make_start_mode()
    #     start_mode.sg = self.initial_sg
    #     return start_mode


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
    # urdf_path = "./src/multi_robot_multi_goal_planning/assets/models/pinocchio/ur5e/ur5e.urdf"
    # urdf_path = "./src/multi_robot_multi_goal_planning/assets/models/pinocchio/ur5e/ur5e_constrained_coll_primitives.urdf"
    urdf_path = str(ASSET_ROOT / 'models/pinocchio/ur5e/ur5e_constrained.urdf')

    coll_mesh_dir = (
        Path(urdf_path).resolve().parent / "ur_description/meshes/ur5e/collision/"
    )
    visual_mesh_dir = (
        Path(urdf_path).resolve().parent / "ur_description/meshes/ur5e/visual/"
    )

    # mesh_dir = "../src/multi_robot_multi_goal_planning/assets/models/pinocchio/ur10e/ur_description/meshes/ur10e/visual/"

    robot_1 = pin.buildModelFromUrdf(urdf_path)
    r1_coll = pin.buildGeomFromUrdf(
        robot_1, urdf_path, pin.GeometryType.COLLISION, coll_mesh_dir
    )
    r1_viz = pin.buildGeomFromUrdf(
        robot_1, urdf_path, pin.GeometryType.VISUAL, visual_mesh_dir
    )

    robot_2 = pin.buildModelFromUrdf(urdf_path)
    r2_coll = pin.buildGeomFromUrdf(
        robot_1, urdf_path, pin.GeometryType.COLLISION, coll_mesh_dir
    )
    r2_viz = pin.buildGeomFromUrdf(
        robot_1, urdf_path, pin.GeometryType.VISUAL, visual_mesh_dir
    )

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
                # print(
                #     "removing",
                #     id_1,
                #     id_2,
                #     collision_model.geometryObjects[id_1].name,
                #     collision_model.geometryObjects[id_2].name,
                #     model.names[collision_model.geometryObjects[id_1].parentJoint],
                #     model.names[
                #         model.parents[collision_model.geometryObjects[id_2].parentJoint]
                #     ],
                # )
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


@register("pinocchio.random_ur5")
class pin_random_dual_ur5_env(SequenceMixin, PinocchioEnvironment):
    def sample_random_valid_state(self):
        while True:
            q = np.random.uniform(low=self.limits[0, :], high=self.limits[1, :])
            q = NpConfiguration.from_list([q[:6], q[6:]])
            if self.is_collision_free(q, None):
                return q

    def __init__(self):
        model, collision_model, visual_model = make_dual_ur5_waypoint_env()

        # dummy start_pose to set the env up
        q = NpConfiguration.from_list([np.zeros(6), np.zeros(6)])
        robots = ["a1", "a2"]

        PinocchioEnvironment.__init__(
            self, model, collision_model, visual_model, q, robots, "table"
        )

        q_rnd_start = self.sample_random_valid_state()
        self.start_pos = q_rnd_start

        q_inter = self.sample_random_valid_state()
        q_goal = self.sample_random_valid_state()

        # print(q_rnd_start.state())
        # print(q_inter.state())
        # print(q_goal.state())

        self.tasks = [
            # r1
            Task(
                "a1_goal",
                ["a1"],
                SingleGoal(q_inter[0]),
            ),
            # r2
            Task(
                "a2_goal",
                ["a2"],
                SingleGoal(q_inter[1]),
            ),
            # terminal mode
            Task(
                "terminal",
                ["a1", "a2"],
                SingleGoal(q_goal.state()),
            ),
        ]

        self.sequence = self._make_sequence_from_names(
            ["a2_goal", "a1_goal", "terminal"]
        )

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

        # q = NpConfiguration.from_list([q[:6], q[6:]])
        # self.show_config(q)

        # for j in range(6):
        #     low = model.lowerPositionLimit
        #     up = model.upperPositionLimit

        #     for i in range(100):
        #         q = np.zeros(6)
        #         q[j] = low[j] + (up[j] - low[j]) / 100 * i
        #         q = NpConfiguration.from_list([q[:6], np.zeros(6)])
        #         self.show_config(q)


def make_dual_ur5_reorientation_env():
    urdf_path = (
        str(ASSET_ROOT / 'models/pinocchio/ur10e/ur10_spherized.urdf')
    )
    # urdf_path = "./src/multi_robot_multi_goal_planning/assets/models/pinocchio/ur10e/ur10e_meshes.urdf"
    # urdf_path = "./src/multi_robot_multi_goal_planning/assets/models/pinocchio/ur10e/ur10e_primitives.urdf"

    coll_mesh_dir = (
        Path(urdf_path).resolve().parent / "ur_description/meshes/ur10e/collision/"
    )
    visual_mesh_dir = (
        Path(urdf_path).resolve().parent / "ur_description/meshes/ur10e/visual/"
    )

    robot_1 = pin.buildModelFromUrdf(urdf_path)
    r1_coll = pin.buildGeomFromUrdf(
        robot_1, urdf_path, pin.GeometryType.COLLISION, coll_mesh_dir
    )
    r1_viz = pin.buildGeomFromUrdf(
        robot_1, urdf_path, pin.GeometryType.VISUAL, visual_mesh_dir
    )

    robot_2 = pin.buildModelFromUrdf(urdf_path)
    r2_coll = pin.buildGeomFromUrdf(
        robot_1, urdf_path, pin.GeometryType.COLLISION, coll_mesh_dir
    )
    r2_viz = pin.buildGeomFromUrdf(
        robot_1, urdf_path, pin.GeometryType.VISUAL, visual_mesh_dir
    )

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

    with open(
        str(ASSET_ROOT / 'desc/box_poses.json'), "r"
    ) as f:
        loaded_box_data = json.load(f)
        # print("Loaded box data:")
        for box in loaded_box_data:
            # print(f"Position: {box['position']}, Quaternion: {box['quaternion']}")

            geom1_name = box["name"]
            shape1 = fcl.Box(0.1, 0.1, 0.1)
            placement = pin.SE3.Identity()
            placement.translation = np.array(box["position"])
            placement.translation[2] += 0.048
            placement.rotation = quaternion_to_rotation_matrix(box["quaternion"])
            geom1_obj = pin.GeometryObject(
                geom1_name, 0, transformation_z_180 * placement, shape1
            )
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
                    "base_link" in collision_model.geometryObjects[id_1].name
                    and (
                        (
                            "ur5_1" in collision_model.geometryObjects[id_1].name
                            and "ur5_1" in collision_model.geometryObjects[id_2].name
                        )
                        or (
                            "ur5_2" in collision_model.geometryObjects[id_1].name
                            and "ur5_2" in collision_model.geometryObjects[id_2].name
                        )
                    )
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
                # TODO: should probably use an sdf here
                # print(
                #     "removing",
                #     id_1,
                #     id_2,
                #     collision_model.geometryObjects[id_1].name,
                #     collision_model.geometryObjects[id_2].name,
                #     model.names[collision_model.geometryObjects[id_1].parentJoint],
                #     model.names[
                #         model.parents[collision_model.geometryObjects[id_2].parentJoint]
                #     ],
                # )
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


@register("pinocchio.reorientation")
class pin_reorientation_dual_ur5_env(SequenceMixin, PinocchioEnvironment):
    def sample_random_valid_state(self):
        while True:
            q = np.random.uniform(low=self.limits[0, :], high=self.limits[1, :])
            q = NpConfiguration.from_list([q[:6], q[6:]])
            if self.is_collision_free(q, None):
                return q

    def __init__(self):
        model, collision_model, visual_model = make_dual_ur5_reorientation_env()

        q = np.array([0, -2, 1.0, -1, -1.57, 1])
        start_pos = NpConfiguration.from_list([q, q])
        robots = ["a1_", "a2_"]

        PinocchioEnvironment.__init__(
            self, model, collision_model, visual_model, start_pos, robots
        )

        self.import_tasks(
            str(ASSET_ROOT / 'desc/box_reorientation_tasks.txt')
        )
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

        self.manipulating_env = True

        BaseModeLogic.__init__(self)

        self.collision_tolerance = 0.01
        # self.collision_resolution = 0.05

    # def make_start_mode(self):
    #     start_mode = super().make_start_mode()
    #     start_mode.sg = self.initial_sg
    #     return start_mode
