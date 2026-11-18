"""Sphere-based Panda welding, authored in RAI and converted without approximation.

The .g assets define the robot, reviewed tool and finite obstacle boxes. JSON
files supply fixed per-robot task queues and link exclusions. Four arms use the
10-weld pilot; eight arms use 40 welds. These are distinct from RoboBallet's
source-mesh presets. Geometry edits require revalidating the saved goals.
Load through this factory to apply the exclusions as well as the .g geometry.

    from multi_robot_multi_goal_planning.problems import get_env_by_name
    from multi_robot_multi_goal_planning.problems.rai_mujoco_env import to_mujoco
    from multi_robot_multi_goal_planning.problems.comotion_env import to_comotion

    env = get_env_by_name("rai.panda_welding_4")
    env.show()
    mujoco_env = to_mujoco(env)
    comotion_env = to_comotion(env)

Both conversions preserve the source geometry, goals and excluded shape pairs.
No MuJoCo/CoMotion installation is required to build or view the RAI source.
"""
from itertools import product
from pathlib import Path
import json

import numpy as np
import robotic as ry

from ..core.dependency_graph import DependencyGraph
from ..core.goals import SingleGoal, GoalSet
from ..core.registry import register
from ..planning_env import BaseModeLogic, DependencyGraphMixin, DependencyType, ManipulationType, Task
from ..rai_base_env import rai_env

ASSETS = Path(__file__).resolve().parents[2] / "assets/models/rai/panda_welding"


@register([
    ("rai.panda_welding_4", {"robots": 4}),
    ("rai.panda_welding_8", {"robots": 8}),
])
class PandaWeldingEnvironment(DependencyGraphMixin, rai_env):
    def __init__(self, robots=4):
        if robots not in (4, 8):
            raise ValueError("Panda welding provides four- and eight-arm scenes")
        description = json.loads((ASSETS / f"welding_{robots}.json").read_text())
        self.robots = [f"panda{i+1}_" for i in range(robots)]
        self.C = ry.Config()
        self.C.addFile(str(ASSETS / f"welding_{robots}.g"))
        rai_env.__init__(self)
        self.collision_resolution = description["collision_resolution"]
        self.collision_tolerance = 0.0
        self.spec.manipulation = ManipulationType.STATIC
        self.spec.dependency = DependencyType.UNORDERED
        self.tasks = []
        self.graph = DependencyGraph()
        last = {}
        for saved in description["tasks"]:
            robot = self.robots[saved["robot"]]
            values = saved.get("q_candidates", [saved["q"]])
            q = np.asarray(values, dtype=float)
            indices = self.robot_idx[robot]
            if (q.ndim != 2 or q.shape[1] != len(indices) or not np.isfinite(q).all()
                    or np.any(q < self.limits[0, indices] - 1e-9)
                    or np.any(q > self.limits[1, indices] + 1e-9)):
                raise ValueError(f"Invalid saved goal for {saved['name']}")
            goal = GoalSet(list(q)) if len(q) > 1 else SingleGoal(q[0])
            self.tasks.append(Task(saved["name"], [robot], goal))
            self.graph.add_node(saved["name"])
            if robot in last:
                self.graph.add_dependency(saved["name"], last[robot])
            last[robot] = saved["name"]
        self.tasks.append(Task("terminal", self.robots, SingleGoal(self.start_pos.state().copy())))
        self.graph.add_node("terminal")
        for name in last.values():
            self.graph.add_dependency("terminal", name)
        self.collision_exclusions = self._collision_exclusions(description)
        BaseModeLogic.__init__(self)

    def _collision_exclusions(self, description):
        """Expand reviewed link exclusions using the shapes in the editable scene."""
        link_pairs = json.loads((ASSETS / "excluded_links.json").read_text())
        links = {n for pair in link_pairs for n in pair}
        shapes = {r + link: [] for r in self.robots for link in links}
        for frame in self.C.getFrames():
            if not frame.info().get("contact", 0):
                continue
            ancestor = frame
            while ancestor is not None and ancestor.name not in shapes:
                ancestor = ancestor.getParent()
            if ancestor is not None:
                shapes[ancestor.name].append(frame.name)
        excluded = set()
        for robot in self.robots:
            for a, b in link_pairs:
                if not shapes[robot+a] or not shapes[robot+b]:
                    raise ValueError(f"Excluded link has no collision shapes: {robot}{a}, {b}")
                excluded.update(frozenset(pair) for pair in product(shapes[robot+a], shapes[robot+b]))
        for obstacle, ignored in description["obstacle_excluded_links"].items():
            if not self.C.getFrame(obstacle).info().get("contact", 0):
                raise ValueError(f"Excluded obstacle has no collision shape: {obstacle}")
            for robot, link in product(self.robots, ignored):
                excluded.update(frozenset((obstacle, s)) for s in shapes[robot+link])
        return frozenset(excluded)

