from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
import copy

from multi_robot_multi_goal_planning.planners.baseplanner import BasePlanner
from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseProblem,
    State,
    SequenceMixin,
    Task,
    SingleGoal,
    GoalRegion,
    GoalSet,
    Mode,
    Configuration,
)
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
    RuntimeTerminationCondition,
)

from multi_robot_multi_goal_planning.planners.composite_prm_planner import (
    CompositePRM,
    CompositePRMConfig,
)


@dataclass
class RecedingHorizonConfig:
    low_level_solver: str = "composite_prm"
    horizon_length: int = 1
    execution_length: int = 1
    low_level_max_time: float = 5
    constrain_free_robots_to_home: bool = False


class RecedingHorizonPlanner(BasePlanner):
    """
    Receding horizon planner wrapper.
    Mostly for tests how suboptimal a solution is if we only consider parts of the whole plan.

    Takes a planner, and only runs it on a subsequence of the whole planning sequence.
    Takes the solution to the subsequence as fixed, and continues on from there.
    To not plan to a nonsensical solution, we always ensure that all agents have > N goals.
    This can imply that some agents might have more than N goals.
    But we need to deal with this somehow. Maybe N = 1 means actually only planning for one goal, all other agents keep home pose?
    We need to decide how long we plan for on a given subsequence.
    Same termination criterion as usually + convergence test + run shortcutter after.

    So, approach: N gives the horizon/number of tasks we consider.
    we can do receding horizon by moving horizon only 1 or N.
    """

    def construct_planner(self, env: BaseProblem) -> BasePlanner:
        planner = CompositePRM(env)
        return planner

    def make_short_horizon_env(
        self, seq_start_idx, start_mode: Mode, start_pos: Configuration
    ) -> BaseProblem:
        short_horizon_env = copy.deepcopy(self.base_env)

        final_task_index = min(
            len(short_horizon_env.sequence), seq_start_idx + self.config.horizon_length
        )
        short_horizon_env.sequence = short_horizon_env.sequence[
            seq_start_idx:final_task_index
        ]
        short_horizon_env.start_mode = start_mode
        short_horizon_env.start_pos = start_pos

        if final_task_index < len(short_horizon_env.sequence):
            if self.config.constrain_free_robots_to_home:
                goal_pose = self.base_env.start_pos.state()

                # get active robot, set it to the correct pose
                task_idx = self.base_env.sequence[final_task_index]
                final_task = self.base_env.tasks[task_idx]
                final_task_goal = final_task.goal

                # this could be adapted to a goal_set
                final_task_pose = final_task_goal.sample(None)

                constrained_robots = final_task.robots

                offset = 0
                for r in self.base_env.robots:
                    if r in constrained_robots:
                        dim = self.base_env.robot_dims[r]
                        goal_pose[offset : offset + dim] = final_task_pose

                new_terminal_task = Task(self.base_env.robots, SingleGoal(goal_pose))
            else:
                goal_region = self.base_env.limits
                # get active robot, set it to the correct pose
                task_idx = self.base_env.sequence[final_task_index]
                final_task = self.base_env.tasks[task_idx]
                final_task_goal = final_task.goal

                # this could be adapted to a goal_set
                final_task_pose = final_task_goal.sample(None)

                constrained_robots = final_task.robots

                offset = 0
                for r in self.base_env.robots:
                    if r in constrained_robots:
                        dim = self.base_env.robot_dims[r]
                        goal_region[0, offset : offset + dim] = final_task_pose
                        goal_region[1, offset : offset + dim] = final_task_pose

                new_terminal_task = Task(self.base_env.robots, GoalRegion(goal_region))

            short_horizon_env.tasks.append(new_terminal_task)
            short_horizon_env.sequence[-1] = len(short_horizon_env.tasks)

        return short_horizon_env

    def __init__(self, env: BaseProblem, config: RecedingHorizonConfig | None = None):
        self.base_env = env
        self.config = config if config is not None else RecedingHorizonConfig()

    def plan(
        self, ptc: PlannerTerminationCondition, optimize: bool = False
    ) -> Tuple[List[State] | None, Dict[str, Any]]:
        """
        Plans a geometric path/trajectory for the specified problem. This function assumes that the problem is solvable by the planner.
        """

        assert self.config.execution_length <= self.config.horizon_length, (
            "Execution length cannot be longer than planing horizon."
        )

        # TODO: ensure that this is a sequence env
        if not isinstance(self.base_env, SequenceMixin):
            assert False, "Needs to be sequence task spec."

        complete_plan = []

        current_task_index = 0

        while True:
            # make short horizon env
            # get index in plan where we at
            if len(complete_plan) == 0:
                start_mode = self.base_env.start_mode
                start_pos = self.base_env.start_pos
            else:
                idx = 0
                for i in range(1, len(complete_plan)):
                    if complete_plan[i].mode != complete_plan[i - 1]:
                        idx += 1

                    if idx == current_task_index:
                        start_mode = None
                        start_pos = None

            sh_env = self.make_short_horizon_env(
                current_task_index, start_mode, start_pos
            )

            short_horizon_planner = self.construct_planner(sh_env)

            inner_ptc = RuntimeTerminationCondition(self.config.low_level_max_time)
            short_horizon_plan, _ = short_horizon_planner.plan(inner_ptc)

            if short_horizon_plan is None:
                return None, {}

            complete_plan.extend(short_horizon_plan)

            if self.base_env.is_terminal_mode(complete_plan[-1].mode):
                break

            current_task_index += self.config.execution_length

        info = {}
        return complete_plan, info
