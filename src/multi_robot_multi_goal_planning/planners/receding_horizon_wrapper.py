from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

from multi_robot_multi_goal_planning.planners.baseplanner import BasePlanner
from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseProblem,
    State,
)
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
    RuntimeTerminationCondition
)

@dataclass
class RecedingHorizonConfig:
    low_level_solver: str = "composite_prm"
    horizon_length: int = 3
    execution_length: int = 1
    low_level_max_time: float = 50

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

    def construct_planner(env: BaseProblem) -> BasePlanner:
        planner = None
        return planner
    
    def make_short_horizon_env(self, start_task) -> BaseProblem:
        short_horizon_env = None
        return short_horizon_env

    def __init__(self, env: BaseProblem, config: RecedingHorizonConfig | None = None):
        self.base_env = env
        self.config = config if config is not None else RecedingHorizonConfig()

    def plan(
        self, planner_termination_criterion: PlannerTerminationCondition
    ) -> Tuple[List[State] | None, Dict[str, Any]]:
        """
        Plans a geometric path/trajectory for the specified problem. This function assumes that the problem is solvable by the planner.
        """

        # ensure that this is a sequence env

        complete_plan = []

        current_task_index = 0

        while True:
            # make short horizon env
            sh_env = self.make_short_horizon_env(current_task_index)
            short_horizon_planner = self.construct_planner(sh_env)

            inner_ptc = RuntimeTerminationCondition()
            short_horizon_plan, _ = short_horizon_planner.plan(inner_ptc)
            
            if short_horizon_plan is None:
                return None, {}

            complete_plan.extend(short_horizon_plan)

            if self.base_env.is_terminal_mode(complete_plan[-1].mode):
                break
            
            current_task_index += self.config.execution_length

        info = {}
        return complete_plan, info