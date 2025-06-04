from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Set, List, Dict, Any, Tuple


from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
)
from multi_robot_multi_goal_planning.problems.planning_env import (
    AgentType,
    BaseProblem,
    State,
    ConstraintType,
    ManipulationType,
    ProblemSpec,
)


@dataclass(frozen=True)
class SolverCapabilities:
    supports_agent_types: Set[AgentType]
    supports_constraints: Set[ConstraintType]
    supports_manipulation: Set[ManipulationType]

    def __repr__(self):
        return (
            f"SolverCapabilities(Agents: {[a.value for a in self.supports_agent_types]}, "
            f"Constraints: {[c.value for c in self.supports_constraints]}, "
            f"Env: {[e.value for e in self.supports_manipulation]}, "
        )


class BasePlanner(ABC):
    solver_capabilities: SolverCapabilities

    def __init__(self, env: BaseProblem):
        pass

    @abstractmethod
    def plan(
        self, planner_termination_criterion: PlannerTerminationCondition
    ) -> Tuple[List[State] | None, Dict[str, Any]]:
        pass

    def can_solve(self, problem_spec: ProblemSpec) -> bool:
        problem_spec = problem_spec
        caps = self.solver_capabilities

        # All conditions must be met
        return (
            problem_spec.agent_type in caps.supports_agent_types
            and problem_spec.constraints in caps.supports_constraints
            and problem_spec.manipulation in caps.supports_manipulation
        )
