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
    GoalType,
    ConstraintType,
    ManipulationType,
    DependencyType,
    DynamicsType,
    ProblemSpec,
)


@dataclass(frozen=True)
class SolverCapabilities:
    supports_goal_type: Set[GoalType]
    supports_dependency_type: Set[DependencyType]
    supports_dynamics_type: Set[DynamicsType]
    supports_agent_types: Set[AgentType]
    supports_constraints: Set[ConstraintType]
    supports_manipulation: Set[ManipulationType]

    def __repr__(self):
        return (
            f"SolverCapabilities(Agents: {[a.value for a in self.supports_agent_types]}, "
            f"Constraints: {[c.value for c in self.supports_constraints]}, "
            f"Env: {[e.value for e in self.supports_manipulation]}, "
            f"Dynamics: {[e.value for e in self.supports_dynamics_type]}, "
            f"Goals: {[e.value for e in self.supports_goal_type]}, "
            f"Dependencies: {[e.value for e in self.supports_dependency_type]}, "
        )


class BasePlanner(ABC):
    solver_capabilities: SolverCapabilities

    def __init__(self, env: BaseProblem):
        pass

    @abstractmethod
    def plan(
        self, planner_termination_criterion: PlannerTerminationCondition
    ) -> Tuple[List[State] | None, Dict[str, Any]]:
        """
        Plans a geometric path/trajectory for the specified problem. This function assumes that the problem is solvable by the planner.
        """
        pass

    def can_solve(self, problem_spec: ProblemSpec) -> bool:
        """
        Determines if a problem can be solved by the planner at hand by comparing
        the attributes of the problem with the attributes that the planner can deal with.
        """
        problem_spec = problem_spec
        caps = self.solver_capabilities

        # All conditions must be met
        return (
            problem_spec.agent_type in caps.supports_agent_types
            and problem_spec.constraints in caps.supports_constraints
            and problem_spec.goal in caps.supports_goal_type
            and problem_spec.dynamics in caps.supports_dependency_type
            and problem_spec.dependency in caps.supports_dependency_type
            and problem_spec.manipulation in caps.supports_manipulation
        )
