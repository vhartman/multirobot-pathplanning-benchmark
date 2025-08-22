from abc import ABC, abstractmethod


class PlannerTerminationCondition(ABC):
    @abstractmethod
    def should_terminate(
        self, current_iterations: int | None = None, current_time: float | None = None
    ) -> bool:
        pass


class IterationTerminationCondition(PlannerTerminationCondition):
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def __repr__(self):
        return f"max iter.: {self.max_iterations}"

    def should_terminate(
        self, current_iterations: int | None = None, current_time: float | None = None
    ) -> bool:
        assert current_iterations
        
        if self.max_iterations < current_iterations:
            return True

        return False


class RuntimeTerminationCondition(PlannerTerminationCondition):
    def __init__(self, max_runtime_in_s: int):
        self.max_runtime_in_s = max_runtime_in_s

    def __repr__(self):
        return f"max runtime.: {self.max_runtime_in_s} s"

    def should_terminate(
        self, current_iterations: int | None = None, current_time: float | None = None
    ) -> bool:
        assert current_time is not None

        if self.max_runtime_in_s < current_time:
            return True

        return False
