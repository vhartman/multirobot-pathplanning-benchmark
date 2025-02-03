import robotic as ry
import numpy as np
import random

from abc import ABC, abstractmethod

from typing import List, Dict, Optional
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
    config_dist,
)
from multi_robot_multi_goal_planning.problems.dependency_graph import DependencyGraph


class Goal(ABC):
    def __init__(self):
        pass

    # TODO: this should be a coficturaiotn not an ndarray
    @abstractmethod
    def satisfies_constraints(self, q: NDArray, mode: "Mode", tolerance: float) -> bool:
        pass

    @abstractmethod
    def sample(self, mode: "Mode") -> NDArray:
        pass


# class DummyGoal(ABC):
#     def __init__(self):
#         pass

#     def satisfies_constraints(self, q, tolerance):
#         return True

#     def sample(self):
#         pass


class GoalRegion(Goal):
    def __init__(self, limits: NDArray):
        self.limits = limits

    def satisfies_constraints(self, q: NDArray, mode: "Mode", _) -> bool:
        if np.all(q > self.limits[0, :]) and np.all(q < self.limits[1, :]):
            return True

    def sample(self, mode: "Mode") -> NDArray:
        q = (
            np.random.rand(len(self.limits[0, :]))
            * (self.limits[1, :] - self.limits[0, :])
            + self.limits[0, :]
        )
        return q


# TODO: implement sampler to sample a goal
class ConstrainedGoal(Goal):
    pass


class ConditionalGoal(Goal):
    def __init__(self, conditions, goals):
        self.conditions = conditions
        self.goals = goals

    def satisfies_constraints(self, q: NDArray, mode: "Mode", tolerance: float) -> bool:
        for c, g in zip(self.conditions, self.goals):
            if (
                np.linalg.norm(mode.entry_configuration[0] - c) < tolerance
                and np.linalg.norm(g - q) < tolerance
            ):
                return True

        return False

    def sample(self, mode: "Mode") -> NDArray:
        for c, g in zip(self.conditions, self.goals):
            if np.linalg.norm(mode.entry_configuration[0] - c) < 1e-5:
                return g

        raise ValueError("No feasible goal in mode")


class GoalSet(Goal):
    def __init__(self, goals):
        self.goals = goals

    def satisfies_constraints(self, q: NDArray, mode: "Mode", tolerance: float) -> bool:
        for g in self.goals:
            if np.linalg.norm(g - q) < tolerance:
                return True

        return False

    def sample(self, mode: "Mode") -> NDArray:
        rnd = np.random.randint(0, len(self.goals))
        return self.goals[rnd]


class SingleGoal(Goal):
    def __init__(self, goal: NDArray):
        self.goal = goal

    def satisfies_constraints(
        self, q: ry.Config, mode: "Mode", tolerance: float
    ) -> bool:
        if np.linalg.norm(self.goal - q) < tolerance:
            return True

        return False

    def sample(self, mode: "Mode") -> NDArray:
        return self.goal


class Task:
    name: str
    robots: List[str]
    goal: Goal

    # things for manipulation
    type: str
    frames: List[str]
    side_effect: str

    # things for the future:
    constraints = List

    def __init__(
        self, robots: List[str], goal: NDArray, type=None, frames=None, side_effect=None
    ):
        self.robots = robots
        self.goal = goal

        # constraints
        self.type = type
        self.frames = frames
        self.side_effect = side_effect


class Mode:
    task_ids: List[int]
    entry_configuration: Configuration
    sg: Dict[str, tuple]

    id: int
    prev_mode: "Mode"
    next_modes: List["Mode"]

    id_counter = 0

    def __init__(self, task_list, entry_configuration):
        self.task_ids = task_list
        self.entry_configuration = entry_configuration

        # TODO: set in constructor?
        self.prev_mode = None
        self.sg = {}
        self.next_modes = []

        self.id = Mode.id_counter
        Mode.id_counter += 1

        self.cached_hash = None

    def __repr__(self):
        return "Tasks: " + str(self.task_ids) + "id: " + str(self.id)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        if self.cached_hash is None:
            entry_hash = 0
            sg_hash = hash(frozenset(self.sg.items()))
            task_hash = hash(tuple(self.task_ids))
            self.cached_hash = hash((entry_hash, sg_hash, task_hash))
        
        return self.cached_hash
        

class State:
    q: Configuration
    m: Mode

    def __init__(self, q: Configuration, m: Mode):
        self.q = q
        self.mode = m

    def to_dict(self):
        return {
            "q": self.q.state().tolist(),
            "mode": self.mode.task_ids
        }        


def state_dist(start: State, end: State) -> float:
    if start.mode != end.mode:
        return np.inf

    return config_dist(start.q, end.q)


class BaseModeLogic(ABC):
    tasks: List[Task]
    _task_name_dict: Dict[str, Task]
    _task_id_dict: Dict[str, int]

    def __init__(self):
        self.start_mode = self.make_start_mode()
        self._terminal_task_ids = self.make_symbolic_end()

    def _get_task_by_name(self, name):
        for t in self.tasks:
            if t.name == name:
                return t

    def _get_task_id_by_name(self, name):
        for i, t in enumerate(self.tasks):
            if t.name == name:
                return i

    @abstractmethod
    def make_start_mode(self):
        pass

    @abstractmethod
    def make_symbolic_end(self):
        pass

    @abstractmethod
    def get_valid_next_task_combinations(self, m: Mode):
        pass

    @abstractmethod
    def is_terminal_mode(self, mode: Mode):
        pass

    @abstractmethod
    def done(self, q: Configuration, m: Mode) -> bool:
        pass

    @abstractmethod
    def get_next_mode(self, q: Configuration, mode: Mode):
        pass

    @abstractmethod
    def is_transition(self, q: Configuration, m: Mode) -> bool:
        pass

    @abstractmethod
    def get_active_task(self, current_mode: Mode, next_task_ids: List[int]) -> Task:
        pass


# concrete implementations of the required abstract classes for the sequence-setting.
# TODO: technically, this is a specialization of the dependency graph below - should we make this explicit?
class SequenceMixin(BaseModeLogic):
    sequence: List[int]
    tasks: List[Task]

    def _make_sequence_from_names(self, names: List[str]) -> List[int]:
        sequence = []

        for name in names:
            no_task_with_name_found = True
            for idx, task in enumerate(self.tasks):
                if name == task.name:
                    sequence.append(idx)
                    no_task_with_name_found = False

            if no_task_with_name_found:
                raise ValueError(f"Task with name {name} not found.")

        return sequence

    def make_start_mode(self) -> Mode:
        mode_dict = {}

        for task_index in self.sequence:
            task_robots = self.tasks[task_index].robots

            for r in task_robots:
                if r not in mode_dict:
                    mode_dict[r] = task_index

        task_ids = []
        for r in self.robots:
            task_ids.append(mode_dict[r])

        start_mode = Mode(task_ids, self.start_pos)
        return start_mode

    def make_symbolic_end(self) -> List[int]:
        mode_dict = {}

        for task_index in self.sequence:
            task_robots = self.tasks[task_index].robots

            # difference to above: we do not check if the robot already has a task assigned
            for r in task_robots:
                mode_dict[r] = task_index

        mode = []
        for r in self.robots:
            mode.append(mode_dict[r])

        return mode

    def is_terminal_mode(self, mode: Mode):
        if mode.task_ids == self._terminal_task_ids:
            return True

        return False

    def get_current_seq_index(self, mode: Mode) -> int:
        # Approach: iterate through all indices, find them in the sequence, and check which is the one
        # that has to be fulfilled first
        min_sequence_pos = len(self.sequence) - 1
        for i, task_id in enumerate(mode.task_ids):
            # print("robots in task:", self.tasks[m].robots, self.sequence.index(m))
            if task_id != self._terminal_task_ids[i]:
                min_sequence_pos = min(self.sequence.index(task_id), min_sequence_pos)

        return min_sequence_pos

    def get_sequence(self):
        return self.sequence

    def get_robot_sequence(self, robot: str):
        pass

    def get_goal_constrained_robots(self, mode: Mode) -> List[str]:
        seq_index = self.get_current_seq_index(mode)
        task = self.tasks[self.sequence[seq_index]]
        return task.robots

    def done(self, q: Configuration, m: Mode) -> bool:
        if not self.is_terminal_mode(m):
            return False

        terminal_task_idx = self.sequence[-1]
        terminal_task = self.tasks[terminal_task_idx]
        involved_robots = terminal_task.robots

        q_concat = []
        for r in involved_robots:
            r_idx = self.robots.index(r)
            q_concat.append(q.robot_state(r_idx))

        q_concat = np.concatenate(q_concat)

        if terminal_task.goal.satisfies_constraints(
            q_concat, mode=m, tolerance=self.tolerance
        ):
            return True

        return False

    def is_transition(self, q: Configuration, m: Mode) -> bool:
        if self.is_terminal_mode(m):
            return False

        task = self.get_active_task(m, None)

        q_concat = []
        for r in task.robots:
            r_idx = self.robots.index(r)
            q_concat.append(q.robot_state(r_idx))

        q_concat = np.concatenate(q_concat)

        if task.goal.satisfies_constraints(q_concat, mode=m, tolerance=self.tolerance):
            return True

        return False

    def get_valid_next_task_combinations(self, mode: Mode) -> List[List[int]]:
        if self.is_terminal_mode(mode):
            return []

        seq_idx = self.get_current_seq_index(mode)

        # find the next mode for the currently constrained one(s)
        task_idx = self.sequence[seq_idx]
        rs = self.tasks[task_idx].robots

        next_task_ids = mode.task_ids.copy()

        # find next occurrence of the robot in the sequence/dep graph
        for r in rs:
            for idx in self.sequence[seq_idx + 1 :]:
                if r in self.tasks[idx].robots:
                    r_idx = self.robots.index(r)
                    next_task_ids[r_idx] = idx
                    break

        return [next_task_ids]

    def get_next_mode(self, q: Optional[Configuration], mode: Mode) -> Mode:
        next_task_ids = self.get_valid_next_task_combinations(mode)[0]

        next_mode = Mode(task_list=next_task_ids, entry_configuration=q)
        next_mode.prev_mode = mode
        mode.next_modes.append(next_mode)

        sg = self.get_scenegraph_info_for_mode(next_mode)
        next_mode.sg = sg

        return next_mode

    def get_active_task(self, current_mode: Mode, next_task_ids: List[int]) -> Task:
        seq_idx = self.get_current_seq_index(current_mode)
        return self.tasks[self.sequence[seq_idx]]


class DependencyGraphMixin(BaseModeLogic):
    graph: DependencyGraph
    tasks: List[Task]

    def _make_sequence_from_names(self, names: List[str]) -> List[int]:
        sequence = []

        for name in names:
            no_task_with_name_found = True
            for idx, task in enumerate(self.tasks):
                if name == task.name:
                    sequence.append(idx)
                    no_task_with_name_found = False

            if no_task_with_name_found:
                raise ValueError(f"Task with name {name} not found.")

        return sequence

    def _make_start_mode_from_sequence(self, sequence) -> Mode:
        mode_dict = {}

        for task_index in sequence:
            task_robots = self.tasks[task_index].robots

            for r in task_robots:
                if r not in mode_dict:
                    mode_dict[r] = task_index

        task_ids = []
        for r in self.robots:
            task_ids.append(mode_dict[r])

        start_mode = Mode(task_ids, self.start_pos)
        return start_mode

    def _make_terminal_mode_from_sequence(self, sequence) -> Mode:
        mode_dict = {}
        _completed_symbolic_tasks_cache: Dict[Mode, List[str]]

        for task_index in sequence:
            task_robots = self.tasks[task_index].robots

            # difference to above: we do not check if the robot already has a task assigned
            for r in task_robots:
                mode_dict[r] = task_index

        mode = []
        for r in self.robots:
            mode.append(mode_dict[r])

        return mode

    def _verify_graph(self) -> bool:
        # ensure that there are no multiple root nodes for the same robot
        # ensure that there is only one leaf node

        return True

    def make_start_mode(self) -> Mode:
        possible_named_sequence = self.graph.get_build_order()
        possible_id_sequence = self._make_sequence_from_names(possible_named_sequence)

        return self._make_start_mode_from_sequence(possible_id_sequence)

    def make_symbolic_end(self) -> Mode:
        possible_named_sequence = self.graph.get_build_order()
        possible_id_sequence = self._make_sequence_from_names(possible_named_sequence)

        return self._make_terminal_mode_from_sequence(possible_id_sequence)

    def _get_finished_tasks_from_mode(self, mode: Mode) -> List[str]:
        completed_tasks = []
        for i, task_id in enumerate(mode.task_ids):
            robot = self.robots[i]
            task_name = self.tasks[task_id].name

            dependencies = self.graph.get_all_dependencies(task_name)

            for dep in dependencies:
                robots = self._get_task_by_name(dep).robots
                if robot in robots:
                    completed_tasks.append(dep)

        # make unique
        completed_tasks = list(set(completed_tasks))

        return completed_tasks

    def get_valid_next_task_combinations(self, m: Mode) -> List[List[int]]:
        # construct set of all already done tasks
        done_tasks = self._get_finished_tasks_from_mode(m)

        if m.task_ids == self._terminal_task_ids:
            return []

        mode_task_names = []
        for task_id in m.task_ids:
            mode_task_names.append(self.tasks[task_id].name)

        possible_next_task_ids = []

        for task_name in mode_task_names:
            dependencies = self.graph.get_all_dependencies(task_name)
            if all(dep in done_tasks or dep == task_name for dep in dependencies):
                # this is a possible next task
                robots = self._get_task_by_name(task_name).robots

                new_task_ids = m.task_ids.copy()

                for r in robots:
                    i = self.robots.index(r)
                    new_task_ids[i] = self._get_task_id_by_name(
                        self._get_next_task_for_robot(task_name, self.robots[i])
                    )

                possible_next_task_ids.append(new_task_ids)

        # print(possible_next_task_ids)
        return possible_next_task_ids

    def _get_next_task_for_robot(self, current_task_name, robot):
        possible_order = self.graph.get_build_order()
        idx = possible_order.index(current_task_name)
        for name in possible_order[idx + 1 :]:
            id = self._get_task_id_by_name(name)
            involved_robots = self.tasks[id].robots
            if robot in involved_robots:
                return name

    def get_next_mode(self, q: Configuration, mode: Mode):
        next_mode_ids = self.get_valid_next_task_combinations(mode)

        # all of this is duplicated with the method below
        # TODO: can it be possible that multiple mode transitions are possible?
        # TODO: should we change this to 'get_next_modes'?
        for next_mode in next_mode_ids:
            for i in range(len(self.robots)):
                if next_mode[i] != mode.task_ids[i]:
                    # need to check if the goal conditions for this task are fulfilled in the current state
                    task = self.tasks[mode.task_ids[i]]
                    q_concat = []
                    for r in task.robots:
                        r_idx = self.robots.index(r)
                        q_concat.append(q.robot_state(r_idx))

                    q_concat = np.concatenate(q_concat)

                    if task.goal.satisfies_constraints(
                        q_concat, mode=mode, tolerance=self.tolerance
                    ):
                        tmp = Mode(task_list=next_mode.copy(), entry_configuration=q)
                        tmp.prev_mode = mode

                        sg = self.get_scenegraph_info_for_mode(tmp)
                        tmp.sg = sg

                        return tmp

        raise ValueError("This does not fulfill the constraints to reach a new mode.")

    def is_transition(self, q: Configuration, m: Mode) -> bool:
        if self.is_terminal_mode(m):
            return False

        next_mode_ids = self.get_valid_next_task_combinations(m)

        for next_mode in next_mode_ids:
            for i in range(len(self.robots)):
                if next_mode[i] != m.task_ids[i]:
                    # need to check if the goal conditions for this task are fulfilled in the current state
                    task = self.tasks[m.task_ids[i]]
                    q_concat = []
                    for r in task.robots:
                        r_idx = self.robots.index(r)
                        q_concat.append(q.robot_state(r_idx))

                    q_concat = np.concatenate(q_concat)

                    if task.goal.satisfies_constraints(
                        q_concat, mode=m, tolerance=self.tolerance
                    ):
                        return True

        return False

    def done(self, q: Configuration, mode: Mode):
        if not self.is_terminal_mode(mode):
            return False

        leaf_nodes = list(self.graph.get_leaf_nodes())
        assert len(leaf_nodes) == 1

        terminal_task_name = leaf_nodes[0]
        terminal_task = self._get_task_by_name(terminal_task_name)
        involved_robots = terminal_task.robots

        q_concat = []
        for r in involved_robots:
            r_idx = self.robots.index(r)
            q_concat.append(q.robot_state(r_idx))

        q_concat = np.concatenate(q_concat)

        if terminal_task.goal.satisfies_constraints(
            q_concat, mode=mode, tolerance=self.tolerance
        ):
            return True

        return False

    def is_terminal_mode(self, mode: Mode):
        if mode.task_ids == self._terminal_task_ids:
            return True

        return False

    def get_active_task(self, current_mode: Mode, next_task_ids: List[int]) -> Task:
        if next_task_ids is None:
            # we should return the terminal task here
            return self.tasks[self._terminal_task_ids[0]]
        else:
            different_tasks = []
            for i, task_id in enumerate(current_mode.task_ids):
                if task_id != next_task_ids[i]:
                    different_tasks.append(task_id)

            # print("next", next_task_ids)
            # print("current", current_mode.task_ids)
            # print("changing task_ids", different_tasks)
            different_tasks = list(set(different_tasks))
            assert len(different_tasks) == 1

            return self.tasks[different_tasks[0]]


# TODO: split into env + problem specification
class BaseProblem(ABC):
    robots: List[str]
    robot_dims: Dict[str, int]
    robot_idx: Dict[str, NDArray]
    start_pos: Configuration

    start_mode: Mode
    _terminal_task_ids: List[int]

    # visualization
    @abstractmethod
    def show_config(self, q: Configuration):
        pass

    @abstractmethod
    def show(self):
        pass

    ## General methods
    def get_start_pos(self):
        return self.start_pos

    def get_start_mode(self):
        return self.start_mode

    def get_robot_dim(self, robot: str):
        return self.robot_dims[robot]

    def get_all_bounds(self):
        self.bounds

    # def get_robot_bounds(self, robot):
    #     self.bounds

    @abstractmethod
    def done(self, q: Configuration, mode: Mode):
        pass

    @abstractmethod
    def is_transition(self, q: Configuration, m: Mode) -> bool:
        pass

    @abstractmethod
    def get_next_mode(self, q: Configuration, mode: Mode):
        pass

    @abstractmethod
    def get_active_task(self, mode: Mode, next_task_ids: List[int]) -> Task:
        pass

    # @abstractmethod
    # def get_tasks_for_mode(self, mode: Mode) -> List[Task]:
    #     pass

    # Collision checking and environment related methods
    @abstractmethod
    def get_scenegraph_info_for_mode(self, mode: Mode):
        pass

    @abstractmethod
    def set_to_mode(self, mode: Mode):
        pass

    @abstractmethod
    def is_collision_free(self, q: Optional[Configuration], mode: Mode) -> bool:
        pass

    def is_collision_free_for_robot(
        self, r: str, q: NDArray, m: Mode, collision_tolerance: float = 0.01
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_edge_collision_free(
        self,
        q1: Configuration,
        q2: Configuration,
        mode: Mode,
        resolution: float = 0.1,
        tolerance: float = 0.01
    ) -> bool:
        pass

    @abstractmethod
    def is_path_collision_free(self, path: List[State]) -> bool:
        pass

    def is_valid_plan(self, path: List[State]) -> bool:
        # check if it is collision free and if all modes are passed in order
        # only take the configuration into account for that
        mode = self.start_mode
        collision = False
        for i in range(len(path)):
            mode = path[i].mode

            # check if the state is collision free
            if not self.is_collision_free(path[i].q, mode):
                print(f"There is a collision at index {i}")
                # col = self.C.getCollisionsTotalPenetration()
                # print(col)
                # self.show()
                collision = True

            # if the next mode is a transition, check where to go
            # if i < len(path) - 1 and self.is_transition(path[i].q, mode):
            #     # TODO: this does not work if multiple switches are possible at the same time
            #     next_mode = self.get_next_mode(path[i].q, mode)

            #     if path[i + 1].mode == next_mode:
            #         mode = next_mode

        if not self.done(path[-1].q, path[-1].mode):
            print("Final mode not reached")
            return False

        if collision:
            print("There was a collision")
            return False

        return True

    @abstractmethod
    def config_cost(self, start: Configuration, goal: Configuration) -> float:
        pass

    @abstractmethod
    def batch_config_cost(
        self,
        starts: List[Configuration],
        ends: List[Configuration],
    ) -> List[float]:
        pass

    def state_cost(self, start: State, end: State) -> float:
        if start.mode != end.mode:
            return np.inf

        return self.config_cost(start.q, end.q)
