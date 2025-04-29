import robotic as ry
import numpy as np
import random

import copy

from abc import ABC, abstractmethod

from typing import List, Dict, Optional
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
    config_dist,
)
from multi_robot_multi_goal_planning.problems.dependency_graph import DependencyGraph
# from multi_robot_multi_goal_planning.problems.util import generate_binary_search_indices


from functools import cache
from collections import deque

from itertools import product


@cache
def generate_binary_search_indices(N):
    sequence = []
    queue = deque([(0, N - 1)])
    while queue:
        start, end = queue.popleft()
        if start > end:
            continue
        mid = (start + end) // 2
        sequence.append(int(mid))
        queue.append((start, mid - 1))
        queue.append((mid + 1, end))
    return tuple(sequence)


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

    @abstractmethod
    def serialize(self):
        pass

    @classmethod
    @abstractmethod
    def from_data(cls, data):
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

    def satisfies_constraints(self, q: NDArray, mode: "Mode", tolerance: float) -> bool:
        if np.all(q > self.limits[0, :]) and np.all(q < self.limits[1, :]):
            return True

    def sample(self, mode: "Mode") -> NDArray:
        q = (
            np.random.rand(len(self.limits[0, :]))
            * (self.limits[1, :] - self.limits[0, :])
            + self.limits[0, :]
        )
        return q

    def serialize(self):
        return self.limits.tolist()

    @classmethod
    def from_data(cls, data):
        return GoalRegion(np.array(data))


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
            if np.linalg.norm(mode.entry_configuration[0] - c) < 1e-8:
                return g

        raise ValueError("No feasible goal in mode")

    def serialize(self):
        print("This is not yet implemented")
        raise NotImplementedError

    @classmethod
    def from_data(cls, data):
        print("This is not yet implemented")
        raise NotImplementedError


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

    def serialize(self):
        return [goal.tolist() for goal in self.goals]

    @classmethod
    def from_data(cls, data):
        return GoalSet([np.array(goal) for goal in data])


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

    def serialize(self):
        return self.goal.tolist()

    @classmethod
    def from_data(cls, data):
        return SingleGoal(np.array(data))


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
        self.name = None

        # constraints
        self.type = type
        self.frames = frames
        self.side_effect = side_effect


class Mode:
    __slots__ = (
        "task_ids",
        "entry_configuration",
        "sg",
        "id",
        "prev_mode",
        "next_modes",
        "additional_hash_info",
        "_cached_hash",
    )

    task_ids: List[int]
    entry_configuration: Configuration
    sg: Dict[str, tuple]

    id: int
    prev_mode: "Mode"
    next_modes: List["Mode"]
    additional_hash_info: any

    id_counter = 0

    def __init__(self, task_list, entry_configuration):
        self.task_ids = task_list
        self.entry_configuration = entry_configuration

        self.prev_mode = None
        self.sg = {}
        self.next_modes = []

        self.id = Mode.id_counter
        Mode.id_counter += 1

        self._cached_hash = None

        self.additional_hash_info = None

    def __repr__(self):
        return f"Tasks: {self.task_ids}, id: {self.id}"

    def __eq__(self, other):
        if not isinstance(other, Mode):
            return False

        if self.id == other.id:
            return True

        if self.task_ids != other.task_ids:
            return False

        return hash(self) == hash(other)

    def __hash__(self):
        if self._cached_hash is None:
            entry_hash = 0
            sg_fitered = {
                k: (v[0], v[1], v[2]) if len(v) > 2 else v for k, v in self.sg.items()
            }
            sg_hash = hash(frozenset(sg_fitered.items()))
            task_hash = hash(tuple(self.task_ids))

            self._cached_hash = hash(
                (entry_hash, sg_hash, task_hash, self.additional_hash_info)
            )

        return self._cached_hash


class State:
    q: Configuration
    mode: Mode

    def __init__(self, q: Configuration, m: Mode):
        self.q = q
        self.mode = m

    def to_dict(self):
        return {"q": self.q.state().tolist(), "mode": self.mode.task_ids}


def state_dist(start: State, end: State) -> float:
    if start.mode != end.mode:
        return np.inf

    return config_dist(start.q, end.q)


class BaseModeLogic(ABC):
    tasks: List[Task]
    _task_name_dict: Dict[str, Task]
    _task_id_dict: Dict[str, int]

    def __init__(self):
        self.prev_mode = None
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
    def get_next_modes(self, q: Configuration, mode: Mode):
        pass

    @abstractmethod
    def is_transition(self, q: Configuration, m: Mode) -> bool:
        pass

    @abstractmethod
    def get_active_task(self, current_mode: Mode, next_task_ids: List[int]) -> Task:
        pass


class UnorderedButAssignedMixin(BaseModeLogic):
    tasks: List[Task]
    per_robot_tasks: List[List[int]]
    task_dependencies: Dict[int, List[int]]

    terminal_task: int

    def make_start_mode(self):
        ids = [0] * self.start_pos.num_agents()
        m = Mode(ids, self.start_pos)
        return m

    def make_symbolic_end(self):
        return [self.terminal_task] * self.start_pos.num_agents()

    @cache
    def get_valid_next_task_combinations(self, mode: Mode):
        # print(f"called get valid next with {mode.task_ids}")
        if self.is_terminal_mode(mode):
            return []

        # check which tasks have been done, return all possible next combinations
        unfinished_tasks_per_robot = copy.deepcopy(self.per_robot_tasks)
        finished_tasks = []

        # there might be a problem that until now we assumed that the mode is markov
        # (i.e it does not matter what we did before for the current task)
        # m.prev_mode

        # print(f"current mode {mode.task_ids}")
        # print("unfinished tasks", unfinished_tasks_per_robot)
        num_agents = len(mode.task_ids)

        if mode.prev_mode is None:
            has_unfinished_tasks = True

            # this can only be the case for the starting mode.
            # construct all possible follow up modes.
            # Important: we want to transition both states here, compared to the case below
            # where we only transition one at a time.
            feasible_tasks_per_robot = [[] for _ in range(num_agents)]

            for i in range(num_agents):
                current_task = mode.task_ids[i]
                for t in unfinished_tasks_per_robot[i]:
                    task_requirements_fulfilled = True
                    if t in self.task_dependencies:
                        for dependency in self.task_dependencies[t]:
                            if (
                                dependency not in finished_tasks
                                and dependency != mode.task_ids[i]
                            ):
                                task_requirements_fulfilled = False

                    if not task_requirements_fulfilled:
                        # print(f"dependency unfulfilled for task {t}")
                        continue
                        
                    feasible_tasks_per_robot[i].append(t)
            
            next_states = [
                list(combo)
                for combo in product(*feasible_tasks_per_robot)
                if combo.count(self.terminal_task) != num_agents
            ]

            return next_states

        else:
            pm = mode.prev_mode
            # print("prev ids")
            while pm:
                # print(pm.task_ids)
                task_ids = pm.task_ids
                for i, task_id in enumerate(task_ids):
                    if (
                        task_id in unfinished_tasks_per_robot[i]
                        and task_id != mode.task_ids[i]
                    ):
                        # print(f"trying to remove {task_id}")
                        unfinished_tasks_per_robot[i].remove(task_id)
                        finished_tasks.append(task_id)

                pm = pm.prev_mode

            has_unfinished_tasks = False
            for i, tasks in enumerate(unfinished_tasks_per_robot):
                # print(i, tasks)
                if len(tasks) > 0:
                    # unfinished_tasks_per_robot[i].append(self.terminal_task)
                    has_unfinished_tasks = True
                if len(tasks) == 1 and tasks[0] == mode.task_ids[i]:
                    unfinished_tasks_per_robot[i].append(self.terminal_task)
                    # print(f"ADDED TERMINAL TO {i}")

        # print("Unfinished tasks", unfinished_tasks_per_robot)

        if has_unfinished_tasks:
            # print("Unfinished tasks", unfinished_tasks_per_robot)
            next_states = []

            for i in range(num_agents):
                current_task = mode.task_ids[i]
                for t in unfinished_tasks_per_robot[i]:
                    task_requirements_fulfilled = True
                    if t in self.task_dependencies:
                        for dependency in self.task_dependencies[t]:
                            if (
                                dependency not in finished_tasks
                                and dependency != mode.task_ids[i]
                            ):
                                task_requirements_fulfilled = False

                    if not task_requirements_fulfilled:
                        # print(f"dependency unfulfilled for task {t}")
                        continue

                    # print(f"dependency done for task {t}")

                    if t != current_task:
                        new_state = copy.deepcopy(mode.task_ids)
                        new_state[i] = t
                        next_states.append(new_state)

            # print("current task ids", mode.task_ids)
            # print("possible next tasks:", next_states)
            # print()

            return next_states

        return [[self.terminal_task] * self.start_pos.num_agents()]

    def is_terminal_mode(self, mode: Mode):
        # check if all task shave been done
        if all([t == self.terminal_task for t in mode.task_ids]):
            return True

        return False

    def done(self, q: Configuration, m: Mode) -> bool:
        if not self.is_terminal_mode(m):
            return False

        # check if this configuration fulfills the final goals
        terminal_task = self.tasks[self.terminal_task]
        involved_robots = terminal_task.robots

        q_concat = []
        for r in involved_robots:
            r_idx = self.robots.index(r)
            q_concat.append(q.robot_state(r_idx))

        q_concat = np.concatenate(q_concat)

        if terminal_task.goal.satisfies_constraints(q_concat, mode=m, tolerance=1e-8):
            return True

        return False

    def get_next_modes(self, q: Configuration, mode: Mode):
        # needs to be changed to get next modes
        valid_next_combinations = self.get_valid_next_task_combinations(mode)

        # print(valid_next_combinations)

        possible_next_mode_ids = []
        for next_mode_ids in valid_next_combinations:
            for i in range(len(self.robots)):
                # for dummy start mode
                if next_mode_ids in possible_next_mode_ids:
                    continue
                if next_mode_ids[i] != mode.task_ids[i]:
                    # need to check if the goal conditions for this task are fulfilled in the current state
                    task = self.tasks[mode.task_ids[i]]
                    q_concat = []
                    for r in task.robots:
                        r_idx = self.robots.index(r)
                        q_concat.append(q.robot_state(r_idx))

                    q_concat = np.concatenate(q_concat)

                    if task.goal.satisfies_constraints(
                        q_concat, mode=mode, tolerance=1e-8
                    ):
                        possible_next_mode_ids.append(next_mode_ids)

        next_modes = []

        for next_id in possible_next_mode_ids:
            next_mode = Mode(next_id, q)

            next_mode.prev_mode = mode
            tmp = tuple(tuple(sublist) for sublist in valid_next_combinations)
            next_mode.additional_hash_info = tmp

            sg = self.get_scenegraph_info_for_mode(next_mode)
            next_mode.sg = sg

            mode_exists = False
            for nm in mode.next_modes:
                if hash(nm) == hash(next_mode):
                    # print("AAAAAAAAAA")
                    next_modes.append(nm)
                    mode_exists = True

                    break

            if not mode_exists:
                mode.next_modes.append(next_mode)
                next_modes.append(next_mode)

        # print(mode)
        # print(mode.next_modes)
        # print()

        return next_modes

    def is_transition(self, q: Configuration, m: Mode) -> bool:
        if self.is_terminal_mode(m):
            return False

        # check if any of the robots is fulfilling its goal constraints
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
                        q_concat, mode=m, tolerance=1e-8
                    ):
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


class FreeMixin(BaseModeLogic):
    tasks: List[Task]
    task_groups: List[
        List[int]
    ]  # describes groups of tasks of which one has to be done
    task_dependencies: Dict[int, List[int]]
    terminal_task: int

    def make_start_mode(self):
        # ids = [0, 1]
        ids = [0] * self.start_pos.num_agents()
        m = Mode(ids, self.start_pos)
        return m

    def make_symbolic_end(self):
        return [self.terminal_task] * self.start_pos.num_agents()

    def _get_finished_groups(self, mode: Mode) -> List[int]:
        finished_task_groups = []
        if mode.prev_mode is None:
            return []
        else:
            pm = mode.prev_mode
            while pm:
                task_ids = pm.task_ids
                for i, task_id in enumerate(task_ids):

                    for j in range(len(self.task_groups)):
                        if task_id in [id for _, id in self.task_groups[j]]:
                            finished_task_groups.append(j)

                pm = pm.prev_mode

        return list(set(sorted(finished_task_groups)))

    @cache
    def get_valid_next_task_combinations(self, mode: Mode) -> List[List[int]]:
        if self.is_terminal_mode(mode):
            return []

        # check which tasks have been done, return all possible next combinations
        unfinished_tasks = copy.deepcopy(self.task_groups)

        # there might be a problem that until now we assumed that the mode is markov
        # (i.e it does not matter what we did before for the current task)
        # m.prev_mode

        finished_task_groups = []
        finished_tasks = []

        if mode.prev_mode is None:
            pass
        else:
            pm = mode.prev_mode
            while pm:
                task_ids = pm.task_ids
                for i, task_id in enumerate(task_ids):
                    if task_id != mode.task_ids[i]:
                        finished_tasks.append(task_id)

                    for j in range(len(self.task_groups)):
                        if task_id in [id for _, id in self.task_groups[j]]:
                            finished_task_groups.append(j)

                pm = pm.prev_mode

            for i in sorted(list(set(finished_task_groups)), reverse=True):
                unfinished_tasks.pop(i)

            # additionally remove the tasks that are already assigned
            task_ids = mode.task_ids
            additional_groups_to_remove = []
            for task_id in task_ids:
                for j in range(len(unfinished_tasks)):
                    if task_id in [id for _, id in unfinished_tasks[j]]:
                        additional_groups_to_remove.append(j)

            for i in sorted(list(set(additional_groups_to_remove)), reverse=True):
                unfinished_tasks.pop(i)

        # print(mode)
        # print(unfinished_tasks)

        next_states = []
        num_agents = len(mode.task_ids)

        if mode.task_ids[0] == 0:
            # this is the dummy start state
            feasible_tasks_per_robot = {}
            for i in range(num_agents):
                feasible_tasks_per_robot[i] = [self.terminal_task]

            for set_of_tasks in unfinished_tasks:
                for robot_idx, t in set_of_tasks:
                    task_requirements_fulfilled = True
                    if t in self.task_dependencies:
                        for dependency in self.task_dependencies[t]:
                            if dependency not in finished_tasks:
                                task_requirements_fulfilled = False

                    if not task_requirements_fulfilled:
                        # print(f"dependency unfulfilled for task {t}")
                        continue

                    feasible_tasks_per_robot[robot_idx].append(t)

            next_states = [
                list(combo)
                for combo in product(*feasible_tasks_per_robot.values())
                if combo.count(self.terminal_task) != num_agents
            ]

            states_to_remove = []
            for state in next_states:
                group_origin = []
                for task_id in state:
                    for j, set_of_tasks in enumerate(self.task_groups):
                        if task_id in [task for r_id, task in set_of_tasks]:
                            if j in group_origin:
                                states_to_remove.append(state)
                            group_origin.append(j)

            for state in states_to_remove:
                next_states.remove(state)
                # print("removing")

            # print("possible states from 0")
            # print(next_states)
            # for task_ids in next_states:
            #     for task in task_ids:
            #         print(self.tasks[task].name)
            #     print()

        else:
            for set_of_tasks in unfinished_tasks:
                for robot_idx, t in set_of_tasks:
                    current_task = mode.task_ids[robot_idx]

                    if current_task == self.terminal_task:
                        continue

                    task_requirements_fulfilled = True
                    if t in self.task_dependencies:
                        for dependency in self.task_dependencies[t]:
                            if (
                                dependency not in finished_tasks
                                and dependency != current_task
                            ):
                                task_requirements_fulfilled = False

                    if not task_requirements_fulfilled:
                        # print(f"dependency unfulfilled for task {t}")
                        continue

                    if t != current_task:
                        new_state = copy.deepcopy(mode.task_ids)
                        new_state[robot_idx] = t
                        next_states.append(new_state)

            # append terminal transition for each robot separately
            for i in range(num_agents):
                new_state = copy.deepcopy(mode.task_ids)

                if self.terminal_task == new_state[i]:
                    continue
                # Check if there are any remaining dependencies for this task
                if [new_state[i]] in self.task_dependencies.values():
                    if next(k for k, v in self.task_dependencies.items() if v == [new_state[i]]) not in finished_tasks:
                        continue
                new_state[i] = self.terminal_task

                # check if this is the terminal_task_id_state
                if new_state == self.make_symbolic_end():
                    # if it is equal to the terminal state, we need to ensure that all tasks are done
                    if len(unfinished_tasks) != 0:
                        continue

                next_states.append(new_state)

            # if len(next_states) == 0:
            #     print("AAAAAAAA")

        # print("current state", mode)
        # print(len(unfinished_tasks))
        # print(unfinished_tasks)
        # print(next_states)

        return next_states

    def is_terminal_mode(self, mode: Mode):
        # check if all task shave been done
        if all([t == self.terminal_task for t in mode.task_ids]):
            return True

        return False

    def done(self, q: Configuration, m: Mode) -> bool:
        if not self.is_terminal_mode(m):
            return False

        # check if this configuration fulfills the final goals
        terminal_task = self.tasks[self.terminal_task]
        involved_robots = terminal_task.robots

        q_concat = []
        for r in involved_robots:
            r_idx = self.robots.index(r)
            q_concat.append(q.robot_state(r_idx))

        q_concat = np.concatenate(q_concat)

        if terminal_task.goal.satisfies_constraints(q_concat, mode=m, tolerance=1e-8):
            return True

        return False

    @cache
    def get_next_modes(self, q: Configuration, mode: Mode):
        # needs to be changed to get next modes
        # print(q.state(), mode)
        valid_next_combinations = self.get_valid_next_task_combinations(mode)

        # for next_mode_ids in valid_next_combinations:
        #     print(next_mode_ids, mode.task_ids)
        #     if mode.task_ids == next_mode_ids:
        #         print("WTFWTFWTW")

        # print(valid_next_combinations)

        possible_next_mode_ids = []
        for next_mode_ids in valid_next_combinations:
            for i in range(len(self.robots)):
                if next_mode_ids[i] != mode.task_ids[i]:
                    # need to check if the goal conditions for this task are fulfilled in the current state
                    task = self.tasks[mode.task_ids[i]]
                    q_concat = []
                    for r in task.robots:
                        r_idx = self.robots.index(r)
                        q_concat.append(q.robot_state(r_idx))

                    q_concat = np.concatenate(q_concat)

                    if task.goal.satisfies_constraints(
                        q_concat, mode=mode, tolerance=1e-8
                    ):
                        possible_next_mode_ids.append(next_mode_ids)

        next_modes = []

        for next_id in possible_next_mode_ids:
            next_mode = Mode(next_id, q)

            next_mode.prev_mode = mode
            tmp = tuple(self._get_finished_groups(next_mode))
            next_mode.additional_hash_info = copy.deepcopy(tmp)

            sg = self.get_scenegraph_info_for_mode(next_mode)
            next_mode.sg = sg

            mode_exists = False
            for nm in mode.next_modes:
                if hash(nm) == hash(next_mode):
                    # print("AAAAAAAAAA")
                    next_modes.append(nm)
                    mode_exists = True

                    break

            if not mode_exists:
                mode.next_modes.append(next_mode)
                next_modes.append(next_mode)

        # print(mode)
        # print(mode.next_modes)
        # print()

        return next_modes

    def is_transition(self, q: Configuration, m: Mode) -> bool:
        if self.is_terminal_mode(m):
            return False

        # check if any of the robots is fulfilling its goal constraints
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
                        q_concat, mode=m, tolerance=1e-8
                    ):
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
        sg = self.get_scenegraph_info_for_mode(start_mode, is_start_mode=True)
        start_mode.sg = sg
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

        if terminal_task.goal.satisfies_constraints(q_concat, mode=m, tolerance=1e-8):
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

        if task.goal.satisfies_constraints(q_concat, mode=m, tolerance=1e-8):
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

    def get_next_modes(self, q: Optional[Configuration], mode: Mode) -> Mode:
        next_task_ids = self.get_valid_next_task_combinations(mode)[0]

        next_mode = Mode(task_list=next_task_ids, entry_configuration=q)
        next_mode.prev_mode = mode

        sg = self.get_scenegraph_info_for_mode(next_mode)
        next_mode.sg = sg

        for nm in mode.next_modes:
            if hash(nm) == hash(next_mode):
                return [nm]

        mode.next_modes.append(next_mode)

        return [next_mode]

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
        sg = self.get_scenegraph_info_for_mode(start_mode, is_start_mode=True)
        start_mode.sg = sg
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

    def get_next_modes(self, q: Configuration, mode: Mode):
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
                        q_concat, mode=mode, tolerance=1e-8
                    ):
                        tmp = Mode(task_list=next_mode.copy(), entry_configuration=q)
                        tmp.prev_mode = mode

                        sg = self.get_scenegraph_info_for_mode(tmp)
                        tmp.sg = sg

                        for nm in mode.next_modes:
                            if hash(nm) == hash(tmp):
                                return [nm]

                        mode.next_modes.append(tmp)
                        # print(mode.next_modes)

                        return [tmp]

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
                        q_concat, mode=m, tolerance=1e-8
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
            q_concat, mode=mode, tolerance=1e-8
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

    # misc
    # collision_tolerance: float
    # collision_resolution: float

    # def __init__(self):
    #     self.collision_tolerance = 0.01
    #     self.collision_resolution = 0.01

    def serialize_tasks(self):
        # open file
        task_list = []

        for t in self.tasks:
            task_data = {
                "name": t.name,
                "robots": t.robots,
                "goal_type": type(t.goal).__name__,
                "goal": t.goal.serialize(),
                "type": t.type,
                "frames": t.frames,
                "side_effect": t.side_effect,
            }

            task_list.append(task_data)

        return task_list

    def export_tasks(self, path):
        task_list = self.serialize_tasks()
        with open(path, "w") as file:
            for task_data in task_list:
                file.write(f"{task_data}\n")

    def import_tasks(self, path):
        with open(path, "r") as file:
            task_list = []
            for line in file:
                task_data = eval(
                    line.strip()
                )  # Convert string representation back to dictionary
                goal_type = task_data["goal_type"]

                goal = None
                if goal_type == "SingleGoal":
                    goal = SingleGoal.from_data(task_data["goal"])
                elif goal_type == "GoalRegion":
                    goal = GoalRegion.from_data(task_data["goal"])
                elif goal_type == "GoalSet":
                    goal = GoalSet.from_data(task_data["goal"])
                elif goal_type == "ConditionalGoal":
                    goal = ConditionalGoal.from_data(task_data["goal"])
                elif goal_type == "ConstrainedGoal":
                    goal = ConstrainedGoal.from_data(task_data["goal"])

                task = Task(
                    robots=task_data["robots"],
                    goal=goal,
                    type=task_data["type"],
                    frames=task_data["frames"],
                    side_effect=task_data["side_effect"],
                )
                task.name = task_data["name"]
                task_list.append(task)

            self.tasks = task_list

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
    def get_next_modes(self, q: Configuration, mode: Mode):
        pass

    @abstractmethod
    def get_active_task(self, mode: Mode, next_task_ids: List[int]) -> Task:
        pass

    # @abstractmethod
    # def get_tasks_for_mode(self, mode: Mode) -> List[Task]:
    #     pass

    # Collision checking and environment related methods
    @abstractmethod
    def get_scenegraph_info_for_mode(self, mode: Mode, is_start_mode: bool = False):
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
        m: Mode,
        resolution: float = None,
        tolerance: float = None,
        include_endpoints: bool = False,
        N_start: int = 0,
        N_max: int = None,
        N: int = None,
    ) -> bool:
        pass

    def is_path_collision_free(
        self,
        path: List[State],
        binary_order=True,
        resolution=None,
        tolerance=None,
        check_edges_in_order=False,
    ) -> bool:
        if tolerance is None:
            tolerance = self.collision_tolerance

        if resolution is None:
            resolution = self.collision_resolution

        if binary_order:
            # idx = list(range(len(path) - 1))
            # np.random.shuffle(idx)
            idx = generate_binary_search_indices(len(path) - 1)
        else:
            idx = list(range(len(path) - 1))

        # valid_edges = 0

        if check_edges_in_order:
            for i in idx:
                # skip transition nodes
                # if path[i].mode != path[i + 1].mode:
                #     continue

                q1 = path[i].q
                q2 = path[i + 1].q
                mode = path[i].mode

                if not self.is_collision_free(q1, mode):
                    return False

                if not self.is_edge_collision_free(
                    q1,
                    q2,
                    mode,
                    resolution=resolution,
                    tolerance=tolerance,
                    include_endpoints=False,
                ):
                    return False

                # valid_edges += 1

            if not self.is_collision_free(path[-1].q, path[-1].mode):
                return False

            # print("checked edges in shortcutting: ", valid_edges)
        else:
            edge_queue = list(idx)
            checks_per_iteration = 10
            N_start = 0
            N_max = N_start + checks_per_iteration
            while edge_queue:
                edges_to_remove = []
                for i in edge_queue:
                    q1 = path[i].q
                    q2 = path[i + 1].q
                    mode = path[i].mode

                    if N_start == 0:
                        if not self.is_collision_free(q1, mode):
                            return False

                    res = self.is_edge_collision_free(
                        q1,
                        q2,
                        mode,
                        resolution=resolution,
                        tolerance=tolerance,
                        include_endpoints=False,
                        N_start=N_start,
                        N_max=N_max,
                    )

                    if res is None:
                        edges_to_remove.append(i)
                        continue

                    if not res:
                        return False

                for i in edges_to_remove:
                    edge_queue.remove(i)

                N_start += checks_per_iteration
                N_max += checks_per_iteration

            if not self.is_collision_free(path[-1].q, path[-1].mode):
                return False

        return True

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
