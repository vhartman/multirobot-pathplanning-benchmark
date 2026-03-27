from typing import List
from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseProblem,
    Mode,
)
import random
import numpy as np


class ModeValidation:
    # TODO: get rid of apply - if we call this, stuff should happen
    def __init__(
        self,
        env: BaseProblem,
        apply: bool,
        with_noise: bool = False,
    ):
        self.env = env
        self.blacklist_modes = set()
        self.whitelist_modes = set()
        self.invalid_next_ids = {}
        self.counter = 0
        self.apply = apply
        self.with_noise = with_noise

    def get_valid_modes(self, prev_mode: Mode, modes: List[Mode]) -> List[Mode]:
        """
        Filters a list of modes based on their validity (logically/geometrically) and updates the cache of invalid modes.
        Args:
            prev_mode (Mode): Previous mode of all modes in the list.
            modes (List[Mode]): List of modes to be validated.
        Returns:
            List[Mode]: List of valid modes.
        """
        if not self.apply:
            return list(modes)

        self.counter += 1
        # TODO what if its a Goal Region? Does that matter?
        # TODO: the start position is not necessarily always free
        start_pos = self.env.start_pos.state()
        valid_modes = []
        for mode in modes:
            if mode in self.blacklist_modes:
                self.add_invalid_mode(mode)
                continue

            if mode in self.whitelist_modes:
                valid_modes.append(mode)
                continue

            if mode == self.env.start_mode:
                self.whitelist_modes.add(mode)
                valid_modes.append(mode)
                continue

            possible_next_task_combinations = self.env.get_valid_next_task_combinations(
                mode
            )
            if not possible_next_task_combinations and not self.env.is_terminal_mode(
                mode
            ):
                self.add_invalid_mode(mode)
                return []

            is_in_collision = False
            whitelist_robots = set()
            set_mode = True
            for next_ids in possible_next_task_combinations:
                active_task = self.env.get_active_task(mode, next_ids)
                constrained_robots = active_task.robots
                if len(whitelist_robots) == len(self.env.robots):
                    break

                if all(elem in whitelist_robots for elem in constrained_robots):
                    continue

                goal = active_task.goal.sample(mode)

                for robot in self.env.robots:
                    if robot in constrained_robots:
                        q = start_pos * 1
                        end_idx = 0
                        for r in constrained_robots:
                            robot_indices = self.env.robot_idx[r]
                            dim = self.env.robot_dims[r]
                            indices = list(range(end_idx, end_idx + dim))
                            if self.with_noise:
                                q[robot_indices] = goal[indices] + np.random.normal(
                                    loc=0.0, scale=0.03, size=goal[indices].shape
                                )
                            else:
                                q[robot_indices] = goal[indices]

                            end_idx += dim
                        # checks if the mode has a possible goal configuration
                        # if not self.env.is_collision_free_for_robot(robot, q, mode, self.env.collision_tolerance, set_mode):
                        # assert(self.env.is_collision_free_np(q, mode, self.env.collision_tolerance, set_mode) == self.env.is_collision_free_for_robot(robot, q, mode, self.env.collision_tolerance, set_mode)),(
                        #     "ghjkl"
                        # )
                        if not self.env.is_collision_free_for_robot(
                            robot, q, mode, collision_tolerance=None, set_mode=set_mode
                        ):
                            self.add_invalid_mode(mode)
                            # when one task in mode cannot be reached -> it can never be reached later having this mode sequence (remove mode completely)
                            # print("AAA")
                            # self.env.show()
                            is_in_collision = True
                            break
                        # self.env.show()

                        whitelist_robots.add(robot)
                        set_mode = False
                if is_in_collision:
                    break

            if is_in_collision:
                continue

            valid_modes.append(mode)
            self.whitelist_modes.add(mode)

        return valid_modes

    def get_valid_next_ids(self, mode: Mode) -> List[int] | None:
        """
        Retrieves valid combination of next task IDs for given mode.

        Args:
            mode (Mode): Current operational mode for which to retrieve valid next task ID combinations.

        Returns:
            Optional[List[int]]: Randomly selected valid combination of all next task ID combinations if available
        """

        possible_next_task_combinations = self.env.get_valid_next_task_combinations(
            mode
        )
        if not self.apply:
            if not possible_next_task_combinations:
                return None

            return random.choice(possible_next_task_combinations)

        # if we have no possible task combinations
        if not possible_next_task_combinations:
            # and the mode is not terminal -> add the mode to the invalid ones
            if not self.env.is_terminal_mode(mode):
                self.propagate_invalid(mode)
            return None

        invalid_next_modes = self.invalid_next_ids.get(mode, set())

        while True:
            next_task = random.choice(possible_next_task_combinations)
            if tuple(next_task) in invalid_next_modes:
                continue
            return next_task

    def add_invalid_mode(self, mode: Mode) -> None:
        """
        Add invalid mode to the blacklist of the previous mode and mark it as an invalid descendant of the previous mode.

        Args:
            mode (Mode): Current operational mode.

        Returns:
            None: This method does not return any value.
        """
        if not self.apply:
            return

        # if the mode that we ar trying to add is the start mode, we dont
        if mode == self.env.start_mode:
            # self.env.C.view(True)
            assert False, "Tried to add initial mode to the invalid modes."
            return

        self.blacklist_modes.add(mode)
        # print("added to blacklist")
        if mode.prev_mode not in self.invalid_next_ids:
            self.invalid_next_ids[mode.prev_mode] = set()
        self.invalid_next_ids[mode.prev_mode].add(tuple(mode.task_ids))

        # assert False

    def propagate_invalid(self, mode: Mode) -> None:
        """Walk backwards from mode, blacklisting any mode that has no remaining valid
        next-task combinations. Stops as soon as a mode still has feasible successors."""
        if not self.apply:
            return

        while True:
            invalid_next_ids = self.invalid_next_ids.get(mode, set())
            possible_next_task_combinations = self.env.get_valid_next_task_combinations(mode)
            # if there are more possible task combinations than invalid ones, a feasible
            # combination still exists — stop propagating.
            if len(invalid_next_ids) < len(possible_next_task_combinations):
                break
            self.add_invalid_mode(mode)
            if mode == self.env.start_mode:
                break
            mode = mode.prev_mode

    def remove_invalid_modes(self, modes: List[Mode]) -> List[Mode]:
        """Return a filtered copy of modes with all blacklisted modes removed."""
        return [m for m in modes if m not in self.blacklist_modes]
