from typing import (
    List, Tuple
)
from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseProblem,
    Mode,
)
import random
import numpy as np

class ModeValidation():

    def __init__(
        self,
        env: BaseProblem,
        apply:bool,
        with_noise:bool = False,
    ):
        self.env = env
        self.blacklist_modes = set()
        self.whitelist_modes = set()
        self.invalid_next_ids = {}
        self.counter = 0
        self.apply = apply
        self.with_noise = with_noise

    def get_valid_modes(self, prev_mode:Mode, modes:tuple[Mode]) -> List[Mode]:
        """
        Filters a list of modes based on their validity (logically/geometrically) and updates the cache of invalid modes.
        Args:
            prev_mode (Mode): Previous mode of all modes in the list.
            modes (tuple[Mode]): List of modes to be validated.
        Returns:
            List[Mode]: List of valid modes.
        """
        if not self.apply:
            return list(modes)
        self.counter += 1
        #TODO what if its a Goal Region? Does that matter?
        start_pos = self.env.start_pos.state()        
        valid_modes = []
        for mode in modes:
            if mode in self.blacklist_modes:
                self.update_cache_of_invalid_modes(mode)
                continue

            if mode in self.whitelist_modes:
                valid_modes.append(mode)
                continue
            
            if mode.prev_mode is None:
                self.whitelist_modes.add(mode)
                valid_modes.append(mode)
                continue
            
            possible_next_task_combinations = self.env.get_valid_next_task_combinations(mode)
            if not possible_next_task_combinations and not self.env.is_terminal_mode(mode):
                self.update_cache_of_invalid_modes(mode)
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
                end_idx = 0
            
                for robot in self.env.robots:
                    if robot in constrained_robots:
                        q = start_pos *1
                        robot_indices = self.env.robot_idx[robot]
                        dim = self.env.robot_dims[robot]
                        indices = list(range(end_idx, end_idx + dim))
                        if self.with_noise:
                            q[robot_indices] = goal[indices]+ np.random.normal(loc=0.0, scale=0.03, size=goal[indices].shape)
                        else:
                            q[robot_indices] = goal[indices]

                        # 
                        end_idx += dim 
                        # checks if the mode has a possible goal configuration
                        # if not self.env.is_collision_free_for_robot(robot, q, mode, self.env.collision_tolerance, set_mode):
                        # assert(self.env.is_collision_free_np(q, mode, self.env.collision_tolerance, set_mode) == self.env.is_collision_free_for_robot(robot, q, mode, self.env.collision_tolerance, set_mode)),(
                        #     "ghjkl"
                        # )
                        if not self.env.is_collision_free_for_robot(robot, q, mode, collision_tolerance=None, set_mode=set_mode):
                            self.update_cache_of_invalid_modes(mode)
                            #when one task in mode cannot be reached -> it can never be reached later having this mode sequence (remove mode completely)
                            is_in_collision = True
                            break

                        whitelist_robots.add(robot)
                        set_mode = False
                if is_in_collision:
                    break

            if is_in_collision:
                continue
            
            valid_modes.append(mode)
            self.whitelist_modes.add(mode)
   
        return valid_modes
    
    def get_valid_next_ids(self, mode:Mode) -> List[int]:
        """
        Retrieves valid combination of next task IDs for given mode.

        Args:
            mode (Mode): Current operational mode for which to retrieve valid next task ID combinations.

        Returns:
            Optional[List[int]]: Randomly selected valid combination of all next task ID combinations if available
        """

        possible_next_task_combinations = self.env.get_valid_next_task_combinations(mode)
        if not self.apply:
            if not possible_next_task_combinations:
                return
            return random.choice(possible_next_task_combinations)
        if not possible_next_task_combinations:
            if not self.env.is_terminal_mode(mode):
                _ = self.track_invalid_modes(mode)
            return
        invalid_next_modes = self.invalid_next_ids.get(mode, set())
        while True:
            next_task = random.choice(possible_next_task_combinations)
            if tuple(next_task) in invalid_next_modes:
                continue
            return next_task

    def update_cache_of_invalid_modes(self, mode:Mode) -> None:
        """
        Add invalid mode to the blacklist of the previous mode and mark it as an invalid descendant of the previous mode.

        Args:
            mode (Mode): Current operational mode.

        Returns:
            None: This method does not return any value.
        """
        if not self.apply:
            return
        if mode.prev_mode is None:
            return
        self.blacklist_modes.add(mode)
        if mode.prev_mode not in self.invalid_next_ids:
            self.invalid_next_ids[mode.prev_mode] = set()
        self.invalid_next_ids[mode.prev_mode].add(tuple(mode.task_ids))      

    def track_invalid_modes(self, mode:Mode, modes:List[Mode] | None = None) -> Tuple[List[Mode], bool]:
        """
        Tracks invalid modes by adding them to blacklist and removing them from the list.

        Args:
            mode (Mode): The mode to be tracked as invalid.

        Returns:
            None: This method does not return any value.
        """
        if not self.apply:
            return modes, False
        if mode.id == 77:
            pass
        updated = False
        while True: 
            invalid_next_ids = self.invalid_next_ids.get(mode, set())
            possible_next_task_combinations = self.env.get_valid_next_task_combinations(mode)
            if len(invalid_next_ids) < len(possible_next_task_combinations):
                break
            updated = True
            modes.remove(mode)
            self.update_cache_of_invalid_modes(mode)
            if mode.prev_mode is None:
                break
            mode = mode.prev_mode
        return modes, updated