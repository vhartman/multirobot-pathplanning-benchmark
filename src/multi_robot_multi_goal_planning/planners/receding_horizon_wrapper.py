from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
import copy
import time

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
from multi_robot_multi_goal_planning.problems.util import path_cost


@dataclass
class RecedingHorizonConfig:
    low_level_solver: str = "composite_prm"
    horizon_length: int = 2
    execution_length: int = 1
    low_level_max_time: float = 10
    constrain_free_robots_to_home: bool = True


class RecedingHorizonPlanner(BasePlanner):
    """
    Receding horizon planner wrapper.
    Mostly for tests how suboptimal a solution is if we only consider parts of the whole plan.

    Takes a planner, and only runs it on a subsequence of the whole planning sequence.
    'Executes" part of the solution, and continues from there.
    """

    def construct_planner(self, env: BaseProblem) -> BasePlanner:
        planner = CompositePRM(env)
        return planner

    def make_short_horizon_env(
        self, seq_start_idx, start_pos: Configuration, path
    ) -> BaseProblem:
        print()
        print("Constructing short horizon env")
        print(f"Start index {seq_start_idx}")

        # What this does:
        # - copy the env
        # - rewrite the sequence
        # - go through the path and add all the modes so far via get next mode

        short_horizon_env = copy.deepcopy(self.base_env)

        final_task_index = min(
            len(self.base_env.sequence), seq_start_idx + self.config.horizon_length
        )

        original_named_sequence = [
            self.base_env.tasks[idx].name for idx in self.base_env.sequence
        ]

        short_horizon_env.start_pos = copy.deepcopy(start_pos)

        if final_task_index <  len(self.base_env.sequence):
            if self.config.constrain_free_robots_to_home:
                goal_pose = self.base_env.start_pos.state() * 1.0

                # get active robot, set it to the correct pose
                task_idx = self.base_env.sequence[final_task_index - 1]
                final_task = self.base_env.tasks[task_idx]
                final_task_goal = final_task.goal

                # this could be adapted to a goal_set
                final_task_pose = final_task_goal.sample(None) * 1.0

                constrained_robots = final_task.robots

                offset = 0
                goal_sample_offset = 0
                for r in self.base_env.robots:
                    dim = self.base_env.robot_dims[r]
                    if r in constrained_robots:
                        goal_pose[offset : offset + dim] = final_task_pose[
                            goal_sample_offset : goal_sample_offset + dim
                        ]
                        goal_sample_offset += dim
                    offset += dim

                new_terminal_task = Task(self.base_env.robots, SingleGoal(goal_pose))
            else:
                goal_region = self.base_env.limits
                # get active robot, set it to the correct pose
                task_idx = self.base_env.sequence[final_task_index - 1]
                final_task = self.base_env.tasks[task_idx]
                final_task_goal = final_task.goal

                # this could be adapted to a goal_set
                final_task_pose = final_task_goal.sample(None) * 1.0

                constrained_robots = final_task.robots

                offset = 0
                for r in self.base_env.robots:
                    if r in constrained_robots:
                        dim = self.base_env.robot_dims[r]
                        goal_region[0, offset : offset + dim] = final_task_pose
                        goal_region[1, offset : offset + dim] = final_task_pose

                new_terminal_task = Task(self.base_env.robots, GoalRegion(goal_region))

            short_horizon_env.tasks.append(new_terminal_task)
            short_horizon_env.tasks[-1].name = f"dummy_terminal_{seq_start_idx}"
        
        short_horizon_sequence = original_named_sequence[
            0:final_task_index
        ]
        if final_task_index <  len(self.base_env.sequence):
            short_horizon_sequence[-1] = f"dummy_terminal_{seq_start_idx}"

        short_horizon_env.sequence = short_horizon_env._make_sequence_from_names(
            short_horizon_sequence
        )

        curr_mode = short_horizon_env.make_start_mode()

        for i in range(1, len(path)):
            if path[i].mode != path[i - 1].mode:
                next_mode = short_horizon_env.get_next_modes(
                    path[i - 1].q, curr_mode
                )[0]
                curr_mode = next_mode
                short_horizon_env.set_to_mode(next_mode)

            elif i == len(path) - 1:
                next_mode = short_horizon_env.get_next_modes(path[i].q, curr_mode)[
                    0
                ]
                curr_mode = next_mode
                short_horizon_env.set_to_mode(next_mode)

        short_horizon_env.start_mode = curr_mode

        if final_task_index <  len(self.base_env.sequence):
            short_horizon_env._terminal_task_ids = [
                len(short_horizon_env.tasks) - 1
            ] * len(self.base_env.robots)
            
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

        if not isinstance(self.base_env, SequenceMixin):
            assert False, "Needs to be sequence task spec."

        complete_plan = []
        current_task_index = 0
        start_mode = None

        start_time = time.time()
        while True:
            # make short horizon env
            # get index in plan where we at
            if len(complete_plan) == 0:
                start_pos = copy.deepcopy(self.base_env.start_pos)
            else:
                idx = 0
                start_pos = None
                for i in range(1, len(complete_plan)):
                    # print(complete_plan[i].mode)
                    if complete_plan[i].mode != complete_plan[i - 1].mode:
                        idx += 1

                    if idx == current_task_index:
                        start_pos = copy.deepcopy(complete_plan[i].q)
                        break

                if start_pos is None:
                    start_pos = copy.deepcopy(complete_plan[-1].q)

            sh_env = self.make_short_horizon_env(
                current_task_index, start_pos, complete_plan
            )

            short_horizon_planner = self.construct_planner(sh_env)

            inner_ptc = RuntimeTerminationCondition(self.config.low_level_max_time)
            short_horizon_plan, _ = short_horizon_planner.plan(inner_ptc, optimize=True)

            if short_horizon_plan is None:
                print("Was not able to find a path.")
                return None, {}

            sh_env.sequence = self.base_env.sequence

            if self.base_env.is_terminal_mode(short_horizon_plan[-1].mode):
                complete_plan.extend(short_horizon_plan)
            else:
                # print("overwritten")
                idx = 0
                if start_mode is None:
                    curr_mode = self.base_env.start_mode
                else:
                    curr_mode = start_mode
                for i in range(0, len(short_horizon_plan)):
                    complete_plan.append(copy.deepcopy(short_horizon_plan[i]))
                    complete_plan[-1].mode = copy.deepcopy(curr_mode)

                    if i == 0:
                        continue

                    if idx == self.config.execution_length:
                        break

                    # print(short_horizon_plan[i].mode)
                    if not self.base_env.is_terminal_mode(short_horizon_plan[i].mode):
                        if short_horizon_plan[i].mode != short_horizon_plan[i - 1].mode:
                            idx += 1
                            next_mode = self.base_env.get_next_modes(
                                short_horizon_plan[i - 1].q, curr_mode
                            )[0]
                            start_mode = next_mode
                            curr_mode = next_mode

                        elif i == len(short_horizon_plan) - 1:
                            idx += 1
                            next_mode = self.base_env.get_next_modes(
                                short_horizon_plan[i].q, curr_mode
                            )[0]
                            start_mode = next_mode
                            curr_mode = next_mode

           
            # env_cpy = copy.deepcopy(self.base_env)
            # env_cpy.display_path(complete_plan)

            if self.base_env.is_terminal_mode(complete_plan[-1].mode):
                break

            current_task_index += self.config.execution_length

        print("Finished planning")
        print()

        costs = [path_cost(complete_plan, self.base_env.batch_config_cost)]
        times = [time.time() - start_time]
        info = {"costs": costs, "times": times, "paths": [complete_plan]}
        return complete_plan, info
