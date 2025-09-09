import numpy as np
import time as time
import math as math
from typing import Tuple, List, Dict, Any
from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
    Mode,
)
from .rrtstar_base import (
    BaseRRTConfig,
    BaseRRTstar,
    Node,
    SingleTree,
    save_data,
)
from .termination_conditions import (
    PlannerTerminationCondition,
)


class RRTstar(BaseRRTstar):
    """Represents the class for the RRT* based planner"""

    def __init__(self, env: BaseProblem, config: BaseRRTConfig):
        super().__init__(env=env, config=config)

    def update_cost(self, mode: Mode, n: Node) -> None:
        stack = [n]
        while stack:
            current_node = stack.pop()
            children = current_node.children
            if children:
                for _, child in enumerate(children):
                    child.cost = current_node.cost + child.cost_to_parent
                    # child.agent_dists = current_node.agent_dists + child.agent_dists_to_parent
                stack.extend(children)

    def manage_transition(self, mode: Mode, n_new: Node) -> None:
        # check if transition is reached
        if self.env.is_transition(n_new.state.q, mode):
            self.save_tree_data()
            self.add_new_mode(n_new.state.q, mode, SingleTree)
            self.save_tree_data()
            self.convert_node_to_transition_node(mode, n_new)
        # check if termination is reached
        if self.env.done(n_new.state.q, mode):
            self.convert_node_to_transition_node(mode, n_new)
            if not self.operation.init_sol:
                self.operation.init_sol = True
        self.find_lb_transition_node()

    def initialize_planner(self) -> None:
        self.set_gamma_rrtstar()
        # Initilaize first Mode
        self.add_new_mode(tree_instance=SingleTree)
        active_mode = self.modes[-1]

        # Create start node
        start_state = State(self.env.start_pos, active_mode)
        start_node = Node(start_state, self.operation)
        self.trees[active_mode].add_node(start_node)
        start_node.cost = 0.0
        start_node.cost_to_parent = 0.0
        # in case a dummy start is defined
        self.manage_transition(active_mode, start_node)

    def save_tree_data(self) -> None:
        if not self.config.with_tree_visualization:
            return
        data = {}
        data["all_nodes"] = [
            self.trees[m].subtree[id].state.q.state()
            for m in self.modes
            for id in self.trees[m].get_node_ids_subtree()
        ]

        try:
            data["all_transition_nodes"] = [
                self.trees[m].subtree[id].state.q.state()
                for m in self.modes
                for id in self.transition_node_ids[m]
            ]
            data["all_transition_nodes_mode"] = [
                self.trees[m].subtree[id].state.mode.task_ids
                for m in self.modes
                for id in self.transition_node_ids[m]
            ]
        except Exception:
            data["all_transition_nodes"] = []
            data["all_transition_nodes_mode"] = []
        data["all_nodes_mode"] = [
            self.trees[m].subtree[id].state.mode.task_ids
            for m in self.modes
            for id in self.trees[m].get_node_ids_subtree()
        ]

        for i, type in enumerate(["forward", "reverse"]):
            data[type] = {}
            data[type]["nodes"] = []
            data[type]["parents"] = []
            data[type]["modes"] = []
            for m in self.modes:
                for id in self.trees[m].get_node_ids_subtree():
                    node = self.trees[m].subtree[id]
                    data[type]["nodes"].append(node.state.q.state())
                    data[type]["modes"].append(node.state.mode.task_ids)
                    parent = node.parent
                    if parent is not None:
                        data[type]["parents"].append(parent.state.q.state())
                    else:
                        data[type]["parents"].append(None)
            break
        data["pathnodes"] = []
        data["pathparents"] = []
        if self.operation.path_nodes is not None:
            for node in self.operation.path_nodes:
                data["pathnodes"].append(node.state.q.state())
                parent = node.parent
                if parent is not None:
                    data["pathparents"].append(parent.state.q.state())
                else:
                    data["pathparents"].append(None)

        save_data(data, True)

    def plan(
        self,
        ptc: PlannerTerminationCondition,
        optimize: bool = True,
    ) -> Tuple[List[State] | None, Dict[str, Any]]:
        i = 0
        self.initialize_planner()
        while True:
            i += 1
            # Mode selection
            active_mode = self.random_mode()
            # RRT* core
            q_rand = self.sample_configuration(active_mode)
            if not q_rand:
                continue
            n_nearest, dist, set_dists, n_nearest_idx = self.nearest(
                active_mode, q_rand
            )
            state_new = self.steer(active_mode, n_nearest, q_rand, dist)
            if not state_new:
                continue
            if self.env.is_collision_free(
                state_new.q, active_mode
            ) and self.env.is_edge_collision_free(
                n_nearest.state.q, state_new.q, active_mode
            ):
                n_new = Node(state_new, self.operation)
                if np.equal(n_new.state.q.state(), q_rand.state()).all():
                    N_near_batch, n_near_costs, node_indices = self.near(
                        active_mode, n_new, n_nearest_idx, set_dists
                    )
                else:
                    N_near_batch, n_near_costs, node_indices = self.near(
                        active_mode, n_new, n_nearest_idx
                    )
                batch_cost = self.env.batch_config_cost(n_new.state.q, N_near_batch)
                self.find_parent(
                    active_mode,
                    node_indices,
                    n_new,
                    n_nearest,
                    batch_cost,
                    n_near_costs,
                )
                if self.rewire(
                    active_mode, node_indices, n_new, batch_cost, n_near_costs
                ):
                    self.update_cost(active_mode, n_new)
                self.manage_transition(active_mode, n_new)

            if not optimize and self.operation.init_sol:
                self.save_tree_data()
                break

            if ptc.should_terminate(i, time.time() - self.start_time):
                print("Number of iterations: ", i)
                break

        self.update_results_tracking(self.operation.cost, self.operation.path)
        info = {"costs": self.costs, "times": self.times, "paths": self.all_paths}
        # print('Path is collision free:', self.env.is_path_collision_free(self.operation.path))

        # ensure that the mode-switch nodes are there once in every mode.
        path_w_doubled_modes = []
        for i in range(len(self.operation.path)):
            path_w_doubled_modes.append(self.operation.path[i])

            if i + 1 < len(self.operation.path) and self.operation.path[i].mode != self.operation.path[i + 1].mode:
                path_w_doubled_modes.append(State(self.operation.path[i].q, self.operation.path[i + 1].mode))

        self.operation.path = path_w_doubled_modes

        return self.operation.path, info
