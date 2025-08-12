import numpy as np

from typing import List, Dict, Tuple, Optional, Union, Any, Set
import time
import math
from itertools import chain
from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
)
from multi_robot_multi_goal_planning.problems.configuration import (
    batch_config_dist,
)
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
)
from multi_robot_multi_goal_planning.planners.itstar_base import (
    BaseITConfig,
    BaseITstar,
    BaseGraph,
    BaseOperation,
    BaseNode,
    BaseTree,
)
from functools import cache
from multi_robot_multi_goal_planning.planners.planner_aitstar import EdgeQueue


class InadmissibleHeuristics:
    """Stores and updates inadmissible heuristic estimates used during reverse search."""

    __slots__ = [
        "lb_cost_to_go",
        "effort",
    ]

    lb_cost_to_go: float
    effort: float

    def __init__(self):
        self.lb_cost_to_go = np.inf
        self.effort = np.inf

    def update(self, pot_parent: "Node", edge_cost: float, edge_effort: float) -> None:
        """
        Updates inadmissible heuristic values using a potential parent node and associated edge cost and effort.

        Args:
            pot_parent (Node): The potential parent node from which the update is derived.
            edge_cost (float): The cost associated with the edge to the current node.
            edge_effort (float): The effort associated with the edge to the current node.

        Returns:
            None: This method does not return any value.
        """
        self.lb_cost_to_go = min(
            self.lb_cost_to_go, pot_parent.inad.lb_cost_to_go + edge_cost
        )
        new_effort = pot_parent.inad.effort + edge_effort
        if new_effort < self.effort:
            self.effort = new_effort

    def reset_goal_nodes(self) -> None:
        """
        Resets the heuristic values for goal nodes to zero.

        Args:
            None

        Returns:
            None: This method does not return any value.
        """
        self.lb_cost_to_go = 0.0
        self.effort = 0.0

    def reset(self) -> None:
        """
        Resets the heuristic values to infinity.

        Args:
            None

        Returns:
            None: This method does not return any value.
        """
        self.lb_cost_to_go = np.inf
        self.effort = np.inf


class Graph(BaseGraph):
    def __init__(
        self,
        root_state,
        operation,
        distance_metric,
        batch_dist_fun,
        batch_cost_fun,
        is_edge_collision_free,
        get_next_modes,
        collision_resolution,
        node_cls,
    ):
        super().__init__(
            root_state=root_state,
            operation=operation,
            distance_metric=distance_metric,
            batch_dist_fun=batch_dist_fun,
            batch_cost_fun=batch_cost_fun,
            is_edge_collision_free=is_edge_collision_free,
            get_next_modes=get_next_modes,
            collision_resolution=collision_resolution,
            node_cls=node_cls,
            including_effort=True,
        )

        self.reverse_queue = None
        self.effort_estimate_queue = None
        self.cost_bound_queue = None
        self.epsilon = np.inf

    def reset_reverse_tree(self) -> None:
        [
            (
                setattr(self.nodes[node_id], "lb_cost_to_go", math.inf),
                self.nodes[node_id].rev.reset(),
                self.nodes[node_id].inad.reset(),
            )
            for node_id in list(chain.from_iterable(self.node_ids.values()))
        ]
        [
            (
                setattr(self.nodes[node_id], "lb_cost_to_go", math.inf),
                self.nodes[node_id].rev.reset(),
                self.nodes[node_id].inad.reset(),
            )
            for node_id in list(chain.from_iterable(self.transition_node_ids.values()))
        ]  # also includes goal nodes
        [
            (
                setattr(self.nodes[node_id], "lb_cost_to_go", math.inf),
                self.nodes[node_id].rev.reset(),
                self.nodes[node_id].inad.reset(),
            )
            for node_id in list(
                chain.from_iterable(self.reverse_transition_node_ids.values())
            )
        ]

    def reset_all_goal_nodes_lb_costs_to_go(self) -> None:
        """
        Sets the lower bound cost-to-go of all goal nodes (and virtual goal nodes, if present) to zero,
        resets their reverse cost-to-parent, and resets their inadmissible heuristic attributes.

        Args:
            None

        Returns:
            None: This method does not return any value.
        """
        for node in self.goal_nodes:
            node.lb_cost_to_go = 0
            node.rev.cost_to_parent = 0
            node.inad.reset_goal_nodes()
        # if long horizon planning
        if self.virtual_goal_nodes:
            for node in self.virtual_goal_nodes:
                node.lb_cost_to_go = 0
                node.rev.cost_to_parent = 0
                node.inad.reset_goal_nodes()

    def update_forward_queue(
        self, edge_cost: float, edge: Tuple["Node", "Node"], edge_effort: float
    ) -> None:
        item = (edge_cost, edge, edge_effort)
        self.effort_estimate_queue.heappush(item)
        self.cost_bound_queue.heappush(item)
        # self.g.cost_estimate_queue.heappush((edge_cost, edge))

    def update_forward_queue_keys(
        self, type: str, node_ids: Optional[Set["Node"]] = None
    ) -> None:
        if node_ids is not None and len(node_ids) == 0:
            return
        # if node_ids is not None:
        #     self.effort_estimate_queue.update(node_ids, type)
        # self.cost_estimate_queue.update(node_ids, type)
        self.cost_bound_queue.update(node_ids, type)
        if type == "target":
            self.effort_estimate_queue.update(node_ids, type)

    def update_reverse_queue_keys(
        self, type: str, node_ids: Optional[Set["Node"]] = None
    ) -> None:
        if node_ids is not None and len(node_ids) == 0:
            return
        self.reverse_queue.update(node_ids, type)

    def remove_forward_queue(
        self, edge_cost: float, n0: "Node", n1: "Node", edge_effort: float
    ) -> None:
        item1 = (edge_cost, (n1, n0), edge_effort)
        item2 = (edge_cost, (n0, n1), edge_effort)
        self.cost_bound_queue.remove(item1)
        self.effort_estimate_queue.remove(item1)
        # self.g.cost_estimate_queue.remove((edge_cost, (n1, n0)))
        self.cost_bound_queue.remove(item2)
        self.effort_estimate_queue.remove(item2)
        # self.g.cost_estimate_queue.remove((edge_cost, (n0, n1)))

    def get_best_forward_edge(self):
        key_cb, item_cb = self.cost_bound_queue.peek_first_element()
        bound = key_cb * self.epsilon
        _, item_ee = self.effort_estimate_queue.peek_first_element()
        # if np.isinf(item_cb[1][1].lb_cost_to_go):
        # assert(np.isinf(key_cb)), (
        #     "key_cb is not inf"
        # )
        item = item_cb
        if np.isinf(self.epsilon):
            key_ee = self.cost_bound_queue.key(item_ee)
            if key_ee <= bound:
                item = item_ee

        self.effort_estimate_queue.remove(item)
        self.cost_bound_queue.remove(item)

        return item


class Operation(BaseOperation):
    def __init__(self) -> None:
        super().__init__()

    def update(
        self,
        node: "Node",
        lb_cost_to_go: float = np.inf,
        cost: float = np.inf,
        lb_cost_to_come: float = np.inf,
    ) -> None:
        node.lb_cost_to_go = lb_cost_to_go
        self.lb_costs_to_come = self.ensure_capacity(self.lb_costs_to_come, node.id)
        node.lb_cost_to_come = lb_cost_to_come
        self.costs = self.ensure_capacity(self.costs, node.id)
        node.cost = cost


class Node(BaseNode):
    __slots__ = ["inad", "lb_effort_to_come"]

    inad: InadmissibleHeuristics
    lb_effort_to_come: float

    def __init__(
        self, operation: "Operation", state: "State", is_transition: bool = False
    ) -> None:
        super().__init__(operation, state, is_transition)
        self.inad = InadmissibleHeuristics()
        self.lb_effort_to_come = np.inf

    def close(self, resolution: Optional[float] = None) -> None:
        self.inad.lb_cost_to_go = self.lb_cost_to_go
        self.inad.effort = (
            self.rev.parent.inad.effort + self.rev.cost_to_parent / resolution
        )

    def set_to_goal_node(self) -> None:
        self.lb_cost_to_go = 0.0
        self.inad.reset_goal_nodes()


class ReverseQueue(EdgeQueue):
    def __init__(self, alpha=1.0, collision_resolution: Optional[float] = None):
        super().__init__(alpha, collision_resolution)

    def key(self, item: Tuple[Any]) -> Tuple[float, float]:
        # item[0]: edge_cost from n0 to n1
        # item[1]: edge (n0, n1)
        # item[2]: edge effort from n0, n1

        start_node, target_node = item[1]
        return (
            start_node.inad.lb_cost_to_go + item[0] + target_node.lb_cost_to_come,
            start_node.inad.effort + item[2] + target_node.lb_effort_to_come,
        )


class EffortEstimateQueue(EdgeQueue):
    def __init__(self, alpha=1.0, collision_resolution: Optional[float] = None):
        super().__init__(alpha, collision_resolution)
        self.cost_bound = {}

    # remaining effort through an edge
    def key(self, item: Tuple[Any]) -> float:
        # item[0]: edge cost from n0 to n1
        # item[1]: edge (n0, n1)
        # item[2]: edge effort from n0, n1
        return item[1][1].inad.effort + item[2]

    def add_and_sync(self, item: Tuple[Any]) -> None:
        target_node_id = item[1][1].id

        self.target_nodes.add(target_node_id)
        self.target_nodes_with_item.setdefault(target_node_id, set()).add(item)

    def remove(self, item: Tuple[Any], in_current_entries: bool = False) -> None:
        if not in_current_entries and item not in self.current_entries:
            return
        target_node_id = item[1][1].id
        self.target_nodes_with_item[target_node_id].remove(item)
        del self.current_entries[item]

        if not self.target_nodes_with_item[target_node_id]:
            self.target_nodes.discard(target_node_id)


class CostBoundQueue(EdgeQueue):
    def key(self, item: Tuple[Any]) -> float:
        # item[0]: edge_cost from n0 to n1
        # item[1]: edge (n0, n1)
        start_node, target_node = item[1]
        return start_node.cost + item[0] + target_node.lb_cost_to_go


# class CostEstimateQueue(EdgeQueue):
#     def key(self, item: Tuple[Any]) -> float:
#         # item[0]: edge_cost from n0 to n1
#         # item[1]: edge (n0, n1)
#         return (item[1][0].cost + item[0] + item[1][1].inad.lb_cost_to_go)

# class LongHorizon(BaseLongHorizon):
#     def __init__(self):
#         super().__init__()

#     def


class EITstar(BaseITstar):
    """Represents the class for the EIT* based planner"""

    def __init__(self, env: BaseProblem, config: BaseITConfig):
        super().__init__(env=env, config=config)

        self.sparse_number_of_points = 2
        self.reverse_tree_set = set()
        self.reduce_neighbors = False
        self.sparesly_checked_edges = {}
        self.check = set()
        self.start_mode = None
        self.intermediate_end_mode = None

    def _create_operation(self) -> BaseOperation:
        return Operation()

    def _create_graph(self, root_state: State) -> BaseGraph:
        return Graph(
            root_state=root_state,
            operation=self.operation,
            distance_metric=self.config.distance_metric,
            batch_dist_fun=lambda a, b, c=None: batch_config_dist(
                a, b, c or self.config.distance_metric
            ),
            batch_cost_fun=lambda a, b: self.env.batch_config_cost(a, b),
            is_edge_collision_free=self.env.is_edge_collision_free,
            get_next_modes=self.env.get_next_modes,
            collision_resolution=self.env.collision_resolution,
            node_cls=Node,
        )

    def continue_reverse_search(self, iter: int) -> bool:
        if len(self.g.reverse_queue) == 0 or len(self.g.cost_bound_queue) == 0:
            return False
        if iter > 0 and not self.updated_target_nodes:
            return True
        # update forward queue
        self.g.update_forward_queue_keys("target", self.updated_target_nodes)
        self.updated_target_nodes = set()
        if np.isinf(
            self.g.epsilon
        ) and not self.g.cost_bound_queue.target_nodes.isdisjoint(
            self.reverse_tree_set
        ):
            return False
        not_closed_nodes = (
            self.g.cost_bound_queue.target_nodes - self.reverse_closed_set
        )
        if len(not_closed_nodes) == 0:
            return False
        forward_key, forward_item = self.g.cost_bound_queue.peek_first_element()
        reverse_key, _ = self.g.reverse_queue.peek_first_element()
        if (
            forward_item[1][1].id in self.reverse_closed_set
            and forward_key <= reverse_key[0]
        ):
            return False
        return True

    def update_reverse_sets(self, node: Node) -> None:
        self.reverse_closed_set.add(node.id)
        self.reverse_tree_set.add(node.id)

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def expand_node_reverse(
        self, nodes: List[Node], first_search: bool = False
    ) -> None:
        """
        Expands nodes in reverse search and updates the reverse search tree.

        Args:
            nodes (List[Node]): A list of nodes to be expanded in reverse.
            first_search (bool, optional): Whether this is the initial reverse expansion. Defaults to False.

        Returns:
            None
        """
        g_nodes = self.g.nodes

        root_id = self.g.root.id
        # goal_node_ids = self.g.goal_node_ids
        
        goal_ids           = {n.id for n in self.g.goal_nodes}
        virtual_goal_ids   = {n.id for n in self.g.virtual_goal_nodes}

        skip_goal_ids = goal_ids | virtual_goal_ids | {root_id}

        # push = self.g.reverse_queue.heappush  # localize function ref

        if not self.dynamic_reverse_search_update:
            nodes_to_skip = self.reverse_closed_set | self.reverse_tree_set

        for node in nodes:
            if node.id == root_id or node == self.g.virtual_root:
                return
            
            self.reverse_tree_set.add(node.id)
            self.updated_target_nodes.add(node.id)
            neighbors = self.g.get_neighbors(
                node,
                space_extent=self.approximate_space_extent,
                first_search=first_search,
            )
            
            if neighbors.size == 0:
                return
            
            edge_costs = self.g.tot_neighbors_batch_cost_cache[node.id]
            edge_efforts = self.g.tot_neighbors_batch_effort_cache[node.id]
            # assert (len(edge_costs) == len(edge_efforts)), (
            #     "neighbors and edge_costs are not the same length"
            # )

            node_lb_cost_to_go = node.lb_cost_to_go
            node_rev_parent_id = (
                node.rev.parent.id if node.rev.parent is not None else None
            )
            # node_blacklist = node.blacklist
            node_is_transition = node.is_transition
            node_mode = getattr(node.state, "mode", None)

            if not self.dynamic_reverse_search_update:
                # nodes_to_skip = self.reverse_closed_set | self.reverse_tree_set
                mask_valid = np.array(
                    [nid not in nodes_to_skip for nid in neighbors], dtype=bool
                )
                # mask_valid = ~np.isin(neighbors, np.array(nodes_to_skip))

                # assert (mask_valid == mask_valid_new).all()

                neighbors = neighbors[mask_valid]
                edge_costs = edge_costs[mask_valid]
                edge_efforts = edge_efforts[mask_valid]

            # for id, edge_cost, edge_effort in zip(neighbors, edge_costs, edge_efforts):
            tmp = node.operation.lb_costs_to_come
            for i in range(len(neighbors)):
                id = neighbors[i]

                # if id == self.g.root.id:
                #     continue

                n = g_nodes[id]
                
                if n.is_transition:
                    if not n.is_reverse_transition:
                        continue
                    else:
                        if node_is_transition:
                            if node_mode != n.state.mode:
                                continue
                
                # if self.current_best_cost is None and edge in self.check:
                #     continue
                # if edge in self.check:
                # print(node.id, n.id)
                # assert (n.forward.parent == node) == (n.id in node.forward.children), (
                #         f"Parent and children don't coincide (reverse): parent: {node.id}, child: {n.id}"
                #         )
                if (
                    id in skip_goal_ids
                ):
                    continue
                # assert(id not in node_blacklist), (
                # "neighbors are wrong")

                if node_rev_parent_id == id:
                    continue

                edge_cost = edge_costs[i]

                if id in self.reverse_tree_set:
                    if n.lb_cost_to_go < node_lb_cost_to_go + edge_cost:
                        # assert(n.lb_cost_to_go- (node_lb_cost_to_go + edge_cost) < 1e-5), (
                        #     f"{id}, {node.id}, qwdfertzj"
                        # )
                        continue

                if self.current_best_cost is not None:
                    if (
                        # node_lb_cost_to_go + edge_cost + n.lb_cost_to_come
                        node_lb_cost_to_go + edge_cost + tmp[n.id]
                        > self.current_best_cost
                    ):
                        continue

                # edge_effort = edge_efforts[i]

                # edge = (node, n)
                self.g.reverse_queue.heappush((edge_cost, (node, n), edge_efforts[i]))
                # push((edge_cost, (node, n), edge_effort))

    def update_inflation_factor(self) -> None:
        """
        Updates the inflation factor (epsilon) used in reverse search.

        Args:
            None

        Returns:
            None
        """
        # if self.current_best_cost is None:
        if not self.dynamic_reverse_search_update:
            self.g.epsilon = np.inf
        else:
            self.g.epsilon = 1

    @cache
    def get_collision_start_and_end_index(self, sparse_N) -> Tuple[int, int]:
        """
        Computes the start and end index for collision checking based on the number of samples.

        Args:
            sparse_N (int): The number of samples along a path or edge.

        Returns:
            Tuple[int, int]: The start and end index for collision checking.
                - The start index for collision checking.
                - The end index (equal to sparse_N).
        """
        if sparse_N > 2:
            start = int(sparse_N / 2)
        else:
            start = 0
        return start, int(sparse_N)

    def reverse_search(self) -> None:
        """
        Performs the reverse search process, updating lower bound costs and tree structure.

        Args:
            None.

        Returns:
            None: This method does not return any value.
        """
        self.updated_target_nodes = set()
        iter = 0
        while self.continue_reverse_search(iter):
            iter += 1
            edge_cost, edge, edge_effort = self.g.reverse_queue.heappop()
            n0, n1 = edge

            # assert (n0.state.mode in self.long_horizon.mode_sequence), (
            #     "fghjklö"
            # )

            self.reverse_closed_set.add(n0.id)
            is_transition = n1.is_transition and n1.is_reverse_transition

            potential_lb_cost_to_go = n0.lb_cost_to_go + edge_cost
            if (
                is_transition
                and n1.transition_neighbors[0].lb_cost_to_go < potential_lb_cost_to_go
            ):
                # don't change the parent
                if (
                    self.config.apply_long_horizon
                    and n1.transition_neighbors[0].state.mode
                    not in self.long_horizon.mode_sequence
                ):
                    continue
                self.expand_node_reverse([n1.transition_neighbors[0]])
                continue

            if n0.id not in n1.whitelist:
                sparsely_collision_free = False
                if n0.id not in n1.blacklist:
                    n_start, n_end = (n0, n1) if n1.id > n0.id else (n1, n0)
                    N_start, N_max = self.get_collision_start_and_end_index(
                        self.sparse_number_of_points
                    )
                    N = max(2, int(edge_effort) + 1)

                    edge_id = (n_start.id, n_end.id)

                    previously_checked = edge_id in self.sparesly_checked_edges
                    valid_check = previously_checked and (
                        self.sparesly_checked_edges[edge_id] >= N_max
                    )
                    if valid_check:
                        sparsely_collision_free = True
                    else:
                        # if edge_id in self.check:
                        #     print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                        # self.check.add(edge_id)
                        sparsely_collision_free = self.env.is_edge_collision_free(
                            n_start.state.q,
                            n_end.state.q,
                            n_start.state.mode,
                            N_start=N_start,
                            N_max=N_max,
                            N=N,
                        )
                        self.sparesly_checked_edges[edge_id] = N_max

                if not sparsely_collision_free:
                    self.g.remove_forward_queue(edge_cost, n0, n1, edge_effort)
                    self.g.update_edge_collision_cache(n0, n1, False)
                    self.manage_edge_in_collision(n0, n1, clear=True)
                    continue

                if N_max >= N:
                    # checked it already with env resolution
                    self.g.update_edge_collision_cache(n0, n1, True)

            n1.inad.update(n0, edge_cost, edge_effort)

            if is_transition:
                # assert(len(n1.transition_neighbors) ==1), (
                #         "Transition node has more than one neighbor"
                #     )
                if self.config.with_tree_visualization and iter > 0:
                    self.save_tree_data((BaseTree.all_vertices, self.reverse_tree_set))
                n1.transition_neighbors[0].inad.update(n1, 0.0, 0.0)

            if n1.lb_cost_to_go > potential_lb_cost_to_go:
                self.g.update_connectivity(
                    n0, n1, edge_cost, potential_lb_cost_to_go, "reverse", is_transition
                )
                # assert (n1.lb_cost_to_go == n1.inad.lb_cost_to_go), (
                #     "ghjklö"
                # )
                # assert(n1.inad.effort != np.inf), (
                #     "effort is inf"
                # )
                if is_transition:
                    if self.config.with_tree_visualization and iter > 0:
                        self.save_tree_data(
                            (BaseTree.all_vertices, self.reverse_tree_set)
                        )
                    self.reverse_tree_set.add(n1.id)
                    if (
                        self.config.apply_long_horizon
                        and n1.transition_neighbors[0].state.mode
                        not in self.long_horizon.mode_sequence
                    ):
                        continue

                    self.expand_node_reverse([n1.transition_neighbors[0]])
                    continue

                self.expand_node_reverse([n1])

        if self.config.with_tree_visualization and iter > 0:
            self.save_tree_data((BaseTree.all_vertices, self.reverse_tree_set))

    def clear_reverse_edge_in_collision(self, n0: Node, n1: Node) -> None:
        """
        Handles the removal of a reverse edge due to a detected collision and updates affected nodes.

        Args:
            n0 (Node): One endpoint of the edge in collision.
            n1 (Node): The other endpoint of the edge in collision.

        Returns:
            None
        """
        if self.config.with_tree_visualization and (
            BaseTree.all_vertices or self.reverse_tree_set
        ):
            self.save_tree_data((BaseTree.all_vertices, self.reverse_tree_set))
        if n1.rev.parent is not None and n1.rev.parent.id == n0.id:
            nodes_to_update = self.update_reverse_node_of_children(n1)
        elif n0.rev.parent is not None and n0.rev.parent.id == n1.id:
            nodes_to_update = self.update_reverse_node_of_children(n0)
        self.g.update_forward_queue_keys("target", nodes_to_update)
        self.g.update_reverse_queue_keys("start", nodes_to_update)

        if self.config.with_tree_visualization and (
            BaseTree.all_vertices or self.reverse_tree_set
        ):
            self.save_tree_data((BaseTree.all_vertices, self.reverse_tree_set))

    def update_reverse_node_of_children(self, n: Node) -> set[int]:
        """
        Resets the reverse search state of a node and all its reverse children, and collects affected node IDs.

        Args:
            n (Node): The node whose associated reverse search structure and descendants are to be updated.

        Returns:
            set[int]: Set of node IDs that need their queue entries updated due to reverse tree modifications.
        """
        nodes_to_update = set()
        n.lb_cost_to_go = np.inf
        n.inad.reset()
        self.reverse_closed_set.discard(n.id)
        self.reverse_tree_set.discard(n.id)
        n.rev.parent.rev.children.remove(n.id)
        n.rev.parent.rev.fam.remove(n.id)
        if n.is_transition and n.transition_neighbors:
            # need to update the edges in the queue with the transition as well
            transition_node_ids = [n.id for n in n.transition_neighbors]
            nodes_to_update.update(transition_node_ids)
        nodes_to_update.add(n.id)
        if len(n.rev.children) == 0:
            n.rev.reset()
            return nodes_to_update
        stack = [n.id]
        while stack:
            current_id = stack.pop()
            current_node = self.g.nodes[current_id]
            children = current_node.rev.children
            current_node.rev.reset()
            if children:
                for _, id in enumerate(children):
                    child = self.g.nodes[id]
                    child.lb_cost_to_go = np.inf
                    if child in self.g.goal_nodes or child in self.g.virtual_goal_nodes:
                        child.lb_cost_to_go = 0.0
                    child.inad.reset()
                    self.reverse_closed_set.discard(child.id)
                    self.reverse_tree_set.discard(child.id)
                    nodes_to_update.add(child.id)
                stack.extend(children)
        return nodes_to_update

    def manage_edge_in_collision(self, n0: Node, n1: Node, clear: bool = False) -> None:
        """
        Handles reverse edge collisions by either clearing the edge or increasing resolution and reinitializing the search.

        Args:
            n0 (Node): One endpoint of the edge in collision.
            n1 (Node): The other endpoint of the edge in collision.
            clear (bool, optional): Whether to immediately clear the reverse edge. Defaults to False.

        Returns:
            None
        """
        if n0.rev.parent == n1 or n1.rev.parent == n0:
            if not self.dynamic_reverse_search_update or clear:
                self.clear_reverse_edge_in_collision(n0, n1)
                # self.initialize_reverse_search(False)
                return
            self.sparse_number_of_points *= 2
            # self.clear_reverse_edge_in_collision(n0, n1)
            self.initialize_reverse_search(False)
        # elif len(self.g.reverse_queue) == 0:
        #     self.initialize_reverse_search(False)

    def initialize_lb(self) -> None:
        self.g.compute_transition_lb_cost_to_come()
        self.g.compute_transition_lb_effort_to_come()
        self.g.compute_node_lb_to_come()

    def initialize_forward_search(self) -> None:
        self.update_inflation_factor()
        self.g.effort_estimate_queue = EffortEstimateQueue(
            collision_resolution=self.env.collision_resolution
        )
        self.g.cost_bound_queue = CostBoundQueue()
        # self.g.cost_estimate_queue = CostEstimateQueue()
        if self.config.apply_long_horizon:
            start = self.g.virtual_root
        else:
            if self.current_best_cost is None and self.g.virtual_root is not None:
                start = self.g.virtual_root
            else:
                start = self.g.root
        self.expand_node_forward(
            start, regardless_forward_closed_set=True, first_search=self.first_search
        )

    def initialize_reverse_search(self, reset: bool = True) -> None:
        if len(BaseTree.all_vertices) > 1:
            if self.config.with_tree_visualization and (
                BaseTree.all_vertices or self.reverse_tree_set
            ):
                self.save_tree_data((BaseTree.all_vertices, self.reverse_tree_set))
        self.reverse_closed_set = (
            set()
        )  # node was as start node visited in the reverse search
        self.reverse_tree_set = set()  # node was extended in the reverse search
        self.g.reverse_queue = ReverseQueue(
            collision_resolution=self.env.collision_resolution
        )
        self.g.reset_reverse_tree()
        self.g.reset_all_goal_nodes_lb_costs_to_go()
        self.updated_target_nodes = set()  # lb_cost_to_go was updated in reverse search
        if reset:
            self.sparse_number_of_points = 2
        if self.config.apply_long_horizon:
            goal_nodes = self.g.virtual_goal_nodes
        else:
            goal_nodes = self.g.goal_nodes
        self.expand_node_reverse(goal_nodes, first_search=self.first_search)
        self.g.update_forward_queue_keys("target")

    def plan(
        self,
        ptc: PlannerTerminationCondition,
        optimize: bool = True,
    ) -> Tuple[List[State] | None, Dict[str, Any]]:
        self.collision_cache = set()
        self.PlannerInitialization()
        num_iter = 0
        n1 = None
        while True:
            num_iter += 1
            self.reverse_search()

            if num_iter % 5000 == 0:
                print(f"Samples: {self.cnt}; time: {time.time() - self.start_time:.2f}s; {ptc}")
            
            if num_iter % 100000 == 0:
                print("Forward Queue: ", len(self.g.cost_bound_queue))

            if len(self.g.cost_bound_queue) < 1:
                print(
                    "------------------------", n1.state.mode.task_ids, n1.id, num_iter
                )
                self.initialize_search(num_iter)
                continue

            key, _ = self.g.cost_bound_queue.peek_first_element()
            if self.current_best_cost is None or key < self.current_best_cost:
                edge_cost, (n0, n1), edge_effort = self.g.get_best_forward_edge()
                if n0.id in n1.blacklist:
                    # needed because of rewiring
                    continue
                # assert n0.id not in n1.blacklist, (
                #     "askdflö"
                # )
                is_transition = False
                if (
                    n1.is_transition
                    and not n1.is_reverse_transition
                    and n1.transition_neighbors
                ):
                    is_transition = True

                if (
                    n1 in BaseTree.all_vertices
                    and not self.dynamic_reverse_search_update
                ):
                    continue

                # if np.isinf(n1.lb_cost_to_go):
                #     assert not [self.g.cost_bound_queue.key(item) for item in self.g.cost_bound_queue.current_entries if not np.isinf(self.g.cost_bound_queue.key(item))], (
                #     "Forward queue contains non-infinite keys!")
                # already found best possible parent
                if n1.forward.parent == n0:  # if its already the parent
                    if is_transition:
                        for transition in n1.transition_neighbors:
                            # if self.config.apply_long_horizon and transition.state.mode not in self.long_horizon.mode_sequence:
                            #     continue
                            if (
                                not self.config.apply_long_horizon
                                and self.current_best_cost is not None
                                and transition.state.mode
                                not in self.sorted_reached_modes
                            ):
                                continue
                            self.expand_node_forward(transition)
                    else:
                        self.expand_node_forward(n1)
                elif n0.cost + edge_cost < n1.cost:  # parent can improve the cost
                    # assert n0.id not in n1.forward.children, (
                    #     "Potential parent is already a child (forward)"
                    # )
                    # check edge sparsely now. if it is not valid, blacklist it, and continue with the next edge
                    if n0.id not in n1.whitelist:
                        collision_free = False
                        if n0.id not in n1.blacklist:
                            n_start, n_end = (n0, n1) if n1.id > n0.id else (n1, n0)
                            edge_id = (n_start.id, n_end.id)
                            N_start = 0
                            if edge_id in self.sparesly_checked_edges:
                                _, N_start = self.get_collision_start_and_end_index(
                                    self.sparse_number_of_points
                                )
                            N = max(2, int(edge_effort))
                            collision_free = self.env.is_edge_collision_free(
                                n_start.state.q,
                                n_end.state.q,
                                n_start.state.mode,
                                N_start=N_start,
                                N_max=N,
                                N=N,
                            )

                        if not collision_free:
                            self.g.update_edge_collision_cache(n0, n1, collision_free)
                            self.g.remove_forward_queue(edge_cost, n0, n1, edge_effort)
                            self.manage_edge_in_collision(n0, n1)
                            continue
                    self.g.update_connectivity(
                        n0, n1, edge_cost, n0.cost + edge_cost, "forward", is_transition
                    )
                    # if self.current_best_cost is not None:
                    # assert (n1.cost + n1.lb_cost_to_go <= self.current_best_cost), (
                    #         "hjklö"
                    #     )
                    if is_transition:
                        for transition in n1.transition_neighbors:
                            # if self.config.apply_long_horizon and transition.state.mode not in self.long_horizon.mode_sequence:
                            #     continue
                            if (
                                not self.config.apply_long_horizon
                                and self.current_best_cost is not None
                                and transition.state.mode
                                not in self.sorted_reached_modes
                            ):
                                continue
                            self.expand_node_forward(transition)
                    else:
                        self.expand_node_forward(n1)
                    update = False
                    if n1 in self.g.goal_nodes or n1 in self.g.virtual_goal_nodes:
                        update = True
                    if self.dynamic_reverse_search_update or update:
                        path = self.generate_path()
                        if len(path) > 0:
                            if self.config.with_tree_visualization and (
                                BaseTree.all_vertices or self.reverse_tree_set
                            ):
                                self.save_tree_data(
                                    (BaseTree.all_vertices, self.reverse_tree_set)
                                )
                            self.process_valid_path(
                                path, with_shortcutting=update, with_queue_update=update
                            )
                            self.update_inflation_factor()
                            if self.config.with_tree_visualization and (
                                BaseTree.all_vertices or self.reverse_tree_set
                            ):
                                self.save_tree_data(
                                    (BaseTree.all_vertices, self.reverse_tree_set)
                                )
                            if (
                                self.config.apply_long_horizon
                                and self.current_best_path_nodes[
                                    -1
                                ].transition_neighbors
                            ):
                                self.initialize_search(num_iter, True)

            else:
                self.initialize_search(num_iter)

            if not optimize and self.current_best_cost is not None:
                if (
                    self.config.apply_long_horizon
                    and self.current_best_path_nodes[-1].transition_neighbors
                ):
                    continue
                break

            if ptc.should_terminate(self.cnt, time.time() - self.start_time):
                break

        if self.costs != []:
            self.update_results_tracking(self.costs[-1], self.current_best_path)
        info = {"costs": self.costs, "times": self.times, "paths": self.all_paths}
        print(self.mode_validation.counter)
        return self.current_best_path, info
