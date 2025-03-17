import numpy as np
import random

from typing import (
    List,
    Dict,
    Tuple,
    Optional,
    Set,
    ClassVar,
    Any,
    Union,
    Generic,
    TypeVar,
)
from numpy.typing import NDArray

import heapq
import time
import math
from abc import ABC, abstractmethod
from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
    Mode,
)
from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
    batch_config_dist,
)
from multi_robot_multi_goal_planning.problems.util import path_cost, interpolate_path

from multi_robot_multi_goal_planning.planners import shortcutting

from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
)
from multi_robot_multi_goal_planning.planners.itstar_base import Informed
from multi_robot_multi_goal_planning.planners.rrtstar_base import find_nearest_indices
from multi_robot_multi_goal_planning.planners.rrtstar_base import save_data

T = TypeVar("T")



class Operation:
    """Represents an operation instance responsible for managing variables related to path planning and cost optimization. """
    def __init__(self):
    
        self.lb_costs_to_go_expanded = np.empty(10000000, dtype=np.float64)
        self.forward_costs = np.empty(10000000, dtype=np.float64)

    def get_lb_cost_to_go_expanded(self, idx:int) -> float:
        """
        Returns cost of node with the specified index.

        Args: 
            idx (int): Index of node whose cost is to be retrieved.

        Returns: 
            float: Cost associated with the specified node."""
        return self.lb_costs_to_go_expanded[idx]
    
    def get_forward_cost(self, idx:int) -> float:
        """
        Returns cost of node with the specified index.

        Args: 
            idx (int): Index of node whose cost is to be retrieved.

        Returns: 
            float: Cost associated with the specified node."""
        return self.forward_costs[idx]
    
    def _resize_array(self, array: NDArray, current_capacity: int, new_capacity: int) -> NDArray:
        """
        Dynamically resizes a NumPy array to a new capacity.

        Args: 
            array (NDArray): Array to be resized. 
            current_capacity (int): Current capacity of array.
            new_capacity (int):Target capacity for the array.

        Returns: 
            NDArray: The resized array. 
        """
        new_array = np.empty((new_capacity, *array.shape[1:]), dtype=array.dtype)
        new_array[:current_capacity] = array  # Copy existing data
        del array  # Free old array (Python garbage collector will handle memory)
        return new_array

    def ensure_capacity(self, array: NDArray, required_capacity: int) -> NDArray:
        """ 
        Ensures that a NumPy array has sufficient capacity to accommodate new elements and resizes it if necessary.

        Args: 
            array (NDArray): The array to be checked and potentially resized. 
            required_capacity (int): The minimum required capacity for the array.

        Returns: 
            NDArray: The array with ensured capacity. """
        current_size = array.shape[0]

        if required_capacity == current_size:
            return self._resize_array(array, current_size, required_capacity * 2)  # Double the size

        return array
    
    



class Node:
    __slots__ = [
        "state",
        "forward_parent",
        "reverse_parent",
        "forward_children",
        "reverse_children",
        "forward_cost_to_parent",
        "reverse_cost_to_parent",
        "lb_cost_to_go",
        "is_transition",
        "transition",
        "forward_fam",
        "reverse_fam",
        "whitelist",
        "blacklist",
        "id",
        "operation"
    ]

    # Class attribute
    id_counter: ClassVar[int] = 0

    # Instance attributes
    state: State
    forward_parent: Optional["Node"]
    reverse_parent: Optional["Node"]
    forward_children: List[int]
    reverse_children: List[int]
    forward_cost_to_parent: float
    reverse_cost_to_parent: float
    lb_cost_to_go: Optional[
        float
    ]  # cost to go to the goal from the node when it was connected the last time
    # lb_cost_to_go_expanded: Optional[
    #     float
    # ]  # cost to go to the goal from the node when it was expanded the last time
    is_transition: bool
    transition: Optional["Node"]
    whitelist: Set[int]
    blacklist: Set[int]
    forward_fam: Set[int] 
    reverse_fam: Set[int]
    id: int
    operation:Operation

    def __init__(self, operation: Operation, state: State, is_transition: bool = False) -> None:
        self.state = state
        # self.forward_cost = np.inf
        self.forward_parent = None
        self.reverse_parent = None
        self.forward_children = []  # children in the forward tree
        self.reverse_children = []  # children in the reverse tree
        self.forward_cost_to_parent = np.inf
        self.reverse_cost_to_parent = np.inf
        self.lb_cost_to_go = np.inf
        # self.lb_cost_to_go_expanded = np.inf
        self.is_transition = is_transition
        self.transition = None
        self.whitelist = set()
        self.blacklist = set()
        self.forward_fam = set()
        self.reverse_fam = set()
        self.id = Node.id_counter
        Node.id_counter += 1
        self.operation = operation

    def __lt__(self, other: "Node") -> bool:
        return self.id < other.id

    def __hash__(self) -> int:
        return self.id

    @property
    def lb_cost_to_go_expanded(self):
        return self.operation.get_lb_cost_to_go_expanded(self.id)
    
    @lb_cost_to_go_expanded.setter
    def lb_cost_to_go_expanded(self, value) -> None:
        """Set the cost in the shared operation costs array.

        Args:
            value (float): Cost value to assign to the current node.

        Returns: 
            None: This method does not return any value."""
        self.operation.lb_costs_to_go_expanded[self.id] = value
    @property
    def forward_cost(self):
        return self.operation.get_forward_cost(self.id)
    
    @forward_cost.setter
    def forward_cost(self, value) -> None:
        """Set the cost in the shared operation costs array.

        Args:
            value (float): Cost value to assign to the current node.

        Returns: 
            None: This method does not return any value."""
        self.operation.forward_costs[self.id] = value

class DictIndexHeap(ABC, Generic[T]):
    __slots__ = ["queue", "items", "current_entries"]

    queue: List[Tuple[float, int]]  # (priority, index)
    items: Dict[int, Any]  # Dictionary for storing active items
    current_entries: Dict[T, Tuple[float, int]]

    idx = 0

    def __init__(self) -> None:
        self.queue = []
        self.items = {}
        self.current_entries = {}  
        heapq.heapify(self.queue)

    def __len__(self):
        return len(self.current_entries)
    
    def __contains__(self, item: T) -> bool:
        return item in self.current_entries
    
    @abstractmethod
    def key(self, node: Node) -> float:
        pass

    def heappush(self, item: T) -> None:
        """Push a single item into the heap."""
        # idx = len(self.items)
        self.items[DictIndexHeap.idx] = item  # Store only valid items
        priority = self.key(item)
        self.current_entries[item] = (priority, DictIndexHeap.idx) # always up to date with the newest one!
        # self.nodes_in_queue[item[1]] = (priority, DictIndexHeap.idx)
        heapq.heappush(self.queue, (priority, DictIndexHeap.idx))
        DictIndexHeap.idx += 1

    def peek_priority(self) -> Any:
        """Return the smallest priority (the key) without removing the item."""
        if not self.queue:
            raise IndexError("peek from an empty heap")
        # Return only the priority (i.e. the first element of the tuple)
        return self.queue[0][0]

    def is_smallest_priority_less_than_root_priority(self, root: Node) -> bool:
        """Return True if the smallest key in the heap is less than other_key."""
        root_key = self.key(root)
        return self.peek_priority() < root_key

    def remove(self, item):
        if item in self.current_entries:
            del self.current_entries[item]
  
class ForwarrdQueue(DictIndexHeap[Tuple[Any]]):
    def __init__(self, alpha = 1.0):
        super().__init__()
        self.alpha = alpha

    def key(self, item: Tuple[Any]) -> float:
        # item[0]: edge_cost from n0 to n1
        # item[1]: edge (n0, n1)
        return (
            item[1][0].forward_cost + item[0] + item[1][1].lb_cost_to_go*self.alpha,
            item[1][0].forward_cost + item[0],
            item[1][0].forward_cost,
        )

    def heappop(self) -> Node:
        """Pop the item with the smallest priority from the heap."""
        if not self.queue:
            raise IndexError("pop from an empty heap")

         # Remove from dictionary (Lazy approach)
        while self.current_entries:
            priority, idx = heapq.heappop(self.queue)
            item = self.items.pop(idx)
            if item in self.current_entries:
                current_priority, current_idx = self.current_entries[item]
                new_priority = self.key(item)
                if current_priority == priority and current_idx == idx:
                    if new_priority != priority: #needed if reverse search changed priorities
                        self.heappush(item) 
                        continue
                    del self.current_entries[item]
                    return item
            else:
                continue
                
        raise IndexError("pop from an empty queue")

class ReverseQueue(DictIndexHeap[Node]):
    def __init__(self):
        super().__init__()

    def key(self, node: Node) -> float:
        return (
            min(node.lb_cost_to_go, node.lb_cost_to_go_expanded) + node.lb_cost_to_go,
            min(node.lb_cost_to_go, node.lb_cost_to_go_expanded),
        )

    def heappop(self) -> Node:
        """Pop the item with the smallest priority from the heap."""
        if not self.queue:
            raise IndexError("pop from an empty heap")

         # Remove from dictionary (Lazy approach)
        while self.current_entries:
            priority, idx = heapq.heappop(self.queue)
            item = self.items.pop(idx)
            if item in self.current_entries:
                current_priority, current_idx = self.current_entries[item]
                if current_priority == priority and current_idx == idx:
                    del self.current_entries[item]
                    return item
            else:
                continue
                
        raise IndexError("pop from an empty queue")
    
class Graph:
    root: Node
    nodes: Dict
    node_ids: Dict
    vertices: Dict

    def __init__(self, operation:Operation, start: State, batch_dist_fun, batch_cost_fun):
        self.operation = operation
        self.dim = len(start.q.state())
        self.root = Node(self.operation, start)
        self.root.forward_cost = 0
        self.root.lb_cost_to_go_expanded = np.inf

        self.batch_dist_fun = batch_dist_fun
        self.batch_cost_fun = batch_cost_fun

        self.nodes = {}  # contains all the nodes ever created
        self.nodes[self.root.id] = self.root
        
        self.node_ids = {}
        self.node_ids[self.root.state.mode] = [self.root.id]
        self.vertices = [self.root.id]
        self.transition_node_ids = {}  # contains the transitions at the end of the mode
        self.reverse_transition_node_ids = {}
        self.reverse_transition_node_ids[self.root.state.mode] = [self.root.id]
        self.goal_nodes = []
        self.goal_node_ids = []
        self.lb_costs_to_go_expanded = np.empty(10000000, dtype=np.float64)
        self.initialize_cache()
        # self.current_best_path_nodes = []
        self.current_best_path = []
           
    def get_lb_cost_to_go_expand(self, id:int) -> float:
        """
        Returns cost of node with the specified index.

        Args: 
            id (int): Index of node whose cost is to be retrieved.

        Returns: 
            float: Cost associated with the specified node."""
        return self.lb_costs_to_go_expanded[id]
    
    def reset_reverse_tree(self):
        [
            (
                setattr(self.nodes[node_id], "lb_cost_to_go", math.inf),
                setattr(self.nodes[node_id], "lb_cost_to_go_expanded", math.inf),
                setattr(self.nodes[node_id], "reverse_parent", None),
                setattr(self.nodes[node_id], "reverse_children", []),
                setattr(self.nodes[node_id], "reverse_cost_to_parent", np.inf),
                setattr(self.nodes[node_id], "reverse_fam", set()),
            )
            for sublist in self.node_ids.values()
            for node_id in sublist
        ]

        [
            (
                setattr(self.nodes[node_id], "lb_cost_to_go", math.inf),
                setattr(self.nodes[node_id], "lb_cost_to_go_expanded", math.inf),
                setattr(self.nodes[node_id], "reverse_parent", None),
                setattr(self.nodes[node_id], "reverse_children", []),
                setattr(self.nodes[node_id], "reverse_cost_to_parent", np.inf),
                setattr(self.nodes[node_id], "reverse_fam", set()),
            )
            for sublist in self.transition_node_ids.values()
            for node_id in sublist
        ]  # also includes goal nodes

        [
            (
                setattr(self.nodes[node_id], "lb_cost_to_go", math.inf),
                setattr(self.nodes[node_id], "lb_cost_to_go_expanded", math.inf),
                setattr(self.nodes[node_id], "reverse_parent", None),
                setattr(self.nodes[node_id], "reverse_children", []),
                setattr(self.nodes[node_id], "reverse_cost_to_parent", np.inf),
                setattr(self.nodes[node_id], "reverse_fam", set()),
            )
            for sublist in self.reverse_transition_node_ids.values()
            for node_id in sublist
        ]

        # for id in self.vertices:
        #     node = self.nodes[id]
        #     node.lb_cost_to_go = np.inf
        #     node.lb_cost_to_go_expanded = np.inf

    def reset_all_goal_nodes_lb_costs_to_go(self):
        for node in self.goal_nodes:
            node.lb_cost_to_go = 0
            node.reverse_cost_to_parent = 0
            self.reverse_queue.heappush(node)

    def get_num_samples(self) -> int:
        num_samples = 0
        for k, v in self.node_ids.items():
            num_samples += len(v)

        num_transition_samples = 0
        for k, v in self.transition_node_ids.items():
            num_transition_samples += len(v)

        return num_samples + num_transition_samples

    def add_node(self, new_node: Node) -> None:
        self.nodes[new_node.id] = new_node
        key = new_node.state.mode
        if key not in self.node_ids:
            self.node_ids[key] = []
        self.node_ids[key].append(new_node.id)
        self.operation.lb_costs_to_go_expanded = self.operation.ensure_capacity(self.operation.lb_costs_to_go_expanded, new_node.id) 
        new_node.lb_cost_to_go_expanded = np.inf
        self.operation.forward_costs = self.operation.ensure_capacity(self.operation.forward_costs, new_node.id) 
        new_node.forward_cost = np.inf

    def add_vertex(self, node: Node) -> None:  # TODO really needed??????
        if node.id in self.vertices:
            return
        self.vertices.append(node.id)
        # if node.is_transition and node.transition is not None:
        #     self.vertices.append(node.transition.id)

    def add_states(self, states: List[State]):
        for s in states:
            self.add_node(Node(self.operation, s))

    def add_nodes(self, nodes: List[Node]):  # EXTEND not better??? TODO
        for n in nodes:
            self.add_node(n)

    def add_transition_nodes(self, transitions: Tuple[Configuration, Mode, Mode]):
        
        for q, this_mode, next_mode in transitions:
            node_this_mode = Node(self.operation, State( q, this_mode), True)
            node_next_mode = Node(self.operation, State( q, next_mode), True)


            if this_mode in self.transition_node_ids:
                if self.transition_is_already_present(node_this_mode):
                    # if path:
                    #     print("QQQQQQQQQQQQQQQQQQQQQQQQQQQ")
                    #     self.add_vertex(node_this_mode)
                    #     self.add_vertex(node_next_mode)
                    continue
            is_goal = True
            if next_mode is not None:
                is_goal = False
                node_next_mode.transition = node_this_mode
                node_this_mode.transition = node_next_mode
                assert this_mode.task_ids != next_mode.task_ids

            self.add_transition_node(node_this_mode, is_goal=is_goal)
            self.add_transition_node(node_next_mode, reverse=True)

    def add_transition_node(self, node:Node, is_goal:bool = False, reverse:bool = False):
        mode = node.state.mode
        if mode is None:
            return 
        
        if is_goal:
            self.goal_nodes.append(node)
            self.goal_node_ids.append(node.id)

        if not reverse:
            if mode in self.transition_node_ids:
                self.transition_node_ids[mode].append(node.id)
            else:
                self.transition_node_ids[mode] = [node.id]
            self.nodes[node.id] = node

        if reverse: 
            if mode in self.reverse_transition_node_ids:
                self.reverse_transition_node_ids[mode].append(node.id)  # TODO for nearest neighbor search, correct ??
            else:
                self.reverse_transition_node_ids[mode] = [node.id]
            self.nodes[node.id] = node
        self.operation.lb_costs_to_go_expanded = self.operation.ensure_capacity(self.operation.lb_costs_to_go_expanded, node.id) 
        node.lb_cost_to_go_expanded = np.inf
        self.operation.forward_costs = self.operation.ensure_capacity(self.operation.forward_costs, node.id) 
        node.forward_cost = np.inf
    
    def add_path_node(self, node:Node, parent:Node, edge_cost:float, is_transition:bool, next_mode:Mode): 
        self.update_connectivity(parent, node, edge_cost)
        if is_transition:
            is_goal = True
            node_next_mode = Node(self.operation, State(node.state.q, next_mode), is_transition)            
            if next_mode is not None:
                # self.current_best_path_nodes.append(node_next_mode)
                self.current_best_path.append(node_next_mode.state)
                is_goal = False
                self.update_connectivity(node, node_next_mode, 0.0)

                node.transition = node_next_mode
                node_next_mode.transition = node

            self.add_transition_node(node, is_goal=is_goal)
            self.add_transition_node(node_next_mode, reverse=True)
        else:
            self.add_node(node)

    def add_path_states(self, path:List[State]):
        # self.current_best_path_nodes = []
        self.current_best_path = []
        batch_edge_cost = self.batch_cost_fun(path[:-1], path[1:])
        parent = self.root
        # self.current_best_path_nodes.append(parent)
        self.current_best_path.append(parent.state)
        for i in range(len(path)):
            if i == 0:
                continue
            is_transition = False
            next_mode = None
            edge_cost = batch_edge_cost[i-1]
            # if not self.env.is_collision_free(s.q, s.mode):
            #     continue #TODO needed?

            if (
                i < len(path) - 1
                and path[i].mode
                != path[i + 1].mode
            ):
                is_transition = True
                next_mode = path[i+1].mode
            if i == len(path)-1:
                is_transition = True
            node = Node(self.operation, path[i], is_transition)
            # self.current_best_path_nodes.append(node)
            self.current_best_path.append(node.state)
            self.add_path_node(node, parent, edge_cost, is_transition, next_mode)
            parent = node
        return self.current_best_path, None

    def transition_is_already_present(self, node:Node, is_goal:bool = False):
        if not is_goal:
            configs_list = [
                                self.nodes[id].state.q
                                for id in self.transition_node_ids[node.state.mode]
                            ]
        else:
            configs_list = [
                                g.state.q
                                for g in self.goal_nodes
                            ]
        dists = self.batch_dist_fun(node.state.q, configs_list)
        if min(dists) < 1e-6:
            return True
        return False

    def update_cache(self, key: Mode):
        if key in self.node_ids:
            if key not in self.node_array_cache:
                self.node_array_cache[key] = np.array(
                    [self.nodes[id].state.q.q for id in self.node_ids[key]],
                    dtype=np.float64,
                )
        if key in self.transition_node_ids:
            if key not in self.transition_node_array_cache:
                self.transition_node_array_cache[key] = np.array(
                    [self.nodes[id].state.q.q for id in self.transition_node_ids[key]],
                    dtype=np.float64,
                )
        if key in self.reverse_transition_node_ids:
            if key not in self.reverse_transition_node_array_cache:
                self.reverse_transition_node_array_cache[key] = np.array(
                    [
                        self.nodes[id].state.q.q
                        for id in self.reverse_transition_node_ids[key]
                    ],
                    dtype=np.float64,
                )
        if key.prev_mode is None:
            if key not in self.reverse_transition_node_array_cache:
                self.reverse_transition_node_array_cache[key] = np.array(
                    [self.root.state.q.q], dtype=np.float64
                )

    def initialize_cache(self):
        # modes as keys
        self.node_array_cache = {}
        self.transition_node_array_cache = {}
        self.reverse_transition_node_array_cache = {}

        self.reverse_transition_node_lb_cache = {}
        self.transition_node_lb_cache = {}

        # node ids as keys
        self.neighbors_node_ids_cache = {}
        self.reverse_neighbors_node_ids_cache = {}

        self.neighbors_array_cache = {}
        self.reverse_neighbors_array_cache = {}

        self.neighbors_fam_ids_cache = {}
        self.reverse_neighbors_fam_ids_cache = {}

        self.tot_neighbors_batch_cost_cache = {}
        self.reverse_tot_neighbors_batch_cost_cache = {}

        self.tot_neighbors_id_cache = {}
        self.reverse_tot_neighbors_id_cache = {}
  
    def get_r_star(
        self, number_of_nodes, informed_measure, unit_n_ball_measure, wheight=1
    ):
        r_star = (
            1.001
            * wheight
            * (
                (2 * (1 + 1 / self.dim))
                * (informed_measure / unit_n_ball_measure)
                * (np.log(number_of_nodes) / number_of_nodes)
            )
            ** (1 / self.dim)
        )
        return r_star

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def get_neighbors(
        self, node: Node, space_extent: float = None, search: str = "forward", in_closed_set:bool = False
    ) -> set:
        
        if search == "forward" and node.id in self.neighbors_node_ids_cache:
            if in_closed_set:
                return self.update_neighbors_with_family_of_node(node, search)
            return self.update_neighbors(node, search)
        if search == "reverse" and node.id in self.reverse_neighbors_node_ids_cache:
            if in_closed_set:
                return self.update_neighbors_with_family_of_node(node, search)
            return self.update_neighbors(node, search)

        key = node.state.mode
        self.update_cache(key)

        best_nodes_arr = np.zeros((0, self.dim))
        best_transitions_arr = np.zeros((0, self.dim))

        unit_n_ball_measure = ((np.pi**0.5) ** self.dim) / math.gamma(self.dim / 2 + 1)
        informed_measure = 1
        if space_extent is not None:
            informed_measure = space_extent
        
        indices_transitions, indices_reverse_transitions, indices = np.empty((0,), dtype=int), np.empty((0,), dtype=int), np.empty((0,), dtype=int)
        node_ids = np.empty((0,), dtype=int)
        transition_node_ids = np.empty((0,), dtype=int)
        if key in self.node_ids:
            dists = self.batch_dist_fun(node.state.q, self.node_array_cache[key])
            r_star = self.get_r_star(
                len(self.node_ids[key]), informed_measure, unit_n_ball_measure
            )
            indices = find_nearest_indices(dists, r_star)
            node_ids = np.array(self.node_ids[key])[indices]

            best_nodes_arr = self.node_array_cache[key][indices]
        if key.task_ids == [0,1]:
            pass
        if key in self.transition_node_ids:
            transition_dists = self.batch_dist_fun(
                node.state.q, self.transition_node_array_cache[key]
            )

            r_star = self.get_r_star(
                len(self.transition_node_ids[key]),
                informed_measure,
                unit_n_ball_measure
            )
            if len(self.transition_node_ids[key]) == 1:
                r_star = 1e6

            indices_transitions = find_nearest_indices(transition_dists, r_star)
            transition_node_ids = np.array(self.transition_node_ids[key])[
                indices_transitions
            ]
            best_transitions_arr = self.transition_node_array_cache[key][indices_transitions]

            # assert len(indices_transitions) != 0, (
            #     "didn't find a transition node"
            # )
               
        if key in self.reverse_transition_node_array_cache and search == "reverse":  # TODO make it better!!!
            reverse_transition_dists = self.batch_dist_fun(
                node.state.q, self.reverse_transition_node_array_cache[key]
            )
            r_star = self.get_r_star(
                len(self.reverse_transition_node_ids[key]),
                informed_measure,
                unit_n_ball_measure
               
            )

            # if len(self.reverse_transition_node_array_cache[key]) == 1:
            #     r_star = 1e6

            indices_reverse_transitions = find_nearest_indices(
                reverse_transition_dists, r_star
            )
            if indices_reverse_transitions.size > 0:
                reverse_transition_node_ids = np.array(
                    self.reverse_transition_node_ids[key]
                )[indices_reverse_transitions]
                
                # reverse_transition_node_ids = reverse_transition_node_ids[~np.isin(reverse_transition_node_ids, list(node.blacklist))]
                best_reverse_transitions_arr = self.reverse_transition_node_array_cache[
                    key
                ][indices_reverse_transitions]

                indices_transitions = np.concatenate((indices_transitions, indices_reverse_transitions))

                transition_node_ids = np.concatenate(
                    (reverse_transition_node_ids, transition_node_ids)
                )
                best_transitions_arr = np.vstack(
                    [best_reverse_transitions_arr, best_transitions_arr],
                    dtype=np.float64,
                )
        all_ids = np.concatenate((node_ids, transition_node_ids)) 
        arr = np.vstack([best_nodes_arr, best_transitions_arr], dtype=np.float64)

        if (
            node.is_transition
            and node.transition is not None
        ):
            all_ids = np.concatenate((all_ids, np.array([node.transition.id])))
            arr = np.vstack([arr, node.transition.state.q.state()])
        
        if search == "forward":
            assert node.id not in self.neighbors_node_ids_cache,("2 already calculated")
            self.neighbors_node_ids_cache[node.id] = all_ids 
            self.neighbors_array_cache[node.id] = arr
        if search == "reverse":
            assert node.id not in self.reverse_neighbors_node_ids_cache,("4 already calculated")
            self.reverse_neighbors_node_ids_cache[node.id] = all_ids
            self.reverse_neighbors_array_cache[node.id] = arr
        
        return self.update_neighbors(node, search)

    def update_neighbors_with_family_of_node(self, node:Node, search:str):
        neighbors_fam = set()
        if search == "forward":
            if node.id in self.neighbors_fam_ids_cache:
                neighbors_fam = self.neighbors_fam_ids_cache[node.id]
        if search == "reverse":
            if node.id in self.reverse_neighbors_fam_ids_cache:
                neighbors_fam = self.reverse_neighbors_fam_ids_cache[node.id]

        combined_fam = node.forward_fam | node.reverse_fam
        blacklist = node.blacklist
        if len(blacklist) > 0 and len(combined_fam) > 0:
            combined_fam =  combined_fam - blacklist
        if  neighbors_fam != combined_fam:
            if search == "forward":
                self.neighbors_fam_ids_cache[node.id] = combined_fam 
                node_ids = self.neighbors_node_ids_cache[node.id]
                arr = self.neighbors_array_cache[node.id]
            if search == "reverse":
                self.reverse_neighbors_fam_ids_cache[node.id] = combined_fam 
                node_ids = self.reverse_neighbors_node_ids_cache[node.id]
                arr = self.reverse_neighbors_array_cache[node.id]
            mask_node_ids =  np.array(list(combined_fam - set(node_ids)))
            if mask_node_ids.size > 0:
                arr = np.array(
                    [self.nodes[id].state.q.q for id in mask_node_ids],
                    dtype=np.float64,
                )
                if search == "forward":
                    arr = np.concatenate((arr, self.neighbors_array_cache[node.id]))
                    self.tot_neighbors_batch_cost_cache[node.id] = self.batch_cost_fun(node.state.q, arr)
                    self.tot_neighbors_id_cache[node.id] = np.concatenate((mask_node_ids, self.neighbors_node_ids_cache[node.id]))
                    assert len(self.tot_neighbors_id_cache[node.id]) == len(self.tot_neighbors_batch_cost_cache[node.id]),(
                        "forward not right"
                    )
                    return self.tot_neighbors_id_cache[node.id]
                else:
                    arr = np.concatenate((arr, self.reverse_neighbors_array_cache[node.id]))
                    self.reverse_tot_neighbors_batch_cost_cache[node.id] = self.batch_cost_fun(node.state.q, arr)
                    self.reverse_tot_neighbors_id_cache[node.id] = np.concatenate((mask_node_ids, self.reverse_neighbors_node_ids_cache[node.id]))
                    assert len(self.reverse_tot_neighbors_id_cache[node.id]) == len(self.reverse_tot_neighbors_batch_cost_cache[node.id]),(
                        " reverse not right"
                    )
                    return self.reverse_tot_neighbors_id_cache[node.id]

        if search == "forward":
            if node.id not in self.tot_neighbors_id_cache:
                arr = self.neighbors_array_cache[node.id]
                self.tot_neighbors_batch_cost_cache[node.id] = self.batch_cost_fun(node.state.q, arr)
                self.tot_neighbors_id_cache[node.id] = self.neighbors_node_ids_cache[node.id]
            assert len(self.tot_neighbors_id_cache[node.id]) == len(self.tot_neighbors_batch_cost_cache[node.id]),(
                "forward not right"
            )
            return self.tot_neighbors_id_cache[node.id]
        else:
            if node.id not in self.reverse_tot_neighbors_id_cache:
                arr = self.reverse_neighbors_array_cache[node.id]
                self.reverse_tot_neighbors_batch_cost_cache[node.id] = self.batch_cost_fun(node.state.q, arr)
                self.reverse_tot_neighbors_id_cache[node.id] = self.reverse_neighbors_node_ids_cache[node.id]
            assert len(self.reverse_tot_neighbors_id_cache[node.id]) == len(self.reverse_tot_neighbors_batch_cost_cache[node.id]),(
                " reverse not right"
            )
            return self.reverse_tot_neighbors_id_cache[node.id]

    def update_neighbors(self, node:Node, search:str): # only needed for forward
        blacklist = node.blacklist
        if len(blacklist) > 0:
            if search == "forward":
                node_ids = self.neighbors_node_ids_cache[node.id]
                arr =  self.neighbors_array_cache[node.id]
            if search == "reverse":
                node_ids = self.reverse_neighbors_node_ids_cache[node.id]
                arr = self.reverse_neighbors_array_cache[node.id]
            mask_node_ids =  ~np.isin(node_ids, blacklist)
            
            if not mask_node_ids.all():
                node_ids = node_ids[mask_node_ids]
                arr = arr[mask_node_ids]
                if search == "forward":
                    self.neighbors_node_ids_cache[node.id] = node_ids
                    self.neighbors_array_cache[node.id] = arr
                else:
                    self.reverse_neighbors_node_ids_cache[node.id] = node_ids
                    self.reverse_neighbors_array_cache[node.id] = arr
        return self.update_neighbors_with_family_of_node(node, search)
            
    def update_forward_cost(self, n: Node) -> None:
        stack = [n.id]
        while stack:
            current_id = stack.pop()
            current_node = self.nodes[current_id]
            children = current_node.forward_children
            if children:
                for _, id in enumerate(children):
                    child = self.nodes[id]
                    new_cost = current_node.forward_cost + child.forward_cost_to_parent
                    child.forward_cost = new_cost
                    self.forward_queue.heappush((child.forward_cost_to_parent, (current_node, child)))
                    # child.agent_dists = current_node.agent_dists + child.agent_dists_to_parent
                stack.extend(children)

    def update_connectivity(self, n0: Node, n1: Node, edge_cost, tree: str = "forward"):
        if tree == "forward":
            if n1.forward_parent is not None and n1.forward_parent.id != n0.id:
                n1.forward_parent.forward_children.remove(n1.id)
                n1.forward_parent.forward_fam.remove(n1.id)
                n1.forward_fam.remove(n1.forward_parent.id)
            n1.forward_parent = n0
            updated_cost = n0.forward_cost + edge_cost
            n1.forward_cost_to_parent = edge_cost
            n0.forward_children.append(n1.id)
            if updated_cost != n1.forward_cost:
                n1.forward_cost = updated_cost
                if len(n1.forward_children) > 0:
                    self.update_forward_cost(n1)
            else:
                print("")

            n1.forward_fam.add(n0.id)
            n0.forward_fam.add(n1.id)
        else:
            n1.lb_cost_to_go = n0.lb_cost_to_go_expanded + edge_cost
            if n1.reverse_parent is not None:
                if n1.reverse_parent.id == n0.id:
                    return
                if n1.reverse_parent.id != n0.id:
                    assert [
                                (self.nodes[child].reverse_parent, child)
                                for child in n1.reverse_parent.reverse_children
                                if self.nodes[child].reverse_parent is None
                                or self.nodes[child].reverse_parent.id != n1.reverse_parent.id
                            ] == [], "parent and children not correct"

                    n1.reverse_parent.reverse_children.remove(n1.id)
                    n1.reverse_parent.reverse_fam.remove(n1.id)
                    n1.reverse_fam.remove(n1.reverse_parent.id)


            n1.reverse_parent = n0
            assert n1.id not in n0.reverse_children, "already a child"
            n0.reverse_children.append(n1.id)
            n1.reverse_cost_to_parent = edge_cost
            
            assert [
                        (self.nodes[child].reverse_parent, child)
                        for child in n1.reverse_parent.reverse_children
                        if self.nodes[child].reverse_parent is None
                        or self.nodes[child].reverse_parent.id != n1.reverse_parent.id
                    ] == [], "new parent and new children not correct"
            n1.reverse_fam.add(n0.id)
            n0.reverse_fam.add(n1.id)

            


# taken from https://github.com/marleyshan21/Batch-informed-trees/blob/master/python/BIT_Star.py
# needed adaption to work.


class AITstar:
    def __init__(
        self,
        env: BaseProblem,
        ptc: PlannerTerminationCondition,
        optimize: bool = True,
        mode_sampling_type: str = "greedy",
        distance_metric: str = "euclidean",
        try_sampling_around_path: bool = True,
        use_k_nearest: bool = True,
        try_informed_sampling: bool = True,
        uniform_batch_size: int = 200,
        uniform_transition_batch_size: int = 500,
        informed_batch_size: int = 500,
        informed_transition_batch_size: int = 500,
        path_batch_size: int = 500,
        locally_informed_sampling: bool = True,
        try_informed_transitions: bool = True,
        try_shortcutting: bool = True,
        try_direct_informed_sampling: bool = True,
        informed_with_lb:bool = True
    ):
        self.env = env
        self.ptc = ptc
        self.optimize = optimize
        self.mode_sampling_type = mode_sampling_type
        self.distance_metric = distance_metric
        self.try_sampling_around_path = try_sampling_around_path
        self.use_k_nearest = use_k_nearest
        self.try_informed_sampling = try_informed_sampling
        self.uniform_batch_size = uniform_batch_size
        self.uniform_transition_batch_size = uniform_transition_batch_size
        self.informed_batch_size = informed_batch_size
        self.informed_transition_batch_size = informed_transition_batch_size
        self.path_batch_size = path_batch_size
        self.locally_informed_sampling = locally_informed_sampling
        self.try_informed_transitions = try_informed_transitions
        self.try_shortcutting = try_shortcutting
        self.try_direct_informed_sampling = try_direct_informed_sampling
        self.informed_with_lb = informed_with_lb

        self.reached_modes = []
        self.start_time = time.time()
        self.costs = []
        self.times = []
        self.all_paths = []
        self.conf_type = type(env.get_start_pos())
        self.informed = Informed(env, self.sample_mode, self.conf_type, self.informed_with_lb)
        self.approximate_space_extent = np.prod(np.diff(env.limits, axis=0))
        self.current_best_cost = None
        self.current_best_path = None
        # self.current_best_path_nodes = None
        self.cnt = 0
        self.operation = Operation()
        self.alpha = 4.5
        
    def sample_valid_uniform_batch(self, batch_size: int, cost: float) -> List[State]:
        new_samples = []
        num_attempts = 0
        num_valid = 0

        if len(self.g.goal_nodes) > 0:
            focal_points = np.array(
                [self.g.root.state.q.state(), self.g.goal_nodes[0].state.q.state()],
                dtype=np.float64,
            )

        while len(new_samples) < batch_size:
            num_attempts += 1
            # print(len(new_samples))
            # sample mode
            m = self.sample_mode(
                self.reached_modes, "uniform_reached", cost is not None
            )

            # print(m)

            # sample configuration
            q = []
            for i in range(len(self.env.robots)):
                r = self.env.robots[i]
                lims = self.env.limits[:, self.env.robot_idx[r]]
                if lims[0, 0] < lims[1, 0]:
                    qr = (
                        np.random.rand(self.env.robot_dims[r])
                        * (lims[1, :] - lims[0, :])
                        + lims[0, :]
                    )
                else:
                    qr = np.random.rand(self.env.robot_dims[r]) * 6 - 3

                q.append(qr)

            q = self.conf_type.from_list(q)

            if cost is not None:
                if sum(self.env.batch_config_cost(q, focal_points)) > cost:
                    continue

            if self.env.is_collision_free(q, m):
                new_samples.append(State(q, m))
                num_valid += 1

        print("Percentage of succ. attempts", num_valid / num_attempts)

        return new_samples, num_attempts

    def sample_valid_uniform_transitions(self, transistion_batch_size, cost):
        transitions = []

        if len(self.g.goal_nodes) > 0:
            focal_points = np.array(
                [self.g.root.state.q.state(), self.g.goal_nodes[0].state.q.state()],
                dtype=np.float64,
            )

        while len(transitions) < transistion_batch_size:
            # sample mode
            mode = self.sample_mode(self.reached_modes, "uniform_reached", None)

            # sample transition at the end of this mode
            possible_next_task_combinations = self.env.get_valid_next_task_combinations(
                mode
            )
            # print(mode, possible_next_task_combinations)

            if len(possible_next_task_combinations) > 0:
                ind = random.randint(0, len(possible_next_task_combinations) - 1)
                active_task = self.env.get_active_task(
                    mode, possible_next_task_combinations[ind]
                )
            else:
                active_task = self.env.get_active_task(mode, None)

            goals_to_sample = active_task.robots

            goal_sample = active_task.goal.sample(mode)

            # if mode.task_ids == [3, 8]:
            #     print(active_task.name)

            q = []
            for i in range(len(self.env.robots)):
                r = self.env.robots[i]
                if r in goals_to_sample:
                    offset = 0
                    for _, task_robot in enumerate(active_task.robots):
                        if task_robot == r:
                            q.append(
                                goal_sample[
                                    offset : offset + self.env.robot_dims[task_robot]
                                ]
                            )
                            break
                        offset += self.env.robot_dims[task_robot]
                else:  # uniform sample
                    lims = self.env.limits[:, self.env.robot_idx[r]]
                    if lims[0, 0] < lims[1, 0]:
                        qr = (
                            np.random.rand(self.env.robot_dims[r])
                            * (lims[1, :] - lims[0, :])
                            + lims[0, :]
                        )
                    else:
                        qr = np.random.rand(self.env.robot_dims[r]) * 6 - 3

                    q.append(qr)

            q = self.conf_type.from_list(q)

            if cost is not None:
                if sum(self.env.batch_config_cost(q, focal_points)) > cost:
                    continue

            if self.env.is_collision_free(q, mode):
                if self.env.is_terminal_mode(mode):
                    next_mode = None
                else:
                    next_mode = self.env.get_next_mode(q, mode)
                    if next_mode.task_ids == [2,2] and mode.task_ids == [2,1]:
                        pass
                    if next_mode.task_ids == [2,2] and mode.task_ids == [0,2]:
                        next_mode.prev_mode = mode
                        pass

                transitions.append((q, mode, next_mode))

                # print(mode, mode.next_modes)

                if next_mode not in self.reached_modes and next_mode is not None:
                    self.reached_modes.append(next_mode)
            # else:
            #     if mode.task_ids == [3, 8]:
            #         self.env.show(True)

        return transitions

    def add_sample_batch(self):
        # add new batch of nodes
        while True:
            
            if len(self.g.nodes) > 1:
                sample_batch_size = 500
                transition_batch_size = 500
            else:
                sample_batch_size = 350
                transition_batch_size = 400



            effective_uniform_batch_size = (
                self.uniform_batch_size if self.current_best_cost is not None else sample_batch_size
            )
            effective_uniform_transition_batch_size = (
                self.uniform_transition_batch_size
                if self.current_best_cost is not None
                else transition_batch_size
            )

            # nodes_per_state = []
            # for m in reached_modes:
            #     num_nodes = 0
            #     for n in new_states:
            #         if n.mode == m:
            #             num_nodes += 1

            #     nodes_per_state.append(num_nodes)

            # plt.figure("Uniform states")
            # plt.bar([str(mode) for mode in reached_modes], nodes_per_state)

            # if self.env.terminal_mode not in reached_modes:   
            print("--------------------")
            print("Sampling transitions")
            new_transitions = self.sample_valid_uniform_transitions(
                transistion_batch_size=effective_uniform_transition_batch_size,
                cost=self.current_best_cost,
            )
            self.g.add_transition_nodes(new_transitions)
            print(f"Adding {len(new_transitions)} transitions")
            print("--------------------")
            
            print("Sampling uniform")
            new_states, required_attempts_this_batch = self.sample_valid_uniform_batch(
                    batch_size=effective_uniform_batch_size, cost=self.current_best_cost
                )
            self.g.add_states(new_states)
            print(f"Adding {len(new_states)} new states")

            self.approximate_space_extent = (
                np.prod(np.diff(self.env.limits, axis=0))
                * len(new_states)
                / required_attempts_this_batch
            )

            # print(reached_modes)

            if len(self.g.goal_nodes) == 0:
                continue

        # g.compute_lb_cost_to_go(self.env.batch_config_cost)
        # g.compute_lower_bound_from_start(self.env.batch_config_cost)

            if self.current_best_cost is not None and (
                self.try_informed_sampling or self.try_informed_transitions
            ):
                interpolated_path = interpolate_path(self.current_best_path)
                # interpolated_path = current_best_path

                if self.try_informed_sampling:
                    print("Generating informed samples")
                    new_informed_states = self.informed.generate_samples(
                        self.g,
                        self.reached_modes,
                        self.informed_batch_size,
                        interpolated_path,
                        locally_informed_sampling=self.locally_informed_sampling,
                        try_direct_sampling=self.try_direct_informed_sampling,
                    )
                    self.g.add_states(new_informed_states)

                    print(f"Adding {len(new_informed_states)} informed samples")

                if self.try_informed_transitions:
                    print("Generating informed transitions")
                    new_informed_transitions = self.informed.generate_transitions(
                        self.g,
                        self.reached_modes,
                        self.informed_transition_batch_size,
                        interpolated_path,
                        locally_informed_sampling=self.locally_informed_sampling,
                    )
                    self.g.add_transition_nodes(new_informed_transitions)
                    print(f"Adding {len(new_informed_transitions)} informed transitions")

                    # g.compute_lb_cost_to_go(self.env.batch_config_cost)
                    # g.compute_lower_bound_from_start(self.env.batch_config_cost)

            if self.try_sampling_around_path and self.current_best_path is not None:
                print("Sampling around path")
                path_samples, path_transitions = self.sample_around_path(
                    self.current_best_path
                )

                self.g.add_states(path_samples)
                print(f"Adding {len(path_samples)} path samples")

                self.g.add_transition_nodes(path_transitions)
                print(f"Adding {len(path_transitions)} path transitions")
            
            if len(self.g.goal_nodes) != 0:
                break
        

        # self.g.compute_lb_reverse_cost_to_come(self.env.batch_config_cost)
        # self.g.compute_lb_cost_to_come(self.env.batch_config_cost)

    def sample_mode(
        self,
        reached_modes,
        mode_sampling_type: str = "uniform_reached",
        found_solution: bool = False,
    ) -> Mode:
        if mode_sampling_type == "uniform_reached":
            m_rnd = random.choice(reached_modes)

        return m_rnd

    def sample_manifold(self) -> None:
        print("====================")
        while True:
            self.g.initialize_cache()
            if self.current_best_path is not None:
                # prune
                # for mode in self.reached_modes:
                #     q_samples = [self.g.nodes[id].state.q.state() for id in self.g.node_ids[mode]]
                #     modes = [self.g.nodes[id].state.mode.task_ids for id in self.g.node_ids[mode]]
                #     data = {
                #         "q_samples": q_samples,
                #         "modes": modes,
                #         "path": self.current_best_path
                #     }
                #     save_data(data)
                #     print()
                # for mode in self.reached_modes:
                #     q_samples = [self.g.nodes[id].state.q.state() for id in self.g.transition_node_ids[mode]]
                #     modes = [self.g.nodes[id].state.mode.task_ids for id in self.g.transition_node_ids[mode]]
                #     data = {
                #         "q_samples": q_samples,
                #         "modes": modes,
                #         "path": self.current_best_path
                #     }
                #     save_data(data)
                #     print()

                self.remove_nodes_in_graph()

                # for mode in self.reached_modes:
                #     q_samples = [self.g.nodes[id].state.q.state() for id in self.g.node_ids[mode]]
                #     modes = [self.g.nodes[id].state.mode.task_ids for id in self.g.node_ids[mode]]
                #     data = {
                #         "q_samples": q_samples,
                #         "modes": modes,
                #         "path": self.current_best_path
                #     }
                #     save_data(data)
                #     print()
                # for mode in self.reached_modes:
                #     q_samples = [self.g.nodes[id].state.q.state() for id in self.g.transition_node_ids[mode]]
                #     modes = [self.g.nodes[id].state.mode.task_ids for id in self.g.transition_node_ids[mode]]
                #     data = {
                #         "q_samples": q_samples,
                #         "modes": modes,
                #         "path": self.current_best_path
                #     }
                #     save_data(data)
                #     print()


            print(f"Samples: {self.cnt}; {self.ptc}")

            samples_in_graph_before = self.g.get_num_samples()
            self.add_sample_batch()
            self.g.initialize_cache()

            samples_in_graph_after = self.g.get_num_samples()
            self.cnt += samples_in_graph_after - samples_in_graph_before

            # search over nodes:
            # 1. search from goal state with sparse check
            reached_terminal_mode = False
            for m in self.reached_modes:
                if self.env.is_terminal_mode(m):
                    reached_terminal_mode = True
                    break

            if reached_terminal_mode:
                print("====================")
                break

    def remove_nodes_in_graph(self):
        num_pts_for_removal = 0
        focal_points = np.array(
            [self.g.root.state.q.state(), self.g.goal_nodes[0].state.q.state()],
            dtype=np.float64,
        )
        # Remove elements from g.nodes_ids
        for mode in list(
            self.g.node_ids.keys()
        ):  # Avoid modifying dict while iterating
            original_count = len(self.g.node_ids[mode])
            self.g.node_ids[mode] = [
                id
                for id in self.g.node_ids[mode]
                if id in self.g.vertices or sum(
                    self.env.batch_config_cost(self.g.nodes[id].state.q, focal_points)
                )
                <= self.current_best_cost
            ]
            num_pts_for_removal += original_count - len(self.g.node_ids[mode])

        # Remove elements from g.transition_node_ids
        for mode in list(self.g.transition_node_ids.keys()):
            original_count = len(self.g.transition_node_ids[mode])
            self.g.transition_node_ids[mode] = [
                id
                for id in self.g.transition_node_ids[mode]
                if id in self.g.vertices or sum(
                    self.env.batch_config_cost(self.g.nodes[id].state.q, focal_points)
                )
                <= self.current_best_cost
            ]
            num_pts_for_removal += original_count - len(
                self.g.transition_node_ids[mode]
            )
        # Update elements from g.reverse_transition_node_ids
        for mode in list(self.g.reverse_transition_node_ids.keys()):
            original_count = len(self.g.reverse_transition_node_ids[mode])
            if mode == self.env.get_start_mode():
                num_pts_for_removal += original_count - len(
                self.g.reverse_transition_node_ids[mode]
                )
                continue
            
            self.g.reverse_transition_node_ids[mode] = [
                self.g.nodes[id].transition.id
                for id in self.g.transition_node_ids[mode.prev_mode]
            ]
            num_pts_for_removal += original_count - len(
                self.g.reverse_transition_node_ids[mode]
            )


        print(f"Removed {num_pts_for_removal} nodes")

    def process_valid_path(self, valid_path):
        path = [node.state for node in valid_path]
        new_path_cost = path_cost(path, self.env.batch_config_cost)
        if self.current_best_cost is None or new_path_cost < self.current_best_cost:
            self.alpha = 1.0
            self.current_best_path = path
            self.current_best_cost = new_path_cost
            # self.current_best_path_nodes = valid_path

            # extract modes
            modes = [path[0].mode]
            for p in path:
                if p.mode != modes[-1]:
                    modes.append(p.mode)

            print("Modes of new path")
            print([m.task_ids for m in modes])

            print(f"New cost: {new_path_cost} at time {time.time() - self.start_time}")
            self.update_results_tracking(new_path_cost, path)

            if self.try_shortcutting:
                print("Shortcutting path")
                shortcut_path, _ = shortcutting.robot_mode_shortcut(
                    self.env,
                    path,
                    250,
                    resolution=self.env.collision_resolution,
                    tolerance=self.env.collision_tolerance,
                )

                shortcut_path = shortcutting.remove_interpolated_nodes(shortcut_path)

                shortcut_path_cost = path_cost(
                    shortcut_path, self.env.batch_config_cost
                )

                if shortcut_path_cost < self.current_best_cost:
                    print("New cost: ", shortcut_path_cost)
                    self.update_results_tracking(shortcut_path_cost, shortcut_path)

                    self.current_best_path = shortcut_path
                    self.current_best_cost = shortcut_path_cost

                    interpolated_path = shortcut_path
                    self.current_best_path, _ = self.g.add_path_states(interpolated_path)   
            if not self.optimize and self.current_best_cost is not None:
                return   
            self.initialize_samples_and_forward_queue()       

    def generate_path(self) -> List[Node]:
        path = []
        goal, cost = self.get_lb_goal_node_and_cost()
        if math.isinf(cost) or goal is None or (self.current_best_cost is not None and cost >= self.current_best_cost):
            return path
        path.append(goal)

        n = goal

        while n.forward_parent is not None:
            path.append(n.forward_parent)
            n = n.forward_parent

        path.append(n)
        path = path[::-1]
        return path

    def expand(self, node: Optional[Node] = None) -> None:
        if node is None:
            node = self.g.root
            self.g.forward_queue = ForwarrdQueue(self.alpha)
            self.forward_closed_set = set()
            assert node.forward_cost == 0 ,(
                " root node wrong"
            )
            print("Restart reverse search ...")
            self.update_heuristic()
            print("... finished")
            
        neighbors = self.g.get_neighbors(node, space_extent=self.approximate_space_extent)
        if len(neighbors) == 0:
            return
        # add neighbors to open_queue
        edge_costs = self.g.tot_neighbors_batch_cost_cache[node.id]
        for id, edge_cost in zip(neighbors, edge_costs):
            n = self.g.nodes[id]
            assert (n.forward_parent == node) == (n.id in node.forward_children), (
                    f"Parent and children don't coincide (reverse): parent: {node.id}, child: {n.id}"
                    )
            if n.id in node.blacklist or n.forward_parent == node:
                continue
            edge = (node, n)
            if n.id != node.id and edge:
                self.g.forward_queue.heappush((edge_cost, edge))
 
    def update_heuristic(self, edge: Optional[Tuple[Node, Node]] = None) -> float:
        self.reversed_closed_set = set() #do it correctly
        if edge is None:
            self.g.reverse_queue = ReverseQueue()
            self.reversed_closed_set = set() #do it correctly
            self.g.reset_reverse_tree()
            self.g.reset_all_goal_nodes_lb_costs_to_go()
        else:
            self.update_edge_collision_cache(edge[0], edge[1], False)
            self.reversed_closed_set.add(edge[0].id)
            self.update_state(edge[0])
            # if edge[0].id not in self.reversed_closed_set:
            #     self.reversed_closed_set.add(edge[0].id)
            # else:
            #     print("")

        # Process the reverse queue until stopping conditions are met.
        num_iter = 0
        while len(self.g.reverse_queue) > 0 and (
            self.g.reverse_queue.is_smallest_priority_less_than_root_priority(self.g.root)
            or self.g.root.lb_cost_to_go_expanded < self.g.root.lb_cost_to_go
            or len(self.g.forward_queue) > 0
        ):  # TODO
            if not self.optimize and self.current_best_cost is not None:
                break

            if self.ptc.should_terminate(self.cnt, time.time() - self.start_time):
                break
            n = self.g.reverse_queue.heappop()
            num_iter += 1
            if num_iter % 100000 == 0:
                print(num_iter, ": Reverse Queue: ", len(self.g.reverse_queue))
            # If the connected cost is lower than the expanded cost, confirm it.
            if n.lb_cost_to_go < n.lb_cost_to_go_expanded:
                n.lb_cost_to_go_expanded = n.lb_cost_to_go
            else:
                n.lb_cost_to_go_expanded = np.inf
                self.update_state(n)

            self.reversed_closed_set.add(n.id)
            neighbors = self.g.get_neighbors(n, self.approximate_space_extent, search="reverse")# TODO no reverse?? check transition nodes???
            for id in neighbors:  
                if id == n.id:
                    continue
                nb = self.g.nodes[id]
                self.update_state(nb)
                # self.reversed_closed_set.add(nb.id)
        
    def update_edge_collision_cache(
        self, n0: Node, n1: Node, is_edge_collision_free: bool
    ):
        if is_edge_collision_free:
            n1.whitelist.add(n0.id)
            n0.whitelist.add(n1.id)
        else:
            n0.blacklist.add(n1.id)
            n1.blacklist.add(n0.id)

    def inconcistency_check(self, node: Node):
        self.g.reverse_queue.remove(node)
        if node.lb_cost_to_go != node.lb_cost_to_go_expanded:
            self.g.reverse_queue.heappush(node)
            
    def update_node_without_available_reverse_parent(self, node:Node):
        node.lb_cost_to_go = np.inf
        if node.reverse_parent is not None:
            node.reverse_parent.reverse_children.remove(node.id)
        node.reverse_parent = None
        node.reverse_cost_to_parent = np.inf

    def update_state(self, node: Node) -> None:
        if node.id == self.g.root.id:
            return

        # Retrieve neighbors for node.
        if node not in self.g.goal_nodes:
            in_closed_set = node.id in self.reversed_closed_set
            if node.lb_cost_to_go == node.lb_cost_to_go_expanded and in_closed_set:
                self.inconcistency_check(node)
                return
            neighbors = list(self.g.get_neighbors(node, self.approximate_space_extent, "reverse", in_closed_set))
            if not neighbors:
                self.update_node_without_available_reverse_parent(node)
                self.inconcistency_check(node)
                return
                            
            batch_cost = self.g.reverse_tot_neighbors_batch_cost_cache[node.id]

            #cannot cache them as they are always updated
            
            lb_costs_to_go_expanded = self.operation.lb_costs_to_go_expanded[neighbors]
            candidates =  lb_costs_to_go_expanded + batch_cost

            sorted_candidates_indices = np.argsort(candidates)
            compare_node = None
            best_idx = sorted_candidates_indices[0]
            if candidates[best_idx] == node.lb_cost_to_go and node.reverse_parent is not None: 
                self.inconcistency_check(node)
                return
                compare_node = self.g.nodes[neighbors[sorted_candidates_indices[0]]]
                n = self.g.nodes[neighbors[best_idx]]
                if n.state.mode == node.state.mode.prev_mode or n.id != node.reverse_parent.id:
                    if candidates[sorted_candidates_indices[1]] == node.lb_cost_to_go:
                        compare_node = self.g.nodes[neighbors[sorted_candidates_indices[1]]]

            best_parent = None
            for idx in sorted_candidates_indices:
                n = self.g.nodes[neighbors[idx]]
                if n.state.mode == node.state.mode.prev_mode:
                    continue
                if n.id == node.id:
                    continue
                assert (n.reverse_parent == node) == (n.id in node.reverse_children), (
                    f"Parent and children don't coincide (reverse): parent {node.id} of {n.id}"
                )
                #to avoid inner cycles in reverse tree (can happen when a parent appears on blacklist at some point):
                if n.reverse_parent == node:
                    continue
                if n.reverse_parent is not None and n.reverse_parent.id in node.reverse_children:
                    continue
                if n.id in node.reverse_children: 
                    continue

                best_parent = n
                best_edge_cost = batch_cost[idx]
                break

            if best_parent is not None:
                if best_parent.reverse_parent is not None:
                    assert best_parent.reverse_parent.id not in node.reverse_children, (
                        "Parent of potential parent of node is one of its children"
                    )
                assert best_parent.id not in node.reverse_children, (
                    "Potential parent of node is one of its children"
                )
                assert best_parent.reverse_parent != node, (
                    "Parent of potential parent of node is node itself"
                )
                # if best_value < node.lb_cost_to_go:
                if node.reverse_parent is not None:
                    assert node.id in node.reverse_parent.reverse_children, (
                        f"Parent and children don't coincide (reverse): parent {node.id} of {n.id}"
                    )
                self.g.update_connectivity(
                    best_parent, node, best_edge_cost, "reverse"
                ) 
                if node.reverse_parent is not None:
                    assert (node.id in best_parent.reverse_children) and (
                        node.reverse_parent == best_parent
                    ), "Not correct connected"
                if node.reverse_parent is not None:
                    assert node.id in node.reverse_parent.reverse_children, (
                        f"Parent and children don't coincide (reverse): parent {node.id} of {n.id}"
                    )
                # if compare_node is not None:
                #     n = compare_node
                #     assert compare_node.id == best_parent.id ,(
                #         "shit"
                #     )
            
                
            else:
                self.update_node_without_available_reverse_parent(node)                
        self.inconcistency_check(node)
       
    def update_results_tracking(self, cost, path):
        self.costs.append(cost)
        self.times.append(time.time() - self.start_time)
        self.all_paths.append(path)

    def initialize_samples_and_forward_queue(self):
        self.sample_manifold()
        self.expand()
        self.wasted_pops = 0
        self.processed_edges = 0

    def PlannerInitialization(self) -> None:
        q0 = self.env.get_start_pos()
        m0 = self.env.get_start_mode()

        self.reached_modes.append(m0)

        self.g = Graph(self.operation,
            State(q0, m0), 
            lambda a, b: batch_config_dist(a, b, self.distance_metric),
            lambda a, b: self.env.batch_config_cost(a, b)
        )
        self.initialize_samples_and_forward_queue()

    def get_lb_goal_node_and_cost(self) -> Node:
        min_id = np.argmin(self.operation.forward_costs[self.g.goal_node_ids], axis=0)
        best_cost = self.operation.forward_costs[self.g.goal_node_ids][min_id]
        best_node = self.g.goal_nodes[min_id]
        return best_node, best_cost

    def Plan(
        self,
    ) -> Tuple[List[State], Dict[str, List[Union[float, float, List[State]]]]]:
        self.PlannerInitialization()
        num_iter = 0
        while True:
            num_iter += 1
            if num_iter % 100000 == 0:
                print("Forward Queue: ", len(self.g.forward_queue))
            if len(self.g.forward_queue) < 1:
                self.initialize_samples_and_forward_queue()
                continue
            edge_cost, (n0, n1) = self.g.forward_queue.heappop()
            # if (n0,n1) in self.forward_closed_set:
            #     print("AAAAAAAAAAAAAA")
            self.forward_closed_set.add((n0, n1))
            # if n1.state.mode.task_ids == [3,2]:
            #     pass
            # if n1.state.mode.task_ids == [4,2]:
            #     pass
            # if n1.state.mode.task_ids == [5,2]:
            #     pass
            if (
                self.current_best_cost is None
                or n0.forward_cost + edge_cost + n1.lb_cost_to_go
                < self.current_best_cost
            ):
                if n1.forward_parent == n0:  # if its already the parent
                    self.expand(n1)
                elif (
                    n0.forward_cost + edge_cost < n1.forward_cost
                ):  # parent can improve the cost
                    assert n0.id not in n1.forward_children, (
                        "Potential parent is already a child (forward)"
                    )
                    # if n1.id in self.closed_set:
                    #     wasted_pops += 1
                    #     continue
                    # check edge sparsely now. if it is not valid, blacklist it, and continue with the next edge
                    if n0.id not in n1.whitelist:
                        if n1.id in n0.blacklist:
                            self.update_heuristic((n0, n1))
                            continue 

                        collision_free = self.env.is_edge_collision_free(
                            n0.state.q,
                            n1.state.q,
                            n0.state.mode,
                            self.env.collision_resolution,
                        )
                        self.update_edge_collision_cache(n0, n1, collision_free)
                        if not collision_free:
                            self.update_heuristic((n0, n1))
                            continue
                    self.processed_edges += 1
                    self.g.update_connectivity(n0, n1, edge_cost, "forward")
                    self.g.add_vertex(n1)
                    self.expand(n1)
                    path = self.generate_path()
                    if len(path) > 0:
                        self.process_valid_path(path)
                            
            else:
                self.initialize_samples_and_forward_queue()

            if not self.optimize and self.current_best_cost is not None:
                break

            if self.ptc.should_terminate(self.cnt, time.time() - self.start_time):
                break
        if self.costs != []:
            self.update_results_tracking(self.costs[-1], self.current_best_path)
        info = {"costs": self.costs, "times": self.times, "paths": self.all_paths}
        return self.current_best_path, info
