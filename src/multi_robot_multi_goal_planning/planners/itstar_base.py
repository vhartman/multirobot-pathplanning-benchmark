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
from itertools import chain
from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
    Mode,
)
from multi_robot_multi_goal_planning.problems.configuration import (
    Configuration,
)
from multi_robot_multi_goal_planning.problems.util import path_cost, interpolate_path

from multi_robot_multi_goal_planning.planners import shortcutting

from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
)
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
        self.vertices = set()
        self.vertices.add(self.root.id)
        self.transition_node_ids = {}  # contains the transitions at the end of the mode
        self.reverse_transition_node_ids = {}
        self.reverse_transition_node_ids[self.root.state.mode] = [self.root.id]
        self.goal_nodes = []
        self.goal_node_ids = []
        self.lb_costs_to_go_expanded = np.empty(10000000, dtype=np.float64)
        self.initialize_cache()
        # self.current_best_path_nodes = []
        self.current_best_path = []
        self.unit_n_ball_measure = ((np.pi**0.5) ** self.dim) / math.gamma(self.dim / 2 + 1)
           
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
        pass

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

    def add_node(self, new_node: Node, forward_cost:float = np.inf, lb_cost_to_go:float = np.inf) -> None:
        self.nodes[new_node.id] = new_node
        key = new_node.state.mode
        if key not in self.node_ids:
            self.node_ids[key] = []
        self.node_ids[key].append(new_node.id)
        self.operation.lb_costs_to_go_expanded = self.operation.ensure_capacity(self.operation.lb_costs_to_go_expanded, new_node.id) 
        new_node.lb_cost_to_go_expanded = lb_cost_to_go
        new_node.lb_cost_to_go = lb_cost_to_go
        self.operation.forward_costs = self.operation.ensure_capacity(self.operation.forward_costs, new_node.id) 
        new_node.forward_cost = forward_cost

    def add_vertex(self, node: Node) -> None:  
        if node.id in self.vertices:
            return
        self.vertices.add(node.id)
    
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
                    # continue
                    return False
            is_goal = True
            if next_mode is not None:
                is_goal = False
                node_next_mode.transition = node_this_mode
                node_this_mode.transition = node_next_mode
                assert this_mode.task_ids != next_mode.task_ids

            self.add_transition_node(node_this_mode, is_goal=is_goal)
            self.add_transition_node(node_next_mode, reverse=True)
            return True

    def add_transition_node(self, node:Node, is_goal:bool = False, reverse:bool = False, forward_cost:float = np.inf, lb_cost_to_go:float = np.inf):
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

        if reverse: 
            if mode in self.reverse_transition_node_ids:
                self.reverse_transition_node_ids[mode].append(node.id)  # TODO for nearest neighbor search, correct ??
            else:
                self.reverse_transition_node_ids[mode] = [node.id]

        self.nodes[node.id] = node
        self.operation.lb_costs_to_go_expanded = self.operation.ensure_capacity(self.operation.lb_costs_to_go_expanded, node.id) 
        node.lb_cost_to_go_expanded = lb_cost_to_go
        node.lb_cost_to_go = lb_cost_to_go
        self.operation.forward_costs = self.operation.ensure_capacity(self.operation.forward_costs, node.id) 
        node.forward_cost = forward_cost
        
    def add_path_node(self, node:Node, parent:Node, edge_cost:float, is_transition:bool, next_mode:Mode, current_best_cost:float): 
        self.update_connectivity(parent, node, edge_cost)
        if is_transition:
            is_goal = True
            node_next_mode = Node(self.operation, State(node.state.q, next_mode), is_transition)            
            if next_mode is not None:
                self.current_best_path_nodes.append(node_next_mode)
                self.current_best_path.append(node_next_mode.state)
                is_goal = False
                self.update_connectivity(node, node_next_mode, 0.0)

                node.transition = node_next_mode
                node_next_mode.transition = node

            self.add_transition_node(node, is_goal=is_goal, forward_cost=node.forward_cost, lb_cost_to_go=current_best_cost-node.forward_cost)
            self.add_transition_node(node_next_mode, reverse=True, forward_cost=node.forward_cost, lb_cost_to_go=current_best_cost-node.forward_cost)
        else:
            self.add_node(node, node.forward_cost, current_best_cost-node.forward_cost)

    def add_path_states(self, path:List[State], current_best_cost:float):
        self.current_best_path_nodes = []
        self.current_best_path = []
        batch_edge_cost = self.batch_cost_fun(path[:-1], path[1:])
        parent = self.root
        self.current_best_path_nodes.append(parent)
        self.current_best_path.append(parent.state)
        parent.lb_cost_to_go = current_best_cost
        parent.lb_cost_to_go_expanded = current_best_cost
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
            self.current_best_path_nodes.append(node)
            self.current_best_path.append(node.state)
            self.add_path_node(node, parent, edge_cost, is_transition, next_mode, current_best_cost)
            parent = self.current_best_path_nodes[-1]
        return self.current_best_path, self.current_best_path_nodes

    def transition_is_already_present(self, node:Node, is_goal:bool = False):
        if len(self.transition_node_ids[node.state.mode]) > 0:
            configs_list = [
                            self.nodes[id].state.q
                            for id in self.transition_node_ids[node.state.mode]
                            ]
        
            dists = self.batch_dist_fun(node.state.q, configs_list)

            if min(dists) < 1e-6:
                return True
        else:
            pass
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

        # node ids as keys
        self.neighbors_node_ids_cache = {}
        self.neighbors_array_cache = {}
        self.neighbors_fam_ids_cache = {}
        self.tot_neighbors_batch_cost_cache = {}
        self.tot_neighbors_id_cache = {}
        self.transition_node_lb_cache = {}
        self.reverse_transition_node_lb_cache = {}
        self.reverse_transition_node_array_cache = {}
         
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
        self, node: Node, space_extent: float = None, in_closed_set:bool = False
    ) -> set:
        
        
        if node.id in self.neighbors_node_ids_cache:
            if in_closed_set:
                return self.update_neighbors_with_family_of_node(node)
            return self.update_neighbors(node)

        key = node.state.mode
        self.update_cache(key)

        best_nodes_arr, best_transitions_arr= np.zeros((0, self.dim)), np.zeros((0, self.dim))
        informed_measure = 1
        if space_extent is not None:
            informed_measure = space_extent
        
        indices_transitions, indices = np.empty((0,), dtype=int), np.empty((0,), dtype=int)
        node_ids, transition_node_ids = np.empty((0,), dtype=int), np.empty((0,), dtype=int)
        if key in self.node_ids:
            dists = self.batch_dist_fun(node.state.q, self.node_array_cache[key])
            r_star = self.get_r_star(
                len(self.node_ids[key]), informed_measure, self.unit_n_ball_measure
            )
            indices = find_nearest_indices(dists, r_star)
            node_ids = np.array(self.node_ids[key])[indices]

            best_nodes_arr = self.node_array_cache[key][indices]

        if key in self.transition_node_ids:
            transition_dists = self.batch_dist_fun(
                node.state.q, self.transition_node_array_cache[key]
            )

            r_star = self.get_r_star(
                len(self.transition_node_ids[key]),
                informed_measure,
                self.unit_n_ball_measure
            )
            if len(self.transition_node_ids[key]) == 1:
                r_star = 1e6

            indices_transitions = find_nearest_indices(transition_dists, r_star)
            transition_node_ids = np.array(self.transition_node_ids[key])[
                indices_transitions
            ]
            best_transitions_arr = self.transition_node_array_cache[key][indices_transitions]

        if key in self.reverse_transition_node_array_cache:
            reverse_transition_dists = self.batch_dist_fun(
                node.state.q, self.reverse_transition_node_array_cache[key]
            )
            r_star = self.get_r_star(
                len(self.reverse_transition_node_ids[key]),
                informed_measure,
                self.unit_n_ball_measure
               
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

        if node.is_transition and node.transition is not None:
            all_ids = np.concatenate((all_ids, np.array([node.transition.id])))
            arr = np.vstack([arr, node.transition.state.q.state()])


        assert node.id in all_ids, (
           " ohhh nooooooooooooooo"
        )
        # if node.id not in all_ids:
        #     pass

        #remove node itself
        mask = all_ids != node.id
        all_ids = all_ids[mask]
        arr = arr[mask]

        assert node.id not in self.neighbors_node_ids_cache,("2 already calculated")
        self.neighbors_node_ids_cache[node.id] = all_ids 
        self.neighbors_array_cache[node.id] = arr

        return self.update_neighbors(node)

    def update_neighbors_with_family_of_node(self, node:Node):
        neighbors_fam = set()
        if node.id in self.neighbors_fam_ids_cache:
            neighbors_fam = self.neighbors_fam_ids_cache[node.id]

        combined_fam = node.forward_fam | node.reverse_fam
        blacklist = node.blacklist
        if len(blacklist) > 0 and len(combined_fam) > 0:
            combined_fam =  combined_fam - blacklist
        if  neighbors_fam != combined_fam:
            self.neighbors_fam_ids_cache[node.id] = combined_fam 
            node_ids = self.neighbors_node_ids_cache[node.id]
            arr = self.neighbors_array_cache[node.id]
            mask_node_ids =  np.array(list(combined_fam - set(node_ids)))
            if mask_node_ids.size > 0:
                arr = np.array(
                    [self.nodes[id].state.q.q for id in mask_node_ids],
                    dtype=np.float64,
                )
                arr = np.concatenate((arr, self.neighbors_array_cache[node.id]))
                self.tot_neighbors_batch_cost_cache[node.id] = self.batch_cost_fun(node.state.q, arr)
                self.tot_neighbors_id_cache[node.id] = np.concatenate((mask_node_ids, self.neighbors_node_ids_cache[node.id]))
                assert len(self.tot_neighbors_id_cache[node.id]) == len(self.tot_neighbors_batch_cost_cache[node.id]),(
                    "forward not right"
                )
                return self.tot_neighbors_id_cache[node.id]
        if node.id not in self.tot_neighbors_id_cache:
            arr = self.neighbors_array_cache[node.id]
            self.tot_neighbors_batch_cost_cache[node.id] = self.batch_cost_fun(node.state.q, arr)
            self.tot_neighbors_id_cache[node.id] = self.neighbors_node_ids_cache[node.id]
        assert len(self.tot_neighbors_id_cache[node.id]) == len(self.tot_neighbors_batch_cost_cache[node.id]),(
            "forward not right"
        )
        return self.tot_neighbors_id_cache[node.id]

    def update_neighbors(self, node:Node): # only needed for forward
        blacklist = node.blacklist
        if len(blacklist) > 0:
            node_ids = self.neighbors_node_ids_cache[node.id]
            arr =  self.neighbors_array_cache[node.id]
            mask_node_ids =  ~np.isin(node_ids, blacklist)
            if not mask_node_ids.all():
                node_ids = node_ids[mask_node_ids]
                arr = arr[mask_node_ids]
                self.neighbors_node_ids_cache[node.id] = node_ids
                self.neighbors_array_cache[node.id] = arr

        return self.update_neighbors_with_family_of_node(node)
            
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
                print("uhhh")
            self.add_vertex(n1)
            self.add_vertex(n0)
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
            assert n1.id not in n0.reverse_children, (
                "already a child")
            n0.reverse_children.append(n1.id)
            n1.reverse_cost_to_parent = edge_cost
            assert [
                        (self.nodes[child].reverse_parent, child)
                        for child in n1.reverse_parent.reverse_children
                        if self.nodes[child].reverse_parent is None
                        or self.nodes[child].reverse_parent.id != n1.reverse_parent.id
                    ] == [], (
                        "new parent and new children not correct")
            n1.reverse_fam.add(n0.id)
            n0.reverse_fam.add(n1.id)

class Informed():
    def __init__(self, env:BaseProblem, sample_mode, conf_type, informed_with_lb):
        self.env = env
        self.conf_type = conf_type
        self.sample_mode = sample_mode
        self.informed_with_lb = informed_with_lb

    def sample_unit_ball(self, dim:int) -> NDArray:
        """ 
        Samples a point uniformly from a n-dimensional unit ball centered at the origin.

        Args: 
            n (int): Dimension of the ball.

        Returns: 
            NDArray: Sampled point from the unit ball. """
        # u = np.random.uniform(-1, 1, dim)
        # norm = np.linalg.norm(u)
        # r = np.random.random() ** (1.0 / dim)
        # return r * u / norm
        u = np.random.normal(0, 1, dim)
        norm = np.linalg.norm(u)
        # Generate radius with correct distribution
        r = np.random.random() ** (1.0 / dim)
        return (r / norm) * u

    def compute_PHS_matrices(self, a:NDArray, b:NDArray, c:float) -> Tuple[NDArray, NDArray]:
        """
        Computes transformation matrix and center for a Prolate Hyperspheroid (PHS) defined by endpoints a and b and cost parameter c.

        Args:
            a (NDArray): Start point of the PHS.
            b (NDArray): End point of the PHS.
            c (float): Cost parameter defining scaling of the PHS.

        Returns:
            Tuple:    
                - NDArray: Transformation matrix (rotation and scaling) of the PHS.
                - NDArray: Center of the PHS, calculated as the midpoint between a and b.
               
        """

        dim = len(a)
        diff = b - a

        # Calculate the center of the PHS.
        center = (a + b) / 2
        # The transverse axis in the world frame.
        c_min = np.linalg.norm(diff)

        # The first column of the identity matrix.
        # one_1 = np.eye(a1.shape[0])[:, 0]
        a1 = diff / c_min
        e1 = np.zeros(dim)
        e1[0] = 1.0

        # Optimized rotation matrix calculation
        U, S, Vt = np.linalg.svd(np.outer(a1, e1))
        # Sigma = np.diag(S)
        # lam = np.eye(Sigma.shape[0])
        lam = np.eye(dim)
        lam[-1, -1] = np.linalg.det(U) * np.linalg.det(Vt.T)
        # Calculate the rotation matrix.
        # cwe = np.matmul(U, np.matmul(lam, Vt))
        cwe = U @ lam @ Vt
        # Get the radius of the first axis of the PHS.
        # r1 = c / 2
        # Get the radius of the other axes of the PHS.
        # rn = [np.sqrt(c**2 - c_min**2) / 2] * (dim - 1)
        # Create a vector of the radii of the PHS.
        # r = np.diag(np.array([r1] + rn))
        # sqrt_term = c**2 - c_min**2

        # if sqrt_term < 0 or np.isnan(sqrt_term):
        #     print("hallo")
        r = np.diag([c * 0.5] + [np.sqrt(c**2 - c_min**2) * 0.5] * (dim - 1))


        return cwe @ r, center

    def sample_phs_with_given_matrices(self, rot:NDArray, center:NDArray) -> NDArray:
        """
        Samples point from a prolate hyperspheroid (PHS) defined by the given rotation matrix and center.

        Args:
            rot (NDArray): Transformation matrix (rotation and scaling) for the PHS.
            center (NDArray): Center point of the PHS.

        Returns:
            NDArray: Sampled point from the PHS.
        """

        dim = len(center)
        x_ball = self.sample_unit_ball(dim)
        return rot @ x_ball + center

    def lb_cost_from_start(self, g, state):
        if state.mode not in g.reverse_transition_node_array_cache:
            g.reverse_transition_node_array_cache[state.mode] = np.array(
                [g.nodes[id].state.q.q for id in g.reverse_transition_node_ids[state.mode]],
                dtype=np.float64,
            )

        if state.mode not in g.reverse_transition_node_lb_cache:
            g.reverse_transition_node_lb_cache[state.mode] = np.array(
                [
                    g.nodes[id].forward_cost
                    for id in g.reverse_transition_node_ids[state.mode]
                ],
                dtype=np.float64,
            )

        costs_to_transitions = self.env.batch_config_cost(
            state.q,
            g.reverse_transition_node_array_cache[state.mode],
        )

        min_cost = np.min(
            g.reverse_transition_node_lb_cache[state.mode] + costs_to_transitions
        )
        return min_cost

    def lb_cost_from_goal(self, g, state):
        if state.mode not in g.transition_node_ids:
            return np.inf

        if state.mode not in g.transition_node_array_cache:
            g.transition_node_array_cache[state.mode] = np.array(
                [g.nodes[id].state.q.q for id in g.transition_node_ids[state.mode]],
                dtype=np.float64,
            )

        if state.mode not in g.transition_node_lb_cache:
            g.transition_node_lb_cache[state.mode] = np.array(
                [g.nodes[id].lb_cost_to_go for id in g.transition_node_ids[state.mode]],
                dtype=np.float64,
            )

        costs_to_transitions = self.env.batch_config_cost(
            state.q,
            g.transition_node_array_cache[state.mode],
        )

        min_cost = np.min(
            g.transition_node_lb_cache[state.mode] + costs_to_transitions
        )
        return min_cost

    def can_improve(
        self, 
        g,
        rnd_state: State, 
        path: List[State], 
        start_index, 
        end_index, 
        path_segment_costs
        ) -> bool:
        # path_segment_costs = env.batch_config_cost(path[:-1], path[1:])

        # compute the local cost
        path_cost_from_start_to_index = np.sum(path_segment_costs[:start_index])
        path_cost_from_goal_to_index = np.sum(path_segment_costs[end_index:])
        path_cost = np.sum(path_segment_costs)

        if start_index == 0:
            assert path_cost_from_start_to_index == 0
        if end_index == len(path) - 1:
            assert path_cost_from_goal_to_index == 0

        path_cost_from_index_to_index = (
            path_cost - path_cost_from_goal_to_index - path_cost_from_start_to_index
        )

        # print(path_cost_from_index_to_index)

        lb_cost_from_start_index_to_state = self.env.config_cost(
            rnd_state.q, path[start_index].q
        )
        if path[start_index].mode != rnd_state.mode and self.informed_with_lb:
            start_state = path[start_index]
            lb_cost_from_start_to_state = self.lb_cost_from_start(g, rnd_state)
            lb_cost_from_start_to_index = self.lb_cost_from_start(g, start_state)

            lb_cost_from_start_index_to_state = max(
                (lb_cost_from_start_to_state - lb_cost_from_start_to_index),
                lb_cost_from_start_index_to_state,
            )


        lb_cost_from_state_to_end_index = self.env.config_cost(
            rnd_state.q, path[end_index].q
        )
        if path[end_index].mode != rnd_state.mode and self.informed_with_lb:
            goal_state = path[end_index]
            lb_cost_from_goal_to_state = self.lb_cost_from_goal(g, rnd_state)
            lb_cost_from_goal_to_index = self.lb_cost_from_goal(g, goal_state)
            lb_cost_from_state_to_end_index = max(
                (lb_cost_from_goal_to_state - lb_cost_from_goal_to_index),
                lb_cost_from_state_to_end_index,
            )

        # print("can_imrpove")

        # print("start", lb_cost_from_start_index_to_state)
        # print("end", lb_cost_from_state_to_end_index)

        # print('start index', start_index)
        # print('end_index', end_index)

        # assert(lb_cost_from_start_index_to_state >= 0)
        # assert(lb_cost_from_state_to_end_index >= 0)

        # print("segment cost", path_cost_from_index_to_index)
        # print(
        #     "lb cost",
        #     lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index,
        # )

        if (
            path_cost_from_index_to_index
            > lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index
        ):
            return True

        return False

    def get_inbetween_modes(self, start_mode, end_mode):
        """
        Find all possible paths from start_mode to end_mode.

        Args:
            start_mode: The starting mode object
            end_mode: The ending mode object

        Returns:
            A list of lists, where each inner list represents a valid path
            from start_mode to end_mode (inclusive of both).
        """
        # Store all found paths
        open_paths = [[start_mode]]

        in_between_modes = set()
        in_between_modes.add(start_mode)
        in_between_modes.add(end_mode)

        while len(open_paths) > 0:
            p = open_paths.pop()
            last_mode = p[-1]

            if last_mode == end_mode:
                for m in p:
                    in_between_modes.add(m)
                continue

            if len(last_mode.next_modes) > 0:
                for mode in last_mode.next_modes:
                    new_path = p.copy()
                    new_path.append(mode)
                    open_paths.append(new_path)

        return list(in_between_modes)

    def generate_samples(
        self,
        g, 
        reached_modes,
        batch_size,
        path,
        max_attempts_per_sample=200,
        locally_informed_sampling=True,
        try_direct_sampling=True,
    ):
        new_samples = []
        path_segment_costs = self.env.batch_config_cost(path[:-1], path[1:])

        in_between_mode_cache = {}

        num_attempts = 0
        while len(new_samples) < batch_size:
            if num_attempts > batch_size:
                break

            num_attempts += 1
            # print(len(new_samples))
            # sample mode
            if locally_informed_sampling:
                for _ in range(500):
                    start_ind = random.randint(0, len(path) - 1)
                    end_ind = random.randint(0, len(path) - 1)

                    if end_ind - start_ind > 2:
                        # if end_ind - start_ind > 2 and end_ind - start_ind < 50:
                        current_cost = sum(path_segment_costs[start_ind:end_ind])
                        lb_cost = self.env.config_cost(path[start_ind].q, path[end_ind].q)

                        if lb_cost < current_cost:
                            break

                # TODO: we need to sample from the set of all reachable modes here
                # not only from the modes on the path
                if (
                    path[start_ind].mode,
                    path[end_ind].mode,
                ) not in in_between_mode_cache:
                    in_between_modes = self.get_inbetween_modes(
                        path[start_ind].mode, path[end_ind].mode
                    )
                    in_between_mode_cache[
                        (path[start_ind].mode, path[end_ind].mode)
                    ] = in_between_modes

                # print(in_between_mode_cache[(path[start_ind].mode, path[end_ind].mode)])

                m = random.choice(
                    in_between_mode_cache[(path[start_ind].mode, path[end_ind].mode)]
                )

                # k = random.randint(start_ind, end_ind)
                # m = path[k].mode
            else:
                start_ind = 0
                end_ind = len(path) - 1
                m = self.sample_mode(reached_modes, "uniform_reached", True)

            # print(m)

            current_cost = sum(path_segment_costs[start_ind:end_ind])

            # tmp = 0
            # for i in range(start_ind, end_ind):
            #     tmp += self.env.config_cost(path[i].q, path[i+1].q)

            # print(current_cost, tmp)

            # plt.figure()
            # samples = []
            # for _ in range(500):
            #     sample = samplePHS(np.array([-1, 1, 0]), np.array([1, -1, 0]), 3)
            #     # sample = samplePHS(np.array([[-1], [0]]), np.array([[1], [0]]), 3)
            #     samples.append(sample[:2])
            #     print("sample", sample)

            # plt.scatter([a[0] for a in samples], [a[1] for a in samples])
            # plt.show()

            focal_points = np.array(
                [path[start_ind].q.state(), path[end_ind].q.state()], dtype=np.float64
            )

            precomputed_phs_matrices = {}
            precomputed_robot_cost_bounds = {}

            obv_inv_attempts = 0
            sample_in_collision = 0

            for k in range(max_attempts_per_sample):
                had_to_be_clipped = False
                if not try_direct_sampling or self.env.cost_metric != "euclidean":
                    # completely random sample configuration from the (valid) domain robot by robot
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
                else:
                    # sample by sampling each agent separately
                    q = []
                    for i in range(len(self.env.robots)):
                        r = self.env.robots[i]
                        lims = self.env.limits[:, self.env.robot_idx[r]]
                        if lims[0, 0] < lims[1, 0]:
                            if i not in precomputed_robot_cost_bounds:
                                if self.env.cost_reduction == "sum":
                                    precomputed_robot_cost_bounds[i] = (
                                        current_cost
                                        - sum(
                                            [
                                                np.linalg.norm(
                                                    path[start_ind].q[j]
                                                    - path[end_ind].q[j]
                                                )
                                                for j in range(len(self.env.robots))
                                                if j != i
                                            ]
                                        )
                                    )
                                else:
                                    precomputed_robot_cost_bounds[i] = current_cost

                            if (
                                np.linalg.norm(
                                    path[start_ind].q[i] - path[end_ind].q[i]
                                )
                                < 1e-3
                            ):
                                qr = (
                                    np.random.rand(self.env.robot_dims[r])
                                    * (lims[1, :] - lims[0, :])
                                    + lims[0, :]
                                )
                            else:
                                # print("cost", current_cost)
                                # print("robot cst", c_robot_bound)
                                # print(
                                #     np.linalg.norm(
                                #         path[start_ind].q[i] - path[end_ind].q[i]
                                #     )
                                # )

                                if i not in precomputed_phs_matrices:
                                    precomputed_phs_matrices[i] = self.compute_PHS_matrices(
                                        path[start_ind].q[i],
                                        path[end_ind].q[i],
                                        precomputed_robot_cost_bounds[i],
                                    )

                                qr = self.sample_phs_with_given_matrices(
                                    *precomputed_phs_matrices[i]
                                )

                                # plt.figure()
                                # samples = []
                                # for _ in range(500):
                                #     sample = sample_phs_with_given_matrices(
                                #         *precomputed_phs_matrices[i]
                                #     )
                                #     # sample = samplePHS(np.array([[-1], [0]]), np.array([[1], [0]]), 3)
                                #     samples.append(sample[:2])
                                #     print("sample", sample)

                                # plt.scatter(
                                #     [a[0] for a in samples], [a[1] for a in samples]
                                # )
                                # plt.show()

                                # qr = samplePHS(path[start_ind].q[i], path[end_ind].q[i], c_robot_bound)
                                # qr = rejection_sample_from_ellipsoid(
                                #     path[start_ind].q[i], path[end_ind].q[i], c_robot_bound
                                # )

                                # if np.linalg.norm(path[start_ind].q[i] - qr) + np.linalg.norm(path[end_ind].q[i] - qr) > c_robot_bound:
                                #     print("AAAAAAAAA")
                                #     print(np.linalg.norm(path[start_ind].q[i] - qr) + np.linalg.norm(path[end_ind].q[i] - qr), c_robot_bound)

                                clipped = np.clip(qr, lims[0, :], lims[1, :])
                                if not np.array_equal(clipped, qr):
                                    had_to_be_clipped = True
                                    break
                                    # print("AAA")

                        q.append(qr)

                if had_to_be_clipped:
                    continue

                q = self.conf_type.from_list(q)

                if sum(self.env.batch_config_cost(q, focal_points)) > current_cost:
                    # print(path[start_ind].mode, path[end_ind].mode, m)
                    # print(
                    #     current_cost,
                    #     self.self.env.config_cost(path[start_ind].q, q)
                    #     + self.env.config_cost(path[end_ind].q, q),
                    # )
                    # if can_improve(State(q, m), path, start_ind, end_ind):
                    #     assert False

                    obv_inv_attempts += 1

                    continue

                # if can_improve(State(q, m), path, 0, len(path)-1):
                # if can_improve(State(q, m), path, start_ind, end_ind):
                if not self.env.is_collision_free(q, m):
                    sample_in_collision += 1
                    continue

                if self.can_improve(
                    g, State(q, m), path, start_ind, end_ind, path_segment_costs
                ):
                    # if self.env.is_collision_free(q, m) and can_improve(State(q, m), path, 0, len(path)-1):
                    new_samples.append(State(q, m))
                    break

            # print("inv attempt", obv_inv_attempts)
            # print("coll", sample_in_collision)

        print(len(new_samples) / num_attempts)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter([a.q[0][0] for a in new_samples], [a.q[0][1] for a in new_samples], [a.q[0][2] for a in new_samples])
        # ax.scatter([a.q[1][0] for a in new_samples], [a.q[1][1] for a in new_samples], [a.q[1][2] for a in new_samples])
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.scatter([a.q[0][0] for a in new_samples], [a.q[0][1] for a in new_samples])
        # ax.scatter([a.q[1][0] for a in new_samples], [a.q[1][1] for a in new_samples])
        # plt.show()

        return new_samples

    def can_transition_improve(
            self, 
            g, 
            transition, 
            path, 
            start_index, 
            end_index):
        path_segment_costs = self.env.batch_config_cost(path[:-1], path[1:])

        # compute the local cost
        path_cost_from_start_to_index = np.sum(path_segment_costs[:start_index])
        path_cost_from_goal_to_index = np.sum(path_segment_costs[end_index:])
        path_cost = np.sum(path_segment_costs)

        if start_index == 0:
            assert path_cost_from_start_to_index == 0
        if end_index == len(path) - 1:
            assert path_cost_from_goal_to_index == 0

        path_cost_from_index_to_index = (
            path_cost - path_cost_from_goal_to_index - path_cost_from_start_to_index
        )

        # print(path_cost_from_index_to_index)

        rnd_state_mode_1 = State(transition[0], transition[1])
        rnd_state_mode_2 = State(transition[0], transition[2])

        lb_cost_from_start_index_to_state = self.env.config_cost(
            rnd_state_mode_1.q, path[start_index].q
        )
        if path[start_index].mode != rnd_state_mode_1.mode and self.informed_with_lb:
            start_state = path[start_index]
            lb_cost_from_start_to_state = self.lb_cost_from_start(g, rnd_state_mode_1)
            lb_cost_from_start_to_index = self.lb_cost_from_start(g, start_state)

            lb_cost_from_start_index_to_state = max(
                (lb_cost_from_start_to_state - lb_cost_from_start_to_index),
                lb_cost_from_start_index_to_state,
            )

        lb_cost_from_state_to_end_index = self.env.config_cost(
            rnd_state_mode_2.q, path[end_index].q
        )
        if path[end_index].mode != rnd_state_mode_2.mode and self.informed_with_lb:
            goal_state = path[end_index]
            lb_cost_from_goal_to_state = self.lb_cost_from_goal(g, rnd_state_mode_2)
            lb_cost_from_goal_to_index = self.lb_cost_from_goal(g, goal_state)

            lb_cost_from_state_to_end_index = max(
                (lb_cost_from_goal_to_state - lb_cost_from_goal_to_index),
                lb_cost_from_state_to_end_index,
            )

        # print("can_imrpove")

        # print("start", lb_cost_from_start_index_to_state)
        # print("end", lb_cost_from_state_to_end_index)

        # print('start index', start_index)
        # print('end_index', end_index)

        # assert(lb_cost_from_start_index_to_state >= 0)
        # assert(lb_cost_from_state_to_end_index >= 0)

        # print("segment cost", path_cost_from_index_to_index)
        # print(
        #     "lb cost",
        #     lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index,
        # )

        if (
            path_cost_from_index_to_index
            > lb_cost_from_start_index_to_state + lb_cost_from_state_to_end_index
        ):
            return True

        return False

    def generate_transitions(
        self,
        g, 
        reached_modes,
        batch_size, 
        path, 
        locally_informed_sampling=False, 
        max_attempts_per_sample=100
        ):
        if len(self.env.tasks) == 1:
            return []

        new_transitions = []
        num_attempts = 0
        path_segment_costs = self.env.batch_config_cost(path[:-1], path[1:])

        in_between_mode_cache = {}

        while len(new_transitions) < batch_size:
            num_attempts += 1

            if num_attempts > batch_size:
                break

            # print(len(new_samples))
            # sample mode
            if locally_informed_sampling:
                while True:
                    start_ind = random.randint(0, len(path) - 1)
                    end_ind = random.randint(0, len(path) - 1)

                    if (
                        path[end_ind].mode != path[start_ind].mode
                        and end_ind - start_ind > 2
                    ):
                        # if end_ind - start_ind > 2 and end_ind - start_ind < 50:
                        current_cost = sum(path_segment_costs[start_ind:end_ind])
                        lb_cost = self.env.config_cost(path[start_ind].q, path[end_ind].q)

                        if lb_cost < current_cost:
                            break

                if (
                    path[start_ind].mode,
                    path[end_ind].mode,
                ) not in in_between_mode_cache:
                    in_between_modes = self.get_inbetween_modes(
                        path[start_ind].mode, path[end_ind].mode
                    )
                    in_between_mode_cache[
                        (path[start_ind].mode, path[end_ind].mode)
                    ] = in_between_modes

                # print(in_between_mode_cache[(path[start_ind].mode, path[end_ind].mode)])

                mode = random.choice(
                    in_between_mode_cache[(path[start_ind].mode, path[end_ind].mode)]
                )

                # k = random.randint(start_ind, end_ind)
                # mode = path[k].mode
            else:
                start_ind = 0
                end_ind = len(path) - 1
                mode = self.sample_mode(reached_modes,"uniform_reached", True)

            # print(m)

            current_cost = sum(path_segment_costs[start_ind:end_ind])

            # sample transition at the end of this mode
            possible_next_task_combinations = self.env.get_valid_next_task_combinations(mode)
            if len(possible_next_task_combinations) > 0:
                ind = random.randint(0, len(possible_next_task_combinations) - 1)
                active_task = self.env.get_active_task(
                    mode, possible_next_task_combinations[ind]
                )
            else:
                continue

            goals_to_sample = active_task.robots

            goal_sample = active_task.goal.sample(mode)

            for k in range(max_attempts_per_sample):
                # completely random sample configuration from the (valid) domain robot by robot
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

                if (
                    self.env.config_cost(path[start_ind].q, q)
                    + self.env.config_cost(path[end_ind].q, q)
                    > current_cost
                ):
                    continue

                if self.env.is_terminal_mode(mode):
                    assert False
                else:
                    next_mode = self.env.get_next_mode(q, mode)

                if self.can_transition_improve(
                    g, (q, mode, next_mode), path, start_ind, end_ind
                ) and self.env.is_collision_free(q, mode):
                    new_transitions.append((q, mode, next_mode))
                    break

        print(len(new_transitions) / num_attempts)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter([a.q[0][0] for a in new_samples], [a.q[0][1] for a in new_samples], [a.q[0][2] for a in new_samples])
        # ax.scatter([a.q[1][0] for a in new_samples], [a.q[1][1] for a in new_samples], [a.q[1][2] for a in new_samples])
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.scatter([a[0][0][0] for a in new_transitions], [a[0][0][1] for a in new_transitions])
        # ax.scatter([a[0][1][0] for a in new_transitions], [a[0][1][1] for a in new_transitions])
        # plt.show()

        return new_transitions

# taken from https://github.com/marleyshan21/Batch-informed-trees/blob/master/python/BIT_Star.py
# needed adaption to work.
class BaseITstar(ABC):
    """
    Represents the base class for IT*-based algorithms, providing core functionalities for motion planning.
    """
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
        self.current_best_path_nodes = None
        self.cnt = 0
        self.operation = Operation()

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
        added_transitions = 0
        # while len(transitions) < transistion_batch_size:
        while added_transitions < transistion_batch_size:

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

                transitions.append((q, mode, next_mode))

                # print(mode, mode.next_modes)

                if next_mode not in self.reached_modes and next_mode is not None:
                    self.reached_modes.append(next_mode)

                if self.g.add_transition_nodes(transitions):
                    added_transitions +=1
                transitions = []
            # else:
            #     if mode.task_ids == [3, 8]:
            #         self.env.show(True)
        print(f"Adding {added_transitions} transitions")
        return

    def sample_around_path(self, path):
        # sample index
        interpolated_path = interpolate_path(path)
        # interpolated_path = current_best_path
        new_states_from_path_sampling = []
        new_transitions_from_path_sampling = []
        for _ in range(200):
            idx = random.randint(0, len(interpolated_path) - 2)
            state = interpolated_path[idx]

            # this is a transition. we would need to figure out which robots are active and not sample those
            q = []
            if (
                state.mode != interpolated_path[idx + 1].mode
                and np.linalg.norm(
                    state.q.state() - interpolated_path[idx + 1].q.state()
                )
                < 1e-5
            ):
                next_task_ids = interpolated_path[idx + 1].mode.task_ids

                # TODO: this seems to move transitions around
                task = self.env.get_active_task(state.mode, next_task_ids)
                involved_robots = task.robots
                for i in range(len(self.env.robots)):
                    r = self.env.robots[i]
                    if r in involved_robots:
                        qr = state.q[i] * 1.0
                    else:
                        qr_mean = state.q[i] * 1.0

                        qr = np.random.rand(len(qr_mean)) * 0.5 + qr_mean

                        lims = self.env.limits[:, self.env.robot_idx[r]]
                        if lims[0, 0] < lims[1, 0]:
                            qr = np.clip(qr, lims[0, :], lims[1, :])

                    q.append(qr)

                q = self.conf_type.from_list(q)

                if self.env.is_collision_free(q, state.mode):
                    new_transitions_from_path_sampling.append(
                        (q, state.mode, interpolated_path[idx + 1].mode)
                    )

            else:
                for i in range(len(self.env.robots)):
                    r = self.env.robots[i]
                    qr_mean = state.q[i]

                    qr = np.random.rand(len(qr_mean)) * 0.5 + qr_mean

                    lims = self.env.limits[:, self.env.robot_idx[r]]
                    if lims[0, 0] < lims[1, 0]:
                        qr = np.clip(qr, lims[0, :], lims[1, :])

                    q.append(qr)

                q = self.conf_type.from_list(q)

                if self.env.is_collision_free(q, state.mode):
                    rnd_state = State(q, state.mode)
                    new_states_from_path_sampling.append(rnd_state)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter([a.q[0][0] for a in new_states_from_path_sampling], [a.q[0][1] for a in new_states_from_path_sampling], [a.q[0][2] for a in new_states_from_path_sampling])
        # ax.scatter([a.q[1][0] for a in new_states_from_path_sampling], [a.q[1][1] for a in new_states_from_path_sampling], [a.q[1][2] for a in new_states_from_path_sampling])
        # ax.scatter([a.q[2][0] for a in new_states_from_path_sampling], [a.q[2][1] for a in new_states_from_path_sampling], [a.q[1][2] for a in new_states_from_path_sampling])
        # plt.show()

        return new_states_from_path_sampling, new_transitions_from_path_sampling

    def add_sample_batch(self):
        # add new batch of nodes
        while True:
            
            if len(self.g.nodes) > 1:
                sample_batch_size = 350
                transition_batch_size = 350
            else:
                sample_batch_size = 250
                transition_batch_size = 250



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
            self.sample_valid_uniform_transitions(
                transistion_batch_size=effective_uniform_transition_batch_size,
                cost=self.current_best_cost,
            )
            # new_transitions = self.sample_valid_uniform_transitions(
            #     transistion_batch_size=effective_uniform_transition_batch_size,
            #     cost=self.current_best_cost,
            # )


            # self.g.add_transition_nodes(new_transitions)
            # print(f"Adding {len(new_transitions)} transitions")
            
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
    
    def prune_set_of_expanded_nodes(self):
        if not self.remove_nodes:
            return self.g.vertices, []

        vertices = []
        stack = [self.g.root.id]
        children_to_be_removed = []
        focal_points = {}
        focal_points_transition = {}
        inter_costs = {}
        count = 0
        while len(stack) > 0:
            id = stack.pop()
            node = self.g.nodes[id]
            key = node.state.mode
            
            if node.is_transition:
                if key in self.g.reverse_transition_node_ids:
                    if node.id in self.g.reverse_transition_node_ids[key]:
                        stack.extend(node.forward_children)
                        count +=1
                        continue
            
            if key not in self.start_transition_arrays:
                if key not in focal_points:
                    focal_points[key] = np.array(
                    [self.g.root.state.q.state(), self.current_best_path[-1].q.state()],
                    dtype=np.float64,
                    )
                    focal_points_transition[key] =  focal_points[key]

                    inter_costs[key] = self.current_best_cost
            else:
                if key not in focal_points:
                    focal_points[key] = np.array(
                        [self.start_transition_arrays[key], self.end_transition_arrays[key]],
                        dtype=np.float64,
                    )
                    focal_points_transition[key] = np.array(
                    [self.start_transition_arrays[key]],
                    dtype=np.float64,
                    )
                    inter_costs[key] =  self.intermediate_mode_costs[key]
            

            if node in self.current_best_path_nodes:
                flag = False
            elif not node.is_transition:
                flag = sum(self.env.batch_config_cost(node.state.q, focal_points[key])) > inter_costs[key]
                if not flag:
                        if node.id not in self.g.vertices:
                            flag = True
            else:
                flag = sum(self.env.batch_config_cost(node.state.q, focal_points_transition[key])) > inter_costs[key]
                
            if flag:
                children = [node.id]
                if node.forward_parent is not None:
                    node.forward_parent.forward_children.remove(node.id)
                    node.forward_parent.forward_fam.remove(node.id)
                    node.forward_fam.remove(node.forward_parent.id)
                    node.forward_parent = None
                to_be_removed = []
                while len(children) > 0:
                    child_id = children.pop()
                    to_be_removed.append(child_id)
                    children.extend(self.g.nodes[child_id].forward_children)
                children_to_be_removed.extend(to_be_removed)
            else:
                vertices.append(id)
                stack.extend(node.forward_children)

        goal_mask = np.array([item in children_to_be_removed for item in self.g.goal_node_ids])
        goal_nodes = np.array(self.g.goal_node_ids)[goal_mask]
        if goal_nodes.size > 0:
            for goal in goal_nodes:
                goal_node = self.g.nodes[goal]
                goal_node.forward_cost = np.inf
                goal_node.forward_cost_to_parent = np.inf
                goal_node.forward_children = []
                goal_node.forward_parent = None
                goal_node.forward_fam = set()
                children_to_be_removed.remove(goal)
                key = goal_node.state.mode
                vertices.append(goal)
        
        return vertices, children_to_be_removed

    def remove_nodes_in_graph(self):
        num_pts_for_removal = 0
        vertices_to_keep, vertices_to_be_removed = self.prune_set_of_expanded_nodes()
        # vertices_to_keep = list(chain.from_iterable(self.g.vertices.values()))
        # Remove elements from g.nodes_ids
        for mode in list(self.g.node_ids.keys()):# Avoid modifying dict while iterating
            # start = random.choice(self.g.reverse_transition_node_ids[mode])
            # goal = random.choice(self.g.re)
            if mode not in self.start_transition_arrays:
                focal_points = np.array(
                [self.g.root.state.q.state(), self.current_best_path[-1].q.state()],
                dtype=np.float64,
                )
                cost = self.current_best_cost
                
            else:
                focal_points = np.array(
                    [self.start_transition_arrays[mode], self.end_transition_arrays[mode]],
                    dtype=np.float64,
                )
                cost =  self.intermediate_mode_costs[mode]
            original_count = len(self.g.node_ids[mode])
            self.g.node_ids[mode] = [
                id
                for id in self.g.node_ids[mode]
                if id in vertices_to_keep or (id not in vertices_to_be_removed and sum(
                    self.env.batch_config_cost(self.g.nodes[id].state.q, focal_points))
                <= cost)
            ]
            assert[id for id in self.g.node_ids[mode] if id in vertices_to_be_removed]== [],(
                "hoh"
            )
            num_pts_for_removal += original_count - len(self.g.node_ids[mode])
        # Remove elements from g.transition_node_ids
        self.g.reverse_transition_node_ids = {}

        for mode in list(self.g.transition_node_ids.keys()):
            self.g.reverse_transition_node_ids[mode] = []
            if self.env.is_terminal_mode(mode):
                continue
            
            if mode not in self.start_transition_arrays:
                focal_points = np.array(
                [self.g.root.state.q.state(), self.current_best_path[-1].q.state()],
                dtype=np.float64,
                )
                cost = self.current_best_cost
            else:
                focal_points = np.array(
                    [self.start_transition_arrays[mode]],
                    dtype=np.float64,
                )
                cost =  self.intermediate_mode_costs[mode]
            original_count = len(self.g.transition_node_ids[mode])
            self.g.transition_node_ids[mode] = [
                id
                for id in self.g.transition_node_ids[mode]
                if id in vertices_to_keep or (id not in vertices_to_be_removed and sum(
                    self.env.batch_config_cost(self.g.nodes[id].state.q, focal_points))
                <= cost)
            ]
            num_pts_for_removal += original_count - len(
                self.g.transition_node_ids[mode]
            )
        # Update elements from g.reverse_transition_node_ids
        self.g.reverse_transition_node_ids[self.env.get_start_mode()] = [self.g.root.id]
        all_transitions = list(chain.from_iterable(self.g.transition_node_ids.values()))
        transition_nodes = np.array([self.g.nodes[id] for id in all_transitions])
        valid_mask = np.array([node.transition is not None for node in transition_nodes])
        valid_nodes = transition_nodes[valid_mask]
        reverse_transition_ids = [node.transition.id for node in valid_nodes]
        reverse_transition_modes =[node.transition.state.mode for node in valid_nodes]
        for mode, t_id in zip(reverse_transition_modes, reverse_transition_ids):
            self.g.reverse_transition_node_ids[mode].append(t_id)
        if self.remove_nodes:
            self.g.vertices = set()
            self.g.vertices.add(self.g.root.id)
        print(f"Removed {num_pts_for_removal} nodes")

    def process_valid_path(self, valid_path):
        path = [node.state for node in valid_path]
        new_path_cost = path_cost(path, self.env.batch_config_cost)
        if self.current_best_cost is None or new_path_cost < self.current_best_cost:
            self.alpha = 1.0
            self.current_best_path = path
            self.current_best_cost = new_path_cost
            self.current_best_path_nodes = valid_path
            self.remove_nodes = True

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
                    self.current_best_path, self.current_best_path_nodes = self.g.add_path_states(interpolated_path, self.current_best_cost)   
                    self.current_best_cost = path_cost(self.current_best_path, self.env.batch_config_cost) 
            if not self.optimize and self.current_best_cost is not None:
                return  
            # extract modes
            self.start_transition_arrays, self.end_transition_arrays, self.intermediate_mode_costs = {}, {}, {}
            self.start_transition_arrays[self.current_best_path_nodes[0].state.mode] = self.g.root.state.q.state()
            modes = [self.current_best_path_nodes[0].state.mode]
            start_cost = 0
            for n in self.current_best_path_nodes:
                if n.state.mode != modes[-1]:
                    self.end_transition_arrays[modes[-1]] = n.state.q.state()
                    self.start_transition_arrays[n.state.mode] = n.state.q.state()
                    self.intermediate_mode_costs[modes[-1]] = n.forward_cost - start_cost
                    start_cost = n.forward_cost
                    modes.append(n.state.mode)
            self.end_transition_arrays[modes[-1]] = self.current_best_path_nodes[-1].state.q.state()
            self.intermediate_mode_costs[modes[-1]] = self.current_best_path_nodes[-1].forward_cost - start_cost
            print("Modes of new path")
            print([m.task_ids for m in modes])

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
        # assert all([(self.g.nodes[id].state.mode in node.state.mode.next_modes or self.g.nodes[id].state.mode == node.state.mode) for id in neighbors]),  (
        #     "Not good"
        # )
        edge_costs = self.g.tot_neighbors_batch_cost_cache[node.id]
        for id, edge_cost in zip(neighbors, edge_costs):
            n = self.g.nodes[id]
            assert (n.forward_parent == node) == (n.id in node.forward_children), (
                    f"Parent and children don't coincide (reverse): parent: {node.id}, child: {n.id}"
                    )
            if n.id in node.blacklist or n.forward_parent == node:
                continue
            if node.state.mode in n.state.mode.next_modes:
                continue
            if n.is_transition:
                if n.transition is not None and n.id in self.g.reverse_transition_node_ids[node.state.mode]: 
                    if node.id != n.transition.id:
                        continue
            edge = (node, n)
            if n.id != node.id and edge:
                self.g.forward_queue.heappush((edge_cost, edge))
    
    def update_edge_collision_cache(
        self, n0: Node, n1: Node, is_edge_collision_free: bool
    ):
        if is_edge_collision_free:
            n1.whitelist.add(n0.id)
            n0.whitelist.add(n1.id)
        else:
            n0.blacklist.add(n1.id)
            n1.blacklist.add(n0.id)

    def update_results_tracking(self, cost, path):
        self.costs.append(cost)
        self.times.append(time.time() - self.start_time)
        self.all_paths.append(path)

    def get_lb_goal_node_and_cost(self) -> Node:
        min_id = np.argmin(self.operation.forward_costs[self.g.goal_node_ids], axis=0)
        best_cost = self.operation.forward_costs[self.g.goal_node_ids][min_id]
        best_node = self.g.goal_nodes[min_id]
        return best_node, best_cost

    @abstractmethod
    def initialize_samples_and_forward_queue(self):
        pass

    @abstractmethod
    def PlannerInitialization(self) -> None:
        """
        Initializes planner by setting parameters, creating the initial mode, and adding start node.

        Args:
            None

        Returns:
            None: None: This method does not return any value.
        """
        pass

    @abstractmethod
    def Plan(self) -> Tuple[List[State], Dict[str, List[Union[float, float, List[State]]]]]:
        """
        Executes planning process using an RRT* framework.

        Args:
            None

        Returns:
            Tuple:
                - List[State]: The planned path as a list of states.
                - Dict[str, List]: A dictionary containing:
                    - "costs" (List[float]): Recorded path costs.
                    - "times" (List[float]): Recorded execution times.
                    - "paths" (List[List[State]]): All explored paths.


        """
        pass
