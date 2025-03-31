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
    Type
)

from numpy.typing import NDArray
from itertools import chain
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
)
from multi_robot_multi_goal_planning.problems.util import path_cost, interpolate_path

from multi_robot_multi_goal_planning.planners import shortcutting

from multi_robot_multi_goal_planning.planners.termination_conditions import (
    PlannerTerminationCondition,
)
from multi_robot_multi_goal_planning.planners.rrtstar_base import find_nearest_indices
from multi_robot_multi_goal_planning.planners.rrtstar_base import save_data

T = TypeVar("T")
class BaseOperation(ABC):
    """Represents an operation instance responsible for managing variables related to path planning and cost optimization. """
    def __init__(self):

        self.costs = np.empty(10000000, dtype=np.float64)
        self.lb_costs_to_come = np.empty(10000000, dtype=np.float64)

    def get_cost(self, idx:int) -> float:
        """
        Returns cost of node with the specified index.

        Args: 
            idx (int): Index of node whose cost is to be retrieved.

        Returns: 
            float: Cost associated with the specified node."""
        return self.costs[idx]
    
    def get_lb_cost_to_come(self, idx:int) -> float:
        """
        Returns cost of node with the specified index.

        Args: 
            idx (int): Index of node whose cost is to be retrieved.

        Returns: 
            float: Cost associated with the specified node."""
        return self.lb_costs_to_come[idx]
    
    @abstractmethod
    def update(self):
        pass

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

class BaseNode(ABC):
    __slots__ = [
        "state",
        "forward",
        "rev",
        "is_transition",
        "is_reverse_transition",
        "transition",
        "whitelist",
        "blacklist",
        "id",
        "operation",
        "lb_cost_to_go"
    ]
    
    id_counter: ClassVar[int] = 0
    
    # Instance attributes
    forward: "Relatives"
    rev: "Relatives"
    state: State
    is_transition: bool
    is_reverse_transition:bool
    transition: Optional["BaseNode"]
    whitelist: Set[int]
    blacklist: Set[int]
    id: int
    operation:BaseOperation
    lb_cost_to_go: float

    def __init__(self, operation: BaseOperation, state: State, is_transition: bool = False) -> None:
        self.state = state
        self.forward = Relatives()
        self.rev = Relatives()
        self.is_transition = is_transition
        self.is_reverse_transition = False
        self.transition = None
        self.whitelist = set()
        self.blacklist = set()
        self.id = BaseNode.id_counter
        BaseNode.id_counter += 1
        self.operation = operation
        self.lb_cost_to_go = np.inf
    
    @abstractmethod
    def close(self):
        pass
    @abstractmethod
    def set_to_goal_node(self):
        pass

    def __lt__(self, other: "BaseNode") -> bool:
        return self.id < other.id

    def __hash__(self) -> int:
        return self.id

    @property
    def lb_cost_to_come(self):
        return self.operation.get_lb_cost_to_come(self.id)
    
    @lb_cost_to_come.setter
    def lb_cost_to_come(self, value) -> None:
        """Set the cost in the shared operation costs array.

        Args:
            value (float): Cost value to assign to the current node.

        Returns: 
            None: This method does not return any value."""
        self.operation.lb_costs_to_come[self.id] = value

    @property
    def cost(self):
        return self.operation.get_cost(self.id)
    
    @cost.setter
    def cost(self, value) -> None:
        """Set the cost in the shared operation costs array.

        Args:
            value (float): Cost value to assign to the current node.

        Returns: 
            None: This method does not return any value."""
        self.operation.costs[self.id] = value
        # if not np.isinf(value):
        #     self.lb_cost_to_come = value

class Relatives:
    __slots__ = [
        "parent",
        "children",
        "cost_to_parent",
        "fam"        
    ]
    parent: Optional["BaseNode"]
    children: List[int]
    cost_to_parent: Optional[float]
    fam: Set

    def __init__(self):
        self.parent = None
        self.children = []
        self.cost_to_parent = np.inf
        self.fam = set()

    def reset(self):
        self.parent = None
        self.children = []
        self.cost_to_parent = np.inf
        self.fam.clear()  
class DictIndexHeap(ABC, Generic[T]):
    __slots__ = ["queue", "items", "current_entries", "nodes"]

    queue: List[Tuple[float, int]]  # (priority, index)
    items: Dict[int, Any]  # Dictionary for storing active items
    current_entries: Dict[T, Tuple[float, int]]
    nodes: set
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
    def key(self, node: BaseNode) -> float:
        pass

    def heappush(self, item: T) -> None:
        """Push a single item into the heap."""
        # idx = len(self.items)
        priority = self.push_and_sync(item)
        # self.nodes_in_queue[item[1]] = (priority, DictIndexHeap.idx)
        heapq.heappush(self.queue, (priority, DictIndexHeap.idx))
        DictIndexHeap.idx += 1

    def pop_and_sync(self, idx:int):
        return self.items.pop(idx)

    def push_and_sync(self, item):
        self.items[DictIndexHeap.idx] = item  # Store only valid items
        priority = self.key(item)
        self.current_entries[item] = (priority, DictIndexHeap.idx) # always up to date with the newest one!
        return priority

    def peek_first_element(self) -> Any:
        while self.current_entries:
            priority, idx = self.queue[0]
            item = self.items[idx]
            if item in self.current_entries:
                current_priority, current_idx = self.current_entries[item]
                new_priority = self.key(item)
                if current_priority == priority and current_idx == idx:
                    if type(item) is BaseNode:
                        assert(new_priority == priority), (
                            "queue wrong"
                        )
                    if new_priority != priority: #needed if reverse search changed priorities
                        _, idx = heapq.heappop(self.queue)
                        item = self.pop_and_sync(idx)
                        self.heappush(item) 
                        continue
                    return priority, item
            _, idx = heapq.heappop(self.queue)
            _ = self.pop_and_sync(idx)
            continue

    def remove(self, item, in_current_entries:bool = False):
        if not in_current_entries and item not in self.current_entries:
           return
        del self.current_entries[item]
    
    def heappop(self) -> BaseNode:
        """Pop the item with the smallest priority from the heap."""
        if not self.queue:
            raise IndexError("pop from an empty heap")

         # Remove from dictionary (Lazy approach)
        while self.current_entries:
            priority, idx = heapq.heappop(self.queue)
            item = self.pop_and_sync(idx)
            if item in self.current_entries:
                current_priority, current_idx = self.current_entries[item]
                new_priority = self.key(item)
                if current_priority == priority and current_idx == idx:
                    assert(new_priority == priority), (
                        "queue wrong"
                    )
                    # if new_priority != priority: #needed if reverse search changed priorities
                    #     if type(item) is BaseNode:
                    #         pass
                    #     self.heappush(item) 
                    #     continue
                    self.remove(item, True)
                    return item
            
                
        raise IndexError("pop from an empty queue")
class BaseTree():
    all_vertices = set()
    def __init__(self, 
                 operation:BaseOperation, 
                 robot_dims, 
                 batch_dist_fun, 
                 batch_cost_fun, 
                 update_connectivity, 
                 is_edge_collision_free
                 ):

        self.operation = operation
        self.cnt = 0
        self.vertices = {}
        self.vertices_node_ids = np.empty(100000, dtype=np.int64)
        self.vertices_batch_array = np.empty((100000, robot_dims), dtype=np.float64)
        self.vertices_position = {}
        self.batch_dist_fun = batch_dist_fun
        self.batch_cost_fun = batch_cost_fun
        self.update_connectivity = update_connectivity
        self.is_edge_collision_free = is_edge_collision_free

    def ensure_capacity(self,array: NDArray, required_capacity: int) -> NDArray:
        return self.operation.ensure_capacity(array, required_capacity)
       
    def add_vertex(self, node: BaseNode) -> None:  
        if node.id in BaseTree.all_vertices:
            return
        BaseTree.all_vertices.add(node.id)         
        position = self.cnt
        self.vertices_position[node.id] = position
        self.vertices_batch_array = self.ensure_capacity(self.vertices_batch_array, position)
        self.vertices_batch_array[position,:] = node.state.q.state()
        self.vertices_node_ids = self.ensure_capacity(self.vertices_node_ids, position)
        self.vertices_node_ids[position] = node.id
        self.vertices[node.id] = node
        self.cnt +=1
    
    def get_vertices_batch_array(self)-> NDArray:
        return self.vertices_batch_array[:self.cnt]

    def get_vertices_node_ids(self)-> NDArray:
        return self.vertices_node_ids[:self.cnt]
    
    def get_position_of_node(self, node:BaseNode) -> int:
        return self.vertices_position[node.id]

    
    def neighbors(self, 
             node: BaseNode, 
             potential_parent:BaseNode, 
             r:float,
             ) -> Tuple[NDArray, NDArray, NDArray]:      
        """
        Retrieves neighbors of a node within a calculated radius for the given mode.

        Args:
            node (Node): New node for which neighbors are being identified.
            n_nearest_idx (int): Index of the nearest node to node.
            set_dists (Optional[NDArray]): Precomputed distances from n_new to all nodes in the specified subtree.
            tree (str): Identifier of subtree in which the nearest node is searched for

        Returns:
            Tuple:   
                - NDArray: Batch of neighbors near n_new.
                - NDArray: Corresponding cost values of these nodes.
                - NDArray: Corresponding IDs of these nodes.
        """
        #node is not yet in the tree
        batch_array= self.get_vertices_batch_array()
        set_dists = self.batch_dist_fun(node.state.q, batch_array)
        indices = find_nearest_indices(set_dists, r) # indices of batch_subtree
        if potential_parent is not None:
            potential_parent_idx = self.get_position_of_node(potential_parent)
            if potential_parent_idx not in indices:
                indices = np.insert(indices, 0, potential_parent_idx)
        node_indices = self.get_vertices_node_ids()[indices]
        n_near_costs = self.operation.costs[node_indices]
        N_near_batch = batch_array[indices]
        batch_cost = self.batch_cost_fun(node.state.q, N_near_batch)
        return batch_cost, n_near_costs, node_indices
    
    def find_parent(self, 
                    node: BaseNode, 
                    potential_parent:BaseNode,  
                    batch_cost: NDArray, 
                    n_near_costs: NDArray,
                    node_indices: NDArray,
                    ) -> None:
        """
        Sets the optimal parent for a new node by evaluating connection costs among candidate nodes.

        Args:
            mode (Mode): Current operational mode.
            node_indices (NDArray): Array of IDs representing candidate neighboring nodes.
            n_new (Node): New node that needs a parent connection.
            n_nearest (Node): Nearest candidate node to n_new.
            batch_cost (NDArray): Costs associated from n_new to all candidate neighboring nodes.
            n_near_costs (NDArray): Cost values for all candidate neighboring nodes.

        Returns:
            None: This method does not return any value.
        """

        idx =  np.where(node_indices == potential_parent.id)[0][0]
        c_new_tensor = n_near_costs + batch_cost
        c_min = c_new_tensor[idx]
        c_min_to_parent = batch_cost[idx]
        n_min = potential_parent
        if node.is_transition and node.is_reverse_transition:
            return n_min, c_min_to_parent 
        valid_mask = c_new_tensor < c_min
        if np.any(valid_mask):
            sorted_indices = np.where(valid_mask)[0][np.argsort(c_new_tensor[valid_mask])]
            for idx in sorted_indices:
                n = self.vertices[node_indices[idx].item()]
                if self.is_edge_collision_free(n.state.q, node.state.q, node.state.mode):
                    c_min = c_new_tensor[idx]
                    c_min_to_parent = batch_cost[idx]      
                    n_min = n                            
                    break
        return n_min, c_min_to_parent

    def rewire(self, 
                node: BaseNode, 
                batch_cost: NDArray, 
                n_near_costs: NDArray,
                node_indices: NDArray,
               ) -> bool:
        """
        Rewires neighboring nodes by updating their parent connection to n_new if a lower-cost path is established.

        Args:
            mode (Mode): Current operational mode.
            node_indices (NDArray): Array of IDs representing candidate neighboring nodes.
            n_new (Node): New node as potential parent for neighboring nodes.
            batch_cost (NDArray): Costs associated from n_new to all candidate neighboring nodes.
            n_near_costs (NDArray): Cost values for all candidate neighboring nodes.

        Returns:
            bool: True if any neighbor's parent connection is updated to n_new; otherwise, False.
        """

        c_potential_tensor = node.cost + batch_cost

        improvement_mask = c_potential_tensor < n_near_costs
        
        if np.any(improvement_mask):
            improved_indices = np.nonzero(improvement_mask)[0]

            for idx in improved_indices:
                n_near = self.vertices[node_indices[idx].item()]
                if n_near.is_transition and n_near.is_reverse_transition:
                    continue
                if n_near == node.forward.parent or n_near.cost == np.inf or n_near.id == node.id:
                    continue
                if node.state.mode == n_near.state.mode or node.state.mode == n_near.state.mode.prev_mode:
                    if self.is_edge_collision_free(node.state.q, n_near.state.q, node.state.mode):
                        edge_cost = float(batch_cost[idx])
                        self.update_connectivity(node, n_near, edge_cost, node.cost + edge_cost)


    
    

class BaseGraph(ABC):
    root: BaseNode
    nodes: Dict
    node_ids: Dict
    tree: BaseTree

    def __init__(self, root, operation:BaseOperation, batch_dist_fun, batch_cost_fun, is_edge_collision_free, node_cls: Type["BaseNode"] = BaseNode):
        self.operation = operation
        self.dim = len(root.state.q.state())
        self.root = root
        self.operation.update(root, lb_cost_to_go=np.inf, cost = 0, lb_cost_to_come = 0.0)
        self.node_cls = node_cls

        self.batch_dist_fun = batch_dist_fun
        self.batch_cost_fun = batch_cost_fun
        self.is_edge_collision_free = is_edge_collision_free

        self.nodes = {}  # contains all the nodes ever created
        self.nodes[self.root.id] = self.root
        
        self.node_ids = {}
        self.node_ids[self.root.state.mode] = [self.root.id]
        self.tree = {}
        self.transition_node_ids = {}  # contains the transitions at the end of the mode
        self.reverse_transition_node_ids = {}
        self.reverse_transition_node_ids[self.root.state.mode] = [self.root.id]
        self.goal_nodes = []
        self.goal_node_ids = []
        self.initialize_cache()
        self.unit_n_ball_measure = ((np.pi**0.5) ** self.dim) / math.gamma(self.dim / 2 + 1) 
        self.new_path = None

    def get_num_samples(self) -> int:
        num_samples = 0
        for k, v in self.node_ids.items():
            num_samples += len(v)

        num_transition_samples = 0
        for k, v in self.transition_node_ids.items():
            num_transition_samples += len(v)

        return num_samples + num_transition_samples
    
    def add_node(self, new_node, cost:float = np.inf, lb_cost_to_go:float = np.inf) -> None:
        self.nodes[new_node.id] = new_node
        key = new_node.state.mode
        if key not in self.node_ids:
            self.node_ids[key] = []
        self.node_ids[key].append(new_node.id)
        self.operation.update(new_node, lb_cost_to_go, cost)
        
    def add_vertex_to_tree(self, node: BaseNode) -> None:  
        mode = node.state.mode
        if mode not in self.tree: 
            self.tree[mode] = BaseTree(self.operation, 
                                       self.dim, 
                                       batch_dist_fun = self.batch_dist_fun,
                                       batch_cost_fun = self.batch_cost_fun,
                                       update_connectivity = self.update_connectivity,
                                       is_edge_collision_free = self.is_edge_collision_free)
        self.tree[mode].add_vertex(node)

    def add_states(self, states: List[State]):
        for s in states:
            self.add_node(self.node_cls(self.operation, s))

    def add_nodes(self, nodes: List[BaseNode]):
        for n in nodes:
            self.add_node(n)

    def add_transition_nodes(self, transitions: Tuple[Configuration, Mode, Mode]):
        
        for q, this_mode, next_mode in transitions:
            node_this_mode = self.node_cls(self.operation, State( q, this_mode), True)
            node_next_mode = self.node_cls(self.operation, State( q, next_mode), True)

            if this_mode in self.transition_node_ids:
                if self.transition_or_goal_is_already_present(node_this_mode) is not None:
                    return False
            is_goal = True
            if next_mode is not None:
                is_goal = False
                node_next_mode.transition = node_this_mode
                node_this_mode.transition = node_next_mode
                assert this_mode.task_ids != next_mode.task_ids
                self.update_edge_collision_cache(node_this_mode,node_next_mode, True)

            self.add_transition_node(node_this_mode, is_goal=is_goal)
            self.add_transition_node(node_next_mode, reverse=True)
            return True

    def add_transition_node(self, node, is_goal:bool = False, reverse:bool = False, cost:float = np.inf, lb_cost_to_go:float = np.inf):
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
            node.is_reverse_transition = True 
            if mode in self.reverse_transition_node_ids:
                self.reverse_transition_node_ids[mode].append(node.id) 
            else:
                self.reverse_transition_node_ids[mode] = [node.id]

        self.nodes[node.id] = node
        self.operation.update(node, lb_cost_to_go, cost)
    
    def add_path_states(self, path:List[State], space_extent):
        self.new_path = []
        parent = self.root
        for i in range(len(path)):
            if i == 0:
                continue
            is_transition = False
            next_mode = None
            if (
                i < len(path) - 1
                and path[i].mode
                != path[i + 1].mode
            ):
                is_transition = True
                next_mode = path[i+1].mode
            if i == len(path)-1:
                is_transition = True
            node = self.node_cls(self.operation, path[i], is_transition)
            if is_transition:
                if parent.is_transition and parent.is_reverse_transition and self.reverse_transition_is_already_present(node):
                    continue
                n = self.transition_or_goal_is_already_present(node)
                if n is not None:
                    edge_cost = float(self.batch_cost_fun([parent.state], [n.state]))
                    self.update_connectivity(parent, n, edge_cost, parent.cost + edge_cost)
                    self.new_path.append(n)
                    if n.transition is not None:
                        self.new_path.append(n.transition)
                        self.update_connectivity(n, n.transition, 0.0, n.cost)
                        r = self.get_r_star(self.tree[n.transition.state.mode].cnt, space_extent, self.unit_n_ball_measure)
                        batch_cost, n_near_costs, node_indices = self.tree[n.transition.state.mode].neighbors(n.transition, None, r)
                        self.tree[n.transition.state.mode].rewire(n.transition, batch_cost, n_near_costs, node_indices)
                    parent = self.new_path[-1]
                    continue

            self.add_path_node(node, parent, is_transition, next_mode, space_extent)
            parent = self.new_path[-1]     
        return 

    def add_path_node(self, node, parent, is_transition:bool, next_mode:Mode, space_extent): 
        self.new_path.append(node)
        if is_transition:
            edge_cost = float(self.batch_cost_fun([parent.state], [node.state]))
            self.update_connectivity(parent, node, edge_cost, parent.cost + edge_cost)
            is_goal = True
            node_next_mode = self.node_cls(self.operation, State(node.state.q, next_mode), is_transition)   
            self.add_vertex_to_tree(node)         
            if next_mode is not None:
                self.new_path.append(node_next_mode)
                is_goal = False
                self.update_connectivity(node, node_next_mode, 0.0, node.cost)

                node.transition = node_next_mode
                node_next_mode.transition = node
                
                r = self.get_r_star(self.tree[node_next_mode.state.mode].cnt, space_extent, self.unit_n_ball_measure)
                batch_cost, n_near_costs, node_indices = self.tree[node_next_mode.state.mode].neighbors(node_next_mode, None, r) 
                self.tree[node_next_mode.state.mode].rewire(node_next_mode, batch_cost, n_near_costs, node_indices)
                self.add_vertex_to_tree(node_next_mode)
            else:
                node.set_to_goal_node()

            self.add_transition_node(node, is_goal=is_goal, cost=node.cost)
            self.add_transition_node(node_next_mode, reverse=True, cost=node.cost)
            
            
        else:
            r = self.get_r_star(self.tree[node.state.mode].cnt, space_extent, self.unit_n_ball_measure)
            batch_cost, n_near_costs, node_indices = self.tree[node.state.mode].neighbors(node, parent, r) 
            true_parent, true_edge_cost = self.tree[node.state.mode].find_parent(node, parent, batch_cost, n_near_costs, node_indices) 
            self.update_connectivity(true_parent, node, float(true_edge_cost), true_parent.cost + float(true_edge_cost))
            self.tree[node.state.mode].rewire(node, batch_cost, n_near_costs, node_indices)
            self.add_node(node, cost=node.cost)
            self.add_vertex_to_tree(node)

    def transition_or_goal_is_already_present(self, node:BaseNode):  
        if len(self.transition_node_ids[node.state.mode]) > 0:
            configs_list = [
                            self.nodes[id].state.q
                            for id in self.transition_node_ids[node.state.mode]
                            ]
        
            dists = self.batch_dist_fun(node.state.q, configs_list)
            min_index = np.argmin(dists)
            min_dist = dists[min_index]

            if min_dist < 1e-6:
                id = self.transition_node_ids[node.state.mode][min_index]
                return self.nodes[id]
        return None
    
    def reverse_transition_is_already_present(self, node:BaseNode):  
        if len(self.transition_node_ids[node.state.mode]) > 0:
            configs_list = [
                            self.nodes[id].state.q
                            for id in self.reverse_transition_node_ids[node.state.mode]
                            ]
        
            dists = self.batch_dist_fun(node.state.q, configs_list)

            if np.min(dists) < 1e-6:
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

        # node ids as keys
        self.neighbors_node_ids_cache = {}
        self.neighbors_batch_cost_cache = {}
        self.neighbors_array_cache = {}
        self.neighbors_fam_ids_cache = {}
        self.tot_neighbors_batch_cost_cache = {}
        self.tot_neighbors_id_cache = {}
        self.transition_node_lb_cache = {}
        self.reverse_transition_node_lb_cache = {}
        self.reverse_transition_node_array_cache = {}

        self.blacklist_cache = {}

    def get_r_star(
        self, number_of_nodes, informed_measure, unit_n_ball_measure, wheight=1):
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

        # r_star = (
        #     1.001
        #     * 2* wheight
        #     * (
        #         ((1 + 1 / self.dim))
        #         * (informed_measure / unit_n_ball_measure)
        #         * (np.log(number_of_nodes) / number_of_nodes)
        #     )
        #     ** (1 / self.dim)
        # )
        return r_star

    def get_neighbors(
        self, node: BaseNode, space_extent: float = None, in_closed_set:bool = False
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
        self.neighbors_batch_cost_cache[node.id] = self.batch_cost_fun(node.state.q, arr)
        self.blacklist_cache[node.id] = set()
        return self.update_neighbors(node)

    def update_neighbors_with_family_of_node(self, node:BaseNode, update:bool = False):
        neighbors_fam = set()
        if node.id in self.neighbors_fam_ids_cache:
            neighbors_fam = self.neighbors_fam_ids_cache[node.id]

        combined_fam = node.forward.fam | node.rev.fam
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
                self.tot_neighbors_batch_cost_cache[node.id] = np.concatenate((self.batch_cost_fun(node.state.q, arr), self.neighbors_batch_cost_cache[node.id]))
                arr = np.concatenate((arr, self.neighbors_array_cache[node.id]))
                self.tot_neighbors_id_cache[node.id] = np.concatenate((mask_node_ids, self.neighbors_node_ids_cache[node.id]))
                assert len(self.tot_neighbors_id_cache[node.id]) == len(self.tot_neighbors_batch_cost_cache[node.id]),(
                    "forward not right"
                )
                return self.tot_neighbors_id_cache[node.id]
        if update or node.id not in self.tot_neighbors_id_cache:
            arr = self.neighbors_array_cache[node.id]
            self.tot_neighbors_batch_cost_cache[node.id] = self.neighbors_batch_cost_cache[node.id]
            self.tot_neighbors_id_cache[node.id] = self.neighbors_node_ids_cache[node.id]
        assert len(self.tot_neighbors_id_cache[node.id]) == len(self.tot_neighbors_batch_cost_cache[node.id]),(
            "forward not right"
        )
        return self.tot_neighbors_id_cache[node.id]

    def update_neighbors(self, node:BaseNode): # only needed for forward
        blacklist = node.blacklist
        diff = blacklist -  self.blacklist_cache[node.id]
        if len(diff) > 0:
            self.blacklist_cache[node.id] = set(blacklist)
            node_ids = self.neighbors_node_ids_cache[node.id]
            arr =  self.neighbors_array_cache[node.id]
            # mask_node_ids =  ~np.isin(node_ids, list(diff))
            mask_node_ids =  np.fromiter((id_ not in diff for id_ in node_ids), dtype=bool)
            if not mask_node_ids.all():
                node_ids = node_ids[mask_node_ids]
                arr = arr[mask_node_ids]
                self.neighbors_node_ids_cache[node.id] = node_ids
                self.neighbors_array_cache[node.id] = arr
                self.neighbors_batch_cost_cache[node.id] = self.neighbors_batch_cost_cache[node.id][mask_node_ids]

        return self.update_neighbors_with_family_of_node(node, True)
            
    def update_forward_cost_of_children(self, n: BaseNode, start_nodes_to_update:set[int]) -> set[int]:
        stack = [n.id]
        while stack:
            current_id = stack.pop()
            current_node = self.nodes[current_id]
            children = current_node.forward.children
            if children:
                for _, id in enumerate(children):
                    child = self.nodes[id]
                    new_cost = current_node.cost + child.forward.cost_to_parent
                    child.cost = new_cost
                    start_nodes_to_update.add(child.id)
                stack.extend(children)
        return start_nodes_to_update
    
    def update_forward_cost(self, node:BaseNode):
        start_nodes_to_update = set()
        start_nodes_to_update.add(node.id)
        if len(node.forward.children) > 0:
            start_nodes_to_update = self.update_forward_cost_of_children(node, start_nodes_to_update)
        self.update_forward_queue_keys('start', start_nodes_to_update)
        
    @abstractmethod
    def update_forward_queue_keys(self, node_ids:Optional[Set[BaseNode]], type:str):
        pass

    def update_connectivity(self, n0: BaseNode, n1: BaseNode, edge_cost, updated_cost, tree: str = "forward", is_transition:bool = False):
        if tree == "forward":
            if n1.forward.parent is None: 
                n0.forward.children.append(n1.id)
            elif n1.forward.parent.id != n0.id:
                n1.forward.parent.forward.children.remove(n1.id)
                n1.forward.parent.forward.fam.remove(n1.id)
                n1.forward.fam.remove(n1.forward.parent.id)
                n0.forward.children.append(n1.id)
            n1.forward.parent = n0
            # assert (n1.lb_cost_to_come - updated_cost)<=1e-15, (
            #         "ohhhhhhhhhhhhhh something is wrong with the lb cost to go"
            #     )
            n1.forward.cost_to_parent = edge_cost
            assert n1.id in n0.forward.children, (
                "not a child")
            
            
            if updated_cost != n1.cost:
                n1.cost = updated_cost
                self.update_forward_cost(n1)
                
                
            # else:
            #     print("uhhh")
            self.add_vertex_to_tree(n1)
            self.add_vertex_to_tree(n0)
            n1.forward.fam.add(n0.id)
            n0.forward.fam.add(n1.id)
            if is_transition and n1.transition.forward.parent is None:
                self.update_connectivity(n1, n1.transition, 0.0, n1.cost ,"forward")

        else:
            n1.lb_cost_to_go = updated_cost
            if n1.rev.parent is not None:
                if n1.rev.parent.id == n0.id:
                    if is_transition:
                        self.update_connectivity(n1, n1.transition, 0.0, n1.lb_cost_to_go,"reverse", False)
                    return
                if n1.rev.parent.id != n0.id:
                    # assert [
                    #             (self.nodes[child].rev.parent, child)
                    #             for child in n1.rev.parent.rev.children
                    #             if self.nodes[child].rev.parent is None
                    #             or self.nodes[child].rev.parent.id != n1.rev.parent.id
                    #         ] == [], "parent and children not correct"

                    n1.rev.parent.rev.children.remove(n1.id)
                    n1.rev.parent.rev.fam.remove(n1.id)
                    n1.rev.fam.remove(n1.rev.parent.id)
                    n0.rev.children.append(n1.id)
            else:
                n0.rev.children.append(n1.id)

            # assert (n1.lb_cost_to_go != n0.lb_cost_to_go_expanded + edge_cost), (
            # "asdf")
            n1.rev.parent = n0
            assert n1.id in n0.rev.children, (
                "not a child")
            
            n1.rev.cost_to_parent = edge_cost
            # assert [
            #             (self.nodes[child].rev.parent, child)
            #             for child in n1.rev.parent.rev.children
            #             if self.nodes[child].rev.parent is None
            #             or self.nodes[child].rev.parent.id != n1.rev.parent.id
            #         ] == [], (
            #             "new parent and new children not correct")
            if n0.state.mode.next_modes == []:
                assert n1.lb_cost_to_go >= float(self.batch_cost_fun([n1.state], [n0.state])), (
                "something is wrong with the lb cost to go"
            )
            else:
                goal_nodes = np.array([n.state.q.state() for n in self.goal_nodes])
                assert min((self.batch_cost_fun(n1.state.q, goal_nodes))) - n1.lb_cost_to_go <=1e-15 , (
                    "something is wrong with the lb cost to go"
                )

            # if (n1.test - n1.lb_cost_to_go )>1e-15:
            #     return
            # if not np.isinf(n1.test):
            #     assert (n1.test - n1.lb_cost_to_go )<=1e-15, (
            #             "ohhhhhhhhhhhhhh something is wrong with the lb cost to go"
            #         )
            # if not n1.is_transition:
            #     assert not np.isinf(n1.lb_cost_to_go), (
            #         "wrongg update"
            #     )
            n1.rev.fam.add(n0.id)
            n0.rev.fam.add(n1.id)
            if is_transition:
                self.update_connectivity(n1, n1.transition, 0.0, n1.lb_cost_to_go, "reverse", False)

    def compute_transition_lb_cost_to_come(self):
        # run a reverse search on the transition nodes without any collision checking
        costs = {}
        transition_nodes = {}
        processed = 0

        closed_set = set()

        queue = []
        heapq.heappush(queue, (0, self.root))
        costs[self.root.id] = 0

        while len(queue) > 0:
            _, node = heapq.heappop(queue)
            mode = node.state.mode
            if node.id in closed_set:
                continue
            closed_set.add(node.id)
            if mode not in self.transition_node_ids:
                continue
            if mode not in transition_nodes:
                if mode.task_ids == self.goal_nodes[0].state.mode.task_ids:
                    transition_nodes[mode] = self.goal_nodes
                else:
                    transition_nodes[mode] = [self.nodes[id].transition for id in self.transition_node_ids[mode]]

            if len(transition_nodes[mode]) == 0:
                continue
            self.update_cache(mode)
            
            if mode not in self.transition_node_array_cache:
                continue

            # add neighbors to open_queue
            edge_costs = self.batch_cost_fun(
                node.state.q,
                self.transition_node_array_cache[mode],
            )

            parent_cost = costs[node.id]
            for edge_cost, n in zip(edge_costs, transition_nodes[mode]):
                cost = parent_cost + edge_cost
                id = n.id
                if id not in costs or cost < costs[id]:
                    costs[id] = cost
                    n.lb_cost_to_come = cost
                    if n.transition is not None:
                        n.transition.lb_cost_to_come = cost
                    processed += 1
                    heapq.heappush(queue, (cost, n))
        print(processed)
        
    def compute_node_lb_cost_to_come(self):
        processed = 0
        reverse_transition_node_lb_cache = {}
        for mode in self.node_ids:
            for id in self.node_ids[mode]:
                n = self.nodes[id]
                mode = n.state.mode
                if mode not in self.reverse_transition_node_array_cache:
                    continue

                if mode not in reverse_transition_node_lb_cache:
                    reverse_transition_node_lb_cache[mode] = np.array(
                        [
                            self.nodes[id].lb_cost_to_come
                            for id in self.reverse_transition_node_ids[mode]
                        ],
                        dtype=np.float64,
                    )

                costs_to_transitions = self.batch_cost_fun(
                    n.state.q,
                    self.reverse_transition_node_array_cache[mode],
                )

                min_cost = np.min(
                    reverse_transition_node_lb_cache[mode] + costs_to_transitions
                )
                n.lb_cost_to_come = min_cost
                processed +=1
        print(processed)

    def update_edge_collision_cache(
            self, n0: BaseNode, n1: BaseNode, is_edge_collision_free: bool
        ):
            if is_edge_collision_free:
                n1.whitelist.add(n0.id)
                n0.whitelist.add(n1.id)
            else:
                n0.blacklist.add(n1.id)
                n1.blacklist.add(n0.id)

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
                    g.nodes[id].lb_cost_to_come
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
        informed_with_lb:bool = True,
        remove_based_on_modes:bool = False
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
        self.remove_based_on_modes = remove_based_on_modes

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
        self.operation = self._create_operation()
        self.consistent_nodes = set()
    
    def _create_operation(self) -> BaseOperation:
        return BaseOperation()

    def sample_valid_uniform_batch(self, batch_size: int, cost: float) -> List[State]:
        new_samples = []
        num_attempts = 0
        num_valid = 0
        failed_attemps = 0

        if len(self.g.goal_nodes) > 0:
            focal_points = np.array(
                [self.g.root.state.q.state(), self.g.goal_nodes[0].state.q.state()],
                dtype=np.float64,
            )

        while len(new_samples) < batch_size:
            # if failed_attemps > 10000000:
            #     break
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
                    failed_attemps += 1
                    continue

            if self.env.is_collision_free(q, m):
                new_samples.append(State(q, m))
                num_valid += 1
            else:
                 failed_attemps += 1

        print("Percentage of succ. attempts", num_valid / num_attempts)

        return new_samples, num_attempts

    def sample_valid_uniform_transitions(self, transistion_batch_size, cost):
        transitions = []
        failed_attemps = 0

        if len(self.g.goal_nodes) > 0:
            focal_points = np.array(
                [self.g.root.state.q.state(), self.g.goal_nodes[0].state.q.state()],
                dtype=np.float64,
            )
        added_transitions = 0
        # while len(transitions) < transistion_batch_size:
        while added_transitions < transistion_batch_size:
            # if failed_attemps > 100000000:
            #     break
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
                    failed_attemps +=1
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
                else:
                    failed_attemps+=1
                transitions = []
            else:
                failed_attemps +=1
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

                        qr = np.random.rand(len(qr_mean)) * 0.2 + qr_mean

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

                    qr = np.random.rand(len(qr_mean)) * 0.2 + qr_mean

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
            return BaseTree.all_vertices, []

        vertices_node_ids = []
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
            if node.is_transition and node.is_reverse_transition:
                if node.transition.id in vertices_node_ids:
                    vertices_node_ids.append(id)
                    stack.extend(node.forward.children)
                    count +=1
                continue
            
            if not self.remove_based_on_modes or key not in self.start_transition_arrays:
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
                    #not in the current expanded tree
                    if node.id not in BaseTree.all_vertices:
                        flag = True
            else:
                flag = sum(self.env.batch_config_cost(node.state.q, focal_points_transition[key])) > inter_costs[key]
                
            if flag:
                children = [node.id]
                if node.forward.parent is not None:
                    node.forward.parent.forward.children.remove(node.id)
                    node.forward.parent.forward.fam.remove(node.id)
                    node.forward.fam.remove(node.forward.parent.id)
                    node.forward.parent = None
                to_be_removed = []
                while len(children) > 0:
                    child_id = children.pop()
                    to_be_removed.append(child_id)
                    children.extend(self.g.nodes[child_id].forward.children)
                children_to_be_removed.extend(to_be_removed)
            else:
                vertices_node_ids.append(id)
                stack.extend(node.forward.children)

        goal_mask = np.array([item in children_to_be_removed for item in self.g.goal_node_ids])
        goal_nodes = np.array(self.g.goal_node_ids)[goal_mask]
        if goal_nodes.size > 0:
            for goal in goal_nodes:
                goal_node = self.g.nodes[goal]
                goal_node.cost = np.inf
                goal_node.forward.cost_to_parent = np.inf
                goal_node.forward.children = []
                goal_node.forward.parent = None
                goal_node.forward.fam = set()
                children_to_be_removed.remove(goal)
                key = goal_node.state.mode
                vertices_node_ids.append(goal)
        
        return vertices_node_ids, children_to_be_removed

    def remove_nodes_in_graph(self):
        num_pts_for_removal = 0
        vertices_to_keep, vertices_to_be_removed = self.prune_set_of_expanded_nodes()
        # vertices_to_keep = list(chain.from_iterable(self.g.vertices_node_ids.values()))
        # Remove elements from g.nodes_ids
        for mode in list(self.g.node_ids.keys()):# Avoid modifying dict while iterating
            # start = random.choice(self.g.reverse_transition_node_ids[mode])
            # goal = random.choice(self.g.re)
            if not self.remove_based_on_modes or mode not in self.start_transition_arrays:
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
                if id == self.g.root.id or id in vertices_to_keep or (id not in vertices_to_be_removed and sum(
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
            if self.env.is_terminal_mode(mode):
                continue
            
            if not self.remove_based_on_modes or mode not in self.start_transition_arrays:
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
            if mode not in self.g.reverse_transition_node_ids:
                self.g.reverse_transition_node_ids[mode] = []
            self.g.reverse_transition_node_ids[mode].append(t_id)
        if self.remove_nodes:
            self.g.tree = {}
            BaseTree.all_vertices.clear()
            self.g.add_vertex_to_tree(self.g.root)
        print(f"Removed {num_pts_for_removal} nodes")

    def process_valid_path(self, 
                           valid_path, 
                           with_queue_update:bool = True, 
                           with_shortcutting:bool = False, 
                           force_update:bool = False):
        path = [node.state for node in valid_path]
        new_path_cost = path_cost(path, self.env.batch_config_cost)
        update_forward_queue = False
        if self.current_best_cost is None:
            update_forward_queue = True
        if force_update or self.current_best_cost is None or new_path_cost < self.current_best_cost:
            self.alpha = 1.0
            print()
            print([n.id for n in valid_path])
            self.current_best_path = path
            self.current_best_cost = new_path_cost
            self.current_best_path_nodes = valid_path
            self.remove_nodes = True
            self.dynamic_reverse_search_update = True
            if not with_shortcutting and with_queue_update:
                # self.g.add_path_states(path, self.approximate_space_extent)   
                # self.current_best_path_nodes = self.generate_path(force_generation=True)
                # self.current_best_path = [node.state for node in self.current_best_path_nodes]
                # self.g.initialize_cache()
                self.add_reverse_connectivity_to_path(self.current_best_path_nodes, with_queue_update)
            if update_forward_queue:
                self.g.update_forward_queue_keys('target')

            print(f"New cost: {new_path_cost} at time {time.time() - self.start_time}")
            self.update_results_tracking(new_path_cost, path)

            if self.try_shortcutting and with_shortcutting:
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
                    self.g.add_path_states(interpolated_path, self.approximate_space_extent)   
                    self.current_best_path_nodes = self.generate_path(force_generation=True)
                    self.current_best_path = [node.state for node in self.current_best_path_nodes]
                    self.g.initialize_cache()
                    # self.g.compute_transition_lb_cost_to_come()
                    # self.g.compute_node_lb_cost_to_come() 
                    self.add_reverse_connectivity_to_path(self.current_best_path_nodes, with_queue_update)
                    self.current_best_cost = path_cost(self.current_best_path, self.env.batch_config_cost) 
                    print("new cost: " ,self.current_best_cost)
            if not self.optimize and self.current_best_cost is not None:
                return  
            print([n.id for n in self.current_best_path_nodes])
            # extract modes for removal strategy
    
    def update_removal_conditions(self):
        self.start_transition_arrays, self.end_transition_arrays, self.intermediate_mode_costs = {}, {}, {}
        self.start_transition_arrays[self.current_best_path_nodes[0].state.mode] = self.g.root.state.q.state()
        modes = [self.current_best_path_nodes[0].state.mode]
        start_cost = 0
        for n in self.current_best_path_nodes:
            if n.state.mode != modes[-1]:
                self.end_transition_arrays[modes[-1]] = n.state.q.state()
                self.start_transition_arrays[n.state.mode] = n.state.q.state()
                self.intermediate_mode_costs[modes[-1]] = n.cost - start_cost
                start_cost = n.cost
                modes.append(n.state.mode)
        self.end_transition_arrays[modes[-1]] = self.current_best_path_nodes[-1].state.q.state()
        self.intermediate_mode_costs[modes[-1]] = self.current_best_path_nodes[-1].cost - start_cost
        print("Modes of new path")
        print([m.task_ids for m in modes])     
            
    def add_reverse_connectivity_to_path(self, path, with_queue_update):
        parent = path[-1]
        to_update = set()
        to_update.add(parent.id)
        for node in reversed(path[1:-1]):
            edge_cost = parent.forward.cost_to_parent
            # lb_cost_to_go_expanded = lb_cost_to_go for AIT*
            self.g.update_connectivity(parent, node, edge_cost , parent.lb_cost_to_go + edge_cost,'reverse')
            node.close(self.env.collision_resolution)
            if with_queue_update:
                if node.is_transition and not node.is_reverse_transition:
                    self.expand_node_forward(node.transition)
                else:
                    self.expand_node_forward(node)
            self.consistent_nodes.add(node.id)
            to_update.add(node.id)
            parent = node
        to_update.add(path[0].id)
        if with_queue_update:
            self.g.update_forward_queue_keys('target', to_update)

    def generate_path(self, force_generation:bool = False) -> List[BaseNode]:
        path = []
        goal, cost = self.get_lb_goal_node_and_cost()
        if not force_generation and (math.isinf(cost) or goal is None or (self.current_best_cost is not None and cost >= self.current_best_cost)):
            return path
        path.append(goal)

        n = goal

        while n.forward.parent is not None:
            path.append(n.forward.parent)
            n = n.forward.parent

        # path.append(n)
        path = path[::-1]
        return path

    def update_results_tracking(self, cost, path):
        self.costs.append(cost)
        self.times.append(time.time() - self.start_time)
        self.all_paths.append(path)

    def get_lb_goal_node_and_cost(self) -> BaseNode:
        min_id = np.argmin(self.operation.costs[self.g.goal_node_ids], axis=0)
        best_cost = self.operation.costs[self.g.goal_node_ids][min_id]
        best_node = self.g.goal_nodes[min_id]
        return best_node, best_cost
    
    def expand_node_forward(self, node: Optional[BaseNode] = None) -> None:
        self.forward_closed_set.add(node.id)
        if node in self.g.goal_nodes:
            return   
        neighbors = self.g.get_neighbors(node, space_extent=self.approximate_space_extent)
        if neighbors.size == 0:
            return
        
        edge_costs = self.g.tot_neighbors_batch_cost_cache[node.id]
        for id, edge_cost in zip(neighbors, edge_costs):
            n = self.g.nodes[id]
            if n.id in self.forward_closed_set:
                continue
            assert (n.forward.parent == node) == (n.id in node.forward.children), (
                    f"Parent and children don't coincide (reverse): parent: {node.id}, child: {n.id}"
                    )
            if n.id == self.g.root.id:
                continue
            assert(n.id not in node.blacklist), (
            "neighbors are wrong")
            if node.forward.parent is not None and node.forward.parent.id == n.id:
                continue
            if n.is_transition and n.is_reverse_transition:
                continue
            edge = (node, n)
            self.update_forward_queue(edge_cost, edge)

    @abstractmethod
    def update_forward_queue(self, edge_cost, edge):
        pass

    @abstractmethod
    def initialize_search(self):
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
