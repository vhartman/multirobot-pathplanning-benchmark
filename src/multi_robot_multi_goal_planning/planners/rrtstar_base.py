import numpy as np
import time as time
import math as math
import random
from multi_robot_multi_goal_planning.problems.util import interpolate_path
from multi_robot_multi_goal_planning.planners.shortcutting import robot_mode_shortcut, remove_interpolated_nodes
import os
import pickle
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, List, Dict, Callable, ClassVar
from numpy.typing import NDArray
from numba import njit
from dataclasses import dataclass, field

from scipy.stats.qmc import Halton

from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem,
    Mode,
)
from multi_robot_multi_goal_planning.problems.core.configuration import (
    Configuration,
    batch_config_dist,
)
from .sampling_informed import InformedSampling
from .mode_validation import ModeValidation
from .baseplanner import BasePlanner


class Operation:
    """Represents an operation instance responsible for managing variables related to path planning and cost optimization."""

    def __init__(self):
        self.path = []
        self.path_modes = []
        self.path_nodes = None
        self.cost = np.inf
        self.cost_change = np.inf
        self.init_sol = False
        self.costs = np.empty(10000000, dtype=np.float64)
        self.paths_inter = []
        self.path_shortcutting = []
        self.path_shortcutting_interpolated = []

    def get_cost(self, idx: int) -> float:
        """
        Returns cost of node with the specified index.

        Args:
            idx (int): Index of node whose cost is to be retrieved.

        Returns:
            float: Cost associated with the specified node."""
        return self.costs[idx]


class Node:
    """Represents a node in the planning structure"""

    id_counter = 0

    def __init__(self, state: State, operation: Operation):
        self.state = state
        self.parent = None
        self.children = []
        self.transition = False
        self.cost_to_parent = None
        self.operation = operation
        self.id = Node.id_counter
        Node.id_counter += 1
        self.neighbors = {}
        self.hash = None

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

    def __repr__(self):
        return f"<N- {self.state.q.state()}, c: {self.cost}>"

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(
                (self.state.q.state().data.tobytes(), tuple(self.state.mode.task_ids))
            )
        return self.hash


class BaseTree(ABC):
    """
    Represents base structure for different tree implementations.
    """

    def __init__(self):
        pass

    def _resize_array(
        self, array: NDArray, current_capacity: int, new_capacity: int
    ) -> NDArray:
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
            NDArray: The array with ensured capacity."""
        current_size = array.shape[0]

        if required_capacity == current_size:
            return self._resize_array(
                array, current_size, required_capacity * 2
            )  # Double the size

        return array

    @abstractmethod
    def add_node(self, n: Node, tree: str = "") -> None:
        """
        Adds a node to the specified subtree.

        Args:
            n (Node): Node to be added.
            tree (str, optional): Identifier of subtree where the node will be added.

        Returns:
            None: This method does not return any value.
        """

        pass

    @abstractmethod
    def remove_node(self, n: Node, tree: str = "") -> None:
        """
        Removes a node from the specified subtree.

        Args:
            n (Node): The node to be removed.
            tree (str, optional): Identifier of subtree from which the node will be removed.

        Returns:
            None: This method does not return any value.
        """

        pass

    @abstractmethod
    def get_batch_subtree(self, tree: str = "") -> NDArray:
        """
        Retrieves batch representation of specified subtree.

        Args:
            tree (str, optional): Identifier of subtree to be retrieved.

        Returns:
            NDArray: A NumPy array representing the batch data of the subtree.
        """

        pass

    @abstractmethod
    def get_node_ids_subtree(self, tree: str = "") -> NDArray:
        """
        Retrieves node IDs of specified subtree.

        Args:
            tree (str, optional): Identifier of subtree.

        Returns:
            NDArray: A NumPy array containing node IDs of the subtree.
        """

        pass

    @abstractmethod
    def add_transition_node_as_start_node(self, n: Node, tree: str = "") -> None:
        """
        Adds transition node as a start node in the specified subtree.

        Args:
            n (Node): The transition node to be added as the start node.
            tree (str, optional):  Identifier of subtree.

        Returns:
            None: This method does not return any value.
        """

        pass

    @abstractmethod
    def get_node(self, id: int, tree: str = "") -> Node:
        """
        Retrieves node from the specified subtree by its id.

        Args:
            id (int): The unique ID of the node to be retrieved.
            tree (str, optional):  Identifier of subtree.

        Returns:
            Node: The node with the desired id.
        """

        pass

    @abstractmethod
    def get_number_of_nodes_in_tree(self) -> int:
        """
        Returns total number of nodes in the tree.

        Args:
            None

        Returns:
            int: Number of nodes present in the tree.
        """

        pass


class SingleTree(BaseTree):
    """
    Represents single tree structure.
    """

    def __init__(self, env: BaseProblem):
        self.order = 1
        # self.informed = Informed()
        robot_dims = sum(env.robot_dims.values())
        self.subtree = {}
        self.initial_capacity = 100000
        self.batch_subtree = np.empty(
            (self.initial_capacity, robot_dims), dtype=np.float64
        )
        self.node_ids_subtree = np.empty(self.initial_capacity, dtype=np.int64)

    def add_node(self, n: Node, tree: str = "") -> None:
        self.subtree[n.id] = n
        position = len(self.subtree) - 1
        self.batch_subtree = self.ensure_capacity(self.batch_subtree, position)
        self.batch_subtree[position, :] = n.state.q.state()
        self.node_ids_subtree = self.ensure_capacity(self.node_ids_subtree, position)
        self.node_ids_subtree[position] = n.id

    def remove_node(self, n: Node, tree: str = "") -> None:
        # old: mask = self.node_ids_subtree != n.id
        # old: self.node_ids_subtree = self.node_ids_subtree[mask]
        # old: self.batch_subtree = self.batch_subtree[mask]
        # old: del self.subtree[n.id]

        size = len(self.subtree)
        idx = int(np.where(self.node_ids_subtree[:size] == n.id)[0][0])
        last = size - 1
        if idx != last:
            self.node_ids_subtree[idx] = self.node_ids_subtree[last]
            self.batch_subtree[idx] = self.batch_subtree[last]
        del self.subtree[n.id]

    def get_batch_subtree(self, tree: str = "") -> NDArray:
        return self.batch_subtree[: len(self.subtree)]

    def get_node_ids_subtree(self, tree: str = "") -> NDArray:
        return self.node_ids_subtree[: len(self.subtree)]

    def add_transition_node_as_start_node(self, n: Node, tree: str = "") -> None:
        if n.id not in self.subtree:
            self.add_node(n)

    def get_node(self, id: int, tree: str = "") -> Node:
        return self.subtree.get(id)

    def get_number_of_nodes_in_tree(self) -> int:
        return len(self.subtree)


class BidirectionalTree(BaseTree):
    """
    Represents bidirectional tree structure.
    """

    def __init__(self, env: BaseProblem):
        self.order = 1
        # self.informed = Informed()
        robot_dims = sum(env.robot_dims.values())
        self.subtree = {}
        self.initial_capacity = 100000
        self.batch_subtree = np.empty(
            (self.initial_capacity, robot_dims), dtype=np.float64
        )
        self.node_ids_subtree = np.empty(self.initial_capacity, dtype=np.int64)
        self.subtree_b = {}
        self.batch_subtree_b = np.empty(
            (self.initial_capacity, robot_dims), dtype=np.float64
        )
        self.node_ids_subtree_b = np.empty(self.initial_capacity, dtype=np.int64)
        self.connected = False

    def add_node(self, n: Node, tree: str = "") -> None:
        if tree == "A" or tree == "":
            self.subtree[n.id] = n
            position = len(self.subtree) - 1
            self.batch_subtree = self.ensure_capacity(self.batch_subtree, position)
            self.batch_subtree[position, :] = n.state.q.state()
            self.node_ids_subtree = self.ensure_capacity(
                self.node_ids_subtree, position
            )
            self.node_ids_subtree[position] = n.id
        if tree == "B":
            self.subtree_b[n.id] = n
            position = len(self.subtree_b) - 1
            self.batch_subtree_b = self.ensure_capacity(self.batch_subtree_b, position)
            self.batch_subtree_b[position, :] = n.state.q.state()
            self.node_ids_subtree_b = self.ensure_capacity(
                self.node_ids_subtree_b, position
            )
            self.node_ids_subtree_b[position] = n.id

    def remove_node(self, n: Node, tree: str = "") -> None:
        if tree == "A" or tree == "":
            # old: mask = self.node_ids_subtree != n.id
            # old: self.node_ids_subtree = self.node_ids_subtree[mask]
            # old: self.batch_subtree = self.batch_subtree[mask]
            # old: del self.subtree[n.id]

            size = len(self.subtree)
            idx = int(np.where(self.node_ids_subtree[:size] == n.id)[0][0])
            last = size - 1
            if idx != last:
                self.node_ids_subtree[idx] = self.node_ids_subtree[last]
                self.batch_subtree[idx] = self.batch_subtree[last]
            del self.subtree[n.id]

        if tree == "B":
            # old: mask = self.node_ids_subtree_b != n.id
            # old: self.node_ids_subtree_b = self.node_ids_subtree_b[mask]
            # old: self.batch_subtree_b = self.batch_subtree_b[mask]
            # old: del self.subtree_b[n.id]

            size = len(self.subtree_b)
            idx = int(np.where(self.node_ids_subtree_b[:size] == n.id)[0][0])
            last = size - 1
            if idx != last:
                self.node_ids_subtree_b[idx] = self.node_ids_subtree_b[last]
                self.batch_subtree_b[idx] = self.batch_subtree_b[last]
            del self.subtree_b[n.id]

    def get_batch_subtree(self, tree: str = "") -> NDArray:
        if tree == "A" or tree == "":
            return self.batch_subtree[: len(self.subtree)]
        if tree == "B":
            return self.batch_subtree_b[: len(self.subtree_b)]

    def get_node_ids_subtree(self, tree: str = "") -> NDArray:
        if tree == "A" or tree == "":
            return self.node_ids_subtree[: len(self.subtree)]
        if tree == "B":
            return self.node_ids_subtree_b[: len(self.subtree_b)]

    def add_transition_node_as_start_node(self, n: Node, tree: str = "") -> None:
        if self.order == 1:
            if n.id not in self.subtree:
                self.add_node(n)
        else:
            if n.id not in self.subtree_b:
                self.add_node(n, "B")

    def get_node(self, id: int, tree: str = "") -> Node:
        if tree == "A" or tree == "":
            return self.subtree.get(id)
        if tree == "B":
            return self.subtree_b.get(id)

    def swap(self, balanced=False) -> None:
        """
        Swaps the forward (primary) and reversed growing subtree representations if not already connected.

        Args:
        None

        Returns:
        None: This method does not return any value.
        """
        # if we want a balanced bidirectional tree, we do not swap s long as the
        # primary tree is smaller
        if balanced and len(self.subtree) < len(self.subtree_b):
            return

        if self.connected:
            return 
            
        self.subtree, self.subtree_b = self.subtree_b, self.subtree
        self.batch_subtree, self.batch_subtree_b = (
            self.batch_subtree_b,
            self.batch_subtree,
        )
        self.node_ids_subtree, self.node_ids_subtree_b = (
            self.node_ids_subtree_b,
            self.node_ids_subtree,
        )
        self.order *= -1

    def get_number_of_nodes_in_tree(self) -> int:
        return len(self.subtree) + len(self.subtree_b)


@njit(fastmath=True, cache=True)
def find_nearest_indices(set_dists: NDArray, r: float) -> NDArray:
    """
    Finds the indices of elements in the distance array that are less than or equal to the specified threshold r.

    Args:
        set_dists (NDArray): Array of distance values.
        r (float): Threshold value for comparison (.

    Returns:
        NDArray: Array of indices where the distance values are less than or equal to the threshold.
    """

    r += 1e-10  # a small epsilon is added to mitigate floating point issues
    return np.nonzero(set_dists <= r)[0]


def save_data(data: dict, tree: bool = False):
    # Directory Handling: Ensure directory exists
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    directory = os.path.join(parent_path, "out")
    if tree:
        dir = os.path.join(directory, "Analysis/Tree")
    else:
        dir = os.path.join(directory, "Analysis")
    os.makedirs(dir, exist_ok=True)

    # Determine Next File Number: Use generator expressions for efficiency
    next_file_number = (
        max(
            (
                int(file.split(".")[0])
                for file in os.listdir(dir)
                if file.endswith(".pkl") and file.split(".")[0].isdigit()
            ),
            default=-1,
        )
        + 1
    )

    # Save Data as Pickle File
    filename = os.path.join(dir, f"{next_file_number:04d}.pkl")
    with open(filename, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


class BaseLongHorizon:
    counter: ClassVar[int] = 1

    def __init__(self, horizon_length: int = 4):
        self.reached_terminal_mode = False
        self.reached_modes = set()
        self.new_section = True
        self.reached_horizon = False
        self.horizon_length = horizon_length
        self.horizon_idx = BaseLongHorizon.counter * self.horizon_length
        self.terminal_mode = None

    def update(self, mode: Mode, init):
        if mode not in self.reached_modes:
            self.reached_modes.add(mode)

        if len(self.reached_modes) >= self.horizon_idx:
            BaseLongHorizon.counter += 1
            self.horizon_idx = BaseLongHorizon.counter * self.horizon_length
            self.reached_horizon = True
            self.terminal_mode = mode
        if init:
            self.reached_terminal_mode = True

    def reset(self):
        if not self.reached_terminal_mode:
            self.reached_horizon = False


@dataclass
class BaseRRTConfig:
    informed_sampling: bool = True
    locally_informed_sampling: bool = True
    informed_batch_size: int = 300
    
    distance_metric: str = "max_euclidean"
    p_goal: float = 0.4
    
    shortcutting: bool = True
    
    init_mode_sampling_type: str = "frontier"
    frontier_mode_sampling_probability: float = 0.98
    
    remove_redundant_nodes: bool = True
    apply_long_horizon: bool = False
    horizon_length: int = 1
    with_mode_validation: bool = True
    
    sampler: str = "auto"  # "joint" | "per_robot" | "gibbs" | "auto"
    sampler_gibbs_sweeps: int = 1  # number of full Gibbs sweeps per sample attempt
    # "auto" mode: use expected collision checks per successful sample to pick between
    # joint and gibbs per mode. Switch when gibbs becomes cheaper, switch back if not.
    # Warmup: minimum joint attempts before evaluating (avoids premature switching).
    sampler_auto_warmup: int = 20
    
    # probability of using home pose for free robots instead of the normal sampler
    # quick experiment to gauge value of per-robot ellipsoid sampling idea
    p_home_bias: float = 0.0
    with_noise: bool = False
    with_tree_visualization: bool = False
    
    # BidirectionalRRTstar
    transition_nodes: int = 10
    p_reseed_transition: float = 0.00
    birrtstar_version: int = 2
    stepsize: float = 0

    balanced_trees: bool = False


class BaseRRTstar(BasePlanner):
    """
    Represents the base class for RRT*-based algorithms, providing core functionalities for motion planning.
    """

    def __init__(
        self,
        env: BaseProblem,
        config: BaseRRTConfig = field(default_factory=BaseRRTConfig),
    ):
        self.env = env
        self.config = config
        self.dim = sum(self.env.robot_dims.values())

        self.eta = self.config.stepsize
        if self.eta == 0:
            self.eta = np.sqrt(self.dim)
            
        self.operation = Operation()
        self.modes = []
        self.trees = {}
        self.transition_node_ids = {}
        # pre-warm JIT-compiled helpers so compilation is not charged to plan()
        find_nearest_indices(np.zeros(1, dtype=np.float64), 0.0)
        self.start_time = time.time()
        self.costs = []
        self.times = []
        self.all_paths = []
        self.informed = InformedSampling(
            self.env, "sampling_based", self.config.locally_informed_sampling
        )
        self.long_horizon = BaseLongHorizon(self.config.horizon_length)
        self.mode_validation = ModeValidation(
            self.env,
            self.config.with_mode_validation,
            with_noise=self.config.with_noise,
        )
        self.check = set()
        self.blacklist_mode = set()
        self.home_pose_attempted_modes: set = set()
        # per-mode sampling stats for "auto" sampler
        # joint: [attempts, successes]
        # gibbs: [total_checks, successful_calls]
        self._joint_sampling_stats: Dict[Mode, List[int]] = {}
        self._gibbs_sampling_stats: Dict[Mode, List[int]] = {}
        # timing instrumentation
        self._sampling_time: float = 0.0
        self._coll_checking_time: float = 0.0
        self._edge_check_time_success: float = 0.0
        self._edge_check_time_failure: float = 0.0
        
        # Toggle timed vs untimed edge collision checking (change one line):
        # self._check_edge_cf = self._timed_edge_collision_free  # timed
        self._check_edge_cf = self._edge_collision_free      # untimed
        
        # self._check_config_cf = self._timed_collision_free      # timed
        self._check_config_cf = self._collision_free      # untimed

        # self._sample_config_fn = self._timed_sample_configuration  # timed
        self._sample_config_fn = self.sample_configuration             # untimed

        # self._sample_transition_config_fn = self._timed_sample_transition_configuration  # timed
        self._sample_transition_config_fn = self.sample_transition_configuration             # untimed

        # running c_free estimate via uniform samples from _sample_uniform
        self._c_free_n_total: int = 0
        self._c_free_n_free: int = 0
        self._c_free_total_volume: float = 0.0

    def _edge_collision_free(
        self, q1: "Configuration", q2: "Configuration", mode: "Mode"
    ) -> bool:
        return self.env.is_edge_collision_free(q1, q2, mode)

    def _timed_edge_collision_free(
        self, q1: "Configuration", q2: "Configuration", mode: "Mode"
    ) -> bool:
        t0 = time.perf_counter()
        result = self.env.is_edge_collision_free(q1, q2, mode)
        dt = time.perf_counter() - t0
        if result:
            self._edge_check_time_success += dt
        else:
            self._edge_check_time_failure += dt
        return result

    def _timed_collision_free(
        self, q: "Configuration", mode: "Mode"
    ) -> bool:
        t0 = time.perf_counter()
        result = self.env.is_collision_free(q, mode)
        dt = time.perf_counter() - t0
        self._coll_checking_time += dt
        return result

    def _collision_free(
        self, q: "Configuration", mode: "Mode"
    ) -> bool:
        return self.env.is_collision_free(q, mode)

        # calling once for compilation
        # find_nearest_indices(np.zeros(5), 0)

    def add_tree(
        self,
        mode: Mode,
        tree_instance: Optional[Union["SingleTree", "BidirectionalTree"]] = None,
    ) -> None:
        """
        Initializes new tree instance for the given mode.

        Args:
            mode (Mode): Current operational mode.
            tree_instance (Optional[Union["SingleTree", "BidirectionalTree"]]): Type of tree instance to initialize. Must be either SingleTree or BidirectionalTree.

        Returns:
            None: This method does not return any value.
        """

        if tree_instance is None:
            raise ValueError(
                "You must provide a tree instance type: SingleTree or BidirectionalTree."
            )

        # Check type and initialize the tree
        if tree_instance == SingleTree:
            self.trees[mode] = SingleTree(self.env)
        elif tree_instance == BidirectionalTree:
            self.trees[mode] = BidirectionalTree(self.env)
        else:
            raise TypeError("tree_instance must be SingleTree or BidirectionalTree.")

    def add_new_mode(
        self,
        q: Optional[Configuration] = None,
        mode: Optional[Mode] = None,
        tree_instance: Optional[Union["SingleTree", "BidirectionalTree"]] = None,
    ) -> None:
        """
        Initializes a new mode (including its corresponding tree instance and performs informed initialization).

        Args:
            q (Configuration): Configuration used to determine the new mode.
            mode (Mode): The current mode from which to get the next mode.
            tree_instance (Optional[Union["SingleTree", "BidirectionalTree"]]): Type of tree instance to initialize for the next mode. Must be either SingleTree or BidirectionalTree.

        Returns:
            None: This method does not return any value.
        """
        if mode is None:
            new_modes = [self.env.get_start_mode()]
        else:
            self.check.add(mode)
            new_modes = self.env.get_next_modes(q, mode)
            new_modes = self.mode_validation.get_valid_modes(mode, tuple(new_modes))
            if new_modes == []:
                self.mode_validation.propagate_invalid(mode)
                self.modes = self.mode_validation.remove_invalid_modes(self.modes)
            self.save_tree_data()

        for new_mode in new_modes:
            if new_mode in self.modes:
                continue
            if new_mode in self.blacklist_mode:
                continue

            self.modes.append(new_mode)
            self.add_tree(new_mode, tree_instance)

        if self.config.apply_long_horizon and mode in self.modes:
            self.long_horizon.update(mode, self.operation.init_sol)

    def mark_node_as_transition(self, mode: Mode, n: Node) -> None:
        """
        Marks node as a potential transition node for the specified mode.

        Args:
            mode (Mode): Current operational mode in which the node is marked as a transition node.
            n (Node): Node to be marked as a transition node.

        Returns:
            None: This method does not return any value.
        """

        n.transition = True
        if mode not in self.transition_node_ids:
            self.transition_node_ids[mode] = []

        if n.id not in self.transition_node_ids[mode]:
            self.transition_node_ids[mode].append(n.id)

    def convert_node_to_transition_node(self, mode: Mode, n: Node) -> None:
        """
        Marks a node as potential transition node in the specified mode and adds it as a start node for each subsequent mode.

        Args:
            mode (Mode): Current operational mode in which the node is converted to a transition node.
            n (Node): Node to convert into a transition node.

        Returns:
            None: This method does not return any value.
        """
        self.mark_node_as_transition(mode, n)
        if self.env.is_terminal_mode(mode):
            return
        if mode not in self.modes:
            return
        next_modes = self.env.get_next_modes(n.state.q, mode)
        next_modes = self.mode_validation.get_valid_modes(mode, tuple(next_modes))

        if next_modes == []:
            self.mode_validation.propagate_invalid(mode)
            self.modes = self.mode_validation.remove_invalid_modes(self.modes)

        for next_mode in next_modes:
            if next_mode not in self.modes:
                tree_type = type(self.trees[mode])
                if tree_type == BidirectionalTree:
                    self.trees[mode].connected = True
                self.add_new_mode(n.state.q, mode, tree_type)
            self.trees[next_mode].add_transition_node_as_start_node(n)
            if self.trees[next_mode].order == 1:
                index = len(self.trees[next_mode].subtree) - 1
                tree = "A"
            else:
                index = len(self.trees[next_mode].subtree_b) - 1
                tree = "B"
            # index = np.where(self.trees[mode].get_node_ids_subtree() == n.id)

            # need to rewire tree of next mode as well
            if index != 0:
                N_near_batch, n_near_costs, node_indices = self.near(
                    next_mode, n, index, tree=tree
                )
                batch_cost = self.env.batch_config_cost(n.state.q, N_near_batch)
                if self.rewire(
                    next_mode, node_indices, n, batch_cost, n_near_costs, tree
                ):
                    self.update_cost(next_mode, n)

    def get_lb_transition_node_id(
        self, modes: List[Mode]
    ) -> Tuple[Tuple[float, int], Mode]:
        """
        Retrieves the lower bound cost and corresponding transition node ID from a list of modes.

        Args:
            modes (List[Mode]): List of modes to search for the transition node with the minimum cost.

        Returns:
            Tuple:
                - float: A tuple with the lower bound cost.
                - int: Corresponding transition node id.
                - Mode: Mode from which the transition node was selected.
        """

        indices, costs = [], []
        for mode in modes:
            i = np.argmin(self.operation.costs[self.transition_node_ids[mode]], axis=0)
            indices.append(i)
            costs.append(self.operation.costs[self.transition_node_ids[mode]][i])

        idx = np.argmin(costs, axis=0)
        m = modes[idx]
        node_id = self.transition_node_ids[m][indices[idx]]
        return (costs[idx], node_id), m
        # sorted_indices = costs.argsort()
        # for idx in sorted_indices:

        #     if self.trees[mode].order == 1 and node_id in self.trees[mode].subtree:

        #     elif self.trees[mode].order == -1 and node_id in self.trees[mode].subtree_b:
        #         return (costs[idx], node_id)

    def get_transition_node(self, mode: Mode, id: int) -> Node:
        """
        Retrieves transition node from the primary subtree of the given mode by its id.

        Args:
            mode (Mode): Current operational mode from which the transition node is to be retrieved.
            id (int): The unique ID of the transition node.

        Returns:
            Node: Transition node from the primary subtree (i.e. 'subtree' if order is 1; otherwise 'subtree_b' as trees are swapped).
        """

        if self.trees[mode].order == 1:
            return self.trees[mode].subtree.get(id)
        else:
            return self.trees[mode].subtree_b.get(id)

    def get_lebesgue_measure_of_free_configuration_space(
        self, num_samples: int = 1000
    ) -> None:
        """
        Sets the free configuration space parameter by estimating its Lebesgue measure using Halton sequence sampling.

        Args:
            num_samples (int): Number of samples to generate the estimation.

        Returns:
            None: This method does not return any value.
        """

        total_volume = 1.0
        limits = []

        for robot in self.env.robots:
            r_indices = self.env.robot_idx[robot]  # Get joint indices for the robot
            lims = self.env.limits[:, r_indices]  # Extract joint limits
            limits.append(lims)
            total_volume *= np.prod(lims[1] - lims[0])  # Compute volume product

        # Generate Halton sequence samples
        try:
            halton_sampler = Halton(self.d, scramble=False)
            halton_samples = halton_sampler.random(num_samples)  # Scaled [0,1] samples
        except ImportError:
            halton_samples = np.random.rand(
                num_samples, self.d
            )  # Fallback to uniform random sampling

        # Map Halton samples to configuration space
        free_samples = 0
        q_robots = np.empty(len(self.env.robots), dtype=object)
        for i, (robot, lims) in enumerate(zip(self.env.robots, limits)):
            q_robots[i] = lims[0] + halton_samples[:, self.env.robot_idx[robot]] * (
                lims[1] - lims[0]
            )
        # idx = 0
        # q_ellipse = []
        for i in range(num_samples):
            q = [q_robot[i] for q_robot in q_robots]
            q = type(self.env.get_start_pos()).from_list(q)
            # if idx < 800:
            #     q_ellipse.append(q.state())
            #     idx+=1

            # Check if sample is collision-free
            if self.env.is_collision_free(q, self.env.start_mode):
                free_samples += 1
        # Estimate C_free measure
        self.c_free = (free_samples / num_samples) * total_volume

    def set_gamma_rrtstar(self) -> None:
        """
        Sets the constant gamma parameter used to define the search radius in RRT* and the dimension of state space.

        Args:
            None

        Returns:
            None: This method does not return any value.
        """

        self.d = sum(self.env.robot_dims.values())
        unit_ball_volume = math.pi ** (self.d / 2) / math.gamma((self.d / 2) + 1)
        self._unit_ball_volume = unit_ball_volume

        # compute total C-space volume from joint limits (no collision checking)
        total_volume = 1.0
        for robot in self.env.robots:
            lims = self.env.limits[:, self.env.robot_idx[robot]]
            total_volume *= np.prod(lims[1] - lims[0])
        self._c_free_total_volume = total_volume

        # initialise c_free optimistically with total volume; refined online by _sample_uniform
        self.c_free = total_volume
        self.gamma_rrtstar = (
            (2 * (1 + 1 / self.d)) ** (1 / self.d)
            * (self.c_free / unit_ball_volume) ** (1 / self.d)
        ) * self.eta

    _C_FREE_MIN_SAMPLES = 200   # don't update until we have this many uniform draws
    _C_FREE_UPDATE_EVERY = 500  # update every this many additional draws

    def _update_c_free_from_samples(self) -> None:
        """Recompute c_free and gamma_rrtstar from running uniform-sample counts."""
        self.c_free = (self._c_free_n_free / self._c_free_n_total) * self._c_free_total_volume
        self.gamma_rrtstar = (
            (2 * (1 + 1 / self.d)) ** (1 / self.d)
            * (self.c_free / self._unit_ball_volume) ** (1 / self.d)
        ) * self.eta

    def get_home_poses(self, mode: Mode) -> List[NDArray]:
        """
        Retrieves home poses (i.e., the most recent completed task configurations) for all agents of given mode.

        Args:
            mode (Mode): Current operational mode.

        Returns:
            List[NDArray]: Representing home poses for each agent.
        """

        # Start mode
        if mode.prev_mode is None:
            q = self.env.start_pos
            q_new = []
            for r_idx in range(len(self.env.robots)):
                q_new.append(q.robot_state(r_idx))
        # all other modes
        else:
            previous_task = self.env.get_active_task(mode.prev_mode, mode.task_ids)
            goal = previous_task.goal.sample(mode.prev_mode)
            q = self.get_home_poses(mode.prev_mode)
            q_new = []
            end_idx = 0
            for robot in self.env.robots:
                r_idx = self.env.robots.index(robot)
                if robot in previous_task.robots:
                    dim = self.env.robot_dims[robot]
                    indices = list(range(end_idx, end_idx + dim))
                    q_new.append(goal[indices])
                    end_idx += dim
                    continue
                q_new.append(q[r_idx])
        return q_new

    def get_task_goal_of_agent(self, mode: Mode, r: str) -> NDArray:
        """Returns task goal of agent in current mode"""
        """
        Retrieves task goal configuration of given mode for the specified robot.

        Args:
            mode (Mode): Current operational mode.
            r (str): The identifier of the robot whose task goal is to be retrieved.

        Returns:
            NDArray: Goal configuration for the specified robot. 
        """

        # task = self.env.get_active_task(mode, self.mode_validation.get_valid_next_ids(mode))
        # if r not in task.robots:
        r_idx = self.env.robots.index(r)
        goal = self.env.tasks[mode.task_ids[r_idx]].goal.sample(mode)
        if len(goal) == self.env.robot_dims[r]:
            return goal
        else:
            constrained_robot = self.env.get_active_task(
                mode, self.mode_validation.get_valid_next_ids(mode)
            ).robots
            end_idx = 0
            for robot in constrained_robot:
                dim = self.env.robot_dims[r]
                if robot == r:
                    indices = list(range(end_idx, end_idx + dim))
                    return goal[indices]

                end_idx += dim
        # goal = task.goal.sample(mode)
        # if len(goal) == self.env.robot_dims[r]:
        #    return goal
        # else:
        #     return goal[self.env.robot_idx[r]]

    def _sample_collision_free_joint(
        self, mode: Mode, pinned: Dict[str, NDArray] = {}
    ) -> Configuration | None:
        """Joint rejection sampling: sample all free robots at once, accept if collision-free."""

        if not hasattr(self, "sample_attempts"):
            self.sample_attempts = 0

        if not hasattr(self, "succ_samples"):
            self.succ_samples = 0

        if not hasattr(self, "coll_checks"):
            self.coll_checks = 0

        self.sample_attempts += 1

        q = self.env.sample_config_uniform_in_limits()
        for robot, values in pinned.items():
            i = self.env.robots.index(robot)
            q[i] = values

        self.coll_checks += 1
        if self.env.is_collision_free(q, mode):
            self.succ_samples += 1
            return q
        
        return None

    def _sample_collision_free_per_robot(
        self, mode: Mode, pinned: Dict[str, NDArray] = {}
    ) -> Configuration | None:
        """
        Per-robot rejection sampling: retry each free robot independently using
        per-robot collision checking, then do a final full check for inter-robot
        collisions. Avoids the exponential curse of dimensionality of joint sampling.

        Returns None if any free robot exceeds its per-robot attempt budget (100).
        """
        q = self.env.sample_config_uniform_in_limits()

        for robot, values in pinned.items():
            i = self.env.robots.index(robot)
            q[i] = values

        for i, robot in enumerate(self.env.robots):
            if robot in pinned:
                continue
            lims = self.env.limits[:, self.env.robot_idx[robot]]
            for _ in range(100):
                if self.env.is_collision_free_for_robot([robot], q.state(), mode):
                    break
                q[i] = np.random.uniform(lims[0], lims[1])
            else:
                return None

        if self.env.is_collision_free(q, mode):
            return q
        return None

    def _resolve_sampler(self, mode: Mode | None = None) -> str:
        if self.config.sampler == "auto":
            if mode is None:
                return "joint"
            joint_stats = self._joint_sampling_stats.get(mode)
            if joint_stats is None or joint_stats[0] < self.config.sampler_auto_warmup:
                return "joint"
            # expected checks per successful sample:
            #   joint: 1/p_success = attempts/successes (inf if no success yet)
            #   gibbs: total_checks/successful_calls (inf if no success yet)
            joint_attempts, joint_successes = joint_stats
            expected_joint = joint_attempts / max(joint_successes, 1e-9)
            gibbs_stats = self._gibbs_sampling_stats.get(mode)
            if gibbs_stats is None or gibbs_stats[1] == 0:
                # no gibbs data yet — switch speculatively if joint is very expensive
                return "gibbs" if joint_successes == 0 else "joint"
            expected_gibbs = gibbs_stats[0] / gibbs_stats[1]
            return "gibbs" if expected_gibbs < expected_joint else "joint"
        return self.config.sampler

    def _get_seed_for_mode(self, mode: Mode) -> Configuration:
        """Returns a random existing node config for the given mode, or start_pos as fallback."""
        if mode in self.trees:
            node_ids = self.trees[mode].get_node_ids_subtree()
            if len(node_ids) > 0:
                node = self.trees[mode].get_node(np.random.choice(node_ids))
                return node.state.q
        return self.env.get_start_pos()

    def _sample_collision_free_gibbs(
        self, mode: Mode, pinned: Dict[str, NDArray] = {}
    ) -> Tuple[Configuration | None, int]:
        """
        Approximate Gibbs sampling: seed from an existing tree node, then perform
        multiple full sweeps over free robots. Each robot draws a fresh candidate
        and accepts via a full collision check (including inter-robot).

        Returns (config, n_checks) so the caller can track expected cost.

        NOTE: This deviates from true Gibbs sampling. Standard Gibbs loops on each
        variable until a valid sample is found, keeping the chain in a valid state
        after every update. Here, if a robot exhausts its per-robot budget (100
        attempts), we keep its current (potentially invalid) position and move on.
        This is necessary to avoid false rejections caused by not-yet-swept robots
        whose seed positions collide with the newly pinned robots. The consequence
        is that intermediate states within a sweep can be invalid, which breaks the
        theoretical convergence guarantee. In practice this is resolved over multiple
        sweeps: the first sweep is approximate, subsequent sweeps are genuine Gibbs
        steps since all robots have been resampled at least once.
        """
        seed = self._get_seed_for_mode(mode)
        n_checks = 0

        q = self.env.sample_config_uniform_in_limits()
        for i in range(len(self.env.robots)):
            q[i] = seed[i].copy()

        for robot, values in pinned.items():
            i = self.env.robots.index(robot)
            q[i] = values

        for _ in range(self.config.sampler_gibbs_sweeps):
            for i, robot in enumerate(self.env.robots):
                if robot in pinned:
                    continue
                lims = self.env.limits[:, self.env.robot_idx[robot]]
                for _ in range(100):
                    q[i] = np.random.uniform(lims[0], lims[1])
                    n_checks += 1
                    if self.env.is_collision_free(q, mode):
                        break
                # no early return on failure — robot j may be causing false rejections
                # for robot i; continue sweeping and let subsequent sweeps resolve it

        n_checks += 1
        if self.env.is_collision_free(q, mode):
            return q, n_checks

        return None, n_checks

    def _sample_collision_free(
        self, mode: Mode, pinned: Dict[str, NDArray] = {}
    ) -> Configuration | None:
        # TODO: caching opportunity — every is_collision_free hit inside the per-robot
        # and Gibbs samplers already produces a valid config, not just the final one.
        # A per-mode cache (Dict[Mode, deque[Configuration]]) could store these and be
        # consumed here first. On consumption, overwrite pinned slots with current goal
        # values and re-validate cheaply. Unpinned calls skip the overwrite.
        # Cost: one copy per cache push; benefit: fewer full sampling rounds, especially
        # for high robot counts. Needs care around goal resampling (GoalRegion vs point).

        if self.config.p_home_bias > 0 and np.random.random() < self.config.p_home_bias:
            next_ids = self.mode_validation.get_valid_next_ids(mode)
            task = self.env.get_active_task(mode, next_ids)
            active_robots = set(task.robots)
            home = self.env.get_start_pos()
            q = self.env.sample_config_uniform_in_limits()
            for i, robot in enumerate(self.env.robots):
                if robot in pinned:
                    q[i] = pinned[robot]
                elif robot not in active_robots:
                    q[i] = home[i].copy()  # truly free robots go to home pose
                # active robots keep their random value from sample_config_uniform_in_limits
            if self.env.is_collision_free(q, mode):
                return q

        sampler = self._resolve_sampler(mode)
        if sampler == "joint":
            q = self._sample_collision_free_joint(mode, pinned)
            if self.config.sampler == "auto":
                stats = self._joint_sampling_stats.setdefault(mode, [0, 0])
                stats[0] += 1           # attempts = collision checks (1 per joint attempt)
                stats[1] += q is not None  # successes
            return q
        elif sampler == "gibbs":
            q, n_checks = self._sample_collision_free_gibbs(mode, pinned)
            if self.config.sampler == "auto":
                stats = self._gibbs_sampling_stats.setdefault(mode, [0, 0])
                stats[0] += n_checks    # total collision checks
                stats[1] += q is not None  # successful calls
            return q
        else:
            return self._sample_collision_free_per_robot(mode, pinned)

    def sample_transition_configuration(self, mode) -> Configuration | None:
        """
        Samples a collision-free transition configuration for the given mode.

        Args:
            mode (Mode): Current operational mode.

        Returns:
            Configuration: Collision-free configuration constructed by combining goal samples (active robots) with random samples (non-active robots).
        """
        failed_attempts = 0
        while True:
            if failed_attempts > 10000:
                print("Failed to sample transition configuration after 10000 attempts.")
                if self.config.with_mode_validation:
                    self.mode_validation.add_invalid_mode(mode)
                    self.mode_validation.propagate_invalid(mode.prev_mode)
                    self.modes = self.mode_validation.remove_invalid_modes(self.modes)
                else:
                    self.blacklist_mode.add(mode)
                    self.modes.remove(mode)

                return

            next_ids = self.mode_validation.get_valid_next_ids(mode)
            if not next_ids and not self.env.is_terminal_mode(mode):
                return

            task = self.env.get_active_task(mode, next_ids)
            goal = task.goal.sample(mode)
            pinned = {}
            end_idx = 0
            for robot in task.robots:
                dim = self.env.robot_dims[robot]
                pinned[robot] = goal[end_idx : end_idx + dim]
                end_idx += dim

            # Try home pose + pinned once per mode across all calls
            if mode not in self.home_pose_attempted_modes:
                self.home_pose_attempted_modes.add(mode)
                home = self.env.get_start_pos()
                q = self.env.sample_config_uniform_in_limits()
                for i, robot in enumerate(self.env.robots):
                    q[i] = pinned[robot] if robot in pinned else home[i].copy()
                if self.env.is_collision_free(q, mode):
                    return q

            q = self._sample_collision_free(mode, pinned)
            if q is not None:
                # self.env.show(False)
                return q

            failed_attempts += 1

    def _sample_goal(
        self,
        mode: Mode,
        transition_node_ids: Dict[Mode, List[int]],
        tree_order: int = 1,
    ) -> Configuration | None:
        # goal sampling
        while True:
            if tree_order == -1:
                if mode.prev_mode is None or mode == self.env.start_mode:
                    return self.env.start_pos
                else:
                    transition_nodes_id = transition_node_ids[mode.prev_mode]
                    if transition_nodes_id == []:
                        return self.sample_transition_configuration(mode.prev_mode)

                    else:
                        node_id = np.random.choice(transition_nodes_id)
                        node = self.trees[mode.prev_mode].subtree.get(node_id)
                        if node is None:
                            node = self.trees[mode.prev_mode].subtree_b.get(node_id)
                        return node.state.q

            if self.operation.init_sol:
                if (
                    not self.env.is_terminal_mode(mode)
                ):
                    q = self.informed.generate_transitions(
                        self.modes,
                        self.config.informed_batch_size,
                        self.operation.path_shortcutting_interpolated,
                        active_mode=mode,
                    )
                    if q == []:
                        return

                    if self.env.is_collision_free(q, mode):
                        return q
                    continue

            q = self.sample_transition_configuration(mode)

            if q is None:
                return

            return q

            # if random.choice([0, 1]) == 0:
            #     return q

            # while True:
            #     q_noise = []
            #     for r in range(len(self.env.robots)):
            #         q_robot = q.robot_state(r)
            #         noise = np.random.normal(0, 0.1, q_robot.shape)
            #         q_noise.append(q_robot + noise)
            #     q = type(self.env.get_start_pos()).from_list(q_noise)
            #     if self.env.is_collision_free(q, mode):
            #         return q

    def _sample_uniform(self, mode: Mode):
        while True:
            q = self.env.sample_config_uniform_in_limits()
            self._c_free_n_total += 1
            if self.env.is_collision_free(q, mode):
                self._c_free_n_free += 1
                if (
                    self._c_free_n_total >= self._C_FREE_MIN_SAMPLES
                    and self._c_free_n_total % self._C_FREE_UPDATE_EVERY == 0
                ):
                    self._update_c_free_from_samples()
                return q
            # self.env.show(False)

    def sample_informed(
        self,
        mode: Mode,
    ) -> Configuration | None:
        """
        Samples a collision-free configuration for the given mode using the specified sampling strategy.

        Args:
            mode (Mode): Current operational mode.

        Returns:
            Configuration: Collision-free configuration within specified limits for the robots based on the sampling strategy.
        """
        next_ids = self.mode_validation.get_valid_next_ids(mode)
        if not next_ids and not self.env.is_terminal_mode(mode):
            return

        while True:
            # informed sampling
            q = self.informed.generate_samples(
                self.modes,
                self.config.informed_batch_size,
                self.operation.path_shortcutting_interpolated,
                active_mode=mode,
            )
            if q == []:
                return

            if self.env.is_collision_free(q, mode):
                return q

    def get_termination_modes(self) -> List[Mode]:
        """
        Retrieves a list of modes that are considered terminal based on the environment's criteria.

        Args:
            None

        Returns:
            List[Mode]:List containing all terminal modes."""

        termination_modes = []
        for mode in self.modes:
            if self.env.is_terminal_mode(mode):
                termination_modes.append(mode)
        return termination_modes

    def nearest(
        self, mode: Mode, q_rand: Configuration, tree: str = ""
    ) -> Tuple[Node, float, NDArray, int]:
        """
        Retrieves nearest node to a configuration in the specified subree for a given mode.

        Args:
            mode (Mode): Current operational mode.
            q_rand (Configuration): Random configuration for which the nearest node is sought.
            tree (str): Identifier of subtree in which the nearest node is searched for
        Returns:
            Tuple:
                - Node: Nearest node.
                - float: Distance from q_rand to nearest node.
                - NDArray: Array of distances from q_rand to all nodes in the specified subtree.
                - int: Index of the nearest node in the distance array.
        """

        set_dists = batch_config_dist(
            q_rand,
            self.trees[mode].get_batch_subtree(tree),
            self.config.distance_metric,
        )
        idx = np.argmin(set_dists)
        node_id = self.trees[mode].get_node_ids_subtree(tree)[idx]
        # print([float(set_dists[idx])])
        return self.trees[mode].get_node(node_id, tree), set_dists[idx], set_dists, idx

    def steer(
        self,
        mode: Mode,
        n_nearest: Node,
        q_rand: Configuration,
        dist: NDArray,
        i: int = 1,
    ) -> Optional[State]:
        """
        Steers from the nearest node toward the target configuration by taking an incremental step.

        Args:
            mode (Mode): Current operational mode.
            n_nearest (Node): Nearest node from which steering begins.
            q_rand (Configuration): Target random configuration for steering.
            dist (NDArray): Distance between the nearest node and q_rand.
            i (int): Step index for incremental movement.

        Returns:
            State: New obtained state by steering from n_nearest towards q_rand (i.e. q_rand, if within the allowed step size; otherwise, advances by one incremental step).
        """

        if np.equal(n_nearest.state.q.state(), q_rand.state()).all():
            return None
        q_nearest = n_nearest.state.q.state()
        direction = q_rand.q - q_nearest
        if self.config.distance_metric != "max_euclidean":
            # most independent of the number of robots and their dimension
            dist = batch_config_dist(
                n_nearest.state.q, [q_rand], metric="max_euclidean"
            )
        N = float((dist / self.eta))  # to have exactly the step size

        if N <= 1 or int(N) == i - 1:  # for bidirectional or drrt
            q_new = q_rand.q
        else:
            q_new = q_nearest + (direction * (i / N))
        state_new = State(self.env.get_start_pos().from_flat(q_new), mode)
        return state_new

    def near(
        self,
        mode: Mode,
        n_new: Node,
        n_nearest_idx: int,
        set_dists: NDArray = None,
        tree: str = "",
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Retrieves neighbors of a node within a calculated radius for the given mode.

        Args:
            mode (Mode):  Current operational mode.
            n_new (Node): New node for which neighbors are being identified.
            n_nearest_idx (int): Index of the nearest node to n_new.
            set_dists (Optional[NDArray]): Precomputed distances from n_new to all nodes in the specified subtree.
            tree (str): Identifier of subtree in which the nearest node is searched for

        Returns:
            Tuple:
                - NDArray: Batch of neighbors near n_new.
                - NDArray: Corresponding cost values of these nodes.
                - NDArray: Corresponding IDs of these nodes.
        """

        batch_subtree = self.trees[mode].get_batch_subtree(tree)
        if set_dists is None:
            set_dists = batch_config_dist(
                n_new.state.q, batch_subtree, self.config.distance_metric
            )
        vertices = self.trees[mode].get_number_of_nodes_in_tree()
        r = np.minimum(
            self.gamma_rrtstar * (np.log(vertices) / vertices) ** (1 / self.d), self.eta
        )
        indices = find_nearest_indices(set_dists, r)  # indices of batch_subtree
        if n_nearest_idx not in indices:
            indices = np.insert(indices, 0, n_nearest_idx)

        node_indices = self.trees[mode].get_node_ids_subtree(tree)[indices]
        n_near_costs = self.operation.costs[node_indices]
        N_near_batch = batch_subtree[indices]
        return N_near_batch, n_near_costs, node_indices

    def find_parent(
        self,
        mode: Mode,
        node_indices: NDArray,
        n_new: Node,
        n_nearest: Node,
        batch_cost: NDArray,
        n_near_costs: NDArray,
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

        idx = np.where(node_indices == n_nearest.id)[0][0]
        c_new_tensor = n_near_costs + batch_cost
        c_min = c_new_tensor[idx]
        c_min_to_parent = batch_cost[idx]
        n_min = n_nearest
        valid_mask = c_new_tensor < c_min
        if np.any(valid_mask):
            sorted_indices = np.where(valid_mask)[0][
                np.argsort(c_new_tensor[valid_mask])
            ]
            for idx in sorted_indices:
                node = self.trees[mode].subtree.get(node_indices[idx].item())
                if self._check_edge_cf(node.state.q, n_new.state.q, mode):
                    c_min = c_new_tensor[idx]
                    c_min_to_parent = batch_cost[idx]  # Update minimum cost
                    n_min = node  # Update parent node
                    break
        n_new.parent = n_min
        n_new.cost_to_parent = c_min_to_parent
        n_min.children.append(n_new)  # Set child
        self.operation.costs = self.trees[mode].ensure_capacity(
            self.operation.costs, n_new.id
        )
        n_new.cost = c_min
        self.trees[mode].add_node(n_new)

    def rewire(
        self,
        mode: Mode,
        node_indices: NDArray,
        n_new: Node,
        batch_cost: NDArray,
        n_near_costs: NDArray,
        tree: str = "",
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

        rewired = False
        c_potential_tensor = n_new.cost + batch_cost

        improvement_mask = c_potential_tensor < n_near_costs

        if np.any(improvement_mask):
            improved_indices = np.nonzero(improvement_mask)[0]

            for idx in improved_indices:
                n_near = self.trees[mode].get_node(node_indices[idx].item(), tree)
                if (
                    n_near == n_new.parent
                    or n_near.cost == np.inf
                    or n_near.id == n_new.id
                ):
                    continue
                if (
                    n_new.state.mode == n_near.state.mode
                    or n_new.state.mode == n_near.state.mode.prev_mode
                ):
                    if self._check_edge_cf(
                        n_new.state.q, n_near.state.q, mode
                    ):
                        if n_near.parent is not None:
                            n_near.parent.children.remove(n_near)
                        n_near.parent = n_new
                        n_new.children.append(n_near)

                        n_near.cost = c_potential_tensor[idx]
                        n_near.cost_to_parent = batch_cost[idx]
                        if n_near.children != []:
                            rewired = True
        return rewired

    def update_results_tracking(self, cost: float, path: List[State]) -> None:
        """
        Updates the tracking of costs, times, and all explored paths if the path ends in a terminal mode.

        Args:
            cost (float): The cost of the path.
            path (List[State]): The path as a list of states.

        Returns:
            None: This method does not return any value.
        """
        if path == []:
            return
        if not self.env.is_terminal_mode(path[-1].mode):
            return
        self.costs.append(cost)
        self.times.append(time.time() - self.start_time)
        self.all_paths.append(path)

        modes = []
        for s in path:
            if len(modes) == 0 or s.mode.task_ids != modes[-1]:
                modes.append(s.mode.task_ids)

        print(f"New cost: {cost} at time {time.time() - self.start_time}")
        print("Modes: ", modes)

    def generate_path(
        self, mode: Mode, n: Node, shortcutting_bool: bool = True
    ) -> None:
        """
        Sets path from the specified node back to the root by following parent links, with optional shortcutting.

        Args:
            mode (Mode): Current operational Mode.
            n (Node): Starting node from which the path is generated.
            shortcutting_bool (bool): If True, applies shortcutting to the generated path.

        Returns:
            None: This method does not return any value.
        """

        path_nodes, path, path_modes, path_shortcutting = [], [], [], []
        while n:
            path_nodes.append(n)
            path_modes.append(n.state.mode.task_ids)
            path.append(n.state)
            # if shortcutting_bool:
            path_shortcutting.append(n.state)
            if n.parent is not None and n.parent.state.mode != n.state.mode:
                new_state = State(n.parent.state.q, n.state.mode)
                path_shortcutting.append(new_state)
            n = n.parent
        path_in_order = path[::-1]
        self.operation.path_modes = path_modes[::-1]
        self.operation.path = path_in_order
        self.operation.path_nodes = path_nodes[::-1]
        self.operation.cost = self.operation.path_nodes[-1].cost
        self.update_results_tracking(self.operation.cost, self.operation.path)
        self.operation.path_shortcutting = path_shortcutting[
            ::-1
        ]  # includes transiiton node twice
        self.operation.path_shortcutting_interpolated = (
            interpolate_path(path_shortcutting)
        )
        if (
            (
                self.operation.init_sol
                or self.config.apply_long_horizon
                and self.long_horizon.reached_horizon
            )
            and self.config.shortcutting
            and shortcutting_bool
        ):
            # print(f"-- M", mode.task_ids, "Cost: ", self.operation.cost.item())
            shortcut_path_, result = robot_mode_shortcut(
                self.env,
                self.operation.path_shortcutting,
                100,
                resolution=self.env.collision_resolution,
                tolerance=self.env.collision_tolerance,
            )
            if self.config.remove_redundant_nodes:
                # print(np.sum(self.env.batch_config_cost(shortcut_path[:-1], shortcut_path[1:])))
                shortcut_path = remove_interpolated_nodes(
                    shortcut_path_
                )
            else:
                shortcut_path = shortcut_path_
                # print(np.sum(self.env.batch_config_cost(shortcut_path[:-1], shortcut_path[1:])))
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(3, 2, figsize=(12, 15))

            # samples = [sample.q.state()[3:5] for sample in shortcut_path]
            # axes[1, 0].scatter([s[0] for s in samples], [s[1] for s in samples])
            # axes[1, 0].set_title('Agent 2')

            # samples = [sample.q.state()[:2] for sample in shortcut_path]
            # axes[0, 0].scatter([s[0] for s in samples], [s[1] for s in samples])
            # axes[0, 0].set_title('Agent 1')

            # samples = [sample.q.state()[3:5] for sample in shortcut_path_]
            # axes[1, 1].scatter([s[0] for s in samples], [s[1] for s in samples])
            # axes[1, 1].set_title('Agent 2')

            # samples = [sample.q.state()[:2] for sample in shortcut_path_]
            # axes[0, 1].scatter([s[0] for s in samples], [s[1] for s in samples])
            # axes[0, 1].set_title('Agent 1')

            # samples = [sample.q.state()[5:] for sample in shortcut_path]
            # axes[2, 1].scatter([s[0] for s in samples], [s[1] for s in samples])
            # axes[2, 1].set_title('Agent 3')

            # samples = [sample.q.state()[5:] for sample in shortcut_path_]
            # axes[2, 0].scatter([s[0] for s in samples], [s[1] for s in samples])
            # axes[2, 0].set_title('Agent 3')

            # # Adjust layout and display the plots
            # plt.tight_layout()
            # plt.show()
            # batch_cost = self.env.batch_config_cost(shortcut_path[:-1], shortcut_path[1:])
            # shortcut_path_costs = cumulative_sum(batch_cost)
            # shortcut_path_costs = np.insert(shortcut_path_costs, 0, 0.0)
            if result[0][-1] < self.operation.cost:
                self.update_results_tracking(result[0][-1], shortcut_path)
                self.tree_extension(shortcut_path)

            if (
                self.config.apply_long_horizon
                and not self.long_horizon.reached_terminal_mode
            ):
                self.long_horizon.reset()
                self.operation.cost = np.inf

    def random_mode(self) -> Mode:
        """
        Randomly selects a mode based on the current mode sampling strategy.

        Args:
            None

        Returns:
            Mode: Sampled mode based on the mode sampling strategy.
        """

        mode_sampling_type = self.config.init_mode_sampling_type
        if self.operation.init_sol:
            mode_sampling_type = "uniform_reached"

        num_modes = len(self.modes)
        if num_modes == 1:
            return random.choice(self.modes)
        # if self.operation.task_sequence == [] and self.config.mode_sampling != 0:
        # elif self.operation.init_sol and mode_sampling_type != "weighted":
        #     p = [1 / num_modes] * num_modes

        elif mode_sampling_type == "uniform_reached":# is None:
            # equally (= mode uniformly)
            return random.choice(self.modes)

        elif mode_sampling_type == "greedy": # 1:
            # greedy (only latest mode is selected until initial paths are found and then it continues with equally)
            probability = [0] * (num_modes)
            probability[-1] = 1

            p = probability

        elif mode_sampling_type == "weighted": #0:
            # Uniformly
            total_nodes = sum(
                self.trees[mode].get_number_of_nodes_in_tree() for mode in self.modes
            )
            # Calculate probabilities inversely proportional to node counts
            inverse_probabilities = [
                1 - (len(self.trees[mode].subtree) / total_nodes) for mode in self.modes
            ]
            # Normalize the probabilities to sum to 1
            total_inverse = sum(inverse_probabilities)
            p = [inv_prob / total_inverse for inv_prob in inverse_probabilities]

        elif mode_sampling_type == "frontier":
        # else:
            # not working for unordered envs
            frontier_modes = set()

            for m in self.modes:
                if not m.next_modes:
                    frontier_modes.add(m)

            p_frontier = self.config.frontier_mode_sampling_probability
            # p_frontier = self.config.mode_sampling
            p_remaining = 1 - p_frontier

            total_nodes = sum(
                self.trees[mode].get_number_of_nodes_in_tree() for mode in self.modes
            )
            # Calculate probabilities inversely proportional to node counts
            subtree_lengths = {
                mode: len(self.trees[mode].subtree) for mode in self.modes
            }
            inverse_probabilities = [
                1 - (subtree_lengths[mode] / total_nodes)
                for mode in self.modes
                if mode not in frontier_modes
            ]
            total_inverse = sum(inverse_probabilities)

            p = []

            # print(len(frontier_modes))

            for m in self.modes:
                if m in frontier_modes:
                    tmp = p_frontier / len(frontier_modes)
                else:
                    tmp = (
                        (1 - (subtree_lengths[m] / total_nodes))
                        / total_inverse
                        * p_remaining
                    )

                p.append(tmp)

        # print(len(self.modes), sum(p))

        # return np.random.choice(self.modes, p = p)
        return random.choices(self.modes, weights=p, k=1)[0]

    def sample_configuration(self, mode: Mode) -> Configuration | None:
        """
        Samples a node configuration from the manifold based on various probabilistic strategies.

        Args:
            mode (Mode): Current operational mode.

        Returns:
            Configuration: Configuration obtained by a sampling strategy based on preset probabilities and operational conditions.
        """
        if np.random.uniform(0, 1) < self.config.p_goal:
            return self._sample_goal(
                mode, self.transition_node_ids, self.trees[mode].order
            )

        if self.config.informed_sampling and self.operation.init_sol:
            return self.sample_informed(mode)

        sampler = self._resolve_sampler(mode)
        if sampler == "gibbs":
            q, n_checks = self._sample_collision_free_gibbs(mode)
            if self.config.sampler == "auto":
                stats = self._gibbs_sampling_stats.setdefault(mode, [0, 0])
                stats[0] += n_checks
                stats[1] += q is not None
            return q
        elif sampler == "per_robot":
            return self._sample_collision_free_per_robot(mode)
        return self._sample_uniform(mode)

    def _timed_sample_configuration(self, mode: Mode) -> Configuration | None:
        _t0 = time.perf_counter()
        result = self.sample_configuration(mode)
        self._sampling_time += time.perf_counter() - _t0
        return result

    def _timed_sample_transition_configuration(self, mode: Mode) -> Configuration | None:
        _t0 = time.perf_counter()
        result = self.sample_transition_configuration(mode)
        self._sampling_time += time.perf_counter() - _t0
        return result

    def find_lb_transition_node(self, shortcutting_bool: bool = True) -> None:
        """
        Searches lower-bound transition node and generates its corresponding path if a valid candidate is found.

        Args:
            shortcutting_bool (bool): Flag to either apply shortcutting or not

        Returns:
            None: This method does not return any value.
        """

        if (
            self.operation.init_sol
            or self.config.apply_long_horizon
            and self.long_horizon.reached_horizon
        ):
            modes = self.get_termination_modes()
            if (
                self.config.apply_long_horizon
                and self.long_horizon.reached_horizon
                and not self.operation.init_sol
            ):
                modes = [self.long_horizon.terminal_mode]
            result, mode = self.get_lb_transition_node_id(modes)
            if not result:
                return
            valid_mask = result[0] < self.operation.cost
            if valid_mask.any():
                lb_transition_node = self.get_transition_node(mode, result[1])
                self.generate_path(
                    mode, lb_transition_node, shortcutting_bool=shortcutting_bool
                )

    def tree_extension(self, discretized_path: List[State]) -> None:
        """
        Extends the tree by adding path states as nodes and updating parent-child relationships.

        Args:
            active_mode (Mode): Current mode for tree extension.
            discretized_path (List[State]):Sequence of states forming the discretized path.

        Returns:
            None: This method does not return any value.
        """
        mode = discretized_path[0].mode
        parent = self.operation.path_nodes[0]
        for i in range(1, len(discretized_path)):
            state = discretized_path[i]

            node = Node(state, self.operation)
            # node.parent = parent
            # self.operation.costs = self.trees[discretized_path[i].mode].ensure_capacity(self.operation.costs, node.id)
            # node.cost = discretized_costs[i]
            # node.cost_to_parent = node.cost - node.parent.cost
            # parent.children.append(node)

            if mode == node.state.mode:
                index = np.where(
                    self.trees[node.state.mode].get_node_ids_subtree() == parent.id
                )
                N_near_batch, n_near_costs, node_indices = self.near(
                    node.state.mode, node, index
                )
                batch_cost = self.env.batch_config_cost(node.state.q, N_near_batch)
                self.find_parent(
                    node.state.mode,
                    node_indices,
                    node,
                    parent,
                    batch_cost,
                    n_near_costs,
                )
            else:
                node.parent = parent
                self.operation.costs = self.trees[
                    discretized_path[i].mode
                ].ensure_capacity(self.operation.costs, node.id)
                node.cost_to_parent = self.env.config_cost(node.state.q, parent.state.q)
                node.cost = node.parent.cost + node.cost_to_parent
                parent.children.append(node)
                self.convert_node_to_transition_node(
                    node.parent.state.mode, node.parent
                )
                index = np.where(
                    self.trees[node.state.mode].get_node_ids_subtree() == parent.id
                )
                N_near_batch, n_near_costs, node_indices = self.near(
                    node.state.mode, node, index
                )
                batch_cost = self.env.batch_config_cost(node.state.q, N_near_batch)

            if self.rewire(
                node.state.mode, node_indices, node, batch_cost, n_near_costs
            ):
                self.update_cost(node.state.mode, node)
            if self.trees[discretized_path[i].mode].order == 1:
                self.trees[discretized_path[i].mode].add_node(node)
            else:
                self.trees[discretized_path[i].mode].add_node(node, "B")
            parent = node

            mode = node.state.mode

        self.convert_node_to_transition_node(mode, node)
        self.find_lb_transition_node(shortcutting_bool=False)
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(3, 2, figsize=(12, 15))

        # samples = [sample.q.state()[3:5] for sample in discretized_path]
        # axes[1, 0].scatter([s[0] for s in samples], [s[1] for s in samples])
        # axes[1, 0].set_title('Agent 2')

        # samples = [sample.q.state()[:2] for sample in discretized_path]
        # axes[0, 0].scatter([s[0] for s in samples], [s[1] for s in samples])
        # axes[0, 0].set_title('Agent 1')

        # # samples = [sample.q.state()[5:] for sample in discretized_path]
        # # axes[2, 0].scatter([s[0] for s in samples], [s[1] for s in samples])
        # # axes[2, 0].set_title('Agent 3')

        # samples = [sample.q.state()[3:5] for sample in self.operation.path]
        # axes[1, 1].scatter([s[0] for s in samples], [s[1] for s in samples])
        # axes[1, 1].set_title('Agent 2')

        # samples = [sample.q.state()[:2] for sample in self.operation.path]
        # axes[0, 1].scatter([s[0] for s in samples], [s[1] for s in samples])
        # axes[0, 1].set_title('Agent 1')

        # # samples = [sample.q.state()[5:] for sample in self.operation.path]
        # # axes[2, 1].scatter([s[0] for s in samples], [s[1] for s in samples])
        # # axes[2, 1].set_title('Agent 3')

        # # Adjust layout and display the plots
        # plt.tight_layout()
        # plt.show()
        print("final new cost:", self.operation.cost)

        # self.GeneratePath(active_mode, node, shortcutting_bool=False)

    @abstractmethod
    def update_cost(self, mode: Mode, n: Node) -> None:
        """
        Updates cost for a given node and all its descendants by propagating the cost change down the tree.

        Args:
            mode (Mode): The current operational mode.
            n (Node): Root node from which the cost update begins.

        Returns:
            None: This method does not return any value.
        """
        pass

    @abstractmethod
    def initialize_planner(self) -> None:
        """
        Initializes planner by setting parameters, creating the initial mode, and adding start node.

        Args:
            None

        Returns:
            None: None: This method does not return any value.
        """
        pass

    @abstractmethod
    def manage_transition(self, mode: Mode, n_new: Node) -> None:
        """
        Checks if new node qualifies as a transition or termination node.
        If it does, node is converted into transition node, mode is updated accordingly,
        and a new path is generated if node is the lower-bound transition node.

        Args:
            mode (Mode): The current operational mode.
            n_new (Node): The newly added node to evaluate for triggering a transition or termination.

        Returns:
            None: This method does not return any value.
        """
        pass

    @abstractmethod
    def save_tree_data(self) -> None:
        pass
