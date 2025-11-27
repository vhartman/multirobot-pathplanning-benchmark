import heapq
from typing import (
    Any,
    Dict,
    List,
    Tuple,
)

from sortedcontainers import SortedList
from collections import defaultdict

class HeapQueue:
    __slots__ = ["queue"]

    # Class attribute type hints
    queue: List[Any]

    def __init__(self) -> None:
        self.queue = []
        heapq.heapify(self.queue)

    def __len__(self) -> int:
        return len(self.queue)

    def heappush(self, item: Any) -> None:
        heapq.heappush(self.queue, item)

    def heappop(self) -> Any:
        # if len(self.queue) == 0:
        #     raise IndexError("pop from an empty heap")
        return heapq.heappop(self.queue)

    def remove(self, node: Any) -> None:
        """
        Removes a node from the heap.
        Note: This is a placeholder implementation.
        """
        # (cost, edge_cost, e)
        # To implement removal, you would need to:
        # 1. Find the node in the heap.
        # 2. Mark it as removed (e.g., using a flag or a separate set).
        # 3. Re-heapify the queue if necessary.
        pass


class SortedQueue:
    """
    Edge queue implementation using SortedList for efficient removal operations.
    """

    def __init__(self):
        # SortedList maintains elements in sorted order
        # Elements are tuples: (cost, edge_cost, (node1, node2))
        self.queue = SortedList()

    def heappush(self, item):
        """Add new edge to queue"""
        self.queue.add(item)

    def heappop(self):
        """Remove and return lowest cost edge"""
        if self.queue:
            return self.queue.pop(0)
        return None

    # def remove_by_node(self, node):
    #     """Remove all edges where node2 == node"""
    #     # Create list of indices to remove (in reverse order)
    #     to_remove = [
    #         i for i, (_, _, (_, node2)) in enumerate(self.queue)
    #         if node2 == node
    #     ]

    #     # Remove from highest index to lowest to maintain valid indices
    #     for idx in reversed(to_remove):
    #         del self.queue[idx]

    def remove_by_node(self, node):
        """Remove all edges where node2 == node"""
        i = 0
        while i < len(self.queue):
            _, _, (_, node2) = self.queue[i]
            if node2 == node:
                del self.queue[
                    i
                ]  # Deleting an item shifts elements left, so don't increment i
            else:
                i += 1  # Only increment if an element was not removed

    def __len__(self):
        return len(self.queue)


class EfficientEdgeQueue:
    """
    Edge queue using a min-heap for efficient pops and a dictionary for fast removals.
    """

    def __init__(self):
        # Min-heap of (cost, edge_cost, (node1, node2))
        self.heap = []
        # Dictionary mapping node2 to a set of edges for quick removal
        self.edges_by_node = defaultdict(set)

    def heappush(self, item):
        """Add a new edge to the queue."""
        cost, edge_cost, nodes = item

        heapq.heappush(self.heap, item)
        self.edges_by_node[nodes[1]].add(item)

    def heappop(self):
        """Remove and return the lowest cost edge."""
        while self.heap:
            item = heapq.heappop(self.heap)
            nodes = item[2]
            if item in self.edges_by_node[nodes[1]]:  # Ensure the edge is still valid
                self.edges_by_node[nodes[1]].remove(item)
                return item

        return None

    def remove_by_node(self, node):
        """Remove all edges where node2 == node."""
        if node in self.edges_by_node:
            for item in self.edges_by_node[node]:
                # Lazy deletion: mark edge as removed by excluding it from the dict
                self.heap.remove(item)  # O(n), but happens rarely
            heapq.heapify(self.heap)  # Restore heap property, O(n)
            del self.edges_by_node[node]  # Remove the entry

    def __len__(self):
        return len(self.heap)


class BucketHeapQueue:
    def __init__(self):
        self.queues = {}
        self.priority_lookup = []

        self.len = 0

    def __len__(self):
        return self.len

    def heappush(self, item):
        self.len += 1
        priority = int(item[0] * 10000)

        if priority not in self.queues:
            self.queues[priority] = []
            heapq.heappush(self.priority_lookup, priority)

        heapq.heappush(self.queues[priority], item)

    def heappop(self):
        self.len -= 1

        min_priority = self.priority_lookup[0]
        value = heapq.heappop(self.queues[min_priority])

        if not self.queues[min_priority]:
            del self.queues[min_priority]
            heapq.heappop(self.priority_lookup)

        return value


class IndexHeap:
    __slots__ = ["queue", "items"]

    # Class attribute type hints
    queue: List[Tuple[float, int]]  # (priority, index)
    items: List[Any]  # The actual items

    def __init__(self) -> None:
        self.queue = []
        self.items = []
        heapq.heapify(self.queue)

    def __len__(self) -> int:
        return len(self.queue)

    def heappush_list(self, items: List[Tuple[float, Any]]) -> None:
        for item in items:
            idx = len(self.items)
            self.items.append(item)
            self.queue.append((item[0], idx))

        heapq.heapify(self.queue)

    def heappush(self, item: Tuple[float, Any]) -> None:
        idx = len(self.items)
        self.items.append(item)
        heapq.heappush(self.queue, (item[0], idx))

    def heappop(self) -> Any:
        # if len(self.queue) == 0:
        #     raise IndexError("pop from an empty heap")

        _, idx = heapq.heappop(self.queue)
        return self.items[idx]


class DictIndexHeap:
    __slots__ = ["queue", "items"]

    queue: List[Tuple[float, int]]  # (priority, index)
    items: Dict[int, Any]  # Dictionary for storing active items

    idx = 0

    def __init__(self) -> None:
        self.queue = []
        self.items = {}
        heapq.heapify(self.queue)

    def __len__(self) -> int:
        return len(self.queue)

    def __bool__(self):
        return bool(self.queue)

    # def heappush_list(self, items: List[Tuple[float, Any]]) -> None:
    #     """Push a list of items into the heap."""
    #     for priority, value in items:
    #         idx = len(self.items)
    #         self.items[idx] = value  # Store only valid items
    #         self.queue.append((priority, idx))

    #     heapq.heapify(self.queue)

    def heappush(self, item: Tuple[float, Any, Any]) -> None:
        """Push a single item into the heap."""
        # idx = len(self.items)
        self.items[DictIndexHeap.idx] = item  # Store only valid items
        heapq.heappush(self.queue, (item[0], DictIndexHeap.idx))
        DictIndexHeap.idx += 1

    def heappop(self) -> Any:
        """Pop the item with the smallest priority from the heap."""
        if not self.queue:
            raise IndexError("pop from an empty heap")

        _, idx = heapq.heappop(self.queue)
        value = self.items.pop(idx)  # Remove from dictionary
        return value


class BucketIndexHeap:
    __slots__ = ["granularity", "queues", "priority_lookup", "items", "len"]

    # Class attribute type hints
    granularity: int
    queues: Dict[int, List[Tuple[float, int]]]
    priority_lookup: List[int]
    items: List[Any]
    len: int

    def __init__(self, granularity: int = 100) -> None:
        self.granularity = granularity
        self.len = 0

        self.queues = {}
        self.priority_lookup = []
        self.items = []

    def __len__(self) -> int:
        return self.len

    # @profile # run with kernprof -l examples/run_planner.py [your environment] [your flags]
    def heappush(self, item: Tuple[float, Any]) -> None:
        self.len += 1
        priority: int = int(item[0] * self.granularity)

        idx: int = len(self.items)
        self.items.append(item)

        if priority not in self.queues:
            self.queues[priority] = []
            heapq.heappush(self.priority_lookup, priority)

        heapq.heappush(self.queues[priority], (item[0], idx))

    # def heappush_list(self, items: List[Tuple[float, Any]]) -> None:
    #     for item in items:
    #         self.heappush(item)

    def heappop(self) -> Any:
        # I do not want the possible performance penalty
        # if self.len == 0:
        #     raise IndexError("pop from an empty heap")

        self.len -= 1
        min_priority: int = self.priority_lookup[0]
        _, idx = heapq.heappop(self.queues[min_priority])

        if not self.queues[min_priority]:
            del self.queues[min_priority]
            heapq.heappop(self.priority_lookup)

        value: Any = self.items[idx]
        return value


class DiscreteBucketIndexHeap:
    __slots__ = ["granularity", "queues", "priority_lookup", "items", "len"]

    # Class attribute type hints
    granularity: int
    queues: Dict[int, List[Tuple[float, int]]]
    priority_lookup: List[int]
    items: List[Any]
    len: int

    def __init__(self, granularity: int = 1000) -> None:
        self.granularity = granularity
        self.queues = {}
        self.priority_lookup = []
        self.items = []
        self.len = 0

    def __len__(self) -> int:
        return self.len

    def heappush(self, item: Tuple[float, Any]) -> None:
        self.len += 1
        priority: int = int(item[0] * self.granularity)

        idx: int = len(self.items)
        self.items.append(item)

        if priority not in self.queues:
            self.queues[priority] = []
            heapq.heappush(self.priority_lookup, priority)

        self.queues[priority].append((item[0], idx))

    def heappop(self) -> Any:
        if self.len == 0:
            raise IndexError("pop from an empty heap")

        self.len -= 1

        min_priority: int = self.priority_lookup[0]
        _, idx = self.queues[min_priority].pop()

        if not self.queues[min_priority]:
            del self.queues[min_priority]
            heapq.heappop(self.priority_lookup)

        value: Any = self.items[idx]
        return value