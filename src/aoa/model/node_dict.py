from collections.abc import Generator, Iterable, Iterator, MutableMapping, Set
from dataclasses import dataclass
from typing import Callable, override

from aoa.model.activity import Activity

from .id_generator import id_generator
from .node import Node


@dataclass
class ActivityNodes:
    start_node: Node | None
    end_node: Node | None


class NodeDict(MutableMapping[Set[int], Node]):
    """
    A dict-like mapping from a set of ints to a Node.
    - Accepts any AbstractSet[int] as a key; internally canonicalizes to frozenset[int].
    - Iteration yields frozenset[int] keys.
    """

    __slots__ = (  # pyright: ignore [reportUnannotatedClassAttribute]
        "_store",
        "_node_id2key",
        "_node_id_generator",
        "_activity_node_lut",
        "_end_nodes",
    )

    def __init__(
        self,
        items: Iterable[tuple[Set[int], Node]] | None = None,
    ) -> None:
        self._node_id_generator: Generator[int, None, None] = id_generator(start=-1, increment=1)
        self._node_id2key: dict[int, frozenset[int]] = {}  # Maps node IDs to Node instances for quick lookup
        self._activity_node_lut: dict[int, ActivityNodes] = {}  # Maps activity IDs to Nodes for quick lookup
        self._store: dict[frozenset[int], Node] = {}
        self._end_nodes: list[int] = []
        if items is not None:
            for k, v in items:
                self[k] = v  # go through __setitem__ to validate and normalize

    # --- Internal helpers ---

    def _norm_key(self, key: Set[int]) -> frozenset[int]:
        """Normalize any Set[int] to a frozenset[int] and validate element types."""
        try:
            fs = frozenset(key)
        except TypeError as e:
            raise TypeError("Key must be an abstract set of ints") from e
        return fs

    # --- MutableMapping interface ---
    @override
    def __getitem__(self, key: Set[int]) -> Node:
        return self._store[self._norm_key(key)]

    @override
    def __setitem__(self, key: Set[int], value: Node) -> None:
        # to make sure there are no duplicates in the dict, find entry where the node's id matches and delete it
        if del_key := next((k for k, v in self.items() if v.id == value.id), None):
            del self[del_key]
        self._store[self._norm_key(key)] = value

    @override
    def __delitem__(self, key: Set[int]) -> None:
        del self._store[self._norm_key(key)]

    @override
    def __iter__(self) -> Iterator[frozenset[int]]:
        # Iterates over normalized keys (frozenset[int])
        return iter(self._store)

    @override
    def __len__(self) -> int:
        return len(self._store)

    # Optional: faster membership test
    @override
    def __contains__(self, key: object) -> bool:
        try:
            k = self._norm_key(key)  # pyright: ignore [reportArgumentType]
        except TypeError:
            return False
        return k in self._store

    # Optional: nicer representation
    @override
    def __repr__(self) -> str:
        # Build a dict-like repr but keys are frozenset[int]
        items = ", ".join(f"{k}: {v!r}" for k, v in self._store.items())
        return f"{self.__class__.__name__}({{{items}}})"

    @override
    def __str__(self) -> str:
        return self.__repr__()

    @property
    def start_node(self) -> Node:
        """Get the start node (the node with an empty set of dependencies)."""
        if set() not in self:
            _ = self.new_node()  # Ensure the start node exists
        return self[set()]

    def new_node(self) -> Node:
        """Create a new Node with a unique ID from the generator."""
        node = Node(id=next(self._node_id_generator))

        node.register_inbound_activity_append_callback(self.update_on_inbound_append_change_closure(node))
        node.register_inbound_activity_remove_callback(self.update_on_inbound_remove_change_closure(node))
        node.register_outbound_activity_append_callback(self.update_on_outbound_append_change_closure(node))
        node.register_outbound_activity_remove_callback(self.update_on_outbound_remove_change_closure(node))

        self._end_nodes.append(node.id)
        if node.id == 0:
            self.__setitem__(set(), node)  # Add the initial node with an empty set key

        return node

    def nodes_of(self, activity_id: int) -> ActivityNodes:
        return self._activity_node_lut[activity_id]

    def update_on_inbound_append_change_closure(self, node: Node) -> Callable[[Activity], None]:
        def update_on_inbound_append_change(activity: Activity) -> None:
            # print(f"inbound change received signal for node ID {node.id}")
            new_key = frozenset(node.start_dependencies)
            # when creating floating nodes we can create temporary keys that have the same set of dependencies as an
            # existing node, but we only want to update the dict when the key is unique
            if new_key not in self:
                if node.id in self._node_id2key:
                    old_key = self._node_id2key[node.id]
                    del self[old_key]  # Remove the old entry
                self._node_id2key[node.id] = new_key  # Update the mapping from
                self[new_key] = node  # Add the new entry
            if activity.id in self._activity_node_lut:
                activity_nodes = self._activity_node_lut[activity.id]
                if activity_nodes.end_node:
                    activity_nodes.end_node.inbound_activities.remove(activity)  # Remove the activity from the old end
                activity_nodes.end_node = node
            else:
                activity_nodes = ActivityNodes(start_node=None, end_node=node)
            self._activity_node_lut[activity.id] = activity_nodes  # Update the LUT

        return update_on_inbound_append_change

    def update_on_inbound_remove_change_closure(self, node: Node) -> Callable[[Activity], None]:
        def update_on_inbound_remove(activity: Activity) -> None:
            # update the dict entry
            new_key = frozenset(node.start_dependencies)
            if node.id in self._node_id2key:
                old_key = self._node_id2key[node.id]
                del self[old_key]  # Remove the old entry
            if new_key == frozenset({}):
                del self._node_id2key[node.id]
                if node.id in self._end_nodes:
                    self._end_nodes.remove(node.id)
            else:
                self._node_id2key[node.id] = new_key  # Update the mapping from
                self[new_key] = node  # Add the new entry
            # remove the end node from the activity
            if activity.id in self._activity_node_lut:
                activity_nodes = self._activity_node_lut[activity.id]
                if activity_nodes.end_node and node.id == activity_nodes.end_node.id:
                    activity_nodes.end_node = None
                    self._activity_node_lut[activity.id] = activity_nodes  # Update the LUT

        return update_on_inbound_remove

    def update_on_outbound_append_change_closure(self, node: Node) -> Callable[[Activity], None]:
        def update_on_outbound_append_change(activity: Activity) -> None:
            """"""

            if activity.id in self._activity_node_lut:
                activity_nodes = self._activity_node_lut[activity.id]
                activity_nodes.start_node = node
            else:
                activity_nodes = ActivityNodes(start_node=node, end_node=None)
            activity_nodes.start_node = node
            self._activity_node_lut[activity.id] = activity_nodes  # Update the LUT
            if node.id in self._end_nodes:
                self._end_nodes.remove(node.id)

        return update_on_outbound_append_change

    def update_on_outbound_remove_change_closure(self, node: Node) -> Callable[[Activity], None]:
        def update_on_outbound_remove_change(activity: Activity) -> None:
            # remove the start node
            if activity.id in self._activity_node_lut:
                activity_nodes = self._activity_node_lut[activity.id]
                if activity_nodes.start_node and activity_nodes.start_node.id == node.id:
                    activity_nodes.start_node = None
                self._activity_node_lut[activity.id] = activity_nodes  # Update the LUT
            # if there are no outbound activities left on the node add it to the end nodes list
            if node.id in self._end_nodes:
                if not node.outbound_activities:
                    self._end_nodes.append(node.id)

        return update_on_outbound_remove_change

    @property
    def end_nodes(self) -> list[Node]:
        return [self[self._node_id2key[i]] for i in self._end_nodes]
