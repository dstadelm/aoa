from collections.abc import Generator, Iterable, Iterator, MutableMapping, Set
from dataclasses import dataclass
from typing import override

from aoa.model.activity import Activity

from .exception import AoAException
from .id_generator import id_generator
from .node import Node


@dataclass
class ActivityNodes:
    """The start and end nodes for an activity in the network."""

    start_node: Node | None
    end_node: Node | None


class NodeDict(MutableMapping[Set[int], Node]):
    """A dict-like mapping from a set of predecessor activity IDs to a Node.

    Accepts any ``AbstractSet[int]`` as a key; internally canonicalizes to
    ``frozenset[int]``.  Maintains several internal lookup tables that are
    updated explicitly through :meth:`attach_activity` and
    :meth:`move_activity_to_node`.

    Internal state maintained:
        ``_store``              – The primary mapping of frozenset keys → Nodes.
        ``_node_id2key``        – Maps ``node.id`` → frozenset key for reverse lookup.
        ``_activity_node_lut``  – Maps ``activity.id`` → :class:`ActivityNodes`.
        ``_end_nodes``          – List of node IDs that have no outbound activities.
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
        self._node_id2key: dict[int, frozenset[int]] = {}
        self._activity_node_lut: dict[int, ActivityNodes] = {}
        self._store: dict[frozenset[int], Node] = {}
        self._end_nodes: list[int] = []
        if items is not None:
            for k, v in items:
                self[k] = v

    # --- Internal helpers ---

    def _norm_key(self, key: Set[int]) -> frozenset[int]:
        """Normalize any Set[int] to a frozenset[int]."""
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
        if del_key := next((k for k, v in self.items() if v.id == value.id), None):
            del self[del_key]
        self._store[self._norm_key(key)] = value

    @override
    def __delitem__(self, key: Set[int]) -> None:
        del self._store[self._norm_key(key)]

    @override
    def __iter__(self) -> Iterator[frozenset[int]]:
        return iter(self._store)

    @override
    def __len__(self) -> int:
        return len(self._store)

    @override
    def __contains__(self, key: object) -> bool:
        try:
            k = self._norm_key(key)  # pyright: ignore [reportArgumentType]
        except TypeError:
            return False
        return k in self._store

    @override
    def __repr__(self) -> str:
        items = ", ".join(f"{k}: {v!r}" for k, v in self._store.items())
        return f"{self.__class__.__name__}({{{items}}})"

    @override
    def __str__(self) -> str:
        return self.__repr__()

    # --- Node creation ---

    @property
    def start_node(self) -> Node:
        """Get the start node (the node with an empty set of dependencies)."""
        if set() not in self:
            _ = self.new_node()
        return self[set()]

    def new_node(self) -> Node:
        """Create a new Node with a unique ID from the generator."""
        node = Node(_id=next(self._node_id_generator))
        self._end_nodes.append(node.id)
        if node.id == 0:
            self.__setitem__(set(), node)
        return node

    # --- Lookup ---

    def nodes_of(self, activity_id: int) -> ActivityNodes:
        """Return the start and end nodes for a given activity.

        Arguments:
            activity_id: The ID of the activity to look up.

        Returns:
            ActivityNodes: The start and end node for the given activity.
        """
        return self._activity_node_lut[activity_id]

    # --- Explicit state mutation methods ---

    def attach_activity(self, activity: Activity, start_node: Node, end_node: Node) -> None:
        """Attach an activity between a start and end node, updating all internal state.

        This is the primary method for adding activities to the network. It:
        1. Adds the activity to the start node's outbound list.
        2. Adds the activity to the end node's inbound list.
        3. Updates the activity-to-node lookup table.
        4. Updates the store key for the end node.
        5. Removes the start node from the end-nodes list.
        6. Updates the end node's max depth.

        Arguments:
            activity: The activity to attach.
            start_node: The node the activity departs from.
            end_node: The node the activity arrives at.
        """
        # Add to start node's outbound list
        start_node.outbound_activities.append(activity)

        # Update activity LUT with start node
        if activity.id in self._activity_node_lut:
            self._activity_node_lut[activity.id].start_node = start_node
        else:
            self._activity_node_lut[activity.id] = ActivityNodes(start_node=start_node, end_node=None)

        # Start node now has outbound activities — remove from end nodes
        if start_node.id in self._end_nodes:
            self._end_nodes.remove(start_node.id)

        # Add to end node's inbound list
        end_node.inbound_activities.append(activity)

        # Update store key for end node
        self._update_node_key(end_node)

        # Update activity LUT with end node
        self._activity_node_lut[activity.id].end_node = end_node

        # Update end node's max depth
        if start_node.max_depth >= end_node.max_depth:
            end_node.max_depth = start_node.max_depth + 1

    def move_activity_to_node(self, activity: Activity, target_node: Node) -> None:
        """Move an activity's end point from its current node to the target node.

        Removes the activity from its current end node's inbound list, cleans
        up the old node if it becomes orphaned, and adds the activity to the
        target node's inbound list.

        Arguments:
            activity: The activity to move.
            target_node: The node to move the activity's end point to.
        """
        # Remove from current end node
        current_end = self._activity_node_lut[activity.id].end_node
        if current_end:
            current_end.inbound_activities.remove(activity)
            self._handle_inbound_removal(current_end)

        # Add to target node's inbound list
        target_node.inbound_activities.append(activity)

        # Update store key for target node
        self._update_node_key(target_node)

        # Update activity LUT with new end node
        self._activity_node_lut[activity.id].end_node = target_node

        # Update target node max depth
        start_node = self._activity_node_lut[activity.id].start_node
        if start_node and start_node.max_depth >= target_node.max_depth:
            target_node.max_depth = start_node.max_depth + 1

    def renumber_nodes(self) -> None:
        """Renumber nodes to have consecutive IDs based on topological depth.

        During network generation, some nodes may be merged, leaving gaps in
        the numbering. This function reassigns sequential IDs and rebuilds the
        internal ``_node_id2key`` and ``_end_nodes`` lookup tables.

        Raises:
            AoAException: If there are zero or multiple end nodes.
        """
        if len(self._end_nodes) > 1:
            end_node_ids = [node.id for node in self.end_nodes]
            raise AoAException(f"Undefined end_node, multiple end nodes detected {end_node_ids}")
        if len(self._end_nodes) == 0:
            raise AoAException("Undefined end_node, no end node defined")

        sorted_nodes = sorted(self.values(), key=lambda n: n.max_depth)

        # Rebuild _node_id2key and _end_nodes with new IDs
        new_node_id2key: dict[int, frozenset[int]] = {}
        new_end_nodes: list[int] = []

        for index, node in enumerate(sorted_nodes):
            old_id = node.id

            # Transfer key mapping
            if old_id in self._node_id2key:
                new_node_id2key[index] = self._node_id2key[old_id]

            # Transfer end node status
            if old_id in self._end_nodes:
                new_end_nodes.append(index)

            # Assign new ID
            node.id = index

        self._node_id2key = new_node_id2key
        self._end_nodes = new_end_nodes

    # --- Ancestor check ---

    def have_common_ancestor(self, left_node_id: int, right_node_id: int) -> bool:
        """Check if two nodes share a common ancestor.

        Two nodes have a common ancestor if any of their inbound activities
        originate from the same start node.

        Arguments:
            left_node_id: The ID of the first node.
            right_node_id: The ID of the second node.

        Returns:
            bool: True if the nodes share a common ancestor.
        """
        ids_left = {
            self.nodes_of(activity.id).start_node.id  # pyright: ignore [reportOptionalMemberAccess]
            for activity in self[self._node_id2key[left_node_id]].inbound_activities
            if self.nodes_of(activity.id).start_node
        }
        ids_right = {
            self.nodes_of(activity.id).start_node.id  # pyright: ignore [reportOptionalMemberAccess]
            for activity in self[self._node_id2key[right_node_id]].inbound_activities
            if self.nodes_of(activity.id).start_node
        }
        return bool(ids_left.intersection(ids_right))

    # --- Properties ---

    @property
    def end_nodes(self) -> list[Node]:
        """Return all nodes that have no outbound activities."""
        return [self[self._node_id2key[i]] for i in self._end_nodes]

    # --- Private helpers ---

    def _update_node_key(self, node: Node) -> None:
        """Update the store key for a node based on its current start_dependencies.

        Only updates if the new key does not already exist in the store
        (to avoid conflicts with temporary floating nodes during construction).
        """
        new_key = frozenset(node.start_dependencies)
        if new_key not in self:
            if node.id in self._node_id2key:
                old_key = self._node_id2key[node.id]
                del self[old_key]
            self._node_id2key[node.id] = new_key
            self[new_key] = node

    def _handle_inbound_removal(self, node: Node) -> None:
        """Update internal state after an activity has been removed from a node's inbound list.

        If the node has no remaining inbound activities (start_dependencies is
        empty), it is removed from the store and all lookup tables.  Otherwise,
        its store key and max depth are updated.
        """
        new_key = frozenset(node.start_dependencies)

        # Remove old key from store
        if node.id in self._node_id2key:
            old_key = self._node_id2key[node.id]
            del self[old_key]

        if new_key == frozenset():
            # Node is orphaned — remove from all lookups
            if node.id in self._node_id2key:
                del self._node_id2key[node.id]
            if node.id in self._end_nodes:
                self._end_nodes.remove(node.id)
        else:
            # Update with new key
            self._node_id2key[node.id] = new_key
            self[new_key] = node

        # Recalculate max depth from remaining inbound activities
        self._recalculate_max_depth(node)

    def _recalculate_max_depth(self, node: Node) -> None:
        """Recalculate a node's max_depth from its inbound activities' start nodes."""
        node.max_depth = 0
        for act in node.inbound_activities:
            if act.id in self._activity_node_lut:
                activity_nodes = self._activity_node_lut[act.id]
                if activity_nodes.start_node and activity_nodes.start_node.max_depth >= node.max_depth:
                    node.max_depth = activity_nodes.start_node.max_depth + 1
