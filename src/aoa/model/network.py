"""Network creation module for Activity-on-Arrow (AoA) diagrams.

This module implements the core algorithm for building an AoA network from a
collection of activities. The process follows four stages:

1. **Allocation sequencing** — Topologically sort activities so each activity's
   predecessors are processed first.
2. **Activity allocation** — For each activity, find or create appropriate start
   and end nodes, inserting dummy activities where necessary to maintain correct
   dependency semantics.
3. **End-node tying** — Merge all leaf (end) nodes into a single terminal node.
4. **Node renumbering** — Reassign sequential node IDs based on topological depth.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field

from more_itertools import powerset

from .activity import Activity, ActivityCollection
from .exception import AllocationException, AoAException
from .node import Node
from .node_dict import ActivityNodes, NodeDict


@dataclass
class Network:
    """Represents an Activity-on-Arrow network.

    Attributes:
        activities: The collection of activities (edges) in the network.
        node_dict: The mapping of predecessor sets to nodes (events).
    """

    activities: ActivityCollection
    node_dict: NodeDict = field(default_factory=NodeDict, compare=False)

    def get_node_list_sorted_by_depth(self) -> list[Node]:
        """Return all nodes sorted by depth (ascending node ID order).

        Returns:
            list[Node]: All nodes sorted by depth.
        """
        return sorted(self.node_dict.values(), key=lambda x: x.id)

    def get_activity_nodes(self, activity: Activity) -> ActivityNodes:
        """Return the start and end nodes for a given activity.

        Arguments:
            activity: The activity to look up.

        Returns:
            ActivityNodes: The start and end node for the given activity.
        """
        return self.node_dict.nodes_of(activity.id)

    def get_activity_start_node(self, activity: Activity) -> Node:
        """Return the start node for a given activity."""
        nodes = self.node_dict.nodes_of(activity.id)
        assert nodes.start_node is not None
        return nodes.start_node

    def get_activity_end_node(self, activity: Activity) -> Node:
        """Return the end node for a given activity."""
        nodes = self.node_dict.nodes_of(activity.id)
        assert nodes.end_node is not None
        return nodes.end_node


def create_network(activities: ActivityCollection) -> Network:
    """Factory method to create a network from a collection of activities.

    The network is built in four stages:
    1. Determine allocation sequence (topological sort).
    2. Allocate each activity to start/end nodes.
    3. Tie all leaf nodes into a single end node.
    4. Renumber nodes sequentially by depth.

    Arguments:
        activities: The collection of activities to create the network from.

    Returns:
        Network: The created network.
    """
    network = Network(activities)

    sorted_activities = _get_allocation_sequence(copy(network.activities))
    for activity in sorted_activities:
        _allocate_activity(activity, network)
    _tie_end_node(network)
    _renumber_nodes(network.node_dict)

    return network


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _sorted_power_set(predecessors: list[int]) -> list[set[int]]:
    """Return all subsets of the given list, sorted from largest to smallest.

    Arguments:
        predecessors: The list of predecessor IDs.

    Returns:
        list[set[int]]: All possible subsets, largest first.

    Example:
        >>> _sorted_power_set([1, 2, 3])
        [{1, 2, 3}, {1, 2}, {1, 3}, {2, 3}, {1}, {2}, {3}, set()]
    """
    power_sets = [set(x) for x in list(powerset(predecessors))]
    return sorted(power_sets, key=lambda x: len(x), reverse=True)


def _get_union(list_of_sets: list[set[int]]) -> set[int]:
    """Return the union of all sets in the list, or an empty set if the list is empty.

    Arguments:
        list_of_sets: The list of sets to union.

    Returns:
        set[int]: The union of all sets.
    """
    if list_of_sets:
        return set.union(*list_of_sets)  # pyright: ignore [reportUnknownMemberType, reportUnknownVariableType]
    return set()


# ---------------------------------------------------------------------------
# Allocation sequence
# ---------------------------------------------------------------------------


def _get_allocation_sequence(activities: ActivityCollection) -> list[Activity]:
    """Determine the order in which activities must be allocated.

    Activities are returned in topological order where each activity's
    predecessors always appear before the activity itself. Uses an iterative
    approach to avoid stack depth issues with large activity sets.

    Arguments:
        activities: The collection of activities to sequence.

    Returns:
        list[Activity]: The activities sorted into a valid allocation order.

    Raises:
        AllocationException: If dependencies cannot be resolved (e.g. cycles).
    """
    result: list[Activity] = []
    allocated_ids: set[int] = set()
    remaining = dict(activities)

    while remaining:
        allocated_in_pass = False
        for idx, activity in list(remaining.items()):
            if activity.predecessors.issubset(allocated_ids):
                result.append(activity)
                allocated_ids.add(activity.id)
                del remaining[idx]
                allocated_in_pass = True
                break

        if not allocated_in_pass:
            unresolved = [a.id for a in remaining.values()]
            raise AllocationException(
                f"Activities {unresolved}, can't be allocated as their dependencies can't be resolved"
            )

    return result


# ---------------------------------------------------------------------------
# Node renumbering
# ---------------------------------------------------------------------------


def _renumber_nodes(node_dict: NodeDict) -> None:
    """Renumber nodes to have consecutive IDs based on topological depth.

    Delegates to :meth:`NodeDict.renumber_nodes` which reassigns sequential
    IDs and rebuilds internal lookup tables.

    Raises:
        AoAException: If there are zero or multiple end nodes.
    """
    node_dict.renumber_nodes()


# ---------------------------------------------------------------------------
# End-node tying
# ---------------------------------------------------------------------------


def _tie_end_node(network: Network) -> None:
    """Tie all leaf nodes into a single end node.

    The allocation algorithm leaves end nodes as-is. This function merges them
    into one common end node. Nodes sharing a common ancestor with the chosen
    end node are linked via dummy activities; others have their inbound
    activities moved directly.

    Arguments:
        network: The network being constructed.

    Raises:
        AoAException: If there are no end nodes to tie.
    """
    node_dict = network.node_dict

    if not node_dict.end_nodes:
        raise AoAException("No end nodes available to tie.")

    tie_node = max(node_dict.end_nodes, key=lambda n: n.max_depth)

    for node in node_dict.end_nodes:
        if node.id == tie_node.id:
            continue
        if node_dict.have_common_ancestor(node.id, tie_node.id):
            _create_dummy_activity(node, tie_node, network)
        else:
            for activity in list(node.inbound_activities):
                node_dict.move_activity_to_node(activity, tie_node)


# ---------------------------------------------------------------------------
# Activity attachment helpers
# ---------------------------------------------------------------------------


def _create_dummy_activity(start_node: Node, end_node: Node, network: Network) -> None:
    """Create and attach a zero-effort dummy activity between two nodes.

    Arguments:
        start_node: The starting node for the dummy activity.
        end_node: The ending node for the dummy activity.
        network: The network being constructed.
    """
    dummy_activity = network.activities.new_dummy_activity()
    dummy_activity.predecessors = start_node.start_dependencies
    network.node_dict.attach_activity(dummy_activity, start_node, end_node)


# ---------------------------------------------------------------------------
# Predecessor resolution
# ---------------------------------------------------------------------------


def _find_related_predecessor_sets(activity_ids: set[int], activities: ActivityCollection) -> list[set[int]]:
    """Return all predecessor sets that contain any ID from the given set.

    A predecessor set is the full set of predecessors for some activity. This
    function finds every such set that overlaps with *activity_ids*.

    Arguments:
        activity_ids: A set of activity IDs to search for.
        activities: The activity collection providing the reverse lookup.

    Returns:
        list[set[int]]: All predecessor sets containing at least one ID from *activity_ids*.

    Example:
        If activity 4 has predecessors {1, 2, 3} and activity 5 has predecessors
        {1, 2}, then for activity_ids={2} the result is [{1, 2, 3}, {1, 2}].
    """
    return [pred_set for aid in activity_ids for pred_set in activities.reverse_predecessor_lut[aid]]


def _find_resolvable_predecessors(predecessors: set[int], network: Network) -> set[int]:
    """Find the subset of predecessors that can be bound to a single existing node.

    Iterates over all existing predecessor sets and removes IDs that are
    "locked" by another set (i.e. already committed to a different grouping),
    returning only the IDs that are free to merge.

    Arguments:
        predecessors: The full set of predecessor activity IDs.
        network: The network being constructed.

    Returns:
        set[int]: The subset of predecessors that can be resolved to one node.
                  May be empty if no predecessors can be merged.
    """
    if not predecessors or predecessors in network.node_dict:
        return predecessors

    remaining = predecessors.copy()
    for pred_set in _find_related_predecessor_sets(predecessors, network.activities):
        if not predecessors.issubset(pred_set):
            remaining.difference_update(pred_set)
        if not remaining:
            return remaining

    return remaining


def _find_mergeable_subset(candidate: set[int], network: Network) -> set[int] | None:
    """Search for a subset of IDs that can be merged without violating existing groupings.

    Returns the mergeable subset only if it contains more than one element
    (single-element subsets don't need merging).

    Arguments:
        candidate: The candidate set of activity IDs to check.
        network: The network being constructed.

    Returns:
        set[int] | None: The mergeable subset, or None if merging is not possible.
    """
    subset = _find_resolvable_predecessors(candidate, network)
    return subset if subset and len(subset) > 1 else None


# ---------------------------------------------------------------------------
# Minimal covering sets
# ---------------------------------------------------------------------------


def _minimal_covering_sets(list_of_sets: list[set[int]]) -> list[set[int]]:
    """Find the smallest list of sets that covers all elements without duplicates.

    Given a list of sets (which may contain overlapping elements), return a
    minimal sub-list whose union equals the union of the original list. Sets
    are considered smallest-first, so larger sets are preferred as they cover
    more elements at once.

    Arguments:
        list_of_sets: The candidate sets, potentially with overlapping elements.

    Returns:
        list[set[int]]: A minimal sub-list covering all elements.
    """
    sorted_sets = sorted(list_of_sets, key=lambda x: len(x))
    result: list[set[int]] = []
    target = _get_union(sorted_sets)

    for i, candidate in enumerate(sorted_sets):
        result_union = _get_union(result)
        remaining_union = _get_union(sorted_sets[i + 1 :])
        if result_union.union(remaining_union) != target:
            result.append(candidate)

    return result


# ---------------------------------------------------------------------------
# Activity allocation
# ---------------------------------------------------------------------------


def _allocate_activity(activity: Activity, network: Network) -> None:
    """Allocate an activity to start and end nodes in the network.

    Finds or creates appropriate start and end nodes for the given activity,
    inserting dummy activities where needed to maintain dependency semantics.

    The process involves:
    1. Finding predecessors that can be resolved to an existing node.
    2. Finding and merging additional predecessor subsets.
    3. Linking remaining predecessor nodes via dummy activities.
    4. Attaching the activity between the resolved start node and a new end node.

    Arguments:
        activity: The activity to allocate.
        network: The network being constructed.
    """
    predecessors = activity.predecessors.copy()

    # Step 1: Find the largest resolvable predecessor set and merge its nodes
    tie_node_id = _find_resolvable_predecessors(predecessors.copy(), network)
    if tie_node_id:
        _merge_subset(tie_node_id, network)
        predecessors.difference_update(tie_node_id)

    # Step 2: Find and merge additional predecessor subsets
    mergeable_nodes: list[set[int]] = []
    for subset in _sorted_power_set(list(predecessors)):
        node_id = _find_mergeable_subset(subset, network)
        if node_id:
            predecessors.difference_update(node_id)
            mergeable_nodes.append(node_id)

    for node_set in mergeable_nodes:
        _merge_subset(node_set, network)

    # Step 3: Collect remaining predecessor nodes already present in the network
    remaining_nodes = [subset for subset in _sorted_power_set(list(predecessors)) if subset in network.node_dict]

    # Step 4: Find the minimal set of nodes needed for dummy links
    dummy_link_sources = _minimal_covering_sets(mergeable_nodes + remaining_nodes)

    # Step 5: Determine the start node for this activity
    if tie_node_id:
        start_node = network.node_dict[tie_node_id]
    elif dummy_link_sources:
        start_node = network.node_dict.new_node()
    else:
        start_node = network.node_dict.start_node

    # Step 6: Create dummy activities linking predecessor nodes to the start node
    for node_set in dummy_link_sources:
        _create_dummy_activity(network.node_dict[node_set], start_node, network)

    # Step 7: Attach the activity from start node to a new end node
    network.node_dict.attach_activity(activity, start_node, end_node=network.node_dict.new_node())


# ---------------------------------------------------------------------------
# Node merging
# ---------------------------------------------------------------------------


def _merge_subset(merge_set: set[int], network: Network) -> None:
    """Decompose a merge set into existing nodes and merge them together.

    Since subsets of the merge set may already have been merged into existing
    nodes, this function:
    1. Finds existing nodes that together cover the merge set (largest first).
    2. Merges all found nodes into a single node.

    Arguments:
        merge_set: The set of predecessor activity IDs to merge.
        network: The network being constructed.

    Raises:
        AoAException: If the merge set cannot be decomposed into existing nodes.
    """
    node_sets_to_link: list[set[int]] = []
    remaining = merge_set.copy()

    while remaining:
        previous_remaining = remaining.copy()
        for subset in _sorted_power_set(list(remaining)):
            if set(subset) in network.node_dict:
                node_sets_to_link.append(set(subset))
                remaining.difference_update(subset)
                break
        if previous_remaining == remaining:
            raise AoAException("Unable to merge subset")

    _merge_node_sets(node_sets_to_link, network)


def _merge_node_sets(node_sets: list[set[int]], network: Network) -> None:
    """Merge a list of node sets into a single node.

    Iterates through the node sets, merging each into the first (head) node.
    Nodes that share a common ancestor with the head are linked via a dummy
    activity; otherwise their inbound activities are moved directly.

    Arguments:
        node_sets: The node sets to merge (first set is the initial merge target).
        network: The network being constructed.
    """
    if len(node_sets) < 2:
        return

    node_dict = network.node_dict
    head_node = node_dict[node_sets[0]]

    for tail_key in node_sets[1:]:
        tail_node = node_dict[tail_key]

        if node_dict.have_common_ancestor(head_node.id, tail_node.id):
            _create_dummy_activity(tail_node, head_node, network)
        else:
            for activity in list(tail_node.inbound_activities):
                node_dict.move_activity_to_node(activity, head_node)
