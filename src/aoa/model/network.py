#!/usr/bin/env python3
from __future__ import annotations

from collections import OrderedDict
from copy import copy
from dataclasses import dataclass, field
from functools import reduce

from more_itertools import powerset

from .activity import Activity, ActivityCollection
from .exception import AllocationException
from .node import Node
from .node_dict import ActivityNodes, NodeDict


@dataclass
class Network:
    activities: ActivityCollection
    node_dict: NodeDict = field(default_factory=NodeDict, compare=False)

    def get_node_list_sorted_by_depth(self) -> list[Node]:
        """Iterate over all nodes and sort them by depth (depth of the graph from the root).

        Returns:
            list[Node]: All nodes sorted by depth
        """
        nodes = list(sorted(self.node_dict.values(), key=lambda x: x.id))
        return nodes

    def get_activity_nodes(self, activity: Activity) -> ActivityNodes:
        """Returns the start and end node for a given activity

        Arguments:
            activity (Activity): The activity

        Returns:
            ActivityNodes: The start and end node for the given activity
        """
        return self.node_dict.nodes_of(activity.id)

    def get_activity_start_node(self, activity: Activity) -> Node | None:
        """Returns the start node for a given activity

        Arguments:
            activty (Activity): The activity

        Returns:
            Node: The start node of the given activity
        """
        return self.node_dict.nodes_of(activity.id).start_node

    def get_activity_end_node(self, activity: Activity) -> Node | None:
        """Returns the start node for a given activity

        Arguments:
            activty (Activity): The activity

        Returns:
            Node: The start node of the given activity
        """
        return self.node_dict.nodes_of(activity.id).end_node


def create_network(activities: ActivityCollection) -> Network:
    """Factory method to create a network from a list of activities.

    Arguments:
        activities (list[Activity]): The list of activities to create the network from
    Returns:
        Network: The created network
    """
    network = Network(activities)

    sorted_activities: list[Activity] = _get_allocation_sequence(copy(network.activities))
    for activity in sorted_activities:
        _allocate_activity(activity, network.activities, network.node_dict)
    _tie_end_node(network.activities, network.node_dict)
    _renumber_nodes(network.node_dict)

    return network


def power_subset(predecessors: list[int]) -> list[set[int]]:
    """For a list of values returns all possible power sets from largest to smallest.

    Arguments:
        predecessors (list[int]): The list of all predecessors

    Returns:
        list[set[int]]: A List of all possible sets from the giver predecessors

    Example:
        >>> Network.power_subset([1, 2, 3])
        [{1, 2, 3}, {1, 2}, {1, 3}, {2, 3}, {1}, {2}, {3}, set()]
    """
    powersets = [set(x) for x in list(powerset(predecessors))]
    return sorted(powersets, key=lambda x: len(x), reverse=True)


def _get_sets_that_contain_ids_in_id_set(id_set: set[int], activities: ActivityCollection) -> list[set[int]]:
    """For a given set of ids returns all sets that contain any one of those ids.

    Where a set is a group of activities which are together the predecessors of an activity

    Arguments:
        id_set (set[int]): A set of Ids

    Returns:
        list[set[int]]: A List of all sets that contain any one of the ids int the id_set

    Example:

        Activity 4 has predecessors 1, 2 and 3
        Activity 5 has predecessors 1, 2

        So for Activity 2 the sets in which it exists are {1, 2, 3} and {1, 2}
    """

    return [subset for id in id_set for subset in activities.reverse_predecessor_lut[id]]


def _get_allocation_sequence(activities: ActivityCollection) -> list[Activity]:
    allocated_activities: OrderedDict[int, Activity] = OrderedDict()

    def get_allocation_sequence_recursion(
        unallocated_activities: ActivityCollection,
    ) -> list[Activity]:
        """Recursive Function to determine a sequence in which the activities can be allocated.

        An activity can be allocated, when all activities which the activity depends on have been allocated.

        Arguments:
            unallocated_activities (list[Activity]): The list of activities to allocate

        Returns:
            list[Activity]: The list of activities in the order of allocation
        """
        if not unallocated_activities:
            return list(allocated_activities.values())

        allocated_set = set(allocated_activities.keys())
        for idx, activity in unallocated_activities.items():
            id: int = activity.id
            if activity.predecessors.issubset(allocated_set):
                allocated_activities[id] = activity
                del unallocated_activities[idx]
                return get_allocation_sequence_recursion(unallocated_activities)

        raise AllocationException(
            f"Activities {[activity.id for activity in unallocated_activities.values()]}, can't be allocated as ther depencecies can't be resolved"
        )

    return get_allocation_sequence_recursion(activities)


def _renumber_nodes(node_dict: NodeDict) -> None:
    """Renumber nodes to have a consecutive numbering for the nodes.

    During the generation of the nodes some nodes can be tied together resulting in nodes getting dropped, therefore
    the numbering is in order but there might be blanks. This method renumbers all nodes in sequential manner
    without changing their order.
    """
    if len(node_dict.end_nodes) > 1:
        end_node_ids = [node.id for node in node_dict.end_nodes]
        raise Exception(f"Undefined end_node, multiple end nodes detected {end_node_ids}")
    if len(node_dict.end_nodes) == 0:
        raise Exception("Undefined end_node, no end node defined")

    sorted_nodes: list[Node] = []
    for node in list(sorted(node_dict.values(), key=lambda x: x.max_depth)):
        sorted_nodes.append(node)

    for index, node in enumerate(sorted_nodes):
        node.id = index


def _tie_end_node(activities: ActivityCollection, node_dict: NodeDict) -> None:
    """Ties leaf nodes to one end node.

    The algorithm for allocating nodes leaves end nodes as they are. This function ties them to one common end node.
    """

    tie_node = reduce(
        lambda a, b: a if a.max_depth > b.max_depth else b,
        node_dict.end_nodes,
    )

    for node in node_dict.end_nodes:
        if node.id == tie_node.id:
            continue
        if node_dict.have_common_ancestor(node.id, tie_node.id):
            _create_dummy_activity(node, tie_node, activities)
        else:
            for activity in node.inbound_activities:
                # update the inbound activities and start dependencies
                tie_node.inbound_activities.append(activity)


def _attach_activity(activity: Activity, start_node: Node, end_node: Node) -> None:
    """Attach an activity to given start node and create an end node.

    * The outbound activities of the start node are updated.
    * The earliest start of the end node is set.
    * The activity is added to the inbound activities of the end node
    * The start and end node are added to the activity
    * The node lookup table is updated with the new activity

    Arguments:
        activity(Activity): The activity to attach to the start node
        start_node(Node): The start node to which the activity is attached
    """
    start_node.outbound_activities.append(activity)
    end_node.inbound_activities.append(activity)


def _create_dummy_activity(start_node: Node, end_node: Node, activities: ActivityCollection) -> None:
    """Add an dummy node between a start and end node.

    Arguments:
        start_node(Node): The starting node for the dummy activity
        end_node(Node): The ending node for the dummy activity

    Returns:
        set[int]: The updated id of the end node
    """
    dummy_activity = activities.new_dummy_activity()
    dummy_activity.predecessors = start_node.start_dependencies
    _attach_activity(dummy_activity, start_node, end_node)


def _find_mergable_subset_for_set(
    id_set: set[int], activities: ActivityCollection, node_dict: NodeDict
) -> set[int] | None:
    """
    Search for nodes that can be merged with the provided set without violating the existing set

    Arguments:
        id_set(Set(int)): The set to be checked and merged

    Returns:
        Optional[set[int]]: The mergable subset that can be merged with the given id_set
    """
    subset = _find_tieable_node_for_set(id_set, activities, node_dict)
    return subset if subset and len(subset) > 1 else None


def _find_tieable_node_for_set(predecessors: set[int], activities: ActivityCollection, node_dict: NodeDict) -> set[int]:
    """
    Searches over all existing sets and removes nodes which are bound by a existing set

    Arguments:
        predecessors(set[int]): A set of predecessor activities

    Returns:
        Optional[set[int]]: The activities that can be merged to one end node

    """

    # if there are no predecessors or the predecessors are already reprexented by a node then return the
    # predecessors as they are
    if (not predecessors) or (predecessors in node_dict):
        return predecessors

    mutable_node_id = predecessors.copy()
    for pred_set in _get_sets_that_contain_ids_in_id_set(predecessors, activities):
        if not predecessors.issubset(pred_set):
            mutable_node_id.difference_update(pred_set)
        # early exit if there are no more nodes that can be tied together
        if not mutable_node_id:
            return mutable_node_id

    return mutable_node_id


def _allocate_activity(activity: Activity, activities: ActivityCollection, node_dict: NodeDict) -> None:
    """Add the provided activity and link all dependencies to it."""
    predecessors = activity.predecessors.copy()

    def update_predecessors(node_id: set[int]) -> set[int]:
        predecessors.difference_update(node_id)
        return node_id

    # if it exists find the node to which all predecessors can be bound
    if tie_node_id := _find_tieable_node_for_set(predecessors.copy(), activities, node_dict):
        _merge_subset(tie_node_id, activities, node_dict)
        predecessors.difference_update(tie_node_id)

    # find subsets that can be created from existing sub-subsets
    mergeable_nodes = [
        update_predecessors(node_id)
        for subset in power_subset(list(predecessors))
        if (node_id := _find_mergable_subset_for_set(subset, activities, node_dict))
    ]

    for node in mergeable_nodes:
        _merge_subset(node, activities, node_dict)

    remaining_nodes = [subset for subset in power_subset(list(predecessors)) if subset in node_dict]

    dummy_link_start_nodes = _minimal_viable_list(mergeable_nodes + remaining_nodes)

    # There are three cases what the tie node can be
    # 1. A tie node determined earlier in the code
    # 2. A dummy link start node (aggregate) -> a new node has to be created
    # 3. No nodes are applicable so the remaining option is the start node
    tie_node = (
        node_dict[tie_node_id]
        if tie_node_id
        else node_dict.new_node() if dummy_link_start_nodes else node_dict.start_node  # floating node
    )

    for node in dummy_link_start_nodes:
        _create_dummy_activity(
            node_dict[node],
            tie_node,
            activities,
        )
    _attach_activity(activity, tie_node, end_node=node_dict.new_node())


def _minimal_viable_list(list_of_sets: list[set[int]]) -> list[set[int]]:
    """
    From a list of sets which can contain nodes multiple times find the minimal set of sets that contains all dependencies but no duplicates
    """
    list_of_sets = sorted(list_of_sets, key=lambda x: len(x))
    return _minimal_viable_list_recursion(list_of_sets, [])


def _minimal_viable_list_recursion(start: list[set[int]], result: list[set[int]]) -> list[set[int]]:
    if not start:
        return result
    result_union = _get_union(result)
    target = result_union.union(_get_union(start))
    if result_union.union(_get_union(start[1:])) != target:
        result.append(start[0])
    return _minimal_viable_list_recursion(start[1:], result)


def _get_union(list_of_sets: list[set[int]]) -> set[int]:
    if list_of_sets:
        return set.union(*list_of_sets)  # pyright: ignore [reportUnknownMemberType, reportUnknownVariableType]
    else:
        return set()


def _merge_subset(merge_set: set[int], activities: ActivityCollection, node_dict: NodeDict) -> None:
    """
    As subsets of the to be merged subset could potentially already have been merged the following steps are required
    1. go through each subset of the subset and check if there is a activity with that id
    2. if activity with such an id exists add the activity id to the list of activity ids to link
    3. remove vitual node subset from orig subset
    4. if len of orig subset > 0 goto 1
    """
    activity_ids_to_link: list[set[int]] = []
    mutable_merge_set = merge_set.copy()
    while mutable_merge_set:
        old_mutable_merge_set = mutable_merge_set.copy()
        for subset in power_subset(list(mutable_merge_set)):
            if set(subset) in node_dict:
                activity_ids_to_link.append(set(subset))
                mutable_merge_set.difference_update(subset)
                break
        if old_mutable_merge_set == mutable_merge_set:
            raise Exception("Unable to merge subset")
    if activity_ids_to_link:
        _recursive_merge(activity_ids_to_link[0], activity_ids_to_link[1:], activities, node_dict)


def _recursive_merge(head: set[int], tail: list[set[int]], activities: ActivityCollection, node_dict: NodeDict) -> None:
    new_head: set[int] = set()
    if tail:
        if node_dict.have_common_ancestor(node_dict[head].id, node_dict[tail[0]].id):
            _create_dummy_activity(node_dict[tail[0]], node_dict[head], activities)
            new_head = node_dict[head].start_dependencies
        else:
            for activity in node_dict[tail[0]].inbound_activities:
                node_dict[head].inbound_activities.append(activity)

        _recursive_merge(new_head, tail[1:], activities, node_dict)
