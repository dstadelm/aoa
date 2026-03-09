#!/usr/bin/env python3
from __future__ import annotations

from collections import OrderedDict
from copy import copy
from functools import reduce
from typing import override

from more_itertools import powerset

from .activity import Activity, ActivityCollection
from .exception import AllocationException
from .node import Node
from .node_dict import ActivityNodes, NodeDict


class Network:
    """Class that creates a network of nodes and activities based on the provided activities.
    Arguments:
        activities (list[Activity]): The list of activities to create the network from
    Raises:
        AllocationException: When the activities can't be allocated due to cyclic dependencies or overconstraining
        NonUniqueIdException: When duplicate activity ID's are detected in the provided activities
    """

    def __init__(self, activities: ActivityCollection):
        self.node_dict: NodeDict = NodeDict()
        self.activities: ActivityCollection = activities

        self._activities: list[Activity] = self._get_allocation_sequence(copy(activities))
        for activity in self._activities:
            self._allocate_activity(activity)
        self._tie_end_node()
        self._renumber_nodes()

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

    @classmethod
    def power_subset(cls, predecessors: list[int]) -> list[set[int]]:
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

    def _get_sets_that_contain_ids_in_id_set(self, id_set: set[int]) -> list[set[int]]:
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

        return [subset for id in id_set for subset in self.activities.reverse_predecessor_lut[id]]

    def _get_allocation_sequence(self, activities: ActivityCollection) -> list[Activity]:
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

    @override
    def __repr__(self) -> str:
        nodes = "Nodes key, node id:\n"
        for key, node in self.node_dict.items():
            nodes += f"{key} => {node.id}\n"
            for activity in node.inbound_activities:
                nodes += activity.__repr__()
                nodes += "\n"
        return nodes

    def _renumber_nodes(self) -> None:
        """Renumber nodes to have a consecutive numbering for the nodes.

        During the generation of the nodes some nodes can be tied together resulting in nodes getting dropped, therefore
        the numbering is in order but there might be blanks. This method renumbers all nodes in sequential manner
        without changing their order.
        """
        if not len(self.node_dict.end_nodes) == 1:
            raise Exception("Undefined end_node")

        sorted_nodes: list[Node] = []
        for node in list(sorted(self.node_dict.values(), key=lambda x: x.max_depth)):
            sorted_nodes.append(node)

        for index, node in enumerate(sorted_nodes):
            node.id = index

    def _tie_end_node(self) -> None:
        """Ties leaf nodes to one end node.

        The algorithm for allocating nodes leaves end nodes as they are. This function ties them to one common end node.
        """

        tie_node = reduce(
            lambda a, b: a if a.max_depth > b.max_depth else b,
            self.node_dict.end_nodes,
        )

        for node in self.node_dict.end_nodes:
            if node.id == tie_node.id:
                continue
            if self._have_common_ancestor(node, tie_node):
                _ = self._create_dummy_activity(node, tie_node)
            else:
                for activity in node.inbound_activities:
                    # update the inbound activities and start dependencies
                    tie_node.inbound_activities.append(activity)

                # if node.id != 0:
                #     _ = self.node_dict.pop(node.start_dependencies)

    def _attach_activity(self, activity: Activity, start_node: Node) -> None:
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
        end_node = self.node_dict.new_node()
        end_node.max_depth = start_node.max_depth + 1
        start_node.outbound_activities.append(activity)
        end_node.inbound_activities.append(activity)

    def _create_dummy_activity(self, start_node: Node, end_node: Node) -> set[int]:
        """Add an dummy node between a start and end node.

        Arguments:
            start_node(Node): The starting node for the dummy activity
            end_node(Node): The ending node for the dummy activity

        Returns:
           set[int]: The updated id of the end node
        """
        dummy_activity = self.activities.new_dummy_activity()
        dummy_activity.predecessors = start_node.start_dependencies
        start_node.outbound_activities.append(dummy_activity)
        end_node.inbound_activities.append(dummy_activity)

        end_node.max_depth = max([start_node.max_depth + 1, end_node.max_depth])

        return end_node.start_dependencies

    def _find_mergable_subset_for_set(self, id_set: set[int]) -> set[int] | None:
        """
        Search for nodes that can be merged with the provided set without violating the existing set

        Arguments:
            id_set(Set(int)): The set to be checked and merged

        Returns:
            Optional[set[int]]: The mergable subset that can be merged with the given id_set
        """
        subset = self._find_tieable_node_for_set(id_set)
        return subset if subset and len(subset) > 1 else None

    def _find_tieable_node_for_set(self, predecessors: set[int]) -> set[int] | None:
        """
        Searches over all existing sets and removes nodes which are bound by a existing set

        Arguments:
            predecessors(set[int]): A set of predecessor activities

        Returns:
            Optional[set[int]]: The activities that can be merged to one end node

        """
        if (not predecessors) or (predecessors in self.node_dict):
            return predecessors

        mutable_node_id = predecessors.copy()
        for pred_set in self._get_sets_that_contain_ids_in_id_set(predecessors):
            if not predecessors.issubset(pred_set):
                mutable_node_id.difference_update(pred_set)
            if not mutable_node_id:
                return None

        if mutable_node_id:
            return mutable_node_id
        else:
            return None

    def _allocate_activity(self, activity: Activity) -> None:
        """Add the provided activity and link all dependencies to it."""
        predecessors = activity.predecessors.copy()

        def update_predecessors(node_id: set[int]) -> set[int]:
            predecessors.difference_update(node_id)
            return node_id

        # if it exists find the node to which all predecessors can be bound
        if tie_node_id := self._find_tieable_node_for_set(predecessors.copy()):
            self._merge_subset(tie_node_id)
            predecessors.difference_update(tie_node_id)

        # find subsets that can be created from existing sub-subsets
        mergeable_nodes = [
            update_predecessors(node_id)
            for subset in Network.power_subset(list(predecessors))
            if (node_id := self._find_mergable_subset_for_set(subset))
        ]

        for node in mergeable_nodes:
            self._merge_subset(node)

        remaining_nodes = [subset for subset in Network.power_subset(list(predecessors)) if subset in self.node_dict]

        dummy_link_start_nodes = self._minimal_viable_list(mergeable_nodes + remaining_nodes)

        # There are three cases what the tie node can be
        # 1. A tie node determined earlier in the code
        # 2. A dummy link start node (aggregate) -> a new node has to be created
        # 3. No nodes are applicable so the remaining option is the start node
        tie_node = (
            self.node_dict[tie_node_id]
            if tie_node_id
            else self.node_dict.new_node() if dummy_link_start_nodes else self.node_dict.start_node  # floating node
        )

        for node in dummy_link_start_nodes:
            _ = self._create_dummy_activity(
                self.node_dict[node],
                tie_node,
            )
        self._attach_activity(activity, tie_node)

    def _minimal_viable_list(self, list_of_sets: list[set[int]]) -> list[set[int]]:
        """
        From a list of sets which can contain nodes multiple times find the minimal set of sets that contains all dependencies but no duplicates
        """
        list_of_sets = sorted(list_of_sets, key=lambda x: len(x))
        return self._minimal_viable_list_recursion(list_of_sets, [])

    def _minimal_viable_list_recursion(self, start: list[set[int]], result: list[set[int]]) -> list[set[int]]:
        if not start:
            return result
        result_union = self._get_union(result)
        target = result_union.union(self._get_union(start))
        if result_union.union(self._get_union(start[1:])) != target:
            result.append(start[0])
        return self._minimal_viable_list_recursion(start[1:], result)

    def _get_union(self, list_of_sets: list[set[int]]) -> set[int]:
        if list_of_sets:
            return set.union(*list_of_sets)  # pyright: ignore [reportUnknownMemberType, reportUnknownVariableType]
        else:
            return set()

    def _merge_subset(self, merge_set: set[int]) -> None:
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
            for subset in Network.power_subset(list(mutable_merge_set)):
                if set(subset) in self.node_dict:
                    activity_ids_to_link.append(set(subset))
                    mutable_merge_set.difference_update(subset)
                    break
            if old_mutable_merge_set == mutable_merge_set:
                raise Exception("Unable to merge subset")
        if activity_ids_to_link:
            self._recursive_merge(activity_ids_to_link[0], activity_ids_to_link[1:])

    def _recursive_merge(self, head: set[int], tail: list[set[int]]) -> None:
        new_head: set[int] = set()
        if tail:
            if self._have_common_ancestor(self.node_dict[head], self.node_dict[tail[0]]):
                new_head = self._create_dummy_activity(self.node_dict[tail[0]], self.node_dict[head])
            else:
                for activity in self.node_dict[tail[0]].inbound_activities:
                    self.node_dict[head].inbound_activities.append(activity)

            self._recursive_merge(new_head, tail[1:])

    def _have_common_ancestor(self, node_left: Node, node_right: Node) -> bool:
        ids_left = {
            self.node_dict.nodes_of(activity.id).start_node.id  # pyright: ignore [reportOptionalMemberAccess]
            for activity in node_left.inbound_activities
            if self.node_dict.nodes_of(activity.id).start_node
        }
        ids_right = {
            self.node_dict.nodes_of(activity.id).start_node.id  # pyright: ignore [reportOptionalMemberAccess]
            for activity in node_right.inbound_activities
            if self.node_dict.nodes_of(activity.id).start_node
        }
        return True if ids_left.intersection(ids_right) else False
