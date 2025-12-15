#!/usr/bin/env python3
from __future__ import annotations

import copy
from collections.abc import Generator
from dataclasses import dataclass
from functools import cached_property
from typing import override

from more_itertools import powerset

from .activity import Activity, DummyActivity
from .node import Node
from .node_dict import NodeDict


@dataclass
class ActivityNodes:
    start_node: Node
    end_node: Node


def id_generator(start: int = -1, increment: int = 1) -> Generator[int, None, None]:
    current_id = start
    while True:
        current_id += increment
        yield current_id


class Network:
    def __init__(self, activities: list[Activity]):
        self.node_dict: NodeDict = NodeDict()
        self.activities: list[Activity] = copy.deepcopy(activities)

        self._activity_node_lut: dict[int, ActivityNodes] = dict()

        self.node_id: Generator[int, None, None] = id_generator(start=-1, increment=1)
        self.dummy_activity_id: Generator[int, None, None] = id_generator(start=0, increment=-1)

        self.start_node: Node = Node(next(self.node_id))
        self.end_node: Node | None = None

        allocation_sequence = self._get_allocation_sequence(activities, list(), set())
        for activity in allocation_sequence:
            self._allocate_activity(activity)
        self._tie_end_node()
        self._renumber_nodes()
        # self.calculate_latest_start(self.get_node_list_sorted_by_depth())

    def get_node_list_sorted_by_depth(self) -> list[Node]:
        """Iterate over all nodes and sort them by depth (depth of the graph from the root).

        Returns:
            list[Node]: All nodes sorted by depth
        """
        nodes = [self.start_node]
        nodes += list(sorted(self.node_dict.values(), key=lambda x: x.id))
        return nodes

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

    @cached_property
    def _reverse_predecessor_lut(self) -> dict[int, list[set[int]]]:
        _reverse_predecessor_lut: dict[int, list[set[int]]] = dict()
        for activity in self.activities:
            for id in activity.predecessors:
                _reverse_predecessor_lut.setdefault(id, []).append(activity.predecessors)
        return _reverse_predecessor_lut

    def _get_sets_that_contain_ids_in_set(self, id_set: set[int]) -> list[set[int]]:
        """For a given set of ids returns all all sets that contain any one of those ids.

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

        return [subset for id in id_set for subset in self._reverse_predecessor_lut[id]]

    def _get_allocation_sequence(
        self, activities: list[Activity], allocated_activities: list[Activity], allocated_ids: set[int]
    ) -> list[Activity]:
        """Recursive Function to determine a sequence in which the activities can be allocated.

        An activity can be allocated, when all activities which the activity depends on have been allocated. Returns a
        list of activities in the order of allocation of the activities
        """
        if not activities:
            return allocated_activities

        allocateable_activities: list[Activity] = []
        allocateable_activity_ids: list[set[int]] = list()
        unallocateable_activities: list[Activity] = []

        for activity in activities:
            if activity.predecessors.issubset(allocated_ids):
                allocateable_activities.append(activity)
                allocateable_activity_ids.append({activity.id})
            else:
                unallocateable_activities.append(activity)

        sorted_allocateable_activies = sorted(allocateable_activities, key=lambda x: len(x.predecessors))
        if len(activities) == len(unallocateable_activities):
            raise Exception("Unable to find allocation sequence")

        return self._get_allocation_sequence(
            unallocateable_activities,
            allocated_activities + sorted_allocateable_activies,
            allocated_ids.union(*allocateable_activity_ids),
        )

    # def calculate_latest_start(self, nodes_sorted_by_depth: list[Node]) -> None:
    #     """Iterate over all nodes and determines the latest possible start.
    #
    #     The nodes are updated with the according latest possible start value.
    #
    #     Attributes:
    #         nodes_sorted_by_depth(list[Node]): Nodes of the network sorted by depth
    #
    #     """
    #     reversed_nodes = [nodes_sorted_by_depth[i] for i in range(len(nodes_sorted_by_depth) - 1, -1, -1)]
    #     for node in reversed_nodes:
    #         latest_starts: list[int] = [
    #             self._activities[activity.id].end_node.latest_start - activity.duration
    #             for activity in node.outbound_activities
    #             if type(activity) is Activity
    #         ]
    #         latest_starts += [
    #             self._activities[activity.id].end_node.latest_start
    #             for activity in node.outbound_activities
    #             if type(activity) is DummyActivity
    #         ]
    #         if latest_starts:
    #             node.latest_start = min(latest_starts)
    #         else:
    #             node.latest_start = node.earliest_start
    #
    #         for activity in node.outbound_activities:
    #             if type(activity) is Activity:
    #                 activity.total_float = (
    #                     self._activities[activity.id].end_node.latest_start - activity.duration - node.earliest_start
    #                 )
    #                 activity.free_float = (
    #                     self._activities[activity.id].end_node.earliest_start - activity.duration - node.earliest_start
    #                 )

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
        if not self.end_node:
            raise Exception("Undefined end_node")

        sorted_nodes: list[Node] = [self.start_node]
        for node in list(sorted(self.node_dict.values(), key=lambda x: x.max_depth)):
            if node.id != self.end_node.id:
                sorted_nodes.append(node)
        sorted_nodes.append(self.end_node)

        for index, node in enumerate(sorted_nodes):
            node.id = index

    def _tie_end_node(self) -> None:
        """Ties leaf nodes to one end node.

        The algorithm for allocating nodes leaves end nodes as they are. This function ties them to one common end node.
        """
        end_nodes: NodeDict = NodeDict()
        tie_node: Node = Node(-1)
        max_depth: int = -1
        for id, node in self.node_dict.items():
            if not node.outbound_activities:
                end_nodes[id] = node
                if node.max_depth > max_depth:
                    max_depth = node.max_depth
                    tie_node = node

        if tie_node.start_dependencies:
            del end_nodes[tie_node.start_dependencies]

        for id, node in end_nodes.items():
            if node.id == tie_node.id:
                continue
            if self._have_common_ancestor(node, tie_node):
                _ = self._create_dummy_activity(node, tie_node)
            else:
                for activity in node.inbound_activities:
                    tie_node.inbound_activities.append(activity)
                    self._activity_node_lut[activity.id].end_node = tie_node
                    # if isinstance(activity, Activity):
                    #     earliest_start = self._activities[activity.id].start_node.earliest_start + activity.duration
                    #     if tie_node.earliest_start < earliest_start:
                    #         tie_node.earliest_start = earliest_start
                if node.id != 0:
                    _ = self.node_dict.pop(node.start_dependencies)

        tie_node.latest_start = tie_node.earliest_start
        self.end_node = tie_node

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
        end_node = Node(next(self.node_id), max_depth=start_node.max_depth + 1)
        start_node.outbound_activities.append(activity)
        end_node.inbound_activities.append(activity)
        # end_node.earliest_start = start_node.earliest_start + activity.duration
        self._activity_node_lut[activity.id] = ActivityNodes(start_node, end_node)
        end_node.start_dependencies = {activity.id}
        self.node_dict[{activity.id}] = end_node

    def _create_dummy_activity(self, start_node: Node, end_node: Node) -> set[int]:
        """Add an dummy node between a start and end node.

        Arguments:
            start_node(Node): The starting node for the dummy activity
            end_node(Node): The ending node for the dummy activity

        Returns:
           set[int]: The updated id of the end node
        """
        dummy_activity = DummyActivity(next(self.dummy_activity_id))
        start_node.outbound_activities.append(dummy_activity)
        end_node.inbound_activities.append(dummy_activity)

        end_node.max_depth = max([start_node.max_depth + 1, end_node.max_depth])
        end_node.earliest_start = max([start_node.earliest_start, end_node.earliest_start])

        if end_node.start_dependencies in self.node_dict:
            # Only delete the entry if it references this end_node
            # For building nodes a node with this can already exist, and that node shall not be deleted
            if self.node_dict[end_node.start_dependencies].id == end_node.id:
                _ = self.node_dict.pop(end_node.start_dependencies)

        end_node.start_dependencies = end_node.start_dependencies.union(start_node.start_dependencies)

        # When building a new floating node we will create temporary node ids which already exist
        if end_node.start_dependencies not in self.node_dict:
            self.node_dict[end_node.start_dependencies] = end_node

        self._activity_node_lut[dummy_activity.id] = ActivityNodes(start_node, end_node)
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
        for pred_set in self._get_sets_that_contain_ids_in_set(predecessors):
            if not predecessors.issubset(pred_set):
                mutable_node_id.difference_update(pred_set)
            if not mutable_node_id:
                return None

        if mutable_node_id:
            return mutable_node_id
        else:
            return None

    def _allocate_activity(self, activity: Activity) -> None:
        """
        Add the provided activity and link all dependencies to it


        """
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

        tie_node = (
            self.node_dict[tie_node_id]
            if tie_node_id
            else Node(next(self.node_id)) if dummy_link_start_nodes else self.start_node  # floating node
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
            for subset in Network.power_subset(list(mutable_merge_set)):
                if set(subset) in self.node_dict:
                    activity_ids_to_link.append(set(subset))
                    mutable_merge_set.difference_update(subset)
                    break

        self._recursive_merge(activity_ids_to_link[0], activity_ids_to_link[1:])

    def _recursive_merge(self, head: set[int], tail: list[set[int]]) -> None:
        new_head: set[int] = set()
        if tail:
            if self._have_common_ancestor(self.node_dict[head], self.node_dict[tail[0]]):
                new_head = self._create_dummy_activity(self.node_dict[tail[0]], self.node_dict[head])
            else:
                for activity in self.node_dict[tail[0]].inbound_activities:
                    self._activity_node_lut[activity.id].end_node = self.node_dict[head]
                    self.node_dict[head].inbound_activities.append(activity)
                new_head = head.union(tail[0])
                self.node_dict[head].start_dependencies = new_head
                self.node_dict[new_head] = self.node_dict[head]  # Update lookup with new key
                _ = self.node_dict.pop(tail[0])
                _ = self.node_dict.pop(head)

            self._recursive_merge(new_head, tail[1:])

    def _have_common_ancestor(self, node_left: Node, node_right: Node) -> bool:
        ids_left = {self._activity_node_lut[activity.id].start_node.id for activity in node_left.inbound_activities}
        ids_right = {self._activity_node_lut[activity.id].start_node.id for activity in node_right.inbound_activities}
        return True if ids_left.intersection(ids_right) else False
