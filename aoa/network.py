#!/usr/bin/env python3
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Set

from activity import Activity, DummyActivity
from more_itertools import powerset
from node import Node
from node_dict import NodeDict


@dataclass
class ActivityNodeLut:
    start_node: Node
    end_node: Node


class Network:
    def __init__(self, activities: List[Activity]):
        self.node_lut: NodeDict = NodeDict()
        self.activities = copy.deepcopy(activities)

        self._activities: dict[int, ActivityNodeLut] = dict()

        self.start_node: Node = Node(self.allocate_node_id())
        self.end_node: Optional[Node] = None

        allocation_sequence = self.get_allocation_sequence(activities, list(), set())
        for activity in allocation_sequence:
            self.allocate_activity(activity)
        self._tie_end_node()
        self._renumber_nodes()
        self.calculate_latest_start(self.get_node_list_sorted_by_depth())

    @classmethod
    def power_subset(cls, predecessors: List[int]) -> List[Set[int]]:
        """For a list of values returns all possible power sets from largest to smallest.

        Arguments:
            predecessors (List[int]): The list of all predecessors

        Returns:
            List[Set[int]]: A List of all possible sets from the giver predecessors

        Example:
            >>> Network.power_subset([1, 2, 3])
            [{1, 2, 3}, {1, 2}, {1, 3}, {2, 3}, {1}, {2}, {3}, set()]
        """
        powersets = [set(x) for x in list(powerset(predecessors))]
        return sorted(powersets, key=lambda x: len(x), reverse=True)

    def get_sets_that_contain_ids_in_set(self, id_set: Set[int]) -> List[Set[int]]:
        """For a given set of ids returns all all sets that contain any one of those ids.

        Where a set is a group of activities which are together the predecessors of an activity

        Arguments:
            id_set (Set[int]): A set of Ids

        Returns:
            List[Set[int]]: A List of all sets that contain any one of the ids int the id_set

        Example:

            Activity 4 has predecessors 1, 2 and 3
            Activity 5 has predecessors 1, 2

            So for Activity 2 the sets in which it exists are {1, 2, 3} and {1, 2}
        """

        if not hasattr(Network.get_sets_that_contain_ids_in_set, "reverse_predecessor_lut"):
            Network.get_sets_that_contain_ids_in_set.reverse_predecessor_lut = dict()
            # create cached entry
            for activity in self.activities:
                for id in activity.predecessors:
                    Network.get_sets_that_contain_ids_in_set.reverse_predecessor_lut.setdefault(id, []).append(
                        activity.predecessors
                    )

        return [
            subset for id in id_set for subset in Network.get_sets_that_contain_ids_in_set.reverse_predecessor_lut[id]
        ]

    def get_allocation_sequence(
        self, activities: List[Activity], allocated_activities: List[Activity], allocated_ids: Set[int]
    ) -> List[Activity]:
        """Recursive Function to determine a sequence in which the activities can be allocated.

        An activity can be allocated, when all activities which the activity depends on have been allocated. Returns a
        list of activities in the order of allocation of the activities
        """
        if not activities:
            return allocated_activities

        allocateable_activities = []
        allocateable_activity_ids: List[Set[int]] = list()
        unallocateable_activities = []

        for activity in activities:
            if activity.predecessors.issubset(allocated_ids):
                allocateable_activities.append(activity)
                allocateable_activity_ids.append({activity.id})
            else:
                unallocateable_activities.append(activity)

        sorted_allocateable_activies = sorted(allocateable_activities, key=lambda x: len(x.predecessors))
        if len(activities) == len(unallocateable_activities):
            raise Exception("Unable to find allocation sequence")

        return self.get_allocation_sequence(
            unallocateable_activities,
            allocated_activities + sorted_allocateable_activies,
            allocated_ids.union(*allocateable_activity_ids),
        )

    def calculate_latest_start(self, nodes_sorted_by_depth: List[Node]) -> None:
        """Iterate over all nodes and determines the latest possible start.

        The nodes are updated with the according latest possible start value.

        Attributes:
            nodes_sorted_by_depth(List[Node]): Nodes of the network sorted by depth

        """
        reversed_nodes = [nodes_sorted_by_depth[i] for i in range(len(nodes_sorted_by_depth) - 1, -1, -1)]
        for node in reversed_nodes:
            latest_starts = [
                self._activities[activity.id].end_node.latest_start - activity.duration
                for activity in node.outbound_activities
                if type(activity) == Activity
            ]
            latest_starts += [
                self._activities[activity.id].end_node.latest_start
                for activity in node.outbound_activities
                if type(activity) == DummyActivity
            ]
            if latest_starts:
                node.latest_start = min(latest_starts)
            else:
                node.latest_start = node.earliest_start

            for activity in node.outbound_activities:
                if type(activity) == Activity:
                    activity.total_float = (
                        self._activities[activity.id].end_node.latest_start - activity.duration - node.earliest_start
                    )
                    activity.free_float = (
                        self._activities[activity.id].end_node.earliest_start - activity.duration - node.earliest_start
                    )

    def get_node_list_sorted_by_depth(self) -> List[Node]:
        """Iterate over all nodes and sort them by depth (depth of the graph from the root).

        Returns:
            List[Node]: All nodes sorted by depth
        """
        nodes = [self.start_node]
        nodes += list(sorted(self.node_lut.values(), key=lambda x: x.id))
        return nodes

    def __repr__(self) -> str:
        nodes = "Nodes key, node id:\n"
        for key, node in self.node_lut.items():
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

        sorted_nodes: List[Node] = [self.start_node]
        for node in list(sorted(self.node_lut.values(), key=lambda x: x.max_depth)):
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
        for id, node in self.node_lut.items():
            if not node.outbound_activities:
                end_nodes[id] = node
                if node.max_depth > max_depth:
                    max_depth = node.max_depth
                    tie_node = node

        del end_nodes[tie_node.start_dependencies]

        for id, node in end_nodes.items():
            if self.have_common_ancestor(node, tie_node):
                self.create_dummy_activity(node, tie_node)
            else:
                for activity in node.inbound_activities:
                    tie_node.inbound_activities.append(activity)
                    self._activities[activity.id].end_node = tie_node
                    if isinstance(activity, Activity):
                        earliest_start = self._activities[activity.id].start_node.earliest_start + activity.duration
                        if tie_node.earliest_start < earliest_start:
                            tie_node.earliest_start = earliest_start
                if node.id != 0:
                    self.node_lut.pop(node.start_dependencies)

        tie_node.latest_start = tie_node.earliest_start
        self.end_node = tie_node

    def allocate_node_id(self) -> int:
        """Creates a new node id based on the existing largest node id"""
        if not hasattr(Network.allocate_node_id, "id"):
            Network.allocate_node_id.id = -1
        Network.allocate_node_id.id += 1
        return Network.allocate_node_id.id

    def dummy_activity_id_generator(self) -> int:
        if not hasattr(Network.dummy_activity_id_generator, "id"):
            Network.dummy_activity_id_generator.id = 0
        Network.dummy_activity_id_generator.id -= 1
        return Network.dummy_activity_id_generator.id

    def attach_activity(self, activity: Activity, start_node: Node) -> Activity:
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
        end_node = Node(self.allocate_node_id(), max_depth=start_node.max_depth + 1)
        start_node.outbound_activities.append(activity)
        end_node.inbound_activities.append(activity)
        end_node.earliest_start = start_node.earliest_start + activity.duration
        self._activities[activity.id] = ActivityNodeLut(start_node, end_node)
        self.node_lut[{activity.id}] = end_node
        return activity

    def create_dummy_activity(self, start_node: Node, end_node: Node) -> Set[int]:
        dummy_activity = DummyActivity(self.dummy_activity_id_generator())
        """Add an dummy node between a start and end node.

        Arguments:
            start_node(Node): The starting node for the dummy activity
            end_node(Node): The ending node for the dummy activity

        Returns:
           Set[int]: The updated id of the end node
        """
        start_node.outbound_activities.append(dummy_activity)
        end_node.inbound_activities.append(dummy_activity)

        end_node.max_depth = max([start_node.max_depth + 1, end_node.max_depth])
        end_node.earliest_start = max([start_node.earliest_start, end_node.earliest_start])

        if end_node.start_dependencies in self.node_lut:
            # Only delete the entry if it references this end_node
            # For building nodes a node with this can already exist, and that node shall not be deleted
            if self.node_lut[end_node.start_dependencies].id == end_node.id:
                self.node_lut.pop(end_node.start_dependencies)

        end_node.start_dependencies = end_node.start_dependencies.union(start_node.start_dependencies)

        # When building a new floating node we will create temporary node ids which already exist
        if end_node.start_dependencies not in self.node_lut:
            self.node_lut[end_node.start_dependencies] = end_node

        self._activities[dummy_activity.id] = ActivityNodeLut(start_node, end_node)
        return end_node.start_dependencies

    def find_mergable_subset_for_set(self, id_set: Set[int]) -> Optional[Set[int]]:
        """
        Search for nodes that can be merged with the provided set without violating the existing set

        Arguments:
            id_set(Set(int)): The set to be checked and merged

        Returns:
            Optional[Set[int]]: The mergable subset that can be merged with the given id_set
        """
        subset = self.find_tieable_node_for_set(id_set)
        return subset if subset and len(subset) > 1 else None

    def find_tieable_node_for_set(self, predecessors: Set[int]) -> Optional[Set[int]]:
        """
        Searches over all existing sets and removes nodes which are bound by a existing set

        Arguments:
            predecessors(Set[int]): A set of predecessor activities

        Returns:
            Optional[Set[int]]: The activities that can be merged to one end node

        """
        if not predecessors or predecessors in self.node_lut:
            return predecessors

        mutable_node_id = predecessors.copy()
        for pred_set in self.get_sets_that_contain_ids_in_set(predecessors):
            if not predecessors.issubset(pred_set):
                mutable_node_id.difference_update(pred_set)
            if not mutable_node_id:
                return None

        if mutable_node_id:
            return mutable_node_id
        else:
            return None

    def allocate_activity(self, activity: Activity) -> None:
        """
        Add the provided activity and link all dependencies to it


        """
        predecessors = activity.predecessors.copy()

        def update_predecessors(node_id):
            predecessors.difference_update(node_id)
            return node_id

        # if it exists find the node to which all predecessors can be bound
        if tie_node_id := self.find_tieable_node_for_set(predecessors.copy()):
            self.merge_subset(tie_node_id)
            predecessors.difference_update(tie_node_id)

        # find subsets that can be created from existing sub-subsets
        mergeable_nodes = [
            update_predecessors(node_id)
            for subset in Network.power_subset(list(predecessors))
            if (node_id := self.find_mergable_subset_for_set(subset))
        ]

        for node in mergeable_nodes:
            self.merge_subset(node)

        remaining_nodes = [subset for subset in Network.power_subset(list(predecessors)) if subset in self.node_lut]

        dummy_link_start_nodes = self.minimal_viable_list(mergeable_nodes + remaining_nodes)

        tie_node = (
            self.node_lut[tie_node_id]
            if tie_node_id
            else Node(self.allocate_node_id())  # floating node
            if dummy_link_start_nodes
            else self.start_node
        )

        for node in dummy_link_start_nodes:
            self.create_dummy_activity(
                self.node_lut[node],
                tie_node,
            )
        self.attach_activity(activity, tie_node)

    def minimal_viable_list(self, list_of_sets: List[Set[int]]) -> List[Set[int]]:
        """
        From a list of sets which can contain nodes multiple times find the minimal set of sets that contains all dependencies but no duplicates
        """
        list_of_sets = sorted(list_of_sets, key=lambda x: len(x))
        return self.minimal_viable_list_recursion(list_of_sets, [])

    def minimal_viable_list_recursion(self, start: List[Set[int]], result: List[Set[int]]) -> List[Set[int]]:
        if not start:
            return result
        result_union = self.get_union(result)
        target = result_union.union(self.get_union(start))
        if result_union.union(self.get_union(start[1:])) != target:
            result.append(start[0])
        return self.minimal_viable_list_recursion(start[1:], result)

    def get_union(self, list_of_sets: List[Set[int]]) -> Set[int]:
        if list_of_sets:
            return set.union(*list_of_sets)
        else:
            return set()

    def get_multiple_allocated_activity_ids(self, los: List[Set[int]]) -> List[int]:
        return list(
            map(
                lambda id: id,
                [id for id in set.union(*los) if sum([1 for id_set in los if id in id_set]) > 1],
            )
        )

    def merge_subset(self, merge_set: Set[int]) -> None:
        """
        As subsets of the to be merged subset could potentially already have been merged the following steps are required
        1. go through each subset of the subset and check if there is a activity with that id
        2. if activity with such an id exists add the activity id to the list of activity ids to link
        3. remove vitual node subset from orig subset
        4. if len of orig subset > 0 goto 1
        """
        activity_ids_to_link: List[Set[int]] = []
        mutable_merge_set = merge_set.copy()
        while mutable_merge_set:
            for subset in Network.power_subset(list(mutable_merge_set)):
                if set(subset) in self.node_lut:
                    activity_ids_to_link.append(set(subset))
                    mutable_merge_set.difference_update(subset)
                    break

        self.recursive_merge(activity_ids_to_link[0], activity_ids_to_link[1:])

    def recursive_merge(self, head: Set[int], tail: List[Set[int]]) -> None:
        new_head: Set[int] = set()
        if tail:
            if self.have_common_ancestor(self.node_lut[head], self.node_lut[tail[0]]):
                new_head = self.create_dummy_activity(self.node_lut[tail[0]], self.node_lut[head])
            else:
                for activity in self.node_lut[tail[0]].inbound_activities:
                    self._activities[activity.id].end_node = self.node_lut[head]
                    self.node_lut[head].inbound_activities.append(activity)
                new_head = head.union(tail[0])
                self.node_lut[new_head] = self.node_lut[head]
                self.node_lut.pop(tail[0])
                self.node_lut.pop(head)

            self.recursive_merge(new_head, tail[1:])

    def have_common_ancestor(self, node_left: Node, node_right: Node) -> bool:
        ids_left = {self._activities[activity.id].start_node.id for activity in node_left.inbound_activities}
        ids_right = {self._activities[activity.id].start_node.id for activity in node_right.inbound_activities}
        return True if ids_left.intersection(ids_right) else False
