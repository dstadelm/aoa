#!/usr/bin/env python3
from __future__ import annotations

import copy
from functools import cache
from typing import List, Optional, Set

from more_itertools import powerset

import networkx as nx
from aoa.activity import Activity
from aoa.node import Node


def to_key(id_set: Set[int]) -> str:
    return "-".join(map(str, sorted(id_set)))


def to_set(id: str) -> Set[int]:
    return set([int(i) for i in id.split("-")])


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


class Network:
    START_LABLE = "start"
    END_LABLE = "end"

    def __init__(self, activities: List[Activity]) -> None:
        self.graph = nx.DiGraph()
        self.start_node_id = 0
        self.graph.add_node(str(self.start_node_id), data=Node())
        self.end_activity_ids = self.get_end_activity_ids(activities)

        self.activities = copy.deepcopy(activities)

        self.end_node_id = len(self.activities)

        allocation_sequence = self.get_allocation_sequence(activities, list(), set())
        for activity in allocation_sequence:
            self.allocate_activity(activity)

        nx.relabel_nodes(self.graph, {to_key({self.start_node_id}): Network.START_LABLE}, copy=False)
        nx.relabel_nodes(self.graph, {to_key({self.end_node_id}): Network.END_LABLE}, copy=False)

        self.graph.nodes[Network.END_LABLE]["data"].latest_finish = self.graph.nodes[Network.END_LABLE][
            "data"
        ].earliest_start
        self.backtrack_cpm_values()
        self.cpm_free_float_values()

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @graph.setter
    def graph(self, value: nx.DiGraph):
        self._graph = value

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

    def cpm_free_float_values(self):
        for start_node, end_node, data in self.graph.in_edges(data=True):
            activity = data["activity"]
            start_node = self.graph.nodes[start_node]["data"]
            end_node = self.graph.nodes[end_node]["data"]
            free_float = end_node.latest_finish - start_node.latest_finish - activity.duration
            activity.free_float = free_float

    def backtrack_cpm_values(self):
        def recursion(end_nodes: List[str]):
            new_end_nodes = []
            for end_node in end_nodes:
                for start_node, _, data in self.graph.in_edges(end_node, data=True):
                    en = self.graph.nodes[end_node]["data"]
                    activity = data["activity"]
                    latest_finish = en.latest_finish - activity.duration
                    sn = self.graph.nodes[start_node]["data"]
                    activity.latest_finish = en.latest_finish
                    activity.earliest_start = sn.earliest_start
                    sn.update_latest_finish(latest_finish)
                    new_end_nodes.append(start_node)

            if new_end_nodes:
                recursion(new_end_nodes)

        recursion([Network.END_LABLE])

    def get_end_activity_ids(self, activities: list[Activity]) -> Set[int]:
        non_end_activities = {p for activity in activities for p in activity.predecessors}
        activity_ids = {a.id for a in activities}
        return set(activity_ids).difference(non_end_activities)

    @cache
    def reverse_predecessor_lut(self) -> dict[int, list[Set[int]]]:
        ret = dict()
        for activity in self.activities:
            for id in activity.predecessors:
                ret.setdefault(id, []).append(activity.predecessors)
        return ret

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

        return [subset for id in id_set for subset in self.reverse_predecessor_lut()[id]]

    def get_allocation_sequence(
        self, activities: List[Activity], allocated_activities: List[Activity], allocated_ids: Set[int]
    ) -> List[Activity]:
        """Recursive Function to determine a sequence in which the activities can be allocated.

        An activity can be allocated, when all activities which the activity depends on have been allocated.

        Returns:
            A list of activities in the order of allocation of the activities
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

    def has_node(self, s: set[int]) -> bool:
        return self.graph.has_node(to_key(s))

    def attach_activity(self, activity: Activity, start_node: Set[int]):
        end_node = str(self.end_node_id) if activity.id in self.end_activity_ids else str(activity.id)
        self.graph.add_node(end_node, data=Node())
        self.graph.add_edge(to_key(start_node), end_node, activity=activity)
        self.update_earliest_start_and_max_depth(end_node)
        return activity

    def create_dummy_activity(self, start_node: Set[int], end_node: Set[int]) -> Set[int]:
        dummy_activity = Activity()
        self.graph.add_edge(to_key(start_node), to_key(end_node), activity=dummy_activity)
        new_id = start_node.union(end_node)
        nx.relabel_nodes(self.graph, {to_key(end_node): to_key(new_id)}, copy=False)
        self.update_earliest_start_and_max_depth(to_key(new_id))
        return new_id

    def update_earliest_start_and_max_depth(self, end_node: str) -> None:
        for start, _ in self.graph.in_edges(end_node):  # pyright: ignore [reportArgumentType]
            self.evaluate_earliest_start_and_max_depth(start, end_node)

    def evaluate_earliest_start_and_max_depth(self, start_node_name: str, end_node_name: str):
        activity = self.graph.get_edge_data(start_node_name, end_node_name)["activity"]
        duration = activity.duration
        earliest_start_end_node = self.graph.nodes[start_node_name]["data"].earliest_start + duration

        start_node = self.graph.nodes[start_node_name]["data"]
        max_depth = start_node.max_depth + 1
        end_node = self.graph.nodes[end_node_name]["data"]
        end_node.update_earliest_start(earliest_start_end_node)
        end_node.update_max_depth(max_depth)

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
        if not predecessors or self.has_node(predecessors):
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

        remaining_nodes = [
            subset for subset in Network.power_subset(list(predecessors)) if self.graph.has_node(to_key(subset))
        ]

        dummy_link_start_nodes = self.minimal_viable_list(mergeable_nodes + remaining_nodes)

        tie_node = (
            tie_node_id
            if tie_node_id
            else (
                set.union(*dummy_link_start_nodes)
                if dummy_link_start_nodes
                else {self.start_node_id}  # new floating node
            )  # the root node
        )

        for node in dummy_link_start_nodes:
            tie_node = self.create_dummy_activity(
                node,
                tie_node,
            )
        self.attach_activity(activity, tie_node)

    def minimal_viable_list(self, list_of_sets: List[Set[int]]) -> List[Set[int]]:
        """From a list of sets which can contain nodes multiple times find the minimal set of sets that contains all
        dependencies but no duplicates.
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
        """Merge multiple subsets.
        As subsets of the to be merged subset could potentially already have been merged the following steps are
        required
        1. go through each subset of the subset and check if there is a activity with that id
        2. if activity with such an id exists add the activity id to the list of activity ids to link
        3. remove vitual node subset from orig subset
        4. if len of orig subset > 0 goto 1
        """
        activity_ids_to_link: List[Set[int]] = []
        mutable_merge_set = merge_set.copy()
        while mutable_merge_set:
            for subset in Network.power_subset(list(mutable_merge_set)):
                # if set(subset) in self.node_lut:
                #     activity_ids_to_link.append(set(subset))
                #     mutable_merge_set.difference_update(subset)
                #     break
                if self.has_node(set(subset)):
                    activity_ids_to_link.append(set(subset))
                    mutable_merge_set.difference_update(subset)
                    break

        self.recursive_merge(activity_ids_to_link[0], activity_ids_to_link[1:])

    def recursive_merge(self, head: Set[int], tail: List[Set[int]]) -> None:
        new_head: Set[int] = set()
        if tail:
            if self.have_common_ancestor(head, tail[0]):
                new_head = self.create_dummy_activity(tail[0], head)
            else:
                new_head = head.union(tail[0])
                self.merge_nodes(new_head, head)
                self.merge_nodes(new_head, tail[0])

            self.recursive_merge(new_head, tail[1:])

    def merge_nodes(self, survivor: Set[int], loser: Set[int]):
        survivor_key = to_key(survivor)
        loser_key = to_key(loser)
        if not self.graph.has_node(survivor_key):
            self.graph.add_node(survivor_key, data=Node())
        for start, _, data in self.graph.in_edges(loser_key, data=True):
            self.graph.add_edge(start, survivor_key, activity=data["activity"])
            self.evaluate_earliest_start_and_max_depth(start, survivor_key)

        self.graph.remove_node(loser_key)

    def have_common_ancestor(self, node_left: Set[int], node_right: Set[int]) -> bool:
        ids_left = set(self.graph.predecessors(to_key(node_left)))
        ids_right = set(self.graph.predecessors(to_key(node_right)))
        return True if ids_left.intersection(ids_right) else False
