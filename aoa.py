#!/usr/bin/env python3
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

import yaml
from more_itertools import powerset

YamlActivity = Dict[str, Any]
YamlActivities = List[YamlActivity]


class KeyMapping(Enum):
    ID = "Id"
    PREDECESSORS = "Predecessors"
    ACTIVITY = "Activity"
    DURATION = "Duration"
    OWNER = "Owner"
    RESSOURCE = "Ressource"


@dataclass
class Node:
    id: int
    erliest_start: Optional[int] = None
    latest_start: Optional[int] = None
    inbound_activities: List[Union[Activity, DummyActivity]] = field(default_factory=list)
    outbound_activities: List[Union[Activity, DummyActivity]] = field(default_factory=list)


@dataclass
class Activity:
    id: int
    start_node: Node
    end_node: Node
    duration: int  # what better way to define time?
    description: str
    float: int = 0


@dataclass
class DummyActivity:
    id: str
    start_node: Node
    end_node: Node


class Network:
    largest_node_id = -1
    dummy_id = -1

    unallocated_activities: YamlActivities = []

    activities: YamlActivities
    start_node: Node
    end_node: Optional[Node] = None
    activity_id_lut: Dict[str, Activity] = {}

    node_dict: Dict[int, Node] = {}

    def allocate_node_id(self) -> int:
        self.largest_node_id += 1
        return self.largest_node_id

    def allocate_dummy_id(self) -> str:
        self.largest_node_id += 1
        return f"d{self.largest_node_id}"

    @classmethod
    def successor_nodes(cls, node: Node) -> Generator[Node, None, None]:
        yield node
        for activity in node.outbound_activities:
            Network.successor_nodes(activity.end_node)

    @classmethod
    def successor_activities(cls, node: Node) -> Generator[Union[Activity, DummyActivity], None, None]:
        for node in Network.successor_nodes(node):
            for activity in node.outbound_activities:
                yield activity

    @classmethod
    def sort_activities_by_predecessor_length(cls, yaml_activities: YamlActivities):
        """shortest first"""

        def sort_func(x):
            return len(x[KeyMapping.PREDECESSORS.value])

        yaml_activities.sort(key=sort_func)

    @classmethod
    def power_subset(cls, predecessors: Tuple[int]) -> List[Tuple[int]]:
        all_subsets = list(powerset(predecessors))
        return sorted(all_subsets, key=lambda x: len(x), reverse=True)

    @classmethod
    def remove_subset_from_list(cls, target: List[int], subset: Tuple[int]) -> None:
        for i in list(subset):
            target.remove(i)

    def __init__(self, activities: YamlActivities):
        self.activities = activities
        self.unallocated_activities = copy.deepcopy(activities)
        self.create_start_node()
        len_previous_unallocated_activities = 0
        # loop or recursion
        while self.unallocated_activities and len_previous_unallocated_activities != len(self.unallocated_activities):
            len_previous_unallocated_activities = len(self.unallocated_activities)
            allocatable_activities = self.get_allocatable_activities()
            Network.sort_activities_by_predecessor_length(allocatable_activities)
            self.allocate_single_predecessor_activities(allocatable_activities)
            for activity in allocatable_activities:
                self.allocate_multi_predessor_activity(activity)

    def create_start_node(self):
        self.start_node = Node(self.allocate_node_id())
        self.node_dict[self.start_node.id] = self.start_node
        for activity in self.activities:
            if KeyMapping.PREDECESSORS.value not in activity or not self.get_predecessors(activity):
                self.create_activity_from_dict(activity, self.start_node)

    def create_activity_from_dict(self, yaml_activity: YamlActivity, start_node: Node) -> Activity:
        activity = Activity(
            id=yaml_activity[KeyMapping.ID.value],
            start_node=start_node,
            end_node=Node(self.allocate_node_id()),
            duration=yaml_activity[KeyMapping.DURATION.value],
            description=yaml_activity[KeyMapping.ACTIVITY.value],
        )
        self.node_dict[activity.end_node.id] = activity.end_node
        self.activity_id_lut[str(activity.id)] = activity
        start_node.outbound_activities.append(activity)
        activity.end_node.inbound_activities.append(activity)
        self.unallocated_activities.remove(yaml_activity)
        return activity

    def create_dummy_activity(self, start_node: Node, end_node: Node) -> None:
        dummy_activity = DummyActivity(
            id=self.allocate_dummy_id(),
            start_node=start_node,
            end_node=end_node,
        )
        start_node.outbound_activities.append(dummy_activity)
        end_node.inbound_activities.append(dummy_activity)

    def get_allocatable_activities(self) -> YamlActivities:
        """activities are allocatable if all there predecessors have been allocated"""
        allocated_activity_ids = self.get_allocated_activity_ids()
        allocatable_activities = []
        for activity in self.unallocated_activities:
            if set(self.get_predecessors(activity)).issubset(allocated_activity_ids):
                allocatable_activities.append(copy.deepcopy(activity))

        return allocatable_activities

    def allocate_single_predecessor_activities(self, allocatable_activities: YamlActivities) -> None:
        local_alloc_activities = copy.deepcopy(allocatable_activities)
        for yaml_activity in local_alloc_activities:
            if len(self.get_predecessors(yaml_activity)) == 1:
                self.link_to_activity_by_id(self.get_predecessors(yaml_activity)[0], yaml_activity)
                allocatable_activities.remove(yaml_activity)

    def get_predecessors(self, activity: YamlActivity) -> List[int]:
        if KeyMapping.PREDECESSORS.value in activity:
            return activity[KeyMapping.PREDECESSORS.value]
        else:
            return []

    def allocate_multi_predessor_activity(self, yaml_activity: YamlActivity) -> None:
        """
        a. find largest subset of predecessors which exist exlusively throughout all unallocated activities
           exlusive meaning that that no element of the subset exists in an activity without the whole subset
        b. if subset found
            c. merge subset
            d. link to subset
            e. remove subset from predecessor list
            f. goto a)
        g. if predecessor list has at least one entry
            h. find largest subset for which a virtual predecessor exists
                i. link the subset
                j. remove subset from predecessor list
                k. got f)
        """

        def find_max_subset(predecessors: Tuple[int]) -> Optional[Tuple[int]]:
            found = False
            subset: Tuple[int] = tuple()
            for subset in Network.power_subset(predecessors):
                if len(subset) > 1:
                    for activity in self.activities:
                        pred = self.get_predecessors(activity)
                        if set(subset) <= set(pred):
                            found = True
                        elif any([i in pred for i in subset]):
                            found = False
                            break
                else:
                    found_count = 0
                    for activity in self.activities:
                        pred = self.get_predecessors(activity)
                        if set(subset) <= set(pred):
                            found = True
                            found_count += 1
                            if found_count > 1:
                                found = False
                                break
                if found:
                    break

            return subset if found else None

        predecessors = sorted(set(copy.deepcopy(self.get_predecessors(yaml_activity))))
        activity_direct_link_list: List[str] = []
        activity_dummy_link_list: List[str] = []
        while predecessors:
            subset = find_max_subset(tuple(predecessors))
            if subset:
                self.merge_subset(subset)
                new_id = self.activity_id(subset)
                activity_direct_link_list.append(new_id)
                Network.remove_subset_from_list(predecessors, subset)
            else:
                break

        while predecessors:
            for subset in Network.power_subset(tuple(predecessors)):
                if self.activity_id(subset) in self.activity_id_lut:
                    activity_dummy_link_list.append(self.activity_id(subset))
                    Network.remove_subset_from_list(predecessors, subset)
                    break

        if activity_direct_link_list:
            new_activity_id = 0
            for index, activity in enumerate(activity_direct_link_list):
                if index == 0:
                    new_activity_id = self.create_activity_from_dict(
                        yaml_activity, self.activity_id_lut[activity].end_node
                    ).id
                else:
                    self.create_dummy_activity(
                        self.activity_id_lut[activity].end_node, self.activity_id_lut[str(new_activity_id)].start_node
                    )
            for activity in activity_dummy_link_list:
                self.create_dummy_activity(
                    self.activity_id_lut[activity].end_node, self.activity_id_lut[str(new_activity_id)].start_node
                )
        else:
            new_activity = self.create_activity_from_dict(yaml_activity, Node(self.allocate_node_id()))
            self.node_dict[new_activity.start_node.id] = new_activity.start_node
            for link in activity_dummy_link_list:
                self.create_dummy_activity(self.activity_id_lut[link].end_node, new_activity.start_node)

    def activity_id(self, id_set: Tuple[int]) -> str:
        return "-".join(map(str, sorted(id_set)))

    def activities_from_id(self, id: str) -> List[int]:
        str_ids = id.split("-")
        return [int(i) for i in str_ids]

    def merge_ids(self, id_l: str, id_r) -> str:
        activities_l = self.activities_from_id(id_l)
        activities_r = self.activities_from_id(id_r)
        activities = activities_l + activities_r
        sorted_activities = sorted(activities)
        activity_tuple = tuple(set(sorted_activities))
        return self.activity_id(activity_tuple)

    def merge_subset(self, merge_set: Tuple[int]) -> None:
        """
        As subsets of the to be merged subset could potentially allready have been merged the following steps are required
        1. go through each subset of the subset and check if there is a activity with that id
        2. if activity with such an id exists add the activity id to the list of activity ids to link
        3. remove vitual node subset from orig subset
        4. if len of orig subset > 0 goto 1
        """
        activity_ids_to_link: List[str] = []
        mutable_merge_set = list(merge_set)
        while mutable_merge_set:
            for subset in Network.power_subset(tuple(mutable_merge_set)):
                if self.activity_id(subset) in self.activity_id_lut:
                    activity_ids_to_link.append(self.activity_id(subset))
                    for item in subset:
                        mutable_merge_set.remove(item)
                    break

        self.recursive_merge(activity_ids_to_link[0], activity_ids_to_link[1:])

    def have_common_ancestor(self, id_l: str, id_r: str) -> bool:
        return True if self.activity_id_lut[id_l].start_node == self.activity_id_lut[id_r].start_node else False

    def recursive_merge(self, head: str, tail: List[str]) -> None:
        new_id = ""
        if tail:
            if self.have_common_ancestor(head, tail[0]):
                self.create_dummy_activity(self.activity_id_lut[head].end_node, self.activity_id_lut[tail[0]].end_node)
                new_id = self.merge_ids(head, tail[0])
                self.activity_id_lut[new_id] = self.activity_id_lut.pop(tail[0])
            else:
                self.activity_id_lut[tail[0]].end_node = self.activity_id_lut[head].end_node
                self.activity_id_lut[head].end_node.inbound_activities.append(self.activity_id_lut[tail[0]])
                new_id = self.merge_ids(head, tail[0])
                self.activity_id_lut[new_id] = self.activity_id_lut.pop(head)
                self.activity_id_lut.pop(tail[0])

            self.recursive_merge(new_id, tail[1:])

    def link_to_activity_by_id(self, id: int, yaml_activity: YamlActivity) -> None:
        start_node = self.activity_id_lut[str(id)].end_node
        if start_node:
            self.create_activity_from_dict(yaml_activity, start_node)

    def get_allocated_activity_ids(self) -> Set[int]:
        return {num for id in self.activity_id_lut.keys() for num in self.activities_from_id(id)}

    def all_successor_nodes(self) -> Generator[Node, None, None]:
        for node in Network.successor_nodes(self.start_node):
            yield node

    def all_successor_activities(self) -> Generator[Union[Activity, DummyActivity], None, None]:
        for activity in Network.successor_activities(self.start_node):
            yield activity


def main(file: Path) -> None:
    project = parse(file)
    network = Network(project["Activities"])

    sorted_nodes = dict(sorted(network.node_dict.items()))
    for k, _ in sorted_nodes.items():
        print(f"map {k}", "{\n}")

    for k, v in sorted_nodes.items():
        for a in v.outbound_activities:
            if type(a) == Activity:
                print(f"{a.start_node.id} --> {a.end_node.id} : {a.description} ({a.id})")
            else:
                print(f"{a.start_node.id} --> {a.end_node.id} #line.dashed")


def parse(file: Path):
    with open(file, "r") as f:
        project = yaml.safe_load(f)
    return project


def get_owner_formatting(activity: YamlActivity, formatting: Dict[str, str]) -> str:
    return f"text.{formatting[activity['Owner']]}"


def create_plantuml_header() -> str:
    print("@startuml PERT")
    print("left to right direction")
    print("' Horizontal lines: -->, <--, <-->")
    print("' Vertical lines: ->, <-, <->")
    print("title Pert: Project Design")
    return ""


def create_plantuml_footer() -> str:
    print("@enduml")
    return ""


def create_plantuml_maps(activitys: List[YamlActivity]) -> str:
    for index in range(0, len(activitys) + 1):
        print(f"map {str(index)} ", "{")
        print("  earliest start => ")
        print("  latest start => ")
        print("  float => ")
    return ""


def create_plantuml_network(dependencies: List[YamlActivity], formatting: Dict[str, str]) -> str:
    for dependency in dependencies:
        create_plantuml_dependencies(dependency, formatting)
    return ""


def create_plantuml_dependencies(activity: YamlActivity, formatting: YamlActivity) -> str:
    create_plantuml_activity(activity, formatting)
    create_plantuml_non_activity_dependency(activity, formatting)
    return ""


def create_plantuml_activity(activity: YamlActivity, formatting: YamlActivity) -> str:
    predecessor = activity["Predecessors"][0]
    print(
        f"{str(predecessor)} --> {str(activity['NodeId'])} #{get_owner_formatting(activity, formatting)} : {activity['Action']}"
    )
    return ""


def create_plantuml_non_activity_dependency(activity: YamlActivity, formatting: YamlActivity) -> str:
    for predecessor in activity["Predecessors"][1:]:
        print(f"{str(predecessor)} --> {str(activity['NodeId'])} #line.dashed")
    return ""


if __name__ == "__main__":
    # args = sys.argv[1:]
    # parse(Path(args[0]))
    # d = parse(Path("AoA.yaml"))
    # create_plantuml_header()
    # create_plantuml_maps(d["Nodes"])
    #
    # create_plantuml_network(d["Nodes"], d["Formatting"])
    # create_plantuml_footer()
    main(Path("AoA.yaml"))
