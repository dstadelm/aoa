#!/usr/bin/env python3
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

import yaml
from more_itertools import powerset

YamlActivity = Dict[str, Any]
YamlActivities = List[YamlActivity]

logger = logging.getLogger(__name__)


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
    erliest_start: Optional[int] = field(default=None, compare=False)
    latest_start: Optional[int] = field(default=None, compare=False)
    inbound_activities: List[Union[Activity, DummyActivity]] = field(default_factory=list, repr=False, compare=False)
    outbound_activities: List[Union[Activity, DummyActivity]] = field(default_factory=list, repr=False, compare=False)
    max_depth: int = field(default=0, compare=False)


@dataclass
class Activity:
    id: int
    start_node: Node = field(compare=False)
    end_node: Node = field(compare=False)
    duration: int = field(compare=False)  # what better way to define time?
    description: str = field(compare=False)
    float: int = field(default=0, compare=False)


@dataclass
class DummyActivity:
    id: int
    start_node: Node
    end_node: Node


class NodeLut(dict):
    def to_key(self, id_set: Set[int]) -> str:
        return "-".join(map(str, sorted(id_set)))

    def to_set(self, id: str) -> Set[int]:
        return set([int(i) for i in id.split("-")])

    def __setitem__(self, key: Set[int], value: Node) -> None:
        super().__setitem__(self.to_key(key), value)

    def __getitem__(self, key: Set[int]) -> Node:
        return super().__getitem__(self.to_key(key))

    def __delitem__(self, key: Set[int]) -> None:
        return super().__delitem__(self.to_key(key))

    def __contains__(self, key: Set[int]) -> bool:
        return super().__contains__(self.to_key(key))

    def items(self):
        for k, v in super().items():
            yield self.to_set(k), v

    def keys(self):
        for k in super().keys():
            yield self.to_set(k)

    def __repr__(self) -> str:
        m = "set, node id\n"
        for k, v in self.items():
            m += f"{k}, {v.id}"
        return m


class Network:
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
    def power_subset(cls, predecessors: List[int]) -> List[Set[int]]:
        powersets = [set(x) for x in list(powerset(predecessors))]
        return sorted(powersets, key=lambda x: len(x), reverse=True)

    @classmethod
    def remove_subset_from_list(cls, target: List[int], subset: Set[int]) -> None:
        for i in list(subset):
            target.remove(i)

    def __init__(self, activities: YamlActivities):
        self.largest_node_id = -1
        self.dummy_id = -1
        self.node_lut: NodeLut = NodeLut()
        self.node_dict: Dict[int, Node] = {}
        self.activities = activities
        self.unallocated_activities: YamlActivities = copy.deepcopy(activities)
        self.start_node: Node = self.create_start_node()
        self.end_node: Optional[Node] = None
        len_previous_unallocated_activities = 0
        # loop or recursion
        while self.unallocated_activities and len_previous_unallocated_activities != len(self.unallocated_activities):
            len_previous_unallocated_activities = len(self.unallocated_activities)
            allocatable_activities = self.get_allocatable_activities()
            Network.sort_activities_by_predecessor_length(allocatable_activities)
            self.allocate_single_predecessor_activities(allocatable_activities)
            for activity in allocatable_activities:
                print(self)
                self.allocate_multi_predessor_activity(activity)
        self.tie_end_node()

    def __repr__(self) -> str:
        nodes = "Nodes by inbound acticityies:\n"
        for ia in self.node_lut.keys():
            nodes += f"  {ia}\n"
        return nodes

    def tie_end_node(self) -> None:
        end_nodes: Dict[int, Node] = {}
        for k, node in enumerate(self.node_dict.values()):
            if not node.outbound_activities:
                end_nodes[k] = node
        if len(end_nodes) > 1:
            tie_node: Optional[Node] = None
            max_depth = -1
            for node in end_nodes.values():
                if node.max_depth > max_depth:
                    tie_node = node
                    max_depth = node.max_depth

            if tie_node:
                for node in end_nodes.values():
                    if node.id != tie_node.id:
                        if self.have_common_ancestor(node, tie_node):
                            self.create_dummy_activity(node, tie_node)
                        else:
                            for activity in node.inbound_activities:
                                self.node_dict.pop(activity.end_node.id)
                                activity.end_node = tie_node

    def allocate_node_id(self) -> int:
        self.largest_node_id += 1
        return self.largest_node_id

    def create_start_node(self) -> Node:
        start_node = Node(self.allocate_node_id())
        self.node_dict[start_node.id] = start_node
        for activity in self.activities:
            if KeyMapping.PREDECESSORS.value not in activity or not self.get_predecessors(activity):
                self.create_activity_from_dict(activity, start_node)
        return start_node

    def create_activity_from_dict(self, yaml_activity: YamlActivity, start_node: Node) -> Activity:
        activity = Activity(
            id=yaml_activity[KeyMapping.ID.value],
            start_node=start_node,
            end_node=Node(self.allocate_node_id(), max_depth=start_node.max_depth + 1),
            duration=yaml_activity[KeyMapping.DURATION.value],
            description=yaml_activity[KeyMapping.ACTIVITY.value],
        )
        self.node_dict[activity.end_node.id] = activity.end_node
        self.node_lut[{activity.id}] = activity.end_node
        start_node.outbound_activities.append(activity)
        activity.end_node.inbound_activities.append(activity)
        self.unallocated_activities.remove(yaml_activity)
        return activity

    def create_dummy_activity(self, start_node: Node, end_node: Node) -> None:
        dummy_activity = DummyActivity(
            id=self.allocate_node_id(),
            start_node=start_node,
            end_node=end_node,
        )
        start_node.outbound_activities.append(dummy_activity)
        end_node.inbound_activities.append(dummy_activity)

        end_node.max_depth = max([start_node.max_depth + 1, end_node.max_depth])

        bindable_activities = [activity for activity in end_node.inbound_activities if type(activity) == Activity]
        bindable_activities += [activity for activity in start_node.inbound_activities if type(activity) == Activity]
        dummy_activities_start_nodes = [
            acticity.start_node for acticity in end_node.inbound_activities if type(acticity) == DummyActivity
        ]
        propagated_acticities = [
            activity
            for start_node in dummy_activities_start_nodes
            for activity in start_node.inbound_activities
            if type(activity) == Activity
        ]

        ids = [activity.id for activity in bindable_activities]
        ids += [acticity.id for acticity in propagated_acticities]

        new_id = set(ids)
        self.node_lut[new_id] = end_node

        if not bindable_activities:
            logger.warning(f"Unable to bind {new_id}, as the end node only has dummy activities")
        if new_id in self.node_lut:
            logger.warning(f"{new_id} already exists in activity lut")

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

        def find_max_subset(predecessors: List[int]) -> Tuple[Set[int], Set[int]]:
            found = False
            largest_subset: Set[int] = set()
            non_end_nodes_of_largest_subset: Set[int] = set()
            subset: Set[int] = set()
            non_end_nodes: Set[int] = set()
            for subset in Network.power_subset(predecessors):
                if len(subset) > 1:
                    for activity in self.activities:
                        pred = self.get_predecessors(activity)
                        if set(subset) <= set(pred):
                            found = True
                        else:
                            non_end_nodes = non_end_nodes.union({i for i in subset if i in pred})
                        if non_end_nodes == set(predecessors):
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
                    if len(set(subset).difference(non_end_nodes)) > len(largest_subset):
                        largest_subset = set(subset)
                        non_end_nodes_of_largest_subset = non_end_nodes

            return largest_subset, non_end_nodes_of_largest_subset

        predecessors = sorted(set(copy.deepcopy(self.get_predecessors(yaml_activity))))
        activity_direct_link_list: List[Set[int]] = []
        activity_dummy_link_set_list: List[Set[int]] = []
        while predecessors:
            subset, non_end_nodes = find_max_subset(predecessors)
            mergable_subset = subset.difference(non_end_nodes)
            if mergable_subset:
                self.merge_subset(mergable_subset)
                activity_direct_link_list.append(mergable_subset)
                Network.remove_subset_from_list(predecessors, mergable_subset)
            else:
                break

        for subset in Network.power_subset(predecessors):
            if set(subset) in self.node_lut:
                if set(subset) <= set().union(*activity_dummy_link_set_list):
                    continue
                activity_dummy_link_set_list.append(set(subset))
                if set.union(*activity_dummy_link_set_list) == set(predecessors):
                    self.minimal_viable_list_update(activity_dummy_link_set_list)
                    break

        if activity_direct_link_list:
            new_activity: Optional[Activity] = None
            for index, activity in enumerate(activity_direct_link_list):
                if index == 0:
                    new_activity = self.create_activity_from_dict(
                        yaml_activity, self.node_lut[activity]
                    )
                else:
                    if new_activity:
                        self.create_dummy_activity(self.node_lut[activity], new_activity.start_node)
            for activity in activity_dummy_link_set_list:
                if new_activity:
                    self.create_dummy_activity(
                        self.node_lut[activity],
                        new_activity.start_node,
                    )
        else:
            floating_node = Node(self.allocate_node_id())
            for link in activity_dummy_link_set_list:
                self.create_dummy_activity(self.node_lut[link], floating_node)

            new_activity = self.create_activity_from_dict(yaml_activity, floating_node)
            self.node_dict[new_activity.start_node.id] = new_activity.start_node

    def minimal_viable_list_update(self, los: List[Set[int]]) -> None:
        required_ids = set.union(*los)
        multi_ids = self.get_multiple_allocated_activity_ids(los)
        for subset in Network.power_subset(multi_ids):
            if subset in los:
                los.remove(subset)
                if set.union(*los) != required_ids:
                    los.append(subset)

    def get_multiple_allocated_activity_ids(self, los: List[Set[int]]) -> List[int]:
        return list(
            map(
                lambda id: id,
                [id for id in set.union(*los) if sum([1 for id_set in los if id in id_set]) > 1],
            )
        )

    def activity_id(self, id_set: Set[int]) -> str:
        return "-".join(map(str, sorted(id_set)))

    def activities_from_id(self, id: str) -> List[int]:
        str_ids = id.split("-")
        return [int(i) for i in str_ids]

    def merge_ids(self, id_l: str, id_r) -> str:
        activities_l = self.activities_from_id(id_l)
        activities_r = self.activities_from_id(id_r)
        activities = activities_l + activities_r
        sorted_activities = sorted(activities)
        activity_tuple = set(sorted_activities)
        return self.activity_id(activity_tuple)

    def merge_subset(self, merge_set: Set[int]) -> None:
        """
        As subsets of the to be merged subset could potentially allready have been merged the following steps are required
        1. go through each subset of the subset and check if there is a activity with that id
        2. if activity with such an id exists add the activity id to the list of activity ids to link
        3. remove vitual node subset from orig subset
        4. if len of orig subset > 0 goto 1
        """
        activity_ids_to_link: List[Set[int]] = []
        mutable_merge_set = list(merge_set)
        while mutable_merge_set:
            for subset in Network.power_subset(mutable_merge_set):
                if set(subset) in self.node_lut:
                    activity_ids_to_link.append(set(subset))
                    for item in subset:
                        mutable_merge_set.remove(item)
                    break

        self.recursive_merge(activity_ids_to_link[0], activity_ids_to_link[1:])

    def have_common_ancestor(self, node_left: Node, node_right: Node) -> bool:
        ids_left = {activity.start_node.id for activity in node_left.inbound_activities}
        ids_right = {activity.start_node.id for activity in node_right.inbound_activities}
        return True if ids_left.intersection(ids_right) else False

    def recursive_merge(self, head: Set[int], tail: List[Set[int]]) -> None:
        new_head : Set[int] = set()
        if tail:
            if self.have_common_ancestor(self.node_lut[head], self.node_lut[tail[0]]):
                self.create_dummy_activity(self.node_lut[head], self.node_lut[tail[0]])
                new_head = head.union(tail[0])
                self.node_lut[new_head] = self.node_lut.pop(tail[0])
            else:
                for activity in self.node_lut[tail[0]].inbound_activities:
                    self.node_lut[head].inbound_activities.append(activity)
                self.node_dict.pop(self.node_lut[tail[0]].id)
                self.node_lut[tail[0]] = self.node_lut[head]
                new_head = head.union(tail[0])
                self.node_lut[new_head] = self.node_lut.pop(head)
                self.node_lut.pop(tail[0])

            self.recursive_merge(new_head, tail[1:])

    def link_to_activity_by_id(self, id: int, yaml_activity: YamlActivity) -> None:
        start_node = self.node_lut[{id}]
        if start_node:
            self.create_activity_from_dict(yaml_activity, start_node)

    def get_allocated_activity_ids(self) -> Set[int]:
        return {num for id in self.node_lut.keys() for num in id}

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
    for k, v in sorted_nodes.items():
        print(f"map {k}", "{\n}")

    for _, v in sorted_nodes.items():
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
    logging.basicConfig(level=logging.WARN)
    main(Path("more_tricky.yaml"))
