#!/usr/bin/env python3
from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, OrderedDict, Set, Tuple, Union

import yaml
from more_itertools import powerset

YamlActivity = Dict[str, Any]
YamlActivities = List[YamlActivity]

logger = logging.getLogger(__name__)


class KeyMapping(Enum):
    ID = "Id"
    PREDECESSORS = "Predecessors"
    ACTIVITY = "Activity"
    ACTIVITIES = "Activities"
    EFFORT = "Effort"
    OWNER = "Owner"
    RESOURCE = "Resource"
    RESOURCES = "Resources"
    DURATION = "Duration"
    PENSUM = "Pensum"


ID = KeyMapping.ID.value
PREDECESSORS = KeyMapping.PREDECESSORS.value
ACTIVITY = KeyMapping.ACTIVITY.value
ACTIVITIES = KeyMapping.ACTIVITIES.value
EFFORT = KeyMapping.EFFORT.value
OWNER = KeyMapping.OWNER.value
RESOURCE = KeyMapping.RESOURCE.value
RESOURCES = KeyMapping.RESOURCES.value
DURATION = KeyMapping.DURATION.value
PENSUM = KeyMapping.PENSUM.value


@dataclass
class Node:
    id: int
    earliest_start: int = field(default=0, compare=False)
    latest_start: int = field(default=0, compare=False)
    inbound_activities: List[Union[Activity, DummyActivity]] = field(default_factory=list, repr=False, compare=False)
    outbound_activities: List[Union[Activity, DummyActivity]] = field(default_factory=list, repr=False, compare=False)
    max_depth: int = field(default=0, compare=False)
    start_dependencies: Set[int] = field(default_factory=set)

    def __post_init__(self):
        pass  # place init logging stuff here


@dataclass
class Activity:
    id: int
    duration: int = field(compare=False)
    description: str = field(compare=False)
    start_node: Node = field(default=Node(-1), compare=False)
    end_node: Node = field(default=Node(-1), compare=False)
    predecessors: Set[int] = field(default_factory=set, compare=False)
    total_float: int = field(default=0, compare=False)
    free_float: int = field(default=0, compare=False)
    float: int = field(default=0, compare=False)

    def __repr__(self):
        return f"    Activity {str(self.id)}"


@dataclass
class DummyActivity:
    start_node: Node
    end_node: Node

    def __repr__(self):
        return f"    Dummy Activity from start node {str(self.start_node.id)}"


class NodeDict(dict):
    def to_key(self, id_set: Set[int]) -> str:
        return "-".join(map(str, sorted(id_set)))

    def to_set(self, id: str) -> Set[int]:
        return set([int(i) for i in id.split("-")])

    def __setitem__(self, key: Set[int], value: Node) -> None:
        value.start_dependencies = key
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

    def pop(self, key: Set[int]) -> Node:
        return super().pop(self.to_key(key))

    def keys(self):
        for k in super().keys():
            yield self.to_set(k)

    def __repr__(self) -> str:
        m = "set, node id\n"
        for k, v in self.items():
            m += f"{k}:{v.start_dependencies} {v.id}\n"
        return m


class Network:
    @classmethod
    def power_subset(cls, predecessors: List[int]) -> List[Set[int]]:
        powersets = [set(x) for x in list(powerset(predecessors))]
        return sorted(powersets, key=lambda x: len(x), reverse=True)

    @classmethod
    def remove_subset_from_list(cls, target: List[int], subset: Set[int]) -> None:
        for i in list(subset):
            target.remove(i)

    def __init__(self, activities: List[Activity]):
        self.largest_node_id = -1
        self.dummy_id = -1
        self.node_lut: NodeDict = NodeDict()
        self.activities = copy.deepcopy(activities)
        self.start_node: Node = Node(self.allocate_node_id())
        self.end_node: Optional[Node] = None

        allocation_sequence = self.get_allocation_sequence(activities, list(), set())
        for activity in allocation_sequence:
            self.allocate_multi_predessor_activity(activity)
        self.tie_end_node()
        self.renumber_nodes()
        self.calculate_latest_start()

    def get_allocation_sequence(
        self, activities: List[Activity], allocated_activities: List[Activity], allocated_ids: Set[int]
    ) -> List[Activity]:
        if activities:
            allocateable_activities = []
            allocateable_activity_ids: List[Set[int]] = list()
            unallocateable_activities = []

            for activity in activities:
                if not activity.predecessors:
                    allocateable_activities.append(activity)
                    allocateable_activity_ids.append({activity.id})
                elif activity.predecessors.issubset(allocated_ids):
                    allocateable_activities.append(activity)
                    allocateable_activity_ids.append({activity.id})
                else:
                    unallocateable_activities.append(activity)

            sorted_allocateable_activies = sorted(allocateable_activities, key=lambda x: len(x.predecessors))
            if len(activities) == unallocateable_activities:
                raise Exception("Unable to find allocation sequence")
            return self.get_allocation_sequence(
                unallocateable_activities,
                allocated_activities + sorted_allocateable_activies,
                allocated_ids.union(*allocateable_activity_ids),
            )
        else:
            return allocated_activities

    def calculate_latest_start(self) -> None:
        nodes = self.get_node_list_sorted_by_depth()
        nodes_reversed = [nodes[i] for i in range(len(nodes) - 1, -1, -1)]
        for node in nodes_reversed:
            latest_starts = [
                activity.end_node.latest_start - activity.duration
                for activity in node.outbound_activities
                if type(activity) == Activity
            ]
            latest_starts += [
                activity.end_node.latest_start
                for activity in node.outbound_activities
                if type(activity) == DummyActivity
            ]
            if latest_starts:
                node.latest_start = min(latest_starts)
            else:
                node.latest_start = node.earliest_start

            for activity in node.outbound_activities:
                if type(activity) == Activity:
                    activity.total_float = activity.end_node.latest_start - activity.duration - node.earliest_start
                    activity.free_float = activity.end_node.earliest_start - activity.duration - node.earliest_start

    def get_node_list_sorted_by_depth(self) -> List[Node]:
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

    def renumber_nodes(self) -> None:
        if not self.end_node:
            raise Exception("Undefined end_node")

        sorted_nodes: List[Node] = [self.start_node]
        for node in list(sorted(self.node_lut.values(), key=lambda x: x.max_depth)):
            if node.id != self.end_node.id:
                sorted_nodes.append(node)
        sorted_nodes.append(self.end_node)

        for index, node in enumerate(sorted_nodes):
            node.id = index

    def tie_end_node(self) -> None:
        end_nodes: NodeDict = NodeDict()
        tie_node: Node = Node(-1)
        max_depth: int = -1
        for id, node in self.node_lut.items():
            if not node.outbound_activities:
                end_nodes[id] = node
                if node.max_depth > max_depth:
                    max_depth = node.max_depth
                    tie_node = node

        if len(end_nodes) > 1:
            for id, node in end_nodes.items():
                if node.id != tie_node.id:
                    if self.have_common_ancestor(node, tie_node):
                        id_to_unlink = copy.deepcopy(tie_node.start_dependencies)
                        self.create_dummy_activity(node, tie_node)
                        self.node_lut.pop(id_to_unlink)
                    else:
                        for activity in node.inbound_activities:
                            tie_node.inbound_activities.append(activity)
                            activity.end_node = tie_node
                        if node.id != 0:
                            self.node_lut.pop(node.start_dependencies)

        self.end_node = tie_node

    def allocate_node_id(self) -> int:
        self.largest_node_id += 1
        return self.largest_node_id

    def attach_activity(self, activity: Activity, start_node: Node) -> Activity:
        end_node = Node(self.allocate_node_id(), max_depth=start_node.max_depth + 1)
        start_node.outbound_activities.append(activity)
        end_node.inbound_activities.append(activity)
        end_node.earliest_start = start_node.earliest_start + activity.duration
        activity.end_node = end_node
        activity.start_node = start_node
        self.node_lut[{activity.id}] = activity.end_node
        return activity

    def create_dummy_activity(self, start_node: Node, end_node: Node) -> Set[int]:
        dummy_activity = DummyActivity(
            start_node=start_node,
            end_node=end_node,
        )
        start_node.outbound_activities.append(dummy_activity)
        end_node.inbound_activities.append(dummy_activity)

        end_node.max_depth = max([start_node.max_depth + 1, end_node.max_depth])
        end_node.earliest_start = max([start_node.earliest_start, end_node.earliest_start])

        start_dependencies = []
        for activity in end_node.inbound_activities:
            if activity.start_node.id == 0 and type(activity) == Activity:
                start_dependencies.append({activity.id})
            else:
                start_dependencies.append(activity.start_node.start_dependencies)

        new_id = set.union(*start_dependencies)
        if new_id not in self.node_lut:
            self.node_lut[new_id] = end_node

        return new_id

    def allocate_multi_predessor_activity(self, activity: Activity) -> None:
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
            if len(predecessors) == 1:
                return ({predecessors[0]}, set())
            found = False
            largest_subset: Set[int] = set()
            non_end_nodes_of_largest_subset: Set[int] = set()
            subset: Set[int] = set()
            non_end_nodes: Set[int] = set()
            for subset in Network.power_subset(predecessors):
                if len(subset) > 1:
                    for activity in self.activities:
                        pred = activity.predecessors
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
                        pred = activity.predecessors
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

        predecessors = sorted(set(copy.deepcopy(activity.predecessors)))
        direct_link_start_node: Set[int] = set()
        dummy_link_start_nodes: List[Set[int]] = []
        while predecessors:
            subset, non_end_nodes = find_max_subset(predecessors)
            mergable_subset = subset.difference(non_end_nodes)
            if len(mergable_subset):
                if set.union(mergable_subset) not in self.node_lut:
                    self.merge_subset(mergable_subset)
                if direct_link_start_node:
                    dummy_link_start_nodes.append(mergable_subset)
                else:
                    direct_link_start_node = mergable_subset
                Network.remove_subset_from_list(predecessors, mergable_subset)
            else:
                break

        for subset in Network.power_subset(predecessors):
            if set(subset) in self.node_lut:
                if set(subset) <= set().union(*dummy_link_start_nodes):
                    continue
                dummy_link_start_nodes.append(set(subset))
                if set.union(*dummy_link_start_nodes) == set(predecessors):
                    self.minimal_viable_list_update(dummy_link_start_nodes)
                    break

        if direct_link_start_node:
            linked_start_node: Set[int] = direct_link_start_node
            for start_node in dummy_link_start_nodes:
                node_to_unlink = linked_start_node
                linked_start_node = self.create_dummy_activity(
                    self.node_lut[start_node],
                    self.node_lut[direct_link_start_node],
                )
                self.node_lut.pop(node_to_unlink)
            if linked_start_node:
                self.attach_activity(activity, self.node_lut[linked_start_node])
        elif dummy_link_start_nodes:
            floating_node = Node(self.allocate_node_id())
            for start_node in dummy_link_start_nodes:
                self.create_dummy_activity(self.node_lut[start_node], floating_node)

            self.attach_activity(activity, floating_node)
        else:
            self.attach_activity(activity, self.start_node)

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
        new_head: Set[int] = set()
        if tail:
            if self.have_common_ancestor(self.node_lut[head], self.node_lut[tail[0]]):
                new_head = self.create_dummy_activity(self.node_lut[tail[0]], self.node_lut[head])
                self.node_lut.pop(head)
            else:
                for activity in self.node_lut[tail[0]].inbound_activities:
                    activity.end_node = self.node_lut[head]
                    self.node_lut[head].inbound_activities.append(activity)
                new_head = head.union(tail[0])
                self.node_lut[new_head] = self.node_lut[head]
                self.node_lut.pop(tail[0])
                self.node_lut.pop(head)

            self.recursive_merge(new_head, tail[1:])


@dataclass
class CriticalPathFormatting:
    color: str = field(default="black")
    thickness: int = field(default=4)


class DummyActivityFormatting:
    style: str = field(default="dashed")


class PlantUml:
    def __init__(self, network: Network):
        self.plantuml: str = ""
        self.sorted_nodes = network.get_node_list_sorted_by_depth()

    def get_txt(self) -> str:
        return self._get_header() + self._get_map() + "\n" + self._get_network() + self._get_trailer()

    def write_txt(self, file: Path) -> None:
        with open(file, "w") as f:
            f.write(self.get_txt())

    def _get_header(self) -> str:
        return """@startuml PERT
left to right direction
' Horizontal lines: -->, <--, <-->
' Vertical lines: ->, <-, <->
title Pert: Project Design

"""

    def _get_trailer(self) -> str:
        return "\n@enduml"

    def _get_map(self) -> str:
        map_list = [
            f"""map {node.id} {{
    earliest start => {node.earliest_start}
    latest start => {node.latest_start}
}}"""
            for node in self.sorted_nodes
        ]
        return "\n".join(map_list)

    def _get_network(self) -> str:
        network = [
            f"{a.start_node.id} -{self._line_fmt(a)}-> {a.end_node.id} : {a.description} (Id={a.id}, D={a.duration}, TF={a.total_float}, FF={a.free_float})"
            if type(a) == Activity
            else f"{a.start_node.id} -{self._line_fmt(a)}-> {a.end_node.id}"
            for node in self.sorted_nodes
            for a in node.outbound_activities
        ]

        return "\n".join(network)

    def _line_fmt(self, activity: Union[Activity, DummyActivity]) -> str:
        if type(activity) == Activity:
            if activity.total_float == 0:
                return "[thickness=4]"
            else:
                return ""
        if type(activity) == DummyActivity:
            if activity.start_node.earliest_start == activity.end_node.latest_start:
                return "[dashed,thickness=4]"
            else:
                return "[dashed]"
        return ""


def main(file: Path) -> None:
    project = parse(file)
    annotate_with_duration(project)
    network = Network(get_activities(project))
    plantuml = PlantUml(network)
    # print(plantuml.get_txt())
    plantuml.write_txt(file.with_suffix(".txt"))


def get_activities(project: Dict[str, Any]) -> List[Activity]:
    return [
        Activity(
            id=yaml_activity[ID],
            duration=yaml_activity[DURATION],
            description=yaml_activity[ACTIVITY],
            predecessors=set(yaml_activity[PREDECESSORS]) if PREDECESSORS in yaml_activity else set(),
        )
        for yaml_activity in project[ACTIVITIES]
    ]


def get_resources(project: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if RESOURCES in project:
        return {entry["Id"]: entry for entry in project[RESOURCES]}
    return dict()


def annotate_with_duration(project: Dict[str, Any]):
    resources = get_resources(project)
    for activity in project[ACTIVITIES]:
        effort = activity[EFFORT]
        pensum = resources[activity[RESOURCE]][PENSUM] if resources else 1.0
        duration = math.ceil(effort / pensum)
        activity[DURATION] = duration


def parse(file: Path):
    with open(file, "r") as f:
        project = yaml.safe_load(f)
    return project


if __name__ == "__main__":
    # args = sys.argv[1:]
    logging.basicConfig(level=logging.WARN)
    main(Path("test_case_3.yaml"))
