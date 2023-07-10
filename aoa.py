#!/usr/bin/env python3
from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

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

    def __init__(self, activities: List[Activity]):
        self.largest_node_id = -1
        self.node_lut: NodeDict = NodeDict()
        self.reverse_predecessor_lut: Dict[int, List[Set[int]]] = dict()
        self.activities = copy.deepcopy(activities)
        self.start_node: Node = Node(self.allocate_node_id())
        self.end_node: Optional[Node] = None

        allocation_sequence = self.get_allocation_sequence(activities, list(), set())
        for activity in allocation_sequence:
            self.allocate_activity(activity)
        self.tie_end_node()
        self.renumber_nodes()
        self.calculate_latest_start()

    def get_sets_that_contain_ids_in_set(self, id_set: Set[int]) -> List[Set[int]]:
        if not self.reverse_predecessor_lut:
            for activity in self.activities:
                for id in activity.predecessors:
                    self.reverse_predecessor_lut.setdefault(id, []).append(activity.predecessors)

        return [subset for id in id_set for subset in self.reverse_predecessor_lut[id]]

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

        del end_nodes[tie_node.start_dependencies]

        for id, node in end_nodes.items():
            if self.have_common_ancestor(node, tie_node):
                self.create_dummy_activity(node, tie_node)
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

        if end_node.start_dependencies in self.node_lut:
            if self.node_lut[end_node.start_dependencies].id == end_node.id:
                self.node_lut.pop(end_node.start_dependencies)

        end_node.start_dependencies = end_node.start_dependencies.union(start_node.start_dependencies)

        if end_node.start_dependencies not in self.node_lut:
            self.node_lut[end_node.start_dependencies] = end_node

        return end_node.start_dependencies

    def create_start_node(self, predecessors: Set[int]) -> Optional[Set[int]]:
        if predecessors in self.node_lut:
            return predecessors

        mutable_node_id = predecessors.copy()
        for pred_sets in self.get_sets_that_contain_ids_in_set(predecessors):
            if not predecessors.issubset(pred_sets):
                mutable_node_id.difference_update(pred_sets)
            if not mutable_node_id:
                return None

        if mutable_node_id:
            self.merge_subset(mutable_node_id)
            return mutable_node_id
        else:
            return None

    def allocate_activity(self, activity: Activity) -> None:
        predecessors = activity.predecessors.copy()

        # if it exists find the node to which all predecessors can be bound
        if tie_node_id := self.create_start_node(predecessors.copy()):
            predecessors.difference_update(tie_node_id)

        dummy_link_start_nodes: List[Set[int]] = list()
        # find subsets that can be created from existing sub-subsets
        for subset in Network.power_subset(list(predecessors))[1:]:
            if dummy_link_start_node := self.create_start_node(predecessors.copy()):
                dummy_link_start_nodes.append(dummy_link_start_node)
                predecessors.difference_update(dummy_link_start_node)
            if not predecessors:
                break

        # find existing subsets
        for subset in Network.power_subset(list(predecessors)):
            if subset in self.node_lut:
                dummy_link_start_nodes.append(subset)
                predecessors.difference_update(subset)
            if not predecessors:
                break

        dummy_link_start_nodes = self.minimal_viable_list(dummy_link_start_nodes)

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

    def minimal_viable_list(self, los: List[Set[int]]) -> List[Set[int]]:
        los = sorted(los, key=lambda x: len(x))
        return self.mvl_recursion(los, [])

    def mvl_recursion(self, start: List[Set[int]], result: List[Set[int]]) -> List[Set[int]]:
        if not start:
            return result
        result_union = self.get_union(result)
        target = result_union.union(self.get_union(start))
        if result_union.union(self.get_union(start[1:])) != target:
            result.append(start[0])
        return self.mvl_recursion(start[1:], result)

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
    logger = logging.getLogger(__name__)
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(logging.DEBUG)
    main(Path("AoA.yaml"))
    # main(Path("tricky.yaml"))
    # main(Path("more_tricky.yaml"))
    # main(Path("test_case_3.yaml"))
    # main(Path("test_case_4.yaml"))
