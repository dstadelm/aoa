#!/usr/bin/env python3
from __future__ import annotations

import logging
import math
import timeit
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import pygraphviz as pgvz
import yaml
from matplotlib import image as mpimg
from matplotlib import pyplot as plt

import networkx as nx
from aoa.activity import Activity
from aoa.dot import set_dot_attributes
from aoa.network import Network

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
class CriticalPathFormatting:
    color: str = field(default="black")
    thickness: int = field(default=4)


class DummyActivityFormatting:
    style: str = field(default="dashed")


class Runnable:
    def __init__(self, file: Path):
        self.file = file

    def __call__(self):
        parse_time = timeit.Timer(self.parse).timeit(1)
        annotation_time = timeit.Timer(self.annotate).timeit(1)
        extract_activities = timeit.Timer(self.extract_activities).timeit(1)
        create_graph = timeit.Timer(self.create_graph).timeit(1)
        create_dot = timeit.Timer(self.create_dot).timeit(1)
        layout_dot = timeit.Timer(self.layout_dot).timeit(1)
        display_dot = timeit.Timer(self.display_dot).timeit(1)

        print("parse_time " + str(parse_time))
        print("annotation_time " + str(annotation_time))
        print("extract_activities " + str(extract_activities))
        print("create_graph " + str(create_graph))
        print("create_dot " + str(create_dot))
        print("layout_dot " + str(layout_dot))
        print("display_dot " + str(display_dot))

    def parse(self):
        self.project = parse(self.file)

    def annotate(self):
        annotate_with_duration(self.project)

    def extract_activities(self):
        self.activities = get_activities(self.project)

    def create_graph(self):
        self.network = Network(self.activities)

    def create_dot(self):
        set_dot_attributes(self.network.graph)
        self.gvz: pgvz.AGraph = nx.nx_agraph.to_agraph(self.network.graph)

    def layout_dot(self):
        self.gvz.layout(prog="dot", args="-Nshape=Mrecord -Nrankdir=LR")

    def display_dot(self):
        self.gvz.draw(self.file.with_suffix(".png"))
        self.gvz.draw(self.file.with_suffix(".svg"))
        image = mpimg.imread(self.file.with_suffix(".png"))
        plt.title(str(self.file.with_suffix("")))
        plt.axis("off")
        plt.imshow(image)
        plt.show()


def main(file: Path) -> None:
    project = parse(file)
    annotate_with_duration(project)
    network = Network(get_activities(project))
    set_dot_attributes(network.graph)
    gvz: pgvz.AGraph = nx.nx_agraph.to_agraph(network.graph)
    # gvz.graph_attr["rankdir"] = "LR"

    gvz.layout(prog="dot", args="-Nshape=Mrecord -Nrankdir=LR")
    # dot = gvz.string()
    gvz.draw(file.with_suffix(".png"))
    gvz.draw(file.with_suffix(".svg"), format="svg")
    image = mpimg.imread(file.with_suffix(".png"))
    plt.title(str(file.with_suffix("")))
    plt.axis("off")
    plt.imshow(image)
    plt.show()

    # plantuml = PlantUml(network)
    # # print(plantuml.get_txt())
    # plantuml.write_txt(file.with_suffix(".txt"))


def get_activities(project: Dict[str, Any]) -> List[Activity]:
    return [
        Activity(
            id=yaml_activity[ID],
            effort=yaml_activity[EFFORT],
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
