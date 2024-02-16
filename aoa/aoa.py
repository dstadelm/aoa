#!/usr/bin/env python3
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import yaml
from activity import Activity
from network import Network
from plantuml import PlantUml

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


def main(file: Path) -> None:
    project = parse(file)
    annotate_with_duration(project)
    network = Network(get_activities(project))
    plantuml = PlantUml(network)
    # print(plantuml.get_txt())
    plantuml.write_txt(file.with_suffix(".md"))


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
    main(Path("../AoA.yaml"))
    # main(Path("tricky.yaml"))
    # main(Path("more_tricky.yaml"))
    # main(Path("test_case_3.yaml"))
    # main(Path("test_case_5.yaml"))
