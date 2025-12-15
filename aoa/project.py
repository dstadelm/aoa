from datetime import date
from pathlib import Path
from typing import TypeAlias

import cattrs
import yaml
from attr import define, field

from .activity import Activity
from .milestones import Milestone
from .resources import Resource
from .state import State


def date_hook(date: date, _: type) -> date:
    return date


def status_hook(status: str, _: type) -> State:
    return State[status.upper()]


cattrs.register_structure_hook(date, date_hook)
cattrs.register_structure_hook(State, status_hook)

ProjectDictType: TypeAlias = dict[str, list[dict[str, str | date | int | float | list[int] | list[date]]] | date]


@define
class Project:
    start: date = field(default=date.today())
    milestones: list[Milestone] = field(factory=list)
    activities: list[Activity] = field(factory=list)
    resources: list[Resource] = field(factory=list)
    _activity_index: dict[int, Activity] = field(init=False, factory=dict, repr=False)
    _resource_index: dict[str, Resource] = field(init=False, factory=dict, repr=False)

    def get_activity_by_id(self, activity_id: int) -> Activity | None:
        if self._activity_index == {}:
            self._activity_index = {activity.id: activity for activity in self.activities}
        return self._activity_index.get(activity_id)

    def get_resource_by_id(self, resource_id: str) -> Resource | None:
        if self._resource_index == {}:
            self._resource_index = {resource.id: resource for resource in self.resources}
        return self._resource_index.get(resource_id)


def deserialize_project(project_dict: ProjectDictType) -> Project:
    print(project_dict)
    return cattrs.structure(project_dict, Project)


def load_yaml_project(config: Path) -> Project:
    with open(config, "r") as f:
        _yaml_project: ProjectDictType = yaml.safe_load(f)  # pyright: ignore [reportAny]
        return deserialize_project(_yaml_project)
