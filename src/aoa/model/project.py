from datetime import date
from pathlib import Path
from typing import TypeAlias

import cattrs
import yaml
from attr import define, field

from .activity import Activity, ActivityCollection
from .exception import NonUniqueIdException
from .milestones import Milestone, MilestoneCollection
from .resources import Resource, ResourceCollection
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

    def get_activities(self) -> ActivityCollection:
        return ActivityCollection(activities=self.activities)

    def get_resources(self) -> ResourceCollection:
        return ResourceCollection(resources=self.resources)

    def get_milestones(self) -> MilestoneCollection:
        return MilestoneCollection(milestones=self.milestones)


def check_for_unique_activity_ids(activities: list[Activity]) -> None:
    activity_ids = [activity.id for activity in activities]
    if len(activity_ids) != len(set(activity_ids)):
        duplicate_ids = [activity_id for activity_id in set(activity_ids) if activity_ids.count(activity_id) > 1]
        raise NonUniqueIdException(
            "Activity IDs must be unique. Duplicate IDs found: " + ", ".join(str(id) for id in duplicate_ids)
        )


def deserialize_project(project_dict: ProjectDictType) -> Project:
    project = cattrs.structure(project_dict, Project)
    check_for_unique_activity_ids(activities=project.activities)
    return project


def load_yaml_project(config: Path) -> Project:
    with open(config, "r") as f:
        _yaml_project: ProjectDictType = yaml.safe_load(f)  # pyright: ignore [reportAny]
        return deserialize_project(_yaml_project)


def _serialize_project(project: Project) -> ProjectDictType:
    """Convert a Project back into a plain dict suitable for YAML serialization."""
    result: ProjectDictType = {}

    result["start"] = project.start

    if project.resources:
        result["resources"] = [
            {
                k: v
                for k, v in {
                    "id": r.id,
                    "name": r.name,
                    "workload": r.workload,
                    "weekdays": r.weekdays,
                    "holidays": r.holidays if r.holidays else None,
                }.items()
                if v is not None and v != "" and v != []
            }
            for r in project.resources
        ]

    if project.milestones:
        result["milestones"] = [
            {
                k: v
                for k, v in {
                    "id": m.id,
                    "description": m.description,
                    "owner": m.owner,
                    "due_date": m.due_date,
                    "state": m.state.name,
                }.items()
                if v is not None and v != ""
            }
            for m in project.milestones
        ]

    if project.activities:
        result["activities"] = [
            {
                k: v
                for k, v in {
                    "id": a.id,
                    "activity": a.activity,
                    "predecessors": sorted(a.predecessors) if a.predecessors else None,
                    "planned_effort": a.planned_effort if a.planned_effort else None,
                    "actual_effort": a.actual_effort if a.actual_effort else None,
                    "owner": a.owner,
                    "resource": a.resource,
                    "state": a.state.name if a.state != State.OPEN else None,
                }.items()
                if v is not None and v != "" and v != 0
            }
            for a in project.activities
        ]

    return result


def save_yaml_project(project: Project, config: Path) -> None:
    """Serialize a Project and write it to a YAML file.

    Arguments:
        project: The project to serialize.
        config: The file path to write the YAML output to.
    """
    project_dict = _serialize_project(project)
    with open(config, "w") as f:
        yaml.dump(project_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
