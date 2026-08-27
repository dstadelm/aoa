from datetime import datetime

import pytest

from aoa.model.project import ProjectDictType, deserialize_project
from aoa.model.state import State


def test_load_start_date() -> None:
    project_dict: ProjectDictType = {"start": datetime(2023, 10, 1).date()}
    project = deserialize_project(project_dict)
    assert project.start == datetime(2023, 10, 1).date()


def test_load_milestone() -> None:
    project_dict: ProjectDictType = {
        "milestones": [
            {
                "id": "MS 010",
                "description": "description",
                "owner": "owner",
                "due_date": datetime(2024, 3, 16).date(),
                "state": "DONE",
            }
        ]
    }

    project = deserialize_project(project_dict)
    for p in project.milestones:
        print(p)
    assert project.milestones[0].id == "MS 010"
    assert project.milestones[0].description == "description"
    assert project.milestones[0].owner == "owner"
    assert project.milestones[0].due_date == datetime(2024, 3, 16).date()
    assert project.milestones[0].state.name == "DONE"


def test_load_milestones() -> None:
    project_dict: ProjectDictType = {
        "milestones": [
            {
                "id": "MS 010",
                "description": "description1",
                "owner": "owner1",
                "due_date": datetime(2024, 3, 16).date(),
                "state": "DONE",
            },
            {
                "id": "MS 020",
                "description": "description2",
                "owner": "owner2",
                "due_date": datetime(2024, 4, 20).date(),
                "state": "IN_PROGRESS",
            },
        ]
    }

    project = deserialize_project(project_dict)
    assert len(project.milestones) == 2

    milestone1 = project.milestones[0]
    assert milestone1.id == "MS 010"
    assert milestone1.description == "description1"
    assert milestone1.owner == "owner1"
    assert milestone1.due_date == datetime(2024, 3, 16).date()
    assert milestone1.state.name == "DONE"

    milestone2 = project.milestones[1]
    assert milestone2.id == "MS 020"
    assert milestone2.description == "description2"
    assert milestone2.owner == "owner2"
    assert milestone2.due_date == datetime(2024, 4, 20).date()
    assert milestone2.state.name == "IN_PROGRESS"


def test_load_activity() -> None:
    project_dict: ProjectDictType = {
        "activities": [
            {
                "id": 1,
                "activity": "activity 1",
                "state": "DONE",
                "planned_effort": 40,
                "owner": "owner",
                "resource": "ressource",
            }
        ]
    }

    project = deserialize_project(project_dict)
    assert len(project.activities) == 1

    activity = project.activities[0]
    assert activity.id == 1
    assert activity.activity == "activity 1"
    assert activity.state == State.DONE
    assert activity.planned_effort == 40
    assert activity.owner == "owner"
    assert activity.resource == "ressource"
    assert activity.predecessors == set()


def test_load_activities() -> None:

    project_dict: ProjectDictType = {
        "activities": [
            {
                "id": 1,
                "activity": "activity 1",
                "state": "DONE",
                "planned_effort": 40,
                "owner": "owner_a",
                "resource": "ressource_a",
            },
            {
                "id": 2,
                "predecessors": [1],
                "activity": "activity 2",
                "state": "IN_PROGRESS",
                "planned_effort": 5,
                "owner": "owner_b",
                "resource": "resource_b",
            },
        ]
    }

    project = deserialize_project(project_dict)
    assert len(project.activities) == 2

    activity_1 = project.activities[0]
    assert activity_1.id == 1
    assert activity_1.activity == "activity 1"
    assert activity_1.state == State.DONE
    assert activity_1.planned_effort == 40
    assert activity_1.owner == "owner_a"
    assert activity_1.resource == "ressource_a"
    assert activity_1.predecessors == set()

    activity_2 = project.activities[1]
    assert activity_2.id == 2
    assert activity_2.activity == "activity 2"
    assert activity_2.state == State.IN_PROGRESS
    assert activity_2.planned_effort == 5
    assert activity_2.owner == "owner_b"
    assert activity_2.resource == "resource_b"
    assert activity_2.predecessors == {1}


def test_load_resource() -> None:
    project_dict: ProjectDictType = {
        "resources": [{"id": "TJ", "name": "Tom and Jerry", "weekdays": "1110100", "workload": 0.8, "holidays": []}]
    }
    project = deserialize_project(project_dict)
    assert len(project.resources) == 1

    resource = project.resources[0]
    assert resource.id == "TJ"
    assert resource.name == "Tom and Jerry"
    assert resource.workload == "0.8"
    assert resource.weekdays == "1110100"
    assert resource.holidays == []


def test_load_resources() -> None:
    project_dict: ProjectDictType = {
        "resources": [
            {"id": "TJ", "name": "Tom and Jerry", "weekdays": "1110100", "workload": 0.8, "holidays": []},
            {
                "id": "SC",
                "name": "Sylvester and Cat",
                "weekdays": "1111100",
                "workload": 1.0,
                "holidays": [datetime(2024, 12, 25).date(), datetime(2024, 12, 26).date()],
            },
        ]
    }
    project = deserialize_project(project_dict)
    assert len(project.resources) == 2

    resource_1 = project.resources[0]
    assert resource_1.id == "TJ"
    assert resource_1.name == "Tom and Jerry"
    assert resource_1.workload == "0.8"
    assert resource_1.weekdays == "1110100"
    assert resource_1.holidays == []

    resource_2 = project.resources[1]
    assert resource_2.id == "SC"
    assert resource_2.name == "Sylvester and Cat"
    assert resource_2.workload == "1.0"
    assert resource_2.weekdays == "1111100"
    assert resource_2.holidays == ["2024-12-25", "2024-12-26"]
