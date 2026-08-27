import pytest

from aoa.model.activity import Activity, ActivityCollection
from aoa.model.exception import AllocationException, NonUniqueIdException


def test_validate_unique_activity_ids() -> None:
    activities = [
        Activity(id=1),
        Activity(id=1),
    ]

    with pytest.raises(NonUniqueIdException, match=r"Activity IDs must be unique. Duplicate IDs found: ID\[1]"):
        _ = ActivityCollection(activities)


def test_validate_cycle_detection() -> None:
    activities = [
        Activity(id=0),
        Activity(id=1, predecessors={0, 3}),
        Activity(id=2, predecessors={1}),
        Activity(id=3, predecessors={2}),
    ]

    with pytest.raises(
        AllocationException, match=r"Cycle detected in the network involving activities ID\[1], ID\[2], ID\[3]"
    ):
        _ = ActivityCollection(activities)


def test_overconstraining() -> None:

    activities = [
        Activity(
            id=1,
            planned_effort=10,
        ),
        Activity(
            id=2,
            planned_effort=1,
        ),
        Activity(
            id=3,
            planned_effort=1,
            predecessors={2},
        ),
        Activity(
            id=4,
            planned_effort=1,
            predecessors={3},
        ),
        Activity(
            id=5,
            planned_effort=1,
            predecessors={2, 3},
        ),
    ]

    with pytest.raises(AllocationException):
        _ = ActivityCollection(activities)
