import pytest

from aoa.model.activity import Activity, ActivityCollection
from aoa.model.exception import NonUniqueIdException


def test_validate_unique_activity_ids() -> None:
    activities = [
        Activity(id=1),
        Activity(id=1),
    ]

    with pytest.raises(NonUniqueIdException, match=r"Activity IDs must be unique. Duplicate IDs found: ID\[1]"):
        _ = ActivityCollection(activities)
