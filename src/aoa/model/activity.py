from dataclasses import dataclass
from typing import Generator, override

from attr import define, field

from .exception import NonUniqueIdException
from .id_generator import id_generator
from .state import State


@define
class ActivityProtocol:
    id: int
    predecessors: set[int]
    effort: float
    duration: float
    earliest_start: float
    latest_finish: float
    ealiest_finish: float
    latest_start: float
    total_float: float
    free_float: float


@define
class Activity:
    id: int
    effort: float = field(default=0.0, eq=False)
    resource: str = field(default="", eq=False)
    owner: str = field(default="", eq=False)
    activity: str = field(default="", eq=False)
    predecessors: set[int] = field(factory=set, eq=False)
    state: State = field(default=State.OPEN, eq=False)
    earliest_start: float = field(default=0.0, eq=False)
    latest_finish: float = field(default=0.0, eq=False)
    free_float: float = field(default=0.0, eq=False)

    @override
    def __repr__(self):
        return f"    Activity {str(self.id)}"

    @property
    def earliest_finish(self) -> float:
        return self.earliest_start + self.duration

    @property
    def latest_start(self) -> float:
        return self.latest_finish - self.duration

    @property
    def duration(self) -> float:
        return self.effort

    @property
    def total_float(self) -> float:
        return self.latest_start - self.earliest_start

    @property
    def critical(self) -> bool:
        """Returns true if the activity is on the critical path"""
        return self.latest_finish == self.earliest_finish

    @property
    def is_dummy(self) -> bool:
        """Returns true if the activity is a dummy activity (i.e., has zero effort and no resource)"""
        return self.id < 0


@dataclass
class ActivityCollection(dict[int, Activity]):
    def __init__(self, activities: list[Activity]):
        check_for_unique_ids(activities)
        self._dummy_activity_id_generator: Generator[int, None, None] = id_generator(start=0, increment=-1)
        self._dummy_activities: list[Activity] = []
        super().__init__((activity.id, activity) for activity in activities)

    def new_dummy_activity(self) -> Activity:
        da = Activity(next(self._dummy_activity_id_generator))
        self._dummy_activities.append(da)
        return da


def check_for_unique_ids(activities: list[Activity]) -> None:
    """Checks that all activity IDs are unique. Raises a ValueError if duplicate IDs are found."""
    activity_ids = [activity.id for activity in activities]
    if len(activity_ids) != len(set(activity_ids)):
        duplicate_ids = [
            "ID[" + str(activity_id) + "]" for activity_id in set(activity_ids) if activity_ids.count(activity_id) > 1
        ]
        raise NonUniqueIdException("Activity IDs must be unique. Duplicate IDs found: " + ", ".join(duplicate_ids))
