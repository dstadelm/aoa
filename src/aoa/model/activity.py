from collections.abc import Generator
from dataclasses import dataclass
from functools import cached_property
from typing import override

from attr import define, field

from .exception import AllocationException, NonUniqueIdException
from .id_generator import id_generator
from .state import State


@define
class ActivityProtocol:
    id: int
    predecessors: set[int]
    planned_effort: float
    earliest_start: float
    latest_finish: float
    earliest_finish: float
    latest_start: float
    total_float: float
    free_float: float


@define
class Activity:
    id: int
    planned_effort: float = field(default=0.0, eq=False)
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
        return self.planned_effort

    @property
    def total_float(self) -> float:
        return self.latest_start - self.earliest_start

    @property
    def critical(self) -> bool:
        """Returns true if the activity is on the critical path"""
        return self.latest_finish == self.earliest_finish

    @property
    def is_dummy(self) -> bool:
        """Returns true if the activity is a dummy activity (i.e., has zero planned effort and no resource)"""
        return self.id < 0


@dataclass
class ActivityCollection(dict[int, Activity]):
    def __init__(self, activities: list[Activity]):
        check_for_unique_ids(activities)
        self._dummy_activity_id_generator: Generator[int, None, None] = id_generator(start=0, increment=-1)
        super().__init__((activity.id, activity) for activity in activities)
        check_no_cycles_exist(self)
        check_for_overconstraining(self)

    def new_dummy_activity(self) -> Activity:
        da = Activity(next(self._dummy_activity_id_generator))
        self[da.id] = da
        return da

    @cached_property
    def reverse_predecessor_lut(self) -> dict[int, list[set[int]]]:
        _reverse_predecessor_lut: dict[int, list[set[int]]] = dict()
        for activity in self.values():
            for id in activity.predecessors:
                _reverse_predecessor_lut.setdefault(id, []).append(activity.predecessors)
        return _reverse_predecessor_lut


def check_for_overconstraining(activities: ActivityCollection) -> None:
    """Check that no overconstraining exists in the provided activities.

    Overconstraining exists when an activity is a direct predecessor of another activity but also a downstream predecessor of the same activity.

    Arguments:
        activities (list[Activity]): The list of activities to check
    """
    downstream_predecessor_lut: dict[int, set[int]] = dict()

    for activity in activities.values():
        # update downstream predecessors
        for predecessor_id in activity.predecessors:

            downstream_predecessor_lut.setdefault(predecessor_id, set()).update(activities[predecessor_id].predecessors)
            downstream_predecessor_lut.setdefault(predecessor_id, set()).update(
                downstream_predecessor_lut.get(predecessor_id, set())
            )
        intersection = downstream_predecessor_lut.get(activity.id, set()).intersection(activity.predecessors)
        if intersection:
            raise AllocationException(
                f"Downstream activities {intersection} detected as direct predecessors of activity {activity.id}"
            )


def check_no_cycles_exist(activities: ActivityCollection) -> None:
    """Check that no cycles exist in the provided activities.

    A cycle exists when an activity is both a predecessor and a downstream predecessor of another activity.

    Arguments:
        activities (list[Activity]): The list of activities to check
    """
    for activity in activities.values():
        visited: set[int] = set()

        def visit(act: Activity) -> None:
            if act.id in visited:
                pruned = prune_involved(visited, activities)
                message = ", ".join([f"ID[{id}]" for id in pruned])
                raise AllocationException(f"Cycle detected in the network involving activities {message}")
            visited.add(act.id)
            for pred_id in act.predecessors:
                visit(activities[pred_id])

        visit(activity)


def prune_involved(involved: set[int], activities: ActivityCollection) -> set[int]:
    """Prune the involved set by removing activities that have no predecessors recursively.

    Given activities A, B, C, D and E where A has no predecessors, B has A as predecessor C has B and E as
    predecessor, D has C as predecessor and E has D as predecessor. The involved set is {A, B, C, D, E}. Pruning
    this set will remove A first as it has no predecessors resulting in the set {B, C, D, E}. B will then be
    modified to by removing A as predecessor resulting in B having no predecessors. Pruning will then remove B
    resulting in the set {C, D, E}. B will be removed from C but C will still have E as predecessor so it won't be
    removed. The final pruned set will be {C, D, E}.

    Arguments:
        involved (set[int]): The set of involved activity ids
        activity_dict (dict[int, Activity]): The dictionary of activities by id
    Returns:
        set[int]: The pruned set of involved activity ids

    """
    pruned = involved.copy()
    for act_id in involved:
        activity = activities[act_id]
        if not activity.predecessors:
            pruned.discard(act_id)
            for activity in activities.values():
                activity.predecessors.discard(act_id)

    if pruned != involved:
        return prune_involved(pruned, activities)
    else:
        return pruned


def check_for_unique_ids(activities: list[Activity]) -> None:
    """Checks that all activity IDs are unique. Raises a ValueError if duplicate IDs are found."""
    activity_ids = [activity.id for activity in activities]
    if len(activity_ids) != len(set(activity_ids)):
        duplicate_ids = [
            "ID[" + str(activity_id) + "]" for activity_id in set(activity_ids) if activity_ids.count(activity_id) > 1
        ]
        raise NonUniqueIdException("Activity IDs must be unique. Duplicate IDs found: " + ", ".join(duplicate_ids))
