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
    actual_effort: float = field(default=0.0, eq=False)
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

    Overconstraining exists when an activity has a direct predecessor that is also transitively reachable via
    another of its predecessors. Assumes the graph is acyclic (call ``check_no_cycles_exist`` first).

    Arguments:
        activities (ActivityCollection): The activities to check
    """
    transitive: dict[int, set[int]] = {}

    def ancestors(aid: int) -> set[int]:
        if aid in transitive:
            return transitive[aid]
        acc: set[int] = set()
        for p in activities[aid].predecessors:
            acc.add(p)
            acc |= ancestors(p)
        transitive[aid] = acc
        return acc

    for activity in activities.values():
        for direct_pred in activity.predecessors:
            indirect = ancestors(direct_pred)
            redundant = indirect & activity.predecessors
            if redundant:
                raise AllocationException(
                    f"Downstream activities {redundant} detected as direct predecessors of activity {activity.id}"
                )


def check_no_cycles_exist(activities: ActivityCollection) -> None:
    """Check that no cycles exist in the provided activities.

    Performs an iterative DFS using two sets:
      * ``visited`` — nodes fully explored; revisiting them is fine (diamond DAGs).
      * ``on_stack`` — nodes on the current DFS path; revisiting one indicates a cycle.

    When a back-edge is detected, the exact cycle path is extracted from the DFS stack for reporting.

    Arguments:
        activities (ActivityCollection): The activities to check
    """
    visited: set[int] = set()

    for root in activities.values():
        if root.id in visited:
            continue

        # Stack entries: (activity_id, iterator over its predecessors)
        stack: list[tuple[int, "iter"]] = [(root.id, iter(activities[root.id].predecessors))]
        on_stack: set[int] = {root.id}
        path: list[int] = [root.id]

        while stack:
            current_id, pred_iter = stack[-1]
            try:
                pred_id = next(pred_iter)
            except StopIteration:
                stack.pop()
                on_stack.discard(current_id)
                path.pop()
                visited.add(current_id)
                continue

            if pred_id in on_stack:
                # Found a back-edge: cycle is path[path.index(pred_id):] + [pred_id]
                start_idx = path.index(pred_id)
                cycle_ids = sorted(path[start_idx:])
                message = ", ".join(f"ID[{aid}]" for aid in cycle_ids)
                raise AllocationException(f"Cycle detected in the network involving activities {message}")

            if pred_id in visited:
                continue

            stack.append((pred_id, iter(activities[pred_id].predecessors)))
            on_stack.add(pred_id)
            path.append(pred_id)


def prune_involved(involved: set[int], activities: ActivityCollection) -> set[int]:
    """Prune the involved set by iteratively removing activities whose predecessors are all outside the set.

    Only activities whose remaining predecessors are all within `involved` can participate in a cycle among
    `involved`. Anything else is trimmed. Iterates until stable. Does not mutate the underlying activities.

    Arguments:
        involved (set[int]): The set of involved activity ids
        activities (ActivityCollection): Lookup for activities by id
    Returns:
        set[int]: The pruned set of involved activity ids
    """
    pruned = set(involved)
    while True:
        removable = {a for a in pruned if not (activities[a].predecessors & pruned)}
        if not removable:
            return pruned
        pruned -= removable


def check_for_unique_ids(activities: list[Activity]) -> None:
    """Checks that all activity IDs are unique. Raises a ValueError if duplicate IDs are found."""
    activity_ids = [activity.id for activity in activities]
    if len(activity_ids) != len(set(activity_ids)):
        duplicate_ids = [
            "ID[" + str(activity_id) + "]" for activity_id in set(activity_ids) if activity_ids.count(activity_id) > 1
        ]
        raise NonUniqueIdException("Activity IDs must be unique. Duplicate IDs found: " + ", ".join(duplicate_ids))
