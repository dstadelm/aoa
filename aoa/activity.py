from dataclasses import dataclass, field
from typing import Set


@dataclass
class Activity:
    """The Activity class for the PERT and CPM.

    This class holds the data for a activity and provides derived values

    Args:
        id: unique ID of the activity
        effort: the amount of work that is required to accomplish the activity
        duration: the amount of time that is required to accomplish the activity (maybe it is done by an external party
                  and the effort is therefor zero)
        description: the description of the activity
        predecessors: a list of ids of preceding activities
        free_float: the amount of time an activity can be delayed without delaying any subsequent activity
        earliest_start: the earliest moment at which the activity can start
        latest_finish: the latest moment at which the activity must finish

    """

    id: int = field(default=-1, compare=True)
    effort: int = field(default=0, compare=False)
    duration: int = field(default=0, compare=False)
    description: str = field(default="Dummy Activity", compare=False)
    predecessors: Set[int] = field(default_factory=set, compare=False)
    free_float: int = field(default=0, compare=False)
    earliest_start: int = field(default=0, compare=False)
    latest_finish: int = field(default=0, compare=False)

    def __repr__(self):
        return f"    Activity {str(self.id)}"

    @property
    def earliest_finish(self):
        """The earliest moment at which the activity can be finished."""
        return self.earliest_start + self.duration

    @property
    def latest_start(self):
        """The latest moment at which the activity is allowed to start without delaying the project"""
        return self.latest_finish - self.duration

    @property
    def critical(self) -> bool:
        """Returns true if the activity is on the critical path"""
        return self.latest_finish == self.earliest_finish

    @property
    def total_float(self) -> int:
        """The amount of time an activity can be delayed without delaying the end of the project"""
        return self.latest_finish - self.earliest_finish

    def update_earliest_start(self, contender: int) -> None:
        if contender > self.earliest_start:
            self.earliest_start = contender
