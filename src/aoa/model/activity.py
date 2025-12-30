from typing import override

from attr import define, field

from .state import State


@define
class ActivityProtocol:
    id: int
    predecessors: set[int]


@define
class Activity:
    id: int
    effort: float = field(default=0, eq=False)
    resource: str = field(default="", eq=False)
    owner: str = field(default="", eq=False)
    activity: str = field(default="", eq=False)
    predecessors: set[int] = field(factory=set, eq=False)
    state: State = field(default=State.OPEN, eq=False)
    earliest_start: float = field(default=0, eq=False)
    latest_finish: float = field(default=0, eq=False)
    total_float: int = field(default=0, eq=False)
    free_float: int = field(default=0, eq=False)

    @override
    def __repr__(self):
        return f"    Activity {str(self.id)}"

    @property
    def earliest_finish(self) -> float:
        return self.earliest_start + self.effort

    @property
    def latest_start(self) -> float:
        return self.latest_finish - self.effort

    @property
    def duration(self) -> float:
        return self.effort


@define
class DummyActivity:
    id: int
    predecessors: set[int] = field(factory=set, eq=False)
    effort: float = field(default=0, eq=False)
    duration: float = field(default=0, eq=False)
    earliest_start: float = field(default=0, eq=False)
    latest_finish: float = field(default=0, eq=False)

    @property
    def earliest_finish(self) -> float:
        return self.earliest_start

    @property
    def latest_start(self) -> float:
        return self.latest_finish
