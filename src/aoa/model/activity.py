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
    effort: int = field(default=0, eq=False)
    resource: str = field(default="", eq=False)
    owner: str = field(default="", eq=False)
    activity: str = field(default="", eq=False)
    predecessors: set[int] = field(factory=set, eq=False)
    state: State = field(default=State.OPEN, eq=False)
    total_float: int = field(default=0, eq=False)
    free_float: int = field(default=0, eq=False)
    float: int = field(default=0, eq=False)

    @override
    def __repr__(self):
        return f"    Activity {str(self.id)}"


@define
class DummyActivity:
    id: int
    predecessors: set[int] = field(factory=set, eq=False)
