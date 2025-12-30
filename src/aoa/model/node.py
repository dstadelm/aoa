from dataclasses import dataclass, field
from typing import override

from .activity import Activity, DummyActivity


@dataclass
class Node:
    id: int
    is_end: bool = field(default=False, compare=False)
    inbound_activities: list[Activity | DummyActivity] = field(default_factory=list, repr=False, compare=False)
    outbound_activities: list[Activity | DummyActivity] = field(default_factory=list, repr=False, compare=False)
    max_depth: int = field(default=0, compare=False)
    start_dependencies: set[int] = field(default_factory=set)

    @override
    def __str__(self) -> str:
        if not self.start_dependencies:
            return "start"
        elif self.is_end:
            return "end"
        else:
            return "-".join(str(v) for v in sorted(self.start_dependencies))

    def __post_init__(self):
        pass  # place init logging stuff here
