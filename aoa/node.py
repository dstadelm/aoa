from dataclasses import dataclass, field
from typing import List, Set, Union

from activity import Activity, DummyActivity


@dataclass
class Node:
    id: int
    earliest_start: int = field(default=0, compare=False)
    latest_start: int = field(default=0, compare=False)
    inbound_activities: List[Union[Activity, DummyActivity]] = field(default_factory=list, repr=False, compare=False)
    outbound_activities: List[Union[Activity, DummyActivity]] = field(default_factory=list, repr=False, compare=False)
    max_depth: int = field(default=0, compare=False)
    start_dependencies: Set[int] = field(default_factory=set)

    def __post_init__(self):
        pass  # place init logging stuff here
