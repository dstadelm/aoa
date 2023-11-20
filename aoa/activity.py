from dataclasses import dataclass, field
from typing import Set


@dataclass
class Activity:
    id: int
    duration: int = field(compare=False)
    description: str = field(compare=False)
    predecessors: Set[int] = field(default_factory=set, compare=False)
    total_float: int = field(default=0, compare=False)
    free_float: int = field(default=0, compare=False)
    float: int = field(default=0, compare=False)

    def __repr__(self):
        return f"    Activity {str(self.id)}"


@dataclass
class DummyActivity:
    id: int
    predecessors: Set[int] = field(default_factory=set, compare=False)
