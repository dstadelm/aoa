from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Union


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


@dataclass
class Activity:
    id: int
    duration: int = field(compare=False)
    description: str = field(compare=False)
    start_node: Node = field(default=Node(-1), compare=False)
    end_node: Node = field(default=Node(-1), compare=False)
    predecessors: Set[int] = field(default_factory=set, compare=False)
    total_float: int = field(default=0, compare=False)
    free_float: int = field(default=0, compare=False)
    float: int = field(default=0, compare=False)

    def __repr__(self):
        return f"    Activity {str(self.id)}"


@dataclass
class DummyActivity:
    start_node: Node
    end_node: Node

    def __repr__(self):
        return f"    Dummy Activity from start node {str(self.start_node.id)}"
