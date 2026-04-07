"""Node model for Activity-on-Arrow (AoA) network diagrams.

Nodes represent events (points in time) in the network. Activities are
attached to nodes via their inbound and outbound activity lists, which
are maintained by :class:`NodeDict` through explicit method calls.
"""

from dataclasses import dataclass, field
from typing import override

from .activity import Activity


@dataclass
class Node:
    """A node (event) in the AoA network.

    Attributes:
        _id: The unique identifier for this node.
        inbound_activities: Activities arriving at this node.
        outbound_activities: Activities departing from this node.
        max_depth: The maximum topological depth of this node (used for ordering).
    """

    _id: int
    inbound_activities: list[Activity] = field(default_factory=list, repr=False, compare=False)
    outbound_activities: list[Activity] = field(default_factory=list, repr=False, compare=False)
    max_depth: int = field(default=0, compare=False)

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        self._id = value

    @property
    def is_end(self) -> bool:
        """Return True if this node has no outbound activities."""
        return len(self.outbound_activities) == 0

    @property
    def start_dependencies(self) -> set[int]:
        """Return the set of activity IDs that this node depends on.

        For real activities (id >= 0), the activity ID itself is a dependency.
        For dummy activities (id < 0), the dummy's predecessors are used instead.
        """
        start_dependencies = {activity.id for activity in self.inbound_activities if activity.id >= 0}
        dummy_start_dependencies = [activity.predecessors for activity in self.inbound_activities if activity.id < 0]
        return start_dependencies.union(*dummy_start_dependencies)

    @property
    def earliest_start(self) -> float:
        if self.is_end:
            return max(
                [activity.earliest_start + activity.duration for activity in self.inbound_activities], default=0.0
            )
        else:
            return self.outbound_activities[0].earliest_start

    @property
    def latest_start(self) -> float:
        if self.is_end:
            return self.earliest_start
        else:
            return min([activity.latest_start for activity in self.outbound_activities], default=0.0)

    @property
    def latest_finish(self) -> float:
        return self.earliest_start

    @override
    def __str__(self) -> str:
        if not self.start_dependencies:
            return "start"
        elif self.is_end:
            return "end"
        else:
            return "-".join(str(v) for v in sorted(self.start_dependencies))


@dataclass
class NodeCollection(dict[int, Node]):
    def __init__(self, nodes: list[Node]):
        super().__init__((node.id, node) for node in nodes)
