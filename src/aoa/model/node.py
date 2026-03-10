from dataclasses import dataclass, field
from typing import Callable, TypeVar, override

from .activity import Activity

T = TypeVar("T")


# observable list that notives observers when an item is added or removed, used for inbound and outbound activities of a node
class ObservableList(list[T]):
    def __init__(self):
        super().__init__()
        self.append_callback: Callable[[T], None] = (
            lambda x: None
        )  # placeholder, will be set by Node when creating the list
        self.remove_callback: Callable[[T], None] = (
            lambda x: None
        )  # placeholder, will be set by Node when creating the list

    @override
    def append(self, item: T):
        super().append(item)
        self.append_callback(item)

    @override
    def remove(self, item: T):
        super().remove(item)
        self.remove_callback(item)

    def register_append_callback(self, callback: Callable[[T], None]) -> None:
        self.append_callback = callback

    def register_remove_callback(self, callback: Callable[[T], None]) -> None:
        self.remove_callback = callback


@dataclass
class Node:
    _id: int
    inbound_activities: ObservableList[Activity] = field(default_factory=ObservableList, repr=False, compare=False)
    outbound_activities: ObservableList[Activity] = field(default_factory=ObservableList, repr=False, compare=False)
    max_depth: int = field(default=0, compare=False)
    _id_change_callback: Callable[[int], None] = field(default=lambda x: None, repr=False, compare=False)

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        self._id = value

    @property
    def is_end(self) -> bool:
        return len(self.outbound_activities) == 0

    @property
    def start_dependencies(self) -> set[int]:
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

    def register_inbound_activity_append_callback(self, callback: Callable[[Activity], None]) -> None:
        self.inbound_activities.register_append_callback(callback)

    def register_outbound_activity_append_callback(self, callback: Callable[[Activity], None]) -> None:
        self.outbound_activities.register_append_callback(callback)

    def register_inbound_activity_remove_callback(self, callback: Callable[[Activity], None]) -> None:
        self.inbound_activities.register_remove_callback(callback)

    def register_outbound_activity_remove_callback(self, callback: Callable[[Activity], None]) -> None:
        self.outbound_activities.register_remove_callback(callback)

    def register_node_id_change_callback(self, callback: Callable[[int], None]) -> None:
        self._id_change_callback = callback


@dataclass
class NodeCollection(dict[int, Node]):
    def __init__(self, nodes: list[Node]):
        super().__init__((node.id, node) for node in nodes)
