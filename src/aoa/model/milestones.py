import datetime
from dataclasses import dataclass

from attr import define, field

from .state import State


@define
class Milestone:
    id: str = field(default="")
    description: str = field(default="")
    owner: str = field(default="")
    due_date: datetime.date = field(default=datetime.date.today())
    state: State = field(default=State.OPEN)


@dataclass
class MilestoneCollection:
    milestones: list[Milestone]
    _milestones_dict: dict[str, Milestone] = field(init=False, factory=dict[str, Milestone])

    def __get_item__(self, key: str) -> Milestone:
        if not self._milestones_dict:
            self._milestones_dict = {r.id: r for r in self.milestones}

        if not (milestone := self._milestones_dict.get(key)):
            raise KeyError(f"Resource with id {key} not found")
        return milestone
