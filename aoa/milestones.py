import datetime
from dataclasses import dataclass
from enum import Enum


class Status(Enum):
    DONE = 0
    IN_PROGRESS = 1
    ON_HOLD = 2
    OPEN = 3
    CANCELLED = 4


@dataclass
class Milestone:
    id: str
    description: str
    owner: str
    due_date: datetime.date
    status: Status


@dataclass
class MilestoneCollection:
    milestones: list[Milestone]
