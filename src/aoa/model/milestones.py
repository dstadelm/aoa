import datetime

from attr import define, field

from .state import State


@define
class Milestone:
    id: str = field(default="")
    description: str = field(default="")
    owner: str = field(default="")
    due_date: datetime.date = field(default=datetime.date.today())
    state: State = field(default=State.OPEN)
