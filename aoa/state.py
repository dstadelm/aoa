from enum import Enum


class State(Enum):
    DONE = 0
    IN_PROGRESS = 1
    ON_HOLD = 2
    OPEN = 3
    CANCELLED = 4
