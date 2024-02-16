from dataclasses import dataclass, field


@dataclass
class Node:
    id: str
    earliest_start: int = field(default=0, compare=False)
    latest_start: int = field(default=0, compare=False)
    max_depth: int = field(default=0, compare=False)

    def __post_init__(self):
        pass  # place init logging stuff here
