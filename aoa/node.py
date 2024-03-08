from dataclasses import dataclass, field


@dataclass
class Node:
    earliest_start: int = field(default=0, compare=False)
    latest_finish: int = field(default=-1, compare=False)
    max_depth: int = field(default=0, compare=False)

    def __init__(self, earliest_start: int = 0, latest_finish: int = -1, max_depth: int = 0):
        self.earliest_start = earliest_start
        self.latest_finish = latest_finish
        self.max_depth = max_depth

    def update_earliest_start(self, contender: int):
        if contender > self.earliest_start:
            self.earliest_start = contender

    def update_max_depth(self, contender: int):
        if contender > self.max_depth:
            self.max_depth = contender

    def update_latest_finish(self, contender: int):
        if self.latest_finish < 0 or contender < self.latest_finish:
            self.latest_finish = contender

    def __post_init__(self):
        pass  # place init logging stuff here
