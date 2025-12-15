# TODO:
# - implement holidays
from dataclasses import dataclass

from attrs import define, field


@define
class Resource:
    id: str = field(default="")
    name: str = field(default="")
    pensum: str = field(default="")
    weekdays: str = field(default="1111100")
    holidays: list[str] = field(factory=list)


@dataclass
class ResourceCollection:
    resources: list[Resource]
