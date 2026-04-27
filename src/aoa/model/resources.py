# TODO:
# - implement holidays
from dataclasses import dataclass

from attrs import define, field


@define
class Resource:
    id: str = field(default="")
    name: str = field(default="")
    workload: str = field(default="")
    weekdays: str = field(default="1111100")
    holidays: list[str] = field(factory=list)


@dataclass
class ResourceCollection(dict[str, Resource]):
    def __init__(self, resources: list[Resource]):
        super().__init__({resource.id: resource for resource in resources})
