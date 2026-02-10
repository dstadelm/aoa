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
    _resource_dict: dict[str, Resource] = field(init=False, factory=dict[str, Resource])

    def __get_item__(self, resource_id: str) -> Resource:
        if not self._resource_dict:
            self._resource_dict = {r.id: r for r in self.resources}

        if not (resource := self._resource_dict.get(resource_id)):
            raise KeyError(f"Resource with id {resource_id} not found")
        return resource
