from typing import Any, Generator, Set, Tuple

from node import Node


class NodeDict(dict):
    def to_key(self, id_set: Set[int]) -> str:
        return "-".join(map(str, sorted(id_set)))

    def to_set(self, id: str) -> Set[int]:
        return set([int(i) for i in id.split("-")])

    def __setitem__(self, key: Set[int], value: Node) -> None:
        value.start_dependencies = key
        super().__setitem__(self.to_key(key), value)

    def __getitem__(self, key: Set[int]) -> Node:
        return super().__getitem__(self.to_key(key))

    def __delitem__(self, key: Set[int]) -> None:
        return super().__delitem__(self.to_key(key))

    def __contains__(self, key: Set[int]) -> bool:
        return super().__contains__(self.to_key(key))

    def items(self) -> Generator[Tuple[Set[int], Node], Any, Any]:
        for k, v in super().items():
            yield self.to_set(k), v

    def pop(self, key: Set[int]) -> Node:
        return super().pop(self.to_key(key))

    def keys(self) -> Generator[Set[int], Any, Any]:
        for k in super().keys():
            yield self.to_set(k)

    def values(self) -> Generator[Node, Any, Any]:
        for v in super().values():
            yield v

    def __repr__(self) -> str:
        m = "set, node id\n"
        for k, v in self.items():
            m += f"{k}:{v.start_dependencies} {v.id}\n"
        return m
