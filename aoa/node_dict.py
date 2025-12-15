from collections.abc import Iterable, Iterator, MutableMapping, Set
from typing import override

from .node import Node


class NodeDict(MutableMapping[Set[int], Node]):
    """
    A dict-like mapping from a set of ints to a Node.
    - Accepts any AbstractSet[int] as a key; internally canonicalizes to frozenset[int].
    - Iteration yields frozenset[int] keys.
    """

    __slots__ = ("_store",)  # pyright: ignore [reportUnannotatedClassAttribute]

    def __init__(
        self,
        items: Iterable[tuple[Set[int], Node]] | None = None,
    ) -> None:
        self._store: dict[frozenset[int], Node] = {}
        if items is not None:
            for k, v in items:
                self[k] = v  # go through __setitem__ to validate and normalize

    # --- Internal helpers ---

    def _norm_key(self, key: Set[int]) -> frozenset[int]:
        """Normalize any Set[int] to a frozenset[int] and validate element types."""
        try:
            fs = frozenset(key)
        except TypeError as e:
            raise TypeError("Key must be an abstract set of ints") from e
        return fs

    # --- MutableMapping interface ---
    @override
    def __getitem__(self, key: Set[int]) -> Node:
        return self._store[self._norm_key(key)]

    @override
    def __setitem__(self, key: Set[int], value: Node) -> None:
        self._store[self._norm_key(key)] = value

    @override
    def __delitem__(self, key: Set[int]) -> None:
        del self._store[self._norm_key(key)]

    @override
    def __iter__(self) -> Iterator[frozenset[int]]:
        # Iterates over normalized keys (frozenset[int])
        return iter(self._store)

    @override
    def __len__(self) -> int:
        return len(self._store)

    # Optional: faster membership test
    @override
    def __contains__(self, key: object) -> bool:
        try:
            k = self._norm_key(key)  # pyright: ignore [reportArgumentType]
        except TypeError:
            return False
        return k in self._store

    # Optional: nicer representation
    @override
    def __repr__(self) -> str:
        # Build a dict-like repr but keys are frozenset[int]
        items = ", ".join(f"{k}: {v!r}" for k, v in self._store.items())
        return f"{self.__class__.__name__}({{{items}}})"
