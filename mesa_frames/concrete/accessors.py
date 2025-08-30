"""Concrete implementations of agent set accessors.

This module contains the concrete implementation of the AgentSetsAccessor,
which provides a user-friendly interface for accessing and manipulating
collections of agent sets within the mesa-frames library.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, Literal, TypeVar, cast

from mesa_frames.abstract.accessors import AbstractAgentSetsAccessor
from mesa_frames.abstract.agents import AgentSetDF
from mesa_frames.types_ import KeyBy

TSet = TypeVar("TSet", bound=AgentSetDF)


class AgentSetsAccessor(AbstractAgentSetsAccessor):
    def __init__(self, parent: mesa_frames.concrete.agents.AgentsDF) -> None:
        self._parent = parent

    def __getitem__(
        self, key: int | str | type[AgentSetDF]
    ) -> AgentSetDF | list[AgentSetDF]:
        sets = self._parent._agentsets
        if isinstance(key, int):
            try:
                return sets[key]
            except IndexError as e:
                raise IndexError(
                    f"Index {key} out of range for {len(sets)} agent sets"
                ) from e
        if isinstance(key, str):
            for s in sets:
                if s.name == key:
                    return s
            available = [getattr(s, "name", None) for s in sets]
            raise KeyError(f"No agent set named '{key}'. Available: {available}")
        if isinstance(key, type):
            matches = [s for s in sets if isinstance(s, key)]
            # Always return list for type keys to maintain consistent shape
            return matches  # type: ignore[return-value]
        raise TypeError("Key must be int | str | type[AgentSetDF]")

    def get(
        self,
        key: int | str | type[TSet],
        default: AgentSetDF | list[TSet] | None = None,
    ) -> AgentSetDF | list[TSet] | None:
        try:
            val = self[key]  # type: ignore[return-value]
            # For type keys, if no matches and a default was provided, return default
            if (
                isinstance(key, type)
                and isinstance(val, list)
                and len(val) == 0
                and default is not None
            ):
                return default
            return val
        except (KeyError, IndexError, TypeError):
            return default

    def first(self, t: type[TSet]) -> TSet:
        match = next((s for s in self._parent._agentsets if isinstance(s, t)), None)
        if not match:
            raise KeyError(f"No agent set of type {getattr(t, '__name__', t)} found.")
        return match

    def all(self, t: type[TSet]) -> list[TSet]:
        return [s for s in self._parent._agentsets if isinstance(s, t)]  # type: ignore[return-value]

    def at(self, index: int) -> AgentSetDF:
        return self[index]  # type: ignore[return-value]

    # ---------- key generation and views ----------
    def _gen_key(self, aset: AgentSetDF, idx: int, mode: str) -> Any:
        if mode == "name":
            return aset.name
        if mode == "index":
            return idx
        if mode == "object":
            return aset
        if mode == "type":
            return type(aset)
        raise ValueError("key_by must be 'name'|'index'|'object'|'type'")

    def keys(self, *, key_by: KeyBy = "name") -> Iterable[Any]:
        for i, s in enumerate(self._parent._agentsets):
            yield self._gen_key(s, i, key_by)

    def items(self, *, key_by: KeyBy = "name") -> Iterable[tuple[Any, AgentSetDF]]:
        for i, s in enumerate(self._parent._agentsets):
            yield self._gen_key(s, i, key_by), s

    def values(self) -> Iterable[AgentSetDF]:
        return iter(self._parent._agentsets)

    def iter(self, *, key_by: KeyBy = "name") -> Iterable[tuple[Any, AgentSetDF]]:
        return self.items(key_by=key_by)

    def dict(self, *, key_by: KeyBy = "name") -> dict[Any, AgentSetDF]:
        return {k: v for k, v in self.items(key_by=key_by)}

    # ---------- read-only snapshots ----------
    @property
    def by_name(self) -> Mapping[str, AgentSetDF]:
        return MappingProxyType({cast(str, s.name): s for s in self._parent._agentsets})

    @property
    def by_type(self) -> Mapping[type, list[AgentSetDF]]:
        d: dict[type, list[AgentSetDF]] = defaultdict(list)
        for s in self._parent._agentsets:
            d[type(s)].append(s)
        return MappingProxyType(dict(d))

    # ---------- membership & iteration ----------
    def rename(
        self,
        target: AgentSetDF
        | str
        | dict[AgentSetDF | str, str]
        | list[tuple[AgentSetDF | str, str]],
        new_name: str | None = None,
        *,
        on_conflict: Literal["canonicalize", "raise"] = "canonicalize",
        mode: Literal["atomic", "best_effort"] = "atomic",
    ) -> str | dict[AgentSetDF, str]:
        """
        Rename agent sets. Supports single and batch renaming with deterministic conflict handling.

        Parameters
        ----------
        target : AgentSetDF | str | dict[AgentSetDF | str, str] | list[tuple[AgentSetDF | str, str]]
            Either:
            - Single: AgentSet or name string (must provide new_name)
            - Batch: {target: new_name} dict or [(target, new_name), ...] list
        new_name : str | None, optional
            New name (only used for single renames)
        on_conflict : "Literal['canonicalize', 'raise']"
            Conflict resolution: "canonicalize" (default) appends suffixes, "raise" raises ValueError
        mode : "Literal['atomic', 'best_effort']"
            Rename mode: "atomic" applies all or none (default), "best_effort" skips failed renames

        Returns
        -------
        str | dict[AgentSetDF, str]
            Single rename: final name string
            Batch: {agentset: final_name} mapping

        Examples
        --------
        Single rename:
        >>> agents.sets.rename("old_name", "new_name")

        Batch rename (dict):
        >>> agents.sets.rename({"set1": "new_name", "set2": "another_name"})

        Batch rename (list):
        >>> agents.sets.rename([("set1", "new_name"), ("set2", "another_name")])
        """
        return self._parent._rename_set(
            target, new_name, on_conflict=on_conflict, mode=mode
        )

    def __contains__(self, x: str | AgentSetDF) -> bool:
        sets = self._parent._agentsets
        if isinstance(x, str):
            return any(s.name == x for s in sets)
        if isinstance(x, AgentSetDF):
            return any(s is x for s in sets)
        return False

    def __len__(self) -> int:
        return len(self._parent._agentsets)

    def __iter__(self) -> Iterator[AgentSetDF]:
        return iter(self._parent._agentsets)
