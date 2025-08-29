from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, cast

from types_ import KeyBy

from mesa_frames.abstract.agents import AgentSetDF
from mesa_frames.concrete.agents import AgentsDF


class AgentSetsAccessor(AgentSetsAccessorBase):
    def __init__(self, parent: "AgentsDF") -> None:
        self._parent = parent

    def __getitem__(
        self, key: int | str | type[AgentSetDF]
    ) -> AgentSetDF | list[AgentSetDF]:
        p = self._parent
        if isinstance(key, int):
            try:
                return p._agentsets[key]
            except IndexError as e:
                raise IndexError(
                    f"Index {key} out of range for {len(p._agentsets)} agent sets"
                ) from e
        if isinstance(key, str):
            for s in p._agentsets:
                if s.name == key:
                    return s
            available = [getattr(s, "name", None) for s in p._agentsets]
            raise KeyError(f"No agent set named '{key}'. Available: {available}")
        if isinstance(key, type):
            return [s for s in p._agentsets if isinstance(s, key)]
        raise TypeError("Key must be int | str | type[AgentSetDF]")

    def get(
        self, key: int | str | type[AgentSetDF], default: Any | None = None
    ) -> AgentSetDF | list[AgentSetDF] | Any | None:
        try:
            val = self[key]
            if isinstance(key, type) and val == [] and default is None:
                return []
            return val
        except (KeyError, IndexError, TypeError):
            # For type keys, preserve list shape by default
            if isinstance(key, type) and default is None:
                return []
            return default

    def first(self, t: type[AgentSetDF]) -> AgentSetDF:
        matches = [s for s in self._parent._agentsets if isinstance(s, t)]
        if not matches:
            raise KeyError(f"No agent set of type {getattr(t, '__name__', t)} found.")
        return matches[0]

    def all(self, t: type[AgentSetDF]) -> list[AgentSetDF]:
        return [s for s in self._parent._agentsets if isinstance(s, t)]

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

    def mapping(self, *, key_by: KeyBy = "name") -> dict[Any, AgentSetDF]:
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
    def __contains__(self, x: str | AgentSetDF) -> bool:
        if isinstance(x, str):
            return any(s.name == x for s in self._parent._agentsets)
        if isinstance(x, AgentSetDF):
            return any(s is x for s in self._parent._agentsets)
        return False

    def __len__(self) -> int:
        return len(self._parent._agentsets)

    def __iter__(self) -> Iterator[AgentSetDF]:
        return iter(self._parent._agentsets)
