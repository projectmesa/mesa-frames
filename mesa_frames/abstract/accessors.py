"""Abstract accessors for agent sets collections.

This module provides abstract base classes for accessors that enable
flexible querying and manipulation of collections of agent sets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Literal, overload, TypeVar

from mesa_frames.abstract.agents import AgentSetDF
from mesa_frames.types_ import KeyBy

TSet = TypeVar("TSet", bound=AgentSetDF)


class AbstractAgentSetsAccessor(ABC):
    """Abstract accessor for collections of agent sets.

    This interface defines a flexible, user-friendly API to access agent sets
    by name, positional index, or class/type, and to iterate or view the
    collection under different key domains.

    Notes
    -----
    Concrete implementations should:
    - Support ``__getitem__`` with ``int`` | ``str`` | ``type[AgentSetDF]``.
    - Return a list for type-based queries (even when there is one match).
    - Provide keyed iteration via ``keys/items/iter/mapping`` with ``key_by``.
    - Expose read-only snapshots ``by_name`` and ``by_type``.

    Examples
    --------
    Assuming ``agents`` is an :class:`~mesa_frames.concrete.agents.AgentsDF`:

    >>> sheep = agents.sets["Sheep"]             # name lookup
    >>> first = agents.sets[0]                    # index lookup
    >>> wolves = agents.sets[Wolf]                # type lookup → list
    >>> len(wolves) >= 0
    True

    Choose a key view when iterating:

    >>> for k, aset in agents.sets.items(key_by="index"):
    ...     print(k, aset.name)
    0 Sheep
    1 Wolf
    """

    # __getitem__ — exact shapes per key kind
    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> AgentSetDF: ...

    @overload
    @abstractmethod
    def __getitem__(self, key: str) -> AgentSetDF: ...

    @overload
    @abstractmethod
    def __getitem__(self, key: type[TSet]) -> list[TSet]: ...

    @abstractmethod
    def __getitem__(self, key: int | str | type[TSet]) -> AgentSetDF | list[TSet]:
        """Retrieve agent set(s) by index, name, or type.

        Parameters
        ----------
        key : int | str | type[TSet]
            - ``int``: positional index (supports negative indices).
            - ``str``: agent set name.
            - ``type``: class or subclass of :class:`AgentSetDF`.

        Returns
        -------
        AgentSetDF | list[TSet]
            A single agent set for ``int``/``str`` keys; a list of matching
            agent sets for ``type`` keys (possibly empty).

        Raises
        ------
        IndexError
            If an index is out of range.
        KeyError
            If a name is missing.
        TypeError
            If the key type is unsupported.
        """

    # get — mirrors dict.get, but preserves list shape for type keys
    @overload
    @abstractmethod
    def get(self, key: int, default: None = ...) -> AgentSetDF | None: ...

    @overload
    @abstractmethod
    def get(self, key: str, default: None = ...) -> AgentSetDF | None: ...

    @overload
    @abstractmethod
    def get(self, key: type[TSet], default: None = ...) -> list[TSet]: ...

    @overload
    @abstractmethod
    def get(self, key: int, default: AgentSetDF) -> AgentSetDF: ...

    @overload
    @abstractmethod
    def get(self, key: str, default: AgentSetDF) -> AgentSetDF: ...

    @overload
    @abstractmethod
    def get(self, key: type[TSet], default: list[TSet]) -> list[TSet]: ...

    @abstractmethod
    def get(
        self,
        key: int | str | type[TSet],
        default: AgentSetDF | list[TSet] | None = None,
    ) -> AgentSetDF | list[TSet] | None:
        """
        Safe lookup variant that returns a default on miss.

        Parameters
        ----------
        key : int | str | type[TSet]
            Lookup key; see :meth:`__getitem__`.
        default : AgentSetDF | list[TSet] | None, optional
            Value to return when the lookup fails. For type keys, if no matches
            are found and default is None, implementers should return [] to keep
            list shape stable.

        Returns
        -------
        AgentSetDF | list[TSet] | None
            - int/str keys: return the set or default/None if missing
            - type keys: return list of matching sets; if none and default is None,
              return [] (stable list shape)
        """

    @abstractmethod
    def first(self, t: type[TSet]) -> TSet:
        """Return the first agent set matching a type.

        Parameters
        ----------
        t : type[TSet]
            The concrete class (or base class) to match.

        Returns
        -------
        TSet
            The first matching agent set in iteration order.

        Raises
        ------
        KeyError
            If no agent set matches ``t``.

        Examples
        --------
        >>> agents.sets.first(Wolf)  # doctest: +SKIP
        <Wolf ...>
        """

    @abstractmethod
    def all(self, t: type[TSet]) -> list[TSet]:
        """Return all agent sets matching a type.

        Parameters
        ----------
        t : type[TSet]
            The concrete class (or base class) to match.

        Returns
        -------
        list[TSet]
            A list of all matching agent sets (possibly empty).

        Examples
        --------
        >>> agents.sets.all(Wolf)  # doctest: +SKIP
        [<Wolf ...>, <Wolf ...>]
        """

    @abstractmethod
    def at(self, index: int) -> AgentSetDF:
        """Return the agent set at a positional index.

        Parameters
        ----------
        index : int
            Positional index; negative indices are supported.

        Returns
        -------
        AgentSetDF
            The agent set at the given position.

        Raises
        ------
        IndexError
            If ``index`` is out of range.

        Examples
        --------
        >>> agents.sets.at(0) is agents.sets[0]
        True
        """

    @overload
    @abstractmethod
    def keys(self, *, key_by: Literal["name"]) -> Iterable[str]: ...

    @overload
    @abstractmethod
    def keys(self, *, key_by: Literal["index"]) -> Iterable[int]: ...

    @overload
    @abstractmethod
    def keys(self, *, key_by: Literal["object"]) -> Iterable[AgentSetDF]: ...

    @overload
    @abstractmethod
    def keys(self, *, key_by: Literal["type"]) -> Iterable[type[AgentSetDF]]: ...

    @abstractmethod
    def keys(
        self, *, key_by: KeyBy = "name"
    ) -> Iterable[str | int | AgentSetDF | type[AgentSetDF]]:
        """Iterate keys under a chosen key domain.

        Parameters
        ----------
        key_by : KeyBy
            - ``"name"`` → agent set names. (Default)
            - ``"index"`` → positional indices.
            - ``"object"`` → the :class:`AgentSetDF` objects.
            - ``"type"`` → the concrete classes of each set.

        Returns
        -------
        Iterable[str | int | AgentSetDF | type[AgentSetDF]]
            An iterable of keys corresponding to the selected domain.
        """

    @overload
    @abstractmethod
    def items(self, *, key_by: Literal["name"]) -> Iterable[tuple[str, AgentSetDF]]: ...

    @overload
    @abstractmethod
    def items(
        self, *, key_by: Literal["index"]
    ) -> Iterable[tuple[int, AgentSetDF]]: ...

    @overload
    @abstractmethod
    def items(
        self, *, key_by: Literal["object"]
    ) -> Iterable[tuple[AgentSetDF, AgentSetDF]]: ...

    @overload
    @abstractmethod
    def items(
        self, *, key_by: Literal["type"]
    ) -> Iterable[tuple[type[AgentSetDF], AgentSetDF]]: ...

    @abstractmethod
    def items(
        self, *, key_by: KeyBy = "name"
    ) -> Iterable[tuple[str | int | AgentSetDF | type[AgentSetDF], AgentSetDF]]:
        """Iterate ``(key, AgentSetDF)`` pairs under a chosen key domain.

        See :meth:`keys` for the meaning of ``key_by``.
        """

    @abstractmethod
    def values(self) -> Iterable[AgentSetDF]:
        """Iterate over agent set values only (no keys)."""

    @abstractmethod
    def iter(self, *, key_by: KeyBy = "name") -> Iterable[tuple[Any, AgentSetDF]]:
        """Alias for :meth:`items` for convenience."""

    @overload
    @abstractmethod
    def dict(self, *, key_by: Literal["name"]) -> dict[str, AgentSetDF]: ...

    @overload
    @abstractmethod
    def dict(self, *, key_by: Literal["index"]) -> dict[int, AgentSetDF]: ...

    @overload
    @abstractmethod
    def dict(self, *, key_by: Literal["object"]) -> dict[AgentSetDF, AgentSetDF]: ...

    @overload
    @abstractmethod
    def dict(
        self, *, key_by: Literal["type"]
    ) -> dict[type[AgentSetDF], AgentSetDF]: ...

    @abstractmethod
    def dict(
        self, *, key_by: KeyBy = "name"
    ) -> dict[str | int | AgentSetDF | type[AgentSetDF], AgentSetDF]:
        """Return a dictionary view keyed by the chosen domain.

        Notes
        -----
        ``key_by="type"`` will keep the last set per type. For one-to-many
        grouping, prefer the read-only :attr:`by_type` snapshot.
        """

    @property
    @abstractmethod
    def by_name(self) -> Mapping[str, AgentSetDF]:
        """Read-only mapping of names to agent sets.

        Returns
        -------
        Mapping[str, AgentSetDF]
            An immutable snapshot that maps each agent set name to its object.

        Notes
        -----
        Implementations should return a read-only mapping such as
        ``types.MappingProxyType`` over an internal dict to avoid accidental
        mutation.

        Examples
        --------
        >>> sheep = agents.sets.by_name["Sheep"]  # doctest: +SKIP
        >>> sheep is agents.sets["Sheep"]  # doctest: +SKIP
        True
        """

    @property
    @abstractmethod
    def by_type(self) -> Mapping[type, list[AgentSetDF]]:
        """Read-only mapping of types to lists of agent sets.

        Returns
        -------
        Mapping[type, list[AgentSetDF]]
            An immutable snapshot grouping agent sets by their concrete class.

        Notes
        -----
        This supports one-to-many relationships where multiple sets share the
        same type. Prefer this over ``mapping(key_by="type")`` when you need
        grouping instead of last-write-wins semantics.
        """

    @abstractmethod
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

    @abstractmethod
    def __contains__(self, x: str | AgentSetDF) -> bool:
        """Return ``True`` if a name or object is present.

        Parameters
        ----------
        x : str | AgentSetDF
            A name to test by equality, or an object to test by identity.

        Returns
        -------
        bool
            ``True`` if present, else ``False``.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Return number of agent sets in the collection."""

    @abstractmethod
    def __iter__(self) -> Iterator[AgentSetDF]:
        """Iterate over agent set values in insertion order."""
