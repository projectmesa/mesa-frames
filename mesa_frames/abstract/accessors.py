"""Abstract accessors for agent sets collections.

This module provides abstract base classes for accessors that enable
flexible querying and manipulation of collections of agent sets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from mesa_frames.abstract.agents import AgentSetDF
from mesa_frames.types_ import KeyBy


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

    @abstractmethod
    def __getitem__(
        self, key: int | str | type[AgentSetDF]
    ) -> AgentSetDF | list[AgentSetDF]:
        """Retrieve agent set(s) by index, name, or type.

        Parameters
        ----------
        key : int | str | type[AgentSetDF]
            - ``int``: positional index (supports negative indices).
            - ``str``: agent set name.
            - ``type``: class or subclass of :class:`AgentSetDF`.

        Returns
        -------
        AgentSetDF | list[AgentSetDF]
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

    @abstractmethod
    def get(self, key: int | str | type[AgentSetDF], default: Any | None = None) -> Any:
        """Safe lookup variant that returns a default on miss.

        Parameters
        ----------
        key : int | str | type[AgentSetDF]
            Lookup key; see :meth:`__getitem__`.
        default : Any | None, optional
            Value to return when the lookup fails. If ``key`` is a type and no
            matches are found, implementers may prefer returning ``[]`` when
            ``default`` is ``None`` to keep list shape stable.

        Returns
        -------
        Any
            The resolved value or ``default``.
        """

    @abstractmethod
    def first(self, t: type[AgentSetDF]) -> AgentSetDF:
        """Return the first agent set matching a type.

        Parameters
        ----------
        t : type[AgentSetDF]
            The concrete class (or base class) to match.

        Returns
        -------
        AgentSetDF
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
    def all(self, t: type[AgentSetDF]) -> list[AgentSetDF]:
        """Return all agent sets matching a type.

        Parameters
        ----------
        t : type[AgentSetDF]
            The concrete class (or base class) to match.

        Returns
        -------
        list[AgentSetDF]
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

    @abstractmethod
    def keys(self, *, key_by: KeyBy = "name") -> Iterable[Any]:
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
        Iterable[Any]
            An iterable of keys corresponding to the selected domain.
        """

    @abstractmethod
    def items(self, *, key_by: KeyBy = "name") -> Iterable[tuple[Any, AgentSetDF]]:
        """Iterate ``(key, AgentSetDF)`` pairs under a chosen key domain.

        See :meth:`keys` for the meaning of ``key_by``.
        """

    @abstractmethod
    def values(self) -> Iterable[AgentSetDF]:
        """Iterate over agent set values only (no keys)."""

    @abstractmethod
    def iter(self, *, key_by: KeyBy = "name") -> Iterable[tuple[Any, AgentSetDF]]:
        """Alias for :meth:`items` for convenience."""

    @abstractmethod
    def mapping(self, *, key_by: KeyBy = "name") -> dict[Any, AgentSetDF]:
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
