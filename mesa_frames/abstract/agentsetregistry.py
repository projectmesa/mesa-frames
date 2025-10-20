"""
Abstract base classes for agent containers in mesa-frames.

This module defines the core abstractions for agent containers in the mesa-frames
extension. It provides the foundation for implementing agent storage and
manipulation using DataFrame-based approaches.

Classes:
    AbstractAgentSetRegistry(CopyMixin):
        An abstract base class that defines the common interface for all agent
        containers in mesa-frames. It inherits from CopyMixin to provide fast
        copying functionality.

    AbstractAgentSet(AbstractAgentSetRegistry, DataFrameMixin):
        An abstract base class for agent sets that use DataFrames as the underlying
        storage mechanism. It inherits from both AbstractAgentSetRegistry and DataFrameMixin
        to combine agent container functionality with DataFrame operations.

These abstract classes are designed to be subclassed by concrete implementations
that use Polars library as their backend.

Usage:
    These classes should not be instantiated directly. Instead, they should be
    subclassed to create concrete implementations:

    from mesa_frames.abstract.agents import AbstractAgentSet

    class AgentSet(AbstractAgentSet):
        def __init__(self, model):
            super().__init__(model)
            # Implementation using a DataFrame backend
            ...

        # Implement other abstract methods

Note:
    The abstract methods in these classes use Python's @abstractmethod decorator,
    ensuring that concrete subclasses must implement these methods.

Attributes and methods of each class are documented in their respective docstrings.
"""

from __future__ import annotations  # PEP 563: postponed evaluation of type annotations

from abc import abstractmethod
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from contextlib import suppress
from typing import Any, Literal, Self, overload

from numpy.random import Generator

from mesa_frames.abstract.mixin import CopyMixin
from mesa_frames.types_ import (
    AbstractAgentSetSelector as AgentSetSelector,
)
from mesa_frames.types_ import (
    BoolSeries,
    Index,
    KeyBy,
    Series,
)


class AbstractAgentSetRegistry(CopyMixin):
    """An abstract class for containing agents. Defines the common interface for AbstractAgentSet and AgentSetRegistry."""

    _copy_only_reference: list[str] = [
        "_model",
    ]
    _model: mesa_frames.concrete.model.Model

    @abstractmethod
    def __init__(self) -> None: ...

    def discard(
        self,
        sets: AgentSetSelector,
        inplace: bool = True,
    ) -> Self:
        """Remove AgentSets selected by ``sets``. Ignores missing.

        Parameters
        ----------
        sets : AgentSetSelector
            Which AgentSets to remove (instance, type, name, or collection thereof).
        inplace : bool
            Whether to remove in place. Defaults to True.

        Returns
        -------
        Self
            The updated AbstractAgentSetRegistry.
        """
        with suppress(KeyError, ValueError):
            return self.remove(sets, inplace=inplace)
        return self._get_obj(inplace)

    @abstractmethod
    def rename(
        self,
        target: (
            mesa_frames.abstract.agentset.AbstractAgentSet
            | str
            | dict[mesa_frames.abstract.agentset.AbstractAgentSet | str, str]
            | list[tuple[mesa_frames.abstract.agentset.AbstractAgentSet | str, str]]
        ),
        new_name: str | None = None,
        *,
        on_conflict: Literal["canonicalize", "raise"] = "canonicalize",
        mode: Literal["atomic", "best_effort"] = "atomic",
        inplace: bool = True,
    ) -> Self:
        """Rename AgentSets in this registry, handling conflicts.

        Parameters
        ----------
        target : mesa_frames.abstract.agentset.AbstractAgentSet | str | dict[mesa_frames.abstract.agentset.AbstractAgentSet | str, str] | list[tuple[mesa_frames.abstract.agentset.AbstractAgentSet | str, str]]
            Single target (instance or existing name) with ``new_name`` provided,
            or a mapping/sequence of (target, new_name) pairs for batch rename.
        new_name : str | None
            New name for single-target rename.
        on_conflict : Literal["canonicalize", "raise"]
            When a desired name collides, either canonicalize by appending a
            numeric suffix (default) or raise ``ValueError``.
        mode : Literal["atomic", "best_effort"]
            In "atomic" mode, validate all renames before applying any. In
            "best_effort" mode, apply what can be applied and skip failures.

        Returns
        -------
        Self
            Updated registry (or a renamed copy when ``inplace=False``).

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform the rename in place. If False, a renamed copy is
            returned, by default True.
        """
        ...

    @abstractmethod
    def add(
        self,
        sets: (
            mesa_frames.abstract.agentset.AbstractAgentSet
            | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
        ),
        inplace: bool = True,
    ) -> Self:
        """Add AgentSets to the AbstractAgentSetRegistry.

        Parameters
        ----------
        sets : mesa_frames.abstract.agentset.AbstractAgentSet | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
            The AgentSet(s) to add.
        inplace : bool
            Whether to add in place. Defaults to True.

        Returns
        -------
        Self
            The updated AbstractAgentSetRegistry.
        """
        ...

    @overload
    @abstractmethod
    def contains(
        self,
        sets: (
            mesa_frames.abstract.agentset.AbstractAgentSet
            | type[mesa_frames.abstract.agentset.AbstractAgentSet]
            | str
        ),
    ) -> bool: ...

    @overload
    @abstractmethod
    def contains(
        self,
        sets: Collection[
            mesa_frames.abstract.agentset.AbstractAgentSet
            | type[mesa_frames.abstract.agentset.AbstractAgentSet]
            | str
        ],
    ) -> BoolSeries: ...

    @abstractmethod
    def contains(self, sets: AgentSetSelector) -> bool | BoolSeries:
        """Check if selected AgentSets are present in the registry.

        Parameters
        ----------
        sets : AgentSetSelector
            An AgentSet instance, class/type, name string, or a collection of
            those. For collections, returns a BoolSeries aligned with input order.

        Returns
        -------
        bool | BoolSeries
            Boolean for single selector values; BoolSeries for collections.
        """

    @overload
    @abstractmethod
    def do(
        self,
        method_name: str,
        *args: Any,
        sets: AgentSetSelector | None = None,
        return_results: Literal[False] = False,
        inplace: bool = True,
        key_by: KeyBy = "name",
        **kwargs: Any,
    ) -> Self: ...

    @overload
    @abstractmethod
    def do(
        self,
        method_name: str,
        *args: Any,
        sets: AgentSetSelector,
        return_results: Literal[True],
        inplace: bool = True,
        key_by: KeyBy = "name",
        **kwargs: Any,
    ) -> (
        Any
        | dict[str, Any]
        | dict[int, Any]
        | dict[type[mesa_frames.abstract.agentset.AbstractAgentSet], Any]
    ): ...

    @abstractmethod
    def do(
        self,
        method_name: str,
        *args: Any,
        sets: AgentSetSelector = None,
        return_results: bool = False,
        inplace: bool = True,
        key_by: KeyBy = "name",
        **kwargs: Any,
    ) -> (
        Self
        | Any
        | dict[str, Any]
        | dict[int, Any]
        | dict[type[mesa_frames.abstract.agentset.AbstractAgentSet], Any]
    ):
        """Invoke a method on the AbstractAgentSetRegistry.

        Parameters
        ----------
        method_name : str
            The name of the method to invoke.
        *args : Any
            Positional arguments to pass to the method
        sets : AgentSetSelector, optional
            Which AgentSets to target (instance, type, name, or collection thereof). Defaults to all.
        return_results : bool, optional
            Whether to return per-set results as a dictionary, by default False.
        inplace : bool, optional
            Whether the operation should be done inplace, by default True
        key_by : KeyBy, optional
            Key domain for the returned mapping when ``return_results`` is True.
            - "name" (default) → keys are set names (str)
            - "index" → keys are positional indices (int)
            - "type" → keys are concrete set classes (type)
        **kwargs : Any
            Keyword arguments to pass to the method

        Returns
        -------
        Self | Any | dict[str, Any] | dict[int, Any] | dict[type[mesa_frames.abstract.agentset.AbstractAgentSet], Any]
            The updated registry, or the method result(s). When ``return_results``
            is True, returns a dictionary keyed per ``key_by``.
        """
        ...

    @overload
    @abstractmethod
    def get(
        self, key: int, default: None = ...
    ) -> mesa_frames.abstract.agentset.AbstractAgentSet | None: ...

    @overload
    @abstractmethod
    def get(
        self, key: str, default: None = ...
    ) -> mesa_frames.abstract.agentset.AbstractAgentSet | None: ...

    @overload
    @abstractmethod
    def get(
        self,
        key: type[mesa_frames.abstract.agentset.AbstractAgentSet],
        default: None = ...,
    ) -> list[mesa_frames.abstract.agentset.AbstractAgentSet]: ...

    @overload
    @abstractmethod
    def get(
        self,
        key: int | str | type[mesa_frames.abstract.agentset.AbstractAgentSet],
        default: mesa_frames.abstract.agentset.AbstractAgentSet
        | list[mesa_frames.abstract.agentset.AbstractAgentSet]
        | None,
    ) -> (
        mesa_frames.abstract.agentset.AbstractAgentSet
        | list[mesa_frames.abstract.agentset.AbstractAgentSet]
        | None
    ): ...

    @abstractmethod
    def get(
        self,
        key: int | str | type[mesa_frames.abstract.agentset.AbstractAgentSet],
        default: mesa_frames.abstract.agentset.AbstractAgentSet
        | list[mesa_frames.abstract.agentset.AbstractAgentSet]
        | None = None,
    ) -> (
        mesa_frames.abstract.agentset.AbstractAgentSet
        | list[mesa_frames.abstract.agentset.AbstractAgentSet]
        | None
    ):
        """Safe lookup for AgentSet(s) by index, name, or type."""

    @abstractmethod
    def remove(
        self,
        sets: AgentSetSelector,
        inplace: bool = True,
    ) -> Self:
        """Remove AgentSets from the AbstractAgentSetRegistry.

        Parameters
        ----------
        sets : AgentSetSelector
            Which AgentSets to remove (instance, type, name, or collection thereof).
        inplace : bool, optional
            Whether to remove the agent in place.

        Returns
        -------
        Self
            The updated AbstractAgentSetRegistry.
        """
        ...


    @abstractmethod
    def replace(
        self,
        mapping: (
            dict[int | str, mesa_frames.abstract.agentset.AbstractAgentSet]
            | list[tuple[int | str, mesa_frames.abstract.agentset.AbstractAgentSet]]
        ),
        *,
        inplace: bool = True,
        atomic: bool = True,
    ) -> Self:
        """Batch assign/replace AgentSets by index or name.

        Parameters
        ----------
        mapping : dict[int | str, mesa_frames.abstract.agentset.AbstractAgentSet] | list[tuple[int | str, mesa_frames.abstract.agentset.AbstractAgentSet]]
            Keys are indices or names to assign; values are AgentSets bound to the same model.
        inplace : bool, optional
            Whether to apply on this registry or return a copy, by default True.
        atomic : bool, optional
            When True, validates all keys and name invariants before applying any
            change; either all assignments succeed or none are applied.

        Returns
        -------
        Self
            Updated registry.
        """
        ...

    @abstractmethod
    def shuffle(self, inplace: bool = False) -> Self:
        """Shuffle the order of AgentSets in the registry.

        Parameters
        ----------
        inplace : bool
            Whether to shuffle in place.

        Returns
        -------
        Self
            A new or updated AbstractAgentSetRegistry.
        """

    @abstractmethod
    def sort(
        self,
        by: str | Sequence[str],
        ascending: bool | Sequence[bool] = True,
        inplace: bool = True,
        **kwargs,
    ) -> Self:
        """
        Sort the AgentSets in the registry based on the given criteria.

        Parameters
        ----------
        by : str | Sequence[str]
            The attribute(s) to sort by.
        ascending : bool | Sequence[bool]
            Whether to sort in ascending order.
        inplace : bool
            Whether to sort the agents in place.
        **kwargs
            Keyword arguments to pass to the sort

        Returns
        -------
        Self
            A new or updated AbstractAgentSetRegistry.
        """

    def __add__(
        self,
        other: mesa_frames.abstract.agentset.AbstractAgentSet
        | Collection[mesa_frames.abstract.agentset.AbstractAgentSet],
    ) -> Self:
        """Add AgentSets to a new AbstractAgentSetRegistry through the + operator."""
        return self.add(sets=other, inplace=False)

    def __contains__(
        self, sets: mesa_frames.abstract.agentset.AbstractAgentSet
    ) -> bool:
        """Check if an AgentSet is in the AbstractAgentSetRegistry."""
        return bool(self.contains(sets=sets))

    @overload
    def __getitem__(
        self, key: int
    ) -> mesa_frames.abstract.agentset.AbstractAgentSet: ...

    @overload
    def __getitem__(
        self, key: str
    ) -> mesa_frames.abstract.agentset.AbstractAgentSet: ...

    @overload
    def __getitem__(
        self, key: type[mesa_frames.abstract.agentset.AbstractAgentSet]
    ) -> list[mesa_frames.abstract.agentset.AbstractAgentSet]: ...

    def __getitem__(
        self, key: int | str | type[mesa_frames.abstract.agentset.AbstractAgentSet]
    ) -> (
        mesa_frames.abstract.agentset.AbstractAgentSet
        | list[mesa_frames.abstract.agentset.AbstractAgentSet]
    ):
        """Retrieve AgentSet(s) by index, name, or type."""

    def __iadd__(
        self,
        other: (
            mesa_frames.abstract.agentset.AbstractAgentSet
            | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
        ),
    ) -> Self:
        """Add AgentSets to the registry through the += operator.

        Parameters
        ----------
        other : mesa_frames.abstract.agentset.AbstractAgentSet | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
            The AgentSets to add.

        Returns
        -------
        Self
            The updated AbstractAgentSetRegistry.
        """
        return self.add(sets=other, inplace=True)

    def __isub__(
        self,
        other: (
            mesa_frames.abstract.agentset.AbstractAgentSet
            | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
        ),
    ) -> Self:
        """Remove AgentSets from the registry through the -= operator.

        Parameters
        ----------
        other : mesa_frames.abstract.agentset.AbstractAgentSet | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
            The AgentSets to remove.

        Returns
        -------
        Self
            The updated AbstractAgentSetRegistry.
        """
        return self.discard(other, inplace=True)

    def __sub__(
        self,
        other: (
            mesa_frames.abstract.agentset.AbstractAgentSet
            | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
        ),
    ) -> Self:
        """Remove AgentSets from a new registry through the - operator.

        Parameters
        ----------
        other : mesa_frames.abstract.agentset.AbstractAgentSet | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
            The AgentSets to remove.

        Returns
        -------
        Self
            A new AbstractAgentSetRegistry with the removed AgentSets.
        """
        return self.discard(other, inplace=False)

    def __setitem__(
        self,
        key: int | str,
        value: mesa_frames.abstract.agentset.AbstractAgentSet,
    ) -> None:
        """Assign/replace a single AgentSet at an index or name.

        Mirrors the invariants of ``replace`` for single-key assignment:
        - Names remain unique across the registry
        - ``value.model is self.model``
        - For name keys, the key is authoritative for the assigned set's name
        - For index keys, collisions on a different entry's name must raise
        """
        if value.model is not self.model:
            raise TypeError("Assigned AgentSet must belong to the same model")
        if isinstance(key, int):
            # Delegate to replace() so subclasses centralize invariant handling.
            self.replace({key: value}, inplace=True, atomic=True)
            return
        if isinstance(key, str):
            for existing in self:
                if existing.name == key:
                    self.replace({key: value}, inplace=True, atomic=True)
                    return
            value.rename(key, inplace=True)
            self.add(value, inplace=True)
            return
        raise TypeError("Key must be int index or str name")

    @abstractmethod
    def __getattr__(self, name: str) -> Any | dict[str, Any]:
        """Fallback for retrieving attributes of the AgentSetRegistry."""

    @abstractmethod
    def __iter__(self) -> Iterator[mesa_frames.abstract.agentset.AbstractAgentSet]:
        """Iterate over AgentSets in the registry."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Get the number of AgentSets in the registry."""
        ...

    @abstractmethod
    def __repr__(self) -> str:
        """Get a string representation of the AgentSets in the registry."""
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator:
        """Iterate over AgentSets in reverse order."""
        ...

    @abstractmethod
    def __str__(self) -> str:
        """Get a string representation of the AgentSets in the registry."""
        ...

    def keys(
        self, *, key_by: KeyBy = "name"
    ) -> Iterable[str | int | type[mesa_frames.abstract.agentset.AbstractAgentSet]]:
        """Iterate keys for contained AgentSets (by name|index|type)."""
        if key_by == "index":
            yield from range(len(self))
            return
        if key_by == "type":
            for agentset in self:
                yield type(agentset)
            return
        if key_by != "name":
            raise ValueError("key_by must be 'name'|'index'|'type'")
        for agentset in self:
            if agentset.name is not None:
                yield agentset.name

    def items(
        self, *, key_by: KeyBy = "name"
    ) -> Iterable[
        tuple[
            str | int | type[mesa_frames.abstract.agentset.AbstractAgentSet],
            mesa_frames.abstract.agentset.AbstractAgentSet,
        ]
    ]:
        """Iterate (key, AgentSet) pairs for contained sets."""
        if key_by == "index":
            for idx, agentset in enumerate(self):
                yield idx, agentset
            return
        if key_by == "type":
            for agentset in self:
                yield type(agentset), agentset
            return
        if key_by != "name":
            raise ValueError("key_by must be 'name'|'index'|'type'")
        for agentset in self:
            if agentset.name is not None:
                yield agentset.name, agentset

    def values(self) -> Iterable[mesa_frames.abstract.agentset.AbstractAgentSet]:
        """Iterate contained AgentSets (values view)."""
        yield from self

    @property
    def model(self) -> mesa_frames.concrete.model.Model:
        """The model that the AbstractAgentSetRegistry belongs to.

        Returns
        -------
        mesa_frames.concrete.model.Model
        """
        return self._model

    @property
    def random(self) -> Generator:
        """The random number generator of the model.

        Returns
        -------
        Generator
        """
        return self.model.random

    @property
    def space(self) -> mesa_frames.abstract.space.Space | None:
        """The space of the model.

        Returns
        -------
        mesa_frames.abstract.space.Space | None
        """
        return self.model.space

    @property
    @abstractmethod
    def ids(self) -> Series:
        """Public view of all agent unique_id values across contained sets.

        Returns
        -------
        Series
            Concatenated unique_id Series for all AgentSets.
        """
        ...
