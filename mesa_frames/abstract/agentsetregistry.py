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
<<<<<<< HEAD
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
=======
from collections.abc import Callable, Collection, Iterator, Sequence
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
from contextlib import suppress
from typing import Any, Literal, Self, overload

from numpy.random import Generator

from mesa_frames.abstract.mixin import CopyMixin
from mesa_frames.types_ import (
<<<<<<< HEAD
    AbstractAgentSetSelector as AgentSetSelector,
)
from mesa_frames.types_ import (
    BoolSeries,
    Index,
    KeyBy,
=======
    AgentMask,
    BoolSeries,
    DataFrame,
    DataFrameInput,
    IdsLike,
    Index,
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
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
<<<<<<< HEAD
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
=======
        agents: IdsLike
        | AgentMask
        | mesa_frames.abstract.agentset.AbstractAgentSet
        | Collection[mesa_frames.abstract.agentset.AbstractAgentSet],
        inplace: bool = True,
    ) -> Self:
        """Remove agents from the AbstractAgentSetRegistry. Does not raise an error if the agent is not found.

        Parameters
        ----------
        agents : IdsLike | AgentMask | mesa_frames.abstract.agentset.AbstractAgentSet | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
            The agents to remove
        inplace : bool
            Whether to remove the agent in place. Defaults to True.
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

        Returns
        -------
        Self
            The updated AbstractAgentSetRegistry.
        """
        with suppress(KeyError, ValueError):
<<<<<<< HEAD
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
=======
            return self.remove(agents, inplace=inplace)
        return self._get_obj(inplace)

    @abstractmethod
    def add(
        self,
        agents: DataFrame
        | DataFrameInput
        | mesa_frames.abstract.agentset.AbstractAgentSet
        | Collection[mesa_frames.abstract.agentset.AbstractAgentSet],
        inplace: bool = True,
    ) -> Self:
        """Add agents to the AbstractAgentSetRegistry.

        Parameters
        ----------
        agents : DataFrame | DataFrameInput | mesa_frames.abstract.agentset.AbstractAgentSet | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
            The agents to add.
        inplace : bool
            Whether to add the agents in place. Defaults to True.
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

        Returns
        -------
        Self
            The updated AbstractAgentSetRegistry.
        """
        ...

    @overload
    @abstractmethod
<<<<<<< HEAD
    def contains(
        self,
        sets: (
            mesa_frames.abstract.agentset.AbstractAgentSet
            | type[mesa_frames.abstract.agentset.AbstractAgentSet]
            | str
        ),
    ) -> bool: ...
=======
    def contains(self, agents: int) -> bool: ...
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

    @overload
    @abstractmethod
    def contains(
<<<<<<< HEAD
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
=======
        self, agents: mesa_frames.abstract.agentset.AbstractAgentSet | IdsLike
    ) -> BoolSeries: ...

    @abstractmethod
    def contains(
        self, agents: mesa_frames.abstract.agentset.AbstractAgentSet | IdsLike
    ) -> bool | BoolSeries:
        """Check if agents with the specified IDs are in the AbstractAgentSetRegistry.

        Parameters
        ----------
        agents : mesa_frames.abstract.agentset.AbstractAgentSet | IdsLike
            The ID(s) to check for.
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

        Returns
        -------
        bool | BoolSeries
<<<<<<< HEAD
            Boolean for single selector values; BoolSeries for collections.
=======
            True if the agent is in the AbstractAgentSetRegistry, False otherwise.
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
        """

    @overload
    @abstractmethod
    def do(
        self,
        method_name: str,
        *args: Any,
<<<<<<< HEAD
        sets: AgentSetSelector | None = None,
        return_results: Literal[False] = False,
        inplace: bool = True,
        key_by: KeyBy = "name",
=======
        mask: AgentMask | None = None,
        return_results: Literal[False] = False,
        inplace: bool = True,
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
        **kwargs: Any,
    ) -> Self: ...

    @overload
    @abstractmethod
    def do(
        self,
        method_name: str,
        *args: Any,
<<<<<<< HEAD
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
=======
        mask: AgentMask | None = None,
        return_results: Literal[True],
        inplace: bool = True,
        **kwargs: Any,
    ) -> Any | dict[mesa_frames.abstract.agentset.AbstractAgentSet, Any]: ...
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

    @abstractmethod
    def do(
        self,
        method_name: str,
        *args: Any,
<<<<<<< HEAD
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
=======
        mask: AgentMask | None = None,
        return_results: bool = False,
        inplace: bool = True,
        **kwargs: Any,
    ) -> Self | Any | dict[mesa_frames.abstract.agentset.AbstractAgentSet, Any]:
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
        """Invoke a method on the AbstractAgentSetRegistry.

        Parameters
        ----------
        method_name : str
            The name of the method to invoke.
        *args : Any
            Positional arguments to pass to the method
<<<<<<< HEAD
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
=======
        mask : AgentMask | None, optional
            The subset of agents on which to apply the method
        return_results : bool, optional
            Whether to return the result of the method, by default False
        inplace : bool, optional
            Whether the operation should be done inplace, by default False
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
        **kwargs : Any
            Keyword arguments to pass to the method

        Returns
        -------
<<<<<<< HEAD
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
=======
        Self | Any | dict[mesa_frames.abstract.agentset.AbstractAgentSet, Any]
            The updated AbstractAgentSetRegistry or the result of the method.
        """
        ...

    @abstractmethod
    @overload
    def get(self, attr_names: str) -> Series | dict[str, Series]: ...

    @abstractmethod
    @overload
    def get(
        self, attr_names: Collection[str] | None = None
    ) -> DataFrame | dict[str, DataFrame]: ...
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

    @abstractmethod
    def get(
        self,
<<<<<<< HEAD
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
=======
        attr_names: str | Collection[str] | None = None,
        mask: AgentMask | None = None,
    ) -> Series | dict[str, Series] | DataFrame | dict[str, DataFrame]:
        """Retrieve the value of a specified attribute for each agent in the AbstractAgentSetRegistry.

        Parameters
        ----------
        attr_names : str | Collection[str] | None, optional
            The attributes to retrieve. If None, all attributes are retrieved. Defaults to None.
        mask : AgentMask | None, optional
            The AgentMask of agents to retrieve the attribute for. If None, attributes of all agents are returned. Defaults to None.

        Returns
        -------
        Series | dict[str, Series] | DataFrame | dict[str, DataFrame]
            The attribute values.
        """
        ...
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

    @abstractmethod
    def remove(
        self,
<<<<<<< HEAD
        sets: AgentSetSelector,
        inplace: bool = True,
    ) -> Self:
        """Remove AgentSets from the AbstractAgentSetRegistry.

        Parameters
        ----------
        sets : AgentSetSelector
            Which AgentSets to remove (instance, type, name, or collection thereof).
=======
        agents: (
            IdsLike
            | AgentMask
            | mesa_frames.abstract.agentset.AbstractAgentSet
            | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
        ),
        inplace: bool = True,
    ) -> Self:
        """Remove the agents from the AbstractAgentSetRegistry.

        Parameters
        ----------
        agents : IdsLike | AgentMask | mesa_frames.abstract.agentset.AbstractAgentSet | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
            The agents to remove.
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
        inplace : bool, optional
            Whether to remove the agent in place.

        Returns
        -------
        Self
            The updated AbstractAgentSetRegistry.
        """
        ...

<<<<<<< HEAD
    # select() intentionally removed from the abstract API.

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
=======
    @abstractmethod
    def select(
        self,
        mask: AgentMask | None = None,
        filter_func: Callable[[Self], AgentMask] | None = None,
        n: int | None = None,
        negate: bool = False,
        inplace: bool = True,
    ) -> Self:
        """Select agents in the AbstractAgentSetRegistry based on the given criteria.

        Parameters
        ----------
        mask : AgentMask | None, optional
            The AgentMask of agents to be selected, by default None
        filter_func : Callable[[Self], AgentMask] | None, optional
            A function which takes as input the AbstractAgentSetRegistry and returns a AgentMask, by default None
        n : int | None, optional
            The maximum number of agents to be selected, by default None
        negate : bool, optional
            If the selection should be negated, by default False
        inplace : bool, optional
            If the operation should be performed on the same object, by default True
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

        Returns
        -------
        Self
<<<<<<< HEAD
            Updated registry.
=======
            A new or updated AbstractAgentSetRegistry.
        """
        ...

    @abstractmethod
    @overload
    def set(
        self,
        attr_names: dict[str, Any],
        values: None,
        mask: AgentMask | None = None,
        inplace: bool = True,
    ) -> Self: ...

    @abstractmethod
    @overload
    def set(
        self,
        attr_names: str | Collection[str],
        values: Any,
        mask: AgentMask | None = None,
        inplace: bool = True,
    ) -> Self: ...

    @abstractmethod
    def set(
        self,
        attr_names: DataFrameInput | str | Collection[str],
        values: Any | None = None,
        mask: AgentMask | None = None,
        inplace: bool = True,
    ) -> Self:
        """Set the value of a specified attribute or attributes for each agent in the mask in AbstractAgentSetRegistry.

        Parameters
        ----------
        attr_names : DataFrameInput | str | Collection[str]
            The key can be:
            - A string: sets the specified column of the agents in the AbstractAgentSetRegistry.
            - A collection of strings: sets the specified columns of the agents in the AbstractAgentSetRegistry.
            - A dictionary: keys should be attributes and values should be the values to set. Value should be None.
        values : Any | None
            The value to set the attribute to. If None, attr_names must be a dictionary.
        mask : AgentMask | None
            The AgentMask of agents to set the attribute for.
        inplace : bool
            Whether to set the attribute in place.

        Returns
        -------
        Self
            The updated agent set.
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
        """
        ...

    @abstractmethod
    def shuffle(self, inplace: bool = False) -> Self:
<<<<<<< HEAD
        """Shuffle the order of AgentSets in the registry.
=======
        """Shuffles the order of agents in the AbstractAgentSetRegistry.
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

        Parameters
        ----------
        inplace : bool
<<<<<<< HEAD
            Whether to shuffle in place.
=======
            Whether to shuffle the agents in place.
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

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
<<<<<<< HEAD
        Sort the AgentSets in the registry based on the given criteria.
=======
        Sorts the agents in the agent set based on the given criteria.
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

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
<<<<<<< HEAD
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
=======
        other: DataFrame
        | DataFrameInput
        | mesa_frames.abstract.agentset.AbstractAgentSet
        | Collection[mesa_frames.abstract.agentset.AbstractAgentSet],
    ) -> Self:
        """Add agents to a new AbstractAgentSetRegistry through the + operator.

        Parameters
        ----------
        other : DataFrame | DataFrameInput | mesa_frames.abstract.agentset.AbstractAgentSet | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
            The agents to add.

        Returns
        -------
        Self
            A new AbstractAgentSetRegistry with the added agents.
        """
        return self.add(agents=other, inplace=False)

    def __contains__(
        self, agents: int | mesa_frames.abstract.agentset.AbstractAgentSet
    ) -> bool:
        """Check if an agent is in the AbstractAgentSetRegistry.

        Parameters
        ----------
        agents : int | mesa_frames.abstract.agentset.AbstractAgentSet
            The ID(s) or AbstractAgentSet to check for.

        Returns
        -------
        bool
            True if the agent is in the AbstractAgentSetRegistry, False otherwise.
        """
        return self.contains(agents=agents)

    @overload
    def __getitem__(
        self, key: str | tuple[AgentMask, str]
    ) -> Series | dict[mesa_frames.abstract.agentset.AbstractAgentSet, Series]: ...

    @overload
    def __getitem__(
        self,
        key: AgentMask | Collection[str] | tuple[AgentMask, Collection[str]],
    ) -> (
        DataFrame | dict[mesa_frames.abstract.agentset.AbstractAgentSet, DataFrame]
    ): ...

    def __getitem__(
        self,
        key: (
            str
            | Collection[str]
            | AgentMask
            | tuple[AgentMask, str]
            | tuple[AgentMask, Collection[str]]
            | tuple[
                dict[mesa_frames.abstract.agentset.AbstractAgentSet, AgentMask], str
            ]
            | tuple[
                dict[mesa_frames.abstract.agentset.AbstractAgentSet, AgentMask],
                Collection[str],
            ]
        ),
    ) -> (
        Series
        | DataFrame
        | dict[mesa_frames.abstract.agentset.AbstractAgentSet, Series]
        | dict[mesa_frames.abstract.agentset.AbstractAgentSet, DataFrame]
    ):
        """Implement the [] operator for the AbstractAgentSetRegistry.

        The key can be:
        - An attribute or collection of attributes (eg. AbstractAgentSetRegistry["str"], AbstractAgentSetRegistry[["str1", "str2"]]): returns the specified column(s) of the agents in the AbstractAgentSetRegistry.
        - An AgentMask (eg. AbstractAgentSetRegistry[AgentMask]): returns the agents in the AbstractAgentSetRegistry that satisfy the AgentMask.
        - A tuple (eg. AbstractAgentSetRegistry[AgentMask, "str"]): returns the specified column of the agents in the AbstractAgentSetRegistry that satisfy the AgentMask.
        - A tuple with a dictionary (eg. AbstractAgentSetRegistry[{AbstractAgentSet: AgentMask}, "str"]): returns the specified column of the agents in the AbstractAgentSetRegistry that satisfy the AgentMask from the dictionary.
        - A tuple with a dictionary (eg. AbstractAgentSetRegistry[{AbstractAgentSet: AgentMask}, Collection[str]]): returns the specified columns of the agents in the AbstractAgentSetRegistry that satisfy the AgentMask from the dictionary.

        Parameters
        ----------
        key : str | Collection[str] | AgentMask | tuple[AgentMask, str] | tuple[AgentMask, Collection[str]] | tuple[dict[mesa_frames.abstract.agentset.AbstractAgentSet, AgentMask], str] | tuple[dict[mesa_frames.abstract.agentset.AbstractAgentSet, AgentMask], Collection[str]]
            The key to retrieve.

        Returns
        -------
        Series | DataFrame | dict[mesa_frames.abstract.agentset.AbstractAgentSet, Series] | dict[mesa_frames.abstract.agentset.AbstractAgentSet, DataFrame]
            The attribute values.
        """
        # TODO: fix types
        if isinstance(key, tuple):
            return self.get(mask=key[0], attr_names=key[1])
        else:
            if isinstance(key, str) or (
                isinstance(key, Collection) and all(isinstance(k, str) for k in key)
            ):
                return self.get(attr_names=key)
            else:
                return self.get(mask=key)
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

    def __iadd__(
        self,
        other: (
<<<<<<< HEAD
            mesa_frames.abstract.agentset.AbstractAgentSet
            | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
        ),
    ) -> Self:
        """Add AgentSets to the registry through the += operator.

        Parameters
        ----------
        other : mesa_frames.abstract.agentset.AbstractAgentSet | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
            The AgentSets to add.
=======
            DataFrame
            | DataFrameInput
            | mesa_frames.abstract.agentset.AbstractAgentSet
            | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
        ),
    ) -> Self:
        """Add agents to the AbstractAgentSetRegistry through the += operator.

        Parameters
        ----------
        other : DataFrame | DataFrameInput | mesa_frames.abstract.agentset.AbstractAgentSet | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
            The agents to add.
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

        Returns
        -------
        Self
            The updated AbstractAgentSetRegistry.
        """
<<<<<<< HEAD
        return self.add(sets=other, inplace=True)
=======
        return self.add(agents=other, inplace=True)
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

    def __isub__(
        self,
        other: (
<<<<<<< HEAD
            mesa_frames.abstract.agentset.AbstractAgentSet
            | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
        ),
    ) -> Self:
        """Remove AgentSets from the registry through the -= operator.

        Parameters
        ----------
        other : mesa_frames.abstract.agentset.AbstractAgentSet | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
            The AgentSets to remove.
=======
            IdsLike
            | AgentMask
            | mesa_frames.abstract.agentset.AbstractAgentSet
            | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
        ),
    ) -> Self:
        """Remove agents from the AbstractAgentSetRegistry through the -= operator.

        Parameters
        ----------
        other : IdsLike | AgentMask | mesa_frames.abstract.agentset.AbstractAgentSet | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
            The agents to remove.
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

        Returns
        -------
        Self
            The updated AbstractAgentSetRegistry.
        """
        return self.discard(other, inplace=True)

    def __sub__(
        self,
        other: (
<<<<<<< HEAD
            mesa_frames.abstract.agentset.AbstractAgentSet
            | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
        ),
    ) -> Self:
        """Remove AgentSets from a new registry through the - operator.

        Parameters
        ----------
        other : mesa_frames.abstract.agentset.AbstractAgentSet | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
            The AgentSets to remove.
=======
            IdsLike
            | AgentMask
            | mesa_frames.abstract.agentset.AbstractAgentSet
            | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
        ),
    ) -> Self:
        """Remove agents from a new AbstractAgentSetRegistry through the - operator.

        Parameters
        ----------
        other : IdsLike | AgentMask | mesa_frames.abstract.agentset.AbstractAgentSet | Collection[mesa_frames.abstract.agentset.AbstractAgentSet]
            The agents to remove.
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

        Returns
        -------
        Self
<<<<<<< HEAD
            A new AbstractAgentSetRegistry with the removed AgentSets.
=======
            A new AbstractAgentSetRegistry with the removed agents.
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
        """
        return self.discard(other, inplace=False)

    def __setitem__(
        self,
<<<<<<< HEAD
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

    @abstractmethod
    def __getattr__(self, name: str) -> Any | dict[str, Any]:
        """Fallback for retrieving attributes of the AgentSetRegistry."""

    @abstractmethod
    def __iter__(self) -> Iterator[mesa_frames.abstract.agentset.AbstractAgentSet]:
        """Iterate over AgentSets in the registry."""
=======
        key: (
            str
            | Collection[str]
            | AgentMask
            | tuple[AgentMask, str | Collection[str]]
            | tuple[
                dict[mesa_frames.abstract.agentset.AbstractAgentSet, AgentMask], str
            ]
            | tuple[
                dict[mesa_frames.abstract.agentset.AbstractAgentSet, AgentMask],
                Collection[str],
            ]
        ),
        values: Any,
    ) -> None:
        """Implement the [] operator for setting values in the AbstractAgentSetRegistry.

        The key can be:
        - A string (eg. AbstractAgentSetRegistry["str"]): sets the specified column of the agents in the AbstractAgentSetRegistry.
        - A list of strings(eg. AbstractAgentSetRegistry[["str1", "str2"]]): sets the specified columns of the agents in the AbstractAgentSetRegistry.
        - A tuple (eg. AbstractAgentSetRegistry[AgentMask, "str"]): sets the specified column of the agents in the AbstractAgentSetRegistry that satisfy the AgentMask.
        - A AgentMask (eg. AbstractAgentSetRegistry[AgentMask]): sets the attributes of the agents in the AbstractAgentSetRegistry that satisfy the AgentMask.
        - A tuple with a dictionary (eg. AbstractAgentSetRegistry[{AbstractAgentSet: AgentMask}, "str"]): sets the specified column of the agents in the AbstractAgentSetRegistry that satisfy the AgentMask from the dictionary.
        - A tuple with a dictionary (eg. AbstractAgentSetRegistry[{AbstractAgentSet: AgentMask}, Collection[str]]): sets the specified columns of the agents in the AbstractAgentSetRegistry that satisfy the AgentMask from the dictionary.

        Parameters
        ----------
        key : str | Collection[str] | AgentMask | tuple[AgentMask, str | Collection[str]] | tuple[dict[mesa_frames.abstract.agentset.AbstractAgentSet, AgentMask], str] | tuple[dict[mesa_frames.abstract.agentset.AbstractAgentSet, AgentMask], Collection[str]]
            The key to set.
        values : Any
            The values to set for the specified key.
        """
        # TODO: fix types as in __getitem__
        if isinstance(key, tuple):
            self.set(mask=key[0], attr_names=key[1], values=values)
        else:
            if isinstance(key, str) or (
                isinstance(key, Collection) and all(isinstance(k, str) for k in key)
            ):
                try:
                    self.set(attr_names=key, values=values)
                except KeyError:  # key=AgentMask
                    self.set(attr_names=None, mask=key, values=values)
            else:
                self.set(attr_names=None, mask=key, values=values)

    @abstractmethod
    def __getattr__(self, name: str) -> Any | dict[str, Any]:
        """Fallback for retrieving attributes of the AbstractAgentSetRegistry. Retrieve an attribute of the underlying DataFrame(s).

        Parameters
        ----------
        name : str
            The name of the attribute to retrieve.

        Returns
        -------
        Any | dict[str, Any]
            The attribute value
        """

    @abstractmethod
    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over the agents in the AbstractAgentSetRegistry.

        Returns
        -------
        Iterator[dict[str, Any]]
            An iterator over the agents.
        """
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
        ...

    @abstractmethod
    def __len__(self) -> int:
<<<<<<< HEAD
        """Get the number of AgentSets in the registry."""
=======
        """Get the number of agents in the AbstractAgentSetRegistry.

        Returns
        -------
        int
            The number of agents in the AbstractAgentSetRegistry.
        """
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
        ...

    @abstractmethod
    def __repr__(self) -> str:
<<<<<<< HEAD
        """Get a string representation of the AgentSets in the registry."""
=======
        """Get a string representation of the DataFrame in the AbstractAgentSetRegistry.

        Returns
        -------
        str
            A string representation of the DataFrame in the AbstractAgentSetRegistry.
        """
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator:
<<<<<<< HEAD
        """Iterate over AgentSets in reverse order."""
=======
        """Iterate over the agents in the AbstractAgentSetRegistry in reverse order.

        Returns
        -------
        Iterator
            An iterator over the agents in reverse order.
        """
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
        ...

    @abstractmethod
    def __str__(self) -> str:
<<<<<<< HEAD
        """Get a string representation of the AgentSets in the registry."""
        ...

    @abstractmethod
    def keys(
        self, *, key_by: KeyBy = "name"
    ) -> Iterable[str | int | type[mesa_frames.abstract.agentset.AbstractAgentSet]]:
        """Iterate keys for contained AgentSets (by name|index|type)."""
        ...

    @abstractmethod
    def items(
        self, *, key_by: KeyBy = "name"
    ) -> Iterable[
        tuple[
            str | int | type[mesa_frames.abstract.agentset.AbstractAgentSet],
            mesa_frames.abstract.agentset.AbstractAgentSet,
        ]
    ]:
        """Iterate (key, AgentSet) pairs for contained sets."""
        ...

    @abstractmethod
    def values(self) -> Iterable[mesa_frames.abstract.agentset.AbstractAgentSet]:
        """Iterate contained AgentSets (values view)."""
=======
        """Get a string representation of the agents in the AbstractAgentSetRegistry.

        Returns
        -------
        str
            A string representation of the agents in the AbstractAgentSetRegistry.
        """
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
        ...

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
<<<<<<< HEAD
    def ids(self) -> Series:
        """Public view of all agent unique_id values across contained sets.

        Returns
        -------
        Series
            Concatenated unique_id Series for all AgentSets.
=======
    def df(self) -> DataFrame | dict[str, DataFrame]:
        """The agents in the AbstractAgentSetRegistry.

        Returns
        -------
        DataFrame | dict[str, DataFrame]
        """

    @df.setter
    @abstractmethod
    def df(
        self, agents: DataFrame | list[mesa_frames.abstract.agentset.AbstractAgentSet]
    ) -> None:
        """Set the agents in the AbstractAgentSetRegistry.

        Parameters
        ----------
        agents : DataFrame | list[mesa_frames.abstract.agentset.AbstractAgentSet]
        """

    @property
    @abstractmethod
    def active_agents(self) -> DataFrame | dict[str, DataFrame]:
        """The active agents in the AbstractAgentSetRegistry.

        Returns
        -------
        DataFrame | dict[str, DataFrame]
        """

    @active_agents.setter
    @abstractmethod
    def active_agents(
        self,
        mask: AgentMask,
    ) -> None:
        """Set the active agents in the AbstractAgentSetRegistry.

        Parameters
        ----------
        mask : AgentMask
            The mask to apply.
        """
        self.select(mask=mask, inplace=True)

    @property
    @abstractmethod
    def inactive_agents(
        self,
    ) -> DataFrame | dict[mesa_frames.abstract.agentset.AbstractAgentSet, DataFrame]:
        """The inactive agents in the AbstractAgentSetRegistry.

        Returns
        -------
        DataFrame | dict[mesa_frames.abstract.agentset.AbstractAgentSet, DataFrame]
        """

    @property
    @abstractmethod
    def index(
        self,
    ) -> Index | dict[mesa_frames.abstract.agentset.AbstractAgentSet, Index]:
        """The ids in the AbstractAgentSetRegistry.

        Returns
        -------
        Index | dict[mesa_frames.abstract.agentset.AbstractAgentSet, Index]
        """
        ...

    @property
    @abstractmethod
    def pos(
        self,
    ) -> DataFrame | dict[mesa_frames.abstract.agentset.AbstractAgentSet, DataFrame]:
        """The position of the agents in the AbstractAgentSetRegistry.

        Returns
        -------
        DataFrame | dict[mesa_frames.abstract.agentset.AbstractAgentSet, DataFrame]
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
        """
        ...
