"""
Abstract base classes for agent sets in mesa-frames.

This module defines the core abstractions for agent sets in the mesa-frames
extension. It provides the foundation for implementing agent set storage and
manipulation.

Classes:
    AbstractAgentSet:
        An abstract base class for agent sets that combines agent container
        functionality with DataFrame operations. It inherits from both
        AbstractAgentSetRegistry and DataFrameMixin to provide comprehensive
        agent management capabilities.

This abstract class is designed to be subclassed to create concrete
implementations that use specific DataFrame backends.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Collection, Iterable, Iterator
from contextlib import suppress
from typing import Any, Literal, Self, overload

from collections.abc import Callable, Sequence

from numpy.random import Generator

from mesa_frames.abstract.mixin import CopyMixin, DataFrameMixin
from mesa_frames.types_ import (
    AgentMask,
    BoolSeries,
    DataFrame,
    DataFrameInput,
    IdsLike,
    Index,
    Series,
)


class AbstractAgentSet(CopyMixin, DataFrameMixin):
    """The AbstractAgentSet class is a container for agents of the same type.

    Parameters
    ----------
    model : mesa_frames.concrete.model.Model
        The model that the agent set belongs to.
    """

    _copy_only_reference: list[str] = ["_model"]
    _df: DataFrame  # The agents in the AbstractAgentSet
    _mask: AgentMask  # The underlying mask used for the active agents in the AbstractAgentSet.
    _model: (
        mesa_frames.concrete.model.Model
    )  # The model that the AbstractAgentSet belongs to.

    @abstractmethod
    def __init__(self, model: mesa_frames.concrete.model.Model) -> None: ...

    @abstractmethod
    def add(
        self,
        agents: DataFrame | DataFrameInput,
        inplace: bool = True,
    ) -> Self:
        """Add agents to the AbstractAgentSet.

        Agents can be the input to the DataFrame constructor. So, the input can be:
        - A DataFrame: adds the agents from the DataFrame.
        - A DataFrameInput: passes the input to the DataFrame constructor.

        Parameters
        ----------
        agents : DataFrame | DataFrameInput
            The agents to add.
        inplace : bool, optional
            If True, perform the operation in place, by default True

        Returns
        -------
        Self
            A new AbstractAgentSet with the added agents.
        """
        ...

    @overload
    @abstractmethod
    def contains(self, agents: int) -> bool: ...

    @overload
    @abstractmethod
    def contains(self, agents: IdsLike) -> BoolSeries: ...

    @abstractmethod
    def contains(self, agents: IdsLike) -> bool | BoolSeries:
        """Check if agents with the specified IDs are in the AgentSet.

        Parameters
        ----------
        agents : IdsLike
            The ID(s) to check for.

        Returns
        -------
        bool | BoolSeries
            True if the agent is in the AgentSet, False otherwise.
        """
        ...

    def discard(self, agents: IdsLike | AgentMask, inplace: bool = True) -> Self:
        """Remove an agent from the AbstractAgentSet. Does not raise an error if the agent is not found.

        Parameters
        ----------
        agents : IdsLike | AgentMask
            The ids to remove
        inplace : bool, optional
            Whether to remove the agent in place, by default True

        Returns
        -------
        Self
            The updated AbstractAgentSet.
        """
        with suppress(KeyError, ValueError):
            return self.remove(agents, inplace=inplace)
        return self._get_obj(inplace)

    @abstractmethod
    def remove(self, agents: IdsLike | AgentMask, inplace: bool = True) -> Self:
        """Remove agents from this AbstractAgentSet.

        Parameters
        ----------
        agents : IdsLike | AgentMask
            The agents or mask to remove.
        inplace : bool, optional
            Whether to remove in place, by default True.

        Returns
        -------
        Self
            The updated agent set.
        """
        ...

    @overload
    @abstractmethod
    def do(
        self,
        method_name: str,
        *args: Any,
        mask: AgentMask | None = None,
        return_results: Literal[False] = False,
        inplace: bool = True,
        **kwargs: Any,
    ) -> Self: ...

    @overload
    @abstractmethod
    def do(
        self,
        method_name: str,
        *args: Any,
        mask: AgentMask | None = None,
        return_results: Literal[True],
        inplace: bool = True,
        **kwargs: Any,
    ) -> Any: ...

    @abstractmethod
    def do(
        self,
        method_name: str,
        *args: Any,
        mask: AgentMask | None = None,
        return_results: bool = False,
        inplace: bool = True,
        **kwargs: Any,
    ) -> Self | Any:
        """Invoke a method on the AgentSet.

        Parameters
        ----------
        method_name : str
            The name of the method to invoke.
        *args : Any
            Positional arguments to pass to the method
        mask : AgentMask | None, optional
            The subset of agents on which to apply the method
        return_results : bool, optional
            Whether to return the result of the method, by default False
        inplace : bool, optional
            Whether the operation should be done inplace, by default False
        **kwargs : Any
            Keyword arguments to pass to the method

        Returns
        -------
        Self | Any
            The updated AgentSet or the result of the method.
        """
        ...

    @abstractmethod
    @overload
    def get(
        self,
        attr_names: str,
        mask: AgentMask | None = None,
    ) -> Series: ...

    @abstractmethod
    @overload
    def get(
        self,
        attr_names: Collection[str] | None = None,
        mask: AgentMask | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    def get(
        self,
        attr_names: str | Collection[str] | None = None,
        mask: AgentMask | None = None,
    ) -> Series | DataFrame:
        """Retrieve agent attributes as a Series or DataFrame.

        Parameters
        ----------
        attr_names : str | Collection[str] | None, optional
            Column name or collection of names to fetch. When ``None``, return
            all agent attributes (excluding any internal identifiers).
        mask : AgentMask | None, optional
            Subset selector limiting which agents are included. ``None`` means
            operate on the full set.

        Returns
        -------
        Series | DataFrame
            A Series when selecting a single attribute, otherwise a DataFrame
            containing the requested columns.
        """
        ...

    @abstractmethod
    def step(self) -> None:
        """Run a single step of the AbstractAgentSet. This method should be overridden by subclasses."""
        ...

    @abstractmethod
    def _concatenate_agentsets(
        self,
        objs: Iterable[Self],
        duplicates_allowed: bool = True,
        keep_first_only: bool = True,
        original_masked_index: Index | None = None,
    ) -> Self: ...

    @abstractmethod
    def _get_bool_mask(self, mask: AgentMask) -> BoolSeries:
        """Get the equivalent boolean mask based on the input mask.

        Parameters
        ----------
        mask : AgentMask

        Returns
        -------
        BoolSeries
        """
        ...

    @abstractmethod
    def _get_masked_df(self, mask: AgentMask) -> DataFrame:
        """Get the df filtered by the input mask.

        Parameters
        ----------
        mask : AgentMask

        Returns
        -------
        DataFrame
        """

    @overload
    @abstractmethod
    def _get_obj_copy(self, obj: DataFrame) -> DataFrame: ...

    @overload
    @abstractmethod
    def _get_obj_copy(self, obj: Series) -> Series: ...

    @overload
    @abstractmethod
    def _get_obj_copy(self, obj: Index) -> Index: ...

    @abstractmethod
    def _get_obj_copy(
        self, obj: DataFrame | Series | Index
    ) -> DataFrame | Series | Index: ...

    @abstractmethod
    def _discard(self, ids: IdsLike) -> Self:
        """Remove an agent from the DataFrame of the AbstractAgentSet. Gets called by self.model.sets.remove and self.model.sets.discard.

        Parameters
        ----------
        ids : IdsLike

            The ids to remove

        Returns
        -------
        Self
        """
        ...

    @abstractmethod
    def _update_mask(
        self, original_active_indices: Index, new_active_indices: Index | None = None
    ) -> None: ...

    def __add__(self, other: DataFrame | DataFrameInput) -> Self:
        """Add agents to a new AbstractAgentSet through the + operator.

        Other can be:
        - A DataFrame: adds the agents from the DataFrame.
        - A DataFrameInput: passes the input to the DataFrame constructor.

        Parameters
        ----------
        other : DataFrame | DataFrameInput
            The agents to add.

        Returns
        -------
        Self
            A new AbstractAgentSet with the added agents.
        """
        return self.add(other, inplace=False)

    def __iadd__(self, other: DataFrame | DataFrameInput) -> Self:
        """
        Add agents to the AbstractAgentSet through the += operator.

        Other can be:
        - A DataFrame: adds the agents from the DataFrame.
        - A DataFrameInput: passes the input to the DataFrame constructor.

        Parameters
        ----------
        other : DataFrame | DataFrameInput
            The agents to add.

        Returns
        -------
        Self
            The updated AbstractAgentSet.
        """
        return self.add(other, inplace=True)

    def __isub__(self, other: IdsLike | AgentMask | DataFrame) -> Self:
        """Remove agents via -= operator."""
        return self.discard(other, inplace=True)

    def __sub__(self, other: IdsLike | AgentMask | DataFrame) -> Self:
        """Return a new set with agents removed via - operator."""
        return self.discard(other, inplace=False)

    @abstractmethod
    def __getattr__(self, name: str) -> Any:
        if __debug__:  # Only execute in non-optimized mode
            if name == "_df":
                raise AttributeError(
                    "The _df attribute is not set. You probably forgot to call super().__init__ in the __init__ method."
                )

    @overload
    def __getitem__(self, key: str | tuple[AgentMask, str]) -> Series | DataFrame: ...

    @overload
    def __getitem__(
        self,
        key: AgentMask | Collection[str] | tuple[AgentMask, Collection[str]],
    ) -> DataFrame: ...

    def __getitem__(
        self,
        key: (
            str
            | Collection[str]
            | AgentMask
            | tuple[AgentMask, str]
            | tuple[AgentMask, Collection[str]]
        ),
    ) -> Series | DataFrame:
        # Mirror registry/old container behavior: delegate to get()
        if isinstance(key, tuple):
            return self.get(mask=key[0], attr_names=key[1])
        else:
            if isinstance(key, str) or (
                isinstance(key, Collection) and all(isinstance(k, str) for k in key)
            ):
                return self.get(attr_names=key)
            else:
                return self.get(mask=key)

    def __contains__(self, agents: int) -> bool:
        """Membership test for an agent id in this set."""
        return bool(self.contains(agents))

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}\n {str(self._df)}"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}\n {str(self._df)}"

    def __reversed__(self) -> Iterator:
        return reversed(self._df)

    @property
    def df(self) -> DataFrame:
        """Return the full backing DataFrame for this agent set.

        Returns
        -------
        DataFrame
            Table containing every agent, including inactive records.
        """
        return self._df

    @df.setter
    def df(self, agents: DataFrame) -> None:
        """Set the agents in the AbstractAgentSet.

        Parameters
        ----------
        agents : DataFrame
            The agents to set.
        """
        self._df = agents

    @property
    @abstractmethod
    def active_agents(self) -> DataFrame:
        """Return the subset of agents currently marked as active.

        Returns
        -------
        DataFrame
            DataFrame view containing only active agents.
        """
        ...

    @property
    @abstractmethod
    def inactive_agents(self) -> DataFrame:
        """Return the subset of agents currently marked as inactive.

        Returns
        -------
        DataFrame
            DataFrame view containing only inactive agents.
        """
        ...

    @property
    @abstractmethod
    def index(self) -> Index:
        """Return the unique identifier index for agents in this set.

        Returns
        -------
        Index
            Collection of unique agent identifiers.
        """
        ...

    @property
    def pos(self) -> DataFrame:
        """Return positional data for agents from the attached space.

        Returns
        -------
        DataFrame
            Position records aligned with each agent's ``unique_id``.

        Raises
        ------
        AttributeError
            If the model has no space attached.
        """
        if self.space is None:
            raise AttributeError(
                "Attempted to access `pos`, but the model has no space attached."
            )
        pos = self._df_get_masked_df(
            df=self.space.agents, index_cols="agent_id", mask=self.index
        )
        pos = self._df_reindex(
            pos, self.index, new_index_cols="unique_id", original_index_cols="agent_id"
        )
        return pos

    @property
    def name(self) -> str:
        """The name of the agent set.

        Returns
        -------
        str
            The name of the agent set
        """
        return self._name

    @property
    def model(self) -> mesa_frames.concrete.model.Model:
        """Return the parent model for this agent set.

        Returns
        -------
        mesa_frames.concrete.model.Model
            The model instance that owns this agent set.
        """
        return self._model

    @property
    def random(self) -> Generator:
        """Return the random number generator shared with the model.

        Returns
        -------
        Generator
            Generator used for stochastic operations.
        """
        return self.model.random

    @property
    def space(self) -> mesa_frames.abstract.space.Space | None:
        """Return the space attached to the parent model, if any.

        Returns
        -------
        mesa_frames.abstract.space.Space | None
            Spatial structure registered on the model, or ``None`` when absent.
        """
        return self.model.space

    @abstractmethod
    def rename(self, new_name: str, inplace: bool = True) -> Self:
        """Rename this AgentSet.

        Concrete subclasses must implement the mechanics for coordinating with
        any containing registry and managing ``inplace`` semantics. The method
        should update the set's name (or return a renamed copy when
        ``inplace=False``) while preserving registry invariants.

        Parameters
        ----------
        new_name : str
            Desired new name for this AgentSet.
        inplace : bool, optional
            Whether to perform the rename in place. If False, a renamed copy is
            returned, by default True.

        Returns
        -------
        Self
            The updated AgentSet (or a renamed copy when ``inplace=False``).
        """
        ...

    @abstractmethod
    def set(
        self,
        attr_names: str | Collection[str] | dict[str, Any] | None = None,
        values: Any | None = None,
        mask: AgentMask | None = None,
        inplace: bool = True,
    ) -> Self:
        """Update agent attributes, optionally on a masked subset.

        Parameters
        ----------
        attr_names : str | Collection[str] | dict[str, Any] | None, optional
            Attribute(s) to assign. When ``None``, concrete implementations may
            derive targets from ``values``.
        values : Any | None, optional
            Replacement value(s) aligned with ``attr_names``.
        mask : AgentMask | None, optional
            Subset selector limiting which agents are updated.
        inplace : bool, optional
            Whether to mutate in place or return an updated copy, by default True.

        Returns
        -------
        Self
            The updated AgentSet (or a modified copy when ``inplace=False``).
        """
        ...

    @abstractmethod
    def select(
        self,
        mask: AgentMask | None = None,
        filter_func: Callable[[Self], BoolSeries] | None = None,
        n: int | None = None,
        negate: bool = False,
        inplace: bool = True,
    ) -> Self:
        """Update the active-agent mask using selection criteria.

        Parameters
        ----------
        mask : AgentMask | None, optional
            Pre-computed mask identifying agents to activate.
        filter_func : Callable[[Self], BoolSeries] | None, optional
            Callable evaluated on the agent set to produce an additional mask.
        n : int | None, optional
            Randomly sample ``n`` agents from the selected mask when provided.
        negate : bool, optional
            Invert the effective mask, by default False.
        inplace : bool, optional
            Whether to mutate in place or return an updated copy, by default True.

        Returns
        -------
        Self
            The updated AgentSet (or a modified copy when ``inplace=False``).
        """
        ...

    @abstractmethod
    def shuffle(self, inplace: bool = True) -> Self:
        """Randomly permute agent order.

        Parameters
        ----------
        inplace : bool, optional
            Whether to mutate in place or return a shuffled copy, by default True.

        Returns
        -------
        Self
            The shuffled AgentSet (or a shuffled copy when ``inplace=False``).
        """
        ...

    @abstractmethod
    def sort(
        self,
        by: str | Sequence[str],
        ascending: bool | Sequence[bool] = True,
        inplace: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Sort agents by one or more columns.

        Parameters
        ----------
        by : str | Sequence[str]
            Column name(s) to sort on.
        ascending : bool | Sequence[bool], optional
            Sort order per column, by default True.
        inplace : bool, optional
            Whether to mutate in place or return a sorted copy, by default True.
        **kwargs : Any
            Backend-specific keyword arguments forwarded to the concrete sorter.

        Returns
        -------
        Self
            The sorted AgentSet (or a sorted copy when ``inplace=False``).
        """
        ...

    def __setitem__(
        self,
        key: str
        | Collection[str]
        | AgentMask
        | tuple[AgentMask, str | Collection[str]],
        values: Any,
    ) -> None:
        """Set values using [] syntax, delegating to set()."""
        if isinstance(key, tuple):
            self.set(mask=key[0], attr_names=key[1], values=values)
        else:
            if isinstance(key, str) or (
                isinstance(key, Collection) and all(isinstance(k, str) for k in key)
            ):
                try:
                    self.set(attr_names=key, values=values)
                except KeyError:  # key may actually be a mask
                    self.set(attr_names=None, mask=key, values=values)
            else:
                self.set(attr_names=None, mask=key, values=values)
