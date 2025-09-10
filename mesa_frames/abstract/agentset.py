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
from typing import Any, Literal, Self, overload

from mesa_frames.abstract.agentsetregistry import AbstractAgentSetRegistry
from mesa_frames.abstract.mixin import DataFrameMixin
from mesa_frames.types_ import (
    AgentMask,
    BoolSeries,
    DataFrame,
    DataFrameInput,
    IdsLike,
    Index,
    Series,
)


class AbstractAgentSet(DataFrameMixin):
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
            A new AbstractAgentSetRegistry with the added agents.
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
        agents : mesa_frames.concrete.agents.AgentSetDF | IdsLike
            The ID(s) to check for.

        Returns
        -------
        bool | BoolSeries
            True if the agent is in the AgentSet, False otherwise.
        """
        ...

    @abstractmethod
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
    ) -> Series | DataFrame: ...

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
            A new AbstractAgentSetRegistry with the added agents.
        """
        return super().__add__(other)

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
            The updated AbstractAgentSetRegistry.
        """
        return super().__iadd__(other)

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
        attr = super().__getitem__(key)
        assert isinstance(attr, (Series, DataFrame, Index))
        return attr

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
    def active_agents(self) -> DataFrame: ...

    @property
    @abstractmethod
    def inactive_agents(self) -> DataFrame: ...

    @property
    def index(self) -> Index: ...

    @property
    def pos(self) -> DataFrame:
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
