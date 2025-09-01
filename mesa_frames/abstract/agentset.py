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


class AbstractAgentSet(AbstractAgentSetRegistry, DataFrameMixin):
    """The AbstractAgentSet class is a container for agents of the same type.

    Parameters
    ----------
    model : mesa_frames.concrete.model.Model
        The model that the agent set belongs to.
    """

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
        return super().discard(agents, inplace)

    @overload
    def do(
        self,
        method_name: str,
        *args,
        mask: AgentMask | None = None,
        return_results: Literal[False] = False,
        inplace: bool = True,
        **kwargs,
    ) -> Self: ...

    @overload
    def do(
        self,
        method_name: str,
        *args,
        mask: AgentMask | None = None,
        return_results: Literal[True],
        inplace: bool = True,
        **kwargs,
    ) -> Any: ...

    def do(
        self,
        method_name: str,
        *args,
        mask: AgentMask | None = None,
        return_results: bool = False,
        inplace: bool = True,
        **kwargs,
    ) -> Self | Any:
        masked_df = self._get_masked_df(mask)
        # If the mask is empty, we can use the object as is
        if len(masked_df) == len(self._df):
            obj = self._get_obj(inplace)
            method = getattr(obj, method_name)
            result = method(*args, **kwargs)
        else:  # If the mask is not empty, we need to create a new masked AbstractAgentSet and concatenate the AbstractAgentSets at the end
            obj = self._get_obj(inplace=False)
            obj._df = masked_df
            original_masked_index = obj._get_obj_copy(obj.index)
            method = getattr(obj, method_name)
            result = method(*args, **kwargs)
            obj._concatenate_agentsets(
                [self],
                duplicates_allowed=True,
                keep_first_only=True,
                original_masked_index=original_masked_index,
            )
            if inplace:
                for key, value in obj.__dict__.items():
                    setattr(self, key, value)
                obj = self
        if return_results:
            return result
        else:
            return obj

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

    def remove(self, agents: IdsLike | AgentMask, inplace: bool = True) -> Self:
        if isinstance(agents, str) and agents == "active":
            agents = self.active_agents
        if agents is None or (isinstance(agents, Iterable) and len(agents) == 0):
            return self._get_obj(inplace)
        agents = self._df_index(self._get_masked_df(agents), "unique_id")
        sets = self.model.sets.remove(agents, inplace=inplace)
        # TODO: Refactor AgentSetRegistry to return dict[str, AbstractAgentSet] instead of dict[AbstractAgentSet, DataFrame]
        # And assign a name to AbstractAgentSet? This has to be replaced by a nicer API of AgentSetRegistry
        for agentset in sets.df.keys():
            if isinstance(agentset, self.__class__):
                return agentset
        return self

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
