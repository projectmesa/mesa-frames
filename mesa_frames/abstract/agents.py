"""
Abstract base classes for agent containers in mesa-frames.

This module defines the core abstractions for agent containers in the mesa-frames
extension. It provides the foundation for implementing agent storage and
manipulation using DataFrame-based approaches.

Classes:
    AgentContainer(CopyMixin):
        An abstract base class that defines the common interface for all agent
        containers in mesa-frames. It inherits from CopyMixin to provide fast
        copying functionality.

    AgentSetDF(AgentContainer, DataFrameMixin):
        An abstract base class for agent sets that use DataFrames as the underlying
        storage mechanism. It inherits from both AgentContainer and DataFrameMixin
        to combine agent container functionality with DataFrame operations.

These abstract classes are designed to be subclassed by concrete implementations
that use Polars library as their backend.

Usage:
    These classes should not be instantiated directly. Instead, they should be
    subclassed to create concrete implementations:

    from mesa_frames.abstract.agents import AgentSetDF

    class AgentSetPolars(AgentSetDF):
        def __init__(self, model):
            super().__init__(model)
            # Implementation using polars DataFrame
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
from typing import TYPE_CHECKING, Literal

from numpy.random import Generator
from typing_extensions import Any, Self, overload

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

if TYPE_CHECKING:
    from mesa_frames.abstract.space import SpaceDF
    from mesa_frames.concrete.agents import AgentSetDF
    from mesa_frames.concrete.model import ModelDF


class AgentContainer(CopyMixin):
    """An abstract class for containing agents. Defines the common interface for AgentSetDF and AgentsDF."""

    _copy_only_reference: list[str] = [
        "_model",
    ]
    _model: ModelDF

    @abstractmethod
    def __init__(self) -> None: ...

    def discard(
        self,
        agents: IdsLike | AgentSetDF | Collection[AgentSetDF],
        inplace: bool = True,
    ) -> Self:
        """Remove agents from the AgentContainer. Does not raise an error if the agent is not found.

        Parameters
        ----------
        agents : IdsLike | 'AgentSetDF' | Collection['AgentSetDF']
            The agents to remove
        inplace : bool
            Whether to remove the agent in place. Defaults to True.

        Returns
        -------
        Self
        """
        with suppress(KeyError, ValueError):
            return self.remove(agents, inplace=inplace)
        return self._get_obj(inplace)

    @abstractmethod
    def add(
        self,
        agents: DataFrameInput | AgentSetDF | Collection[AgentSetDF],
        inplace: bool = True,
    ) -> Self:
        """Add agents to the AgentContainer.

        Parameters
        ----------
        agents : DataFrameInput | AgentSetDF | Collection[AgentSetDF]
            The agents to add.
        inplace : bool
            Whether to add the agents in place. Defaults to True.

        Returns
        -------
        Self
            The updated AgentContainer.
        """
        ...

    @overload
    @abstractmethod
    def contains(self, agents: int) -> bool: ...

    @overload
    @abstractmethod
    def contains(self, agents: AgentSetDF | IdsLike) -> BoolSeries: ...

    @abstractmethod
    def contains(self, agents: IdsLike) -> bool | BoolSeries:
        """Check if agents with the specified IDs are in the AgentContainer.

        Parameters
        ----------
        agents : IdsLike
            The ID(s) to check for.

        Returns
        -------
        bool | BoolSeries
            True if the agent is in the AgentContainer, False otherwise.
        """

    @overload
    @abstractmethod
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
    @abstractmethod
    def do(
        self,
        method_name: str,
        *args,
        mask: AgentMask | None = None,
        return_results: Literal[True],
        inplace: bool = True,
        **kwargs,
    ) -> Any | dict[AgentSetDF, Any]: ...

    @abstractmethod
    def do(
        self,
        method_name: str,
        *args: Any,
        mask: AgentMask | None = None,
        return_results: bool = False,
        inplace: bool = True,
        **kwargs: Any,
    ) -> Self | Any | dict[AgentSetDF, Any]:
        """Invoke a method on the AgentContainer.

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
        Self | Any | dict[AgentSetDF, Any]
            The updated AgentContainer or the result of the method.
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

    @abstractmethod
    def get(
        self,
        attr_names: str | Collection[str] | None = None,
        mask: AgentMask | None = None,
    ) -> Series | dict[str, Series] | DataFrame | dict[str, DataFrame]:
        """Retrieve the value of a specified attribute for each agent in the AgentContainer.

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

    @abstractmethod
    def remove(
        self,
        agents: IdsLike | AgentSetDF | Collection[AgentSetDF],
        inplace: bool = True,
    ) -> Self:
        """Remove the agents from the AgentContainer.

        Parameters
        ----------
        agents : IdsLike | AgentSetDF | Collection[AgentSetDF]
            The agents to remove.
        inplace : bool, optional
            Whether to remove the agent in place.

        Returns
        -------
        Self
            The updated AgentContainer.
        """
        ...

    @abstractmethod
    def select(
        self,
        mask: AgentMask | None = None,
        filter_func: Callable[[Self], AgentMask] | None = None,
        n: int | None = None,
        negate: bool = False,
        inplace: bool = True,
    ) -> Self:
        """Select agents in the AgentContainer based on the given criteria.

        Parameters
        ----------
        mask : AgentMask | None, optional
            The AgentMask of agents to be selected, by default None
        filter_func : Callable[[Self], AgentMask] | None, optional
            A function which takes as input the AgentContainer and returns a AgentMask, by default None
        n : int | None, optional
            The maximum number of agents to be selected, by default None
        negate : bool, optional
            If the selection should be negated, by default False
        inplace : bool, optional
            If the operation should be performed on the same object, by default True

        Returns
        -------
        Self
            A new or updated AgentContainer.
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
        """Set the value of a specified attribute or attributes for each agent in the mask in AgentContainer.

        Parameters
        ----------
        attr_names : DataFrameInput | str | Collection[str]
            The key can be:
            - A string: sets the specified column of the agents in the AgentContainer.
            - A collection of strings: sets the specified columns of the agents in the AgentContainer.
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
        """
        ...

    @abstractmethod
    def shuffle(self, inplace: bool = False) -> Self:
        """Shuffles the order of agents in the AgentContainer.

        Parameters
        ----------
        inplace : bool
            Whether to shuffle the agents in place.

        Returns
        -------
        Self
            A new or updated AgentContainer.
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
        Sorts the agents in the agent set based on the given criteria.

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
            A new or updated AgentContainer.
        """

    def __add__(
        self, other: DataFrameInput | AgentSetDF | Collection[AgentSetDF]
    ) -> Self:
        """Add agents to a new AgentContainer through the + operator.

        Parameters
        ----------
        other : DataFrameInput | AgentSetDF | Collection[AgentSetDF]
            The agents to add.

        Returns
        -------
        Self
            A new AgentContainer with the added agents.
        """
        return self.add(agents=other, inplace=False)

    def __contains__(self, agents: int | AgentSetDF) -> bool:
        """Check if an agent is in the AgentContainer.

        Parameters
        ----------
        agents : int | AgentSetDF
            The ID(s) or AgentSetDF to check for.

        Returns
        -------
        bool
            True if the agent is in the AgentContainer, False otherwise.
        """
        return self.contains(agents=agents)

    @overload
    def __getitem__(
        self, key: str | tuple[AgentMask, str]
    ) -> Series | dict[str, Series]: ...

    @overload
    def __getitem__(
        self, key: AgentMask | Collection[str] | tuple[AgentMask, Collection[str]]
    ) -> DataFrame | dict[str, DataFrame]: ...

    def __getitem__(
        self,
        key: (
            str
            | Collection[str]
            | AgentMask
            | tuple[AgentMask, str]
            | tuple[AgentMask, Collection[str]]
        ),
    ) -> Series | DataFrame | dict[str, Series] | dict[str, DataFrame]:
        """Implement the [] operator for the AgentContainer.

        The key can be:
        - An attribute or collection of attributes (eg. AgentContainer["str"], AgentContainer[["str1", "str2"]]): returns the specified column(s) of the agents in the AgentContainer.
        - An AgentMask (eg. AgentContainer[AgentMask]): returns the agents in the AgentContainer that satisfy the AgentMask.
        - A tuple (eg. AgentContainer[AgentMask, "str"]): returns the specified column of the agents in the AgentContainer that satisfy the AgentMask.

        Parameters
        ----------
        key : str| Collection[str] | AgentMask | tuple[AgentMask, str] | tuple[AgentMask, Collection[str]]
            The key to retrieve.

        Returns
        -------
        Series | DataFrame | dict[str, Series] | dict[str, DataFrame]
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

    def __iadd__(
        self, other: DataFrameInput | AgentSetDF | Collection[AgentSetDF]
    ) -> Self:
        """Add agents to the AgentContainer through the += operator.

        Parameters
        ----------
        other : DataFrameInput | AgentSetDF | Collection[AgentSetDF]
            The agents to add.

        Returns
        -------
        Self
            The updated AgentContainer.
        """
        return self.add(agents=other, inplace=True)

    def __isub__(self, other: IdsLike | AgentSetDF | Collection[AgentSetDF]) -> Self:
        """Remove agents from the AgentContainer through the -= operator.

        Parameters
        ----------
        other : IdsLike | AgentSetDF | Collection[AgentSetDF]
            The agents to remove.

        Returns
        -------
        Self
            The updated AgentContainer.
        """
        return self.discard(other, inplace=True)

    def __sub__(self, other: IdsLike | AgentSetDF | Collection[AgentSetDF]) -> Self:
        """Remove agents from a new AgentContainer through the - operator.

        Parameters
        ----------
        other : IdsLike | AgentSetDF | Collection[AgentSetDF]
            The agents to remove.

        Returns
        -------
        Self
            A new AgentContainer with the removed agents.
        """
        return self.discard(other, inplace=False)

    def __setitem__(
        self,
        key: (
            str | Collection[str] | AgentMask | tuple[AgentMask, str | Collection[str]]
        ),
        values: Any,
    ) -> None:
        """Implement the [] operator for setting values in the AgentContainer.

        The key can be:
        - A string (eg. AgentContainer["str"]): sets the specified column of the agents in the AgentContainer.
        - A list of strings(eg. AgentContainer[["str1", "str2"]]): sets the specified columns of the agents in the AgentContainer.
        - A tuple (eg. AgentContainer[AgentMask, "str"]): sets the specified column of the agents in the AgentContainer that satisfy the AgentMask.
        - A AgentMask (eg. AgentContainer[AgentMask]): sets the attributes of the agents in the AgentContainer that satisfy the AgentMask.

        Parameters
        ----------
        key : str | Collection[str] | AgentMask | tuple[AgentMask, str | Collection[str]]
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
        """Fallback for retrieving attributes of the AgentContainer. Retrieve an attribute of the underlying DataFrame(s).

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
        """Iterate over the agents in the AgentContainer.

        Returns
        -------
        Iterator[dict[str, Any]]
            An iterator over the agents.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Get the number of agents in the AgentContainer.

        Returns
        -------
        int
            The number of agents in the AgentContainer.
        """
        ...

    @abstractmethod
    def __repr__(self) -> str:
        """Get a string representation of the DataFrame in the AgentContainer.

        Returns
        -------
        str
            A string representation of the DataFrame in the AgentContainer.
        """
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator:
        """Iterate over the agents in the AgentContainer in reverse order.

        Returns
        -------
        Iterator
            An iterator over the agents in reverse order.
        """
        ...

    @abstractmethod
    def __str__(self) -> str:
        """Get a string representation of the agents in the AgentContainer.

        Returns
        -------
        str
            A string representation of the agents in the AgentContainer.
        """
        ...

    @property
    def model(self) -> ModelDF:
        """The model that the AgentContainer belongs to.

        Returns
        -------
        ModelDF
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
    def space(self) -> SpaceDF:
        """The space of the model.

        Returns
        -------
        SpaceDF
        """
        return self.model.space

    @property
    @abstractmethod
    def agents(self) -> DataFrame | dict[str, DataFrame]:
        """The agents in the AgentContainer.

        Returns
        -------
        DataFrame | dict[str, DataFrame]
        """

    @agents.setter
    @abstractmethod
    def agents(self, agents: DataFrame | list[AgentSetDF]) -> None:
        """Set the agents in the AgentContainer.

        Parameters
        ----------
        agents : DataFrame | list[AgentSetDF]
        """

    @property
    @abstractmethod
    def active_agents(self) -> DataFrame | dict[str, DataFrame]:
        """The active agents in the AgentContainer.

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
        """Set the active agents in the AgentContainer.

        Parameters
        ----------
        mask : AgentMask
            The mask to apply.
        """
        self.select(mask=mask, inplace=True)

    @property
    @abstractmethod
    def inactive_agents(self) -> DataFrame | dict[AgentSetDF, DataFrame]:
        """The inactive agents in the AgentContainer.

        Returns
        -------
        DataFrame | dict[AgentSetDF, DataFrame]
        """

    @property
    @abstractmethod
    def index(self) -> Index | dict[AgentSetDF, Index]:
        """The ids in the AgentContainer.

        Returns
        -------
        Index | dict[AgentSetDF, Index]
        """
        ...

    @property
    @abstractmethod
    def pos(self) -> DataFrame | dict[str, DataFrame]:
        """The position of the agents in the AgentContainer.

        Returns
        -------
        DataFrame | dict[str, DataFrame]
        """
        ...


class AgentSetDF(AgentContainer, DataFrameMixin):
    """The AgentSetDF class is a container for agents of the same type.

    Parameters
    ----------
    model : ModelDF
        The model that the agent set belongs to.
    """

    _agents: DataFrame  # The agents in the AgentSetDF
    _mask: (
        AgentMask  # The underlying mask used for the active agents in the AgentSetDF.
    )
    _model: ModelDF  # The model that the AgentSetDF belongs to.

    @abstractmethod
    def __init__(self, model: ModelDF) -> None: ...

    @abstractmethod
    def add(
        self,
        agents: DataFrameInput,
        inplace: bool = True,
    ) -> Self:
        """Add agents to the AgentSetDF.

        Agents can be the input to the DataFrame constructor. So, the input can be:
        - A DataFrame: adds the agents from the DataFrame.
        - A dictionary: keys should be attributes and values should be the values to add.
        - A Sequence[Sequence]: each inner sequence should be one single agent to add.

        Parameters
        ----------
        agents : DataFrameInput
            The agents to add.
        inplace : bool, optional
            If True, perform the operation in place, by default True

        Returns
        -------
        Self
            A new AgentContainer with the added agents.
        """
        ...

    def discard(self, agents: IdsLike, inplace: bool = True) -> Self:
        """Remove an agent from the AgentSetDF. Does not raise an error if the agent is not found.

        Parameters
        ----------
        agents : IdsLike
            The ids to remove
        inplace : bool, optional
            Whether to remove the agent in place, by default True

        Returns
        -------
        Self
            The updated AgentSetDF.
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
        if len(masked_df) == len(self._agents):
            obj = self._get_obj(inplace)
            method = getattr(obj, method_name)
            result = method(*args, **kwargs)
        else:  # If the mask is not empty, we need to create a new masked AgentSetDF and concatenate the AgentSetDFs at the end
            obj = self._get_obj(inplace=False)
            obj._agents = masked_df
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
        """Run a single step of the AgentSetDF. This method should be overridden by subclasses."""
        ...

    def remove(self, agents: IdsLike, inplace: bool = True) -> Self:
        if agents is None or (isinstance(agents, Iterable) and len(agents) == 0):
            return self._get_obj(inplace)
        agents = self._df_index(self._get_masked_df(agents), "unique_id")
        agentsdf = self.model.agents.remove(agents, inplace=inplace)
        # TODO: Refactor AgentsDF to return dict[str, AgentSetDF] instead of dict[AgentSetDF, DataFrame]
        # And assign a name to AgentSetDF? This has to be replaced by a nicer API of AgentsDF
        for agentset in agentsdf.agents.keys():
            if isinstance(agentset, self.__class__):
                return agentset

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
        """Remove an agent from the DataFrame of the AgentSetDF. Gets called by self.model.agents.remove and self.model.agents.discard.

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

    def __add__(self, other: DataFrame | Sequence[Any] | dict[str, Any]) -> Self:
        """Add agents to a new AgentSetDF through the + operator.

        Other can be:
        - A DataFrame: adds the agents from the DataFrame.
        - A Sequence[Any]: should be one single agent to add.
        - A dictionary: keys should be attributes and values should be the values to add.

        Parameters
        ----------
        other : DataFrame | Sequence[Any] | dict[str, Any]
            The agents to add.

        Returns
        -------
        Self
            A new AgentContainer with the added agents.
        """
        return super().__add__(other)

    def __iadd__(self, other: DataFrame | Sequence[Any] | dict[str, Any]) -> Self:
        """
        Add agents to the AgentSetDF through the += operator.

        Other can be:
        - A DataFrame: adds the agents from the DataFrame.
        - A Sequence[Any]: should be one single agent to add.
        - A dictionary: keys should be attributes and values should be the values to add.

        Parameters
        ----------
        other : DataFrame | Sequence[Any] | dict[str, Any]
            The agents to add.

        Returns
        -------
        Self
            The updated AgentContainer.
        """
        return super().__iadd__(other)

    @abstractmethod
    def __getattr__(self, name: str) -> Any:
        if __debug__:  # Only execute in non-optimized mode
            if name == "_agents":
                raise AttributeError(
                    "The _agents attribute is not set. You probably forgot to call super().__init__ in the __init__ method."
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
        return len(self._agents)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}\n {str(self._agents)}"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}\n {str(self._agents)}"

    def __reversed__(self) -> Iterator:
        return reversed(self._agents)

    @property
    def agents(self) -> DataFrame:
        return self._agents

    @agents.setter
    def agents(self, agents: DataFrame) -> None:
        """Set the agents in the AgentSetDF.

        Parameters
        ----------
        agents : DataFrame
            The agents to set.
        """
        self._agents = agents

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
        pos = self._df_constructor(self.space.agents, index_cols="agent_id")
        pos = self._df_get_masked_df(df=pos, index_cols="agent_id", mask=self.index)
        pos = self._df_reindex(
            pos, self.index, new_index_cols="unique_id", original_index_cols="agent_id"
        )
        return pos
