from __future__ import annotations  # PEP 563: postponed evaluation of type annotations

from abc import abstractmethod
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Literal

from numpy.random import Generator
from typing_extensions import Any, Self, overload

from mesa_frames.abstract.mixin import CopyMixin
from mesa_frames.types import BoolSeries, DataFrame, IdsLike, Index, MaskLike, Series

if TYPE_CHECKING:
    from mesa_frames.concrete.agents import AgentSetDF
    from mesa_frames.concrete.model import ModelDF


class AgentContainer(CopyMixin):
    """An abstract class for containing agents. Defines the common interface for AgentSetDF and AgentsDF.

    Attributes
    ----------
    _model : ModelDF
        The model that the AgentContainer belongs to.

    Methods
    -------
    copy(deep: bool = False, memo: dict | None = None) -> Self
        Create a copy of the AgentContainer.
    discard(ids: IdsLike, inplace: bool = True) -> Self
        Removes an agent from the AgentContainer. Does not raise an error if the agent is not found.
    add(other: Any, inplace: bool = True) -> Self
        Add agents to the AgentContainer.
    contains(ids: IdsLike) -> bool | BoolSeries
        Check if agents with the specified IDs are in the AgentContainer.
    do(method_name: str, *args, return_results: bool = False, inplace: bool = True, **kwargs) -> Self | Any | dict[str, Any]
        Invoke a method on the AgentContainer.
    get(attr_names: str | Collection[str] | None = None, mask: MaskLike | None = None) -> Series | DataFrame | dict[str, Series] | dict[str, DataFrame]
        Retrieve the value of a specified attribute for each agent in the AgentContainer.
    remove(ids: IdsLike, inplace: bool = True) -> Self
        Removes an agent from the AgentContainer.
    select(mask: MaskLike | None = None, filter_func: Callable[[Self], MaskLike] | None = None, n: int | None = None, negate: bool = False, inplace: bool = True) -> Self
        Select agents in the AgentContainer based on the given criteria.
    set(attr_names: str | dict[str, Any] | Collection[str], values: Any | None = None, mask: MaskLike | None = None, inplace: bool = True) -> Self
        Sets the value of a specified attribute or attributes for each agent in the mask in AgentContainer.
    shuffle(inplace: bool = False) -> Self
        Shuffles the order of agents in the AgentContainer.
    sort(by: str | Sequence[str], ascending: bool | Sequence[bool] = True, inplace: bool = True, **kwargs) -> Self
        Sorts the agents in the agent set based on the given criteria.

    Properties
    ----------
    model : ModelDF
        Get the model associated with the AgentContainer.
    random : Generator
        Get the random number generator associated with the model.
    agents : DataFrame | dict[str, DataFrame]
        Get or set the agents in the AgentContainer.
    active_agents : DataFrame | dict[str, DataFrame]
        Get or set the active agents in the AgentContainer.
    inactive_agents : DataFrame | dict[str, DataFrame]
        Get the inactive agents in the AgentContainer.
    """

    _copy_only_reference: list[str] = [
        "_model",
    ]
    _model: ModelDF

    @abstractmethod
    def __init__(self) -> None: ...

    def discard(self, agents, inplace: bool = True) -> Self:
        """Removes agents from the AgentContainer. Does not raise an error if the agent is not found.

        Parameters
        ----------
        agents
            The agents to remove
        inplace : bool
            Whether to remove the agent in place. Defaults to True.

        Returns
        ----------
        Self
        """
        with suppress(KeyError, ValueError):
            return self.remove(agents, inplace=inplace)
        return self._get_obj(inplace)

    @abstractmethod
    def add(self, agents, inplace: bool = True) -> Self:
        """Add agents to the AgentContainer.

        Parameters
        ----------
        agents
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
        mask: MaskLike | None = None,
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
        mask: MaskLike | None = None,
        return_results: Literal[True],
        inplace: bool = True,
        **kwargs,
    ) -> Any | dict[AgentSetDF, Any]: ...

    @abstractmethod
    def do(
        self,
        method_name: str,
        *args,
        mask: MaskLike | None = None,
        return_results: bool = False,
        inplace: bool = True,
        **kwargs,
    ) -> Self | Any | dict[AgentSetDF, Any]:
        """Invoke a method on the AgentContainer.

        Parameters
        ----------
        method_name : str
            The name of the method to invoke.
        *args : Any
            Positional arguments to pass to the method
        mask : MaskLike, optional
            The subset of agents on which to apply the method
        return_results : bool, optional
            Whether to return the result of the method, by default False
        inplace : bool, optional
            Whether the operation should be done inplace, by default False

        Returns
        -------
        Self | Any
            The updated AgentContainer or the result of the method.
        """
        ...

    @abstractmethod
    @overload
    def get(self, attr_names: str) -> Series | dict[str, Series]: ...

    @abstractmethod
    @overload
    def get(self, attr_names: Collection[str]) -> DataFrame | dict[str, DataFrame]: ...

    @abstractmethod
    def get(
        self,
        attr_names: str | Collection[str] | None = None,
        mask: MaskLike | None = None,
    ) -> Series | DataFrame | dict[str, Series] | dict[str, DataFrame]:
        """Retrieves the value of a specified attribute for each agent in the AgentContainer.

        Parameters
        ----------
        attr_names : str | Collection[str] | None
            The attributes to retrieve. If None, all attributes are retrieved. Defaults to None.
        MaskLike : MaskLike | None
            The MaskLike of agents to retrieve the attribute for. If None, attributes of all agents are returned. Defaults to None.

        Returns
        ----------
        Series | DataFrame | dict[str, Series | DataFrame]
            The attribute values.
        """
        ...

    @abstractmethod
    def remove(self, agents, inplace: bool = True) -> Self:
        """Removes the agents from the AgentContainer

        Parameters
        ----------
        agents
            The agents to remove.
        inplace : bool
            Whether to remove the agent in place.

        Returns
        ----------
        Self
            The updated AgentContainer.
        """
        ...

    @abstractmethod
    def select(
        self,
        mask: MaskLike | None = None,
        filter_func: Callable[[Self], MaskLike] | None = None,
        n: int | None = None,
        negate: bool = False,
        inplace: bool = True,
    ) -> Self:
        """Select agents in the AgentContainer based on the given criteria.

        Parameters
        ----------
        mask : MaskLike | None, optional
            The MaskLike of agents to be selected, by default None
        filter_func : Callable[[Self], MaskLike] | None, optional
            A function which takes as input the AgentContainer and returns a MaskLike, by default None
        n : int, optional
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
        mask: MaskLike | None = None,
        inplace: bool = True,
    ) -> Self: ...

    @abstractmethod
    @overload
    def set(
        self,
        attr_names: str | Collection[str],
        values: Any,
        mask: MaskLike | None = None,
        inplace: bool = True,
    ) -> Self: ...

    @abstractmethod
    def set(
        self,
        attr_names: str | dict[str, Any] | Collection[str],
        values: Any | None = None,
        mask: MaskLike | None = None,
        inplace: bool = True,
    ) -> Self:
        """Sets the value of a specified attribute or attributes for each agent in the mask in AgentContainer.

        Parameters
        ----------
        attr_names : str | dict[str, Any] | Collection[str] | None
            The key can be:
            - A string: sets the specified column of the agents in the AgentContainer.
            - A collection of strings: sets the specified columns of the agents in the AgentContainer.
            - A dictionary: keys should be attributes and values should be the values to set. Value should be None.
        value : Any | None
            The value to set the attribute to. If None, attr_names must be a dictionary.
        mask : MaskLike | None
            The MaskLike of agents to set the attribute for.
        inplace : bool
            Whether to set the attribute in place.

        Returns
        ----------
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
        ----------
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
        ----------
        Self
            A new or updated AgentContainer.
        """

    def __add__(self, other) -> Self:
        return self.add(agents=other, inplace=False)

    def __contains__(self, agents: int | IdsLike | AgentSetDF) -> bool:
        """Check if an agent is in the AgentContainer.

        Parameters
        ----------
        id : int | IdsLike | AgentSetDF
            The ID(s) or AgentSetDF to check for.

        Returns
        -------
        bool
            True if the agent is in the AgentContainer, False otherwise.
        """
        return self.contains(agents=agents)

    def __getitem__(
        self,
        key: (
            str
            | Collection[str]
            | MaskLike
            | tuple[MaskLike, str]
            | tuple[MaskLike, Collection[str]]
        ),
    ) -> Series | DataFrame | dict[str, Series] | dict[str, DataFrame]:
        """Implements the [] operator for the AgentContainer.

        The key can be:
        - An attribute or collection of attributes (eg. AgentContainer["str"], AgentContainer[["str1", "str2"]]): returns the specified column(s) of the agents in the AgentContainer.
        - A MaskLike (eg. AgentContainer[MaskLike]): returns the agents in the AgentContainer that satisfy the MaskLike.
        - A tuple (eg. AgentContainer[MaskLike, "str"]): returns the specified column of the agents in the AgentContainer that satisfy the MaskLike.

        Parameters
        ----------
        key : Attributes | MaskLike | tuple[MaskLike, Attributes]
            The key to retrieve.

        Returns
        -------
        Series | DataFrame
            The attribute values.
        """
        # TODO: fix types
        if isinstance(key, tuple):
            return self.get(mask=key[0], attr_names=key[1])
        else:
            try:
                return self.get(attr_names=key)
            except KeyError:
                return self.get(mask=key)

    def __iadd__(self, other) -> Self:
        """Add agents to the AgentContainer through the += operator.

        Parameters
        ----------
        other
            The agents to add.

        Returns
        -------
        Self
            The updated AgentContainer.
        """
        return self.add(agents=other, inplace=True)

    def __isub__(self, other: AgentSetDF | IdsLike) -> Self:
        """Remove agents from the AgentContainer through the -= operator.

        Parameters
        ----------
        other : MaskLike
            The agents to remove.

        Returns
        -------
        Self
            The updated AgentContainer.
        """
        return self.discard(other, inplace=True)

    def __sub__(self, other: AgentSetDF | IdsLike) -> Self:
        """Remove agents from a new AgentContainer through the - operator.

        Parameters
        ----------
        other : DataFrame | ListLike
            The agents to remove.

        Returns
        -------
        Self
            A new AgentContainer with the removed agents.
        """
        return self.discard(other, inplace=False)

    def __setitem__(
        self,
        key: str | Collection[str] | MaskLike | tuple[MaskLike, str | Collection[str]],
        values: Any,
    ) -> None:
        """Implement the [] operator for setting values in the AgentContainer.

        The key can be:
        - A string (eg. AgentContainer["str"]): sets the specified column of the agents in the AgentContainer.
        - A list of strings(eg. AgentContainer[["str1", "str2"]]): sets the specified columns of the agents in the AgentContainer.
        - A tuple (eg. AgentContainer[MaskLike, "str"]): sets the specified column of the agents in the AgentContainer that satisfy the MaskLike.
        - A MaskLike (eg. AgentContainer[MaskLike]): sets the attributes of the agents in the AgentContainer that satisfy the MaskLike.

        Parameters
        ----------
        key : str | list[str] | MaskLike | tuple[MaskLike, str | list[str]]
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
                except KeyError:  # key=MaskLike
                    self.set(attr_names=None, mask=key, values=values)
            else:
                self.set(attr_names=None, mask=key, values=values)

    @abstractmethod
    def __getattr__(self, name: str) -> Any | dict[str, Any]:
        """Fallback for retrieving attributes of the AgentContainer. Retrieves an attribute of the underlying DataFrame(s).

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
        Iterator
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
        Generator"""
        return self.model.random

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
        DataFrame
        """

    @active_agents.setter
    @abstractmethod
    def active_agents(
        self,
        mask: MaskLike,
    ) -> None:
        """Set the active agents in the AgentContainer.

        Parameters
        ----------
        mask : MaskLike
            The mask to apply.
        """
        self.select(mask=mask, inplace=True)

    @property
    @abstractmethod
    def inactive_agents(self) -> DataFrame | dict[str, DataFrame]:
        """The inactive agents in the AgentContainer.

        Returns
        -------
        DataFrame
        """


class AgentSetDF(AgentContainer):
    """The AgentSetDF class is a container for agents of the same type.

    Attributes
    ----------
    _agents : DataFrame
        The agents in the AgentSetDF.
    _copy_only_reference : list[str]
        A list of attributes to copy with a reference only.
    _copy_with_method : dict[str, tuple[str, list[str]]]
        A dictionary of attributes to copy with a specified method and arguments.
    _mask : MaskLike
        The underlying mask used for the active agents in the AgentSetDF.
    _model : ModelDF
        The model that the AgentSetDF belongs to.

    Methods
    -------
    __init__(self, model: ModelDF) -> None
        Create a new AgentSetDF.
    add(self, other: DataFrame | Sequence[Any] | dict[str, Any], inplace: bool = True) -> Self
        Add agents to the AgentSetDF.
    contains(self, ids: Hashable | Collection[Hashable]) -> bool | BoolSeries
        Check if agents with the specified IDs are in the AgentSetDF.
    copy(self, deep: bool = False, memo: dict | None = None) -> Self
        Create a copy of the AgentSetDF.
    discard(self, ids: MaskLike, inplace: bool = True) -> Self
        Removes an agent from the AgentSetDF. Does not raise an error if the agent is not found.
    do(self, method_name: str, *args, return_results: bool = False, inplace: bool = True, **kwargs) -> Self | Any
        Invoke a method on the AgentSetDF.
    get(self, attr_names: str | Collection[str] | None = None, mask: MaskLike | None = None) -> Series | DataFrame
        Retrieve the value of a specified attribute for each agent in the AgentSetDF.
    remove(self, ids: MaskLike, inplace: bool = True) -> Self
        Removes an agent from the AgentSetDF.
    select(self, mask: MaskLike | None = None, filter_func: Callable[[Self], MaskLike] | None = None, n: int | None = None, negate: bool = False, inplace: bool = True) -> Self
        Select agents in the AgentSetDF based on the given criteria.
    set(self, attr_names: str | dict[str, Any] | Collection[str], values: Any | None = None, mask: MaskLike | None = None, inplace: bool = True) -> Self
        Sets the value of a specified attribute or attributes for each agent in the mask in AgentSetDF.
    shuffle(self, inplace: bool = False) -> Self
        Shuffles the order of agents in the AgentSetDF.
    sort(self, by: str | Sequence[str], ascending: bool | Sequence[bool] = True, inplace: bool = True, **kwargs) -> Self
        Sorts the agents in the AgentSetDF based on the given criteria.
    _get_obj(self, inplace: bool) -> Self
        Get the appropriate object, either the current instance or a copy, based on the `inplace` parameter.
    __add__(self, other: Self | DataFrame | Sequence[Any] | dict[str, Any]) -> Self
        Add agents to a new AgentSetDF through the + operator.
    __iadd__(self, other: Self | DataFrame | Sequence[Any] | dict[str, Any]) -> Self
        Add agents to the AgentSetDF through the += operator.
    __getattr__(self, name: str) -> Any
        Retrieve an attribute of the AgentSetDF.
    __getitem__(self, key: str | Collection[str] | MaskLike | tuple[MaskLike, str] | tuple[MaskLike, Collection[str]]) -> Series | DataFrame
        Retrieve an item from the AgentSetDF.
    __iter__(self) -> Iterator
        Get an iterator for the agents in the AgentSetDF.
    __len__(self) -> int
        Get the number of agents in the AgentSetDF.
    __repr__(self) -> str
        Get the string representation of the AgentSetDF.
    __reversed__(self) -> Iterator
        Get a reversed iterator for the agents in the AgentSetDF.
    __str__(self) -> str
        Get the string representation of the AgentSetDF.

    Properties
    ----------
    active_agents(self) -> DataFrame
        Get the active agents in the AgentSetDF.
    agents(self) -> DataFrame
        Get or set the agents in the AgentSetDF.
    inactive_agents(self) -> DataFrame
        Get the inactive agents in the AgentSetDF.
    model(self) -> ModelDF
        Get the model associated with the AgentSetDF.
    random(self) -> Generator
        Get the random number generator associated with the model.
    """

    _agents: DataFrame
    _mask: MaskLike
    _model: ModelDF

    @abstractmethod
    def __init__(self, model: ModelDF) -> None:
        """Create a new AgentSetDF.

        Parameters
        ----------
        model : ModelDF
            The model that the agent set belongs to.

        Returns
        -------
        None
        """
        ...

    @abstractmethod
    def add(
        self, agents: DataFrame | Sequence[Any] | dict[str, Any], inplace: bool = True
    ) -> Self:
        """Add agents to the AgentSetDF

        Other can be:
        - A DataFrame: adds the agents from the DataFrame.
        - A Sequence[Any]: should be one single agent to add.
        - A dictionary: keys should be attributes and values should be the values to add.

        Parameters
        ----------
        other : DataFrame | Sequence[Any] | dict[str, Any]
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
        return super().discard(agents, inplace)

    @overload
    def do(
        self,
        method_name: str,
        *args,
        mask: MaskLike | None = None,
        return_results: Literal[False] = False,
        inplace: bool = True,
        **kwargs,
    ) -> Self: ...

    @overload
    def do(
        self,
        method_name: str,
        *args,
        mask: MaskLike | None = None,
        return_results: Literal[True],
        inplace: bool = True,
        **kwargs,
    ) -> Any: ...

    def do(
        self,
        method_name: str,
        *args,
        mask: MaskLike | None = None,
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
        mask: MaskLike | None = None,
    ) -> Series: ...

    @abstractmethod
    @overload
    def get(
        self,
        attr_names: Collection[str] | None = None,
        mask: MaskLike | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    def get(
        self,
        attr_names: str | Collection[str] | None = None,
        mask: MaskLike | None = None,
    ) -> Series | DataFrame: ...

    @abstractmethod
    def remove(self, agents: IdsLike, inplace: bool = True) -> Self: ...

    @abstractmethod
    def _concatenate_agentsets(
        self,
        objs: Iterable[Self],
        duplicates_allowed: bool = True,
        keep_first_only: bool = True,
        original_masked_index: Index | None = None,
    ) -> Self: ...

    @abstractmethod
    def _get_bool_mask(self, mask: MaskLike) -> BoolSeries:
        """Get the equivalent boolean mask based on the input mask

        Parameters
        ----------
        mask : MaskLike

        Returns
        -------
        BoolSeries
        """
        ...

    @abstractmethod
    def _get_masked_df(self, mask: MaskLike) -> DataFrame:
        """Get the df filtered by the input mask

        Parameters
        ----------
        mask : MaskLike

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
        if name == "_agents":
            raise RuntimeError(
                "The _agents attribute is not set. You probably forgot to call super().__init__ in the __init__ method."
            )

    @overload
    def __getitem__(self, key: str | tuple[MaskLike, str]) -> Series | DataFrame: ...

    @overload
    def __getitem__(
        self, key: MaskLike | Collection[str] | tuple[MaskLike, Collection[str]]
    ) -> DataFrame: ...

    def __getitem__(
        self,
        key: (
            str
            | Collection[str]
            | MaskLike
            | tuple[MaskLike, str]
            | tuple[MaskLike, Collection[str]]
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
