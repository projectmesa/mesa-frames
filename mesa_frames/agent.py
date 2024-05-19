from __future__ import annotations  # PEP 563: postponed evaluation of type annotations

from abc import ABC, abstractmethod
from contextlib import suppress
from copy import copy, deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterable,
    Literal,
    Self,
    Sequence,
    overload,
)

import pandas as pd
import polars as pl
from numpy import int64, ndarray
from pandas.core.arrays.base import ExtensionArray
from polars.datatypes import N_INFER_DEFAULT

from mesa_frames.model import ModelDF

# For AgentSetPandas.select
ArrayLike = ExtensionArray | ndarray
AnyArrayLike = ArrayLike | pd.Index | pd.Series
ListLike = AnyArrayLike | list | range

# For AgentSetPandas.drop
IndexLabel = Hashable | Sequence[Hashable]

# For AgentContainer.__getitem__ and AgentContainer.__setitem__
DataFrame = pd.DataFrame | pl.DataFrame

Series = pd.Series | pl.Series

BoolSeries = pd.Series | pl.Expr | pl.Series

PandasMaskLike = (
    Literal["active"] | Literal["all"] | pd.Series | pd.DataFrame | ListLike | Hashable
)

PolarsMaskLike = (
    Literal["active"]
    | Literal["all"]
    | pl.Expr
    | pl.Series
    | pl.DataFrame
    | ListLike
    | Hashable
)

MaskLike = PandasMaskLike | PolarsMaskLike

if TYPE_CHECKING:

    # For AgentSetDF
    from numpy.random import Generator

    ValueKeyFunc = Callable[[pd.Series], pd.Series | AnyArrayLike] | None

    # For AgentSetPolars
    from polars.type_aliases import (
        FrameInitTypes,
        IntoExpr,
        Orientation,
        SchemaDefinition,
        SchemaDict,
    )


### The AgentContainer class defines the interface for AgentSetDF and AgentsDF. It contains methods for selecting, shuffling, sorting, and manipulating agents. ###


class AgentContainer(ABC):
    model: ModelDF
    _mask: BoolSeries
    _skip_copy: list[str] = ["model", "_mask"]
    """An abstract class for containing agents. Defines the common interface for AgentSetDF and AgentsDF.
    
    Attributes
    ----------
    model : ModelDF
        The model to which the AgentContainer belongs.
    _mask : Series
        A boolean mask indicating which agents are active.
    _skip_copy : list[str]
        A list of attributes to skip during the copy process.    
    """

    def __new__(cls, model: ModelDF) -> Self:
        """Create a new AgentContainer object.

        Parameters
        ----------
        model : ModelDF
            The model to which the AgentContainer belongs.

        Returns
        -------
        Self
            A new AgentContainer object.
        """
        obj = super().__new__(cls)
        obj.model = model
        return obj

    def __add__(self, other: Self | DataFrame | ListLike | dict[str, Any]) -> Self:
        """Add agents to a new AgentContainer through the + operator.

        Other can be:
        - A Self: adds the agents from the other AgentContainer.
        - A DataFrame: adds the agents from the DataFrame.
        - A ListLike: should be one single agent to add.
        - A dictionary: keys should be attributes and values should be the values to add.

        Parameters
        ----------
        other : Self | DataFrame | ListLike | dict[str, Any]
            The agents to add.

        Returns
        -------
        Self
            A new AgentContainer with the added agents.
        """
        new_obj = deepcopy(self)
        return new_obj.add(other)

    def __iadd__(self, other: Self | DataFrame | ListLike | dict[str, Any]) -> Self:
        """Add agents to the AgentContainer through the += operator.

        Other can be:
        - A Self: adds the agents from the other AgentContainer.
        - A DataFrame: adds the agents from the DataFrame.
        - A ListLike: should be one single agent to add.
        - A dictionary: keys should be attributes and values should be the values to add.

        Parameters
        ----------
        other : Self | DataFrame | ListLike | dict[str, Any]
            The agents to add.

        Returns
        -------
        Self
            The updated AgentContainer.
        """
        return self.add(other)

    @abstractmethod
    def __contains__(self, id: Hashable) -> bool:
        """Check if an agent is in the AgentContainer.

        Parameters
        ----------
        id : Hashable
            The ID(s) to check for.

        Returns
        -------
        bool
            True if the agent is in the AgentContainer, False otherwise.
        """

    def __copy__(self) -> Self:
        """Create a shallow copy of the AgentContainer.

        Returns
        -------
        Self
            A shallow copy of the AgentContainer.
        """
        return self.copy(deep=False)

    def __deepcopy__(self, memo: dict) -> Self:
        """Create a deep copy of the AgentContainer.

        Parameters
        ----------
        memo : dict
            A dictionary to store the copied objects.

        Returns
        -------
        Self
            A deep copy of the AgentContainer.
        """
        return self.copy(deep=True, memo=memo)

    def __getattr__(self, name: str) -> Series:
        """Fallback for retrieving attributes of the AgentContainer. Retrieves an attribute column of the agents in the AgentContainer.

        Parameters
        ----------
        name : str
            The name of the attribute to retrieve.

        Returns
        -------
        Series
            The attribute values.
        """
        return self.get_attribute(name)

    @overload
    def __getitem__(self, key: str | tuple[MaskLike, str]) -> Series: ...

    @overload
    def __getitem__(self, key: list[str]) -> DataFrame: ...

    def __getitem__(
        self, key: str | list[str] | MaskLike | tuple[MaskLike, str | list[str]]
    ) -> Series | DataFrame:  # tuple is not generic so it is not type hintable
        """Implement the [] operator for the AgentContainer.

        The key can be:
        - A string (eg. AgentContainer["str"]): returns the specified column of the agents in the AgentContainer.
        - A list of strings(eg. AgentContainer[["str1", "str2"]]): returns the specified columns of the agents in the AgentContainer.
        - A tuple (eg. AgentContainer[mask, "str"]): returns the specified column of the agents in the AgentContainer that satisfy the mask.
        - A mask (eg. AgentContainer[mask]): returns the agents in the AgentContainer that satisfy the mask.

        Parameters
        ----------
        key : str | list[str] | MaskLike | tuple[MaskLike, str | list[str]]
            The key to retrieve.

        Returns
        -------
        Series | DataFrame
            The attribute values.
        """
        if isinstance(key, (str, list)):
            return self.get_attribute(attr_names=key)

        elif isinstance(key, tuple):
            return self.get_attribute(mask=key[0], attr_names=key[1])

        else:  # MaskLike
            return self.get_attribute(mask=key)

    @abstractmethod
    def __iter__(self) -> Iterable:
        """Iterate over the agents in the AgentContainer.

        Returns
        -------
        Iterable
            An iterator over the agents.
        """

    def __isub__(self, other: MaskLike) -> Self:
        """Remove agents from the AgentContainer through the -= operator.

        Parameters
        ----------
        other : Self | DataFrame | ListLike
            The agents to remove.

        Returns
        -------
        Self
            The updated AgentContainer.
        """
        return self.discard(other)

    @abstractmethod
    def __len__(self) -> int | dict[str, int]:
        """Get the number of agents in the AgentContainer.

        Returns
        -------
        int | dict[str, int]
            The number of agents in the AgentContainer.
        """

    @abstractmethod
    def __repr__(self) -> str:
        """Get a string representation of the DataFrame in the AgentContainer.

        Returns
        -------
        str
            A string representation of the DataFrame in the AgentContainer.
        """
        return repr(self.agents)

    def __setitem__(
        self,
        key: str | list[str] | MaskLike | tuple[MaskLike, str | list[str]],
        value: Any,
    ) -> None:
        """Implement the [] operator for setting values in the AgentContainer.

        The key can be:
        - A string (eg. AgentContainer["str"]): sets the specified column of the agents in the AgentContainer.
        - A list of strings(eg. AgentContainer[["str1", "str2"]]): sets the specified columns of the agents in the AgentContainer.
        - A tuple (eg. AgentContainer[mask, "str"]): sets the specified column of the agents in the AgentContainer that satisfy the mask.
        - A mask (eg. AgentContainer[mask]): sets the attributes of the agents in the AgentContainer that satisfy the mask.

        Parameters
        ----------
        key : str | list[str] | MaskLike | tuple[MaskLike, str | list[str]
            The key to set.
        """
        if isinstance(key, (str, list)):
            self.set_attribute(attr_names=key, value=value)

        elif isinstance(key, tuple):
            self.set_attribute(mask=key[0], attr_names=key[1], value=value)

        else:  # key=MaskLike
            self.set_attribute(mask=key, value=value)

    @abstractmethod
    def __str__(self) -> str:
        """Get a string representation of the DataFrame in the AgentContainer.

        Returns
        -------
        str
            A string representation of the DataFrame in the AgentContainer.
        """

    def __sub__(self, other: MaskLike) -> Self:
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
        new_obj = deepcopy(self)
        return new_obj.discard(other)

    @abstractmethod
    def __reversed__(self) -> Iterable:
        """Iterate over the agents in the AgentContainer in reverse order.

        Returns
        -------
        Iterable
            An iterator over the agents in reverse order.
        """

    @property
    @abstractmethod
    def agents(self) -> DataFrame:
        """The agents in the AgentContainer.

        Returns
        -------
        DataFrame
        """

    @property
    @abstractmethod
    def active_agents(self) -> DataFrame:
        """The active agents in the AgentContainer.

        Returns
        -------
        DataFrame
        """

    @active_agents.setter
    def active_agents(self, mask: MaskLike) -> None:
        """Set the active agents in the AgentContainer.

        Parameters
        ----------
        mask : MaskLike
            The mask to apply.
        """
        self.select(mask=mask)

    @property
    @abstractmethod
    def inactive_agents(self) -> DataFrame:
        """The inactive agents in the AgentContainer.

        Returns
        -------
        DataFrame
        """

    @property
    def random(self) -> Generator:
        """
        Provide access to the model's random number generator.

        Returns
        -------
        np.Generator
        """
        return self.model.random

    def _get_obj(self, inplace: bool) -> Self:
        """Get the object to perform operations on.

        Parameters
        ----------
        inplace : bool
            If inplace, return self. Otherwise, return a copy.

        Returns
        ----------
        Self
            The object to perform operations on.
        """
        if inplace:
            return self
        else:
            return deepcopy(self)

    @abstractmethod
    def contains(self, ids: MaskLike) -> BoolSeries:
        """Check if agents with the specified IDs are in the AgentContainer.

        Parameters
        ----------
        id : MaskLike
            The ID(s) to check for.

        Returns
        -------
        BoolSeries
        """

    def copy(
        self,
        deep: bool = False,
        skip: list[str] | str | None = None,
        memo: dict | None = None,
    ) -> Self:
        """Create a copy of the AgentContainer.

        Parameters
        ----------
        deep : bool, optional
            Flag indicating whether to perform a deep copy of the AgentContainer.
            If True, all attributes of the AgentContainer will be recursively copied (except self.agents, check Pandas/Polars documentation).
            If False, only the top-level attributes will be copied.
            Defaults to False.

        skip : list[str] | str | None, optional
            A list of attribute names or a single attribute name to skip during the copy process.
            If an attribute name is specified, it will be skipped for all levels of the copy.
            If a list of attribute names is specified, they will be skipped for all levels of the copy.
            If None, no attributes will be skipped.
            Defaults to None.

        memo : dict | None, optional
            A dictionary used to track already copied objects during deep copy.
            Defaults to None.

        Returns
        -------
        Self
            A new instance of the AgentContainer class that is a copy of the original instance.
        """
        skip_list = self._skip_copy.copy()
        cls = self.__class__
        obj = cls.__new__(cls, self.model)
        if isinstance(skip, str):
            skip_list.append(skip)
        elif isinstance(skip, list):
            skip_list += skip
        if deep:
            if not memo:
                memo = {}
            memo[id(self)] = obj
            attributes = self.__dict__.copy()
            setattr(obj, "model", attributes.pop("model"))
            [
                setattr(obj, k, deepcopy(v, memo))
                for k, v in attributes.items()
                if k not in skip_list
            ]
        else:
            [
                setattr(obj, k, copy(v))
                for k, v in self.__dict__.items()
                if k not in skip_list
            ]
        return obj

    @abstractmethod
    def select(
        self,
        mask: MaskLike | None = None,
        filter_func: Callable[[Self], MaskLike] | None = None,
        n: int | None = None,
        inplace: bool = True,
    ) -> Self:
        """Select agents in the AgentContainer based on the given criteria.

        Parameters
        ----------
        mask : MaskLike | None, optional
            The mask of agents to be selected, by default None
        filter_func : Callable[[Self], MaskLike] | None, optional
            A function which takes as input the AgentContainer and returns a MaskLike, by default None
        n : int, optional
            The maximum number of agents to be selected, by default None
        inplace : bool, optional
            If the operation should be performed on the same object, by default True

        Returns
        -------
        Self
            A new or updated AgentContainer.
        """

    @abstractmethod
    def shuffle(self, inplace: bool = True) -> Self:
        """
        Shuffles the order of agents in the AgentContainer.

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
    def sort(self, *args, inplace: bool = True, **kwargs) -> Self:
        """
        Sorts the agents in the agent set based on the given criteria.

        Parameters
        ----------
        *args
            Positional arguments to pass to the sort method.
        inplace : bool
            Whether to sort the agents in place.
        **kwargs
            Keyword arguments to pass to the sort

        Returns
        ----------
        Self
            A new or updated AgentContainer.
        """

    @overload
    def do(
        self,
        method_name: str,
        *args,
        return_results: Literal[False] = False,
        inplace: bool = True,
        **kwargs,
    ) -> Self: ...

    @overload
    def do(
        self,
        method_name: str,
        *args,
        return_results: Literal[True],
        inplace: bool = True,
        **kwargs,
    ) -> Any: ...

    def do(
        self,
        method_name: str,
        *args,
        return_results: bool = False,
        inplace: bool = True,
        **kwargs,
    ) -> Self | Any:
        """Invoke a method on the AgentContainer.

        Parameters
        ----------
        method_name : str
            The name of the method to invoke.
        return_results : bool, optional
            Whether to return the result of the method, by default False
        inplace : bool, optional
            Whether the operation should be done inplace, by default True

        Returns
        -------
        Self | Any
            The updated AgentContainer or the result of the method.
        """
        obj = self._get_obj(inplace)
        method = getattr(obj, method_name)
        if return_results:
            return method(*args, **kwargs)
        else:
            method(*args, **kwargs)
            return obj

    @abstractmethod
    @overload
    def get_attribute(
        self,
        attr_names: list[str] | None = None,
        mask: MaskLike | None = None,
    ) -> DataFrame: ...

    @abstractmethod
    @overload
    def get_attribute(
        self,
        attr_names: str,
        mask: MaskLike | None = None,
    ) -> Series: ...

    @abstractmethod
    def get_attribute(
        self,
        attr_names: str | list[str] | None = None,
        mask: MaskLike | None = None,
    ) -> Series | DataFrame:
        """
        Retrieves the value of a specified attribute for each agent in the AgentContainer.

        Parameters
        ----------
        attr_names : str | list[str] | None
            The name of the attribute to retrieve. If None, all attributes are retrieved. Defaults to None.
        mask : MaskLike | None
            The mask of agents to retrieve the attribute for. If None, attributes of all agents are returned. Defaults to None.

        Returns
        ----------
        Series | DataFrame
            The attribute values.
        """

    @abstractmethod
    @overload
    def set_attribute(
        self,
        attr_names: None = None,
        value: Any = Any,
        mask: MaskLike = MaskLike,
        inplace: bool = True,
    ) -> Self: ...

    @abstractmethod
    @overload
    def set_attribute(
        self,
        attr_names: dict[str, Any],
        value: None,
        mask: MaskLike | None = None,
        inplace: bool = True,
    ) -> Self: ...

    @abstractmethod
    @overload
    def set_attribute(
        self,
        attr_names: str | list[str],
        value: Any,
        mask: MaskLike | None = None,
        inplace: bool = True,
    ) -> Self: ...

    @abstractmethod
    def set_attribute(
        self,
        attr_names: str | dict[str, Any] | list[str] | None = None,
        value: Any | None = None,
        mask: MaskLike | None = None,
        inplace: bool = True,
    ) -> Self:
        """
        Sets the value of a specified attribute or attributes for each agent in the AgentContainer.

        The key can be:
        - A string: sets the specified column of the agents in the AgentContainer.
        - A list of strings: sets the specified columns of the agents in the AgentContainer.
        - A dictionary: keys should be attributes and values should be the values to set. Value should be None.

        Parameters
        ----------
        attr_names : str | dict[str, Any]
            The name of the attribute to set.
        value : Any | None
            The value to set the attribute to. If None, attr_names must be a dictionary.
        mask : MaskLike | None
            The mask of agents to set the attribute for.
        inplace : bool
            Whether to set the attribute in place.

        Returns
        ----------
        AgentContainer
            The updated agent set.
        """

    @abstractmethod
    def add(
        self, other: Self | DataFrame | ListLike | dict[str, Any], inplace: bool = True
    ) -> Self:
        """Adds agents to the AgentContainer.

        Other can be:
        - A Self: adds the agents from the other AgentContainer.
        - A DataFrame: adds the agents from the DataFrame.
        - A ListLike: should be one single agent to add.
        - A dictionary: keys should be attributes and values should be the values to add.

        Parameters
        ----------
        other : Self | DataFrame | ListLike | dict[str, Any]
            The agents to add.
        inplace : bool, optional
            Whether the operation is done into place, by default True

        Returns
        -------
        Self
            The updated AgentContainer.
        """

    def discard(self, id: MaskLike, inplace: bool = True) -> Self:
        """
        Removes an agent from the AgentContainer. Does not raise an error if the agent is not found.

        Parameters
        ----------
        id : ListLike | Any
            The ID of the agent to remove.
        inplace : bool
            Whether to remove the agent in place.

        Returns
        ----------
        AgentContainer
            The updated AgentContainer.
        """
        with suppress(KeyError):
            return self.remove(id, inplace=inplace)

    @abstractmethod
    def remove(self, id: MaskLike, inplace: bool = True) -> Self:
        """
        Removes an agent from the AgentContainer.

        Parameters
        ----------
        id : ListLike | Any
            The ID of the agent to remove.
        inplace : bool
            Whether to remove the agent in place.

        Returns
        ----------
        AgentContainer
            The updated AgentContainer.
        """


### The AgentSetDF class is a container for agents of the same type. It has an implementation with Pandas and Polars ###


class AgentSetDF(AgentContainer):
    _agents: DataFrame
    _skip_copy = ["model", "_mask", "_agents"]
    """A container for agents of the same type.
    
    Attributes
    ----------
    model : ModelDF
        The model to which the AgentSetDF belongs.
    _mask : Series
        A boolean mask indicating which agents are active.
    _agents : DataFrame
        The agents in the AgentSetDF.
    _skip_copy : list[str]
        A list of attributes to skip during the copy process.
    """

    @property
    def agents(self) -> DataFrame:
        """The agents in the AgentSetDF."""
        return self._agents

    @agents.setter
    def agents_setter(self, agents: DataFrame) -> None:
        """Set the agents in the AgentSetDF.

        Parameters
        ----------
        agents : DataFrame
            The agents to set.
        """
        self._agents = agents

    def __len__(self) -> int:
        return len(self._agents)

    def __repr__(self) -> str:
        return repr(self._agents)

    def __str__(self) -> str:
        return str(self._agents)

    def contains(self, ids: MaskLike) -> BoolSeries | bool:

        if isinstance(
            ids,
            (ListLike, Series, pl.Expr, DataFrame),
        ) or ids == "all" or ids == "active":
            return self._get_bool_mask(ids)
        else:
            return ids in self

    @abstractmethod
    def _get_bool_mask(self, mask: MaskLike) -> BoolSeries:
        """Get a boolean mask for the agents in the AgentSet.

        The mask can be:
        - "all": all agents are selected.
        - "active": only active agents are selected.
        - A ListLike of IDs: only agents with the specified IDs are selected.
        - A DataFrame: only agents with indices in the DataFrame are selected.
        - A BoolSeries: only agents with True values are selected.
        - Any other value: only the agent with the specified ID value is selected.

        Parameters
        ----------
        mask : MaskLike
            The mask to apply.

        Returns
        -------
        BoolSeries
            The boolean mask for the agents.
        """


class AgentSetPandas(AgentSetDF):
    _agents: pd.DataFrame
    _mask: pd.Series[bool]
    """A pandas-based implementation of the AgentSet.
    
    Attributes
    ----------
    model : ModelDF
        The model to which the AgentSet belongs.
    _mask : pd.Series[bool]
        A boolean mask indicating which agents are active.
    _agents : pd.DataFrame
        The agents in the AgentSet.
    _skip_copy : list[str]
        A list of attributes to skip during the copy process.
    """

    def __new__(cls, model: ModelDF) -> Self:
        obj = super().__new__(cls, model)
        obj._agents = pd.DataFrame(columns=["unique_id"]).set_index("unique_id")
        obj._mask = pd.Series(True, index=obj._agents.index)
        return obj

    def __contains__(self, id: Hashable) -> bool:
        return id in self._agents.index

    def __deepcopy__(self, memo: dict) -> Self:
        obj = super().__deepcopy__(memo)
        obj._agents = self._agents.copy(deep=True)
        return obj

    def __iter__(self):
        return self._agents.iterrows()

    def __reversed__(self) -> Iterable:
        return self._agents[::-1].iterrows()

    @property
    def agents(self) -> pd.DataFrame:
        return self._agents

    @property
    def active_agents(self) -> pd.DataFrame:
        return self._agents.loc[self._mask]

    @active_agents.setter  # When a property is overriden, so it is the getter
    def active_agents(self, mask: PandasMaskLike) -> None:
        return AgentContainer.active_agents.fset(self, mask)  # type: ignore

    @property
    def inactive_agents(self) -> pd.DataFrame:
        return self._agents.loc[~self._mask]

    def _get_bool_mask(
        self,
        mask: PandasMaskLike | None = None,
    ) -> pd.Series:
        if isinstance(mask, pd.Series) and mask.dtype == bool:
            return mask
        elif isinstance(mask, self.__class__):
            return pd.Series(
                self._agents.index.isin(mask.agents.index), index=self._agents.index
            )
        elif isinstance(mask, pd.DataFrame):
            return pd.Series(
                self._agents.index.isin(mask.index), index=self._agents.index
            )
        elif isinstance(mask, list):
            return pd.Series(self._agents.index.isin(mask), index=self._agents.index)
        elif mask is None or mask == "all":
            return pd.Series(True, index=self._agents.index)
        elif mask == "active":
            return self._mask
        else:
            return pd.Series(self._agents.index.isin([mask]), index=self._agents.index)

    def copy(
        self,
        deep: bool = False,
        skip: list[str] | str | None = None,
        memo: dict | None = None,
    ) -> Self:
        obj = super().copy(deep, skip, memo)
        obj._agents = self._agents.copy(deep=deep)
        obj._mask = self._mask.copy(deep=deep)
        return obj

    def select(
        self,
        mask: PandasMaskLike | None = None,
        filter_func: Callable[[Self], PandasMaskLike] | None = None,
        n: int | None = None,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        bool_mask = obj._get_bool_mask(mask)
        if n != None:
            bool_mask = pd.Series(
                obj._agents.index.isin(obj._agents[bool_mask].sample(n).index),
                index=obj._agents.index,
            )
        if filter_func:
            bool_mask = bool_mask & obj._get_bool_mask(filter_func(obj))
        obj._mask = bool_mask
        return obj

    def shuffle(self, inplace: bool = True) -> Self:
        obj = self._get_obj(inplace)
        obj._agents = obj._agents.sample(frac=1)
        return obj

    def sort(
        self,
        by: str | Sequence[str],
        key: ValueKeyFunc | None = None,
        ascending: bool | Sequence[bool] = True,
        inplace: bool = True,
    ) -> Self:
        """
        Sort the agents in the agent set based on the given criteria.

        Parameters
        ----------
        by : str | Sequence[str]
            The attribute(s) to sort by.
        key : ValueKeyFunc | None
            A function to use for sorting.
        ascending : bool | Sequence[bool]
            Whether to sort in ascending order.

        Returns
        ----------
        AgentSetDF: The sorted agent set.
        """
        obj = self._get_obj(inplace)
        obj._agents.sort_values(by=by, key=key, ascending=ascending, inplace=True)
        return obj

    @overload
    def set_attribute(
        self,
        attr_names: None = None,
        value: Any = Any,
        mask: PandasMaskLike = PandasMaskLike,
        inplace: bool = True,
    ) -> Self: ...

    @overload
    def set_attribute(
        self,
        attr_names: dict[str, Any],
        value: None,
        mask: PandasMaskLike | None = None,
        inplace: bool = True,
    ) -> Self: ...

    @overload
    def set_attribute(
        self,
        attr_names: str | list[str],
        value: Any,
        mask: PandasMaskLike | None = None,
        inplace: bool = True,
    ) -> Self: ...

    def set_attribute(
        self,
        attr_names: str | list[str] | dict[str, Any] | None = None,
        value: Any | None = None,
        mask: PandasMaskLike | None = None,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        mask = obj._get_bool_mask(mask)
        if attr_names is None:
            attr_names = obj._agents.columns.values.tolist()
        if isinstance(attr_names, (str, list)) and value is not None:
            obj._agents.loc[mask, attr_names] = value
        elif isinstance(attr_names, dict):
            for key, value in attr_names.items():
                obj._agents.loc[mask, key] = value
        else:
            raise ValueError(
                "attr_names must be a string or a dictionary with columns as keys and values."
            )
        return obj

    @overload
    def get_attribute(
        self,
        attr_names: list[str] | None = None,
        mask: PandasMaskLike | None = None,
    ) -> pd.DataFrame: ...

    @overload
    def get_attribute(
        self,
        attr_names: str,
        mask: PandasMaskLike | None = None,
    ) -> pd.Series: ...

    def get_attribute(
        self,
        attr_names: str | list[str] | None = None,
        mask: PandasMaskLike | None = None,
        inplace: bool = True,
    ) -> pd.Series | pd.DataFrame:
        obj = self._get_obj(inplace)
        mask = obj._get_bool_mask(mask)
        if attr_names is None:
            return obj._agents.loc[mask]
        else:
            return obj._agents.loc[mask, attr_names]

    def add(
        self,
        other: Self | pd.DataFrame | ListLike | dict[str, Any],
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        if isinstance(other, obj.__class__):
            new_agents = other.agents
        elif isinstance(other, pd.DataFrame):
            new_agents = other
            if "unique_id" != other.index.name:
                try:
                    new_agents.set_index("unique_id", inplace=True, drop=True)
                except KeyError:
                    new_agents["unique_id"] = obj.random.random(len(other)) * 10**8
        elif isinstance(other, dict):
            if "unique_id" not in other:
                index = obj.random.random(len(other)) * 10**8
            if not isinstance(other["unique_id"], ListLike):
                index = [other["unique_id"]]
            else:
                index = other["unique_id"]
            new_agents = (
                pd.DataFrame(other, index=pd.Index(index))
                .reset_index(drop=True)
                .set_index("unique_id")
            )
        else:  # ListLike
            if len(other) == len(obj._agents.columns):
                # data missing unique_id
                new_agents = pd.DataFrame([other], columns=obj._agents.columns)
                new_agents["unique_id"] = obj.random.random(1) * 10**8
            elif len(other) == len(obj._agents.columns) + 1:
                new_agents = pd.DataFrame(
                    [other], columns=["unique_id"] + obj._agents.columns.values.tolist()
                )
            else:
                raise ValueError(
                    "Length of data must match the number of columns in the AgentSet if being added as a ListLike."
                )
            new_agents.set_index("unique_id", inplace=True, drop=True)
        obj._agents = pd.concat([obj._agents, new_agents])
        return obj

    def remove(self, id: PandasMaskLike, inplace: bool = True) -> Self:
        initial_len = len(self._agents)
        obj = self._get_obj(inplace)
        mask = obj._get_bool_mask(id)
        remove_ids = obj._agents[mask].index
        obj._agents.drop(remove_ids, inplace=True)
        if len(obj._agents) == initial_len:
            raise KeyError(f"IDs {id} not found in agent set.")
        return obj


class AgentSetPolars(AgentSetDF):
    _agents: pl.DataFrame
    _mask: pl.Expr | pl.Series
    """A polars-based implementation of the AgentSet."""

    def __init__(self, model: ModelDF):
        """Create a new AgentSetDF.

        Parameters
        ----------
        model : ModelDF
            The model to which the AgentSetDF belongs.

        Attributes
        ----------
        agents : pl.DataFrame
            The agents in the AgentSetDF.
        model : ModelDF
            The model to which the AgentSetDF belongs.
        """
        super().__init__(model)
        self._agents = pl.DataFrame(schema={"unique_id": pl.String})
        self._mask = pl.repeat(True, len(self.agents))

    @property
    def agents(self) -> pl.DataFrame:
        if self._agents is None:
            self._agents = pl.DataFrame(schema={"unique_id": pl.String})
        return self._agents

    @property
    def active_agents(self) -> pl.DataFrame:
        return self.agents.filter(self._mask)

    @property
    def inactive_agents(self) -> pl.DataFrame:
        return self.agents.filter(~self._mask)

    def select(
        self,
        mask: pl.Expr | pl.Series | pl.DataFrame | None = None,
        filter_func: Callable[[Self], pl.Series] | None = None,
        n: int = 0,
    ) -> Self:
        if mask is None:  # if not mask doesn't work
            mask = pl.repeat(True, len(self.agents))
        elif isinstance(mask, pl.DataFrame):
            mask = self.agents["unique_id"].is_in(mask["unique_id"])
        if filter_func:
            mask = mask & filter_func(self)
        if n != 0:
            mask = (
                self.agents.filter(mask)
                .sample(n)["unique_id"]
                .is_in(self.agents["unique_id"])
            )
        self._mask = mask
        return self

    def shuffle(self) -> Self:
        self.agents = self.agents.sample(fraction=1)
        return self

    def sort(
        self,
        by: IntoExpr | Iterable[IntoExpr],
        *more_by: IntoExpr,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool = False,
    ) -> Self:
        """Sort the agents in the agent set based on the given criteria.

        Parameters
        ----------
        by (IntoExpr | Iterable[IntoExpr]): The attribute(s) to sort by.
        more_by (IntoExpr): Additional attributes to sort by.
        descending (bool | Sequence[bool]): Whether to sort in descending order.
        nulls_last (bool): Whether to place null values last.

        Returns
        ----------
        AgentSetDF: The sorted agent set.
        """
        self.agents = self.agents.sort(
            by=by, *more_by, descending=descending, nulls_last=nulls_last
        )
        return self

    def get_attribute(self, attr_names: str) -> pl.Series:
        return self.agents.filter(self._mask)[attr_names]

    def set_attribute(self, attr_names: str, value: Any) -> Self:
        if type(value) == pl.Series:
            self.agents.filter(self._mask).with_columns(**{attr_names: value})
        else:
            self.agents.filter(self._mask).with_columns(**{attr_names: pl.lit(value)})
        return self

    def add(
        self,
        n: int,
        data: FrameInitTypes | None = None,
        schema: SchemaDefinition | None = None,
        schema_overrides: SchemaDict | None = None,
        orient: Orientation | None = None,
        infer_schema_length: int | None = N_INFER_DEFAULT,
        nan_to_null: bool = False,
    ) -> Self:
        """Adds new agents to the agent set.

        Parameters
        ----------
        n : int
            The number of agents to add.
        data : dict, Sequence, ndarray, Series, or pandas.DataFrame
            Two-dimensional data in various forms; dict input must contain Sequences, Generators, or a range. Sequence may contain Series or other Sequences.
        schema : Sequence of str, (str,DataType) pairs, or a {str:DataType,} dict
            The DataFrame schema may be declared in several ways:
            - As a dict of {name:type} pairs; if type is None, it will be auto-inferred.
            - As a list of column names; in this case types are automatically inferred.
            - As a list of (name,type) pairs; this is equivalent to the dictionary form.
            If you supply a list of column names that does not match the names in the underlying data, the names given here will overwrite them. The number of names given in the schema should match the underlying data dimensions.
        schema_overrides : dict, default None
            Support type specification or override of one or more columns; note that any dtypes inferred from the schema param will be overridden.
            The number of entries in the schema should match the underlying data dimensions, unless a sequence of dictionaries is being passed, in which case a *partial* schema can be declared to prevent specific fields from being loaded.
        orient : {'col', 'row'}, default None
            Whether to interpret two-dimensional data as columns or as rows.
            If None, the orientation is inferred by matching the columns and data dimensions.
            If this does not yield conclusive results, column orientation is used.
        infer_schema_length : int or None
            The maximum number of rows to scan for schema inference. If set to None, the full data may be scanned *(this is slow)*.
            This parameter only applies if the input data is a sequence or generator of rows; other input is read as-is.
        nan_to_null :  bool, default False
            If the data comes from one or more numpy arrays, can optionally convert input data np.nan values to null instead. This is a no-op for all other input data.

        Returns
        ----------
        AgentSetPolars: The updated agent set.
        """
        new_df = pl.DataFrame(
            data=data,
            schema=schema,
            schema_overrides=schema_overrides,
            orient=orient,
            infer_schema_length=infer_schema_length,
            nan_to_null=nan_to_null,
        )

        if "unique_id" not in new_df.columns:
            new_df = new_df.with_columns(
                unique_id=pl.Series(
                    values=self.random.random(n) * 10**8, dtype=pl.Int64
                )
            )

        old_active_agents = self.agents.filter(self._mask)["unique_id"]
        self.agents = pl.concat([self.agents, new_df])
        self._mask = self.agents["unique_id"].is_in(old_active_agents) | self.agents[
            "unique_id"
        ].is_in(new_df["unique_id"])
        return self

    def discard(self, id: int) -> Self:
        with suppress(KeyError):
            self.agents = self.agents.filter(self.agents["unique_id"] != id)
        return self

    def remove(self, id: int) -> Self:
        self.agents = self.agents.filter(self.agents["unique_id"] != id)
        return self


### The AgentsDF class is a container for AgentSetDFs. It has an implementation with Pandas and Polars ###


class AgentsDF(AgentContainer):
    agentsets: list[AgentSetDF]
    """A collection of AgentSetDFs. All agents of the model are stored here."""

    def __init__(self, model: ModelDF) -> None:
        super().__init__(model)
        self.agentsets = []

    def __len__(self) -> int:
        return sum(len(agentset.agents) for agentset in self.agentsets)

    def __repr__(self):
        return self.agentsets.__repr__()

    def __str__(self) -> str:
        return self.agentsets.__str__()

    def sort(
        self,
        by: str | Sequence[str],
        key: ValueKeyFunc | None,
        ascending: bool | Sequence[bool] = True,
    ) -> Self:
        self.agentsets = [
            agentset.sort(by, key, ascending) for agentset in self.agentsets
        ]
        return self

    @overload
    def do(
        self,
        method_name: str,
        return_results: Literal[False] = False,
        *args,
        **kwargs,
    ) -> Self: ...

    @overload
    def do(
        self,
        method_name: str,
        return_results: Literal[True],
        *args,
        **kwargs,
    ) -> list[Any]: ...

    def do(
        self,
        method_name: str,
        return_results: bool = False,
        *args,
        **kwargs,
    ) -> Self | list[Any]:
        if return_results:
            return [
                agentset.do(method_name, return_results, *args, **kwargs)
                for agentset in self.agentsets
            ]
        else:
            self.agentsets = [
                agentset.do(method_name, return_results, *args, **kwargs)
                for agentset in self.agentsets
            ]
            return self

    def add(self, agentsets: AgentSetDF | list[AgentSetDF]) -> Self:
        if isinstance(agentsets, list):
            self.agentsets += agentsets
        else:
            self.agentsets.append(agentsets)
        return self

    @abstractmethod
    def to_frame(self) -> DataFrame:
        """Convert the AgentsDF to a single DataFrame.

        Returns
        -------
        DataFrame
            A DataFrame containing all agents from all AgentSetDFs.
        """
        pass

    def get_agents_of_type(self, agent_type: type[AgentSetDF]) -> AgentSetDF:
        """Retrieve the AgentSetDF of a specified type.

        Parameters
        ----------
        agent_type : type
            The type of AgentSetDF to retrieve.

        Returns
        -------
        AgentSetDF
            The AgentSetDF of the specified type.
        """
        for agentset in self.agentsets:
            if isinstance(agentset, agent_type):
                return agentset
        raise ValueError(f"No AgentSetDF of type {agent_type} found.")

    def set_attribute(self, attr_names: str, value: Any) -> Self:
        self.agentsets = [
            agentset.set_attribute(attr_names, value) for agentset in self.agentsets
        ]
        return self

    def shuffle(self) -> Self:
        self.agentsets = [agentset.shuffle() for agentset in self.agentsets]
        return self

    def discard(self, id: int) -> Self:
        self.agentsets = [agentset.discard(id) for agentset in self.agentsets]
        return self

    def remove(self, id: int) -> Self:
        for i, agentset in enumerate(self.agentsets):
            original_size = len(agentset.agents)
            self.agentsets[i] = agentset.discard(id)
            if original_size != len(self.agentsets[i].agents):
                return self
        raise KeyError(f"Agent with id {id} not found in any agentset.")


class AgentsPandas(AgentsDF):
    agentsets: list[AgentSetPandas]
    """A pandas implementation of a collection of AgentSetDF. All agents of the model are stored here."""

    def __init__(self, model: ModelDF):
        """Create a new AgentsDF object.

        Parameters
        ----------
        model : ModelDF
            The model to which the AgentsDF object belongs.

        Attributes
        ----------
        agentsets : list[AgentSetDF]
            The AgentSetDFs that make up the AgentsDF object.
        model : ModelDF
            The model to which the AgentSetDF belongs.
        """
        super().__init__(model)

    @property
    def active_agents(self) -> pd.DataFrame:
        return pd.concat([agentset.active_agents for agentset in self.agentsets])

    @property
    def inactive_agents(self) -> pd.DataFrame:
        return pd.concat([agentset.inactive_agents for agentset in self.agentsets])

    def select(
        self,
        mask: pd.Series[bool] | pd.DataFrame | None = None,
        filter_func: Callable[[AgentSetDF], pd.Series[bool]] | None = None,
        n: int = 0,
    ) -> Self:
        n, r = int(n / len(self.agentsets)), n % len(self.agentsets)
        new_agentsets: list[AgentSetPandas] = []
        for agentset in self.agentsets:
            if mask is None:
                agentset_mask = mask
            elif isinstance(mask, pd.DataFrame):
                agentset_mask = pd.Series(
                    agentset.agents.index.isin(mask), index=agentset.agents.index
                )
            else:
                agentset_mask = pd.Series(
                    agentset.agents.index.isin(mask[mask].index),
                    index=agentset.agents.index,
                )
            agentset.select(mask=agentset_mask, filter_func=filter_func, n=n + r)
            if len(agentset.active_agents) > n:
                r = len(agentset.active_agents) - n
            new_agentsets.append(agentset)
        self.agentsets = new_agentsets
        return self

    def get_attribute(self, attr_names: str) -> pd.Series[Any]:
        return pd.concat(
            [agentset.get_attribute(attr_names) for agentset in self.agentsets]
        )

    def add(self, agentsets: AgentSetPandas | list[AgentSetPandas]) -> Self:
        return super().add(agentsets)  # type: ignore


class AgentsPolars(AgentsDF):
    agentsets: list[AgentSetPolars]
    """A polars implementation of a collection of AgentSetDF. All agents of the model are stored here."""

    def __init__(self, model: ModelDF):
        """Create a new AgentsDF object.

        Parameters
        ----------
        model : ModelDF
            The model to which the AgentsDF object belongs.

        Attributes
        ----------
        agentsets : list[AgentSetDF]
            The AgentSetDFs that make up the AgentsDF object.
        model : ModelDF
            The model to which the AgentSetDF belongs.
        """
        super().__init__(model)

    @property
    def active_agents(self) -> pl.DataFrame:
        return pl.concat([agentset.active_agents for agentset in self.agentsets])

    @property
    def inactive_agents(self) -> pl.DataFrame:
        return pl.concat([agentset.inactive_agents for agentset in self.agentsets])

    def select(
        self,
        mask: pl.Expr | pl.Series | pl.DataFrame | None = None,
        filter_func: Callable[[AgentSetDF], pl.Series] | None = None,
        n: int = 0,
    ) -> Self:
        n, r = int(n / len(self.agentsets)), n % len(self.agentsets)
        new_agentsets: list[AgentSetPolars] = []
        for agentset in self.agentsets:
            if mask is None:
                agentset_mask = mask
            elif isinstance(mask, pl.DataFrame):
                agentset_mask = agentset.agents["unique_id"].is_in(mask["unique_id"])
            elif isinstance(mask, pl.Series):
                agentset_mask = agentset.agents["unique_id"].is_in(mask)
            agentset.select(mask=agentset_mask, filter_func=filter_func, n=n + r)
            if len(agentset.active_agents) > n:
                r = len(agentset.active_agents) - n
            new_agentsets.append(agentset)
        self.agentsets = new_agentsets
        return self

    def get_attribute(self, attr_names: str) -> pl.Series:
        return pl.concat(
            [agentset.get_attribute(attr_names) for agentset in self.agentsets]
        )

    def add(self, agentsets: AgentSetPolars | list[AgentSetPolars]) -> Self:
        return super().add(agentsets)  # type: ignore #child classes are not checked?
