from abc import ABC, abstractmethod
from contextlib import suppress
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

from mesa_frames.model import ModelDF

if TYPE_CHECKING:

    # For AgentSetDF
    from numpy.random import Generator

    DataFrameLike = pd.DataFrame | pl.DataFrame
    MaskLike = pd.Series[bool] | pl.Expr | pl.Series

    # For AgentSetPandas
    from numpy import ndarray
    from pandas.core.arrays.base import ExtensionArray

    from .model import ModelDF

    ArrayLike = ExtensionArray | ndarray
    AnyArrayLike = ArrayLike | pd.Index | pd.Series
    ValueKeyFunc = Callable[[pd.Series], pd.Series | AnyArrayLike] | None
    from pandas._typing import Axes, Dtype, ListLikeU
    from polars.datatypes import N_INFER_DEFAULT

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
    _mask: MaskLike
    """An abstract class for containing agents. Defines the common interface for AgentSetDF and AgentsDF."""

    def __init__(self, model: ModelDF) -> None:
        self.model = model
    
    def __get_item__(self, attr_name: str) -> Any:
        return self.get_attribute(attr_name)
    
    def __set_item__(self, attr_name: str, value: Any) -> None:
        self.set_attribute(attr_name, value)

    @property
    @abstractmethod
    def active_agents(self) -> DataFrameLike:
        """The active agents in the AgentContainer (those that are used for the do, set_attribute, get_attribute operations).

        Returns
        -------
        DataFrameLike
        """
        pass
    
    @active_agents.setter
    def active_agents(self, agents: DataFrameLike | MaskLike) -> None:
        self.select(mask=agents)

    @property
    @abstractmethod
    def inactive_agents(self) -> DataFrameLike:
        """The inactive agents in the AgentContainer (those that are not used for the do, set_attribute, get_attribute operations).

        Returns
        -------
        DataFrameLike
        """
        pass

    @inactive_agents.setter
    def inactive_agents(self, agents: DataFrameLike | MaskLike) -> None:
        self.select(mask=agents)
        
    @property
    def random(self) -> Generator:
        """
        Provide access to the model's random number generator.

        Returns:
        ----------
        np.Generator
        """
        return self.model.random

    @abstractmethod
    def select(
        self,
        mask: MaskLike | DataFrameLike | None = None,
        filter_func: Callable[[Self], MaskLike] | None = None,
        n: int = 0,
    ) -> Self:
        """
        Selects a subset of agents based on the given criteria.

        Parameters:
        ----------
            mask (MaskLike | DataFrameLike | None): A boolean mask or DataFrame used to filter the agents.
            filter_func (Callable[[AgentContainer], MaskLike] | None): A function that takes an AgentContainer and returns a boolean mask.
            n (int): The maximum number of agents to select.

        Returns:
        ----------
            AgentContainer: Returns an AgentContainer with selected agents as active.

        """
        pass

    @abstractmethod
    def shuffle(self) -> Self:
        """
        Shuffles the order of agents in the AgentContainer.

        Returns:
        ----------
            AgentContainer: The shuffled agent set.
        """
        pass

    @abstractmethod
    def sort(self, *args, **kwargs) -> Self:
        """
        Sorts the agents in the agent set based on the given criteria.
        """
        pass

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
    ) -> Any: ...

    def do(
        self,
        method_name: str,
        return_results: bool = False,
        *args,
        **kwargs,
    ) -> Self | Any:
        """
        Invokes a method on the AgentContainer.

        Parameters:
        ----------
            method_name (str): The name of the method to call.
            return_results (bool): Whether to return the results of the method call.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
        ----------
            AgentContainer | Any: The updated agent set or the results of the method call.
        """
        method = getattr(self, method_name)
        if return_results:
            return method(*args, **kwargs)
        else:
            method(*args, **kwargs)
            return self

    @abstractmethod
    def get_attribute(self, attr_name: str) -> MaskLike:
        """
        Retrieves the value of a specified attribute for each agent in the AgentContainer.

        Parameters:
        ----------
            attr_name (str): The name of the attribute to retrieve.

        Returns:
        ----------
            MaskLike: The attribute values.
        """
        pass

    @abstractmethod
    def set_attribute(self, attr_name: str, value: Any) -> Self:
        """
        Sets the value of a specified attribute for each agent in the AgentContainer.

        Parameters:
        ----------
            attr_name (str): The name of the attribute to set.
            value (Any): The value to set the attribute to.

        Returns:
        ----------
            AgentContainer: The updated agent set.
        """
        pass

    @abstractmethod
    def add(self, n: int, *args, **kwargs) -> Self:
        """Adds new agents to the AgentContainer."""
        pass

    @abstractmethod
    def discard(self, id: int) -> Self:
        """
        Removes an agent from the AgentContainer.

        Parameters:
        ----------
            id (int): The ID of the agent to remove.

        Returns:
        ----------
            AgentContainer: The updated AgentContainer.
        """
        pass

    @abstractmethod
    def remove(self, id: int) -> Self:
        """
        Removes an agent from the AgentContainer.

        Parameters:
        ----------
            id (int): The ID of the agent to remove.

        Returns:
        ----------
            AgentContainer: The updated AgentContainer.
        """
        pass


### The AgentSetDF class is a container for agents of the same type. It has an implementation with Pandas and Polars ###


class AgentSetDF(AgentContainer):
    agents: DataFrameLike
    """An abstract class for a set of agents of the same type."""


class AgentSetPandas(AgentSetDF):
    agents: pd.DataFrame
    _mask: pd.Series[bool]
    """A pandas-based implementation of the AgentSet."""

    def __init__(self, model: ModelDF):
        """Create a new AgentSetDF.

        Parameters
        ----------
        model : ModelDF
            The model to which the AgentSetDF belongs.

        Attributes
        ----------
        agents : pd.DataFrame
            The agents in the AgentSetDF.
        model : ModelDF
            The model to which the AgentSetDF belongs.
        """
        super().__init__(model)

    @property
    def active_agents(self) -> pd.DataFrame:
        return self.agents.loc[self._mask]

    @property
    def inactive_agents(self) -> pd.DataFrame:
        return self.agents.loc[~self._mask]

    def select(
        self,
        mask: pd.Series[bool] | pd.DataFrame | None = None,
        filter_func: Callable[[Self], pd.Series[bool]] | None = None,
        n: int = 0,
    ) -> Self:
        if mask is None:
            mask = pd.Series(True, index=self.agents.index)
        elif isinstance(mask, pd.DataFrame):
            mask = pd.Series(
                self.agents.index.isin(mask.index), index=self.agents.index
            )
        if filter_func:
            mask = mask & filter_func(self)
        if n != 0:
            mask = pd.Series(self.agents[mask].sample(n).index.isin(self.agents.index))
        self._mask = mask
        return self

    def shuffle(self) -> Self:
        self.agents = self.agents.sample(frac=1)
        return self

    def sort(
        self,
        by: str | Sequence[str],
        key: ValueKeyFunc | None,
        ascending: bool | Sequence[bool] = True,
    ) -> Self:
        """
        Sort the agents in the agent set based on the given criteria.

        Parameters:
        ----------
            by : str | Sequence[str]
                The attribute(s) to sort by.
            key : ValueKeyFunc | None
                A function to use for sorting.
            ascending : bool | Sequence[bool]
                Whether to sort in ascending order.

        Returns:
        ----------
            AgentSetDF: The sorted agent set.
        """
        self.agents.sort_values(by=by, key=key, ascending=ascending, inplace=True)
        return self

    def get_attribute(self, attr_name: str) -> pd.Series[Any]:
        return self.agents.loc[
            self.agents.index.isin(self.active_agents.index), attr_name
        ]

    def set_attribute(self, attr_name: str, value: Any) -> Self:
        self.agents.loc[self.agents.index.isin(self.active_agents.index), attr_name] = (
            value
        )
        return self

    def add(
        self,
        n: int,
        data: (
            ListLikeU
            | pd.DataFrame
            | dict[Any, Any]
            | Iterable[ListLikeU | tuple[Hashable, ListLikeU] | dict[Any, Any]]
            | None
        ) = None,
        index: Axes | None = None,
        copy: bool = False,
        columns: Axes | None = None,
        dtype: Dtype | None = None,
    ) -> Self:
        """
        Adds new agents to the agent set.

        Parameters:
        ----------
            n (int): The number of agents to add.
            data (ListLikeU | pd.DataFrame | dict[Any, Any] | Iterable[ListLikeU | tuple[Hashable, ListLikeU] | dict[Any, Any]] | None): The data for the new agents.
            index (Axes | None): The index for the new agents.
            copy (bool): Whether to copy the data.
            columns (Axes | None): The columns for the new agents.
            dtype (Dtype | None): The data type for the new agents.

        Returns:
        ----------
            AgentSetDF: The updated agent set.
        """
        if not index:
            index = pd.Index((self.random.random(n) * 10**8).astype(int))

        new_df = pd.DataFrame(
            data=data,
            index=index,
            columns=columns,
            dtype=dtype,
            copy=copy,
        )

        self.agents = pd.concat([self.agents, new_df])

        if self._mask.empty:
            self._mask = pd.Series(True, index=new_df.index)
        else:
            self._mask = pd.concat([self._mask, pd.Series(True, index=new_df.index)])

        return self

    def discard(self, id: int) -> Self:
        with suppress(KeyError):
            self.agents.drop(id, inplace=True)
        return self

    def remove(self, id: int) -> Self:
        self.agents.drop(id, inplace=True)
        return self


class AgentSetPolars(AgentSetDF):
    agents: pl.DataFrame
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
        self.agents = pl.DataFrame(schema={"unique_id": pl.String})
        self._mask = pl.repeat(True, len(self.agents))

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

        Parameters:
        ----------
        by (IntoExpr | Iterable[IntoExpr]): The attribute(s) to sort by.
        more_by (IntoExpr): Additional attributes to sort by.
        descending (bool | Sequence[bool]): Whether to sort in descending order.
        nulls_last (bool): Whether to place null values last.

        Returns:
        ----------
        AgentSetDF: The sorted agent set.
        """
        self.agents = self.agents.sort(
            by=by, *more_by, descending=descending, nulls_last=nulls_last
        )
        return self

    def get_attribute(self, attr_name: str) -> pl.Series:
        return self.agents.filter(self._mask)[attr_name]

    def set_attribute(self, attr_name: str, value: Any) -> Self:
        if type(value) == pl.Series:
            self.agents.filter(self._mask).with_columns(**{attr_name: value})
        else:
            self.agents.filter(self._mask).with_columns(**{attr_name: pl.lit(value)})
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

        Parameters:
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

        Returns:
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
    def to_frame(self) -> DataFrameLike:
        """Convert the AgentsDF to a single DataFrame.

        Returns
        -------
        DataFrameLike
            A DataFrame containing all agents from all AgentSetDFs.
        """
        pass

    def get_agents_of_type(self, agent_type: type) -> AgentSetDF:
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

    def set_attribute(self, attr_name: str, value: Any) -> Self:
        self.agentsets = [
            agentset.set_attribute(attr_name, value) for agentset in self.agentsets
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

    def get_attribute(self, attr_name: str) -> pd.Series[Any]:
        return pd.concat(
            [agentset.get_attribute(attr_name) for agentset in self.agentsets]
        )

    def add(self, agentsets: AgentSetPandas | list[AgentSetPandas]) -> Self:
        if isinstance(agentsets, list) and not all(
            isinstance(agentset, AgentSetPandas) for agentset in agentsets
        ):
            raise ValueError("All agentsets must be of type AgentSetPandas.")
        elif not isinstance(agentsets, AgentSetPandas):
            raise ValueError(
                "agentsets must be of type AgentSetPandas or list[AgentSetPandas]."
            )
        return super().add(agentsets)


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

    def get_attribute(self, attr_name: str) -> pl.Series:
        return pl.concat(
            [agentset.get_attribute(attr_name) for agentset in self.agentsets]
        )

    def add(self, agentsets: AgentSetPolars | list[AgentSetPolars]) -> Self:
        if isinstance(agentsets, list) and not all(
            isinstance(agentset, AgentSetPolars) for agentset in agentsets
        ):
            raise ValueError("All agentsets must be of type AgentSetPolars.")
        elif not isinstance(agentsets, AgentSetPolars):
            raise ValueError(
                "agentsets must be of type AgentSetPolars or list[AgentSetPolars]."
            )
        return super().add(agentsets)
