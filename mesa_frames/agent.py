from __future__ import annotations

from contextlib import suppress
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterable,
    Literal,
    Sequence,
    overload,
)

import pandas as pd
from numpy import ndarray

# import polars as pl
from numpy.random import Generator

if TYPE_CHECKING:
    from pandas.core.arrays.base import ExtensionArray

    from .model import ModelDF

    ArrayLike = ExtensionArray | ndarray
    AnyArrayLike = ArrayLike | pd.Index | pd.Series
    ValueKeyFunc = Callable[[pd.Series], pd.Series | AnyArrayLike] | None

    from pandas._typing import Axes, Dtype, ListLikeU


class AgentsDF:
    """A collection of AgentSetDFs. All agents of the model are stored here."""

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
        self.agentsets: list[AgentSetDF] = []
        self.model: ModelDF = model

    @property
    def active_agents(self) -> pd.DataFrame:
        """The active agents in the AgentsDF (those that are used for the do, set_attribute, get_attribute operations).

        Returns
        -------
        pd.DataFrame
        """
        return pd.concat([agentset.active_agents for agentset in self.agentsets])

    @property
    def inactive_agents(self) -> pd.DataFrame:
        """The inactive agents in the AgentsDF (those that are not used for the do, set_attribute, get_attribute operations).

        Returns
        -------
        pd.DataFrame
        """
        return pd.concat([agentset.inactive_agents for agentset in self.agentsets])

    @property
    def random(self) -> Generator | None:
        """
        Provide access to the model's random number generator.

        Returns:
        ----------
        np.Generator
        """
        return self.agentsets[0].model.random

    def select(
        self,
        mask: pd.Series[bool] | pd.DataFrame | None = None,
        filter_func: Callable[[AgentSetDF], pd.Series[bool]] | None = None,
        n: int = 0,
    ) -> AgentsDF:
        """
        Change active_agents to a subset of agents from the AgentSet based on a mask, filter function and/or quantity limit.

        Attributes:
        ----------
        mask : pd.Series[bool] | pd.DataFrame | None, optional
            A boolean mask indicating which agents should be included in the result.
            If it's a DataFrame, it uses the indexes present in that dataframe.
            If None, no filtering is applied. Defaults to None.
        filter_func : Callable[[AgentSetDF],  pd.Series[bool]], optional
            A function that takes the AgentSet and returns a boolean mask over the agents indicating which agents
            should be included in the result. Defaults to None, meaning no filtering is applied.
        n : int, optional
            The number of agents to select. If 0, all matching agents are selected. Defaults to 0.

        Returns:
        ----------
        AgentsDF
            The same AgentsDF with the updated active_agents property for each AgentSetDF.
        """
        n, r = int(n / len(self.agentsets)), n % len(self.agentsets)
        new_agentsets: list[AgentSetDF] = []
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

    def shuffle(self) -> AgentsDF:
        """Randomly shuffles the agents in each AgentSetDF.

        Returns:
        ----------
        AgentsDF
            The same AgentsDF with the agents shuffled in each AgentSetDF.
        """
        self.agentsets = [agentset.shuffle() for agentset in self.agentsets]
        return self

    def sort(
        self,
        by: str | Sequence[str],
        key: ValueKeyFunc | None,
        ascending: bool | Sequence[bool] = True,
    ) -> AgentsDF:
        """
        Sort the agents in each AgentSetDF based on a specified attribute or custom function.

        Parameters:
        ----------
        by : str | Sequence[str])
            A single attribute name or a list of attribute names based on which the agents are sorted.
        key : ValueKeyFunc | None
            A function or attribute name based on which the agents are sorted.
        ascending : bool, optional
            If True, the agents are sorted in ascending order. Defaults to True.

        Returns:
        ----------
        AgentsDF
            The same AgentsDF with the agents sorted in each AgentSetDF.
        """
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
    ) -> AgentsDF: ...

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
    ) -> AgentsDF | list[Any]:
        """Invoke a method on each AgentSetDF.


        Parameters
        ----------
        method_name : str
            The name of the method to call on each agent.
        return_results : bool, optional
            If True, the results of the method calls are returned. Defaults to False.
        *args
            Variable length argument list passed to the method being called.
        **kwargs
            Arbitrary keyword arguments passed to the method being called.

        Returns
        -------
        AgentsDF | list[Any]
            The same AgentsDF with each AgentSetDF updated based on the method call or the results of the method calls.

        """
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

    def get_attribute(self, attr_name: str) -> pd.Series[Any]:
        """
        Retrieve a specified attribute for active agents in AgentsDF.

        Parameters:
        ----------
        attr_name : str
            The name of the attribute to retrieve.

        Returns:
        ----------
        pd.Series[Any]
            A list of attribute values from each active agent in AgentsDF.
        """
        return pd.concat(
            [agentset.get_attribute(attr_name) for agentset in self.agentsets]
        )

    def set_attribute(self, attr_name: str, value: Any) -> AgentsDF:
        """
        Set a specified attribute for active agents in AgentsDF.

        Parameters:
        ----------
        attr_name : str
            The name of the attribute to set for each agent.
        value : Any
            The value assigned to the attribute. If the value is a scalar, it is assigned to all active agents.
            If the value is array-like, it must be the same length as the number of active agents.

        Returns:
        ----------
        AgentsDF
            The updated Agents
        """
        self.agentsets = [
            agentset.set_attribute(attr_name, value) for agentset in self.agentsets
        ]
        return self

    def add(self, agentsets: AgentSetDF | list[AgentSetDF]) -> AgentsDF:
        """Add an AgentSetDF or a list of AgentSetDFs to the AgentsDF.

        Parameters
        ----------
        agentsets : AgentSetDF | list[AgentSetDF]

        Returns
        -------
        AgentsDF
            The updated AgentsDF.
        """
        if isinstance(agentsets, AgentSetDF):
            self.agentsets.append(agentsets)
        else:
            self.agentsets = self.agentsets + agentsets
        return self

    def discard(self, id: int) -> AgentsDF:
        """Remove a specified agent. If the agent is not found, does not raise an error.

        Parameters
        ----------
        id : int
            The ID of the agent to remove.

        Returns
        ----------
        Agents
            The updated Agents."""
        self.agentsets = [agentset.discard(id) for agentset in self.agentsets]
        return self

    def remove(self, id: int) -> AgentsDF:
        """Remove an agent from the AgentsDF. If the agent is not found, raises a KeyError.

        Parameters
        ----------
        id : int
            The ID of the agent to remove.

        Returns
        ----------
        AgentsDF
            The updated AgentsDF.
        """
        for i, agentset in enumerate(self.agentsets):
            original_size = len(agentset.agents)
            self.agentsets[i] = agentset.discard(id)
            if original_size != len(self.agentsets[i].agents):
                return self
        raise KeyError(f"Agent with id {id} not found in any agentset.")

    def to_frame(self) -> pd.DataFrame:
        """Convert the AgentsDF to a single DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing all agents from all AgentSetDFs.
        """
        return pd.concat([agentset.agents for agentset in self.agentsets])

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


class AgentSetDF:
    """A DataFrame-based implementation of the AgentSet."""

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
        self.agents: pd.DataFrame = pd.DataFrame()
        self.model: ModelDF = model
        self._mask: pd.Series[bool] = pd.Series(True, index=self.agents.index)

    @property
    def active_agents(self) -> pd.DataFrame:
        """The active agents in the AgentSetDF (those that are used for the do, set_attribute, get_attribute operations).

        Returns
        -------
        pd.DataFrame
        """
        return self.agents.loc[self._mask]

    @property
    def inactive_agents(self) -> pd.DataFrame:
        """The inactive agents in the AgentSetDF (those that are not used for the do, set_attribute, get_attribute operations).

        Returns
        -------
        pd.DataFrame
        """
        return self.agents.loc[~self._mask]

    @property
    def random(self) -> Generator:
        """
        Provide access to the model's random number generator.

        Returns:
        ----------
        np.Generator
        """
        return self.model.random

    def select(
        self,
        mask: pd.Series[bool] | pd.DataFrame | None = None,
        filter_func: Callable[[AgentSetDF], pd.Series[bool]] | None = None,
        n: int = 0,
    ) -> AgentSetDF:
        """
        Change active_agents to a subset of agents from the AgentSetDF based on a mask, filter function and/or quantity limit.

        Attributes:
        ----------
        mask : pd.Series[bool] | pd.DataFrame | None, optional
            A boolean mask indicating which agents should be included in the result.
            If it's a DataFrame, it uses the indexes present in that dataframe.
            If None, no filtering is applied. Defaults to None.
        filter_func : Callable[[AgentSetDF],  pd.Series[bool]], optional
            A function that takes the AgentSet and returns a boolean mask over the agents indicating which agents
            should be included in the result. Defaults to None, meaning no filtering is applied.
        n : int, optional
            The number of agents to select. If 0, all matching agents are selected. Defaults to 0.

        Returns:
        ----------
        AgentSetDF
            The same AgentSetDF with the updated active_agents property.
        """
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

    def shuffle(self) -> AgentSetDF:
        """Randomly shuffles the agents in the AgentSetDF.

        Returns:
        ----------
        AgentSetDF
            The same AgentSetDF with the agents shuffled.
        """
        self.agents = self.agents.sample(frac=1)
        return self

    def sort(
        self,
        by: str | Sequence[str],
        key: ValueKeyFunc | None,
        ascending: bool | Sequence[bool] = True,
    ) -> AgentSetDF:
        """
        Sort the agents in the AgentSetDF based on a specified attribute or custom function.

        Parameters:
        ----------
        by : str | Sequence[str])
            A single attribute name or a list of attribute names based on which the agents are sorted.
        key : ValueKeyFunc | None
            A function or attribute name based on which the agents are sorted.
        ascending : bool, optional
            If True, the agents are sorted in ascending order. Defaults to True.

        Returns:
        ----------
        AgentSetDF
            The same AgentSetDF with the agents sorted.
        """
        self.agents.sort_values(by=by, key=key, ascending=ascending, inplace=True)
        return self

    @overload
    def do(
        self,
        method_name: str,
        return_results: Literal[False] = False,
        *args,
        **kwargs,
    ) -> AgentSetDF: ...

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
    ) -> AgentSetDF | Any:
        """
        Invoke a method on the AgentSetDF.

        Parameters:
        ----------
        method_name : str
            The name of the method to call on each agent.
        return_results : bool, optional
            If True, the results of the method calls are returned. Defaults to False.
        *args
            Variable length argument list passed to the method being called.
        **kwargs
            Arbitrary keyword arguments passed to the method being called.

        Returns:
        ----------
        AgentSetDF | Any
            The same AgentSetDF with the agents updated based on the method call or the results of the method calls.
        """
        method = getattr(self, method_name)
        if return_results:
            return method(*args, **kwargs)
        else:
            method(*args, **kwargs)
            return self

    def get_attribute(self, attr_name: str) -> pd.Series[Any]:
        """
        Retrieve a specified attribute for active agents in the AgentSetDF.

        Parameters:
        ----------
        attr_name : str
            The name of the attribute to retrieve.

        Returns:
        ----------
        pd.Series[Any]
            A list of attribute values from each active agent in AgentSetDF.
        """
        return self.agents.loc[
            self.agents.index.isin(self.active_agents.index), attr_name
        ]

    def set_attribute(self, attr_name: str, value: Any) -> AgentSetDF:
        """
        Set a specified attribute for active agents in the AgentSetDF.

        Parameters:
        ----------
        attr_name : str
            The name of the attribute to set for each agent.
        value : Any
            The value assigned to the attribute. If the value is a scalar, it is assigned to all active agents.
            If the value is array-like, it must be the same length as the number of active agents.

        Returns:
        ----------
        AgentSetDF
            The updated AgentSetDF.
        """
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
    ) -> AgentSetDF:
        """Add n agents to the AgentSet.
        If index is not specified, the agents are assigned 64-bit random unique IDs using the model's generator.
        The other arguments are passed to the pandas.DataFrame constructor.

        Attributes
        ----------
        n : int
            The number of agents to add.
        data : array-like, Iterable, dict, or DataFrame, optional
                Dict can contain Series, arrays, constants, dataclass or list-like objects. If data is a dict, column order follows insertion-order. If a dict contains Series which have an index defined, it is aligned by its index. This alignment also occurs if data is a Series or a DataFrame itself. Alignment is done on Series/DataFrame inputs.
                If data is a list of dicts, column order follows insertion-order.
        index : Index or array-like
                Index to use for resulting frame. Will default to RangeIndex if no indexing information part of input data and no index provided.
        columns : Index or array-like
                Column labels to use for resulting frame when data does not have them, defaulting to RangeIndex(0, 1, 2, …, n). If data contains column labels, will perform column selection instead.
        dtype : dtype, default None
                Data type to force. Only a single dtype is allowed. If None, infer.

        Returns
        ----------
        AgentSetDF
            The updated self
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

    def discard(self, id: int) -> AgentSetDF:
        """Remove a specified agent. If the agent is not found, does not raise an error.

        Parameters
        ----------
        id : int
            The ID of the agent to remove.

        Returns
        ----------
        AgentSetDF
            The updated AgentSetDF."""
        with suppress(KeyError):
            self.agents.drop(id, inplace=True)
        return self

    def remove(self, id: int) -> AgentSetDF:
        """Remove an agent from the AgentSetDF. If the agent is not found, raises a KeyError.

        Parameters
        ----------
        id : int
            The ID of the agent to remove.

        Returns
        ----------
        AgentSetDF
            The updated AgentSetDF.
        """
        self.agents.drop(id, inplace=True)
        return self
