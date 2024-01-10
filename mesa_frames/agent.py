from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Callable, Generator, Sequence, cast

import numpy as np
import pandas as pd

# import polars as pl
from mesa import Agent
from model import ModelDF
from numpy.random import Generator
from pandas import DataFrame

if TYPE_CHECKING:
    from mesa.model import Model
    from mesa.space import Position
    from pandas import Index, Series
    from pandas.core.arrays.base import ExtensionArray

    ArrayLike = ExtensionArray | np.ndarray
    AnyArrayLike = ArrayLike | Index | Series
    ValueKeyFunc = Callable[[Series], Series | AnyArrayLike] | None
    from mesa.space import Position


class AgentSetPandas(DataFrame):
    """
    Attributes
    ----------
    agent_type : Agent
        The type of the Agent.
    model : model
        model: The ABM model instance to which this AgentSet belongs."""

    model: ModelDF
    agent_type: type[Agent]

    @property
    def _constructor(self):
        return AgentSetPandas

    def __new__(cls, n: int, agent_type: type[Agent], model: ModelDF, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self, n: int, agent_type: type[Agent], model: ModelDF, *args, **kwargs
    ):
        super(DataFrame, self).__init__(*args, **kwargs)
        self.model = model
        self.agent_type = agent_type
        self.add_agents(n)

    def __getitem__(self, key) -> AgentSetPandas:
        result = super().__getitem__(key)
        if isinstance(result, DataFrame):
            # Create AgentSetPandas with DataFrame-specific data
            return AgentSetPandas(0, self.agent_type, self.model, data=result)
        elif isinstance(result, pd.Series):
            # Convert Series to DataFrame and then create AgentSetPandas
            return AgentSetPandas(
                0, self.agent_type, self.model, data=result.to_frame()
            )
        else:
            return result

    def select(
        self,
        filter_func: Callable[[AgentSetPandas], pd.Series[bool]] | None = None,
        n: int = 0,
        inplace: bool = False,
    ) -> AgentSetPandas | None:
        """
        Select a subset of agents from the AgentSet based on a filter function and/or quantity limit.

        Attributes:
        ----------
        filter_func : Callable[[AgentSetPandas],  pd.Series[bool]], optional
            A function that takes the AgentSet and returns a boolean mask over the agents indicating which agents
            should be included in the result. Defaults to None, meaning no filtering is applied.
        n : int, optional
            The number of agents to select. If 0, all matching agents are selected. Defaults to 0.
        inplace : bool, optional
            If True, modifies the current AgentSet; otherwise, returns a new AgentSet. Defaults to False.

        Returns:
        ----------
        AgentSet: A new AgentSet containing the selected agents, unless inplace is True, in which case the current AgentSet is updated.
        """
        mask = pd.Series(True, index=self.index)
        if filter_func:
            mask = filter_func(self)
        mask = mask & self.sample(n).index.isin(mask.index)
        if inplace:
            # Apply the mask in-place
            self.loc[:, :] = self[mask]
        else:
            # Return a new instance
            return AgentSetPandas(0, self.agent_type, self.model, self[mask])

    def shuffle(self, inplace: bool = False) -> AgentSetPandas | None:
        """Randomly shuffle the agents in the AgentSet."""
        if inplace:
            self.loc[:, :] = self.sample(frac=1)
        else:
            return AgentSetPandas(0, self.agent_type, self.model, self.sample(frac=1))

    def sort(
        self,
        by: str | Sequence[str],
        key: ValueKeyFunc | None,
        ascending: bool | Sequence[bool] = True,
        inplace: bool = False,
    ) -> AgentSetPandas | None:
        """
        Sort the agents in the AgentSetPandas based on a specified attribute or custom function.

        Args:
            key (Callable[[Agent], Any] | str): A function or attribute name based on which the agents are sorted.
            ascending (bool, optional): If True, the agents are sorted in ascending order. Defaults to False.
            inplace (bool, optional): If True, sorts the agents in the current AgentSetPandas; otherwise, returns a new sorted AgentSet. Defaults to False.

        Returns:
            AgentSetPandas: A sorted AgentSetPandas. Returns the current AgentSetPandas if inplace is True.
        """
        return cast(
            "AgentSetPandas",
            self.sort_values(by=by, key=key, ascending=ascending, inplace=inplace),
        )

    def do(self, method_name: str, *args, sequential=False, **kwargs) -> AgentSetPandas:
        """
        Invoke a method on each agent in the AgentSet.

        Parameters:
        ----------
        method_name (str): The name of the method to call on each agent.
        *args: Variable length argument list passed to the method being called.
        sequential = False
        **kwargs: Arbitrary keyword arguments passed to the method being called.

        Returns:
        ----------
        AgentSetPandas: The results of the method calls
        """
        method = getattr(self, method_name)
        if sequential:
            return self.apply(method, axis=0, args=args, **kwargs)
        else:
            return self.apply(method, axis=1, args=args, **kwargs)

    def get_attribute(self, attr_name: str) -> AgentSetPandas:
        """
        Retrieve a specified attribute from each agent in the AgentSet.

        Args:
            attr_name (str): The name of the attribute to retrieve from each agent.

        Returns:
            list[Any]: A list of attribute values from each agent in the set.
        """
        return self[attr_name]

    def add_agents(self, n: int):
        """Add n agents to the AgentSet.

        Attributes
        ----------
        n : int
            The number of agents to add.
        """
        # First, let's collect attributes from each agent_type.
        callables = []
        values = []
        attributes = []
        for agent_type in reversed(self.agent_type.__mro__):
            for attribute in agent_type.__dict__.keys():
                if attribute[:2] != "__":
                    attributes.append(attribute)
                    value = getattr(agent_type, attribute)
                    if callable(value):
                        callables.append((attribute, value))
                    else:
                        values.append((attribute, value))
        # Now, let's create the agents.
        self.index = pd.Index(self.model.random.random(n) % 1)
        self.columns = list(attributes)

        # Finally, let's assign the values to the attribtutes.

        for attribute, value in values:
            self[attribute] = value

        for attribute, value in callables:
            self[attribute] = value(self)

    def discard(self, agent: Agent) -> AgentSetPandas | None:
        """Remove an agent from the agentset."""
        with suppress(KeyError):
            self.drop(agent.unique_id, inplace=True)

    def remove(self, agent: Agent):
        """Remove an agent from the agentset."""
        self.drop(agent.unique_id, inplace=True)

    @property
    def random(self) -> Generator:
        """
        Provide access to the model's random number generator.

        Returns:
            Random: The random number generator associated with the model.
        """
        return self.model.random


class AgentDF(Agent):
    unique_id: int

    def __init__(self, unique_id: int, model: ModelDF):
        self.unique_id = unique_id
        self.model = model
        self.pos: Position | None = None
