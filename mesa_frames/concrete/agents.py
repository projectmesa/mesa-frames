from operator import ne
from typing import Any, Callable, Iterable, Iterator, Literal, Self, Sequence, overload

import polars as pl

from mesa_frames.abstract.agents import AgentContainer, AgentSetDF, Collection, Hashable
from mesa_frames.concrete.agentset_pandas import AgentSetPandas
from mesa_frames.concrete.agentset_polars import AgentSetPolars
from mesa_frames.types import BoolSeries, DataFrame, IdsLike, MaskLike, Series


class AgentsDF(AgentContainer):
    _agentsets: list[AgentSetDF]
    _copy_with_method: dict[str, tuple[str, list[str]]] = {
        "_agentsets": ("copy", []),
    }
    _backend: str
    _ids: pl.Series
    """A collection of AgentSetDFs. All agents of the model are stored here.

    Attributes
    ----------
    _agentsets : list[AgentSetDF]
        The agent sets contained in this collection.
    _copy_with_method : dict[str, tuple[str, list[str]]]
        A dictionary of attributes to copy with a specified method and arguments.
    _backend : str
        The backend used for data operations.

    Properties
    ----------
    active_agents(self) -> dict[str, pd.DataFrame]
        Get the active agents in the AgentsDF.
    agents(self) -> dict[str, pd.DataFrame]
        Get or set the agents in the AgentsDF.
    inactive_agents(self) -> dict[str, pd.DataFrame]
        Get the inactive agents in the AgentsDF.
    model(self) -> ModelDF
        Get the model associated with the AgentsDF.
    random(self) -> np.random.Generator
        Get the random number generator associated with the model.

    Methods
    -------
    __init__(self) -> None
        Initialize a new AgentsDF.
    add(self, other: AgentSetDF | Iterable[AgentSetDF], inplace: bool = True) -> Self
        Add agents to the AgentsDF.
    contains(self, ids: IdsLike) -> bool | pl.Series
        Check if agents with the specified IDs are in the AgentsDF.
    copy(self, deep: bool = False, memo: dict | None = None) -> Self
        Create a copy of the AgentsDF.
    discard(self, ids: IdsLike, inplace: bool = True) -> Self
        Remove an agent from the AgentsDF. Does not raise an error if the agent is not found.
    do(self, method_name: str, *args, return_results: bool = False, inplace: bool = True, **kwargs) -> Self | Any
        Invoke a method on the AgentsDF.
    get(self, attr_names: str | Collection[str] | None = None, mask: MaskLike = None) -> dict[str, Series] | dict[str, DataFrame]
        Retrieve the value of a specified attribute for each agent in the AgentsDF.
    remove(self, ids: IdsLike, inplace: bool = True) -> Self
        Remove agents from the AgentsDF.
    select(self, mask: MaskLike = None, filter_func: Callable[[Self], MaskLike] | None = None, n: int | None = None, negate: bool = False, inplace: bool = True) -> Self
        Select agents in the AgentsDF based on the given criteria.
    set(self, attr_names: str | Collection[str] | dict[str, Any] | None = None, values: Any | None = None, mask: MaskLike | None = None, inplace: bool = True) -> Self
        Set the value of a specified attribute or attributes for each agent in the mask in the AgentsDF.
    shuffle(self, inplace: bool = True) -> Self
        Shuffle the order of agents in the AgentsDF.
    sort(self, by: str | Sequence[str], ascending: bool | Sequence[bool] = True, inplace: bool = True, **kwargs) -> Self
        Sort the agents in the AgentsDF based on the given criteria.
    _check_ids(self, other: AgentSetDF | Iterable[AgentSetDF]) -> None
        Check if the IDs of the agents to be added are unique.
    __add__(self, other: AgentSetDF | Iterable[AgentSetDF]) -> Self
        Add AgentSetDFs to a new AgentsDF through the + operator.
    __getattr__(self, key: str) -> Any
        Retrieve an attribute of the underlying agent sets.
    __iadd__(self, other: AgentSetDF | Iterable[AgentSetDF]) -> Self
        Add AgentSetDFs to the AgentsDF through the += operator.
    __iter__(self) -> Iterator
        Get an iterator for the agents in the AgentsDF.
    __len__(self) -> int
        Get the number of agents in the AgentsDF.
    __repr__(self) -> str
        Get the string representation of the AgentsDF.
    __reversed__(self) -> Iterator
        Get a reversed iterator for the agents in the AgentsDF.
    __str__(self) -> str
        Get the string representation of the AgentsDF.
    """

    def __init__(self) -> None:
        self._agentsets = []
        self._ids = pl.Series(name="unique_id", dtype=pl.Int64)

    def add(
        self, other: AgentSetDF | Iterable[AgentSetDF], inplace: bool = True
    ) -> Self:
        """Add an AgentSetDF to the AgentsDF.

        Parameters
        ----------
        other : AgentSetDF
            The AgentSetDF to add.
        inplace : bool
            Whether to add the AgentSetDF in place.

        Returns
        ----------
        Self
            The updated AgentsDF.
        """
        obj = self._get_obj(inplace)
        self._check_ids(other)
        if isinstance(other, Iterable):
            obj._agentsets += other
        else:
            obj._agentsets.append(other)
        return self

    @overload
    def contains(self, ids: int) -> bool: ...

    @overload
    def contains(self, ids: IdsLike) -> pl.Series: ...

    def contains(self, ids: IdsLike) -> bool | pl.Series:
        if isinstance(ids, int):
            return ids in self._ids
        else:
            return pl.Series(ids).is_in(self._ids)

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
    ) -> dict[str, Any]: ...

    def do(
        self,
        method_name: str,
        *args,
        return_results: bool = False,
        inplace: bool = True,
        **kwargs,
    ) -> Self | Any:
        obj = self._get_obj(inplace)
        if return_results:
            return {
                agentset.__class__.__name__: agentset.do(
                    method_name,
                    *args,
                    return_results=return_results,
                    **kwargs,
                    inplace=inplace,
                )
                for agentset in obj._agentsets
            }
        else:
            obj._agentsets = [
                agentset.do(
                    method_name,
                    *args,
                    return_results=return_results,
                    **kwargs,
                    inplace=inplace,
                )
                for agentset in obj._agentsets
            ]
            return obj

    def get(
        self,
        attr_names: str | list[str] | None = None,
        mask: MaskLike | None = None,
    ) -> dict[str, Series] | dict[str, DataFrame]:
        return {
            agentset.__class__.__name__: agentset.get(attr_names, mask)
            for agentset in self._agentsets
        }

    def remove(self, ids: IdsLike, inplace: bool = True) -> Self:
        obj = self._get_obj(inplace)
        deleted = 0
        for agentset in obj._agentsets:
            initial_len = len(agentset)
            agentset.discard(ids, inplace=True)
            deleted += initial_len - len(agentset)
        if deleted < len(list(ids)):  # TODO: fix type hint
            raise KeyError(f"None of the agentsets contain the ID {MaskLike}.")
        return obj

    def set(
        self,
        attr_names: str | dict[str, Any] | Collection[str],
        values: Any | None = None,
        mask: MaskLike | None = None,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        obj._agentsets = [
            agentset.set(
                attr_names=attr_names, values=values, mask=mask, inplace=inplace
            )
            for agentset in obj._agentsets
        ]
        return obj

    def select(
        self,
        mask: MaskLike | None = None,
        filter_func: Callable[[AgentSetDF], MaskLike] | None = None,
        n: int | None = None,
        inplace: bool = True,
        negate: bool = False,
    ) -> Self:
        obj = self._get_obj(inplace)
        obj._agentsets = [
            agentset.select(
                mask=mask, filter_func=filter_func, n=n, negate=negate, inplace=inplace
            )
            for agentset in obj._agentsets
        ]
        return obj

    def shuffle(self, inplace: bool = True) -> Self:
        obj = self._get_obj(inplace)
        obj._agentsets = [agentset.shuffle(inplace) for agentset in obj._agentsets]
        return obj

    def sort(
        self,
        by: str | Sequence[str],
        ascending: bool | Sequence[bool] = True,
        inplace: bool = True,
        **kwargs,
    ) -> Self:
        obj = self._get_obj(inplace)
        obj._agentsets = [
            agentset.sort(by=by, ascending=ascending, inplace=inplace, **kwargs)
            for agentset in obj._agentsets
        ]
        return obj

    def _check_ids(self, other: AgentSetDF | Iterable[AgentSetDF]) -> None:
        """Check if the IDs of the agents to be added are unique.

        Parameters
        ----------
        other : AgentSetDF | Iterable[AgentSetDF]
            The AgentSetDFs to check.

        Raises
        ------
        ValueError
            If the agent set contains IDs already present in agents.
        """
        for agentset in other if isinstance(other, Iterable) else [other]:
            if isinstance(agentset, AgentSetPandas):
                new_ids = pl.Series(agentset._agents.index)
            elif isinstance(agentset, AgentSetPolars):
                new_ids = agentset._agents["unique_id"]
            if new_ids.is_in(self._ids).any():
                raise ValueError(
                    "The agent set contains IDs already present in agents."
                )

    def __add__(self, other: AgentSetDF | Iterable[AgentSetDF]) -> Self:
        """Add AgentSetDFs to a new AgentsDF through the + operator.

        Parameters
        ----------
        other : AgentSetDF | Iterable[AgentSetDF]
            The AgentSetDFs to add.

        Returns
        -------
        AgentsDF
            A new AgentsDF with the added AgentSetDFs.
        """
        return super().__add__(other)

    def __getattr__(self, name: str) -> dict[str, Any]:
        return {
            agentset.__class__.__name__: getattr(agentset, name)
            for agentset in self._agentsets
        }

    def __iadd__(self, other: AgentSetDF | Iterable[AgentSetDF]) -> Self:
        """Add AgentSetDFs to the AgentsDF through the += operator.

        Parameters
        ----------
        other : Self | AgentSetDF | Iterable[AgentSetDF]
            The AgentSetDFs to add.

        Returns
        -------
        AgentsDF
            The updated AgentsDF.
        """
        return super().__iadd__(other)

    def __iter__(self) -> Iterator:
        return (
            agent for agentset in self._agentsets for agent in iter(agentset._backend)
        )

    def __repr__(self) -> str:
        return str(
            {
                agentset.__class__.__name__: repr(agentset)
                for agentset in self._agentsets
            }
        )

    def __str__(self) -> str:
        return str(
            {agentset.__class__.__name__: str(agentset) for agentset in self._agentsets}
        )

    def __reversed__(self) -> Iterator:
        return (
            agent
            for agentset in self._agentsets
            for agent in reversed(agentset._backend)
        )

    def __len__(self) -> int:
        return sum(len(agentset._agents) for agentset in self._agentsets)

    @property
    def agents(self) -> dict[str, DataFrame]:
        return {
            agentset.__class__.__name__: agentset.agents for agentset in self._agentsets
        }

    @agents.setter
    def agents(self, other: Iterable[AgentSetDF]) -> None:
        """Set the agents in the AgentsDF.

        Parameters
        ----------
        other : Iterable[AgentSetDF]
            The AgentSetDFs to set.
        """
        self._agentsets = list(other)

    @property
    def active_agents(self) -> dict[str, DataFrame]:
        return {
            agentset.__class__.__name__: agentset.active_agents
            for agentset in self._agentsets
        }

    @property
    def inactive_agents(self):
        return {
            agentset.__class__.__name__: agentset.inactive_agents
            for agentset in self._agentsets
        }
