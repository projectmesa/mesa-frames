from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Literal, Self, cast, overload

from collections.abc import Collection

import polars as pl

from mesa_frames.abstract.agents import AgentContainer, AgentSetDF
from mesa_frames.types import DataFrame, IdsLike, MaskLike, Series


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
    _copy_with_method : dict[AgentSetDF, tuple[str, list[str]]]
        A dictionary of attributes to copy with a specified method and arguments.
    _backend : str
        The backend used for data operations.

    Properties
    ----------
    active_agents(self) -> dict[AgentSetDF, pd.DataFrame]
        Get the active agents in the AgentsDF.
    agents(self) -> dict[AgentSetDF, pd.DataFrame]
        Get or set the agents in the AgentsDF.
    inactive_agents(self) -> dict[AgentSetDF, pd.DataFrame]
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
    get(self, attr_names: str | Collection[str] | None = None, mask: MaskLike = None) -> dict[AgentSetDF, Series] | dict[AgentSetDF, DataFrame]
        Retrieve the value of a specified attribute for each agent in the AgentsDF.
    remove(self, ids: IdsLike, inplace: bool = True) -> Self
        Remove agents from the AgentsDF.
    select(self, mask: MaskLike = None, filter_func: Callable[[Self], MaskLike] | None = None, n: int | None = None, negate: bool = False, inplace: bool = True) -> Self
        Select agents in the AgentsDF based on the given criteria.
    set(self, attr_names: str | Collection[str] | dict[AgentSetDF, Any] | None = None, values: Any | None = None, mask: MaskLike | None = None, inplace: bool = True) -> Self
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
        self, agentsets: AgentSetDF | Iterable[AgentSetDF], inplace: bool = True
    ) -> Self:
        """Add an AgentSetDF to the AgentsDF.

        Parameters
        ----------
        agentsets : AgentSetDF | Iterable[AgentSetDF]
            The AgentSetDF to add.
        inplace : bool
            Whether to add the AgentSetDF in place.

        Returns
        ----------
        Self
            The updated AgentsDF.

        Raises
        ------
        ValueError
            If some agentsets are already present in the AgentsDF or if the IDs are not unique.
        """
        obj = self._get_obj(inplace)
        other_list = obj._return_agentsets_list(agentsets)
        if obj._check_agentsets_presence(other_list).any():
            raise ValueError("Some agentsets are already present in the AgentsDF.")
        new_ids = pl.concat(
            [obj._ids] + [pl.Series(agentset["unique_id"]) for agentset in other_list]
        )
        if new_ids.is_duplicated().any():
            raise ValueError("Some of the agent IDs are not unique.")
        obj._agentsets.extend(other_list)
        obj._ids = new_ids
        return obj

    @overload
    def contains(self, agents: int | AgentSetDF) -> bool: ...

    @overload
    def contains(self, agents: IdsLike | Iterable[AgentSetDF]) -> pl.Series: ...

    def contains(
        self, agents: AgentSetDF | IdsLike | Iterable[AgentSetDF]
    ) -> bool | pl.Series:
        if isinstance(agents, AgentSetDF):
            return self._check_agentsets_presence([agents]).any()
        elif isinstance(agents, Iterable) and isinstance(
            next(iter(agents)), AgentSetDF
        ):
            agents = cast(Iterable[AgentSetDF], agents)
            return self._check_agentsets_presence(list(agents))
        else:
            agents = cast(IdsLike, agents)
            return pl.Series(agents).is_in(self._ids)

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
    ) -> dict[AgentSetDF, Any]: ...

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
                agentset: agentset.do(
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
    ) -> dict[AgentSetDF, Series] | dict[AgentSetDF, DataFrame]:
        return {
            agentset: agentset.get(attr_names, mask) for agentset in self._agentsets
        }

    def remove(
        self, agents: AgentSetDF | Iterable[AgentSetDF] | IdsLike, inplace: bool = True
    ) -> Self:
        obj = self._get_obj(inplace)
        deleted = 0
        if isinstance(agents, AgentSetDF):
            self._agentsets.remove(agents)
        elif isinstance(agents, Iterable) and isinstance(
            next(iter(agents)), AgentSetDF
        ):  # Faster than controlling every AgentSetDF
            for agentset in iter(agents):
                self._agentsets.remove(agentset)  # type: ignore (Pylance can't recognize agents as Iterable[AgentSetDF])
        else:  # IDsLike
            for agentset in obj._agentsets:
                initial_len = len(agentset)
                agentset.discard(agents, inplace=True)
                deleted += initial_len - len(agentset)
            if deleted < len(list(agents)):  # TODO: fix type hint
                raise KeyError(
                    "There exist some IDs which are not present in any agentset"
                )
        return obj

    def set(
        self,
        attr_names: str | dict[AgentSetDF, Any] | Collection[str],
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

    def _check_ids_presence(self, other: list[AgentSetDF]) -> pl.DataFrame:
        """Check if the IDs of the agents to be added are unique.

        Parameters
        ----------
        other : list[AgentSetDF]
            The AgentSetDFs to check.

        Raises
        ------
        ValueError
            If the agent set contains IDs already present in agents.
        """
        presence_df = pl.DataFrame(
            data={"unique_id": self._ids},
            schema={"unique_id": pl.Int64, "present": pl.Boolean},
        )
        for agentset in other:
            new_ids = pl.Series(agentset["unique_id"])
            presence_df = pl.concat(
                [
                    presence_df,
                    (
                        new_ids.is_in(presence_df["unique_id"])
                        .to_frame()
                        .with_columns("unique_id", new_ids)
                    ),
                ]
            )
        return presence_df

    def _check_agentsets_presence(self, other: list[AgentSetDF]) -> pl.Series:
        """Check if the agent sets to be added are already present in the AgentsDF.

        Parameters
        ----------
        other : list[AgentSetDF]
            The AgentSetDFs to check.

        Raises
        ------
        ValueError
            If the agent sets are already present in the AgentsDF.
        """
        other_set = set(other)
        return pl.Series(
            [agentset in other_set for agentset in self._agentsets], dtype=pl.Boolean
        )

    def _return_agentsets_list(
        self, agentsets: AgentSetDF | Iterable[AgentSetDF]
    ) -> list[AgentSetDF]:
        """Convert the agentsets to a list of AgentSetDF

        Parameters
        ----------
        agentsets : AgentSetDF | Iterable[AgentSetDF]

        Returns
        -------
        list[AgentSetDF]
        """
        return [agentsets] if isinstance(agentsets, AgentSetDF) else list(agentsets)

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

    def __getattr__(self, name: str) -> dict[AgentSetDF, Any]:
        return {agentset: getattr(agentset, name) for agentset in self._agentsets}

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
        return "\n".join([repr(agentset) for agentset in self._agentsets])

    def __str__(self) -> str:
        return "\n".join([str(agentset) for agentset in self._agentsets])

    def __reversed__(self) -> Iterator:
        return (
            agent
            for agentset in self._agentsets
            for agent in reversed(agentset._backend)
        )

    def __len__(self) -> int:
        return sum(len(agentset._agents) for agentset in self._agentsets)

    @property
    def agents(self) -> dict[AgentSetDF, DataFrame]:
        return {agentset: agentset.agents for agentset in self._agentsets}

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
    def active_agents(self) -> dict[AgentSetDF, DataFrame]:
        return {agentset: agentset.active_agents for agentset in self._agentsets}

    @property
    def agentsets_by_type(self) -> dict[type[AgentSetDF], list[AgentSetDF]]:
        dictionary = defaultdict(list)
        for agentset in self._agentsets:
            dictionary[agentset.__class__] = dictionary[agentset.__class__] + [agentset]
        return dictionary

    @property
    def inactive_agents(self):
        return {agentset: agentset.inactive_agents for agentset in self._agentsets}
