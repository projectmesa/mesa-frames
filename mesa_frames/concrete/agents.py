"""
Concrete implementation of the agents collection for mesa-frames.

This module provides the concrete implementation of the agents collection class
for the mesa-frames library. It defines the AgentsDF class, which serves as a
container for all agent sets in a model, leveraging DataFrame-based storage for
improved performance.

Classes:
    AgentsDF(AgentContainer):
        A collection of AgentSetDFs. This class acts as a container for all
        agents in the model, organizing them into separate AgentSetDF instances
        based on their types.

The AgentsDF class is designed to be used within ModelDF instances to manage
all agents in the simulation. It provides methods for adding, removing, and
accessing agents and agent sets, while taking advantage of the performance
benefits of DataFrame-based agent storage.

Usage:
    The AgentsDF class is typically instantiated and used within a ModelDF subclass:

    from mesa_frames.concrete.model import ModelDF
    from mesa_frames.concrete.agents import AgentsDF
    from mesa_frames.concrete import AgentSetPolars

    class MyCustomModel(ModelDF):
        def __init__(self):
            super().__init__()
            # Adding agent sets to the collection
            self.agents += AgentSetPolars(self)
            self.agents += AnotherAgentSetPolars(self)

        def step(self):
            # Step all agent sets
            self.agents.do("step")

Note:
    This concrete implementation builds upon the abstract AgentContainer class
    defined in the mesa_frames.abstract package, providing a ready-to-use
    agents collection that integrates with the DataFrame-based agent storage system.

For more detailed information on the AgentsDF class and its methods, refer to
the class docstring.
"""

from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Literal, cast

import polars as pl
from typing_extensions import Any, Self, overload

from mesa_frames.abstract.agents import AgentContainer, AgentSetDF
from mesa_frames.types_ import (
    AgentMask,
    AgnosticAgentMask,
    BoolSeries,
    DataFrame,
    IdsLike,
    Index,
    Series,
)

if TYPE_CHECKING:
    from mesa_frames.concrete.model import ModelDF


class AgentsDF(AgentContainer):
    """A collection of AgentSetDFs. All agents of the model are stored here."""

    _agentsets: list[AgentSetDF]
    _ids: pl.Series

    def __init__(self, model: "ModelDF") -> None:
        """Initialize a new AgentsDF.

        Parameters
        ----------
        model : ModelDF
            The model associated with the AgentsDF.
        """
        self._model = model
        self._agentsets = []
        self._ids = pl.Series(name="unique_id", dtype=pl.Int64)

    def add(
        self, agents: AgentSetDF | Iterable[AgentSetDF], inplace: bool = True
    ) -> Self:
        """Add an AgentSetDF to the AgentsDF.

        Parameters
        ----------
        agents : AgentSetDF | Iterable[AgentSetDF]
            The AgentSetDF to add.
        inplace : bool, optional
            Whether to add the AgentSetDF in place.

        Returns
        -------
        Self
            The updated AgentsDF.

        Raises
        ------
        ValueError
            If some agentsets are already present in the AgentsDF or if the IDs are not unique.
        """
        obj = self._get_obj(inplace)
        other_list = obj._return_agentsets_list(agents)
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
        self, agents: IdsLike | AgentSetDF | Iterable[AgentSetDF]
    ) -> bool | pl.Series:
        if isinstance(agents, int):
            return agents in self._ids
        elif isinstance(agents, AgentSetDF):
            return self._check_agentsets_presence([agents]).any()
        elif isinstance(agents, Iterable):
            if len(agents) == 0:
                return True
            elif isinstance(next(iter(agents)), AgentSetDF):
                agents = cast(Iterable[AgentSetDF], agents)
                return self._check_agentsets_presence(list(agents))
            else:  # IDsLike
                agents = cast(IdsLike, agents)

                return pl.Series(agents).is_in(self._ids)

    @overload
    def do(
        self,
        method_name: str,
        *args,
        mask: AgnosticAgentMask | IdsLike | dict[AgentSetDF, AgentMask] = None,
        return_results: Literal[False] = False,
        inplace: bool = True,
        **kwargs,
    ) -> Self: ...

    @overload
    def do(
        self,
        method_name: str,
        *args,
        mask: AgnosticAgentMask | IdsLike | dict[AgentSetDF, AgentMask] = None,
        return_results: Literal[True],
        inplace: bool = True,
        **kwargs,
    ) -> dict[AgentSetDF, Any]: ...

    def do(
        self,
        method_name: str,
        *args,
        mask: AgnosticAgentMask | IdsLike | dict[AgentSetDF, AgentMask] = None,
        return_results: bool = False,
        inplace: bool = True,
        **kwargs,
    ) -> Self | Any:
        obj = self._get_obj(inplace)
        agentsets_masks = obj._get_bool_masks(mask)
        if return_results:
            return {
                agentset: agentset.do(
                    method_name,
                    *args,
                    mask=mask,
                    return_results=return_results,
                    **kwargs,
                    inplace=inplace,
                )
                for agentset, mask in agentsets_masks.items()
            }
        else:
            obj._agentsets = [
                agentset.do(
                    method_name,
                    *args,
                    mask=mask,
                    return_results=return_results,
                    **kwargs,
                    inplace=inplace,
                )
                for agentset, mask in agentsets_masks.items()
            ]
            return obj

    def get(
        self,
        attr_names: str | Collection[str] | None = None,
        mask: AgnosticAgentMask | IdsLike | dict[AgentSetDF, AgentMask] = None,
    ) -> dict[AgentSetDF, Series] | dict[AgentSetDF, DataFrame]:
        agentsets_masks = self._get_bool_masks(mask)
        return {
            agentset: agentset.get(attr_names, mask)
            for agentset, mask in agentsets_masks.items()
        }

    def remove(
        self, agents: AgentSetDF | Iterable[AgentSetDF] | IdsLike, inplace: bool = True
    ) -> Self:
        obj = self._get_obj(inplace)
        if agents is None or (isinstance(agents, Iterable) and len(agents) == 0):
            return obj
        if isinstance(agents, AgentSetDF):
            agents = [agents]
        if isinstance(agents, Iterable) and isinstance(next(iter(agents)), AgentSetDF):
            # We have to get the index of the original AgentSetDF because the copy made AgentSetDFs with different hash
            ids = [self._agentsets.index(agentset) for agentset in iter(agents)]
            ids.sort(reverse=True)
            removed_ids = pl.Series(dtype=pl.Int64)
            for id in ids:
                removed_ids = pl.concat(
                    [removed_ids, pl.Series(obj._agentsets[id].index)]
                )
                obj._agentsets.pop(id)

        else:  # IDsLike
            if isinstance(agents, int):
                agents = [agents]
            elif isinstance(agents, DataFrame):
                agents = agents["unique_id"]
            removed_ids = pl.Series(agents)
            deleted = 0

            for agentset in obj._agentsets:
                initial_len = len(agentset)
                agentset._discard(removed_ids)
                deleted += initial_len - len(agentset)
                if deleted == len(removed_ids):
                    break
            if deleted < len(removed_ids):  # TODO: fix type hint
                raise KeyError(
                    "There exist some IDs which are not present in any agentset"
                )
        try:
            obj.space.remove_agents(removed_ids, inplace=True)
        except ValueError:
            pass
        obj._ids = obj._ids.filter(obj._ids.is_in(removed_ids).not_())
        return obj

    def select(
        self,
        mask: AgnosticAgentMask | IdsLike | dict[AgentSetDF, AgentMask] = None,
        filter_func: Callable[[AgentSetDF], AgentMask] | None = None,
        n: int | None = None,
        inplace: bool = True,
        negate: bool = False,
    ) -> Self:
        obj = self._get_obj(inplace)
        agentsets_masks = obj._get_bool_masks(mask)
        if n is not None:
            n = n // len(agentsets_masks)
        obj._agentsets = [
            agentset.select(
                mask=mask, filter_func=filter_func, n=n, negate=negate, inplace=inplace
            )
            for agentset, mask in agentsets_masks.items()
        ]
        return obj

    def set(
        self,
        attr_names: str | dict[AgentSetDF, Any] | Collection[str],
        values: Any | None = None,
        mask: AgnosticAgentMask | IdsLike | dict[AgentSetDF, AgentMask] = None,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        agentsets_masks = obj._get_bool_masks(mask)
        if isinstance(attr_names, dict):
            for agentset, values in attr_names.items():
                if not inplace:
                    # We have to get the index of the original AgentSetDF because the copy made AgentSetDFs with different hash
                    id = self._agentsets.index(agentset)
                    agentset = obj._agentsets[id]
                agentset.set(
                    attr_names=values, mask=agentsets_masks[agentset], inplace=True
                )
        else:
            obj._agentsets = [
                agentset.set(
                    attr_names=attr_names, values=values, mask=mask, inplace=True
                )
                for agentset, mask in agentsets_masks.items()
            ]
        return obj

    def shuffle(self, inplace: bool = True) -> Self:
        obj = self._get_obj(inplace)
        obj._agentsets = [agentset.shuffle(inplace=True) for agentset in obj._agentsets]
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

    def step(self, inplace: bool = True) -> Self:
        """Advance the state of the agents in the AgentsDF by one step.

        Parameters
        ----------
        inplace : bool, optional
            Whether to update the AgentsDF in place, by default True

        Returns
        -------
        Self
        """
        obj = self._get_obj(inplace)
        for agentset in obj._agentsets:
            agentset.step()
        return obj

    def _check_ids_presence(self, other: list[AgentSetDF]) -> pl.DataFrame:
        """Check if the IDs of the agents to be added are unique.

        Parameters
        ----------
        other : list[AgentSetDF]
            The AgentSetDFs to check.

        Returns
        -------
        pl.DataFrame
            A DataFrame with the unique IDs and a boolean column indicating if they are present.
        """
        presence_df = pl.DataFrame(
            data={"unique_id": self._ids, "present": True},
            schema={"unique_id": pl.Int64, "present": pl.Boolean},
        )
        for agentset in other:
            new_ids = pl.Series(agentset.index)
            presence_df = pl.concat(
                [
                    presence_df,
                    (
                        new_ids.is_in(presence_df["unique_id"])
                        .to_frame("present")
                        .with_columns(unique_id=new_ids)
                        .select(["unique_id", "present"])
                    ),
                ]
            )
        presence_df = presence_df.slice(self._ids.len())
        return presence_df

    def _check_agentsets_presence(self, other: list[AgentSetDF]) -> pl.Series:
        """Check if the agent sets to be added are already present in the AgentsDF.

        Parameters
        ----------
        other : list[AgentSetDF]
            The AgentSetDFs to check.

        Returns
        -------
        pl.Series
            A boolean Series indicating if the agent sets are present.

        Raises
        ------
        ValueError
            If the agent sets are already present in the AgentsDF.
        """
        other_set = set(other)
        return pl.Series(
            [agentset in other_set for agentset in self._agentsets], dtype=pl.Boolean
        )

    def _get_bool_masks(
        self,
        mask: AgnosticAgentMask | IdsLike | dict[AgentSetDF, AgentMask] = None,
    ) -> dict[AgentSetDF, BoolSeries]:
        return_dictionary = {}
        if not isinstance(mask, dict):
            mask = {agentset: mask for agentset in self._agentsets}
        for agentset, mask in mask.items():
            return_dictionary[agentset] = agentset._get_bool_mask(mask)
        return return_dictionary

    def _return_agentsets_list(
        self, agentsets: AgentSetDF | Iterable[AgentSetDF]
    ) -> list[AgentSetDF]:
        """Convert the agentsets to a list of AgentSetDF.

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
        Self
            A new AgentsDF with the added AgentSetDFs.
        """
        return super().__add__(other)

    def __getattr__(self, name: str) -> dict[AgentSetDF, Any]:
        # Avoids infinite recursion of private attributes
        if __debug__:  # Only execute in non-optimized mode
            if name.startswith("_"):
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )
        return {agentset: getattr(agentset, name) for agentset in self._agentsets}

    @overload
    def __getitem__(
        self, key: str | tuple[dict[AgentSetDF, AgentMask], str]
    ) -> dict[str, Series]: ...

    @overload
    def __getitem__(
        self,
        key: (
            Collection[str]
            | AgnosticAgentMask
            | IdsLike
            | tuple[dict[AgentSetDF, AgentMask], Collection[str]]
        ),
    ) -> dict[str, DataFrame]: ...

    def __getitem__(
        self,
        key: (
            str
            | Collection[str]
            | AgnosticAgentMask
            | IdsLike
            | tuple[dict[AgentSetDF, AgentMask], str]
            | tuple[dict[AgentSetDF, AgentMask], Collection[str]]
        ),
    ) -> dict[str, Series] | dict[str, DataFrame]:
        return super().__getitem__(key)

    def __iadd__(self, agents: AgentSetDF | Iterable[AgentSetDF]) -> Self:
        """Add AgentSetDFs to the AgentsDF through the += operator.

        Parameters
        ----------
        agents : AgentSetDF | Iterable[AgentSetDF]
            The AgentSetDFs to add.

        Returns
        -------
        Self
            The updated AgentsDF.
        """
        return super().__iadd__(agents)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return (agent for agentset in self._agentsets for agent in iter(agentset))

    def __isub__(self, agents: AgentSetDF | Iterable[AgentSetDF] | IdsLike) -> Self:
        """Remove AgentSetDFs from the AgentsDF through the -= operator.

        Parameters
        ----------
        agents : AgentSetDF | Iterable[AgentSetDF] | IdsLike
            The AgentSetDFs to remove.

        Returns
        -------
        Self
            The updated AgentsDF.
        """
        return super().__isub__(agents)

    def __len__(self) -> int:
        return sum(len(agentset._agents) for agentset in self._agentsets)

    def __repr__(self) -> str:
        return "\n".join([repr(agentset) for agentset in self._agentsets])

    def __reversed__(self) -> Iterator:
        return (
            agent
            for agentset in self._agentsets
            for agent in reversed(agentset._backend)
        )

    def __setitem__(
        self,
        key: (
            str
            | Collection[str]
            | AgnosticAgentMask
            | IdsLike
            | tuple[dict[AgentSetDF, AgentMask], str]
            | tuple[dict[AgentSetDF, AgentMask], Collection[str]]
        ),
        values: Any,
    ) -> None:
        super().__setitem__(key, values)

    def __str__(self) -> str:
        return "\n".join([str(agentset) for agentset in self._agentsets])

    def __sub__(self, agents: IdsLike | AgentSetDF | Iterable[AgentSetDF]) -> Self:
        """Remove AgentSetDFs from a new AgentsDF through the - operator.

        Parameters
        ----------
        agents : IdsLike | AgentSetDF | Iterable[AgentSetDF]
            The AgentSetDFs to remove.

        Returns
        -------
        Self
            A new AgentsDF with the removed AgentSetDFs.
        """
        return super().__sub__(agents)

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

    @active_agents.setter
    def active_agents(
        self, agents: AgnosticAgentMask | IdsLike | dict[AgentSetDF, AgentMask]
    ) -> None:
        self.select(agents, inplace=True)

    @property
    def agentsets_by_type(self) -> dict[type[AgentSetDF], Self]:
        """Get the agent sets in the AgentsDF grouped by type.

        Returns
        -------
        dict[type[AgentSetDF], Self]
            A dictionary mapping agent set types to the corresponding AgentsDF.
        """

        def copy_without_agentsets() -> Self:
            return self.copy(deep=False, skip=["_agentsets"])

        dictionary: dict[type[AgentSetDF], Self] = defaultdict(copy_without_agentsets)

        for agentset in self._agentsets:
            agents_df = dictionary[agentset.__class__]
            agents_df._agentsets = []
            agents_df._agentsets = agents_df._agentsets + [agentset]
            dictionary[agentset.__class__] = agents_df
        return dictionary

    @property
    def inactive_agents(self) -> dict[AgentSetDF, DataFrame]:
        return {agentset: agentset.inactive_agents for agentset in self._agentsets}

    @property
    def index(self) -> dict[AgentSetDF, Index]:
        return {agentset: agentset.index for agentset in self._agentsets}

    @property
    def pos(self) -> dict[AgentSetDF, DataFrame]:
        return {agentset: agentset.pos for agentset in self._agentsets}
