"""Abstract space interface."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Collection, Sequence, Sized
from typing import Self

import numpy as np
from numpy.random import Generator

from mesa_frames.abstract.agentset import AbstractAgentSet
from mesa_frames.abstract.agentsetregistry import AbstractAgentSetRegistry
from mesa_frames.abstract.mixin import CopyMixin, DataFrameMixin
from mesa_frames.abstract.space.neighborhood import AbstractNeighborhood
from mesa_frames.types_ import (
    DataFrame,
    IdsLike,
    Series,
    SpaceCoordinate,
    SpaceCoordinates,
)


class Space(CopyMixin, DataFrameMixin):
    """The Space class is an abstract class that defines the interface for all space classes in mesa_frames."""

    _agents: DataFrame  # | GeoDataFrame  # Stores the agents placed in the space
    _neighborhood_obj: AbstractNeighborhood
    _center_col_names: list[
        str
    ]  # The column names of the center pos/agents in the neighbors/neighborhood method (eg. ['dim_0_center', 'dim_1_center', ...] in Grids, ['node_id_center', 'edge_id_center'] in Networks)
    _pos_col_names: list[
        str
    ]  # The column names of the positions in the _agents dataframe (eg. ['dim_0', 'dim_1', ...] in Grids, ['node_id', 'edge_id'] in Networks)

    def __init__(self, model: mesa_frames.concrete.model.Model) -> None:
        """Create a new Space.

        Parameters
        ----------
        model : mesa_frames.concrete.model.Model
        """
        self._model = model

    def move_agents(
        self,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        pos: SpaceCoordinate | SpaceCoordinates,
        inplace: bool = True,
    ) -> Self:
        """Move agents in the Space to the specified coordinates.

        If some agents are not placed,raises a RuntimeWarning.

        Parameters
        ----------
        agents : IdsLike | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry]
            The agents to move
        pos : SpaceCoordinate | SpaceCoordinates
            The coordinates for each agents. The length of the coordinates must match the number of agents.
        inplace : bool, optional
            Whether to perform the operation inplace, by default True

        Raises
        ------
        RuntimeWarning
            If some agents are not placed in the space.
        ValueError
            - If some agents are not part of the model.
            - If agents is IdsLike and some agents are present multiple times.

        Returns
        -------
        Self
        """
        obj = self._get_obj(inplace=inplace)
        return obj._place_or_move_agents(agents=agents, pos=pos, is_move=True)

    def place_agents(
        self,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        pos: SpaceCoordinate | SpaceCoordinates,
        inplace: bool = True,
    ) -> Self:
        """Place agents in the space according to the specified coordinates. If some agents are already placed, raises a RuntimeWarning.

        Parameters
        ----------
        agents : IdsLike | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry]
            The agents to place in the space
        pos : SpaceCoordinate | SpaceCoordinates
            The coordinates for each agents. The length of the coordinates must match the number of agents.
        inplace : bool, optional
            Whether to perform the operation inplace, by default True

        Returns
        -------
        Self

        Raises
        ------
        RuntimeWarning
            If some agents are already placed in the space.
        ValueError
            - If some agents are not part of the model.
            - If agents is IdsLike and some agents are present multiple times.
        """
        obj = self._get_obj(inplace=inplace)
        return obj._place_or_move_agents(agents=agents, pos=pos, is_move=False)

    def random_agents(
        self,
        n: int,
    ) -> DataFrame:
        """Return a random sample of agents from the space.

        Parameters
        ----------
        n : int
            The number of agents to sample

        Returns
        -------
        DataFrame
            A DataFrame with the sampled agents
        """
        seed = self.random.integers(np.iinfo(np.int32).max)
        return self._df_sample(self._agents, n=n, seed=int(seed))

    def swap_agents(
        self,
        agents0: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        agents1: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        inplace: bool = True,
    ) -> Self:
        """Swap the positions of the agents in the space.

        agents0 and agents1 must have the same length and all agents must be placed in the space.

        Parameters
        ----------
        agents0 : IdsLike | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry]
            The first set of agents to swap
        agents1 : IdsLike | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry]
            The second set of agents to swap
        inplace : bool, optional
            Whether to perform the operation inplace, by default True

        Returns
        -------
        Self
        """
        # Normalize inputs to Series of ids for validation and operations
        ids0 = self._get_ids_srs(agents0)
        ids1 = self._get_ids_srs(agents1)
        if __debug__:
            if len(ids0) != len(ids1):
                raise ValueError("The two sets of agents must have the same length")
            if not self._df_contains(self._agents, "agent_id", ids0).all():
                raise ValueError("Some agents in agents0 are not in the space")
            if not self._df_contains(self._agents, "agent_id", ids1).all():
                raise ValueError("Some agents in agents1 are not in the space")
            if self._srs_contains(ids0, ids1).any():
                raise ValueError("Some agents are present in both agents0 and agents1")
        obj = self._get_obj(inplace)
        agents0_df = obj._df_get_masked_df(
            obj._agents, index_cols="agent_id", mask=ids0
        )
        agents1_df = obj._df_get_masked_df(
            obj._agents, index_cols="agent_id", mask=ids1
        )
        agents0_df = obj._df_set_index(agents0_df, "agent_id", ids1)
        agents1_df = obj._df_set_index(agents1_df, "agent_id", ids0)
        obj._agents = obj._df_combine_first(
            agents0_df, obj._agents, index_cols="agent_id"
        )
        obj._agents = obj._df_combine_first(
            agents1_df, obj._agents, index_cols="agent_id"
        )

        return obj

    @property
    @abstractmethod
    def neighborhood(self) -> AbstractNeighborhood:
        """Access neighborhood queries via a unified interface."""
        ...

    @abstractmethod
    def get_directions(
        self,
        pos0: SpaceCoordinate | SpaceCoordinates | None = None,
        pos1: SpaceCoordinate | SpaceCoordinates | None = None,
        agents0: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None = None,
        agents1: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None = None,
        normalize: bool = False,
    ) -> DataFrame:
        """Return the directions from pos0 to pos1 or agents0 and agents1.

        If the space is a Network, the direction is the shortest path between the two nodes.
        In all other cases, the direction is the direction vector between the two positions.
        Either positions (pos0, pos1) or agents (agents0, agents1) must be specified, not both and they must have the same length.

        Parameters
        ----------
        pos0 : SpaceCoordinate | SpaceCoordinates | None, optional
            The starting positions
        pos1 : SpaceCoordinate | SpaceCoordinates | None, optional
            The ending positions
        agents0 : IdsLike | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry] | None, optional
            The starting agents
        agents1 : IdsLike | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry] | None, optional
            The ending agents
        normalize : bool, optional
            Whether to normalize the vectors to unit norm. By default False

        Returns
        -------
        DataFrame
            A DataFrame where each row represents the direction from pos0 to pos1 or agents0 to agents1
        """
        ...

    @abstractmethod
    def get_distances(
        self,
        pos0: SpaceCoordinate | SpaceCoordinates | None = None,
        pos1: SpaceCoordinate | SpaceCoordinates | None = None,
        agents0: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None = None,
        agents1: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None = None,
    ) -> DataFrame:
        """Return the distances from pos0 to pos1 or agents0 and agents1.

        If the space is a Network, the distance is the number of nodes of the shortest path between the two nodes.
        In all other cases, the distance is Euclidean/l2/Frobenius norm.
        You should specify either positions (pos0, pos1) or agents (agents0, agents1), not both and they must have the same length.

        Parameters
        ----------
        pos0 : SpaceCoordinate | SpaceCoordinates | None, optional
            The starting positions
        pos1 : SpaceCoordinate | SpaceCoordinates | None, optional
            The ending positions
        agents0 : IdsLike | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry] | None, optional
            The starting agents
        agents1 : IdsLike | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry] | None, optional
            The ending agents

        Returns
        -------
        DataFrame
            A DataFrame where each row represents the distance from pos0 to pos1 or agents0 to agents1
        """
        return ...

    @abstractmethod
    def move_to_empty(
        self,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        inplace: bool = True,
    ) -> Self:
        """Move agents to empty cells/positions in the space (cells/positions where there isn't any single agent).

        Parameters
        ----------
        agents : IdsLike | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry]
            The agents to move to empty cells/positions
        inplace : bool, optional
            Whether to perform the operation inplace, by default True

        Returns
        -------
        Self
        """
        ...

    @abstractmethod
    def place_to_empty(
        self,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        inplace: bool = True,
    ) -> Self:
        """Place agents in empty cells/positions in the space (cells/positions where there isn't any single agent).

        Parameters
        ----------
        agents : IdsLike | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry]
            The agents to place in empty cells/positions
        inplace : bool, optional
            Whether to perform the operation inplace, by default True

        Returns
        -------
        Self
        """
        ...

    @abstractmethod
    def remove_agents(
        self,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        inplace: bool = True,
    ) -> Self:
        """Remove agents from the space.

        Does not remove the agents from the model.

        Parameters
        ----------
        agents : IdsLike | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry]
            The agents to remove from the space
        inplace : bool, optional
            Whether to perform the operation inplace, by default True

        Raises
        ------
        ValueError
            If some agents are not part of the model.

        Returns
        -------
        Self
        """
        return ...

    def _get_ids_srs(
        self,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
    ) -> Series:
        if isinstance(agents, Sized) and len(agents) == 0:
            return self._srs_constructor([], name="agent_id", dtype="uint64")
        if isinstance(agents, AbstractAgentSet):
            return self._srs_constructor(
                self._df_index(agents.df, "unique_id"),
                name="agent_id",
                dtype="uint64",
            )
        elif isinstance(agents, AbstractAgentSetRegistry):
            return self._srs_constructor(agents.ids, name="agent_id", dtype="uint64")
        elif isinstance(agents, Collection) and (
            isinstance(agents[0], AbstractAgentSet)
            or isinstance(agents[0], AbstractAgentSetRegistry)
        ):
            ids = []
            for a in agents:
                if isinstance(a, AbstractAgentSet):
                    ids.append(
                        self._srs_constructor(
                            self._df_index(a.df, "unique_id"),
                            name="agent_id",
                            dtype="uint64",
                        )
                    )
                elif isinstance(a, AbstractAgentSetRegistry):
                    ids.append(
                        self._srs_constructor(a.ids, name="agent_id", dtype="uint64")
                    )
            return self._df_concat(ids, ignore_index=True)
        elif isinstance(agents, int):
            return self._srs_constructor([agents], name="agent_id", dtype="uint64")
        else:  # IDsLike
            return self._srs_constructor(agents, name="agent_id", dtype="uint64")

    @abstractmethod
    def _place_or_move_agents(
        self,
        agents: IdsLike
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSetRegistry],
        pos: SpaceCoordinate | SpaceCoordinates,
        is_move: bool,
    ) -> Self:
        """Move or place agents.

        Only the runtime warning change.

        Parameters
        ----------
        agents : IdsLike | AbstractAgentSetRegistry | Collection[AbstractAgentSetRegistry]
            The agents to move/place
        pos : SpaceCoordinate | SpaceCoordinates
            The position to move/place agents to
        is_move : bool
            Whether the operation is "move" or "place"

        Returns
        -------
        Self
        """

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the Space.

        Returns
        -------
        str
        """
        ...

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the Space.

        Returns
        -------
        str
        """
        ...

    @property
    def agents(self) -> DataFrame:  # | GeoDataFrame:
        """Get the ids of the agents placed in the cell set, along with their coordinates or geometries.

        Returns
        -------
        DataFrame
        """
        return self._agents

    @property
    def model(self) -> mesa_frames.concrete.model.Model:
        """The model to which the space belongs.

        Returns
        -------
        'mesa_frames.concrete.model.Model'
        """
        return self._model

    @property
    def random(self) -> Generator:
        """The model's random number generator.

        Returns
        -------
        Generator
        """
        return self.model.random
