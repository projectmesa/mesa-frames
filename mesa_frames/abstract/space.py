"""
Abstract base classes for spatial components in mesa-frames.

This module defines the core abstractions for spatial structures in the mesa-frames
extension. It provides the foundation for implementing various types of spaces,
including discrete spaces and grids, using DataFrame-based approaches for improved
performance and scalability.

Classes:
    Space(CopyMixin, DataFrameMixin):
        An abstract base class that defines the common interface for all space
        classes in mesa-frames. It combines fast copying functionality with
        DataFrame operations.

    AbstractDiscreteSpace(Space):
        An abstract base class for discrete space implementations, such as grids
        and networks. It extends Space with methods specific to discrete spaces.

    AbstractGrid(AbstractDiscreteSpace):
        An abstract base class for grid-based spaces. It inherits from
        AbstractDiscreteSpace and adds grid-specific functionality.

These abstract classes are designed to be subclassed by concrete implementations
that use Polars library as their backend.
They provide a common interface and shared functionality across different types
of spatial structures in agent-based models.

Usage:
    These classes should not be instantiated directly. Instead, they should be
    subclassed to create concrete implementations:

    from mesa_frames.abstract.space import AbstractGrid

    class Grid(AbstractGrid):
        def __init__(self, model, dimensions, torus, capacity, neighborhood_type):
            super().__init__(model, dimensions, torus, capacity, neighborhood_type)
            # Implementation using polars DataFrame
            ...

        # Implement other abstract methods

Note:
    The abstract methods in these classes use Python's @abstractmethod decorator,
    ensuring that concrete subclasses must implement these methods.

Attributes and methods of each class are documented in their respective docstrings.
For more detailed information on each class, refer to their individual docstrings.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Collection, Sequence, Sized
from itertools import product
from typing import Any, Literal, Self, cast
from warnings import warn

import numpy as np
import polars as pl
from numpy.random import Generator

from mesa_frames.abstract.agentset import AbstractAgentSet
from mesa_frames.abstract.agentsetregistry import (
    AbstractAgentSetRegistry,
)
from mesa_frames.abstract.mixin import CopyMixin, DataFrameMixin
from mesa_frames.types_ import (
    ArrayLike,
    BoolSeries,
    DataFrame,
    DataFrameInput,
    DiscreteCoordinate,
    DiscreteCoordinates,
    DiscreteSpaceCapacity,
    GridCapacity,
    GridCoordinate,
    GridCoordinates,
    IdsLike,
    Infinity,
    Series,
    SpaceCoordinate,
    SpaceCoordinates,
)

ESPG = int


class Space(CopyMixin, DataFrameMixin):
    """The Space class is an abstract class that defines the interface for all space classes in mesa_frames."""

    _agents: DataFrame  # | GeoDataFrame  # Stores the agents placed in the space
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
    def get_neighbors(
        self,
        radius: int | float | Sequence[int] | Sequence[float] | ArrayLike,
        pos: SpaceCoordinate | SpaceCoordinates | None = None,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None = None,
        include_center: bool = False,
    ) -> DataFrame:
        """Get the neighboring agents from given positions or agents according to the specified radiuses.

        Either positions (pos0, pos1) or agents (agents0, agents1) must be specified, not both and they must have the same length.

        Parameters
        ----------
        radius : int | float | Sequence[int] | Sequence[float] | ArrayLike
            The radius(es) of the neighborhood
        pos : SpaceCoordinate | SpaceCoordinates | None, optional
            The coordinates of the cell to get the neighborhood from, by default None
        agents : IdsLike | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry] | None, optional
            The id of the agents to get the neighborhood from, by default None
        include_center : bool, optional
            If the center cells or agents should be included in the result, by default False

        Returns
        -------
        DataFrame
            A dataframe with neighboring agents.
            The columns with '_center' suffix represent the center agent/position.

        Raises
        ------
        ValueError
            If both pos and agent are None or if both pos and agent are not None.
        """
        ...

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
    def random_pos(
        self,
        n: int,
    ) -> DataFrame:
        """Return a random sample of positions from the space.

        Parameters
        ----------
        n : int
            The number of positions to sample

        Returns
        -------
        DataFrame
            A DataFrame with the sampled positions
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

    @property
    def seed(self) -> int | Sequence[int]:
        """Return the seed for the model's random number generator.

        Returns
        -------
        int | Sequence[int]
            The seed that initialized the model's random number generator.
        """
        return self.model.seed


class AbstractDiscreteSpace(Space):
    """The AbstractDiscreteSpace class is an abstract class that defines the interface for all discrete space classes (Grids and Networks) in mesa_frames."""

    _agents: DataFrame
    _capacity: int | None  # The maximum capacity for cells (default is infinite)
    _cells: DataFrame  # Stores the properties of the cells
    _cells_capacity: (
        DiscreteSpaceCapacity  # Storing the remaining capacity of the cells in the grid
    )

    def __init__(
        self,
        model: mesa_frames.concrete.model.Model,
        capacity: int | None = None,
    ):
        """Create a new AbstractDiscreteSpace.

        Parameters
        ----------
        model : mesa_frames.concrete.model.Model
            The model to which the space belongs
        capacity : int | None, optional
            The maximum capacity for cells (default is infinite), by default None
        """
        super().__init__(model)
        self._capacity = capacity

    def is_available(self, pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame:
        """Check whether the input positions are available (there exists at least one remaining spot in the cells).

        Parameters
        ----------
        pos : DiscreteCoordinate | DiscreteCoordinates
            The positions to check for

        Returns
        -------
        DataFrame
            A dataframe with positions and a boolean column "available"
        """
        return self._check_cells(pos, "available")

    def is_empty(self, pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame:
        """Check whether the input positions are empty (there isn't any single agent in the cells).

        Parameters
        ----------
        pos : DiscreteCoordinate | DiscreteCoordinates
            The positions to check for

        Returns
        -------
        DataFrame
            A dataframe with positions and a boolean column "empty"
        """
        return self._check_cells(pos, "empty")

    def is_full(self, pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame:
        """Check whether the input positions are full (there isn't any spot available in the cells).

        Parameters
        ----------
        pos : DiscreteCoordinate | DiscreteCoordinates
            The positions to check for

        Returns
        -------
        DataFrame
            A dataframe with positions and a boolean column "full"
        """
        return self._check_cells(pos, "full")

    def move_to_empty(
        self,
        agents: IdsLike
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSetRegistry]
        | AbstractAgentSet
        | Collection[AbstractAgentSet],
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        return obj._place_or_move_agents_to_cells(
            agents, cell_type="empty", is_move=True
        )

    def move_to_available(
        self,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        inplace: bool = True,
    ) -> Self:
        """Move agents to available cells/positions in the space (cells/positions where there is at least one spot available).

        Parameters
        ----------
        agents : IdsLike | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry]
            The agents to move to available cells/positions
        inplace : bool, optional
            Whether to perform the operation inplace, by default True

        Returns
        -------
        Self
        """
        obj = self._get_obj(inplace)

        return obj._place_or_move_agents_to_cells(
            agents, cell_type="available", is_move=True
        )

    def place_to_empty(
        self,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)

        return obj._place_or_move_agents_to_cells(
            agents, cell_type="empty", is_move=False
        )

    def place_to_available(
        self,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)
        return obj._place_or_move_agents_to_cells(
            agents, cell_type="available", is_move=False
        )

    def random_pos(self, n: int) -> DataFrame | pl.DataFrame:
        return self.sample_cells(n, cell_type="any", with_replacement=True)

    def sample_cells(
        self,
        n: int,
        cell_type: Literal["any", "empty", "available", "full"] = "any",
        with_replacement: bool = True,
        respect_capacity: bool = True,
    ) -> DataFrame:
        """Sample cells from the grid according to the specified cell_type.

        Parameters
        ----------
        n : int
            The number of cells to sample
        cell_type : Literal["any", "empty", "available", "full"], optional
            The type of cells to sample, by default "any"
        with_replacement : bool, optional
            If the sampling should be with replacement, by default True
        respect_capacity : bool, optional
            If the capacity of the cells should be respected in the sampling.
            This is only relevant if cell_type is "empty" or "available", by default True

        Returns
        -------
        DataFrame
            A DataFrame with the sampled cells
        """
        match cell_type:
            case "any":
                condition = self._any_cell_condition
            case "empty":
                condition = self._empty_cell_condition
            case "available":
                condition = self._available_cell_condition
            case "full":
                condition = self._full_cell_condition
        return self._sample_cells(
            n,
            with_replacement,
            condition=condition,
            respect_capacity=respect_capacity,
        )

    def set_cells(
        self,
        cells: DataFrame | DiscreteCoordinate | DiscreteCoordinates,
        properties: DataFrame | dict[str, Any] | None = None,
        inplace: bool = True,
    ) -> Self:
        """Set the properties of the specified cells.

        This method mirrors the functionality of mesa's PropertyLayer, but allows also to set properties only of specific cells.
        Either the cells DF must contain both the cells' coordinates and the properties
        or the cells' coordinates can be specified separately with the cells argument.
        If the Space is a Grid, the cell coordinates must be GridCoordinates.
        If the Space is a Network, the cell coordinates must be NetworkCoordinates.


        Parameters
        ----------
        cells : DataFrame | DiscreteCoordinate | DiscreteCoordinates
            The cells to set the properties for. It can contain the coordinates of the cells or both the coordinates and the properties.
        properties : DataFrame | dict[str, Any] | None, optional
            The properties of the cells, by default None if the cells argument contains the properties
        inplace : bool
            Whether to perform the operation inplace

        Returns
        -------
        Self
        """
        obj = self._get_obj(inplace)

        # Convert cells to DataFrame
        if isinstance(cells, DataFrame):
            cells_df = cells
        else:
            cells_df = obj._get_df_coords(cells)
        cells_df = obj._df_set_index(cells_df, index_name=obj._pos_col_names)

        if __debug__:
            if isinstance(cells_df, DataFrame) and any(
                k not in obj._df_column_names(cells_df) for k in obj._pos_col_names
            ):
                raise ValueError(
                    f"The cells DataFrame must have the columns {obj._pos_col_names}"
                )

        if properties:
            cells_df = obj._df_constructor(
                data=properties, index=self._df_index(cells_df, obj._pos_col_names)
            )

        if "capacity" in obj._df_column_names(cells_df):
            obj._cells_capacity = obj._update_capacity_cells(cells_df)

        obj._cells = obj._df_combine_first(
            cells_df, obj._cells, index_cols=obj._pos_col_names
        )
        return obj

    @abstractmethod
    def get_neighborhood(
        self,
        radius: int | float | Sequence[int] | Sequence[float] | ArrayLike,
        pos: DiscreteCoordinate | DiscreteCoordinates | None = None,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry] = None,
        include_center: bool = False,
    ) -> DataFrame:
        """Get the neighborhood cells from the given positions (pos) or agents according to the specified radiuses.

        Either positions (pos) or agents must be specified, not both.

        Parameters
        ----------
        radius : int | float | Sequence[int] | Sequence[float] | ArrayLike
            The radius(es) of the neighborhoods
        pos : DiscreteCoordinate | DiscreteCoordinates | None, optional
            The coordinates of the cell(s) to get the neighborhood from
        agents : IdsLike | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry], optional
            The agent(s) to get the neighborhood from
        include_center : bool, optional
            If the cell in the center of the neighborhood should be included in the result, by default False

        Returns
        -------
        DataFrame
            A dataframe where
             - Columns are called according to the coordinates of the space(['dim_0', 'dim_1', ...] in Grids, ['node_id', 'edge_id'] in Networks)
             - Rows represent the coordinates of a neighboring cells
        """
        ...

    @abstractmethod
    def get_cells(
        self, coords: DiscreteCoordinate | DiscreteCoordinates | None = None
    ) -> DataFrame:
        """Retrieve a dataframe of specified cells with their properties and agents.

        Parameters
        ----------
        coords : DiscreteCoordinate | DiscreteCoordinates | None, optional
            The cells to retrieve. Default is None (all cells retrieved)

        Returns
        -------
        DataFrame
            A DataFrame with the properties of the cells and the agents placed in them.
        """
        ...

    # We define the cell conditions here, because ruff does not allow lambda functions

    def _any_cell_condition(
        self, cap: DiscreteSpaceCapacity
    ) -> BoolSeries | np.ndarray:
        return self._cells_capacity

    @abstractmethod
    def _empty_cell_condition(
        self, cap: DiscreteSpaceCapacity
    ) -> BoolSeries | np.ndarray: ...

    def _available_cell_condition(
        self, cap: DiscreteSpaceCapacity
    ) -> BoolSeries | np.ndarray:
        return cap > 0

    def _full_cell_condition(
        self, cap: DiscreteSpaceCapacity
    ) -> BoolSeries | np.ndarray:
        return cap == 0

    def _check_cells(
        self,
        pos: DiscreteCoordinate | DiscreteCoordinates,
        state: Literal["empty", "full", "available"],
    ) -> DataFrame:
        """
        Check the state of cells at given positions.

        Parameters
        ----------
        pos : DiscreteCoordinate | DiscreteCoordinates
            The positions to check
        state : Literal["empty", "full", "available"]
            The state to check for ("empty", "full", or "available")

        Returns
        -------
        DataFrame
            A dataframe with positions and a boolean column indicating the state
        """
        pos_df = self._get_df_coords(pos)

        if state == "empty":
            mask = self.empty_cells
        elif state == "full":
            mask = self.full_cells
        elif state == "available":
            mask = self.available_cells

        return self._df_with_columns(
            original_df=pos_df,
            data=self._df_get_bool_mask(
                pos_df,
                index_cols=self._pos_col_names,
                mask=mask,
            ),
            new_columns=state,
        )

    def _place_or_move_agents_to_cells(
        self,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        cell_type: Literal["any", "empty", "available"],
        is_move: bool,
    ) -> Self:
        # Get Ids of agents
        agents = self._get_ids_srs(agents)

        if __debug__:
            # Check ids presence in model using public API
            b_contained = agents.is_in(self.model.sets.ids)
            if (isinstance(b_contained, Series) and not b_contained.all()) or (
                isinstance(b_contained, bool) and not b_contained
            ):
                raise ValueError("Some agents are not in the model")

        # Get cells of specified type
        cells = self.sample_cells(len(agents), cell_type=cell_type)

        # Place agents
        if is_move:
            self.move_agents(agents, cells)
        else:
            self.place_agents(agents, cells)
        return self

    @abstractmethod
    def _get_df_coords(
        self,
        pos: DiscreteCoordinate | DiscreteCoordinates | None = None,
        agents: IdsLike
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSetRegistry]
        | None = None,
    ) -> DataFrame:
        """Get the DataFrame of coordinates from the specified positions or agents.

        Parameters
        ----------
        pos : DiscreteCoordinate | DiscreteCoordinates | None, optional
            The positions to get the DataFrame from, by default None
        agents : IdsLike | AbstractAgentSetRegistry | Collection[AbstractAgentSetRegistry] | None, optional
            The agents to get the DataFrame from, by default None

        Returns
        -------
        DataFrame
            A dataframe where the columns are the coordinates col_names and the rows are the positions

        Raises
        ------
        ValueError
            If neither pos or agents are specified
        """
        ...

    @abstractmethod
    def _sample_cells(
        self,
        n: int | None,
        with_replacement: bool,
        condition: Callable[[DiscreteSpaceCapacity], BoolSeries | np.ndarray],
        respect_capacity: bool = True,
    ) -> DataFrame:
        """Sample cells from the grid according to a condition on the capacity.

        Parameters
        ----------
        n : int | None
            The number of cells to sample. If None, samples the maximum available.
        with_replacement : bool
            If the sampling should be with replacement
        condition : Callable[[DiscreteSpaceCapacity], BoolSeries | np.ndarray]
            The condition to apply on the capacity
        respect_capacity : bool, optional
            If the capacity should be respected in the sampling.
            This is only relevant if cell_type is "empty" or "available", by default True

        Returns
        -------
        DataFrame
        """
        ...

    @abstractmethod
    def _update_capacity_cells(self, cells: DataFrame) -> DiscreteSpaceCapacity:
        """Update the cells' capacity after setting new properties.

        Parameters
        ----------
        cells : DataFrame
            A DF with the cells to update the capacity and the 'capacity' column

        Returns
        -------
        DiscreteSpaceCapacity
            The updated cells' capacity
        """
        ...

    @abstractmethod
    def _update_capacity_agents(
        self, agents: DataFrame | Series, operation: Literal["movement", "removal"]
    ) -> DiscreteSpaceCapacity:
        """Update the cells' capacity after moving agents.

        Parameters
        ----------
        agents : DataFrame | Series
            The moved agents with their new positions
        operation : Literal["movement", "removal"]
            The operation that was performed on the agents

        Returns
        -------
        DiscreteSpaceCapacity
            The updated cells' capacity
        """
        ...

    def __getitem__(self, cells: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame:
        """Get the properties and agents of the specified cells.

        Parameters
        ----------
        cells : DiscreteCoordinate | DiscreteCoordinates
            The cells to get the properties for

        Returns
        -------
        DataFrame
            A DataFrame with the properties and agents of the cells
        """
        return self.get_cells(cells)

    def __getattr__(self, key: str) -> DataFrame:
        """Get the properties of the cells.

        Parameters
        ----------
        key : str
            The property to get

        Returns
        -------
        DataFrame
            A DataFrame with the properties of the cells
        """
        # Fallback, if key (property) is not found in the object,
        # then it must mean that it's in the _cells dataframe
        return self._cells[key]

    def __setitem__(
        self, cells: DiscreteCoordinates, properties: DataFrame | DataFrameInput
    ):
        """Set the properties of the specified cells.

        Parameters
        ----------
        cells : DiscreteCoordinates
            The cells to set the properties for
        properties : DataFrame | DataFrameInput
            The properties to set
        """
        self.set_cells(cells=cells, properties=properties, inplace=True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}\n{str(self.cells)}"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}\n{str(self.cells)}"

    @property
    def cells(self) -> DataFrame:
        """
        Obtain the properties and agents of the cells in the grid.

        Returns
        -------
        DataFrame
            A Dataframe with all cells, their properties and their agents
        """
        return self.get_cells()

    @cells.setter
    def cells(self, df: DataFrame):
        return self.set_cells(df, inplace=True)

    @property
    def empty_cells(self) -> DataFrame:
        """Get the empty cells (cells without any agent) in the grid.

        Returns
        -------
        DataFrame
            A DataFrame with the empty cells
        """
        return self._sample_cells(
            None, with_replacement=False, condition=self._empty_cell_condition
        )

    @property
    def available_cells(self) -> DataFrame:
        """Get the available cells (cells with at least one spot available) in the grid.

        Returns
        -------
        DataFrame
            A DataFrame with the available cells
        """
        return self._sample_cells(
            None, with_replacement=False, condition=self._available_cell_condition
        )

    @property
    def full_cells(self) -> DataFrame:
        """Get the full cells (cells without any spot available) in the grid.

        Returns
        -------
        DataFrame
            A DataFrame with the full cells
        """
        return self._sample_cells(
            None, with_replacement=False, condition=self._full_cell_condition
        )

    @property
    @abstractmethod
    def remaining_capacity(self) -> int | Infinity:
        """The remaining capacity of the cells in the grid.

        Returns
        -------
        int | Infinity
            None if the capacity is infinite, otherwise the remaining capacity
        """
        ...


class AbstractGrid(AbstractDiscreteSpace):
    """The AbstractGrid class is an abstract class that defines the interface for all grid classes in mesa-frames.

    Inherits from AbstractDiscreteSpace.

    Warning
    -------
    For rectangular grids:
    In this implementation, [0, ..., 0] is the bottom-left corner and
    [dimensions[0]-1, ..., dimensions[n-1]-1] is the top-right corner, consistent with
    Cartesian coordinates and Matplotlib/Seaborn plot outputs.
    The convention is different from `np.genfromtxt`_ and its use in the
    `mesa-examples Sugarscape model`_, where [0, ..., 0] is the top-left corner
    and [dimensions[0]-1, ..., dimensions[n-1]-1] is the bottom-right corner.

    For hexagonal grids:
    The coordinates are ordered according to the axial coordinate system.
    In this system, the hexagonal grid uses two axes (q and r) at 60 degrees to each other.
    The q-axis points to the right, and the r-axis points up and to the right.
    The [0, 0] coordinate is at the bottom-left corner of the grid.

    .. _np.genfromtxt: https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html
    .. _mesa-examples Sugarscape model: https://github.com/mesa/mesa-examples/blob/e137a60e4e2f2546901bec497e79c4a7b0cc69bb/examples/sugarscape_g1mt/sugarscape_g1mt/model.py#L93-L94
    """

    _cells_capacity: (
        GridCapacity  # Storing the remaining capacity of the cells in the grid
    )
    _neighborhood_type: Literal[
        "moore", "von_neumann", "hexagonal"
    ]  # The type of neighborhood to consider
    _offsets: DataFrame  # The offsets to compute the neighborhood of a cell
    _torus: bool  # If the grid is a torus

    def __init__(
        self,
        model: mesa_frames.concrete.model.Model,
        dimensions: Sequence[int],
        torus: bool = False,
        capacity: int | None = None,
        neighborhood_type: str = "moore",
    ):
        """Create a new AbstractGrid.

        Parameters
        ----------
        model : mesa_frames.concrete.model.Model
            The model to which the space belongs
        dimensions : Sequence[int]
            The dimensions of the grid
        torus : bool, optional
            If the grid is a torus, by default False
        capacity : int | None, optional
            The maximum capacity for cells (default is infinite), by default None
        neighborhood_type : str, optional
            The type of neighborhood to consider, by default "moore"
        """
        super().__init__(model, capacity)
        self._dimensions = dimensions
        self._torus = torus
        self._pos_col_names = [f"dim_{k}" for k in range(len(dimensions))]
        self._center_col_names = [x + "_center" for x in self._pos_col_names]
        self._agents = self._df_constructor(
            columns=["agent_id"] + self._pos_col_names,
            index_cols="agent_id",
            dtypes={"agent_id": "uint64"} | {col: int for col in self._pos_col_names},
        )

        cells_df_dtypes = {col: int for col in self._pos_col_names}
        cells_df_dtypes.update(
            {"capacity": float}  # Capacity can be float if we want to represent np.nan
        )
        self._cells = self._df_constructor(
            columns=self._pos_col_names + ["capacity"],
            index_cols=self._pos_col_names,
            dtypes=cells_df_dtypes,
        )
        self._offsets = self._compute_offsets(neighborhood_type)
        self._cells_capacity = self._generate_empty_grid(dimensions, capacity)
        self._neighborhood_type = neighborhood_type

    def get_directions(
        self,
        pos0: GridCoordinate | GridCoordinates | None = None,
        pos1: GridCoordinate | GridCoordinates | None = None,
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
        result = self._calculate_differences(pos0, pos1, agents0, agents1)
        if normalize:
            result = self._df_div(result, other=self._df_norm(result))
        return result

    def get_distances(
        self,
        pos0: GridCoordinate | GridCoordinates | None = None,
        pos1: GridCoordinate | GridCoordinates | None = None,
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
        result = self._calculate_differences(pos0, pos1, agents0, agents1)
        return self._df_norm(result, "distance", True)

    def get_neighbors(
        self,
        radius: int | Sequence[int],
        pos: GridCoordinate | GridCoordinates | None = None,
        agents: IdsLike
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSetRegistry]
        | None = None,
        include_center: bool = False,
    ) -> DataFrame:
        neighborhood_df = self.get_neighborhood(
            radius=radius, pos=pos, agents=agents, include_center=include_center
        )
        return self._df_get_masked_df(
            df=self._agents,
            index_cols=self._pos_col_names,
            mask=neighborhood_df,
        )

    def get_neighborhood(
        self,
        radius: int | Sequence[int] | ArrayLike,
        pos: DiscreteCoordinate | DiscreteCoordinates | None = None,
        agents: IdsLike
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSetRegistry]
        | None = None,
        include_center: bool = False,
    ) -> DataFrame:
        pos_df = self._get_df_coords(pos, agents)

        if __debug__:
            if isinstance(radius, ArrayLike):
                if len(radius) != len(pos_df):
                    raise ValueError(
                        "The length of the radius sequence must be equal to the number of positions/agents"
                    )

        ## Create all possible neighbors by multiplying offsets by the radius and adding original pos

        # If radius is a sequence, get the maximum radius (we will drop unnecessary neighbors later, time-efficient but memory-inefficient)
        if isinstance(radius, ArrayLike):
            radius_srs = self._srs_constructor(radius, name="radius")
            radius_df = self._srs_to_df(radius_srs)
            max_radius = radius_srs.max()
        else:
            max_radius = radius

        range_df = self._srs_to_df(
            self._srs_range(name="radius", start=1, end=max_radius + 1)
        )

        neighbors_df = self._df_join(
            self._offsets,
            range_df,
            how="cross",
        )

        neighbors_df = self._df_with_columns(
            neighbors_df,
            data=self._df_mul(
                neighbors_df[self._pos_col_names], neighbors_df["radius"]
            ),
            new_columns=self._pos_col_names,
        )

        if self.neighborhood_type == "hexagonal":
            # We need to add in-between cells for hexagonal grids
            # In-between offsets (for every radius k>=2, we need k-1 in-between cells)
            in_between_cols = ["in_between_dim_0", "in_between_dim_1"]
            radius_srs = self._srs_constructor(
                np.repeat(np.arange(1, max_radius + 1), np.arange(0, max_radius)),
                name="radius",
            )
            radius_df = self._srs_to_df(radius_srs)
            radius_df = self._df_with_columns(
                radius_df,
                self._df_groupby_cumcount(radius_df, "radius", name="offset"),
                new_columns="offset",
            )

            in_between_df = self._df_join(
                self._in_between_offsets,
                radius_df,
                how="cross",
            )
            # We multiply the radius to get the directional cells
            in_between_df = self._df_with_columns(
                in_between_df,
                data=self._df_mul(
                    in_between_df[self._pos_col_names], in_between_df["radius"]
                ),
                new_columns=self._pos_col_names,
            )
            # We multiply the offset (from the directional cells) to get the in-between offset for each radius
            in_between_df = self._df_with_columns(
                in_between_df,
                data=self._df_mul(
                    in_between_df[in_between_cols], in_between_df["offset"]
                ),
                new_columns=in_between_cols,
            )
            # We add the in-between offset to the directional cells to obtain the in-between cells
            in_between_df = self._df_with_columns(
                in_between_df,
                data=self._df_add(
                    in_between_df[self._pos_col_names],
                    self._df_rename_columns(
                        in_between_df[in_between_cols],
                        in_between_cols,
                        self._pos_col_names,
                    ),
                ),
                new_columns=self._pos_col_names,
            )

            in_between_df = self._df_drop_columns(
                in_between_df, in_between_cols + ["offset"]
            )

            neighbors_df = self._df_concat(
                [neighbors_df, in_between_df], how="vertical"
            )
            radius_df = self._df_drop_columns(radius_df, "offset")

        neighbors_df = self._df_join(
            neighbors_df, pos_df, how="cross", suffix="_center"
        )

        center_df = self._df_rename_columns(
            neighbors_df[self._center_col_names],
            self._center_col_names,
            self._pos_col_names,
        )  # We rename the columns to the original names for the addition

        neighbors_df = self._df_with_columns(
            original_df=neighbors_df,
            new_columns=self._pos_col_names,
            data=self._df_add(
                neighbors_df[self._pos_col_names],
                center_df,
            ),
        )

        # If radius is a sequence, filter unnecessary neighbors
        if isinstance(radius, ArrayLike):
            radius_df = self._df_rename_columns(
                self._df_concat([pos_df, radius_df], how="horizontal"),
                self._pos_col_names + ["radius"],
                self._center_col_names + ["max_radius"],
            )

            neighbors_df = self._df_join(
                neighbors_df,
                radius_df,
                on=self._center_col_names,
            )
            neighbors_df = self._df_get_masked_df(
                neighbors_df, mask=neighbors_df["radius"] <= neighbors_df["max_radius"]
            )
            neighbors_df = self._df_drop_columns(neighbors_df, "max_radius")

        # If torus, "normalize" (take modulo) for out-of-bounds cells
        if self._torus:
            neighbors_df = self._df_with_columns(
                neighbors_df,
                data=self.torus_adj(neighbors_df[self._pos_col_names]),
                new_columns=self._pos_col_names,
            )
            # Remove duplicates
            neighbors_df = self._df_drop_duplicates(neighbors_df, self._pos_col_names)

        # Filter out-of-bound neighbors
        mask = self._df_all(
            self._df_and(
                self._df_lt(
                    neighbors_df[self._pos_col_names], self._dimensions, axis="columns"
                ),
                neighbors_df >= 0,
            )
        )
        neighbors_df = self._df_get_masked_df(neighbors_df, mask=mask)

        if include_center:
            center_df = self._df_rename_columns(
                pos_df, self._pos_col_names, self._center_col_names
            )
            pos_df = self._df_with_columns(
                pos_df,
                data=0,
                new_columns="radius",
            )
            pos_df = self._df_concat([pos_df, center_df], how="horizontal")

            neighbors_df = self._df_concat(
                [pos_df, neighbors_df], how="vertical", ignore_index=True
            )

        return neighbors_df

    def get_cells(
        self, coords: GridCoordinate | GridCoordinates | None = None
    ) -> DataFrame:
        # TODO : Consider whether not outputting the agents at all (fastest),
        # outputting a single agent per cell (current)
        # or outputting all agents per cell in a imploded list (slowest, https://stackoverflow.com/a/66018377)
        if not coords:
            cells_df = self._cells
        else:
            coords_df = self._get_df_coords(pos=coords)
            cells_df = self._df_get_masked_df(
                df=self._cells, index_cols=self._pos_col_names, mask=coords_df
            )
        return self._df_join(
            left=cells_df,
            right=self._agents,
            index_cols=self._pos_col_names,
            on=self._pos_col_names,
        )

    def out_of_bounds(self, pos: GridCoordinate | GridCoordinates) -> DataFrame:
        """Check if a position is out of bounds in a non-toroidal grid.

        Parameters
        ----------
        pos : GridCoordinate | GridCoordinates
            The position to check

        Returns
        -------
        DataFrame
            A DataFrame with the coordinates and an 'out_of_bounds' containing boolean values.

        Raises
        ------
        ValueError
            If the grid is a torus
        """
        if self.torus:
            raise ValueError("This method is only valid for non-torus grids")
        pos_df = self._get_df_coords(pos, check_bounds=False)
        out_of_bounds = self._df_all(
            self._df_or(
                pos_df < 0,
                self._df_ge(
                    pos_df,
                    self._dimensions,
                    axis="columns",
                    index_cols=self._pos_col_names,
                ),
            ),
            name="out_of_bounds",
        )
        return self._df_concat(
            objs=[pos_df, self._srs_to_df(out_of_bounds)], how="horizontal"
        )

    def remove_agents(
        self,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)

        agents = obj._get_ids_srs(agents)

        if __debug__:
            # Check ids presence in model via public ids
            b_contained = agents.is_in(obj.model.sets.ids)
            if (isinstance(b_contained, Series) and not b_contained.all()) or (
                isinstance(b_contained, bool) and not b_contained
            ):
                raise ValueError("Some agents are not in the model")

        # Remove agents
        obj._cells_capacity = obj._update_capacity_agents(agents, operation="removal")

        obj._agents = obj._df_remove(obj._agents, mask=agents, index_cols="agent_id")

        return obj

    def torus_adj(self, pos: GridCoordinate | GridCoordinates) -> DataFrame:
        """Get the toroidal adjusted coordinates of a position.

        Parameters
        ----------
        pos : GridCoordinate | GridCoordinates
            The coordinates to adjust

        Returns
        -------
        DataFrame
            The adjusted coordinates
        """
        df_coords = self._get_df_coords(pos)
        df_coords = self._df_mod(df_coords, self._dimensions, axis="columns")
        return df_coords

    def _calculate_differences(
        self,
        pos0: GridCoordinate | GridCoordinates | None,
        pos1: GridCoordinate | GridCoordinates | None,
        agents0: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None,
        agents1: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None,
    ) -> DataFrame:
        """Calculate the differences between two positions or agents.

        Parameters
        ----------
        pos0 : GridCoordinate | GridCoordinates | None
            The starting positions
        pos1 : GridCoordinate | GridCoordinates | None
            The ending positions
        agents0 : IdsLike | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry] | None
            The starting agents
        agents1 : IdsLike | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry] | None
            The ending agents

        Returns
        -------
        DataFrame

        Raises
        ------
        ValueError
            If objects do not have the same length
        """
        pos0_df = self._get_df_coords(pos0, agents0)
        pos1_df = self._get_df_coords(pos1, agents1)
        if __debug__ and len(pos0_df) != len(pos1_df):
            raise ValueError("objects must have the same length")
        return pos1_df - pos0_df

    def _compute_offsets(self, neighborhood_type: str) -> DataFrame:
        """Generate offsets for the neighborhood.

        Parameters
        ----------
        neighborhood_type : str
            The type of neighborhood to consider

        Returns
        -------
        DataFrame
            A DataFrame with the offsets

        Raises
        ------
        ValueError
            If the neighborhood type is invalid
        ValueError
            If the grid has more than 2 dimensions and the neighborhood type is 'hexagonal'
        """
        if neighborhood_type == "moore":
            ranges = [range(-1, 2) for _ in self._dimensions]
            directions = [d for d in product(*ranges) if any(d)]
        elif neighborhood_type == "von_neumann":
            ranges = [range(-1, 2) for _ in self._dimensions]
            directions = [
                d for d in product(*ranges) if sum(map(abs, d)) <= 1 and any(d)
            ]
        elif neighborhood_type == "hexagonal":
            if __debug__ and len(self._dimensions) > 2:
                raise ValueError(
                    "Hexagonal neighborhood is only valid for 2-dimensional grids"
                )
            directions = [
                (1, 0),  # East
                (1, -1),  # South-West
                (0, -1),  # South-East
                (-1, 0),  # West
                (-1, 1),  # North-West
                (0, 1),  # North-East
            ]
            in_between = [
                (-1, -1),  # East -> South-East
                (0, 1),  # South-West -> West
                (-1, 0),  # South-East -> South-West
                (1, 1),  # West -> North-West
                (1, 0),  # North-West -> North-East
                (0, -1),  # North-East -> East
            ]
            df = self._df_constructor(data=directions, columns=self._pos_col_names)
            self._in_between_offsets = self._df_with_columns(
                df,
                data=in_between,
                new_columns=["in_between_dim_0", "in_between_dim_1"],
            )
            return df
        else:
            raise ValueError("Invalid neighborhood type specified")
        return self._df_constructor(data=directions, columns=self._pos_col_names)

    def _get_df_coords(
        self,
        pos: GridCoordinate | GridCoordinates | None = None,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None = None,
        check_bounds: bool = True,
    ) -> DataFrame:
        """Get the DataFrame of coordinates from the specified positions or agents.

        Parameters
        ----------
        pos : GridCoordinate | GridCoordinates | None, optional
            The positions to get the DataFrame from, by default None
        agents : IdsLike | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry] | None, optional
            The agents to get the DataFrame from, by default None
        check_bounds: bool, optional
            If the positions should be checked for out-of-bounds in non-toroidal grids, by default True

        Returns
        -------
        DataFrame
            A dataframe where the columns are "dim_0, dim_1, ..." and the rows are the coordinates

        Raises
        ------
        ValueError
            If neither pos or agents are specified
        """
        if __debug__:
            if pos is None and agents is None:
                raise ValueError("Neither pos or agents are specified")
            elif pos is not None and agents is not None:
                raise ValueError("Both pos and agents are specified")
            # If the grid is non-toroidal, we have to check whether any position is out of bounds
            if not self.torus and pos is not None and check_bounds:
                pos = self.out_of_bounds(pos)
                if pos["out_of_bounds"].any():
                    raise ValueError(
                        "If the grid is non-toroidal, every position must be in-bound"
                    )
            if agents is not None:
                agents = self._get_ids_srs(agents)
                # Check ids presence in model
                b_contained = agents.is_in(self.model.sets.ids)
                if (isinstance(b_contained, Series) and not b_contained.all()) or (
                    isinstance(b_contained, bool) and not b_contained
                ):
                    raise ValueError("Some agents are not present in the model")

                # Check ids presence in the grid
                b_contained = self._df_contains(self._agents, "agent_id", agents)
                if (isinstance(b_contained, Series) and not b_contained.all()) or (
                    isinstance(b_contained, bool) and not b_contained
                ):
                    raise ValueError("Some agents are not placed in the grid")
                # Check ids are unique
                agents = pl.Series(agents)
                if agents.n_unique() != len(agents):
                    raise ValueError("Some agents are present multiple times")
        if agents is not None:
            df = self._df_get_masked_df(
                self._agents, index_cols="agent_id", mask=agents
            )
            df = self._df_reindex(df, agents, "agent_id")
            return self._df_reset_index(df, index_cols="agent_id", drop=True)
        if isinstance(pos, DataFrame):
            return self._df_reset_index(pos[self._pos_col_names], drop=True)
        elif (
            isinstance(pos, Collection)
            and isinstance(pos[0], Collection)
            and (len(pos[0]) == len(self._dimensions))
        ):  # We only test the first coordinate for performance
            # This means that we have a collection of coordinates
            return self._df_constructor(
                data=pos,
                columns=self._pos_col_names,
                dtypes={col: int for col in self._pos_col_names},
            )
        elif isinstance(pos, ArrayLike) and len(pos) == len(self._dimensions):
            # This means that the sequence is already a sequence where each element is the
            # sequence of coordinates for dimension i
            for i, c in enumerate(pos):
                if isinstance(c, slice):
                    start = c.start if c.start is not None else 0
                    step = c.step if c.step is not None else 1
                    stop = c.stop if c.stop is not None else self._dimensions[i]
                    pos[i] = self._srs_range(start=start, stop=stop, step=step)
            return self._df_constructor(
                data=[pos],
                columns=self._pos_col_names,
                dtypes={col: int for col in self._pos_col_names},
            )
        elif isinstance(pos, int) and len(self._dimensions) == 1:
            return self._df_constructor(
                data=[pos],
                columns=self._pos_col_names,
                dtypes={col: int for col in self._pos_col_names},
            )
        else:
            raise ValueError("Invalid coordinates")

    def _place_or_move_agents(
        self,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        pos: GridCoordinate | GridCoordinates,
        is_move: bool,
    ) -> Self:
        agents = self._get_ids_srs(agents)

        if __debug__:
            # Warn if agents are already placed
            if is_move:
                if not self._df_contains(self._agents, "agent_id", agents).all():
                    warn("Some agents are not present in the grid", RuntimeWarning)
            else:  # is "place"
                if self._df_contains(self._agents, "agent_id", agents).any():
                    warn("Some agents are already present in the grid", RuntimeWarning)

            # Check if agents are present in the model using the public ids
            b_contained = agents.is_in(self.model.sets.ids)
            if (isinstance(b_contained, Series) and not b_contained.all()) or (
                isinstance(b_contained, bool) and not b_contained
            ):
                raise ValueError("Some agents are not present in the model")

            # Check if there is enough capacity
            if self._capacity:
                # If len(agents) > remaining_capacity + len(agents that will move)
                if len(agents) > self.remaining_capacity + len(
                    self._df_get_masked_df(
                        self._agents,
                        index_cols="agent_id",
                        mask=agents,
                    )
                ):
                    raise ValueError("Not enough capacity in the space for all agents")

        # Place or move agents (checking that capacity is respected)
        pos_df = self._get_df_coords(pos)
        agents_df = self._srs_to_df(agents)

        if __debug__:
            if len(agents_df) != len(pos_df):
                raise ValueError("The number of agents and positions must be equal")

        new_df = self._df_concat(
            [agents_df, pos_df], how="horizontal", index_cols="agent_id"
        )
        self._cells_capacity = self._update_capacity_agents(
            new_df, operation="movement"
        )
        self._agents = self._df_combine_first(
            new_df, self._agents, index_cols="agent_id"
        )
        return self

    @abstractmethod
    def _generate_empty_grid(
        self, dimensions: Sequence[int], capacity: int | None
    ) -> GridCapacity:
        """Generate an empty grid with the specified dimensions and capacity.

        Parameters
        ----------
        dimensions : Sequence[int]
            The dimensions of the grid
        capacity : int | None
            The capacity of the grid

        Returns
        -------
        GridCapacity
        """
        ...

    @property
    def dimensions(self) -> Sequence[int]:
        """The dimensions of the grid.

        They are set uniquely at the creation of the grid.

        Returns
        -------
        Sequence[int]
            The dimensions of the grid
        """
        return self._dimensions

    @property
    def neighborhood_type(self) -> Literal["moore", "von_neumann", "hexagonal"]:
        """The type of neighborhood to consider (moore, von_neumann, hexagonal).

        It is set uniquely at the creation of the grid.

        Returns
        -------
        Literal['moore', 'von_neumann', 'hexagonal']
        """
        return self._neighborhood_type

    @property
    def torus(self) -> bool:
        """If the grid is a torus (wraps around at the edges).

        Can be set uniquely at the creation of the grid.

        Returns
        -------
        bool
            Whether the grid is a torus
        """
        return self._torus
