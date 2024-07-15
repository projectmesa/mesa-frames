"""
Mesa Frames Space Module
=================

Objects used to add a spatial component to a model.

Grid: base grid, which creates a rectangular grid.
SingleGrid: extension to Grid which strictly enforces one agent per cell.
MultiGrid: extension to Grid where each cell can contain a set of agents.
HexGrid: extension to Grid to handle hexagonal neighbors.
ContinuousSpace: a two-dimensional space where each agent has an arbitrary
                 position of `float`'s.
NetworkGrid: a network where each node contains zero or more agents.
"""

from mesa_frames.abstract.agents import AgentContainer
from mesa_frames.types_ import IdsLike, PositionsLike


class SpaceDF:
    def _check_empty_pos(pos: PositionsLike) -> bool:
        """Check if the given positions are empty.

        Parameters
        ----------
        pos : DataFrame | tuple[Series, Series] | Series
            Input positions to check.

        Returns
        -------
        Series[bool]
            Whether
        """


class SingleGrid(SpaceDF):
    """Rectangular grid where each cell contains exactly at most one agent.

    Grid cells are indexed by [x, y], where [0, 0] is assumed to be the
    bottom-left and [width-1, height-1] is the top-right. If a grid is
    toroidal, the top and bottom, and left and right, edges wrap to each other.

    This class provides a property `empties` that returns a set of coordinates
    for all empty cells in the grid. It is automatically updated whenever
    agents are added or removed from the grid. The `empties` property should be
    used for efficient access to current empty cells rather than manually
    iterating over the grid to check for emptiness.

    """

    def place_agents(self, agents: IdsLike | AgentContainer, pos: PositionsLike):
        """Place agents on the grid at the coordinates specified in pos.
        NOTE: The cells must be empty.


        Parameters
        ----------
        agents : IdsLike | AgentContainer

        pos : DataFrame | tuple[Series, Series]
            _description_
        """

    def _check_empty_pos(pos: PositionsLike) -> bool:
        """Check if the given positions are empty.

        Parameters
        ----------
        pos : DataFrame | tuple[Series, Series]
            Input positions to check.

        Returns
        -------
        bool
            _description_
        """

"""
Mesa Frames Space Module
=================

Objects used to add a spatial component to a model.

"""

from abc import abstractmethod
from functools import lru_cache
from itertools import product
from typing import cast
from warnings import warn

import geopandas as gpd
import networkx as nx
import numpy as np

# if TYPE_CHECKING:
import pandas as pd
import polars as pl
import shapely as shp
from numpy.random import Generator
from pyproj import CRS
from typing_extensions import (
    Any,
    Callable,
    Collection,
    Iterable,
    Iterator,
    Self,
    Sequence,
)

from mesa_frames.abstract.agents import AgentContainer, AgentSetDF
from mesa_frames.abstract.mixin import CopyMixin, DataFrameMixin
from mesa_frames.concrete.model import ModelDF
from mesa_frames.concrete.pandas.mixin import PandasMixin
from mesa_frames.concrete.polars.mixin import PolarsMixin
from mesa_frames.types_ import (
    DataFrame,
    DiscreteCoordinate,
    DiscreteCoordinates,
    DiscreteSpaceCapacity,
    GeoDataFrame,
    GridCapacity,
    GridCoordinate,
    GridCoordinates,
    SpaceCoordinate,
    SpaceCoordinates,
)

ESPG = int


class SpaceDF(CopyMixin, DataFrameMixin):
    _model: ModelDF
    _agents: DataFrame | GeoDataFrame

    def __init__(self, model: ModelDF) -> None:
        """Create a new CellSet object.

        Parameters
        ----------
        model : ModelDF

        Returns
        -------
        None
        """
        self._model = model

    def iter_neighbors(
        self,
        radius: int | float | Sequence[int] | Sequence[float],
        pos: SpaceCoordinate | SpaceCoordinates | None = None,
        agents: int
        | Collection[int]
        | AgentContainer
        | Collection[AgentContainer]
        | None = None,
        include_center: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Return an iterator over the neighboring agents from the given positions or agents according to specified radiuses.
        Either the positions or the agents must be specified (not both).

        Parameters
        ----------
        radius : int | float
            The radius of the neighborhood
        pos : SpaceCoordinate | SpaceCoordinates | None, optional
            The positions to get the neighbors from, by default None
        agents : int | Sequence[int] | None, optional
            The agents to get the neigbors from, by default None
        include_center : bool, optional
            If the position or agent should be included in the result, by default False

        Yields
        ------
        Iterator[dict[str, Any]]
            An iterator over neighboring agents where each agent is a dictionary with:
            - Attributes of the agent (the columns of its AgentSetDF dataframe)
            - Keys which are suffixed by '_center' to indicate the original center (eg. ['dim_0_center', 'dim_1_center', ...] for Grids, ['node_id_center', 'edge_id_center'] for Networks, 'agent_id_center' for agents)

        Raises
        ------
        ValueError
            If both pos and agents are None or if both pos and agents are not None.
        """
        return self._df_iterator(
            self.get_neighbors(
                radius=radius, pos=pos, agents=agents, include_center=include_center
            )
        )

    def iter_directions(
        self,
        pos0: SpaceCoordinate | SpaceCoordinates | None = None,
        pos1: SpaceCoordinate | SpaceCoordinates | None = None,
        agents0: int
        | Collection[int]
        | AgentContainer
        | Collection[AgentContainer]
        | None = None,
        agents1: int
        | Collection[int]
        | AgentContainer
        | Collection[AgentContainer]
        | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Return an iterator over the direction from pos0 to pos1 or agents0 to agents1.

        Parameters
        ----------
        pos0 : SpaceCoordinate | SpaceCoordinates | None, optional
            The starting positions, by default None
        pos1 : SpaceCoordinate | SpaceCoordinates | None, optional
            The ending positions, by default None
        agents0 : int | Sequence[int] | None, optional
            The starting agents, by default None
        agents1 : int | Sequence[int] | None, optional
            The ending agents, by default None

        Yields
        ------
        Iterator[dict[str, Any]]
            An iterator over the direction from pos0 to pos1 or agents0 to agents1 where each direction is a dictionary with:
            - Keys called according to the coordinates of the space(['dim_0', 'dim_1', ...] in Grids, ['node_id', 'edge_id'] in Networks)
            - Values representing the value of coordinates according to the dimension
        """
        return self._df_iterator(
            self.get_directions(pos0=pos0, pos1=pos1, agents0=agents0, agents1=agents1)
        )

    def iter_distances(
        self,
        pos0: SpaceCoordinate | SpaceCoordinates | None = None,
        pos1: SpaceCoordinate | SpaceCoordinates | None = None,
        agents0: int
        | Collection[int]
        | AgentContainer
        | Collection[AgentContainer]
        | None = None,
        agents1: int
        | Collection[int]
        | AgentContainer
        | Collection[AgentContainer]
        | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Return an iterator over the distance from pos0 to pos1 or agents0 to agents1.

        Parameters
        ----------
        pos0 : SpaceCoordinate | SpaceCoordinates | None, optional
            The starting positions, by default None
        pos1 : SpaceCoordinate | SpaceCoordinates | None, optional
            The ending positions, by default None
        agents0 : int | Sequence[int] | None, optional
            The starting agents, by default None
        agents1 : int | Sequence[int] | None, optional
            The ending agents, by default None

        Yields
        ------
        Iterator[dict[str, Any]]
            An iterator over the distance from pos0 to pos1 or agents0 to agents1 where each distance is a dictionary with:
            - A single key 'distance' representing the distance between the two positions
        """
        return self._df_iterator(
            self.get_distances(pos0=pos0, pos1=pos1, agents0=agents0, agents1=agents1)
        )

    def random_agents(
        self,
        n: int,
        seed: int | None = None,
    ) -> DataFrame:
        """Return a random sample of agents from the space.

        Parameters
        ----------
        n : int
            The number of agents to sample
        seed : int | None, optional
            The seed for the sampling, by default None
            If None, an integer from the model's random number generator is used.

        Returns
        -------
        DataFrame
            A DataFrame with the sampled agents
        """
        if seed is None:
            seed = self.random.integers(0)
        return self._df_sample(self._agents, n=n, seed=seed)

    @abstractmethod
    def get_directions(
        self,
        pos0: SpaceCoordinate | SpaceCoordinates | None = None,
        pos1: SpaceCoordinate | SpaceCoordinates | None = None,
        agents0: int
        | Collection[int]
        | AgentContainer
        | Collection[AgentContainer]
        | None = None,
        agents1: int
        | Collection[int]
        | AgentContainer
        | Collection[AgentContainer]
        | None = None,
    ) -> DataFrame:
        """Returns the direction from pos0 to pos1 or agents0 and agents1.
        If the space is a Network, the direction is the shortest path between the two nodes.
        In all other cases, the direction is the direction vector between the two positions.
        Either positions (pos0, pos1) or agents (agents0, agents1) must be specified, not both.
        They must have the same length.

        Parameters
        ----------
        pos0 : SpaceCoordinate | SpaceCoordinates | None = None
            The starting positions
        pos1 : SpaceCoordinate | SpaceCoordinates | None = None
            The ending positions
        agents0 : int | Sequence[int] | None = None
            The starting agents
        agents1 : int | Sequence[int] | None = None
            The ending agents

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
        agents0: int
        | Collection[int]
        | AgentContainer
        | Collection[AgentContainer]
        | None = None,
        agents1: int
        | Collection[int]
        | AgentContainer
        | Collection[AgentContainer]
        | None = None,
    ) -> DataFrame:
        """Returns the distance from pos0 to pos1.
        If the space is a Network, the distance is the number of nodes of the shortest path between the two nodes.
        In all other cases, the distance is Euclidean/l2/Frobenius norm.
        You should specify either positions (pos0, pos1) or agents (agents0, agents1), not both.
        pos0 and pos1 must be the same type of coordinates and have the same length.
        agents0 and agents1 must have the same length.

        Parameters
        ----------
        pos0 : SpaceCoordinate | SpaceCoordinates | None = None
            The starting positions
        pos1 : SpaceCoordinate | SpaceCoordinates | None = None
            The ending positions
        agents0 : int | Sequence[int] | None = None
            The starting agents
        agents1 : int | Sequence[int] | None = None
            The ending agents

        Returns
        -------
        DataFrame
            A DataFrame where each row represents the distance from pos0 to pos1 or agents0 to agents1
        """
        ...

    @abstractmethod
    def get_neighbors(
        self,
        radius: int | float | Sequence[int] | Sequence[float],
        pos: SpaceCoordinate | SpaceCoordinates | None = None,
        agents: int
        | Collection[int]
        | AgentContainer
        | Collection[AgentContainer]
        | None = None,
        include_center: bool = False,
    ) -> DataFrame:
        """Get the neighboring agents from given positions or agents according to a radius.
        Either positions or agents must be specified, not both.

        Parameters
        ----------
        radius : int | float
            The radius of the neighborhood
        pos : SpaceCoordinates | None, optional
            The coordinates of the cell to get the neighborhood from, by default None
        agent : int | None, optional
            The id of the agent to get the neighborhood from, by default None
        include_center : bool, optional
            If the cell or agent should be included in the result, by default False

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
    def move_agents(
        self,
        agents: int | Collection[int] | AgentContainer | Collection[AgentContainer],
        pos: SpaceCoordinate | SpaceCoordinates,
        inplace: bool = True,
    ) -> Self:
        """Place agents in the space according to the specified coordinates. If some agents are already placed,
        raises a RuntimeWarning.

        Parameters
        ----------
        agents : AgentContainer | Collection[AgentContainer] | int | Sequence[int]
            The agents to place in the space
        pos : SpaceCoordinates
            The coordinates for each agents. The length of the coordinates must match the number of agents.
        inplace : bool, optional
            Whether to perform the operation inplace, by default True

        Raises
        ------
        RuntimeWarning
            If some agents are already placed in the space.
        ValueError
            - If some agents are not part of the model.
            - If agents is int | Sequence[int] and some agents are present multiple times.

        Returns
        -------
        Self
        """
        ...

    @abstractmethod
    def move_to_empty(
        self,
        agents: int
        | Collection[int]
        | AgentContainer
        | Collection[AgentContainer]
        | None,
        inplace: bool = True,
    ) -> Self:
        """Move agents to empty cells/positions in the space.

        Parameters
        ----------
        agents : AgentContainer | Collection[AgentContainer] | int | Sequence[int]
            The agents to move to empty cells/positions
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
        seed: int | None = None,
    ) -> DataFrame:
        """Return a random sample of positions from the space.

        Parameters
        ----------
        n : int
            The number of positions to sample
        seed : int | None, optional
            The seed for the sampling, by default None
            If None, an integer from the model's random number generator is used.

        Returns
        -------
        DataFrame
            A DataFrame with the sampled positions
        """
        ...

    @abstractmethod
    def remove_agents(
        self,
        agents: int
        | Collection[int]
        | AgentContainer
        | Collection[AgentContainer]
        | None,
        inplace: bool = True,
    ):
        """Remove agents from the space

        Parameters
        ----------
        agents : AgentContainer | Collection[AgentContainer] | int | Sequence[int]
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
        ...

    @abstractmethod
    def swap_agents(
        self,
        agents0: int
        | Collection[int]
        | AgentContainer
        | Collection[AgentContainer]
        | None,
        agents1: int
        | Collection[int]
        | AgentContainer
        | Collection[AgentContainer]
        | None,
    ) -> Self:
        """Swap the positions of the agents in the space.
        agents0 and agents1 must have the same length and all agents must be placed in the space.

        Parameters
        ----------
        agents0 : AgentContainer | Collection[AgentContainer] | int | Sequence[int]
            The first set of agents to swap
        agents1 : AgentContainer | Collection[AgentContainer] | int | Sequence[int]
            The second set of agents to swap

        Returns
        -------
        Self
        """

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __str__(self) -> str: ...

    @property
    def agents(self) -> DataFrame | GeoDataFrame:
        """Get the ids of the agents placed in the cell set, along with their coordinates or geometries

        Returns
        -------
        AgentsDF
        """
        return self._agents

    @property
    def model(self) -> ModelDF:
        """The model to which the space belongs.

        Returns
        -------
        ModelDF
        """
        self._model

    @property
    def random(self) -> Generator:
        """The model's random number generator.

        Returns
        -------
        Generator
        """
        return self.model.random


class GeoSpaceDF(SpaceDF): ...


class DiscreteSpaceDF(SpaceDF):
    _capacity: int | None
    _cells: DataFrame
    _cells_col_names: list[str]
    _center_col_names: list[str]

    def __init__(
        self,
        model: ModelDF,
        capacity: int | None = None,
    ):
        super().__init__(model)
        self._capacity = capacity

    def iter_neighborhood(
        self,
        radius: int | Sequence[int],
        pos: DiscreteCoordinate | DataFrame | None = None,
        agents: int | Sequence[int] | None = None,
        include_center: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Return an iterator over the neighborhood cells from a given position according to a radius.

        Parameters
        ----------
        pos : DiscreteCoordinates
            The coordinates of the cell to get the neighborhood from
        radius : int
            The radius of the neighborhood
        include_center : bool, optional
            If the cell in the center of the neighborhood should be included in the result, by default False

        Returns
        ------
        Iterator[dict[str, Any]]
            An iterator over neighboring cell where each cell is a dictionary with:
            - Keys called according to the coordinates of the space(['dim_0', 'dim_1', ...] in Grids, ['node_id', 'edge_id'] in Networks)
            - Values representing the value of coordinates according to the dimension

        """
        return self._df_iterator(
            self.get_neighborhood(
                radius=radius, pos=pos, agents=agents, include_center=include_center
            )
        )

    def move_to_empty(
        self,
        agents: int
        | Collection[int]
        | AgentContainer
        | Collection[AgentContainer]
        | None,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)

        # Get Ids of agents
        # TODO: fix this
        if isinstance(agents, AgentContainer | Collection[AgentContainer]):
            agents = agents.index

        # Check ids presence in model
        b_contained = obj.model.agents.contains(agents)
        if (isinstance(b_contained, pl.Series) and not b_contained.all()) or (
            isinstance(b_contained, bool) and not b_contained
        ):
            raise ValueError("Some agents are not in the model")

        # Get empty cells
        empty_cells = obj._get_empty_cells(skip_agents=agents)
        if len(empty_cells) < len(agents):
            raise ValueError("Not enough empty cells to move agents")

        # Place agents
        obj._agents = obj.move_agents(agents, empty_cells)
        return obj

    def get_empty_cells(
        self,
        n: int | None = None,
        with_replacement: bool = True,
    ) -> DataFrame:
        """Get the empty cells in the space (cells without any agent).


        Parameters
        ----------
        n : int | None, optional
            _description_, by default None
        with_replacement : bool, optional
            If with_replacement is False, all cells are different.
            If with_replacement is True, some cells could be the same (but such that the total number of selection per cells is less or equal than the capacity), by default True

        Returns
        -------
        DataFrame
            _description_
        """
        return self._sample_cells(
            n, with_replacement, condition=lambda cap: cap == self._capacity
        )

    def get_free_cells(
        self,
        n: int | None = None,
        with_replacement: bool = True,
    ) -> DataFrame:
        """Get the free cells in the space (cells that have not reached maximum capacity).

        Parameters
        ----------
        n : int
            The number of empty cells to get
        with_replacement : bool, optional
            If with_replacement is False, all cells are different.
            If with_replacement is True, some cells could be the same (but such that the total number of selection per cells is at less or equal than the remaining capacity), by default True

        Returns
        -------
        DataFrame
            A DataFrame with free cells
        """
        return self._sample_cells(n, with_replacement, condition=lambda cap: cap > 0)

    def get_full_cells(
        self,
        n: int | None = None,
        with_replacement: bool = True,
    ) -> DataFrame:
        """Get the full cells in the space.

        Parameters
        ----------
        n : int
            The number of full cells to get

        Returns
        -------
        DataFrame
            A DataFrame with full cells
        """
        return self._sample_cells(n, with_replacement, condition=lambda cap: cap == 0)

    @abstractmethod
    def get_neighborhood(
        self,
        radius: int | float | Sequence[int] | Sequence[float],
        pos: DiscreteCoordinate | DataFrame | None = None,
        agents: int | Sequence[int] | None = None,
        include_center: bool = False,
    ) -> DataFrame:
        """Get the neighborhood cells from a given position.

        Parameters
        ----------
        pos : DiscreteCoordinates
            The coordinates of the cell to get the neighborhood from
        radius : int
            The radius of the neighborhood
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
    def get_cells(self, cells: DiscreteCoordinates | None = None) -> DataFrame:
        """Retrieve the dataframe of specified cells with their properties and agents.

        Parameters
        ----------
        cells : CellCoordinates, default is optional (all cells retrieved)

        Returns
        -------
        DataFrame
            A DataFrame where columns representing the CellCoordiantes
            (['x', 'y' in Grids, ['node_id', 'edge_id'] in Network]), an agent_id columns containing a list of agents
            in the cell and the properties of the cell
        """
        ...

    @abstractmethod
    def set_cells(
        self,
        properties: DataFrame,
        cells: DiscreteCoordinates | None = None,
        inplace: bool = True,
    ) -> Self:
        """Set the properties of the specified cells.
        Either the properties df must contain both the cell coordinates and the properties or
        the cell coordinates must be specified separately.
        If the Space is a Grid, the cell coordinates must be GridCoordinates.
        If the Space is a Network, the cell coordinates must be NetworkCoordinates.


        Parameters
        ----------
        properties : DataFrame
            The properties of the cells
        inplace : bool
            Whether to perform the operation inplace

        Returns
        -------
        Self
        """
        ...

    @abstractmethod
    def _get_empty_cells(
        self,
        skip_agents: Collection[int] | None = None,
    ): ...

    @abstractmethod
    def _sample_cells(
        self,
        n: int | None,
        with_replacement: bool,
        condition: Callable[[DiscreteSpaceCapacity], DiscreteSpaceCapacity],
    ) -> DataFrame:
        """Sample cells from the grid according to a condition on the capacity.

        Parameters
        ----------
        n : int | None
            The number of cells to sample
        with_replacement : bool
            If the sampling should be with replacement
        condition : Callable[[DiscreteSpaceCapacity], DiscreteSpaceCapacity]
            The condition to apply on the capacity

        Returns
        -------
        DataFrame
        """
        ...

    def __getitem__(self, cells: DiscreteCoordinates):
        return self.get_cells(cells)

    def __setitem__(self, cells: DiscreteCoordinates, properties: DataFrame):
        self.set_cells(properties=properties, cells=cells)

    def __getattr__(self, key: str) -> DataFrame:
        # Fallback, if key is not found in the object,
        # then it must mean that it's in the _cells dataframe
        return self._cells[key]

    def is_free(self, pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame:
        """Check whether the input positions are free (there exists at least one remaining spot in the cells)

        Parameters
        ----------
        pos : GridCoordinate | GridCoordinates
            The positions to check for

        Returns
        -------
        DataFrame
            A dataframe with positions and a boolean column "free"
        """
        df = self._df_constructor(data=pos, columns=self._cells_col_names)
        return self._df_add_columns(
            df, ["free"], self._df_get_bool_mask(df, mask=self.full_cells, negate=True)
        )

    def is_empty(self, pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame:
        """Check whether the input positions are empty (there isn't any single agent in the cells)

        Parameters
        ----------
        pos : GridCoordinate | GridCoordinates
            The positions to check for

        Returns
        -------
        DataFrame
            A dataframe with positions and a boolean column "empty"
        """
        df = self._df_constructor(data=pos, columns=self._cells_col_names)
        return self._df_add_columns(
            df, ["empty"], self._df_get_bool_mask(df, mask=self._cells, negate=True)
        )

    def is_full(self, pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame:
        """Check whether the input positions are full (there isn't any spot available in the cells)

        Parameters
        ----------
        pos : GridCoordinate | GridCoordinates
            The positions to check for

        Returns
        -------
        DataFrame
            A dataframe with positions and a boolean column "full"
        """
        df = self._df_constructor(data=pos, columns=self._cells_col_names)
        return self._df_add_columns(
            df, ["full"], self._df_get_bool_mask(df, mask=self.full_cells, negate=True)
        )

    # We use lru_cache because cached_property does not support a custom setter.
    # TODO: Test if there's an effective increase in performance
    @property
    @lru_cache(maxsize=1)
    def cells(self) -> DataFrame:
        return self.get_cells()

    @cells.setter
    def cells(self, df: DataFrame):
        return self.set_cells(df, inplace=True)

    @property
    def full_cells(self) -> DataFrame:
        df = self.cells
        return self._df_get_masked_df(
            self._cells, mask=df["n_agents"] == df["capacity"]
        )


class GridDF(DiscreteSpaceDF):
    _agents: DataFrame
    _cells: DataFrame
    _empty_grid: GridCapacity
    _torus: bool
    _offsets: DataFrame

    def __init__(
        self,
        model: ModelDF,
        dimensions: Sequence[int],
        torus: bool = False,
        capacity: int | None = None,
        neighborhood_type: str = "moore",
    ):
        """Grid cells are indexed, where [0, ..., 0] is assumed to be the
        bottom-left and [dimensions[0]-1, ..., dimensions[n]-1] is the top-right. If a grid is
        toroidal, the top and bottom, and left and right, edges wrap to each other.

        Parameters
        ----------
        model : ModelDF
            The model selfect to which the grid belongs
        dimensions: Sequence[int]
            The dimensions of the grid
        torus : bool, optional
            If the grid should be a torus, by default False
        capacity : int | None, optional
            The maximum number of agents that can be placed in a cell, by default None
        neighborhood_type: str, optional
            The type of neighborhood to consider, by default 'moore'.
            If 'moore', the neighborhood is the 8 cells around the center cell.
            If 'von_neumann', the neighborhood is the 4 cells around the center cell.
            If 'hexagonal', the neighborhood is 6 cells around the center cell.
        """
        super().__init__(model, capacity)
        self._dimensions = dimensions
        self._torus = torus
        self._cells_col_names = [f"dim_{k}" for k in range(len(dimensions))]
        self._center_col_names = [x + "_center" for x in self._cells_col_names]
        self._agents = self._df_constructor(
            columns=["agent_id"] + self._cells_col_names, index_col="agent_id"
        )
        self._cells = self._df_constructor(
            columns=self._cells_col_names + ["capacity"],
            index_cols=self._cells_col_names,
        )
        self._offsets = self._compute_offsets(neighborhood_type)
        self._empty_grid = self._generate_empty_grid(dimensions)

    def get_directions(
        self,
        pos0: GridCoordinate | GridCoordinates | None = None,
        pos1: GridCoordinate | GridCoordinates | None = None,
        agents0: int | Sequence[int] | None = None,
        agents1: int | Sequence[int] | None = None,
    ) -> DataFrame:
        pos0_df = self._get_df_coords(pos0, agents0)
        pos1_df = self._get_df_coords(pos1, agents1)
        assert len(pos0_df) == len(pos1_df), "objects must have the same length"
        return pos1_df - pos0_df

    def get_neighbors(
        self,
        radius: int | Sequence[int],
        pos: GridCoordinate | GridCoordinates | None = None,
        agents: int | Sequence[int] | None = None,
        include_center: bool = False,
    ) -> DataFrame:
        assert (
            pos is None and agents is not None or pos is not None and agents is None
        ), "Either pos or agents must be specified"
        neighborhood_df = self.get_neighborhood(
            radius=radius, pos=pos, agents=agents, include_center=include_center
        )
        return self._df_get_masked_df(
            neighborhood_df, index_col="agent_id", columns=self._agents.columns
        )

    def get_cells(self, cells: GridCoordinates | None = None) -> DataFrame:
        coords = self._get_df_coords(cells)
        return self._get_cells_df(coords)

    def move_agents(
        self,
        agents: AgentSetDF | Iterable[AgentSetDF] | int | Sequence[int],
        pos: GridCoordinates,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)

        # Get Ids of agents
        if isinstance(agents, AgentContainer | Collection[AgentContainer]):
            agents = agents.index

        # Check ids presence in model
        b_contained = obj.model.agents.contains(agents)
        if (isinstance(b_contained, pl.Series) and not b_contained.all()) or (
            isinstance(b_contained, bool) and not b_contained
        ):
            raise ValueError("Some agents are not in the model")

        # Check ids are unique
        agents = pl.Series(agents)
        if agents.unique_counts() != len(agents):
            raise ValueError("Some agents are present multiple times")

        # Warn if agents are already placed
        if agents.is_in(obj._agents["agent_id"]):
            warn("Some agents are already placed in the grid", RuntimeWarning)

        # Place agents (checking that capacity is not )
        coords = obj._get_df_coords(pos)
        obj._agents = obj._place_agents_df(agents, coords)
        return obj

    def out_of_bounds(self, pos: SpaceCoordinates) -> DataFrame:
        """Check if a position is out of bounds.

        Parameters
        ----------
        pos : SpaceCoordinates


        Returns
        -------
        DataFrame
            A DataFrame with a' column representing the coordinates and an 'out_of_bounds' containing boolean values.
        """
        pos_df = self._get_df_coords(pos)
        out_of_bounds = pos_df < 0 | pos_df >= self._dimensions
        return self._df_constructor(
            data=[pos_df, out_of_bounds],
        )

    def remove_agents(
        self,
        agents: AgentContainer | Collection[AgentContainer] | int | Sequence[int],
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)

        # Get Ids of agents
        if isinstance(agents, AgentContainer | Collection[AgentContainer]):
            agents = agents.index

        # Check ids presence in model
        b_contained = obj.model.agents.contains(agents)
        if (isinstance(b_contained, pl.Series) and not b_contained.all()) or (
            isinstance(b_contained, bool) and not b_contained
        ):
            raise ValueError("Some agents are not in the model")

        # Remove agents
        obj._agents = obj._df_remove(obj._agents, ids=agents, index_col="agent_id")

        return obj

    def torus_adj(self, pos: GridCoordinates) -> DataFrame:
        """Get the toroidal adjusted coordinates of a position.

        Parameters
        ----------
        pos : GridCoordinates
            The coordinates to adjust

        Returns
        -------
        DataFrame
            The adjusted coordinates
        """
        df_coords = self._get_df_coords(pos)
        df_coords = df_coords % self._dimensions
        return df_coords

    @abstractmethod
    def get_neighborhood(
        self,
        radius: int | Sequence[int],
        pos: GridCoordinate | GridCoordinates | None = None,
        agents: int | Sequence[int] | None = None,
        include_center: bool = False,
    ) -> DataFrame: ...

    def _get_df_coords(
        self,
        pos: GridCoordinate | GridCoordinates | None = None,
        agents: int | Sequence[int] | None = None,
    ) -> DataFrame:
        """Get the DataFrame of coordinates from the specified positions or agents.

        Parameters
        ----------
        pos : GridCoordinate | GridCoordinates | None, optional
        agents : int | Sequence[int] | None, optional

        Returns
        -------
        DataFrame
            A dataframe where each column represent a column

        Raises
        ------
        ValueError
            If neither pos or agents are specified
        """
        assert (
            pos is not None or agents is not None
        ), "Either pos or agents must be specified"
        if agents:
            return self._df_get_masked_df(
                self._agents, index_col="agent_id", mask=agents
            )
        if isinstance(pos, DataFrame):
            return pos[self._cells_col_names]
        elif isinstance(pos, Sequence) and len(pos) == len(self._dimensions):
            # This means that the sequence is already a sequence where each element is the
            # sequence of coordinates for dimension i
            for i, c in enumerate(pos):
                if isinstance(c, slice):
                    start = c.start if c.start is not None else 0
                    step = c.step if c.step is not None else 1
                    stop = c.stop if c.stop is not None else self._dimensions[i]
                    pos[i] = pl.arange(start=start, end=stop, step=step)
                elif isinstance(c, int):
                    pos[i] = [c]
            return self._df_constructor(data=pos, columns=self._cells_col_names)
        elif isinstance(pos, Collection) and all(
            len(c) == len(self._dimensions) for c in pos
        ):
            # This means that we have a collection of coordinates
            sequences = []
            for i in range(len(self._dimensions)):
                sequences.append([c[i] for c in pos])
            return self._df_constructor(data=sequences, columns=self._cells_col_names)
        elif isinstance(pos, int) and len(self._dimensions) == 1:
            return self._df_constructor(data=[pos], columns=self._cells_col_names)
        else:
            raise ValueError("Invalid coordinates")

    def _compute_offsets(self, neighborhood_type: str) -> DataFrame:
        """Generate offsets for the neighborhood.

        Parameters
        ----------
        neighborhood_type : str
            _description_

        Returns
        -------
        DataFrame
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
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
            if len(self._dimensions) != 2:
                raise ValueError("Hexagonal grid only supports 2 dimensions")
            directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        else:
            raise ValueError("Invalid neighborhood type specified")

        return self._df_constructor(data=directions, columns=self._cells_col_names)

    @abstractmethod
    def _generate_empty_grid(self, dimensions: Sequence[int]) -> Any:
        """Generate an empty grid with the specified dimensions.

        Parameters
        ----------
        dimensions : Sequence[int]

        Returns
        -------
        Any
        """

    @abstractmethod
    def _get_cells_df(self, coords: GridCoordinates) -> DataFrame: ...

    @abstractmethod
    def _place_agents_df(
        self, agents: int | Sequence[int], coords: GridCoordinates
    ) -> DataFrame: ...

    @abstractmethod
    def _sample_cells(
        self,
        n: int | None,
        with_replacement: bool,
        condition: Callable[[GridCapacity], GridCapacity],
    ) -> DataFrame: ...

    def __getitem__(self, cells: GridCoordinates):
        return super().__getitem__(cells)

    def __setitem__(self, cells: GridCoordinates, properties: DataFrame):
        return super().__setitem__(cells, properties)


class GridPandas(GridDF, PandasMixin):
    _agents: pd.DataFrame
    _cells: pd.DataFrame
    _empty_grid: np.ndarray
    _offsets: pd.DataFrame

    def get_distances(
        self,
        pos0: SpaceCoordinate | SpaceCoordinates | None = None,
        pos1: SpaceCoordinate | SpaceCoordinates | None = None,
        agents0: int | Sequence[int] | None = None,
        agents1: int | Sequence[int] | None = None,
    ) -> pd.DataFrame:
        pos0_df = self._get_df_coords(pos0, agents0)
        pos1_df = self._get_df_coords(pos1, agents1)
        return pd.DataFrame(np.linalg.norm(pos1_df - pos0_df, axis=1))

    def get_neighborhood(
        self,
        radius: int | Sequence[int],
        pos: GridCoordinate | GridCoordinates | None = None,
        agents: int | Sequence[int] | None = None,
        include_center: bool = False,
    ) -> pd.DataFrame:
        pos_df = self._get_df_coords(pos)

        # Create all possible neighbors by multipling directions by the radius and adding original pos
        neighbors_df = self._offsets.join(
            [pd.Series(np.arange(1, radius + 1), name="radius"), pos_df],
            how="cross",
            rsuffix="_center",
        )

        neighbors_df = (
            neighbors_df[self._cells_col_names] * neighbors_df["radius"]
            + neighbors_df[self._center_col_names]
        ).drop(columns=["radius"])

        # If torus, "normalize" (take modulo) for out-of-bounds cells
        if self._torus:
            neighbors_df = self.torus_adj(neighbors_df)

        # Filter out-of-bound neighbors (all ensures that if any coordinates violates, it gets excluded)
        neighbors_df = neighbors_df[
            ((neighbors_df >= 0) & (neighbors_df < self._dimensions)).all(axis=1)
        ]

        if include_center:
            pos_df[self._center_col_names] = pos_df[self._cells_col_names]
            neighbors_df = pd.concat([neighbors_df, pos_df], ignore_index=True)

        return neighbors_df

    def set_cells(self, df: pd.DataFrame, inplace: bool = True) -> Self:
        if df.index.names != self._cells_col_names or not all(
            k in df.columns for k in self._cells_col_names
        ):
            raise ValueError(
                "The dataframe must have columns/MultiIndex 'dim_0', 'dim_1', ..."
            )
        obj = self._get_obj(inplace)
        df = df.set_index(self._cells_col_names)
        obj._cells = df.combine_first(obj._cells)
        return obj

    def _generate_empty_grid(
        self, dimensions: Sequence[int], capacity: int
    ) -> np.ogrid:
        return np.full(dimensions, capacity, dtype=int)

    def _get_df_coords(
        self,
        pos: GridCoordinate | GridCoordinates | None = None,
        agents: int | Sequence[int] | None = None,
    ) -> pd.DataFrame:
        return super()._get_df_coords(pos=pos, agents=agents)

    def _get_cells_df(self, coords: GridCoordinates) -> pd.DataFrame:
        return (
            pd.DataFrame({k: v for k, v in zip(self._cells_col_names, coords)})
            .set_index(self._cells_col_names)
            .merge(
                self._agents.reset_index(),
                how="left",
                left_index=True,
                right_on=self._cells_col_names,
            )
            .groupby(level=self._cells_col_names)
            .agg(agents=("index", list), n_agents=("index", "size"))
            .merge(self._cells, how="left", left_index=True, right_index=True)
        )

    def _place_agents_df(
        self, agents: int | Sequence[int], coords: GridCoordinates
    ) -> pd.DataFrame:
        new_df = pd.DataFrame(
            {k: v for k, v in zip(self._cells_col_names, coords)},
            index=pd.Index(agents, name="agent_id"),
        )
        new_df = self._agents.combine_first(new_df)

        # Check if the capacity is respected
        capacity_df = (
            new_df.value_counts(subset=self._cells_col_names)
            .to_frame("n_agents")
            .merge(self._cells["capacity"], on=self._cells_col_names)
        )
        capacity_df["capacity"] = capacity_df["capacity"].fillna(self._capacity)
        if (capacity_df["n_agents"] > capacity_df["capacity"]).any():
            raise ValueError(
                "There is at least a cell where the number of agents would be higher than the capacity of the cell"
            )

        return new_df

    def _sample_cells(
        self,
        n: int | None,
        with_replacement: bool,
        condition: Callable[[np.ndarray], np.ndarray],
    ) -> pd.DataFrame:
        # Get the coordinates and remaining capacities of the cells
        coords = np.array(np.where(condition(self._empty_grid))).T
        capacities = self._empty_grid[tuple(coords.T)]

        if n is not None:
            if with_replacement:
                assert (
                    n <= capacities.sum()
                ), "Requested sample size exceeds the total available capacity."

                # Initialize the sampled coordinates list
                sampled_coords = []

                # Resample until we have the correct number of samples with valid capacities
                while len(sampled_coords) < n:
                    # Calculate the remaining samples needed
                    remaining_samples = n - len(sampled_coords)

                    # Compute uniform probabilities for sampling (excluding full cells)
                    probabilities = np.ones(len(coords)) / len(coords)

                    # Sample with replacement using uniform probabilities
                    sampled_indices = np.random.choice(
                        len(coords),
                        size=remaining_samples,
                        replace=True,
                        p=probabilities,
                    )
                    new_sampled_coords = coords[sampled_indices]

                    # Update capacities
                    unique_coords, counts = np.unique(
                        new_sampled_coords, axis=0, return_counts=True
                    )
                    self._empty_grid[tuple(unique_coords.T)] -= counts

                    # Check if any cells exceed their capacity and need to be resampled
                    over_capacity_mask = self._empty_grid[tuple(unique_coords.T)] < 0
                    valid_coords = unique_coords[~over_capacity_mask]
                    invalid_coords = unique_coords[over_capacity_mask]

                    # Add valid coordinates to the sampled list
                    sampled_coords.extend(valid_coords)

                    # Restore capacities for invalid coordinates
                    if len(invalid_coords) > 0:
                        self._empty_grid[tuple(invalid_coords.T)] += counts[
                            over_capacity_mask
                        ]

                    # Update coords based on the current state of the grid
                    coords = np.array(np.where(condition(self._empty_grid))).T

                sampled_coords = np.array(sampled_coords[:n])
            else:
                assert n <= len(
                    coords
                ), "Requested sample size exceeds the number of available cells."

                # Sample without replacement
                sampled_indices = np.random.choice(len(coords), size=n, replace=False)
                sampled_coords = coords[sampled_indices]

                # No need to update capacities as sampling is without replacement
        else:
            sampled_coords = coords

        # Convert the coordinates to a DataFrame
        sampled_cells = pd.DataFrame(sampled_coords, columns=self._cells_col_names)

        return sampled_cells


class GridPolars(GridDF, PolarsMixin):
    _agents: pl.DataFrame
    _cells: pl.DataFrame
    _empty_grid: list[pl.Expr]
    _offsets: pl.DataFrame

    def get_distances(
        self,
        pos0: SpaceCoordinate | SpaceCoordinates | None = None,
        pos1: SpaceCoordinate | SpaceCoordinates | None = None,
        agents0: int | Sequence[int] | None = None,
        agents1: int | Sequence[int] | None = None,
    ) -> pl.DataFrame:
        pos0_df = self._get_df_coords(pos0, agents0)
        pos1_df = self._get_df_coords(pos1, agents1)
        return pos0_df - pos1_df

    def get_neighborhood(
        self,
        radius: int | Sequence[int],
        pos: GridCoordinate | GridCoordinates | None = None,
        agents: int | Sequence[int] | None = None,
        include_center: bool = False,
    ) -> pl.DataFrame:
        pos_df = self._get_df_coords(pos)

        # Create all possible neighbors by multiplying directions by the radius and adding original pos
        neighbors_df = self._offsets.join(
            [pl.arange(1, radius + 1, eager=True).to_frame(name="radius"), pos_df],
            how="cross",
            suffix="_center",
        )

        neighbors_df = neighbors_df.with_columns(
            (
                pl.col(self._cells_col_names) * pl.col("radius")
                + pl.col(self._center_col_names)
            ).alias(pl.col(self._cells_col_names))
        ).drop("radius")

        # If torus, "normalize" (take modulo) for out-of-bounds cells
        if self._torus:
            neighbors_df = self.torus_adj(neighbors_df)
            neighbors_df = cast(
                pl.DataFrame, neighbors_df
            )  # Previous return is Any according to linter but should be DataFrame

        # Filter out-of-bound neighbors
        neighbors_df = neighbors_df.filter(
            pl.all((neighbors_df < self._dimensions) & (neighbors_df >= 0))
        )

        if include_center:
            pos_df.with_columns(
                pl.col(self._cells_col_names).alias(self._center_col_names)
            )
            neighbors_df = pl.concat([neighbors_df, pos_df], how="vertical")

        return neighbors_df

    def set_cells(self, df: pl.DataFrame, inplace: bool = True) -> Self:
        if not all(k in df.columns for k in self._cells_col_names):
            raise ValueError(
                "The dataframe must have an columns/MultiIndex 'dim_0', 'dim_1', ..."
            )
        obj = self._get_obj(inplace)
        obj._cells = obj._combine_first(obj._cells, df, on=self._cells_col_names)
        return obj

    def _generate_empty_grid(self, dimensions: Sequence[int]) -> list[pl.Expr]:
        return [pl.arange(0, d, eager=False) for d in dimensions]

    def _get_df_coords(
        self,
        pos: GridCoordinate | GridCoordinates | None = None,
        agents: int | Sequence[int] | None = None,
    ) -> pl.DataFrame:
        return super()._get_df_coords(pos, agents)

    def _get_cells_df(self, coords: GridCoordinates) -> pl.DataFrame:
        return (
            pl.DataFrame({k: v for k, v in zip(self._cells_col_names, coords)})
            .join(self._agents, how="left", on=self._cells_col_names)
            .group_by(self._cells_col_names)
            .agg(
                pl.col("agent_id").list().alias("agents"),
                pl.col("agent_id").count().alias("n_agents"),
            )
            .join(self._cells, on=self._cells_col_names, how="left")
        )

    def _place_agents_df(
        self, agents: int | Sequence[int], coords: GridCoordinates
    ) -> pl.DataFrame:
        new_df = pl.DataFrame(
            {"agent_id": agents}.update(
                {k: v for k, v in zip(self._cells_col_names, coords)}
            )
        )
        new_df: pl.DataFrame = self._df_combine_first(
            self._agents, new_df, on="agent_id"
        )

        # Check if the capacity is respected
        capacity_df = (
            new_df.group_by(self._cells_col_names)
            .count()
            .join(
                self._cells[self._cells_col_names + ["capacity"]],
                on=self._cells_col_names,
            )
        )
        capacity_df = capacity_df.with_columns(
            capacity=pl.col("capacity").fill_null(self._capacity)
        )
        if (capacity_df["count"] > capacity_df["capacity"]).any():
            raise ValueError(
                "There is at least a cell where the number of agents would be higher than the capacity of the cell"
            )

        return new_df

    def _sample_cells_lazy(
        self,
        n: int | None,
        with_replacement: bool,
        condition: Callable[[pl.Expr], pl.Expr],
    ) -> pl.DataFrame:
        # Create a base DataFrame with all grid coordinates and default capacities
        grid_df = pl.DataFrame(self._empty_grid).with_columns(
            [pl.lit(self._capacity).alias("capacity")]
        )

        # Apply the condition to filter the cells
        grid_df = grid_df.filter(condition(pl.col("capacity")))

        if n is not None:
            if with_replacement:
                assert (
                    n <= grid_df.select(pl.sum("capacity")).item()
                ), "Requested sample size exceeds the total available capacity."

                # Initialize the sampled DataFrame
                sampled_df = pl.DataFrame()

                # Resample until we have the correct number of samples with valid capacities
                while sampled_df.shape[0] < n:
                    # Calculate the remaining samples needed
                    remaining_samples = n - sampled_df.shape[0]

                    # Sample with replacement using uniform probabilities
                    sampled_part = grid_df.sample(
                        n=remaining_samples, with_replacement=True
                    )

                    # Count occurrences of each sampled coordinate
                    count_df = sampled_part.group_by(self._cells_col_names).agg(
                        pl.count("capacity").alias("sampled_count")
                    )

                    # Adjust capacities based on counts
                    grid_df = (
                        grid_df.join(count_df, on=self._cells_col_names, how="left")
                        .with_columns(
                            [
                                (
                                    pl.col("capacity")
                                    - pl.col("sampled_count").fill_null(0)
                                ).alias("capacity")
                            ]
                        )
                        .drop("sampled_count")
                    )

                    # Ensure no cell exceeds its capacity
                    valid_sampled_part = sampled_part.join(
                        grid_df.filter(pl.col("capacity") >= 0),
                        on=self._cells_col_names,
                        how="inner",
                    )

                    # Add valid samples to the result
                    sampled_df = pl.concat([sampled_df, valid_sampled_part])

                    # Filter out over-capacity cells from the grid
                    grid_df = grid_df.filter(pl.col("capacity") > 0)

                sampled_df = sampled_df.head(n)  # Ensure we have exactly n samples
            else:
                assert (
                    n <= grid_df.height
                ), "Requested sample size exceeds the number of available cells."

                # Sample without replacement
                sampled_df = grid_df.sample(n=n, with_replacement=False)
        else:
            sampled_df = grid_df

        return sampled_df

    def _sample_cells_eager(
        self,
        n: int | None,
        with_replacement: bool,
        condition: Callable[[pl.Expr], pl.Expr],
    ) -> pl.DataFrame:
        # Create a base DataFrame with all grid coordinates and default capacities
        grid_df = pl.DataFrame(self._empty_grid).with_columns(
            [pl.lit(self._capacity).alias("capacity")]
        )

        # If there are any specific capacities in self._cells, update the grid_df with these values
        if not self._cells.is_empty():
            grid_df = (
                grid_df.join(self._cells, on=self._cells_col_names, how="left")
                .with_columns(
                    [
                        pl.col("capacity_right")
                        .fill_null(pl.col("capacity"))
                        .alias("capacity")
                    ]
                )
                .drop("capacity_right")
            )

        # Apply the condition to filter the cells
        grid_df = grid_df.filter(condition(pl.col("capacity")))

        if n is not None:
            if with_replacement:
                assert (
                    n <= grid_df.select(pl.sum("capacity")).item()
                ), "Requested sample size exceeds the total available capacity."

                # Initialize the sampled DataFrame
                sampled_df = pl.DataFrame()

                # Resample until we have the correct number of samples with valid capacities
                while sampled_df.shape[0] < n:
                    # Calculate the remaining samples needed
                    remaining_samples = n - sampled_df.shape[0]

                    # Sample with replacement using uniform probabilities
                    sampled_part = grid_df.sample(
                        n=remaining_samples, with_replacement=True
                    )

                    # Count occurrences of each sampled coordinate
                    count_df = sampled_part.group_by(self._cells_col_names).agg(
                        pl.count("capacity").alias("sampled_count")
                    )

                    # Adjust capacities based on counts
                    grid_df = (
                        grid_df.join(count_df, on=self._cells_col_names, how="left")
                        .with_columns(
                            [
                                (
                                    pl.col("capacity")
                                    - pl.col("sampled_count").fill_null(0)
                                ).alias("capacity")
                            ]
                        )
                        .drop("sampled_count")
                    )

                    # Ensure no cell exceeds its capacity
                    valid_sampled_part = sampled_part.join(
                        grid_df.filter(pl.col("capacity") >= 0),
                        on=self._cells_col_names,
                        how="inner",
                    )

                    # Add valid samples to the result
                    sampled_df = pl.concat([sampled_df, valid_sampled_part])

                    # Filter out over-capacity cells from the grid
                    grid_df = grid_df.filter(pl.col("capacity") > 0)

                sampled_df = sampled_df.head(n)  # Ensure we have exactly n samples
            else:
                assert (
                    n <= grid_df.height
                ), "Requested sample size exceeds the number of available cells."

                # Sample without replacement
                sampled_df = grid_df.sample(n=n, with_replacement=False)
        else:
            sampled_df = grid_df

        return sampled_df

    def _sample_cells(
        self,
        n: int | None,
        with_replacement: bool,
        condition: Callable[[pl.Expr], pl.Expr],
    ) -> pl.DataFrame:
        if "capacity" not in self._cells.columns:
            return self._sample_cells_lazy(n, with_replacement, condition)
        else:
            return self._sample_cells_eager(n, with_replacement, condition)


class GeoGridDF(GridDF, GeoSpaceDF): ...


class NetworkDF(DiscreteSpaceDF):
    _network: nx.Graph
    _nodes: pd.DataFrame
    _links: pd.DataFrame

    def torus_adj(self, pos):
        raise NotImplementedError("No concept of torus in Networks")

    @abstractmethod
    def __iter__(self) -> Iterable:
        pass

    @abstractmethod
    def connect(self, nodes0, nodes1):
        pass

    @abstractmethod
    def disconnect(self, nodes0, nodes1):
        pass


class ContinousSpaceDF(GeoSpaceDF):
    _agents: gpd.GeoDataFrame
    _limits: Sequence[float]

    def __init__(self, model: ModelDF, ref_sys: CRS | ESPG | str | None = None) -> None:
        """Create a new CellSet object.

        Parameters
        ----------
        model : ModelDF
        ref_sys : CRS | ESPG | str | None, optional
            Coordinate Reference System. ESPG is an integer, by default None

        Returns
        -------
        None
        """
        super().__init__(model)
        self._cells = gpd.GeoDataFrame(columns=["agent_id", "geometry"], crs=ref_sys)

    def get_neighborhood(
        self,
        pos: shp.Point | Sequence[tuple[int | float, int | float]],
        radius: float | int,
        include_center: bool = False,
        inplace: bool = True,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """Get the neighborhood cells from a given position.

        Parameters
        ----------
        pos : shp.Point | Sequence[int, int]
            The selfect to get the neighborhood from.
        radius : float | int
            The radius of the neighborhood
        include_center : bool, optional
            If the cell in the center of the neighborhood should be included in the result, by default False
        inplace : bool, optional
            If the method should return a new instance of the class or modify the current one, by default True
        **kwargs
            Extra arguments to be passed to shp.Point.buffer.

        Returns
        -------
        GeoDataFrame
            Cells in the neighborhood
        """
        if isinstance(pos, Sequence[int, int]):
            pos = shp.Point(pos)
            pos = pos.buffer(distance=radius, **kwargs)
        if include_center:
            return self._cells[self._cells.within(other=pos)]
        else:
            return self._cells[
                self._cells.within(other=pos) & ~self._cells.intersects(other=pos)
            ]

    def get_direction(self, pos0, pos1):
        pass

    @property
    def crs(self) -> CRS:
        if self._agents.crs is None:
            raise ValueError("CRS not set")
        return self._agents.crs

    @crs.setter
    def crs(self, ref_sys: CRS | ESPG | str | None):
        if isinstance(ref_sys, ESPG):
            self._agents = self._agents.to_crs(espg=ref_sys)
        else:
            self._agents = self._agents.to_crs(crs=ref_sys)
        return self


class MultiSpaceDF(Collection[SpaceDF]):
    _spaces: Collection[SpaceDF]
