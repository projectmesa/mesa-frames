from abc import abstractmethod
from collections.abc import Callable, Collection, Iterable, Sequence
from functools import lru_cache
from itertools import product
from typing import TYPE_CHECKING
from warnings import warn

import polars as pl
from numpy.random import Generator
from typing_extensions import Any, Self

from collections.abc import Iterator

from mesa_frames.abstract.agents import AgentContainer
from mesa_frames.abstract.mixin import CopyMixin, DataFrameMixin
from mesa_frames.types_ import (
    DataFrame,
    DiscreteCoordinate,
    DiscreteCoordinates,
    DiscreteSpaceCapacity,
    GeoDataFrame,
    GridCapacity,
    GridCoordinate,
    GridCoordinates,
    IdsLike,
    SpaceCoordinate,
    SpaceCoordinates,
)

ESPG = int

if TYPE_CHECKING:
    from mesa_frames.abstract.agents import AgentSetDF
    from mesa_frames.concrete.model import ModelDF


class SpaceDF(CopyMixin, DataFrameMixin):
    _model: "ModelDF"
    _agents: DataFrame | GeoDataFrame

    def __init__(self, model: "ModelDF") -> None:
        """Create a new SpaceDF object.

        Parameters
        ----------
        model : 'ModelDF'

        Returns
        -------
        None
        """
        self._model = model

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
        agents0: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
        agents1: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
        normalize: bool = False,
    ) -> DataFrame:
        """Returns the directions from pos0 to pos1 or agents0 and agents1.
        If the space is a Network, the direction is the shortest path between the two nodes.
        In all other cases, the direction is the direction vector between the two positions.
        Either positions (pos0, pos1) or agents (agents0, agents1) must be specified, not both and they must have the same length.

        Parameters
        ----------
        pos0 : SpaceCoordinate | SpaceCoordinates | None, optional
            The starting positions
        pos1 : SpaceCoordinate | SpaceCoordinates | None, optional
            The ending positions
        agents0 : IdsLike | AgentContainer | Collection[AgentContainer] | None, optional
            The starting agents
        agents1 : IdsLike | AgentContainer | Collection[AgentContainer] | None, optional
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
        agents0: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
        agents1: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
    ) -> DataFrame:
        """Returns the distances from pos0 to pos1 or agents0 and agents1.
        If the space is a Network, the distance is the number of nodes of the shortest path between the two nodes.
        In all other cases, the distance is Euclidean/l2/Frobenius norm.
        You should specify either positions (pos0, pos1) or agents (agents0, agents1), not both and they must have the same length.

        Parameters
        ----------
        pos0 : SpaceCoordinate | SpaceCoordinates | None, optional
            The starting positions
        pos1 : SpaceCoordinate | SpaceCoordinates | None, optional
            The ending positions
        agents0 : IdsLike | AgentContainer | Collection[AgentContainer], optional
            The starting agents
        agents1 : IdsLike | AgentContainer | Collection[AgentContainer], optional
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
        agents: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
        include_center: bool = False,
    ) -> DataFrame:
        """Get the neighboring agents from given positions or agents according to the specified radiuses.
        Either positions (pos0, pos1) or agents (agents0, agents1) must be specified, not both and they must have the same length.

        Parameters
        ----------
        radius : int | float | Sequence[int] | Sequence[float]
            The radius(es) of the neighborhood
        pos : SpaceCoordinate | SpaceCoordinates | None, optional
            The coordinates of the cell to get the neighborhood from, by default None
        agents : IdsLike | AgentContainer | Collection[AgentContainer] | None, optional
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
    def move_agents(
        self,
        agents: IdsLike | AgentContainer | Collection[AgentContainer],
        pos: SpaceCoordinate | SpaceCoordinates,
        inplace: bool = True,
    ) -> Self:
        """Place agents in the space according to the specified coordinates. If some agents are already placed,
        raises a RuntimeWarning.

        Parameters
        ----------
        agents : IdsLike | AgentContainer | Collection[AgentContainer]
            The agents to place in the space
        pos : SpaceCoordinate | SpaceCoordinates
            The coordinates for each agents. The length of the coordinates must match the number of agents.
        inplace : bool, optional
            Whether to perform the operation inplace, by default True

        Raises
        ------
        RuntimeWarning
            If some agents are already placed in the space.
        ValueError
            - If some agents are not part of the model.
            - If agents is IdsLike and some agents are present multiple times.

        Returns
        -------
        Self
        """
        ...

    @abstractmethod
    def move_to_empty(
        self,
        agents: IdsLike | AgentContainer | Collection[AgentContainer],
        inplace: bool = True,
    ) -> Self:
        """Move agents to empty cells/positions in the space (cells/positions where there isn't any single agent).

        Parameters
        ----------
        agents : IdsLike | AgentContainer | Collection[AgentContainer]
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
        agents: IdsLike | AgentContainer | Collection[AgentContainer],
        inplace: bool = True,
    ):
        """Remove agents from the space.

        Parameters
        ----------
        agents : IdsLike | AgentContainer | Collection[AgentContainer]
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
        agents0: IdsLike | AgentContainer | Collection[AgentContainer],
        agents1: IdsLike | AgentContainer | Collection[AgentContainer],
    ) -> Self:
        """Swap the positions of the agents in the space.
        agents0 and agents1 must have the same length and all agents must be placed in the space.

        Parameters
        ----------
        agents0 : IdsLike | AgentContainer | Collection[AgentContainer]
            The first set of agents to swap
        agents1 : IdsLike | AgentContainer | Collection[AgentContainer]
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
    def model(self) -> "ModelDF":
        """The model to which the space belongs.

        Returns
        -------
        'ModelDF'
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


class DiscreteSpaceDF(SpaceDF):
    _capacity: int | None
    _cells: DataFrame
    _cells_col_names: list[str]
    _center_col_names: list[str]

    def __init__(
        self,
        model: "ModelDF",
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
        model: "ModelDF",
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
        model : 'ModelDF'
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
