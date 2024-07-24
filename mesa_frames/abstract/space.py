from abc import abstractmethod
from collections.abc import Callable, Collection, Sequence
from itertools import product
from typing import TYPE_CHECKING, Literal
from warnings import warn

import polars as pl
from numpy.random import Generator
from typing_extensions import Self

from mesa_frames.abstract.agents import AgentContainer
from mesa_frames.abstract.mixin import CopyMixin, DataFrameMixin
from mesa_frames.types_ import (
    BoolSeries,
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
    from mesa_frames.concrete.model import ModelDF


class SpaceDF(CopyMixin, DataFrameMixin):
    """The SpaceDF class is an abstract class that defines the interface for all space classes in mesa_frames.

    Methods
    -------
    __init__(model: 'ModelDF')
        Create a new SpaceDF object.
    random_agents(n: int, seed: int | None = None) -> DataFrame
        Return a random sample of agents from the space.
    get_directions(
        pos0: SpaceCoordinate | SpaceCoordinates | None = None,
        pos1: SpaceCoordinate | SpaceCoordinates | None = None,
        agents0: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
        agents1: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
        normalize: bool = False,
    ) -> DataFrame
        Returns the directions from pos0 to pos1 or agents0 and agents1.
    get_distances(
        pos0: SpaceCoordinate | SpaceCoordinates | None = None,
        pos1: SpaceCoordinate | SpaceCoordinates | None = None,
        agents0: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
        agents1: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
    ) -> DataFrame
        Returns the distances from pos0 to pos1 or agents0 and agents1.
    get_neighbors(
        radius: int | float | Sequence[int] | Sequence[float],
        pos: Space
    ) -> DataFrame
        Get the neighboring agents from given positions or agents according to the specified radiuses.
    move_agents(
        agents: IdsLike | AgentContainer | Collection[AgentContainer],
        pos
    ) -> Self
        Place agents in the space according to the specified coordinates.
    move_to_empty(
        agents: IdsLike | AgentContainer | Collection[AgentContainer],
        inplace: bool = True,
    ) -> Self
        Move agents to empty cells/positions in the space.
    random_pos(
        n: int,
        seed: int | None = None,
    ) -> DataFrame
        Return a random sample of positions from the space.
    remove_agents(
        agents: IdsLike | AgentContainer | Collection[AgentContainer],
        inplace: bool = True,
    )
        Remove agents from the space.
    swap_agents(
        agents0: IdsLike | AgentContainer | Collection[AgentContainer],
        agents1: IdsLike | AgentContainer | Collection[AgentContainer],
    ) -> Self
        Swap the positions of the agents in the space.
    """

    _model: "ModelDF"
    _agents: DataFrame | GeoDataFrame  # Stores the agents placed in the space

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
    """The DiscreteSpaceDF class is an abstract class that defines the interface for all discrete space classes (Grids and Networks) in mesa_frames.

    Methods
    -------
    __init__(model: 'ModelDF', capacity: int | None = None)
        Create a new DiscreteSpaceDF object.
    is_available(pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame
        Check whether the input positions are available (there exists at least one remaining spot in the cells).
    is_empty(pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame
        Check whether the input positions are empty (there isn't any single agent in the cells).
    is_full(pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame
        Check whether the input positions are full (there isn't any spot available in the cells).
    move_to_empty(agents: IdsLike | AgentContainer | Collection[AgentContainer], inplace: bool = True) -> Self
        Move agents to empty cells in the space (cells where there isn't any single agent).
    move_to_available(agents: IdsLike | AgentContainer | Collection[AgentContainer], inplace: bool = True) -> Self
        Move agents to available cells in the space (cells where there is at least one spot available).
    sample_cells(n: int, cell_type: Literal["any", "empty", "available", "full"] = "any", with_replacement: bool = True) -> DataFrame
        Sample cells from the grid according to the specified cell_type.
    get_neighborhood(radius: int | float | Sequence[int] | Sequence[float], pos: DiscreteCoordinate | DiscreteCoordinates | None = None, agents: IdsLike | AgentContainer | Collection[AgentContainer] = None, include_center: bool = False) -> DataFrame
        Get the neighborhood cells from the given positions (pos) or agents according to the specified radiuses.
    get_cells(coords: DiscreteCoordinate | DiscreteCoordinates | None = None) -> DataFrame
        Retrieve a dataframe of specified cells with their properties and agents.
    set_cells(properties: DataFrame, cells: DiscreteCoordinates | None = None, inplace: bool = True) -> Self
        Set the properties of the specified cells.
    """

    _capacity: int | None  # The maximum capacity for cells (default is infinite)
    _cells: DataFrame  # Stores the properties of the cells
    _cells_col_names: list[
        str
    ]  # The column names of the _cells dataframe (eg. ['dim_0', 'dim_1', ...] in Grids, ['node_id', 'edge_id'] in Networks)
    _center_col_names: list[
        str
    ]  # The column names of the center cells/agents in the get_neighbors method (eg. ['dim_0_center', 'dim_1_center', ...] in Grids, ['node_id_center', 'edge_id_center'] in Networks)

    def __init__(
        self,
        model: "ModelDF",
        capacity: int | None = None,
    ):
        """Create a DiscreteSpaceDF object.
        NOTE: The capacity specified here is the default capacity,
        it can be set also per cell through the set_cells method.

        Parameters
        ----------
        model : ModelDF
            The model to which the space belongs
        capacity : int | None, optional
            The maximum capacity for cells, by default None (infinite)
        """
        super().__init__(model)
        self._capacity = capacity

    def is_available(self, pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame:
        """Check whether the input positions are available (there exists at least one remaining spot in the cells)

        Parameters
        ----------
        pos : GridCoordinate | GridCoordinates
            The positions to check for

        Returns
        -------
        DataFrame
            A dataframe with positions and a boolean column "available"
        """
        df = self._df_constructor(data=pos, columns=self._cells_col_names)
        return self._df_add_columns(
            df,
            ["available"],
            self._df_get_bool_mask(df, mask=self.full_cells, negate=True),
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

    def move_to_empty(
        self,
        agents: IdsLike | AgentContainer | Collection[AgentContainer],
        inplace: bool = True,
    ) -> Self:
        return self._move_agents_to_cells(agents, cell_type="empty", inplace=inplace)

    def move_to_available(
        self,
        agents: IdsLike | AgentContainer | Collection[AgentContainer],
        inplace: bool = True,
    ) -> Self:
        """Move agents to available cells/positions in the space (cells/positions where there is at least one spot available).

        Parameters
        ----------
        agents : IdsLike | AgentContainer | Collection[AgentContainer]
            The agents to move to available cells/positions
        inplace : bool, optional
            Whether to perform the operation inplace, by default True

        Returns
        -------
        Self
        """
        return self._move_agents_to_cells(
            agents, cell_type="available", inplace=inplace
        )

    def random_pos(self, n: int, seed: int | None = None) -> DataFrame | pl.DataFrame:
        return self.sample_cells(n, cell_type="any", with_replacement=True, seed=seed)

    def sample_cells(
        self,
        n: int,
        cell_type: Literal["any", "empty", "available", "full"] = "any",
        with_replacement: bool = True,
        seed: int | None = None,
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
        seed : int | None, optional
            The seed for the sampling, by default None
            If None, an integer from the model's random number generator is used.

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
        return self._sample_cells(n, with_replacement, condition=condition, seed=seed)

    @abstractmethod
    def get_neighborhood(
        self,
        radius: int | float | Sequence[int] | Sequence[float],
        pos: DiscreteCoordinate | DiscreteCoordinates | None = None,
        agents: IdsLike | AgentContainer | Collection[AgentContainer] = None,
        include_center: bool = False,
    ) -> DataFrame:
        """Get the neighborhood cells from the given positions (pos) or agents according to the specified radiuses.
        Either positions (pos) or agents must be specified, not both.

        Parameters
        ----------
        radius : int | float | Sequence[int] | Sequence[float]
            The radius(es) of the neighborhoods
        pos : DiscreteCoordinate | DiscreteCoordinates | None, optional
            The coordinates of the cell(s) to get the neighborhood from
        agents : IdsLike | AgentContainer | Collection[AgentContainer], optional
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

    @abstractmethod
    def set_cells(
        self,
        properties: DataFrame,
        cells: DiscreteCoordinates | None = None,
        inplace: bool = True,
    ) -> Self:
        """Set the properties of the specified cells.
        This method mirrors the functionality of mesa's PropertyLayer, but allows also to set properties only of specific cells.
        Either the properties DF must contain both the cell coordinates and the properties
        or the cell coordinates must be specified separately with the cells argument.
        If the Space is a Grid, the cell coordinates must be GridCoordinates.
        If the Space is a Network, the cell coordinates must be NetworkCoordinates.


        Parameters
        ----------
        properties : DataFrame
            The properties of the cells
        cells : DiscreteCoordinates | None, optional
            The coordinates of the cells to set the properties for, by default None (all cells)
        inplace : bool
            Whether to perform the operation inplace

        Returns
        -------
        Self
        """
        ...

    def _move_agents_to_cells(
        self,
        agents: IdsLike | AgentContainer | Collection[AgentContainer],
        cell_type: Literal["any", "empty", "available"],
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

        # Get cells of specified type
        cells = obj.sample_cells(len(agents), cell_type=cell_type)

        # Place agents
        obj._agents = obj.move_agents(agents, cells)
        return obj

    # We define the cell conditions here, because ruff does not allow lambda functions

    def _any_cell_condition(self, cap: DiscreteSpaceCapacity) -> BoolSeries:
        return True

    def _empty_cell_condition(self, cap: DiscreteSpaceCapacity) -> BoolSeries:
        return cap == self._capacity

    def _available_cell_condition(self, cap: DiscreteSpaceCapacity) -> BoolSeries:
        return cap > 0

    def _full_cell_condition(self, cap: DiscreteSpaceCapacity) -> BoolSeries:
        return cap == 0

    @abstractmethod
    def _sample_cells(
        self,
        n: int | None,
        with_replacement: bool,
        condition: Callable[[DiscreteSpaceCapacity], BoolSeries],
        seed: int | None = None,
    ) -> DataFrame:
        """Sample cells from the grid according to a condition on the capacity.

        Parameters
        ----------
        n : int | None
            The number of cells to sample
        with_replacement : bool
            If the sampling should be with replacement
        condition : Callable[[DiscreteSpaceCapacity], BoolSeries]
            The condition to apply on the capacity
        seed : int | None, optional
            The seed for the sampling, by default None
            If None, an integer from the model's random number generator is used.

        Returns
        -------
        DataFrame
        """
        ...

    def __getitem__(self, cells: DiscreteCoordinates):
        return self.get_cells(cells)

    def __getattr__(self, key: str) -> DataFrame:
        # Fallback, if key (property) is not found in the object,
        # then it must mean that it's in the _cells dataframe
        return self._cells[key]

    def __setitem__(self, cells: DiscreteCoordinates, properties: DataFrame):
        self.set_cells(properties=properties, cells=cells)

    def __repr__(self) -> str:
        return self._cells.__repr__()

    def __str__(self) -> str:
        return self._cells.__str__()

    @property
    def cells(self) -> DataFrame:
        return self.get_cells()

    @cells.setter
    def cells(self, df: DataFrame):
        return self.set_cells(df, inplace=True)

    @property
    def empty_cells(self) -> DataFrame:
        return self._sample_cells(
            None, with_replacement=False, condition=self._empty_cell_condition
        )

    @property
    def available_cells(self) -> DataFrame:
        return self._sample_cells(
            None, with_replacement=False, condition=self._available_cell_condition
        )

    @property
    def full_cells(self) -> DataFrame:
        return self._sample_cells(
            None, with_replacement=False, condition=self._full_cell_condition
        )


class GridDF(DiscreteSpaceDF):
    """The GridDF class is an abstract class that defines the interface for all grid classes in mesa-frames.
    Inherits from DiscreteSpaceDF.

    Warning
    -------
    In this implementation, [0, ..., 0] is the bottom-left corner and
    [dimensions[0]-1, ..., dimensions[n-1]-1] is the top-right corner, consistent with
    Cartesian coordinates and Matplotlib/Seaborn plot outputs.
    The convention is different from `np.genfromtxt`_ and its use in the
    `mesa-examples Sugarscape model`_, where [0, ..., 0] is the top-left corner
    and [dimensions[0]-1, ..., dimensions[n-1]-1] is the bottom-right corner.

    .. _np.genfromtxt: https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html
    .. _mesa-examples Sugarscape model: https://github.com/projectmesa/mesa-examples/blob/e137a60e4e2f2546901bec497e79c4a7b0cc69bb/examples/sugarscape_g1mt/sugarscape_g1mt/model.py#L93-L94


    Methods
    -------
    __init__(model: 'ModelDF', dimensions: Sequence[int], torus: bool = False, capacity: int | None = None, neighborhood_type: str = 'moore')
        Create a new GridDF object.
    out_of_bounds(pos: GridCoordinate | GridCoordinates) -> DataFrame
        Check whether the input positions are out of bounds in a non-toroidal grid.

    Properties
    ----------
    dimensions : Sequence[int]
        The dimensions of the grid
    neighborhood_type : Literal['moore', 'von_neumann', 'hexagonal']
        The type of neighborhood to consider
    torus : bool
        If the grid is a torus
    """

    _grid_capacity: (
        GridCapacity  # Storing the remaining capacity of the cells in the grid
    )
    _neighborhood_type: Literal[
        "moore", "von_neumann", "hexagonal"
    ]  # The type of neighborhood to consider
    _offsets: DataFrame  # The offsets to compute the neighborhood of a cell
    _torus: bool  # If the grid is a torus

    def __init__(
        self,
        model: "ModelDF",
        dimensions: Sequence[int],
        torus: bool = False,
        capacity: int | None = None,
        neighborhood_type: str = "moore",
    ):
        """Create a new GridDF object.

        Warning
        -------
        In this implementation, [0, ..., 0] is the bottom-left corner and
        [dimensions[0]-1, ..., dimensions[n-1]-1] is the top-right corner, consistent with
        Cartesian coordinates and Matplotlib/Seaborn plot outputs.
        The convention is different from `np.genfromtxt`_ and its use in the
        `mesa-examples Sugarscape model`_, where [0, ..., 0] is the top-left corner
        and [dimensions[0]-1, ..., dimensions[n-1]-1] is the bottom-right corner.

        .. _np.genfromtxt: https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html
        .. _mesa-examples Sugarscape model: https://github.com/projectmesa/mesa-examples/blob/e137a60e4e2f2546901bec497e79c4a7b0cc69bb/examples/sugarscape_g1mt/sugarscape_g1mt/model.py#L93-L94

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
            If 'moore', the neighborhood is the 8 cells around the center cell (up, down, left, right, and diagonals).
            If 'von_neumann', the neighborhood is the 4 cells around the center cell (up, down, left, right).
            If 'hexagonal', the neighborhood are 6 cells around the center cell distributed in a hexagonal shape.
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
        self._grid_capacity = self._generate_empty_grid(dimensions, capacity)
        self._neighborhood_type = neighborhood_type

    def get_directions(
        self,
        pos0: GridCoordinate | GridCoordinates | None = None,
        pos1: GridCoordinate | GridCoordinates | None = None,
        agents0: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
        agents1: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
        normalize: bool = False,
    ) -> DataFrame:
        result = self._calculate_differences(pos0, pos1, agents0, agents1)
        if normalize:
            result = result / self._df_norm(result)
        return result

    def get_distances(
        self,
        pos0: GridCoordinate | GridCoordinates | None = None,
        pos1: GridCoordinate | GridCoordinates | None = None,
        agents0: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
        agents1: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
    ) -> DataFrame:
        result = self._calculate_differences(pos0, pos1, agents0, agents1)
        return self._df_norm(result)

    def get_neighbors(
        self,
        radius: int | Sequence[int],
        pos: GridCoordinate | GridCoordinates | None = None,
        agents: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
        include_center: bool = False,
    ) -> DataFrame:
        neighborhood_df = self.get_neighborhood(
            radius=radius, pos=pos, agents=agents, include_center=include_center
        )
        return self._df_get_masked_df(
            df=self._agents,
            index_col="agent_id",
            mask=neighborhood_df,
            columns=self._agents.columns,
        )

    def get_cells(
        self, coords: GridCoordinate | GridCoordinates | None = None
    ) -> DataFrame:
        coords_df = self._get_df_coords(pos=coords)
        return self._df_get_masked_df(
            df=self._cells,
            index_cols=self._cells_col_names,
            mask=coords_df,
            columns=self._cells.columns,
        )

    def move_agents(
        self,
        agents: IdsLike | AgentContainer | Collection[AgentContainer],
        pos: GridCoordinate | GridCoordinates,
        inplace: bool = True,
    ) -> Self:
        obj = self._get_obj(inplace)

        # Get Ids of agents
        if isinstance(agents, AgentContainer | Collection[AgentContainer]):
            agents = agents.index

        if __debug__:
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

        # Place agents (checking that capacity is not)
        coords = obj._get_df_coords(pos)
        obj._agents = obj._place_agents_df(agents, coords)
        return obj

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
        if self._torus:
            raise ValueError("This method is only valid for non-torus grids")
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

        if __debug__:
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

    def _calculate_differences(
        self,
        pos0: GridCoordinate | GridCoordinates | None,
        pos1: GridCoordinate | GridCoordinates | None,
        agents0: IdsLike | AgentContainer | Collection[AgentContainer] | None,
        agents1: IdsLike | AgentContainer | Collection[AgentContainer] | None,
    ) -> DataFrame:
        """Calculate the differences between two positions or agents.

        Parameters
        ----------
        pos0 : GridCoordinate | GridCoordinates | None
            The starting positions
        pos1 : GridCoordinate | GridCoordinates | None
            The ending positions
        agents0 : IdsLike | AgentContainer | Collection[AgentContainer] | None
            The starting agents
        agents1 : IdsLike | AgentContainer | Collection[AgentContainer] | None
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
            even_offsets = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
            odd_offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

            # Create a DataFrame with three columns: dim_0, dim_1, and is_even
            offsets_data = [(d[0], d[1], True) for d in even_offsets] + [
                (d[0], d[1], False) for d in odd_offsets
            ]
            return self._df_constructor(
                data=offsets_data, columns=self._cells_col_names + ["is_even"]
            )
        else:
            raise ValueError("Invalid neighborhood type specified")
        return self._df_constructor(data=directions, columns=self._cells_col_names)

    def _get_df_coords(
        self,
        pos: GridCoordinate | GridCoordinates | None = None,
        agents: IdsLike | AgentContainer | Collection[AgentContainer] | None = None,
    ) -> DataFrame:
        """Get the DataFrame of coordinates from the specified positions or agents.

        Parameters
        ----------
        pos : GridCoordinate | GridCoordinates | None, optional
        agents : int | Sequence[int] | None, optional

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

    @abstractmethod
    def _generate_empty_grid(
        self, dimensions: Sequence[int], capacity: int
    ) -> GridCapacity:
        """Generate an empty grid with the specified dimensions and capacity.

        Parameters
        ----------
        dimensions : Sequence[int]

        Returns
        -------
        GridCapacity
        """
        ...

    @abstractmethod
    def _place_agents_df(self, agents: IdsLike, coords: DataFrame) -> DataFrame:
        """Place agents in the grid according to the specified coordinates.

        Parameters
        ----------
        agents : IDsLike
            The agents to place in the grid
        coords : DataFrame
            The coordinates for each agent

        Returns
        -------
        DataFrame
            A DataFrame with the agents placed in the grid
        """
        ...

    @property
    def dimensions(self) -> Sequence[int]:
        return self._dimensions

    @property
    def neighborhood_type(self) -> str:
        return self._neighborhood_type

    @property
    def torus(self) -> bool:
        return self._torus
