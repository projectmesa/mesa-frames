from abc import abstractmethod
from collections.abc import Callable, Collection, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

import polars as pl
from numpy.random import Generator
from typing_extensions import Self

from typing import Literal

from mesa_frames.abstract.agents import AgentContainer
from mesa_frames.abstract.mixin import CopyMixin, DataFrameMixin
from mesa_frames.types_ import (
    BoolSeries,
    DataFrame,
    DiscreteCoordinate,
    DiscreteCoordinates,
    DiscreteSpaceCapacity,
    GeoDataFrame,
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
    is_free(pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame
        Check whether the input positions are free (there exists at least one remaining spot in the cells).
    is_empty(pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame
        Check whether the input positions are empty (there isn't any single agent in the cells).
    is_full(pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame
        Check whether the input positions are full (there isn't any spot available in the cells).
    move_to_empty(agents: IdsLike | AgentContainer | Collection[AgentContainer], inplace: bool = True) -> Self
        Move agents to empty cells in the space (cells where there isn't any single agent).
    move_to_free(agents: IdsLike | AgentContainer | Collection[AgentContainer], inplace: bool = True) -> Self
        Move agents to free cells in the space (cells where there is at least one spot available).
    sample_cells(n: int, cell_type: Literal["any", "empty", "free", "full"] = "any", with_replacement: bool = True) -> DataFrame
        Sample cells from the grid according to the specified cell_type.
    get_neighborhood(radius: int | float | Sequence[int] | Sequence[float], pos: DiscreteCoordinate | Discrete
        Get the neighborhood cells from a given position.
    get_cells(cells: DiscreteCoordinates | None = None) -> DataFrame
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

    def move_to_empty(
        self,
        agents: IdsLike | AgentContainer | Collection[AgentContainer],
        inplace: bool = True,
    ) -> Self:
        return self._move_agents_to_cells(agents, cell_type="empty", inplace=inplace)

    def move_to_free(
        self,
        agents: IdsLike | AgentContainer | Collection[AgentContainer],
        inplace: bool = True,
    ) -> Self:
        """Move agents to free cells/positions in the space (cells/positions where there is at least one spot available).

        Parameters
        ----------
        agents : IdsLike | AgentContainer | Collection[AgentContainer]
            The agents to move to free cells/positions
        inplace : bool, optional
            Whether to perform the operation inplace, by default True

        Returns
        -------
        Self
        """
        return self._move_agents_to_cells(agents, cell_type="free", inplace=inplace)

    def sample_cells(
        self,
        n: int,
        cell_type: Literal["any", "empty", "free", "full"] = "any",
        with_replacement: bool = True,
    ) -> DataFrame:
        """Sample cells from the grid according to the specified cell_type.

        Parameters
        ----------
        n : int
            The number of cells to sample
        cell_type : Literal["any", "empty", "free", "full"], optional
            The type of cells to sample, by default "any"
        with_replacement : bool, optional
            If the sampling should be with replacement, by default True

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
            case "free":
                condition = self._free_cell_condition
            case "full":
                condition = self._full_cell_condition
        return self._sample_cells(n, with_replacement, condition=condition)

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
        """Retrieve a dataframe of specified cells with their properties and agents.

        Parameters
        ----------
        cells : CellCoordinates, default is optional (all cells retrieved)

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

    def _move_agents_to_cells(
        self,
        agents: IdsLike | AgentContainer | Collection[AgentContainer],
        cell_type: Literal["empty", "free"],
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

    def _free_cell_condition(self, cap: DiscreteSpaceCapacity) -> BoolSeries:
        return cap > 0

    def _full_cell_condition(self, cap: DiscreteSpaceCapacity) -> BoolSeries:
        return cap == 0

    @abstractmethod
    def _sample_cells(
        self,
        n: int | None,
        with_replacement: bool,
        condition: Callable[[DiscreteSpaceCapacity], BoolSeries],
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
        # Fallback, if key (property) is not found in the object,
        # then it must mean that it's in the _cells dataframe
        return self._cells[key]

    # We use lru_cache because cached_property does not support a custom setter.
    # It should improve performance if cell properties haven't changed between accesses.
    # TODO: Test if there's an effective increase in performance

    @property
    @lru_cache(maxsize=1)
    def cells(self) -> DataFrame:
        return self.get_cells()

    @cells.setter
    def cells(self, df: DataFrame):
        return self.set_cells(df, inplace=True)

    @property
    @lru_cache(maxsize=1)
    def empty_cells(self) -> DataFrame:
        return self._sample_cells(
            None, with_replacement=False, condition=self._empty_cell_condition
        )

    @property
    @lru_cache(maxsize=1)
    def free_cells(self) -> DataFrame:
        return self._sample_cells(
            None, with_replacement=False, condition=self._free_cell_condition
        )

    @property
    @lru_cache(maxsize=1)
    def full_cells(self) -> DataFrame:
        return self._sample_cells(
            None, with_replacement=False, condition=self._full_cell_condition
        )
