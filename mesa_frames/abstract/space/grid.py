"""Abstract grid interface."""

from __future__ import annotations

from collections.abc import Collection, Sequence, Sized
from itertools import product
from typing import Literal, Self
from warnings import warn

import numpy as np

from mesa_frames.abstract.agentset import AbstractAgentSet
from mesa_frames.abstract.agentsetregistry import AbstractAgentSetRegistry
from mesa_frames.abstract.space.cells import AbstractCells
from mesa_frames.abstract.space.neighborhood import AbstractNeighborhood
from mesa_frames.abstract.space.discrete import AbstractDiscreteSpace
from mesa_frames.types_ import (
    ArrayLike,
    DataFrame,
    DiscreteCoordinate,
    DiscreteCoordinates,
    GridCoordinate,
    GridCoordinates,
    IdsLike,
    Series,
)


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
        self._offsets = self._compute_offsets(neighborhood_type)
        self._neighborhood_type = neighborhood_type

    @property
    def cells(self) -> AbstractCells:
        return self._cells_obj

    @cells.setter
    def cells(self, cells: AbstractCells) -> None:
        self._cells_obj = cells

    @property
    def neighborhood(self) -> AbstractNeighborhood:
        return self._neighborhood_obj

    @neighborhood.setter
    def neighborhood(self, neighborhood: AbstractNeighborhood) -> None:
        self._neighborhood_obj = neighborhood

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
        obj.cells._update_capacity_agents(agents, operation="removal")

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
        agents_ids = None
        if agents is not None:
            agents_ids = self._get_ids_srs(agents)

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
                # Check ids presence in model
                b_contained = agents_ids.is_in(self.model.sets.ids)
                if (isinstance(b_contained, Series) and not b_contained.all()) or (
                    isinstance(b_contained, bool) and not b_contained
                ):
                    raise ValueError("Some agents are not present in the model")

                # Check ids presence in the grid
                b_contained = self._df_contains(self._agents, "agent_id", agents_ids)
                if (isinstance(b_contained, Series) and not b_contained.all()) or (
                    isinstance(b_contained, bool) and not b_contained
                ):
                    raise ValueError("Some agents are not placed in the grid")
                # Check ids are unique
                if agents_ids.n_unique() != len(agents_ids):
                    raise ValueError("Some agents are present multiple times")

        if agents_ids is not None:
            df = self._df_get_masked_df(
                self._agents, index_cols="agent_id", mask=agents_ids
            )
            df = self._df_reindex(df, agents_ids, "agent_id")
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
                if len(agents) > self.cells.remaining_capacity + len(
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
        self.cells._update_capacity_agents(new_df, operation="movement")
        full_move = (
            is_move
            and (not self._agents.is_empty())
            and new_df.height == self._agents.height
            and agents.n_unique() == len(agents)
        )
        if full_move:
            full_move = bool(agents.is_in(self._agents["agent_id"]).all())

        if full_move:
            self._agents = new_df
        else:
            self._agents = self._df_combine_first(
                new_df, self._agents, index_cols="agent_id"
            )
        return self

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
