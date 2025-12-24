"""Abstract discrete space interface."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Collection, Sequence
from typing import Literal, Self

from mesa_frames.abstract.agentset import AbstractAgentSet
from mesa_frames.abstract.agentsetregistry import AbstractAgentSetRegistry
from mesa_frames.abstract.space.cells import AbstractCells
from mesa_frames.abstract.space.space import Space
from mesa_frames.types_ import (
    DataFrame,
    DiscreteCoordinate,
    DiscreteCoordinates,
    IdsLike,
    Series,
)


class AbstractDiscreteSpace(Space):
    """The AbstractDiscreteSpace class is an abstract class that defines the interface for all discrete space classes (Grids and Networks) in mesa_frames."""

    _agents: DataFrame
    _capacity: int | None  # The maximum capacity for cells (default is infinite)
    _cells_obj: AbstractCells

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

    def copy(
        self,
        deep: bool = False,
        memo: dict | None = None,
        skip: list[str] | None = None,
    ) -> Self:
        skip_list = list(skip or [])
        skip_list.append("_cells_obj")
        skip_list.append("_neighborhood_obj")
        obj = super().copy(deep=deep, memo=memo, skip=skip_list)
        obj._cells_obj = self._cells_obj.copy(obj)
        obj._neighborhood_obj = self._neighborhood_obj.copy(obj)
        return obj

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
        cells = self.cells.sample(len(agents), cell_type=cell_type)

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
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None = None,
    ) -> DataFrame:
        """Get the DataFrame of coordinates from the specified positions or agents.

        Parameters
        ----------
        pos : DiscreteCoordinate | DiscreteCoordinates | None, optional
            The positions to get the DataFrame from, by default None
        agents : IdsLike | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry] | None, optional
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

    @property
    @abstractmethod
    def cells(self) -> AbstractCells:
        """Access cell data via a unified get/set interface."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}\n{str(self.cells())}"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}\n{str(self.cells())}"
