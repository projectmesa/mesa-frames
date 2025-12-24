"""Abstract grid interface.

This module defines the *interface only* for grid-like spaces.
Concrete behavior lives in concrete implementations (e.g. the Polars-backed
:class:`mesa_frames.concrete.space.grid.Grid`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from typing import Literal, Self

from mesa_frames.abstract.agentset import AbstractAgentSet
from mesa_frames.abstract.agentsetregistry import AbstractAgentSetRegistry
from mesa_frames.abstract.space.cells import AbstractCells
from mesa_frames.abstract.space.discrete import AbstractDiscreteSpace
from mesa_frames.abstract.space.neighborhood import AbstractNeighborhood
from mesa_frames.types_ import DataFrame, GridCoordinate, GridCoordinates, IdsLike


class AbstractGrid(AbstractDiscreteSpace, ABC):
    """Interface for grid-based discrete spaces."""

    @property
    @abstractmethod
    def cells(self) -> AbstractCells: ...

    @cells.setter
    @abstractmethod
    def cells(self, cells: AbstractCells) -> None: ...

    @property
    @abstractmethod
    def neighborhood(self) -> AbstractNeighborhood: ...

    @neighborhood.setter
    @abstractmethod
    def neighborhood(self, neighborhood: AbstractNeighborhood) -> None: ...

    @abstractmethod
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
    ) -> DataFrame: ...

    @abstractmethod
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
    ) -> DataFrame: ...

    @abstractmethod
    def out_of_bounds(self, pos: GridCoordinate | GridCoordinates) -> DataFrame: ...

    @abstractmethod
    def remove_agents(
        self,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        inplace: bool = True,
    ) -> Self: ...

    @abstractmethod
    def move_all(
        self,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        pos: GridCoordinate | GridCoordinates,
        inplace: bool = True,
    ) -> Self: ...

    @abstractmethod
    def torus_adj(self, pos: GridCoordinate | GridCoordinates) -> DataFrame: ...

    @property
    @abstractmethod
    def dimensions(self) -> Sequence[int]: ...

    @property
    @abstractmethod
    def neighborhood_type(self) -> Literal["moore", "von_neumann", "hexagonal"]: ...

    @property
    @abstractmethod
    def torus(self) -> bool: ...
