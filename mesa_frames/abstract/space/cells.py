"""
Abstract cells interface for discrete spaces in mesa-frames.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection
from typing import Literal

import numpy as np

from mesa_frames.abstract.agentset import AbstractAgentSet
from mesa_frames.abstract.agentsetregistry import AbstractAgentSetRegistry
from mesa_frames.types_ import (
    BoolSeries,
    DataFrame,
    DataFrameInput,
    DiscreteCoordinate,
    DiscreteCoordinates,
    DiscreteSpaceCapacity,
    IdsLike,
    Infinity,
    Series,
)


class AbstractCells(ABC):
    """Abstract interface for cell storage and queries."""

    def __init__(self, space: object) -> None:
        self._space = space

    @abstractmethod
    def copy(self, space: object) -> "AbstractCells":
        """Return a copy of the cells bound to a new space."""

    @abstractmethod
    def __call__(
        self,
        target: DiscreteCoordinate
        | DiscreteCoordinates
        | DataFrame
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None = None,
        *,
        include: Literal["properties", "agents", "both"] = "both",
    ) -> DataFrame: ...

    @abstractmethod
    def set(
        self,
        target: DiscreteCoordinate
        | DiscreteCoordinates
        | DataFrame
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        properties: DataFrame | DataFrameInput | None = None,
        *,
        inplace: bool = True,
    ) -> object: ...

    @property
    @abstractmethod
    def capacity(self) -> DiscreteSpaceCapacity: ...

    @property
    @abstractmethod
    def remaining_capacity(self) -> int | Infinity: ...

    @property
    @abstractmethod
    def empty(self) -> DataFrame: ...

    @property
    @abstractmethod
    def available(self) -> DataFrame: ...

    @property
    @abstractmethod
    def full(self) -> DataFrame: ...

    @abstractmethod
    def sample(
        self,
        n: int,
        cell_type: Literal["any", "empty", "available", "full"] = "any",
        *,
        with_replacement: bool = True,
        respect_capacity: bool = True,
    ) -> DataFrame: ...

    @abstractmethod
    def is_available(self, pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame: ...

    @abstractmethod
    def is_empty(self, pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame: ...

    @abstractmethod
    def is_full(self, pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame: ...

    @abstractmethod
    def _update_capacity_cells(self, cells: DataFrame) -> DiscreteSpaceCapacity: ...

    @abstractmethod
    def _update_capacity_agents(
        self, agents: DataFrame | Series, operation: Literal["movement", "removal"]
    ) -> DiscreteSpaceCapacity: ...

    @abstractmethod
    def _empty_cell_condition(
        self, cap: DiscreteSpaceCapacity
    ) -> BoolSeries | np.ndarray: ...

    @abstractmethod
    def _sample_cells(
        self,
        n: int | None,
        with_replacement: bool,
        condition: Callable[[DiscreteSpaceCapacity], BoolSeries | np.ndarray],
        respect_capacity: bool = True,
    ) -> DataFrame: ...
