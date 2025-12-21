"""
Abstract neighborhood interface for discrete spaces in mesa-frames.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Literal

from mesa_frames.abstract.agentset import AbstractAgentSet
from mesa_frames.abstract.agentsetregistry import AbstractAgentSetRegistry
from mesa_frames.types_ import (
    ArrayLike,
    DataFrame,
    DiscreteCoordinate,
    DiscreteCoordinates,
    IdsLike,
)


class AbstractNeighborhood(ABC):
    """Abstract interface for neighborhood queries."""

    def __init__(self, space: object) -> None:
        self._space = space

    @abstractmethod
    def copy(self, space: object) -> "AbstractNeighborhood":
        """Return a copy of the neighborhood accessor bound to a new space."""

    @abstractmethod
    def __call__(
        self,
        radius: int | float | Collection[int] | Collection[float] | ArrayLike,
        target: DiscreteCoordinate
        | DiscreteCoordinates
        | DataFrame
        | IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
        *,
        include: Literal["coords", "agents", "both"] = "coords",
        include_center: bool = False,
    ) -> DataFrame: ...
