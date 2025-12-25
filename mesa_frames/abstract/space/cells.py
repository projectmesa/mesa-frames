"""Abstract cells interface for discrete spaces in mesa-frames."""

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
    def copy(self, space: object) -> AbstractCells:
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
    def update(
        self,
        target: DiscreteCoordinate
        | DiscreteCoordinates
        | DataFrame
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None = None,
        updates: dict[str, object] | None = None,
        *,
        mask: str | DataFrame | Series | np.ndarray | None = None,
        backend: Literal["auto", "polars"] = "auto",
        mask_col: str | None = None,
    ) -> None:
        """Update existing cell properties.

        This is the primary write API for cells.

        Parameters
        ----------
        target : DiscreteCoordinate | DiscreteCoordinates | DataFrame | AbstractAgentSet | AbstractAgentSetRegistry | Collection[AbstractAgentSet] | Collection[AbstractAgentSetRegistry] | None, optional
            Target cells to update. When provided, this selects the cells to be
            updated (by coordinates or via agents/registries). When ``target`` is
            a DataFrame, it is interpreted as a coordinate mask or a full cells
            table (when ``updates`` is ``None``).

        updates : dict[str, object] | None, optional
            Mapping of property name to update value.

            Accepted value types are:

            - scalar (int/float/bool)
            - array-like (NumPy array, list, or backend Series)
            - Polars expression (``pl.Expr``; triggers Polars fallback)
            - column name (str), interpreted as "copy values from that column"

            Callables are not accepted.

        mask : str | DataFrame | Series | np.ndarray | None, optional
            Optional selector for which cells to update. Supported string masks
            include ``"all"``, ``"empty"``, and ``"full"``.

        backend : Literal["auto", "polars"], optional
            Selects the implementation backend.

        mask_col : str | None, optional
            When ``mask`` is a DataFrame, optional name of a boolean column
            indicating the selected rows.
        """
        ...

    @abstractmethod
    def lookup(
        self,
        target: DiscreteCoordinates | DataFrame | IdsLike | np.ndarray,
        columns: list[str] | None = None,
        *,
        as_df: bool = True,
    ) -> DataFrame | dict[str, np.ndarray] | np.ndarray:
        """Fetch cell rows by key without joins."""
        ...

    @property
    @abstractmethod
    def capacity(self) -> DiscreteSpaceCapacity: ...

    @capacity.setter
    @abstractmethod
    def capacity(self, cap: DiscreteSpaceCapacity) -> None: ...

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
    def is_available(
        self, pos: DiscreteCoordinate | DiscreteCoordinates
    ) -> DataFrame: ...

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

    def update_capacity_cells(self, cells: DataFrame) -> DiscreteSpaceCapacity:
        """Public wrapper around internal capacity updates.

        This is intentionally public so other components (e.g. grids) can
        request capacity recomputation without reaching into protected APIs.
        """
        return self._update_capacity_cells(cells)

    def update_capacity_agents(
        self, agents: DataFrame | Series, operation: Literal["movement", "removal"]
    ) -> DiscreteSpaceCapacity:
        """Public wrapper around internal capacity updates.

        This is intentionally public so other components (e.g. grids) can
        update capacity without reaching into protected APIs.
        """
        return self._update_capacity_agents(agents, operation=operation)

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
