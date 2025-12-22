"""
Concrete cells implementation for polars-backed grids.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from typing import Any, Literal

import numpy as np
import polars as pl

from mesa_frames.abstract.agentset import AbstractAgentSet
from mesa_frames.abstract.agentsetregistry import AbstractAgentSetRegistry
from mesa_frames.abstract.space import AbstractDiscreteSpace, AbstractGrid
from mesa_frames.abstract.space.cells import AbstractCells
from mesa_frames.types_ import (
    BoolSeries,
    DataFrame,
    DataFrameInput,
    DiscreteCoordinate,
    DiscreteCoordinates,
    DiscreteSpaceCapacity,
    Infinity,
    Series,
)


class GridCells(AbstractCells):
    """Polars-based cells implementation for grids."""

    _cells: pl.DataFrame
    _cells_capacity: np.ndarray

    def __init__(self, space: AbstractGrid) -> None:
        super().__init__(space)
        cells_df_dtypes = {col: int for col in self._space._pos_col_names}
        cells_df_dtypes.update(
            {"capacity": float}  # Capacity can be float if we want to represent np.nan
        )
        self._cells = self._space._df_constructor(
            columns=self._space._pos_col_names + ["capacity"],
            index_cols=self._space._pos_col_names,
            dtypes=cells_df_dtypes,
        )
        self._cells_capacity = self._generate_empty_grid(
            self._space._dimensions, self._space._capacity
        )

    @property
    def coords(self) -> pl.DataFrame:
        """Return a DataFrame of cell coordinates.

        This is the canonical public accessor for cell positions.
        """
        return self._cells.select(self._space._pos_col_names)

    def _cell_id_from_coords(self, coords: np.ndarray) -> np.ndarray:
        """Convert coordinate rows to flat cell ids (row-major)."""
        if coords.ndim != 2 or coords.shape[1] != len(self._space._dimensions):
            raise ValueError("coords must be a (n, ndim) array")
        dims = self._space._dimensions
        if len(dims) == 2:
            height = int(dims[1])
            return coords[:, 0].astype(np.int64, copy=False) * height + coords[:, 1].astype(
                np.int64, copy=False
            )
        return np.ravel_multi_index(coords.T, dims)

    def _coords_from_cell_id(self, cell_id: np.ndarray) -> np.ndarray:
        """Convert flat cell ids (row-major) to coordinate rows."""
        dims = self._space._dimensions
        if len(dims) == 2:
            height = int(dims[1])
            dim0 = (cell_id // height).astype(np.int64, copy=False)
            dim1 = (cell_id % height).astype(np.int64, copy=False)
            return np.stack([dim0, dim1], axis=1)
        coords = np.array(np.unravel_index(cell_id, dims)).T
        return coords.astype(np.int64, copy=False)

    def _ensure_dense_row_major_cells(self) -> None:
        """Ensure dense grid cells are stored in row-major coordinate order."""
        space = self._space
        if getattr(space, "_cells_row_major_ok", False):
            return
        if self._cells.is_empty():
            return
        if self._cells.height != int(np.prod(space._dimensions)):
            return
        # Sort into canonical (dim_0, dim_1, ...) order once and cache the fact.
        self._cells = self._cells.sort(space._pos_col_names)
        setattr(space, "_cells_row_major_ok", True)

    def copy(self, space: AbstractGrid) -> "GridCells":
        obj = self.__class__(space)
        if isinstance(self._cells, pl.DataFrame):
            obj._cells = self._cells.clone()
        else:
            obj._cells = space._df_constructor(self._cells)
        obj._cells_capacity = self._cells_capacity.copy()
        return obj

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
    ) -> DataFrame:
        if include not in {"properties", "agents", "both"}:
            raise ValueError("include must be one of: properties, agents, both")

        coords, agents = self._split_target(target)
        space = self._space

        if coords is None and agents is None:
            if include == "properties":
                return self._cells
            if include == "agents":
                return space._agents
            return space._df_join(
                left=self._cells,
                right=space._agents,
                index_cols=space._pos_col_names,
                on=space._pos_col_names,
            )

        if include == "agents":
            if agents is not None:
                agent_ids = space._get_ids_srs(agents)
                df = space._df_get_masked_df(
                    space._agents, index_cols="agent_id", mask=agent_ids
                )
                return space._df_reindex(df, agent_ids, "agent_id")
            coords_df = space._get_df_coords(pos=coords)
            return space._df_get_masked_df(
                df=space._agents,
                index_cols=space._pos_col_names,
                mask=coords_df,
            )

        coords_df = space._get_df_coords(pos=coords, agents=agents)
        cells_df = space._df_get_masked_df(
            df=self._cells, index_cols=space._pos_col_names, mask=coords_df
        )
        if include == "properties":
            return cells_df
        return space._df_join(
            left=cells_df,
            right=space._agents,
            index_cols=space._pos_col_names,
            on=space._pos_col_names,
        )

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
    ) -> AbstractDiscreteSpace:
        coords, agents = self._split_target(target)
        obj = self._space._get_obj(inplace)
        cells_obj = obj.cells

        if agents is not None:
            coords = obj._get_df_coords(agents=agents)

        if isinstance(coords, DataFrame):
            cells_df = coords
        else:
            cells_df = obj._get_df_coords(coords)
        cells_df = obj._df_set_index(cells_df, index_name=obj._pos_col_names)

        if __debug__:
            if isinstance(cells_df, DataFrame) and any(
                k not in obj._df_column_names(cells_df) for k in obj._pos_col_names
            ):
                raise ValueError(
                    f"The cells DataFrame must have the columns {obj._pos_col_names}"
                )

        if properties:
            cells_df = obj._df_constructor(
                data=properties, index=obj._df_index(cells_df, obj._pos_col_names)
            )

        if "capacity" in obj._df_column_names(cells_df):
            cells_obj._cells_capacity = cells_obj._update_capacity_cells(cells_df)

        if isinstance(cells_obj._cells, pl.DataFrame) and isinstance(
            cells_df, pl.DataFrame
        ):
            if cells_obj._cells.is_empty():
                cells_obj._cells = cells_df
                try:
                    object.__delattr__(obj, "_cells_row_major_ok")
                except AttributeError:
                    pass
                return obj
            update_cols = [c for c in cells_df.columns if c not in obj._pos_col_names]
            if update_cols:
                is_dense_grid = cells_obj._cells.height == int(
                    np.prod(obj._dimensions)
                )
                if is_dense_grid:
                    # Delta-aware update: for small updates on dense grids, avoid a join.
                    # We rely on a stable row-major layout and update only the touched rows.
                    n_updates = int(cells_df.height)
                    n_total = int(cells_obj._cells.height)
                    if n_updates and n_updates <= max(1, int(0.05 * n_total)):
                        cells_obj._ensure_dense_row_major_cells()
                        coords_np = cells_df.select(obj._pos_col_names).to_numpy()
                        row_idx = cells_obj._cell_id_from_coords(coords_np)

                        updated = cells_obj._cells
                        for col in update_cols:
                            values = cells_df[col]
                            if col in updated.columns:
                                base = updated[col]
                            else:
                                # Create a null base column for new properties.
                                base_expr = pl.repeat(None, n_total, eager=True)
                                if values.dtype != pl.Null:
                                    base_expr = base_expr.cast(values.dtype)
                                updated = updated.with_columns(base_expr.alias(col))
                                base = updated[col]

                            scattered = base.scatter(row_idx, values)
                            updated = updated.with_columns(scattered.alias(col))

                        cells_obj._cells = updated
                        return obj

                    updates = cells_df.select([*obj._pos_col_names, *update_cols])
                    merged = cells_obj._cells.join(
                        updates,
                        on=obj._pos_col_names,
                        how="left",
                        suffix="_new",
                        maintain_order="left",
                    )
                    existing_cols = set(cells_obj._cells.columns)
                    coalesce_exprs: list[pl.Expr] = []
                    drop_cols: list[str] = []
                    for c in update_cols:
                        if c in existing_cols:
                            new_c = f"{c}_new"
                            coalesce_exprs.append(
                                pl.coalesce(pl.col(new_c), pl.col(c)).alias(c)
                            )
                            drop_cols.append(new_c)
                    if coalesce_exprs:
                        merged = merged.with_columns(coalesce_exprs).drop(drop_cols)
                    cells_obj._cells = merged
                else:
                    cells_obj._cells = obj._df_combine_first(
                        cells_df, cells_obj._cells, index_cols=obj._pos_col_names
                    )
                    try:
                        object.__delattr__(obj, "_cells_row_major_ok")
                    except AttributeError:
                        pass
        else:
            cells_obj._cells = obj._df_combine_first(
                cells_df, cells_obj._cells, index_cols=obj._pos_col_names
            )
            try:
                object.__delattr__(obj, "_cells_row_major_ok")
            except AttributeError:
                pass
        return obj

    @property
    def capacity(self) -> DiscreteSpaceCapacity:
        return self._cells_capacity

    @capacity.setter
    def capacity(self, cap: DiscreteSpaceCapacity) -> None:
        # GridCells stores capacity as a dense ndarray.
        if cap is np.inf:
            raise ValueError("Grid cells capacity cannot be infinity")
        if not isinstance(cap, np.ndarray):
            raise TypeError("capacity must be a numpy ndarray")
        self._cells_capacity = cap

    @property
    def remaining_capacity(self) -> int | Infinity:
        if not self._space._capacity:
            return np.inf
        return int(self._cells_capacity.sum())

    def is_available(self, pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame:
        return self._check(pos, "available")

    def is_empty(self, pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame:
        return self._check(pos, "empty")

    def is_full(self, pos: DiscreteCoordinate | DiscreteCoordinates) -> DataFrame:
        return self._check(pos, "full")

    def _check(
        self,
        pos: DiscreteCoordinate | DiscreteCoordinates,
        state: Literal["empty", "full", "available"],
    ) -> DataFrame:
        pos_df = self._space._get_df_coords(pos)

        if state == "empty":
            mask = self.empty
        elif state == "full":
            mask = self.full
        else:
            mask = self.available

        return self._space._df_with_columns(
            original_df=pos_df,
            data=self._space._df_get_bool_mask(
                pos_df,
                index_cols=self._space._pos_col_names,
                mask=mask,
            ),
            new_columns=state,
        )

    @property
    def empty(self) -> DataFrame:
        return self._sample_cells(
            None, with_replacement=False, condition=self._empty_cell_condition
        )

    @property
    def available(self) -> DataFrame:
        return self._sample_cells(
            None,
            with_replacement=False,
            condition=self._available_cell_condition,
        )

    @property
    def full(self) -> DataFrame:
        return self._sample_cells(
            None,
            with_replacement=False,
            condition=self._full_cell_condition,
            respect_capacity=False,
        )

    def sample(
        self,
        n: int,
        cell_type: Literal["any", "empty", "available", "full"] = "any",
        *,
        with_replacement: bool = True,
        respect_capacity: bool = True,
    ) -> DataFrame:
        match cell_type:
            case "any":
                condition = self._any_cell_condition
            case "empty":
                condition = self._empty_cell_condition
            case "available":
                condition = self._available_cell_condition
            case "full":
                condition = self._full_cell_condition
                respect_capacity = False
        return self._sample_cells(
            n,
            with_replacement,
            condition=condition,
            respect_capacity=respect_capacity,
        )

    def _any_cell_condition(
        self, cap: DiscreteSpaceCapacity
    ) -> BoolSeries | np.ndarray:
        return cap

    def _available_cell_condition(
        self, cap: DiscreteSpaceCapacity
    ) -> BoolSeries | np.ndarray:
        return cap > 0

    def _full_cell_condition(
        self, cap: DiscreteSpaceCapacity
    ) -> BoolSeries | np.ndarray:
        return cap == 0

    def _empty_cell_condition(
        self, cap: DiscreteSpaceCapacity
    ) -> BoolSeries | np.ndarray:
        empty_mask = np.ones_like(cap, dtype=bool)

        if not self._space._agents.is_empty():
            agent_coords = self._space._agents[self._space._pos_col_names].to_numpy()
            empty_mask[tuple(agent_coords.T)] = False

        return empty_mask

    def _sample_cells(
        self,
        n: int | None,
        with_replacement: bool,
        condition: Callable[[DiscreteSpaceCapacity], BoolSeries | np.ndarray],
        respect_capacity: bool = True,
    ) -> DataFrame:
        coords = np.array(np.where(condition(self._cells_capacity))).T

        if respect_capacity:
            capacities = self._cells_capacity[tuple(coords.T)]
        else:
            capacities = np.ones(len(coords), dtype=int)

        if n is not None:
            if with_replacement:
                if respect_capacity:
                    assert n <= capacities.sum(), (
                        "Requested sample size exceeds the total available capacity."
                    )

                sampled_coords = np.empty((0, coords.shape[1]), dtype=coords.dtype)
                while len(sampled_coords) < n:
                    remaining_samples = n - len(sampled_coords)
                    sampled_indices = self._space.random.choice(
                        len(coords),
                        size=remaining_samples,
                        replace=True,
                    )
                    unique_indices, counts = np.unique(
                        sampled_indices, return_counts=True
                    )

                    if respect_capacity:
                        valid_counts = np.minimum(counts, capacities[unique_indices])
                        capacities[unique_indices] -= valid_counts
                    else:
                        valid_counts = counts

                    new_coords = np.repeat(coords[unique_indices], valid_counts, axis=0)
                    sampled_coords = np.vstack((sampled_coords, new_coords))

                    if respect_capacity:
                        mask = capacities > 0
                        coords = coords[mask]
                        capacities = capacities[mask]

                sampled_coords = sampled_coords[:n]
                self._space.random.shuffle(sampled_coords)
            else:
                assert n <= len(coords), (
                    "Requested sample size exceeds the number of available cells."
                )
                sampled_indices = self._space.random.choice(
                    len(coords), size=n, replace=False
                )
                sampled_coords = coords[sampled_indices]
        else:
            sampled_coords = coords

        sampled_cells = pl.DataFrame(
            sampled_coords, schema=self._space._pos_col_names, orient="row"
        )
        return sampled_cells

    def _update_capacity_agents(
        self,
        agents: DataFrame | Series,
        operation: Literal["movement", "removal"],
    ) -> np.ndarray:
        masked_df: pl.DataFrame
        if (
            operation == "movement"
            and isinstance(agents, pl.DataFrame)
            and (not self._space._agents.is_empty())
            and agents.height == self._space._agents.height
            and "agent_id" in agents.columns
            and agents["agent_id"].n_unique() == agents.height
            and bool(agents["agent_id"].is_in(self._space._agents["agent_id"]).all())
        ):
            masked_df = self._space._agents
        else:
            masked_df = self._space._df_get_masked_df(
                self._space._agents, index_cols="agent_id", mask=agents
            )

        if operation == "movement":
            old_positions = tuple(masked_df[self._space._pos_col_names].to_numpy().T)
            np.add.at(self._cells_capacity, old_positions, 1)

            new_positions = tuple(agents[self._space._pos_col_names].to_numpy().T)
            np.add.at(self._cells_capacity, new_positions, -1)
        elif operation == "removal":
            positions = tuple(masked_df[self._space._pos_col_names].to_numpy().T)
            np.add.at(self._cells_capacity, positions, 1)
        return self._cells_capacity

    def _update_capacity_cells(self, cells: DataFrame) -> np.ndarray:
        coords = cells[self._space._pos_col_names]

        current_capacity = (
            coords.join(self._cells, on=self._space._pos_col_names, how="left")
            .fill_null(self._space._capacity)["capacity"]
            .to_numpy()
        )

        agents_in_cells = (
            current_capacity - self._cells_capacity[tuple(zip(*coords.to_numpy()))]
        )

        new_capacity = cells["capacity"].to_numpy() - agents_in_cells

        assert np.all(new_capacity >= 0), (
            "New capacity of a cell cannot be less than the number of agents in it."
        )

        self._cells_capacity[tuple(zip(*coords.to_numpy()))] = new_capacity

        return self._cells_capacity

    def _generate_empty_grid(
        self, dimensions: Sequence[int], capacity: int | None
    ) -> np.ndarray:
        if not capacity:
            capacity = np.inf
        return np.full(dimensions, capacity)

    def _split_target(
        self,
        target: DiscreteCoordinate
        | DiscreteCoordinates
        | DataFrame
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | None,
    ) -> tuple[DiscreteCoordinate | DiscreteCoordinates | DataFrame | None, Any | None]:
        if target is None:
            return None, None
        if isinstance(target, DataFrame):
            return target, None
        if isinstance(target, (AbstractAgentSet, AbstractAgentSetRegistry)):
            return None, target
        if isinstance(target, Collection):
            if not target:
                return None, target
            first = next(iter(target))
            if isinstance(first, (AbstractAgentSet, AbstractAgentSetRegistry)):
                return None, target
        return target, None
