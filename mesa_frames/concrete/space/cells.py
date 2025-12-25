"""Concrete cells implementation for polars-backed grids."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from typing import Any, Literal

import numpy as np
import polars as pl

from mesa_frames.abstract.agentset import AbstractAgentSet
from mesa_frames.abstract.agentsetregistry import AbstractAgentSetRegistry
from mesa_frames.abstract.space import AbstractGrid
from mesa_frames.abstract.space.cells import AbstractCells
from mesa_frames.concrete._update_masked import _MaskedUpdateMixin
from mesa_frames.types_ import (
    BoolSeries,
    DataFrame,
    DiscreteCoordinate,
    DiscreteCoordinates,
    DiscreteSpaceCapacity,
    Infinity,
    Series,
)


class GridCells(_MaskedUpdateMixin, AbstractCells):
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
            return coords[:, 0].astype(np.int64, copy=False) * height + coords[
                :, 1
            ].astype(np.int64, copy=False)
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

    def _property_buffer(self, name: str) -> np.ndarray:
        """Return a dense row-major NumPy view of a cell property.

        This is an internal helper used by fast movement paths. It intentionally
        does not cache/buffer values; callers can memoize externally if needed.
        """
        self._ensure_dense_row_major_cells()
        if name not in self._cells.columns:
            raise KeyError(f"Unknown cell property: {name}")
        return self._cells[name].to_numpy()

    def _ensure_dense_row_major_cells(self) -> None:
        """Ensure dense grid cells are stored in row-major coordinate order."""
        space = self._space
        if getattr(space, "_cells_row_major_ok", False):
            return
        n_total = int(np.prod(space._dimensions))
        if self._cells.is_empty():
            # Initialize a dense, row-major base table on first use.
            coords = (
                np.stack(np.indices(space._dimensions), axis=-1)
                .reshape(-1, len(space._dimensions))
                .astype(np.int64, copy=False)
            )
            base = pl.DataFrame(coords, schema=space._pos_col_names, orient="row")
            cap = getattr(space, "_capacity", None)
            if cap is None:
                base = base.with_columns(
                    pl.lit(np.nan).cast(pl.Float64).alias("capacity")
                )
            else:
                base = base.with_columns(
                    pl.lit(float(cap)).cast(pl.Float64).alias("capacity")
                )
            self._cells = base
            setattr(space, "_cells_row_major_ok", True)
            return
        if self._cells.height != n_total:
            # Rebuild a dense, row-major table once (coords left-join preserves order).
            coords = (
                np.stack(np.indices(space._dimensions), axis=-1)
                .reshape(-1, len(space._dimensions))
                .astype(np.int64, copy=False)
            )
            base = pl.DataFrame(coords, schema=space._pos_col_names, orient="row")
            cap = getattr(space, "_capacity", None)
            if cap is None:
                base = base.with_columns(
                    pl.lit(np.nan).cast(pl.Float64).alias("capacity")
                )
            else:
                base = base.with_columns(
                    pl.lit(float(cap)).cast(pl.Float64).alias("capacity")
                )
            joined = base.join(
                self._cells,
                on=space._pos_col_names,
                how="left",
                maintain_order="left",
                suffix="_old",
            )
            if "capacity_old" in joined.columns:
                joined = joined.with_columns(
                    pl.coalesce(pl.col("capacity_old"), pl.col("capacity")).alias(
                        "capacity"
                    )
                ).drop("capacity_old")
            self._cells = joined
        else:
            # Sort into canonical (dim_0, dim_1, ...) order once and cache the fact.
            self._cells = self._cells.sort(space._pos_col_names)
        setattr(space, "_cells_row_major_ok", True)

    def copy(self, space: AbstractGrid) -> GridCells:
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

    def update(
        self,
        target: DiscreteCoordinate
        | DiscreteCoordinates
        | DataFrame
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry]
        | dict[str, object]
        | None = None,
        updates: dict[str, object] | None = None,
        *,
        mask: str | DataFrame | Series | np.ndarray | None = None,
        backend: Literal["auto", "polars"] = "auto",
        mask_col: str | None = None,
    ) -> None:
        """Update cell properties.

        See the abstract interface for accepted values and masks.
        """
        if updates is None and isinstance(target, dict):
            updates = target
            target = None

        if updates is not None:
            self._reject_callables(updates)

        # Special case: full-table updates via DataFrame input.
        if updates is None:
            if target is None:
                raise ValueError(
                    "update() requires either updates or a target DataFrame"
                )
            if not isinstance(target, pl.DataFrame):
                raise TypeError("When updates is None, target must be a DataFrame")
            try:
                object.__delattr__(self._space, "_cells_row_major_ok")
            except AttributeError:
                pass
            self._update_polars_cells_table(target)
            return

        if backend not in {"auto", "polars"}:
            raise ValueError('backend must be one of: "auto", "polars"')

        if target is not None and mask is not None:
            raise ValueError("Provide either target or mask, not both")

        selector: object = target if target is not None else mask
        if selector is None:
            selector = "all"

        if isinstance(selector, pl.Expr):
            raise TypeError(
                "update(mask=pl.Expr) is not supported; pass coords/ids, a boolean mask, or a string mask"
            )

        mask_arg: object = selector
        if not isinstance(mask_arg, (str, pl.DataFrame, pl.Series, np.ndarray)):
            coords, agents = self._split_target(mask_arg)
            if agents is not None:
                mask_arg = self._space._get_df_coords(agents=agents)
            elif coords is None:
                mask_arg = "all"
            elif isinstance(coords, pl.DataFrame):
                mask_arg = coords
            else:
                mask_arg = self._space._get_df_coords(coords)

        # Coordinate-target updates should not force dense row-major storage.
        # If a DataFrame of coordinates is provided (or derived) and no explicit
        # boolean mask column is specified, treat it as an upsert by coordinates.
        pos_cols = self._space._pos_col_names
        if (
            isinstance(mask_arg, pl.DataFrame)
            and mask_col is None
            and all(c in mask_arg.columns for c in pos_cols)
        ):
            coords_df = mask_arg.select(pos_cols)
            n_sel = int(coords_df.height)
            if n_sel == 0:
                return

            coords_np = coords_df.to_numpy().astype(np.int64, copy=False)
            target_cell_id = self._cell_id_from_coords(coords_np)

            existing_row_idx: list[int] = []
            sel_existing_idx: list[int] = []
            sel_new_idx: list[int] = []

            if not self._cells.is_empty():
                existing_coords = (
                    self._cells.select(pos_cols).to_numpy().astype(np.int64, copy=False)
                )
                existing_cell_id = self._cell_id_from_coords(existing_coords)
                row_by_cell_id = {int(cid): i for i, cid in enumerate(existing_cell_id)}

                for sel_i, cid in enumerate(target_cell_id):
                    row_i = row_by_cell_id.get(int(cid))
                    if row_i is None:
                        sel_new_idx.append(sel_i)
                    else:
                        sel_existing_idx.append(sel_i)
                        existing_row_idx.append(int(row_i))
            else:
                sel_new_idx = list(range(n_sel))

            def _values_for_selection(value: object, n: int) -> object:
                if isinstance(value, pl.Series):
                    if int(value.len()) == n:
                        return value.to_numpy()
                    if int(value.len()) == 1:
                        return np.repeat(value.to_numpy()[0], n)
                    return value
                if isinstance(value, np.ndarray):
                    arr = value
                    if arr.ndim == 0:
                        return arr.item()
                    if int(arr.shape[0]) == n:
                        return arr
                    if int(arr.shape[0]) == 1:
                        return np.repeat(arr[0], n)
                    return value
                if isinstance(value, (list, tuple)):
                    if len(value) == n:
                        return np.asarray(value)
                    if len(value) == 1:
                        return np.repeat(value[0], n)
                    return value
                return value

            # Update remaining-capacity array BEFORE mutating self._cells, because
            # _update_capacity_cells derives agents-in-cells from the current capacity.
            if "capacity" in updates:
                cap_val = _values_for_selection(updates["capacity"], n_sel)
                if isinstance(cap_val, (np.ndarray, list, tuple)):
                    cap_arr = np.asarray(cap_val)
                    if cap_arr.ndim != 1 or int(cap_arr.shape[0]) != n_sel:
                        raise ValueError("capacity update length mismatch")
                    cap_series = pl.Series("capacity", cap_arr)
                else:
                    cap_series = pl.repeat(float(cap_val), n_sel, eager=True).alias(
                        "capacity"
                    )
                cap_updates = coords_df.with_columns(
                    pl.Series("capacity", cap_series).cast(pl.Float64)
                    if isinstance(cap_series, pl.Series)
                    else cap_series.cast(pl.Float64)
                )
                self._cells_capacity = self._update_capacity_cells(cap_updates)

            # Update existing rows.
            if existing_row_idx:
                mask_bool = np.zeros(int(self._cells.height), dtype=bool)
                row_idx_arr = np.asarray(existing_row_idx, dtype=np.int64)
                mask_bool[row_idx_arr] = True

                updates_existing: dict[str, object] = {}
                for col, raw in updates.items():
                    val = _values_for_selection(raw, n_sel)

                    if isinstance(val, (np.ndarray, list, tuple)):
                        arr = np.asarray(val)
                        if arr.ndim == 1 and int(arr.shape[0]) == n_sel:
                            full = np.empty(int(self._cells.height), dtype=arr.dtype)
                            fill_val = arr[0] if full.size else 0
                            full.fill(fill_val)
                            full[row_idx_arr] = arr[
                                np.asarray(sel_existing_idx, dtype=np.int64)
                            ]
                            updates_existing[col] = pl.Series(col, full)
                        else:
                            updates_existing[col] = raw
                    elif isinstance(val, pl.Series):
                        if int(val.len()) == n_sel:
                            arr = val.to_numpy()
                            full = np.empty(int(self._cells.height), dtype=arr.dtype)
                            fill_val = arr[0] if full.size else 0
                            full.fill(fill_val)
                            full[row_idx_arr] = arr[
                                np.asarray(sel_existing_idx, dtype=np.int64)
                            ]
                            updates_existing[col] = pl.Series(col, full)
                        else:
                            updates_existing[col] = raw
                    else:
                        updates_existing[col] = raw

                self._cells = self._apply_masked_updates(
                    self._cells, mask_bool, updates_existing
                )

            # Append new rows.
            if sel_new_idx:
                new_coords_df = coords_df[sel_new_idx]
                new_rows = new_coords_df

                existing_cols = set(self._cells.columns)
                copy_updates: list[tuple[str, str]] = []

                for col, raw in updates.items():
                    val = _values_for_selection(raw, n_sel)
                    if isinstance(val, str) and (
                        val in existing_cols or val in updates
                    ):
                        copy_updates.append((col, val))
                        continue

                    if isinstance(val, (np.ndarray, list, tuple)):
                        arr = np.asarray(val)
                        if arr.ndim == 1 and int(arr.shape[0]) == n_sel:
                            arr = arr[np.asarray(sel_new_idx, dtype=np.int64)]
                            new_rows = new_rows.with_columns(pl.Series(col, arr))
                        else:
                            new_rows = new_rows.with_columns(pl.lit(raw).alias(col))
                    elif isinstance(val, pl.Series):
                        if int(val.len()) == n_sel:
                            arr = val.to_numpy()[
                                np.asarray(sel_new_idx, dtype=np.int64)
                            ]
                            new_rows = new_rows.with_columns(pl.Series(col, arr))
                        else:
                            new_rows = new_rows.with_columns(raw.alias(col))
                    else:
                        new_rows = new_rows.with_columns(pl.lit(raw).alias(col))

                if copy_updates:
                    new_rows = new_rows.with_columns(
                        [pl.col(src).alias(dst) for dst, src in copy_updates]
                    )

                self._cells = pl.concat(
                    [self._cells, new_rows], how="diagonal_relaxed", rechunk=True
                )

            return

        mask_bool = self._mask_to_bool(mask_arg, mask_col=mask_col)
        self._cells = self._apply_masked_updates(self._cells, mask_bool, updates)

        if "capacity" in updates:
            # For mask-based updates, update remaining capacity before mutating _cells.
            # (This path assumes dense row-major cells.)
            selected_idx = np.flatnonzero(mask_bool)
            if selected_idx.size:
                coords_np = self._coords_from_cell_id(
                    selected_idx.astype(np.int64, copy=False)
                )
                coords_df = pl.DataFrame(
                    coords_np,
                    schema=self._space._pos_col_names,
                    orient="row",
                )

                cap_val = updates["capacity"]
                if isinstance(cap_val, pl.Series):
                    if int(cap_val.len()) == int(selected_idx.size):
                        cap_series = pl.Series("capacity", cap_val.to_numpy())
                    else:
                        cap_series = pl.repeat(
                            float(cap_val.to_numpy()[0]),
                            int(selected_idx.size),
                            eager=True,
                        ).alias("capacity")
                elif isinstance(cap_val, (list, tuple, np.ndarray)):
                    arr = np.asarray(cap_val)
                    if arr.ndim == 0:
                        cap_series = pl.repeat(
                            float(arr.item()), int(selected_idx.size), eager=True
                        ).alias("capacity")
                    elif int(arr.shape[0]) == int(selected_idx.size):
                        cap_series = pl.Series("capacity", arr)
                    elif int(arr.shape[0]) == 1:
                        cap_series = pl.repeat(
                            float(arr[0]), int(selected_idx.size), eager=True
                        ).alias("capacity")
                    else:
                        raise ValueError("capacity update length mismatch")
                else:
                    cap_series = pl.repeat(
                        float(cap_val), int(selected_idx.size), eager=True
                    ).alias("capacity")

                cap_updates = coords_df.with_columns(
                    pl.Series("capacity", cap_series).cast(pl.Float64)
                    if isinstance(cap_series, pl.Series)
                    else cap_series.cast(pl.Float64)
                )
                self._cells_capacity = self._update_capacity_cells(cap_updates)

    def _mask_to_bool(self, mask: object, *, mask_col: str | None = None) -> np.ndarray:
        self._ensure_dense_row_major_cells()
        n_total = int(np.prod(self._space._dimensions))
        if int(self._cells.height) != n_total:
            raise ValueError("Dense row-major cells are required for fast updates")

        if mask is None or (isinstance(mask, str) and mask == "all"):
            return np.ones(n_total, dtype=bool)

        if isinstance(mask, str):
            remaining = self._cells_capacity.ravel(order="C")
            if mask == "full":
                return remaining == 0
            if mask == "empty":
                cap = self._cells["capacity"].to_numpy().astype(float, copy=False)
                return remaining.astype(float, copy=False) == cap
            if mask == "available":
                return remaining > 0
            raise ValueError(
                "Unsupported mask string; expected 'all', 'empty', 'full', or 'available'"
            )

        if isinstance(mask, np.ndarray):
            arr = mask
            if arr.dtype == bool:
                if arr.ndim == 2:
                    arr = arr.ravel(order="C")
                if arr.ndim != 1 or int(arr.shape[0]) != n_total:
                    raise ValueError("Boolean mask length mismatch")
                return arr.astype(bool, copy=False)
            raise TypeError("Mask ndarray must be boolean")

        if isinstance(mask, pl.Series):
            if mask.dtype != pl.Boolean:
                raise TypeError("Mask Series must be boolean")
            if int(mask.len()) != n_total:
                raise ValueError("Boolean mask Series length mismatch")
            return mask.to_numpy().astype(bool, copy=False)

        if isinstance(mask, pl.DataFrame):
            pos_cols = self._space._pos_col_names
            if any(c not in mask.columns for c in pos_cols):
                raise KeyError(f"Mask DataFrame must include columns: {pos_cols}")

            bool_values: np.ndarray
            if mask_col is not None:
                if mask_col not in mask.columns:
                    raise KeyError(f"mask_col not found in DataFrame: {mask_col}")
                s = mask[mask_col]
                if s.dtype != pl.Boolean:
                    raise TypeError("mask_col must be a boolean column")
                bool_values = s.to_numpy().astype(bool, copy=False)
            else:
                extra_cols = [c for c in mask.columns if c not in pos_cols]
                if len(extra_cols) == 1 and mask[extra_cols[0]].dtype == pl.Boolean:
                    bool_values = (
                        mask[extra_cols[0]].to_numpy().astype(bool, copy=False)
                    )
                else:
                    bool_values = np.ones(int(mask.height), dtype=bool)

            coords_np = mask.select(pos_cols).to_numpy()
            cell_id = self._cell_id_from_coords(coords_np)
            out = np.zeros(n_total, dtype=bool)
            np.logical_or.at(out, cell_id.astype(np.int64, copy=False), bool_values)
            return out

        raise TypeError("Unsupported mask type")

    def lookup(
        self,
        target: DiscreteCoordinates | DataFrame | np.ndarray,
        columns: list[str] | None = None,
        *,
        as_df: bool = True,
    ) -> pl.DataFrame | dict[str, np.ndarray] | np.ndarray:
        """Fetch cell rows by coords or cell_id without joins."""
        self._ensure_dense_row_major_cells()
        n_total = int(np.prod(self._space._dimensions))
        if int(self._cells.height) != n_total:
            raise ValueError("Dense row-major cells are required for lookup")

        if isinstance(target, pl.DataFrame):
            if "cell_id" in target.columns:
                cell_id = target["cell_id"].to_numpy().astype(np.int64, copy=False)
            else:
                pos_cols = self._space._pos_col_names
                if any(c not in target.columns for c in pos_cols):
                    raise KeyError(
                        f"lookup target DataFrame must have 'cell_id' or coords columns: {pos_cols}"
                    )
                coords_np = target.select(pos_cols).to_numpy()
                cell_id = self._cell_id_from_coords(coords_np)
        else:
            arr = np.asarray(target)
            if arr.ndim == 2 and arr.shape[1] == len(self._space._dimensions):
                cell_id = self._cell_id_from_coords(arr.astype(np.int64, copy=False))
            else:
                cell_id = arr.astype(np.int64, copy=False)

        if cell_id.ndim == 0:
            cell_id = cell_id.reshape(1)

        if (cell_id < 0).any() or (cell_id >= n_total).any():
            raise IndexError("cell_id out of bounds")

        out = self._cells[cell_id]
        if columns is not None:
            out = out.select(columns)

        if as_df:
            return out
        if columns is not None and len(columns) == 1:
            return out[columns[0]].to_numpy()
        return {col: out[col].to_numpy() for col in out.columns}

    def _update_polars_cells_table(self, cells_df: pl.DataFrame) -> None:
        """Set/merge the cells table (legacy set(table) behavior)."""
        space = self._space
        if any(k not in cells_df.columns for k in space._pos_col_names):
            raise ValueError(
                f"The cells DataFrame must have the columns {space._pos_col_names}"
            )

        if "capacity" in cells_df.columns:
            self._cells_capacity = self._update_capacity_cells(cells_df)

        if self._cells.is_empty():
            # Keep a capacity column available for named masks ("empty", "full").
            # Some models initialize dense properties (e.g. sugar/max_sugar) without
            # explicitly providing a capacity column.
            if "capacity" not in cells_df.columns:
                cap = getattr(space, "_capacity", None)
                if cap and cap != np.inf:
                    cells_df = cells_df.with_columns(
                        pl.lit(int(cap)).cast(pl.Float64).alias("capacity")
                    )
                else:
                    cells_df = cells_df.with_columns(pl.lit(np.nan).alias("capacity"))
            self._cells = cells_df
            return

        update_cols = [c for c in cells_df.columns if c not in space._pos_col_names]
        if not update_cols:
            return

        is_dense_grid = self._cells.height == int(np.prod(space._dimensions))
        if is_dense_grid:
            # Join+coalesce on dense grids, preserving existing order.
            updates_only = cells_df.select([*space._pos_col_names, *update_cols])
            merged = self._cells.join(
                updates_only,
                on=space._pos_col_names,
                how="left",
                suffix="_new",
                maintain_order="left",
            )
            existing_cols = set(self._cells.columns)
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
            self._cells = merged
        else:
            self._cells = space._df_combine_first(
                cells_df, self._cells, index_cols=space._pos_col_names
            )

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
            if len(target) == 0:
                return None, target
            first = next(iter(target))
            if isinstance(first, (AbstractAgentSet, AbstractAgentSetRegistry)):
                return None, target
        return target, None
