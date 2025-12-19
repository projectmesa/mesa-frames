"""
Polars-based implementation of spatial structures for mesa-frames.

This module provides concrete implementations of spatial structures using Polars
as the backend for DataFrame operations. It defines the Grid class, which
implements a 2D grid structure using Polars DataFrames for efficient spatial
operations and agent positioning.

Classes:
    Grid(AbstractGrid, PolarsMixin):
        A Polars-based implementation of a 2D grid. This class uses Polars
        DataFrames to store and manipulate spatial data, providing high-performance
        operations for large-scale spatial simulations.

The Grid class is designed to be used within Model instances to represent
the spatial environment of the simulation. It leverages the power of Polars for
fast and efficient data operations on spatial attributes and agent positions.

Usage:
    The Grid class can be used directly in a model to represent the
    spatial environment:

    from mesa_frames.concrete.model import Model
    from mesa_frames.concrete.space import Grid
    from mesa_frames.concrete.agentset import AgentSet

    class MyAgents(AgentSet):
        # ... agent implementation ...

    class MyModel(Model):
        def __init__(self, width, height):
            super().__init__()
            self.space = Grid(self, [width, height])
            self.sets += MyAgents(self)

        def step(self):
            # Move agents
            self.space.move_agents(self.sets)
            # ... other model logic ...

For more detailed information on the Grid class and its methods,
refer to the class docstring.
"""

from collections.abc import Callable, Collection, Sequence
from math import inf
from typing import Literal

import numpy as np
import polars as pl

from mesa_frames.abstract.agentsetregistry import AbstractAgentSetRegistry
from mesa_frames.abstract.agentset import AbstractAgentSet
from mesa_frames.abstract.space import AbstractGrid
from mesa_frames.concrete.mixin import PolarsMixin
from mesa_frames.types_ import (
    ArrayLike,
    DataFrame,
    DiscreteCoordinate,
    DiscreteCoordinates,
    IdsLike,
    Infinity,
)
from mesa_frames.utils import copydoc


@copydoc(AbstractGrid)
class Grid(AbstractGrid, PolarsMixin):
    """Polars-based implementation of AbstractGrid."""

    _agents: pl.DataFrame
    _copy_with_method: dict[str, tuple[str, list[str]]] = {
        "_agents": ("clone", []),
        "_cells": ("clone", []),
        "_cells_capacity": ("copy", []),
        "_offsets": ("clone", []),
    }
    _cells: pl.DataFrame
    _cells_capacity: np.ndarray
    _offsets: pl.DataFrame

    def _empty_cell_condition(self, cap: np.ndarray) -> np.ndarray:
        # Create a boolean mask of the same shape as cap
        empty_mask = np.ones_like(cap, dtype=bool)

        if not self._agents.is_empty():
            # Get the coordinates of all agents
            agent_coords = self._agents[self._pos_col_names].to_numpy()

            # Mark cells containing agents as not empty
            empty_mask[tuple(agent_coords.T)] = False

        return empty_mask

    def _generate_empty_grid(
        self, dimensions: Sequence[int], capacity: int | None
    ) -> np.ndarray:
        if not capacity:
            capacity = np.inf
        return np.full(dimensions, capacity)

    def _sample_cells(
        self,
        n: int | None,
        with_replacement: bool,
        condition: Callable[[np.ndarray], np.ndarray],
        respect_capacity: bool = True,
    ) -> pl.DataFrame:
        # Get the coordinates of cells that meet the condition
        coords = np.array(np.where(condition(self._cells_capacity))).T

        if respect_capacity and condition != self._full_cell_condition:
            capacities = self._cells_capacity[tuple(coords.T)]
        else:
            # If not respecting capacity or for full cells, set capacities to 1
            capacities = np.ones(len(coords), dtype=int)

        if n is not None:
            if with_replacement:
                if respect_capacity and condition != self._full_cell_condition:
                    assert n <= capacities.sum(), (
                        "Requested sample size exceeds the total available capacity."
                    )

                sampled_coords = np.empty((0, coords.shape[1]), dtype=coords.dtype)
                while len(sampled_coords) < n:
                    remaining_samples = n - len(sampled_coords)
                    sampled_indices = self.random.choice(
                        len(coords),
                        size=remaining_samples,
                        replace=True,
                    )
                    unique_indices, counts = np.unique(
                        sampled_indices, return_counts=True
                    )

                    if respect_capacity and condition != self._full_cell_condition:
                        # Calculate valid counts for each unique index
                        valid_counts = np.minimum(counts, capacities[unique_indices])
                        # Update capacities
                        capacities[unique_indices] -= valid_counts
                    else:
                        valid_counts = counts

                    # Create array of repeated coordinates
                    new_coords = np.repeat(coords[unique_indices], valid_counts, axis=0)
                    # Extend sampled_coords
                    sampled_coords = np.vstack((sampled_coords, new_coords))

                    if respect_capacity and condition != self._full_cell_condition:
                        # Update coords and capacities
                        mask = capacities > 0
                        coords = coords[mask]
                        capacities = capacities[mask]

                sampled_coords = sampled_coords[:n]
                self.random.shuffle(sampled_coords)
            else:
                assert n <= len(coords), (
                    "Requested sample size exceeds the number of available cells."
                )
                sampled_indices = self.random.choice(len(coords), size=n, replace=False)
                sampled_coords = coords[sampled_indices]
        else:
            sampled_coords = coords

        # Convert the coordinates to a DataFrame
        sampled_cells = pl.DataFrame(
            sampled_coords, schema=self._pos_col_names, orient="row"
        )
        return sampled_cells

    def _update_capacity_agents(
        self,
        agents: pl.DataFrame | pl.Series,
        operation: Literal["movement", "removal"],
    ) -> np.ndarray:
        # Update capacity for agents that were already on the grid
        masked_df: pl.DataFrame
        if (
            operation == "movement"
            and isinstance(agents, pl.DataFrame)
            and (not self._agents.is_empty())
            and agents.height == self._agents.height
            and "agent_id" in agents.columns
            and agents["agent_id"].n_unique() == agents.height
            and bool(agents["agent_id"].is_in(self._agents["agent_id"]).all())
        ):
            masked_df = self._agents
        else:
            masked_df = self._df_get_masked_df(
                self._agents, index_cols="agent_id", mask=agents
            )

        if operation == "movement":
            # Increase capacity at old positions
            old_positions = tuple(masked_df[self._pos_col_names].to_numpy().T)
            np.add.at(self._cells_capacity, old_positions, 1)

            # Decrease capacity at new positions
            new_positions = tuple(agents[self._pos_col_names].to_numpy().T)
            np.add.at(self._cells_capacity, new_positions, -1)
        elif operation == "removal":
            # Increase capacity at the positions of removed agents
            positions = tuple(masked_df[self._pos_col_names].to_numpy().T)
            np.add.at(self._cells_capacity, positions, 1)
        return self._cells_capacity

    def _update_capacity_cells(self, cells: pl.DataFrame) -> np.ndarray:
        # Get the coordinates of the cells to update
        coords = cells[self._pos_col_names]

        # Get the current capacity of updatable cells
        current_capacity = (
            coords.join(self._cells, on=self._pos_col_names, how="left")
            .fill_null(self._capacity)["capacity"]
            .to_numpy()
        )

        # Calculate the number of agents currently in each cell
        agents_in_cells = (
            current_capacity - self._cells_capacity[tuple(zip(*coords.to_numpy()))]
        )

        # Update the capacity in self._cells_capacity
        new_capacity = cells["capacity"].to_numpy() - agents_in_cells

        # Assert that no new capacity is negative
        assert np.all(new_capacity >= 0), (
            "New capacity of a cell cannot be less than the number of agents in it."
        )

        self._cells_capacity[tuple(zip(*coords.to_numpy()))] = new_capacity

        return self._cells_capacity

    @property
    def remaining_capacity(self) -> int | Infinity:
        if not self._capacity:
            return inf
        return int(self._cells_capacity.sum())

    def _expanded_offsets(self, max_radius: int) -> pl.DataFrame:
        """Return cached, radius-expanded neighborhood offsets.

        The returned dataframe has columns:
        - ``radius`` (i64)
        - ``<dim>_offset`` for each position column (i64)

        Ordering matches the legacy implementation: offsets in the order of
        ``self._offsets`` with radius increasing for each offset.
        """
        if max_radius <= 0:
            schema: dict[str, pl.DataType] = {"radius": pl.Int64}
            schema.update({f"{c}_offset": pl.Int64 for c in self._pos_col_names})
            return pl.DataFrame(schema=schema)

        try:
            cache: dict[int, pl.DataFrame] = object.__getattribute__(
                self, "_expanded_offsets_cache"
            )
        except AttributeError:
            cache = {}
        cached = cache.get(max_radius)
        if cached is not None:
            return cached

        range_df = pl.DataFrame(
            {
                "radius": pl.arange(1, max_radius + 1, eager=True).cast(pl.Int64),
            }
        )
        base = self._offsets.join(range_df, how="cross")

        offset_exprs = [
            (pl.col(c) * pl.col("radius")).cast(pl.Int64).alias(f"{c}_offset")
            for c in self._pos_col_names
        ]
        base = base.with_columns(offset_exprs).select(
            ["radius", *[f"{c}_offset" for c in self._pos_col_names]]
        )

        if self.neighborhood_type == "hexagonal":
            in_between_offsets = getattr(self, "_in_between_offsets", None)
            if in_between_offsets is None:
                raise AttributeError(
                    "Hexagonal neighborhood requires `_in_between_offsets`."
                )

            in_between_cols = ["in_between_dim_0", "in_between_dim_1"]
            radius_values: list[int] = []
            for r in range(1, max_radius + 1):
                radius_values.extend([r] * (r - 1))

            if radius_values:
                radius_df = pl.DataFrame({"radius": pl.Series(radius_values)})
                radius_df = radius_df.with_columns(
                    pl.cum_count("radius").over("radius").alias("offset")
                )

                in_between_df = in_between_offsets.join(radius_df, how="cross")

                in_between_df = in_between_df.with_columns(
                    [
                        (pl.col(self._pos_col_names[0]) * pl.col("radius")).alias(
                            self._pos_col_names[0]
                        ),
                        (pl.col(self._pos_col_names[1]) * pl.col("radius")).alias(
                            self._pos_col_names[1]
                        ),
                        (pl.col(in_between_cols[0]) * pl.col("offset")).alias(
                            in_between_cols[0]
                        ),
                        (pl.col(in_between_cols[1]) * pl.col("offset")).alias(
                            in_between_cols[1]
                        ),
                    ]
                )

                in_between_df = in_between_df.with_columns(
                    [
                        (
                            pl.col(self._pos_col_names[0]) + pl.col(in_between_cols[0])
                        ).alias(self._pos_col_names[0]),
                        (
                            pl.col(self._pos_col_names[1]) + pl.col(in_between_cols[1])
                        ).alias(self._pos_col_names[1]),
                    ]
                ).drop(in_between_cols + ["offset"])

                in_between_df = in_between_df.with_columns(
                    [
                        pl.col(self._pos_col_names[0])
                        .cast(pl.Int64)
                        .alias(f"{self._pos_col_names[0]}_offset"),
                        pl.col(self._pos_col_names[1])
                        .cast(pl.Int64)
                        .alias(f"{self._pos_col_names[1]}_offset"),
                    ]
                ).select(
                    [
                        "radius",
                        f"{self._pos_col_names[0]}_offset",
                        f"{self._pos_col_names[1]}_offset",
                    ]
                )
                base = pl.concat([base, in_between_df], how="vertical")

        cache[max_radius] = base
        object.__setattr__(self, "_expanded_offsets_cache", cache)
        return base

    def get_neighborhood(
        self,
        radius: int | Sequence[int] | ArrayLike,
        pos: DiscreteCoordinate | DiscreteCoordinates | None = None,
        agents: IdsLike
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSetRegistry]
        | None = None,
        include_center: bool = False,
    ) -> DataFrame:
        """Get neighborhood coordinates around positions/agents.

        This is the Polars concrete implementation. It optimizes the common case
        of per-center radii by expanding radii per center (via `int_ranges` +
        `explode`) and joining on `radius`, avoiding generating rows for all
        centers up to the global maximum radius.
        """
        pos_df = self._get_df_coords(pos, agents)
        if pos_df.is_empty():
            schema: dict[str, pl.DataType] = {
                c: pl.Int64
                for c in [*self._pos_col_names, "radius", *self._center_col_names]
            }
            return pl.DataFrame(schema=schema)

        center_df = pos_df.rename(
            {c: f"{c}_center" for c in self._pos_col_names}
        ).select(self._center_col_names)

        if isinstance(radius, ArrayLike):
            if __debug__ and len(radius) != len(pos_df):
                raise ValueError(
                    "The length of the radius sequence must be equal to the number of positions/agents"
                )
            radius_srs = pl.Series("max_radius", radius).cast(pl.Int64)
            max_radius = int(radius_srs.max())

            centers = center_df.with_columns(radius_srs)
            centers = (
                centers.with_columns(
                    pl.int_ranges(1, pl.col("max_radius") + 1).alias("radius")
                )
                .explode("radius")
                .drop_nulls("radius")
                .select(["radius", *self._center_col_names])
            )

            offsets = self._expanded_offsets(max_radius)
            neighbors_df = offsets.join(centers, on="radius", how="inner")
        else:
            max_radius = int(radius)
            offsets = self._expanded_offsets(max_radius)
            neighbors_df = offsets.join(center_df, how="cross")

        # Convert offsets into absolute coordinates.
        abs_exprs = [
            (pl.col(f"{c}_offset") + pl.col(f"{c}_center")).alias(c)
            for c in self._pos_col_names
        ]
        neighbors_df = neighbors_df.with_columns(abs_exprs).select(
            [
                *self._pos_col_names,
                "radius",
                *self._center_col_names,
            ]
        )

        if self._torus:
            neighbors_df = self._df_with_columns(
                neighbors_df,
                data=self.torus_adj(neighbors_df[self._pos_col_names]),
                new_columns=self._pos_col_names,
            )
            neighbors_df = neighbors_df.unique(
                subset=self._pos_col_names, keep="first", maintain_order=True
            )
        else:
            in_bounds_exprs = [
                (pl.col(c) >= 0) & (pl.col(c) < int(self._dimensions[i]))
                for i, c in enumerate(self._pos_col_names)
            ]
            neighbors_df = neighbors_df.filter(pl.all_horizontal(in_bounds_exprs))

        if include_center:
            center_rows = center_df.with_columns(
                [
                    pl.lit(0, dtype=pl.Int64).alias("radius"),
                    *[
                        pl.col(c).alias(c.replace("_center", ""))
                        for c in self._center_col_names
                    ],
                ]
            ).select([*self._pos_col_names, "radius", *self._center_col_names])
            neighbors_df = pl.concat([center_rows, neighbors_df], how="vertical")

        return neighbors_df

    def get_neighborhood_for_agents(
        self,
        radius: int | Sequence[int] | ArrayLike,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSetRegistry]
        | None = None,
        include_center: bool = False,
    ) -> pl.DataFrame:
        """Get neighborhood coordinates around agents, including ``agent_id``.

        This is a convenience wrapper around :meth:`get_neighborhood` that
        keeps the originating agent id in the output, avoiding an extra join
        back to agent positions in downstream code.
        """
        if agents is None:
            raise ValueError("agents must be provided")

        agent_ids = self._get_ids_srs(agents)
        if agent_ids.is_empty():
            schema: dict[str, pl.DataType] = {
                "agent_id": pl.UInt64,
                **{c: pl.Int64 for c in self._pos_col_names},
                "radius": pl.Int64,
                **{c: pl.Int64 for c in self._center_col_names},
            }
            return pl.DataFrame(schema=schema)

        centers = self._df_get_masked_df(
            self._agents, index_cols="agent_id", mask=agent_ids
        )
        centers = self._df_reindex(centers, agent_ids, new_index_cols="agent_id")
        centers = self._df_reset_index(
            centers, index_cols="agent_id", drop=False
        ).select(["agent_id", *self._pos_col_names])

        center_df = centers.rename(
            {c: f"{c}_center" for c in self._pos_col_names}
        ).select(["agent_id", *self._center_col_names])

        if isinstance(radius, ArrayLike):
            if __debug__ and len(radius) != len(centers):
                raise ValueError(
                    "The length of the radius sequence must be equal to the number of agents"
                )
            radius_srs = pl.Series("max_radius", radius).cast(pl.Int64)
            max_radius = int(radius_srs.max())

            per_radius_centers = (
                center_df.with_columns(radius_srs)
                .with_columns(
                    pl.int_ranges(1, pl.col("max_radius") + 1).alias("radius")
                )
                .explode("radius")
                .drop_nulls("radius")
                .select(["radius", "agent_id", *self._center_col_names])
            )
            offsets = self._expanded_offsets(max_radius)
            neighbors_df = offsets.join(per_radius_centers, on="radius", how="inner")
        else:
            max_radius = int(radius)
            offsets = self._expanded_offsets(max_radius)
            neighbors_df = offsets.join(center_df, how="cross")

        abs_exprs = [
            (pl.col(f"{c}_offset") + pl.col(f"{c}_center")).alias(c)
            for c in self._pos_col_names
        ]
        neighbors_df = neighbors_df.with_columns(abs_exprs).select(
            ["agent_id", *self._pos_col_names, "radius", *self._center_col_names]
        )

        if self._torus:
            neighbors_df = self._df_with_columns(
                neighbors_df,
                data=self.torus_adj(neighbors_df[self._pos_col_names]),
                new_columns=self._pos_col_names,
            )
            neighbors_df = neighbors_df.unique(
                subset=["agent_id", *self._pos_col_names],
                keep="first",
                maintain_order=True,
            )
        else:
            in_bounds_exprs = [
                (pl.col(c) >= 0) & (pl.col(c) < int(self._dimensions[i]))
                for i, c in enumerate(self._pos_col_names)
            ]
            neighbors_df = neighbors_df.filter(pl.all_horizontal(in_bounds_exprs))

        if include_center:
            center_rows = center_df.with_columns(
                [
                    pl.lit(0, dtype=pl.Int64).alias("radius"),
                    *[
                        pl.col(c).alias(c.replace("_center", ""))
                        for c in self._center_col_names
                    ],
                ]
            ).select(
                ["agent_id", *self._pos_col_names, "radius", *self._center_col_names]
            )
            neighbors_df = pl.concat([center_rows, neighbors_df], how="vertical")

        return neighbors_df

    def get_cell_property(
        self,
        coords: pl.DataFrame,
        column: str,
    ) -> pl.Series:
        """Return a cell property Series aligned with ``coords``.

        When the grid stores a dense, row-major cell table, this method uses a
        fast integer gather (no join). Otherwise it falls back to a join.
        """
        coords_df = coords.select(self._pos_col_names)
        if any(coords_df[c].null_count() for c in self._pos_col_names):
            return coords_df.join(
                self._cells.select([*self._pos_col_names, column]),
                on=self._pos_col_names,
                how="left",
            )[column]

        if self._torus:
            coords_df = self.torus_adj(coords_df[self._pos_col_names])
        else:
            if __debug__:
                out_of_bounds = False
                for i, c in enumerate(self._pos_col_names):
                    srs = coords_df[c]
                    if bool((srs < 0).any()) or bool(
                        (srs >= int(self._dimensions[i])).any()
                    ):
                        out_of_bounds = True
                        break
                if out_of_bounds:
                    return coords_df.join(
                        self._cells.select([*self._pos_col_names, column]),
                        on=self._pos_col_names,
                        how="left",
                    )[column]

        expected_rows = int(np.prod(self._dimensions))
        if self._cells.height == expected_rows and not self._cells.is_empty():
            try:
                row_major_ok = object.__getattribute__(self, "_cells_row_major_ok")
            except AttributeError:
                strides: list[int] = []
                acc = 1
                for d in reversed(self._dimensions[1:]):
                    acc *= int(d)
                    strides.insert(0, acc)
                strides.append(1)
                idx_expr = pl.lit(0, dtype=pl.Int64)
                for c, stride in zip(self._pos_col_names, strides):
                    idx_expr = idx_expr + (pl.col(c).cast(pl.Int64) * int(stride))
                idx_expr = idx_expr.alias("_cell_row")
                row_ids = self._cells.select(idx_expr)["_cell_row"]
                expected = pl.arange(0, expected_rows, eager=True).cast(pl.Int64)
                row_major_ok = bool((row_ids == expected).all())
                object.__setattr__(self, "_cells_row_major_ok", row_major_ok)

            if row_major_ok:
                strides = []
                acc = 1
                for d in reversed(self._dimensions[1:]):
                    acc *= int(d)
                    strides.insert(0, acc)
                strides.append(1)
                idx_expr = pl.lit(0, dtype=pl.Int64)
                for c, stride in zip(self._pos_col_names, strides):
                    idx_expr = idx_expr + (pl.col(c).cast(pl.Int64) * int(stride))
                idx = coords_df.select(idx_expr.alias("_cell_row"))["_cell_row"]
                return self._cells[column].gather(idx)

        # Fallback for sparse/unordered cell tables.
        return coords_df.join(
            self._cells.select([*self._pos_col_names, column]),
            on=self._pos_col_names,
            how="left",
        )[column]

    def set_cell_property(
        self,
        coords: pl.DataFrame,
        column: str,
        values: ArrayLike | pl.Series,
        *,
        inplace: bool = True,
    ) -> "Grid":
        """Set a single cell property aligned with ``coords``.

        For dense, row-major grids this uses a fast scatter into the backing
        column. Otherwise, it falls back to :meth:`set_cells`.
        """
        obj = self._get_obj(inplace=inplace)
        coords_df = coords.select(obj._pos_col_names)

        if isinstance(values, pl.Series):
            values_srs = values
        else:
            values_srs = pl.Series(name=column, values=values)

        if coords_df.height != len(values_srs):
            raise ValueError("coords and values must have the same length")

        if any(coords_df[c].null_count() for c in obj._pos_col_names):
            return obj.set_cells(coords_df, {column: values_srs}, inplace=True)

        if obj._torus:
            coords_df = obj.torus_adj(coords_df[obj._pos_col_names])

        expected_rows = int(np.prod(obj._dimensions))
        if obj._cells.height == expected_rows and not obj._cells.is_empty():
            try:
                row_major_ok = object.__getattribute__(obj, "_cells_row_major_ok")
            except AttributeError:
                strides: list[int] = []
                acc = 1
                for d in reversed(obj._dimensions[1:]):
                    acc *= int(d)
                    strides.insert(0, acc)
                strides.append(1)
                idx_expr = pl.lit(0, dtype=pl.Int64)
                for c, stride in zip(obj._pos_col_names, strides):
                    idx_expr = idx_expr + (pl.col(c).cast(pl.Int64) * int(stride))
                idx_expr = idx_expr.alias("_cell_row")
                row_ids = obj._cells.select(idx_expr)["_cell_row"]
                expected = pl.arange(0, expected_rows, eager=True).cast(pl.Int64)
                row_major_ok = bool((row_ids == expected).all())
                object.__setattr__(obj, "_cells_row_major_ok", row_major_ok)

            if row_major_ok:
                strides = []
                acc = 1
                for d in reversed(obj._dimensions[1:]):
                    acc *= int(d)
                    strides.insert(0, acc)
                strides.append(1)
                idx_expr = pl.lit(0, dtype=pl.Int64)
                for c, stride in zip(obj._pos_col_names, strides):
                    idx_expr = idx_expr + (pl.col(c).cast(pl.Int64) * int(stride))
                idx = coords_df.select(idx_expr.alias("_cell_row"))["_cell_row"]

                if not obj._torus and __debug__:
                    in_bounds_exprs = [
                        (pl.col(c) >= 0) & (pl.col(c) < int(obj._dimensions[i]))
                        for i, c in enumerate(obj._pos_col_names)
                    ]
                    ok = coords_df.select(
                        pl.all_horizontal(in_bounds_exprs).alias("_ok")
                    )["_ok"]
                    if not ok.all():
                        idx = idx.filter(ok)
                        values_srs = values_srs.filter(ok)
                        if len(values_srs) == 0:
                            return obj

                if column in obj._cells.columns:
                    base = obj._cells[column]
                else:
                    base = pl.repeat(None, obj._cells.height, eager=True).cast(
                        values_srs.dtype
                    )
                updated = base.scatter(idx, values_srs)
                obj._cells = obj._cells.with_columns(updated.alias(column))
                return obj

        return obj.set_cells(coords_df, {column: values_srs}, inplace=True)
