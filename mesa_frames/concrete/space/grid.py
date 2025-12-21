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

from collections.abc import Collection, Sequence
import numpy as np
import polars as pl

from mesa_frames.abstract.agentsetregistry import AbstractAgentSetRegistry
from mesa_frames.abstract.agentset import AbstractAgentSet
from mesa_frames.abstract.space import AbstractGrid
from .cells import GridCells
from mesa_frames.concrete.mixin import PolarsMixin
from mesa_frames.types_ import (
    ArrayLike,
    DataFrame,
    DiscreteCoordinate,
    DiscreteCoordinates,
    IdsLike,
)
from mesa_frames.utils import copydoc


@copydoc(AbstractGrid)
class Grid(AbstractGrid, PolarsMixin):
    """Polars-based implementation of AbstractGrid."""

    _agents: pl.DataFrame
    _copy_with_method: dict[str, tuple[str, list[str]]] = {
        "_agents": ("clone", []),
        "_offsets": ("clone", []),
    }
    _offsets: pl.DataFrame

    def __init__(
        self,
        model: "mesa_frames.concrete.model.Model",
        dimensions: Sequence[int],
        torus: bool = False,
        capacity: int | None = None,
        neighborhood_type: str = "moore",
    ) -> None:
        super().__init__(
            model=model,
            dimensions=dimensions,
            torus=torus,
            capacity=capacity,
            neighborhood_type=neighborhood_type,
        )
        self.cells = GridCells(self)

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
