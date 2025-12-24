"""Concrete neighborhood implementation for polars-backed grids."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Literal

import numpy as np
import polars as pl

from mesa_frames.abstract.agentset import AbstractAgentSet
from mesa_frames.abstract.agentsetregistry import AbstractAgentSetRegistry
from mesa_frames.abstract.space import AbstractGrid
from mesa_frames.abstract.space.neighborhood import AbstractNeighborhood
from mesa_frames.types_ import (
    ArrayLike,
    DataFrame,
    DiscreteCoordinate,
    DiscreteCoordinates,
    IdsLike,
    Series,
)


class GridNeighborhood(AbstractNeighborhood):
    """Polars-based neighborhood implementation for grids."""

    def __init__(self, space: AbstractGrid) -> None:
        super().__init__(space)

    def copy(self, space: AbstractGrid) -> GridNeighborhood:
        return self.__class__(space)

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
    ) -> DataFrame:
        if include not in {"coords", "agents", "both"}:
            raise ValueError("include must be one of: coords, agents, both")

        coords, agents = self._split_target(target)
        if agents is None:
            neighbors_df = self._neighbors_for_coords(radius, coords, include_center)
        else:
            neighbors_df = self._neighbors_for_agents(radius, agents, include_center)

        if include == "coords":
            return neighbors_df

        space = self._space
        if include == "agents":
            return space._df_get_masked_df(
                df=space._agents,
                index_cols=space._pos_col_names,
                mask=neighbors_df,
            )

        return space._df_join(
            left=neighbors_df,
            right=space._agents,
            index_cols=space._pos_col_names,
            on=space._pos_col_names,
        )

    def _expanded_offsets(self, max_radius: int) -> pl.DataFrame:
        """Return cached, radius-expanded neighborhood offsets."""
        space = self._space
        if max_radius <= 0:
            schema: dict[str, pl.DataType] = {"radius": pl.Int64}
            schema.update({f"{c}_offset": pl.Int64 for c in space._pos_col_names})
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
        base = space._offsets.join(range_df, how="cross")

        offset_exprs = [
            (pl.col(c) * pl.col("radius")).cast(pl.Int64).alias(f"{c}_offset")
            for c in space._pos_col_names
        ]
        base = base.with_columns(offset_exprs).select(
            ["radius", *[f"{c}_offset" for c in space._pos_col_names]]
        )

        if space.neighborhood_type == "hexagonal":
            in_between_offsets = getattr(space, "_in_between_offsets", None)
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
                        (pl.col(space._pos_col_names[0]) * pl.col("radius")).alias(
                            space._pos_col_names[0]
                        ),
                        (pl.col(space._pos_col_names[1]) * pl.col("radius")).alias(
                            space._pos_col_names[1]
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
                            pl.col(space._pos_col_names[0]) + pl.col(in_between_cols[0])
                        ).alias(space._pos_col_names[0]),
                        (
                            pl.col(space._pos_col_names[1]) + pl.col(in_between_cols[1])
                        ).alias(space._pos_col_names[1]),
                    ]
                ).drop(in_between_cols + ["offset"])

                in_between_df = in_between_df.with_columns(
                    [
                        pl.col(space._pos_col_names[0])
                        .cast(pl.Int64)
                        .alias(f"{space._pos_col_names[0]}_offset"),
                        pl.col(space._pos_col_names[1])
                        .cast(pl.Int64)
                        .alias(f"{space._pos_col_names[1]}_offset"),
                    ]
                ).select(
                    [
                        "radius",
                        f"{space._pos_col_names[0]}_offset",
                        f"{space._pos_col_names[1]}_offset",
                    ]
                )
                base = pl.concat([base, in_between_df], how="vertical")

        cache[max_radius] = base
        object.__setattr__(self, "_expanded_offsets_cache", cache)
        return base

    def _neighbors_for_coords(
        self,
        radius: int | float | Collection[int] | Collection[float] | ArrayLike,
        coords: DiscreteCoordinate | DiscreteCoordinates | DataFrame | None,
        include_center: bool,
    ) -> DataFrame:
        space = self._space
        pos_df = space._get_df_coords(coords, None)
        if pos_df.is_empty():
            schema: dict[str, pl.DataType] = {
                c: pl.Int64
                for c in [*space._pos_col_names, "radius", *space._center_col_names]
            }
            return pl.DataFrame(schema=schema)

        center_df = pos_df.rename(
            {c: f"{c}_center" for c in space._pos_col_names}
        ).select(space._center_col_names)

        neighbors_df = self._neighbors_from_centers(
            radius=radius,
            centers=center_df,
            center_cols=space._center_col_names,
            center_id_cols=(),
        )

        return self._finalize_neighbors(
            neighbors_df=neighbors_df,
            center_df=center_df,
            center_id_cols=(),
            include_center=include_center,
        )

    def _neighbors_for_agents(
        self,
        radius: int | float | Collection[int] | Collection[float] | ArrayLike,
        agents: IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSetRegistry]
        | Collection[AbstractAgentSet]
        | None,
        include_center: bool,
    ) -> DataFrame:
        if agents is None:
            raise ValueError("agents must be provided")

        space = self._space
        agent_ids = space._get_ids_srs(agents)
        if agent_ids.is_empty():
            schema: dict[str, pl.DataType] = {
                "agent_id": pl.UInt64,
                **{c: pl.Int64 for c in space._pos_col_names},
                "radius": pl.Int64,
                **{c: pl.Int64 for c in space._center_col_names},
            }
            return pl.DataFrame(schema=schema)

        centers = space._df_get_masked_df(
            space._agents, index_cols="agent_id", mask=agent_ids
        )
        centers = space._df_reindex(centers, agent_ids, new_index_cols="agent_id")
        centers = space._df_reset_index(
            centers, index_cols="agent_id", drop=False
        ).select(["agent_id", *space._pos_col_names])

        center_df = centers.rename(
            {c: f"{c}_center" for c in space._pos_col_names}
        ).select(["agent_id", *space._center_col_names])

        neighbors_df = self._neighbors_from_centers(
            radius=radius,
            centers=center_df,
            center_cols=space._center_col_names,
            center_id_cols=("agent_id",),
        )

        return self._finalize_neighbors(
            neighbors_df=neighbors_df,
            center_df=center_df,
            center_id_cols=("agent_id",),
            include_center=include_center,
        )

    def _neighbors_from_centers(
        self,
        radius: int | float | Collection[int] | Collection[float] | ArrayLike,
        centers: pl.DataFrame,
        center_cols: list[str],
        center_id_cols: Collection[str],
    ) -> pl.DataFrame:
        is_sequence = isinstance(radius, (np.ndarray, Series)) or (
            isinstance(radius, Collection) and not isinstance(radius, (str, bytes))
        )
        if is_sequence:
            if __debug__ and len(radius) != len(centers):
                raise ValueError(
                    "The length of the radius sequence must be equal to the number of centers"
                )
            radius_srs = pl.Series("max_radius", radius).cast(pl.Int64)
            max_radius = int(radius_srs.max())

            per_radius_centers = (
                centers.with_columns(radius_srs)
                .with_columns(
                    pl.int_ranges(1, pl.col("max_radius") + 1).alias("radius")
                )
                .explode("radius")
                .drop_nulls("radius")
                .select(["radius", *center_id_cols, *center_cols])
            )
            offsets = self._expanded_offsets(max_radius)
            neighbors_df = offsets.join(per_radius_centers, on="radius", how="inner")
        else:
            max_radius = int(radius)
            offsets = self._expanded_offsets(max_radius)
            neighbors_df = offsets.join(centers, how="cross")

        abs_exprs = [
            (pl.col(f"{c}_offset") + pl.col(f"{c}_center")).alias(c)
            for c in self._space._pos_col_names
        ]
        return neighbors_df.with_columns(abs_exprs).select(
            [*center_id_cols, *self._space._pos_col_names, "radius", *center_cols]
        )

    def _finalize_neighbors(
        self,
        neighbors_df: pl.DataFrame,
        center_df: pl.DataFrame,
        center_id_cols: Collection[str],
        include_center: bool,
    ) -> pl.DataFrame:
        space = self._space
        if space._torus:
            neighbors_df = space._df_with_columns(
                neighbors_df,
                data=space.torus_adj(neighbors_df[space._pos_col_names]),
                new_columns=space._pos_col_names,
            )
            subset = [*center_id_cols, *space._pos_col_names]
            neighbors_df = neighbors_df.unique(
                subset=subset, keep="first", maintain_order=True
            )
        else:
            in_bounds_exprs = [
                (pl.col(c) >= 0) & (pl.col(c) < int(space._dimensions[i]))
                for i, c in enumerate(space._pos_col_names)
            ]
            neighbors_df = neighbors_df.filter(pl.all_horizontal(in_bounds_exprs))

        # Ensure a stable, canonical column order before any concat operations.
        neighbors_df = neighbors_df.select(
            [
                *center_id_cols,
                *space._pos_col_names,
                "radius",
                *space._center_col_names,
            ]
        )

        if include_center:
            center_rows = center_df.with_columns(
                [
                    pl.lit(0, dtype=pl.Int64).alias("radius"),
                    *[
                        pl.col(c).alias(c.replace("_center", ""))
                        for c in space._center_col_names
                    ],
                ]
            ).select(
                [
                    *center_id_cols,
                    *space._pos_col_names,
                    "radius",
                    *space._center_col_names,
                ]
            )
            neighbors_df = pl.concat([center_rows, neighbors_df], how="vertical")

        return neighbors_df

    def _split_target(
        self,
        target: DiscreteCoordinate
        | DiscreteCoordinates
        | DataFrame
        | IdsLike
        | AbstractAgentSet
        | AbstractAgentSetRegistry
        | Collection[AbstractAgentSet]
        | Collection[AbstractAgentSetRegistry],
    ) -> tuple[DiscreteCoordinate | DiscreteCoordinates | DataFrame | None, Any | None]:
        space = self._space
        if target is None:
            raise ValueError("target must be provided")
        if isinstance(target, DataFrame):
            cols = set(space._df_column_names(target))
            if set(space._pos_col_names).issubset(cols):
                return target, None
            if "agent_id" in cols:
                return None, target["agent_id"]
            return target, None
        if isinstance(target, (AbstractAgentSet, AbstractAgentSetRegistry)):
            return None, target
        if isinstance(target, Series):
            return None, target
        if isinstance(target, np.ndarray):
            if target.ndim == 1:
                if len(target) == len(space._dimensions):
                    return target, None
                return None, target
            return target, None
        if isinstance(target, Collection):
            if not target:
                return target, None
            first = next(iter(target))
            if isinstance(first, (AbstractAgentSet, AbstractAgentSetRegistry)):
                return None, target
            if isinstance(first, Collection) and not isinstance(first, (str, bytes)):
                return target, None
            if len(target) == len(space._dimensions):
                return target, None
            return None, target
        return None, target
