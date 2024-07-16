from collections.abc import Callable, Sequence
from typing import cast
from typing_extensions import Self
import polars as pl

from mesa_frames.abstract.space import GridDF
from mesa_frames.concrete.polars.mixin import PolarsMixin
from mesa_frames.types_ import (
    GridCoordinate,
    GridCoordinates,
    SpaceCoordinate,
    SpaceCoordinates,
)


class GridPolars(GridDF, PolarsMixin):
    _agents: pl.DataFrame
    _cells: pl.DataFrame
    _empty_grid: list[pl.Expr]
    _offsets: pl.DataFrame

    def get_distances(
        self,
        pos0: SpaceCoordinate | SpaceCoordinates | None = None,
        pos1: SpaceCoordinate | SpaceCoordinates | None = None,
        agents0: int | Sequence[int] | None = None,
        agents1: int | Sequence[int] | None = None,
    ) -> pl.DataFrame:
        pos0_df = self._get_df_coords(pos0, agents0)
        pos1_df = self._get_df_coords(pos1, agents1)
        return pos0_df - pos1_df

    def get_neighborhood(
        self,
        radius: int | Sequence[int],
        pos: GridCoordinate | GridCoordinates | None = None,
        agents: int | Sequence[int] | None = None,
        include_center: bool = False,
    ) -> pl.DataFrame:
        pos_df = self._get_df_coords(pos)

        # Create all possible neighbors by multiplying directions by the radius and adding original pos
        neighbors_df = self._offsets.join(
            [pl.arange(1, radius + 1, eager=True).to_frame(name="radius"), pos_df],
            how="cross",
            suffix="_center",
        )

        neighbors_df = neighbors_df.with_columns(
            (
                pl.col(self._cells_col_names) * pl.col("radius")
                + pl.col(self._center_col_names)
            ).alias(pl.col(self._cells_col_names))
        ).drop("radius")

        # If torus, "normalize" (take modulo) for out-of-bounds cells
        if self._torus:
            neighbors_df = self.torus_adj(neighbors_df)
            neighbors_df = cast(
                pl.DataFrame, neighbors_df
            )  # Previous return is Any according to linter but should be DataFrame

        # Filter out-of-bound neighbors
        neighbors_df = neighbors_df.filter(
            pl.all((neighbors_df < self._dimensions) & (neighbors_df >= 0))
        )

        if include_center:
            pos_df.with_columns(
                pl.col(self._cells_col_names).alias(self._center_col_names)
            )
            neighbors_df = pl.concat([neighbors_df, pos_df], how="vertical")

        return neighbors_df

    def set_cells(self, df: pl.DataFrame, inplace: bool = True) -> Self:
        if not all(k in df.columns for k in self._cells_col_names):
            raise ValueError(
                "The dataframe must have an columns/MultiIndex 'dim_0', 'dim_1', ..."
            )
        obj = self._get_obj(inplace)
        obj._cells = obj._combine_first(obj._cells, df, on=self._cells_col_names)
        return obj

    def _generate_empty_grid(self, dimensions: Sequence[int]) -> list[pl.Expr]:
        return [pl.arange(0, d, eager=False) for d in dimensions]

    def _get_df_coords(
        self,
        pos: GridCoordinate | GridCoordinates | None = None,
        agents: int | Sequence[int] | None = None,
    ) -> pl.DataFrame:
        return super()._get_df_coords(pos, agents)

    def _get_cells_df(self, coords: GridCoordinates) -> pl.DataFrame:
        return (
            pl.DataFrame({k: v for k, v in zip(self._cells_col_names, coords)})
            .join(self._agents, how="left", on=self._cells_col_names)
            .group_by(self._cells_col_names)
            .agg(
                pl.col("agent_id").list().alias("agents"),
                pl.col("agent_id").count().alias("n_agents"),
            )
            .join(self._cells, on=self._cells_col_names, how="left")
        )

    def _place_agents_df(
        self, agents: int | Sequence[int], coords: GridCoordinates
    ) -> pl.DataFrame:
        new_df = pl.DataFrame(
            {"agent_id": agents}.update(
                {k: v for k, v in zip(self._cells_col_names, coords)}
            )
        )
        new_df: pl.DataFrame = self._df_combine_first(
            self._agents, new_df, on="agent_id"
        )

        # Check if the capacity is respected
        capacity_df = (
            new_df.group_by(self._cells_col_names)
            .count()
            .join(
                self._cells[self._cells_col_names + ["capacity"]],
                on=self._cells_col_names,
            )
        )
        capacity_df = capacity_df.with_columns(
            capacity=pl.col("capacity").fill_null(self._capacity)
        )
        if (capacity_df["count"] > capacity_df["capacity"]).any():
            raise ValueError(
                "There is at least a cell where the number of agents would be higher than the capacity of the cell"
            )

        return new_df

    def _sample_cells_lazy(
        self,
        n: int | None,
        with_replacement: bool,
        condition: Callable[[pl.Expr], pl.Expr],
    ) -> pl.DataFrame:
        # Create a base DataFrame with all grid coordinates and default capacities
        grid_df = pl.DataFrame(self._empty_grid).with_columns(
            [pl.lit(self._capacity).alias("capacity")]
        )

        # Apply the condition to filter the cells
        grid_df = grid_df.filter(condition(pl.col("capacity")))

        if n is not None:
            if with_replacement:
                assert (
                    n <= grid_df.select(pl.sum("capacity")).item()
                ), "Requested sample size exceeds the total available capacity."

                # Initialize the sampled DataFrame
                sampled_df = pl.DataFrame()

                # Resample until we have the correct number of samples with valid capacities
                while sampled_df.shape[0] < n:
                    # Calculate the remaining samples needed
                    remaining_samples = n - sampled_df.shape[0]

                    # Sample with replacement using uniform probabilities
                    sampled_part = grid_df.sample(
                        n=remaining_samples, with_replacement=True
                    )

                    # Count occurrences of each sampled coordinate
                    count_df = sampled_part.group_by(self._cells_col_names).agg(
                        pl.count("capacity").alias("sampled_count")
                    )

                    # Adjust capacities based on counts
                    grid_df = (
                        grid_df.join(count_df, on=self._cells_col_names, how="left")
                        .with_columns(
                            [
                                (
                                    pl.col("capacity")
                                    - pl.col("sampled_count").fill_null(0)
                                ).alias("capacity")
                            ]
                        )
                        .drop("sampled_count")
                    )

                    # Ensure no cell exceeds its capacity
                    valid_sampled_part = sampled_part.join(
                        grid_df.filter(pl.col("capacity") >= 0),
                        on=self._cells_col_names,
                        how="inner",
                    )

                    # Add valid samples to the result
                    sampled_df = pl.concat([sampled_df, valid_sampled_part])

                    # Filter out over-capacity cells from the grid
                    grid_df = grid_df.filter(pl.col("capacity") > 0)

                sampled_df = sampled_df.head(n)  # Ensure we have exactly n samples
            else:
                assert (
                    n <= grid_df.height
                ), "Requested sample size exceeds the number of available cells."

                # Sample without replacement
                sampled_df = grid_df.sample(n=n, with_replacement=False)
        else:
            sampled_df = grid_df

        return sampled_df

    def _sample_cells_eager(
        self,
        n: int | None,
        with_replacement: bool,
        condition: Callable[[pl.Expr], pl.Expr],
    ) -> pl.DataFrame:
        # Create a base DataFrame with all grid coordinates and default capacities
        grid_df = pl.DataFrame(self._empty_grid).with_columns(
            [pl.lit(self._capacity).alias("capacity")]
        )

        # If there are any specific capacities in self._cells, update the grid_df with these values
        if not self._cells.is_empty():
            grid_df = (
                grid_df.join(self._cells, on=self._cells_col_names, how="left")
                .with_columns(
                    [
                        pl.col("capacity_right")
                        .fill_null(pl.col("capacity"))
                        .alias("capacity")
                    ]
                )
                .drop("capacity_right")
            )

        # Apply the condition to filter the cells
        grid_df = grid_df.filter(condition(pl.col("capacity")))

        if n is not None:
            if with_replacement:
                assert (
                    n <= grid_df.select(pl.sum("capacity")).item()
                ), "Requested sample size exceeds the total available capacity."

                # Initialize the sampled DataFrame
                sampled_df = pl.DataFrame()

                # Resample until we have the correct number of samples with valid capacities
                while sampled_df.shape[0] < n:
                    # Calculate the remaining samples needed
                    remaining_samples = n - sampled_df.shape[0]

                    # Sample with replacement using uniform probabilities
                    sampled_part = grid_df.sample(
                        n=remaining_samples, with_replacement=True
                    )

                    # Count occurrences of each sampled coordinate
                    count_df = sampled_part.group_by(self._cells_col_names).agg(
                        pl.count("capacity").alias("sampled_count")
                    )

                    # Adjust capacities based on counts
                    grid_df = (
                        grid_df.join(count_df, on=self._cells_col_names, how="left")
                        .with_columns(
                            [
                                (
                                    pl.col("capacity")
                                    - pl.col("sampled_count").fill_null(0)
                                ).alias("capacity")
                            ]
                        )
                        .drop("sampled_count")
                    )

                    # Ensure no cell exceeds its capacity
                    valid_sampled_part = sampled_part.join(
                        grid_df.filter(pl.col("capacity") >= 0),
                        on=self._cells_col_names,
                        how="inner",
                    )

                    # Add valid samples to the result
                    sampled_df = pl.concat([sampled_df, valid_sampled_part])

                    # Filter out over-capacity cells from the grid
                    grid_df = grid_df.filter(pl.col("capacity") > 0)

                sampled_df = sampled_df.head(n)  # Ensure we have exactly n samples
            else:
                assert (
                    n <= grid_df.height
                ), "Requested sample size exceeds the number of available cells."

                # Sample without replacement
                sampled_df = grid_df.sample(n=n, with_replacement=False)
        else:
            sampled_df = grid_df

        return sampled_df

    def _sample_cells(
        self,
        n: int | None,
        with_replacement: bool,
        condition: Callable[[pl.Expr], pl.Expr],
    ) -> pl.DataFrame:
        if "capacity" not in self._cells.columns:
            return self._sample_cells_lazy(n, with_replacement, condition)
        else:
            return self._sample_cells_eager(n, with_replacement, condition)
