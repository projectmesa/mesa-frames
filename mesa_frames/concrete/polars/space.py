from collections.abc import Callable, Sequence

import polars as pl

from mesa_frames.abstract.space import GridDF
from mesa_frames.concrete.polars.mixin import PolarsMixin
from mesa_frames.types_ import (
    GridCoordinates,
)


class GridPolars(GridDF, PolarsMixin):
    _agents: pl.DataFrame
    _cells: pl.DataFrame
    _empty_grid: list[pl.Expr]
    _offsets: pl.DataFrame

    def _generate_empty_grid(self, dimensions: Sequence[int]) -> list[pl.Expr]:
        return [pl.arange(0, d, eager=False) for d in dimensions]

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
