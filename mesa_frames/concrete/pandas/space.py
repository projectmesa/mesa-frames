from collections.abc import Callable, Sequence
from typing_extensions import Self
import numpy as np
import pandas as pd

from mesa_frames.abstract.space import GridDF
from mesa_frames.concrete.pandas.mixin import PandasMixin
from mesa_frames.types_ import (
    GridCoordinate,
    GridCoordinates,
    SpaceCoordinate,
    SpaceCoordinates,
)


class GridPandas(GridDF, PandasMixin):
    _agents: pd.DataFrame
    _cells: pd.DataFrame
    _empty_grid: np.ndarray
    _offsets: pd.DataFrame

    def get_distances(
        self,
        pos0: SpaceCoordinate | SpaceCoordinates | None = None,
        pos1: SpaceCoordinate | SpaceCoordinates | None = None,
        agents0: int | Sequence[int] | None = None,
        agents1: int | Sequence[int] | None = None,
    ) -> pd.DataFrame:
        pos0_df = self._get_df_coords(pos0, agents0)
        pos1_df = self._get_df_coords(pos1, agents1)
        return pd.DataFrame(np.linalg.norm(pos1_df - pos0_df, axis=1))

    def get_neighborhood(
        self,
        radius: int | Sequence[int],
        pos: GridCoordinate | GridCoordinates | None = None,
        agents: int | Sequence[int] | None = None,
        include_center: bool = False,
    ) -> pd.DataFrame:
        pos_df = self._get_df_coords(pos)

        # Create all possible neighbors by multiplying directions by the radius and adding original pos
        neighbors_df = self._offsets.join(
            [pd.Series(np.arange(1, radius + 1), name="radius"), pos_df],
            how="cross",
            rsuffix="_center",
        )

        neighbors_df = (
            neighbors_df[self._cells_col_names] * neighbors_df["radius"]
            + neighbors_df[self._center_col_names]
        ).drop(columns=["radius"])

        # If torus, "normalize" (take modulo) for out-of-bounds cells
        if self._torus:
            neighbors_df = self.torus_adj(neighbors_df)

        # Filter out-of-bound neighbors (all ensures that if any coordinates violates, it gets excluded)
        neighbors_df = neighbors_df[
            ((neighbors_df >= 0) & (neighbors_df < self._dimensions)).all(axis=1)
        ]

        if include_center:
            pos_df[self._center_col_names] = pos_df[self._cells_col_names]
            neighbors_df = pd.concat([neighbors_df, pos_df], ignore_index=True)

        return neighbors_df

    def set_cells(self, df: pd.DataFrame, inplace: bool = True) -> Self:
        if df.index.names != self._cells_col_names or not all(
            k in df.columns for k in self._cells_col_names
        ):
            raise ValueError(
                "The dataframe must have columns/MultiIndex 'dim_0', 'dim_1', ..."
            )
        obj = self._get_obj(inplace)
        df = df.set_index(self._cells_col_names)
        obj._cells = df.combine_first(obj._cells)
        return obj

    def _generate_empty_grid(
        self, dimensions: Sequence[int], capacity: int
    ) -> np.ogrid:
        return np.full(dimensions, capacity, dtype=int)

    def _get_df_coords(
        self,
        pos: GridCoordinate | GridCoordinates | None = None,
        agents: int | Sequence[int] | None = None,
    ) -> pd.DataFrame:
        return super()._get_df_coords(pos=pos, agents=agents)

    def _get_cells_df(self, coords: GridCoordinates) -> pd.DataFrame:
        return (
            pd.DataFrame({k: v for k, v in zip(self._cells_col_names, coords)})
            .set_index(self._cells_col_names)
            .merge(
                self._agents.reset_index(),
                how="left",
                left_index=True,
                right_on=self._cells_col_names,
            )
            .groupby(level=self._cells_col_names)
            .agg(agents=("index", list), n_agents=("index", "size"))
            .merge(self._cells, how="left", left_index=True, right_index=True)
        )

    def _place_agents_df(
        self, agents: int | Sequence[int], coords: GridCoordinates
    ) -> pd.DataFrame:
        new_df = pd.DataFrame(
            {k: v for k, v in zip(self._cells_col_names, coords)},
            index=pd.Index(agents, name="agent_id"),
        )
        new_df = self._agents.combine_first(new_df)

        # Check if the capacity is respected
        capacity_df = (
            new_df.value_counts(subset=self._cells_col_names)
            .to_frame("n_agents")
            .merge(self._cells["capacity"], on=self._cells_col_names)
        )
        capacity_df["capacity"] = capacity_df["capacity"].fillna(self._capacity)
        if (capacity_df["n_agents"] > capacity_df["capacity"]).any():
            raise ValueError(
                "There is at least a cell where the number of agents would be higher than the capacity of the cell"
            )

        return new_df

    def _sample_cells(
        self,
        n: int | None,
        with_replacement: bool,
        condition: Callable[[np.ndarray], np.ndarray],
    ) -> pd.DataFrame:
        # Get the coordinates and remaining capacities of the cells
        coords = np.array(np.where(condition(self._empty_grid))).T
        capacities = self._empty_grid[tuple(coords.T)]

        if n is not None:
            if with_replacement:
                assert (
                    n <= capacities.sum()
                ), "Requested sample size exceeds the total available capacity."

                # Initialize the sampled coordinates list
                sampled_coords = []

                # Resample until we have the correct number of samples with valid capacities
                while len(sampled_coords) < n:
                    # Calculate the remaining samples needed
                    remaining_samples = n - len(sampled_coords)

                    # Compute uniform probabilities for sampling (excluding full cells)
                    probabilities = np.ones(len(coords)) / len(coords)

                    # Sample with replacement using uniform probabilities
                    sampled_indices = np.random.choice(
                        len(coords),
                        size=remaining_samples,
                        replace=True,
                        p=probabilities,
                    )
                    new_sampled_coords = coords[sampled_indices]

                    # Update capacities
                    unique_coords, counts = np.unique(
                        new_sampled_coords, axis=0, return_counts=True
                    )
                    self._empty_grid[tuple(unique_coords.T)] -= counts

                    # Check if any cells exceed their capacity and need to be resampled
                    over_capacity_mask = self._empty_grid[tuple(unique_coords.T)] < 0
                    valid_coords = unique_coords[~over_capacity_mask]
                    invalid_coords = unique_coords[over_capacity_mask]

                    # Add valid coordinates to the sampled list
                    sampled_coords.extend(valid_coords)

                    # Restore capacities for invalid coordinates
                    if len(invalid_coords) > 0:
                        self._empty_grid[tuple(invalid_coords.T)] += counts[
                            over_capacity_mask
                        ]

                    # Update coords based on the current state of the grid
                    coords = np.array(np.where(condition(self._empty_grid))).T

                sampled_coords = np.array(sampled_coords[:n])
            else:
                assert n <= len(
                    coords
                ), "Requested sample size exceeds the number of available cells."

                # Sample without replacement
                sampled_indices = np.random.choice(len(coords), size=n, replace=False)
                sampled_coords = coords[sampled_indices]

                # No need to update capacities as sampling is without replacement
        else:
            sampled_coords = coords

        # Convert the coordinates to a DataFrame
        sampled_cells = pd.DataFrame(sampled_coords, columns=self._cells_col_names)

        return sampled_cells
