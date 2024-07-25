from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from mesa_frames.abstract.space import GridDF
from mesa_frames.concrete.pandas.mixin import PandasMixin


class GridPandas(GridDF, PandasMixin):
    _agents: pd.DataFrame
    _cells: pd.DataFrame
    _grid_capacity: np.ndarray
    _offsets: pd.DataFrame

    def _generate_empty_grid(
        self, dimensions: Sequence[int], capacity: int
    ) -> np.ndarray:
        return np.full(dimensions, capacity, dtype=int)

    def _sample_cells(
        self,
        n: int | None,
        with_replacement: bool,
        condition: Callable[[np.ndarray], np.ndarray],
    ) -> pd.DataFrame:
        # Get the coordinates and remaining capacities of the cells
        coords = np.array(np.where(condition(self._grid_capacity))).T
        capacities = self._grid_capacity[tuple(coords.T)]

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
                    self._grid_capacity[tuple(unique_coords.T)] -= counts

                    # Check if any cells exceed their capacity and need to be resampled
                    over_capacity_mask = self._grid_capacity[tuple(unique_coords.T)] < 0
                    valid_coords = unique_coords[~over_capacity_mask]
                    invalid_coords = unique_coords[over_capacity_mask]

                    # Add valid coordinates to the sampled list
                    sampled_coords.extend(valid_coords)

                    # Restore capacities for invalid coordinates
                    if len(invalid_coords) > 0:
                        self._grid_capacity[tuple(invalid_coords.T)] += counts[
                            over_capacity_mask
                        ]

                    # Update coords based on the current state of the grid
                    coords = np.array(np.where(condition(self._grid_capacity))).T

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
        sampled_cells = pd.DataFrame(sampled_coords, columns=self._pos_col_names)

        return sampled_cells

    def _update_capacity_cells(self, cells: pd.DataFrame) -> None:
        # Update the grid capacity based on the sampled cells
        self._grid_capacity[tuple(cells[self._pos_col_names].to_numpy().T)] += cells[
            "capacity"
        ]

    def _update_capacity_agents(self, agents: pd.DataFrame) -> None:
        # Update capacity for agents that were already on the grid
        masked_df = self._df_get_masked_df(
            self._agents, index_col="agent_id", mask=agents
        )
        self._grid_capacity[tuple(masked_df[self._pos_col_names].to_numpy().T)] += 1

        # Update capacity on new positions
        self._grid_capacity[tuple(agents[self._pos_col_names].to_numpy().T)] -= 1
        return self._grid_capacity

    @property
    def remaining_capacity(self) -> int:
        return self._grid_capacity.sum()
