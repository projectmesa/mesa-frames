"""
Pandas-based implementation of spatial structures for mesa-frames.

This module provides concrete implementations of spatial structures using pandas
as the backend for DataFrame operations. It defines the GridPandas class, which
implements a 2D grid structure using pandas DataFrames for efficient spatial
operations and agent positioning.

Classes:
    GridPandas(GridDF, PandasMixin):
        A pandas-based implementation of a 2D grid. This class uses pandas
        DataFrames to store and manipulate spatial data, providing high-performance
        operations for large-scale spatial simulations.

The GridPandas class is designed to be used within ModelDF instances to represent
the spatial environment of the simulation. It leverages the power of pandas for
fast and efficient data operations on spatial attributes and agent positions.

Usage:
    The GridPandas class can be used directly in a model to represent the
    spatial environment:

    from mesa_frames.concrete.model import ModelDF
    from mesa_frames.concrete.pandas.space import GridPandas
    from mesa_frames.concrete.pandas.agentset import AgentSetPandas

    class MyAgents(AgentSetPandas):
        # ... agent implementation ...

    class MyModel(ModelDF):
        def __init__(self, width, height):
            super().__init__()
            self.space = GridPandas(self, [width, height])
            self.agents += MyAgents(self)

        def step(self):
            # Move agents
            self.space.move_agents(self.agents, positions)
            # ... other model logic ...

Features:
    - Efficient storage and retrieval of agent positions
    - Fast operations for moving agents and querying neighborhoods
    - Seamless integration with pandas-based agent sets
    - Support for various boundary conditions (e.g., wrapped, bounded)

Note:
    This implementation relies on pandas, so users should ensure that pandas
    is installed and imported. The performance characteristics of this class
    will depend on the pandas version and the specific operations used.

For more detailed information on the GridPandas class and its methods,
refer to the class docstring.
"""

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from mesa_frames.abstract.space import GridDF
from mesa_frames.concrete.pandas.mixin import PandasMixin
from mesa_frames.utils import copydoc
import warnings


@copydoc(GridDF)
class GridPandas(GridDF, PandasMixin):
    """
    WARNING: GridPandas is deprecated and will be removed in the next release of mesa-frames.
    pandas-based implementation of GridDF.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "GridPandas is deprecated and will be removed in the next release of mesa-frames.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    _agents: pd.DataFrame
    _copy_with_method: dict[str, tuple[str, list[str]]] = {
        "_agents": ("copy", ["deep"]),
        "_cells": ("copy", ["deep"]),
        "_cells_capacity": ("copy", []),
        "_offsets": ("copy", ["deep"]),
    }
    _cells: pd.DataFrame
    _cells_capacity: np.ndarray
    _offsets: pd.DataFrame

    def _empty_cell_condition(self, cap: np.ndarray) -> np.ndarray:
        # Create a boolean mask of the same shape as cap
        empty_mask = np.ones_like(cap, dtype=bool)

        if not self._agents.empty:
            # Get the coordinates of all agents
            agent_coords = self._agents[self._pos_col_names].to_numpy(int)

            # Mark cells containing agents as not empty
            empty_mask[tuple(agent_coords.T)] = False

        return empty_mask

    def _generate_empty_grid(
        self, dimensions: Sequence[int], capacity: int
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
    ) -> pd.DataFrame:
        # Get the coordinates of cells that meet the condition
        coords = np.array(np.where(condition(self._cells_capacity))).T

        # If the grid has infinite capacity, there is no need to respect capacity
        if np.any(self._cells_capacity == np.inf):
            respect_capacity = False

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
        sampled_cells = pd.DataFrame(sampled_coords, columns=self._pos_col_names)
        return sampled_cells

    def _update_capacity_agents(
        self,
        agents: pd.DataFrame,
        operation: Literal["movement", "removal"],
    ) -> np.ndarray:
        # Update capacity for agents that were already on the grid
        masked_df = self._df_get_masked_df(
            self._agents, index_cols="agent_id", mask=agents
        )

        if operation == "movement":
            # Increase capacity at old positions
            old_positions = tuple(masked_df[self._pos_col_names].to_numpy(int).T)
            np.add.at(self._cells_capacity, old_positions, 1)

            # Decrease capacity at new positions
            new_positions = tuple(agents[self._pos_col_names].to_numpy(int).T)
            np.add.at(self._cells_capacity, new_positions, -1)
        elif operation == "removal":
            # Increase capacity at the positions of removed agents
            positions = tuple(masked_df[self._pos_col_names].to_numpy(int).T)
            np.add.at(self._cells_capacity, positions, 1)
        return self._cells_capacity

    def _update_capacity_cells(self, cells: pd.DataFrame) -> np.ndarray:
        # Get the coordinates of the cells to update
        coords = cells.index

        # Get the current capacity of updatable cells
        current_capacity = self._cells.reindex(coords, fill_value=self._capacity)[
            "capacity"
        ].to_numpy()

        # Calculate the number of agents currently in each cell
        agents_in_cells = current_capacity - self._cells_capacity[tuple(zip(*coords))]

        # Update the capacity in self._cells_capacity
        new_capacity = cells["capacity"].to_numpy() - agents_in_cells

        # Assert that no new capacity is negative
        assert np.all(new_capacity >= 0), (
            "New capacity of a cell cannot be less than the number of agents in it."
        )

        self._cells_capacity[tuple(zip(*coords))] = new_capacity

        return self._cells_capacity

    @property
    def remaining_capacity(self) -> int:
        if not self._capacity:
            return np.inf
        return self._cells_capacity.sum()
