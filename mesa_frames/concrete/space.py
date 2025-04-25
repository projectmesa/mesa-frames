"""
Polars-based implementation of spatial structures for mesa-frames.

This module provides concrete implementations of spatial structures using Polars
as the backend for DataFrame operations. It defines the GridPolars class, which
implements a 2D grid structure using Polars DataFrames for efficient spatial
operations and agent positioning.

Classes:
    GridPolars(GridDF, PolarsMixin):
        A Polars-based implementation of a 2D grid. This class uses Polars
        DataFrames to store and manipulate spatial data, providing high-performance
        operations for large-scale spatial simulations.

The GridPolars class is designed to be used within ModelDF instances to represent
the spatial environment of the simulation. It leverages the power of Polars for
fast and efficient data operations on spatial attributes and agent positions.

Usage:
    The GridPolars class can be used directly in a model to represent the
    spatial environment:

    from mesa_frames.concrete.model import ModelDF
    from mesa_frames.concrete.space import GridPolars
    from mesa_frames.concrete.agentset import AgentSetPolars

    class MyAgents(AgentSetPolars):
        # ... agent implementation ...

    class MyModel(ModelDF):
        def __init__(self, width, height):
            super().__init__()
            self.space = GridPolars(self, [width, height])
            self.agents += MyAgents(self)

        def step(self):
            # Move agents
            self.space.move_agents(self.agents)
            # ... other model logic ...

For more detailed information on the GridPolars class and its methods,
refer to the class docstring.
"""

from math import inf
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from mesa_frames.abstract.space import GridDF
from mesa_frames.concrete.mixin import PolarsMixin
from mesa_frames.types_ import Infinity
from mesa_frames.utils import copydoc


@beartype
@copydoc(GridDF)
class GridPolars(GridDF, PolarsMixin):
    """Polars-based implementation of GridDF."""

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
