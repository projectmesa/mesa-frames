"""Agent implementations for the Sugarscape IG example (mesa-frames).

This module provides the parallel (synchronous) movement variant as in the
advanced tutorial. The code and comments mirror
docs/general/tutorials/3_advanced_tutorial.py.
"""

from __future__ import annotations

import polars as pl

from mesa_frames import AgentSet, Model


class AntsBase(AgentSet):
    """Base agent set for the Sugarscape tutorial.

    This class implements the common behaviour shared by all agent
    movement variants (sequential, numba-accelerated and parallel).

    Notes
    -----
    - Agents are expected to have integer traits: ``sugar``, ``metabolism``
      and ``vision``. These are validated in :meth:`__init__`.
    - Subclasses must implement :meth:`move` which changes agent positions
      on the grid (via :meth:`mesa_frames.Grid` helpers).
    """

    def __init__(self, model: Model, agent_frame: pl.DataFrame) -> None:
        """Initialise the agent set and validate required trait columns.

        Parameters
        ----------
        model : Model
            The parent model which provides RNG and space.
        agent_frame : pl.DataFrame
            A Polars DataFrame with at least the columns ``sugar``,
            ``metabolism`` and ``vision`` for each agent.

        Raises
        ------
        ValueError
            If required trait columns are missing from ``agent_frame``.
        """
        super().__init__(model)
        required = {"sugar", "metabolism", "vision"}
        missing = required.difference(agent_frame.columns)
        if missing:
            raise ValueError(
                f"Initial agent frame must include columns {sorted(required)}; missing {sorted(missing)}."
            )
        self.add(agent_frame.clone())

    def step(self) -> None:
        """Advance the agent set by one time step.

        The update order is important: agents are first shuffled to randomise
        move order (this is important only for sequential variants), then they move, harvest sugar
        from their occupied cells, and finally any agents whose sugar falls
        to zero or below are removed.
        """
        # Randomise ordering for movement decisions when required by the
        # implementation (e.g. sequential update uses this shuffle).
        self.shuffle(inplace=True)
        # Movement policy implemented by subclasses.
        self.move()
        # Agents harvest sugar on their occupied cells.
        self.eat()
        # Remove agents that starved after eating.
        self._remove_starved()

    def move(self) -> None:  # pragma: no cover
        """Abstract movement method.

        Subclasses must override this method to update agent positions on the
        grid. Implementations should use :meth:`mesa_frames.Grid.move_agents`
        or similar helpers provided by the space API.
        """
        raise NotImplementedError

    def eat(self) -> None:
        """Agents harvest sugar from the cells they currently occupy.

        Behaviour:
        - Look up the set of occupied cells (cells that reference an agent
          id).
        - For each occupied cell, add the cell sugar to the agent's sugar
          stock and subtract the agent's metabolism cost.
        - After agents harvest, set the sugar on those cells to zero (they
          were consumed).
        """
        positions = self.pos.select(
            [
                pl.col("unique_id").alias("agent_id"),
                "dim_0",
                "dim_1",
            ]
        )
        sugar_df = self.space.cells.lookup(positions, columns=["sugar"], as_df=True)
        sugar = sugar_df["sugar"].fill_null(0)
        agent_ids = positions["agent_id"]

        traits = self.lookup(agent_ids, columns=["sugar", "metabolism"], as_df=True)
        new_sugar = traits["sugar"] + sugar - traits["metabolism"]
        self.update({"sugar": new_sugar}, mask=agent_ids)
        # After harvesting, occupied cells have zero sugar.
        self.space.cells.update(
            positions.select(["dim_0", "dim_1"]),
            {"sugar": 0},
        )

    def _remove_starved(self) -> None:
        """Discard agents whose sugar stock has fallen to zero or below.

        This method performs a vectorised filter on the agent frame and
        removes any matching rows from the set.
        """
        starved = self.df.filter(pl.col("sugar") <= 0)
        if not starved.is_empty():
            # Remove starved agents from the grid so full-move paths stay valid.
            self.space.remove_agents(starved["unique_id"])
            # ``discard`` accepts a DataFrame of agents to remove.
            self.discard(starved)


class AntsParallel(AntsBase):
    def step(self) -> None:
        """Advance the agent set by one time step (parallel semantics)."""
        # In synchronous parallel updates, agent ordering does not affect the outcome,
        # so we skip the per-step shuffle to avoid unnecessary overhead.
        self.move()
        self.eat()
        self._remove_starved()

    def move(self) -> None:
        """Move agents in parallel using the space-level `move_to_best` API.

        This example keeps model logic (traits, metabolism, harvesting) in the
        agent set, while delegating neighborhood generation, ranking, and
        conflict resolution to the grid implementation.
        """
        if len(self.df) == 0:
            return

        # Per-agent vision is supported by passing a 1-D integer radius array/Series
        # aligned with the provided agent ids.
        self.space.move_to_best(
            self, radius=self["vision"], property="sugar", include_center=True
        )


__all__ = [
    "AntsBase",
    "AntsParallel",
]
