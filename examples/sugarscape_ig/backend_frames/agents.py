"""Agent implementations for the Sugarscape IG example (mesa-frames).

This module provides the parallel (synchronous) movement variant as in the
advanced tutorial. The code and comments mirror
docs/general/tutorials/3_advanced_tutorial.py.
"""

from __future__ import annotations

import numpy as np
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
        # Join agent positions against the underlying cell properties to look
        # up sugar values, avoiding an O(|cells|) scan + `is_in` filter.
        positions = self.pos.select(
            [
                pl.col("unique_id").alias("agent_id"),
                "dim_0",
                "dim_1",
            ]
        )
        sugar = (
            positions.join(
                self.space.cells(include="properties").select(
                    ["dim_0", "dim_1", "sugar"]
                ),
                on=["dim_0", "dim_1"],
                how="left",
            )
            .with_columns(pl.col("sugar").fill_null(0))
            .select("sugar")["sugar"]
        )
        agent_ids = positions["agent_id"]
        self[agent_ids, "sugar"] = (
            self[agent_ids, "sugar"] + sugar - self[agent_ids, "metabolism"]
        )
        # After harvesting, occupied cells have zero sugar.
        self.space.cells.set(
            positions.select(["dim_0", "dim_1"]),
            {"sugar": pl.repeat(0, positions.height, eager=True).cast(pl.Int64)},
        )

    def _remove_starved(self) -> None:
        """Discard agents whose sugar stock has fallen to zero or below.

        This method performs a vectorised filter on the agent frame and
        removes any matching rows from the set.
        """
        starved = self.df.filter(pl.col("sugar") <= 0)
        if not starved.is_empty():
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
        """Move agents in parallel by ranking visible cells and resolving conflicts.

        Declarative mental model: express *what* each agent wants (ranked candidates),
        then use dataframe ops to *allocate* (joins, group_by with a lottery).
        Performance is handled by Polars/LazyFrames; avoid premature micro-optimisations.

        Returns
        -------
        None
            Movement updates happen in-place on the underlying space.
        """
        # Early exit if there are no agents.
        if len(self.df) == 0:
            return

        # current_pos columns:
        # ┌──────────┬────────────────┬────────────────┐
        # │ agent_id ┆ dim_0_center   ┆ dim_1_center   │
        # │ ---      ┆ ---            ┆ ---            │
        # │ u64      ┆ i64            ┆ i64            │
        # ╞══════════╪════════════════╪════════════════╡
        current_pos = self.pos.select(
            [
                pl.col("unique_id").alias("agent_id"),
                pl.col("dim_0").alias("dim_0_center"),
                pl.col("dim_1").alias("dim_1_center"),
            ]
        )
        neighborhood = self._build_neighborhood_frame()
        choices, origins, max_rank = self._rank_candidates(neighborhood, current_pos)
        if choices.is_empty():
            return
        assigned = self._resolve_conflicts_in_rounds(choices, origins, max_rank)
        if assigned.is_empty():
            return

        agent_ids = assigned["agent_id"]
        pos_df = assigned.select(
            [
                pl.col("dim_0_candidate").alias("dim_0"),
                pl.col("dim_1_candidate").alias("dim_1"),
            ]
        )
        self.space.move_all(agent_ids, pos_df)

    def _build_neighborhood_frame(self) -> pl.DataFrame:
        """Assemble the sugar-weighted neighbourhood for each sensing agent.

        Parameters
        ----------
        Returns
        -------
        pl.DataFrame
            DataFrame with columns ``agent_id``, ``radius``, ``dim_0_candidate``,
            ``dim_1_candidate`` and ``sugar`` describing the visible cells for
            each agent.
        """
        # Build a neighbourhood frame: for each agent and visible cell we
        # attach the cell sugar. The raw offsets contain the candidate
        # cell coordinates and the center coordinates for the sensing agent.
        # Raw neighborhood columns:
        # ┌──────────┬────────────┬────────────┬────────┬────────────────┬────────────────┐
        # │ agent_id ┆ dim_0      ┆ dim_1      ┆ radius ┆ dim_0_center   ┆ dim_1_center   │
        # │ ---      ┆ ---        ┆ ---        ┆ ---    ┆ ---            ┆ ---            │
        # │ u64      ┆ i64        ┆ i64        ┆ i64    ┆ i64            ┆ i64            │
        # ╞══════════╪════════════╪════════════╪════════╪════════════════╪════════════════╡
        neighborhood_cells = self.space.neighborhood(
            radius=self["vision"], target=self, include_center=True
        )

        # sugar_cells columns:
        # ┌────────────┬────────────┬────────┐
        # │ dim_0      ┆ dim_1      ┆ sugar  │
        # │ ---        ┆ ---        ┆ ---    │
        # │ i64        ┆ i64        ┆ i64    │
        # ╞════════════╪════════════╪════════╡

        neighborhood_cells = (
            neighborhood_cells.join(
                self.space.cells(include="properties").select(
                    ["dim_0", "dim_1", "sugar"]
                ),
                on=["dim_0", "dim_1"],
                how="left",
            )
            .with_columns(pl.col("sugar").fill_null(0))
        )

        # Final neighborhood columns:
        # ┌──────────┬────────┬──────────────────┬──────────────────┬────────┐
        # │ agent_id ┆ radius ┆ dim_0_candidate  ┆ dim_1_candidate  ┆ sugar  │
        # │ ---      ┆ ---    ┆ ---              ┆ ---              ┆ ---    │
        # │ u64      ┆ i64    ┆ i64              ┆ i64              ┆ i64    │
        # ╞══════════╪════════╪══════════════════╪══════════════════╪════════╡
        neighborhood_cells = (
            neighborhood_cells.drop(["dim_0_center", "dim_1_center"])
            .rename({"dim_0": "dim_0_candidate", "dim_1": "dim_1_candidate"})
            .select(
                ["agent_id", "radius", "dim_0_candidate", "dim_1_candidate", "sugar"]
            )
        )

        return neighborhood_cells

    def _rank_candidates(
        self,
        neighborhood: pl.DataFrame,
        current_pos: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Rank candidate destination cells for each agent.

        Parameters
        ----------
        neighborhood : pl.DataFrame
            Output of :meth:`_build_neighborhood_frame` with columns
            ``agent_id``, ``radius``, ``dim_0_candidate``, ``dim_1_candidate``
            and ``sugar``.
        current_pos : pl.DataFrame
            Frame with columns ``agent_id``, ``dim_0_center`` and
            ``dim_1_center`` describing where each agent currently stands.

        Returns
        -------
        choices : pl.DataFrame
            Ranked candidates per agent with columns ``agent_id``,
            ``dim_0_candidate``, ``dim_1_candidate``, ``sugar``, ``radius`` and
            ``rank``.
        origins : pl.DataFrame
            Original coordinates per agent with columns ``agent_id``,
            ``dim_0`` and ``dim_1``.
        max_rank : pl.DataFrame
            Maximum available rank per agent with columns ``agent_id`` and
            ``max_rank``.
        """
        # Create ranked choices per agent: sort by sugar (desc), radius
        # (asc), then coordinates. Keep the first unique entry per cell.
        # choices columns (after select):
        # ┌──────────┬──────────────────┬──────────────────┬────────┬────────┐
        # │ agent_id ┆ dim_0_candidate  ┆ dim_1_candidate  ┆ sugar  ┆ radius │
        # │ ---      ┆ ---              ┆ ---              ┆ ---    ┆ ---    │
        # │ u64      ┆ i64              ┆ i64              ┆ i64    ┆ i64    │
        # ╞══════════╪══════════════════╪══════════════════╪════════╪════════╡
        choices = (
            neighborhood.select(
                [
                    "agent_id",
                    "dim_0_candidate",
                    "dim_1_candidate",
                    "sugar",
                    "radius",
                ]
            )
            .sort(
                ["agent_id", "sugar", "radius", "dim_0_candidate", "dim_1_candidate"],
                descending=[False, True, False, False, False],
            )
            .unique(
                subset=["agent_id", "dim_0_candidate", "dim_1_candidate"],
                keep="first",
                maintain_order=True,
            )
            .with_columns(pl.col("agent_id").cum_count().over("agent_id").alias("rank"))
        )

        # Precompute per‑agent candidate rank once so conflict resolution can
        # promote losers by incrementing a cheap `current_rank` counter,
        # without re-sorting after each round. Alternative: drop taken cells
        # and re-rank by sugar every round; simpler conceptually but requires
        # repeated sorts and deduplication, which is heavier than filtering by
        # `rank >= current_rank`.

        # Origins for fallback (if an agent exhausts candidates it stays put).
        # origins columns:
        # ┌──────────┬────────────┬────────────┐
        # │ agent_id ┆ dim_0      ┆ dim_1      │
        # │ ---      ┆ ---        ┆ ---        │
        # │ u64      ┆ i64        ┆ i64        │
        # ╞══════════╪════════════╪════════════╡
        origins = current_pos.select(
            [
                "agent_id",
                pl.col("dim_0_center").alias("dim_0"),
                pl.col("dim_1_center").alias("dim_1"),
            ]
        )

        # Track the maximum available rank per agent to clamp promotions.
        # This bounds `current_rank`; once an agent reaches `max_rank` and
        # cannot secure a cell, they fall back to origin cleanly instead of
        # chasing nonexistent ranks.
        # max_rank columns:
        # ┌──────────┬───────────┐
        # │ agent_id ┆ max_rank │
        # │ ---      ┆ ---       │
        # │ u64      ┆ u32       │
        # ╞══════════╪═══════════╡
        max_rank = choices.group_by("agent_id").agg(
            pl.col("rank").max().alias("max_rank")
        )
        return choices, origins, max_rank

    def _resolve_conflicts_in_rounds(
        self,
        choices: pl.DataFrame,
        origins: pl.DataFrame,
        max_rank: pl.DataFrame,
    ) -> pl.DataFrame:
        """Resolve movement conflicts through iterative lottery rounds.

        Parameters
        ----------
        choices : pl.DataFrame
            Ranked candidate cells per agent with headers matching the
            ``choices`` frame returned by :meth:`_rank_candidates`.
        origins : pl.DataFrame
            Agent origin coordinates with columns ``agent_id``, ``dim_0`` and
            ``dim_1``.
        max_rank : pl.DataFrame
            Maximum rank offset per agent with columns ``agent_id`` and
            ``max_rank``.

        Returns
        -------
        pl.DataFrame
            Allocated movements with columns ``agent_id``, ``dim_0_candidate``
            and ``dim_1_candidate``; each row records the destination assigned
            to an agent.
        """
        # Prepare unresolved agents and working tables.
        agent_ids = choices["agent_id"].unique(maintain_order=True)

        # unresolved columns:
        # ┌──────────┬────────────────┐
        # │ agent_id ┆ current_rank  │
        # │ ---      ┆ ---            │
        # │ u64      ┆ i64            │
        # ╞══════════╪════════════════╡
        unresolved = pl.DataFrame(
            {
                "agent_id": agent_ids,
                "current_rank": pl.Series(np.zeros(len(agent_ids), dtype=np.int64)),
            }
        )

        # assigned columns:
        # ┌──────────┬──────────────────┬──────────────────┐
        # │ agent_id ┆ dim_0_candidate  ┆ dim_1_candidate  │
        # │ ---      ┆ ---              ┆ ---              │
        # │ u64      ┆ i64              ┆ i64              │
        # ╞══════════╪══════════════════╪══════════════════╡
        assigned = pl.DataFrame(
            {
                "agent_id": pl.Series(
                    name="agent_id", values=[], dtype=agent_ids.dtype
                ),
                "dim_0_candidate": pl.Series(
                    name="dim_0_candidate", values=[], dtype=pl.Int64
                ),
                "dim_1_candidate": pl.Series(
                    name="dim_1_candidate", values=[], dtype=pl.Int64
                ),
            }
        )
        assigned_parts: list[pl.DataFrame] = []

        # taken columns:
        # ┌─────────┐
        # │ cell_id │
        # │ ---     │
        # │ i64     │
        # ╞═════════╡
        # Treat all currently occupied cells (origins) as taken from the start.
        # Each agent may still target its own origin; we handle that exception
        # when filtering candidate pools.
        # Use a single integer key for cells to speed up joins.
        # For a 2D grid, `cell_id = dim_0 * height + dim_1` is unique.
        height = int(self.space.dimensions[1])
        choices = choices.with_columns(
            (pl.col("dim_0_candidate") * height + pl.col("dim_1_candidate"))
            .cast(pl.Int64)
            .alias("cell_id")
        )
        origins_for_filter = origins.rename(
            {"dim_0": "dim_0_origin", "dim_1": "dim_1_origin"}
        )
        origins_for_filter = origins_for_filter.with_columns(
            (pl.col("dim_0_origin") * height + pl.col("dim_1_origin"))
            .cast(pl.Int64)
            .alias("origin_cell_id")
        )
        # Attach origin ids to choices once; they are constant for the step and
        # repeatedly needed during conflict resolution rounds.
        choices = choices.join(
            origins_for_filter.select(["agent_id", "origin_cell_id"]),
            on="agent_id",
            how="left",
        )

        taken = origins_for_filter.select(
            pl.col("origin_cell_id").alias("cell_id"),
        )
        taken_ids = taken["cell_id"]

        # Resolve in rounds: each unresolved agent proposes its current-ranked
        # candidate; winners per-cell are selected at random and losers are
        # promoted to their next choice.
        while unresolved.height > 0:
            # Using precomputed `rank` lets us select candidates with
            # `rank >= current_rank` and avoid re-ranking after each round.
            # Alternative: remove taken cells and re-sort remaining candidates
            # by sugar/distance per round (heavier due to repeated sort/dedupe).
            # candidate_pool columns (after join with unresolved):
            # ┌──────────┬──────────────────┬──────────────────┬────────┬────────┬──────┬──────────────┐
            # │ agent_id ┆ dim_0_candidate  ┆ dim_1_candidate  ┆ sugar  ┆ radius ┆ rank ┆ current_rank │
            # │ ---      ┆ ---              ┆ ---              ┆ ---    ┆ ---    ┆ ---  ┆ ---          │
            # │ u64      ┆ i64              ┆ i64              ┆ i64    ┆ i64    ┆ u32  ┆ i64          │
            # ╞══════════╪══════════════════╪══════════════════╪════════╪════════╪══════╪══════════════╡
            candidate_pool = choices.join(
                unresolved, on="agent_id", how="inner", maintain_order="left"
            )
            candidate_pool = candidate_pool.filter(
                pl.col("rank") >= pl.col("current_rank")
            )
            candidate_pool = candidate_pool.filter(
                (~pl.col("cell_id").is_in(taken_ids))
                | (pl.col("cell_id") == pl.col("origin_cell_id"))
            )

            if candidate_pool.is_empty():
                # No available candidates — everyone falls back to origin.
                # Note: this covers both agents with no visible cells left and
                # the case where all remaining candidates are already taken.
                # fallback columns:
                # ┌──────────┬────────────┬────────────┬──────────────┐
                # │ agent_id ┆ dim_0      ┆ dim_1      ┆ current_rank │
                # │ ---      ┆ ---        ┆ ---        ┆ ---          │
                # │ u64      ┆ i64        ┆ i64        ┆ i64          │
                # ╞══════════╪════════════╪════════════╪══════════════╡
                fallback = unresolved.join(origins, on="agent_id", how="left")
                assigned_parts.append(
                    fallback.select(
                        [
                            "agent_id",
                            pl.col("dim_0").alias("dim_0_candidate"),
                            pl.col("dim_1").alias("dim_1_candidate"),
                        ]
                    )
                )
                break

            # best_candidates columns (per agent first choice):
            # ┌──────────┬──────────────────┬──────────────────┬────────┬────────┬──────┬──────────────┐
            # │ agent_id ┆ dim_0_candidate  ┆ dim_1_candidate  ┆ sugar  ┆ radius ┆ rank ┆ current_rank │
            # │ ---      ┆ ---              ┆ ---              ┆ ---    ┆ ---    ┆ ---  ┆ ---          │
            # │ u64      ┆ i64              ┆ i64              ┆ i64    ┆ i64    ┆ u32  ┆ i64          │
            # ╞══════════╪══════════════════╪══════════════════╪════════╪════════╪══════╪══════════════╡
            # We sort and then take the first row per agent. Using `unique`
            # here is noticeably faster than `group_by(...).first()` for large
            # frames while keeping the same semantics (first after sort).
            best_candidates = candidate_pool.unique(
                subset=["agent_id"], keep="first", maintain_order=True
            )

            # Agents that had no candidate this round fall back to origin.
            # missing columns:
            # ┌──────────┬──────────────┐
            # │ agent_id ┆ current_rank │
            # │ ---      ┆ ---          │
            # │ u64      ┆ i64          │
            # ╞══════════╪══════════════╡
            missing_ids = best_candidates["agent_id"]
            missing = unresolved.filter(~pl.col("agent_id").is_in(missing_ids))
            if not missing.is_empty():
                # fallback (missing) columns match fallback table above.
                fallback = missing.join(origins, on="agent_id", how="left")
                assigned_parts.append(
                    fallback.select(
                        [
                            "agent_id",
                            pl.col("dim_0").alias("dim_0_candidate"),
                            pl.col("dim_1").alias("dim_1_candidate"),
                        ]
                    )
                )
                unresolved = unresolved.filter(
                    ~pl.col("agent_id").is_in(missing["agent_id"])
                )
                best_candidates = best_candidates.filter(
                    ~pl.col("agent_id").is_in(missing["agent_id"])
                )
                if unresolved.is_empty() or best_candidates.is_empty():
                    continue

            # Add a small random lottery to break ties deterministically for
            # each candidate set.
            lottery = pl.Series("lottery", self.random.random(best_candidates.height))
            best_candidates = best_candidates.with_columns(lottery)

            # winners columns:
            # ┌──────────┬──────────────────┬──────────────────┬────────┬────────┬──────┬──────────────┬─────────┐
            # │ agent_id ┆ dim_0_candidate  ┆ dim_1_candidate  ┆ sugar  ┆ radius ┆ rank ┆ current_rank │ lottery │
            # │ ---      ┆ ---              ┆ ---              ┆ ---    ┆ ---    ┆ ---  ┆ ---          ┆ f64     │
            # ╞══════════╪══════════════════╪══════════════════╪════════╪════════╪══════╪══════════════╪═════════╡
            # Winners are the first row per destination cell after sorting by
            # destination and a random lottery. Again, `unique` avoids a more
            # expensive group_by while preserving the "first after sort" rule.
            winners = best_candidates.sort(
                ["cell_id", "lottery"],
            ).unique(
                subset=["cell_id"],
                keep="first",
                maintain_order=True,
            )

            assigned_parts.append(
                winners.select(
                    [
                        "agent_id",
                        pl.col("dim_0_candidate"),
                        pl.col("dim_1_candidate"),
                    ]
                )
            )
            movers = winners.filter(pl.col("cell_id") != pl.col("origin_cell_id"))

            # Keep `taken` compact: origins are already present, so only add
            # destinations for agents that actually move away from their origin.
            if not movers.is_empty():
                taken_ids = pl.concat([taken_ids, movers["cell_id"]], how="vertical")
                # Origins of agents that move away become available to others in
                # subsequent rounds.
                taken_ids = taken_ids.filter(~taken_ids.is_in(movers["origin_cell_id"]))

            winner_ids = winners.select("agent_id")
            unresolved = unresolved.filter(
                ~pl.col("agent_id").is_in(winner_ids["agent_id"])
            )
            if unresolved.is_empty():
                break

            # loser candidates columns mirror best_candidates (minus winners).
            losers = best_candidates.filter(
                ~pl.col("agent_id").is_in(winner_ids["agent_id"])
            )
            if losers.is_empty():
                continue

            # loser_updates columns (after select):
            # ┌──────────┬───────────┐
            # │ agent_id ┆ next_rank │
            # │ ---      ┆ ---       │
            # │ u64      ┆ i64       │
            # ╞══════════╪═══════════╡
            loser_updates = (
                losers.select(
                    "agent_id",
                    (pl.col("rank") + 1).cast(pl.Int64).alias("next_rank"),
                )
                .join(max_rank, on="agent_id", how="left")
                .with_columns(
                    pl.min_horizontal(pl.col("next_rank"), pl.col("max_rank")).alias(
                        "next_rank"
                    )
                )
                .select(["agent_id", "next_rank"])
            )

            # Promote losers' current_rank (if any) and continue.
            # unresolved (updated) retains columns agent_id/current_rank.
            unresolved = (
                unresolved.join(loser_updates, on="agent_id", how="left")
                .with_columns(
                    pl.when(pl.col("next_rank").is_not_null())
                    .then(pl.col("next_rank"))
                    .otherwise(pl.col("current_rank"))
                    .alias("current_rank")
                )
                .drop("next_rank")
            )

        if assigned_parts:
            return pl.concat([assigned, *assigned_parts], how="vertical")
        return assigned

__all__ = [
    "AntsBase",
    "AntsParallel",
]
