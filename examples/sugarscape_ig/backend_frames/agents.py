"""Agent implementations for the Sugarscape IG example (mesa-frames).

This module provides the parallel (synchronous) movement variant as in the
advanced tutorial. The code and comments mirror
docs/general/user-guide/3_advanced_tutorial.py.
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
        # Map of currently occupied agent ids on the grid.
        occupied_ids = self.index
        # `occupied_ids` is a Polars Series; calling `is_in` with a Series
        # of the same datatype is ambiguous in newer Polars. Use `implode`
        # to collapse the Series into a list-like value for membership checks.
        occupied_cells = self.space.cells.filter(
            pl.col("agent_id").is_in(occupied_ids.implode())
        )
        if occupied_cells.is_empty():
            return
        # The agent ordering here uses the agent_id values stored in the
        # occupied cells frame; indexing the agent set with that vector updates
        # the matching agents' sugar values in one vectorised write.
        agent_ids = occupied_cells["agent_id"]
        self[agent_ids, "sugar"] = (
            self[agent_ids, "sugar"]
            + occupied_cells["sugar"]
            - self[agent_ids, "metabolism"]
        )
        # After harvesting, occupied cells have zero sugar.
        self.space.set_cells(
            occupied_cells.select(["dim_0", "dim_1"]),
            {"sugar": pl.Series(np.zeros(len(occupied_cells), dtype=np.int64))},
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

        neighborhood = self._build_neighborhood_frame(current_pos)
        choices, origins, max_rank = self._rank_candidates(neighborhood, current_pos)
        if choices.is_empty():
            return

        assigned = self._resolve_conflicts_in_rounds(choices, origins, max_rank)
        if assigned.is_empty():
            return

        # move_df columns:
        # ┌────────────┬────────────┬────────────┐
        # │ unique_id  ┆ dim_0      ┆ dim_1      │
        # │ ---        ┆ ---        ┆ ---        │
        # │ u64        ┆ i64        ┆ i64        │
        # ╞════════════╪════════════╪════════════╡
        move_df = pl.DataFrame(
            {
                "unique_id": assigned["agent_id"],
                "dim_0": assigned["dim_0_candidate"],
                "dim_1": assigned["dim_1_candidate"],
            }
        )
        # `move_agents` accepts IdsLike and SpaceCoordinates (Polars Series/DataFrame),
        # so pass Series/DataFrame directly rather than converting to Python lists.
        self.space.move_agents(move_df["unique_id"], move_df.select(["dim_0", "dim_1"]))

    def _build_neighborhood_frame(self, current_pos: pl.DataFrame) -> pl.DataFrame:
        """Assemble the sugar-weighted neighbourhood for each sensing agent.

        Parameters
        ----------
        current_pos : pl.DataFrame
            DataFrame with columns ``agent_id``, ``dim_0_center`` and
            ``dim_1_center`` describing the current position of each agent.

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
        # ┌────────────┬────────────┬────────┬────────────────┬────────────────┐
        # │ dim_0      ┆ dim_1      ┆ radius ┆ dim_0_center   ┆ dim_1_center   │
        # │ ---        ┆ ---        ┆ ---    ┆ ---            ┆ ---            │
        # │ i64        ┆ i64        ┆ i64    ┆ i64            ┆ i64            │
        # ╞════════════╪════════════╪════════╪════════════════╪════════════════╡
        neighborhood_cells = self.space.get_neighborhood(
            radius=self["vision"], agents=self, include_center=True
        )

        # sugar_cells columns:
        # ┌────────────┬────────────┬────────┐
        # │ dim_0      ┆ dim_1      ┆ sugar  │
        # │ ---        ┆ ---        ┆ ---    │
        # │ i64        ┆ i64        ┆ i64    │
        # ╞════════════╪════════════╪════════╡

        sugar_cells = self.space.cells.select(["dim_0", "dim_1", "sugar"])

        neighborhood_cells = (
            neighborhood_cells.join(sugar_cells, on=["dim_0", "dim_1"], how="left")
            .with_columns(pl.col("sugar").fill_null(0))
            .rename({"dim_0": "dim_0_candidate", "dim_1": "dim_1_candidate"})
        )

        neighborhood_cells = neighborhood_cells.join(
            current_pos,
            left_on=["dim_0_center", "dim_1_center"],
            right_on=["dim_0_center", "dim_1_center"],
            how="left",
        )

        # Final neighborhood columns:
        # ┌──────────┬────────┬──────────────────┬──────────────────┬────────┐
        # │ agent_id ┆ radius ┆ dim_0_candidate  ┆ dim_1_candidate  ┆ sugar  │
        # │ ---      ┆ ---    ┆ ---              ┆ ---              ┆ ---    │
        # │ u64      ┆ i64    ┆ i64              ┆ i64              ┆ i64    │
        # ╞══════════╪════════╪══════════════════╪══════════════════╪════════╡
        neighborhood_cells = neighborhood_cells.drop(
            ["dim_0_center", "dim_1_center"]
        ).select(["agent_id", "radius", "dim_0_candidate", "dim_1_candidate", "sugar"])

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
            .with_columns(pl.col("radius"))
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

        # taken columns:
        # ┌──────────────────┬──────────────────┐
        # │ dim_0_candidate  ┆ dim_1_candidate  │
        # │ ---              ┆ ---              │
        # │ i64              ┆ i64              │
        # ╞══════════════════╪══════════════════╡
        taken = pl.DataFrame(
            {
                "dim_0_candidate": pl.Series(
                    name="dim_0_candidate", values=[], dtype=pl.Int64
                ),
                "dim_1_candidate": pl.Series(
                    name="dim_1_candidate", values=[], dtype=pl.Int64
                ),
            }
        )

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
            candidate_pool = choices.join(unresolved, on="agent_id")
            candidate_pool = candidate_pool.filter(
                pl.col("rank") >= pl.col("current_rank")
            )
            if not taken.is_empty():
                candidate_pool = candidate_pool.join(
                    taken,
                    on=["dim_0_candidate", "dim_1_candidate"],
                    how="anti",
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
                assigned = pl.concat(
                    [
                        assigned,
                        fallback.select(
                            [
                                "agent_id",
                                pl.col("dim_0").alias("dim_0_candidate"),
                                pl.col("dim_1").alias("dim_1_candidate"),
                            ]
                        ),
                    ],
                    how="vertical",
                )
                break

            # best_candidates columns (per agent first choice):
            # ┌──────────┬──────────────────┬──────────────────┬────────┬────────┬──────┬──────────────┐
            # │ agent_id ┆ dim_0_candidate  ┆ dim_1_candidate  ┆ sugar  ┆ radius ┆ rank ┆ current_rank │
            # │ ---      ┆ ---              ┆ ---              ┆ ---    ┆ ---    ┆ ---  ┆ ---          │
            # │ u64      ┆ i64              ┆ i64              ┆ i64    ┆ i64    ┆ u32  ┆ i64          │
            # ╞══════════╪══════════════════╪══════════════════╪════════╪════════╪══════╪══════════════╡
            best_candidates = (
                candidate_pool.sort(["agent_id", "rank"])
                .group_by("agent_id", maintain_order=True)
                .first()
            )

            # Agents that had no candidate this round fall back to origin.
            # missing columns:
            # ┌──────────┬──────────────┐
            # │ agent_id ┆ current_rank │
            # │ ---      ┆ ---          │
            # │ u64      ┆ i64          │
            # ╞══════════╪══════════════╡
            missing = unresolved.join(
                best_candidates.select("agent_id"), on="agent_id", how="anti"
            )
            if not missing.is_empty():
                # fallback (missing) columns match fallback table above.
                fallback = missing.join(origins, on="agent_id", how="left")
                assigned = pl.concat(
                    [
                        assigned,
                        fallback.select(
                            [
                                "agent_id",
                                pl.col("dim_0").alias("dim_0_candidate"),
                                pl.col("dim_1").alias("dim_1_candidate"),
                            ]
                        ),
                    ],
                    how="vertical",
                )
                taken = pl.concat(
                    [
                        taken,
                        fallback.select(
                            [
                                pl.col("dim_0").alias("dim_0_candidate"),
                                pl.col("dim_1").alias("dim_1_candidate"),
                            ]
                        ),
                    ],
                    how="vertical",
                )
                unresolved = unresolved.join(
                    missing.select("agent_id"), on="agent_id", how="anti"
                )
                best_candidates = best_candidates.join(
                    missing.select("agent_id"), on="agent_id", how="anti"
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
            winners = (
                best_candidates.sort(["dim_0_candidate", "dim_1_candidate", "lottery"])
                .group_by(["dim_0_candidate", "dim_1_candidate"], maintain_order=True)
                .first()
            )

            assigned = pl.concat(
                [
                    assigned,
                    winners.select(
                        [
                            "agent_id",
                            pl.col("dim_0_candidate"),
                            pl.col("dim_1_candidate"),
                        ]
                    ),
                ],
                how="vertical",
            )
            taken = pl.concat(
                [
                    taken,
                    winners.select(["dim_0_candidate", "dim_1_candidate"]),
                ],
                how="vertical",
            )

            winner_ids = winners.select("agent_id")
            unresolved = unresolved.join(winner_ids, on="agent_id", how="anti")
            if unresolved.is_empty():
                break

            # loser candidates columns mirror best_candidates (minus winners).
            losers = best_candidates.join(winner_ids, on="agent_id", how="anti")
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

        return assigned


__all__ = [
    "AntsBase",
    "AntsParallel",
]

