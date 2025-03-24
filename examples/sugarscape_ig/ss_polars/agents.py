from abc import abstractmethod

import numpy as np
import polars as pl
from numba import b1, guvectorize, int32

from mesa_frames import AgentSetPolars, ModelDF


class AntPolarsBase(AgentSetPolars):
    def __init__(
        self,
        model: ModelDF,
        n_agents: int,
        initial_sugar: np.ndarray | None = None,
        metabolism: np.ndarray | None = None,
        vision: np.ndarray | None = None,
    ):
        super().__init__(model)

        if initial_sugar is None:
            initial_sugar = model.random.integers(6, 25, n_agents)
        if metabolism is None:
            metabolism = model.random.integers(2, 4, n_agents)
        if vision is None:
            vision = model.random.integers(1, 6, n_agents)

        agents = pl.DataFrame(
            {
                "unique_id": pl.arange(n_agents, eager=True),
                "sugar": initial_sugar,
                "metabolism": metabolism,
                "vision": vision,
            }
        )
        self.add(agents)

    def eat(self):
        cells = self.space.cells.filter(pl.col("agent_id").is_not_null())
        self[cells["agent_id"], "sugar"] = (
            self[cells["agent_id"], "sugar"]
            + cells["sugar"]
            - self[cells["agent_id"], "metabolism"]
        )

    def step(self):
        self.shuffle().do("move").do("eat")
        self.discard(self.agents.filter(pl.col("sugar") <= 0))

    def move(self):
        neighborhood = self._get_neighborhood()
        agent_order = self._get_agent_order(neighborhood)
        neighborhood = self._prepare_neighborhood(neighborhood, agent_order)
        best_moves = self.get_best_moves(neighborhood)
        self.space.move_agents(agent_order["agent_id_center"], best_moves)

    def _get_neighborhood(self) -> pl.DataFrame:
        """Get the neighborhood of each agent, completed with the sugar of the cell and the agent_id of the center cell

        NOTE: This method should be unnecessary if get_neighborhood/get_neighbors return the agent_id of the center cell and the properties of the cells

        Returns
        -------
        pl.DataFrame
            Neighborhood DataFrame
        """
        neighborhood: pl.DataFrame = self.space.get_neighborhood(
            radius=self["vision"], agents=self, include_center=True
        )
        # Join self.space.cells to obtain properties ('sugar') per cell

        neighborhood = neighborhood.join(self.space.cells, on=["dim_0", "dim_1"])

        # Join self.pos to obtain the agent_id of the center cell
        # TODO: get_neighborhood/get_neighbors should return 'agent_id_center' instead of center position when input is AgentLike

        neighborhood = neighborhood.with_columns(
            agent_id_center=neighborhood.join(
                self.pos,
                left_on=["dim_0_center", "dim_1_center"],
                right_on=["dim_0", "dim_1"],
            )["unique_id"]
        )
        return neighborhood

    def _get_agent_order(self, neighborhood: pl.DataFrame) -> pl.DataFrame:
        """Get the order of agents based on the original order of agents

        Parameters
        ----------
        neighborhood : pl.DataFrame
            Neighborhood DataFrame

        Returns
        -------
        pl.DataFrame
            DataFrame with 'agent_id_center' and 'agent_order' columns
        """
        # Order of agents moves based on the original order of agents.
        # The agent in his cell has order 0 (highest)

        return (
            neighborhood.unique(
                subset=["agent_id_center"], keep="first", maintain_order=True
            )
            .with_row_count("agent_order")
            .select(["agent_id_center", "agent_order"])
        )

    def _prepare_neighborhood(
        self, neighborhood: pl.DataFrame, agent_order: pl.DataFrame
    ) -> pl.DataFrame:
        """Prepare the neighborhood DataFrame to find the best moves

        Parameters
        ----------
        neighborhood : pl.DataFrame
            Neighborhood DataFrame
        agent_order : pl.DataFrame
            DataFrame with 'agent_id_center' and 'agent_order' columns

        Returns
        -------
        pl.DataFrame
            Prepared neighborhood DataFrame
        """
        neighborhood = neighborhood.join(agent_order, on="agent_id_center")

        # Add blocking agent order
        neighborhood = neighborhood.join(
            agent_order.select(
                pl.col("agent_id_center").alias("agent_id"),
                pl.col("agent_order").alias("blocking_agent_order"),
            ),
            on="agent_id",
            how="left",
        ).rename({"agent_id": "blocking_agent_id"})

        # Filter only possible moves (agent is in his cell, blocking agent has moved before him or there is no blocking agent)
        neighborhood = neighborhood.filter(
            (pl.col("agent_order") >= pl.col("blocking_agent_order"))
            | pl.col("blocking_agent_order").is_null()
        )

        # Sort neighborhood by agent_order & max_sugar (max_sugar because we will check anyway if the cell is empty)
        # However, we need to make sure that the current agent cell is ordered by current sugar (since it's 0 until agent hasn't moved)
        neighborhood = neighborhood.with_columns(
            max_sugar=pl.when(pl.col("blocking_agent_id") == pl.col("agent_id_center"))
            .then(pl.lit(0))
            .otherwise(pl.col("max_sugar"))
        ).sort(
            ["agent_order", "max_sugar", "radius", "dim_0"],
            descending=[False, True, False, False],
        )
        return neighborhood

    def get_best_moves(self, neighborhood: pl.DataFrame) -> pl.DataFrame:
        """Get the best moves for each agent

        Parameters
        ----------
        neighborhood : pl.DataFrame
            Neighborhood DataFrame

        Returns
        -------
        pl.DataFrame
            DataFrame with the best moves for each agent
        """
        raise NotImplementedError("Subclasses must implement this method")


class AntPolarsLoopDF(AntPolarsBase):
    def get_best_moves(self, neighborhood: pl.DataFrame):
        best_moves = pl.DataFrame()

        # While there are agents that do not have a best move, keep looking for one
        while len(best_moves) < len(self.agents):
            # Check if there are previous agents that might make the same move (priority for the given move is > 1)
            neighborhood = neighborhood.with_columns(
                priority=pl.col("agent_order").cum_count().over(["dim_0", "dim_1"])
            )

            # Get the best moves for each agent:
            # If duplicates are found, select the one with the highest order
            new_best_moves = (
                neighborhood.group_by("agent_id_center", maintain_order=True)
                .first()
                .unique(subset=["dim_0", "dim_1"], keep="first", maintain_order=True)
            )
            # Agents can make the move if:
            # - There is no blocking agent
            # - The agent is in its own cell
            # - The blocking agent has moved before him
            # - There isn't a higher priority agent that might make the same move

            condition = pl.col("blocking_agent_id").is_null() | (
                pl.col("blocking_agent_id") == pl.col("agent_id_center")
            )
            if len(best_moves) > 0:
                condition = condition | pl.col("blocking_agent_id").is_in(
                    best_moves["agent_id_center"]
                )

            condition = condition & (pl.col("priority") == 1)

            new_best_moves = new_best_moves.filter(condition)

            best_moves = pl.concat([best_moves, new_best_moves])

            # Remove agents that have already moved
            neighborhood = neighborhood.filter(
                ~pl.col("agent_id_center").is_in(best_moves["agent_id_center"])
            )

            # Remove cells that have been already selected
            neighborhood = neighborhood.join(
                best_moves.select(["dim_0", "dim_1"]), on=["dim_0", "dim_1"], how="anti"
            )

            # Check if there are previous agents that might make the same move (priority for the given move is > 1)
            neighborhood = neighborhood.with_columns(
                priority=pl.col("agent_order").cum_count().over(["dim_0", "dim_1"])
            )
        return best_moves.sort("agent_order").select(["dim_0", "dim_1"])


class AntPolarsLoop(AntPolarsBase):
    numba_target = None

    def get_best_moves(self, neighborhood: pl.DataFrame):
        occupied_cells, free_cells, target_cells = self._prepare_cells(neighborhood)
        best_moves_func = self._get_best_moves()

        processed_agents = np.zeros(len(self.agents), dtype=np.bool_)

        if self.numba_target is None:
            # Non-vectorized case: we need to create and pass the best_moves array
            map_batches_func = lambda df: best_moves_func(
                occupied_cells,
                free_cells,
                target_cells,
                df.struct.field("agent_order"),
                df.struct.field("blocking_agent_order"),
                processed_agents,
                best_moves=np.full(len(self.agents), -1, dtype=np.int32),
            )
        else:
            # Vectorized case: Polars will create the output array (best_moves) automatically
            map_batches_func = lambda df: best_moves_func(
                occupied_cells.astype(np.int32),
                free_cells.astype(np.bool_),
                target_cells.astype(np.int32),
                df.struct.field("agent_order"),
                df.struct.field("blocking_agent_order"),
                processed_agents.astype(np.bool_),
            )

        best_moves = (
            neighborhood.fill_null(-1)
            .cast({"agent_order": pl.Int32, "blocking_agent_order": pl.Int32})
            .select(
                pl.struct(["agent_order", "blocking_agent_order"]).map_batches(
                    map_batches_func
                )
            )
            .with_columns(
                dim_0=pl.col("agent_order") // self.space.dimensions[1],
                dim_1=pl.col("agent_order") % self.space.dimensions[1],
            )
            .drop("agent_order")
        )
        return best_moves

    # Resolved method with proper docstring
    def _prepare_cells(
        self, neighborhood: pl.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the occupied and free cells and the target cells for each agent,
        based on the neighborhood DataFrame such that the arrays refer to a flattened version of the grid

        Parameters
        ----------
        neighborhood : pl.DataFrame
            Neighborhood DataFrame

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
            - occupied_cells: Array of currently occupied cell positions
            - free_cells: Boolean array indicating which cells are free
            - target_cells: Array of target cell positions for each agent
        """
        occupied_cells = (
            neighborhood[["agent_id_center", "agent_order"]]
            .unique()
            .join(self.pos, left_on="agent_id_center", right_on="unique_id")
            .with_columns(
                flattened=(pl.col("dim_0") * self.space.dimensions[1] + pl.col("dim_1"))
            )
            .sort("agent_order")["flattened"]
            .to_numpy()
        )
        free_cells = np.ones(
            self.space.dimensions[0] * self.space.dimensions[1], dtype=np.bool_
        )
        free_cells[occupied_cells] = False

        target_cells = (
            neighborhood["dim_0"] * self.space.dimensions[1] + neighborhood["dim_1"]
        ).to_numpy()
        return occupied_cells, free_cells, target_cells

    def _get_best_moves(self):
        raise NotImplementedError("Subclasses must implement this method")


class AntPolarsLoopNoVec(AntPolarsLoop):
    # Non-vectorized case
    def _get_best_moves(self):
        def inner_get_best_moves(
            occupied_cells: np.ndarray,
            free_cells: np.ndarray,
            target_cells: np.ndarray,
            agent_id_center: np.ndarray,
            blocking_agent: np.ndarray,
            processed_agents: np.ndarray,
            best_moves: np.ndarray,
        ) -> np.ndarray:
            for i, agent in enumerate(agent_id_center):
                # If the agent has not moved yet
                if not processed_agents[agent]:
                    # If the target cell is free
                    if free_cells[target_cells[i]] or blocking_agent[i] == agent:
                        best_moves[agent] = target_cells[i]
                        # Free current cell
                        free_cells[occupied_cells[agent]] = True
                        # Occupy target cell
                        free_cells[target_cells[i]] = False
                        processed_agents[agent] = True
            return best_moves

        return inner_get_best_moves


class AntPolarsNumba(AntPolarsLoop):
    # Vectorized case
    def _get_best_moves(self):
        @guvectorize(
            [
                (
                    int32[:],
                    b1[:],
                    int32[:],
                    int32[:],
                    int32[:],
                    b1[:],
                    int32[:],
                )
            ],
            "(n), (m), (p), (p), (p), (n)->(n)",
            nopython=True,
            target=self.numba_target,
            # Writable inputs should be declared according to https://numba.pydata.org/numba-doc/dev/user/vectorize.html#overwriting-input-values
            # In this case, there doesn't seem to be a difference. I will leave it commented for reference so that we can use CUDA target (which doesn't support writable_args)
            # writable_args=(
            #    "free_cells",
            #    "processed_agents",
            # ),
        )
        def vectorized_get_best_moves(
            occupied_cells,
            free_cells,
            target_cells,
            agent_id_center,
            blocking_agent,
            processed_agents,
            best_moves,
        ):
            for i, agent in enumerate(agent_id_center):
                # If the agent has not moved yet
                if not processed_agents[agent]:
                    # If the target cell is free
                    if free_cells[target_cells[i]] or blocking_agent[i] == agent:
                        best_moves[agent] = target_cells[i]
                        # Free current cell
                        free_cells[occupied_cells[agent]] = True
                        # Occupy target cell
                        free_cells[target_cells[i]] = False
                        processed_agents[agent] = True

        return vectorized_get_best_moves


class AntPolarsNumbaCPU(AntPolarsNumba):
    numba_target = "cpu"


class AntPolarsNumbaParallel(AntPolarsNumba):
    numba_target = "parallel"


class AntPolarsNumbaGPU(AntPolarsNumba):
    numba_target = "cuda"
