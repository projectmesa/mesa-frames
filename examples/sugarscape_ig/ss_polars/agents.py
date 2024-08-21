import numpy as np
import polars as pl

from mesa_frames import AgentSetPolars, ModelDF


class AntPolars(AgentSetPolars):
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
                "sugar": model.random.integers(6, 25, n_agents),
                "metabolism": model.random.integers(2, 4, n_agents),
                "vision": model.random.integers(1, 6, n_agents),
            }
        )
        self.add(agents)

    def move(self):
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

        # Order of agents moves based on the original order of agents.
        # The agent in his cell has order 0 (highest)
        agent_order = neighborhood.unique(
            subset=["agent_id_center"], keep="first", maintain_order=True
        ).with_row_count("agent_order")

        neighborhood = neighborhood.join(agent_order, on="agent_id_center")

        neighborhood = neighborhood.join(
            agent_order.select(
                pl.col("agent_id_center").alias("agent_id"),
                pl.col("agent_order").alias("blocking_agent_order"),
            ),
            on="agent_id",
        )

        # Filter impossible moves
        neighborhood = neighborhood.filter(
            pl.col("agent_order") >= pl.col("blocking_agent_order")
        )

        # Sort cells by sugar and radius (nearest first)
        neighborhood = neighborhood.sort(["sugar", "radius"], descending=[True, False])

        best_moves = pl.DataFrame()
        # While there are agents that do not have a best move, keep looking for one
        while len(best_moves) < len(self.agents):
            # Get the best moves for each agent and if duplicates are found, select the one with the highest order
            new_best_moves = (
                neighborhood.group_by("agent_id_center", maintain_order=True)
                .first()
                .sort("agent_order")
                .unique(subset=["dim_0", "dim_1"], keep="first")
            )

            # Agents can make the move if:
            # - There is no blocking agent
            # - The agent is in its own cell
            # - The blocking agent has moved before him
            condition = pl.col("agent_id").is_null() | (
                pl.col("agent_id") == pl.col("agent_id_center")
            )
            if len(best_moves) > 0:
                condition = condition | pl.col("agent_id").is_in(
                    best_moves["agent_id_center"]
                )
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

        self.space.move_agents(self, best_moves.select(["dim_0", "dim_1"]))

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
