import numpy as np
import pandas as pd

from mesa_frames import AgentSetPandas, ModelDF


class AntPandas(AgentSetPandas):
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

        agents = pd.DataFrame(
            {
                "unique_id": np.arange(n_agents),
                "sugar": model.random.integers(6, 25, n_agents),
                "metabolism": model.random.integers(2, 4, n_agents),
                "vision": model.random.integers(1, 6, n_agents),
            }
        )
        self.add(agents)

    def move(self):
        neighborhood: pd.DataFrame = self.space.get_neighborhood(
            radius=self["vision"], agents=self, include_center=True
        )

        # Merge self.space.cells to obtain properties ('sugar') per cell
        neighborhood = neighborhood.merge(self.space.cells, on=["dim_0", "dim_1"])

        # Merge self.pos to obtain the agent_id of the center cell
        # TODO: get_neighborhood/get_neighbors should return 'agent_id_center' instead of center position when input is AgentLike
        neighborhood["agent_id_center"] = neighborhood.merge(
            self.pos.reset_index(),
            left_on=["dim_0_center", "dim_1_center"],
            right_on=["dim_0", "dim_1"],
        )["unique_id"]

        # Order of agents moves based on the original order of agents.
        # The agent in his cell has order 0 (highest)
        agent_order = neighborhood.groupby(["agent_id_center"], sort=False).ngroup()
        neighborhood["agent_order"] = agent_order
        agent_order = neighborhood[["agent_id_center", "agent_order"]].drop_duplicates()

        neighborhood = neighborhood.merge(
            agent_order.rename(
                columns={
                    "agent_id_center": "agent_id",
                    "agent_order": "blocking_agent_order",
                }
            ),
            on="agent_id",
        )

        # Filter impossible moves
        neighborhood = neighborhood[
            neighborhood["agent_order"] >= neighborhood["blocking_agent_order"]
        ]

        # Sort cells by sugar and radius (nearest first)
        neighborhood = neighborhood.sort_values(
            ["sugar", "radius"], ascending=[False, True]
        )

        best_moves = pd.DataFrame()

        # While there are agents that do not have a best move, keep looking for one
        while len(best_moves) < len(self.agents):
            # Get the best moves for each agent and if duplicates are found, select the one with the highest order
            new_best_moves = (
                neighborhood.groupby("agent_id_center", sort=False)
                .first()
                .sort_values("agent_order")
                .drop_duplicates(["dim_0", "dim_1"], keep="first")
            )

            # Agents can make the move if:
            # - There is no blocking agent
            # - The agent is in its own cell
            # - The blocking agent has moved before him
            new_best_moves = new_best_moves[
                (new_best_moves["agent_id"].isna())
                | (new_best_moves["agent_id"] == new_best_moves.index)
                | (new_best_moves["agent_id"].isin(best_moves.index))
            ]

            best_moves = pd.concat([best_moves, new_best_moves])

            # Remove agents that have already moved
            neighborhood = neighborhood[
                ~neighborhood["agent_id_center"].isin(best_moves.index)
            ]

            # Remove cells that have been already selected
            neighborhood = neighborhood.merge(
                best_moves[["dim_0", "dim_1"]],
                on=["dim_0", "dim_1"],
                how="left",
                indicator=True,
            )

            neighborhood = neighborhood[neighborhood["_merge"] == "left_only"].drop(
                columns="_merge"
            )

        self.space.move_agents(self, best_moves[["dim_0", "dim_1"]])

    def eat(self):
        cells = self.space.cells[self.space.cells["agent_id"].notna()].reset_index()
        self[cells["agent_id"], "sugar"] = (
            self[cells["agent_id"], "sugar"]
            + cells["sugar"]
            - self[cells["agent_id"], "metabolism"]
        )

    def step(self):
        self.shuffle().do("move").do("eat")
        self.discard(self[self["sugar"] <= 0])
