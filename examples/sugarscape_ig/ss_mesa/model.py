import mesa
import numpy as np
import polars as pl

from .agents import AntMesa, Sugar


class SugarscapeMesa(mesa.Model):
    """
    Sugarscape 2 Instant Growback
    """

    def __init__(
        self,
        n_agents: int,
        sugar_grid: np.ndarray | None = None,
        initial_sugar: np.ndarray | None = None,
        metabolism: np.ndarray | None = None,
        vision: np.ndarray | None = None,
        initial_positions: pl.DataFrame | None = None,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        """
        Create a new Instant Growback model with the given parameters.

        """
        # Initialize the Mesa base class (required in Mesa 3.0)
        super().__init__(seed=seed)

        # Set parameters
        if sugar_grid is None:
            sugar_grid = np.random.randint(0, 4, (width, height))
        if initial_sugar is None:
            initial_sugar = np.random.randint(6, 25, n_agents)
        if metabolism is None:
            metabolism = np.random.randint(2, 4, n_agents)
        if vision is None:
            vision = np.random.randint(1, 6, n_agents)

        self.width, self.height = sugar_grid.shape
        self.n_agents = n_agents
        self.space = mesa.space.MultiGrid(self.width, self.height, torus=False)

        # Create sugar resources
        sugar_count = 0
        for _, (x, y) in self.space.coord_iter():
            max_sugar = sugar_grid[x, y]
            sugar = Sugar(self, max_sugar)
            self.space.place_agent(sugar, (x, y))
            sugar_count += 1

        # Create AntMesa agents
        ant_count = 0
        for i in range(self.n_agents):
            # Determine position
            if initial_positions is not None:
                x = initial_positions["dim_0"][i]
                y = initial_positions["dim_1"][i]
            else:
                x = self.random.randrange(self.width)
                y = self.random.randrange(self.height)

            # Create and place agent
            ant = AntMesa(
                self,
                moore=False,
                sugar=initial_sugar[i],
                metabolism=metabolism[i],
                vision=vision[i],
            )
            self.space.place_agent(ant, (x, y))
            ant_count += 1

        self.running = True

    def step(self):
        # Get all AntMesa agents
        ant_agents = [agent for agent in self.agents if isinstance(agent, AntMesa)]

        # Randomize the order
        self.random.shuffle(ant_agents)

        # Step each ant agent directly
        for ant in ant_agents:
            ant.step()

        # Process Sugar agents directly
        for sugar in [agent for agent in self.agents if isinstance(agent, Sugar)]:
            sugar.step()

    def run_model(self, step_count=200):
        for i in range(step_count):
            if self.agents.count(AntMesa) == 0:
                return
            self.step()
