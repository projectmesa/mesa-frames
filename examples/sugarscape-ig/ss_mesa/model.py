import mesa
import numpy as np

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
        width: int | None = None,
        height: int | None = None,
    ):
        """
        Create a new Instant Growback model with the given parameters.

        """
        super().__init__()

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
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=False)
        self.agents: list = []

        agent_id = 0
        self.sugars = []
        for _, (x, y) in self.grid.coord_iter():
            max_sugar = sugar_grid[x, y]
            sugar = Sugar(agent_id, self, max_sugar)
            agent_id += 1
            self.grid.place_agent(sugar, (x, y))
            self.sugars.append(sugar)

        # Create agent:
        for i in range(self.n_agents):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            ssa = AntMesa(
                agent_id, self, False, initial_sugar[i], metabolism[i], vision[i]
            )
            agent_id += 1
            self.grid.place_agent(ssa, (x, y))
            self.agents.append(ssa)

        self.running = True

    def step(self):
        self.random.shuffle(self.agents)
        [agent.step() for agent in self.agents]
        [sugar.step() for sugar in self.sugars]

    def run_model(self, step_count=200):
        for i in range(step_count):
            if len(self.agents) == 0:
                return
            self.step()
